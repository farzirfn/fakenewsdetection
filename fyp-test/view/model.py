import streamlit as st
import mysql.connector
import pickle
import torch
import copy
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import os

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
PRETRAINED   = "distilbert-base-uncased"
SAVE_DIR     = "fake_news_distilbert"          # best-fold model
FULL_SAVE_DIR= "fake_news_distilbert_full"     # full-retrain model
LABEL_PATH   = "label_encoder.pkl"

MAX_LEN      = 128
BATCH_SIZE   = 16
MAX_EPOCHS   = 10
PATIENCE     = 2
LR           = 2e-5          # recommended for full fine-tune of DistilBERT
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1           # 10 % of total steps used for LR warmup
MAX_GRAD_NORM= 1.0           # gradient clipping

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ─────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────
def create_connection():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        port=st.secrets["mysql"]["port"],
        user=st.secrets["mysql"]["username"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"]
    )
def save_results_to_db(acc, prec, rec, f1, cm, classes, epochs_used):
    conn   = create_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO train_results
           (accuracy, prec, recall, f1, confusion_matrix, classes, epochs_used)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (acc, prec, rec, f1, str(cm.tolist()), str(classes.tolist()), epochs_used),
    )
    conn.commit()
    cursor.close()
    conn.close()

# ─────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────
def build_model(num_labels, device):
    """Load DistilBERT with a classification head — ALL layers unfrozen."""
    model = DistilBertForSequenceClassification.from_pretrained(
        PRETRAINED,
        num_labels=num_labels,
    )
    # Explicitly confirm all parameters require gradients (full fine-tune)
    for param in model.parameters():
        param.requires_grad = True
    return model.to(device)

def build_optimizer_and_scheduler(model, train_loader, num_epochs):
    """
    AdamW with weight decay, linear warmup + linear decay scheduler.
    Weight decay is NOT applied to bias / LayerNorm parameters.
    """
    no_decay = {"bias", "LayerNorm.weight"}
    grouped_params = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer   = AdamW(grouped_params, lr=LR)
    total_steps = len(train_loader) * num_epochs
    warmup_steps= int(total_steps * WARMUP_RATIO)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    losses = []
    for batch in loader:
        batch    = {k: v.to(device) for k, v in batch.items()}
        outputs  = model(**batch)
        loss     = outputs.loss
        loss.backward()
        clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)   # ← gradient clipping
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return float(np.mean(losses))

@torch.no_grad()
def evaluate(model, loader, device, num_labels):
    model.eval()
    preds, truths = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        truths.extend(batch["labels"].cpu().numpy())

    acc  = accuracy_score(truths, preds)
    prec = precision_score(truths, preds, average="weighted", zero_division=0)
    rec  = recall_score(truths, preds,    average="weighted", zero_division=0)
    f1   = f1_score(truths, preds,        average="weighted", zero_division=0)
    cm   = confusion_matrix(truths, preds, labels=list(range(num_labels)))
    return acc, prec, rec, f1, cm

# ─────────────────────────────────────────────
# Update live loss chart
# ─────────────────────────────────────────────
def update_loss_chart(plot_placeholder, all_fold_losses, current_losses, fold, epoch, avg_loss):
    fig = go.Figure()
    for i, fl in enumerate(all_fold_losses):
        fig.add_trace(go.Scatter(
            y=fl, mode="lines+markers",
            name=f"Fold {i+1}", opacity=0.35,
            line=dict(width=1),
        ))
    fig.add_trace(go.Scatter(
        y=current_losses, mode="lines+markers",
        line=dict(color="crimson", width=2.5),
        marker=dict(size=7),
        name=f"Fold {fold+1} (current)",
    ))
    fig.update_layout(
        title=f"Fold {fold+1} | Epoch {epoch+1} | Train Loss: {avg_loss:.4f}",
        xaxis_title="Epoch", yaxis_title="Loss",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=50, b=60),
    )
    plot_placeholder.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# MAIN RETRAINING FUNCTION
# ─────────────────────────────────────────────
def retrain_model(progress_bar, status_box, plot_placeholder, n_splits=5):
    start_time = time.time()
    try:
        # ── 1. Load data ──────────────────────────────────────────
        status_box.text("📂 Loading dataset from database…")
        progress_bar.progress(3)

        conn   = create_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT text, status FROM dataset")
        rows   = cursor.fetchall()
        cursor.close()
        conn.close()

        df = pd.DataFrame(rows, columns=["text", "status"]).dropna()
        df["status"] = df["status"].str.lower().str.strip()
        texts  = np.array(df["text"].tolist())
        labels_raw = df["status"].tolist()

        label_encoder  = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels_raw)
        num_labels     = len(label_encoder.classes_)

        tokenizer = DistilBertTokenizer.from_pretrained(PRETRAINED)
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── 2. Cross-validation ───────────────────────────────────
        skf          = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []
        all_fold_losses = []
        agg_cm       = np.zeros((num_labels, num_labels), dtype=int)

        best_fold_f1    = -1.0
        best_fold_state = None          # ← best fold checkpoint (deep copy)
        best_fold_idx   = -1

        progress_per_fold = 75 / n_splits   # CV occupies 3 % → 78 %

        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels_encoded)):
            status_box.text(f"🔁 Fold {fold+1}/{n_splits} — fine-tuning all layers…")

            X_train, X_val = texts[train_idx], texts[val_idx]
            y_train, y_val = labels_encoded[train_idx], labels_encoded[val_idx]

            train_loader = DataLoader(
                NewsDataset(X_train.tolist(), y_train.tolist(), tokenizer),
                batch_size=BATCH_SIZE, shuffle=True,
            )
            val_loader = DataLoader(
                NewsDataset(X_val.tolist(), y_val.tolist(), tokenizer),
                batch_size=BATCH_SIZE,
            )

            # Fresh fine-tune from pre-trained weights each fold
            model               = build_model(num_labels, device)
            optimizer, scheduler= build_optimizer_and_scheduler(model, train_loader, MAX_EPOCHS)

            best_val_f1   = -1.0
            best_state    = None        # best checkpoint within this fold
            no_improve    = 0
            epoch_losses  = []

            for epoch in range(MAX_EPOCHS):
                train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
                epoch_losses.append(train_loss)

                # Evaluate on validation fold after every epoch
                acc, prec, rec, f1, _ = evaluate(model, val_loader, device, num_labels)

                update_loss_chart(plot_placeholder, all_fold_losses,
                                  epoch_losses, fold, epoch, train_loss)

                status_box.text(
                    f"🔁 Fold {fold+1} | Epoch {epoch+1} | "
                    f"Loss: {train_loss:.4f} | Val F1: {f1:.4f}"
                )

                # ── Save best checkpoint within fold (by val F1) ──
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    no_improve  = 0
                    # Deep-copy state dict to CPU to save GPU memory
                    best_state  = copy.deepcopy(
                        {k: v.cpu() for k, v in model.state_dict().items()}
                    )
                else:
                    no_improve += 1

                if no_improve >= PATIENCE:
                    status_box.text(
                        f"⏹️ Fold {fold+1} — early stopping at epoch {epoch+1}"
                    )
                    break

                # Progress bar update
                base     = 3 + fold * progress_per_fold
                ep_frac  = (epoch + 1) / MAX_EPOCHS
                progress_bar.progress(int(min(base + ep_frac * progress_per_fold, 78)))

            all_fold_losses.append(epoch_losses)

            # ── Restore best checkpoint, then evaluate ────────────
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            acc, prec, rec, f1, cm = evaluate(model, val_loader, device, num_labels)
            agg_cm += cm

            fold_metrics.append({
                "acc": acc, "prec": prec, "rec": rec, "f1": f1,
                "epochs": len(epoch_losses),
            })

            status_box.text(
                f"✅ Fold {fold+1} done — Acc: {acc:.4f} | F1: {f1:.4f}"
            )

            # ── Track best fold across all folds ──────────────────
            if f1 > best_fold_f1:
                best_fold_f1    = f1
                best_fold_state = best_state          # already on CPU
                best_fold_idx   = fold

        # ── 3. Aggregate CV metrics ───────────────────────────────
        avg_acc  = float(np.mean([m["acc"]  for m in fold_metrics]))
        avg_prec = float(np.mean([m["prec"] for m in fold_metrics]))
        avg_rec  = float(np.mean([m["rec"]  for m in fold_metrics]))
        avg_f1   = float(np.mean([m["f1"]   for m in fold_metrics]))
        std_acc  = float(np.std( [m["acc"]  for m in fold_metrics]))
        std_f1   = float(np.std( [m["f1"]   for m in fold_metrics]))
        avg_epochs_used = max(1, round(
            sum(m["epochs"] for m in fold_metrics) / n_splits
        ))

        progress_bar.progress(80)

        # ── 4. Save BEST-FOLD model ───────────────────────────────
        status_box.text(
            f"💾 Saving best-fold model (Fold {best_fold_idx+1}, F1={best_fold_f1:.4f})…"
        )
        os.makedirs(SAVE_DIR, exist_ok=True)
        best_model = build_model(num_labels, device)
        best_model.load_state_dict(
            {k: v.to(device) for k, v in best_fold_state.items()}
        )
        best_model.save_pretrained(SAVE_DIR, safe_serialization=False)
        tokenizer.save_pretrained(SAVE_DIR)
        with open(os.path.join(SAVE_DIR, LABEL_PATH), "wb") as f:
            pickle.dump(label_encoder, f)

        progress_bar.progress(85)

        # ── 5. Full-data retrain ──────────────────────────────────
        status_box.text(
            f"🏁 Retraining on full dataset for {avg_epochs_used} epoch(s)…"
        )
        full_loader = DataLoader(
            NewsDataset(texts.tolist(), labels_encoded.tolist(), tokenizer),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        full_model              = build_model(num_labels, device)
        full_optimizer, full_scheduler = build_optimizer_and_scheduler(
            full_model, full_loader, avg_epochs_used
        )

        for epoch in range(avg_epochs_used):
            loss = train_one_epoch(
                full_model, full_loader, full_optimizer, full_scheduler, device
            )
            status_box.text(
                f"🏁 Full retrain | Epoch {epoch+1}/{avg_epochs_used} | Loss: {loss:.4f}"
            )
            progress_bar.progress(85 + int((epoch + 1) / avg_epochs_used * 10))

        os.makedirs(FULL_SAVE_DIR, exist_ok=True)
        full_model.save_pretrained(FULL_SAVE_DIR, safe_serialization=False)
        tokenizer.save_pretrained(FULL_SAVE_DIR)
        with open(os.path.join(FULL_SAVE_DIR, LABEL_PATH), "wb") as f:
            pickle.dump(label_encoder, f)

        progress_bar.progress(96)

        # ── 6. Save metrics to DB ─────────────────────────────────
        save_results_to_db(avg_acc, avg_prec, avg_rec, avg_f1,
                           agg_cm, label_encoder.classes_, avg_epochs_used)

        progress_bar.progress(100)

        return {
            "success":        True,
            "accuracy":       avg_acc,
            "precision":      avg_prec,
            "recall":         avg_rec,
            "f1":             avg_f1,
            "std_acc":        std_acc,
            "std_f1":         std_f1,
            "confusion_matrix": agg_cm,
            "classes":        label_encoder.classes_,
            "fold_metrics":   fold_metrics,
            "epochs_used":    avg_epochs_used,
            "best_fold":      best_fold_idx + 1,
            "best_fold_f1":   best_fold_f1,
            "elapsed_time":   time.time() - start_time,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────
# STREAMLIT PAGE
# ─────────────────────────────────────────────
def model_page():
    st.title("🤖 Model Retraining — Full Fine-Tune + Cross-Validation")

    col_a, col_b = st.columns(2)
    with col_a:
        n_splits = st.slider("CV Folds (k)", 3, 10, 5)
    with col_b:
        st.info(
            f"**Strategy:** Full fine-tune (all layers)  \n"
            f"**LR:** {LR} | **Warmup:** {int(WARMUP_RATIO*100)}%  \n"
            f"**Grad clip:** {MAX_GRAD_NORM} | **Weight decay:** {WEIGHT_DECAY}"
        )

    if st.button("🚀 Start Retraining"):
        progress   = st.progress(0)
        status     = st.empty()
        loss_chart = st.empty()

        result = retrain_model(progress, status, loss_chart, n_splits=n_splits)

        progress.empty()
        status.empty()
        loss_chart.empty()

        if result["success"]:
            st.success(
                f"✅ {n_splits}-Fold CV complete | "
                f"⏱ {result['elapsed_time']:.1f}s | "
                f"Best fold: #{result['best_fold']} (F1={result['best_fold_f1']:.4f})"
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",  f"{result['accuracy']:.4f}",  f"±{result['std_acc']:.4f}")
            c2.metric("Precision", f"{result['precision']:.4f}")
            c3.metric("Recall",    f"{result['recall']:.4f}")
            c4.metric("F1 Score",  f"{result['f1']:.4f}",        f"±{result['std_f1']:.4f}")

            st.caption(
                f"📁 Best-fold model saved → `{SAVE_DIR}/`  \n"
                f"📁 Full-retrain model saved → `{FULL_SAVE_DIR}/`"
            )

            # Per-fold table
            st.subheader("📋 Per-Fold Results")
            fold_df = pd.DataFrame(result["fold_metrics"])
            fold_df.index = [
                f"⭐ Fold {i+1}" if i == result["best_fold"]-1 else f"Fold {i+1}"
                for i in range(len(fold_df))
            ]
            fold_df.columns = ["Accuracy", "Precision", "Recall", "F1", "Epochs Used"]
            st.dataframe(
                fold_df.style.format(
                    "{:.4f}", subset=["Accuracy", "Precision", "Recall", "F1"]
                ).highlight_max(
                    subset=["F1"], color="#d4edda"
                )
            )

            # Aggregated confusion matrix
            st.subheader("🔲 Aggregated Confusion Matrix (all folds)")
            fig = go.Figure(data=go.Heatmap(
                z=result["confusion_matrix"],
                x=result["classes"],
                y=result["classes"],
                text=result["confusion_matrix"],
                texttemplate="%{text}",
                colorscale="Blues",
            ))
            fig.update_layout(
                xaxis_title="Predicted",
                yaxis_title="Actual",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(f"❌ Training failed: {result['error']}")


if __name__ == "__main__":
    model_page()