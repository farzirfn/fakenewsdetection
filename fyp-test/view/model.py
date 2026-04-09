import streamlit as st
import mysql.connector
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import time

# -------------------------------
# Dataset Class
# -------------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -------------------------------
# DB Connection
# -------------------------------
def create_connection():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        port=st.secrets["mysql"]["port"],
        user=st.secrets["mysql"]["username"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"]
    )

# -------------------------------
# Retrain Model dengan K-Fold CV
# -------------------------------
def retrain_model(progress_placeholder, status_placeholder, plot_placeholder,
                  n_splits=5, max_epochs=10, patience=3):
    start_time = time.time()
    try:
        # ===========================
        # LOAD DATASET
        # ===========================
        status_placeholder.text("📂 Loading dataset...")
        progress_placeholder.progress(5)

        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT text, status FROM dataset")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        df = pd.DataFrame(rows, columns=["text", "status"]).dropna()
        df["status"] = df["status"].str.lower().str.strip()
        texts = np.array(df["text"].tolist())
        labels_raw = df["status"].tolist()

        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = np.array(label_encoder.fit_transform(labels_raw))

        # Semak kelas
        st.write(f"📊 Agihan kelas: {dict(zip(*np.unique(labels_encoded, return_counts=True)))}")
        progress_placeholder.progress(10)

        # ===========================
        # K-FOLD SETUP
        # ===========================
        BATCH_SIZE = 16
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_labels = len(label_encoder.classes_)

        # Simpan metrics tiap fold
        fold_metrics = []
        all_cms = []

        # ===========================
        # K-FOLD LOOP
        # ===========================
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels_encoded)):
            status_placeholder.text(f"🔄 Fold {fold+1}/{n_splits} — Training...")

            X_train, X_val = texts[train_idx], texts[val_idx]
            y_train, y_val = labels_encoded[train_idx], labels_encoded[val_idx]

            # Dataset & DataLoader
            train_dataset = NewsDataset(X_train, y_train, tokenizer)
            val_dataset   = NewsDataset(X_val,   y_val,   tokenizer)
            train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
            val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, pin_memory=True)

            # Model baru setiap fold
            model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=num_labels
            ).to(device)

            optimizer = AdamW(model.parameters(), lr=5e-5)
            lr_scheduler = get_scheduler(
                "linear",
                optimizer,
                num_warmup_steps=0,
                num_training_steps=len(train_loader) * max_epochs
            )

            best_loss = float("inf")
            no_improve = 0
            epoch_losses = []

            # ------- Training Loop -------
            for epoch in range(max_epochs):
                model.train()
                batch_losses = []

                for batch_idx, batch in enumerate(train_loader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    batch_losses.append(loss.item())

                    # Progress bar
                    fold_progress = int((fold / n_splits) * 80)
                    epoch_progress = int((batch_idx / len(train_loader)) * (80 / n_splits))
                    progress_placeholder.progress(min(10 + fold_progress + epoch_progress, 89))

                avg_loss = np.mean(batch_losses)
                epoch_losses.append(avg_loss)

                # Update loss chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=epoch_losses,
                    mode='lines+markers',
                    line=dict(color='red'),
                    name=f'Fold {fold+1} Loss'
                ))
                fig.update_layout(
                    title=f"Fold {fold+1} | Epoch {epoch+1} | Loss: {avg_loss:.4f}",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    yaxis=dict(range=[0, max(epoch_losses) + 0.05])
                )
                plot_placeholder.plotly_chart(fig, use_container_width=True)

                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    status_placeholder.text(f"⏹️ Early stopping — Fold {fold+1} Epoch {epoch+1}")
                    break

            # ------- Evaluation per Fold -------
            model.eval()
            preds, true_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                    true_labels.extend(batch["labels"].cpu().numpy())

            acc  = accuracy_score(true_labels, preds)
            prec = precision_score(true_labels, preds, average="weighted", zero_division=0)
            rec  = recall_score(true_labels, preds, average="weighted", zero_division=0)
            f1   = f1_score(true_labels, preds, average="weighted", zero_division=0)
            cm   = confusion_matrix(true_labels, preds)

            fold_metrics.append({
                "fold": fold + 1,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "epochs_used": epoch + 1
            })
            all_cms.append(cm)

            status_placeholder.text(
                f"✅ Fold {fold+1} selesai — Acc: {acc:.4f} | F1: {f1:.4f}"
            )

        # ===========================
        # AVERAGE METRICS
        # ===========================
        progress_placeholder.progress(90)

        avg_acc  = np.mean([m["accuracy"]  for m in fold_metrics])
        avg_prec = np.mean([m["precision"] for m in fold_metrics])
        avg_rec  = np.mean([m["recall"]    for m in fold_metrics])
        avg_f1   = np.mean([m["f1"]        for m in fold_metrics])

        std_acc  = np.std([m["accuracy"]  for m in fold_metrics])
        std_f1   = np.std([m["f1"]        for m in fold_metrics])

        avg_cm = np.mean(all_cms, axis=0).astype(int)

        # ===========================
        # TRAIN FINAL MODEL (full data)
        # ===========================
        status_placeholder.text("🔁 Training final model pada full dataset...")

        full_dataset  = NewsDataset(texts, labels_encoded, tokenizer)
        full_loader   = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        final_model   = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        ).to(device)

        final_optimizer = AdamW(final_model.parameters(), lr=5e-5)
        final_scheduler = get_scheduler(
            "linear", final_optimizer,
            num_warmup_steps=0,
            num_training_steps=len(full_loader) * 3  # 3 epoch sahaja untuk final
        )

        final_model.train()
        for epoch in range(3):
            for batch in full_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = final_model(**batch)
                outputs.loss.backward()
                final_optimizer.step()
                final_scheduler.step()
                final_optimizer.zero_grad()

        # Save final model
        final_model.save_pretrained("fake_news_distilbert", safe_serialization=False)
        tokenizer.save_pretrained("fake_news_distilbert")
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)

        # ===========================
        # SAVE TO DB
        # ===========================
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO train_results
            (accuracy, prec, recall, f1, confusion_matrix, classes, epochs_used)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            avg_acc, avg_prec, avg_rec, avg_f1,
            str(avg_cm.tolist()),
            str(label_encoder.classes_.tolist()),
            int(np.mean([m["epochs_used"] for m in fold_metrics]))
        ))
        conn.commit()
        cursor.close()
        conn.close()

        progress_placeholder.progress(100)

        return {
            "success": True,
            "fold_metrics": fold_metrics,
            "avg_accuracy": avg_acc,
            "avg_precision": avg_prec,
            "avg_recall": avg_rec,
            "avg_f1": avg_f1,
            "std_accuracy": std_acc,
            "std_f1": std_f1,
            "avg_confusion_matrix": avg_cm,
            "classes": label_encoder.classes_,
            "elapsed_time": time.time() - start_time
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# -------------------------------
# STREAMLIT PAGE
# -------------------------------
def model_page():
    st.title("🤖 Model Retraining — K-Fold Cross Validation")

    # ===========================
    # TETAPAN PENGGUNA
    # ===========================
    st.subheader("⚙️ Tetapan Training")

    col1, col2, col3 = st.columns(3)

    with col1:
        n_splits = st.slider(
            "Bilangan Fold (K)",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Lebih besar = lebih teliti tapi lebih lama"
        )

    with col2:
        max_epochs = st.slider(
            "Max Epochs per Fold",
            min_value=3,
            max_value=15,
            value=10,
            step=1
        )

    with col3:
        patience = st.slider(
            "Early Stopping Patience",
            min_value=1,
            max_value=5,
            value=3,
            step=1
        )

    # Tunjuk anggaran masa
    st.info(
        f"Anggaran: **{n_splits} fold** × max **{max_epochs} epoch** "
        f"= max **{n_splits * max_epochs} epoch** total"
    )

    if st.button("🚀 Start Retraining"):
        progress   = st.progress(0)
        status     = st.empty()
        loss_chart = st.empty()

        result = retrain_model(
            progress, status, loss_chart,
            n_splits=n_splits,         # ← hantar parameter
            max_epochs=max_epochs,
            patience=patience
        )

        progress.empty()
        status.empty()

        if result["success"]:
            st.success(f"✅ Training selesai dalam {result['elapsed_time']:.1f}s")

            # ------- Per-Fold Results Table -------
            st.subheader("📋 Keputusan setiap fold")
            df_folds = pd.DataFrame(result["fold_metrics"])
            st.dataframe(df_folds.style.format({
                "accuracy": "{:.4f}", "precision": "{:.4f}",
                "recall": "{:.4f}", "f1": "{:.4f}"
            }), use_container_width=True)

            # ------- Average Metrics -------
            st.subheader("📊 Purata (Mean ± Std)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{result['avg_accuracy']:.4f}", f"±{result['std_accuracy']:.4f}")
            col2.metric("Precision", f"{result['avg_precision']:.4f}")
            col3.metric("Recall",    f"{result['avg_recall']:.4f}")
            col4.metric("F1 Score",  f"{result['avg_f1']:.4f}", f"±{result['std_f1']:.4f}")

            # ------- Fold F1 Bar Chart -------
            st.subheader("📈 F1 Score per Fold")
            fig_bar = go.Figure(go.Bar(
                x=[f"Fold {m['fold']}" for m in result["fold_metrics"]],
                y=[m["f1"] for m in result["fold_metrics"]],
                marker_color="steelblue"
            ))
            fig_bar.add_hline(
                y=result["avg_f1"], line_dash="dash",
                line_color="red", annotation_text=f"Mean F1: {result['avg_f1']:.4f}"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ------- Confusion Matrix (purata) -------
            st.subheader("🗂️ Confusion Matrix (purata)")
            fig_cm = go.Figure(data=go.Heatmap(
                z=result["avg_confusion_matrix"],
                x=result["classes"],
                y=result["classes"],
                text=result["avg_confusion_matrix"],
                texttemplate="%{text}",
                colorscale="Blues"
            ))
            st.plotly_chart(fig_cm, use_container_width=True)

        else:
            st.error(result["error"])


if __name__ == "__main__":
    model_page()