import streamlit as st
import mysql.connector
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# -------------------------------
# Dataset Class
# -------------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
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
    host="localhost",
    user="farzifae_fyp",
    password="AK9CYVY#),&2",
    database="farzifae_fyp"
)

# -------------------------------
# Retrain Model with Live Loss Chart + Early Stopping + DB Save
# -------------------------------
def retrain_model(progress_placeholder, status_placeholder, plot_placeholder):
    start_time = time.time()
    try:
        # Load dataset
        status_placeholder.text("📂 Loading dataset...")
        progress_placeholder.progress(10)

        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT text, status FROM dataset")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        df = pd.DataFrame(rows, columns=["text", "status"]).dropna()
        df["status"] = df["status"].str.lower().str.strip()
        texts = df["text"].tolist()
        labels = df["status"].tolist()

        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        progress_placeholder.progress(20)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels_encoded, test_size=0.2, random_state=42
        )
        progress_placeholder.progress(30)

        # Tokenizer + Dataset
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        train_dataset = NewsDataset(X_train, y_train, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        progress_placeholder.progress(40)

        # Model + Optimizer + Scheduler
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_labels = len(label_encoder.classes_)

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_labels
        )
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=5e-5)

        max_epochs = 10
        patience = 2
        best_loss = float("inf")
        no_improve = 0
        epoch_losses = []

        lr_scheduler = get_scheduler(
            "linear",
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * max_epochs
        )

        # ===========================
        # TRAINING LOOP
        # ===========================
        for epoch in range(max_epochs):
            status_placeholder.text(f"🔥 Training Epoch {epoch+1}/{max_epochs}")
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

                # Optional: small progress update per batch
                progress = 40 + int((batch_idx / len(train_loader)) * 30)
                progress_placeholder.progress(min(progress, 70))

            avg_loss = np.mean(batch_losses)
            epoch_losses.append(avg_loss)

            # Update live loss chart using Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=epoch_losses,
                mode='lines+markers',
                line=dict(color='red'),
                name='Loss'
            ))
            fig.update_layout(
                title=f"Epoch {epoch+1} Loss: {avg_loss:.4f}",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                yaxis=dict(range=[0, max(epoch_losses)+0.05])
            )
            plot_placeholder.plotly_chart(fig, use_container_width=True)

            status_placeholder.text(f"📊 Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                status_placeholder.text("⏹️ Early stopping triggered")
                break

        # ===========================
        # EVALUATION
        # ===========================
        model.eval()
        test_dataset = NewsDataset(X_test, y_test, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=16)

        preds, true_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                true_labels.extend(batch["labels"].cpu().numpy())

        acc = accuracy_score(true_labels, preds)
        prec = precision_score(true_labels, preds, average="weighted")
        rec = recall_score(true_labels, preds, average="weighted")
        f1 = f1_score(true_labels, preds, average="weighted")
        cm = confusion_matrix(true_labels, preds)
        progress_placeholder.progress(90)

        # Save model + tokenizer + label encoder
        model.save_pretrained("fake_news_distilbert", safe_serialization=False)
        tokenizer.save_pretrained("fake_news_distilbert")
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)

        # Save training metrics to DB
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO train_results
            (accuracy, prec, recall, f1, confusion_matrix, classes, epochs_used)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            acc,
            prec,
            rec,
            f1,
            str(cm.tolist()),
            str(label_encoder.classes_.tolist()),
            epoch+1
        ))
        conn.commit()
        cursor.close()
        conn.close()

        progress_placeholder.progress(100)

        return {
            "success": True,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "classes": label_encoder.classes_,
            "epochs_used": epoch+1,
            "elapsed_time": time.time()-start_time
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

# -------------------------------
# STREAMLIT PAGE
# -------------------------------
def model_page():
    st.title("🤖 Model Retraining")

    if st.button("🚀 Start Retraining"):
        progress = st.progress(0)
        status = st.empty()
        loss_chart = st.empty()  # placeholder for live loss chart

        result = retrain_model(progress, status, loss_chart)

        progress.empty()
        status.empty()
        loss_chart.empty()

        if result["success"]:
            st.success(f"""
            ✅ Training completed  
            ⏱ Time: {result['elapsed_time']:.1f}s  
            🛑 Epochs Used: {result['epochs_used']}
            """)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{result['accuracy']:.4f}")
            col2.metric("Precision", f"{result['precision']:.4f}")
            col3.metric("Recall", f"{result['recall']:.4f}")
            col4.metric("F1 Score", f"{result['f1']:.4f}")

            fig = go.Figure(
                data=go.Heatmap(
                    z=result["confusion_matrix"],
                    x=result["classes"],
                    y=result["classes"],
                    text=result["confusion_matrix"],
                    texttemplate="%{text}",
                    colorscale="Blues"
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(result["error"])

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    model_page()
