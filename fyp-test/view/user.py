import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# ================================
# MODEL LOADING
# ================================
@st.cache_resource
def load_model():
    """Load the trained model and tokenizer"""

    model = DistilBertForSequenceClassification.from_pretrained(
        "farzirfn/fake-news-distilbert"
    )
    tokenizer = DistilBertTokenizer.from_pretrained(
        "farzirfn/fake-news-distilbert"
    )

    # ✅ Use built-in label mapping (NO .pkl needed)
    label_map = model.config.id2label

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, label_map, device


# ================================
# PREDICTION FUNCTION
# ================================
def predict_news(model, tokenizer, label_map, device, text):
    """Predict whether news is real or fake"""

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

    pred_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_id].item()

    # ✅ Get label from model config
    if isinstance(label_map, dict):
        label = label_map[str(pred_id)] if str(pred_id) in label_map else label_map[pred_id]
    else:
        label = label_map[pred_id]

    return label, confidence, probs[0].cpu().numpy()


# ================================
# GAUGE CHART
# ================================
def create_confidence_gauge(confidence, label):
    color = "#667eea" if label.lower() == "real" else "#f5576c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence Score"},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "#f5576c"},
                {'range': [50, 80], 'color': "#f5a623"},
                {'range': [80, 100], 'color': "#667eea"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'value': 90
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


# ================================
# MAIN APP
# ================================
def user_home():
    # Header
    st.markdown(
        "<h2 style='text-align:center; color:#2E86C1;'>🔍 Fake News Detector</h2>",
        unsafe_allow_html=True
    )
    st.write(
        "<p style='text-align:center; color:gray;'>Powered by AI • Analyze news articles in seconds</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # Load model
    try:
        model, tokenizer, label_map, device = load_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

    # Input section
    st.subheader("📝 Enter News Article")
    text_input = st.text_area(
        label="Paste the news headline or article text below:",
        height=150,
        placeholder="Example: Breaking news! Scientists discover new planet..."
    )

    predict_button = st.button("🔍 Analyze News", use_container_width=True)

    # Prediction
    if predict_button:
        if text_input.strip():
            with st.spinner("🤖 Analyzing..."):
                label, confidence, probs = predict_news(
                    model, tokenizer, label_map, device, text_input
                )
                confidence_pct = round(confidence * 100, 2)

            # Result
            if label.lower() == "real":
                st.success("✅ REAL NEWS")
                st.metric("Confidence", f"{confidence_pct:.2f}%")
            else:
                st.error("❌ FAKE NEWS")
                st.metric("Confidence", f"{confidence_pct:.2f}%")

            # Gauge
            st.subheader("📊 Confidence Analysis")
            fig = create_confidence_gauge(confidence, label)
            st.plotly_chart(fig, use_container_width=True)

            # Probabilities
            with st.expander("📈 View Detailed Probabilities"):
                df = pd.DataFrame({
                    "Class": list(label_map.values()),
                    "Probability (%)": [round(p * 100, 2) for p in probs]
                })

                fig_probs = px.bar(
                    df,
                    x="Class",
                    y="Probability (%)",
                    text="Probability (%)",
                    color="Class",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )

                fig_probs.update_traces(textposition="outside")
                fig_probs.update_layout(height=320)

                st.plotly_chart(fig_probs, use_container_width=True)

        else:
            st.warning("⚠️ Please enter some text to analyze.")

    # Info
    if not predict_button or not text_input.strip():
        st.info("""
        **ℹ️ How It Works**
        
        This AI-powered system uses **DistilBERT**, a state-of-the-art NLP model, to classify news as real or fake.
        
        - 🎯 High Accuracy  
        - ⚡ Instant Results  
        - 🔒 Privacy Friendly  
        """)

    # Footer
    st.markdown("---")
    st.caption("⚡ Powered by DistilBERT • Made with Streamlit • © 2026")


# ================================
# RUN APP
# ================================
if __name__ == "__main__":
    user_home()