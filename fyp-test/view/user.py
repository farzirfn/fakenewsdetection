import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import pickle
import plotly.graph_objects as go
import plotly.express as px
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from view.xai import display_xai_report

# ================================
# MODEL LOADING
# ================================
@st.cache_resource
def load_model():
    """Load the trained model, tokenizer, and label encoder"""
    model = DistilBertForSequenceClassification.from_pretrained("farzirfn/fake-news-distilbert")
    tokenizer = DistilBertTokenizer.from_pretrained("farzirfn/fake-news-distilbert")
    from huggingface_hub import hf_hub_download
    encoder_path = hf_hub_download(repo_id="farzirfn/fake-news-distilbert", filename="label_encoder.pkl")
    label_encoder = pickle.load(open(encoder_path, "rb"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, label_encoder, device

# ================================
# PREDICTION FUNCTION
# ================================
def predict_news(model, tokenizer, label_encoder, device, text):
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
    label = label_encoder.inverse_transform([pred_id])[0]

    return label, confidence, probs[0].cpu().numpy()

# ================================
# GAUGE CHART
# ================================
def create_confidence_gauge(confidence, label):
    """Create a gauge chart for confidence"""
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
            'threshold': {'line': {'color': "black", 'width': 4}, 'value': 90}
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
    st.write("<p style='text-align:center; color:gray;'>Powered by AI • Analyze news articles in seconds</p>", unsafe_allow_html=True)
    st.divider()
    
    # Load model
    try:
        model, tokenizer, label_encoder, device = load_model()
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
    
    # Predict button
    predict_button = st.button("🔍 Analyze News", use_container_width=True)
    
    # Prediction
    if predict_button:
        if text_input.strip():
            with st.spinner("🤖 Analyzing..."):
                label, confidence, probs = predict_news(model, tokenizer, label_encoder, device, text_input)
                confidence_pct = round(confidence * 100, 2)
            
            # Result section
            if label.lower() == "real":
                st.success(f"✅ REAL NEWS", icon="✅")
                st.metric("Confidence", f"{confidence_pct:.2f}%")
            else:
                st.error(f"❌ FAKE NEWS", icon="❌")
                st.metric("Confidence", f"{confidence_pct:.2f}%")
            
            # Confidence gauge
            st.subheader("📊 Confidence Analysis")
            fig = create_confidence_gauge(confidence, label)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed probabilities
            with st.expander("📈 View Detailed Probabilities"):
                df = pd.DataFrame({
                    "Class": label_encoder.classes_,
                    "Probability (%)": [round(p*100, 2) for p in probs]
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
            
            # Explainable AI Analysis
            st.divider()
            display_xai_report(model, tokenizer, label_encoder, device, text_input, label, confidence)
            
        else:
            st.warning("⚠️ Please enter some text to analyze.")
    
    # Info section
    if not predict_button or not text_input.strip():
        st.info("""
        **ℹ️ How It Works**
        
        This AI-powered system uses **DistilBERT**, a state-of-the-art natural language processing model, to analyze news articles and determine their credibility.
        
        - 🎯 **High Accuracy:** Trained on thousands of verified news articles  
        - ⚡ **Instant Results:** Get predictions in seconds  
        - 🔒 **Privacy First:** Your text is processed locally and not stored  
        """)
    
    # Footer
    st.markdown("---")
    st.caption("⚡ Powered by DistilBERT • Made with Streamlit • © 2026 Fake News Detector")

# ================================
# RUN APP
# ================================
if __name__ == "__main__":
    user_home()