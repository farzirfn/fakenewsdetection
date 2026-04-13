import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import pickle
import plotly.graph_objects as go
import plotly.express as px
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Verity · News Verifier",
    page_icon="◎",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ================================
# GLOBAL STYLES
# ================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 680px;
    }

    /* Hero badge */
    .hero-badge {
        display: inline-block;
        background: #F0F4FF;
        color: #3D52D5;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 5px 14px;
        border-radius: 20px;
        margin-bottom: 1.2rem;
    }

    /* Hero heading */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 600;
        color: #0F1117;
        line-height: 1.2;
        letter-spacing: -0.02em;
        margin: 0 0 0.6rem 0;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #6B7280;
        font-weight: 400;
        margin: 0 0 2.5rem 0;
        line-height: 1.6;
    }

    /* Section label */
    .section-label {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: #9CA3AF;
        margin-bottom: 0.5rem;
    }

    /* Textarea override */
    textarea {
        border: 1.5px solid #E5E7EB !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 15px !important;
        color: #1F2937 !important;
        padding: 14px 16px !important;
        background: #FAFAFA !important;
        transition: border-color 0.2s ease !important;
        resize: none !important;
    }
    textarea:focus {
        border-color: #3D52D5 !important;
        background: #fff !important;
        box-shadow: 0 0 0 3px rgba(61,82,213,0.08) !important;
    }

    /* Primary button */
    .stButton > button {
        background: #0F1117 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        padding: 0.7rem 1.5rem !important;
        letter-spacing: 0.01em;
        transition: background 0.2s ease, transform 0.1s ease !important;
    }
    .stButton > button:hover {
        background: #3D52D5 !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Result cards */
    .result-card {
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        margin: 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .result-real {
        background: #F0FDF4;
        border: 1.5px solid #BBF7D0;
    }
    .result-fake {
        background: #FFF1F2;
        border: 1.5px solid #FECDD3;
    }
    .result-icon {
        font-size: 2.2rem;
        line-height: 1;
    }
    .result-label {
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0 0 3px 0;
    }
    .result-real .result-label { color: #15803D; }
    .result-fake .result-label { color: #BE123C; }
    .result-desc {
        font-size: 13.5px;
        color: #6B7280;
        margin: 0;
    }
    .result-confidence {
        margin-left: auto;
        text-align: right;
    }
    .conf-pct {
        font-size: 1.6rem;
        font-weight: 600;
        line-height: 1;
    }
    .result-real .conf-pct { color: #15803D; }
    .result-fake .conf-pct { color: #BE123C; }
    .conf-label {
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #9CA3AF;
        margin-top: 3px;
    }

    /* How it works cards */
    .how-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 1rem;
    }
    .how-card {
        background: #FAFAFA;
        border: 1px solid #F3F4F6;
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
    }
    .how-icon {
        font-size: 1.4rem;
        margin-bottom: 8px;
    }
    .how-title {
        font-size: 13px;
        font-weight: 600;
        color: #1F2937;
        margin: 0 0 4px 0;
    }
    .how-desc {
        font-size: 12px;
        color: #9CA3AF;
        line-height: 1.5;
        margin: 0;
    }

    /* Divider */
    .subtle-divider {
        border: none;
        border-top: 1px solid #F3F4F6;
        margin: 2rem 0;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        font-size: 12px;
        color: #D1D5DB;
        margin-top: 3rem;
        letter-spacing: 0.03em;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #6B7280 !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #3D52D5 !important;
    }

    /* Warning / info */
    .stAlert {
        border-radius: 10px !important;
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)


# ================================
# MODEL LOADING
# ================================
@st.cache_resource
def load_model():
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
# PREDICTION
# ================================
def predict_news(model, tokenizer, label_encoder, device, text):
    encoding = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=256, return_tensors="pt"
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
        probs = F.softmax(outputs.logits, dim=1)
    pred_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_id].item()
    label = label_encoder.inverse_transform([pred_id])[0]
    return label, confidence, probs[0].cpu().numpy()


# ================================
# CHARTS
# ================================
def create_gauge(confidence, label):
    bar_color = "#15803D" if label.lower() == "real" else "#BE123C"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "family": "Inter", "color": bar_color}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 11, "family": "Inter"}, "tickcolor": "#D1D5DB"},
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "#F9FAFB",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#FEF2F2"},
                {"range": [50, 75], "color": "#FEF9C3"},
                {"range": [75, 100], "color": "#F0FDF4"},
            ],
        }
    ))
    fig.update_layout(
        height=200,
        margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def create_prob_bars(label_encoder, probs):
    classes = label_encoder.classes_
    colors = ["#15803D" if c.lower() == "real" else "#BE123C" for c in classes]
    fig = go.Figure()
    for i, (cls, prob, color) in enumerate(zip(classes, probs, colors)):
        fig.add_trace(go.Bar(
            x=[round(prob * 100, 1)],
            y=[cls.upper()],
            orientation="h",
            marker_color=color,
            marker_line_width=0,
            text=[f"{round(prob*100,1)}%"],
            textposition="outside",
            textfont={"size": 13, "family": "Inter", "color": "#374151"},
        ))
    fig.update_layout(
        height=130,
        margin=dict(t=5, b=5, l=0, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont={"size": 12, "family": "Inter", "color": "#6B7280"}),
        showlegend=False,
        barmode="overlay",
    )
    return fig


# ================================
# MAIN
# ================================
def user_home():

    # ── Hero ──────────────────────────────────────────────
    st.markdown('<div class="hero-badge">◎ AI-Powered Verification</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Is this news real?</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Paste any headline or article. Our AI will assess its credibility in seconds.</p>',
        unsafe_allow_html=True
    )

    # ── Load model ────────────────────────────────────────
    try:
        model, tokenizer, label_encoder, device = load_model()
    except Exception as e:
        st.error(f"Could not load model: {str(e)}")
        st.stop()

    # ── Input ─────────────────────────────────────────────
    st.markdown('<p class="section-label">Article or headline</p>', unsafe_allow_html=True)
    text_input = st.text_area(
        label="",
        height=140,
        placeholder="Paste a news headline or full article here…",
        label_visibility="collapsed"
    )
    analyze_btn = st.button("Analyze →", use_container_width=True)

    # ── Result ────────────────────────────────────────────
    if analyze_btn:
        if text_input.strip():
            with st.spinner("Analyzing…"):
                label, confidence, probs = predict_news(model, tokenizer, label_encoder, device, text_input)
                conf_pct = round(confidence * 100, 1)

            is_real = label.lower() == "real"
            card_class = "result-real" if is_real else "result-fake"
            icon = "✓" if is_real else "✕"
            verdict = "Real News" if is_real else "Fake News"
            desc = "This article appears credible based on our analysis." if is_real else "This article shows patterns consistent with misinformation."

            st.markdown(f"""
            <div class="result-card {card_class}">
                <div class="result-icon">{icon}</div>
                <div>
                    <p class="result-label">{verdict}</p>
                    <p class="result-desc">{desc}</p>
                </div>
                <div class="result-confidence">
                    <div class="conf-pct">{conf_pct}%</div>
                    <div class="conf-label">Confidence</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Gauge + probability bars ───────────────────
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown('<p class="section-label" style="margin-top:1rem">Confidence meter</p>', unsafe_allow_html=True)
                st.plotly_chart(create_gauge(confidence, label), use_container_width=True, config={"displayModeBar": False})
            with col2:
                st.markdown('<p class="section-label" style="margin-top:1rem">Class probabilities</p>', unsafe_allow_html=True)
                st.plotly_chart(create_prob_bars(label_encoder, probs), use_container_width=True, config={"displayModeBar": False})

        else:
            st.warning("Please enter some text before analyzing.")

    # ── How it works (shown when idle) ────────────────────
    if not analyze_btn or not text_input.strip():
        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">How it works</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="how-grid">
            <div class="how-card">
                <div class="how-icon">⌨</div>
                <p class="how-title">Paste text</p>
                <p class="how-desc">Any headline, paragraph, or full article works</p>
            </div>
            <div class="how-card">
                <div class="how-icon">⚡</div>
                <p class="how-title">AI analyzes</p>
                <p class="how-desc">DistilBERT reads patterns across thousands of signals</p>
            </div>
            <div class="how-card">
                <div class="how-icon">◎</div>
                <p class="how-title">Get verdict</p>
                <p class="how-desc">Instant result with a confidence score</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────
    st.markdown(
        '<p class="app-footer">Verity · Powered by DistilBERT · © 2026</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    user_home()