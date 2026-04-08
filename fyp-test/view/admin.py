import streamlit as st
import pandas as pd
import mysql.connector
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Global Styles
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400&family=DM+Sans:wght@300;400;500&display=swap');

/* Reset & Base */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F5F2;
    color: #1A1A1A;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 3rem 4rem 2rem 4rem; max-width: 1400px; }

/* ---- Typography ---- */
h1, h2, h3 { font-family: 'DM Serif Display', serif; font-weight: 400; }

/* ---- Page Header ---- */
.page-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding-bottom: 2.5rem;
    border-bottom: 1px solid #E0DDD8;
    margin-bottom: 3rem;
}
.page-header h1 {
    font-size: 2.6rem;
    letter-spacing: -0.02em;
    color: #1A1A1A;
    line-height: 1;
}
.page-header span {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #999;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ---- Section Labels ---- */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #E0DDD8;
}

/* ---- Metric Cards ---- */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E8E5E0;
    border-radius: 4px;
    padding: 1.8rem 2rem;
    position: relative;
    transition: border-color 0.2s ease;
}
.metric-card:hover { border-color: #C8C4BE; }
.metric-card .label {
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #AAAAAA;
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.9rem;
}
.metric-card .value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #1A1A1A;
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: #AAAAAA;
    margin-top: 0.5rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
}
.metric-card .accent-bar {
    position: absolute;
    top: 0; left: 0;
    width: 3px;
    height: 100%;
    border-radius: 4px 0 0 4px;
    background: #1A1A1A;
}

/* ---- Status Badge ---- */
.status-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.25rem 0.6rem;
    border-radius: 2px;
    background: #F0EDE8;
    color: #666;
}

/* ---- Info Bar ---- */
.info-bar {
    background: #FFFFFF;
    border: 1px solid #E8E5E0;
    border-left: 3px solid #1A1A1A;
    border-radius: 4px;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}
.info-bar .info-item { display: flex; flex-direction: column; gap: 0.2rem; }
.info-bar .info-item span:first-child {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #AAAAAA;
}
.info-bar .info-item span:last-child { font-size: 0.85rem; color: #1A1A1A; }

/* ---- Divider ---- */
.thin-divider { border: none; border-top: 1px solid #E8E5E0; margin: 3rem 0; }

/* ---- Warning / Error ---- */
.stAlert { border-radius: 4px; }

/* ---- Dataframe ---- */
.stDataFrame { border: 1px solid #E8E5E0; border-radius: 4px; overflow: hidden; }

/* ---- Expander ---- */
.streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #999 !important;
    background: transparent !important;
    border: 1px solid #E8E5E0 !important;
    border-radius: 4px !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Plotly theme helper
# -------------------------------
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#555", size=12),
    margin=dict(l=0, r=0, t=20, b=0),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="left", x=0,
        font=dict(size=11, color="#888")
    ),
    xaxis=dict(showgrid=False, zeroline=False, color="#AAA", tickfont=dict(size=11)),
    yaxis=dict(showgrid=True, gridcolor="#F0EDE8", zeroline=False, color="#AAA", tickfont=dict(size=11)),
)

MONO_PALETTE = ["#1A1A1A", "#555555", "#999999", "#CCCCCC", "#E8E5E0"]
ACCENT_PALETTE = ["#1A1A1A", "#8B7355", "#C4956A", "#D4B483", "#E8D5B0"]

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
# Data loaders
# -------------------------------
def load_dataset_summary():
    conn = create_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT COUNT(*) as total FROM dataset")
    total = cursor.fetchone()['total']
    cursor.execute("SELECT subject, COUNT(*) as count FROM dataset GROUP BY subject")
    subjects = cursor.fetchall()
    cursor.execute("SELECT status, COUNT(*) as count FROM dataset GROUP BY status")
    statuses = cursor.fetchall()
    cursor.close()
    conn.close()
    return {
        'total': total,
        'subjects': pd.DataFrame(subjects),
        'statuses': pd.DataFrame(statuses)
    }

def load_train_results():
    conn = create_connection()
    df = pd.read_sql("SELECT * FROM train_results ORDER BY timestamp DESC LIMIT 1", conn)
    conn.close()
    return df

def load_training_history():
    conn = create_connection()
    df = pd.read_sql("SELECT * FROM train_results ORDER BY timestamp DESC LIMIT 10", conn)
    conn.close()
    return df

# -------------------------------
# Metric card HTML
# -------------------------------
def metric_card(label, value, sub=None, accent=True):
    accent_bar = '<div class="accent-bar"></div>' if accent else ''
    sub_html = f'<div class="sub">{sub}</div>' if sub else ''
    return f"""
    <div class="metric-card">
        {accent_bar}
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {sub_html}
    </div>"""

# -------------------------------
# Dashboard
# -------------------------------
def stats_page():
    # Page header
    st.markdown("""
    <div class="page-header">
        <h1>Admin Dashboard</h1>
        <span>Performance &amp; Analytics</span>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    try:
        dataset_summary = load_dataset_summary()
        df_train = load_train_results()
    except Exception as e:
        st.error(f"Could not connect to database: {str(e)}")
        return

    # ── SECTION 1: Dataset ──────────────────────────────────────────
    st.markdown('<div class="section-label">Dataset Overview</div>', unsafe_allow_html=True)

    accuracy_val = float(df_train['accuracy'][0]) * 100 if not df_train.empty else None
    accuracy_display = f"{accuracy_val:.1f}%" if accuracy_val else "—"

    cols = st.columns(4, gap="medium")
    cards = [
        ("Total Records",  f"{dataset_summary['total']:,}", None),
        ("Subjects",       str(len(dataset_summary['subjects'])), None),
        ("Status Types",   str(len(dataset_summary['statuses'])), None),
        ("Model Accuracy", accuracy_display, "latest run"),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(metric_card(label, value, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="section-label" style="margin-top:0;">Status distribution</div>', unsafe_allow_html=True)
        df_status = dataset_summary['statuses']
        fig = go.Figure(go.Pie(
            labels=df_status['status'],
            values=df_status['count'],
            hole=0.6,
            marker=dict(colors=ACCENT_PALETTE, line=dict(color="#F7F5F2", width=3)),
            textinfo="percent",
            textfont=dict(size=11, family="DM Mono, monospace"),
            hovertemplate="<b>%{label}</b><br>%{value} records<extra></extra>"
        ))
        fig.update_layout(**{**PLOT_LAYOUT, "height": 300, "showlegend": True})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        st.markdown('<div class="section-label" style="margin-top:0;">Subject distribution</div>', unsafe_allow_html=True)
        df_subject = dataset_summary['subjects']
        fig = go.Figure(go.Bar(
            x=df_subject['subject'],
            y=df_subject['count'],
            marker=dict(color="#1A1A1A", line=dict(width=0)),
            text=df_subject['count'],
            textposition="outside",
            textfont=dict(family="DM Mono, monospace", size=10, color="#888"),
            hovertemplate="<b>%{x}</b><br>%{y} records<extra></extra>"
        ))
        fig.update_layout(**{**PLOT_LAYOUT, "height": 300, "showlegend": False})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander("View raw table"):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.dataframe(df_status, use_container_width=True, hide_index=True)
        with c2:
            st.dataframe(df_subject, use_container_width=True, hide_index=True)

    # ── SECTION 2: Model Performance ────────────────────────────────
    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)

    if df_train.empty:
        st.info("No training results found. Train your model to see metrics here.")
        return

    # Info bar
    st.markdown(f"""
    <div class="info-bar">
        <div class="info-item">
            <span>Training ID</span>
            <span>#{df_train['id'][0]}</span>
        </div>
        <div class="info-item">
            <span>Last Trained</span>
            <span>{df_train['timestamp'][0]}</span>
        </div>
        <div class="info-item">
            <span>Status</span>
            <span class="status-badge">Complete</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    accuracy  = float(df_train['accuracy'][0])
    precision = float(df_train['prec'][0])
    recall    = float(df_train['recall'][0])
    f1        = float(df_train['f1'][0])

    metric_cols = st.columns(4, gap="medium")
    perf_cards = [
        ("Accuracy",  f"{accuracy:.4f}",  f"{accuracy*100:.1f}%"),
        ("Precision", f"{precision:.4f}", f"{precision*100:.1f}%"),
        ("Recall",    f"{recall:.4f}",    f"{recall*100:.1f}%"),
        ("F1 Score",  f"{f1:.4f}",        f"{f1*100:.1f}%"),
    ]
    for col, (label, value, sub) in zip(metric_cols, perf_cards):
        with col:
            st.markdown(metric_card(label, value, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="section-label" style="margin-top:0;">Metric overview</div>', unsafe_allow_html=True)
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Score":  [accuracy, precision, recall, f1]
        })
        fig = go.Figure(go.Bar(
            x=metrics_df["Metric"],
            y=metrics_df["Score"],
            marker=dict(
                color=["#1A1A1A", "#555555", "#8B7355", "#C4956A"],
                line=dict(width=0)
            ),
            text=[f"{v:.4f}" for v in metrics_df["Score"]],
            textposition="outside",
            textfont=dict(family="DM Mono, monospace", size=10, color="#888"),
            hovertemplate="<b>%{x}</b><br>%{y:.4f}<extra></extra>"
        ))
        fig.update_layout(**{
            **PLOT_LAYOUT,
            "height": 320,
            "showlegend": False,
            "yaxis": dict(range=[0, 1.15], showgrid=True, gridcolor="#F0EDE8", zeroline=False, color="#AAA", tickfont=dict(size=11))
        })
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        st.markdown('<div class="section-label" style="margin-top:0;">Confusion matrix</div>', unsafe_allow_html=True)
        try:
            cm = np.array(eval(df_train["confusion_matrix"][0]))
            labels = ["Class 0", "Class 1"]
            if "classes" in df_train.columns and pd.notna(df_train["classes"][0]):
                try:
                    labels = list(eval(df_train["classes"][0]))
                except Exception:
                    pass

            fig = go.Figure(go.Heatmap(
                z=cm,
                x=[f"Pred · {l}" for l in labels],
                y=[f"Actual · {l}" for l in labels],
                text=cm.astype(int),
                texttemplate="%{text}",
                textfont=dict(family="DM Mono, monospace", size=14, color="#FFF"),
                colorscale=[[0, "#F7F5F2"], [1, "#1A1A1A"]],
                showscale=False,
                hovertemplate="%{y} → %{x}<br>Count: %{z}<extra></extra>"
            ))
            fig.update_layout(**{**PLOT_LAYOUT, "height": 320, "showlegend": False})
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.error(f"Error displaying confusion matrix: {str(e)}")

    # Training trend
    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Training history</div>', unsafe_allow_html=True)

    df_history = load_training_history()
    if len(df_history) > 1:
        df_history = df_history.sort_values('timestamp')
        fig = go.Figure()
        line_styles = [
            ("accuracy", "#1A1A1A", "solid"),
            ("prec",     "#8B7355", "dot"),
            ("recall",   "#C4956A", "dash"),
            ("f1",       "#999999", "longdash"),
        ]
        for col_name, color, dash in line_styles:
            fig.add_trace(go.Scatter(
                x=df_history["timestamp"],
                y=df_history[col_name],
                name=col_name.replace("prec", "precision").title(),
                mode="lines+markers",
                line=dict(color=color, width=2, dash=dash),
                marker=dict(color=color, size=5),
                hovertemplate=f"<b>{col_name}</b><br>%{{y:.4f}}<extra></extra>"
            ))
        fig.update_layout(**{**PLOT_LAYOUT, "height": 300, "hovermode": "x unified"})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Train the model multiple times to see performance trends.")

    with st.expander("View detailed metrics"):
        detail_df = pd.DataFrame({
            "Metric":     ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Score":      [accuracy, precision, recall, f1],
            "Percentage": [f"{v*100:.2f}%" for v in [accuracy, precision, recall, f1]]
        })
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    # Footer
    st.markdown("""
    <hr class="thin-divider">
    <p style="text-align:center; font-family:'DM Mono',monospace; font-size:0.65rem;
              letter-spacing:0.12em; text-transform:uppercase; color:#CCC; padding-bottom:2rem;">
        Secure Admin Dashboard
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    stats_page()