import streamlit as st
import pandas as pd
import mysql.connector
import numpy as np
import plotly.graph_objects as go

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Admin Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Global Styles
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D0D0D;
    color: #E8E4DE;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 3rem 3rem 3rem;
    max-width: 1300px;
    margin: 0 auto;
}

/* ── Page Header ── */
.page-title {
    font-size: 1.9rem;
    font-weight: 400;
    color: #E8E4DE;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.page-subtitle {
    font-size: 0.82rem;
    color: #555;
    font-weight: 300;
    margin-bottom: 2.2rem;
}

/* ── Section Header ── */
.section-header {
    font-size: 0.63rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #444;
    font-weight: 500;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}

/* ── Metric Card ── */
.metric-card {
    background: #141414;
    border: 1px solid #1E1E1E;
    border-radius: 12px;
    padding: 1.4rem 1.6rem 1.2rem 1.6rem;
    height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.mc-label { font-size: 0.75rem; color: #555; font-weight: 400; }
.mc-value { font-size: 2.15rem; font-weight: 300; color: #E8E4DE; letter-spacing: -0.03em; line-height: 1; }
.mc-sub   { font-size: 0.72rem; color: #444; font-weight: 300; }

/* ── Chart Card ── */
.chart-card {
    background: #141414;
    border: 1px solid #1E1E1E;
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
}
.cc-title { font-size: 0.95rem; font-weight: 400; color: #C8C4BE; margin-bottom: 1.4rem; }

/* ── Performance outer card ── */
.perf-outer-card {
    background: #141414;
    border: 1px solid #1E1E1E;
    border-radius: 12px;
    padding: 1.6rem 1.8rem 1.2rem 1.8rem;
}

/* ── Perf metric mini-card ── */
.perf-card {
    background: #1A1A1A;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.pc-label { font-size: 0.7rem; color: #555; margin-bottom: 0.5rem; font-weight: 400; }
.pc-value { font-size: 1.65rem; font-weight: 300; color: #E8E4DE; letter-spacing: -0.02em; line-height: 1; }
.pc-pct   { font-size: 0.7rem; color: #444; margin-top: 0.35rem; }

/* ── Progress row ── */
.prog-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.85rem;
}
.prog-label { font-size: 0.82rem; color: #777; width: 72px; flex-shrink: 0; font-weight: 400; }
.prog-track { flex: 1; height: 7px; background: #1E1E1E; border-radius: 99px; overflow: hidden; }
.prog-fill  { height: 100%; border-radius: 99px; }
.prog-val   { font-size: 0.78rem; color: #666; width: 42px; text-align: right; flex-shrink: 0; }

/* ── Footer bar ── */
.footer-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 1.2rem;
    padding-top: 1rem;
    border-top: 1px solid #1E1E1E;
}
.footer-meta { font-size: 0.72rem; color: #3A3A3A; }
.ready-badge {
    font-size: 0.7rem; color: #777;
    background: #1A1A1A; border: 1px solid #252525;
    border-radius: 99px; padding: 0.28rem 0.9rem;
}

/* ── Donut legend ── */
.donut-legend { display: flex; flex-direction: column; gap: 0.6rem; justify-content: center; height: 100%; }
.donut-legend-item { display: flex; align-items: center; gap: 0.55rem; font-size: 0.8rem; color: #888; }
.donut-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }

</style>
""", unsafe_allow_html=True)

# ── Plotly base ───────────────────────────────────────────────────────────────
PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#555", size=11),
    margin=dict(l=0, r=0, t=4, b=0),
    showlegend=False,
)

SUBJECT_COLORS = ["#4C8EF7", "#3DBF7F", "#D97B3A", "#7DC94B", "#E05A4E", "#777777"]
STATUS_COLORS  = ["#4C8EF7", "#E05A4E", "#3DBF7F", "#D97B3A"]
PROG_COLORS    = ["#4C8EF7", "#3DBF7F", "#D97B3A", "#7DC94B"]

# ── DB helpers ────────────────────────────────────────────────────────────────
def create_connection():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        port=st.secrets["mysql"]["port"],
        user=st.secrets["mysql"]["username"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"]
    )

def load_dataset_summary():
    conn = create_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT COUNT(*) as total FROM dataset")
    total = cur.fetchone()['total']
    cur.execute("SELECT subject, COUNT(*) as count FROM dataset GROUP BY subject ORDER BY count DESC")
    subjects = cur.fetchall()
    cur.execute("SELECT status, COUNT(*) as count FROM dataset GROUP BY status ORDER BY count DESC")
    statuses = cur.fetchall()
    cur.close(); conn.close()
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

# ── HTML builders ─────────────────────────────────────────────────────────────
def metric_card(label, value, sub):
    return f"""<div class="metric-card">
        <div class="mc-label">{label}</div>
        <div class="mc-value">{value}</div>
        <div class="mc-sub">{sub}</div>
    </div>"""

def perf_card(label, value, pct):
    return f"""<div class="perf-card">
        <div class="pc-label">{label}</div>
        <div class="pc-value">{value}</div>
        <div class="pc-pct">{pct}</div>
    </div>"""

def prog_row(label, value, color):
    pct = value * 100
    return f"""<div class="prog-row">
        <span class="prog-label">{label}</span>
        <div class="prog-track"><div class="prog-fill" style="width:{pct:.1f}%;background:{color};"></div></div>
        <span class="prog-val">{pct:.1f}%</span>
    </div>"""

# ── Dashboard ─────────────────────────────────────────────────────────────────
def stats_page():

    try:
        summary  = load_dataset_summary()
        df_train = load_train_results()
    except Exception as e:
        st.error(f"Database error: {e}")
        return

    # Values
    accuracy  = float(df_train['accuracy'][0]) if not df_train.empty else 0
    precision = float(df_train['prec'][0])     if not df_train.empty else 0
    recall    = float(df_train['recall'][0])   if not df_train.empty else 0
    f1        = float(df_train['f1'][0])       if not df_train.empty else 0
    df_status  = summary['statuses']
    df_subject = summary['subjects']
    status_labels = " · ".join(df_status['status'].tolist()) if not df_status.empty else "—"

    # ── Header ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-title">Admin dashboard</div>
    <div class="page-subtitle">Fake news detection · Last updated just now</div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    # OVERVIEW — 4 metric cards
    # ════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1: st.markdown(metric_card("Total records",  f"{summary['total']:,}", "Dataset entries"),      unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Subjects",       str(len(df_subject)),    "Unique categories"),    unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Status types",   str(len(df_status)),     status_labels),         unsafe_allow_html=True)
    with c4: st.markdown(metric_card("Model accuracy", f"{accuracy*100:.1f}%",  "Latest run"),          unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Two distribution charts ───────────────────────────────────────
    col_l, col_r = st.columns([3, 2], gap="small")

    # LEFT — horizontal bar chart by subject
    with col_l:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="cc-title">Distribution by subject</div>', unsafe_allow_html=True)

        max_cnt = df_subject['count'].max() if not df_subject.empty else 1
        html = ""
        for i, row in df_subject.iterrows():
            color = SUBJECT_COLORS[i % len(SUBJECT_COLORS)]
            pct   = row['count'] / max_cnt
            k     = f"{row['count']/1000:.1f}k" if row['count'] >= 1000 else str(row['count'])
            html += f"""
            <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.85rem;">
                <span style="font-size:0.82rem;color:#777;width:68px;flex-shrink:0;">{row['subject']}</span>
                <div style="flex:1;height:7px;background:#1E1E1E;border-radius:99px;overflow:hidden;">
                    <div style="width:{pct*100:.1f}%;height:100%;background:{color};border-radius:99px;"></div>
                </div>
                <span style="font-size:0.78rem;color:#555;width:38px;text-align:right;flex-shrink:0;">{k}</span>
            </div>"""
        st.markdown(html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT — donut by status
    with col_r:
        st.markdown('<div class="chart-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<div class="cc-title">Distribution by status</div>', unsafe_allow_html=True)

        if not df_status.empty:
            total_s   = df_status['count'].sum()
            top       = df_status.iloc[0]
            top_pct   = int(top['count'] / total_s * 100)

            fig = go.Figure(go.Pie(
                labels=df_status['status'],
                values=df_status['count'],
                hole=0.68,
                marker=dict(colors=STATUS_COLORS[:len(df_status)], line=dict(color="#141414", width=3)),
                textinfo="none",
                hovertemplate="<b>%{label}</b><br>%{value:,}<extra></extra>"
            ))
            fig.add_annotation(text=f"<b>{top_pct}%</b>", x=0.5, y=0.58, showarrow=False,
                               font=dict(size=20, color="#E8E4DE", family="Inter"), xanchor="center")
            fig.add_annotation(text=top['status'].lower(), x=0.5, y=0.42, showarrow=False,
                               font=dict(size=11, color="#555", family="Inter"), xanchor="center")
            fig.update_layout(**PLOT_BASE, height=175, margin=dict(l=0,r=0,t=0,b=0))

            d_col, l_col = st.columns([1, 1], gap="small")
            with d_col:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            with l_col:
                leg = '<div class="donut-legend">'
                for i, row in df_status.iterrows():
                    color = STATUS_COLORS[i % len(STATUS_COLORS)]
                    leg  += f'<div class="donut-legend-item"><div class="donut-dot" style="background:{color};"></div><span>{row["status"]} — {row["count"]:,}</span></div>'
                leg += '</div>'
                st.markdown(leg, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    # MODEL PERFORMANCE
    # ════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

    if df_train.empty:
        st.info("No training results found. Train your model first.")
        return

    st.markdown('<div class="perf-outer-card">', unsafe_allow_html=True)

    # 4 metric mini-cards
    p1, p2, p3, p4 = st.columns(4, gap="small")
    for col, (lbl, val) in zip(
        [p1, p2, p3, p4],
        [("Accuracy", accuracy), ("Precision", precision), ("Recall", recall), ("F1 score", f1)]
    ):
        with col:
            st.markdown(perf_card(lbl, f"{val:.4f}", f"{val*100:.2f}%"), unsafe_allow_html=True)

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

    # Progress bars
    bars = ""
    for lbl, val, color in [
        ("Accuracy",  accuracy,  PROG_COLORS[0]),
        ("Precision", precision, PROG_COLORS[1]),
        ("Recall",    recall,    PROG_COLORS[2]),
        ("F1 score",  f1,        PROG_COLORS[3]),
    ]:
        bars += prog_row(lbl, val, color)
    st.markdown(bars, unsafe_allow_html=True)

    # Footer
    tid = df_train['id'][0]        if not df_train.empty else "—"
    tts = df_train['timestamp'][0] if not df_train.empty else "—"
    st.markdown(f"""
    <div class="footer-bar">
        <span class="footer-meta">Training #{tid} · {tts}</span>
        <span class="ready-badge">Model ready</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    stats_page()