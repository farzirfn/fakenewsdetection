import streamlit as st
import pandas as pd
import mysql.connector
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------
# Custom CSS — Minimalist Theme
# -------------------------------
def inject_css():
    st.markdown("""
    <style>
    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: show; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* Section labels */
    .section-label {
        font-size: 25px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #888780;
        margin-bottom: 0.5rem;
        margin-top: 1.5rem;
    }

    /* Metric cards */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-bottom: 1.25rem;
    }
    .metric-card {
        background: #f7f6f2;
        border-radius: 10px;
        padding: 14px 16px;
        border-left: 3px solid #ccc;
    }
    .metric-card.blue  { border-left-color: #378ADD; }
    .metric-card.green { border-left-color: #639922; }
    .metric-card.amber { border-left-color: #BA7517; }
    .metric-card.teal  { border-left-color: #1D9E75; }
    .metric-card .m-label {
        font-size: 15px;
        color: #888;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-card .m-value {
        font-size: 30px;
        font-weight: 500;
        color: #2c2c2a;
        line-height: 1.2;
    }
    .metric-card .m-sub {
        font-size: 15px;
        color: #aaa;
        margin-top: 2px;
    }

    /* Status pill */
    .pill-ok {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 11px;
        padding: 3px 10px;
        border-radius: 99px;
        background: #eaf3de;
        color: #3b6d11;
    }
    .pill-warn {
        background: #faeeda;
        color: #854f0b;
    }

    /* Divider */
    .thin-divider {
        border: none;
        border-top: 0.5px solid #e0ded8;
        margin: 1rem 0;
    }

    /* Dark mode overrides */
    @media (prefers-color-scheme: dark) {
        .metric-card { background: #1e1e1c; }
        .metric-card .m-value { color: #e0ded8; }
        .metric-card .m-label { color: #666; }
        .metric-card .m-sub   { color: #555; }
        .pill-ok  { background: #17340a; color: #9fd065; }
        .pill-warn{ background: #412402; color: #f9cb42; }
    }
    </style>
    """, unsafe_allow_html=True)


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
# Data Loaders
# -------------------------------
def load_dataset_summary():
    conn = create_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT COUNT(*) as total FROM dataset")
    total = cursor.fetchone()['total']

    cursor.execute("SELECT subject, COUNT(*) as count FROM dataset GROUP BY subject ORDER BY count DESC")
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
    query = "SELECT * FROM train_results ORDER BY timestamp DESC LIMIT 1"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_training_history():
    conn = create_connection()
    query = "SELECT * FROM train_results ORDER BY timestamp ASC"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# -------------------------------
# Plotly theme helper
# -------------------------------
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", size=12, color="#888780"),
    margin=dict(l=0, r=0, t=10, b=0),
    showlegend=False,
)

COLORS = {
    "blue":  "#378ADD",
    "green": "#639922",
    "amber": "#BA7517",
    "teal":  "#1D9E75",
    "red":   "#E24B4A",
    "gray":  "#B4B2A9",
}


# -------------------------------
# Admin Dashboard
# -------------------------------
def stats_page():
    inject_css()

    # Header
    st.title("📊 Admin Dashboard")
    st.write("Real-time insights and model performance metrics")

    st.divider()

    # Load data
    try:
        summary = load_dataset_summary()
        df_train = load_train_results()
    except Exception as e:
        st.error(f"❌ Database error: {str(e)}")
        return

    # ─── Section: Overview ───────────────────────────────────
    st.markdown("<div class='section-label'>Overview</div>", unsafe_allow_html=True)

    accuracy_val = float(df_train['accuracy'].iloc[0]) * 100 if not df_train.empty else None

    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-card blue'>
            <div class='m-label'>Total records</div>
            <div class='m-value'>{summary['total']:,}</div>
            <div class='m-sub'>Dataset entries</div>
        </div>
        <div class='metric-card green'>
            <div class='m-label'>Subjects</div>
            <div class='m-value'>{len(summary['subjects'])}</div>
            <div class='m-sub'>Unique categories</div>
        </div>
        <div class='metric-card amber'>
            <div class='m-label'>Status types</div>
            <div class='m-value'>{len(summary['statuses'])}</div>
            <div class='m-sub'>Real · Fake</div>
        </div>
        <div class='metric-card teal'>
            <div class='m-label'>Model accuracy</div>
            <div class='m-value'>{"N/A" if accuracy_val is None else f"{accuracy_val:.1f}%"}</div>
            <div class='m-sub'>Latest training run</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Charts: Subject + Status ─────────────────────────────
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown("<div class='section-label'>Distribution by subject</div>", unsafe_allow_html=True)
        df_subj = summary['subjects']
        fig = go.Figure(go.Bar(
            x=df_subj['count'],
            y=df_subj['subject'],
            orientation='h',
            marker=dict(
                color=df_subj['count'],
                colorscale=[[0, "#B5D4F4"], [1, "#185FA5"]],
                showscale=False
            ),
            text=df_subj['count'].apply(lambda x: f"{x:,}"),
            textposition='outside',
            textfont=dict(size=11, color="#888780"),
        ))
        fig.update_layout(
            **PLOT_LAYOUT,
            height=260,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=12)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown("<div class='section-label'>Distribution by status</div>", unsafe_allow_html=True)
        df_stat = summary['statuses']
        fig2 = go.Figure(go.Pie(
            labels=df_stat['status'],
            values=df_stat['count'],
            hole=0.62,
            marker=dict(colors=[COLORS["blue"], COLORS["red"]]),
            textinfo='none',
        ))
        fig2.add_annotation(
            text=f"{df_stat['count'].iloc[0]:,}",
            x=0.5, y=0.55,
            font=dict(size=18, color="#2c2c2a"),
            showarrow=False
        )
        fig2.add_annotation(
            text=df_stat['status'].iloc[0],
            x=0.5, y=0.4,
            font=dict(size=11, color="#888780"),
            showarrow=False
        )
        fig2.update_layout(**PLOT_LAYOUT, height=260)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ─── Section: Model Performance ──────────────────────────
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Model performance</div>", unsafe_allow_html=True)

    if df_train.empty:
        st.info("No training results yet. Train the model to see metrics here.")
        return

    acc  = float(df_train['accuracy'].iloc[0])
    prec = float(df_train['prec'].iloc[0])
    rec  = float(df_train['recall'].iloc[0])
    f1   = float(df_train['f1'].iloc[0])
    ts   = df_train['timestamp'].iloc[0]
    tid  = df_train['id'].iloc[0]

    # Perf metric cards
    col1, col2, col3, col4 = st.columns(4, gap="small")
    for col, name, val, color in zip(
        [col1, col2, col3, col4],
        ["Accuracy", "Precision", "Recall", "F1 score"],
        [acc, prec, rec, f1],
        ["blue", "green", "amber", "teal"]
    ):
        with col:
            st.markdown(f"""
            <div class='metric-card {color}'>
                <div class='m-label'>{name}</div>
                <div class='m-value'>{val:.4f}</div>
                <div class='m-sub'>{val*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Horizontal bar chart for metrics
    col1, col2 = st.columns([2, 3], gap="medium")

    with col1:
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 score"],
            "Score": [acc, prec, rec, f1],
            "Color": [COLORS["blue"], COLORS["green"], COLORS["amber"], COLORS["teal"]]
        })
        fig3 = go.Figure()
        for _, row in metrics_df.iterrows():
            fig3.add_trace(go.Bar(
                x=[row['Score']],
                y=[row['Metric']],
                orientation='h',
                marker=dict(color=row['Color']),
                text=[f"{row['Score']*100:.1f}%"],
                textposition='outside',
                textfont=dict(size=11, color="#888780"),
                name=row['Metric']
            ))
        fig3.update_layout(
            **PLOT_LAYOUT,
            height=220,
            barmode='stack',
            xaxis=dict(range=[0, 1.1], showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=12)),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with col2:
        try:
            cm = np.array(eval(df_train["confusion_matrix"].iloc[0]))
            labels = ["Real", "Fake"]
            if "classes" in df_train.columns and pd.notna(df_train["classes"].iloc[0]):
                try:
                    labels = list(eval(df_train["classes"].iloc[0]))
                except Exception:
                    pass

            fig4 = go.Figure(go.Heatmap(
                z=cm,
                x=[f"Pred {l}" for l in labels],
                y=[f"Actual {l}" for l in labels],
                text=cm.astype(int),
                texttemplate="%{text}",
                colorscale=[[0, "#E6F1FB"], [1, "#185FA5"]],
                showscale=False,
            ))
            fig4.update_layout(**PLOT_LAYOUT, height=220)
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Confusion matrix unavailable: {e}")

    # ─── Training History (original) ─────────────────────────
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Training History Trends</div>", unsafe_allow_html=True)
    df_history = load_training_history()
    if len(df_history) > 1:
        df_history = df_history.sort_values('timestamp')
        fig_trend = px.line(
            df_history,
            x="timestamp",
            y=["accuracy", "prec", "recall", "f1"],
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_trend.update_layout(height=320, hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("📊 Train the model multiple times to see performance trends over time.")
    st.markdown(f"""
    <div style='display:flex;align-items:center;justify-content:space-between;'>
        <span style='font-size:11px;color:#aaa;'>Training #{tid} · {ts}</span>
        <span class='pill-ok'>● Model ready</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='text-align:center; color:gray;'>⚡ Admin Dashboard</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    stats_page()