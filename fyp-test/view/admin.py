import streamlit as st
import pandas as pd
import mysql.connector
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Admin Dashboard", layout="wide")

# Custom minimalist CSS
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.block-container {
    padding-top: 2rem;
}
.metric-card {
    background: #111827;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.metric-title {
    color: #9ca3af;
    font-size: 14px;
}
.metric-value {
    color: white;
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# DB CONNECTION
# ===============================
def create_connection():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        port=st.secrets["mysql"]["port"],
        user=st.secrets["mysql"]["username"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"]
    )

# ===============================
# DATA LOADERS
# ===============================
def load_dataset_summary():
    conn = create_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT COUNT(*) as total FROM dataset")
    total = cursor.fetchone()['total']

    cursor.execute("SELECT subject, COUNT(*) as count FROM dataset GROUP BY subject")
    subjects = pd.DataFrame(cursor.fetchall())

    cursor.execute("SELECT status, COUNT(*) as count FROM dataset GROUP BY status")
    statuses = pd.DataFrame(cursor.fetchall())

    cursor.close()
    conn.close()

    return total, subjects, statuses


def load_train_results():
    conn = create_connection()
    df = pd.read_sql("SELECT * FROM train_results ORDER BY timestamp DESC LIMIT 1", conn)
    conn.close()
    return df

# ===============================
# UI COMPONENTS
# ===============================
def metric_card(title, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# MAIN DASHBOARD
# ===============================
def stats_page():

    st.markdown("<h2 style='text-align:center; color:white;'>Admin Dashboard</h2>", unsafe_allow_html=True)
    st.caption("Minimal • Clean • Real-time insights")
    st.divider()

    try:
        total, subjects, statuses = load_dataset_summary()
        df_train = load_train_results()
    except Exception as e:
        st.error(f"Error: {e}")
        return

    # ===========================
    # METRICS
    # ===========================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Total Records", f"{total:,}")

    with col2:
        metric_card("Subjects", len(subjects))

    with col3:
        metric_card("Statuses", len(statuses))

    with col4:
        if not df_train.empty:
            metric_card("Accuracy", f"{float(df_train['accuracy'][0])*100:.1f}%")
        else:
            metric_card("Accuracy", "N/A")

    st.divider()

    # ===========================
    # CHARTS
    # ===========================
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.pie(statuses, names="status", values="count", hole=0.6)
        fig1.update_layout(showlegend=True, height=300, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(subjects, x="subject", y="count")
        fig2.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ===========================
    # MODEL PERFORMANCE
    # ===========================
    if not df_train.empty:
        st.subheader("Model Performance")

        acc = float(df_train['accuracy'][0])
        prec = float(df_train['prec'][0])
        rec = float(df_train['recall'][0])
        f1 = float(df_train['f1'][0])

        df_metrics = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1"],
            "Value": [acc, prec, rec, f1]
        })

        fig3 = px.bar(df_metrics, x="Metric", y="Value", text="Value")
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

        try:
            cm = np.array(eval(df_train["confusion_matrix"][0]))
            fig_cm = go.Figure(data=go.Heatmap(z=cm, colorscale="Greys"))
            fig_cm.update_layout(height=300)
            st.plotly_chart(fig_cm, use_container_width=True)
        except:
            st.warning("No confusion matrix available")

    st.divider()
    st.caption("Secure Admin Panel • Minimal UI")


if __name__ == "__main__":
    stats_page()