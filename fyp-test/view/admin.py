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

# ===============================
# GLOBAL CSS (CLEAN UI)
# ===============================
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.card {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.metric-card {
    background-color: #eaf2f8;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.metric-title {
    color: #2E86C1;
    font-size: 14px;
}
.metric-value {
    font-size: 26px;
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
# LOAD DATA
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


def load_training_history():
    conn = create_connection()
    df = pd.read_sql("SELECT * FROM train_results ORDER BY timestamp DESC LIMIT 10", conn)
    conn.close()
    return df


# ===============================
# MAIN PAGE
# ===============================
def stats_page():

    st.markdown("<h2 style='text-align:center;'>📊 Admin Dashboard</h2>", unsafe_allow_html=True)
    st.caption("Minimal • Clean • Real-time insights")
    st.divider()

    try:
        total, subjects, statuses = load_dataset_summary()
        df_train = load_train_results()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return

    # ===========================
    # METRICS
    # ===========================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📊 Total Records</div>
            <div class="metric-value">{total:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📚 Subjects</div>
            <div class="metric-value">{len(subjects)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🏷️ Status Types</div>
            <div class="metric-value">{len(statuses)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        if not df_train.empty:
            acc = float(df_train['accuracy'][0]) * 100
            value = f"{acc:.1f}%"
        else:
            value = "N/A"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🎯 Accuracy</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ===========================
    # CHARTS (SIDE BY SIDE)
    # ===========================
    col1, col2 = st.columns(2)

    # LEFT CARD
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 style="text-align:center;">📊 Distribution by Status</h4>', unsafe_allow_html=True)

        fig1 = px.pie(
            statuses,
            names="status",
            values="count",
            hole=0.5,
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig1.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10))

        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT CARD
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 style="text-align:center;">📚 Distribution by Subject</h4>', unsafe_allow_html=True)

        fig2 = px.bar(
            subjects,
            x="subject",
            y="count",
            text="count",
            color="subject",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10))

        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ===========================
    # MODEL PERFORMANCE
    # ===========================
    st.divider()
    st.header("🎯 Model Performance")

    if df_train.empty:
        st.warning("No training results yet")
        return

    acc = float(df_train['accuracy'][0])
    prec = float(df_train['prec'][0])
    rec = float(df_train['recall'][0])
    f1 = float(df_train['f1'][0])

    col1, col2 = st.columns(2)

    with col1:
        df_metrics = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1"],
            "Value": [acc, prec, rec, f1]
        })

        fig3 = px.bar(df_metrics, x="Metric", y="Value", text="Value")
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        try:
            cm = np.array(eval(df_train["confusion_matrix"][0]))
            fig_cm = go.Figure(data=go.Heatmap(z=cm, colorscale="Blues"))
            fig_cm.update_layout(height=300)
            st.plotly_chart(fig_cm, use_container_width=True)
        except:
            st.warning("No confusion matrix available")

    # ===========================
    # TRAINING TREND
    # ===========================
    st.subheader("📈 Training Trend")
    df_history = load_training_history()

    if len(df_history) > 1:
        df_history = df_history.sort_values('timestamp')

        fig4 = px.line(
            df_history,
            x="timestamp",
            y=["accuracy", "prec", "recall", "f1"],
            markers=True
        )
        fig4.update_layout(height=300)

        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Train more to see trend")

    st.divider()
    st.caption("🔒 Secure Admin Dashboard")


# RUN
if __name__ == "__main__":
    stats_page()