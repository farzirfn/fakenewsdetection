import streamlit as st
import pandas as pd
import mysql.connector
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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
# Load dataset statistics
# -------------------------------
def load_stats():
    conn = create_connection()
    query = "SELECT status, COUNT(*) as count FROM dataset GROUP BY status"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

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

# -------------------------------
# Load training results
# -------------------------------
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
# Admin Dashboard Page
# -------------------------------
def stats_page():
    # Header
    st.markdown(
        "<h2 style='text-align:center; color:#2E86C1;'>📊 Admin Dashboard</h2>",
        unsafe_allow_html=True
    )
    st.write(
        "<p style='text-align:center; color:gray;'>Real-time insights and model performance metrics</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # Load data
    try:
        dataset_summary = load_dataset_summary()
        df_train = load_train_results()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return

    # ================================
    # SECTION 1: Dataset Overview
    # ================================
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Total Records", f"{dataset_summary['total']:,}")

    with col2:
        st.metric("📚 Subjects", len(dataset_summary['subjects']))

    with col3:
        st.metric("🏷️ Status Types", len(dataset_summary['statuses']))

    with col4:
        if not df_train.empty:
            accuracy = float(df_train['accuracy'].iloc[0]) * 100
            st.metric("🎯 Model Accuracy", f"{accuracy:.1f}%")
        else:
            st.metric("🎯 Model Accuracy", "N/A")

    # Dataset distribution charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution by Status")
        df_status = dataset_summary['statuses']

        fig_status = px.pie(
            df_status,
            names="status",
            values="count",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig_status.update_layout(height=320)
        st.plotly_chart(fig_status, width='stretch')

    with col2:
        st.subheader("Distribution by Subject")
        df_subject = dataset_summary['subjects']

        fig_subject = px.bar(
            df_subject,
            x="subject",
            y="count",
            text="count",
            color="subject",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_subject.update_traces(textposition="outside")
        fig_subject.update_layout(height=320)
        st.plotly_chart(fig_subject, width='stretch')

    # Detailed table
    with st.expander("📋 View Detailed Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**By Status:**")
            st.dataframe(df_status, width='stretch', hide_index=True)
        with col2:
            st.write("**By Subject:**")
            st.dataframe(df_subject, width='stretch', hide_index=True)

    # ================================
    # SECTION 2: Model Performance
    # ================================
    st.header("🎯 Model Performance")

    if df_train.empty:
        st.warning("⚠️ No training results available. Train your model to see performance metrics here.")
        return

    # Latest training info
    st.info(f"**Last Trained:** {df_train['timestamp'].iloc[0]}   |   **Training ID:** #{df_train['id'].iloc[0]}")

    # Performance metrics
    accuracy  = float(df_train['accuracy'].iloc[0])
    precision = float(df_train['prec'].iloc[0])
    recall    = float(df_train['recall'].iloc[0])
    f1        = float(df_train['f1'].iloc[0])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy",  f"{accuracy*100:.2f}%")
    with col2:
        st.metric("Precision", f"{precision*100:.2f}%")
    with col3:
        st.metric("Recall",    f"{recall*100:.2f}%")
    with col4:
        st.metric("F1 Score",  f"{f1*100:.2f}%")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Performance Metrics")
        metrics_table = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [accuracy, precision, recall, f1]
        })

        fig_bar = px.bar(
            metrics_table,
            x="Metric",
            y="Value",
            text=[f"{v:.4f}" for v in metrics_table["Value"]],
            color="Metric",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(height=360, yaxis=dict(range=[0, 1.1]))
        st.plotly_chart(fig_bar, width='stretch')

    with col2:
        st.subheader("Confusion Matrix")
        try:
            cm = np.array(eval(df_train["confusion_matrix"].iloc[0]))
            labels = ["Class 0", "Class 1"]
            if "classes" in df_train.columns and pd.notna(df_train["classes"].iloc[0]):
                try:
                    labels = list(eval(df_train["classes"].iloc[0]))
                except Exception:
                    pass

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=[f"Pred {l}" for l in labels],
                y=[f"Actual {l}" for l in labels],
                text=cm.astype(int),
                texttemplate="%{text}",
                colorscale="Blues"
            ))
            fig_cm.update_layout(height=360)
            st.plotly_chart(fig_cm, width='stretch')
        except Exception as e:
            st.error(f"Error displaying confusion matrix: {str(e)}")

    # Training history trend
    st.subheader("Training History Trend")
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
        st.plotly_chart(fig_trend, width='stretch')
    else:
        st.info("📊 Train the model multiple times to see performance trends over time.")

    # Detailed metrics table
    with st.expander("📊 Detailed Metrics Table"):
        detailed_metrics = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Score": [accuracy, precision, recall, f1],
            "Percentage": [f"{v*100:.2f}%" for v in [accuracy, precision, recall, f1]]
        })
        st.dataframe(detailed_metrics, width='stretch', hide_index=True)

    # Footer
    st.divider()
    st.caption("🔒 Secure Admin Dashboard")


if __name__ == "__main__":
    stats_page()