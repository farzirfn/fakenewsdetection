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
    """Load dataset distribution by status"""
    conn = create_connection()
    query = "SELECT status, COUNT(*) as count FROM dataset GROUP BY status"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_dataset_summary():
    """Load comprehensive dataset statistics"""
    conn = create_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Total records
    cursor.execute("SELECT COUNT(*) as total FROM dataset")
    total = cursor.fetchone()['total']
    
    # Records by subject
    cursor.execute("SELECT subject, COUNT(*) as count FROM dataset GROUP BY subject")
    subjects = cursor.fetchall()
    
    # Records by status
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
    """Load latest training result"""
    conn = create_connection()
    query = "SELECT * FROM train_results ORDER BY timestamp DESC LIMIT 1"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_training_history():
    """Load all training history for trends"""
    conn = create_connection()
    query = "SELECT * FROM train_results ORDER BY timestamp DESC LIMIT 10"
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
    st.write("<p style='text-align:center; color:gray;'>Real-time insights and model performance metrics</p>", unsafe_allow_html=True)
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
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div style="background-color:#eaf2f8; padding:20px; border-radius:10px; text-align:center; box-shadow:2px 2px 5px rgba(0,0,0,0.1);">
                <h4 style="color:#2E86C1;">📊 Total Records</h4>
                <h2 style="color:#000;">{dataset_summary['total']:,}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    
    with col2:
        num_subjects = len(dataset_summary['subjects'])
        st.markdown(
            f"""
            <div style="background-color:#eaf2f8; padding:20px; border-radius:10px; text-align:center; box-shadow:2px 2px 5px rgba(0,0,0,0.1);">
                <h4 style="color:#2E86C1;">📚 Subjects</h4>
                <h2 style="color:#000;">{num_subjects}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        num_statuses = len(dataset_summary['statuses'])
        st.markdown(
            f"""
            <div style="background-color:#eaf2f8; padding:20px; border-radius:10px; text-align:center; box-shadow:2px 2px 5px rgba(0,0,0,0.1);">
                <h4 style="color:#2E86C1;">🏷️ Status Types</h4>
                <h2 style="color:#000;">{num_statuses}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        if not df_train.empty:
            accuracy = float(df_train['accuracy'][0]) * 100
            st.markdown(
                f"""
                <div style="background-color:#eaf2f8; padding:20px; border-radius:10px; text-align:center; box-shadow:2px 2px 5px rgba(0,0,0,0.1);">
                    <h4 style="color:#2E86C1;">🎯 Model Accuracy</h4>
                    <h2 style="color:#000;">{accuracy:.1f}%</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color:#eaf2f8; padding:20px; border-radius:10px; text-align:center; box-shadow:2px 2px 5px rgba(0,0,0,0.1);">
                    <h4 style="color:#2E86C1;">🎯 Model Accuracy</h4>
                    <h2 style="color:#000;">N/A</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Dataset distribution charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
        """
        <div style="background-color:#f9f9f9; padding:15px; border-radius:10px; 
                    box-shadow:2px 2px 5px rgba(0,0,0,0.1);">
            <h4 style="text-align:center; color:#2E86C1;">Distribution by Status</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    df_status = dataset_summary['statuses']
    fig_status = px.pie(
        df_status,
        names="status",
        values="count",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig_status.update_layout(height=320)
    st.plotly_chart(fig_status, use_container_width=True)

    with col2:
        st.markdown(
        """
        <div style="background-color:#f9f9f9; padding:15px; border-radius:10px; 
                    box-shadow:2px 2px 5px rgba(0,0,0,0.1);">
            <h4 style="text-align:center; color:#16A085;">Distribution by Subject</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
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
    st.plotly_chart(fig_subject, use_container_width=True)

    
    # Detailed table
    with st.expander("📋 View Detailed Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**By Status:**")
            st.dataframe(df_status, use_container_width=True, hide_index=True)
        with col2:
            st.write("**By Subject:**")
            st.dataframe(df_subject, use_container_width=True, hide_index=True)
    
    # ================================
    # SECTION 2: Model Performance
    # ================================
    st.header("🎯 Model Performance")
    
    if df_train.empty:
        st.warning("⚠️ No Training Results Available\n\nTrain your model to see performance metrics here.")
        return
    
    # Latest training info
    st.info(f"**Last Trained:** {df_train['timestamp'][0]}\n\n**Training ID:** #{df_train['id'][0]}")
    
    # Performance metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    accuracy = float(df_train['accuracy'][0])
    precision = float(df_train['prec'][0])
    recall = float(df_train['recall'][0])
    f1 = float(df_train['f1'][0])
    
    with col1:
        st.markdown(
            f"""
            <div style="background-color:#eaf2f8; padding:20px; border-radius:10px; text-align:center; box-shadow:2px 2px 5px rgba(0,0,0,0.1);">
                <h4 style="color:#2E86C1;">🎯 Model Accuracy</h4>
                <h2 style="color:#000;">{accuracy*100:.2f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="background-color:#eaf2f8; padding:20px; border-radius:10px; text-align:center; box-shadow:2px 2px 5px rgba(0,0,0,0.1);">
                <h4 style="color:#2E86C1;">📊 Precision</h4>
                <h2 style="color:#000;">{precision*100:.2f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="background-color:#eaf2f8; padding:20px; border-radius:10px; text-align:center; box-shadow:2px 2px 5px rgba(0,0,0,0.1);">
                <h4 style="color:#2E86C1;">📈 Recall</h4>
                <h2 style="color:#000;">{recall*100:.2f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div style="background-color:#eaf2f8; padding:20px; border-radius:10px; text-align:center; box-shadow:2px 2px 5px rgba(0,0,0,0.1);">
                <h4 style="color:#2E86C1;">📊 F1 Score</h4>
                <h2 style="color:#000;">{f1*100:.2f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Visualizations
    col1, col2 = st.columns([1, 1])
    
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
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("Confusion Matrix")
        try:
            cm = np.array(eval(df_train["confusion_matrix"][0]))
            labels = ["Class 0", "Class 1"]
            if "classes" in df_train.columns and pd.notna(df_train["classes"][0]):
                try:
                    labels = list(eval(df_train["classes"][0]))
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
            st.plotly_chart(fig_cm, use_container_width=True)
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
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("📊 Train the model multiple times to see performance trends over time.")
    
    # Performance summary
    with st.expander("📊 Detailed Metrics Table"):
        detailed_metrics = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Score": [accuracy, precision, recall, f1],
            "Percentage": [f"{v*100:.2f}%" for v in [accuracy, precision, recall, f1]]
        })
        st.dataframe(detailed_metrics, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown(
        "<p style='text-align:center; color:gray;'>🔒 Secure Admin Dashboard </p>",
        unsafe_allow_html=True
    )
    st.divider()

# Run the stats page
if __name__ == "__main__":
    stats_page()