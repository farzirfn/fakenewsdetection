import streamlit as st
import pandas as pd
import mysql.connector
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------
# Page Configuration & CSS
# -------------------------------
st.set_page_config(page_title="Admin Dashboard", layout="wide")

def inject_custom_css():
    st.markdown("""
        <style>
        /* Minimalist typography and spacing */
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            color: #1E293B;
        }
        .subtitle {
            color: #64748B;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        /* Metric card styling */
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #E2E8F0;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
            transition: all 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            box-shadow: 0 4px 6px rgba(0,0,0,0.04);
            border-color: #CBD5E1;
        }
        /* Minimalist expanders */
        .streamlit-expanderHeader {
            font-weight: 500;
            color: #475569;
        }
        </style>
    """, unsafe_allow_html=True)

# -------------------------------
# DB Connection & Data Loading
# -------------------------------
@st.cache_resource
def create_connection():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        port=st.secrets["mysql"]["port"],
        user=st.secrets["mysql"]["username"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"]
    )

@st.cache_data(ttl=60)
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

@st.cache_data(ttl=60)
def load_train_results():
    conn = create_connection()
    query = "SELECT * FROM train_results ORDER BY timestamp DESC LIMIT 1"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

@st.cache_data(ttl=60)
def load_training_history():
    conn = create_connection()
    query = "SELECT * FROM train_results ORDER BY timestamp DESC LIMIT 10"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# -------------------------------
# Chart Styling Helper
# -------------------------------
def apply_minimalist_layout(fig):
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color="#475569", family="Inter, sans-serif"),
        showlegend=False
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color="#94A3B8")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False, color="#94A3B8")
    return fig

# -------------------------------
# Admin Dashboard Page
# -------------------------------
def stats_page():
    inject_custom_css()
    
    # Clean Header
    st.markdown("<h1>System Overview</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Dataset metrics and model performance analytics</div>", unsafe_allow_html=True)
    
    try:
        dataset_summary = load_dataset_summary()
        df_train = load_train_results()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # ================================
    # SECTION 1: Dataset Overview
    # ================================
    st.markdown("<h3>Dataset Distribution</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{dataset_summary['total']:,}")
    with col2:
        st.metric("Subjects", len(dataset_summary['subjects']))
    with col3:
        st.metric("Status Types", len(dataset_summary['statuses']))
    with col4:
        if not df_train.empty:
            accuracy = float(df_train['accuracy'][0]) * 100
            st.metric("Model Accuracy", f"{accuracy:.1f}%")
        else:
            st.metric("Model Accuracy", "N/A")
            
    st.write("") # Spacer
    
    col1, col2 = st.columns(2)
    with col1:
        df_status = dataset_summary['statuses']
        fig_status = px.pie(
            df_status, names="status", values="count", hole=0.6,
            color_discrete_sequence=["#3B82F6", "#93C5FD", "#1E40AF", "#DBEAFE"]
        )
        fig_status.update_traces(textinfo='percent+label', textposition='outside', hoverinfo='label+percent+name')
        fig_status = apply_minimalist_layout(fig_status)
        fig_status.update_layout(title_text="By Status", title_x=0.5, title_font=dict(size=14))
        st.plotly_chart(fig_status, use_container_width=True)
        
    with col2:
        df_subject = dataset_summary['subjects']
        fig_subject = px.bar(
            df_subject, x="subject", y="count",
            color_discrete_sequence=["#64748B"]
        )
        fig_subject.update_traces(marker_line_width=0, opacity=0.8)
        fig_subject = apply_minimalist_layout(fig_subject)
        fig_subject.update_layout(title_text="By Subject", title_x=0.5, title_font=dict(size=14))
        st.plotly_chart(fig_subject, use_container_width=True)

    with st.expander("View Raw Data"):
        c1, c2 = st.columns(2)
        c1.dataframe(df_status, use_container_width=True, hide_index=True)
        c2.dataframe(df_subject, use_container_width=True, hide_index=True)

    st.markdown("<br><br>", unsafe_allow_html=True) # Spacer

    # ================================
    # SECTION 2: Model Performance
    # ================================
    st.markdown("<h3>Model Performance</h3>", unsafe_allow_html=True)
    
    if df_train.empty:
        st.info("No training results available yet.")
        return
        
    st.caption(f"Latest Build: {df_train['timestamp'][0]} • ID: {df_train['id'][0]}")
    
    accuracy = float(df_train['accuracy'][0])
    precision = float(df_train['prec'][0])
    recall = float(df_train['recall'][0])
    f1 = float(df_train['f1'][0])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")
    col4.metric("F1 Score", f"{f1:.3f}")
    
    st.write("") # Spacer
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        df_history = load_training_history()
        if len(df_history) > 1:
            df_history = df_history.sort_values('timestamp')
            fig_trend = px.line(
                df_history, x="timestamp", y=["accuracy", "prec", "recall", "f1"],
                color_discrete_sequence=["#0F172A", "#3B82F6", "#10B981", "#6366F1"]
            )
            fig_trend.update_traces(line=dict(width=2))
            fig_trend = apply_minimalist_layout(fig_trend)
            fig_trend.update_layout(
                title_text="Performance History", title_x=0.05, title_font=dict(size=14),
                showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.write("Insufficient data for trend analysis.")
            
    with col2:
        try:
            cm = np.array(eval(df_train["confusion_matrix"][0]))
            labels = ["Class 0", "Class 1"]
            if "classes" in df_train.columns and pd.notna(df_train["classes"][0]):
                try:
                    labels = list(eval(df_train["classes"][0]))
                except Exception:
                    pass
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=labels, y=labels,
                text=cm.astype(int), texttemplate="%{text}",
                colorscale="Blues", showscale=False,
                hoverinfo="skip"
            ))
            fig_cm = apply_minimalist_layout(fig_cm)
            fig_cm.update_layout(
                title_text="Confusion Matrix", title_x=0.5, title_font=dict(size=14),
                xaxis_title="Predicted", yaxis_title="Actual",
                yaxis=dict(autorange="reversed") # Standard CM orientation
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        except Exception:
            st.write("Confusion matrix data unavailable.")

    # Footer
    st.markdown("<br><hr style='border-top: 1px solid #E2E8F0;'><p style='text-align:center; color:#94A3B8; font-size: 0.8rem;'>Secure Admin Environment</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    stats_page()