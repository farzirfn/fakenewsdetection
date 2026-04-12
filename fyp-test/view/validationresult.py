import streamlit as st
import mysql.connector
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ================================
# PAGE CONFIG
# ================================

# ================================
# DB CONNECTION
# ================================
def create_connection():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        port=st.secrets["mysql"]["port"],
        user=st.secrets["mysql"]["username"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"]
    )

# ================================
# LOAD VALIDATION RESULTS
# ================================
def load_pre_augmentation_validation():
    """Load pre-augmentation validation results from database"""
    conn = create_connection()
    query = """
        SELECT 
            id,
            balance_column,
            minority_classes,
            total_samples,
            validation_method,
            vector_magnitude,
            is_valid,
            interpretation,
            validated_at
        FROM pre_augmentation_validation
        ORDER BY validated_at DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_post_augmentation_validation():
    """Load post-augmentation validation results from database"""
    conn = create_connection()
    query = """
        SELECT 
            id,
            total_minority_records,
            total_augmented_generated,
            validation_method,
            mean_cosine_similarity,
            min_cosine_similarity,
            max_cosine_similarity,
            std_cosine_similarity,
            similarity_threshold,
            is_valid,
            created_at
        FROM augmentation_validation_results
        ORDER BY created_at DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_augmented_data_summary():
    """Load summary of augmented data"""
    conn = create_connection()
    query = """
        SELECT 
            subject,
            status,
            COUNT(*) as count,
            AVG(cosine_similarity) as avg_similarity,
            MIN(cosine_similarity) as min_similarity,
            MAX(cosine_similarity) as max_similarity
        FROM augmented_dataset
        GROUP BY subject, status
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ================================
# MAIN APP
# ================================
def main():
    st.title("📊 Data Augmentation Validation Results")
    st.markdown("View and analyze validation results from data augmentation process using **Cosine Similarity**")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Pre-Augmentation Validation", 
        "📈 Post-Augmentation Stats",
        "📊 Augmented Data Details",
        "📉 Comparison & Analytics"
    ])
    
    # ================================
    # TAB 1: Pre-Augmentation Validation
    # ================================
    with tab1:
        st.subheader("🔍 Pre-Augmentation Validation Results")
        st.markdown("**Purpose:** Validate minority class data quality BEFORE augmentation")
        
        try:
            df_pre = load_pre_augmentation_validation()
            
            if len(df_pre) == 0:
                st.info("📭 No validation results found. Run the augmentation process first.")
            else:
                # Summary metrics
                st.markdown("### 📊 Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Validations", len(df_pre))
                with col2:
                    valid_count = df_pre['is_valid'].sum()
                    st.metric("Valid Results", valid_count)
                with col3:
                    invalid_count = len(df_pre) - valid_count
                    st.metric("Warning Results", invalid_count)
                with col4:
                    avg_magnitude = df_pre['vector_magnitude'].mean()
                    st.metric("Avg Vector Magnitude", f"{avg_magnitude:.4f}")
                
                # Detailed table
                st.markdown("### 📋 Detailed Results")
                
                # Format the dataframe for display
                display_df = df_pre.copy()
                display_df['is_valid'] = display_df['is_valid'].apply(
                    lambda x: "✅ Valid" if x else "⚠️ Warning"
                )
                display_df['vector_magnitude'] = display_df['vector_magnitude'].apply(lambda x: f"{x:.4f}")
                display_df['validated_at'] = pd.to_datetime(display_df['validated_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(display_df, use_container_width=True)
                
                # Visualization: Vector magnitude distribution
                st.markdown("### 📈 Vector Magnitude Distribution")
                fig = px.histogram(
                    df_pre, 
                    x='vector_magnitude',
                    nbins=20,
                    title="Distribution of Vector Magnitudes",
                    labels={'vector_magnitude': 'Vector Magnitude', 'count': 'Frequency'},
                    color_discrete_sequence=['#667eea']
                )
                fig.add_vline(x=0.01, line_dash="dash", line_color="red", 
                             annotation_text="Threshold (0.01)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Visualization: Vector magnitude over time
                st.markdown("### 📉 Vector Magnitude Over Time")
                fig2 = px.line(
                    df_pre, 
                    x='validated_at', 
                    y='vector_magnitude',
                    title="Vector Magnitude Trend",
                    markers=True,
                    color_discrete_sequence=['#764ba2']
                )
                st.plotly_chart(fig2, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error loading pre-augmentation validation: {str(e)}")
            st.info("💡 Make sure you've run the augmentation process with the updated schema.")
    
    # ================================
    # TAB 2: Post-Augmentation Stats
    # ================================
    with tab2:
        st.subheader("📈 Post-Augmentation Statistics")
        st.markdown("**Purpose:** Statistics from the augmentation and validation process using Cosine Similarity")
        
        try:
            df_post = load_post_augmentation_validation()
            
            if len(df_post) == 0:
                st.info("📭 No augmentation results found. Complete the augmentation process first.")
            else:
                # Latest run summary
                latest = df_post.iloc[0]
                
                st.markdown("### 📊 Latest Augmentation Run")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Minority Records", latest['total_minority_records'])
                with col2:
                    st.metric("Augmented Generated", latest['total_augmented_generated'])
                with col3:
                    st.metric("Mean Cosine Sim", f"{latest['mean_cosine_similarity']:.4f}")
                with col4:
                    status = "✅ Valid" if latest['is_valid'] else "⚠️ Failed"
                    st.metric("Validation Status", status)
                
                # Detailed metrics
                st.markdown("### 📊 Similarity Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Min Similarity", f"{latest['min_cosine_similarity']:.4f}")
                with col2:
                    st.metric("Max Similarity", f"{latest['max_cosine_similarity']:.4f}")
                with col3:
                    st.metric("Std Deviation", f"{latest['std_cosine_similarity']:.4f}")
                with col4:
                    st.metric("Threshold", f"{latest['similarity_threshold']:.2f}")
                
                # Detailed table
                st.markdown("### 📋 All Augmentation Runs")
                display_post = df_post.copy()
                display_post['is_valid'] = display_post['is_valid'].apply(
                    lambda x: "✅ Valid" if x else "⚠️ Failed"
                )
                display_post['mean_cosine_similarity'] = display_post['mean_cosine_similarity'].apply(lambda x: f"{x:.4f}")
                display_post['min_cosine_similarity'] = display_post['min_cosine_similarity'].apply(lambda x: f"{x:.4f}")
                display_post['max_cosine_similarity'] = display_post['max_cosine_similarity'].apply(lambda x: f"{x:.4f}")
                display_post['std_cosine_similarity'] = display_post['std_cosine_similarity'].apply(lambda x: f"{x:.4f}")
                display_post['created_at'] = pd.to_datetime(display_post['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(display_post, use_container_width=True)
                
                # Visualization: Cosine similarity trend over time
                st.markdown("### 📈 Mean Cosine Similarity Trend")
                fig = px.line(
                    df_post, 
                    x='created_at', 
                    y='mean_cosine_similarity',
                    title="Mean Cosine Similarity Over Time",
                    markers=True,
                    color_discrete_sequence=['#667eea']
                )
                fig.add_hline(y=0.85, line_dash="dash", line_color="green", 
                             annotation_text="Threshold (0.85)")
                fig.update_yaxes(title_text="Mean Cosine Similarity")
                st.plotly_chart(fig, use_container_width=True)
                
                # Box plot: Similarity range
                st.markdown("### 📊 Similarity Range Distribution")
                similarity_data = pd.DataFrame({
                    'Metric': ['Min', 'Mean', 'Max'] * len(df_post),
                    'Similarity': (
                        list(df_post['min_cosine_similarity']) + 
                        list(df_post['mean_cosine_similarity']) + 
                        list(df_post['max_cosine_similarity'])
                    ),
                    'Run': list(range(len(df_post))) * 3
                })
                fig_box = px.box(
                    similarity_data,
                    x='Metric',
                    y='Similarity',
                    title="Cosine Similarity Distribution Across Runs",
                    color='Metric',
                    color_discrete_sequence=['#f5576c', '#667eea', '#4facfe']
                )
                fig_box.add_hline(y=0.85, line_dash="dash", line_color="green", 
                                 annotation_text="Threshold")
                st.plotly_chart(fig_box, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error loading post-augmentation stats: {str(e)}")
            st.info("💡 Make sure you've run the augmentation process with the updated schema.")
    
    # ================================
    # TAB 3: Augmented Data Details
    # ================================
    with tab3:
        st.subheader("📊 Augmented Data Details")
        st.markdown("**Purpose:** Detailed breakdown of augmented data by class")
        
        try:
            df_summary = load_augmented_data_summary()
            
            if len(df_summary) == 0:
                st.info("📭 No augmented data found. Complete the augmentation process first.")
            else:
                # Total augmented records
                total_augmented = df_summary['count'].sum()
                avg_similarity = df_summary['avg_similarity'].mean()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Augmented Records", total_augmented)
                with col2:
                    st.metric("Overall Avg Similarity", f"{avg_similarity:.4f}")
                with col3:
                    classes = df_summary['subject'].nunique() + df_summary['status'].nunique()
                    st.metric("Classes Augmented", classes)
                
                # Detailed table
                st.markdown("### 📋 Breakdown by Class")
                display_summary = df_summary.copy()
                display_summary['avg_similarity'] = display_summary['avg_similarity'].apply(lambda x: f"{x:.4f}")
                display_summary['min_similarity'] = display_summary['min_similarity'].apply(lambda x: f"{x:.4f}")
                display_summary['max_similarity'] = display_summary['max_similarity'].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(display_summary, use_container_width=True)
                
                # Visualization: Count by class
                st.markdown("### 📊 Augmented Records by Subject")
                fig1 = px.bar(
                    df_summary,
                    x='subject',
                    y='count',
                    title="Number of Augmented Records per Subject",
                    color='subject',
                    labels={'count': 'Number of Records', 'subject': 'Subject'}
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Visualization: Average similarity by class
                st.markdown("### 📈 Average Similarity by Subject")
                fig2 = px.bar(
                    df_summary,
                    x='subject',
                    y='avg_similarity',
                    title="Average Cosine Similarity per Subject",
                    color='avg_similarity',
                    color_continuous_scale='Viridis',
                    labels={'avg_similarity': 'Average Similarity', 'subject': 'Subject'}
                )
                fig2.add_hline(y=0.85, line_dash="dash", line_color="red", 
                              annotation_text="Threshold (0.85)")
                st.plotly_chart(fig2, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error loading augmented data: {str(e)}")
            st.info("💡 Make sure you've saved augmented data to the database.")
    
    # ================================
    # TAB 4: Comparison & Analytics
    # ================================
    with tab4:
        st.subheader("📉 Comparison & Analytics")
        
        try:
            df_pre = load_pre_augmentation_validation()
            df_post = load_post_augmentation_validation()
            
            if len(df_pre) == 0 or len(df_post) == 0:
                st.info("📭 Need both pre and post augmentation data for comparison.")
            else:
                st.markdown("### 🔄 Pre vs Post Augmentation Metrics")
                
                # Compare average metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Pre-Augmentation (Minority Quality Check)")
                    pre_metrics = pd.DataFrame({
                        'Metric': ['Avg Vector Magnitude', 'Validation Success Rate', 'Total Samples'],
                        'Value': [
                            f"{df_pre['vector_magnitude'].mean():.4f}",
                            f"{(df_pre['is_valid'].sum() / len(df_pre) * 100):.1f}%",
                            f"{df_pre['total_samples'].sum()}"
                        ]
                    })
                    st.dataframe(pre_metrics, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("#### Post-Augmentation (Augmented vs Minority)")
                    latest_post = df_post.iloc[0]
                    post_metrics = pd.DataFrame({
                        'Metric': ['Mean Cosine Similarity', 'Validation Status', 'Augmented Generated'],
                        'Value': [
                            f"{latest_post['mean_cosine_similarity']:.4f}",
                            "✅ Valid" if latest_post['is_valid'] else "⚠️ Failed",
                            f"{latest_post['total_augmented_generated']}"
                        ]
                    })
                    st.dataframe(post_metrics, use_container_width=True, hide_index=True)
                
                # Success rate comparison
                st.markdown("### 📊 Success Rate Comparison")
                pre_success_rate = (df_pre['is_valid'].sum() / len(df_pre)) * 100
                post_success_rate = (df_post['is_valid'].sum() / len(df_post)) * 100
                
                comparison_data = pd.DataFrame({
                    'Stage': ['Pre-Augmentation', 'Post-Augmentation'],
                    'Success Rate (%)': [pre_success_rate, post_success_rate]
                })
                
                fig = px.bar(
                    comparison_data,
                    x='Stage',
                    y='Success Rate (%)',
                    title="Validation Success Rate: Pre vs Post",
                    color='Stage',
                    color_discrete_sequence=['#667eea', '#764ba2'],
                    text='Success Rate (%)'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
                # Quality metrics comparison
                st.markdown("### 📈 Quality Metrics Over Time")
                
                # Create combined timeline
                timeline_data = []
                for _, row in df_pre.iterrows():
                    timeline_data.append({
                        'Date': row['validated_at'],
                        'Metric': 'Vector Magnitude',
                        'Value': row['vector_magnitude'],
                        'Stage': 'Pre-Augmentation'
                    })
                
                for _, row in df_post.iterrows():
                    timeline_data.append({
                        'Date': row['created_at'],
                        'Metric': 'Cosine Similarity',
                        'Value': row['mean_cosine_similarity'],
                        'Stage': 'Post-Augmentation'
                    })
                
                timeline_df = pd.DataFrame(timeline_data)
                
                fig_timeline = px.scatter(
                    timeline_df,
                    x='Date',
                    y='Value',
                    color='Stage',
                    symbol='Metric',
                    title="Quality Metrics Timeline",
                    color_discrete_sequence=['#667eea', '#764ba2']
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Insights
                st.markdown("### 💡 Insights & Recommendations")
                
                if pre_success_rate >= 80:
                    st.success("✅ Pre-augmentation validation shows excellent minority data quality (≥80% pass rate)")
                elif pre_success_rate >= 60:
                    st.info("ℹ️ Pre-augmentation validation shows acceptable minority data quality (60-80% pass rate)")
                else:
                    st.warning("⚠️ Pre-augmentation validation shows potential quality issues in minority data (<60% pass rate)")
                
                if post_success_rate >= 80:
                    st.success("✅ High augmentation validation rate indicates excellent quality augmented data")
                elif post_success_rate >= 60:
                    st.info("ℹ️ Augmentation validation rate is acceptable - augmented data is usable")
                else:
                    st.warning("⚠️ Lower augmentation validation rate - consider adjusting augmentation parameters")
                
                # Recommendations
                st.markdown("#### 📝 Recommendations:")
                recommendations = []
                
                if latest_post['mean_cosine_similarity'] < 0.85:
                    recommendations.append("- Consider adjusting augmentation parameters to increase similarity")
                
                if latest_post['std_cosine_similarity'] > 0.1:
                    recommendations.append("- High standard deviation suggests inconsistent augmentation quality")
                
                if df_pre['vector_magnitude'].mean() < 0.05:
                    recommendations.append("- Low vector magnitude suggests minority data might need more text content")
                
                if recommendations:
                    for rec in recommendations:
                        st.markdown(rec)
                else:
                    st.success("✅ All metrics look good! No immediate action needed.")
                
        except Exception as e:
            st.error(f"❌ Error in comparison: {str(e)}")
            st.info("💡 Make sure you have both pre and post augmentation data.")

# ================================
# RUN APP
# ================================
if __name__ == "__main__":
    main()