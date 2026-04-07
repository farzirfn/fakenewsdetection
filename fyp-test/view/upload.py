import streamlit as st
import pandas as pd
import mysql.connector
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, Dict
import hashlib

# ================================
# NLTK SETUP
# ================================
@st.cache_resource
def setup_nltk():
    """Download NLTK data if not already present"""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download("stopwords", quiet=True)
    
    return set(stopwords.words("english")), PorterStemmer()

stop_words, stemmer = setup_nltk()

# ================================
# DATABASE CONNECTION
# ================================
def create_connection():
    """Create database connection with error handling"""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="fyp",
            autocommit=False
        )
        return conn
    except mysql.connector.Error as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        return None

# ================================
# DATA VALIDATION
# ================================
def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate dataframe has required columns
    Returns: (is_valid, error_message)
    """
    required_columns = ['title', 'text', 'subject', 'status']
    
    if df.empty:
        return False, "❌ DataFrame is empty"
    
    df.columns = df.columns.str.strip().str.lower()
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        return False, f"❌ Missing required columns: {', '.join(missing_cols)}"
    
    return True, "✅ Validation passed"

def validate_column_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate and clean column data
    Returns: (cleaned_df, stats_dict)
    """
    stats = {
        'empty_titles': 0,
        'empty_texts': 0,
        'invalid_status': 0
    }
    
    stats['empty_titles'] = df['title'].isna().sum()
    stats['empty_texts'] = df['text'].isna().sum()
    
    valid_statuses = ['real', 'fake', 'true', 'false', '0', '1', 'Real', 'Fake', 'True', 'False']
    stats['invalid_status'] = (~df['status'].astype(str).isin(valid_statuses)).sum()
    
    return df, stats

# ================================
# DATA PREPROCESSING
# ================================
def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove rows with missing values and clean column names"""
    original_count = len(df)
    
    df.columns = df.columns.str.strip().str.lower()
    
    df = df.dropna(subset=['title', 'text', 'status'], how='any')
    
    df['subject'] = df['subject'].fillna('Unknown')
    
    df = df.drop_duplicates(subset=['title', 'text'], keep='first')
    
    removed_count = original_count - len(df)
    
    return df, removed_count

def clean_and_stem(text: str) -> str:
    """Clean text: lowercase, remove punctuation, remove stopwords, stem"""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    
    text = text.lower()
    
    text = re.sub(r"[^a-z\s]", "", text)
    
    tokens = text.split()
    
    tokens = [t for t in tokens if t and t not in stop_words]
    
    try:
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
    except Exception:
        stemmed_tokens = tokens
    
    result = " ".join(stemmed_tokens)
    
    return result if result.strip() else text[:100]

def generate_content_hash(title: str, text: str) -> str:
    """Generate hash for duplicate detection"""
    content = f"{title}|{text}".lower().strip()
    return hashlib.md5(content.encode()).hexdigest()

# ================================
# DATABASE OPERATIONS
# ================================
def get_database_stats() -> Dict:
    """Get current database statistics"""
    try:
        conn = create_connection()
        if not conn:
            return {'total': 0, 'by_status': []}
        
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT COUNT(*) as total FROM dataset")
        total = cursor.fetchone()['total']
        
        cursor.execute("SELECT status, COUNT(*) as count FROM dataset GROUP BY status")
        by_status = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {'total': total, 'by_status': by_status}
    except Exception as e:
        st.warning(f"⚠️ Could not fetch database stats: {str(e)}")
        return {'total': 0, 'by_status': []}

def check_existing_records(conn, df: pd.DataFrame) -> int:
    """
    Check how many records already exist in database
    Returns: count of duplicates
    """
    try:
        cursor = conn.cursor()
        
        sample_size = min(len(df), 100)
        duplicates = 0
        
        for _, row in df.head(sample_size).iterrows():
            content_hash = generate_content_hash(
                str(row.get('title', '')), 
                str(row.get('text', ''))
            )
            
            cursor.execute(
                "SELECT COUNT(*) FROM dataset WHERE MD5(CONCAT(title, '|', text)) = %s",
                (content_hash,)
            )
            if cursor.fetchone()[0] > 0:
                duplicates += 1
        
        cursor.close()
        
        if sample_size < len(df):
            duplicates = int(duplicates * (len(df) / sample_size))
        
        return duplicates
        
    except Exception as e:
        st.warning(f"⚠️ Could not check duplicates: {str(e)}")
        return 0

def save_to_database(df: pd.DataFrame) -> Tuple[bool, str, Dict]:
    """
    Save dataframe to database with transaction support
    Returns: (success, message, stats)
    """
    conn = create_connection()
    if not conn:
        return False, "❌ Database connection failed", {}
    
    try:
        cursor = conn.cursor()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        stats = {
            'attempted': len(df),
            'inserted': 0,
            'duplicates': 0,
            'errors': 0
        }
        
        status_text.text("🔄 Checking for duplicates...")
        
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            try:
                title_clean = clean_and_stem(str(row.get("title", "Untitled")))
                text_clean = clean_and_stem(str(row.get("text", "")))
                subject_clean = clean_and_stem(str(row.get("subject", "Unknown")))
                status_val = str(row.get("status", "Pending")).strip()
                
                if not title_clean or len(title_clean) < 3:
                    stats['errors'] += 1
                    continue
                
                if not text_clean or len(text_clean) < 10:
                    stats['errors'] += 1
                    continue
                
                content_hash = generate_content_hash(
                    title_clean,
                    text_clean
                )
                
                cursor.execute(
                    "SELECT COUNT(*) FROM dataset WHERE MD5(CONCAT(title, '|', text)) = %s",
                    (content_hash,)
                )
                
                if cursor.fetchone()[0] > 0:
                    stats['duplicates'] += 1
                else:
                    cursor.execute(
                        """INSERT INTO dataset (title, text, subject, status) 
                           VALUES (%s, %s, %s, %s)""",
                        (title_clean, text_clean, subject_clean, status_val)
                    )
                    stats['inserted'] += 1
                
            except Exception as e:
                stats['errors'] += 1
                st.warning(f"⚠️ Error on row {i}: {str(e)}")
            
            progress = i / len(df)
            progress_bar.progress(progress)
            
            if i % 10 == 0:
                status_text.text(
                    f"🔄 Processing... {i}/{len(df)} rows "
                    f"(Inserted: {stats['inserted']}, Duplicates: {stats['duplicates']}, Errors: {stats['errors']})"
                )
        
        conn.commit()
        
        cursor.execute("SELECT COUNT(*) FROM dataset")
        final_count = cursor.fetchone()[0]
        stats['total_in_db'] = final_count
        
        cursor.close()
        conn.close()
        
        progress_bar.empty()
        status_text.empty()
        
        if stats['inserted'] > 0:
            return True, "✅ Upload completed successfully", stats
        else:
            return False, "❌ No new records inserted", stats
        
    except mysql.connector.Error as e:
        conn.rollback()
        conn.close()
        return False, f"❌ Database error: {str(e)}", {}
    except Exception as e:
        conn.rollback()
        conn.close()
        return False, f"❌ Unexpected error: {str(e)}", {}

# ================================
# MAIN UPLOAD PAGE
# ================================
def upload_page():
    # Header
    st.title("📤 Upload Dataset")
    st.write("Upload and preprocess your dataset with automatic cleaning")
    
    # Current database stats
    db_stats = get_database_stats()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        **📋 Upload Instructions**
        
        ✅ Supported formats: CSV, XLS, XLSX
                
        ✅ Required columns: `title`, `text`, `subject`, `status`
                
        ✅ Automatic preprocessing: lowercase, remove punctuation, remove stopwords, stemming
                
        ✅ Rows with missing values will be automatically removed
                
        ✅ Duplicate detection based on title and text
        """)
    
    with col2:
        st.metric("Current Database", f"{db_stats['total']:,}", "📊 Total Records")
    
    # File uploader
    st.subheader("📁 Select File to Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV, XLS, or XLSX file",
        type=["csv", "xls", "xlsx"],
        help="Upload your dataset file. Supported formats: CSV, Excel (XLS, XLSX)"
    )
    
    if uploaded_file is not None:
        try:
            # Load file based on extension
            with st.spinner("📂 Loading file..."):
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
                elif uploaded_file.name.endswith((".xls", ".xlsx")):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("❌ Unsupported file type.")
                    return
            
            st.success(f"✅ File loaded: **{uploaded_file.name}**")
            
            # Validate dataframe structure
            is_valid, validation_msg = validate_dataframe(df)
            
            if not is_valid:
                st.error(f"{validation_msg}\n\n**Your columns:** {', '.join(df.columns.tolist())}\n\n**Required columns:** title, text, subject, status")
                return
            
            # Validate column data
            df_validated, validation_stats = validate_column_data(df)
            
            # Show validation warnings
            if any(validation_stats.values()):
                warning_messages = []
                if validation_stats['empty_titles'] > 0:
                    warning_messages.append(f"{validation_stats['empty_titles']} empty titles")
                if validation_stats['empty_texts'] > 0:
                    warning_messages.append(f"{validation_stats['empty_texts']} empty texts")
                if validation_stats['invalid_status'] > 0:
                    warning_messages.append(f"{validation_stats['invalid_status']} invalid status values")
                
                if warning_messages:
                    st.warning(f"⚠️ Data quality issues detected: {', '.join(warning_messages)}\n\nThese rows will be removed during preprocessing.")
            
            # Show stats BEFORE preprocessing
            st.subheader("📊 Dataset Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Before Preprocessing", f"{df.shape[0]:,}", "Rows")
            
            # Preprocess data
            df_clean, removed_count = preprocess_data(df)
            
            with col2:
                st.metric("After Preprocessing", f"{df_clean.shape[0]:,}", "Rows")
            
            with col3:
                removed_pct = (removed_count / df.shape[0] * 100) if df.shape[0] > 0 else 0
                st.metric("Removed", f"{removed_count:,}", f"{removed_pct:.1f}%")
            
            # Check if data remains after cleaning
            if len(df_clean) == 0:
                st.error("❌ **No valid data remaining after preprocessing!**\n\nAll rows were removed due to missing or invalid data. Please check your dataset and try again.")
                return
            
            # Show warning if too many rows were removed
            if removed_count > 0:
                st.warning(f"⚠️ **{removed_count:,} rows** ({removed_pct:.1f}%) were removed due to:\n- Missing values in title, text, or status\n- Duplicate entries")
            
            # Preprocessing preview
            st.subheader("🔍 Preprocessing Preview")
            st.write("See how your data will be cleaned before saving to database:")
            
            # Create preview with before/after
            preview_size = min(5, len(df_clean))
            preview_df = df_clean.head(preview_size).copy()
            preview_df["title_clean"] = preview_df["title"].apply(clean_and_stem)
            preview_df["text_clean"] = preview_df["text"].apply(clean_and_stem)
            preview_df["subject_clean"] = preview_df["subject"].apply(clean_and_stem)
            
            # Show in tabs
            tab1, tab2, tab3 = st.tabs(["📝 Original Data", "✨ Cleaned Data", "🔄 Comparison"])
            
            with tab1:
                st.dataframe(
                    df_clean[["title", "text", "subject", "status"]].head(10),
                    use_container_width=True
                )
            
            with tab2:
                cleaned_preview = pd.DataFrame({
                    "title": preview_df["title_clean"],
                    "text": preview_df["text_clean"],
                    "subject": preview_df["subject_clean"],
                    "status": preview_df["status"]
                })
                st.dataframe(cleaned_preview, use_container_width=True)
            
            with tab3:
                if len(preview_df) > 0:
                    comparison = pd.DataFrame({
                        "Column": ["Title", "Text", "Subject"],
                        "Original": [
                            str(preview_df["title"].iloc[0])[:50] + "...",
                            str(preview_df["text"].iloc[0])[:50] + "...",
                            str(preview_df["subject"].iloc[0])[:50] + "..."
                        ],
                        "Cleaned": [
                            str(preview_df["title_clean"].iloc[0])[:50] + "...",
                            str(preview_df["text_clean"].iloc[0])[:50] + "...",
                            str(preview_df["subject_clean"].iloc[0])[:50] + "..."
                        ]
                    })
                    st.dataframe(comparison, use_container_width=True, hide_index=True)
            
            # Data distribution visualization
            st.subheader("📊 Data Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution by status
                status_dist = df_clean['status'].value_counts()
                fig_status = px.pie(
                    values=status_dist.values,
                    names=status_dist.index,
                    title='Distribution by Status',
                    hole=0.4
                )
                fig_status.update_traces(textposition='inside', textinfo='percent+label')
                fig_status.update_layout(height=300)
                st.plotly_chart(fig_status, use_container_width=True)
            
            with col2:
                # Distribution by subject
                subject_dist = df_clean['subject'].value_counts().head(10)
                fig_subject = px.bar(
                    x=subject_dist.index,
                    y=subject_dist.values,
                    title='Top 10 Subjects',
                    labels={'x': 'Subject', 'y': 'Count'}
                )
                fig_subject.update_layout(height=300)
                fig_subject.update_xaxes(tickangle=45)
                st.plotly_chart(fig_subject, use_container_width=True)
            
            # Save to database button
            st.subheader("💾 Save to Database")
            
            if st.button("🚀 Save to Database", use_container_width=True):
                success, message, stats = save_to_database(df_clean)
                
                if success:
                    st.success(f"""
                    **✅ Upload Successful!**
                    
                    **{stats['inserted']:,}** new records have been processed and saved to the database.
                    
                    **📊 Upload Summary:**
                    - Total attempted: {stats['attempted']:,}
                    - Successfully inserted: {stats['inserted']:,}
                    - Duplicates skipped: {stats['duplicates']:,}
                    - Errors: {stats['errors']:,}
                    - Total records in database: {stats['total_in_db']:,}
                    """)
                    
                    st.balloons()
                else:
                    if stats:
                        st.warning(f"""
                        **⚠️ Upload Completed with Warnings**
                        
                        {message}
                        
                        **📊 Upload Summary:**
                        - Total attempted: {stats.get('attempted', 0):,}
                        - Successfully inserted: {stats.get('inserted', 0):,}
                        - Duplicates skipped: {stats.get('duplicates', 0):,}
                        - Errors: {stats.get('errors', 0):,}
                        """)
                    else:
                        st.error(message)
        
        except pd.errors.EmptyDataError:
            st.error("❌ The file is empty or corrupted.")
        except pd.errors.ParserError:
            st.error("❌ Error parsing file. Please check the file format.")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.exception(e)
    
    else:
        st.info("""
        **📁 No File Selected**
        
        Click "Browse files" above to upload your dataset
        
        - Supported formats: CSV, XLS, XLSX
        - Required columns: title, text, subject, status
        """)

# ================================
# RUN PAGE
# ================================
if __name__ == "__main__":
    upload_page()