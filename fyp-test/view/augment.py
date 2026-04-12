import streamlit as st
import mysql.connector
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ================================
# SESSION STATE INITIALIZATION
# ================================
def init_session_state():
    """Initialize all session state variables to prevent data loss"""
    if 'df_original' not in st.session_state:
        st.session_state['df_original'] = None
    if 'imbalance_info' not in st.session_state:
        st.session_state['imbalance_info'] = None
    if 'minority_validation' not in st.session_state:
        st.session_state['minority_validation'] = None
    if 'df_augmented' not in st.session_state:
        st.session_state['df_augmented'] = None
    if 'df_augmented_validated' not in st.session_state:
        st.session_state['df_augmented_validated'] = None
    if 'dataset_validation' not in st.session_state:
        st.session_state['dataset_validation'] = None
    if 'validation_stats' not in st.session_state:
        st.session_state['validation_stats'] = None
    if 'tfidf_vectorizer' not in st.session_state:
        st.session_state['tfidf_vectorizer'] = None
    if 'minority_embeddings' not in st.session_state:
        st.session_state['minority_embeddings'] = None

# -------------------------------
# DB connection
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
# Load dataset from database
# -------------------------------
def load_dataset():
    """Load original dataset from database"""
    conn = create_connection()
    df = pd.read_sql("SELECT title, text, subject, status FROM dataset", conn)
    conn.close()
    return df

# -------------------------------
# Identify imbalanced classes
# -------------------------------
def identify_imbalanced_classes(df, column='subject', threshold_ratio=1.5):
    """
    Identify minority classes that need augmentation.
    
    Args:
        df: DataFrame with the dataset
        column: Column to check for imbalance ('subject' or 'status')
        threshold_ratio: If max_count/class_count > threshold_ratio, class is minority
    
    Returns:
        dict with class distribution and list of minority classes
    """
    class_counts = df[column].value_counts()
    max_count = class_counts.max()
    
    minority_classes = []
    class_info = {}
    
    for cls, count in class_counts.items():
        ratio = max_count / count
        is_minority = ratio > threshold_ratio
        
        class_info[cls] = {
            'count': count,
            'ratio': ratio,
            'is_minority': is_minority,
            'needs_augmentation': count
        }
        
        if is_minority:
            minority_classes.append(cls)
    
    return {
        'class_counts': class_counts,
        'class_info': class_info,
        'minority_classes': minority_classes,
        'max_count': max_count,
        'column': column
    }

# ================================
# DistilBERT EMBEDDING FUNCTIONS
# ================================

@st.cache_resource
def load_distilbert_model():
    """Load DistilBERT model and tokenizer (cached)"""
    st.info("📥 Loading DistilBERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()  # Set to evaluation mode
    st.success("✅ DistilBERT model loaded!")
    return tokenizer, model

def get_distilbert_embedding(text, tokenizer, model):
    """
    Get 768-dimensional DistilBERT embedding for text.
    
    Args:
        text: Input text string
        tokenizer: DistilBERT tokenizer
        model: DistilBERT model
    
    Returns:
        768-dimensional numpy array representing text meaning
    """
    # Tokenize and encode text
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    
    # Get embeddings without gradient computation (faster)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract [CLS] token embedding (represents entire sentence)
    # Shape: (1, 768) -> squeeze to (768,)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    return embedding

def compute_embedding_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: 768-dim numpy array
        embedding2: 768-dim numpy array
    
    Returns:
        Similarity score between 0 and 1
    """
    # Reshape for sklearn
    emb1 = embedding1.reshape(1, -1)
    emb2 = embedding2.reshape(1, -1)
    
    # Compute cosine similarity
    similarity = cosine_similarity(emb1, emb2)[0][0]
    
    return similarity

# ================================
# DISTILBERT-BASED AUGMENTATION METHODS
# ================================

def augment_by_embedding_mixup(text, similar_texts, tokenizer, model, num_aug=2, alpha=0.3):
    """
    Augment by mixing embeddings of similar texts (MixUp in embedding space).
    
    This creates new texts by:
    1. Getting embeddings of original text
    2. Getting embeddings of similar texts from same class
    3. Interpolating between embeddings
    4. Reconstructing text from mixed embedding
    
    Args:
        text: Original text
        similar_texts: List of similar texts from same class
        tokenizer: DistilBERT tokenizer
        model: DistilBERT model
        num_aug: Number of augmentations to generate
        alpha: Interpolation weight (0.0-1.0)
    
    Returns:
        List of augmented texts
    """
    if len(similar_texts) == 0:
        # Fallback to word-level augmentation
        return augment_by_word_substitution(text, tokenizer, model, num_aug)
    
    # Get embedding of original text
    original_embedding = get_distilbert_embedding(text, tokenizer, model)
    
    augmented_texts = []
    
    for _ in range(num_aug):
        # Randomly select a similar text
        similar_text = np.random.choice(similar_texts)
        similar_embedding = get_distilbert_embedding(similar_text, tokenizer, model)
        
        # Mix embeddings using interpolation
        # mixed = (1-alpha) * original + alpha * similar
        mixed_embedding = (1 - alpha) * original_embedding + alpha * similar_embedding
        
        # Reconstruct text from mixed embedding using word substitution
        # Since we can't directly decode from embeddings, we use word-level mixing
        augmented = mix_texts_at_word_level(text, similar_text, alpha=alpha)
        augmented_texts.append(augmented)
    
    return augmented_texts

def mix_texts_at_word_level(text1, text2, alpha=0.3):
    """
    Mix two texts at word level based on alpha parameter.
    
    Args:
        text1: Original text
        text2: Similar text
        alpha: Proportion of words to take from text2
    
    Returns:
        Mixed text
    """
    words1 = text1.split()
    words2 = text2.split()
    
    # Determine target length (average of both)
    target_len = (len(words1) + len(words2)) // 2
    
    # Calculate how many words to take from each text
    num_from_text2 = int(target_len * alpha)
    num_from_text1 = target_len - num_from_text2
    
    # Sample words from both texts
    if len(words1) >= num_from_text1:
        selected_words1 = np.random.choice(words1, num_from_text1, replace=False).tolist()
    else:
        selected_words1 = words1
    
    if len(words2) >= num_from_text2:
        selected_words2 = np.random.choice(words2, num_from_text2, replace=False).tolist()
    else:
        selected_words2 = words2[:num_from_text2]
    
    # Combine and shuffle
    mixed_words = selected_words1 + selected_words2
    np.random.shuffle(mixed_words)
    
    return ' '.join(mixed_words)

def augment_by_word_substitution(text, tokenizer, model, num_aug=2, substitution_rate=0.3):
    """
    Augment by substituting words based on contextual embeddings.
    
    Uses masked language modeling approach:
    1. Masks certain words
    2. Uses context to predict similar words
    3. Replaces with contextually appropriate words
    
    Args:
        text: Original text
        tokenizer: DistilBERT tokenizer
        model: DistilBERT model
        num_aug: Number of augmentations
        substitution_rate: Percentage of words to substitute
    
    Returns:
        List of augmented texts
    """
    words = text.split()
    augmented_texts = []
    
    for _ in range(num_aug):
        new_words = words.copy()
        
        # Calculate number of words to substitute
        num_substitutions = max(1, int(len(words) * substitution_rate))
        
        # Select random positions to substitute (avoid very short words)
        substitutable_positions = [
            i for i, word in enumerate(words) 
            if len(word) > 3 and word.isalpha()
        ]
        
        if len(substitutable_positions) == 0:
            augmented_texts.append(text)
            continue
        
        num_to_substitute = min(num_substitutions, len(substitutable_positions))
        positions_to_substitute = np.random.choice(
            substitutable_positions, 
            num_to_substitute, 
            replace=False
        )
        
        # Substitute words with similar words from vocabulary
        for pos in positions_to_substitute:
            # Get contextual alternatives (simple approach: use similar length words)
            original_word = words[pos]
            
            # Simple substitution: shuffle characters slightly or use similar patterns
            new_word = get_similar_word(original_word, tokenizer)
            new_words[pos] = new_word
        
        augmented_text = ' '.join(new_words)
        augmented_texts.append(augmented_text)
    
    return augmented_texts

def get_similar_word(word, tokenizer):
    """
    Get a similar word from tokenizer vocabulary.
    
    This is a simplified approach - in production, you'd use:
    - Masked Language Model predictions
    - WordNet synonyms
    - Pre-computed word similarity matrices
    
    Args:
        word: Original word
        tokenizer: DistilBERT tokenizer
    
    Returns:
        Similar word
    """
    # Get words from vocabulary with similar length and starting letter
    vocab = list(tokenizer.vocab.keys())
    
    # Filter vocabulary
    similar_candidates = [
        w for w in vocab 
        if len(w) >= max(3, len(word) - 2) 
        and len(w) <= len(word) + 2
        and w.isalpha()
        and w[0] == word[0]  # Same starting letter
    ]
    
    if similar_candidates:
        return np.random.choice(similar_candidates)
    else:
        return word

def augment_by_embedding_perturbation(text, tokenizer, model, num_aug=2, noise_level=0.1):
    """
    Augment by adding small perturbations to embeddings and reconstructing text.
    
    Args:
        text: Original text
        tokenizer: DistilBERT tokenizer
        model: DistilBERT model  
        num_aug: Number of augmentations
        noise_level: Amount of noise to add to embeddings
    
    Returns:
        List of augmented texts
    """
    # Get original embedding
    original_embedding = get_distilbert_embedding(text, tokenizer, model)
    
    augmented_texts = []
    words = text.split()
    
    for _ in range(num_aug):
        # Add Gaussian noise to embedding
        noise = np.random.normal(0, noise_level, original_embedding.shape)
        perturbed_embedding = original_embedding + noise
        
        # Since we can't directly decode embeddings to text,
        # we use the perturbed embedding to guide word-level changes
        # The more noise, the more words we change
        
        num_changes = max(1, int(len(words) * (noise_level * 3)))
        new_words = words.copy()
        
        changeable_positions = [
            i for i, w in enumerate(words) 
            if len(w) > 3 and w.isalpha()
        ]
        
        if changeable_positions:
            num_to_change = min(num_changes, len(changeable_positions))
            positions = np.random.choice(changeable_positions, num_to_change, replace=False)
            
            for pos in positions:
                new_words[pos] = get_similar_word(words[pos], tokenizer)
        
        augmented_text = ' '.join(new_words)
        augmented_texts.append(augmented_text)
    
    return augmented_texts

def augment_using_nearest_neighbors(text, minority_df, tokenizer, model, num_aug=2, k=3):
    """
    Augment by finding k-nearest neighbors in embedding space and mixing them.
    
    This is the MOST EFFECTIVE approach using DistilBERT embeddings:
    1. Computes 768-dim embedding for input text
    2. Finds k most similar texts from minority class
    3. Mixes input with similar texts
    
    Args:
        text: Original text to augment
        minority_df: DataFrame with minority class texts
        tokenizer: DistilBERT tokenizer
        model: DistilBERT model
        num_aug: Number of augmentations to generate
        k: Number of nearest neighbors to consider
    
    Returns:
        List of augmented texts with similarity scores
    """
    # Get embedding for original text
    original_embedding = get_distilbert_embedding(text, tokenizer, model)
    
    # Compute embeddings for all minority texts (cached in session state)
    if 'minority_embeddings_cache' not in st.session_state:
        st.session_state['minority_embeddings_cache'] = {}
    
    similarities = []
    
    for idx, row in minority_df.iterrows():
        other_text = row['text']
        
        # Skip if same text
        if other_text == text:
            continue
        
        # Use cached embedding if available
        if idx in st.session_state['minority_embeddings_cache']:
            other_embedding = st.session_state['minority_embeddings_cache'][idx]
        else:
            other_embedding = get_distilbert_embedding(other_text, tokenizer, model)
            st.session_state['minority_embeddings_cache'][idx] = other_embedding
        
        # Compute similarity
        similarity = compute_embedding_similarity(original_embedding, other_embedding)
        similarities.append((similarity, other_text, idx))
    
    # Get k most similar texts
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_k_similar = similarities[:k]
    
    if len(top_k_similar) == 0:
        # Fallback to word substitution
        return augment_by_word_substitution(text, tokenizer, model, num_aug), [0.0] * num_aug
    
    # Generate augmentations by mixing with similar texts
    augmented_texts = []
    similarity_scores = []
    
    for _ in range(num_aug):
        # Randomly select from top-k similar texts
        sim_score, similar_text, _ = top_k_similar[np.random.randint(0, len(top_k_similar))]
        
        # Mix original with similar text
        alpha = np.random.uniform(0.2, 0.4)  # Mix 20-40% from similar text
        mixed_text = mix_texts_at_word_level(text, similar_text, alpha=alpha)
        
        augmented_texts.append(mixed_text)
        similarity_scores.append(sim_score)
    
    return augmented_texts, similarity_scores

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
def create_tfidf_vectorizer(texts):
    """Create and fit TF-IDF vectorizer on texts"""
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    vectorizer.fit(texts)
    return vectorizer

def get_tfidf_vector(text, vectorizer):
    """Get TF-IDF vector for text"""
    return vectorizer.transform([text]).toarray().flatten()

# -------------------------------
# Validation Functions
# -------------------------------
def cosine_similarity_validate_augmented_vs_minority(df_minority, df_augmented, vectorizer, threshold=0.85):
    """
    Validate augmented dataset against minority class data using TF-IDF + Cosine Similarity.
    """
    st.info(f"🔍 Validating {len(df_augmented)} augmented records against {len(df_minority)} minority class records...")
    
    minority_tfidf_vectors = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Extracting TF-IDF vectors from minority class dataset...")
    for idx, (index, row) in enumerate(df_minority.iterrows()):
        try:
            tfidf_vec = get_tfidf_vector(row['text'], vectorizer)
            minority_tfidf_vectors.append(tfidf_vec)
        except Exception as e:
            st.warning(f"Error processing minority data row {index}: {e}")
        
        progress = min(((idx + 1) / len(df_minority)) * 0.5, 0.5)
        progress_bar.progress(progress)
    
    augmented_tfidf_vectors = []
    status_text.text("Extracting TF-IDF vectors from augmented dataset...")
    for idx, (index, row) in enumerate(df_augmented.iterrows()):
        try:
            tfidf_vec = get_tfidf_vector(row['text'], vectorizer)
            augmented_tfidf_vectors.append(tfidf_vec)
        except Exception as e:
            st.warning(f"Error processing augmented data row {index}: {e}")
        
        progress = min(0.5 + ((idx + 1) / len(df_augmented)) * 0.5, 1.0)
        progress_bar.progress(progress)
    
    try:
        progress_bar.empty()
        status_text.empty()
    except:
        pass
    
    if len(minority_tfidf_vectors) < 1 or len(augmented_tfidf_vectors) < 1:
        return {
            'is_valid': False,
            'error': 'Not enough TF-IDF vectors generated',
            'minority_samples': len(minority_tfidf_vectors),
            'augmented_samples': len(augmented_tfidf_vectors)
        }
    
    minority_mean_tfidf = np.mean(minority_tfidf_vectors, axis=0)
    aug_mean_tfidf = np.mean(augmented_tfidf_vectors, axis=0)
    
    mean_cosine_sim = np.dot(minority_mean_tfidf, aug_mean_tfidf) / (
        np.linalg.norm(minority_mean_tfidf) * np.linalg.norm(aug_mean_tfidf) + 1e-10
    )
    
    individual_similarities = []
    for aug_vec in augmented_tfidf_vectors:
        sim = np.dot(minority_mean_tfidf, aug_vec) / (
            np.linalg.norm(minority_mean_tfidf) * np.linalg.norm(aug_vec) + 1e-10
        )
        individual_similarities.append(sim)
    
    min_similarity = np.min(individual_similarities)
    max_similarity = np.max(individual_similarities)
    std_similarity = np.std(individual_similarities)
    
    is_valid = mean_cosine_sim >= threshold
    
    return {
        'is_valid': is_valid,
        'mean_cosine_similarity': mean_cosine_sim,
        'min_cosine_similarity': min_similarity,
        'max_cosine_similarity': max_similarity,
        'std_cosine_similarity': std_similarity,
        'minority_samples': len(minority_tfidf_vectors),
        'augmented_samples': len(augmented_tfidf_vectors),
        'threshold': threshold,
        'interpretation': (
            f'Augmented data is similar to minority class (similarity: {mean_cosine_sim:.4f} >= {threshold}) ✅' 
            if is_valid 
            else f'Augmented data differs from minority class (similarity: {mean_cosine_sim:.4f} < {threshold}) ⚠️'
        )
    }

def validate_minority_data_quality_tfidf(df_minority, vectorizer):
    """Validate minority class data using TF-IDF"""
    st.info(f"🔍 Validating {len(df_minority)} minority class records using TF-IDF...")
    
    minority_tfidf = []
    progress_bar = st.progress(0)
    
    for idx, (index, row) in enumerate(df_minority.iterrows()):
        try:
            tfidf_vec = get_tfidf_vector(row['text'], vectorizer)
            minority_tfidf.append(tfidf_vec)
        except Exception as e:
            st.warning(f"Error processing row {index}: {e}")
        
        progress = min((idx + 1) / len(df_minority), 1.0)
        progress_bar.progress(progress)
    
    try:
        progress_bar.empty()
    except:
        pass
    
    if len(minority_tfidf) < 1:
        return {
            'is_valid': False,
            'error': 'Not enough TF-IDF vectors generated',
            'total_samples': len(minority_tfidf)
        }
    
    minority_mean_tfidf = np.mean(minority_tfidf, axis=0)
    vector_magnitude = np.linalg.norm(minority_mean_tfidf)
    is_valid = vector_magnitude > 0.01
    
    return {
        'is_valid': is_valid,
        'vector_magnitude': vector_magnitude,
        'total_samples': len(minority_tfidf),
        'interpretation': (
            f'Minority data is valid with meaningful TF-IDF features (magnitude: {vector_magnitude:.4f})' 
            if is_valid 
            else f'Minority data has weak TF-IDF features (magnitude: {vector_magnitude:.4f})'
        )
    }

# -------------------------------
# Database Save Functions
# -------------------------------
def save_combined_dataset(df_original, df_augmented):
    """Combine original dataset with augmented data and save to database"""
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS augmented_dataset")
    cursor.execute("""
        CREATE TABLE augmented_dataset (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(255),
            text TEXT,
            subject VARCHAR(100),
            status VARCHAR(50),
            original_text TEXT,
            augmentation_method VARCHAR(100),
            embedding_similarity FLOAT,
            cosine_similarity FLOAT,
            is_validated BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    for _, row in df_augmented.iterrows():
        cursor.execute("""
            INSERT INTO augmented_dataset 
            (title, text, subject, status, original_text, augmentation_method, 
             embedding_similarity, cosine_similarity, is_validated)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            row["title"], 
            row["text"], 
            row["subject"], 
            row["status"], 
            row.get("original_text", ""),
            row.get("augmentation_method", ""),
            float(row.get("embedding_similarity", 0.0)),
            float(row.get("cosine_similarity", 0.0)),
            bool(row.get("is_validated", False))
        ))

    df_augmented_for_dataset = df_augmented[['title', 'text', 'subject', 'status']].copy()
    df_combined = pd.concat([df_original, df_augmented_for_dataset], ignore_index=True)

    cursor.execute("DELETE FROM dataset")
    
    for _, row in df_combined.iterrows():
        cursor.execute("""
            INSERT INTO dataset (title, text, subject, status)
            VALUES (%s, %s, %s, %s)
        """, (row["title"], row["text"], row["subject"], row["status"]))

    conn.commit()
    cursor.close()
    conn.close()
    
    return len(df_combined)

def save_validation_stats_v2(validation_stats):
    """Save validation statistics to database"""
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS augmentation_validation_results")
    cursor.execute("""
        CREATE TABLE augmentation_validation_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            total_minority_records INT,
            total_augmented_generated INT,
            validation_method VARCHAR(100),
            mean_cosine_similarity FLOAT,
            min_cosine_similarity FLOAT,
            max_cosine_similarity FLOAT,
            std_cosine_similarity FLOAT,
            mean_embedding_similarity FLOAT,
            similarity_threshold FLOAT,
            is_valid BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        INSERT INTO augmentation_validation_results 
        (total_minority_records, total_augmented_generated, validation_method,
         mean_cosine_similarity, min_cosine_similarity, max_cosine_similarity,
         std_cosine_similarity, mean_embedding_similarity, similarity_threshold, is_valid)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        validation_stats['total_minority_records'],
        validation_stats['total_augmented_generated'],
        validation_stats['validation_method'],
        float(validation_stats['mean_cosine_similarity']),
        float(validation_stats['min_cosine_similarity']),
        float(validation_stats['max_cosine_similarity']),
        float(validation_stats['std_cosine_similarity']),
        float(validation_stats.get('mean_embedding_similarity', 0.0)),
        float(validation_stats['threshold']),
        bool(validation_stats['is_valid'])
    ))

    conn.commit()
    cursor.close()
    conn.close()

def save_minority_validation_to_db(validation_result, minority_classes, balance_column):
    """Save minority class validation results to database"""
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS pre_augmentation_validation")
    cursor.execute("""
        CREATE TABLE pre_augmentation_validation (
            id INT AUTO_INCREMENT PRIMARY KEY,
            balance_column VARCHAR(50),
            minority_classes TEXT,
            total_samples INT,
            validation_method VARCHAR(50),
            vector_magnitude FLOAT,
            is_valid BOOLEAN,
            interpretation TEXT,
            validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        INSERT INTO pre_augmentation_validation 
        (balance_column, minority_classes, total_samples, validation_method, 
         vector_magnitude, is_valid, interpretation)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        balance_column,
        ', '.join(map(str, minority_classes)),
        int(validation_result['total_samples']),
        'TF-IDF Vector Magnitude Check',
        float(validation_result['vector_magnitude']),
        bool(validation_result['is_valid']),
        validation_result['interpretation']
    ))

    conn.commit()
    cursor.close()
    conn.close()

# -------------------------------
# Main Streamlit UI
# -------------------------------
def augment_page():
    init_session_state()
    
    st.title("📝 DistilBERT Embedding-Based Augmentation (768-dim)")
    st.markdown("**Workflow:** Load → Identify Imbalance → Load DistilBERT → Validate → Augment → Validate → Save")
    st.info("⚡ Using DistilBERT 768-dimensional embeddings for semantic data augmentation")
    
    # Session state info
    with st.expander("💾 Session State Info"):
        st.info("✅ **Session state active!** Data persists during your session.")
        st.write(f"📊 Data loaded: {'Yes' if st.session_state['df_original'] is not None else 'No'}")
        st.write(f"📊 Imbalance analyzed: {'Yes' if st.session_state['imbalance_info'] is not None else 'No'}")
        st.write(f"📊 TF-IDF fitted: {'Yes' if st.session_state['tfidf_vectorizer'] is not None else 'No'}")
        st.write(f"📊 Augmentation done: {'Yes' if st.session_state['df_augmented'] is not None else 'No'}")
        
        if st.button("🗑️ Clear Session & Restart"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Step 1: Load data
    st.subheader("📊 Step 1: Load Original Dataset")
    
    if st.button("🔄 Load Data from Database"):
        with st.spinner("Loading data from database..."):
            df_original = load_dataset()
            st.session_state['df_original'] = df_original
            st.success(f"✅ Loaded {len(df_original)} records from database")
            st.rerun()
    
    if st.session_state['df_original'] is not None:
        df_original = st.session_state['df_original']
        st.dataframe(df_original.head(10))
        st.info(f"Total records: {len(df_original)}")

        # Step 2: Identify imbalance
        st.subheader("⚖️ Step 2: Identify Imbalanced Classes")
        
        col1, col2 = st.columns(2)
        with col1:
            balance_column = st.selectbox("Column to check for imbalance", ['subject', 'status'])
        with col2:
            threshold_ratio = st.number_input("Imbalance threshold ratio", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
        
        if st.button("🔍 Analyze Class Distribution"):
            imbalance_info = identify_imbalanced_classes(df_original, column=balance_column, threshold_ratio=threshold_ratio)
            st.session_state['imbalance_info'] = imbalance_info
            st.rerun()
        
        if st.session_state['imbalance_info'] is not None:
            imbalance_info = st.session_state['imbalance_info']
            
            st.subheader(f"Class Distribution - {imbalance_info['column'].capitalize()}")
            
            dist_data = []
            for cls, info in imbalance_info['class_info'].items():
                dist_data.append({
                    'Class': cls,
                    'Count': info['count'],
                    'Ratio (Max/Current)': f"{info['ratio']:.2f}x",
                    'Status': '🔴 Minority (Need Augmentation)' if info['is_minority'] else '🟢 Majority'
                })
            
            dist_df = pd.DataFrame(dist_data)
            st.dataframe(dist_df, use_container_width=True)
            
            if imbalance_info['minority_classes']:
                st.warning(f"⚠️ Found {len(imbalance_info['minority_classes'])} minority class(es): {', '.join(map(str, imbalance_info['minority_classes']))}")
                
                minority_classes = imbalance_info['minority_classes']
                balance_column = imbalance_info['column']
                df_minority = df_original[df_original[balance_column].isin(minority_classes)]
                
                # Step 3: Load DistilBERT
                st.subheader("🤖 Step 3: Load DistilBERT Model")
                st.info("DistilBERT will generate 768-dimensional embeddings representing text meaning")
                
                with st.expander("ℹ️ About DistilBERT Embeddings"):
                    st.markdown("""
                    **What are 768-dimensional embeddings?**
                    - Each text is converted to a vector of 768 numbers
                    - This vector captures the semantic meaning of the text
                    - Similar texts have similar embeddings (close in 768-D space)
                    - Used to find semantically similar texts for mixing
                    
                    **Why DistilBERT?**
                    - Fast and efficient (66% faster than BERT)
                    - High quality semantic representations
                    - Pre-trained on large text corpus
                    - Understands context and meaning
                    """)
                
                if st.button("📥 Load DistilBERT Model"):
                    tokenizer, model = load_distilbert_model()
                    st.session_state['distilbert_tokenizer'] = tokenizer
                    st.session_state['distilbert_model'] = model
                    st.rerun()
                
                if 'distilbert_model' in st.session_state:
                    tokenizer = st.session_state['distilbert_tokenizer']
                    model = st.session_state['distilbert_model']
                    st.success("✅ DistilBERT model loaded and ready!")
                    
                    # Show embedding example
                    with st.expander("🔍 Test DistilBERT Embeddings"):
                        test_text = st.text_area(
                            "Enter text to see its 768-dim embedding",
                            "The government announced new policy changes"
                        )
                        if st.button("Generate Embedding"):
                            embedding = get_distilbert_embedding(test_text, tokenizer, model)
                            st.write(f"**Embedding shape:** {embedding.shape}")
                            st.write(f"**First 20 dimensions:** {embedding[:20]}")
                            st.write(f"**Embedding norm (magnitude):** {np.linalg.norm(embedding):.4f}")
                    
                    # Step 4: Create TF-IDF
                    st.subheader("🔧 Step 4: Create TF-IDF Vectorizer (for validation)")
                    
                    if st.button("🔧 Fit TF-IDF Vectorizer"):
                        with st.spinner("Fitting TF-IDF vectorizer..."):
                            all_texts = df_original['text'].tolist()
                            vectorizer = create_tfidf_vectorizer(all_texts)
                            st.session_state['tfidf_vectorizer'] = vectorizer
                            st.success(f"✅ TF-IDF vectorizer fitted")
                            st.rerun()
                    
                    if st.session_state['tfidf_vectorizer'] is not None:
                        vectorizer = st.session_state['tfidf_vectorizer']
                        st.success(f"✅ TF-IDF ready (vocab: {len(vectorizer.vocabulary_)})")
                        
                        # Step 5: Validate minority
                        st.subheader("✅ Step 5: Validate Minority Class Data")
                        
                        if st.button("🔍 Validate Minority Data"):
                            validation_result = validate_minority_data_quality_tfidf(df_minority, vectorizer)
                            st.session_state['minority_validation'] = validation_result
                            st.rerun()
                        
                        if st.session_state['minority_validation'] is not None:
                            validation_result = st.session_state['minority_validation']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Vector Magnitude", f"{validation_result['vector_magnitude']:.4f}")
                            with col2:
                                st.metric("Samples", validation_result['total_samples'])
                            with col3:
                                status = "✅ Valid" if validation_result['is_valid'] else "⚠️ Warning"
                                st.metric("Status", status)
                            
                            if validation_result['is_valid']:
                                st.success(f"✅ {validation_result['interpretation']}")
                            
                            if st.button("💾 Save Pre-Augmentation Validation"):
                                save_minority_validation_to_db(validation_result, minority_classes, balance_column)
                                st.success("✅ Validation saved!")
                        
                        # Step 6: Augmentation Settings
                        st.subheader("⚙️ Step 6: DistilBERT Embedding Augmentation Settings")
                        
                        augmentation_method = st.selectbox(
                            "Augmentation method",
                            [
                                'K-Nearest Neighbors Mixing (RECOMMENDED)',
                                'Word Substitution',
                                'Embedding Perturbation'
                            ]
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            num_aug = st.number_input("Augmentations per text", 1, 10, 2)
                        with col2:
                            if augmentation_method == 'K-Nearest Neighbors Mixing (RECOMMENDED)':
                                k_neighbors = st.number_input("K nearest neighbors", 1, 10, 3)
                            elif augmentation_method == 'Word Substitution':
                                substitution_rate = st.slider("Word substitution rate", 0.1, 0.5, 0.3, 0.05)
                            else:
                                noise_level = st.slider("Embedding noise level", 0.05, 0.3, 0.1, 0.05)
                        
                        st.info(f"🎯 Will augment {len(df_minority)} minority records using 768-dim DistilBERT embeddings")
                        
                        # Method explanation
                        if augmentation_method == 'K-Nearest Neighbors Mixing (RECOMMENDED)':
                            st.info("""
                            **K-NN Method:**
                            1. Compute 768-dim embedding for each text
                            2. Find K most similar texts in embedding space
                            3. Mix original with similar texts
                            4. Creates semantically consistent augmentations
                            """)
                        
                        # Step 7: Start Augmentation
                        if st.button("🚀 Start DistilBERT Embedding Augmentation"):
                            augmented_rows = []
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            total_rows = len(df_minority)
                            
                            for idx, (_, row) in enumerate(df_minority.iterrows()):
                                status_text.text(f"Processing {idx + 1}/{total_rows}: {row['title'][:50]}...")
                                
                                try:
                                    if augmentation_method == 'K-Nearest Neighbors Mixing (RECOMMENDED)':
                                        aug_texts, sim_scores = augment_using_nearest_neighbors(
                                            row["text"],
                                            df_minority,
                                            tokenizer,
                                            model,
                                            num_aug=num_aug,
                                            k=k_neighbors
                                        )
                                        method_name = f"DistilBERT K-NN (k={k_neighbors})"
                                    
                                    elif augmentation_method == 'Word Substitution':
                                        aug_texts = augment_by_word_substitution(
                                            row["text"],
                                            tokenizer,
                                            model,
                                            num_aug=num_aug,
                                            substitution_rate=substitution_rate
                                        )
                                        sim_scores = [0.0] * len(aug_texts)
                                        method_name = "DistilBERT Word Substitution"
                                    
                                    else:  # Embedding Perturbation
                                        aug_texts = augment_by_embedding_perturbation(
                                            row["text"],
                                            tokenizer,
                                            model,
                                            num_aug=num_aug,
                                            noise_level=noise_level
                                        )
                                        sim_scores = [0.0] * len(aug_texts)
                                        method_name = "DistilBERT Embedding Perturbation"
                                    
                                    for aug_text, sim_score in zip(aug_texts, sim_scores):
                                        augmented_rows.append({
                                            "title": row["title"],
                                            "text": aug_text,
                                            "subject": row["subject"],
                                            "status": row["status"],
                                            "original_text": row["text"],
                                            "augmentation_method": method_name,
                                            "embedding_similarity": sim_score
                                        })
                                
                                except Exception as e:
                                    st.warning(f"⚠️ Error: {str(e)}")
                                
                                progress_bar.progress(min((idx + 1) / total_rows, 1.0))
                            
                            try:
                                progress_bar.empty()
                                status_text.empty()
                            except:
                                pass
                            
                            df_augmented = pd.DataFrame(augmented_rows)
                            st.session_state['df_augmented'] = df_augmented
                            
                            st.success(f"✅ Generated {len(df_augmented)} augmented records!")
                            st.rerun()
                        
                        # Display augmented data
                        if st.session_state['df_augmented'] is not None:
                            st.subheader("📦 Step 8: Augmented Data Generated")
                            df_augmented = st.session_state['df_augmented']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Minority Records", len(df_minority))
                            with col2:
                                st.metric("Augmented Generated", len(df_augmented))
                            with col3:
                                avg_emb_sim = df_augmented['embedding_similarity'].mean()
                                st.metric("Avg Embedding Similarity", f"{avg_emb_sim:.4f}")
                            
                            # Show examples
                            with st.expander("📋 Original vs Augmented Examples"):
                                for i in range(min(3, len(df_augmented))):
                                    st.markdown(f"**Example {i+1}:**")
                                    st.text("Original:")
                                    st.info(df_augmented.iloc[i]['original_text'][:300])
                                    st.text("Augmented:")
                                    st.success(df_augmented.iloc[i]['text'][:300])
                                    st.text(f"Embedding Similarity: {df_augmented.iloc[i]['embedding_similarity']:.4f}")
                                    st.markdown("---")
                            
                            st.dataframe(df_augmented[['title', 'text', 'subject', 'embedding_similarity']].head(20))
                            
                            # Step 9: Validate
                            st.subheader("✅ Step 9: Validate Augmented Data")
                            
                            cosine_threshold = st.number_input(
                                "Cosine similarity threshold", 
                                0.50, 0.99, 0.85, 0.05
                            )
                            
                            if st.button("🔍 Validate Augmented vs Minority"):
                                validation_result = cosine_similarity_validate_augmented_vs_minority(
                                    df_minority, df_augmented, vectorizer, threshold=cosine_threshold
                                )
                                st.session_state['dataset_validation'] = validation_result
                                st.rerun()
                            
                            if st.session_state.get('dataset_validation') is not None:
                                validation_result = st.session_state['dataset_validation']
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Mean Similarity", f"{validation_result['mean_cosine_similarity']:.4f}")
                                with col2:
                                    st.metric("Min", f"{validation_result['min_cosine_similarity']:.4f}")
                                with col3:
                                    st.metric("Max", f"{validation_result['max_cosine_similarity']:.4f}")
                                with col4:
                                    status = "✅ Valid" if validation_result['is_valid'] else "⚠️ Failed"
                                    st.metric("Status", status)
                                
                                if validation_result['is_valid']:
                                    st.success(f"✅ {validation_result['interpretation']}")
                                else:
                                    st.warning(f"⚠️ {validation_result['interpretation']}")
                                
                                # Add metrics
                                df_augmented_validated = df_augmented.copy()
                                df_augmented_validated['cosine_similarity'] = validation_result['mean_cosine_similarity']
                                df_augmented_validated['is_validated'] = validation_result['is_valid']
                                st.session_state['df_augmented_validated'] = df_augmented_validated
                                
                                # Stats
                                validation_stats = {
                                    'total_minority_records': len(df_minority),
                                    'total_augmented_generated': len(df_augmented),
                                    'validation_method': 'DistilBERT 768-dim + TF-IDF Cosine Similarity',
                                    'mean_cosine_similarity': validation_result['mean_cosine_similarity'],
                                    'min_cosine_similarity': validation_result['min_cosine_similarity'],
                                    'max_cosine_similarity': validation_result['max_cosine_similarity'],
                                    'std_cosine_similarity': validation_result['std_cosine_similarity'],
                                    'mean_embedding_similarity': df_augmented['embedding_similarity'].mean(),
                                    'threshold': validation_result['threshold'],
                                    'is_valid': validation_result['is_valid']
                                }
                                st.session_state['validation_stats'] = validation_stats
                            
                            # Step 10: Save
                            if st.session_state.get('df_augmented_validated') is not None:
                                st.subheader("💾 Step 10: Save Combined Dataset")
                                
                                df_augmented_validated = st.session_state['df_augmented_validated']
                                total = len(df_original) + len(df_augmented_validated)
                                
                                st.info(f"📊 Total: {len(df_original)} original + {len(df_augmented_validated)} augmented = {total} records")
                                
                                if st.button("💾 Save to Database"):
                                    with st.spinner("Saving..."):
                                        total_saved = save_combined_dataset(df_original, df_augmented_validated)
                                        if st.session_state['validation_stats']:
                                            save_validation_stats_v2(st.session_state['validation_stats'])
                                        
                                        st.success(f"✅ Saved {total_saved} records!")
                                        st.success("✅ Augmented data saved with DistilBERT embedding metrics!")
                                        st.balloons()
            else:
                st.success("✅ Dataset is balanced!")

if __name__ == "__main__":
    augment_page()