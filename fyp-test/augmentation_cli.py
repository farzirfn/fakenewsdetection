#!/usr/bin/env python3
"""
Text Augmentation CLI Tool
Run data augmentation process from command line with progress tracking
"""

import argparse
import sys
import mysql.connector
import pandas as pd
import torch
import nlpaug.augmenter.word as naw
from transformers import DistilBertTokenizer, DistilBertModel
from scipy.stats import ks_2samp
import numpy as np
from datetime import datetime
from tqdm import tqdm
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('augmentation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------
# DB Connection
# -------------------------------
def create_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="fyp"
    )

# -------------------------------
# Load Dataset
# -------------------------------
def load_dataset():
    """Load original dataset from database"""
    logger.info("📂 Loading dataset from database...")
    conn = create_connection()
    df = pd.read_sql("SELECT title, text, subject, status FROM dataset", conn)
    conn.close()
    logger.info(f"✅ Loaded {len(df)} records")
    return df

# -------------------------------
# Identify Imbalanced Classes
# -------------------------------
def identify_imbalanced_classes(df, column='subject', threshold_ratio=1.5):
    """Identify minority classes that need augmentation"""
    logger.info(f"⚖️ Analyzing class distribution in '{column}' column...")
    
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
            'is_minority': is_minority
        }
        
        if is_minority:
            minority_classes.append(cls)
            logger.info(f"  🔴 MINORITY: {cls} - {count} records (ratio: {ratio:.2f}x)")
        else:
            logger.info(f"  🟢 MAJORITY: {cls} - {count} records")
    
    return {
        'class_counts': class_counts,
        'class_info': class_info,
        'minority_classes': minority_classes,
        'max_count': max_count,
        'column': column
    }

# -------------------------------
# Augmentation
# -------------------------------
def augment_text_with_bert(text, num_aug=2):
    """Augment text using DistilBERT"""
    distilbert_aug = naw.ContextualWordEmbsAug(
        model_path='distilbert-base-uncased',
        action="substitute"
    )
    augmented = distilbert_aug.augment(text, n=num_aug)
    if isinstance(augmented, str):
        return [augmented]
    return augmented

# -------------------------------
# Validation
# -------------------------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def get_embedding(text):
    """Get DistilBERT embedding"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:,0,:].squeeze().numpy()

def ks_validate(original_text, augmented_text, threshold=0.05):
    """Validate augmented text using K-S test"""
    emb_orig = get_embedding(original_text)
    emb_aug = get_embedding(augmented_text)
    
    if len(emb_orig.shape) > 1:
        emb_orig = emb_orig.flatten()
    if len(emb_aug.shape) > 1:
        emb_aug = emb_aug.flatten()
    
    stat, p_value = ks_2samp(emb_orig, emb_aug)
    is_valid = p_value > threshold
    
    return is_valid, {"ks_stat": stat, "p_value": p_value}

def validate_minority_data_quality(df_minority):
    """Validate minority class data quality"""
    logger.info(f"🔍 Validating {len(df_minority)} minority records...")
    
    minority_embeddings = []
    
    for idx, row in tqdm(df_minority.iterrows(), total=len(df_minority), desc="Generating embeddings"):
        try:
            emb = get_embedding(row['text'])
            minority_embeddings.append(emb.flatten())
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}")
    
    if len(minority_embeddings) < 2:
        return {
            'is_valid': False,
            'error': 'Not enough embeddings generated',
            'total_samples': len(minority_embeddings)
        }
    
    # Split and test
    mid_point = len(minority_embeddings) // 2
    first_half = minority_embeddings[:mid_point]
    second_half = minority_embeddings[mid_point:]
    
    first_half_dist = np.concatenate(first_half)
    second_half_dist = np.concatenate(second_half)
    
    stat, p_value = ks_2samp(first_half_dist, second_half_dist)
    is_valid = p_value > 0.05
    
    result = {
        'is_valid': is_valid,
        'ks_stat': float(stat),
        'p_value': float(p_value),
        'total_samples': len(minority_embeddings),
        'first_half_samples': len(first_half),
        'second_half_samples': len(second_half),
        'interpretation': 'Minority data is internally consistent' if is_valid else 'Minority data shows internal inconsistency'
    }
    
    logger.info(f"  K-S Statistic: {stat:.4f}")
    logger.info(f"  P-Value: {p_value:.4f}")
    logger.info(f"  Status: {'✅ Valid' if is_valid else '⚠️ Warning'}")
    logger.info(f"  {result['interpretation']}")
    
    return result

# -------------------------------
# Save Functions
# -------------------------------
def save_minority_validation_to_db(validation_result, minority_classes, balance_column):
    """Save pre-augmentation validation to database"""
    logger.info("💾 Saving validation results to database...")
    
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pre_augmentation_validation (
            id INT AUTO_INCREMENT PRIMARY KEY,
            balance_column VARCHAR(50),
            minority_classes TEXT,
            total_samples INT,
            first_half_samples INT,
            second_half_samples INT,
            ks_statistic FLOAT,
            p_value FLOAT,
            is_valid BOOLEAN,
            interpretation TEXT,
            validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        INSERT INTO pre_augmentation_validation 
        (balance_column, minority_classes, total_samples, first_half_samples, 
         second_half_samples, ks_statistic, p_value, is_valid, interpretation)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        balance_column,
        ', '.join(map(str, minority_classes)),
        int(validation_result['total_samples']),
        int(validation_result['first_half_samples']),
        int(validation_result['second_half_samples']),
        float(validation_result['ks_stat']),
        float(validation_result['p_value']),
        bool(validation_result['is_valid']),
        validation_result['interpretation']
    ))

    conn.commit()
    cursor.close()
    conn.close()
    logger.info("✅ Validation results saved")

def save_combined_dataset(df_combined):
    """Save combined dataset to database"""
    logger.info("💾 Saving combined dataset to database...")
    
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS combined_dataset (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(255),
            text TEXT,
            subject VARCHAR(100),
            status VARCHAR(50),
            is_augmented BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("TRUNCATE TABLE combined_dataset")

    for _, row in tqdm(df_combined.iterrows(), total=len(df_combined), desc="Saving to database"):
        cursor.execute("""
            INSERT INTO combined_dataset (title, text, subject, status, is_augmented)
            VALUES (%s, %s, %s, %s, %s)
        """, (row["title"], row["text"], row["subject"], row["status"], row["is_augmented"]))

    conn.commit()
    cursor.close()
    conn.close()
    logger.info(f"✅ Saved {len(df_combined)} records to database")

def save_validation_stats(validation_stats):
    """Save validation statistics to database"""
    logger.info("💾 Saving validation statistics...")
    
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS validation_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            total_original INT,
            total_augmented_generated INT,
            total_valid_augmented INT,
            total_invalid_augmented INT,
            avg_ks_stat FLOAT,
            avg_p_value FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        INSERT INTO validation_results 
        (total_original, total_augmented_generated, total_valid_augmented, 
         total_invalid_augmented, avg_ks_stat, avg_p_value)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        validation_stats['total_original'],
        validation_stats['total_augmented_generated'],
        validation_stats['total_valid_augmented'],
        validation_stats['total_invalid_augmented'],
        validation_stats['avg_ks_stat'],
        validation_stats['avg_p_value']
    ))

    conn.commit()
    cursor.close()
    conn.close()
    logger.info("✅ Statistics saved")

# -------------------------------
# Main Augmentation Process
# -------------------------------
def run_augmentation(column='subject', threshold_ratio=1.5, num_aug=2, ks_threshold=0.05, skip_validation=False):
    """Main augmentation process"""
    
    start_time = datetime.now()
    logger.info("="*80)
    logger.info("🚀 STARTING AUGMENTATION PROCESS")
    logger.info("="*80)
    logger.info(f"Parameters:")
    logger.info(f"  - Balance column: {column}")
    logger.info(f"  - Threshold ratio: {threshold_ratio}")
    logger.info(f"  - Augmentations per text: {num_aug}")
    logger.info(f"  - K-S threshold: {ks_threshold}")
    logger.info(f"  - Skip validation: {skip_validation}")
    logger.info("")
    
    # Step 1: Load data
    df_original = load_dataset()
    
    # Step 2: Identify imbalance
    imbalance_info = identify_imbalanced_classes(df_original, column=column, threshold_ratio=threshold_ratio)
    
    if not imbalance_info['minority_classes']:
        logger.info("✅ Dataset is balanced! No augmentation needed.")
        return
    
    logger.info(f"\n⚠️ Found {len(imbalance_info['minority_classes'])} minority class(es):")
    logger.info(f"   {', '.join(map(str, imbalance_info['minority_classes']))}")
    
    # Step 3: Filter minority data
    minority_classes = imbalance_info['minority_classes']
    df_minority = df_original[df_original[column].isin(minority_classes)]
    logger.info(f"\n🎯 Will augment {len(df_minority)} records from minority classes")
    
    # Step 4: Validate minority data
    if not skip_validation:
        logger.info("\n" + "="*80)
        logger.info("STEP: PRE-AUGMENTATION VALIDATION")
        logger.info("="*80)
        validation_result = validate_minority_data_quality(df_minority)
        
        if not validation_result['is_valid']:
            logger.warning("⚠️ Validation warning: Minority data shows inconsistency")
            logger.warning("   You can still proceed, but review data manually")
        
        # Save validation results
        save_minority_validation_to_db(validation_result, minority_classes, column)
    else:
        logger.info("\n⚠️ Skipping pre-augmentation validation")
    
    # Step 5: Augmentation
    logger.info("\n" + "="*80)
    logger.info("STEP: AUGMENTATION & VALIDATION")
    logger.info("="*80)
    
    augmented_rows = []
    validation_results = []
    
    for idx, (_, row) in enumerate(tqdm(df_minority.iterrows(), total=len(df_minority), desc="Augmenting")):
        try:
            # Augment
            aug_texts = augment_text_with_bert(row["text"], num_aug=num_aug)
            
            for aug_text in aug_texts:
                # Validate
                is_valid, ks_result = ks_validate(row["text"], aug_text, threshold=ks_threshold)
                
                validation_results.append({
                    'is_valid': is_valid,
                    'ks_stat': ks_result['ks_stat'],
                    'p_value': ks_result['p_value']
                })
                
                if is_valid:
                    augmented_rows.append({
                        "title": row["title"],
                        "text": aug_text,
                        "subject": row["subject"],
                        "status": row["status"],
                        "is_augmented": True
                    })
        
        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
    
    # Step 6: Combine
    logger.info("\n" + "="*80)
    logger.info("STEP: COMBINING DATASETS")
    logger.info("="*80)
    
    df_original_flagged = df_original.copy()
    df_original_flagged['is_augmented'] = False
    df_augmented = pd.DataFrame(augmented_rows)
    
    if len(df_augmented) > 0:
        df_combined = pd.concat([df_original_flagged, df_augmented], ignore_index=True)
    else:
        df_combined = df_original_flagged
    
    # Statistics
    logger.info(f"  Original records: {len(df_original)}")
    logger.info(f"  Minority records: {len(df_minority)}")
    logger.info(f"  Generated augmentations: {len(validation_results)}")
    logger.info(f"  Valid augmentations: {len(df_augmented)}")
    logger.info(f"  Invalid augmentations: {len(validation_results) - len(df_augmented)}")
    logger.info(f"  Total combined: {len(df_combined)}")
    
    if validation_results:
        avg_ks = np.mean([v['ks_stat'] for v in validation_results])
        avg_p = np.mean([v['p_value'] for v in validation_results])
        valid_rate = len(df_augmented) / len(validation_results) * 100
        
        logger.info(f"  Average K-S stat: {avg_ks:.4f}")
        logger.info(f"  Average P-value: {avg_p:.4f}")
        logger.info(f"  Validation rate: {valid_rate:.1f}%")
    
    # Step 7: Save to database
    logger.info("\n" + "="*80)
    logger.info("STEP: SAVING TO DATABASE")
    logger.info("="*80)
    
    save_combined_dataset(df_combined)
    
    if validation_results:
        validation_stats = {
            'total_original': len(df_original),
            'total_augmented_generated': len(validation_results),
            'total_valid_augmented': len(df_augmented),
            'total_invalid_augmented': len(validation_results) - len(df_augmented),
            'avg_ks_stat': avg_ks,
            'avg_p_value': avg_p
        }
        save_validation_stats(validation_stats)
    
    # Summary
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*80)
    logger.info("✅ AUGMENTATION COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"⏱️  Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"📊 Total records in database: {len(df_combined)}")
    logger.info(f"🎯 Check 'combined_dataset' table for results")
    logger.info("="*80)

# -------------------------------
# CLI Interface
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Text Augmentation CLI Tool - Augment imbalanced text data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (default settings)
  python augmentation_cli.py

  # Specify column and threshold
  python augmentation_cli.py --column status --threshold 2.0

  # More augmentations per text
  python augmentation_cli.py --num-aug 5

  # Skip pre-validation
  python augmentation_cli.py --skip-validation

  # Full custom
  python augmentation_cli.py --column subject --threshold 1.5 --num-aug 3 --ks-threshold 0.1
        """
    )
    
    parser.add_argument(
        '--column',
        type=str,
        default='subject',
        choices=['subject', 'status'],
        help='Column to check for imbalance (default: subject)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=1.5,
        help='Imbalance threshold ratio (default: 1.5)'
    )
    
    parser.add_argument(
        '--num-aug',
        type=int,
        default=2,
        help='Number of augmentations per text (default: 2)'
    )
    
    parser.add_argument(
        '--ks-threshold',
        type=float,
        default=0.05,
        help='K-S test p-value threshold (default: 0.05)'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip pre-augmentation validation'
    )
    
    args = parser.parse_args()
    
    try:
        run_augmentation(
            column=args.column,
            threshold_ratio=args.threshold,
            num_aug=args.num_aug,
            ks_threshold=args.ks_threshold,
            skip_validation=args.skip_validation
        )
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()