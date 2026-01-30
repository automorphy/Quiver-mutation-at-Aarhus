import sqlite3
import numpy as np
import sys
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lib.config import get_db_path
from lib.svm_utils import compute_exact_kernel, compute_diag_kernel, normalize_kernel_cosine

def load_data(limit=None, threshold=5000):
    db_path = get_db_path("quivers_rank4_250k.db")
    print(f"Loading data from {db_path}...")
    conn = sqlite3.connect(db_path)
    query = """
        SELECT b12, b13, b14, b23, b24, b34, label 
        FROM quivers
    """
    if limit:
        # Load more initially to account for filtering
        query += f" LIMIT {limit * 2}"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Filter by threshold
    mask = (df[['b12', 'b13', 'b14', 'b23', 'b24', 'b34']].abs() <= threshold).all(axis=1)
    df_filtered = df[mask]
    
    if limit and len(df_filtered) > limit:
        df_filtered = df_filtered.iloc[:limit]
        
    print(f"Loaded {len(df)} samples. Retained {len(df_filtered)} after filtering (threshold={threshold}).")
    
    X = df_filtered[['b12', 'b13', 'b14', 'b23', 'b24', 'b34']].values
    y = df_filtered['label'].values
    return X, y

def train_and_evaluate(degree=3, limit=10000):
    X, y = load_data(limit=limit)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Compute Kernels
    K_train_raw = compute_exact_kernel(X_train, X_train, degree=degree)
    K_test_raw = compute_exact_kernel(X_test, X_train, degree=degree)
    
    # Compute Diagonals for normalization
    diag_train = compute_diag_kernel(X_train, degree=degree)
    diag_test = compute_diag_kernel(X_test, degree=degree)
    
    # Normalize
    K_train, K_test = normalize_kernel_cosine(K_train_raw, K_test_raw, diag_train, diag_test)
    
    # Train SVM
    print("Training SVM with precomputed kernel...")
    svm = SVC(kernel='precomputed', C=1.0, class_weight='balanced')
    svm.fit(K_train, y_train)
    
    # Predict
    print("Evaluating...")
    y_pred = svm.predict(K_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*40)
    print(f"Results for Polynomial Kernel (Degree {degree})")
    print("="*40)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return acc

if __name__ == "__main__":
    # Test a few degrees
    degrees_to_test = [5, 6, 7, 8]
    results = {}
    
    # Using a smaller limit for faster iteration during initial test
    # Increase this for full run
    SAMPLE_LIMIT = 100000
    
    for d in degrees_to_test:
        print(f"\n\n>>> Starting experiment for degree={d}")
        acc = train_and_evaluate(degree=d, limit=SAMPLE_LIMIT)
        results[d] = acc
        
    print("\n\n=== Final Summary ===")
    for d, acc in results.items():
        print(f"Degree {d}: {acc:.4f}")
