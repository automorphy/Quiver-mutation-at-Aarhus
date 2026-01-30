import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time
import random
import os
import sys

# Add repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lib.config import get_external_data_path
from lib.svm_utils import compute_exact_kernel, compute_diag_kernel, normalize_kernel_cosine

def load_external_data(limit=10000):
    """Loads quiver data from text files as in Experiment 3."""
    
    def read_exchange_matrix(file_path):
        data = []
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            return []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    row = [float(x) for x in line.strip().split(',')]
                    if len(row) == 16:
                        # Map 16 elements to our 6 features: b12, b13, b14, b23, b24, b34
                        # Indices: 1, 2, 3, 6, 7, 11
                        features = [row[1], row[2], row[3], row[6], row[7], row[11]]
                        data.append(features)
                except ValueError:
                    continue
        return data

    def read_categories(file_path):
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            return []
        with open(file_path, 'r') as f:
            return [int(float(line.strip())) for line in f if line.strip()]

    print("Loading external data...")
    ma_x = read_exchange_matrix(get_external_data_path("Experiment_3", "MA_ALL_DEPTH_7_type_0.txt"))
    ma_y = read_categories(get_external_data_path("Experiment_3", "MA_ALL_DEPTH_7_type_0_cat.txt"))
    nma_x = read_exchange_matrix(get_external_data_path("Experiment_3", "NMA_ALL_Depth_5__type_1.txt"))
    nma_y = read_categories(get_external_data_path("Experiment_3", "NMA_ALL_Depth_5__type_1_cat.txt"))

    # Balance classes
    if not ma_x or not nma_x:
        print("Error: No data loaded.")
        return None, None
    splicing = min(len(ma_x), len(nma_x))
    print(f"Balanced class size: {splicing}")
    
    # Shuffle and sample
    ma_indices = random.sample(range(len(ma_x)), splicing)
    nma_indices = random.sample(range(len(nma_x)), splicing)
    
    X = [ma_x[i] for i in ma_indices] + [nma_x[i] for i in nma_indices]
    y = [ma_y[i] for i in ma_indices] + [nma_y[i] for i in nma_indices]
    
    X = np.array(X, dtype=object) 
    y = np.array(y)
    
    if limit and len(X) > limit:
        indices = random.sample(range(len(X)), limit)
        X = X[indices]
        y = y[indices]
        print(f"Sampled down to {limit} points.")

    return X, y

def train_and_evaluate(X, y, degree=3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    K_train_raw = compute_exact_kernel(X_train, X_train, degree=degree)
    K_test_raw = compute_exact_kernel(X_test, X_train, degree=degree)
    diag_train = compute_diag_kernel(X_train, degree=degree)
    diag_test = compute_diag_kernel(X_test, degree=degree)
    K_train, K_test = normalize_kernel_cosine(K_train_raw, K_test_raw, diag_train, diag_test)
    
    print("Training SVM...")
    svm = SVC(kernel='precomputed', C=1.0, class_weight='balanced')
    svm.fit(K_train, y_train)
    
    print("Evaluating...")
    y_pred = svm.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nResults for Polynomial Kernel (Degree {degree})")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    return acc

if __name__ == "__main__":
    SAMPLE_LIMIT = 5000 
    X, y = load_external_data(limit=SAMPLE_LIMIT)
    if X is not None:
        degrees_to_test = [3, 4, 5]
        results = {}
        for d in degrees_to_test:
            print(f"\n>>> Starting experiment for degree={d}")
            acc = train_and_evaluate(X, y, degree=d)
            results[d] = acc
        print("\n=== Final Summary ===")
        for d, acc in results.items():
            print(f"Degree {d}: {acc:.4f}")