import numpy as np
import os
import sys
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, matthews_corrcoef

# Add repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lib.config import get_external_data_path, get_db_path

def sign_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

# --- Data Loading Functions ---

def load_exp3_data(limit=10000):
    ma_file = get_external_data_path("Experiment_3", "MA_ALL_DEPTH_7_type_0.txt")
    nma_file = get_external_data_path("Experiment_3", "NMA_ALL_Depth_5__type_1.txt")
    def read_matrix(path):
        data = []
        if not os.path.exists(path): raise FileNotFoundError(f"File not found: {path}")
        with open(path, 'r') as f:
            for line in f:
                try:
                    row = [float(x) for x in line.strip().split(',')]
                    if len(row) == 16:
                        data.append([row[1], row[2], row[3], row[6], row[7], row[11]])
                except: continue
        return data
    ma_X = read_matrix(ma_file)
    nma_X = read_matrix(nma_file)
    
    # Shuffle and combine
    X = np.array(ma_X + nma_X)
    y = np.array([1] * len(ma_X) + [0] * len(nma_X))
    idx = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    
    if limit:
        X, y = X[:limit], y[:limit]
    return X, y, "Exp3 (Hard)"

def load_exp4_data(limit=10000):
    ma_file = get_external_data_path("Experiment_4", "Fourth_Experiment_MA_data.txt")
    nma_file = get_external_data_path("Experiment_4", "Fourth_Experiment_NMA_data.txt")
    def read_matrix(path):
        data = []
        if not os.path.exists(path): raise FileNotFoundError(f"File not found: {path}")
        with open(path, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                if len(row) > 11:
                    data.append([float(row[1]), float(row[2]), float(row[3]), float(row[6]), float(row[7]), float(row[11])])
        return data
    ma_X = read_matrix(ma_file)
    nma_X = read_matrix(nma_file)
    X = np.array(ma_X + nma_X)
    y = np.array([1] * len(ma_X) + [0] * len(nma_X))
    idx = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    if limit:
        X, y = X[:limit], y[:limit]
    return X, y, "Exp4 (Easy)"

def load_our_data(limit=10000):
    db_path = get_db_path("quivers_rank4_250k.db")
    if not os.path.exists(db_path):
        return np.array([]), np.array([]), "Our Data (Medium)"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT b12, b13, b14, b23, b24, b34, label FROM quivers ORDER BY RANDOM() LIMIT " + str(limit), conn)
    conn.close()
    return df[['b12', 'b13', 'b14', 'b23', 'b24', 'b34']].values, df['label'].values, "Our Data (Medium)"

def run_comparison():
    TRAIN_SIZE = 5000 
    TEST_SIZE = 2000
    
    datasets = [
        load_exp3_data(limit=TRAIN_SIZE+TEST_SIZE),
        load_exp4_data(limit=TRAIN_SIZE+TEST_SIZE),
        load_our_data(limit=TRAIN_SIZE+TEST_SIZE)
    ]
    
    poly = PolynomialFeatures(degree=4, include_bias=True)
    
    print("\nStarting Cross-Evaluation Matrix with LOG SCALING")
    print(f"{'Train Set':<15} | {'Test Set':<15} | {'Accuracy':<10} | {'MCC':<10}")
    print("-" * 60)
    
    # Pre-split all datasets into train/test to be fair
    processed_datasets = []
    for X, y, name in datasets:
        if len(y) == 0:
            continue
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE, random_state=42, stratify=y)
        processed_datasets.append({'X_tr': X_tr, 'X_te': X_te, 'y_tr': y_tr, 'y_te': y_te, 'name': name})

    for train_pkg in processed_datasets:
        # Scale and Poly Transform Train
        X_train_scaled = sign_log(train_pkg['X_tr'])
        X_train_poly = poly.fit_transform(X_train_scaled)
        
        # Class weights
        n_pos = np.sum(train_pkg['y_tr'] == 1)
        n_neg = np.sum(train_pkg['y_tr'] == 0)
        clf = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=0.1, 
                        class_weight={0: 1, 1: n_neg/n_pos if n_pos > 0 else 1}, 
                        max_iter=10000, random_state=42)
        clf.fit(X_train_poly, train_pkg['y_tr'])
            
        for test_pkg in processed_datasets:
            X_eval_scaled = sign_log(test_pkg['X_te'])
            X_eval_poly = poly.transform(X_eval_scaled)
            
            y_pred = clf.predict(X_eval_poly)
            acc = accuracy_score(test_pkg['y_te'], y_pred)
            mcc = matthews_corrcoef(test_pkg['y_te'], y_pred)
            
            print(f"{train_pkg['name']:<15} | {test_pkg['name']:<15} | {acc:<10.4f} | {mcc:<10.4f}")

if __name__ == "__main__":
    run_comparison()