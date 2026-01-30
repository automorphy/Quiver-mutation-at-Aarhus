import numpy as np
import os
import sys
import sqlite3
import pandas as pd

# Add repo root to sys.path to allow importing lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.config import get_external_data_path, get_db_path

# --- Loading Functions ---

def load_exp3_data():
    ma_file = get_external_data_path("Experiment_3", "MA_ALL_DEPTH_7_type_0.txt")
    nma_file = get_external_data_path("Experiment_3", "NMA_ALL_Depth_5__type_1.txt")
    
    def read_matrix(path):
        data = []
        if not os.path.exists(path): return []
        with open(path, 'r') as f:
            for line in f:
                try:
                    row = [float(x) for x in line.strip().split(',')]
                    if len(row) == 16:
                        # indices: 1, 2, 3, 6, 7, 11
                        data.append([row[1], row[2], row[3], row[6], row[7], row[11]])
                except: continue
        return data

    ma_X = read_matrix(ma_file)
    nma_X = read_matrix(nma_file)
    
    X = np.array(ma_X + nma_X)
    # Exp 3 labels: 0 for MA, 1 for NMA (based on filenames/context)
    # But wait, let's normalize to our convention: 1 for MA, 0 for NMA (or -1 for NMA)
    # Our DB: 1=MA, 0=NMA. 
    # Exp 4: -1=MA, 1=NMA (wait, need to check)
    
    # In RAW_SVM.py (Exp 4):
    # NON_NMA_DATA -> not_nma_class = -1
    # NMA_DATA -> nma_class = 1
    
    # In Exp 3:
    # MA...type_0 -> 0
    # NMA...type_1 -> 1
    
    # Let's standardize to: MA=1, NMA=0 for this analysis output
    y = np.array([1] * len(ma_X) + [0] * len(nma_X))
    return X, y, "Experiment 3"

def load_exp4_data():
    ma_file = get_external_data_path("Experiment_4", "Fourth_Experiment_MA_data.txt")
    nma_file = get_external_data_path("Experiment_4", "Fourth_Experiment_NMA_data.txt")
    
    def read_matrix(path):
        data = []
        if not os.path.exists(path): return []
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
    return X, y, "Experiment 4"

def load_our_data(limit=None):
    db_path = get_db_path("quivers_rank4_250k.db")
    if not os.path.exists(db_path):
        return np.array([]), np.array([]), "Our Data (Not Found)"
        
    conn = sqlite3.connect(db_path)
    query = "SELECT b12, b13, b14, b23, b24, b34, label FROM quivers"
    if limit:
        query += f" LIMIT {limit}"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    X = df[['b12', 'b13', 'b14', 'b23', 'b24', 'b34']].values
    y = df['label'].values # Already 1=MA, 0=NMA (check main.rs: label = if stype == "acyclic" { 1 } else { 0 })
    
    return X, y, "Our Data (250k)"

# --- Analysis ---

def analyze_dataset(X, y, name):
    print(f"\n--- Analysis: {name} ---")
    if len(X) == 0:
        print("No data found.")
        return

    print(f"Total Samples: {len(X)}")
    
    # Class Balance
    n_ma = np.sum(y == 1)
    n_nma = np.sum(y == 0)
    print(f"MA (1): {n_ma} ({n_ma/len(y)*100:.1f}%)")
    print(f"NMA (0): {n_nma} ({n_nma/len(y)*100:.1f}%)")
    
    # Feature Stats
    # Flatten X to see range of weights
    weights = X.flatten()
    print(f"Weight Range: [{np.min(weights)}, {np.max(weights)}]")
    print(f"Weight Mean: {np.mean(np.abs(weights)):.2f}")
    print(f"Weight Std: {np.std(weights):.2f}")
    
    # Unique Vectors
    # Convert to tuple for hashing
    unique_vectors = set(tuple(x) for x in X)
    print(f"Unique Vectors: {len(unique_vectors)} (Duplication rate: {100*(1 - len(unique_vectors)/len(X)):.1f}%)")

if __name__ == "__main__":
    X3, y3, name3 = load_exp3_data()
    analyze_dataset(X3, y3, name3)
    
    X4, y4, name4 = load_exp4_data()
    analyze_dataset(X4, y4, name4)
    
    X_our, y_our, name_our = load_our_data()
    analyze_dataset(X_our, y_our, name_our)