import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, matthews_corrcoef

# Add repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lib.config import get_external_data_path
from lib.svm_utils import compute_exact_kernel

def data_reading_exchange_matrix(file_name_exchange):
    data = []
    if not os.path.exists(file_name_exchange):
        print(f"File not found: {file_name_exchange}")
        return []
    with open(file_name_exchange, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            string_list = line.split(',')
            float_list = [float(k) for k in string_list]
            data.append(float_list)
    return data

def components_finder_6(lists):
    output = []
    for i in lists:
        if len(i) > 11:
            outs = [i[1], i[2], i[3], i[6], i[7], i[11]]
            output.append(outs)
    return output

def load_exp4_data():
    ma_file = get_external_data_path("Experiment_4", "Fourth_Experiment_MA_data.txt")
    nma_file = get_external_data_path("Experiment_4", "Fourth_Experiment_NMA_data.txt")
    nma_data_raw = data_reading_exchange_matrix(nma_file)
    ma_data_raw = data_reading_exchange_matrix(ma_file)
    nma_data = components_finder_6(nma_data_raw)
    ma_data = components_finder_6(ma_data_raw)
    X = np.array(ma_data + nma_data)
    y = np.array([-1] * len(ma_data) + [1] * len(nma_data))
    seed_random = 30
    np.random.seed(seed_random)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    proportion = float(len(ma_data)/len(nma_data))
    return X, y, proportion

def run_regularization_sweep():
    X, y, proportion = load_exp4_data()
    # Use 2000 samples for reasonable execution time
    X_train, X_test, y_train, y_test = train_test_split(X[:2000], y[:2000], test_size=0.2, random_state=42)
    
    print("\nRegularization Sweep (C parameter)")
    print(f"Data Split: {len(X_train)} Train, {len(X_test)} Test")
    
    degree = 4
    class_weight = {-1: 1, 1: proportion}
    
    # Precompute Kernels once
    K_train = compute_exact_kernel(X_train, X_train, degree=degree, gamma=1, coef0=1).astype(np.float64)
    K_test = compute_exact_kernel(X_test, X_train, degree=degree, gamma=1, coef0=1).astype(np.float64)

    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    print(f"\n{'C':<8} | {'Test MCC':<10} | {'Test Acc':<10} | {'SVs':<8}")
    print("-" * 55)

    for c in c_values:
        svm_model = SVC(kernel='precomputed', C=c, class_weight=class_weight)
        svm_model.fit(K_train, y_train)
        
        y_pred_test = svm_model.predict(K_test)
        mcc = matthews_corrcoef(y_test, y_pred_test)
        acc = accuracy_score(y_test, y_pred_test)
        n_sv = svm_model.n_support_.sum()
        
        print(f"{c:<8} | {mcc:<10.4f} | {acc:<10.4f} | {n_sv:<8}")

if __name__ == "__main__":
    run_regularization_sweep()