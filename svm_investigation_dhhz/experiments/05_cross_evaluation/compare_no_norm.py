import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

# Add repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lib.config import get_external_data_path
from lib.svm_utils import train_custom_svm

# Reuse data loading from previous scripts
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

    print("Loading Experiment 4 data...")
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

def run_comparison():
    X, y, proportion = load_exp4_data()
    # Use a smaller subset for faster execution
    X_train, X_test, y_train, y_test = train_test_split(X[:2000], y[:2000], test_size=0.2, random_state=42)
    
    print("\nComparing Normalization vs No-Normalization (Custom Int Kernel)")
    print(f"Data Split: {len(X_train)} Train, {len(X_test)} Test")
    
    degrees = [4]
    class_weight = {-1: 1, 1: proportion}
    
    print(f"\n{'Deg':<4} | {'Norm?':<6} | {'Test MCC':<10} | {'Test Acc':<10} | {'SVs':<8}")
    print("-" * 55)

    for d in degrees:
        # With Normalization
        metrics_norm = train_custom_svm(
            X_train, y_train, X_test, y_test, 
            degree=d, gamma=1, coef0=1, normalize=True, class_weight=class_weight
        )
        print(f"{d:<4} | {'Yes':<6} | {metrics_norm['test_mcc']:<10.4f} | {metrics_norm['test_acc']:<10.4f} | {metrics_norm['support_vectors']:<8}")
        
        # Without Normalization
        metrics_no_norm = train_custom_svm(
            X_train, y_train, X_test, y_test, 
            degree=d, gamma=1, coef0=1, normalize=False, class_weight=class_weight
        )
        print(f"{d:<4} | {'No':<6} | {metrics_no_norm['test_mcc']:<10.4f} | {metrics_no_norm['test_acc']:<10.4f} | {metrics_no_norm['support_vectors']:<8}")
        print("-" * 55)

if __name__ == "__main__":
    run_comparison()