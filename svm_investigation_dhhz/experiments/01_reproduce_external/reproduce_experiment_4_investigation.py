import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, accuracy_score
import os
import sys

# Add repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lib.config import get_external_data_path

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
            # indices: 1,2,3, 6,7, 11
            outs = [i[1], i[2], i[3], i[6], i[7], i[11]]
            output.append(outs)
    return output

def run_investigation():
    ma_file = get_external_data_path("Experiment_4", "Fourth_Experiment_MA_data.txt")
    nma_file = get_external_data_path("Experiment_4", "Fourth_Experiment_NMA_data.txt")

    print("Loading Experiment 4 data...")
    nma_data_raw = data_reading_exchange_matrix(nma_file)
    ma_data_raw = data_reading_exchange_matrix(ma_file)
    
    nma_data = components_finder_6(nma_data_raw)
    ma_data = components_finder_6(ma_data_raw)

    X = np.array(ma_data + nma_data)
    y = np.array([-1] * len(ma_data) + [1] * len(nma_data))

    # Shuffle
    seed_random = 30
    np.random.seed(seed_random)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    proportion = float(len(ma_data)/len(nma_data))
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data Split: {len(X_train)} Train, {len(X_test)} Test")
    print(f"Class weight: {proportion:.2f}")

    degrees = [4, 6, 8]
    
    print(f"\n{'Deg':<4} | {'Coef0':<5} | {'Set':<5} | {'MCC':<8} | {'Acc':<8}")
    print("-" * 45)

    for d in degrees:
        for coef0 in [0.0, 1.0]:
            # Train
            clf = svm.SVC(kernel='poly', C=1, degree=d, gamma='scale', coef0=coef0, class_weight={-1: 1, 1: proportion})
            clf.fit(X_train, y_train)
            
            # Eval Train
            y_pred_train = clf.predict(X_train)
            mcc_train = matthews_corrcoef(y_train, y_pred_train)
            acc_train = accuracy_score(y_train, y_pred_train)
            
            print(f"{d:<4} | {coef0:<5} | Train | {mcc_train:<8.4f} | {acc_train:<8.4f}")

            # Eval Test
            y_pred_test = clf.predict(X_test)
            mcc_test = matthews_corrcoef(y_test, y_pred_test)
            acc_test = accuracy_score(y_test, y_pred_test)
            
            print(f"{d:<4} | {coef0:<5} | Test  | {mcc_test:<8.4f} | {acc_test:<8.4f}")
            print("-" * 45)

if __name__ == "__main__":
    run_investigation()