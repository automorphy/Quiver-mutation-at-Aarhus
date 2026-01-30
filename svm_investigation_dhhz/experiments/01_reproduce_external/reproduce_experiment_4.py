import numpy as np
from sklearn import svm
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

def run_experiment():
    ma_file = get_external_data_path("Experiment_4", "Fourth_Experiment_MA_data.txt")
    nma_file = get_external_data_path("Experiment_4", "Fourth_Experiment_NMA_data.txt")

    print("Loading data...")
    nma_data_raw = data_reading_exchange_matrix(nma_file)
    ma_data_raw = data_reading_exchange_matrix(ma_file)

    print(f"NMA samples: {len(nma_data_raw)}")
    print(f"MA samples: {len(ma_data_raw)}")

    nma_data = components_finder_6(nma_data_raw)
    ma_data = components_finder_6(ma_data_raw)

    X = np.array(ma_data + nma_data)
    y = np.array([-1] * len(ma_data) + [1] * len(nma_data))

    print(f"Total samples: {len(X)}")

    # Shuffle as in their script
    seed_random = 30
    np.random.seed(seed_random)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    proportion = float(len(ma_data)/len(nma_data))
    print(f"Class weight proportion (MA/NMA): {proportion:.2f}")

    degrees = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    
    print("\nStarting SVM training (Full Dataset - No Train/Test Split as per original script)...")
    print(f"{'Degree':<8} | {'MCC':<10} | {'Accuracy':<10} | {'Support Vectors':<15}")
    print("-" * 55)

    for d in degrees:
        clf = svm.SVC(kernel='poly', C=1, degree=d, class_weight={-1: 1, 1: proportion})
        clf.fit(X, y)
        
        y_pred = clf.predict(X)
        mcc = matthews_corrcoef(y, y_pred)
        acc = accuracy_score(y, y_pred)
        n_sv = clf.n_support_.sum()
        
        print(f"{d:<8} | {mcc:<10.4f} | {acc:<10.4f} | {n_sv:<15}")

if __name__ == "__main__":
    run_experiment()