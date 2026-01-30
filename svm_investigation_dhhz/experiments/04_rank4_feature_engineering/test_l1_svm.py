import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, matthews_corrcoef

# Add repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lib.config import get_external_data_path
from lib.svm_utils import analyze_boundary_distances

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

def run_l1_svm():
    X, y, proportion = load_exp4_data()
    # Use 2000 samples subset for speed
    X_train, X_test, y_train, y_test = train_test_split(X[:2000], y[:2000], test_size=0.2, random_state=42)
    
    print("\nL1-Regularized Linear SVM on Polynomial Features (Degree 4)")
    print(f"Data Split: {len(X_train)} Train, {len(X_test)} Test")
    
    degree = 4
    print(f"Generating polynomial features (degree={degree})...")
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    n_features = X_train_poly.shape[1]
    print(f"Number of polynomial features: {n_features}")
    
    c_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    print(f"\n{'C':<8} | {'Test MCC':<10} | {'Test Acc':<10} | {'Non-Zero Coefs':<15}")
    print("-" * 55)

    for c in c_values:
        # LinearSVC with l1 penalty requires dual=False
        clf = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=c, class_weight={-1: 1, 1: proportion}, max_iter=10000, random_state=42)
        clf.fit(X_train_poly, y_train)
        
        y_pred = clf.predict(X_test_poly)
        mcc = matthews_corrcoef(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        n_nonzero = np.sum(clf.coef_ != 0)
        
        print(f"{c:<8} | {mcc:<10.4f} | {acc:<10.4f} | {n_nonzero:<15}")
        
        # New shared analysis
        analyze_boundary_distances(clf, X_train_poly, y_train, X_test_poly, y_test, class_labels=(-1, 1))
        print("-" * 55)

if __name__ == "__main__":
    run_l1_svm()