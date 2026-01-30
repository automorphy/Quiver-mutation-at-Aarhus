import numpy as np
import sys
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, matthews_corrcoef

# Add repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lib.svm_utils import load_canonical_class_0

def sign_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

def add_cycle_features(X):
    abs_X = np.abs(X)
    def cycle_feat(w1, w2, w3):
        return np.sign(w1 - (w2 * w3))

    # X columns: 0:b01, 1:b02, 2:b03, 3:b12, 4:b13, 5:b23
    w_03 = abs_X[:, 2]
    w_13 = abs_X[:, 4]
    w_01 = abs_X[:, 0]
    
    # Cycle 1 (0,3,1) features
    c1_1 = cycle_feat(w_03, w_13, w_01)
    c1_2 = cycle_feat(w_13, w_01, w_03)
    c1_3 = cycle_feat(w_01, w_03, w_13)
    
    w_23 = abs_X[:, 5]
    w_02 = abs_X[:, 1]
    
    # Cycle 2 (0,3,2) features
    c2_1 = cycle_feat(w_03, w_23, w_02)
    c2_2 = cycle_feat(w_23, w_02, w_03)
    c2_3 = cycle_feat(w_02, w_03, w_23)
    
    new_feats = np.column_stack([c1_1, c1_2, c1_3, c2_1, c2_2, c2_3])
    return new_feats

def run_extraction():
    X_raw, y = load_canonical_class_0(limit=5000)
    if X_raw is None: return

    # 1. Prepare Base Features
    X_log = sign_log(X_raw)
    X_cycles = add_cycle_features(X_raw)
    
    # Combine
    X_combined = np.hstack([X_log, X_cycles])
    
    # Feature Names
    base_names = ['b01', 'b02', 'b03', 'b12', 'b13', 'b23']
    cycle_names = ['C1_03', 'C1_13', 'C1_01', 'C2_03', 'C2_23', 'C2_02']
    all_names = base_names + cycle_names
    
    print(f"Total Input Features: {len(all_names)}")
    
    # 2. Polynomial Expansion (Degree 4)
    poly = PolynomialFeatures(degree=4, include_bias=True)
    X_poly = poly.fit_transform(X_combined)
    poly_names = poly.get_feature_names_out(all_names)
    
    print(f"Expanded Features: {X_poly.shape[1]}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    
    # 3. Train L1 Linear SVM
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    class_weight = {0: 1, 1: n_neg/n_pos if n_pos > 0 else 1}
    
    print("Training L1 Linear SVM...")
    clf = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=0.01, class_weight=class_weight, max_iter=5000, random_state=42)
    clf.fit(X_train, y_train)
    
    # 4. Results
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"\nAccuracy: {acc:.4f}")
    print(f"MCC:      {mcc:.4f}")
    
    # 5. Extract Top Features
    coefs = clf.coef_[0]
    n_nonzero = np.sum(coefs != 0)
    print(f"Non-Zero Terms: {n_nonzero}")
    
    indices = np.argsort(np.abs(coefs))[::-1]
    
    print(f"\nTop 20 Terms:")
    for i in range(20):
        idx = indices[i]
        if coefs[idx] == 0: break
        print(f"{i+1:<3} | {coefs[idx]:<10.4f} | {poly_names[idx]}")

if __name__ == "__main__":
    run_extraction()