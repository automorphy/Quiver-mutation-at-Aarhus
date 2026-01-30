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

def get_cycle_bases(X):
    abs_X = np.abs(X)
    w1_03, w1_13, w1_01 = abs_X[:, 2], abs_X[:, 4], abs_X[:, 0]
    w2_03, w2_23, w2_02 = abs_X[:, 2], abs_X[:, 5], abs_X[:, 1]
    return [(w1_03, w1_13, w1_01), (w2_03, w2_23, w2_02)]

def make_abs_diff_features(cycles):
    feats = []
    for (w1, w2, w3) in cycles:
        feats.append(np.abs(w1 - w2*w3))
        feats.append(np.abs(w2 - w3*w1))
        feats.append(np.abs(w3 - w1*w2))
    return sign_log(np.column_stack(feats))

def run_extraction():
    X_raw, y = load_canonical_class_0(limit=5000)
    if X_raw is None: return

    # Prepare Features: Base + AbsDiff
    X_base = sign_log(X_raw)
    cycles = get_cycle_bases(X_raw)
    X_abs_diff = make_abs_diff_features(cycles)
    X_combined = np.hstack([X_base, X_abs_diff])
    
    # Feature Names
    base_names = ['b01', 'b02', 'b03', 'b12', 'b13', 'b23']
    # AbsDiff names: | |w1| - |w2w3| |
    # Cycle 1: (0,3,1) -> Edges 03, 13, 01
    cycle_names = [
        '|b03-b13b01|', '|b13-b01b03|', '|b01-b03b13|', # Cycle 1
        '|b03-b23b02|', '|b23-b02b03|', '|b02-b03b23|'  # Cycle 2
    ]
    all_names = base_names + cycle_names
    
    # Polynomial Expansion
    poly = PolynomialFeatures(degree=4, include_bias=True)
    X_poly = poly.fit_transform(X_combined)
    poly_names = poly.get_feature_names_out(all_names)
    
    # Train
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    cw = {0: 1, 1: n_neg/n_pos if n_pos > 0 else 1}
    
    print("Training L1 Linear SVM (Base + AbsDiff)...")
    clf = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=0.01, class_weight=cw, max_iter=5000, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}, MCC: {mcc:.4f}")
    
    coefs = clf.coef_[0]
    indices = np.argsort(np.abs(coefs))[::-1]
    
    print(f"\nTop 20 Terms:")
    for i in range(20):
        idx = indices[i]
        if coefs[idx] == 0: break
        print(f"{i+1:<3} | {coefs[idx]:<10.4f} | {poly_names[idx]}")

if __name__ == "__main__":
    run_extraction()