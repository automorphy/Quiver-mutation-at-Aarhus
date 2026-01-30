import numpy as np
import sys
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, matthews_corrcoef

# Add repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lib.svm_utils import load_canonical_class_0, analyze_boundary_distances

def sign_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

def get_cycle_bases(X):
    abs_X = np.abs(X)
    # Cycle 1 (0,3,1): Edges (0,3), (3,1), (1,0) -> Indices 2, 4, 0
    w1_03, w1_13, w1_01 = abs_X[:, 2], abs_X[:, 4], abs_X[:, 0]
    
    # Cycle 2 (0,3,2): Edges (0,3), (3,2), (2,0) -> Indices 2, 5, 1
    w2_03, w2_23, w2_02 = abs_X[:, 2], abs_X[:, 5], abs_X[:, 1]
    
    return [(w1_03, w1_13, w1_01), (w2_03, w2_23, w2_02)]

def make_sign_features(cycles):
    feats = []
    for (w1, w2, w3) in cycles:
        feats.append(np.sign(w1 - w2*w3))
        feats.append(np.sign(w2 - w3*w1))
        feats.append(np.sign(w3 - w1*w2))
    return np.column_stack(feats)

def make_abs_diff_features(cycles):
    # | |w1| - |w2w3| |
    # Note: weights passed here are already absolute values from get_cycle_bases
    feats = []
    for (w1, w2, w3) in cycles:
        feats.append(np.abs(w1 - w2*w3))
        feats.append(np.abs(w2 - w3*w1))
        feats.append(np.abs(w3 - w1*w2))
    # Log scale these differences because they can be huge
    return sign_log(np.column_stack(feats))

def run_comparison():
    X_raw, y = load_canonical_class_0(limit=3000)
    if X_raw is None: return

    # Base Features
    X_base = sign_log(X_raw)
    cycles = get_cycle_bases(X_raw)
    
    # Feature Sets
    sets = {
        "Base Only": X_base,
        "Base + Sign": np.hstack([X_base, make_sign_features(cycles)]),
        "Base + AbsDiff": np.hstack([X_base, make_abs_diff_features(cycles)]),
        "Base + Sign + AbsDiff": np.hstack([X_base, make_sign_features(cycles), make_abs_diff_features(cycles)])
    }
    
    print("\nComparing Feature Sets (L1 SVM, Poly Degree 4, C=0.01)")
    print(f"{'Feature Set':<25} | {'Input Dim':<10} | {'Poly Dim':<10} | {'Acc':<8} | {'MCC':<8}")
    print("-" * 75)
    
    # Fixed Split
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
    
    for name, X in sets.items():
        poly = PolynomialFeatures(degree=4, include_bias=True)
        # Use subset of training for poly fit to save memory if needed, but 3000 is small enough
        X_poly = poly.fit_transform(X)
        
        X_tr, X_te = X_poly[idx_train], X_poly[idx_test]
        y_tr, y_te = y[idx_train], y[idx_test]
        
        n_pos = np.sum(y_tr == 1)
        n_neg = np.sum(y_tr == 0)
        cw = {0: 1, 1: n_neg/n_pos if n_pos > 0 else 1}
        
        clf = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=0.01, class_weight=cw, max_iter=5000, random_state=42)
        clf.fit(X_tr, y_tr)
        
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        mcc = matthews_corrcoef(y_te, y_pred)
        
        print(f"{name:<25} | {X.shape[1]:<10} | {X_poly.shape[1]:<10} | {acc:<8.4f} | {mcc:<8.4f}")
        
        # Analyze boundary
        analyze_boundary_distances(clf, X_tr, y_tr, X_te, y_te, class_labels=(0, 1))
        print("-" * 75)

if __name__ == "__main__":
    run_comparison()