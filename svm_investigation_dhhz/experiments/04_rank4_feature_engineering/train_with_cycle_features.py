import numpy as np
import sys
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report

# Add repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lib.svm_utils import load_canonical_class_0

def sign_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

def add_cycle_features(X):
    # X columns: 0:b01, 1:b02, 2:b03, 3:b12, 4:b13, 5:b23
    
    # Cycle 1: (0,3,1) -> Edges: (0,3), (3,1), (1,0)
    # Indices: 2, 4, 0
    # Weights are in X[:, idx]. We need absolute values.
    
    abs_X = np.abs(X)
    
    # Helper for cycle feature: sign(|w1| - |w2*w3|)
    def cycle_feat(w1, w2, w3):
        return np.sign(w1 - (w2 * w3))

    # Cycle 1 Edges: e1=b03(2), e2=b13(4), e3=b01(0)
    w_03 = abs_X[:, 2]
    w_13 = abs_X[:, 4]
    w_01 = abs_X[:, 0]
    
    c1_f1 = cycle_feat(w_03, w_13, w_01)
    c1_f2 = cycle_feat(w_13, w_01, w_03)
    c1_f3 = cycle_feat(w_01, w_03, w_13)
    
    # Cycle 2: (0,3,2) -> Edges: (0,3), (3,2), (2,0)
    # Indices: 2, 5, 1
    w_03 = abs_X[:, 2] # Same edge
    w_23 = abs_X[:, 5]
    w_02 = abs_X[:, 1]
    
    c2_f1 = cycle_feat(w_03, w_23, w_02)
    c2_f2 = cycle_feat(w_23, w_02, w_03)
    c2_f3 = cycle_feat(w_02, w_03, w_23)
    
    new_feats = np.column_stack([c1_f1, c1_f2, c1_f3, c2_f1, c2_f2, c2_f3])
    
    return np.hstack([X, new_feats])

def train_svm():
    X_raw, y = load_canonical_class_0(limit=20000)
    if X_raw is None: return

    print(f"Original Features: {X_raw.shape[1]}")
    
    # Log scale original features
    X_log = sign_log(X_raw)
    
    # Add cycle features (computed on RAW weights, but output is {-1,0,1}, so no scaling needed)
    X_augmented = add_cycle_features(X_raw)
    # We essentially append the 6 signs to the unscaled X, but we want X_log for the main features
    # So:
    X_final = np.hstack([X_log, X_augmented[:, 6:]])
    
    print(f"Augmented Features: {X_final.shape[1]}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
    
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    class_weight = {0: 1, 1: n_neg/n_pos if n_pos > 0 else 1}
    
    print("Training SVM (Poly Degree 4, C=1.0)...")
    clf = SVC(kernel='poly', degree=4, gamma='scale', coef0=1.0, C=1.0, class_weight=class_weight)
    clf.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"\nAccuracy: {acc:.4f}")
    print(f"MCC:      {mcc:.4f}")
    print(classification_report(y_test, y_pred))

    # --- Boundary Analysis ---
    from lib.svm_utils import analyze_boundary_distances
    analyze_boundary_distances(clf, X_train, y_train, X_test, y_test, class_labels=(0, 1))

if __name__ == "__main__":
    train_svm()

if __name__ == "__main__":
    train_svm()