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

def extract_features_degree(degree, X_train, y_train, X_test, y_test):
    print(f"\n--- Degree {degree} ---")
    print(f"Generating Polynomial Features...")
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    
    # Fit transform on smaller subset if degree is high to avoid OOM
    if degree >= 6:
        # Use subset for feature gen
        X_tr_poly = poly.fit_transform(X_train)
        X_te_poly = poly.transform(X_test)
    else:
        X_tr_poly = poly.fit_transform(X_train)
        X_te_poly = poly.transform(X_test)
        
    feature_names = poly.get_feature_names_out(['b12', 'b13', 'b14', 'b23', 'b24', 'b34'])
    print(f"Num Features: {X_tr_poly.shape[1]}")
    
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    class_weight = {0: 1, 1: n_neg/n_pos if n_pos > 0 else 1}
    
    print("Training L1 Linear SVM...")
    # Increase C slightly for higher degrees to allow more complex fit? Or decrease for sparsity?
    # Keep C=0.01 as baseline
    clf = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=0.01, class_weight=class_weight, max_iter=5000, random_state=42)
    clf.fit(X_tr_poly, y_train)
    
    y_pred = clf.predict(X_te_poly)
    mcc = matthews_corrcoef(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    coefs = clf.coef_[0]
    n_nonzero = np.sum(coefs != 0)
    
    print(f"Results -> Accuracy: {acc:.4f}, MCC: {mcc:.4f}")
    print(f"Non-Zero Terms: {n_nonzero}")
    
    indices = np.argsort(np.abs(coefs))[::-1]
    
    print(f"Top 10 Terms:")
    for i in range(10):
        idx = indices[i]
        if coefs[idx] == 0: break
        print(f"{i+1:<3} | {coefs[idx]:<10.4f} | {feature_names[idx]}")

def run_all():
    X, y = load_canonical_class_0(limit=3000) # Keep small for high degree expansion
    if X is None: return

    print("Applying Log Scaling...")
    X_scaled = sign_log(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    for d in [4, 6, 8]:
        extract_features_degree(d, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    run_all()