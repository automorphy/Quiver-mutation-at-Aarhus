import numpy as np
import sys
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report

# Add repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lib.svm_utils import load_canonical_class_0

def sign_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

def train_svm():
    X, y = load_canonical_class_0(limit=20000) # Use a decent chunk
    
    # Log Scaling
    print("Applying Log Scaling...")
    X_scaled = sign_log(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    
    # Degree 4 Polynomial Kernel (Implicit via SVC 'poly')
    # Since we established Aut(G) is trivial, we don't need a custom invariant kernel.
    # Standard poly kernel is fine.
    
    # Class weights
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    print(f"Class Balance: MA(1): {n_pos}, NMA(0): {n_neg}")
    
    class_weight = {0: 1, 1: n_neg/n_pos if n_pos > 0 else 1}
    
    print("Training SVM (Poly Degree 4, C=1.0)...")
    clf = SVC(kernel='poly', degree=4, gamma='scale', coef0=1.0, C=1.0, class_weight=class_weight)
    clf.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print("\n--- Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"MCC:      {mcc:.4f}")
    print(f"Support Vectors: {clf.n_support_.sum()}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_svm()