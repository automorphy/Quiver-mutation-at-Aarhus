import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef
import time

def generate_rank3_data(n_samples=5000, val_range=10):
    """
    Generates Rank 3 quiver data (b12, b13, b23) and labels.
    Label is 1 (MA/Acyclic) if Markov Constant > 4 or min(|weight|) < 2.
    Otherwise 0 (NMA/Cyclic).
    """
    # Generate random signed weights
    X = np.random.randint(-val_range, val_range + 1, size=(n_samples, 3))
    
    y = []
    for row in X:
        x, y_val, z = abs(row[0]), abs(row[1]), abs(row[2])
        c = x**2 + y_val**2 + z**2 - x*y_val*z
        min_v = min(x, y_val, z)
        is_acyclic = (c > 4) or (min_v < 2)
        y.append(1 if is_acyclic else 0)
        
    return X, np.array(y)

# --- Custom Invariant Kernel Logic ---

def cyclic_invariant_kernel(X, Y, degree=3, gamma=1.0, coef0=1.0):
    """
    Computes K_sym(x, y) = sum_{sigma in C3} (gamma * <x, sigma(y)> + coef0)^degree
    """
    # Base dot product
    # We need to compute <x, y>, <x, rot(y)>, <x, rot2(y)>
    
    # Y permutations
    Y_rot1 = np.roll(Y, 1, axis=1) # (z, x, y)
    Y_rot2 = np.roll(Y, 2, axis=1) # (y, z, x)
    
    K = np.zeros((X.shape[0], Y.shape[0]))
    
    for Y_perm in [Y, Y_rot1, Y_rot2]:
        # Compute standard poly kernel for this permutation
        # (gamma <x, y_perm> + coef0)^d
        dot = np.dot(X, Y_perm.T)
        k_part = (gamma * dot + coef0) ** degree
        K += k_part
        
    return K

def run_experiment():
    print("--- Rank 3 Markov Polynomial Experiment (Symmetry & Invariance) ---")
    
    X, y = generate_rank3_data(n_samples=5000, val_range=15)
    print(f"Data Generated: {len(X)} samples")
    print(f"Class Balance: MA (1): {np.sum(y==1)}, NMA (0): {np.sum(y==0)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Baseline: Canonical Absolute Data
    print("\n[Model A] Baseline: Absolute Data (|x|,|y|,|z|)")
    X_train_abs = np.abs(X_train)
    X_test_abs = np.abs(X_test)
    
    clf_base = SVC(kernel='poly', degree=3, coef0=1, C=1.0)
    clf_base.fit(X_train_abs, y_train)
    mcc_base = matthews_corrcoef(y_test, clf_base.predict(X_test_abs))
    print(f"MCC: {mcc_base:.4f}, SVs: {len(clf_base.support_)}")

    # 2. Invariant Features (Elementary Symmetric Polynomials)
    # e1 = x+y+z, e2 = xy+yz+zx, e3 = xyz
    print("\n[Model B] Invariant Features (e1, e2, e3)")
    def get_sym_features(X_in):
        X_a = np.abs(X_in)
        x, y_v, z = X_a[:,0], X_a[:,1], X_a[:,2]
        e1 = x + y_v + z
        e2 = x*y_v + y_v*z + z*x
        e3 = x*y_v*z
        return np.column_stack([e1, e2, e3])

    X_train_sym = get_sym_features(X_train)
    X_test_sym = get_sym_features(X_test)
    
    # Use Linear SVM here because the invariant IS a linear combo of e1^2, e2, e3
    # Wait, Markov is x^2+y^2+z^2 - xyz = e1^2 - 2e2 - e3.
    # So we need Degree 2 on e1, or Degree 1 if we feed x^2+y^2+z^2 directly.
    # Let's use Poly Degree 2 on (e1, e2, e3) to be safe.
    clf_feat = SVC(kernel='poly', degree=2, coef0=1, C=1.0)
    clf_feat.fit(X_train_sym, y_train)
    mcc_feat = matthews_corrcoef(y_test, clf_feat.predict(X_test_sym))
    print(f"MCC: {mcc_feat:.4f}, SVs: {len(clf_feat.support_)}")

    # 3. Invariant Kernel (Reynolds Operator)
    print("\n[Model C] Invariant Kernel (Sum over C3 orbit)")
    # We pass Absolute data to this kernel, assuming the user has already canonicalized signs
    # This kernel handles the cyclic permutation invariance.
    
    K_train = cyclic_invariant_kernel(X_train_abs, X_train_abs, degree=3)
    K_test = cyclic_invariant_kernel(X_test_abs, X_train_abs, degree=3)
    
    clf_kern = SVC(kernel='precomputed', C=1.0)
    clf_kern.fit(K_train, y_train)
    mcc_kern = matthews_corrcoef(y_test, clf_kern.predict(K_test))
    print(f"MCC: {mcc_kern:.4f}, SVs: {len(clf_kern.support_)}")

if __name__ == "__main__":
    run_experiment()
