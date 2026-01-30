import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef
import math
from collections import defaultdict

# --- Simple Polynomial Class ---
class Polynomial:
    def __init__(self, terms=None):
        # Dictionary mapping tuple of powers (a,b,c) -> coefficient
        # Variables are x, y, z corresponding to indices 0, 1, 2
        self.terms = defaultdict(float)
        if terms:
            for k, v in terms.items():
                self.terms[k] = v
    
    def add(self, other):
        res = Polynomial(self.terms)
        for k, v in other.terms.items():
            res.terms[k] += v
        return res
        
    def multiply_scalar(self, scalar):
        res = Polynomial()
        for k, v in self.terms.items():
            res.terms[k] = v * scalar
        return res
    
    def power(self, p):
        # Naive power implementation for (poly)^p
        # Only needed for (gamma*dot + coef0)^d where base is linear
        # But here base is a polynomial? No, base is dot product + constant.
        pass

    def __repr__(self):
        sorted_terms = sorted(self.terms.items(), key=lambda x: abs(x[1]), reverse=True)
        s = []
        for (powers, coeff) in sorted_terms:
            if abs(coeff) < 0.1: continue
            term_s = []
            vars = ['x', 'y', 'z']
            for i, p in enumerate(powers):
                if p == 0: continue
                if p == 1: term_s.append(vars[i])
                else: term_s.append(f"{vars[i]}^{p}")
            t_str = "*".join(term_s) if term_s else "1"
            s.append(f"{coeff:.4f} * {t_str}")
        return "\n".join(s)

def expand_kernel_term(sv, degree, gamma, coef0):
    # (gamma * (sv . x) + coef0)^degree
    # sv . x = sv[0]*x + sv[1]*y + sv[2]*z
    # Base expression: gamma*sv[0]*x + gamma*sv[1]*y + gamma*sv[2]*z + coef0
    
    # We need to expand (c0*x + c1*y + c2*z + c3)^degree
    # This is a multinomial expansion.
    
    coeffs = [gamma*sv[0], gamma*sv[1], gamma*sv[2], coef0] # x, y, z, 1
    
    poly = Polynomial()
    
    # Iterate over all partitions of degree into 4 parts
    # i+j+k+l = degree
    from itertools import product
    
    for indices in product(range(degree + 1), repeat=4):
        if sum(indices) != degree: continue
        
        i, j, k, l = indices
        # Multinomial coeff: degree! / (i! j! k! l!)
        multinomial = math.factorial(degree) / (math.factorial(i) * math.factorial(j) * math.factorial(k) * math.factorial(l))
        
        term_coeff = multinomial * (coeffs[0]**i) * (coeffs[1]**j) * (coeffs[2]**k) * (coeffs[3]**l)
        
        # Power tuple for x, y, z
        powers = (i, j, k)
        poly.terms[powers] += term_coeff
        
    return poly

def generate_rank3_data(n_samples=5000, val_range=10):
    X = np.random.randint(-val_range, val_range + 1, size=(n_samples, 3))
    y = []
    for row in X:
        x, y_val, z = abs(row[0]), abs(row[1]), abs(row[2])
        c = x**2 + y_val**2 + z**2 - x*y_val*z
        min_v = min(x, y_val, z)
        is_acyclic = (c > 4) or (min_v < 2)
        y.append(1 if is_acyclic else -1)
    return np.abs(X), np.array(y)

def cyclic_invariant_kernel(X, Y, degree=3, gamma=1.0, coef0=1.0):
    K = np.zeros((X.shape[0], Y.shape[0]))
    perms = [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
    for p in perms:
        Y_perm = Y[:, p]
        dot = np.dot(X, Y_perm.T)
        K += (gamma * dot + coef0) ** degree
    return K

def extract_polynomial(clf, X_train, degree=3, gamma=1.0, coef0=1.0):
    print("\n--- Extracting Symbolic Polynomial ---")
    
    sv_indices = clf.support_
    sv_vecs = X_train[sv_indices]
    dual_coefs = clf.dual_coef_[0]
    intercept = clf.intercept_[0]
    
    print(f"Number of Support Vectors: {len(sv_vecs)}")
    
    final_poly = Polynomial()
    final_poly.terms[(0,0,0)] = intercept
    
    perms = [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
    
    for idx, sv in enumerate(sv_vecs):
        alpha = dual_coefs[idx]
        
        # Sum over permutations
        for p in perms:
            # Permute SV to match the kernel logic: <x, sigma(sv)>
            # Actually kernel was <x, sigma(y)>.
            # f(x) = sum alpha * K(sv, x) = sum alpha * sum_sigma (sv . sigma(x) + c)^d
            # (sv . sigma(x) + c)^d is equivalent to (sigma^-1(sv) . x + c)^d
            
            # Inverse permutations:
            # p=[0,1,2] -> inv=[0,1,2]
            # p=[2,0,1] (z,x,y) -> inv=[1,2,0] (y,z,x)
            # p=[1,2,0] (y,z,x) -> inv=[2,0,1] (z,x,y)
            
            # Actually easier: dot(sv, perm(x)) is same as dot(inv_perm(sv), x)
            # So we construct polynomial for inv_perm(sv) . x
            
            sv_perm = sv[p] # This applies p to sv indices? No, numpy indexing reorders.
            # sv[p] means [sv[p[0]], sv[p[1]]...]
            # If p=[2,0,1], sv_perm = [sv[2], sv[0], sv[1]] = (z, x, y)
            # dot((z,x,y), (x,y,z)) = zx + xy + yz
            # This matches dot(sv, (y,z,x))
            
            # Let's just use the fact that sum_sigma (sv . sigma(x)) is symmetric.
            # We expand (sv . sigma(x) + c)^d
            
            # Coefficients for x, y, z in the dot product:
            # sigma=[0,1,2] -> coeff = sv
            # sigma=[2,0,1] -> x pairs with sv[2], y with sv[0], z with sv[1]. Coeffs: [sv[2], sv[0], sv[1]]
            
            # Wait, dot(sv, sigma(x))
            # sigma(x) = [x[p[0]], x[p[1]], x[p[2]]]
            # dot = sv[0]*x[p[0]] + sv[1]*x[p[1]] + sv[2]*x[p[2]]
            
            # We need to map this back to coeffs of x[0], x[1], x[2].
            # term for x[k] has coeff sv[j] where p[j] == k.
            
            current_coeffs = [0.0, 0.0, 0.0]
            for j in range(3):
                target_var = p[j] # 0 for x, 1 for y, 2 for z
                current_coeffs[target_var] = sv[j]
            
            poly_term = expand_kernel_term(current_coeffs, degree, gamma, coef0)
            weighted_term = poly_term.multiply_scalar(alpha)
            final_poly = final_poly.add(weighted_term)
            
    print("\nLearned Polynomial (Rounded Coefficients):")
    print(final_poly)

def run_experiment():
    X, y = generate_rank3_data(n_samples=2000, val_range=8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Invariant Kernel SVM...")
    K_train = cyclic_invariant_kernel(X_train, X_train, degree=3)
    clf = SVC(kernel='precomputed', C=10.0)
    clf.fit(K_train, y_train)
    
    K_test = cyclic_invariant_kernel(X_test, X_train, degree=3)
    y_pred = clf.predict(K_test)
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"Test MCC: {mcc:.4f}")
    
    extract_polynomial(clf, X_train, degree=3)

if __name__ == "__main__":
    run_experiment()
