import numpy as np
import time
import sqlite3
import pandas as pd
import ast
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report
from lib.config import get_db_path

def compute_exact_kernel(X1, X2, degree=3, gamma=1, coef0=1):
    """
    Computes the polynomial kernel using arbitrary-precision integers.
    K(x, y) = (gamma * <x, y> + coef0)^degree
    """
    print(f"Computing exact kernel matrix ({len(X1)}x{len(X2)}) with degree={degree}...")
    start_time = time.time()
    
    # Convert to object type to hold Python integers (arbitrary precision)
    # Ensure inputs are integers first
    X1_obj = X1.astype(int).astype(object)
    X2_obj = X2.astype(int).astype(object)
    
    # Compute dot product
    K_raw = np.dot(X1_obj, X2_obj.T)
    
    # Apply polynomial function
    K_raw = K_raw * int(gamma) + int(coef0)
    K_raw = K_raw ** int(degree)
    
    print(f"Kernel computation took {time.time() - start_time:.2f}s")
    return K_raw

def compute_diag_kernel(X, degree=3, gamma=1, coef0=1):
    """Computes only the diagonal of the kernel matrix."""
    X_obj = X.astype(int).astype(object)
    dot_self = np.sum(X_obj * X_obj, axis=1)
    K_diag = (dot_self * int(gamma) + int(coef0)) ** int(degree)
    return K_diag

def normalize_kernel_cosine(K_train, K_test, diag_train, diag_test):
    """
    Normalizes the kernel matrix using cosine normalization:
    K'(x, y) = K(x, y) / sqrt(K(x, x) * K(y, y))
    """
    print("Normalizing kernel (cosine normalization)...")
    
    diag_train_float = diag_train.astype(np.float64)
    diag_test_float = diag_test.astype(np.float64)
    
    diag_train_float[diag_train_float == 0] = 1e-15
    diag_test_float[diag_test_float == 0] = 1e-15
    
    inv_sqrt_diag_train = 1.0 / np.sqrt(diag_train_float)
    inv_sqrt_diag_test = 1.0 / np.sqrt(diag_test_float)
    
    K_train_float = K_train.astype(np.float64)
    K_train_norm = K_train_float * inv_sqrt_diag_train[:, np.newaxis] * inv_sqrt_diag_train[np.newaxis, :]
    
    K_test_float = K_test.astype(np.float64)
    K_test_norm = K_test_float * inv_sqrt_diag_test[:, np.newaxis] * inv_sqrt_diag_train[np.newaxis, :]
    
    return K_train_norm, K_test_norm

def train_custom_svm(X_train, y_train, X_test, y_test, degree=3, gamma=1, coef0=1, normalize=True, class_weight=None):
    # Compute Kernels
    K_train = compute_exact_kernel(X_train, X_train, degree=degree, gamma=gamma, coef0=coef0)
    K_test = compute_exact_kernel(X_test, X_train, degree=degree, gamma=gamma, coef0=coef0)
    
    if normalize:
        diag_train = compute_diag_kernel(X_train, degree=degree, gamma=gamma, coef0=coef0)
        diag_test = compute_diag_kernel(X_test, degree=degree, gamma=gamma, coef0=coef0)
        K_train, K_test = normalize_kernel_cosine(K_train, K_test, diag_train, diag_test)
    else:
        print("Skipping cosine normalization...")
        # Convert to float for SVM
        K_train = K_train.astype(np.float64)
        K_test = K_test.astype(np.float64)

    print("Training SVM with precomputed kernel...")
    svm_model = SVC(kernel='precomputed', C=1.0, class_weight=class_weight)
    svm_model.fit(K_train, y_train)
    
    print("Evaluating...")
    y_pred_train = svm_model.predict(K_train)
    y_pred_test = svm_model.predict(K_test)
    
    metrics = {
        'train_acc': accuracy_score(y_train, y_pred_train),
        'train_mcc': matthews_corrcoef(y_train, y_pred_train),
        'test_acc': accuracy_score(y_test, y_pred_test),
        'test_mcc': matthews_corrcoef(y_test, y_pred_test),
        'support_vectors': svm_model.n_support_.sum()
    }
    
    return metrics

def load_canonical_class_0(limit=None):
    """Loads aligned weights for the largest isomorphism class from the database."""
    db_path = get_db_path("quivers_rank4_canonical.db")
    conn = sqlite3.connect(db_path)
    
    # Class 0 Key
    target_class = "[-1, -1, 1, -1, -1, -1]"
    
    print(f"Loading data for Class {target_class}...")
    
    query = "SELECT aligned_weights, label FROM quivers WHERE digraph_class = ?"
    if limit:
        query += f" LIMIT {limit}"
        
    df = pd.read_sql_query(query, conn, params=(target_class,))
    conn.close()
    
    if df.empty:
        print("No data found!")
        return None, None
        
    # Parse aligned weights
    X_raw = [ast.literal_eval(w) for w in df['aligned_weights']]
    X = np.array(X_raw)
    y = df['label'].values
    
    print(f"Loaded {len(X)} samples.")
    return X, y

def analyze_boundary_distances(model, X_train, y_train, X_test, y_test, class_labels=(0, 1)):
    """
    Analyzes the distance of data points from the SVM decision boundary.
    
    Metrics:
    1. Max SV Distance: The furthest distance of a support vector from the boundary (per class).
    2. Max Error Distance: The furthest distance of a misclassified point from the boundary (per class).
    3. Boundedness: Whether Max Error <= Max SV.
    4. Margin vs Range: Comparison of the Max Error Distance to the full span of the data 
       projected onto the normal vector (decision function range).
       
    Args:
        model: Trained sklearn estimator (must have decision_function or similar).
        X_train, y_train: Training data (to find SVs).
        X_test, y_test: Test data (to find Errors).
        class_labels: Tuple of (Negative Class Label, Positive Class Label). 
                      Default (0, 1). For Exp 4 data use (-1, 1).
    """
    neg_label, pos_label = class_labels
    
    print("\n--- Boundary Distance Analysis ---")
    
    # --- 1. Support Vector Analysis (Train Set) ---
    try:
        # Get decision function values for training data
        df_train = model.decision_function(X_train)
        
        # Handle LinearSVC vs SVC
        if hasattr(model, 'support_'):
            # Kernel SVC
            sv_indices = model.support_
            sv_labels = y_train[sv_indices]
            sv_dists = df_train[sv_indices]
        else:
            # LinearSVC (approximate SVs as points within margin or on wrong side)
            # For LinearSVC, standard margin is 1.0. 
            # We treat "Support Vectors" loosely as points that constrain the margin 
            # (usually |df| <= 1), but user asked for "furthest" SVs. 
            # In soft-margin SVM, SVs can be anywhere on the wrong side or inside the margin.
            # So we look at all points with y * f(x) < 1.0 (Margin violators)
            margin_violators = (y_train * df_train) < 1.0
            sv_labels = y_train[margin_violators]
            sv_dists = df_train[margin_violators]
            
        sv_neg_dists = sv_dists[sv_labels == neg_label]
        sv_pos_dists = sv_dists[sv_labels == pos_label]
        
        # Max absolute distance (how "deep" into the wrong side or margin do they go?)
        # Note: Large positive dist for Pos class is GOOD. Large negative for Neg is GOOD.
        # "Furthest SV" usually refers to the Slack Variable xi being large.
        # i.e. Points on the WRONG side of the margin.
        # For Pos Class: Smallest df value (most negative if misclassified).
        # For Neg Class: Largest df value (most positive if misclassified).
        # But the user logic in previous turn used np.max(np.abs(sv_dists)). 
        # Let's stick to the previous implementation's interpretation: 
        # "How far from 0 is the furthest SV?" (Whether it's correctly or incorrectly classified).
        # Actually, for Soft Margin, SVs are defined by alpha > 0. 
        # If alpha = C (bounded), they can be far away.
        # We will report the Max Absolute Value of the decision function for SVs.
        
        max_sv_dist_neg = np.max(np.abs(sv_neg_dists)) if len(sv_neg_dists) > 0 else 0
        max_sv_dist_pos = np.max(np.abs(sv_pos_dists)) if len(sv_pos_dists) > 0 else 0
        
        print(f"Max SV Distance (Class {neg_label}): {max_sv_dist_neg:.4f}")
        print(f"Max SV Distance (Class {pos_label}): {max_sv_dist_pos:.4f}")
        
    except Exception as e:
        print(f"Could not compute SV distances: {e}")
        max_sv_dist_neg, max_sv_dist_pos = 0, 0

    # --- 2. Error Analysis (Test Set) ---
    df_test = model.decision_function(X_test)
    y_pred = model.predict(X_test)
    error_mask = (y_test != y_pred)
    
    if np.sum(error_mask) > 0:
        # Class Neg Errors: True Neg, Pred Pos (False Positives). Decision function > 0.
        neg_errors_mask = error_mask & (y_test == neg_label)
        neg_error_dists = df_test[neg_errors_mask]
        max_error_dist_neg = np.max(np.abs(neg_error_dists)) if np.any(neg_errors_mask) else 0.0
        
        # Class Pos Errors: True Pos, Pred Neg (False Negatives). Decision function < 0.
        pos_errors_mask = error_mask & (y_test == pos_label)
        pos_error_dists = df_test[pos_errors_mask]
        max_error_dist_pos = np.max(np.abs(pos_error_dists)) if np.any(pos_errors_mask) else 0.0
        
        print(f"Max Error Distance (Class {neg_label} - FP): {max_error_dist_neg:.4f}")
        print(f"Max Error Distance (Class {pos_label} - FN): {max_error_dist_pos:.4f}")
        
        # Check bounds
        if max_sv_dist_neg > 0:
            status = "BOUNDED" if max_error_dist_neg <= max_sv_dist_neg else "NOT BOUNDED"
            print(f">> Class {neg_label} errors: {status} ({max_error_dist_neg:.4f} vs {max_sv_dist_neg:.4f})")
            
        if max_sv_dist_pos > 0:
            status = "BOUNDED" if max_error_dist_pos <= max_sv_dist_pos else "NOT BOUNDED"
            print(f">> Class {pos_label} errors: {status} ({max_error_dist_pos:.4f} vs {max_sv_dist_pos:.4f})")
    else:
        print("No errors in test set.")
        max_error_dist_neg, max_error_dist_pos = 0, 0

    # --- 3. Full Data Range Analysis (Perpendicular to Boundary) ---
    # We use the combined Train + Test decision function values to estimate the full span
    all_df = np.concatenate([df_train, df_test])
    min_df = np.min(all_df)
    max_df = np.max(all_df)
    full_range = max_df - min_df
    abs_range = np.max(np.abs(all_df)) # Max deviation from 0
    
    print(f"\n--- Range vs Boundary Analysis ---")
    print(f"Full Decision Function Range: [{min_df:.4f}, {max_df:.4f}] (Span: {full_range:.4f})")
    
    overall_max_error = max(max_error_dist_neg, max_error_dist_pos)
    
    # Ratio: How much of the "feature space width" is occupied by errors?
    if full_range > 0:
        error_ratio = overall_max_error / full_range
        # Also compare to the max absolute value (distance from origin in projection)
        # This tells us if errors are "near the middle" or "at the edges"
        # If errors are near 0, and data goes to +/- 100, then ratio is small.
        print(f"Max Error / Full Range Ratio: {error_ratio:.4%}")
        
    print(f"Max Error is {overall_max_error:.4f} units from boundary.")
    print(f"Data extends up to {abs_range:.4f} units from boundary.")
    
    norm_depth_neg = 0.0
    norm_depth_pos = 0.0
    
    if abs_range > 0:
        norm_depth_neg = max_error_dist_neg / abs_range
        norm_depth_pos = max_error_dist_pos / abs_range
        print(f"Normalized Error Depth (Class {neg_label}): {norm_depth_neg:.4%}")
        print(f"Normalized Error Depth (Class {pos_label}): {norm_depth_pos:.4%}")
        
    return {
        "max_sv_dist_neg": max_sv_dist_neg,
        "max_sv_dist_pos": max_sv_dist_pos,
        "max_error_dist_neg": max_error_dist_neg,
        "max_error_dist_pos": max_error_dist_pos,
        "full_range": full_range,
        "norm_depth_neg": norm_depth_neg,
        "norm_depth_pos": norm_depth_pos
    }
