# Strategy: Approximating Rank 4 NMA/MA Classification via Polynomial SVM

## 1. Objective

The goal is to train a Support Vector Machine (SVM) with a polynomial kernel to classify Rank 4 quivers as Mutation-Acyclic (MA) or Non-Mutation-Acyclic (NMA). We hypothesize that, similar to the Rank 3 case (which is determined by the polynomial inequality $x^2 + y^2 + z^2 - xyz > 4$), the Rank 4 decision boundary can be approximated by a polynomial level set in the 6-dimensional space of the quiver coordinates.

## 2. Theoretical Motivation

- **Rank 3 Case**: The boundary is exactly defined by a degree-3 polynomial (Markov constant).
- **Rank 4 Case**: The boundary is known to be complex and related to the properties of $3 \times 3$ subquivers and potentially other invariants. A polynomial kernel SVM is the natural choice to attempt to capture this non-linear decision boundary.

## 3. Data Pipeline

- **Source**: `../data/quivers_rank4_250k.db`
- **Features**: The 6 upper-triangular entries of the skew-symmetric matrix $B$:
  $X = [b_{12}, b_{13}, b_{14}, b_{23}, b_{24}, b_{34}]$
- **Labels**: $y \in \{0, 1\}$ (0 for NMA, 1 for MA)

## 6. Implementation: Arbitrary-Precision Integers

Since our input features are integers, we can bypass floating-point issues entirely during the critical kernel computation step by utilizing Python's native arbitrary-precision integers.

**The Pipeline**:
1. **Raw Inputs**: Load the quiver vectors $X$ as standard integers.
2. **Exact Kernel Computation**:
   - Compute the kernel matrix $K$ using Python `int`.
   - Formula: $K_{ij} = (u_i^\top v_j + c_0)^d$
   - We set $\gamma=1$ and $c_0 \in \mathbb{Z}$ (e.g., 0 or 1) to ensure the result remains an integer.
   - Python handles the resulting massive integers (e.g., $10^{36}$) automatically without overflow.
3. **Kernel Normalization**:
   - The values in $K$ will be too large for the SVM solver.
   - We normalize the matrix: $K_{norm} = K / \max(K)$.
   - This maps all similarities to the $[0, 1]$ range, preserving the relative structure exactly.
4. **Final Cast**:
   - Cast $K_{norm}$ to `float64`.
   - Pass this precomputed kernel to `SVC(kernel='precomputed')`.

*Note: This approach is numerically superior to `float128` for integer data as it has zero precision loss during the polynomial expansion.*

## 7. Model Configuration (Scikit-Learn)

- **Classifier**: `sklearn.svm.SVC`
- **Kernel**:
  - Standard: `'poly'` (for initial `float64` tests).
  - High-Precision: `'precomputed'` (using the exact integer matrix strategy described above).
- **Hyperparameters to Tune**:
  - `degree` ($d$):
    - Start with **3** (analogy to Rank 3).
    - Test **4** (natural for Rank 4 matrix determinants/pfaffians).
    - Explore **2** through **6**.
  - `C` (Regularization): Controls hardness of the margin.
  - `coef0`: Important for polynomial kernels (allows interaction terms with lower degree).

## 8. Implementation Steps

1. **Data Loading**: Write a Python script to query the SQLite database.
2. **Preprocessing**:
   - Filter by `max_weight` if necessary.
   - Separate features ($X$) and targets ($y$).
   - Split into **Train** (80%) and **Test** (20%) sets.
   - **Skip Scaling**: Do not apply Min-Max scaling. Use raw integers.
3. **Training**: Use the precomputed kernel strategy with arbitrary precision integers.
4. **Evaluation**:
   - Compute Accuracy, Precision, Recall, and F1-Score.
   - **Critical**: Check if the model learns the "obvious" NMA cases (like Markov subquivers).
5. **Analysis**:
   - Inspect Support Vectors (count).
   - If accuracy is high (>95%), it suggests a polynomial structure exists.

## 9. Deliverables

- `train_svm.py`: Script to load data, train the model, and print metrics.
- `svm_results.txt`: Log of performance for different degrees.
