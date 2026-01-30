# Quiver Mutation Acyclicity Experiments

This directory contains the various machine learning experiments conducted to investigate invariants for Rank 4 quivers.

## Structure

*   **`01_reproduce_external/`**: Scripts to reproduce the results from prior work (Experiments 3 & 4), identifying issues with data scale and train/test leakage.
*   **`02_rank3_isomorphism/`**: Controlled experiments on Rank 3 quivers to demonstrate the necessity of canonicalization and symmetrized kernels for learning algebraic invariants like the Markov Constant.
*   **`03_rank4_canonical_svm/`**: Application of the "Invariant Kernel" and Canonicalization strategy to the largest isomorphism class of Rank 4 quivers.
*   **`04_rank4_feature_engineering/`**: Experiments using L1-Regularized Linear SVMs on explicit feature sets (Polynomials, Cycle Inequalities) to extract interpretable mathematical terms.
*   **`05_cross_evaluation/`**: Cross-dataset generalization tests (Exp 3 vs Exp 4 vs Our Data) and hyperparameter sweeps (Regularization, Normalization).
