# WARNING: This code is almost entirely AI-generated, and may have lots of bugs or subtle mistakes!

# Quiver Mutation Acyclicity: Machine Learning Investigation

This repository contains code, data generation tools, and analysis scripts for investigating the **Mutation-Acyclicity** property of Rank 4 quivers using Machine Learning.

## Directory Structure

*   **`rust_generation/`**: High-performance Rust code to generate, classify, and canonicalize Rank 4 quivers. Includes logic for computing Markov constants and verifying acyclicity.
*   **`experiments/`**: Python scripts for various ML experiments, organized by topic:
    *   `01_reproduce_external/`: Replication of prior work (Experiments 3 & 4) to understand baselines.
    *   `02_rank3_isomorphism/`: Controlled experiments on Rank 3 quivers to solve the graph isomorphism problem using symmetrized kernels.
    *   `03_rank4_canonical_svm/`: Application of canonicalization strategies to the largest Rank 4 isomorphism class.
    *   `04_rank4_feature_engineering/`: Advanced feature engineering (L1 sparsity, cycle inequalities) to extract mathematical invariants.
    *   `05_cross_evaluation/`: rigorous cross-dataset evaluation and hyperparameter sweeps.
*   **`analysis/`**: Tools for inspecting the generated databases (`.db`), verifying data consistency, and visualizing quiver structures.
*   **`lib/`**: Shared Python libraries, including `svm_utils.py` for custom kernel implementations and `config.py` for path management.
*   **`docs/`**: Documentation and detailed reports, including `investigation_report.md`.

## Setup and Configuration

The scripts in this repository rely on a central configuration file (`lib/config.py`) to locate data files and external repositories.

### Environment Variables

You can configure the paths using the following environment variables. If not set, they default to the relative paths shown below.

| Variable | Description | Default Value |
| :--- | :--- | :--- |
| **`QUIVERS_DATA_DIR`** | Path to the directory containing `.db` files and generated data. | `../data` |
| **`EXTERNAL_REPO_DIR`** | Path to the cloned `MACHINE_LEARNING_MUTATION_ACYCLICITY_OF_QUIVERS` repository (for Exp 3/4 data). | `../MACHINE_LEARNING_MUTATION_ACYCLICITY_OF_QUIVERS` |

### Installation & Setup

Since the virtual environment (`venv/`) is not included in the repository, you must create it and install the dependencies locally.

1.  **Create the Virtual Environment**:
    ```bash
    python3 -m venv venv
    ```

2.  **Install Dependencies**:
    ```bash
    ./venv/bin/pip install -r requirements.txt
    ```
    (Dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `networkx`)

    *Note: `rust_generation` requires a working Rust toolchain (cargo).*

### Python Environment Usage

**To execute scripts, you must either activate the virtual environment or use the binary directly.**

To use the binary directly (recommended for one-off commands):
```bash
./venv/bin/python <script_path>
```

To activate the environment (for interactive sessions):
```bash
source venv/bin/activate
```

## Usage

### 1. Generating Data
To generate the canonical Rank 4 database:
```bash
cd rust_generation
cargo run --release
```
This will create `quivers_rank4_canonical.db` in your configured `QUIVERS_DATA_DIR`.

### 2. Running Experiments
All python scripts can be run directly from the repository root using the virtual environment. They will automatically resolve paths.

Example: Running the feature extraction experiment
```bash
./venv/bin/python experiments/04_rank4_feature_engineering/extract_cycle_features_importance.py
```

Example: Verifying database consistency
```bash
./venv/bin/python analysis/verify_canonicalization.py
```
