# Rank 4 Quiver Data Generation (Rust)

High-performance Rust code to generate, classify, and canonicalize Rank 4 quivers.

## Features
*   **Generation**: Random mutation exploration and seed-based expansion.
*   **Verification**: Checks acyclicity and computes Markov constants (for Rank 3 subquivers).
*   **Canonicalization**: Identifies the isomorphism class of the underlying unweighted digraph and aligns edge weights to a standard form.
*   **Database**: Stores results in a SQLite database.

## Usage
```bash
cargo build --release
./target/release/quiver_mutation
```
