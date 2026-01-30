# Data Analysis and Verification

Scripts for inspecting the generated quiver databases.

*   **`dataset_analysis.py`**: Computes statistical profiles (weight distribution, class balance) for the three main datasets.
*   **`analyze_canonical_db.py`**: Connects to the canonical database, counts isomorphism classes, and generates visualizations of sample quivers.
*   **`verify_canonicalization.py`**: Rigorously checks that the stored `aligned_weights` in the database match the sign pattern dictated by their `digraph_class` key.
