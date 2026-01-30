import sys
import os
import sqlite3
import pandas as pd
import ast
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Add repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.config import get_db_path, get_output_dir

def visualize_quivers(samples, class_counts, output_file="digraph_samples.png"):
    n_classes = len(samples)
    n_samples = 5
    
    # Create a figure with GridSpec to have a separate column for row labels/titles if needed,
    # or just rely on the first column axes.
    # We will increase height per row to make it clear.
    
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(15, 4 * n_classes))
    fig.suptitle("Sample Quivers per Digraph Isomorphism Class\n(Rows = Distinct Isomorphism Classes)", fontsize=20, y=0.98)

    # Convert dictionary keys to list for indexing
    classes = list(samples.keys())

    for i, cls_key in enumerate(classes):
        quivers = samples[cls_key]
        count = class_counts.get(cls_key, 0)
        
        # Add a row title spanning the columns
        # We can use fig.text relative coordinates, but it's tricky with subplots.
        # Instead, we set the ylabel of the first axes to be very large and descriptive.
        
        # Better: Add a text annotation to the left of the first plot
        
        for j in range(n_samples):
            # Handle 1D axes array if n_classes=1
            if n_classes > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]
            
            if j >= len(quivers):
                ax.axis('off')
                continue

            q_vec = quivers[j]
            
            # Reconstruct 4x4 matrix
            B = np.zeros((4, 4))
            idx = 0
            for r in range(4):
                for c in range(r + 1, 4):
                    B[r, c] = q_vec[idx]
                    B[c, r] = -q_vec[idx]
                    idx += 1
            
            G = nx.DiGraph()
            G.add_nodes_from([1, 2, 3, 4])
            
            # Positions: Square
            pos = {1: (0, 1), 2: (1, 1), 3: (0, 0), 4: (1, 0)}
            
            # Add edges
            for r in range(4):
                for c in range(4):
                    w = B[r, c]
                    if w > 0:
                        G.add_edge(r+1, c+1, weight=w)
            
            # Draw
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', 
                    node_size=600, font_weight='bold', arrowsize=20, edge_color='gray')
            
            # Edge labels
            edge_labels = {(u, v): f"{d['weight']:.0f}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, label_pos=0.5, font_color='red', font_weight='bold')
            
            # Print Matrix for Verification
            if j < 2 and i < 2: # Limit output to first few
                print(f"\n--- Class {i+1} Sample {j+1} ---")
                print(f"Key: {cls_key}")
                print(f"Weights: {q_vec}")
                print("Adjacency Matrix (B):")
                print(B)
            
            # Row Labeling on the first column
            if j == 0:
                # Add a visible rectangle or line to separate?
                # Just a big label
                label_text = f"Class #{i+1}\nCount: {count}\nKey: {cls_key[:15]}..."
                ax.set_ylabel(label_text, rotation=0, labelpad=60, size=14, weight='bold', ha='center', va='center')
                
            ax.set_title(f"Sample {j+1}", fontsize=10)

    plt.tight_layout(rect=[0.05, 0.02, 1, 0.96]) # Leave room on left for y-labels
    
    # Draw horizontal lines between rows to separate classes
    # Get figure coordinates
    # We can just rely on the spacing and labels.
    
    plt.savefig(output_file)
    print(f"Saved visualization to {output_file}")

def analyze_db():
    db_path = get_db_path("quivers_rank4_canonical.db")
    conn = sqlite3.connect(db_path)
    
    print(f"Connected to {db_path}")
    
    # 1. Total Count
    total = pd.read_sql_query("SELECT COUNT(*) as count FROM quivers", conn).iloc[0]['count']
    print(f"Total Quivers: {total}")

    # 1.5 Unique Isomorphism Classes
    n_classes = pd.read_sql_query("SELECT COUNT(DISTINCT digraph_class) as count FROM quivers", conn).iloc[0]['count']
    print(f"Total Isomorphism Classes: {n_classes}")
    
    # 2. Isomorphism Class Counts
    print("\n--- Top 20 Digraph Isomorphism Classes ---")
    query = """
        SELECT digraph_class, COUNT(*) as count 
        FROM quivers 
        GROUP BY digraph_class 
        ORDER BY count DESC 
        LIMIT 20
    """
    df_classes = pd.read_sql_query(query, conn)
    print(df_classes)
    
    # 3. Sample for Top 3 Classes
    top_classes = df_classes['digraph_class'].head(3).values
    samples = {}
    class_counts = {}
    
    for cls in top_classes:
        query_sample = f"SELECT aligned_weights FROM quivers WHERE digraph_class = ? LIMIT 5"
        df_sample = pd.read_sql_query(query_sample, conn, params=(cls,))
        # aligned_weights is a string "[...]", need to parse it
        parsed_samples = [ast.literal_eval(row) for row in df_sample['aligned_weights']]
        samples[cls] = parsed_samples
        class_counts[cls] = df_classes.loc[df_classes['digraph_class'] == cls, 'count'].values[0]
        
    out_dir = get_output_dir(__file__)
    visualize_quivers(samples, class_counts, output_file=os.path.join(out_dir, "digraph_samples.png"))
    
    conn.close()

if __name__ == "__main__":
    analyze_db()
