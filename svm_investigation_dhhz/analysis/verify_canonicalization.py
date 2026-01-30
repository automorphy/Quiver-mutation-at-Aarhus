import sys
import os
import sqlite3
import pandas as pd
import ast
import numpy as np

# Add repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.config import get_db_path

def verify_db():
    db_path = get_db_path("quivers_rank4_canonical.db")
    conn = sqlite3.connect(db_path)
    
    print(f"Checking consistency of {db_path}...")
    
    # Load all unique classes first
    # query = "SELECT id, digraph_class, aligned_weights FROM quivers LIMIT 10000" # Check a subset first
    query = "SELECT id, digraph_class, aligned_weights FROM quivers" # Full check
    
    chunk_size = 10000
    offset = 0
    mismatches = 0
    checked = 0
    
    while True:
        df = pd.read_sql_query(f"{query} LIMIT {chunk_size} OFFSET {offset}", conn)
        if df.empty:
            break
            
        for _, row in df.iterrows():
            checked += 1
            key_str = row['digraph_class']
            weights_str = row['aligned_weights']
            
            try:
                target_signs = ast.literal_eval(key_str)
                weights = ast.literal_eval(weights_str)
                
                # Compute actual signs from weights
                actual_signs = [int(np.sign(w)) for w in weights]
                
                # Signum of 0 is 0. 
                # Rust signum: 0 -> 0.
                
                if actual_signs != target_signs:
                    mismatches += 1
                    if mismatches < 10:
                        print(f"Mismatch at ID {row['id']}:")
                        print(f"  Target Key: {target_signs}")
                        print(f"  Actual Sgn: {actual_signs}")
                        print(f"  Weights:    {weights}")
            except Exception as e:
                print(f"Error parsing row {row['id']}: {e}")
                
        offset += chunk_size
        if checked % 50000 == 0:
            print(f"Checked {checked} rows...")

    print("\nVerification Complete.")
    print(f"Total Checked: {checked}")
    print(f"Total Mismatches: {mismatches}")
    
    conn.close()

if __name__ == "__main__":
    verify_db()