import os
import sys
from pathlib import Path

# Project Root (repo/)
# Assumes this file is in repo/lib/config.py
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 1. Internal Data Directory
# Default: repo/../data -> quivers/data
# Can be overridden by QUIVERS_DATA_DIR env var
_default_data_dir = PROJECT_ROOT.parent / "data"
DATA_DIR = Path(os.environ.get("QUIVERS_DATA_DIR", _default_data_dir)).resolve()

# 2. External Repository Directory
# Default: repo/../MACHINE_LEARNING...
# Can be overridden by EXTERNAL_REPO_DIR env var
_default_ext_repo = PROJECT_ROOT.parent / "MACHINE_LEARNING_MUTATION_ACYCLICITY_OF_QUIVERS"
EXTERNAL_REPO_DIR = Path(os.environ.get("EXTERNAL_REPO_DIR", _default_ext_repo)).resolve()

def get_db_path(db_name="quivers_rank4_canonical.db"):
    path = DATA_DIR / db_name
    if not path.exists():
        print(f"[WARN] Database not found at {path}")
    return str(path)

def get_external_data_path(experiment, filename):
    """
    Resolves paths like 'Experiment_3/Machine_Learning/MA...txt'
    relative to the external repo root.
    """
    path = EXTERNAL_REPO_DIR / experiment / "Machine_Learning" / filename
    if not path.exists():
         print(f"[WARN] External data file not found at {path}")
    return str(path)

def get_output_dir(script_file):
    """
    Creates and returns an 'output' directory next to the script file.
    """
    out_dir = Path(script_file).parent / "output"
    out_dir.mkdir(exist_ok=True)
    return str(out_dir)

# Add repo root to sys.path so scripts can import lib.config
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
