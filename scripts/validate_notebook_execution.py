
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
import os
import sys

def validate_notebook():
    nb_path = Path("notebooks/analysis.ipynb")
    print(f"Validating {nb_path}...")
    
    # Read the notebook
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    # Configure the executor
    # We use the current python kernel/env
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    # Try to execute
    try:
        # We need to run in the notebooks/ or root directory?
        # The notebook assumes "../data" paths, so it expects to be run from "notebooks/"
        # We will change CWD to notebooks/ temporarily
        cwd = os.getcwd()
        nb_dir = nb_path.parent.resolve()
        
        print(f"Switching CWD to {nb_dir} for execution context...")
        os.chdir(nb_dir)
        
        # Execute
        print("Starting cell-by-cell execution (this may take a few minutes)...")
        ep.preprocess(nb, {'metadata': {'path': str(nb_dir)}})
        
        print("\nSUCCESS: All cells in the notebook executed without errors!")
        
    except Exception as e:
        print(f"\nFAILURE: Notebook validation failed.\nError: {e}")
        # Identify the cell index
        # This catch is broad, ExecutePreprocessor usually raises CellExecutionError
        sys.exit(1)
        
    finally:
        os.chdir(cwd)

if __name__ == "__main__":
    validate_notebook()
