"""Verify current repository layout without depending on historical filenames."""
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
required = ["README.md", "01_data_preprocessing/README.md", "02_feature_selection/README.md", "03_model_training/README.md", "04_model_analysis/README.md", "05_icu_optimization/README.md", "06_streamlit_frontend/app.py", "07_validation/README.md", "08_results/README.md"]
errors = [f"Missing required file: {p}" for p in required if not (root / p).is_file()]
for path in root.rglob("*"):
    if path.is_file() and (path.name.endswith(".py.py") or path.name.endswith(".ipynb.ipynb")):
        errors.append(f"Duplicate file extension: {path.relative_to(root)}")
for error in errors:
    print("ERROR:", error)
print(f"Repository layout: {len(errors)} errors")
sys.exit(bool(errors))
