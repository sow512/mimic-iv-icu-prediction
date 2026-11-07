"""Fail CI if tracked source contains machine-specific absolute paths."""
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
patterns = [
    re.compile(r"[A-Za-z]:[\\\\/](?:Users|Documents and Settings)[\\\\/]", re.I),
    re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    re.compile(r"(?i)(?:OneDrive|Downloads)[\\\\/](?:Desktop|MIMIC-IV|MIMIC_IV)"),
]
errors = []
for folder in ("01_data_preprocessing", "02_feature_selection", "03_model_training",
               "04_model_analysis", "05_icu_optimization", "06_streamlit_frontend",
               "07_validation", "notebooks", "tools"):
    for path in (root / folder).rglob("*"):
        if path.suffix not in (".py", ".ipynb", ".md"):
            continue
        if path.name == "check_portable_paths.py":
            continue
        text = path.read_text(encoding="utf-8")
        for pat in patterns:
            if pat.search(text):
                errors.append(str(path.relative_to(root)))
                break
for path in errors:
    print("Machine-specific path detected:", path)
print(f"Portable path scan: {len(errors)} files flagged")
sys.exit(bool(errors))
