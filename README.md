# MIMIC-IV ICU Transfer Prediction — Research Code Archive

A code-only collection of the renal ICU-transfer project and subsequent hospital-wide, competing-risk, binary neural, feature-engineering, and dynamic landmark experiments.

## Layout

- `src/early_renal/phases/`: original renal phases, audits, subgroup evaluation, tuning and competing risks.
- `src/early_renal/experiments/`: supporting benchmarks and comparisons.
- `src/early_renal/masters/`: distinct versions of the renal master experiment (kept when their contents differ).
- `src/hospital_wide/phases/`: later hospital-wide and renal-subgroup model experiments.
- `src/hospital_wide/experiments/`: supporting benchmarks.
- `src/dynamic/`: dynamic renal landmark pilot.
- `tools/`: local packaging and extraction utilities.
- `notebooks/`: notebook with cleared outputs.
- `CODE_INVENTORY.json`: source-to-destination mapping and checksums.
- `CLEANUP_REPORT.md`: deduplication and quality checks.

## Important

This repository intentionally contains **no MIMIC-IV data, feature tables, predictions, trained models or evaluation labels**. Obtain MIMIC-IV access independently and use only approved storage.

These are **historical experiment scripts**, not one unified, tested package. Some scripts require other scripts on `PYTHONPATH` or in the same working directory; their original CLI paths and expected output filenames were deliberately not changed. To reproduce a stage, inspect `python path/to/script.py --help`, and point arguments to the corresponding local authorized dataset/phase outputs.

The original renal cohorts and later hospital-wide cohorts have different eligibility/label policies and sometimes different patient splits. Their reported metrics must not be directly compared without harmonization. Feature availability, cached-lab provenance, competing-risk interval definitions, and repeated validation-set inspection remain open methodological issues. This code is retrospective research only, not clinical decision support.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall -q src tools
```

For safety, run `git status` and inspect staged files before any push. The `.gitignore` is defense-in-depth, not a substitute for checking that restricted data is absent.
