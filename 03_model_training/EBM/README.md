# Explainable Boosting Machine (EBM) experiments

This folder contains the EBM training scripts supplied for the renal ICU-transfer research experiments. Aggregate metrics and experiment notes are indexed in [EBM results](../../08_results/ebm_experiments/README.md); the associated plots are grouped under [EBM figures](../../08_results/ebm_experiments/figures/).

## Scripts

- `EBM_20_features.py` — 20-feature EBM.
- `EBM_75.py` — baseline 75-feature EBM.
- `EBM_75_tuned.py` — tuned 75-feature EBM.
- `EBM_75_SMOTE(Tuned).py` — tuned 75-feature EBM with SMOTE/SMOTENC.
- `EBM_200.py` — 200-feature EBM.
- `EBM_SMOTE_200.py` — baseline vs SMOTE ratios 0.2, 0.5, and 1.0 on the 200-feature dataset.
- `EBM_SMOTE75.py` — audited patient-window cohort, feature selection, and SMOTE comparison.

These are historical experiment scripts, not a turnkey pipeline. They require authorized local MIMIC-IV-derived input files and compatible dependencies. Review each script's CLI arguments, label definition, feature availability assumptions, and expected input format before execution.

## File index

| File | Description |
|---|---|
| [`EBM_200.py`](EBM_200.py) | EBM 200 (python research script) |
| [`EBM_20_features.py`](EBM_20_features.py) | EBM 20 features (python research script) |
| [`EBM_75.py`](EBM_75.py) | EBM 75 (python research script) |
| [`EBM_75_SMOTE(Tuned).py`](EBM_75_SMOTE%28Tuned%29.py) | EBM 75 SMOTE(Tuned) (python research script) |
| [`EBM_75_tuned.py`](EBM_75_tuned.py) | EBM 75 tuned (python research script) |
| [`EBM_SMOTE75.py`](EBM_SMOTE75.py) | EBM SMOTE75 (python research script) |
| [`EBM_SMOTE_200.py`](EBM_SMOTE_200.py) | EBM SMOTE 200 (python research script) |

## Reproducibility status

This repository documents retrospective experiments. Access-controlled MIMIC-IV datasets and some fitted model artifacts are not included. The automated checks validate syntax and repository structure, not end-to-end reproduction or clinical safety. Different cohort definitions and evaluation splits must not be compared as a single leaderboard.

