# Explainable Boosting Machine (EBM) experiments

This folder contains the EBM training scripts supplied for the renal ICU-transfer research experiments. Aggregate metrics and experiment notes are indexed in [EBM results](../../08_results/ebm_experiments/README.md); the associated plots are grouped under [EBM figures](../../08_results/ebm_experiments/figures/).

## Scripts

- `ebm_20_feature_training.py` — 20-feature EBM.
- `ebm_75_feature_training.py` — baseline 75-feature EBM.
- `ebm_75_feature_tuning.py` — tuned 75-feature EBM.
- `ebm_75_feature_tuned_smote.py` — tuned 75-feature EBM with SMOTE/SMOTENC.
- `ebm_200_feature_training.py` — 200-feature EBM.
- `ebm_200_feature_smote_comparison.py` — baseline vs SMOTE ratios 0.2, 0.5, and 1.0 on the 200-feature dataset.
- `ebm_75_feature_smote_comparison.py` — audited patient-window cohort, feature selection, and SMOTE comparison.

These are historical experiment scripts, not a turnkey pipeline. They require authorized local MIMIC-IV-derived input files and compatible dependencies. Review each script's CLI arguments, label definition, feature availability assumptions, and expected input format before execution.

## File index

| File | Description |
|---|---|
| [`ebm_200_feature_training.py`](ebm_200_feature_training.py) | EBM 200 (python research script) |
| [`ebm_20_feature_training.py`](ebm_20_feature_training.py) | EBM 20 features (python research script) |
| [`ebm_75_feature_training.py`](ebm_75_feature_training.py) | EBM 75 (python research script) |
| [`ebm_75_feature_tuned_smote.py`](EBM_75_SMOTE%28Tuned%29.py) | EBM 75 SMOTE(Tuned) (python research script) |
| [`ebm_75_feature_tuning.py`](ebm_75_feature_tuning.py) | EBM 75 tuned (python research script) |
| [`ebm_75_feature_smote_comparison.py`](ebm_75_feature_smote_comparison.py) | EBM SMOTE75 (python research script) |
| [`ebm_200_feature_smote_comparison.py`](ebm_200_feature_smote_comparison.py) | EBM SMOTE 200 (python research script) |

## Reproducibility status

This repository documents retrospective experiments. Access-controlled MIMIC-IV datasets and some fitted model artifacts are not included. The automated checks validate syntax and repository structure, not end-to-end reproduction or clinical safety. Different cohort definitions and evaluation splits must not be compared as a single leaderboard.

