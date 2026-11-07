# Explainable Boosting Machine (EBM) experiments

This folder contains the EBM training scripts supplied for the renal ICU-transfer research experiments. Aggregate metrics and experiment notes are indexed in [EBM results](../../08_results/ebm_experiments/README.md); the associated plots are stored under [08_results](../../08_results/).

## Scripts

- `EBM_20_features.py` — 20-feature EBM.
- `EBM_75.py` — baseline 75-feature EBM.
- `EBM_75_tuned.py` — tuned 75-feature EBM.
- `EBM_75_SMOTE(Tuned).py` — tuned 75-feature EBM with SMOTE/SMOTENC.
- `EBM_200.py` — 200-feature EBM.
- `EBM_SMOTE_200.py` — baseline vs SMOTE ratios 0.2, 0.5, and 1.0 on the 200-feature dataset.
- `EBM_SMOTE75.py` — audited patient-window cohort, feature selection, and SMOTE comparison.

These are historical experiment scripts, not a turnkey pipeline. They require authorized local MIMIC-IV-derived input files and compatible dependencies. Review each script's CLI arguments, label definition, feature availability assumptions, and expected input format before execution.
