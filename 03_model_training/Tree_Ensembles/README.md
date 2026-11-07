# Tree Ensembles

LightGBM, CatBoost, weighted blends, clinical feature ablations, Optuna tuning and hospital-wide benchmarks.

## Contents

- [`clinical_feature_ablation_lgbm_catboost.py`](clinical_feature_ablation_lgbm_catboost.py) — Python experiment script.
- [`hospitalwide_multimodel_benchmark.py`](hospitalwide_multimodel_benchmark.py) — Python experiment script.
- [`optuna_tree_model_tuning.py`](optuna_tree_model_tuning.py) — Python experiment script.
- [`original_style_lgbm_catboost.py`](original_style_lgbm_catboost.py) — Python experiment script.
- [`renal_lightgbm_catboost_training.py`](renal_lightgbm_catboost_training.py) — Python experiment script.

## Scope and reproducibility

Experiments may use different cohorts, feature definitions and validation partitions. Compare reported metrics only when these match. MIMIC-IV data is access-controlled and not included; scripts may require authorized local input files. These results are research-only and not validated for clinical use.
