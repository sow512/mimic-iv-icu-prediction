# Experiment index

This index is a navigation aid, **not** a pooled leaderboard. Phase labels reflect the experiment reports; filenames may not contain the phase number.

| Experiment | Scope | Model / method | Where to look | Evaluation caveat |
| --- | --- | --- | --- | --- |
| Phase 9C | Original retrospective renal cohort | LightGBM, CatBoost, weighted blend; five feature configurations (15 rows) | `03_model_training/Tree_Ensembles/clinical_feature_ablation_lgbm_catboost.py`; `08_results/phase9c_feature_ablation.csv` | 6,030 evaluation admissions, 302 transfers; not directly comparable to later renal subgroup |
| Renal competing-risk benchmark (Phase 21/22 reports) | Newer renal subgroup | Linear discrete-time competing risks; deep discrete-time competing risks | `03_model_training/Competing_Risks/renal_competing_risk_benchmark.py`; `08_results/renal_neural_comparison.csv` | Confirm phase naming and training provenance before publication |
| Phase 26 renal models | Newer renal subgroup | SHAP-selected binary MLP; SHAP-selected neural competing risks | `03_model_training/Neural_Networks/renal_binary_mlp_training.py`; `03_model_training/Competing_Risks/renal_only_competing_risk_evaluation.py`; `08_results/renal_neural_comparison.csv` | 100 selected numerical / 122 encoded inputs; validation results |
| Phase 29 | Dynamic renal risk sets | Repeated-landmark prediction | `03_model_training/Dynamic_Models/dynamic_renal_landmark_prediction.py` | Landmark populations and event prevalences change |
| Phase 30 | Original renal cohort, development partitions | Optuna-tuned MLP (1,009 inputs in selected configuration) | `03_model_training/Neural_Networks/renal_neural_hyperparameter_tuning.py`; `08_results/phase30_mlp_development.csv` | Model-selection AUPRC is not untouched test performance |
| Phase 31 | Not indexed | Not verified | No clearly named phase31 script/result in current tree | Do not invent metrics |

## Interpretation rules

- Always state **training population** separately from **evaluation population**.
- Record prediction landmark and horizon, positive event definition, cohort size and prevalence.
- For neural networks, distinguish architecture (MLP), output objective (binary vs competing risks), and feature policy (full vs SHAP-selected).
- Timestamp availability and retrospective renal diagnosis eligibility remain important methodological concerns.
- ICU transfer predicts observed utilization; it is not a validated measure of ICU need.

## New experiment record template

```text
Experiment ID:
Contributor / PR:
Training cohort:
Evaluation cohort:
Prediction landmark / horizon:
Feature source and availability audit:
Feature count (raw / encoded):
Model architecture and objective:
Data split (patient-disjoint?):
AUROC / AUPRC / Brier / recall at fixed alert capacity:
Evaluation partition (selection / calibration / untouched test):
Results location:
Known limitations:
```
