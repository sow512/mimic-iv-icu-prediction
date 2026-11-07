# Model training

The team experiments are organized by **model family**, not contributor. Different subfolders may use different cohorts and feature sets; results should only be compared when cohort, split, landmark, and outcome are matched.

| Model family | Directory | Contents |
|---|---|---|
| XGBoost | [XGBoost](XGBoost/) | 20/75/200-feature experiments, SMOTE comparisons, 75-feature training |
| Tree ensembles | [Tree_Ensembles](Tree_Ensembles/) | LightGBM, CatBoost, ensemble ablations and benchmarks |
| Neural networks | [Neural_Networks](Neural_Networks/) | Binary MLP and renal tuning |
| Competing risks | [Competing_Risks](Competing_Risks/) | Linear/neural discrete-time competing-event modeling |
| Dynamic models | [Dynamic_Models](Dynamic_Models/) | Repeated-landmark prediction |
| SVM/MLP | [SVM_MLP](SVM_MLP/) | Team notebooks |
| Random Forest | [Random_Forest](Random_Forest/) | Reserved experiment folders |
| EBM | [EBM](EBM/) | Explainable boosting, feature-budget, tuning and SMOTE experiments |
| Historical experiments | [Research_Archives](Research_Archives/) | Master scripts and supporting benchmarks |

Random Forest directories are currently placeholders; the EBM directory contains completed experiment scripts. See [curated results](../08_results/README.md) for aggregate metrics and limitations.

## File index

- [Competing_Risks](Competing_Risks/) — experiment group or supporting materials.
- [Dynamic_Models](Dynamic_Models/) — experiment group or supporting materials.
- [EBM](EBM/) — experiment group or supporting materials.
- [Neural_Networks](Neural_Networks/) — experiment group or supporting materials.
- [Random_Forest](Random_Forest/) — experiment group or supporting materials.
- [Research_Archives](Research_Archives/) — experiment group or supporting materials.
- [SVM_MLP](SVM_MLP/) — experiment group or supporting materials.
- [Tree_Ensembles](Tree_Ensembles/) — experiment group or supporting materials.
- [XGBoost](XGBoost/) — experiment group or supporting materials.


## Reproducibility status

This repository documents retrospective experiments. Access-controlled MIMIC-IV datasets and some fitted model artifacts are not included. The automated checks validate syntax and repository structure, not end-to-end reproduction or clinical safety. Different cohort definitions and evaluation splits must not be compared as a single leaderboard.

