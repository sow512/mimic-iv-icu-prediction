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
| EBM | [EBM](EBM/) | Reserved experiment folder |
| Historical experiments | [Research_Archives](Research_Archives/) | Master scripts and supporting benchmarks |

Some folders are placeholders, not evidence of completed experiments. See [curated results](../08_results/README.md) for aggregate metrics and limitations.
