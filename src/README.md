# Research code index

Original phase scripts now live alongside the team's code in the numbered folders. No duplicate copies are maintained here.

| Topic | Canonical directory |
|---|---|
| Cohort / features | [`01_data_preprocessing/`](../01_data_preprocessing/) |
| Feature selection | [`02_feature_selection/`](../02_feature_selection/) |
| Tree ensembles | [`03_model_training/Tree_Ensembles/`](../03_model_training/Tree_Ensembles/) |
| Neural models | [`03_model_training/Neural_Networks/`](../03_model_training/Neural_Networks/) |
| Competing risks | [`03_model_training/Competing_Risks/`](../03_model_training/Competing_Risks/) |
| Dynamic prediction | [`03_model_training/Dynamic_Models/`](../03_model_training/Dynamic_Models/) |
| Additional research | [`03_model_training/Research_Archives/`](../03_model_training/Research_Archives/) |
| Evaluation | [`04_model_analysis/`](../04_model_analysis/) |
| Optimization | [`05_icu_optimization/`](../05_icu_optimization/) |

See [MOVE_MAP.md](MOVE_MAP.md) for exact file paths. Historical hard-coded paths may need updating before execution; no full runtime validation has been performed.

## File index

| File | Description |
|---|---|
| [`MOVE_MAP.md`](MOVE_MAP.md) | MOVE MAP.md (supporting file) |

## Reproducibility status

This repository documents retrospective experiments. Access-controlled MIMIC-IV datasets and some fitted model artifacts are not included. The automated checks validate syntax and repository structure, not end-to-end reproduction or clinical safety. Different cohort definitions and evaluation splits must not be compared as a single leaderboard.

