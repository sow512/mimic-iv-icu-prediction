# Feature selection

Feature-selection approaches include VIF/RFE, RFECV, TreeSHAP, feature-budget ablations, and timestamp availability audits.

- [RFECV feature selection (484 candidates)](rfecv_feature_selection_484_candidates.py) — exploratory XGBoost RFECV; currently contains machine-specific input paths.
- [Feature budget ablation](feature_budget_ablation.py) — tests reduced feature sets.
- [SHAP-ranked selection](shap_ranked_feature_selection.py) — feature engineering/selection for later experiments.
- [Feature availability audit](feature_timestamp_availability_audit.py) — checks timing assumptions.
- [Original preprocessing notebook](../01_data_preprocessing/mimic_iv_cohort_preprocessing_feature_engineering.ipynb) — initial VIF/RFE workflow.

Feature selection must be fit within development data, never on held-out evaluation labels.

## File index

| File | Description |
|---|---|
| [`feature_budget_ablation.py`](feature_budget_ablation.py) | feature budget ablation (python research script) |
| [`feature_selection_benchmark.py`](feature_selection_benchmark.py) | feature selection benchmark (python research script) |
| [`feature_timestamp_availability_audit.py`](feature_timestamp_availability_audit.py) | feature timestamp availability audit (python research script) |
| [`rfe_selected_20_features.txt`](rfe_selected_20_features.txt) | rfe selected 20 features (feature list) |
| [`rfecv_feature_selection_484_candidates.py`](rfecv_feature_selection_484_candidates.py) | rfecv feature selection 484 candidates (python research script) |
| [`shap_200_feature_analysis.py`](shap_200_feature_analysis.py) | shap 200 feature analysis (python research script) |
| [`shap_feature_count_comparison.py`](shap_feature_count_comparison.py) | shap feature count comparison (python research script) |
| [`shap_lab_test_selection.py`](shap_lab_test_selection.py) | shap lab test selection (python research script) |
| [`shap_ranked_feature_selection.py`](shap_ranked_feature_selection.py) | shap ranked feature selection (python research script) |
| [`vif_removed_17_features.txt`](vif_removed_17_features.txt) | vif removed 17 features (feature list) |
| [`vif_retained_60_features.txt`](vif_retained_60_features.txt) | vif retained 60 features (feature list) |

## Reproducibility status

This repository documents retrospective experiments. Access-controlled MIMIC-IV datasets and some fitted model artifacts are not included. The automated checks validate syntax and repository structure, not end-to-end reproduction or clinical safety. Different cohort definitions and evaluation splits must not be compared as a single leaderboard.

