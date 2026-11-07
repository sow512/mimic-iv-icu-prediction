# Feature selection

Feature-selection approaches include VIF/RFE, RFECV, TreeSHAP, feature-budget ablations, and timestamp availability audits.

- [RFECV feature selection (484 candidates)](rfecv_feature_selection_484_candidates.py) — exploratory XGBoost RFECV; currently contains machine-specific input paths.
- [Feature budget ablation](feature_budget_ablation.py) — tests reduced feature sets.
- [SHAP-ranked selection](shap_ranked_feature_selection.py) — feature engineering/selection for later experiments.
- [Feature availability audit](feature_timestamp_availability_audit.py) — checks timing assumptions.
- [Original preprocessing notebook](../01_data_preprocessing/mimic_iv_cohort_preprocessing_feature_engineering.ipynb) — initial VIF/RFE workflow.

Feature selection must be fit within development data, never on held-out evaluation labels.
