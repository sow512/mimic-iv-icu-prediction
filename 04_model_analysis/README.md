# Model evaluation and explainability

This folder contains calibration, subgroup analyses, bootstrap validation, SHAP explainability, feature provenance audits, and XGBoost evaluation.

- [XGBoost 75-feature extended analysis](xgboost_75_extended_analysis.py)
- [XGBoost evaluation](xgboost_75_model_evaluation.py)
- [XGBoost calibration](xgboost_75_probability_calibration.py)
- [XGBoost SHAP](xgboost_75_shap_explainability.py)
- [Renal subgroup evaluation](renal_subgroup_evaluation.py)
- [Patient-level bootstrap](competing_risk_patient_bootstrap_validation.py)
- [Clinical feature provenance audit](clinical_feature_provenance_audit.py)

For figures and reported metrics see [08_results](../08_results/README.md). Some scripts depend on locally generated derived datasets and are not yet tested end-to-end in this layout.

## File index

| File | Description |
|---|---|
| [`clinical_feature_provenance_audit.py`](clinical_feature_provenance_audit.py) | clinical feature provenance audit (python research script) |
| [`compare_cohort_prevalence.py`](compare_cohort_prevalence.py) | compare cohort prevalence (python research script) |
| [`compare_prediction_horizons.py`](compare_prediction_horizons.py) | compare prediction horizons (python research script) |
| [`compare_renal_models.py`](compare_renal_models.py) | compare renal models (python research script) |
| [`competing_risk_bootstrap_validation.py`](competing_risk_bootstrap_validation.py) | competing risk bootstrap validation (python research script) |
| [`competing_risk_patient_bootstrap_validation.py`](competing_risk_patient_bootstrap_validation.py) | competing risk patient bootstrap validation (python research script) |
| [`feature_leakage_ablation_audit.py`](feature_leakage_ablation_audit.py) | feature leakage ablation audit (python research script) |
| [`probability_calibration_diagnostics.py`](probability_calibration_diagnostics.py) | probability calibration diagnostics (python research script) |
| [`renal_model_validation.py`](renal_model_validation.py) | renal model validation (python research script) |
| [`renal_shap_explainability.py`](renal_shap_explainability.py) | renal shap explainability (python research script) |
| [`renal_subgroup_evaluation.py`](renal_subgroup_evaluation.py) | renal subgroup evaluation (python research script) |
| [`train_and_calibrate_renal_models.py`](train_and_calibrate_renal_models.py) | train and calibrate renal models (python research script) |
| [`xgboost_75_extended_analysis.py`](xgboost_75_extended_analysis.py) | xgboost 75 extended analysis (python research script) |
| [`xgboost_75_model_evaluation.py`](xgboost_75_model_evaluation.py) | xgboost 75 model evaluation (python research script) |
| [`xgboost_75_probability_calibration.py`](xgboost_75_probability_calibration.py) | xgboost 75 probability calibration (python research script) |
| [`xgboost_75_shap_explainability.py`](xgboost_75_shap_explainability.py) | xgboost 75 shap explainability (python research script) |

## Reproducibility status

This repository documents retrospective experiments. Access-controlled MIMIC-IV datasets and some fitted model artifacts are not included. The automated checks validate syntax and repository structure, not end-to-end reproduction or clinical safety. Different cohort definitions and evaluation splits must not be compared as a single leaderboard.

