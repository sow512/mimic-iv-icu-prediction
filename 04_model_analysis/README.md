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
