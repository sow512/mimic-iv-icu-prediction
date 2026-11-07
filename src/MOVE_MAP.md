# Moved research scripts

## Naming convention

Canonical Python filenames describe their function. The original experiment identifiers are retained in the first column below for reproducibility. Existing team-owned scripts were not renamed.


The source blobs were moved without changing their contents.

| Previous path | New path |
|---|---|
| `src/dynamic/phase29_dynamic_renal_pilot.py` | [`03_model_training/Dynamic_Models/dynamic_renal_landmark_prediction.py`](../03_model_training/Dynamic_Models/dynamic_renal_landmark_prediction.py) |
| `src/early_renal/experiments/compare_cohort_prevalence.py` | [`04_model_analysis/compare_cohort_prevalence.py`](../04_model_analysis/compare_cohort_prevalence.py) |
| `src/early_renal/experiments/compare_prediction_horizons.py` | [`04_model_analysis/compare_prediction_horizons.py`](../04_model_analysis/compare_prediction_horizons.py) |
| `src/early_renal/experiments/feature_selection_benchmark.py` | [`02_feature_selection/feature_selection_benchmark.py`](../02_feature_selection/feature_selection_benchmark.py) |
| `src/early_renal/experiments/original_style_lgbm_catboost.py` | [`03_model_training/Tree_Ensembles/original_style_lgbm_catboost.py`](../03_model_training/Tree_Ensembles/original_style_lgbm_catboost.py) |
| `src/early_renal/experiments/renal_icu_all_phases.py` | [`03_model_training/Research_Archives/experiments/renal_icu_all_phases.py`](../03_model_training/Research_Archives/experiments/renal_icu_all_phases.py) |
| `src/early_renal/experiments/renal_icu_megabench.py` | [`03_model_training/Research_Archives/experiments/renal_icu_megabench.py`](../03_model_training/Research_Archives/experiments/renal_icu_megabench.py) |
| `src/early_renal/masters/renal_master.py` | [`03_model_training/Research_Archives/masters/renal_master.py`](../03_model_training/Research_Archives/masters/renal_master.py) |
| `src/early_renal/phases/phase10_validate.py` | [`04_model_analysis/renal_model_validation.py`](../04_model_analysis/renal_model_validation.py) |
| `src/early_renal/phases/phase11_admin_ablation.py` | [`03_model_training/Research_Archives/experiments/administrative_feature_ablation.py`](../03_model_training/Research_Archives/experiments/administrative_feature_ablation.py) |
| `src/early_renal/phases/phase12_subgroup_evaluation.py` | [`04_model_analysis/renal_subgroup_evaluation.py`](../04_model_analysis/renal_subgroup_evaluation.py) |
| `src/early_renal/phases/phase13_optuna_tuning.py` | [`03_model_training/Research_Archives/experiments/optuna_renal_model_tuning.py`](../03_model_training/Research_Archives/experiments/optuna_renal_model_tuning.py) |
| `src/early_renal/phases/phase14_competing_risks.py` | [`03_model_training/Competing_Risks/renal_discrete_time_competing_risks.py`](../03_model_training/Competing_Risks/renal_discrete_time_competing_risks.py) |
| `src/early_renal/phases/phase17_patient_flow_audit.py` | [`01_data_preprocessing/patient_flow_cohort_audit.py`](../01_data_preprocessing/patient_flow_cohort_audit.py) |
| `src/early_renal/phases/phase1_cohort.py` | [`01_data_preprocessing/renal_cohort_24h_landmark.py`](../01_data_preprocessing/renal_cohort_24h_landmark.py) |
| `src/early_renal/phases/phase2_labs.py` | [`01_data_preprocessing/temporal_laboratory_features.py`](../01_data_preprocessing/temporal_laboratory_features.py) |
| `src/early_renal/phases/phase3_train.py` | [`03_model_training/Tree_Ensembles/renal_lightgbm_catboost_training.py`](../03_model_training/Tree_Ensembles/renal_lightgbm_catboost_training.py) |
| `src/early_renal/phases/phase4_calibration_diagnostics.py` | [`04_model_analysis/probability_calibration_diagnostics.py`](../04_model_analysis/probability_calibration_diagnostics.py) |
| `src/early_renal/phases/phase4b_train_calibrate.py` | [`04_model_analysis/train_and_calibrate_renal_models.py`](../04_model_analysis/train_and_calibrate_renal_models.py) |
| `src/early_renal/phases/phase5_shap.py` | [`04_model_analysis/renal_shap_explainability.py`](../04_model_analysis/renal_shap_explainability.py) |
| `src/early_renal/phases/phase6_allocation.py` | [`05_icu_optimization/icu_allocation_simulation.py`](../05_icu_optimization/icu_allocation_simulation.py) |
| `src/early_renal/phases/phase8_feature_availability_audit.py` | [`02_feature_selection/feature_timestamp_availability_audit.py`](../02_feature_selection/feature_timestamp_availability_audit.py) |
| `src/early_renal/phases/phase9_history_omr.py` | [`01_data_preprocessing/patient_history_omr_features.py`](../01_data_preprocessing/patient_history_omr_features.py) |
| `src/early_renal/phases/phase9b_medications_comorbidities.py` | [`01_data_preprocessing/medication_comorbidity_features.py`](../01_data_preprocessing/medication_comorbidity_features.py) |
| `src/early_renal/phases/phase9c_lgbm_catboost_ablation.py` | [`03_model_training/Tree_Ensembles/clinical_feature_ablation_lgbm_catboost.py`](../03_model_training/Tree_Ensembles/clinical_feature_ablation_lgbm_catboost.py) |
| `src/hospital_wide/experiments/benchmark_no_labs.py` | [`03_model_training/Research_Archives/experiments/benchmark_no_labs.py`](../03_model_training/Research_Archives/experiments/benchmark_no_labs.py) |
| `src/hospital_wide/experiments/compare_renal_models.py` | [`04_model_analysis/compare_renal_models.py`](../04_model_analysis/compare_renal_models.py) |
| `src/hospital_wide/experiments/staged_labs.py` | [`01_data_preprocessing/staged_labs.py`](../01_data_preprocessing/staged_labs.py) |
| `src/hospital_wide/phases/phase18_unified_flow.py` | [`01_data_preprocessing/hospitalwide_event_cohort.py`](../01_data_preprocessing/hospitalwide_event_cohort.py) |
| `src/hospital_wide/phases/phase18b_feature_store.py` | [`01_data_preprocessing/hospitalwide_feature_store.py`](../01_data_preprocessing/hospitalwide_feature_store.py) |
| `src/hospital_wide/phases/phase18c_clinical_concepts.py` | [`01_data_preprocessing/clinical_concept_features.py`](../01_data_preprocessing/clinical_concept_features.py) |
| `src/hospital_wide/phases/phase19b_multimodel.py` | [`03_model_training/Tree_Ensembles/hospitalwide_multimodel_benchmark.py`](../03_model_training/Tree_Ensembles/hospitalwide_multimodel_benchmark.py) |
| `src/hospital_wide/phases/phase20_audit_ablation.py` | [`04_model_analysis/feature_leakage_ablation_audit.py`](../04_model_analysis/feature_leakage_ablation_audit.py) |
| `src/hospital_wide/phases/phase21_competing_risk.py` | [`03_model_training/Competing_Risks/hospitalwide_competing_risk_network.py`](../03_model_training/Competing_Risks/hospitalwide_competing_risk_network.py) |
| `src/hospital_wide/phases/phase21_renal_only.py` | [`03_model_training/Competing_Risks/renal_only_competing_risk_evaluation.py`](../03_model_training/Competing_Risks/renal_only_competing_risk_evaluation.py) |
| `src/hospital_wide/phases/phase22_validation.py` | [`04_model_analysis/competing_risk_bootstrap_validation.py`](../04_model_analysis/competing_risk_bootstrap_validation.py) |
| `src/hospital_wide/phases/phase22_validation_fixed.py` | [`04_model_analysis/competing_risk_patient_bootstrap_validation.py`](../04_model_analysis/competing_risk_patient_bootstrap_validation.py) |
| `src/hospital_wide/phases/phase22a_conservative_retrain.py` | [`03_model_training/Competing_Risks/conservative_feature_competing_risk_training.py`](../03_model_training/Competing_Risks/conservative_feature_competing_risk_training.py) |
| `src/hospital_wide/phases/phase22a_conservative_retrain_fixed.py` | [`03_model_training/Competing_Risks/conservative_feature_competing_risk_training_fixed.py`](../03_model_training/Competing_Risks/conservative_feature_competing_risk_training_fixed.py) |
| `src/hospital_wide/phases/phase22b_labs_retrain.py` | [`03_model_training/Competing_Risks/laboratory_feature_competing_risk_training.py`](../03_model_training/Competing_Risks/laboratory_feature_competing_risk_training.py) |
| `src/hospital_wide/phases/phase23_renal_benchmark.py` | [`03_model_training/Competing_Risks/renal_competing_risk_benchmark.py`](../03_model_training/Competing_Risks/renal_competing_risk_benchmark.py) |
| `src/hospital_wide/phases/phase23b_binary_renal_neural.py` | [`03_model_training/Neural_Networks/renal_binary_mlp_training.py`](../03_model_training/Neural_Networks/renal_binary_mlp_training.py) |
| `src/hospital_wide/phases/phase23c_hospitalwide_binary.py` | [`03_model_training/Neural_Networks/hospitalwide_binary_mlp_training.py`](../03_model_training/Neural_Networks/hospitalwide_binary_mlp_training.py) |
| `src/hospital_wide/phases/phase24_feature_reduction.py` | [`02_feature_selection/feature_budget_ablation.py`](../02_feature_selection/feature_budget_ablation.py) |
| `src/hospital_wide/phases/phase25_feature_audit.py` | [`04_model_analysis/clinical_feature_provenance_audit.py`](../04_model_analysis/clinical_feature_provenance_audit.py) |
| `src/hospital_wide/phases/phase26_shap_engineering.py` | [`02_feature_selection/shap_ranked_feature_selection.py`](../02_feature_selection/shap_ranked_feature_selection.py) |
| `src/hospital_wide/phases/phase27_renal_tuning.py` | [`03_model_training/Neural_Networks/renal_neural_hyperparameter_tuning.py`](../03_model_training/Neural_Networks/renal_neural_hyperparameter_tuning.py) |
| `src/hospital_wide/phases/phase28_optuna_trees.py` | [`03_model_training/Tree_Ensembles/optuna_tree_model_tuning.py`](../03_model_training/Tree_Ensembles/optuna_tree_model_tuning.py) |
