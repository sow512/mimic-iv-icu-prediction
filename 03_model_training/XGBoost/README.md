
# XGBoost — ICU Transfer Prediction

This folder contains XGBoost model training experiments for predicting ICU transfer in kidney-disease patients using the MIMIC-IV dataset.

## Feature Configurations

| Feature Set | Experiments |
|---|---|
| **20 Features** | Baseline, class-weighted XGBoost and SMOTENC |
| **44 Features** | Baseline XGBoost vs SMOTENC |
| **75 Features** | Final XGBoost training, baseline vs SMOTENC and hyperparameter tuning |
| **200 Features** | Baseline XGBoost vs SMOTENC |

## Methodology

- **Feature Selection:** VIF/RFE for compact feature sets and SHAP-based selection for expanded feature sets.
- **Class Imbalance:** Comparison of baseline training, class weighting and SMOTE/SMOTENC.
- **Evaluation:** AUROC, AUPRC, precision, recall, F1-score, specificity and confusion matrices.
- **Threshold Selection:** Validation-based threshold selection targeting approximately 70% recall.

## Final Model

The **75-feature XGBoost model (Step 53)** was selected for integration into the Streamlit ICU-transfer prediction and resource-planning dashboard.

The model incorporates laboratory measurements and engineered temporal features to estimate ICU-transfer risk during the 24–72-hour prediction window.

## Important Notes

- The 44-feature experiments use repeated prediction-window observations and are not directly comparable with admission-level datasets.
- SMOTE experiments are exploratory and do not replace the original final model.
- Patient-level data and restricted MIMIC-IV files are not included in this repository.
- This project is intended for research and demonstration purposes, not clinical decision-making.
