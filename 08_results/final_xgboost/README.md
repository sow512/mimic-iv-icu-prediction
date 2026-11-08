# Final 75-Feature XGBoost Results

This folder contains the final evaluation results of the selected 75-feature XGBoost model for predicting ICU transfer within 24–72 hours among hospitalized AKI/CKD patients.

## Included Results

- **ROC Curve:** Evaluates discrimination between ICU-transfer and non-transfer cases.
- **Precision–Recall Curve:** Evaluates performance under severe class imbalance.
- **Confusion Matrix:** Shows true positives, false positives, true negatives and false negatives.
- **Calibration Curve:** Examines agreement between predicted probabilities and observed ICU-transfer frequencies.
- **Test Metrics:** Reports the final model's evaluation metrics, including AUROC, AUPRC, recall, precision, specificity and F1-score.

## Model Selection

The 75-feature XGBoost model was selected for integration into the Streamlit prediction and ICU resource-planning prototype.

The model uses SHAP-selected predictors and a validation-selected classification threshold.

## Limitations

The model was evaluated retrospectively using MIMIC-IV data. Its predictions have not been externally validated, and the results should not be interpreted as evidence of clinical effectiveness.
