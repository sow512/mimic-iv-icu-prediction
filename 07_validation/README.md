
# Model & Simulation Validation

This folder contains validation scripts used to assess data integrity, ICU-transfer prediction consistency, and ICU resource simulation reliability.

## Included Scripts

| File | Description |
|---|---|
| `data_leakage_cohort_validation.py` | Checks cohort consistency, duplicate records, patient-level splits, and potential data leakage. |
| `icu_outcome_dataset_validation.py` | Verifies ICU-transfer labels, prediction-window timing, and 75-feature dataset consistency. |
| `icu_outcome_consistency_audit.py` | Investigates outcome-label inconsistencies and early-discharge cases. |
| `streamlit_prediction_verification.py` | Compares Streamlit predictions against the saved XGBoost model and reference predictions. |
| `model_simulation_verification.py` | Verifies model predictions and independently tests ICU simulation constraints. |
| `icu_simulation_validation.py` | Tests the actual ICU simulation function for capacity limits, request accounting, waiting-time behavior, and reproducibility. |

## Final Verification Results

- **Step 73 — PASS:** Model probabilities and classification-threshold flags matched the reference predictions. Independent simulation checks also passed.
- **Step 74 — PASS:** The actual ICU simulation function passed all tested capacity, request-accounting, and reproducibility checks.

## Limitations

- Outcome-timing inconsistencies and early-discharge cases require further review.
- Independent simulation validation does not establish clinical effectiveness.
- Priority tie-breaking and separate random-allocation implementations have not been fully verified.
- Model validation does not establish generalizability to other hospitals or patient populations.

## Data Privacy

Patient-level MIMIC-IV records, restricted datasets, and private identifiers are excluded from this repository.

**Note:** This project is intended for academic research and demonstration purposes only, not clinical decision-making.
