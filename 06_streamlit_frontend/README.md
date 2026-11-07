# Streamlit Frontend — ICU Transfer Prediction & Resource Planning

This folder contains the final Streamlit dashboard developed for predicting ICU transfers among kidney-disease patients using the MIMIC-IV dataset.

## Application Features

- **ICU Transfer Prediction:** Uses the final 75-feature XGBoost model to estimate ICU-transfer risk within 24–72 hours after hospital admission.
- **Risk Stratification:** Ranks patients based on predicted ICU-transfer probabilities.
- **SHAP Explainability:** Identifies the major features influencing individual patient predictions.
- **ICU Crisis Management:** Simulates bed availability, demand surges, waiting-time constraints and risk-prioritized allocation.
- **Resource Planning:** Evaluates potential ICU shortages under different hypothetical capacity scenarios.

## Files

| File | Description |
|---|---|
| `app.py` | Final Streamlit application integrating prediction, explainability and ICU capacity simulation. |
| `requirements.txt` | Python dependencies required to run the application. |

## Running the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the dashboard:

```bash
streamlit run 06_streamlit_frontend/app.py
```

The application requires the frozen Step 53 XGBoost model, preprocessing bundle and metadata files, which must be supplied locally by authorized users.

## Important Notes

- Patient-level MIMIC-IV data and demonstration datasets are not included in this repository.
- Risk-based ranking reflects predicted ICU-transfer probability, not clinical urgency.
- ICU simulations and capacity alerts are hypothetical and intended for research purposes only.
- The application is not validated for real-world clinical decision-making.
