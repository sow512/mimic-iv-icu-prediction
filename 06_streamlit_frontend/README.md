# Streamlit Frontend — ICU Transfer Prediction & Resource Planning

This folder contains the research Streamlit dashboard developed for predicting ICU transfers among kidney-disease patients using the MIMIC-IV dataset.

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

## File index

| File | Description |
|---|---|
| [`app.py`](app.py) | app (python research script) |
| [`final_inference_bundle.joblib`](final_inference_bundle.joblib) | final inference bundle.joblib (supporting file) |
| [`final_model_metadata.json`](final_model_metadata.json) | final model metadata.json (supporting file) |
| [`final_xgboost_75_model.json`](final_xgboost_75_model.json) | final xgboost 75 model.json (supporting file) |
| [`requirements.txt`](requirements.txt) | requirements (feature list) |
## Live Streamlit Application

🔗[ [Launch ICU Transfer Prediction & Crisis Capacity Management](https://58vgelnnmdyhrdcosbvtga.streamlit.app/)](https://mhhaqtqmgxdhhfga7f5a6k.streamlit.app/)

The application demonstrates:
- ICU-transfer risk prediction using the final 75-feature XGBoost model
- Patient-level SHAP explainability
- ICU capacity stress testing and risk-based prioritization

**Note:** This is a research demonstration, not a clinical decision-support tool.
## Reproducibility status

This repository documents retrospective experiments. Access-controlled MIMIC-IV datasets and some fitted model artifacts are not included. The automated checks validate syntax and repository structure, not end-to-end reproduction or clinical safety. Different cohort definitions and evaluation splits must not be compared as a single leaderboard.

