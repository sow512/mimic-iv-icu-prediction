# Streamlit research demo — ICU transfer and capacity planning

The entry point is **[app.py](app.py)** (not `staylength.py` or `icu_crisis_app.py`). It loads a previously fitted 75-feature XGBoost model and preprocessing bundle; it does **not** train a model.

## Run locally

```bash
pip install -r 06_streamlit_frontend/requirements.txt
streamlit run 06_streamlit_frontend/app.py
```

The app requires `final_xgboost_75_model.json`, `final_inference_bundle.joblib`, and `final_model_metadata.json` from an authorized local model-artifact directory. Configure `ICU_MODEL_DIR` as needed. These artifacts are **not included in GitHub**, so a public deployment will not function until the matching artifacts and dependencies are supplied.

Features include risk ranking, explainability, and hypothetical ICU capacity simulations. Use **synthetic or approved non-identifiable demonstration inputs only**. Predictions represent retrospective observed transfer risk, not clinical urgency or medical advice. Not validated for clinical use.
