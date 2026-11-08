# Artifact and duplication audit

No model artifacts have been deleted or relocated in this cleanup. Paths may be referenced by the deployed Streamlit app or historical scripts.

## Potential duplicate paths requiring verification

- `06_streamlit_frontend/final_xgboost_75_model.json`
- `06_streamlit_frontend/model/final_xgboost_75_model.json`
- `06_streamlit_frontend/final_model_metadata.json`
- `06_streamlit_frontend/model/final_model_metadata.json`
- `06_streamlit_frontend/final_inference_bundle.joblib`
- `06_streamlit_frontend/model/final_preprocessing_pipeline.joblib`

Matching filenames do **not** establish matching content. Compare hashes, then inspect `app.py`, deployment configuration and all references before removing or moving any file.

## Safety and hygiene

- `.DS_Store` is macOS-generated metadata and should not be tracked.
- Patient-level MIMIC-IV records, per-patient predictions, and derived restricted tables must remain outside Git.
- Check tracked artifacts separately from `.gitignore`: ignore rules do not remove files already committed.
- Large model JSON files should be reviewed for provenance and deployment dependency, not automatically deleted.
- Preserve historical results with clear labels rather than silently replacing them.
