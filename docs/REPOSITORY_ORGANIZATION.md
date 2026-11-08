# Repository organization and maintenance

This repository is organized by **research workstream**, not contributor. Preserve Git history for attribution; do not move files solely to identify their author.

## Canonical workstreams

| Path | Purpose |
| --- | --- |
| `01_data_preprocessing/` | Cohorts, landmarks, temporal feature construction |
| `02_feature_selection/` | Feature ranking, ablations, timestamp audits |
| `03_model_training/` | Training grouped by model family |
| `04_model_analysis/` | Calibration, subgroup comparisons, explainability |
| `05_icu_optimization/` | Hypothetical bed-allocation simulations |
| `06_streamlit_frontend/` | Research demonstration and inference assets |
| `07_validation/` | Consistency, leakage, model and simulation checks |
| `08_results/` | Aggregate results, figures and phase-specific summaries |
| `notebooks/` | Historical research notebooks |
| `src/`, `tools/` | Shared utilities and maintenance scripts |

## Model training map

- `XGBoost/`, `Random_Forest/`, `EBM/`, `SVM_MLP/`: team experiments and feature-budget comparisons.
- `Tree_Ensembles/`: LightGBM, CatBoost, weighted blending, clinical ablations.
- `Neural_Networks/`: binary MLP and neural hyperparameter tuning.
- `Competing_Risks/`: linear and neural discrete-time competing-event models.
- `Dynamic_Models/`: repeated renal prediction landmarks.
- `Research_Archives/`: historical/reference implementations; not necessarily maintained entry points.

## Rules for future contributions

1. Put new training scripts in the relevant model-family directory. Use descriptive names, not only phase numbers.
2. Add an experiment record to `docs/EXPERIMENT_INDEX.md` with cohort, landmark, feature policy, split, and evaluation status.
3. Put **aggregate** metrics and de-identified figures under `08_results/`; never commit MIMIC-IV patient-level data or per-admission predictions.
4. Never compare AUPRC across different event prevalences as though it were a matched benchmark.
5. Preserve historical scripts until imports, command paths, notebooks and CI references have been audited.
6. Prefer a feature branch and pull request. Do not commit directly to `main` during reorganization.
7. Keep the Streamlit app's model artifacts in their current locations until all references and deployment settings are verified.

## Known cleanup backlog

- Verify duplicated Streamlit model/metadata artifacts before consolidation.
- Review historical notebooks and scripts for hardcoded paths.
- Review large figures/results and archive outdated exports only after provenance checks.
- Confirm the status of incomplete simulation placeholders.
- Add `phase31` records only when source outputs or scripts are available.
- Automated syntax checks are not end-to-end reproduction or clinical validation.
