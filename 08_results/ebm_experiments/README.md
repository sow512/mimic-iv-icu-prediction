# EBM experiment archive

This collection records the Explainable Boosting Machine (EBM) experiments supplied in the EBM archive. Training scripts are indexed in [03_model_training/EBM](../../03_model_training/EBM/README.md); aggregate results and evaluation figures belong here. Original experiment folder names are retained in the archive layout.

## Included experiments

| Experiment | Description |
|---|---|
| `EBM_20features` | EBM using the 20-feature dataset |
| `EBM_75` | Baseline 75-feature EBM |
| `EBM_75_Tuned` | Hyperparameter-tuned 75-feature EBM |
| `EBM_75_SMOTE(Tuned)` | Tuned 75-feature EBM with SMOTE/SMOTENC |
| `EBM_200` | EBM using the 200-feature dataset |
| `EBM_SMOTE_200` | 200-feature baseline vs SMOTE ratios 0.2, 0.5, and 1.0 |
| `EBM_SMOTE75` | Patient-window cohort audit, feature selection, and SMOTE comparison |

## Aggregate metric summaries

A consolidated CSV is provided in [ebm_test_metrics_summary.csv](ebm_test_metrics_summary.csv). It preserves each experiment's reported test metrics but does **not** establish a matched leaderboard. Cohort size, prevalence, outcome construction, feature set, data split, threshold-selection procedure, and code version differ between experiments.

### Interpretation notes

- Several 75-feature runs explicitly report `window_definition: unverified`. Their results must not be described as verified 24–72-hour forecasts without auditing target construction and feature availability.
- The 200-feature baseline code warns that the upstream 24–72-hour target definition must be verified.
- The 20-feature and 75-feature runs have differing test-set sizes and event prevalence.
- SMOTE was applied during training in the relevant scripts. Validation/test data should retain their original prevalence.
- The 75-feature SMOTE experiment reports lower AUROC/AUPRC and more false positives than its non-SMOTE baseline; oversampling did not improve the reported overall discrimination in that run.
- The 200-feature SMOTE comparison likewise reports the strongest AUROC/AUPRC and precision for its baseline (no SMOTE) row.
- The feature-window cohort audit includes 427,204 verified windows and reports zero label mismatches in that audit. This is an internal consistency check, not independent external validation.
- AKI/CKD diagnosis flags can contain retrospective information; the associated script excludes them from predictors by default. A full feature-availability audit remains necessary.
- ICU transfer is an observed care-utilization outcome, not necessarily unmet clinical need for ICU admission.

## Privacy and artifact policy

The repository intentionally excludes row-level MIMIC-IV-derived datasets, feature-window exports, patient/admission-linked prediction outputs, and serialized fitted models. These files may contain patient-level clinical information or enable reconstruction of patient-level records. Only aggregate metrics, feature-level summaries, cohort-level counts, documentation, and figures are appropriate for version control. Keep restricted data and fitted model artifacts in authorized local storage.

## Reproduction

The scripts are historical experiment sources, not a turnkey pipeline. They require the corresponding MIMIC-IV-derived input files, correct local paths, and compatible Python dependencies. Review each script's command-line arguments and target-label assumptions before running. Some older scripts use different input formats and target columns; they are not interchangeable.

These are retrospective research experiments, not a deployed or clinically validated decision-support system.
