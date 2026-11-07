# Predicting ICU Transfer from Hospital Data

### A MIMIC-IV research project exploring early warning, renal-risk modeling, and dynamic clinical prediction

This repository contains a series of retrospective machine-learning experiments investigating whether routinely collected hospital information can help estimate the risk of **subsequent intensive care unit (ICU) transfer**. The work began with a renal-focused cohort and expanded into hospital-wide modeling, alternative outcome definitions, competing-risk approaches, and repeated predictions at clinical landmarks.

The central question is straightforward: **what can be predicted using only information that would have been available at the time of prediction?** Answering it requires more than fitting a classifier. The experiments also examine cohort eligibility, temporal feature availability, patient-level data separation, calibration, and the impact of discharge and death on outcome interpretation.

> **Research status:** This is an experimental code archive, **not** a deployed clinical tool or a fully reproduced benchmark. The scripts have not been validated end to end in their reorganized layout. No clinical use is intended.

## Research at a glance

| Dimension | Scope |
| --- | --- |
| Data source | MIMIC-IV (access-controlled clinical data) |
| Prediction task | Future ICU transfer following a hospital-based prediction landmark |
| Initial focus | Retrospectively identified renal-risk population |
| Expanded scope | Hospital-wide and renal-subgroup experiments |
| Approaches explored | Tree-based ML, neural baselines, tuning, ablation, competing risks, dynamic landmarks |
| Evaluation explored | AUROC, AUPRC, calibration, subgroup analysis, explainability |
| Deliverable | Research scripts, a walkthrough notebook, and experiment documentation |

## Why this problem matters

Clinical deterioration can occur after hospital admission, before an ICU transfer takes place. Risk estimates may be useful to study how early warning systems could support prioritization and review. But retrospective predictive performance alone does **not** establish clinical usefulness: predictions must be available at the right time, calibrated for the intended population, and evaluated under realistic deployment conditions.

This project therefore treats **time, cohort definition, and outcome definition** as first-class modeling decisions.

## Research workflow

```text
MIMIC-IV (authorized local access)
             |
             v
   Cohort and landmark definition
             |
             v
  Time-constrained feature assembly
  (labs, history, medications, etc.)
             |
             v
     Patient-disjoint data splits
             |
             v
  Model training and comparison
             |
             v
 Discrimination / calibration / subgroups
             |
             v
  Exploratory extensions:
  competing risks + dynamic landmarks
```

This diagram summarizes the research themes; it is **not** a single tested command-line pipeline.

## What was investigated

### 1. Early renal-focused prediction

The initial experiments construct a retrospective renal cohort and study ICU transfer after a fixed prediction landmark. They cover cohort assembly, time-bounded laboratory features, model fitting, calibration, feature selection, explainability, and subgroup assessment.

A key methodological caveat is that a cohort defined using hospital diagnosis codes can incorporate diagnoses documented **after** the prediction landmark. That makes the original renal cohort suitable for retrospective investigation but not automatically equivalent to a prospectively identifiable screening population.

### 2. Broader cohorts and model comparisons

Later experiments extend beyond the initial renal cohort, investigate additional clinical features, compare model families, and explore tuning and feature ablation. These scripts do not necessarily share identical eligibility criteria, labels, or splits; results from different phases should **not** be compared as though they came from a single benchmark.

### 3. Competing events and dynamic prediction

Discharge and death can prevent an ICU transfer from being observed within a prediction window. The repository includes exploratory competing-risk formulations alongside binary models. It also includes a dynamic renal pilot that updates predictions at multiple landmarks.

These formulations answer related but **different** statistical questions. Repeated predictions from the same patient also require appropriate handling when estimating uncertainty.

## Repository map

The numbered directories are the canonical team workflow. Research scripts have been moved into these folders alongside existing contributions; `src/` now contains navigation and a complete [old-to-new path map](src/MOVE_MAP.md).

| Directory | Contents |
|---|---|
| [01_data_preprocessing](01_data_preprocessing/) | Team preprocessing and renal research cohort/feature builders |
| [02_feature_selection](02_feature_selection/) | Team feature selection and renal feature-selection experiments |
| [03_model_training](03_model_training/) | XGBoost, tree ensembles, neural networks, competing risks, dynamic models, EBM experiments, and archived research experiments |
| [04_model_analysis](04_model_analysis/) | Calibration, validation, explainability, and ablation scripts |
| [05_icu_optimization](05_icu_optimization/) | ICU allocation and simulation work |
| [06_streamlit_frontend](06_streamlit_frontend/) | Interactive application |
| [07_validation](07_validation/) | Team verification and validation scripts |
| [08_results](08_results/) | Non-sensitive aggregate results, including the EBM experiment archive |
| [src](src/) | Navigation index and file move map; no duplicate model source |
| [notebooks](notebooks/) | Research walkthrough notebook (historical paths may need updating) |

**Where to start:** See [src/MOVE_MAP.md](src/MOVE_MAP.md) for each original phase script and its canonical location. The EBM experiment scripts are indexed in [03_model_training/EBM](03_model_training/EBM/README.md), with aggregate summaries in [08_results/ebm_experiments](08_results/ebm_experiments/README.md). The original notebook remains a historical walkthrough and is not a verified runnable pipeline.

## Local data configuration

Historical scripts use external, authorized MIMIC-IV data. Set `MIMIC_PROJECT_ROOT` to your local research workspace for the legacy XGBoost, SHAP, RFECV and validation scripts. Set `ICU_75_FEATURE_WORKBOOK` to the 75-feature Excel workbook for the standalone SMOTE experiment. The SVM/MLP notebooks use `ICU_DATA_DIR` for their input spreadsheets. No personal machine paths or restricted data are stored in the current source tree. Some older scripts still assume their original generated subfolder structure, and runtime compatibility is not yet guaranteed.

## Getting started

### Prerequisites

- Python and a suitable local environment for scientific computing
- Authorized access to the required MIMIC-IV tables and any derived inputs
- Sufficient storage and compute resources for the experiments you choose to run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency list is an **approximate union of imports**, not a pinned, tested environment. Some scripts may need additional packages or compatible versions.

To inspect an individual script's interface:

```bash
python 01_data_preprocessing/renal_cohort_24h_landmark.py --help
```

Not every historical script is guaranteed to expose the same CLI or run independently. Some still assume legacy paths, filenames, working directories, or sibling scripts. Review its source and required input tables before running it. A syntax-only check can be performed with:

```bash
python -m compileall -q 01_data_preprocessing 02_feature_selection 03_model_training 04_model_analysis 05_icu_optimization 06_streamlit_frontend 07_validation tools
```

**Passing compilation does not mean the experiments run successfully.** A GitHub Actions static-check workflow compiles Python sources and checks the research move map. It does not execute experiments or validate MIMIC-IV data access.


## Results and figures

Browse the [curated aggregate results and nine-figure gallery](08_results/README.md), including model discrimination, calibration, SHAP feature attribution, hypothetical event capture, and Phase 30 neural-network diagnostics. The EBM experiment archive is separately indexed in [08_results/ebm_experiments](08_results/ebm_experiments/README.md). Figures come from different experiments and should not be interpreted as a single matched leaderboard.

## Data access and privacy

MIMIC-IV is a controlled-access clinical dataset. **No patient-level data, feature tables, labels, trained model artifacts, or prediction exports are included in this repository.** Obtain access through the dataset's official process and follow its data-use conditions. Keep restricted inputs and outputs outside version control and in approved storage.

## Evaluation and reproducibility

The code explores discrimination (including AUROC and AUPRC), calibration, explainability, and subgroup analyses. **This README intentionally reports no headline performance number:** the archive does not contain a single verified final evaluation with a unified cohort, label policy, and held-out test protocol.

Before treating any experiment as a reproducible research result, the following need to be established:

1. A prospectively identifiable prediction population and clearly defined landmark.
2. A harmonized outcome definition, including treatment of discharge and death.
3. Verification that all predictor values were available by prediction time.
4. Fixed patient-disjoint development and evaluation partitions, with appropriate uncertainty estimates.
5. A pinned environment, executable end-to-end workflow, and automated tests.
6. Independent temporal or external validation before any clinical interpretation.

## Known limitations

- **Potential future-informed cohort selection:** Retrospective renal diagnosis codes may reflect information acquired after the landmark.
- **Outcome heterogeneity:** Some experiments exclude early competing events; others handle them differently.
- **Temporal availability:** Laboratory timestamps receive explicit attention in parts of the code, but all feature families need a consistent availability audit.
- **Multiple experiments:** Cohorts, splits, and model-selection decisions vary across phases; the archive does not designate one final model.
- **Validation:** There is no demonstrated external validation or completed end-to-end reproduction of the reorganized repository.
- **Research-only status:** Predictive metrics, even if reproduced, would not establish clinical safety or utility.

## Project history

The repository preserves a chronological reconstruction of research milestones based on file modification metadata from an earlier archive. **Historical author dates are estimates of file development activity; the Git commits themselves were created during repository reconstruction.** The later organization and documentation changes are separate from the original experiments.

## Acknowledgment

This work uses the MIMIC-IV research ecosystem. Users should consult the official MIMIC-IV documentation for access requirements, dataset versions, and citation instructions. This repository is independent research code and is not affiliated with the dataset maintainers.

---

**Status:** Exploratory retrospective clinical ML research · **Intended audience:** Researchers and technical reviewers · **Not for clinical decision-making**
