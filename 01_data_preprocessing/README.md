# Data preprocessing and feature engineering

This folder contains the team's initial MIMIC-IV preprocessing notebook and additional research scripts for renal and hospital-wide cohorts. The notebook [mimic_iv_cohort_preprocessing_feature_engineering.ipynb](mimic_iv_cohort_preprocessing_feature_engineering.ipynb) documents the midterm workflow; the Python scripts cover 24-hour landmarks, laboratory features, clinical history, medication/comorbidity features, and cohort audits.

**Important:** These are related but not necessarily identical cohorts. Do not combine their output tables without checking labels, timestamps, and patient-disjoint splits. MIMIC-IV data is not included.

## File index

| File | Description |
|---|---|
| [`clinical_concept_features.py`](clinical_concept_features.py) | clinical concept features (python research script) |
| [`hospitalwide_event_cohort.py`](hospitalwide_event_cohort.py) | hospitalwide event cohort (python research script) |
| [`hospitalwide_feature_store.py`](hospitalwide_feature_store.py) | hospitalwide feature store (python research script) |
| [`medication_comorbidity_features.py`](medication_comorbidity_features.py) | medication comorbidity features (python research script) |
| [`mimic_iv_cohort_preprocessing_feature_engineering.ipynb`](mimic_iv_cohort_preprocessing_feature_engineering.ipynb) | mimic iv cohort preprocessing feature engineering (research notebook) |
| [`patient_flow_cohort_audit.py`](patient_flow_cohort_audit.py) | patient flow cohort audit (python research script) |
| [`patient_history_omr_features.py`](patient_history_omr_features.py) | patient history omr features (python research script) |
| [`renal_cohort_24h_landmark.py`](renal_cohort_24h_landmark.py) | renal cohort 24h landmark (python research script) |
| [`staged_labs.py`](staged_labs.py) | staged labs (python research script) |
| [`temporal_laboratory_features.py`](temporal_laboratory_features.py) | temporal laboratory features (python research script) |

## Reproducibility status

This repository documents retrospective experiments. Access-controlled MIMIC-IV datasets and some fitted model artifacts are not included. The automated checks validate syntax and repository structure, not end-to-end reproduction or clinical safety. Different cohort definitions and evaluation splits must not be compared as a single leaderboard.

