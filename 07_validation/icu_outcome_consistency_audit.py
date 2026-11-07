
# ============================================================
# STEP 57: OUTCOME CONSISTENCY AND EARLY DISCHARGE POLICY
# Read-only audit. No labels or models overwritten.
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT = Path(
    r"C:\Users\sowmy\Downloads\MIMIC-IV\MIMIC_IV\MIMIC_IV"
)

SOURCE = PROJECT / "model training figures" / "expanded_lab_xgboost"
OUT = SOURCE / "step57_outcome_policy"
OUT.mkdir(parents=True, exist_ok=True)

ADM_FILE = (
    PROJECT / "physionet.org/files/mimiciv/3.1/hosp/admissions.csv"
)
ICU_FILE = (
    PROJECT / "physionet.org/files/mimiciv/3.1/icu/icustays.csv"
)
COHORT_FILE = SOURCE / "cohort_audit.csv"

print("STEP 57: Loading original clinical records...")

admissions = pd.read_csv(
    ADM_FILE,
    usecols=[
        "subject_id", "hadm_id", "admittime", "dischtime"
    ],
    parse_dates=["admittime", "dischtime"]
)

icu = pd.read_csv(
    ICU_FILE,
    usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    parse_dates=["intime", "outtime"]
)

cohort = pd.read_csv(
    COHORT_FILE,
    usecols=["subject_id", "hadm_id", "icu_transfer"]
)

if cohort.hadm_id.duplicated().any():
    raise ValueError("Duplicate cohort admission IDs")

if admissions.hadm_id.duplicated().any():
    raise ValueError("Duplicate admission IDs")

# ============================================================
# 1. RECONSTRUCT HOSPITAL AND ICU TIMING
# ============================================================

audit = cohort.merge(
    admissions,
    on=["subject_id", "hadm_id"],
    how="left",
    validate="one_to_one"
)

if audit.admittime.isna().any():
    raise ValueError("Missing admission times")

audit["prediction_time"] = (
    audit.admittime + pd.Timedelta(hours=24)
)

audit["window_end"] = (
    audit.admittime + pd.Timedelta(hours=72)
)

# Keep every ICU stay to investigate timing anomalies.
icu = icu.dropna(subset=["hadm_id", "intime"])

first_icu = (
    icu.groupby("hadm_id", as_index=False)["intime"]
    .min()
    .rename(columns={"intime": "first_icu_time"})
)

audit = audit.merge(
    first_icu,
    on="hadm_id",
    how="left",
    validate="one_to_one"
)

# ============================================================
# 2. IDENTIFY OUTCOME AND TIMING ISSUES
# ============================================================

audit["positive_after_discharge"] = (
    audit.icu_transfer.eq(1)
    & audit.first_icu_time.notna()
    & audit.dischtime.notna()
    & audit.first_icu_time.gt(audit.dischtime)
)

audit["early_discharge_negative"] = (
    audit.icu_transfer.eq(0)
    & audit.dischtime.notna()
    & audit.dischtime.gt(audit.prediction_time)
    & audit.dischtime.lt(audit.window_end)
)

audit["transfer_in_window"] = (
    audit.first_icu_time.gt(audit.prediction_time)
    & audit.first_icu_time.le(audit.window_end)
)

audit["label_mismatch"] = (
    audit.icu_transfer.astype(bool)
    != audit.transfer_in_window
)

audit["hours_until_discharge"] = (
    (audit.dischtime - audit.admittime)
    .dt.total_seconds() / 3600
)

audit["hours_until_icu"] = (
    (audit.first_icu_time - audit.admittime)
    .dt.total_seconds() / 3600
)

audit["icu_after_discharge_hours"] = (
    (audit.first_icu_time - audit.dischtime)
    .dt.total_seconds() / 3600
)

# ============================================================
# 3. REVIEW FOUR INCONSISTENT POSITIVE ADMISSIONS
# ============================================================

inconsistent = audit.loc[
    audit.positive_after_discharge
].copy()

print("\nINCONSISTENT POSITIVE ADMISSIONS:", len(inconsistent))

review_columns = [
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
    "prediction_time",
    "window_end",
    "first_icu_time",
    "hours_until_icu",
    "icu_after_discharge_hours",
    "icu_transfer"
]

inconsistent[review_columns].to_csv(
    OUT / "four_inconsistent_admissions.csv",
    index=False
)

# Retrieve all ICU stay records associated with these admissions.
icu_details = icu[
    icu.hadm_id.isin(inconsistent.hadm_id)
].copy()

icu_details.to_csv(
    OUT / "inconsistent_icu_stay_details.csv",
    index=False
)

# Create manual review template.
manual_review = inconsistent[review_columns].copy()

manual_review["review_status"] = "PENDING"
manual_review["possible_explanation"] = ""
manual_review["reviewer_decision"] = ""
manual_review["reviewer_notes"] = ""

manual_review.to_csv(
    OUT / "manual_review_template.csv",
    index=False
)

# ============================================================
# 4. EARLY DISCHARGE ANALYSIS
# ============================================================

early = audit.loc[
    audit.early_discharge_negative
].copy()

print("Early-discharge negative admissions:", len(early))

early["discharge_interval"] = pd.cut(
    early.hours_until_discharge,
    bins=[24, 36, 48, 60, 72],
    labels=["24-36h", "36-48h", "48-60h", "60-72h"],
    include_lowest=True
)

early_summary = (
    early.groupby("discharge_interval", observed=False)
    .size()
    .reset_index(name="admissions")
)

early_summary.to_csv(
    OUT / "early_discharge_summary.csv",
    index=False
)

early[
    ["subject_id", "hadm_id", "hours_until_discharge"]
].to_csv(
    OUT / "early_discharge_admissions.csv",
    index=False
)

# ============================================================
# 5. DEFINE OUTCOME POLICY
# ============================================================

# Policy:
# - Prediction made at hour 24.
# - Positive = observed in-hospital ICU entry during hours (24,72].
# - Early discharge without ICU transfer = observed negative
#   for the in-hospital transfer outcome.
# - Such admissions have shorter hospital observation.
# - Inconsistent positive transfers require manual review.
# - A sensitivity analysis can exclude early discharges,
#   but must be reported as a different selected cohort.

policy = """# Group 28 - ICU Transfer Outcome Policy

## Prediction landmark
Predictions are made 24 hours after hospital admission.

## Positive outcome
An observed ICU transfer occurring after hour 24 and
no later than hour 72 of the hospital admission.

## Negative outcome
No observed in-hospital ICU transfer within the
specified prediction window.

## Early hospital discharge
Admissions discharged between hours 24 and 72 without
a recorded ICU transfer are retained as negative under
the observed in-hospital ICU-transfer definition.

These admissions have a shorter period at risk of an
in-hospital transfer. They must not be interpreted as
having a fully observed 72-hour hospitalization.

A sensitivity analysis excluding early discharges may
be performed, but this changes the study population and
can introduce selection bias.

## Inconsistent ICU timestamps
Positive admissions with ICU entry after discharge are
flagged for manual investigation.

Do not automatically change their labels.
Review the admission and ICU stay records first.

## Important limitations
- ICU transfer reflects both clinical deterioration and
  hospital admission/bed-management decisions.
- Discharge and death are competing events.
- ICD-based AKI/CKD selection may rely on retrospective
  discharge diagnoses.
- Earlier models were trained using the original labels.
  Any changed cohort or labels require renewed evaluation.
- Existing test results are exploratory because the test
  set has already been examined repeatedly.
"""

(OUT / "outcome_definition_policy.md").write_text(
    policy,
    encoding="utf-8"
)

# ============================================================
# 6. SAVE AUDIT SUMMARY
# ============================================================

summary = {
    "total_admissions": int(len(audit)),
    "positive_admissions": int(audit.icu_transfer.sum()),
    "positive_after_discharge": int(len(inconsistent)),
    "early_discharge_negatives": int(len(early)),
    "label_mismatches": int(audit.label_mismatch.sum()),
    "outcome_policy": "observed_in_hospital_icu_transfer_24_to_72h",
    "inconsistent_records_status": "manual_review_pending",
    "model_retrained": False,
    "labels_modified": False
}

with open(
    OUT / "step57_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\nEARLY DISCHARGE BREAKDOWN")
print(early_summary.to_string(index=False))

print("\n" + "=" * 60)
print("STEP 57 COMPLETE")
print("=" * 60)

print(json.dumps(summary, indent=2))
print("\nResults saved to:", OUT)
print("No datasets, labels, or models overwritten.")
