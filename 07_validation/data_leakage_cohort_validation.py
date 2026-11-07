
# ============================================================
# STEP 49: COHORT, FEATURE AND DATA LEAKAGE AUDIT
# ============================================================

import os
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

PROJECT = Path(
    os.environ["MIMIC_PROJECT_ROOT"]
)

SOURCE = (
    PROJECT / "model training figures"
    / "expanded_lab_xgboost"
)

MODEL_DIR = SOURCE / "step44_final_200_model"
OUT = SOURCE / "step49_data_audit"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("STEP 49: DATA LEAKAGE AND COHORT VALIDATION")
print("=" * 65)

# ============================================================
# 1. LOAD EXISTING DATA
# ============================================================

cohort = pd.read_csv(
    SOURCE / "cohort_audit.csv"
)

features = pd.read_parquet(
    SOURCE / "expanded_candidate_features.parquet"
)

bundle = joblib.load(
    MODEL_DIR / "final_inference_bundle.joblib"
)

with open(
    MODEL_DIR / "final_model_metadata.json",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

selected_features = list(
    bundle["selected_feature_names"]
)

checks = []
warnings = []

def record_check(name, passed, detail):
    checks.append({
        "check": name,
        "passed": bool(passed),
        "detail": str(detail)
    })
    print(
        f"{'PASS' if passed else 'CHECK'}: "
        f"{name} | {detail}"
    )

# ============================================================
# 2. COHORT SIZE AND CLASS DISTRIBUTION
# ============================================================

print("\n1. COHORT SUMMARY")

print("Admissions:", len(cohort))
print("Unique patients:", cohort.subject_id.nunique())

positive = int(cohort.icu_transfer.sum())
negative = len(cohort) - positive

print("ICU transfers:", positive)
print("Non-ICU:", negative)
print("Positive prevalence:", round(positive / len(cohort), 4))

record_check(
    "Binary target",
    set(cohort.icu_transfer.dropna().unique()).issubset({0, 1})
    and cohort.icu_transfer.notna().all(),
    "icu_transfer must contain only 0 and 1"
)

# ============================================================
# 3. DUPLICATE ADMISSION CHECK
# ============================================================

print("\n2. DUPLICATE CHECK")

cohort_duplicates = int(
    cohort.hadm_id.duplicated().sum()
)

feature_duplicates = int(
    features.hadm_id.duplicated().sum()
)

record_check(
    "Unique cohort admissions",
    cohort_duplicates == 0,
    f"Duplicates: {cohort_duplicates}"
)

record_check(
    "Unique feature admissions",
    feature_duplicates == 0,
    f"Duplicates: {feature_duplicates}"
)

record_check(
    "Matching admission IDs",
    set(cohort.hadm_id) == set(features.hadm_id),
    "Cohort and feature table membership"
)

# ============================================================
# 4. REPRODUCE PATIENT-DISJOINT SPLIT
# ============================================================

print("\n3. PATIENT-LEVEL SPLIT CHECK")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

dev_idx, test_idx = next(
    splitter.split(
        cohort,
        cohort.icu_transfer,
        groups=cohort.subject_id
    )
)

dev = cohort.iloc[dev_idx]
test = cohort.iloc[test_idx]

overlap = (
    set(dev.subject_id)
    & set(test.subject_id)
)

record_check(
    "No patient overlap",
    len(overlap) == 0,
    f"Overlapping patients: {len(overlap)}"
)

print("Development admissions:", len(dev))
print("Test admissions:", len(test))

record_check(
    "Saved development size",
    len(dev) == metadata["development_admissions"],
    f"Observed: {len(dev)}"
)

record_check(
    "Saved test size",
    len(test) == metadata["reserved_test_admissions"],
    f"Observed: {len(test)}"
)

# ============================================================
# 5. TARGET CONSISTENCY
# ============================================================

print("\n4. TARGET CONSISTENCY")

if "icu_transfer" in features.columns:

    target_lookup = cohort.set_index(
        "hadm_id"
    )["icu_transfer"]

    expected = features.hadm_id.map(target_lookup)

    mismatch = int(
        (
            expected != features.icu_transfer
        ).sum()
    )

    record_check(
        "Outcome alignment",
        mismatch == 0,
        f"Mismatched labels: {mismatch}"
    )

# ============================================================
# 6. SUSPICIOUS FEATURE NAMES
# ============================================================

print("\n5. POTENTIAL LEAKAGE COLUMNS")

suspicious_terms = [
    "discharge",
    "death",
    "expire",
    "mortality",
    "hospital_los",
    "icu_los",
    "future",
    "outcome",
    "transfer_time",
    "icu_transfer",
    "readmission"
]

raw_columns = bundle["raw_columns"]

suspicious = [
    c for c in raw_columns
    if any(term in c.lower() for term in suspicious_terms)
]

record_check(
    "No obvious future-information predictors",
    len(suspicious) == 0,
    suspicious if suspicious else "None detected"
)

# ============================================================
# 7. SELECTED FEATURE CONSISTENCY
# ============================================================

print("\n6. SELECTED FEATURE CHECK")

record_check(
    "Exactly 200 selected features",
    len(selected_features) == 200,
    f"Found: {len(selected_features)}"
)

record_check(
    "No duplicate selected features",
    len(selected_features) == len(set(selected_features)),
    "Feature names must be unique"
)

pre = bundle["preprocessor"]

all_encoded = list(
    pre.get_feature_names_out()
)

selected_indices = np.asarray(
    bundle["selected_indices"],
    dtype=int
)

valid_indices = (
    len(selected_indices) == len(selected_features)
    and np.all(selected_indices >= 0)
    and np.all(selected_indices < len(all_encoded))
)

record_check(
    "Valid selected feature indices",
    valid_indices,
    f"Encoded features: {len(all_encoded)}"
)

if valid_indices:
    reconstructed = [
        all_encoded[i] for i in selected_indices
    ]

    record_check(
        "Feature names match saved indices",
        reconstructed == selected_features,
        "Exact model feature ordering"
    )

# ============================================================
# 8. MISSING VALUE AUDIT
# ============================================================

print("\n7. MISSING VALUE ANALYSIS")

numeric_columns = [
    c for c in raw_columns
    if c in features.columns
    and pd.api.types.is_numeric_dtype(features[c])
]

missing_summary = (
    features[numeric_columns]
    .isna()
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

missing_summary.columns = [
    "feature",
    "missing_fraction"
]

missing_summary.to_csv(
    OUT / "missing_value_summary.csv",
    index=False
)

print("\nTop 10 missing features:")
print(missing_summary.head(10).to_string(index=False))

# ============================================================
# 9. TIMESTAMP AVAILABILITY AUDIT
# ============================================================

print("\n8. TIMESTAMP AUDIT")

timestamp_terms = [
    "admittime",
    "dischtime",
    "charttime",
    "storetime",
    "intime",
    "outtime"
]

timestamp_columns = [
    c for c in cohort.columns
    if any(t in c.lower() for t in timestamp_terms)
]

print("Available timestamp columns:", timestamp_columns)

if not timestamp_columns:
    warnings.append(
        "No raw timestamps available in cohort audit. "
        "Cannot verify first-24-hour availability or "
        "24-72-hour outcome timing from these saved files."
    )

if "charttime" not in features.columns:
    warnings.append(
        "Raw laboratory charttime is not present. "
        "Feature extraction timing must be checked "
        "against the original extraction script."
    )

if "storetime" not in features.columns:
    warnings.append(
        "Laboratory result availability at prediction "
        "time cannot be verified from this feature table."
    )

# ============================================================
# 10. CLINICAL FEATURE AUDIT
# ============================================================

print("\n9. CLINICAL FEATURE SUMMARY")

lab_features = [
    c for c in selected_features
    if c.startswith("lab_")
]

other_features = [
    c for c in selected_features
    if not c.startswith("lab_")
]

lab_ids = set()

for name in lab_features:
    parts = name.split("_")
    if len(parts) >= 3 and parts[1].isdigit():
        lab_ids.add(parts[1])

print("Total selected predictors:", len(selected_features))
print("Laboratory predictors:", len(lab_features))
print("Unique laboratory item IDs:", len(lab_ids))
print("Other predictors:", len(other_features))

# ============================================================
# 11. SAVE AUDIT RESULTS
# ============================================================

pd.DataFrame(checks).to_csv(
    OUT / "audit_checks.csv",
    index=False
)

summary = {
    "total_admissions": len(cohort),
    "unique_patients": int(cohort.subject_id.nunique()),
    "positive_admissions": positive,
    "negative_admissions": negative,
    "development_admissions": len(dev),
    "test_admissions": len(test),
    "patient_overlap": len(overlap),
    "selected_features": len(selected_features),
    "unique_lab_ids": len(lab_ids),
    "all_checks_passed": all(
        x["passed"] for x in checks
    ),
    "warnings": warnings
}

with open(
    OUT / "audit_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 65)
print("FINAL AUDIT SUMMARY")
print("=" * 65)

print("Checks passed:",
      sum(x["passed"] for x in checks),
      "/",
      len(checks))

for warning in warnings:
    print("\nWARNING:", warning)

print("\nSTEP 49 COMPLETE")
print("Results saved to:", OUT)
print("No model retraining performed.")
