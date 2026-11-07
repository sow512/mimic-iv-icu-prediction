
# ============================================================
# STEP 56: OUTCOME AUDIT + FINAL EXCEL VERIFICATION
# Does NOT retrain models or overwrite previous files.
# ============================================================

from pathlib import Path
import json
import pandas as pd
import numpy as np

PROJECT = Path(
    r"C:\Users\sowmy\Downloads\MIMIC-IV\MIMIC_IV\MIMIC_IV"
)

SOURCE = PROJECT / "model training figures" / "expanded_lab_xgboost"

EXCEL = (
    SOURCE / "FINAL_75_FEATURE_DATASET"
    / "75_Feature_Dataset.xlsx"
)

COHORT_FILE = SOURCE / "cohort_audit.csv"
OUT = SOURCE / "step56_outcome_review"
OUT.mkdir(parents=True, exist_ok=True)

# Locate original MIMIC-IV files
def find_source(filename):
    if filename.startswith("admissions"):
        return PROJECT / "physionet.org/files/mimiciv/3.1/hosp/admissions.csv"

    elif filename.startswith("icustays"):
        return PROJECT / "physionet.org/files/mimiciv/3.1/icu/icustays.csv"

    raise FileNotFoundError(f"Unknown file: {filename}")

print("STEP 56: Loading original admission and ICU records...")

admissions = pd.read_csv(
    find_source("admissions.csv.gz"),
    usecols=["hadm_id", "admittime", "dischtime"],
    parse_dates=["admittime", "dischtime"]
)

icu = pd.read_csv(
    find_source("icustays.csv.gz"),
    usecols=["hadm_id", "intime"],
    parse_dates=["intime"]
)

cohort = pd.read_csv(COHORT_FILE)

required = {"hadm_id", "subject_id", "icu_transfer"}
if not required.issubset(cohort.columns):
    raise ValueError("Missing required cohort columns")

if cohort.hadm_id.duplicated().any():
    raise ValueError("Duplicate cohort admissions")

# ============================================================
# 1. RECONSTRUCT ICU TRANSFER TIMING
# ============================================================

icu = icu.dropna(subset=["hadm_id", "intime"])

first_icu = (
    icu.groupby("hadm_id")["intime"]
    .min()
    .rename("first_icu_time")
)

audit = cohort.merge(
    admissions,
    on="hadm_id",
    how="left",
    validate="one_to_one"
)

audit = audit.merge(
    first_icu,
    on="hadm_id",
    how="left",
    validate="one_to_one"
)

if audit.admittime.isna().any():
    raise ValueError("Missing admission timestamps")

audit["hour24"] = (
    audit["admittime"] + pd.Timedelta(hours=24)
)

audit["hour72"] = (
    audit["admittime"] + pd.Timedelta(hours=72)
)

audit["transfer_24_72"] = (
    audit.first_icu_time.gt(audit.hour24)
    & audit.first_icu_time.le(audit.hour72)
)

audit["positive_after_discharge"] = (
    audit.icu_transfer.eq(1)
    & audit.first_icu_time.notna()
    & audit.dischtime.notna()
    & audit.first_icu_time.gt(audit.dischtime)
)

audit["early_discharge_negative"] = (
    audit.icu_transfer.eq(0)
    & audit.dischtime.notna()
    & audit.dischtime.gt(audit.hour24)
    & audit.dischtime.lt(audit.hour72)
)

audit["label_mismatch"] = (
    audit.icu_transfer.astype(bool)
    != audit.transfer_24_72
)

# ============================================================
# 2. SAVE REVIEW REPORTS
# ============================================================

audit.to_csv(
    OUT / "complete_outcome_timing_audit.csv",
    index=False
)

inconsistent = audit[
    audit.positive_after_discharge
].copy()

inconsistent.to_csv(
    OUT / "positive_after_discharge_review.csv",
    index=False
)

early = audit[
    audit.early_discharge_negative
].copy()

early.to_csv(
    OUT / "early_discharge_negative_review.csv",
    index=False
)

print("\nOUTCOME AUDIT")
print("Total admissions:", len(audit))
print("Positive transfers after discharge:", len(inconsistent))
print("Early-discharge negatives:", len(early))
print("Label mismatches:", int(audit.label_mismatch.sum()))

# ============================================================
# 3. VERIFY EXISTING 75-FEATURE EXCEL WORKBOOK
# ============================================================

print("\nVERIFYING EXCEL WORKBOOK...")

if not EXCEL.exists():
    raise FileNotFoundError(
        f"Excel workbook not found: {EXCEL}"
    )

df = pd.read_excel(EXCEL, sheet_name=0)

metadata_columns = {
    "subject_id",
    "hadm_id",
    "dataset_split",
    "icu_transfer"
}

missing = metadata_columns - set(df.columns)
if missing:
    raise ValueError(f"Missing workbook columns: {missing}")

feature_columns = [
    c for c in df.columns
    if c not in metadata_columns
]

if len(feature_columns) != 75:
    raise ValueError(
        f"Expected 75 predictors, found {len(feature_columns)}"
    )

if df.hadm_id.duplicated().any():
    raise ValueError("Duplicate admissions in Excel")

if set(df.hadm_id) != set(cohort.hadm_id):
    raise ValueError("Excel admissions do not match cohort")

if not set(df.dataset_split.unique()).issubset(
    {"train", "test"}
):
    raise ValueError("Unexpected split labels")

train_patients = set(
    df.loc[df.dataset_split == "train", "subject_id"]
)

test_patients = set(
    df.loc[df.dataset_split == "test", "subject_id"]
)

if train_patients & test_patients:
    raise ValueError("Patient overlap between splits")

lookup = cohort.set_index("hadm_id")["icu_transfer"]

if not np.array_equal(
    df.icu_transfer.astype(int).to_numpy(),
    df.hadm_id.map(lookup).astype(int).to_numpy()
):
    raise ValueError("Excel outcome labels differ from cohort")

print("Excel verification PASSED")
print("Rows:", len(df))
print("Predictors:", len(feature_columns))
print("Training:", (df.dataset_split == "train").sum())
print("Testing:", (df.dataset_split == "test").sum())

# ============================================================
# 4. CREATE REVIEWED DATASET
# ============================================================

# Exclude inconsistent after-discharge positives.
# Do NOT silently convert their labels to 0.

exclude_ids = set(inconsistent.hadm_id)

reviewed = df[
    ~df.hadm_id.isin(exclude_ids)
].copy()

# Early discharge remains negative under the
# in-hospital transfer definition, but is flagged.
early_ids = set(early.hadm_id)

reviewed["early_discharge_before_72h"] = (
    reviewed.hadm_id.isin(early_ids).astype(int)
)

# Secondary sensitivity-analysis dataset:
# excludes early-discharge negatives too.
sensitivity = reviewed[
    reviewed.early_discharge_before_72h == 0
].copy()

# ============================================================
# 5. EXPORT TO ONE WORKBOOK
# ============================================================

output_excel = OUT / "Group28_75_Features_Outcome_Reviewed.xlsx"

with pd.ExcelWriter(
    output_excel,
    engine="openpyxl"
) as writer:

    reviewed.to_excel(
        writer,
        sheet_name="Reviewed_Dataset",
        index=False
    )

    sensitivity.to_excel(
        writer,
        sheet_name="Sensitivity_Analysis",
        index=False
    )

    notes = pd.DataFrame({
        "Item": [
            "Predictors",
            "Primary outcome",
            "After-discharge positives",
            "Early-discharge negatives",
            "Dataset status",
            "Training guidance"
        ],
        "Explanation": [
            "75 model predictors; exclude IDs, target, split and flags",
            "In-hospital ICU transfer between hours 24 and 72",
            "Excluded pending manual review",
            "Retained and flagged in primary dataset",
            "Outcome-reviewed exploratory dataset",
            "Retrain and validate models before reporting new performance"
        ]
    })

    notes.to_excel(
        writer,
        sheet_name="README",
        index=False
    )

summary = {
    "original_admissions": len(df),
    "inconsistent_positive_admissions": len(inconsistent),
    "early_discharge_negatives": len(early),
    "label_mismatches": int(audit.label_mismatch.sum()),
    "reviewed_admissions": len(reviewed),
    "sensitivity_admissions": len(sensitivity),
    "predictors": len(feature_columns),
    "excel_verified": True,
    "new_workbook": str(output_excel)
}

with open(
    OUT / "step56_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\nSTEP 56 COMPLETE")
print(json.dumps(summary, indent=2))
print("\nSaved:", output_excel)
print("Original model and workbook unchanged.")
