
# ============================================================
# BN5212 FINAL PROJECT
# SIMULATED ALLOCATION OF 10 ICU BEDS
# RANDOM FOREST - 200 FEATURES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

# ============================================================
# STEP 1: SIMULATION SETTINGS
# ============================================================

NUMBER_OF_PATIENTS = 100
AVAILABLE_BEDS = 10
RANDOM_SEED = 42

results_folder = Path("results_icu_simulation")
results_folder.mkdir(exist_ok=True)

print("\n======================================")
print("ICU BED ALLOCATION SIMULATION")
print("======================================")

print("Candidate admissions:", NUMBER_OF_PATIENTS)
print("Available ICU beds:", AVAILABLE_BEDS)

# ============================================================
# STEP 2: LOAD TESTING DATA
# ============================================================

print("\nLoading testing dataset...")

data = pd.read_excel(
    "cleaned dataset (200 features).xlsx",
    sheet_name="Testing Data",
    engine="openpyxl"
)

print("Total test records:", len(data))

# ============================================================
# STEP 3: LOAD TRAINED RANDOM FOREST
# ============================================================

model = joblib.load(
    "results_rf_200/random_forest_200.pkl"
)

print("Trained Random Forest loaded!")

# ============================================================
# STEP 4: SELECT 100 ADMISSIONS
# ============================================================

if NUMBER_OF_PATIENTS > len(data):
    raise ValueError("Not enough testing records.")

# Select at most one admission per patient
candidate_pool = data.drop_duplicates(
    subset=["subject_id"]
)

if NUMBER_OF_PATIENTS > len(candidate_pool):
    raise ValueError("Not enough unique patients.")

patients = candidate_pool.sample(
    n=NUMBER_OF_PATIENTS,
    random_state=RANDOM_SEED
).copy()

print("\nSelected", len(patients), "patient admissions.")

# ============================================================
# STEP 5: PREPARE MODEL INPUTS
# ============================================================

exclude = ["subject_id", "hadm_id", "icu_transfer"]

# Use the exact feature order from training
features = list(model.feature_names_in_)

X = patients[features].copy()

for col in features:
    X[col] = pd.to_numeric(
        X[col], errors="raise"
    )

# ============================================================
# STEP 6: PREDICT ICU TRANSFER SCORES
# ============================================================

print("\nCalculating ICU transfer scores...")

patients["predicted_risk"] = (
    model.predict_proba(X)[:, 1]
)

# ============================================================
# STEP 7: RANK PATIENTS
# ============================================================

patients = patients.sort_values(
    by="predicted_risk",
    ascending=False
).reset_index(drop=True)

patients["priority_rank"] = (
    np.arange(len(patients)) + 1
)

# ============================================================
# STEP 8: RECOMMEND 10 ICU BEDS
# ============================================================

patients["recommended_bed"] = 0

patients.loc[
    patients.index[:AVAILABLE_BEDS],
    "recommended_bed"
] = 1

patients["allocation_status"] = np.where(
    patients["recommended_bed"] == 1,
    "Recommended for ICU bed",
    "Not selected in top 10"
)

# ============================================================
# STEP 9: DISPLAY TOP 10 PATIENTS
# ============================================================

display_columns = [
    "priority_rank",
    "subject_id",
    "hadm_id",
    "predicted_risk",
    "icu_transfer",
    "allocation_status"
]

print("\n======================================")
print("TOP 10 ICU BED RECOMMENDATIONS")
print("======================================")

print(
    patients[display_columns]
    .head(AVAILABLE_BEDS)
    .to_string(index=False)
)

# ============================================================
# STEP 10: EVALUATE RETROSPECTIVE RANKING
# ============================================================

selected = patients.head(AVAILABLE_BEDS)

actual_icu = int(patients["icu_transfer"].sum())

correctly_selected = int(
    selected["icu_transfer"].sum()
)

missed = int(
    patients.iloc[AVAILABLE_BEDS:]["icu_transfer"].sum()
)

precision_at_10 = (
    correctly_selected / AVAILABLE_BEDS
)

recall_at_10 = (
    correctly_selected / actual_icu
    if actual_icu > 0 else np.nan
)

print("\n======================================")
print("SIMULATION RESULTS")
print("======================================")

print("Available beds:", AVAILABLE_BEDS)

print("Actual ICU transfers in candidate group:",
      actual_icu)

print("Actual ICU transfers among top 10:",
      correctly_selected)

print("Actual ICU transfers outside top 10:",
      missed)

print("Precision at 10:",
      round(precision_at_10, 4))

print("Recall at 10:",
      round(recall_at_10, 4)
      if not np.isnan(recall_at_10)
      else "Undefined (no actual ICU transfers)")

# ============================================================
# STEP 11: SAVE EXCEL RESULTS
# ============================================================

output_file = results_folder / "icu_allocation_results.xlsx"

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    patients[display_columns].to_excel(
        writer,
        sheet_name="All Candidates",
        index=False
    )

    selected[display_columns].to_excel(
        writer,
        sheet_name="Top 10",
        index=False
    )

    summary = pd.DataFrame({
        "Metric": [
            "Candidate admissions",
            "Available beds",
            "Actual ICU transfers",
            "Actual transfers in top 10",
            "Actual transfers outside top 10",
            "Precision at 10",
            "Recall at 10"
        ],
        "Value": [
            NUMBER_OF_PATIENTS,
            AVAILABLE_BEDS,
            actual_icu,
            correctly_selected,
            missed,
            precision_at_10,
            recall_at_10
        ]
    })

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

print("\nExcel results saved!")

# ============================================================
# STEP 12: PLOT TOP 10 RISK SCORES
# ============================================================

top10 = patients.head(AVAILABLE_BEDS).copy()

plt.figure(figsize=(10, 6))

plt.barh(
    top10["priority_rank"].astype(str),
    top10["predicted_risk"]
)

plt.gca().invert_yaxis()

plt.xlabel("Predicted ICU transfer score")
plt.ylabel("Priority rank")
plt.title("Top 10 Ranked Admissions")

plt.tight_layout()

plt.savefig(
    results_folder / "top10_risk_scores.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 13: PLOT RANKING OUTCOMES
# ============================================================

plt.figure(figsize=(9, 5))

labels = [
    "Actual transfers in top 10",
    "Actual transfers outside top 10"
]

values = [
    correctly_selected,
    missed
]

plt.bar(labels, values)

plt.ylabel("Number of ICU transfer records")
plt.title("Retrospective ICU Transfer Capture")

plt.tight_layout()

plt.savefig(
    results_folder / "allocation_outcomes.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 14: SAVE SUMMARY TEXT
# ============================================================

with open(
    results_folder / "simulation_summary.txt",
    "w"
) as f:

    f.write("ICU BED ALLOCATION SIMULATION\n\n")
    f.write(f"Candidates: {NUMBER_OF_PATIENTS}\n")
    f.write(f"Available beds: {AVAILABLE_BEDS}\n")
    f.write(f"Actual ICU transfers: {actual_icu}\n")
    f.write(f"Transfers in top 10: {correctly_selected}\n")
    f.write(f"Transfers outside top 10: {missed}\n")
    f.write(f"Precision at 10: {precision_at_10:.4f}\n")
    f.write(f"Recall at 10: {recall_at_10:.4f}\n")

print("\n======================================")
print("SIMULATION COMPLETED SUCCESSFULLY!")
print("======================================")
