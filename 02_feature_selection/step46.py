# ============================================================
# STEP 46: ANALYZE FINAL 200 SHAP-SELECTED FEATURES
# ============================================================

from pathlib import Path
from collections import defaultdict
import pandas as pd

PROJECT_DIR = Path(
    r"C:\Users\sowmy\Downloads\MIMIC-IV\MIMIC_IV\MIMIC_IV"
)

MODEL_DIR = (
    PROJECT_DIR
    / "model training figures"
    / "expanded_lab_xgboost"
    / "step44_final_200_model"
)

OUTPUT_DIR = MODEL_DIR / "feature_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 1: LOAD FINAL SELECTED FEATURES
# ============================================================

feature_file = MODEL_DIR / "final_200_selected_features.csv"

df = pd.read_csv(feature_file)

print("\n" + "=" * 60)
print("FINAL 200-FEATURE ANALYSIS")
print("=" * 60)

print("Total selected features:", len(df))

# ============================================================
# STEP 2: IDENTIFY LABORATORY TESTS
# ============================================================

# The expanded pipeline uses lab_<itemid>_<statistic>
# e.g. lab_50912_mean_value

lab_groups = defaultdict(list)
non_lab_features = []

for feature in df["feature"]:

    if feature.startswith("lab_"):

        parts = feature.split("_")

        if len(parts) >= 3 and parts[1].isdigit():
            lab_id = parts[1]
            lab_groups[lab_id].append(feature)
        else:
            non_lab_features.append(feature)

    else:
        non_lab_features.append(feature)

# ============================================================
# STEP 3: DISPLAY SUMMARY
# ============================================================

print("\nUNIQUE LABORATORY TESTS:", len(lab_groups))

print("LAB-DERIVED FEATURES:",
      sum(len(v) for v in lab_groups.values()))

print("NON-LAB FEATURES:", len(non_lab_features))

print("\n" + "=" * 60)
print("LABORATORY TEST BREAKDOWN")
print("=" * 60)

lab_summary = []

for lab_id, features in lab_groups.items():

    lab_summary.append({
        "Lab_Item_ID": lab_id,
        "Number_of_Features": len(features),
        "Selected_Features": ", ".join(features)
    })

lab_summary_df = pd.DataFrame(lab_summary)

if not lab_summary_df.empty:

    lab_summary_df = lab_summary_df.sort_values(
        "Number_of_Features",
        ascending=False
    )

    print(
        lab_summary_df[
            ["Lab_Item_ID", "Number_of_Features"]
        ].to_string(index=False)
    )

# ============================================================
# STEP 4: DISPLAY NON-LAB FEATURES
# ============================================================

print("\nNON-LAB FEATURES:")

for feature in non_lab_features:
    print("-", feature)

# ============================================================
# STEP 5: SAVE RESULTS
# ============================================================

lab_summary_df.to_csv(
    OUTPUT_DIR / "selected_lab_test_summary.csv",
    index=False
)

pd.DataFrame({
    "Non_Lab_Feature": non_lab_features
}).to_csv(
    OUTPUT_DIR / "non_lab_features.csv",
    index=False
)

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("Total predictors:", len(df))
print("Unique laboratory test IDs:", len(lab_groups))
print("Laboratory-derived predictors:",
      sum(len(v) for v in lab_groups.values()))
print("Other predictors:", len(non_lab_features))

print("\nSTEP 46 COMPLETE")
print("Results saved to:", OUTPUT_DIR)