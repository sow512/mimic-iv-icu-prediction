
# ============================================================
# BN5212 FINAL PROJECT
# 75-FEATURE RANDOM FOREST
# ICU TRANSFER PREDICTION + 10-BED SIMULATION
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    brier_score_loss
)

# ============================================================
# STEP 1: SETTINGS
# ============================================================

FILE = Path("75_Feature_Dataset.xlsx")
OUT = Path("results_rf_75")

OUT.mkdir(exist_ok=True)

TARGET = "icu_transfer"

EXCLUDE = [
    "subject_id",
    "hadm_id",
    "dataset_split",
    TARGET
]

SEED = 42

N_CANDIDATES = 100
N_BEDS = 10

# ============================================================
# STEP 2: LOAD EXCEL DATASET
# ============================================================

print("\nLoading 75-feature dataset...", flush=True)

df = pd.read_excel(
    FILE,
    sheet_name="Final_75_Features",
    engine="openpyxl"
)

df.columns = df.columns.str.strip()

df["dataset_split"] = (
    df["dataset_split"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print("Dataset shape:", df.shape)

# ============================================================
# STEP 3: SEPARATE TRAINING AND TESTING DATA
# ============================================================

assert set(df["dataset_split"].unique()) == {
    "train", "test"
}, "Unexpected dataset_split values"

train = df[df["dataset_split"] == "train"].copy()
test = df[df["dataset_split"] == "test"].copy()

print("\nTraining records:", len(train))
print("Testing records:", len(test))

# ============================================================
# STEP 4: CHECK PATIENT OVERLAP
# ============================================================

if (
    train["subject_id"].isna().any()
    or test["subject_id"].isna().any()
):
    raise ValueError("Missing patient IDs detected.")

train_ids = set(train["subject_id"])
test_ids = set(test["subject_id"])

overlap = train_ids.intersection(test_ids)

print("\nPatients appearing in both sets:", len(overlap))

if overlap:
    raise ValueError(
        "Patient overlap detected! "
        "Training and testing sets must be separated."
    )

# ============================================================
# STEP 5: SELECT 75 FEATURES
# ============================================================

features = [
    col for col in df.columns
    if col not in EXCLUDE
]

print("\nNumber of input features:", len(features))

assert len(features) == 75, (
    f"Expected 75 features, found {len(features)}"
)

X = train[features].apply(
    pd.to_numeric,
    errors="raise"
)

y = train[TARGET].astype(int)

X_test = test[features].apply(
    pd.to_numeric,
    errors="raise"
)

y_test = test[TARGET].astype(int)

print("\nTraining class distribution:")
print(y.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

if set(y.unique()) != {0, 1}:
    raise ValueError("Training target must contain 0 and 1.")

if set(y_test.unique()) != {0, 1}:
    raise ValueError("Testing target must contain 0 and 1.")

# ============================================================
# STEP 6: PATIENT-LEVEL VALIDATION SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

train_idx, val_idx = next(
    splitter.split(
        X,
        y,
        groups=train["subject_id"]
    )
)

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

print("\nModel training records:", len(X_train))
print("Validation records:", len(X_val))

if y_train.nunique() != 2 or y_val.nunique() != 2:
    raise ValueError(
        "Both classes must be present "
        "in training and validation."
    )

# ============================================================
# STEP 7: CREATE RANDOM FOREST
# ============================================================

def make_model():

    return Pipeline([

        ("imputer", SimpleImputer(
            strategy="median",
            keep_empty_features=True
        )),

        ("forest", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        ))

    ])

# ============================================================
# STEP 8: TRAIN VALIDATION MODEL
# ============================================================

print("\nTraining Random Forest...", flush=True)

model = make_model()

model.fit(X_train, y_train)

print("Training completed!", flush=True)

# ============================================================
# STEP 9: VALIDATION PREDICTIONS
# ============================================================

val_prob = model.predict_proba(X_val)[:, 1]

val_auroc = roc_auc_score(y_val, val_prob)
val_auprc = average_precision_score(y_val, val_prob)

print("\nVALIDATION RESULTS")

print("AUROC:", round(val_auroc, 4))
print("AUPRC:", round(val_auprc, 4))

# ============================================================
# STEP 10: THRESHOLD TUNING
# ============================================================

thresholds = np.round(
    np.arange(0.05, 0.951, 0.05),
    2
)

results = []

for threshold in thresholds:

    pred = (val_prob >= threshold).astype(int)

    results.append({

        "Threshold": threshold,

        "Precision": precision_score(
            y_val, pred, zero_division=0
        ),

        "Recall": recall_score(
            y_val, pred, zero_division=0
        ),

        "F1": f1_score(
            y_val, pred, zero_division=0
        ),

        "Balanced Accuracy": balanced_accuracy_score(
            y_val, pred
        )

    })

threshold_table = pd.DataFrame(results)

print("\nTHRESHOLD COMPARISON:")
print(threshold_table.round(4).to_string(index=False))

threshold_table.to_csv(
    OUT / "validation_thresholds.csv",
    index=False
)

best_index = threshold_table["F1"].idxmax()

best_threshold = float(
    threshold_table.loc[best_index, "Threshold"]
)

print("\nSelected threshold:", best_threshold)

# ============================================================
# STEP 11: TRAIN FINAL MODEL
# ============================================================

print("\nTraining final Random Forest...", flush=True)

final_model = make_model()

final_model.fit(X, y)

print("Final training completed!", flush=True)

# ============================================================
# STEP 12: TEST PREDICTIONS
# ============================================================

test_prob = final_model.predict_proba(
    X_test
)[:, 1]

test_pred = (
    test_prob >= best_threshold
).astype(int)

# ============================================================
# STEP 13: EVALUATE MODEL
# ============================================================

print("\n================================")
print("FINAL 75-FEATURE RANDOM FOREST")
print("================================")

report = classification_report(
    y_test,
    test_pred,
    labels=[0, 1],
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=[0, 1]
)

auroc = roc_auc_score(y_test, test_prob)

auprc = average_precision_score(y_test, test_prob)
brier = brier_score_loss(y_test, test_prob)

print("AUPRC:", round(auprc, 4))
print("Brier Score:", round(brier, 4))

print("\nCLASSIFICATION REPORT:")
print(report)

print("\nCONFUSION MATRIX:")
print(cm)

print("\nAUROC:", round(auroc, 4))

# ============================================================
# STEP 14: SAVE MODEL RESULTS
# ============================================================

with open(OUT / "test_results.txt", "w") as f:

    f.write("75-FEATURE RANDOM FOREST RESULTS\n\n")

    f.write(f"Threshold: {best_threshold}\n")
    f.write(f"AUROC: {auroc:.5f}\n")
    f.write(f"AUPRC: {auprc:.5f}\n\n")

    f.write("Classification Report:\n")
    f.write(report)

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

# ============================================================
# STEP 15: CONFUSION MATRIX GRAPH
# ============================================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_pred,
    values_format="d",
    cmap="Blues"
)

plt.title("75-Feature RF - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUT / "confusion_matrix.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 16: ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    test_prob
)

plt.title("75-Feature RF - ROC Curve")
plt.tight_layout()

plt.savefig(
    OUT / "roc_curve.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 17: PRECISION-RECALL CURVE
# ============================================================

PrecisionRecallDisplay.from_predictions(
    y_test,
    test_prob
)

plt.title("75-Feature RF - PR Curve")
plt.tight_layout()

plt.savefig(
    OUT / "precision_recall_curve.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 18: FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    final_model.named_steps["forest"].feature_importances_,
    index=features
).sort_values(ascending=False)

importance.to_csv(
    OUT / "feature_importance.csv",
    header=["Importance"]
)

plt.figure(figsize=(11, 9))

importance.head(25).sort_values().plot(
    kind="barh"
)

plt.title("75-Feature RF - Top 25 Features")
plt.xlabel("Feature Importance")
plt.tight_layout()

plt.savefig(
    OUT / "feature_importance.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 19: SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    final_model,
    OUT / "random_forest_75.pkl"
)

print("\nModel and graphs saved!")

# ============================================================
# STEP 20: ICU BED ALLOCATION SIMULATION
# ============================================================

print("\n================================")
print("10-BED ICU RANKING SIMULATION")
print("================================")

# One admission per patient
pool = test.drop_duplicates(
    subset=["subject_id"]
)

if len(pool) < N_CANDIDATES:
    raise ValueError("Not enough unique patients.")

# Select 100 random patient admissions
candidates = pool.sample(
    n=N_CANDIDATES,
    random_state=SEED
).copy()

# Prepare model input
X_candidates = candidates[features].apply(
    pd.to_numeric,
    errors="raise"
)

# Calculate predicted ICU risk scores
candidates["risk_score"] = (
    final_model.predict_proba(X_candidates)[:, 1]
)

# Rank highest risk first
candidates = candidates.sort_values(
    by="risk_score",
    ascending=False
).reset_index(drop=True)

candidates["priority_rank"] = (
    np.arange(len(candidates)) + 1
)

# Recommend top 10
candidates["top_10_recommended"] = (
    candidates["priority_rank"] <= N_BEDS
)

# ============================================================
# STEP 21: EVALUATE SIMULATION
# ============================================================

actual_icu = int(
    candidates[TARGET].sum()
)

captured = int(
    candidates.head(N_BEDS)[TARGET].sum()
)

missed = actual_icu - captured

precision_at_10 = captured / N_BEDS

recall_at_10 = (
    captured / actual_icu
    if actual_icu > 0
    else float("nan")
)

random_expected = (
    N_BEDS * actual_icu / N_CANDIDATES
)

print("\nCandidate admissions:", N_CANDIDATES)
print("Available beds:", N_BEDS)

print("Actual ICU transfers:", actual_icu)

print("Actual transfers in top 10:", captured)

print("Actual transfers outside top 10:", missed)

print("Precision@10:", round(precision_at_10, 4))

print("Recall@10:", round(recall_at_10, 4))

print(
    "Random-selection expected captures:",
    round(random_expected, 2)
)

# ============================================================
# STEP 22: DISPLAY TOP 10
# ============================================================

columns = [
    "priority_rank",
    "subject_id",
    "hadm_id",
    "risk_score",
    TARGET,
    "top_10_recommended"
]

print("\nTOP 10 RECOMMENDED ADMISSIONS:")

print(
    candidates[columns]
    .head(N_BEDS)
    .to_string(index=False)
)

# ============================================================
# STEP 23: SAVE SIMULATION TO EXCEL
# ============================================================

with pd.ExcelWriter(
    OUT / "simulation_top10.xlsx",
    engine="openpyxl"
) as writer:

    candidates[columns].to_excel(
        writer,
        sheet_name="All 100",
        index=False
    )

    candidates.head(N_BEDS)[columns].to_excel(
        writer,
        sheet_name="Top 10",
        index=False
    )

    summary = pd.DataFrame([{

        "Candidates": N_CANDIDATES,
        "Available Beds": N_BEDS,
        "Actual ICU Transfers": actual_icu,
        "Captured in Top 10": captured,
        "Missed Outside Top 10": missed,
        "Precision@10": precision_at_10,
        "Recall@10": recall_at_10,
        "Random Expected Captures": random_expected

    }])

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

# ============================================================
# STEP 24: SIMULATION GRAPH
# ============================================================

top10 = candidates.head(N_BEDS)

plt.figure(figsize=(10, 6))

plt.barh(
    top10["priority_rank"].astype(str),
    top10["risk_score"]
)

plt.gca().invert_yaxis()

plt.xlabel("Predicted ICU Transfer Score")
plt.ylabel("Priority Rank")

plt.title("Top 10 ICU Risk Ranking")

plt.tight_layout()

plt.savefig(
    OUT / "simulation_top10_scores.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 25: FINISH
# ============================================================

print("\n================================")
print("ALL RESULTS SAVED SUCCESSFULLY!")
print("================================")

print("\nResults folder:", OUT.resolve())
