
# ==========================================================
# BN5212 FINAL PROJECT
# RANDOM FOREST - CHOSEN FEATURE DATASET
# TARGET: ICU TRANSFER WITHIN 24 HOURS
# ==========================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)

# ==========================================================
# STEP 1: SETTINGS
# ==========================================================

FILE = "Chosen Feature Dataset.csv.gz"
OUT = Path("results_chosen_rf")
OUT.mkdir(exist_ok=True)

TARGET = "icu_transfer_within_24h"
SEED = 42

N_CANDIDATES = 100
N_BEDS = 10

# ==========================================================
# STEP 2: LOAD DATA
# ==========================================================

print("\nLoading chosen-feature dataset...", flush=True)

df = pd.read_csv(FILE, compression="gzip")

print("Dataset shape:", df.shape)

print("\nDataset splits:")
print(df["dataset_split"].value_counts())

# ==========================================================
# STEP 3: SEPARATE DATASETS
# ==========================================================

train = df[df["dataset_split"] == "train"].copy()
val = df[df["dataset_split"] == "validation"].copy()
test = df[df["dataset_split"] == "test"].copy()

print("\nTraining records:", len(train))
print("Validation records:", len(val))
print("Testing records:", len(test))

# ==========================================================
# STEP 4: CHECK PATIENT OVERLAP
# ==========================================================

train_ids = set(train["subject_id"])
val_ids = set(val["subject_id"])
test_ids = set(test["subject_id"])

if (
    train_ids & val_ids
    or train_ids & test_ids
    or val_ids & test_ids
):
    raise ValueError(
        "Patient overlap detected between data splits."
    )

print("\nPatient-level split check passed!")

# ==========================================================
# STEP 5: SELECT FEATURES
# ==========================================================

exclude = [
    "dataset_split",
    "row_id",
    "subject_id",
    "hadm_id",
    "index_time",
    TARGET
]

features = [
    c for c in df.columns if c not in exclude
]

print("\nNumber of features:", len(features))

assert len(features) == 44, (
    f"Expected 44 features, found {len(features)}"
)

X_train = train[features].astype("float32")
X_val = val[features].astype("float32")
X_test = test[features].astype("float32")

y_train = train[TARGET].astype(int)
y_val = val[TARGET].astype(int)
y_test = test[TARGET].astype(int)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nValidation class distribution:")
print(y_val.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# ==========================================================
# STEP 6: CREATE RANDOM FOREST
# ==========================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)

# ==========================================================
# STEP 7: TRAIN MODEL
# ==========================================================

print("\nTraining Random Forest...", flush=True)

model.fit(X_train, y_train)

print("Training completed!", flush=True)

# ==========================================================
# STEP 8: VALIDATION RESULTS
# ==========================================================

val_prob = model.predict_proba(X_val)[:, 1]

print("\nVALIDATION RESULTS")

print("AUROC:",
      roc_auc_score(y_val, val_prob))

print("AUPRC:",
      average_precision_score(y_val, val_prob))

# ==========================================================
# STEP 9: THRESHOLD TUNING
# ==========================================================

thresholds = np.round(
    np.arange(0.05, 0.951, 0.05), 2
)

rows = []

for t in thresholds:

    pred = (val_prob >= t).astype(int)

    rows.append({
        "Threshold": t,
        "Precision": precision_score(
            y_val, pred, zero_division=0
        ),
        "Recall": recall_score(
            y_val, pred, zero_division=0
        ),
        "F1": f1_score(
            y_val, pred, zero_division=0
        )
    })

threshold_df = pd.DataFrame(rows)

print("\nTHRESHOLD COMPARISON:")
print(threshold_df.round(4).to_string(index=False))

threshold_df.to_csv(
    OUT / "threshold_comparison.csv",
    index=False
)

best_threshold = float(
    threshold_df.loc[
        threshold_df["F1"].idxmax(),
        "Threshold"
    ]
)

print("\nSelected threshold:", best_threshold)

# ==========================================================
# STEP 10: FINAL TEST EVALUATION
# ==========================================================

test_prob = model.predict_proba(X_test)[:, 1]

test_pred = (
    test_prob >= best_threshold
).astype(int)

report = classification_report(
    y_test, test_pred, zero_division=0
)

cm = confusion_matrix(
    y_test, test_pred, labels=[0, 1]
)

auroc = roc_auc_score(y_test, test_prob)
auprc = average_precision_score(y_test, test_prob)
brier = brier_score_loss(y_test, test_prob)

print("AUPRC:", round(auprc, 4))
print("Brier Score:", round(brier, 4))
print("\n================================")
print("FINAL RANDOM FOREST RESULTS")
print("================================")

print("\nSelected threshold:", best_threshold)

print("\nCLASSIFICATION REPORT:")
print(report)

print("\nCONFUSION MATRIX:")
print(cm)

print("\nAUROC:", round(auroc, 4))

# ==========================================================
# STEP 11: SAVE GRAPHS
# ==========================================================

ConfusionMatrixDisplay.from_predictions(
    y_test, test_pred,
    values_format="d", cmap="Blues"
)
plt.title("Chosen Features - Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=300)
plt.close()

RocCurveDisplay.from_predictions(
    y_test, test_prob
)
plt.title("Chosen Features - ROC Curve")
plt.tight_layout()
plt.savefig(OUT / "roc_curve.png", dpi=300)
plt.close()

PrecisionRecallDisplay.from_predictions(
    y_test, test_prob
)
plt.title("Chosen Features - PR Curve")
plt.tight_layout()
plt.savefig(
    OUT / "precision_recall_curve.png", dpi=300
)
plt.close()

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

importance.to_csv(
    OUT / "feature_importance.csv"
)

plt.figure(figsize=(10, 9))
importance.head(25).sort_values().plot(kind="barh")
plt.title("Top 25 Feature Importances")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=300)
plt.close()

# ==========================================================
# STEP 12: SAVE MODEL
# ==========================================================

joblib.dump(
    model,
    OUT / "random_forest_chosen.pkl"
)

with open(OUT / "test_results.txt", "w") as f:
    f.write(f"Threshold: {best_threshold}\n")
    f.write(f"AUROC: {auroc:.5f}\n")
    f.write(f"AUPRC: {auprc:.5f}\n\n")
    f.write(report)
    f.write("\nConfusion matrix:\n")
    f.write(str(cm))

print("\nModel and graphs saved!")

# ==========================================================
# STEP 13: 10-BED SIMULATION
# ==========================================================

print("\n================================")
print("10-BED ICU RANKING SIMULATION")
print("================================")

# Use the last observation per admission,
# so each admission is represented only once.
pool = (
    test.sort_values("index_time")
    .drop_duplicates("hadm_id", keep="last")
)

# Keep only one admission per patient.
pool = pool.drop_duplicates(
    "subject_id", keep="last"
)

candidates = pool.sample(
    n=N_CANDIDATES,
    random_state=SEED
).copy()

X_candidates = (
    candidates[features].astype("float32")
)

candidates["risk_score"] = (
    model.predict_proba(X_candidates)[:, 1]
)

candidates = candidates.sort_values(
    "risk_score",
    ascending=False
).reset_index(drop=True)

candidates["priority_rank"] = (
    np.arange(len(candidates)) + 1
)

candidates["top10_recommended"] = (
    candidates["priority_rank"] <= N_BEDS
)

# ==========================================================
# STEP 14: SIMULATION RESULTS
# ==========================================================

actual = int(candidates[TARGET].sum())

captured = int(
    candidates.head(N_BEDS)[TARGET].sum()
)

missed = actual - captured

precision_at_10 = captured / N_BEDS

recall_at_10 = (
    captured / actual
    if actual > 0 else float("nan")
)

print("\nCandidate admissions:", N_CANDIDATES)
print("Available beds:", N_BEDS)
print("Actual ICU transfers within 24h:", actual)
print("Captured in top 10:", captured)
print("Missed outside top 10:", missed)
print("Precision@10:", round(precision_at_10, 4))
print("Recall@10:", round(recall_at_10, 4))

print("\nRandom expected captures:",
      round(N_BEDS * actual / N_CANDIDATES, 2))

columns = [
    "priority_rank",
    "subject_id",
    "hadm_id",
    "index_time",
    "risk_score",
    TARGET,
    "top10_recommended"
]

print("\nTOP 10 RANKED ADMISSIONS:")

print(
    candidates[columns]
    .head(N_BEDS)
    .to_string(index=False)
)

# ==========================================================
# STEP 15: SAVE SIMULATION RESULTS
# ==========================================================

with pd.ExcelWriter(
    OUT / "simulation_results.xlsx",
    engine="openpyxl"
) as writer:

    candidates[columns].to_excel(
        writer,
        sheet_name="All Candidates",
        index=False
    )

    candidates.head(N_BEDS)[columns].to_excel(
        writer,
        sheet_name="Top 10",
        index=False
    )

plt.figure(figsize=(10, 6))

top10 = candidates.head(N_BEDS)

plt.barh(
    top10["priority_rank"].astype(str),
    top10["risk_score"]
)

plt.gca().invert_yaxis()
plt.xlabel("Predicted ICU Transfer Score")
plt.ylabel("Priority Rank")
plt.title("Top 10 ICU Transfer Risk Ranking")
plt.tight_layout()

plt.savefig(
    OUT / "simulation_top10.png",
    dpi=300
)

plt.close()

print("\n================================")
print("ALL RESULTS SAVED SUCCESSFULLY!")
print("================================")
