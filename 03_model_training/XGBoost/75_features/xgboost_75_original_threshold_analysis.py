
# ============================================================
# BN5212 FINAL PROJECT
# 75-FEATURE XGBOOST - ORIGINAL MODEL + THRESHOLD ANALYSIS
# MAC VERSION - WITHOUT HYPERPARAMETER TUNING
# ============================================================

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay
)

# ============================================================
# STEP 1: PATHS AND SETTINGS
# ============================================================

BASE = Path(__file__).resolve().parent

DATASET = BASE / "75_Feature_Dataset.xlsx"

OUT = BASE / "results_xgb_75_original_threshold"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
RECALL_TARGET = 0.70

print("=" * 55)
print("75-FEATURE ORIGINAL XGBOOST + THRESHOLD ANALYSIS")
print("=" * 55)

print("Dataset:", DATASET.name)
print("Output folder:", OUT)

# ============================================================
# STEP 2: ORIGINAL XGBOOST HYPERPARAMETERS
# ============================================================

# These are the ORIGINAL parameters from your friend's code.
# No tuned parameters are used.

ORIGINAL_PARAMS = dict(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.90,
    colsample_bytree=0.80,
    min_child_weight=5,
    reg_lambda=5,
    scale_pos_weight=1.0,
    tree_method="hist",
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=4
)

def make_model():
    return XGBClassifier(**ORIGINAL_PARAMS)

# ============================================================
# STEP 3: LOAD EXCEL DATASET
# ============================================================

print("\nLoading Excel dataset...", flush=True)

if not DATASET.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET}"
    )

df = pd.read_excel(DATASET, engine="openpyxl")

print("Dataset shape:", df.shape)

required = [
    "subject_id",
    "hadm_id",
    "dataset_split",
    "icu_transfer"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["dataset_split"] = (
    df["dataset_split"].astype(str).str.strip().str.lower()
)

print("\nDataset split distribution:")
print(df["dataset_split"].value_counts())

development = df[
    df["dataset_split"] == "train"
].copy()

test = df[
    df["dataset_split"] == "test"
].copy()

if development.empty or test.empty:
    raise ValueError("Train or test dataset is empty.")

# ============================================================
# STEP 4: CHECK PATIENT OVERLAP
# ============================================================

dev_ids = set(development["subject_id"].dropna())
test_ids = set(test["subject_id"].dropna())

if dev_ids.intersection(test_ids):
    raise ValueError(
        "Patient overlap detected between development and test."
    )

print("\nPatient-level split check passed!")

# ============================================================
# STEP 5: CREATE VALIDATION SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=2026
)

fit_idx, val_idx = next(
    splitter.split(
        development,
        development["icu_transfer"],
        groups=development["subject_id"]
    )
)

fit = development.iloc[fit_idx].copy()
val = development.iloc[val_idx].copy()

assert not (
    set(fit["subject_id"])
    & set(val["subject_id"])
)

print("\nTraining records:", len(fit))
print("Validation records:", len(val))
print("Reserved test records:", len(test))

# ============================================================
# STEP 6: SELECT 75 FEATURES
# ============================================================

exclude = [
    "subject_id",
    "hadm_id",
    "dataset_split",
    "icu_transfer"
]

features = [
    c for c in df.columns if c not in exclude
]

if len(features) != 75:
    raise ValueError(
        f"Expected 75 features, found {len(features)}"
    )

# The Excel file already contains encoded features.
# No additional one-hot encoding is needed.

X_fit = fit[features].apply(
    pd.to_numeric, errors="raise"
).astype(float)

X_val = val[features].apply(
    pd.to_numeric, errors="raise"
).astype(float)

X_dev = development[features].apply(
    pd.to_numeric, errors="raise"
).astype(float)

X_test = test[features].apply(
    pd.to_numeric, errors="raise"
).astype(float)

y_fit = fit["icu_transfer"].astype(int)
y_val = val["icu_transfer"].astype(int)
y_dev = development["icu_transfer"].astype(int)
y_test = test["icu_transfer"].astype(int)

for name, y in [
    ("training", y_fit),
    ("validation", y_val),
    ("test", y_test)
]:
    if y.nunique() != 2:
        raise ValueError(
            f"Both classes required in {name} data."
        )

print("\nNumber of features:", len(features))

print("\nTraining class distribution:")
print(y_fit.value_counts())

print("\nValidation class distribution:")
print(y_val.value_counts())

print("\nTest class distribution:")
print(y_test.value_counts())

# ============================================================
# STEP 7: TRAIN ORIGINAL XGBOOST
# ============================================================

print("\nTraining original XGBoost...", flush=True)

model = make_model()
model.fit(X_fit, y_fit)

print("Training completed!")

# ============================================================
# STEP 8: VALIDATION PROBABILITIES
# ============================================================

val_prob = model.predict_proba(X_val)[:, 1]

val_auroc = roc_auc_score(y_val, val_prob)
val_auprc = average_precision_score(y_val, val_prob)
val_brier = brier_score_loss(y_val, val_prob)

print("\nORIGINAL MODEL - VALIDATION RESULTS")
print("AUROC:", round(val_auroc, 5))
print("AUPRC:", round(val_auprc, 5))
print("Brier Score:", round(val_brier, 5))

# ============================================================
# STEP 9: THRESHOLD ANALYSIS
# ============================================================

print("\nPerforming threshold analysis...")

thresholds = np.unique(
    np.concatenate([
        np.linspace(0.001, 0.20, 400),
        [0.5]
    ])
)

def calculate_metrics(y_true, prob, threshold):

    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    return {
        "Threshold": float(threshold),
        "Precision": precision_score(
            y_true, pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, pred, zero_division=0
        ),
        "Specificity": (
            tn / (tn + fp) if tn + fp else 0
        ),
        "F1": f1_score(
            y_true, pred, zero_division=0
        ),
        "Balanced Accuracy": balanced_accuracy_score(
            y_true, pred
        ),
        "Accuracy": (tn + tp) / len(y_true),
        "Predicted Positives": int(pred.sum()),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

threshold_df = pd.DataFrame([
    calculate_metrics(y_val, val_prob, t)
    for t in thresholds
])

threshold_df.to_csv(
    OUT / "threshold_comparison.csv",
    index=False
)

# ============================================================
# STEP 10: SELECT THRESHOLD FOR 70% RECALL
# ============================================================

# Exact probability-based threshold selection.
# Among thresholds meeting recall >= 70%,
# choose the one with highest precision.

def select_threshold(y_true, prob, target=0.70):

    y_array = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)

    order = np.argsort(-prob, kind="stable")

    sorted_y = y_array[order]
    sorted_prob = prob[order]

    ends = np.r_[
        np.flatnonzero(np.diff(sorted_prob) != 0),
        len(sorted_prob) - 1
    ]

    tp = np.cumsum(sorted_y)[ends]
    fp = ends + 1 - tp

    recall = tp / max(y_array.sum(), 1)
    precision = tp / np.maximum(tp + fp, 1)

    eligible = np.flatnonzero(recall >= target)

    if len(eligible) == 0:
        raise ValueError("Recall target unavailable.")

    best_precision = precision[eligible].max()

    candidates = eligible[
        np.isclose(
            precision[eligible],
            best_precision,
            rtol=0,
            atol=1e-12
        )
    ]

    best = candidates[
        np.argmax(sorted_prob[ends[candidates]])
    ]

    return float(sorted_prob[ends[best]])

selected_threshold = select_threshold(
    y_val, val_prob, RECALL_TARGET
)

selected_metrics = calculate_metrics(
    y_val, val_prob, selected_threshold
)

print("\nSELECTED ORIGINAL MODEL THRESHOLD")
print("Threshold:", selected_threshold)

print("\nVALIDATION PERFORMANCE:")
print(pd.Series(selected_metrics).to_string())

pd.Series(selected_metrics).to_csv(
    OUT / "selected_threshold_metrics.csv"
)

# ============================================================
# STEP 11: RECALL TARGET COMPARISON
# ============================================================

target_rows = []

for target in [0.60, 0.70, 0.80, 0.90]:

    t = select_threshold(y_val, val_prob, target)

    row = calculate_metrics(y_val, val_prob, t)
    row["Recall Target"] = target

    target_rows.append(row)

target_df = pd.DataFrame(target_rows)

print("\nRECALL TARGET COMPARISON:")
print(target_df.round(4).to_string(index=False))

target_df.to_csv(
    OUT / "recall_target_comparison.csv",
    index=False
)

# ============================================================
# STEP 12: VALIDATION THRESHOLD GRAPH
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    label="F1-score"
)

plt.axvline(
    selected_threshold,
    color="black",
    linestyle="--",
    label="Original selected threshold"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")

plt.title(
    "75-Feature Original XGBoost - Validation Threshold Analysis"
)

plt.xlim(0, 0.20)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUT / "original_threshold_analysis.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 13: TRAIN FINAL ORIGINAL MODEL
# ============================================================

print("\nRetraining original model on full development data...")

final_model = make_model()
final_model.fit(X_dev, y_dev)

print("Final model training completed!")

# ============================================================
# STEP 14: RESERVED TEST EVALUATION
# ============================================================

print("\nEvaluating reserved test admissions...")

test_prob = final_model.predict_proba(X_test)[:, 1]

test_pred = (
    test_prob >= selected_threshold
).astype(int)

test_metrics = calculate_metrics(
    y_test, test_prob, selected_threshold
)

test_auroc = roc_auc_score(y_test, test_prob)
test_auprc = average_precision_score(y_test, test_prob)
test_brier = brier_score_loss(y_test, test_prob)

test_metrics["AUROC"] = test_auroc
test_metrics["AUPRC"] = test_auprc
test_metrics["Brier Score"] = test_brier

print("\n" + "=" * 55)
print("ORIGINAL XGBOOST - FINAL TEST RESULTS")
print("=" * 55)

print(pd.Series(test_metrics).to_string())

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_test, test_pred, zero_division=0
))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, test_pred))

pd.Series(test_metrics).to_csv(
    OUT / "original_test_metrics.csv"
)

pd.DataFrame({
    "subject_id": test["subject_id"].to_numpy(),
    "hadm_id": test["hadm_id"].to_numpy(),
    "actual": y_test.to_numpy(),
    "probability": test_prob,
    "predicted": test_pred
}).to_csv(
    OUT / "original_test_predictions.csv",
    index=False
)

# ============================================================
# STEP 15: CONFUSION MATRIX
# ============================================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_pred,
    labels=[0, 1],
    values_format="d",
    cmap="Blues"
)

plt.title(
    "75-Feature Original XGBoost - Test Confusion Matrix"
)
plt.tight_layout()

plt.savefig(
    OUT / "original_confusion_matrix.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 16: ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test, test_prob
)

plt.title("75-Feature Original XGBoost - Test ROC Curve")
plt.tight_layout()

plt.savefig(
    OUT / "original_roc_curve.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 17: PRECISION-RECALL CURVE
# ============================================================

PrecisionRecallDisplay.from_predictions(
    y_test, test_prob
)

plt.title("75-Feature Original XGBoost - Test PR Curve")
plt.tight_layout()

plt.savefig(
    OUT / "original_pr_curve.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 18: FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    final_model.feature_importances_,
    index=features
).sort_values(ascending=False)

importance.to_csv(
    OUT / "original_feature_importance.csv",
    header=["Importance"]
)

plt.figure(figsize=(11, 9))

importance.head(25).sort_values().plot(
    kind="barh"
)

plt.title("75-Feature Original XGBoost - Feature Importance")
plt.xlabel("Feature Importance")
plt.tight_layout()

plt.savefig(
    OUT / "original_feature_importance.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 19: COMPARISON WITH PREVIOUS TUNED RESULTS
# ============================================================

# These tuned results come from your earlier experiment.
# They are reference values, NOT recomputed by this script.

tuned_reference = {
    "AUROC": 0.764776,
    "AUPRC": 0.131873,
    "Recall": 0.730399,
    "Specificity": 0.63756,
    "Precision": 0.063807,
    "F1": 0.117361,
    "Accuracy": 0.640598,
    "Brier Score": 0.030179,
    "Threshold": 0.029568,
    "TN": 13705,
    "FP": 7791,
    "FN": 196,
    "TP": 531
}

original_reference = {
    "AUROC": test_auroc,
    "AUPRC": test_auprc,
    "Recall": test_metrics["Recall"],
    "Specificity": test_metrics["Specificity"],
    "Precision": test_metrics["Precision"],
    "F1": test_metrics["F1"],
    "Accuracy": test_metrics["Accuracy"],
    "Brier Score": test_brier,
    "Threshold": selected_threshold,
    "TN": test_metrics["TN"],
    "FP": test_metrics["FP"],
    "FN": test_metrics["FN"],
    "TP": test_metrics["TP"]
}

comparison = pd.DataFrame({
    "Original XGBoost - New Run": original_reference,
    "Tuned XGBoost - Previous Run": tuned_reference
})

print("\n" + "=" * 55)
print("ORIGINAL VS TUNED XGBOOST - TEST COMPARISON")
print("=" * 55)

print(comparison.round(5).to_string())

comparison.to_csv(
    OUT / "original_vs_tuned_comparison.csv"
)

# ============================================================
# STEP 20: SAVE MODEL AND METADATA
# ============================================================

joblib.dump(
    final_model,
    OUT / "original_xgboost_75.pkl"
)

metadata = {
    "model": "Original 75-Feature XGBoost",
    "parameters": ORIGINAL_PARAMS,
    "features": features,
    "selected_threshold": selected_threshold,
    "recall_target": RECALL_TARGET,
    "validation_auroc": val_auroc,
    "validation_auprc": val_auprc,
    "test_auroc": test_auroc,
    "test_auprc": test_auprc,
    "test_brier": test_brier
}

with open(
    OUT / "original_model_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metadata, f, indent=2)

with open(
    OUT / "original_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("75-FEATURE ORIGINAL XGBOOST\n\n")
    f.write("ORIGINAL PARAMETERS:\n")
    f.write(str(ORIGINAL_PARAMS))

    f.write("\n\nVALIDATION RESULTS:\n")
    f.write(f"AUROC: {val_auroc:.6f}\n")
    f.write(f"AUPRC: {val_auprc:.6f}\n")
    f.write(f"Brier: {val_brier:.6f}\n")

    f.write("\nFINAL TEST RESULTS:\n")
    f.write(pd.Series(test_metrics).to_string())

    f.write("\n\nCLASSIFICATION REPORT:\n")
    f.write(classification_report(
        y_test, test_pred, zero_division=0
    ))

    f.write("\n\nCOMPARISON WITH TUNED MODEL:\n")
    f.write(comparison.to_string())

print("\n" + "=" * 55)
print("ALL ORIGINAL MODEL RESULTS SAVED SUCCESSFULLY!")
print("=" * 55)
print("Results folder:", OUT)
