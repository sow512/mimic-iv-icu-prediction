
# ============================================================
# BN5212 - RF 44 FEATURES + SMOTE
# TARGET: ICU TRANSFER WITHIN 24 HOURS
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import gc

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay,
    PrecisionRecallDisplay, roc_auc_score,
    average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score,
    accuracy_score, balanced_accuracy_score
)
from imblearn.over_sampling import SMOTE, SMOTENC

SEED = 42
OUT = Path("results_rf_chosen_smote")
OUT.mkdir(exist_ok=True)

TARGET = "icu_transfer_within_24h"

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading chosen-feature dataset...", flush=True)

df = pd.read_csv(
    "Chosen Feature Dataset.csv.gz",
    compression="gzip"
)

print("Dataset shape:", df.shape)

df["dataset_split"] = (
    df["dataset_split"].astype(str).str.strip().str.lower()
)

train = df[df["dataset_split"] == "train"].copy()
val = df[df["dataset_split"] == "validation"].copy()
test = df[df["dataset_split"] == "test"].copy()

print("Training:", len(train))
print("Validation:", len(val))
print("Testing:", len(test))

# ------------------------------------------------------------
# PATIENT SPLIT CHECK
# ------------------------------------------------------------

for part in [train, val, test]:
    if part["subject_id"].isna().any():
        raise ValueError("Missing patient IDs")

train_ids = set(train["subject_id"])
val_ids = set(val["subject_id"])
test_ids = set(test["subject_id"])

if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
    raise ValueError("Patient overlap detected!")

print("Patient-level split check passed!")

# ------------------------------------------------------------
# SELECT FEATURES
# ------------------------------------------------------------

exclude = [
    "dataset_split",
    "row_id",
    "subject_id",
    "hadm_id",
    "index_time",
    TARGET
]

features = [c for c in df.columns if c not in exclude]

assert len(features) == 44, (
    f"Expected 44 features, found {len(features)}"
)

X_train = train[features].astype("float32")
X_val = val[features].astype("float32")
X_test = test[features].astype("float32")

y_train = train[TARGET].astype(int)
y_val = val[TARGET].astype(int)
y_test = test[TARGET].astype(int)

if not (
    np.isfinite(X_train.to_numpy()).all()
    and np.isfinite(X_val.to_numpy()).all()
    and np.isfinite(X_test.to_numpy()).all()
):
    raise ValueError("Missing or infinite feature values detected")

print("\nNumber of features:", len(features))

print("\nTraining distribution:")
print(y_train.value_counts())

# Free unnecessary memory
del df
gc.collect()

# ------------------------------------------------------------
# IDENTIFY BINARY FEATURES
# ------------------------------------------------------------

binary_features = [
    c for c in features
    if X_train[c].isin([0, 1]).all()
]

categorical_indices = [
    features.index(c) for c in binary_features
]

print("\nBinary features detected:", len(binary_features))

# ------------------------------------------------------------
# SMOTE FUNCTION
# ------------------------------------------------------------

def apply_smote(X, y):
    counts = y.value_counts()
    ratio = 0.15

    if counts.min() / counts.max() >= ratio:
        raise ValueError("SMOTE ratio must exceed existing ratio")

    if categorical_indices and len(categorical_indices) < len(features):
        sampler = SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy=ratio,
            random_state=SEED,
            k_neighbors=5
        )
    else:
        sampler = SMOTE(
            sampling_strategy=ratio,
            random_state=SEED,
            k_neighbors=5
        )

    X_res, y_res = sampler.fit_resample(X, y)

    return np.asarray(X_res, dtype=np.float32), y_res

def make_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight=None,
        random_state=SEED,
        n_jobs=-1
    )

# ------------------------------------------------------------
# TRAINING WITH SMOTE
# ------------------------------------------------------------

print("\nApplying SMOTE to training data...", flush=True)

X_res, y_res = apply_smote(X_train, y_train)

print("Class distribution after SMOTE:")
print(pd.Series(y_res).value_counts())

model = make_model()

print("\nTraining Random Forest...", flush=True)

model.fit(X_res, y_res)

print("Training completed!", flush=True)

del X_res, y_res
gc.collect()

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

val_prob = model.predict_proba(X_val)[:, 1]

print("\nVALIDATION RESULTS")

print("AUROC:", roc_auc_score(y_val, val_prob))
print("AUPRC:", average_precision_score(y_val, val_prob))

# ------------------------------------------------------------
# THRESHOLD TUNING
# ------------------------------------------------------------

rows = []

for threshold in np.round(np.arange(0.05, 0.951, 0.05), 2):

    pred = (val_prob >= threshold).astype(int)

    rows.append({
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

threshold_df = pd.DataFrame(rows)

print("\nTHRESHOLD COMPARISON")
print(threshold_df.round(4).to_string(index=False))

threshold_df.to_csv(
    OUT / "validation_thresholds.csv", index=False
)

best_threshold = float(
    threshold_df.loc[
        threshold_df["F1"].idxmax(), "Threshold"
    ]
)

print("\nSelected threshold:", best_threshold)

# ------------------------------------------------------------
# TEST EVALUATION
# ------------------------------------------------------------

test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_pred, labels=[0, 1]
).ravel()

auroc = roc_auc_score(y_test, test_prob)
auprc = average_precision_score(y_test, test_prob)
brier = brier_score_loss(y_test, test_prob)

precision = precision_score(y_test, test_pred, zero_division=0)
recall = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)
accuracy = accuracy_score(y_test, test_pred)
specificity = tn / (tn + fp)
npv = tn / (tn + fn)
balanced_acc = (recall + specificity) / 2

metrics = {
    "AUROC": auroc,
    "AUPRC": auprc,
    "Brier Score": brier,
    "Precision": precision,
    "Recall": recall,
    "Specificity": specificity,
    "Accuracy": accuracy,
    "F1": f1,
    "NPV": npv,
    "Balanced Accuracy": balanced_acc,
    "FPR": 1 - specificity,
    "FNR": 1 - recall,
    "Threshold": best_threshold,
    "TN": tn, "FP": fp, "FN": fn, "TP": tp
}

print("\n================================")
print("FINAL 44-FEATURE RF + SMOTE RESULTS")
print("================================")

for key, value in metrics.items():
    print(f"{key}: {value:.5f}")

report = classification_report(
    y_test, test_pred, zero_division=0
)

print("\nCLASSIFICATION REPORT:")
print(report)

pd.DataFrame([metrics]).to_csv(
    OUT / "test_metrics.csv", index=False
)

with open(OUT / "test_results.txt", "w") as f:
    f.write("RF 44 FEATURES + SMOTE\n\n")
    for key, value in metrics.items():
        f.write(f"{key}: {value:.5f}\n")
    f.write("\n" + report)

# ------------------------------------------------------------
# SAVE GRAPHS
# ------------------------------------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test, test_pred,
    values_format="d", cmap="Blues"
)
plt.title("RF 44 + SMOTE - Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=300)
plt.close()

RocCurveDisplay.from_predictions(y_test, test_prob)
plt.title("RF 44 + SMOTE - ROC Curve")
plt.tight_layout()
plt.savefig(OUT / "roc_curve.png", dpi=300)
plt.close()

PrecisionRecallDisplay.from_predictions(y_test, test_prob)
plt.title("RF 44 + SMOTE - PR Curve")
plt.tight_layout()
plt.savefig(OUT / "precision_recall_curve.png", dpi=300)
plt.close()

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

importance.to_csv(OUT / "feature_importance.csv")

importance.head(25).sort_values().plot(
    kind="barh", figsize=(10, 9)
)
plt.title("RF 44 + SMOTE - Feature Importance")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

joblib.dump(
    {
        "model": model,
        "features": features,
        "threshold": best_threshold
    },
    OUT / "rf_chosen_smote.pkl"
)

print("\nALL 44-FEATURE SMOTE RESULTS SAVED!")
