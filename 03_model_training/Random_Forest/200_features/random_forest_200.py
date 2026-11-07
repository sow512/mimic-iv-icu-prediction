
# ============================================================
# BN5212 FINAL PROJECT
# RANDOM FOREST - NEW 200 FEATURE DATASET
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit

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
    balanced_accuracy_score,
    brier_score_loss
)

# ============================================================
# STEP 1: CREATE RESULTS FOLDER
# ============================================================

folder = Path("results_rf_200")
folder.mkdir(exist_ok=True)

print("Results folder created!", flush=True)

# ============================================================
# STEP 2: LOAD EXCEL DATA
# ============================================================

file = "cleaned dataset (200 features).xlsx"

print("\nLoading training data...", flush=True)

train = pd.read_excel(
    file,
    sheet_name="Training Data",
    engine="openpyxl"
)

print("\nLoading testing data...", flush=True)

test = pd.read_excel(
    file,
    sheet_name="Testing Data",
    engine="openpyxl"
)

print("\nTraining shape:", train.shape)
print("Testing shape:", test.shape)

# ============================================================
# STEP 3: CHECK PATIENT OVERLAP
# ============================================================

train_ids = set(train["subject_id"].dropna())
test_ids = set(test["subject_id"].dropna())

overlap = train_ids.intersection(test_ids)

print("\nPatients appearing in both sets:", len(overlap))

if overlap:
    raise ValueError(
        "Patient overlap detected between training and "
        "testing datasets. Fix the split before proceeding."
    )

# ============================================================
# STEP 4: SEPARATE FEATURES AND TARGET
# ============================================================

target = "icu_transfer"

exclude = ["subject_id", "hadm_id", target]

features = [
    col for col in train.columns
    if col not in exclude
]

X = train[features].copy()
y = train[target].astype(int)

X_test = test[features].copy()
y_test = test[target].astype(int)

print("\nNumber of input features:", len(features))

print("\nTraining class distribution:")
print(y.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# Make sure all input columns are numerical
for col in features:
    X[col] = pd.to_numeric(X[col], errors="raise")
    X_test[col] = pd.to_numeric(
        X_test[col], errors="raise"
    )

# ============================================================
# STEP 5: CREATE PATIENT-LEVEL VALIDATION SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
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
        "Both classes must be present in training "
        "and validation sets."
    )

# ============================================================
# STEP 6: CREATE RANDOM FOREST
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
            random_state=42,
            n_jobs=-1
        ))
    ])

model = make_model()

# ============================================================
# STEP 7: TRAIN ON TRAINING SUBSET
# ============================================================

print("\nTraining Random Forest...", flush=True)

model.fit(X_train, y_train)

print("Training completed!", flush=True)

# ============================================================
# STEP 8: VALIDATION PROBABILITIES
# ============================================================

val_prob = model.predict_proba(X_val)[:, 1]

print("\nVALIDATION AUROC:",
      roc_auc_score(y_val, val_prob))

print("VALIDATION AUPRC:",
      average_precision_score(y_val, val_prob))

# ============================================================
# STEP 9: THRESHOLD COMPARISON
# ============================================================

thresholds = np.arange(0.05, 0.96, 0.05)

results = []

for threshold in thresholds:

    pred = (val_prob >= threshold).astype(int)

    results.append({
        "Threshold": round(float(threshold), 2),
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

results_df = pd.DataFrame(results)

print("\nTHRESHOLD COMPARISON:")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    folder / "threshold_comparison.csv",
    index=False
)

# Choose threshold by validation F1
best_row = results_df.loc[
    results_df["F1"].idxmax()
]

best_threshold = float(best_row["Threshold"])

print("\nBest validation F1 threshold:",
      best_threshold)

# ============================================================
# STEP 10: RETRAIN ON ALL TRAINING DATA
# ============================================================

print("\nRetraining on full training data...", flush=True)

final_model = make_model()

final_model.fit(X, y)

print("Final training completed!", flush=True)

# ============================================================
# STEP 11: TEST PREDICTIONS
# ============================================================

test_prob = final_model.predict_proba(
    X_test
)[:, 1]

test_pred = (
    test_prob >= best_threshold
).astype(int)

print("\n================================")
print("FINAL RANDOM FOREST TEST RESULTS")
print("================================")

print("\nSelected threshold:", best_threshold)

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_test, test_pred, zero_division=0
))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, test_pred))

auroc = roc_auc_score(y_test, test_prob)
auprc = average_precision_score(y_test, test_prob)
brier = brier_score_loss(y_test, test_prob)

print("AUPRC:", round(auprc, 4))
print("Brier Score:", round(brier, 4))

print("\nAUROC:", auroc)

# ============================================================
# STEP 12: CONFUSION MATRIX
# ============================================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_pred,
    values_format="d",
    cmap="Blues"
)

plt.title("RF 200 Features - Confusion Matrix")
plt.tight_layout()
plt.savefig(
    folder / "confusion_matrix.png",
    dpi=300
)
plt.close()

# ============================================================
# STEP 13: ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test, test_prob
)

plt.title("RF 200 Features - ROC Curve")
plt.tight_layout()
plt.savefig(
    folder / "roc_curve.png",
    dpi=300
)
plt.close()

# ============================================================
# STEP 14: PRECISION-RECALL CURVE
# ============================================================

PrecisionRecallDisplay.from_predictions(
    y_test, test_prob
)

plt.title("RF 200 Features - PR Curve")
plt.tight_layout()
plt.savefig(
    folder / "precision_recall_curve.png",
    dpi=300
)
plt.close()

# ============================================================
# STEP 15: FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    final_model.named_steps["forest"].feature_importances_,
    index=features
).sort_values(ascending=False)

importance.to_csv(
    folder / "feature_importance.csv",
    header=["Importance"]
)

plt.figure(figsize=(11, 9))

importance.head(25).sort_values().plot(
    kind="barh"
)

plt.title("Top 25 Random Forest Features")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig(
    folder / "feature_importance.png",
    dpi=300
)
plt.close()

# ============================================================
# STEP 16: SAVE MODEL AND RESULTS
# ============================================================

joblib.dump(
    final_model,
    folder / "random_forest_200.pkl"
)

with open(folder / "results.txt", "w") as f:

    f.write("RANDOM FOREST - 200 FEATURES\n\n")
    f.write(f"Threshold: {best_threshold}\n")
    f.write(f"AUROC: {auroc:.4f}\n")
    f.write(f"AUPRC: {auprc:.4f}\n\n")

    f.write("Classification Report:\n")
    f.write(classification_report(
        y_test, test_pred, zero_division=0
    ))

    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(
        y_test, test_pred
    )))

from sklearn.metrics import brier_score_loss

brier = brier_score_loss(y_test, test_prob)

print("Brier Score:", round(brier, 4))
print("\n================================")
print("ALL RESULTS SAVED SUCCESSFULLY!")
print("================================")
