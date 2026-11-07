
# ============================================================
# BN5212 FINAL PROJECT
# RANDOM FOREST - 20 FEATURES WITH SMOTE
# ORIGINAL BALANCED RF + SMOTE
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

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
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss
)

# ============================================================
# STEP 1: CREATE RESULTS FOLDER
# ============================================================

results = Path("results_rf_20_smote_balanced")
results.mkdir(exist_ok=True)

print("Results folder created!", flush=True)

# ============================================================
# STEP 2: LOAD DATA
# ============================================================

print("\nLoading 20-feature dataset...", flush=True)

train = pd.read_csv("kidney_final_train_dataset.csv")
test = pd.read_csv("kidney_final_test_dataset.csv")

target = "icu_transfer"

features = [
    col for col in train.columns
    if col != target
]

X_train = train[features].copy()
y_train = train[target].astype(int)

X_test = test[features].copy()
y_test = test[target].astype(int)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ============================================================
# STEP 3: CHECK DATA
# ============================================================

if len(features) != 20:
    raise ValueError(
        f"Expected 20 features, found {len(features)}"
    )

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "Training and testing features do not match."
    )

for col in features:
    X_train[col] = pd.to_numeric(
        X_train[col], errors="raise"
    )

    X_test[col] = pd.to_numeric(
        X_test[col], errors="raise"
    )

if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError(
        "Training data contains missing or infinite values."
    )

if not np.isfinite(X_test.to_numpy()).all():
    raise ValueError(
        "Testing data contains missing or infinite values."
    )

print("\nBEFORE SMOTE:")
print(y_train.value_counts())

# ============================================================
# STEP 4: CREATE SMOTE + RANDOM FOREST PIPELINE
# ============================================================

model = Pipeline([
    (
        "smote",
        SMOTE(
            sampling_strategy=0.50,
            random_state=42,
            k_neighbors=5
        )
    ),

    (
        "forest",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
])

# ============================================================
# STEP 5: TRAIN MODEL
# ============================================================

print("\nTraining Random Forest with SMOTE...", flush=True)

model.fit(X_train, y_train)

print("Training completed!", flush=True)

# ============================================================
# STEP 6: MAKE PREDICTIONS
# ============================================================

threshold = 0.50

test_prob = model.predict_proba(X_test)[:, 1]

test_pred = (
    test_prob >= threshold
).astype(int)

# ============================================================
# STEP 7: CALCULATE PERFORMANCE METRICS
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_pred,
    labels=[0, 1]
).ravel()

auroc = roc_auc_score(
    y_test,
    test_prob
)

auprc = average_precision_score(
    y_test,
    test_prob
)

brier = brier_score_loss(
    y_test,
    test_prob
)

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

accuracy = accuracy_score(
    y_test,
    test_pred
)

specificity = tn / (tn + fp)

npv = tn / (tn + fn)

balanced_acc = balanced_accuracy_score(
    y_test,
    test_pred
)

fpr = fp / (fp + tn)

fnr = fn / (fn + tp)

# ============================================================
# STEP 8: COMPLETE RESULTS TABLE
# ============================================================

metrics = {
    "AUROC": auroc,
    "AUPRC (Average Precision)": auprc,
    "Sensitivity / Recall": recall,
    "Specificity": specificity,
    "Precision (PPV)": precision,
    "Accuracy": accuracy,
    "F1-Score": f1,
    "Negative Predictive Value (NPV)": npv,
    "Balanced Accuracy": balanced_acc,
    "False Positive Rate": fpr,
    "False Negative Rate": fnr,
    "Brier Score": brier,
    "Classification Threshold": threshold,
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp)
}

print("\n======================================")
print("20 FEATURE RANDOM FOREST WITH SMOTE")
print("======================================")

for metric, value in metrics.items():
    print(f"{metric}: {value:.5f}")

# ============================================================
# STEP 9: CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    test_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=[0, 1]
)

print("\nCLASSIFICATION REPORT:")
print(report)

print("\nCONFUSION MATRIX:")
print(cm)

# ============================================================
# STEP 10: SAVE METRICS AS CSV
# ============================================================

metrics_df = pd.DataFrame(
    list(metrics.items()),
    columns=["Performance Metric", "Value"]
)

metrics_df.to_csv(
    results / "performance_metrics.csv",
    index=False
)

# ============================================================
# STEP 11: SAVE RESULTS AS TEXT
# ============================================================

with open(results / "results.txt", "w") as f:

    f.write(
        "20 FEATURE RANDOM FOREST WITH SMOTE\n\n"
    )

    f.write("PERFORMANCE METRICS:\n\n")

    for metric, value in metrics.items():
        f.write(
            f"{metric}: {value:.5f}\n"
        )

    f.write("\nCLASSIFICATION REPORT:\n")
    f.write(report)

    f.write("\nCONFUSION MATRIX:\n")
    f.write(str(cm))

# ============================================================
# STEP 12: SAVE CONFUSION MATRIX
# ============================================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_pred,
    labels=[0, 1],
    values_format="d",
    cmap="Blues"
)

plt.title(
    "20 Feature RF + SMOTE - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    results / "confusion_matrix.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 13: SAVE ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    test_prob
)

plt.title(
    "20 Feature RF + SMOTE - ROC Curve"
)

plt.tight_layout()

plt.savefig(
    results / "roc_curve.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 14: SAVE PRECISION-RECALL CURVE
# ============================================================

PrecisionRecallDisplay.from_predictions(
    y_test,
    test_prob
)

plt.title(
    "20 Feature RF + SMOTE - PR Curve"
)

plt.tight_layout()

plt.savefig(
    results / "precision_recall_curve.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 15: FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    model.named_steps["forest"].feature_importances_,
    index=features
).sort_values(ascending=False)

importance.to_csv(
    results / "feature_importance.csv",
    header=["Importance"]
)

plt.figure(figsize=(10, 8))

importance.sort_values().plot(
    kind="barh"
)

plt.title(
    "20 Feature RF + SMOTE - Feature Importance"
)

plt.xlabel("Feature Importance")

plt.tight_layout()

plt.savefig(
    results / "feature_importance.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 16: SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    {
        "model": model,
        "features": features,
        "threshold": threshold
    },
    results / "random_forest_20_smote.pkl"
)

# ============================================================
# STEP 17: FINISH
# ============================================================

print("\n======================================")
print("ALL SMOTE RESULTS SAVED SUCCESSFULLY!")
print("======================================")
