
# ============================================================
# BN5212 FINAL PROJECT
# TUNED RANDOM FOREST FOR ICU TRANSFER PREDICTION
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# ============================================================
# STEP 1: CREATE RESULTS FOLDER
# ============================================================

results_folder = Path("results_tuned")
results_folder.mkdir(exist_ok=True)

print("Results folder created!")

# ============================================================
# STEP 2: LOAD TRAINING DATA
# ============================================================

print("\nLoading kidney patient dataset...")

train = pd.read_csv("kidney_final_train_dataset.csv")

print("Training dataset shape:", train.shape)

# ============================================================
# STEP 3: SEPARATE FEATURES AND TARGET
# ============================================================

X = train.drop(columns=["icu_transfer"])
y = train["icu_transfer"]

print("\nTarget distribution:")
print(y.value_counts())

# ============================================================
# STEP 4: SPLIT INTO TRAINING AND VALIDATION SETS
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))

# ============================================================
# STEP 5: CREATE RANDOM FOREST MODEL
# ============================================================

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=1
)

# ============================================================
# STEP 6: DEFINE HYPERPARAMETERS
# ============================================================

parameters = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_leaf": [2, 5, 10],
    "min_samples_split": [2, 5, 10],
    "class_weight": [None, "balanced"]
}

# ============================================================
# STEP 7: HYPERPARAMETER TUNING
# ============================================================

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=parameters,
    n_iter=15,
    scoring="average_precision",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nSearching for best Random Forest settings...")
print("This may take several minutes.\n")

search.fit(X_train, y_train)

print("\nBest hyperparameters:")
print(search.best_params_)

print("\nBest cross-validation AUPRC:")
print(search.best_score_)

# ============================================================
# STEP 8: GET BEST MODEL
# ============================================================

best_model = search.best_estimator_

print("\nBest Random Forest model selected!")

# ============================================================
# STEP 9: MAKE VALIDATION PREDICTIONS
# ============================================================

y_pred = best_model.predict(X_val)

y_prob = best_model.predict_proba(X_val)[:, 1]

# ============================================================
# STEP 10: EVALUATE MODEL
# ============================================================

print("\n================================")
print("TUNED RANDOM FOREST RESULTS")
print("================================")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_val, y_pred))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_val, y_pred))

auroc = roc_auc_score(y_val, y_prob)
auprc = average_precision_score(y_val, y_prob)

print("\nAUROC:", auroc)
print("AUPRC:", auprc)

# ============================================================
# STEP 11: SAVE CONFUSION MATRIX
# ============================================================

ConfusionMatrixDisplay.from_predictions(
    y_val,
    y_pred,
    values_format="d",
    cmap="Blues"
)

plt.title("Tuned Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    results_folder / "confusion_matrix.png",
    dpi=300
)

plt.close()

print("Confusion matrix saved!")

# ============================================================
# STEP 12: SAVE ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_val,
    y_prob
)

plt.title("Tuned Random Forest - ROC Curve")
plt.tight_layout()

plt.savefig(
    results_folder / "roc_curve.png",
    dpi=300
)

plt.close()

print("ROC curve saved!")

# ============================================================
# STEP 13: SAVE PRECISION-RECALL CURVE
# ============================================================

PrecisionRecallDisplay.from_predictions(
    y_val,
    y_prob
)

plt.title("Tuned Random Forest - Precision-Recall Curve")
plt.tight_layout()

plt.savefig(
    results_folder / "precision_recall_curve.png",
    dpi=300
)

plt.close()

print("Precision-recall curve saved!")

# ============================================================
# STEP 14: FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    best_model.feature_importances_,
    index=X_train.columns
).sort_values()

plt.figure(figsize=(10, 8))

importance.plot(
    kind="barh"
)

plt.title("Tuned Random Forest - Feature Importance")
plt.xlabel("Feature Importance")
plt.ylabel("Clinical Features")
plt.tight_layout()

plt.savefig(
    results_folder / "feature_importance.png",
    dpi=300
)

plt.close()

print("Feature importance graph saved!")

# ============================================================
# STEP 15: SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    results_folder / "random_forest_tuned.pkl"
)

print("\nTuned model saved successfully!")

# ============================================================
# STEP 16: SAVE RESULTS AS TEXT
# ============================================================

with open(results_folder / "model_results.txt", "w") as f:

    f.write("TUNED RANDOM FOREST RESULTS\n\n")

    f.write("Best Hyperparameters:\n")
    f.write(str(search.best_params_))

    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_val, y_pred))

    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_val, y_pred)))

    f.write("\n\nAUROC: " + str(auroc))
    f.write("\nAUPRC: " + str(auprc))

print("\n================================")
print("ALL RESULTS SAVED SUCCESSFULLY!")
print("================================")
