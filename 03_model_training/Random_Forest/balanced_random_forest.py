
# BN5212 FINAL PROJECT
# BALANCED RANDOM FOREST
# WITHOUT FEATURE ENGINEERING

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from imblearn.ensemble import BalancedRandomForestClassifier

# STEP 1: CREATE RESULTS FOLDER

folder = Path("results_balanced_rf")
folder.mkdir(exist_ok=True)

# STEP 2: LOAD TRAINING DATA

data = pd.read_csv("kidney_final_train_dataset.csv")

X = data.drop(columns=["icu_transfer"])
y = data["icu_transfer"]

# STEP 3: SPLIT TRAINING DATA

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# STEP 4: CREATE BALANCED RANDOM FOREST

model = BalancedRandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

# STEP 5: TRAIN

print("Training Balanced Random Forest...")
model.fit(X_train, y_train)
print("Training completed!")

# STEP 6: PREDICT PROBABILITIES

y_prob = model.predict_proba(X_val)[:, 1]

# STEP 7: TEST DIFFERENT THRESHOLDS

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:

    y_pred = (y_prob >= threshold).astype(int)

    print("\n======================")
    print("THRESHOLD:", threshold)
    print("======================")

    print(classification_report(
        y_val,
        y_pred,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

# STEP 8: AUROC AND AUPRC

print("\nAUROC:", roc_auc_score(y_val, y_prob))
print("AUPRC:", average_precision_score(y_val, y_prob))

# STEP 9: SAVE CONFUSION MATRIX AT 0.5

y_pred = (y_prob >= 0.5).astype(int)

ConfusionMatrixDisplay.from_predictions(
    y_val, y_pred, values_format="d"
)
plt.title("Balanced RF Confusion Matrix")
plt.tight_layout()
plt.savefig(folder / "confusion_matrix.png", dpi=300)
plt.close()

# STEP 10: SAVE ROC CURVE

RocCurveDisplay.from_predictions(y_val, y_prob)
plt.title("Balanced Random Forest ROC")
plt.tight_layout()
plt.savefig(folder / "roc_curve.png", dpi=300)
plt.close()

# STEP 11: SAVE PRECISION-RECALL CURVE

PrecisionRecallDisplay.from_predictions(
    y_val, y_prob
)
plt.title("Balanced Random Forest PR Curve")
plt.tight_layout()
plt.savefig(folder / "precision_recall_curve.png", dpi=300)
plt.close()

# STEP 12: SAVE FEATURE IMPORTANCE

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values()

importance.plot(kind="barh", figsize=(10, 8))
plt.title("Balanced RF Feature Importance")
plt.tight_layout()
plt.savefig(folder / "feature_importance.png", dpi=300)
plt.close()

print("\nAll Balanced RF results saved!")
