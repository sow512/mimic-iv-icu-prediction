
import pandas as pd
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
    PrecisionRecallDisplay
)

# STEP 1: CREATE RESULTS FOLDER
results = Path("results_smote")
results.mkdir(exist_ok=True)

# STEP 2: LOAD DATA
train = pd.read_csv("kidney_final_train_dataset.csv")
test = pd.read_csv("kidney_final_test_dataset.csv")

X_train = train.drop(columns=["icu_transfer"])
y_train = train["icu_transfer"]

X_test = test.drop(columns=["icu_transfer"])
y_test = test["icu_transfer"]

# Check that features match
assert list(X_train.columns) == list(X_test.columns)

# SMOTE requires numeric, non-missing inputs
assert X_train.select_dtypes(exclude="number").empty
assert not X_train.isna().any().any()

print("BEFORE SMOTE:")
print(y_train.value_counts())

# STEP 3: CREATE SMOTE + RANDOM FOREST PIPELINE
model = Pipeline([
    ("smote", SMOTE(
        sampling_strategy=0.5,
        random_state=42,
        k_neighbors=5
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

# STEP 4: TRAIN MODEL
print("\nTraining Random Forest with SMOTE...")
model.fit(X_train, y_train)
print("Training completed!")

# STEP 5: MAKE PREDICTIONS
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# STEP 6: EVALUATE
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\nAUROC:", roc_auc_score(y_test, y_prob))
print("AUPRC:", average_precision_score(y_test, y_prob))

# STEP 7: SAVE CONFUSION MATRIX
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Random Forest with SMOTE")
plt.tight_layout()
plt.savefig(results / "confusion_matrix.png", dpi=300)
plt.close()

# STEP 8: SAVE ROC CURVE
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve - Random Forest with SMOTE")
plt.tight_layout()
plt.savefig(results / "roc_curve.png", dpi=300)
plt.close()

# STEP 9: SAVE PRECISION-RECALL CURVE
PrecisionRecallDisplay.from_predictions(y_test, y_prob)
plt.title("Precision-Recall Curve - RF with SMOTE")
plt.tight_layout()
plt.savefig(results / "precision_recall_curve.png", dpi=300)
plt.close()

# STEP 10: FEATURE IMPORTANCE
importance = pd.Series(
    model.named_steps["forest"].feature_importances_,
    index=X_train.columns
).sort_values()

importance.plot(kind="barh", figsize=(10, 8))
plt.title("Feature Importance - RF with SMOTE")
plt.tight_layout()
plt.savefig(results / "feature_importance.png", dpi=300)
plt.close()

# STEP 11: SAVE MODEL
joblib.dump(model, results / "random_forest_smote.pkl")

print("\nAll SMOTE results saved successfully!")
