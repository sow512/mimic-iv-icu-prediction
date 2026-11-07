
# ============================================================
# BN5212 - RANDOM FOREST 75 FEATURES + SMOTE
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
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
OUT = Path("results_rf_75_smote")
OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("Loading 75-feature dataset...", flush=True)

df = pd.read_excel(
    "75_Feature_Dataset.xlsx",
    sheet_name="Final_75_Features"
)

df.columns = df.columns.str.strip()
df["dataset_split"] = (
    df["dataset_split"].astype(str).str.strip().str.lower()
)

if set(df["dataset_split"].unique()) != {"train", "test"}:
    raise ValueError("Unexpected dataset split values")

train = df[df["dataset_split"] == "train"].copy()
test = df[df["dataset_split"] == "test"].copy()

target = "icu_transfer"
exclude = ["subject_id", "hadm_id", "dataset_split", target]
features = [c for c in df.columns if c not in exclude]

assert len(features) == 75

if train["subject_id"].isna().any() or test["subject_id"].isna().any():
    raise ValueError("Missing patient IDs")

if set(train["subject_id"]) & set(test["subject_id"]):
    raise ValueError("Patient overlap detected")

X = train[features].apply(pd.to_numeric, errors="raise")
y = train[target].astype(int)

X_test = test[features].apply(pd.to_numeric, errors="raise")
y_test = test[target].astype(int)

print("Training records:", len(X))
print("Testing records:", len(X_test))
print("Features:", len(features))

# ------------------------------------------------------------
# PATIENT-LEVEL VALIDATION SPLIT
# ------------------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=SEED
)

tr_idx, val_idx = next(
    splitter.split(X, y, groups=train["subject_id"])
)

X_train = X.iloc[tr_idx]
X_val = X.iloc[val_idx]
y_train = y.iloc[tr_idx]
y_val = y.iloc[val_idx]

# ------------------------------------------------------------
# IDENTIFY BINARY FEATURES
# ------------------------------------------------------------

binary_features = [
    c for c in features
    if X_train[c].dropna().isin([0, 1]).all()
    and X_train[c].notna().any()
]

categorical_indices = [
    features.index(c) for c in binary_features
]

print("Detected binary features:", len(binary_features))

def resample_data(X_fit, y_fit, X_eval):
    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True
    )

    X_fit_imp = imputer.fit_transform(X_fit)
    X_eval_imp = imputer.transform(X_eval)

    ratio = 0.15
    counts = y_fit.value_counts()

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

    X_res, y_res = sampler.fit_resample(
        X_fit_imp, y_fit
    )

    return X_res, y_res, X_eval_imp, imputer

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
# VALIDATION MODEL
# ------------------------------------------------------------

print("\nApplying SMOTE...", flush=True)

X_res, y_res, X_val_imp, _ = resample_data(
    X_train, y_train, X_val
)

print("Class distribution after SMOTE:")
print(pd.Series(y_res).value_counts())

model = make_model()

print("\nTraining Random Forest...", flush=True)
model.fit(X_res, y_res)

val_prob = model.predict_proba(X_val_imp)[:, 1]

print("Validation AUROC:", roc_auc_score(y_val, val_prob))
print("Validation AUPRC:", average_precision_score(y_val, val_prob))

# ------------------------------------------------------------
# THRESHOLD TUNING
# ------------------------------------------------------------

rows = []

for threshold in np.round(np.arange(0.05, 0.951, 0.05), 2):
    pred = (val_prob >= threshold).astype(int)

    rows.append({
        "Threshold": threshold,
        "Precision": precision_score(y_val, pred, zero_division=0),
        "Recall": recall_score(y_val, pred, zero_division=0),
        "F1": f1_score(y_val, pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_val, pred)
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
# FINAL MODEL - FULL TRAINING DATA WITH SMOTE
# ------------------------------------------------------------

print("\nApplying SMOTE to full training data...", flush=True)

X_res, y_res, X_test_imp, final_imputer = resample_data(
    X, y, X_test
)

final_model = make_model()

print("Training final Random Forest...", flush=True)
final_model.fit(X_res, y_res)

# ------------------------------------------------------------
# TEST EVALUATION
# ------------------------------------------------------------

test_prob = final_model.predict_proba(X_test_imp)[:, 1]
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

print("\nFINAL RESULTS - 75 FEATURES + SMOTE")
for key, value in metrics.items():
    print(f"{key}: {value:.5f}")

report = classification_report(
    y_test, test_pred, zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

pd.DataFrame([metrics]).to_csv(
    OUT / "test_metrics.csv", index=False
)

with open(OUT / "test_results.txt", "w") as f:
    f.write("RF 75 FEATURES + SMOTE\n\n")
    for key, value in metrics.items():
        f.write(f"{key}: {value:.5f}\n")
    f.write("\n" + report)

# ------------------------------------------------------------
# SAVE GRAPHS
# ------------------------------------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test, test_pred, values_format="d", cmap="Blues"
)
plt.title("RF 75 + SMOTE - Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=300)
plt.close()

RocCurveDisplay.from_predictions(y_test, test_prob)
plt.title("RF 75 + SMOTE - ROC")
plt.tight_layout()
plt.savefig(OUT / "roc_curve.png", dpi=300)
plt.close()

PrecisionRecallDisplay.from_predictions(y_test, test_prob)
plt.title("RF 75 + SMOTE - PR Curve")
plt.tight_layout()
plt.savefig(OUT / "precision_recall_curve.png", dpi=300)
plt.close()

importance = pd.Series(
    final_model.feature_importances_,
    index=features
).sort_values(ascending=False)

importance.to_csv(OUT / "feature_importance.csv")

importance.head(25).sort_values().plot(
    kind="barh", figsize=(10, 9)
)
plt.title("RF 75 + SMOTE - Feature Importance")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# SAVE MODEL AND PREPROCESSING
# ------------------------------------------------------------

joblib.dump(
    {
        "model": final_model,
        "imputer": final_imputer,
        "features": features,
        "threshold": best_threshold
    },
    OUT / "rf_75_smote.pkl"
)

print("\nALL 75-FEATURE SMOTE RESULTS SAVED!")
