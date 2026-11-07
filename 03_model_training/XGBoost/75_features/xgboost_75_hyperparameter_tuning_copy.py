
# ============================================================
# BN5212 FINAL PROJECT
# XGBOOST - 75 FEATURES - HYPERPARAMETER TUNING
# MAC / VS CODE / EXCEL VERSION
# ============================================================

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    ParameterSampler,
    GroupShuffleSplit
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_recall_curve,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# ============================================================
# STEP 1: SETTINGS AND FILE PATHS
# ============================================================

PROJECT = Path(__file__).resolve().parent

# EXACT filename shown in your VS Code screenshot
EXCEL_FILE = PROJECT / "75_Feature_Dataset.xlsx"

OUT = PROJECT / "results_xgb_75_tuning"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
RECALL_TARGET = 0.70
N_CANDIDATES = 25

if not EXCEL_FILE.exists():
    raise FileNotFoundError(
        f"Excel dataset not found: {EXCEL_FILE}"
    )

print("\n========================================")
print("75-FEATURE XGBOOST HYPERPARAMETER TUNING")
print("========================================")

print("\nDataset file:", EXCEL_FILE.name)
print("Results folder:", OUT)

# ============================================================
# STEP 2: LOAD EXCEL DATASET
# ============================================================

print("\nLoading 75-feature Excel dataset...", flush=True)

data = pd.read_excel(
    EXCEL_FILE,
    sheet_name="Final_75_Features",
    engine="openpyxl"
)

print("\nDataset shape:", data.shape)

required_columns = [
    "subject_id",
    "hadm_id",
    "dataset_split",
    "icu_transfer"
]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

data["dataset_split"] = (
    data["dataset_split"]
    .astype(str)
    .str.strip()
    .str.lower()
)

allowed_splits = {"train", "validation", "test"}

if not set(data["dataset_split"]).issubset(allowed_splits):
    raise ValueError(
        "Unexpected dataset_split labels found."
    )

if data["subject_id"].isna().any():
    raise ValueError("Missing subject_id values detected.")

if data["hadm_id"].isna().any():
    raise ValueError("Missing hadm_id values detected.")

if data["hadm_id"].duplicated().any():
    raise ValueError("Duplicate admission IDs detected.")

print("\nDataset split distribution:")
print(data["dataset_split"].value_counts())


# ============================================================
# STEP 3: CREATE PATIENT-LEVEL VALIDATION SPLIT
# ============================================================

# Original Excel file contains only train and test.
# We create validation data from the original train set.
# The reserved test set is not used during tuning.

development = data[
    data["dataset_split"] == "train"
].copy().reset_index(drop=True)

test = data[
    data["dataset_split"] == "test"
].copy().reset_index(drop=True)

if len(development) == 0 or len(test) == 0:
    raise ValueError("Training or testing data is empty.")

# Check that no patient appears in both original groups
development_ids = set(development["subject_id"])
test_ids = set(test["subject_id"])

if development_ids & test_ids:
    raise ValueError(
        "Patient overlap between development and test data."
    )

# Create an 80/20 patient-level split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        development,
        development["icu_transfer"],
        groups=development["subject_id"]
    )
)

train = development.iloc[
    train_idx
].copy().reset_index(drop=True)

val = development.iloc[
    val_idx
].copy().reset_index(drop=True)

# Verify patient separation
train_ids = set(train["subject_id"])
val_ids = set(val["subject_id"])

if train_ids & val_ids:
    raise ValueError(
        "Patient overlap between training and validation."
    )

if train_ids & test_ids:
    raise ValueError(
        "Patient overlap between training and testing."
    )

if val_ids & test_ids:
    raise ValueError(
        "Patient overlap between validation and testing."
    )

# Verify both classes exist in each partition
for name, frame in [
    ("Training", train),
    ("Validation", val),
    ("Testing", test)
]:
    if frame["icu_transfer"].nunique() != 2:
        raise ValueError(
            f"{name} set does not contain both classes."
        )

print("\nPatient-level split check passed!")

print("\nTraining records:", len(train))
print("Validation records:", len(val))
print("Reserved testing records:", len(test))

print("\nTraining class distribution:")
print(train["icu_transfer"].value_counts())

print("\nValidation class distribution:")
print(val["icu_transfer"].value_counts())

print("\nTesting class distribution:")
print(test["icu_transfer"].value_counts())


# ============================================================
# STEP 4: SELECT THE 75 FEATURES
# ============================================================

exclude_columns = [
    "subject_id",
    "hadm_id",
    "dataset_split",
    "icu_transfer"
]

feature_columns = [
    col for col in data.columns
    if col not in exclude_columns
]

print("\nNumber of input features:", len(feature_columns))

if len(feature_columns) != 75:
    raise ValueError(
        f"Expected 75 features, found {len(feature_columns)}"
    )

X_train = train[feature_columns].apply(
    pd.to_numeric,
    errors="raise"
)

X_val = val[feature_columns].apply(
    pd.to_numeric,
    errors="raise"
)

X_test = test[feature_columns].apply(
    pd.to_numeric,
    errors="raise"
)

y_train = train["icu_transfer"].astype(int).to_numpy()
y_val = val["icu_transfer"].astype(int).to_numpy()
y_test = test["icu_transfer"].astype(int).to_numpy()

for name, labels in [
    ("TRAINING", y_train),
    ("VALIDATION", y_val),
    ("TESTING", y_test)
]:

    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(
            f"Both target classes required in {name}."
        )

    print(f"\n{name} CLASS DISTRIBUTION:")
    print(pd.Series(labels).value_counts().sort_index())

# ============================================================
# STEP 5: PREPROCESS MISSING VALUES
# ============================================================

print("\nPreprocessing dataset...")

imputer = SimpleImputer(
    strategy="median",
    keep_empty_features=True
)

# Fit imputer only on training data
X_train_processed = imputer.fit_transform(
    X_train
).astype(np.float32)

X_val_processed = imputer.transform(
    X_val
).astype(np.float32)

print("Preprocessing completed!")

# ============================================================
# STEP 6: ORIGINAL XGBOOST PARAMETERS
# ============================================================

original_params = {
    "n_estimators": 250,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 5,
    "scale_pos_weight": 1.0
}

fixed_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": 4
}

# ============================================================
# STEP 7: HYPERPARAMETER SEARCH SPACE
# ============================================================

search_space = {
    "n_estimators": [150, 250, 400, 600],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.10],
    "subsample": [0.7, 0.85, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "reg_lambda": [1, 5, 10],
    "scale_pos_weight": [1.0, 5.0, 10.0, 20.0]
}

candidates = [original_params.copy()]

candidates += list(
    ParameterSampler(
        search_space,
        n_iter=N_CANDIDATES,
        random_state=SEED
    )
)

print("\nTotal configurations:", len(candidates))

# ============================================================
# STEP 8: HYPERPARAMETER TUNING
# ============================================================

print("\n========================================")
print("STARTING HYPERPARAMETER TUNING")
print("========================================", flush=True)

tuning_results = []

best_auprc = -1.0
best_params = None

for i, params in enumerate(candidates):

    print(
        f"\nTesting combination {i + 1}/{len(candidates)}",
        flush=True
    )

    model = XGBClassifier(
        **fixed_params,
        **params
    )

    model.fit(
        X_train_processed,
        y_train
    )

    probabilities = model.predict_proba(
        X_val_processed
    )[:, 1]

    auroc = roc_auc_score(
        y_val,
        probabilities
    )

    auprc = average_precision_score(
        y_val,
        probabilities
    )

    brier = brier_score_loss(
        y_val,
        probabilities
    )

    tuning_results.append({
        "Combination": i + 1,
        "Type": "Original" if i == 0 else "Tuned",
        "AUROC": auroc,
        "AUPRC": auprc,
        "Brier Score": brier,
        **params
    })

    print("AUROC:", round(auroc, 5), flush=True)
    print("AUPRC:", round(auprc, 5), flush=True)
    print("Brier Score:", round(brier, 5), flush=True)

    if auprc > best_auprc:

        best_auprc = auprc
        best_params = params.copy()

# ============================================================
# STEP 9: SAVE TUNING RESULTS
# ============================================================

results_df = pd.DataFrame(tuning_results)

results_df = results_df.sort_values(
    "AUPRC",
    ascending=False
)

results_df.to_csv(
    OUT / "hyperparameter_tuning_results.csv",
    index=False
)

print("\n========================================")
print("HYPERPARAMETER TUNING COMPLETED")
print("========================================")

print("\nOriginal hyperparameters:")
print(original_params)

print("\nBest hyperparameters:")
print(best_params)

print("\nBest validation AUPRC:")
print(best_auprc)

print("\nTop 5 configurations:")
print(results_df.head(5).to_string(index=False))

# ============================================================
# STEP 10: TRAIN ORIGINAL AND BEST MODELS
# ============================================================

print("\nTraining original model...")

original_model = XGBClassifier(
    **fixed_params,
    **original_params
)

original_model.fit(
    X_train_processed,
    y_train
)

print("\nTraining best tuned model...")

best_model = XGBClassifier(
    **fixed_params,
    **best_params
)

best_model.fit(
    X_train_processed,
    y_train
)

original_prob = original_model.predict_proba(
    X_val_processed
)[:, 1]

tuned_prob = best_model.predict_proba(
    X_val_processed
)[:, 1]

# ============================================================
# STEP 11: SELECT CLASSIFICATION THRESHOLD
# ============================================================

def choose_threshold(y_true, probabilities, recall_target):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    eligible = np.where(
        recall[:-1] >= recall_target
    )[0]

    if len(eligible) == 0:
        raise ValueError(
            "No threshold satisfies the recall target."
        )

    best_precision = precision[eligible].max()

    tied = eligible[
        np.isclose(
            precision[eligible],
            best_precision
        )
    ]

    best_index = tied[
        np.argmax(thresholds[tied])
    ]

    return float(thresholds[best_index])


original_threshold = choose_threshold(
    y_val,
    original_prob,
    RECALL_TARGET
)

tuned_threshold = choose_threshold(
    y_val,
    tuned_prob,
    RECALL_TARGET
)

print("\nOriginal model threshold:", original_threshold)
print("Tuned model threshold:", tuned_threshold)

# ============================================================
# STEP 12: EVALUATION FUNCTION
# ============================================================

def evaluate_model(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0.0
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0 else 0.0
    )

    metrics = {
        "AUROC": roc_auc_score(
            y_true,
            probabilities
        ),
        "AUPRC": average_precision_score(
            y_true,
            probabilities
        ),
        "Sensitivity / Recall": sensitivity,
        "Specificity": specificity,
        "Precision (PPV)": precision,
        "Accuracy": (tp + tn) / len(y_true),
        "F1-Score": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "Negative Predictive Value (NPV)": npv,
        "Balanced Accuracy": balanced_accuracy_score(
            y_true,
            predictions
        ),
        "False Positive Rate": 1 - specificity,
        "False Negative Rate": 1 - sensitivity,
        "Brier Score": brier_score_loss(
            y_true,
            probabilities
        ),
        "Classification Threshold": threshold,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    return metrics, predictions

# ============================================================
# STEP 13: COMPARE ORIGINAL AND TUNED MODELS
# ============================================================

original_metrics, original_pred = evaluate_model(
    y_val,
    original_prob,
    original_threshold
)

tuned_metrics, tuned_pred = evaluate_model(
    y_val,
    tuned_prob,
    tuned_threshold
)

comparison = pd.DataFrame({
    "Original XGBoost": original_metrics,
    "Tuned XGBoost": tuned_metrics
})

print("\n========================================")
print("ORIGINAL VS TUNED XGBOOST")
print("VALIDATION RESULTS")
print("========================================")

print(comparison.round(5).to_string())

comparison.to_csv(
    OUT / "original_vs_tuned_comparison.csv"
)

print("\nTUNED CLASSIFICATION REPORT:")
print(classification_report(
    y_val,
    tuned_pred,
    zero_division=0
))

print("\nTUNED CONFUSION MATRIX:")
print(confusion_matrix(
    y_val,
    tuned_pred
))

# ============================================================
# STEP 14: SAVE VALIDATION PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "hadm_id": val["hadm_id"].to_numpy(),
    "actual": y_val,
    "original_probability": original_prob,
    "tuned_probability": tuned_prob,
    "original_prediction": original_pred,
    "tuned_prediction": tuned_pred
})

predictions_df.to_csv(
    OUT / "validation_predictions.csv",
    index=False
)

# ============================================================
# STEP 15: SAVE CONFUSION MATRIX
# ============================================================

ConfusionMatrixDisplay.from_predictions(
    y_val,
    tuned_pred,
    values_format="d",
    cmap="Blues"
)

plt.title("75-Feature Tuned XGBoost - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUT / "confusion_matrix.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 16: SAVE ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_val,
    tuned_prob
)

plt.title("75-Feature Tuned XGBoost - ROC Curve")
plt.tight_layout()

plt.savefig(
    OUT / "roc_curve.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 17: SAVE PRECISION-RECALL CURVE
# ============================================================

PrecisionRecallDisplay.from_predictions(
    y_val,
    tuned_prob
)

plt.title("75-Feature Tuned XGBoost - PR Curve")
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
    best_model.feature_importances_,
    index=feature_columns
).sort_values(ascending=False)

importance.to_csv(
    OUT / "feature_importance.csv",
    header=["Importance"]
)

plt.figure(figsize=(11, 9))

importance.head(25).sort_values().plot(
    kind="barh"
)

plt.title("75-Feature Tuned XGBoost - Feature Importance")
plt.xlabel("Importance")

plt.tight_layout()

plt.savefig(
    OUT / "feature_importance.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 19: TRAIN FINAL MODEL ON DEVELOPMENT DATA
# ============================================================

print("\nTraining final model on training + validation...")

development = pd.concat(
    [train, val],
    ignore_index=True
)

X_development = development[
    feature_columns
].apply(
    pd.to_numeric,
    errors="raise"
)

y_development = development[
    "icu_transfer"
].astype(int).to_numpy()

final_imputer = SimpleImputer(
    strategy="median",
    keep_empty_features=True
)

X_development_processed = final_imputer.fit_transform(
    X_development
).astype(np.float32)

final_model = XGBClassifier(
    **fixed_params,
    **best_params
)

final_model.fit(
    X_development_processed,
    y_development
)

print("Final model training completed!")

# ============================================================
# STEP 20: SAVE FINAL MODEL
# ============================================================

final_model.save_model(
    str(OUT / "final_xgboost_75_model.json")
)

joblib.dump(
    {
        "model": final_model,
        "imputer": final_imputer,
        "feature_columns": feature_columns,
        "validation_threshold": tuned_threshold
    },
    OUT / "final_model_bundle.joblib"
)

# ============================================================
# STEP 21: SAVE METADATA
# ============================================================

metadata = {
    "dataset": EXCEL_FILE.name,
    "worksheet": "Final_75_Features",
    "number_of_features": len(feature_columns),
    "training_records": len(train),
    "validation_records": len(val),
    "reserved_testing_records": len(test),
    "recall_target": RECALL_TARGET,
    "original_parameters": original_params,
    "best_parameters": best_params,
    "original_validation_metrics": original_metrics,
    "tuned_validation_metrics": tuned_metrics,
    "selected_validation_threshold": tuned_threshold,
    "note": (
        "Hyperparameters and threshold were selected "
        "using the same validation set. Validation "
        "performance is optimistic. Final model was "
        "refitted on all development data. Threshold "
        "requires revalidation. Reserved test data "
        "was not evaluated."
    )
}

with open(
    OUT / "model_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        metadata,
        file,
        indent=2
    )


# ============================================================
# STEP 22: FINAL EVALUATION ON RESERVED TEST ADMISSIONS
# ============================================================

print("\n========================================")
print("75-FEATURE XGBOOST - FINAL TEST EVALUATION")
print("========================================", flush=True)

print("\nTotal reserved test admissions:", len(test))
print("Actual ICU transfers:", int(np.sum(y_test == 1)))
print("Actual non-ICU transfers:", int(np.sum(y_test == 0)))

# IMPORTANT:
# The test set has not been used for hyperparameter
# selection, threshold selection, or model training.

# Use the final imputer fitted on development data only.
X_test_processed = final_imputer.transform(
    X_test
).astype(np.float32)

# ============================================================
# STEP 23: RETRAIN ORIGINAL MODEL FOR FAIR COMPARISON
# ============================================================

print("\nTraining original XGBoost on full development data...",
      flush=True)

original_final_model = XGBClassifier(
    **fixed_params,
    **original_params
)

original_final_model.fit(
    X_development_processed,
    y_development
)

print("Original model training completed!", flush=True)

# The tuned final_model was already trained in STEP 19.

# ============================================================
# STEP 24: GENERATE TEST PROBABILITIES
# ============================================================

print("\nGenerating test predictions...", flush=True)

original_test_prob = original_final_model.predict_proba(
    X_test_processed
)[:, 1]

tuned_test_prob = final_model.predict_proba(
    X_test_processed
)[:, 1]

# Use thresholds previously selected on validation data.
# Do NOT select new thresholds using test labels.

original_test_metrics, original_test_pred = evaluate_model(
    y_test,
    original_test_prob,
    original_threshold
)

tuned_test_metrics, tuned_test_pred = evaluate_model(
    y_test,
    tuned_test_prob,
    tuned_threshold
)

# ============================================================
# STEP 25: COMPARE ORIGINAL VS TUNED TEST RESULTS
# ============================================================

test_comparison = pd.DataFrame({
    "Original XGBoost": original_test_metrics,
    "Tuned XGBoost": tuned_test_metrics
})

print("\n========================================")
print("ORIGINAL VS TUNED XGBOOST")
print("FINAL TEST RESULTS")
print("========================================")

print(test_comparison.round(5).to_string())

test_comparison.to_csv(
    OUT / "final_test_comparison.csv"
)

# ============================================================
# STEP 26: CLASSIFICATION REPORT AND CONFUSION MATRIX
# ============================================================

print("\n========================================")
print("TUNED XGBOOST - FINAL TEST CLASSIFICATION")
print("========================================")

report = classification_report(
    y_test,
    tuned_test_pred,
    labels=[0, 1],
    target_names=["No ICU Transfer", "ICU Transfer"],
    zero_division=0
)

print("\nCLASSIFICATION REPORT:")
print(report)

cm = confusion_matrix(
    y_test,
    tuned_test_pred,
    labels=[0, 1]
)

print("\nCONFUSION MATRIX:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nTrue Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# ============================================================
# STEP 27: SAVE FINAL TEST CONFUSION MATRIX
# ============================================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    tuned_test_pred,
    labels=[0, 1],
    display_labels=["No ICU", "ICU"],
    values_format="d",
    cmap="Blues"
)

plt.title("75-Feature Tuned XGBoost - Test Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUT / "final_test_confusion_matrix.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 28: SAVE FINAL TEST ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    tuned_test_prob
)

plt.title("75-Feature Tuned XGBoost - Test ROC Curve")
plt.tight_layout()

plt.savefig(
    OUT / "final_test_roc_curve.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 29: SAVE FINAL TEST PRECISION-RECALL CURVE
# ============================================================

PrecisionRecallDisplay.from_predictions(
    y_test,
    tuned_test_prob
)

plt.title("75-Feature Tuned XGBoost - Test PR Curve")
plt.tight_layout()

plt.savefig(
    OUT / "final_test_precision_recall_curve.png",
    dpi=300
)

plt.close()

# ============================================================
# STEP 30: SAVE TEST PREDICTIONS
# ============================================================

test_predictions = pd.DataFrame({
    "subject_id": test["subject_id"].to_numpy(),
    "hadm_id": test["hadm_id"].to_numpy(),
    "actual_icu_transfer": y_test,
    "original_probability": original_test_prob,
    "tuned_probability": tuned_test_prob,
    "original_prediction": original_test_pred,
    "tuned_prediction": tuned_test_pred
})

test_predictions.to_csv(
    OUT / "final_test_predictions.csv",
    index=False
)

# ============================================================
# STEP 31: SAVE ALL FINAL TEST RESULTS AS TEXT
# ============================================================

with open(
    OUT / "final_test_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("75-FEATURE XGBOOST FINAL TEST RESULTS\n")
    f.write("====================================\n\n")

    f.write(f"Test admissions: {len(test)}\n")
    f.write(f"Actual ICU transfers: {int(np.sum(y_test == 1))}\n\n")

    f.write("BEST HYPERPARAMETERS:\n")
    f.write(str(best_params))
    f.write("\n\n")

    f.write("ORIGINAL VS TUNED TEST RESULTS:\n")
    f.write(test_comparison.round(5).to_string())
    f.write("\n\n")

    f.write("TUNED CLASSIFICATION REPORT:\n")
    f.write(report)
    f.write("\n")

    f.write("TUNED CONFUSION MATRIX:\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write("TN: " + str(tn) + "\n")
    f.write("FP: " + str(fp) + "\n")
    f.write("FN: " + str(fn) + "\n")
    f.write("TP: " + str(tp) + "\n\n")

    f.write(
        "Thresholds were selected on validation data "
        "and applied unchanged to the reserved test set.\n"
    )

# ============================================================
# STEP 32: UPDATE METADATA
# ============================================================

metadata["original_test_metrics"] = original_test_metrics
metadata["tuned_test_metrics"] = tuned_test_metrics
metadata["test_evaluated"] = True
metadata["test_evaluation_note"] = (
    "Reserved test set evaluated once after model and "
    "threshold selection. Original and tuned models "
    "were fitted on all development admissions. "
    "Test labels were not used to select thresholds."
)

with open(
    OUT / "model_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metadata, f, indent=2)

# ============================================================
# STEP 33: FINAL SUMMARY
# ============================================================

print("\n========================================")
print("FINAL TEST EVALUATION COMPLETED")
print("========================================")

print("\nBest hyperparameters:")
print(best_params)

print("\nOriginal validation AUPRC:",
      round(original_metrics["AUPRC"], 5))

print("Tuned validation AUPRC:",
      round(tuned_metrics["AUPRC"], 5))

print("\nOriginal test AUPRC:",
      round(original_test_metrics["AUPRC"], 5))

print("Tuned test AUPRC:",
      round(tuned_test_metrics["AUPRC"], 5))

print("\nTuned test AUROC:",
      round(tuned_test_metrics["AUROC"], 5))

print("Tuned test Brier Score:",
      round(tuned_test_metrics["Brier Score"], 5))

print("\nSelected tuned threshold:",
      round(tuned_threshold, 6))

print("\nResults saved in:")
print(OUT)

print("\nALL FINAL TEST RESULTS SAVED SUCCESSFULLY!")
