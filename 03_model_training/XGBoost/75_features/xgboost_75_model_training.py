
# ============================================================
# STEP 53: FINALIZE TIMING-CORRECTED 75-FEATURE XGBOOST
# ============================================================

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier, DMatrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

# ============================================================
# 1. PATHS AND SETTINGS - MAC / VS CODE
# ============================================================
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

EXCEL_FILE = PROJECT / "75_Feature_Dataset.xlsx"

OUT = PROJECT / "results_xgb_75_tuning"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FEATURES = 75
RECALL_TARGET = 0.70


# ============================================================
# 2. LOAD 75-FEATURE EXCEL DATASET
# ============================================================

print("\nLoading 75-feature Excel dataset...")

features = pd.read_excel(
    EXCEL_FILE,
    engine="openpyxl"
)

print("Dataset shape:", features.shape)
print("Number of columns:", len(features.columns))

required = [
    "subject_id",
    "hadm_id",
    "dataset_split",
    "icu_transfer"
]

missing = [
    col for col in required
    if col not in features.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if features["hadm_id"].duplicated().any():
    raise ValueError("Duplicate admission IDs detected")

print("\nDataset loaded successfully!")

# ============================================================
# 3. USE THE DATASET'S EXISTING SPLIT
# ============================================================

print("\nAvailable dataset splits:")
print(features["dataset_split"].value_counts())

features["dataset_split"] = (
    features["dataset_split"]
    .astype(str)
    .str.strip()
    .str.lower()
)

allowed = {"train", "validation", "test"}

if not set(features["dataset_split"]).issubset(allowed):
    raise ValueError("Unexpected dataset_split values")

dev = features[
    features["dataset_split"].isin(["train", "validation"])
].copy().reset_index(drop=True)

test_cohort = features[
    features["dataset_split"] == "test"
].copy().reset_index(drop=True)

if dev.empty or test_cohort.empty:
    raise ValueError("Development or test data is empty")

if set(dev["subject_id"]) & set(test_cohort["subject_id"]):
    raise ValueError("Patient overlap detected")

print("\nDevelopment admissions:", len(dev))
print("Reserved test admissions:", len(test_cohort))


# ============================================================
# 4. PREPROCESSING
# ============================================================

from sklearn.impute import SimpleImputer

EXCLUDE_COLS = [
    "subject_id",
    "hadm_id",
    "dataset_split",
    "icu_transfer"
]

FEATURE_COLS = [
    col for col in features.columns
    if col not in EXCLUDE_COLS
]

print("\nNumber of features:", len(FEATURE_COLS))

if len(FEATURE_COLS) != 75:
    raise ValueError("Expected exactly 75 features")

def fit_preprocessor(frame):

    data = frame[FEATURE_COLS].copy()

    for col in FEATURE_COLS:
        data[col] = pd.to_numeric(
            data[col],
            errors="raise"
        )

    pre = SimpleImputer(
        strategy="median",
        keep_empty_features=True
    )

    X = pre.fit_transform(data).astype(np.float32)

    return pre, X, FEATURE_COLS.copy(), FEATURE_COLS.copy()


def transform_data(pre, frame, kept):

    data = frame[kept].copy()

    for col in kept:
        data[col] = pd.to_numeric(
            data[col],
            errors="raise"
        )

    return pre.transform(data).astype(np.float32)

# ============================================================
# 5. SHAP FEATURE RANKING
# ============================================================

def rank_features(X, y, feature_names, seed):

    ranker = XGBClassifier(**PARAMS)
    ranker.fit(X, y)

    rng = np.random.default_rng(seed)

    sample_idx = rng.choice(
        len(X),
        size=min(2000, len(X)),
        replace=False
    )

    shap_values = ranker.get_booster().predict(
        DMatrix(
            X[sample_idx],
            feature_names=feature_names
        ),
        pred_contribs=True
    )

    importance = np.abs(
        shap_values[:, :-1]
    ).mean(axis=0)

    ranking = np.argsort(
        -importance,
        kind="stable"
    )

    return ranking, importance

# ============================================================
# 6. VALIDATION-BASED THRESHOLD SELECTION
# ============================================================

def choose_threshold(y_true, probabilities, target=0.70):

    order = np.argsort(
        -probabilities,
        kind="stable"
    )

    sorted_y = y_true[order]
    sorted_p = probabilities[order]

    ends = np.r_[
        np.flatnonzero(np.diff(sorted_p) != 0),
        len(sorted_p) - 1
    ]

    tp = np.cumsum(sorted_y)[ends]
    fp = ends + 1 - tp

    recall = tp / max(int(y_true.sum()), 1)
    precision = tp / np.maximum(tp + fp, 1)

    eligible = np.flatnonzero(recall >= target)

    if len(eligible) == 0:
        raise ValueError("Recall target unavailable")

    best_precision = precision[eligible].max()

    candidates = eligible[
        precision[eligible] == best_precision
    ]

    best = candidates[
        np.argmax(sorted_p[ends[candidates]])
    ]

    return float(sorted_p[ends[best]])

# ============================================================
# 7. CREATE DEVELOPMENT-ONLY THRESHOLD SPLIT
# ============================================================

print("\nSelecting threshold using development data...")

fit = features[
    features["dataset_split"] == "train"
].copy().reset_index(drop=True)

val = features[
    features["dataset_split"] == "validation"
].copy().reset_index(drop=True)

if set(fit["subject_id"]) & set(val["subject_id"]):
    raise ValueError("Patient overlap between train and validation")

print("Training records:", len(fit))
print("Validation records:", len(val))
assert not (
    set(fit.subject_id) & set(val.subject_id)
)

y_fit = fit.icu_transfer.astype(int).to_numpy()
y_val = val.icu_transfer.astype(int).to_numpy()

pre_fit, X_fit, fit_names, kept_fit = fit_preprocessor(fit)

X_val = transform_data(
    pre_fit,
    val,
    kept_fit
)

# Rank on fitting patients only
ranking_fit, _ = rank_features(
    X_fit,
    y_fit,
    fit_names,
    SEED
)

selected_fit = np.arange(N_FEATURES)

validation_model = XGBClassifier(**PARAMS)

validation_model.fit(
    X_fit[:, selected_fit],
    y_fit
)

p_val = validation_model.predict_proba(
    X_val[:, selected_fit]
)[:, 1]

threshold = choose_threshold(
    y_val,
    p_val,
    RECALL_TARGET
)

pred_val = (p_val >= threshold).astype(int)

tn, fp, fn, tp = map(
    int,
    confusion_matrix(
        y_val,
        pred_val,
        labels=[0, 1]
    ).ravel()
)

validation_metrics = {
    "auroc": float(roc_auc_score(y_val, p_val)),
    "auprc": float(average_precision_score(y_val, p_val)),
    "recall": tp / (tp + fn),
    "precision": tp / (tp + fp) if tp + fp else 0,
    "specificity": tn / (tn + fp),
    "tp": tp,
    "fp": fp,
    "fn": fn,
    "tn": tn
}

print("\nVALIDATION METRICS")
print(json.dumps(validation_metrics, indent=2))
print("Selected threshold:", round(threshold, 6))

pd.DataFrame({
    "hadm_id": val.hadm_id.to_numpy(),
    "actual": y_val,
    "probability": p_val,
    "predicted": pred_val
}).to_csv(
    OUT / "threshold_validation_predictions.csv",
    index=False
)

# ============================================================
# 8. FINAL PREPROCESSING ON ALL DEVELOPMENT DATA
# ============================================================

print("\nRefitting preprocessing on all development admissions...")

pre_final, X_dev, all_names, kept_final = fit_preprocessor(dev)

y_dev = dev.icu_transfer.astype(int).to_numpy()

# ============================================================
# 9. FINAL SHAP FEATURE SELECTION
# ============================================================

print("Selecting final 75 features...")

ranking_final, importance_final = rank_features(
    X_dev,
    y_dev,
    all_names,
    SEED + 1
)

selected_idx = np.arange(N_FEATURES)

selected_names = [
    all_names[i]
    for i in selected_idx
]

# ============================================================
# 10. TRAIN FINAL MODEL
# ============================================================

print("\nTraining final 75-feature XGBoost...")

final_model = XGBClassifier(**PARAMS)

final_model.fit(
    X_dev[:, selected_idx],
    y_dev
)

# ============================================================
# 11. SAVE MODEL AND PREPROCESSING
# ============================================================

final_model.save_model(
    str(OUT / "final_xgboost_75_model.json")
)

joblib.dump(
    pre_final,
    OUT / "final_preprocessing_pipeline.joblib"
)

pd.DataFrame({
    "rank": np.arange(1, N_FEATURES + 1),
    "feature": selected_names,
    "mean_abs_shap": importance_final[selected_idx]
}).to_csv(
    OUT / "final_75_selected_features.csv",
    index=False
)

# Save inference bundle
bundle = {
    "preprocessor": pre_final,
    "raw_columns": FEATURE_COLS,
    "categorical_columns": [],
    "selected_indices": selected_idx.tolist(),
    "selected_feature_names": selected_names
}

joblib.dump(
    bundle,
    OUT / "final_inference_bundle.joblib"
)

# ============================================================
# 12. SAVE METADATA
# ============================================================

lab_ids = {
    name.split("_")[1]
    for name in selected_names
    if name.startswith("lab_")
    and len(name.split("_")) >= 3
    and name.split("_")[1].isdigit()
}

metadata = {
    "model": "XGBoost",
    "feature_count": N_FEATURES,
    "unique_lab_ids": len(lab_ids),
    "selected_threshold": threshold,
    "recall_target": RECALL_TARGET,
    "validation_metrics": validation_metrics,
    "development_admissions": len(dev),
    "reserved_test_admissions": len(test_cohort),
    "model_params": PARAMS,
    "data_source": str(EXCEL_FILE),
    "note": (
        "Uses Step 51 storetime-corrected laboratory features. "
        "Threshold selected on development-only validation. "
        "Final model refitted on all development admissions. "
        "Threshold is provisional after refitting. "
        "Outcome review flags and early-discharge policy "
        "remain unresolved. No test evaluation performed."
    )
}

with open(
    OUT / "final_model_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metadata, f, indent=2)

# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("STEP 53 COMPLETE")
print("=" * 65)

print("Final model: XGBoost")
print("Selected predictors:", len(selected_names))
print("Unique laboratory IDs:", len(lab_ids))
print("Development admissions:", len(dev))
print("Selected threshold:", round(threshold, 6))

print("\nFiles saved to:")
print(OUT)

print("\nNo reserved test data evaluated.")
print("Previous models remain unchanged.")
