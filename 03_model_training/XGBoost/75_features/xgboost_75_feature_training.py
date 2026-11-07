
# ============================================================
# STEP 53: FINALIZE TIMING-CORRECTED 75-FEATURE XGBOOST
# ============================================================

import os
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

PROJECT = Path(
    os.environ["MIMIC_PROJECT_ROOT"]
)

SOURCE = (
    PROJECT / "model training figures"
    / "expanded_lab_xgboost"
)

CORRECTED_FILE = (
    SOURCE / "step51_timing_corrected"
    / "expanded_candidate_features_storetime_corrected.parquet"
)

COHORT_FILE = SOURCE / "cohort_audit.csv"

OUT = SOURCE / "step53_final_75_model"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FEATURES = 75
RECALL_TARGET = 0.70

PARAMS = dict(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=5,
    scale_pos_weight=1.0,
    tree_method="hist",
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=4
)

# ============================================================
# 2. LOAD TIMING-CORRECTED DATA
# ============================================================

print("\nSTEP 53: Loading corrected dataset...")

features = pd.read_parquet(CORRECTED_FILE)

cohort = pd.read_csv(
    COHORT_FILE,
    usecols=["hadm_id", "subject_id", "icu_transfer"]
)

if features.hadm_id.duplicated().any():
    raise ValueError("Duplicate feature admission IDs")

if cohort.hadm_id.duplicated().any():
    raise ValueError("Duplicate cohort admission IDs")

if set(features.hadm_id) != set(cohort.hadm_id):
    raise ValueError("Feature/cohort admission mismatch")

# Verify target alignment
target_lookup = cohort.set_index("hadm_id")["icu_transfer"]

if not np.array_equal(
    features.hadm_id.map(target_lookup).astype(int).to_numpy(),
    features.icu_transfer.astype(int).to_numpy()
):
    raise ValueError("Outcome mismatch")

# ============================================================
# 3. REPRODUCE ORIGINAL DEVELOPMENT / TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

dev_idx, test_idx = next(
    splitter.split(
        cohort,
        cohort.icu_transfer,
        groups=cohort.subject_id
    )
)

dev_cohort = cohort.iloc[dev_idx]
test_cohort = cohort.iloc[test_idx]

assert not (
    set(dev_cohort.subject_id)
    & set(test_cohort.subject_id)
)

dev = features[
    features.hadm_id.isin(dev_cohort.hadm_id)
].copy().reset_index(drop=True)

if len(dev) != len(dev_cohort):
    raise ValueError("Development size mismatch")

print("Development admissions:", len(dev))
print("Reserved test admissions:", len(test_cohort))

# ============================================================
# 4. PREPROCESSING FUNCTIONS
# ============================================================

CAT_COLS = [
    "gender",
    "admission_type",
    "admission_location"
]

NUM_COLS = ["age"] + [
    c for c in dev.columns
    if c.startswith("lab_")
]

def fit_preprocessor(frame):

    kept = [
        c for c in NUM_COLS
        if frame[c].notna().mean() >= 0.10
    ]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", kept),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CAT_COLS
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    data = frame[kept + CAT_COLS].copy()

    for col in CAT_COLS:
        data[col] = data[col].fillna("Unknown").astype(str)

    X = np.asarray(
        pre.fit_transform(data),
        dtype=np.float32
    )

    return pre, X, list(pre.get_feature_names_out()), kept


def transform_data(pre, frame, kept):

    data = frame[kept + CAT_COLS].copy()

    for col in CAT_COLS:
        data[col] = data[col].fillna("Unknown").astype(str)

    return np.asarray(
        pre.transform(data),
        dtype=np.float32
    )

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

inner = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=2026
)

fit_idx, val_idx = next(
    inner.split(
        dev,
        dev.icu_transfer,
        groups=dev.subject_id
    )
)

fit = dev.iloc[fit_idx].copy()
val = dev.iloc[val_idx].copy()

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

selected_fit = ranking_fit[:N_FEATURES]

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

selected_idx = ranking_final[:N_FEATURES]

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
    "raw_columns": kept_final + CAT_COLS,
    "categorical_columns": CAT_COLS,
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
    "data_source": str(CORRECTED_FILE),
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
