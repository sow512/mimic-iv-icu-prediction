
# ============================================================
# STEP 48: SMALL-FEATURE XGBOOST COMPARISON
# SHAP FEATURE SELECTION
# ============================================================

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    PROJECT / "model training figures" /
    "expanded_lab_xgboost"
)

OUT = SOURCE / "step48_small_feature_selection"
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_COUNTS = [15, 20, 30, 40, 50, 75, 200]
RECALL_TARGET = 0.70
SEED = 42

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
# 2. LOAD SAVED DATA
# ============================================================

print("\nSTEP 48: Loading expanded dataset...")

features = pd.read_parquet(
    SOURCE / "expanded_candidate_features.parquet"
)

cohort = pd.read_csv(
    SOURCE / "cohort_audit.csv",
    usecols=["hadm_id", "subject_id", "icu_transfer"]
)

if cohort.hadm_id.duplicated().any():
    raise ValueError("Duplicate admission IDs in cohort")

if features.hadm_id.duplicated().any():
    raise ValueError("Duplicate admission IDs in features")

if set(cohort.hadm_id) != set(features.hadm_id):
    raise ValueError("Cohort and feature IDs do not match")

# ============================================================
# 3. PRESERVE ORIGINAL PATIENT-LEVEL TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

dev_idx, test_idx = next(
    splitter.split(
        cohort,
        cohort["icu_transfer"],
        groups=cohort["subject_id"]
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
    raise ValueError("Development cohort mismatch")

# Verify target alignment
labels = dev_cohort.set_index("hadm_id")["icu_transfer"]

if not np.array_equal(
    dev.hadm_id.map(labels).astype(int).to_numpy(),
    dev.icu_transfer.astype(int).to_numpy()
):
    raise ValueError("Outcome labels do not match")

print("Development admissions:", len(dev))
print("Reserved test admissions:", len(test_cohort))

# ============================================================
# 4. PATIENT-GROUPED FIT / VALIDATION SPLIT
# ============================================================

inner = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=2026
)

fit_idx, val_idx = next(
    inner.split(
        dev,
        dev["icu_transfer"],
        groups=dev["subject_id"]
    )
)

fit = dev.iloc[fit_idx].copy()
val = dev.iloc[val_idx].copy()

assert not (
    set(fit.subject_id) & set(val.subject_id)
)

y_fit = fit.icu_transfer.astype(int).to_numpy()
y_val = val.icu_transfer.astype(int).to_numpy()

print("Fitting admissions:", len(fit))
print("Validation admissions:", len(val))
print("Validation ICU cases:", int(y_val.sum()))

# ============================================================
# 5. PREPROCESSING: FIT ONLY ON TRAINING PATIENTS
# ============================================================

cat_cols = [
    "gender",
    "admission_type",
    "admission_location"
]

num_candidates = ["age"] + [
    c for c in fit.columns
    if c.startswith("lab_")
]

# Keep numeric columns with at least 10% coverage
kept = [
    c for c in num_candidates
    if fit[c].notna().mean() >= 0.10
]

for frame in [fit, val]:
    for col in cat_cols:
        frame[col] = (
            frame[col].fillna("Unknown").astype(str)
        )

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", kept),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            cat_cols
        )
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

raw_columns = kept + cat_cols

X_fit = np.asarray(
    preprocessor.fit_transform(fit[raw_columns]),
    dtype=np.float32
)

X_val = np.asarray(
    preprocessor.transform(val[raw_columns]),
    dtype=np.float32
)

feature_names = list(
    preprocessor.get_feature_names_out()
)

print("Available encoded predictors:", len(feature_names))

# ============================================================
# 6. TRAIN RANKING MODEL
# ============================================================

print("\nTraining SHAP ranking model...")

ranking_model = XGBClassifier(**PARAMS)
ranking_model.fit(X_fit, y_fit)

rng = np.random.default_rng(SEED)

sample_idx = rng.choice(
    len(X_fit),
    size=min(2000, len(X_fit)),
    replace=False
)

contributions = ranking_model.get_booster().predict(
    DMatrix(
        X_fit[sample_idx],
        feature_names=feature_names
    ),
    pred_contribs=True
)

importance = np.abs(
    contributions[:, :-1]
).mean(axis=0)

ranking = np.argsort(
    -importance,
    kind="stable"
)

ranking_df = pd.DataFrame({
    "rank": np.arange(1, len(ranking) + 1),
    "feature": np.array(feature_names)[ranking],
    "mean_abs_shap": importance[ranking]
})

ranking_df.to_csv(
    OUT / "shap_feature_ranking.csv",
    index=False
)

# ============================================================
# 7. THRESHOLD SELECTION
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
        raise ValueError("Cannot achieve recall target")

    best_precision = precision[eligible].max()

    candidates = eligible[
        precision[eligible] == best_precision
    ]

    best = candidates[
        np.argmax(sorted_p[ends[candidates]])
    ]

    return float(sorted_p[ends[best]])

# ============================================================
# 8. TRAIN MODELS WITH DIFFERENT FEATURE COUNTS
# ============================================================

results = []

for count in FEATURE_COUNTS:

    if count > len(feature_names):
        continue

    selected_idx = ranking[:count]

    model = XGBClassifier(**PARAMS)

    model.fit(
        X_fit[:, selected_idx],
        y_fit
    )

    prob = model.predict_proba(
        X_val[:, selected_idx]
    )[:, 1]

    threshold = choose_threshold(
        y_val,
        prob,
        RECALL_TARGET
    )

    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = map(
        int,
        confusion_matrix(
            y_val,
            pred,
            labels=[0, 1]
        ).ravel()
    )

    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if tp + fp else 0
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / len(y_val)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    selected_names = [
        feature_names[i] for i in selected_idx
    ]

    lab_ids = {
        name.split("_")[1]
        for name in selected_names
        if name.startswith("lab_")
        and len(name.split("_")) >= 3
        and name.split("_")[1].isdigit()
    }

    results.append({
        "features": count,
        "unique_lab_ids": len(lab_ids),
        "auroc": roc_auc_score(y_val, prob),
        "auprc": average_precision_score(y_val, prob),
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": threshold
    })

    pd.DataFrame({
        "feature": selected_names
    }).to_csv(
        OUT / f"top_{count}_features.csv",
        index=False
    )

    print(
        f"{count:3d} features | "
        f"labs={len(lab_ids):2d} | "
        f"AUROC={results[-1]['auroc']:.4f} | "
        f"AP={results[-1]['auprc']:.4f} | "
        f"Recall={recall:.3f} | "
        f"Precision={precision:.3f} | "
        f"FP={fp:,}"
    )

# ============================================================
# 9. COMPARE RESULTS
# ============================================================

comparison = pd.DataFrame(results)

comparison.to_csv(
    OUT / "small_feature_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("FEATURE COUNT COMPARISON")
print("=" * 65)

print(comparison.to_string(index=False))

# Exploratory model selection:
# maximize precision at >=70% recall;
# break ties using AUPRC and smaller feature count.

eligible = comparison[
    comparison["recall"] >= RECALL_TARGET
].copy()

best = eligible.sort_values(
    ["precision", "auprc", "features"],
    ascending=[False, False, True]
).iloc[0]

best_count = int(best["features"])

print("\nBEST FEATURE COUNT:", best_count)
print("Precision:", round(float(best["precision"]), 4))
print("Recall:", round(float(best["recall"]), 4))
print("False positives:", int(best["fp"]))
print("Unique lab IDs:", int(best["unique_lab_ids"]))

with open(
    OUT / "selection_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "best_feature_count": best_count,
            "selection": "Max precision at recall >= 0.70",
            "validation_only": True,
            "note": (
                "Exploratory comparison. Thresholds and "
                "feature count were selected on the same "
                "validation split. Confirm with grouped CV."
            )
        },
        f,
        indent=2
    )

# ============================================================
# 10. PLOT COMPARISONS
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    comparison["features"],
    comparison["precision"],
    marker="o",
    label="Precision"
)

ax.plot(
    comparison["features"],
    comparison["recall"],
    marker="s",
    label="Recall"
)

ax.set_xlabel("Number of Features")
ax.set_ylabel("Score")
ax.set_title("Precision and Recall vs Feature Count")
ax.legend()
fig.tight_layout()

fig.savefig(
    OUT / "precision_recall_vs_features.png",
    dpi=300
)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    comparison["features"],
    comparison["auprc"],
    marker="o"
)

ax.set_xlabel("Number of Features")
ax.set_ylabel("AUPRC")
ax.set_title("AUPRC vs Feature Count")
fig.tight_layout()

fig.savefig(
    OUT / "auprc_vs_features.png",
    dpi=300
)
plt.close(fig)

print("\nSTEP 48 COMPLETE")
print("Results saved to:", OUT)
print("Original 200-feature model unchanged.")
print("Reserved test set not evaluated.")
