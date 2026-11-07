
# ============================================================
# STEP 47: LAB TEST-LEVEL SHAP FEATURE SELECTION
# ============================================================

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, DMatrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# ============================================================
# 1. PATHS
# ============================================================

PROJECT = Path(
    r"C:\Users\sowmy\Downloads\MIMIC-IV\MIMIC_IV\MIMIC_IV"
)

SOURCE = (
    PROJECT
    / "model training figures"
    / "expanded_lab_xgboost"
)

MODEL_DIR = SOURCE / "step44_final_200_model"
OUT = SOURCE / "step47_lab_selection"

OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
RECALL_TARGET = 0.70

LAB_COUNTS = [10, 15, 20, 25, 30, 46]

# ============================================================
# 2. LOAD SAVED DATA
# ============================================================

print("\nSTEP 47: Loading saved data...")

features = pd.read_parquet(
    SOURCE / "expanded_candidate_features.parquet"
)

cohort = pd.read_csv(
    SOURCE / "cohort_audit.csv"
)

bundle = joblib.load(
    MODEL_DIR / "final_inference_bundle.joblib"
)

selected_names = bundle["selected_feature_names"]

print("Original selected features:", len(selected_names))

# ============================================================
# 3. REPRODUCE DEVELOPMENT SPLIT
# ============================================================

split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

dev_idx, test_idx = next(
    split.split(
        cohort,
        cohort["icu_transfer"],
        cohort["subject_id"]
    )
)

dev_ids = set(cohort.iloc[dev_idx]["hadm_id"])
test_ids = set(cohort.iloc[test_idx]["hadm_id"])

assert not (
    set(cohort.iloc[dev_idx]["subject_id"])
    & set(cohort.iloc[test_idx]["subject_id"])
)

dev = features[
    features["hadm_id"].isin(dev_ids)
].copy().reset_index(drop=True)

print("Development admissions:", len(dev))
print("Reserved test admissions:", len(test_ids))

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
        dev["subject_id"]
    )
)

fit = dev.iloc[fit_idx].copy()
val = dev.iloc[val_idx].copy()

y_fit = fit["icu_transfer"].astype(int).to_numpy()
y_val = val["icu_transfer"].astype(int).to_numpy()

print("Fitting:", len(fit))
print("Validation:", len(val))

# ============================================================
# 5. FIT PREPROCESSING ON FITTING DATA ONLY
# ============================================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

cat_cols = [
    "gender",
    "admission_type",
    "admission_location"
]

num_candidates = [
    "age"
] + [
    c for c in fit.columns
    if c.startswith("lab_")
]

kept = [
    c for c in num_candidates
    if fit[c].notna().mean() >= 0.10
]

pre = ColumnTransformer(
    [
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

for frame in (fit, val):
    for col in cat_cols:
        frame[col] = frame[col].fillna("Unknown").astype(str)

raw_cols = kept + cat_cols

X_fit = np.asarray(
    pre.fit_transform(fit[raw_cols]),
    dtype=np.float32
)

X_val = np.asarray(
    pre.transform(val[raw_cols]),
    dtype=np.float32
)

names = list(pre.get_feature_names_out())

# ============================================================
# 6. XGBOOST SETTINGS
# ============================================================

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
    random_state=42,
    n_jobs=4
)

# ============================================================
# 7. TRAIN RANKING MODEL AND CALCULATE SHAP
# ============================================================

print("\nCalculating SHAP importance...")

ranker = XGBClassifier(**PARAMS)
ranker.fit(X_fit, y_fit)

rng = np.random.default_rng(SEED)

sample_idx = rng.choice(
    len(X_fit),
    size=min(2000, len(X_fit)),
    replace=False
)

contributions = ranker.get_booster().predict(
    DMatrix(
        X_fit[sample_idx],
        feature_names=names
    ),
    pred_contribs=True
)

importance = np.abs(
    contributions[:, :-1]
).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": names,
    "importance": importance
})

# ============================================================
# 8. GROUP FEATURES BY LAB TEST ID
# ============================================================

def lab_id_from_feature(name):

    if not name.startswith("lab_"):
        return None

    parts = name.split("_")

    if len(parts) >= 3 and parts[1].isdigit():
        return parts[1]

    return None


importance_df["lab_id"] = (
    importance_df["feature"].map(
        lab_id_from_feature
    )
)

lab_importance = (
    importance_df
    .dropna(subset=["lab_id"])
    .groupby("lab_id")
    .agg(
        total_shap=("importance", "sum"),
        feature_count=("feature", "count")
    )
    .reset_index()
    .sort_values(
        "total_shap",
        ascending=False
    )
)

lab_importance.to_csv(
    OUT / "lab_shap_ranking.csv",
    index=False
)

print("\nTop 20 laboratory test IDs:")
print(lab_importance.head(20).to_string(index=False))

# ============================================================
# 9. IDENTIFY NON-LAB FEATURES
# ============================================================

non_lab_indices = [
    i for i, name in enumerate(names)
    if lab_id_from_feature(name) is None
]

lab_ids = lab_importance["lab_id"].tolist()

# ============================================================
# 10. THRESHOLD SELECTION
# ============================================================

def choose_threshold(y_true, probabilities, target=0.70):

    order = np.argsort(-probabilities)
    sorted_y = y_true[order]
    sorted_p = probabilities[order]

    ends = np.r_[
        np.flatnonzero(np.diff(sorted_p) != 0),
        len(sorted_p) - 1
    ]

    tp = np.cumsum(sorted_y)[ends]
    fp = ends + 1 - tp

    recall = tp / max(y_true.sum(), 1)
    precision = tp / np.maximum(tp + fp, 1)

    eligible = np.where(recall >= target)[0]

    if len(eligible) == 0:
        raise ValueError("Recall target cannot be achieved")

    best_precision = precision[eligible].max()

    candidates = eligible[
        precision[eligible] == best_precision
    ]

    best = candidates[np.argmax(sorted_p[ends[candidates]])]

    return float(sorted_p[ends[best]])


# ============================================================
# 11. COMPARE LAB TEST COUNTS
# ============================================================

results = []

for count in LAB_COUNTS:

    selected_labs = set(lab_ids[:count])

    selected_indices = [
        i for i, name in enumerate(names)
        if (
            lab_id_from_feature(name) in selected_labs
            or i in non_lab_indices
        )
    ]

    model = XGBClassifier(**PARAMS)

    model.fit(
        X_fit[:, selected_indices],
        y_fit
    )

    probabilities = model.predict_proba(
        X_val[:, selected_indices]
    )[:, 1]

    threshold = choose_threshold(
        y_val,
        probabilities,
        RECALL_TARGET
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    ).ravel()

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)

    results.append({
        "lab_tests": count,
        "total_features": len(selected_indices),
        "auroc": roc_auc_score(
            y_val, probabilities
        ),
        "auprc": average_precision_score(
            y_val, probabilities
        ),
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "false_positives": int(fp),
        "true_positives": int(tp),
        "threshold": threshold
    })

    print(
        f"{count} labs | "
        f"{len(selected_indices)} features | "
        f"AUROC={results[-1]['auroc']:.4f} | "
        f"AUPRC={results[-1]['auprc']:.4f} | "
        f"Recall={recall:.3f} | "
        f"Precision={precision:.3f} | "
        f"FP={fp:,}"
    )

# ============================================================
# 12. SAVE COMPARISON
# ============================================================

comparison = pd.DataFrame(results)

comparison.to_csv(
    OUT / "lab_count_comparison.csv",
    index=False
)

print("\nLAB TEST COMPARISON:")
print(comparison.to_string(index=False))

# Exploratory selection: highest precision at recall >= 70%
best = comparison.sort_values(
    ["precision", "auprc", "total_features"],
    ascending=[False, False, True]
).iloc[0]

best_count = int(best["lab_tests"])

print("\nBEST LAB COUNT:", best_count)

# Save selected laboratory IDs
pd.DataFrame({
    "lab_id": lab_ids[:best_count]
}).to_csv(
    OUT / "selected_lab_ids.csv",
    index=False
)

# ============================================================
# 13. VISUALIZATIONS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    comparison["lab_tests"],
    comparison["precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    comparison["lab_tests"],
    comparison["recall"],
    marker="s",
    label="Recall"
)

plt.xlabel("Number of Laboratory Tests")
plt.ylabel("Score")
plt.title("Precision and Recall vs Lab Test Count")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "lab_count_precision_recall.png",
    dpi=300
)

plt.close()

plt.figure(figsize=(8, 5))

plt.plot(
    comparison["lab_tests"],
    comparison["auprc"],
    marker="o"
)

plt.xlabel("Number of Laboratory Tests")
plt.ylabel("AUPRC")
plt.title("AUPRC vs Number of Laboratory Tests")
plt.tight_layout()

plt.savefig(
    OUT / "lab_count_auprc.png",
    dpi=300
)

plt.close()

print("\nSTEP 47 COMPLETE")
print("Results saved to:", OUT)
print("Original 200-feature model unchanged.")
print("Reserved test set not evaluated.")
