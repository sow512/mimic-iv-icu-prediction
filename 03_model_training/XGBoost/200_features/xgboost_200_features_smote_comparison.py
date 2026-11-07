
# ============================================================
# STEP 67: 200-FEATURE XGBOOST + SMOTENC EXPERIMENT
# Original Step 44 dataset; original model untouched
# ============================================================

import os
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_score, recall_score,
    f1_score, accuracy_score, brier_score_loss,
    log_loss, roc_curve, precision_recall_curve
)

# ============================================================
# 1. PATHS
# ============================================================

PROJECT = Path(
    os.environ["MIMIC_PROJECT_ROOT"]
)

SOURCE = PROJECT / "model training figures" / "expanded_lab_xgboost"
MODEL_DIR = SOURCE / "step44_final_200_model"

OUT = SOURCE / "step67_smote_200_experiment"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET_RECALL = 0.70
SMOTE_RATIO = 0.30

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
# 2. LOAD ORIGINAL FEATURES AND SAVED FEATURE SELECTION
# ============================================================

print("STEP 67: Loading original 200-feature data...")

features = pd.read_parquet(
    SOURCE / "expanded_candidate_features.parquet"
)

cohort = pd.read_csv(
    SOURCE / "cohort_audit.csv",
    usecols=["hadm_id", "subject_id", "icu_transfer"]
)

bundle = joblib.load(
    MODEL_DIR / "final_inference_bundle.joblib"
)

with open(
    MODEL_DIR / "final_model_metadata.json",
    encoding="utf-8"
) as f:
    original_meta = json.load(f)

if cohort.hadm_id.duplicated().any():
    raise ValueError("Duplicate cohort admissions")

if features.hadm_id.duplicated().any():
    raise ValueError("Duplicate feature admissions")

if set(cohort.hadm_id) != set(features.hadm_id):
    raise ValueError("Cohort and features do not match")

# ============================================================
# 3. REPRODUCE ORIGINAL PATIENT-LEVEL SPLIT
# ============================================================

split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

dev_idx, test_idx = next(
    split.split(
        cohort,
        cohort.icu_transfer,
        groups=cohort.subject_id
    )
)

dev_cohort = cohort.iloc[dev_idx]
test_cohort = cohort.iloc[test_idx]

assert not (
    set(dev_cohort.subject_id) &
    set(test_cohort.subject_id)
)

# Preserve the exact admission order
features = features.set_index("hadm_id", drop=False)

dev = features.loc[
    dev_cohort.hadm_id
].copy().reset_index(drop=True)

test = features.loc[
    test_cohort.hadm_id
].copy().reset_index(drop=True)

y_dev = dev_cohort.icu_transfer.astype(int).to_numpy()
y_test = test_cohort.icu_transfer.astype(int).to_numpy()

print("Development admissions:", len(dev))
print("Test admissions:", len(test))

# ============================================================
# 4. TRANSFORM EXACT SAVED 200 FEATURES
# ============================================================

pre = bundle["preprocessor"]
raw_cols = bundle["raw_columns"]
cat_cols = bundle["categorical_columns"]
selected_idx = np.asarray(
    bundle["selected_indices"], dtype=int
)
selected_names = list(bundle["selected_feature_names"])

if len(selected_names) != 200:
    raise ValueError("Expected 200 features")

def transform(frame):
    data = frame[raw_cols].copy()

    for c in cat_cols:
        data[c] = data[c].fillna("Unknown").astype(str)

    X = np.asarray(
        pre.transform(data),
        dtype=np.float32
    )

    return X[:, selected_idx]

X_dev = transform(dev)
X_test = transform(test)

encoded_names = list(pre.get_feature_names_out())

if [encoded_names[i] for i in selected_idx] != selected_names:
    raise ValueError("Feature ordering mismatch")

# Identify encoded categorical indicators.
categorical_idx = [
    i for i, name in enumerate(selected_names)
    if any(
        name.startswith(c + "_")
        for c in cat_cols
    )
]

numeric_idx = [
    i for i in range(200)
    if i not in categorical_idx
]

print("Predictors:", X_dev.shape[1])
print("Categorical indicators:", len(categorical_idx))
print("Numerical predictors:", len(numeric_idx))

# ============================================================
# 5. DEVELOPMENT-ONLY THRESHOLD VALIDATION
# ============================================================

inner = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=2026
)

fit_idx, val_idx = next(
    inner.split(
        dev, y_dev,
        groups=dev.subject_id
    )
)

assert not (
    set(dev.iloc[fit_idx].subject_id) &
    set(dev.iloc[val_idx].subject_id)
)

X_fit = X_dev[fit_idx]
X_val = X_dev[val_idx]

y_fit = y_dev[fit_idx]
y_val = y_dev[val_idx]

# ============================================================
# 6. SMOTENC PREPARATION
# ============================================================

def prepare_smote(X_train, X_other):

    # Median imputation is fitted on training data only.
    # Missingness indicators preserve whether values
    # were originally unavailable.

    numeric_train = X_train[:, numeric_idx]
    numeric_other = X_other[:, numeric_idx]

    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
        keep_empty_features=True
    )

    num_train = imputer.fit_transform(numeric_train)
    num_other = imputer.transform(numeric_other)

    cat_train = X_train[:, categorical_idx]
    cat_other = X_other[:, categorical_idx]

    if len(categorical_idx):
        A = np.column_stack([num_train, cat_train])
        B = np.column_stack([num_other, cat_other])

        cat_positions = list(
            range(num_train.shape[1], A.shape[1])
        )
    else:
        A = num_train
        B = num_other
        cat_positions = []

    return (
        np.asarray(A, dtype=np.float32),
        np.asarray(B, dtype=np.float32),
        cat_positions,
        imputer
    )

def make_oversampled(X, y, cat_positions):

    if not cat_positions:
        raise ValueError(
            "No categorical indicators found. "
            "SMOTENC requires categorical columns."
        )

    sampler = SMOTENC(
        categorical_features=cat_positions,
        sampling_strategy=SMOTE_RATIO,
        random_state=SEED,
        k_neighbors=5
    )

    return sampler.fit_resample(X, y)

# ============================================================
# 7. THRESHOLD SELECTION
# ============================================================

def choose_threshold(y, prob, target=0.70):

    order = np.argsort(-prob, kind="stable")
    sorted_y = y[order]
    sorted_p = prob[order]

    ends = np.r_[
        np.flatnonzero(np.diff(sorted_p) != 0),
        len(sorted_p) - 1
    ]

    tp = np.cumsum(sorted_y)[ends]
    fp = ends + 1 - tp

    recall = tp / max(int(y.sum()), 1)
    precision = tp / np.maximum(tp + fp, 1)

    eligible = np.flatnonzero(recall >= target)

    if len(eligible) == 0:
        raise ValueError("Recall target not achievable")

    best_precision = precision[eligible].max()
    candidates = eligible[
        precision[eligible] == best_precision
    ]

    best = candidates[
        np.argmax(sorted_p[ends[candidates]])
    ]

    return float(sorted_p[ends[best]])

# ============================================================
# 8. TRAIN VALIDATION MODELS
# ============================================================

print("\nSTEP 1: Selecting thresholds...")

X_fit_ready, X_val_ready, cat_positions, _ = (
    prepare_smote(X_fit, X_val)
)

thresholds = {}

for name in ["baseline", "smotenc"]:

    if name == "smotenc":
        X_train, y_train = make_oversampled(
            X_fit_ready, y_fit, cat_positions
        )
    else:
        X_train, y_train = X_fit_ready, y_fit

    model = XGBClassifier(**PARAMS)
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_val_ready)[:, 1]

    threshold = choose_threshold(
        y_val, prob, TARGET_RECALL
    )

    thresholds[name] = threshold

    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, pred, labels=[0, 1]
    ).ravel()

    print(
        f"{name}: threshold={threshold:.6f} "
        f"AP={average_precision_score(y_val, prob):.4f} "
        f"recall={tp/(tp+fn):.3f} "
        f"precision={tp/(tp+fp):.3f} "
        f"FP={fp:,}"
    )

# ============================================================
# 9. REFIT ON FULL DEVELOPMENT DATA
# ============================================================

print("\nSTEP 2: Training final comparison models...")

X_dev_ready, X_test_ready, cat_positions, imputer = (
    prepare_smote(X_dev, X_test)
)

results = []
predictions = pd.DataFrame({
    "hadm_id": test.hadm_id.to_numpy(),
    "actual": y_test
})

curves = {}

for name in ["baseline", "smotenc"]:

    if name == "smotenc":
        X_train, y_train = make_oversampled(
            X_dev_ready, y_dev, cat_positions
        )
    else:
        X_train, y_train = X_dev_ready, y_dev

    print(
        name,
        "training counts:",
        np.bincount(y_train)
    )

    model = XGBClassifier(**PARAMS)
    model.fit(X_train, y_train)

    model.save_model(
        str(OUT / f"{name}_xgboost_200.json")
    )

    threshold = thresholds[name]

    prob = model.predict_proba(X_test_ready)[:, 1]
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = map(
        int,
        confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).ravel()
    )

    metrics = {
        "model": name,
        "threshold": threshold,
        "auroc": roc_auc_score(y_test, prob),
        "auprc": average_precision_score(y_test, prob),
        "recall": recall_score(y_test, pred),
        "precision": precision_score(
            y_test, pred, zero_division=0
        ),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_test, pred),
        "accuracy": accuracy_score(y_test, pred),
        "brier": brier_score_loss(y_test, prob),
        "log_loss": log_loss(y_test, prob),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "alerts": int(pred.sum())
    }

    results.append(metrics)

    predictions[f"{name}_probability"] = prob
    predictions[f"{name}_prediction"] = pred

    curves[name] = prob

    # Confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i,j]:,}",
                ha="center", va="center",
                fontsize=13,
                color=(
                    "white"
                    if cm[i,j] > cm.max()/2
                    else "black"
                )
            )

    ax.set_xticks([0, 1], ["No ICU", "ICU"])
    ax.set_yticks([0, 1], ["No ICU", "ICU"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"{name.upper()} - 200 features\n"
        f"Threshold={threshold:.5f}"
    )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(
        OUT / f"{name}_confusion_matrix.png",
        dpi=300
    )
    plt.close(fig)

# ============================================================
# 10. SAVE METRICS
# ============================================================

comparison = pd.DataFrame(results)

comparison.to_csv(
    OUT / "test_metrics_comparison.csv",
    index=False
)

predictions.to_csv(
    OUT / "test_predictions_comparison.csv",
    index=False
)

joblib.dump(
    imputer,
    OUT / "smote_experiment_imputer.joblib"
)

# ============================================================
# 11. ROC CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

for name, prob in curves.items():
    fpr, tpr, _ = roc_curve(y_test, prob)

    ax.plot(
        fpr, tpr,
        label=(
            f"{name} AUROC="
            f"{roc_auc_score(y_test, prob):.4f}"
        )
    )

ax.plot([0, 1], [0, 1], "--", label="Chance")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("200-feature XGBoost: ROC comparison")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "roc_comparison.png", dpi=300)
plt.close(fig)

# ============================================================
# 12. PRECISION-RECALL CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

for name, prob in curves.items():
    precision, recall, _ = precision_recall_curve(
        y_test, prob
    )

    ax.plot(
        recall, precision,
        label=(
            f"{name} AP="
            f"{average_precision_score(y_test, prob):.4f}"
        )
    )

ax.axhline(
    y_test.mean(),
    linestyle="--",
    label="Positive prevalence"
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("200-feature XGBoost: PR comparison")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "pr_comparison.png", dpi=300)
plt.close(fig)

# ============================================================
# 13. CALIBRATION CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    [0, 1], [0, 1],
    "--", label="Perfect calibration"
)

for name, prob in curves.items():

    table = pd.DataFrame({
        "probability": prob,
        "actual": y_test
    })

    table["bin"] = pd.qcut(
        table.probability,
        q=10,
        labels=False,
        duplicates="drop"
    )

    grouped = table.groupby("bin").agg(
        predicted=("probability", "mean"),
        observed=("actual", "mean")
    )

    ax.plot(
        grouped.predicted,
        grouped.observed,
        marker="o",
        label=name
    )

ax.set_xlim(0, 0.30)
ax.set_ylim(0, 0.30)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed ICU transfer frequency")
ax.set_title("Calibration comparison")
ax.legend()
fig.tight_layout()
fig.savefig(
    OUT / "calibration_comparison.png",
    dpi=300
)
plt.close(fig)

# ============================================================
# 14. METRIC COMPARISON CHART
# ============================================================

plot_metrics = [
    "auroc", "auprc", "recall",
    "precision", "specificity", "f1"
]

plot_data = comparison.set_index("model")[plot_metrics].T

fig, ax = plt.subplots(figsize=(10, 5))
plot_data.plot(kind="bar", ax=ax)

ax.set_ylabel("Score")
ax.set_title("200-feature XGBoost: Baseline vs SMOTENC")
ax.tick_params(axis="x", rotation=35)
ax.legend(title="Model")
fig.tight_layout()
fig.savefig(
    OUT / "metrics_comparison.png",
    dpi=300
)
plt.close(fig)

# ============================================================
# 15. FINAL SUMMARY
# ============================================================

with open(
    OUT / "experiment_notes.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump({
        "feature_count": 200,
        "smote_ratio": SMOTE_RATIO,
        "target_recall": TARGET_RECALL,
        "development_admissions": len(dev),
        "test_admissions": len(test),
        "original_model_unchanged": True,
        "test_evaluation": "exploratory",
        "limitations": [
            "Uses original pre-timing-correction features",
            "Saved feature selection reused",
            "SMOTENC treats encoded categorical dummies separately",
            "Median imputation differs from original XGBoost",
            "Thresholds selected on development validation",
            "Test set previously examined"
        ]
    }, f, indent=2)

print("\nTEST METRICS COMPARISON")
print(comparison.to_string(index=False))

print("\nSTEP 67 COMPLETE")
print("Results saved to:", OUT)
print("Original 200-feature model unchanged.")
