
# ============================================================
# STEP 68: 20-FEATURE XGBOOST + SMOTENC EXPERIMENT
# Complete corrected standalone script
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

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
    roc_curve,
    precision_recall_curve,
)

from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT = Path(
    os.environ["MIMIC_PROJECT_ROOT"]
)

TRAIN_FILE = PROJECT / "kidney_final_train_dataset.csv"
TEST_FILE = PROJECT / "kidney_final_test_dataset.csv"

OUT = PROJECT / "SMOTE_20_XGBOOST_RESULTS"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET_RECALL = 0.70
SMOTE_RATIO = 0.30

FEATURES = [
    "admission_hour",
    "anion_gap_latest_value",
    "calcium_latest_value",
    "creatinine_latest_value",
    "glucose_latest_value",
    "hematocrit_latest_value",
    "pt_latest_value",
    "wbc_latest_value",
    "inr_count",
    "phosphate_count",
    "platelets_count",
    "bun_change",
    "wbc_change",
    "admission_type_DIRECT OBSERVATION",
    "admission_type_ELECTIVE",
    "admission_type_EU OBSERVATION",
    "admission_type_SURGICAL SAME DAY ADMISSION",
    "admission_type_URGENT",
    "admission_location_PHYSICIAN REFERRAL",
    "admission_location_TRANSFER FROM HOSPITAL",
]

CAT_COLS = [
    c for c in FEATURES
    if c.startswith("admission_type_")
    or c.startswith("admission_location_")
]

NUM_COLS = [
    c for c in FEATURES
    if c not in CAT_COLS
]

BASE_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=SEED,
    n_jobs=4,
)


# ============================================================
# 2. LOAD ORIGINAL DATA
# ============================================================

print("\nSTEP 1: Loading original 20-feature datasets...")

train = pd.read_csv(TRAIN_FILE, low_memory=False)
test = pd.read_csv(TEST_FILE, low_memory=False)

required = set(
    FEATURES + ["subject_id", "icu_transfer"]
)

for name, df in [("train", train), ("test", test)]:

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} missing columns: {missing}"
        )

    if df.subject_id.isna().any():
        raise ValueError(
            f"{name} contains missing subject IDs"
        )

    if not df.icu_transfer.isin([0, 1]).all():
        raise ValueError(
            f"{name} contains invalid outcome labels"
        )

if set(train.subject_id) & set(test.subject_id):
    raise ValueError(
        "Patient overlap between train and test"
    )

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Predictors:", len(FEATURES))
print("Training positives:", int(train.icu_transfer.sum()))
print("Testing positives:", int(test.icu_transfer.sum()))


# ============================================================
# 3. PATIENT-DISJOINT VALIDATION SPLIT
# ============================================================

print("\nSTEP 2: Creating validation split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=2026,
)

fit_idx, val_idx = next(
    splitter.split(
        train,
        train.icu_transfer,
        groups=train.subject_id,
    )
)

fit = train.iloc[fit_idx].copy()
val = train.iloc[val_idx].copy()

if set(fit.subject_id) & set(val.subject_id):
    raise ValueError(
        "Patient overlap between fitting and validation"
    )

y_fit = fit.icu_transfer.astype(int).to_numpy()
y_val = val.icu_transfer.astype(int).to_numpy()
y_dev = train.icu_transfer.astype(int).to_numpy()
y_test = test.icu_transfer.astype(int).to_numpy()

for labels in [y_fit, y_val, y_dev, y_test]:
    if len(np.unique(labels)) != 2:
        raise ValueError(
            "Both outcome classes must be present"
        )

print("Fitting rows:", len(fit))
print("Validation rows:", len(val))


# ============================================================
# 4. CORRECTED PREPROCESSING FUNCTION
# ============================================================

def fit_transform_pair(train_df, other_df):

    # --------------------------------------------------------
    # Numeric imputation
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
        add_indicator=True,
    )

    A_num = imputer.fit_transform(
        train_df[NUM_COLS]
    )

    B_num = imputer.transform(
        other_df[NUM_COLS]
    )

    # --------------------------------------------------------
    # Numeric scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    A_num = scaler.fit_transform(A_num)
    B_num = scaler.transform(B_num)

    # --------------------------------------------------------
    # Load standardized categorical indicators
    # .copy() fixes the read-only NumPy array error.
    # --------------------------------------------------------

    A_cat = (
        train_df[CAT_COLS]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=np.float32)
        .copy()
    )

    B_cat = (
        other_df[CAT_COLS]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=np.float32)
        .copy()
    )

    # --------------------------------------------------------
    # Convert standardized binary values back to 0 and 1
    # Determine levels using training data only.
    # --------------------------------------------------------

    categorical_levels = {}

    for j, col in enumerate(CAT_COLS):

        observed = np.unique(
            A_cat[:, j]
        )

        observed = observed[
            np.isfinite(observed)
        ]

        if len(observed) != 2:
            raise ValueError(
                f"{col}: Expected exactly two standardized "
                f"binary levels in training data. "
                f"Found: {observed}"
            )

        low = float(observed[0])
        high = float(observed[1])

        midpoint = (low + high) / 2

        if not np.isfinite(A_cat[:, j]).all():
            raise ValueError(
                f"Missing categorical values in {col}"
            )

        if not np.isfinite(B_cat[:, j]).all():
            raise ValueError(
                f"Missing categorical values in {col}"
            )

        # Check that validation/test values match
        # one of the two training levels.
        for values in [A_cat[:, j], B_cat[:, j]]:

            valid = (
                np.isclose(values, low, atol=1e-5)
                | np.isclose(values, high, atol=1e-5)
            )

            if not valid.all():
                raise ValueError(
                    f"Unexpected standardized values in {col}"
                )

        A_cat[:, j] = (
            A_cat[:, j] > midpoint
        ).astype(np.float32)

        B_cat[:, j] = (
            B_cat[:, j] > midpoint
        ).astype(np.float32)

        categorical_levels[col] = {
            "original_zero": low,
            "original_one": high,
        }

    # --------------------------------------------------------
    # Verify binary conversion
    # --------------------------------------------------------

    for arr in [A_cat, B_cat]:

        if not np.isin(arr, [0, 1]).all():
            raise ValueError(
                "Categorical conversion failed"
            )

    print(
        "Categorical indicators successfully "
        "converted to binary."
    )

    # --------------------------------------------------------
    # Combine numeric and categorical predictors
    # --------------------------------------------------------

    A = np.column_stack(
        [A_num, A_cat]
    ).astype(np.float32)

    B = np.column_stack(
        [B_num, B_cat]
    ).astype(np.float32)

    cat_positions = list(
        range(A_num.shape[1], A.shape[1])
    )

    preprocess = {
        "imputer": imputer,
        "scaler": scaler,
        "numeric_columns": NUM_COLS,
        "categorical_columns": CAT_COLS,
        "categorical_levels": categorical_levels,
    }

    return A, B, cat_positions, preprocess


# ============================================================
# 5. SMOTENC OVERSAMPLING
# ============================================================

def apply_smotenc(X, y, cat_positions):

    counts = np.bincount(y)

    current_ratio = counts[1] / counts[0]

    if current_ratio >= SMOTE_RATIO:
        raise ValueError(
            "Current minority ratio already exceeds target"
        )

    sampler = SMOTENC(
        categorical_features=cat_positions,
        sampling_strategy=SMOTE_RATIO,
        random_state=SEED,
        k_neighbors=5,
    )

    X_resampled, y_resampled = sampler.fit_resample(
        X, y
    )

    return X_resampled, y_resampled


# ============================================================
# 6. MODEL SETTINGS
# ============================================================

def build_model(method, original_labels):

    params = BASE_PARAMS.copy()

    if method == "weighted":

        counts = np.bincount(original_labels)

        params["scale_pos_weight"] = (
            counts[0] / counts[1]
        )

    else:
        params["scale_pos_weight"] = 1.0

    return XGBClassifier(**params)


# ============================================================
# 7. THRESHOLD OPTIMIZATION
# ============================================================

def choose_threshold(
    y_true,
    probabilities,
    target=0.70,
):

    order = np.argsort(
        -probabilities,
        kind="stable",
    )

    sorted_y = y_true[order]
    sorted_p = probabilities[order]

    ends = np.r_[
        np.flatnonzero(
            np.diff(sorted_p) != 0
        ),
        len(sorted_p) - 1,
    ]

    tp = np.cumsum(sorted_y)[ends]
    fp = ends + 1 - tp

    recall = tp / max(
        int(y_true.sum()), 1
    )

    precision = tp / np.maximum(
        tp + fp, 1
    )

    eligible = np.flatnonzero(
        recall >= target
    )

    if len(eligible) == 0:
        raise ValueError(
            "Cannot achieve target recall"
        )

    best_precision = precision[eligible].max()

    candidates = eligible[
        precision[eligible] == best_precision
    ]

    best = candidates[
        np.argmax(
            sorted_p[ends[candidates]]
        )
    ]

    return float(
        sorted_p[ends[best]]
    )


# ============================================================
# 8. EVALUATION FUNCTION
# ============================================================

def evaluate(
    method,
    split_name,
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = map(
        int,
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel(),
    )

    return {
        "model": method,
        "split": split_name,
        "threshold": threshold,
        "auroc": roc_auc_score(
            y_true, probabilities
        ),
        "auprc": average_precision_score(
            y_true, probabilities
        ),
        "recall": recall_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "specificity": tn / (tn + fp),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "accuracy": accuracy_score(
            y_true, predictions
        ),
        "brier": brier_score_loss(
            y_true, probabilities
        ),
        "log_loss": log_loss(
            y_true, probabilities
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "alerts": int(predictions.sum()),
        "alert_rate": float(predictions.mean()),
    }


# ============================================================
# 9. VALIDATION TRAINING
# ============================================================

print("\nSTEP 3: Training validation models...")

X_fit, X_val, cat_pos, _ = fit_transform_pair(
    fit, val
)

METHODS = [
    "weighted",
    "unweighted",
    "smotenc",
]

thresholds = {}
validation_results = []

for method in METHODS:

    if method == "smotenc":

        X_train, y_train = apply_smotenc(
            X_fit,
            y_fit,
            cat_pos,
        )

    else:
        X_train = X_fit
        y_train = y_fit

    model = build_model(
        method,
        y_fit,
    )

    model.fit(
        X_train,
        y_train,
    )

    prob = model.predict_proba(
        X_val
    )[:, 1]

    threshold = choose_threshold(
        y_val,
        prob,
        TARGET_RECALL,
    )

    thresholds[method] = threshold

    row = evaluate(
        method,
        "validation",
        y_val,
        prob,
        threshold,
    )

    validation_results.append(row)

    print(
        f"{method}: "
        f"threshold={threshold:.6f} | "
        f"AP={row['auprc']:.4f} | "
        f"recall={row['recall']:.3f} | "
        f"precision={row['precision']:.3f} | "
        f"FP={row['fp']:,}"
    )

pd.DataFrame(
    validation_results
).to_csv(
    OUT / "validation_metrics.csv",
    index=False,
)


# ============================================================
# 10. FINAL TRAINING ON FULL DEVELOPMENT DATA
# ============================================================

print("\nSTEP 4: Training full-development models...")

X_dev, X_test, cat_pos, preprocess = (
    fit_transform_pair(train, test)
)

joblib.dump(
    preprocess,
    OUT / "preprocessing.joblib",
)

test_results = []
curves = {}

predictions_df = pd.DataFrame({
    "subject_id": test.subject_id.to_numpy(),
    "actual": y_test,
})

for method in METHODS:

    if method == "smotenc":

        X_train, y_train = apply_smotenc(
            X_dev,
            y_dev,
            cat_pos,
        )

    else:
        X_train = X_dev
        y_train = y_dev

    print(
        f"{method} training counts:",
        np.bincount(y_train),
    )

    model = build_model(
        method,
        y_dev,
    )

    model.fit(
        X_train,
        y_train,
    )

    model.save_model(
        str(
            OUT / f"{method}_xgboost.json"
        )
    )

    prob = model.predict_proba(
        X_test
    )[:, 1]

    threshold = thresholds[method]

    row = evaluate(
        method,
        "exploratory_test",
        y_test,
        prob,
        threshold,
    )

    test_results.append(row)

    pred = (
        prob >= threshold
    ).astype(int)

    predictions_df[
        f"{method}_probability"
    ] = prob

    predictions_df[
        f"{method}_prediction"
    ] = pred

    curves[method] = prob

    print(
        f"{method}: "
        f"AUROC={row['auroc']:.4f} | "
        f"AP={row['auprc']:.4f} | "
        f"Recall={row['recall']:.3f} | "
        f"Precision={row['precision']:.3f} | "
        f"FP={row['fp']:,}"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        pred,
        labels=[0, 1],
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        cm,
        cmap="Blues",
    )

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=13,
                color=(
                    "white"
                    if cm[i, j] > cm.max() / 2
                    else "black"
                ),
            )

    ax.set_xticks(
        [0, 1],
        ["No ICU", "ICU"],
    )

    ax.set_yticks(
        [0, 1],
        ["No ICU", "ICU"],
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_title(
        f"{method.upper()} XGBoost - 20 Features"
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        OUT / f"{method}_confusion_matrix.png",
        dpi=300,
    )

    plt.close(fig)

    print(
        "\nClassification report:",
        method,
    )

    print(
        classification_report(
            y_test,
            pred,
            zero_division=0,
        )
    )


# ============================================================
# 11. SAVE RESULTS
# ============================================================

comparison = pd.DataFrame(
    test_results
)

comparison.to_csv(
    OUT / "test_metrics_comparison.csv",
    index=False,
)

predictions_df.to_csv(
    OUT / "test_predictions_comparison.csv",
    index=False,
)


# ============================================================
# 12. ROC COMPARISON
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

for method, prob in curves.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        prob,
    )

    auc = roc_auc_score(
        y_test,
        prob,
    )

    ax.plot(
        fpr,
        tpr,
        label=f"{method}: AUROC={auc:.4f}",
    )

ax.plot(
    [0, 1],
    [0, 1],
    "--",
    label="Chance",
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("20-Feature XGBoost ROC Comparison")
ax.legend()

fig.tight_layout()

fig.savefig(
    OUT / "roc_comparison.png",
    dpi=300,
)

plt.close(fig)


# ============================================================
# 13. PRECISION-RECALL COMPARISON
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

for method, prob in curves.items():

    precision, recall, _ = (
        precision_recall_curve(
            y_test,
            prob,
        )
    )

    ap = average_precision_score(
        y_test,
        prob,
    )

    ax.plot(
        recall,
        precision,
        label=f"{method}: AP={ap:.4f}",
    )

ax.axhline(
    y_test.mean(),
    linestyle="--",
    label="Positive prevalence",
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("20-Feature XGBoost PR Comparison")
ax.legend()

fig.tight_layout()

fig.savefig(
    OUT / "pr_comparison.png",
    dpi=300,
)

plt.close(fig)


# ============================================================
# 14. CALIBRATION COMPARISON
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.plot(
    [0, 1],
    [0, 1],
    "--",
    label="Perfect calibration",
)

for method, prob in curves.items():

    table = pd.DataFrame({
        "probability": prob,
        "actual": y_test,
    })

    table["bin"] = pd.qcut(
        table["probability"],
        q=10,
        labels=False,
        duplicates="drop",
    )

    grouped = (
        table.groupby("bin")
        .agg(
            predicted=("probability", "mean"),
            observed=("actual", "mean"),
        )
    )

    ax.plot(
        grouped["predicted"],
        grouped["observed"],
        marker="o",
        label=method,
    )

ax.set_xlim(0, 0.4)
ax.set_ylim(0, 0.4)

ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Observed ICU Transfer Frequency")
ax.set_title("20-Feature XGBoost Calibration")
ax.legend()

fig.tight_layout()

fig.savefig(
    OUT / "calibration_comparison.png",
    dpi=300,
)

plt.close(fig)


# ============================================================
# 15. METRICS COMPARISON CHART
# ============================================================

plot_metrics = [
    "auroc",
    "auprc",
    "recall",
    "precision",
    "specificity",
    "f1",
]

fig, ax = plt.subplots(
    figsize=(10, 5)
)

comparison.set_index(
    "model"
)[plot_metrics].T.plot(
    kind="bar",
    ax=ax,
)

ax.set_ylabel("Score")
ax.set_title("20-Feature XGBoost: Imbalance Methods")
ax.tick_params(
    axis="x",
    rotation=35,
)

ax.legend(title="Model")

fig.tight_layout()

fig.savefig(
    OUT / "metrics_comparison.png",
    dpi=300,
)

plt.close(fig)


# ============================================================
# 16. SAVE EXPERIMENT SETTINGS
# ============================================================

settings = {
    "features": FEATURES,
    "smote_ratio": SMOTE_RATIO,
    "target_recall": TARGET_RECALL,
    "hyperparameters": BASE_PARAMS,
    "methods": METHODS,
    "thresholds": thresholds,
    "train_rows": len(train),
    "test_rows": len(test),
    "original_models_unchanged": True,
    "test_evaluation": "exploratory",
    "limitations": [
        "Input CSV files already contain standardized predictors",
        "Numeric predictors are scaled again for SMOTENC",
        "Categorical dummy variables restored to binary",
        "SMOTENC treats one-hot indicators independently",
        "Synthetic categorical combinations may be invalid",
        "Preprocessing differs from original 20-feature model",
        "Test set previously examined",
    ],
}

with open(
    OUT / "experiment_settings.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        settings,
        f,
        indent=2,
    )


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TEST METRICS COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)

print("\nSTEP 68 COMPLETE")
print("Results saved to:", OUT)
print("Original XGBoost models unchanged.")
