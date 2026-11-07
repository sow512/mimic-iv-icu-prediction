
# ============================================================
# STEP 60: XGBOOST PROBABILITY CALIBRATION
# Timing-corrected 75-feature model
# Patient-grouped CV + Platt + Isotonic
# ============================================================

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss, log_loss,
    roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier, DMatrix

# ============================================================
# 1. PATHS
# ============================================================

PROJECT = Path(
    r"C:\Users\sowmy\Downloads\MIMIC-IV\MIMIC_IV\MIMIC_IV"
)

SOURCE = (
    PROJECT / "model training figures"
    / "expanded_lab_xgboost"
)

DATA_FILE = (
    SOURCE / "step51_timing_corrected"
    / "expanded_candidate_features_storetime_corrected.parquet"
)

MODEL_DIR = SOURCE / "step53_final_75_model"
OUT = SOURCE / "step60_probability_calibration"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FEATURES = 75
N_FOLDS = 3

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

CAT_COLS = [
    "gender",
    "admission_type",
    "admission_location"
]

# ============================================================
# 2. LOAD CORRECTED DATA
# ============================================================

print("\nSTEP 60: Loading timing-corrected data...")

features = pd.read_parquet(DATA_FILE)

cohort = pd.read_csv(
    SOURCE / "cohort_audit.csv",
    usecols=["hadm_id", "subject_id", "icu_transfer"]
)

if features.hadm_id.duplicated().any():
    raise ValueError("Duplicate feature admissions")

if cohort.hadm_id.duplicated().any():
    raise ValueError("Duplicate cohort admissions")

if set(features.hadm_id) != set(cohort.hadm_id):
    raise ValueError("Cohort mismatch")

# ============================================================
# 3. PRESERVE ORIGINAL TEST SPLIT
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

y = dev.icu_transfer.astype(int).to_numpy()
groups = dev.subject_id.to_numpy()

print("Development admissions:", len(dev))
print("Reserved test admissions:", len(test_cohort))
print("Development positives:", int(y.sum()))

# ============================================================
# 4. PREPROCESSING WITHIN EACH FOLD
# ============================================================

NUM_COLS = ["age"] + [
    c for c in dev.columns
    if c.startswith("lab_")
]

def prepare_fold(train, validation):

    kept = [
        c for c in NUM_COLS
        if train[c].notna().mean() >= 0.10
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
                CAT_COLS
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    train_data = train[kept + CAT_COLS].copy()
    val_data = validation[kept + CAT_COLS].copy()

    for data in (train_data, val_data):
        for col in CAT_COLS:
            data[col] = (
                data[col].fillna("Unknown").astype(str)
            )

    X_train = np.asarray(
        pre.fit_transform(train_data),
        dtype=np.float32
    )

    X_val = np.asarray(
        pre.transform(val_data),
        dtype=np.float32
    )

    return X_train, X_val, list(
        pre.get_feature_names_out()
    )

# ============================================================
# 5. SHAP RANKING INSIDE EACH TRAINING FOLD
# ============================================================

def select_features(X, labels, names, seed):

    ranker = XGBClassifier(**PARAMS)
    ranker.fit(X, labels)

    rng = np.random.default_rng(seed)

    sample_idx = rng.choice(
        len(X),
        size=min(2000, len(X)),
        replace=False
    )

    shap_values = ranker.get_booster().predict(
        DMatrix(
            X[sample_idx],
            feature_names=names
        ),
        pred_contribs=True
    )

    importance = np.abs(
        shap_values[:, :-1]
    ).mean(axis=0)

    return np.argsort(
        -importance,
        kind="stable"
    )[:N_FEATURES]

# ============================================================
# 6. GENERATE OUT-OF-FOLD PREDICTIONS
# ============================================================

print("\nGenerating patient-grouped OOF predictions...")

cv = StratifiedGroupKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)

oof_prob = np.full(len(dev), np.nan)
oof_fold = np.full(len(dev), -1, dtype=int)

for fold, (fit_idx, val_idx) in enumerate(
    cv.split(dev, y, groups),
    start=1
):

    fit = dev.iloc[fit_idx].copy()
    val = dev.iloc[val_idx].copy()

    if set(fit.subject_id) & set(val.subject_id):
        raise ValueError("Patient overlap in CV fold")

    X_fit, X_val, names = prepare_fold(fit, val)

    selected = select_features(
        X_fit,
        y[fit_idx],
        names,
        SEED + fold
    )

    model = XGBClassifier(**PARAMS)

    model.fit(
        X_fit[:, selected],
        y[fit_idx]
    )

    probabilities = model.predict_proba(
        X_val[:, selected]
    )[:, 1]

    oof_prob[val_idx] = probabilities
    oof_fold[val_idx] = fold

    print(
        f"Fold {fold}/{N_FOLDS} | "
        f"Validation={len(val):,} | "
        f"AUROC={roc_auc_score(y[val_idx], probabilities):.4f} | "
        f"AP={average_precision_score(y[val_idx], probabilities):.4f}",
        flush=True
    )

if np.isnan(oof_prob).any():
    raise ValueError("Missing OOF predictions")

# ============================================================
# 7. CALIBRATION FUNCTIONS
# ============================================================

EPS = 1e-7

def clip_prob(p):
    return np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1 - EPS
    )

def logit(p):
    p = clip_prob(p)
    return np.log(p / (1 - p))

def fit_platt(p, labels):

    calibrator = LogisticRegression(
        max_iter=1000,
        C=1e6,
        random_state=SEED
    )

    calibrator.fit(
        logit(p).reshape(-1, 1),
        labels
    )

    return calibrator

def predict_platt(calibrator, p):

    return calibrator.predict_proba(
        logit(p).reshape(-1, 1)
    )[:, 1]

def fit_isotonic(p, labels):

    calibrator = IsotonicRegression(
        out_of_bounds="clip",
        y_min=0,
        y_max=1
    )

    calibrator.fit(p, labels)
    return calibrator

# ============================================================
# 8. CROSS-FIT CALIBRATION
# ============================================================

# Each calibration evaluation fold is calibrated
# using predictions from OTHER patient groups.
# This avoids evaluating calibration on the same
# labels used to fit that calibrator.

print("\nCross-fitting probability calibration...")

cal_cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=2026
)

platt_oof = np.full(len(dev), np.nan)
isotonic_oof = np.full(len(dev), np.nan)

for fit_idx, val_idx in cal_cv.split(
    dev, y, groups
):

    platt = fit_platt(
        oof_prob[fit_idx],
        y[fit_idx]
    )

    iso = fit_isotonic(
        oof_prob[fit_idx],
        y[fit_idx]
    )

    platt_oof[val_idx] = predict_platt(
        platt,
        oof_prob[val_idx]
    )

    isotonic_oof[val_idx] = iso.predict(
        oof_prob[val_idx]
    )

if (
    np.isnan(platt_oof).any()
    or np.isnan(isotonic_oof).any()
):
    raise ValueError("Missing calibrated predictions")

# ============================================================
# 9. METRICS
# ============================================================

def calibration_error(labels, p, bins=10):

    edges = np.quantile(
        p,
        np.linspace(0, 1, bins + 1)
    )

    edges = np.unique(edges)

    if len(edges) < 2:
        return np.nan

    assignment = np.searchsorted(
        edges[1:-1], p, side="right"
    )

    error = 0.0

    for k in np.unique(assignment):
        mask = assignment == k

        error += (
            mask.mean()
            * abs(
                labels[mask].mean()
                - p[mask].mean()
            )
        )

    return float(error)

def evaluate(name, p):

    p = clip_prob(p)

    return {
        "method": name,
        "brier_score": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "ece_10_quantile_bins": calibration_error(y, p),
        "mean_predicted_risk": float(p.mean()),
        "observed_prevalence": float(y.mean())
    }

results = pd.DataFrame([
    evaluate("Uncalibrated", oof_prob),
    evaluate("Platt scaling", platt_oof),
    evaluate("Isotonic regression", isotonic_oof)
])

print("\nCALIBRATION COMPARISON")
print(results.to_string(index=False))

results.to_csv(
    OUT / "calibration_comparison.csv",
    index=False
)

# ============================================================
# 10. CALIBRATION CURVES AND RISK-BIN TABLES
# ============================================================

methods = {
    "Uncalibrated": oof_prob,
    "Platt scaling": platt_oof,
    "Isotonic regression": isotonic_oof
}

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(
    [0, 1], [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

bin_tables = []

for name, probabilities in methods.items():

    # Quantile bins handle the low risk prevalence
    # better than uniformly spaced bins.
    bins = pd.qcut(
        probabilities,
        q=10,
        labels=False,
        duplicates="drop"
    )

    temp = pd.DataFrame({
        "bin": bins,
        "actual": y,
        "probability": probabilities
    })

    summary = temp.groupby("bin").agg(
        admissions=("actual", "size"),
        observed_risk=("actual", "mean"),
        predicted_risk=("probability", "mean")
    ).reset_index()

    summary["method"] = name
    bin_tables.append(summary)

    ax.plot(
        summary["predicted_risk"],
        summary["observed_risk"],
        marker="o",
        label=name
    )

ax.set_xlabel("Mean predicted ICU-transfer probability")
ax.set_ylabel("Observed ICU-transfer frequency")
ax.set_title("Patient-grouped OOF calibration")
ax.legend()
ax.grid(alpha=0.25)

# Focus on clinically relevant low probabilities.
ax.set_xlim(0, 0.30)
ax.set_ylim(0, 0.30)

fig.tight_layout()
fig.savefig(
    OUT / "calibration_curve.png",
    dpi=300
)
plt.close(fig)

pd.concat(
    bin_tables,
    ignore_index=True
).to_csv(
    OUT / "calibration_risk_bins.csv",
    index=False
)

# ============================================================
# 11. SAVE OOF PREDICTIONS
# ============================================================

pd.DataFrame({
    "hadm_id": dev.hadm_id.to_numpy(),
    "subject_id": dev.subject_id.to_numpy(),
    "actual": y,
    "fold": oof_fold,
    "uncalibrated_probability": oof_prob,
    "platt_probability": platt_oof,
    "isotonic_probability": isotonic_oof
}).to_csv(
    OUT / "development_oof_calibration_predictions.csv",
    index=False
)

# ============================================================
# 12. SAVE CALIBRATORS FOR FROZEN FINAL MODEL
# ============================================================

# Fit final calibrators using ALL development OOF
# predictions. These are intended to calibrate the
# frozen Step 53 model, which was refitted on all
# development patients.
#
# Calibration transfer to that refitted model
# must still be evaluated independently.

final_platt = fit_platt(oof_prob, y)
final_isotonic = fit_isotonic(oof_prob, y)

joblib.dump(
    final_platt,
    OUT / "platt_calibrator.joblib"
)

joblib.dump(
    final_isotonic,
    OUT / "isotonic_calibrator.joblib"
)

# Choose method by cross-fitted Brier score.
best = results.sort_values(
    "brier_score"
).iloc[0]

summary = {
    "best_method_by_crossfit_brier": best["method"],
    "best_crossfit_brier": float(best["brier_score"]),
    "feature_count": N_FEATURES,
    "cv_folds": N_FOLDS,
    "test_evaluated": False,
    "note": (
        "Calibration comparison is development-only. "
        "Feature selection is repeated within each model fold. "
        "Calibration is cross-fitted on OOF predictions. "
        "The saved calibrators were fitted on all development "
        "OOF predictions and have not been independently "
        "evaluated on the frozen full-development model. "
        "Outcome-label review remains pending."
    )
}

with open(
    OUT / "calibration_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("STEP 60 COMPLETE")
print("=" * 60)

print("Best calibration method:", best["method"])
print("Best cross-fitted Brier score:", best["brier_score"])
print("Results saved to:", OUT)
print("Original XGBoost model unchanged.")
print("Reserved test set not evaluated.")
