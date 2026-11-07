
# ============================================================
# BN5212 FINAL PROJECT
# 75-FEATURE XGBOOST - FURTHER MODEL ANALYSIS
#
# 1. Threshold sensitivity analysis
# 2. Probability calibration analysis
# 3. ICU bed ranking simulation
#
# Uses the EXISTING trained tuned XGBoost model.
# No hyperparameter tuning or model retraining.
# ============================================================

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss
)

from sklearn.calibration import calibration_curve


# ============================================================
# STEP 1: SETTINGS
# ============================================================

PROJECT = Path(__file__).resolve().parent

EXCEL_FILE = PROJECT / "75_Feature_Dataset.xlsx"

MODEL_FOLDER = PROJECT / "results_xgb_75_tuning"

OUTPUT = PROJECT / "results_xgb_75_further_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42

N_CANDIDATES = 100
N_BEDS = 10
N_SIMULATIONS = 1000

rng = np.random.default_rng(SEED)

print("=" * 55)
print("75-FEATURE XGBOOST - FURTHER ANALYSIS")
print("=" * 55)

print("\nOutput folder:", OUTPUT)


# ============================================================
# STEP 2: LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained XGBoost model...", flush=True)

bundle_path = MODEL_FOLDER / "final_model_bundle.joblib"

if not bundle_path.exists():
    raise FileNotFoundError(
        f"Model bundle not found: {bundle_path}"
    )

bundle = joblib.load(bundle_path)

model = bundle["model"]
imputer = bundle["imputer"]
features = bundle["feature_columns"]

selected_threshold = float(
    bundle["validation_threshold"]
)

print("Trained model loaded successfully!")
print("Number of features:", len(features))
print("Original selected threshold:", selected_threshold)


# ============================================================
# STEP 3: LOAD VALIDATION PREDICTIONS
# ============================================================

print("\nLoading validation predictions...", flush=True)

val_path = MODEL_FOLDER / "validation_predictions.csv"

if not val_path.exists():
    raise FileNotFoundError(
        f"Validation predictions not found: {val_path}"
    )

validation = pd.read_csv(val_path)

required_val = ["actual", "tuned_probability"]

if not set(required_val).issubset(validation.columns):
    raise ValueError(
        "Validation predictions have missing columns."
    )

y_val = validation["actual"].astype(int).to_numpy()

val_prob = validation[
    "tuned_probability"
].to_numpy(dtype=float)

print("Validation records:", len(validation))


# ============================================================
# STEP 4: LOAD RESERVED TEST DATA
# ============================================================

print("\nLoading reserved test dataset...", flush=True)

data = pd.read_excel(
    EXCEL_FILE,
    sheet_name="Final_75_Features",
    engine="openpyxl"
)

data["dataset_split"] = (
    data["dataset_split"]
    .astype(str)
    .str.strip()
    .str.lower()
)

test = data[
    data["dataset_split"] == "test"
].copy().reset_index(drop=True)

if len(test) == 0:
    raise ValueError("Test dataset is empty.")

if test["hadm_id"].duplicated().any():
    raise ValueError("Duplicate test admission IDs found.")

X_test = test[features].apply(
    pd.to_numeric,
    errors="raise"
)

y_test = test[
    "icu_transfer"
].astype(int).to_numpy()

X_test_processed = imputer.transform(
    X_test
).astype(np.float32)

test_prob = model.predict_proba(
    X_test_processed
)[:, 1]

print("Test admissions:", len(test))
print("Actual ICU transfers:", int(y_test.sum()))

# Check consistency with previously saved predictions.

previous_path = MODEL_FOLDER / "final_test_predictions.csv"

if previous_path.exists():

    previous = pd.read_csv(previous_path)

    required = {
        "hadm_id",
        "tuned_probability"
    }

    if required.issubset(previous.columns):

        current = pd.DataFrame({
            "hadm_id": test["hadm_id"].to_numpy(),
            "new_probability": test_prob
        })

        check = current.merge(
            previous[[
                "hadm_id",
                "tuned_probability"
            ]],
            on="hadm_id",
            validate="one_to_one"
        )

        if len(check) != len(test):
            raise ValueError(
                "Saved predictions do not match all test admissions."
            )

        max_difference = np.max(
            np.abs(
                check["new_probability"].to_numpy()
                - check["tuned_probability"].to_numpy()
            )
        )

        print("Maximum probability difference:",
              max_difference)

        if max_difference > 1e-5:
            raise ValueError(
                "Model predictions differ from saved test results."
            )


# ============================================================
# STEP 5: THRESHOLD ANALYSIS FUNCTION
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if tp + fn else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp else 0.0
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    return {
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "Balanced Accuracy": balanced_accuracy_score(
            y_true,
            predictions
        ),
        "Predicted Positives": int(predictions.sum()),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


# ============================================================
# STEP 6: VALIDATION THRESHOLD SENSITIVITY ANALYSIS
# ============================================================

print("\n" + "=" * 55)
print("EXPERIMENT 1: THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 55)

thresholds = np.unique(
    np.concatenate([
        np.linspace(0.005, 0.20, 196),
        [selected_threshold]
    ])
)

threshold_results = []

for threshold in thresholds:

    threshold_results.append(
        calculate_metrics(
            y_val,
            val_prob,
            float(threshold)
        )
    )

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    OUTPUT / "validation_threshold_comparison.csv",
    index=False
)

print("\nSelected threshold performance on validation:")

selected_metrics = calculate_metrics(
    y_val,
    val_prob,
    selected_threshold
)

print(pd.Series(selected_metrics).to_string())


# ============================================================
# VALIDATION PREDICTION CONSISTENCY CHECK
# ============================================================

print("\nVALIDATION CONSISTENCY CHECK")

full_precision_pred = (
    val_prob >= selected_threshold
).astype(int)

rounded_pred = (
    np.round(val_prob, 6) >= selected_threshold
).astype(int)

changed = np.where(
    full_precision_pred != rounded_pred
)[0]

print("Predictions changed by rounding:", len(changed))

print("\nConfusion matrix - saved probabilities:")
print(confusion_matrix(
    y_val, full_precision_pred, labels=[0, 1]
))

print("\nConfusion matrix - rounded probabilities:")
print(confusion_matrix(
    y_val, rounded_pred, labels=[0, 1]
))

if len(changed) > 0:
    comparison = pd.DataFrame({
        "row_index": changed,
        "actual": y_val[changed],
        "saved_probability": val_prob[changed],
        "rounded_probability": np.round(
            val_prob[changed], 6
        ),
        "selected_threshold": selected_threshold
    })

    comparison.to_csv(
        OUTPUT / "threshold_rounding_check.csv",
        index=False
    )

    print("\nRows affected by rounding:")
    print(comparison.to_string(index=False))

# ============================================================
# STEP 7: COMPARE RECALL TARGETS
# ============================================================

recall_targets = [0.60, 0.70, 0.80, 0.90]

target_results = []

for target in recall_targets:

    eligible = threshold_df[
        threshold_df["Recall"] >= target
    ]

    if eligible.empty:
        continue

    # Choose highest precision among thresholds
    # satisfying the recall requirement.

    chosen = eligible.sort_values(
        ["Precision", "Threshold"],
        ascending=[False, False]
    ).iloc[0].copy()

    chosen["Recall Target"] = target

    target_results.append(chosen)

target_df = pd.DataFrame(target_results)

target_df.to_csv(
    OUTPUT / "recall_target_comparison.csv",
    index=False
)

print("\nRECALL TARGET COMPARISON:")
print(target_df.round(4).to_string(index=False))

# These thresholds are exploratory alternatives.
# Do not select a final threshold using test labels.


# ============================================================
# STEP 8: THRESHOLD GRAPHS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    label="F1-score"
)

plt.axvline(
    selected_threshold,
    linestyle="--",
    color="black",
    label="Original selected threshold"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("75-Feature XGBoost - Validation Threshold Analysis")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT / "threshold_analysis.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 9: PROBABILITY CALIBRATION ANALYSIS
# ============================================================

print("\n" + "=" * 55)
print("EXPERIMENT 2: PROBABILITY CALIBRATION")
print("=" * 55)

brier = brier_score_loss(
    y_test,
    test_prob
)

auc = roc_auc_score(
    y_test,
    test_prob
)

ap = average_precision_score(
    y_test,
    test_prob
)

loss = log_loss(
    y_test,
    test_prob,
    labels=[0, 1]
)

prevalence = float(np.mean(y_test))

baseline_prob = np.full(
    len(y_test),
    prevalence
)

baseline_brier = brier_score_loss(
    y_test,
    baseline_prob
)

brier_skill = (
    1 - brier / baseline_brier
    if baseline_brier > 0 else np.nan
)

print("\nTEST PROBABILITY METRICS:")
print("AUROC:", round(auc, 5))
print("AUPRC:", round(ap, 5))
print("Brier Score:", round(brier, 5))
print("Log Loss:", round(loss, 5))
print("ICU prevalence:", round(prevalence, 5))
print("Baseline Brier:", round(baseline_brier, 5))
print("Brier Skill Score:", round(brier_skill, 5))

prob_true, prob_pred = calibration_curve(
    y_test,
    test_prob,
    n_bins=10,
    strategy="quantile"
)

calibration_df = pd.DataFrame({
    "Mean Predicted Probability": prob_pred,
    "Observed ICU Frequency": prob_true
})

calibration_df.to_csv(
    OUTPUT / "calibration_results.csv",
    index=False
)


# ============================================================
# CALIBRATION BIN COUNTS
# ============================================================

calibration_bins = pd.DataFrame({
    "actual": y_test,
    "predicted_probability": test_prob
})

calibration_bins["bin"] = pd.qcut(
    calibration_bins["predicted_probability"],
    q=10,
    labels=False,
    duplicates="drop"
)

calibration_bin_summary = (
    calibration_bins
    .groupby("bin")
    .agg(
        admissions=("actual", "size"),
        actual_icu_transfers=("actual", "sum"),
        mean_predicted_probability=(
            "predicted_probability", "mean"
        ),
        observed_icu_frequency=("actual", "mean")
    )
    .reset_index()
)

calibration_bin_summary.to_csv(
    OUTPUT / "calibration_bin_summary.csv",
    index=False
)

print("\nCALIBRATION BIN SUMMARY:")
print(calibration_bin_summary.to_string(index=False))

plt.figure(figsize=(8, 7))

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
    label="Perfect calibration"
)

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label="Tuned XGBoost"
)

plt.xlabel("Mean Predicted ICU Probability")
plt.ylabel("Observed ICU Transfer Frequency")
plt.title("75-Feature XGBoost - Calibration Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT / "calibration_curve.png",
    dpi=300
)

plt.close()

# Focused view for rare ICU events.

plt.figure(figsize=(8, 7))

plt.plot(
    [0, 0.3],
    [0, 0.3],
    linestyle="--",
    color="black",
    label="Perfect calibration"
)

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label="Tuned XGBoost"
)

plt.xlim(0, 0.3)
plt.ylim(0, 0.3)

plt.xlabel("Mean Predicted ICU Probability")
plt.ylabel("Observed ICU Transfer Frequency")
plt.title("Calibration Curve - Zoomed View")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT / "calibration_curve_zoomed.png",
    dpi=300
)

plt.close()



# ============================================================
# STEP 10: IMPROVED ICU BED RANKING SIMULATION
# ============================================================

print("\n" + "=" * 55)
print("EXPERIMENT 3: ICU BED RANKING SIMULATION")
print("=" * 55)

print("Candidate admissions:", N_CANDIDATES)
print("Available ICU beds:", N_BEDS)
print("Number of simulations:", N_SIMULATIONS)

simulation_results = []
first_ranked = None

for simulation in range(N_SIMULATIONS):

    indices = rng.choice(
        len(test),
        size=N_CANDIDATES,
        replace=False
    )

    group = test.iloc[indices].copy()
    group["risk_score"] = test_prob[indices]
    group["actual_icu_transfer"] = y_test[indices]

    group = group.sort_values(
        "risk_score",
        ascending=False
    ).reset_index(drop=True)

    group["priority_rank"] = np.arange(
        1, len(group) + 1
    )

    group["top10_recommended"] = (
        group["priority_rank"] <= N_BEDS
    )

    top = group.head(N_BEDS)

    actual_total = int(
        group["actual_icu_transfer"].sum()
    )

    captured = int(
        top["actual_icu_transfer"].sum()
    )

    missed = actual_total - captured

    precision_at_10 = captured / N_BEDS

    recall_at_10 = (
        captured / actual_total
        if actual_total > 0 else np.nan
    )

    # Random baseline using the SAME candidate group
    random_indices = rng.choice(
        len(group),
        size=N_BEDS,
        replace=False
    )

    random_captured = int(
        group.iloc[random_indices][
            "actual_icu_transfer"
        ].sum()
    )

    random_expected = (
        N_BEDS * actual_total / N_CANDIDATES
    )

    simulation_results.append({
        "Simulation": simulation + 1,
        "Candidate Admissions": N_CANDIDATES,
        "Available Beds": N_BEDS,
        "Actual ICU Transfers": actual_total,
        "Captured in Top 10": captured,
        "Missed Outside Top 10": missed,
        "Precision@10": precision_at_10,
        "Recall@10": recall_at_10,
        "Random Captured": random_captured,
        "Random Expected Captures": random_expected,
        "Improvement Over Random": (
            captured - random_captured
        )
    })

    if simulation == 0:
        first_ranked = group.copy()

# ============================================================
# STEP 11: SAVE SIMULATION RESULTS
# ============================================================

simulation_df = pd.DataFrame(simulation_results)

simulation_df.to_csv(
    OUTPUT / "icu_simulation_1000_runs.csv",
    index=False
)

first_ranked.to_csv(
    OUTPUT / "first_simulation_all_100.csv",
    index=False
)

top10 = first_ranked.head(N_BEDS).copy()

display_columns = [
    "priority_rank",
    "subject_id",
    "hadm_id",
    "risk_score",
    "actual_icu_transfer",
    "top10_recommended"
]

top10[display_columns].to_csv(
    OUTPUT / "first_simulation_top10.csv",
    index=False
)

print("\nFIRST SIMULATION - TOP 10:")
print(
    top10[display_columns].to_string(index=False)
)


# ============================================================
# STEP 12: SUMMARIZE 1000 SIMULATIONS
# ============================================================


summary = {
    "Number of Simulations": N_SIMULATIONS,
    "Candidate Admissions": N_CANDIDATES,
    "Available Beds": N_BEDS,

    "Mean Actual ICU Transfers":
        simulation_df["Actual ICU Transfers"].mean(),

    "Mean Captured in Top 10":
        simulation_df["Captured in Top 10"].mean(),

    "Mean Missed Outside Top 10":
        simulation_df["Missed Outside Top 10"].mean(),

    "Mean Precision@10":
        simulation_df["Precision@10"].mean(),

    "Mean Recall@10 (nonempty groups)":
        simulation_df["Recall@10"].mean(),

    "Pooled Recall@10":
        simulation_df["Captured in Top 10"].sum()
        / simulation_df["Actual ICU Transfers"].sum(),

    "Mean Random Captured":
        simulation_df["Random Captured"].mean(),

    "Mean Random Expected Captures":
        simulation_df["Random Expected Captures"].mean(),

    "Mean Improvement Over Random":
        simulation_df["Improvement Over Random"].mean(),

    "Capture Rate Improvement Ratio":
        simulation_df["Captured in Top 10"].mean()
        / simulation_df["Random Expected Captures"].mean(),

    "Capture Count 2.5th Percentile":
        simulation_df["Captured in Top 10"].quantile(0.025),

    "Capture Count 97.5th Percentile":
        simulation_df["Captured in Top 10"].quantile(0.975),

    "Simulations With No Actual Transfers":
        int((simulation_df["Actual ICU Transfers"] == 0).sum())
}


summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["Metric", "Value"]
)

summary_df.to_csv(
    OUTPUT / "icu_simulation_summary.csv",
    index=False
)

print("\nSIMULATION SUMMARY:")
print(summary_df.to_string(index=False))


# ============================================================
# STEP 13: VISUALIZE SIMULATION PERFORMANCE
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    simulation_df["Captured in Top 10"],
    bins=np.arange(-0.5, N_BEDS + 1.5, 1),
    edgecolor="black"
)

plt.xlabel("Actual ICU Transfers Captured in Top 10")
plt.ylabel("Number of Simulations")
plt.title("ICU Ranking Performance Across 1000 Simulations")
plt.xticks(range(N_BEDS + 1))
plt.tight_layout()

plt.savefig(
    OUTPUT / "icu_capture_distribution.png",
    dpi=300
)

plt.close()

# Compare model with expected random selection.

comparison_values = [
    summary["Mean Captured in Top 10"],
    summary["Mean Random Expected Captures"]
]

plt.figure(figsize=(8, 6))

plt.bar(
    ["XGBoost Ranking", "Random Selection"],
    comparison_values
)

plt.ylabel("Mean Actual ICU Transfers Captured")
plt.title("Top-10 Ranking vs Random Selection")
plt.tight_layout()

plt.savefig(
    OUTPUT / "icu_ranking_vs_random.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 14: SAVE FINAL SUMMARY
# ============================================================

report_path = OUTPUT / "further_analysis_summary.txt"

with open(report_path, "w", encoding="utf-8") as f:

    f.write("75-FEATURE XGBOOST - FURTHER ANALYSIS\n")
    f.write("=" * 50 + "\n\n")

    f.write("TEST PROBABILITY METRICS\n")
    f.write(f"AUROC: {auc:.5f}\n")
    f.write(f"AUPRC: {ap:.5f}\n")
    f.write(f"Brier Score: {brier:.5f}\n")
    f.write(f"Log Loss: {loss:.5f}\n")
    f.write(f"ICU Prevalence: {prevalence:.5f}\n")
    f.write(f"Baseline Brier: {baseline_brier:.5f}\n")
    f.write(f"Brier Skill Score: {brier_skill:.5f}\n\n")

    f.write("ORIGINAL VALIDATION THRESHOLD\n")
    f.write(f"{selected_threshold:.6f}\n\n")

    f.write("RECALL TARGET ANALYSIS\n")
    f.write(target_df.round(5).to_string(index=False))
    f.write("\n\n")

    f.write("ICU SIMULATION SUMMARY\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n\n")

    f.write(
        "NOTE: Threshold alternatives were examined "
        "using validation predictions. The test set "
        "was not used to choose new thresholds.\n"
    )

    f.write(
        "ICU ranking is a retrospective simulation "
        "and does not establish clinical bed-allocation "
        "safety or effectiveness.\n"
    )


# ============================================================
# ADDITIONAL STEP: SAVE STANDARDIZED TEST METRICS
# ============================================================

test_metrics = calculate_metrics(
    y_test,
    test_prob,
    selected_threshold
)

tn = test_metrics["TN"]
fp = test_metrics["FP"]
fn = test_metrics["FN"]
tp = test_metrics["TP"]

total = tn + fp + fn + tp

npv = tn / (tn + fn) if (tn + fn) else 0.0

standardized_metrics = {
    "Model": "75-Feature Tuned XGBoost",
    "Dataset": "Reserved Test",
    "AUROC": auc,
    "AUPRC": ap,
    "Sensitivity / Recall": test_metrics["Recall"],
    "Specificity": test_metrics["Specificity"],
    "Precision (PPV)": test_metrics["Precision"],
    "Accuracy": (tp + tn) / total,
    "F1-Score": test_metrics["F1"],
    "Negative Predictive Value (NPV)": npv,
    "Balanced Accuracy": test_metrics["Balanced Accuracy"],
    "False Positive Rate": fp / (fp + tn),
    "False Negative Rate": fn / (fn + tp),
    "Brier Score": brier,
    "Log Loss": loss,
    "Brier Skill Score": brier_skill,
    "Classification Threshold": selected_threshold,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp
}

metrics_df = pd.DataFrame(
    list(standardized_metrics.items()),
    columns=["Performance Metric", "Value"]
)

metrics_df.to_csv(
    OUTPUT / "standardized_test_metrics.csv",
    index=False
)

print("\nSTANDARDIZED TEST METRICS:")
print(metrics_df.to_string(index=False))

# ============================================================
# STEP 15: FINISHED
# ============================================================

print("\n" + "=" * 55)
print("ALL THREE EXPERIMENTS COMPLETED!")
print("=" * 55)

print("\nResults saved in:")
print(OUTPUT)

print("\nGenerated:")
print("- Threshold comparison CSV")
print("- Recall target comparison CSV")
print("- Threshold analysis graph")
print("- Calibration results and graphs")
print("- ICU ranking simulation results")
print("- ICU ranking comparison graphs")
print("- Final summary text file")

print("\nSCRIPT COMPLETED SUCCESSFULLY!")
