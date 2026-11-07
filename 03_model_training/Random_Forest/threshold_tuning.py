
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load training dataset
data = pd.read_csv("kidney_final_train_dataset.csv")

X = data.drop(columns=["icu_transfer"])
y = data["icu_transfer"]

# Recreate original validation split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Load your already trained tuned model
model = joblib.load(
    "results_tuned/random_forest_tuned.pkl"
)

# Get predicted ICU probabilities
y_prob = model.predict_proba(X_val)[:, 1]

# Test different thresholds
thresholds = np.arange(0.05, 0.51, 0.05)

results = []

print("\nTHRESHOLD COMPARISON\n")

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_val, y_pred, zero_division=0
    )

    recall = recall_score(
        y_val, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_val, y_pred, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val, y_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "Threshold": round(threshold, 2),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn
    })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

# Save threshold results
results_df.to_csv(
    "results_tuned/threshold_comparison.csv",
    index=False
)

print("\nThreshold comparison saved successfully!")
