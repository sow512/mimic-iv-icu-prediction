"""Step 54: Read-only exploratory evaluation of saved Step 53 XGBoost.
No training, threshold tuning, or recalibration. Requires original saved artifacts.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score, brier_score_loss)
from sklearn.calibration import calibration_curve

PROJECT = Path(r'C:\Users\sowmy\Downloads\MIMIC-IV\MIMIC_IV\MIMIC_IV')
SOURCE = PROJECT / 'model training figures' / 'expanded_lab_xgboost'
MODEL_DIR = SOURCE / 'step53_final_75_model'
DATA_FILE = SOURCE / 'step51_timing_corrected' / 'expanded_candidate_features_storetime_corrected.parquet'
OUT = SOURCE / 'step54_75_evaluation'
OUT.mkdir(parents=True, exist_ok=True)

print('STEP 54: Loading saved corrected features and frozen model...', flush=True)
cohort = pd.read_csv(SOURCE / 'cohort_audit.csv', usecols=['hadm_id', 'subject_id', 'icu_transfer'])
features = pd.read_parquet(DATA_FILE)
with open(MODEL_DIR / 'final_model_metadata.json', encoding='utf-8') as f:
    meta = json.load(f)
bundle = joblib.load(MODEL_DIR / 'final_inference_bundle.joblib')
model = XGBClassifier()
model.load_model(str(MODEL_DIR / 'final_xgboost_75_model.json'))
threshold = float(meta['selected_threshold'])
if not (0 < threshold < 1):
    raise ValueError('Invalid saved threshold')
if cohort.hadm_id.duplicated().any() or features.hadm_id.duplicated().any():
    raise ValueError('Duplicate hadm_id')
if set(cohort.hadm_id) != set(features.hadm_id):
    raise ValueError('Cohort and corrected features do not match')

split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
dev_idx, test_idx = next(split.split(cohort, cohort.icu_transfer, groups=cohort.subject_id))
dev_cohort, test_cohort = cohort.iloc[dev_idx], cohort.iloc[test_idx]
if set(dev_cohort.subject_id) & set(test_cohort.subject_id):
    raise ValueError('Patient overlap')
if len(test_cohort) != int(meta['reserved_test_admissions']):
    raise ValueError('Test count mismatch with saved model')
test = features.loc[features.hadm_id.isin(test_cohort.hadm_id)].copy().reset_index(drop=True)
if len(test) != len(test_cohort):
    raise ValueError('Missing test admissions')
lookup = test_cohort.set_index('hadm_id')['icu_transfer']
y = test.hadm_id.map(lookup).astype(int).to_numpy()
if 'icu_transfer' in test and not np.array_equal(y, test.icu_transfer.astype(int).to_numpy()):
    raise ValueError('Target mismatch')

raw_cols = bundle['raw_columns']
cat_cols = bundle['categorical_columns']
missing = [c for c in raw_cols if c not in test.columns]
if missing:
    raise ValueError(f'Missing raw columns: {missing[:10]}')
for c in cat_cols:
    test[c] = test[c].fillna('Unknown').astype(str)
X_all = np.asarray(bundle['preprocessor'].transform(test[raw_cols]), dtype=np.float32)
selected_idx = np.asarray(bundle['selected_indices'], dtype=int)
if len(selected_idx) != 75 or selected_idx.min() < 0 or selected_idx.max() >= X_all.shape[1]:
    raise ValueError('Saved selected indices do not match preprocessing')
all_names = list(bundle['preprocessor'].get_feature_names_out())
selected_names = [all_names[i] for i in selected_idx]
if selected_names != list(bundle['selected_feature_names']):
    raise ValueError('Saved selected feature order mismatch')
X = X_all[:, selected_idx]
if model.n_features_in_ != 75:
    raise ValueError('Saved model expects a different feature count')

p = model.predict_proba(X)[:, 1]
pred = (p >= threshold).astype(int)
tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
metrics = {
    'threshold': threshold,
    'auroc': float(roc_auc_score(y, p)),
    'auprc_average_precision': float(average_precision_score(y, p)),
    'recall': float(recall_score(y, pred)),
    'precision': float(precision_score(y, pred, zero_division=0)),
    'specificity': tn / (tn + fp),
    'f1': float(f1_score(y, pred, zero_division=0)),
    'accuracy': float(accuracy_score(y, pred)),
    'brier_score': float(brier_score_loss(y, p)),
    'mean_predicted_probability': float(p.mean()),
    'observed_prevalence': float(y.mean()),
    'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    'test_admissions': len(y), 'feature_count': 75,
}
print('\nEXPLORATORY TEST METRICS')
for k, v in metrics.items():
    print(f'{k}: {v:.6f}' if isinstance(v, float) else f'{k}: {v}')
print('\nConfusion matrix [TN FP; FN TP]:\n', np.array([[tn, fp], [fn, tp]]))
print('\nClassification report:\n', classification_report(y, pred, labels=[0, 1], zero_division=0))
pd.DataFrame([metrics]).to_csv(OUT / 'test_metrics.csv', index=False)
with open(OUT / 'test_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2)

results = pd.DataFrame({'hadm_id': test.hadm_id.to_numpy(), 'actual': y,
    'predicted_probability': p, 'predicted': pred, 'correct': pred == y})
results.to_csv(OUT / 'test_predictions.csv', index=False)
examples = pd.concat([results.loc[results.actual == label].sample(n=5, random_state=42)
    for label in [0, 1]], ignore_index=True).sample(frac=1, random_state=42)
examples.to_csv(OUT / 'random_patient_check.csv', index=False)
print('\nRANDOM BALANCED CHECK (not representative of population):')
print(examples[['actual', 'predicted_probability', 'predicted', 'correct']].to_string(index=False))
print('Correct:', int(examples.correct.sum()), '/', len(examples))

cm = np.array([[tn, fp], [fn, tp]])
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(cm, cmap='Blues')
fig.colorbar(im, ax=ax, label='Number of admissions')
ax.set_xticks([0, 1], ['No ICU', 'ICU'])
ax.set_yticks([0, 1], ['No ICU', 'ICU'])
ax.set_xlabel('Predicted outcome')
ax.set_ylabel('Actual outcome')
ax.set_title(f'75-feature XGBoost confusion matrix (threshold {threshold:.5f})')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]:,}\n{["TN", "FP", "FN", "TP"][i*2+j]}',
            ha='center', va='center', fontsize=13,
            color='white' if cm[i,j] > cm.max()/2 else 'black')
fig.tight_layout()
fig.savefig(OUT / 'confusion_matrix.png', dpi=250)
plt.close(fig)

fpr, tpr, _ = roc_curve(y, p)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f'AUROC = {metrics["auroc"]:.4f}')
ax.plot([0, 1], [0, 1], '--', label='Chance')
ax.set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curve')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'roc_curve.png', dpi=250)
plt.close(fig)

precision, recall, _ = precision_recall_curve(y, p)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(recall, precision, label=f'AP = {metrics["auprc_average_precision"]:.4f}')
ax.axhline(y.mean(), linestyle='--', label=f'Prevalence = {y.mean():.4f}')
ax.set(xlabel='Recall', ylabel='Precision', title='Precision-recall curve')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'precision_recall_curve.png', dpi=250)
plt.close(fig)

# Reliability curve is descriptive only; no calibrator is fit to the test set.
frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy='quantile')
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], '--', label='Perfect calibration')
ax.plot(mean_pred, frac_pos, 'o-', label='Observed risk by decile')
ax.set(xlabel='Mean predicted probability', ylabel='Observed ICU-transfer fraction',
       title=f'Calibration curve (Brier = {metrics["brier_score"]:.4f})')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'calibration_curve.png', dpi=250)
plt.close(fig)

with open(OUT / 'evaluation_notes.txt', 'w', encoding='utf-8') as f:
    f.write('EXPLORATORY evaluation only: test set previously examined.\n'
            'Frozen Step 53 model and threshold; no retraining, tuning or recalibration.\n'
            'Four after-discharge positive flags and early-discharge policy remain unresolved.\n'
            'Calibration curve and Brier score are descriptive, not calibration correction.\n'
            'External or genuinely untouched validation is needed for unbiased claims.\n')
print('\nSTEP 54 COMPLETE. Saved to:', OUT)
