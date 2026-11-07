"""Step 58: SHAP explanations for frozen Step 53 timing-corrected 75-feature XGBoost.
Read-only with respect to saved model/data. Test set is previously examined: exploratory.
"""
from pathlib import Path
import json
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier, DMatrix

PROJECT = Path(r'C:\Users\sowmy\Downloads\MIMIC-IV\MIMIC_IV\MIMIC_IV')
SOURCE = PROJECT / 'model training figures' / 'expanded_lab_xgboost'
MODEL_DIR = SOURCE / 'step53_final_75_model'
FEATURE_FILE = SOURCE / 'step51_timing_corrected' / 'expanded_candidate_features_storetime_corrected.parquet'
COHORT_FILE = SOURCE / 'cohort_audit.csv'
OUT = SOURCE / 'step58_shap_explainability'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
SAMPLE_SIZE = 2000
TOP_N = 20

print('STEP 58: Loading frozen 75-feature model and corrected dataset...', flush=True)
for f in [FEATURE_FILE, COHORT_FILE, MODEL_DIR / 'final_inference_bundle.joblib',
          MODEL_DIR / 'final_xgboost_75_model.json', MODEL_DIR / 'final_model_metadata.json']:
    if not f.exists():
        raise FileNotFoundError(f)
cohort = pd.read_csv(COHORT_FILE, usecols=['subject_id', 'hadm_id', 'icu_transfer'])
features = pd.read_parquet(FEATURE_FILE)
if cohort.hadm_id.duplicated().any() or features.hadm_id.duplicated().any():
    raise ValueError('Duplicate admission IDs')
if set(cohort.hadm_id) != set(features.hadm_id):
    raise ValueError('Cohort and feature admissions differ')
with open(MODEL_DIR / 'final_model_metadata.json', encoding='utf-8') as f:
    metadata = json.load(f)
threshold = float(metadata['selected_threshold'])
bundle = joblib.load(MODEL_DIR / 'final_inference_bundle.joblib')
model = XGBClassifier()
model.load_model(str(MODEL_DIR / 'final_xgboost_75_model.json'))

split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
_, test_idx = next(split.split(cohort, cohort.icu_transfer, groups=cohort.subject_id))
test_cohort = cohort.iloc[test_idx]
test = features[features.hadm_id.isin(test_cohort.hadm_id)].copy().reset_index(drop=True)
if len(test) != len(test_cohort):
    raise ValueError('Test cohort size mismatch')
if len(test) != int(metadata['reserved_test_admissions']):
    raise ValueError('Test count differs from Step 53')
y = test.hadm_id.map(test_cohort.set_index('hadm_id').icu_transfer).astype(int).to_numpy()
if 'icu_transfer' in test and not np.array_equal(y, test.icu_transfer.astype(int).to_numpy()):
    raise ValueError('Outcome labels mismatch')

raw_cols = list(bundle['raw_columns'])
cat_cols = list(bundle['categorical_columns'])
missing = sorted(set(raw_cols) - set(test.columns))
if missing:
    raise ValueError(f'Missing raw columns: {missing[:10]}')
input_frame = test[raw_cols].copy()
for c in cat_cols:
    input_frame[c] = input_frame[c].fillna('Unknown').astype(str)
pre = bundle['preprocessor']
all_X = np.asarray(pre.transform(input_frame), dtype=np.float32)
selected_idx = np.asarray(bundle['selected_indices'], dtype=int)
names = list(bundle['selected_feature_names'])
if len(names) != 75 or len(selected_idx) != 75:
    raise ValueError('Expected exactly 75 features')
encoded_names = list(pre.get_feature_names_out())
if [encoded_names[i] for i in selected_idx] != names:
    raise ValueError('Saved feature order mismatch')
X = np.ascontiguousarray(all_X[:, selected_idx])
if model.n_features_in_ != X.shape[1]:
    raise ValueError('Model and feature count mismatch')
prob = model.predict_proba(X)[:, 1]
pred = (prob >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
print(f'Test admissions={len(test):,}; features={len(names)}; threshold={threshold:.6f}')
print(f'Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}', flush=True)

# For interpretability, sample across true/predicted classes, not for performance estimates.
rng = np.random.default_rng(SEED)
strata = {(a, b): np.flatnonzero((y == a) & (pred == b)) for a in (0, 1) for b in (0, 1)}
selected_rows = []
for (a, b), indices in strata.items():
    n = min(len(indices), SAMPLE_SIZE // 4)
    if n:
        selected_rows.extend(rng.choice(indices, size=n, replace=False).tolist())
remaining = SAMPLE_SIZE - len(selected_rows)
if remaining > 0:
    unused = np.setdiff1d(np.arange(len(y)), np.asarray(selected_rows, dtype=int))
    selected_rows.extend(rng.choice(unused, size=min(remaining, len(unused)), replace=False).tolist())
sample_idx = np.asarray(selected_rows, dtype=int)
X_sample = X[sample_idx]

# TreeSHAP contributions from XGBoost: last column is bias; contributions in raw margin/log-odds.
booster = model.get_booster()
contrib = booster.predict(DMatrix(X_sample), pred_contribs=True, validate_features=False)
shap_values = np.asarray(contrib[:, :-1], dtype=float)
bias = np.asarray(contrib[:, -1], dtype=float)
if shap_values.shape != X_sample.shape:
    raise ValueError('SHAP dimensions mismatch')
raw_margin = booster.predict(DMatrix(X_sample), output_margin=True, validate_features=False)
if not np.allclose(shap_values.sum(axis=1) + bias, raw_margin, atol=1e-3):
    raise ValueError('SHAP additivity check failed')
importance = np.abs(shap_values).mean(axis=0)
rank_idx = np.argsort(-importance, kind='stable')
ranked = pd.DataFrame({'rank': np.arange(1, len(names)+1),
                       'feature': np.asarray(names)[rank_idx],
                       'mean_absolute_shap_log_odds': importance[rank_idx]})
ranked.to_csv(OUT / 'shap_feature_importance.csv', index=False)

# Native matplotlib charts avoid SHAP version-specific plotting dependencies.
order = rank_idx[:TOP_N][::-1]
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(np.asarray(names)[order], importance[order])
ax.set_xlabel('Mean absolute SHAP value (log-odds)')
ax.set_title('75-feature XGBoost: top 20 feature importance')
fig.tight_layout()
fig.savefig(OUT / 'shap_top20_bar.png', dpi=250, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 8))
for pos, j in enumerate(order):
    vals = X_sample[:, j]
    finite = np.isfinite(vals)
    if finite.sum() > 0:
        vmin, vmax = np.nanpercentile(vals[finite], [5, 95])
        if vmax <= vmin:
            vmin, vmax = float(np.nanmin(vals[finite])), float(np.nanmax(vals[finite]))
        colors = np.clip((vals - vmin) / (vmax - vmin), 0, 1) if vmax > vmin else np.full(len(vals), 0.5)
    else:
        colors = np.full(len(vals), 0.5)
    jitter = rng.normal(0, 0.13, len(vals))
    ax.scatter(shap_values[:, j], np.full(len(vals), pos) + jitter,
               c=colors, cmap='coolwarm', vmin=0, vmax=1, s=7, alpha=0.4, rasterized=True)
ax.axvline(0, color='gray', linewidth=0.8)
ax.set_yticks(np.arange(len(order)), np.asarray(names)[order])
ax.set_xlabel('SHAP contribution to ICU-transfer log-odds')
ax.set_title('SHAP beeswarm (blue = lower value, red = higher value)')
fig.tight_layout()
fig.savefig(OUT / 'shap_beeswarm_top20.png', dpi=250, bbox_inches='tight')
plt.close(fig)

# Dependence: SHAP value vs raw (preprocessed) feature value, for top three lab features.
lab_top = [j for j in rank_idx if re.match(r'^lab_\d+_', names[j])][:3]
for k, j in enumerate(lab_top, 1):
    xval = X_sample[:, j]
    finite = np.isfinite(xval) & np.isfinite(shap_values[:, j])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xval[finite], shap_values[finite, j], s=10, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel(f'{names[j]} (model input units)')
    ax.set_ylabel('SHAP contribution (log-odds)')
    ax.set_title(f'SHAP dependence: {names[j]}')
    fig.tight_layout()
    fig.savefig(OUT / f'shap_dependence_{k}.png', dpi=250, bbox_inches='tight')
    plt.close(fig)

# Group SHAP values by lab item ID; aggregate absolute grouped contribution per patient.
groups = {}
for j, name in enumerate(names):
    match = re.match(r'^lab_(\d+)_', name)
    group = f'lab_{match.group(1)}' if match else 'non_lab'
    groups.setdefault(group, []).append(j)
group_rows = []
for group, indices in groups.items():
    grouped_contrib = shap_values[:, indices].sum(axis=1)
    group_rows.append({'group': group, 'feature_count': len(indices),
                       'mean_abs_group_shap': float(np.abs(grouped_contrib).mean()),
                       'sum_individual_mean_abs_shap': float(np.abs(shap_values[:, indices]).mean(axis=0).sum())})
group_df = pd.DataFrame(group_rows).sort_values('mean_abs_group_shap', ascending=False)
group_df.to_csv(OUT / 'lab_group_shap_importance.csv', index=False)
fig, ax = plt.subplots(figsize=(9, 7))
top_groups = group_df.head(20).iloc[::-1]
ax.barh(top_groups.group, top_groups.mean_abs_group_shap)
ax.set_xlabel('Mean absolute grouped SHAP contribution (log-odds)')
ax.set_title('Top lab-ID groups and admission features')
fig.tight_layout()
fig.savefig(OUT / 'lab_group_importance.png', dpi=250, bbox_inches='tight')
plt.close(fig)

# Case-level waterfall-style contribution charts for each confusion-matrix category.
case_rows = []
for label, actual, predicted in [('true_positive', 1, 1), ('true_negative', 0, 0),
                                 ('false_positive', 0, 1), ('false_negative', 1, 0)]:
    candidates = np.flatnonzero((y == actual) & (pred == predicted))
    if not len(candidates):
        continue
    # Representative example: closest probability to category median.
    chosen = candidates[np.argmin(np.abs(prob[candidates] - np.median(prob[candidates])))]
    one = booster.predict(DMatrix(X[chosen:chosen+1]), pred_contribs=True, validate_features=False)[0]
    values = one[:-1]
    top = np.argsort(-np.abs(values))[:12]
    other = float(values.sum() - values[top].sum())
    plot_names = [names[j] for j in top] + ['Other features']
    plot_values = np.r_[values[top], other]
    # Cumulative additive decomposition, sorted by absolute contribution.
    base = float(one[-1])
    starts = base + np.r_[0, np.cumsum(plot_values[:-1])]
    fig, ax = plt.subplots(figsize=(10, 7))
    for pos, (start, contribution) in enumerate(zip(starts, plot_values)):
        ax.barh(pos, abs(contribution), left=min(start, start + contribution),
                color='tab:red' if contribution > 0 else 'tab:blue', alpha=0.85)
        ax.text(start + contribution, pos, f' {contribution:+.3f}', va='center', fontsize=8)
    ax.set_yticks(np.arange(len(plot_names)), plot_names)
    ax.invert_yaxis()
    ax.set_xlabel('Cumulative model output (log-odds)')
    ax.set_title(f'{label.replace("_", " ").title()} | p={prob[chosen]:.4f}, threshold={threshold:.4f}')
    fig.tight_layout()
    fig.savefig(OUT / f'shap_waterfall_{label}.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    case_rows.append({'case_type': label, 'hadm_id': int(test.iloc[chosen].hadm_id),
                      'actual': actual, 'predicted': predicted,
                      'probability': float(prob[chosen]), 'baseline_log_odds': base,
                      'total_shap_log_odds': float(values.sum())})
pd.DataFrame(case_rows).to_csv(OUT / 'selected_case_explanations.csv', index=False)

# Summary is exploratory, not a clinical explanation or new performance estimate.
summary = {'model': 'Step 53 frozen timing-corrected XGBoost', 'features': len(names),
           'test_admissions': len(test), 'sampled_for_shap': len(sample_idx),
           'threshold': threshold, 'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
           'shap_units': 'raw model margin (log-odds)',
           'notes': ['Model and preprocessor were not retrained.',
                     'Test set has been examined in prior experiments; explanations are exploratory.',
                     'SHAP is associational, not causal.',
                     'Four after-discharge positive admissions remain pending review.']}
(OUT / 'shap_analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('\nTop 15 SHAP features:')
print(ranked.head(15).to_string(index=False))
print('\nTop 10 grouped lab/importances:')
print(group_df.head(10).to_string(index=False))
print('\nSTEP 58 COMPLETE. Results saved to:', OUT)
