#!/usr/bin/env python3
"""Phase 9C: controlled LightGBM/CatBoost feature-group ablation.

Research only. Inputs: existing Phase 1/2/9A/9B Parquet files.
Runs five prespecified feature groups with two models and an ensemble each.
Patient-disjoint splits: train / early-stop / blend / threshold / evaluation.
No test-based model or threshold selection. All output rows are admissions.
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit

SEED = 20261017
TARGET = 'icu_transfer_24_72h'
IDS = {'subject_id', 'hadm_id', 'landmark_time', TARGET}
FORBIDDEN = ('icu_transfer', 'icu_intime', 'first_icu', 'dischtime',
             'deathtime', 'horizon_end', 'label_status', 'hours_to_icu')


def grouped_partition(indices, groups, fraction, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    a, b = next(splitter.split(np.zeros(len(indices)), groups=groups.iloc[indices]))
    return indices[a], indices[b]


def load_feature_table(path, base, name):
    df = pd.read_parquet(path)
    if 'hadm_id' not in df or df.hadm_id.isna().any() or df.hadm_id.duplicated().any():
        raise ValueError(f'{name}: missing/duplicated hadm_id')
    if 'subject_id' in df:
        link = base[['hadm_id', 'subject_id']].merge(
            df[['hadm_id', 'subject_id']], on='hadm_id', how='left',
            validate='one_to_one', suffixes=('_base', '_new'))
        mismatch = link.subject_id_new.notna() & (link.subject_id_base != link.subject_id_new)
        if mismatch.any():
            raise ValueError(f'{name}: subject_id mismatch')
    return df.drop(columns=['subject_id', 'landmark_time'], errors='ignore')


def build_data(args):
    base = pd.read_parquet(Path(args.phase1) / 'phase1_landmark_predictors.parquet')
    labels = pd.read_parquet(Path(args.phase1) / 'phase1_labels.parquet')
    if base.hadm_id.duplicated().any() or labels.hadm_id.duplicated().any():
        raise ValueError('Duplicated admission in base or labels')
    labs = load_feature_table(Path(args.phase2) / 'phase2_temporal_lab_features.parquet', base, 'labs')
    history = load_feature_table(Path(args.phase9) / 'phase9a_history_omr_features.parquet', base, 'history')
    clinical = load_feature_table(Path(args.phase9) / 'phase9b_medication_comorbidity_features.parquet', base, 'clinical')
    for name, df in [('labs', labs), ('history', history), ('clinical', clinical)]:
        redundant = (set(df.columns) & set(base.columns)) - {'hadm_id'}
        if redundant:
            df.drop(columns=sorted(redundant), inplace=True)
        illegal = [c for c in df if c != 'hadm_id' and any(t in c.lower() for t in FORBIDDEN)]
        if illegal:
            raise ValueError(f'{name}: suspicious feature columns {illegal}')
    rx = [c for c in clinical if c.startswith('rx_')]
    comorb = [c for c in clinical if c != 'hadm_id' and c not in rx]
    if not rx or not comorb:
        raise ValueError('Expected rx_ and historical comorbidity features in Phase 9B')
    if set(labs.columns) & set(history.columns) - {'hadm_id'}:
        raise ValueError('Lab/history columns overlap')
    if set(labs.columns) & set(clinical.columns) - {'hadm_id'}:
        raise ValueError('Lab/clinical columns overlap')
    if set(history.columns) & set(clinical.columns) - {'hadm_id'}:
        raise ValueError('History/clinical columns overlap')
    data = base.merge(labs, on='hadm_id', how='left', validate='one_to_one')
    data = data.merge(history, on='hadm_id', how='left', validate='one_to_one')
    data = data.merge(clinical, on='hadm_id', how='left', validate='one_to_one')
    data = data.merge(labels[['subject_id', 'hadm_id', TARGET]],
                      on=['subject_id', 'hadm_id'], how='inner', validate='one_to_one')
    if len(data) != len(base) or len(data) != len(labels):
        raise ValueError('Lost or gained admissions during merge')
    lab_cols = [c for c in labs if c != 'hadm_id']
    history_cols = [c for c in history if c != 'hadm_id']
    base_cols = [c for c in base if c not in IDS]
    groups = {
        'A_baseline': base_cols + lab_cols,
        'B_history': base_cols + lab_cols + history_cols,
        'C_medications': base_cols + lab_cols + rx,
        'D_comorbidities': base_cols + lab_cols + comorb,
        'E_all': base_cols + lab_cols + history_cols + rx + comorb,
    }
    for name, cols in groups.items():
        if len(cols) != len(set(cols)):
            raise ValueError(f'Duplicate predictor columns in {name}')
        bad = [c for c in cols if any(t in c.lower() for t in FORBIDDEN)]
        if bad:
            raise ValueError(f'Leakage-suspect predictors in {name}: {bad}')
    return data, groups


def prepare_features(data, columns, train_idx):
    X = data[columns].copy()
    cats = list(X.select_dtypes(include=['object', 'string', 'category']).columns)
    for c in cats:
        X[c] = X[c].astype('string').fillna('__MISSING__').astype('category')
    nums = [c for c in X if c not in cats]
    X[nums] = X[nums].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
    # Training-only missingness screen, same rule for each feature set.
    keep = X.iloc[train_idx].columns[X.iloc[train_idx].notna().mean() >= .02].tolist()
    cats = [c for c in cats if c in keep]
    return X[keep], cats


def cb_frame(frame, cats):
    out = frame.copy()
    for c in cats:
        out[c] = out[c].astype('string').fillna('__MISSING__').astype(str)
    return out


def scores(y, p, threshold):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(auroc=float(roc_auc_score(y, p)),
                auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                log_loss=float(log_loss(y, p)),
                threshold=float(threshold),
                accuracy=float((tp + tn) / len(y)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                sensitivity=float(recall_score(y, pred, zero_division=0)),
                specificity=float(tn / (tn + fp)) if (tn + fp) else float('nan'),
                precision=float(precision_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def select_threshold(y, p):
    # Thresholds are selected exclusively on threshold-validation patients.
    thresholds = np.unique(np.r_[np.linspace(.005, .5, 200), .5])
    vals = [(balanced_accuracy_score(y, p >= t), float(t)) for t in thresholds]
    return max(vals, key=lambda v: (v[0], v[1]))[1]


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data, feature_sets = build_data(args)
    y = data[TARGET].astype(int)
    groups = data.subject_id
    all_idx = np.arange(len(data))
    # Approx. 55/15/10/10/10, same patient split across all 15 models.
    remaining, evaluation = grouped_partition(all_idx, groups, .10, SEED)
    remaining, threshold = grouped_partition(remaining, groups, .10/.90, SEED + 1)
    remaining, blend = grouped_partition(remaining, groups, .10/.80, SEED + 2)
    train, early = grouped_partition(remaining, groups, .15/.70, SEED + 3)
    splits = {'train': train, 'early_stop': early, 'blend': blend,
              'threshold': threshold, 'evaluation': evaluation}
    for name, idx in splits.items():
        if y.iloc[idx].nunique() != 2:
            raise ValueError(f'{name}: missing a class')
        for other, idx2 in splits.items():
            if name < other and set(groups.iloc[idx]) & set(groups.iloc[idx2]):
                raise ValueError(f'Patient leakage: {name}/{other}')
    pd.concat([
        data.iloc[idx][['subject_id', 'hadm_id']].assign(split=name)
        for name, idx in splits.items()
    ]).to_csv(out / 'split_assignments.csv', index=False)
    manifest = {'seed': SEED, 'total_admissions': len(data),
                'total_positives': int(y.sum()),
                'splits': {k: {'admissions': len(v), 'positives': int(y.iloc[v].sum()),
                               'patients': int(groups.iloc[v].nunique())}
                           for k, v in splits.items()},
                'feature_groups': {k: len(v) for k, v in feature_sets.items()},
                'selection': 'All 5 feature sets and 3 models are prespecified; no evaluation-based winner selection.',
                'limitations': [
                    'Retrospective renal diagnosis-defined cohort; index diagnoses may be established after landmark.',
                    'Historical diagnoses come from prior discharged encounters, not verified condition onset.',
                    'Medication orders are not proof of administration; approximate lexical drug grouping.',
                    'Historical outpatient measurements are not acute ward vital signs.',
                    'Binary outcome excludes early discharge/death before horizon and may have selection bias.',
                    'Evaluation remains internal MIMIC-IV reassessment; prior test results were inspected.'
                ]}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    for set_name, cols in feature_sets.items():
        print(f'\n=== {set_name} ({len(cols)} raw predictors) ===', flush=True)
        X, cats = prepare_features(data, cols, train)
        xtr, xearly, xblend, xth, xte = [X.iloc[splits[k]].copy() for k in splits]
        ytr, yearly, yblend, yth, yte = [y.iloc[splits[k]].to_numpy() for k in splits]
        lgb = LGBMClassifier(n_estimators=1200, learning_rate=.035, num_leaves=23,
                             min_child_samples=70, colsample_bytree=.8,
                             subsample=.8, subsample_freq=1, reg_lambda=5,
                             random_state=SEED, n_jobs=args.threads, verbosity=-1)
        lgb.fit(xtr, ytr, eval_set=[(xearly, yearly)], eval_metric='average_precision',
                categorical_feature=cats,
                callbacks=[early_stopping(75, verbose=False), log_evaluation(0)])
        cat = CatBoostClassifier(iterations=1400, depth=5, learning_rate=.04,
                                 l2_leaf_reg=6, loss_function='Logloss',
                                 eval_metric='PRAUC', random_seed=SEED,
                                 thread_count=args.threads, verbose=False)
        cat.fit(cb_frame(xtr, cats), ytr, cat_features=cats,
                eval_set=(cb_frame(xearly, cats), yearly),
                early_stopping_rounds=90, verbose=False)
        def both(frame):
            return (lgb.predict_proba(frame)[:, 1],
                    cat.predict_proba(cb_frame(frame, cats))[:, 1])
        pb_l, pb_c = both(xblend)
        weight = max((float(average_precision_score(yblend, w*pb_l + (1-w)*pb_c)), float(w))
                     for w in np.linspace(0, 1, 21))[1]
        pth_l, pth_c = both(xth)
        pte_l, pte_c = both(xte)
        pred_th = {'lightgbm': pth_l, 'catboost': pth_c,
                   'blend': weight*pth_l + (1-weight)*pth_c}
        pred_te = {'lightgbm': pte_l, 'catboost': pte_c,
                   'blend': weight*pte_l + (1-weight)*pte_c}
        preds = data.iloc[evaluation][['subject_id', 'hadm_id']].copy()
        preds['outcome'] = yte
        for name in ('lightgbm', 'catboost', 'blend'):
            cutoff = select_threshold(yth, pred_th[name])
            result = {'feature_set': set_name, 'model': name,
                      'n_features': len(X.columns),
                      'n_evaluation': len(yte), 'n_positives': int(yte.sum()),
                      'blend_lightgbm_weight': weight if name == 'blend' else None,
                      **scores(yte, pred_te[name], cutoff)}
            rows.append(result)
            preds[name + '_probability'] = pred_te[name]
            print(f'{name:9s} AUROC={result["auroc"]:.4f} AUPRC={result["auprc"]:.4f} '
                  f'recall={result["sensitivity"]:.3f} balanced={result["balanced_accuracy"]:.3f}', flush=True)
        preds.to_parquet(out / f'{set_name}_evaluation_predictions.parquet', index=False)
        pd.DataFrame(rows).to_csv(out / 'ablation_results.csv', index=False)
        if args.save_models:
            joblib.dump({'lightgbm': lgb, 'catboost': cat, 'columns': list(X.columns),
                         'categorical': cats, 'blend_lightgbm_weight': weight},
                        out / f'{set_name}_models.joblib')
    result = pd.DataFrame(rows)
    print('\nFINAL PRESPECIFIED ABLATION RESULTS (descriptive, do not reselect on test):')
    print(result[['feature_set', 'model', 'n_features', 'auroc', 'auprc',
                  'balanced_accuracy', 'sensitivity', 'specificity', 'precision']]
          .to_string(index=False, float_format=lambda z: f'{z:.4f}'))
    print('\nSaved to:', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase1', required=True)
    p.add_argument('--phase2', required=True)
    p.add_argument('--phase9', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--save-models', action='store_true')
    run(p.parse_args())
