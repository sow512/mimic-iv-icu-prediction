#!/usr/bin/env python3
"""Phase 13: resumable Optuna tuning for MIMIC-IV renal ICU transfer.

Research only. Reuses Phase 9C patient-disjoint split assignments, Phase 1/2/9
features. Does NOT access evaluation labels or predictions during tuning.

Example:
 python3 phase13_optuna_tuning.py --phase1 ... --phase2 ... --phase9 ... \
   --phase9c ... --output ... --trials 100 --threads 4

After reviewing development results, optional ONE-TIME evaluation:
 python3 phase13_optuna_tuning.py [same arguments] --finalize
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss, log_loss, balanced_accuracy_score,
                             confusion_matrix, precision_score, recall_score, f1_score)
from sklearn.model_selection import GroupShuffleSplit

SEED = 20261017
TARGET = 'icu_transfer_24_72h'
FORBIDDEN = ('icu_transfer', 'icu_intime', 'first_icu', 'dischtime',
             'deathtime', 'horizon_end', 'label_status', 'hours_to_icu')


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_table(path, base, name):
    df = pd.read_parquet(path)
    if 'hadm_id' not in df or df.hadm_id.isna().any() or df.hadm_id.duplicated().any():
        raise ValueError(f'{name}: invalid or duplicated hadm_id')
    if 'subject_id' in df:
        check = base[['hadm_id', 'subject_id']].merge(
            df[['hadm_id', 'subject_id']], on='hadm_id', how='left',
            validate='one_to_one', suffixes=('_base', '_source'))
        bad = check.subject_id_source.notna() & (check.subject_id_base != check.subject_id_source)
        if bad.any():
            raise ValueError(f'{name}: subject_id mismatch')
    return df.drop(columns=['subject_id', 'landmark_time'], errors='ignore')


def build_data(args):
    p1, p2, p9 = Path(args.phase1), Path(args.phase2), Path(args.phase9)
    base = pd.read_parquet(p1 / 'phase1_landmark_predictors.parquet')
    labels = pd.read_parquet(p1 / 'phase1_labels.parquet')
    for name, d in [('base', base), ('labels', labels)]:
        if d.hadm_id.isna().any() or d.hadm_id.duplicated().any():
            raise ValueError(f'{name}: invalid hadm_id')
    paths = [p2 / 'phase2_temporal_lab_features.parquet',
             p9 / 'phase9a_history_omr_features.parquet',
             p9 / 'phase9b_medication_comorbidity_features.parquet']
    data = base.copy()
    for name, path in zip(['labs', 'history', 'clinical'], paths):
        t = load_table(path, base, name)
        t = t.drop(columns=list((set(t.columns) & set(data.columns)) - {'hadm_id'}))
        illegal = [c for c in t.columns if c != 'hadm_id' and any(k in c.lower() for k in FORBIDDEN)]
        if illegal:
            raise ValueError(f'{name}: leakage-suspect columns: {illegal}')
        data = data.merge(t, on='hadm_id', how='left', validate='one_to_one')
    data = data.merge(labels[['subject_id', 'hadm_id', TARGET]],
                      on=['subject_id', 'hadm_id'], how='inner', validate='one_to_one')
    if len(data) != len(base) or len(data) != len(labels):
        raise ValueError('Merge changed eligible cohort')
    columns = [c for c in data if c not in ('subject_id', 'hadm_id', 'landmark_time', TARGET)]
    if any(any(k in c.lower() for k in FORBIDDEN) for c in columns):
        raise ValueError('Potential leakage in features')
    return data, columns


def prepare(data, columns, train_idx):
    X = data[columns].copy()
    cats = list(X.select_dtypes(include=['object', 'string', 'category']).columns)
    for c in cats:
        X[c] = X[c].astype('string').fillna('__MISSING__').astype('category')
    nums = [c for c in X if c not in cats]
    X[nums] = X[nums].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
    keep = X.iloc[train_idx].columns[X.iloc[train_idx].notna().mean() >= .02].tolist()
    X = X[keep]
    cats = [c for c in cats if c in keep]
    return X, cats


def cb_frame(x, cats):
    z = x.copy()
    for c in cats:
        z[c] = z[c].astype('string').fillna('__MISSING__').astype(str)
    return z


def load_splits(data, path):
    s = pd.read_csv(path)
    required = {'subject_id', 'hadm_id', 'split'}
    if not required.issubset(s.columns):
        raise ValueError(f'Split assignments missing {required-set(s.columns)}')
    if s.hadm_id.duplicated().any() or len(s) != len(data):
        raise ValueError('Split file has duplicated/missing admissions')
    linked = data[['subject_id', 'hadm_id']].merge(s, on=['subject_id', 'hadm_id'],
                                                  how='left', validate='one_to_one', sort=False)
    if linked.split.isna().any():
        raise ValueError('Split file does not match cohort')
    expected = {'train', 'early_stop', 'blend', 'threshold', 'evaluation'}
    if set(linked.split.unique()) != expected:
        raise ValueError('Unexpected Phase 9C split names')
    if linked.groupby('subject_id').split.nunique().gt(1).any():
        raise ValueError('Patient overlap between splits')
    return {name: np.flatnonzero(linked.split.to_numpy() == name) for name in sorted(expected)}


def make_model(arch, params, threads, seed, n_estimators=None):
    if arch == 'lightgbm':
        return LGBMClassifier(n_estimators=n_estimators or 1800, objective='binary',
                              random_state=seed, n_jobs=threads, verbosity=-1, **params)
    return CatBoostClassifier(iterations=n_estimators or 1800,
                              loss_function='Logloss', eval_metric='PRAUC',
                              random_seed=seed, thread_count=threads,
                              verbose=False, allow_writing_files=False, **params)


def fit_model(arch, params, Xfit, yfit, Xstop, ystop, cats, threads, seed, rounds=1800):
    model = make_model(arch, params, threads, seed, rounds)
    if arch == 'lightgbm':
        model.fit(Xfit, yfit, eval_set=[(Xstop, ystop)], eval_metric='average_precision',
                  categorical_feature=cats,
                  callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        best = int(model.best_iteration_ or rounds)
    else:
        model.fit(cb_frame(Xfit, cats), yfit, cat_features=cats,
                  eval_set=(cb_frame(Xstop, cats), ystop),
                  early_stopping_rounds=100, verbose=False)
        best = int(model.best_iteration_ + 1)
    return model, max(1, best)


def predict(model, arch, X, cats):
    return model.predict_proba(cb_frame(X, cats) if arch == 'catboost' else X)[:, 1]


def metrics(y, p, threshold=None):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    out = {'n': int(len(y)), 'positives': int(np.sum(y)),
           'auroc': float(roc_auc_score(y, p)),
           'auprc': float(average_precision_score(y, p)),
           'brier': float(brier_score_loss(y, p)),
           'log_loss': float(log_loss(y, p))}
    if threshold is not None:
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        out.update(threshold=float(threshold), accuracy=float((tn+tp)/len(y)),
                   balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                   sensitivity=float(recall_score(y, pred, zero_division=0)),
                   specificity=float(tn/(tn+fp)),
                   precision=float(precision_score(y, pred, zero_division=0)),
                   f1=float(f1_score(y, pred, zero_division=0)),
                   tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return out


def select_threshold(y, p):
    thresholds = np.unique(np.r_[np.linspace(.005, .5, 250), .5])
    return float(max(((balanced_accuracy_score(y, p >= t), float(t)) for t in thresholds),
                     key=lambda a: (a[0], a[1]))[1])


def get_params(trial, arch):
    if arch == 'lightgbm':
        depth = trial.suggest_int('max_depth', 3, 12)
        return {'learning_rate': trial.suggest_float('learning_rate', .008, .12, log=True),
                'max_depth': depth,
                'num_leaves': trial.suggest_int('num_leaves', 8, min(127, 2**depth)),
                'min_child_samples': trial.suggest_int('min_child_samples', 25, 180),
                'subsample': trial.suggest_float('subsample', .65, 1),
                'subsample_freq': 1,
                'colsample_bytree': trial.suggest_float('colsample_bytree', .65, 1),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 25, log=True)}
    return {'depth': trial.suggest_int('depth', 4, 9),
            'learning_rate': trial.suggest_float('learning_rate', .008, .12, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 30, log=True),
            'random_strength': trial.suggest_float('random_strength', 0, 4),
            'border_count': trial.suggest_categorical('border_count', [64, 128, 254]),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 2),
            'bootstrap_type': 'Bayesian'}


def cv_objective(trial, arch, X, y, groups, train_indices, cats, args):
    # Three patient-disjoint score folds within Phase 9C TRAIN ONLY.
    # Each fold's early-stopping patients are distinct from its score patients.
    local_groups = groups.iloc[train_indices].to_numpy()
    splitter = GroupShuffleSplit(n_splits=args.folds, test_size=.20, random_state=SEED+100)
    params = get_params(trial, arch)
    fold_scores, best_rounds = [], []
    for fold, (dev_loc, score_loc) in enumerate(splitter.split(np.zeros(len(train_indices)),
                                                                groups=local_groups)):
        dev_idx = train_indices[dev_loc]
        score_idx = train_indices[score_loc]
        inner = GroupShuffleSplit(n_splits=1, test_size=.18, random_state=SEED+200+fold)
        fit_loc, stop_loc = next(inner.split(np.zeros(len(dev_idx)), groups=groups.iloc[dev_idx]))
        fit_idx, stop_idx = dev_idx[fit_loc], dev_idx[stop_loc]
        for idx in (fit_idx, stop_idx, score_idx):
            if np.unique(y[idx]).size < 2:
                raise ValueError('Fold contains only one class')
        model, rounds = fit_model(arch, params, X.iloc[fit_idx], y[fit_idx],
                                  X.iloc[stop_idx], y[stop_idx], cats,
                                  args.threads, SEED+fold)
        p = predict(model, arch, X.iloc[score_idx], cats)
        fold_scores.append(float(average_precision_score(y[score_idx], p)))
        best_rounds.append(rounds)
        trial.report(float(np.mean(fold_scores)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr('fold_auprc', fold_scores)
    trial.set_user_attr('best_rounds', best_rounds)
    return float(np.mean(fold_scores))


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)


def main(args):
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    data, raw_cols = build_data(args)
    split_file = Path(args.phase9c) / 'split_assignments.csv'
    splits = load_splits(data, split_file)
    y = data[TARGET].astype(int).to_numpy()
    groups = data.subject_id
    X, cats = prepare(data, raw_cols, splits['train'])
    saved_bundle = Path(args.phase9c) / 'E_all_models.joblib'
    if not saved_bundle.exists():
        raise FileNotFoundError(f'Required Phase 9C reference model missing: {saved_bundle}')
    original_bundle = joblib.load(saved_bundle)
    original_columns = list(original_bundle['columns'])
    current_columns = list(X.columns)
    missing = sorted(set(original_columns) - set(current_columns))
    extra = sorted(set(current_columns) - set(original_columns))
    print('\nPHASE 9C FEATURE SCHEMA CHECK', flush=True)
    print(f'Phase 9C: {len(original_columns)} | Phase 13: {len(current_columns)}', flush=True)
    print(f'Missing: {len(missing)} | Extra: {len(extra)}', flush=True)
    if missing:
        print('Missing feature names:', missing[:80], flush=True)
        raise ValueError('Required Phase 9C features missing; cannot safely continue')
    if extra:
        print('Extra reconstructed features excluded to match saved Phase 9C schema:', extra[:80], flush=True)
    if len(original_columns) != len(set(original_columns)):
        raise ValueError('Duplicate names in saved Phase 9C schema')
    X = X.loc[:, original_columns].copy()
    cats = [c for c in cats if c in original_columns]
    saved_cats = list(original_bundle['categorical'])
    if set(cats) != set(saved_cats):
        raise ValueError(f'Categorical mismatch: reconstructed={cats}, saved={saved_cats}')
    cats = saved_cats
    print('Schema aligned to Phase 9C reference; checking 600 predictor values remains important.', flush=True)
    if not set(np.unique(y)) == {0, 1}:
        raise ValueError('Outcome is not binary')
    manifest = {'seed': SEED, 'phase9c_split_sha256': file_hash(split_file),
                'n_admissions': len(data), 'n_features': X.shape[1],
                'feature_names_sha256': hashlib.sha256('\n'.join(X.columns).encode()).hexdigest(),
                'folds': args.folds, 'search_version': 2,
                'note': 'Tuning uses only Phase 9C train patients. Evaluation held out.'}
    manifest_path = out / 'tuning_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Existing tuning manifest differs: use a new output directory')
    else:
        atomic_json(manifest_path, manifest)
    print(f'Loaded {len(data):,} admissions; {X.shape[1]} predictors; '
          f'{len(splits["train"]):,} training admissions; {args.folds} CV folds.', flush=True)
    if args.finalize:
        finalize(args, out, data, X, y, groups, cats, splits)
        return
    for arch in ('lightgbm', 'catboost'):
        db = out / f'{arch}_optuna.sqlite3'
        study = optuna.create_study(study_name=f'phase13_{arch}', direction='maximize',
                                    storage=f'sqlite:///{db}', load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(seed=SEED),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=12))
        complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        remaining = max(0, args.trials - complete)
        print(f'\n{arch}: {complete} complete trials; running up to {remaining} more.', flush=True)
        attempts = 0
        while remaining > 0:
            before = len(study.trials)
            study.optimize(lambda t: cv_objective(t, arch, X, y, groups, splits['train'], cats, args),
                           n_trials=remaining, gc_after_trial=True, show_progress_bar=False,
                           catch=(ValueError,))
            complete_now = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
            remaining = max(0, args.trials - complete_now)
            attempts += len(study.trials) - before
            if remaining and attempts >= max(30, args.trials * 3):
                raise RuntimeError(f'{arch}: too many failed/pruned trials; inspect study')
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            raise RuntimeError(f'{arch}: no successful trials')
        best = study.best_trial
        report = {'architecture': arch, 'best_cv_auprc': study.best_value,
                  'best_trial_number': best.number, 'best_params': best.params,
                  'best_rounds_by_fold': best.user_attrs.get('best_rounds', []),
                  'completed_trials': len(complete), 'total_trials': len(study.trials)}
        atomic_json(out / f'{arch}_best_params.json', report)
        study.trials_dataframe().to_csv(out / f'{arch}_trial_history.csv', index=False)
        print(json.dumps(report, indent=2), flush=True)
    print('\nTuning finished. Review CV results. Run --finalize ONCE to evaluate frozen models.', flush=True)


def finalize(args, out, data, X, y, groups, cats, splits):
    marker = out / 'FINAL_EVALUATION_COMPLETED.json'
    if marker.exists():
        raise RuntimeError('Final evaluation already completed. Refusing repeat evaluation.')
    results = {}
    fitted = {}
    pred_blend = {}
    # Use Phase 9C train for fitting and early_stop ONLY for choosing boosting rounds.
    # Blend weights on Phase 9C blend; thresholds on Phase 9C threshold.
    for arch in ('lightgbm', 'catboost'):
        report_path = out / f'{arch}_best_params.json'
        if not report_path.exists():
            raise RuntimeError(f'Missing {report_path}; finish tuning first')
        report = json.loads(report_path.read_text())
        if report['completed_trials'] < args.trials:
            raise RuntimeError(f'{arch}: fewer than {args.trials} completed trials')
        params = report['best_params'].copy()
        # Fixed params include constants not represented as Optuna search dimensions.
        if arch == 'lightgbm':
            params['subsample_freq'] = 1
        else:
            params['bootstrap_type'] = 'Bayesian'
        fit, early = splits['train'], splits['early_stop']
        model, rounds = fit_model(arch, params, X.iloc[fit], y[fit],
                                  X.iloc[early], y[early], cats, args.threads, SEED)
        fitted[arch] = model
        results[arch] = {'best_rounds': rounds, 'cv_auprc': report['best_cv_auprc']}
        pred_blend[arch] = predict(model, arch, X.iloc[splits['blend']], cats)
    blend_idx = splits['blend']
    weights = np.linspace(0, 1, 21)
    w = float(max(((average_precision_score(y[blend_idx],
                       z*pred_blend['lightgbm']+(1-z)*pred_blend['catboost']), float(z))
                   for z in weights), key=lambda item: item[0])[1])
    threshold_idx = splits['threshold']
    th_probs = {a: predict(m, a, X.iloc[threshold_idx], cats) for a, m in fitted.items()}
    th_probs['blend'] = w*th_probs['lightgbm']+(1-w)*th_probs['catboost']
    thresholds = {a: select_threshold(y[threshold_idx], p) for a, p in th_probs.items()}
    # Evaluation labels are accessed only AFTER params, blend weight and thresholds freeze.
    eval_idx = splits['evaluation']
    eval_probs = {a: predict(m, a, X.iloc[eval_idx], cats) for a, m in fitted.items()}
    eval_probs['blend'] = w*eval_probs['lightgbm']+(1-w)*eval_probs['catboost']
    for a, p in eval_probs.items():
        results[a] = {**results.get(a, {}), **metrics(y[eval_idx], p, thresholds[a])}
    predictions = data.iloc[eval_idx][['subject_id', 'hadm_id']].copy()
    predictions['outcome'] = y[eval_idx]
    for a, p in eval_probs.items():
        predictions[a+'_probability'] = p
    joblib.dump({'lightgbm': fitted['lightgbm'], 'catboost': fitted['catboost'],
                 'columns': list(X.columns), 'categorical': cats,
                 'blend_lightgbm_weight': w, 'thresholds': thresholds},
                out / 'phase13_models.joblib')
    predictions.to_parquet(out / 'phase13_evaluation_predictions.parquet', index=False)
    final = {'models': results, 'blend_lightgbm_weight': w,
             'thresholds_selected_on': 'Phase 9C threshold split',
             'limitations': ['Previously inspected MIMIC-IV evaluation data: exploratory reassessment.',
                             'Retrospective AKI/CKD coding and follow-up exclusions may bias results.',
                             'No external validation. Do not use for clinical decisions.']}
    atomic_json(out / 'phase13_final_report.json', final)
    atomic_json(marker, {'completed': True, 'evaluation_admissions': len(eval_idx)})
    print(json.dumps(final, indent=2), flush=True)
    print('Finalized once. Saved models, predictions, and report to', out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase1', required=True)
    p.add_argument('--phase2', required=True)
    p.add_argument('--phase9', required=True)
    p.add_argument('--phase9c', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--finalize', action='store_true')
    a = p.parse_args()
    if a.trials < 1 or a.folds < 2 or a.threads < 1:
        p.error('--trials >=1, --folds >=2, --threads >=1 required')
    main(a)
