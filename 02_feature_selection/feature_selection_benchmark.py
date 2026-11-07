#!/usr/bin/env python3
"""Renal ICU feature-selection benchmark. Retrospective research only.

36 configurations: 3 selectors x 4 feature counts x 3 classifiers,
plus 3 full-feature baselines. Selection, threshold tuning, and evaluation
are patient-disjoint. Test is scored once after configurations are fixed.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score,
    balanced_accuracy_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

SEED = 42
COUNTS = (20, 40, 80, 160)
SELECTORS = ('mutual_information', 'l1_logistic', 'lightgbm_importance')
MODELS = ('logistic', 'lightgbm', 'catboost')


def split_grouped(frame):
    # 60% fit, 15% reserved validation, 15% threshold validation, 10% test.
    ids = np.arange(len(frame))
    g = frame.subject_id
    outer = GroupShuffleSplit(n_splits=1, test_size=.10, random_state=SEED)
    dev, test = next(outer.split(ids, groups=g))
    split1 = GroupShuffleSplit(n_splits=1, test_size=1/3, random_state=SEED+1)
    fit, rest = next(split1.split(dev, groups=g.iloc[dev]))
    train = dev[fit]; rest = dev[rest]
    split2 = GroupShuffleSplit(n_splits=1, test_size=.5, random_state=SEED+2)
    va, vb = next(split2.split(rest, groups=g.iloc[rest]))
    splits = {'train':train, 'reserved_validation':rest[va],
              'threshold_validation':rest[vb], 'test':test}
    for a in splits:
        for b in splits:
            if a < b:
                assert not (set(g.iloc[splits[a]]) & set(g.iloc[splits[b]])), (a,b)
    return splits


def make_data(phase1, phase2):
    base = pd.read_parquet(phase1/'phase1_landmark_predictors.parquet')
    lab = pd.read_parquet(phase2/'phase2_temporal_lab_features.parquet')
    labels = pd.read_parquet(phase1/'phase1_labels.parquet')
    for name, d in [('base',base),('lab',lab),('labels',labels)]:
        if d.hadm_id.isna().any() or d.hadm_id.duplicated().any():
            raise ValueError(f'{name} invalid or duplicate hadm_id')
    lab = lab.drop(columns=['subject_id','landmark_time'], errors='ignore')
    # Phase 2 may contain repeated baseline columns; keep Phase 1 baseline only.
    repeated = set(lab.columns).intersection(base.columns) - {'hadm_id'}
    lab = lab.drop(columns=list(repeated))
    data = base.merge(lab, on='hadm_id', how='left', validate='one_to_one')
    data = data.merge(labels[['subject_id','hadm_id','icu_transfer_24_72h']],
                      on=['subject_id','hadm_id'], how='inner', validate='one_to_one')
    if len(data) != len(base) or len(data) != len(labels):
        raise ValueError('Cohort row mismatch')
    forbidden = ['subject_id','hadm_id','landmark_time','icu_transfer_24_72h']
    X = data.drop(columns=forbidden)
    leakage_tokens = ('icu_transfer','icu_intime','dischtime','deathtime',
                      'horizon_end','label_status','hours_to_icu','first_icu')
    suspicious = [c for c in X if any(t in c.lower() for t in leakage_tokens)]
    if suspicious: raise ValueError(f'Suspected leakage columns: {suspicious}')
    y = data.icu_transfer_24_72h.astype(int)
    if set(y.unique()) != {0,1}: raise ValueError('Expected binary outcome')
    return data, X, y


def prep(X, train_idx):
    """Fit ALL preprocessing on train only. Dense representation for common feature universe."""
    X = X.copy()
    categorical = X.select_dtypes(include=['object','string','category']).columns.tolist()
    numeric = [c for c in X if c not in categorical]
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors='raise').replace([np.inf,-np.inf],np.nan)
    keep_num = [c for c in numeric if X.iloc[train_idx][c].notna().mean() >= .02]
    keep_cat = [c for c in categorical if X.iloc[train_idx][c].notna().any()]
    # Avoid imputer's all-missing columns; median learned only from training.
    numeric_imp = SimpleImputer(strategy='median')
    num = numeric_imp.fit_transform(X.iloc[train_idx][keep_num])
    all_num = numeric_imp.transform(X[keep_num])
    scaler = StandardScaler()
    scaler.fit(num)
    num_arr = scaler.transform(all_num).astype('float32')
    parts = [pd.DataFrame(num_arr, columns=keep_num, index=X.index)]
    if keep_cat:
        cat_train = X.iloc[train_idx][keep_cat].astype('string').fillna('__MISSING__').astype(str)
        cat_all = X[keep_cat].astype('string').fillna('__MISSING__').astype(str)
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
        encoder.fit(cat_train)
        cat_arr = encoder.transform(cat_all)
        names = encoder.get_feature_names_out(keep_cat).tolist()
        parts.append(pd.DataFrame(cat_arr, columns=names, index=X.index))
    result = pd.concat(parts, axis=1)
    if result.columns.duplicated().any() or result.isna().any().any():
        raise ValueError('Invalid preprocessed feature matrix')
    return result


def ranking(Xtr, ytr, selector, threads):
    if selector == 'mutual_information':
        scores = mutual_info_classif(Xtr, ytr, random_state=SEED, n_neighbors=3)
    elif selector == 'l1_logistic':
        model = LogisticRegression(penalty='l1', solver='saga', C=.05,
                                   class_weight='balanced', max_iter=2500,
                                   random_state=SEED, n_jobs=threads)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(Xtr,ytr)
        scores = np.abs(model.coef_[0])
    else:
        model = LGBMClassifier(n_estimators=250, learning_rate=.04, num_leaves=20,
                               min_child_samples=80, reg_lambda=5,
                               random_state=SEED, n_jobs=threads, verbosity=-1)
        model.fit(Xtr,ytr)
        scores = model.feature_importances_.astype(float)
    # Stable tie break by column name.
    return pd.DataFrame({'feature':Xtr.columns, 'importance':scores}).sort_values(
        ['importance','feature'], ascending=[False,True], kind='mergesort').reset_index(drop=True)


def build_model(kind, threads):
    if kind == 'logistic':
        return LogisticRegression(class_weight='balanced', max_iter=1500,
                                  random_state=SEED)
    if kind == 'lightgbm':
        return LGBMClassifier(n_estimators=250, learning_rate=.04, num_leaves=20,
                              min_child_samples=80, reg_lambda=5,
                              class_weight='balanced', random_state=SEED,
                              n_jobs=threads, verbosity=-1)
    return CatBoostClassifier(iterations=350, depth=5, learning_rate=.04,
                              l2_leaf_reg=6, auto_class_weights='Balanced',
                              random_seed=SEED, thread_count=threads, verbose=False)


def tune_threshold(y, p):
    # Threshold tuned ONLY on threshold-validation data; optimize balanced accuracy.
    thresholds = np.unique(np.r_[np.linspace(.01,.99,99), np.quantile(p, np.linspace(.01,.99,99))])
    scored = [(balanced_accuracy_score(y,p>=t), float(t)) for t in thresholds]
    return max(scored, key=lambda z:(z[0], -abs(z[1]-.5)))[1]


def evaluate(y,p,threshold):
    pred = (p>=threshold).astype(int)
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'auroc':float(roc_auc_score(y,p)),
            'auprc':float(average_precision_score(y,p)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'accuracy':float(accuracy_score(y,pred)),
            'sensitivity':float(recall_score(y,pred,zero_division=0)),
            'specificity':float(tn/(tn+fp)),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'f1':float(f1_score(y,pred,zero_division=0)),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def main(args):
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    data, raw, y = make_data(Path(args.phase1),Path(args.phase2))
    splits = split_grouped(data)
    X = prep(raw, splits['train'])
    ytr = y.iloc[splits['train']]
    Xtr = X.iloc[splits['train']]
    val_idx = splits['threshold_validation']
    test_idx = splits['test']
    yval = y.iloc[val_idx]; ytest = y.iloc[test_idx]
    report = {'split': {k: {'admissions':len(idx), 'patients':int(data.iloc[idx].subject_id.nunique()),
                           'positives':int(y.iloc[idx].sum())} for k,idx in splits.items()},
              'feature_universe':len(X.columns),
              'warning':'Retrospective diagnosis cohort; observed ICU transfer != clinical need.'}
    (out/'split_summary.json').write_text(json.dumps(report,indent=2))
    # Save exact patient split IDs for reproducibility.
    split_rows = []
    for name,idx in splits.items():
        split_rows.extend({'subject_id':int(s),'hadm_id':int(h),'split':name}
                          for s,h in zip(data.iloc[idx].subject_id,data.iloc[idx].hadm_id))
    pd.DataFrame(split_rows).to_csv(out/'patient_splits.csv',index=False)
    all_configs = [('full',len(X.columns),X.columns.tolist())]
    for method in SELECTORS:
        print(f'Ranking features: {method}',flush=True)
        rank = ranking(Xtr,ytr,method,args.threads)
        rank.to_csv(out/f'ranking_{method}.csv',index=False)
        for n in COUNTS:
            all_configs.append((method,min(n,len(rank)),rank.feature.iloc[:n].tolist()))
    rows=[]
    for i,(method,n,features) in enumerate(all_configs,1):
        for kind in MODELS:
            print(f'[{i}/{len(all_configs)}] {method} / {n} / {kind}',flush=True)
            model=build_model(kind,args.threads)
            model.fit(Xtr[features],ytr)
            pv=model.predict_proba(X.iloc[val_idx][features])[:,1]
            threshold=tune_threshold(yval,pv)
            pt=model.predict_proba(X.iloc[test_idx][features])[:,1]
            scores=evaluate(ytest,pt,threshold)
            rows.append({'selector':method,'n_features':n,'architecture':kind,
                         'threshold':threshold,**scores})
            pd.DataFrame(rows).to_csv(out/'benchmark_results.csv',index=False)
    result=pd.DataFrame(rows).sort_values('auprc',ascending=False)
    result.to_csv(out/'benchmark_results_sorted.csv',index=False)
    print('\nTOP 10 BY TEST AUPRC (DESCRIPTIVE ONLY; DO NOT SELECT FINAL MODEL ON TEST):')
    print(result[['selector','n_features','architecture','auroc','auprc','balanced_accuracy','sensitivity']].head(10).to_string(index=False))
    print('\nSaved to',out)
    print('CAUTION: Choosing a winner from test results is exploratory; use nested validation or a new holdout for final confirmation.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--phase1',required=True)
    p.add_argument('--phase2',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--threads',type=int,default=4)
    main(p.parse_args())
