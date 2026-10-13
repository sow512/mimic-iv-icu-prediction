#!/usr/bin/env python3
"""Phase 4: exploratory validation calibration diagnostics (no test access).
Reconstructs Phase 3 grouped split exactly; never fits a calibrator.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, log_loss)

SEED = 42

def calibration_bins(y, p, n_bins=10):
    df = pd.DataFrame({'y': np.asarray(y), 'p': np.asarray(p)})
    df['bin'] = pd.qcut(df.p.rank(method='first'), q=n_bins, labels=False)
    return df.groupby('bin', as_index=False).agg(n=('y','size'), observed=('y','mean'), predicted=('p','mean'), positives=('y','sum'))

def main(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(Path(args.phase2) / 'phase2_temporal_lab_features.parquet')
    base = pd.read_parquet(Path(args.phase1) / 'phase1_landmark_predictors.parquet')
    labels = pd.read_parquet(Path(args.phase1) / 'phase1_labels.parquet')
    for name, df in [('features',features),('baseline',base),('labels',labels)]:
        if df.hadm_id.isna().any() or df.hadm_id.duplicated().any():
            raise ValueError(f'{name}: invalid or duplicated hadm_id')
    features = features.drop(columns=['subject_id','landmark_time'], errors='ignore')
    data = base.merge(features, on='hadm_id', how='left', validate='one_to_one')
    data = data.merge(labels[['subject_id','hadm_id','icu_transfer_24_72h']],
                      on=['subject_id','hadm_id'], how='inner', validate='one_to_one')
    if len(data) != len(base) or len(data) != len(labels):
        raise ValueError('Row mismatch')
    X = data.drop(columns=['subject_id','hadm_id','landmark_time','icu_transfer_24_72h'])
    suspicious = [c for c in X if any(t in c.lower() for t in
       ['icu_transfer','icu_intime','dischtime','deathtime','horizon_end','label_status','hours_to_icu','first_icu'])]
    if suspicious: raise ValueError('Potential leakage: '+str(suspicious))
    categorical = list(X.select_dtypes(include=['object','string','category']).columns)
    for col in categorical:
        X[col] = X[col].astype('string').fillna('__MISSING__').astype('category')
    numeric = [c for c in X if c not in categorical]
    X[numeric] = X[numeric].apply(pd.to_numeric, errors='raise').replace([np.inf,-np.inf], np.nan)
    X = X.drop(columns=[c for c in X if X[c].isna().all()])
    y = data.icu_transfer_24_72h.astype(int)
    groups = data.subject_id
    outer = GroupShuffleSplit(n_splits=1, test_size=.15, random_state=SEED)
    dev_idx, test_idx = next(outer.split(X,y,groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=.15/.85, random_state=SEED+1)
    tr_sub, va_sub = next(inner.split(X.iloc[dev_idx], y.iloc[dev_idx], groups.iloc[dev_idx]))
    train_idx, val_idx = dev_idx[tr_sub], dev_idx[va_sub]
    if set(groups.iloc[train_idx]) & set(groups.iloc[val_idx]): raise AssertionError('Patient overlap')
    if set(groups.iloc[val_idx]) & set(groups.iloc[test_idx]): raise AssertionError('Patient overlap')
    if set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]): raise AssertionError('Patient overlap')
    bundle = joblib.load(Path(args.phase3) / 'phase3_models.joblib')
    cols = bundle['columns']; cats = bundle['categorical']
    missing = set(cols) - set(X.columns)
    if missing: raise ValueError('Missing saved model features: '+str(missing))
    Xv = X.iloc[val_idx][cols].copy()
    for c in cats:
        Xv[c] = Xv[c].astype('string').fillna('__MISSING__').astype('category')
    cb = Xv.copy()
    for c in cats: cb[c] = cb[c].astype('string').fillna('__MISSING__').astype(str)
    p_l = bundle['lightgbm'].predict_proba(Xv)[:,1]
    p_c = bundle['catboost'].predict_proba(cb)[:,1]
    w = float(bundle['blend_lightgbm_weight'])
    p = w*p_l+(1-w)*p_c
    yy = y.iloc[val_idx].to_numpy()
    old = json.loads((Path(args.phase3)/'phase3_report.json').read_text())
    if len(yy) != old['split']['validation']['admissions'] or int(yy.sum()) != old['split']['validation']['positives']:
        raise AssertionError('Reconstructed validation counts differ from Phase 3')
    ap = average_precision_score(yy,p)
    auc = roc_auc_score(yy,p)
    if not np.isclose(ap, old['validation']['blend']['auprc'], atol=1e-7) or not np.isclose(auc,old['validation']['blend']['auroc'],atol=1e-7):
        raise AssertionError('Predictions do not reproduce Phase 3 validation metrics; STOP')
    bins = calibration_bins(yy,p)
    bins.to_csv(out/'validation_reliability_bins.csv',index=False)
    pd.DataFrame({'subject_id':data.iloc[val_idx].subject_id.values,
                  'hadm_id':data.iloc[val_idx].hadm_id.values,
                  'outcome':yy,'blend_probability':p}).to_parquet(out/'validation_predictions.parquet',index=False)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot([0,1],[0,1],'--',label='Perfect calibration')
    ax.plot(bins.predicted,bins.observed,'o-',label='Validation (10 equal-count bins)')
    ax.set(xlabel='Mean predicted ICU-transfer probability',ylabel='Observed ICU-transfer rate',title='Exploratory validation reliability')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'validation_reliability.png',dpi=180);plt.close(fig)
    summary={'validation_admissions':int(len(yy)),'positives':int(yy.sum()),
             'prevalence':float(yy.mean()),'auroc':float(auc),'auprc':float(ap),
             'brier':float(brier_score_loss(yy,p)),
             'log_loss':float(log_loss(yy,np.clip(p,1e-8,1-1e-8))),
             'note':'Exploratory validation diagnostics only. Validation was used for early stopping and blend selection; no calibration model fitted. Test set not accessed.'}
    (out/'phase4_diagnostics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print('Saved to:',out)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for arg in ('phase1','phase2','phase3','output'):parser.add_argument('--'+arg,required=True)
    main(parser.parse_args())
