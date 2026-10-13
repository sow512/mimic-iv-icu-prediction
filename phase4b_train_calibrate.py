#!/usr/bin/env python3
"""Research-only renal ICU prediction: patient-disjoint development and calibration.
Usage: python3 phase4b_train_calibrate.py --phase1 PATH --phase2 PATH --output PATH
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
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier

SEED = 20261013

def partition(indices, groups, fraction, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    a, b = next(splitter.split(np.zeros(len(indices)), groups=groups.iloc[indices]))
    return indices[a], indices[b]

def evaluate(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(np.sum(y)), prevalence=float(np.mean(y)),
                auroc=float(roc_auc_score(y,p)), auprc=float(average_precision_score(y,p)),
                brier=float(brier_score_loss(y,p)), log_loss=float(log_loss(y,p)))

def cb_frame(frame, cats):
    result = frame.copy()
    for col in cats:
        result[col] = result[col].astype('string').fillna('__MISSING__').astype(str)
    return result

def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(Path(args.phase1) / 'phase1_landmark_predictors.parquet')
    labs = pd.read_parquet(Path(args.phase2) / 'phase2_temporal_lab_features.parquet')
    labels = pd.read_parquet(Path(args.phase1) / 'phase1_labels.parquet')
    for name, table in [('base',base), ('labs',labs), ('labels',labels)]:
        if table.hadm_id.isna().any() or table.hadm_id.duplicated().any():
            raise ValueError(f'{name}: missing or duplicate hadm_id')
    labs = labs.drop(columns=['subject_id','landmark_time'], errors='ignore')
    # Base demographics are canonical: eliminate redundant columns from Phase 2.
    overlap = set(labs.columns).intersection(base.columns) - {'hadm_id'}
    labs = labs.drop(columns=sorted(overlap))
    data = base.merge(labs, on='hadm_id', how='left', validate='one_to_one')
    data = data.merge(labels[['subject_id','hadm_id','icu_transfer_24_72h']],
                      on=['subject_id','hadm_id'], how='inner', validate='one_to_one')
    if len(data) != len(base) or len(data) != len(labels):
        raise ValueError('Row count mismatch')
    target = 'icu_transfer_24_72h'
    forbidden = ['subject_id','hadm_id','landmark_time',target]
    X = data.drop(columns=forbidden)
    suspicious = [c for c in X if any(t in c.lower() for t in
                  ['icu_transfer','icu_intime','dischtime','deathtime','horizon_end',
                   'label_status','hours_to_icu','first_icu'])]
    if suspicious:
        raise ValueError('Possible outcome leakage: ' + str(suspicious))
    y = data[target].astype(int)
    groups = data.subject_id
    cats = list(X.select_dtypes(include=['object','string','category']).columns)
    for c in cats:
        X[c] = X[c].astype('string').fillna('__MISSING__').astype('category')
    numeric = [c for c in X if c not in cats]
    X[numeric] = X[numeric].apply(pd.to_numeric, errors='raise').replace([np.inf,-np.inf],np.nan)
    all_indices = np.arange(len(data))
    # Patient-disjoint target fractions: train 55%, early-stop 15%, blend 10%, calibration 10%, evaluation 10%.
    remaining, test = partition(all_indices, groups, .10, SEED)
    remaining, calibration = partition(remaining, groups, .10/.90, SEED+1)
    remaining, blend = partition(remaining, groups, .10/.80, SEED+2)
    train, early = partition(remaining, groups, .15/.70, SEED+3)
    splits = {'train':train,'early_stop':early,'blend':blend,'calibration':calibration,'evaluation':test}
    for a, ia in splits.items():
        if y.iloc[ia].nunique() != 2:
            raise ValueError(f'{a}: only one outcome class')
        for b, ib in splits.items():
            if a < b and set(groups.iloc[ia]) & set(groups.iloc[ib]):
                raise AssertionError(f'Patient overlap: {a}, {b}')
    keep = X.iloc[train].columns[X.iloc[train].notna().mean() >= .02].tolist()
    cats = [c for c in cats if c in keep]
    X = X[keep]
    xtr, xearly, xblend, xcal, xte = [X.iloc[splits[k]].copy() for k in splits]
    ytr, yearly, yblend, ycal, yte = [y.iloc[splits[k]] for k in splits]
    lgb = LGBMClassifier(n_estimators=1200,learning_rate=.035,num_leaves=23,
        min_child_samples=70,colsample_bytree=.8,subsample=.8,subsample_freq=1,
        reg_lambda=5,random_state=SEED,n_jobs=args.threads,verbosity=-1)
    lgb.fit(xtr,ytr,eval_set=[(xearly,yearly)],eval_metric='average_precision',
            categorical_feature=cats,callbacks=[early_stopping(75,verbose=False),log_evaluation(0)])
    cat = CatBoostClassifier(iterations=1400,depth=5,learning_rate=.04,l2_leaf_reg=6,
        loss_function='Logloss',eval_metric='PRAUC',random_seed=SEED,
        thread_count=args.threads,verbose=False)
    cat.fit(cb_frame(xtr,cats),ytr,cat_features=cats,
            eval_set=(cb_frame(xearly,cats),yearly),early_stopping_rounds=90,verbose=False)
    def both(frame):
        return (lgb.predict_proba(frame)[:,1],cat.predict_proba(cb_frame(frame,cats))[:,1])
    pb_l,pb_c = both(xblend)
    scores = [(average_precision_score(yblend,w*pb_l+(1-w)*pb_c),float(w))
              for w in np.linspace(0,1,21)]
    _, weight = max(scores,key=lambda item:item[0])
    pc_l,pc_c = both(xcal)
    pc = weight*pc_l + (1-weight)*pc_c
    # Platt scaling on log-odds; independent calibration set.
    logits = np.log(np.clip(pc,1e-6,1-1e-6) / (1-np.clip(pc,1e-6,1-1e-6)))
    sigmoid = LogisticRegression(C=1e3,max_iter=1000,random_state=SEED)
    sigmoid.fit(logits.reshape(-1,1),ycal)
    isotonic = IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip')
    isotonic.fit(pc,ycal)
    sigmoid_cal = sigmoid.predict_proba(logits.reshape(-1,1))[:,1]
    isotonic_cal = isotonic.predict(pc)
    # Select calibrator ONLY using calibration data. This comparison is optimistic;
    # independent evaluation split is used once below for reporting.
    candidates = {'uncalibrated':pc,'sigmoid':sigmoid_cal,'isotonic':isotonic_cal}
    selected = min(candidates,key=lambda k:log_loss(ycal,np.clip(candidates[k],1e-8,1-1e-8)))
    pt_l,pt_c = both(xte)
    pt = weight*pt_l+(1-weight)*pt_c
    pt_logit = np.log(np.clip(pt,1e-6,1-1e-6)/(1-np.clip(pt,1e-6,1-1e-6)))
    test_probs = {'uncalibrated':pt,'sigmoid':sigmoid.predict_proba(pt_logit.reshape(-1,1))[:,1],
                  'isotonic':isotonic.predict(pt)}
    report = {'split':{k:{'admissions':int(len(v)),'patients':int(groups.iloc[v].nunique()),
                           'positives':int(y.iloc[v].sum())} for k,v in splits.items()},
              'features':len(keep),'categorical':cats,
              'best_iterations':{'lightgbm':int(lgb.best_iteration_),'catboost':int(cat.best_iteration_)},
              'lightgbm_blend_weight':weight,'selected_calibration':selected,
              'calibration_metrics':{k:evaluate(ycal,v) for k,v in candidates.items()},
              'evaluation_metrics':{k:evaluate(yte,v) for k,v in test_probs.items()},
              'limitations':['Retrospective diagnosis-defined cohort',
                 'Observed ICU transfer does not equal clinical need',
                 'Previously inspected MIMIC-IV test outcomes; this is an internal reassessment, not an untouched external validation',
                 'Calibrator selection on calibration data may be optimistic; evaluation set was not used for selection']}
    (output/'phase4b_report.json').write_text(json.dumps(report,indent=2))
    joblib.dump({'lightgbm':lgb,'catboost':cat,'columns':keep,'categorical':cats,
                 'blend_lightgbm_weight':weight,'sigmoid':sigmoid,'isotonic':isotonic,
                 'selected_calibration':selected},output/'phase4b_models.joblib')
    pd.DataFrame({'subject_id':data.iloc[test].subject_id.values,
                  'hadm_id':data.iloc[test].hadm_id.values,'outcome':yte.values,
                  **{k+'_probability':v for k,v in test_probs.items()}}).to_parquet(
                      output/'phase4b_evaluation_predictions.parquet',index=False)
    fig,ax=plt.subplots(figsize=(7,6))
    ax.plot([0,1],[0,1],'--',label='Perfect calibration')
    for name,p in test_probs.items():
        d=pd.DataFrame({'p':p,'y':yte.to_numpy()})
        d['bin']=pd.qcut(d['p'].rank(method='first'),10,labels=False)
        bins=d.groupby('bin').agg(predicted=('p','mean'),observed=('y','mean'))
        ax.plot(bins.predicted,bins.observed,'o-',label=name)
    ax.set(xlabel='Mean predicted probability',ylabel='Observed ICU-transfer rate',
           title='Internal evaluation reliability (10 equal-count bins)')
    ax.legend();fig.tight_layout();fig.savefig(output/'phase4b_reliability.png',dpi=200)
    plt.close(fig)
    print(json.dumps(report,indent=2))
    print('Saved to:',output)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase1',required=True)
    parser.add_argument('--phase2',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--threads',type=int,default=4)
    main(parser.parse_args())
