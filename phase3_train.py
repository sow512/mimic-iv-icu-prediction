#!/usr/bin/env python3
"""Phase 3: leakage-aware grouped train/validation/test evaluation. Research only."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier

SEED=42

def metrics(y, p):
    return {'n':int(len(y)), 'positives':int(y.sum()), 'prevalence':float(y.mean()),
            'auroc':float(roc_auc_score(y,p)), 'auprc':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p)), 'log_loss':float(log_loss(y,np.clip(p,1e-8,1-1e-8)))}

def main(args):
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    features=pd.read_parquet(Path(args.phase2)/'phase2_temporal_lab_features.parquet')
    base=pd.read_parquet(Path(args.phase1)/'phase1_landmark_predictors.parquet')
    labels=pd.read_parquet(Path(args.phase1)/'phase1_labels.parquet')
    for name,df in [('features',features),('baseline',base),('labels',labels)]:
        if df.hadm_id.isna().any() or df.hadm_id.duplicated().any(): raise ValueError(f'{name}: invalid or duplicated hadm_id')
    if 'subject_id' in features.columns:
        features=features.drop(columns='subject_id')
    if 'landmark_time' in features.columns:
        features=features.drop(columns='landmark_time')
    if 'subject_id' not in labels.columns: raise ValueError('Labels must contain subject_id')
    data=base.merge(features,on='hadm_id',how='left',validate='one_to_one')
    data=data.merge(labels[['subject_id','hadm_id','icu_transfer_24_72h']],on=['subject_id','hadm_id'],how='inner',validate='one_to_one')
    if len(data)!=len(base) or len(data)!=len(labels): raise ValueError('Row mismatch across phases')
    forbidden=['subject_id','hadm_id','landmark_time','icu_transfer_24_72h']
    X=data.drop(columns=forbidden)
    suspicious=[c for c in X if any(token in c.lower() for token in ['icu_transfer','icu_intime','dischtime','deathtime','horizon_end','label_status','hours_to_icu','first_icu'])]
    if suspicious: raise ValueError('Potential outcome leakage: '+str(suspicious))
    categorical=list(X.select_dtypes(include=['object','string','category']).columns)
    for col in categorical: X[col]=X[col].astype('string').fillna('__MISSING__').astype('category')
    numeric=[c for c in X if c not in categorical]
    X[numeric]=X[numeric].apply(pd.to_numeric,errors='raise').replace([np.inf,-np.inf],np.nan)
    all_null=[c for c in X if X[c].isna().all()]
    if all_null: X=X.drop(columns=all_null); categorical=[c for c in categorical if c not in all_null]
    y=data.icu_transfer_24_72h.astype(int)
    if set(y.unique())!={0,1}: raise ValueError('Expected binary outcome with both classes')
    groups=data.subject_id
    outer=GroupShuffleSplit(n_splits=1,test_size=.15,random_state=SEED)
    dev_idx,test_idx=next(outer.split(X,y,groups))
    inner=GroupShuffleSplit(n_splits=1,test_size=.15/.85,random_state=SEED+1)
    tr_sub,va_sub=next(inner.split(X.iloc[dev_idx],y.iloc[dev_idx],groups.iloc[dev_idx]))
    train_idx=dev_idx[tr_sub]; val_idx=dev_idx[va_sub]
    splits={'train':train_idx,'validation':val_idx,'test':test_idx}
    for a in splits:
        for b in splits:
            if a<b and set(groups.iloc[splits[a]]) & set(groups.iloc[splits[b]]): raise AssertionError('Patient leakage')
    for name,idx in splits.items():
        if y.iloc[idx].nunique()!=2: raise ValueError(f'{name} contains only one outcome class')
    Xtr,Xva,Xte=[X.iloc[splits[k]].copy() for k in ['train','validation','test']]
    ytr,yva,yte=[y.iloc[splits[k]] for k in ['train','validation','test']]
    # Missingness filter fit on training only. Native NaNs retained for both algorithms.
    keep=Xtr.columns[Xtr.notna().mean()>=.02].tolist()
    Xtr,Xva,Xte=[d[keep].copy() for d in (Xtr,Xva,Xte)]
    cats=[c for c in categorical if c in keep]
    lgb=LGBMClassifier(n_estimators=1200,learning_rate=.035,num_leaves=23,max_depth=-1,
        min_child_samples=70,colsample_bytree=.8,subsample=.8,subsample_freq=1,
        reg_lambda=5,random_state=SEED,n_jobs=args.threads,verbosity=-1)
    lgb.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric='average_precision',
            categorical_feature=cats,callbacks=[early_stopping(75,verbose=False),log_evaluation(0)])
    # CatBoost categorical inputs must be strings, not pandas category/NaN.
    def cb_frame(frame):
        z=frame.copy()
        for c in cats: z[c]=z[c].astype('string').fillna('__MISSING__').astype(str)
        return z
    cb=CatBoostClassifier(iterations=1400,depth=5,learning_rate=.04,l2_leaf_reg=6,
        loss_function='Logloss',eval_metric='PRAUC',random_seed=SEED,
        thread_count=args.threads,verbose=False)
    cb.fit(cb_frame(Xtr),ytr,cat_features=cats,eval_set=(cb_frame(Xva),yva),early_stopping_rounds=90,verbose=False)
    pv_l=lgb.predict_proba(Xva)[:,1]; pv_c=cb.predict_proba(cb_frame(Xva))[:,1]
    # Weight selected only on validation; no test-set model selection.
    weights=np.linspace(0,1,21)
    scored=[(float(average_precision_score(yva,w*pv_l+(1-w)*pv_c)),float(w)) for w in weights]
    best_ap,w=max(scored,key=lambda t:t[0])
    pt_l=lgb.predict_proba(Xte)[:,1]; pt_c=cb.predict_proba(cb_frame(Xte))[:,1]
    pred_val=w*pv_l+(1-w)*pv_c; pred_test=w*pt_l+(1-w)*pt_c
    report={'split':{k:{'admissions':len(idx),'patients':int(groups.iloc[idx].nunique()),'positives':int(y.iloc[idx].sum())} for k,idx in splits.items()},
      'predictors':len(keep),'categorical':cats,'dropped_high_missingness':len(X.columns)-len(keep),
      'best_lightgbm_iteration':int(lgb.best_iteration_), 'best_catboost_iteration':int(cb.best_iteration_),
      'blend_lightgbm_weight':w,'blend_catboost_weight':1-w,
      'validation':{'lightgbm':metrics(yva,pv_l),'catboost':metrics(yva,pv_c),'blend':metrics(yva,pred_val)},
      'test':{'lightgbm':metrics(yte,pt_l),'catboost':metrics(yte,pt_c),'blend':metrics(yte,pred_test)},
      'notes':['Retrospective renal ICD cohort may include diagnoses established after 24h.',
               'Observed ICU transfer is not equivalent to clinical ICU need.',
               'Test metrics should be inspected only once; subsequent model changes require a new holdout.']}
    (out/'phase3_report.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame({'subject_id':data.iloc[test_idx].subject_id.values,'hadm_id':data.iloc[test_idx].hadm_id.values,
                  'outcome':yte.values,'lightgbm_probability':pt_l,'catboost_probability':pt_c,
                  'blend_probability':pred_test}).to_parquet(out/'phase3_test_predictions.parquet',index=False)
    pd.DataFrame({'feature':keep,'lightgbm_importance':lgb.feature_importances_}).sort_values('lightgbm_importance',ascending=False).to_csv(out/'phase3_feature_importance.csv',index=False)
    import joblib
    joblib.dump({'lightgbm':lgb,'catboost':cb,'columns':keep,'categorical':cats,'blend_lightgbm_weight':w},out/'phase3_models.joblib')
    print(json.dumps(report,indent=2))
    print('Saved outputs to',out)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--phase1',required=True);p.add_argument('--phase2',required=True);p.add_argument('--output',required=True)
    p.add_argument('--threads',type=int,default=4)
    main(p.parse_args())
