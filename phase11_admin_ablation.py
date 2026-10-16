#!/usr/bin/env python3
"""Phase 11: Remove admission_type/location from Phase 9C E_all models.

Reuses Phase 9C's frozen patient splits and compares on identical evaluation
admissions. Research only; prior evaluation outcomes have been inspected.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, brier_score_loss
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier
import joblib

SEED = 20261017
TARGET = 'icu_transfer_24_72h'
DROP = ['admission_type', 'admission_location']
FORBIDDEN = ('icu_transfer', 'icu_intime', 'first_icu', 'dischtime', 'deathtime', 'horizon_end', 'label_status', 'hours_to_icu')

def feature_table(path, base):
    df = pd.read_parquet(path)
    if 'hadm_id' not in df or df.hadm_id.isna().any() or df.hadm_id.duplicated().any():
        raise ValueError(f'Invalid hadm_id in {path}')
    if 'subject_id' in df:
        chk = base[['hadm_id','subject_id']].merge(df[['hadm_id','subject_id']], on='hadm_id', validate='one_to_one', suffixes=('_base','_new'))
        if not (chk.subject_id_base == chk.subject_id_new).all():
            raise ValueError(f'Subject mismatch: {path}')
    return df.drop(columns=['subject_id','landmark_time'], errors='ignore')

def prepare(data, columns, train_idx):
    x = data[columns].copy()
    cats = list(x.select_dtypes(include=['object','string','category']).columns)
    for c in cats:
        x[c] = x[c].astype('string').fillna('__MISSING__').astype('category')
    nums = [c for c in x if c not in cats]
    x[nums] = x[nums].apply(pd.to_numeric, errors='raise').replace([np.inf,-np.inf],np.nan)
    keep = x.iloc[train_idx].columns[x.iloc[train_idx].notna().mean() >= .02].tolist()
    return x[keep], [c for c in cats if c in keep]

def cat_frame(x, cats):
    z = x.copy()
    for c in cats:
        z[c] = z[c].astype('string').fillna('__MISSING__').astype(str)
    return z

def threshold_for(y,p):
    ts = np.unique(np.r_[np.linspace(.005,.5,200),.5])
    return max(((balanced_accuracy_score(y,p>=t),float(t)) for t in ts),key=lambda q:(q[0],q[1]))[1]

def evaluate(y,p,t):
    pred = (p>=t).astype(int)
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                sensitivity=float(recall_score(y,pred)),specificity=float(tn/(tn+fp)),
                precision=float(precision_score(y,pred,zero_division=0)),
                brier=float(brier_score_loss(y,p)),threshold=float(t),
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def main(args):
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    root=Path(args.phase9c)
    base=pd.read_parquet(Path(args.phase1)/'phase1_landmark_predictors.parquet')
    labels=pd.read_parquet(Path(args.phase1)/'phase1_labels.parquet')
    if base.hadm_id.duplicated().any() or labels.hadm_id.duplicated().any(): raise ValueError('Duplicate IDs')
    data=base.copy()
    for path in [Path(args.phase2)/'phase2_temporal_lab_features.parquet',
                 Path(args.phase9)/'phase9a_history_omr_features.parquet',
                 Path(args.phase9)/'phase9b_medication_comorbidity_features.parquet']:
        df=feature_table(path,base)
        df=df.drop(columns=[c for c in df if c!='hadm_id' and c in data.columns])
        bad=[c for c in df if c!='hadm_id' and any(k in c.lower() for k in FORBIDDEN)]
        if bad: raise ValueError(f'Suspicious columns: {bad}')
        data=data.merge(df,on='hadm_id',how='left',validate='one_to_one')
    data=data.merge(labels[['subject_id','hadm_id',TARGET]],on=['subject_id','hadm_id'],how='inner',validate='one_to_one')
    if len(data)!=len(base): raise ValueError('Row count changed')
    saved=pd.read_csv(root/'split_assignments.csv')
    if saved.hadm_id.duplicated().any() or len(saved)!=len(data): raise ValueError('Split file has duplicate/missing admissions')
    aligned=data[['subject_id','hadm_id']].merge(saved,on=['subject_id','hadm_id'],how='left',validate='one_to_one')
    if aligned.split.isna().any(): raise ValueError('Split IDs do not match')
    splits={name:np.flatnonzero(aligned.split.to_numpy()==name) for name in ['train','early_stop','blend','threshold','evaluation']}
    if any(len(idx)==0 for idx in splits.values()): raise ValueError('Missing split')
    if any(data.iloc[idx][TARGET].nunique()!=2 for idx in splits.values()): raise ValueError('Single-class split')
    original=pd.read_parquet(root/'E_all_evaluation_predictions.parquet')
    eval_ids=data.iloc[splits['evaluation']][['subject_id','hadm_id',TARGET]].rename(columns={TARGET:'outcome'})
    if len(original)!=len(eval_ids) or set(map(tuple,original[['subject_id','hadm_id']].to_numpy()))!=set(map(tuple,eval_ids[['subject_id','hadm_id']].to_numpy())):
        raise ValueError('Existing Phase 9C evaluation set does not match')
    check=eval_ids.merge(original[['subject_id','hadm_id','outcome']],on=['subject_id','hadm_id'],suffixes=('_new','_old'),validate='one_to_one')
    if not (check.outcome_new==check.outcome_old).all(): raise ValueError('Label mismatch')
    columns=[c for c in data if c not in ['subject_id','hadm_id','landmark_time',TARGET]+DROP]
    x,cats=prepare(data,columns,splits['train'])
    print(f'No-admin features: {len(x.columns)}; removed: {DROP}',flush=True)
    tr,es,bl,th,te=[x.iloc[splits[s]].copy() for s in splits]
    ytr,yes,ybl,yth,yte=[data.iloc[splits[s]][TARGET].astype(int).to_numpy() for s in splits]
    lgb=LGBMClassifier(n_estimators=1200,learning_rate=.035,num_leaves=23,min_child_samples=70,
        colsample_bytree=.8,subsample=.8,subsample_freq=1,reg_lambda=5,random_state=SEED,
        n_jobs=args.threads,verbosity=-1)
    lgb.fit(tr,ytr,eval_set=[(es,yes)],eval_metric='average_precision',categorical_feature=cats,
            callbacks=[early_stopping(75,verbose=False),log_evaluation(0)])
    cat=CatBoostClassifier(iterations=1400,depth=5,learning_rate=.04,l2_leaf_reg=6,
        loss_function='Logloss',eval_metric='PRAUC',random_seed=SEED,thread_count=args.threads,verbose=False)
    cat.fit(cat_frame(tr,cats),ytr,cat_features=cats,eval_set=(cat_frame(es,cats),yes),early_stopping_rounds=90,verbose=False)
    def predict(frame): return lgb.predict_proba(frame)[:,1],cat.predict_proba(cat_frame(frame,cats))[:,1]
    pb_l,pb_c=predict(bl)
    w=max(((float(average_precision_score(ybl,v*pb_l+(1-v)*pb_c)),float(v)) for v in np.linspace(0,1,21)),key=lambda t:t[0])[1]
    pt_l,pt_c=predict(th); pe_l,pe_c=predict(te)
    preds={'lightgbm':pe_l,'catboost':pe_c,'blend':w*pe_l+(1-w)*pe_c}
    th_preds={'lightgbm':pt_l,'catboost':pt_c,'blend':w*pt_l+(1-w)*pt_c}
    new=eval_ids.copy()
    for model,p in preds.items(): new[f'{model}_probability']=p
    new.to_parquet(out/'no_admin_evaluation_predictions.parquet',index=False)
    rows=[]
    old=original.set_index(['subject_id','hadm_id']).loc[pd.MultiIndex.from_frame(eval_ids[['subject_id','hadm_id']])]
    for model,p in preds.items():
        t=threshold_for(yth,th_preds[model]); score=evaluate(yte,p,t)
        old_p=old[f'{model}_probability'].to_numpy()
        rows.append(dict(model=model,variant='without_admin',n_features=len(x.columns),**score))
        rows.append(dict(model=model,variant='with_admin_existing',n_features=600,
            auroc=float(roc_auc_score(yte,old_p)),auprc=float(average_precision_score(yte,old_p))))
        print(f'{model}: without admin AUROC={score["auroc"]:.4f}, AUPRC={score["auprc"]:.4f}; '
              f'with admin AUROC={roc_auc_score(yte,old_p):.4f}, AUPRC={average_precision_score(yte,old_p):.4f}',flush=True)
    pd.DataFrame(rows).to_csv(out/'admin_ablation_comparison.csv',index=False)
    joblib.dump(dict(lightgbm=lgb,catboost=cat,columns=list(x.columns),categorical=cats,
                     blend_lightgbm_weight=w),out/'no_admin_models.joblib')
    (out/'report.json').write_text(json.dumps({'evaluation_admissions':len(yte),'positives':int(yte.sum()),
        'removed_columns':DROP,'no_admin_features':len(x.columns),'blend_lightgbm_weight':w,
        'note':'Existing with-admin test predictions reused. Differences are descriptive on a previously inspected internal evaluation set.'},indent=2))
    print('Saved:',out)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ['phase1','phase2','phase9','phase9c','output']: parser.add_argument('--'+arg,required=True)
    parser.add_argument('--threads',type=int,default=4)
    main(parser.parse_args())
