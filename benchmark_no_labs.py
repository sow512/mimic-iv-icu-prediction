#!/usr/bin/env python3
"""Patient-disjoint ICU-transfer benchmark using Phase 18 + 18B + 18C (NO LABS).
Binary logistic regression, LightGBM, CatBoost, and discrete-time multinomial hazard SGD.
Train/early_stop/validation only. Reserved evaluation never accessed.
"""
import argparse, json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight

def scores(y,p):
    y=np.asarray(y,dtype=int);p=np.clip(np.asarray(p,dtype=float),0,1)
    return dict(n=len(y),positives=int(y.sum()),prevalence=float(y.mean()),auroc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,auprc=float(average_precision_score(y,p)) if y.sum() else None,brier=float(brier_score_loss(y,p)))

def load_feature(path,prefix):
    x=pd.read_parquet(path)
    if 'hadm_id' not in x:raise ValueError(f'{path}: missing hadm_id')
    if x.hadm_id.duplicated().any():raise ValueError(f'{path}: duplicate hadm_id')
    x=x.drop(columns=[c for c in x if c in ['subject_id','split'] or c.endswith('_first_event_time') or c.endswith('_last_event_time')])
    # Only numeric pre-landmark counts/clinical concept features; never use absolute timestamps.
    bad=[c for c in x if c!='hadm_id' and (pd.api.types.is_datetime64_any_dtype(x[c]) or not pd.api.types.is_numeric_dtype(x[c]))]
    if bad:print('Dropping nonnumeric columns',prefix,bad,flush=True);x=x.drop(columns=bad)
    return x

def cif(prob):
    # (N,K,4): remain, ICU, discharge, death; conditional interval hazards.
    surv=np.ones(len(prob));icu=np.zeros(len(prob))
    for k in range(prob.shape[1]):
        icu+=surv*prob[:,k,1];surv*=prob[:,k,0]
    return np.clip(icu,0,1)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase18',required=True);ap.add_argument('--phase18b',required=True);ap.add_argument('--phase18c',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--threads',type=int,default=2);ap.add_argument('--interval-hours',type=int,default=12)
    ap.add_argument('--hazard-epochs',type=int,default=3);ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args();out=Path(a.output).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    f=pd.read_parquet(Path(a.phase18)/'prelandmark_features.parquet')
    e=pd.read_parquet(Path(a.phase18)/'first_events.parquet')
    if f.hadm_id.duplicated().any() or e.hadm_id.duplicated().any():raise ValueError('Duplicate admission IDs')
    data=f.merge(e[['hadm_id','first_event','event_hours','icu_24_72']],on='hadm_id',validate='one_to_one')
    for group,base,names in [('18B',a.phase18b,['history','history_diagnoses','prescriptions','emar','orders','microbiology']),('18C',a.phase18c,['history_diagnoses','prescriptions','emar','orders'])]:
        for name in names:
            p=Path(base)/(name+'.parquet')
            if not p.exists():print('MISSING optional module:',p,flush=True);continue
            x=load_feature(p,group+'_'+name)
            rename={c:group.lower()+'_'+c for c in x if c!='hadm_id'}
            x=x.rename(columns=rename)
            marker=group.lower()+'_'+name+'_observed'
            x[marker]=1
            data=data.merge(x,on='hadm_id',how='left',validate='one_to_one')
            data[marker]=data[marker].fillna(0).astype('int8')
            print(f'Loaded {group}/{name}: {len(x):,} admissions, {len(rename)} numeric features',flush=True)
    if data.groupby('subject_id').split.nunique().max()!=1:raise ValueError('Patient split leakage')
    if data.hadm_id.duplicated().any():raise ValueError('Merge multiplied admissions')
    allowed={'train','early_stop','validation','evaluation'}
    if set(data.split.unique())!=allowed:raise ValueError('Unexpected split labels')
    # Explicitly drop held-out evaluation rows BEFORE all training and model selection.
    data=data[data.split!='evaluation'].copy()
    labels=['first_event','event_hours','icu_24_72','subject_id','hadm_id','split']
    cats=[c for c in ['gender','admission_type','admission_location'] if c in data]
    numeric=[c for c in data.columns if c not in labels+cats and pd.api.types.is_numeric_dtype(data[c])]
    for c in numeric:data[c]=pd.to_numeric(data[c],errors='coerce')
    # Missing values in event-count features imply no observed records, not confirmed clinical absence.
    prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value=0,add_indicator=False)),('scale',StandardScaler(with_mean=False))]),numeric),('cat',OneHotEncoder(handle_unknown='ignore'),cats)],sparse_threshold=1.0)
    train=data.split=='train';stop=data.split=='early_stop';val=data.split=='validation'
    Xtr=prep.fit_transform(data.loc[train,numeric+cats]);Xst=prep.transform(data.loc[stop,numeric+cats]);Xv=prep.transform(data.loc[val,numeric+cats]);
    Xtr=sparse.csr_matrix(Xtr);Xst=sparse.csr_matrix(Xst);Xv=sparse.csr_matrix(Xv)
    ytr=data.loc[train,'icu_24_72'].astype(int).to_numpy();ys=data.loc[stop,'icu_24_72'].astype(int).to_numpy();yv=data.loc[val,'icu_24_72'].astype(int).to_numpy()
    results={};preds=pd.DataFrame({'hadm_id':data.loc[val,'hadm_id'].to_numpy(),'outcome':yv})
    lr=LogisticRegression(max_iter=300,solver='liblinear',random_state=a.seed)
    lr.fit(Xtr,ytr);p=lr.predict_proba(Xv)[:,1];results['logistic']=scores(yv,p);preds['logistic']=p
    print('logistic:',results['logistic'],flush=True)
    try:
        import lightgbm as lgb
        model=lgb.LGBMClassifier(n_estimators=650,learning_rate=.035,num_leaves=24,max_depth=6,min_child_samples=90,reg_lambda=8,verbosity=-1,n_jobs=a.threads,random_state=a.seed)
        model.fit(Xtr,ytr,eval_set=[(Xst,ys)],eval_metric='average_precision',callbacks=[lgb.early_stopping(45,verbose=False)])
        p=model.predict_proba(Xv)[:,1];results['lightgbm']=scores(yv,p);preds['lightgbm']=p
        print('lightgbm:',results['lightgbm'],flush=True)
    except ImportError:print('LightGBM not installed; skipped',flush=True)
    try:
        from catboost import CatBoostClassifier
        cb=CatBoostClassifier(iterations=650,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function='Logloss',eval_metric='PRAUC',thread_count=a.threads,random_seed=a.seed,verbose=False)
        cb.fit(Xtr,ytr,eval_set=(Xst,ys),early_stopping_rounds=45,verbose=False)
        p=cb.predict_proba(Xv)[:,1];results['catboost']=scores(yv,p);preds['catboost']=p
        print('catboost:',results['catboost'],flush=True)
    except ImportError:print('CatBoost not installed; skipped',flush=True)
    # Conditional competing-risk multinomial logistic hazards via streaming SGD.
    # Events after horizon contribute 'remain' in all horizon intervals; early competing events end risk exposure.
    horizon=48.0;K=int(np.ceil(horizon/a.interval_hours));classes=np.array([0,1,2,3])
    if K<1:raise ValueError('Invalid interval')
    hazard=SGDClassifier(loss='log_loss',alpha=0.0001,random_state=a.seed,average=True)
    def hazard_rows(X,ev,times):
        # Generate rows per interval with a one-hot time indicator, without building full N*K dataset.
        for k in range(K):
            left=k*a.interval_hours;right=min((k+1)*a.interval_hours,horizon)
            at_risk=times>left
            idx=np.flatnonzero(at_risk)
            if not len(idx):continue
            y=np.zeros(len(idx),dtype=int)
            occurring=(times[idx]<=right)
            for event,label in [('icu',1),('discharge',2),('death',3)]:y[occurring & (ev[idx]==event)]=label
            time=sparse.csr_matrix((np.ones(len(idx)),(np.arange(len(idx)),np.full(len(idx),k))),shape=(len(idx),K))
            yield sparse.hstack([X[idx],time],format='csr'),y
    evtr=data.loc[train,'first_event'].astype(str).to_numpy();ttr=data.loc[train,'event_hours'].astype(float).to_numpy()
    # Crucial: event at exact boundary is assigned to preceding interval, never repeated.
    for epoch in range(a.hazard_epochs):
        for xb,yb in hazard_rows(Xtr,evtr,ttr):hazard.partial_fit(xb,yb,classes=classes)
        print(f'hazard SGD epoch {epoch+1}/{a.hazard_epochs}',flush=True)
    def predict_hazard(X):
        prob=np.zeros((X.shape[0],K,4),dtype=float)
        for k in range(K):
            time=sparse.csr_matrix((np.ones(X.shape[0]),(np.arange(X.shape[0]),np.full(X.shape[0],k))),shape=(X.shape[0],K))
            z=hazard.predict_proba(sparse.hstack([X,time],format='csr'))
            prob[:,k,hazard.classes_]=z
        return cif(prob)
    p=predict_hazard(Xv);results['discrete_competing_risk']=scores(yv,p);preds['discrete_competing_risk']=p
    print('discrete_competing_risk:',results['discrete_competing_risk'],flush=True)
    preds.to_parquet(out/'validation_predictions.parquet',index=False)
    (out/'validation_metrics.json').write_text(json.dumps({'models':results,'n_train':int(train.sum()),'n_early_stop':int(stop.sum()),'n_validation':int(val.sum()),'n_numeric_features':len(numeric),'n_categorical_features':len(cats),'evaluation_accessed':False,'warning':'No labs. Descriptive internal validation only; audit availability of medication and microbiology before clinical use.'},indent=2))
    print('Saved:',out,flush=True)
if __name__=='__main__':main()
