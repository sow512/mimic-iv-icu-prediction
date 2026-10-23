#!/usr/bin/env python3
"""Phase 23: matched renal-history benchmark (binary trees vs competing-risk neural).
Exploratory internal validation; no held-out evaluation access.
Requires phase22b_labs_retrain.py in the same directory or --phase22b-script.
"""
import argparse, importlib.util, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib
import torch

def load_module(path):
    spec=importlib.util.spec_from_file_location('phase22b_module',str(path))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def score(y,p):
    y=np.asarray(y,int);p=np.clip(np.asarray(p,float),0,1)
    k=max(1,int(np.ceil(.1*len(y))));ix=np.argsort(-p,kind='stable')[:k]
    return dict(n=int(len(y)),events=int(y.sum()),prevalence=float(y.mean()),auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),brier=float(brier_score_loss(y,p)),recall_top10=float(y[ix].sum()/y.sum()),precision_top10=float(y[ix].mean()),expected_events=float(p.sum()))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('phase18','phase18b','phase18c','labs','staged-labs','output'):ap.add_argument('--'+key,required=True)
    ap.add_argument('--phase22b-script',default=None)
    ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--bootstraps',type=int,default=500)
    ap.add_argument('--hidden',type=int,default=128);ap.add_argument('--dropout',type=float,default=.2)
    ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--weight-decay',type=float,default=.001)
    ap.add_argument('--batch-size',type=int,default=1024);ap.add_argument('--patience',type=int,default=7)
    ap.add_argument('--horizon-hours',type=int,default=48);ap.add_argument('--interval-hours',type=int,default=12)
    a=ap.parse_args()
    if a.horizon_hours!=48 or a.interval_hours!=12:ap.error('This benchmark uses 48h after landmark, in 12h bins')
    script=Path(a.phase22b_script) if a.phase22b_script else Path(__file__).with_name('phase22b_labs_retrain.py')
    if not script.exists():ap.error(f'Phase22B source missing: {script}; use --phase22b-script')
    out=Path(a.output).expanduser();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    m=load_module(script)
    d=m.load_data(a)
    # Renal eligibility uses ONLY completed historical admissions, never current discharge ICD.
    renal_cols=[c for c in ('b_hist_dx_prior_renal','c_hist_renal_aki','c_hist_renal_ckd') if c in d]
    if not renal_cols:raise ValueError('No historical renal columns found. Check Phase18B/C inputs.')
    renal=np.zeros(len(d),dtype=bool)
    for c in renal_cols:renal|=d[c].fillna(0).to_numpy(dtype=float)>0
    d=d.loc[renal].copy()
    if d.empty:raise ValueError('No renal-history admissions')
    if d.groupby('subject_id').split.nunique().max()!=1:raise ValueError('Patient split violation')
    print('Renal definition:',renal_cols, 'cohort:',len(d),flush=True)
    cats=[c for c in ('gender','admission_type','admission_location') if c in d]
    excluded={'subject_id','hadm_id','split','icu_24_72','first_event','event_hours','landmark','landmark_time','admittime','dischtime','deathtime'}
    nums=[c for c in d if c not in excluded and c not in cats and pd.api.types.is_numeric_dtype(d[c])]
    allow={'b_hist_days_since_discharge'}
    bad=[c for c in nums if c not in allow and any(t in c.lower() for t in ('time','date','outcome','target','discharge','death'))]
    if bad:raise ValueError('Potential unsafe columns: '+str(bad))
    splits={s:d.split.eq(s).to_numpy() for s in ('train','early_stop','validation')}
    if any(splits[s].sum()==0 for s in splits):raise ValueError('Empty partition')
    if d.loc[splits['validation'],'icu_24_72'].nunique()<2:raise ValueError('Validation lacks both classes')
    for s in splits:print(s,int(splits[s].sum()),'events',int(d.loc[splits[s],'icu_24_72'].sum()),flush=True)
    for c in nums:d[c]=pd.to_numeric(d[c],errors='coerce').astype('float32')
    pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value=0)),('scale',StandardScaler())]),nums),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)],sparse_threshold=0)
    X={}
    for s in splits:
        x=d.loc[splits[s],nums+cats]
        X[s]=np.nan_to_num(np.asarray(pre.fit_transform(x) if s=='train' else pre.transform(x),dtype=np.float32),nan=0,posinf=0,neginf=0)
    joblib.dump({'preprocessor':pre,'numeric_columns':nums,'categorical_columns':cats},out/'preprocessing.joblib')
    y={s:d.loc[splits[s],'icu_24_72'].astype(int).to_numpy() for s in splits}
    val=d.loc[splits['validation']]
    preds=pd.DataFrame({'subject_id':val.subject_id.to_numpy(),'hadm_id':val.hadm_id.to_numpy(),'outcome':y['validation']})
    results={}
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    models={
      'lightgbm':lgb.LGBMClassifier(n_estimators=700,learning_rate=.035,num_leaves=24,max_depth=6,min_child_samples=90,reg_lambda=8,verbosity=-1,n_jobs=a.threads,random_state=a.seed),
      'catboost':CatBoostClassifier(iterations=700,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function='Logloss',eval_metric='PRAUC',thread_count=a.threads,random_seed=a.seed,verbose=False)
    }
    for name,model in models.items():
        print('TRAINING',name,flush=True);start=time.time()
        if name=='lightgbm':model.fit(X['train'],y['train'],eval_set=[(X['early_stop'],y['early_stop'])],eval_metric='average_precision',callbacks=[lgb.early_stopping(50,verbose=False)])
        else:model.fit(X['train'],y['train'],eval_set=(X['early_stop'],y['early_stop']),early_stopping_rounds=50,verbose=False)
        p=model.predict_proba(X['validation'])[:,1];preds[name]=p
        results[name]={**score(y['validation'],p),'seconds':round(time.time()-start,2)}
        joblib.dump(model,out/f'{name}.joblib');print(name,results[name],flush=True)
    Y={};M={}
    for s in splits:Y[s],M[s]=m.make_targets(d.loc[splits[s]],4,12)
    print('TRAINING neural competing-risk',flush=True);start=time.time()
    neural,info=m.train_model('neural',X['train'],Y['train'],M['train'],X['early_stop'],Y['early_stop'],M['early_stop'],a,out)
    cumulative,_=m.predict(neural,X['validation']);p=cumulative[:,-1]
    preds['neural_competing_risk']=p
    results['neural_competing_risk']={**score(y['validation'],p),'training':info,'seconds':round(time.time()-start,2)}
    print('neural_competing_risk',results['neural_competing_risk'],flush=True)
    preds.to_parquet(out/'validation_predictions.parquet',index=False)
    # Paired patient bootstrap: preserve repeated admissions for the same patient.
    groups=list(preds.groupby('subject_id',sort=False).indices.values());rng=np.random.default_rng(a.seed)
    names=list(results);boot={n:[] for n in names};pair={}
    for _ in range(a.bootstraps):
        idx=np.concatenate([groups[i] for i in rng.integers(0,len(groups),size=len(groups))]);yy=y['validation'][idx]
        if yy.min()==yy.max():continue
        for n in names:
            v=score(yy,preds[n].to_numpy()[idx]);boot[n].append({k:v[k] for k in ('auroc','auprc','brier','recall_top10')})
    ci=lambda vals:[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
    for n in names:
        results[n]['patient_bootstrap_95ci']={k:ci([v[k] for v in boot[n]]) for k in ('auroc','auprc','brier','recall_top10')}
    for n in ('lightgbm','catboost'):
        pair['neural_minus_'+n]={k:ci([v[k]-u[k] for v,u in zip(boot['neural_competing_risk'],boot[n])]) for k in ('auroc','auprc','brier','recall_top10')}
    report={'scope':'renal-history development validation only','landmark_hours':24,'horizon_hours_after_landmark':48,'renal_eligibility_columns':renal_cols,'feature_policy':'Phase22B conservative + labs; excludes current-admission medication, orders, microbiology','n_features_encoded':int(X['train'].shape[1]),'n_numeric':len(nums),'n_categorical':len(cats),'n_unique_validation_patients':int(val.subject_id.nunique()),'models':results,'paired_patient_bootstrap_ci':pair,'bootstrap_unit':'patient','limitations':['Binary LightGBM/CatBoost estimate fixed-horizon event probability; neural estimates competing-risk CIF. Compare prediction quality, not identical likelihoods.','Historical diagnosis code availability and lab cache provenance not independently certified.','Same validation set reused in prior exploratory experiments.','This is a renal-history cohort, not necessarily active renal dysfunction.','No held-out evaluation split accessed.']}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print('COMPLETE:',out)
if __name__=='__main__':main()
