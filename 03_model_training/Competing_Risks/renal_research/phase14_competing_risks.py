#!/usr/bin/env python3
"""Phase 14: landmarked 24-72h discrete-time competing-risk ICU prediction.
Research only. ICU=1, discharge=2, death=3, no event=0.
Full mode uses baseline-only covariates for ALL landmark-eligible renal admissions.
Enriched mode uses the already selected Phase-1 cohort and Phase-2/9 features;
its results are NOT a full competing-risk cohort estimate.
"""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier

SEED=20261022
EVENTS=['none','icu','discharge','death']

def locate(base, folder, stem):
    hits=list(Path(base).glob(f'physionet.org/files/mimiciv/*/{folder}/{stem}.csv*'))
    if len(hits)!=1: raise FileNotFoundError(f'Expected one {folder}/{stem}.csv[.gz] under {base}; found {hits}')
    return hits[0]

def read(base,folder,stem,cols):
    p=locate(base,folder,stem);print('Reading',p,flush=True)
    return pd.read_csv(p,usecols=cols,low_memory=False)

def hashed_split(ids):
    # deterministic subject-level assignment, independent of row order
    return np.array([('train' if k<65 else 'stop' if k<77 else 'validation' if k<88 else 'evaluation') for k in
        (int.from_bytes(hashlib.sha256(f'phase14-{int(s)}'.encode()).digest()[:8],'big')%100 for s in ids)])

def full_cohort(args):
    pat=read(args.base,'hosp','patients',['subject_id','gender','anchor_age','anchor_year'])
    adm=read(args.base,'hosp','admissions',['subject_id','hadm_id','admittime','dischtime','deathtime','admission_type','admission_location'])
    dx=read(args.base,'hosp','diagnoses_icd',['hadm_id','icd_code','icd_version'])
    icu=read(args.base,'icu','icustays',['subject_id','hadm_id','intime','outtime'])
    dx['icd_code']=dx.icd_code.astype(str).str.upper().str.replace('.','',regex=False)
    version=pd.to_numeric(dx.icd_version,errors='coerce')
    renal=((version==9)&dx.icd_code.str.match(r'^(584|585)'))|((version==10)&dx.icd_code.str.match(r'^(N17|N18)'))
    renal_hadm=set(dx.loc[renal,'hadm_id'].dropna().astype('int64'))
    adm=adm[adm.hadm_id.isin(renal_hadm)].copy()
    adm=adm.merge(pat,on='subject_id',validate='many_to_one')
    for c in ['admittime','dischtime','deathtime']:adm[c]=pd.to_datetime(adm[c],errors='coerce')
    adm=adm[adm.admittime.notna() & adm.dischtime.notna() & (adm.dischtime>adm.admittime)].copy()
    adm['age']=adm.anchor_age + (adm.admittime.dt.year-adm.anchor_year)
    adm=adm[adm.age>=18].copy()
    adm['landmark_time']=adm.admittime+pd.Timedelta(hours=24)
    adm=adm[(adm.dischtime>adm.landmark_time)&(adm.deathtime.isna()|(adm.deathtime>adm.landmark_time))].copy()
    for c in ['intime','outtime']:icu[c]=pd.to_datetime(icu[c],errors='coerce')
    icu=icu[icu.hadm_id.isin(adm.hadm_id)].copy()
    # Fail closed for invalid ICU linkage/timestamps, rather than silently using bad times.
    links=icu.merge(adm[['hadm_id','subject_id','admittime','dischtime']],on='hadm_id',suffixes=('_icu','_adm'),validate='many_to_one')
    bad=links.intime.isna()|links.outtime.isna()|(links.intime>=links.outtime)|(links.subject_id_icu!=links.subject_id_adm)|\
        (links.intime<links.admittime)|(links.intime>links.dischtime)|(links.outtime>links.dischtime+pd.Timedelta(minutes=1))
    invalid=set(links.loc[bad,'hadm_id']);print('Excluded malformed ICU-linked admissions:',len(invalid))
    adm=adm[~adm.hadm_id.isin(invalid)].copy()
    first=links.loc[~bad].groupby('hadm_id').intime.min().rename('icu_time')
    adm=adm.join(first,on='hadm_id')
    adm=adm[adm.icu_time.isna()|(adm.icu_time>adm.landmark_time)].copy()
    return adm[['subject_id','hadm_id','landmark_time','admittime','dischtime','deathtime','icu_time','age','gender','admission_type','admission_location']]

def restricted_cohort(args):
    p=Path(args.phase1)
    d=pd.read_parquet(p/'phase1_cohort_audit.parquet')
    required=['subject_id','hadm_id','landmark_time','admittime','dischtime','deathtime','first_icu_intime','age','gender','admission_type','admission_location']
    if not set(required).issubset(d):raise ValueError('Phase 1 audit missing: '+str(set(required)-set(d)))
    d=d[required].rename(columns={'first_icu_intime':'icu_time'}).copy()
    for c in ['landmark_time','admittime','dischtime','deathtime','icu_time']:d[c]=pd.to_datetime(d[c],errors='coerce')
    for path in [Path(args.phase2)/'phase2_temporal_lab_features.parquet',Path(args.phase9)/'phase9a_history_omr_features.parquet',Path(args.phase9)/'phase9b_medication_comorbidity_features.parquet']:
        f=pd.read_parquet(path)
        if f.hadm_id.duplicated().any():raise ValueError('Duplicate hadm_id in '+str(path))
        f=f.drop(columns=['subject_id','landmark_time'],errors='ignore')
        dup=[c for c in f if c!='hadm_id' and c in d]
        if dup:raise ValueError('Duplicate feature names '+str(dup[:10]))
        d=d.merge(f,on='hadm_id',how='left',validate='one_to_one')
    return d

def event_table(d,horizon):
    d=d.copy();land=d.landmark_time
    # Death at/after discharge timestamp is not necessarily an in-hospital death.
    death=d.deathtime.where(d.deathtime.notna()&(d.deathtime<=d.dischtime))
    # Death takes precedence over discharge on exact ties; ICU takes precedence over either on ties.
    dt=pd.DataFrame({'icu':d.icu_time,'death':death,'discharge':d.dischtime})
    earliest=dt.min(axis=1)
    code=np.select([d.icu_time.eq(earliest),death.eq(earliest),d.dischtime.eq(earliest)],[1,3,2],default=0)
    end=land+pd.Timedelta(hours=horizon)
    observed=(earliest<=end)
    d['event_code']=np.where(observed,code,0).astype('int8')
    d['duration_hours']=((earliest.where(observed,end)-land).dt.total_seconds()/3600).clip(lower=0,upper=horizon)
    if (d.duration_hours<=0).any():raise ValueError('Event at/before landmark: inspect eligibility')
    if d.hadm_id.duplicated().any() or d.subject_id.isna().any():raise ValueError('Invalid cohort identifiers')
    return d

def make_long(d,interval):
    # Each row corresponds to one at-risk interval. Event time t in ((j-1)*dt, j*dt].
    chunks=[];horizon=48
    for j in range(1,horizon//interval+1):
        left=(j-1)*interval;right=j*interval
        at_risk=d.duration_hours.to_numpy()>left
        rows=np.flatnonzero(at_risk)
        labels=np.where((d.duration_hours.to_numpy()[rows]<=right)&(d.event_code.to_numpy()[rows]>0),d.event_code.to_numpy()[rows],0)
        chunks.append((rows,np.full(len(rows),j-1,dtype='int16'),labels.astype('int8')))
    return tuple(np.concatenate([c[k] for c in chunks]) for k in range(3))

def feature_frame(d,mode):
    forbidden={'subject_id','hadm_id','admittime','landmark_time','dischtime','deathtime','icu_time','duration_hours','event_code'}
    cols=[c for c in d if c not in forbidden]
    if mode=='full':cols=['age','gender','admission_type','admission_location']
    leak=[c for c in cols if any(x in c.lower() for x in ['icu_intime','icu_transfer','first_icu','horizon_end','label_status','dischtime','deathtime'])]
    if leak:raise ValueError('Potential leakage: '+str(leak))
    x=d[cols].copy();cats=list(x.select_dtypes(include=['object','string','category']).columns)
    for c in cats:x[c]=x[c].astype('string').fillna('__MISSING__').astype('category')
    for c in x.columns.difference(cats):x[c]=pd.to_numeric(x[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    return x,cats

def frame_long(x,rows,bins,cats,arch):
    z=x.iloc[rows].copy().reset_index(drop=True)
    z['time_bin']=bins.astype('int16')
    if arch=='catboost':
        for c in cats:z[c]=z[c].astype('string').fillna('__MISSING__').astype(str)
        z=z.fillna({c:'__MISSING__' for c in cats})
    return z

def predict_cif(model,x,cats,arch,interval):
    # 4-class conditional hazard probabilities; multiply survival through each interval.
    n=len(x);surv=np.ones(n);cif=np.zeros((n,3));curves=[]
    for j in range(48//interval):
        z=frame_long(x,np.arange(n),np.full(n,j),cats,arch)
        p=model.predict_proba(z)
        ordered=np.zeros((n,4))
        for i,k in enumerate(model.classes_):ordered[:,int(k)]=p[:,i]
        hazards=np.clip(ordered[:,1:],0,1)
        cif+=surv[:,None]*hazards
        surv*=np.clip(ordered[:,0],0,1)
        curves.append(cif[:,0].copy())
    return np.column_stack(curves),cif,surv

def scores(y,p):
    if len(np.unique(y))<2:return {'auroc':None,'auprc':None}
    return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p)), 'prevalence':float(y.mean()),'n':int(len(y)),
            'positives':int(y.sum())}

def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.interval<=0 or 48%a.interval:raise ValueError('interval must divide 48')
    if a.mode=='full':d=full_cohort(a)
    else:
        if not all([a.phase1,a.phase2,a.phase9]):raise ValueError('enriched mode needs --phase1 --phase2 --phase9')
        d=restricted_cohort(a)
    d=event_table(d,48)
    split=hashed_split(d.subject_id)
    if d.assign(split=split).groupby('subject_id').split.nunique().max()!=1:raise AssertionError('Patient leakage')
    x,cats=feature_frame(d,a.mode)
    train=np.flatnonzero(split=='train');stop=np.flatnonzero(split=='stop');val=np.flatnonzero(split=='validation');ev=np.flatnonzero(split=='evaluation')
    # train-only missingness filtering
    keep=x.columns[x.iloc[train].notna().mean()>=.02].tolist();x=x[keep];cats=[c for c in cats if c in keep]
    rows,bins,y=make_long(d,a.interval)
    partition=split[rows]
    if len(np.unique(y[partition=='train']))!=4:raise ValueError('Not all competing event types in training data')
    print('Mode:',a.mode,'Admissions:',len(d),'Patients:',d.subject_id.nunique(),'Features:',x.shape[1],flush=True)
    print('Event counts by 72h:',dict(zip(EVENTS,np.bincount(d.event_code,minlength=4))),flush=True)
    print('Split admissions:',{k:int((split==k).sum()) for k in ['train','stop','validation','evaluation']},flush=True)
    results={};models={}
    for arch in ['lightgbm','catboost']:
        print('Training',arch,flush=True)
        tr=partition=='train';st=partition=='stop'
        xtr=frame_long(x,rows[tr],bins[tr],cats,arch);xst=frame_long(x,rows[st],bins[st],cats,arch)
        if arch=='lightgbm':
            m=LGBMClassifier(objective='multiclass',num_class=4,n_estimators=1200,learning_rate=.035,
                num_leaves=25,min_child_samples=80,colsample_bytree=.85,reg_lambda=6,
                random_state=SEED,n_jobs=a.threads,verbosity=-1)
            m.fit(xtr,y[tr],eval_set=[(xst,y[st])],eval_metric='multi_logloss',categorical_feature=cats,
                callbacks=[early_stopping(65,verbose=False)])
        else:
            m=CatBoostClassifier(loss_function='MultiClass',eval_metric='MultiClass',iterations=1400,
                depth=5,learning_rate=.04,l2_leaf_reg=7,random_seed=SEED,thread_count=a.threads,
                verbose=False,allow_writing_files=False)
            m.fit(xtr,y[tr],cat_features=cats,eval_set=(xst,y[st]),early_stopping_rounds=80,verbose=False)
        models[arch]=m
        curves,cif,surv=predict_cif(m,x.iloc[val],cats,arch,a.interval)
        yy=(d.event_code.iloc[val].to_numpy()==1).astype(int)
        results[arch]={'validation_icu_by_72h':scores(yy,curves[:,-1]),
                       'best_iteration':int(m.best_iteration_ if arch=='lightgbm' else m.best_iteration_)}
        print(arch,results[arch],flush=True)
        pd.DataFrame({'subject_id':d.subject_id.iloc[val].to_numpy(),'hadm_id':d.hadm_id.iloc[val].to_numpy(),
            'icu_by_72h':yy,'icu_cif':curves[:,-1],'discharge_cif':cif[:,1],
            'death_cif':cif[:,2],'event_code':d.event_code.iloc[val].to_numpy()}).to_parquet(out/f'{arch}_validation_predictions.parquet',index=False)
    # Fixed 50:50 blend for descriptive validation; no test-driven selection.
    l=pd.read_parquet(out/'lightgbm_validation_predictions.parquet');c=pd.read_parquet(out/'catboost_validation_predictions.parquet')
    if not l[['subject_id','hadm_id']].equals(c[['subject_id','hadm_id']]):raise AssertionError('Prediction alignment')
    blend=.5*l.icu_cif.to_numpy()+.5*c.icu_cif.to_numpy()
    results['blend']={'validation_icu_by_72h':scores(l.icu_by_72h.to_numpy(),blend)}
    pd.DataFrame({'subject_id':l.subject_id,'hadm_id':l.hadm_id,'outcome':l.icu_by_72h,'blend_icu_cif':blend}).to_parquet(out/'blend_validation_predictions.parquet',index=False)
    manifest={'mode':a.mode,'interval_hours':a.interval,'prediction_landmark_hours':24,'prediction_horizon_hours':72,
      'admissions':len(d),'patients':int(d.subject_id.nunique()),'features':list(x.columns),'categorical':cats,
      'events_by_72h':dict(zip(EVENTS,map(int,np.bincount(d.event_code,minlength=4)))),
      'validation':results,'warning':('FULL risk-set with baseline predictors only; diagnoses retrospective; not comparable to 600-feature model' if a.mode=='full' else
       'SELECTED 59,949-admission cohort: discharge/death before 72h previously excluded; NOT full competing-risk evaluation'),
      'evaluation_status':'NOT ACCESSED' if not a.evaluate else 'EXPLORATORY: PREVIOUSLY INSPECTED DATA'}
    if a.evaluate:
        if (out/'evaluation_complete.json').exists():raise RuntimeError('Evaluation already completed in this output directory')
        yev=(d.event_code.iloc[ev].to_numpy()==1).astype(int)
        preds={}
        for arch,m in models.items():
            curve,cif,surv=predict_cif(m,x.iloc[ev],cats,arch,a.interval)
            preds[arch]=curve[:,-1]
        preds['blend']=(preds['lightgbm']+preds['catboost'])/2
        manifest['evaluation']={k:scores(yev,v) for k,v in preds.items()}
        pd.DataFrame({'subject_id':d.subject_id.iloc[ev].to_numpy(),'hadm_id':d.hadm_id.iloc[ev].to_numpy(),
          'outcome':yev,**{k+'_icu_cif':v for k,v in preds.items()}}).to_parquet(out/'evaluation_predictions.parquet',index=False)
        (out/'evaluation_complete.json').write_text(json.dumps({'evaluated':True,'warning':'Previously inspected MIMIC-IV; exploratory only'},indent=2))
    joblib.dump({'models':models,'columns':list(x.columns),'categorical':cats,'interval_hours':a.interval,'mode':a.mode},out/'phase14_models.joblib')
    (out/'phase14_report.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'mode':a.mode,'validation':results,'evaluation_accessed':a.evaluate},indent=2),flush=True)
    print('Saved:',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True);p.add_argument('--phase1');p.add_argument('--phase2');p.add_argument('--phase9')
    p.add_argument('--mode',choices=['full','enriched'],default='full')
    p.add_argument('--output',required=True);p.add_argument('--interval',type=int,default=12)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--evaluate',action='store_true')
    main(p.parse_args())
