#!/usr/bin/env python3
"""Phase 18: reproducible hospital-wide ICU-transfer / multi-state dataset.
No model fitting. Patient-level output contains deidentified MIMIC IDs: keep locally.
Time-zero = hospital admission + landmark-hours. No post-landmark predictors.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

def load(root, rel, cols):
    p=root/rel
    if not p.exists(): raise FileNotFoundError(p)
    print('Reading',p,flush=True)
    return pd.read_csv(p,usecols=cols,low_memory=False)

def datetime(df, cols):
    for c in cols: df[c]=pd.to_datetime(df[c],errors='coerce')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--landmark-hours',type=float,default=24)
    ap.add_argument('--horizon-hours',type=float,default=72)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--labs',action='store_true',help='Optional costly first-24h lab extraction')
    ap.add_argument('--chunksize',type=int,default=1000000)
    a=ap.parse_args()
    if not 0<a.landmark_hours<a.horizon_hours: ap.error('Require 0 < landmark < horizon')
    base=Path(a.base).expanduser()
    root=base if (base/'hosp/admissions.csv').exists() else base/'physionet.org/files/mimiciv/3.1'
    out=Path(a.output).expanduser();out.mkdir(parents=True,exist_ok=True)
    p=load(root,'hosp/patients.csv',['subject_id','anchor_age','anchor_year','gender'])
    adm=load(root,'hosp/admissions.csv',['subject_id','hadm_id','admittime','dischtime','deathtime','admission_type','admission_location'])
    icu=load(root,'icu/icustays.csv',['subject_id','hadm_id','stay_id','intime','outtime'])
    datetime(adm,['admittime','dischtime','deathtime']);datetime(icu,['intime','outtime'])
    adm=adm.merge(p,on='subject_id',validate='many_to_one')
    adm['age']=adm.anchor_age+adm.admittime.dt.year-adm.anchor_year
    adm=adm.loc[(adm.age>=18)&(adm.dischtime>adm.admittime)&adm.admittime.notna()].copy()
    if adm.hadm_id.duplicated().any(): raise ValueError('Duplicate admission IDs')
    links=icu.merge(adm[['hadm_id','subject_id','admittime','dischtime']],on='hadm_id',suffixes=('_icu','_adm'),validate='many_to_one')
    valid=(links.subject_id_icu==links.subject_id_adm)&links.intime.notna()&(links.intime>=links.admittime)&(links.intime<=links.dischtime)
    invalid=int((~valid).sum());starts=links.loc[valid].sort_values(['hadm_id','intime','stay_id']).copy()
    first=starts.groupby('hadm_id').intime.min().rename('first_icu_intime')
    adm=adm.join(first,on='hadm_id')
    adm['landmark']=adm.admittime+pd.to_timedelta(a.landmark_hours,unit='h')
    eligible=(adm.dischtime>adm.landmark)&(adm.first_icu_intime.isna()|(adm.first_icu_intime>adm.landmark))
    cohort=adm.loc[eligible].copy()
    death=cohort.deathtime.where(cohort.deathtime.between(cohort.landmark,cohort.dischtime,inclusive='both'))
    times=pd.DataFrame({'icu':cohort.first_icu_intime,'death':death,'discharge':cohort.dischtime},index=cohort.index)
    t=times.min(axis=1)
    event=np.select([times.icu.eq(t),times.death.eq(t),times.discharge.eq(t)],['icu','death','discharge'],default='unknown')
    if (event=='unknown').any(): raise ValueError('Unknown event times')
    cohort['first_event']=event
    cohort['event_time']=t
    cohort['event_hours']=(t-cohort.landmark).dt.total_seconds()/3600
    cohort['icu_24_72']=((cohort.first_event=='icu')&(cohort.event_time<=cohort.admittime+pd.to_timedelta(a.horizon_hours,unit='h'))).astype('int8')
    # Stable patient-level split; no admission from the same patient crosses partitions.
    subjects=cohort.subject_id.drop_duplicates().sort_values().to_numpy()
    rng=np.random.default_rng(a.seed);subjects = rng.permutation(subjects)
    n=len(subjects);cut=np.floor(np.array([.7,.8,.9])*n).astype(int)
    splits={int(s):name for name,arr in zip(['train','early_stop','validation','evaluation'],np.split(subjects,cut)) for s in arr}
    cohort['split']=cohort.subject_id.map(splits)
    # Prior completed hospitalization counts, strictly before current admission, calculated from ALL valid adult admissions.
    prior=adm[['subject_id','hadm_id','admittime','dischtime']].sort_values(['subject_id','dischtime','hadm_id'])
    # Merge-asof grouped by subject, using completed discharges only. Count earlier discharges at current admittime.
    prior['prior_rank']=prior.groupby('subject_id').cumcount()+1
    left=cohort[['hadm_id','subject_id','admittime']].sort_values('admittime')
    right=prior[['subject_id','dischtime','prior_rank']].rename(columns={'dischtime':'prior_dischtime'}).sort_values('prior_dischtime')
    history=pd.merge_asof(left,right,left_on='admittime',right_on='prior_dischtime',by='subject_id',direction='backward',allow_exact_matches=False)
    cohort['prior_completed_admissions']=cohort.hadm_id.map(history.set_index('hadm_id').prior_rank).fillna(0).astype('int32')
    # ICU transitions: each valid ICU entry after landmark; only valid positive-duration exits for LOS.
    valid_stays=starts.loc[starts.hadm_id.isin(cohort.hadm_id)].copy()
    valid_stays=valid_stays.loc[valid_stays.intime>valid_stays.admittime+pd.to_timedelta(a.landmark_hours,unit='h')]
    valid_stays['valid_exit']=valid_stays.outtime.notna()&(valid_stays.outtime>valid_stays.intime)&(valid_stays.outtime<=valid_stays.dischtime+pd.Timedelta(minutes=1))
    valid_stays['los_hours']=((valid_stays.outtime-valid_stays.intime).dt.total_seconds()/3600).where(valid_stays.valid_exit)
    valid_stays['icu_sequence']=valid_stays.groupby('hadm_id').cumcount()+1
    transitions=valid_stays[['hadm_id','stay_id','intime','outtime','valid_exit','los_hours','icu_sequence']].copy()
    transitions['hours_from_landmark']=(transitions.intime-cohort.set_index('hadm_id').landmark.reindex(transitions.hadm_id).to_numpy()).dt.total_seconds()/3600
    # Keep predictor columns separate from outcome/event columns.
    features=cohort[['subject_id','hadm_id','split','age','gender','admission_type','admission_location','prior_completed_admissions']].copy()
    if a.labs:
        # Lab item selection is defined by the data dictionary, never the outcome.
        dictlab=load(root,'hosp/d_labitems.csv',['itemid','label','fluid'])
        targets={'creatinine','urea nitrogen','potassium','sodium','bicarbonate','chloride','glucose','lactate','hemoglobin','platelet count','white blood cells','anion gap','calcium, total','albumin'}
        selected=dictlab.loc[dictlab.label.str.lower().isin(targets)&dictlab.fluid.str.lower().isin(['blood','serum','plasma']),['itemid','label']].drop_duplicates('itemid')
        mapping=dict(zip(selected.itemid,selected.label.str.lower().str.replace(r'[^a-z0-9]+','_',regex=True)))
        print('Selected lab item IDs:',len(mapping),flush=True)
        key=cohort.set_index('hadm_id')[['admittime','landmark']]
        acc=[];read=0;kept=0
        for chunk in pd.read_csv(root/'hosp/labevents.csv',usecols=['hadm_id','itemid','charttime','storetime','valuenum'],chunksize=a.chunksize,low_memory=False):
            read+=len(chunk)
            chunk=chunk.loc[chunk.hadm_id.isin(key.index)&chunk.itemid.isin(mapping)&chunk.valuenum.notna()].copy()
            if len(chunk):
                chunk['charttime']=pd.to_datetime(chunk.charttime,errors='coerce');chunk['storetime']=pd.to_datetime(chunk.storetime,errors='coerce')
                chunk=chunk.join(key,on='hadm_id')
                chunk=chunk.loc[(chunk.charttime>=chunk.admittime)&(chunk.charttime<chunk.landmark)&chunk.storetime.notna()&(chunk.storetime<chunk.landmark)]
                if len(chunk):
                    chunk['lab']=chunk.itemid.map(mapping)
                    acc.append(chunk[['hadm_id','lab','valuenum']]);kept+=len(chunk)
            if read%10000000<a.chunksize: print('Lab rows scanned:',read,'retained:',kept,flush=True)
        if acc:
            all_labs=pd.concat(acc,ignore_index=True)
            aggregated=all_labs.groupby(['hadm_id','lab']).valuenum.agg(['count','mean','min','max']).unstack('lab')
            aggregated.columns=[f'lab_{lab}_{stat}' for stat,lab in aggregated.columns]
            features=features.join(aggregated,on='hadm_id')
            del all_labs,acc
    # Deidentified IDs remain sensitive governed research data; do not upload these files publicly.
    features.to_parquet(out/'prelandmark_features.parquet',index=False)
    cohort[['subject_id','hadm_id','split','landmark','first_event','event_time','event_hours','icu_24_72']].to_parquet(out/'first_events.parquet',index=False)
    transitions.to_parquet(out/'icu_stays_after_landmark.parquet',index=False)
    assert not cohort.groupby('subject_id').split.nunique().gt(1).any()
    counts=cohort.groupby(['split','first_event']).size().unstack(fill_value=0)
    counts.to_csv(out/'split_event_counts.csv')
    meta={'cohort_admissions':len(cohort),'unique_patients':cohort.subject_id.nunique(),'invalid_icu_link_rows':invalid,'icu_stays_after_landmark':len(transitions),'valid_icu_exit_rows':int(transitions.valid_exit.sum()),'feature_columns':list(features.columns),'landmark_hours':a.landmark_hours,'horizon_hours':a.horizon_hours,'seed':a.seed,'labs_extracted':a.labs,'source':'MIMIC-IV 3.1','warnings':['Retrospective cohort; no clinical deployment validation','Cross-patient shifted dates cannot reconstruct hospital occupancy','All events after landmark are targets, not predictors','Do not use discharge codes from current admission as prediction-time features','Hospital admission timestamps may be shifted; only within-patient timing is interpreted','Historical admission count excludes non-adult and invalid-timestamp admissions','ICU stays with uncertain outtime retained for entry but excluded from valid LOS']}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2,default=str))
    print('\nCOHORT:',len(cohort),'admissions;',cohort.subject_id.nunique(),'patients')
    print('Events:',cohort.first_event.value_counts().to_dict())
    print('ICU by horizon:',int(cohort.icu_24_72.sum()))
    print('Feature columns:',len(features.columns)-3)
    print('Saved to',out,flush=True)
if __name__=='__main__': main()
