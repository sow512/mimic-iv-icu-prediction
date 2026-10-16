#!/usr/bin/env python3
"""Audit pre-landmark clinical data availability for MIMIC-IV 3.1 renal cohort.

Outputs aggregate coverage only (no patient-level data). Research use only.
Optional ED data requires a separately available MIMIC-IV-ED dataset.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def locate(base, module, name):
    roots = [base, base/'physionet.org'/'files'/'mimiciv'/'3.1', base/'mimiciv'/'3.1']
    for root in roots:
        for suffix in ('.csv.gz', '.csv'):
            p = root/module/(name+suffix)
            if p.is_file():
                return p
    return None


def read_cols(path):
    return list(pd.read_csv(path, nrows=0).columns)


def audit(base, phase1, output, chunk_size):
    output.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(phase1/'phase1_landmark_predictors.parquet')
    required = {'subject_id','hadm_id','landmark_time'}
    if not required.issubset(cohort):
        raise ValueError(f'Phase 1 predictors missing {required-set(cohort)}')
    if cohort.hadm_id.isna().any() or cohort.hadm_id.duplicated().any():
        raise ValueError('Expected unique, non-null hadm_id')
    cohort['landmark_time'] = pd.to_datetime(cohort.landmark_time, errors='raise')
    cohort['start_time'] = cohort.landmark_time-pd.Timedelta(hours=24)
    ids = cohort[['subject_id','hadm_id','start_time','landmark_time']]
    n = len(ids)
    rows = []
    manifest = {'cohort_admissions': n, 'unique_patients':int(ids.subject_id.nunique()), 'sources': {},
                'policy':'Availability only; strict event-time and recorded-time checks when present. No outcome labels loaded.'}

    def add(group, feature, observed, details):
        observed = int(observed)
        rows.append({'group':group,'feature':feature,'admissions_with_available_data':observed,
                     'eligible_admissions':n,'coverage_percent':round(100*observed/n,3), 'details':details})

    def source_status(key, path, note=''):
        manifest['sources'][key] = {'found':bool(path),'path':str(path) if path else None,'note':note}
        print(f'{key}: {path if path else "NOT FOUND"}',flush=True)

    # Baseline variables in Phase 1 are available by design; report coverage.
    for col in ['age','gender','admission_type','admission_location']:
        if col in cohort:
            add('baseline',col,cohort[col].notna().sum(),'Present in Phase 1 landmark predictor table')

    # Prior hospitalizations: count only admissions discharged before index admission time.
    path=locate(base,'hosp','admissions')
    source_status('hosp/admissions',path)
    if path:
        cols=['subject_id','hadm_id','admittime','dischtime']
        history=pd.read_csv(path,usecols=cols,parse_dates=['admittime','dischtime'])
        # Only previous admissions with completed discharge before the current index admission.
        # Avoid self-joins that generate large admission x admission intermediates.
        hist=history.loc[history.dischtime.notna(),['subject_id','hadm_id','dischtime']].copy()
        starts=ids[['subject_id','hadm_id','start_time']].copy()
        counts=np.zeros(n,dtype=np.int32)
        # Search sorted per-subject discharge times, excluding index admission by time boundary.
        from collections import defaultdict
        discharge_by_subject=defaultdict(list)
        for subject, disch in zip(hist.subject_id, hist.dischtime):
            discharge_by_subject[subject].append(disch.to_datetime64())
        for subject in discharge_by_subject:
            discharge_by_subject[subject]=np.sort(np.array(discharge_by_subject[subject],dtype='datetime64[ns]'))
        for i,(subject,start) in enumerate(zip(starts.subject_id,starts.start_time)):
            arr=discharge_by_subject.get(subject)
            if arr is not None:
                counts[i]=np.searchsorted(arr,start.to_datetime64(),side='left')
        add('history','prior_discharged_hospital_admission',np.count_nonzero(counts),
            'Prior discharge strictly before index admission; subject-level longitudinal history')
        del history,hist,discharge_by_subject

    # Prior ICU stay: require ICU outtime prior to index admission, not future/current stay.
    path=locate(base,'icu','icustays')
    source_status('icu/icustays',path)
    if path:
        icu=pd.read_csv(path,usecols=['subject_id','outtime'],parse_dates=['outtime'])
        icu=icu.loc[icu.outtime.notna()]
        by_subject={s:np.sort(g.outtime.to_numpy(dtype='datetime64[ns]')) for s,g in icu.groupby('subject_id')}
        prior=0
        for subject,start in zip(ids.subject_id,ids.start_time):
            arr=by_subject.get(subject)
            if arr is not None and np.searchsorted(arr,start.to_datetime64())>0:
                prior+=1
        add('history','prior_completed_icu_stay',prior,'ICU outtime strictly before index admission')
        del icu,by_subject

    # OMR is an outpatient measurement record: chartdate has day-level resolution.
    # To avoid ambiguous within-day ordering, only use chartdate strictly before index admission date.
    path=locate(base,'hosp','omr')
    source_status('hosp/omr',path,'Outpatient measures; only prior calendar days considered')
    if path:
        cols=read_cols(path)
        needed={'subject_id','chartdate','result_name'}
        if needed.issubset(cols):
            subj_dates=ids[['subject_id','start_time']].copy()
            subj_dates['admission_date']=subj_dates.start_time.dt.normalize()
            available={}
            use=['subject_id','chartdate','result_name']
            for chunk in pd.read_csv(path,usecols=use,chunksize=chunk_size):
                chunk=chunk.loc[chunk.subject_id.isin(subj_dates.subject_id)].copy()
                if chunk.empty:continue
                chunk['chartdate']=pd.to_datetime(chunk.chartdate,errors='coerce')
                chunk=chunk.loc[chunk.chartdate.notna()]
                for name,g in chunk.groupby('result_name',dropna=True):
                    k=str(name)
                    if k not in available:available[k]=[]
                    available[k].append(g[['subject_id','chartdate']])
            for name,parts in available.items():
                dates=pd.concat(parts,ignore_index=True).drop_duplicates()
                merged=subj_dates.merge(dates,on='subject_id',how='left')
                good=merged.loc[merged.chartdate.lt(merged.admission_date),'subject_id']
                # Coverage per admission, not per subject.
                eligible=merged.loc[merged.chartdate.lt(merged.admission_date),['subject_id','start_time']].drop_duplicates()
                add('outpatient_measurement',name,len(eligible),
                    'OMR chartdate strictly before index admission date; outpatient, not ward vital signs')
        else:
            manifest['sources']['hosp/omr']['note']='Unexpected schema: '+str(cols)

    # Medication prescriptions: order timing only; does not establish administration.
    path=locate(base,'hosp','prescriptions')
    source_status('hosp/prescriptions',path,'Medication orders, not proof of administration')
    if path:
        cols=read_cols(path)
        need={'subject_id','hadm_id','starttime'}
        if need.issubset(cols):
            # Use order starttime conservatively; no claim of clinician availability or administration.
            seen=set(); retained=0
            for i,chunk in enumerate(pd.read_csv(path,usecols=list(need),chunksize=chunk_size),1):
                chunk=chunk.loc[chunk.hadm_id.isin(ids.hadm_id)].copy()
                if chunk.empty:continue
                chunk=chunk.merge(ids,on=['subject_id','hadm_id'],how='inner',validate='many_to_one')
                chunk['starttime']=pd.to_datetime(chunk.starttime,errors='coerce')
                chunk=chunk.loc[chunk.starttime.ge(chunk.start_time)&chunk.starttime.lt(chunk.landmark_time)]
                retained+=len(chunk)
                seen.update(chunk.hadm_id.unique().tolist())
                if i%25==0:print(f'  prescriptions chunks {i}',flush=True)
            add('medication_orders','prescription_started_within_first_24h',len(seen),
                f'{retained} orders with starttime in [admission, landmark); not verified administration')
        else:
            manifest['sources']['hosp/prescriptions']['note']='Unexpected schema: '+str(cols)

    # Check other sources without claiming their data are pre-landmark or ward-level.
    for module,name in [('hosp','emar'),('hosp','pharmacy'),('hosp','microbiologyevents'),
                        ('hosp','procedures_icd'),('hosp','diagnoses_icd'),('icu','chartevents')]:
        p=locate(base,module,name)
        source_status(f'{module}/{name}',p)
        if p:
            manifest['sources'][f'{module}/{name}']['columns']=read_cols(p)
    manifest['warnings']=[
        'ICU chartevents are not general ward vital signs; do not use post-ICU measurements.',
        'Current-admission discharge diagnoses and procedures are not assumed known by hour 24.',
        'OMR is historical/outpatient data and uses date-level timestamps.',
        'Prescription starttime indicates an order window, not medication administration.',
        'MIMIC-IV core hospital tables do not guarantee complete ward vital signs.',
        'This script measures coverage, not predictive utility or causal effects.'
    ]
    result=pd.DataFrame(rows).sort_values(['group','coverage_percent','feature'],ascending=[True,False,True])
    result.to_csv(output/'phase8_feature_coverage.csv',index=False)
    (output/'phase8_source_manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    print('\nFEATURE COVERAGE (admissions):\n',result.to_string(index=False),flush=True)
    print('\nSaved:',output,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--phase1',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chunk-size',type=int,default=300000)
    args=p.parse_args()
    audit(args.base,args.phase1,args.output,args.chunk_size)
