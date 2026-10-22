#!/usr/bin/env python3
"""Phase 17: retrospective hospital patient-flow audit for MIMIC-IV 3.1.
No model training. Aggregate outputs only; patient-level rows are not exported.
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

HOURS = 3600.0

def read_csv(root, rel, cols):
    path = root / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    print(f"Reading {path}", flush=True)
    return pd.read_csv(path, usecols=cols, low_memory=False)

def dt(df, cols):
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def qstats(x):
    a = pd.to_numeric(pd.Series(x), errors='coerce').dropna()
    if a.empty: return {'n':0,'median':None,'p10':None,'p25':None,'p75':None,'p90':None,'mean':None}
    return {k:float(v) for k,v in {'n':len(a),'median':a.median(),'p10':a.quantile(.1),'p25':a.quantile(.25),'p75':a.quantile(.75),'p90':a.quantile(.9),'mean':a.mean()}.items()}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', required=True, help='MIMIC_IV directory or mimiciv/3.1 directory')
    ap.add_argument('--output',required=True)
    ap.add_argument('--landmark-hours',type=float,default=24)
    ap.add_argument('--horizon-hours',type=float,default=72,help='Hours after hospital admission, not after landmark')
    ap.add_argument('--adult-age',type=int,default=18)
    a=ap.parse_args()
    if a.horizon_hours <= a.landmark_hours: ap.error('horizon-hours must exceed landmark-hours')
    base=Path(a.base).expanduser()
    if (base/'hosp'/'admissions.csv').exists(): root=base
    elif (base/'physionet.org/files/mimiciv/3.1/hosp/admissions.csv').exists(): root=base/'physionet.org/files/mimiciv/3.1'
    else: raise FileNotFoundError('Cannot locate hosp/admissions.csv beneath --base')
    out=Path(a.output).expanduser();out.mkdir(parents=True,exist_ok=True)
    patients=read_csv(root,'hosp/patients.csv',['subject_id','anchor_age','anchor_year','gender'])
    admissions=read_csv(root,'hosp/admissions.csv',['subject_id','hadm_id','admittime','dischtime','deathtime','admission_type','admission_location','hospital_expire_flag'])
    icu=read_csv(root,'icu/icustays.csv',['subject_id','hadm_id','stay_id','intime','outtime','los'])
    diagnoses=read_csv(root,'hosp/diagnoses_icd.csv',['subject_id','hadm_id','icd_code','icd_version'])
    dt(admissions,['admittime','dischtime','deathtime']);dt(icu,['intime','outtime'])
    admissions=admissions.merge(patients,on='subject_id',how='left',validate='many_to_one')
    admissions['age']=admissions.anchor_age+(admissions.admittime.dt.year-admissions.anchor_year)
    # Age >= 90 is top-coded in MIMIC; adult classification remains valid.
    valid_adm=admissions.admittime.notna() & admissions.dischtime.notna() & (admissions.dischtime>admissions.admittime)
    admissions=admissions.loc[valid_adm & (admissions.age>=a.adult_age)].copy()
    admissions=admissions.drop_duplicates('hadm_id')
    print('Adult admissions with valid hospital intervals:',len(admissions),flush=True)
    
    # Validate ICU stay links, but do not exclude entire admissions for invalid ICU rows.
    links=icu.merge(admissions[['hadm_id','subject_id','admittime','dischtime']],on='hadm_id',how='inner',suffixes=('_icu','_adm'),validate='many_to_one')
    reasons={
      'missing_icu_time':links.intime.isna()|links.outtime.isna(),
      'nonpositive_icu_duration':links.intime>=links.outtime,
      'subject_mismatch':links.subject_id_icu!=links.subject_id_adm,
      'icu_start_before_admission':links.intime<links.admittime,
      'icu_start_after_discharge':links.intime>links.dischtime,
      'icu_end_after_discharge_plus_1min':links.outtime>links.dischtime+pd.Timedelta(minutes=1),
    }
    flags=pd.DataFrame(reasons,index=links.index)
    flags['any']=flags.any(axis=1)
    audit=[]
    for k in reasons:
        m=flags[k]
        audit.append({'reason':k,'icu_stays':int(m.sum()),'admissions':int(links.loc[m,'hadm_id'].nunique())})
    audit.append({'reason':'any_flag','icu_stays':int(flags['any'].sum()),'admissions':int(links.loc[flags['any'],'hadm_id'].nunique())})
    pd.DataFrame(audit).to_csv(out/'icu_timestamp_audit.csv',index=False)
    # For event definitions, only require valid ICU start, matching patient, and start within admission.
    start_ok=(links.intime.notna() & (links.subject_id_icu==links.subject_id_adm) & (links.intime>=links.admittime) & (links.intime<=links.dischtime))
    starts=links.loc[start_ok].copy()
    starts=starts.sort_values(['hadm_id','intime','stay_id'])
    first=starts.groupby('hadm_id',sort=False).first()[['intime']].rename(columns={'intime':'first_icu_intime'})
    admissions=admissions.join(first,on='hadm_id')
    landmark=admissions.admittime+pd.to_timedelta(a.landmark_hours,unit='h')
    horizon=admissions.admittime+pd.to_timedelta(a.horizon_hours,unit='h')
    admissions['landmark']=landmark
    admissions['horizon']=horizon
    # Restrict to admissions still hospitalized at landmark, not already in ICU by then.
    # First ICU start at landmark is considered already ICU, conservatively.
    eligible=(admissions.dischtime>landmark)&(admissions.first_icu_intime.isna() | (admissions.first_icu_intime>landmark))
    cohort=admissions.loc[eligible].copy()
    # Death is a competing event only when timestamp is within hospital admission.
    death=cohort.deathtime.where((cohort.deathtime>=cohort.landmark)&(cohort.deathtime<=cohort.dischtime))
    icut=cohort.first_icu_intime.where(cohort.first_icu_intime>cohort.landmark)
    discharge=cohort.dischtime
    # First event after landmark; tie priority ICU, death, discharge.
    eventtime=pd.concat([icut.rename('icu'),death.rename('death'),discharge.rename('discharge')],axis=1)
    earliest=eventtime.min(axis=1)
    event=np.select([icut.eq(earliest),death.eq(earliest),discharge.eq(earliest)],['icu','death','discharge'],default='unknown')
    cohort['first_event']=event
    cohort['first_event_time']=earliest
    cohort['event_hours_from_landmark']=(earliest-cohort.landmark).dt.total_seconds()/HOURS
    cohort['event_by_horizon']=np.where(earliest<=cohort.horizon,event,'none_by_horizon')
    cohort['icu_24_72']=cohort.event_by_horizon.eq('icu')
    # Full hospital-followup event distribution, not artificially truncated.
    
    # Diagnoses are retrospective; only for subgroup descriptive comparisons, never assumed available at landmark.
    dx=diagnoses.loc[diagnoses.hadm_id.isin(cohort.hadm_id),['hadm_id','icd_code','icd_version']].copy()
    dx['code']=dx.icd_code.fillna('').astype(str).str.upper().str.replace('.','',regex=False)
    ver=pd.to_numeric(dx.icd_version,errors='coerce')
    rules={
       'renal_aki_ckd': ((ver==9)&dx.code.str.startswith(('584','585'))) | ((ver==10)&dx.code.str.startswith(('N17','N18'))),
       'renal_urinary_chapter':((ver==9)&dx.code.str.match(r'^5[89]')) | ((ver==10)&dx.code.str.match(r'^N\d')),
       'cardiovascular_chapter':((ver==9)&dx.code.str.match(r'^(39\d|4[0-5]\d)')) | ((ver==10)&dx.code.str.match(r'^I\d')),
       'respiratory_chapter':((ver==9)&dx.code.str.match(r'^4[6-9]\d|^5[0-1]\d')) | ((ver==10)&dx.code.str.match(r'^J\d')),
       'infectious_chapter':((ver==9)&dx.code.str.match(r'^0\d\d|^1[0-3]\d')) | ((ver==10)&dx.code.str.match(r'^[AB]\d')),
    }
    for name,mask in rules.items():
        positive=set(dx.loc[mask,'hadm_id'])
        cohort[name]=cohort.hadm_id.isin(positive)
    cohort['all_adults']=True
    names=['all_adults']+list(rules)
    
    # Calculate valid ICU stay LOS, readmission and return-to-ward metrics on actual observed stays.
    valid_los=starts.loc[starts.outtime.notna() & (starts.outtime>starts.intime)].copy()
    valid_los['icu_los_hours']=(valid_los.outtime-valid_los.intime).dt.total_seconds()/HOURS
    valid_los=valid_los.loc[valid_los.icu_los_hours.between(0,24*365)]
    valid_los['icu_stay_rank']=valid_los.groupby('hadm_id').cumcount()+1
    stay_counts=starts.groupby('hadm_id').size()
    cohort['n_icu_stays']=cohort.hadm_id.map(stay_counts).fillna(0).astype(int)
    # ICU length-of-stay after first post-landmark transfer, if outtime is valid.
    first_post=valid_los.loc[valid_los.intime>valid_los.admittime+pd.to_timedelta(a.landmark_hours,unit='h')].sort_values(['hadm_id','intime']).drop_duplicates('hadm_id')
    cohort['first_postlandmark_icu_los_hours']=cohort.hadm_id.map(first_post.set_index('hadm_id').icu_los_hours)
    cohort['time_to_icu_hours']=(cohort.first_icu_intime-cohort.landmark).dt.total_seconds()/HOURS
    cohort['hospital_los_hours']=(cohort.dischtime-cohort.admittime).dt.total_seconds()/HOURS
    
    rows=[];dists=[];transitions=[]
    for name in names:
        sub=cohort.loc[cohort[name]]
        n=len(sub)
        if not n: continue
        events=sub.event_by_horizon.value_counts()
        all_events=sub.first_event.value_counts()
        icus=sub.loc[sub.first_event=='icu']
        rows.append({'cohort':name,'admissions':n,'patients':sub.subject_id.nunique(),
            'icu_by_horizon':int(events.get('icu',0)), 'discharge_by_horizon':int(events.get('discharge',0)),
            'death_by_horizon':int(events.get('death',0)),'none_by_horizon':int(events.get('none_by_horizon',0)),
            'icu_pct_by_horizon':100*events.get('icu',0)/n,
            'icu_ever_before_hospital_exit':int(all_events.get('icu',0)),
            'discharge_first_full_followup':int(all_events.get('discharge',0)),
            'death_first_full_followup':int(all_events.get('death',0)),
            'icu_ever_pct':100*all_events.get('icu',0)/n,
            'icu_admissions_with_multiple_stays':int((icus.n_icu_stays>1).sum())})
        for variable, values in [('time_to_icu_from_landmark_hours',icus.time_to_icu_hours),
                                 ('first_postlandmark_icu_los_hours',icus.first_postlandmark_icu_los_hours),
                                 ('hospital_los_hours',sub.hospital_los_hours)]:
            dists.append({'cohort':name,'variable':variable,**qstats(values)})
        for state,cnt in all_events.items():
            transitions.append({'cohort':name,'first_event':state,'count':int(cnt)})
    pd.DataFrame(rows).to_csv(out/'cohort_comparison.csv',index=False)
    pd.DataFrame(dists).to_csv(out/'timing_distributions.csv',index=False)
    pd.DataFrame(transitions).to_csv(out/'first_event_counts.csv',index=False)
    # Event timing histogram for first ICU entry, exclusive 12-hour bins from landmark.
    event_rows=[]
    for name in names:
        sub=cohort.loc[cohort[name] & cohort.first_event.eq('icu')]
        for start in [0,12,24,36,48,72,96,120,168,240,336,504,720]:
            end={0:12,12:24,24:36,36:48,48:72,72:96,96:120,120:168,168:240,240:336,336:504,504:720,720:float('inf')}[start]
            count=int(((sub.time_to_icu_hours>=start)&(sub.time_to_icu_hours<end)).sum())
            event_rows.append({'cohort':name,'start_hours_after_landmark':start,'end_hours_after_landmark':end if np.isfinite(end) else 'infinity','icu_events':count})
    pd.DataFrame(event_rows).to_csv(out/'icu_transfer_timing_bins.csv',index=False)
    # Eventual ICU readmission after first ICU discharge, per admission, based on starts with valid outtime.
    readmit=[]
    for hadm, grp in valid_los.groupby('hadm_id',sort=False):
        grp=grp.sort_values('intime')
        if len(grp)<2: continue
        previous_out=grp.outtime.iloc[:-1].to_numpy(dtype='datetime64[ns]')
        next_in=grp.intime.iloc[1:].to_numpy(dtype='datetime64[ns]')
        if np.any(next_in>previous_out): readmit.append(hadm)
    cohort['has_icu_readmission']=cohort.hadm_id.isin(readmit)
    readmission_summary=[]
    for name in names:
        sub=cohort.loc[cohort[name]]
        icu_admissions=sub.loc[sub.n_icu_stays>0]
        readmission_summary.append({'cohort':name,'icu_admissions':len(icu_admissions),'with_subsequent_icu_stay':int(icu_admissions.has_icu_readmission.sum())})
    pd.DataFrame(readmission_summary).to_csv(out/'icu_readmission_summary.csv',index=False)
    meta={'source_root':str(root),'landmark_hours_from_admission':a.landmark_hours,
          'horizon_hours_from_admission':a.horizon_hours,'eligible_admissions':int(len(cohort)),
          'eligible_patients':int(cohort.subject_id.nunique()),'adult_admissions_valid_timestamps':int(len(admissions)),
          'icu_linked_rows':int(len(links)),'icu_start_usable_rows':int(start_ok.sum()),
          'timestamp_audit':'icu_timestamp_audit.csv',
          'privacy':'Aggregate outputs only; no patient-level export.',
          'limitations':[
            'ICD cohort membership is based on retrospective discharge codes; not valid as a prediction-time feature.',
            'ICU entry is observed utilization, not a direct measure of physiological deterioration or bed demand.',
            'MIMIC-IV patient dates are independently shifted; cross-patient overlap cannot reconstruct hospital occupancy.',
            'A valid ICU start can be retained despite an invalid ICU end; ICU LOS excludes invalid ends.',
            'Hospital death is identified from deathtime when within admission; discharge and death ties prioritize death.',
            'ICU readmissions are identified only when two valid ICU intervals are chronologically separated.',
            'Disease chapters are broad approximations and overlap; do not sum cohort counts.',
            'No administrative censoring beyond hospital discharge is assumed for admissions with valid discharge times.'
          ]}
    (out/'audit_manifest.json').write_text(json.dumps(meta,indent=2))
    print('\nCOHORT SUMMARY (horizon is hours since admission):',flush=True)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print('\nSaved aggregate audit to',out,flush=True)

if __name__=='__main__':
    main()
