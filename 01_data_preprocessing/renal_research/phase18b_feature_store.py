#!/usr/bin/env python3
"""Phase 18B: resumable, leakage-aware MIMIC-IV hospital-wide feature store.
Reads Phase 18 prelandmark_features.parquet and first_events.parquet.
Writes one admission-keyed parquet per source; never reads outcomes for feature selection.
No current-admission discharge ICD/DRG codes used. No ICU post-landmark data used.
Requires pandas, numpy, pyarrow. No model fitting.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

KEY = 'hadm_id'

def slug(s):
    return re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_')[:70]

def locate(base):
    base = Path(base).expanduser().resolve()
    if (base/'hosp'/'admissions.csv').exists(): return base
    p = base/'physionet.org/files/mimiciv/3.1'
    if p.exists(): return p
    raise FileNotFoundError(f'Cannot locate MIMIC-IV hosp/ under {base}')

def dates(df, *names):
    for c in names:
        if c in df: df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

def read(root, rel, usecols):
    path = root/rel
    if not path.exists(): raise FileNotFoundError(path)
    print('Reading',path,flush=True)
    return pd.read_csv(path,usecols=usecols,low_memory=False)

def write_module(out, name, df, expected, force):
    if KEY not in df: raise ValueError(f'{name} lacks hadm_id')
    if df[KEY].duplicated().any(): raise ValueError(f'{name} has duplicate admission keys')
    if not set(df[KEY]).issubset(expected): raise ValueError(f'{name} has foreign admissions')
    path = out/(name+'.parquet')
    tmp = out/(name+'.tmp.parquet')
    df.to_parquet(tmp,index=False)
    tmp.replace(path)
    print(f'Saved {name}: {len(df):,} admissions, {len(df.columns)-1} features',flush=True)

def history(root, cohort):
    a = read(root,'hosp/admissions.csv',['subject_id','hadm_id','admittime','dischtime'])
    dates(a,'admittime','dischtime')
    a = a[a.dischtime.notna() & a.admittime.notna() & (a.dischtime>a.admittime)].copy()
    # Only completed historical admissions, before the index admission begins.
    left=cohort[['subject_id','hadm_id','admittime']].sort_values('admittime')
    prior=a.sort_values(['subject_id','dischtime','hadm_id']).copy()
    prior['prior_count']=prior.groupby('subject_id').cumcount()+1
    prior['prior_total_days']=(prior.dischtime-prior.admittime).dt.total_seconds()/86400
    prior['prior_total_days']=prior.groupby('subject_id').prior_total_days.cumsum()
    right=prior[['subject_id','dischtime','prior_count','prior_total_days']].sort_values('dischtime')
    z=pd.merge_asof(left,right,by='subject_id',left_on='admittime',right_on='dischtime',direction='backward',allow_exact_matches=False)
    z['hist_prior_admissions']=z.prior_count.fillna(0).astype('int32')
    z['hist_prior_total_days']=z.prior_total_days.fillna(0)
    z['hist_days_since_discharge']=(z.admittime-z.dischtime).dt.total_seconds()/86400
    # Historical ICU stays: ICU stay must end before the index admission starts.
    icu=read(root,'icu/icustays.csv',['subject_id','intime','outtime'])
    dates(icu,'intime','outtime')
    icu=icu[icu.outtime.notna() & icu.intime.notna() & (icu.outtime>icu.intime)].copy()
    icu['duration_days']=(icu.outtime-icu.intime).dt.total_seconds()/86400
    icu=icu.sort_values(['subject_id','outtime'])
    icu['icu_count']=icu.groupby('subject_id').cumcount()+1
    icu['icu_days']=icu.groupby('subject_id').duration_days.cumsum()
    i=pd.merge_asof(left.sort_values('admittime'),icu[['subject_id','outtime','icu_count','icu_days']].sort_values('outtime'),by='subject_id',left_on='admittime',right_on='outtime',direction='backward',allow_exact_matches=False)
    z=z.merge(i[['hadm_id','icu_count','icu_days','outtime']],on='hadm_id',validate='one_to_one')
    z['hist_prior_icu_stays']=z.icu_count.fillna(0).astype('int32')
    z['hist_prior_icu_days']=z.icu_days.fillna(0)
    z['hist_days_since_icu']=(z.admittime-z.outtime).dt.total_seconds()/86400
    return z[['hadm_id','hist_prior_admissions','hist_prior_total_days','hist_days_since_discharge','hist_prior_icu_stays','hist_prior_icu_days','hist_days_since_icu']]

def history_diagnoses(root, cohort, chunksize):
    # Only codes belonging to an admission discharged before the index admission.
    a=read(root,'hosp/admissions.csv',['subject_id','hadm_id','dischtime'])
    dates(a,'dischtime')
    index=cohort[['subject_id','hadm_id','admittime']].rename(columns={'hadm_id':'index_hadm_id'})
    prev=a.merge(index,on='subject_id',how='inner')
    prev=prev[prev.dischtime.notna() & (prev.dischtime<prev.admittime)][['hadm_id','index_hadm_id']]
    # To avoid massive patient cross-products for recurrent admissions, restrict to ICD groups and aggregate prior-admission flags.
    groups={'renal':r'^(N0[0-9]|N1[0-9]|N17|N18|N19|58[0-9])',
            'cardiac':r'^(I[0-5][0-9]|4[0-4][0-9])',
            'respiratory':r'^(J[0-9]|4[6-9][0-9]|5[0-1][0-9])',
            'diabetes':r'^(E0[8-9]|25[0-9])',
            'infection':r'^(A[0-9]|B[0-9]|0[0-9][0-9])'}
    # Build a per-hospitalization code indicator first, then link only historical stays.
    src=root/'hosp/diagnoses_icd.csv'
    chunks=[]
    for chunk in pd.read_csv(src,usecols=['hadm_id','icd_code','icd_version'],chunksize=chunksize,dtype={'icd_code':'string'}):
        c=chunk.icd_code.fillna('').str.upper().str.replace('.','',regex=False)
        frame=pd.DataFrame({'hadm_id':chunk.hadm_id})
        for name,pat in groups.items(): frame[name]=c.str.match(pat).astype('int8')
        chunks.append(frame.groupby('hadm_id',sort=False).max())
    if not chunks: return pd.DataFrame({KEY:cohort.hadm_id})
    codes=pd.concat(chunks).groupby(level=0).max().reset_index()
    z=prev.merge(codes,on='hadm_id',how='left')
    vals=z.groupby('index_hadm_id')[list(groups)].sum().add_prefix('hist_dx_prior_').reset_index().rename(columns={'index_hadm_id':'hadm_id'})
    return vals

def aggregate_events(root, rel, cohort, chunksize, module, itemcol, timecol, extra_time=None, textcol=None):
    path=root/rel
    if not path.exists(): raise FileNotFoundError(path)
    keys=cohort.set_index('hadm_id')[['admittime','landmark']]
    cols=['hadm_id',timecol,itemcol]
    if extra_time and extra_time not in cols: cols.append(extra_time)
    if textcol and textcol not in cols: cols.append(textcol)
    counts={}; unique=set(); read_n=0; kept_n=0
    for ch in pd.read_csv(path,usecols=cols,chunksize=chunksize,low_memory=False):
        read_n+=len(ch)
        ch=ch[ch.hadm_id.isin(keys.index)].copy()
        if ch.empty: continue
        dates(ch,timecol,*([extra_time] if extra_time else []))
        ch=ch.join(keys,on='hadm_id')
        mask=ch[timecol].notna() & (ch[timecol]>=ch.admittime) & (ch[timecol]<ch.landmark)
        if extra_time: mask &= ch[extra_time].notna() & (ch[extra_time]<ch.landmark)
        ch=ch[mask]
        if ch.empty: continue
        kept_n+=len(ch)
        # No unbounded raw-row retention: aggregate incrementally by admission.
        for hadm,part in ch.groupby('hadm_id',sort=False):
            h=int(hadm)
            entry=counts.setdefault(h,{'count':0,'concepts':set(),'first':pd.NaT,'last':pd.NaT})
            entry['count']+=len(part)
            entry['concepts'].update(part[itemcol].dropna().astype(str).str.lower().str.strip().tolist())
            lo=part[timecol].min();hi=part[timecol].max()
            if pd.isna(entry['first']) or lo<entry['first']:entry['first']=lo
            if pd.isna(entry['last']) or hi>entry['last']:entry['last']=hi
        if read_n//(chunksize*10) != (read_n-len(ch))//(chunksize*10):
            print(f'{module}: scanned {read_n:,}, retained {kept_n:,}',flush=True)
    rows=[]
    for h,d in counts.items():
        rows.append({'hadm_id':h,f'{module}_events_24h':d['count'],f'{module}_distinct_concepts_24h':len(d['concepts']),f'{module}_first_event_time':d['first'],f'{module}_last_event_time':d['last']})
    z=pd.DataFrame(rows)
    if z.empty: z=pd.DataFrame(columns=['hadm_id',f'{module}_events_24h',f'{module}_distinct_concepts_24h',f'{module}_first_event_time',f'{module}_last_event_time'])
    # Keep timestamp columns in audit output; don't feed directly to a model without transforming relative to landmark.
    return z

def labs(root,cohort,chunksize):
    # Streaming aggregates, retaining per-admission/item/unit sufficient statistics only.
    keys=cohort.set_index('hadm_id')[['admittime','landmark']]
    d=read(root,'hosp/d_labitems.csv',['itemid','label','fluid'])
    lab_map=d.set_index('itemid').label.to_dict()
    accum={}; scanned=0; retained=0
    path=root/'hosp/labevents.csv'
    cols=['hadm_id','itemid','charttime','storetime','valuenum','valueuom']
    for ch in pd.read_csv(path,usecols=cols,chunksize=chunksize,low_memory=False):
        scanned+=len(ch)
        ch=ch[ch.hadm_id.isin(keys.index)&ch.valuenum.notna()].copy()
        if ch.empty:continue
        dates(ch,'charttime','storetime')
        ch=ch.join(keys,on='hadm_id')
        ch=ch[ch.charttime.notna() & ch.storetime.notna() & (ch.charttime>=ch.admittime) & (ch.charttime<ch.landmark) & (ch.storetime<ch.landmark)]
        if ch.empty:continue
        retained+=len(ch)
        # Preserve item ID and unit separately; do not combine incompatible units.
        for (hadm,item,unit),part in ch.groupby(['hadm_id','itemid','valueuom'],dropna=False,sort=False):
            key=(int(hadm),int(item),str(unit))
            d0=accum.setdefault(key,[0,0.,float('inf'),float('-inf'),pd.NaT,np.nan])
            v=part.valuenum.to_numpy(dtype=float)
            d0[0]+=len(v);d0[1]+=float(v.sum());d0[2]=min(d0[2],float(v.min()));d0[3]=max(d0[3],float(v.max()))
            last=part.sort_values('charttime').iloc[-1]
            if pd.isna(d0[4]) or last.charttime>d0[4]:d0[4]=last.charttime;d0[5]=float(last.valuenum)
        if scanned//10000000 != (scanned-len(ch))//10000000:
            print(f'labs: scanned {scanned:,}; retained {retained:,}; groups {len(accum):,}',flush=True)
    # Long-form feature store avoids gigantic, sparse wide matrices and preserves every item/unit.
    rows=[]
    for (hadm,item,unit),v in accum.items():
        rows.append({'hadm_id':hadm,'itemid':item,'label':lab_map.get(item,''),'unit':unit,'n':v[0],'mean':v[1]/v[0],'min':v[2],'max':v[3],'latest':v[5]})
    return pd.DataFrame(rows,columns=['hadm_id','itemid','label','unit','n','mean','min','max','latest'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True);p.add_argument('--phase18',required=True);p.add_argument('--output',required=True)
    p.add_argument('--modules',nargs='+',choices=['history','history_diagnoses','labs','prescriptions','emar','orders','microbiology'],default=['history','prescriptions','emar','orders','microbiology'])
    p.add_argument('--chunksize',type=int,default=500000);p.add_argument('--force',action='store_true')
    a=p.parse_args();root=locate(a.base);out=Path(a.output).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    phase=Path(a.phase18).expanduser().resolve()
    f=pd.read_parquet(phase/'prelandmark_features.parquet')
    ev=pd.read_parquet(phase/'first_events.parquet',columns=['hadm_id','landmark'])
    assert f.hadm_id.is_unique and ev.hadm_id.is_unique
    c=f[['subject_id','hadm_id','split']].merge(ev,on='hadm_id',validate='one_to_one')
    ad=read(root,'hosp/admissions.csv',['hadm_id','admittime'])
    dates(ad,'admittime');c=c.merge(ad,on='hadm_id',validate='one_to_one')
    if c.landmark.isna().any() or c.admittime.isna().any():raise ValueError('Missing timestamps')
    if (c.landmark<=c.admittime).any():raise ValueError('Invalid landmark')
    if c.groupby('subject_id').split.nunique().gt(1).any():raise ValueError('Patient split leakage')
    print(f'Phase18: {len(c):,} admissions, {c.subject_id.nunique():,} patients',flush=True)
    expected=set(c.hadm_id)
    config={
        'prescriptions':('hosp/prescriptions.csv','drug','starttime',None),
        'emar':('hosp/emar.csv','medication','charttime','storetime'),
        'orders':('hosp/poe.csv','order_type','ordertime',None),
        'microbiology':('hosp/microbiologyevents.csv','test_name','charttime','storetime'),
    }
    completed={};errors={}
    for name in a.modules:
        target=out/(name+'.parquet')
        if target.exists() and not a.force:
            print('Skipping existing',target,flush=True);completed[name]='existing';continue
        print('\nMODULE',name,flush=True)
        try:
            if name=='history': z=history(root,c)
            elif name=='history_diagnoses': z=history_diagnoses(root,c,a.chunksize)
            elif name=='labs': z=labs(root,c,a.chunksize)
            else:
                rel,item,t,available=config[name]
                z=aggregate_events(root,rel,c,a.chunksize,name,item,t,available)
            if name=='labs':
                # Multiple rows per admission by design for lab item/unit groups.
                temp=out/'labs.tmp.parquet';z.to_parquet(temp,index=False);temp.replace(target)
                print(f'Saved labs: {len(z):,} admission-item-unit groups',flush=True)
            else:write_module(out,name,z,expected,a.force)
            completed[name]=int(len(z))
        except Exception as exc:
            errors[name]=f'{type(exc).__name__}: {exc}'
            print('FAILED',name,errors[name],flush=True)
        (out/'manifest.json').write_text(json.dumps({'source_root':str(root),'phase18':str(phase),'cohort_admissions':len(c),'modules':completed,'errors':errors,'notes':['No outcome labels are read','Current-admission diagnosis codes are not used','Laboratory groups preserve item IDs and units','Timestamp cutoff uses chart and store times where available','Medication orders are not verified administrations','History diagnoses require completed prior admissions','Per-module files are resumable; use --force to replace','No notes or ICU post-landmark data extracted']},indent=2))
    if errors:raise SystemExit(f'Finished with errors: {errors}')
    print('\nCOMPLETE:',out,flush=True)
if __name__=='__main__':main()
