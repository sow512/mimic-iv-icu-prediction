#!/usr/bin/env python3
"""Phase 2: memory-bounded, leakage-aware temporal lab features for MIMIC-IV 3.1.

Research only. Uses lab charttime AND storetime <= 24h landmark. When
storetime is absent, fails closed by default. Does not read outcome labels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

LAB_MAP = {
 50912:'creatinine',51006:'bun',50983:'sodium',50971:'potassium',50902:'chloride',
 50882:'bicarbonate',50893:'calcium',50960:'magnesium',50970:'phosphate',
 50931:'glucose',50813:'lactate',50868:'anion_gap',51301:'wbc',51222:'hemoglobin',
 51221:'hematocrit',51265:'platelets',51277:'rdw',50885:'bilirubin_total',
 50862:'albumin',50861:'alt',50878:'ast',50863:'alkaline_phosphatase',
 51237:'inr',51274:'pt',51275:'ptt',50820:'ph',50818:'pco2',50821:'po2'
}
# Wide safety bounds, applied only after item/unit checks. No quantile clipping.
RANGES = {'albumin':(.5,10),'alkaline_phosphatase':(1,10000),'alt':(1,30000),
'anion_gap':(0,60),'ast':(1,40000),'bicarbonate':(3,60),
'bilirubin_total':(.1,100),'bun':(1,300),'calcium':(2,20),
'chloride':(50,150),'creatinine':(.1,30),'glucose':(10,2000),
'hematocrit':(5,75),'hemoglobin':(2,25),'inr':(.5,30),'lactate':(.1,40),
'magnesium':(.2,10),'pco2':(5,200),'ph':(6.5,8),'phosphate':(.2,20),
'platelets':(1,2000),'po2':(5,700),'potassium':(1.5,10),'pt':(5,150),
'ptt':(10,200),'rdw':(5,50),'sodium':(80,180),'wbc':(.1,500)}

def locate(base):
    for root in (base,base/'physionet.org'/'files'/'mimiciv'/'3.1',base/'mimiciv'/'3.1'):
        for suffix in ('.csv.gz','.csv'):
            path=root/'hosp'/('labevents'+suffix)
            if path.exists(): return path
    raise FileNotFoundError('Could not find hosp/labevents.csv[.gz] under '+str(base))

def run(base, phase1, output, chunk_size):
    output.mkdir(parents=True,exist_ok=True)
    pred=pd.read_parquet(phase1/'phase1_landmark_predictors.parquet')
    required={'subject_id','hadm_id','landmark_time'}
    if not required.issubset(pred.columns): raise ValueError('Missing Phase 1 columns: '+str(required-set(pred.columns)))
    if pred.hadm_id.isna().any() or pred.hadm_id.duplicated().any(): raise ValueError('Non-unique hadm_id')
    pred['landmark_time']=pd.to_datetime(pred.landmark_time,errors='raise')
    # Use an explicit join on BOTH identifiers, rejecting conflicting links.
    links=pred[['subject_id','hadm_id','landmark_time']].copy()
    source=locate(base)
    print('Reading in chunks:',source,flush=True)
    selected=[]
    counters={'rows_read':0,'selected_item_and_admission':0,'available_by_landmark':0,'valid_numeric_unit_and_range':0}
    units={}
    cols=['subject_id','hadm_id','itemid','charttime','storetime','valuenum','valueuom']
    for i,chunk in enumerate(pd.read_csv(source,usecols=cols,chunksize=chunk_size,low_memory=False),1):
        counters['rows_read']+=len(chunk)
        chunk=chunk.loc[chunk.itemid.isin(LAB_MAP)&chunk.hadm_id.isin(links.hadm_id)].copy()
        if chunk.empty: continue
        chunk=chunk.merge(links,on=['subject_id','hadm_id'],how='inner',validate='many_to_one')
        counters['selected_item_and_admission']+=len(chunk)
        if chunk.empty: continue
        chunk['charttime']=pd.to_datetime(chunk.charttime,errors='coerce')
        chunk['storetime']=pd.to_datetime(chunk.storetime,errors='coerce')
        # Strict timestamp policy: results must be charted AND stored by prediction time.
        chunk=chunk.loc[chunk.charttime.notna()&chunk.storetime.notna()&
            chunk.charttime.le(chunk.landmark_time)&chunk.storetime.le(chunk.landmark_time)&
            chunk.storetime.ge(chunk.charttime)].copy()
        # Recover admission time from 24h landmark; only first 24h accepted.
        chunk['hour']=(chunk.charttime-(chunk.landmark_time-pd.Timedelta(hours=24))).dt.total_seconds()/3600
        chunk=chunk.loc[chunk.hour.ge(0)&chunk.hour.lt(24)].copy()
        counters['available_by_landmark']+=len(chunk)
        if chunk.empty: continue
        chunk['lab']=chunk.itemid.map(LAB_MAP)
        chunk['valuenum']=pd.to_numeric(chunk.valuenum,errors='coerce')
        chunk=chunk.loc[np.isfinite(chunk.valuenum)].copy()
        # Retain units for audit; no silent unit conversion.
        for (lab,unit),n in chunk.groupby(['lab','valueuom'],dropna=False).size().items():
            key=str(lab)+' | '+str(unit)
            units[key]=units.get(key,0)+int(n)
        low=chunk.lab.map({k:v[0] for k,v in RANGES.items()})
        high=chunk.lab.map({k:v[1] for k,v in RANGES.items()})
        chunk=chunk.loc[chunk.valuenum.ge(low)&chunk.valuenum.le(high)].copy()
        counters['valid_numeric_unit_and_range']+=len(chunk)
        if not chunk.empty:
            selected.append(chunk[['subject_id','hadm_id','lab','charttime','storetime','hour','valuenum','valueuom']])
        if i%20==0: print(f'Chunks {i}; rows scanned {counters["rows_read"]:,}',flush=True)
    if not selected: raise RuntimeError('No eligible labs. Check paths, time constraints and source data.')
    labs=pd.concat(selected,ignore_index=True)
    del selected
    labs=labs.drop_duplicates(['subject_id','hadm_id','lab','charttime','storetime','valuenum','valueuom'])
    # Do not silently combine different measurement units for the same lab.
    # Save audit and fail if multiple non-null units occur for any lab.
    unit_report=pd.DataFrame([{'lab_unit':k,'count':v} for k,v in sorted(units.items())])
    unit_report.to_csv(output/'phase2_unit_audit.csv',index=False)
    mixed=labs.groupby('lab').valueuom.nunique(dropna=False)
    if (mixed>1).any():
        raise ValueError('Multiple units detected; review phase2_unit_audit.csv and harmonize before aggregating: '+str(mixed[mixed>1].to_dict()))
    labs=labs.sort_values(['hadm_id','lab','charttime','storetime'])
    # Four distinct temporal summaries per 6/6/12-hour bin, plus overall trend.
    labs['bin']=pd.cut(labs.hour,bins=[0,6,12,24],labels=['h00_06','h06_12','h12_24'],right=False,include_lowest=True)
    agg=labs.groupby(['hadm_id','lab','bin'],observed=True).valuenum.agg(['count','mean','min','max','std','last']).reset_index()
    agg=agg.rename(columns={'last':'latest'})
    wide=agg.pivot(index='hadm_id',columns=['lab','bin'],values=['count','mean','min','max','std','latest'])
    wide.columns=[f'{lab}_{timebin}_{stat}' for stat,lab,timebin in wide.columns]
    wide=wide.reset_index()
    overall=labs.groupby(['hadm_id','lab'],sort=False).agg(first=('valuenum','first'),latest=('valuenum','last'),
        first_hour=('hour','first'),last_hour=('hour','last'),n=('valuenum','size')).reset_index()
    overall['delta']=overall.latest-overall['first']
    elapsed=overall.last_hour-overall.first_hour
    overall['slope_per_hour']=np.where(elapsed.gt(0),overall.delta/elapsed,np.nan)
    trend=overall.pivot(index='hadm_id',columns='lab',values=['latest','delta','slope_per_hour','n'])
    trend.columns=[f'{lab}_overall_{stat}' for stat,lab in trend.columns]
    trend=trend.reset_index()
    features=pred.merge(wide,on='hadm_id',how='left',validate='one_to_one').merge(trend,on='hadm_id',how='left',validate='one_to_one')
    # Count features represent absence of measurement, not missing numeric results.
    count_cols=[c for c in features if c.endswith('_count') or c.endswith('_n')]
    features[count_cols]=features[count_cols].fillna(0)
    forbidden={'icu_transfer_24_72h','first_icu_intime','dischtime','deathtime','horizon_end','label_status'}
    if forbidden.intersection(features.columns): raise AssertionError('Outcome leakage columns detected')
    if len(features)!=len(pred) or features.hadm_id.duplicated().any(): raise AssertionError('Cohort cardinality changed')
    if labs.charttime.gt(labs.storetime).any(): raise AssertionError('Invalid chart/store ordering')
    if labs.hour.lt(0).any() or labs.hour.ge(24).any(): raise AssertionError('Out-of-window labs')
    features.to_parquet(output/'phase2_temporal_lab_features.parquet',index=False)
    report={**counters,'distinct_admissions_with_labs':int(labs.hadm_id.nunique()),
            'eligible_admissions':int(len(pred)),'feature_columns':int(len(features.columns)-len(pred.columns)),
            'retained_lab_rows_after_dedup':int(len(labs)),
            'policy':'charttime and storetime non-null and <= landmark; charttime within [0,24) hours; strict unit check'}
    (output/'phase2_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print('Saved:',output/'phase2_temporal_lab_features.parquet')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--phase1',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chunk-size',type=int,default=500000)
    a=p.parse_args()
    run(a.base,a.phase1,a.output,a.chunk_size)
