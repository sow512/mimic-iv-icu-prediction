#!/usr/bin/env python3
"""Dynamic renal-history ICU-transfer pilot, MIMIC-IV 3.1.
Uses Phase18 24h-eligible cohort and its patient-disjoint splits. Landmarks >=24h.
At each landmark predicts ICU-first-event in NEXT 48h (discharge/death compete).
Rebuilds time-dependent lab summaries from raw events with charttime AND storetime cutoffs.
This is a retrospective research experiment, not a clinical model.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib

LABS={50912:'creatinine',51006:'bun',50971:'potassium',50983:'sodium',50882:'bicarbonate',50902:'chloride',50868:'anion_gap',50931:'glucose',50813:'lactate',51301:'wbc',51222:'hemoglobin',51265:'platelets',51275:'ptt',51274:'pt',50885:'bilirubin',50970:'phosphate',50893:'calcium'}

def locate(p):
    p=Path(p).expanduser()
    for q in [p,p/'physionet.org/files/mimiciv/3.1']:
        if (q/'hosp/admissions.csv').exists():return q
    raise FileNotFoundError('MIMIC-IV root with hosp/admissions.csv not found')

def scores(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    k=max(1,int(np.ceil(.1*len(y))));top=np.argsort(-p,kind='stable')[:k]
    return {'n':int(len(y)),'events':int(y.sum()),'prevalence':float(y.mean()),'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,'auprc':float(average_precision_score(y,p)) if y.sum() else None,'brier':float(brier_score_loss(y,p)),'recall_top10':float(y[top].sum()/y.sum()) if y.sum() else None,'expected_events':float(p.sum())}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',required=True);ap.add_argument('--phase18',required=True);ap.add_argument('--phase18c',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--landmarks',type=int,nargs='+',default=[24,36,48,60]);ap.add_argument('--horizon',type=int,default=48)
    ap.add_argument('--chunksize',type=int,default=200000);ap.add_argument('--max-iter',type=int,default=250)
    a=ap.parse_args();t0=time.time()
    if min(a.landmarks)<24 or len(set(a.landmarks))!=len(a.landmarks):ap.error('Landmarks must be unique and >=24h (Phase18 eligibility starts at 24h)')
    root=locate(a.base);out=Path(a.output).expanduser();out.mkdir(parents=True,exist_ok=True)
    f=pd.read_parquet(Path(a.phase18)/'prelandmark_features.parquet')
    e=pd.read_parquet(Path(a.phase18)/'first_events.parquet',columns=['hadm_id','subject_id','landmark','first_event','event_time'])
    hist=pd.read_parquet(Path(a.phase18c)/'history_diagnoses.parquet')
    keys=['hist_renal_aki','hist_renal_ckd']
    for k in keys:
        if k not in hist:raise ValueError('Missing renal-history column: '+k)
    renal=hist.loc[hist[keys].fillna(0).gt(0).any(axis=1),'hadm_id']
    f=f.loc[f.hadm_id.isin(renal)].copy()
    if not f.hadm_id.is_unique:raise ValueError('Duplicate admissions')
    if f.groupby('subject_id').split.nunique().max()!=1:raise ValueError('Patient split overlap')
    e=e.loc[e.hadm_id.isin(f.hadm_id)].copy()
    if not e.hadm_id.is_unique:raise ValueError('Duplicate event keys')
    adm=pd.read_csv(root/'hosp/admissions.csv',usecols=['hadm_id','admittime','dischtime','deathtime'],low_memory=False)
    adm=adm.loc[adm.hadm_id.isin(f.hadm_id)].copy()
    for c in ['admittime','dischtime','deathtime']:adm[c]=pd.to_datetime(adm[c],errors='coerce')
    d=f.merge(e[['hadm_id','first_event','event_time']],on='hadm_id',validate='one_to_one').merge(adm,on='hadm_id',validate='one_to_one')
    d['event_time']=pd.to_datetime(d.event_time,errors='coerce')
    d=d.merge(hist[['hadm_id']+keys],on='hadm_id',how='left',validate='one_to_one')
    if d[['admittime','event_time']].isna().any().any():raise ValueError('Missing essential timestamps')
    rows=[]
    for h in sorted(a.landmarks):
        z=d.copy();z['landmark_hour']=h;z['prediction_time']=z.admittime+pd.to_timedelta(h,unit='h')
        # Eligible only while still in hospital and before any ICU/discharge/death first event.
        z=z.loc[z.event_time>z.prediction_time].copy()
        z['hours_until_event']=(z.event_time-z.prediction_time).dt.total_seconds()/3600
        z['outcome']=((z.first_event=='icu')&(z.hours_until_event<=a.horizon)).astype('int8')
        z['competing_event']=np.where(z.hours_until_event<=a.horizon,z.first_event,'none_within_horizon')
        rows.append(z)
    panel=pd.concat(rows,ignore_index=True)
    if panel.empty:raise ValueError('No eligible landmark rows')
    if panel[['hadm_id','landmark_hour']].duplicated().any():raise ValueError('Duplicate landmark keys')
    print('Renal-history admissions:',len(d),'landmark rows:',len(panel),flush=True)
    # Avoid using any future event/discharge times as predictors. Static predictors are known at admission.
    static=['age','prior_completed_admissions','hist_renal_aki','hist_renal_ckd']
    categorical=[c for c in ['gender','admission_type','admission_location'] if c in panel]
    for c in static:
        if c not in panel:raise ValueError('Missing baseline feature '+c)
    # Restrict raw lab extraction to renal cohort, specified labs, and <= max landmark.
    # Use chunk-wise cumulative stats at each landmark; do not retain raw events in memory.
    key=panel[['hadm_id','landmark_hour','prediction_time','admittime']].copy()
    key['row_id']=np.arange(len(key),dtype=np.int64)
    groups={};scanned=0;kept=0
    use=['hadm_id','itemid','charttime','storetime','valuenum']
    for ch in pd.read_csv(root/'hosp/labevents.csv',usecols=use,chunksize=a.chunksize,low_memory=False):
        scanned+=len(ch)
        ch=ch.loc[ch.hadm_id.isin(d.hadm_id)&ch.itemid.isin(LABS)&ch.valuenum.notna()].copy()
        if ch.empty:continue
        ch['charttime']=pd.to_datetime(ch.charttime,errors='coerce');ch['storetime']=pd.to_datetime(ch.storetime,errors='coerce')
        ch=ch.loc[ch.charttime.notna()&ch.storetime.notna()];
        if ch.empty:continue
        ch=ch.merge(key,on='hadm_id',how='inner',validate='many_to_many')
        ch=ch.loc[(ch.charttime>=ch.admittime)&(ch.charttime<ch.prediction_time)&(ch.storetime<ch.prediction_time)&np.isfinite(ch.valuenum)].copy()
        if ch.empty:continue
        kept+=len(ch)
        for (rid,item),sub in ch.groupby(['row_id','itemid'],sort=False):
            g=groups.setdefault((int(rid),int(item)),[0,0.,float('inf'),float('-inf'),None,None,None,None])
            vals=sub.valuenum.to_numpy(dtype=float);g[0]+=len(vals);g[1]+=float(vals.sum());g[2]=min(g[2],float(vals.min()));g[3]=max(g[3],float(vals.max()))
            # First/last based on charttime, storetime tiebreaker.
            first=sub.sort_values(['charttime','storetime']).iloc[0];last=sub.sort_values(['charttime','storetime']).iloc[-1]
            if g[4] is None or first.charttime<g[4]:g[4]=first.charttime;g[5]=float(first.valuenum)
            if g[6] is None or last.charttime>=g[6]:g[6]=last.charttime;g[7]=float(last.valuenum)
        if scanned//5000000 != (scanned-len(ch))//5000000:print('Lab rows scanned:',scanned,'eligible row-observations:',kept,flush=True)
    feat={}
    for item,name in LABS.items():
        for stat in ['n','mean','min','max','first','last','delta','range']:feat[name+'_'+stat]=np.full(len(panel),np.nan,dtype=np.float32)
    for (rid,item),g in groups.items():
        name=LABS[item];n,s,lo,hi,_,first,__,last=g
        for stat,val in [('n',n),('mean',s/n),('min',lo),('max',hi),('first',first),('last',last),('delta',last-first),('range',hi-lo)]:feat[name+'_'+stat][rid]=val
    X=pd.DataFrame(feat)
    for col in X:panel[col]=X[col].to_numpy()
    # Explicit missingness and utilization: counts for unmeasured tests = 0; values remain NaN.
    labcols=list(feat)
    for c in [x for x in labcols if x.endswith('_n')]:panel[c]=panel[c].fillna(0)
    num=static+['landmark_hour']+labcols
    panel['split']=panel.split.astype(str)
    tr=panel.split=='train';st=panel.split=='early_stop';va=panel.split=='validation'
    if not (tr.any() and st.any() and va.any()):raise ValueError('Required splits missing')
    # StandardScaler sparse-friendly optional; logistic provides a fast CPU-friendly first dynamic benchmark.
    prep=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),num),('cat',OneHotEncoder(handle_unknown='ignore'),categorical)],sparse_threshold=0)
    Xtr=prep.fit_transform(panel.loc[tr,num+categorical]);Xst=prep.transform(panel.loc[st,num+categorical]);Xv=prep.transform(panel.loc[va,num+categorical])
    ytr=panel.loc[tr,'outcome'].to_numpy();yst=panel.loc[st,'outcome'].to_numpy();yv=panel.loc[va,'outcome'].to_numpy()
    # Repeated landmark rows are dependent: keep patient splits and report results per landmark.
    model=LogisticRegression(max_iter=a.max_iter,solver='lbfgs',C=0.1)
    model.fit(Xtr,ytr)
    p=model.predict_proba(Xv)[:,1];pst=model.predict_proba(Xst)[:,1]
    validation=panel.loc[va,['subject_id','hadm_id','landmark_hour','outcome','competing_event','hours_until_event']].copy()
    validation['predicted_risk']=p
    validation.to_parquet(out/'validation_dynamic_predictions.parquet',index=False)
    reports={'early_stop':scores(yst,pst),'validation_overall':scores(yv,p),'validation_by_landmark':{},'cohort_definition':'Historical AKI/CKD from Phase18C, originally eligible at admission hour 24','target':'ICU first event within next 48h; discharge/death are negatives in this binary pilot','landmarks':sorted(a.landmarks),'feature_count':len(num)+len(categorical),'source_rows':{'renal_24h_cohort':len(d),'dynamic_rows':len(panel),'lab_rows_scanned':scanned,'eligible_lab_row_observations':kept},'limitations':['Selection conditioned on being eligible at hour 24; landmarks earlier than 24 prohibited','Binary outcome does not separately estimate discharge/death competing incidence','No external or temporal validation; original patient split retained','Clinical timestamp provenance only checked via charttime/storetime for labs','No ward vital signs; no treatment or medication features','Repeated landmark predictions correlated; do not treat rows as independent in confidence intervals','Potential source missingness and changing risk sets; compare horizons with caution']}
    for h,sub in validation.groupby('landmark_hour'):
        reports['validation_by_landmark'][str(h)]=scores(sub.outcome,sub.predicted_risk)
    joblib.dump({'model':model,'preprocessor':prep,'numeric_columns':num,'categorical_columns':categorical},out/'dynamic_renal_model.joblib')
    # Save only development labels; evaluation split remains excluded from exported data.
    panel.loc[panel.split!='evaluation',['subject_id','hadm_id','split','landmark_hour','outcome','competing_event','hours_until_event']].to_parquet(out/'development_landmark_labels.parquet',index=False)
    reports['seconds']=round(time.time()-t0,1)
    (out/'report.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2),flush=True)
    print('COMPLETE:',out,flush=True)
if __name__=='__main__':main()
