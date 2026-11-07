#!/usr/bin/env python3
"""Phase 20: pre-landmark feature provenance checks and grouped ablations.
No access to evaluation split. No claim of row-level timestamp verification from aggregated files.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

SOURCES = {
 'history':('b','history'), 'historical_dx_b':('b','history_diagnoses'),
 'prescriptions_b':('b','prescriptions'), 'emar_b':('b','emar'),
 'orders_b':('b','orders'), 'microbiology_b':('b','microbiology'),
 'historical_dx_c':('c','history_diagnoses'), 'prescriptions_c':('c','prescriptions'),
 'emar_c':('c','emar'), 'orders_c':('c','orders')}
SETS = {
 'baseline':['history'],
 'baseline_plus_diagnoses':['history','historical_dx_b','historical_dx_c'],
 'baseline_plus_prescriptions':['history','prescriptions_b','prescriptions_c'],
 'baseline_plus_emar':['history','emar_b','emar_c'],
 'baseline_plus_orders':['history','orders_b','orders_c'],
 'baseline_plus_microbiology':['history','microbiology_b'],
 'all_except_orders':[k for k in SOURCES if not k.startswith('orders')],
 'all_except_medications':[k for k in SOURCES if not (k.startswith('prescriptions') or k.startswith('emar'))],
 'all_except_administration':[k for k in SOURCES if not k.startswith('emar')],
 'all_except_microbiology':[k for k in SOURCES if not k.startswith('microbiology')],
 'all_except_history':[k for k in SOURCES if k!='history' and not k.startswith('historical_dx')],
 'full':list(SOURCES)}
CATS=['gender','admission_type','admission_location']
FORBIDDEN={'subject_id','hadm_id','split','first_event','event_hours','icu_24_72','icu_ever','icu_intime','dischtime','deathtime','outcome','label','target'}

def metrics(y,p):
 return {'n':int(len(y)),'positives':int(y.sum()),'prevalence':float(y.mean()),'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p))}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for arg in ('phase18','phase18b','phase18c','output'):ap.add_argument('--'+arg,required=True)
 ap.add_argument('--threads',type=int,default=2);ap.add_argument('--seed',type=int,default=42)
 ap.add_argument('--models',nargs='+',choices=['lightgbm','logistic','catboost'],default=['lightgbm'])
 ap.add_argument('--sets',nargs='+',choices=list(SETS),default=list(SETS))
 a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 features=pd.read_parquet(Path(a.phase18)/'prelandmark_features.parquet')
 events=pd.read_parquet(Path(a.phase18)/'first_events.parquet')
 for name,frame in [('features',features),('events',events)]:
  if frame.hadm_id.isna().any() or frame.hadm_id.duplicated().any():raise ValueError(name+' has missing/duplicate hadm_id')
 data=features.merge(events[['hadm_id','icu_24_72']],on='hadm_id',validate='one_to_one')
 if data.groupby('subject_id').split.nunique().max()!=1:raise ValueError('Patient-disjoint split violation')
 if set(data.split.unique())!={'train','early_stop','validation','evaluation'}:raise ValueError('Unexpected split labels')
 # Remove evaluation rows before loading any other features or fitting anything.
 data=data.loc[data.split!='evaluation'].copy()
 provenance=[]; groups={}
 for group,(kind,name) in SOURCES.items():
  file=Path(a.phase18b if kind=='b' else a.phase18c)/(name+'.parquet')
  if not file.exists():raise FileNotFoundError(f'Required module missing: {file}')
  x=pd.read_parquet(file)
  if 'hadm_id' not in x or x.hadm_id.duplicated().any():raise ValueError(f'Invalid hadm_id in {file}')
  columns=[]; dropped=[]
  for c in x.columns:
   if c=='hadm_id':continue
   if (c.lower() in FORBIDDEN or 'time' in c.lower() or 'date' in c.lower() or
       not pd.api.types.is_numeric_dtype(x[c])):dropped.append(c)
   else:columns.append(c)
  x=x[['hadm_id']+columns].rename(columns={c:f'{group}__{c}' for c in columns})
  marker=f'{group}__record_present';x[marker]=1
  data=data.merge(x,on='hadm_id',how='left',validate='one_to_one')
  data[marker]=data[marker].fillna(0).astype('int8')
  groups[group]=[f'{group}__{c}' for c in columns]+[marker]
  provenance.append({'group':group,'source':str(file),'sha256':sha(file),'rows':int(len(x)),'retained_columns':columns,'dropped_columns':dropped,'matched_development_rows':int(data[marker].sum()),'availability_verified_at_row_level':False})
  print(f'{group}: {len(x):,} source rows, {len(columns)} numeric features',flush=True)
 base_num=[c for c in features if c not in FORBIDDEN and c not in CATS and pd.api.types.is_numeric_dtype(features[c])]
 base_cats=[c for c in CATS if c in data]
 for c in base_num:
  if any(z in c.lower() for z in ('time','date','outcome','label','icu_')):raise ValueError(f'Potential unsafe baseline column: {c}')
 audit={'provenance':provenance,'base_numeric':base_num,'base_categorical':base_cats,
 'limitations':['Aggregated Phase 18B/18C tables do not contain source event timestamps. This script cannot certify temporal validity of individual source rows.',
 'Prescription starttime may not equal order availability time; inspect raw order creation/verification times.',
 'Order status and microbiology results may reflect post-landmark updates even when an earlier chart timestamp exists.',
 'Missing source records are distinguished using record_present flags, but imputed zeros remain a modeling assumption.',
 'Validation was used for exploratory comparisons; do not select on the reserved evaluation set.']}
 (out/'feature_audit.json').write_text(json.dumps(audit,indent=2))
 import lightgbm as lgb
 from sklearn.linear_model import LogisticRegression
 from scipy import sparse
 results=[];preds=None
 for setname in a.sets:
  selected=SETS[setname]
  numeric=base_num+[col for g in selected for col in groups[g]]
  # Keep feature selection fixed; fit all preprocessing only on train.
  train=data.split.eq('train');stop=data.split.eq('early_stop');val=data.split.eq('validation')
  transformer=ColumnTransformer([
   ('num',Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True))]),numeric),
   ('cat',OneHotEncoder(handle_unknown='ignore'),base_cats)],sparse_threshold=1.0)
  Xtr=transformer.fit_transform(data.loc[train,numeric+base_cats]);Xst=transformer.transform(data.loc[stop,numeric+base_cats]);Xv=transformer.transform(data.loc[val,numeric+base_cats])
  ytr=data.loc[train,'icu_24_72'].astype(int).to_numpy();yst=data.loc[stop,'icu_24_72'].astype(int).to_numpy();yv=data.loc[val,'icu_24_72'].astype(int).to_numpy()
  for modelname in a.models:
   if modelname=='lightgbm':
    model=lgb.LGBMClassifier(n_estimators=650,learning_rate=.035,num_leaves=24,max_depth=6,min_child_samples=90,reg_lambda=8,verbosity=-1,n_jobs=a.threads,random_state=a.seed)
    model.fit(Xtr,ytr,eval_set=[(Xst,yst)],eval_metric='average_precision',callbacks=[lgb.early_stopping(45,verbose=False)])
   elif modelname=='logistic':
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler(with_mean=False);Xtr2=scaler.fit_transform(Xtr);Xv2=scaler.transform(Xv)
    model=LogisticRegression(max_iter=500,solver='liblinear',random_state=a.seed)
    model.fit(Xtr2,ytr)
   else:
    from catboost import CatBoostClassifier
    model=CatBoostClassifier(iterations=650,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function='Logloss',eval_metric='PRAUC',thread_count=a.threads,random_seed=a.seed,verbose=False)
    model.fit(Xtr,ytr,eval_set=(Xst,yst),early_stopping_rounds=45,verbose=False)
   p=model.predict_proba(Xv2 if modelname=='logistic' else Xv)[:,1]
   result={'feature_set':setname,'model':modelname,'numeric_features':len(numeric),'categorical_features':len(base_cats),**metrics(yv,p)}
   results.append(result);print(result,flush=True)
   if preds is None:preds=pd.DataFrame({'hadm_id':data.loc[val,'hadm_id'].to_numpy(),'outcome':yv})
   preds[f'{setname}__{modelname}']=p
   pd.DataFrame(results).to_csv(out/'ablation_results.csv',index=False)
   preds.to_parquet(out/'validation_predictions.parquet',index=False)
 (out/'run_manifest.json').write_text(json.dumps({'n_development':int(len(data)),'n_validation':int((data.split=='validation').sum()),'evaluation_accessed':False,'seed':a.seed,'feature_sets':a.sets,'models':a.models,'notes':audit['limitations']},indent=2))
 print('COMPLETE:',out,flush=True)
if __name__=='__main__':main()
