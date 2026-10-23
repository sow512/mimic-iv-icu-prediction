#!/usr/bin/env python3
"""Phase 24: renal-history feature-budget screening, train/early-stop selection.
Uses Phase22B conservative + labs loader. Evaluation split is never loaded.
Feature ranking uses LightGBM gain on TRAIN only (no validation label access for selection).
Development validation is reported for transparency, not used to select budget.
"""
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load(path):
    spec=importlib.util.spec_from_file_location('phase22b_loader',str(path))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def metrics(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float)
    k=max(1,int(np.ceil(.1*len(y))))
    top=np.argsort(-p,kind='stable')[:k]
    return {'n':len(y),'events':int(y.sum()),'auroc':float(roc_auc_score(y,p)),
            'auprc':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p)),
            'recall_top10':float(y[top].sum()/max(1,y.sum())),
            'expected_events':float(p.sum())}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('phase18','phase18b','phase18c','labs','staged-labs','output'):
        ap.add_argument('--'+k,required=True)
    ap.add_argument('--phase22b-script',default=None)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--budgets',nargs='+',type=int,default=[10,25,50,100,200])
    ap.add_argument('--top-labs-only',action='store_true',help='Rank lab features separately from history features (experimental)')
    a=ap.parse_args()
    import lightgbm as lgb
    source=Path(a.phase22b_script) if a.phase22b_script else Path(__file__).with_name('phase22b_labs_retrain.py')
    if not source.exists():ap.error(f'Missing Phase22B loader: {source}; use --phase22b-script')
    out=Path(a.output).expanduser();out.mkdir(parents=True,exist_ok=True)
    d=load(source).load_data(a)
    renal_cols=[c for c in ('b_hist_dx_prior_renal','c_hist_renal_aki','c_hist_renal_ckd') if c in d]
    if not renal_cols:raise ValueError('Historical renal eligibility columns missing')
    mask=np.zeros(len(d),dtype=bool)
    for c in renal_cols:mask|=d[c].fillna(0).to_numpy(dtype=float)>0
    d=d.loc[mask].copy()
    if d.groupby('subject_id').split.nunique().max()!=1:raise ValueError('Patient overlap')
    cats=[c for c in ('gender','admission_type','admission_location') if c in d]
    exclude={'subject_id','hadm_id','split','icu_24_72','first_event','event_hours','landmark','landmark_time','admittime','dischtime','deathtime'}
    nums=[c for c in d if c not in exclude and c not in cats and pd.api.types.is_numeric_dtype(d[c])]
    allow={'b_hist_days_since_discharge'}
    bad=[c for c in nums if c not in allow and any(s in c.lower() for s in ('time','date','outcome','target','discharge','death'))]
    if bad:raise ValueError('Unsafe feature names: '+str(bad))
    tr=d.split.eq('train');st=d.split.eq('early_stop');va=d.split.eq('validation')
    if min(tr.sum(),st.sum(),va.sum())==0:raise ValueError('Empty split')
    ytr=d.loc[tr,'icu_24_72'].astype(int).to_numpy();yst=d.loc[st,'icu_24_72'].astype(int).to_numpy();yv=d.loc[va,'icu_24_72'].astype(int).to_numpy()
    if min(ytr.sum(),yst.sum(),yv.sum())==0:raise ValueError('Split without positive outcomes')
    # Numeric missing values are preserved as NaN for LightGBM; categorical encoding is train-fit.
    for c in nums:d[c]=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan).astype('float32')
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    enc.fit(d.loc[tr,cats]); cat_names=enc.get_feature_names_out(cats).tolist()
    catX={s:enc.transform(d.loc[m,cats]).astype('float32') for s,m in [('train',tr),('early_stop',st),('validation',va)]}
    joblib.dump(enc,out/'categorical_encoder.joblib')
    def matrix(cols,split):
        z=d.loc[{'train':tr,'early_stop':st,'validation':va}[split],cols].to_numpy(dtype='float32')
        return np.column_stack((z,catX[split]))
    def fit(cols,seed):
        model=lgb.LGBMClassifier(n_estimators=700,learning_rate=.035,num_leaves=24,max_depth=6,
                min_child_samples=90,reg_lambda=8,verbosity=-1,n_jobs=a.threads,random_state=seed)
        model.fit(matrix(cols,'train'),ytr,eval_set=[(matrix(cols,'early_stop'),yst)],
                  eval_metric='average_precision',callbacks=[lgb.early_stopping(50,verbose=False)])
        return model
    print('Training ranking model on',len(nums),'numeric +',len(cat_names),'encoded categorical features',flush=True)
    ranker=fit(nums,a.seed)
    gains=ranker.booster_.feature_importance(importance_type='gain')[:len(nums)]
    rank=pd.DataFrame({'feature':nums,'gain':gains}).sort_values(['gain','feature'],ascending=[False,True])
    rank['rank']=np.arange(1,len(rank)+1)
    rank.to_csv(out/'feature_importance_train.csv',index=False)
    joblib.dump(ranker,out/'ranking_model.joblib')
    candidates=[('full',nums)]
    for k in sorted(set(a.budgets)):
        if k<=0:raise ValueError('Feature budgets must be positive')
        if k<len(nums):candidates.append((f'top_{k}',rank.feature.head(k).tolist()))
    records=[];preds=pd.DataFrame({'subject_id':d.loc[va,'subject_id'].to_numpy(),
                                  'hadm_id':d.loc[va,'hadm_id'].to_numpy(),'outcome':yv})
    for name,cols in candidates:
        start=time.time();model=ranker if name=='full' else fit(cols,a.seed)
        stop_p=model.predict_proba(matrix(cols,'early_stop'))[:,1]
        val_p=model.predict_proba(matrix(cols,'validation'))[:,1]
        row={'model':name,'numeric_features':len(cols),'encoded_features':len(cols)+len(cat_names),
             'early_stop':metrics(yst,stop_p),'validation':metrics(yv,val_p),
             'training_seconds':round(time.time()-start,2)}
        records.append(row);preds[name]=val_p
        joblib.dump({'model':model,'numeric_columns':cols,'categorical_columns':cats,
                     'categorical_encoder':enc,'encoded_categorical_columns':cat_names},out/f'{name}.joblib')
        (out/f'{name}_features.json').write_text(json.dumps(cols,indent=2))
        print(name,'stop AUPRC',round(row['early_stop']['auprc'],4),
              'val AUPRC',round(row['validation']['auprc'],4),flush=True)
    # Selection is based ONLY on early_stop AUPRC; 98% tolerance to best score.
    best=max(r['early_stop']['auprc'] for r in records)
    viable=[r for r in records if r['early_stop']['auprc']>=.98*best]
    chosen=min(viable,key=lambda r:(r['numeric_features'],-r['early_stop']['auprc']))
    preds.to_parquet(out/'validation_predictions.parquet',index=False)
    report={'selection_rule':'Smallest numeric budget with early-stop AUPRC >=98% of best; validation not used for selection',
            'selected':chosen['model'],'selected_numeric_features':chosen['numeric_features'],
            'n_renal_admissions':len(d),'renal_columns':renal_cols,'n_validation':int(va.sum()),
            'n_validation_patients':int(d.loc[va,'subject_id'].nunique()),'models':records,
            'limitations':['Exploratory validation has been repeatedly inspected in earlier phases.',
            'Importance ranking is computed using training labels, and early-stop outcomes select the budget.',
            'Correlated features may divide gain importance; feature selection is not a causal analysis.',
            'Historical ICD availability and cached laboratory provenance remain incompletely verified.',
            'Evaluation split never loaded; feature selection uses development data only.',
            'This experiment benchmarks LightGBM feature budgets, not yet neural compression.']}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print('SELECTED:',chosen['model'],'features',chosen['numeric_features'],'output',out,flush=True)
if __name__=='__main__':main()
