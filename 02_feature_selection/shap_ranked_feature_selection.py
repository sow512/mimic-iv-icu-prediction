#!/usr/bin/env python3
"""Train-only TreeSHAP selection + prelandmark feature engineering; matched binary/competing neural benchmarks.
Three evaluation tracks: hospital->hospital, hospital->renal, renal->renal; both objectives.
Development only. Source provenance NOT independently certified. No reserved evaluation rows.
"""
import argparse, importlib.util, json, random, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def loadmod(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def engineered(d):
    """Only transformations of prelandmark aggregates. No post-landmark values."""
    groups={}
    for col in d:
        match=re.fullmatch(r'(lab_\d+_[a-f0-9]+)_(firstval|lastval|minval|maxval|mean|n)',col)
        if match:groups.setdefault(match.group(1),{})[match.group(2)]=col
    created=[]
    # Concatenate once to avoid pandas fragmentation.
    out={}
    for key,g in groups.items():
        if 'firstval' in g and 'lastval' in g:
            first=pd.to_numeric(d[g['firstval']],errors='coerce');last=pd.to_numeric(d[g['lastval']],errors='coerce')
            out[key+'_delta']=((last-first).replace([np.inf,-np.inf],np.nan)).astype('float32')
            out[key+'_relative_delta']=((last-first)/(first.abs()+1.0)).replace([np.inf,-np.inf],np.nan).astype('float32')
        if 'minval' in g and 'maxval' in g:
            low=pd.to_numeric(d[g['minval']],errors='coerce');high=pd.to_numeric(d[g['maxval']],errors='coerce')
            out[key+'_range']=(high-low).replace([np.inf,-np.inf],np.nan).astype('float32')
    if out:
        created=list(out)
        d=pd.concat([d,pd.DataFrame(out,index=d.index)],axis=1)
    return d,created

def score(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float)
    if len(np.unique(y))<2:return {'n':len(y),'events':int(y.sum()),'error':'single class'}
    k=max(1,int(np.ceil(.1*len(y))));idx=np.argsort(-p,kind='stable')[:k]
    return {'n':int(len(y)),'events':int(y.sum()),'auroc':float(roc_auc_score(y,p)),
            'auprc':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),
            'recall_top10':float(y[idx].sum()/y.sum()),'expected_events':float(p.sum())}

def transform(d,nums,cats,train_mask):
    prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value=0)),('scale',StandardScaler())]),nums),
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)],sparse_threshold=0)
    tr=prep.fit_transform(d.loc[train_mask,nums+cats])
    return prep,np.nan_to_num(np.asarray(tr,dtype=np.float32),nan=0,posinf=0,neginf=0)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('phase18','phase18b','phase18c','labs','staged-labs','output'):ap.add_argument('--'+key,required=True)
    ap.add_argument('--script-dir',default=str(Path(__file__).resolve().parent))
    ap.add_argument('--policy',choices=['full','conservative'],default='full')
    ap.add_argument('--budgets',type=int,nargs='+',default=[25,50,100,200])
    ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--hidden',type=int,default=128);ap.add_argument('--dropout',type=float,default=.2)
    ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--weight-decay',type=float,default=.001)
    ap.add_argument('--batch-size',type=int,default=1024);ap.add_argument('--patience',type=int,default=8)
    ap.add_argument('--shap-sample',type=int,default=3000);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--horizon-hours',type=int,default=48);ap.add_argument('--interval-hours',type=int,default=12)
    a=ap.parse_args()
    if a.horizon_hours!=48 or a.interval_hours!=12:ap.error('Expected 48h after 24h landmark, 12h intervals')
    out=Path(a.output).expanduser();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads);np.random.seed(a.seed);random.seed(a.seed);torch.manual_seed(a.seed)
    base=Path(a.script_dir)
    src=loadmod(base/('phase21_competing_risk.py' if a.policy=='full' else 'phase22b_labs_retrain.py'),'source')
    binary=loadmod(base/'phase23c_hospitalwide_binary.py','binary_source')
    d=src.load_data(a)
    if d.hadm_id.duplicated().any() or d.groupby('subject_id').split.nunique().gt(1).any():raise ValueError('Patient split violation')
    renal_cols=['b_hist_dx_prior_renal','c_hist_renal_aki','c_hist_renal_ckd']
    if not set(renal_cols).issubset(d):raise ValueError('Missing renal history columns')
    d['renal_history']=d[renal_cols].fillna(0).gt(0).any(axis=1)
    d,created=engineered(d)
    cats=[c for c in ['gender','admission_type','admission_location'] if c in d]
    excluded={'subject_id','hadm_id','split','icu_24_72','first_event','event_hours','landmark','landmark_time','admittime','dischtime','deathtime','renal_history'}
    nums=[c for c in d if c not in excluded and c not in cats and pd.api.types.is_numeric_dtype(d[c])]
    allowed={'b_hist_days_since_discharge'}
    suspect=[c for c in nums if c not in allowed and any(z in c.lower() for z in ('time','date','outcome','target','discharge','death'))]
    if suspect:raise ValueError('Unsafe column names: '+str(suspect))
    for c in nums:d[c]=pd.to_numeric(d[c],errors='coerce').astype('float32')
    from lightgbm import LGBMClassifier
    results={};saved=[]
    for population in ('hospital','renal'):
        folder=out/population;folder.mkdir(exist_ok=True)
        eligible=np.ones(len(d),dtype=bool) if population=='hospital' else d.renal_history.to_numpy(dtype=bool)
        masks={s:(eligible & d.split.eq(s).to_numpy()) for s in ('train','early_stop','validation')}
        y={s:d.loc[masks[s],'icu_24_72'].to_numpy(dtype=int) for s in masks}
        print('\nPOPULATION',population, {s:(len(y[s]),int(y[s].sum())) for s in y},flush=True)
        if any(len(y[s])==0 or len(np.unique(y[s]))<2 for s in y):raise ValueError('Insufficient cases')
        # TreeSHAP rankings are calculated on TRAIN ONLY; no validation/stop labels used for selection ranking.
        # Raw numeric columns only, categorical features always retained in neural fits.
        tr=d.loc[masks['train'],nums].replace([np.inf,-np.inf],np.nan).fillna(0)
        ranking=LGBMClassifier(n_estimators=180,learning_rate=.045,num_leaves=20,max_depth=5,
                               min_child_samples=80,reg_lambda=8,verbosity=-1,n_jobs=a.threads,random_state=a.seed)
        ranking.fit(tr,y['train'])
        rng=np.random.default_rng(a.seed)
        sample=rng.choice(len(tr),min(len(tr),a.shap_sample),replace=False)
        contrib=ranking.booster_.predict(tr.iloc[sample],pred_contrib=True)
        # LightGBM binary pred_contrib: final column is expected value, not a feature.
        if isinstance(contrib,list):contrib=contrib[-1]
        if contrib.shape[1]!=len(nums)+1:raise ValueError('Unexpected TreeSHAP contribution shape')
        importance=np.abs(contrib[:,:-1]).mean(axis=0)
        rank=pd.DataFrame({'feature':nums,'mean_abs_treeshap':importance}).sort_values('mean_abs_treeshap',ascending=False)
        rank.to_csv(folder/'train_treeshap_ranking.csv',index=False)
        joblib.dump(ranking,folder/'ranking_lightgbm.joblib')
        ordered=rank.feature.tolist()
        budgets=sorted(set([min(len(nums),b) for b in a.budgets if b>0]+[len(nums)]))
        # Select budget on EARLY_STOP only, not validation. The neural models are trained AFTER budget selection.
        selection=[]
        for budget in budgets:
            columns=ordered[:budget]
            tree=LGBMClassifier(n_estimators=220,learning_rate=.04,num_leaves=20,max_depth=5,
                               min_child_samples=80,reg_lambda=8,verbosity=-1,n_jobs=a.threads,random_state=a.seed)
            tree.fit(d.loc[masks['train'],columns].fillna(0),y['train'])
            p=tree.predict_proba(d.loc[masks['early_stop'],columns].fillna(0))[:,1]
            value=float(average_precision_score(y['early_stop'],p))
            selection.append({'budget':budget,'early_stop_auprc':value})
            print(population,'budget',budget,'stop AUPRC',round(value,5),flush=True)
        pd.DataFrame(selection).to_csv(folder/'budget_selection_early_stop.csv',index=False)
        best=sorted(selection,key=lambda x:(-x['early_stop_auprc'],x['budget']))[0]['budget']
        chosen=ordered[:best]
        (folder/'selected_features.json').write_text(json.dumps({'budget':best,'numeric_features':chosen,'categorical_features':cats,'engineered_candidates':created,'selection':'TreeSHAP ranked on train; budget chosen by early_stop LightGBM AUPRC'},indent=2))
        pre,Xtrain=transform(d,chosen,cats,masks['train'])
        X={s:(Xtrain if s=='train' else np.nan_to_num(np.asarray(pre.transform(d.loc[masks[s],chosen+cats]),dtype=np.float32),nan=0,posinf=0,neginf=0)) for s in masks}
        joblib.dump({'preprocessor':pre,'numeric_columns':chosen,'categorical_columns':cats},folder/'preprocessing.joblib')
        # Train binary network and competing-risk network on IDENTICAL selected features and population.
        torch.manual_seed(a.seed)
        binmodel,bininfo=binary.train(X['train'],y['train'],X['early_stop'],y['early_stop'],a,folder/'binary_best.pt')
        pbin=binary.infer(binmodel,X['validation'])
        Y={};M={}
        for s in masks:Y[s],M[s]=src.make_targets(d.loc[masks[s]],4,12)
        torch.manual_seed(a.seed)
        riskmodel,riskinfo=src.train_model('neural',X['train'],Y['train'],M['train'],X['early_stop'],Y['early_stop'],M['early_stop'],a,folder)
        pcomp=src.predict(riskmodel,X['validation'])[0][:,-1]
        val=d.loc[masks['validation'],['subject_id','hadm_id','icu_24_72','renal_history']].copy()
        val=val.rename(columns={'icu_24_72':'outcome'})
        val['binary']=pbin;val['competing_risk']=pcomp
        val.to_parquet(folder/'validation_predictions.parquet',index=False)
        evaluations={}
        for target in (['hospital','renal'] if population=='hospital' else ['renal']):
            v=val if target=='hospital' else val[val.renal_history]
            for model in ('binary','competing_risk'):
                evaluations[f'{population}_trained_{model}_on_{target}']=score(v.outcome,v[model])
        report={'population':population,'policy':a.policy,'selected_numeric_features':best,'encoded_features':Xtrain.shape[1],
                'binary_training':bininfo,'competing_training':riskinfo,'metrics':evaluations,
                'warnings':['TreeSHAP explains a LightGBM surrogate/ranker, NOT the neural models.',
                            'Historical diagnosis and cached lab provenance not independently certified.',
                            'Development validation repeatedly inspected; reserved evaluation untouched.',
                            'Renal history is not equivalent to active AKI/CKD at the landmark.',
                            'Feature engineering uses prelandmark summaries; no new raw-event timestamps audited.']}
        (folder/'report.json').write_text(json.dumps(report,indent=2));results.update(evaluations)
        print(json.dumps(report,indent=2),flush=True)
    (out/'summary.json').write_text(json.dumps(results,indent=2))
    print('\nCOMPLETE:',out,flush=True)
if __name__=='__main__':main()
