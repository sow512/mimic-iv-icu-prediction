#!/usr/bin/env python3
"""MIMIC-IV renal ICU architecture x selection x imbalance benchmark.
Research only. One patient-disjoint selection/threshold/evaluation split.
No claim of clinical ICU need or treatment benefit. Python >=3.10.
"""
import argparse, json, os, time, traceback, warnings, gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    brier_score_loss, log_loss)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier

SEED=42
ARCHS=('logistic','xgboost','ebm','random_forest','catboost','lightgbm','mlp')
SELECTORS=('full','mutual_information','l1_logistic','lightgbm_importance')
BALANCERS=('none','class_weight','oversample','undersample')
COUNTS=(20,40,80,160)


def read_data(p1,p2):
    base=pd.read_parquet(p1/'phase1_landmark_predictors.parquet')
    labs=pd.read_parquet(p2/'phase2_temporal_lab_features.parquet')
    labels=pd.read_parquet(p1/'phase1_labels.parquet')
    for name,df in [('base',base),('labs',labs),('labels',labels)]:
        if df.hadm_id.isna().any() or df.hadm_id.duplicated().any():
            raise ValueError(f'{name}: invalid/duplicate hadm_id')
    labs=labs.drop(columns=['subject_id','landmark_time'],errors='ignore')
    labs=labs.drop(columns=list((set(labs.columns)&set(base.columns)) - {'hadm_id'}))
    d=base.merge(labs,on='hadm_id',how='left',validate='one_to_one')
    d=d.merge(labels[['subject_id','hadm_id','icu_transfer_24_72h']],
              on=['subject_id','hadm_id'],how='inner',validate='one_to_one')
    if len(d)!=len(base) or len(d)!=len(labels): raise ValueError('Row count mismatch')
    banned=['subject_id','hadm_id','landmark_time','icu_transfer_24_72h']
    X=d.drop(columns=banned)
    tokens=('icu_transfer','icu_intime','dischtime','deathtime','horizon_end','label_status','hours_to_icu','first_icu')
    if any(any(t in c.lower() for t in tokens) for c in X): raise ValueError('Suspected leakage column')
    y=d.icu_transfer_24_72h.astype(int)
    if set(y.unique())!={0,1}: raise ValueError('Nonbinary outcome')
    return d,X,y


def grouped_splits(d):
    ix=np.arange(len(d)); groups=d.subject_id
    outer=GroupShuffleSplit(n_splits=1,test_size=.10,random_state=SEED)
    development,test=next(outer.split(ix,groups=groups))
    middle=GroupShuffleSplit(n_splits=1,test_size=1/3,random_state=SEED+1)
    a,b=next(middle.split(development,groups=groups.iloc[development]))
    train=development[a]; remaining=development[b]
    inner=GroupShuffleSplit(n_splits=1,test_size=.5,random_state=SEED+2)
    a,b=next(inner.split(remaining,groups=groups.iloc[remaining]))
    s={'train':train,'selection':remaining[a], 'threshold':remaining[b],'test':test}
    for k,ix1 in s.items():
        if len(ix1)==0: raise ValueError('Empty split')
        for k2,ix2 in s.items():
            if k<k2 and set(groups.iloc[ix1]) & set(groups.iloc[ix2]): raise AssertionError('Patient leakage')
    return s


def preprocess(X,train):
    cats=X.select_dtypes(include=['object','string','category']).columns.tolist()
    nums=[c for c in X if c not in cats]
    raw=X.copy()
    for c in nums: raw[c]=pd.to_numeric(raw[c],errors='raise').replace([np.inf,-np.inf],np.nan)
    keep_nums=[c for c in nums if raw.iloc[train][c].notna().mean()>=.02]
    keep_cats=[c for c in cats if raw.iloc[train][c].notna().any()]
    imp=SimpleImputer(strategy='median')
    z=imp.fit_transform(raw.iloc[train][keep_nums]); zn=imp.transform(raw[keep_nums])
    scaler=StandardScaler().fit(z)
    parts=[pd.DataFrame(scaler.transform(zn).astype('float32'),columns=keep_nums,index=raw.index)]
    if keep_cats:
        ct=raw.iloc[train][keep_cats].astype('string').fillna('__MISSING__').astype(str)
        ca=raw[keep_cats].astype('string').fillna('__MISSING__').astype(str)
        enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
        enc.fit(ct)
        parts.append(pd.DataFrame(enc.transform(ca),columns=enc.get_feature_names_out(keep_cats),index=raw.index))
    out=pd.concat(parts,axis=1)
    if out.columns.duplicated().any() or out.isna().any().any(): raise ValueError('Bad feature matrix')
    return out


def rank_features(X,y,method,threads):
    if method=='mutual_information':
        score=mutual_info_classif(X,y,random_state=SEED,n_neighbors=3)
    elif method=='l1_logistic':
        m=LogisticRegression(penalty='l1',solver='saga',C=.05,max_iter=2500,
                             class_weight='balanced',random_state=SEED,n_jobs=threads)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');m.fit(X,y)
        score=np.abs(m.coef_[0])
    elif method=='lightgbm_importance':
        m=LGBMClassifier(n_estimators=250,learning_rate=.04,num_leaves=20,
                          min_child_samples=80,reg_lambda=5,random_state=SEED,
                          n_jobs=threads,verbosity=-1)
        m.fit(X,y);score=m.feature_importances_
    else: raise ValueError(method)
    return pd.DataFrame({'feature':X.columns,'score':score}).sort_values(
        ['score','feature'],ascending=[False,True],kind='mergesort').reset_index(drop=True)


def build_model(name,weighted,threads):
    if name=='logistic':
        return LogisticRegression(max_iter=1500,class_weight='balanced' if weighted else None,
                                  random_state=SEED)
    if name=='lightgbm':
        return LGBMClassifier(n_estimators=300,learning_rate=.04,num_leaves=20,
              min_child_samples=80,reg_lambda=5,class_weight='balanced' if weighted else None,
              random_state=SEED,n_jobs=threads,verbosity=-1)
    if name=='catboost':
        return CatBoostClassifier(iterations=400,depth=5,learning_rate=.04,l2_leaf_reg=6,
              auto_class_weights='Balanced' if weighted else None,random_seed=SEED,
              thread_count=threads,verbose=False,allow_writing_files=False)
    if name=='xgboost':
        return XGBClassifier(n_estimators=350,max_depth=4,learning_rate=.04,subsample=.85,
              colsample_bytree=.85,reg_lambda=5,tree_method='hist',eval_metric='logloss',
              random_state=SEED,n_jobs=threads)
    if name=='random_forest':
        return RandomForestClassifier(n_estimators=250,min_samples_leaf=12,max_features='sqrt',
              class_weight='balanced_subsample' if weighted else None,
              random_state=SEED,n_jobs=threads)
    if name=='mlp':
        return MLPClassifier(hidden_layer_sizes=(64,32),alpha=.01,max_iter=180,
              early_stopping=False,random_state=SEED,batch_size=256)
    if name=='ebm':
        return ExplainableBoostingClassifier(interactions=0,max_rounds=250,
              learning_rate=.04,outer_bags=4,inner_bags=0,n_jobs=threads,random_state=SEED)
    raise ValueError(name)


def weighted_sample_weights(y):
    counts=np.bincount(np.asarray(y,dtype=int),minlength=2)
    return np.asarray([len(y)/(2*counts[int(v)]) for v in y],dtype=float)


def resample(X,y,method):
    rng=np.random.default_rng(SEED)
    yy=np.asarray(y,dtype=int)
    classes=[np.flatnonzero(yy==i) for i in (0,1)]
    if not len(classes[0]) or not len(classes[1]): raise ValueError('Missing class')
    n=max(map(len,classes)) if method=='oversample' else min(map(len,classes))
    chosen=np.concatenate([rng.choice(ix,size=n,replace=len(ix)<n) for ix in classes])
    rng.shuffle(chosen)
    return X.iloc[chosen],y.iloc[chosen]


def fit_model(architecture,balancer,X,y,threads):
    weighted=balancer=='class_weight'
    m=build_model(architecture,weighted,threads)
    xx,yy=X,y
    if balancer in ('oversample','undersample'):xx,yy=resample(X,y,balancer)
    if weighted and architecture in ('xgboost','mlp','ebm'):
        # XGBoost, EBM and recent sklearn MLP accept sample_weight in fit;
        # compatibility errors are logged rather than silently using unweighted fit.
        m.fit(xx,yy,sample_weight=weighted_sample_weights(yy))
    else:
        m.fit(xx,yy)
    return m


def metrics(y,p,t):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    pred=(p>=t).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),
        accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        sensitivity=float(recall_score(y,pred,zero_division=0)),specificity=float(tn/(tn+fp)),
        precision=float(precision_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
        brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,np.clip(p,1e-8,1-1e-8))),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def threshold_for_balanced_accuracy(y,p):
    ts=np.unique(np.r_[np.linspace(.01,.99,99),np.quantile(p,np.linspace(.01,.99,99))])
    scores=[(balanced_accuracy_score(y,p>=t),-abs(t-.5),float(t)) for t in ts]
    return max(scores)[2]


def atomic_csv(df,path):
    tmp=path.with_name(path.name+'.tmp');df.to_csv(tmp,index=False);os.replace(tmp,path)


def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    data,raw,y=read_data(Path(a.phase1),Path(a.phase2))
    splits=grouped_splits(data)
    X=preprocess(raw,splits['train'])
    train,select,threshold,test=[splits[k] for k in ('train','selection','threshold','test')]
    manifest={'seed':SEED,'features':len(X.columns),'splits':{k:{'admissions':len(ix),
        'patients':int(data.iloc[ix].subject_id.nunique()),'positives':int(y.iloc[ix].sum())}
        for k,ix in splits.items()},'limitations':['Retrospective ICD-defined renal cohort',
        'Observed transfer is not ICU clinical need','Prior MIMIC-IV results have been inspected; internal benchmark only']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    pd.DataFrame([{'subject_id':int(s),'hadm_id':int(h),'split':name}
        for name,ix in splits.items() for s,h in zip(data.iloc[ix].subject_id,data.iloc[ix].hadm_id)]).to_csv(out/'splits.csv',index=False)
    configs=[('full',len(X.columns),X.columns.tolist())]
    for method in SELECTORS[1:]:
        path=out/f'ranking_{method}.csv'
        if path.exists() and a.resume:
            r=pd.read_csv(path)
            if set(r.feature)!=set(X.columns):raise ValueError(f'Ranking mismatch: {path}')
        else:
            print('Ranking:',method,flush=True)
            r=rank_features(X.iloc[train],y.iloc[train],method,a.threads)
            atomic_csv(r,path)
        for n in COUNTS:configs.append((method,min(n,len(X.columns)),r.feature.iloc[:n].tolist()))
    selection_path=out/'selection_results.csv'
    if selection_path.exists() and a.resume:
        existing=pd.read_csv(selection_path).to_dict('records')
    else:existing=[]
    done={str(r['run_id']) for r in existing}
    total=len(configs)*len(ARCHS)*len(BALANCERS)
    counter=0
    for method,n,cols in configs:
        for arch in ARCHS:
            if a.architectures and arch not in a.architectures:continue
            for bal in BALANCERS:
                if a.balancers and bal not in a.balancers:continue
                counter+=1
                key=f'{method}__{n}__{arch}__{bal}'
                if key in done:continue
                print(f'[{counter}/{total}] {key}',flush=True)
                record={'run_id':key,'selector':method,'n_features':n,
                        'architecture':arch,'imbalance':bal,'status':'ok'}
                t0=time.time()
                try:
                    model=fit_model(arch,bal,X.iloc[train][cols],y.iloc[train],a.threads)
                    p=model.predict_proba(X.iloc[select][cols])[:,1]
                    record.update({'selection_'+k:v for k,v in metrics(y.iloc[select],p,.5).items()})
                except Exception as e:
                    record.update(status='failed',error=f'{type(e).__name__}: {e}')
                    with (out/'failures.log').open('a') as f:f.write(key+'\n'+traceback.format_exc()+'\n')
                    if a.fail_fast:raise
                record['seconds']=round(time.time()-t0,2)
                existing.append(record);done.add(key)
                atomic_csv(pd.DataFrame(existing),selection_path)
                gc.collect()
    results=pd.DataFrame(existing)
    good=results[results.status=='ok'].copy()
    if good.empty:raise RuntimeError('No successful runs; inspect failures.log')
    good=good.sort_values(['selection_auprc','selection_auroc'],ascending=False)
    atomic_csv(good,out/'selection_ranked.csv')
    print('\nTOP VALIDATION CONFIGURATIONS:')
    print(good[['run_id','selection_auroc','selection_auprc']].head(15).to_string(index=False))
    if not a.finalize:
        print('\nSelection complete. Inspect selection_ranked.csv, then rerun with --resume --finalize.')
        return
    # Freeze selection before touching test; finalists chosen on selection validation only.
    finalists=good.head(a.top_k).copy()
    frozen=out/'finalists.json'
    ids=finalists.run_id.tolist()
    if frozen.exists():
        prior=json.loads(frozen.read_text())
        if prior!=ids:raise ValueError('Frozen finalists differ; use a new output directory')
    else:frozen.write_text(json.dumps(ids,indent=2))
    final_path=out/'final_evaluation.csv'
    final_rows=pd.read_csv(final_path).to_dict('records') if final_path.exists() and a.resume else []
    completed={r['run_id'] for r in final_rows}
    for row in finalists.to_dict('records'):
        key=row['run_id']
        if key in completed:continue
        method,n,arch,bal=row['selector'],int(row['n_features']),row['architecture'],row['imbalance']
        cols=X.columns.tolist() if method=='full' else pd.read_csv(out/f'ranking_{method}.csv').feature.iloc[:n].tolist()
        print('Finalist:',key,flush=True)
        model=fit_model(arch,bal,X.iloc[train][cols],y.iloc[train],a.threads)
        pv=model.predict_proba(X.iloc[threshold][cols])[:,1]
        cut=threshold_for_balanced_accuracy(y.iloc[threshold],pv)
        pt=model.predict_proba(X.iloc[test][cols])[:,1]
        record={'run_id':key,'selector':method,'n_features':n,'architecture':arch,
                'imbalance':bal,'threshold':cut,'status':'ok',**metrics(y.iloc[test],pt,cut)}
        final_rows.append(record);atomic_csv(pd.DataFrame(final_rows),final_path)
        gc.collect()
    print('\nFINALISTS TEST RESULTS (do not reselect using test):')
    print(pd.DataFrame(final_rows)[['run_id','auroc','auprc','balanced_accuracy','sensitivity']].to_string(index=False))
    print('Saved to',out)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase1',required=True);ap.add_argument('--phase2',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--finalize',action='store_true')
    ap.add_argument('--top-k',type=int,default=7)
    ap.add_argument('--fail-fast',action='store_true')
    ap.add_argument('--architectures',nargs='*',choices=ARCHS)
    ap.add_argument('--balancers',nargs='*',choices=BALANCERS)
    main(ap.parse_args())
