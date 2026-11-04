#!/usr/bin/env python3
"""Renal ICU 24-72h master research benchmark (MIMIC-IV 3.1).

Reuses Phase 1/2/9A/9B/9C. Adds renal-physiology interactions,
feature-group ablations, LightGBM/CatBoost seed ensembles, validation-only
blend and threshold selection, and optional one-time evaluation.

No raw ICU chartevents, no synthetic ward vitals, no test-set optimization.
Research use only. Never use for clinical decisions.
"""
import argparse, hashlib, json, os, re, sys, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, balanced_accuracy_score, confusion_matrix)

TARGET='icu_transfer_24_72h'
SPLITS=('train','early_stop','blend','threshold','evaluation')
FORBIDDEN=('icu_transfer','icu_intime','first_icu','dischtime','deathtime','horizon_end','label_status','hours_to_icu')
SEED=20261022


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def dump_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,default=lambda x:float(x) if isinstance(x,np.floating) else int(x) if isinstance(x,np.integer) else str(x)))
    os.replace(tmp,path)


def load(args):
    p1,p2,p9=map(Path,(args.phase1,args.phase2,args.phase9))
    base=pd.read_parquet(p1/'phase1_landmark_predictors.parquet')
    labels=pd.read_parquet(p1/'phase1_labels.parquet')
    if base.hadm_id.duplicated().any() or labels.hadm_id.duplicated().any():
        raise ValueError('Duplicate hadm_id in baseline or labels')
    data=base.copy()
    groups={}
    sources={
      'labs':p2/'phase2_temporal_lab_features.parquet',
      'history':p9/'phase9a_history_omr_features.parquet',
      'medcom':p9/'phase9b_medication_comorbidity_features.parquet'}
    for name,path in sources.items():
        t=pd.read_parquet(path)
        if t.hadm_id.isna().any() or t.hadm_id.duplicated().any():
            raise ValueError(f'{name}: invalid admission identifiers')
        if 'subject_id' in t:
            chk=base[['hadm_id','subject_id']].merge(t[['hadm_id','subject_id']],on='hadm_id',how='left',suffixes=('_base','_source'),validate='one_to_one')
            if ((chk.subject_id_source.notna()) & (chk.subject_id_base!=chk.subject_id_source)).any():
                raise ValueError(f'{name}: patient linkage mismatch')
        t=t.drop(columns=['subject_id','landmark_time'],errors='ignore')
        t=t.drop(columns=[c for c in t if c!='hadm_id' and c in data],errors='ignore')
        bad=[c for c in t if c!='hadm_id' and any(k in c.lower() for k in FORBIDDEN)]
        if bad: raise ValueError(f'{name}: suspicious predictors {bad}')
        groups[name]=[c for c in t if c!='hadm_id']
        data=data.merge(t,on='hadm_id',how='left',validate='one_to_one')
    data=data.merge(labels[['subject_id','hadm_id',TARGET]],on=['subject_id','hadm_id'],how='inner',validate='one_to_one')
    if len(data)!=len(base) or len(data)!=len(labels): raise ValueError('Cohort row mismatch')
    splitfile=Path(args.phase9c)/'split_assignments.csv'
    sp=pd.read_csv(splitfile)
    if not {'hadm_id','subject_id','split'}.issubset(sp): raise ValueError('Invalid split file')
    if sp.hadm_id.duplicated().any() or len(sp)!=len(data): raise ValueError('Invalid split assignments')
    linked=data[['hadm_id','subject_id']].merge(sp,on=['hadm_id','subject_id'],how='left',validate='one_to_one',sort=False)
    if linked.split.isna().any() or set(linked.split)!=set(SPLITS): raise ValueError('Split mismatch')
    if linked.groupby('subject_id').split.nunique().gt(1).any(): raise ValueError('Patient leakage across splits')
    ix={s:np.flatnonzero(linked.split.to_numpy()==s) for s in SPLITS}
    y=data[TARGET].astype(int).to_numpy()
    for name,idx in ix.items():
        if np.unique(y[idx]).size!=2: raise ValueError(f'{name}: missing outcome class')
    groups['baseline']=[c for c in base if c not in ('subject_id','hadm_id','landmark_time')]
    return data,groups,ix,y,splitfile,sources


def safe_ratio(a,b):
    return a/b.where(b.abs()>1e-8)


def advanced_renal_features(data, available):
    """Only transformations of previously time-filtered lab features.
    Never interpolate raw future measurements or use outcome columns.
    """
    added={}
    def col(name):
        return pd.to_numeric(data[name],errors='coerce') if name in available else None
    for suffix in ('overall_latest','overall_first','overall_mean'):
        cr,bun=col('creatinine_'+suffix),col('bun_'+suffix)
        if cr is not None and bun is not None:
            added['renal_bun_creatinine_ratio_'+suffix]=safe_ratio(bun,cr)
        na,cl,hco3=col('sodium_'+suffix),col('chloride_'+suffix),col('bicarbonate_'+suffix)
        if all(z is not None for z in (na,cl,hco3)):
            added['renal_calculated_anion_gap_'+suffix]=na-cl-hco3
        ca,phos=col('calcium_'+suffix),col('phosphate_'+suffix)
        if ca is not None and phos is not None:
            added['renal_calcium_phosphate_product_'+suffix]=ca*phos
    for lab,low,high in [('potassium',3.5,5.5),('sodium',135,145),('bicarbonate',22,29),('creatinine',0.6,1.3),('lactate',0.5,2.0)]:
        v=col(lab+'_overall_latest')
        if v is not None:
            added['renal_'+lab+'_below_reference']=np.where(v.notna(),(v<low).astype(float),np.nan)
            added['renal_'+lab+'_above_reference']=np.where(v.notna(),(v>high).astype(float),np.nan)
            added['renal_'+lab+'_distance_from_reference']=np.where(v.notna(),np.maximum(low-v,np.maximum(v-high,0)),np.nan)
    for lab in ('creatinine','bun','potassium','bicarbonate','lactate'):
        v=col(lab+'_overall_delta')
        if v is not None:
            added['renal_'+lab+'_absolute_change']=v.abs()
        first,last=col(lab+'_overall_first'),col(lab+'_overall_latest')
        if first is not None and last is not None:
            added['renal_'+lab+'_relative_change']=safe_ratio(last-first,first.abs())
    if not added:
        print('WARNING: no advanced features constructed; check Phase 2 column names',flush=True)
        return pd.DataFrame(index=data.index)
    return pd.DataFrame(added,index=data.index).replace([np.inf,-np.inf],np.nan)


def prepare(data,columns,train_idx):
    X=data[columns].copy()
    cats=list(X.select_dtypes(include=['object','string','category']).columns)
    for c in cats: X[c]=X[c].astype('string').fillna('__MISSING__').astype('category')
    numeric=[c for c in X if c not in cats]
    X[numeric]=X[numeric].apply(pd.to_numeric,errors='raise').replace([np.inf,-np.inf],np.nan)
    keep=X.iloc[train_idx].columns[X.iloc[train_idx].notna().mean()>=0.02].tolist()
    X=X[keep]
    return X,[c for c in cats if c in keep]


def catframe(X,cats):
    z=X.copy()
    for c in cats: z[c]=z[c].astype('string').fillna('__MISSING__').astype(str)
    return z


def models(arch,params,seed,threads,rounds):
    if arch=='lightgbm':
        p=dict(objective='binary',n_estimators=rounds,learning_rate=.035,num_leaves=23,
            min_child_samples=70,colsample_bytree=.8,subsample=.8,subsample_freq=1,
            reg_lambda=5,verbosity=-1,random_state=seed,n_jobs=threads)
        p.update(params)
        return LGBMClassifier(**p)
    p=dict(iterations=rounds,depth=5,learning_rate=.04,l2_leaf_reg=6,
        loss_function='Logloss',eval_metric='PRAUC',random_seed=seed,
        thread_count=threads,verbose=False,allow_writing_files=False)
    p.update(params)
    return CatBoostClassifier(**p)


def train(arch,params,X,y,fit,stop,cats,seed,threads):
    m=models(arch,params,seed,threads,1800)
    if arch=='lightgbm':
        m.fit(X.iloc[fit],y[fit],eval_set=[(X.iloc[stop],y[stop])],
              eval_metric='average_precision',categorical_feature=cats,
              callbacks=[early_stopping(100,verbose=False),log_evaluation(0)])
    else:
        m.fit(catframe(X.iloc[fit],cats),y[fit],cat_features=cats,
              eval_set=(catframe(X.iloc[stop],cats),y[stop]),
              early_stopping_rounds=100,verbose=False)
    return m


def pred(m,arch,X,cats):
    return m.predict_proba(catframe(X,cats) if arch=='catboost' else X)[:,1]


def score(y,p,threshold=None):
    p=np.clip(p,1e-8,1-1e-8)
    result=dict(n=len(y),positives=int(y.sum()),prevalence=float(y.mean()),
        auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),
        brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p)))
    if threshold is not None:
        tn,fp,fn,tp=confusion_matrix(y,p>=threshold,labels=[0,1]).ravel()
        result.update(threshold=float(threshold),sensitivity=float(tp/(tp+fn)),
            specificity=float(tn/(tn+fp)),precision=float(tp/(tp+fp)) if tp+fp else 0.,
            balanced_accuracy=float((tp/(tp+fn)+tn/(tn+fp))/2),
            tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
    return result


def threshold_at_specificity(y,p,target):
    neg=np.asarray(p)[np.asarray(y)==0]
    # Strictly exceed empirical negative quantile to meet target specificity.
    candidate=np.unique(np.r_[0,np.asarray(p),1])
    spec=np.searchsorted(np.sort(neg),candidate,side='left')/len(neg)
    eligible=candidate[spec>=target]
    return float(eligible[0]) if len(eligible) else 1.0


def best_params(args,arch):
    if not args.phase13: return {}
    path=Path(args.phase13)/f'{arch}_best_params.json'
    if not path.exists():
        print(f'No tuned parameters for {arch}; using Phase 9C-like defaults',flush=True)
        return {}
    report=json.loads(path.read_text())
    params=report['best_params'].copy()
    if arch=='lightgbm': params['subsample_freq']=1
    else: params['bootstrap_type']='Bayesian'
    print(f'Using Phase 13 tuned parameters for {arch}: {path}',flush=True)
    return params


def main(args):
    out=Path(args.output).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    data,grp,ix,y,splitfile,sources=load(args)
    basecols=grp['baseline']+grp['labs']
    history=grp['history']
    med=[c for c in grp['medcom'] if c.startswith('rx_') or c=='has_rx_order_24h']
    com=[c for c in grp['medcom'] if c not in med]
    original_bundle=joblib.load(Path(args.phase9c)/'E_all_models.joblib')
    refcols=original_bundle['columns']
    if any(c not in data for c in refcols):
        raise ValueError('Phase 9C reference features missing; check input versions')
    if len(refcols)!=600 or len(set(refcols))!=600:
        raise ValueError('Expected 600 unique Phase 9C predictors; refusing incompatible dataset')
    # Preserve Phase 9C's training-derived 600-feature selection as fixed reference.
    extras=advanced_renal_features(data,set(basecols))
    for c in extras:
        if c in data: raise ValueError(f'Derived feature collision: {c}')
    data=pd.concat([data,extras],axis=1)
    sets={
      'A_original_600':refcols,
      'B_renal_engineered':refcols+list(extras.columns),
      'C_no_admin': [c for c in refcols if c not in ('admission_type','admission_location')],
      'D_no_medications':[c for c in refcols if c not in med],
      'E_no_history':[c for c in refcols if c not in history],
      'F_no_comorbidities':[c for c in refcols if c not in com],
      'G_no_history_medications':[c for c in refcols if c not in set(history+med)],
    }
    if args.quick: sets={k:v for k,v in sets.items() if k in ('A_original_600','B_renal_engineered')}
    manifest={'version':1,'split_sha256':sha(splitfile),
        'source_hashes':{k:sha(v) for k,v in sources.items()},
        'phase1_hash':sha(Path(args.phase1)/'phase1_labels.parquet'),
        'phase13_parameter_hashes':{a:sha(Path(args.phase13)/f'{a}_best_params.json') if args.phase13 and (Path(args.phase13)/f'{a}_best_params.json').exists() else None for a in ('lightgbm','catboost')},
        'seeds':args.seeds,'sets':sets,'rows':len(data),'phase9c_columns':refcols,
        'note':'Previously inspected MIMIC-IV evaluation; any finalization is exploratory.'}
    mp=out/'master_manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:
        raise RuntimeError('Manifest changed: use a NEW output directory to avoid mixing experiments')
    if not mp.exists(): dump_json(mp,manifest)
    if args.finalize:
        if (out/'FINALIZED.json').exists(): raise RuntimeError('Final evaluation already completed')
        selected_path=out/'frozen_selection.json'
        if not selected_path.exists(): raise RuntimeError('Run development mode first')
        selected=json.loads(selected_path.read_text())
        key=selected['feature_set']
        if key not in sets: raise RuntimeError('Frozen feature set missing')
        X,cats=prepare(data,sets[key],ix['train'])
        if list(X.columns)!=selected['columns']: raise RuntimeError('Frozen schema differs')
        result=fit_ensemble(X,cats,y,ix,args,selected['params'],evaluate=True)
        predictions=data.iloc[ix['evaluation']][['subject_id','hadm_id']].copy()
        predictions['outcome']=y[ix['evaluation']]
        for name,p in result['evaluation_probabilities'].items():
            predictions[name+'_probability']=p
        predictions.to_parquet(out/'master_evaluation_predictions.parquet',index=False)
        joblib.dump(result['models'],out/'master_models.joblib')
        dump_json(out/'FINALIZED.json',{'feature_set':key,'evaluation':result['evaluation_metrics'],
            'warning':'Exploratory reassessment: evaluation patients have been examined in prior phases.'})
        print(json.dumps(result['evaluation_metrics'],indent=2),flush=True)
        return
    results=[]
    for name,cols in sets.items():
        print(f'\n=== {name}: {len(cols)} candidate features ===',flush=True)
        X,cats=prepare(data,cols,ix['train'])
        # Use early_stop for boosting stopping; blend for blend selection; threshold for threshold selection.
        r=fit_ensemble(X,cats,y,ix,args,{a:best_params(args,a) for a in ('lightgbm','catboost')},evaluate=False)
        for model,metrics in r['development_metrics'].items():
            row={'feature_set':name,'model':model,'features':X.shape[1],**metrics}
            results.append(row)
            pd.DataFrame(results).to_csv(out/'development_results.csv',index=False)
        print('blend validation AUPRC:',round(r['development_metrics']['blend']['auprc'],5),flush=True)
    ranked=pd.DataFrame(results).sort_values('auprc',ascending=False)
    ranked.to_csv(out/'development_ranked.csv',index=False)
    # Freeze choice by blend AUPRC, NEVER evaluation scores.
    blend=ranked[ranked.model=='blend'].iloc[0]
    choice=blend.feature_set
    X,cats=prepare(data,sets[choice],ix['train'])
    frozen={'feature_set':choice,'columns':list(X.columns),'categorical':cats,
            'params':{a:best_params(args,a) for a in ('lightgbm','catboost')},
            'selection_metric':'blend AUPRC on Phase 9C blend split',
            'warning':'Multiple feature groups selected on development data; test is previously inspected.'}
    dump_json(out/'frozen_selection.json',frozen)
    print('\nFROZEN SELECTION:',choice,'(development only)',flush=True)
    print('To run ONE exploratory evaluation, rerun identical command with --finalize.',flush=True)


def fit_ensemble(X,cats,y,ix,args,params,evaluate=False):
    trained={};blends={};threshold_preds={};eval_preds={}
    for arch in ('lightgbm','catboost'):
        trained[arch]=[];blends[arch]=[];threshold_preds[arch]=[];eval_preds[arch]=[]
        for k in range(args.seeds):
            m=train(arch,params[arch],X,y,ix['train'],ix['early_stop'],cats,
                    SEED+k*71,args.threads)
            trained[arch].append(m)
            blends[arch].append(pred(m,arch,X.iloc[ix['blend']],cats))
            threshold_preds[arch].append(pred(m,arch,X.iloc[ix['threshold']],cats))
            if evaluate: eval_preds[arch].append(pred(m,arch,X.iloc[ix['evaluation']],cats))
    b={a:np.mean(v,axis=0) for a,v in blends.items()}
    t={a:np.mean(v,axis=0) for a,v in threshold_preds.items()}
    weights=np.linspace(0,1,21)
    w=float(max(weights,key=lambda z:average_precision_score(y[ix['blend']],z*b['lightgbm']+(1-z)*b['catboost'])))
    b['blend']=w*b['lightgbm']+(1-w)*b['catboost']
    t['blend']=w*t['lightgbm']+(1-w)*t['catboost']
    thresholds={a:threshold_at_specificity(y[ix['threshold']],p,args.specificity) for a,p in t.items()}
    dev={a:{**score(y[ix['blend']],p),
         'threshold_validation_recall':score(y[ix['threshold']],t[a],thresholds[a])['sensitivity'],
         'threshold_validation_specificity':score(y[ix['threshold']],t[a],thresholds[a])['specificity']}
         for a,p in b.items()}
    result={'development_metrics':dev}
    if evaluate:
        e={a:np.mean(v,axis=0) for a,v in eval_preds.items()}
        e['blend']=w*e['lightgbm']+(1-w)*e['catboost']
        result['evaluation_probabilities']=e
        result['evaluation_metrics']={a:score(y[ix['evaluation']],p,thresholds[a]) for a,p in e.items()}
        result['models']={'models':trained,'columns':list(X.columns),'categorical':cats,
            'blend_lightgbm_weight':w,'thresholds':thresholds,'seed_count':args.seeds}
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ('phase1','phase2','phase9','phase9c','output'):
        ap.add_argument('--'+arg,required=True)
    ap.add_argument('--phase13',default=None,help='Optional completed Phase 13 best_params JSON directory')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--seeds',type=int,default=3)
    ap.add_argument('--specificity',type=float,default=.8)
    ap.add_argument('--quick',action='store_true',help='Only original vs renal engineered')
    ap.add_argument('--finalize',action='store_true',help='One-time exploratory evaluation')
    a=ap.parse_args()
    if a.seeds<1 or a.threads<1 or not (0<a.specificity<1): ap.error('Invalid seeds/threads/specificity')
    main(a)
