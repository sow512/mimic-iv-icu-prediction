#!/usr/bin/env python3
"""Original-style renal ICU transfer prediction, replacing logistic regression with
class-balanced LightGBM and CatBoost. Retrospective research only.

Uses the SAME Phase 1/2 cohort and features for both models, and does not
re-run the raw 158M-row laboratory extraction. Feature selection, imputers,
encoders and thresholds are fitted on development data only.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

SEED = 42

def scores(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(threshold=float(threshold), accuracy=float(accuracy_score(y,pred)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        sensitivity=float(recall_score(y,pred,zero_division=0)),
        specificity=float(tn/(tn+fp)) if tn+fp else None,
        precision=float(precision_score(y,pred,zero_division=0)),
        f1=float(f1_score(y,pred,zero_division=0)),
        auroc=float(roc_auc_score(y,p)), auprc=float(average_precision_score(y,p)),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def choose_threshold(y, p, objective):
    # Validation-only threshold selection; never optimize on test labels.
    thresholds=np.unique(np.r_[0.0, np.quantile(p,np.linspace(0,1,501)),1.0])
    vals=[]
    for t in thresholds:
        pred=p>=t
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        sensitivity=tp/(tp+fn) if tp+fn else 0
        specificity=tn/(tn+fp) if tn+fp else 0
        precision=tp/(tp+fp) if tp+fp else 0
        if objective=='balanced_accuracy': metric=(sensitivity+specificity)/2
        else: metric=2*precision*sensitivity/(precision+sensitivity) if precision+sensitivity else 0
        vals.append((metric, float(t)))
    return max(vals, key=lambda z:z[0])[1]

def main(a):
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    p1=Path(a.phase1);p2=Path(a.phase2)
    base=pd.read_parquet(p1/'phase1_landmark_predictors.parquet')
    labels=pd.read_parquet(p1/'phase1_labels.parquet')
    labs=pd.read_parquet(p2/'phase2_temporal_lab_features.parquet')
    for name, frame in [('base',base),('labels',labels),('labs',labs)]:
        if frame.hadm_id.isna().any() or frame.hadm_id.duplicated().any():
            raise ValueError(f'{name}: missing or duplicate hadm_id')
    # Phase 2 can contain repeated baseline columns; Phase 1 is authoritative.
    labs=labs.drop(columns=[c for c in ['subject_id','landmark_time','age','gender',
        'admission_type','admission_location'] if c in labs],errors='ignore')
    data=base.merge(labs,on='hadm_id',how='left',validate='one_to_one')
    data=data.merge(labels[['subject_id','hadm_id','icu_transfer_24_72h']],
        on=['subject_id','hadm_id'],how='inner',validate='one_to_one')
    if len(data)!=len(base) or len(data)!=len(labels): raise ValueError('Row mismatch')
    forbidden=['subject_id','hadm_id','landmark_time','icu_transfer_24_72h']
    X=data.drop(columns=forbidden)
    suspicious=[c for c in X if any(s in c.lower() for s in ['icu_transfer','icu_intime','dischtime','deathtime','horizon_end','label_status','first_icu','hours_to_icu'])]
    if suspicious: raise ValueError('Possible label leakage: '+str(suspicious))
    y=data.icu_transfer_24_72h.astype(int).to_numpy()
    groups=data.subject_id.to_numpy()
    # Train/validation/test patient-disjoint split: 70/15/15 approximately.
    outer=GroupShuffleSplit(n_splits=1,test_size=.15,random_state=SEED)
    dev,test=next(outer.split(X,y,groups))
    inner=GroupShuffleSplit(n_splits=1,test_size=.15/.85,random_state=SEED+1)
    ti,vi=next(inner.split(X.iloc[dev],y[dev],groups[dev]))
    train,val=dev[ti],dev[vi]
    assert not (set(groups[train])&set(groups[val]) or set(groups[train])&set(groups[test]) or set(groups[val])&set(groups[test]))
    for name,ids in [('train',train),('validation',val),('test',test)]:
        print(f'{name}: admissions={len(ids):,}, positives={y[ids].sum():,}, patients={len(set(groups[ids])):,}')
    # Match original preprocessing: drop high-missingness features, median/mode
    # imputation, one-hot encoding, standardization, VIF then RFE.
    xtr=X.iloc[train].copy(); xva=X.iloc[val].copy(); xte=X.iloc[test].copy()
    keep=xtr.columns[xtr.isna().mean()<=a.max_missing].tolist()
    xtr=xtr[keep];xva=xva[keep];xte=xte[keep]
    cats=xtr.select_dtypes(include=['object','string','category']).columns.tolist()
    nums=[c for c in keep if c not in cats]
    for frame in (xtr,xva,xte):
        for c in nums: frame[c]=pd.to_numeric(frame[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
        for c in cats: frame[c]=frame[c].astype('object').where(frame[c].notna(),np.nan)
    # A completely missing training column cannot be imputed reliably.
    nums=[c for c in nums if xtr[c].notna().any()]
    cats=[c for c in cats if xtr[c].notna().any()]
    nimp=SimpleImputer(strategy='median');cimp=SimpleImputer(strategy='most_frequent')
    encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    def transform(frame, fit=False):
        n=(nimp.fit_transform(frame[nums]) if fit else nimp.transform(frame[nums])) if nums else np.empty((len(frame),0))
        c=(cimp.fit_transform(frame[cats]) if fit else cimp.transform(frame[cats])) if cats else np.empty((len(frame),0))
        enc=(encoder.fit_transform(c) if fit else encoder.transform(c)) if cats else np.empty((len(frame),0))
        names=nums+(encoder.get_feature_names_out(cats).tolist() if cats else [])
        return pd.DataFrame(np.column_stack([n,enc]),columns=names,index=frame.index)
    Xtr=transform(xtr,True); Xva=transform(xva); Xte=transform(xte)
    # Remove constant features, then scale like the original (not required for trees).
    nonconstant=Xtr.columns[Xtr.nunique(dropna=False)>1].tolist()
    Xtr=Xtr[nonconstant];Xva=Xva[nonconstant];Xte=Xte[nonconstant]
    scaler=StandardScaler()
    Xtr=pd.DataFrame(scaler.fit_transform(Xtr),columns=nonconstant,index=Xtr.index)
    Xva=pd.DataFrame(scaler.transform(Xva),columns=nonconstant,index=Xva.index)
    Xte=pd.DataFrame(scaler.transform(Xte),columns=nonconstant,index=Xte.index)
    if a.vif:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        cols=Xtr.columns.tolist()
        while len(cols)>a.select_features:
            arr=Xtr[cols].to_numpy()
            vifs=np.array([variance_inflation_factor(arr,i) for i in range(len(cols))])
            bad=np.where(~np.isfinite(vifs),np.inf,vifs)
            idx=int(np.argmax(bad))
            if bad[idx]<=10: break
            print('VIF remove:',cols[idx],bad[idx]);cols.pop(idx)
        Xtr=Xtr[cols];Xva=Xva[cols];Xte=Xte[cols]
    if a.select_features and Xtr.shape[1]>a.select_features:
        selector=RFE(LGBMClassifier(n_estimators=120,num_leaves=15,learning_rate=.05,
            class_weight='balanced',random_state=SEED,n_jobs=a.threads,verbosity=-1),
            n_features_to_select=a.select_features,step=max(1,a.rfe_step))
        selector.fit(Xtr,y[train]);cols=Xtr.columns[selector.support_].tolist()
        Xtr=Xtr[cols];Xva=Xva[cols];Xte=Xte[cols]
    print('Final features:',len(Xtr.columns))
    # Original-style class weighting for BOTH models.
    lgb=LGBMClassifier(n_estimators=450,learning_rate=.035,num_leaves=23,
        min_child_samples=60,reg_lambda=5,class_weight='balanced',
        random_state=SEED,n_jobs=a.threads,verbosity=-1)
    cat=CatBoostClassifier(iterations=500,depth=5,learning_rate=.04,l2_leaf_reg=6,
        auto_class_weights='Balanced',loss_function='Logloss',random_seed=SEED,
        thread_count=a.threads,verbose=False)
    lgb.fit(Xtr,y[train]);cat.fit(Xtr,y[train])
    pva={'lightgbm':lgb.predict_proba(Xva)[:,1], 'catboost':cat.predict_proba(Xva)[:,1]}
    pte={'lightgbm':lgb.predict_proba(Xte)[:,1], 'catboost':cat.predict_proba(Xte)[:,1]}
    # Weight chosen on validation AUPRC only.
    candidates=np.linspace(0,1,21)
    weight=max(candidates,key=lambda w:average_precision_score(y[val],w*pva['lightgbm']+(1-w)*pva['catboost']))
    pva['blend']=weight*pva['lightgbm']+(1-weight)*pva['catboost']
    pte['blend']=weight*pte['lightgbm']+(1-weight)*pte['catboost']
    report={'split':{n:{'admissions':int(len(i)),'positives':int(y[i].sum())} for n,i in [('train',train),('validation',val),('test',test)]},
        'final_features':Xtr.columns.tolist(),'blend_lightgbm_weight':float(weight),
        'threshold_objective':a.threshold_objective,'models':{}}
    for name in pva:
        t=choose_threshold(y[val],pva[name],a.threshold_objective)
        report['models'][name]={'validation':scores(y[val],pva[name],t),
                                'test':scores(y[test],pte[name],t),
                                'test_at_0_5':scores(y[test],pte[name],.5)}
        print(f'\n{name.upper()} | validation-selected threshold={t:.4f}')
        print(json.dumps(report['models'][name]['test'],indent=2))
    pred_df=pd.DataFrame({'subject_id':groups[test],'hadm_id':data.iloc[test].hadm_id.to_numpy(),'outcome':y[test]})
    for name,p in pte.items():pred_df[name+'_probability']=p
    pred_df.to_parquet(out/'test_predictions.parquet',index=False)
    (out/'comparison_report.json').write_text(json.dumps(report,indent=2))
    joblib.dump({'lightgbm':lgb,'catboost':cat,'numeric_imputer':nimp,'categorical_imputer':cimp,
                 'encoder':encoder,'scaler':scaler,'original_numeric_columns':nums,
                 'original_categorical_columns':cats,'nonconstant_columns':nonconstant,
                 'selected_columns':Xtr.columns.tolist(),'blend_lightgbm_weight':float(weight)},out/'models.joblib')
    for name,p in pte.items():
        fpr,tpr,_=roc_curve(y[test],p)
        precision,recall,_=precision_recall_curve(y[test],p)
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        axs[0].plot(fpr,tpr,label=f'AUROC {roc_auc_score(y[test],p):.3f}')
        axs[0].plot([0,1],[0,1],'--');axs[0].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC');axs[0].legend()
        axs[1].plot(recall,precision,label=f'AUPRC {average_precision_score(y[test],p):.3f}')
        axs[1].axhline(y[test].mean(),linestyle='--');axs[1].set(xlabel='Recall',ylabel='Precision',title='Precision–recall');axs[1].legend()
        fig.suptitle(name);fig.tight_layout();fig.savefig(out/f'{name}_curves.png',dpi=180);plt.close(fig)
    print('\nSaved outputs:',out)
    print('NOTE: Same retrospective ICD cohort; not a direct comparison with the original 3.07% cohort.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--phase1',required=True);p.add_argument('--phase2',required=True)
    p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--max-missing',type=float,default=.5)
    p.add_argument('--select-features',type=int,default=20)
    p.add_argument('--rfe-step',type=int,default=5)
    p.add_argument('--vif',action='store_true',help='Optional expensive original-style iterative VIF filtering')
    p.add_argument('--threshold-objective',choices=['balanced_accuracy','f1'],default='balanced_accuracy')
    main(p.parse_args())
