#!/usr/bin/env python3
"""Validate and join three previously extracted MIMIC-IV kidney datasets.
Compare EBM without SMOTE, with SMOTE 0.2, and with SMOTE 0.5.
All preprocessing/selection/SMOTE fitted on training patients only.
"""
import argparse, json, logging, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from interpret.glassbox import ExplainableBoostingClassifier
import joblib

LOG=logging.getLogger('kidney_compare')
TARGET='icu_transfer_24_72h'
DROP={'hadm_id','subject_id','prediction_time',TARGET,'admittime','dischtime','deathtime','intime','outtime','stay_id','anchor_year','anchor_age','gender','icu_intime','icu_outtime','observation_end','verified_label'}
RETROSPECTIVE_FLAGS={'AKI','CKD','Glomerular','Other renal','diabetes','hypertension','heart_failure','sepsis'}

def barplot(labels,counts,title,path,ylabel='Count'):
    fig,ax=plt.subplots(figsize=(max(8,len(labels)*1.2),5))
    bars=ax.bar(labels,counts,color='#2879a8')
    ax.set(title=title,ylabel=ylabel);ax.tick_params(axis='x',rotation=25)
    for b,n in zip(bars,counts): ax.annotate(f'{int(n):,}',(b.get_x()+b.get_width()/2,b.get_height()),ha='center',va='bottom',fontsize=9)
    fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)

def splits(df,seed):
    g=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed)
    dev,test=next(g.split(df,groups=df.subject_id))
    d=df.iloc[dev];t=df.iloc[test]
    g=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1)
    tr,va=next(g.split(d,groups=d.subject_id))
    return d.iloc[tr].copy(),d.iloc[va].copy(),t.copy()

def select_features(tr,va,te,output,max_features,missing_max,corr_max,seed):
    initial=[c for c in tr.columns if c not in DROP and pd.api.types.is_numeric_dtype(tr[c])]
    steps=[('Initial numeric features',len(initial))]
    keep=[c for c in initial if tr[c].notna().mean()>=1-missing_max]
    if not keep: raise ValueError('No features remain after missingness filtering. Increase --max-missing.')
    steps.append(('Missingness filter',len(keep)))
    before_variance=keep.copy()
    imp=SimpleImputer(strategy='median')
    a=imp.fit_transform(tr[keep]);b=imp.transform(va[keep]);c=imp.transform(te[keep])
    vt=VarianceThreshold(threshold=1e-10)
    a=vt.fit_transform(a);b=vt.transform(b);c=vt.transform(c)
    keep=np.array(keep)[vt.get_support()].tolist()
    variance_keep=keep.copy()
    steps.append(('Variance filter',len(keep)))
    if len(keep)>1:
        cor=np.abs(np.corrcoef(a,rowvar=False))
        cor=np.nan_to_num(cor,nan=0)
        discard=set()
        for j in range(1,len(keep)):
            if any(cor[i,j]>corr_max for i in range(j) if i not in discard): discard.add(j)
        indices=[i for i in range(len(keep)) if i not in discard]
        a=a[:,indices];b=b[:,indices];c=c[:,indices];keep=[keep[i] for i in indices]
    steps.append(('Correlation filter',len(keep)))
    y=tr[TARGET].to_numpy(dtype=int)
    mi=mutual_info_classif(a,y,random_state=seed)
    order=np.argsort(-mi,kind='stable')
    n=min(max_features,len(keep));idx=order[:n]
    ranking=pd.DataFrame({'feature':keep,'mutual_information':mi}).sort_values('mutual_information',ascending=False)
    ranking.to_csv(output/'feature_selection_ranking.csv',index=False)
    chosen=[keep[i] for i in idx]
    steps.append(('Mutual information top K',len(chosen)))
    pd.DataFrame(steps,columns=['method','features_remaining']).to_csv(output/'feature_selection_counts.csv',index=False)
    barplot([s[0] for s in steps],[s[1] for s in steps], 'Features retained after each selection stage',output/'feature_selection_counts.png','Features remaining')
    (output/'selected_features.txt').write_text('\n'.join(chosen)+'\n')
    prep={'imputer':imp,'input_features':before_variance,'variance_support':vt.get_support(),
          'post_variance_features':variance_keep,'post_correlation_features':keep,'selected_features':chosen}
    return a[:,idx],b[:,idx],c[:,idx],chosen,prep,keep,steps

def metric(y,p,threshold):
    pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),
      'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),
      'specificity':tn/max(1,tn+fp),'f1':f1_score(y,pred,zero_division=0),
      'auroc':roc_auc_score(y,p),'pr_auc':average_precision_score(y,p),
      'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
      'alerts_per_100':100*pred.mean(),'threshold':float(threshold)}

def choose_threshold(y,p,goal):
    precision,recall,thresholds=precision_recall_curve(y,p)
    valid=np.flatnonzero(recall[:-1]>=goal)
    if not len(valid):return float(np.min(p))
    ix=max(valid,key=lambda i:(precision[i],thresholds[i]))
    return float(thresholds[ix])

def diagnostics(y,p,threshold,name,folder):
    cm=confusion_matrix(y,p>=threshold,labels=[0,1]);fig,ax=plt.subplots(figsize=(5,4))
    ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['No ICU','ICU'],yticklabels=['No ICU','ICU'],xlabel='Predicted',ylabel='Actual',title=f'{name} confusion matrix')
    fig.tight_layout();fig.savefig(folder/f'{name}_confusion_matrix.png',dpi=170);plt.close(fig)
    fpr,tpr,_=roc_curve(y,p);pr,re,_=precision_recall_curve(y,p)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].plot(fpr,tpr,label=f'AUROC {roc_auc_score(y,p):.3f}');axs[0].plot([0,1],[0,1],'--',color='gray');axs[0].set(xlabel='FPR',ylabel='TPR',title='ROC');axs[0].legend()
    axs[1].plot(re,pr,label=f'AP {average_precision_score(y,p):.3f}');axs[1].axhline(np.mean(y),color='gray',linestyle='--');axs[1].set(xlabel='Recall',ylabel='Precision',title='PR');axs[1].legend()
    fig.tight_layout();fig.savefig(folder/f'{name}_roc_pr.png',dpi=170);plt.close(fig)

def read_three_and_validate(args):
    """One row per existing prediction window; verify ICU labels against stay times."""
    LOG.info('Reading prediction windows: %s', args.windows)
    w=pd.read_csv(args.windows,low_memory=False)
    LOG.info('Reading kidney admission cohort: %s', args.patients)
    patients=pd.read_csv(args.patients,low_memory=False)
    LOG.info('Reading ICU stays: %s', args.icu_stays)
    stays=pd.read_csv(args.icu_stays,low_memory=False)
    for name,df,cols in [('windows',w,{'subject_id','hadm_id','prediction_time',TARGET}),
                          ('patients',patients,{'subject_id','hadm_id','admittime','dischtime'}),
                          ('icu_stays',stays,{'subject_id','hadm_id','intime','outtime'})]:
        missing=cols-set(df.columns)
        if missing: raise ValueError(f'{name} missing required columns: {sorted(missing)}')
    for frame in [w,patients,stays]:
        for col in ['subject_id','hadm_id']:
            frame[col]=pd.to_numeric(frame[col],errors='coerce').astype('Int64')
    for frame,cols in [(w,['prediction_time']),(patients,['admittime','dischtime','deathtime']),
                       (stays,['intime','outtime'])]:
        for col in cols:
            if col in frame: frame[col]=pd.to_datetime(frame[col],errors='coerce')
    w=w.dropna(subset=['subject_id','hadm_id','prediction_time',TARGET]).copy()
    w[TARGET]=pd.to_numeric(w[TARGET],errors='raise').astype(int)
    if not w[TARGET].isin([0,1]).all(): raise ValueError('Outcome must be binary')
    if w.duplicated(['subject_id','hadm_id','prediction_time']).any():
        raise ValueError('Duplicate patient/admission/prediction_time rows in window file')
    patients=patients.dropna(subset=['subject_id','hadm_id']).copy()
    if patients.duplicated(['subject_id','hadm_id']).any():
        raise ValueError('kidney_patients contains duplicate subject_id/hadm_id; inspect before joining')
    patient_cols=['subject_id','hadm_id','admittime','dischtime']
    if 'deathtime' in patients: patient_cols+=['deathtime']
    extra=[c for c in patients.columns if c not in w and c not in patient_cols]
    w=w.merge(patients[patient_cols+extra],on=['subject_id','hadm_id'],how='inner',validate='many_to_one')
    if w.empty: raise ValueError('No windows matched kidney patient admissions')
    if 'deathtime' in w:
        w['observation_end']=w[['dischtime','deathtime']].min(axis=1)
    else: w['observation_end']=w['dischtime']
    stays=stays.dropna(subset=['subject_id','hadm_id','intime']).copy()
    stay_groups={(int(k[0]),int(k[1])):g[['intime','outtime']].itertuples(index=False,name=None)
                 for k,g in stays.groupby(['subject_id','hadm_id'])}
    stay_groups={k:list(v) for k,v in stay_groups.items()}
    valid=[]; relabeled=[]; reasons={}; mismatch=0
    for row in w[['subject_id','hadm_id','prediction_time','admittime','observation_end',TARGET]].itertuples(index=False,name=None):
        sid,hid,t,admit,end,old=row
        why=None
        if pd.isna(admit) or pd.isna(end) or t<admit+pd.Timedelta(hours=24): why='invalid_landmark'
        elif t+pd.Timedelta(hours=72)>end: why='incomplete_followup'
        ss=stay_groups.get((int(sid),int(hid)),[])
        if why is None and any(start<=t and (pd.isna(finish) or finish>t) for start,finish in ss): why='already_in_icu'
        future=[(start-t).total_seconds()/3600 for start,finish in ss if start>t]
        if why is None and future and min(future)<24: why='icu_within_24h'
        if why:
            valid.append(False);relabeled.append(0);reasons[why]=reasons.get(why,0)+1
        else:
            label=int(any(24<=delta<=72 for delta in future))
            valid.append(True);relabeled.append(label)
            mismatch+=int(label!=old)
    w['verified_label']=relabeled
    w=w.loc[valid].copy()
    w[TARGET]=w.pop('verified_label').astype(int)
    w=w.drop(columns=['observation_end'],errors='ignore')
    if not args.allow_retrospective_diagnosis_features:
        w=w.drop(columns=list(RETROSPECTIVE_FLAGS),errors='ignore')
    LOG.info('Verified windows=%s; ICU positives=%s; label mismatches=%s; exclusions=%s',
             f'{len(w):,}',f'{w[TARGET].sum():,}',f'{mismatch:,}',reasons)
    audit={'input_windows':int(len(valid)),'verified_windows':int(len(w)),
           'positive_windows':int(w[TARGET].sum()),'original_label_mismatches':int(mismatch),
           'exclusions':reasons,'patient_admissions':int(len(patients)),
           'unique_kidney_patients':int(patients.subject_id.nunique()),
           'icu_stays':int(len(stays)),'diagnosis_predictors_included':bool(args.allow_retrospective_diagnosis_features)}
    (args.output/'cohort_label_audit.json').write_text(json.dumps(audit,indent=2))
    diseases=[c for c in ['AKI','CKD','Glomerular','Other renal'] if c in patients]
    if diseases:
        counts=[int(pd.to_numeric(patients[c],errors='coerce').fillna(0).sum()) for c in diseases]
        pd.DataFrame({'disease':diseases,'admissions':counts}).to_csv(args.output/'kidney_disease_counts.csv',index=False)
        barplot(diseases,counts,'Kidney diagnoses by admission (overlap allowed)',args.output/'kidney_diseases.png','Admissions')
    w.to_csv(args.output/'verified_feature_engineered_windows.csv.gz',index=False,compression='gzip')
    return w,audit

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).resolve().parent/'Kidney_EBM_SMOTE'
    ap.add_argument('--windows',type=Path,default=base/'feature_engineered_windows.csv.gz')
    ap.add_argument('--patients',type=Path,default=base/'kidney_patients.csv.gz')
    ap.add_argument('--icu-stays',type=Path,default=base/'kidney_icu_stays.csv.gz')
    ap.add_argument('--allow-retrospective-diagnosis-features',action='store_true',
                    help='Not recommended: ICD flags may be assigned after prediction time')
    ap.add_argument('--output',type=Path,default=Path('Kidney_EBM_SMOTE_Optimized'))
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--target-recall',type=float,default=.60)
    ap.add_argument('--max-features',type=int,default=70)
    ap.add_argument('--max-missing',type=float,default=.95)
    ap.add_argument('--correlation-limit',type=float,default=.98)
    ap.add_argument('--max-rounds',type=int,default=600)
    ap.add_argument('--outer-bags',type=int,default=2)
    ap.add_argument('--n-jobs',type=int,default=2)
    args=ap.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    for path in [args.windows,args.patients,args.icu_stays]:
        if not path.is_file(): ap.error(f'Missing input file: {path}')
    if not 0<args.target_recall<=1:ap.error('--target-recall must be between 0 and 1')
    args.output.mkdir(parents=True,exist_ok=True)
    t=time.monotonic()
    df,audit=read_three_and_validate(args)
    if df[TARGET].nunique()!=2:raise ValueError('Need both classes after ICU label validation')
    tr,va,te=splits(df,args.seed)
    for name,part in [('train',tr),('validation',va),('test',te)]:
        LOG.info('%s: %s windows, %s patients, positives=%s',name,len(part),part.subject_id.nunique(),part[TARGET].sum())
    a,b,c,chosen,preprocessor,postcorr,steps=select_features(tr,va,te,args.output,args.max_features,args.max_missing,args.correlation_limit,args.seed)
    LOG.info('Feature selection stages: %s',steps)
    ytr=tr[TARGET].to_numpy(dtype=int);yv=va[TARGET].to_numpy(dtype=int);yt=te[TARGET].to_numpy(dtype=int)
    for y in [ytr,yv,yt]:
        if len(np.unique(y))<2:raise ValueError('A split contains only one class; change --seed')
    original=np.bincount(ytr,minlength=2)
    barplot(['No ICU','ICU transfer'],original,'Training imbalance before SMOTE',args.output/'class_imbalance_before_smote.png')
    records=[];models={};probs={}
    for name,ratio in [('no_smote',None),('smote_0_2',.2),('smote_0_5',.5)]:
        LOG.info('Starting %s',name)
        folder=args.output/name;folder.mkdir(exist_ok=True)
        prevalence=original[1]/max(1,original[0]);sampler=None
        if ratio is not None and ratio>prevalence:
            sampler=SMOTE(sampling_strategy=ratio,random_state=args.seed,k_neighbors=min(5,original[1]-1)) if original[1]>1 else RandomOverSampler(sampling_strategy=ratio,random_state=args.seed)
        X,y=sampler.fit_resample(a,ytr) if sampler is not None else (a,ytr)
        counts=np.bincount(y,minlength=2)
        barplot(['No ICU','ICU transfer'],counts,f'{name}: training class distribution',folder/'class_imbalance_after_resampling.png')
        LOG.info('%s: training %s rows x %s features (positive=%s)',name,len(y),len(chosen),counts[1])
        model=ExplainableBoostingClassifier(interactions=0,max_bins=128,learning_rate=.03,max_rounds=args.max_rounds,outer_bags=args.outer_bags,n_jobs=args.n_jobs,random_state=args.seed)
        model.fit(X,y)
        pv=model.predict_proba(b)[:,1];pt=model.predict_proba(c)[:,1]
        threshold=choose_threshold(yv,pv,args.target_recall)
        for split,yy,pp,frame in [('validation',yv,pv,va),('test',yt,pt,te)]:
            row={'model':name,'split':split,'train_rows_after_resampling':len(y),'selected_features':len(chosen),**metric(yy,pp,threshold)}
            records.append(row)
            diagnostics(yy,pp,threshold,split,folder)
            pd.DataFrame({'subject_id':frame.subject_id.to_numpy(),'hadm_id':frame.hadm_id.to_numpy(),
                'prediction_time':frame.prediction_time.to_numpy(),'actual':yy,'probability':pp,'predicted':(pp>=threshold).astype(int)}).to_csv(folder/f'{split}_predictions.csv',index=False)
        imp=pd.DataFrame({'feature':chosen,'importance':model.term_importances()[:len(chosen)]}).sort_values('importance',ascending=False)
        imp.to_csv(folder/'ebm_feature_importance.csv',index=False)
        ax=imp.head(20).iloc[::-1].plot.barh(x='feature',y='importance',legend=False,figsize=(9,7),title=f'{name} feature importance')
        ax.set_xlabel('Importance');plt.tight_layout();plt.savefig(folder/'feature_importance.png',dpi=170);plt.close()
        joblib.dump({'model':model,'preprocessor':preprocessor,'selected_features':chosen,'threshold':threshold},folder/'model.joblib')
        LOG.info('%s test: %s',name,records[-1]);models[name]=model;probs[name]=pt
    metrics=pd.DataFrame(records);metrics.to_csv(args.output/'model_comparison_metrics.csv',index=False)
    print('\nMODEL COMPARISON\n'+metrics.to_string(index=False,float_format=lambda v:f'{v:.4f}'),flush=True)
    test=metrics[metrics.split=='test'].copy()
    for metric_name in ['precision','recall','pr_auc','auroc','f1','fp','alerts_per_100']:
        fig,ax=plt.subplots(figsize=(8,4));ax.bar(test.model,test[metric_name]);ax.set(title=f'Test {metric_name} by model',ylabel=metric_name)
        fig.tight_layout();fig.savefig(args.output/f'comparison_{metric_name}.png',dpi=170);plt.close(fig)
    valid=metrics[metrics.split=='validation'].sort_values(['pr_auc','precision'],ascending=False)
    winner=valid.iloc[0]['model']
    (args.output/'run_summary.json').write_text(json.dumps({'source_datasets':[str(x.resolve()) for x in (args.windows,args.patients,args.icu_stays)],'cohort_audit':audit,'selected_model_by_validation_pr_auc':winner,
      'target_validation_recall':args.target_recall,'feature_selection_stages':steps,
      'warning':'Diagnosis flags can reflect retrospective information; full 72-hour observation requirement may induce selection bias. ICU transfer is not the same as unmet ICU need.'},indent=2))
    LOG.info('Best model by validation PR-AUC: %s. Finished in %.1f minutes',winner,(time.monotonic()-t)/60)

if __name__=='__main__':main()
