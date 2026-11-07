#!/usr/bin/env python3
"""EBM vs SMOTE (0.2, 0.5, 1.0) on the supplied 200-feature workbook.
Install: pip install interpret imbalanced-learn pandas openpyxl scikit-learn matplotlib joblib
Run: python EBM_SMOTE_200.py --input 'cleaned dataset (200 features)(1).xlsx' --output EBM_SMOTE_200
Optional disease graph: --diagnoses kidney_diagnoses.csv (columns: subject_id or hadm_id, and kidney_disease or icd_code)
SMOTE operates on training rows only. All evaluation uses original unsampled data.
"""
import argparse, json, logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from interpret.glassbox import ExplainableBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    brier_score_loss, log_loss, roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve

LOG=logging.getLogger('smote_ebm')
TARGET='icu_transfer'

def args_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',default='cleaned dataset (200 features).xlsx')
    p.add_argument('--output',default='EBM_SMOTE_200')
    p.add_argument('--diagnoses',default=None)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--min-recall',type=float,default=.60)
    p.add_argument('--max-alert-rate',type=float,default=.15)
    p.add_argument('--smote-ratios',nargs='+',type=float,default=[.2,.5,1.0])
    p.add_argument('--max-rounds',type=int,default=1500)
    return p.parse_args()

def savefig(path):
    plt.tight_layout();plt.savefig(path,dpi=190,bbox_inches='tight');plt.close()

def metric(y,p,t):
    z=(p>=t).astype(int);tn,fp,fn,tp=map(int,confusion_matrix(y,z,labels=[0,1]).ravel())
    return dict(threshold=float(t),n=int(len(y)),prevalence=float(y.mean()),
      accuracy=float(accuracy_score(y,z)),balanced_accuracy=float(balanced_accuracy_score(y,z)),
      precision_ppv=float(precision_score(y,z,zero_division=0)),sensitivity_recall=float(recall_score(y,z,zero_division=0)),
      specificity=tn/max(tn+fp,1),npv=tn/max(tn+fn,1),f1=float(f1_score(y,z,zero_division=0)),
      auroc=float(roc_auc_score(y,p)),auprc_average_precision=float(average_precision_score(y,p)),
      mcc=float(matthews_corrcoef(y,z)),brier_score=float(brier_score_loss(y,p)),
      log_loss=float(log_loss(y,np.clip(p,1e-12,1-1e-12))),true_positives=tp,true_negatives=tn,
      false_positives=fp,false_negatives=fn,alerts_per_100_windows=float(100*z.mean()),
      false_positive_rate=float(fp/max(fp+tn,1)))

def choose_threshold(y,p,recall_goal,alert_budget):
    prec,rec,ts=precision_recall_curve(y,p);feasible=np.flatnonzero(rec[:-1]>=recall_goal)
    if not len(feasible):raise RuntimeError('No threshold satisfies validation recall requirement')
    ix=feasible[-1];t=float(ts[ix]);m=metric(y,p,t)
    return t,m['alerts_per_100_windows']<=100*alert_budget

def disease_plot(diag_path,train,test,out):
    if not diag_path:
        (out/'kidney_disease_graph_README.txt').write_text('No diagnosis columns exist in the 200-feature workbook. Supply --diagnoses FILE with subject_id or hadm_id and kidney_disease or icd_code. Do not interpret renal laboratory values as diagnoses.\n');return
    d=pd.read_csv(diag_path);key=next((c for c in ['hadm_id','subject_id'] if c in d and c in train and c in test),None)
    disease=next((c for c in ['kidney_disease','disease','diagnosis','icd_code','icd_code_10'] if c in d),None)
    if key is None or disease is None:raise ValueError('Diagnosis file requires a matching ID and kidney_disease/diagnosis/icd_code column')
    ids=pd.concat([train[[key]],test[[key]]]).drop_duplicates()
    matched=d.merge(ids,on=key,how='inner')[[key,disease]].dropna().drop_duplicates()
    counts=matched[disease].astype(str).value_counts().head(20)
    counts.rename_axis('diagnosis').reset_index(name='unique_id_count').to_csv(out/'kidney_disease_counts.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,7));ax.barh(counts.index[::-1],counts.values[::-1]);ax.set_xlabel('Unique matching IDs per diagnosis')
    ax.set_title('Kidney diagnoses (multiple diagnoses per patient possible)');savefig(out/'kidney_disease_distribution.png')

def main():
    a=args_parser();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if not 0<a.min_recall<=1 or not 0<a.max_alert_rate<=1:raise ValueError('Invalid recall or alert budget')
    LOG.info('Reading training sheet (large Excel files can take minutes)')
    train=pd.read_excel(a.input,sheet_name='Training Data',engine='openpyxl')
    LOG.info('Reading testing sheet')
    test=pd.read_excel(a.input,sheet_name='Testing Data',engine='openpyxl')
    train.columns=train.columns.astype(str).str.strip();test.columns=test.columns.astype(str).str.strip()
    if TARGET not in train or TARGET not in test:raise ValueError('Missing icu_transfer target')
    for df in [train,test]:
        if df[TARGET].isna().any() or not set(df[TARGET].unique()).issubset({0,1}):raise ValueError('Target must be complete binary 0/1')
    if 'subject_id' not in train or 'subject_id' not in test:raise ValueError('subject_id required for patient-disjoint validation')
    if train.subject_id.isna().any() or test.subject_id.isna().any():raise ValueError('Missing subject_id')
    overlap=set(train.subject_id)&set(test.subject_id)
    if overlap:raise ValueError(f'Patient leakage across train/test: {len(overlap)} subject IDs')
    forbidden=('icu_transfer_time','hours_to_icu','time_to_icu','future_icu','icu_admit','icu_intime','icu_outtime','post_outcome')
    features=[c for c in train if c in test and c not in ['subject_id','hadm_id','stay_id',TARGET] and not any(s in c.lower() for s in forbidden)]
    if not features:raise ValueError('No shared predictors')
    X=train[features].apply(pd.to_numeric,errors='coerce').astype('float32')
    Xt=test[features].apply(pd.to_numeric,errors='coerce').astype('float32')
    keep=X.columns[X.notna().any() & (X.nunique(dropna=True)>1)];X=X[keep];Xt=Xt[keep]
    y=train[TARGET].astype(int).to_numpy();yt=test[TARGET].astype(int).to_numpy()
    if min(np.bincount(y))==0 or min(np.bincount(yt))==0:raise ValueError('Both classes required')
    split=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=a.seed);fi,vi=next(split.split(X,y,groups=train.subject_id))
    if set(train.subject_id.iloc[fi])&set(train.subject_id.iloc[vi]):raise ValueError('Validation patient leakage')
    yf=y[fi];yv=y[vi]
    if min(np.bincount(yf))<2 or min(np.bincount(yv))<2:raise ValueError('Insufficient positives in fit/validation')
    missing_cols=X.columns[X.iloc[fi].isna().any()].tolist()
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    xf=imputer.fit_transform(X.iloc[fi]).astype('float32');xv=imputer.transform(X.iloc[vi]).astype('float32')
    xfull=imputer.transform(X).astype('float32');xt=imputer.transform(Xt).astype('float32')
    miss_idx=[X.columns.get_loc(c) for c in missing_cols]
    if miss_idx:
        xf=np.column_stack([xf,X.iloc[fi,miss_idx].isna().to_numpy(dtype='float32')])
        xv=np.column_stack([xv,X.iloc[vi,miss_idx].isna().to_numpy(dtype='float32')])
        xfull=np.column_stack([xfull,X.iloc[:,miss_idx].isna().to_numpy(dtype='float32')])
        xt=np.column_stack([xt,Xt.iloc[:,miss_idx].isna().to_numpy(dtype='float32')])
    feature_names=list(keep)+[c+'_missing' for c in missing_cols]
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler();xf=scaler.fit_transform(xf).astype('float32');xv=scaler.transform(xv).astype('float32')
    xfull=scaler.transform(xfull).astype('float32');xt=scaler.transform(xt).astype('float32')
    ratios=[('Baseline',None)]+[(f'SMOTE_{r:g}',r) for r in a.smote_ratios]
    history=[];test_rows=[];curves={}
    for name,ratio in ratios:
        LOG.info('Fitting %s',name);original_ratio=float(np.sum(yf==1)/np.sum(yf==0))
        if ratio is not None and ratio<=original_ratio:
            LOG.warning('Skipping %s: ratio %.3f is <= original ratio %.3f',name,ratio,original_ratio);continue
        if ratio is None:xx,yy=xf,yf
        else:xx,yy=SMOTE(sampling_strategy=ratio,random_state=a.seed,k_neighbors=min(5,int(sum(yf==1))-1)).fit_resample(xf,yf)
        params=dict(random_state=a.seed,interactions=0,n_jobs=-1,outer_bags=2,
                    learning_rate=.02,max_leaves=3,min_samples_leaf=30,max_bins=128,
                    max_rounds=a.max_rounds,early_stopping_rounds=75)
        model=ExplainableBoostingClassifier(**params);model.fit(xx,yy);pv=model.predict_proba(xv)[:,1]
        threshold,budget_feasible=choose_threshold(yv,pv,a.min_recall,a.max_alert_rate);vm=metric(yv,pv,threshold)
        history.append(dict(model=name,smote_ratio=ratio if ratio is not None else 0,
            original_fit_negative=int(sum(yf==0)),original_fit_positive=int(sum(yf==1)),
            resampled_negative=int(sum(yy==0)),resampled_positive=int(sum(yy==1)),
            alert_budget_feasible=budget_feasible,**vm))
        pt=model.predict_proba(xt)[:,1];tm=metric(yt,pt,threshold);test_rows.append(dict(model=name,**tm));curves[name]=(pt,threshold)
        LOG.info('%s validation recall %.3f test recall %.3f test precision %.3f FP %d',name,vm['sensitivity_recall'],tm['sensitivity_recall'],tm['precision_ppv'],tm['false_positives'])
        joblib.dump(dict(model=model,imputer=imputer,scaler=scaler,features=list(keep),missing_cols=missing_cols,
            feature_names=feature_names,threshold=threshold,trained_on='fit split only',smote_ratio=ratio),out/f'{name}_model.joblib')
        pd.DataFrame({'actual':yt,'probability':pt,'prediction':(pt>=threshold).astype(int)}).to_csv(out/f'{name}_test_predictions.csv',index=False)
        importance=pd.DataFrame({'feature':model.term_names_,'importance':model.term_importances()}).sort_values('importance',ascending=False)
        importance.to_csv(out/f'{name}_feature_importance.csv',index=False)
        fig,ax=plt.subplots(figsize=(10,7));top=importance.head(20).iloc[::-1];ax.barh(top.feature,top.importance);ax.set_title(f'{name}: top 20 EBM terms');savefig(out/f'{name}_feature_importance.png')
        cm=confusion_matrix(yt,pt>=threshold,labels=[0,1]);fig,ax=plt.subplots(figsize=(6,5));ax.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',color='black',fontsize=14)
        ax.set_xticks([0,1],['No ICU','ICU']);ax.set_yticks([0,1],['No ICU','ICU']);ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(f'{name} confusion matrix');savefig(out/f'{name}_confusion_matrix.png')
        fig,ax=plt.subplots(figsize=(7,5));frac,means=calibration_curve(yt,pt,n_bins=10,strategy='quantile');ax.plot(means,frac,'o-',label=name);ax.plot([0,1],[0,1],'--',label='Ideal');ax.set(xlabel='Predicted risk',ylabel='Observed rate',title=f'{name} calibration');ax.legend();savefig(out/f'{name}_calibration.png')
        fig,ax=plt.subplots(figsize=(7,5));ax.hist(pt[yt==0],bins=40,alpha=.5,density=True,label='No ICU');ax.hist(pt[yt==1],bins=40,alpha=.5,density=True,label='ICU');ax.axvline(threshold,color='black',ls='--');ax.legend();ax.set_title(f'{name} predicted probabilities');savefig(out/f'{name}_probability_distribution.png')
        ts=np.unique(np.r_[np.quantile(pt,np.linspace(0,1,201)),threshold]);thr=[]
        for t in ts:
            mm=metric(yt,pt,float(t));thr.append({k:mm[k] for k in ['threshold','precision_ppv','sensitivity_recall','f1','false_positives','alerts_per_100_windows']})
        pd.DataFrame(thr).to_csv(out/f'{name}_threshold_sensitivity.csv',index=False)
    if not test_rows:raise RuntimeError('No variants were trained')
    val=pd.DataFrame(history);te=pd.DataFrame(test_rows);val.to_csv(out/'validation_metrics.csv',index=False);te.to_csv(out/'test_metrics.csv',index=False)
    (out/'test_metrics.json').write_text(json.dumps(test_rows,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,yy,title in [(axes[0],y,'Original training'),(axes[1],yt,'Untouched test')]:
        counts=np.bincount(yy,minlength=2);ax.bar(['No ICU','ICU'],counts);ax.set_title(title)
        for i,c in enumerate(counts):ax.text(i,c,f'{c:,} ({c/len(yy):.1%})',ha='center',va='bottom')
    savefig(out/'original_class_imbalance.png')
    fig,ax=plt.subplots(figsize=(9,5));labels=val.model.tolist();x=np.arange(len(labels));w=.35
    ax.bar(x-w/2,val.resampled_negative,width=w,label='No ICU');ax.bar(x+w/2,val.resampled_positive,width=w,label='ICU');ax.set_xticks(x,labels);ax.set_ylabel('Fit rows');ax.legend();ax.set_title('Class counts before and after SMOTE');savefig(out/'smote_class_balance.png')
    fig,ax=plt.subplots(figsize=(7,6))
    for name,(p,t) in curves.items():
        fpr,tpr,_=roc_curve(yt,p);ax.plot(fpr,tpr,label=f'{name}: {roc_auc_score(yt,p):.3f}')
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Held-out test ROC');ax.legend();savefig(out/'roc_comparison.png')
    fig,ax=plt.subplots(figsize=(7,6))
    for name,(p,t) in curves.items():
        precision,recall,_=precision_recall_curve(yt,p);ax.plot(recall,precision,label=f'{name}: {average_precision_score(yt,p):.3f}')
    ax.axhline(yt.mean(),ls='--',color='gray',label='Prevalence');ax.set(xlabel='Recall',ylabel='Precision',title='Held-out test PR curves');ax.legend();savefig(out/'pr_comparison.png')
    fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(te));ax.bar(x-.2,te.false_positives,width=.4,label='False positives');ax.bar(x+.2,te.true_positives,width=.4,label='True positives');ax.set_xticks(x,te.model);ax.set_ylabel('Prediction windows');ax.legend();ax.set_title('Alert outcomes by model');savefig(out/'false_positives_comparison.png')
    fig,ax=plt.subplots(figsize=(9,5));ax.scatter(te.sensitivity_recall,te.false_positives,s=90)
    for row in te.itertuples():ax.annotate(row.model,(row.sensitivity_recall,row.false_positives),xytext=(5,5),textcoords='offset points')
    ax.axvline(a.min_recall,ls='--',color='gray');ax.set(xlabel='Test recall',ylabel='False positives',title='Recall vs false positives');savefig(out/'false_positives_vs_recall.png')
    fig,ax=plt.subplots(figsize=(10,5));cols=['accuracy','precision_ppv','sensitivity_recall','specificity','f1','auroc','auprc_average_precision']
    for i,row in te.iterrows():ax.plot(cols,[row[c] for c in cols],marker='o',label=row.model)
    ax.tick_params(axis='x',rotation=30);ax.legend();ax.set_ylim(0,1.05);ax.set_title('Evaluation metric comparison');savefig(out/'metric_comparison.png')
    table_cols=['model','threshold','accuracy','balanced_accuracy','precision_ppv','sensitivity_recall','specificity','npv','f1','auroc','auprc_average_precision','mcc','brier_score','log_loss','true_positives','true_negatives','false_positives','false_negatives','alerts_per_100_windows']
    data=te[table_cols].copy()
    for c in data:
        if c!='model':data[c]=data[c].map(lambda v:f'{v:,.0f}' if c in ['true_positives','true_negatives','false_positives','false_negatives'] else f'{v:.4f}')
    fig,ax=plt.subplots(figsize=(17,11));ax.axis('off');tab=ax.table(cellText=data.drop(columns='model').T.values,colLabels=list(data.model),rowLabels=table_cols[1:],cellLoc='center',loc='center',bbox=[.20,.02,.78,.92]);tab.auto_set_font_size(False);tab.set_fontsize(10)
    for (r,c),cell in tab.get_celld().items():
        if r==0:cell.set_facecolor('#DCEAF3');cell.set_text_props(weight='bold')
        if r>0 and r%2==0:cell.set_facecolor('#F2F5F8')
    ax.set_title('EBM 200-feature SMOTE comparison | Held-out test evaluation',fontsize=16,pad=20);savefig(out/'evaluation_parameters_table.png')
    disease_plot(a.diagnoses,train,test,out)
    (out/'README_results.txt').write_text('Validation threshold selected to target recall >= %.2f. Test set untouched; test recall may differ.\nSMOTE applied only to fit rows; patient-disjoint validation. EBM models trained on fit subset for fair comparison.\nNo test-based model selection. SMOTE on one-hot/missingness features can create fractional synthetic values.\n24-72h target timing cannot be independently verified from this workbook.\n'%a.min_recall)
    print('\nHELD-OUT TEST RESULTS\n',te[['model','sensitivity_recall','precision_ppv','auroc','auprc_average_precision','false_positives','false_negatives']].to_string(index=False))
    print('\nOutputs:',out.resolve())

if __name__=='__main__':main()
