#!/usr/bin/env python3
"""Train and evaluate an Explainable Boosting Machine on the supplied MIMIC-IV 200-feature workbook.

Install: pip install 'interpret>=0.6' pandas openpyxl scikit-learn matplotlib joblib
Run: python EBM_200.py --input 'cleaned dataset (200 features).xlsx' --output EBM_200

IMPORTANT: The workbook does not include prediction/transfer timestamps. Verify upstream that
icu_transfer means transfer 24-72 hours AFTER feature measurement and excludes earlier ICU transfer.
"""
import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
    confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score,
    brier_score_loss, accuracy_score, matthews_corrcoef, log_loss, roc_curve)
from sklearn.model_selection import GroupShuffleSplit, train_test_split, ParameterSampler

LOG=logging.getLogger('ebm')
TARGET='icu_transfer'
ID_COLUMNS=['subject_id','hadm_id','stay_id']

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='cleaned dataset (200 features).xlsx')
    p.add_argument('--output',default='EBM_200')
    p.add_argument('--trials',type=int,default=12)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--min-recall',type=float,default=.65)
    p.add_argument('--max-alert-rate',type=float,default=.20)
    p.add_argument('--interactions',type=int,default=0)
    return p.parse_args()

def load_sheet(path,name):
    LOG.info('Reading sheet: %s',name)
    df=pd.read_excel(path,sheet_name=name,engine='openpyxl')
    df.columns=df.columns.astype(str).str.strip()
    if TARGET not in df:raise ValueError(f'Missing {TARGET} in {name}')
    if df[TARGET].isna().any():raise ValueError(f'Missing labels in {name}')
    if not set(df[TARGET].unique()).issubset({0,1,False,True}):raise ValueError(f'{TARGET} must be binary')
    return df

def prepare(train,test):
    excluded=set(ID_COLUMNS+[TARGET])
    forbidden=('icu_transfer_time','hours_to_icu','time_to_icu','future_icu','icu_admit','icu_intime','icu_outtime','post_outcome')
    features=[c for c in train if c in test and c not in excluded and not any(s in c.lower() for s in forbidden)]
    if not features:raise ValueError('No shared predictor columns found')
    xtr=train[features].apply(pd.to_numeric,errors='coerce')
    xte=test[features].apply(pd.to_numeric,errors='coerce')
    dropped=sorted(set(xtr.columns[xtr.isna().all()].tolist()+xtr.columns[xtr.nunique(dropna=True)<=1].tolist()))
    xtr=xtr.drop(columns=dropped);xte=xte.drop(columns=dropped)
    return xtr.astype('float32'),xte.astype('float32'),dropped

def choose_threshold(y,p,min_recall,max_alert_rate):
    precision,recall,thresholds=precision_recall_curve(y,p)
    alert_rates=np.array([(p>=t).mean() for t in thresholds])
    eligible=np.where((recall[:-1]>=min_recall)&(alert_rates<=max_alert_rate))[0]
    rule='highest_precision_at_recall_and_alert_budget'
    if not len(eligible):
        eligible=np.where(recall[:-1]>=min_recall)[0]
        rule='highest_precision_at_min_recall_budget_infeasible'
    if not len(eligible):
        f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
        idx=int(np.argmax(f1));rule='max_f1_recall_target_infeasible'
    else:idx=int(eligible[np.argmax(precision[eligible])])
    return float(thresholds[idx]),rule

def compute_metrics(y,p,threshold):
    pred=(p>=threshold).astype(int);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),
      'prevalence':float(np.mean(y)),'threshold':float(threshold),
      'sensitivity_recall':float(recall_score(y,pred,zero_division=0)),
      'specificity':float(tn/max(tn+fp,1)),'precision_ppv':float(precision_score(y,pred,zero_division=0)),
      'f1':float(f1_score(y,pred,zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
      'accuracy':float(accuracy_score(y,pred)),'npv':float(tn/max(tn+fn,1)),
      'mcc':float(matthews_corrcoef(y,pred)),'log_loss':float(log_loss(y,np.clip(p,1e-12,1-1e-12))),
      'brier_score':float(brier_score_loss(y,p)),'true_positives':int(tp),'true_negatives':int(tn),
      'false_positives':int(fp),'false_negatives':int(fn),'test_alerts_per_100_windows':float(100*pred.mean())}

def plot_results(y,p,threshold,history,test,out):
    pred=(p>=threshold).astype(int)
    cm=confusion_matrix(y,pred,labels=[0,1])
    fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center')
    ax.set_xticks([0,1],['No transfer','ICU transfer']);ax.set_yticks([0,1],['No transfer','ICU transfer'])
    ax.set(xlabel='Predicted',ylabel='Actual',title=f'Confusion matrix (threshold={threshold:.4f})')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=180);plt.close(fig)
    fpr,tpr,_=roc_curve(y,p);pr,rec,_=precision_recall_curve(y,p)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].plot(fpr,tpr);axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].set(title='ROC curve',xlabel='False positive rate',ylabel='True positive rate')
    axes[1].plot(rec,pr);axes[1].axhline(y.mean(),ls='--',color='gray');axes[1].set(title='Precision-recall curve',xlabel='Recall',ylabel='Precision')
    fig.tight_layout();fig.savefig(out/'roc_pr_curves.png',dpi=180);plt.close(fig)
    frac,mean=calibration_curve(y,p,n_bins=10,strategy='quantile')
    fig,ax=plt.subplots(figsize=(7,5));ax.plot(mean,frac,'o-');ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(xlabel='Mean predicted probability',ylabel='Observed transfer rate',title='Calibration curve');fig.tight_layout();fig.savefig(out/'calibration_curve.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));ax.hist(p[y==0],bins=40,density=True,alpha=.55,label='No ICU')
    ax.hist(p[y==1],bins=40,density=True,alpha=.55,label='ICU');ax.axvline(threshold,ls='--',color='black')
    ax.set(xlabel='Predicted probability',ylabel='Density',title='Probability distribution');ax.legend()
    fig.tight_layout();fig.savefig(out/'probability_distribution.png',dpi=180);plt.close(fig)
    rows=[]
    for t in np.unique(np.r_[np.linspace(.001,.999,200),threshold]):
        q=p>=t;tp=int(np.sum(q&(y==1)));fp=int(np.sum(q&(y==0)));fn=int(np.sum(~q&(y==1)))
        precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
        rows.append([t,precision,recall,2*precision*recall/max(precision+recall,1e-12),100*q.mean(),fp])
    td=pd.DataFrame(rows,columns=['threshold','precision','recall','f1','alerts_per_100','false_positives'])
    td.to_csv(out/'test_threshold_sensitivity_analysis.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5))
    for col in ['precision','recall','f1']:ax.plot(td.threshold,td[col],label=col.title())
    ax.axvline(threshold,ls='--',color='black',label='Selected threshold');ax.set(xlabel='Threshold',ylabel='Score',title='Threshold sensitivity');ax.legend()
    fig.tight_layout();fig.savefig(out/'threshold_metrics.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));ax.plot(td.threshold,td.alerts_per_100);ax.axvline(threshold,ls='--',color='black')
    ax.set(xlabel='Threshold',ylabel='Alerts / 100 windows',title='Alert burden');fig.tight_layout();fig.savefig(out/'alert_burden.png',dpi=180);plt.close(fig)
    hd=pd.DataFrame(history);fig,ax=plt.subplots(figsize=(8,5));ax.bar(hd.trial.astype(str),hd.validation_pr_auc)
    ax.set(xlabel='Trial',ylabel='Validation PR-AUC',title='Hyperparameter trial comparison');fig.tight_layout();fig.savefig(out/'hyperparameter_pr_auc.png',dpi=180);plt.close(fig)
    renal={'Creatinine (mg/dL)':'lab_50912_last','BUN (mg/dL)':'lab_51006_last','Potassium (mEq/L)':'lab_50971_last'}
    available=[(label,col) for label,col in renal.items() if col in test.columns]
    if available:
        fig,axes=plt.subplots(1,len(available),figsize=(5*len(available),4.5),squeeze=False);summary=[]
        for ax,(label,col) in zip(axes[0],available):
            values=pd.to_numeric(test[col],errors='coerce').to_numpy(dtype=float)
            groups=[values[(y==cls)&np.isfinite(values)] for cls in [0,1]]
            for cls,name in [(0,'No ICU transfer'),(1,'ICU transfer')]:
                v=values[(y==cls)&np.isfinite(values)]
                summary.append({'biomarker':label,'group':name,'n':len(v),'median':float(np.median(v)) if len(v) else None,
                  'q1':float(np.quantile(v,.25)) if len(v) else None,'q3':float(np.quantile(v,.75)) if len(v) else None})
            if all(len(v) for v in groups):ax.boxplot(groups,tick_labels=['No transfer','ICU transfer'],showfliers=False)
            ax.set_title(label)
        fig.suptitle('Renal-related lab features vs ICU transfer (not disease diagnoses)')
        fig.tight_layout();fig.savefig(out/'renal_biomarker_feature_engineering.png',dpi=180);plt.close(fig)
        pd.DataFrame(summary).to_csv(out/'renal_biomarker_summary.csv',index=False)

def main():
    args=parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if not 0<args.min_recall<=1 or not 0<args.max_alert_rate<=1:raise ValueError('min-recall and max-alert-rate must be in (0,1]')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train=load_sheet(args.input,'Training Data');test=load_sheet(args.input,'Testing Data')
    X,Xtest,dropped=prepare(train,test);y=train[TARGET].astype(int).to_numpy();yt=test[TARGET].astype(int).to_numpy()
    if len(np.unique(y))<2 or len(np.unique(yt))<2:raise ValueError('Both classes must exist in train and test')
    if 'subject_id' in train and 'subject_id' in test:
        overlap=set(train.subject_id.dropna())&set(test.subject_id.dropna())
        if overlap:raise ValueError(f'Patient leakage: {len(overlap)} subject_ids overlap train/test')
        splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=args.seed)
        fit_idx,val_idx=next(splitter.split(X,y,groups=train.subject_id.to_numpy()))
    else:
        LOG.warning('No subject_id in both sheets: cannot verify patient-level separation')
        fit_idx,val_idx=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=args.seed)
    Xfit,Xval=X.iloc[fit_idx],X.iloc[val_idx];yfit,yval=y[fit_idx],y[val_idx]
    if len(np.unique(yfit))<2 or len(np.unique(yval))<2:raise ValueError('Validation split has only one class')
    grid=dict(learning_rate=[.005,.01,.02,.03],max_leaves=[2,3,4],min_samples_leaf=[10,20,50,100],max_bins=[128,256],smoothing_rounds=[0,100,200])
    trials=list(ParameterSampler(grid,n_iter=args.trials,random_state=args.seed));history=[];best_score=-np.inf
    for i,config in enumerate(trials,1):
        LOG.info('Trial %d/%d: %s',i,len(trials),config)
        model=ExplainableBoostingClassifier(random_state=args.seed,interactions=args.interactions,n_jobs=-1,
            outer_bags=4,validation_size=.15,early_stopping_rounds=75,**config)
        npos=int(yfit.sum());nneg=len(yfit)-npos
        weights=np.where(yfit==1,len(yfit)/(2*npos),len(yfit)/(2*nneg))
        model.fit(Xfit,yfit,sample_weight=weights);pval=model.predict_proba(Xval)[:,1]
        score=float(average_precision_score(yval,pval))
        history.append(dict(trial=i,validation_pr_auc=score,validation_roc_auc=float(roc_auc_score(yval,pval)),**config))
        if score>best_score:best_score=score;best_config=config;best_val_prob=pval
    pd.DataFrame(history).sort_values('validation_pr_auc',ascending=False).to_csv(out/'hyperparameter_trials.csv',index=False)
    threshold,rule=choose_threshold(yval,best_val_prob,args.min_recall,args.max_alert_rate)
    val_metrics=compute_metrics(yval,best_val_prob,threshold)
    full_model=ExplainableBoostingClassifier(random_state=args.seed,interactions=args.interactions,n_jobs=-1,
        outer_bags=4,validation_size=.15,early_stopping_rounds=75,**best_config)
    npos=int(y.sum());nneg=len(y)-npos;full_weights=np.where(y==1,len(y)/(2*npos),len(y)/(2*nneg))
    full_model.fit(X,y,sample_weight=full_weights);ptest=full_model.predict_proba(Xtest)[:,1]
    result=compute_metrics(yt,ptest,threshold)
    result.update({'threshold_selection_rule':rule,'validation_metrics':val_metrics,'best_hyperparameters':best_config,
      'n_train':int(len(y)),'n_test':int(len(yt)),'n_features':int(X.shape[1]),'dropped_features':dropped,
      'target_definition_warning':'Verify upstream that icu_transfer means transfer 24-72h AFTER feature measurement and excludes earlier ICU transfer.'})
    (out/'test_metrics.json').write_text(json.dumps(result,indent=2))
    importance=pd.DataFrame({'feature':full_model.term_names_,'importance':full_model.term_importances()}).sort_values('importance',ascending=False)
    importance.to_csv(out/'ebm_global_importance.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,8));top=importance.head(20).iloc[::-1];ax.barh(top.feature,top.importance)
    ax.set_xlabel('Mean absolute EBM term contribution');fig.tight_layout();fig.savefig(out/'top_20_feature_importance.png',dpi=170);plt.close(fig)
    plot_results(yt,ptest,threshold,history,test,out)
    numeric=[{'metric':k,'value':v} for k,v in result.items() if isinstance(v,(int,float))]
    pd.DataFrame(numeric).to_csv(out/'evaluation_metrics.csv',index=False)
    joblib.dump({'model':full_model,'features':list(X.columns),'threshold':threshold,'hyperparameters':best_config},out/'ebm_icu_transfer.joblib')
    print(json.dumps(result,indent=2));LOG.info('All outputs saved to %s',out.resolve())

if __name__=='__main__':main()
