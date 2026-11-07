#!/usr/bin/env python3
"""EBM ICU transfer model; requires pip install interpret scikit-learn pandas openpyxl matplotlib joblib."""
import argparse,json,os,inspect
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit,train_test_split
from sklearn.metrics import (roc_auc_score,average_precision_score,roc_curve,precision_recall_curve,confusion_matrix,ConfusionMatrixDisplay,accuracy_score,precision_score,recall_score,f1_score,balanced_accuracy_score,matthews_corrcoef,brier_score_loss,classification_report)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from joblib import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',default='75_Feature_Dataset.xlsx');p.add_argument('--output',default='EBM_75_SMOTE');p.add_argument('--smote-ratio',type=float,default=1.0);p.add_argument('--smote-k',type=int,default=5);p.add_argument('--target-recall',type=float,default=.60);p.add_argument('--trials',type=int,default=16);p.add_argument('--window-definition',default='unverified',choices=['unverified','verified_24_72h']);args=p.parse_args()
 from interpret.glassbox import ExplainableBoostingClassifier
 from imblearn.over_sampling import SMOTE, SMOTENC
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_excel(args.input,engine='openpyxl');target='icu_transfer'
 required=['subject_id','dataset_split',target];assert all(c in df for c in required),f'Missing required columns: {required}'
 df=df[df[target].isin([0,1])].copy();df['dataset_split']=df.dataset_split.astype(str).str.lower().str.strip()
 tr=df[df.dataset_split=='train'].copy();te=df[df.dataset_split=='test'].copy()
 assert len(tr)>0 and len(te)>0,'Train/test split missing'
 overlap=set(tr.subject_id.dropna())&set(te.subject_id.dropna())
 if overlap:raise ValueError(f'Patient leakage: {len(overlap)} subject_ids appear in train and test')
 excluded={'subject_id','hadm_id','dataset_split',target}
 feats=[c for c in df if c not in excluded]
 for f in feats:tr[f]=pd.to_numeric(tr[f],errors='coerce');te[f]=pd.to_numeric(te[f],errors='coerce')
 groups=tr.subject_id.fillna(-1).astype(str)
 if groups.nunique()<10:raise ValueError('Insufficient distinct patient IDs for grouped validation')
 splitter=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=42)
 fit_idx,val_idx=next(splitter.split(tr, tr[target],groups=groups))
 fit,val=tr.iloc[fit_idx],tr.iloc[val_idx]
 if min(fit[target].nunique(),val[target].nunique(),te[target].nunique())<2:raise ValueError('Both classes required in each split')
 imputer=SimpleImputer(strategy='median',keep_empty_features=True)
 x_fit=pd.DataFrame(imputer.fit_transform(fit[feats]),columns=feats)
 x_val=pd.DataFrame(imputer.transform(val[feats]),columns=feats)
 x_test=pd.DataFrame(imputer.transform(te[feats]),columns=feats)
 y_fit=fit[target].astype(int).reset_index(drop=True);counts_before=y_fit.value_counts().reindex([0,1],fill_value=0)
 if not 0 < args.smote_ratio <= 1:raise ValueError('--smote-ratio must be in (0,1]')
 if counts_before[1]/counts_before[0] >= args.smote_ratio:raise ValueError('SMOTE ratio must exceed original minority/majority ratio')
 categorical_idx=[i for i,c in enumerate(feats) if c.startswith(('admission_type_','admission_location_'))]
 sampler=(SMOTENC(categorical_features=categorical_idx,sampling_strategy=args.smote_ratio,random_state=42,k_neighbors=args.smote_k) if categorical_idx else SMOTE(sampling_strategy=args.smote_ratio,random_state=42,k_neighbors=args.smote_k))
 x_resampled,y_resampled=sampler.fit_resample(x_fit,y_fit)
 x_resampled=pd.DataFrame(x_resampled,columns=feats);y_resampled=pd.Series(y_resampled,name=target)
 counts_after=y_resampled.value_counts().reindex([0,1],fill_value=0)
 fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
 for ax,counts,title in zip(axes,[counts_before,counts_after],['Before SMOTE (fit only)','After SMOTE (fit only)']):
  bars=ax.bar(['No ICU transfer','ICU transfer'],counts.values,color=['#4472c4','#e38a32']);ax.set_title(title);ax.set_ylabel('Training records');ax.set_ylim(0,max(counts_before.max(),counts_after.max())*1.16)
  for b,v in zip(bars,counts.values):ax.text(b.get_x()+b.get_width()/2,v+max(counts_before.max(),counts_after.max())*.015,f'{v:,} ({v/sum(counts):.1%})',ha='center',fontsize=10)
 fig.suptitle('Class imbalance before and after SMOTE | EBM 75 features',fontsize=14);fig.tight_layout();fig.savefig(out/'01_before_after_smote_class_imbalance.png',dpi=180,bbox_inches='tight');plt.close(fig)
 pd.DataFrame([{'stage':'Before SMOTE','non_icu':int(counts_before[0]),'icu':int(counts_before[1])},{'stage':'After SMOTE','non_icu':int(counts_after[0]),'icu':int(counts_after[1])}]).to_csv(out/'01_smote_class_counts.csv',index=False)
 print('SMOTE counts before:',counts_before.to_dict(),'after:',counts_after.to_dict(),flush=True)
 rng=np.random.default_rng(42);base={'interactions':0,'max_bins':128,'learning_rate':.025,'max_rounds':2500,'outer_bags':4,'inner_bags':0,'validation_size':.15,'early_stopping_rounds':100,'n_jobs':-1,'random_state':42}
 space={'interactions':[0,0,0,5,10], 'max_bins':[64,128,256], 'learning_rate':[.01,.025,.04,.06], 'max_rounds':[700,1200,2000,3000], 'outer_bags':[2,4,6], 'min_samples_leaf':[2,5,10,20], 'max_leaves':[2,3,4], 'smoothing_rounds':[0,50,100,200]}
 supported=set(inspect.signature(ExplainableBoostingClassifier).parameters);trials=[base.copy()]
 for _ in range(max(0,args.trials-1)):
  cfg=base.copy()
  for key,values in space.items():
   if key in supported:cfg[key]=values[int(rng.integers(len(values)))]
  trials.append(cfg)
 vy=val[target].astype(int).to_numpy();search=[];best_model=None;best_score=(-1,-1,-1);best_cfg=None
 for idx,cfg in enumerate(trials,1):
  print(f'Tuning {idx}/{len(trials)}: {cfg}',flush=True)
  try:
   m=ExplainableBoostingClassifier(**{k:v for k,v in cfg.items() if k in supported});m.fit(x_resampled,y_resampled)
   probs=m.predict_proba(x_val)[:,1];p_curve,r_curve,t_curve=precision_recall_curve(vy,probs);eligible=np.flatnonzero(r_curve[:-1]>=args.target_recall)
   if len(eligible):
    bi=max(eligible,key=lambda j:(p_curve[j],r_curve[j]));ppv=float(p_curve[bi]);cutoff=float(t_curve[bi]);rec=float(r_curve[bi])
   else:ppv=0.;cutoff=1.;rec=0.
   ap=float(average_precision_score(vy,probs));auc=float(roc_auc_score(vy,probs));score=(round(ap,4),ppv,auc)
   search.append({'trial':idx,'status':'ok','validation_auprc':ap,'validation_auroc':auc,'validation_precision_at_target_recall':ppv,'validation_recall':rec,'threshold':cutoff,**cfg})
   if score>best_score:best_score=score;best_model=m;best_cfg=cfg
  except Exception as exc:search.append({'trial':idx,'status':'failed','error':str(exc),**cfg})
 pd.DataFrame(search).to_csv(out/'09_hyperparameter_trials.csv',index=False)
 if best_model is None:raise RuntimeError('All EBM tuning trials failed; inspect 09_hyperparameter_trials.csv')
 model=best_model;(out/'best_hyperparameters.json').write_text(json.dumps(best_cfg,indent=2))
 good=pd.DataFrame([x for x in search if x['status']=='ok'])
 fig,ax=plt.subplots(figsize=(8,5));ax.scatter(good.trial,good.validation_auprc,label='Validation AUPRC');ax.plot(good.trial,good.validation_auprc,alpha=.5);ax.set(xlabel='Trial',ylabel='Average precision',title='EBM hyperparameter tuning');ax.legend();fig.tight_layout();fig.savefig(out/'09_hyperparameter_tuning.png',dpi=180);plt.close(fig)
 vp=model.predict_proba(x_val)[:,1];vy=val[target].astype(int).to_numpy();precision,recall,thresholds=precision_recall_curve(vy,vp);candidates=np.where(recall[:-1]>=args.target_recall)[0]
 if not len(candidates):raise RuntimeError('Target recall unattainable')
 best=max(candidates,key=lambda i:(precision[i],2*precision[i]*recall[i]/max(precision[i]+recall[i],1e-12),thresholds[i]))
 threshold=float(thresholds[best]);tp=model.predict_proba(x_test)[:,1];ty=te[target].astype(int).to_numpy();pred=(tp>=threshold).astype(int)
 tn,fp,fn,tpos=confusion_matrix(ty,pred,labels=[0,1]).ravel()
 metrics={'decision_threshold':threshold,'target_validation_recall':args.target_recall,'validation_recall_at_threshold':float(recall[best]),'validation_precision_at_threshold':float(precision[best]),'test_n':len(ty),'test_prevalence':float(ty.mean()),'auroc':roc_auc_score(ty,tp),'auprc_average_precision':average_precision_score(ty,tp),'accuracy':accuracy_score(ty,pred),'balanced_accuracy':balanced_accuracy_score(ty,pred),'precision_ppv':precision_score(ty,pred,zero_division=0),'sensitivity_recall':recall_score(ty,pred,zero_division=0),'specificity':float(tn/(tn+fp)),'f1_score':f1_score(ty,pred,zero_division=0),'npv':float(tn/(tn+fn)) if tn+fn else None,'mcc':matthews_corrcoef(ty,pred),'brier_score':brier_score_loss(ty,tp),'false_positive_rate':float(fp/(fp+tn)),'false_negative_rate':float(fn/(fn+tpos)),'alerts_per_100':float(100*pred.mean()),'true_negatives':int(tn),'false_positives':int(fp),'false_negatives':int(fn),'true_positives':int(tpos),'smote_ratio':args.smote_ratio,'fit_original_icu':int(counts_before[1]),'fit_original_non_icu':int(counts_before[0]),'fit_smote_icu':int(counts_after[1]),'fit_smote_non_icu':int(counts_after[0]),'smote_method':'SMOTENC' if categorical_idx else 'SMOTE','train_count':len(tr),'validation_count':len(val),'fit_count':len(fit),'window_definition':args.window_definition,'patient_group_split':True,'tuning_trials':len(trials),'best_validation_auprc':float(best_score[0])}
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2));(out/'classification_report.txt').write_text(classification_report(ty,pred,zero_division=0));pd.DataFrame({'subject_id':te.subject_id,'hadm_id':te.hadm_id if 'hadm_id' in te else np.nan,'actual':ty,'risk_score':tp,'predicted':pred}).to_csv(out/'test_predictions.csv',index=False)
 pd.DataFrame([metrics]).to_csv(out/'metrics.csv',index=False)
 def save(name):plt.tight_layout();plt.savefig(out/name,dpi=180,bbox_inches='tight');plt.close()
 fig,ax=plt.subplots(figsize=(7,5));counts=df[target].value_counts().reindex([0,1],fill_value=0);ax.bar(['No ICU transfer','ICU transfer'],counts,color=['#4472c4','#e38a32']);ax.set_ylabel('Records');ax.set_title('Overall class imbalance');[ax.text(i,v+500,f'{v:,} ({v/len(df):.2%})',ha='center') for i,v in enumerate(counts)];save('01_class_imbalance.png')
 fpr,tpr,_=roc_curve(ty,tp);plt.figure(figsize=(7,5));plt.plot(fpr,tpr,label=f'AUROC = {metrics["auroc"]:.4f}');plt.plot([0,1],[0,1],'--',color='gray');plt.xlabel('False positive rate');plt.ylabel('Sensitivity');plt.title('Test ROC curve');plt.legend();save('02_auroc.png')
 pr,re,_=precision_recall_curve(ty,tp);plt.figure(figsize=(7,5));plt.plot(re,pr,label=f'Average precision = {metrics["auprc_average_precision"]:.4f}');plt.axhline(ty.mean(),linestyle='--',color='gray',label=f'Prevalence = {ty.mean():.4f}');plt.xlabel('Recall');plt.ylabel('Precision');plt.title('Test precision-recall curve');plt.legend();save('03_auprc.png')
 fig,ax=plt.subplots(figsize=(6,5));ConfusionMatrixDisplay.from_predictions(ty,pred,display_labels=['No ICU','ICU'],cmap='Blues',ax=ax,values_format='d');ax.set_title(f'Test confusion matrix (threshold {threshold:.3f})');save('04_confusion_matrix.png')
 prob_true,prob_pred=calibration_curve(ty,tp,n_bins=10,strategy='quantile');plt.figure(figsize=(6,5));plt.plot(prob_pred,prob_true,'o-',label='EBM');plt.plot([0,1],[0,1],'--',color='gray');plt.xlabel('Mean predicted probability');plt.ylabel('Observed ICU fraction');plt.title('Test calibration');plt.legend();save('05_calibration.png')
 plt.figure(figsize=(7,5));plt.hist(tp[ty==0],bins=40,alpha=.65,label='No ICU',density=True);plt.hist(tp[ty==1],bins=40,alpha=.65,label='ICU',density=True);plt.axvline(threshold,color='black',linestyle='--',label='Threshold');plt.xlabel('Predicted ICU risk');plt.ylabel('Density');plt.title('Test risk-score distribution');plt.legend();save('06_score_distribution.png')
 imp=pd.Series(model.term_importances(),index=model.term_names_).sort_values(ascending=False).head(20).iloc[::-1];plt.figure(figsize=(9,7));plt.barh(imp.index,imp.values);plt.title('EBM global feature importance');plt.xlabel('Mean absolute contribution');save('07_feature_importance.png')
 labels=[('AUROC','auroc'),('AUPRC (average precision)','auprc_average_precision'),('Accuracy','accuracy'),('Balanced accuracy','balanced_accuracy'),('Sensitivity / recall','sensitivity_recall'),('Specificity','specificity'),('Precision / PPV','precision_ppv'),('F1 score','f1_score'),('Negative predictive value','npv'),('Matthews correlation coefficient','mcc'),('Brier score (lower is better)','brier_score'),('False-positive rate','false_positive_rate'),('False-negative rate','false_negative_rate'),('Alerts per 100 patients','alerts_per_100'),('True positives','true_positives'),('False positives','false_positives'),('True negatives','true_negatives'),('False negatives','false_negatives'),('Test sample size','test_n'),('Test ICU prevalence','test_prevalence'),('Decision threshold','decision_threshold'),('Validation recall','validation_recall_at_threshold'),('Validation precision','validation_precision_at_threshold')]
 rows=[];count_keys={'true_positives','false_positives','true_negatives','false_negatives','test_n'};decimal_keys={'mcc','brier_score','auroc','auprc_average_precision','decision_threshold'}
 for label,key in labels:
  v=metrics[key];shown=f'{v:,}' if key in count_keys else (f'{v:.4f}' if key in decimal_keys else (f'{v:.2f}' if key=='alerts_per_100' else f'{100*v:.2f}%'));rows.append((label,shown))
 pd.DataFrame(rows,columns=['Evaluation parameter','Value']).to_csv(out/'08_evaluation_metrics_table.csv',index=False)
 fig,ax=plt.subplots(figsize=(10,12));ax.axis('off');ax.set_title('EBM 75 SMOTE | Evaluation metrics',fontsize=17,fontweight='bold',pad=20)
 table=ax.table(cellText=rows,colLabels=['Evaluation parameter','Value'],colWidths=[.68,.26],loc='center',cellLoc='left',colLoc='left');table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.55)
 for (r,c),cell in table.get_celld().items():
  cell.set_linewidth(0)
  if r==0:cell.set_facecolor('#163a5c');cell.get_text().set_color('white');cell.get_text().set_weight('bold')
  else:cell.set_facecolor('#edf4f8' if r%2 else '#ffffff')
  if c==1:cell.get_text().set_ha('right')
 fig.text(.5,.02,'Test-set metrics unless explicitly labelled validation. Window label requires timestamp audit.',ha='center',fontsize=9,color='#555555');save('08_evaluation_metrics_table.png')
 dump({'model':model,'features':feats,'threshold':threshold,'imputer':imputer},out/'ebm_model.joblib')
 (out/'README.txt').write_text('SMOTE-tuned EBM ICU-transfer prediction. SMOTE applied ONLY to original fitting partition after train-only median imputation. Validation/test preserve original prevalence. IMPORTANT: The uploaded dataset does not contain timestamps or a verifiable 24-72h target definition. Do not describe these metrics as 24-72h forecasting until target derivation and feature observation cutoff are audited. Patient-level train/test leakage is checked. Validation selects threshold; test is used once for evaluation. EBM is fitted on fit partition, not refitted on validation.\n')
 print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
