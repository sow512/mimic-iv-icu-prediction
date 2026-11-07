#!/usr/bin/env python3
"""EBM ICU transfer model; requires pip install interpret scikit-learn pandas openpyxl matplotlib joblib."""
import argparse,json,os
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
 p=argparse.ArgumentParser();p.add_argument('--input',default='75_Feature_Dataset.xlsx');p.add_argument('--output',default='EBM_75');p.add_argument('--target-recall',type=float,default=.60);p.add_argument('--window-definition',default='unverified',choices=['unverified','verified_24_72h']);args=p.parse_args()
 from interpret.glassbox import ExplainableBoostingClassifier
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
 model=ExplainableBoostingClassifier(interactions=0,max_bins=128,learning_rate=.025,max_rounds=2500,outer_bags=4,inner_bags=0,validation_size=.15,early_stopping_rounds=100,n_jobs=-1,random_state=42)
 model.fit(fit[feats],fit[target].astype(int))
 vp=model.predict_proba(val[feats])[:,1];vy=val[target].astype(int).to_numpy()
 precision,recall,thresholds=precision_recall_curve(vy,vp)
 candidates=np.where(recall[:-1]>=args.target_recall)[0]
 if not len(candidates):raise RuntimeError('Target recall unattainable')
 best=max(candidates,key=lambda i:(precision[i],2*precision[i]*recall[i]/max(precision[i]+recall[i],1e-12),thresholds[i]))
 threshold=float(thresholds[best]);tp=model.predict_proba(te[feats])[:,1];ty=te[target].astype(int).to_numpy();pred=(tp>=threshold).astype(int)
 tn,fp,fn,tpos=confusion_matrix(ty,pred,labels=[0,1]).ravel()
 metrics={'decision_threshold':threshold,'target_validation_recall':args.target_recall,'validation_recall_at_threshold':float(recall[best]),'validation_precision_at_threshold':float(precision[best]),'test_n':len(ty),'test_prevalence':float(ty.mean()),'auroc':roc_auc_score(ty,tp),'auprc_average_precision':average_precision_score(ty,tp),'accuracy':accuracy_score(ty,pred),'balanced_accuracy':balanced_accuracy_score(ty,pred),'precision_ppv':precision_score(ty,pred,zero_division=0),'sensitivity_recall':recall_score(ty,pred,zero_division=0),'specificity':float(tn/(tn+fp)),'f1_score':f1_score(ty,pred,zero_division=0),'npv':float(tn/(tn+fn)) if tn+fn else None,'mcc':matthews_corrcoef(ty,pred),'brier_score':brier_score_loss(ty,tp),'false_positive_rate':float(fp/(fp+tn)),'false_negative_rate':float(fn/(fn+tpos)),'alerts_per_100':float(100*pred.mean()),'true_negatives':int(tn),'false_positives':int(fp),'false_negatives':int(fn),'true_positives':int(tpos),'train_count':len(tr),'validation_count':len(val),'fit_count':len(fit),'window_definition':args.window_definition,'patient_group_split':True}
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
 rows=[]
 count_keys={'true_positives','false_positives','true_negatives','false_negatives','test_n'}
 decimal_keys={'mcc','brier_score','auroc','auprc_average_precision','decision_threshold'}
 for label,key in labels:
  v=metrics[key]
  shown=f'{v:,}' if key in count_keys else (f'{v:.4f}' if key in decimal_keys else (f'{v:.2f}' if key=='alerts_per_100' else f'{100*v:.2f}%'))
  rows.append((label,shown))
 pd.DataFrame(rows,columns=['Evaluation parameter','Value']).to_csv(out/'08_evaluation_metrics_table.csv',index=False)
 fig,ax=plt.subplots(figsize=(10,12));ax.axis('off');ax.set_title('EBM ICU-transfer prediction | Evaluation metrics',fontsize=17,fontweight='bold',pad=20)
 table=ax.table(cellText=rows,colLabels=['Evaluation parameter','Value'],colWidths=[.68,.26],loc='center',cellLoc='left',colLoc='left');table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.55)
 for (r,c),cell in table.get_celld().items():
  cell.set_linewidth(0)
  if r==0:cell.set_facecolor('#163a5c');cell.get_text().set_color('white');cell.get_text().set_weight('bold')
  else:cell.set_facecolor('#edf4f8' if r%2 else '#ffffff')
  if c==1:cell.get_text().set_ha('right')
 fig.text(.5,.02,'Test-set metrics unless explicitly labelled validation. Window label requires timestamp audit.',ha='center',fontsize=9,color='#555555')
 save('08_evaluation_metrics_table.png')
 dump({'model':model,'features':feats,'threshold':threshold},out/'ebm_model.joblib')
 (out/'README.txt').write_text('EBM ICU-transfer prediction. IMPORTANT: The uploaded dataset does not contain timestamps or a verifiable 24-72h target definition. Do not describe these metrics as 24-72h forecasting until target derivation and feature observation cutoff are audited. Patient-level train/test leakage is checked. Validation selects threshold; test is used once for evaluation. EBM is fitted on fit partition, not refitted on validation.\n')
 print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
