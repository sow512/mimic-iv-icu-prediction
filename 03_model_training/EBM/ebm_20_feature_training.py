import os,json,argparse
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--train',default='kidney_final_train_dataset.csv');p.add_argument('--test',default='kidney_final_test_dataset.csv');p.add_argument('--output',default='EBM_20_24-72');p.add_argument('--recall',type=float,default=.60);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
try:from interpret.glassbox import ExplainableBoostingClassifier
except ImportError:raise SystemExit('Install the real EBM implementation first: python -m pip install interpret-core')
tr=pd.read_csv(a.train);te=pd.read_csv(a.test);target='icu_transfer';cols=[c for c in tr if c!=target];assert set(cols)==set(te.columns)-{target}
X=tr[cols];y=tr[target].astype(int);Xt=te[cols];yt=te[target].astype(int)
xa,xv,ya,yv=train_test_split(X,y,test_size=.20,stratify=y,random_state=42)
model=ExplainableBoostingClassifier(interactions=0,max_bins=128,learning_rate=.025,max_rounds=2000,outer_bags=4,random_state=42,n_jobs=-1)
model.fit(xa,ya)
v=model.predict_proba(xv)[:,1];t=model.predict_proba(Xt)[:,1]
pr,re,th=precision_recall_curve(yv,v)
valid=np.flatnonzero(re[:-1]>=a.recall)
threshold=float(th[valid[np.argmax(pr[:-1][valid])]]) if len(valid) else 0.5
pred=(t>=threshold).astype(int);tn,fp,fn,tp=confusion_matrix(yt,pred,labels=[0,1]).ravel()
metrics={'threshold':threshold,'target_validation_recall':a.recall,'validation_recall':float(recall_score(yv,v>=threshold)),'test_samples':len(yt),'test_positives':int(yt.sum()),'prevalence':float(yt.mean()),'AUROC':roc_auc_score(yt,t),'PR_AUC_average_precision':average_precision_score(yt,t),'accuracy':accuracy_score(yt,pred),'balanced_accuracy':balanced_accuracy_score(yt,pred),'precision_PPV':precision_score(yt,pred,zero_division=0),'recall_sensitivity':recall_score(yt,pred),'specificity':tn/(tn+fp),'F1':f1_score(yt,pred),'NPV':tn/(tn+fn),'MCC':matthews_corrcoef(yt,pred),'Brier_score':brier_score_loss(yt,t),'false_positives':int(fp),'false_negatives':int(fn),'true_positives':int(tp),'true_negatives':int(tn),'alerts_per_100':100*float(pred.mean())}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2));pd.DataFrame(list(metrics.items()),columns=['Metric','Value']).to_csv(out/'metrics.csv',index=False)
pd.DataFrame({'actual':yt,'probability':t,'prediction':pred}).to_csv(out/'test_predictions.csv',index=False)
def save(name):plt.tight_layout();plt.savefig(out/name,dpi=180,bbox_inches='tight');plt.close()
plt.figure(figsize=(9,7));plt.axis('off');rows=[[k,f'{v:.5f}' if isinstance(v,float) else str(v)] for k,v in metrics.items()];tab=plt.table(cellText=rows,colLabels=['Evaluation parameter','Test value'],loc='center',cellLoc='left');tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,1.23);save('evaluation_metrics_table.png')
fig,ax=plt.subplots();ConfusionMatrixDisplay.from_predictions(yt,pred,ax=ax,cmap='Blues',values_format='d');ax.set_title('Test confusion matrix');save('confusion_matrix.png')
fpr,tpr,_=roc_curve(yt,t);plt.plot(fpr,tpr,label=f'AUROC={metrics["AUROC"]:.3f}');plt.plot([0,1],[0,1],'--');plt.xlabel('False positive rate');plt.ylabel('True positive rate');plt.legend();save('roc_curve.png')
prec,rec,_=precision_recall_curve(yt,t);plt.plot(rec,prec,label=f'AP={metrics["PR_AUC_average_precision"]:.3f}');plt.axhline(yt.mean(),ls='--',color='gray');plt.xlabel('Recall');plt.ylabel('Precision');plt.legend();save('precision_recall_curve.png')
for data,name in [(y,'train'),(yt,'test')]:
 c=data.value_counts().reindex([0,1],fill_value=0);plt.bar(['No ICU transfer','ICU transfer'],c);plt.title(f'{name.title()} class imbalance');plt.ylabel('Windows');save(f'{name}_class_imbalance.png')
frac,mean=calibration_curve(yt,t,n_bins=10,strategy='quantile');plt.plot(mean,frac,'o-',label='EBM');plt.plot([0,1],[0,1],'--');plt.xlabel('Predicted probability');plt.ylabel('Observed fraction');plt.legend();save('calibration_curve.png')
thresholds=np.linspace(0.001,max(.10,float(np.quantile(t,.995))),160);rs=[];ps=[];fs=[]
for z in thresholds:
 q=t>=z;rs.append(recall_score(yt,q));ps.append(precision_score(yt,q,zero_division=0));fs.append(int(((yt==0)&q).sum()))
plt.plot(thresholds,rs,label='Recall');plt.plot(thresholds,ps,label='Precision');plt.axvline(threshold,ls='--',label='Chosen threshold');plt.xlabel('Threshold');plt.legend();save('threshold_tradeoff.png')
plt.hist(t[yt==0],bins=60,alpha=.6,label='No ICU transfer');plt.hist(t[yt==1],bins=60,alpha=.6,label='ICU transfer');plt.axvline(threshold,ls='--');plt.yscale('log');plt.xlabel('Predicted risk');plt.ylabel('Count (log)');plt.legend();save('predicted_risk_distribution.png')
imp=permutation_importance(model,Xt,yt,n_repeats=2,scoring='average_precision',random_state=42,n_jobs=1);order=np.argsort(imp.importances_mean)[-20:];plt.figure(figsize=(10,7));plt.barh(np.array(cols)[order],imp.importances_mean[order]);plt.xlabel('Decrease in average precision after permutation');save('feature_importance.png')
pd.DataFrame({'feature':cols,'permutation_importance_AP':imp.importances_mean}).sort_values('permutation_importance_AP',ascending=False).to_csv(out/'feature_importance.csv',index=False)
if 'kidney_disease' in tr.columns or 'icd_code' in tr.columns:pass
else:(out/'kidney_disease_graph_README.txt').write_text('Kidney diagnosis categories and ICD codes are absent from both input datasets. Disease-specific graph requires original cohort/diagnosis data with a join key. Do not infer disease from creatinine alone.\n')
import pickle
with open(out/'ebm_model.pkl','wb') as f:pickle.dump(model,f)
print(json.dumps(metrics,indent=2));print('Saved to',out.resolve())
