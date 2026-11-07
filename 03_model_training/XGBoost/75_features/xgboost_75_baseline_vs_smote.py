"""Standalone exploratory 75-feature XGBoost: no oversampling vs SMOTENC.
Requires the already-engineered 75-feature Excel workbook with dataset_split.
Never changes the workbook or the previously trained model.
"""
import os
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report, brier_score_loss, log_loss,
    accuracy_score, f1_score, precision_score, recall_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTENC, SMOTE

# === EDIT ONLY THIS PATH IF NEEDED ===
DATA_FILE = Path(
    os.environ["ICU_75_FEATURE_WORKBOOK"]
)
OUT = DATA_FILE.parent / 'SMOTE_75_XGBOOST_EXPERIMENT'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
TARGET_RECALL = 0.70
SMOTE_RATIO = 0.30  # minority/majority AFTER SMOTENC, e.g. 30%; increase to 1.0 for full balance
K_NEIGHBORS = 5
PARAMS = dict(n_estimators=250, max_depth=4, learning_rate=0.05,
              subsample=0.9, colsample_bytree=0.8, min_child_weight=5,
              reg_lambda=5, scale_pos_weight=1.0, tree_method='hist',
              objective='binary:logistic', eval_metric='logloss',
              random_state=SEED, n_jobs=4)

print('STEP 1: Loading pre-engineered 75-feature workbook...', flush=True)
if not DATA_FILE.exists():
    raise FileNotFoundError(f'Cannot find {DATA_FILE}')
df = pd.read_excel(DATA_FILE, sheet_name=0, engine='openpyxl')
needed = {'subject_id','hadm_id','dataset_split','icu_transfer'}
if needed - set(df.columns):
    raise ValueError(f'Missing required columns: {needed-set(df.columns)}')
if df.hadm_id.duplicated().any():
    raise ValueError('Duplicate hadm_id')
if not df.icu_transfer.isin([0,1]).all():
    raise ValueError('Target must be 0/1')
df.dataset_split = df.dataset_split.astype(str).str.strip().str.lower()
if set(df.dataset_split.unique()) != {'train','test'}:
    raise ValueError(f'Expected train/test splits; got {df.dataset_split.unique()}')
exclude = {'subject_id','hadm_id','dataset_split','icu_transfer','early_discharge_before_72h'}
feature_cols = [c for c in df.columns if c not in exclude]
if len(feature_cols) != 75:
    raise ValueError(f'Expected 75 predictors, found {len(feature_cols)}. Check workbook.')
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='raise')
train = df[df.dataset_split=='train'].reset_index(drop=True)
test = df[df.dataset_split=='test'].reset_index(drop=True)
if set(train.subject_id) & set(test.subject_id):
    raise ValueError('Patient leakage across original train/test splits')
print(f'Rows: train={len(train):,}, test={len(test):,}; features={len(feature_cols)}')
print(f'Positives: train={int(train.icu_transfer.sum()):,}, test={int(test.icu_transfer.sum()):,}')

# Validation split for threshold choice; same patients used for both experiment arms.
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=2026)
fit_ix, val_ix = next(gss.split(train, train.icu_transfer, groups=train.subject_id))
fit = train.iloc[fit_ix].copy()
val = train.iloc[val_ix].copy()
if set(fit.subject_id) & set(val.subject_id):
    raise ValueError('Patient overlap in threshold-validation split')

# Already one-hot-encoded admission categories. SMOTENC preserves 0/1 categorical flags.
cat_ix = [i for i,c in enumerate(feature_cols) if c.startswith(('admission_type_', 'admission_location_', 'gender_'))]
num_ix = [i for i in range(len(feature_cols)) if i not in cat_ix]
print(f'Encoded categorical indicators: {len(cat_ix)}; numeric predictors: {len(num_ix)}')

# Fit imputers ONLY on fitting/development patients, never validation/test.
def fit_imputers(frame):
    num = SimpleImputer(strategy='median', keep_empty_features=True)
    cat = SimpleImputer(strategy='most_frequent', keep_empty_features=True)
    if num_ix: num.fit(frame[feature_cols].iloc[:, num_ix])
    if cat_ix: cat.fit(frame[feature_cols].iloc[:, cat_ix])
    return num,cat

def transform(frame, imputers):
    num,cat = imputers
    X = np.zeros((len(frame), len(feature_cols)), dtype=np.float32)
    if num_ix: X[:,num_ix] = num.transform(frame[feature_cols].iloc[:,num_ix]).astype(np.float32)
    if cat_ix: X[:,cat_ix] = cat.transform(frame[feature_cols].iloc[:,cat_ix]).astype(np.float32)
    return X

def oversample(X, y):
    counts = np.bincount(y, minlength=2)
    current_ratio = counts[1]/counts[0]
    if SMOTE_RATIO <= current_ratio:
        raise ValueError(f'SMOTE_RATIO={SMOTE_RATIO} must exceed original ratio {current_ratio:.4f}')
    k = min(K_NEIGHBORS, int(counts[1])-1)
    if k < 1: raise ValueError('Not enough positive samples for SMOTE')
    if cat_ix:
        sampler = SMOTENC(categorical_features=cat_ix, sampling_strategy=SMOTE_RATIO,
                          k_neighbors=k, random_state=SEED)
    else:
        sampler = SMOTE(sampling_strategy=SMOTE_RATIO, k_neighbors=k, random_state=SEED)
    Xr,yr = sampler.fit_resample(X,y)
    return np.asarray(Xr,dtype=np.float32),np.asarray(yr,dtype=int)

def threshold_for_recall(y, p, target):
    # Choose highest threshold achieving target recall (fewer alerts among those meeting target).
    order = np.argsort(-p, kind='stable')
    sorted_y, sorted_p = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p)!=0), len(sorted_p)-1]
    tps = np.cumsum(sorted_y)[ends]
    recalls = tps / max(1,int(y.sum()))
    eligible = np.flatnonzero(recalls>=target)
    if not len(eligible): raise ValueError('Cannot reach target recall')
    return float(sorted_p[ends[eligible[0]]])

def metrics(y,p,t):
    pred=(p>=t).astype(int)
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    return dict(threshold=float(t), auroc=float(roc_auc_score(y,p)),
                auprc=float(average_precision_score(y,p)),
                recall=float(recall_score(y,pred)),
                precision=float(precision_score(y,pred,zero_division=0)),
                specificity=tn/(tn+fp), f1=float(f1_score(y,pred,zero_division=0)),
                accuracy=float(accuracy_score(y,pred)),
                brier=float(brier_score_loss(y,p)),
                log_loss=float(log_loss(y,np.clip(p,1e-7,1-1e-7))),
                tp=tp,tn=tn,fp=fp,fn=fn, alerts=tp+fp,
                alert_rate=(tp+fp)/len(y))

print('\nSTEP 2: Validation threshold selection...', flush=True)
fit_imp = fit_imputers(fit)
Xfit=transform(fit,fit_imp); Xval=transform(val,fit_imp)
yfit=fit.icu_transfer.to_numpy(dtype=int); yval=val.icu_transfer.to_numpy(dtype=int)
thresholds={}
validation_rows=[]
for name,use_smote in [('baseline',False),('smotenc',True)]:
    xx,yy=oversample(Xfit,yfit) if use_smote else (Xfit,yfit)
    print(f'  {name}: training counts {np.bincount(yy).tolist()}',flush=True)
    model=XGBClassifier(**PARAMS)
    model.fit(xx,yy)
    p=model.predict_proba(Xval)[:,1]
    t=threshold_for_recall(yval,p,TARGET_RECALL)
    thresholds[name]=t
    row=dict(model=name,split='validation',**metrics(yval,p,t))
    validation_rows.append(row)
    print(f'  {name}: threshold={t:.6f}, AP={row["auprc"]:.4f}, recall={row["recall"]:.3f}, precision={row["precision"]:.3f}, FP={row["fp"]:,}')

# Refit both models on ALL development admissions. This mirrors Step 53 final refit.
# Note: validation thresholds may drift after full-development refit.
print('\nSTEP 3: Refitting on full development set...',flush=True)
full_imp=fit_imputers(train)
Xtrain=transform(train,full_imp); Xtest=transform(test,full_imp)
ytrain=train.icu_transfer.to_numpy(dtype=int); ytest=test.icu_transfer.to_numpy(dtype=int)
results=[]
probs={}
for name,use_smote in [('baseline',False),('smotenc',True)]:
    xx,yy=oversample(Xtrain,ytrain) if use_smote else (Xtrain,ytrain)
    model=XGBClassifier(**PARAMS)
    model.fit(xx,yy)
    p=model.predict_proba(Xtest)[:,1]
    probs[name]=p
    row=dict(model=name,split='exploratory_test',**metrics(ytest,p,thresholds[name]))
    results.append(row)
    model.save_model(str(OUT/f'{name}_xgboost.json'))
    print(f'  {name}: AUROC={row["auroc"]:.4f} AP={row["auprc"]:.4f} recall={row["recall"]:.3f} precision={row["precision"]:.3f} FP={row["fp"]:,}',flush=True)

pd.DataFrame(validation_rows).to_csv(OUT/'validation_metrics.csv',index=False)
pd.DataFrame(results).to_csv(OUT/'test_metrics_comparison.csv',index=False)
predictions=pd.DataFrame({'hadm_id':test.hadm_id,'subject_id':test.subject_id,'actual':ytest})
for name,p in probs.items():
    predictions[f'{name}_probability']=p
    predictions[f'{name}_prediction']=(p>=thresholds[name]).astype(int)
predictions.to_csv(OUT/'test_predictions_comparison.csv',index=False)

# === FIGURES ===
print('\nSTEP 4: Generating graphs...',flush=True)
for row in results:
    name=row['model']; p=probs[name]; pred=(p>=thresholds[name]).astype(int)
    cm=confusion_matrix(ytest,pred,labels=[0,1])
    fig,ax=plt.subplots(figsize=(7,5.5))
    im=ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f'{cm[i,j]:,}\n{["TN","FP","FN","TP"][2*i+j]}',ha='center',va='center',fontsize=13,
                    color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set_xticks([0,1],['No ICU','ICU']); ax.set_yticks([0,1],['No ICU','ICU'])
    ax.set(xlabel='Predicted',ylabel='Actual',title=f'{name} confusion matrix | threshold {thresholds[name]:.5f}')
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(OUT/f'{name}_confusion_matrix.png',dpi=250);plt.close(fig)

fig,ax=plt.subplots(figsize=(7,5.5))
for name,p in probs.items():
    fpr,tpr,_=roc_curve(ytest,p)
    ax.plot(fpr,tpr,label=f'{name} (AUROC={roc_auc_score(ytest,p):.4f})')
ax.plot([0,1],[0,1],'--',label='Chance')
ax.set(xlabel='False positive rate',ylabel='True positive rate',title='ROC: baseline vs SMOTENC')
ax.legend();fig.tight_layout();fig.savefig(OUT/'roc_comparison.png',dpi=250);plt.close(fig)

fig,ax=plt.subplots(figsize=(7,5.5))
for name,p in probs.items():
    precision,recall,_=precision_recall_curve(ytest,p)
    ax.plot(recall,precision,label=f'{name} (AP={average_precision_score(ytest,p):.4f})')
ax.axhline(ytest.mean(),ls='--',label=f'Prevalence={ytest.mean():.4f}')
ax.set(xlabel='Recall',ylabel='Precision',title='Precision-recall: baseline vs SMOTENC')
ax.legend();fig.tight_layout();fig.savefig(OUT/'pr_comparison.png',dpi=250);plt.close(fig)

fig,ax=plt.subplots(figsize=(7,5.5))
ax.plot([0,0.3],[0,0.3],'--',label='Perfect calibration')
for name,p in probs.items():
    # Equal-sized risk bins, not evenly spaced risk ranges
    bins=pd.qcut(pd.Series(p),q=10,labels=False,duplicates='drop')
    temp=pd.DataFrame({'bin':bins,'p':p,'y':ytest})
    g=temp.groupby('bin').agg(mean_pred=('p','mean'),observed=('y','mean'))
    ax.plot(g.mean_pred,g.observed,marker='o',label=name)
ax.set(xlim=(0,0.3),ylim=(0,0.3),xlabel='Mean predicted probability',ylabel='Observed transfer rate',title='Exploratory test calibration')
ax.legend();fig.tight_layout();fig.savefig(OUT/'calibration_comparison.png',dpi=250);plt.close(fig)

fig,ax=plt.subplots(figsize=(8,5))
for name,p in probs.items():
    ax.hist(np.clip(p,0,0.30),bins=60,density=True,alpha=0.45,label=name)
ax.set(xlabel='Predicted probability (clipped at 0.30)',ylabel='Density',title='Predicted probability distributions')
ax.legend();fig.tight_layout();fig.savefig(OUT/'probability_distributions.png',dpi=250);plt.close(fig)

comparison=pd.DataFrame(results).set_index('model')
fig,ax=plt.subplots(figsize=(8,5))
comparison[['recall','precision','specificity']].T.plot(kind='bar',ax=ax)
ax.set(ylabel='Fraction',title='Operating-point metrics at validation-selected thresholds')
ax.tick_params(axis='x',rotation=0);fig.tight_layout();fig.savefig(OUT/'operating_metrics_comparison.png',dpi=250);plt.close(fig)

# Same threshold comparison is NOT appropriate after oversampling because probability scales shift.
# Save feature list and all experiment parameters for reproducibility.
pd.DataFrame({'feature':feature_cols}).to_csv(OUT/'feature_list.csv',index=False)
with open(OUT/'experiment_metadata.json','w',encoding='utf-8') as f:
    json.dump({'source':str(DATA_FILE),'seed':SEED,'feature_count':75,
               'params':PARAMS,'smote_ratio':SMOTE_RATIO,'k_neighbors':K_NEIGHBORS,
               'categorical_columns':[feature_cols[i] for i in cat_ix],
               'validation_thresholds':thresholds,
               'notes':['SMOTENC used only on training patients, never validation/test.',
                        'Both arms use identical original 75 selected features and train/test membership.',
                        'Original 75 features are already engineered and selected; raw-data processing is not repeated.',
                        'The test split has been viewed before: exploratory comparison only.',
                        'One-hot admission flags may form impossible combinations under independent SMOTENC treatment; interpret results cautiously.',
                        'Thresholds selected on inner validation then used after full-development refit; recall can drift.',
                        'Previous outcome-label inconsistencies and early-discharge policy remain a study limitation.']},f,indent=2)

print('\nTEST METRICS COMPARISON')
print(pd.DataFrame(results).to_string(index=False))
print('\nDONE. Saved everything to:',OUT)
print('Existing Step 53 model and Step 54 figures were not modified.')
