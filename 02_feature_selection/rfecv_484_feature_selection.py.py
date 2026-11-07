"""Step 70 FAST: grouped XGBoost RFECV without VIF, using Step 51 corrected candidate features.
Exploratory evaluation; original models remain untouched.
"""
from pathlib import Path
import json, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_curve, brier_score_loss,
                             precision_score, recall_score, f1_score, accuracy_score)
from xgboost import XGBClassifier
import joblib

ROOT = Path(r'C:\Users\sowmy\Downloads\MIMIC-IV\MIMIC_IV\MIMIC_IV')
BASE = ROOT / 'model training figures' / 'expanded_lab_xgboost'
FEATURE_FILE = BASE / 'step51_timing_corrected' / 'expanded_candidate_features_storetime_corrected.parquet'
OUT = BASE / 'step70_fast_rfecv_no_vif'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
MIN_FEATURES = 10
RFE_STEP = 10
TARGET_RECALL = 0.70

# Discover the cohort table by required schema; avoid silently choosing an unrelated file.
def locate_cohort():
    paths = [BASE / 'cohort_audit.csv', ROOT / 'cohort_audit.csv']
    for p in paths:
        if p.exists():
            return p
    matches = []
    for p in BASE.rglob('*.csv'):
        if any(x in str(p).lower() for x in ['step67', 'step68', 'smote', 'predictions']):
            continue
        try:
            cols = set(pd.read_csv(p, nrows=0).columns)
            if {'hadm_id','subject_id','icu_transfer'}.issubset(cols):
                matches.append(p)
        except Exception:
            pass
    if len(matches) != 1:
        raise FileNotFoundError('Could not uniquely identify cohort CSV. Set COHORT_FILE manually. Candidates: ' + str(matches[:20]))
    return matches[0]

COHORT_FILE = locate_cohort()
print('Features:', FEATURE_FILE, '\nCohort:', COHORT_FILE, flush=True)
if not FEATURE_FILE.exists():
    raise FileNotFoundError('Step 51 corrected feature file not found: ' + str(FEATURE_FILE))
cohort = pd.read_csv(COHORT_FILE, usecols=['hadm_id','subject_id','icu_transfer'])
feat = pd.read_parquet(FEATURE_FILE)
if not {'hadm_id'}.issubset(feat.columns):
    raise ValueError('Corrected features must contain hadm_id')
if cohort.hadm_id.duplicated().any() or feat.hadm_id.duplicated().any():
    raise ValueError('Duplicate admission IDs detected')
if not cohort.icu_transfer.isin([0,1]).all():
    raise ValueError('Non-binary outcome')
if cohort.subject_id.isna().any():
    raise ValueError('Missing subject_id')
if set(cohort.hadm_id) != set(feat.hadm_id):
    raise ValueError('Cohort and features have different admission memberships')
joined = cohort.merge(feat.drop(columns=['subject_id','icu_transfer'], errors='ignore'), on='hadm_id', validate='one_to_one', sort=False)

# Same patient-level development/test split as earlier experiments.
splitter = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED)
dev_i, test_i = next(splitter.split(joined, joined.icu_transfer, groups=joined.subject_id))
dev, test = joined.iloc[dev_i].copy(), joined.iloc[test_i].copy()
if set(dev.subject_id) & set(test.subject_id):
    raise ValueError('Patient overlap')
print(f'Development={len(dev):,}, test={len(test):,}, predictors before encoding={len(joined.columns)-3}', flush=True)
if (len(dev),len(test)) != (66118,22223):
    raise ValueError('Split sizes differ from previous project. Verify cohort ordering and split before continuing.')

# Hold out a patient-disjoint validation subset for threshold selection.
inner = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=2026)
fit_i, val_i = next(inner.split(dev, dev.icu_transfer, groups=dev.subject_id))
fit, val = dev.iloc[fit_i].copy(), dev.iloc[val_i].copy()

# Only include actual predictors, not IDs, target or any temporal/outcome fields.
excluded = {'subject_id','hadm_id','icu_transfer','hours_to_icu','icu_intime','admittime','dischtime','deathtime','stay_id'}
raw_cols = [c for c in joined.columns if c not in excluded]
suspicious = [c for c in raw_cols if re.search(r'(icu|transfer|outcome|label|discharg|death|future|after_24|after_72|time_to)', c, re.I)]
if suspicious:
    raise ValueError('Potential target/leakage predictors. Inspect before proceeding: ' + str(suspicious))
cat_cols = [c for c in raw_cols if not pd.api.types.is_numeric_dtype(joined[c])]
num_cols = [c for c in raw_cols if c not in cat_cols]

# Refit imputation/encoding on the fitting subset. Do not use test for feature selection.
def prep_fit(frame):
    transformers = []
    if num_cols:
        transformers.append(('num', SimpleImputer(strategy='median', keep_empty_features=True), num_cols))
    if cat_cols:
        transformers.append(('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols))
    pre = ColumnTransformer(transformers, sparse_threshold=0, verbose_feature_names_out=False)
    arr = np.asarray(pre.fit_transform(frame[raw_cols]), dtype=np.float32)
    names = list(pre.get_feature_names_out())
    return pre, arr, names

pre, Xfit, names = prep_fit(fit)
Xval = np.asarray(pre.transform(val[raw_cols]), dtype=np.float32)
yfit, yval = fit.icu_transfer.to_numpy(), val.icu_transfer.to_numpy()

# Remove constant and exactly duplicate columns using fitting data only.
# No VIF: tree-based XGBoost tolerates correlated predictors.
std = np.nanstd(Xfit, axis=0)
keep = np.flatnonzero(np.isfinite(std) & (std > 1e-10))
Xfit = Xfit[:, keep]
Xval = Xval[:, keep]
names = [names[i] for i in keep]
# Drop exact duplicates (including encoded indicator duplicates).
unique = ~pd.DataFrame(Xfit.T).duplicated().to_numpy()
Xfit = Xfit[:, unique]
Xval = Xval[:, unique]
names = [name for name, ok in zip(names, unique) if ok]
print('Features after constant/duplicate removal:', len(names), flush=True)
if len(names) < MIN_FEATURES:
    raise ValueError('Fewer than minimum number of predictors remain')

params = dict(n_estimators=100, max_depth=4, learning_rate=.05, subsample=.8,
              colsample_bytree=.8, min_child_weight=5, reg_lambda=5,
              scale_pos_weight=1, objective='binary:logistic', eval_metric='aucpr',
              tree_method='hist', random_state=SEED, n_jobs=4)
cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
selector = RFECV(estimator=XGBClassifier(**params), step=RFE_STEP,
                 min_features_to_select=min(MIN_FEATURES,len(names)), cv=cv,
                 scoring='average_precision', n_jobs=1)
print('Starting grouped RFECV on',len(names),'features (step=10, 100 trees).', flush=True)
selector.fit(Xfit, yfit, groups=fit.subject_id.to_numpy())
chosen = list(np.array(names)[selector.support_])
print('RFECV selected',len(chosen),'features', flush=True)
pd.DataFrame({'feature':names,'selected':selector.support_,'rfe_rank':selector.ranking_}).to_csv(OUT/'feature_ranking.csv',index=False)
pd.Series(chosen,name='selected_feature').to_csv(OUT/'selected_features.csv',index=False)

# Validation threshold selection; test is untouched until all choices are frozen.
model = XGBClassifier(**params)
model.fit(Xfit[:,selector.support_], yfit)
pval = model.predict_proba(Xval[:,selector.support_])[:,1]
pr, rec, th = precision_recall_curve(yval,pval)
valid = np.flatnonzero(rec[:-1]>=TARGET_RECALL)
if not len(valid):
    raise RuntimeError('No validation threshold achieves target recall')
best = max(valid, key=lambda i:(pr[i],th[i]))
threshold = float(th[best])
print('Validation AP:',average_precision_score(yval,pval),'threshold:',threshold,flush=True)

# Refit preprocessing and selected-feature model on all development admissions.
# Map selected names to newly fitted encoding names (stable for this cohort).
pre_final, Xdev_all, final_names = prep_fit(dev)
Xt_all = np.asarray(pre_final.transform(test[raw_cols]),dtype=np.float32)
if not set(chosen).issubset(final_names):
    raise ValueError('Some selected feature names absent after full development refit')
indices = [final_names.index(c) for c in chosen]
final_model = XGBClassifier(**params)
final_model.fit(Xdev_all[:,indices],dev.icu_transfer.to_numpy())
p = final_model.predict_proba(Xt_all[:,indices])[:,1]
y = test.icu_transfer.to_numpy()
pred = (p>=threshold).astype(int)
tn,fp,fn,tp = map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
metrics = {'features':len(chosen),'threshold':threshold,'auroc':roc_auc_score(y,p),
           'auprc':average_precision_score(y,p),'recall':recall_score(y,pred),
           'precision':precision_score(y,pred,zero_division=0),
           'specificity':tn/(tn+fp),'f1':f1_score(y,pred),
           'accuracy':accuracy_score(y,pred),'brier':brier_score_loss(y,p),
           'tp':tp,'fp':fp,'fn':fn,'tn':tn,'test_admissions':len(test)}
print('\nEXPLORATORY TEST RESULTS\n',json.dumps(metrics,indent=2),flush=True)
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
pd.DataFrame({'hadm_id':test.hadm_id,'subject_id':test.subject_id,'actual':y,'probability':p,'predicted':pred}).to_csv(OUT/'test_predictions.csv',index=False)
joblib.dump({'preprocessor':pre_final,'selected_names':chosen,'model':final_model,'threshold':threshold,'raw_columns':raw_cols},OUT/'inference_bundle.joblib')

fig,ax=plt.subplots(figsize=(6,5))
cm=np.array([[tn,fp],[fn,tp]])
ax.imshow(cm,cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center')
ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['No ICU','ICU'],yticklabels=['No ICU','ICU'],xlabel='Predicted',ylabel='Actual',title='XGBoost RFECV confusion matrix')
fig.tight_layout();fig.savefig(OUT/'confusion_matrix.png',dpi=250);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,5))
fpr,tpr,_=roc_curve(y,p)
ax.plot(fpr,tpr,label=f'AUROC={metrics["auroc"]:.3f}')
ax.plot([0,1],[0,1],'--');ax.legend();ax.set(xlabel='FPR',ylabel='TPR',title='ROC curve')
fig.tight_layout();fig.savefig(OUT/'roc.png',dpi=250);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,5))
prec,rec,_=precision_recall_curve(y,p)
ax.plot(rec,prec,label=f'AP={metrics["auprc"]:.3f}')
ax.axhline(y.mean(),ls='--',label='Prevalence');ax.legend();ax.set(xlabel='Recall',ylabel='Precision',title='PR curve')
fig.tight_layout();fig.savefig(OUT/'pr.png',dpi=250);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
cvres=pd.DataFrame(selector.cv_results_)
cvres.to_csv(OUT/'rfecv_cv_results.csv',index=False)
ax.plot(cvres['n_features'],cvres['mean_test_score'],marker='o')
ax.axvline(len(chosen),ls='--');ax.set(xlabel='Feature count',ylabel='CV average precision',title='RFECV feature count selection')
fig.tight_layout();fig.savefig(OUT/'rfecv_feature_curve.png',dpi=250);plt.close(fig)
print('\nSaved results to:',OUT)
print('CAVEATS: Exploratory test previously used; preprocessing outside inner RFECV folds; Step 57 outcome inconsistencies remain.')
