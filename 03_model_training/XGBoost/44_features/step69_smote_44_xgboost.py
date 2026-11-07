"""Step 69: Baseline vs SMOTENC on teammate's 24-72h time-point dataset.
Research-only row-level evaluation. Does not overwrite prior models.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve, brier_score_loss, log_loss)
from imblearn.over_sampling import SMOTENC, SMOTE
from xgboost import XGBClassifier

DATA_FILE = Path(r'C:\Users\sowmy\OneDrive\Desktop\44f\New Chosen Feature Dataset.csv')
OUT = DATA_FILE.parent / 'smote_44_45_xgboost_results'
OUT.mkdir(parents=True, exist_ok=True)
TARGET = 'icu_transfer_24_to_72h'
META = ['dataset_split', 'row_id', 'subject_id', 'hadm_id', 'index_time', TARGET]
RECALL_TARGET = 0.70
SMOTE_RATIO = 0.30
SEED = 42
PARAMS = dict(n_estimators=350, max_depth=4, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
    reg_lambda=5, scale_pos_weight=1.0, tree_method='hist',
    objective='binary:logistic', eval_metric='logloss', random_state=SEED, n_jobs=4)

print('STEP 69: Loading', DATA_FILE, flush=True)
if not DATA_FILE.is_file():
    raise FileNotFoundError(DATA_FILE)
# The original dataset is gzip-compressed despite its .csv extension.
df = pd.read_csv(DATA_FILE, compression='gzip', low_memory=False)
missing = (set(META) | {'hours_since_admission'}) - set(df.columns)
if missing:
    raise ValueError(f'Missing columns: {sorted(missing)}')
if df[TARGET].isna().any() or not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError('Target must be binary without missing values')
if df[['subject_id', 'hadm_id', 'dataset_split']].isna().any().any():
    raise ValueError('Missing identifiers or split labels')
df['dataset_split'] = df['dataset_split'].astype(str).str.strip().str.lower()
if set(df.dataset_split.unique()) != {'train', 'validation', 'test'}:
    raise ValueError(f'Unexpected split labels: {df.dataset_split.unique()}')
feature_cols = [c for c in df.columns if c not in META]
if any(not pd.api.types.is_numeric_dtype(df[c]) for c in feature_cols):
    raise ValueError('Non-numeric predictors detected')
parts = {s: df.loc[df.dataset_split.eq(s)].copy() for s in ('train', 'validation', 'test')}
for a, b in [('train','validation'),('train','test'),('validation','test')]:
    if set(parts[a].subject_id) & set(parts[b].subject_id):
        raise ValueError(f'Patient overlap: {a} vs {b}')
for s, part in parts.items():
    print(f'{s}: {len(part):,} rows, {int(part[TARGET].sum()):,} positives, {part.subject_id.nunique():,} patients')
print('Predictor count:', len(feature_cols))

# The one-hot columns are already encoded. SMOTENC on individual dummies
# can produce invalid combinations; validate and document this limitation.
cat_cols = [c for c in feature_cols if c.startswith(('gender_', 'admission_type_', 'admission_location_', 'insurance_')) or c.endswith('_missing')]
num_cols = [c for c in feature_cols if c not in cat_cols]
for col in cat_cols:
    values = pd.to_numeric(df[col], errors='coerce').dropna().unique()
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f'Categorical indicator is not binary: {col}, examples={values[:8]}')
print(f'Numerical predictors: {len(num_cols)}; binary indicator predictors: {len(cat_cols)}')

def preprocess(fit, other):
    # Fit preprocessing on training only, separately for validation/full development.
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    train_num = imputer.fit_transform(fit[num_cols].replace([np.inf,-np.inf], np.nan))
    other_num = imputer.transform(other[num_cols].replace([np.inf,-np.inf], np.nan))
    scaler = StandardScaler()
    train_num = scaler.fit_transform(train_num)
    other_num = scaler.transform(other_num)
    if cat_cols:
        train_cat = fit[cat_cols].fillna(0).to_numpy(dtype=np.float32)
        other_cat = other[cat_cols].fillna(0).to_numpy(dtype=np.float32)
        if not np.isin(train_cat, [0,1]).all() or not np.isin(other_cat, [0,1]).all():
            raise ValueError('Nonbinary categorical indicators after missing-value handling')
        train_x = np.column_stack([train_num, train_cat]).astype(np.float32)
        other_x = np.column_stack([other_num, other_cat]).astype(np.float32)
        cat_idx = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    else:
        train_x, other_x = train_num.astype(np.float32), other_num.astype(np.float32)
        cat_idx = []
    return train_x, other_x, cat_idx, imputer, scaler

def oversample(X, y, cat_idx):
    ratio = np.bincount(y)[1] / np.bincount(y)[0]
    if ratio >= SMOTE_RATIO:
        raise ValueError('Minority ratio already exceeds SMOTE target')
    if cat_idx and len(cat_idx) < X.shape[1]:
        sampler = SMOTENC(categorical_features=cat_idx, sampling_strategy=SMOTE_RATIO,
                          random_state=SEED, k_neighbors=5)
    else:
        sampler = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=SEED, k_neighbors=5)
    return sampler.fit_resample(X, y)

def threshold_for_recall(y, p):
    order = np.argsort(-p, kind='stable')
    ys, ps = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps)-1]
    tp = np.cumsum(ys)[ends]
    precision = tp / (ends + 1)
    recall = tp / int(y.sum())
    eligible = np.flatnonzero(recall >= RECALL_TARGET)
    if not len(eligible):
        raise ValueError('Cannot reach target recall')
    candidates = eligible[np.isclose(precision[eligible], precision[eligible].max())]
    return float(ps[ends[candidates[np.argmax(ps[ends[candidates]])]]])

def metrics(name, y, p, threshold, split):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0,1]).ravel())
    return dict(model=name, split=split, threshold=threshold,
        auroc=float(roc_auc_score(y,p)), auprc=float(average_precision_score(y,p)),
        recall=tp/(tp+fn), precision=tp/(tp+fp) if tp+fp else 0.,
        specificity=tn/(tn+fp), f1=2*tp/(2*tp+fp+fn),
        accuracy=(tp+tn)/len(y), brier=float(brier_score_loss(y,p)),
        log_loss=float(log_loss(y,p)), tn=tn, fp=fp, fn=fn, tp=tp,
        alerts=tp+fp, alert_rate=(tp+fp)/len(y))

def confusion_plot(row):
    cm = np.array([[row['tn'],row['fp']],[row['fn'],row['tp']]])
    fig, ax = plt.subplots(figsize=(6.8,5.5))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f'{cm[i,j]:,}\n{["TN","FP","FN","TP"][i*2+j]}',
                ha='center',va='center',fontweight='bold',
                color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set_xticks([0,1],['No ICU','ICU'])
    ax.set_yticks([0,1],['No ICU','ICU'])
    ax.set(xlabel='Predicted', ylabel='Actual',
           title=f'{row["model"].upper()} - 24-72h row-level\nThreshold={row["threshold"]:.5f}')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT/f'{row["model"]}_confusion_matrix.png',dpi=250)
    plt.close(fig)

train, val, test = (parts[s] for s in ('train','validation','test'))
ytr = train[TARGET].to_numpy(dtype=int)
yv = val[TARGET].to_numpy(dtype=int)
yt = test[TARGET].to_numpy(dtype=int)
if any(len(np.unique(y)) != 2 for y in (ytr,yv,yt)):
    raise ValueError('Both classes required in every split')

print('\nSTEP 1: Validation threshold selection...',flush=True)
Xtr, Xv, cat_idx, _, _ = preprocess(train,val)
thresholds = {}
validation_rows = []
for name in ('baseline','smotenc'):
    if name == 'smotenc':
        Xfit,yfit = oversample(Xtr,ytr,cat_idx)
    else:
        Xfit,yfit = Xtr,ytr
    print(name, 'training counts:',np.bincount(yfit),flush=True)
    model = XGBClassifier(**PARAMS)
    model.fit(Xfit,yfit)
    p = model.predict_proba(Xv)[:,1]
    threshold = threshold_for_recall(yv,p)
    thresholds[name] = threshold
    row = metrics(name,yv,p,threshold,'validation')
    validation_rows.append(row)
    print(f'  {name}: threshold={threshold:.6f}, AP={row["auprc"]:.4f}, '
          f'recall={row["recall"]:.3f}, precision={row["precision"]:.3f}, FP={row["fp"]:,}',flush=True)
pd.DataFrame(validation_rows).to_csv(OUT/'validation_metrics.csv',index=False)

print('\nSTEP 2: Training comparison models on supplied training split...',flush=True)
# Do not merge validation into training: this keeps the supplied split semantics
# and baseline closer to the original script. All thresholds remain frozen.
rows=[]
probs={}
predictions = test[['row_id','subject_id','hadm_id','index_time']].copy()
predictions['actual'] = yt
for name in ('baseline','smotenc'):
    Xfit,yfit = oversample(Xtr,ytr,cat_idx) if name=='smotenc' else (Xtr,ytr)
    model=XGBClassifier(**PARAMS)
    model.fit(Xfit,yfit)
    model.save_model(str(OUT/f'{name}_xgboost.json'))
    p=model.predict_proba(preprocess(train,test)[1])[:,1]
    row=metrics(name,yt,p,thresholds[name],'exploratory_test')
    rows.append(row)
    probs[name]=p
    predictions[f'{name}_probability']=p
    predictions[f'{name}_prediction']=(p>=thresholds[name]).astype(int)
    confusion_plot(row)
    imp=pd.DataFrame({'feature':num_cols+cat_cols,'importance':model.feature_importances_}).sort_values('importance',ascending=False)
    imp.to_csv(OUT/f'{name}_feature_importance.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,7))
    top=imp.head(20).iloc[::-1]
    ax.barh(top.feature,top.importance)
    ax.set(xlabel='XGBoost importance',title=f'{name}: Top 20 features')
    fig.tight_layout();fig.savefig(OUT/f'{name}_top20_importance.png',dpi=250);plt.close(fig)
    print(f'{name}: AUROC={row["auroc"]:.4f}, AP={row["auprc"]:.4f}, '
          f'recall={row["recall"]:.3f}, precision={row["precision"]:.3f}, FP={row["fp"]:,}',flush=True)
    print(classification_report(yt,predictions[f'{name}_prediction'],zero_division=0))

comparison=pd.DataFrame(rows)
comparison.to_csv(OUT/'test_metrics_comparison.csv',index=False)
predictions.to_csv(OUT/'test_predictions_comparison.csv',index=False)

fig,ax=plt.subplots(figsize=(7,5))
for name,p in probs.items():
    fpr,tpr,_=roc_curve(yt,p)
    ax.plot(fpr,tpr,label=f'{name} AUROC={roc_auc_score(yt,p):.4f}')
ax.plot([0,1],[0,1],'--',label='Chance')
ax.set(xlabel='False positive rate',ylabel='True positive rate',title='ROC comparison')
ax.legend();fig.tight_layout();fig.savefig(OUT/'roc_comparison.png',dpi=250);plt.close(fig)

fig,ax=plt.subplots(figsize=(7,5))
for name,p in probs.items():
    precision,recall,_=precision_recall_curve(yt,p)
    ax.plot(recall,precision,label=f'{name} AP={average_precision_score(yt,p):.4f}')
ax.axhline(yt.mean(),ls='--',label=f'Prevalence={yt.mean():.4f}')
ax.set(xlabel='Recall',ylabel='Precision',title='Precision-recall comparison')
ax.legend();fig.tight_layout();fig.savefig(OUT/'pr_comparison.png',dpi=250);plt.close(fig)

fig,ax=plt.subplots(figsize=(7,5))
ax.plot([0,.3],[0,.3],'--',label='Perfect calibration')
for name,p in probs.items():
    d=pd.DataFrame({'p':p,'y':yt})
    d['bin']=pd.qcut(d.p,10,labels=False,duplicates='drop')
    g=d.groupby('bin').agg(predicted=('p','mean'),observed=('y','mean'))
    ax.plot(g.predicted,g.observed,marker='o',label=name)
ax.set(xlim=(0,.3),ylim=(0,.3),xlabel='Mean predicted risk',ylabel='Observed transfer frequency',title='Calibration comparison')
ax.legend();fig.tight_layout();fig.savefig(OUT/'calibration_comparison.png',dpi=250);plt.close(fig)

fig,ax=plt.subplots(figsize=(9,5))
comparison.set_index('model')[['auroc','auprc','recall','precision','specificity','f1']].T.plot(kind='bar',ax=ax)
ax.set(ylabel='Score',title='Baseline vs SMOTENC (row level)')
ax.tick_params(axis='x',rotation=30)
fig.tight_layout();fig.savefig(OUT/'metrics_comparison.png',dpi=250);plt.close(fig)

(OUT/'experiment_notes.txt').write_text(
    'Exploratory row-level 24-72h target, repeated observation rows per patient.\n'
    'Original split is patient-disjoint; SMOTENC applied to train only.\n'
    'SMOTENC on one-hot indicators may create invalid categorical combinations.\n'
    'Feature scaling was added for SMOTENC distance calculations; baseline is retrained under the same preprocessing.\n'
    'The supplied labels and timing have not been independently audited.\n'
    'Do not compare counts directly with one-admission-per-patient models.\n'
    'The test set has already been examined; treat results as exploratory.\n',encoding='utf-8')
with open(OUT/'settings.json','w',encoding='utf-8') as f:
    json.dump({'target':TARGET,'feature_count':len(feature_cols),'smote_ratio':SMOTE_RATIO,
               'target_recall':RECALL_TARGET,'parameters':PARAMS,'thresholds':thresholds,
               'test_is_exploratory':True},f,indent=2)
print('\nTEST METRICS COMPARISON\n',comparison.to_string(index=False))
print('\nSTEP 69 COMPLETE. Results saved to:',OUT)
print('Original model and results unchanged.')
