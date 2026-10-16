#!/usr/bin/env python3
"""Phase 12: Retrospective AKI/CKD subgroup evaluation (research only).

Reads saved evaluation predictions; NEVER refits models or selects thresholds on test labels.
Bootstrap resamples unique patients, retaining all their admissions.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

GROUPS = ['AKI only', 'CKD only', 'AKI + CKD']
MODELS = {'lightgbm':'lightgbm_probability', 'catboost':'catboost_probability', 'blend':'blend_probability'}

def scores(frame, col, threshold):
    y = frame['outcome'].to_numpy(dtype=int)
    p = frame[col].to_numpy(dtype=float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    n, positives = len(y), int(y.sum())
    result = {'n':int(n), 'positives':positives, 'prevalence':float(y.mean()) if n else None,
              'auroc':float(roc_auc_score(y,p)) if 0<positives<n else None,
              'auprc':float(average_precision_score(y,p)) if positives else None}
    if threshold is not None and n:
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
        result.update({'threshold':float(threshold), 'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),
                       'sensitivity':float(tp/(tp+fn)) if tp+fn else None,
                       'specificity':float(tn/(tn+fp)) if tn+fp else None,
                       'precision':float(tp/(tp+fp)) if tp+fp else 0.0,
                       'balanced_accuracy':float((tp/(tp+fn)+tn/(tn+fp))/2) if (tp+fn) and (tn+fp) else None})
    return result

def bootstrap(frame, col, n_boot, seed):
    # Resample patients with replacement; preserve multiplicity of sampled clusters.
    rng = np.random.default_rng(seed)
    by_patient = [g for _,g in frame.groupby('subject_id', sort=False)]
    if not by_patient:
        return {}
    vals = {'auroc':[], 'auprc':[]}
    for _ in range(n_boot):
        picks = rng.integers(0,len(by_patient),len(by_patient))
        sample = pd.concat([by_patient[i] for i in picks],ignore_index=True)
        m = scores(sample,col,None)
        for k in vals:
            if m[k] is not None: vals[k].append(m[k])
    return {f'{k}_ci_low':float(np.quantile(v,.025)) if v else None for k,v in vals.items()} | \
           {f'{k}_ci_high':float(np.quantile(v,.975)) if v else None for k,v in vals.items()} | \
           {'bootstrap_valid_auroc':len(vals['auroc']), 'bootstrap_valid_auprc':len(vals['auprc'])}

def load_predictions(path):
    df = pd.read_parquet(path)
    needed = ['subject_id','hadm_id','outcome'] + list(MODELS.values())
    missing = set(needed)-set(df.columns)
    if missing: raise ValueError(f'{path}: missing columns {sorted(missing)}')
    if df.hadm_id.duplicated().any(): raise ValueError(f'{path}: duplicate hadm_id')
    return df[needed].copy()

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--phase1',required=True)
    a.add_argument('--phase9c',required=True)
    a.add_argument('--phase11',default=None,help='Optional folder with no-admin evaluation predictions')
    a.add_argument('--no-admin-predictions',default=None,help='Explicit Parquet file; takes precedence over --phase11')
    a.add_argument('--output',required=True)
    a.add_argument('--bootstrap',type=int,default=500)
    a.add_argument('--seed',type=int,default=42)
    a.add_argument('--threshold',type=float,default=None,help='Only use a threshold fixed independently of evaluation labels. Omit for threshold-free evaluation.')
    args=a.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    audit=pd.read_parquet(Path(args.phase1)/'phase1_cohort_audit.parquet',columns=['subject_id','hadm_id','AKI','CKD','kidney_group'])
    if audit.hadm_id.duplicated().any():raise ValueError('Duplicate audit admissions')
    audit['kidney_group']=np.select([(audit.AKI==1)&(audit.CKD==0),(audit.AKI==0)&(audit.CKD==1),(audit.AKI==1)&(audit.CKD==1)],GROUPS,default='Unknown')
    full=load_predictions(Path(args.phase9c)/'E_all_evaluation_predictions.parquet')
    frames={'full':full}
    noadmin_path=Path(args.no_admin_predictions) if args.no_admin_predictions else None
    if noadmin_path is None and args.phase11:
        folder=Path(args.phase11)
        candidates=sorted(folder.glob('*predictions*.parquet'))
        if len(candidates)==1: noadmin_path=candidates[0]
        elif len(candidates)>1:
            print('Multiple Phase 11 prediction files found; specify --no-admin-predictions explicitly:',[p.name for p in candidates])
    if noadmin_path:
        frames['without_admin']=load_predictions(noadmin_path)
    for name,df in frames.items():
        joined=df.merge(audit,on=['subject_id','hadm_id'],how='left',validate='one_to_one',indicator=True)
        if not (joined._merge=='both').all():raise ValueError(f'{name}: predictions missing from audit')
        if (joined.kidney_group=='Unknown').any():raise ValueError(f'{name}: unknown subgroup')
        frames[name]=joined.drop(columns=['_merge'])
    if 'without_admin' in frames:
        left=frames['full'][['subject_id','hadm_id','outcome']].sort_values('hadm_id').reset_index(drop=True)
        right=frames['without_admin'][['subject_id','hadm_id','outcome']].sort_values('hadm_id').reset_index(drop=True)
        if not left.equals(right):raise ValueError('Full and no-admin evaluation sets do not match exactly')
    records=[]
    for variant,df in frames.items():
        for subgroup in ['Overall']+GROUPS:
            sub=df if subgroup=='Overall' else df[df.kidney_group==subgroup]
            for model,col in MODELS.items():
                m=scores(sub,col,args.threshold)
                ci=bootstrap(sub,col,args.bootstrap,args.seed) if args.bootstrap>0 else {}
                records.append({'variant':variant,'subgroup':subgroup,'model':model,'patients':int(sub.subject_id.nunique()),**m,**ci})
    result=pd.DataFrame(records)
    result.to_csv(out/'subgroup_metrics.csv',index=False)
    summary={'models':list(MODELS),'variants':list(frames),'bootstrap_replicates':args.bootstrap,
             'threshold':args.threshold,'warning':'Retrospective ICD subgroups; diagnosis timing may be after landmark. Threshold-free metrics are primary. Any threshold must be selected independently of evaluation labels. Confidence intervals are descriptive and do not correct for earlier model selection.'}
    (out/'phase12_report.json').write_text(json.dumps(summary,indent=2))
    print(result[['variant','subgroup','model','patients','n','positives','prevalence','auroc','auprc','auroc_ci_low','auroc_ci_high','auprc_ci_low','auprc_ci_high']].round(4).to_string(index=False))
    print('Saved:',out)
if __name__=='__main__':main()
