#!/usr/bin/env python3
"""Compare saved hospital-wide and renal-trained Phase21 predictions on identical renal validation admissions.
No retraining, no held-out evaluation access. Requires original Phase18 features for patient clusters.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def metrics(y,p):
    n=len(y); k=max(1,int(np.ceil(.1*n)))
    top=np.argsort(-p,kind='stable')[:k]
    return dict(n=int(n),events=int(y.sum()),prevalence=float(y.mean()),auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),brier=float(brier_score_loss(y,p)),recall_top10=float(y[top].sum()/y.sum()),precision_top10=float(y[top].mean()),expected_events=float(p.sum()))

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--phase18',required=True)
    a.add_argument('--general',required=True,help='Phase21 hospital-wide output directory')
    a.add_argument('--renal',required=True,help='Phase21 renal output directory')
    a.add_argument('--output',required=True)
    a.add_argument('--bootstraps',type=int,default=1000)
    a.add_argument('--seed',type=int,default=42)
    args=a.parse_args()
    if args.bootstraps<1: a.error('bootstraps must be positive')
    base=pd.read_parquet(Path(args.phase18)/'prelandmark_features.parquet',columns=['hadm_id','subject_id','split'])
    if base.hadm_id.duplicated().any():raise ValueError('Duplicate Phase18 admissions')
    if base.groupby('subject_id').split.nunique().max()!=1:raise ValueError('Patient split overlap')
    g=pd.read_parquet(Path(args.general)/'validation_predictions.parquet')
    r=pd.read_parquet(Path(args.renal)/'validation_predictions.parquet')
    for name,d in [('general',g),('renal',r)]:
        if d.hadm_id.duplicated().any():raise ValueError(f'{name} duplicate admissions')
        if not {'hadm_id','outcome','neural_icu_by_48h'}.issubset(d):raise ValueError(f'{name} missing expected columns: {d.columns.tolist()}')
    if not set(r.hadm_id).issubset(set(g.hadm_id)):raise ValueError('Renal predictions not subset of general validation predictions')
    d=r[['hadm_id','outcome','neural_icu_by_48h']].rename(columns={'outcome':'renal_outcome','neural_icu_by_48h':'renal_neural'}).merge(g[['hadm_id','outcome','neural_icu_by_48h']].rename(columns={'outcome':'general_outcome','neural_icu_by_48h':'general_neural'}),on='hadm_id',validate='one_to_one').merge(base,on='hadm_id',validate='one_to_one')
    if len(d)!=len(r):raise ValueError('Missing predictions or Phase18 identifiers')
    if not (d.split=='validation').all():raise ValueError('Non-validation admissions included')
    if not (d.renal_outcome==d.general_outcome).all():raise ValueError('Outcome mismatch')
    y=d.renal_outcome.to_numpy(dtype=int)
    if not set(np.unique(y))=={0,1}:raise ValueError('Both classes required')
    for c in ['general_neural','renal_neural']:
        if not np.isfinite(d[c]).all() or not d[c].between(0,1).all():raise ValueError('Invalid probabilities '+c)
    p1=d.general_neural.to_numpy(float);p2=d.renal_neural.to_numpy(float)
    names=['auroc','auprc','brier','recall_top10','precision_top10']
    result={'scope':'renal validation admissions only; no evaluation split accessed','horizon':'48h after 24h landmark (72h after admission)','general':metrics(y,p1),'renal':metrics(y,p2),'bootstrap_unit':'patient','bootstraps_requested':args.bootstraps}
    result['renal_minus_general']={k:result['renal'][k]-result['general'][k] for k in names}
    groups=list(d.groupby('subject_id',sort=False).indices.values()); rng=np.random.default_rng(args.seed)
    deltas=[]; skipped=0
    for _ in range(args.bootstraps):
        chosen=rng.integers(len(groups),size=len(groups));idx=np.concatenate([groups[i] for i in chosen]);yy=y[idx]
        if yy.min()==yy.max():skipped+=1;continue
        m1=metrics(yy,p1[idx]);m2=metrics(yy,p2[idx]);deltas.append({k:m2[k]-m1[k] for k in names})
    if not deltas:raise ValueError('No valid resamples')
    result['paired_patient_bootstrap_95ci']={k:{'lower':float(np.quantile([z[k] for z in deltas],.025)),'upper':float(np.quantile([z[k] for z in deltas],.975))} for k in names}
    result['valid_resamples']=len(deltas);result['skipped_resamples']=skipped;result['unique_patients']=len(groups)
    result['limitations']=['Retrospective internal validation; full Phase21 feature-availability audit incomplete','Original Phase21 interval-boundary training behavior unchanged','Renal subgroup was selected using historical diagnosis features','Subgroup comparison does not establish benefit from renal specialization on other populations']
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'renal_model_comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));print('Saved:',out/'renal_model_comparison.json')
if __name__=='__main__':main()
