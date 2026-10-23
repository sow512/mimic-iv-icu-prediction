#!/usr/bin/env python3
"""Phase 22 validation of SAVED admission-level predictions; does not audit leakage.
No patient IDs are saved, so intervals are admission-bootstrap CIs, not patient-cluster CIs.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def metrics(y,p):
    k=max(1,int(np.ceil(.1*len(y))))
    top=np.argpartition(-p,k-1)[:k]
    return {'auroc':float(roc_auc_score(y,p)), 'auprc':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p)), 'recall_top10':float(y[top].sum()/y.sum()),
            'precision_top10':float(y[top].mean()),'expected_events':float(p.sum()),'observed_events':int(y.sum())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase21-output',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--bootstraps',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--horizon',type=int,default=None,help='Saved prediction suffix in hours, e.g. 48; default largest common horizon')
    a=ap.parse_args()
    if a.bootstraps<1: ap.error('--bootstraps must be >= 1')
    src=Path(a.phase21_output)
    d=pd.read_parquet(src/'validation_predictions.parquet')
    if not {'hadm_id','outcome'}.issubset(d.columns): raise ValueError(f'Missing hadm_id/outcome: {d.columns.tolist()}')
    if d.hadm_id.duplicated().any(): raise ValueError('Duplicate hadm_id; cannot safely evaluate')
    cols={}
    for c in d.columns:
        m=re.fullmatch(r'(linear|neural)_icu_by_(\d+)h',c)
        if m: cols[(m.group(1),int(m.group(2)))]=c
    common=sorted({h for model,h in cols if model=='linear'} & {h for model,h in cols if model=='neural'})
    if not common: raise ValueError(f'No common linear/neural prediction horizons. Columns: {d.columns.tolist()}')
    h=a.horizon if a.horizon is not None else common[-1]
    if h not in common: raise ValueError(f'Horizon {h} unavailable; common horizons: {common}')
    selected=[cols[('linear',h)],cols[('neural',h)]]
    y=d.outcome.to_numpy()
    if not np.isin(y,[0,1]).all(): raise ValueError('Outcome must be binary 0/1')
    y=y.astype(int)
    if y.sum()==0 or y.sum()==len(y): raise ValueError('Both outcome classes required')
    probs={}
    for c in selected:
        p=d[c].to_numpy(dtype=float)
        if not np.isfinite(p).all() or np.any((p<0)|(p>1)): raise ValueError(f'Invalid probabilities: {c}')
        probs[c]=p
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    warning=('No subject_id column in saved predictions: confidence intervals resample admissions, '
             'not independent patients. Correlation from repeated admissions may make intervals too narrow. '
             'Saved horizon suffix may be hours after landmark, not hours after admission; '
             'verify label and time origin in training code. No feature-leakage audit performed.')
    report={'n':len(y),'events':int(y.sum()),'selected_horizon_suffix_hours':h,
            'available_common_horizons':common,'models':{},'paired_difference_neural_minus_linear':{},
            'bootstrap_unit':'admission','warning':warning}
    bins=[]
    for c,p in probs.items():
        report['models'][c]=metrics(y,p)
        rank=np.empty(len(p),dtype=int);rank[np.argsort(p,kind='stable')]=np.arange(len(p))
        dec=np.minimum(9,rank*10//len(p))
        for j in range(10):
            ix=dec==j
            bins.append({'model':c,'risk_decile':j+1,'n':int(ix.sum()),'events':int(y[ix].sum()),
                         'observed_rate':float(y[ix].mean()),'predicted_rate':float(p[ix].mean())})
    pd.DataFrame(bins).to_csv(out/'calibration_deciles.csv',index=False)
    rng=np.random.default_rng(a.seed)
    samples={c:[] for c in selected}; diffs=[]; skipped=0
    compare_keys=['auroc','auprc','brier','recall_top10','precision_top10']
    for _ in range(a.bootstraps):
        ix=rng.integers(len(y),size=len(y)); yy=y[ix]
        if yy.min()==yy.max(): skipped+=1;continue
        res={c:metrics(yy,probs[c][ix]) for c in selected}
        for c in selected:samples[c].append(res[c])
        diffs.append({k:res[selected[1]][k]-res[selected[0]][k] for k in compare_keys})
    if not diffs: raise ValueError('No valid bootstrap samples')
    def ci(rows,keys):
        return {k:{'lower_95':float(np.quantile([v[k] for v in rows],.025)),
                   'upper_95':float(np.quantile([v[k] for v in rows],.975))} for k in keys}
    for c in selected: report['models'][c]['bootstrap_95ci']=ci(samples[c],compare_keys)
    report['paired_difference_neural_minus_linear']={'point':{k:report['models'][selected[1]][k]-report['models'][selected[0]][k] for k in compare_keys},
        'bootstrap_95ci':ci(diffs,compare_keys),'valid_resamples':len(diffs),'skipped':skipped}
    (out/'phase22_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));print('\nWARNING:',warning,'\nSaved to:',out)
if __name__=='__main__':main()
