#!/usr/bin/env python3
"""Phase 22: paired patient-cluster bootstrap and calibration for Phase21 saved validation predictions.
Does not touch the held-out evaluation split. Requires only phase21_outputs/validation_predictions.parquet.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss

def metric(y,p):
    n=len(y); k=max(1,int(np.ceil(.1*n))); top=np.argpartition(-p,k-1)[:k]
    return dict(auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),brier=float(brier_score_loss(y,p)),recall_top10=float(y[top].sum()/y.sum()),precision_top10=float(y[top].mean()),expected=float(p.sum()),observed=int(y.sum()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase21-output',required=True);ap.add_argument('--output',required=True);ap.add_argument('--bootstraps',type=int,default=1000);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args()
    d=pd.read_parquet(Path(a.phase21_output)/'validation_predictions.parquet');out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    models=[c for c in d if c.endswith('_icu_by_72h')];assert len(models)>=2,models
    assert {'subject_id','hadm_id','outcome'}.issubset(d.columns)
    assert not d.hadm_id.duplicated().any()
    y=d.outcome.to_numpy().astype(int);assert set(np.unique(y))<= {0,1}
    for c in models:assert np.isfinite(d[c]).all() and d[c].between(0,1).all()
    report={'scope':'internal validation only; no held-out evaluation accessed','n':len(d),'events':int(y.sum()),'models':{},'paired_bootstrap':{}}
    bins=[]
    for c in models:
        p=d[c].to_numpy();report['models'][c]=metric(y,p)
        # Equal-count risk deciles; ties may cross decile boundaries.
        order=np.argsort(p,kind='stable');rank=np.empty(len(p),dtype=int);rank[order]=np.arange(len(p));dec=np.minimum(9,rank*10//len(p))
        for j in range(10):
            z=dec==j;bins.append({'model':c,'decile':j+1,'n':int(z.sum()),'observed_rate':float(y[z].mean()),'predicted_rate':float(p[z].mean()),'events':int(y[z].sum())})
    pd.DataFrame(bins).to_csv(out/'calibration_deciles.csv',index=False)
    groups=d.groupby('subject_id',sort=False).indices;keys=list(groups);rng=np.random.default_rng(a.seed)
    boot={c:[] for c in models};diffs=[];fail=0
    for b in range(a.bootstraps):
        sampled=rng.integers(0,len(keys),size=len(keys));idx=np.concatenate([groups[keys[i]] for i in sampled]);yy=y[idx]
        if yy.min()==yy.max():fail+=1;continue
        results={c:metric(yy,d[c].to_numpy()[idx]) for c in models}
        for c in models:boot[c].append(results[c])
        diffs.append({k:results[models[1]][k]-results[models[0]][k] for k in ['auroc','auprc','brier','recall_top10','precision_top10']})
    def summary(rows):
        return {k:{'lower_95':float(np.quantile([r[k] for r in rows],.025)),'upper_95':float(np.quantile([r[k] for r in rows],.975))} for k in rows[0]}
    for c in models:report['models'][c]['bootstrap_95ci']=summary(boot[c])
    report['paired_bootstrap']={'comparison':f'{models[1]} minus {models[0]}','intervals':summary(diffs),'valid_resamples':len(diffs),'skipped':fail}
    (out/'phase22_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    print('Saved to',out)
if __name__=='__main__':main()
