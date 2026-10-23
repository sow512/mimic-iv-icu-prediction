#!/usr/bin/env python3
"""Map Phase24 feature rankings to MIMIC lab labels/units and flag redundancy.
No outcomes, evaluation labels, or model fitting are accessed.
"""
import argparse, json, re, hashlib
from pathlib import Path
import pandas as pd

LAB_RE=re.compile(r'^lab_(\d+)_([0-9a-f]{7})_(n|mean|minval|maxval|firstval|lastval)$')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase24',required=True)
    p.add_argument('--phase19b',required=True)
    p.add_argument('--base',help='Optional MIMIC-IV root for d_labitems.csv labels')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    out=Path(a.output).expanduser();out.mkdir(parents=True,exist_ok=True)
    ranked=pd.read_csv(Path(a.phase24)/'feature_importance_train.csv')
    if not {'feature','gain','rank'}.issubset(ranked):raise ValueError('Ranking file lacks feature/gain/rank')
    meta=Path(a.phase19b)/'lab_selection.json'
    if not meta.exists():raise FileNotFoundError(meta)
    selection=json.loads(meta.read_text()).get('selected',[])
    mapping={}
    for item in selection:
        key=str(item.get('feature_key',''))
        if not key:
            unit=str(item.get('unit',''))
            key='lab_'+str(item['itemid'])+'_'+hashlib.sha1(unit.encode()).hexdigest()[:7]
        mapping[key]=item
    labels={}
    if a.base:
        root=Path(a.base).expanduser()
        if not (root/'hosp/d_labitems.csv').exists():root=root/'physionet.org/files/mimiciv/3.1'
        f=root/'hosp/d_labitems.csv'
        if not f.exists():raise FileNotFoundError(f)
        d=pd.read_csv(f,usecols=['itemid','label','fluid'])
        labels={int(r.itemid):(str(r.label),str(r.fluid)) for r in d.itertuples(index=False)}
    rows=[]
    for r in ranked.itertuples(index=False):
        name=str(r.feature);m=LAB_RE.match(name)
        if m:
            itemid=int(m.group(1));key=f'lab_{m.group(1)}_{m.group(2)}'
            entry=mapping.get(key)
            label,fluid=labels.get(itemid,('UNVERIFIED','UNVERIFIED'))
            rows.append(dict(feature=name,rank=int(r.rank),gain=float(r.gain),source='lab',itemid=itemid,unit=entry.get('unit') if entry else 'UNVERIFIED',label=label,fluid=fluid,statistic=m.group(3),selection_match=entry is not None))
        else:
            rows.append(dict(feature=name,rank=int(r.rank),gain=float(r.gain),source='history_or_baseline',itemid='',unit='',label='',fluid='',statistic='',selection_match=True))
    result=pd.DataFrame(rows).sort_values('rank')
    result.to_csv(out/'annotated_feature_importance.csv',index=False)
    lab=result[result.source=='lab'].copy()
    if not lab.empty:
        group=lab.groupby(['itemid','unit','label'],dropna=False).agg(total_gain=('gain','sum'),best_rank=('rank','min'),n_statistics=('statistic','nunique'),statistics=('statistic',lambda s:', '.join(sorted(set(s))))).reset_index().sort_values('total_gain',ascending=False)
        group.to_csv(out/'lab_group_importance.csv',index=False)
    report={'ranked_features':len(result),'laboratory_features':len(lab),'lab_groups':int(lab[['itemid','unit']].drop_duplicates().shape[0]),'unmatched_selection_keys':int((~lab.selection_match).sum()),'warnings':['Gain is model-dependent and correlated statistics split importance.','Lab unit hashes must be verified against lab_selection.json.','This audit does not establish source-event timestamp validity or patient-level leakage freedom.','Feature selection on development data should not use held-out evaluation labels.']}
    (out/'audit_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print('\nTop 30 annotated features:')
    print(result.head(30).to_string(index=False))
    print('\nSaved to',out)
if __name__=='__main__':main()
