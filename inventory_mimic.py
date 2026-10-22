#!/usr/bin/env python3
"""Read-only exhaustive local MIMIC file/column inventory. No patient values exported."""
import argparse, csv, json, os, re
from pathlib import Path
from collections import Counter
import pandas as pd

KEYS={'subject_id','hadm_id','stay_id','transfer_id','ed_stay_id','encounter_id'}
TIME_RE=re.compile(r'(time|date|chart|store|start|stop|end)',re.I)

def files_under(root):
    exts=('.csv','.csv.gz','.parquet','.tsv','.tsv.gz')
    return sorted(p for p in root.rglob('*') if p.is_file() and p.name.lower().endswith(exts))

def csv_header(path,sep):
    with ( __import__('gzip').open(path,'rt',encoding='utf-8-sig',errors='replace',newline='') if path.name.endswith('.gz') else open(path,'r',encoding='utf-8-sig',errors='replace',newline='')) as f:
        return next(csv.reader(f,delimiter=sep))

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--base',required=True,help='Root folder containing MIMIC files')
    a.add_argument('--output',required=True)
    a.add_argument('--sample-rows',type=int,default=1000)
    a.add_argument('--cohort-ids',help='Optional CSV/parquet with subject_id and/or hadm_id for SAMPLE coverage only')
    args=a.parse_args()
    root=Path(args.base).expanduser().resolve(); out=Path(args.output).expanduser().resolve()
    if not root.is_dir(): raise SystemExit(f'No such input folder: {root}')
    if out==root or root in out.parents: print('Note: output inside input tree; outputs are not scanned as source files.')
    out.mkdir(parents=True,exist_ok=True)
    cohort=None
    if args.cohort_ids:
        cp=Path(args.cohort_ids)
        cohort=pd.read_parquet(cp) if cp.suffix=='.parquet' else pd.read_csv(cp,usecols=lambda c:c in KEYS)
        if not any(k in cohort for k in ('subject_id','hadm_id')): raise SystemExit('Cohort file must contain subject_id and/or hadm_id')
        cohort=cohort[[k for k in ('subject_id','hadm_id') if k in cohort]].drop_duplicates()
    paths=files_under(root); tables=[]; cols=[]; errors=[]
    print(f'Found {len(paths)} data files',flush=True)
    for i,p in enumerate(paths,1):
        rel=str(p.relative_to(root)); kind='parquet' if p.suffix=='.parquet' else ('tsv' if '.tsv' in p.name else 'csv'); sep='\t' if kind=='tsv' else ','
        try:
            if kind=='parquet':
                import pyarrow.parquet as pq
                pf=pq.ParquetFile(p); header=pf.schema_arrow.names; total_rows=pf.metadata.num_rows
                sample=next(pf.iter_batches(batch_size=max(1,args.sample_rows)),None)
                df=sample.to_pandas() if sample is not None else pd.DataFrame(columns=header)
            else:
                header=csv_header(p,sep)
                df=pd.read_csv(p,sep=sep,nrows=max(1,args.sample_rows),low_memory=False,on_bad_lines='skip')
                total_rows=None
            if len(header)!=len(set(header)): errors.append({'file':rel,'issue':'duplicate header names'})
            keys=[k for k in header if k in KEYS]
            tables.append({'file':rel,'module':p.relative_to(root).parts[0],'table':p.name.split('.')[0], 'bytes':p.stat().st_size,'format':kind,'columns':len(header),'sample_rows':len(df),'parquet_rows':total_rows,'join_keys':','.join(keys),'time_columns':','.join(c for c in header if TIME_RE.search(c))})
            sample_cohort=df
            if cohort is not None:
                common=[k for k in ('hadm_id','subject_id') if k in df and k in cohort]
                if common:
                    key='hadm_id' if 'hadm_id' in common else 'subject_id'
                    sample_cohort=df[df[key].isin(set(cohort[key].dropna()))]
                else: sample_cohort=df.iloc[0:0]
            for pos,c in enumerate(header):
                if c in df:
                    s=df[c]; nonnull=int(s.notna().sum()); distinct=int(s.nunique(dropna=True))
                    cohort_nonnull=int(sample_cohort[c].notna().sum()) if c in sample_cohort else None
                    dtype=str(s.dtype)
                else: nonnull=distinct=0; cohort_nonnull=None; dtype='unknown'
                cols.append({'file':rel,'table':p.name.split('.')[0],'column_position':pos,'column':c,'sample_dtype':dtype,'sample_nonnull':nonnull,'sample_rows':len(df),'sample_nonnull_percent':round(100*nonnull/max(len(df),1),2),'sample_distinct':distinct,'sample_cohort_rows':len(sample_cohort) if cohort is not None else None,'sample_cohort_nonnull':cohort_nonnull,'is_join_key':c in KEYS,'is_time_like':bool(TIME_RE.search(c))})
            print(f'[{i}/{len(paths)}] {rel}: {len(header)} columns',flush=True)
        except Exception as e:
            errors.append({'file':rel,'issue':f'{type(e).__name__}: {str(e)[:400]}'})
            print(f'[{i}/{len(paths)}] ERROR {rel}: {e}',flush=True)
    pd.DataFrame(tables).to_csv(out/'tables.csv',index=False)
    pd.DataFrame(cols).to_csv(out/'all_columns.csv',index=False)
    with open(out/'all_columns_by_table.txt','w') as f:
        for t in tables:
            f.write(f"\n## {t['file']} ({t['columns']} columns)\n")
            for c in cols:
                if c['file']==t['file']: f.write(f"  {c['column']} [{c['sample_dtype']}] sample non-null {c['sample_nonnull_percent']}%\n")
    pd.DataFrame(errors,columns=['file','issue']).to_csv(out/'errors.csv',index=False)
    report={'source_root':str(root),'files_found':len(paths),'files_inspected':len(tables),'columns_total_across_tables':len(cols),'unique_column_names':len(set(x['column'] for x in cols)),'errors':len(errors),'sample_rows_per_table':args.sample_rows,'notes':['All columns are enumerated from file schemas, including empty columns.','Null fractions and types are estimates from the first N rows, not whole-table statistics.','No patient-level data values are exported.','This inventory does not establish feature availability at a 24-hour landmark.','Cohort sample coverage is not population-level coverage.','Parquet row counts are exact metadata; CSV row counts are not calculated.']}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print('\nSUMMARY',json.dumps(report,indent=2),flush=True)
    print(f'Outputs: {out}',flush=True)
if __name__=='__main__': main()
