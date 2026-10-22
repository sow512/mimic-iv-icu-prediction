#!/usr/bin/env python3
"""Crash-resumable MIMIC-IV lab extraction. Checkpoint one source CSV chunk per file.
Only charted AND stored measurements before the 24-hour landmark are retained.
Final aggregation is disk-backed DuckDB, preserving (hadm_id,itemid,unit).
"""
import argparse, json, os, time
from pathlib import Path
import pandas as pd

def root_path(base):
    b=Path(base).expanduser()
    return b if (b/'hosp/labevents.csv').exists() else b/'physionet.org/files/mimiciv/3.1'

def atomic_parquet(frame,path):
    tmp=path.with_suffix('.tmp.parquet');frame.to_parquet(tmp,index=False);os.replace(tmp,path)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',required=True);ap.add_argument('--phase18',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--chunksize',type=int,default=200000);ap.add_argument('--memory-limit',default='2GB')
    ap.add_argument('--threads',type=int,default=2);ap.add_argument('--stage',choices=['extract','aggregate','both'],default='both')
    a=ap.parse_args();root=root_path(a.base);out=Path(a.output).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    parts=out/'parts';parts.mkdir(exist_ok=True)
    if a.chunksize<1000:raise ValueError('chunksize must be >=1000')
    if a.stage in ('extract','both'):
        events=pd.read_parquet(Path(a.phase18)/'first_events.parquet',columns=['hadm_id','landmark'])
        if events.hadm_id.duplicated().any():raise ValueError('Duplicate hadm_id in Phase 18')
        adm=pd.read_csv(root/'hosp/admissions.csv',usecols=['hadm_id','admittime'])
        adm['admittime']=pd.to_datetime(adm.admittime,errors='coerce')
        key=events.merge(adm,on='hadm_id',validate='one_to_one').set_index('hadm_id')[['admittime','landmark']]
        key['landmark']=pd.to_datetime(key.landmark,errors='coerce')
        if key.isna().any().any():raise ValueError('Missing timestamps in cohort')
        meta={'phase18':str(Path(a.phase18).resolve()),'chunk_size':a.chunksize,'cohort_n':len(key),'source_size':(root/'hosp/labevents.csv').stat().st_size}
        meta_path=out/'extraction_config.json'
        if meta_path.exists() and json.loads(meta_path.read_text())!=meta:
            raise ValueError('Extraction settings/source changed. Use a NEW output directory; never mix chunk partitions.')
        meta_path.write_text(json.dumps(meta,indent=2))
        src=root/'hosp/labevents.csv';scanned=0;retained=0
        use=['hadm_id','itemid','charttime','storetime','valuenum','valueuom']
        for i,ch in enumerate(pd.read_csv(src,usecols=use,chunksize=a.chunksize,low_memory=False)):
            scanned+=len(ch);path=parts/f'chunk_{i:06d}.parquet'
            if path.exists():
                if i%25==0:print(f'Chunk {i}: checkpoint exists; skipped (CSV read still required)',flush=True)
                continue
            ch=ch[ch.hadm_id.isin(key.index)&ch.valuenum.notna()&ch.itemid.notna()].copy()
            if not ch.empty:
                ch['charttime']=pd.to_datetime(ch.charttime,errors='coerce');ch['storetime']=pd.to_datetime(ch.storetime,errors='coerce')
                ch=ch.join(key,on='hadm_id')
                ch=ch[ch.charttime.ge(ch.admittime)&ch.charttime.lt(ch.landmark)&ch.storetime.notna()&ch.storetime.lt(ch.landmark)].copy()
            if ch.empty:
                ch=pd.DataFrame({'hadm_id':pd.Series(dtype='int64'),'itemid':pd.Series(dtype='int64'), 'unit':pd.Series(dtype='str'),'charttime':pd.Series(dtype='datetime64[ns]'),'valuenum':pd.Series(dtype='float64')})
            else:
                ch=ch[['hadm_id','itemid','valueuom','charttime','valuenum']].rename(columns={'valueuom':'unit'})
                ch['hadm_id']=ch.hadm_id.astype('int64');ch['itemid']=ch.itemid.astype('int64');ch['unit']=ch.unit.fillna('UNKNOWN').astype(str)
            retained+=len(ch);atomic_parquet(ch,path)
            if i%10==0:print(f'chunk={i} scanned={scanned:,} newly_retained={retained:,} checkpoint={path.name}',flush=True)
        (out/'extraction_complete.json').write_text(json.dumps({'chunks':i+1 if scanned else 0,'scanned':scanned,'newly_retained':retained},indent=2))
        print('Extraction checkpoints complete',flush=True)
    if a.stage in ('aggregate','both'):
        if not (out/'extraction_complete.json').exists():raise RuntimeError('Extraction incomplete. Run --stage extract first.')
        import duckdb
        db=duckdb.connect(str(out/'lab_aggregation.duckdb'))
        db.execute(f"SET memory_limit='{a.memory_limit}'")
        db.execute(f'SET threads={a.threads}')
        (out/'duckdb_tmp').mkdir(exist_ok=True)
        db.execute(f"SET temp_directory='{str(out/'duckdb_tmp')}'")
        glob=str(parts/'chunk_*.parquet').replace("'","''")
        # DuckDB streams input and spills to disk; avoid a Python dict with millions of keys.
        sql=f"""COPY (
            SELECT hadm_id, itemid, unit,
                   COUNT(*)::BIGINT AS n, AVG(valuenum) AS mean,
                   MIN(valuenum) AS min, MAX(valuenum) AS max,
                   ARG_MAX(valuenum, charttime) AS latest
            FROM read_parquet('{glob}')
            GROUP BY hadm_id,itemid,unit
        ) TO '{str(out/'labs_aggregated.tmp.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
        db.execute(sql);db.close()
        os.replace(out/'labs_aggregated.tmp.parquet',out/'labs_aggregated.parquet')
        print('Saved:',out/'labs_aggregated.parquet',flush=True)
if __name__=='__main__':main()
