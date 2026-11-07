#!/usr/bin/env python3
"""Create a compact pre-landmark longitudinal companion to MIMIC_IV_TEAM_PACKAGE.zip.

Only for use in an institutionally approved MIMIC-IV research environment.
Requires pandas, pyarrow, numpy. Does not include held-out evaluation patients.
"""
import argparse, hashlib, io, json, os, shutil, zipfile
from pathlib import Path
import pandas as pd

SOURCES = {
 'labs': dict(file='hosp/labevents.csv', columns=['hadm_id','itemid','charttime','storetime','valuenum','valueuom'], time='charttime', available='storetime', values=['itemid','valuenum','valueuom'], strict=True),
 'emar': dict(file='hosp/emar.csv', columns=['hadm_id','charttime','storetime','medication','event_txt'], time='charttime', available='storetime', values=['medication'], strict=True),
 'prescriptions': dict(file='hosp/prescriptions.csv', columns=['hadm_id','starttime','drug','route'], time='starttime', available=None, values=['drug','route'], strict=False),
 'orders': dict(file='hosp/poe.csv', columns=['hadm_id','ordertime','order_type','order_subtype'], time='ordertime', available=None, values=['order_type','order_subtype'], strict=False),
 'microbiology': dict(file='hosp/microbiologyevents.csv', columns=['hadm_id','charttime','storetime','test_name'], time='charttime', available='storetime', values=['test_name'], strict=True),
}

def locate(base):
 p=Path(base).expanduser()
 if (p/'hosp/admissions.csv').exists(): return p
 q=p/'physionet.org/files/mimiciv/3.1'
 if (q/'hosp/admissions.csv').exists(): return q
 raise FileNotFoundError('Expected hosp/admissions.csv below --base')

def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--base',required=True,help='Local MIMIC-IV 3.1 root or parent')
 ap.add_argument('--team-zip',required=True,help='Existing MIMIC_IV_TEAM_PACKAGE.zip')
 ap.add_argument('--output',required=True,help='New output directory, e.g. ./outputs/temporal_package')
 ap.add_argument('--chunksize',type=int,default=200000)
 ap.add_argument('--sources',nargs='+',choices=list(SOURCES),default=['labs','emar','prescriptions','orders','microbiology'])
 ap.add_argument('--strict-only',action='store_true',help='Export only sources with a second recorded availability timestamp (labs/eMAR/microbiology)')
 ap.add_argument('--keep-folder',action='store_true',help='Retain extracted Parquet folder after ZIP is created')
 a=ap.parse_args()
 if a.chunksize<1000:ap.error('--chunksize must be >= 1000')
 root=locate(a.base);zp=Path(a.team_zip).expanduser();out=Path(a.output).expanduser()
 if not zp.exists():raise FileNotFoundError(zp)
 if out.exists() and any(out.iterdir()):raise FileExistsError('Output directory is not empty: '+str(out))
 out.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(zp) as z:
  names=z.namelist()
  fkey=next((n for n in names if n.endswith('/data/phase18/prelandmark_features.parquet')),None)
  ekey=next((n for n in names if n.endswith('/data/phase18/first_events.parquet')),None)
  if not fkey or not ekey:raise ValueError('Team ZIP lacks Phase18 features/events')
  f=pd.read_parquet(io.BytesIO(z.read(fkey)),columns=['hadm_id','subject_id','split'])
  ev=pd.read_parquet(io.BytesIO(z.read(ekey)))
  # Protect reserved evaluation: explicitly use only development split labels.
  f=f.loc[f.split.isin(['train','early_stop','validation'])].copy()
  ev=ev.loc[ev.hadm_id.isin(f.hadm_id)].copy()
  if f.hadm_id.duplicated().any() or ev.hadm_id.duplicated().any():raise ValueError('Duplicate admission keys')
  cohort=f.merge(ev,on='hadm_id',validate='one_to_one')
  if cohort.groupby('subject_id').split.nunique().gt(1).any():raise ValueError('Patient split violation')
  if 'landmark' not in cohort:raise ValueError('Missing landmark in first_events')
 adm=pd.read_csv(root/'hosp/admissions.csv',usecols=['hadm_id','admittime'],low_memory=False)
 adm['admittime']=pd.to_datetime(adm.admittime,errors='coerce')
 cohort=cohort.merge(adm,on='hadm_id',validate='one_to_one')
 cohort['landmark']=pd.to_datetime(cohort.landmark,errors='coerce')
 if cohort[['admittime','landmark']].isna().any().any():raise ValueError('Missing admission/landmark times')
 if not (cohort.landmark>cohort.admittime).all():raise ValueError('Bad landmark times')
 cohort[['hadm_id','subject_id','split','admittime','landmark','first_event','event_hours','icu_24_72']].to_parquet(out/'cohort_landmarks_and_outcomes.parquet',index=False,compression='zstd')
 keys=cohort.set_index('hadm_id')[['admittime','landmark']]
 selected=[s for s in a.sources if not a.strict_only or SOURCES[s]['strict']]
 report={'source_version':'MIMIC-IV 3.1','team_zip_sha256':sha256(zp),'development_admissions':int(len(cohort)),'unique_patients':int(cohort.subject_id.nunique()),'landmark_policy':'source chart/event time >= admission and < landmark; availability/storetime also < landmark when supplied','sources':{},'limitations':[
  'Only development admissions; reserved evaluation admissions and labels excluded.',
  'Prescription starttime is not a verified entry/availability time. Orders may be updated retrospectively; only basic type fields exported.',
  'eMAR and microbiology storetime cutoffs do not guarantee all text fields were finalized by landmark.',
  'No ward vital signs exported: MIMIC-IV hosp does not provide a comprehensive ward vital-sign time series.',
  'This dataset supports pre-landmark sequences and fixed 24h-landmark survival outcomes, but NOT repeated later landmarks without new extraction.',
  'The cohort already conditions on being in hospital and not in ICU at 24h; it cannot support predictions at admission without rebuilding eligibility.',
  'Restricted patient-level MIMIC-IV-derived data: use only approved storage and authorized users.'
 ]}
 for name in selected:
  cfg=SOURCES[name];src=root/cfg['file'];sub=out/'events'/name;sub.mkdir(parents=True,exist_ok=True)
  if not src.exists():raise FileNotFoundError(src)
  header=set(pd.read_csv(src,nrows=0).columns)
  missing=set(cfg['columns'])-header
  if missing:raise ValueError(f'{name}: missing columns {sorted(missing)}')
  scanned=kept=0;parts=0
  for i,ch in enumerate(pd.read_csv(src,usecols=cfg['columns'],chunksize=a.chunksize,low_memory=False)):
   scanned+=len(ch)
   ch=ch.loc[ch.hadm_id.isin(keys.index)].copy()
   if ch.empty:continue
   ch[cfg['time']]=pd.to_datetime(ch[cfg['time']],errors='coerce')
   if cfg['available']:ch[cfg['available']]=pd.to_datetime(ch[cfg['available']],errors='coerce')
   ch=ch.join(keys,on='hadm_id')
   ok=ch[cfg['time']].ge(ch.admittime)&ch[cfg['time']].lt(ch.landmark)
   if cfg['available']:ok &= ch[cfg['available']].notna()&ch[cfg['available']].lt(ch.landmark)
   ch=ch.loc[ok].copy()
   if ch.empty:continue
   ch['hours_since_admission']=(ch[cfg['time']]-ch.admittime).dt.total_seconds().div(3600).astype('float32')
   ch['hours_before_landmark']=(ch.landmark-ch[cfg['time']]).dt.total_seconds().div(3600).astype('float32')
   if cfg['available']:
    ch['available_hours_since_admission']=(ch[cfg['available']]-ch.admittime).dt.total_seconds().div(3600).astype('float32')
   cols=['hadm_id','hours_since_admission','hours_before_landmark']+(['available_hours_since_admission'] if cfg['available'] else [])+cfg['values']
   ch=ch[cols]
   ch.to_parquet(sub/f'part_{i:06d}.parquet',index=False,compression='zstd')
   kept+=len(ch);parts+=1
   if i%50==0:print(f'{name}: scanned {scanned:,}, kept {kept:,}',flush=True)
  report['sources'][name]={'rows_scanned':scanned,'rows_retained':kept,'parts':parts,'strict_availability_timestamp':bool(cfg['strict']),'source':cfg['file']}
  print(f'FINISHED {name}: {kept:,} events in {parts} parts',flush=True)
 (out/'manifest.json').write_text(json.dumps(report,indent=2))
 (out/'README.md').write_text('''# MIMIC-IV Temporal Companion Dataset\n\nThis package complements MIMIC_IV_TEAM_PACKAGE.zip. It contains only development admissions, not the reserved evaluation cohort.\n\nEach events/<source>/part_*.parquet is long-form and keyed by hadm_id. Sort by hours_since_admission for a within-admission sequence.\n\nThe cohort_landmarks_and_outcomes.parquet file provides subject_id, split, first_event and event_hours (hours AFTER the 24-hour landmark), and icu_24_72.\n\nFor leakage-sensitive work, prefer labs. eMAR and microbiology have charttime+storetime cutoffs but field-finalization remains uncertain. Prescriptions and orders have only single event timestamps and are NOT independently availability-certified. Exclude them from claims of strict prediction-time validity pending audit.\n\nThis is NOT a complete time-series dataset of hospital vitals. It does NOT enable arbitrary future landmarks or hospital-wide occupancy reconstruction. See manifest.json for further limitations.\n\nAll files are restricted MIMIC-IV-derived patient-level data. Use only authorized research environments and approved transfer mechanisms.\n''')
 dest=out.with_suffix('.zip')
 if dest.exists():raise FileExistsError(dest)
 print('Compressing ZIP...',flush=True)
 with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
  for p in sorted(out.rglob('*')):
   if p.is_file():z.write(p,arcname=str(Path(out.name)/p.relative_to(out)))
 print('ZIP:',dest,'size MB:',round(dest.stat().st_size/1024**2,2))
 if not a.keep_folder:shutil.rmtree(out)
 print('COMPLETE. Original source data unchanged.')

if __name__=='__main__':main()
