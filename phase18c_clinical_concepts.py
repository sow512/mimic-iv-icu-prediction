#!/usr/bin/env python3
"""Phase 18C: MIMIC-IV clinical concept extraction for hospital-wide ICU prediction.

Outputs admission-keyed concept-count features (wide Parquet), with one module per file.
Never uses outcomes, future ICU records, or current-admission discharge diagnoses.
Requires pandas, numpy, pyarrow. Read the accompanying README before modeling.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

MODULES = ('prescriptions', 'emar', 'orders', 'history_diagnoses')
MED_PATTERNS = {
    'vasopressor_inotrope': r'\b(norepinephrine|noradrenaline|epinephrine|adrenaline|vasopressin|phenylephrine|dopamine|dobutamine|milrinone)\b',
    'diuretic': r'\b(furosemide|lasix|bumetanide|bumex|torsemide|metolazone|spironolactone|chlorothiazide)\b',
    'antibiotic': r'\b(vancomycin|ceftriaxone|cefepime|cefazolin|piperacillin|tazobactam|zosyn|meropenem|imipenem|azithromycin|levofloxacin|ciprofloxacin|metronidazole|linezolid|daptomycin|ampicillin|amoxicillin|clindamycin|gentamicin)\b',
    'anticoagulant': r'\b(heparin|enoxaparin|warfarin|apixaban|rivaroxaban|dabigatran|fondaparinux|argatroban|bivalirudin)\b',
    'antiplatelet': r'\b(aspirin|clopidogrel|ticagrelor|prasugrel)\b',
    'insulin': r'\b(insulin|humalog|novolog|lantus|lispro|aspart|glargine)\b',
    'glucocorticoid': r'\b(hydrocortisone|prednisone|prednisolone|methylprednisolone|dexamethasone)\b',
    'opioid': r'\b(morphine|fentanyl|hydromorphone|oxycodone|methadone|tramadol)\b',
    'sedative': r'\b(propofol|midazolam|lorazepam|dexmedetomidine|diazepam|ketamine)\b',
    'antihypertensive': r'\b(lisinopril|losartan|valsartan|amlodipine|metoprolol|carvedilol|labetalol|hydralazine|nicardipine|diltiazem)\b',
    'electrolyte_replacement': r'\b(potassium chloride|magnesium sulfate|calcium gluconate|sodium bicarbonate|sodium phosphate)\b',
    'iv_fluid': r'\b(normal saline|sodium chloride|lactated ringers|ringer.s|dextrose 5|d5w|albumin)\b',
}
DX_PATTERNS = {
    'renal_aki': (r'^(N17)', r'^584'),
    'renal_ckd': (r'^(N18)', r'^585'),
    'heart_failure': (r'^(I50)', r'^428'),
    'ischemic_heart': (r'^(I2[0-5])', r'^41[0-4]'),
    'arrhythmia': (r'^(I4[4-9])', r'^427'),
    'diabetes': (r'^(E0[8-9]|E1[0-3])', r'^250'),
    'copd': (r'^(J4[0-4])', r'^49[0-6]'),
    'pneumonia': (r'^(J1[2-8])', r'^48[0-6]'),
    'sepsis': (r'^(A40|A41|R652)', r'^(038|99591|99592)'),
    'stroke': (r'^(I6[0-9])', r'^(43[0-8])'),
    'liver_disease': (r'^(K7[0-7])', r'^(57[0-3])'),
    'malignancy': (r'^(C[0-9])', r'^(1[4-9][0-9]|20[0-8])'),
}

def locate(base):
    p = Path(base).expanduser().resolve()
    if (p/'hosp'/'admissions.csv').exists(): return p
    q = p/'physionet.org/files/mimiciv/3.1'
    if (q/'hosp'/'admissions.csv').exists(): return q
    raise FileNotFoundError('Cannot locate hosp/admissions.csv')

def slug(value):
    return re.sub('[^a-z0-9]+','_',str(value).lower()).strip('_')[:45] or 'unknown'


def load_cohort(phase18, root=None):
    import pandas as pd
    from pathlib import Path

    p = Path(phase18) / "prelandmark_features.parquet"
    df = pd.read_parquet(p)

    if "hadm_id" not in df.columns:
        raise ValueError("Phase 18 features missing hadm_id")

    if "admittime" not in df.columns:
        if root is None:
            raise ValueError("MIMIC-IV root required for admission timestamps")

        admissions = pd.read_csv(
            Path(root) / "hosp" / "admissions.csv",
            usecols=["hadm_id", "admittime"]
        )

        admissions["admittime"] = pd.to_datetime(
            admissions["admittime"], errors="coerce"
        )

        df = df.merge(
            admissions,
            on="hadm_id",
            how="left",
            validate="one_to_one"
        )

    df["admittime"] = pd.to_datetime(
        df["admittime"], errors="coerce"
    )

    if df["admittime"].isna().any():
        raise ValueError("Missing admission timestamps after merge")

    if "landmark_time" not in df.columns:
        df["landmark_time"] = (
            df["admittime"] + pd.Timedelta(hours=24)
        )

    return df

def write(out, name, df, cohort, force):
    dest=out/f'{name}.parquet'
    if dest.exists() and not force:
        print('Already exists, skipping:',dest,flush=True);return
    if df.hadm_id.duplicated().any():raise ValueError('Duplicate hadm_id in '+name)
    if not df.hadm_id.isin(cohort.hadm_id).all():raise ValueError('Unexpected admission in '+name)
    temp=out/f'{name}.tmp.parquet'
    df.to_parquet(temp,index=False)
    temp.replace(dest)
    report={'module':name,'rows':len(df),'feature_columns':len(df.columns)-1,
            'cohort_admissions':len(cohort),'output':str(dest),
            'feature_names':list(df.columns[1:]),
            'note':'Missing admission row means no qualifying observed record, not clinical absence.'}
    (out/f'{name}.json').write_text(json.dumps(report,indent=2))
    print(f'Saved {name}: {len(df):,} admissions, {len(df.columns)-1} features',flush=True)

def medication_labels(s):
    s=s.fillna('').astype(str).str.lower()
    return {k:s.str.contains(p,regex=True,na=False) for k,p in MED_PATTERNS.items()}

def stream_events(root,cohort,module,chunksize):
    if module=='prescriptions':
        path=root/'hosp/prescriptions.csv';cols=['hadm_id','starttime','drug','route']
        timestamp='starttime';availability=None;concept='drug'
    elif module=='emar':
        path=root/'hosp/emar.csv';cols=['hadm_id','charttime','storetime','medication','event_txt']
        timestamp='charttime';availability='storetime';concept='medication'
    else:
        path=root/'hosp/poe.csv';cols=['hadm_id','ordertime','order_type','order_subtype','transaction_type','order_status']
        timestamp='ordertime';availability=None;concept='order_type'
    if not path.exists():raise FileNotFoundError(path)
    keys=cohort.set_index('hadm_id')[['admittime','landmark_time']]
    counts=Counter(); scanned=0; retained=0
    for ch in pd.read_csv(path,usecols=cols,chunksize=chunksize,low_memory=False):
        scanned+=len(ch)
        ch=ch[ch.hadm_id.isin(keys.index)].copy()
        if ch.empty:continue
        ch[timestamp]=pd.to_datetime(ch[timestamp],errors='coerce')
        if availability:ch[availability]=pd.to_datetime(ch[availability],errors='coerce')
        ch=ch.join(keys,on='hadm_id')
        mask=ch[timestamp].notna()&(ch[timestamp]>=ch.admittime)&(ch[timestamp]<ch.landmark_time)
        if availability:mask &= ch[availability].notna()&(ch[availability]<ch.landmark_time)
        ch=ch.loc[mask]
        if ch.empty:continue
        retained+=len(ch)
        feats={'total_events':pd.Series(True,index=ch.index)}
        if module in ('prescriptions','emar'):
            feats.update({f'med_{k}':v for k,v in medication_labels(ch[concept]).items()})
            if module=='prescriptions':
                route=ch.route.fillna('').astype(str).str.lower()
                feats['route_iv']=route.str.contains(r'\b(iv|intravenous)\b',regex=True)
                feats['route_oral']=route.str.contains(r'\b(po|oral|by mouth)\b',regex=True)
            else:
                evt=ch.event_txt.fillna('').astype(str).str.lower()
                feats['event_not_given']=evt.str.contains(r'not given|refused|held|missed',regex=True)
        else:
            typ=ch.order_type.fillna('').astype(str).map(slug)
            # Categories derive from actual order_type values; avoid enormous vocabulary.
            for k in ('lab','radiology','medication','procedure','consult','diet','nursing','blood','respiratory'):
                feats[f'order_{k}']=typ.str.contains(k,regex=False)
            feats['order_discontinued']=ch.transaction_type.fillna('').astype(str).str.lower().str.contains('discontinue|cancel',regex=True)
        feat_df=pd.DataFrame(feats,index=ch.index).astype('int8')
        feat_df.insert(0,'hadm_id',ch.hadm_id.to_numpy())
        grouped=feat_df.groupby('hadm_id',sort=False).sum()
        for h,row in grouped.iterrows():
            h=int(h)
            for k,v in row.items():
                if v:counts[(h,k)]+=int(v)
        if scanned//5000000 != (scanned-len(ch))//5000000:
            print(f'{module}: scanned {scanned:,}; kept {retained:,}; counters {len(counts):,}',flush=True)
    if not counts:return pd.DataFrame({'hadm_id':pd.Series(dtype='int64')})
    records=[(h,k,v) for (h,k),v in counts.items()]
    long=pd.DataFrame(records,columns=['hadm_id','feature','value'])
    wide=long.pivot(index='hadm_id',columns='feature',values='value').fillna(0)
    wide.columns=[f'{module}_{c}_24h' for c in wide.columns]
    return wide.reset_index().rename_axis(columns=None)

def history_diagnoses(root,cohort,chunksize):
    # Map codes to flags per *historical* admission; use ICD version to prevent ambiguity.
    path=root/'hosp/diagnoses_icd.csv'
    valid_hadm=pd.read_csv(root/'hosp/admissions.csv',usecols=['hadm_id','subject_id','admittime','dischtime'],low_memory=False)
    valid_hadm['admittime']=pd.to_datetime(valid_hadm.admittime,errors='coerce')
    valid_hadm['dischtime']=pd.to_datetime(valid_hadm.dischtime,errors='coerce')
    valid_hadm=valid_hadm[valid_hadm.dischtime.notna() & (valid_hadm.dischtime>valid_hadm.admittime)]
    known=set(valid_hadm.hadm_id)
    prior_flags={};scanned=0
    for ch in pd.read_csv(path,usecols=['hadm_id','icd_code','icd_version'],dtype={'icd_code':'string'},chunksize=chunksize):
        scanned+=len(ch)
        ch=ch[ch.hadm_id.isin(known)].copy()
        if ch.empty:continue
        code=ch.icd_code.fillna('').str.upper().str.replace('.','',regex=False)
        vers=ch.icd_version
        flags={}
        for k,(icd10,icd9) in DX_PATTERNS.items():
            flags[k]=((vers.eq(10)&code.str.match(icd10,na=False))|(vers.eq(9)&code.str.match(icd9,na=False))).astype('int8')
        frame=pd.DataFrame(flags)
        frame.insert(0,'hadm_id',ch.hadm_id.to_numpy())
        g=frame.groupby('hadm_id',sort=False).max()
        for h,row in g.iterrows():
            mask=prior_flags.get(int(h),0)
            for i,k in enumerate(DX_PATTERNS):
                if row[k]:mask|=(1<<i)
            prior_flags[int(h)]=mask
        if scanned//5000000 != (scanned-len(ch))//5000000:
            print(f'history_diagnoses: scanned {scanned:,}; indexed admissions {len(prior_flags):,}',flush=True)
    # Per-patient chronological cumulative counts, querying only discharges STRICTLY before index admission.
    hist=valid_hadm[['subject_id','hadm_id','dischtime']].copy()
    hist['mask']=hist.hadm_id.map(prior_flags).fillna(0).astype('int64')
    hist=hist.sort_values(['subject_id','dischtime','hadm_id'])
    names=list(DX_PATTERNS)
    for i,k in enumerate(names):
        hist[f'hist_{k}']=((hist['mask'].to_numpy()>>i)&1).astype('int32')
        hist[f'hist_{k}']=hist.groupby('subject_id')[f'hist_{k}'].cumsum()
    hist['hist_dx_prior_distinct_admissions']=hist.groupby('subject_id').cumcount()+1
    left=cohort[['hadm_id','subject_id','admittime']].sort_values('admittime')
    right=hist[['subject_id','dischtime','hist_dx_prior_distinct_admissions']+[f'hist_{k}' for k in names]].sort_values('dischtime')
    joined=pd.merge_asof(left,right,left_on='admittime',right_on='dischtime',by='subject_id',direction='backward',allow_exact_matches=False)
    columns=['hist_dx_prior_distinct_admissions']+[f'hist_{k}' for k in names]
    joined[columns]=joined[columns].fillna(0).astype('int32')
    return joined[['hadm_id']+columns]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True);p.add_argument('--phase18',required=True);p.add_argument('--output',required=True)
    p.add_argument('--modules',nargs='+',choices=MODULES,default=list(MODULES))
    p.add_argument('--chunksize',type=int,default=500000)
    p.add_argument('--force',action='store_true')
    args=p.parse_args()
    if args.chunksize<1000: p.error('chunksize must be >=1000')
    root=locate(args.base);cohort=load_cohort(args.phase18, root)
    out=Path(args.output).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    print(f'Phase 18C: {len(cohort):,} admissions; modules={args.modules}',flush=True)
    for module in args.modules:
        dest=out/f'{module}.parquet'
        if dest.exists() and not args.force:
            print('Skipping existing',dest,flush=True);continue
        print('\nMODULE',module,flush=True)
        if module=='history_diagnoses':df=history_diagnoses(root,cohort,args.chunksize)
        else:df=stream_events(root,cohort,module,args.chunksize)
        write(out,module,df,cohort,args.force)
    print('COMPLETE:',out,flush=True)
if __name__=='__main__':main()
