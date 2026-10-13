#!/usr/bin/env python3
"""Compare 24-72h ICU-transfer prevalence across retrospective diagnosis cohorts.

Research-only. Reuses Phase 1 eligibility and competing-event policy.
Diagnosis codes are discharge-coded and must not be used as 24h predictors.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def locate(base, module, stem):
    roots = [base, base / 'physionet.org/files/mimiciv/3.1', base / 'mimiciv/3.1']
    for root in roots:
        for ext in ('.csv.gz', '.csv'):
            path = root / module / (stem + ext)
            if path.is_file():
                return path
    raise FileNotFoundError(f'Cannot locate {module}/{stem}.csv[.gz] under {base}')


def read(base, module, stem, cols, dates=()):
    path = locate(base, module, stem)
    print('Reading:', path, flush=True)
    frame = pd.read_csv(path, usecols=cols, low_memory=False)
    for col in dates:
        frame[col] = pd.to_datetime(frame[col], errors='coerce')
    return frame


def require_unique(df, column, name):
    if df[column].isna().any() or df[column].duplicated().any():
        raise ValueError(f'{name}: invalid or duplicate {column}')


def main(args):
    base = Path(args.base)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    patients = read(base, 'hosp', 'patients', ['subject_id','anchor_age','anchor_year'])
    adm = read(base, 'hosp', 'admissions', ['subject_id','hadm_id','admittime','dischtime','deathtime'], ['admittime','dischtime','deathtime'])
    dx = read(base, 'hosp', 'diagnoses_icd', ['subject_id','hadm_id','icd_code','icd_version'])
    icu = read(base, 'icu', 'icustays', ['subject_id','hadm_id','stay_id','intime','outtime'], ['intime','outtime'])
    for frame, key, name in [(patients,'subject_id','patients'),(adm,'hadm_id','admissions'),(icu,'stay_id','icustays')]:
        require_unique(frame,key,name)
    adm = adm.loc[adm.admittime.notna() & adm.dischtime.notna() & adm.dischtime.gt(adm.admittime)].copy()
    adm = adm.merge(patients,on='subject_id',how='inner',validate='many_to_one')
    adm['age'] = adm.anchor_age + adm.admittime.dt.year - adm.anchor_year
    adm = adm.loc[adm.age.ge(18)].copy()
    adult_count = len(adm)

    # Diagnose membership per admission. Categories intentionally overlap.
    dx['icd_code'] = dx.icd_code.astype('string').str.upper().str.replace('.', '', regex=False).str.strip()
    ver = pd.to_numeric(dx.icd_version, errors='coerce')
    code = dx.icd_code
    icd9_three = pd.to_numeric(code.str.slice(0,3), errors='coerce')
    v9, v10 = ver.eq(9), ver.eq(10)
    masks = {
        'AKI_CKD': (v9 & (code.str.startswith('584',na=False) | code.str.startswith('585',na=False))) | (v10 & (code.str.startswith('N17',na=False) | code.str.startswith('N18',na=False))),
        'renal_urinary': (v9 & icd9_three.between(580,599)) | (v10 & code.str.match(r'^N[0-3][0-9]',na=False)),
        'cardiovascular': (v9 & icd9_three.between(390,459)) | (v10 & code.str.match(r'^I[0-9]{2}',na=False)),
        'respiratory': (v9 & icd9_three.between(460,519)) | (v10 & code.str.match(r'^J[0-9]{2}',na=False)),
        'infectious': (v9 & icd9_three.between(1,139)) | (v10 & code.str.match(r'^[AB][0-9]{2}',na=False)),
    }
    flags = dx[['subject_id','hadm_id']].copy()
    for name, mask in masks.items():
        flags[name] = mask.fillna(False).astype('int8')
    flags = flags.groupby(['subject_id','hadm_id'],as_index=False)[list(masks)].max()
    adm = adm.merge(flags,on=['subject_id','hadm_id'],how='left',validate='one_to_one')
    for name in masks:
        adm[name] = adm[name].fillna(0).astype('int8')

    # Fail closed for any invalid or conflicting ICU record, as in Phase 1.
    valid = icu.hadm_id.notna() & icu.subject_id.notna() & icu.intime.notna() & icu.outtime.notna() & icu.outtime.gt(icu.intime)
    bad = set(icu.loc[~valid & icu.hadm_id.notna(),'hadm_id'])
    icu = icu.loc[valid].copy()
    links = icu.merge(adm[['hadm_id','subject_id','admittime','dischtime']],on='hadm_id',how='inner',suffixes=('_icu','_adm'),validate='many_to_one')
    inconsistent = links.subject_id_icu.ne(links.subject_id_adm) | links.intime.lt(links.admittime) | links.intime.gt(links.dischtime) | links.outtime.lt(links.admittime)
    bad.update(links.loc[inconsistent,'hadm_id'].tolist())
    adm = adm.loc[~adm.hadm_id.isin(bad)].copy()
    icu = icu.loc[icu.hadm_id.isin(adm.hadm_id)].copy()
    first = icu.sort_values(['hadm_id','intime','stay_id']).drop_duplicates('hadm_id')[['hadm_id','intime']].rename(columns={'intime':'first_icu_intime'})
    adm = adm.merge(first,on='hadm_id',how='left',validate='one_to_one')
    adm['landmark_time'] = adm.admittime + pd.Timedelta(hours=24)
    adm['horizon_end'] = adm.admittime + pd.Timedelta(hours=72)
    adm = adm.loc[adm.dischtime.gt(adm.landmark_time) & (adm.deathtime.isna() | adm.deathtime.gt(adm.landmark_time))].copy()
    adm = adm.loc[adm.first_icu_intime.isna() | adm.first_icu_intime.gt(adm.landmark_time)].copy()
    positive = adm.first_icu_intime.notna() & adm.first_icu_intime.le(adm.horizon_end)
    negative = (adm.first_icu_intime.isna() | adm.first_icu_intime.gt(adm.horizon_end)) & adm.dischtime.ge(adm.horizon_end) & (adm.deathtime.isna() | adm.deathtime.ge(adm.horizon_end))
    adm['outcome_status'] = np.select([positive,negative],['positive','negative'],default='censored')
    adm['label'] = positive.astype('int8')
    rows = []
    categories = [('AKI/CKD (current)',adm.AKI_CKD.eq(1)),('Renal + urinary (ICD chapter)',adm.renal_urinary.eq(1)),('Cardiovascular',adm.cardiovascular.eq(1)),('Respiratory',adm.respiratory.eq(1)),('Infectious',adm.infectious.eq(1)),('All adult admissions',pd.Series(True,index=adm.index))]
    for name, mask in categories:
        group = adm.loc[mask]
        labeled = group.loc[group.outcome_status.ne('censored')]
        n = len(labeled)
        pos = int(labeled.label.sum())
        neg = n-pos
        rows.append({'cohort':name,'landmark_eligible':len(group),'excluded_censored_competing':len(group)-n,'labeled_admissions':n,'unique_patients':int(labeled.subject_id.nunique()),'positive_transfers':pos,'negative_admissions':neg,'positive_percent':round(100*pos/n,4) if n else None,'negative_to_positive_ratio':round(neg/pos,3) if pos else None})
    summary = pd.DataFrame(rows)
    summary.to_csv(out/'cohort_prevalence_comparison.csv',index=False)
    metadata = {'adult_admissions_before_landmark':adult_count,'landmark_eligible_all_adults':len(adm),'definition':'ICU first entry after 24h and by 72h; no ICU by 24h; still hospitalized/alive at 24h; negatives must be observable through 72h','caveats':['Diagnosis groups overlap and are based on discharge-coded ICD data.','Renal + urinary includes non-renal urinary diagnoses, including stones and bladder disorders.','Early discharge/death without ICU is excluded as censored/competing.','No diagnosis groups are used as model predictors.','Comparison of prevalence does not establish which population is clinically preferable.']}
    (out/'comparison_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('\nCOHORT COMPARISON\n')
    print(summary.to_string(index=False))
    print('\nSaved:',out/'cohort_prevalence_comparison.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',required=True)
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
