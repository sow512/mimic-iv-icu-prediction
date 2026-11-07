#!/usr/bin/env python3
"""Phase 1: retrospective renal ICU-transfer landmark cohort for MIMIC-IV 3.1."""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


def locate(base, module, stem):
    roots = [base, base / 'physionet.org' / 'files' / 'mimiciv' / '3.1', base / 'mimiciv' / '3.1']
    for root in roots:
        for suffix in ('.csv.gz', '.csv'):
            p = root / module / (stem + suffix)
            if p.is_file():
                return p
    raise FileNotFoundError(f'Cannot locate {module}/{stem}.csv[.gz] under {base}')


def load(base, module, stem, columns, dates=()):
    p = locate(base, module, stem)
    print('Reading', p, flush=True)
    df = pd.read_csv(p, usecols=columns, low_memory=False)
    for c in dates:
        df[c] = pd.to_datetime(df[c], errors='coerce')
    return df


def assert_unique(df, col, label):
    if df[col].isna().any() or df[col].duplicated().any():
        raise ValueError(f'{label}: {col} must be unique and non-null')


def build(base, output):
    output.mkdir(parents=True, exist_ok=True)
    patients = load(base, 'hosp', 'patients', ['subject_id','gender','anchor_age','anchor_year'])
    adm = load(base, 'hosp', 'admissions', ['subject_id','hadm_id','admittime','dischtime','deathtime','admission_type','admission_location'], ['admittime','dischtime','deathtime'])
    dx = load(base, 'hosp', 'diagnoses_icd', ['subject_id','hadm_id','icd_code','icd_version'])
    icu = load(base, 'icu', 'icustays', ['subject_id','hadm_id','stay_id','intime','outtime'], ['intime','outtime'])
    assert_unique(patients, 'subject_id', 'patients')
    assert_unique(adm, 'hadm_id', 'admissions')
    assert_unique(icu, 'stay_id', 'icustays')
    report = {}
    def count(name, df):
        report[name] = int(len(df))
        print(f'{name}: {len(df):,}', flush=True)

    count('raw_admissions', adm)
    adm = adm.loc[adm.admittime.notna() & adm.dischtime.notna() & (adm.dischtime > adm.admittime)].copy()
    count('valid_hospital_intervals', adm)
    # Fail closed on conflicting subject links.
    adm = adm.merge(patients, on='subject_id', how='inner', validate='many_to_one')
    adm['age'] = adm.anchor_age + adm.admittime.dt.year - adm.anchor_year
    adm = adm.loc[adm.age.ge(18) & adm.age.notna()].copy()
    count('adult_admissions', adm)

    dx['icd_code'] = dx.icd_code.astype('string').str.upper().str.replace('.', '', regex=False).str.strip()
    version = pd.to_numeric(dx.icd_version, errors='coerce')
    aki = ((version.eq(9) & dx.icd_code.str.startswith('584', na=False)) |
           (version.eq(10) & dx.icd_code.str.startswith('N17', na=False)))
    ckd = ((version.eq(9) & dx.icd_code.str.startswith('585', na=False)) |
           (version.eq(10) & dx.icd_code.str.startswith('N18', na=False)))
    renal = dx.loc[aki | ckd, ['subject_id','hadm_id']].copy()
    renal['AKI'] = aki.loc[renal.index].astype('int8')
    renal['CKD'] = ckd.loc[renal.index].astype('int8')
    renal = renal.groupby(['subject_id','hadm_id'], as_index=False)[['AKI','CKD']].max()
    adm = adm.merge(renal, on=['subject_id','hadm_id'], how='inner', validate='one_to_one')
    count('adult_retrospective_renal_admissions', adm)
    adm['kidney_group'] = np.select([adm.AKI.eq(1)&adm.CKD.eq(1), adm.AKI.eq(1), adm.CKD.eq(1)], ['AKI + CKD','AKI only','CKD only'], default='unknown')

    # ICU data integrity: exclude admissions with ANY malformed/conflicting ICU record,
    # rather than incorrectly treating them as non-ICU admissions.
    valid_icu = (icu.hadm_id.notna() & icu.subject_id.notna() & icu.intime.notna() & icu.outtime.notna() & (icu.outtime > icu.intime))
    malformed_keys = set(icu.loc[~valid_icu & icu.hadm_id.notna(), 'hadm_id'])
    icu = icu.loc[valid_icu].copy()
    joined = icu.merge(adm[['hadm_id','subject_id','admittime','dischtime']], on='hadm_id', how='inner', suffixes=('_icu','_adm'), validate='many_to_one')
    inconsistent = joined.subject_id_icu.ne(joined.subject_id_adm) | joined.intime.lt(joined.admittime) | joined.intime.gt(joined.dischtime) | joined.outtime.lt(joined.admittime)
    malformed_keys.update(joined.loc[inconsistent, 'hadm_id'].tolist())
    adm = adm.loc[~adm.hadm_id.isin(malformed_keys)].copy()
    count('after_excluding_malformed_icu_links', adm)
    icu = icu.loc[~icu.hadm_id.isin(malformed_keys)].copy()
    icu = icu.loc[icu.hadm_id.isin(adm.hadm_id)].copy()
    first = icu.sort_values(['hadm_id','intime','stay_id']).drop_duplicates('hadm_id')[['hadm_id','intime']].rename(columns={'intime':'first_icu_intime'})
    adm = adm.merge(first, on='hadm_id', how='left', validate='one_to_one')
    adm['landmark_time'] = adm.admittime + pd.Timedelta(hours=24)
    adm['horizon_end'] = adm.admittime + pd.Timedelta(hours=72)
    # Death timestamp is an additional conservative eligibility check; discharge
    # remains the primary observation boundary.
    adm = adm.loc[adm.dischtime.gt(adm.landmark_time) & (adm.deathtime.isna() | adm.deathtime.gt(adm.landmark_time))].copy()
    count('hospitalized_and_alive_at_24h', adm)
    adm = adm.loc[adm.first_icu_intime.isna() | adm.first_icu_intime.gt(adm.landmark_time)].copy()
    count('not_in_icu_during_first_24h', adm)

    positive = adm.first_icu_intime.notna() & adm.first_icu_intime.le(adm.horizon_end)
    # Early discharge or death before 72h with no ICU is a competing event:
    # do not silently label it as a fully observed negative.
    fully_observed_negative = (adm.first_icu_intime.isna() | adm.first_icu_intime.gt(adm.horizon_end)) & adm.dischtime.ge(adm.horizon_end) & (adm.deathtime.isna() | adm.deathtime.ge(adm.horizon_end))
    adm['label_status'] = np.select([positive, fully_observed_negative], ['positive','negative'], default='censored')
    report['censored_competing_events'] = int(adm.label_status.eq('censored').sum())
    cohort = adm.loc[adm.label_status.ne('censored')].copy()
    cohort['icu_transfer_24_72h'] = cohort.label_status.eq('positive').astype('int8')
    count('final_labeled_admissions', cohort)
    report['positive_admissions'] = int(cohort.icu_transfer_24_72h.sum())
    report['negative_admissions'] = int((cohort.icu_transfer_24_72h == 0).sum())
    report['unique_patients'] = int(cohort.subject_id.nunique())
    report['prevalence'] = float(cohort.icu_transfer_24_72h.mean()) if len(cohort) else None

    # No future timestamps or outcome fields in the predictor-facing table.
    predictor_columns = ['subject_id','hadm_id','landmark_time','age','gender','admission_type','admission_location']
    predictor_table = cohort[predictor_columns].copy()
    labels = cohort[['subject_id','hadm_id','icu_transfer_24_72h']].copy()
    assert_unique(predictor_table, 'hadm_id', 'predictor_table')
    assert_unique(labels, 'hadm_id', 'labels')
    assert not (cohort.first_icu_intime.notna() & cohort.first_icu_intime.le(cohort.landmark_time)).any()
    assert (cohort.dischtime > cohort.landmark_time).all()
    assert not (cohort.icu_transfer_24_72h.eq(0) & cohort.dischtime.lt(cohort.horizon_end)).any()
    assert not set(['first_icu_intime','dischtime','deathtime','horizon_end','label_status','icu_transfer_24_72h']).intersection(predictor_table.columns)
    predictor_table.to_parquet(output/'phase1_landmark_predictors.parquet', index=False)
    labels.to_parquet(output/'phase1_labels.parquet', index=False)
    # Restricted to research workspace: audit contains post-landmark timestamps.
    cohort.to_parquet(output/'phase1_cohort_audit.parquet', index=False)
    (output/'phase1_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nWrote outputs to:', output)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, type=Path, help='MIMIC-IV root folder')
    parser.add_argument('--output', default=Path('phase1_outputs'), type=Path)
    args = parser.parse_args()
    build(args.base, args.output)
