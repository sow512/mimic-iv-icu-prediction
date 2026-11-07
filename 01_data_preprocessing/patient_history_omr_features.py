#!/usr/bin/env python3
"""Phase 9A: pre-landmark longitudinal history and outpatient OMR features.
Research only. Does not use discharge diagnoses from the index admission.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def table(base, section, name):
    candidates = [base / 'physionet.org/files/mimiciv/3.1' / section,
                  base / section, base]
    for root in candidates:
        for suffix in ('.csv.gz', '.csv'):
            p = root / (name + suffix)
            if p.exists():
                return p
    raise FileNotFoundError(f'Cannot locate {section}/{name}.csv[.gz] beneath {base}')


def parse_dt(df, cols):
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--phase1', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--no-omr', action='store_true')
    args = ap.parse_args()
    base = Path(args.base)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(Path(args.phase1) / 'phase1_landmark_predictors.parquet')
    audit = pd.read_parquet(Path(args.phase1) / 'phase1_cohort_audit.parquet', columns=['subject_id','hadm_id','admittime'])
    if cohort.hadm_id.duplicated().any() or audit.hadm_id.duplicated().any():
        raise ValueError('Duplicate hadm_id in phase 1')
    idx = cohort[['subject_id','hadm_id','landmark_time']].merge(audit, on=['subject_id','hadm_id'], validate='one_to_one')
    idx = parse_dt(idx, ['landmark_time','admittime'])
    if idx[['landmark_time','admittime']].isna().any().any():
        raise ValueError('Missing landmark/admission time')
    print('Eligible admissions:', len(idx), flush=True)
    adm = pd.read_csv(table(base, 'hosp', 'admissions'), usecols=['subject_id','hadm_id','admittime','dischtime'])
    adm = parse_dt(adm, ['admittime','dischtime'])
    # Only completed earlier hospitalizations; exclude overlapping or future encounters.
    prior = idx[['subject_id','hadm_id','admittime']].merge(adm, on='subject_id', suffixes=('_index','_prior'))
    prior = prior.loc[(prior.hadm_id_index != prior.hadm_id_prior) &
                      (prior.dischtime < prior.admittime_index) &
                      prior.dischtime.notna()].copy()
    prior['prior_los_days'] = (prior.dischtime - prior.admittime_prior).dt.total_seconds()/86400
    prior = prior.loc[prior.prior_los_days >= 0]
    h = prior.groupby('hadm_id_index').agg(
        prior_hospital_admissions=('hadm_id_prior','nunique'),
        prior_total_hospital_days=('prior_los_days','sum'),
        prior_mean_hospital_days=('prior_los_days','mean'),
        prior_last_discharge=('dischtime','max')
    ).reset_index().rename(columns={'hadm_id_index':'hadm_id'})
    h = idx[['hadm_id','admittime']].merge(h, on='hadm_id', how='left', validate='one_to_one')
    h['days_since_prior_discharge'] = (h.admittime - h.prior_last_discharge).dt.total_seconds()/86400
    h = h.drop(columns=['admittime','prior_last_discharge'])
    h['has_prior_hospitalization'] = (h.prior_hospital_admissions.fillna(0)>0).astype('int8')
    for c in ['prior_hospital_admissions','prior_total_hospital_days']:
        h[c] = h[c].fillna(0)
    print('History: prior hospitalization coverage:', int(h.has_prior_hospitalization.sum()), flush=True)
    icu = pd.read_csv(table(base,'icu','icustays'), usecols=['subject_id','stay_id','intime','outtime'])
    icu = parse_dt(icu, ['intime','outtime'])
    prior_icu = idx[['subject_id','hadm_id','admittime']].merge(icu, on='subject_id')
    prior_icu = prior_icu.loc[(prior_icu.outtime < prior_icu.admittime) &
                              (prior_icu.intime <= prior_icu.outtime) & prior_icu.outtime.notna()].copy()
    prior_icu['icu_los_days'] = (prior_icu.outtime - prior_icu.intime).dt.total_seconds()/86400
    i = prior_icu.groupby('hadm_id').agg(
        prior_icu_stays=('stay_id','nunique'),
        prior_icu_total_days=('icu_los_days','sum'),
        prior_last_icu_discharge=('outtime','max')
    ).reset_index()
    i = idx[['hadm_id','admittime']].merge(i,on='hadm_id',how='left',validate='one_to_one')
    i['days_since_prior_icu'] = (i.admittime-i.prior_last_icu_discharge).dt.total_seconds()/86400
    i = i.drop(columns=['admittime','prior_last_icu_discharge'])
    i['has_prior_icu'] = (i.prior_icu_stays.fillna(0)>0).astype('int8')
    for c in ['prior_icu_stays','prior_icu_total_days']:
        i[c] = i[c].fillna(0)
    result = idx[['subject_id','hadm_id']].merge(h,on='hadm_id',validate='one_to_one').merge(i,on='hadm_id',validate='one_to_one')
    report = {'eligible_admissions':len(idx), 'prior_hospitalization_coverage':int(h.has_prior_hospitalization.sum()), 'prior_icu_coverage':int(i.has_prior_icu.sum())}
    if not args.no_omr:
        omr_path = table(base,'hosp','omr')
        print('Reading outpatient OMR:', omr_path, flush=True)
        omr = pd.read_csv(omr_path, usecols=['subject_id','chartdate','result_name','result_value'], low_memory=False)
        omr['chartdate'] = pd.to_datetime(omr.chartdate, errors='coerce')
        # OMR has date-only timestamps: strictly before admission date avoids same-day ambiguity.
        omr = omr.loc[omr.result_name.isin(['Blood Pressure','BMI (kg/m2)','Weight (Lbs)','Height (Inches)'])]
        omr = idx[['subject_id','hadm_id','admittime']].merge(omr,on='subject_id')
        omr = omr.loc[omr.chartdate < omr.admittime.dt.normalize()].copy()
        omr = omr.sort_values(['hadm_id','result_name','chartdate'])
        bp = omr.loc[omr.result_name == 'Blood Pressure'].copy()
        parts = bp.result_value.astype('string').str.extract(r'^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$')
        bp['prior_systolic_bp'] = pd.to_numeric(parts[0], errors='coerce')
        bp['prior_diastolic_bp'] = pd.to_numeric(parts[1], errors='coerce')
        bp.loc[~bp.prior_systolic_bp.between(50,300),'prior_systolic_bp']=np.nan
        bp.loc[~bp.prior_diastolic_bp.between(20,200),'prior_diastolic_bp']=np.nan
        bp = bp.groupby('hadm_id')[['prior_systolic_bp','prior_diastolic_bp']].last().reset_index()
        result = result.merge(bp,on='hadm_id',how='left',validate='one_to_one')
        for source,target,lo,hi in [('BMI (kg/m2)','prior_bmi',8,100),('Weight (Lbs)','prior_weight_lbs',40,700),('Height (Inches)','prior_height_inches',36,90)]:
            z = omr.loc[omr.result_name == source, ['hadm_id','result_value']].copy()
            z[target] = pd.to_numeric(z.result_value,errors='coerce')
            z.loc[~z[target].between(lo,hi),target]=np.nan
            z = z.groupby('hadm_id')[target].last().reset_index()
            result = result.merge(z,on='hadm_id',how='left',validate='one_to_one')
        for c in ['prior_systolic_bp','prior_diastolic_bp','prior_bmi','prior_weight_lbs','prior_height_inches']:
            report[c+'_available'] = int(result[c].notna().sum())
    if len(result)!=len(idx) or result.hadm_id.duplicated().any():
        raise AssertionError('Output lost or duplicated admissions')
    if result.columns.str.contains('icu_transfer|dischtime|deathtime|first_icu_intime').any():
        raise AssertionError('Outcome leakage column')
    path=out/'phase9a_history_omr_features.parquet'
    result.to_parquet(path,index=False)
    report['feature_columns'] = [c for c in result if c not in ['subject_id','hadm_id']]
    (out/'phase9a_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print('Saved:',path)

if __name__ == '__main__':
    main()
