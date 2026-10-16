#!/usr/bin/env python3
"""Phase 9B: pre-landmark prescription orders and historical discharge-coded comorbidities.
Research only. Discharge diagnoses are used ONLY from prior admissions discharged
before the index admission; prescription orders do not imply administration.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd


def locate(base, section, stem):
    for root in (base / 'physionet.org/files/mimiciv/3.1' / section,
                 base / section, base):
        for ext in ('.csv.gz', '.csv'):
            p = root / (stem + ext)
            if p.is_file():
                return p
    raise FileNotFoundError(f'Cannot find {section}/{stem}.csv[.gz] under {base}')

# Broad screening groups, not validated clinical phenotypes.
# Codes normalized without dots; versions checked separately.
CONDITIONS = {
    'diabetes': {'9': ('250',), '10': ('E08', 'E09', 'E10', 'E11', 'E13')},
    'hypertension': {'9': ('401','402','403','404','405'), '10': ('I10','I11','I12','I13','I15')},
    'heart_failure': {'9': ('428',), '10': ('I50',)},
    'coronary_disease': {'9': ('410','411','412','413','414'), '10': ('I20','I21','I22','I23','I24','I25')},
    'chronic_lung_disease': {'9': ('490','491','492','493','494','495','496'), '10': ('J40','J41','J42','J43','J44','J45','J46','J47')},
    'sepsis': {'9': ('038','99591','99592'), '10': ('A40','A41','R652')},
    'stroke': {'9': ('430','431','432','433','434','436'), '10': ('I60','I61','I62','I63','I64')},
    'liver_disease': {'9': ('570','571','572','573'), '10': ('K70','K71','K72','K73','K74','K75','K76','K77')},
    'malignancy': {'9': tuple(str(i) for i in range(140, 209)), '10': ('C',)},
    'prior_aki': {'9': ('584',), '10': ('N17',)},
    'prior_ckd': {'9': ('585',), '10': ('N18',)},
}
# Lexical medication-order groups: intentionally broad; inspect false matches.
MED_GROUPS = {
    'diuretic': r'\b(furosemide|lasix|bumetanide|torsemide|spironolactone|metolazone|hydrochlorothiazide)\b',
    'vasopressor_order': r'\b(norepinephrine|levophed|epinephrine|phenylephrine|vasopressin|dopamine)\b',
    'antibiotic': r'\b(vancomycin|ceftriaxone|cefepime|piperacillin|tazobactam|meropenem|azithromycin|levofloxacin|ciprofloxacin|ampicillin|amoxicillin|metronidazole)\b',
    'anticoagulant': r'\b(heparin|enoxaparin|warfarin|apixaban|rivaroxaban|dabigatran)\b',
    'insulin': r'\binsulin\b|\b(lispro|glargine|aspart|detemir)\b',
    'opioid': r'\b(morphine|fentanyl|hydromorphone|oxycodone|tramadol)\b',
    'corticosteroid': r'\b(prednisone|prednisolone|methylprednisolone|dexamethasone|hydrocortisone)\b',
    'ace_arb': r'\b(lisinopril|enalapril|losartan|valsartan|candesartan|irbesartan)\b',
}

def build_prior_conditions(base, cohort):
    admissions = pd.read_csv(locate(base, 'hosp', 'admissions'),
                             usecols=['subject_id','hadm_id','admittime','dischtime'])
    admissions['dischtime'] = pd.to_datetime(admissions['dischtime'], errors='coerce')
    admissions['admittime'] = pd.to_datetime(admissions['admittime'], errors='coerce')
    admissions = admissions.dropna(subset=['subject_id','hadm_id','dischtime','admittime'])
    admissions = admissions.loc[admissions.dischtime >= admissions.admittime]
    subjects = set(cohort.subject_id)
    admissions = admissions.loc[admissions.subject_id.isin(subjects)].copy()
    diag = pd.read_csv(locate(base, 'hosp', 'diagnoses_icd'),
                       usecols=['hadm_id','icd_code','icd_version'], dtype={'icd_code':'string'})
    diag = diag.loc[diag.hadm_id.isin(admissions.hadm_id)].copy()
    code = diag.icd_code.fillna('').str.upper().str.replace('.', '', regex=False).str.strip()
    ver = diag.icd_version.astype('Int64').astype('string')
    for name, versions in CONDITIONS.items():
        mask = pd.Series(False, index=diag.index)
        for v, prefixes in versions.items():
            mask |= (ver.eq(v).fillna(False) & code.str.startswith(prefixes).fillna(False))
        diag[name] = mask.astype('int8')
    flags = diag.groupby('hadm_id', as_index=False)[list(CONDITIONS)].max()
    prior = admissions.merge(flags, on='hadm_id', how='left')
    prior[list(CONDITIONS)] = prior[list(CONDITIONS)].fillna(0).astype('int8')
    # A prior encounter is known only after its discharge; cumulative counts
    # are as-of joined strictly before the index admission time.
    prior = prior.sort_values(['subject_id','dischtime','hadm_id'])
    for name in CONDITIONS:
        prior['prior_' + name + '_count'] = prior.groupby('subject_id')[name].cumsum()
    left = cohort[['subject_id','hadm_id','admittime']].copy().sort_values('admittime')
    rightcols = ['subject_id','dischtime'] + ['prior_' + x + '_count' for x in CONDITIONS]
    right = prior[rightcols].sort_values('dischtime')
    result = pd.merge_asof(left, right, left_on='admittime', right_on='dischtime',
                           by='subject_id', direction='backward', allow_exact_matches=False)
    # Remove any theoretical overlapping prior admission: discharge strictly before
    # index admittime already ensures it cannot be the current admission.
    result = result.drop(columns=['dischtime','admittime'])
    for name in CONDITIONS:
        count_col = 'prior_' + name + '_count'
        result[count_col] = result[count_col].fillna(0).astype('int32')
        result['has_prior_' + name] = (result[count_col] > 0).astype('int8')
    return result


def build_orders(base, cohort, chunk_size):
    p = locate(base, 'hosp', 'prescriptions')
    index = cohort[['hadm_id','subject_id','admittime','landmark_time']].set_index('hadm_id')
    ids = set(index.index)
    cols = ['hadm_id','starttime','drug']
    totals = {}
    distinct = {}
    groups = {k: set() for k in MED_GROUPS}
    scanned = selected = valid = 0
    for i, chunk in enumerate(pd.read_csv(p, usecols=cols, chunksize=chunk_size, low_memory=False), 1):
        scanned += len(chunk)
        chunk = chunk.loc[chunk.hadm_id.isin(ids)].copy()
        selected += len(chunk)
        if chunk.empty:
            continue
        chunk['starttime'] = pd.to_datetime(chunk.starttime, errors='coerce')
        chunk = chunk.join(index[['admittime','landmark_time']], on='hadm_id', how='left')
        chunk = chunk.loc[chunk.starttime.notna() &
                          (chunk.starttime >= chunk.admittime) &
                          (chunk.starttime < chunk.landmark_time)].copy()
        valid += len(chunk)
        if chunk.empty:
            continue
        chunk['drug_clean'] = chunk.drug.fillna('').astype(str).str.strip().str.lower()
        for hadm, count in chunk.groupby('hadm_id').size().items():
            hadm = int(hadm)
            totals[hadm] = totals.get(hadm, 0) + int(count)
        for hadm, values in chunk.groupby('hadm_id')['drug_clean']:
            distinct.setdefault(int(hadm), set()).update(x for x in values if x)
        for name, pattern in MED_GROUPS.items():
            matches = chunk.drug_clean.str.contains(pattern, case=False, regex=True, na=False)
            groups[name].update(int(x) for x in chunk.loc[matches,'hadm_id'].unique())
        if i % 25 == 0:
            print(f'Prescription chunks {i}; rows scanned {scanned:,}; valid orders {valid:,}', flush=True)
    out = cohort[['hadm_id']].copy()
    out['rx_order_count_24h'] = out.hadm_id.map(totals).fillna(0).astype('int32')
    out['rx_distinct_drug_names_24h'] = out.hadm_id.map(
        {k: len(v) for k,v in distinct.items()}).fillna(0).astype('int32')
    out['has_rx_order_24h'] = (out.rx_order_count_24h > 0).astype('int8')
    for name, ids_set in groups.items():
        out['rx_' + name + '_order_24h'] = out.hadm_id.isin(ids_set).astype('int8')
    return out, {'prescription_rows_scanned':scanned,
                 'prescription_rows_matching_cohort':selected,
                 'prescription_orders_with_valid_starttime':valid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--phase1', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--chunk-size', type=int, default=500000)
    args = ap.parse_args()
    base = Path(args.base)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(Path(args.phase1) / 'phase1_cohort_audit.parquet',
                             columns=['subject_id','hadm_id','admittime','landmark_time'])
    for c in ['admittime','landmark_time']:
        cohort[c] = pd.to_datetime(cohort[c], errors='coerce')
    if cohort.hadm_id.isna().any() or cohort.hadm_id.duplicated().any():
        raise ValueError('Invalid or duplicate hadm_id in Phase 1 audit')
    if cohort[['subject_id','admittime','landmark_time']].isna().any().any():
        raise ValueError('Missing essential cohort identifiers/timestamps')
    if not (cohort.landmark_time > cohort.admittime).all():
        raise ValueError('Invalid landmark')
    print(f'Eligible admissions: {len(cohort):,}', flush=True)
    history = build_prior_conditions(base, cohort)
    print('Prior-diagnosis history extracted', flush=True)
    orders, stats = build_orders(base, cohort, args.chunk_size)
    result = cohort[['subject_id','hadm_id']].merge(history.drop(columns='subject_id'),
                    on='hadm_id', how='left', validate='one_to_one')
    result = result.merge(orders, on='hadm_id', how='left', validate='one_to_one')
    if len(result) != len(cohort) or result.hadm_id.duplicated().any():
        raise AssertionError('Cohort size changed')
    feature_cols = [x for x in result.columns if x not in ('subject_id','hadm_id')]
    if result[feature_cols].isna().any().any():
        raise AssertionError('Unexpected missing feature values')
    dest = outdir / 'phase9b_medication_comorbidity_features.parquet'
    result.to_parquet(dest, index=False)
    report = {'eligible_admissions':len(cohort), 'feature_count':len(feature_cols),
              'feature_columns':feature_cols,
              'admissions_with_rx_orders':int(result.has_rx_order_24h.sum()),
              **stats,
              'limitations': [
                'Prescription starttime indicates order timing, not administration or exposure.',
                'Drug groups use approximate name matching, not validated RxNorm/ATC mappings.',
                'Historical diagnosis flags derive only from prior admissions discharged before index admission.',
                'No diagnoses from the index admission are used as predictors.',
                'Missing historical evidence is encoded as zero, not proof of disease absence.',
                'Date-shifted MIMIC-IV timestamps support within-patient chronological comparisons.'
              ]}
    (outdir/'phase9b_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('Saved:', dest)

if __name__ == '__main__':
    main()
