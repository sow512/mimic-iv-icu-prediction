#!/usr/bin/env python3
"""Renal-history subgroup retraining of the supplied Phase 21 model.

Eligibility is determined ONLY from prior-admission diagnosis aggregates, never
current-admission discharge codes or outcome labels. The model otherwise uses
Phase 21's original feature set, architecture, and training routine.

This is a renal-history subgroup, NOT all AKI/renal dysfunction presentations.
The upstream medication/order/lab availability limitations remain.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ORIGINAL = Path(__file__).with_name('phase21_competing_risk.py')
if not ORIGINAL.exists():
    raise SystemExit('Place the uploaded phase21_competing_risk.py alongside this script.')
spec = importlib.util.spec_from_file_location('original_phase21', ORIGINAL)
phase21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase21)
original_loader = phase21.load_data


def renal_loader(args):
    # Filter only after the original loader has performed its patient-split checks.
    data = original_loader(args)
    candidates = ['b_hist_dx_prior_renal', 'c_hist_renal_aki', 'c_hist_renal_ckd']
    available = [col for col in candidates if col in data.columns]
    if not available:
        raise ValueError('No historical renal diagnosis columns found: ' + str(candidates))
    eligible = pd.Series(False, index=data.index)
    for col in available:
        eligible |= pd.to_numeric(data[col], errors='coerce').fillna(0).gt(0)
    selected = data.loc[eligible].copy()
    counts = selected.groupby('split').agg(admissions=('hadm_id', 'size'), patients=('subject_id', 'nunique'), icu_events=('icu_24_72', 'sum'))
    print('RENAL ELIGIBILITY: prior-admission renal diagnoses only; columns:', available, flush=True)
    print(counts.to_string(), flush=True)
    for split in ('train', 'early_stop', 'validation'):
        if split not in counts.index or counts.loc[split, 'icu_events'] < 2:
            raise ValueError(f'Insufficient ICU events in {split} for renal-only training/evaluation')
    if selected.groupby('subject_id').split.nunique().max() > 1:
        raise ValueError('Patient split leakage')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'renal_cohort_manifest.json').write_text(json.dumps({
        'definition': 'Prior-admission ICD-derived renal disease, AKI or CKD',
        'selection_columns': available,
        'split_counts': counts.reset_index().to_dict(orient='records'),
        'warning': 'Retrospective historical ICD availability not independently audited. Full Phase21 feature provenance limitations persist. Original Phase21 interval-boundary behavior is unchanged.'
    }, indent=2))
    return selected

phase21.load_data = renal_loader
if __name__ == '__main__':
    phase21.main()
