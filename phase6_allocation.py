#!/usr/bin/env python3
"""Phase 6: retrospective ICU-transfer capture under hypothetical capacity.
This is NOT a clinical allocation or treatment-benefit model.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CAPACITIES = [0.05, 0.10, 0.15, 0.20, 0.30]

def choose_col(df, requested, candidates, purpose):
    if requested:
        if requested not in df.columns:
            raise ValueError(f'{purpose} column {requested!r} not found. Available: {list(df.columns)}')
        return requested
    matches = [c for c in candidates if c in df.columns]
    if len(matches) != 1:
        raise ValueError(f'Cannot unambiguously identify {purpose} column. Candidates present: {matches}. Available: {list(df.columns)}. Pass explicit --{purpose}-col.')
    return matches[0]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase4b', required=True, help='Directory containing phase4b_evaluation_predictions.parquet')
    ap.add_argument('--output', required=True)
    ap.add_argument('--risk-col', default=None)
    ap.add_argument('--outcome-col', default=None)
    ap.add_argument('--random-repeats', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.random_repeats < 1:
        raise ValueError('--random-repeats must be >=1')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    infile = Path(args.phase4b) / 'phase4b_evaluation_predictions.parquet'
    df = pd.read_parquet(infile)
    print('Available columns:', df.columns.tolist())
    risk_col = choose_col(df, args.risk_col, ['uncalibrated_probability', 'raw_ensemble_probability', 'blend_probability', 'uncalibrated', 'raw_probability', 'ensemble_probability', 'probability_uncalibrated'], 'risk')
    outcome_col = choose_col(df, args.outcome_col, ['outcome', 'icu_transfer_24_72h', 'icu_transfer', 'label', 'y_true'], 'outcome')
    if df.empty or df[[risk_col, outcome_col]].isna().any().any():
        raise ValueError('Empty dataset or missing outcome/risk values')
    if 'hadm_id' in df and df.hadm_id.duplicated().any():
        raise ValueError('Duplicate hadm_id: expected one row per admission')
    p = pd.to_numeric(df[risk_col], errors='raise').to_numpy(dtype=float)
    y = pd.to_numeric(df[outcome_col], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Predicted risks must be finite probabilities in [0,1]')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Outcomes must be binary 0/1')
    y = y.astype(int)
    n, positives = len(y), int(y.sum())
    if positives == 0:
        raise ValueError('No positive outcomes in evaluation dataset')
    rng = np.random.default_rng(args.seed)
    # Stable ranking; ties resolved by row order (not by observed outcome).
    order = np.argsort(-p, kind='stable')
    rows = []
    selected = []
    for fraction in CAPACITIES:
        k = max(1, int(np.floor(n * fraction)))
        chosen = order[:k]
        hits = int(y[chosen].sum())
        random_hits = np.empty(args.random_repeats, dtype=int)
        for j in range(args.random_repeats):
            sample = rng.choice(n, size=k, replace=False)
            random_hits[j] = int(y[sample].sum())
        rows.append({
            'capacity_fraction': fraction, 'capacity_admissions': k,
            'total_admissions': n, 'total_eventual_transfers': positives,
            'risk_ranked_captured': hits,
            'risk_ranked_precision_at_capacity': hits/k,
            'risk_ranked_recall_at_capacity': hits/positives,
            'random_expected_captured_analytic': k*positives/n,
            'random_mean_captured_simulated': float(random_hits.mean()),
            'random_captured_p025': float(np.quantile(random_hits, 0.025)),
            'random_captured_p975': float(np.quantile(random_hits, 0.975)),
            'random_mean_precision_at_capacity': float(random_hits.mean()/k),
            'random_mean_recall_at_capacity': float(random_hits.mean()/positives),
            'capture_lift_vs_random_expectation': float(hits/(k*positives/n)),
        })
        for rank, idx in enumerate(chosen, start=1):
            selected.append({'capacity_fraction': fraction, 'rank': rank,
                             'hadm_id': df.iloc[idx]['hadm_id'] if 'hadm_id' in df else int(idx),
                             'predicted_risk': float(p[idx]), 'observed_transfer': int(y[idx])})
    result = pd.DataFrame(rows)
    result.to_csv(out/'phase6_capacity_comparison.csv', index=False)
    pd.DataFrame(selected).to_csv(out/'phase6_ranked_selection_audit.csv', index=False)
    fig, ax = plt.subplots(figsize=(8,5))
    x = result['capacity_fraction'].to_numpy()*100
    ax.plot(x, result['risk_ranked_recall_at_capacity'], marker='o', label='Risk ranking')
    ax.plot(x, result['random_mean_recall_at_capacity'], marker='s', label='Random allocation (mean)')
    ax.fill_between(x, result['random_captured_p025']/positives, result['random_captured_p975']/positives, alpha=.15, label='Random 95% simulation interval')
    ax.set(xlabel='Hypothetical capacity (% of evaluation admissions)', ylabel='Fraction of eventual ICU transfers captured', title='Retrospective event capture by hypothetical capacity')
    ax.set_ylim(0,1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/'phase6_capture_curve.png', dpi=220)
    plt.close(fig)
    report = {'source_file': str(infile), 'risk_column': risk_col, 'outcome_column': outcome_col,
              'evaluation_admissions': n, 'eventual_transfers': positives,
              'prevalence': positives/n, 'random_repeats': args.random_repeats,
              'note': 'This is a retrospective ranking simulation. It does not model real ICU bed occupancy, clinical eligibility, resource needs, competing risks, patient outcomes under alternative treatment, or lives saved. A one-resource equal-cost linear allocation objective reduces to top-k risk ranking.'}
    (out/'phase6_report.json').write_text(json.dumps(report, indent=2))
    print('\n'+result[['capacity_fraction','capacity_admissions','risk_ranked_captured','risk_ranked_precision_at_capacity','risk_ranked_recall_at_capacity','random_expected_captured_analytic','capture_lift_vs_random_expectation']].to_string(index=False))
    print('\nSaved outputs to', out)

if __name__ == '__main__':
    main()
