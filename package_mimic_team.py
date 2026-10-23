#!/usr/bin/env python3
"""Package compact MIMIC-IV-derived research files for an authorized team.

Copies (never moves) selected Phase 18/18B/18C/19B outputs and code, removes
reserved evaluation rows from shared development Parquets, and creates a ZIP.
Do not distribute outside a DUA/institution-approved secure environment.
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MODULE_B = ['history', 'history_diagnoses', 'prescriptions', 'emar', 'orders', 'microbiology']
MODULE_C = ['history_diagnoses', 'prescriptions', 'emar', 'orders']
CODE_NAMES = ['phase18_unified_flow', 'phase18b_feature_store', 'phase18c_clinical_concepts',
              'staged_labs', 'phase19b_multimodel', 'phase21_competing_risk',
              'phase22b_labs_retrain', 'phase23c_hospitalwide_binary',
              'phase26_shap_engineering', 'phase23_renal_benchmark',
              'phase23b_binary_renal_neural', 'phase24_feature_reduction',
              'phase25_feature_audit', 'phase27_renal_tuning', 'phase28_optuna_trees']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='~/Downloads/MIMIC_IV', help='Directory containing phase*_outputs')
    p.add_argument('--destination', default='~/Downloads/MIMIC_IV_TEAM_PACKAGE')
    p.add_argument('--code-dir', default='~/Downloads')
    p.add_argument('--include-checkpoints', action='store_true', help='Include Phase21/23C model files (may be large)')
    p.add_argument('--include-validation-predictions', action='store_true', help='Include saved development validation predictions')
    p.add_argument('--overwrite', action='store_true', help='Replace existing destination directory and ZIP')
    args = p.parse_args()
    root = Path(args.root).expanduser().resolve()
    dest = Path(args.destination).expanduser().resolve()
    archive = dest.with_suffix('.zip')
    if not root.is_dir():
        p.error(f'Root directory not found: {root}')
    if dest == root or root in dest.parents:
        p.error('Destination must be OUTSIDE the MIMIC_IV root to prevent recursive packaging')
    if dest.exists() or archive.exists():
        if not args.overwrite:
            p.error(f'Destination or ZIP already exists: {dest} / {archive}; use --overwrite')
        if dest.exists():
            if not dest.is_dir(): p.error('Destination exists and is not a directory')
            shutil.rmtree(dest)
        archive.unlink(missing_ok=True)
    dest.mkdir(parents=True)
    copied = []
    missing = []

    def write_df(source, relative, *, filter_ids=None, split_filter=False):
        source = Path(source)
        if not source.is_file():
            missing.append(str(source))
            return
        data = pd.read_parquet(source)
        before = len(data)
        if split_filter:
            if 'split' not in data.columns:
                raise ValueError(f'Cannot filter evaluation split in {source}: split missing')
            data = data.loc[data['split'].isin(['train', 'early_stop', 'validation'])].copy()
        elif filter_ids is not None:
            if 'hadm_id' not in data.columns:
                raise ValueError(f'Cannot filter by hadm_id in {source}')
            data = data.loc[data.hadm_id.isin(filter_ids)].copy()
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(target, index=False, compression='zstd')
        copied.append({'file': relative, 'rows': len(data), 'source_rows': before,
                       'bytes': target.stat().st_size, 'sha256': sha256(target)})
        print(f'  {relative}: {before:,} -> {len(data):,} rows; {target.stat().st_size / 1048576:.1f} MiB', flush=True)

    phase18 = root / 'phase18_outputs'
    base_features = phase18 / 'prelandmark_features.parquet'
    events = phase18 / 'first_events.parquet'
    if not base_features.exists() or not events.exists():
        p.error(f'Required Phase18 files missing under {phase18}')
    print('Reading Phase18 split IDs...', flush=True)
    key = pd.read_parquet(base_features, columns=['hadm_id', 'subject_id', 'split'])
    if key.hadm_id.duplicated().any() or key.groupby('subject_id').split.nunique().gt(1).any():
        raise ValueError('Duplicate admission or patient-disjoint split violation')
    key = key.loc[key.split.isin(['train', 'early_stop', 'validation'])].copy()
    ids = set(key.hadm_id)
    print(f'Packaging {len(ids):,} development admissions; evaluation split excluded.', flush=True)

    print('Phase18...', flush=True)
    write_df(base_features, 'data/phase18/prelandmark_features.parquet', split_filter=True)
    write_df(events, 'data/phase18/first_events.parquet', filter_ids=ids)

    print('Phase18B and Phase18C...', flush=True)
    for folder, names in [('phase18b_outputs', MODULE_B), ('phase18c_outputs', MODULE_C)]:
        for name in names:
            write_df(root / folder / f'{name}.parquet', f'data/{folder}/{name}.parquet', filter_ids=ids)

    print('Phase19B cached labs...', flush=True)
    write_df(root / 'phase19b_outputs/lab_features.parquet', 'data/phase19b_outputs/lab_features.parquet', filter_ids=ids)
    lab_selection = root / 'phase19b_outputs/lab_selection.json'
    if lab_selection.is_file():
        target = dest / 'metadata/lab_selection.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lab_selection, target)
        copied.append({'file': str(target.relative_to(dest)), 'bytes': target.stat().st_size, 'sha256': sha256(target)})
    else:
        missing.append(str(lab_selection))

    print('Copying reproducibility scripts...', flush=True)
    code_dir = Path(args.code_dir).expanduser().resolve()
    for name in CODE_NAMES:
        candidates = sorted(code_dir.glob(name + '*.py'), key=lambda x: (len(x.name), x.name))
        if not candidates:
            missing.append(f'code: {name}*.py')
            continue
        source = candidates[0]
        target = dest / 'code' / (name + '.py')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append({'file': str(target.relative_to(dest)), 'bytes': target.stat().st_size, 'sha256': sha256(target)})

    if args.include_checkpoints or args.include_validation_predictions:
        for phase in ['phase21_outputs', 'phase23c_outputs']:
            source_dir = root / phase
            if not source_dir.is_dir():
                missing.append(str(source_dir))
                continue
            for source in sorted(source_dir.iterdir()):
                if not source.is_file():
                    continue
                if source.suffix in ('.pt', '.joblib') and not args.include_checkpoints:
                    continue
                if source.name == 'validation_predictions.parquet' and not args.include_validation_predictions:
                    continue
                if source.suffix not in ('.pt', '.joblib', '.json', '.parquet'):
                    continue
                target = dest / 'models' / phase / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.suffix == '.parquet':
                    write_df(source, str(target.relative_to(dest)), filter_ids=ids)
                else:
                    shutil.copy2(source, target)
                    copied.append({'file': str(target.relative_to(dest)), 'bytes': target.stat().st_size, 'sha256': sha256(target)})

    readme = '''# MIMIC-IV authorized-team development package\n\nThis package contains MIMIC-IV-derived patient-level data. Access and storage\nare restricted by the applicable PhysioNet DUA and institutional requirements.\nShare only in an approved, access-controlled research environment with\nindividually authorized team members. Do not upload this ZIP to public GitHub.\n\n## Contents\n- data/phase18: admission-level baseline predictors and outcome/event records\n- data/phase18b_outputs, data/phase18c_outputs: admission-keyed feature modules\n- data/phase19b_outputs: selected aggregated laboratory features\n- metadata/lab_selection.json: laboratory feature name/unit mapping (when available)\n- code/: reproduction scripts available from the local Downloads folder\n- models/: optional checkpoints and validation predictions\n\n## Development-only access\nThe reserved `evaluation` split was removed from the packaged patient-level\ndata. The package includes train, early_stop, and validation admissions only.\nUse the original `split` field and preserve patient-disjoint assignments.\nDo not fit preprocessing on validation or early_stop rows.\n\n## Renal subgroup\nConstruct the renal-history eligibility indicator by joining the Phase18B\n`history_diagnoses` and Phase18C `history_diagnoses` tables on hadm_id, and\nchecking whether b.hist_dx_prior_renal, c.hist_renal_aki or c.hist_renal_ckd\nis greater than zero. Missing historical records do not prove disease absence.\n\n## Caveats\nThe full feature policy includes prescriptions, eMAR, orders and microbiology\nwith unresolved availability-time concerns. Historical diagnosis and laboratory\ncache provenance are not independently certified. This is an exploratory\nresearch package, not a clinically validated or deployment-ready dataset.\n\nOriginal MIMIC-IV raw CSV files and staged raw lab chunks are not included.\nTo engineer new raw-event temporal features, access to the source database\nthrough an authorized environment is still required.\n'''
    (dest / 'README.md').write_text(readme)
    (dest / 'requirements.txt').write_text('numpy\npandas\npyarrow\nscikit-learn\nscipy\ntorch\njoblib\nlightgbm\ncatboost\nxgboost\noptuna\nduckdb\n')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'development only; evaluation excluded',
                'development_admissions': len(ids),
                'files': copied, 'missing_optional_files': missing,
                'warning': 'Credentialed MIMIC-IV-derived patient data; approved sharing/storage only.'}
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Creating ZIP...', flush=True)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5, allowZip64=True) as z:
        for source in sorted(dest.rglob('*')):
            if source.is_file():
                z.write(source, source.relative_to(dest.parent))
    print(f'\nPACKAGE: {dest}\nZIP: {archive}\nZIP SIZE: {archive.stat().st_size / 1048576:.1f} MiB')
    print('Missing optional files:', len(missing))
    for item in missing: print('  -', item)
    print('Original files were not modified.')


if __name__ == '__main__':
    main()
