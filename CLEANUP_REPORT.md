# Cleanup report

## Changes
- Deduplicated identical files by SHA-256 (not filename alone).
- Normalized duplicate upload suffixes such as `(2)` from filenames.
- Preserved differing versions with `_variantN` suffixes rather than silently discarding code.
- Separated original renal phases, master variants, hospital-wide phases, dynamic pilot, and packaging tools.
- Preserved original Python source behavior; no untested refactoring of training logic.
- Cleared Jupyter notebook outputs and execution counts.
- Excluded patient-level files, model binaries and data ZIPs.

## Duplicates removed
- 10 byte-identical redundant files.

## Python syntax check
- 0 syntax failures out of 50 scripts.


## Different master versions
- Renal master versions are kept separately when their bytes differ. Verify intended canonical version before consolidating.

## Remaining limitations
- Full runtime integration, clinical leakage audits, and reproducibility against MIMIC-IV were not performed.
- Code references to legacy script names or local paths may need updating after reorganization.
- Source code should be reviewed for credentials and confidential strings before public publication.

## Subsequent team integration
- Moved research Python files into numbered team workflow folders without modifying their source bytes.
- Updated the root README and code inventory to reflect canonical paths.
- Added a static CI check for Python syntax and mapped file existence.
- Full runtime execution and historical notebook path remediation still require an authorized MIMIC-IV environment.

## Descriptive script naming
- Renamed 38 scripts by their purpose and model family, without changing their source bytes.
- The phase-to-filename mapping is retained in `src/MOVE_MAP.md`.
- Imports and dataset-dependent runtime behavior have not been verified end-to-end.
