# Uploaded archive completeness audit

Audit date: 2026-11-07. Compared the original user-provided ZIP archives with the repository's `main` branch using **Git blob SHA-1 of exact file bytes**. Renamed source files count as matches if their content hash matches a repository blob. This is a source-presence audit, not a runtime test.

## Code archive: Archive(3).zip

- 52 Python source files.
- **49/52 byte-identical Python files already present in GitHub** (including renamed scripts).
- **Three Python files absent**:
  1. `phase14b_renalrisknet.py` — renal competing-risk neural model; **add to `03_model_training/Competing_Risks/renal_competing_risk_mlp_baseline.py`**.
  2. `inventory_mimic.py` — read-only local MIMIC table/column inventory; **add to `tools/inventory_mimic_tables.py`**.
  3. `build_mail_index.py` — MBOX/email indexing utility; **exclude from this clinical ML repository** (unrelated).
- One `.py.bak` file (`phase18c_clinical_concepts.py.bak`) is a backup, not counted as a Python source file; preserve in the local archive, not the public repository.

## Results archive: Archive(2).zip

- 1,901 archive members: 1,647 Parquet, 84 JSON, 76 CSV, 38 Joblib, 31 PNG, 19 PyTorch `.pt`, 2 TXT, 2 SQLite, 1 DuckDB, and 1 Python file.
- Only **curated aggregate tables and nine figures** were intentionally added to `08_results/`. The other experiment outputs were not exhaustively mirrored to GitHub.
- The sole Python file, `test.py`, computes threshold-dependent classification metrics from a Phase 4B prediction Parquet but uses an absolute personal filesystem path. **Refactor to accept `--predictions` before publishing**.
- Do **not** upload patient-level Parquet data, prediction exports, model artifacts, or unreviewed database files to a public repository.

## Suggested follow-up

1. Add the missing renal competing-risk model and local dataset-inventory tool, preserving original source bytes where possible.
2. Convert `test.py` to a portable CLI and add a synthetic-data smoke test.
3. Validate any newly added scripts with `compileall` and test `--help` interfaces.
4. Link each published aggregate result to its source experiment, cohort, and evaluation split.
5. Do not claim the full ZIP is mirrored; only source completeness and selected aggregate outputs are intended.

**Scope limitation:** The 49 matching scripts were verified by Git blob hashes. This audit does not independently verify that all results in the ZIP are represented, that scripts run end-to-end, or that historical model outputs can be reproduced.
