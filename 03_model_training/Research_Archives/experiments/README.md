# experiments

Earlier feature ablations, Optuna tuning, no-laboratory benchmarks and combined experiments.

## Contents

- [`administrative_feature_ablation.py`](administrative_feature_ablation.py) — Python experiment script.
- [`benchmark_no_labs.py`](benchmark_no_labs.py) — Python experiment script.
- [`optuna_renal_model_tuning.py`](optuna_renal_model_tuning.py) — Python experiment script.
- [`renal_icu_all_phases.py`](renal_icu_all_phases.py) — Python experiment script.
- [`renal_icu_megabench.py`](renal_icu_megabench.py) — Python experiment script.

## Scope and reproducibility

Experiments may use different cohorts, feature definitions and validation partitions. Compare reported metrics only when these match. MIMIC-IV data is access-controlled and not included; scripts may require authorized local input files. These results are research-only and not validated for clinical use.
