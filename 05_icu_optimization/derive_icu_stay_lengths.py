"""Build data/icu_stay_hours.csv (ICU stay lengths in hours) from MIMIC-IV icustays.

Usage (PowerShell, from the ICU_STREAMLIT_APP folder):
  python tools/make_stay_lengths.py "D:/mimiciv/icu/icustays.csv.gz"
  python tools/make_stay_lengths.py "D:/mimiciv/icu/icustays.csv.gz" --cohort "D:/.../expanded_lab_xgboost/cohort_audit.csv"

--cohort keeps only ICU stays of admissions in your kidney cohort whose icu_transfer == 1 (needs columns hadm_id, icu_transfer).
Each admission contributes its FIRST ICU stay. The output has one column (los_hours) and no patient identifiers.
Keep it local: it is derived from credentialed MIMIC data, so show summary numbers rather than sharing the file.
"""
import argparse
from pathlib import Path
import pandas as pd

APP = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("icustays")
ap.add_argument("--cohort")
ap.add_argument("--out", default=str(APP / "data" / "icu_stay_hours.csv"))
a = ap.parse_args()

d = pd.read_csv(a.icustays)
need = {"hadm_id", "los"}
if not need <= set(d.columns):
    raise SystemExit(f"icustays file must contain {sorted(need)}; found {list(d.columns)}")
if "intime" in d.columns:
    d = d.sort_values("intime")
d = d.drop_duplicates("hadm_id", keep="first")                       # first ICU stay per admission
if a.cohort:
    c = pd.read_csv(a.cohort, usecols=["hadm_id", "icu_transfer"])
    d = d[d.hadm_id.isin(c.loc[c.icu_transfer == 1, "hadm_id"])]
hours = (d["los"].astype(float) * 24).round(2)
hours = hours[hours > 0].reset_index(drop=True)
if len(hours) < 30:
    raise SystemExit(f"Only {len(hours)} stays found; need at least 30.")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
hours.to_frame("los_hours").to_csv(a.out, index=False)
q = hours.quantile([.25, .5, .75, .9, .99]) / 24
print(f"Wrote {len(hours):,} stays to {a.out}")
print(f"Median {q[.5]:.2f} d | mean {hours.mean()/24:.2f} d | IQR {q[.25]:.2f}-{q[.75]:.2f} d | 90th pct {q[.9]:.2f} d | 99th pct {q[.99]:.2f} d")
