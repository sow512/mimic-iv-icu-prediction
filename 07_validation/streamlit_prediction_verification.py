"""Step 72: Independently reproduce Streamlit inference and compare Step 54 scores.
Uses existing local MIMIC-IV artifacts only. No model training.
"""
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

PROJECT = Path(os.environ["MIMIC_PROJECT_ROOT"])
BASE = PROJECT / 'model training figures' / 'expanded_lab_xgboost'
MODEL_DIR = BASE / 'step53_final_75_model'
SAMPLE_DIR = PROJECT / 'ICU_STREAMLIT_APP' / 'real_patient_test_samples'
UPLOAD = SAMPLE_DIR / 'streamlit_real_10_patients.csv'
RANKED = SAMPLE_DIR / 'ranked_icu_risk.csv'  # Put the downloaded Streamlit file here
TRUTH = SAMPLE_DIR / 'actual_outcomes_LOCAL_ONLY.csv'
MAPPING = SAMPLE_DIR / 'private_sample_id_mapping_DO_NOT_UPLOAD.csv'
STEP54 = BASE / 'step54_75_evaluation' / 'test_predictions.csv'
OUT = SAMPLE_DIR / 'step72_verification.csv'
TOLERANCE = 1e-5

for p in (UPLOAD, RANKED, TRUTH, MODEL_DIR/'final_inference_bundle.joblib', MODEL_DIR/'final_xgboost_75_model.json', MODEL_DIR/'final_model_metadata.json'):
    if not p.is_file():
        raise FileNotFoundError(f'Missing: {p}')

raw = pd.read_csv(UPLOAD)
ranked = pd.read_csv(RANKED)
truth = pd.read_csv(TRUTH)
if not {'patient_row', 'predicted_risk'}.issubset(ranked.columns):
    raise ValueError('Ranked CSV needs patient_row and predicted_risk columns')
if not {'sample_id','actual_icu_transfer_24_72h'}.issubset(truth.columns):
    raise ValueError('Actual outcomes CSV has unexpected columns')
if not raw.sample_id.is_unique or not truth.sample_id.is_unique or not ranked.patient_row.is_unique:
    raise ValueError('Duplicate sample_id or patient_row detected')
if len(raw) != len(ranked) or set(ranked.patient_row.astype(int)) != set(range(len(raw))):
    raise ValueError('Ranked predictions do not match uploaded rows; download results from the SAME upload')

bundle = joblib.load(MODEL_DIR/'final_inference_bundle.joblib')  # Trusted local artifact only
meta = json.loads((MODEL_DIR/'final_model_metadata.json').read_text(encoding='utf-8'))
model = XGBClassifier()
model.load_model(str(MODEL_DIR/'final_xgboost_75_model.json'))
required = list(bundle['raw_columns'])
missing = sorted(set(required) - set(raw.columns))
if missing:
    raise ValueError(f'Input missing required columns: {missing[:15]}')

# Reproduce app.py preprocessing exactly.
frame = raw[required].copy()
cat = list(bundle['categorical_columns'])
for c in cat:
    frame[c] = frame[c].fillna('Unknown').astype(str)
for c in required:
    if c not in cat:
        frame[c] = pd.to_numeric(frame[c], errors='coerce').replace([np.inf,-np.inf], np.nan)
encoded = np.asarray(bundle['preprocessor'].transform(frame), dtype=np.float32)
indices = np.asarray(bundle['selected_indices'], dtype=int)
X = encoded[:, indices]
if X.shape[1] != 75:
    raise ValueError(f'Expected 75 model inputs, got {X.shape[1]}')
probs = model.predict_proba(X)[:,1]

check = pd.DataFrame({'patient_row': np.arange(len(raw)), 'sample_id': raw.sample_id, 'direct_model_probability': probs})
check = check.merge(ranked[['patient_row','predicted_risk']], on='patient_row', validate='one_to_one')
check = check.merge(truth, on='sample_id', validate='one_to_one')
check['absolute_difference_vs_streamlit'] = (check.direct_model_probability-check.predicted_risk).abs()
threshold = float(meta['selected_threshold'])
check['direct_flag'] = check.direct_model_probability >= threshold
check['streamlit_flag'] = check.predicted_risk >= threshold
check['actual'] = check.actual_icu_transfer_24_72h.astype(int)
check['correct'] = check.streamlit_flag.astype(int) == check.actual

print('\nMODEL vs STREAMLIT')
print('Threshold:', threshold)
print('Maximum absolute probability difference:', check.absolute_difference_vs_streamlit.max())
print('Exact agreement within tolerance:', bool((check.absolute_difference_vs_streamlit <= TOLERANCE).all()))
print('All threshold flags agree:', bool((check.direct_flag == check.streamlit_flag).all()))

# Optional comparison to Step 54 saved predictions, matching actual hadm_id, not row order.
if STEP54.exists() and MAPPING.exists():
    original = pd.read_csv(STEP54)
    mapping = pd.read_csv(MAPPING)
    if {'hadm_id','predicted_probability'}.issubset(original.columns) and {'sample_id','hadm_id'}.issubset(mapping.columns):
        if original.hadm_id.duplicated().any() or mapping.sample_id.duplicated().any():
            raise ValueError('Duplicate identifiers in Step 54 or private mapping')
        check = check.merge(mapping[['sample_id','hadm_id']], on='sample_id', validate='one_to_one')
        check = check.merge(original[['hadm_id','predicted_probability']].rename(columns={'predicted_probability':'step54_probability'}), on='hadm_id', how='left', validate='one_to_one')
        if check.step54_probability.isna().any():
            print('WARNING: Some admissions not found in Step 54 predictions')
        else:
            check['absolute_difference_vs_step54'] = (check.direct_model_probability - check.step54_probability).abs()
            print('\nSTEP 54 vs SAVED MODEL')
            print('Maximum absolute difference:', check.absolute_difference_vs_step54.max())
            print('Agreement within tolerance:', bool((check.absolute_difference_vs_step54 <= TOLERANCE).all()))
            if not (check.absolute_difference_vs_step54 <= TOLERANCE).all():
                print('NOTE: A difference may indicate a different model/preprocessing/version. Investigate before relying on app.')
    else:
        print('\nStep 54 or mapping lacks required columns; skipping Step 54 comparison.')
else:
    print('\nStep 54 comparison skipped (predictions or private ID mapping missing).')

print('\n10-PATIENT CHECK (deliberately balanced; NOT population performance)')
y = check.actual.to_numpy()
yhat = check.streamlit_flag.astype(int).to_numpy()
tn,fp,fn,tp = confusion_matrix(y,yhat,labels=[0,1]).ravel()
print(f'TP={tp}, FP={fp}, TN={tn}, FN={fn}')
print(f'Accuracy={accuracy_score(y,yhat):.3f}; Recall={recall_score(y,yhat,zero_division=0):.3f}; Precision={precision_score(y,yhat,zero_division=0):.3f}; F1={f1_score(y,yhat,zero_division=0):.3f}')
print(check[['sample_id','direct_model_probability','predicted_risk','actual','streamlit_flag','correct']].to_string(index=False))

# Avoid exporting the private admission mapping in the public-facing report.
check.drop(columns=['hadm_id'], errors='ignore').to_csv(OUT,index=False)
print('\nSaved:', OUT)
print('Research validation only; do not share patient-level exports publicly.')
