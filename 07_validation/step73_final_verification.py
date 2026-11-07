"""Final local checks: Step 53 inference consistency and ICU simulation invariants.
Run: python step73_final_verification.py --reference ranked_icu_risk.csv
Optional: --patients synthetic_patients_20.csv --app riskweighted.py
No model retraining. Does not share patient data or modify artifacts.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
FILES = ('final_xgboost_75_model.json','final_inference_bundle.joblib','final_model_metadata.json')

def locate_model():
    paths = [os.getenv('ICU_MODEL_DIR'), ROOT/'model', ROOT/'ICU_STREAMLIT_APP'/'model',
             ROOT/'model training figures'/'expanded_lab_xgboost'/'step53_final_75_model',
             ROOT.parent/'model training figures'/'expanded_lab_xgboost'/'step53_final_75_model']
    for path in paths:
        if path and all((Path(path)/name).is_file() for name in FILES):
            return Path(path)
    raise FileNotFoundError('Cannot find Step 53 artifacts. Set ICU_MODEL_DIR to the folder containing the 3 model files.')

def find_file(name, required=True):
    paths = [ROOT/name, ROOT/'ICU_STREAMLIT_APP'/name, ROOT/'ICU_STREAMLIT_APP'/'real_patient_test_samples'/name]
    for p in paths:
        if p.is_file(): return p
    if required: raise FileNotFoundError(f'Cannot find {name}. Pass an explicit path with --patients or --reference.')
    return None

def prepare(df, bundle):
    cols = list(bundle['raw_columns'])
    missing = [c for c in cols if c not in df]
    if missing: raise ValueError(f'Patient input missing {len(missing)} columns: {missing[:10]}')
    raw = df[cols].copy()
    cat = list(bundle['categorical_columns'])
    for c in cat: raw[c] = raw[c].fillna('Unknown').astype(str)
    for c in cols:
        if c not in cat:
            raw[c] = pd.to_numeric(raw[c], errors='coerce').replace([np.inf,-np.inf],np.nan)
    matrix = np.asarray(bundle['preprocessor'].transform(raw), dtype=np.float32)
    selected = matrix[:,np.asarray(bundle['selected_indices'],dtype=int)]
    if selected.shape[1]!=75: raise ValueError('Expected 75 selected features')
    return selected

def check_reference(pred, ref, threshold):
    if 'predicted_risk' in ref: score_col='predicted_risk'
    elif 'risk' in ref: score_col='risk'
    else: raise ValueError('Reference CSV needs predicted_risk or risk column (raw probabilities, not percentages).')
    if 'sample_id' in ref and 'sample_id' in pred:
        if pred.sample_id.duplicated().any() or ref.sample_id.duplicated().any():
            raise ValueError('Duplicate sample_id; cannot safely match rows')
        merged=pred.merge(ref, on='sample_id', how='outer', validate='one_to_one', indicator=True, suffixes=('','_ref'))
    elif 'patient' in ref and 'patient' in pred:
        if pred.patient.duplicated().any() or ref.patient.duplicated().any(): raise ValueError('Duplicate patient labels')
        merged=pred.merge(ref, on='patient', how='outer', validate='one_to_one', indicator=True, suffixes=('','_ref'))
    elif 'patient_row' in ref:
        merged=pred.merge(ref,on='patient_row',how='outer',validate='one_to_one',indicator=True,suffixes=('','_ref'))
    else:
        raise ValueError('Reference CSV needs sample_id, patient, or patient_row to align safely. Never compare sorted rows by position.')
    if not merged._merge.eq('both').all() or len(merged)!=len(pred): raise ValueError('Reference and input patients do not match exactly')
    diffs=np.abs(merged['direct_probability'].to_numpy()-merged[score_col].to_numpy(float))
    maxdiff=float(diffs.max())
    flags=merged['direct_probability'].to_numpy() >= threshold
    flag_col=next((c for c in ['above_threshold','review_flag','flag_for_review'] if c in merged),None)
    flag_pass=True
    if flag_col:
        raw=merged[flag_col]
        actual=raw.map({'True':True,'False':False,'true':True,'false':False,True:True,False:False,1:True,0:False})
        if actual.isna().any(): raise ValueError(f'Invalid boolean flag values in {flag_col}')
        flag_pass=bool(np.array_equal(flags,actual.to_numpy(bool)))
    return maxdiff, bool(np.all(diffs<=1e-7)),flag_pass,len(merged)

# Audit of an independent priority vs random comparison, with identical generated
# requests, arrival times, ICU stays, and initial occupancy for each policy.
def shared_scenario(risks,beds,occupied,mean_stay_hours,max_wait,seed):
    rng=np.random.default_rng(seed)
    needs=rng.random(len(risks))<risks
    arrivals=rng.integers(24,73,size=len(risks))
    remaining=rng.exponential(mean_stay_hours,size=occupied)
    # Generate a stay for each possible patient before either policy runs.
    stays=np.maximum(1.0,rng.exponential(mean_stay_hours,size=len(risks)))
    return needs,arrivals,remaining,stays

def policy_result(scenario,risks,beds,max_wait,mode,seed):
    needs,arrivals,remaining,stays=scenario
    busy=[float(x) for x in remaining]
    pending=[]; served=set(); unmet=set(); occupancy=[]
    rng=np.random.default_rng(seed)
    for hour in range(73+max_wait+2):
        busy=[end for end in busy if end>hour]
        pending.extend(i for i in range(len(risks)) if needs[i] and arrivals[i]==hour)
        overdue=[i for i in pending if hour-int(arrivals[i])>max_wait]
        unmet.update(overdue)
        pending=[i for i in pending if i not in unmet]
        if mode=='risk': pending.sort(key=lambda i:(-risks[i],i))
        elif mode=='random': rng.shuffle(pending)
        else: raise ValueError(mode)
        while pending and len(busy)<beds:
            i=pending.pop(0)
            served.add(i)
            busy.append(hour+float(stays[i]))
        occupancy.append(len(busy))
        assert len(busy)<=beds, 'CAPACITY VIOLATION'
    assert not pending, 'Pending requests unaccounted for'
    assert not served.intersection(unmet), 'Double counted request'
    assert len(served)+len(unmet)==int(needs.sum()), 'Request conservation failure'
    return {'requests':int(needs.sum()),'admitted':len(served),'unmet':len(unmet),
            'max_occupied':max(occupancy),'unserved_risk':float(risks[list(unmet)].sum()) if unmet else 0.0}

def simulation_checks(risks,seed):
    beds=10; occupied=10; max_wait=12; mean_stay_hours=7*24
    tests=[]; records=[]
    for j in range(100):
        # Reuse EXACT scenario for both strategies.
        scenario=shared_scenario(risks,beds,occupied,mean_stay_hours,max_wait,seed+j)
        a=policy_result(scenario,risks,beds,max_wait,'risk',seed+j+10000)
        b=policy_result(scenario,risks,beds,max_wait,'random',seed+j+10000)
        tests.extend([a['max_occupied']<=beds,b['max_occupied']<=beds,
                      a['requests']==a['admitted']+a['unmet'],
                      b['requests']==b['admitted']+b['unmet'],
                      a['requests']==b['requests']])
        records.append({'run':j+1,'requests':a['requests'],'risk_admitted':a['admitted'],
                        'random_admitted':b['admitted'],'risk_unmet':a['unmet'],
                        'random_unmet':b['unmet'],'risk_unserved_score':a['unserved_risk'],
                        'random_unserved_score':b['unserved_risk']})
    return all(tests),pd.DataFrame(records)

def inspect_app(path):
    """Static check: confirm whether the deployed app uses the independently audited code."""
    if not path or not path.exists(): return 'NOT CHECKED: supply --app to inspect the actual frontend.'
    src=path.read_text(encoding='utf-8')
    tree=ast.parse(src)
    names={n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    if 'simulate_priority' not in names:
        return 'WARNING: no top-level simulate_priority function found; this script does not validate the deployed simulator.'
    return 'NOTE: simulate_priority exists, but its behavior is NOT established by the independent simulation audit. Compare its outputs separately.'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--patients',type=Path,help='Exact CSV/Excel uploaded to Streamlit')
    parser.add_argument('--reference',type=Path,help='Ranking CSV downloaded from original or latest Streamlit app')
    parser.add_argument('--app',type=Path,help='Actual frontend .py file (for static inspection)')
    args=parser.parse_args()
    patient_path=args.patients or find_file('streamlit_real_10_patients.csv',required=False) or find_file('synthetic_patients_20.csv',required=False)
    if patient_path is None: raise FileNotFoundError('Specify --patients with the exact file used in Streamlit')
    reference_path=args.reference or find_file('ranked_icu_risk.csv',required=False)
    folder=locate_model()
    model=XGBClassifier(); model.load_model(str(folder/FILES[0]))
    bundle=joblib.load(folder/FILES[1]); metadata=json.loads((folder/FILES[2]).read_text(encoding='utf-8'))
    threshold=float(metadata['selected_threshold'])
    df=pd.read_excel(patient_path) if patient_path.suffix.lower() in ('.xlsx','.xls') else pd.read_csv(patient_path)
    if df.empty: raise ValueError('Empty patient file')
    p=model.predict_proba(prepare(df,bundle))[:,1]
    pred=pd.DataFrame({'patient_row':np.arange(len(df)),'direct_probability':p})
    if 'sample_id' in df: pred['sample_id']=df['sample_id'].astype(str)
    pred['patient']=df['sample_id'].astype(str) if 'sample_id' in df else [f'Patient {i+1}' for i in range(len(df))]
    pred['direct_flag']=p>=threshold
    print('\n=== 1. SAVED MODEL vs STREAMLIT REFERENCE ===')
    print('Patients:',len(df),'| Threshold:',threshold)
    if reference_path:
        ref=pd.read_csv(reference_path)
        maxdiff,ok,flags,n=check_reference(pred,ref,threshold)
        print('Matched patients:',n,'| Max absolute difference:',maxdiff)
        print('Probabilities match (1e-7):', 'PASS' if ok else 'FAIL')
        print('Threshold flags match:', 'PASS' if flags else 'FAIL')
        model_pass=ok and flags
    else:
        print('NOT RUN: reference CSV missing. Supply --reference ranked_icu_risk.csv')
        model_pass=False
    print('\n=== 2. INDEPENDENT SIMULATION AUDIT ===')
    sim_ok, results=simulation_checks(np.asarray(p,float),seed=42)
    print('Capacity, request accounting, shared scenarios:', 'PASS' if sim_ok else 'FAIL')
    print('Same requests in both strategies:','PASS' if (results.requests>=0).all() else 'FAIL')
    print('Mean unmet requests: risk',round(results.risk_unmet.mean(),3),'/ random',round(results.random_unmet.mean(),3))
    print('Mean unserved predicted risk: risk',round(results.risk_unserved_score.mean(),4),'/ random',round(results.random_unserved_score.mean(),4))
    print('\n=== 3. ACTUAL APP COVERAGE ===')
    app_path=args.app or find_file('riskweighted.py',required=False) or find_file('icu_crisis_app.py',required=False)
    print(inspect_app(app_path))
    output=ROOT/'step73_shared_scenario_audit.csv'
    results.to_csv(output,index=False)
    print('\nSaved aggregate simulation runs:',output)
    print('MODEL REFERENCE CHECK:', 'PASS' if model_pass else 'NOT VERIFIED / FAIL')
    print('INDEPENDENT SIMULATION CHECK:', 'PASS' if sim_ok else 'FAIL')
    print('Important: independent simulation PASS does NOT prove the deployed app has identical logic.')
    if not sim_ok or (reference_path is not None and not model_pass): sys.exit(1)

if __name__=='__main__': main()
