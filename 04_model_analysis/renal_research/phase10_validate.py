#!/usr/bin/env python3
"""Phase 10: paired evaluation, patient bootstrap, and SHAP for MIMIC-IV research.
No model retraining. Do not use evaluation-set thresholds as prospective operating thresholds.
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

SETS = ['A_baseline','B_history','C_medications','D_comorbidities','E_all']
MODELS = {'lightgbm':'lightgbm_probability','catboost':'catboost_probability','blend':'blend_probability'}

def score(y,p):
    return {'auroc':float(roc_auc_score(y,p)), 'auprc':float(average_precision_score(y,p))}

def recall_at_empirical_specificity(y,p,target):
    # Descriptive, in-sample operating point ONLY. No clinical threshold selection.
    fpr,tpr,thresholds=roc_curve(y,p,drop_intermediate=False)
    ok=np.flatnonzero(fpr<=1-target+1e-12)
    i=ok[np.argmax(tpr[ok])]
    return {'recall':float(tpr[i]),'specificity':float(1-fpr[i]),'threshold_exploratory':float(thresholds[i])}

def load_predictions(folder):
    result={}
    for name in SETS:
        p=folder/f'{name}_evaluation_predictions.parquet'
        d=pd.read_parquet(p)
        required=['subject_id','hadm_id','outcome',*MODELS.values()]
        if not set(required).issubset(d.columns):
            raise ValueError(f'{p} missing {set(required)-set(d.columns)}')
        if d.hadm_id.isna().any() or d.hadm_id.duplicated().any():
            raise ValueError(f'Invalid admission IDs in {p}')
        result[name]=d[required].copy()
    ref=result['A_baseline'].set_index('hadm_id').sort_index()
    for name,d in result.items():
        z=d.set_index('hadm_id').sort_index()
        if not ref.index.equals(z.index): raise ValueError(f'Admissions differ: {name}')
        if not np.array_equal(ref.outcome.to_numpy(),z.outcome.to_numpy()): raise ValueError(f'Labels differ: {name}')
        if not np.array_equal(ref.subject_id.to_numpy(),z.subject_id.to_numpy()): raise ValueError(f'Patient IDs differ: {name}')
        result[name]=z
    return result

def paired_bootstrap(data, n_boot, seed):
    # Resample patients, retaining ALL their admissions; same draws for every model.
    base=data['A_baseline']
    patient_groups=base.reset_index().groupby('subject_id',sort=False).indices
    patients=np.array(list(patient_groups.keys()))
    indices=[np.asarray(patient_groups[p],dtype=np.int64) for p in patients]
    rng=np.random.default_rng(seed)
    y=base.outcome.to_numpy(dtype=int)
    rows=[]
    for b in range(n_boot):
        sampled=rng.integers(0,len(indices),len(indices))
        ix=np.concatenate([indices[i] for i in sampled])
        yy=y[ix]
        if np.unique(yy).size<2: continue
        for model,col in MODELS.items():
            baseline=data['A_baseline'][col].to_numpy()[ix]
            expanded=data['E_all'][col].to_numpy()[ix]
            sb=score(yy,baseline); se=score(yy,expanded)
            rows.append({'bootstrap':b,'model':model,
                         'delta_auroc':se['auroc']-sb['auroc'],
                         'delta_auprc':se['auprc']-sb['auprc']})
    return pd.DataFrame(rows)

def reconstruct_expanded_features(args, bundle, target_ids):
    # Rebuild the exact Phase 9C source columns, but validate model compatibility
    # before SHAP. We never use outcome information as a feature.
    base=pd.read_parquet(Path(args.phase1)/'phase1_landmark_predictors.parquet')
    labs=pd.read_parquet(Path(args.phase2)/'phase2_temporal_lab_features.parquet')
    hist=pd.read_parquet(Path(args.phase9)/'phase9a_history_omr_features.parquet')
    med=pd.read_parquet(Path(args.phase9)/'phase9b_medication_comorbidity_features.parquet')
    frames=[base,labs,hist,med]
    for i,d in enumerate(frames):
        if 'hadm_id' not in d: raise ValueError(f'Feature source {i} has no hadm_id')
        if d.hadm_id.duplicated().any(): raise ValueError(f'Feature source {i} duplicates hadm_id')
    x=base.copy()
    for d in [labs,hist,med]:
        drop=[c for c in ['subject_id','landmark_time','age','gender','admission_type','admission_location'] if c in d.columns]
        x=x.merge(d.drop(columns=drop),on='hadm_id',how='left',validate='one_to_one')
    x=x.set_index('hadm_id').loc[target_ids]
    cols=bundle['columns']; cats=bundle['categorical']
    absent=[c for c in cols if c not in x.columns]
    if absent: raise ValueError(f'Missing {len(absent)} saved model features: {absent[:15]}')
    x=x[cols].copy()
    for c in cats: x[c]=x[c].astype('string').fillna('__MISSING__').astype('category')
    for c in x.columns:
        if c not in cats: x[c]=pd.to_numeric(x[c],errors='raise').replace([np.inf,-np.inf],np.nan)
    return x

def shap_analysis(args, output, data):
    import joblib
    import shap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bundle=joblib.load(Path(args.phase9c)/'E_all_models.joblib')
    rng=np.random.default_rng(args.seed)
    all_ids=data['E_all'].index.to_numpy()
    ids=rng.choice(all_ids,size=min(args.shap_sample,len(all_ids)),replace=False)
    x=reconstruct_expanded_features(args,bundle,ids)
    cats=bundle['categorical']
    for model_name in ['lightgbm','catboost']:
        model=bundle[model_name]
        z=x.copy()
        if model_name=='catboost':
            for c in cats: z[c]=z[c].astype('string').fillna('__MISSING__').astype(str)
        explainer=shap.TreeExplainer(model)
        vals=explainer.shap_values(z)
        if isinstance(vals,list): vals=vals[-1]
        vals=np.asarray(vals)
        if vals.ndim==3: vals=vals[:,:,-1]
        if vals.shape != (len(z),len(z.columns)):
            raise ValueError(f'Unexpected SHAP shape for {model_name}: {vals.shape}')
        imp=pd.DataFrame({'feature':z.columns,'mean_abs_shap':np.abs(vals).mean(axis=0)})
        imp=imp.sort_values('mean_abs_shap',ascending=False)
        imp.to_csv(output/f'{model_name}_shap_importance.csv',index=False)
        top=imp.head(25).iloc[::-1]
        fig,ax=plt.subplots(figsize=(11,9))
        ax.barh(top.feature,top.mean_abs_shap)
        ax.set_xlabel('Mean absolute SHAP contribution (model output units)')
        ax.set_title(f'{model_name}: global feature importance')
        fig.tight_layout(); fig.savefig(output/f'{model_name}_shap_importance.png',dpi=180); plt.close(fig)
        print(f'{model_name} SHAP top 10: {imp.feature.head(10).tolist()}')
    (output/'shap_note.txt').write_text('Model-specific SHAP values are not additive explanations of the blended probability. SHAP is associational, not causal.\n')

def main(args):
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    data=load_predictions(Path(args.phase9c))
    base=data['A_baseline']; y=base.outcome.to_numpy(dtype=int)
    if set(np.unique(y))!={0,1}: raise ValueError('Binary labels required')
    print(f'Paired evaluation: {len(y)} admissions, {base.subject_id.nunique()} patients, {y.sum()} positives')
    rows=[]
    for name,d in data.items():
        for model,col in MODELS.items():
            p=d[col].to_numpy(dtype=float)
            if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError(f'Invalid probabilities: {name}/{model}')
            m=score(y,p)
            op80=recall_at_empirical_specificity(y,p,.8)
            op90=recall_at_empirical_specificity(y,p,.9)
            rows.append({'feature_set':name,'model':model,**m,
                         'recall_at_80pct_specificity_exploratory':op80['recall'],
                         'recall_at_90pct_specificity_exploratory':op90['recall']})
    metrics=pd.DataFrame(rows)
    metrics.to_csv(out/'paired_evaluation_metrics.csv',index=False)
    print('\nPaired AUROC/AUPRC:')
    print(metrics[['feature_set','model','auroc','auprc']].to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    boot=paired_bootstrap(data,args.bootstrap,args.seed)
    if boot.empty: raise ValueError('No valid bootstrap replicates')
    summary=[]
    for model,g in boot.groupby('model'):
        for metric in ['delta_auroc','delta_auprc']:
            point=metrics.query('feature_set == "E_all" and model == @model')[metric.replace('delta_','')].iloc[0]-metrics.query('feature_set == "A_baseline" and model == @model')[metric.replace('delta_','')].iloc[0]
            lo,hi=np.quantile(g[metric],[.025,.975])
            summary.append({'model':model,'metric':metric,'observed_difference':point,'ci_lower':lo,'ci_upper':hi,'bootstrap_replicates':len(g)})
    pd.DataFrame(summary).to_csv(out/'paired_bootstrap_ci.csv',index=False)
    print('\nPaired patient-bootstrap differences (E_all minus A_baseline):')
    print(pd.DataFrame(summary).to_string(index=False,float_format=lambda v:f'{v:.5f}'))
    if not args.skip_shap:
        shap_analysis(args,out,data)
    report={'n_admissions':len(y),'n_patients':int(base.subject_id.nunique()),'positives':int(y.sum()),
            'bootstrap_requested':args.bootstrap,'notes':[
                'All five feature sets evaluated on identical admissions and labels.',
                'Bootstrap resamples patients, retaining repeated admissions and using paired draws.',
                'Recall at fixed specificity is computed on evaluation outcomes and is exploratory, not a validation-selected operating point.',
                'Previous test-set inspection and architecture searches mean these are internal exploratory comparisons.',
                'Medication orders are not confirmed administrations; retrospective renal diagnosis cohort is not prospectively defined.'
            ]}
    (out/'phase10_report.json').write_text(json.dumps(report,indent=2))
    print('\nSaved to',out)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--phase9c',required=True)
    p.add_argument('--phase1',required=True)
    p.add_argument('--phase2',required=True)
    p.add_argument('--phase9',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--shap-sample',type=int,default=300)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--skip-shap',action='store_true')
    main(p.parse_args())
