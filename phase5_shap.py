#!/usr/bin/env python3
"""Phase 5: retrospective SHAP explanations for the raw LightGBM/CatBoost ensemble."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

SEED=42

def main(a):
    import shap
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    bundle=joblib.load(Path(a.phase4b)/'phase4b_models.joblib')
    cols=bundle['columns']; cats=bundle['categorical']; w=float(bundle['blend_lightgbm_weight'])
    base=pd.read_parquet(Path(a.phase1)/'phase1_landmark_predictors.parquet')
    features=pd.read_parquet(Path(a.phase2)/'phase2_temporal_lab_features.parquet')
    labels=pd.read_parquet(Path(a.phase1)/'phase1_labels.parquet')
    for name,df in [('base',base),('features',features),('labels',labels)]:
        if df.hadm_id.isna().any() or df.hadm_id.duplicated().any(): raise ValueError(f'{name} invalid hadm_id')
    features=features.drop(columns=['subject_id','landmark_time'],errors='ignore')
    # Match the Phase 4B training feature schema; duplicate baseline fields from Phase 2 are ignored.
    duplicates=[c for c in features if c in base.columns and c!='hadm_id']
    features=features.drop(columns=duplicates)
    data=base.merge(features,on='hadm_id',how='left',validate='one_to_one').merge(
        labels[['subject_id','hadm_id','icu_transfer_24_72h']],on=['subject_id','hadm_id'],how='inner',validate='one_to_one')
    if len(data)!=len(base): raise ValueError('Cohort size changed')
    X=data.reindex(columns=cols).copy()
    if X.columns.tolist()!=cols: raise AssertionError('Feature order mismatch')
    if X.columns.duplicated().any(): raise ValueError('Duplicate features')
    for c in cats: X[c]=X[c].astype('string').fillna('__MISSING__').astype('category')
    for c in X.columns.difference(cats): X[c]=pd.to_numeric(X[c],errors='raise').replace([np.inf,-np.inf],np.nan)
    # Verify model-compatible feature types and avoid missing columns silently.
    missing=set(cols)-set(data.columns)
    if missing: raise ValueError('Missing model features: '+str(sorted(missing)[:20]))
    # Sample only patients in the saved evaluation predictions (not used to tune models).
    pred=pd.read_parquet(Path(a.phase4b)/'phase4b_evaluation_predictions.parquet')
    if 'hadm_id' not in pred: raise ValueError('Evaluation predictions missing hadm_id')
    indices=data.index[data.hadm_id.isin(pred.hadm_id)]
    if len(indices)!=len(pred): raise ValueError('Evaluation IDs do not match reconstructed cohort')
    rng=np.random.default_rng(SEED)
    indices=rng.choice(indices,size=min(a.sample,len(indices)),replace=False)
    sample=X.loc[indices].copy(); info=data.loc[indices,['subject_id','hadm_id','icu_transfer_24_72h']].copy()
    cb=sample.copy()
    for c in cats: cb[c]=cb[c].astype('string').fillna('__MISSING__').astype(str)
    lgb=bundle['lightgbm']; cat=bundle['catboost']
    p_l=lgb.predict_proba(sample)[:,1]; p_c=cat.predict_proba(cb)[:,1]
    raw=w*p_l+(1-w)*p_c
    # SHAP values in raw margin space: separate model explanations, NOT additively composable
    # in probability space. Use model-specific SHAP, and ensemble probability separately.
    lgb_ex=shap.TreeExplainer(lgb)
    lv=lgb_ex.shap_values(sample)
    if isinstance(lv,list): lv=lv[-1]
    lv=np.asarray(lv)
    if lv.ndim==3: lv=lv[:,:,1]
    if lv.shape!=(len(sample),len(cols)): raise ValueError(f'Unexpected SHAP shape {lv.shape}')
    # Categorical string features can cause problems with CatBoost SHAP APIs; native CatBoost
    # SHAP uses its Pool and supports categorical values.
    from catboost import Pool
    pool=Pool(cb,cat_features=cats)
    cv=np.asarray(cat.get_feature_importance(pool,type='ShapValues'))
    if cv.shape!=(len(sample),len(cols)+1): raise ValueError(f'Unexpected CatBoost SHAP shape {cv.shape}')
    cat_vals=cv[:,:-1]
    def rank(values,name):
        df=pd.DataFrame({'feature':cols,'mean_abs_shap_margin':np.abs(values).mean(axis=0)}).sort_values('mean_abs_shap_margin',ascending=False)
        df.to_csv(out/f'{name}_shap_importance.csv',index=False)
        top=df.head(20).iloc[::-1]
        fig,ax=plt.subplots(figsize=(10,8));ax.barh(top.feature,top.mean_abs_shap_margin)
        ax.set_xlabel('Mean absolute SHAP value (model margin units)');ax.set_title(name+' feature importance')
        fig.tight_layout();fig.savefig(out/f'{name}_importance.png',dpi=180);plt.close(fig)
        return df
    rank(lv,'lightgbm');rank(cat_vals,'catboost')
    # Numerical feature beeswarm (categorical codes deliberately not fabricated).
    numeric=[c for c in cols if c not in cats]
    idx=[cols.index(c) for c in numeric]
    shap.summary_plot(lv[:,idx],sample[numeric],show=False,max_display=20)
    plt.title('LightGBM SHAP summary (margin units)');plt.tight_layout()
    plt.savefig(out/'lightgbm_beeswarm.png',dpi=180,bbox_inches='tight');plt.close('all')
    # Renal-feature dependence plots for features actually in the trained model.
    for term in ['creatinine','bun','potassium','anion_gap']:
        choices=[c for c in numeric if term in c and ('latest' in c or 'mean' in c)]
        if not choices: continue
        feature=choices[0]; j=cols.index(feature)
        fig,ax=plt.subplots(figsize=(7,5));ax.scatter(sample[feature],lv[:,j],s=10,alpha=.4)
        ax.set_xlabel(feature);ax.set_ylabel('LightGBM SHAP (margin)');ax.set_title(feature+' dependence')
        fig.tight_layout();fig.savefig(out/f'dependence_{term}.png',dpi=180);plt.close(fig)
    result=info.reset_index(drop=True).copy()
    result['lightgbm_probability']=p_l;result['catboost_probability']=p_c;result['raw_ensemble_probability']=raw
    result.to_parquet(out/'phase5_sample_predictions.parquet',index=False)
    # Patient-specific contributions for each model, rather than mathematically invalid
    # weighted averaging of log-odds SHAP values.
    for tag,pos in [('highest',int(np.argmax(raw))),('lowest',int(np.argmin(raw)))]:
        for model,vals in [('lightgbm',lv),('catboost',cat_vals)]:
            ix=np.argsort(np.abs(vals[pos]))[-15:][::-1]
            pd.DataFrame({'feature':np.array(cols)[ix],'shap_margin':vals[pos,ix]}).to_csv(out/f'{tag}_{model}_explanation.csv',index=False)
    report={'sample_n':len(sample),'model_features':len(cols),'lightgbm_weight':w,
        'raw_ensemble_mean_probability':float(raw.mean()),'observed_event_rate':float(info.icu_transfer_24_72h.mean()),
        'interpretation':'Model-specific SHAP values are in margin units and must not be averaged to explain blended probability. SHAP is associational, not causal.',
        'note':'Research-only retrospective explanations. No protected patient information included in plots.'}
    (out/'phase5_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));print('Saved to',out)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['phase1','phase2','phase4b','output']: p.add_argument('--'+k,required=True)
    p.add_argument('--sample',type=int,default=500)
    main(p.parse_args())
