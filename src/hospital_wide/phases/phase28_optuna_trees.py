#!/usr/bin/env python3
"""Optuna tuning of LightGBM, CatBoost, XGBoost for hospital-wide or renal-history ICU transfer.
Uses Phase22B leakage-reduced feature loader. Evaluation split is never accessed.
Hyperparameters selected by early_stop AUPRC; final validation assessed once per best model.
"""
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def score(y,p):
    k=max(1,int(np.ceil(len(y)*.1))); idx=np.argsort(-p,kind='stable')[:k]
    return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p)),'recall_top10':float(y[idx].sum()/max(1,y.sum())),
            'expected_events':float(p.sum()),'observed_events':int(y.sum())}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['phase18','phase18b','phase18c','labs','output']:ap.add_argument('--'+key,required=True)
    ap.add_argument('--phase22b-script',default=None)
    ap.add_argument('--population',choices=['renal','hospital'],default='renal')
    ap.add_argument('--models',nargs='+',choices=['lightgbm','catboost','xgboost'],default=['lightgbm','catboost','xgboost'])
    ap.add_argument('--trials',type=int,default=40);ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--timeout',type=int,default=None)
    a=ap.parse_args()
    import optuna
    out=Path(a.output).expanduser();out.mkdir(parents=True,exist_ok=True)
    src=Path(a.phase22b_script) if a.phase22b_script else Path(__file__).with_name('phase22b_labs_retrain.py')
    if not src.exists():ap.error('Missing Phase22B script; supply --phase22b-script')
    spec=importlib.util.spec_from_file_location('phase22b_loader',src); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    d=mod.load_data(a)
    if a.population=='renal':
        cols=[c for c in ['b_hist_dx_prior_renal','c_hist_renal_aki','c_hist_renal_ckd'] if c in d]
        if not cols:raise ValueError('Missing renal history features')
        mask=np.zeros(len(d),bool)
        for c in cols:mask |= d[c].fillna(0).to_numpy()>0
        d=d.loc[mask].copy()
    if d.groupby('subject_id').split.nunique().max()>1:raise ValueError('Patient overlap across partitions')
    if 'evaluation' in set(d.split):d=d.loc[d.split!='evaluation'].copy()
    cats=[c for c in ['gender','admission_type','admission_location'] if c in d]
    exclude={'subject_id','hadm_id','split','icu_24_72','first_event','event_hours','landmark','landmark_time','admittime','dischtime','deathtime'}
    nums=[c for c in d if c not in exclude and c not in cats and pd.api.types.is_numeric_dtype(d[c])]
    unsafe=[c for c in nums if c!='b_hist_days_since_discharge' and any(z in c.lower() for z in ['discharge','death','outcome','target','timestamp'])]
    if unsafe:raise ValueError('Unsafe predictors: '+str(unsafe))
    split={s:d.split.eq(s).to_numpy() for s in ['train','early_stop','validation']}
    pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value=0))]),nums),('cat',OneHotEncoder(handle_unknown='ignore'),cats)],sparse_threshold=0.3)
    X={s:(pre.fit_transform(d.loc[split[s],nums+cats]) if s=='train' else pre.transform(d.loc[split[s],nums+cats])) for s in split}
    y={s:d.loc[split[s],'icu_24_72'].astype(int).to_numpy() for s in split}
    print('Population',a.population,'n:',{s:int(split[s].sum()) for s in split},'features:',X['train'].shape[1],flush=True)
    if any(len(np.unique(y[s]))!=2 for s in split):raise ValueError('A partition has only one class')
    joblib.dump({'preprocessor':pre,'numeric_columns':nums,'categorical_columns':cats},out/'preprocessing.joblib')
    def build(name,trial):
        if name=='lightgbm':
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=trial.suggest_int('n_estimators',150,1000),learning_rate=trial.suggest_float('learning_rate',.008,.15,log=True),num_leaves=trial.suggest_int('num_leaves',8,96),max_depth=trial.suggest_int('max_depth',3,10),min_child_samples=trial.suggest_int('min_child_samples',20,200),subsample=trial.suggest_float('subsample',.65,1),subsample_freq=1,colsample_bytree=trial.suggest_float('colsample_bytree',.6,1),reg_lambda=trial.suggest_float('reg_lambda',.1,50,log=True),reg_alpha=trial.suggest_float('reg_alpha',1e-5,5,log=True),verbosity=-1,n_jobs=a.threads,random_state=a.seed)
        if name=='catboost':
            from catboost import CatBoostClassifier
            return CatBoostClassifier(iterations=trial.suggest_int('iterations',200,1000),depth=trial.suggest_int('depth',3,8),learning_rate=trial.suggest_float('learning_rate',.008,.15,log=True),l2_leaf_reg=trial.suggest_float('l2_leaf_reg',1,30,log=True),random_strength=trial.suggest_float('random_strength',.01,3,log=True),loss_function='Logloss',thread_count=a.threads,random_seed=a.seed,verbose=False,allow_writing_files=False)
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=trial.suggest_int('n_estimators',150,1000),max_depth=trial.suggest_int('max_depth',2,8),learning_rate=trial.suggest_float('learning_rate',.008,.15,log=True),subsample=trial.suggest_float('subsample',.6,1),colsample_bytree=trial.suggest_float('colsample_bytree',.6,1),min_child_weight=trial.suggest_float('min_child_weight',1,30,log=True),gamma=trial.suggest_float('gamma',1e-5,5,log=True),reg_lambda=trial.suggest_float('reg_lambda',.1,50,log=True),reg_alpha=trial.suggest_float('reg_alpha',1e-5,5,log=True),tree_method='hist',eval_metric='logloss',n_jobs=a.threads,random_state=a.seed)
    predictions=pd.DataFrame({'subject_id':d.loc[split['validation'],'subject_id'].to_numpy(),'hadm_id':d.loc[split['validation'],'hadm_id'].to_numpy(),'outcome':y['validation']})
    report={'population':a.population,'feature_policy':'conservative plus cached labs','n_features_encoded':int(X['train'].shape[1]),'models':{},'warning':'Exploratory; repeated validation use; cached lab provenance and historical coding availability not independently certified.'}
    for name in a.models:
        print('\nOPTUNA',name,flush=True)
        study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=a.seed),storage=f'sqlite:///{(out/(name+"_optuna.sqlite3")).resolve()}',study_name=f'{a.population}_{name}',load_if_exists=True)
        def objective(trial):
            model=build(name,trial)
            model.fit(X['train'],y['train'])
            p=model.predict_proba(X['early_stop'])[:,1]
            value=float(average_precision_score(y['early_stop'],p))
            trial.set_user_attr('stop_auroc',float(roc_auc_score(y['early_stop'],p)))
            return value
        remaining=max(0,a.trials-len(study.trials))
        if remaining:study.optimize(objective,n_trials=remaining,timeout=a.timeout,gc_after_trial=True)
        best=study.best_trial
        # Train on original training only. Early_stop was used for tuning; validation untouched until this point.
        from optuna.trial import FixedTrial
        model=build(name,FixedTrial(best.params));start=time.time();model.fit(X['train'],y['train'])
        pv=model.predict_proba(X['validation'])[:,1];predictions[name]=pv
        record={'best_trial':best.number,'stop_auprc':best.value,'params':best.params,'validation':score(y['validation'],pv),'fit_seconds':round(time.time()-start,2)}
        report['models'][name]=record
        joblib.dump(model,out/(name+'_best.joblib'))
        (out/'report.json').write_text(json.dumps(report,indent=2))
        predictions.to_parquet(out/'validation_predictions.parquet',index=False)
        print(name,record,flush=True)
    print('COMPLETE:',out)
if __name__=='__main__':main()
