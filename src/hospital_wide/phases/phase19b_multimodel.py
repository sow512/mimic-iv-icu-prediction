#!/usr/bin/env python3
"""Phase 19B: disk-backed lab aggregation + patient-disjoint multimodel ICU benchmark.
Uses staged lab Parquet parts, Phase18/18B/18C. Does not read evaluation labels.
Requires pandas, numpy, pyarrow, duckdb, scipy, scikit-learn; optional xgboost, lightgbm, catboost.
"""
import argparse, json, hashlib, os, gc, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from scipy import sparse

def metrics(y,p):
    y=np.asarray(y,int);p=np.clip(np.asarray(p,float),0,1)
    order=np.argsort(-p,kind='stable'); top=max(1,int(np.ceil(.1*len(y))))
    return dict(auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),brier=float(brier_score_loss(y,p)),recall_at_top_10pct=float(y[order[:top]].sum()/max(1,y.sum())),expected_events=float(p.sum()),observed_events=int(y.sum()))

def load_modules(data,b,c):
    for tag,root,names in [('b',b,['history','history_diagnoses','prescriptions','emar','orders','microbiology']),('c',c,['history_diagnoses','prescriptions','emar','orders'])]:
        for name in names:
            f=Path(root)/(name+'.parquet')
            if not f.exists():print('Missing optional:',f,flush=True);continue
            x=pd.read_parquet(f)
            if x.hadm_id.duplicated().any():raise ValueError(f'Duplicate hadm_id: {f}')
            cols=[k for k in x if k!='hadm_id' and pd.api.types.is_numeric_dtype(x[k])]
            x=x[['hadm_id']+cols].rename(columns={k:f'{tag}_{k}' for k in cols})
            x[f'{tag}_{name}_present']=np.int8(1)
            data=data.merge(x,on='hadm_id',how='left',validate='one_to_one')
            print(f'Loaded {tag}/{name}: {len(x):,} rows, {len(cols)} numeric features',flush=True)
    return data

def aggregate_labs(a,features,train_ids):
    import duckdb
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cache=out/'lab_features.parquet'; selection=out/'lab_selection.json'
    if cache.exists() and selection.exists() and not a.rebuild_labs:
        old=json.loads(selection.read_text())
        if old['top_n']!=a.top_labs:raise ValueError('Cached lab selection uses different --top-labs; add --rebuild-labs')
        print('Using existing lab feature cache',flush=True)
        return pd.read_parquet(cache)
    path=str((Path(a.staged_labs)/'parts'/'*.parquet').resolve())
    if not list((Path(a.staged_labs)/'parts').glob('*.parquet')):raise FileNotFoundError('No staged lab parts found')
    db=duckdb.connect(str(out/'labs_work.duckdb'))
    db.execute(f"SET memory_limit='{a.memory_limit}'")
    db.execute(f"SET threads={a.threads}")
    temp=out/'duckdb_temp';temp.mkdir(exist_ok=True)
    db.execute(f"SET temp_directory='{str(temp)}'")
    db.register('train_ids_df',pd.DataFrame({'hadm_id':np.asarray(train_ids,dtype='int64')}))
    db.execute('CREATE OR REPLACE TEMP TABLE train_ids AS SELECT DISTINCT hadm_id FROM train_ids_df')
    print('Selecting laboratory concepts on TRAIN patients only...',flush=True)
    # A concept is an item/unit pair; only availability is used for selection, never outcomes.
    counts=db.execute('''SELECT l.itemid, coalesce(l.unit,'UNKNOWN') AS unit, count(DISTINCT l.hadm_id) AS n
        FROM read_parquet(?) l INNER JOIN train_ids t USING(hadm_id)
        WHERE l.valuenum IS NOT NULL AND isfinite(l.valuenum)
        GROUP BY 1,2 ORDER BY n DESC, itemid, unit LIMIT ?''',[path,a.top_labs]).df()
    if counts.empty:raise ValueError('No lab measurements for training admissions')
    counts['feature_key']=['lab_'+str(i)+'_'+hashlib.sha1(str(u).encode()).hexdigest()[:7] for i,u in zip(counts.itemid,counts.unit)]
    db.register('selected_df',counts[['itemid','unit','feature_key']]);db.execute('CREATE OR REPLACE TEMP TABLE selected AS SELECT * FROM selected_df')
    print(f'Aggregating {len(counts)} item/unit concepts across staged files...',flush=True)
    # Output is long format (~admissions x selected concepts), not 9.7m source rows.
    long=db.execute('''SELECT l.hadm_id, s.feature_key, count(*) AS n, avg(l.valuenum) AS mean,
      min(l.valuenum) AS minval, max(l.valuenum) AS maxval,
      arg_min(l.valuenum,l.charttime) AS firstval,
      arg_max(l.valuenum,l.charttime) AS lastval
      FROM read_parquet(?) l INNER JOIN selected s
      ON l.itemid=s.itemid AND coalesce(l.unit,'UNKNOWN')=s.unit
      WHERE l.valuenum IS NOT NULL AND isfinite(l.valuenum)
      GROUP BY 1,2''',[path]).df()
    db.close()
    # Pivot one statistic at a time, keeping peak memory bounded.
    wide=None
    for col in ['n','mean','minval','maxval','firstval','lastval']:
        pivot=long.pivot(index='hadm_id',columns='feature_key',values=col)
        pivot.columns=[f'{k}_{col}' for k in pivot.columns]
        pivot=pivot.astype('float32')
        wide=pivot if wide is None else wide.join(pivot,how='outer')
        del pivot;gc.collect()
    wide=wide.reset_index();wide.to_parquet(cache,index=False)
    selection.write_text(json.dumps({'top_n':a.top_labs,'selected':counts.to_dict(orient='records'),'warning':'Verify original staged extractor enforced storetime before landmark; staged files lack storetime.'},indent=2,default=str))
    print('Saved lab features:',cache,'rows',len(wide),'columns',len(wide.columns),flush=True)
    return wide

def run(a):
    os.environ['OMP_NUM_THREADS']=str(a.threads)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    f=pd.read_parquet(Path(a.phase18)/'prelandmark_features.parquet')
    e=pd.read_parquet(Path(a.phase18)/'first_events.parquet')
    if f.hadm_id.duplicated().any() or e.hadm_id.duplicated().any():raise ValueError('Duplicate admission keys')
    # Do not read event outcome from reserved evaluation rows.
    d=f.merge(e[['hadm_id','icu_24_72']],on='hadm_id',validate='one_to_one')
    if d.groupby('subject_id').split.nunique().max()!=1:raise ValueError('Patient split leakage')
    if set(d.split.unique())!={'train','early_stop','validation','evaluation'}:raise ValueError('Unexpected splits')
    d=d[d.split!='evaluation'].copy()
    d=load_modules(d,a.phase18b,a.phase18c)
    train_ids=d.loc[d.split=='train','hadm_id'].to_numpy()
    labs=aggregate_labs(a,f,train_ids)
    if labs.hadm_id.duplicated().any():raise ValueError('Duplicate lab admission keys')
    d=d.merge(labs,on='hadm_id',how='left',validate='one_to_one')
    d['labs_present']=d.hadm_id.isin(labs.hadm_id).astype('int8')
    del labs;gc.collect()
    excluded={'subject_id','hadm_id','split','icu_24_72','first_event','event_hours','landmark','landmark_time','admittime','dischtime','deathtime'}
    cats=[x for x in ['gender','admission_type','admission_location'] if x in d]
    numeric=[x for x in d if x not in excluded and x not in cats and pd.api.types.is_numeric_dtype(d[x])]
    labcols=[x for x in numeric if x.startswith('lab_')]
    basecols=[x for x in numeric if x not in labcols and x!='labs_present']
    if not labcols:raise ValueError('No lab columns')
    # No feature selection using validation labels; all input transformations fit train only.
    train=d.split=='train'; stop=d.split=='early_stop';val=d.split=='validation'
    ytr=d.loc[train,'icu_24_72'].astype(int).to_numpy();ys=d.loc[stop,'icu_24_72'].astype(int).to_numpy();yv=d.loc[val,'icu_24_72'].astype(int).to_numpy()
    print('Splits:',train.sum(),stop.sum(),val.sum(),'validation positives:',yv.sum(),flush=True)
    allresults=[];preds=pd.DataFrame({'hadm_id':d.loc[val,'hadm_id'].to_numpy(),'outcome':yv})
    for view,cols in [('no_labs',basecols),('labs_only',labcols+([c for c in ['age'] if c in basecols])),('combined',basecols+labcols+['labs_present'])]:
        print('\nVIEW',view,'features',len(cols)+len(cats),flush=True)
        prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value=0)),('scale',StandardScaler(with_mean=False))]),cols),('cat',OneHotEncoder(handle_unknown='ignore'),cats)],sparse_threshold=1.0)
        columns=cols+cats
        Xtr=sparse.csr_matrix(prep.fit_transform(d.loc[train,columns]));Xst=sparse.csr_matrix(prep.transform(d.loc[stop,columns]));Xv=sparse.csr_matrix(prep.transform(d.loc[val,columns]))
        for modelname in a.models:
            print('Training',view,modelname,flush=True)
            if modelname=='logistic': model=LogisticRegression(max_iter=350,solver='liblinear',random_state=a.seed)
            elif modelname=='extratrees': model=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=20,max_features=.6,n_jobs=a.threads,random_state=a.seed)
            elif modelname=='mlp': model=MLPClassifier(hidden_layer_sizes=(128,64),alpha=.01,early_stopping=True,n_iter_no_change=12,max_iter=80,random_state=a.seed)
            elif modelname=='xgboost':
                from xgboost import XGBClassifier
                model=XGBClassifier(n_estimators=350,max_depth=4,learning_rate=.05,subsample=.85,colsample_bytree=.85,reg_lambda=8,tree_method='hist',n_jobs=a.threads,random_state=a.seed,eval_metric='logloss')
            elif modelname=='lightgbm':
                import lightgbm as lgb
                model=lgb.LGBMClassifier(n_estimators=650,learning_rate=.035,num_leaves=24,max_depth=6,min_child_samples=90,reg_lambda=8,verbosity=-1,n_jobs=a.threads,random_state=a.seed)
            elif modelname=='catboost':
                from catboost import CatBoostClassifier
                model=CatBoostClassifier(iterations=650,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function='Logloss',thread_count=a.threads,random_seed=a.seed,verbose=False)
            else:raise ValueError(modelname)
            try:
                if modelname=='lightgbm':
                    import lightgbm as lgb
                    model.fit(Xtr,ytr,eval_set=[(Xst,ys)],callbacks=[lgb.early_stopping(45,verbose=False)])
                elif modelname=='catboost':model.fit(Xtr,ytr,eval_set=(Xst,ys),early_stopping_rounds=45,verbose=False)
                else:model.fit(Xtr,ytr)
                p=model.predict_proba(Xv)[:,1]
                record={'view':view,'model':modelname,'features':len(cols)+len(cats),**metrics(yv,p)}
                allresults.append(record);preds[f'{view}__{modelname}']=p.astype('float32')
                pd.DataFrame(allresults).to_csv(out/'validation_results.csv',index=False)
                preds.to_parquet(out/'validation_predictions.parquet',index=False)
                print(record,flush=True)
            except (ImportError,ModuleNotFoundError) as ex:print('Optional model unavailable:',ex,flush=True)
            finally:
                del model;gc.collect()
        del Xtr,Xst,Xv,prep;gc.collect()
    (out/'manifest.json').write_text(json.dumps({'evaluation_accessed':False,'train_n':int(train.sum()),'stop_n':int(stop.sum()),'validation_n':int(val.sum()),'lab_columns':len(labcols),'nonlab_columns':len(basecols),'categorical_columns':cats,'models':a.models,'warning':'Exploratory internal validation; staged lab source storetime cutoff and medication/order availability require source-level audit. Absolute timestamps excluded. No time-to-event model in this script.'},indent=2))
    print('\nFinished:',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['phase18','phase18b','phase18c','staged-labs','output']:p.add_argument('--'+arg,required=True)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--memory-limit',default='2GB');p.add_argument('--top-labs',type=int,default=35)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--rebuild-labs',action='store_true')
    p.add_argument('--models',nargs='+',choices=['logistic','extratrees','mlp','xgboost','lightgbm','catboost'],default=['logistic','extratrees','mlp','xgboost','lightgbm'])
    run(p.parse_args())
