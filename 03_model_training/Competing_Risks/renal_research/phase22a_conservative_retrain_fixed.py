#!/usr/bin/env python3
"""Phase 21: discrete-time neural competing-risk ICU-transfer models.
Same Phase18 patient splits, Phase18B/C clinical features and Phase19B cached labs.
No held-out evaluation labels are loaded. Exploratory retrospective research only.
"""
import argparse, json, os, random, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib

EVENTS = {'icu':1, 'discharge':2, 'death':3}

def load_data(a):
    f=pd.read_parquet(Path(a.phase18)/'prelandmark_features.parquet')
    # Select only non-evaluation event rows before accessing event columns.
    f=f.loc[f['split'].isin(['train','early_stop','validation'])].copy()
    e=pd.read_parquet(Path(a.phase18)/'first_events.parquet', columns=['hadm_id','first_event','event_hours','icu_24_72'])
    e=e.loc[e.hadm_id.isin(f.hadm_id)]
    if f.hadm_id.duplicated().any() or e.hadm_id.duplicated().any(): raise ValueError('Duplicate admission IDs')
    d=f.merge(e,on='hadm_id',validate='one_to_one')
    if len(d)!=len(f):raise ValueError('Missing event rows')
    if d.groupby('subject_id').split.nunique().max()!=1:raise ValueError('Patient leakage across splits')
    for prefix,root,names in [('b',a.phase18b,['history','history_diagnoses']),('c',a.phase18c,['history_diagnoses'])]:
        for name in names:
            p=Path(root)/(name+'.parquet')
            if not p.exists():print('Missing optional:',p,flush=True);continue
            x=pd.read_parquet(p)
            if x.hadm_id.duplicated().any():raise ValueError(f'Duplicate keys in {p}')
            cols=[c for c in x if c!='hadm_id' and pd.api.types.is_numeric_dtype(x[c])]
            x=x[['hadm_id']+cols].rename(columns={c:f'{prefix}_{c}' for c in cols})
            marker=f'{prefix}_{name}_present';x[marker]=np.int8(1)
            d=d.merge(x,on='hadm_id',how='left',validate='one_to_one')
            d[marker]=d[marker].fillna(0).astype('int8')
            print(f'Loaded {prefix}/{name}: {len(x):,} admissions, {len(cols)} features',flush=True)
    if a.labs:
        if not a.staged_labs: raise ValueError('--labs requires --staged-labs for provenance verification')
        stage=Path(a.staged_labs)
        cfg=stage/'extraction_config.json'
        complete=stage/'extraction_complete.json'
        if not cfg.exists() or not complete.exists(): raise ValueError('Missing staged lab extraction provenance/completion; rebuild staged labs')
        config=json.loads(cfg.read_text())
        if Path(config['phase18']).resolve()!=Path(a.phase18).resolve(): raise ValueError('Staged lab source Phase18 mismatch')
        if not (stage/'parts').exists(): raise ValueError('Staged lab parts missing')
        selection=Path(a.labs)/'lab_selection.json'
        if not selection.exists(): raise ValueError('Missing lab selection metadata')
        print('CAUTION: cached labs provenance checked by path, not independently by row/hash',flush=True)
        p=Path(a.labs)/'lab_features.parquet'
        if not p.exists():raise FileNotFoundError(f'{p} missing. Run Phase19B lab aggregation first.')
        x=pd.read_parquet(p)
        if x.hadm_id.duplicated().any():raise ValueError('Duplicate lab keys')
        d=d.merge(x,on='hadm_id',how='left',validate='one_to_one')
        d['labs_present']=d.hadm_id.isin(x.hadm_id).astype('int8')
        print('Loaded cached labs:',len(x),'admissions',len(x.columns)-1,'features',flush=True)
    return d

def make_targets(df,k,interval):
    times=df.event_hours.to_numpy(dtype=float)
    events=df.first_event.astype(str).to_numpy()
    if not np.isfinite(times).all() or (times<0).any():raise ValueError('Invalid event times')
    if not set(events).issubset(set(EVENTS)):raise ValueError('Unexpected event types')
    target=np.zeros((len(df),k),dtype=np.int64)
    mask=np.ones((len(df),k),dtype=np.float32)
    for j in range(k):
        left=j*interval; right=(j+1)*interval
        # Event at 0 is assigned to first interval; right boundary belongs to preceding bin.
        occurred=(times>left)&(times<=right) if j else (times>=0)&(times<=right)
        for ev,idx in EVENTS.items():target[occurred&(events==ev),j]=idx
        # Once any competing event occurs, subsequent intervals are not in risk set.
        # Every event on a boundary belongs to the preceding interval.
        # The interval is at risk iff no event occurred at/before its left edge.
        mask[:,j]=((times>left) if j else (times>=0)).astype('float32')
    return target,mask

class HazardNet(nn.Module):
    def __init__(self,d,k,hidden,dropout,linear=False):
        super().__init__();self.k=k
        if linear:self.net=nn.Linear(d,k*4)
        else:self.net=nn.Sequential(nn.Linear(d,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,hidden//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden//2,k*4))
    def forward(self,x):return self.net(x).reshape(-1,self.k,4)

def loss_fn(logits,y,mask):
    logp=torch.log_softmax(logits,dim=-1)
    chosen=logp.gather(2,y.unsqueeze(-1)).squeeze(-1)
    return -(chosen*mask).sum()/mask.sum().clamp_min(1)

@torch.no_grad()
def predict(model,x,batch=4096):
    model.eval();outputs=[]
    for i in range(0,len(x),batch):
        p=torch.softmax(model(torch.from_numpy(x[i:i+batch])),dim=-1).numpy()
        outputs.append(p)
    prob=np.concatenate(outputs)
    surv=np.ones(len(prob));out=np.zeros((len(prob),prob.shape[1]),dtype=np.float64)
    for j in range(prob.shape[1]):
        prev=out[:,j-1] if j else 0
        out[:,j]=prev+surv*prob[:,j,1]
        surv*=prob[:,j,0]
    if (np.diff(out,axis=1)<-1e-8).any() or (out<0).any() or (out>1+1e-6).any():raise ValueError('Invalid CIF')
    return out,prob

def scores(y,p):
    y=np.asarray(y,dtype=int);p=np.clip(p,0,1)
    n_top=max(1,int(np.ceil(.1*len(y)))); top=np.argsort(-p,kind='stable')[:n_top]
    return dict(n=int(len(y)),positives=int(y.sum()),prevalence=float(y.mean()),auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),brier=float(brier_score_loss(y,p)),recall_at_top_10pct=float(y[top].sum()/max(y.sum(),1)),expected_events=float(p.sum()),observed_events=int(y.sum()))

def train_model(name,xtr,ytr,mtr,xst,yst,mst,a,out):
    model=HazardNet(xtr.shape[1],a.horizon_hours//a.interval_hours,a.hidden,a.dropout,linear=name=='linear')
    optim=torch.optim.AdamW(model.parameters(),lr=a.lr if name=='neural' else a.lr*.5,weight_decay=a.weight_decay)
    ds=TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr),torch.from_numpy(mtr))
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=0)
    best=float('inf');wait=0;best_epoch=0
    for epoch in range(1,a.epochs+1):
        model.train();total=0
        for xb,yb,mb in loader:
            optim.zero_grad(set_to_none=True);loss=loss_fn(model(xb),yb,mb)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optim.step();total+=loss.item()
        model.eval()
        with torch.no_grad():
            stop_losses=[]
            for i in range(0,len(xst),a.batch_size):
                stop_losses.append((loss_fn(model(torch.from_numpy(xst[i:i+a.batch_size])),torch.from_numpy(yst[i:i+a.batch_size]),torch.from_numpy(mst[i:i+a.batch_size])).item(),min(a.batch_size,len(xst)-i)))
            stop=sum(v*n for v,n in stop_losses)/sum(n for _,n in stop_losses)
        if stop<best-1e-5:
            best=stop;best_epoch=epoch;wait=0
            torch.save({'state_dict':model.state_dict(),'input_dim':xtr.shape[1],'intervals':a.horizon_hours//a.interval_hours,'hidden':a.hidden,'dropout':a.dropout,'linear':name=='linear','best_epoch':epoch},out/f'{name}_best.pt')
        else:wait+=1
        if epoch==1 or epoch%5==0 or wait>=a.patience:print(f'{name} epoch {epoch} train_batch_loss={total/max(len(loader),1):.5f} stop_loss={stop:.5f} best={best:.5f}',flush=True)
        if wait>=a.patience:break
    checkpoint=torch.load(out/f'{name}_best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    return model,dict(best_epoch=best_epoch,stop_loss=best)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for x in ['phase18','phase18b','phase18c','output']:ap.add_argument('--'+x,required=True)
    ap.add_argument('--labs',help='Optional Phase19B lab cache (requires --staged-labs)')
    ap.add_argument('--staged-labs',help='Original staged lab extraction directory for provenance check')
    ap.add_argument('--horizon-hours',type=int,default=48);ap.add_argument('--interval-hours',type=int,default=12)
    ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--patience',type=int,default=7)
    ap.add_argument('--batch-size',type=int,default=1024);ap.add_argument('--hidden',type=int,default=128)
    ap.add_argument('--dropout',type=float,default=.2);ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--weight-decay',type=float,default=.001);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()
    if a.horizon_hours!=48:raise ValueError('Phase18 icu_24_72 benchmark is defined for 48 hours after landmark; keep --horizon-hours 48')
    if a.interval_hours<=0 or a.horizon_hours%a.interval_hours:raise ValueError('Interval must divide horizon')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    d=load_data(a)
    if a.interval_hours != 12: raise ValueError('Use 12h intervals for the audited benchmark')
    if not (d['split'].isin(['train','early_stop','validation'])).all(): raise ValueError('Unexpected split')
    provenance={'included':['phase18_demographics','phase18_prior_completed_admissions','18b_history','18b_history_diagnoses','18c_history_diagnoses'], 'excluded':['18b_prescriptions','18b_emar','18b_orders','18b_microbiology','18c_prescriptions','18c_emar','18c_orders'], 'labs_enabled':bool(a.labs), 'labs_warning':'Lab cache verified by extraction configuration path and completion only; cannot prove cached values derive from those parts.', 'landmark':'24h after admission', 'horizon':'48h after landmark = 72h after admission'}
    if a.labs:provenance['included'].append('phase19b_cached_labs')
    (out/'feature_provenance.json').write_text(json.dumps(provenance,indent=2))
    cats=[c for c in ['gender','admission_type','admission_location'] if c in d]
    excluded={'subject_id','hadm_id','split','icu_24_72','first_event','event_hours','landmark','landmark_time','admittime','dischtime','deathtime'}
    nums=[c for c in d if c not in excluded and c not in cats and pd.api.types.is_numeric_dtype(d[c])]
    # Historical time-since-discharge is derived exclusively from completed prior stays.
    # Allow only this exact provenance-reviewed column, not arbitrary discharge fields.
    audited_historical_allowlist={'b_hist_days_since_discharge'}
    suspicious=[c for c in nums if c not in audited_historical_allowlist and any(t in c.lower() for t in ('time','date','outcome','target','discharge','death'))]
    if suspicious: raise ValueError('Potential leakage columns: '+str(suspicious))
    for c in nums:d[c]=pd.to_numeric(d[c],errors='coerce').astype('float32')
    prep=ColumnTransformer([('numeric',Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value=0)),('scaler',StandardScaler())]),nums),('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)],sparse_threshold=0)
    split={s:d.split.eq(s).to_numpy() for s in ['train','early_stop','validation']}
    X={}
    for s in split:
        sub=d.loc[split[s],nums+cats]
        X[s]=np.asarray(prep.fit_transform(sub) if s=='train' else prep.transform(sub),dtype=np.float32)
        X[s]=np.nan_to_num(X[s],nan=0,posinf=0,neginf=0)
    joblib.dump({'preprocessor':prep,'numeric_columns':nums,'categorical_columns':cats},out/'preprocessing.joblib')
    Y={};M={}
    for s in split:Y[s],M[s]=make_targets(d.loc[split[s]],a.horizon_hours//a.interval_hours,a.interval_hours)
    val=d.loc[split['validation']]
    y=val.icu_24_72.to_numpy(dtype=int)
    preds=pd.DataFrame({'subject_id':val.subject_id.to_numpy(),'hadm_id':val.hadm_id.to_numpy(),'outcome':y})
    results={}
    for name in ['linear','neural']:
        print('TRAINING',name,'train',len(X['train']),'stop',len(X['early_stop']),'validation',len(X['validation']),flush=True)
        model,info=train_model(name,X['train'],Y['train'],M['train'],X['early_stop'],Y['early_stop'],M['early_stop'],a,out)
        cumulative,probs=predict(model,X['validation'])
        for j in range(cumulative.shape[1]):preds[f'{name}_icu_by_{(j+1)*a.interval_hours}h']=cumulative[:,j].astype('float32')
        results[name]={'validation_icu_by_72h_admission':scores(y,cumulative[:,-1]),'training':info}
        print(name,results[name],flush=True)
        preds.to_parquet(out/'validation_predictions.parquet',index=False)
        (out/'validation_metrics.json').write_text(json.dumps({'models':results,'n_features_encoded':X['train'].shape[1],'n_numeric':len(nums),'n_categorical':len(cats),'evaluation_accessed':False,'warning':'Conservative feature subset; historical code availability not proven; optional lab cache provenance not independently certified. Internal validation only.'},indent=2))
    (out/'manifest.json').write_text(json.dumps({'interval_hours':a.interval_hours,'horizon_hours_after_landmark':a.horizon_hours,'landmark_hours_after_admission':24,'n_train':len(X['train']),'n_stop':len(X['early_stop']),'n_validation':len(X['validation']),'lab_cache':a.labs,'conservative_feature_subset':True,'evaluation_accessed':False,'models':['linear','neural'],'event_codes':{'remain':0,**EVENTS},'limitations':['Retrospective cohort and care-utilization endpoint','No patient calendar-wide occupancy reconstruction from MIMIC-IV','Original event-time availability not independently audited','Validation reused for exploratory comparisons']},indent=2))
    print('COMPLETE:',out,flush=True)
if __name__=='__main__':main()
