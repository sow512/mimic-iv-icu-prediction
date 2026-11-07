#!/usr/bin/env python3
"""Hospital-wide binary MLP: full Phase21 vs conservative+labs Phase22B.
Development splits only. Renal subgroup defined by historical renal diagnoses.
Requires phase21_competing_risk.py and phase22b_labs_retrain.py alongside this file.
"""
import argparse, json, random, time
from pathlib import Path
import importlib.util
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


def module(path, name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

class BinaryNet(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout),
                               nn.Linear(hidden,hidden//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden//2,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def metric(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    if not len(y) or len(np.unique(y))<2:return {'n':int(len(y)),'events':int(y.sum()),'error':'both classes required'}
    k=max(1,int(np.ceil(.1*len(y)))); idx=np.argsort(-p,kind='stable')[:k]
    return {'n':int(len(y)),'events':int(y.sum()),'prevalence':float(y.mean()),
            'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p)),'recall_top10':float(y[idx].sum()/y.sum()),
            'precision_top10':float(y[idx].mean()),'expected_events':float(p.sum())}

def renal_mask(d):
    cols=['b_hist_dx_prior_renal','c_hist_renal_aki','c_hist_renal_ckd']
    missing=[c for c in cols if c not in d]
    if missing:raise ValueError('Missing renal eligibility features: '+str(missing))
    return d[cols].fillna(0).gt(0).any(axis=1).to_numpy(),cols

@torch.no_grad()
def infer(net,X,batch=4096):
    net.eval();return np.concatenate([torch.sigmoid(net(torch.from_numpy(X[i:i+batch]))).numpy() for i in range(0,len(X),batch)])

def train(Xtr,ytr,Xstop,ystop,args,path):
    net=BinaryNet(Xtr.shape[1],args.hidden,args.dropout)
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    loader=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr.astype('float32'))),
                      batch_size=args.batch_size,shuffle=True,num_workers=0)
    loss=nn.BCEWithLogitsLoss(); best=float('inf'); bestepoch=0;wait=0
    xt=torch.from_numpy(Xstop);yt=torch.from_numpy(ystop.astype('float32'))
    for epoch in range(1,args.epochs+1):
        net.train();total=0.
        for xb,yb in loader:
            opt.zero_grad(set_to_none=True); l=loss(net(xb),yb);l.backward()
            nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();total+=l.item()
        net.eval()
        with torch.no_grad():
            parts=[]
            for i in range(0,len(xt),4096):
                parts.append((loss(net(xt[i:i+4096]),yt[i:i+4096]).item(),len(xt[i:i+4096])))
            stop=sum(v*n for v,n in parts)/sum(n for _,n in parts)
        if stop<best-1e-5:
            best=stop;bestepoch=epoch;wait=0
            torch.save({'state_dict':net.state_dict(),'input_dim':Xtr.shape[1],
                        'hidden':args.hidden,'dropout':args.dropout,'best_epoch':epoch},path)
        else:wait+=1
        if epoch==1 or epoch%5==0 or wait>=args.patience:
            print(f'epoch {epoch} train_batch_loss={total/len(loader):.5f} stop_bce={stop:.5f} best={best:.5f}',flush=True)
        if wait>=args.patience:break
    net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'])
    return net,{'best_epoch':bestepoch,'early_stop_bce':best}

def run_policy(policy,a,out):
    script=Path(a.script_dir)/('phase21_competing_risk.py' if policy=='full' else 'phase22b_labs_retrain.py')
    if not script.exists():raise FileNotFoundError(f'Missing dependency {script}. Copy your uploaded Phase21 and Phase22B scripts here or pass --script-dir.')
    src=module(script,'source_'+policy)
    print('\nLOADING',policy,flush=True)
    d=src.load_data(a)
    if d.hadm_id.duplicated().any() or d.groupby('subject_id').split.nunique().gt(1).any():
        raise ValueError('Duplicate admissions or patient split leakage')
    if set(d.split.unique())!={'train','early_stop','validation'}:raise ValueError('Unexpected splits')
    cats=[c for c in ['gender','admission_type','admission_location'] if c in d]
    excluded={'subject_id','hadm_id','split','icu_24_72','first_event','event_hours','landmark','landmark_time','admittime','dischtime','deathtime'}
    nums=[c for c in d if c not in excluded and c not in cats and pd.api.types.is_numeric_dtype(d[c])]
    allowed={'b_hist_days_since_discharge'}
    suspect=[c for c in nums if c not in allowed and any(t in c.lower() for t in ('time','date','outcome','target','discharge','death'))]
    if suspect:raise ValueError('Potential unsafe predictors: '+str(suspect))
    renal,renal_cols=renal_mask(d)
    for c in nums:d[c]=pd.to_numeric(d[c],errors='coerce').astype('float32')
    prep=ColumnTransformer([
        ('numeric',Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value=0)),('scale',StandardScaler())]),nums),
        ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)],sparse_threshold=0)
    masks={s:d.split.eq(s).to_numpy() for s in ['train','early_stop','validation']}
    X={}
    for s in masks:
        sub=d.loc[masks[s],nums+cats]
        X[s]=np.nan_to_num(np.asarray(prep.fit_transform(sub) if s=='train' else prep.transform(sub),dtype=np.float32),nan=0,posinf=0,neginf=0)
    y={s:d.loc[masks[s],'icu_24_72'].to_numpy(dtype=int) for s in masks}
    folder=out/policy;folder.mkdir(parents=True,exist_ok=True)
    joblib.dump({'preprocessor':prep,'numeric_columns':nums,'categorical_columns':cats},folder/'preprocessing.joblib')
    print(f'{policy}: train={len(y["train"])} stop={len(y["early_stop"])} val={len(y["validation"])} features={X["train"].shape[1]}',flush=True)
    t=time.perf_counter()
    net,training=train(X['train'],y['train'],X['early_stop'],y['early_stop'],a,folder/'binary_best.pt')
    p=infer(net,X['validation']); v=d.loc[masks['validation'],['subject_id','hadm_id']].copy()
    v['outcome']=y['validation'];v['prediction']=p.astype('float32');v['renal_history']=renal[masks['validation']]
    v.to_parquet(folder/'validation_predictions.parquet',index=False)
    result={'policy':policy,'n_numeric':len(nums),'n_encoded':X['train'].shape[1],
            'training':training,'seconds':round(time.perf_counter()-t,2),
            'hospital_wide':metric(v.outcome,v.prediction),
            'renal_subgroup':metric(v.loc[v.renal_history,'outcome'],v.loc[v.renal_history,'prediction']),
            'renal_eligibility_columns':renal_cols}
    (folder/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    return v,result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['phase18','phase18b','phase18c','labs','staged-labs','output']:ap.add_argument('--'+key,required=True)
    ap.add_argument('--script-dir',default=str(Path(__file__).resolve().parent))
    ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--patience',type=int,default=10)
    ap.add_argument('--batch-size',type=int,default=1024);ap.add_argument('--hidden',type=int,default=128)
    ap.add_argument('--dropout',type=float,default=.2);ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--weight-decay',type=float,default=.001);ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--policies',nargs='+',choices=['full','conservative'],default=['full','conservative'])
    a=ap.parse_args();out=Path(a.output).expanduser();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    results={};tables={}
    for policy in a.policies:
        v,res=run_policy(policy,a,out);tables[policy]=v;results[policy]=res
    if len(tables)==2:
        f=tables['full'];c=tables['conservative']
        merged=f.merge(c,on=['hadm_id','subject_id'],suffixes=('_full','_conservative'),validate='one_to_one')
        if len(merged)!=len(f) or len(merged)!=len(c):raise ValueError('Validation admission mismatch')
        if not (merged.outcome_full==merged.outcome_conservative).all():raise ValueError('Outcome mismatch')
        if not (merged.renal_history_full==merged.renal_history_conservative).all():raise ValueError('Renal definition mismatch')
        merged.to_parquet(out/'matched_validation_predictions.parquet',index=False)
    report={'models':results,'evaluation_accessed':False,'prediction_landmark':'24h after admission',
            'outcome':'ICU first event within 48h after landmark (72h after admission)',
            'warnings':['Full policy includes unaudited prescriptions/orders/medications/microbiology.',
                        'Conservative labs provenance only checked by staged source path and metadata.',
                        'Historical diagnosis availability not independently certified.',
                        'Internal validation reused for prior experiments; not final test performance.',
                        'Comparing policies changes input features; differences cannot isolate architecture.']}
    (out/'comparison_report.json').write_text(json.dumps(report,indent=2))
    print('\nCOMPLETE:',out,flush=True)
if __name__=='__main__':main()
