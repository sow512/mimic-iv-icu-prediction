#!/usr/bin/env python3
"""Renal-only binary and competing-risk neural hyperparameter search.
Uses Phase21 feature loader, historical-renal cohort, fixed patient splits.
All tuning uses early_stop only; validation evaluated once for each selected objective.
Full policy is exploratory and not leakage-certified.
"""
import argparse, importlib.util, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib

RENAL=['b_hist_dx_prior_renal','c_hist_renal_aki','c_hist_renal_ckd']
CATS=['gender','admission_type','admission_location']
EXCLUDED={'subject_id','hadm_id','split','icu_24_72','first_event','event_hours','landmark','landmark_time','admittime','dischtime','deathtime'}

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def metrics(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);k=max(1,int(np.ceil(len(y)*.1)))
    ix=np.argsort(-p,kind='stable')[:k]
    return {'n':len(y),'events':int(y.sum()),'auroc':float(roc_auc_score(y,p)),
            'auprc':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),
            'recall_top10':float(y[ix].sum()/max(1,y.sum())), 'expected_events':float(p.sum())}

class Net(nn.Module):
    def __init__(self,d,hidden,dropout,objective):
        super().__init__();self.objective=objective
        self.net=nn.Sequential(nn.Linear(d,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout),
                               nn.Linear(hidden,hidden//2),nn.GELU(),nn.Dropout(dropout),
                               nn.Linear(hidden//2,1 if objective=='binary' else 16))
    def forward(self,x):
        z=self.net(x)
        return z.squeeze(-1) if self.objective=='binary' else z.reshape(-1,4,4)

def targets(ev,t):
    y=np.zeros((len(t),4),dtype=np.int64);m=np.zeros((len(t),4),dtype=np.float32)
    mapping={'icu':1,'discharge':2,'death':3}
    for j in range(4):
        left=12*j;right=12*(j+1)
        # right-closed intervals; boundary event belongs to previous interval.
        at_risk=(t>left) if j else (t>=0)
        m[:,j]=at_risk.astype(np.float32)
        occurring=((t>left)&(t<=right)) if j else ((t>=0)&(t<=right))
        for name,code in mapping.items():y[occurring&(ev==name),j]=code
    return y,m

def lossfn(net,xb,yb,mb,objective):
    z=net(xb)
    if objective=='binary':return nn.functional.binary_cross_entropy_with_logits(z,yb.float())
    lp=nn.functional.log_softmax(z,dim=-1)
    return -(lp.gather(2,yb.long().unsqueeze(-1)).squeeze(-1)*mb).sum()/mb.sum().clamp_min(1)

@torch.no_grad()
def predict(net,x,objective):
    net.eval();allp=[]
    for i in range(0,len(x),4096):
        z=net(torch.from_numpy(x[i:i+4096]))
        if objective=='binary':p=torch.sigmoid(z).numpy()
        else:
            q=torch.softmax(z,dim=-1).numpy();surv=np.ones(len(q));p=np.zeros(len(q))
            for j in range(4):p+=surv*q[:,j,1];surv*=q[:,j,0]
        allp.append(p)
    return np.concatenate(allp)

def fit(xtr,ytr,mtr,xst,yst,mst,objective,cfg,epochs,patience,seed):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    net=Net(xtr.shape[1],cfg['hidden'],cfg['dropout'],objective)
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    ds=TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr),torch.from_numpy(mtr))
    loader=DataLoader(ds,batch_size=cfg['batch_size'],shuffle=True,num_workers=0)
    xs=torch.from_numpy(xst);ys=torch.from_numpy(yst);ms=torch.from_numpy(mst)
    best=float('inf');state=None;best_epoch=0;wait=0
    for epoch in range(1,epochs+1):
        net.train()
        for xb,yb,mb in loader:
            opt.zero_grad(set_to_none=True);l=lossfn(net,xb,yb,mb,objective)
            l.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        net.eval();total=0;count=0
        with torch.no_grad():
            for i in range(0,len(xs),4096):
                n=min(4096,len(xs)-i)
                val=lossfn(net,xs[i:i+n],ys[i:i+n],ms[i:i+n],objective)
                total+=float(val)*n;count+=n
        stop=total/count
        if stop<best-1e-5:
            best=stop;best_epoch=epoch;wait=0
            state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        else:wait+=1
        if wait>=patience:break
    net.load_state_dict(state)
    return net,{'best_epoch':best_epoch,'stop_loss':best,'trained_epochs':epoch}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('phase18','phase18b','phase18c','labs','output'):ap.add_argument('--'+key,required=True)
    ap.add_argument('--policy',choices=['full','conservative'],default='conservative')
    ap.add_argument('--script-dir',default=str(Path(__file__).resolve().parent))
    ap.add_argument('--trials',type=int,default=12)
    ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--patience',type=int,default=8)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--selection',choices=['auprc','auroc'],default='auprc')
    a=ap.parse_args();out=Path(a.output).expanduser();out.mkdir(parents=True,exist_ok=True)
    if a.trials<1:ap.error('--trials must be >=1')
    torch.set_num_threads(a.threads)
    dependency=Path(a.script_dir)/('phase21_competing_risk.py' if a.policy=='full' else 'phase22b_labs_retrain.py')
    if not dependency.exists():raise FileNotFoundError(f'Missing {dependency}; use --script-dir')
    src=load_module(dependency,'feature_loader')
    # Some conservative loaders accept staged-labs metadata args.
    a.staged_labs=str(Path(a.labs).parent/'staged_labs')
    d=src.load_data(a)
    if d.hadm_id.duplicated().any() or d.groupby('subject_id').split.nunique().gt(1).any():raise ValueError('Patient split or admission key violation')
    if not set(d.split.unique()).issubset({'train','early_stop','validation'}):raise ValueError('Unexpected evaluation split loaded')
    if any(c not in d for c in RENAL):raise ValueError('Missing renal history flags')
    renal=d[RENAL].fillna(0).gt(0).any(axis=1)
    d=d.loc[renal].copy()
    cats=[c for c in CATS if c in d]
    nums=[c for c in d if c not in EXCLUDED and c not in cats and pd.api.types.is_numeric_dtype(d[c])]
    allowed={'b_hist_days_since_discharge'}
    suspicious=[c for c in nums if c not in allowed and any(t in c.lower() for t in ('time','date','outcome','target','discharge','death'))]
    if suspicious:raise ValueError('Unsafe-looking predictors: '+str(suspicious))
    for c in nums:d[c]=pd.to_numeric(d[c],errors='coerce').astype('float32')
    masks={s:d.split.eq(s).to_numpy() for s in ('train','early_stop','validation')}
    prep=ColumnTransformer([
        ('numeric',Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value=0)),('scale',StandardScaler())]),nums),
        ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)],sparse_threshold=0)
    x={}
    for s in masks:
        frame=d.loc[masks[s],nums+cats]
        x[s]=np.nan_to_num(np.asarray(prep.fit_transform(frame) if s=='train' else prep.transform(frame),dtype=np.float32),nan=0,posinf=0,neginf=0)
    joblib.dump({'preprocessor':prep,'numeric_columns':nums,'categorical_columns':cats},out/'preprocessing.joblib')
    y={s:d.loc[masks[s],'icu_24_72'].to_numpy(dtype=np.int64) for s in masks}
    ev={s:d.loc[masks[s],'first_event'].astype(str).to_numpy() for s in masks}
    times={s:d.loc[masks[s],'event_hours'].to_numpy(float) for s in masks}
    hz={s:targets(ev[s],times[s]) for s in masks}
    print('Renal cohort',len(d),'train',len(y['train']),'stop',len(y['early_stop']),'validation',len(y['validation']),'features',x['train'].shape[1],flush=True)
    rng=np.random.default_rng(a.seed)
    configs=[{'hidden':128,'dropout':.2,'lr':.001,'weight_decay':.001,'batch_size':1024}]
    while len(configs)<a.trials:
        cfg={'hidden':int(rng.choice([32,64,128,256])),
             'dropout':float(rng.choice([0.05,.1,.2,.3,.4])),
             'lr':float(rng.choice([.0002,.0005,.001,.002])),
             'weight_decay':float(rng.choice([.00001,.0001,.001,.01])),
             'batch_size':int(rng.choice([512,1024,2048]))}
        if cfg not in configs:configs.append(cfg)
    summary={'policy':a.policy,'selection_metric':a.selection,'selection_partition':'early_stop','trials':a.trials,'models':{},'warnings':['Repeated validation use in prior phases; this is exploratory.','Feature availability not fully audited.','Lab cache provenance requires independent verification.']}
    preds=d.loc[masks['validation'],['subject_id','hadm_id']].copy();preds['outcome']=y['validation']
    for objective in ('binary','competing'):
        folder=out/objective;folder.mkdir(exist_ok=True)
        rows=[];best_score=-np.inf;best_cfg=None;best_net=None;best_info=None
        ytr=y['train'].astype(np.float32) if objective=='binary' else hz['train'][0]
        yst=y['early_stop'].astype(np.float32) if objective=='binary' else hz['early_stop'][0]
        mtr=np.ones(len(ytr),dtype=np.float32) if objective=='binary' else hz['train'][1]
        mst=np.ones(len(yst),dtype=np.float32) if objective=='binary' else hz['early_stop'][1]
        for j,cfg in enumerate(configs):
            tic=time.perf_counter()
            net,info=fit(x['train'],ytr,mtr,x['early_stop'],yst,mst,objective,cfg,a.epochs,a.patience,a.seed+j)
            p=predict(net,x['early_stop'],objective)
            met=metrics(y['early_stop'],p)
            score=met[a.selection]
            rows.append({'trial':j,**cfg,**info,**{f'stop_{k}':v for k,v in met.items()},'seconds':round(time.perf_counter()-tic,2)})
            print(objective,'trial',j+1,'/',len(configs),'stop AUPRC',round(met['auprc'],5),'AUROC',round(met['auroc'],5),'best epoch',info['best_epoch'],flush=True)
            if score>best_score:
                best_score=score;best_cfg=cfg;best_net=net;best_info=info
                torch.save({'state_dict':net.state_dict(),'input_dim':x['train'].shape[1],
                            'objective':objective,'config':cfg,'training':info},folder/'best.pt')
        pd.DataFrame(rows).to_csv(folder/'tuning_early_stop.csv',index=False)
        # Validation is accessed only after configuration selection is finished.
        ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        selected=Net(ck['input_dim'],best_cfg['hidden'],best_cfg['dropout'],objective)
        selected.load_state_dict(ck['state_dict'])
        pv=predict(selected,x['validation'],objective)
        preds[objective]=pv.astype('float32')
        result={'best_config':best_cfg,'best_training':best_info,'selected_stop_score':best_score,
                'validation':metrics(y['validation'],pv)}
        (folder/'report.json').write_text(json.dumps(result,indent=2))
        summary['models'][objective]=result
    preds.to_parquet(out/'validation_predictions.parquet',index=False)
    summary['n_features_encoded']=x['train'].shape[1]
    summary['n_renal_validation']=len(y['validation'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2));print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
