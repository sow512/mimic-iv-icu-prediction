#!/usr/bin/env python3
"""Binary renal ICU-transfer neural benchmark, aligned with Phase23 feature policy.
Outcome: ICU transfer during hours 24-72 after hospital admission.
Only train/early_stop/validation used. No evaluation split accessed.
"""
import argparse, importlib.util, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
import joblib

def load_script(path):
    spec=importlib.util.spec_from_file_location('phase22b_loader',str(path));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def metrics(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);k=max(1,int(np.ceil(len(y)*.1)));ix=np.argsort(-p,kind='stable')[:k]
    return {'n':int(len(y)),'events':int(y.sum()),'prevalence':float(y.mean()),'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),'recall_top10':float(y[ix].sum()/y.sum()),'precision_top10':float(y[ix].mean()),'expected_events':float(p.sum())}

class BinaryNet(nn.Module):
    def __init__(self,dim,hidden,dropout):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,hidden//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden//2,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def predict(model,X,batch=4096):
    model.eval();z=[]
    with torch.no_grad():
        for i in range(0,len(X),batch):z.append(torch.sigmoid(model(torch.from_numpy(X[i:i+batch]))).numpy())
    return np.concatenate(z)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('phase18','phase18b','phase18c','labs','staged-labs','output'):ap.add_argument('--'+name,required=True)
    ap.add_argument('--phase22b-script',default=None);ap.add_argument('--phase23-output',default=None,help='Optional Phase23 predictions for matched comparisons')
    ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--patience',type=int,default=10);ap.add_argument('--batch-size',type=int,default=1024)
    ap.add_argument('--hidden',type=int,default=128);ap.add_argument('--dropout',type=float,default=.2);ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--weight-decay',type=float,default=.001);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--bootstraps',type=int,default=1000)
    a=ap.parse_args();out=Path(a.output).expanduser();out.mkdir(parents=True,exist_ok=True)
    script=Path(a.phase22b_script) if a.phase22b_script else Path(__file__).with_name('phase22b_labs_retrain.py')
    if not script.exists():ap.error(f'Missing {script}; pass --phase22b-script')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    m=load_script(script);d=m.load_data(a)
    renal_cols=[c for c in ('b_hist_dx_prior_renal','c_hist_renal_aki','c_hist_renal_ckd') if c in d]
    if not renal_cols:raise ValueError('Missing historical renal eligibility features')
    renal=np.zeros(len(d),dtype=bool)
    for c in renal_cols:renal|=d[c].fillna(0).to_numpy(dtype=float)>0
    d=d.loc[renal].copy()
    if d.groupby('subject_id').split.nunique().max()!=1:raise ValueError('Patient split violation')
    cats=[c for c in ('gender','admission_type','admission_location') if c in d]
    excluded={'subject_id','hadm_id','split','icu_24_72','first_event','event_hours','landmark','landmark_time','admittime','dischtime','deathtime'}
    nums=[c for c in d if c not in excluded and c not in cats and pd.api.types.is_numeric_dtype(d[c])]
    allow={'b_hist_days_since_discharge'}
    bad=[c for c in nums if c not in allow and any(t in c.lower() for t in ('time','date','outcome','target','discharge','death'))]
    if bad:raise ValueError('Unsafe columns: '+str(bad))
    splits={s:d.split.eq(s).to_numpy() for s in ('train','early_stop','validation')}
    if any(not splits[s].any() for s in splits):raise ValueError('Empty split')
    for c in nums:d[c]=pd.to_numeric(d[c],errors='coerce').astype('float32')
    pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value=0)),('scale',StandardScaler())]),nums),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)],sparse_threshold=0)
    X={};y={}
    for s in splits:
        x=d.loc[splits[s],nums+cats]
        X[s]=np.nan_to_num(np.asarray(pre.fit_transform(x) if s=='train' else pre.transform(x),dtype=np.float32),nan=0,posinf=0,neginf=0)
        y[s]=d.loc[splits[s],'icu_24_72'].astype(int).to_numpy()
        print(s,len(y[s]),'events',int(y[s].sum()),flush=True)
    joblib.dump({'preprocessor':pre,'numeric_columns':nums,'categorical_columns':cats},out/'preprocessing.joblib')
    model=BinaryNet(X['train'].shape[1],a.hidden,a.dropout)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    loss=nn.BCEWithLogitsLoss() # unweighted: probabilities remain interpretable without class-weight correction
    ds=TensorDataset(torch.from_numpy(X['train']),torch.from_numpy(y['train'].astype(np.float32)))
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=0)
    best=float('inf');best_epoch=0;wait=0;start=time.time()
    for epoch in range(1,a.epochs+1):
        model.train();total=0.
        for xb,yb in loader:
            opt.zero_grad(set_to_none=True);z=model(xb);l=loss(z,yb);l.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();total+=float(l.item())
        model.eval();loss_sum=0.;count=0
        with torch.no_grad():
            for i in range(0,len(y['early_stop']),a.batch_size):
                xb=torch.from_numpy(X['early_stop'][i:i+a.batch_size]);yb=torch.from_numpy(y['early_stop'][i:i+a.batch_size].astype(np.float32));n=len(yb)
                loss_sum+=float(loss(model(xb),yb).item())*n;count+=n
        stop=loss_sum/count
        if stop<best-1e-5:
            best=stop;best_epoch=epoch;wait=0
            torch.save({'state_dict':model.state_dict(),'input_dim':X['train'].shape[1],'hidden':a.hidden,'dropout':a.dropout,'best_epoch':epoch},out/'binary_neural_best.pt')
        else:wait+=1
        if epoch==1 or epoch%5==0 or wait>=a.patience:print(f'epoch {epoch} train_loss={total/len(loader):.5f} stop_loss={stop:.5f} best={best:.5f}',flush=True)
        if wait>=a.patience:break
    ckpt=torch.load(out/'binary_neural_best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ckpt['state_dict'])
    val=d.loc[splits['validation']];p=predict(model,X['validation'])
    preds=pd.DataFrame({'subject_id':val.subject_id.to_numpy(),'hadm_id':val.hadm_id.to_numpy(),'outcome':y['validation'],'binary_neural':p})
    result=metrics(y['validation'],p)
    report={'scope':'renal-history validation; held-out evaluation excluded','model':'binary MLP, BCEWithLogitsLoss','target':'ICU first event by 72h after admission, conditional on eligibility at 24h','n_features_encoded':int(X['train'].shape[1]),'n_numeric':len(nums),'n_categorical':len(cats),'renal_eligibility_columns':renal_cols,'best_epoch':best_epoch,'early_stop_bce':best,'train_seconds':round(time.time()-start,2),'binary_neural':result,'limitations':['Historical diagnoses and cached lab provenance require independent availability audit.','Renal eligibility is historical diagnosis based.','Validation repeatedly explored; reserved evaluation untouched.']}
    if a.phase23_output:
        ref=pd.read_parquet(Path(a.phase23_output)/'validation_predictions.parquet')
        if ref.hadm_id.duplicated().any() or preds.hadm_id.duplicated().any():raise ValueError('Duplicate admission IDs')
        merged=preds.merge(ref,on='hadm_id',how='inner',suffixes=('','_phase23'),validate='one_to_one')
        if len(merged)!=len(preds) or len(ref)!=len(preds):raise ValueError('Phase23 validation admission sets differ')
        if not np.array_equal(merged.outcome.to_numpy(),merged.outcome_phase23.to_numpy()):raise ValueError('Outcomes differ')
        names=[c for c in ('lightgbm','catboost','neural_competing_risk') if c in merged]
        report['phase23_matched']={n:metrics(merged.outcome.to_numpy(),merged[n].to_numpy()) for n in names}
        groups=list(merged.groupby('subject_id',sort=False).indices.values());rng=np.random.default_rng(a.seed)
        keys=('auroc','auprc','brier','recall_top10');boot={n:{k:[] for k in keys} for n in names}
        for _ in range(a.bootstraps):
            idx=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]);yy=merged.outcome.to_numpy()[idx]
            if len(np.unique(yy))<2:continue
            binary=metrics(yy,merged.binary_neural.to_numpy()[idx])
            for n in names:
                other=metrics(yy,merged[n].to_numpy()[idx])
                for k in keys:boot[n][k].append(binary[k]-other[k])
        report['paired_patient_bootstrap_binary_minus_phase23_95ci']={n:{k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in vals.items()} for n,vals in boot.items()}
    preds.to_parquet(out/'validation_predictions.parquet',index=False)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print('COMPLETE:',out)
if __name__=='__main__':main()
