#!/usr/bin/env python3
"""Phase 14B: Research-only competing-risk baseline vs PyTorch MLP.
Full renal diagnosis-defined landmark cohort, 24-72h, 12h bins.
Time-stamped laboratory observations must have charttime AND storetime <= landmark.
Do not use evaluation patients for model selection.
"""
import argparse, hashlib, json, os, random, copy, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.exceptions import ConvergenceWarning
SEED=20261022
EVENTS=['none','icu','discharge','death']
def locate(base, folder, stem):
    hits=list(Path(base).glob(f'physionet.org/files/mimiciv/*/{folder}/{stem}.csv*'))
    if len(hits)!=1: raise FileNotFoundError(f'Expected one {folder}/{stem}.csv[.gz] under {base}; found {hits}')
    return hits[0]

def read(base,folder,stem,cols):
    p=locate(base,folder,stem);print('Reading',p,flush=True)
    return pd.read_csv(p,usecols=cols,low_memory=False)

def hashed_split(ids):
    # deterministic subject-level assignment, independent of row order
    return np.array([('train' if k<65 else 'stop' if k<77 else 'validation' if k<88 else 'evaluation') for k in
        (int.from_bytes(hashlib.sha256(f'phase14-{int(s)}'.encode()).digest()[:8],'big')%100 for s in ids)])

def full_cohort(args):
    pat=read(args.base,'hosp','patients',['subject_id','gender','anchor_age','anchor_year'])
    adm=read(args.base,'hosp','admissions',['subject_id','hadm_id','admittime','dischtime','deathtime','admission_type','admission_location'])
    dx=read(args.base,'hosp','diagnoses_icd',['hadm_id','icd_code','icd_version'])
    icu=read(args.base,'icu','icustays',['subject_id','hadm_id','intime','outtime'])
    dx['icd_code']=dx.icd_code.astype(str).str.upper().str.replace('.','',regex=False)
    version=pd.to_numeric(dx.icd_version,errors='coerce')
    renal=((version==9)&dx.icd_code.str.match(r'^(584|585)'))|((version==10)&dx.icd_code.str.match(r'^(N17|N18)'))
    renal_hadm=set(dx.loc[renal,'hadm_id'].dropna().astype('int64'))
    adm=adm[adm.hadm_id.isin(renal_hadm)].copy()
    adm=adm.merge(pat,on='subject_id',validate='many_to_one')
    for c in ['admittime','dischtime','deathtime']:adm[c]=pd.to_datetime(adm[c],errors='coerce')
    adm=adm[adm.admittime.notna() & adm.dischtime.notna() & (adm.dischtime>adm.admittime)].copy()
    adm['age']=adm.anchor_age + (adm.admittime.dt.year-adm.anchor_year)
    adm=adm[adm.age>=18].copy()
    adm['landmark_time']=adm.admittime+pd.Timedelta(hours=24)
    adm=adm[(adm.dischtime>adm.landmark_time)&(adm.deathtime.isna()|(adm.deathtime>adm.landmark_time))].copy()
    for c in ['intime','outtime']:icu[c]=pd.to_datetime(icu[c],errors='coerce')
    icu=icu[icu.hadm_id.isin(adm.hadm_id)].copy()
    # Fail closed for invalid ICU linkage/timestamps, rather than silently using bad times.
    links=icu.merge(adm[['hadm_id','subject_id','admittime','dischtime']],on='hadm_id',suffixes=('_icu','_adm'),validate='many_to_one')
    bad=links.intime.isna()|links.outtime.isna()|(links.intime>=links.outtime)|(links.subject_id_icu!=links.subject_id_adm)|\
        (links.intime<links.admittime)|(links.intime>links.dischtime)|(links.outtime>links.dischtime+pd.Timedelta(minutes=1))
    invalid=set(links.loc[bad,'hadm_id']);print('Excluded malformed ICU-linked admissions:',len(invalid))
    adm=adm[~adm.hadm_id.isin(invalid)].copy()
    first=links.loc[~bad].groupby('hadm_id').intime.min().rename('icu_time')
    adm=adm.join(first,on='hadm_id')
    adm=adm[adm.icu_time.isna()|(adm.icu_time>adm.landmark_time)].copy()
    return adm[['subject_id','hadm_id','landmark_time','admittime','dischtime','deathtime','icu_time','age','gender','admission_type','admission_location']]

def event_table(d,horizon):
    d=d.copy();land=d.landmark_time
    # Death at/after discharge timestamp is not necessarily an in-hospital death.
    death=d.deathtime.where(d.deathtime.notna()&(d.deathtime<=d.dischtime))
    # Death takes precedence over discharge on exact ties; ICU takes precedence over either on ties.
    dt=pd.DataFrame({'icu':d.icu_time,'death':death,'discharge':d.dischtime})
    earliest=dt.min(axis=1)
    code=np.select([d.icu_time.eq(earliest),death.eq(earliest),d.dischtime.eq(earliest)],[1,3,2],default=0)
    end=land+pd.Timedelta(hours=horizon)
    observed=(earliest<=end)
    d['event_code']=np.where(observed,code,0).astype('int8')
    d['duration_hours']=((earliest.where(observed,end)-land).dt.total_seconds()/3600).clip(lower=0,upper=horizon)
    if (d.duration_hours<=0).any():raise ValueError('Event at/before landmark: inspect eligibility')
    if d.hadm_id.duplicated().any() or d.subject_id.isna().any():raise ValueError('Invalid cohort identifiers')
    return d


def lab_features(base, d, out, chunksize=500000):
    cache=out/'prelandmark_labs.parquet'
    if cache.exists():
        z=pd.read_parquet(cache)
        if set(z.hadm_id)==set(d.hadm_id) and not z.hadm_id.duplicated().any():
            print('Reusing cached lab features',flush=True)
            return z
        raise ValueError('Cached labs do not match cohort; use a fresh output directory')
    path=locate(base,'hosp','labevents')
    # Small, predeclared set of clinical analytes; item IDs resolved from d_labitems.
    dictpath=locate(base,'hosp','d_labitems')
    items=pd.read_csv(dictpath,usecols=['itemid','label','fluid'])
    wanted=['creatinine','urea nitrogen','potassium','sodium','bicarbonate','chloride',
            'anion gap','glucose','lactate','albumin','calcium, total','phosphate',
            'hemoglobin','platelet count','white blood cells','ph','pco2','po2']
    mapping={}
    for _,r in items.iterrows():
        label=str(r['label']).strip().lower()
        fluid=str(r['fluid']).strip().lower()
        if label in wanted and fluid in ('blood','serum','plasma'):
            mapping[int(r['itemid'])]=label.replace(' ','_').replace(',','')
    if not mapping: raise ValueError('No lab itemids found; inspect d_labitems')
    print('Selected laboratory item IDs:',len(mapping),flush=True)
    cohort=d.set_index('hadm_id')[['admittime','landmark_time']]
    ids=set(cohort.index)
    frames=[]; scanned=0; retained=0
    use=['hadm_id','itemid','charttime','storetime','valuenum','valueuom']
    for chunk in pd.read_csv(path,usecols=use,chunksize=chunksize,low_memory=False):
        scanned+=len(chunk)
        z=chunk.loc[chunk.hadm_id.isin(ids)&chunk.itemid.isin(mapping)].copy()
        if z.empty: continue
        z=z.join(cohort,on='hadm_id',how='inner')
        for c in ['charttime','storetime']:z[c]=pd.to_datetime(z[c],errors='coerce')
        z=z.loc[z.charttime.notna()&z.storetime.notna()&
                (z.charttime>=z.admittime)&(z.charttime<z.landmark_time)&
                (z.storetime<=z.landmark_time)].copy()
        z['valuenum']=pd.to_numeric(z.valuenum,errors='coerce')
        z=z.loc[z.valuenum.notna()&np.isfinite(z.valuenum)].copy()
        # Unit differences can invalidate aggregation; keep separate unit-specific features.
        z['analyte']=z.itemid.map(mapping)+'__'+z.valueuom.fillna('unknown').astype(str).str.lower().str.replace(r'[^a-z0-9]+','_',regex=True)
        z=z[['hadm_id','analyte','charttime','valuenum']]
        retained+=len(z)
        if len(z):frames.append(z)
        if scanned%10000000<chunksize:print('Lab rows scanned:',scanned,'retained:',retained,flush=True)
    if not frames:raise ValueError('No qualifying pre-landmark labs')
    z=pd.concat(frames,ignore_index=True)
    z=z.sort_values(['hadm_id','analyte','charttime'])
    agg=z.groupby(['hadm_id','analyte'],observed=True).valuenum.agg(['count','min','max','mean','std','first','last'])
    agg['delta']=agg['last']-agg['first']
    # Avoid proliferation of rare unit/analyte combinations: threshold based on coverage, no labels used.
    coverage=z.groupby('analyte').hadm_id.nunique()
    keep=coverage[coverage>=max(30,int(.005*len(d)))].index
    agg=agg.loc[agg.index.get_level_values('analyte').isin(keep)]
    wide=agg.unstack('analyte')
    wide.columns=['lab_'+str(metric)+'__'+str(analyte) for metric,analyte in wide.columns]
    wide=wide.reset_index()
    result=d[['hadm_id']].merge(wide,on='hadm_id',how='left',validate='one_to_one')
    result.to_parquet(cache,index=False)
    print('Lab feature columns:',result.shape[1]-1,'rows:',len(result),flush=True)
    return result

def prepare(d, train):
    forbidden={'subject_id','hadm_id','admittime','landmark_time','dischtime','deathtime',
               'icu_time','duration_hours','event_code'}
    cols=[c for c in d if c not in forbidden]
    x=d[cols].copy()
    cat=[c for c in cols if x[c].dtype=='object' or str(x[c].dtype) in ('category','string')]
    num=[c for c in cols if c not in cat]
    x[num]=x[num].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    keep=[c for c in num if x.iloc[train][c].notna().mean()>=.02]
    cat=[c for c in cat if x.iloc[train][c].notna().mean()>=.02]
    imp=SimpleImputer(strategy='median',add_indicator=False)
    scaler=StandardScaler()
    numeric=scaler.fit_transform(imp.fit_transform(x.iloc[train][keep]))
    allnum=scaler.transform(imp.transform(x[keep])).astype('float32')
    # Explicit missingness indicators, calculated before imputation.
    missing=x[keep].isna().to_numpy(dtype='float32')
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    catdata=x[cat].fillna('__MISSING__').astype(str)
    enc.fit(catdata.iloc[train])
    categorical=enc.transform(catdata).astype('float32')
    a=np.concatenate([allnum,missing,categorical],axis=1).astype('float32')
    if not np.isfinite(a).all():raise ValueError('Nonfinite transformed features')
    return a,{'numeric':keep,'categorical':cat,'imputer':imp,'scaler':scaler,'encoder':enc}

def long_rows(d, interval):
    rows=[]; bins=[]; labels=[]
    duration=d.duration_hours.to_numpy(); event=d.event_code.to_numpy()
    for k in range(48//interval):
        at=np.flatnonzero(duration>k*interval)
        lab=np.where((duration[at]<=(k+1)*interval)&(event[at]>0),event[at],0)
        rows.append(at);bins.append(np.full(len(at),k,dtype='int32'));labels.append(lab)
    return np.concatenate(rows),np.concatenate(bins),np.concatenate(labels).astype('int64')

def add_time(x,bins,n_bins):
    t=np.zeros((len(bins),n_bins),dtype='float32')
    t[np.arange(len(bins)),bins]=1
    return np.concatenate([x,t],axis=1)

def cif(model, x, bins_count, predict_fn):
    survival=np.ones(len(x),dtype='float64'); cumulative=np.zeros((len(x),3)); curves=[]
    for k in range(bins_count):
        a=add_time(x,np.full(len(x),k,dtype='int32'),bins_count)
        p=predict_fn(model,a)
        if p.shape!=(len(x),4):raise ValueError('Expected four conditional event probabilities')
        cumulative+=survival[:,None]*p[:,1:]
        survival*=p[:,0]
        curves.append(cumulative[:,0].copy())
    if np.max(np.abs(cumulative.sum(axis=1)+survival-1))>1e-4:
        raise AssertionError('Competing risk probabilities do not sum to one')
    return np.column_stack(curves),cumulative,survival

class RiskNet(nn.Module):
    def __init__(self,dim,width,dropout):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(dim,width),nn.ReLU(),nn.LayerNorm(width),
                                  nn.Dropout(dropout),nn.Linear(width,width//2),nn.ReLU(),
                                  nn.Dropout(dropout),nn.Linear(width//2,4))
    def forward(self,x):return self.layers(x)

def neural_train(x,y, stopx,stopy, args, out):
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    torch.set_num_threads(args.threads)
    model=RiskNet(x.shape[1],args.width,args.dropout)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    train_ds=torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y))
    loader=torch.utils.data.DataLoader(train_ds,batch_size=args.batch,shuffle=True)
    sx=torch.from_numpy(stopx);sy=torch.from_numpy(stopy)
    best=float('inf');best_epoch=-1;patience=0
    for epoch in range(args.epochs):
        model.train();losses=[]
        for xx,yy in loader:
            optimizer.zero_grad();loss=nn.functional.cross_entropy(model(xx),yy)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.0);optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val=float(nn.functional.cross_entropy(model(sx),sy).item())
        if val<best-1e-5:
            best=val;best_epoch=epoch+1;patience=0
            torch.save({'state_dict':model.state_dict(),'dim':x.shape[1],
                        'width':args.width,'dropout':args.dropout,'epoch':best_epoch},out/'risknet_best.pt')
        else:patience+=1
        if epoch==0 or (epoch+1)%5==0:
            print(f'Epoch {epoch+1}: train loss={np.mean(losses):.5f} stop loss={val:.5f} best={best:.5f}',flush=True)
        if patience>=args.patience:break
    state=torch.load(out/'risknet_best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state['state_dict']);model.eval()
    return model,{'best_epoch':best_epoch,'stop_log_loss':best}

def nn_prob(model,a):
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(a)),dim=1).numpy()

def baseline_prob(model,a):
    p=model.predict_proba(a);z=np.zeros((len(a),4))
    for j,k in enumerate(model.classes_):z[:,int(k)]=p[:,j]
    return z

def metrics(y,p):
    if len(np.unique(y))<2:return {'n':len(y),'positives':int(y.sum()),'auroc':None,'auprc':None}
    return {'n':len(y),'positives':int(y.sum()),'prevalence':float(np.mean(y)),
            'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p))}

def main(args):
    if args.interval<=0 or 48%args.interval:raise ValueError('Interval must divide 48')
    out=Path(args.output).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Output already completed; choose new directory')
    d=full_cohort(args)
    d=event_table(d,48)
    split=hashed_split(d.subject_id)
    if pd.DataFrame({'id':d.subject_id,'split':split}).groupby('id').split.nunique().max()!=1:
        raise AssertionError('Patient leakage')
    print('Eligible admissions:',len(d),'patients:',d.subject_id.nunique(),flush=True)
    print('Events:',dict(zip(EVENTS,np.bincount(d.event_code,minlength=4).tolist())),flush=True)
    labs=lab_features(args.base,d,out,args.chunksize)
    d=d.merge(labs,on='hadm_id',how='left',validate='one_to_one')
    tr=np.flatnonzero(split=='train');st=np.flatnonzero(split=='stop');va=np.flatnonzero(split=='validation')
    x,pre=prepare(d,tr)
    print('Encoded input dimension:',x.shape[1],flush=True)
    rows,bins,yy=long_rows(d,args.interval);part=split[rows];nb=48//args.interval
    train=part=='train';stop=part=='stop'
    tx=add_time(x[rows[train]],bins[train],nb);ty=yy[train]
    sx=add_time(x[rows[stop]],bins[stop],nb);sy=yy[stop]
    if set(np.unique(ty))!={0,1,2,3}:raise ValueError('Training split missing competing event type')
    # Reference multinomial model, fitted on training patients only.
    baseline=LogisticRegression(max_iter=300,solver='lbfgs',C=.3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',ConvergenceWarning)
        baseline.fit(tx,ty)
    joblib.dump({'model':baseline,'preprocessing':pre,'interval':args.interval},out/'baseline.joblib')
    network,train_info=neural_train(tx,ty,sx,sy,args,out)
    joblib.dump(pre,out/'preprocessing.joblib')
    target=(d.event_code.iloc[va].to_numpy()==1).astype(int)
    results={}
    for name,model,fn in [('multinomial_logistic',baseline,baseline_prob),('renalrisknet',network,nn_prob)]:
        curves,cumulative,survival=cif(model,x[va],nb,fn)
        results[name]=metrics(target,curves[:,-1])
        pred=pd.DataFrame({'subject_id':d.subject_id.iloc[va].to_numpy(),
            'hadm_id':d.hadm_id.iloc[va].to_numpy(),'icu_by_72h':target,
            'icu_cif':curves[:,-1],'discharge_cif':cumulative[:,1],
            'death_cif':cumulative[:,2],'survival':survival})
        for k in range(nb):pred[f'icu_cif_hour_{24+(k+1)*args.interval}']=curves[:,k]
        pred.to_parquet(out/f'{name}_validation_predictions.parquet',index=False)
        print(name,results[name],flush=True)
    report={'purpose':'exploratory, retrospective diagnosis-defined renal cohort',
            'cohort_admissions':len(d),'cohort_patients':int(d.subject_id.nunique()),
            'event_counts':dict(zip(EVENTS,np.bincount(d.event_code,minlength=4).tolist())),
            'split_counts':{s:int(np.sum(split==s)) for s in ['train','stop','validation','evaluation']},
            'lab_features':labs.shape[1]-1,'encoded_features':x.shape[1],
            'interval_hours':args.interval,'validation':results,'neural_training':train_info,
            'evaluation_accessed':False,
            'limitations':['AKI/CKD diagnosis codes may be assigned after landmark',
                           'ICU entry is utilization, not physiological deterioration',
                           'CIF AUROC/AUPRC on validation only; no external validation',
                           'Lab aggregation uses charttime and storetime cutoff; unit-specific features',
                           'Event priority on timestamp ties: ICU, death, discharge',
                           'No ICU transfers after the landmark are used as predictors',
                           'Death recorded after discharge is not an in-hospital competing event']}
    (out/'complete.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    p.add_argument('--interval',type=int,default=12);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--chunksize',type=int,default=500000)
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--patience',type=int,default=12)
    p.add_argument('--batch',type=int,default=2048);p.add_argument('--width',type=int,default=128)
    p.add_argument('--dropout',type=float,default=.25);p.add_argument('--lr',type=float,default=.0007)
    p.add_argument('--weight-decay',type=float,default=.001)
    main(p.parse_args())
