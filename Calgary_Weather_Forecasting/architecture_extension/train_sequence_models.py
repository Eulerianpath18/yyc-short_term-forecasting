"""Controlled flattened-MLP versus corrected TCN experiment.

Both architectures receive exactly the same 24-hour multistation tensor:
24 value channels (4 variables x 6 stations), 24 missingness-mask channels,
and 4 cyclic calendar features. Both jointly predict 1/6/12/24/48 h.
"""
from __future__ import annotations
import argparse,json,random,time
import numpy as np,pandas as pd,torch
from torch import nn
from sequence_data import ROOT,HORIZONS,LOOKBACK,build,split_masks,fit_preprocessor,transformed_source,summary
from tcn_v2 import TCNCompact
OUT=ROOT/"architecture_extension"/"results";CKPT=OUT/"checkpoints";AUDIT=ROOT/"results"/"sanity_checks"

def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
class SequenceMLP(nn.Module):
    def __init__(self,T,C,S=4,O=5):
        super().__init__();self.net=nn.Sequential(nn.Linear(T*C+S,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,O))
    def forward(self,x,s):return self.net(torch.cat([x.flatten(1),s],1))

def batch_windows(matrix,origins,indices,offsets):
    o=origins[indices];rows=o[:,None]+offsets[None,:];return matrix[rows]

def evaluate(model,matrix,origins,static,targets,batch=4096):
    model.eval();pp=[]
    offsets=torch.arange(-LOOKBACK+1,1)
    with torch.no_grad():
        for a in range(0,len(origins),batch):
            ind=torch.arange(a,min(a+batch,len(origins)));x=batch_windows(matrix,origins,ind,offsets);pp.append(model(x,static[ind]).numpy())
    return np.concatenate(pp),targets.numpy()

def train_one(kind,seed,matrix_np,src,masks):
    seed_all(seed);torch.set_num_threads(min(8,torch.get_num_threads()));matrix=torch.from_numpy(matrix_np);offsets=torch.arange(-LOOKBACK+1,1)
    splits={sp:{"o":torch.from_numpy(src.origins[masks[sp]]),"s":torch.from_numpy(src.static[masks[sp]]),"y":torch.from_numpy(src.targets[masks[sp]])} for sp in ("train","validation","test")}
    mu=splits["train"]["y"].mean(0);sd=splits["train"]["y"].std(0,unbiased=False);sd[sd==0]=1
    ytr=(splits["train"]["y"]-mu)/sd;yva=(splits["validation"]["y"]-mu)/sd
    C=matrix.shape[1];model=SequenceMLP(LOOKBACK,C) if kind=="MLP" else TCNCompact(C,hidden=16,output_dim=len(HORIZONS),dropout=.05)
    params=sum(p.numel() for p in model.parameters());opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-5);loss=nn.MSELoss();best=float("inf");state=None;stale=0;bestep=0;hist=[];start=time.time();bs=4096
    n=len(splits["train"]["o"])
    for ep in range(1,16):
        model.train();perm=torch.randperm(n,generator=torch.Generator().manual_seed(seed+ep));ls=[]
        for a in range(0,n,bs):
            ind=perm[a:a+bs];x=batch_windows(matrix,splits["train"]["o"],ind,offsets);s=splits["train"]["s"][ind];y=ytr[ind];opt.zero_grad();p=model(x,s);l=loss(p,y);l.backward();opt.step();ls.append(float(l.item()))
        model.eval();vls=[];nv=len(splits["validation"]["o"])
        with torch.no_grad():
            for a in range(0,nv,bs):
                ind=torch.arange(a,min(a+bs,nv));x=batch_windows(matrix,splits["validation"]["o"],ind,offsets);vls.append(float(loss(model(x,splits["validation"]["s"][ind]),yva[ind]).item()))
        vv=float(np.mean(vls));hist.append({"model":kind,"seed":seed,"epoch":ep,"train_loss":float(np.mean(ls)),"validation_loss":vv})
        if vv<best-1e-5:best=vv;bestep=ep;stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=3:break
    model.load_state_dict(state);evals={}
    for sp in ("validation","test"):
        rawp,y=evaluate(model,matrix,splits[sp]["o"],splits[sp]["s"],splits[sp]["y"]);evals[sp]=(y,rawp*sd.numpy()+mu.numpy())
    torch.save({"state_dict":state,"model":kind,"seed":seed,"target_mean":mu.numpy(),"target_std":sd.numpy(),"best_epoch":bestep,"input_channels":C},CKPT/f"{kind.lower()}_seed{seed}.pt")
    return evals,hist,{"model":kind,"seed":seed,"best_epoch":bestep,"best_validation_loss":best,"runtime_seconds":time.time()-start,"parameters":params}

def metric_rows(kind,seed,sp,y,p):
    rows=[]
    for q,h in enumerate(HORIZONS):
        e=p[:,q]-y[:,q];rows.append({"model":kind,"seed":seed,"split":sp,"horizon_hours":h,"n_samples":len(y),"rmse_C":float(np.sqrt(np.mean(e*e))),"mae_C":float(np.mean(np.abs(e))),"bias_C":float(np.mean(e))})
    return rows

def _merge_existing(path, new_df, keys):
    if path.exists():
        old=pd.read_csv(path);new_df=pd.concat([old,new_df],ignore_index=True)
    return new_df.drop_duplicates(keys,keep="last").sort_values(keys).reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--seeds",default="2403,2026,17");ap.add_argument("--models",default="MLP,TCN");a=ap.parse_args();seeds=[int(x) for x in a.seeds.split(",")];models=[x.strip().upper() for x in a.models.split(",")]
    if any(model not in {"MLP", "TCN"} for model in models):
        ap.error("--models must contain only MLP or TCN")
    OUT.mkdir(parents=True,exist_ok=True);CKPT.mkdir(parents=True,exist_ok=True);AUDIT.mkdir(parents=True,exist_ok=True);src=build(True);m=split_masks(src);summary(src).to_csv(AUDIT/"sequence_dataset_summary.csv",index=False);prep=fit_preprocessor(src,m["train"]);np.savez(OUT/"sequence_preprocessing_stats.npz",**prep);matrix=transformed_source(src,prep)
    rows=[];hist=[];runs=[]
    for kind in models:
        for seed in seeds:
            ev,hh,rr=train_one(kind,seed,matrix,src,m);hist+=hh;runs.append(rr)
            for sp,(y,p) in ev.items():rows+=metric_rows(kind,seed,sp,y,p)
            print(kind,seed,"best",rr["best_epoch"],"sec",round(rr["runtime_seconds"],1),flush=True)
    metrics_path=OUT/"sequence_model_metrics.csv";history_path=OUT/"sequence_training_history.csv";runs_path=OUT/"sequence_run_summary.csv"
    df=_merge_existing(metrics_path,pd.DataFrame(rows),["model","seed","split","horizon_hours"]);df.to_csv(metrics_path,index=False)
    hdf = pd.DataFrame(hist)
    if history_path.exists():
        # Replace the entire history of rerun model/seed pairs, even if a rerun stops earlier.
        old_history = pd.read_csv(history_path)
        reruns = pd.MultiIndex.from_frame(hdf[["model", "seed"]].drop_duplicates())
        keep = ~pd.MultiIndex.from_frame(old_history[["model", "seed"]]).isin(reruns)
        hdf = pd.concat([old_history.loc[keep], hdf], ignore_index=True)
    hdf.sort_values(["model", "seed", "epoch"]).to_csv(history_path, index=False)
    rdf=_merge_existing(runs_path,pd.DataFrame(runs),["model","seed"]);rdf.to_csv(runs_path,index=False)
    robust=df[df.split=="test"].groupby(["model","horizon_hours"]).agg(rmse_mean_C=("rmse_C","mean"),rmse_std_C=("rmse_C","std"),rmse_min_C=("rmse_C","min"),rmse_max_C=("rmse_C","max"),n_seeds=("seed","nunique")).reset_index();robust.to_csv(OUT/"sequence_seed_robustness.csv",index=False)
    all_seeds=sorted(int(x) for x in df.seed.unique());json.dump({"lookback_hours":24,"stations":6,"variables_per_station":["temperature","dew_point","relative_humidity","wind_speed"],"value_channels":24,"mask_channels":24,"total_sequence_channels":48,"static_features":4,"horizons":list(HORIZONS),"seeds":all_seeds,"same_information_for_mlp_and_tcn":True,"same_training_rule":True,"max_epochs":15,"patience":3,"batch_size":4096,"tcn":"TCNCompact hidden=16; causal residual blocks; dilations 1,2,4,8","tcn_receptive_field_hours":31,"note":"CPU-budgeted controlled comparison; not an exhaustive TCN search."},open(OUT/"sequence_experiment_config.json","w"),indent=2)
    print("\n",robust.to_string(index=False))
if __name__=="__main__":main()
