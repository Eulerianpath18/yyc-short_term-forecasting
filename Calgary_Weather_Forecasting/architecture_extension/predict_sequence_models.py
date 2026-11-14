"""Recreate test predictions from saved controlled sequence checkpoints."""
from pathlib import Path
import numpy as np,pandas as pd,torch
from sequence_data import ROOT,HORIZONS,LOOKBACK,build,split_masks,transformed_source
from train_sequence_models import SequenceMLP
from tcn_v2 import TCNCompact
OUT=ROOT/'architecture_extension/results';prep=dict(np.load(OUT/'sequence_preprocessing_stats.npz'));src=build(False);m=split_masks(src);matrix=torch.from_numpy(transformed_source(src,prep));orig=torch.from_numpy(src.origins[m['test']]);static=torch.from_numpy(src.static[m['test']]);truth=src.targets[m['test']];times=src.origin_timestamps[m['test']];offs=torch.arange(-LOOKBACK+1,1)
rows=[]
for kind in ('MLP','TCN'):
  for seed in (2403,2026,17):
    ck=torch.load(OUT/'checkpoints'/f'{kind.lower()}_seed{seed}.pt',map_location='cpu',weights_only=False);mu=np.asarray(ck['target_mean']);sd=np.asarray(ck['target_std'])
    model=SequenceMLP(LOOKBACK,matrix.shape[1]) if kind=='MLP' else TCNCompact(matrix.shape[1],hidden=16,output_dim=5,dropout=.05);model.load_state_dict(ck['state_dict']);model.eval();pp=[]
    with torch.no_grad():
      for a in range(0,len(orig),4096):
        ind=torch.arange(a,min(a+4096,len(orig)));rr=orig[ind][:,None]+offs[None,:];pp.append(model(matrix[rr],static[ind]).numpy()*sd+mu)
    pred=np.concatenate(pp)
    for q,h in enumerate(HORIZONS):
      for t,y,p in zip(times,truth[:,q],pred[:,q]):rows.append({'timestamp':t,'model':kind,'seed':seed,'horizon_hours':h,'observed_temp_C':y,'predicted_temp_C':p})
pd.DataFrame(rows).to_csv(OUT/'sequence_test_predictions.csv',index=False)
print('wrote',len(rows),'prediction rows')
