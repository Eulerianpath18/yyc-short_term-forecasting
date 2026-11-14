"""LightGBM benchmark on the same 67 engineered spatial predictors."""
from pathlib import Path
import time
import numpy as np,pandas as pd
from lightgbm import LGBMRegressor,early_stopping,log_evaluation
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'spatial_extension/data/processed/model_ready_spatial.csv';MAN=ROOT/'spatial_extension/data/processed/spatial_feature_manifest.csv';OUT=ROOT/'architecture_extension/results';OUT.mkdir(parents=True,exist_ok=True)
H=(1,6,12,24,48)
df=pd.read_csv(DATA);df['timestamp']=pd.to_datetime(df.timestamp);features=pd.read_csv(MAN).feature.tolist();rows=[];info=[];predrows=[]
for h in H:
    target=f'target_temp_{h}h';valid=df.timestamp+pd.to_timedelta(h,unit='h');tr=valid<pd.Timestamp('2024-01-01');va=(df.timestamp>=pd.Timestamp('2024-01-01'))&(valid<pd.Timestamp('2025-01-01'));te=(df.timestamp>=pd.Timestamp('2025-01-01'))&(valid<pd.Timestamp('2026-01-01'))
    med=df.loc[tr,features].median();Xtr=df.loc[tr,features].fillna(med);Xv=df.loc[va,features].fillna(med);Xt=df.loc[te,features].fillna(med);ytr=df.loc[tr,target].to_numpy();yv=df.loc[va,target].to_numpy();yt=df.loc[te,target].to_numpy()
    t=time.time();m=LGBMRegressor(n_estimators=500,learning_rate=.04,num_leaves=31,min_child_samples=30,colsample_bytree=.8,subsample=.85,reg_lambda=1.0,random_state=2403,n_jobs=8,verbosity=-1)
    m.fit(Xtr,ytr,eval_set=[(Xv,yv)],callbacks=[early_stopping(40,verbose=False),log_evaluation(0)]);pv=m.predict(Xv,num_iteration=m.best_iteration_);pt=m.predict(Xt,num_iteration=m.best_iteration_)
    info.append({'horizon_hours':h,'best_iteration':int(m.best_iteration_),'runtime_seconds':time.time()-t})
    for sp,y,p in [('validation',yv,pv),('test',yt,pt)]:
        e=p-y;rows.append({'model':'LightGBM','split':sp,'horizon_hours':h,'n_samples':len(y),'rmse_C':float(np.sqrt(np.mean(e*e))),'mae_C':float(np.mean(np.abs(e))),'bias_C':float(np.mean(e))})
    for ts,obs,p in zip(df.loc[te,'timestamp'],yt,pt):predrows.append({'timestamp':ts,'horizon_hours':h,'observed_temp_C':obs,'lightgbm_temp_C':p})
    print(h,rows[-1],flush=True)
pd.DataFrame(rows).to_csv(OUT/'lightgbm_metrics.csv',index=False);pd.DataFrame(info).to_csv(OUT/'lightgbm_training_info.csv',index=False);pd.DataFrame(predrows).to_csv(OUT/'lightgbm_test_predictions.csv',index=False)
