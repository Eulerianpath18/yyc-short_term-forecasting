"""Principal-component regression baseline on 67 engineered spatial predictors."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'spatial_extension/data/processed/model_ready_spatial.csv';MAN=ROOT/'spatial_extension/data/processed/spatial_feature_manifest.csv';OUT=ROOT/'architecture_extension/results';OUT.mkdir(parents=True,exist_ok=True)
H=(1,6,12,24,48);KS=(16,32,48,64);df=pd.read_csv(DATA);df.timestamp=pd.to_datetime(df.timestamp);features=pd.read_csv(MAN).feature.tolist();search=[];metrics=[]
for h in H:
    target=f'target_temp_{h}h';tt=df.timestamp+pd.to_timedelta(h,unit='h');tr=tt<pd.Timestamp('2024-01-01');va=(df.timestamp>=pd.Timestamp('2024-01-01'))&(tt<pd.Timestamp('2025-01-01'));te=(df.timestamp>=pd.Timestamp('2025-01-01'))&(tt<pd.Timestamp('2026-01-01'))
    med=df.loc[tr,features].median();Xtr=df.loc[tr,features].fillna(med);Xv=df.loc[va,features].fillna(med);Xt=df.loc[te,features].fillna(med);mu=Xtr.mean();sd=Xtr.std(ddof=0).replace(0,1);Xtr=(Xtr-mu)/sd;Xv=(Xv-mu)/sd;Xt=(Xt-mu)/sd;ytr=df.loc[tr,target].to_numpy();yv=df.loc[va,target].to_numpy();yt=df.loc[te,target].to_numpy()
    pca=PCA(n_components=max(KS),random_state=2403);Ptr=pca.fit_transform(Xtr);Pv=pca.transform(Xv);Pt=pca.transform(Xt);trial=[]
    for k in KS:
        lr=LinearRegression().fit(Ptr[:,:k],ytr);p=lr.predict(Pv[:,:k]);r=float(np.sqrt(np.mean((p-yv)**2)));trial.append((r,k,lr));search.append({'horizon_hours':h,'n_components':k,'validation_rmse_C':r})
    _,bestk,lr=min(trial,key=lambda z:z[0]);pv=lr.predict(Pv[:,:bestk]);pt=lr.predict(Pt[:,:bestk])
    for sp,y,p in [('validation',yv,pv),('test',yt,pt)]:
        e=p-y;metrics.append({'model':'PCR','split':sp,'horizon_hours':h,'selected_components':bestk,'n_samples':len(y),'rmse_C':float(np.sqrt(np.mean(e*e))),'mae_C':float(np.mean(np.abs(e))),'bias_C':float(np.mean(e))})
    print(h,'best k',bestk,'test',metrics[-1]['rmse_C'])
pd.DataFrame(search).to_csv(OUT/'pcr_component_search.csv',index=False);pd.DataFrame(metrics).to_csv(OUT/'pcr_metrics.csv',index=False)
