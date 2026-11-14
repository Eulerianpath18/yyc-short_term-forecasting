"""Memory-efficient, leakage-safe common sequence dataset for MLP-vs-TCN."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CLEAN_DIR=ROOT/"spatial_extension"/"data"/"clean"
AUDIT_DIR=ROOT/"results"/"sanity_checks"
STATIONS=("yyc","bow_valley","sundre","drumheller","claresholm","lacombe")
# Keep the sequence compact and robust: the same 4 variables are reasonably complete at all stations.
BASE_VARS=("temperature","dew_point","relative_humidity","wind_speed")
LOOKBACK=24; HORIZONS=(1,6,12,24,48)
TRAIN_END=pd.Timestamp("2024-01-01"); VAL_START=pd.Timestamp("2024-01-01"); VAL_END=pd.Timestamp("2025-01-01"); TEST_START=pd.Timestamp("2025-01-01"); TEST_END=pd.Timestamp("2026-01-01")

@dataclass
class SequenceSource:
    raw: np.ndarray                 # full hourly matrix (time, value_channel), NaNs allowed
    timestamps: pd.DatetimeIndex
    channel_names: list[str]
    origins: np.ndarray             # row index of each valid forecast origin
    origin_timestamps: pd.DatetimeIndex
    targets: np.ndarray             # N,5
    static: np.ndarray              # N,4


def _station(slug):
    p=CLEAN_DIR/f"{slug}_hourly.csv";d=pd.read_csv(p);d["timestamp"]=pd.to_datetime(d.timestamp,errors="raise");d=d.sort_values("timestamp").reset_index(drop=True)
    if d.timestamp.duplicated().any(): raise ValueError(f"duplicate timestamps: {p}")
    if not (d.timestamp.diff().dropna()==pd.Timedelta(hours=1)).all(): raise ValueError(f"non-hourly grid: {p}")
    return d


def build(write_audit=True):
    frames={s:_station(s) for s in STATIONS}; ref=frames["yyc"].timestamp
    for s,d in frames.items():
        if not ref.equals(d.timestamp): raise ValueError(f"timestamp grid differs at {s}")
    names=[]; cols=[]
    for s,d in frames.items():
        for v in BASE_VARS:
            names.append(f"{s}__{v}"); cols.append(pd.to_numeric(d[v],errors="coerce").to_numpy(np.float32))
    raw=np.column_stack(cols).astype(np.float32); target=pd.to_numeric(frames["yyc"].temperature,errors="coerce").to_numpy(np.float32); ts=pd.DatetimeIndex(ref)
    origins=np.arange(LOOKBACK-1,len(ts)-max(HORIZONS),dtype=np.int64)
    y=np.column_stack([target[origins+h] for h in HORIZONS]).astype(np.float32); current=target[origins]
    keep=np.isfinite(y).all(1)&np.isfinite(current);origins=origins[keep];y=y[keep];ots=ts[origins]
    hour=ots.hour.to_numpy(float);doy=ots.dayofyear.to_numpy(float);static=np.column_stack([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25)]).astype(np.float32)
    src=SequenceSource(raw,ts,names,origins,ots,y,static)
    if write_audit: alignment_audit(src)
    return src


def split_masks(src):
    target48=src.origin_timestamps+pd.to_timedelta(48,unit="h")
    return {"train":np.asarray(target48<TRAIN_END),"validation":np.asarray((src.origin_timestamps>=VAL_START)&(target48<VAL_END)),"test":np.asarray((src.origin_timestamps>=TEST_START)&(target48<TEST_END))}


def fit_preprocessor(src,train_mask):
    # Only source rows actually available by the latest training origin are used.
    max_row=int(src.origins[train_mask].max()); rows=src.raw[:max_row+1];med=np.nanmedian(rows,axis=0)
    if np.isnan(med).any(): raise ValueError("entirely missing training channel")
    imp=np.where(np.isfinite(rows),rows,med[None,:]);mean=imp.mean(0);std=imp.std(0);std[std==0]=1
    return {"median":med.astype(np.float32),"mean":mean.astype(np.float32),"std":std.astype(np.float32),"max_training_source_row":np.array(max_row,dtype=np.int64)}


def transformed_source(src,prep):
    observed=np.isfinite(src.raw);imp=np.where(observed,src.raw,prep["median"][None,:]);z=(imp-prep["mean"][None,:])/prep["std"][None,:]
    return np.concatenate([z.astype(np.float32),observed.astype(np.float32)],axis=1)


def alignment_audit(src):
    AUDIT_DIR.mkdir(parents=True,exist_ok=True);rows=[]
    for i in [0,len(src.origins)//2,len(src.origins)-1]:
        o=int(src.origins[i]);t=src.origin_timestamps[i]
        for lag in [0,1,6,23]:
            r=o-lag
            for c,name in enumerate(src.channel_names[:8]):
                exp=src.raw[r,c]
                # Direct row-index check plus timestamp check.
                ok=src.timestamps[r]==t-pd.Timedelta(hours=lag)
                rows.append({"origin":t,"kind":"input","offset_hours":-lag,"channel":name,"source_time":src.timestamps[r],"value":exp,"pass":bool(ok)})
        for q,h in enumerate(HORIZONS):
            r=o+h;ok=(src.timestamps[r]==t+pd.Timedelta(hours=h)) and np.isclose(src.targets[i,q],src.raw[r,0],equal_nan=True)
            rows.append({"origin":t,"kind":"target","offset_hours":h,"channel":"yyc_temperature","source_time":src.timestamps[r],"value":src.targets[i,q],"pass":bool(ok)})
    a=pd.DataFrame(rows);a.to_csv(AUDIT_DIR/"sequence_alignment_check.csv",index=False)
    if not a["pass"].all():raise AssertionError("sequence alignment failed")


def summary(src):
    rows=[]
    for sp,m in split_masks(src).items():
        # Missingness is measured on source rows up through split origins, not materialized windows.
        rows.append({"split":sp,"n_samples":int(m.sum()),"first_origin":src.origin_timestamps[m].min(),"last_origin":src.origin_timestamps[m].max()})
    return pd.DataFrame(rows)
