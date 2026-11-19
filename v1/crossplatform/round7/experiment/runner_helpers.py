from pathlib import Path
import numpy as np

def sample_points(row):
    p=Path(row['pointpath']);raw=np.load(p,allow_pickle=False) if p.suffix=='.npy' else np.fromfile(p,np.float32).reshape(-1,4)
    assert raw.ndim==2 and raw.shape[1]>=3 and len(raw)>0 and np.isfinite(raw[:,:3]).all()
    # R5 exact convention: raw float32 and in-place float32 Drone shift.
    raw=np.asarray(raw,np.float32);ix=np.random.default_rng(row['seed']).choice(len(raw),16384,replace=len(raw)<16384)
    xyz=raw[ix,:3].copy()
    if row['platform']=='drone':xyz[:,2]+=1.8
    return np.ascontiguousarray(xyz,np.float32),ix.astype(np.int64),len(raw)
