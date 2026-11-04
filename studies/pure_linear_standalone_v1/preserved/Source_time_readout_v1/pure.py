"""Only eight expert boxes, their physical times, and requested video frames."""
import numpy as np

def interpolate(anchor_times,anchor_xyxy,frame_ids):
    t=np.asarray(anchor_times,dtype=np.float64)
    b=np.asarray(anchor_xyxy,dtype=np.float64)
    f=np.asarray(frame_ids,dtype=np.float64)
    assert t.shape==(8,)and b.shape==(8,4)and np.all(np.diff(t)>0)
    assert np.isfinite(b).all()and np.all(b[:,2:]>b[:,:2])and np.all(np.diff(f)>0)
    # np.interp holds the nearest endpoint on both tails.
    out=np.column_stack([np.interp(f,t,b[:,k])for k in range(4)])
    for j,at in enumerate(t):out[f==at]=b[j]
    return out
