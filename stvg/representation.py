"""Native spatial appearance BEFORE textfusion; no separate motion/word head required.

Observers read original projected input tensors without changing arguments/outputs.
The TA adapter retains the model's native spatial grid and time mapping.
"""
from contextlib import contextmanager
import numpy as np

def weights(box,height,width,valid):
    b=np.asarray(box,dtype=np.float64).copy();b[2:]*=1.5
    lo,hi=b[:2]-b[2:]/2,b[:2]+b[2:]/2
    xx,yy=np.arange(width)/width,np.arange(height)/height
    dx=np.maximum(0,np.minimum(xx+1/width,hi[0])-np.maximum(xx,lo[0]))*width
    dy=np.maximum(0,np.minimum(yy+1/height,hi[1])-np.maximum(yy,lo[1]))*height
    out=dy[:,None]*dx[None,:]*np.asarray(valid,bool);fallback=not bool(out.sum()>0)
    if fallback:out=np.asarray(valid,dtype=np.float64).copy()
    assert out.sum()>0
    return out,fallback

def pool(features,support):
    a=np.asarray(features,dtype=np.float64);w=np.asarray(support,dtype=np.float64)
    assert a.shape[:2]==w.shape and np.isfinite(a).all()
    return np.einsum('hwc,hw->c',a,w)/w.sum()

@contextmanager
def observe_native_grid(model):
    observed=[]
    def keep(module,args,kwargs):
        assert not args
        v=kwargs['videos'];observed.append((v.tensors.detach().cpu().clone(),(~v.mask).cpu().clone()))
    hook=model.ground_encoder.register_forward_pre_hook(keep,with_kwargs=True)
    try:yield observed
    finally:hook.remove()

def extract(cap,aux):
    ids=list(map(int,cap['frame_ids']));grid=cap['native_local_grid'];assert len(grid)==2
    dimensions=int(grid[0][0].shape[1]);out=np.empty((len(ids),dimensions));stats=[];seen=[]
    for off,(v,valid)in enumerate(grid):
        x=v.permute(0,2,3,1).numpy();n,h,w,c=x.shape;assert c==dimensions
        for j in range(n):
            t=off+2*j;assert t<len(ids);support,fallback=weights(aux[t],h,w,valid[j].numpy());out[t]=pool(x[j],support);stats.append(dict(frame=t,fallback=bool(fallback),area=float(support.sum()),grid=h*w));seen.append(t)
    assert sorted(seen)==list(range(len(ids)))and np.isfinite(out).all()
    return out,stats
