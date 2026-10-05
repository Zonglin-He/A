"""Fixed, label-free input transforms and consistency; no optimizer or gates."""
import re
import numpy as np


def temporal_specs(ids, native):
    ids=np.asarray(ids,dtype=np.int64);n=len(ids)
    assert n>=4 and np.all(np.diff(ids)>0)
    gap=max(1,int(np.median(np.diff(ids))));pad=max(2,int(np.ceil(n/8)))
    delta=pad*gap;origin=int(ids[0])
    shifted=np.r_[np.arange(pad)*gap,ids-origin+delta].tolist()
    start=int(np.searchsorted(ids,native[0]));end=int(np.searchsorted(ids,native[1],side='left'))
    assert 0<=start<end<=n
    a=start//2;b=end+(n-end+1)//2
    if b-a<4:a,b=0,n
    assert ids[a]<=native[0] and ids[b-1]+1>=native[1]
    return dict(shift=dict(ids=shifted,pad=pad,delta=delta,origin=origin,effective=True),
                crop=dict(ids=(ids[a:b]-ids[a]).tolist(),a=a,b=b,origin=int(ids[a]),effective=(a>0 or b<n)))


def temporal_pixels(frames, spec, name):
    if name=='shift':return np.concatenate([np.repeat(frames[:1],spec['pad'],axis=0),frames],axis=0)
    if name=='crop':return frames[spec['a']:spec['b']].copy()
    raise ValueError(name)


def inverse_interval(interval,spec,name,ids,native):
    offset=spec['origin']-(spec['delta'] if name=='shift' else 0)
    raw=[int(x)+offset for x in interval]
    mapped=[max(int(ids[0]),raw[0]),min(int(ids[-1])+1,raw[1])]
    invalid=mapped[1]<=mapped[0]
    return dict(interval=list(native) if invalid else mapped,raw=raw,
                clipped=mapped!=raw,invalid=invalid,fallback=invalid)


def consensus(native,shift,crop):
    out=np.median(np.array([native,shift,crop],dtype=np.int64),axis=0).astype(int).tolist()
    assert out[0]<out[1]
    return out


def spatial_pixels(frames,name):
    if name=='flip':return frames[:,:,::-1,:].copy()
    if name=='dim95':return np.rint(frames.astype(np.float64)*.95).astype(np.uint8)
    raise ValueError(name)


def inverse_boxes(boxes,name):
    out=np.array(boxes,dtype=np.float64,copy=True)
    if name=='flip':out[...,0]=1-out[...,0]
    return out


def per_frame_iou(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    aa=np.c_[a[:,:2]-a[:,2:]/2,a[:,:2]+a[:,2:]/2]
    bb=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]
    inter=np.maximum(np.minimum(aa[:,2:],bb[:,2:])-np.maximum(aa[:,:2],bb[:,:2]),0).prod(1)
    union=np.maximum(aa[:,2:]-aa[:,:2],0).prod(1)+np.maximum(bb[:,2:]-bb[:,:2],0).prod(1)-inter
    return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)


def consistency(a,b,ids,native):
    val=per_frame_iou(a,b);ids=np.asarray(ids)
    mask=(ids>=native[0])&(ids<native[1]);assert mask.any()
    return dict(event=float(val[mask].mean()),full=float(val.mean()),event_frames=int(mask.sum()))


def directional_query(text):
    return bool(re.search(r'\b(left|right|leftmost|rightmost|clockwise|counterclockwise)\b',text.lower()))
