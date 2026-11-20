"""Single-frame face metrology: keep unobserved anchor faces; use independent return halves."""
import numpy as np
from physical_evidence import axes


def split_returns(x):
    values=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32).astype(np.uint64)
    bits=values[:,0]*np.uint64(73856093)^values[:,1]*np.uint64(19349663)^values[:,2]*np.uint64(83492791)
    bits^=bits>>np.uint64(16)
    bits*=np.uint64(0x45d9f3b)
    bits^=bits>>np.uint64(16)
    return (bits & 1).astype(bool)


def bottom(box,x,ground,split):
    initial=box[2]-box[5]/2
    if not np.isfinite(ground).all():return initial
    off=x[:,:2]-box[:2]
    local=off@axes(box)[:2,:2].T
    half=box[3:5]/2
    ring=(abs(local)>half).any(1) & (abs(local)<=2*half).all(1)
    terrain=abs(x[:,2]-x[:,:2]@ground[:2]-ground[2])<=.20
    estimates=[]
    for side in (False,True):
        mask=ring&terrain&(split==side)
        if mask.sum()<3:return initial
        xy=off[mask]
        angles=np.sort(np.arctan2(xy[:,1],xy[:,0]))
        if np.diff(np.r_[angles,angles[0]+2*np.pi]).max()>=np.pi:return initial
        v,_,rank,_=np.linalg.lstsq(np.c_[xy,np.ones(len(xy))],x[mask,2],rcond=None)
        if rank<3 or np.linalg.norm(v[:2])>np.tan(np.deg2rad(25.)):return initial
        estimates.append(v[2])
    value=np.clip(initial,min(estimates),max(estimates))
    return value if value<box[2]+box[5]/2 else initial


def weighted_median(x,w):
    order=np.argsort(x,kind='stable')
    return x[order[np.searchsorted(np.cumsum(w[order]),w.sum()/2)]]


def geometry(anchor,x,split,labels,parts,i,j,ground_bottom,conditional):
    axes3=axes(anchor)
    local=(x-anchor[:3])@axes3.T
    half=anchor[3:6]/2
    estimates=[];weights=[]
    for level,t,r,b in parts:
        if not len(t.index) or not len(r.index):continue
        mass=t.assign[i]*(b@r.assign[j] if conditional else b.sum(1))
        for g in np.flatnonzero(mass>0):
            ids=labels[level]==t.index[g]
            first=local[ids & split];second=local[ids & ~split]
            if not len(first) or not len(second):continue
            mins=np.maximum(first.min(0),second.min(0))
            maxs=np.minimum(first.max(0),second.max(0))
            if np.any(mins>maxs):continue
            estimates.append([min(-half[0],mins[0]),min(-half[1],mins[1]),
                              max(half[0],maxs[0]),max(half[1],maxs[1]),max(half[2],maxs[2])])
            weights.append(mass[g])
    if estimates:
        values=np.asarray(estimates);w=np.asarray(weights)
        a,b,c,d,top=[weighted_median(column,w) for column in values.T]
    else:a,b,c,d,top=-half[0],-half[1],half[0],half[1],half[2]
    lo=np.array([a,b,ground_bottom-anchor[2]])
    hi=np.array([c,d,top])
    return np.r_[anchor[:3]+((lo+hi)/2)@axes3,hi-lo,anchor[6]]


def only_ground(anchor,z):
    b=anchor.copy();top=anchor[2]+anchor[5]/2
    b[2]=(top+z)/2;b[5]=top-z
    return b
