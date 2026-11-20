"""Consensus of box intersections with the three lines through a fixed physical anchor."""
import numpy as np

def axes(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])

def chords(anchor,boxes):
    """Returns signed entrance/exit distances along the anchor's axes for boxes containing its center."""
    co,si=np.cos(boxes[:,6]),np.sin(boxes[:,6])
    delta=anchor[:3]-boxes[:,:3]
    local=np.column_stack((co*delta[:,0]+si*delta[:,1],-si*delta[:,0]+co*delta[:,1],delta[:,2]))
    half=boxes[:,3:6]/2
    inside=(np.abs(local)<=half+1.e-10).all(1)
    bounds=[];base=axes(anchor[6])
    for u in base:
        direc=np.column_stack((co*u[0]+si*u[1],-si*u[0]+co*u[1],np.full(len(boxes),u[2])))
        live=np.abs(direc)>1.e-12
        t0=np.divide(-half-local,direc,out=np.zeros_like(direc),where=live)
        t1=np.divide( half-local,direc,out=np.zeros_like(direc),where=live)
        low=np.where(live,np.minimum(t0,t1),-np.inf).max(1)
        high=np.where(live,np.maximum(t0,t1),np.inf).min(1)
        bounds.append(np.column_stack((low,high)))
    return np.stack(bounds,axis=1),inside

def weighted_median(values,weights,prefer):
    order=np.lexsort((weights,values))
    v=values[order]
    cumulative=np.cumsum(weights[order],dtype=np.longdouble)
    half=cumulative[-1]/2
    # Each view has half the mass. Floating point error can place a true tie
    # on either side of searchsorted, including immediately before its index.
    tol=8*np.finfo(float).eps*cumulative[-1]
    ties=np.flatnonzero(np.abs(cumulative[:-1]-half)<=tol)
    if len(ties):
        return float(np.clip(prefer,v[ties[0]],v[ties[-1]+1]))
    return float(v[np.searchsorted(cumulative,half)])

def estimate(anchor,intervals,weight):
    if weight.sum()==0:return anchor.copy()
    use=weight>0
    endpoints=np.array([[weighted_median(intervals[use,axis,j],weight[use],
                          (j-.5)*anchor[axis+3]) for j in (0,1)] for axis in range(3)])
    dims=endpoints[:,1]-endpoints[:,0]
    if np.min(dims)<=1.e-9:return anchor.copy()
    return np.r_[anchor[:3]+endpoints.mean(1) @ axes(anchor[6]),dims,anchor[6]]

def blend_views(target_weight,full_weight,inside,n):
    combined=np.zeros(len(inside))
    for part,w in ((slice(None,n),target_weight),(slice(n,None),full_weight)):
        masked=w*inside[part]
        if masked.sum()>0:combined[part]=masked/masked.sum()
    if combined.sum()>0:combined/=combined.sum()
    return combined
