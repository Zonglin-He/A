"""Lower bound on physical size from two sets of observed returns, evaluated at fixed seeds."""
import numpy as np
from scipy.spatial import cKDTree, ConvexHull, QhullError
from scipy.spatial.distance import pdist
from scene_faces import split_returns


def diameter(xy):
    if len(xy)<2:
        return 0.
    if len(xy)>24:
        try:
            xy=xy[ConvexHull(xy).vertices]
        except QhullError:
            axis=np.argmax(np.ptp(xy,axis=0))
            return float(np.linalg.norm(xy[xy[:,axis].argmax()]-xy[xy[:,axis].argmin()]))
    return float(np.sqrt(pdist(xy,'sqeuclidean').max()))


def observed_extent(x, parity):
    # A shared lower bound remains meaningful when one side of an object is invisible.
    halves=(x[parity],x[~parity])
    if min(map(len,halves))<2:
        return np.zeros(2)
    return np.min([[diameter(h[:,:2]),np.ptp(h[:,2])] for h in halves],axis=0)


def predicted_extent(sizes):
    horizontal=np.hypot(sizes[...,0],sizes[...,1])
    return np.stack((horizontal,sizes[...,2]),axis=-1)


def compatibility(measured, predicted):
    # An object can be much larger than the observed returns. Penalize only a violated lower bound.
    q=np.ones_like(predicted,dtype=float)
    np.divide(predicted,measured,out=q,where=(predicted>0)&(measured>0))
    return np.minimum(q,1.).prod(axis=-1)


def prepare(seg,word):
    x=seg['xyz'].astype(float); parity=split_returns(x)
    xyz=word['xyz'].astype(float); sz=word['extent'].astype(float)
    # Duplicate seed coordinates describe one physical location; merge them before lookup.
    seeds,inv=np.unique(xyz,axis=0,return_inverse=True)
    n_nonpositive=int((sz<=0).sum())
    summed=np.zeros((len(seeds),)+sz.shape[1:])
    # A negative regression output has no metric length. Apply the production box size floor.
    np.add.at(summed,inv,np.maximum(sz,1.e-6))
    sz=summed/np.bincount(inv)[:,None,None]
    prediction=predicted_extent(sz)
    near=cKDTree(seeds).query(x)[1]
    # Use the same scene and class-noun interventions, with local seed correspondence removed.
    global_proto=np.median(prediction,axis=0)
    answer={}; coverage=[]
    for k,lab in enumerate(seg['labels']):
        base=seg['p'+str(k)]
        n,nclasses=base.shape
        local=np.ones((n,nclasses))
        global_like=local.copy()
        ids=np.flatnonzero(lab>=0)
        if len(ids):
            order=ids[np.argsort(lab[ids],kind='stable')]
            groups=np.split(order,np.flatnonzero(np.diff(lab[order]))+1)
            for group in groups:
                c=int(lab[group[0]])
                extent=observed_extent(x[group],parity[group])
                if not extent.any():continue
                seed_ids=np.unique(near[group])
                proto=np.median(prediction[seed_ids],axis=0)
                local[c,:-1]=compatibility(extent,proto)
                global_like[c,:-1]=compatibility(extent,global_proto)
        wrong=local.copy()
        wrong[:,:-1]=np.roll(local[:,:-1],1,axis=1)
        for kind,weights in [('local',local),('global',global_like),('wrong',wrong)]:
            unnorm=base*weights
            answer[kind+str(k)]=unnorm/unnorm.sum(1,keepdims=True)
        gap=np.ptp(local[:,:-1],axis=1)
        coverage.append([n,int((gap>1e-7).sum()),float(gap.sum()),int((base.argmax(1)!=answer['local'+str(k)].argmax(1)).sum())])
    answer['coverage']=np.asarray(coverage)
    answer['nonpositive']=np.asarray(n_nonpositive,dtype=np.int64)
    return answer
