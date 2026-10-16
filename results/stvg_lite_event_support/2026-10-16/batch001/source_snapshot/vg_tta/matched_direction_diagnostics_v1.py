"""GT-only matched interventions and exact physical-duration attribution.

Coordinates are normalized image x/y, not pixel-isotropic distances. No
per-direction clipping, no GT-informed parameter optimization, no TTA method.
"""
import itertools
import numpy as np
import torch
from vg_tta.metrics import interval_from_logits
from vg_tta.st_component_diagnostics_v1 import inclusion


def spatial_directions(predicted, target):
    p=np.asarray(predicted,dtype=np.float64);g=np.asarray(target,dtype=np.float64)
    assert p.shape==g.shape and p.ndim==2 and p.shape[1]==2
    assert np.isfinite(p).all() and np.isfinite(g).all()
    assert ((p>=0)&(p<=1)).all() and ((g>=0)&(g<=1)).all()
    d=g-p
    vectors={'GT_direction':d,'anti_GT':-d,'orthogonal_plus':np.stack([-d[:,1],d[:,0]],1),
             'orthogonal_minus':np.stack([d[:,1],-d[:,0]],1)}
    alpha=np.ones(len(p))
    for v in vectors.values():
        for j in range(2):
            positive=v[:,j]>0;negative=v[:,j]<0
            alpha[positive]=np.minimum(alpha[positive],(1-p[positive,j])/v[positive,j])
            alpha[negative]=np.minimum(alpha[negative],-p[negative,j]/v[negative,j])
    alpha=np.maximum(0,np.minimum(1,alpha))
    # A single shared numerical inward margin, never per-arm clipping.
    alpha[alpha<1]*=1-1e-12
    centers={name:p+alpha[:,None]*v for name,v in vectors.items()}
    lengths=np.linalg.norm(alpha[:,None]*d,axis=1)
    for c in centers.values():
        assert ((c>=-1e-12)&(c<=1+1e-12)).all()
        assert np.allclose(np.linalg.norm(c-p,axis=1),lengths,atol=1e-12)
    return centers,{'alpha':alpha.tolist(),'requested_displacement':np.linalg.norm(d,axis=1).tolist(),
        'actual_displacement':lengths.tolist(),'exact_GT_fraction':float(np.mean(alpha==1)),
        'boundary_shrunk_fraction':float(np.mean(alpha<1)),
        'zero_effect_fraction':float(np.mean(lengths<1e-8)),
        'coordinate_system':'normalized image x/y; equal continuous normalized distance, not identical token displacement'}


def selected_spatial_tokens(centers, grid, fraction=.25):
    """Audit the same deterministic nearest-token rule used by the intervention."""
    h,w=grid;yy,xx=np.meshgrid((np.arange(h)+.5)/h,(np.arange(w)+.5)/w,indexing='ij')
    xy=np.stack([xx.ravel(),yy.ravel()],1).astype(np.float32)
    c=np.asarray(centers,np.float32)
    distance=((xy[None]-c[:,None])**2).sum(-1)
    k=max(1,int(np.ceil(h*w*fraction)))
    chosen=np.argsort(distance,axis=1,kind='stable')[:,:k]
    return {'indices':chosen.tolist(),'K':k,'centroid':xy[chosen].mean(1).tolist()}


def endpoint_directions(native_interval,gt_interval,frame_ids):
    ids=np.asarray(frame_ids,np.int64);lo=float(ids[0]);hi=float(ids[-1]+1)
    p=np.asarray(native_interval,float);g=np.asarray(gt_interval,float);d=g-p
    alpha=1.
    # Shared feasible scale: both endpoints inside bounds, interval length>=1.
    for sign in [1.,-1.]:
        v=sign*d
        for j in range(2):
            if v[j]>0:alpha=min(alpha,(hi-p[j])/v[j])
            if v[j]<0:alpha=min(alpha,(lo-p[j])/v[j])
        dv=v[1]-v[0]
        if dv<0:alpha=min(alpha,(p[1]-p[0]-1)/(-dv))
    alpha=max(0.,min(1.,alpha))
    if alpha<1:alpha*=1-1e-12
    for _ in range(60):
        intervals={'GT_endpoint_direction':p+alpha*d,'anti_GT_endpoint':p-alpha*d}
        masks={k:inclusion(ids,v) for k,v in intervals.items()}
        if all(m.any() for m in masks.values()):break
        alpha*=.5
    else:raise AssertionError('Could not construct nonempty matched contexts')
    assert all(lo<=v[0]+1e-8<v[1]+1e-8<=hi+1e-7 for v in intervals.values())
    return {k:{'mask':masks[k].tolist(),'requested_interval':v.tolist()} for k,v in intervals.items()}, {
        'alpha':alpha,'requested_endpoint_displacement':float(np.linalg.norm(d)),
        'actual_endpoint_displacement':float(alpha*np.linalg.norm(d)),
        'continuous_equal_magnitude':True,'discrete_key_count_may_differ':True}


def equal_length_center_pair(native_indices,gt_interval,frame_ids):
    """Exact count, physical length, and +/- physical center shift on this grid.

If no nonzero symmetric pair is feasible, retain the zero pair and flag it;
never move a boundary independently or silently use a nearest length.
"""
    ids=np.asarray(frame_ids,np.int64);i,j=native_indices;k=j-i+1
    length=int(ids[j]+1-ids[i]);basecenter=(ids[i]+ids[j]+1)/2
    request=(gt_interval[0]+gt_interval[1])/2-basecenter
    candidates={}
    for s in range(len(ids)-k+1):
        e=s+k-1
        if int(ids[e]+1-ids[s])==length:
            delta=float((ids[s]+ids[e]+1)/2-basecenter);candidates[delta]=(s,e)
    assert 0. in candidates
    feasible=[abs(v) for v in candidates if -v in candidates and abs(v)<=abs(request)+1e-9]
    amplitude=max(feasible);direction=1 if request>=0 else -1
    result={}
    for name,delta in [('GT_center_same_length',direction*amplitude),('anti_center_same_length',-direction*amplitude)]:
        s,e=candidates[delta];mask=np.zeros(len(ids),bool);mask[s:e+1]=True
        result[name]={'mask':mask.tolist(),'requested_interval':[int(ids[s]),int(ids[e]+1)]}
    return result,{'K':k,'physical_length':length,'requested_center_shift':float(request),
        'actual_center_shift':float(direction*amplitude),'degenerate':amplitude==0,
        'same_physical_length_and_key_count':True}


def same_length_decode(base_logits,frame_ids,target_length):
    logits=base_logits.detach().to('cpu',torch.float64)
    if logits.ndim==3:assert logits.shape[0]==1;logits=logits[0]
    ids=torch.as_tensor(frame_ids,dtype=torch.int64);n=len(ids)
    durations=ids[None,:]+1-ids[:,None]
    legal=torch.triu(torch.ones((n,n),dtype=torch.bool),diagonal=1 if n>1 else 0)
    legal&=durations==int(target_length)
    assert legal.any(),'Exact duration must exist: intervention prediction uses this same frame grid'
    scores=logits[:,0,None]+logits[None,:,1]
    flat=int(scores.masked_fill(~legal,-torch.inf).argmax());i,j=divmod(flat,n)
    return (int(ids[i]),int(ids[j]+1)),{'candidate_count':int(legal.sum()),'exact':True,'mismatch':0,'indices':[i,j]}


def interval_metrics(interval,gt_interval,frame_ids):
    a,b=map(float,interval);g,h=map(float,gt_interval);assert a<b and g<h
    inter=max(0.,min(b,h)-max(a,g));span=float(frame_ids[-1]+1-frame_ids[0]);assert span>0
    pi=inclusion(frame_ids,interval);gi=inclusion(frame_ids,gt_interval)
    return {'tIoU':inter/(max(b,h)-min(a,g)),'precision':inter/(b-a),'recall':inter/(h-g),
        'length':b-a,'center':(a+b)/2,'length_fraction':(b-a)/span,
        'center_fraction':((a+b)/2-frame_ids[0])/span,'center_error_fraction':abs((a+b-g-h)/2)/span,
        'sample_precision':float((pi&gi).sum()/pi.sum()) if pi.any() else 0.,
        'sample_recall':float((pi&gi).sum()/gi.sum()),'interval':[a,b]}


def signflip_p_greater(values):
    """Exploratory exact paired sign-flip reference; assumes sign exchangeability."""
    values=np.asarray(values,float);assert values.ndim==1 and len(values)<=16
    signs=np.asarray(list(itertools.product([-1.,1.],repeat=len(values))))
    distribution=(signs*values[None]).mean(1)
    return float(np.mean(distribution>=values.mean()-1e-12))


def holm_adjust(pvalues):
    names=sorted(pvalues,key=lambda x:pvalues[x]);out={};previous=0.
    for rank,name in enumerate(names):
        previous=max(previous,min(1.,(len(names)-rank)*pvalues[name]));out[name]=previous
    return out
