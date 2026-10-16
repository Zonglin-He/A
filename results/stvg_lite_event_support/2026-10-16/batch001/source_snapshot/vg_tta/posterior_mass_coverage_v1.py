"""Training-free posterior-mass extent selection on the native sampled grid.

This is not parameter TTA. Inclusion mass is normalized expected frame support,
not an event-coverage guarantee and not identical to boundary uncertainty.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.special import softmax


def inputs(logits, frame_ids):
    if torch.is_tensor(logits):
        z=logits.detach().cpu().double().numpy()
    else:z=np.asarray(logits,dtype=np.float64)
    if z.ndim==3 and z.shape[0]==1:z=z[0]
    ids=np.asarray(frame_ids,dtype=np.int64)
    if z.ndim!=2 or z.shape!=(len(ids),2) or len(ids)==0:
        raise ValueError('Expected finite T x 2 logits and nonempty frame IDs')
    if not np.isfinite(z).all() or (np.diff(ids)<=0).any():
        raise ValueError('Require finite logits and strictly increasing frame IDs')
    return z,ids


def entropy(p):
    p=np.asarray(p,dtype=float);positive=p>0
    return float(-(p[positive]*np.log(p[positive])).sum())


def native_posterior(logits,frame_ids):
    z,ids=inputs(logits,frame_ids);t=len(ids)
    valid=np.triu(np.ones((t,t),bool),int(t>1))
    joint_score=z[:,0,None]+z[None,:,1]
    joint=np.zeros((t,t));joint[valid]=softmax(joint_score[valid])
    # P(s<=t) - P(e<t); inclusive sampled endpoints with strict s<e.
    starts=joint.sum(1);ends=joint.sum(0)
    r=np.cumsum(starts)-np.concatenate(([0.],np.cumsum(ends)[:-1]))
    assert r.min()>=-1e-12 and r.max()<=1+1e-12
    r=np.clip(r,0,1);mass=r/r.sum();h=entropy(mass)
    neff=float(np.exp(h));normalizer=max(int(valid.sum()),1)
    joint_h=entropy(joint[valid])
    raw_endpoint_h=[entropy(softmax(z[:,c])) for c in range(2)]
    valid_endpoint_h=[entropy(starts),entropy(ends)]
    expected_duration=float(np.dot(ends,ids+1)-np.dot(starts,ids))
    return {'inclusion':r,'mass':mass,'joint':joint,'joint_scores':joint_score,
            'legal':valid,'stats':{
                'time_count':t,'log_time_count':float(np.log(t)),
                'inclusion_entropy':h,'normalized_inclusion_entropy':h/np.log(t) if t>1 else 0.,
                'effective_support':neff,'normalized_effective_support':neff/t,
                'joint_entropy':joint_h,'normalized_joint_entropy':joint_h/np.log(normalizer) if normalizer>1 else 0.,
                'raw_endpoint_entropy_mean':float(np.mean(raw_endpoint_h)),
                'normalized_raw_endpoint_entropy':float(np.mean(raw_endpoint_h)/np.log(t)) if t>1 else 0.,
                'valid_endpoint_entropy_mean':float(np.mean(valid_endpoint_h)),
                'posterior_expected_sample_support':float(r.sum()),
                'posterior_expected_physical_duration':expected_duration,
                'posterior_expected_span_fraction':expected_duration/(ids[-1]+1-ids[0]),
                'sampling_gap_CV':float(np.std(np.diff(ids))/np.mean(np.diff(ids))) if t>1 else 0.}}


def cell_widths(frame_ids):
    ids=np.asarray(frame_ids,dtype=float)
    if len(ids)==1:return np.ones(1)
    edges=np.r_[ids[0],(ids[:-1]+ids[1:])/2,ids[-1]+1]
    return np.diff(edges)


def select(logits,frame_ids,tau=.9,physical_mass=False,posterior=None):
    """Shortest physical-duration interval meeting the sampled mass threshold.

    Ties: maximum ORIGINAL native endpoint joint logit among mass-feasible,
    minimum-duration intervals, then ascending (start,end). A subsequent
    unconstrained relocation can violate tau and is not this method.
    """
    if not np.isfinite(tau) or not 0<tau<=1:raise ValueError('tau must lie in (0,1]')
    z,ids=inputs(logits,frame_ids)
    p=native_posterior(z,ids) if posterior is None else posterior
    weights=cell_widths(ids) if physical_mass else np.ones(len(ids))
    raw=p['inclusion']*weights;mass=raw/raw.sum()
    cumulative=np.r_[0.,mass.cumsum()]
    enclosed=cumulative[None,1:]-cumulative[:-1,None]
    duration=ids[None,:]+1-ids[:,None]
    feasible=p['legal']&(enclosed>=float(tau)-1e-12)
    assert feasible.any()
    minimum=int(duration[feasible].min())
    allowed=feasible&(duration==minimum)
    score=np.where(allowed,p['joint_scores'],-np.inf)
    k=int(score.argmax());ij=(k//len(ids),k%len(ids))
    return {'indices':ij,'tau':float(tau),'mass_fraction':float(enclosed[ij]),
            'physical_duration':minimum,'sample_count':ij[1]-ij[0]+1,
            'minimum_length_feasible_count':int(allowed.sum()),'all_feasible_count':int(feasible.sum()),
            'mass_weighting':'midpoint_cell_sensitivity' if physical_mass else 'literal_sample_sum_primary'}


def mass_fraction(mass,indices):
    return float(np.asarray(mass)[indices[0]:indices[1]+1].sum())


def correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12:return None
    return float(np.corrcoef(x,y)[0,1])


def residual_correlation(x,y,controls):
    """Linear partial correlation; rank inputs beforehand for rank-partial."""
    x=np.asarray(x,float);y=np.asarray(y,float);c=np.asarray(controls,float)
    if c.ndim==1:c=c[:,None]
    if len(x)<c.shape[1]+3:return None
    c=np.column_stack((np.ones(len(x)),c))
    return correlation(x-c@np.linalg.lstsq(c,x,rcond=None)[0],y-c@np.linalg.lstsq(c,y,rcond=None)[0])
