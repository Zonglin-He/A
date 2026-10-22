"""Conditional event evidence, without labels or model execution."""
import numpy as np
from scipy.special import logsumexp

def word_logits(premax,groups):
    x=np.asarray(premax,dtype=np.float64)
    return np.stack([x[:,:,g].mean(-1)for g in groups],-1)

def conditional_event(premax,verbs,nouns,selected):
    """Remove the observed slotwise nominal response, not a unit-slope ratio.

    For each existing teacher frame and predicate, regress verb logit on the
    mean target-noun logit in the OTHER14 tracked slots. They are calibration
    observations, NOT assumed negative identities or negative actions. Evaluate
    the residual on the held-out target slot. No target-slot fitting or labels.
    """
    v=word_logits(premax,verbs);n=word_logits(premax,nouns).mean(-1)
    keep=np.arange(v.shape[1])!=selected;out=[];diagnostics=[]
    for t in range(len(v)):
        x=n[t,keep];mean=x.mean();scale=max(float(x.std()),1e-12)
        design=np.c_[np.ones(keep.sum()),(x-mean)/scale]
        coefficient=np.linalg.lstsq(design,v[t,keep],rcond=None)[0]
        predicted=np.r_[1.,(n[t,selected]-mean)/scale]@coefficient
        out.append(v[t,selected]-predicted)
        diagnostics.append(dict(rank=int(np.linalg.matrix_rank(design)),noun_scale=scale,extrapolation_z=float((n[t,selected]-mean)/scale),slopes=(coefficient[1]/scale).tolist()))
    return np.stack(out),diagnostics

def full_reweight(q,pairs,score):
    z=np.log(np.maximum(q[pairs[:,0],pairs[:,1]],1e-300))+score
    w=np.exp(z-z.max());w/=w.sum();out=np.zeros_like(q);out[pairs[:,0],pairs[:,1]]=w
    return out
