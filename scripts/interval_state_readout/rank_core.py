"""Calibration-free, candidate-internal vs external event ordering."""
import numpy as np
from scripts.interval_state_readout.action_core import choose,posterior_sd,roi_reliability

def likelihood(ids,intervals,logits,reliability=None):
 """Weighted rank AUC of candidate inside vs outside, without GT labels.

Strictly increasing transformations, including any constant logit offset, leave
this score unchanged. Physical soft membership uses the same saved node cells.
"""
 ids=np.asarray(ids,float);z=np.asarray(logits,float);r=np.ones_like(z)if reliability is None else np.asarray(reliability,float);edges=np.r_[ids[0],(ids[:-1]+ids[1:])/2,ids[-1]+1];width=np.diff(edges);w=width*r;comparison=(z[:,None]>z[None,:]).astype(float)+.5*(z[:,None]==z[None,:]);scores=[]
 for a,b in intervals:
  inside=np.maximum(0,np.minimum(edges[1:],b)-np.maximum(edges[:-1],a))/width;pos=w*inside;neg=w*(1-inside);mass=pos.sum()*neg.sum();scores.append(float(pos@comparison@neg/mass)if mass>1e-12 else .5)
 return np.asarray(scores)
