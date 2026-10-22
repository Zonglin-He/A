"""Backbone-agnostic expected tube-overlap readout; boxes/posterior only.

No hidden, attention, auxiliary head, labels, neural framework or trained
parameters. Quality is a hypothesis-consensus proxy, not a calibrated GT IoU.
"""
import numpy as np

def box_iou(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);alo=a[...,:2]-a[...,2:]/2;ahi=a[...,:2]+a[...,2:]/2;blo=b[...,:2]-b[...,2:]/2;bhi=b[...,:2]+b[...,2:]/2;intersection=np.maximum(0,np.minimum(ahi,bhi)-np.maximum(alo,blo)).prod(-1);union=a[...,2:].prod(-1)+b[...,2:].prod(-1)-intersection;return intersection/np.maximum(union,1e-15)
def consensus(track,hypotheses,valid):
 iou=box_iou(np.asarray(track)[:,None,:],hypotheses);w=np.asarray(valid,float);assert(w.sum(-1)>0).all();return (iou*w).sum(-1)/w.sum(-1)
def expected_overlap(ids,q,interval,quality):
 """Exact posterior expectation with piecewise-constant physical quality.

Constant quality1 exactly reduces to expected temporal IoU. Node cells use
physical midpoints; no event/GT boundaries are involved in quality creation.
"""
 ids=np.asarray(ids,float);quality=np.asarray(quality,float);assert quality.shape==ids.shape and np.isfinite(quality).all()and(quality>=0).all()and(quality<=1+1e-12).all();edges=np.r_[ids[0],(ids[:-1]+ids[1:])/2,ids[-1]+1];cum=np.r_[0,np.cumsum(np.diff(edges)*quality)];i,j=np.nonzero(q);mass=q[i,j];s=ids[i];e=ids[j]+1;a,b=interval;lo=np.maximum(a,s);hi=np.minimum(b,e);num=np.maximum(0,np.interp(hi,edges,cum)-np.interp(lo,edges,cum));inter=np.maximum(0,hi-lo);union=(b-a)+(e-s)-inter;return float(mass@(num/union))
def select(scores,temporal_utilities):
 scores=np.asarray(scores,float);u=np.asarray(temporal_utilities,float)
 # Protect incumbent under exact or numerical equality.
 best=float(scores.max());eligible=np.flatnonzero(scores>=best-1e-12);return int(max(eligible,key=lambda i:(u[i],-i)))
