"""Whole-interval Bernoulli evidence from the original frozen actionness head."""
import numpy as np

def likelihood(ids,intervals,logits,reliability=None):
 ids=np.asarray(ids,float);z=np.asarray(logits,float);r=np.ones_like(z)if reliability is None else np.asarray(reliability,float);edges=np.r_[ids[0],(ids[:-1]+ids[1:])/2,ids[-1]+1];w=np.diff(edges)*r
 if w.sum()<=1e-12:return np.zeros(len(intervals))
 result=[]
 for a,b in intervals:
  inside=np.maximum(0,np.minimum(edges[1:],b)-np.maximum(edges[:-1],a))/np.diff(edges)
  result.append(float((w*(inside*z-np.logaddexp(0,z))).sum()/w.sum()))
 return np.asarray(result)
def choose(utilities,scores,posterior_sd,constrained=True):
 u=np.asarray(utilities);s=np.asarray(scores);ok=s>s[0]+1e-12
 if constrained:ok&=u>=u[0]-posterior_sd-1e-12
 if not ok.any():return 0
 valid=np.flatnonzero(ok);return int(max(valid,key=lambda i:(s[i],u[i],-i)))
def posterior_sd(ids,q,interval):
 ids=np.asarray(ids,float);a,b=interval;start=ids[:,None];end=ids[None,:]+1;intersection=np.maximum(0,np.minimum(b,end)-np.maximum(a,start));denom=(b-a)+(end-start)-intersection;v=np.divide(intersection,denom,out=np.zeros_like(intersection),where=denom>0);m=float((q*v).sum());return float(np.sqrt(max(0,float((q*(v-m)**2).sum()))))
def roi_reliability(z,permuted=False):
 area=z['roi_mass']-z['roi_enrichment'];coverage=z['permuted_roi_mass']if permuted else z['roi_mass'];return np.clip((coverage-area)/np.maximum(1-area,1e-12),0,1)
