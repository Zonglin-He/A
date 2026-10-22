"""Whole-interval state evidence with ordinal, two-evidence dominance.

No GT, learned head, scalar reward mixing, RGB or model execution. The cached
motion memory is already frozen text-conditioned Source output.
"""
import numpy as np
from scipy.stats import rankdata

def detrend(x,t):
 x=np.asarray(x,float);x=x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12);u=(t-t[0])/max(t[-1]-t[0],1e-12);Q=np.column_stack([np.ones(len(t)),u,u*u]);return x-Q@np.linalg.lstsq(Q,x,rcond=None)[0]
def segment_gain(x,ids,intervals):
 """Fraction of feature energy explained by prefix/event/suffix means.

Voronoi cells on the physical frame axis permit partial-node membership. This
is a descriptive state-separation statistic, not an event probability.
"""
 x=np.asarray(x,float);ids=np.asarray(ids,float);edges=np.r_[ids[0],(ids[:-1]+ids[1:])/2,ids[-1]+1];w=np.diff(edges);mu=(w[:,None]*x).sum(0)/w.sum();x=x-mu;total=float((w[:,None]*x*x).sum());out=[]
 for start,end in intervals:
  cuts=[(edges[0],start),(start,end),(end,edges[-1])];between=0.
  for a,b in cuts:
   mass=np.maximum(0,np.minimum(edges[1:],b)-np.maximum(edges[:-1],a));n=mass.sum()
   if n>1e-12:between+=float(np.square((mass[:,None]*x).sum(0)).sum()/n)
  out.append(between/total if total>1e-12 else 0.)
 z=np.asarray(out);assert np.isfinite(z).all()and z.min()>=-1e-8 and z.max()<=1+1e-8;return np.clip(z,0,1)
def rank(x):return (rankdata(x,method='average')-.5)/len(x)
def choose(utilities,hidden,spatial,mode='joint'):
 """Only accept candidates that strictly improve all available criteria.

Candidate0 is incumbent. Utility is a tie-breaker, not an additive penalty.
No reward temperature or mixing weight is selected.
"""
 u=np.asarray(utilities);h=np.asarray(hidden);s=np.asarray(spatial)
 if mode=='hidden':ok=h>h[0]+1e-10;score=rank(h)-rank(h)[0]
 elif mode=='spatial':ok=s>s[0]+1e-10;score=rank(s)-rank(s)[0]
 else:
  assert mode=='joint';ok=(h>h[0]+1e-10)&(s>s[0]+1e-10);score=np.minimum(rank(h)-rank(h)[0],rank(s)-rank(s)[0])
 if not ok.any():return 0
 ids=np.flatnonzero(ok);return int(max(ids,key=lambda i:(score[i],u[i],-i)))
