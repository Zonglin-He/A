"""Two boundaries assign all frames to before/event/after, with native prior."""
import numpy as np
from research.native_event_semantic_prototype_v1.core import update,EPS

def state_membership(q):
 pre=np.cumsum(q.sum(1)[::-1])[::-1]-q.sum(1);post=np.cumsum(q.sum(0))-q.sum(0);inside=1-pre-post
 return np.c_[pre,inside,post]

def evidence(p,q,mode='ordered'):
 weights=state_membership(q);LL=np.zeros((len(p),3));stats=[]
 for parity in (0,1):
  train=np.arange(len(p))%2!=parity;held=~train;v=p[train];center=np.where(np.ptp(v,axis=0)==0,v[0],v.mean(0));d=v-center;mu=[];undefined=[]
  for state in range(3):
   w=weights[train,state];fallback=bool(w.sum()<=0);m=center.copy()if fallback else center+(w[:,None]*d).sum(0)/w.sum();mu.append(np.clip(m,EPS,1-EPS));undefined.append(fallback)
  mu=np.stack(mu);scores=(p[held,:,None]*np.log(mu.T[None])+(1-p[held,:,None])*np.log1p(-mu.T[None])).sum(1)
  if mode=='swap':scores=scores[:,[2,1,0]]
  if mode=='unordered':
   background=np.logaddexp(scores[:,0],scores[:,2])-np.log(2);scores[:,0]=background;scores[:,2]=background
  LL[held]=scores;stats.append(dict(heldout_parity=parity,prototypes=mu,undefined_seed=undefined))
 # Subtract common before-state density at EVERYframe: fullvideo likelihood unchanged up to a candidate-independent constant.
 rel=LL-LL[:,0,None];event=np.r_[0.,np.cumsum(rel[:,1])];after=np.r_[0.,np.cumsum(rel[:,2])];S=event[1:][None,:]-event[:-1][:,None]+after[-1]-after[1:][None,:]
 return S,dict(loglikelihood=LL,prototypes=stats)

def loop(p,q0,mode='ordered'):
 q=q0.copy();steps=[]
 for _ in range(2):
  S,stats=evidence(p,q,mode);q=update(q0,S);steps.append(dict(q=q.copy(),S=S,statistics=stats))
 return q,steps
