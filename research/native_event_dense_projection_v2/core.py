"""Reuse old fixed ridge1 projection with new dense trained-semantic feedback."""
import numpy as np
from research.joint_event_v1.core import ridge_predict
from research.native_event_semantic_prototype_v1.core import evidence,inclusive_scores,update

def hidden_features(offsets):
 out=[]
 for o in offsets:
  h=np.asarray(o['hidden'],np.float64)[:,0].copy();h/=np.maximum(np.linalg.norm(h,axis=-1,keepdims=True),1e-30);out.append(h.transpose(1,0,2).reshape(h.shape[1],-1)/np.sqrt(h.shape[0]))
 n=sum(len(x)for x in out);return np.stack([out[i%2][i//2]for i in range(n)])

def position_features(offsets):
 out=[]
 for o in offsets:
  x=np.asarray(o['inputs']['query_time'],np.float64)[:,0].copy();x/=np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-30);out.append(x)
 return np.stack([out[i%2][i//2]for i in range(sum(len(x)for x in out))])

def project(x,L):
 result=np.empty(len(x));stats=[]
 for heldout in (0,1):
  train=np.arange(len(x))%2!=heldout;indices=np.flatnonzero(train);pred,mu,sd=ridge_predict(x,L[train,None],np.arange(len(indices)),indices);v=(pred*sd+mu)[:,0];result[~train]=v[~train];stats.append(dict(heldout=heldout,count=len(indices),training_mean=float(mu[0]),training_std=float(sd[0]),prediction=v))
 return result,stats

def loop(p,q0,x):
 q=q0.copy();steps=[]
 for _ in range(2):
  L,proto=evidence(p,q);projected,fit=project(x,L);S=inclusive_scores(projected);q=update(q0,S);steps.append(dict(q=q.copy(),raw_L=L,projected_L=projected,S=S,fit=fit,prototypes=proto))
 return q,steps
