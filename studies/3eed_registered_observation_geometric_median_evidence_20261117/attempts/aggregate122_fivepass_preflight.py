"""Bounded smoothed-Weiszfeld aggregation of existing aligned seed features.
No network/label/GT/appearance-rule input. No hard outlier or target identity claim.
"""
import numpy as np
VARS=['robust','weight_shuffle','weight_mean','wrong_robust','wrong_mean']
def solve122(A,obs):
 X=np.stack([A]+[x['vis'] for x in obs]).astype(np.float64);valid=np.stack([np.ones(len(A),bool)]+[x['valid'].astype(bool) for x in obs]);n=valid.sum(0);w=valid/n[None];mean=(X*w[:,:,None]).sum(0);out=mean.copy();triple=n==3;history=[];eps=1e-6
 if triple.any():
  XX=X[:,triple];z=mean[triple].copy()
  def objective(t):
   dist=np.linalg.norm(XX-t[None],axis=-1);return np.where(dist>=eps,dist,dist**2/(2*eps)+eps/2).mean(0)
  old=objective(z);history.append(float(old.mean()))
  for _ in range(4):
   dist=np.linalg.norm(XX-z[None],axis=-1);alpha=1/np.maximum(dist,eps);alpha/=alpha.sum(0,keepdims=True);z=(alpha[:,:,None]*XX).sum(0);new=objective(z);assert np.all(new<=old+1e-10);old=new;history.append(float(new.mean()))
  w[:,triple]=alpha;out[triple]=z
 assert np.isfinite(out).all() and np.allclose(w.sum(0),1) and np.all(w>=0)
 return out.astype(np.float32),w,triple,history
