"""Geometric median of at most3 existing aligned observations, no learned/GT input.
Three vectors span a triangle; Fermat barycentric formula or >=120-degree vertex.
Fewer than3 valid vectors keep their equal-weight mean. No iterative fit/strength.
"""
import numpy as np
VARS=['robust','weight_shuffle','weight_mean','wrong_robust','wrong_mean']
def exact3(X):
 # [3,N,D]. Barycentric coordinates a*csc(A+pi/3), with sidea=|BC|.
 n=X.shape[1];side=np.stack([np.linalg.norm(X[1]-X[2],axis=-1),np.linalg.norm(X[2]-X[0],axis=-1),np.linalg.norm(X[0]-X[1],axis=-1)]);w=np.ones((3,n))/3;duplicate=side.min(0)<=1e-12
 for k in range(3):
  ok=duplicate&(side[k]<=1e-12)&(side.max(0)>1e-12);w[:,ok]=0;w[(k+1)%3,ok]=.5;w[(k+2)%3,ok]=.5
 non=~duplicate
 if non.any():
  s=side[:,non];cos=np.stack([(s[1]**2+s[2]**2-s[0]**2)/(2*s[1]*s[2]),(s[2]**2+s[0]**2-s[1]**2)/(2*s[2]*s[0]),(s[0]**2+s[1]**2-s[2]**2)/(2*s[0]*s[1])]);cos=np.clip(cos,-1,1);corner=cos.min(0)<=-.5;ww=np.zeros_like(s)
  if corner.any():ww[cos[:,corner].argmin(0),np.flatnonzero(corner)]=1
  interior=~corner
  if interior.any():
   c=cos[:,interior];den=.5*np.sqrt(np.maximum(0,1-c*c))+np.sqrt(3)/2*c;v=s[:,interior]/np.maximum(den,1e-15);ww[:,interior]=v/v.sum(0,keepdims=True)
  w[:,non]=ww
 out=(w[:,:,None]*X).sum(0);return out,w

def solve122(A,obs):
 X=np.stack([A]+[x['vis'] for x in obs]).astype(np.float64);valid=np.stack([np.ones(len(A),bool)]+[x['valid'].astype(bool) for x in obs]);n=valid.sum(0);w=valid/n[None];mean=(X*w[:,:,None]).sum(0);out=mean.copy();triple=n==3;history=[]
 if triple.any():
  XX=X[:,triple];z,alpha=exact3(XX);old=np.linalg.norm(XX-mean[triple][None],axis=-1).mean(0);new=np.linalg.norm(XX-z[None],axis=-1).mean(0);assert np.all(new<=old+1e-9);history=[float(old.mean()),float(new.mean())];w[:,triple]=alpha;out[triple]=z
 assert np.isfinite(out).all() and np.allclose(w.sum(0),1) and np.all(w>=0)
 return out.astype(np.float32),w,triple,history
