"""Shared-covariance Gaussian episode feedback in frozen semantic coordinates."""
import numpy as np
from research.native_event_semantic_prototype_v1.core import membership,inclusive_scores,update

def oas(v):
 # Same modified Eq23 as sklearn1.9.0 _oas; exactflat values have zero covariance.
 center=np.where(np.ptp(v,axis=0)==0,v[0],v.mean(0));x=v-center;n,C=v.shape;S=x.T@x/n;mu=np.trace(S)/C
 if C==1:return S,0.
 if mu==0:return S,1.
 alpha=np.mean(S**2);den=(n+1)*(alpha-mu**2/C);rho=1. if den==0 else min((alpha+mu**2)/den,1.)
 cov=(1-rho)*S;cov.flat[::C+1]+=rho*mu;return cov,float(rho)

def covariances(p,mode='full'):
 result=[]
 for parity in (0,1):
  v=p[np.arange(len(p))%2!=parity];S,rho=oas(v)
  if mode=='diagonal':S=np.diag(np.diag(S))
  elif mode=='isotropic':S=np.eye(S.shape[0])*np.trace(S)/S.shape[0]
  else:assert mode=='full'
  result.append((S,rho))
 return result

def evidence(p,q,covs):
 w=membership(q);L=np.zeros(len(p));stats=[]
 for parity in (0,1):
  train=np.arange(len(p))%2!=parity;held=~train;v=p[train];center=np.where(np.ptp(v,axis=0)==0,v[0],v.mean(0));d=v-center;wp=w[train];wn=1-wp;undefined=bool(wp.sum()==0 or wn.sum()==0);plus=center.copy()if undefined else center+(wp[:,None]*d).sum(0)/wp.sum();minus=center.copy()if undefined else center+(wn[:,None]*d).sum(0)/wn.sum();S,rho=covs[parity];flat=bool(np.trace(S)==0)
  direction=np.zeros(p.shape[1])if flat else np.linalg.solve(S,plus-minus);L[held]=(p[held]-(plus+minus)/2)@direction
  stats.append(dict(heldout_parity=parity,positive=plus,negative=minus,positive_weight=float(wp.sum()),negative_weight=float(wn.sum()),undefined_seed=undefined,flat_covariance=flat,shrinkage=rho,covariance=S))
 return L,stats

def loop(p,q0,mode='full'):
 p=np.asarray(p,dtype=np.float64);covs=covariances(p,mode);q=q0.copy();steps=[]
 for _ in range(2):
  L,stats=evidence(p,q,covs);S=inclusive_scores(L);q=update(q0,S);steps.append(dict(q=q.copy(),L=L,S=S,statistics=stats))
 return q,steps
