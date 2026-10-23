"""Frozen one-round common-mean prototype shrinkage kappa16; no optimizer."""
import numpy as np
EPS=1e-6

def membership(q):
 return np.diag(np.cumsum(np.cumsum(q,axis=0)[:,::-1],axis=1)[:,::-1]).copy()

def inclusive_scores(L):
 c=np.r_[0.,np.cumsum(L)];return c[1:][None,:]-c[:-1][:,None]

def evidence(p,q):
 p=np.asarray(p,dtype=np.float64);w=membership(q);L=np.zeros(len(p));stats=[]
 for parity in (0,1):
  train=np.arange(len(p))%2!=parity;held=~train;v=p[train];center=np.where(np.ptp(v,axis=0)==0,v[0],v.mean(0));d=v-center;wp=w[train];wn=1-wp
  # Centering yields EXACT zero for constant class features.
  undefined=bool(wp.sum()==0 or wn.sum()==0)
  plus=center.copy()if undefined else center+(wp[:,None]*d).sum(0)/wp.sum()
  minus=center.copy()if undefined else center+(wn[:,None]*d).sum(0)/wn.sum()
  plus=np.clip(plus,EPS,1-EPS);minus=np.clip(minus,EPS,1-EPS)
  L[held]=(p[held]*np.log(plus/minus)+(1-p[held])*np.log((1-plus)/(1-minus))).sum(1)
  stats.append(dict(heldout_parity=parity,positive_weight=float(wp.sum()),negative_weight=float(wn.sum()),undefined_seed=undefined,positive=plus,negative=minus))
 return L,stats

def update(prior,S):
 if np.all(S[prior>0]==0):return prior.copy()
 take=prior>0;x=np.log(prior[take])+S[take];x-=x.max();out=np.zeros_like(prior);out[take]=np.exp(x);out/=out.sum();return out

def shrunk_evidence(p,q0,kappa):
 assert kappa>=0
 if kappa==0:return evidence(p,q0)
 p=np.asarray(p,dtype=np.float64);w=membership(q0);L=np.zeros(len(p));stats=[]
 for parity in (0,1):
  train=np.arange(len(p))%2!=parity;held=~train;v=p[train];mu0=np.where(np.ptp(v,axis=0)==0,v[0],v.mean(0));d=v-mu0;wp=w[train];wn=1-wp
  plus=mu0+(wp[:,None]*d).sum(0)/(wp.sum()+kappa);minus=mu0+(wn[:,None]*d).sum(0)/(wn.sum()+kappa)
  plus=np.clip(plus,EPS,1-EPS);minus=np.clip(minus,EPS,1-EPS)
  L[held]=(p[held]*np.log(plus/minus)+(1-p[held])*np.log((1-plus)/(1-minus))).sum(1)
  stats.append(dict(heldout_parity=parity,n_train=int(train.sum()),positive_weight=float(wp.sum()),negative_weight=float(wn.sum()),mu0=mu0,positive=plus,negative=minus,kappa=kappa))
 return L,stats

def one_step(p,q0):
 kappa=16.;beta=1.
 assert not(kappa>0 and beta!=1.),'No stacked shrink+temperature arms'
 L,stats=shrunk_evidence(p,q0,kappa);S=inclusive_scores(L);q=update(q0,beta*S)
 return q,dict(L=L,S=S,prototypes=stats,kappa=kappa,beta=beta)
