"""One-step cross-fit prototype-mean regularization, separate evidence control."""
import numpy as np
from research.native_event_semantic_prototype_v1.core import membership,inclusive_scores,update,evidence,loop,EPS
KAPPAS=(1.,4.,16.);BETAS=(.25,.5,.75)

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

def one_step(p,q0,kappa=0.,beta=1.):
 assert not(kappa>0 and beta!=1.),'No stacked shrink+temperature arms'
 L,stats=shrunk_evidence(p,q0,kappa);S=inclusive_scores(L);q=update(q0,beta*S)
 return q,dict(L=L,S=S,prototypes=stats,kappa=kappa,beta=beta)

def arms(p,q0,randomp,wrongp,reflectedp):
 old,oldsteps=loop(p,q0);q1,one=one_step(p,q0);assert np.array_equal(q1,oldsteps[0]['q'])
 out=dict(NativeG6=q0.copy(),Loop2=old,Loop1=q1);tr=dict(Loop1=one,Loop2=oldsteps)
 for k in KAPPAS:
  name='ShrinkK'+str(int(k));out[name],tr[name]=one_step(p,q0,kappa=k)
  for suffix,v in [('Random',randomp),('WrongTime',wrongp),('Reflected',reflectedp)]:out[name+suffix],tr[name+suffix]=one_step(v,q0,kappa=k)
 for b in BETAS:
  name='WeakB'+f'{int(b*100):03d}';out[name],tr[name]=one_step(p,q0,beta=b)
 return out,tr
