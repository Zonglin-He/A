"""Per-query temporal probability state; no learned or pretrained network."""
import numpy as np
import torch
from scipy.special import softmax

class Policy:
 def __init__(self,p0,target):
  self.p0=np.asarray(p0,dtype=np.float64);self.target=np.asarray(target,dtype=np.float64);self.mask=self.p0>0;assert np.array_equal(self.mask,self.target>0);assert abs(self.p0.sum()-1)<1e-10 and abs(self.target.sum()-1)<1e-10;self.n=len(p0);self.i,self.j=np.where(self.mask)
  self.l0=torch.from_numpy(np.log(self.p0[self.mask]));self.tf=torch.from_numpy(self.target[self.mask]);self.lt=torch.log(self.tf);self.ii=torch.from_numpy(self.i);self.jj=torch.from_numpy(self.j);self.eta=torch.nn.Parameter(torch.zeros(2,self.n,dtype=torch.float64))
  full=np.full(self.mask.shape,-1e30);full[self.mask]=np.log(self.p0[self.mask]);self.full=torch.from_numpy(full)
  self.ts=self.target.sum(1);self.te=self.target.sum(0);ce=np.divide(self.target,self.ts[:,None],out=np.zeros_like(self.target),where=self.ts[:,None]>0);cs=np.divide(self.target.T,self.te[:,None],out=np.zeros_like(self.target),where=self.te[:,None]>0);self.ce=torch.from_numpy(ce);self.cs=torch.from_numpy(cs)
  self.lce=torch.from_numpy(np.log(ce,out=np.zeros_like(ce),where=ce>0));self.lcs=torch.from_numpy(np.log(cs,out=np.zeros_like(cs),where=cs>0))
 def logflat(self):return torch.log_softmax(self.l0+self.eta[0,self.ii]+self.eta[1,self.jj],0)
 def probability(self):
  if not bool(self.eta.detach().any()):return self.p0.copy()
  q=np.zeros_like(self.p0);q[self.mask]=self.logflat().detach().exp().numpy();return q
 def conditional_terms(self):
  ls=torch.log_softmax(self.full.T+self.eta[0][None],1);le=torch.log_softmax(self.full+self.eta[1][None],1)
  return (self.cs*(self.lcs-ls)).sum(1),(self.ce*(self.lce-le)).sum(1)
 def loss(self,mode,pairs=None):
  if mode=='DirectJoint':return .5*(self.tf*(self.lt-self.logflat())).sum()
  ks,ke=self.conditional_terms()
  if mode=='DirectConditional':return .5*(ks@torch.from_numpy(self.te)+ke@torch.from_numpy(self.ts))
  assert pairs is not None
  return .5*(ks[torch.from_numpy(pairs[:,1])].mean()+ke[torch.from_numpy(pairs[:,0])].mean())
 def manual_gradient(self,mode,pairs=None):
  eta=self.eta.detach().numpy();l0=self.full.numpy()
  if mode=='DirectJoint':
   q=self.probability();return .5*np.stack([q.sum(1)-self.ts,q.sum(0)-self.te])
  ps=softmax(l0.T+eta[0][None],axis=1);pe=softmax(l0+eta[1][None],axis=1)
  if mode=='DirectConditional':ws,we=self.te,self.ts
  else:ws=np.bincount(pairs[:,1],minlength=self.n)/len(pairs);we=np.bincount(pairs[:,0],minlength=self.n)/len(pairs)
  return .5*np.stack([ws@(ps-self.cs.numpy()),we@(pe-self.ce.numpy())])

def fit(p0,target,mode,seed,steps=40,lr=.1):
 import hashlib
 model=Policy(p0,target);rng=np.random.default_rng(seed);opt=torch.optim.Adam([model.eta],lr=lr);trace=[];maxerr=0.
 if np.array_equal(p0,target):return dict(q=np.array(p0,copy=True),eta=np.zeros((2,len(p0))),trace=[],actual_updates=0,scheduled_updates=steps,teacher_equal_prior_noop=True,max_manual_gradient_error=0.)
 for step in range(steps):
  before=model.eta.detach().numpy().copy();q=model.probability();pairs=None;uniforms=None;digest=None;selected=None
  if mode in ['Fixed','Live']:
   probs=(p0 if mode=='Fixed'else q)[model.mask];probs=probs/probs.sum();uniforms=rng.random(32);cdf=np.cumsum(probs);cdf[-1]=1.;indices=np.searchsorted(cdf,uniforms,side='right');pairs=np.column_stack([model.i[indices],model.j[indices]]);selected=probs[indices];target_s=model.cs[pairs[:,1]].numpy();target_e=model.ce[pairs[:,0]].numpy();digest=hashlib.sha256(target_s.tobytes()+target_e.tobytes()).hexdigest()
  opt.zero_grad(set_to_none=True);loss=model.loss(mode,pairs);loss.backward();gradient=model.eta.grad.detach().numpy().copy();manual=model.manual_gradient(mode,pairs);err=float(abs(gradient-manual).max());assert err<2e-12;maxerr=max(maxerr,err);opt.step()
  with torch.no_grad():model.eta-=model.eta.mean(1,keepdim=True)
  trace.append(dict(step=step,eta_before=before,eta_after=model.eta.detach().numpy().copy(),gradient=gradient,loss=float(loss.detach()),sampled_pairs=pairs,uniforms=uniforms,selected_sampling_probability=selected,teacher_conditional_digest=digest,manual_gradient_error=err,current_joint_digest=hashlib.sha256(q.tobytes()).hexdigest()))
 return dict(q=model.probability(),eta=model.eta.detach().numpy().copy(),trace=trace,actual_updates=steps,scheduled_updates=steps,teacher_equal_prior_noop=False,max_manual_gradient_error=maxerr)

def properties():
 p=np.array([[0.,.09,.16,.05],[0.,0.,.2,.1],[0.,0.,0.,.4],[0.,0.,0.,0.]])
 t=p*np.array([[1.,1.4,.5,2.],[1.,1.,1.5,.4],[1.,1.,1.,.8],[1.,1.,1.,1.]]);t/=t.sum();pairs=np.array([[0,1],[1,2],[0,3]])
 m=Policy(p,t);m.eta.data.copy_(torch.tensor([[.1,-.2,.05,.2],[-.1,.3,.1,0.]],dtype=torch.float64));errs=[];gradients={}
 for mode in ['DirectJoint','DirectConditional','Live']:
  m.eta.grad=None;loss=m.loss(mode,pairs);loss.backward();g=m.eta.grad.detach().numpy().copy();assert np.max(abs(g-m.manual_gradient(mode,pairs)))<1e-12;gradients[mode]=g
  for row in range(2):
   for col in range(4):
    orig=float(m.eta[row,col]);eps=1e-6
    with torch.no_grad():m.eta[row,col]=orig+eps
    high=float(m.loss(mode,pairs).detach())
    with torch.no_grad():m.eta[row,col]=orig-eps
    low=float(m.loss(mode,pairs).detach())
    with torch.no_grad():m.eta[row,col]=orig
    errs.append(abs((high-low)/(2*eps)-g[row,col]))
 assert max(errs)<2e-8 and np.max(abs(gradients['DirectJoint']-gradients['DirectConditional']))>1e-5
 n=fit(p,p,'Live',41);assert n['actual_updates']==0 and np.array_equal(n['q'],p)
 a=fit(p,t,'Fixed',41,steps=2);b=fit(p,t,'Live',41,steps=2)
 for k in ['sampled_pairs','gradient','eta_after']:assert np.array_equal(a['trace'][0][k],b['trace'][0][k])
 assert not np.array_equal(a['q'],p)
 return dict(status='pass',max_finite_difference_error=max(errs),teacher_equal_prior_noop=True,first_step_Fixed_Live_exact=True,conditional_and_joint_objectives_differ=True,temporary_state_changes=True)
