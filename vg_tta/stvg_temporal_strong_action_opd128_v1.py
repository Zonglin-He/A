"""Cached observation-scale teacher construction and diagnostics; no networks."""
import numpy as np
from vg_tta.stvg_temporal_duration_condition128_v1 import posterior,marginal

def high_curve(observation):
 x=np.empty(65,dtype=np.float64)
 assert len(observation['actions'])==2
 for off,v in enumerate(observation['actions']):
  arr=v.detach().cpu().numpy();assert arr.dtype==np.float32;x[off::2]=arr.reshape(-1)
 assert np.isfinite(x).all();return x

def wrong_curve(ids,x,donor_ids,donor):
 u=np.asarray(ids,dtype=np.float64);v=np.asarray(donor_ids,dtype=np.float64);u=(u-u[0])/(u[-1]-u[0]);v=(v-v[0])/(v[-1]-v[0]);w=np.interp(u,v,donor);mu=float(x.mean());sd=float(x.std());ds=float(w.std());deg=sd<=1e-12 or ds<=1e-12
 out=np.full(65,mu)if deg else mu+(w-w.mean())*(sd/ds)
 assert abs(out.mean()-mu)<1e-12
 if not deg:assert abs(out.std()-sd)<1e-12
 return out,dict(degenerate=deg,recipient_std=sd,interpolated_donor_std=ds)

def targets(base,curves):return {k:posterior(base,v,True)for k,v in curves.items()}

def joint_kl(t,q):
 mask=t>0;assert np.all(q[mask]>0);return float(np.sum(t[mask]*np.log(t[mask]/q[mask])))

def summary(base,t,q):
 return dict(initial_joint_KL=joint_kl(t,base),final_joint_KL=joint_kl(t,q),node_count_TV_to_base=float(abs(marginal(q)-marginal(base)).sum()/2),start_marginal_TV_to_target=float(abs(q.sum(1)-t.sum(1)).sum()/2),end_marginal_TV_to_target=float(abs(q.sum(0)-t.sum(0)).sum()/2))

# Identical existing Policy/Adam arithmetic, with bounded-run progress hooks.
from vg_tta.stvg_temporal_conditional_opd128_v1 import Policy
import torch
def fit(p0,target,mode,seed,steps=40,lr=.1,observer=None,check=None):
 import hashlib
 model=Policy(p0,target);rng=np.random.default_rng(seed);opt=torch.optim.Adam([model.eta],lr=lr);trace=[];maxerr=0.
 if np.array_equal(p0,target):return dict(q=np.array(p0,copy=True),eta=np.zeros((2,len(p0))),trace=[],actual_updates=0,scheduled_updates=steps,teacher_equal_prior_noop=True,max_manual_gradient_error=0.)
 for step in range(steps):
  if check is not None:check()
  before=model.eta.detach().numpy().copy();q=model.probability();pairs=None;uniforms=None;digest=None;selected=None
  if mode in ['Fixed','Live']:
   probs=(p0 if mode=='Fixed'else q)[model.mask];probs=probs/probs.sum();uniforms=rng.random(32);cdf=np.cumsum(probs);cdf[-1]=1.;indices=np.searchsorted(cdf,uniforms,side='right');pairs=np.column_stack([model.i[indices],model.j[indices]]);selected=probs[indices];target_s=model.cs[pairs[:,1]].numpy();target_e=model.ce[pairs[:,0]].numpy();digest=hashlib.sha256(target_s.tobytes()+target_e.tobytes()).hexdigest()
  opt.zero_grad(set_to_none=True);loss=model.loss(mode,pairs);loss.backward();gradient=model.eta.grad.detach().numpy().copy();manual=model.manual_gradient(mode,pairs);err=float(abs(gradient-manual).max());assert err<2e-12;maxerr=max(maxerr,err);opt.step()
  if observer is not None:observer('adam',trace)
  with torch.no_grad():model.eta-=model.eta.mean(1,keepdim=True)
  trace.append(dict(step=step,eta_before=before,eta_after=model.eta.detach().numpy().copy(),gradient=gradient,loss=float(loss.detach()),sampled_pairs=pairs,uniforms=uniforms,selected_sampling_probability=selected,teacher_conditional_digest=digest,manual_gradient_error=err,current_joint_digest=hashlib.sha256(q.tobytes()).hexdigest()))
  if observer is not None:observer('trace',trace)
 return dict(q=model.probability(),eta=model.eta.detach().numpy().copy(),trace=trace,actual_updates=steps,scheduled_updates=steps,teacher_equal_prior_noop=False,max_manual_gradient_error=maxerr)

def instrumentation_properties():
 from vg_tta.stvg_temporal_conditional_opd128_v1 import fit as original
 p=np.array([[0.,.1,.2],[0.,0.,.7],[0.,0.,0.]])
 t=np.array([[0.,.2,.4],[0.,0.,.4],[0.,0.,0.]])
 cases=0
 for mode in ['DirectJoint','DirectConditional','Fixed','Live']:
  for seed in [41,42,43]:
   calls=[];a=original(p,t,mode,seed,steps=2);b=fit(p,t,mode,seed,steps=2,observer=lambda e,tr:calls.append(e))
   assert np.array_equal(a['q'],b['q'])and np.array_equal(a['eta'],b['eta'])and calls==['adam','trace']*2
   for x,y in zip(a['trace'],b['trace']):
    assert x.keys()==y.keys()
    for k,v in x.items():
     if isinstance(v,np.ndarray):assert np.array_equal(v,y[k])
     else:assert v==y[k]
   cases+=1
 return dict(status='pass',original_fitter_bitexact_cases=cases,all_trace_fields_exact=True,progress_hooks_only=True)
