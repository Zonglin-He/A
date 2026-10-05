"""Persistent STVG task ports of TENT/EATA/SAR. No experts or label access.

This is a disclosed task port, not an unchanged classification reproduction.
Native entropy, masks, two offsets and physical grids are validated upstream.
EATA compares relative-clip endpoint profiles because STVG has variable grids,
and uses source-only pseudo-native Fisher; it is never run with Fisher=None.
"""
import copy,math
import torch
import torch.nn.functional as F
from vg_tta.native_probability_interface_v1 import offsets_entropy,entropy_ceiling

CONFIGS={
 'TENT':dict(lr=.001,steps=1,optimizer='Adam'),
 'EATA':dict(lr=.00025,steps=1,optimizer='SGD',momentum=.9,margin_fraction=.4,d_margin=.05,fisher_alpha=2000.,profile_bins=32),
 'SAR':dict(lr=.001,steps=3,optimizer='SGD',momentum=.9,rho=.05,margin_fraction=.4,reset_entropy=.2),
}

def profile(outputs,bins=32):
 """Detached normalized-time endpoint mass, plus binary mean actionness.

Each endpoint contributes a probability histogram, not a new input frame grid.
Physical IDs are normalized over the observed clip. Averaging original offsets
is only for the reliability descriptor; entropy still treats offsets separately.
"""
 out=[]
 for o in outputs:
  p,a=o.probabilities();ids=torch.tensor(o.frame_ids,device=p.device,dtype=p.dtype)[o.valid]
  scaled=((ids-ids.min())/(ids.max()-ids.min()).clamp_min(1))*bins
  ind=scaled.long().clamp(0,bins-1);hist=p.new_zeros(bins,2).scatter_add_(0,ind[:,None].expand(-1,2),p)
  out.append(torch.cat([hist[:,0],hist[:,1],torch.stack([a.mean(),1-a.mean()])])/3)
 z=torch.stack(out).mean(0).detach();assert torch.isfinite(z).all() and torch.isclose(z.sum(),z.new_tensor(1.))
 return z

def pseudo_native_loss(outputs):
 """Source-only squared-gradient Fisher surrogate, without source/target GT."""
 loss=[]
 for o in outputs:
  o.probabilities();z=o.sted[o.valid].float();a=o.actionness_logits.reshape(-1)[o.valid].float()
  target=z.detach().argmax(0)
  loss.append(F.cross_entropy(z.T,target)+F.binary_cross_entropy_with_logits(a,(a.detach()>=0).float()))
 return torch.stack(loss).mean()

class OnlineBaseline:
 def __init__(self,names,parameters,method,fisher=None,config=None):
  assert method in CONFIGS
  self.names=list(names);self.params=list(parameters);self.method=method;self.cfg=dict(CONFIGS[method])
  if config is not None:
   assert set(config)==set(self.cfg);self.cfg.update(config)
  assert self.params and len(set(self.names))==len(self.params) and len({id(p) for p in self.params})==len(self.params)
  self.initial=[p.detach().clone() for p in self.params];self.flags=[p.requires_grad for p in self.params]
  self.fisher=None
  if method=='EATA':
   assert fisher is not None,'Fisher=None is ETA, not EATA'
   assert fisher['names']==self.names and fisher['source_inputs_only'] and fisher['source_queries']==2000 and not fisher['target_labels_used']
   assert len(fisher['values'])==len(self.params)
   self.fisher=[v.to(p) for v,p in zip(fisher['values'],self.params)]
   assert all(v.shape==p.shape and torch.isfinite(v).all() and (v>=0).all() for v,p in zip(self.fisher,self.params))
   assert all(torch.equal(v.to(p),p.detach()) for v,p in zip(fisher['source_parameters'],self.params))
  self.optimizer=(torch.optim.Adam(self.params,lr=self.cfg['lr']) if method=='TENT' else torch.optim.SGD(self.params,lr=self.cfg['lr'],momentum=self.cfg['momentum']))
  self.opt_initial=copy.deepcopy(self.optimizer.state_dict());self.ema=None;self.probs=None;self.arrivals=0;self.audit_updates=False
 def copy(self,values):
  with torch.no_grad():
   for p,v in zip(self.params,values):p.copy_(v)
 def reset(self):
  self.copy(self.initial);self.optimizer.load_state_dict(copy.deepcopy(self.opt_initial));self.ema=None;self.probs=None;self.arrivals=0
 def state_dict(self):
  return dict(method=self.method,config=self.cfg,names=self.names,parameters=[p.detach().cpu().clone() for p in self.params],optimizer=copy.deepcopy(self.optimizer.state_dict()),ema=self.ema,probs=None if self.probs is None else self.probs.cpu(),arrivals=self.arrivals)
 def load_state_dict(self,z):
  assert z['method']==self.method and z['config']==self.cfg and z['names']==self.names
  self.copy([v.to(p) for v,p in zip(z['parameters'],self.params)]);self.optimizer.load_state_dict(z['optimizer'])
  self.ema=z['ema'];self.probs=None if z['probs'] is None else z['probs'].to(self.params[0]);self.arrivals=z['arrivals']
 @torch.enable_grad()
 def arrive(self,closure,inference):
  trace=[];evidence=[];counts=dict(forward_closures=0,backward_steps=0,optimizer_steps=0,recoveries=0)
  def output():
   with torch.no_grad():return inference()
  def grad(loss):
   assert loss.requires_grad and torch.isfinite(loss)
   loss.backward();counts['backward_steps']+=1
   gs=[p.grad for p in self.params if p.grad is not None]
   assert gs and all(torch.isfinite(g).all() for g in gs)
   return torch.sqrt(sum(g.float().square().sum() for g in gs))
  pre=output()
  try:
   for p in self.params:p.requires_grad_(True)
   for _ in range(self.cfg['steps']):
    self.optimizer.zero_grad(set_to_none=True);native=closure();counts['forward_closures']+=1
    entropy=offsets_entropy(native);assert torch.isfinite(entropy) and entropy.requires_grad
    rec=dict(entropy=float(entropy.detach()),updated=False)
    if self.method in ('EATA','SAR'):
     margin=self.cfg['margin_fraction']*sum(entropy_ceiling(o) for o in native)/len(native);rec['margin']=margin
     if entropy.detach()>=margin:rec['skip']='entropy';trace.append(rec);continue
    if self.method=='EATA':
     vector=profile(native,self.cfg['profile_bins']);sim=None if self.probs is None else float(F.cosine_similarity(vector[None],self.probs[None]))
     rec['profile_cosine']=sim
     if sim is not None and abs(sim)>=self.cfg['d_margin']:rec['skip']='redundancy';trace.append(rec);continue
     self.probs=vector if self.probs is None else .9*self.probs+.1*vector
     reg=sum((f*(p-p0).square()).sum() for f,p,p0 in zip(self.fisher,self.params,self.initial))
     loss=entropy*torch.exp(entropy.new_tensor(margin)-entropy.detach())+self.cfg['fisher_alpha']*reg
     rec['fisher_regularization']=float(reg.detach());rec['loss']=float(loss.detach());rec['gradient_norm']=float(grad(loss))
    elif self.method=='SAR':
     magnitude=grad(entropy);saved=[p.detach().clone() for p in self.params]
     try:
      with torch.no_grad():
       for p in self.params:
        if p.grad is not None:p.add_(p.grad*self.cfg['rho']/(magnitude+1e-12))
      self.optimizer.zero_grad(set_to_none=True);second=offsets_entropy(closure());counts['forward_closures']+=1
      assert torch.isfinite(second);rec['second_entropy']=float(second.detach())
      if second.detach()>=margin:rec['skip']='second_entropy';trace.append(rec);continue
      rec['gradient_norm']=float(grad(second));self.ema=float(second.detach()) if self.ema is None else .9*self.ema+.1*float(second.detach())
     finally:self.copy(saved)
    else:rec['gradient_norm']=float(grad(entropy))
    if self.audit_updates:
     from methods.decota_final_simplified_v1.tensors import detached
     ev=dict(parameters_before=[p.detach().cpu().clone() for p in self.params],gradients=[None if p.grad is None else p.grad.detach().cpu().clone() for p in self.params],optimizer_before=detached(copy.deepcopy(self.optimizer.state_dict()),'cpu'))
    self.optimizer.step();counts['optimizer_steps']+=1;rec['updated']=True
    if self.audit_updates:ev['parameters_after']=[p.detach().cpu().clone() for p in self.params];evidence.append(ev)
    assert all(torch.isfinite(p).all() for p in self.params)
    if self.method=='SAR' and self.ema<self.cfg['reset_entropy']:
     self.copy(self.initial);self.optimizer.load_state_dict(copy.deepcopy(self.opt_initial));self.ema=None;counts['recoveries']+=1;rec['recovered']=True
    trace.append(rec)
   post=output();self.arrivals+=1
   return pre,post,dict(method=self.method,arrival=self.arrivals,trace=trace,counts=counts,update_evidence=evidence,GT_read=False,expert_calls=0)
  finally:
   for p,f in zip(self.params,self.flags):p.requires_grad_(f);p.grad=None
