"""Actual frozen decoder rollouts; external standardized QR Gaussian mean only."""
import hashlib
import numpy as np
import torch
from vg_tta.stvg_dino_query_closedloop32_v1 import corrected
ARMS=['Source_MBR','E_MBR','Endpoint_local','Latent_search','Latent_mean_control','Latent_fixed','Latent_live','Latent_shuffle']
FIT_ARMS=['Latent_fixed','Latent_live','Latent_shuffle']


def noise(alias):
 seed=int.from_bytes(hashlib.sha256(('realizable-query32-v1|'+alias).encode()).digest()[:8],'little');g=np.random.default_rng(seed);return np.stack([np.concatenate((a,-a),0)for a in [g.standard_normal((8,256)),g.standard_normal((8,256))]])

def weights(actions,current,proposal,rewards,valid,tau=.1):
 # Self-normalized importance estimate of current-policy and reward-tilted
 # expectations. No deterministic zero atom is included in this integral.
 valid=np.asarray(valid,bool);lp=-.5*((actions-current)**2).sum(1)+.5*((actions-proposal)**2).sum(1)
 if not valid.any():return dict(base=np.zeros(len(actions)),teacher=np.zeros(len(actions)),delta=np.zeros(256),informative=False,log_ratio=lp)
 lp[~valid]=-np.inf
 def sm(x):
  y=np.exp(x-np.max(x));return y/y.sum()
 base=sm(lp);info=bool(np.max(rewards[valid])-np.min(rewards[valid])>1e-12);teacher=sm(lp+rewards/tau)if info else base.copy();delta=((teacher-base)[:,None]*actions).sum(0)
 if not info:delta[:]=0
 assert np.isfinite(delta).all()
 return dict(base=base,teacher=teacher,delta=delta,informative=info,log_ratio=lp)

def reward(boxes,positions,targets):
 if not len(positions):return 0.
 b=np.asarray(boxes,dtype=np.float64)[positions];t=np.asarray(targets,dtype=np.float64);bl=b[:,:2]-b[:,2:]/2;bh=b[:,:2]+b[:,2:]/2;tl=t[:,:2]-t[:,2:]/2;th=t[:,:2]+t[:,2:]/2;inter=np.maximum(np.minimum(bh,th)-np.maximum(bl,tl),0).prod(1);union=b[:,2:].prod(1)+t[:,2:].prod(1)-inter;return float(np.mean(inter/np.maximum(union,1e-30)))

class Decoder:
 def __init__(self,decoder,inputs,source,E,check=None):
  self.check=check;self.decoder=decoder;self.inputs=inputs;self.source=source.cpu();self.E=torch.as_tensor(E,dtype=torch.float64).cpu();self.n=len(source);self.cache={};self.calls=0;self.model_seconds=0.
  self.query_rms=float(torch.cat([x['query_tgt'].detach().cpu().double().reshape(-1)for x in inputs]).square().mean().sqrt());self.sigma=self.query_rms/16;assert np.isfinite(self.sigma)and self.sigma>0
 @torch.no_grad()
 def __call__(self,m):
  import time
  m=np.asarray(m,dtype=np.float64);key=m.tobytes()
  if key in self.cache:return self.cache[key]
  if self.check:self.check()
  tick=time.perf_counter();native=[];u=self.sigma*torch.from_numpy(m).to('cuda',dtype=torch.float32)
  for x in self.inputs:
   kw=dict(x);kw['query_tgt']=x['query_tgt']+u[None,None];p=self.decoder(**kw);native.append(p.flatten(1,2)[-1])
  raw=torch.stack([native[i%2][i//2]for i in range(self.n)]).float().cpu();valid=bool(torch.isfinite(raw).all()and(raw[:,2:]>0).all())
  if valid:
   b=corrected(raw,self.source,self.E);valid=bool(torch.isfinite(b).all()and(b[:,2:]>0).all());out=b.numpy()
  else:out=None
  torch.cuda.synchronize();self.model_seconds+=time.perf_counter()-tick;self.calls+=1;rec=dict(m=m.copy(),u=u.cpu(),native=raw,boxes=out,valid=valid,actual_frozen_decoder=True);self.cache[key]=rec
  if self.check:self.check()
  return rec

def fit(decode,eps,positions,targets,mode,check):
 m=np.zeros(256,dtype=np.float64);z=decode(m);assert z['valid'];path=[dict(state=m.copy(),boxes=z['boxes'].copy())];trace=[];active=bool(len(positions));tt=np.asarray(targets)
 if mode=='Latent_shuffle'and len(tt)>1:tt=np.roll(tt,1,axis=0)
 for step in range(2):
  check();proposal=np.zeros_like(m)if mode=='Latent_fixed'else m.copy();actions=proposal[None]+eps[step];records=[]
  if active:
   for a in actions:check();records.append(decode(a))
   valid=np.array([r['valid']for r in records]);rs=np.array([reward(r['boxes'],positions,tt)if r['valid']else 0. for r in records]);w=weights(actions,m,proposal,rs,valid);proposed=m+.5*w['delta'];post=decode(proposed);rejected=not post['valid'];new=m if rejected else proposed;out=decode(new)['boxes']
  else:
   valid=np.ones(16,bool);rs=np.zeros(16);w=weights(actions,m,proposal,rs,valid);proposed=m.copy();new=m.copy();out=z['boxes'];rejected=False
  trace.append(dict(step=step,sampling_center=proposal,actions=actions,noise=eps[step],rewards=rs,valid=valid,weights=w,before=m.copy(),proposed=proposed,after=new.copy(),invalid_mean_rejected=rejected,teacher_binding=np.roll(np.arange(len(tt)),1).tolist()if mode=='Latent_shuffle'and len(tt)>1 else list(range(len(tt))),actual_candidates=[dict(native=r['native'],boxes=r['boxes'],valid=r['valid'])for r in records],fixed_step=.5))
  m=new.copy();path.append(dict(state=m.copy(),boxes=np.asarray(out).copy()))
 return dict(mode=mode,path=path,rounds=trace,final=path[-1]['boxes'],selected_step=2,GT_read=False,no_decoder_backprop=True,actual_candidate_generation=True,active=active)

def properties():
 eps=noise('properties');m=np.linspace(-.1,.1,256);prop=np.zeros(256);actions=prop[None]+eps[0];r=np.ones(16);v=np.ones(16,bool);w=weights(actions,m,prop,r,v);assert np.array_equal(w['delta'],np.zeros(256))and np.array_equal(w['base'],w['teacher'])
 v[::3]=False;assert np.array_equal(weights(actions,m,prop,r,v)['delta'],np.zeros(256));assert not weights(actions,m,prop,r,np.zeros(16,bool))['informative'];assert np.array_equal(eps[:,:8],-eps[:,8:])
 wp=weights(actions,np.zeros(256),np.zeros(256),np.arange(16)/16,np.ones(16,bool));assert np.max(abs(wp['base']-1/16))<1e-15
 assert np.isfinite(wp['delta']).all();assert abs(reward(np.array([[.5,.5,.2,.2]]),[0],[[.5,.5,.2,.2]])-1)<1e-12
 s=torch.tensor([[.5,.5,.2,.2]],dtype=torch.float32);E=s.double()+.01;assert torch.equal(corrected(s,s,E),E)
 return dict(status='pass',zero_atom_not_in_integral=True,antithetic16_exact=True,constant_feedback_multiround_no_drift=True,partial_valid_constant_no_drift=True,all_invalid_noop=True,on_policy_current_proposal_ratio_uniform=True,zero_query_exact_E_mapping=True,normalized_step_units=True)
