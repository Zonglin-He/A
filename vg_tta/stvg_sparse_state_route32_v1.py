"""Per-query final native CA-content bias: sparse direct versus trajectory OPD."""
import hashlib,time
import numpy as np
import torch
from methods.decota_final_simplified_v1.tensors import detached
from vg_tta.stvg_dino_query_closedloop32_v1 import corrected
ARMS=['E_MBR','Fine_center','Direct_anchor','State_fixed','State_live']
FIT_ARMS=['Direct_anchor','State_fixed','State_live']

def noise(alias):
 seed=int.from_bytes(hashlib.sha256(('sparse-state-route32-v1|'+alias).encode()).digest()[:8],'little');g=np.random.default_rng(seed);return np.stack([np.concatenate((a,-a),0)for a in [g.normal(size=(8,256)),g.normal(size=(8,256))]])

def box_loss(boxes,positions,targets):
 if not positions:return boxes.sum()*0
 b=boxes[positions];t=torch.as_tensor(targets,device=b.device,dtype=b.dtype);bl=b[:,:2]-b[:,2:]/2;bh=b[:,:2]+b[:,2:]/2;tl=t[:,:2]-t[:,2:]/2;th=t[:,:2]+t[:,2:]/2;inter=(torch.minimum(bh,th)-torch.maximum(bl,tl)).clamp_min(0).prod(-1);union=b[:,2:].prod(-1)+t[:,2:].prod(-1)-inter;cover=(torch.maximum(bh,th)-torch.minimum(bl,tl)).clamp_min(0).prod(-1);giou=inter/union.clamp_min(1e-30)-(cover-union)/cover.clamp_min(1e-30)
 return 5*(b-t).abs().sum(-1).mean()+2*(1-giou).mean()

def weights(actions,current,proposal,losses,valid):
 valid=np.asarray(valid,bool);logratio=-.5*((actions-current)**2).sum(1)+.5*((actions-proposal)**2).sum(1)
 if not valid.any():return dict(base=np.zeros(len(actions)),teacher=np.zeros(len(actions)),gradient=np.zeros(256),informative=False,log_ratio=logratio)
 lp=logratio.copy();lp[~valid]=-np.inf
 def sm(z):
  a=np.exp(z-np.max(z));return a/a.sum()
 base=sm(lp);informative=bool(losses[valid].max()-losses[valid].min()>1e-12);teacher=sm(lp-losses)if informative else base.copy();grad=((base-teacher)[:,None]*actions).sum(0)
 if not informative:grad[:]=0
 return dict(base=base,teacher=teacher,gradient=grad,informative=informative,log_ratio=logratio)

class Decoder:
 def __init__(self,decoder,inputs,source,E,check):
  self.decoder=decoder;self.inputs=inputs;self.source=source.cpu();self.E=torch.as_tensor(E,dtype=torch.float64);self.check=check;self.current=None;self.scale=None;self.zero_q=[];self.calls=0;self.backward_calls=0;self.model_seconds=0.;self.cache={};self.n=len(source);self.last_projection=decoder.layers[-1].ca_qcontent_proj;self.prefix_native=[];self.prefix_checks=0
  assert len(decoder.layers)==6
  def hook(mod,args,out):
   if self.scale is None:self.zero_q.append(out.detach().cpu())
   return out if self.current is None else out+self.scale*self.current[None,None]
  self.hook=self.last_projection.register_forward_hook(hook)
  z=self.raw(None,False);assert torch.equal(z['native'],self.source)and np.array_equal(z['boxes'],self.E.numpy());self.scale=float(torch.cat([v.double().reshape(-1)for v in self.zero_q]).square().mean().sqrt());assert self.scale>0 and np.isfinite(self.scale);zz=self.raw(np.zeros(256),False);assert torch.equal(zz['native'],self.source)and np.array_equal(zz['boxes'],self.E.numpy());self.cache[np.zeros(256,dtype=np.float64).tobytes()]=zz
 def close(self):self.hook.remove()
 def raw(self,m,gradient=False):
  self.check();begin=time.perf_counter();native=[]
  if m is None:self.current=None
  elif isinstance(m,torch.Tensor):self.current=m
  else:self.current=torch.as_tensor(m,device='cuda',dtype=torch.float32)
  with torch.set_grad_enabled(gradient):
   for off,kw in enumerate(self.inputs):
    out=self.decoder(**kw);all_layers=out.flatten(1,2);pre=all_layers[:-1].detach().cpu()
    if len(self.prefix_native)<=off:self.prefix_native.append(pre.clone())
    else:assert torch.equal(self.prefix_native[off],pre);self.prefix_checks+=1
    native.append(all_layers[-1])
   raw=torch.stack([native[i%2][i//2]for i in range(self.n)]).float();b=corrected(raw,self.source.to(raw),self.E.to(raw.device));valid=bool(torch.isfinite(b).all()and(b[:,2:]>0).all());cpu=raw.detach().cpu()
  torch.cuda.synchronize();self.calls+=1;self.model_seconds+=time.perf_counter()-begin;self.check()
  return dict(native=cpu,boxes=b if gradient else b.detach().cpu().numpy(),valid=valid,actual_decoder=True)
 @torch.no_grad()
 def __call__(self,m):
  m=np.asarray(m,dtype=np.float64);key=m.tobytes()
  if key not in self.cache:self.cache[key]=self.raw(m,False)
  return self.cache[key]
 def gradient_check(self,alias):
  # Smooth output probe, no GT/anchor loss kinks. Only final-layer bias varies.
  rng=np.random.default_rng(int.from_bytes(hashlib.sha256(alias.encode()).digest()[:8],'little'));direction=rng.choice([-1.,1.],256)/16;probe=torch.as_tensor(rng.choice([-1.,1.],(self.n,4))/np.sqrt(4*self.n),device='cuda',dtype=torch.float64);m=torch.zeros(256,device='cuda',requires_grad=True);z=self.raw(m,True);v=(z['boxes']*probe).sum();g=torch.autograd.grad(v,m)[0].detach().cpu().double().numpy();eps=.01;plus=self(eps*direction)['boxes'];minus=self(-eps*direction)['boxes'];self.backward_calls+=1;fd=float(((plus-minus)*probe.cpu().numpy()).sum()/(2*eps));ad=float(g@direction);assert abs(fd-ad)<2e-4+.05*abs(fd)
  return dict(FD=fd,autograd=ad,absolute_error=abs(fd-ad),tolerance=2e-4+.05*abs(fd),smooth_output_probe=True,no_GT=True)

def fit(dec,eps,positions,targets,arm,lr,check):
 assert arm in FIT_ARMS;m=torch.zeros(256,device='cuda',dtype=torch.float32,requires_grad=True);opt=torch.optim.Adam([m],lr=lr,betas=(.9,.999),eps=1e-8);path=[dict(state=np.zeros(256),boxes=dec.E.numpy().copy())];rounds=[];active=bool(positions)
 for step in range(2):
  check();before=m.detach().cpu().double().numpy();current=dec(before);lbefore=float(box_loss(torch.as_tensor(current['boxes']),positions,targets));records=[];w=None;proposal=None;actions=None;losses=None;valid=None;updated=False
  if active:
   if arm=='Direct_anchor':
    out=dec.raw(m,True);loss=box_loss(out['boxes'],positions,targets);grad=torch.autograd.grad(loss,m)[0];dec.backward_calls+=1;updated=bool(torch.count_nonzero(grad).item());g=grad.detach().cpu().double().numpy()
   else:
    proposal=np.zeros(256)if arm=='State_fixed'else before.copy();actions=proposal[None]+eps[step];records=[dec(a)for a in actions];valid=np.array([r['valid']for r in records]);losses=np.array([float(box_loss(torch.as_tensor(r['boxes']),positions,targets))if r['valid']else 0. for r in records]);w=weights(actions,before,proposal,losses,valid);g=w['gradient'];grad=torch.as_tensor(g,device='cuda',dtype=torch.float32);updated=w['informative']and bool(np.any(g!=0))
   assert torch.isfinite(grad).all();opt.zero_grad(set_to_none=True)
   if updated:m.grad=grad;opt.step()
  else:g=np.zeros(256)
  after=m.detach().cpu().double().numpy();post=dec(after);assert post['valid'],'nonfinite/invalid updated mean, engineering fail without label/quality rollback';lafter=float(box_loss(torch.as_tensor(post['boxes']),positions,targets));rounds.append(dict(step=step,before=before,after=after,gradient=g,updated=updated,loss_before=lbefore,loss_after=lafter,proposal=proposal,actions=actions,noise=eps[step],losses=losses,valid=valid,weights=w,Adam=detached(opt.state_dict(),'cpu'),actual_candidates=[dict(native=r['native'],boxes=r['boxes'],valid=r['valid'])for r in records],post_native=post['native'],post_boxes=post['boxes']));path.append(dict(state=after,boxes=post['boxes'].copy()))
 return dict(arm=arm,path=path,rounds=rounds,final=path[-1]['boxes'],selected_step=2,temporary_native_bias_dimension=256,lr=lr,GT_used=False,network_weights_unchanged=True)

def properties():
 e=noise('properties');assert np.array_equal(e[:,:8],-e[:,8:]);a=e[0];w=weights(a,np.zeros(256),np.zeros(256),np.ones(16),np.ones(16,bool));assert np.array_equal(w['gradient'],np.zeros(256));b=torch.tensor([[.5,.5,.2,.2]],dtype=torch.float64,requires_grad=True);assert abs(float(box_loss(b,[0],b.detach().numpy())))<1e-12;assert float(box_loss(b,[],[]))==0
 return dict(status='pass',antithetic=True,constant_feedback_no_drift=True,exact_box_zero_loss=True,no_reference_loss_zero=True,no_pretrained_training=True)
