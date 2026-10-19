"""Original query+last-three-LN1792, with fixed E output map, no new network.

Samples remain in the original invertible native logit chart. Both candidates
and deterministic predictions are pushed through the same fixed E map; E may
legitimately exceed [0,1]. Never clamp E to fabricate an invertible chart.
"""
import copy,time
import torch
from methods.decota_final_simplified_v1.tensors import ParameterState,detached
from methods.decota_final_simplified_v1.objectives import SpatialLoss
from vg_tta.decota_spatial_opd_tunable_v1 import mean_coordinates,action_boxes,likelihood_loss,optimizer_step
from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
from vg_tta.stvg_dino_query_closedloop32_v1 import corrected
import hashlib

class Replay1792(ParameterState):
 def __init__(self,model,cache,source,E,check):
  from vg_tta.stvg_opd_frozen_capture_cpu007 import transfer
  self.decoder=copy.deepcopy(model.ground_decoder.decoder).eval().requires_grad_(False);self.inputs=transfer(cache['spatial_inputs'],'cuda');self.source=source.to('cuda');self.E=torch.as_tensor(E,device='cuda',dtype=torch.float64);self.check=check;self.n=len(source);self.calls=0;self.backward_calls=0;self.seconds=0.
  self.delta=torch.zeros(256,device='cuda',requires_grad=True);self.named=[('spatial.query_residual',self.delta)]
  for ln in ['norm1','norm3','norm4']:
   for name,p in getattr(self.decoder.layers[5],ln).named_parameters():p.requires_grad_(True);self.named.append((f'spatial.layers.5.{ln}.{name}',p))
  assert sum(p.numel()for _,p in self.named)==1792
  for n,p in self.named:assert torch.equal(p.detach().cpu(),cache['initial'][n])
  self.initial=self.state()
  with torch.no_grad():z=self.values()
  assert torch.equal(z['native'].cpu(),source)and torch.equal(z['boxes'],self.E)
 def values(self):
  self.check();t=time.perf_counter();raw=[]
  with torch.autocast('cuda',enabled=False):
   for inp in self.inputs:
    kw=dict(inp);q=kw['query_tgt'];kw['query_tgt']=q+self.delta[None,None,:];raw.append(self.decoder(**kw).flatten(1,2)[-1])
   native=torch.stack([raw[i%2][i//2]for i in range(self.n)]).float();out=corrected(native,self.source,self.E)
  torch.cuda.synchronize();self.calls+=1;self.seconds+=time.perf_counter()-t;self.check();assert torch.isfinite(out).all()and(out[:,2:]>0).all()
  return dict(native=native,boxes=out)
 def candidate_boxes(self,native_samples,positions):
  # Same E mapping, broadcasting the frame-specific fixed correction.
  b=action_boxes(native_samples).double();s=self.source[positions].double()[:,None];e=self.E[positions][:,None]
  return torch.cat([e[...,:2]+b[...,:2]-s[...,:2],e[...,2:]*torch.exp(torch.log(b[...,2:])-torch.log(s[...,2:]))],-1)

@torch.no_grad()
def rollout(rp,mu,targets,pos,key,k,cfg):
 seed=int(hashlib.sha256(f'spatial-opd-v1|{key}|{k}'.encode()).hexdigest()[:15],16);g=torch.Generator(device='cpu').manual_seed(seed);noise=torch.randn(len(mu),16,4,generator=g,dtype=torch.float32).to(mu);noise=torch.cat([noise,-noise],1);samples=(mu[:,None]+cfg['sigma']*noise).detach();boxes=rp.candidate_boxes(samples,pos);rewards=aligned_iou(boxes,targets[:,None,:]).detach();weights=torch.softmax(rewards/cfg['tau'],-1).detach()
 return dict(mean=mu.detach().clone(),samples=samples,rewards=rewards,weights=weights,informative=bool((rewards!=rewards[:,:1]).any()),seed=seed,candidate_boxes=boxes)

@torch.enable_grad()
def fit(rp,anchors,key,mode,cfg):
 assert mode in ['Direct','Fixed','Live'];rp.restore(rp.initial);initial=detached(rp.state(),'cpu');params=[p for _,p in rp.named];opt=torch.optim.Adam(params,lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.);pos=[int(a['position'])for a in anchors];target=torch.tensor([a['box']for a in anchors],device='cuda',dtype=torch.float64).reshape(-1,4);lossfn=SpatialLoss(anchors,rp.E);path=[];rounds=[];frozen=None;back=0;start=time.perf_counter()
 try:
  for k in range(cfg['steps']+1):
   rp.check();v=rp.values();loss_anchor=lossfn(v['boxes']);entry=dict(step=k,boxes=detached(v['boxes'],'cpu'),native=detached(v['native'],'cpu'),state=detached(rp.state(),'cpu'),anchor_loss=float(loss_anchor.detach()));path.append(entry)
   if k==0:assert torch.equal(v['boxes'],rp.E)
   if k==cfg['steps']or not pos:break
   rr=None;gmu=None;error=None;double_error=None;gradient_scale=None;tolerance=None
   if mode=='Direct':loss=loss_anchor;info=True
   else:
    mu=mean_coordinates(v['native'][pos]);rr=frozen if mode=='Fixed'and frozen is not None else rollout(rp,mu.detach(),target,pos,key,k,cfg)
    if mode=='Fixed'and frozen is None:frozen=rr
    loss=likelihood_loss(mu,rr['mean'],rr['samples'],rr['weights'],sigma=cfg['sigma']);gmu=torch.autograd.grad(loss,mu,retain_graph=True)[0];analytic=((mu.detach()-rr['mean'])/cfg['sigma']**2-((rr['weights']-1/32)[:,:,None]*(rr['samples']-mu.detach()[:,None,:])).sum(1)/cfg['sigma']**2)/len(pos);error=float((gmu-analytic).abs().max());double_error=None;gradient_scale=float(gmu.abs().max());tolerance=2e-5+8*torch.finfo(mu.dtype).eps*max(1.,gradient_scale)
    if error>=2e-5:
     md=mu.detach().double().requires_grad_(True);ld=likelihood_loss(md,rr['mean'].double(),rr['samples'].double(),rr['weights'].double(),sigma=cfg['sigma']);gd=torch.autograd.grad(ld,md)[0];double_error=float((gmu.double()-gd).abs().max());assert double_error<tolerance,('float32-vs-double',mode,k,double_error,tolerance,gradient_scale)
    assert error<tolerance,('analytic',mode,k,error,tolerance,gradient_scale);info=rr['informative']
   update=optimizer_step(opt,loss,params,info)
   if update is not None:back+=1;rp.backward_calls+=1;entry['update']=update
   rounds.append(dict(step=k,updated=update is not None,loss=float(loss.detach()),rollout=detached(rr,'cpu'),autograd_mean_gradient=detached(gmu,'cpu'),analytic_gradient_error=error,double_reference_gradient_error=double_error,gradient_scale=gradient_scale,gradient_tolerance=tolerance,Adam=detached(opt.state_dict(),'cpu')))
  finalstate=detached(rp.state(),'cpu');assert all(torch.equal(finalstate[n],path[-1]['state'][n])for n in finalstate)
  if not pos:assert torch.equal(path[-1]['boxes'],rp.E.cpu())
  return dict(mode=mode,initial=initial,state=finalstate,path=path,rounds=rounds,final=path[-1]['boxes'],native_final=path[-1]['native'],gradient_calls=back,positions=pos,config=cfg,selected_step=cfg['steps']if pos else 0,actual_rounds=len(rounds),seconds=time.perf_counter()-start,active_parameters=1792,empty=not pos,zero_exact_E=True)
 finally:rp.restore(rp.initial)
