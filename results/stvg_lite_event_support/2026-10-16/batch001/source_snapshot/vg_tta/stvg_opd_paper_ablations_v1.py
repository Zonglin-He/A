"""Explicit, frame-aligned Gaussian action policy with detached expert feedback.

The native merged cxcywh output remains the exact deterministic central readout.
The stochastic extension uses its inverse-sigmoid coordinates and a fixed scale;
it is exploration, not trained or calibrated uncertainty. No expert coordinates
enter the likelihood objective. Frozen rollout is a named distillation control,
not an importance-corrected off-policy algorithm.
"""
import hashlib,math
import torch
from methods.decota_final_simplified_v1.tensors import detached
from vg_tta.c1_enabling_tricks_v1 import TrickReplay
from vg_tta.decota_optimizer_posterior_r1_v1 import Energy,flat
from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou

CONFIG=dict(sigma=.25,tau=.25,samples=32,steps=10,lr=.03,writeback=1/16)
ARMS=['on_policy','frozen_rollout','shuffled_feedback']

def mean_coordinates(boxes):
 assert torch.isfinite(boxes).all() and ((boxes>0)&(boxes<1)).all(), 'Policy requires an invertible native box chart; do not silently clamp'
 return torch.logit(boxes)

def action_boxes(z):return torch.sigmoid(z)

def likelihood_loss(mean,rollout_mean,samples,weights,sigma=.25):
 """Control-variate estimate of the forward-KL gradient; detached actions.

The analytic baseline is KL(pi_rollout || pi_mean). It is not a constraint on
the post-Adam movement. Each on-policy round has one optimizer step.
"""
 assert not rollout_mean.requires_grad and not samples.requires_grad and not weights.requires_grad
 assert samples.shape==(*mean.shape[:-1],32,4) and weights.shape==samples.shape[:-1]
 kl=(mean-rollout_mean).square().sum(-1)/(2*sigma**2)
 logp=-(samples-mean.unsqueeze(-2)).square().sum(-1)/(2*sigma**2)-2*math.log(2*math.pi*sigma**2)
 return (kl-((weights-1/32)*logp).sum(-1)).mean()

@torch.no_grad()
def rollout(mean,evidence,key,round_index,shuffle=False,*,config=None):
 cfg=dict(CONFIG if config is None else config)
 assert mean.ndim==2 and mean.shape==evidence.shape and mean.shape[-1]==4
 seed=int(hashlib.sha256(f'spatial-opd-v1|{key}|{round_index}'.encode()).hexdigest()[:15],16)
 gen=torch.Generator(device='cpu').manual_seed(seed)
 noise=torch.randn(len(mean),16,4,generator=gen,dtype=torch.float32).to(mean)
 noise=torch.cat((noise,-noise),1)
 z=(mean[:,None,:]+cfg['sigma']*noise).detach()
 r=aligned_iou(action_boxes(z),evidence[:,None,:]).detach()
 if shuffle:
  permutations=torch.stack([torch.randperm(32,generator=gen) for _ in mean]).to(mean.device)
  used=r.gather(1,permutations)
 else:permutations=None;used=r
 w=torch.softmax(used/cfg['tau'],-1).detach()
 # Exact all-equal comparisons; no learned threshold or acceptance gate.
 informative=(used!=used[:,:1]).any()
 return dict(mean=mean.detach().clone(),samples=z,rewards=r,used_rewards=used,weights=w,
  permutations=permutations,informative=bool(informative),seed=seed)

@torch.enable_grad()
def optimizer_step(opt,loss,params,informative):
 opt.zero_grad(set_to_none=True)
 if not informative:return None
 gs=torch.autograd.grad(loss,params)
 assert all(torch.isfinite(g).all() for g in gs)
 before=flat(params).detach().clone()
 for p,g in zip(params,gs):p.grad=g.detach()
 opt.step();after=flat(params).detach().clone();assert torch.isfinite(after).all()
 return dict(gradient=flat(gs).detach().cpu(),raw=(after-before).cpu())

@torch.enable_grad()
def component_fit(base,initial,expert,ids,key,arm,*,config=None,scope="joint",direct=False):
 cfg=dict(CONFIG if config is None else config)
 assert set(cfg)==set(CONFIG) and cfg['samples']==32
 assert cfg['sigma']>0 and cfg['tau']>0 and cfg['lr']>0
 assert isinstance(cfg['steps'],int) and cfg['steps']>=1 and 0<=cfg['writeback']<=1
 assert arm in ARMS and scope in ["joint","query_only","LN_only"]
 base.restore(initial);rp=TrickReplay(base,ids,key,{})
 origin=rp.state();allnamed=list(rp.named)
 selected=[(n,p) for n,p in allnamed if scope=="joint" or (scope=="query_only")== (n=="spatial.query_residual")]
 params=[p for _,p in selected];names=[n for n,_ in selected]
 active=sum(p.numel() for p in params);assert active=={"joint":1792,"query_only":256,"LN_only":1536}[scope]
 assert sum(p.numel() for _,p in allnamed)==1792
 with torch.no_grad():before=rp.values()['boxes'].detach().clone()
 # Reuse the original admission + highest score within each admitted frame.
 support=Energy(expert,before,'top1')
 positions=[int(f[0]) for f in support.frames]
 evidence=torch.cat([f[1] for f in support.frames],0).detach() if positions else before.new_empty((0,4))
 opt=torch.optim.Adam(params,lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
 path=[];audits=[];frozen=None;backwards=0
 try:
  for k in range(cfg['steps']):
   boxes=rp.values()['boxes'];entry=dict(step=k,boxes=detached(boxes,'cpu'),state=detached(rp.state(),'cpu'))
   path.append(entry)
   if not positions:
    audits.append(dict(round=k,updated=False,no_information=True));continue
   if direct:
    from methods.decota_final_simplified_v1.objectives import generalized_iou
    pred=boxes[positions];loss=(5*(pred-evidence).abs().sum(-1)+2*(1-generalized_iou(pred,evidence))).mean()
    update=optimizer_step(opt,loss,params,True)
    entry['update']=dict(**update,optimizer='adam',lr=cfg['lr'],names=names);backwards+=1
    with torch.no_grad():post=rp.values()['boxes']
    audits.append(dict(round=k,updated=True,no_information=False,loss=float(loss.detach()),
     direct_L1_GIoU=True,weights={'L1':5.,'GIoU':2.},
     central_reward_before=aligned_iou(pred.detach(),evidence).cpu(),central_reward_after=aligned_iou(post[positions],evidence).cpu()))
    continue
   mu=mean_coordinates(boxes[positions])
   if arm=='frozen_rollout' and frozen is not None:rr=frozen
   else:
    rr=rollout(mu.detach(),evidence,key,k,arm=='shuffled_feedback',config=cfg)
    if arm=='frozen_rollout':frozen=rr
   loss=likelihood_loss(mu,rr['mean'],rr['samples'],rr['weights'],sigma=cfg['sigma'])
   gmu=torch.autograd.grad(loss,mu,retain_graph=True)[0]
   analytic=((mu.detach()-rr['mean'])/cfg['sigma']**2-
    ((rr['weights']-1/32)[:,:,None]*(rr['samples']-mu.detach()[:,None,:])).sum(1)/cfg['sigma']**2)/len(positions)
   error=float((gmu-analytic).abs().max());assert error<2e-5
   update=optimizer_step(opt,loss,params,rr['informative'])
   if update is not None:entry['update']=dict(**update,optimizer='adam',lr=cfg['lr'],names=names);backwards+=1
   with torch.no_grad():post=rp.values()['boxes'];newmu=mean_coordinates(post[positions])
   mask=torch.ones(len(ids),device=boxes.device,dtype=torch.bool);mask[positions]=False
   aud=dict(round=k,updated=update is not None,no_information=not rr['informative'],loss=float(loss.detach()),
    reward_variance=rr['rewards'].var(1,unbiased=False).cpu(),weight_max=rr['weights'].max(1).values.cpu(),
    ESS=(1/rr['weights'].square().sum(1)).cpu(),mean_before=mu.detach().cpu(),mean_after=newmu.cpu(),
    rollout=detached(rr,'cpu'),autograd_mean_gradient=gmu.detach().cpu(),analytic_gradient_error=error,
    observed_movement=float((post[positions]-boxes[positions].detach()).abs().mean()),
    unobserved_movement=float((post[mask]-boxes[mask].detach()).abs().mean()) if mask.any() else None,
    central_reward_before=aligned_iou(boxes[positions].detach(),evidence).cpu(),
    central_reward_after=aligned_iou(post[positions],evidence).cpu(),
    weighted_reward=float((rr['weights']*rr['rewards']).sum(1).mean()),
    mean_movement=float((newmu-mu.detach()).norm()),gradient_only_student_likelihood=True)
   audits.append(aud)
  finalstate=detached(rp.state(),'cpu')
  with torch.no_grad():final=detached(rp.values()['boxes'],'cpu')
  path.append(dict(step=cfg['steps'],boxes=final,state=finalstate))
  if not positions:assert torch.equal(final,before.cpu()) and all(torch.equal(finalstate[n],origin[n].cpu()) for n in origin)
  for n,p in allnamed:
   if n not in names:
    assert all(torch.equal(h['state'][n],origin[n].cpu()) for h in path)
  return dict(arm=arm,initial=detached(origin,'cpu'),state=finalstate,before=before.cpu(),final=final,
   selected_step=cfg['steps'],path=path,rounds=audits,gradient_calls=backwards,active_parameters=active,scope=scope,direct_objective=direct,optimizer_parameter_names=names,
   positions=positions,frame_metadata=support.frame_metadata,empty=not positions,GT_used=False,config=cfg)
 finally:rp.restore(origin)


def fit_variant(base,initial,expert,ids,key,variant,config):
 """Full dispatches to the registered implementation; isolated controls change only declared factors."""
 from vg_tta.decota_spatial_opd_tunable_v1 import fit as main_fit
 if variant in ARMS:
  return main_fit(base,initial,expert,ids,key,variant,config=config)
 if variant=='joint_alpha0':
  return main_fit(base,initial,expert,ids,key,'on_policy',config={**config,'writeback':0.})
 if variant=='direct_L1_GIoU':
  return component_fit(base,initial,expert,ids,key,'on_policy',config=config,scope='joint',direct=True)
 assert variant in ['query_only','LN_only']
 return component_fit(base,initial,expert,ids,key,'on_policy',config=config,scope=variant)
