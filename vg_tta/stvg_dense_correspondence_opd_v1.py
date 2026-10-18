"""Four-arm on-policy interface experiment; immutable image feedback, no readers.

Dense frames each contribute a mean likelihood term. Rank transfer matches the
current interpolation weight multiset, not a probability calibration claim.
"""
import hashlib
import torch
from methods.decota_final_simplified_v1.tensors import detached
from vg_tta.c1_enabling_tricks_v1 import TrickReplay
from vg_tta.decota_optimizer_posterior_r1_v1 import Energy, flat
from vg_tta.decota_spatial_opd_tunable_v1 import mean_coordinates,rollout,likelihood_loss,optimizer_step,fit as sparse_fit
from vg_tta.stvg_soft_correspondence_v1 import soft_iou
from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
ARMS=('A_sparse','B_raw','C_rank','D_interpolation')

@torch.no_grad()
def rank_weights(rewards,reference_weights):
 assert rewards.shape==reference_weights.shape and rewards.ndim==2
 order=torch.argsort(rewards,dim=1,stable=True);sr=rewards.gather(1,order)
 rw=reference_weights.sort(dim=1).values
 ties=(sr[:,:,None]==sr[:,None,:]);assigned=(ties.to(rw)@rw[:,:,None]).squeeze(-1)/ties.sum(-1)
 out=torch.empty_like(assigned).scatter_(1,order,assigned)
 flatrow=(rewards==rewards[:,:1]).all(1)
 return torch.where(flatrow[:,None],torch.full_like(out,1/rewards.shape[1]),out).detach()

def contract():
 import numpy as np
 gen=torch.Generator().manual_seed(20261018)
 r=torch.randn(10,32,generator=gen);r[0]=0;r[1,:8]=2;r[2]=torch.arange(32)//4
 w=torch.softmax(torch.randn(10,32,generator=gen),1);actual=rank_weights(r,w).numpy();expected=np.empty((10,32))
 for i in range(10):
  vals=r[i].numpy();sortedw=np.sort(w[i].numpy());order=np.argsort(vals,kind='stable')
  if np.all(vals==vals[0]):expected[i]=1/32;continue
  for v in np.unique(vals):
   inds=np.flatnonzero(vals[order]==v);expected[i,order[inds]]=float(sortedw[inds].mean())
 assert np.max(abs(expected-actual))<1e-7
 return dict(status='pass',rank_transfer_rows=10,ties_average=True,flat_uniform=True,max_error=float(np.max(abs(expected-actual))))

@torch.no_grad()
def dense_rollout(mean,softmaps,interpolated,key,k,cfg,arm):
 seed=int(hashlib.sha256(f'dense-correspondence-v1|{key}|{k}'.encode()).hexdigest()[:15],16)
 gen=torch.Generator(device='cpu').manual_seed(seed);noise=torch.randn(len(mean),16,4,generator=gen).to(mean);noise=torch.cat((noise,-noise),1)
 z=(mean[:,None]+cfg['sigma']*noise).detach();actions=z.sigmoid()
 geom=aligned_iou(actions,interpolated[:,None]).detach();gw=torch.softmax(geom/cfg['tau'],-1)
 raw=torch.stack([soft_iou(actions,m[:,None])for m in softmaps]).mean(0).to(mean)
 w=(rank_weights(raw,gw)if arm=='C_rank'else gw if arm=='D_interpolation'else torch.softmax(raw/cfg['tau'],-1))
 reward=geom if arm=='D_interpolation'else raw
 return dict(mean=mean.detach().clone(),samples=z,weights=w.detach(),rewards=reward.detach(),raw_rewards=raw,interpolation_rewards=geom,seed=seed,informative=bool((w!=1/32).any()))

def grad_check(loss,mu,rr,sigma):
 actual=torch.autograd.grad(loss,mu,retain_graph=True)[0]
 expected=((mu.detach()-rr['mean'])-((rr['weights']-1/32)[:,:,None]*(rr['samples']-mu.detach()[:,None])).sum(1))/(sigma*sigma*len(mu))
 error=float((actual-expected).abs().max());assert error<3e-5,error
 return actual.detach(),error

@torch.enable_grad()
def fit(base,initial,expert,ids,key,arm,*,config,softmaps,interpolated,unobserved):
 assert arm in ARMS
 if arm=='A_sparse':return sparse_fit(base,initial,expert,ids,key,'on_policy',config=config)
 cfg=dict(config);base.restore(initial);rp=TrickReplay(base,ids,key,{})
 origin=rp.state();params=[v for _,v in rp.named];names=[n for n,_ in rp.named];assert sum(v.numel()for v in params)==1792
 with torch.no_grad():before=rp.values()['boxes'].detach().clone()
 support=Energy(expert,before,'top1');positions=[int(f[0])for f in support.frames];evidence=torch.cat([f[1]for f in support.frames],0).detach()if positions else before.new_empty((0,4))
 dense=list(unobserved);maps=softmaps[:,dense].to(before);interpolated=interpolated[dense].to(before)
 assert len(maps)==len(positions) and not maps.requires_grad and not interpolated.requires_grad
 opt=torch.optim.Adam(params,lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
 path=[];audits=[];backwards=0
 try:
  for k in range(cfg['steps']):
   boxes=rp.values()['boxes'];entry=dict(step=k,boxes=detached(boxes,'cpu'),state=detached(rp.state(),'cpu'));path.append(entry)
   if not positions:
    audits.append(dict(round=k,updated=False,no_information=True));continue
   smu=mean_coordinates(boxes[positions]);sr=rollout(smu.detach(),evidence,key,k,config=cfg)
   sloss=likelihood_loss(smu,sr['mean'],sr['samples'],sr['weights'],cfg['sigma']);sg,se=grad_check(sloss,smu,sr,cfg['sigma'])
   if dense:
    dmu=mean_coordinates(boxes[dense]);dr=dense_rollout(dmu.detach(),maps,interpolated,key,k,cfg,arm)
    dloss=likelihood_loss(dmu,dr['mean'],dr['samples'],dr['weights'],cfg['sigma']);dg,de=grad_check(dloss,dmu,dr,cfg['sigma'])
   else:dloss=boxes.sum()*0;dr=None;dg=None;de=0.
   # Two separately averaged supports; coefficient one fixed before all fitting.
   loss=sloss+dloss;info=sr['informative']or(dr is not None and dr['informative']);update=optimizer_step(opt,loss,params,info)
   if update is not None:entry['update']=dict(**update,optimizer='adam',lr=cfg['lr'],names=names);backwards+=1
   with torch.no_grad():post=rp.values()['boxes'];movement=(post-boxes.detach()).abs().mean(1)
   audits.append(dict(round=k,updated=update is not None,no_information=not info,sparse_loss=float(sloss.detach()),dense_loss=float(dloss.detach()),loss=float(loss.detach()),sparse_rollout=detached(sr,'cpu'),dense_rollout=detached(dr,'cpu'),sparse_mean_gradient=sg.cpu(),dense_mean_gradient=None if dg is None else dg.cpu(),analytic_gradient_error=max(se,de),sparse_gradient_norm=float(sg.norm()),dense_gradient_norm=None if dg is None else float(dg.norm()),observed_movement=float(movement[positions].mean()),unobserved_movement=float(movement[dense].mean())if dense else None,parameter_step_norm=None if update is None else float(update['raw'].norm()),parameter_gradient_norm=None if update is None else float(update['gradient'].norm()),sparse_ESS=(1/sr['weights'].square().sum(1)).cpu(),dense_ESS=None if dr is None else(1/dr['weights'].square().sum(1)).cpu(),current_Student_resampled=True))
  finalstate=detached(rp.state(),'cpu')
  with torch.no_grad():final=detached(rp.values()['boxes'],'cpu')
  path.append(dict(step=cfg['steps'],boxes=final,state=finalstate))
  if not positions:assert torch.equal(final,before.cpu())
  return dict(arm=arm,initial=detached(origin,'cpu'),state=finalstate,before=before.cpu(),final=final,selected_step=cfg['steps'],path=path,rounds=audits,gradient_calls=backwards,active_parameters=1792,positions=positions,unobserved_positions=dense,empty=not positions,GT_used=False,config=cfg,dense_coefficient=1.,dense_per_frame_mean=True)
 finally:rp.restore(origin)
