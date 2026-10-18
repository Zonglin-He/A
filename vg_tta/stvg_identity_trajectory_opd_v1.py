"""Frozen association hypotheses and finite, normalized whole-tube policies.

Association weights are evidence, not calibrated identity correctness. No labels.
All proposed tubes are actions; final prediction must be actual Student decode.
"""
import hashlib,math
import numpy as np
import torch
from methods.decota_final_simplified_v1.tensors import detached
from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
from vg_tta.decota_spatial_opd_tunable_v1 import mean_coordinates

@torch.no_grad()
def hypotheses(base,expert,ids,membership):
 from torchvision.ops import roi_align
 # Frozen Source query-conditioned appearance, identical for every arm.
 maps=[]
 for view in base.views:
  H=base.norm(view['prefix']).detach().float();h,w=view['info']['fea_map_size'];n=h*w
  maps.append(H[:n].permute(1,2,0).reshape(H.shape[1],256,h,w))
 pools=[];positions=[]
 for (_,pos),ob in sorted(expert['observations'].items()):
  p=ob['probe'];b=torch.as_tensor(p['boxes'],device=base.delta.device,dtype=torch.float32);s=torch.as_tensor(p['target_scores'],device=b.device,dtype=b.dtype)
  valid=torch.isfinite(b).all(1)&(b[:,2:]>0).all(1)&(s>=.35)&((b>0)&(b<1)).all(1);b=b[valid];s=s[valid]
  if not len(b):continue
  feat=maps[pos%2][pos//2:pos//2+1];h,w=feat.shape[-2:];xy=torch.cat((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),1).clamp(0,1)
  rois=torch.cat((b.new_zeros(len(b),1),xy*b.new_tensor([w,h,w,h])),1);f=roi_align(feat,rois,(2,2),aligned=True,sampling_ratio=2).mean((-1,-2));f=torch.nn.functional.normalize(f,dim=-1)
  pools.append((b,s,f));positions.append(pos)
 if not pools:return dict(empty=True,positions=[],GT_used=False)
 # Beam paths can skip a frame, but cannot switch independently at reward time.
 beam=[(0.,[],None,0)]
 for b,s,f in pools:
  expanded=[]
  for score,path,last,nedge in beam:
   expanded.append((score+math.log(.35),path+[-1],last,nedge))
   for j in range(len(b)):
    edge=0. if last is None else float((1-torch.dot(last,f[j]))/2)
    expanded.append((score+math.log(float(s[j]))-edge,path+[j],f[j],nedge+(last is not None)))
  beam=sorted(expanded,key=lambda x:(-x[0],x[1]))[:8]
 paths=[v for v in beam if any(j>=0 for j in v[1])][:3];paths.append((len(pools)*math.log(.35),[-1]*len(pools),None,0))
 logw=torch.tensor([v[0]/len(pools)for v in paths],device=base.delta.device);logw=logw-torch.logsumexp(logw,0);targets=[];mask=[]
 for _,path,_,_ in paths:
  targets.append(torch.stack([pools[k][0][j]if j>=0 else pools[k][0].new_zeros(4)for k,j in enumerate(path)]));mask.append([j>=0 for j in path])
 weights=torch.as_tensor(membership[positions],device=base.delta.device,dtype=torch.float32);denom=weights.sum()
 if float(denom)>0:weights=weights/denom
 return dict(empty=not paths[:-1]or float(denom)==0,positions=positions,targets=torch.stack(targets),mask=torch.tensor(mask,device=base.delta.device,dtype=torch.bool),logw=logw,temporal_weights=weights,paths=[v[1]for v in paths],candidate_counts=[len(x[0])for x in pools],GT_used=False,association='cached query-conditioned frozen appearance + detector evidence; uncalibrated',outside_event_evidence_retained=True)

def reward(boxes,hyp,tau):
 # [...,N,4] -> [...,H]. Accumulate one identity first, marginalize second.
 p=boxes[...,hyp['positions'],:];ious=aligned_iou(p[...,None,:,:],hyp['targets']);w=hyp['temporal_weights']*hyp['mask'];compat=(ious*w).sum(-1)
 return tau*torch.logsumexp(hyp['logw']+compat/tau,dim=-1)

def covariance(ids,device,dtype,structured):
 n=len(ids)
 if not structured:return torch.eye(n,device=device,dtype=dtype)
 t=torch.as_tensor(ids,device=device,dtype=dtype);t=(t-t.mean())/(t.std(unbiased=False).clamp_min(1))
 K=.5*torch.eye(n,device=device,dtype=dtype)+.25*torch.ones(n,n,device=device,dtype=dtype)+.25*t[:,None]*t[None,:]
 d=K.diag().sqrt();return K/d[:,None]/d[None,:]

@torch.no_grad()
def proposal_offsets(mean,hyp,ids,structured):
 if not structured:return mean.new_zeros((1,*mean.shape)),mean.new_zeros(1)
 offsets=[];t=np.asarray(ids,dtype=float)
 for h in range(len(hyp['logw'])):
  valid=hyp['mask'][h];pos=np.array(hyp['positions'])[valid.cpu().numpy()]
  if not len(pos):offsets.append(torch.zeros_like(mean));continue
  ref=mean_coordinates(hyp['targets'][h,valid]);d=(ref-mean[pos]).cpu().numpy()
  # Individual identity proposals, never an average across hypotheses.
  offsets.append(torch.as_tensor(np.stack([np.interp(t,t[pos],d[:,c])for c in range(4)],1),device=mean.device,dtype=mean.dtype))
 return torch.stack(offsets),hyp['logw']

def log_density(samples,mean,offsets,logw,inv,logdet,sigma):
 residual=samples[:,None]-mean[None,None]-offsets[None]
 quad=torch.einsum('mhnd,nk,mhkd->mh',residual,inv,residual)/sigma**2
 normalizer=4*(len(mean)*math.log(2*math.pi*sigma**2)+logdet)
 return torch.logsumexp(logw[None]-.5*(quad+normalizer),dim=1)

@torch.enable_grad()
def fit(base,initial,hyp,ids,key,arm,cfg):
 from vg_tta.c1_enabling_tricks_v1 import TrickReplay
 from vg_tta.decota_optimizer_posterior_r1_v1 import flat
 assert arm in ['multi_frame','multi_tube','multi_direct']
 base.restore(initial);rp=TrickReplay(base,ids,key,{});origin=rp.state();params=[p for _,p in rp.named];assert sum(p.numel()for p in params)==1792
 opt=torch.optim.Adam(params,lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.);aud=[];path=[];grad=0
 with torch.no_grad():before=rp.values()['boxes'].detach().clone()
 structured=arm=='multi_tube';policy_positions=list(range(len(ids)))if structured else hyp.get('positions',[])
 if not hyp['empty']:
  K=covariance([ids[i]for i in policy_positions],before.device,torch.float64,structured);chol=torch.linalg.cholesky(K);inv=torch.cholesky_inverse(chol);ld=2*chol.diag().log().sum()
 try:
  for k in range(cfg['steps']):
   boxes=rp.values()['boxes'];path.append(dict(step=k,boxes=boxes.detach().cpu()));opt.zero_grad(set_to_none=True)
   if hyp['empty']:aud.append(dict(step=k,no_information=True,updated=False));continue
   if arm=='multi_direct':
    loss=-reward(boxes,hyp,cfg['tau']);rr=dict(reward_before=float(-loss.detach()),no_information=False)
   else:
    mu=mean_coordinates(boxes[policy_positions]).double();old=mu.detach().clone();offs,lw=proposal_offsets(old,hyp,ids,True)if structured else(old.new_zeros((1,*old.shape)),old.new_zeros(1))
    seed=int(hashlib.sha256(f'identity-tube-v1|{key}|{k}'.encode()).hexdigest()[:15],16);gen=torch.Generator().manual_seed(seed)
    eps=torch.randn((16,len(old),4),generator=gen,dtype=torch.float64);eps=torch.cat((eps,-eps),0).to(old);components=torch.multinomial(lw.exp().cpu(),32,replacement=True,generator=gen).to(old.device)
    noise=torch.einsum('nk,mkd->mnd',chol,eps);samples=(old[None]+offs[components]+cfg['sigma']*noise).detach()
    actions=boxes.detach().double()[None].expand(32,-1,-1).clone();actions[:,policy_positions]=samples.sigmoid();rw=reward(actions,hyp,cfg['tau']).detach();weights=torch.softmax(rw/cfg['tau'],0).detach();informative=bool((rw!=rw[:1]).any())
    logp=log_density(samples,mu,offs,lw,inv,ld,cfg['sigma']);loss=-((weights-1/32)*logp).sum()/len(policy_positions)
    rr=dict(seed=seed,samples=samples.cpu(),offsets=offs.cpu(),logw=lw.cpu(),components=components.cpu(),rewards=rw.cpu(),weights=weights.cpu(),old_mean=old.cpu(),log_density=logp.detach().cpu(),loss=float(loss.detach()),no_information=not informative,ESS=float(1/weights.square().sum()),policy_full_tube=structured)
   assert torch.isfinite(loss)
   if rr['no_information']:rr['updated']=False
   else:
    g=torch.autograd.grad(loss,params);assert all(torch.isfinite(v).all()for v in g)
    start=flat(params).detach().clone()
    for p,v in zip(params,g):p.grad=v
    opt.step();rr.update(updated=True,gradient=flat(g).detach().cpu(),parameter_delta=(flat(params).detach()-start).cpu());grad+=1
   with torch.no_grad():rr['reward_after']=float(reward(rp.values()['boxes'],hyp,cfg['tau']))
   rr['step']=k;aud.append(rr)
  state=detached(rp.state(),'cpu')
  with torch.no_grad():final=rp.values()['boxes'].detach().cpu()
  path.append(dict(step=cfg['steps'],boxes=final));assert torch.isfinite(final).all()
  return dict(initial=detached(origin,'cpu'),state=state,before=before.cpu(),final=final,rounds=aud,path=path,gradient_calls=grad,active_parameters=1792,empty=hyp['empty'],config=cfg,GT_used=False,arm=arm,policy_positions=policy_positions)
 finally:rp.restore(origin)
