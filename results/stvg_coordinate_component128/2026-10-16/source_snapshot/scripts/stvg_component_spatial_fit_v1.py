"""Matched deletion variants of the saved M2 fitter, in a new namespace.

Only declared boolean component switches differ. Fixed source, reference,
config, decoder and seed interfaces are inherited; this file is not deployed.
"""
import copy,time,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from methods.decota_final_simplified_v1.tensors import detached
from scripts import stvg_target_teacher_student_v1 as core
from scripts.stvg_weightfree_coordinate_fit002 import loss_per_anchor,accept_reference_step
from vg_tta.decota_spatial_opd_tunable_v1 import mean_coordinates,rollout
from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou

def flatten(values):return torch.cat([p.reshape(-1)for p in values])
def put(rp,state,d=None):
 with torch.no_grad():
  k=0
  for n,p in rp.named:
   active=n in getattr(rp,'_ablation_active_names',[x for x,_ in rp.named])
   if d is None or not active:p.copy_(state[n].to(p))
   else:p.copy_(state[n].to(p)+d[k:k+p.numel()].to(p).reshape_as(p));k+=p.numel()
   p.grad=None

def validate_config(cfg, mode):
    """Validate one of the pre-locked coordinate configurations."""
    if mode not in {'M1', 'M2'}:
        raise ValueError('generalized Soft EP is only used for M1/M2')
    if not isinstance(cfg, dict):
        raise TypeError('config must be a mapping')
    if isinstance(cfg.get('steps'), bool) or not isinstance(cfg.get('steps'), int) or not 1 <= cfg['steps'] <= 8:
        raise ValueError('steps must be an integer in [1, 8]')
    if cfg.get('samples') != 32:
        raise ValueError('the locked antithetic sample count must remain 32')
    from scripts import stvg_target_teacher_student_v1 as core
    if cfg.get('writeback') == 0.0625:
        dataset = 'hc2'
    elif cfg.get('writeback') == 0.125:
        dataset = 'vidstg'
    else:
        raise ValueError('writeback must match a locked source domain')
    original = core.load_original_opd_config(dataset)
    tunable = {'lr', 'sigma', 'tau', 'steps'}
    if set(cfg) != set(original):
        raise ValueError('unexpected OPD config fields')
    if any(cfg[k] != original[k] for k in set(original) - tunable):
        raise ValueError('non-coordinate OPD settings differ from the original config')
    for name, low, high in (('lr', 1e-6, 1.5), ('sigma', 1e-4, 0.4), ('tau', 1e-4, 2.0)):
        value = cfg.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not low <= float(value) <= high:
            raise ValueError(f'{name} is outside the fixed coordinate-search bounds [{low}, {high}]')
    return dataset

ALLOWED_FLAGS={'uniform_reliability','no_control_variate','no_policy_KL','iid_sampling','no_reference_gate','no_response_gate','no_backtracking','no_rejection_stop','no_flat_skip','no_equal_norm_diagnostic','query_only','ln_only','no_soft_direction'}
def loss_ablation(mean,policy,sigma,flags):
 if not flags.get('no_control_variate')and not flags.get('no_policy_KL'):return loss_per_anchor(mean,policy,sigma)
 kl=(mean-policy['mean']).square().sum(-1)/(2*sigma**2)
 logp=-(policy['samples']-mean[:,None,:]).square().sum(-1)/(2*sigma**2)-2*math.log(2*math.pi*sigma**2)
 return (torch.zeros_like(kl)if flags.get('no_policy_KL')else kl)-((policy['weights']-(0. if flags.get('no_control_variate')else 1/32))*logp).sum(-1)
@torch.no_grad()
def rollout_ablation(mean,evidence,key,k,cfg,flags):
 if not flags.get('iid_sampling'):return rollout(mean,evidence,key,k,config=cfg)
 import hashlib
 from vg_tta.decota_spatial_opd_tunable_v1 import action_boxes
 seed=int(hashlib.sha256(f'spatial-opd-v1|{key}|{k}'.encode()).hexdigest()[:15],16);gen=torch.Generator(device='cpu').manual_seed(seed);noise=torch.randn(len(mean),32,4,generator=gen,dtype=torch.float32).to(mean);z=(mean[:,None,:]+cfg['sigma']*noise).detach();r=aligned_iou(action_boxes(z),evidence[:,None,:]).detach();w=torch.softmax(r/cfg['tau'],-1).detach()
 return dict(mean=mean.detach().clone(),samples=z,rewards=r,used_rewards=r,weights=w,permutations=None,informative=bool((r!=r[:,:1]).any()),seed=seed)

def fit(base,initial,anchors,ids,key,cfg,mode,settings,qualification=False,checkpoint=None,flags=None):
 from scripts.stvg_soft_ep_response_v1 import solve_from_graph,graph_jvp
 validate_config(cfg, mode);flags=dict(flags or {});assert set(flags)<=ALLOWED_FLAGS and all(isinstance(v,bool)for v in flags.values())
 assert not(flags.get('query_only')and flags.get('ln_only'))
 begin=time.perf_counter()
 base.restore(initial);rp,selected=core._trainable_replay(base,ids,key,'joint');origin=rp.state();assert sum(p.numel()for _,p in selected)==1792
 if flags.get('query_only'):selected=[(n,p)for n,p in selected if n=='spatial.query_residual']
 if flags.get('ln_only'):selected=[(n,p)for n,p in selected if n!='spatial.query_residual']
 assert not(flags.get('query_only')and flags.get('ln_only'))
 rp._ablation_active_names=[n for n,_ in selected]
 for n,p in rp.named:p.requires_grad_(n in rp._ablation_active_names)
 params=[p for _,p in selected]
 active=[a for a in anchors if a['weight']>0];pos=[a['position']for a in active];u=[i for i in range(len(ids))if i not in set(pos)]
 evidence=torch.stack([a['box'].to(base.zero['boxes'])for a in active]);weights=base.zero['boxes'].new_tensor([a['weight']for a in active])
 if flags.get('uniform_reliability'):weights=torch.ones_like(weights)
 shadows=[torch.nn.Parameter(p.detach().clone())for p in params];opt=torch.optim.Adam(shadows,lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
 path=[];accepted=proposals=0;stopped=False
 try:
  for k in range(cfg['steps']):
   state=rp.state();before=rp.values()['boxes'];b=before.detach().clone();ob=core._copy_cpu(opt.state_dict());row=dict(step=k,state_before=detached(state,'cpu'),boxes_before=b.cpu(),optimizer_before=ob)
   if stopped:
    row.update(skipped=True,reason='stopped_after_rejected_step',accepted=False);path.append(row);continue
   mean=mean_coordinates(before[pos]);policy=rollout_ablation(mean,evidence,key,k,cfg,flags);pa=loss_ablation(mean,policy,cfg['sigma'],flags);loss=(weights*pa).mean()
   assert torch.isfinite(loss)
   row.update(loss=float(loss.detach()),per_anchor_loss=pa.detach().cpu(),rollout=detached(policy,'cpu'),reliability=weights.cpu(),reference_iou_before=aligned_iou(b[pos],evidence).cpu())
   if not policy['informative']and not flags.get('no_flat_skip'):
    row.update(skipped=True,reason='no_information',accepted=False);path.append(row);continue
   grads=torch.autograd.grad(loss,params,retain_graph=True);assert all(torch.isfinite(g).all()for g in grads)
   opt.zero_grad(set_to_none=True)
   with torch.no_grad():
    for sh,p,g in zip(shadows,params,grads):sh.copy_(p);sh.grad=g.detach().clone()
   opt.step();d=flatten([sh.detach()-p.detach()for sh,p in zip(shadows,params)]);op=core._copy_cpu(opt.state_dict());proposals+=1
   solver={};qcheck=None
   if mode=='M2' and u and d.norm()>0 and not flags.get('no_soft_direction'):
    y=before[u].flatten()/math.sqrt(len(u)*4)
    direction,solver=solve_from_graph(y,params,d,mu=settings['mu'],max_iter=settings['cg_max_iter'],rtol=settings['cg_rtol'])
    if qualification and k==0:
     gen=torch.Generator(device='cpu').manual_seed(20261015);w=torch.randn(len(y),generator=gen).to(y);w=w/w.norm();v=d.detach()/d.norm()
     jv=graph_jvp(y,params,v);jt=flatten(torch.autograd.grad(y,params,grad_outputs=w,retain_graph=True))
     left=torch.dot(w,jv);right=torch.dot(jt,v);qcheck=dict(adjoint_left=float(left),adjoint_right=float(right),adjoint_abs_error=float(abs(left-right)),adjoint_pass=bool(abs(left-right)<=1e-7+1e-4*max(abs(left),abs(right))))
     qcheck.update(unit_direction=v.cpu(),linear_jvp=jv.detach().cpu(),output_before=y.detach().cpu(),cotangent=w.cpu(),vjp=jt.detach().cpu())
   else:direction=d.clone()
   # All graph derivatives finish before mutating leaf parameter versions.
   if qcheck is not None:
    fd=[]
    for eps in [.002,.001]:
     put(rp,state,qcheck['unit_direction'].to(d)*eps)
     with torch.no_grad():yp=rp.values()['boxes'][u].flatten()/math.sqrt(len(u)*4)
     put(rp,state,-qcheck['unit_direction'].to(d)*eps)
     with torch.no_grad():ym=rp.values()['boxes'][u].flatten()/math.sqrt(len(u)*4)
     observed=(yp-ym)/(2*eps);pred=qcheck['linear_jvp'].to(observed);err=(observed-pred).norm()
     fd.append(dict(h=eps,plus=yp.cpu(),minus=ym.cpu(),absolute_error=float(err),predicted_norm=float(pred.norm()),relative_error=float(err/pred.norm().clamp_min(1e-12)),gross_agreement=bool(err<=1e-3+.25*pred.norm())))
    qcheck['finite_differences']=fd;put(rp,state)
   # Original Adam proposal defines one common relative response budget.
   with torch.no_grad():
    for (_,p),sh in zip(selected,shadows):p.copy_(sh)
    original=rp.values()['boxes'].detach().clone()
   original_rms=(original[u]-b[u]).square().mean().sqrt()if u else b.new_tensor(0.)
   budget=original_rms*settings['response_ratio'];norm_control=None
   if mode=='M2' and d.norm()>0 and not flags.get('no_equal_norm_diagnostic'):
    matched=d*(direction.norm()/d.norm());put(rp,state,matched)
    with torch.no_grad():mb=rp.values()['boxes'].detach().clone()
    put(rp,state,direction)
    with torch.no_grad():sb=rp.values()['boxes'].detach().clone()
    norm_control=dict(original_direction=d.cpu(),soft_direction=direction.cpu(),matched_direction=matched.cpu(),matched_boxes=mb.cpu(),soft_boxes=sb.cpu(),matched_U_RMS=float((mb[u]-b[u]).square().mean().sqrt())if u else 0.,soft_U_RMS=float((sb[u]-b[u]).square().mean().sqrt())if u else 0.,matched_anchor_IoU=aligned_iou(mb[pos],evidence).cpu(),soft_anchor_IoU=aligned_iou(sb[pos],evidence).cpu())
   candidates=[];approved=False;scale=0.;chosen=b
   for a in ([1.]if flags.get('no_backtracking')else settings['backtracking']):
    if mode=='M1' and a==1.:
     with torch.no_grad():
      for (_,p),sh in zip(selected,shadows):p.copy_(sh)
    else:put(rp,state,direction*a)
    with torch.no_grad():candidate=rp.values()['boxes'].detach().clone()
    riou=aligned_iou(candidate[pos],evidence);rms=(candidate[u]-b[u]).square().mean().sqrt()if u else b.new_tensor(0.)
    train_pass=flags.get('no_reference_gate',False)or accept_reference_step(row['reference_iou_before'].to(riou),riou,weights);response_pass=flags.get('no_response_gate',False)or bool(rms<=budget+settings['response_abs_tolerance']);ok=train_pass and response_pass and bool(torch.isfinite(candidate).all())
    candidates.append(dict(scale=a,train_pass=train_pass,response_pass=response_pass,accepted=ok,anchor_IoU=riou.cpu(),U_RMS=float(rms)))
    if ok:approved=True;scale=a;chosen=candidate;break
   if approved:accepted+=1
   else:
    put(rp,state);opt.load_state_dict(copy.deepcopy(ob));stopped=not flags.get('no_rejection_stop');chosen=b
    assert core._tree_equal(core._copy_cpu(opt.state_dict()),ob)
   final=rp.state()
   row.update(skipped=False,accepted=approved,rollback=not approved,gradient=flatten(grads).detach().cpu(),original_adam_delta=d.detach().cpu(),filtered_delta=direction.detach().cpu(),accepted_scale=scale,optimizer_proposed=op,optimizer_after=core._copy_cpu(opt.state_dict()),state_after=detached(final,'cpu'),boxes_after=chosen.cpu(),original_proposal_boxes=original.cpu(),response_original_RMS=float(original_rms),response_budget=float(budget),candidates=candidates,solver=solver,norm_control=norm_control,derivative_qualification=qcheck)
   path.append(row)
   if checkpoint is not None:checkpoint(dict(next_step=k+1,path=path,state=detached(final,'cpu'),GT_read=False))
  with torch.no_grad():boxes=rp.values()['boxes'].detach().cpu()
  return dict(schema='stvg-soft-ep-coordinate-fit-v1',mode=mode,initial_state=detached(origin,'cpu'),final_state=detached(rp.state(),'cpu'),fit_final_state=detached(rp.state(),'cpu'),final_cached_decoder_boxes=boxes,path=path,optimizer_steps=accepted,proposed_optimizer_steps=proposals,active_parameters=sum(p.numel()for p in params),active_names=list(rp._ablation_active_names),ablation_flags=dict(flags),config=dict(cfg),settings=copy.deepcopy(settings),seed_key=key,observation_positions=pos,protection_positions=u,GT_read=False,wall_seconds=time.perf_counter()-begin)
 finally:
  rp.restore(origin);base.restore(initial)
  if hasattr(rp,'_ablation_active_names'):del rp._ablation_active_names
