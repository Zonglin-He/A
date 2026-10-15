"""Post-Adam decoder-response direction filter. Fixed references; no GT."""
import copy,time,math
import torch
from methods.decota_final_simplified_v1.tensors import detached
from scripts import stvg_target_teacher_student_v1 as core
from scripts.stvg_weightfree_coordinate_fit_v1 import loss_per_anchor,accept_reference_step
from vg_tta.decota_spatial_opd_tunable_v1 import mean_coordinates,rollout
from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou

def flatten(values):return torch.cat([p.reshape(-1)for p in values])
def put(rp,state,d=None):
 with torch.no_grad():
  k=0
  for n,p in rp.named:
   p.copy_(state[n].to(p)if d is None else state[n].to(p)+d[k:k+p.numel()].to(p).reshape_as(p));p.grad=None;k+=p.numel()

def fit(base,initial,anchors,ids,key,cfg,mode,settings,qualification=False,checkpoint=None):
 from scripts.stvg_soft_ep_response_v1 import solve_from_graph,graph_jvp
 assert mode in ['M1','M2'] and cfg['steps']==5 and cfg['samples']==32
 begin=time.perf_counter();base.restore(initial);rp,selected=core._trainable_replay(base,ids,key,'joint');origin=rp.state();params=[p for _,p in selected];assert sum(p.numel()for p in params)==1792
 active=[a for a in anchors if a['weight']>0];pos=[a['position']for a in active];u=[i for i in range(len(ids))if i not in set(pos)]
 evidence=torch.stack([a['box'].to(base.zero['boxes'])for a in active]);weights=base.zero['boxes'].new_tensor([a['weight']for a in active])
 shadows=[torch.nn.Parameter(p.detach().clone())for p in params];opt=torch.optim.Adam(shadows,lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
 path=[];accepted=proposals=0;stopped=False
 try:
  for k in range(cfg['steps']):
   state=rp.state();before=rp.values()['boxes'];b=before.detach().clone();ob=core._copy_cpu(opt.state_dict());row=dict(step=k,state_before=detached(state,'cpu'),boxes_before=b.cpu(),optimizer_before=ob)
   if stopped:
    row.update(skipped=True,reason='stopped_after_rejected_step',accepted=False);path.append(row);continue
   mean=mean_coordinates(before[pos]);policy=rollout(mean,evidence,key,k,config=cfg);pa=loss_per_anchor(mean,policy,cfg['sigma']);loss=(weights*pa).mean()
   assert torch.isfinite(loss)
   row.update(loss=float(loss.detach()),per_anchor_loss=pa.detach().cpu(),rollout=detached(policy,'cpu'),reliability=weights.cpu(),reference_iou_before=aligned_iou(b[pos],evidence).cpu())
   if not policy['informative']:
    row.update(skipped=True,reason='no_information',accepted=False);path.append(row);continue
   grads=torch.autograd.grad(loss,params,retain_graph=True);assert all(torch.isfinite(g).all()for g in grads)
   opt.zero_grad(set_to_none=True)
   with torch.no_grad():
    for sh,p,g in zip(shadows,params,grads):sh.copy_(p);sh.grad=g.detach().clone()
   opt.step();d=flatten([sh.detach()-p.detach()for sh,p in zip(shadows,params)]);op=core._copy_cpu(opt.state_dict());proposals+=1
   solver={};qcheck=None
   if mode=='M2' and u and d.norm()>0:
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
    for (_,p),sh in zip(rp.named,shadows):p.copy_(sh)
    original=rp.values()['boxes'].detach().clone()
   original_rms=(original[u]-b[u]).square().mean().sqrt()if u else b.new_tensor(0.)
   budget=original_rms*settings['response_ratio'];norm_control=None
   if mode=='M2' and d.norm()>0:
    matched=d*(direction.norm()/d.norm());put(rp,state,matched)
    with torch.no_grad():mb=rp.values()['boxes'].detach().clone()
    put(rp,state,direction)
    with torch.no_grad():sb=rp.values()['boxes'].detach().clone()
    norm_control=dict(original_direction=d.cpu(),soft_direction=direction.cpu(),matched_direction=matched.cpu(),matched_boxes=mb.cpu(),soft_boxes=sb.cpu(),matched_U_RMS=float((mb[u]-b[u]).square().mean().sqrt())if u else 0.,soft_U_RMS=float((sb[u]-b[u]).square().mean().sqrt())if u else 0.,matched_anchor_IoU=aligned_iou(mb[pos],evidence).cpu(),soft_anchor_IoU=aligned_iou(sb[pos],evidence).cpu())
   candidates=[];approved=False;scale=0.;chosen=b
   for a in settings['backtracking']:
    if mode=='M1' and a==1.:
     with torch.no_grad():
      for (_,p),sh in zip(rp.named,shadows):p.copy_(sh)
    else:put(rp,state,direction*a)
    with torch.no_grad():candidate=rp.values()['boxes'].detach().clone()
    riou=aligned_iou(candidate[pos],evidence);rms=(candidate[u]-b[u]).square().mean().sqrt()if u else b.new_tensor(0.)
    train_pass=accept_reference_step(row['reference_iou_before'].to(riou),riou,weights);response_pass=bool(rms<=budget+settings['response_abs_tolerance']);ok=train_pass and response_pass and bool(torch.isfinite(candidate).all())
    candidates.append(dict(scale=a,train_pass=train_pass,response_pass=response_pass,accepted=ok,anchor_IoU=riou.cpu(),U_RMS=float(rms)))
    if ok:approved=True;scale=a;chosen=candidate;break
   if approved:accepted+=1
   else:
    put(rp,state);opt.load_state_dict(copy.deepcopy(ob));stopped=True;chosen=b
    assert core._tree_equal(core._copy_cpu(opt.state_dict()),ob)
   final=rp.state()
   row.update(skipped=False,accepted=approved,rollback=not approved,gradient=flatten(grads).detach().cpu(),original_adam_delta=d.detach().cpu(),filtered_delta=direction.detach().cpu(),accepted_scale=scale,optimizer_proposed=op,optimizer_after=core._copy_cpu(opt.state_dict()),state_after=detached(final,'cpu'),boxes_after=chosen.cpu(),original_proposal_boxes=original.cpu(),response_original_RMS=float(original_rms),response_budget=float(budget),candidates=candidates,solver=solver,norm_control=norm_control,derivative_qualification=qcheck)
   path.append(row)
   if checkpoint is not None:checkpoint(dict(next_step=k+1,path=path,state=detached(final,'cpu'),GT_read=False))
  with torch.no_grad():boxes=rp.values()['boxes'].detach().cpu()
  return dict(schema='stvg-soft-ep-fit-v1',mode=mode,initial_state=detached(origin,'cpu'),final_state=detached(rp.state(),'cpu'),fit_final_state=detached(rp.state(),'cpu'),final_cached_decoder_boxes=boxes,path=path,optimizer_steps=accepted,proposed_optimizer_steps=proposals,active_parameters=1792,config=cfg,settings=settings,seed_key=key,observation_positions=pos,protection_positions=u,GT_read=False,wall_seconds=time.perf_counter()-begin)
 finally:rp.restore(origin);base.restore(initial)
