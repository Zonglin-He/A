"""R51 root-called episodic update primitive; no file/model/G/GT loaders.

Preferences, P/N groups, A/B correspondences, G evidence and output box pools
must be sealed before this function. No teacher refresh or candidate remap.
It returns 0/1/3-step scores on the SAME cached canonical slots and restores
source parameters; root owns forward/readout bridges and final scoring.
"""
import math
import numpy as np
import torch
from model51 import replay_matching
from lora51 import QVLoRA,MatchingMLP


def checked_groups(cache,pair,require_pair=True):
 if pair is None:pair={'P':[],'N':[]}
 assert set(pair)=={'P','N'}
 p=np.asarray(pair['P'],np.int64);n=np.asarray(pair['N'],np.int64)
 assert p.ndim==n.ndim==1
 if require_pair:assert len(p) and len(n)
 assert len(set(p.tolist()))==len(p) and len(set(n.tolist()))==len(n)
 assert ((p>=0)&(p<256)).all() and ((n>=0)&(n<256)).all() and not set(p)&set(n)
 outside=np.array(sorted(set(range(256))-set(p)-set(n)),np.int64)
 anchor=int(np.asarray(cache['scores']).argmax())
 return p,n,outside,anchor


def objective(outputs,baseline,groups,tau=.1,beta=1.):
 """Original max-group RankNet and R30 relative-z keep, current input only."""
 assert tau==.1 and beta==1.
 gaps=[];keeps=[]
 for view in ('A','B'):
  z=outputs[view]['logodds'];p,n,outside,anchor=groups[view]
  p=torch.as_tensor(p,device=z.device);n=torch.as_tensor(n,device=z.device)
  gaps.append(z[p].max()-z[n].max())
  if len(outside):
   u=torch.as_tensor(outside,device=z.device)
   difference=z-z[anchor];initial=baseline[view]-baseline[view][anchor]
   keeps.append((difference[u]-initial[u]).square().mean())
 rank=torch.nn.functional.softplus(-torch.stack(gaps)/tau).mean()
 keep=torch.stack(keeps).mean() if keeps else rank.new_zeros(())
 return rank+beta*keep,rank,keep,torch.stack(gaps)


def episodic(model,caches,pairs,*,interface,accepted,lr=.0005,
             initial_probability_bound=2e-5,initial_logodds_bound=2e-4,seed=0):
 """One source-reset 3-step trajectory, recording its 1/3 states (no sweep)."""
 assert set(caches)==set(pairs)=={'A','B'} and type(accepted) is bool
 assert interface in ('mlp','lora_qv') and lr==.0005
 assert 0<=initial_probability_bound<=2e-5 and 0<=initial_logodds_bound<=2e-4
 controller=MatchingMLP(model) if interface=='mlp' else QVLoRA(model,rank=4,alpha=4,seed=seed)
 parameters=controller.parameters();controller.reset()
 optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=0.)
 groups={view:checked_groups(caches[view],pairs[view],require_pair=accepted) for view in ('A','B')}
 trace=[];snapshots={};baseline={};initial_reports=[]
 def forward():return {view:replay_matching(model,caches[view],mode=interface) for view in ('A','B')}
 def save(step,outputs):
  scores={view:{'probability':o['probability'].detach().cpu().numpy(),
   'logodds':o['logodds'].detach().cpu().numpy(),
   'selected_original_query':int(o['probability'].detach().argmax()),
   'boxes_original':np.asarray(caches[view]['boxes']).copy()} for view,o in outputs.items()}
  module=controller.module if interface=='mlp' else controller.adapter
  state={n:p.detach().cpu().numpy().copy() for n,p in module.named_parameters()}
  snapshots[step]={'views':scores,'state':state,'update_report':controller.report()}
 try:
  with torch.no_grad():initial=forward()
  for view in ('A','B'):
   out=initial[view];cache=caches[view]
   pe=float(np.max(np.abs(out['probability'].cpu().numpy()-cache['scores'])))
   ze=float(np.max(np.abs(out['logodds'].cpu().numpy()-cache['logodds'])))
   assert pe<=initial_probability_bound and ze<=initial_logodds_bound
   assert int(out['probability'].argmax())==int(np.asarray(cache['scores']).argmax())
   baseline[view]=out['logodds'].detach().clone()
   p,n,outside,anchor=groups[view]
   initial_reports.append({'view':view,'probability_max_abs_error':pe,'logodds_max_abs_error':ze,
    'source_argmax_exact':True,'keep_uncovered_candidates':len(outside),'source_q0_anchor':anchor,
    'anchor_is_in_supervised_pair':bool(anchor in p or anchor in n)})
  save(0,initial)
  if accepted:
   for step in (1,2,3):
    optimizer.zero_grad(set_to_none=True);outputs=forward()
    loss,rank,keep,gap=objective(outputs,baseline,groups)
    assert torch.isfinite(loss);loss.backward()
    grad2=sum((p.grad.detach().square().sum() for p in parameters if p.grad is not None),loss.new_zeros(()))
    grad=float(grad2.sqrt());assert math.isfinite(grad)
    gradient_report=controller.report();optimizer.step()
    with torch.no_grad():post=forward();after,rank_after,keep_after,gap_after=objective(post,baseline,groups)
    trace.append({'step':step,'loss_before':float(loss.detach()),'ranknet_before':float(rank.detach()),
     'keep_before':float(keep.detach()),'gaps_before':gap.detach().cpu().tolist(),'gradient_L2':grad,
     'gradient_nonzero':grad>0,'parameter_gradients_before_step':gradient_report,
     'loss_after':float(after),'ranknet_after':float(rank_after),'keep_after':float(keep_after),
     'gaps_after':gap_after.cpu().tolist()})
    if step in (1,3):save(step,post)
  else:
   save(1,initial);save(3,initial)
  return {'interface':interface,'accepted_prefixed_preference':accepted,'steps_recorded':[0,1,3],
   'optimizer_steps':3 if accepted else 0,'snapshots':snapshots,'trace':trace,'initial_bridges':initial_reports,
   'ranknet_tau':.1,'keep_beta':1.,'keep_scope':'CURRENT_INPUT_A_B_UNCOVERED_PN_CANDIDATES_RELATIVE_TO_FIXED_SOURCE_Q0',
   'R30_U189_cross_observation_replay_not_used':True,'preferences_or_geometry_recomputed_after_update':False,
   'output_geometry':'ONLY_CACHED_ORIGINAL256_BOXES_PER_VIEW_ROOT_MAPS_TO_CANONICAL_A',
   'same_information_0_step_scores_included':True}
 finally:
  controller.reset()
  if interface=='lora_qv':controller.remove()
  else:model.requires_grad_(False)
