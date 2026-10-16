"""Synthetic CPU-only qualification for future, not-yet-run component ablations."""
from pathlib import Path
import sys,copy,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def contracts():
 import torch
 from scripts import stvg_component_spatial_fit_v1 as new
 from scripts import stvg_soft_ep_coordinate_fit002 as old
 from scripts import stvg_component_temporal_ablation_v1 as temporal
 from scripts import stvg_target_teacher_student_v1 as core
 torch.set_num_threads(2)
 class Replay:
  def __init__(self):
   g=torch.Generator().manual_seed(91)
   self.named=[('spatial.query_residual',torch.nn.Parameter(torch.zeros(256))),('spatial.ln.weight',torch.nn.Parameter(torch.zeros(768))),('spatial.ln.bias',torch.nn.Parameter(torch.zeros(768)))]
   self.matrix=torch.randn(1792,32,generator=g)*.04
  def state(self):return {n:p.detach().clone() for n,p in self.named}
  def restore(self,s):
   with torch.no_grad():
    for n,p in self.named:p.copy_(s[n]);p.grad=None
  def values(self):
   x=torch.cat([p.reshape(-1) for _,p in self.named]);return {'boxes':torch.sigmoid((x@self.matrix).reshape(8,4))*.5+.15}
 class Base:
  def __init__(self):self.zero={'boxes':torch.zeros(8,4)};self.restored=False
  def restore(self,s):self.restored=True
 original=core._trainable_replay
 def fake(base,ids,key,scope):
  rp=Replay();return rp,rp.named
 core._trainable_replay=fake
 cfg=core.load_original_opd_config('hc2');cfg=dict(cfg,steps=2,lr=.005)
 settings={'mu':.125,'cg_max_iter':12,'cg_rtol':1e-5,'response_ratio':1.,'response_abs_tolerance':1e-7,'backtracking':[1.,.5,.25,.125]}
 anchors=[{'position':i,'box':torch.tensor([.35,.35,.4,.4]),'weight':w}for i,w in [(1,.7),(6,.9)]]
 checks=0;results={}
 try:
  b=Base();orig=old.fit(b,{},anchors,list(range(8)),'cpu_fixture',cfg,'M2',settings)
  baseline=new.fit(b,{},anchors,list(range(8)),'cpu_fixture',cfg,'M2',settings)
  def science(x):
   if isinstance(x,dict):return {k:science(v)for k,v in x.items()if k not in {'elapsed_seconds','wall_seconds'}}
   if isinstance(x,list):return [science(v)for v in x]
   return x
  for k in ['final_state','final_cached_decoder_boxes','path']:
   assert core._tree_equal(science(orig[k]),science(baseline[k])),k
   checks+=1
  from scripts.review_stvg_component_trial_v1 import fit_audit
  reference={'anchors':[{'position':a['position'],'box_cxcywh':a['box'].tolist(),'weight':a['weight']}for a in anchors]}
  fit_audit(baseline,reference,{})
  for flag in sorted(new.ALLOWED_FLAGS):
   b=Base();r=new.fit(b,{},anchors,list(range(8)),'cpu_fixture',cfg,'M2',settings,flags={flag:True})
   assert b.restored and torch.isfinite(r['final_cached_decoder_boxes']).all();checks+=1
   n=256 if flag=='query_only' else 1536 if flag=='ln_only' else 1792
   assert r['active_parameters']==n;checks+=1
   for name,value in r['final_state'].items():
    if name not in r['active_names']:assert torch.equal(value,r['initial_state'][name]);checks+=1
   fit_audit(r,reference,{flag:True});checks+=1
   results[flag]={'active_parameters':n,'accepted_count':r['optimizer_steps']}
  from scripts.stvg_component_ablation_common_v1 import SPATIAL
  for tid,name,flags in SPATIAL:
   if len(flags)>1:
    r=new.fit(Base(),{},anchors,list(range(8)),'cpu_fixture',cfg,'M2',settings,flags=flags);fit_audit(r,reference,flags);checks+=1
  for mutation in ('loss','gradient','filtered_delta'):
   corrupt=copy.deepcopy(baseline);row=next(x for x in corrupt['path']if not x['skipped'])
   if mutation=='loss':row['loss']+=1.
   else:row[mutation][0]+=1.
   try:fit_audit(corrupt,reference,{})
   except AssertionError:checks+=1
   else:raise AssertionError('reader accepted corrupted '+mutation)
  # Uniform proposal baseline has exactly zero sampled score-function mean for antithetic pairs.
  mean=torch.tensor([[.3,.4,.2,.2]],requires_grad=True)
  p=new.rollout_ablation(mean,torch.tensor([[.35,.4,.25,.2]]),'contract',0,cfg,{})
  g0=torch.autograd.grad(new.loss_ablation(mean,p,cfg['sigma'],{}).sum(),mean,retain_graph=True)[0]
  g1=torch.autograd.grad(new.loss_ablation(mean,p,cfg['sigma'],{'no_policy_KL':True}).sum(),mean,retain_graph=True)[0]
  assert torch.equal(g0,g1);checks+=1
  a=new.rollout_ablation(mean,torch.tensor([[.35,.4,.25,.2]]),'contract',0,cfg,{'iid_sampling':True});bb=new.rollout_ablation(mean,torch.tensor([[.35,.4,.25,.2]]),'contract',0,cfg,{'iid_sampling':True})
  assert torch.equal(a['samples'],bb['samples']) and list(a['samples'].shape)==[1,32,4];checks+=1
  try:new.fit(Base(),{},anchors,list(range(8)),'cpu_fixture',cfg,'M2',settings,flags={'invalid':True})
  except AssertionError:checks+=1
  else:raise AssertionError('unknown switch accepted')
 finally:core._trainable_replay=original
 tr=temporal.contracts()
 return dict(status='pass',scope='synthetic fixture only; no real model/GT/GPU or method-effect qualification',CPU_only=True,GT_read=False,GPU_used=False,model_loaded=False,checks=checks,temporal=tr,variants=results,unmodified_M2_exact_fixture_match=True,policy_KL_current_mean_gradient_exact_zero=True)
if __name__=='__main__':print(json.dumps(contracts(),indent=2,sort_keys=True))
