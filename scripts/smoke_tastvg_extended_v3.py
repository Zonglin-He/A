"""No-GT live default parity and multi-step qualification on fixed cached inputs."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_extended_common_v3 import *

def run(dataset):
 import torch,numpy as np
 from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
 from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
 from scripts.run_final_simplification_v1 import lease
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.tastvg_native_spatial_rollout_s05_v1 import central_state
 from vg_tta.tastvg_optuna_method_v1 import OnlineMethod as Original
 from vg_tta.tastvg_extended_method_v3 import OnlineMethod,rollout_states
 p=verify(dataset);out=BASE/dataset;cfg=anchor(p);cap=read(out/'CAPTURE_BARRIER.json')
 sys.addaudithook(guard);install_clean_loader();bind_decode(dataset);budget()
 from scripts.run_spatial_regression_alignment_v1 import model_load
 torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 handle=lease();model=None;actor=None
 try:
  model=model_load('hcstvg1_test' if dataset=='vidstg' else 'vidstg_test').eval().requires_grad_(False);mh=state_hash(model.state_dict());assert cap['checkpoint_state_sha256']==mh
  center=central_state(model)
  def deltas(count):
   states,_,_=rollout_states(center,cfg['rho'],count)
   return [{n:v-center[n] for n,v in s.items()} for s in states]
  parents=p['splits']['search']['orders']['order1'][::4][:2];results=[];multi_valid=False
  for parent in parents:
   rf=out/'capture/clean'/f'{parent:05}.json';assert sha(rf)==cap['files'][str(rf.relative_to(out))];r=read(rf);assert sha(out/r['cache'])==r['sha256'];data=device_tree(load(out/r['cache']),'cuda')
   def providers(calls):
    def get(stage):
     er=read(out/'experts'/stage/'clean'/f'{parent:05}.json');assert er['pixel_sha256']==r['pixel_sha256'];f=out/'experts'/er['cache'];assert sha(f)==er['cache_sha256'];calls.append(stage);return {**load(f),'pixel_sha256':r['pixel_sha256']}
    return lambda:get('temporal'),lambda:get('spatial')
   actor=Original(model,deltas(4),lr=cfg['lr'],teacher_temperature=cfg['teacher_temperature']);a,_=actor.arrive(data,True,*providers([]));actor.close()
   actor=OnlineMethod(model,deltas(4),lr=cfg['lr'],teacher_temperature=cfg['teacher_temperature'],student_temperature=1.,steps=1);calls=[];b,_=actor.arrive(data,True,*providers(calls));actor.close()
   assert calls==['temporal','spatial'] and a['post_state_sha256']==b['post_state_sha256']
   assert a['output_prediction']['indices']==b['output_prediction']['indices'] and torch.equal(a['output_prediction']['boxes'],b['output_prediction']['boxes'])
   if a['updated']:
    for key in ['loss_before','loss_after','update_scale']:assert a['update'][key]==b['update'][key]
    for n,v in a['update']['gradients'].items():assert torch.equal(v,b['update']['gradients'][n])
   actor=OnlineMethod(model,deltas(8),lr=cfg['lr'],teacher_temperature=cfg['teacher_temperature'],student_temperature=.3,steps=3);calls=[];c,_=actor.arrive(data,True,*providers(calls));actor.close();actor=None
   assert calls==['temporal','spatial']
   assert c['output_prediction']['indices']==a['output_prediction']['indices'] and torch.equal(c['output_prediction']['boxes'],a['output_prediction']['boxes'])
   trace=c['update_steps'];assert all(trace[i]['pre_state_sha256']==trace[i-1]['post_state_sha256'] for i in range(1,len(trace)))
   assert all(torch.isfinite(v).all() for v in c['post_state'].values())
   if len(trace)==3 and c['updated']:multi_valid=True
   results.append(dict(parent=parent,anchor_parity=True,multi_steps=len(trace),updated=c['updated'],providers=calls,compute=c['compute']))
  assert multi_valid,'Fixed two-input smoke did not exercise valid multi-step evidence'
  assert state_hash(model.state_dict())==mh
  write(out/'DEFAULT_PARITY.json',dict(status='pass',native_boxes_bitwise=True,temporal_indices=True,post_state_bitwise=True,gradients_bitwise=True,GT_read=False,time=time.time()))
  write(out/'SMOKE.json',dict(status='pass',GT_read=False,default_bitwise_parity=True,multi_step_qualified=True,inputs=results,model_restored=True,time=time.time()))
 finally:
  if actor:actor.close()
  handle.close()

if __name__=='__main__':run(sys.argv[1])
