"""Fixed original clean inputs, no GT; exact baseline and bound parity."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_proximal_common_v1 import *
def run(ds):
 import torch,numpy as np
 from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
 from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
 from scripts.run_final_simplification_v1 import lease as gpu_lease
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.tastvg_proximal_method_v1 import OnlineMethod,rollout_states
 from vg_tta.tastvg_extended_method_v3 import OnlineMethod as Original
 from vg_tta.tastvg_native_spatial_rollout_s05_v1 import central_state
 p=verify(ds);cfg=p['params'];pool=POOL/ds;sys.addaudithook(guard);install_clean_loader()
 lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 from scripts.run_spatial_regression_alignment_v1 import model_load
 model=model_load('hcstvg1_test' if ds=='vidstg' else 'vidstg_test').eval().requires_grad_(False);mh=state_hash(model.state_dict())
 center=central_state(model);states,spec,_=rollout_states(center,cfg['rho'],4);deltas=[{n:v-center[n] for n,v in s.items()} for s in states]
 results=[];seq=p['splits']['search']['orders']['order1']
 try:
  for parent in [seq[0],seq[4]]:
   rr=read(pool/'capture/clean'/f'{parent:05}.json');data=device_tree(load(pool/rr['cache']),'cuda')
   def provider(stage):
    r=read(pool/'experts'/stage/'clean'/f'{parent:05}.json');f=pool/'experts'/r['cache'];assert sha(f)==r['cache_sha256']
    return {**load(f),'pixel_sha256':rr['pixel_sha256']}
   opts={k:cfg[k] for k in ['lr','teacher_temperature','student_temperature','steps']}
   a=Original(model,deltas,**opts);old,_=a.arrive(data,True,lambda:provider('temporal'),lambda:provider('spatial'));a.close()
   a=OnlineMethod(model,deltas,**opts,target_mode='rank');new,_=a.arrive(data,True,lambda:provider('temporal'),lambda:provider('spatial'));a.close()
   assert old['pre_state_sha256']==new['pre_state_sha256'] and old['post_state_sha256']==new['post_state_sha256']
   assert torch.equal(old['output_prediction']['boxes'],new['output_prediction']['boxes']) and old['output_prediction']['indices']==new['output_prediction']['indices']
   for x,y in zip(old['update_steps'],new['update_steps']):
    if x['update'] is not None:
     assert x['update']['loss_before']==y['update']['loss_before']
     for n,g in x['update']['gradients'].items():assert torch.equal(g,y['update']['gradients'][n])
   a=OnlineMethod(model,deltas,**opts,target_mode='proximal',s_ref=.02,arrival_radius=.1*spec['radius'])
   limited,_=a.arrive(data,True,lambda:provider('temporal'),lambda:provider('spatial'));a.close()
   for step in limited['update_steps']:
    if step['update'] is not None:assert step['update']['actual_arrival_norm']<=.1*spec['radius']+2e-6 and step['update']['arrival_center_sha256']==limited['pre_state_sha256']
   results.append(dict(parent=parent,steps=len(limited['update_steps']),updated=limited['updated'],baseline_bitwise=True,shared_arrival_bound=True))
  assert state_hash(model.state_dict())==mh
  write(BASE/ds/'SMOKE.json',dict(status='pass',results=results,GT_read=False,model_restored=True,time=time.time()))
 finally:lease.close()
if __name__=='__main__':run(sys.argv[1])
