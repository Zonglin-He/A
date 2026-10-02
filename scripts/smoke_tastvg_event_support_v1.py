"""First two fixed clean expert inputs; no labels; exact A and H algebra."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_event_support_common_v1 import *
def run(ds):
 import torch,numpy as np
 from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
 from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
 from scripts.run_final_simplification_v1 import lease as gpu_lease
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.tastvg_event_support_v1 import OnlineMethod,rollout_states
 from vg_tta.tastvg_selected_rollout_v1 import OnlineMethod as Original
 from vg_tta.tastvg_native_spatial_rollout_s05_v1 import central_state
 from scripts.tastvg_event_support_metrics_v1 import weights_from_candidates,weighted_rewards,geometry
 p=verify(ds);cfg=p['params'];pool=POOL/ds;sys.addaudithook(guard);install_clean_loader();bind_decode(ds)
 lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 from scripts.run_spatial_regression_alignment_v1 import model_load
 model=model_load('hcstvg1_test' if ds=='vidstg' else 'vidstg_test').eval().requires_grad_(False);mh=state_hash(model.state_dict())
 center=central_state(model);states,spec,basis=rollout_states(center,cfg['rho'],4)
 deltas=[{n:v-center[n] for n,v in s.items()} for s in states];results=[]
 try:
  seq=p['splits']['search']['orders']['order1']
  for parent in [seq[0],seq[4]]:
   cr=read(pool/'capture/clean'/f'{parent:05}.json');data=device_tree(load(pool/cr['cache']),'cuda')
   def provider(stage,calls):
    calls.append(stage);rec=read(pool/'experts'/stage/'clean'/f'{parent:05}.json');f=pool/'experts'/rec['cache'];assert sha(f)==rec['cache_sha256']
    return {**load(f),'pixel_sha256':cr['pixel_sha256']}
   opts={k:cfg[k] for k in ['lr','teacher_temperature','student_temperature','steps']}
   calls=[];actor=Original(model,deltas,**opts,target_mode='rank',actuation='rkl',basis=basis)
   old,_=actor.arrive(data,True,lambda:provider('temporal',calls),lambda:provider('spatial',calls));actor.close()
   for arm in ['A','H']:
    calls=[];actor=OnlineMethod(model,deltas,**opts,basis=basis,**read(BASE/ds/arm/'REQUEST.json')['method'])
    new,_=actor.arrive(data,True,lambda:provider('temporal',calls),lambda:provider('spatial',calls));actor.close()
    assert calls==['temporal','spatial']
    assert torch.equal(old['output_prediction']['boxes'],new['output_prediction']['boxes']) and old['output_prediction']['indices']==new['output_prediction']['indices']
    if arm=='A':
     assert old['pre_state_sha256']==new['pre_state_sha256'] and old['post_state_sha256']==new['post_state_sha256']
     for x,y in zip(old['update_steps'],new['update_steps']):
      if x['update'] is not None:
       assert x['update']['loss_before']==y['update']['loss_before']
       for n,g in x['update']['gradients'].items():assert torch.equal(g,y['update']['gradients'][n])
    else:
     es=provider('spatial',[]);last=None
     for step in new['update_steps']:
      e=step['event_support'];w=weights_from_candidates(e['candidates'],data['frame_ids'])
      np.testing.assert_allclose(w,e['weights'],atol=1e-15,rtol=0)
      cs=[c['prediction']['boxes'] for c in step['candidates']]
      rew=weighted_rewards(cs,es['boxes'],es['valid'],w);u=step['update']
      if rew is None:assert u is None and step['pre_state_sha256']==step['post_state_sha256']
      else:
       np.testing.assert_allclose(rew,step['rewards'],atol=1e-14,rtol=0)
       d=geometry(step['prediction']['boxes'],np.stack(cs),u['coefficients'],w)
       np.testing.assert_allclose(d,u['distances'],atol=3e-6,rtol=2e-5)
       for n,v in step['pre_state'].items():
        expected=v.clone().add_(u['gradients'][n],alpha=-cfg['lr']);assert torch.equal(expected,step['post_state'][n])
       assert u['rkl_backward_calls']==1 and u['srd_backward_calls']==0
      if last is not None:assert last==step['pre_state_sha256']
      last=step['post_state_sha256']
    results.append(dict(parent=parent,arm=arm,steps=len(new['update_steps']),current_output_bitwise=True,A_baseline_bitwise=arm=='A',expert_fetched_once=True,support_reward_geometry_SGD_checked=arm=='H',GT_read=False))
  assert state_hash(model.state_dict())==mh
  write(BASE/ds/'SMOKE.json',dict(status='pass',results=results,model_restored=True,GT_read=False,time=time.time()))
 finally:lease.close()
if __name__=='__main__':run(sys.argv[1])
