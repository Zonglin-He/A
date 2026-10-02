"""Fixed first two scheduled clean inputs, no GT; original parity and direction."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_directional_common_v1 import *
def run(ds):
 import torch,numpy as np
 from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
 from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
 from scripts.run_final_simplification_v1 import lease as gpu_lease
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.tastvg_directional_preference_v1 import OnlineMethod,rollout_states
 from vg_tta.tastvg_proximal_method_v1 import OnlineMethod as Original
 from vg_tta.tastvg_native_spatial_rollout_s05_v1 import central_state
 p=verify(ds);cfg=p['params'];pool=POOL/ds;sys.addaudithook(guard);install_clean_loader()
 lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 from scripts.run_spatial_regression_alignment_v1 import model_load
 model=model_load('hcstvg1_test' if ds=='vidstg' else 'vidstg_test').eval().requires_grad_(False);mh=state_hash(model.state_dict())
 center=central_state(model);states,spec,basis=rollout_states(center,cfg['rho'],4);deltas=[{n:v-center[n] for n,v in s.items()} for s in states]
 results=[];seq=p['splits']['search']['orders']['order1']
 try:
  for parent in [seq[0],seq[4]]:
   rr=read(pool/'capture/clean'/f'{parent:05}.json');data=device_tree(load(pool/rr['cache']),'cuda')
   def provider(stage,calls):
    calls.append(stage);r=read(pool/'experts'/stage/'clean'/f'{parent:05}.json');f=pool/'experts'/r['cache'];assert sha(f)==r['cache_sha256']
    return {**load(f),'pixel_sha256':rr['pixel_sha256']}
   opts={k:cfg[k] for k in ['lr','teacher_temperature','student_temperature','steps']}
   calls=[];a=Original(model,deltas,**opts,target_mode='rank');old,_=a.arrive(data,True,lambda:provider('temporal',calls),lambda:provider('spatial',calls));a.close()
   for arm in ['A','E','F']:
    calls=[];a=OnlineMethod(model,deltas,**opts,basis=basis,**read(BASE/ds/arm/'REQUEST.json')['method'])
    new,_=a.arrive(data,True,lambda:provider('temporal',calls),lambda:provider('spatial',calls));a.close()
    assert calls==['temporal','spatial']
    assert torch.equal(old['output_prediction']['boxes'],new['output_prediction']['boxes']) and old['output_prediction']['indices']==new['output_prediction']['indices']
    if arm=='A':
     assert old['pre_state_sha256']==new['pre_state_sha256'] and old['post_state_sha256']==new['post_state_sha256']
     for x,y in zip(old['update_steps'],new['update_steps']):
      if x['update'] is not None:
       assert x['update']['loss_before']==y['update']['loss_before']
       for n,g in x['update']['gradients'].items():assert torch.equal(g,y['update']['gradients'][n])
    usable=0;zero=0
    for st in new['update_steps']:
     u=st['update']
     if u is None:continue
     ranks=np.asarray(u['rank']);c=ranks[2::2]-ranks[1::2];np.testing.assert_array_equal(c,u['rank_contrasts'])
     assert u['top_pair']==int(np.argmax(np.abs(c)))
     target=cfg['lr']*u['global_gradient_norm'];assert abs(target-u['counterfactual_rkl_step_norm'])<1e-12
     if arm!='A' and u['no_op_reason']=='no_rank_direction':
      assert st['pre_state_sha256']==st['post_state_sha256'];zero+=1
     else:
      assert abs(u['actual_step_norm']-target)<2e-6+1e-5*target;usable+=1
    results.append(dict(parent=parent,arm=arm,steps=len(new['update_steps']),magnitude_matched_steps=usable,zero_direction_steps=zero,current_output_bitwise=True,baseline_bitwise=arm=='A',expert_fetched_once=True))
  assert state_hash(model.state_dict())==mh
  write(BASE/ds/'SMOKE.json',dict(status='pass',results=results,GT_read=False,model_restored=True,time=time.time()))
 finally:lease.close()
if __name__=='__main__':run(sys.argv[1])
