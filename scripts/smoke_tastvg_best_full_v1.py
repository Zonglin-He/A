"""Bitwise parity of diagnostic logging under actual selected parameters; no GT."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_best_full_common_v1 import *
def run(dataset):
 import torch,numpy as np
 from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
 from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
 from scripts.run_final_simplification_v1 import lease
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.tastvg_native_spatial_rollout_s05_v1 import central_state
 from vg_tta.tastvg_extended_method_v3 import OnlineMethod as Original
 from vg_tta.tastvg_best_full_method_v1 import OnlineMethod,rollout_states
 p=verify(dataset);cfg=p['params'];old=ROOT/'artifacts/tastvg_extended_sensitivity_v3'/dataset;op=read(old/'PLAN.json');cap=read(old/'CAPTURE_BARRIER.json');out=BASE/dataset
 sys.addaudithook(guard);install_clean_loader();bind_decode(dataset);budget()
 from scripts.run_spatial_regression_alignment_v1 import model_load
 torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 handle=lease();actor=None
 try:
  model=model_load('hcstvg1_test' if dataset=='vidstg' else 'vidstg_test').eval().requires_grad_(False);mh=state_hash(model.state_dict());assert mh==cap['checkpoint_state_sha256'];center=central_state(model);states,_,_=rollout_states(center,cfg['rho'],cfg['direction_count']);delta=[{n:v-center[n] for n,v in s.items()} for s in states]
  res=[]
  for parent in op['splits']['search']['orders']['order1'][::4][:2]:
   rf=old/'capture/clean'/f'{parent:05}.json';assert sha(rf)==cap['files'][str(rf.relative_to(old))];r=read(rf);assert sha(old/r['cache'])==r['sha256'];data=device_tree(load(old/r['cache']),'cuda')
   def providers(calls):
    def get(stage):
     er=read(old/'experts'/stage/'clean'/f'{parent:05}.json');f=old/'experts'/er['cache'];assert sha(f)==er['cache_sha256'];calls.append(stage);return {**load(f),'pixel_sha256':r['pixel_sha256']}
    return lambda:get('temporal'),lambda:get('spatial')
   kwargs={k:cfg[k] for k in ['lr','teacher_temperature','student_temperature','steps']}
   actor=Original(model,delta,**kwargs);ca=[];a,_=actor.arrive(data,True,*providers(ca));actor.close()
   actor=OnlineMethod(model,delta,**kwargs);cb=[];b,_=actor.arrive(data,True,*providers(cb));actor.close();actor=None
   assert ca==cb==['temporal','spatial'] and a['compute']==b['compute'];assert a['post_state_sha256']==b['post_state_sha256']
   for field in ['prediction','output_prediction','post_prediction']:
    assert a[field]['indices']==b[field]['indices'] and torch.equal(a[field]['boxes'],b[field]['boxes'])
   assert len(a['update_steps'])==len(b['update_steps'])
   for x,y in zip(a['update_steps'],b['update_steps']):
    assert x['pre_state_sha256']==y['pre_state_sha256'] and x['post_state_sha256']==y['post_state_sha256'];assert len(y['candidates'])==9
    if x['update']:
     for k in ['loss_before','loss_after']:assert x['update'][k]==y['update'][k]
     for n,g in x['update']['gradients'].items():assert torch.equal(g,y['update']['gradients'][n])
   # Actual lossless serialization, including newly logged candidates.
   path=out/'smoke'/f'{parent:05}.pt.gz';savez(path,b);c=loadz(path);assert c['post_state_sha256']==b['post_state_sha256'];assert torch.equal(c['update_steps'][0]['candidates'][0]['prediction']['boxes'],b['prediction']['boxes'])
   res.append(dict(parent=parent,steps=len(b['update_steps']),updated=b['updated'],bitwise_parity=True,compute=b['compute'],payload_sha256=sha(path),payload_bytes=path.stat().st_size))
  assert state_hash(model.state_dict())==mh
  write(out/'SMOKE.json',dict(status='pass',inputs=res,actual_selected_params=cfg,GT_read=False,model_restored=True,no_additional_forward_or_provider_calls=True,time=time.time()))
 finally:
  if actor:actor.close()
  handle.close()
if __name__=='__main__':run(sys.argv[1])
