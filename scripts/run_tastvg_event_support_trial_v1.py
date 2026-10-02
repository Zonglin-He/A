import sys,time,gc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_event_support_common_v1 import *
def run(dataset,request):
 p=verify(dataset);req=read(Path(request));out=Path(request).parent;pool=POOL/dataset;cfg=req['params'];split=req['split'];sp=p['splits'][split];done=0;lease=actor=None;t=time.time()
 try:
  import torch,numpy as np
  from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
  from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
  from scripts.run_final_simplification_v1 import lease as gpu_lease
  from methods.decota_final_simplified_v1.tensors import state_hash,detached
  from vg_tta.tastvg_event_support_v1 import OnlineMethod,rollout_states
  from vg_tta.tastvg_native_spatial_rollout_s05_v1 import central_state,reinsert
  from scripts.run_tastvg_full_b1_experts_v1 import observation
  from scripts.run_tastvg_paper48_p5_online_v1 import compact_prediction
  sys.addaudithook(guard);install_clean_loader();decode=bind_decode(dataset);lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  from scripts.run_spatial_regression_alignment_v1 import model_load
  model=model_load('hcstvg1_test' if dataset=='vidstg' else 'vidstg_test').eval().requires_grad_(False);mh=state_hash(model.state_dict());capbar=read(pool/'CAPTURE_BARRIER.json');assert capbar['checkpoint_state_sha256']==mh
  center=central_state(model);states,spec,basis=rollout_states(center,cfg['rho'],cfg['direction_count']);deltas=[{n:v-center[n] for n,v in state.items()} for state in states]
  if not (out/'SUPPORT.json').exists():write(out/'SUPPORT.json',dict(spec=spec,center_sha256=state_hash(center),params=cfg))
  def input_data(parent,cond):
   rf=pool/'capture'/cond/f'{parent:05}.json';assert sha(rf)==capbar['files'][str(rf.relative_to(pool))];r=read(rf);f=pool/r['cache'];assert sha(f)==r['sha256'];return device_tree(load(f),'cuda'),r
  def providers(parent,cond,pixel,calls):
   def get(stage):
    r=read(pool/'experts'/stage/cond/f'{parent:05}.json');assert r['pixel_sha256']==pixel;f=pool/'experts'/r['cache'];assert sha(f)==r['cache_sha256'];calls.append(stage);return {**load(f),'pixel_sha256':pixel}
   return lambda:get('temporal'),lambda:get('spatial')
  # Arm A must reproduce the full sealed selected development state stream.
  for cond in p['conditions']:
   for order,seq in sp['orders'].items():
    actor=OnlineMethod(model,deltas,lr=cfg['lr'],teacher_temperature=cfg['teacher_temperature'],student_temperature=cfg['student_temperature'],steps=cfg['steps'],basis=basis,**req['method']);previous=state_hash(actor.actor.initial);last=None
    for at,parent in enumerate(seq):
     budget();f=out/'online'/cond/order/f'{at:05}.pt';rf=f.with_suffix('.json')
     if rf.exists():
      r=read(rf);assert sha(f)==r['sha256'] and r['pre_sha']==previous and r['parent']==parent;previous=r['post_sha'];last=f;done+=1;continue
     if last is not None:actor.actor.restore(device_tree(load(last)['post_state'],'cuda'));last=None
     data,cr=input_data(parent,cond);assert state_hash(actor.actor.state())==previous;calls=[];scheduled=at%4==0;tp,ss=providers(parent,cond,cr['pixel_sha256'],calls);x,ev=actor.arrive(data,scheduled,tp,ss)
     assert calls==(['temporal','spatial'] if scheduled else []) and x['pre_state_sha256']==previous
     if not torch.isfinite(x['output_prediction']['boxes']).all() or any(not torch.isfinite(v).all() for v in x['post_state'].values()):raise FloatingPointError('nonfinite output or updated state')
     rein=None
     payload=dict(arm=req['arm'],parent=parent,condition=cond,order=order,arrival=at,expert_scheduled=scheduled,updated=x['updated'],pixel_sha256=cr['pixel_sha256'],pre_state=x['pre_state'],post_state=x['post_state'],pre_sha=x['pre_state_sha256'],post_sha=x['post_state_sha256'],update=x.get('update'),update_steps=x['update_steps'],compute=x['compute'],source_native=compact_prediction(data['prediction']),slow=compact_prediction(x['prediction']),post_prediction=compact_prediction(x['post_prediction']),final_indices=x['output_prediction']['indices'],temporal=x.get('temporal'),reinsertion=rein,GT_read=False)
     if req['arm']=='A':
      old=load(ROOT/p['baseline_result_dir']/'online'/cond/order/f'{at:05}.pt')
      assert old['pre_sha']==payload['pre_sha'] and old['post_sha']==payload['post_sha']
      assert torch.equal(old['slow']['boxes'],payload['slow']['boxes']) and old['final_indices']==payload['final_indices']
      for aa,bb in zip(old['update_steps'],payload['update_steps']):
       if aa['update'] is not None:
        assert aa['update']['loss_before']==bb['update']['loss_before']
        for nn,gg in aa['update']['gradients'].items():assert torch.equal(gg,bb['update']['gradients'][nn])
     save(f,payload);write(rf,dict(sha256=sha(f),parent=parent,pre_sha=payload['pre_sha'],post_sha=payload['post_sha'],updated=x['updated']));previous=payload['post_sha'];done+=1
     if done%12==0:status(out/'STATUS.json',dict(status='running',done=done,total=sp['total'],condition=cond,order=order,seconds=time.time()-t));print('TRIAL',dataset,req['tag'],done,sp['total'],flush=True)
     del data,x,ev,payload;gc.collect();torch.cuda.empty_cache()
    actor.close();actor=None;assert state_hash(model.state_dict())==mh
  assert done==sp['total'];write(out/'PREDICTION_BARRIER.json',dict(cells=done,GT_read=False,model_restored=True,files={str(f.relative_to(out)):sha(f) for f in (out/'online').rglob('*.json')},time=time.time()));status(out/'STATUS.json',dict(status='completed',done=done,total=done,seconds=time.time()-t))
 except BaseException as e:status(out/'STATUS.json',dict(status='numerical_failure' if isinstance(e,FloatingPointError) else 'failed',done=done,error=repr(e),traceback=traceback.format_exc(),seconds=time.time()-t));raise
 finally:
  if actor:actor.close()
  if lease:lease.close()
if __name__=='__main__':
 try:run(*sys.argv[1:])
 except FloatingPointError:sys.exit(42)
