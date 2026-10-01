"""Full query streams; source capture once per observation, never per inner step."""
import sys,time,gc,traceback,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_best_full_common_v1 import *
def run(dataset):
 p=verify(dataset);out=BASE/dataset;cfg=p['params'];done=0;actor=None;lease=None;t=time.time()
 try:
  import torch,numpy as np
  from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
  from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
  from scripts.run_final_simplification_v1 import lease as gpu_lease
  from methods.decota_final_simplified_v1.tensors import state_hash
  from vg_tta.tastvg_best_full_method_v1 import OnlineMethod,rollout_states
  from vg_tta.tastvg_native_spatial_rollout_s05_v1 import central_state,reinsert
  from scripts.run_tastvg_full_b1_experts_v1 import observation
  from scripts.run_tastvg_paper48_p5_online_v1 import source_capture,compact_prediction
  assert read(out/'SMOKE.json')['status']=='pass'
  for stage in ['SPATIAL','TEMPORAL']:assert read(out/'experts'/f'{stage}_BARRIER.json')['cells']==read(out/'EXPERT_PLAN.json')['total']
  subjects=read(out/'SUBJECT_BARRIER.json');assert subjects['count']==p['queries']
  sys.addaudithook(guard);install_clean_loader();decode=bind_decode(dataset);lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  from scripts.run_spatial_regression_alignment_v1 import model_load
  model=model_load('hcstvg1_test' if dataset=='vidstg' else 'vidstg_test').eval().requires_grad_(False);mh=state_hash(model.state_dict());center=central_state(model);states,spec,basis=rollout_states(center,cfg['rho'],cfg['direction_count']);deltas=[{n:v-center[n] for n,v in s.items()} for s in states]
  if not (out/'SUPPORT.json').exists():write(out/'SUPPORT.json',dict(spec=spec,center_sha256=state_hash(center),checkpoint_state_sha256=mh,params=cfg))
  else:assert read(out/'SUPPORT.json')['checkpoint_state_sha256']==mh
  scratch=out/'scratch_H';scratch.mkdir(exist_ok=True);cache={f.stem:f for f in scratch.glob('*.pt')};sizes={k:f.stat().st_size for k,f in cache.items()};size=sum(sizes.values());captures=0
  for cond in p['conditions']:
   for order,seq in p['orders'].items():
    actor=OnlineMethod(model,deltas,**{k:cfg[k] for k in ['lr','teacher_temperature','student_temperature','steps']});previous=state_hash(actor.actor.initial);last=None
    for at,parent in enumerate(seq):
     f=out/'online'/cond/order/f'{at:05}.pt.gz';rf=f.with_suffix('.json')
     if rf.exists():
      r=read(rf);assert sha(f)==r['sha256'] and r['pre_sha']==previous and r['parent']==parent;previous=r['post_sha'];last=f;done+=1;continue
     if last is not None:actor.actor.restore(device_tree(loadz(last)['post_state'],'cuda'));last=None
     budget();row=p['rows'][parent];frames,ids=decode(row['input']);assert ids==row['frame_ids'];shifted,pixel,generation=observation(row,cond,frames)
     sf=out/'subjects'/f'{parent:05}.json';assert sha(sf)==subjects['files'][sf.name];subject=read(sf)['parses']['subject'];learned=actor.actor.state();assert state_hash(learned)==previous
     key=hashlib.sha256((row['key']+'|'+pixel+'|'+mh).encode()).hexdigest();cf=scratch/f'{key}.pt'
     if cf.exists():data=load(cf);cf.touch()
     else:
      actor.actor.restore(actor.actor.initial);data=source_capture(model,shifted,row,subject);data.update(pixel_sha256=pixel,checkpoint_state_sha256=mh);actor.actor.restore(learned);save(cf,data);captures+=1;cache[key]=cf;sizes[key]=cf.stat().st_size;size+=sizes[key]
      while size>2**30 and len(cache)>1:
       old=min((k for k in cache if k!=key),key=lambda k:cache[k].stat().st_mtime);cache[old].unlink();size-=sizes.pop(old);cache.pop(old)
     assert data['pixel_sha256']==pixel and data['checkpoint_state_sha256']==mh
     data=device_tree(data,'cuda');calls=[];scheduled=at%4==0
     def get(stage):
      assert scheduled;r=read(out/'experts'/stage/cond/f'{parent:05}.json');assert r['pixel_sha256']==pixel;cf=out/'experts'/r['cache'];assert sha(cf)==r['cache_sha256'];calls.append(stage);return {**load(cf),'pixel_sha256':pixel}
     x,ev=actor.arrive(data,scheduled,lambda:get('temporal'),lambda:get('spatial'));assert x['pre_state_sha256']==previous and calls==(['temporal','spatial'] if scheduled else [])
     assert torch.isfinite(x['output_prediction']['boxes']).all() and all(torch.isfinite(v).all() for v in x['post_state'].values())
     rein=None
     if at in [0,len(seq)-1]:rein=reinsert(model,shifted,{**row,'parses':dict(subject=subject)},actor.actor.state(),ev)
     payload=dict(parent=parent,source=row['source'],condition=cond,order=order,arrival=at,expert_scheduled=scheduled,updated=x['updated'],pixel_sha256=pixel,generation=generation,pre_state=x['pre_state'],post_state=x['post_state'],pre_sha=x['pre_state_sha256'],post_sha=x['post_state_sha256'],update_steps=x['update_steps'],compute=x['compute'],source_native=compact_prediction(data['prediction']),slow=compact_prediction(x['prediction']),post_prediction=compact_prediction(x['post_prediction']),final_indices=x['output_prediction']['indices'],temporal=x.get('temporal'),reinsertion=rein,GT_read=False)
     savez(f,payload);write(rf,dict(sha256=sha(f),parent=parent,pre_sha=payload['pre_sha'],post_sha=payload['post_sha'],updated=x['updated']));previous=x['post_state_sha256'];done+=1
     if done%12==0 or at==0:status(out/'ONLINE_STATUS.json',dict(status='running',done=done,total=p['total'],condition=cond,order=order,arrival=at,captures_this_run=captures,seconds=time.time()-t));print('ONLINE',dataset,done,p['total'],cond,order,at,flush=True)
     del frames,shifted,data,x,ev,payload;gc.collect();torch.cuda.empty_cache()
    actor.close();actor=None;assert state_hash(model.state_dict())==mh
  assert done==p['total'];files={str(f.relative_to(out)):sha(f) for f in (out/'online').rglob('*.json')};assert len(files)==done
  write(out/'PREDICTION_BARRIER.json',dict(cells=done,GT_read=False,model_restored=True,files=files,time=time.time()));status(out/'ONLINE_STATUS.json',dict(status='completed',done=done,total=done,seconds=time.time()-t,captures_this_run=captures))
 except BaseException as e:status(out/'ONLINE_STATUS.json',dict(status='failed',done=done,error=repr(e),traceback=traceback.format_exc(),seconds=time.time()-t));raise
 finally:
  if actor:actor.close()
  if lease:lease.close()
if __name__=='__main__':run(sys.argv[1])
