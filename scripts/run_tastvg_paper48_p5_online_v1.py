"""HC2 binding of the frozen Paper48 stream, with HC2-relative probe support."""
import sys,time,gc,traceback,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
from scripts.tastvg_paper48_common_v1 import guard,budget
from scripts.tastvg_paper48_p5_common_v1 import BASE,verify as verify_panel
from scripts.run_tastvg_full_b1_experts_v1 import observation
OUT=BASE
def verify():return verify_panel()

def source_capture(model,frames,row,subject):
 import torch
 from methods.decota_final_simplified_v1.backbone import make_batch,query_subject,capture
 from methods.decota_final_simplified_v1.tensors import floating32,detached
 from vg_tta.tastvg_evidence_capture_v1 import combined
 batch=make_batch(frames,row['frame_ids'],row['input'],model)
 with torch.no_grad(),query_subject(model,batch,subject):
  views,records=capture(model,batch);views=floating32(views);fields=[model.ground_encoder.encoder.norm(v['prefix']) for v in views]
  ev,boxes,pred=combined(model,views,fields,records,row['frame_ids'])
  compact=[dict(info={k:v for k,v in view['info'].items() if k not in ['encoded_feature','frames_cls','videos_cls']},vis_pos=view['vis_pos'],H=H) for view,H in zip(views,fields)]
 return detached(dict(views=compact,records=records,prediction=pred,frame_ids=row['frame_ids']),'cpu')

def compact_prediction(p):return dict(boxes=p['boxes'].cpu(),indices=p['indices'],physical_interval=p['physical_interval'])

def run(limit=0):
 p=verify();tick=time.monotonic();done=created=0;failure=None;state='failed';lease=None;policy=None
 try:
  import torch,numpy as np
  from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
  from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
  from scripts.run_final_simplification_v1 import lease as gpu_lease
  from methods.decota_final_simplified_v1.tensors import state_hash,detached
  from methods.tastvg_dual_evidence_j0_v1.method import OnlineMethod,central_with_candidates,fast_rerank
  from vg_tta.exact_frame_decode_audit_v2 import decode
  from vg_tta.tastvg_native_spatial_rollout_s05_v1 import reinsert
  from scripts.run_tastvg_schedule_j01_v1 import full_temporal_check
  sys.addaudithook(guard);install_clean_loader();lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  for stage in ['SPATIAL','TEMPORAL']:assert read(BASE/'experts'/f'{stage}_BARRIER.json')['cells']==read(BASE/'EXPERT_PLAN.json')['total']
  subjects=read(OUT/'SUBJECT_BARRIER.json');assert subjects['count']==len(p['rows'])
  from scripts.run_spatial_regression_alignment_v1 import model_load
  model=model_load('vidstg_test').eval().requires_grad_(False);mh=state_hash(model.state_dict());assert model.cfg.DATASET.NAME=='HC-STVG'
  from vg_tta.tastvg_native_spatial_rollout_s05_v1 import central_state,rollout_states
  center=central_state(model);states,spec,basis=rollout_states(center)
  sf=BASE/'PARAMETER_SUPPORT.pt'
  if not sf.exists():
   save(sf,detached(dict(center=center,states=states,spec=spec,basis=basis),'cpu'));write(sf.with_suffix('.json'),dict(sha256=sha(sf),model_state_sha256=mh,spec=spec))
  sr=read(sf.with_suffix('.json'));assert sr['sha256']==sha(sf) and sr['model_state_sha256']==mh
  assert state_hash(load(sf)['center'])==state_hash(center)
  support=load(BASE/'PARAMETER_SUPPORT.pt');deltas=[{n:(v-support['center'][n]).cuda() for n,v in x.items()} for x in support['states']]
  # Scratch H tensors are reproducible accelerator inputs, never sealed predictions.
  scratch=BASE/'scratch_H';scratch.mkdir(exist_ok=True);cache={f.stem:f for f in scratch.glob('*.pt')};sizes={k:f.stat().st_size for k,f in cache.items()};totalbytes=sum(sizes.values())
  for cond in p['conditions']:
   for order,seq in p['orders'].items():
    policy=OnlineMethod(model,deltas);previous=state_hash(policy.actor.initial);seen_sources=set();last_existing=None;stream_updates=0
    for arrival,parent in enumerate(seq):
     row=p['rows'][parent];name=f'{arrival:05}.pt';f=OUT/'online'/cond/order/name;receipt=f.with_suffix('.json');firstsource=row['source'] not in seen_sources;seen_sources.add(row['source'])
     if receipt.exists():
      z=read(receipt);assert sha(f)==z['sha256'] and z['parent']==parent and z['pre_sha']==previous
      previous=z['post_sha'];last_existing=f;done+=1;stream_updates+=z['updated'];continue
     if limit and created>=limit:break
     if last_existing is not None:policy.actor.restore(device_tree(load(last_existing)['post_state'],'cuda'));last_existing=None
     assert state_hash(policy.actor.state())==previous;budget(tick);learned=policy.actor.state();frames,ids=decode(row['input']);assert ids==row['frame_ids'];shifted,pixel,spec=observation(row,cond,frames)
     sp=OUT/'subjects'/f'{parent:05}.json';assert sha(sp)==subjects['files'][sp.name];subject=read(sp)['parses']['subject'];row={**row,'parses':dict(subject=subject)}
     key=hashlib.sha256((row['key']+'|'+pixel).encode()).hexdigest();cp=scratch/f'{key}.pt'
     if cp.exists():
      data=load(cp);cp.touch()
     else:
      policy.actor.restore(policy.actor.initial);data=source_capture(model,shifted,row,subject);data['pixel_sha256']=pixel;save(cp,data);cache[key]=cp;sizes[key]=cp.stat().st_size;totalbytes+=sizes[key];policy.actor.restore(learned)
      while totalbytes>8*2**30 and len(cache)>1:
       old=min((k for k in cache if k!=key),key=lambda k:cache[k].stat().st_mtime);cache[old].unlink();totalbytes-=sizes.pop(old);cache.pop(old)
     assert data['pixel_sha256']==pixel and state_hash(policy.actor.state())==previous;data=device_tree(data,'cuda');scheduled=p['availability']==100 or (p['availability']==25 and arrival%4==0);calls=[];temporal=[]
     def expert(stage):
      assert scheduled;rr=read(BASE/'experts'/stage/cond/f'{parent:05}.json');assert rr['pixel_sha256']==pixel;cf=BASE/'experts'/rr['cache'];assert sha(cf)==rr['cache_sha256'];calls.append(stage);return {**load(cf),'pixel_sha256':pixel}
     def tp():
      e=expert('temporal');temporal.append(e);return e
     def sp_provider():return expert('spatial')
     native=data['prediction']
     x,ev=policy.arrive(data,scheduled,tp,sp_provider);assert calls==(['temporal','spatial'] if scheduled else []) and x['pre_state_sha256']==previous
     if p['availability']==0:
      assert not x['updated'] and torch.equal(x['output_prediction']['boxes'],native['boxes'].cpu()) and x['output_prediction']['indices']==native['indices']
     rein=None
     if arrival in [0,len(seq)-1]:
      rein=reinsert(model,shifted,row,policy.actor.state(),ev)
      if scheduled:full_temporal_check(model,shifted,row,device_tree(x['pre_state'],'cuda'),x['temporal_layers'])
     u=x.get('update');payload=dict(parent=parent,order=order,condition=cond,arrival=arrival,first_query_of_source=firstsource,expert_scheduled=scheduled,updated=x['updated'],pixel_sha256=pixel,generation=spec,pre_state=x['pre_state'],post_state=x['post_state'],pre_sha=x['pre_state_sha256'],post_sha=x['post_state_sha256'],update=u,rewards=x.get('rewards'),source_native=compact_prediction(native),slow=compact_prediction(x['prediction']),final_indices=x['output_prediction']['indices'],temporal=x.get('temporal'),reinsertion=rein,GT_read=False)
     save(f,payload);write(receipt,dict(sha256=sha(f),parent=parent,pre_sha=x['pre_state_sha256'],post_sha=x['post_state_sha256'],updated=x['updated']));previous=x['post_state_sha256'];stream_updates+=x['updated'];done+=1;created+=1
     if created%20==0:status(OUT/'ONLINE_STATUS.json',dict(status='running',done=done,total=p['total'],condition=cond,order=order,arrival=arrival,seconds=time.monotonic()-tick));print('ONLINE',done,p['total'],cond,order,arrival,'seconds',round(time.monotonic()-tick,2),flush=True)
     del data,x,ev,frames,shifted,payload;gc.collect();torch.cuda.empty_cache()
    policy.close();policy=None;assert state_hash(model.state_dict())==mh
    if limit and created>=limit:break
   if limit and created>=limit:break
  verify();state='completed' if done==p['total'] else 'bounded_batch_complete'
  if state=='completed':
   manifest={str(f.relative_to(OUT)):sha(f) for f in (OUT/'online').rglob('*.json')};assert len(manifest)==p['total']
   write(OUT/'PREDICTION_BARRIER.json',dict(cells=p['total'],files=manifest,GT_read=False,model_restored=True,time=time.time()))
 except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  if policy:policy.close()
  r=dict(stage='online',status=state,done=done,new_arrivals=created,seconds=time.monotonic()-tick,failure=failure,time=time.time());write(OUT/'allocations'/f'{time.time_ns()}.json',r);status(OUT/'ONLINE_STATUS.json',r)
  if lease:lease.close()
if __name__=='__main__':run()
