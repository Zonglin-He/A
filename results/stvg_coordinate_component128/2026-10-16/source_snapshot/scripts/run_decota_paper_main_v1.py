"""Real complete-query online inference, bounded memory and no diagnostic labels."""
import os,sys,time,gc,json,collections,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_paper_common_v1 import *

def lock():
 if (BASE/'RUNTIME_LOCK.json').exists():return verify()
 d=read(BASE/'DESIGN_LOCK.json')
 own=['protocols/decota_paper_experiments_v1.md','scripts/decota_paper_common_v1.py',
  'scripts/prepare_decota_paper_v1.py','scripts/run_decota_paper_main_v1.py',
  'scripts/continue_decota_paper_v1.py','vg_tta/decota_fixed_full_audit_v1.py',
  'vg_tta/decota_identity_commitment_v1.py','vg_tta/decota_optimizer_posterior_r1_v1.py',
  'vg_tta/tastvg_decota_critic_p0_v1.py','vg_tta/decota_actuation_scope_v1.py',
  'vg_tta/c1_enabling_tricks_v1.py','vg_tta/spatial_online_state_v1.py',
  'vg_tta/tastvg_decota_c1_same_domain_v1.py','scripts/run_spatial_ssl_gpu_v1.py',
  'scripts/run_spatial_regression_alignment_v1.py','scripts/run_tastvg_full_b1_experts_v1.py',
  'scripts/c1_controlled_corruption_v1.py','vg_tta/tastvg_deployment_corruption_v2.py',
  'vg_tta/exact_frame_decode_audit_v2.py','vg_tta/tastvg_paper48_hc2_decode_v1.py',
  'vg_tta/unanchored_dense_shift_data_v1.py','scripts/run_tastvg_evidence_vulnerability_v2.py']
 own += [str(f.relative_to(ROOT)) for f in (ROOT/'methods/decota_final_simplified_v1').glob('*.py')]
 from methods.decota_final_simplified_v1.config import MethodConfig,EXPERT_SHA256,EXPERT_SNAPSHOT
 weights={}
 for direct in ['vid_to_hc1','hc2_to_vid']:
  c=MethodConfig.for_direction(direct);assert sha(ROOT/c.checkpoint)==c.checkpoint_sha256;weights[c.source_dataset]=dict(path=c.checkpoint,sha256=c.checkpoint_sha256)
 assert sha(ROOT/EXPERT_SNAPSHOT/'model.safetensors')==EXPERT_SHA256
 write(BASE/'RUNTIME_LOCK.json',dict(version=d['version'],pins={f:sha(ROOT/f) for f in own},inputs=d['inputs'],protected=d['protected'],
  design_sha256=sha(BASE/'DESIGN_LOCK.json'),checkpoints=weights,DINO_sha256=EXPERT_SHA256,expert_snapshot=EXPERT_SNAPSHOT,
  all_query_total=41355,GT_inference=False,time=time.time()))
 return verify()

def gpu():
 import torch,numpy as np
 from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
 install_clean_loader();sys.addaudithook(guard)
 torch.set_num_threads(4);torch.manual_seed(20260920);np.random.seed(20260920)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 from scripts.run_final_simplification_v1 import lease
 return lease()

def model_for(source):
 from scripts.run_spatial_regression_alignment_v1 import model_load
 from methods.decota_final_simplified_v1.tensors import state_hash
 model=model_load('hcstvg1_test' if source=='vidstg' else 'vidstg_test').eval().requires_grad_(False)
 model._fixed_full_state_hash=state_hash(model.state_dict())
 return model

def read_row(ds,parent):
 p=read(BASE/ds/'PLAN.json');r=p['rows'][parent];sf=BASE/ds/'subjects'/f'{parent:05}.json';s=read(sf)
 assert s['ordinal']==parent and s['caption_sha256']==hashlib.sha256(r['input']['caption'].encode()).hexdigest() and not s['GT_read']
 return {**r,'parses':s['parses'],'subject_sha256':sha(sf)}

def frames_for(ds,row,cache):
 assert ds in ('hc2','vidstg')
 if ds=='hc2':
  from vg_tta.tastvg_paper48_hc2_decode_v1 import decode
  decoder_id='HC2_official_output_timing'
 else:
  from vg_tta.exact_frame_decode_audit_v2 import decode
  decoder_id='Vid_original_frame_vsync0'
 # Decoder routing is local. Smoke uses both datasets in one process, so
 # changing the shared module's decode binding contaminates its second half.
 key=digest(dict(dataset=ds,decoder=decoder_id,input={k:row['input'][k] for k in ['video_path','video_sha256','frame_ids','width','height']}))
 if key in cache:frames,ids=cache.pop(key);cache[key]=(frames,ids);return frames,ids,True
 frames,ids=decode(row['input']);assert ids==row['frame_ids'];cache[key]=(frames,ids)
 while sum(x[0].nbytes for x in cache.values())>512*2**20 and len(cache)>1:cache.popitem(last=False)
 return frames,ids,False

def capture_input(model,expert,job,ds,row,cond,frame_cache):
 import torch
 from scripts.run_tastvg_full_b1_experts_v1 import observation
 from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
 from methods.decota_final_simplified_v1.observations import observations
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import detached
 f=BASE/job/'inputs'/cond/f'{row["ordinal"]:05}.npz';frames,ids,reuse_decode=frames_for(ds,row,frame_cache)
 begin=time.perf_counter();shifted,pixel,spec=observation(row,cond,frames)
 batch,records,base=frozen_forward(model,shifted,row);native=prediction(base.zero['logits'],base.zero['boxes'],records,ids)
 key=digest(dict(input=row['input'],subject_sha256=row['subject_sha256'],parses=row['parses'],condition=cond,pixel_sha256=pixel,
  native_indices=native['indices'],source_state=model._fixed_full_state_hash))
 if f.with_suffix('.json').exists():
  a,md,rc=load_npz(f);assert md['signature']==key and md['pixel_sha256']==pixel
  assert torch.equal(native['boxes'].cpu(),torch.from_numpy(a['native_boxes'])) and native['physical_interval']==md['interval']
  ex=unpack_expert(md['expert']);new_DINO=0
 else:
  assert not f.exists(),'Preserve unreceipted input; root engineering recovery required'
  ex=observations(expert,row['parses'],shifted,ids,native['indices'],audit=True);new_DINO=ex['new_DINO']
  md=dict(job=job,dataset=ds,parent=row['ordinal'],condition=cond,signature=key,pixel_sha256=pixel,
   interval=native['physical_interval'],indices=native['indices'],frame_ids=ids,expert=pack_expert(ex),corruption_spec=spec,
   GT_read=False,subject_sha256=row['subject_sha256'],source_model_state_sha256=model._fixed_full_state_hash)
  rc=save_npz(f,dict(native_boxes=native['boxes'].cpu().numpy()),md)
 del frames,shifted,batch
 return base,native,ex,rc,dict(capture_seconds=time.perf_counter()-begin,new_DINO=new_DINO,decoded_frame_reuse=reuse_decode,
  new_backbone_prefix=1,input_bytes=rc['bytes'],pixel_sha256=pixel)

def pack_expert(ex):
 import torch
 def js(x):
  if torch.is_tensor(x):return x.detach().cpu().tolist()
  if isinstance(x,dict):return {str(k):js(v) for k,v in x.items()}
  if isinstance(x,(list,tuple)):return [js(v) for v in x]
  if hasattr(x,'item'):return x.item()
  return x
 return dict(positions4=ex['positions4'],anchors=js(ex['anchors']),new_DINO=ex['new_DINO'],actual_observation_positions=ex['actual_observation_positions'],
  observations=[dict(view=v,position=p,probe=js(a['probe']),receipt=js(a['receipt'])) for (v,p),a in sorted(ex['observations'].items())])

def unpack_expert(z):
 import torch
 out={k:v for k,v in z.items() if k!='observations'};out['observations']={}
 for x in z['observations']:
  probe=dict(x['probe']);probe['boxes']=torch.tensor(probe['boxes'],dtype=torch.float32).reshape(-1,4)
  out['observations'][x['view'],x['position']]=dict(probe=probe,receipt=x['receipt'])
 return out

def smoke(output_root=None):
 output_root=BASE if output_root is None else Path(output_root)
 import torch
 from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
 from methods.decota_final_simplified_v1.observations import SpatialExpert
 from methods.decota_final_simplified_v1.tensors import detached,state_hash
 from vg_tta.decota_identity_commitment_v1 import fit
 from vg_tta.tastvg_decota_c1_same_domain_v1 import NormalizedSpatialReplay
 from vg_tta.decota_actuation_scope_v1 import commit_state
 from vg_tta.spatial_online_state_v1 import arrival
 from vg_tta.decota_fixed_full_audit_v1 import audit
 lease=gpu();verify();expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);stats=[]
 try:
  for job,ds,source in JOBS:
   model=model_for(source);previous=None;modelhash=state_hash(model.state_dict());cache=collections.OrderedDict()
   for parent in [0,1]:
    row=read_row(ds,parent);base,native,ex,rc,cost=capture_input(model,expert,job,ds,row,'clean',cache)
    views=[dict(H=base.norm(v['prefix']),info={k:a for k,a in v['info'].items() if k not in ('encoded_feature','frames_cls','videos_cls')},vis_pos=v['vis_pos']) for v in base.views]
    data=dict(views=detached(views),records=[],frame_ids=row['frame_ids'],prediction=native)
    norm=NormalizedSpatialReplay(model,data);initial=arrival(base.initial,previous,'O-split')
    z=fit(base,initial,ex,row['frame_ids'],row['key'],'top1');w=fit(norm,initial,ex,row['frame_ids'],row['key'],'top1')
    assert z['selected_step']==w['selected_step'] and len(z['path'])==len(w['path'])
    for a,b in zip(z['path'],w['path']):
     assert a['loss']==b['loss'] and torch.equal(a['boxes'],b['boxes']) and state_hash(a['state'])==state_hash(b['state'])
     if 'update' in a:assert torch.equal(a['update']['gradient'],b['update']['gradient']) and torch.equal(a['update']['raw'],b['update']['raw'])
    check=audit(z,ex);previous=commit_state(detached(initial,'cpu'),z['state'])
    from scripts.decota_matrix_common_v1 import save
    sf=output_root/'smoke'/job/f'{parent:05}.pt';save(sf,dict(fit=z,expert=detached(ex,'cpu'),native=detached(native,'cpu'),committed=previous,audit=check))
    stats.append(dict(job=job,parent=parent,sha256=sha(sf),steps=z['gradient_calls'],raw_normalized_all_steps_bitwise=True,
     inherited_prestate=True,math_audit=check,compute=cost,GT_read=False))
    del base,norm,z,w,native,ex,views,data;gc.collect();torch.cuda.empty_cache()
   assert state_hash(model.state_dict())==modelhash;del model;cache.clear();gc.collect();torch.cuda.empty_cache()
  write(output_root/'SMOKE_ROOT_ACCEPTANCE.json',dict(status='pass',records=stats,actual_queries=4,GT_read=False,time=time.time()))
  print('PAPER_MAIN_SMOKE_PASS',4,flush=True)
 finally:lease.close()

def execute(job):
 import torch,numpy as np
 from methods.decota_final_simplified_v1.observations import SpatialExpert
 from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
 from methods.decota_final_simplified_v1.tensors import detached,state_hash
 from vg_tta.decota_identity_commitment_v1 import fit
 from vg_tta.spatial_online_state_v1 import arrival
 from vg_tta.decota_actuation_scope_v1 import commit_state
 from vg_tta.decota_fixed_full_audit_v1 import audit,vector
 job,ds,source=next(x for x in JOBS if x[0]==job);p=read(BASE/ds/'PLAN.json');out=BASE/job
 verify();assert read(BASE/'SMOKE_ROOT_ACCEPTANCE.json')['status']=='pass'
 subjects=read(BASE/ds/'SUBJECT_BARRIER.json');assert subjects['count']==len(p['rows'])
 for name,h in subjects['files'].items():assert sha(BASE/ds/'subjects'/name)==h
 lease=gpu();model=model_for(source);mh=state_hash(model.state_dict());expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT)
 eh=state_hash(expert.model.state_dict());cache=collections.OrderedDict();files={};tick=time.time();count=0;backwards=0;DINO=0;new_prefix=0;bytes_used=0
 schema=None
 try:
  for cond in p['conditions']:
   for order,seq in p['orders'].items():
    previous=None;prevsha=None
    for at,parent in enumerate(seq):
     budget();f=out/'online'/cond/order/f'{at:05}.npz';row={**p['rows'][parent]}
     if f.with_suffix('.json').exists():
      a,md,rc=load_npz(f);assert md['parent']==parent and md['previous_payload_sha256']==prevsha and not md['GT_read']
      schema=read(out/'STATE_SCHEMA.json');previous=unvector(a['committed'],schema)
     else:
      assert not f.exists(),'Preserve unreceipted prediction for recovery'
      sf=BASE/ds/'subjects'/f'{parent:05}.json';ss=read(sf);assert ss['caption_sha256']==hashlib.sha256(row['input']['caption'].encode()).hexdigest()
      row.update(parses=ss['parses'],subject_sha256=sha(sf))
      base,native,ex,inputrc,cost=capture_input(model,expert,job,ds,row,cond,cache)
      if schema is None:
       schema=[dict(name=n,shape=list(v.shape),count=v.numel()) for n,v in base.initial.items()];write(out/'STATE_SCHEMA.json',schema)
      initial=arrival(base.initial,previous,'O-split');assert torch.count_nonzero(initial['spatial.query_residual'])==0
      torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
      z=fit(base,initial,ex,row['frame_ids'],row['key'],'top1');torch.cuda.synchronize();model_fit_seconds=time.perf_counter()-begin
      audit_begin=time.perf_counter();check=audit(z,ex);CPU_math_seconds=time.perf_counter()-audit_begin
      assert state_hash(base.state())==state_hash(initial);committed=commit_state(detached(initial,'cpu'),z['state'])
      names=list(initial);a=dict(box_path=np.stack([h['boxes'].numpy() for h in z['path']]),initial=vector(initial,names),
       selected=vector(z['state'],names),committed=vector(committed,names))
      md=dict(job=job,dataset=ds,source_checkpoint=source,parent=parent,condition=cond,order=order,arrival=at,
       previous_payload_sha256=prevsha,input_receipt_sha256=sha((out/'inputs'/cond/f'{parent:05}.json')),
       pixel_sha256=cost['pixel_sha256'],subject_sha256=row['subject_sha256'],
       interval=native['physical_interval'],selected_step=z['selected_step'],losses=[h['loss'] for h in z['path']],
       empty=z['empty'],gradient_calls=z['gradient_calls'],frame_metadata=z['metadata']['frame_metadata'],math_audit=check,
       initial_sha256=state_hash(initial),selected_sha256=state_hash(z['state']),committed_sha256=state_hash(committed),
       compute={**cost,'fit_and_CPU_math_seconds':time.perf_counter()-begin,'model_fit_seconds':model_fit_seconds,'CPU_math_seconds':CPU_math_seconds,'CUDA_peak_allocated':torch.cuda.max_memory_allocated(),'CUDA_peak_reserved':torch.cuda.max_memory_reserved()},query_reset=True,Adam_reset=True,LN_writeback=1/16,GT_read=False)
      if at%100==0:
       from scripts.decota_matrix_common_v1 import save
       full=out/'audit_samples'/cond/order/f'{at:05}.pt';save(full,dict(fit=z,expert=detached(ex,'cpu'),source_initial=detached(base.initial,'cpu'),committed=committed))
       md['raw_math_sample']=dict(path=str(full.relative_to(BASE)),sha256=sha(full))
      rc=save_npz(f,a,md);previous=committed
      DINO+=cost['new_DINO'];new_prefix+=1;backwards+=z['gradient_calls']
      del base,z,ex,native,a,initial,committed;gc.collect()
     prevsha=rc['sha256'];files[str(f.relative_to(BASE))]=rc['sha256'];bytes_used+=rc['bytes'];count+=1
     if count%10==0 or count==p['arrivals_per_checkpoint']:
      status(out/'STATUS.json',dict(status='online_running',pid=os.getpid(),done=count,total=p['arrivals_per_checkpoint'],condition=cond,order=order,arrival=at,
       seconds=time.time()-tick,output_bytes=bytes_used,new_DINO=DINO,new_backbone_prefix=new_prefix,backwards=backwards,GT_read=False,time=time.time()))
      print('PAPER_MAIN_PROGRESS',job,cond,order,count,p['arrivals_per_checkpoint'],round(time.time()-tick,1),flush=True)
  assert count==p['arrivals_per_checkpoint'] and state_hash(model.state_dict())==mh and state_hash(expert.model.state_dict())==eh
  write(out/'PREDICTION_BARRIER.json',dict(status='sealed',job=job,dataset=ds,source_checkpoint=source,files=files,cells=count,
   source_model_state_sha256=mh,DINO_state_sha256=eh,GT_read=False,time=time.time()))
  status(out/'STATUS.json',dict(status='completed_sealed_pending_global',done=count,total=count,GT_read=False,time=time.time()))
 finally:lease.close()

def unvector(a,schema):
 import torch
 result={};offset=0
 for r in schema:
  result[r['name']]=torch.from_numpy(a[offset:offset+r['count']].copy()).reshape(r['shape']);offset+=r['count']
 assert offset==1792==len(a);return result

def seal():
 verify();bars={};count=0
 for job,_,_ in JOBS:
  f=BASE/job/'PREDICTION_BARRIER.json';b=read(f);assert b['status']=='sealed' and not b['GT_read']
  for rel,h in b['files'].items():assert sha(BASE/rel)==h and read((BASE/rel).with_suffix('.json'))['sha256']==h
  count+=b['cells'];bars[job]=sha(f)
 assert count==41355
 write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',jobs=bars,cells=count,GT_read=False,time=time.time()))

if __name__=='__main__':
 action=sys.argv[1]
 try:
  if action=='lock':lock()
  elif action=='smoke':smoke(Path(sys.argv[2]) if len(sys.argv)>2 else None)
  elif action=='seal':seal()
  else:execute(action)
 except BaseException:
  fd=BASE/'failures'/str(time.time_ns());fd.mkdir(parents=True,exist_ok=True)
  (fd/'traceback.txt').write_text(traceback.format_exc());status(BASE/'WORKER_FAILURE.json',dict(status='failed',action=action,pid=os.getpid(),failure_path=str(fd),time=time.time()))
  raise
