"""Finite no-GT online spatial OPD stages with independent per-arm LN chains."""
import sys,os,time,collections,gc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_spatial_opd_common_v1 import *
from scripts.decota_paper_common_v1 import load_npz

def capture(model,expert,stage,row,condition,cache):
 import torch
 from scripts.run_decota_paper_main_v1 import frames_for,pack_expert,unpack_expert
 from scripts.run_tastvg_full_b1_experts_v1 import observation
 from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
 from methods.decota_final_simplified_v1.observations import observations
 from methods.decota_final_simplified_v1.objectives import prediction
 ds=stage['dataset'];source=stage['source'];frames,ids,reuse=frames_for(ds,row,cache)
 shifted,pixel,spec=observation(row,condition,frames);batch,records,base=frozen_forward(model,shifted,row)
 native=prediction(base.zero['logits'],base.zero['boxes'],records,ids)
 f=BASE/'inputs'/f'{source}_to_{ds}'/condition/f'{row["ordinal"]:05}.pt'
 newcalls=0;oldrc=None
 if f.exists():
  rc=read(f.with_suffix('.json'));assert sha(f)==rc['sha256'];inp=load(f)
  assert inp['pixel_sha256']==pixel and torch.equal(native['boxes'].cpu(),inp['native_boxes']) and inp['interval']==native['physical_interval']
  ex=unpack_expert(inp['expert'])
 else:
  oldjob='t1_ours_hc2' if ds=='hc2' else 't1_ours_vid'
  oldpath=PAPER/oldjob/'inputs'/condition/f'{row["ordinal"]:05}.npz'
  if stage['split']!='mechanism' and oldpath.exists():
   a,md,oldrc=load_npz(oldpath)
   assert md['pixel_sha256']==pixel and md['interval']==native['physical_interval']
   assert torch.equal(native['boxes'].cpu(),torch.from_numpy(a['native_boxes']))
   ex=unpack_expert(md['expert'])
  else:ex=observations(expert,row['parses'],shifted,ids,native['indices'],audit=True);newcalls=ex['new_DINO']
  inp=dict(dataset=ds,source=source,parent=row['ordinal'],pixel_sha256=pixel,frame_ids=ids,
   native_boxes=native['boxes'].detach().cpu(),interval=native['physical_interval'],indices=native['indices'],
   expert=pack_expert(ex),corruption_spec=spec,new_DINO_calls=newcalls,old_cache_sha256=None if oldrc is None else oldrc['sha256'],GT_read=False)
  save(f,inp);rc=dict(sha256=sha(f),runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),GT_read=False,time=time.time());write(f.with_suffix('.json'),rc)
 del frames,shifted,batch
 return base,native,ex,dict(path=str(f.relative_to(BASE)),sha256=sha(f),new_DINO_calls=newcalls,decoded_frame_reuse=reuse)

def execute(stage_name,qualify=False):
 import torch
 from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row
 from methods.decota_final_simplified_v1.observations import SpatialExpert
 from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
 from methods.decota_final_simplified_v1.tensors import detached,state_hash
 from vg_tta.spatial_online_state_v1 import arrival
 from vg_tta.decota_actuation_scope_v1 import commit_state
 from vg_tta.decota_spatial_opd_v1 import fit,ARMS,mean_coordinates,action_boxes
 from vg_tta.decota_spatial_opd_audit_v1 import audit
 lock();plan=read(BASE/'DESIGN_LOCK.json');stage=dict(plan['stages'][stage_name]);ds=stage['dataset']
 if not qualify:assert read(BASE/'QUALIFICATION.json')['status']=='pass'
 dest=BASE/('qualification/'+stage_name if qualify else 'stages/'+stage_name)
 if (dest/'PREDICTION_BARRIER.json').exists():return
 lease=gpu();model=model_for(stage['source']);modelhash=state_hash(model.state_dict());expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT)
 cache=collections.OrderedDict();files={};count=0;costtime=time.time();records=[]
 try:
  for condition in stage['conditions']:
   for order,seq in stage['orders'].items():
    if qualify:
     if order!='order1':continue
     seq=seq[:2]
    previous={a:None for a in ARMS};prevsha={a:None for a in ARMS}
    for at,parent in enumerate(seq):
     budget();row=read_row(ds,parent)
     paths={a:dest/condition/order/a/f'{at:05}.pt' for a in ARMS}
     if all(f.exists() for f in paths.values()):
      for arm,f in paths.items():
       rc=read(f.with_suffix('.json'));assert sha(f)==rc['sha256'];z=load(f)
       assert z['parent']==parent and z['previous_payload_sha256']==prevsha[arm]
       previous[arm]=z['committed'];prevsha[arm]=sha(f);files[str(f.relative_to(BASE))]=sha(f);count+=1
      continue
     begin=time.perf_counter();base,native,ex,inputrc=capture(model,expert,stage,row,condition,cache);captureseconds=time.perf_counter()-begin
     with torch.no_grad():
      assert torch.equal(base.values()['boxes'],base.zero['boxes'])
      roundtrip=float((action_boxes(mean_coordinates(base.values()['boxes']))-base.values()['boxes']).abs().max())
      assert roundtrip<2e-7
     for arm in ARMS:
      f=paths[arm]
      if f.exists():
       rc=read(f.with_suffix('.json'));assert sha(f)==rc['sha256'];z=load(f);previous[arm]=z['committed'];prevsha[arm]=sha(f);files[str(f.relative_to(BASE))]=sha(f);count+=1;continue
      initial=arrival(base.initial,previous[arm],'O-split');assert torch.count_nonzero(initial['spatial.query_residual'])==0
      torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
      result=fit(base,initial,ex,row['frame_ids'],row['key'],arm);torch.cuda.synchronize();seconds=time.perf_counter()-begin
      startaudit=time.perf_counter();check=audit(result,ex);cpuseconds=time.perf_counter()-startaudit
      committed=commit_state(detached(initial,'cpu'),result['state'])
      for name,v in initial.items():assert torch.equal(committed[name],torch.zeros_like(v.cpu()) if name=='spatial.query_residual' else v.cpu()+(result['state'][name]-v.cpu())/16)
      z=dict(arm=arm,parent=parent,stage=stage_name,condition=condition,order=order,arrival=at,dataset=ds,source=stage['source'],
       fit=result,committed=committed,previous_payload_sha256=prevsha[arm],input=inputrc,interval=native['physical_interval'],
       math_audit=check,native_exact_central_parity=True,chart_roundtrip_max_error=roundtrip,
       compute=dict(capture_seconds=captureseconds,fit_GPU_seconds=seconds,CPU_math_seconds=cpuseconds,
        CUDA_peak_allocated=torch.cuda.max_memory_allocated(),CUDA_peak_reserved=torch.cuda.max_memory_reserved(),
        backward_steps=result['gradient_calls'],new_DINO_calls=inputrc['new_DINO_calls'] if arm==ARMS[0] else 0),
       query_reset=True,Adam_reset=True,LN_writeback=1/16,GT_read=False,runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),engineering_revisions=revision_receipt())
      save(f,z);h=sha(f);write(f.with_suffix('.json'),dict(sha256=h,runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),engineering_revisions=revision_receipt(),GT_read=False,time=time.time()))
      files[str(f.relative_to(BASE))]=h;prevsha[arm]=h;previous[arm]=committed;count+=1
      if qualify:records.append(dict(dataset=ds,parent=parent,arm=arm,math_audit=check,roundtrip_max_error=roundtrip,informative_rounds=sum(r['updated'] for r in result['rounds']),GT_read=False))
      del z,result,initial
     base.restore(base.initial);del base,native,ex;gc.collect();torch.cuda.empty_cache()
     status(dest/'STATUS.json',dict(status='online_running',stage=stage_name,pid=os.getpid(),done=count,total=6 if qualify else len(stage['conditions'])*sum(len(v) for v in stage['orders'].values())*3,condition=condition,order=order,GT_read=False,time=time.time()))
     print('OPD_PROGRESS',stage_name,condition,order,count,round(time.time()-costtime,1),flush=True)
  assert state_hash(model.state_dict())==modelhash
  expected=12 if qualify else len(stage['conditions'])*sum(len(v) for v in stage['orders'].values())*3
  # Qualification is two queries times three arms = six, for each direction.
  if qualify:expected=6
  assert count==expected,(count,expected)
  write(dest/'PREDICTION_BARRIER.json',dict(status='sealed',stage=stage_name,qualification=qualify,cells=count,files=files,engineering_revisions=revision_receipt(),GT_read=False,time=time.time()))
  if qualify:write(dest/'QUALIFICATION.json',dict(status='pass',records=records,actual_queries=2,GT_read=False,time=time.time()))
  status(dest/'STATUS.json',dict(status='sealed_pending_CPU_audit',done=count,total=count,GT_read=False,time=time.time()))
 finally:lease.close()

def qualification():
 # Each dataset qualification must run in its own process. This aggregator
 # never initializes GPU or installs another clean annotation-loader wrapper.
 verify()
 records=[read(BASE/'qualification'/s/'QUALIFICATION.json') for s in ['dev_hc2','dev_vidstg']]
 assert all(r['status']=='pass' for r in records)
 assert any(q['informative_rounds'] for r in records for q in r['records']),'Qualification has no distinguishable feedback; root must inspect exploration without confirmation GT'
 write(BASE/'QUALIFICATION.json',dict(status='pass',records=records,actual_queries=4,adapted_queries=12,GT_read=False,time=time.time()))

if __name__=='__main__':
 try:
  if sys.argv[1]=='qualification':qualification()
  elif sys.argv[1].startswith('qual_'):execute(sys.argv[1][5:],True)
  else:execute(sys.argv[1])
 except BaseException:
  fd=BASE/'failures'/str(time.time_ns());fd.mkdir(parents=True,exist_ok=True);(fd/'traceback.txt').write_text(traceback.format_exc())
  status(BASE/'FAILURE.json',dict(status='failed',pid=os.getpid(),failure=str(fd),GT_read=False,time=time.time()));raise
