"""Real paired full paths, disk/audit outside the common timer."""
from runtime_v4_common import *
import sys,gc,subprocess,socket,inspect,textwrap,traceback,pickle
from multiprocessing.connection import Connection
from memory import Monitor

class RAMTeacher:
 def __init__(self):
  import torch
  a,b=socket.socketpair();fd=b.fileno();self.log=(D/'teacher.log').open('wb')
  self.process=subprocess.Popen([str(R/'outputs/edgetam_comparison_v1/runtime/bin/python'),'-B',str(S/'teacher_service_v4.py'),str(fd)],cwd=R,pass_fds=(fd,),stdout=self.log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':str(R),'PYTHONDONTWRITEBYTECODE':'1'})
  b.close();self.conn=Connection(a.detach());assert self.conn.poll(180),'teacher startup timeout';self.startup=self.conn.recv();assert self.startup['ready'];self.parent_at_request=[];self.peaks=[]
 def __call__(self,frames,caption,**kwargs):
  import torch
  parent=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved());self.parent_at_request.append(parent);self.pending_Source={k:v for k,v in inspect.currentframe().f_back.f_locals.items()if k in ['src','logits','spatial','native_grid','recs','positions','posterior','costs']}
  self.conn.send(dict(frames=frames,caption=caption));assert self.conn.poll(180),'teacher request timeout';r=pickle.loads(self.conn.recv_bytes());assert r['ok'],r.get('error');z=r['payload'];self.peaks.append(dict(allocated=parent['allocated']+z['peak_GPU_bytes'],reserved=parent['reserved']+z['peak_reserved_GPU_bytes']));return z
 def close(self):
  if self.process.poll()is None:self.conn.send(dict(stop=True));code=self.process.wait(timeout=30)
  else:code=self.process.returncode
  atomic(D/'teacher_exit.json',dict(worker=self.startup['worker'],code=code));self.conn.close();self.log.close();assert code==0

def edge_finish(edge,module,adapter,frames,positions,prompts):
 """Same original adapter numeric steps, no checkpoint/diagnostic work."""
 import torch,numpy as np
 boxes=adapter.pixel_boxes(prompts,*frames.shape[1:3])
 with adapter.scoped_rgb_loader(module):state=edge.init_state(frames,offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False)
 for j in sorted(range(8),key=lambda j:positions[j]):
  edge.add_new_points_or_box(state,frame_idx=positions[j],obj_id=1,box=boxes[j],clear_old_points=True,normalize_coords=True)
 raw={}
 for i,objects,scores in edge.propagate_in_video(state,start_frame_idx=0,reverse=False):
  mask=(scores[0,0]>0).detach().cpu().numpy();raw[i]=adapter.mask_box(mask)
 result=adapter.keep_observed(np.stack([raw[i]for i in range(len(frames))]),prompts,positions)
 return result

def run():
 import numpy as np,torch,importlib.util,types
 from stvg.runtime import setup,lease,load_source
 from stvg.decode import decode
 from stvg.native import frozen_native_forward
 from stvg.readout import prediction
 from stvg.positions import observation_positions
 from stvg.query import QuerySubjectParser
 from stvg.current_shared_roi import predictor_class,finish as original_finish
 plan=validate_runtime();assert not(D/'worker.json').exists(),'No blind timed-model repeat'
 setup();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 lock=lease();who=identity();atomic(D/'worker.json',who);begin=time.perf_counter();items=[];loads={};teacher=None;model=None;edge=None;live={}
 try:
  resources(begin);sys.path.insert(0,str(R/'studies/edgetam_comparison_v1'))
  from model_adapter_v1 import build_cpu,strict_fill
  from memory_dtype_compat_v1 import install
  spec=importlib.util.spec_from_file_location('_fixed_Source_prompt_EdgeTAM_adapter',R/'studies/edgetam_comparison_v1/adapter.py');adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
  t=time.perf_counter();edge,edge_load=build_cpu();handles,dtype_counts=install(edge);loads['EdgeTAM']=dict(seconds=time.perf_counter()-t,load=edge_load)
  import sam2.sam2_video_predictor as module
  import sam2.utils.misc as misc
  module.fill_holes_in_mask_scores=strict_fill(misc)
  teacher=RAMTeacher();loads['ReferDINO']=teacher.startup
  model_class=predictor_class();f=original_finish.__wrapped__;finish=types.FunctionType(f.__code__,{**f.__globals__,'profile':lambda:None},'same_selected_finish_without_percall_profile_IO')
  selection=read(D/'selection.json')
  for ds,rows in selection.items():
   resources(begin);refs=references(ds,rows);model=model_class(ds);model.teacher_service=teacher
   t=time.perf_counter();model.source=load_source(model.config);model.source.cpu();model.parser=QuerySubjectParser(R/'checkpoints/stanza');gc.collect();torch.cuda.empty_cache();loads[ds]=dict(Source_and_parser_startup_seconds=time.perf_counter()-t)
   def once(row,arm,repeat,warmup=False):
    resources(begin);key=row['id'];attempt=D/'attempts'/f'{ds}_{key}_{repeat}_{arm}.json';assert not attempt.exists();atomic(attempt,dict(status='before_actual_Source',worker=who,arm=arm,repeat=repeat,warmup=warmup,GT_read=False))
    torch.backends.cudnn.allow_tf32=True;torch.manual_seed(20260920);np.random.seed(20260920);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();teacher.peaks=[];teacher.parent_at_request=[]
    mon=Monitor([who['pid']]+([teacher.process.pid]if arm=='Current'else[]));mon.start();clock=time.perf_counter();t=clock
    frames,ids=decode(row['metadata'],ds);decode_seconds=time.perf_counter()-t;t=time.perf_counter();parses=dict(subject=model.parser(row['metadata']['caption'])['subject']);parse_seconds=time.perf_counter()-t;t=time.perf_counter();cap=None
    if arm=='Current':
     cap=model.capture(frames,ids,row['metadata'],parses=parses,collect_mask_diagnostics=False);out=finish(cap);boxes=out['boxes'].copy();iv=list(out['interval']);source_boxes=cap['Source'].copy();positions=list(cap['planned_positions']);source_interval=None;source_cost=cap['cost'];stage=dict(Source_seconds=source_cost['Source224_seconds'],expert_request_seconds=source_cost['expert_current_wall_seconds'],geometry_seconds=source_cost['evidence_seconds'],**out['cost']);new_expert=1;new_Edge=0
    else:
     source=model.source.cuda();torch.cuda.synchronize()
     with torch.no_grad():batch,recs,base=frozen_native_forward(source,frames,dict(input=row['metadata'],frame_ids=ids,parses=parses));native=prediction(base.zero['logits'],base.zero['boxes'],recs,ids)
     source_boxes=native['boxes'].cpu().numpy().copy();iv=list(native['physical_interval']);source_interval=iv.copy();positions=list(observation_positions(ids,native['indices']));del batch,base,native;model.source.cpu();gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();source_end=time.perf_counter();torch.backends.cudnn.allow_tf32=False;edge.cuda()
     with torch.no_grad():boxes=edge_finish(edge,module,adapter,frames,positions,source_boxes[positions])
     edge.cpu();gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();stage=dict(Source_and_offload_seconds=source_end-t,EdgeTAM_and_transfer_offload_seconds=time.perf_counter()-source_end);new_expert=0;new_Edge=1
    torch.cuda.synchronize();wall=time.perf_counter()-clock
    parent_peak=torch.cuda.max_memory_allocated();parent_reserved=torch.cuda.max_memory_reserved();mem=mon.stop()
    # Every expensive check and all saving start after the wall/memory windows.
    assert np.isfinite(wall) and wall>0
    ref=refs[key];assert np.array_equal(source_boxes,ref['Source'])and positions==list(ref['positions8'])and list(ids)==list(ref['frame_ids'])
    assert np.array_equal(boxes,ref[arm])and np.array_equal(iv,ref[arm+'_interval']),(ds,key,arm,'numeric drift; preserve, no fallback')
    rgb_sha=hashlib.sha256(frames.tobytes()).hexdigest();assert rgb_sha==ref['RGB_sha256']
    if source_interval is not None:assert np.array_equal(source_interval,ref['Source_interval'])
    peak=max([parent_peak]+[e['allocated']for e in teacher.peaks]);reserved=max([parent_reserved]+[e['reserved']for e in teacher.peaks])
    record=dict(dataset=ds,query=rows.index(row)+1,repeat=repeat,arm=arm,warmup=warmup,query_wall_seconds=wall,RGB_seconds=decode_seconds,parse_seconds=parse_seconds,stage_seconds=stage,peak_pipeline_allocated_bytes=peak,peak_pipeline_reserved_bytes=reserved,parent_peak_allocated_bytes=parent_peak,teacher_phase_peaks=teacher.peaks,NVML=mem,Source_forwards=1,ReferDINO_queries=new_expert,EdgeTAM_queries=new_Edge,RGB_sha256=rgb_sha,positions8=positions,outputs_exact_saved_reference=True,source_native_exact=True,reference_SHA=ref['reference_SHA'],verification_and_experiment_IO_outside_timer=True,GT_read=False,worker=who)
    outpath=D/'predictions'/f'{ds}_{key}_{repeat}_{arm}.npz';outpath.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(outpath,boxes=boxes,interval=iv,Source=source_boxes,positions8=positions,frame_ids=ids)
    record.update(prediction_path=str(outpath.relative_to(D)),prediction_sha256=sha(outpath));atomic(attempt,{**record,'status':'committed_actual_call'});live.clear();del frames,cap,boxes,source_boxes;gc.collect();torch.cuda.empty_cache();return record
   loads[ds]['warmup']=[once(rows[0],a,0,True)for a in ['Current','EdgeTAM']]
   atomic(D/f'{ds}_actual_warmup_bridge.json',dict(status='both_real_complete_paths_output_Source_RGB_positions_exact',calls=loads[ds]['warmup'],GT_read=False))
   for repeat in range(1,4):
    for i,row in enumerate(rows):
     for arm in (['Current','EdgeTAM']if(repeat+i)%2 else['EdgeTAM','Current']):
      item=once(row,arm,repeat);items.append(item);atomic(D/'measurements_partial.json',dict(items=items,loads=loads));atomic(D/'progress.json',dict(completed_calls=len(items),total_calls=1536,dataset=ds,repeat=repeat,query=i+1,worker=who,GT_read=False));print(ds,repeat,i+1,arm,item['query_wall_seconds'],item['peak_pipeline_allocated_bytes'],flush=True)
   atomic(D/f'{ds}_sealed.json',dict(calls=768,queries=128,parents=128,repeats=3,all_outputs_exact=True,worker=who,items_sha256=digest([v for v in items if v['dataset']==ds])))
   del model;model=None;gc.collect();torch.cuda.empty_cache()
  assert len(items)==1536 and sum(v['ReferDINO_queries']for v in items)==768 and sum(v['EdgeTAM_queries']for v in items)==768
  atomic(D/'measurements.json',dict(items=items,loads=loads,plan_fingerprint=plan['fingerprint']));atomic(D/'sealed.json',dict(status='all1536_measured_complete_paths_exact',calls=1536,Source=1536,ReferDINO=768,EdgeTAM=768,warmup_Source=4,warmup_ReferDINO=2,warmup_EdgeTAM=2,measurement_sha256=sha(D/'measurements.json'),worker=who));atomic(D/'state.json',dict(status='measured_verified_pending_independent_report_audit',worker=who))
 except BaseException:
  error=traceback.format_exc();tb=sys.exc_info()[2];paid={}
  while tb is not None:
   if tb.tb_frame.f_code.co_name=='once':
    local=tb.tb_frame.f_locals
    paid={k:local[k]for k in ['frames','ids','source_boxes','positions','iv','boxes','cap','out','parses']if k in local}
   tb=tb.tb_next
  try:
   if teacher is not None:paid['pending_Source']=getattr(teacher,'pending_Source',None)
   if paid:torch.save(paid,D/'failed_paid_in_memory_state.pt')
  except BaseException:atomic(D/'failed_state_preservation_error.json',dict(error=traceback.format_exc()))
  atomic(D/'failure.json',dict(error=error,worker=who,completed_calls=len(items),seconds=time.perf_counter()-begin,paid_in_memory_state_saved=bool(paid)));atomic(D/'state.json',dict(status='failed_preserved_no_automatic_retry',worker=who));raise
 finally:
  if teacher is not None:teacher.close()
  lock.close()
if __name__=='__main__':run()
