"""One precisely bounded engineering replay restores original full raw SHA; no Source."""
from pilot_execution_v2 import *
import traceback,gc

def validate_restore():
 validate_execution();p=read(D/'raw_restore_plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p

def run():
 import torch,numpy as np,importlib.util
 from stvg.runtime import setup,lease
 from model_adapter_v1 import build_cpu,strict_fill
 validate_restore();setup();torch.set_num_threads(1);parent.gt_guard();lock=lease();folder=D/'first_raw_restore_v1';assert not(folder/'attempt.json').exists();who=identity();atomic(folder/'attempt.json',dict(worker=who,status='one_additional_precise_tracking_query_for_original_missing_raw_SHA',max_tracking_queries=1,new_Source=0,new_original_expert=0,GT_read=False));started=time.perf_counter();model=None;model_load=None;request=None;stage='before_decode';rawfiles=[]
 try:
  row=rows('hc2')[0];original=read(D/'pilot_v1/hc2/bridge_waiting.json')['files'][0];tick=time.perf_counter();frames,pos,native,arrays,proof=payload('hc2',row);decode=time.perf_counter()-tick;assert proof==original['input_binding'];stage='model_load';tick=time.perf_counter();model,load=build_cpu();model=model.cuda().eval().requires_grad_(False);torch.cuda.synchronize();model_load=time.perf_counter()-tick;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  import sam2.sam2_video_predictor as module
  import sam2.utils.misc as misc
  old=module.fill_holes_in_mask_scores;calls=[0];strict=strict_fill(misc)
  def fill(*a,**kw):calls[0]+=1;return strict(*a,**kw)
  module.fill_holes_in_mask_scores=fill;spec=importlib.util.spec_from_file_location('_unchanged_EdgeTAM_exact_raw_restore',S/'adapter.py');adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter);N,H,W,_=frames.shape;torch.cuda.reset_peak_memory_stats();resident=torch.cuda.memory_allocated();IO=[0.]
  def checkpoint(i,scores):
   torch.cuda.synchronize();t=time.perf_counter();path=folder/'raw'/f'{int(i)}.pt';save(path,dict(scores=scores.detach().cpu(),frame_index=int(i),input_binding=proof,GT_read=False));e=original['raw_mask_checkpoints'][int(i)];assert int(i)==len(rawfiles)and sha(path)==e['sha256'],'Hard failure: original full float bytes SHA differs; preserve, never retry';assert scores.shape==(1,1,H,W)and torch.isfinite(scores).all();rawfiles.append(dict(original_path=e['path'],restored_path=str(path.relative_to(D)),sha256=e['sha256']));IO[0]+=time.perf_counter()-t
  stage='one_precise_actual_tracking';tick=time.perf_counter();boxes,info=adapter.propagate_boxes(model,module,frames,pos,native,checkpoint);torch.cuda.synchronize();request=time.perf_counter()-tick;assert len(rawfiles)==N and calls[0]>0
  with np.load(D/original['path'])as z:assert np.array_equal(z['EdgeTAM'],boxes)and np.array_equal(z['positions8'],pos)and np.array_equal(z['EdgeTAM_interval'],arrays['Ours_interval'])
  module.fill_holes_in_mask_scores=old;atomic(folder/'completion.json',dict(status='one_exact_original_float_raw_SHA_restored_all_native_frames_prediction_exact',worker=who,queries=1,original_query_count_not_increased=True,additional_engineering_tracking_queries=1,model_load_seconds=model_load,strict_official_model=load,decode_and_provenance_seconds=decode,tracking_request_including_raw_IO_seconds=request,raw_checkpoint_IO_seconds=IO[0],elapsed_wall_seconds=time.perf_counter()-started,resident_GPU_bytes=resident,peak_GPU_bytes=torch.cuda.max_memory_allocated(),actual_strict_postprocess_calls=calls[0],all_original_raw_mask_bytes_SHA_exact=True,original_prediction_sha256=original['sha256'],restored=rawfiles,new_Source=0,new_original_expert=0,new_GT_scoring=0,never_call_this_unpaid_or_original_one_query_cost=True))
 except BaseException:
  atomic(folder/'failure_preserved.json',dict(worker=who,error=traceback.format_exc(),stage=stage,model_load_seconds=model_load,tracking_request_seconds=request,partial_raw_files=rawfiles,elapsed_wall_seconds=time.perf_counter()-started,additional_attempts_allowed=0));raise
 finally:model=None;gc.collect();torch.cuda.empty_cache();lock.close()
if __name__=='__main__':run()
