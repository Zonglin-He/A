"""Six fixed-input calls identify clip independence, no temporal task scoring."""
from pathlib import Path
import os,sys,time,signal,fcntl,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_clip_independence_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_clip_independence';OLD=R/'artifacts/stvg_temporal_local_motion128_v1';EARLY=R/'artifacts/stvg_temporal_early_fusion128_v1'

def prepare():
 p=read(OLD/'PLAN_LOCK_RECOVERY001.json');assert all(sha(R/f)==h for f,h in p['pins'].items());partial=list((OLD/'partials').glob('*.pt'));assert len(partial)==1;assert not(OLD/'GT_EXPOSURE.json').exists()
 files=list(p['pins'])+['protocols/stvg_temporal_clip_independence_v1.md','scripts/run_stvg_temporal_clip_independence_v1.py','artifacts/stvg_temporal_local_motion128_v1/PLAN_LOCK_RECOVERY001.json','artifacts/stvg_temporal_local_motion128_v1/COST_RECOVERY001.json'];q=dict(record=p['records'][0],old_partial=dict(path=str(partial[0].relative_to(R)),sha256=sha(partial[0])),source_checkpoint=p['source_checkpoint'],source_checkpoint_sha256=p['source_checkpoint_sha256'],pins={f:sha(R/f)for f in sorted(set(files))},GPU_seconds_cap=120,created_epoch=time.time(),deadline_epoch=time.time()+7200,no_GT_or_task_scores=True,new_experts=0,new_training=0,spatial_frozen=True)
 put(D/'PLAN_LOCK.json',q);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in q.items()if k not in ['record','old_partial','created_epoch','deadline_epoch']});print('six-call engineering audit locked',flush=True)

def metrics(a,b):
 a=a.double();b=b.double();d=a-b;return dict(max_relative=float((d.abs()/b.abs().clamp_min(1)).max()),max_absolute=float(d.abs().max()),RMS=float(d.square().mean().sqrt()),relative_L2=float(d.norm()/b.norm().clamp_min(1e-30)),exact=bool((a==b).all()))

def measure():
 import torch,numpy as np
 from main_method.runtime import resource_guard
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source
 from methods.stvg_referdino_opd_v1._source_forward import transient_native_masks
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_temporal_local_motion128_recovery001 import decode_all,normalize,clip_indices
 from vg_tta.stvg_temporal_registered_motion128_v1 import tensor_sha
 setup()
 def reject(event,args):
  if event=='open'and args and isinstance(args[0],(str,bytes,Path))and str(args[0]).endswith('/data/hcstvg2_official_metadata/val_v2.json'):raise PermissionError('clip audit never reads GT')
 sys.addaudithook(reject);p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items())and sha(R/p['old_partial']['path'])==p['old_partial']['sha256'];lock=lease();beg=time.perf_counter();partial={};hooks=[];success=False;counts={k+'_'+s:0 for k in ['motion','projection']for s in ['attempts','completed']};receipts=[]
 def check():
  resource_guard();free,total=torch.cuda.mem_get_info();assert free>5*2**30 and time.perf_counter()-beg+15<120 and time.time()<p['deadline_epoch']
 def entered(k):
  def f(m,a):counts[k+'_attempts']+=1;assert counts[k+'_attempts']<=6
  return f
 def returned(k):
  def f(m,a,o):counts[k+'_completed']+=1
  return f
 try:
  resource_guard();free,total=torch.cuda.mem_get_info();fraction=min(.8,(free-5*2**30)/total);assert fraction>.4;torch.cuda.set_per_process_memory_fraction(fraction);check();m=load_source();mh=state_hash(m.state_dict());flags={n:v.training for n,v in m.named_modules()};assert not any(flags.values())
  for mod,k in [(m.vid,'motion'),(m.input_proj2,'projection')]:hooks.extend([mod.register_forward_pre_hook(entered(k)),mod.register_forward_hook(returned(k))])
  r=p['record'];raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'][r['query_ordinal']]['input'];frames=decode_all(raw,R);x=normalize(frames,raw);ix=clip_indices(raw['frame_ids'],len(frames))['DenseLocal'][:8];xx=x[ix].reshape(-1,3,*x.shape[-2:]);z=load(R/p['old_partial']['path']);assert tensor_sha(xx)==z['batches']['DenseLocal'][0]['input_sha256']and tensor_sha(xx[:8])==z['inflight']['input_sha256'];alt=xx.reshape(8,8,3,*x.shape[-2:])[[0,7,6,5,4,3,2,1]].reshape_as(xx);e=load(EARLY/'early_inputs'/f"{r['alias']}.pt");assert sha(EARLY/'early_inputs'/f"{r['alias']}.pt")==r['early_sha256'];values={}
  for role,inp,t in [('Native0',x[raw['frame_ids'][::2]],33),('Native1',x[raw['frame_ids'][1::2]],32),('Batch',xx,8),('Repeat',xx,8),('OtherPermutation',alt,8),('Single',xx[:8],8)]:
   check();partial=dict(role=role,input_sha256=tensor_sha(inp),num_frames=t,projected=None,raw=None)
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16),transient_native_masks(m):v=m.vid(inp.to('cuda'),t)
   partial['raw']=v['3'].detach().cpu();check()
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):pr=m.input_proj2(v['3'])
   partial['projected']=pr.detach().cpu();values[role]=partial;receipts.append(save(D/'stages'/f'{role}.pt',partial));partial={};del v,pr;gc.collect();torch.cuda.empty_cache();check();print('saved before assertions',role,flush=True)
  checks={}
  for off in (0,1):
   pr=values['Native'+str(off)]['projected'];hh,ww=e['original_views'][off]['info']['fea_map_size'];want=e['captured_encoder_inputs'][off]['src'][-hh*ww:].permute(1,2,0).reshape(pr.shape);checks['Native'+str(off)+'_exact']=torch.equal(pr,want)
  for stage in ['raw','projected']:
   checks['same_batch_'+stage]=metrics(values['Batch'][stage],values['Repeat'][stage]);checks['first_clip_independence_'+stage]=metrics(values['OtherPermutation'][stage][:8],values['Batch'][stage][:8]);checks['single_versus_batch_'+stage]=metrics(values['Batch'][stage][:8],values['Single'][stage])
  checks['old8eps16_threshold']=float(8*torch.finfo(torch.float16).eps);checks['original_R30_qualification_passed']=False;checks['current_fullclip_comparison_with_original_bound_passed']=checks['single_versus_batch_projected']['max_relative']<checks['old8eps16_threshold'];checks['old_saved_batch_centers_exact']=torch.equal(values['Batch']['projected'][3::8],z['features']['DenseLocal'][0]);checks['old_saved_single_exact']=torch.equal(values['Single']['projected'],z['inflight']['projection']);assert all(sha(R/f)==h for f,h in p['pins'].items())and state_hash(m.state_dict())==mh and {n:v.training for n,v in m.named_modules()}==flags and all(not v.requires_grad and v.grad is None for v in m.parameters());assert all(v==6 for v in counts.values());check();put(D/'ENGINEERING_RESULTS.json',dict(runtime=dict(torch=torch.__version__,torch_git=torch.version.git_version,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(),cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),audit_completed=True,independence_pass=all(checks[k]['exact']for k in ['same_batch_raw','same_batch_projected','first_clip_independence_raw','first_clip_independence_projected']),checks=checks,actual_calls=counts,no_GT=True,task_predictions=0,model_weights_flags_exact=True,all_outputs_saved_before_assertions=True));put(D/'GLOBAL_STAGE_BARRIER.json',dict(files=receipts,plan_sha256=sha(D/'PLAN_LOCK.json'),no_GT=True));success=True
 finally:
  for h in hooks:h.remove()
  if not success and partial:save(D/'partials'/f'{time.time_ns()}.pt',dict(partial,actual_calls=counts))
  elapsed=time.perf_counter()-beg;put(D/'COST.json',dict(GPU_seconds=elapsed+3,measured_lease_seconds=elapsed,cleanup_reserve_seconds=3,actual_calls=counts,success=success,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated()if torch.cuda.is_initialized()else 0,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved()if torch.cuda.is_initialized()else 0,execution_completed=success,not_task_qualification=True,new_experts=0,new_training=0,spatial_updates=0,student_updates=0));lock.close()

if __name__=='__main__':
 def stop(s,f):raise TimeoutError('owned clip audit RuntimeMax')
 signal.signal(signal.SIGTERM,stop);D.mkdir(exist_ok=True,parents=True);lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:globals()[sys.argv[1]]()
 except BaseException as e:put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e)));raise
 finally:lock.close()
