"""Fixed one-view capability probe, full prediction seal before separate scoring."""
from pathlib import Path
import sys,os,time,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_mirror128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_mirror128';HR=R/'artifacts/stvg_temporal_hires128_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1';FG=R/'artifacts/stvg_temporal_foreground128_v1';JM=R/'artifacts/stvg_temporal_joint_motion128_v1'
ARMS=['Native','HiRes','HiResMix','Mirror','MirrorEnsemble','MirrorMix']

def prepare():
 from vg_tta.stvg_temporal_mirror128_v1 import eligible,properties
 raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];records=[]
 for r in read(HR/'PLAN_LOCK.json')['records']:
  subject=read(R/r['subject_path'])['parses']['subject'];ok=eligible(raw[r['query_ordinal']]['input']['caption'],subject);records.append(dict(r,mirror_eligible=ok,hires_sha256=sha(HR/'predictions'/f"{r['alias']}.pt")))
 own=['protocols/stvg_temporal_mirror128_v1.md','vg_tta/stvg_temporal_mirror128_v1.py','vg_tta/stvg_temporal_hires128_v1.py','scripts/run_stvg_temporal_mirror128_v1.py','scripts/score_stvg_temporal_mirror128_v1.py','scripts/publish_stvg_temporal_mirror128_v1.py','external/TA-STVG/datasets/transforms.py','external/TA-STVG/experiments/vidstg.yaml','methods/stvg_referdino_opd_v1/_source_forward.py','methods/decota_final_simplified_v1/backbone.py','methods/stvg_referdino_opd_v1/config.json']
 count=sum(r['mirror_eligible']for r in records);assert count==119
 p=dict(records=records,count=128,seeds=[41,42,43],arms=ARMS,primary='MirrorMix',eligible=['MirrorMix'],mirror_Source_cap=count,GPU_seconds_cap=1200,process_seconds_cap=900,input_height=448,deadline_epoch=time.time()+21600,created_epoch=time.time(),pins={f:sha(R/f)for f in own},native448_control_sha256=sha(JM/'NATIVE_CONTROL.json'),spatial_frozen=True,new_experts=0,new_training=0,not_OPD=True)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked128/eligible119',flush=True)

def measure():
 import torch,numpy as np,hashlib
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source,decode
 from methods.stvg_referdino_opd_v1._source_forward import frozen_forward
 from methods.decota_final_simplified_v1.tensors import state_hash
 from main_method.runtime import resource_guard
 from vg_tta.stvg_temporal_hires128_v1 import resolution
 from vg_tta.stvg_temporal_mirror128_v1 import eligible,mirror,fuse,properties
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 setup();assert properties()['status']=='pass';p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());assert sha(JM/'NATIVE_CONTROL.json')==p['native448_control_sha256']and read(JM/'NATIVE_CONTROL.json')['status']=='pass';raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];lock=lease();beg=time.perf_counter();prev=[read(f)for f in (D/'cost').glob('*.json')];previous=sum(x['GPU_seconds']for x in prev);calls=0;files=[];times=[]
 def check():
  resource_guard();assert time.time()<p['deadline_epoch']and previous+time.perf_counter()-beg+60<p['GPU_seconds_cap']and time.perf_counter()-beg+60<p['process_seconds_cap']
 try:
  check();free,total=torch.cuda.mem_get_info();fraction=min(.8,(free-5*2**30)/total);assert fraction>.4;torch.cuda.set_per_process_memory_fraction(fraction);m=load_source();mh=state_hash(m.state_dict());loading=time.perf_counter()-beg
  for r in p['records']:
   check();a=r['alias'];dest=D/'predictions'/f'{a}.pt'
   if dest.exists():load(dest);files.append(read(dest.with_suffix('.receipt.json')));continue
   tick=time.perf_counter();assert sha(HR/'predictions'/f'{a}.pt')==r['hires_sha256'];z=load(HR/'predictions'/f'{a}.pt');row=dict(raw[r['query_ordinal']]);row['parses']=read(R/r['subject_path'])['parses'];ok=eligible(row['input']['caption'],row['parses']['subject']);assert ok==r['mirror_eligible'];ids=z['frame_ids'];obs=None
   if ok:
    assert sum(x['mirror_Source_attempts']for x in prev)+calls<p['mirror_Source_cap'];frames,actual_ids=decode(row['input']);assert ids==actual_ids and hashlib.sha256(frames.tobytes()).hexdigest()==z['observation']['RGB_sha256'];flipped=mirror(frames);assert np.array_equal(mirror(flipped),frames);torch.cuda.reset_peak_memory_stats();calls+=1
    with resolution(448):batch,recs,base=frozen_forward(m,flipped,row)
    ll=[x.detach().cpu()for x in base.zero['logits']];actions=[x.detach().cpu()for x in base.zero['actions']];grid,qm=mass_table(probabilities(recs,ll));assert grid==ids
    obs=dict(logits=ll,actions=actions,q=qm,records=recs,normalized_tensor_shape=list(batch['videos'].tensors.shape),raw_RGB_shape=list(frames.shape),peak_GPU_allocated=torch.cuda.max_memory_allocated(),RGB_sha256=hashlib.sha256(flipped.tobytes()).hexdigest(),original_RGB_sha256=z['observation']['RGB_sha256'],raw_double_reflection_exact=True,input_horizontal_reflection=True)
    base.restore(base.initial);del batch,base,frames,flipped;gc.collect();torch.cuda.empty_cache();times.append(time.perf_counter()-tick)
   else:qm=z['observation']['q'].copy()
   qs=dict(Native=z['outputs']['Native']['q'],HiRes=z['outputs']['HiRes']['q'],HiResMix=z['outputs']['HiResMix']['q'],**fuse(z['outputs']['Native']['q'],z['observation']['q'],qm));outputs={k:dict(q=q,interval=final_mbr(ids,q)['interval'])for k,q in qs.items()};assert all(outputs[k]['interval']==z['outputs'][k]['interval']for k in ['Native','HiRes','HiResMix'])
   if not ok:assert np.array_equal(qs['MirrorMix'],qs['HiResMix'])
   files.append(save(dest,dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,outputs=outputs,trajectories=z['trajectories'],observation=obs,mirror_eligible=ok,GT_read=False)));put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time()),True);print('mirror',a,round(time.perf_counter()-beg,2),'eligible',ok,flush=True)
   if len(times)==2 and not(D/'PILOT2.json').exists():
    estimate=previous+loading+max(times)*p['mirror_Source_cap']+30;put(D/'PILOT2.json',dict(case_seconds=times,model_loading=loading,estimated_GPU_seconds=estimate,allocator_fraction=fraction,GT_read=False));assert estimate<p['GPU_seconds_cap']
   del z,outputs,qs,qm;gc.collect();torch.cuda.empty_cache()
  assert state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,weights_unchanged=True));print('sealed128',flush=True)
 finally:
  now=dict(GPU_seconds=time.perf_counter()-beg+3,mirror_Source_attempts=calls,Source_offset_forwards=2*calls,new_DINO=0,new_ReferDINO=0,network_training=0,spatial_updates=0);put(D/'cost'/f'attempt_{time.time_ns()}.json',now);put(D/'COST.json',dict(GPU_seconds=previous+now['GPU_seconds'],attempts=prev+[now],mirror_Source_attempts=sum(x['mirror_Source_attempts']for x in prev)+calls),True);lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
