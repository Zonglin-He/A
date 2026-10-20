"""Finite CPU real preference OPD on sealed existing HC128 inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import sys,time,fcntl,signal,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_strong_relative_preference_opd128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_strong_relative_preference_opd128'
OLD=R/'artifacts/stvg_temporal_local_motion_completion128_v1';OP=R/'results/stvg_error_state/2026-10-20/temporal_local_motion_completion128'
STAGE=R/'artifacts/stvg_temporal_native_stage128_v1';HR=R/'artifacts/stvg_temporal_hires128_v1'
SEEDS=[41,42,43];MODES=['Direct','Fixed','Live'];STATIC=['Native','Strong','TeacherDirect','DenseLocalMix'];FITS=[m+str(s)for m in MODES for s in SEEDS]

def forbid_GT(event,args):
 if event=='open'and args and isinstance(args[0],(str,bytes,Path))and str(args[0]).endswith('/data/hcstvg2_official_metadata/val_v2.json'):
  raise PermissionError('preference OPD inference and preaudit forbid GT')

def prepare():
 import torch
 from vg_tta.stvg_temporal_strong_relative_preference_opd128_v1 import properties
 from main_method.runtime import resource_guard
 torch.set_num_threads(4);sys.addaudithook(forbid_GT);resource_guard()
 old=read(OLD/'PLAN_LOCK.json');bar=read(OLD/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['weights_and_flags_unchanged']
 assert all(sha(R/f)==h for f,h in old['pins'].items())
 own=['protocols/stvg_temporal_strong_relative_preference_opd128_v1.md','scripts/run_stvg_temporal_strong_relative_preference_opd128_v1.py',
 'scripts/score_stvg_temporal_strong_relative_preference_opd128_v1.py','scripts/publish_stvg_temporal_strong_relative_preference_opd128_v1.py',
 'vg_tta/stvg_temporal_strong_relative_preference_opd128_v1.py','vg_tta/stvg_posterior_decision_tree_v1.py','scripts/run_stvg_joint_event_critic_v1.py',
 'scripts/score_stvg_temporal_local_global_mix128_v1.py','scripts/run_stvg_temporal_local_global_mix128_v1.py','scripts/score_stvg_where_when_event128_v1.py',
 'scripts/score_stvg_opd_p1_v1.py','vg_tta/tastvg_oracle_event5_v1.py','main_method/runtime.py','methods/stvg_referdino_opd_v1/runtime.py',
 'methods/CURRENT_METHOD.json','methods/RESEARCH_METHOD.json','methods/stvg_referdino_opd_v1/config.json']
 pins=dict(old['pins']);pins.update({f:sha(R/f)for f in own})
 for f in [OLD/'PLAN_LOCK.json',OLD/'GLOBAL_PREDICTION_BARRIER.json',OP/'RESULTS_PUBLIC.json',OLD/'COST.json',
 R/'results/stvg_error_state/2026-10-20/temporal_denselocal_opd128/RESULTS_PUBLIC.json']:
  pins[str(f.relative_to(R))]=sha(f)
 receipts={x['path']:x['sha256']for x in bar['files']};records=[]
 for r in old['records']:
  rr={k:r[k]for k in ['alias','dataset','parent','query_ordinal','physical_frame_ids8']}
  for name,base,want in [('dense',OLD,None),('stage',STAGE,r['native_stage_sha256']),('hires',HR,r['hires_sha256'])]:
   f=base/'predictions'/f"{r['alias']}.pt";h=sha(f);assert h==(want or receipts[str(f.relative_to(R))])
   assert h==read(f.with_suffix('.receipt.json'))['sha256'];pins[str(f.relative_to(R))]=h;rr[name+'_sha256']=h
  records.append(rr)
 assert len(records)==128 and len({r['parent']for r in records})==128
 p=dict(records=records,count=128,seeds=SEEDS,modes=MODES,arms=STATIC+FITS,steps=40,lr=.1,pairs_per_step=32,
 label='strict four-way sign; weakest signed log preference; sigmoid scale1; binary w=1/0',global_KL_coefficient=1.,
 student='per-query joint log residual on B-positive support',initial='exact saved Strong HiResAuxMix',
 source_checkpoint=old['source_checkpoint'],source_checkpoint_sha256=old['source_checkpoint_sha256'],pins=pins,
 scheduled_fits=1152,scheduled_updates=46080,CPU_seconds_cap=900,deadline_epoch=time.time()+14400,
 spatial_frozen=True,new_GPU=0,new_model=0,new_expert=0,new_network_training=0,not_confirmation=True,
 no_parameter_search=True,decision_method_separate_from_on_policy=True,created_epoch=time.time())
 put(D/'PLAN_LOCK.json',p)
 put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']})
 put(P/'PROPERTIES_PUBLIC.json',properties());print('locked full128 binary-weight preference OPD',flush=True)

def measure():
 import numpy as np,torch
 from main_method.runtime import resource_guard
 from methods.stvg_referdino_opd_v1.runtime import guard
 from vg_tta.stvg_temporal_strong_relative_preference_opd128_v1 import evidence,fit
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 sys.addaudithook(guard);sys.addaudithook(forbid_GT);torch.set_num_threads(4)
 p=read(D/'PLAN_LOCK.json');beg=time.perf_counter();files=[];times=[];fits=updates=first=0;mode_time={m:0. for m in MODES};success=False
 def check():resource_guard();assert time.perf_counter()-beg<880 and time.time()<p['deadline_epoch']
 def pins():assert all(sha(R/f)==h for f,h in p['pins'].items())
 pins();loading=time.perf_counter()-beg
 try:
  for rec in p['records']:
   check();tick=time.perf_counter();a=rec['alias'];old=load(OLD/'predictions'/f'{a}.pt');stage=load(STAGE/'predictions'/f'{a}.pt');hr=load(HR/'predictions'/f'{a}.pt')
   ev=evidence(old,stage,hr);B=old['outputs']['HiResAuxMix']['q']
   assert final_mbr(old['frame_ids'],B)['interval']==old['outputs']['HiResAuxMix']['interval']
   outputs={name:dict(q=old['outputs'][src]['q'],interval=old['outputs'][src]['interval'])for name,src in [('Native','Native'),('Strong','HiResAuxMix'),('TeacherDirect','DenseLocal'),('DenseLocalMix','DenseLocalMix')]}
   for mode in MODES:
    for seed in SEEDS:
     check();t=time.perf_counter();v=fit(B,ev,mode,seed,steps=p['steps'],lr=p['lr'],check=check);mode_time[mode]+=time.perf_counter()-t
     v['interval']=final_mbr(old['frame_ids'],v['q'])['interval'];outputs[mode+str(seed)]=v;fits+=1;updates+=v['actual_updates']
   for s in SEEDS:
    f,l=outputs[f'Fixed{s}'],outputs[f'Live{s}']
    for k in ['pairs','gradient','eta_after','uniforms','loss']:assert np.array_equal(f['trace'][0][k],l['trace'][0][k])
    first+=1
   files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,frame_ids=old['frame_ids'],trajectories=old['trajectories'],q0=B,evidence=ev,outputs=outputs,GT_read=False)))
   times.append(time.perf_counter()-tick);put(D/'STATUS.json',dict(stage='actual_OPD',done=len(files),fits=fits,updates=updates,pid=os.getpid(),epoch=time.time()),True)
   print('preference_OPD',a,round(time.perf_counter()-beg,2),flush=True)
   if len(files)==2:
    estimate=loading+128*max(times)+30;put(D/'PILOT2.json',dict(case_seconds=times,estimated_CPU_seconds=estimate,engineering_only=True,new_GT=False));assert estimate<900
   del old,stage,hr,ev,outputs;gc.collect()
  check();pins();assert fits==1152 and updates==46080 and first==384
  put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,first_Fixed_Live_exact=first));success=True
  print('sealed128 preference OPD',flush=True)
 finally:
  put(D/'COST.json',dict(success=success,CPU_prediction_and_updates_seconds=time.perf_counter()-beg,actual_fits=fits,actual_updates=updates,mode_fit_CPU_seconds=mode_time,
   scheduled_fits=1152,scheduled_updates=46080,first_Fixed_Live_exact=first,GPU_seconds=0,new_RGB=0,new_model_forward=0,new_DINO=0,new_ReferDINO=0,new_network_training=0,spatial_updates=0,
   upstream_cache_generation_not_free_deployment=True,Direct_pair_arithmetic_larger_than_Fixed_Live=True))

if __name__=='__main__':
 def timeout(signum,frame):raise TimeoutError('finite900s preference OPD stage')
 signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,timeout);signal.alarm(900)
 D.mkdir(parents=True,exist_ok=True);lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:globals()[sys.argv[1]]()
 except BaseException as e:put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partial_outputs_preserved=True));raise
 finally:lock.close()
