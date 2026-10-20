"""Actual existing temporal OPD fitting, only teacher replaced by pure DenseLocal."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import sys,time,fcntl,signal,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_denselocal_opd128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_denselocal_opd128'
OLD=R/'artifacts/stvg_temporal_local_motion_completion128_v1'
OP=R/'results/stvg_error_state/2026-10-20/temporal_local_motion_completion128'
SEEDS=[41,42,43];STATIC=['Native','Strong','TeacherDirect','DenseLocalMix']
FITS=['DirectJoint','DirectConditional']+[f'{m}{s}'for m in ['Fixed','Live']for s in SEEDS]

def forbid_GT(event,args):
 if event=='open'and args and isinstance(args[0],(str,bytes,Path))and str(args[0]).endswith('/data/hcstvg2_official_metadata/val_v2.json'):
  raise PermissionError('DenseLocal OPD inference and preaudit forbid GT')

def prepare():
 import torch
 from vg_tta.stvg_temporal_conditional_opd128_v1 import properties
 from vg_tta.stvg_temporal_strong_action_opd128_v1 import instrumentation_properties
 torch.set_num_threads(4);sys.addaudithook(forbid_GT)
 old=read(OLD/'PLAN_LOCK.json');bar=read(OLD/'GLOBAL_PREDICTION_BARRIER.json')
 assert bar['count']==128 and bar['weights_and_flags_unchanged']
 assert all(sha(R/f)==h for f,h in old['pins'].items())
 own=['protocols/stvg_temporal_denselocal_opd128_v1.md','scripts/run_stvg_temporal_denselocal_opd128_v1.py','scripts/score_stvg_temporal_denselocal_opd128_v1.py','scripts/publish_stvg_temporal_denselocal_opd128_v1.py',
 'vg_tta/stvg_temporal_conditional_opd128_v1.py','vg_tta/stvg_temporal_strong_action_opd128_v1.py','vg_tta/stvg_temporal_duration_condition128_v1.py',
 'scripts/score_stvg_temporal_conditional_opd128_v1.py','scripts/run_stvg_temporal_conditional_opd128_v1.py','scripts/run_stvg_temporal_strong_action_opd128_v1.py',
 'scripts/score_stvg_temporal_local_global_mix128_v1.py','scripts/run_stvg_temporal_local_global_mix128_v1.py',
 str((OLD/'PLAN_LOCK.json').relative_to(R)),str((OLD/'GLOBAL_PREDICTION_BARRIER.json').relative_to(R)),str((OP/'RESULTS_PUBLIC.json').relative_to(R)),
 str((OP/'COST_PUBLIC.json').relative_to(R)) if (OP/'COST_PUBLIC.json').exists()else str((OLD/'COST.json').relative_to(R)),
 'methods/CURRENT_METHOD.json','methods/RESEARCH_METHOD.json']
 pins=dict(old['pins']);pins.update({f:sha(R/f)for f in own})
 receipts={x['path']:x['sha256']for x in bar['files']};records=[]
 for r in old['records']:
  f=OLD/'predictions'/f"{r['alias']}.pt";rel=str(f.relative_to(R));assert sha(f)==receipts[rel]
  records.append(dict(r,dense_prediction_sha256=receipts[rel]));pins[rel]=receipts[rel]
 assert len(records)==128 and len({r['parent']for r in records})==128
 p=dict(records=records,count=128,seeds=SEEDS,arms=STATIC+FITS,primary='Mean_Live_minus_Mean_Strong',
 student_initial='exact saved HiResAuxMix Strong posterior',teacher='exact saved PURE DenseLocal final-layer posterior',
 source_checkpoint=old['source_checkpoint'],source_checkpoint_sha256=old['source_checkpoint_sha256'],
 steps=40,lr=.1,M=32,temporary_parameters=130,CPU_seconds_cap=900,deadline_epoch=time.time()+14400,
 pins=pins,created_epoch=time.time(),spatial_frozen=True,new_GPU=0,new_visual_calls=0,new_network_training=0,
 scheduled_fits=1024,scheduled_updates=40960,no_parameter_search=True,not_confirmation=True)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']})
 put(P/'PROPERTIES_PUBLIC.json',dict(existing_policy=properties(),unchanged_instrumentation=instrumentation_properties()))
 print('locked128 actual DenseLocal teacher OPD, original Strong initialization and fitter',flush=True)

def measure():
 import torch,numpy as np
 from main_method.runtime import resource_guard
 from methods.stvg_referdino_opd_v1.runtime import guard
 from vg_tta.stvg_temporal_strong_action_opd128_v1 import fit,summary
 from vg_tta.stvg_temporal_conditional_opd128_v1 import Policy
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 sys.addaudithook(guard);sys.addaudithook(forbid_GT);torch.set_num_threads(4)
 p=read(D/'PLAN_LOCK.json');beg=time.perf_counter();files=[];times=[];counts=dict(attempts=0,adam_steps=0,trace_steps=0);fits=updates=first=0;maxgrad=0.;partial=None;success=False
 def check():resource_guard();assert time.perf_counter()-beg+20<900 and time.time()<p['deadline_epoch']
 def pins():assert all(sha(R/f)==h for f,h in p['pins'].items())
 pins();loading=time.perf_counter()-beg
 try:
  for r in p['records']:
   check();tick=time.perf_counter();a=r['alias'];f=OLD/'predictions'/f'{a}.pt';assert sha(f)==r['dense_prediction_sha256'];old=load(f)
   B=old['outputs']['HiResAuxMix']['q'];T=old['outputs']['DenseLocal']['q'];assert B.shape==T.shape==(65,65)and np.array_equal(B>0,T>0)and not np.tril(B).any()
   policy=Policy(B,T);assert np.array_equal(policy.probability(),B)and final_mbr(old['frame_ids'],B)['interval']==old['outputs']['HiResAuxMix']['interval']
   outputs={name:dict(q=old['outputs'][src]['q'],interval=old['outputs'][src]['interval'])for name,src in [('Native','Native'),('Strong','HiResAuxMix'),('TeacherDirect','DenseLocal'),('DenseLocalMix','DenseLocalMix')]}
   partial=dict(alias=a,q0=B,teacher=T,outputs=outputs,inflight_fit=None,GT_read=False)
   for mode in ['DirectJoint','DirectConditional','Fixed','Live']:
    for seed in SEEDS if mode in ['Fixed','Live']else[41]:
     check();counts['attempts']+=1;partial['inflight_fit']=dict(mode=mode,seed=seed,trace=[],optimizer_steps=0)
     def observe(event,trace):
      counts['adam_steps'if event=='adam'else'trace_steps']+=1
      if event=='adam':partial['inflight_fit']['optimizer_steps']+=1
      else:partial['inflight_fit']['trace']=trace
     v=fit(B,T,mode,seed,steps=p['steps'],lr=p['lr'],observer=observe,check=check)
     partial['inflight_fit']=None;fits+=1;updates+=v['actual_updates'];maxgrad=max(maxgrad,v['max_manual_gradient_error'])
     v.update(mode=mode,seed=seed,interval=final_mbr(old['frame_ids'],v['q'])['interval'],fit_diagnostics=summary(B,T,v['q']))
     outputs[mode+str(seed)if mode in ['Fixed','Live']else mode]=v
   for seed in SEEDS:
    x,y=outputs[f'Fixed{seed}'],outputs[f'Live{seed}'];assert len(x['trace'])==len(y['trace'])
    if x['trace']:
     for k in ['sampled_pairs','gradient','eta_after','uniforms','teacher_conditional_digest']:assert np.array_equal(x['trace'][0][k],y['trace'][0][k])
    first+=1
   assert set(outputs)==set(p['arms'])
   files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,frame_ids=old['frame_ids'],trajectories=old['trajectories'],q0=B,teacher=T,outputs=outputs,GT_read=False,all_network_weights_untouched=True)))
   times.append(time.perf_counter()-tick);put(D/'STATUS.json',dict(stage='actual_OPD',done=len(files),fits=fits,updates=updates,pid=os.getpid(),epoch=time.time()),True)
   print('DenseLocal_OPD',a,round(time.perf_counter()-beg,2),flush=True)
   if len(files)==2:
    estimate=loading+128*max(times)+30;put(D/'PILOT2.json',dict(case_seconds=times,estimated_CPU_seconds=estimate,engineering_timing_only=True,new_GT=False));assert estimate<900
   partial=None;del old,outputs,B,T,policy;gc.collect()
  check();pins();assert counts['adam_steps']==counts['trace_steps']==updates and fits==1024 and first==384;check()
  put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,first_Fixed_Live_exact=first));success=True
  print('sealed128 actual DenseLocal OPD',flush=True)
 finally:
  if not success and partial is not None:save(D/'partials'/f"{partial['alias']}_{time.time_ns()}.pt",partial)
  put(D/'COST.json',dict(success=success,CPU_prediction_and_updates_seconds=time.perf_counter()-beg,actual_fits=fits,fit_attempts=counts['attempts'],actual_updates=counts['adam_steps'],completed_trace_steps=counts['trace_steps'],completed_fit_updates=updates,scheduled_fits=1024,scheduled_updates=40960,
  first_Fixed_Live_exact=first,max_manual_gradient_error=maxgrad,temporary_parameters_per_fit=130,GPU_seconds=0,new_RGB=0,new_model_forward=0,new_Source_encoder=0,new_DINO=0,new_ReferDINO=0,new_network_training=0,spatial_updates=0,upstream_cache_generation_not_free_deployment=True))

if __name__=='__main__':
 def timeout(signum,frame):raise TimeoutError('finite900s DenseLocal temporal OPD stage')
 signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,timeout);signal.alarm(900)
 D.mkdir(parents=True,exist_ok=True);lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:globals()[sys.argv[1]]()
 except BaseException as e:put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partial_outputs_preserved=True));raise
 finally:lock.close()
