"""CPU-only joint-decision probe; original spatial coordinates never change."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
from pathlib import Path
import sys,time,gc,fcntl,shutil
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_joint_risk128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_joint_risk128';MEM=R/'artifacts/stvg_temporal_object_memory128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1';STAGE=R/'artifacts/stvg_temporal_native_stage128_v1'
PRIORS=['N','H','A'];ARMS=[p+'_'+k for p in PRIORS for k in ['Time','Risk','Shift','Flat','Target','FlatTarget']]


def prepare():
 from vg_tta.stvg_temporal_joint_risk128_v1 import properties
 records=[dict(r,stage_sha256=sha(STAGE/'predictions'/f"{r['alias']}.pt"))for r in read(MEM/'PLAN_LOCK.json')['records']]
 own=['protocols/stvg_temporal_joint_risk128_v1.md','vg_tta/stvg_temporal_joint_risk128_v1.py','scripts/run_stvg_temporal_joint_risk128_v1.py','scripts/score_stvg_temporal_joint_risk128_v1.py','scripts/publish_stvg_temporal_joint_risk128_v1.py','results/stvg_error_state/2026-10-20/temporal_joint_risk128/independent_risk_audit.py','vg_tta/stvg_posterior_decision_tree_v1.py','methods/stvg_referdino_opd_v1/config.json']
 p=dict(records=records,count=128,seeds=[41,42,43],arms=ARMS,priors=PRIORS,primary='A_Risk',eligible=['N_Risk','H_Risk','A_Risk'],CPU_seconds_cap=900,GPU_seconds=0,created_epoch=time.time(),deadline_epoch=time.time()+14400,pins={f:sha(R/f)for f in own},spatial_frozen=True,new_experts=0,new_training=0,new_OPD=False)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked128 joint tube utility',flush=True)


def measure():
 import torch,psutil
 from methods.stvg_referdino_opd_v1.runtime import setup
 from vg_tta.stvg_temporal_joint_risk128_v1 import quality,decisions
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 setup();torch.set_num_threads(4);p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);beg=time.perf_counter();files=[];times=[];computations=flatchecks=0
 def guard():
  assert time.time()<p['deadline_epoch']and time.perf_counter()-beg+30<900;assert psutil.virtual_memory().available>6*2**30 and shutil.disk_usage(R).free>30*2**30
 try:
  for r in p['records']:
   guard();tick=time.perf_counter();a=r['alias']
   for path,h in [(STAGE/'predictions'/f'{a}.pt',r['stage_sha256']),(PRE/'inputs'/f'{a}.pt',r['input_sha256']),(FIN/'Live/predictions'/f'{a}.pt',r['spatial_sha256']),(PRE/'teacher0'/f'{a}.pt',r['teacher_sha256'])]:assert sha(path)==h
   inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');teacher=load(PRE/'teacher0'/f'{a}.pt');stage=load(STAGE/'predictions'/f'{a}.pt');ids=inp['frame_ids'];fc=r['frame_count'];ti=np.asarray(ids)[r['positions8']].tolist();priors={k:stage['outputs'][v]['q']for k,v in zip(PRIORS,['Native','HiResMix','HiResAuxMix'])};trajectories=dict(E=np.asarray(inp['E']),**{f'Live{s}':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']});base={k:final_mbr(ids,q)for k,q in priors.items()};assert base['N']['interval']==inp['interval'];outputs={};curves={};riskmeta={}
   for space,boxes in trajectories.items():
    guard();curve,info=quality(ids,boxes,ti,teacher['native']['boxes'].numpy(),fc);shift=np.roll(curve,fc//2);flat=np.full(fc,float(curve.mean()));curves[space]=dict(curve=curve,shift=shift,flat=flat,meta=info);out={};rm={}
    for name,q in priors.items():
     real=decisions(ids,q,curve);wrong=decisions(ids,q,shift);constant=decisions(ids,q,flat);computations+=3;assert constant['interval']==base[name]['interval'];flatchecks+=1
     out[name+'_Time']=dict(interval=base[name]['interval']);out[name+'_Risk']=dict(interval=real['interval']);out[name+'_Shift']=dict(interval=wrong['interval']);out[name+'_Flat']=dict(interval=constant['interval']);out[name+'_Target']=dict(interval=real['target_interval']);out[name+'_FlatTarget']=dict(interval=constant['target_interval']);rm[name]=dict(real_utility=real['utility'],shift_utility=wrong['utility'],flat_utility=constant['utility'],target=real['target'],flat_target=constant['target'],quality_flat_noop=real['quality_flat_noop'])
    assert set(out)==set(ARMS);outputs[space]=out;riskmeta[space]=rm
   files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,frame_count=fc,positions8=r['positions8'],teacher_boxes=teacher['native']['boxes'].numpy(),trajectories=trajectories,priors=priors,outputs=outputs,quality_curves=curves,risk=riskmeta,GT_read=False,new_OPD=False,spatial_updated=False)));times.append(time.perf_counter()-tick);put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time()),True);print('joint_risk',a,round(time.perf_counter()-beg,2),flush=True)
   if len(times)==2:
    estimate=128*max(times)+60;put(D/'PILOT2.json',dict(case_seconds=times,estimated_CPU_seconds=estimate,GPU_seconds=0,GT_read=False));assert estimate<900
   del inp,sp,teacher,stage,outputs,curves,riskmeta,trajectories;gc.collect()
  put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,spatial_unchanged=True));print('sealed128',flush=True)
 finally:
  put(D/'COST.json',dict(CPU_seconds=time.perf_counter()-beg,GPU_seconds=0,joint_utility_computations=computations,flat_original_time_exact_checks=flatchecks,new_Source_encoder=0,new_RGB=0,new_DINO=0,new_ReferDINO=0,new_training=0,spatial_updates=0,new_OPD_updates=0));lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
