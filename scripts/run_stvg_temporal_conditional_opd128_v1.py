"""Actual structured temporal OPD, CPU/cache only, all states reset per video."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import sys,time,shutil
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_conditional_opd128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_conditional_opd128';HR=R/'artifacts/stvg_temporal_hires128_v1'

def prepare():
 import torch
 torch.set_num_threads(4)
 from vg_tta.stvg_temporal_conditional_opd128_v1 import properties
 records=[dict(r,teacher_prediction_sha256=sha(HR/'predictions'/f"{r['alias']}.pt"))for r in read(HR/'PLAN_LOCK.json')['records']]
 own=['protocols/stvg_temporal_conditional_opd128_v1.md','vg_tta/stvg_temporal_conditional_opd128_v1.py','scripts/run_stvg_temporal_conditional_opd128_v1.py','scripts/score_stvg_temporal_conditional_opd128_v1.py','scripts/publish_stvg_temporal_conditional_opd128_v1.py','methods/stvg_referdino_opd_v1/config.json']
 p=dict(records=records,count=128,seeds=[41,42,43],arms=['Native','TeacherDirect','DirectJoint','DirectConditional']+[f'{m}{s}'for m in ['Fixed','Live']for s in [41,42,43]],steps=40,lr=.1,candidate_intervals_per_update=32,CPU_seconds_cap=900,deadline_epoch=time.time()+7200,created_epoch=time.time(),pins={f:sha(R/f)for f in own},teacher='frozen .5Source224+.5Source448 posterior',temporary_parameters=130,spatial_frozen=True,new_GPU=0,new_visual_calls=0,new_network_training=0)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','deadline_epoch','created_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked128',flush=True)

def measure():
 import torch,numpy as np,psutil,gc
 from methods.stvg_referdino_opd_v1.runtime import guard
 from vg_tta.stvg_temporal_conditional_opd128_v1 import fit
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 sys.addaudithook(guard);torch.set_num_threads(4);p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());beg=time.perf_counter();files=[];times=[];updates=0;fits=0;first_exact=0;maxgrad=0.
 def check():
  assert time.time()<p['deadline_epoch']and time.perf_counter()-beg+20<p['CPU_seconds_cap'];assert psutil.virtual_memory().available>6*2**30 and shutil.disk_usage(R).free>30*2**30
 try:
  for r in p['records']:
   check();tick=time.perf_counter();a=r['alias'];assert sha(HR/'predictions'/f'{a}.pt')==r['teacher_prediction_sha256'];z=load(HR/'predictions'/f'{a}.pt');q0=z['outputs']['Native']['q'];target=z['outputs']['HiResMix']['q'];outputs={key:dict(q=q,interval=final_mbr(z['frame_ids'],q)['interval'])for key,q in [('Native',q0),('TeacherDirect',target)]}
   for mode in ['DirectJoint','DirectConditional','Fixed','Live']:
    for seed in p['seeds']if mode in ['Fixed','Live']else [41]:
     check();v=fit(q0,target,mode,seed,steps=p['steps'],lr=p['lr']);v['interval']=final_mbr(z['frame_ids'],v['q'])['interval'];v['mode']=mode;v['seed']=seed;outputs[mode+str(seed)if mode in ['Fixed','Live']else mode]=v;fits+=1;updates+=v['actual_updates'];maxgrad=max(maxgrad,v['max_manual_gradient_error'])
   for seed in p['seeds']:
    f,l=outputs['Fixed'+str(seed)],outputs['Live'+str(seed)]
    if f['trace']:
     for k in ['sampled_pairs','gradient','eta_after','uniforms']:assert np.array_equal(f['trace'][0][k],l['trace'][0][k])
    else:assert not l['trace']and np.array_equal(f['q'],l['q'])
    first_exact+=1
   assert outputs['Native']['interval']==z['outputs']['Native']['interval']and outputs['TeacherDirect']['interval']==z['outputs']['HiResMix']['interval'];files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=z['frame_ids'],trajectories=z['trajectories'],q0=q0,teacher=target,outputs=outputs,GT_read=False,all_network_weights_untouched=True)));times.append(time.perf_counter()-tick);put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time()),True);print('conditional_opd',a,round(time.perf_counter()-beg,2),flush=True)
   if len(files)==2:
    estimate=max(times)*128+10;put(D/'PILOT2.json',dict(case_seconds=times,estimated_CPU_seconds=estimate,GT_read=False));assert estimate<900
   del z,outputs,q0,target;gc.collect()
  put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,first_Fixed_Live_exact=first_exact));print('sealed128',flush=True)
 finally:
  put(D/'COST.json',dict(CPU_seconds=time.perf_counter()-beg,fits=fits,actual_updates=updates,scheduled_updates=40*1024,temporary_parameters_per_fit=130,first_Fixed_Live_exact=first_exact,max_manual_gradient_error=maxgrad,new_GPU=0,new_visual_calls=0,new_experts=0,network_training=0,spatial_updates=0));

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
