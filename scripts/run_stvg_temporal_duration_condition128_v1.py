"""Node-count marginal preserving cache-only assimilation, no model or annotation access."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,fcntl
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_duration_condition128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_duration_condition128';OLD=R/'artifacts/stvg_temporal_foreground_opd128_v1';STRONG=R/'artifacts/stvg_temporal_native_stage128_v1';WRONG=R/'artifacts/stvg_temporal_foreground_info128_v1/revision001/controls';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1'

def prepare():
 from vg_tta.stvg_temporal_duration_condition128_v1 import properties
 import numpy as np
 records=[dict(r,raw_prediction_sha256=sha(OLD/'predictions'/f"{r['alias']}.pt"),strong_prediction_sha256=sha(STRONG/'predictions'/f"{r['alias']}.pt"),wrong_control_sha256=sha(WRONG/f"{r['alias']}.pt"))for r in read(OLD/'PLAN_LOCK.json')['records']]
 assert len(records)==128 and len({r['parent']for r in records})==128 and len({r['alias']for r in records})==128
 own=['protocols/stvg_temporal_duration_condition128_v1.md','vg_tta/stvg_temporal_duration_condition128_v1.py','scripts/run_stvg_temporal_duration_condition128_v1.py','scripts/score_stvg_temporal_duration_condition128_v1.py','scripts/publish_stvg_temporal_duration_condition128_v1.py','methods/stvg_referdino_opd_v1/config.json','vg_tta/stvg_posterior_decision_tree_v1.py']
 own+=['scripts/run_stvg_joint_event_critic_v1.py','scripts/score_stvg_where_when_event128_v1.py','main_method/runtime.py','methods/stvg_referdino_opd_v1/runtime.py','scripts/score_stvg_opd_p1_v1.py','scripts/stvg_opd_paper_common_v1.py','scripts/decota_matrix_common_v1.py','scripts/decota_opd_tuning_common_v1.py','vg_tta/tastvg_oracle_event5_v1.py','vg_tta/tastvg_paper48_metrics_v1.py','vg_tta/tastvg_paper48_hc2_metrics_v1.py','external/TA-STVG/utils/box_utils.py','external/TA-STVG/engine/evaluate.py','external/TA-STVG/datasets/evaluation/vidstg_eval.py','external/TA-STVG/datasets/evaluation/hcstvg_eval.py','artifacts/decota_paper_experiments_v1/hc2/PLAN.json','artifacts/decota_paper_experiments_v1/DESIGN_LOCK.json']
 p=dict(source_checkpoint='checkpoints/TASTVG_VidSTG.pth',source_checkpoint_sha256='5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83',records=records,count=128,seeds=[41,42,43],arms=['Native','RawTeacher','GlobalFuse','DurationConditional','DurationShift32','DurationWrongParent','HiResAuxMix'],primary='DurationConditional',coefficient=1,shift=32,MBR_action_universe_unchanged=True,CPU_seconds_cap=900,pins={f:sha(R/f)for f in own},spatial_frozen=True,new_models=0,new_experts=0,new_training=0,new_student_updates=0,not_OPD=True,created_epoch=time.time())
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties())
 from vg_tta.stvg_temporal_duration_condition128_v1 import marginal
 tv=[];sizes=[];shifts=[]
 for r in records:
  z=load(OLD/'predictions'/f"{r['alias']}.pt");a=marginal(z['q0']);b=marginal(z['teacher']);k=np.arange(len(a));tv.append(float(abs(a-b).sum()/2));sizes.append([float(k@a),float(k@b)]);shifts.append(abs(sizes[-1][0]-sizes[-1][1]))
 put(P/'OLD_CACHE_NO_GT_DIAGNOSTIC_PUBLIC.json',dict(parents=128,GT_read=False,new_teacher_or_prediction=False,duration_is_included_node_count=True,mean_TV=float(np.mean(tv)),TV_above_point3=int((np.array(tv)>.3).sum()),mean_expected_nodes=np.mean(sizes,0).tolist(),mean_absolute_expected_node_shift=float(np.mean(shifts))))
 print('locked128 duration-marginal conditioning')

def measure():
 import numpy as np
 from main_method.runtime import resource_guard
 from methods.stvg_referdino_opd_v1.runtime import guard
 from vg_tta.stvg_temporal_duration_condition128_v1 import posterior,marginal,energy
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 sys.addaudithook(guard);p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());lock=open(D/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);beg=time.perf_counter();files=[]
 try:
  for r in p['records']:
   resource_guard();assert time.perf_counter()-beg<900;a=r['alias'];assert sha(OLD/'predictions'/f'{a}.pt')==r['raw_prediction_sha256']and sha(STRONG/'predictions'/f'{a}.pt')==r['strong_prediction_sha256'];old=load(OLD/'predictions'/f'{a}.pt');strong=load(STRONG/'predictions'/f'{a}.pt');assert sha(WRONG/f'{a}.pt')==r['wrong_control_sha256'];wrong=load(WRONG/f'{a}.pt');assert np.array_equal(old['frame_ids'],strong['frame_ids']);q0=old['q0'];x=np.asarray(old['Action'],dtype=np.float64)
   out={k:dict(q=q,interval=final_mbr(old['frame_ids'],q)['interval'])for k,q in [('Native',q0),('RawTeacher',old['teacher']),('GlobalFuse',posterior(q0,x,False)),('DurationConditional',posterior(q0,x,True)),('DurationShift32',posterior(q0,np.roll(x,32),True)),('DurationWrongParent',posterior(q0,wrong['field'],True)),('HiResAuxMix',strong['outputs']['HiResAuxMix']['q'])]}
   for name,oldname in [('Native','Native'),('RawTeacher','TeacherDirect')]:assert out[name]['interval']==old['outputs'][oldname]['interval']and np.array_equal(out[name]['q'],old['outputs'][oldname]['q'])
   assert out['HiResAuxMix']['interval']==strong['outputs']['HiResAuxMix']['interval']
   ms={k:marginal(v['q'])for k,v in out.items()};diagnostics=dict(marginal_TV_to_Native={k:float(abs(v-ms['Native']).sum()/2)for k,v in ms.items()},raw_curve_std=float(x.std()),expected_nodes={k:float(np.arange(len(v))@v)for k,v in ms.items()})
   files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=old['frame_ids'],outputs=out,trajectories=old['trajectories'],Action=x,WrongAction=np.asarray(wrong['field'],dtype=np.float64),energy=energy(x),marginals=ms,diagnostics=diagnostics,GT_read=False)));put(D/'STATUS.json',dict(stage='measure',done=len(files),epoch=time.time(),pid=os.getpid()),True)
  resource_guard();assert len(files)==128 and time.perf_counter()-beg<900 and all(sha(R/f)==h for f,h in p['pins'].items());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True));print('sealed128 duration conditioning, no model calls')
 finally:
  put(D/'COST.json',dict(CPU_prediction_seconds=time.perf_counter()-beg,GPU_seconds=0,new_models=0,new_RGB=0,new_Source_encoder=0,new_DINO=0,new_ReferDINO=0,network_training=0,spatial_updates=0,new_student_updates=0));lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
