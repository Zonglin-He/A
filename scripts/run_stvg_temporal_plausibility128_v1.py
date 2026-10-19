"""One fixed cache-only plausibility factor, no model or annotation access."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,fcntl
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_plausibility128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_plausibility128';OLD=R/'artifacts/stvg_temporal_motion_counterfactual128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1'

def prepare():
 from vg_tta.stvg_temporal_plausibility128_v1 import properties
 assert read(OLD/'FINAL_IMMUTABILITY_AUDIT.json')['status']=='pass'
 records=[dict(r,counterfactual_sha256=sha(OLD/'predictions'/f"{r['alias']}.pt"))for r in read(OLD/'PLAN_LOCK.json')['records']]
 assert len(records)==128 and len({r['parent']for r in records})==128 and len({r['alias']for r in records})==128
 own=['protocols/stvg_temporal_plausibility128_v1.md','vg_tta/stvg_temporal_plausibility128_v1.py','scripts/run_stvg_temporal_plausibility128_v1.py','scripts/score_stvg_temporal_plausibility128_v1.py','scripts/publish_stvg_temporal_plausibility128_v1.py','methods/stvg_referdino_opd_v1/config.json','vg_tta/stvg_posterior_decision_tree_v1.py']
 own+=['scripts/run_stvg_joint_event_critic_v1.py','scripts/score_stvg_where_when_event128_v1.py','main_method/runtime.py','scripts/score_stvg_opd_p1_v1.py','scripts/stvg_opd_paper_common_v1.py','scripts/decota_matrix_common_v1.py','scripts/decota_opd_tuning_common_v1.py','vg_tta/tastvg_oracle_event5_v1.py','vg_tta/tastvg_paper48_metrics_v1.py','vg_tta/tastvg_paper48_hc2_metrics_v1.py','external/TA-STVG/utils/box_utils.py','external/TA-STVG/engine/evaluate.py','external/TA-STVG/datasets/evaluation/vidstg_eval.py','external/TA-STVG/datasets/evaluation/hcstvg_eval.py','artifacts/decota_paper_experiments_v1/hc2/PLAN.json','artifacts/decota_paper_experiments_v1/DESIGN_LOCK.json','results/stvg_error_state/2026-10-20/temporal_motion_counterfactual128/RESULTS_PUBLIC.json']
 p=dict(records=records,count=128,seeds=[41,42,43],arms=['Native','MeanMotion','PriorSquare','Contrast','HiResAuxMix','NativeHead','SquareHead','ContrastHead'],primary='ContrastHead',beta=.1,head_defined_by='Native only, >= beta max including ties',MBR_action_universe_unchanged=True,CPU_seconds_cap=900,pins={f:sha(R/f)for f in own},spatial_frozen=True,new_models=0,new_experts=0,new_training=0,new_student_updates=0,not_OPD=True,created_epoch=time.time());put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked128 beta.1 pure cache')

def measure():
 import numpy as np
 from main_method.runtime import resource_guard
 from vg_tta.stvg_temporal_plausibility128_v1 import heads
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());lock=open(D/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);beg=time.perf_counter();files=[]
 try:
  for r in p['records']:
   resource_guard();assert time.perf_counter()-beg<900;a=r['alias'];assert sha(OLD/'predictions'/f'{a}.pt')==r['counterfactual_sha256'];old=load(OLD/'predictions'/f'{a}.pt');h,qs=heads(old['outputs']);out=dict(old['outputs']);indices={}
   for name,q in qs.items():dec=final_mbr(old['frame_ids'],q);out[name]=dict(q=q,interval=dec['interval']);indices[name]=dec['indices']
   diagnostics=dict(H_count=int(h.sum()),native_positive_count=int((old['outputs']['Native']['q']>0).sum()),excluded_mass={k:float(old['outputs'][k]['q'][~h].sum())for k in ['Native','PriorSquare','Contrast']},MBR_outside_H={k:not bool(h[tuple(v)])for k,v in indices.items()})
   files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=old['frame_ids'],outputs=out,trajectories=old['trajectories'],H=h,diagnostics=diagnostics,GT_read=False)));put(D/'STATUS.json',dict(stage='measure',done=len(files),epoch=time.time(),pid=os.getpid()),True)
  resource_guard();assert len(files)==128 and time.perf_counter()-beg<900 and all(sha(R/f)==h for f,h in p['pins'].items());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True));print('sealed128 beta.1 no models')
 finally:
  put(D/'COST.json',dict(CPU_prediction_seconds=time.perf_counter()-beg,GPU_seconds=0,new_models=0,new_RGB=0,new_Source_encoder=0,new_DINO=0,new_ReferDINO=0,network_training=0,spatial_updates=0,new_student_updates=0));lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
