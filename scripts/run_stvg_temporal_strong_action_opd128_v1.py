"""Actual matched conditional temporal OPD from a frozen strong starting prior."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import sys,time,fcntl
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_strong_action_opd128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_strong_action_opd128';OLD=R/'artifacts/stvg_temporal_foreground_opd128_v1';HR=R/'artifacts/stvg_temporal_hires128_v1';STAGE=R/'artifacts/stvg_temporal_native_stage128_v1';WRONG=R/'artifacts/stvg_temporal_foreground_info128_v1/revision001/controls';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1'
CONDITIONS=['H','N','WrongH','ShiftH'];SEEDS=[41,42,43]
STATIC=['Native','B']+[f'Teacher{k}'for k in CONDITIONS]
FIT_ARMS=[f'{k}_{m}'+(str(s)if m in ['Fixed','Live']else'')for k in CONDITIONS for m in (['DirectJoint','DirectConditional','Fixed','Live']if k in ['H','N']else['Live'])for s in (SEEDS if m in ['Fixed','Live']else[41])]

def prepare():
 import torch,numpy as np
 from vg_tta.stvg_temporal_conditional_opd128_v1 import properties
 from vg_tta.stvg_temporal_duration_condition128_v1 import properties as p2
 from vg_tta.stvg_temporal_strong_action_opd128_v1 import instrumentation_properties
 torch.set_num_threads(4);records=[]
 for r in read(R/'artifacts/stvg_temporal_duration_condition128_v1/PLAN_LOCK.json')['records']:
  r=dict(r);w=load(WRONG/f"{r['alias']}.pt");r.update(high_sha256=sha(HR/'predictions'/f"{r['alias']}.pt"),raw_sha256=sha(OLD/'predictions'/f"{r['alias']}.pt"),base_sha256=sha(STAGE/'predictions'/f"{r['alias']}.pt"),wrong_mapping_sha256=sha(WRONG/f"{r['alias']}.pt"),donor=w['donor']);records.append(r)
 assert len(records)==128 and len({r['parent']for r in records})==128 and len({r['alias']for r in records})==128;aliases=[r['alias']for r in records];assert all(r['donor']==aliases[(k+1)%128]for k,r in enumerate(records))
 own=['protocols/stvg_temporal_strong_action_opd128_v1.md','vg_tta/stvg_temporal_strong_action_opd128_v1.py','vg_tta/stvg_temporal_conditional_opd128_v1.py','vg_tta/stvg_temporal_duration_condition128_v1.py','vg_tta/stvg_posterior_decision_tree_v1.py','scripts/run_stvg_temporal_strong_action_opd128_v1.py','scripts/score_stvg_temporal_strong_action_opd128_v1.py','scripts/publish_stvg_temporal_strong_action_opd128_v1.py','scripts/score_stvg_temporal_foreground_opd128_v1.py','scripts/run_stvg_temporal_foreground_opd128_v1.py','methods/stvg_referdino_opd_v1/config.json','methods/stvg_referdino_opd_v1/runtime.py','main_method/runtime.py','scripts/run_stvg_joint_event_critic_v1.py','scripts/score_stvg_where_when_event128_v1.py','scripts/score_stvg_opd_p1_v1.py','scripts/stvg_opd_paper_common_v1.py','scripts/decota_matrix_common_v1.py','scripts/decota_opd_tuning_common_v1.py','vg_tta/tastvg_oracle_event5_v1.py','vg_tta/tastvg_paper48_metrics_v1.py','vg_tta/tastvg_paper48_hc2_metrics_v1.py','external/TA-STVG/utils/box_utils.py','external/TA-STVG/engine/evaluate.py','external/TA-STVG/datasets/evaluation/vidstg_eval.py','external/TA-STVG/datasets/evaluation/hcstvg_eval.py','artifacts/decota_paper_experiments_v1/hc2/PLAN.json','artifacts/decota_paper_experiments_v1/DESIGN_LOCK.json']
 p=dict(records=records,count=128,seeds=SEEDS,arms=STATIC+FIT_ARMS+['NoActionLive41'],primary='Mean_H_Live',student_initial='B=HiResAuxMix',source_checkpoint='checkpoints/TASTVG_VidSTG.pth',source_checkpoint_sha256='5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83',steps=40,lr=.1,M=32,temporary_parameters=130,CPU_seconds_cap=900,pins={f:sha(R/f)for f in own},created_epoch=time.time(),new_GPU=0,new_experts=0,new_network_training=0,spatial_frozen=True,primary_aggregation='matching time/spatial seeds within parent; secondary all9 not selected',not_confirmation=True)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch']});put(P/'PROPERTIES_PUBLIC.json',dict(policy=properties(),node_condition=p2(),instrumentation=instrumentation_properties()));print('locked128 actual Strong Action OPD')

def measure():
 import torch,numpy as np,gc
 from methods.stvg_referdino_opd_v1.runtime import guard
 from main_method.runtime import resource_guard
 from vg_tta.stvg_temporal_strong_action_opd128_v1 import high_curve,wrong_curve,targets,summary,fit
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 sys.addaudithook(guard)
 def forbid_actual_GT(event,args):
  if event=='open'and args and isinstance(args[0],(str,bytes,Path))and str(args[0]).endswith('/data/hcstvg2_official_metadata/val_v2.json'):raise PermissionError('measure forbids actual HC GT')
 sys.addaudithook(forbid_actual_GT);torch.set_num_threads(4);p=read(D/'PLAN_LOCK.json');beg=time.perf_counter();files=[];times=[];fits=updates=first=0;counts=dict(attempts=0,adam_steps=0,trace_steps=0);maxgrad=0.;partial=None;success=False;bank={}
 def check():
  resource_guard();assert time.perf_counter()-beg+20<900
 def pins():assert all(sha(R/f)==h for f,h in p['pins'].items())
 pins()
 try:
  for r in p['records']:
   check();a=r['alias']
   for path,h in [(HR/'predictions'/f'{a}.pt',r['high_sha256']),(OLD/'predictions'/f'{a}.pt',r['raw_sha256']),(STAGE/'predictions'/f'{a}.pt',r['base_sha256']),(WRONG/f'{a}.pt',r['wrong_mapping_sha256'])]:assert sha(path)==h
   h=load(HR/'predictions'/f'{a}.pt');n=load(OLD/'predictions'/f'{a}.pt');assert np.array_equal(h['frame_ids'],n['frame_ids']);bank[a]=dict(ids=h['frame_ids'],H=high_curve(h['observation']),N=np.asarray(n['Action'],dtype=np.float64))
  loading=time.perf_counter()-beg
  for r in p['records']:
   check();tick=time.perf_counter();a=r['alias'];old=load(OLD/'predictions'/f'{a}.pt');b=load(STAGE/'predictions'/f'{a}.pt');B=b['outputs']['HiResAuxMix']['q'];v=bank[a];donor=bank[r['donor']];w,wm=wrong_curve(v['ids'],v['H'],donor['ids'],donor['H']);curves=dict(H=v['H'],N=v['N'],WrongH=w,ShiftH=np.roll(v['H'],32));tt=targets(B,curves)
   outputs={k:dict(q=q,interval=final_mbr(v['ids'],q)['interval'])for k,q in [('Native',old['q0']),('B',B)]+[('Teacher'+k,t)for k,t in tt.items()]};partial=dict(alias=a,query_ordinal=r['query_ordinal'],q0=B,teachers=tt,curves=curves,outputs=outputs,GT_read=False,completed_outputs_only=True)
   counts['attempts']+=1;noop=fit(B,B,'Live',41,steps=40,lr=.1);fits+=1;assert noop['actual_updates']==0 and np.array_equal(noop['q'],B);noop.update(mode='Live',seed=41,teacher='B',interval=outputs['B']['interval']);outputs['NoActionLive41']=noop
   for k in CONDITIONS:
    for mode in (['DirectJoint','DirectConditional','Fixed','Live']if k in ['H','N']else['Live']):
     for seed in SEEDS if mode in ['Fixed','Live']else [41]:
      check();counts['attempts']+=1;partial['inflight_fit']=dict(teacher=k,mode=mode,seed=seed,trace=[],optimizer_steps=0)
      def observe(event,trace):
       counts['adam_steps'if event=='adam'else'trace_steps']+=1
       if event=='adam':partial['inflight_fit']['optimizer_steps']+=1
       else:partial['inflight_fit']['trace']=trace
      f=fit(B,tt[k],mode,seed,steps=40,lr=.1,observer=observe,check=check);partial['inflight_fit']=None;fits+=1;updates+=f['actual_updates'];maxgrad=max(maxgrad,f['max_manual_gradient_error']);f.update(mode=mode,seed=seed,teacher=k,interval=final_mbr(v['ids'],f['q'])['interval'],fit_diagnostics=summary(B,tt[k],f['q']));outputs[k+'_'+mode+(str(seed)if mode in ['Fixed','Live']else'')]=f
   for k in ['H','N']:
    for seed in SEEDS:
     f,l=outputs[f'{k}_Fixed{seed}'],outputs[f'{k}_Live{seed}'];assert len(f['trace'])==len(l['trace'])
     if f['trace']:
      for key in ['sampled_pairs','gradient','eta_after','uniforms']:assert np.array_equal(f['trace'][0][key],l['trace'][0][key])
     first+=1
   assert set(outputs)==set(p['arms'])and outputs['Native']['interval']==old['outputs']['Native']['interval']and outputs['B']['interval']==b['outputs']['HiResAuxMix']['interval'];assert all(np.array_equal(x,b['trajectories'][k])for k,x in old['trajectories'].items())
   diag=dict(H_vs_N_curve_RMS=float(np.sqrt(np.mean((v['H']-v['N'])**2))),H_vs_N_target_TV=float(abs(tt['H']-tt['N']).sum()/2),wrong=wm)
   files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=v['ids'],trajectories=old['trajectories'],q0=B,teachers=tt,curves=curves,donor=r['donor'],outputs=outputs,diagnostics=diag,GT_read=False)));times.append(time.perf_counter()-tick);put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time()),True);print('strong_action_OPD',a,round(time.perf_counter()-beg,2),flush=True)
   if len(files)==2:
    estimate=loading+128*max(times)+30;put(D/'PILOT2.json',dict(case_seconds=times,cache_loading=loading,estimated_CPU_seconds=estimate,GT_read=False));assert estimate<900
   partial=None;del old,b,outputs,tt;gc.collect()
  check();pins();assert len(files)==128 and counts['adam_steps']==counts['trace_steps']==updates;put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,first_Fixed_Live_exact=first));success=True;print('sealed128 complete actual transfer matrix')
 finally:
  if not success and partial is not None:save(D/'partials'/f"{partial['alias']}_{time.time_ns()}.pt",partial)
  put(D/'COST.json',dict(CPU_prediction_and_updates_seconds=time.perf_counter()-beg,fits=fits,fit_attempts=counts['attempts'],actual_updates=counts['adam_steps'],completed_fit_updates=updates,completed_trace_steps=counts['trace_steps'],nominal_including_noop_steps=2944*40,scheduled_nontrivial_updates=112640,NoAction_calls=128,temporary_parameters=130,first_Fixed_Live_exact=first,max_manual_gradient_error=maxgrad,GPU_seconds=0,new_RGB=0,new_Source_encoder=0,new_DINO=0,new_ReferDINO=0,network_training=0,spatial_updates=0,upstream448_evidence_is_not_deployment_free=True))

if __name__=='__main__':
 D.mkdir(parents=True,exist_ok=True);lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:globals()[sys.argv[1]]()
 except BaseException as e:put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),preserved_completed_outputs=True));raise
 finally:lock.close()
