"""Finite cached native-pass and auxiliary head knowledge probe, no new pixels."""
from pathlib import Path
import sys,os,time,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_native_stage128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_native_stage128';MEM=R/'artifacts/stvg_temporal_object_memory128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1';HR=R/'artifacts/stvg_temporal_hires128_v1'
LAYERS=[f'P{s}L{k}'for s in [1,2]for k in range(1,7)]
ARMS=['Native','HiResMix']+LAYERS+['FirstPassMix','Pass1Mean6','Pass2Mean6','BothMean12','HiResAuxMix','NativeUniformMix','HiResUniformMix']


def prepare():
 from vg_tta.stvg_temporal_native_stage128_v1 import properties
 records=[dict(r,memory_sha256=sha(MEM/'temporal_inputs'/f"{r['alias']}.pt"),prior_sha256=sha(DEV/'priors'/f"{r['alias']}.pt"),hires_sha256=sha(HR/'predictions'/f"{r['alias']}.pt"))for r in read(MEM/'PLAN_LOCK.json')['records']]
 own=['protocols/stvg_temporal_native_stage128_v1.md','vg_tta/stvg_temporal_native_stage128_v1.py','scripts/run_stvg_temporal_native_stage128_v1.py','scripts/score_stvg_temporal_native_stage128_v1.py','scripts/publish_stvg_temporal_native_stage128_v1.py','results/stvg_error_state/2026-10-20/temporal_native_stage128/independent_stage_audit.py','external/TA-STVG/models/pipeline.py','external/TA-STVG/models/criterion.py','external/TA-STVG/config/defaults.py','methods/stvg_referdino_opd_v1/config.json']
 p=dict(records=records,count=128,seeds=[41,42,43],arms=ARMS,primary='HiResAuxMix',eligible=['FirstPassMix','Pass1Mean6','Pass2Mean6','BothMean12','HiResAuxMix'],individual_layers_diagnostic_not_eligible=True,GPU_seconds_cap=900,created_epoch=time.time(),deadline_epoch=time.time()+14400,pins={f:sha(R/f)for f in own},new_Source_encoder=0,new_RGB=0,new_experts=0,new_training=0,spatial_frozen=True,not_OPD=True)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked128 native stages',flush=True)


def measure():
 import numpy as np,torch
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source
 from methods.stvg_referdino_opd_v1._capture import transfer
 from methods.decota_final_simplified_v1.tensors import state_hash
 from main_method.runtime import resource_guard
 from vg_tta.stvg_temporal_native_stage128_v1 import native_stages,ensemble
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 setup();p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());lock=lease();beg=time.perf_counter();files=[];times=[];timecalls=auxcalls=nativechecks=0
 def guard():
  resource_guard();assert time.time()<p['deadline_epoch']and time.perf_counter()-beg+30<p['GPU_seconds_cap']
 try:
  guard();m=load_source();mh=state_hash(m.state_dict());assert m.use_aux_loss and len(m.ground_decoder.time_decoder.layers)==6;loading=time.perf_counter()-beg
  for r in p['records']:
   guard();tick=time.perf_counter();a=r['alias']
   for path,h in [(MEM/'temporal_inputs'/f'{a}.pt',r['memory_sha256']),(HR/'predictions'/f'{a}.pt',r['hires_sha256']),(PRE/'inputs'/f'{a}.pt',r['input_sha256']),(FIN/'Live/predictions'/f'{a}.pt',r['spatial_sha256']),(DEV/'priors'/f'{a}.pt',r['prior_sha256'])]:assert sha(path)==h
   inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');cache=load(MEM/'temporal_inputs'/f'{a}.pt');old=load(DEV/'priors'/f'{a}.pt');hr=load(HR/'predictions'/f'{a}.pt');ll1=[];ll2=[];traces=[]
   for off,(kw0,s0,g)in enumerate(zip(cache['inputs'],inp['spatial_inputs'],cache['grids'])):
    guard();kw=transfer(kw0,'cuda');kw['query_tgt']=kw['query_tgt'][:1].expand_as(kw['query_tgt']);sw=transfer(s0,'cuda');one,two,meta=native_stages(m,kw,sw,g);assert torch.equal(two[-1],old['logits'][off]);nativechecks+=1;timecalls+=2;auxcalls+=5;ll1.append(one);ll2.append(two);traces.append(meta);del kw,sw;gc.collect();torch.cuda.empty_cache()
   ids=inp['frame_ids'];stageqs=[];qs=dict(Native=old['q'],HiResMix=hr['outputs']['HiResMix']['q'])
   for stage,logits in enumerate([ll1,ll2],1):
    layerqs=[]
    for k in range(6):
     grid,q=mass_table(probabilities(inp['records'],[z[k]for z in logits]));assert grid==ids;layerqs.append(q);qs[f'P{stage}L{k+1}']=q
    stageqs.append(layerqs)
   qs.update(ensemble(old['q'],*stageqs,hr['observation']['q']));out={k:dict(q=q,interval=final_mbr(ids,q)['interval'])for k,q in qs.items()};assert set(out)==set(ARMS)and np.array_equal(qs['P2L6'],qs['Native'])and out['Native']['interval']==inp['interval'];trajectories=dict(E=np.asarray(inp['E']),**{f'Live{s}':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']});files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,outputs=out,trajectories=trajectories,pass_logits=[ll1,ll2],native_stage_traces=traces,records=inp['records'],GT_read=False,spatial_updated=False,new_student_updates=0)));times.append(time.perf_counter()-tick);put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time()),True);print('native_stage',a,round(time.perf_counter()-beg,2),flush=True)
   if len(times)==2:
    estimate=loading+128*max(times)+60;put(D/'PILOT2.json',dict(case_seconds=times,model_loading=loading,estimated_GPU_seconds=estimate,query_and_final_native_logits_exact=True,GT_read=False));assert estimate<900
   del inp,sp,cache,old,hr,ll1,ll2,traces,qs,out,trajectories,stageqs;gc.collect();torch.cuda.empty_cache()
  assert state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,weights_unchanged=True,native_queries_and_logits_exact=nativechecks));print('sealed128',flush=True)
 finally:
  put(D/'COST.json',dict(GPU_seconds=time.perf_counter()-beg+3,time_decoder_offset_calls=timecalls,attribute_and_action_gate_head_calls=auxcalls,time_projection_head_calls=timecalls,native_query_and_logits_exact_checks=nativechecks,new_Source_encoder=0,new_RGB=0,new_DINO=0,new_ReferDINO=0,new_training=0,spatial_updates=0,new_student_updates=0));lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
