"""One full Source stream, cached old full65 numeric control, frozen spatial outputs."""
from pathlib import Path
import sys,time,os,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_full_context128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_full_context128';HR=R/'artifacts/stvg_temporal_hires128_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1';OLD=R/'artifacts/stvg_observation_factorial32_v1'
ARMS=['Native','Full65','Full65Mix','HiResMix']

def prepare():
 from vg_tta.stvg_temporal_full_context128_v1 import properties
 records=[dict(r,hires_sha256=sha(HR/'predictions'/f"{r['alias']}.pt"))for r in read(HR/'PLAN_LOCK.json')['records']];control=next(r for r in read(OLD/'PLAN_LOCK.json')['records']if r['alias']=='NH001');control=dict(control,dense_prediction_sha256=sha(OLD/'predictions/NH001.pt'))
 own=['protocols/stvg_temporal_full_context128_v1.md','vg_tta/stvg_temporal_full_context128_v1.py','scripts/run_stvg_temporal_full_context128_v1.py','scripts/score_stvg_temporal_full_context128_v1.py','scripts/publish_stvg_temporal_full_context128_v1.py','methods/decota_final_simplified_v1/replay.py','methods/decota_final_simplified_v1/backbone.py','methods/stvg_referdino_opd_v1/_visual_host.py','methods/stvg_referdino_opd_v1/_source_forward.py','methods/stvg_referdino_opd_v1/config.json']
 p=dict(records=records,count=128,seeds=[41,42,43],arms=ARMS,primary='Full65Mix',eligible=['Full65','Full65Mix'],native_control=control,GPU_seconds_cap=900,deadline_epoch=time.time()+14400,created_epoch=time.time(),pins={f:sha(R/f)for f in own},new_experts=0,new_training=0,spatial_frozen=True,source_checkpoint_sha256='5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83',input_height=224,not_OPD=True)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','native_control','deadline_epoch','created_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked128 full65',flush=True)

def measure():
 import torch,numpy as np,hashlib
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source,decode
 from methods.decota_final_simplified_v1.tensors import state_hash
 from main_method.runtime import resource_guard
 from vg_tta.stvg_temporal_full_context128_v1 import full_forward,single_posterior
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 from scripts.run_decota_paper_main_v1 import read_row
 setup();p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];lock=lease();beg=time.perf_counter();files=[];calls=controls=0;times=[]
 def check():
  resource_guard();assert time.time()<p['deadline_epoch']and time.perf_counter()-beg+30<p['GPU_seconds_cap']
 try:
  check();free,total=torch.cuda.mem_get_info();fraction=min(.8,(free-5*2**30)/total);assert fraction>.4;torch.cuda.set_per_process_memory_fraction(fraction);m=load_source();mh=state_hash(m.state_dict());loading=time.perf_counter()-beg
  c=p['native_control'];assert sha(OLD/'predictions/NH001.pt')==c['dense_prediction_sha256'];old=load(OLD/'predictions/NH001.pt');cr=read_row('hc2',c['query_ordinal']);frames,ids=decode(cr['input']);assert ids==cr['frame_ids']==old['frame_ids']==old['temporal']['records'][0]['frame_ids'];ct=time.perf_counter();controls+=1;ctrl,meta=full_forward(m,frames,cr);assert torch.equal(ctrl['logits'][0],old['temporal']['logits'][0]);qc=single_posterior(ids,ctrl['logits'][0]);put(D/'NATIVE_FULL65_CONTROL.json',dict(status='pass',first_old_numeric_control='NH001',FP32_full65_logits_exact=True,original65_nodes_exact=True,old_actions_gates_not_cached=True,new_actions_finite=bool(torch.isfinite(ctrl['actions'][0]).all()),seconds=time.perf_counter()-ct,info=meta,GT_read=False));control_seconds=time.perf_counter()-ct;del frames,ctrl,old;gc.collect();torch.cuda.empty_cache()
  for r in p['records']:
   check();a=r['alias'];t=time.perf_counter();assert calls<128
   hr=load(HR/'predictions'/f'{a}.pt');prior=load(DEV/'priors'/f'{a}.pt');z=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt')
   assert sha(HR/'predictions'/f'{a}.pt')==r['hires_sha256']and sha(PRE/'inputs'/f'{a}.pt')==r['input_sha256']and sha(FIN/'Live/predictions'/f'{a}.pt')==r['spatial_sha256'];row=dict(raw[r['query_ordinal']]);row['parses']=read(R/r['subject_path'])['parses'];frames,ids=decode(row['input']);assert ids==z['frame_ids']==prior['ids']and len(ids)==65 and m.cfg.INPUT.MAX_VIDEO_LEN>=65;assert hashlib.sha256(frames.tobytes()).hexdigest()==hr['observation']['RGB_sha256'];torch.cuda.reset_peak_memory_stats();calls+=1;v,info=full_forward(m,frames,row);q=single_posterior(ids,v['logits'][0]);assert info['normalized_shape'][-2]==224 and len(v['logits'])==1
   qs=dict(Native=prior['q'],Full65=q,Full65Mix=.5*prior['q']+.5*q,HiResMix=hr['outputs']['HiResMix']['q']);out={k:dict(q=qq,interval=final_mbr(ids,qq)['interval'])for k,qq in qs.items()};assert out['Native']['interval']==z['interval'];trajectories=dict(E=np.asarray(z['E']),**{f'Live{s}':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']});files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,outputs=out,trajectories=trajectories,logits=v['logits'][0],actions=v['actions'][0],gates=v['gates'][0],full_context_info=info,peak_GPU_allocated=torch.cuda.max_memory_allocated(),raw_RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),GT_read=False)));times.append(time.perf_counter()-t);put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time()),True);print('full65',a,round(time.perf_counter()-beg,2),flush=True)
   if len(times)==2:
    estimate=loading+control_seconds+128*max(times)+60;put(D/'PILOT2.json',dict(case_seconds=times,loading_seconds=loading,control_seconds=control_seconds,estimated_GPU_seconds=estimate,GT_read=False));assert estimate<p['GPU_seconds_cap']
   del hr,prior,z,sp,frames,v,q,qs,out,trajectories;gc.collect();torch.cuda.empty_cache()
  assert state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,weights_unchanged=True));print('sealed128',flush=True)
 finally:
  put(D/'COST.json',dict(GPU_seconds=time.perf_counter()-beg+3,current_full65_Source_attempts=calls,old_full65_control_attempts=controls,full_Source_encoder_forwards=calls+controls,AMP_decoder_passes=2*(calls+controls),FP32_replay_decoder_passes=2*(calls+controls),new_DINO=0,new_ReferDINO=0,new_network_training=0,spatial_updates=0));lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
