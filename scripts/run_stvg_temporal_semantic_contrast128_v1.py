"""Bounded same-Source textual ablation, actual new forwards counted."""
from pathlib import Path
import sys,os,time,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_semantic_contrast128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_semantic_contrast128';OLD=R/'artifacts/stvg_temporal_object_memory128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1'

def prepare():
 from vg_tta.stvg_temporal_semantic_contrast128_v1 import properties
 records=[];old=read(OLD/'PLAN_LOCK.json')
 for r in old['records']:
  phrase=read(R/r['subject_path'])['parses']['old'].get('phrase','');records.append(dict(r,static_caption=phrase,static_missing=not bool(phrase),prior_sha256=sha(DEV/'priors'/f"{r['alias']}.pt")))
 own=['protocols/stvg_temporal_semantic_contrast128_v1.md','vg_tta/stvg_temporal_semantic_contrast128_v1.py','scripts/run_stvg_temporal_semantic_contrast128_v1.py','scripts/score_stvg_temporal_semantic_contrast128_v1.py','scripts/publish_stvg_temporal_semantic_contrast128_v1.py','methods/stvg_referdino_opd_v1/config.json','methods/stvg_referdino_opd_v1/_source_forward.py']
 put(D/'PLAN_LOCK.json',dict(records=records,count=128,seeds=[41,42,43],arms=['Native','PriorSquare','Contrast'],primary='Contrast',GPU_seconds_cap=900,deadline_epoch=time.time()+14400,created_epoch=time.time(),pins={f:sha(R/f)for f in own},no_new_experts=True,spatial_frozen=True,not_OPD=True))
 put(P/'PROTOCOL_PUBLIC.json',dict(count=128,exposure='all128 exposed development',static_nonempty=sum(not r['static_missing']for r in records),rule='existing cached parses.old.phrase; no prompt alternatives',formula='qC proportional p0^2/qS, zero support retained',coefficient=1,arms=['Native','PriorSquare','Contrast'],primary='Contrast',GPU_seconds_cap=900,max_Source_queries=128,max_Source_offsets=256,new_experts=0,new_training=0,spatial_frozen=True,properties=properties(),confirmation_gate='Contrast-Native mean>=2pp CI>0 AND Contrast-PriorSquare CI>0',not_OPD=True));print('locked128',flush=True)

def measure():
 import torch,numpy as np
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source,decode
 from methods.stvg_referdino_opd_v1._source_forward import frozen_forward
 from methods.decota_final_simplified_v1.tensors import state_hash
 from main_method.runtime import resource_guard
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 from vg_tta.stvg_temporal_semantic_contrast128_v1 import posteriors,properties
 setup();assert properties()['status']=='pass';p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];lock=lease();beg=time.perf_counter();files=[];calls=0;times=[]
 def check():
  resource_guard();assert time.time()<p['deadline_epoch']and time.perf_counter()-beg+10<900
 try:
  check();m=load_source();mh=state_hash(m.state_dict());loading=time.perf_counter()-beg
  for r in p['records']:
   check();tick=time.perf_counter();a=r['alias'];old=load(DEV/'priors'/f'{a}.pt');z=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');assert sha(DEV/'priors'/f'{a}.pt')==r['prior_sha256']and sha(PRE/'inputs'/f'{a}.pt')==r['input_sha256']and sha(FIN/'Live/predictions'/f'{a}.pt')==r['spatial_sha256']and sha(R/r['subject_path'])==r['subject_sha256'];row=dict(raw[r['query_ordinal']]);row['parses']=read(R/r['subject_path'])['parses'];static=None;observed=None
   if r['static_caption']:
    if r['static_caption']==row['input']['caption']:static=old['q'];observed=dict(exact_same_caption_reuse=True)
    else:
     frames,ids=decode(row['input']);assert ids==old['ids'];rr=dict(row,input=dict(row['input'],caption=r['static_caption']));batch,recs,base=frozen_forward(m,frames,rr);calls+=1;ll=[x.detach().cpu()for x in base.zero['logits']];grid,static=mass_table(probabilities(recs,ll));assert grid==ids;observed=dict(records=recs,logits=ll,q=static,RGB_sha256=__import__('hashlib').sha256(frames.tobytes()).hexdigest(),same_caption_reuse=False);base.restore(base.initial);del batch,base,frames;gc.collect();torch.cuda.empty_cache()
   qs=dict(Native=old['q'],**posteriors(old['q'],static));outputs={arm:dict(interval=final_mbr(old['ids'],q)['interval'],q=q)for arm,q in qs.items()};assert outputs['Native']['interval']==z['interval'];trajectories=dict(E=np.asarray(z['E']),**{f'Live{s}':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']});files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=old['ids'],outputs=outputs,trajectories=trajectories,static_observation=observed,static_missing=r['static_missing'],GT_read=False)));times.append(time.perf_counter()-tick);put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time()),True);print('semantic',a,round(time.perf_counter()-beg,2),flush=True)
   if len(files)==2:
    estimate=loading+max(times)*128+30;put(D/'PILOT2.json',dict(case_seconds=times,model_loading=loading,estimated_GPU_seconds=estimate,GT_read=False));assert estimate<900
   del old,z,sp,observed,outputs,static;gc.collect();torch.cuda.empty_cache()
  assert state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,all_before_GT=True,plan_sha256=sha(D/'PLAN_LOCK.json'),weights_unchanged=True));print('sealed128',flush=True)
 finally:
  put(D/f'COST_{time.time_ns()}.json',dict(GPU_seconds=time.perf_counter()-beg+3,new_Source_queries=calls,Source_offset_forwards=2*calls,new_DINO=0,new_ReferDINO=0,network_training=0,spatial_updates=0));lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
