"""Finite frozen motion-mean intervention from exact existing encoder inputs."""
from pathlib import Path
import sys,time,os,gc,hashlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_motion_counterfactual128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_motion_counterfactual128';EARLY=R/'artifacts/stvg_temporal_early_fusion128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1';STRONG=R/'artifacts/stvg_temporal_native_stage128_v1'

def prepare():
 from vg_tta.stvg_temporal_motion_counterfactual128_v1 import properties
 assert read(EARLY/'FINAL_IMMUTABILITY_AUDIT.json')['status']=='pass'
 records=[dict(r,early_input_sha256=sha(EARLY/'early_inputs'/f"{r['alias']}.pt"))for r in read(EARLY/'PLAN_LOCK.json')['records']]
 own=['protocols/stvg_temporal_motion_counterfactual128_v1.md','vg_tta/stvg_temporal_motion_counterfactual128_v1.py','scripts/run_stvg_temporal_motion_counterfactual128_v1.py','scripts/score_stvg_temporal_motion_counterfactual128_v1.py','scripts/publish_stvg_temporal_motion_counterfactual128_v1.py','vg_tta/stvg_temporal_early_fusion128_v1.py','vg_tta/stvg_temporal_semantic_contrast128_v1.py','external/TA-STVG/models/grounding_model/modal_encoder.py','methods/decota_final_simplified_v1/replay.py','methods/stvg_referdino_opd_v1/config.json']
 p=dict(records=records,count=128,seeds=[41,42,43],arms=['Native','MeanMotion','PriorSquare','Contrast','HiResAuxMix'],primary='Contrast',GPU_seconds_cap=900,max_early_encoder_reencodes=512,max_prefix_replays=256,deadline_epoch=time.time()+14400,created_epoch=time.time(),pins={f:sha(R/f)for f in own},quotient_coefficient=1,mean_axis='each original offset, each spatial cell/channel, T',mean_accumulation='FP32 no norm compensation',new_experts=0,new_training=0,new_student_updates=0,spatial_frozen=True,not_OPD=True)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','deadline_epoch','created_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked128 mean-motion contrast',flush=True)

def measure():
 import torch,numpy as np
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_temporal_early_fusion128_v1 import fuse,decode_prefix
 from vg_tta.stvg_temporal_motion_counterfactual128_v1 import temporal_mean,posteriors
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table,independent_cdf
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 from main_method.runtime import resource_guard
 setup();p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());lock=lease();beg=time.perf_counter();files=[];prefixfiles=[];times=[];encodes=replays=0;cdfmax=0.
 def check():
  resource_guard();free,total=torch.cuda.mem_get_info();assert free>4*2**30 and torch.cuda.memory_reserved()<total-5*2**30;assert time.time()<p['deadline_epoch']and time.perf_counter()-beg+30<p['GPU_seconds_cap']
 try:
  torch.cuda.set_per_process_memory_fraction(.8);check();m=load_source();mh=state_hash(m.state_dict());loading=time.perf_counter()-beg
  for r in p['records']:
   check();tick=time.perf_counter();a=r['alias']
   for path,key in [(EARLY/'early_inputs'/f'{a}.pt','early_input_sha256'),(PRE/'inputs'/f'{a}.pt','input_sha256'),(FIN/'Live/predictions'/f'{a}.pt','spatial_sha256'),(DEV/'priors'/f'{a}.pt','prior_sha256'),(STRONG/'predictions'/f'{a}.pt','strong_sha256')]:assert sha(path)==r[key]
   cap=load(EARLY/'early_inputs'/f'{a}.pt');inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');prior=load(DEV/'priors'/f'{a}.pt');strong=load(STRONG/'predictions'/f'{a}.pt');ids=list(inp['frame_ids']);logits={};prefixes={};statistics=[]
   for name in ['Native','MeanMotion']:
    views=[]
    for c,v in zip(cap['captured_encoder_inputs'],cap['original_views']):
     check()
     if name=='MeanMotion':
      cc,ss=temporal_mean(c,v);h,w=v['info']['fea_map_size'];ss['actual_input_sha256']=hashlib.sha256(cc['src'].contiguous().numpy().tobytes()).hexdigest();ss['actual_motion_template']=cc['src'][-h*w:,0].clone();statistics.append(ss)
     else:cc=c
     nv,mm=fuse(m,cc,v);encodes+=1;assert mm['encoder_attention_calls']==0 and nv['info']['encoded_mask'].dtype==torch.bool
     if name=='Native':assert torch.equal(nv['prefix'],v['prefix'])
     views.append(nv)
    values=decode_prefix(m,views,len(ids));replays+=1;logits[name]=values['logits'];prefixes[name]=[v['prefix']for v in views]
    if name=='Native':assert all(torch.equal(x,y)for x,y in zip(logits[name],prior['logits']))
    del values,views;gc.collect();torch.cuda.empty_cache()
   qs=dict(HiResAuxMix=strong['outputs']['HiResAuxMix']['q'])
   for name,ll in logits.items():
    grids=probabilities(inp['records'],ll);grid,q=mass_table(grids);assert grid==ids;err=float(abs(q-independent_cdf(grids,ids)).max());assert err<2e-12;cdfmax=max(cdfmax,err);qs[name]=q
   assert np.array_equal(qs['Native'],prior['q']);qs.update(posteriors(qs['Native'],qs['MeanMotion']));out={k:dict(q=q,interval=final_mbr(ids,q)['interval'])for k,q in qs.items()};trajectories=dict(E=np.asarray(inp['E']),**{f'Live{s}':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']})
   prefixfiles.append(save(D/'prefixes'/f'{a}.pt',dict(alias=a,prefixes=prefixes,GT_read=False,native_exact=True)))
   files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,records=inp['records'],outputs=out,trajectories=trajectories,logits=logits,motion_statistics=statistics,native_exact=True,GT_read=False)));times.append(time.perf_counter()-tick);put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time()),True);print('motion_mean',a,round(time.perf_counter()-beg,2),flush=True)
   if len(files)==2:
    estimate=loading+max(times)*128+60;put(D/'PILOT2.json',dict(case_seconds=times,model_loading=loading,estimated_GPU_seconds=estimate,native_prefix_final_time_exact=True,GT_read=False));assert estimate<900
   del cap,inp,sp,prior,strong,logits,prefixes,qs,out,trajectories;gc.collect();torch.cuda.empty_cache()
  assert encodes==512 and replays==256 and state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,prefixes=prefixfiles,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,weights_unchanged=True));print('sealed128 motion counterfactual',flush=True)
 finally:
  if not (D/'GLOBAL_PREDICTION_BARRIER.json').exists()and 'logits'in locals():save(D/f'failed_current_parent_{time.time_ns()}.pt',dict(alias=locals().get('a'),completed_prefixes=locals().get('prefixes'),completed_logits=logits,not_complete=True,GT_read=False))
  put(D/'COST.json',dict(GPU_seconds=time.perf_counter()-beg+3,early_fusion_encoder_reencodes=encodes,full_native_prefix_replays=replays,max_CDF_error=cdfmax,new_Source_RGB_encoder=0,new_RGB=0,new_DINO=0,new_ReferDINO=0,network_training=0,spatial_updates=0,new_student_updates=0));lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
