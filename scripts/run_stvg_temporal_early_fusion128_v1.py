"""Frozen early fusion ROI capability, exact Native and matched old late control."""
from pathlib import Path
import sys,time,os,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_early_fusion128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_early_fusion128';MEM=R/'artifacts/stvg_temporal_object_memory128_v1';STRONG=R/'artifacts/stvg_temporal_native_stage128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1'

def prepare():
 from vg_tta.stvg_temporal_early_fusion128_v1 import properties
 records=[dict(r,late_sha256=sha(MEM/'predictions'/f"{r['alias']}.pt"),strong_sha256=sha(STRONG/'predictions'/f"{r['alias']}.pt"))for r in read(MEM/'PLAN_LOCK.json')['records']]
 own=['protocols/stvg_temporal_early_fusion128_v1.md','vg_tta/stvg_temporal_early_fusion128_v1.py','vg_tta/stvg_temporal_object_memory128_v1.py','scripts/run_stvg_temporal_early_fusion128_v1.py','scripts/score_stvg_temporal_early_fusion128_v1.py','scripts/publish_stvg_temporal_early_fusion128_v1.py','external/TA-STVG/models/grounding_model/modal_encoder.py','methods/decota_final_simplified_v1/replay.py','methods/stvg_referdino_opd_v1/_source_forward.py','methods/stvg_referdino_opd_v1/_capture.py','methods/stvg_referdino_opd_v1/config.json']
 p=dict(records=records,count=128,seeds=[41,42,43],arms=['Native','LateRefer','EarlyRefer','EarlyShift','HiResAuxMix'],primary='EarlyRefer',GPU_seconds_cap=900,deadline_epoch=time.time()+14400,created_epoch=time.time(),pins={f:sha(R/f)for f in own},max_Source_queries=128,original_motion_prior='.5+.5density',early_motion_keys_only=True,recompute_native_queries_gates=True,new_experts=0,new_training=0,new_student_updates=0,spatial_frozen=True,not_OPD=True)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','deadline_epoch','created_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked128 early fusion',flush=True)

def measure():
 import torch,numpy as np
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source,decode
 from methods.stvg_referdino_opd_v1._source_forward import frozen_forward
 from methods.stvg_referdino_opd_v1._capture import transfer
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_temporal_early_fusion128_v1 import fuse,decode_prefix
 from vg_tta.stvg_referdino_native32_v1 import interpolate
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table,independent_cdf
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 from main_method.runtime import resource_guard
 setup();p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];lock=lease();beg=time.perf_counter();files=[];captures=[];prefixfiles=[];times=[];sourcecalls=sourceoffsets=fusioncalls=prefixdecodes=0;cdfmax=0.
 def check():
  resource_guard();free,total=torch.cuda.mem_get_info();assert free>4*2**30 and torch.cuda.memory_reserved()<total-5*2**30;assert time.time()<p['deadline_epoch']and time.perf_counter()-beg+30<p['GPU_seconds_cap']
 try:
  torch.cuda.set_per_process_memory_fraction(.8);check();m=load_source();mh=state_hash(m.state_dict());loading=time.perf_counter()-beg
  for r in p['records']:
   check();tick=time.perf_counter();a=r['alias']
   for path,h in [(MEM/'predictions'/f'{a}.pt',r['late_sha256']),(STRONG/'predictions'/f'{a}.pt',r['strong_sha256']),(PRE/'inputs'/f'{a}.pt',r['input_sha256']),(FIN/'Live/predictions'/f'{a}.pt',r['spatial_sha256']),(DEV/'priors'/f'{a}.pt',r['prior_sha256']),(PRE/'teacher0'/f'{a}.pt',r['teacher_sha256'])]:assert sha(path)==h
   inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');prior=load(DEV/'priors'/f'{a}.pt');late=load(MEM/'predictions'/f'{a}.pt');strong=load(STRONG/'predictions'/f'{a}.pt');teacher=load(PRE/'teacher0'/f'{a}.pt');row=dict(raw[r['query_ordinal']]);assert sha(R/r['subject_path'])==r['subject_sha256'];row['parses']=read(R/r['subject_path'])['parses'];cap=[]
   def capture(module,args,kwargs):
    nonlocal sourceoffsets
    assert len(args)==1;sourceoffsets+=1;cap.append(dict(src=transfer(args[0],'cpu'),kwargs=transfer(kwargs,'cpu')))
   hook=m.ground_encoder.encoder.register_forward_pre_hook(capture,with_kwargs=True)
   try:
    frames,ids=decode(row['input']);sourcecalls+=1;batch,recs,base=frozen_forward(m,frames,row)
   finally:hook.remove()
   assert len(cap)==2 and ids==inp['frame_ids']and torch.equal(base.zero['boxes'].cpu(),inp['Source'])and all(torch.equal(x.cpu(),y)for x,y in zip(base.zero['logits'],prior['logits']));views=transfer(base.views,'cpu');captures.append(save(D/'early_inputs'/f'{a}.pt',dict(alias=a,captured_encoder_inputs=cap,original_views=views,native_exact=True,GT_read=False)));del base,batch,frames;gc.collect();torch.cuda.empty_cache()
   ref=interpolate(ids,r['positions8'],teacher['native']['boxes'].numpy());shift=ref.copy();shift[:,0]=(shift[:,0]+.5)%1;logits={};meta={};prefixes={}
   for name,boxes in [('Native',None),('EarlyRefer',ref),('EarlyShift',shift)]:
    newviews=[];metadata=[]
    for off in [0,1]:
     check();nv,mm=fuse(m,cap[off],views[off],None if boxes is None else boxes[off::2]);fusioncalls+=1
     if name=='Native':assert torch.equal(nv['prefix'],views[off]['prefix'])
     if len(files)<2 and name=='Native':
      full=np.tile([.5,.5,1.,1.],(len(ids[off::2]),1));fv,fm=fuse(m,cap[off],views[off],full);fusioncalls+=1;assert torch.equal(fv['prefix'],views[off]['prefix'])and fm['encoder_attention_calls']==0
     assert torch.equal(nv['info']['encoded_mask'],views[off]['info']['encoded_mask']);newviews.append(nv);metadata.append(mm)
    check();values=decode_prefix(m,newviews,len(ids));prefixdecodes+=1
    if name=='Native':assert all(torch.equal(x,y)for x,y in zip(values['logits'],prior['logits']))
    logits[name]=values['logits'];meta[name]=metadata;prefixes[name]=[v['prefix']for v in newviews];del newviews,values;gc.collect();torch.cuda.empty_cache()
   qs=dict(LateRefer=late['outputs']['ReferROI']['q'],HiResAuxMix=strong['outputs']['HiResAuxMix']['q'])
   for name,ll in logits.items():
    grids=probabilities(recs,ll);grid,q=mass_table(grids);assert grid==ids;err=float(abs(q-independent_cdf(grids,ids)).max());assert err<2e-12;cdfmax=max(cdfmax,err);qs[name]=q
   assert np.array_equal(qs['Native'],prior['q']);out={k:dict(q=q,interval=final_mbr(ids,q)['interval'])for k,q in qs.items()};assert out['LateRefer']['interval']==late['outputs']['ReferROI']['interval']and out['HiResAuxMix']['interval']==strong['outputs']['HiResAuxMix']['interval'];trajectories=dict(E=np.asarray(inp['E']),**{f'Live{s}':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']})
   prefixfiles.append(save(D/'prefixes'/f'{a}.pt',dict(alias=a,new_prefixes=prefixes,native_early_replay_exact=True,GT_read=False)))
   files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,records=recs,outputs=out,trajectories=trajectories,logits=logits,prior_metadata=meta,refer_boxes=ref,shifted_boxes=shift,GT_read=False,Source_native_exact=True,early_native_exact=True)));times.append(time.perf_counter()-tick);put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time()),True);print('early_fusion',a,round(time.perf_counter()-beg,2),flush=True)
   if len(files)==2:
    estimate=loading+max(times)*128+60;put(D/'PILOT2.json',dict(case_seconds=times,model_loading=loading,estimated_GPU_seconds=estimate,Source_encoder_norm_and_final_time_exact=True,GT_read=False));assert estimate<900
   del inp,sp,prior,late,strong,teacher,cap,views,logits,meta,prefixes,qs,out,trajectories;gc.collect();torch.cuda.empty_cache()
  assert sourcecalls==128 and state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,early_inputs=captures,regional_prefixes=prefixfiles,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,weights_unchanged=True));print('sealed128 early fusion',flush=True)
 finally:
  if not (D/'GLOBAL_PREDICTION_BARRIER.json').exists()and 'cap'in locals():
   save(D/f'failed_current_parent_{time.time_ns()}.pt',dict(alias=locals().get('a'),early_inputs=cap,original_views=locals().get('views'),completed_prefixes=locals().get('prefixes'),completed_logits=locals().get('logits'),not_complete_prediction=True,GT_read=False))
  put(D/'COST.json',dict(GPU_seconds=time.perf_counter()-beg+3,new_Source_queries=sourcecalls,Source_offset_captures=sourceoffsets,Source_input_image_slots=sourcecalls*65,early_encoder_reencodes=fusioncalls,frozen_prefix_full_native_replays=prefixdecodes,max_posterior_CDF_error=cdfmax,new_RGB_positions=0,new_DINO=0,new_ReferDINO=0,network_training=0,spatial_updates=0,new_student_updates=0));lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),completed_parents_preserved=True));raise
