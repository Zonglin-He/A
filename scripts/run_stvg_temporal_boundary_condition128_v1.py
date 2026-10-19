"""Actual native decoder boundary masks; finite cached capability probe."""
from pathlib import Path
import sys,os,time,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_boundary_condition128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_boundary_condition128';MEM=R/'artifacts/stvg_temporal_object_memory128_v1';EVENT=R/'artifacts/stvg_temporal_event_query128_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1'

def prepare():
 from vg_tta.stvg_temporal_boundary_condition128_v1 import properties
 records=[dict(r,memory_sha256=sha(MEM/'temporal_inputs'/f"{r['alias']}.pt"),sample_sha256=sha(EVENT/'predictions'/f"{r['alias']}.pt"))for r in read(MEM/'PLAN_LOCK.json')['records']]
 from vg_tta.stvg_temporal_boundary_condition128_v1 import key_mask
 for r in records:
  e=load(EVENT/'predictions'/f"{r['alias']}.pt");ids=e['frame_ids'];unique=[set(),set()]
  for pairs in e['sampled_pairs']:
   for pair in pairs:
    for direction in ['start','end']:
     cut=ids[pair[1]if direction=='start'else pair[0]]
     for variant in ['Correct','Wrong']:
      for off in [0,1]:
       mask=key_mask(ids[off::2],cut,direction,variant)
       if mask.any():unique[off].add(tuple(map(bool,mask)))
  r['planned_unique_condition_decodes']=sum(map(len,unique))
 own=['protocols/stvg_temporal_boundary_condition128_v1.md','vg_tta/stvg_temporal_boundary_condition128_v1.py','scripts/run_stvg_temporal_boundary_condition128_v1.py','scripts/score_stvg_temporal_boundary_condition128_v1.py','scripts/publish_stvg_temporal_boundary_condition128_v1.py','external/TA-STVG/models/grounding_model/query_decoder.py','methods/stvg_referdino_opd_v1/config.json','scripts/stvg_temporal_marginal_decode_util256_v1.py']
 p=dict(records=records,count=128,seeds=[41,42,43],states=['S0','S1'],samples_per_state=4,directions=['start','end'],variants=['Correct','Wrong'],GPU_seconds_cap=900,max_conditioned_offset_decodes=8192,max_native_checks=256,deadline_epoch=time.time()+14400,created_epoch=time.time(),pins={f:sha(R/f)for f in own},spatial_frozen=True,new_Source_encoder=0,new_RGB=0,new_experts=0,new_training=0,new_student_updates=0,not_OPD=True)
 p['total_unique_condition_decodes']=sum(r['planned_unique_condition_decodes']for r in records);p['max_unique_condition_decodes_per_parent']=max(r['planned_unique_condition_decodes']for r in records);put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','deadline_epoch','created_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked128 native boundary masks',flush=True)

def measure():
 import torch,numpy as np
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source
 from methods.stvg_referdino_opd_v1._capture import transfer
 from methods.decota_final_simplified_v1.tensors import state_hash
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table,independent_cdf
 from vg_tta.stvg_temporal_boundary_condition128_v1 import key_mask,conditional
 from vg_tta.stvg_temporal_event_query128_v1 import sampled_states
 from main_method.runtime import resource_guard
 setup();p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());lock=lease();beg=time.perf_counter();files=[];calls=nativechecks=0;times=[];attnchecks=0;maxblocked=0.;cdfmax=0.;active=[None];handles=[]
 def check():
  resource_guard();free,total=torch.cuda.mem_get_info();assert free>4*2**30 and torch.cuda.memory_reserved()<total-5*2**30;assert time.time()<p['deadline_epoch']and time.perf_counter()-beg+30<p['GPU_seconds_cap']
 def audit_attention(module,args,kwargs,out):
  nonlocal attnchecks,maxblocked
  mask=kwargs['key_padding_mask'];assert active[0]is not None and torch.equal(mask,active[0]);w=out[1];assert torch.isfinite(w).all();v=float(w[...,mask[0]].abs().max())if bool(mask.any())else 0.;assert v==0.;attnchecks+=1;maxblocked=max(maxblocked,v)
 try:
  torch.cuda.set_per_process_memory_fraction(.8);check();m=load_source();mh=state_hash(m.state_dict());loading=time.perf_counter()-beg
  for layer in m.ground_decoder.time_decoder.layers:handles.append(layer.self_attn.register_forward_hook(audit_attention,with_kwargs=True))
  for r in p['records']:
   calls_before=calls
   check();tick=time.perf_counter();a=r['alias']
   for path,h in [(MEM/'temporal_inputs'/f'{a}.pt',r['memory_sha256']),(EVENT/'predictions'/f'{a}.pt',r['sample_sha256']),(PRE/'inputs'/f'{a}.pt',r['input_sha256']),(FIN/'Live/predictions'/f'{a}.pt',r['spatial_sha256'])]:assert sha(path)==h
   cache=load(MEM/'temporal_inputs'/f'{a}.pt');event=load(EVENT/'predictions'/f'{a}.pt');prior=load(DEV/'priors'/f'{a}.pt');inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');q0=event['q0'];q1=event['q1'];ids=list(inp['frame_ids']);u,pairs=sampled_states(q0,q1,r['query_ordinal']);assert np.array_equal(u,event['paired_uniforms'])and all(np.array_equal(x,y)for x,y in zip(pairs,event['sampled_pairs']));assert np.array_equal(q0,prior['q']);queries={};native=[];gpu=[];cached_logits=[{},{}];maskrecords=[]
   for view,kw0 in enumerate(cache['inputs']):
    kw=transfer(kw0,'cuda');kw['query_tgt']=kw['query_tgt'][:1].expand_as(kw['query_tgt']);assert not bool(kw['query_mask'].any());active[0]=kw['query_mask']
    with torch.no_grad():ll=m.temp_embed(m.ground_decoder.time_decoder(**kw))[-1].cpu()
    nativechecks+=1;assert torch.equal(ll,prior['logits'][view]);native.append(ll);gpu.append(kw);cached_logits[view][tuple([False]*len(kw['query_tgt']))]=ll
   requests=[]
   for state in [0,1]:
    for k,pair in enumerate(pairs[state]):
     for direction in p['directions']:
      cut=int(ids[pair[1]if direction=='start'else pair[0]])
      for variant in p['variants']:
       ll=[];masks=[]
       for view,kw in enumerate(gpu):
        check();mask=key_mask(inp['records'][view]['frame_ids'],cut,direction,variant);masktuple=tuple(map(bool,mask));masks.append(mask)
        if masktuple not in cached_logits[view]:
         newkw=dict(kw);newkw['query_mask']=torch.as_tensor(mask[None],device='cuda');active[0]=newkw['query_mask']
         with torch.no_grad():zz=m.temp_embed(m.ground_decoder.time_decoder(**newkw))[-1].cpu()
         assert torch.isfinite(zz).all();cached_logits[view][masktuple]=zz;calls+=1
        ll.append(cached_logits[view][masktuple])
       grids=probabilities(inp['records'],ll);grid,q=mass_table(grids);assert grid==ids;err=float(np.max(abs(q-independent_cdf(grids,ids))));assert err<2e-12;cdfmax=max(cdfmax,err);label=conditional(q,q0,pair,direction);base=conditional(q0,q0,pair,direction)
       name=f'S{state}_{k}_{direction}_{variant}';queries[name]=dict(state=state,sample=k,pair=np.array(pair),direction=direction,variant=variant,physical_cut=cut,masks=masks,logits=ll,joint=q,label=label,native_label=base)
   trajectories=dict(E=np.asarray(inp['E']),**{f'Live{s}':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']});files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,records=inp['records'],q0=q0,q1=q1,sampled_pairs=pairs,paired_uniforms=u,queries=queries,trajectories=trajectories,GT_read=False,no_new_student_updates=True)));times.append(time.perf_counter()-tick);assert calls-calls_before==r['planned_unique_condition_decodes'];put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time()),True);print('boundary_condition',a,round(time.perf_counter()-beg,2),flush=True)
   if len(files)==2:
    per_decode=max(t/max(p['records'][k]['planned_unique_condition_decodes'],1)for k,t in enumerate(times));estimate=loading+per_decode*p['max_unique_condition_decodes_per_parent']*128+60;put(D/'PILOT2.json',dict(case_seconds=times,case_unique_decodes=[r['planned_unique_condition_decodes']for r in p['records'][:2]],prelocked_total_unique_decodes=p['total_unique_condition_decodes'],prelocked_max_per_parent=p['max_unique_condition_decodes_per_parent'],allparent_worst_decode_count_estimate_seconds=estimate,estimated_GPU_seconds=estimate,model_loading=loading,native_logits_exact=True,GT_read=False));assert estimate<900
   del cache,event,prior,inp,sp,gpu,cached_logits,queries,trajectories;gc.collect();torch.cuda.empty_cache()
  assert calls==p['total_unique_condition_decodes'] and calls<=8192 and nativechecks==256 and state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,weights_unchanged=True));print('sealed128 boundary condition qualification',flush=True)
 finally:
  if not (D/'GLOBAL_PREDICTION_BARRIER.json').exists()and 'queries'in locals():
   save(D/f'failed_current_parent_{time.time_ns()}.pt',dict(alias=locals().get('a'),completed_queries=queries,cached_offset_logits=locals().get('cached_logits'),native_logits=locals().get('native'),not_a_valid_complete_prediction=True,GT_read=False))
  for h in handles:h.remove()
  put(D/'COST.json',dict(GPU_seconds=time.perf_counter()-beg+3,conditioned_temporal_offset_decodes=calls,native_temporal_checks=nativechecks,attention_layer_calls_audited=attnchecks,max_blocked_attention=maxblocked,max_posterior_CDF_error=cdfmax,new_Source_encoder=0,new_RGB=0,new_DINO=0,new_ReferDINO=0,network_training=0,spatial_updates=0,new_student_updates=0));lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
