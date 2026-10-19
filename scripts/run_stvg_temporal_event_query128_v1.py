"""Cached frozen teacher query feedback on real student temporal states."""
from pathlib import Path
import sys,os,time,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_event_query128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_event_query128';MEM=R/'artifacts/stvg_temporal_object_memory128_v1';ST=R/'artifacts/stvg_temporal_conditional_opd128_v1';HR=R/'artifacts/stvg_temporal_hires128_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1'
BASE=['GAll','GOcc0','RAll','ROcc0','ROcc1'];SAMPLES=[f'Sample{s}_{k}'for s in [0,1]for k in range(4)];ARMS=['Native','Student1']+BASE+SAMPLES+[k+'Mix'for k in BASE]+['Sample0Mean','Sample1Mean']

def prepare():
 from vg_tta.stvg_temporal_event_query128_v1 import properties
 records=[dict(r,memory_sha256=sha(MEM/'temporal_inputs'/f"{r['alias']}.pt"),student1_sha256=sha(ST/'predictions'/f"{r['alias']}.pt"),hires_sha256=sha(HR/'predictions'/f"{r['alias']}.pt"))for r in read(MEM/'PLAN_LOCK.json')['records']]
 own=['protocols/stvg_temporal_event_query128_v1.md','vg_tta/stvg_temporal_event_query128_v1.py','vg_tta/stvg_object_condition32_v1.py','vg_tta/stvg_soft_correspondence_v1.py','scripts/run_stvg_temporal_event_query128_v1.py','scripts/score_stvg_temporal_event_query128_v1.py','scripts/publish_stvg_temporal_event_query128_v1.py','external/TA-STVG/models/grounding_model/query_decoder.py','external/TA-STVG/models/grounding_model/classifier.py','methods/stvg_referdino_opd_v1/config.json']
 p=dict(records=records,count=128,seeds=[41,42,43],arms=ARMS,primary='ROcc0Mix',eligible=['GOcc0Mix','ROcc0Mix'],sampled_intervals_per_state=4,policy_seed=41,student1='completed conditionalOPD Live41 after40 actual updates',GPU_seconds_cap=900,deadline_epoch=time.time()+14400,created_epoch=time.time(),pins={f:sha(R/f)for f in own},new_Source_encoder=0,new_RGB=0,new_experts=0,new_training=0,spatial_frozen=True,not_OPD=True)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','deadline_epoch','created_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked128 arms22',flush=True)

def measure():
 import numpy as np,torch
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source
 from methods.stvg_referdino_opd_v1._capture import transfer
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_temporal_event_query128_v1 import occupancy,sampled_states,coefficients,query,properties
 from vg_tta.stvg_referdino_native32_v1 import interpolate
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 from main_method.runtime import resource_guard
 setup();p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());assert properties()['status']=='pass';lock=lease();beg=time.perf_counter();files=[];conditioned=checks=heads=0;times=[]
 def guard():
  resource_guard();assert time.time()<p['deadline_epoch']and time.perf_counter()-beg+30<p['GPU_seconds_cap']
 try:
  guard();m=load_source();mh=state_hash(m.state_dict());loading=time.perf_counter()-beg
  for r in p['records']:
   guard();t=time.perf_counter();a=r['alias']
   for path,h in [(MEM/'temporal_inputs'/f'{a}.pt',r['memory_sha256']),(ST/'predictions'/f'{a}.pt',r['student1_sha256']),(HR/'predictions'/f'{a}.pt',r['hires_sha256']),(PRE/'inputs'/f'{a}.pt',r['input_sha256']),(PRE/'teacher0'/f'{a}.pt',r['teacher_sha256']),(FIN/'Live/predictions'/f'{a}.pt',r['spatial_sha256'])]:assert sha(path)==h
   cache=load(MEM/'temporal_inputs'/f'{a}.pt');st=load(ST/'predictions'/f'{a}.pt');hr=load(HR/'predictions'/f'{a}.pt');z=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');prior=load(DEV/'priors'/f'{a}.pt');teach=load(PRE/'teacher0'/f'{a}.pt');q0=st['q0'];q1=st['outputs']['Live41']['q'];assert np.array_equal(q0,prior['q']);ids=z['frame_ids'];ref=interpolate(ids,r['positions8'],teach['native']['boxes'].numpy());u,pairs=sampled_states(q0,q1,r['query_ordinal']);o0=occupancy(q0);o1=occupancy(q1);weights={k:np.ones(65)for k in ['GAll','RAll']};weights.update(GOcc0=o0,ROcc0=o0,ROcc1=o1)
   for s in [0,1]:
    for k,(i,j)in enumerate(pairs[s]):weights[f'Sample{s}_{k}']=((np.arange(65)>=i)&(np.arange(65)<=j)).astype(float)
   logits={k:[]for k in BASE+SAMPLES};traces=[];native=[]
   for view,(kw0,g)in enumerate(zip(cache['inputs'],cache['grids'])):
    guard();kw=transfer(kw0,'cuda');kw['query_tgt']=kw['query_tgt'][:1].expand_as(kw['query_tgt'])
    with torch.no_grad():
     ll=m.temp_embed(m.ground_decoder.time_decoder(**kw))[-1].cpu();checks+=1;assert torch.equal(ll,prior['logits'][view]);native.append(ll)
     H,att,meta=coefficients(m,kw,g,torch.as_tensor(ref[view::2],device='cuda',dtype=torch.float64));heads+=1;queries={};source_query=kw['query_tgt'][0,0]
     for name in BASE+SAMPLES:
      qq,qt=query(H,att['Global'if name.startswith('G')else'Refer'],weights[name][view::2],source_query);newkw=dict(kw);newkw['query_tgt']=qq[None,None,:].expand_as(kw['query_tgt']);ll=m.temp_embed(m.ground_decoder.time_decoder(**newkw))[-1].cpu();assert torch.isfinite(ll).all();logits[name].append(ll);queries[name]=qt;conditioned+=1
     traces.append(dict(view=view,attention=meta,queries=queries,native_query=source_query.detach().cpu(),h=g['h'],w=g['w']))
    del kw,H,att,meta,newkw;gc.collect();torch.cuda.empty_cache()
   qs=dict(Native=q0,Student1=q1)
   for name,ll in logits.items():
    grid,q=mass_table(probabilities(z['records'],ll));assert grid==ids;qs[name]=q
   for k in BASE:qs[k+'Mix']=.5*q0+.5*qs[k]
   for s in [0,1]:qs[f'Sample{s}Mean']=sum(qs[f'Sample{s}_{k}']for k in range(4))/4
   out={k:dict(q=q,interval=final_mbr(ids,q)['interval'])for k,q in qs.items()};assert set(out)==set(ARMS)and out['Student1']['interval']==st['outputs']['Live41']['interval']and out['Native']['interval']==z['interval'];trajectories=dict(E=np.asarray(z['E']),**{f'Live{s}':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']});files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,outputs=out,trajectories=trajectories,logits=logits,query_traces=traces,conditioning_boxes=ref,sampled_pairs=pairs,paired_uniforms=u,occupancy0=o0,occupancy1=o1,q0=q0,q1=q1,GT_read=False,new_student_updates=0)));times.append(time.perf_counter()-t);put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time()),True);print('event_query',a,round(time.perf_counter()-beg,2),flush=True)
   if len(times)==2:
    estimate=loading+128*max(times)+60;put(D/'PILOT2.json',dict(case_seconds=times,estimated_GPU_seconds=estimate,GT_read=False,native_time_replay_exact=True));assert estimate<900
   del cache,st,hr,z,sp,prior,teach,logits,traces,qs,out,trajectories;gc.collect();torch.cuda.empty_cache()
  assert state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,weights_unchanged=True));print('sealed128',flush=True)
 finally:
  put(D/'COST.json',dict(GPU_seconds=time.perf_counter()-beg+3,conditioned_temporal_offset_decodes=conditioned,native_temporal_checks=checks,spatial_attention_head_calls=heads,new_Source_encoder=0,new_RGB=0,new_DINO=0,new_ReferDINO=0,network_training=0,spatial_updates=0,new_student_updates=0));lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
