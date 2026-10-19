"""Finite full-span global33 + posterior boundary32 real RGB teacher views."""
from pathlib import Path
import sys,os,time,gc,hashlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_boundary_view128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_boundary_view128';MEM=R/'artifacts/stvg_temporal_object_memory128_v1';ST=R/'artifacts/stvg_temporal_conditional_opd128_v1';HR=R/'artifacts/stvg_temporal_hires128_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1';AUX=R/'artifacts/stvg_temporal_native_stage128_v1'
ARMS=['Native','HiResAuxMix','ActualRawOldActions','ActualMix','ActualAlignedMix','NearestRawOldActions','NearestMix','NearestAlignedMix']


def prepare():
 import numpy as np
 from vg_tta.stvg_temporal_boundary_view128_v1 import sampling,properties
 assert properties()['status']=='pass';records=[]
 for r in read(MEM/'PLAN_LOCK.json')['records']:
  a=r['alias'];old=load(DEV/'priors'/f'{a}.pt');st=load(ST/'predictions'/f'{a}.pt');s0,m0=sampling(old['ids'],old['q']);s1,m1=sampling(old['ids'],st['outputs']['Live41']['q']);records.append(dict(r,schedules=[s0,s1],schedule_meta=[m0,m1],prior_sha256=sha(DEV/'priors'/f'{a}.pt'),student1_sha256=sha(ST/'predictions'/f'{a}.pt'),hires_sha256=sha(HR/'predictions'/f'{a}.pt'),aux_sha256=sha(AUX/'predictions'/f'{a}.pt')))
 own=['protocols/stvg_temporal_boundary_view128_v1.md','vg_tta/stvg_temporal_boundary_view128_v1.py','scripts/run_stvg_temporal_boundary_view128_v1.py','scripts/score_stvg_temporal_boundary_view128_v1.py','scripts/publish_stvg_temporal_boundary_view128_v1.py','results/stvg_error_state/2026-10-20/temporal_boundary_view128/independent_boundary_audit.py','methods/stvg_referdino_opd_v1/_source_forward.py','methods/stvg_referdino_opd_v1/config.json']
 p=dict(records=records,count=128,seeds=[41,42,43],arms=ARMS,primary='ActualMix',eligible=['ActualRawOldActions','ActualMix','ActualAlignedMix'],max_new_Source_calls=257,GPU_seconds_cap=900,deadline_epoch=time.time()+14400,created_epoch=time.time(),pins={f:sha(R/f)for f in own},new_experts=0,new_training=0,spatial_frozen=True,not_OPD=True)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','deadline_epoch','created_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());put(P/'SCHEDULE_QUALIFICATION.json',dict(parents=128,changed_actual_student_schedules=sum(r['schedules'][0]!=r['schedules'][1]for r in records),mean_new_frames=float(np.mean([r['schedule_meta'][0]['new_frames_vs_original']for r in records])),new_S1_accuracy_or_model=False));print('locked128',flush=True)


def measure():
 import numpy as np,torch
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source,decode
 from methods.stvg_referdino_opd_v1._source_forward import frozen_forward
 from methods.decota_final_simplified_v1.tensors import state_hash
 from main_method.runtime import resource_guard
 from vg_tta.stvg_temporal_boundary_view128_v1 import sampling,project_logits,cross_utility,choose
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_posterior_decision_tree_v1 import expected_iou_fast
 setup();p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];lock=lease();beg=time.perf_counter();files=[];calls=controlcalls=0;times=[];total_new=0
 def guard():
  resource_guard();assert time.time()<p['deadline_epoch']and time.perf_counter()-beg+35<p['GPU_seconds_cap']
 try:
  guard();free,total=torch.cuda.mem_get_info();fraction=min(.8,(free-5*2**30)/total);assert fraction>.4;torch.cuda.set_per_process_memory_fraction(fraction);m=load_source();mh=state_hash(m.state_dict());loading=time.perf_counter()-beg;control_seconds=0.
  for r in p['records']:
   guard();tick=time.perf_counter();a=r['alias']
   for path,h in [(ST/'predictions'/f'{a}.pt',r['student1_sha256']),(HR/'predictions'/f'{a}.pt',r['hires_sha256']),(AUX/'predictions'/f'{a}.pt',r['aux_sha256']),(PRE/'inputs'/f'{a}.pt',r['input_sha256']),(FIN/'Live/predictions'/f'{a}.pt',r['spatial_sha256']),(DEV/'priors'/f'{a}.pt',r['prior_sha256'])]:assert sha(path)==h
   inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');old=load(DEV/'priors'/f'{a}.pt');aux=load(AUX/'predictions'/f'{a}.pt');hr=load(HR/'predictions'/f'{a}.pt');ids=old['ids'];new,meta=sampling(ids,old['q']);assert new==r['schedules'][0]and meta==r['schedule_meta'][0];row=dict(raw[r['query_ordinal']]);row['parses']=read(R/r['subject_path'])['parses'];union=sorted(set(ids)|set(new));allframes,returned=decode(dict(row['input'],frame_ids=union));assert returned==union;frames=allframes[np.searchsorted(union,ids)];actual=allframes[np.searchsorted(union,new)];nearest=frames[meta['nearest_original_indices']];original_sha=hashlib.sha256(frames.tobytes()).hexdigest();assert original_sha==hr['observation']['RGB_sha256'];total_new+=meta['new_frames_vs_original'];observations={};dedup={};peak=0
   if controlcalls==0:
    t=time.perf_counter();controlcalls+=1;batch,recs,base=frozen_forward(m,frames,row);assert torch.equal(base.zero['boxes'].cpu(),inp['Source'])and all(torch.equal(x.cpu(),y)for x,y in zip(base.zero['logits'],old['logits']));grid,q=mass_table(probabilities(recs,[v.cpu()for v in base.zero['logits']]));assert grid==ids and np.array_equal(q,old['q']);base.restore(base.initial);del batch,base;gc.collect();torch.cuda.empty_cache();control_seconds=time.perf_counter()-t;put(D/'NATIVE_CONTROL.json',dict(status='pass',alias=a,Source_boxes_logits_q_exact=True,original_RGB_exact=True,seconds=control_seconds,GT_read=False));tick+=control_seconds
   for kind,pixels in [('Actual',actual),('Nearest',nearest)]:
    key=hashlib.sha256(pixels.tobytes()).hexdigest()
    if key in dedup:observations[kind]=dict(dedup[key],reused='identical pixels/text/timestamps');continue
    guard();viewrow=dict(row,frame_ids=new,input=dict(row['input'],frame_ids=new));t=time.perf_counter();torch.cuda.reset_peak_memory_stats();calls+=1;assert calls+controlcalls<=p['max_new_Source_calls'];batch,recs,base=frozen_forward(m,pixels,viewrow);ll=[x.detach().cpu()for x in base.zero['logits']];grid,q=mass_table(probabilities(recs,ll));assert grid==new;shape=list(batch['videos'].tensors.shape);assert shape[0]==65 and shape[-2]==224;peak=max(peak,torch.cuda.max_memory_allocated());base.restore(base.initial);del batch,base;gc.collect();torch.cuda.empty_cache();obs=dict(q=q,logits=ll,records=recs,RGB_sha256=key,shape=list(pixels.shape),normalized_shape=shape,model_seconds=time.perf_counter()-t,reused=False);observations[kind]=obs;dedup[key]=obs
   sources={'Native':[(ids,old['q'],1.)],'HiResAuxMix':[(ids,aux['outputs']['HiResAuxMix']['q'],1.)]}
   for kind in ['Actual','Nearest']:
    ob=observations[kind];al=project_logits(ob['records'],ob['logits'],old['records']);grid,aq=mass_table(probabilities(old['records'],al));assert grid==ids;ob['aligned_q']=aq;ob['aligned_logits']=al;sources[kind+'RawOldActions']=[(new,ob['q'],1.)];sources[kind+'Mix']=[(ids,old['q'],.5),(new,ob['q'],.5)];sources[kind+'AlignedMix']=[(ids,.5*old['q']+.5*aq,1.)]
   out={}
   for arm,parts in sources.items():
    u=None
    for grid,q,w in parts:
     v=expected_iou_fast(ids,q)if grid==ids else cross_utility(ids,grid,q);u=w*v if u is None else u+w*v
    out[arm]=dict(interval=choose(ids,u),sources=[dict(ids=g,q=q,weight=w)for g,q,w in parts])
   assert set(out)==set(ARMS)and out['Native']['interval']==inp['interval'];trajectories=dict(E=np.asarray(inp['E']),**{f'Live{s}':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']});files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,new_ids=new,outputs=out,trajectories=trajectories,observations=observations,original_RGB_sha256=original_sha,peak_GPU_allocated=peak,GT_read=False,new_student_updates=0,spatial_updated=False)));times.append(time.perf_counter()-tick);put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time(),Source_calls=calls),True);print('boundary',a,round(time.perf_counter()-beg,2),meta['new_frames_vs_original'],flush=True)
   if len(times)==2:
    estimate=loading+control_seconds+max(times)*128+35;put(D/'PILOT2.json',dict(case_seconds=times,model_loading=loading,native_control_seconds=control_seconds,conservative_estimated_GPU_seconds=estimate,GT_read=False));assert estimate<p['GPU_seconds_cap']
   del allframes,frames,actual,nearest,inp,sp,old,aux,hr,observations,dedup,out,sources,trajectories;gc.collect();torch.cuda.empty_cache()
  assert state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,weights_unchanged=True));print('sealed128',flush=True)
 finally:
  put(D/'COST.json',dict(GPU_seconds=time.perf_counter()-beg+3,view_Source_calls=calls,native_control_Source_calls=controlcalls,total_Source_offset_forwards=2*(calls+controlcalls),new_unique_Source_frames=total_new,new_DINO=0,new_ReferDINO=0,new_network_training=0,spatial_updates=0,new_student_updates=0));lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
