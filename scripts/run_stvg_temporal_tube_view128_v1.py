"""Finite same-Source full-time RGB region views; all spatial predictions frozen."""
from pathlib import Path
import sys,os,time,gc,hashlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_tube_view128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_tube_view128';MEM=R/'artifacts/stvg_temporal_object_memory128_v1';ST=R/'artifacts/stvg_temporal_conditional_opd128_v1';HR=R/'artifacts/stvg_temporal_hires128_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1'
ARMS=['Native','HiResMix','Student1','Crop0','Crop0Mix','Crop1','Crop1Mix','SameS1Fixed','SameS1Refresh']


def prepare():
 import numpy as np
 from vg_tta.stvg_temporal_tube_view128_v1 import tube_window,properties
 assert properties()['status']=='pass';raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];records=[];required=changed=0
 for r in read(MEM/'PLAN_LOCK.json')['records']:
  a=r['alias'];z=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');st=load(ST/'predictions'/f'{a}.pt');row=raw[r['query_ordinal']]['input'];boxes=np.asarray(sp['outputs']['T20_Live_41']);iv=[z['interval'],st['outputs']['Live41']['interval']];windows=[];meta=[]
  for interval in iv:
   w,m=tube_window(z['frame_ids'],boxes,interval,row['width'],row['height']);windows.append(w);meta.append(m)
  full=(0,0,row['width'],row['height']);required+=len({tuple(w)for w in windows}-{full});changed+=windows[0]!=windows[1]
  records.append(dict(r,windows=windows,window_meta=meta,state_intervals=iv,student1_sha256=sha(ST/'predictions'/f'{a}.pt'),prior_sha256=sha(DEV/'priors'/f'{a}.pt'),hires_sha256=sha(HR/'predictions'/f'{a}.pt')))
 own=['protocols/stvg_temporal_tube_view128_v1.md','vg_tta/stvg_temporal_tube_view128_v1.py','scripts/run_stvg_temporal_tube_view128_v1.py','scripts/score_stvg_temporal_tube_view128_v1.py','scripts/publish_stvg_temporal_tube_view128_v1.py','results/stvg_error_state/2026-10-20/temporal_tube_view128/independent_view_audit.py','methods/stvg_referdino_opd_v1/_source_forward.py','methods/stvg_referdino_opd_v1/config.json']
 p=dict(records=records,count=128,seeds=[41,42,43],arms=ARMS,primary='Crop0Mix',eligible=['Crop0','Crop0Mix'],required_new_crop_Source_calls=required,changed_window_parents=changed,max_new_Source_calls=257,GPU_seconds_cap=900,deadline_epoch=time.time()+14400,created_epoch=time.time(),pins={f:sha(R/f)for f in own},new_experts=0,new_training=0,spatial_frozen=True,not_OPD=True)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','deadline_epoch','created_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked128 crop Source calls',required,'changed actual student windows',changed,flush=True)


def measure():
 import numpy as np,torch
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source,decode
 from methods.stvg_referdino_opd_v1._source_forward import frozen_forward
 from methods.decota_final_simplified_v1.tensors import state_hash
 from main_method.runtime import resource_guard
 from vg_tta.stvg_temporal_tube_view128_v1 import tube_window,crop
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 setup();p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];lock=lease();beg=time.perf_counter();files=[];calls=controlcalls=0;model_times=[];case_times=[];cpu_times=[]
 def guard():
  resource_guard();assert time.time()<p['deadline_epoch']and time.perf_counter()-beg+60<p['GPU_seconds_cap']
 try:
  guard();free,total=torch.cuda.mem_get_info();fraction=min(.8,(free-5*2**30)/total);assert fraction>.4;torch.cuda.set_per_process_memory_fraction(fraction);m=load_source();mh=state_hash(m.state_dict());loading=time.perf_counter()-beg;control_seconds=0.
  for r in p['records']:
   guard();tick=time.perf_counter();model_count_before=len(model_times);a=r['alias']
   for path,h in [(ST/'predictions'/f'{a}.pt',r['student1_sha256']),(HR/'predictions'/f'{a}.pt',r['hires_sha256']),(PRE/'inputs'/f'{a}.pt',r['input_sha256']),(FIN/'Live/predictions'/f'{a}.pt',r['spatial_sha256']),(DEV/'priors'/f'{a}.pt',r['prior_sha256'])]:assert sha(path)==h
   inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');old=load(DEV/'priors'/f'{a}.pt');st=load(ST/'predictions'/f'{a}.pt');hr=load(HR/'predictions'/f'{a}.pt');row=dict(raw[r['query_ordinal']]);row['parses']=read(R/r['subject_path'])['parses'];frames,ids=decode(row['input']);assert ids==inp['frame_ids'];W,H=row['input']['width'],row['input']['height'];full=[0,0,W,H];assert np.array_equal(crop(frames,full),frames);boxes=np.asarray(sp['outputs']['T20_Live_41']);q0=old['q'];q1=st['outputs']['Live41']['q'];observations={};dedup={};peaks=[]
   if not (D/'NATIVE_CONTROL.json').exists():
    t=time.perf_counter();controlcalls+=1;batch,recs,base=frozen_forward(m,frames,row);assert torch.equal(base.zero['boxes'].cpu(),inp['Source'])and all(torch.equal(x.cpu(),y)for x,y in zip(base.zero['logits'],old['logits']));grid,q=mass_table(probabilities(recs,[v.cpu()for v in base.zero['logits']]));assert grid==ids and np.array_equal(q,q0);base.restore(base.initial);del batch,base;gc.collect();torch.cuda.empty_cache();control_seconds=time.perf_counter()-t;put(D/'NATIVE_CONTROL.json',dict(status='pass',alias=a,Source_boxes_logits_q_exact=True,full_RGB_crop_exact=True,seconds=control_seconds,GT_read=False));tick+=control_seconds
   for state,iv in enumerate(r['state_intervals']):
    window,meta=tube_window(ids,boxes,iv,W,H);assert window==r['windows'][state]and meta==r['window_meta'][state];key=tuple(window)
    if key==tuple(full):observations[state]=dict(q=q0,window=window,interval=iv,reused='original whole-image Source',raw_RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),logits=old['logits'],records=old['records'],shape=[65,H,W,3],normalized_shape=None);continue
    if key in dedup:observations[state]=dict(dedup[key],interval=iv,reused='same RGB window and deterministic teacher');continue
    guard();view=crop(frames,window);viewrow=dict(row,input=dict(row['input'],width=view.shape[2],height=view.shape[1]));t=time.perf_counter();torch.cuda.reset_peak_memory_stats();calls+=1;assert calls<=p['required_new_crop_Source_calls'];batch,recs,base=frozen_forward(m,view,viewrow);ll=[x.detach().cpu()for x in base.zero['logits']];actions=[x.detach().cpu()for x in base.zero['actions']];grid,q=mass_table(probabilities(recs,ll));assert grid==ids;shape=list(batch['videos'].tensors.shape);assert shape[0]==65 and shape[-2]==224;peaks.append(torch.cuda.max_memory_allocated());base.restore(base.initial);del batch,base;gc.collect();torch.cuda.empty_cache();model_times.append(time.perf_counter()-t);obs=dict(q=q,window=window,interval=iv,reused=False,raw_RGB_sha256=hashlib.sha256(view.tobytes()).hexdigest(),logits=ll,actions=actions,records=recs,shape=list(view.shape),normalized_shape=shape);observations[state]=obs;dedup[key]=obs;del view
   qs=dict(Native=q0,HiResMix=hr['outputs']['HiResMix']['q'],Student1=q1,Crop0=observations[0]['q'],Crop1=observations[1]['q']);qs.update(Crop0Mix=.5*q0+.5*qs['Crop0'],Crop1Mix=.5*q0+.5*qs['Crop1'],SameS1Fixed=.5*q1+.5*qs['Crop0'],SameS1Refresh=.5*q1+.5*qs['Crop1']);out={k:dict(q=q,interval=final_mbr(ids,q)['interval'])for k,q in qs.items()};assert set(out)==set(ARMS)and out['Native']['interval']==inp['interval'];trajectories=dict(E=np.asarray(inp['E']),**{f'Live{s}':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']});files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,outputs=out,trajectories=trajectories,observations=observations,original_RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),original_RGB_shape=list(frames.shape),peak_GPU_allocated=max(peaks,default=0),GT_read=False,new_student_updates=0,spatial_updated=False)));case_times.append(time.perf_counter()-tick);cpu_times.append(case_times[-1]-sum(model_times[model_count_before:]));put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time(),crop_Source_calls=calls),True);print('tube_view',a,round(time.perf_counter()-beg,2),r['windows'],flush=True)
   if len(case_times)>=2 and (len(model_times)>=2 or p['required_new_crop_Source_calls']<2)and not(D/'PILOT2.json').exists():
    cpu=max(cpu_times);estimate=loading+control_seconds+max(model_times,default=0)*p['required_new_crop_Source_calls']+max(cpu,.5)*128+60;put(D/'PILOT2.json',dict(case_seconds=case_times[:2],model_seconds=model_times[:4],model_loading=loading,native_control_seconds=control_seconds,estimated_GPU_seconds=estimate,known_unique_view_calls=p['required_new_crop_Source_calls'],GT_read=False));assert estimate<900
   del frames,inp,sp,old,st,hr,observations,dedup,out,qs,trajectories;gc.collect();torch.cuda.empty_cache()
  assert calls==p['required_new_crop_Source_calls'] and state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,weights_unchanged=True));print('sealed128',flush=True)
 finally:
  put(D/'COST.json',dict(GPU_seconds=time.perf_counter()-beg+3,crop_Source_calls=calls,native_control_Source_calls=controlcalls,total_Source_offset_forwards=2*(calls+controlcalls),new_DINO=0,new_ReferDINO=0,new_network_training=0,spatial_updates=0,new_student_updates=0));lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
