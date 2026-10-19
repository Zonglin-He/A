"""Finite unaveraged source motion-attribute feedback capability check."""
from pathlib import Path
import sys,os,time,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_motion_attribute128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_motion_attribute128';MEM=R/'artifacts/stvg_temporal_object_memory128_v1';ST=R/'artifacts/stvg_temporal_conditional_opd128_v1';HR=R/'artifacts/stvg_temporal_hires128_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1'
CURVES=['GlobalQuery','GlobalMax','EQuery','LiveQuery','ReferQuery','ReferClassShift','ReferTimeShift','ReferOcc0','ReferOcc1']
ARMS=['Native','HiResMix','Student1']+CURVES

def prepare():
 from vg_tta.stvg_temporal_motion_attribute128_v1 import properties
 records=[dict(r,memory_sha256=sha(MEM/'temporal_inputs'/f"{r['alias']}.pt"),student1_sha256=sha(ST/'predictions'/f"{r['alias']}.pt"),prior_sha256=sha(DEV/'priors'/f"{r['alias']}.pt"),hires_sha256=sha(HR/'predictions'/f"{r['alias']}.pt"))for r in read(MEM/'PLAN_LOCK.json')['records']]
 own=['results/stvg_error_state/2026-10-20/temporal_motion_attribute128/independent_numeric_audit.py','protocols/stvg_temporal_motion_attribute128_v1.md','vg_tta/stvg_temporal_motion_attribute128_v1.py','vg_tta/stvg_temporal_increment128_v1.py','vg_tta/stvg_temporal_event_query128_v1.py','scripts/run_stvg_temporal_motion_attribute128_v1.py','scripts/score_stvg_temporal_motion_attribute128_v1.py','scripts/publish_stvg_temporal_motion_attribute128_v1.py','external/TA-STVG/models/grounding_model/classifier.py','external/TA-STVG/models/bert_model/bert_module.py','external/TA-STVG/models/criterion.py','methods/stvg_referdino_opd_v1/config.json']
 p=dict(records=records,count=128,seeds=[41,42,43],arms=ARMS,curves=CURVES,primary='ReferQuery',eligible=['GlobalQuery','ReferQuery','ReferOcc0'],lambda_fixed=.5,prototype='normalized sigmoid of temporal-mean native 34 motion class logits; no guessed lexicon',GPU_seconds_cap=900,deadline_epoch=time.time()+14400,created_epoch=time.time(),pins={f:sha(R/f)for f in own},new_Source_encoder=0,new_RGB=0,new_experts=0,new_training=0,spatial_frozen=True,not_OPD=True)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','deadline_epoch','created_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked128',flush=True)

def measure():
 import numpy as np,torch
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source
 from methods.stvg_referdino_opd_v1._capture import transfer
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_temporal_motion_attribute128_v1 import coverage,per_frame,prototype,curve,edges,properties
 from vg_tta.stvg_temporal_event_query128_v1 import occupancy
 from vg_tta.stvg_temporal_increment128_v1 import feedback,tilt
 from vg_tta.stvg_referdino_native32_v1 import interpolate
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 from main_method.runtime import resource_guard
 setup();p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());assert properties()['status']=='pass';lock=lease();beg=time.perf_counter();files=[];calls=fullchecks=0;times=[]
 def guard():
  resource_guard();assert time.time()<p['deadline_epoch'] and time.perf_counter()-beg+30<p['GPU_seconds_cap']
 try:
  guard();m=load_source();mh=state_hash(m.state_dict());loading=time.perf_counter()-beg
  for r in p['records']:
   guard();tick=time.perf_counter();a=r['alias']
   for path,h in [(MEM/'temporal_inputs'/f'{a}.pt',r['memory_sha256']),(ST/'predictions'/f'{a}.pt',r['student1_sha256']),(HR/'predictions'/f'{a}.pt',r['hires_sha256']),(PRE/'inputs'/f'{a}.pt',r['input_sha256']),(PRE/'teacher0'/f'{a}.pt',r['teacher_sha256']),(FIN/'Live/predictions'/f'{a}.pt',r['spatial_sha256']),(DEV/'priors'/f'{a}.pt',r['prior_sha256'])]:assert sha(path)==h
   cache=load(MEM/'temporal_inputs'/f'{a}.pt');st=load(ST/'predictions'/f'{a}.pt');hr=load(HR/'predictions'/f'{a}.pt');z=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');prior=load(DEV/'priors'/f'{a}.pt');teach=load(PRE/'teacher0'/f'{a}.pt');ids=z['frame_ids'];q0=prior['q'];q1=st['outputs']['Live41']['q'];ref=interpolate(ids,r['positions8'],teach['native']['boxes'].numpy());boxes=dict(E=np.asarray(z['E']),Live=np.asarray(sp['outputs']['T20_Live_41']),Refer=ref);raw={k:np.zeros((65,34),np.float32)for k in ['Global','E','Live','Refer']};traces=[]
   for off,(kw0,g)in enumerate(zip(cache['inputs'],cache['grids'])):
    guard();kw=transfer(kw0,'cuda');h,w=g['h'],g['w'];l=h*w;f=kw['encoded_feature'];T=f.shape[1];fmap=f[-l:].permute(1,2,0).reshape(T,256,h,w);text=f[:-l].mean(1).unsqueeze(0)[:,:1];valid=g['valid'].cuda();assert valid.all()
    native=per_frame(m.t_spatial_clas,fmap,text);calls+=1;full=per_frame(m.t_spatial_clas,fmap,text,torch.ones(T,l,device='cuda'));calls+=1;assert torch.equal(native,full);fullchecks+=1;raw['Global'][off::2]=native.numpy();meta={}
    for k in boxes:
     area,missing=coverage(torch.as_tensor(boxes[k][off::2],device='cuda',dtype=torch.float64),valid,h,w);att_trace=[] if a=='RC001'else None;v=per_frame(m.t_spatial_clas,fmap,text,area,att_trace);calls+=1;raw[k][off::2]=v.numpy();meta[k]=dict(area=area.cpu(),empty=missing.cpu(),attention_numeric_trace=att_trace)
    assert all('forward'not in layer.attention.self.__dict__ for layer in m.t_spatial_clas.layer_ca);traces.append(dict(offset=off,h=h,w=w,geometry=meta));del kw,f,fmap,text,area;gc.collect();torch.cuda.empty_cache()
   qg=prototype(raw['Global']);o0=occupancy(q0);o1=occupancy(q1);qp0=prototype(raw['Refer'],o0);qp1=prototype(raw['Refer'],o1);qs=dict(Global=qg,Occ0=qp0,Occ1=qp1,ClassShift=np.roll(qg,1));curves=dict(GlobalQuery=curve(raw['Global'],qg),GlobalMax=(1/(1+np.exp(-raw['Global'].astype(float)))).max(-1),EQuery=curve(raw['E'],qg),LiveQuery=curve(raw['Live'],qg),ReferQuery=curve(raw['Refer'],qg),ReferClassShift=curve(raw['Refer'],qs['ClassShift']),ReferOcc0=curve(raw['Refer'],qp0),ReferOcc1=curve(raw['Refer'],qp1));curves['ReferTimeShift']=np.roll(curves['ReferQuery'],32);e=edges(ids,z['frame_count']if 'frame_count'in z else int(r['frame_count']));out=dict(Native=dict(q=q0,interval=final_mbr(ids,q0)['interval']),HiResMix=hr['outputs']['HiResMix'],Student1=st['outputs']['Live41']);fb={}
   for k,x in curves.items():
    zz,au=feedback(ids,e,x);assert np.array_equal(tilt(q0,zz,0),q0);qt=tilt(q0,zz,.5);out[k]=dict(q=qt,interval=final_mbr(ids,qt)['interval']);fb[k]=au
   assert set(out)==set(ARMS) and out['Native']['interval']==z['interval'];trajectories=dict(E=np.asarray(z['E']),**{f'Live{s}':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']});files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,edges=e,outputs=out,trajectories=trajectories,raw_motion_class_logits=raw,prototypes=qs,curves=curves,feedback=fb,geometry_traces=traces,conditioning_boxes=boxes,occupancy0=o0,occupancy1=o1,q0=q0,q1=q1,GT_read=False,spatial_updated=False,new_student_updates=0)));times.append(time.perf_counter()-tick);put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time()),True);print('motion_attribute',a,round(time.perf_counter()-beg,2),flush=True)
   if len(times)==2:
    estimate=loading+128*max(times)+60;put(D/'PILOT2.json',dict(case_seconds=times,estimated_GPU_seconds=estimate,GT_read=False,full_frame_attribute_logits_exact=True));assert estimate<900
   del cache,st,hr,z,sp,prior,teach,raw,boxes,out,traces,curves,trajectories;gc.collect();torch.cuda.empty_cache()
  assert state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,weights_unchanged=True));print('sealed128',flush=True)
 finally:
  put(D/'COST.json',dict(GPU_seconds=time.perf_counter()-beg+3,motion_attribute_head_calls=calls,native_full_geometry_exact_checks=fullchecks,new_Source_encoder=0,new_RGB=0,new_DINO=0,new_ReferDINO=0,network_training=0,spatial_updates=0,new_student_updates=0));lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
