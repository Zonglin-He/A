"""Frozen trajectories select real per-frame RGB observations for time only."""
from pathlib import Path
import sys,os,time,gc,fcntl,hashlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_moving_rgb128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_moving_rgb128';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1';STAGE=R/'artifacts/stvg_temporal_native_stage128_v1';UNION=R/'artifacts/stvg_temporal_tube_view128_v1'
ARMS=['Native','HiResAuxMix','UnionCrop0Mix','MovingLive','MovingSource','MovingLiveMix','MovingSourceMix']

def rectkey(rects):return hashlib.sha256(repr(rects).encode()).hexdigest()

def prepare():
 import numpy as np
 from vg_tta.stvg_temporal_moving_rgb128_v1 import rectangles,independent_rectangles,properties
 assert read(P/'LEGACY_QUALIFICATION_PUBLIC.json')['status']=='pass_scoped_saved_outputs'
 raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];records=[];required=0
 for r in read(R/'artifacts/stvg_temporal_cross_offset128_v1/PLAN_LOCK.json')['records']:
  a=r['alias'];inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');row=raw[r['query_ordinal']]['input'];b=dict(MovingLive=np.asarray(sp['outputs']['T20_Live_41']),MovingSource=inp['Source'].numpy());rect={k:rectangles(v,row['width'],row['height'])for k,v in b.items()};assert all(rect[k]==independent_rectangles(v,row['width'],row['height'])for k,v in b.items());required+=len({rectkey(v)for v in rect.values()});records.append(dict(r,rectangles=rect,union_sha256=sha(UNION/'predictions'/f'{a}.pt'),subject_sha256=sha(R/r['subject_path'])))
 assert len(records)==128 and len({r['parent']for r in records})==128
 old=read(R/'artifacts/stvg_temporal_cross_offset128_v1/PLAN_LOCK.json');own=[f for f in old['pins']if 'cross_offset'not in f]
 own+=['protocols/stvg_temporal_moving_rgb128_v1.md','vg_tta/stvg_temporal_moving_rgb128_v1.py','scripts/run_stvg_temporal_moving_rgb128_v1.py','scripts/score_stvg_temporal_moving_rgb128_v1.py','scripts/publish_stvg_temporal_moving_rgb128_v1.py','scripts/audit_stvg_temporal_moving_rgb_legacy_v1.py','vg_tta/spatial_ssl_diagnostic_v1.py','external/TA-STVG/datasets/build.py','external/TA-STVG/datasets/transforms.py','external/TA-STVG/datasets/collate_batch.py','artifacts/stvg_temporal_moving_rgb128_v1/LEGACY_PRIVATE_QUALIFICATION.json','results/stvg_error_state/2026-10-20/temporal_moving_rgb128/LEGACY_QUALIFICATION_PUBLIC.json']
 own+=list(map(lambda p:str(p.relative_to(R)),(R/'methods/stvg_referdino_opd_v1').glob('*.py')))+list(map(lambda p:str(p.relative_to(R)),(R/'methods/decota_final_simplified_v1').glob('*.py')))+list(map(lambda p:str(p.relative_to(R)),(R/'external/TA-STVG/models').rglob('*.py')))
 own=sorted(set(own));p=dict(records=records,count=128,seeds=[41,42,43],arms=ARMS,primary='MovingLiveMix',pure_teacher='MovingLive',source_checkpoint='checkpoints/TASTVG_VidSTG.pth',source_checkpoint_sha256=old['source_checkpoint_sha256'],required_geometry_unique_view_calls=required,max_Source_calls=required+1,GPU_seconds_cap=900,CPU_seconds_cap=900,deadline_epoch=time.time()+14400,created_epoch=time.time(),pins={f:sha(R/f)for f in own},new_experts=0,new_training=0,new_student_updates=0,spatial_frozen=True,not_OPD=True)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked128 moving RGB geometry views',required)

def measure():
 import numpy as np,torch,cv2
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source,decode
 from methods.stvg_referdino_opd_v1._source_forward import frozen_forward
 from methods.decota_final_simplified_v1.tensors import state_hash
 from main_method.runtime import resource_guard
 from vg_tta.stvg_temporal_moving_rgb128_v1 import crop,digest,independent_rectangles
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 setup();cv2.setNumThreads(2)
 def reject_GT(event,args):
  if event=='open'and args and isinstance(args[0],(str,bytes,Path))and str(args[0]).endswith('/data/hcstvg2_official_metadata/val_v2.json'):raise PermissionError('moving RGB measure forbids actual HC labels')
 sys.addaudithook(reject_GT);p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];lock=lease();beg=time.perf_counter();files=[];hooks=[];partial=None;model_times=[];cpu_times=[];case_times=[];native_seconds=0.;native_attempts=0;native_completed=0;view_attempts=0;view_completed=0;pixelchecks=0;success=False;peakalloc=peakreserve=0
 counts={k:0 for k in ['Source_attempts','Source_completed','image_attempts','image_completed','motion_attempts','motion_completed']}
 def check():
  resource_guard();free,total=torch.cuda.mem_get_info();assert free>5*2**30 and time.time()<p['deadline_epoch']and time.perf_counter()-beg+40<900
 def pins():assert all(sha(R/f)==h for f,h in p['pins'].items())
 try:
  resource_guard();free,total=torch.cuda.mem_get_info();assert free>4*2**30;fraction=min(.8,(free-5*2**30)/total);assert fraction>.4;torch.cuda.set_per_process_memory_fraction(fraction);check();m=load_source();mh=state_hash(m.state_dict());flags={n:x.training for n,x in m.named_modules()};assert not any(flags.values());loading=time.perf_counter()-beg;peakalloc=torch.cuda.max_memory_allocated();peakreserve=torch.cuda.max_memory_reserved()
  def enter(kind):
   def f(module,args):counts[kind+'_attempts']+=1;assert counts[kind+'_attempts']<=2*p['max_Source_calls']
   return f
  def leave(kind):
   def f(module,args,out):counts[kind+'_completed']+=1
   return f
  for module,kind in [(m.vis_encoder,'image'),(m.vid,'motion')]:hooks.extend([module.register_forward_pre_hook(enter(kind)),module.register_forward_hook(leave(kind))])
  def forward(frames,row):
   check();counts['Source_attempts']+=1;assert counts['Source_attempts']<=p['max_Source_calls'];v=frozen_forward(m,frames,row);counts['Source_completed']+=1;return v
  for r in p['records']:
   check();tick=time.perf_counter();before=len(model_times);control=0.;a=r['alias']
   for path,h in [(PRE/'inputs'/f'{a}.pt',r['input_sha256']),(FIN/'Live/predictions'/f'{a}.pt',r['spatial_sha256']),(DEV/'priors'/f'{a}.pt',r['prior_sha256']),(STAGE/'predictions'/f'{a}.pt',r['native_stage_sha256']),(UNION/'predictions'/f'{a}.pt',r['union_sha256']),(R/r['subject_path'],r['subject_sha256'])]:assert sha(path)==h
   inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');old=load(DEV/'priors'/f'{a}.pt');strong=load(STAGE/'predictions'/f'{a}.pt');union=load(UNION/'predictions'/f'{a}.pt');row=dict(raw[r['query_ordinal']]);row['parses']=read(R/r['subject_path'])['parses'];frames,ids=decode(row['input']);assert ids==inp['frame_ids']and digest(frames)==union['original_RGB_sha256'];W,H=row['input']['width'],row['input']['height'];observations={};partial=dict(alias=a,query_ordinal=r['query_ordinal'],observations=observations,GT_read=False,only_completed_CPU_outputs=True)
   if not files:
    t=time.perf_counter();native_attempts+=1;partial['inflight']=dict(stage='Native_Source_entered');batch,recs,base=forward(frames,row);native_completed+=1;partial['inflight']=dict(stage='Native_Source_returned',logits=[v.detach().cpu()for v in base.zero['logits']],actions=[v.detach().cpu()for v in base.zero['actions']],boxes=base.zero['boxes'].detach().cpu(),records=recs);check();assert torch.equal(base.zero['boxes'].cpu(),inp['Source'])and all(torch.equal(x.cpu(),y)for x,y in zip(base.zero['logits'],old['logits']));grid,q=mass_table(probabilities(recs,[v.cpu()for v in base.zero['logits']]));assert grid==ids and np.array_equal(q,old['q']);base.restore(base.initial);del batch,base;gc.collect();torch.cuda.empty_cache();native_seconds=control=time.perf_counter()-t;peakalloc=max(peakalloc,torch.cuda.max_memory_allocated());peakreserve=max(peakreserve,torch.cuda.max_memory_reserved());put(D/'NATIVE_CONTROL.json',dict(status='pass',Source_boxes_logits_q_exact=True,seconds=native_seconds,GT_read=False));save(D/'NATIVE_CONTROL_OUTPUT.pt',dict(partial['inflight'],q=q));partial['inflight']=None
   b=dict(MovingLive=np.asarray(sp['outputs']['T20_Live_41']),MovingSource=inp['Source'].numpy());dedup={}
   for arm in ['MovingLive','MovingSource']:
    rects=r['rectangles'][arm];assert rects==independent_rectangles(b[arm],W,H);key=rectkey(rects)
    if key in dedup:observations[arm]=dict(dedup[key],rectangles=rects,guidance=arm,reuse_origin_guidance=dedup[key]['guidance'],reused='identical frozen rectangle sequence');continue
    check();view=crop(frames,rects);vhash=digest(view)
    if len(files)<2:
     re=np.stack([cv2.resize(frames[i][v[1]:v[3],v[0]:v[2]],(224,224),interpolation=cv2.INTER_LINEAR)for i,v in enumerate(independent_rectangles(b[arm],W,H))]);assert np.array_equal(view,re);pixelchecks+=1;del re
    if vhash in dedup:observations[arm]=dict(dedup[vhash],rectangles=rects,guidance=arm,reuse_origin_guidance=dedup[vhash]['guidance'],reused='identical actual RGB view');dedup[key]=observations[arm];del view;continue
    vr=dict(row,input=dict(row['input'],width=224,height=224));t=time.perf_counter();torch.cuda.reset_peak_memory_stats();view_attempts+=1;partial['inflight']=dict(stage='View_Source_entered',arm=arm,rectangles=rects,raw_RGB_sha256=vhash);batch,recs,base=forward(view,vr);view_completed+=1;partial['inflight']=dict(stage='View_Source_returned',arm=arm,rectangles=rects,raw_RGB_sha256=vhash,logits=[v.detach().cpu()for v in base.zero['logits']],actions=[v.detach().cpu()for v in base.zero['actions']],boxes=base.zero['boxes'].detach().cpu(),records=recs);check();ll=[v.cpu()for v in base.zero['logits']];actions=[v.cpu()for v in base.zero['actions']];grid,q=mass_table(probabilities(recs,ll));assert grid==ids;shape=list(batch['videos'].tensors.shape);assert shape==[65,3,224,224];peak=torch.cuda.max_memory_allocated();peakalloc=max(peakalloc,peak);peakreserve=max(peakreserve,torch.cuda.max_memory_reserved());base.restore(base.initial);del batch,base;gc.collect();torch.cuda.empty_cache();model_times.append(time.perf_counter()-t);obs=dict(guidance=arm,q=q,logits=ll,actions=actions,records=recs,rectangles=rects,raw_RGB_sha256=vhash,shape=list(view.shape),normalized_shape=shape,peak_GPU_allocated=peak,reused=False);observations[arm]=obs;dedup[key]=dedup[vhash]=obs;partial['inflight']=None;del view
   qs=dict(Native=old['q'],HiResAuxMix=strong['outputs']['HiResAuxMix']['q'],UnionCrop0Mix=union['outputs']['Crop0Mix']['q'],MovingLive=observations['MovingLive']['q'],MovingSource=observations['MovingSource']['q']);qs.update(MovingLiveMix=.5*qs['Native']+.5*qs['MovingLive'],MovingSourceMix=.5*qs['Native']+.5*qs['MovingSource']);out={k:dict(q=q,interval=final_mbr(ids,q)['interval'])for k,q in qs.items()};assert set(out)==set(ARMS)and out['Native']['interval']==inp['interval'];assert out['HiResAuxMix']['interval']==strong['outputs']['HiResAuxMix']['interval']and out['UnionCrop0Mix']['interval']==union['outputs']['Crop0Mix']['interval'];tr=dict(E=np.asarray(inp['E']),**{f'Live{s}':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']});assert all(np.array_equal(v,strong['trajectories'][k])for k,v in tr.items())
   files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,outputs=out,trajectories=tr,observations=observations,original_RGB_sha256=digest(frames),original_RGB_shape=list(frames.shape),GT_read=False,new_student_updates=0,spatial_updated=False)));case_times.append(time.perf_counter()-tick-control);cpu_times.append(case_times[-1]-sum(model_times[before:]));put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time(),Source_attempts=counts['Source_attempts'],Source_completed=counts['Source_completed']),True);print('moving_RGB',a,round(time.perf_counter()-beg,2),flush=True)
   if len(files)==2:
    estimate=loading+native_seconds+max(model_times)*p['required_geometry_unique_view_calls']+max(max(cpu_times),.5)*128+60;put(D/'PILOT2.json',dict(case_seconds=case_times,model_seconds=model_times,model_loading=loading,native_control_seconds=native_seconds,estimated_GPU_seconds=estimate,geometry_unique_views=p['required_geometry_unique_view_calls'],independent_actual_RGB_checks=pixelchecks,GT_read=False));assert estimate<900
   partial=None;del frames,inp,sp,old,strong,union,b,observations,dedup,out,qs,tr,ll,actions,obs;gc.collect();torch.cuda.empty_cache()
  check();pins();assert state_hash(m.state_dict())==mh and {n:x.training for n,x in m.named_modules()}==flags and all(not x.requires_grad and x.grad is None for x in m.parameters());assert counts['Source_attempts']==counts['Source_completed']and all(counts[k+'_attempts']==counts[k+'_completed']==2*counts['Source_completed']for k in ['image','motion']);check();put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,weights_and_flags_unchanged=True));success=True;print('sealed128 moving RGB no GT')
 finally:
  for h in hooks:h.remove()
  if not success and partial is not None:save(D/'partials'/f"{partial['alias']}_{time.time_ns()}.pt",dict(partial,actual_calls=dict(counts)))
  if torch.cuda.is_initialized():peakalloc=max(peakalloc,torch.cuda.max_memory_allocated());peakreserve=max(peakreserve,torch.cuda.max_memory_reserved())
  lease_seconds=time.perf_counter()-beg;put(D/'COST.json',dict(GPU_seconds=lease_seconds+3,measured_lease_seconds=lease_seconds,cleanup_reserve_seconds=3,actual_calls=counts,native_control_Source_attempts=native_attempts,native_control_Source_calls=native_completed,new_view_Source_attempts=view_attempts,new_view_Source_calls=view_completed,independent_actual_RGB_checks=pixelchecks,GPU_peak_allocated_bytes=peakalloc,GPU_peak_reserved_bytes=peakreserve,opencv_version=cv2.__version__,opencv_threads=cv2.getNumThreads(),new_DINO=0,new_ReferDINO=0,new_network_training=0,spatial_updates=0,new_student_updates=0));lock.close()

if __name__=='__main__':
 D.mkdir(exist_ok=True,parents=True);lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:globals()[sys.argv[1]]()
 except BaseException as e:put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partial_or_predictions_preserved=bool(list((D/'partials').glob('*.pt'))or list((D/'predictions').glob('*.pt')))));raise
 finally:lock.close()
