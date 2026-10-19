"""Frozen full-context + registered actual crop-motion temporal capability."""
from pathlib import Path
import sys,os,time,gc,fcntl
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from scripts.run_stvg_temporal_moving_rgb128_v1 import D as MOV,PRE,FIN,DEV,STAGE
D=R/'artifacts/stvg_temporal_registered_motion128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_registered_motion128';EARLY=R/'artifacts/stvg_temporal_early_fusion128_v1'
ARMS=['Native','HiResAuxMix','MovingLive','RGBRegistered','FeatureWarp','RGBRegisteredMix','FeatureWarpMix']

def prepare():
 from vg_tta.stvg_temporal_registered_motion128_v1 import properties
 old=read(MOV/'PLAN_LOCK.json');records=[dict(r,early_sha256=sha(EARLY/'early_inputs'/f"{r['alias']}.pt"),moving_sha256=sha(MOV/'predictions'/f"{r['alias']}.pt"))for r in old['records']];assert len(records)==len({r['parent']for r in records})==128
 own=list(old['pins'])+['protocols/stvg_temporal_registered_motion128_v1.md','vg_tta/stvg_temporal_registered_motion128_v1.py','scripts/run_stvg_temporal_registered_motion128_v1.py','scripts/score_stvg_temporal_registered_motion128_v1.py','scripts/publish_stvg_temporal_registered_motion128_v1.py','vg_tta/stvg_temporal_early_fusion128_v1.py','vg_tta/stvg_temporal_object_memory128_v1.py','results/stvg_error_state/2026-10-20/temporal_moving_rgb128/RESULTS_PUBLIC.json']
 p=dict(records=records,count=128,seeds=[41,42,43],arms=ARMS,primary='RGBRegisteredMix',pure_teacher='RGBRegistered',GPU_seconds_cap=900,CPU_seconds_cap=900,source_checkpoint=old['source_checkpoint'],source_checkpoint_sha256=old['source_checkpoint_sha256'],max_Source_controls=2,max_motion_encodes=260,pins={f:sha(R/f)for f in sorted(set(own))},created_epoch=time.time(),deadline_epoch=time.time()+14400,spatial_frozen=True,not_OPD=True,new_experts=0,new_training=0,new_student_updates=0)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked128 registered RGB motion',flush=True)

def measure():
 import torch,numpy as np,cv2
 from main_method.runtime import resource_guard
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source,decode
 from methods.stvg_referdino_opd_v1._source_forward import frozen_forward,transient_native_masks
 from methods.stvg_referdino_opd_v1._capture import transfer
 from methods.decota_final_simplified_v1.backbone import make_batch,offset_batch
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_temporal_moving_rgb128_v1 import crop,digest
 from vg_tta.stvg_temporal_registered_motion128_v1 import geometry,independent_geometry,feature_crop,register,fuse_motion,decode_prefix,tensor_sha
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 setup();cv2.setNumThreads(2)
 def reject_GT(event,args):
  if event=='open'and args and isinstance(args[0],(str,bytes,Path))and str(args[0]).endswith('/data/hcstvg2_official_metadata/val_v2.json'):raise PermissionError('registered motion measure forbids HC labels')
 sys.addaudithook(reject_GT);p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];lock=lease();beg=time.perf_counter();hooks=[];files=[];partial=None;success=False;times=[];control_seconds=0.;Source_attempts=Source_completed=0;standalone_attempts=standalone_completed=0;peakalloc=peakreserve=0;control_receipt=None
 counts={k:0 for k in ['image_attempts','image_completed','motion_attempts','motion_completed','projection_attempts','projection_completed','fusion_attempts','fusion_completed','decoder_attempts','decoder_completed','head_attempts','head_completed']}
 def check():
  resource_guard();free,total=torch.cuda.mem_get_info();assert free>5*2**30 and time.time()<p['deadline_epoch']and time.perf_counter()-beg+40<900
 def hook_enter(k):
  def f(m,a):
   counts[k+'_attempts']+=1;assert counts[k+'_attempts']<={'image':4,'motion':260,'projection':260,'fusion':772,'decoder':1552,'head':776}[k]
  return f
 def hook_leave(k):
  def f(m,a,o):
   counts[k+'_completed']+=1
   if k=='head'and partial is not None and partial.get('inflight')is not None:partial['inflight'].setdefault('returned_head_logits',[]).append(o.detach().cpu())
  return f
 try:
  resource_guard();free,total=torch.cuda.mem_get_info();assert free>4*2**30;fraction=min(.8,(free-5*2**30)/total);assert fraction>.4;torch.cuda.set_per_process_memory_fraction(fraction);check();m=load_source();mh=state_hash(m.state_dict());flags={n:v.training for n,v in m.named_modules()};assert not any(flags.values());loading=time.perf_counter()-beg
  for module,k in [(m.vis_encoder,'image'),(m.vid,'motion'),(m.input_proj2,'projection'),(m.ground_encoder.encoder,'fusion'),(m.ground_decoder,'decoder'),(m.temp_embed,'head')]:hooks.extend([module.register_forward_pre_hook(hook_enter(k)),module.register_forward_hook(hook_leave(k))])
  for r in p['records']:
   check();tick=time.perf_counter();ctrl=0.;a=r['alias']
   for f,h in [(EARLY/'early_inputs'/f'{a}.pt',r['early_sha256']),(MOV/'predictions'/f'{a}.pt',r['moving_sha256']),(PRE/'inputs'/f'{a}.pt',r['input_sha256']),(FIN/'Live/predictions'/f'{a}.pt',r['spatial_sha256']),(DEV/'priors'/f'{a}.pt',r['prior_sha256']),(STAGE/'predictions'/f'{a}.pt',r['native_stage_sha256']),(R/r['subject_path'],r['subject_sha256'])]:assert sha(f)==h
   e=load(EARLY/'early_inputs'/f'{a}.pt');old=load(MOV/'predictions'/f'{a}.pt');inp=load(PRE/'inputs'/f'{a}.pt');prior=load(DEV/'priors'/f'{a}.pt');strong=load(STAGE/'predictions'/f'{a}.pt');row=dict(raw[r['query_ordinal']]);row['parses']=read(R/r['subject_path'])['parses'];frames,ids=decode(row['input']);assert ids==old['frame_ids']and digest(frames)==old['original_RGB_sha256'];rect=r['rectangles']['MovingLive'];view=crop(frames,rect);assert digest(view)==old['observations']['MovingLive']['raw_RGB_sha256'];partial=dict(alias=a,logits={},crop_motion=[],metadata={},GT_read=False);cropcontrol=None
   if not files:
    ct=time.perf_counter();controls={};partial['source_controls']=controls
    for role,pixels,rr,refll,refq in [('Native',frames,row,prior['logits'],old['outputs']['Native']['q']),('Crop',view,dict(row,input=dict(row['input'],width=224,height=224)),old['observations']['MovingLive']['logits'],old['outputs']['MovingLive']['q'])]:
     projections=[];partial['control_projections']=projections;partial['inflight']=dict(stage='Source_control_entered',role=role);h=m.input_proj2.register_forward_hook(lambda mm,aa,oo:projections.append(oo.detach().cpu()));Source_attempts+=1
     try:batch,recs,base=frozen_forward(m,pixels,rr);Source_completed+=1
     finally:h.remove()
     vv=transfer(base.zero,'cpu');partial['inflight']=dict(stage='Source_control_returned',role=role,values=vv,records=recs);check();grid,q=mass_table(probabilities(recs,vv['logits']));assert grid==ids and np.array_equal(q,refq)and all(torch.equal(x,y)for x,y in zip(vv['logits'],refll));assert len(projections)==2
     if role=='Native':
      assert torch.equal(vv['boxes'],inp['Source'])
      for off,pr in enumerate(projections):
       cap=e['captured_encoder_inputs'][off];hh,ww=e['original_views'][off]['info']['fea_map_size'];want=cap['src'][-hh*ww:].permute(1,2,0).reshape(pr.shape);assert torch.equal(pr,want)
     else:cropcontrol=projections
     controls[role]=dict(projected_motion=projections,values=vv,records=recs,q=q,GT_read=False);base.restore(base.initial);del base,batch;gc.collect();torch.cuda.empty_cache();partial['inflight']=None
    control_receipt=save(D/'SOURCE_CONTROLS.pt',controls);put(D/'SOURCE_CONTROLS.json',dict(status='pass',Native_boxes_logits_q_raw_motion_exact=True,Crop_logits_q_exact_R28=True,GT_read=False));ctrl=control_seconds=time.perf_counter()-ct;partial.pop('control_projections',None)
   # Only existing Swin and projection see the crop. No new ResNet path.
   vr=dict(row['input'],height=224,width=224);batch=make_batch(view,ids,vr,m);batch['videos']=batch['videos'].to('cpu');motion=[]
   for off in [0,1]:
    check();vb=offset_batch(batch,off);vb['videos']=vb['videos'].to('cuda');standalone_attempts+=1;partial['inflight']=dict(stage='Swin_entered',offset=off)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16),transient_native_masks(m):features=m.vid(vb['videos'].tensors,len(vb['videos'].tensors))
    partial['inflight']=dict(stage='Swin_returned',offset=off,raw_features=transfer(features,'cpu'));check()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):pm=m.input_proj2(features['3'])
    pm=pm.detach().cpu();motion.append(pm);partial['crop_motion']=motion;standalone_completed+=1;partial['inflight']=None;check();assert pm.shape==(len(ids[off::2]),256,7,7)
    if cropcontrol is not None:assert torch.equal(pm,cropcontrol[off])
    del features,pm,vb;gc.collect();torch.cuda.empty_cache()
   del batch;allviews={k:[]for k in ['Native','RGBRegistered','FeatureWarp']};partial['completed_views']=allviews;metadata={};newlogits={}
   for off in [0,1]:
    cap=e['captured_encoder_inputs'][off];ov=e['original_views'][off];hh,ww=ov['info']['fea_map_size'];l=hh*ww;src=cap['src'];native=src[-l:].permute(1,2,0).reshape(len(ids[off::2]),256,hh,ww);valid=(~cap['kwargs']['src_key_padding_mask'][:,-l:]).reshape(-1,hh,ww);alpha,grid,cg=geometry(rect[off::2],row['input']['width'],row['input']['height'],valid);aa,gg=independent_geometry(rect[off::2],row['input']['width'],row['input']['height'],valid);assert np.array_equal(aa,alpha.numpy())and np.array_equal(gg,grid.numpy());fwlocal=feature_crop(native,cg,tuple(motion[off].shape[-2:])).to(motion[off].dtype);registered=register(native,motion[off],alpha,grid);fw=register(native,fwlocal,alpha,grid)
    metadata[str(off)]=dict(native_dtype=str(native.dtype),local_dtype=str(motion[off].dtype),FeatureWarp_local_dtype=str(fwlocal.dtype),alpha=alpha,grid=grid,valid=valid,native_motion_sha256=tensor_sha(native),new_motion_sha256=tensor_sha(motion[off]),RGBRegistered_sha256=tensor_sha(registered),FeatureWarp_sha256=tensor_sha(fw),FeatureWarp_local_sha256=tensor_sha(fwlocal),conditions={})
    for arm,mm in [('Native',None),('RGBRegistered',registered),('FeatureWarp',fw)]:
     check();partial['inflight']=dict(stage='fusion_entered',arm=arm,offset=off);nv,meta=fuse_motion(m,cap,ov,mm,partial['inflight']);partial['inflight']=dict(stage='fusion_returned',arm=arm,offset=off,prefix=nv['prefix'],metadata=meta);check()
     if arm=='Native':assert torch.equal(nv['prefix'],ov['prefix'])
     allviews[arm].append(nv);metadata[str(off)]['conditions'][arm]=meta;partial['metadata']=metadata;partial['inflight']=None
   for arm,vv in allviews.items():
    check();partial['inflight']=dict(stage='two_pass_decode_entered',arm=arm);values=decode_prefix(m,vv,len(ids));partial['inflight']=dict(stage='two_pass_decode_returned',arm=arm,values=values);newlogits[arm]=values['logits'];partial['logits']=newlogits;check()
    if arm=='Native':assert all(torch.equal(x,y)for x,y in zip(values['logits'],prior['logits']))
    partial['inflight']=None
   qs={k:old['outputs'][k]['q']for k in ['Native','HiResAuxMix','MovingLive']}
   for arm,ll in newlogits.items():
    gridids,q=mass_table(probabilities(old['observations']['MovingLive']['records'],ll));assert gridids==ids
    if arm=='Native':assert np.array_equal(q,qs['Native'])
    qs[arm]=q
   qs.update(RGBRegisteredMix=.5*qs['Native']+.5*qs['RGBRegistered'],FeatureWarpMix=.5*qs['Native']+.5*qs['FeatureWarp']);outputs={k:dict(q=q,interval=final_mbr(ids,q)['interval'])for k,q in qs.items()};assert set(outputs)==set(ARMS)
   for arm in ['Native','HiResAuxMix','MovingLive']:assert outputs[arm]['interval']==old['outputs'][arm]['interval']
   files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,records=old['observations']['MovingLive']['records'],outputs=outputs,trajectories=old['trajectories'],crop_motion=motion,metadata=metadata,logits=newlogits,RGB_sha256=digest(view),GT_read=False,spatial_updated=False,new_student_updates=0)));times.append(time.perf_counter()-tick-ctrl);put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time(),actual_calls=counts,standalone_motion_completed=standalone_completed),True);print('registered_motion',a,round(time.perf_counter()-beg,2),flush=True)
   if len(files)==2:
    estimate=loading+control_seconds+128*max(times)+60;put(D/'PILOT2.json',dict(case_seconds=times,model_loading=loading,control_seconds=control_seconds,estimated_GPU_seconds=estimate,Native_full_Source_and_prefix_time_exact=True,Crop_Source_and_standalone_Swin_motion_exact=True,scalar_geometry_exact=True,GT_read=False));assert estimate<900
   partial=None;del frames,view,e,old,inp,prior,strong,motion,allviews,metadata,newlogits,outputs,qs,cropcontrol,src,native,registered,fw,fwlocal,nv,vv,values;gc.collect();torch.cuda.empty_cache()
  check();assert all(sha(R/f)==h for f,h in p['pins'].items());assert state_hash(m.state_dict())==mh and {n:v.training for n,v in m.named_modules()}==flags and all(not v.requires_grad and v.grad is None for v in m.parameters());assert Source_attempts==Source_completed==2 and standalone_attempts==standalone_completed==256
  for k in ['image','motion','projection','fusion','decoder','head']:assert counts[k+'_attempts']==counts[k+'_completed']
  assert counts['image_completed']==4 and counts['motion_completed']==counts['projection_completed']==260 and counts['fusion_completed']==772 and counts['decoder_completed']==1552 and counts['head_completed']==776;check();put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,weights_and_flags_unchanged=True,source_control_file=control_receipt,source_control_summary_sha256=sha(D/'SOURCE_CONTROLS.json')));success=True;print('sealed128 registered motion beforeGT',flush=True)
 finally:
  for h in hooks:h.remove()
  if not success and partial is not None:save(D/'partials'/f"{partial['alias']}_{time.time_ns()}.pt",dict(partial,actual_calls=counts,Source_attempts=Source_attempts,Source_completed=Source_completed,standalone_attempts=standalone_attempts,standalone_completed=standalone_completed))
  if torch.cuda.is_initialized():peakalloc=max(peakalloc,torch.cuda.max_memory_allocated());peakreserve=max(peakreserve,torch.cuda.max_memory_reserved())
  seconds=time.perf_counter()-beg;put(D/'COST.json',dict(GPU_seconds=seconds+3,measured_lease_seconds=seconds,cleanup_reserve_seconds=3,actual_calls=counts,Source_control_attempts=Source_attempts,Source_control_completed=Source_completed,standalone_motion_attempts=standalone_attempts,standalone_motion_completed=standalone_completed,GPU_peak_allocated_bytes=peakalloc,GPU_peak_reserved_bytes=peakreserve,new_DINO=0,new_ReferDINO=0,new_network_training=0,spatial_updates=0,new_student_updates=0));lock.close()

if __name__=='__main__':
 D.mkdir(parents=True,exist_ok=True);lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:globals()[sys.argv[1]]()
 except BaseException as e:put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partial_or_predictions_preserved=bool(list((D/'partials').glob('*.pt'))or list((D/'predictions').glob('*.pt')))));raise
 finally:lock.close()
