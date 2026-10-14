"""Actual source-reset joint spatial fits and complete full-model final readout."""
import sys,os,json,time,hashlib,traceback,collections,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_caption_query_v1 import catalog,roster,read,write,sha,inference_guard
BASE=ROOT/'artifacts/stvg_video_conditioned_global_referent_v1';OLD=ROOT/'artifacts/stvg_temporal_caption_query_spatial_opd_v1'
def main(phase):
 assert phase in ['method','oracle']
 import torch,numpy as np
 from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
 from scripts.run_decota_paper_main_v1 import model_for,frames_for
 from scripts.stvg_mllm_feedback_common_v1 import raw_row
 from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward,transient_native_masks
 from vg_tta.stvg_opd_frozen_visual_host011 import frozen_visual_host
 from vg_tta.stvg_opd_frozen_attention_lifetime010 import frozen_attention_storage
 from vg_tta.stvg_opd_frozen_module_cpu008 import inactive_modules_on_host
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import state_hash
 from methods.decota_final_simplified_v1.backbone import query_subject
 from scripts import stvg_target_teacher_student_v1 as core
 from scripts.run_stvg_support_aware_spatial_v1 import _xyxy,_same_logits,_direct_step_readback
 from scripts.stvg_video_global_spatial_fit_v1 import fit_spatial_reward_opd
 install_clean_loader();sys.addaudithook(inference_guard);torch.set_num_threads(2);torch.manual_seed(20261014);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 rt=read(BASE/f'{phase.upper()}_STUDENT_RUNTIME.json')
 for p,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/p)==h,p
 refs=read(BASE/f'{phase.upper()}_REFERENCE_LOCK.json');assert refs['status']=='locked' and refs['GT_read']==(phase=='oracle');records=roster();index={r['case_id']:r for r in refs['records']};model=None;current=None;files=[];totalsteps=0;allstart=time.perf_counter()
 for number,rec in enumerate(records,1):
  start=time.perf_counter();case,_=catalog(rec);cid=rec['case_id'];expected=case['source_model_state_sha256'];old=torch.load(OLD/'student'/f'{cid}.pt',map_location='cpu',weights_only=False);assert old['source_case_sha256']==rec['source_case_sha256'] and old['GT_read'] is False
  if current!=case['source']:
   if model is not None:del model;gc.collect();torch.cuda.empty_cache()
   model=model_for(case['source']);current=case['source']
  assert model._fixed_full_state_hash==expected and not model.training
  row=raw_row(case);frames,frame_ids,_=frames_for(rec['dataset'],row,collections.OrderedDict());assert list(frame_ids)==case['frame_ids'] and hashlib.sha256(frames.tobytes()).hexdigest()==case['pixel_sha256'];batch,records0,base=frozen_forward(model,frames,row);native=prediction(base.zero['logits'],base.zero['boxes'],records0,list(frame_ids));initial={k:v.detach().clone()for k,v in base.initial.items()};assert records0==case['records'];assert all(torch.equal(v.cpu(),case['source_light_initial_state'][k])for k,v in initial.items());assert np.array_equal(_xyxy(native['boxes']),old['arms']['F']['boxes']);assert state_hash(model.state_dict())==expected
  arms={'F':old['arms']['F'],'Projection':old['arms']['Projection'],'T':old['arms']['T']};fits={};readouts={};anchorsall={};costs={};config=core.load_original_opd_config(rec['dataset']);config['steps']=5;key='caption_query_fixed_space|'+cid
  for refname,ref in index[cid]['references'].items():
   anchors=[dict(position=int(a['position']),physical_frame_id=int(frame_ids[a['position']]),box=torch.tensor(a['box_cxcywh'],dtype=torch.float32),weight=1.)for a in ref['anchors']];anchorsall[refname]=ref['anchors']
   for algorithm in ['direct','iou','diou']:
    name=refname+'_'+algorithm;td=time.perf_counter()
    def checkpoint(z,name=name):
     dst=BASE/phase/'partial'/cid/name/f"step{z['next_step']}.pt";dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();torch.save(z,dst)
    if algorithm=='direct':fit=core._fit_direct(base,initial,anchors,frame_ids,key,config,scope='joint',checkpoint_callback=checkpoint);trace=_direct_step_readback(core,base,initial,frame_ids,key,fit,torch);fit['exact_step_readback']=trace
    else:fit=fit_spatial_reward_opd(base,initial,anchors,frame_ids,key,config,scope='joint',reward_kind=algorithm,checkpoint_callback=checkpoint,reference_role=refname,reference_sha256=sha(BASE/f'{phase.upper()}_REFERENCE_LOCK.json'))
    totalsteps+=int(fit['optimizer_steps']);assert fit['active_parameters']==1792 and fit['declared_steps']==5
    with torch.no_grad(),transient_native_masks(model),frozen_visual_host(model),frozen_attention_storage(model),inactive_modules_on_host(model),query_subject(model,batch,row['parses']['subject']):out=core._finalize_student_readout(model,batch,frame_ids,records0,fit['final_state'])
    assert torch.equal(out['boxes'].cpu(),fit['final_cached_decoder_boxes'].cpu());assert _same_logits(out['logits'],native['logits']);assert state_hash(model.state_dict())==expected
    arms[name]=dict(boxes=_xyxy(out['boxes']),interval=list(out['physical_interval']));arms['T_'+name]=dict(boxes=_xyxy(out['boxes']),interval=list(old['arms']['T']['interval']));fits[name]=fit;readouts[name]=out;costs[name]=time.perf_counter()-td;torch.cuda.empty_cache()
  out=dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=rec['source_case_sha256'],frame_ids=list(frame_ids),source_model_state_sha256=expected,pixel_sha256=case['pixel_sha256'],source_cold_readback=old['source_cold_readback'],initial_state=core._copy_cpu(initial),source_boxes=base.zero['boxes'].cpu(),source_logits=core._copy_cpu(base.zero['logits']),arms=arms,fits=fits,readouts=readouts,supervision_anchors=anchorsall,configuration=dict(config=config,key=key,scope='joint'),reference_lock_sha256=sha(BASE/f'{phase.upper()}_REFERENCE_LOCK.json'),temporal_exact_reuse_sha256=sha(OLD/'student'/f'{cid}.pt'),source_restored=True,GT_read=phase=='oracle',GT_reference_offline_nondeployment=phase=='oracle',cost=dict(fit_and_readout_seconds=costs,case_end_to_end_seconds=time.perf_counter()-start,source_capture_calls=1,full_model_readouts=len(fits)),GPU_peak_bytes=torch.cuda.max_memory_allocated(),GPU_peak_scope='process_high_water_not_isolated_case')
  dst=BASE/phase/'student'/f'{cid}.pt';dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();torch.save(out,dst);files.append(dict(path=str(dst.relative_to(ROOT)),sha256=sha(dst),bytes=dst.stat().st_size));write(BASE/f'{phase.upper()}_GPU_STATUS.json',dict(status='running',stage=phase+'_spatial_fits',done=number,total=8,pid=os.getpid(),GT_read=phase=='oracle',time=time.time()),mutable=True);del batch,base,frames,fit,fits,readouts,old,out;gc.collect();torch.cuda.empty_cache()
 write(BASE/f'{phase.upper()}_STUDENT_GLOBAL_BARRIER.json',dict(status='complete',count=len(files),files=files,optimizer_steps=totalsteps,runtime_sha256=sha(BASE/f'{phase.upper()}_STUDENT_RUNTIME.json'),reference_lock_sha256=sha(BASE/f'{phase.upper()}_REFERENCE_LOCK.json'),GT_read=phase=='oracle',offline_nondeployment=phase=='oracle',end_to_end_seconds=time.perf_counter()-allstart,time=time.time()))
if __name__=='__main__':
 phase=sys.argv[1]
 try:main(phase)
 except BaseException as e:write(BASE/f'{phase.upper()}_FAILURE_CAPTURE.json',dict(status='actual_failure',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=phase=='oracle',time=time.time()),mutable=True);raise
