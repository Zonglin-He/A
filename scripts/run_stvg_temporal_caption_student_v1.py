"""Actual eight-case shared temporal readout + unchanged spatial Gaussian OPD."""
import os,sys,time,hashlib,traceback,gc,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.stvg_temporal_caption_query_v1 import *

def main():
 import torch
 from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
 from scripts.run_decota_paper_main_v1 import model_for,frames_for
 from scripts.stvg_mllm_feedback_common_v1 import raw_row
 from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward,transient_native_masks
 from vg_tta.stvg_opd_frozen_visual_host011 import frozen_visual_host
 from vg_tta.stvg_opd_frozen_attention_lifetime010 import frozen_attention_storage
 from vg_tta.stvg_opd_frozen_module_cpu008 import inactive_modules_on_host
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import state_hash,detached
 from methods.decota_final_simplified_v1.backbone import query_subject
 from scripts.stvg_temporal_shared_readout_v1 import fit as temporal_fit
 from scripts import stvg_target_teacher_student_v1 as core
 install_clean_loader();sys.addaudithook(inference_guard);torch.set_num_threads(2);torch.manual_seed(20261014);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 rt=read(BASE/'STUDENT_RUNTIME.json')
 for n,h in rt['pins'].items():assert sha(ROOT/n)==h,n
 seal=read(BASE/'TEACHER_GLOBAL_BARRIER.json');assert seal['status']=='complete' and seal['count']==64 and seal['GT_read'] is False
 for f in seal['files']:assert sha(ROOT/f['path'])==f['sha256']
 refs=read(BASE/'SPATIAL_REFERENCE_LOCK.json')['records'];jobs=read(BASE/'CAPTION_JOB_LOCK.json')['records'];files=[];model=None;source=None;count=0
 for rec in roster():
  case,cat=catalog(rec)
  if source!=case['source']:
   if model is not None:del model;gc.collect();torch.cuda.empty_cache()
   model=model_for(case['source']);source=case['source']
  assert model._fixed_full_state_hash==case['source_model_state_sha256'] and not model.training and all(not p.requires_grad for p in model.parameters())
  row=raw_row(case);frames,ids,_=frames_for(rec['dataset'],row,collections.OrderedDict());assert ids==list(case['frame_ids']) and hashlib.sha256(frames.tobytes()).hexdigest()==case['pixel_sha256']
  started=time.perf_counter();batch,records,base=frozen_forward(model,frames,row);native=prediction(base.zero['logits'],base.zero['boxes'],records,ids)
  assert torch.equal(native['boxes'],case['zero']['boxes']) and all(torch.equal(x,y) for x,y in zip(native['logits'],case['zero']['logits'])) and native['physical_interval']==case['Native_interval']
  assert records==case['records'];assert all(torch.equal(v.detach().cpu(),case['source_light_initial_state'][k]) for k,v in base.initial.items())
  coldpath=BASE/'student_sources'/f"{rec['case_id']}.pt";coldpath.parent.mkdir(exist_ok=True);assert not coldpath.exists()
  torch.save(dict(zero=detached(base.zero,'cpu'),temporal_inputs=detached(base.temporal_inputs,'cpu'),records=records,source_case_sha256=rec['source_case_sha256'],GT_read=False),coldpath)
  hidden=[]
  for h,z in zip(base.temporal_inputs,base.zero['logits']):
   assert h.ndim==4 and tuple(h.shape[:2])==(6,1) and h.shape[-1]==256 and h.shape[2]==z.shape[1]
   with torch.no_grad():assert torch.equal(base.head(h)[-1],z)
   hidden.append(h[-1,0].detach().cpu())
  local=[j for j in jobs if j['case_id']==rec['case_id']];rewards=[read(BASE/'teacher'/f"{rec['case_id']}_{j['candidate']['canonical_id']}_alignment.json")['reward'] for j in local]
  temp=temporal_fit(case,hidden,[j['candidate'] for j in local],rewards)
  ref=next(r for r in refs if r['case_id']==rec['case_id']);assert ref['source_case_sha256']==rec['source_case_sha256'] and sha(ROOT/ref['actual_response_path'])==ref['actual_response_sha256']
  anchors=[]
  if ref['candidate'] is not None:
   for pos in ref['observed_positions']:
    b=ref['candidate']['boxes_xyxy_normalized'][pos];x,y,u,v=b;assert 0<=x<u<=1 and 0<=y<v<=1
    anchors.append(dict(position=pos,physical_frame_id=ids[pos],box=torch.tensor([(x+u)/2,(y+v)/2,u-x,v-y],dtype=torch.float32),weight=1.))
  cfg=core.load_original_opd_config(rec['dataset']);cfg['steps']=5
  def save_progress(payload):
   p=BASE/'student_partial'/rec['case_id']/f"step{payload['next_step']}.pt";p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();torch.save(payload,p)
  fit=core._fit_spatial_opd(base,base.initial,anchors,ids,'caption_query_fixed_space|'+rec['case_id'],cfg,checkpoint_callback=save_progress)
  with torch.no_grad(),transient_native_masks(model),frozen_visual_host(model),frozen_attention_storage(model),inactive_modules_on_host(model),query_subject(model,batch,row['parses']['subject']):
   spatial=core._finalize_student_readout(model,batch,ids,records,fit['final_state'])
  assert torch.equal(spatial['boxes'],fit['final_cached_decoder_boxes']) and all(torch.equal(x,y) for x,y in zip(spatial['logits'],native['logits']))
  assert state_hash(model.state_dict())==case['source_model_state_sha256']
  def xyxy(b):
   a=b.detach().cpu().double().numpy();return np.concatenate((a[:,:2]-a[:,2:]/2,a[:,:2]+a[:,2:]/2),1)
  frozen_boxes=xyxy(native['boxes']);space_boxes=xyxy(spatial['boxes']);arms={'F':dict(boxes=frozen_boxes,interval=case['Native_interval']),'Projection':dict(boxes=frozen_boxes,interval=case['Projection_interval']),'T':dict(boxes=frozen_boxes,interval=temp['final_interval']),'S':dict(boxes=space_boxes,interval=case['Native_interval']),'TS':dict(boxes=space_boxes,interval=temp['final_interval'])}
  result=dict(case_id=rec['case_id'],dataset=rec['dataset'],source_case_sha256=rec['source_case_sha256'],frame_ids=ids,arms=arms,temporal_fit=temp,temporal_hidden=hidden,spatial_fit=fit,spatial_native_full_readout=spatial,source_cold_readback={'status':'pass','pixel_sha256':case['pixel_sha256'],'full_model_state_sha256':case['source_model_state_sha256'],'source_boxes_logits_bitwise':True,'hidden_head_bitwise':True,'temporal_feature_shape':[list(h.shape) for h in hidden]},spatial_reference_sha256=ref['actual_response_sha256'],supervised_positions=[a['position'] for a in anchors],full_model_capture_calls=1,full_model_final_readout_calls=1,wall_seconds=time.perf_counter()-started,GPU_peak_bytes=torch.cuda.max_memory_allocated(),GT_read=False,teacher_weight_update=False,spatial_text_loss=False,online_carry=False)
  dst=BASE/'student'/f"{rec['case_id']}.pt";dst.parent.mkdir(exist_ok=True);assert not dst.exists();torch.save(result,dst);files.append(dict(path=str(dst.relative_to(ROOT)),sha256=sha(dst)));count+=1
  write(BASE/'STUDENT_GPU_STATUS.json',dict(status='running',stage='actual_T_and_S_fits_full_readout',done=count,total=8,pid=os.getpid(),GT_read=False,time=time.time()),mutable=True)
  del base,batch,frames,result,fit,temp,spatial,native;gc.collect();torch.cuda.empty_cache()
 if model is not None:del model;gc.collect();torch.cuda.empty_cache()
 write(BASE/'STUDENT_GLOBAL_BARRIER.json',dict(status='complete',files=files,count=8,logical_final_readouts=40,actual_temporal_fit_cases=8,actual_spatial_fit_cases=8,teacher_barrier_sha256=sha(BASE/'TEACHER_GLOBAL_BARRIER.json'),student_runtime_sha256=sha(BASE/'STUDENT_RUNTIME.json'),GT_read=False,time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException as e:write(BASE/'STUDENT_FAILURE_CAPTURE.json',dict(status='actual_failure',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()),mutable=True);raise
