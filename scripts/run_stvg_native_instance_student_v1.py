"""Actual source-reset three-arm sparse native-instance Student adaptation."""
import sys,os,copy,collections,time,argparse,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import BASE,read,write,sha,verify,guard,row

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args();verify('student');sys.addaudithook(guard);rec=row(a.case)
 import torch,numpy as np
 from scripts import run_stvg_adjacent_motion_student_v1 as h
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
 from scripts.stvg_native_instance_spatial_fit_v1 import fit_spatial_instance
 from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
 def source_only_catalog(record):
  path=ROOT/record['source_case_path'];assert sha(path)==record['source_case_sha256']
  return load_source_case(path),{}
 # Reuse only the source capture implementation, not its obsolete candidate builder.
 h.catalog=source_only_catalog
 install_clean_loader();torch.set_num_threads(2);torch.manual_seed(20261014);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 ref=next(x for x in read(BASE/'REFERENCE_LOCK.json')['records']if x['case_id']==a.case);assert ref['source_case_sha256']==rec['source_case_sha256']and ref['GT_read']is False
 model,_,case,cat,raw,frames,ids,records,batch,base,native,expected,pixel,capture_s=h._capture_source(rec,None,None,torch,collections,model_for,frames_for,raw_row,frozen_forward,prediction,state_hash)
 boxes=h._xyxy(native['boxes']);ni=list(case['Native_interval']);pi=list(case['Projection_interval']);bounds=list(case['video_bounds'])
 assert (ROOT/rec['qa_student_path']).resolve()==(h.OLD/'student'/(a.case+'.pt')).resolve()
 assert sha(ROOT/rec['qa_student_path'])==rec['qa_student_sha256']
 old=h._load_old_temporal_case(a.case,rec['source_case_sha256'],expected,ids,pixel,boxes,ni,pi,bounds,torch);ta=copy.deepcopy(old['arms']['T']);assert ta['interval']==rec['qa_interval']
 arms={'F':h._arm(boxes,ni),'T':ta};fits={};outs={};receipts={};cfg=core.load_original_opd_config(rec['dataset']);cfg['steps']=5;initial={k:v.detach().clone()for k,v in base.initial.items()};anchors=h._anchors(ref,ids,torch)
 assert ref['no_update']==(not bool(anchors))
 for algorithm in ['direct','iou','reliability_weighted_iou']:
  valid=anchors if algorithm!='reliability_weighted_iou'else[x for x in anchors if x['weight']>0]
  if not valid:
   arms[algorithm]=copy.deepcopy(arms['F']);arms['T_'+algorithm]=copy.deepcopy(ta);receipts[algorithm]=dict(status='exact_no_update',optimizer_steps=0,final_redecode_calls=0);continue
  partial=BASE/'partial'/a.case/algorithm;partial.mkdir(parents=True,exist_ok=True)
  def checkpoint(payload):
   dst=partial/('step'+str(int(payload['next_step']))+'.pt');assert not dst.exists();torch.save(dict(payload),dst)
  started=time.perf_counter()
  fit=fit_spatial_instance(base=base,initial={k:v.detach().clone()for k,v in initial.items()},anchors=anchors,frame_ids=ids,key='caption_query_fixed_space|'+a.case,config=cfg,algorithm=algorithm,checkpoint_callback=checkpoint,reference_role='teacher',reference_sha256=sha(BASE/'REFERENCE_LOCK.json'))
  out=h._readout(model,batch,ids,records,fit,raw['parses']['subject'],core,torch,transient_native_masks=transient_native_masks,frozen_visual_host=frozen_visual_host,frozen_attention_storage=frozen_attention_storage,inactive_modules_on_host=inactive_modules_on_host,query_subject=query_subject)
  assert h._same_logits(out['logits'],native['logits']);assert state_hash(model.state_dict())==expected
  if 'final_cached_decoder_boxes'in fit:assert torch.equal(out['boxes'].cpu(),fit['final_cached_decoder_boxes'].cpu())
  b=h._xyxy(out['boxes']);arms[algorithm]=h._arm(b,ni);arms['T_'+algorithm]=h._arm(b,ta['interval']);fits[algorithm]=fit;outs[algorithm]=out
  receipts[algorithm]=dict(status='actual_updated_full_redecode',optimizer_steps=int(fit['optimizer_steps']),final_redecode_calls=1,seconds=time.perf_counter()-started)
 result=dict(case_id=a.case,dataset=rec['dataset'],source_case_sha256=rec['source_case_sha256'],source_model_state_sha256=expected,frame_ids=ids,pixel_sha256=pixel,Native_interval=ni,Projection_interval=pi,arms=arms,fits=fits,readouts=outs,reference_record=ref,reference_lock_sha256=sha(BASE/'REFERENCE_LOCK.json'),initial_state=core._copy_cpu(initial),source_boxes=base.zero['boxes'].detach().cpu(),source_logits=core._copy_cpu(native['logits']),source_restored=True,source_cold_readback=dict(status='pass',full_model_state_sha256=expected,pixel_sha256=pixel,source_boxes_logits_bitwise=True),receipts=receipts,configuration=cfg,actual_fit_count=len(fits),optimizer_steps=sum(int(f['optimizer_steps'])for f in fits.values()),actual_final_redecode_calls=len(outs),source_capture_seconds=capture_s,GT_read=False,online_carry=False)
 dst=BASE/'student'/(a.case+'.pt');dst.parent.mkdir(exist_ok=True);assert not dst.exists();torch.save(result,dst);write(dst.with_suffix('.receipt.json'),dict(status='sealed',path=str(dst.relative_to(ROOT)),sha256=sha(dst),bytes=dst.stat().st_size,actual_fit_count=len(fits),optimizer_steps=result['optimizer_steps'],actual_final_redecode_calls=len(outs),GT_read=False,time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(BASE/'STUDENT_FAILURE_CAPTURE.json',dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()),mutable=True);raise
