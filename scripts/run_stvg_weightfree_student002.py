"""Matched original validated OPD, independent full redecode, two proposal budgets."""
import sys,os,time,copy,gc,collections,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_identity_common_v1 import read,write,sha,guard
BASE=ROOT/'artifacts/stvg_weightfree_spatial_v1/consensus002';PRIOR=ROOT/'artifacts/stvg_instance_identity_update_v1';OLD=ROOT/'artifacts/stvg_native_instance_spatial_opd_v1'
def main():
 rt=read(BASE/'STUDENT_RUNTIME.json')
 for p,hsh in rt['pins'].items():assert sha(ROOT/p)==hsh,p
 sys.addaudithook(guard)
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
 from scripts.stvg_instance_validated_fit_v1 import fit_validated
 from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
 def catalog(rec):
  path=ROOT/rec['source_case_path'];assert sha(path)==rec['source_case_sha256'];return load_source_case(path),{}
 h.catalog=catalog;install_clean_loader();torch.set_num_threads(2);torch.manual_seed(20261014);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']};model=None;source=None;outputs=[];start=time.perf_counter()
 for rec in read(OLD/'ROSTER_LOCK.json')['records']:
  cid=rec['case_id'];p=PRIOR/'student'/(cid+'.pt');assert sha(p)==read(p.with_suffix('.receipt.json'))['sha256'];prior=torch.load(p,map_location='cpu',weights_only=False)
  original_path=BASE.parent/'student'/(cid+'.pt');assert sha(original_path)==read(original_path.with_suffix('.receipt.json'))['sha256'];original=torch.load(original_path,map_location='cpu',weights_only=False)
  r=refs[cid];arms={k:copy.deepcopy(prior['arms'][k]) for k in ['F','T','validated_loss_weighted_iou','T_validated_loss_weighted_iou']};fits={};receipts={};readouts={};capture_s=0.
  if any(r[v]['anchors'] for v in ['consensus8','nested8','consensus_nested8']):
   model,source,case,cat,raw,frames,ids,records,batch,base,native,expected,pixel,capture_s=h._capture_source(rec,model,source,torch,collections,model_for,frames_for,raw_row,frozen_forward,prediction,state_hash)
   assert prior['frame_ids']==ids and prior['pixel_sha256']==pixel and np.array_equal(np.asarray(prior['arms']['F']['boxes']),h._xyxy(native['boxes']))
   initial={k:v.detach().clone()for k,v in base.initial.items()};cfg=core.load_original_opd_config(rec['dataset']);cfg['steps']=5
  for variant in ['consensus8','nested8','consensus_nested8']:
   def signature(aa):return [(a['position'],a['physical_frame_id'],a['box_cxcywh'],a['weight'])for a in aa]
   reused=False
   sources=[('original_diverse8',original['reference_record']['diverse8'],original['arms']['diverse8'],original['arms']['T_diverse8'],original['fits'].get('diverse8'),original['readouts'].get('diverse8'))]
   sources += [(name,r[name],arms[name],arms['T_'+name],fits.get(name),readouts.get(name))for name in receipts]
   for label,ref,sp,ts,ft,rd in sources:
    if signature(ref['anchors'])==signature(r[variant]['anchors']):
     arms[variant]=copy.deepcopy(sp);arms['T_'+variant]=copy.deepcopy(ts)
     if ft is not None:fits[variant]=copy.deepcopy(ft);readouts[variant]=copy.deepcopy(rd)
     receipts[variant]=dict(status='exact_geometry_weight_input_seed_fit_reuse',actual_fit_count=0,accepted_steps=0,reused_from=label,original_student_sha256=sha(original_path));reused=True;break
   if reused:continue
   if not r[variant]['anchors']:
    arms[variant]=copy.deepcopy(arms['F']);arms['T_'+variant]=copy.deepcopy(arms['T']);receipts[variant]=dict(status='no_reference_no_update',actual_fit_count=0,accepted_steps=0);continue
   anchors=h._anchors(r[variant],ids,torch)
   partial=BASE/'partial'/cid/variant;partial.mkdir(parents=True,exist_ok=False)
   def checkpoint(payload):
    out=partial/f'step{payload["next_step"]}.pt';assert not out.exists();torch.save(payload,out)
   fit=fit_validated(base=base,initial={k:v.clone()for k,v in initial.items()},anchors=anchors,frame_ids=ids,key='caption_query_fixed_space|'+cid,config=cfg,reference_sha256=sha(BASE/'REFERENCE_LOCK.json'),checkpoint_callback=checkpoint,validate=True)
   out=h._readout(model,batch,ids,records,fit,raw['parses']['subject'],core,torch,transient_native_masks=transient_native_masks,frozen_visual_host=frozen_visual_host,frozen_attention_storage=frozen_attention_storage,inactive_modules_on_host=inactive_modules_on_host,query_subject=query_subject)
   assert h._same_logits(out['logits'],native['logits']) and state_hash(model.state_dict())==expected
   assert torch.equal(out['boxes'].cpu(),fit['final_cached_decoder_boxes'].cpu())
   boxes=h._xyxy(out['boxes']);arms[variant]=h._arm(boxes,arms['F']['interval']);arms['T_'+variant]=h._arm(boxes,arms['T']['interval']);fits[variant]=fit;readouts[variant]=out;receipts[variant]=dict(status='actual_fit_and_full_redecode',actual_fit_count=1,accepted_steps=fit['optimizer_steps'],proposed_steps=fit['proposed_optimizer_steps'])
  d=dict(case_id=cid,dataset=rec['dataset'],arms=arms,fits=fits,readouts=readouts,receipts=receipts,source_capture_seconds=capture_s,frame_ids=prior['frame_ids'],pixel_sha256=prior['pixel_sha256'],source_model_state_sha256=prior['source_model_state_sha256'],source_case_sha256=rec['source_case_sha256'],reference_record=r,GT_read=False,source_restored=True,online_carry=False)
  dst=BASE/'student'/(cid+'.pt');dst.parent.mkdir(exist_ok=True);assert not dst.exists();torch.save(d,dst)
  rr=dict(status='sealed',path=str(dst.relative_to(ROOT)),sha256=sha(dst),bytes=dst.stat().st_size,actual_fit_count=sum(x['actual_fit_count']for x in receipts.values()),full_redecode_count=sum(x['actual_fit_count']for x in receipts.values()),GT_read=False);write(dst.with_suffix('.receipt.json'),rr);outputs.append(rr)
  del d;gc.collect();torch.cuda.empty_cache();write(BASE/'STUDENT_STATUS.json',dict(status='running',done=len(outputs),total=8,GT_read=False),mutable=True);print(cid,receipts,flush=True)
 write(BASE/'STUDENT_GLOBAL_BARRIER.json',dict(status='complete',count=8,files=outputs,GT_read=False,runtime_sha256=sha(BASE/'STUDENT_RUNTIME.json'),wall_seconds=time.perf_counter()-start))
if __name__=='__main__':
 try:main()
 except BaseException as e:write(BASE/'STUDENT_FAILURE_CAPTURE.json',dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False));raise
