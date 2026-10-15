"""32-source frozen-confirmation default versus development-selected OPD."""
import sys,time,copy,gc,collections,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
from scripts.run_stvg_weightfree_expand_v1 import BASE,PARENT,roster,load_case,save_pt
def main():
 for p,h in read(BASE/'STUDENT_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
 sys.addaudithook(guard)
 import torch,numpy as np
 from scripts import run_stvg_adjacent_motion_student_v1 as h
 from scripts import run_stvg_table1_source_v1 as source_helper
 from scripts.run_stvg_table1_student004 import stable_fit_key
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
 from scripts.stvg_weightfree_selected_fit_v1 import fit_validated
 install_clean_loader();torch.set_num_threads(2);torch.manual_seed(20261014);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']};selection=read(PARENT/'METHOD_SELECTION_LOCK.json');model=None;source=None;files=[];start=time.perf_counter()
 for rec in roster():
  cid=rec['probe_alias'];case,cp=load_case(rec);ref=refs[cid];temporal=torch.load(BASE/'temporal'/(cid+'.pt'),map_location='cpu',weights_only=False);assert temporal['source_file_sha256']==sha(cp)
  arms={'F':h._arm(np.asarray(case['native_boxes_xyxy_normalized']),case['Native_interval']),'T':h._arm(np.asarray(case['native_boxes_xyxy_normalized']),temporal['temporal']['final_interval'])};fits={};readouts={};receipts={};capture_s=0.;ids=list(case['frame_ids']);pixel=case['pixel_sha256'];expected=case['source_model_state_sha256']
  if ref['original']['anchors']:
   transient=dict(case);transient['source']=source_helper.SOURCE_MODEL_NAME[rec['dataset']];transient['_source_case_sha256']=case['source_case_sha256'];_,row=source_helper._validate_roster_record(rec);transient['parent']=row['source']
   def catalog(record):
    assert record['case_id']==cid and record['source_case_sha256']==case['source_case_sha256'];return transient,{}
   h.catalog=catalog;rr=dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=case['source_case_sha256'])
   model,source,_,cat,raw,frames,ids,records,batch,base,native,expected,pixel,capture_s=h._capture_source(rr,model,source,torch,collections,model_for,frames_for,raw_row,frozen_forward,prediction,state_hash)
   assert list(ids)==list(case['frame_ids']) and pixel==case['pixel_sha256'] and expected==case['source_model_state_sha256'];assert np.array_equal(np.asarray(arms['F']['boxes']),h._xyxy(native['boxes'])) and h._same_logits(native['logits'],case['zero']['logits'])
   initial={k:v.clone()for k,v in base.initial.items()}
  for v in ['original','selected']:
   cfg=core.load_original_opd_config(rec['dataset']);cfg['steps']=5
   if v=='selected':cfg=dict(selection['configs'][rec['dataset']])
   if v=='selected' and cfg==original_cfg:
    arms[v]=copy.deepcopy(arms['original']);arms['T_'+v]=copy.deepcopy(arms['T_original'])
    if 'original'in fits:fits[v]=copy.deepcopy(fits['original']);readouts[v]=copy.deepcopy(readouts['original'])
    receipts[v]=dict(actual_fit_count=0,status='exact_default_config_reuse');continue
   if v=='original':original_cfg=dict(cfg)
   if not ref[v]['anchors']:
    arms[v]=copy.deepcopy(arms['F']);arms['T_'+v]=copy.deepcopy(arms['T']);receipts[v]=dict(actual_fit_count=0,status='no_reference_no_update');continue
   anchors=h._anchors(ref[v],ids,torch);partial=BASE/'partial'/cid/v;partial.mkdir(parents=True,exist_ok=False)
   def checkpoint(payload):save_pt(partial/f"step{payload['next_step']}.pt",payload)
   fit=fit_validated(base,{k:x.clone()for k,x in initial.items()},anchors,ids,stable_fit_key(rec['dataset'],rec['query_key_sha256']),cfg,validate=True,reference_sha256=sha(BASE/'REFERENCE_LOCK.json'),checkpoint_callback=checkpoint)
   output=h._readout(model,batch,ids,records,fit,raw['parses']['subject'],core,torch,transient_native_masks=transient_native_masks,frozen_visual_host=frozen_visual_host,frozen_attention_storage=frozen_attention_storage,inactive_modules_on_host=inactive_modules_on_host,query_subject=query_subject)
   assert h._same_logits(output['logits'],native['logits']) and state_hash(model.state_dict())==expected and torch.equal(output['boxes'].cpu(),fit['final_cached_decoder_boxes'].cpu());boxes=h._xyxy(output['boxes']);arms[v]=h._arm(boxes,arms['F']['interval']);arms['T_'+v]=h._arm(boxes,arms['T']['interval']);fits[v]=fit;readouts[v]=output;receipts[v]=dict(actual_fit_count=1,status='actual_fit_full_redecode',accepted_steps=fit['optimizer_steps'],config=cfg)
  d=dict(case_id=cid,dataset=rec['dataset'],parent_sha256=rec['parent_sha256'],arms=arms,fits=fits,readouts=readouts,receipts=receipts,source_capture_seconds=capture_s,frame_ids=ids,pixel_sha256=pixel,source_model_state_sha256=expected,source_case_sha256=case['source_case_sha256'],reference_record=ref,source_restored=True,GT_read=False,selection_sha256=sha(PARENT/'METHOD_SELECTION_LOCK.json'))
  f=save_pt(BASE/'student'/(cid+'.pt'),d);f['actual_fit_count']=sum(x['actual_fit_count']for x in receipts.values());f['full_redecode_count']=f['actual_fit_count'];files.append(f);write(BASE/'STUDENT_STATUS.json',dict(status='running',done=len(files),total=32,GT_read=False),mutable=True);print('student',cid,f['actual_fit_count'],flush=True);del d;gc.collect();torch.cuda.empty_cache()
 write(BASE/'STUDENT_GLOBAL_BARRIER.json',dict(status='complete',count=32,files=files,wall_seconds=time.perf_counter()-start,GT_read=False))
if __name__=='__main__':
 try:main()
 except BaseException as e:write(BASE/'STUDENT_FAILURE_CAPTURE.json',dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False));raise
