"""Matched source-reset spatial fits and complete decoder readouts, fixed temporal winner."""
import sys,os,time,gc,collections,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_evidence_common_v1 import *
def main():
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
 from scripts.stvg_weightfree_coordinate_fit_v1 import fit_validated
 verify_runtime();sys.addaudithook(guard);phase=sys.argv[1];
 if phase=='confirmation'and read(BASE/'FROZEN_METHOD_LOCK.json')['output_preservation']:
  from scripts.stvg_evidence_spatial_preservation_v1 import fit_preserved as fit_validated
 cohort='confirmation'if phase=='confirmation'else'development';b=BASE/phase;refs={r['case_id']:r for r in read(b/'REFERENCE_LOCK.json')['records']};selection=read(C/'SPATIAL_SELECTION_LOCK.json');variants=read(b/'VARIANT_LOCK.json')['variants'];install_clean_loader();torch.set_num_threads(2);torch.manual_seed(20261014);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model=None;source=None;files=[];start=time.perf_counter()
 for rec in roster(cohort):
  cid=rec['probe_alias'];case,cp=load_case(rec);ref=refs[cid];temp=torch.load(b/'temporal'/(cid+'.pt'),map_location='cpu',weights_only=False);iv=temp['selected_interval'];arms={'F':native_arm(case),'T':native_arm(case,iv)};fits={};readouts={};receipts={};capture_s=0.;ids=list(case['frame_ids']);pixel=case['pixel_sha256'];expected=case['source_model_state_sha256']
  if any(ref[v]['anchors']for v in variants):
   transient=dict(case);transient['source']=source_helper.SOURCE_MODEL_NAME[rec['dataset']];transient['_source_case_sha256']=case['source_case_sha256'];_,row=source_helper._validate_roster_record(rec);transient['parent']=row['source']
   def catalog(record):
    assert record['case_id']==cid and record['source_case_sha256']==case['source_case_sha256'];return transient,{}
   h.catalog=catalog;rr=dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=case['source_case_sha256']);model,source,_,cat,raw,frames,ids,records,batch,base,native,expected,pixel,capture_s=h._capture_source(rr,model,source,torch,collections,model_for,frames_for,raw_row,frozen_forward,prediction,state_hash)
   assert list(ids)==list(case['frame_ids'])and pixel==case['pixel_sha256']and expected==case['source_model_state_sha256']and np.array_equal(arms['F']['boxes'],h._xyxy(native['boxes']))and h._same_logits(native['logits'],case['zero']['logits']);initial={k:v.clone()for k,v in base.initial.items()}
  for v in variants:
   cfg=dict(selection['configs'][rec['dataset']]);reuse=False
   if phase=='round2_prompt009':
    parentp=BASE/'round2_prompt009/student'/(cid+'.pt');item=next(x for x in read(BASE/'round2_prompt009/STUDENT_GLOBAL_BARRIER.json')['files']if ROOT/x['path']==parentp);assert sha(parentp)==item['sha256'];parent=torch.load(parentp,map_location='cpu',weights_only=False);assert parent['source_case_sha256']==case['source_case_sha256'];semantic=lambda aa:[{k:x for k,x in a.items()if k!='teacher_output_sha256'}for a in aa]
    if semantic(parent['reference_records'][v]['anchors'])==semantic(ref[v]['anchors']):
     arms['S_'+v]=copy.deepcopy(parent['arms']['S_'+v]);arms['TS_'+v]=copy.deepcopy(parent['arms']['TS_'+v]);receipts[v]=dict(actual_fit_count=0,status='exact_parent_source_reference_geometry_weights_config_seed_reuse',parent_file_sha256=sha(parentp),teacher_receipt_metadata_updated=True)
     if v in parent['fits']:fits[v]=copy.deepcopy(parent['fits'][v]);readouts[v]=copy.deepcopy(parent['readouts'][v]);assert fits[v]['config']==cfg
     continue
   if phase=='preservation010'and v!='preserved':
    originalp=BASE/'round2_prompt009/student'/(cid+'.pt');item=next(x for x in read(BASE/'round2_prompt009/STUDENT_GLOBAL_BARRIER.json')['files']if ROOT/x['path']==originalp);assert sha(originalp)==item['sha256'];oldcase=torch.load(originalp,map_location='cpu',weights_only=False);assert oldcase['reference_records'][v]==ref[v];arms['S_'+v]=copy.deepcopy(oldcase['arms']['S_'+v]);arms['TS_'+v]=copy.deepcopy(oldcase['arms']['TS_'+v]);receipts[v]=dict(actual_fit_count=0,status='exact_round2_control_reuse',source_file_sha256=sha(originalp))
    if v in oldcase['fits']:fits[v]=copy.deepcopy(oldcase['fits'][v]);readouts[v]=copy.deepcopy(oldcase['readouts'][v])
    continue
   if phase=='round2'and v=='old_crop':
    old,oldp,z=fixed_dev_spatial(rec);locked=read(C/'05_lr_refine/SELECTION_LOCK.json')['selected'][rec['dataset']]
    assert ref[v]['anchors']==z['reference_record']['original']['anchors'];arms['S_'+v]=copy.deepcopy(old);arms['TS_'+v]=dict(boxes=copy.deepcopy(old['boxes']),interval=list(iv));receipts[v]=dict(actual_fit_count=0,status='exact_original_source_reference_config_seed_reuse',source_file_sha256=sha(oldp))
    if locked in z['fits']:fits[v]=copy.deepcopy(z['fits'][locked]);readouts[v]=copy.deepcopy(z['readouts'][locked])
    continue
   if not ref[v]['anchors']:
    arms['S_'+v]=copy.deepcopy(arms['F']);arms['TS_'+v]=copy.deepcopy(arms['T']);receipts[v]=dict(actual_fit_count=0,status='no_reference_no_update');continue
   # Exact within-case repeated reference reuse does not constitute another fit.
   prev=None if v=='preserved'else next((u for u in variants[:variants.index(v)]if ref[u]['anchors']==ref[v]['anchors']and u in fits),None)
   if prev is not None:
    arms['S_'+v]=copy.deepcopy(arms['S_'+prev]);arms['TS_'+v]=copy.deepcopy(arms['TS_'+prev]);fits[v]=copy.deepcopy(fits[prev]);readouts[v]=copy.deepcopy(readouts[prev]);receipts[v]=dict(actual_fit_count=0,status='exact_reference_config_seed_reuse',reused_variant=prev);continue
   anchors=h._anchors(ref[v],ids,torch);partial=b/'partial'/cid/v;partial.mkdir(parents=True,exist_ok=False)
   def checkpoint(payload):save_pt(partial/f"step{payload['next_step']}.pt",payload)
   fitter=fit_validated
   if v=='preserved':
    from scripts.stvg_evidence_spatial_preservation_v1 import fit_preserved
    fitter=fit_preserved
   f=fitter(base,{k:x.clone()for k,x in initial.items()},anchors,ids,stable_fit_key(rec['dataset'],rec['query_key_sha256']),cfg,validate=True,reference_sha256=sha(b/'REFERENCE_LOCK.json'),checkpoint_callback=checkpoint)
   out=h._readout(model,batch,ids,records,f,raw['parses']['subject'],core,torch,transient_native_masks=transient_native_masks,frozen_visual_host=frozen_visual_host,frozen_attention_storage=frozen_attention_storage,inactive_modules_on_host=inactive_modules_on_host,query_subject=query_subject)
   assert h._same_logits(out['logits'],native['logits'])and state_hash(model.state_dict())==expected and torch.equal(out['boxes'].cpu(),f['final_cached_decoder_boxes'].cpu());boxes=h._xyxy(out['boxes']);arms['S_'+v]=dict(boxes=boxes,interval=list(case['Native_interval']));arms['TS_'+v]=dict(boxes=boxes,interval=list(iv));fits[v]=f;readouts[v]=out;receipts[v]=dict(actual_fit_count=1,status='actual_fit_full_redecode',accepted_steps=f['optimizer_steps'],config=cfg)
  d=dict(case_id=cid,dataset=rec['dataset'],parent_sha256=rec['parent_sha256'],arms=arms,fits=fits,readouts=readouts,receipts=receipts,reference_records={v:ref[v]for v in variants},frame_ids=ids,pixel_sha256=pixel,source_model_state_sha256=expected,source_case_sha256=case['source_case_sha256'],source_capture_seconds=capture_s,temporal_file_sha256=sha(b/'temporal'/(cid+'.pt')),source_restored=True,GT_read=False)
  item=save_pt(b/'student'/(cid+'.pt'),d);item['actual_fit_count']=sum(x['actual_fit_count']for x in receipts.values());files.append(item);write(b/'STUDENT_STATUS.json',dict(status='running',done=len(files),total=32,GT_read=False),mutable=True);print('student',cid,item['actual_fit_count'],flush=True);del d;gc.collect();torch.cuda.empty_cache()
 write(b/'STUDENT_GLOBAL_BARRIER.json',dict(status='complete',count=32,files=files,wall_seconds=time.perf_counter()-start,GT_read=False))
if __name__=='__main__':main()
