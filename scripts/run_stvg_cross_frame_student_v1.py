"""Hot per-source model, exact episodic reset, one new cross-frame arm per case."""
import sys,os,time,copy,gc,collections,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_cross_frame_common_v1 import BASE,PRIOR,OLD,read,write,sha,guard,verify

def main():
 verify();sys.addaudithook(guard)
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
 from scripts.stvg_cross_frame_validated_fit_v1 import fit_cross_frame
 from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
 def catalog(rec):
  path=ROOT/rec['source_case_path']
  if sha(path)!=rec['source_case_sha256']:raise RuntimeError('source catalog changed')
  return load_source_case(path),{}
 h.catalog=catalog
 install_clean_loader();torch.set_num_threads(2);torch.manual_seed(20261014)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']}
 rows=read(OLD/'ROSTER_LOCK.json')['records'];model=None;source=None;outputs=[];loads=0;started=time.perf_counter()
 def one(rec,model,source):
  cid=rec['case_id'];p=PRIOR/'student'/(cid+'.pt');rc=read(p.with_suffix('.receipt.json'))
  if sha(p)!=rc['sha256']:raise RuntimeError('prior student bytes changed')
  prior=torch.load(p,map_location='cpu',weights_only=False);r=refs[cid]
  if prior['GT_read'] or prior['source_case_sha256']!=rec['source_case_sha256']:raise RuntimeError('prior custody failed')
  arms={k:copy.deepcopy(prior['arms'][k]) for k in ['F','T','raw_iou','T_raw_iou','validated_loss_weighted_iou','T_validated_loss_weighted_iou']}
  if not r['training_anchors']:
   arms['cross_frame']=copy.deepcopy(arms['F']);arms['T_cross_frame']=copy.deepcopy(arms['T'])
   return model,source,dict(case_id=cid,dataset=rec['dataset'],arms=arms,fit=None,readout=None,source_capture_seconds=0.,source_capture_executed=False,actual_fit_count=0,actual_final_redecode_calls=0,GT_read=False,source_restored=True,frame_ids=prior['frame_ids'],pixel_sha256=prior['pixel_sha256'],source_model_state_sha256=prior['source_model_state_sha256'],source_case_sha256=rec['source_case_sha256'],reference_record=r,reference_sha256=sha(BASE/'REFERENCE_LOCK.json'),prior_student_sha256=sha(p),online_carry=False)
  model,source,case,cat,raw,frames,ids,records,batch,base,native,expected,pixel,capture_s=h._capture_source(rec,model,source,torch,collections,model_for,frames_for,raw_row,frozen_forward,prediction,state_hash)
  if prior['frame_ids']!=ids or prior['pixel_sha256']!=pixel or not np.array_equal(np.asarray(prior['arms']['F']['boxes']),h._xyxy(native['boxes'])):raise RuntimeError('hot source capture differs from sealed native')
  anchors=h._anchors({'anchors':r['training_anchors']},ids,torch)
  for a,x in zip(anchors,r['training_anchors']):a['weight']=float(x['reliability']['presence_fraction'])
  vals=h._anchors({'anchors':r['validation_anchors']},ids,torch)
  for a in vals:a['weight']=1.
  cfg=core.load_original_opd_config(rec['dataset']);cfg['steps']=5
  partial=BASE/'partial'/cid/('attempt_'+str(time.time_ns()));partial.mkdir(parents=True,exist_ok=False)
  def checkpoint(payload):
   dst=partial/('step'+str(payload['next_step'])+'.pt')
   if dst.exists():raise FileExistsError(dst)
   torch.save(payload,dst)
  fit=fit_cross_frame(base=base,initial={k:v.detach().clone() for k,v in base.initial.items()},anchors=anchors,validation_anchors=vals,frame_ids=ids,key='caption_query_fixed_space|'+cid,config=cfg,reference_sha256=sha(BASE/'REFERENCE_LOCK.json'),checkpoint_callback=checkpoint)
  out=h._readout(model,batch,ids,records,fit,raw['parses']['subject'],core,torch,transient_native_masks=transient_native_masks,frozen_visual_host=frozen_visual_host,frozen_attention_storage=frozen_attention_storage,inactive_modules_on_host=inactive_modules_on_host,query_subject=query_subject)
  if not h._same_logits(out['logits'],native['logits']) or state_hash(model.state_dict())!=expected:raise RuntimeError('source state or frozen time logits changed')
  if not torch.equal(out['boxes'].cpu(),fit['final_cached_decoder_boxes'].cpu()):raise RuntimeError('full decoder differs from cached final')
  boxes=h._xyxy(out['boxes']);arms['cross_frame']=h._arm(boxes,arms['F']['interval']);arms['T_cross_frame']=h._arm(boxes,arms['T']['interval'])
  result=dict(case_id=cid,dataset=rec['dataset'],arms=arms,fit=fit,readout=out,source_capture_seconds=capture_s,source_capture_executed=True,partial_attempt_path=str(partial.relative_to(ROOT)),actual_fit_count=1,actual_final_redecode_calls=1,GT_read=False,source_restored=True,frame_ids=ids,pixel_sha256=pixel,source_model_state_sha256=expected,source_case_sha256=rec['source_case_sha256'],reference_record=r,reference_sha256=sha(BASE/'REFERENCE_LOCK.json'),prior_student_sha256=sha(p),online_carry=False)
  return model,source,result
 for i,rec in enumerate(rows):
  cid=rec['case_id'];dst=BASE/'student'/(cid+'.pt');receipt=dst.with_suffix('.receipt.json')
  if receipt.exists():
   rr=read(receipt)
   if sha(dst)!=rr['sha256'] or rr['reference_sha256']!=sha(BASE/'REFERENCE_LOCK.json'):raise RuntimeError('resume receipt mismatch')
   outputs.append(dict(case_id=cid,path=str(receipt.relative_to(ROOT)),sha256=sha(receipt)));continue
  write(BASE/'GPU_STATUS.json',dict(status='running',done=i,total=8,case_id=cid,pid=os.getpid(),GT_read=False,time=time.time()),mutable=True)
  before_source=source;t=time.perf_counter();model,source,d=one(rec,model,source);loads+=int(source!=before_source)
  dst.parent.mkdir(exist_ok=True)
  if dst.exists():raise FileExistsError('unsealed partial final requires custody review')
  torch.save(d,dst)
  write(receipt,dict(status='sealed',case_id=cid,path=str(dst.relative_to(ROOT)),sha256=sha(dst),bytes=dst.stat().st_size,reference_sha256=sha(BASE/'REFERENCE_LOCK.json'),actual_fit_count=d['actual_fit_count'],actual_final_redecode_calls=d['actual_final_redecode_calls'],accepted_steps=0 if d['fit'] is None else d['fit']['optimizer_steps'],proposal_steps=0 if d['fit'] is None else d['fit']['proposed_optimizer_steps'],wall_seconds=time.perf_counter()-t,GT_read=False,time=time.time()))
  outputs.append(dict(case_id=cid,path=str(receipt.relative_to(ROOT)),sha256=sha(receipt)));del d;gc.collect();torch.cuda.empty_cache()
  write(BASE/'GPU_STATUS.json',dict(status='running',done=i+1,total=8,case_id=cid,pid=os.getpid(),GT_read=False,time=time.time()),mutable=True)
 write(BASE/'STUDENT_GLOBAL_BARRIER.json',dict(status='complete',count=8,files=outputs,GT_read=False,runtime_sha256=sha(BASE/'STUDENT_RUNTIME.json'),source_model_load_count=loads,wall_seconds=time.perf_counter()-started,time=time.time()))
 write(BASE/'GPU_STATUS.json',dict(status='complete',done=8,total=8,GT_read=False,time=time.time()),mutable=True)
 return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as e:
  write(BASE/'STUDENT_FAILURE_CAPTURE.json',dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()),mutable=True);raise
