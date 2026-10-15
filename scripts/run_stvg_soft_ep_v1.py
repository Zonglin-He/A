"""Finite episodic M0/M1/M2 Student comparison using sealed WF Top1 refs."""
import collections,copy,gc,sys,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_reference_recovery_common_v1 import read,write,sha,guard,load_case,save_pt,PARENT
BASE=ROOT/'artifacts/stvg_soft_ep_opd_v1'
QUAL=['HC02','HC14','V04','V09']
def verify():
 d=read(BASE/'RUNTIME.json');assert d['status']=='locked'and not d['GT_read']
 for p,h in {**d['pins'],**d['protected']}.items():assert sha(ROOT/p)==h,p
 return d

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['qualification','development']);a=p.parse_args()
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
 from scripts.stvg_reference_recovery_fit_v1 import fit_reference_recovery,locked_config
 from scripts.stvg_soft_ep_fit_v1 import fit
 verify();sys.addaudithook(guard);install_clean_loader();torch.set_num_threads(2);torch.manual_seed(20261015)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 design=read(BASE/'DESIGN_LOCK.json');settings=design['response_settings'];refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']};roster=read(BASE/'ROSTER_LOCK.json')['records'];parent_files={x['path']:x for x in read(PARENT/'STUDENT_GLOBAL_BARRIER.json')['files']}
 if a.stage=='qualification':roster=sorted([r for r in roster if r['probe_alias']in QUAL],key=lambda r:QUAL.index(r['probe_alias']))
 else:assert read(BASE/'ROOT_QUALIFICATION.json')['status']=='pass'
 model=source=None;files=[];start=time.perf_counter();captures=fits=0
 for rec in roster:
  cid=rec['probe_alias'];path=BASE/'student'/(cid+'.pt')
  if path.exists():
   receipt=read(path.with_suffix('.receipt.json'));assert sha(path)==receipt['sha256']and not receipt['GT_read'];files.append(receipt['file']);continue
  case,source_path=load_case(rec);pp=PARENT/'student'/(cid+'.pt');assert sha(pp)==parent_files[str(pp.relative_to(ROOT))]['sha256'];parent=torch.load(pp,map_location='cpu',weights_only=False);assert not parent['GT_read']and parent['source_case_sha256']==case['source_case_sha256']
  aa=refs[cid]['anchors'];cfg=locked_config(rec['dataset'],'current5');key=stable_fit_key(rec['dataset'],rec['query_key_sha256']);arms={n:copy.deepcopy(parent['arms'][old])for n,old in [('F','F'),('T','T'),('S_M0','S_new_crop'),('TS_M0','TS_new_crop')]};results={};outputs={};baseline=None;source_seconds=0.
  ids=list(map(int,case['frame_ids']));expected=case['source_model_state_sha256'];pixel=case['pixel_sha256']
  if aa or a.stage=='qualification':
   transient=dict(case,source=source_helper.SOURCE_MODEL_NAME[rec['dataset']],_source_case_sha256=case['source_case_sha256']);_,row=source_helper._validate_roster_record(rec);transient['parent']=row['source']
   old_catalog=h.catalog
   def catalog(record):
    assert record['case_id']==cid and record['source_case_sha256']==case['source_case_sha256'];return transient,{}
   h.catalog=catalog
   try:model,source,_,cat,raw,frames,ids,records,batch,base,native,expected,pixel,source_seconds=h._capture_source(dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=case['source_case_sha256']),model,source,torch,collections,model_for,frames_for,raw_row,frozen_forward,prediction,state_hash)
   finally:h.catalog=old_catalog
   captures+=1;assert pixel==case['pixel_sha256']and expected==case['source_model_state_sha256'];assert np.array_equal(arms['F']['boxes'],h._xyxy(native['boxes']))and h._same_logits(native['logits'],case['zero']['logits'])
   initial={k:v.clone()for k,v in base.initial.items()}
   anchors=h._anchors(dict(anchors=aa),ids,torch)
   if aa:
    if a.stage=='qualification':
     baseline=fit_reference_recovery(base,initial,anchors,ids,key,cfg,profile='current5',dataset=rec['dataset'],validate=True,reference_sha256=sha(BASE/'REFERENCE_LOCK.json'))
     oldfit=parent['fits']['new_crop'];assert torch.equal(baseline['final_cached_decoder_boxes'],oldfit['final_cached_decoder_boxes'])and core._tree_equal(baseline['final_state'],oldfit['final_state']);fits+=1
    for arm in ['M1','M2']:
     base.restore(initial);partial=BASE/'partial'/cid/arm;partial.mkdir(parents=True,exist_ok=False)
     def checkpoint(d):save_pt(partial/('step'+str(d['next_step'])+'.pt'),d)
     f=fit(base,initial,anchors,ids,key,cfg,arm,settings,qualification=a.stage=='qualification',checkpoint=checkpoint);fits+=1;base.restore(initial)
     out=h._readout(model,batch,ids,records,f,raw['parses']['subject'],core,torch,transient_native_masks=transient_native_masks,frozen_visual_host=frozen_visual_host,frozen_attention_storage=frozen_attention_storage,inactive_modules_on_host=inactive_modules_on_host,query_subject=query_subject)
     assert state_hash(model.state_dict())==expected and h._same_logits(out['logits'],native['logits'])and torch.equal(out['boxes'].cpu(),f['final_cached_decoder_boxes'])
     results[arm]=f;outputs[arm]=out;arms['S_'+arm]=dict(boxes=h._xyxy(out['boxes']),interval=list(arms['F']['interval']));arms['TS_'+arm]=dict(boxes=h._xyxy(out['boxes']),interval=list(arms['T']['interval']))
  if not aa:
   for arm in ['M1','M2']:results[arm]=None;outputs[arm]=None;arms['S_'+arm]=copy.deepcopy(arms['F']);arms['TS_'+arm]=copy.deepcopy(arms['T'])
  payload=dict(schema='stvg-soft-ep-case-v1',case_id=cid,dataset=rec['dataset'],arms=arms,fits=results,readouts=outputs,M0_qualification_replay=baseline,frame_ids=ids,reference_records=refs[cid],source_capture_seconds=source_seconds,source_case_sha256=case['source_case_sha256'],source_file_sha256=sha(source_path),source_model_state_sha256=expected,pixel_sha256=pixel,parent_student_sha256=sha(pp),source_restored=True,runtime_sha256=sha(BASE/'RUNTIME.json'),GT_read=False)
  item=save_pt(path,payload);receipt=dict(status='complete',case_id=cid,sha256=sha(path),file=item,GT_read=False);write(path.with_suffix('.receipt.json'),receipt);files.append(item)
  write(BASE/'STATUS.json',dict(status='running',stage=a.stage,done=len(files),total=len(roster),new_fit_invocations=fits,source_captures=captures,GT_read=False,time=time.time()),mutable=True)
  print('SOFT_EP',a.stage,cid,'anchors',len(aa),'accepted',{x:y['optimizer_steps']if y else 0 for x,y in results.items()},flush=True);gc.collect();torch.cuda.empty_cache()
 write(BASE/('QUALIFICATION_BARRIER.json'if a.stage=='qualification'else'STUDENT_GLOBAL_BARRIER.json'),dict(status='complete',count=len(files),files=files,stage=a.stage,new_fit_invocations=fits,source_captures=captures,wall_seconds=time.perf_counter()-start,runtime_sha256=sha(BASE/'RUNTIME.json'),GT_read=False))
if __name__=='__main__':main()
