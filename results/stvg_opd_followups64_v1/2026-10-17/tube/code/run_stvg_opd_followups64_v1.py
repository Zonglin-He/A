"""Serial GPU worker; no Teacher and no GT accesses."""
from pathlib import Path
import collections,copy,gc,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_followups64_v1 import BASE,read,sha,put,verify,guard,adapted_fitter,schedules
def main(family):
 import torch,numpy as np
 from scripts import stvg_pipeline_speed_v1 as speed,run_stvg_adjacent_motion_student_v1 as h
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case,save_pt
 from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
 from scripts.run_decota_paper_main_v1 import model_for,frames_for
 from scripts.stvg_mllm_feedback_common_v1 import raw_row
 from scripts import run_stvg_table1_source_v1 as src
 from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import state_hash
 verify();sys.addaudithook(guard);install_clean_loader();torch.set_num_threads(2);torch.manual_seed(20261015);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 rr={r['probe_alias']:r for r in read(BASE/'ROSTER_LOCK.json')['records']};engine=speed.Engine('followups64_'+family,True);model=None;current=None;files=[];qualified=set();tick=time.perf_counter()
 for row in read(BASE/'INPUT_LOCK.json')['records']:
  verify();cid=row['case_id'];rec=rr[cid];case,cp=load_case(rec);assert sha(cp)==row['source_sha256'] and sha(ROOT/row['old_student_path'])==row['old_student_sha256'];old=torch.load(ROOT/row['old_student_path'],map_location='cpu',weights_only=False);anchors=h._anchors(old['reference_records'],row['native_frame_ids'],torch);proposal=torch.load(ROOT/row['proposal_path'],map_location='cpu',weights_only=False);assert sha(ROOT/row['proposal_path'])==row['proposal_sha256'];ids=row['native_frame_ids'];arms=dict(P0_F=copy.deepcopy(old['arms']['T']),P0_O=copy.deepcopy(old['arms']['TS_M2']));fits={};readouts={};captured=None;initial=None;current_fits=0;current_qualified=False
  if family=='tube':arms['P0_R']=proposal['proposal']
  names=['constant','coarse_to_fine','reverse'] if family=='anneal' else ['P0_R_to_S'];eligible=bool(anchors) and (family=='anneal' or proposal['completion']['status']=='completed')
  if eligible:
   _,rawmeta=src._validate_roster_record(rec);transient=dict(case,source=src.SOURCE_MODEL_NAME[case['dataset']],_source_case_sha256=case['source_case_sha256'],parent=rawmeta['source']);catalog=h.catalog;h.catalog=lambda _:(transient,{})
   try:captured=engine.capture(dict(case_id=cid,dataset=case['dataset'],source_case_sha256=case['source_case_sha256']),model,current,torch,collections,model_for,frames_for,raw_row,frozen_forward,prediction,state_hash)
   finally:h.catalog=catalog
   model,current=captured[:2];base=captured[9];records=captured[7];assert captured[12]==case['pixel_sha256'];base.restore(base.initial);initial=base.state();assert torch.equal(base.zero['boxes'].detach().cpu(),case['zero']['boxes']);original=old['fits']['M2'];assert original is not None;cfg=dict(original['config']);settings=original['settings'];key=original['seed_key']
   for name in names:
    if family=='anneal':
     sigmas,slots=schedules(cfg['sigma'])[name];fitfun=adapted_fitter('anneal',(sigmas,slots));actual_cfg=dict(cfg,steps=3);reward_anchors=anchors
    else:
     sigmas=slots=None;fitfun=adapted_fitter('tube',None,row['original_anchor_positions'],torch.stack([a['box'] for a in anchors]));actual_cfg=dict(cfg);reward_anchors=[dict(position=i,box=torch.as_tensor(proposal['proposal_cxcywh'][i],dtype=torch.float32),weight=1.) for i in range(len(ids))]
    checkpoint_files=[]
    def checkpoint(payload):
     p=BASE/family/'checkpoints'/cid/name/(str(payload['next_step'])+'.pt');checkpoint_files.append(save_pt(p,payload))
    fit=fitfun(base,initial,reward_anchors,ids,key,actual_cfg,'M2',settings,qualification=False,checkpoint=checkpoint,recipe='lite_fullstep',response_scope='full_grid',event_support_positions=None);current_fits+=1
    assert fit['protection_positions']==row['protection_positions'] and fit['active_parameters']==1792
    if family=='anneal':
     assert [p['sigma'] for p in fit['path']]==sigmas and [p['noise_slot'] for p in fit['path']]==slots
     for p in fit['path']:
      if p['rollback']:assert __import__('scripts.stvg_target_teacher_student_v1',fromlist=['_tree_equal'])._tree_equal(p['optimizer_before'],p['optimizer_after']) and __import__('scripts.stvg_target_teacher_student_v1',fromlist=['_tree_equal'])._tree_equal(p['state_before'],p['state_after'])
     if name=='coarse_to_fine' and case['dataset'] not in qualified and original['path'][0]['gradient'] is not None:
      for field in ('state_before','rollout','gradient','original_adam_delta','filtered_delta'):speed.compare_tree(fit['path'][0][field],original['path'][0][field],'first_stage_original_exact.'+field)
      qualified.add(case['dataset']);current_qualified=True;put(BASE/family/'qualification'/(case['dataset']+'.json'),dict(status='pass',count=1,first_original_sigma_noise_Adam_and_response_exact=True,no_extra_model_fit=True,GT_read=False))
     fit.update(scale_schedule=sigmas,noise_slots=slots,rejected_stage_does_not_stop_later_stages=True)
    else:fit.update(acceptance_positions=row['original_anchor_positions'],original_sparse_protection_preserved=True,dense_reward_reference_count=len(reward_anchors))
    full=speed.full_prefix_readout(base,ids,records,fit['final_state']);assert torch.equal(full['boxes'].cpu(),fit['final_cached_decoder_boxes'].cpu()) and all(torch.equal(a.cpu(),b.cpu()) for a,b in zip(full['logits'],case['zero']['logits']));pred=dict(boxes=h._xyxy(full['boxes']),interval=list(row['final_Temporal_interval']));arms[name]=pred;fits[name]=fit;readouts[name]=full;assert checkpoint_files and len(checkpoint_files)==actual_cfg['steps'];fit['checkpoint_files']=checkpoint_files
   engine.release();del base,records,captured,initial;captured=None;initial=None
  else:
   for name in names:arms[name]=copy.deepcopy(arms['P0_F']);fits[name]=None;readouts[name]=None
  p=BASE/family/'student'/(cid+'.pt');payload=dict(case_id=cid,dataset=case['dataset'],source_case_sha256=case['source_case_sha256'],source_file_sha256=sha(cp),reference_record=row['reference_record'],fixed_protection_positions=row['protection_positions'],original_anchor_positions=row['original_anchor_positions'],frame_ids=ids,arms=arms,fits=fits,readouts=readouts,proposal_completion=proposal['completion'] if family=='tube' else None,actual_fit_count=current_fits,no_reference_noop=not anchors,chart_noop=family=='tube' and proposal['completion']['status']!='completed',GT_read=False);entry=save_pt(p,payload);entry['case_id']=cid;files.append(entry);put(BASE/family/'STATUS.json',dict(status='running',stage='Student',done=len(files),total=64,count=64,GT_read=False),replace=True);del case,old,proposal,payload,fits,readouts,arms;gc.collect()
 if family=='anneal':assert qualified=={'hc2','vidstg'};put(BASE/family/'QUALIFICATION_COMPLETION.json',dict(status='pass',count=2,first_step_original_exact_both_domains=True,GT_read=False))
 engine.receipt();put(BASE/family/'STUDENT_GLOBAL_BARRIER.json',dict(status='complete',count=64,files=files,wall_seconds=time.perf_counter()-tick,GT_read=False));print(family.upper()+'_STUDENT64_COMPLETE',flush=True)

if __name__=='__main__':main(sys.argv[1])
