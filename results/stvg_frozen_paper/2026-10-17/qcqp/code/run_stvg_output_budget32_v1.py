"""Independent guarded-budget experiment; actual Student parameter updates."""
from pathlib import Path
import collections,copy,gc,math,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_output_budget_qp_v1 import BASE,read,sha,put,verify,guard,fitter

def qualification(base,fit,old,ids,torch):
 import numpy as np
 from scripts import stvg_pipeline_speed_v1 as speed,stvg_target_teacher_student_v1 as core,stvg_spatial_lite_fit_v1 as lite
 for field in ('state_before','rollout','gradient','original_adam_delta'):speed.compare_tree(fit['path'][0][field],old['path'][0][field],'original_first_stage.'+field)
 p=fit['path'][0];proof=p['solver'];assert proof['status']=='pass';base.restore(base.initial);rp,selected=core._trainable_replay(base,ids,fit['seed_key'],'joint');state=p['state_before'];n=sum(x.numel() for _,x in selected);v=torch.sin(torch.arange(n,device=base.zero['boxes'].device,dtype=torch.float32)+1);v=v/v.norm();A=fit['observation_positions'];U=fit['protection_positions'];sigma=fit['config']['sigma'];pred=np.concatenate([proof['JA'],proof['JU']])@v.cpu().double().numpy();observed=[]
 try:
  for eps in (.002,.001):
   vals=[]
   for sign in (1.,-1.):
    lite._put(rp,selected,state,v*(sign*eps));boxes=rp.values()['boxes'].detach();assert bool(((boxes>0)&(boxes<1)).all());vals.append(torch.cat([torch.logit(boxes[A]).flatten()/(sigma*math.sqrt(len(A))),torch.logit(boxes[U]).flatten()/(sigma*math.sqrt(len(U)))]) if U else torch.logit(boxes[A]).flatten()/(sigma*math.sqrt(len(A))))
   fd=((vals[0]-vals[1])/(2*eps)).cpu().double().numpy();error=float(np.linalg.norm(fd-pred));norm=float(np.linalg.norm(pred));observed.append(dict(h=eps,absolute_error=error,relative_error=error/max(norm,1e-12),pass_gross_agreement=error<=.001+.25*norm))
  assert all(x['pass_gross_agreement'] for x in observed)
 finally:lite._put(rp,selected,state);base.restore(base.initial)
 return dict(status='pass',count=1,original_rollout_gradient_Adam_exact=True,actual_dense_Jacobian_central_differences=observed,new_extra_fit_calls=0,extra_decoder_readouts=4,GT_read=False)

def main():
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
 verify();sys.addaudithook(guard);install_clean_loader();torch.set_num_threads(2);torch.manual_seed(20261015);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;rr={r['probe_alias']:r for r in read(BASE/'ROSTER_LOCK.json')['records']};engine=speed.Engine('output_budget32',True);model=current=None;files=[];qualified=set()
 for row in read(BASE/'INPUT_LOCK.json')['records']:
  verify();cid=row['case_id'];rec=rr[cid];case,cp=load_case(rec);assert sha(cp)==row['source_sha256'] and sha(ROOT/row['old_student_path'])==row['old_student_sha256'];old=torch.load(ROOT/row['old_student_path'],map_location='cpu',weights_only=False);ids=row['native_frame_ids'];anchors=h._anchors(old['reference_records'],ids,torch);arms={'Source_T':copy.deepcopy(old['arms']['T']),'M0':copy.deepcopy(old['arms']['TS_M2'])};fits={};readouts={};nofit=None
  if anchors and row['whole_native_logit_chart_valid']:
   _,rawmeta=src._validate_roster_record(rec);transient=dict(case,source=src.SOURCE_MODEL_NAME[case['dataset']],_source_case_sha256=case['source_case_sha256'],parent=rawmeta['source']);catalog=h.catalog;h.catalog=lambda _:(transient,{})
   try:cap=engine.capture(dict(case_id=cid,dataset=case['dataset'],source_case_sha256=case['source_case_sha256']),model,current,torch,collections,model_for,frames_for,raw_row,frozen_forward,prediction,state_hash)
   finally:h.catalog=catalog
   model,current=cap[:2];base=cap[9];records=cap[7];assert cap[12]==case['pixel_sha256'];original=old['fits']['M2'];base.restore(base.initial);initial=base.state()
   for arm in ('M1','M2'):
    checkpoints=[]
    def checkpoint(payload):checkpoints.append(save_pt(BASE/'checkpoints'/cid/arm/(str(payload['next_step'])+'.pt'),payload))
    fit=fitter(arm)(base,initial,anchors,ids,original['seed_key'],original['config'],'M2',original['settings'],qualification=False,checkpoint=checkpoint,recipe='lite_fullstep',response_scope='full_grid');fit.update(new_QCQP_kind=arm,checkpoint_files=checkpoints);assert fit['protection_positions']==row['protection_positions'];full=speed.full_prefix_readout(base,ids,records,fit['final_state']);assert torch.equal(full['boxes'].cpu(),fit['final_cached_decoder_boxes'].cpu()) and all(torch.equal(a.cpu(),b.cpu()) for a,b in zip(full['logits'],case['zero']['logits']));arms[arm]=dict(boxes=h._xyxy(full['boxes']),interval=list(row['final_Temporal_interval']));fits[arm]=fit;readouts[arm]=full
    if case['dataset'] not in qualified and fit['path'][0]['gradient'] is not None:
     q=qualification(base,fit,original,ids,torch);put(BASE/'qualification'/(case['dataset']+'.json'),q);qualified.add(case['dataset'])
   engine.release();del cap,base,records,initial
  else:
   nofit='no_sparse_reference' if not anchors else 'noninvertible_whole_native_logit_chart'
   for arm in ('M1','M2'):arms[arm]=copy.deepcopy(arms['Source_T']);fits[arm]=None;readouts[arm]=None
  p=BASE/'student'/(cid+'.pt');d=dict(case_id=cid,dataset=case['dataset'],source_case_sha256=case['source_case_sha256'],reference_record=row['reference_record'],frame_ids=ids,arms=arms,fits=fits,readouts=readouts,pre_adaptation_chart_valid=row['whole_native_logit_chart_valid'],no_fit_reason=nofit,GT_read=False);f=save_pt(p,d);f['case_id']=cid;files.append(f);put(BASE/'STATUS.json',dict(status='running',stage='Student',done=len(files),total=32,count=32,GT_read=False),replace=True);del case,old,arms,fits,readouts,d;gc.collect()
 assert qualified=={'hc2','vidstg'};engine.receipt();put(BASE/'QUALIFICATION_COMPLETION.json',dict(status='pass',count=2,original_first_rollout_gradient_Adam_exact=True,actual_Jacobian_fd_pass=True,GT_read=False));put(BASE/'STUDENT_GLOBAL_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False));print('OUTPUT_BUDGET32_STUDENT_COMPLETE')
if __name__=='__main__':main()
