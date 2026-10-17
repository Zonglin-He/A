"""Matched historical geometry x configuration factorial, episodic throughout."""
from pathlib import Path
import collections,copy,gc,inspect,json,math,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,sha,write,guard
BASE=ROOT/'artifacts/stvg_geometry_config32_v1'
OLD=ROOT/'artifacts/stvg_reference_recovery_v1/Reference_swap'
def put(p,d,replace=False):write(p,d,mutable=replace)
def verify():
 d=read(BASE/'EXECUTION_RUNTIME001.json');assert time.time()<d['deadline_epoch']
 for p,h in d['pins'].items():assert sha(ROOT/p)==h,p
 return d
def setup():
 from scripts.stvg_reference_recovery_fit_v1 import locked_config,source_hashes
 BASE.mkdir(exist_ok=False)
 assert read(OLD/'ROOT_READBACK.json')['status']=='pass' and read(OLD/'FINAL_COMPLETION.json')['status']=='complete'
 for n in ['ROSTER_LOCK.json','REFERENCE_LOCK.json']:(BASE/n).write_bytes((OLD/n).read_bytes())
 rr=read(BASE/'ROSTER_LOCK.json')['records'];refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']};bar=read(OLD/'STUDENT_GLOBAL_BARRIER.json');files=[]
 for r in bar['files']:
  p=ROOT/r['path']
  if p.suffix=='.pt':assert sha(p)==r['sha256'];files.append(r)
 assert len(files)==32
 for r in rr:
  ref=refs[r['case_id']];a,b=ref['common_wf_anchors'],ref['common_dino_anchors'];assert len(a)==len(b)
  for x,y in zip(a,b):assert all(x[k]==y[k] for k in ['position','physical_frame_id','weight'])
 assert sum(len(r['common_wf_anchors']) for r in refs.values())==38 and sum(bool(r['common_wf_anchors']) for r in refs.values())==19
 put(BASE/'BASELINE_BARRIER.json',dict(status='complete',count=32,files=files,exact_short_WF_and_DINO_reuse=True,GT_read=False))
 configs={ds:{p:locked_config(ds,p) for p in ['current5','original']} for ds in ['hc2','vidstg']}
 put(BASE/'METHOD_LOCK.json',dict(status='locked',count=32,per_domain=16,cohort='historical exposed same-frame swap32, not current core256 or independent confirmation',geometry=['WF_common','DINO_common'],profiles=configs,actual_profile_change=['lr','sigma','tau','steps'],same_common_support_positions_weights_Source_Adam_seed_final_time=True,common_anchors=38,with_common_support=19,without_common_support_noop_retained=13,validation_enabled=True,response_filter=False,LN_carry=False,new_Teacher_calls=0,maximum_new_scientific_fits=38,maximum_extra_qualification_fits=2,primary='difference in config-package gain for DINO versus WF on same32 full denominator',GT_read=False))
 paths=[Path(__file__),ROOT/'scripts/run_stvg_geometry_config32_v1.py',BASE/'ROSTER_LOCK.json',BASE/'REFERENCE_LOCK.json',BASE/'BASELINE_BARRIER.json',BASE/'METHOD_LOCK.json',OLD/'ROOT_READBACK.json',OLD/'ROOT_INDEPENDENT_OPTIMIZER_MATH_READBACK.json',OLD/'STUDENT_GLOBAL_BARRIER.json',ROOT/'methods/CURRENT_METHOD.json',ROOT/'methods/PAPER_METHOD.json',ROOT/'scripts/stvg_reference_recovery_fit_v1.py',ROOT/'scripts/run_stvg_reference_swap_student005.py',ROOT/'scripts/review_stvg_reference_carry_optimizer_v1.py']
 pins=source_hashes();pins.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
 put(BASE/'EXECUTION_RUNTIME001.json',dict(status='locked',count=32,pins=pins,deadline_epoch=time.time()+14400,maximum_new_fits=40,GT_read=False));put(BASE/'STATUS.json',dict(status='prepared',count=32,GT_read=False));print('GEOMETRY_CONFIG32_LOCKED_COMMON38_REUSE_SHORT38_NEW_ORIGINAL38')
def worker():
 import numpy as np,torch
 from scripts import stvg_reference_recovery_common_v1 as common,run_stvg_reference_swap_student005 as old,run_stvg_adjacent_motion_student_v1 as h,stvg_target_teacher_student_v1 as core,stvg_pipeline_speed_v1 as speed,run_stvg_table1_source_v1 as src
 from scripts.stvg_reference_recovery_fit_v1 import fit_reference_recovery,locked_config
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
 verify();sys.addaudithook(guard);install_clean_loader();torch.set_num_threads(2);torch.manual_seed(20261014);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']};orig={r['probe_alias']:r for r in read(ROOT/'artifacts/stvg_reference_recovery_v1/Optimizer_replay/ROSTER_LOCK.json')['records']};prior={Path(r['path']).stem:r for r in read(BASE/'BASELINE_BARRIER.json')['files']};files=[];model=current=None;qualified=set();capture_calls=scientific_fits=qualification_fits=full_RGB_qualification_calls=0
 for rec in read(BASE/'ROSTER_LOCK.json')['records']:
  verify();cid=rec['case_id'];ds=rec['dataset'];ref=refs[cid];p=ROOT/prior[cid]['path'];assert sha(p)==prior[cid]['sha256'];before=torch.load(p,map_location='cpu',weights_only=False);case,cp=common.load_case(rec);ids=list(map(int,case['frame_ids']));assert before['reference_records']==ref and before['source_case_sha256']==case['source_case_sha256'];preds={'F':copy.deepcopy(before['arms']['F']),'T':copy.deepcopy(before['arms']['T'])};fits={};outs={};cap=None
  for role in ['WF_common','DINO_common']:
   for prefix in ['S_','TS_']:preds[prefix+role+'_SHORT']=copy.deepcopy(before['arms'][prefix+role])
  if ref['common_wf_anchors']:
   transient=dict(case,source=src.SOURCE_MODEL_NAME[ds],_source_case_sha256=case['source_case_sha256']);_,row=src._validate_roster_record(orig[cid]);transient['parent']=row['source'];saved=h.catalog;h.catalog=lambda _: (transient,{})
   try:cap=h._capture_source(dict(case_id=cid,dataset=ds,source_case_sha256=case['source_case_sha256']),model,current,torch,collections,model_for,frames_for,raw_row,frozen_forward,prediction,state_hash)
   finally:h.catalog=saved
   capture_calls+=1;model,current=cap[:2];records,batch,base,native=cap[7],cap[8],cap[9],cap[10];assert cap[6]==ids and cap[12]==case['pixel_sha256']==before['pixel_sha256'];assert np.array_equal(h._xyxy(native['boxes']),before['arms']['F']['boxes']);initial={k:v.detach().cpu().clone() for k,v in base.initial.items()};key=stable_fit_key(ds,rec['query_key_sha256'])
   if ds not in qualified:
    anchors=old._matched_anchors(ref['common_wf_anchors'],ids,torch);q=fit_reference_recovery(base,initial,anchors,ids,key,locked_config(ds,'current5'),profile='current5',dataset=ds,validate=True,reference_sha256=sha(OLD/'REFERENCE_LOCK.json'));expected=before['fits']['WF_common']
    for name in ['initial_state','path','rounds','final_state','final_cached_decoder_boxes','optimizer_steps','config','seed_key']:speed.compare_tree(q[name],expected[name],'short_qualification.'+name)
    qualification_fits+=1;put(BASE/'qualification'/(ds+'_SHORT_REPLAY.json'),dict(status='pass',count=1,entire_saved_short_gradient_free_math_trace_and_final_state_exact=True,extra_fit_calls=1,GT_read=False))
   for role in ['WF_common','DINO_common']:
    anchors=old._matched_anchors(ref['common_wf_anchors' if role=='WF_common' else 'common_dino_anchors'],ids,torch)
    def checkpoint(payload):return common.save_pt(BASE/'checkpoints'/cid/role/(str(payload['next_step'])+'.pt'),payload)
    f=fit_reference_recovery(base,initial,anchors,ids,key,locked_config(ds,'original'),profile='original',dataset=ds,validate=True,reference_sha256=sha(BASE/'REFERENCE_LOCK.json'),checkpoint_callback=checkpoint);scientific_fits+=1
    out=speed.full_prefix_readout(base,ids,records,f['final_state']);assert torch.equal(out['boxes'].cpu(),f['final_cached_decoder_boxes'].cpu()) and h._same_logits(out['logits'],native['logits'])
    if ds not in qualified and role=='WF_common':
     actual=old._full_readout(h,model,batch,ids,records,f,cap[4]['parses']['subject'],core,torch,transient_native_masks,frozen_visual_host,frozen_attention_storage,inactive_modules_on_host,query_subject);speed.compare_tree(actual,out,'actual_RGB_vs_prefix_readout');full_RGB_qualification_calls+=1;qualified.add(ds);put(BASE/'qualification'/(ds+'_READOUT.json'),dict(status='pass',count=1,actual_full_RGB_vs_recomputed_prefix_output_exact=True,GT_read=False))
    fits[role]=f;outs[role]=out
    for pre in ['S_','TS_']:preds[pre+role+'_ORIGINAL']=dict(boxes=h._xyxy(out['boxes']),interval=list(preds['F' if pre=='S_' else 'T']['interval']))
   assert state_hash(model.state_dict())==case['source_model_state_sha256'];del base,records,batch,initial
  else:
   for role in ['WF_common','DINO_common']:
    fits[role]=outs[role]=None
    for pre in ['S_','TS_']:preds[pre+role+'_ORIGINAL']=copy.deepcopy(preds['F' if pre=='S_' else 'T'])
  dest=BASE/'student'/(cid+'.pt');payload=dict(case_id=cid,dataset=ds,source_case_sha256=case['source_case_sha256'],source_file_sha256=sha(cp),reference_records=ref,frame_ids=ids,arms=preds,fits=fits,readouts=outs,baseline_sha256=prior[cid]['sha256'],GT_read=False);r=common.save_pt(dest,payload);r['case_id']=cid;files.append(r);put(BASE/'STATUS.json',dict(status='running',stage='Student',done=len(files),total=32,count=32,GT_read=False),replace=True);del case,before,payload,fits,outs,cap;gc.collect();torch.cuda.empty_cache()
 assert qualified=={'hc2','vidstg'} and scientific_fits==38 and qualification_fits==2
 put(BASE/'QUALIFICATION_COMPLETION.json',dict(status='pass',count=2,extra_short_fits=qualification_fits,actual_full_RGB_qualification_calls=full_RGB_qualification_calls,exact_short_replay_and_complete_readout=True,GT_read=False));put(BASE/'STUDENT_GLOBAL_BARRIER.json',dict(status='complete',count=32,files=files,new_scientific_fits=scientific_fits,new_extra_qualification_fits=qualification_fits,source_capture_calls=capture_calls,GT_read=False))
def root():
 import torch,numpy as np
 from scripts import review_stvg_reference_carry_optimizer_v1 as audit
 verify();bar=read(BASE/'STUDENT_GLOBAL_BARRIER.json');assert bar['status']=='complete' and bar['count']==32;assert read(BASE/'QUALIFICATION_COMPLETION.json')['status']=='pass'
 code=inspect.getsource(audit._audit_fit).replace('fit["profile"] == "current5"','fit["profile"] in ("current5", "original")').replace('== 5','== config["steps"]');ns=dict(audit.__dict__);exec(compile(code,'variable-profile-step-math-audit','exec'),ns);refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']};rr={r['case_id']:r for r in read(BASE/'ROSTER_LOCK.json')['records']};checks=0
 for r in bar['files']:
  p=ROOT/r['path'];assert sha(p)==r['sha256'];d=torch.load(p,map_location='cpu',weights_only=False);cid=d['case_id'];assert d['reference_records']==refs[cid]
  for role in ['WF_common','DINO_common']:
   raw=refs[cid]['common_wf_anchors' if role=='WF_common' else 'common_dino_anchors'];f=d['fits'][role]
   if not raw:
    assert f is None
    for pre in ['S_','TS_']:assert np.array_equal(d['arms'][pre+role+'_ORIGINAL']['boxes'],d['arms']['F']['boxes'])
   else:
    ns['_audit_fit'](f,raw,rr[cid],f['config'],sha(BASE/'REFERENCE_LOCK.json'),torch,np);o=d['readouts'][role];assert torch.equal(o['boxes'].cpu(),f['final_cached_decoder_boxes'].cpu());checks+=len(f['path'])
   for pre in ['S_','TS_']:assert d['arms'][pre+role+'_ORIGINAL']['interval']==d['arms']['F' if pre=='S_' else 'T']['interval']
 put(BASE/'ROOT_READBACK.json',dict(status='pass',count=32,declared_step_math_checks=checks,original_five_step_auditor_only_profile_and_loop_bound_adapted=True,GT_read=False));put(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=bar['files'],root_sha256=sha(BASE/'ROOT_READBACK.json'),GT_read=False))
def score():
 import torch,numpy as np
 from scipy.stats import t
 from scripts.stvg_reference_recovery_common_v1 import load_case
 from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
 from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
 verify();bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert bar['status']=='complete' and read(BASE/'ROOT_READBACK.json')['status']=='pass';rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['case_id']:load_case(r)[0] for r in rr};targets,bindings=truth._truths_after_seal(cases);off,ind,_=truth._metric_functions();rows=[];usage=[]
 for r in bar['files']:
  assert sha(ROOT/r['path'])==r['sha256'];d=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False);cid=d['case_id'];case=cases[cid];support={a['position'] for a in d['reference_records']['common_wf_anchors']};vals={}
  for a,p in d['arms'].items():
   v,_=metric._score_student_arm(p,case,targets[cid],support,off,ind);v.update(R_vIoU_at_0p3=float(v['m_vIoU']>=.3),R_vIoU_at_0p5=float(v['m_vIoU']>=.5));rows.append(dict(case_id=cid,dataset=case['dataset'],arm=a,metrics=v));vals[a]=v
  for role,f in d['fits'].items():usage.append(dict(case_id=cid,dataset=case['dataset'],reference=role,actual_fit=f is not None,accepted_steps=f['optimizer_steps'] if f else 0,proposed_steps=f['proposed_optimizer_steps'] if f else 0))
  assert all(vals[a]['m_tIoU']==vals['F' if a.startswith('S_') else 'T']['m_tIoU'] for a in vals if a not in ['F','T'])
 summary,paired=subset_aggregate(rows);idx={(r['case_id'],r['arm']):r['metrics'] for r in rows};interactions=[]
 for ds in ['hc2','vidstg']:
  ids=[r['case_id'] for r in rr if r['dataset']==ds]
  for pre in ['S_','TS_']:
   for metricname in ['m_vIoU','m_sIoU_dense_GT']:
    v={cid:100*((idx[cid,pre+'DINO_common_ORIGINAL'][metricname]-idx[cid,pre+'DINO_common_SHORT'][metricname])-(idx[cid,pre+'WF_common_ORIGINAL'][metricname]-idx[cid,pre+'WF_common_SHORT'][metricname])) for cid in ids};mean=math.fsum(v.values())/len(v);e=float(t.ppf(.975,len(v)-1)*np.std(list(v.values()),ddof=1)/math.sqrt(len(v)));interactions.append(dict(dataset=ds,readout=pre,metric=metricname,valid=len(v),unknown=0,mean_delta_pp=mean,paired_source_t95_interval_pp=[mean-e,mean+e],positive=sum(x>1e-6 for x in v.values()),negative=sum(x< -1e-6 for x in v.values()),neutral=sum(abs(x)<=1e-6 for x in v.values()),per_source_delta_pp=v))
 put(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=32,per_domain=16,rows=rows,summary=summary,paired=paired,interactions=interactions,usage=usage,GT_bindings=bindings,GT_read=True,configuration_package_not_steps_only=True,episodic_not_historical_online_reproduction=True,old_short_profiles_not_current_paper_Lite=True,development_not_confirmation=True));put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=32,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=True))
if __name__=='__main__':globals()[sys.argv[1]]()
