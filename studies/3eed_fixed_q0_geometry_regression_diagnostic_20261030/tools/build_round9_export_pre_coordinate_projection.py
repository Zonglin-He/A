"""Finite R9 public candidate: prepared only until root final publication gates.
No network/Git, model/GPU, target annotation/index or new scoring invocation.
Only public code/protocol/provenance and sealed-derived scalar projections.
"""
from pathlib import Path,PurePosixPath
from functools import lru_cache
import argparse,csv,gzip,hashlib,io,json,math,platform,re,zlib
B=Path(__file__).resolve().parents[1];E=B/'research/round9/experiment';C=B/'crossplatform/round9'
SOURCE='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd';CLIP_REV='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268';CLIP_SAFE='7a7177862f66645c5e923e31e07318fd609769378a197a21903fbbcc07e446b1'
STRICT={'cudnn_allow_tf32':False,'matmul_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}
MODES=('rgb_fulltext','pc_only');NORMAL_ARMS=('source',)+tuple(m+'__'+k for m in MODES for k in ('direct_G','param1','param3','outputspace1','outputspace3'));GT_ARMS=('source',)+tuple('GT_'+g+'__'+k for g in ('C','S','CS') for k in ('direct_target','param1','param3','exactfit'));BRIDGE_ARMS=('source',)+tuple(m+'__pseudo_exactfit' for m in MODES)
STAGES={'normal':{'arms':NORMAL_ARMS,'role':'unlabelled_normal_fixedq0','predstate':'sealed_R9_normal_fixedq0_predictions_and_sourcecaches_before_GT','scorestate':'normal11_fixedq0_scored_postseal','plan':'NORMAL_PLAN_LOCK.json','scorer':'score_normal.py'},'GT':{'arms':GT_ARMS,'role':'label_ORACLE_NOT_TTA','predstate':'sealed_R9_GTcontrolled_samecache_coordinate_capacity_diagnostic','scorestate':'GTcontrolled_samecache_coordinate_capacity_scored_NOT_TTA','plan':'GT_RUN_PLAN_LOCK.json','scorer':'score_gt_controlled.py'},'bridge':{'arms':BRIDGE_ARMS,'role':'post_normal_score_exploratory_NOT_confirmation','predstate':'sealed_post_normal_score_unlabelled_pseudo_exactfit_optimizer_bridge','scorestate':'post_normal_score_exploratory_pseudo_exactfit_bridge_scored','plan':'PSEUDO_BRIDGE_PLAN_LOCK.json','scorer':'score_pseudo_exactfit_bridge.py'}}
CODE_FILES=tuple('crossplatform/round9/'+x for x in ('fixed_q0_geometry.py','head_exact_fit.py','run_diagnostic.py','run_pseudo_exactfit.py','check_fixed_q0_geometry_cpu.py','check_head_exact_fit_cpu.py','check_run_diagnostic_cpu.py','check_pseudo_exactfit_cpu.py'))+tuple('research/round9/evaluation/'+x for x in ('geometry_metrics.py','capacity_metrics.py','score_normal.py','score_gt_controlled.py','score_pseudo_exactfit_bridge.py','audit_normal_unlabelled.py','audit_normal_scored.py','audit_gt_controlled_scored.py','audit_pseudo_exactfit_scored.py','audit_manual_optimizer_supplement.py','check_actual_linear_backward_cpu.py','check_capacity_and_metrics_synthetic.py','check_capacity_array_contract_cpu.py','check_normal_schema_cpu.py','check_pseudo_exactfit_bridge_cpu.py','history/GT_auditor_float32_residual_fix/audit_gt_controlled_scored_v1.py'))+('research/round9/experiment/prepare_inputs.py','research/round9/experiment/review_identity_samples.py','research/round9/experiment/review_identity_samples_pre_io.py','crossplatform/round8/association_geometry.py','crossplatform/round8/semantic_encoder.py','crossplatform/round8/fetch_semantic_assets.py','crossplatform/round8/prepare_safe_semantic_weights.py','crossplatform/round7/regression_adapter.py','crossplatform/round7/camera_projection.py','crossplatform/round7/experiment/runner_helpers.py','crossplatform/round6/factory.py','research/round5/audit/independent_core_v2.py','tools/obb_iou.py','tools/build_round9_export.py')
DOC_FILES=('crossplatform/round9/DIAGNOSTIC_CONTRACT.md','crossplatform/round9/PSEUDO_EXACTFIT_BRIDGE_CONTRACT.md','crossplatform/round8/SEMANTIC_INTERFACE.md','research/round9/evaluation/INDEPENDENT_PROTOCOL_REVIEW.md','research/round9/evaluation/EXECUTION_DRAFT.md','research/round9/experiment/BRIEF.md','research/round6/UPSTREAM_COMPATIBILITY.patch')
STATIC_JSON=tuple('crossplatform/round9/'+x for x in ('CODE_READY.json','CPU_FIXED_Q0_FIXTURE.json','CPU_EXACTFIT_FIXTURE.json','CPU_R9_RUNNER_FIXTURE.json','PSEUDO_EXACTFIT_BRIDGE_READY.json','CPU_PSEUDO_EXACTFIT_BRIDGE_FIXTURE.json'))+tuple('research/round9/evaluation/'+x for x in ('CPU_ACTUAL_BACKWARD_FIXTURE.json','CPU_CAPACITY_ARRAY_CONTRACT_FIXTURE.json','CPU_NORMAL_SCHEMA_FIXTURE.json','CPU_PREPARATION_RECEIPT.json','CPU_SYNTHETIC_FIXTURE.json','CPU_AGGREGATION_FIXTURE.json','CPU_PSEUDO_EXACTFIT_BRIDGE_FIXTURE.json','NORMAL_EVALUATION_CODE_LOCK.json','AUXILIARY_EVALUATION_CODE_LOCK.json','NORMAL_IMPLEMENTATION_REVIEW.json','NORMAL_IMPLEMENTATION_REVIEW_v2.json','PSEUDO_EXACTFIT_IMPLEMENTATION_REVIEW.json','PROTOCOL_DRAFT.json','history/pre_final_normal_evaluation_lock/NORMAL_EVALUATION_CODE_LOCK.json','history/GT_auditor_float32_residual_fix/FAILURE_AND_FIX.json'))+tuple('research/round9/experiment/'+x for x in ('INPUT_PLAN_LOCK.json','COHORT_AVAILABILITY.json','INPUT_RECEIPT.json','GT_CAPACITY_PREPLAN_LOCK.json','GT_PLAN_MATERIALIZATION_RECEIPT.json','OPTIMIZER_BRIDGE_ADDENDUM.json','IDENTITY_OVERLAY_PLAN_LOCK.json','IDENTITY_OVERLAY_PREPARATION.json','IDENTITY_OVERLAY_IO_PREPARATION.json'))
REQUIRED=tuple(k+'_'+suffix for k in STAGES for suffix in ('prediction_seal','scalar_summary','score_release'))+('normal_unlabelled_audit','normal_reduction_audit','GT_capacity_audit','bridge_independent_audit','final_mechanism_review','manual_identity_review_summary','manual_subset_error_summary','optimization_fidelity_diagnostic','manual_optimizer_supplement_review','root_report','root_decision','root_final_review')
COORD_FIELDS=('center_L2_m','size_L2_m','coordinate_Linf_m','center_L2_gt_dimension_normalized','size_relative_abs_mean','native_yaw0_vs_GT_yaw_not_maximalIoU_oracle')
ARM_FIELDS=('iou','Acc25','Acc50','delta_vs_source','selected_center_shift_L2_m','selected_size_changed','status','parameter_steps_executed','nonzero_gradient_steps','nonzero_parameter_update','normal_target_delta_L2_m','normal_target_nonzero','target_delta_le_registered_atol','center_residual_L2_vs_direct_m','center_residual_Linf_vs_direct_m','relative_center_residual_vs_target_shift','coordinate_check_applicable','coordinate_pass','real_backward_and_optimizer_step','nonzero_gradient','selected_changed_vs_source','delta_vs_normal_directG','delta_vs_normal_param3','delta_vs_normal_kernel3')
CAP_FIELDS=('supervised_coordinate_Linf_m','supervised_coordinate_L2_m','all6_coordinate_Linf_vs_direct_m','center_L2_vs_direct_m','size_L2_vs_direct_m')
PRIVATE_MAPS={'source_cache_files','normal_source_cache_files','normal_prediction_files','source_input_paths','weights_relative_private_location','programroot','workspace_root'}
sha=lambda data:hashlib.sha256(data).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def local(rel):
 q=PurePosixPath(rel);assert q.parts and not q.is_absolute() and '..' not in q.parts
 p=B/q.as_posix();assert p.resolve().is_relative_to(B.resolve()) and p.is_file(),rel;return p

def main(args):
 out=args.output.resolve();assert out.is_relative_to(B.resolve()) and not out.exists(),'New bounded output required, never overwrite'
 ready=read(E/'PUBLICATION_READINESS.json')
 for k in ('three_actual_prediction_stages_sealed','three_actual_score_stages_sealed','normal_unlabelled_and_all_actual_independent_audits_passed','GT_ORACLE_and_postscore_bridge_roles_explicit','identity_condition_and_nonblind_manual_review_limits_explicit','root_final_report_decision_review_passed','privacy_reviewed_no_weights_features_GTindices_sidecars_media_captions_coordinates_or_privateIDs'):assert ready.get(k) is True,('Missing final gate',k)
 specs=read(E/'EXPORT_FINAL_SOURCES.json');bound={}
 for k in REQUIRED:
  spec=specs[k];assert set(spec)=={'path','sha256'};p=local(spec['path']);assert file_sha(p)==spec['sha256'];bound[k]=p
 pred={};scores={};scoreseals={}
 for stage,contract in STAGES.items():
  sp=bound[stage+'_prediction_seal'];s=read(sp);pred[stage]=s;assert s['state']==contract['predstate'] and s['n_samples']==s['n_prediction_files']==160 and s['n_arms']==len(contract['arms'])
  assert s['source_checkpoint_sha256']==SOURCE and s['strict_FP32_flags']==STRICT and s['program_sha256_base']=='workspace_root'
  assert s['plan_sha256']==file_sha(E/contract['plan'])
  for rel,digest in s['program_sha256'].items():assert file_sha(local(rel))==digest
  for rel,digest in s['files'].items():
   q=PurePosixPath(rel);assert not q.is_absolute() and '..' not in q.parts;assert file_sha(sp.parent/rel)==digest
  p=bound[stage+'_scalar_summary'];ss=read(p.parent/'SEALED.json');v=read(p);scores[stage]=v;scoreseals[stage]=ss
  assert v['state']==contract['scorestate'] and v['n']==ss['n_samples']==160
  for name in ('RESULTS_PRIVATE.json','SCALAR_SUMMARY.json','SCORER_LOCK.json'):assert ss['files'][name]==file_sha(p.parent/name)
  grant=read(bound[stage+'_score_release']);lock=read(p.parent/'SCORER_LOCK.json');assert grant['CPU_SCORE_RELEASE'] is True and lock['release_sha256']==file_sha(bound[stage+'_score_release'])
  assert grant['scorer_program_sha256']==file_sha(B/'research/round9/evaluation'/contract['scorer']) and grant['source_checkpoint_sha256']==SOURCE
  if stage=='normal':assert grant['prediction_seal_sha256']==file_sha(sp) and grant['complete_normal_predictions_sealed_before_GT'] and s['all_11arms_and_sourcecache_sealed_before_GT'] and s['all_normal_arm_queries_fixed_q0'] and s['grounder_restored_and_CLIP_frozen']
  if stage=='GT':assert grant['GT_prediction_seal_sha256']==file_sha(sp) and grant['normal_sealed_before_GT_injection'] and s['GTcontrolled_not_unlabelled_TTA_or_taskIoU_upperbound'] and s['all_exactfit_coordinate_checks_pass'] and v['all_exactfit_within_registered5e5'] and s['source_model_parameters_buffers_grads_modes_restored']
  if stage=='bridge':assert grant['bridge_prediction_seal_sha256']==file_sha(sp) and s['post_normal_score_exploratory_optimizer_bridge'] and s['same_normal_directG_targets_no_new_G'] and s['no_GT_scored_files_RGB_PC_CLIP_G_or_backbone_forward_consumed'] and s['not_unique_TTA_benefit_or_heldout_claim'] and s['all_applicable_coordinate_checks_pass'] and s['source_state_restored']
 assert pred['normal']['CLIP_resource']['revision']==CLIP_REV and pred['normal']['CLIP_resource']['derived_safetensors_sha256']==CLIP_SAFE
 assert len({v['source_model_state_fingerprint'] for v in pred.values()})==1
 for stage in ('GT','bridge'):assert pred[stage]['normal_prediction_seal_sha256']==file_sha(bound['normal_prediction_seal']) and pred[stage]['normal_source_cache_files']==pred['normal']['source_cache_files'] and pred[stage]['all_query_indices_same_normal_q0']
 u=read(bound['normal_unlabelled_audit']);assert u['state']=='independent_R9_normal11_unlabelled_audit_pass' and u['prediction_seal_sha256']==file_sha(bound['normal_prediction_seal']) and u['n']==160 and u['GT_metadata_index_or_target_annotations_accessed'] is False
 r=read(bound['normal_reduction_audit']);assert r['state']=='independent_R9_normal11_scored_reduction_and_coordinate_audit_pass' and r['n']==160 and r['results_sha256']==file_sha(bound['normal_scalar_summary'].parent/'RESULTS_PRIVATE.json') and r['scored_seal_sha256']==file_sha(bound['normal_scalar_summary'].parent/'SEALED.json')
 g=read(bound['GT_capacity_audit']);assert g['state']=='independent_oracle13_samecache_coordinate_and_reduction_audit_pass' and g['n']==160 and g['GT_prediction_seal_sha256']==file_sha(bound['GT_prediction_seal']) and g['GT_results_sha256']==file_sha(bound['GT_scalar_summary'].parent/'RESULTS_PRIVATE.json') and g['same_normal_forward_query_cache_and_inactive_coordinate_preservation_verified']
 b=read(bound['bridge_independent_audit']);assert b['state']=='independent_actual_bridge3_samecache_target_coordinate_and_reduction_audit_pass' and b['n']==160 and b['bridge_prediction_seal_sha256']==file_sha(bound['bridge_prediction_seal']) and b['bridge_results_sha256']==file_sha(bound['bridge_scalar_summary'].parent/'RESULTS_PRIVATE.json') and b['same_q0_normalcache_sourcepool_shapeprior_directtargets_verified']
 for audit,script in [(u,'audit_normal_unlabelled.py'),(r,'audit_normal_scored.py'),(g,'audit_gt_controlled_scored.py'),(b,'audit_pseudo_exactfit_scored.py')]:assert audit['auditor_sha256']==file_sha(B/'research/round9/evaluation'/script)
 final=read(bound['final_mechanism_review']);assert final['state']=='R9_actual_three_stage_independent_mechanism_review_complete' and set(final['mask_role_boundary'])=={'normal','GT_C','GT_S','GT_CS','pseudo_bridge'}
 for rel,digest in final['bindings'].items():assert file_sha(local(rel))==digest
 manual=read(bound['manual_identity_review_summary']);assert manual['all20_actual_images_viewed'] and manual['human_visual_assessment_not_blinded_to_GT_overlay_or_spatial_status'] and sum(manual['label_counts'].values())==20 and manual['normal_or_GT_predictions_or_G_not_changed_from_visuals']
 subset=read(bound['manual_subset_error_summary']);assert subset['state']=='postseal_human_visual_same_target_plausibility_subset_descriptive_only' and subset['subset_total']==12
 assert subset['manual_review_sha256']==file_sha(E/'MANUAL_IDENTITY_REVIEW_PRIVATE.json'),'Private manual file is hash-checked only; never parsed/copied'
 fidelity=read(bound['optimization_fidelity_diagnostic']);assert fidelity['state']=='post_normal_score_optimizer_fidelity_diagnostic_same_sealed_targets_no_GT_calculation' and fidelity['normal_prediction_seal_sha256']==file_sha(bound['normal_prediction_seal']) and fidelity['bridge_prediction_seal_sha256']==file_sha(bound['bridge_prediction_seal'])
 supplement=read(bound['manual_optimizer_supplement_review']);assert supplement['state']=='independent_manual_subset_and_optimizer_fidelity_supplement_pass'
 assert supplement['auditor_sha256']==file_sha(B/'research/round9/evaluation/audit_manual_optimizer_supplement.py') and supplement['manual_review_sha256']==subset['manual_review_sha256'] and supplement['manual_aggregate_sha256']==file_sha(bound['manual_subset_error_summary']) and supplement['optimization_diagnostic_sha256']==file_sha(bound['optimization_fidelity_diagnostic'])
 assert supplement['human_labels_not_reassessed_or_changed'] and supplement['human_review_nonblind_GT_overlay_status_explicit'] and supplement['manual_same_target_n']==12
 rootreview=read(bound['root_final_review']);assert rootreview['publication_ready'] is True and rootreview['report_sha256']==file_sha(bound['root_report']) and rootreview['decision_sha256']==file_sha(bound['root_decision'])
 # Privacy corpus input-only, never raw target annotations or private GT sidecars.
 private=read(B/'research/round8/experiment/full5912_gpu_inputs.json');ids={x['id'] for x in private};scenes={x['scene'] for x in private};captions={sha(x['caption'].encode()) for x in private};home=str(Path.home())+'/'
 cohort=read(E/'diagnostic160_gpu_inputs.json');assert len(cohort)==160 and pred['normal']['manifest_sha256']==file_sha(E/'diagnostic160_gpu_inputs.json')
 scene_alias={k:'scene_'+str(i+1).zfill(2) for i,k in enumerate(sorted({(x['platform'],x['scene']) for x in cohort}))};case_alias={x['id']:'case_'+str(i+1).zfill(6) for i,x in enumerate(cohort)}
 forbidden={'rows_PRIVATE','scene','frame','caption','captions','utterance','ground_info','bbox_3d','gt_bbox','positive_map','GT7D_model','target6_modelcoords','source_boxes_raw6','arm_boxes_raw6','arm_boxes_eval6','pseudo_target6','pseudo_target6_float64','raw_consensus_center','coordinate_residual6','coordinate_residual6_by_arm','constructed_target6_by_arm','center_features','size_features','proj_queries','proj_tokens','points','image_intrinsic','image_extrinsic','camera_distortion','pose','embeddings'}
 @lru_cache(maxsize=32768)
 def textcheck(s):
  assert home not in s and 'file:///' not in s,'Private absolute path'
  assert not any(x in s for x in ids) and not any(x in s for x in scenes),'Private ID/scene string'
  assert sha(s.encode()) not in captions,'Original fullcaption'
  assert not re.search(r'https?://[^\s]+[?&](?:X-Amz-Signature|Signature)=',s),'Signed URL'
  return s
 def clean(v):
  if isinstance(v,dict):
   assert not forbidden&set(v) and not any(k.endswith('_PRIVATE') for k in v),('Private payload keys',set(v));return {textcheck(str(k)):clean(x) for k,x in v.items()}
  if isinstance(v,(list,tuple)):return [clean(x) for x in v]
  if isinstance(v,str):return textcheck(v.replace(str(B),'<workspace>').replace(home,'<local-home>/'))
  if isinstance(v,float):assert math.isfinite(v),'Nonfinite scalar'
  return v
 def project(v):
  if isinstance(v,dict):return {k:project(x) for k,x in v.items() if not k.endswith('_path') and not k.endswith('_dir') and k not in PRIVATE_MAPS}
  if isinstance(v,list):return [project(x) for x in v]
  return v
 pending={};provenance=[]
 def put(name,data,source=None,mode='generated'):
  p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts and name not in pending and p.suffix.lower() not in ('.pt','.pth','.bin','.npz','.npy','.safetensors','.jpg','.jpeg','.png','.mp4','.wav','.zip','.so','.pkl')
  pending[name]=data
  if source:provenance.append({'path':name,'source':source.relative_to(B).as_posix(),'source_sha256':file_sha(source),'export_sha256':sha(data),'mode':mode})
 def js(name,v,source=None,mode='explicit_public_scalar_projection'):put(name,(json.dumps(clean(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode(),source,mode)
 def exact(rel,name=None):
  p=local(rel);old=p.read_text();s=old
  if p.suffix=='.py':
   anchor='Path(__file__).resolve().parents['+str(len(PurePosixPath(rel).parts)-1)+']';s=s.replace('Path('+repr(str(B))+')',anchor).replace('Path('+json.dumps(str(B))+')',anchor)
  if p.suffix=='.md':s=s.replace(str(B),'<workspace>')
  assert home not in s and not any(x in s for x in ids|scenes),'Private literal in code/document'
  put(name or rel,s.encode(),p,'byte_exact' if old==s else 'workspace_path_only_projection_original_and_export_SHA_retained')
 for rel in CODE_FILES+DOC_FILES:exact(rel)
 for rel in STATIC_JSON:js(rel,project(read(local(rel))),local(rel),'historical_CPU_or_plan_projection_no_private_filemaps_paths')
 exact('research/round9/experiment/REPRODUCTION.md','REPRODUCTION.md');exact(specs['root_report']['path'],'REPORT.md');js('DECISION.json',read(bound['root_decision']),bound['root_decision']);js('audits/ROOT_FINAL_REVIEW.json',{k:rootreview[k] for k in ('publication_ready','report_sha256','decision_sha256')},bound['root_final_review'])
 for stage,contract in STAGES.items():
  p=E/contract['plan'];js('protocol/'+p.name,project(read(p)),p,'plan_private_paths_and_rawID_filemaps_omitted_original_hash_retained')
  sp=bound[stage+'_prediction_seal'];s=pred[stage];js('results/'+stage.upper()+'_PREDICTION_SEAL_SUMMARY.json',project({k:v for k,v in s.items() if k!='files'})|{'original_prediction_seal_sha256':file_sha(sp),'scientific_role':contract['role'],'original_prediction_file_count':len(s['files'])},sp,'seal_metadata_no_prediction_or_feature_filemap')
  p=bound[stage+'_scalar_summary'];assert 'rows_PRIVATE' not in scores[stage];js('results/'+stage.upper()+'_SCALAR_SUMMARY.json',scores[stage],p)
  p=bound[stage+'_score_release'];js('protocol/'+stage.upper()+'_CPU_SCORE_RELEASE.json',project(read(p)),p,'release_paths_and_rawID_filemaps_removed')
 for key,value in [('normal_unlabelled_audit',u),('normal_reduction_audit',r),('GT_capacity_audit',g),('bridge_independent_audit',b)]:js('audits/'+key.upper()+'.json',project(value),bound[key])
 js('audits/FINAL_MECHANISM_REVIEW.json',project(final),bound['final_mechanism_review']);js('audits/MANUAL_IDENTITY_REVIEW_SUMMARY.json',manual,bound['manual_identity_review_summary'],'nonblind20case_manual_aggregate_only_no_privateIDs_or_media')
 for key in ('manual_subset_error_summary','optimization_fidelity_diagnostic','manual_optimizer_supplement_review'):js('audits/'+key.upper()+'.json',project(read(bound[key])),bound[key])
 failure=local('research/round9/experiment/START_FAILURE_ATTEMPT001.json');v=read(failure);js('history/START_FAILURE_ATTEMPT001.json',{k:v[k] for k in ('state','reason','actual_plan_sha256','no_prediction_directory_created','log_sha256','normal_code_or_plan_not_modified')},failure,'startup_status_and_hashes_no_log_payload')
 resource=local('crossplatform/round8/SEMANTIC_RESOURCE_RECEIPT.json');rv=read(resource);assert rv['revision']==CLIP_REV and rv['safe_loading']['derived_safetensors_sha256']==CLIP_SAFE;js('resources/FROZEN_CLIP_PROVENANCE.json',project(rv),resource,'provenance_only_no_weights_or_tokenizer_assets')
 source=local('crossplatform/round6/waymo_source100/TRAINING_COMPLETE.json');sv=read(source);assert sv['primary_checkpoint_sha256']==SOURCE and sv['no_Drone_or_Quad_data_access'];keep=('state','primary_epoch','primary_checkpoint_sha256','completed_optimizer_steps','plan_sha256','no_joint_checkpoint_initialization','no_Drone_or_Quad_data_access');js('source/WAYMO_ONLY_SOURCE100_PROVENANCE.json',{k:sv[k] for k in keep},source)
 # One complete table, explicit missing scalar fields remain empty, never invented.
 cols=['stage','scientific_role','case_alias','platform','scene_alias','arm','q_source','annotated_spatially_confirmed','stage_source_iou','delta_vs_stage_source','negative_vs_stage_source_atol1e8']+list(ARM_FIELDS)+['error_'+k for k in COORD_FIELDS]+['capacity_group']+['capacity_'+k for k in CAP_FIELDS]+['capacity_exactfit_pass','actual_backward_step_count','actual_nonzero_gradient_step_count','actual_parameter_changed','head_lr_center','head_lr_size']
 stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=cols);writer.writeheader();stagecounts={};normalrows=None;normalmap={}
 for stage,contract in STAGES.items():
  p=bound[stage+'_scalar_summary'].parent/'RESULTS_PRIVATE.json';result=read(p);rows=result['rows_PRIVATE'];assert len(rows)==160 and [x['id'] for x in rows]==[x['id'] for x in cohort]
  if stage=='normal':normalrows=rows;normalmap={x['id']:x for x in rows}
  count=0
  for row in rows:
   sid=row['id'];assert set(row['arms'])==set(contract['arms']) and sid in case_alias;sourceiou=row['arms']['source']['iou'];status=row['identity_diagnostic']['annotated_spatially_confirmed'];assert status==normalmap[sid]['identity_diagnostic']['annotated_spatially_confirmed']
   for arm in contract['arms']:
    a=row['arms'][arm];unexpected=set(a)-set(ARM_FIELDS)-{'coordinate_errors'};assert not unexpected,unexpected
    record={'stage':stage,'scientific_role':contract['role'],'case_alias':case_alias[sid],'platform':row['platform'],'scene_alias':scene_alias[(row['platform'],row['scene'])],'arm':arm,'q_source':normalmap[sid]['q_source'],'annotated_spatially_confirmed':status,'stage_source_iou':sourceiou,'delta_vs_stage_source':a['iou']-sourceiou,'negative_vs_stage_source_atol1e8':a['iou']-sourceiou< -1e-8,**{k:a[k] for k in ARM_FIELDS if k in a},**{'error_'+k:a['coordinate_errors'][k] for k in COORD_FIELDS}}
    if stage=='GT' and arm!='source':
     group,kind=arm.split('__');group=group[3:];cap=row['capacity'][group];control=cap['controls']['direct' if kind=='direct_target' else kind];record.update(capacity_group=group,**{'capacity_'+k:control[k] for k in CAP_FIELDS},capacity_exactfit_pass=cap['exactfit_within_registered_tolerance'])
     tr=row['actual_head_update_trace'][group]
     if kind in ('param1','param3'):
      steps=tr['SGD'+kind[-1]]['trace'];record.update(actual_backward_step_count=len(steps),actual_nonzero_gradient_step_count=sum(x['parameter_gradient_norm']>0 and x['nonzero_gradient_parameter_tensors']>0 for x in steps),actual_parameter_changed=bool(steps and any(v>0 for v in steps[-1]['parameter_delta_max_by_name'].values())),head_lr_center=tr['actual_SGD_lr'],head_lr_size=tr['actual_SGD_lr'])
     if kind=='exactfit':
      t=tr['exactfit'];record.update(actual_backward_step_count=int(t['real_backward_and_optimizer_step_executed']),actual_nonzero_gradient_step_count=int(any(v is not None and v>0 for v in t['parameter_gradient_norm_by_name'].values())),actual_parameter_changed=t['observed_parameter_change'],head_lr_center=t['actual_head_learning_rates'].get('center'),head_lr_size=t['actual_head_learning_rates'].get('size'))
    writer.writerow(clean(record));count+=1
  assert count==160*len(contract['arms']);stagecounts[stage]=count
 data=stream.getvalue().encode();compressed=gzip.compress(data,compresslevel=9,mtime=0);assert gzip.decompress(compressed)==data and sum(stagecounts.values())==4320
 put('results/ALL4320_ARM_SCALARS.csv.gz',compressed,mode='lossless_explicit_scalar_union_aliases_no_coordinate_arrays');js('results/SCALAR_TABLE_SCHEMA.json',{'columns':cols,'rows':4320,'stage_rows':stagecounts,'independent_input_cases':160,'five_actual_scene_clusters':True,'roles':{k:v['role'] for k,v in STAGES.items()},'compression':'gzip.compress(level9,mtime0), CRC/decompression byteidentity asserted','encoding':'UTF8 csv.DictWriter excel CRLF; missing/None empty; no rounding','case_alias_rule':'case_000001..160 in locked input order, original IDs/mapping omitted','scene_alias_rule':'five ordinalaliases from sorted(platform,private_scene), mapping omitted','coordinate_policy':'Only norm/error/residual scalars, no GT/source/pseudo vectors or embeddings','empty_field_policy':'Unavailable stage-specific scalar left empty; no fabricated uniform fields or manual-zero steps','uncompressed_bytes':len(data),'uncompressed_sha256':sha(data),'compressed_sha256':sha(compressed),'runtime':{'python':platform.python_version(),'zlib':zlib.ZLIB_VERSION}})
 js('PUBLICATION_READINESS.json',ready,E/'PUBLICATION_READINESS.json');js('EXPORT_FINAL_SOURCE_HASHES.json',{k:specs[k]['sha256'] for k in REQUIRED},E/'EXPORT_FINAL_SOURCES.json','hashes_only_no_private_source_paths');js('EXPORT_PROVENANCE.json',provenance)
 put('README.md',b'# R9: Fixed-q0 geometry and regression capability diagnostic\n\nSee REPORT.md/DECISION.json for actual outcomes and REPRODUCTION.md for stages, source and privacy. Normal160x11, labelled ORACLE160x13 and post-normal-score exploratory pseudo-exactfit160x3 retain4320 related control outputs, not4320 independent inputs. Source is genuine Waymo-only epoch100; targets are64Drone96Quad,32per each of5scenes, historically research-exposed. Fixedq0 removes the reselection route structurally, not a same-cohort causal estimate of R8 recovery.\n\nNormal geometry retains q0 size prior. GT C/S/CS controls measure masked coordinate-fitting capacity and are NOT TTA or maximal-IoU bounds; GTyaw is absent from fit. The exploratory bridge fits existing normal directG targets with one real L2/inverse-headkernel step; loss and step normalization change together, and directG is the same-information control. No unique parameter benefit follows from matching a directly available box.\n\nSpatiallyconfirmed13Drone/47Quad coincides with sourceAcc25 successes here, not independent semantic identity proof. Root manual20image review was nonblind to GT/status:12same-visible-target-plausible/3different/4offtarget/1occluded. These are descriptive labels on selected figures, not population identity error rates. Zero targets, negative arms and all denominators remain. No new GPU experiment is queued by this export.\n\nWeights, sourcefeatures/caches, private GT6/7D indices/sidecars, actual coordinate arrays, media/captions, rawIDs/scenes and manualprivate rows are excluded. Only alias scalar rows, aggregates, public code/protocol and hashes are included. Pre-start wrong CLI digest and float32 auditor expectation correction retain honest historical receipts; experiment plans/results are unchanged.\n')
 for name,data in pending.items():assert len(data)<95*1024**2 and not any(x.encode() in data for x in ids|scenes),(name,'Private ID/scene or unexpectedly large file')
 out.mkdir(parents=True)
 for name,data in pending.items():p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 manifest={'study':'3eed_fixed_q0_geometry_regression_diagnostic_20261030','files':[{'path':k,'bytes':len(v),'sha256':sha(v)} for k,v in sorted(pending.items())],'root_final_review_sha256':file_sha(bound['root_final_review']),'original_stage_prediction_seal_sha256':{k:file_sha(bound[k+'_prediction_seal']) for k in STAGES},'source_checkpoint_sha256':SOURCE,'CLIP_revision':CLIP_REV,'scope':'Finite public code/protocol/alias4320scalar outputs; no private data or weights'};(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'n_files':len(pending)+1,'manifest_sha256':file_sha(out/'MANIFEST.json'),'output':str(out)}))
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=E/'public_export');main(ap.parse_args())
