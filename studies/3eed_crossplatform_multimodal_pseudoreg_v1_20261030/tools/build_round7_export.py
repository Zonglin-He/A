"""Finite R7 public candidate builder. Preparation only until root final gates.
No model/GPU/network/Git call. Never copies media, weights, NPZ or private input.
See research/round7/experiment/REPRODUCTION.md for required final-source schema.
"""
from pathlib import Path,PurePosixPath
import argparse,csv,gzip,hashlib,io,json,math,platform,re,zlib
B=Path(__file__).resolve().parents[1];R=B/'research/round7';E=R/'experiment';C=B/'crossplatform/round7'
HOME_PREFIX=str(Path.home())+'/'
SOURCE_SHA='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
MODES=('rgb_text','pc_only','shuffled_color');KINDS=('direct_G','param1','param3','outputspace1','outputspace3')
ARMS=('source',)+tuple(m+'__'+k for m in MODES for k in KINDS)
CODE_FILES=(
 'crossplatform/round7/regression_adapter.py','crossplatform/round7/camera_projection.py','crossplatform/round7/multimodal_pseudo.py','crossplatform/round7/v1_pipeline.py',
 'crossplatform/round7/check_regression_adapter_cpu.py','crossplatform/round7/check_camera_projection_cpu.py','crossplatform/round7/check_multimodal_pseudo_cpu.py','crossplatform/round7/check_v1_pipeline_cpu.py','crossplatform/round7/check_FOV_repair_cpu.py',
 'crossplatform/round7/experiment/run_crossdomain.py','crossplatform/round7/experiment/runner_helpers.py','crossplatform/round7/experiment/check_runner_cpu.py','crossplatform/round7/experiment/real_head_positive_control.py','crossplatform/round7/experiment/positive_control_v2.py','crossplatform/round7/experiment/positive_control_v3_precision.py',
 'crossplatform/round7/history/before_FOV_fix/multimodal_pseudo.py','crossplatform/round7/history/before_FOV_fix/check_multimodal_pseudo_cpu.py',
 'research/round7/evaluation/score_crossdomain.py','research/round7/evaluation/audit_unlabelled_predictions.py','research/round7/evaluation/check_score_crossdomain_cpu.py','research/round7/evaluation/check_unlabelled_audit_cpu.py',
 'research/round7/evaluation/history/draft_69b5/score_crossdomain.py','research/round7/evaluation/history/draft_69b5/check_score_crossdomain_cpu.py','research/round7/evaluation/history/draft_e384/score_crossdomain.py','research/round7/evaluation/history/draft_e384/check_score_crossdomain_cpu.py',
 'research/round7/experiment/input_audit/prepare_inputs.py','research/round7/experiment/input_audit/strict_manifests_and_projection.py','research/round7/audit_rgb_geometry_interface.py',
 # Pure dependencies and source model constructor, not old mixed-checkpoint drivers.
 'crossplatform/round6/factory.py','research/round5/audit/independent_core_v2.py','research/round6/source_validation_scoring/batch_yaw_iou.py','tools/obb_iou.py','tools/build_round7_export.py')
DOC_FILES=(
 'crossplatform/round7/MODEL_INTERFACE.md','crossplatform/round7/G_CONTRACT.md','crossplatform/round7/experiment/RUNNER_INTERFACE.md','crossplatform/round7/history/before_FOV_fix/G_CONTRACT.md',
 'research/round7/METHOD_V1.md','research/round7/RGB_GEOMETRY_INTERFACE_AUDIT.md','research/round7/evaluation/SCORING_CONTRACT.md')
STATIC_JSON=(
 'crossplatform/round7/MODEL_INTERFACE_AUDIT.json','crossplatform/round7/G_IMPLEMENTATION_LOCK.json','crossplatform/round7/CPU_CAMERA_FIXTURE.json','crossplatform/round7/CPU_INTERFACE_FIXTURE.json','crossplatform/round7/CPU_G_FIXTURE.json','crossplatform/round7/CPU_PIPELINE_FIXTURE.json','crossplatform/round7/CPU_FOV_REPAIR_FIXTURE.json',
 'crossplatform/round7/experiment/RUNNER_READY.json','crossplatform/round7/experiment/CPU_RUNNER_FIXTURE.json','crossplatform/round7/experiment/POSITIVE_CONTROL_PREPARATION.json','crossplatform/round7/experiment/POSITIVE_CONTROL_V2_PREPARATION.json','crossplatform/round7/experiment/POSITIVE_CONTROL_V3_PREPARATION.json',
 'crossplatform/round7/history/before_FOV_fix/G_IMPLEMENTATION_LOCK.json','crossplatform/round7/history/before_FOV_fix/CPU_G_FIXTURE.json',
 'research/round7/METHOD_V1_LOCK.json','research/round7/METHOD_V1_LOCK_before_FOV_fix.json','research/round7/INDEPENDENT_IMPLEMENTATION_REVIEW.json','research/round7/RGB_METADATA_READ_BOUNDARY_CLARIFICATION.json',
 'research/round7/evaluation/SCORER_PROGRAM_LOCK.json','research/round7/evaluation/SCORER_PROGRAM_LOCK_v2.json','research/round7/evaluation/MEASUREMENT_INTERFACE_CORRECTION.json','research/round7/evaluation/SYNTHETIC_SCORER_FIXTURE.json','research/round7/evaluation/SYNTHETIC_UNLABELLED_AUDIT_FIXTURE.json',
 'research/round7/evaluation/history/draft_69b5/SCORER_PROGRAM_LOCK.json','research/round7/evaluation/history/draft_69b5/SYNTHETIC_SCORER_FIXTURE.json','research/round7/evaluation/history/draft_e384/SYNTHETIC_SCORER_FIXTURE.json')
SCORE_FIELDS=('state','n_samples','stage','summary','source_candidate_coverage','geometry_audit_counts','max_selected_3D_vs_batch_abs_error','strict_threshold_disagreements','negative_case_count_all_arms','scope','effective_scene_warning','coverage_geometry_boundary','failure_scope','seconds')
AUDIT_FIELDS=('state','prediction_seal_sha256','auditor_sha256','scorer_loader_sha256','phase','n','summary','max_parameter_outputspace_abs_error','max_source_pseudo_interval_excess','numeric_comparison_atol','numeric_comparison_rtol','fixed_alpha','all_input_code_prediction_hashes_verified','real_source_repeat_Conv_bridge_and_state_restore_receipt_verified','no_scoring_index_or_target_metadata_GT_opened','no_GPU_model_or_task_IoU_scoring','seconds')
ARM_FIELDS=('mode','identity_skip','parameter_steps_executed','nonzero_parameter_update','nonzero_gradient_steps','status','iou','delta_iou','negative_case','changed_nonzero','raw_nonpositive_dimensions','clamped_dimensions')
PSEUDO_FIELDS=('accepted_axis_count','nonzero_displacement_axes','source_center_error_m','pseudo_center_error_m','direction_cosine_to_GT_center','direction_cosine_undefined_reason','accepted_axes_move_toward_GT','accepted_axes_move_away_from_GT','pseudo_iou','postseal_descriptive_only_never_gate_or_select_alpha')
# A completed diagnostic may validly preserve an original-precision failure.
PC_STATE='strict_FP32_only_engineering_pass_original_mode_failed_preserved'
PC_FIELDS=('state','purpose','source_checkpoint_sha256','source_model_state_fingerprint','strict_state_entries','plan_sha256','old_failure_receipt_sha256','sourcepoint_sample_sha256','sourcepoint_indices_sha256','rawpoint_count','artificial_target_delta6_metres','artificial_reliability6','alpha','beta_metres','predeclared_equivalence_atol_metres','original_mode_passed','strict_FP32_mode_passed','mode_results','one_model_load_and_same_source_snapshot_each_mode','mode_source_queries_may_differ_reported_not_GT_selected','no_RGB_G_feedback_target_GT_or_taskscore','main_full_task_caches_metrics_and_precision_not_changed_or_retested','original_mode_failure_not_overridden_by_strict_mode','elapsed_seconds','interpretation')
PC_MODE_FIELDS=('precision_mode','flags','stage','results','mode_pass','current_mode_Conv_bridge_bitexact','current_mode_source_repeat_bitexact','source_model_fingerprint_before','current_mode_query','smoke_query','query_equal_smoke','source_pool_crossprocess','source_scores_crossprocess','nativeTop10_equal_smoke','nativeTop10_changed_positions','same_source_weights_and_P_indices_strict','actual_lr','center_kernel','size_kernel','effective_x_shrink_factor','artificial_target_from_current_mode_source','peak_allocated_bytes')
PC_RESULT_FIELDS=('steps','raw_step_observations','observed_selected_delta6_metres','final_allquery_equivalence_max_abs_error_metres','all_step_max_equivalence_error_metres','nonselected_parameter_version_changed_count','nonselected_gradients_after_restore_count','post_episode_full_model_fingerprint_restored','feature_norms','spillover','expected_real_arithmetic_x_delta_metres','judgement','checks')
PC_OBSERVATION_FIELDS=('step','loss','weighted_SmoothL1_box_gradient6','parameter_gradient_norm','nonzero_gradient_parameter_tensors','clip_enabled','clip_factor','parameter_delta_max_by_name','output_space_replay_max_abs_error','observed_selected_delta6_metres','nonselected_parameter_version_changes_across_entire_call','nonselected_parameter_max_abs_change_if_versions_unchanged','nonselected_version_evidence')
PC_NESTED_FIELDS={
 'flags':('cudnn_allow_tf32','matmul_allow_tf32','cudnn_benchmark','cudnn_deterministic','cudnn_enabled','float32_matmul_precision','deterministic_algorithms_enabled'),
 'drift':('bitexact','max_absolute_difference','mean_absolute_difference','p99_absolute_difference','max_relative_difference_reference_floor1e_minus12','different_value_count'),
 'feature_norms':('center_squared_feature_norm','size_squared_feature_norm','center_self_kernel_with_bias','size_self_kernel_with_bias'),
 'spillover':('other_query_count_changed','other_query_max_abs_center_change','other_query_max_abs_size_change'),
 'checks':('all_equivalence_errors_le_registered5e_minus5','true_nonzero_gradient_and_center_parameter_change','size_parameters_unchanged','only_selected_x_output_changed','closed_form_x_delta_within_registered_atol','nonselected_unchanged','full_model_restored')}
def project_positive_control(pc):
 # Explicit nested schemas fail closed if a future receipt introduces coordinates.
 assert set(pc)==set(PC_FIELDS)
 out={k:pc[k] for k in PC_FIELDS};out['mode_results']={}
 assert set(pc['mode_results'])=={'original_plan','strict_FP32'}
 for name,m in pc['mode_results'].items():
  assert set(m)==set(PC_MODE_FIELDS);mm={k:m[k] for k in PC_MODE_FIELDS};mm['results']={}
  for key,fields_ in [('flags',PC_NESTED_FIELDS['flags']),('source_pool_crossprocess',PC_NESTED_FIELDS['drift']),('source_scores_crossprocess',PC_NESTED_FIELDS['drift'])]:assert set(m[key])==set(fields_)
  assert set(m['results'])=={'1','3'}
  for step,r in m['results'].items():
   assert set(r)==set(PC_RESULT_FIELDS)
   for key in ('feature_norms','spillover','checks'):assert set(r[key])==set(PC_NESTED_FIELDS[key])
   assert len(r['raw_step_observations'])==int(step)
   for obs in r['raw_step_observations']:
    assert set(obs)==set(PC_OBSERVATION_FIELDS)
    assert set(obs['parameter_delta_max_by_name'])=={'prediction_heads.5.'+h+'.net.8.'+v for h in ('center_residual_head','size_pred_head') for v in ('weight','bias')}
   mm['results'][step]={k:r[k] for k in PC_RESULT_FIELDS}
  out['mode_results'][name]=mm
 return out

def validate_positive_control(pc,pcseal,pcplan,plan_sha,receipt_sha,full_sha,smoke_sha,source_fingerprint):
 project_positive_control(pc)
 assert pc['state']==pcseal['state']==PC_STATE
 assert pcseal['files']['ENGINEERING_POSITIVE_CONTROL.json']==receipt_sha
 assert pc['plan_sha256']==pcseal['plan_sha256']==plan_sha
 assert pcplan['purpose']==pc['purpose']=='R7_real_head_positive_control_v3_precision_diagnostic'
 assert pcplan['completed_full_prediction_seal_sha256']==full_sha and pcplan['smoke_prediction_seal_sha256']==smoke_sha
 assert pc['source_checkpoint_sha256']==pcplan['source_checkpoint_sha256']==pcseal['source_checkpoint_sha256']==SOURCE_SHA
 assert pc['source_model_state_fingerprint']==pcplan['source_model_state_fingerprint']==source_fingerprint
 assert pc['original_mode_passed'] is pcseal['original_mode_passed'] is False
 assert pc['strict_FP32_mode_passed'] is pcseal['strict_FP32_mode_passed'] is True
 assert pc['no_RGB_G_feedback_target_GT_or_taskscore'] and pc['main_full_task_caches_metrics_and_precision_not_changed_or_retested'] and pc['original_mode_failure_not_overridden_by_strict_mode'] and pc['one_model_load_and_same_source_snapshot_each_mode']
 assert pc['artificial_target_delta6_metres']==pcplan['target_delta6_metres']==[.05,0.,0.,0.,0.,0.] and pc['artificial_reliability6']==pcplan['reliability6']==[1.,0.,0.,0.,0.,0.]
 assert pc['alpha']==pcplan['alpha']==.25 and pc['beta_metres']==pcplan['beta_metres']==.1
 assert pc['predeclared_equivalence_atol_metres']==pcplan['equivalence_atol_metres']==5e-5
 assert pcplan['mode_order']==['original_plan','strict_FP32'] and pcplan['steps']==[1,3]
 for name,expected in [('original_plan',False),('strict_FP32',True)]:
  m=pc['mode_results'][name];assert m['mode_pass'] is expected and m['precision_mode']==name
  assert m['current_mode_Conv_bridge_bitexact'] and m['current_mode_source_repeat_bitexact'] and m['same_source_weights_and_P_indices_strict']
  assert m['source_model_fingerprint_before']==source_fingerprint
  for flag,value in pcplan['precision_modes'][name].items():assert m['flags'][flag]==value
  for k in ('1','3'):
   r=m['results'][k];assert r['steps']==int(k) and r['post_episode_full_model_fingerprint_restored'] and r['nonselected_parameter_version_changed_count']==0 and r['nonselected_gradients_after_restore_count']==0
   assert r['checks']['true_nonzero_gradient_and_center_parameter_change'] and all(r['checks'][c] for c in ('size_parameters_unchanged','only_selected_x_output_changed','closed_form_x_delta_within_registered_atol','nonselected_unchanged','full_model_restored'))
   assert all(x['parameter_gradient_norm']>0 and x['nonzero_gradient_parameter_tensors']>0 and max(x['parameter_delta_max_by_name'].values())>0 for x in r['raw_step_observations'])
   err=max([r['all_step_max_equivalence_error_metres'],r['final_allquery_equivalence_max_abs_error_metres']]+[x['output_space_replay_max_abs_error'] for x in r['raw_step_observations']])
   assert r['checks']['all_equivalence_errors_le_registered5e_minus5'] is expected
   assert (err<=5e-5) is expected and r['judgement']==('pass' if expected else 'fail_preserved')

sha=lambda b:hashlib.sha256(b).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for part in iter(lambda:f.read(8<<20),b''):h.update(part)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def local(rel):
 q=PurePosixPath(rel);assert q.parts and not q.is_absolute() and '..' not in q.parts
 p=B/q.as_posix();assert p.resolve().is_relative_to(B.resolve());assert p.is_file(),rel;return p

def main(args):
 out=args.output.resolve();assert out.is_relative_to(B.resolve()) and not out.exists(),'New bounded destination required; never overwrite a published candidate.'
 # No outputdirectory or private source scanning before all actual final gates.
 ready=read(E/'PUBLICATION_READINESS.json')
 for key in ('smoke_and_full_predictions_sealed','full5912_scored','full_unlabelled_audit_passed','real_positive_control_completed','root_final_review_passed','report_decision_privacy_reviewed','no_raw_data_media_weights_in_export'):assert ready.get(key) is True,('Unfinished publication gate',key)
 sources=read(E/'EXPORT_FINAL_SOURCES.json');bound={}
 required=('smoke_prediction_seal','full_prediction_seal','smoke_scalar_summary','full_scalar_summary','full_unlabelled_audit','positive_control_receipt','positive_control_seal','positive_control_plan','smoke_score_release','full_score_release','root_report','root_decision','root_final_review')
 for key in required:
  spec=sources[key];assert set(spec)=={'path','sha256'};fp=local(spec['path']);assert file_sha(fp)==spec['sha256'],('Changed final source',key);bound[key]=fp
 for phase,n in [('smoke',20),('full',5912)]:
  sp=bound[phase+'_prediction_seal'];se=read(sp);plan=read(E/(phase.upper()+'_PLAN_LOCK.json'));assert se['state']=='sealed_crossdomain_R7v1_no_target_labels_or_scoring' and se['phase']==phase and se['n_samples']==n
  assert se['all_16arms_sealed_before_CPUscoring'] and se['model_restored_source_state'] and se['source_checkpoint_sha256']==SOURCE_SHA
  assert se['plan_sha256']==file_sha(E/(phase.upper()+'_PLAN_LOCK.json')) and se['manifest_sha256']==plan['manifest_sha256'];assert se['all_BN_eval_dropout_off'] and se['no_teacher_EMA_or_source_retraining']
  for name,digest in se['program_sha256'].items():assert file_sha(local('crossplatform/round7/'+name))==digest
  # Final scored aggregate and privatecase rows must be an actually sealed stage.
  scored=bound[phase+'_scalar_summary'];scoreseal=read(scored.parent/'SEALED.json');assert scoreseal['files'][scored.name]==file_sha(scored);v=read(scored);assert v['state']=='scored_post_stage_seal' and v['n_samples']==n
  assert v['strict_threshold_disagreements']==0 and v['max_selected_3D_vs_batch_abs_error']<1e-8
  grant=read(bound[phase+'_score_release']);assert grant['CPU_SCORE_RELEASE'] is True and grant['all_stage_predictions_sealed_before_stage_GT_score'] is True;assert grant['prediction_seal_sha256']==file_sha(sp) and grant['manifest_sha256']==se['manifest_sha256'] and grant['source_checkpoint_sha256']==SOURCE_SHA
  assert read(scored.parent/'SCORER_LOCK.json')['prediction_seal_sha256']==file_sha(sp)
  for filename in ('RESULTS_PRIVATE.json','SCORER_LOCK.json'):assert scoreseal['files'][filename]==file_sha(scored.parent/filename)
 unlabelled=read(bound['full_unlabelled_audit']);assert unlabelled['state']=='passed_unlabelled_sealed_prediction_contracts' and unlabelled['n']==5912 and unlabelled['prediction_seal_sha256']==file_sha(bound['full_prediction_seal']) and unlabelled['all_input_code_prediction_hashes_verified']
 pc=read(bound['positive_control_receipt']);pcseal=read(bound['positive_control_seal']);pcplan=read(bound['positive_control_plan'])
 validate_positive_control(pc,pcseal,pcplan,file_sha(bound['positive_control_plan']),file_sha(bound['positive_control_receipt']),file_sha(bound['full_prediction_seal']),file_sha(bound['smoke_prediction_seal']),read(bound['full_prediction_seal'])['source_model_state_fingerprint'])
 for name,digest in pcplan['program_sha256'].items():assert file_sha(local('crossplatform/round7/'+name))==digest
 for name,digest in pcseal['files'].items():
  assert PurePosixPath(name).name==name and file_sha(bound['positive_control_seal'].parent/name)==digest
 assert pcseal['files']['registered_plan.json']==file_sha(bound['positive_control_plan'])
 for name,digest in pc['old_failure_receipt_sha256'].items():assert digest==pcplan['old_failure_receipt_sha256'][name]==file_sha(local('crossplatform/round7/experiment/'+name))
 optional={}
 for key in ('component_aggregates','component_plan','cross_stage_reproducibility'):
  if key in sources:
   spec=sources[key];assert set(spec)=={'path','sha256'};fp=local(spec['path']);assert file_sha(fp)==spec['sha256'];optional[key]=fp
 assert ('component_aggregates' in optional)==('component_plan' in optional),'Component aggregate and plan must be included together.'
 if 'component_aggregates' in optional:
  component=read(optional['component_aggregates']);componentplan=read(optional['component_plan'])
  assert component['state']=='bounded_unlabelled_descriptive_diagnostic_no_method_change' and component['purpose']==componentplan['purpose']=='R7_unlabelled_same_component_diagnostic'
  assert component['source_checkpoint_sha256']==componentplan['source_checkpoint_sha256']==SOURCE_SHA and component['full_prediction_seal_sha256']==componentplan['full_prediction_seal_sha256']==file_sha(bound['full_prediction_seal'])
  assert component['plan_sha256']==file_sha(optional['component_plan']) and component['no_GT_instance_identity_or_task_improvement_claim'] and component['no_G_threshold_plan_or_adoption_change']
  assert componentplan['CPU_execution_authorized'] and componentplan['high_overlap_fraction']==component['overlap_threshold_descriptive_only']==.8
  for name,digest in componentplan['diagnostic_program_sha256'].items():assert file_sha(local('crossplatform/round7/'+name))==digest
 finalreview=read(bound['root_final_review']);assert finalreview.get('publication_ready') is True and finalreview['report_sha256']==file_sha(bound['root_report']) and finalreview['decision_sha256']==file_sha(bound['root_decision'])
 train=read(local('crossplatform/round6/waymo_source100/TRAINING_COMPLETE.json'));assert train['primary_epoch']==100 and train['completed_optimizer_steps']==15300 and train['primary_checkpoint_sha256']==SOURCE_SHA and train['no_joint_checkpoint_initialization'] and train['no_Drone_or_Quad_data_access']
 # Privateinputs are consulted solely for the privacy denylist; never copied.
 private=read(E/'full_inputs.json');raw_scenes={r['scene'] for r in private};raw_frames={r['frame'] for r in private};caption_hashes={sha(r['caption'].encode()) for r in private}
 forbidden={'rows_PRIVATE','ground_info','caption','captions','utterance','utterances','frame','scene','scan_id','sequence_name','meta_path','metapath','imagepath','pointpath','image_path','lidar_path','metadata_path','source_boxes_raw6','source_boxes_eval6','arm_boxes_raw6','arm_boxes_eval6','pseudo_target6','points','point_clouds','RGB','rgb','image_intrinsic','image_extrinsic','camera_distortion','pose','bbox_3d','gt_bbox','gt_bboxes','positive_map','proj_queries','proj_tokens','query_features','center_features','size_features'}
 def textcheck(s):
  assert HOME_PREFIX not in s and 'file://' not in s,'Unprojected private workspace path';assert not any(x in s for x in raw_scenes),'Original physicalscene identifier';assert s not in raw_frames,'Original frameID';assert sha(s.encode()) not in caption_hashes,'Original captionvalue';return s
 def clean(v):
  if isinstance(v,dict):
   assert not (forbidden&set(v)),('Forbidden payloadkeys',forbidden&set(v));return {textcheck(str(k)):clean(x) for k,x in v.items()}
  if isinstance(v,(list,tuple)):return [clean(x) for x in v]
  if isinstance(v,str):return textcheck(v.replace(str(B)+'/', '<workspace>/').replace(HOME_PREFIX,'<local-home>/'))
  if isinstance(v,float):assert math.isfinite(v),'NaN/inf cannot silently enter export'
  return v
 pending={};provenance=[]
 def put(dest,data,src=None,mode='generated'):
  q=PurePosixPath(dest);assert q.parts and not q.is_absolute() and '..' not in q.parts and dest not in pending;assert q.suffix.lower() not in ('.pt','.pth','.npy','.npz','.safetensors','.jpg','.jpeg','.png','.mp4','.wav','.zip','.so');pending[dest]=data
  if src:provenance.append({'path':dest,'source':src.relative_to(B).as_posix(),'source_sha256':file_sha(src),'export_sha256':sha(data),'mode':mode})
 def js(dest,v,src=None,mode='explicit_scalar_field_projection'):
  put(dest,(json.dumps(clean(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode(),src,mode)
 def exact(rel,dest=None):
  src=local(rel);s=src.read_text();original=s;mode='byte_exact_code_or_public_document'
  if src.suffix=='.py':
   anchor='Path(__file__).resolve().parents['+str(len(PurePosixPath(rel).parts)-1)+']'
   s=s.replace("Path("+repr(str(B))+")",anchor).replace("Path("+json.dumps(str(B))+")",anchor)
  elif src.suffix=='.md':s=s.replace(str(B)+'/','<workspace>/')
  if s!=original:mode='path_only_portability_projection_no_algorithm_change_original_exportSHA_retained'
  assert HOME_PREFIX not in s and not any(x in s for x in raw_scenes),'Code/document contains unprojected path orreal scene';put(dest or rel,s.encode(),src,mode)
 def fields(src,names):
  v=read(src);return {k:v[k] for k in names}
 for rel in CODE_FILES+DOC_FILES:exact(rel)
 for rel in STATIC_JSON:js(rel,read(local(rel)),local(rel),'synthetic_or_historical_lock_scalar_and_codehash_receipt')
 # Historical code/fixtures are preserved as history, not relabeled current results.
 exact('research/round7/experiment/REPRODUCTION.md','REPRODUCTION.md');exact(sources['root_report']['path'],'REPORT.md');js('DECISION.json',read(bound['root_decision']),bound['root_decision']);js('audits/ROOT_FINAL_REVIEW.json',fields(bound['root_final_review'],('publication_ready','report_sha256','decision_sha256')),bound['root_final_review'])
 for phase in ('smoke','full'):
  p=E/(phase.upper()+'_PLAN_LOCK.json');v=read(p);v.pop('source_checkpoint_path',None);v['source_checkpoint_location']='LOCAL_SOURCE_WEIGHTS_NOT_DISTRIBUTED';js('protocol/'+p.name,v,p,'plan_field_projection_checkpointpath_removed_original_planSHA_retained')
  sp=bound[phase+'_prediction_seal'];v=read(sp);sel={k:v[k] for k in ('state','phase','n_samples','manifest_sha256','plan_sha256','source_checkpoint_sha256','source_model_state_fingerprint','all_16arms_sealed_before_CPUscoring','model_restored_source_state','all_BN_eval_dropout_off','no_teacher_EMA_or_source_retraining')};sel['sealed_sample_file_count']=len([x for x in v['files'] if x.endswith('.npz')]);sel['original_prediction_seal_sha256']=file_sha(sp);js('results/'+phase.upper()+'_PREDICTION_SEAL_SUMMARY.json',sel,sp)
  src=bound[phase+'_scalar_summary'];js('results/'+phase.upper()+'_SCALAR_SUMMARY.json',fields(src,SCORE_FIELDS),src)
  grant=read(bound[phase+'_score_release']);js('protocol/'+phase.upper()+'_CPU_SCORE_RELEASE.json',{k:grant[k] for k in ('CPU_SCORE_RELEASE','all_stage_predictions_sealed_before_stage_GT_score','stage','prediction_seal_sha256','manifest_sha256','scoring_index_sha256','source_checkpoint_sha256','scorer_program_sha256')},bound[phase+'_score_release'])
 js('audits/FULL_UNLABELLED_AUDIT.json',{k:unlabelled[k] for k in AUDIT_FIELDS},bound['full_unlabelled_audit'])
 # Artificialtarget engineering control contains only relative displacement/gradients;
 # explicit fields exclude any real source coordinates, caption, image or pointcache.
 js('engineering/POSITIVE_CONTROL_V3_SEAL_SUMMARY.json',{'state':pcseal['state'],'purpose':pcseal['purpose'],'plan_sha256':pcseal['plan_sha256'],'source_checkpoint_sha256':pcseal['source_checkpoint_sha256'],'original_mode_passed':pcseal['original_mode_passed'],'strict_FP32_mode_passed':pcseal['strict_FP32_mode_passed'],'original_seal_sha256':file_sha(bound['positive_control_seal']),'verified_file_count':len(pcseal['files'])},bound['positive_control_seal'])
 js('engineering/REAL_HEAD_POSITIVE_CONTROL_V3.json',project_positive_control(pc),bound['positive_control_receipt'],'full_explicit_mode_results_relative_deltas_gradients_and_crossprocess_drift_no_box_coordinates')
 pcpublic={k:v for k,v in pcplan.items() if not k.endswith('_path')};js('protocol/POSITIVE_CONTROL_V3_PLAN.json',pcpublic,bound['positive_control_plan'],'plan_paths_removed_hashes_retained')
 for name in ('POSITIVE_CONTROL_V1_FAILURE.json','POSITIVE_CONTROL_V2_FAILURE.json'):
  fp=local('crossplatform/round7/experiment/'+name);v=read(fp)
  keep=('state','attempt','failure','original_failure_line','original_script_sha256','original_plan_sha256','registered_plan_sha256','log_sha256','checkpoint_and_loaded_model_fingerprint_checks_passed_before_failure','reached_real_backward','numeric_pool_drift_not_saved_by_v1','old_files_modified','full_task_predictions_and_scoring_unchanged','interpretation','failure_line','actual_equivalence_error_not_serialized_by_v2','reached_real_nonzero_gradient_and_centerparameter_change_checks','source_weights_and_input_hashes_passed','cross_process_pool_max_absolute_difference_metres_recorded','cause_not_determined_no_TF32_attribution_yet','old_outputs_scripts_plans_logs_not_modified','full_task_predictions_and_metrics_not_modified')
  public={k:v[k] for k in keep if k in v};public['original_failure_receipt_sha256']=file_sha(fp)
  if 'files_sha256' in v:public['historical_file_hashes']=[{'basename':PurePosixPath(k).name,'sha256':x} for k,x in v['files_sha256'].items()]
  js('engineering/'+name,public,fp,'failure_scalar_and_hash_projection_no_logs_or_rawpaths')
 if 'component_aggregates' in optional:
  cf=('state','purpose','full_prediction_seal_sha256','source_checkpoint_sha256','plan_sha256','selected_n','eligible_counts_by_platform','selection','overlap_threshold_descriptive_only','independent_component_definition','overall','platforms','no_GT_instance_identity_or_task_improvement_claim','no_G_threshold_plan_or_adoption_change','seconds')
  assert set(component)==set(cf)
  aggregatekeys={'n','high_overlap_both_ge_0p8','high_overlap_and_no_extra_supported_independent_component','source_coverage_fraction_mean','alt_from_source_component_fraction_mean','extra_independent_supported_component_count_total'}
  assert set(component['overall'])==aggregatekeys and set(component['platforms'])=={'drone','quad'}
  assert all(set(x)==aggregatekeys for x in component['platforms'].values())
  js('audits/COMPONENT_DIAGNOSTIC_PUBLIC_AGGREGATES.json',{k:component[k] for k in cf},optional['component_aggregates'])
  js('protocol/COMPONENT_DIAGNOSTIC_PLAN.json',{k:v for k,v in componentplan.items() if not k.endswith('_path')},optional['component_plan'],'diagnostic_plan_paths_removed_original_hash_retained')
  for rel in ('crossplatform/round7/experiment/diagnose_competing_component.py','crossplatform/round7/experiment/COMPONENT_DIAGNOSTIC_METHOD.md'):exact(rel)
  fp=local('crossplatform/round7/experiment/COMPONENT_DIAGNOSTIC_PREPARATION.json');js('engineering/COMPONENT_DIAGNOSTIC_PREPARATION.json',read(fp),fp)
 if 'cross_stage_reproducibility' in optional:
  v=read(optional['cross_stage_reproducibility'])
  # This optional root-written file must already be aggregate-only. No rows,
  # identifiers or coordinate vectors are allowed anywhere in its JSON tree.
  def aggregate_only(x):
   if isinstance(x,dict):
    for k,value in x.items():
     assert not re.search(r'(^|_)(id|ids|row|rows|scene|frame|caption|boxes|box|coordinates|coords)(_|$)',k,re.I),('Nonaggregate reproducibility field',k)
     aggregate_only(value)
   elif isinstance(x,list):
    assert all(not isinstance(a,(list,dict)) for a in x),'No row/object arrays in aggregate-only file'
    for value in x:aggregate_only(value)
  aggregate_only(v);js('audits/CROSS_STAGE_REPRODUCIBILITY_PUBLIC.json',v,optional['cross_stage_reproducibility'],'root_written_aggregate_only_no_original_rows_IDs_or_coordinates')
 js('source/SOURCE_EPOCH100_PROVENANCE.json',{k:train[k] for k in ('state','primary_epoch','primary_checkpoint_sha256','completed_optimizer_steps','plan_sha256','source_train_contract','source_val_contract','no_joint_checkpoint_initialization','frozen_generic_text_weights_unchanged','no_Drone_or_Quad_data_access','sourceVAL_secondary_only_not_primary_selection','source_val_epoch100_seal_sha256')},local('crossplatform/round6/waymo_source100/TRAINING_COMPLETE.json'))
 inp=read(E/'input_audit/INPUT_RECEIPT.json');scene_counts=[]
 for key,v in inp['by_scene'].items():scene_counts.append({'scene_hash':sha(('round7-public-scene-v1|'+key).encode())[:16],**{k:x for k,x in v.items() if k!='scene'}})
 js('data/INPUT_AGGREGATES.json',{'full_n':inp['full_n'],'full_by_platform':inp['full_by_platform'],'smoke_n':inp['smoke_n'],'smoke_unique_frames':inp['smoke_unique_frames'],'per_scene_counts':scene_counts,'input_plan_sha256':inp['input_plan_sha256'],'selection_code_sha256':inp['selection_code_sha256'],'parser_sha256':inp['frozen_target_color_parser_module_sha256'],'original_input_receipt_sha256':file_sha(E/'input_audit/INPUT_RECEIPT.json')},E/'input_audit/INPUT_RECEIPT.json','counts_scenehash_only_no_originalscene_orcaption')
 noun=read(E/'input_audit/FIRST_NOUN_REGEX_AUDIT.json');assert not noun['unmatched_ids'];js('data/FIRST_NOUN_REGEX_COUNTS.json',{k:noun[k] for k in ('scope','actual_adapter_source_sha256','lexicon','by_platform','no_GT_fallback')},E/'input_audit/FIRST_NOUN_REGEX_AUDIT.json')
 js('protocol/INPUT_PLAN.json',read(E/'input_audit/INPUT_PLAN.json'),E/'input_audit/INPUT_PLAN.json');pr=read(E/'input_audit/PROJECTION_ROOT_REVIEW.json');js('audits/ROOT_RGB_POINT_PROJECTION_REVIEW.json',{k:v for k,v in pr.items() if k!='files'},E/'input_audit/PROJECTION_ROOT_REVIEW.json','rootwritten_scalar_observations_no_images_paths_or_matrices')
 # Preserve every case/negative delta via explicit scalarCSV, not private rows.
 raw=read(bound['full_scalar_summary'].parent/'RESULTS_PRIVATE.json');assert raw['n_samples']==5912 and len(raw['rows_PRIVATE'])==5912
 stream=io.StringIO();columns=['case_id','platform','scene_hash','source_query','source_iou','common_eligible_all_modes','arm']+list(ARM_FIELDS);writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();pseudo=io.StringIO();pw=csv.DictWriter(pseudo,fieldnames=['case_id','platform','scene_hash','mode']+list(PSEUDO_FIELDS));pw.writeheader()
 for row in raw['rows_PRIVATE']:
  assert re.fullmatch(r'[0-9a-f]{16}',row['id']);scenehash=sha(('round7-public-scene-v1|'+row['platform']+'/'+row['scene']).encode())[:16];assert set(row['arms'])==set(ARMS)
  for name,a in row['arms'].items():
   v={'case_id':row['id'],'platform':row['platform'],'scene_hash':scenehash,'source_query':row['source_query'],'source_iou':row['source_iou'],'common_eligible_all_modes':row['common_eligible_all_modes'],'arm':name,**{k:a[k] for k in ARM_FIELDS}};writer.writerow(clean(v))
  for mode,a in row['pseudo_diagnostic'].items():pw.writerow(clean({'case_id':row['id'],'platform':row['platform'],'scene_hash':scenehash,'mode':mode,**{k:a[k] for k in PSEUDO_FIELDS}}))
 csv_bytes=stream.getvalue().encode();pseudo_bytes=pseudo.getvalue().encode();case_gz=gzip.compress(csv_bytes,compresslevel=9,mtime=0);pseudo_gz=gzip.compress(pseudo_bytes,compresslevel=9,mtime=0);assert gzip.decompress(case_gz)==csv_bytes and gzip.decompress(pseudo_gz)==pseudo_bytes
 put('results/FULL_CASE_SCALARS.csv.gz',case_gz,bound['full_scalar_summary'].parent/'RESULTS_PRIVATE.json','explicit_scalarCSV_lossless_gzip_level9_mtime0_opaque_case_scenehash_no_coordinates_or_text');put('results/FULL_PSEUDO_DIAGNOSTIC_SCALARS.csv.gz',pseudo_gz,bound['full_scalar_summary'].parent/'RESULTS_PRIVATE.json','derived_error_scalarCSV_lossless_gzip_level9_mtime0_no_GT_or_predicted_coordinates')
 js('results/SCALAR_TABLE_SCHEMA.json',{'encoding':'UTF-8','CSV_dialect':'csv.DictWriter default excel; header and CRLF line endings','null_encoding':'empty CSV field for None; all scalar float/bool representations unchanged','compression':'gzip.compress(bytes,compresslevel=9,mtime=0); no original filename; decompression equality asserted','compression_runtime':{'python':platform.python_version(),'zlib':zlib.ZLIB_VERSION},'case_table':{'path':'FULL_CASE_SCALARS.csv.gz','row_count':len(raw['rows_PRIVATE'])*len(ARMS),'columns':columns,'uncompressed_bytes':len(csv_bytes),'uncompressed_sha256':sha(csv_bytes),'compressed_sha256':sha(case_gz),'ordering':'sealed scored sample order, then saved arm order'},'pseudo_table':{'path':'FULL_PSEUDO_DIAGNOSTIC_SCALARS.csv.gz','row_count':len(raw['rows_PRIVATE'])*len(MODES),'columns':['case_id','platform','scene_hash','mode']+list(PSEUDO_FIELDS),'uncompressed_bytes':len(pseudo_bytes),'uncompressed_sha256':sha(pseudo_bytes),'compressed_sha256':sha(pseudo_gz),'ordering':'sealed scored sample order, then saved mode order'},'case_ID':'opaque original output hashID retained for evidence join; no raw scene/frame or text','scene_hash':'first16hex SHA256(round7-public-scene-v1|platform/scene); physical scene string not exported'})
 # Optional actual failures: hash and status only, never raw logs/tracebacks/messages.
 for i,spec in enumerate(sources.get('runner_failure_receipts',[]),1):
  fp=local(spec['path']);assert file_sha(fp)==spec['sha256'];v=read(fp);js(f'engineering/RUNNER_FAILURE_{i:03d}.json',{'source_receipt_sha256':file_sha(fp),**{k:v[k] for k in ('state','exception_type','phase','n_completed','no_target_GT_access') if k in v}},fp,'failure_status_hash_projection_no_rawpaths_ortraceback')
 js('PUBLICATION_READINESS.json',ready,E/'PUBLICATION_READINESS.json');js('EXPORT_FINAL_SOURCE_HASHES.json',{k:{'source_sha256':sources[k]['sha256']} for k in required+tuple(optional)},E/'EXPORT_FINAL_SOURCES.json','final_sourcehashes_withoutprivate_paths')
 put('README.md',b'# R7: Waymo-only source to Drone/Quad, single-model multimodal pseudo regression\n\nSee REPORT.md, REPRODUCTION.md and DECISION.json. This is the complete fixed5912 targetVAL evaluation, with an intentionally enriched20sample engineering smoke. Historical target research exposure is explicit. Sourceweights are genuine Waymo-onlyepoch100. No teacher/EMA, no heading/PCA, fixedsourcequery, sizeweightszero. Real artificial positive control passes the registered equivalence gate only under strict FP32; original-main precision fails and V1/V2 failures are retained. It does not prove G quality or task benefit. Compare source, directG, parameterSGD, exactoutputspaceSGD and PC-only/shuffledcolor with the sameinformation. Results preserve allcase/negative scalars and requested/finite/accepted/nonzero counts. Case tables are lossless .csv.gz with mtime0; SCALAR_TABLE_SCHEMA.json records columns, rowcounts and uncompressed hashes.\n\nThis finite bundle excludes raw captions, physicalscene/frame identifiers, GT/prediction coordinates, media/pointclouds, privateindices, NPZcaches and model/optimizer weights. Code is exact except explicitly recorded path-only portability projections; original/export hashes are in EXPORT_PROVENANCE.json. Frozenhistorical locks retain their original pending/synthetic state and are not substitutes for final experimental evidence.\n')
 js('EXPORT_PROVENANCE.json',provenance)
 # Finite pending set only. Allprivacy gates finish before creating the candidate.
 for dest,data in pending.items():assert len(data)<95*1024**2,(dest,'Unexpectedlargepayload');assert not any(x.encode() in data for x in raw_scenes),'Scene leaked in finalbytes'
 out.mkdir(parents=True)
 for dest,data in pending.items():fp=out/dest;fp.parent.mkdir(parents=True,exist_ok=True);fp.write_bytes(data)
 manifest={'scope':'finite_R7_code_protocol_aggregate_and_case_scalars_no_private_data_or_weights','files':[{'path':k,'bytes':len(v),'sha256':sha(v)} for k,v in sorted(pending.items())],'root_final_review_sha256':file_sha(bound['root_final_review']),'full_original_prediction_seal_sha256':file_sha(bound['full_prediction_seal']),'source_checkpoint_sha256':SOURCE_SHA};(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'files':len(pending)+1,'manifest_sha256':file_sha(out/'MANIFEST.json'),'output':str(out)}))
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=E/'public_export');main(ap.parse_args())
