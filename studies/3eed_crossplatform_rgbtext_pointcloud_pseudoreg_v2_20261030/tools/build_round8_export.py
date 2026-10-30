"""Finite R8 public export candidate builder; root final artifacts required.
No model/GPU/GT scoring, network or Git action. Never distribute private data,
weights, captions, original sample IDs, prediction coordinates or embeddings.
"""
from pathlib import Path,PurePosixPath
import argparse,csv,gzip,hashlib,io,json,math,platform,re,zlib
from functools import lru_cache
B=Path(__file__).resolve().parents[1];R=B/'research/round8';E=R/'experiment';C=B/'crossplatform/round8';HOME_PREFIX=str(Path.home())+'/'
SOURCE_SHA='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
CLIP_REVISION='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
CLIP_ORIGINAL_SHA='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
CLIP_SAFE_SHA='7a7177862f66645c5e923e31e07318fd609769378a197a21903fbbcc07e446b1'
STRICT={'cudnn_allow_tf32':False,'matmul_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}
MODES=('rgb_fulltext','pc_only','semantic_shuffle');KINDS=('association_only','direct_G','param1','param3','outputspace1','outputspace3');ARMS=('source',)+tuple(m+'__'+k for m in MODES for k in KINDS)
CODE_FILES=(
 'crossplatform/round8/association_geometry.py','crossplatform/round8/semantic_encoder.py','crossplatform/round8/run_association.py','crossplatform/round8/check_association_geometry_cpu.py','crossplatform/round8/check_semantic_encoder_cpu.py','crossplatform/round8/check_run_association_cpu.py','crossplatform/round8/fetch_semantic_assets.py','crossplatform/round8/prepare_safe_semantic_weights.py',
 'crossplatform/round8/history/pre_FOV_fix/association_geometry.py','crossplatform/round8/history/pre_FOV_fix/check_association_geometry_cpu.py','crossplatform/round8/history/pre_floor_pixel_fix/association_geometry.py','crossplatform/round8/history/pre_floor_pixel_fix/check_association_geometry_cpu.py',
 'research/round8/experiment/prepare_inputs.py','research/round8/experiment/review_samples.py','research/round8/evaluation/score_crossdomain.py','research/round8/evaluation/audit_unlabelled_predictions.py','research/round8/evaluation/check_scoring_synthetic.py','research/round8/evaluation/history/draft3683/score_crossdomain.py','research/round8/evaluation/history/draft3683/check_scoring_synthetic.py',
 'crossplatform/round7/regression_adapter.py','crossplatform/round7/camera_projection.py','crossplatform/round7/experiment/runner_helpers.py','crossplatform/round6/factory.py','research/round5/audit/independent_core_v2.py','research/round6/source_validation_scoring/batch_yaw_iou.py','tools/obb_iou.py','tools/build_round8_export.py','tools/build_round8_export_v1_snapshot.py','research/round8/evaluation/diagnose_scored_results.py','research/round8/evaluation/history/diagnose_scored_results_pre_review.py','research/round8/evaluation/audit_scored_results.py','research/round8/evaluation/check_reduction_synthetic.py','research/round8/evaluation/check_postseal_diagnostic_cpu.py','research/round8/experiment/analyze_smoke_G_mechanism.py')
DOC_FILES=('crossplatform/round8/ASSOCIATION_GEOMETRY_CONTRACT.md','crossplatform/round8/SEMANTIC_INTERFACE.md','crossplatform/round8/RUNNER_INTERFACE.md','research/round8/INDEPENDENT_PROTOCOL_REVIEW.md','research/round8/PROTOCOL_SCALE_DECISION_v2.md','research/round8/experiment/BRIEF.md','research/round6/UPSTREAM_COMPATIBILITY.patch','research/round8/experiment/SMOKE_G_MECHANISM_DIAGNOSTIC.md','research/round8/evaluation/EXECUTION_COMMANDS.md')
STATIC_JSON=('crossplatform/round8/ASSOCIATION_GEOMETRY_LOCK.json','crossplatform/round8/CPU_ASSOCIATION_GEOMETRY_FIXTURE.json','crossplatform/round8/CPU_SEMANTIC_FIXTURE.json','crossplatform/round8/CPU_R8_RUNNER_FIXTURE.json','crossplatform/round8/RUNNER_READY.json','crossplatform/round8/history/pre_FOV_fix/ASSOCIATION_GEOMETRY_LOCK.json','crossplatform/round8/history/pre_FOV_fix/CPU_ASSOCIATION_GEOMETRY_FIXTURE.json','crossplatform/round8/history/pre_floor_pixel_fix/ASSOCIATION_GEOMETRY_LOCK.json','crossplatform/round8/history/pre_floor_pixel_fix/CPU_ASSOCIATION_GEOMETRY_FIXTURE.json','research/round8/INDEPENDENT_PROTOCOL_REVIEW.json','research/round8/INDEPENDENT_G_CODE_REVIEW.json','research/round8/evaluation/SYNTHETIC_SCORING_FIXTURE.json','research/round8/evaluation/history/draft3683/SYNTHETIC_SCORING_FIXTURE.json','research/round8/experiment/REVIEW_TOOL_RECEIPT.json','research/round8/experiment/POSTSEAL_DIAGNOSTIC_PLAN_v1.json','research/round8/experiment/SMOKE_G_MECHANISM_DIAGNOSTIC.json','research/round8/evaluation/EVALUATION_PROGRAM_LOCK.json','research/round8/evaluation/EVALUATION_PROGRAM_LOCK_v2.json','research/round8/evaluation/POSTSEAL_DIAGNOSTIC_INDEPENDENT_REVIEW.json','research/round8/evaluation/POSTSEAL_DIAGNOSTIC_SYNTHETIC_REVIEW.json','research/round8/evaluation/SYNTHETIC_REDUCTION_FIXTURE.json')
SCORE_FIELDS=('state','n','stage','summary','candidate_and_group_coverage','max_selected_3D_vs_batch_abs_error','strict_threshold_disagreements','geometry_counts','scope','geometry_scope','effective_scene_limit','seconds')
AUDIT_FIELDS=('state','source_auditor_sha256','prediction_seal_sha256','phase','n','summary','max_allquery_kernel_equivalence_error_m','registered_kernel_tolerance_m','max_selected_interval_excess_m','hashes_all_verified','matched_fresh_strictFP32_source','CLIP_frozen_resources_and_restore_receipts_verified','raw_target64_actualtarget32_qchange_and_paramchange_separate','no_target_metadata_GT_scoringindex_or_task_IoU_opened','no_GPU_model_or_taskscore_by_auditor','quality_boundary','seconds')
ARM_FIELDS=('mode','delta_vs_outputspace','raw_nonpositive_dimensions','query_index','status','iou','delta_vs_q0','delta_vs_association','delta_vs_directG','association_iou','directG_iou','selected_changed_vs_q1_anchor','parameter_steps_executed','nonzero_gradient_steps','nonzero_parameter_update')
REDUCTION_FIELDS=('state','auditor_sha256','results_sha256','source_scorer_sha256','n','aggregate_cells_checked','max_aggregate_float_error','signed_negative_counts_by_mode_arm_contrast','all19_fixed_query_and_delta_arithmetic_verified','source256_GTassistedcoverage_aggregation_verified','geometry_scope','no_model_GPU_targetmetadata_GT_or_new_prediction_scoring')
G_FIELDS=('q_changed','accepted_axis','raw_consensus_delta_nonzero','final_float64_pseudo_delta_nonzero','actual_float32_pseudo_delta_nonzero','r_global','selected_group')
STATIC_JSON += ('research/round8/evaluation/FINAL_SCORE_RESULT_REVIEW.json',
                'research/round8/experiment/FINAL_MECHANISM_REVIEW.json')
DOC_FILES += ('research/round8/experiment/FINAL_MECHANISM_REVIEW.md',)
REQUIRED=('smoke_prediction_seal','full_prediction_seal','smoke_scalar_summary','full_scalar_summary','smoke_unlabelled_audit','full_unlabelled_audit','smoke_score_release','full_score_release','root_report','root_decision','root_final_review','full_postseal_diagnostic','smoke_score_reduction_audit','full_score_reduction_audit')
def validate_public_diagnostic_stats(diagnostic):
 # Require the corrected descriptive schema, including equal-scene macro and all
 # four strict-threshold transitions. No private rows/identities are admitted.
 fields={'n','defined_scene_clusters','delta_mean','equal_scene_delta_mean','gain_atol1e_minus8','harm_atol1e_minus8','tie_atol1e_minus8','delta_quantiles','strict_Acc_transitions'}
 for value in diagnostic['platforms'].values():
  assert set(value)=={'n','scene_clusters','subgroups','GT_assisted_coverage_categories','coverage_scope'}
  assert isinstance(value['coverage_scope'],str) and value['coverage_scope']
  for contrasts in value['subgroups'].values():
   for stats in contrasts.values():
    assert set(stats)==fields
    n=stats['n'];assert n>=0 and 0<=stats['defined_scene_clusters']<=value['scene_clusters']
    assert stats['gain_atol1e_minus8']+stats['harm_atol1e_minus8']+stats['tie_atol1e_minus8']==n
    assert (stats['defined_scene_clusters']==0)==(n==0)
    assert (stats['delta_mean'] is None)==(n==0) and (stats['equal_scene_delta_mean'] is None)==(n==0)
    assert set(stats['strict_Acc_transitions'])=={'0.25','0.5'}
    for counts in stats['strict_Acc_transitions'].values():
     assert set(counts)=={'gained','lost','retained_success','retained_failure'} and all(v>=0 for v in counts.values()) and sum(counts.values())==n

sha=lambda b:hashlib.sha256(b).hexdigest()
def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def local(rel):
 q=PurePosixPath(rel);assert q.parts and not q.is_absolute() and '..' not in q.parts
 p=B/q.as_posix();assert p.resolve().is_relative_to(B.resolve()) and p.is_file(),rel;return p

def main(args):
 out=args.output.resolve();assert out.is_relative_to(B.resolve()) and not out.exists(),'Refuse overwrite or destination outside workspace'
 ready=read(E/'PUBLICATION_READINESS.json')
 for key in ('smoke40_and_full5912_predictions_sealed','smoke_and_full_independent_scored','smoke_and_full_unlabelled_audits_passed','actual_fixed10_unlabelled_visual_review_completed','root_final_review_passed','report_decision_privacy_reviewed','no_private_data_weights_identifiers_or_coordinates_in_export'):assert ready.get(key) is True,('Unfinished final gate',key)
 sources=read(E/'EXPORT_FINAL_SOURCES.json');bound={}
 for key in REQUIRED:
  spec=sources[key];assert set(spec)=={'path','sha256'};p=local(spec['path']);assert file_sha(p)==spec['sha256'];bound[key]=p
 stage_seals={};scored_seals={}
 for phase,n in [('smoke',40),('full',5912)]:
  sp=bound[phase+'_prediction_seal'];se=read(sp);stage_seals[phase]=se;planpath=E/(phase.upper()+'_PLAN_LOCK.json');plan=read(planpath)
  assert se['state']=='sealed_R8_strictFP32_crossdomain_predictions_no_target_GT_scoring' and se['phase']==phase and se['n_samples']==se['n_prediction_files']==n and se['n_arms']==19
  assert se['source_checkpoint_sha256']==SOURCE_SHA and se['sourceQ0_not_retagged_as_associationQ1'] and se['all_19arms_sealed_before_any_target_GT_scoring'] and se['grounder_restored_and_CLIP_frozen'] and se['no_backbone_or_source_retraining_or_EMA']
  assert se['strict_FP32_flags']==plan['strict_FP32_flags']==STRICT and se['plan_sha256']==file_sha(planpath) and se['manifest_sha256']==plan['manifest_sha256']
  assert se['program_sha256_base']=='workspace_root'
  for rel,digest in se['program_sha256'].items():assert file_sha(local(rel))==digest
  assert se['CLIP_resource']['revision']==CLIP_REVISION and se['CLIP_resource']['derived_safetensors_sha256']==CLIP_SAFE_SHA
  for name,digest in se['files'].items():
   assert PurePosixPath(name).name==name and file_sha(sp.parent/name)==digest
  for name in ('trace.json','registered_plan.json','manifest_input.json','repeat_restore_control.json'):assert name in se['files']
  assert len([name for name in se['files'] if name.endswith('.npz')])==n
  repeat=read(sp.parent/'repeat_restore_control.json');assert all(repeat[k] is True for k in ('source_repeat_bitexact','source_final_Conv_bridge_bitexact','post_all_modes_grounder_fullstate_restored','frozen_CLIP_fullstate_unchanged')) and repeat['strict_FP32_flags']==STRICT
  ap=bound[phase+'_unlabelled_audit'];audit=read(ap);assert audit['state']=='R8_unlabelled_engineering_audit_pass_not_pseudo_quality_score' and audit['prediction_seal_sha256']==file_sha(sp) and audit['n']==n and audit['phase']==phase
  assert audit['hashes_all_verified'] and audit['matched_fresh_strictFP32_source'] and audit['CLIP_frozen_resources_and_restore_receipts_verified'] and audit['no_target_metadata_GT_scoringindex_or_task_IoU_opened']
  assert audit['max_allquery_kernel_equivalence_error_m']<=audit['registered_kernel_tolerance_m']==5e-5
  scored=bound[phase+'_scalar_summary'];seal=read(scored.parent/'SEALED.json');scored_seals[phase]=seal;summary=read(scored)
  assert summary['state']=='R8_scored_after_fullstage_prediction_seal' and summary['n']==seal['n']==n and summary['strict_threshold_disagreements']==0 and summary['max_selected_3D_vs_batch_abs_error']<1e-8
  for name in ('SCALAR_SUMMARY.json','RESULTS_PRIVATE.json','SCORER_LOCK.json'):assert seal['files'][name]==file_sha(scored.parent/name)
  grant=read(bound[phase+'_score_release']);assert grant['CPU_SCORE_RELEASE'] is True and grant['both_smoke_and_full_sealed_before_any_stage_GT_score'] is True and grant['prediction_seal_sha256']==file_sha(sp) and grant['manifest_sha256']==se['manifest_sha256'] and grant['source_checkpoint_sha256']==SOURCE_SHA
  lock=read(scored.parent/'SCORER_LOCK.json');assert lock['prediction_seal_sha256']==file_sha(sp) and lock['release_sha256']==file_sha(bound[phase+'_score_release']) and lock['scorer_sha256']==grant['scorer_program_sha256']==file_sha(local('research/round8/evaluation/score_crossdomain.py'))
 # The independent reduction audit binds sealed score rows, not readiness alone.
 for phase,n in [('smoke',40),('full',5912)]:
  reduction=read(bound[phase+'_score_reduction_audit']);resultpath=bound[phase+'_scalar_summary'].parent/'RESULTS_PRIVATE.json'
  assert set(reduction)==set(REDUCTION_FIELDS) and reduction['state']=='independent_R8_score_reduction_audit_pass' and reduction['n']==n
  assert reduction['results_sha256']==scored_seals[phase]['files']['RESULTS_PRIVATE.json']==file_sha(resultpath)
  assert reduction['auditor_sha256']==file_sha(local('research/round8/evaluation/audit_scored_results.py'))
  assert reduction['source_scorer_sha256']==file_sha(local('research/round8/evaluation/score_crossdomain.py'))
  assert reduction['aggregate_cells_checked']>0 and 0<=reduction['max_aggregate_float_error']<=1e-12
  assert reduction['all19_fixed_query_and_delta_arithmetic_verified'] and reduction['source256_GTassistedcoverage_aggregation_verified'] and reduction['no_model_GPU_targetmetadata_GT_or_new_prediction_scoring']
 visualpath=E/'SMOKE_ROOT_VISUAL_REVIEW.json';visual=read(visualpath)
 assert visual['state']=='real_smoke_prerequisite_pass_not_task_quality_proof' and visual['smoke_seal_sha256']==file_sha(bound['smoke_prediction_seal']) and visual['independent_unlabelled_audit_sha256']==file_sha(bound['smoke_unlabelled_audit'])
 assert visual['actual_visual_review']['fixed_hash_cases_viewed']==list(range(1,11)) and visual['actual_visual_review']['no_GT_or_annotation_access'] is True and visual['source_and_CLIP_restore_pass']
 diagnosticplanpath=E/'POSTSEAL_DIAGNOSTIC_PLAN_v1.json';diagnosticplan=read(diagnosticplanpath)
 assert diagnosticplan['state']=='registered_before_any_R8_target_GT_score' and diagnosticplan['full_plan_sha256']==file_sha(E/'FULL_PLAN_LOCK.json')
 assert diagnosticplan['no_changes_to_G_or_predictions'] and diagnosticplan['GT_is_offline_only'] and diagnosticplan['eligibility_and_hyperparameters_not_refit_using_targetlabels']
 dp=bound['full_postseal_diagnostic'];diagnostic=read(dp);diagnosticseal=read(dp.parent/'SEALED.json')
 assert diagnosticseal['files']['SCALAR_DIAGNOSTIC.json']==file_sha(dp) and diagnosticseal['n']==diagnostic['n']==5912
 assert diagnostic['state']=='descriptive_postseal_diagnostic_no_configuration_selection' and diagnostic['scored_seal_sha256']==file_sha(bound['full_scalar_summary'].parent/'SEALED.json')
 assert diagnostic['plan_sha256']==file_sha(diagnosticplanpath) and diagnostic['program_sha256']==file_sha(local('research/round8/evaluation/diagnose_scored_results.py'))
 assert set(diagnostic)=={'state','scored_seal_sha256','plan_sha256','program_sha256','n','platforms','limits'} and set(diagnostic['platforms'])=={'drone','quad'}
 validate_public_diagnostic_stats(diagnostic)
 smoke_mechanism=read(E/'SMOKE_G_MECHANISM_DIAGNOSTIC.json')
 assert smoke_mechanism['state']=='postseal_unlabelled_smoke_mechanism_scalars_no_task_quality_claim' and smoke_mechanism['smoke_seal_sha256']==file_sha(bound['smoke_prediction_seal'])
 assert smoke_mechanism['trace_sha256']==stage_seals['smoke']['files']['trace.json'] and smoke_mechanism['root_visual_review_sha256']==file_sha(visualpath)
 assert smoke_mechanism['frozen_code_config_predictions_not_changed'] and smoke_mechanism['no_target_GT_score_GPU_or_full_running_outputs_read']
 review=read(bound['root_final_review']);assert review['publication_ready'] is True and review['report_sha256']==file_sha(bound['root_report']) and review['decision_sha256']==file_sha(bound['root_decision'])
 trainpath=local('crossplatform/round6/waymo_source100/TRAINING_COMPLETE.json');train=read(trainpath);assert train['primary_epoch']==100 and train['completed_optimizer_steps']==15300 and train['primary_checkpoint_sha256']==SOURCE_SHA and train['no_joint_checkpoint_initialization'] and train['no_Drone_or_Quad_data_access']
 resourcepath=C/'SEMANTIC_RESOURCE_RECEIPT.json';resource=read(resourcepath);safe=resource['safe_loading'];assert resource['model_id']=='openai/clip-vit-base-patch32' and resource['revision']==CLIP_REVISION and safe['original_pytorch_sha256']==CLIP_ORIGINAL_SHA and safe['derived_safetensors_sha256']==CLIP_SAFE_SHA and safe['tensor_key_count']==398 and safe['all_loaded_tensor_values_bitexact_after_safetensors_roundtrip'] and safe['strict_model_load_all_learned_keys_and_shapes']
 for name,digest in resource['program_sha256'].items():assert file_sha(C/name)==digest
 # Only the private input whitelist is scanned for privacy checks, never GT/index.
 private=read(E/'full5912_gpu_inputs.json');assert len(private)==5912;raw_ids={x['id'] for x in private};raw_scenes={x['scene'] for x in private};captions={sha(x['caption'].encode()) for x in private};raw_paths={x[k] for x in private for k in ('pointpath','imagepath','metapath')}
 scenes=sorted({(r['platform'],r['scene']) for r in private});scene_alias={key:'scene_'+str(i+1).zfill(2) for i,key in enumerate(scenes)}
 forbidden={'rows_PRIVATE','by_scene_PRIVATE','cluster_oracles_PRIVATE','caption','captions','utterance','utterances','scene','frame','scan_id','metapath','pointpath','imagepath','meta_path','image_path','lidar_path','bbox_3d','gt_bbox','gt_bboxes','source_boxes_raw6','source_boxes_eval6','arm_boxes_raw6','arm_boxes_eval6','pseudo_target6','pseudo_target6_float64','raw_consensus_center','image_intrinsic','image_extrinsic','camera_distortion','pose','positive_map','points','point_clouds','embeddings','proj_queries','proj_tokens','query_features','center_features','size_features'}
 @lru_cache(maxsize=32768)
 def textcheck(s):
  assert HOME_PREFIX not in s and 'file://' not in s and 'X-Amz-' not in s and 'Signature=' not in s,'Unprojected private path or signed URL'
  assert not any(x in s for x in raw_scenes) and not any(x in s for x in raw_ids),'Original physical scene or sampleID'
  assert s not in raw_paths and sha(s.encode()) not in captions,'Original path or caption'
  return s
 def clean(v):
  if isinstance(v,dict):
   assert not forbidden&set(v) and not any(k.endswith('_PRIVATE') for k in v),('Forbidden/private payload fields',set(v));return {textcheck(str(k)):clean(x) for k,x in v.items()}
  if isinstance(v,(list,tuple)):return [clean(x) for x in v]
  if isinstance(v,str):return textcheck(v.replace(str(B)+'/', '<workspace>/').replace(HOME_PREFIX,'<local-home>/'))
  if isinstance(v,float):assert math.isfinite(v),'NaN/inf cannot silently enter export'
  return v
 pending={};provenance=[]
 def put(dest,data,src=None,mode='generated'):
  q=PurePosixPath(dest);assert q.parts and not q.is_absolute() and '..' not in q.parts and dest not in pending
  assert q.suffix.lower() not in ('.pt','.pth','.bin','.npy','.npz','.safetensors','.jpg','.jpeg','.png','.wav','.mp4','.zip','.so')
  pending[dest]=data
  if src:provenance.append({'path':dest,'source':src.relative_to(B).as_posix(),'source_sha256':file_sha(src),'export_sha256':sha(data),'mode':mode})
 def js(dest,v,src=None,mode='explicit_field_projection'):
  put(dest,(json.dumps(clean(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode(),src,mode)
 def exact(rel,dest=None):
  p=local(rel);original=p.read_text();s=original
  if p.suffix=='.py':
   anchor='Path(__file__).resolve().parents['+str(len(PurePosixPath(rel).parts)-1)+']';s=s.replace('Path('+repr(str(B))+')',anchor).replace('Path('+json.dumps(str(B))+')',anchor)
  elif p.suffix=='.md':s=s.replace(str(B),'<workspace>')
  assert HOME_PREFIX not in s and not any(x in s for x in raw_scenes) and not any(x in s for x in raw_ids),'Unprojected code/document private identifier'
  assert not re.search(r'https?://[^\s\"\']*[?&](?:X-Amz-Signature|Signature)=',s),'Actual signed URL in code/document'
  put(dest or rel,s.encode(),p,'byte_exact' if s==original else 'path_only_portability_projection_original_and_export_SHA_retained_no_algorithm_change')
 def paths_removed(v):
  if isinstance(v,dict):return {k:paths_removed(x) for k,x in v.items() if not k.endswith('_path') and not k.endswith('_dir') and k not in ('weights_relative_private_location','programroot','workspace_root')}
  if isinstance(v,list):return [paths_removed(x) for x in v]
  return v
 for rel in CODE_FILES+DOC_FILES:exact(rel)
 for rel in STATIC_JSON:js(rel,paths_removed(read(local(rel))),local(rel),'historical_or_synthetic_receipt_paths_removed_original_hash_retained')
 exact('research/round8/experiment/REPRODUCTION.md','REPRODUCTION.md');exact(sources['root_report']['path'],'REPORT.md');js('DECISION.json',read(bound['root_decision']),bound['root_decision']);js('audits/ROOT_FINAL_REVIEW.json',{k:review[k] for k in ('publication_ready','report_sha256','decision_sha256')},bound['root_final_review'])
 for phase in ('smoke','full'):
  p=E/(phase.upper()+'_PLAN_LOCK.json');js('protocol/'+p.name,paths_removed(read(p)),p,'plan_private_paths_and_dirs_omitted_original_planSHA_in_provenance')
  sp=bound[phase+'_prediction_seal'];se=stage_seals[phase];fields=('state','phase','n_samples','n_prediction_files','n_arms','manifest_sha256','plan_sha256','source_checkpoint_sha256','source_model_state_fingerprint','source_strict_state_entries','CLIP_resource','CLIP_state_fingerprint','strict_FP32_flags','program_sha256','program_sha256_base','sourceQ0_not_retagged_as_associationQ1','all_19arms_sealed_before_any_target_GT_scoring','grounder_restored_and_CLIP_frozen','no_backbone_or_source_retraining_or_EMA','elapsed_seconds')
  js('results/'+phase.upper()+'_PREDICTION_SEAL_SUMMARY.json',{**{k:se[k] for k in fields},'original_seal_sha256':file_sha(sp)},sp,'seal_summary_no_raw_sample_filekeys_or_prediction_arrays')
  p=bound[phase+'_scalar_summary'];v=read(p);assert set(v)==set(SCORE_FIELDS);js('results/'+phase.upper()+'_SCALAR_SUMMARY.json',{k:v[k] for k in SCORE_FIELDS},p)
  p=bound[phase+'_unlabelled_audit'];v=read(p);js('audits/'+phase.upper()+'_UNLABELLED_AUDIT.json',{k:v[k] for k in AUDIT_FIELDS},p)
  p=bound[phase+'_score_release'];v=read(p);keep=('CPU_SCORE_RELEASE','both_smoke_and_full_sealed_before_any_stage_GT_score','stage','prediction_seal_sha256','manifest_sha256','scoring_index_sha256','source_checkpoint_sha256','scorer_program_sha256');js('protocol/'+phase.upper()+'_CPU_SCORE_RELEASE.json',{k:v[k] for k in keep},p)
 js('audits/SMOKE_ROOT_VISUAL_REVIEW.json',visual,visualpath,'ten_case_fixed_hash_unlabelled_root_observations_only_no_images')
 for phase in ('smoke','full'):
  p=bound[phase+'_score_reduction_audit'];v=read(p);js('audits/'+phase.upper()+'_SCORE_REDUCTION_AUDIT.json',{**{k:v[k] for k in REDUCTION_FIELDS},'verified_source_score_seal_sha256':file_sha(bound[phase+'_scalar_summary'].parent/'SEALED.json'),'verified_source_prediction_seal_sha256':file_sha(bound[phase+'_prediction_seal']),'original_reduction_receipt_sha256':file_sha(p)},p,'actual_reduction_aggregate_receipt_plus_explicit_builder_verified_seal_hashes_no_private_score_rows')
 js('results/FULL_POSTSEAL_SCALAR_DIAGNOSTIC.json',diagnostic,bound['full_postseal_diagnostic'],'sealed_public_aggregate_only_no_private_scene_rows_or_signed_tails')
 js('resources/SEMANTIC_RESOURCE_RECEIPT.json',paths_removed(resource),resourcepath,'official_model_hashes_processor_provenance_and_CPU_checks_no_assets_or_weights')
 keep=('state','primary_epoch','primary_checkpoint_sha256','completed_optimizer_steps','plan_sha256','no_joint_checkpoint_initialization','frozen_generic_text_weights_unchanged','no_Drone_or_Quad_data_access','sourceVAL_secondary_only_not_primary_selection','source_val_epoch100_seal_sha256');js('source/SOURCE_EPOCH100_PROVENANCE.json',{k:train[k] for k in keep},trainpath)
 inp=read(E/'INPUT_RECEIPT.json');scene_counts=[]
 for row in inp['by_scene_PRIVATE']:scene_counts.append({'scene_alias':scene_alias[(row['platform'],row['scene'])],**{k:v for k,v in row.items() if k!='scene'}})
 js('data/INPUT_AGGREGATES.json',{k:inp[k] for k in ('state','input_plan_sha256','selection_program_sha256','R7_full_source_sha256','R7_smoke20_source_sha256','descriptive_parser_sha256','output_files','ninekey_schema','full_n','full_by_platform','full_bytes_identical_to_R7_strict_inputs','smoke_n','smoke_by_platform','smoke_scene_n','R7smoke20_ID_overlap','R7smoke20_pointpath_overlap','selection_independent_of_color_and_GT','full_text_only_color_description','smoke_text_only_color_description','target_research_exposure','new_R8_source_forward_required_no_R7_modelcache_reuse')}|{'scene_counts':scene_counts},E/'INPUT_RECEIPT.json','counts_only_original_scene_names_replaced_with_ordinal_aliases')
 js('protocol/INPUT_PLAN_LOCK.json',read(E/'INPUT_PLAN_LOCK.json'),E/'INPUT_PLAN_LOCK.json')
 # All case and G scalar rows, including every negative, no original outputID.
 p=bound['full_scalar_summary'].parent/'RESULTS_PRIVATE.json';raw=read(p);rows=raw['rows_PRIVATE'];assert len(rows)==5912 and [x['id'] for x in rows]==[x['id'] for x in private]
 stream=io.StringIO();gstream=io.StringIO();columns=['case_alias','platform','scene_alias','source_query','source_iou','arm','negative_vs_source']+list(ARM_FIELDS);gcols=['case_alias','platform','scene_alias','mode']+list(G_FIELDS);cw=csv.DictWriter(stream,fieldnames=columns);gw=csv.DictWriter(gstream,fieldnames=gcols);cw.writeheader();gw.writeheader()
 for i,row in enumerate(rows):
  alias='case_'+str(i+1).zfill(6);scene=scene_alias[(row['platform'],row['scene'])];assert set(row['arms'])==set(ARMS) and set(row['G'])==set(MODES)
  for arm in ARMS:
   a=row['arms'][arm];assert set(a)==set(ARM_FIELDS);cw.writerow(clean({'case_alias':alias,'platform':row['platform'],'scene_alias':scene,'source_query':row['q_source'],'source_iou':row['source_q0_iou'],'arm':arm,'negative_vs_source':a['delta_vs_q0']<0,**a}))
  for mode in MODES:
   a=row['G'][mode];assert set(a)==set(G_FIELDS);gw.writerow(clean({'case_alias':alias,'platform':row['platform'],'scene_alias':scene,'mode':mode,**a}))
 tables={}
 for name,data,cols,n in [('FULL_CASE_SCALARS.csv.gz',stream.getvalue().encode(),columns,5912*19),('FULL_G_SCALARS.csv.gz',gstream.getvalue().encode(),gcols,5912*3)]:
  compressed=gzip.compress(data,compresslevel=9,mtime=0);assert gzip.decompress(compressed)==data;put('results/'+name,compressed,p,'lossless_explicit_scalar_CSV_level9_mtime0_no_original_IDs_or_coordinates');tables[name]={'columns':cols,'rows':n,'uncompressed_bytes':len(data),'uncompressed_sha256':sha(data),'compressed_sha256':sha(compressed)}
 js('results/SCALAR_TABLE_SCHEMA.json',{'encoding':'UTF8 csv.DictWriter default excel CRLF; None=empty; scalar representations unrounded','compression':'gzip.compress(level9,mtime0), decompression byteidentity asserted','runtime':{'python':platform.python_version(),'zlib':zlib.ZLIB_VERSION},'tables':tables,'case_alias_rule':'case_000001..case_005912 in sealed manifest/scorer order; original inputIDs and mapping excluded','scene_alias_rule':'scene_01..scene_05 from sorted(platform,private_scene), no original scene name or map exported','ordering':'case order then fixed arm/mode order; no positive/negative selection'})
 # Optional finite failed-run/independent-audit receipts: status and hashes only.
 for i,spec in enumerate(sources.get('failure_receipts',[]),1):
  p=local(spec['path']);assert file_sha(p)==spec['sha256'];v=read(p);js(f'history/FAILURE_{i:03d}.json',{'source_receipt_sha256':file_sha(p),**{k:v[k] for k in ('state','exception_type','phase','n_completed','program_sha256','no_target_GT_access','failure_classification') if k in v}},p,'scalar_failure_projection_no_raw_logs_messages_or_tracebacks')
 js('PUBLICATION_READINESS.json',ready,E/'PUBLICATION_READINESS.json');js('EXPORT_FINAL_SOURCE_HASHES.json',{k:{'source_sha256':sources[k]['sha256']} for k in REQUIRED},E/'EXPORT_FINAL_SOURCES.json','hash_only_no_private_source_paths')
 put('README.md',('# R8: Frozen full-caption RGB association and point-cloud pseudo regression\n\nSee REPORT.md for final measured outcomes, DECISION.json for adoption status, and REPRODUCTION.md for exact provenance and commands. This bundle contains the genuine Waymo-only epoch100 source to complete Drone/Quad5912 targetVAL study, with fixed40case engineering smoke; target data were previously research-exposed. All19arms and allnegative scalar cases are retained. The matched source is a fresh strictFP32 forward, never a reused R7sourcecache.\n\nG associates source proposals using frozen CLIP crop/full-caption cosines and aggregates correlated source-model centers, then updates tiny last-layer parameters. Compare source(q0), association-only(q1), same-information directG, realSGD1/3 and exactkernel-outputspace controls, plus PC-only and semanticshuffle. Globalstrength is heuristic, not calibrated trust. CLIP emits similarity and no3Dboxes; neither CLIP nor grouping proves referring-relation understanding or physical-instance identity.\n\nRoot actually viewed10fixed unlabelled images before the full run: the same visible object remained in multiple groups and some ROIs included neighboring vehicles/background or partial people. Dedup is imperfect; correlated consensus can reinforce the wrong object. These conditions remain in the evaluation and are not retuned away. ContinuousFOV/floorpixel preGPU fixes and historicalcode/fixtures are preserved.\n\nThis finite export excludes weights, media, pointclouds, rawcaptions, target annotations, originalsample/scene/frame IDs, coordinatearrays, embeddings, privateindices and signedURLs. Case tables use ordinalaliases and reproducible lossless .csv.gz. Original/exportSHA and any path-only portability projection are in EXPORT_PROVENANCE.json. Preparing this builder is not an experimental result or deployment decision.\n').encode())
 js('EXPORT_PROVENANCE.json',provenance)
 for name,data in pending.items():
  assert len(data)<95*1024**2,(name,'Unexpected large export')
  assert not any(s.encode() in data for s in raw_scenes) and not any(s.encode() in data for s in raw_ids),'Private identifier in export bytes'
 out.mkdir(parents=True)
 for name,data in pending.items():p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 manifest={'study':'3eed_crossplatform_rgbtext_pointcloud_pseudoreg_v2_20261030','scope':'finite_R8_code_protocol_resources_provenance_aggregate_and_allcase_scalars_no_private_data','files':[{'path':name,'bytes':len(data),'sha256':sha(data)} for name,data in sorted(pending.items())],'root_final_review_sha256':file_sha(bound['root_final_review']),'full_original_prediction_seal_sha256':file_sha(bound['full_prediction_seal']),'source_checkpoint_sha256':SOURCE_SHA,'CLIP_revision':CLIP_REVISION,'CLIP_safetensors_sha256':CLIP_SAFE_SHA};(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'n_files':len(pending)+1,'manifest_sha256':file_sha(out/'MANIFEST.json'),'output':str(out)}))
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=E/'public_export');main(ap.parse_args())
