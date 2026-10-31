"""Finite R13 export preparation. No inference/scoring/network/Git.
--preflight-only is ready; Actual final source/scorer/audit schemas are matched; publication requires every final gate.
"""
from pathlib import Path,PurePosixPath
from functools import lru_cache
import argparse,csv,gzip,hashlib,io,json,math
B=Path(__file__).resolve().parents[1];E=B/'research/round13/experiment';V=B/'research/round13/evaluation'
STUDY='3eed_full_description_association_vs_inequality_geometry_diagnostic_20261031'
ARMS=('source','A_full_consensus','A_full_order0','A_noun_consensus','A_deranged_consensus','B_inequality_full','B_inequality_noRGB','B_inequality_half','B_old_surface_equality')
ROLE_COUNTS={'new_content108':108,'regression8':8}
FINAL_SCHEMA_STATUS='actual_R13_composite_and_scored_v1_schema_matched_independent_audits_required'
REQUIRED=('new_source_seal','reused_source_seal','new_mask_seal','reused_mask_seal','association_preflight_seal','association_seal','geometry_seal','prediction_seal','scalar_summary','score_release','scorer_lock','input_audit','capture_audit','association_unlabelled_audit','geometry_unlabelled_audit','scored_audit','progression_gate','root_mechanism_review','root_report','root_decision','independent_final_report_review','root_final_review')
READINESS=('all116_A6calls_B_and_composite9arms_sealed_before_GT','actual_input_capture_A_B_and_scored_independent_audits_passed','new108_and_exposed_regression8_separate_no_pooled_confirmation','A_original_query_selection_B_fixedq0_positive_inequalities_and_controls_explicit','full_noun_deranged_two_numberings_UNKNOWN_and_mapping_failures_preserved','root_report_decision_progression_and_finalreview_actual_hash_bound','final_actual_producer_scorer_audit_schema_review_complete','privacy_no_rawIDs_scenes_captions_coordinates_masks_features_weights_panels_or_raw_answers')
CODE_FILES=tuple('crossplatform/round13/'+x for x in ('input_io.py','association.py','association_check_cpu.py','association_runner.py','support.py','geometry_observation.py','check_geometry_cpu.py','run_geometry.py','assemble_direct.py'))+tuple('research/round13/evaluation/'+x for x in ('audit_inputs.py','audit_capture108.py','audit_geometry116.py','audit_association_actual.py','audit_composite_actual.py','audit_scored_direct.py','audit_mechanism_gate.py','check_scorer_guard_independent.py','cache_readiness.py','check_B_independent_cpu.py','score_direct.py','check_score_direct_cpu.py'))+('research/round13/mechanism/reduce_unlabelled.py','research/round13/mechanism/check_reducer_cpu.py','research/round13/evaluation/history/preflight_scorer_v1_before_string_producer_guard/score_direct.py','research/round13/evaluation/history/preflight_scorer_v1_before_string_producer_guard/check_score_direct_cpu.py','research/round13/evaluation/history/auditor_preparation_v1_before_numpy_counter_cast/audit_association_actual.py','research/round13/evaluation/history/auditor_preparation_v1_before_numpy_counter_cast/audit_composite_actual.py','tools/history/build_round13_export_pre_final_schema.py','research/round13/prepare_inputs.py','research/round13/geometry/check_support_cpu.py','crossplatform/round12/capture_source.py','crossplatform/round12/capture_masks.py','crossplatform/round12/geometry_feedback.py','crossplatform/round12/multiview_feedback.py','crossplatform/round12/temporal_association.py','crossplatform/round12/sensor_pnp.py','crossplatform/round12/sensor_icp.py','crossplatform/round10/rgb_support.py','crossplatform/round9/run_diagnostic.py','crossplatform/round9/fixed_q0_geometry.py','crossplatform/round9/head_exact_fit.py','crossplatform/round8/association_geometry.py','crossplatform/round8/semantic_encoder.py','crossplatform/round7/camera_projection.py','crossplatform/round7/regression_adapter.py','crossplatform/round7/experiment/runner_helpers.py','crossplatform/round6/factory.py','research/round12/evaluation/metrics.py','research/round5/audit/independent_core_v2.py','tools/obb_iou.py','tools/build_round13_export.py')
STATIC_JSON=tuple('research/round13/association/'+x for x in ('ASSET_INVENTORY.json','IMPLEMENTATION_LOCK.json','ASSOCIATION_PLAN_TEMPLATE.json','MODEL_FILES_SHA256.json','CPU_FIXTURE.json','CPU_FIXTURE_NUMPY23.json'))+tuple('research/round13/geometry/'+x for x in ('IMPLEMENTATION_LOCK.json','SUPPORT_IMPLEMENTATION_LOCK.json','CPU_FIXTURE.json','CPU_FIXTURE_prelock_before_duplicate_ID_efficiency.json','SUPPORT_CPU_FIXTURE.json'))+tuple('research/round13/evaluation/'+x for x in ('A_IMPLEMENTATION_PREFLIGHT_REVIEW.json','B_IMPLEMENTATION_INDEPENDENT_REVIEW.json','B_INDEPENDENT_CPU_FIXTURE.json','B_WRAPPER_SUPPORT_PRERUN_REVIEW.json','CACHE_REUSE_READINESS.json','SCORER_CPU_FIXTURE.json','SCORER_CPU_FIXTURE_before_platform_scene_namespace.json','SCORER_GUARD_INDEPENDENT_CPU_FIXTURE.json','SCORER_IMPLEMENTATION_INDEPENDENT_REVIEW.json','ASSOCIATION_COMPOSITE_AUDITOR_PREPARATION_LOCK.json','COMPOSITE116_UNLABELLED_AUDIT.json','HISTORICAL_EXPOSURE_SCOPE_REVIEW.json','MECHANISM_GATE_INDEPENDENT_REVIEW.json','SCORER_PROGRAM_LOCK.json','SCORE_INTERFACE.json'))+tuple('research/round13/experiment/'+x for x in ('AUTHORIZATION.json','EXPERIMENT_CARD.md','SOURCE_CAPTURE_PLAN_LOCK.json','MASK_CAPTURE_PLAN_LOCK.json','ASSOCIATION_PLAN_LOCK.json','GEOMETRY_PLAN_LOCK.json','PROGRESSION_PREPLAN_LOCK.json','SOURCE_EXECUTION_RECEIPT.json','ROOT_PREFLIGHT_INPUT_REVIEW.json','HISTORICAL_EXPOSURE_CLARIFICATION.json','REGRESSION8_LEGACY_REPLAY_BRIDGE.json','ASSEMBLY_PLAN_LOCK.json','ASSOCIATION_ENGINEERING_RECEIPT.json','MECHANISM_READ_RELEASE.json','history/MASK_CAPTURE_PLAN_before_expected_count_fix.json'))+('research/round13/input/INPUT_PLAN_LOCK.json','research/round12/data_audit/PRIOR_PATH_DISJOINTNESS_ERRATUM.json','research/round13/mechanism/PREPARATION_LOCK.json','research/round13/mechanism/CPU_FIXTURE.json','research/round13/mechanism/actual116_attempt001/SUMMARY.json','research/round13/mechanism/actual116_attempt001/SEALED.json','research/round13/evaluation/history/preflight_scorer_v1_before_string_producer_guard/SCORER_PROGRAM_LOCK.json','research/round13/evaluation/history/preflight_scorer_v1_before_string_producer_guard/SCORER_CPU_FIXTURE.json','research/round13/evaluation/history/auditor_preparation_v1_before_numpy_counter_cast/ASSOCIATION_COMPOSITE_AUDITOR_PREPARATION_LOCK.json')
# Narrative files use text-only path projection, never JSON parsing.
STATIC_JSON=tuple(x for x in STATIC_JSON if not x.endswith('.md'))
OPTIONAL_JSON=('research/round13/evaluation/FINAL_REPORT_INDEPENDENT_REVIEW.json','research/round13/experiment/ROOT_ASSOCIATION_GALLERY_REVIEW.json','research/round13/experiment/ROOT_GEOMETRY_GALLERY_REVIEW.json','research/round13/experiment/ROOT_GALLERY_REVIEW.json','research/round13/qualitative/GALLERY_PLAN_LOCK.json','research/round13/qualitative/GALLERY_RENDER_RELEASE.json')
OPTIONAL_CODE=('research/round13/evaluation/audit_final_report.py','research/round13/evaluation/review_final_report.py')
DOC_FILES=('research/round13/association/INTERFACE_AND_MINIMAL_PLAN.md','research/round13/geometry/INTERFACE.md','research/round13/geometry/PROPOSAL.md','research/round13/geometry/SUPPORT_INTERFACE.md','research/round13/evaluation/PROTOCOL_RECOMMENDATION.md','research/round13/evaluation/SCORER_INTERFACE.md','research/round13/mechanism/INTERFACE.md','research/round13/experiment/EXPERIMENT_CARD.md','research/round13/experiment/REPRODUCTION.md')
CODE_FILES=CODE_FILES+tuple('research/round13/association_diagnostic/'+x for x in ('diagnose_association.py','check_diagnostic_cpu.py','inspect_schema.py','inspect_nested_schema.py','inspect_count_alignment.py'))+('research/round13/qualitative/render_gallery.py','research/round13/evaluation/check_spatial_diagnostic_independent.py','tools/history/build_round13_export_pre_association_diagnostic.py')
STATIC_JSON=STATIC_JSON+tuple('research/round13/association_diagnostic/'+x for x in ('PLAN_V2_LOCK.json','COMPUTE_RELEASE.json','DIAGNOSTIC_SCHEMA_RELEASE.json','DIAGNOSTIC_NESTED_SCHEMA_RELEASE.json','DIAGNOSTIC_COUNT_SCHEMA_RELEASE.json','PLAN_STATE.json','NESTED_SCHEMA_CLI_REJECTION.json','DIAGNOSTIC_CPU_FIXTURE.json','DIAGNOSTIC_CPU_FIXTURE_before_primary_bridge_guard.json','schema_attempt001/SCHEMA_KEYS_SUMMARY.json','schema_attempt001/SEALED.json','nested_schema_attempt001/NESTED_SCHEMA_SUMMARY.json','nested_schema_attempt001/SEALED.json','count_schema_attempt001/SUMMARY.json','count_schema_attempt001/SEALED.json','actual116_attempt001/DIAGNOSTIC_LOCK.json','actual116_attempt001/SUMMARY.json','actual116_attempt001/SEALED.json'))+('research/round13/qualitative/PLAN_LOCK.json','research/round13/qualitative/gallery8_attempt001/SEALED.json','research/round13/experiment/ROOT_GALLERY_REVIEW.json','research/round13/experiment/ROOT_POSTSCORE_DIAGNOSTIC_REVIEW.json','research/round13/experiment/ACTUAL_EXECUTION_RECEIPT.json')
OPTIONAL_JSON=tuple(r for r in OPTIONAL_JSON if r!='research/round13/experiment/ROOT_GALLERY_REVIEW.json')+('research/round13/evaluation/SPATIAL_DIAGNOSTIC_INDEPENDENT_REVIEW.json','research/round13/evaluation/SPATIAL_ADAPTER_INDEPENDENT_CPU_FIXTURE.json','research/round13/evaluation/REPORT_BODY_PRELIMINARY_REVIEW.json')
OPTIONAL_CODE=OPTIONAL_CODE+('research/round13/evaluation/audit_spatial_diagnostic.py',)
DOC_FILES=DOC_FILES+('research/round13/association_diagnostic/POSTSCORE_ASSOCIATION_DIAGNOSTIC_PLAN.md','research/round13/association_diagnostic/POSTSCORE_ASSOCIATION_DIAGNOSTIC_PLAN_V2.md')

FORBIDDEN={'id','ids','case_id','scene','frame','caption','captions','shuffled_caption','utterance','ground_info','bbox_3d','gt_bbox','positive_map','GT7D_model','target7','target6','source6','box6','box7','center3','size3','source_boxes_raw6','source_boxes_eval6','arm_boxes_raw6','arm_boxes_eval6','points','raw_points','sampled_points_model','raw_indices','center_features','size_features','image_intrinsic','image_extrinsic','camera_distortion','embeddings','proj_queries','proj_tokens','roi_xyxy','bbox_xyxy','boxes_xyxy','mask2d','masks_bool','uv','depth','pose','K','E','D','T','T_raw_B_to_A','transformation','rotation_matrix','translation_vector','coordinate_residual6','target6_modelcoords','source_box6','box_corners','q0','q_source','q_selected','source_q0','query_index','query_indices','arm_query_index','representative_query_index','query_group_indices','selected_group_id','group_id','membership256','mask_index','selected_mask_index','selected_candidate_id','candidate_id','raw_answer','raw_answer_PRIVATE','raw','answer','answers','generated_ids','input_ids','actual_model_text','prompt','messages','serialized_prompt'}
DROP_MAPS={'files','source_cache_files','source_input_paths','weights_relative_private_location','workspace_root','programroot','rows','cases','candidates','pairs','selected','selected_cases','selected_rows','anchor_ids','frame_aliases','examples','case_alias_notes','notes_by_alias','candidate_id_to_mask_index','candidate_panels','mapping','calls','trace','lineage_private'}
AXES=('cx','cy','cz','sx','sy','sz');sha=lambda x:hashlib.sha256(x).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8<<20),b''):h.update(z)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def local(rel):
 q=PurePosixPath(rel);assert q.parts and not q.is_absolute() and '..' not in q.parts;p=B/q.as_posix();assert p.resolve().is_relative_to(B.resolve()) and p.is_file(),rel;return p

def source_text(p):
 return p.read_text().replace(str(B),'__WORKSPACE_ROOT__').replace(str(B.parent/'mllm'),'__MLLM_ROOT__').replace('crossplatform'+'/'+'dataset_full'+'/'+'3eed_dataset','__DATASET_ROOT__')

def is_helper_mask_index_key(key):
 k=str(key).lower();return k in ('mask_index','mask_indices') or k.endswith(('_mask_index','_mask_indices'))

def privacy_context():
 corpus=read(B/'research/round8/experiment/full5912_gpu_inputs.json');ids={r['id'] for r in corpus};scenes={r['scene'] for r in corpus};captions={r['caption'] for r in corpus};caption_hashes={sha(x.encode()) for x in captions};longcaptions=tuple(x for x in captions if len(x)>=24)
 @lru_cache(maxsize=32768)
 def textcheck(s):
  assert ('/'+'home'+'/') not in s and ('/'+'dataset_full'+'/') not in s and (chr(92)+'Users'+chr(92)) not in s and ('https:'+'//'+'1drv'+'.ms/') not in s
  assert not any(x in s for x in ids|scenes) and sha(s.encode()) not in caption_hashes and not any(x in s for x in longcaptions),'Private input/description string';return s
 def clean(v):
  if isinstance(v,dict):assert not set(v)&FORBIDDEN and not any(k.endswith('_PRIVATE') or is_helper_mask_index_key(k) for k in v);return {textcheck(str(k)):clean(x) for k,x in v.items()}
  if isinstance(v,list):return [clean(x) for x in v]
  if isinstance(v,str):return textcheck(v)
  if isinstance(v,float):assert math.isfinite(v)
  assert v is None or isinstance(v,(int,float,bool));return v
 def project(v):
  if isinstance(v,dict):
   out={}
   for k,x in v.items():
    if k in FORBIDDEN|DROP_MAPS or is_helper_mask_index_key(k) or k.endswith('_PRIVATE') or k.endswith(('_path','_dir')) or k in ids|scenes or any(q in k.lower() for q in ('pointpath','imagepath','metapath')) or '.npz' in k:continue
    if k in ('container_key_presence_counts','ground_info_key_presence_counts','field_type_counts','path_key_type_counts','bbox_shape_counts','scalar_list_shape_counts') and isinstance(x,dict):out[k+'_schema_records']=[{'schema_field':name,'counts':project(z)} for name,z in x.items()]
    elif k in ('changed_axis6_counts','actual_changed_axis_counts','changed_axis6','accepted_axis6','abs_error6_m_mean','equal_scene_abs_error6_m_mean','abs6_error_reduction_m') and isinstance(x,list) and len(x)==6:out[k+'_scalar_components']={a:project(z) for a,z in zip(AXES,x)}
    elif k in ('image_mean','image_std') and isinstance(x,list) and len(x)==3:out[k+'_RGB_components']={a:z for a,z in zip(('red','green','blue'),x)}
    else:
     public_key=('local_path_key_'+str(len(out)+1).zfill(4)) if k.startswith('/') else k
     out[public_key]=project(x)
   return out
  if isinstance(v,list):
   if v and all(isinstance(x,(list,tuple)) for x in v):return {'matrix_or_nested_observation_array_omitted':True,'rows':len(v)}
   if len(v) in (2,3,4,6,7,9,16,256) and all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in v):return {'numeric_vector_omitted_for_coordinate_or_candidate_privacy':True,'length':len(v)}
   return [project(x) for x in v]
  if isinstance(v,str):
   if v.startswith('/') or ('/'+'home'+'/') in v:return '[local_path_omitted]'
   return v.replace(str(B),'__WORKSPACE_ROOT__').replace(str(B.parent/'mllm'),'__MLLM_ROOT__')
  return v
 return textcheck,clean,project

def preflight():
 textcheck,clean,project=privacy_context();checked=[]
 for rel in CODE_FILES+tuple(r for r in OPTIONAL_CODE if (B/r).is_file())+DOC_FILES:textcheck(source_text(local(rel)));checked.append(rel)
 for rel in STATIC_JSON+tuple(r for r in OPTIONAL_JSON if (B/r).is_file()):clean(project(read(local(rel))));checked.append(rel)
 return {'state':'R13_finite_code_doc_protocol_privacy_preflight_pass_actual_schema_matched','code_n':len(CODE_FILES)+sum((B/r).is_file() for r in OPTIONAL_CODE),'doc_n':len(DOC_FILES),'static_json_n':len(STATIC_JSON),'optional_present_n':sum((B/r).is_file() for r in OPTIONAL_JSON),'checked_sources':{r:file_sha(local(r)) for r in checked},'builder_sha256':file_sha(Path(__file__)),'final_schema_status':FINAL_SCHEMA_STATUS,'no_target_GT_private_scoredrows_pixels_model_GPU_actual_build_or_upload':True}

# This scalar projector is tied to the actual current score_direct.evaluate_row
# source schema; authorized actual sealed scalar rows are checked without new GT.
# Every build additionally requires actual producer/scorer/audit final bindings.
def arm_scalar_table(rows,inputs,case_roles,case_aliases=None):
 textcheck,clean,_=privacy_context();assert len(rows)==116 and len(inputs)==116
 im={x['id']:x for x in inputs};aliases=case_aliases or {x['id']:f'case_{i+1:06d}' for i,x in enumerate(inputs)};scenes={s:f'scene_{i+1:02d}' for i,s in enumerate(sorted({(x['platform'],x['scene']) for x in inputs}))}
 fields=('case_alias','cohort','platform','scene_alias','arm','role','iou','Acc25','Acc50','delta_iou_vs_source','query_changed','actual_changed','changed_axis_bitmask','center_L2_error_m','size_L2_error_m','center_shift_L2_m','size_shift_L2_m','raw_nonpositive_sizes','status');buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=fields);w.writeheader();count=0;roles={}
 for r in rows:
  sid=r['id'];item=im[sid];cohort=r['case_role'];assert cohort==case_roles[sid] and cohort in ROLE_COUNTS and r['platform']==item['platform'] and r['scene']==item['scene'] and tuple(r['arms'])==ARMS;roles[cohort]=roles.get(cohort,0)+1
  for arm,x in r['arms'].items():
   role='frozen_source' if arm=='source' else 'A_frozen_VLM_original_query_selection' if arm.startswith('A_') else 'B_fixedq0_inequality_direct' if arm in ARMS[5:7] else 'B_fixedq0_half_interpolation_control' if arm==ARMS[7] else 'B_fixedq0_legacy_surface_equality_control'
   v={'case_alias':aliases[sid],'cohort':cohort,'platform':r['platform'],'scene_alias':scenes[(r['platform'],r['scene'])],'arm':arm,'role':role,'iou':x['iou'],'Acc25':x['Acc25'],'Acc50':x['Acc50'],'delta_iou_vs_source':x['iou']-r['arms']['source']['iou'],'query_changed':x['query_changed'],'actual_changed':x['actual_changed'],'changed_axis_bitmask':sum(int(z)<<j for j,z in enumerate(x['actual_changed_axes'])),'center_L2_error_m':x['coordinate_errors']['center_L2_m'],'size_L2_error_m':x['coordinate_errors']['size_L2_m'],'center_shift_L2_m':x['center_shift_L2_m'],'size_shift_L2_m':x['size_shift_L2_m'],'raw_nonpositive_sizes':x['raw_nonpositive_sizes'],'status':x['status']};assert v['Acc25']==(v['iou']>.25) and v['Acc50']==(v['iou']>.5);clean(v);w.writerow(v);count+=1
 assert roles==ROLE_COUNTS and count==1044
 return gzip.compress(buf.getvalue().encode(),mtime=0),{'fields':list(fields),'rows':1044,'unique_cases':116,'cohort_counts':ROLE_COUNTS,'nine_related_armoutputs_per_case_not_independent_samples':True,'gzip_mtime':0,'no_query_indices_rawIDs_or_coordinates':True,'new_content_and_regression_not_pooled_confirmation':True,'deranged_scope':'Inputintervention diagnostic; original-target score is original-label retention, not correct counterfactual-query accuracy.'}

def verify_seal(path):
 seal=read(path)
 for rel,h in seal['files'].items():
  q=PurePosixPath(rel);assert not q.is_absolute() and '..' not in q.parts
  assert file_sha(path.parent/q.as_posix())==h,(str(path),rel)
 return seal

def check_bindings(mapping):
 assert isinstance(mapping,dict) and mapping
 for name,h in mapping.items():
  path=Path(name);path=path if path.is_absolute() else B/path;assert path.is_file() and file_sha(path)==h

def main(a):
 assert not a.output.exists() and a.output.resolve().is_relative_to(B.resolve()),'New bounded output required'
 ready=read(E/'PUBLICATION_READINESS.json')
 for k in READINESS:assert ready.get(k) is True,('Missing actual final gate',k)
 specs=read(E/'EXPORT_FINAL_SOURCES.json');bound={}
 for k in REQUIRED:
  spec=specs[k];assert set(spec)=={'path','sha256'};p=local(spec['path']);assert file_sha(p)==spec['sha256'];bound[k]=p
 checkpoint='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd';maskweights='73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e'
 source={k:verify_seal(bound[k]) for k in ('new_source_seal','reused_source_seal')};masks={k:verify_seal(bound[k]) for k in ('new_mask_seal','reused_mask_seal')}
 for k,n in (('new_source_seal',108),('reused_source_seal',146)):
  s=source[k];assert s['state']=='sealed_R12_source_forward_and_input_observations_no_G_no_GT' and s['n_samples']==n and s['source_checkpoint_sha256']==checkpoint and s['source_unchanged'] and s['source_repeat_bitexact']
 for k,n in (('new_mask_seal',108),('reused_mask_seal',501)):
  s=masks[k];assert s['state']=='sealed_R12_independent_whole_image_instance_masks_no_q0_caption_or_GT' and s['frame_n']==n and s['weights_sha256']==maskweights and not s['trained_or_updated'] and s['mask_returned_original_image_pixels']
 ap=verify_seal(bound['association_preflight_seal']);aa=verify_seal(bound['association_seal']);gg=verify_seal(bound['geometry_seal']);pred=verify_seal(bound['prediction_seal'])
 assert ap['state']=='sealed_R13_frozen_association_first2_engineering_no_GT' and ap['n_samples']==2 and ap['n_model_calls']==12
 assert aa['state']=='sealed_R13_all116_frozen_association6calls_before_GT' and aa['n_samples']==116 and aa['n_model_calls']==696 and aa['model_frozen_parameters_buffers_unchanged'] and aa['only_original_source_query_boxes_selected'] and aa['GT_or_optimizer_or_neural_update_executed'] is False and aa['formal_preflight_policy']=='rerun_all116_in_separate_output'
 assert gg['state']=='sealed_R13_fixedq0_inequality_vs_source_and_surface_direct_no_GT' and gg['n']==116 and gg['new_n']==108 and gg['regression_n']==8 and gg['all_q_fixed'] and gg['no_neural_update_or_GPU_or_GT']
 assert pred['state']=='sealed_R13_all116_9_direct_arms_before_GT' and pred['n_samples']==116 and pred['n_arms']==9 and tuple(pred['arm_names'])==ARMS and pred['case_role_counts']==ROLE_COUNTS and pred['source_checkpoint_sha256']==checkpoint and pred['all_A_original_candidate_boxes'] and pred['all_B_fixed_source_q0'] and pred['all_direct_no_neural_update']
 assert pred['association_seal_sha256']==file_sha(bound['association_seal']) and pred['geometry_seal_sha256']==file_sha(bound['geometry_seal'])
 for s in (aa,gg,pred):check_bindings(s['program_sha256'])
 summary=read(bound['scalar_summary']);scoresealpath=bound['scalar_summary'].parent/'SEALED.json';ss=verify_seal(scoresealpath);resultpath=bound['scalar_summary'].parent/'RESULTS_PRIVATE.json'
 assert summary['state']=='R13_nine_direct_arms_scored_only_after_complete_seal' and summary['n']==116 and summary['case_role_counts']==ROLE_COUNTS and summary['source_checkpoint_sha256']==checkpoint
 assert ss['state']=='sealed_R13_postrelease_CPU_scoring' and ss['n_samples']==116 and ss['prediction_seal_sha256']==file_sha(bound['prediction_seal']) and ss['scorer_sha256']==file_sha(V/'score_direct.py')
 release=read(bound['score_release']);assert release['CPU_SCORE_RELEASE'] and release['all_A_B_predictions_and_helper_outputs_sealed_before_GT'] and release['stage']=='R13_association_vs_inequality_geometry_direct' and release['prediction_seal_sha256']==file_sha(bound['prediction_seal']) and release['scorer_sha256']==file_sha(V/'score_direct.py') and release['source_checkpoint_sha256']==checkpoint and release['expected_case_role_counts']==ROLE_COUNTS and ss['release_sha256']==file_sha(bound['score_release'])
 assert set(release['required_seal_roles'])=={x['role'] for x in release['sealed_artifacts']}
 for artifact in release['sealed_artifacts']:
  d=Path(artifact['directory']);d=d if d.is_absolute() else B/d;assert file_sha(d/'SEALED.json')==artifact['seal_sha256'];s=verify_seal(d/'SEALED.json')
  program=s.get('program_sha256')
  if isinstance(program,dict):check_bindings(program)
  elif isinstance(program,str):assert program in set(release['program_bindings'].values()),'String producer must have an explicit hashed code binding'
 for k in ('program_bindings','input_bindings','prompt_bindings','geometry_dependency_sha256'):check_bindings(release[k])
 sl=read(bound['scorer_lock']);assert sl['state']=='all_composite_source_helper_prompt_program_hashes_and_unlabelled_contracts_passed_before_GT' and sl['n_samples']==116 and sl['prediction_seal_sha256']==file_sha(bound['prediction_seal']) and sl['release_sha256']==file_sha(bound['score_release']) and sl['source_checkpoint_sha256']==checkpoint
 ia=read(bound['input_audit']);assert ia['state']=='PASS_R13_INPUT_SELECTION_AND_REUSED_CAPTURE_INTERFACE' and not ia['issues'] and ia['new_n']==108 and ia['regression_n']==8 and ia['auditor_sha256']==file_sha(V/'audit_inputs.py')
 ca=read(bound['capture_audit']);assert ca['state']=='PASS_R13_ACTUAL_NEW108_SOURCE_MASK_AND_OLD8_REUSE_AUDIT' and not ca['issues'] and ca['source_seal_sha256']==file_sha(bound['new_source_seal']) and ca['mask_seal_sha256']==file_sha(bound['new_mask_seal']) and ca['auditor_sha256']==file_sha(V/'audit_capture108.py')
 au=read(bound['association_unlabelled_audit']);assert au['state']=='PASS_R13_FORMAL116_ASSOCIATION_UNLABELLED_AUDIT' and au['n_cases']==116 and au['n_model_calls']==696 and not au['issues'] and au['association_seal_sha256']==file_sha(bound['association_seal']) and au['auditor_sha256']==file_sha(V/'audit_association_actual.py')
 gu=read(bound['geometry_unlabelled_audit']);assert gu['state']=='PASS_R13_ACTUAL116_FIXEDQ0_GEOMETRY_UNLABELLED_AUDIT' and gu['n']==116 and not gu['issues'] and gu['prediction_seal_sha256']==file_sha(bound['geometry_seal']) and gu['auditor_sha256']==file_sha(V/'audit_geometry116.py')
 cu=read(V/'COMPOSITE116_UNLABELLED_AUDIT.json');assert cu['state']=='PASS_R13_ACTUAL116_NINEARM_COMPOSITE_UNLABELLED_AUDIT' and cu['n']==116 and cu['n_arms']==9 and not cu['issues'] and cu['composite_seal_sha256']==file_sha(bound['prediction_seal']) and cu['association_audit_sha256']==file_sha(bound['association_unlabelled_audit']) and cu['geometry_audit_sha256']==file_sha(bound['geometry_unlabelled_audit']) and cu['auditor_sha256']==file_sha(V/'audit_composite_actual.py')
 sa=read(bound['scored_audit']);assert sa['state']=='PASS_R13_INDEPENDENT_SEALED_SCALAR_REDUCTION_AUDIT' and sa['n']==116 and not sa['failures'] and sa['results_sha256']==file_sha(resultpath) and sa['scalar_summary_sha256']==file_sha(bound['scalar_summary']) and sa['scored_seal_sha256']==file_sha(scoresealpath) and sa['prediction_seal_sha256']==file_sha(bound['prediction_seal']) and sa['release_sha256']==file_sha(bound['score_release']) and sa['auditor_sha256']==file_sha(V/'audit_scored_direct.py')
 md=B/'research/round13/mechanism/actual116_attempt001';ms=verify_seal(md/'SEALED.json');mu=read(md/'SUMMARY.json');assert ms['state']=='sealed_R13_noGT_mechanism_reduction' and ms['n_cases']==116 and ms['association_seal_sha256']==file_sha(bound['association_seal']) and ms['geometry_seal_sha256']==file_sha(bound['geometry_seal']) and ms['engineering_association_seal_sha256']==file_sha(bound['association_preflight_seal']) and ms['reducer_sha256']==file_sha(B/'research/round13/mechanism/reduce_unlabelled.py')
 assert mu['state']=='R13_sealed_output_noGT_mechanism_reduction' and mu['n_cases']==116 and mu['case_role_counts']==ROLE_COUNTS and mu['formal_actual_calls']==696 and mu['engineering_actual_calls']==12
 mg=read(V/'MECHANISM_GATE_INDEPENDENT_REVIEW.json');assert mg['state']=='PASS_R13_INDEPENDENT_MECHANISM_AND_PRELOCKED_GATE_REVIEW' and not mg['failures'] and mg['mechanism_seal_sha256']==file_sha(md/'SEALED.json') and mg['mechanism_summary_sha256']==file_sha(md/'SUMMARY.json') and mg['association_seal_sha256']==file_sha(bound['association_seal']) and mg['geometry_seal_sha256']==file_sha(bound['geometry_seal']) and mg['scored_seal_sha256']==file_sha(scoresealpath) and mg['scoring_audit_sha256']==file_sha(bound['scored_audit']) and mg['auditor_sha256']==file_sha(V/'audit_mechanism_gate.py')
 exposure=read(E/'HISTORICAL_EXPOSURE_CLARIFICATION.json');er=read(V/'HISTORICAL_EXPOSURE_SCOPE_REVIEW.json');assert exposure['earlier_R7_R8_full_target_query_n']==5912 and exposure['all_current_targetVAL_cases_have_historical_model_and_taskscore_exposure'] is True and er['root_clarification_sha256']==file_sha(E/'HISTORICAL_EXPOSURE_CLARIFICATION.json')
 bridge=read(E/'REGRESSION8_LEGACY_REPLAY_BRIDGE.json');assert bridge['state']=='PASS' and bridge['n']==8 and bridge['new_geometry_seal_sha256']==file_sha(bound['geometry_seal']) and bridge['old_R12_prediction_seal_sha256']=='d3cdf0fff9cef27afbf4030e7ac836e5dd3bfcc5a83a9ee38136252eb26cbbb5' and bridge['no_GT_or_reoptimization']
 dd=B/'research/round13/association_diagnostic';ds=verify_seal(dd/'actual116_attempt001/SEALED.json');du=read(dd/'actual116_attempt001/SUMMARY.json');dl=read(dd/'actual116_attempt001/DIAGNOSTIC_LOCK.json');dr=read(dd/'COMPUTE_RELEASE.json');dp=read(dd/'PLAN_V2_LOCK.json')
 assert ds['state']=='sealed_R13_postscore_spatial_diagnostic' and ds['n']==116 and ds['program_sha256']==file_sha(dd/'diagnose_association.py') and ds['release_sha256']==file_sha(dd/'COMPUTE_RELEASE.json') and du['state']=='R13_postscore_descriptive_spatial_association_completed' and du['n']==116
 assert dr['POSTSCORE_DIAGNOSTIC_RELEASE'] and dr['all_task_predictions_and_main_scores_sealed'] and dr['method_or_primary_metric_modification_permitted'] is False and dr['program_sha256']==file_sha(dd/'diagnose_association.py') and dr['plan_v1_sha256']==file_sha(dd/'POSTSCORE_ASSOCIATION_DIAGNOSTIC_PLAN.md') and dr['plan_v2_sha256']==file_sha(dd/'PLAN_V2_LOCK.json')
 assert dl['release_sha256']==file_sha(dd/'COMPUTE_RELEASE.json') and dl['plan_v1_sha256']==dr['plan_v1_sha256'] and dl['plan_v2_sha256']==dr['plan_v2_sha256'] and dl['n_cases']==116
 check_bindings(dp['sha256']);check_bindings(dp['geometry_dependency_sha256']);check_bindings(dr['program_bindings']);check_bindings(dr['geometry_dependency_sha256'])
 for artifact in dr['sealed_artifacts']:
  d=Path(artifact['directory']);d=d if d.is_absolute() else B/d;assert file_sha(d/'SEALED.json')==artifact['seal_sha256'];verify_seal(d/'SEALED.json')
 assert file_sha(bound['prediction_seal']) in {x['seal_sha256'] for x in dr['sealed_artifacts']} and file_sha(scoresealpath) in {x['seal_sha256'] for x in dr['sealed_artifacts']}
 postroot=read(E/'ROOT_POSTSCORE_DIAGNOSTIC_REVIEW.json');assert postroot['plan_v2_sha256']==file_sha(dd/'PLAN_V2_LOCK.json') and postroot['diagnostic_seal_sha256']==file_sha(dd/'actual116_attempt001/SEALED.json') and postroot['diagnostic_summary_sha256']==file_sha(dd/'actual116_attempt001/SUMMARY.json') and postroot['compute_release_sha256']==file_sha(dd/'COMPUTE_RELEASE.json') and postroot['primary_prediction_seal_unchanged'] is True and postroot['primary_scored_seal_unchanged'] is True
 diagnostic_review=V/'SPATIAL_DIAGNOSTIC_INDEPENDENT_REVIEW.json'
 assert diagnostic_review.is_file(),'Actual completed postscore diagnostic independent review required'
 di=read(diagnostic_review);assert di['state']=='PASS_R13_POSTSCORE_SPATIAL_DIAGNOSTIC_INDEPENDENT_REVIEW'
 for key,path in (('diagnostic_seal_sha256',dd/'actual116_attempt001/SEALED.json'),('diagnostic_summary_sha256',dd/'actual116_attempt001/SUMMARY.json'),('program_sha256',dd/'diagnose_association.py'),('plan_v2_sha256',dd/'PLAN_V2_LOCK.json'),('diagnostic_release_sha256',dd/'COMPUTE_RELEASE.json'),('diagnostic_cases_sha256',dd/'actual116_attempt001/CASES_PRIVATE.json'),('primary_results_sha256',resultpath),('auditor_sha256',V/'audit_spatial_diagnostic.py'),('synthetic_adapter_fixture_sha256',V/'SPATIAL_ADAPTER_INDEPENDENT_CPU_FIXTURE.json')):assert di[key]==file_sha(path)
 gal=B/'research/round13/qualitative';gs=verify_seal(gal/'gallery8_attempt001/SEALED.json');gr=read(E/'ROOT_GALLERY_REVIEW.json');assert gs['n_images']==8 and gs['plan_sha256']==file_sha(gal/'PLAN_LOCK.json') and gr['gallery_seal_sha256']==file_sha(gal/'gallery8_attempt001/SEALED.json') and gr['gallery_plan_sha256']==file_sha(gal/'PLAN_LOCK.json');assert gs['program_sha256']==file_sha(gal/'render_gallery.py')
 rr=read(bound['root_final_review']);rm=read(bound['root_mechanism_review']);ir=read(bound['independent_final_report_review']);assert rr['publication_ready'] and rr['report_sha256']==file_sha(bound['root_report']) and rr['decision_sha256']==file_sha(bound['root_decision']) and rr['mechanism_seal_sha256']==file_sha(md/'SEALED.json') and rm['mechanism_seal_sha256']==file_sha(md/'SEALED.json')
 assert ir['state']=='PASS_R13_FINAL_REPORT_INDEPENDENT_REVIEW' and ir['report_sha256']==file_sha(bound['root_report']) and ir['decision_sha256']==file_sha(bound['root_decision'])
 assert bound['progression_gate'].is_file();gate=read(bound['progression_gate']);assert isinstance(gate,dict) and gate
 for key,path in (('progression_gate_sha256',bound['progression_gate']),('root_postscore_diagnostic_review_sha256',E/'ROOT_POSTSCORE_DIAGNOSTIC_REVIEW.json'),('spatial_diagnostic_review_sha256',diagnostic_review)):
  assert ir[key]==file_sha(path)
 if 'auditor_sha256' in ir:assert ir['auditor_sha256']==file_sha(V/'audit_final_report.py')
 inputs=read(B/'research/round13/input/ALL116_GPU_INPUTS.json');cases=read(B/'research/round13/input/CASE116_PRIVATE.json');assert len(inputs)==len(cases)==116 and file_sha(B/'research/round13/input/ALL116_GPU_INPUTS.json')==pred['manifest_sha256']==release['manifest_sha256'] and file_sha(B/'research/round13/input/CASE116_PRIVATE.json')==pred['cases_sha256']==release['cases_sha256']
 rolemap={c['id']:'new_content108' if c['role']=='new_content' else 'regression8' for c in cases};aliasmap={c['id']:c['case_alias'] for c in cases};assert len(set(aliasmap.values()))==116
 actual=read(resultpath);assert {k:v for k,v in actual.items() if not k.endswith('_PRIVATE')}==summary;rows=actual['rows_PRIVATE'];assert len(rows)==len({r['id'] for r in rows})==116
 scalar_gz,scalar_schema=arm_scalar_table(rows,inputs,rolemap,aliasmap)
 for t in summary['summary']:
  selected=[r for r in rows if r['case_role']==t['case_role'] and (t['platform']=='all_platforms' or r['platform']==t['platform'])];values=[r['arms'][t['arm']]['iou'] for r in selected];assert len(values)==t['n'] and abs(math.fsum(values)/len(values)-t['mean_iou'])<1e-14
  clusters={(r['platform'],r['scene']) for r in selected};means=[math.fsum(r['arms'][t['arm']]['iou'] for r in selected if (r['platform'],r['scene'])==s)/sum((r['platform'],r['scene'])==s for r in selected) for s in clusters];assert abs(math.fsum(means)/len(means)-t['equal_scene_mean_iou'])<1e-14
 textcheck,clean,project=privacy_context();pending={};provenance=[]
 def emit(name,data):assert name not in pending;pending[name]=data
 def js(name,obj,origin=None,mode='explicit_scalar_and_private_payload_field_projection'):
  data=(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n').encode();emit(name,data)
  if origin:provenance.append({'source_relative':str(origin.relative_to(B)),'source_sha256':file_sha(origin),'export_relative':name,'export_sha256':sha(data),'mode':mode})
 actualcode=CODE_FILES+tuple(r for r in OPTIONAL_CODE if (B/r).is_file())
 for group,prefix in ((actualcode,'code/'),(DOC_FILES,'docs/')):
  for rel in group:
   p=local(rel);data=textcheck(source_text(p)).encode();emit(prefix+rel,data);provenance.append({'source_relative':rel,'source_sha256':file_sha(p),'export_relative':prefix+rel,'export_sha256':sha(data),'mode':'workspace_mllm_dataset_path_projection_only'})
 for rel in dict.fromkeys(STATIC_JSON+tuple(r for r in OPTIONAL_JSON if (B/r).is_file())):js('protocol/'+rel,project(read(local(rel))),local(rel))
 for key,p in bound.items():
  if p.suffix=='.md':
   data=textcheck(source_text(p)).encode();emit('final/'+key+p.suffix,data);provenance.append({'source_relative':str(p.relative_to(B)),'source_sha256':file_sha(p),'export_relative':'final/'+key+p.suffix,'export_sha256':sha(data),'mode':'narrative_path_projection_only'})
  else:js('final/'+key+'.json',project(read(p)),p)
 emit('results/ARM_SCALARS.csv.gz',scalar_gz);js('results/ARM_SCALARS_SCHEMA.json',scalar_schema)
 js('EXPORT_PROVENANCE.json',{'sources':provenance,'final_sources':{k:{'source_relative':str(p.relative_to(B)),'source_sha256':file_sha(p)} for k,p in bound.items()},'source_checkpoint_sha256':checkpoint,'mask_weights_sha256':maskweights,'actual_mechanism_seal_sha256':file_sha(md/'SEALED.json'),'scope':'No ever-unseen/pristine confirmation claim. All5912 historically model/taskGT exposed; only recent exact-content exclusions. A original-query selection and B fixedq0 direct geometry, no neuralTTA.'})
 emit('README.md',(STUDY+'\nSee actual final report/decision and reproduction notes.108recent-content-disjoint target queries plus8exposed outputconditioned regression cases,9directarms,1044relatedoutputs. All5912targetVAL already model/taskGT exposed in R7/R8. CandidateID stability is not semantic truth. Partial foreground is not complete object extent. No original IDs/scenes/captions/answers/coordinates/masks/features/weights/panels are included.\n').encode())
 corpus=read(B/'research/round8/experiment/full5912_gpu_inputs.json');privateidentifiers={r['id'] for r in corpus}|{r['scene'] for r in corpus}
 for name,data in pending.items():assert not any(x.encode() in data for x in privateidentifiers),(name,'Private identifier');assert not name.endswith(('.pt','.pth','.bin','.safetensors','.npz','.npy','.jpg','.jpeg','.png','.mp4','.wav'))
 # Output is created only after all actual binding, scalar and privacy checks.
 a.output.mkdir(parents=True)
 for name,data in pending.items():p=a.output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 manifest={'study':STUDY,'unique_cases':116,'arm_rows':1044,'cohort_counts':ROLE_COUNTS,'files':[{'path':name,'bytes':len(data),'sha256':sha(data)} for name,data in sorted(pending.items())],'source_checkpoint_sha256':checkpoint,'root_final_review_sha256':file_sha(bound['root_final_review']),'historical_exposure_clarification_sha256':file_sha(E/'HISTORICAL_EXPOSURE_CLARIFICATION.json')};(a.output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'n_files':len(pending)+1,'manifest_sha256':file_sha(a.output/'MANIFEST.json')}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=E/'public_export');p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 if a.preflight_only:print(json.dumps(preflight(),indent=2))
 else:main(a)
