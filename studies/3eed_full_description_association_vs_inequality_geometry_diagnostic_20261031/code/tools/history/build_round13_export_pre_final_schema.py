"""Finite R13 export preparation. No inference/scoring/network/Git.
--preflight-only is ready; actual final-schema closure remains explicitly pending.
"""
from pathlib import Path,PurePosixPath
from functools import lru_cache
import argparse,csv,gzip,hashlib,io,json,math
B=Path(__file__).resolve().parents[1];E=B/'research/round13/experiment';V=B/'research/round13/evaluation'
STUDY='3eed_full_description_association_vs_inequality_geometry_diagnostic_20261031'
ARMS=('source','A_full_consensus','A_full_order0','A_noun_consensus','A_deranged_consensus','B_inequality_full','B_inequality_noRGB','B_inequality_half','B_old_surface_equality')
ROLE_COUNTS={'new_content108':108,'regression8':8}
FINAL_SCHEMA_STATUS='pending_actual_composite_scored_and_independent_audit_contract_review'
REQUIRED=('new_source_seal','reused_source_seal','new_mask_seal','reused_mask_seal','association_preflight_seal','association_seal','geometry_seal','prediction_seal','scalar_summary','score_release','scorer_lock','input_audit','capture_audit','association_unlabelled_audit','geometry_unlabelled_audit','scored_audit','progression_gate','root_mechanism_review','root_report','root_decision','independent_final_report_review','root_final_review')
READINESS=('all116_A6calls_B_and_composite9arms_sealed_before_GT','actual_input_capture_A_B_and_scored_independent_audits_passed','new108_and_exposed_regression8_separate_no_pooled_confirmation','A_original_query_selection_B_fixedq0_positive_inequalities_and_controls_explicit','full_noun_deranged_two_numberings_UNKNOWN_and_mapping_failures_preserved','root_report_decision_progression_and_finalreview_actual_hash_bound','final_actual_producer_scorer_audit_schema_review_complete','privacy_no_rawIDs_scenes_captions_coordinates_masks_features_weights_panels_or_raw_answers')
CODE_FILES=tuple('crossplatform/round13/'+x for x in ('input_io.py','association.py','association_check_cpu.py','association_runner.py','support.py','geometry_observation.py','check_geometry_cpu.py','run_geometry.py','assemble_direct.py'))+tuple('research/round13/evaluation/'+x for x in ('audit_inputs.py','audit_capture108.py','audit_geometry116.py','cache_readiness.py','check_B_independent_cpu.py','score_direct.py','check_score_direct_cpu.py'))+('research/round13/prepare_inputs.py','research/round13/geometry/check_support_cpu.py','crossplatform/round12/capture_source.py','crossplatform/round12/capture_masks.py','crossplatform/round12/geometry_feedback.py','crossplatform/round10/rgb_support.py','crossplatform/round9/run_diagnostic.py','crossplatform/round9/fixed_q0_geometry.py','crossplatform/round9/head_exact_fit.py','crossplatform/round8/association_geometry.py','crossplatform/round8/semantic_encoder.py','crossplatform/round7/camera_projection.py','crossplatform/round7/regression_adapter.py','crossplatform/round7/experiment/runner_helpers.py','crossplatform/round6/factory.py','research/round12/evaluation/metrics.py','research/round5/audit/independent_core_v2.py','tools/obb_iou.py','tools/build_round13_export.py')
STATIC_JSON=tuple('research/round13/association/'+x for x in ('ASSET_INVENTORY.json','IMPLEMENTATION_LOCK.json','ASSOCIATION_PLAN_TEMPLATE.json','MODEL_FILES_SHA256.json','CPU_FIXTURE.json','CPU_FIXTURE_NUMPY23.json'))+tuple('research/round13/geometry/'+x for x in ('IMPLEMENTATION_LOCK.json','SUPPORT_IMPLEMENTATION_LOCK.json','CPU_FIXTURE.json','CPU_FIXTURE_prelock_before_duplicate_ID_efficiency.json','SUPPORT_CPU_FIXTURE.json'))+tuple('research/round13/evaluation/'+x for x in ('A_IMPLEMENTATION_PREFLIGHT_REVIEW.json','B_IMPLEMENTATION_INDEPENDENT_REVIEW.json','B_INDEPENDENT_CPU_FIXTURE.json','B_WRAPPER_SUPPORT_PRERUN_REVIEW.json','CACHE_REUSE_READINESS.json','SCORER_CPU_FIXTURE.json','SCORER_CPU_FIXTURE_before_platform_scene_namespace.json','SCORER_PROGRAM_LOCK.json','SCORE_INTERFACE.json'))+tuple('research/round13/experiment/'+x for x in ('AUTHORIZATION.json','EXPERIMENT_CARD.md','SOURCE_CAPTURE_PLAN_LOCK.json','MASK_CAPTURE_PLAN_LOCK.json','ASSOCIATION_PLAN_LOCK.json','GEOMETRY_PLAN_LOCK.json','PROGRESSION_PREPLAN_LOCK.json','SOURCE_EXECUTION_RECEIPT.json','ROOT_PREFLIGHT_INPUT_REVIEW.json','history/MASK_CAPTURE_PLAN_before_expected_count_fix.json'))+('research/round13/input/INPUT_PLAN_LOCK.json','research/round12/data_audit/PRIOR_PATH_DISJOINTNESS_ERRATUM.json')
# Narrative files use text-only path projection, never JSON parsing.
STATIC_JSON=tuple(x for x in STATIC_JSON if not x.endswith('.md'))
OPTIONAL_JSON=('research/round13/experiment/ASSOCIATION_PREFLIGHT_EXECUTION_RECEIPT.json','research/round13/experiment/COMPOSITE_PLAN_LOCK.json','research/round13/experiment/PROGRESSION_GATE.json','research/round13/evaluation/A_ASSOCIATION_UNLABELLED_AUDIT.json','research/round13/evaluation/ASSOCIATION116_UNLABELLED_AUDIT.json','research/round13/evaluation/SCORING_INDEPENDENT_AUDIT.json','research/round13/evaluation/FINAL_REPORT_INDEPENDENT_REVIEW.json')
DOC_FILES=('research/round13/association/INTERFACE_AND_MINIMAL_PLAN.md','research/round13/geometry/INTERFACE.md','research/round13/geometry/PROPOSAL.md','research/round13/geometry/SUPPORT_INTERFACE.md','research/round13/evaluation/PROTOCOL_RECOMMENDATION.md','research/round13/evaluation/SCORER_INTERFACE.md','research/round13/experiment/EXPERIMENT_CARD.md','research/round13/experiment/REPRODUCTION.md')
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

def privacy_context():
 corpus=read(B/'research/round8/experiment/full5912_gpu_inputs.json');ids={r['id'] for r in corpus};scenes={r['scene'] for r in corpus};captions={r['caption'] for r in corpus};caption_hashes={sha(x.encode()) for x in captions};longcaptions=tuple(x for x in captions if len(x)>=24)
 @lru_cache(maxsize=32768)
 def textcheck(s):
  assert ('/'+'home'+'/') not in s and ('/'+'dataset_full'+'/') not in s and (chr(92)+'Users'+chr(92)) not in s and ('https:'+'//'+'1drv'+'.ms/') not in s
  assert not any(x in s for x in ids|scenes) and sha(s.encode()) not in caption_hashes and not any(x in s for x in longcaptions),'Private input/description string';return s
 def clean(v):
  if isinstance(v,dict):assert not set(v)&FORBIDDEN and not any(k.endswith('_PRIVATE') for k in v);return {textcheck(str(k)):clean(x) for k,x in v.items()}
  if isinstance(v,list):return [clean(x) for x in v]
  if isinstance(v,str):return textcheck(v)
  if isinstance(v,float):assert math.isfinite(v)
  assert v is None or isinstance(v,(int,float,bool));return v
 def project(v):
  if isinstance(v,dict):
   out={}
   for k,x in v.items():
    if k in FORBIDDEN|DROP_MAPS or k.endswith('_PRIVATE') or k.endswith(('_path','_dir')) or k in ids|scenes or any(q in k.lower() for q in ('pointpath','imagepath','metapath')) or '.npz' in k:continue
    if k in ('actual_changed_axis_counts','changed_axis6','accepted_axis6','abs_error6_m_mean','equal_scene_abs_error6_m_mean','abs6_error_reduction_m') and isinstance(x,list) and len(x)==6:out[k+'_scalar_components']={a:project(z) for a,z in zip(AXES,x)}
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
 for rel in CODE_FILES+DOC_FILES:textcheck(source_text(local(rel)));checked.append(rel)
 for rel in STATIC_JSON+tuple(r for r in OPTIONAL_JSON if (B/r).is_file()):clean(project(read(local(rel))));checked.append(rel)
 return {'state':'R13_finite_code_doc_protocol_privacy_preflight_pass_final_schema_pending','code_n':len(CODE_FILES),'doc_n':len(DOC_FILES),'static_json_n':len(STATIC_JSON),'optional_present_n':sum((B/r).is_file() for r in OPTIONAL_JSON),'checked_sources':{r:file_sha(local(r)) for r in checked},'builder_sha256':file_sha(Path(__file__)),'final_schema_status':FINAL_SCHEMA_STATUS,'no_target_GT_private_scoredrows_pixels_model_GPU_actual_build_or_upload':True}

# This scalar projector is tied to the actual current score_direct.evaluate_row
# source schema, never exercised on private R13 rows during preparation.
# Final activation additionally requires actual producer/scorer/audit review.
def arm_scalar_table(rows,inputs,case_roles):
 textcheck,clean,_=privacy_context();assert len(rows)==116 and len(inputs)==116
 im={x['id']:x for x in inputs};aliases={x['id']:f'case_{i+1:06d}' for i,x in enumerate(inputs)};scenes={s:f'scene_{i+1:02d}' for i,s in enumerate(sorted({(x['platform'],x['scene']) for x in inputs}))}
 fields=('case_alias','cohort','platform','scene_alias','arm','role','iou','Acc25','Acc50','delta_iou_vs_source','query_changed','actual_changed','changed_axis_bitmask','center_L2_error_m','size_L2_error_m','center_shift_L2_m','size_shift_L2_m','raw_nonpositive_sizes','status');buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=fields);w.writeheader();count=0;roles={}
 for r in rows:
  sid=r['id'];item=im[sid];cohort=r['case_role'];assert cohort==case_roles[sid] and cohort in ROLE_COUNTS and r['platform']==item['platform'] and r['scene']==item['scene'] and tuple(r['arms'])==ARMS;roles[cohort]=roles.get(cohort,0)+1
  for arm,x in r['arms'].items():
   role='frozen_source' if arm=='source' else 'A_frozen_VLM_original_query_selection' if arm.startswith('A_') else 'B_fixedq0_inequality_direct' if arm in ARMS[5:7] else 'B_fixedq0_half_interpolation_control' if arm==ARMS[7] else 'B_fixedq0_legacy_surface_equality_control'
   v={'case_alias':aliases[sid],'cohort':cohort,'platform':r['platform'],'scene_alias':scenes[(r['platform'],r['scene'])],'arm':arm,'role':role,'iou':x['iou'],'Acc25':x['Acc25'],'Acc50':x['Acc50'],'delta_iou_vs_source':x['iou']-r['arms']['source']['iou'],'query_changed':x['query_changed'],'actual_changed':x['actual_changed'],'changed_axis_bitmask':sum(int(z)<<j for j,z in enumerate(x['actual_changed_axes'])),'center_L2_error_m':x['coordinate_errors']['center_L2_m'],'size_L2_error_m':x['coordinate_errors']['size_L2_m'],'center_shift_L2_m':x['center_shift_L2_m'],'size_shift_L2_m':x['size_shift_L2_m'],'raw_nonpositive_sizes':x['raw_nonpositive_sizes'],'status':x['status']};assert v['Acc25']==(v['iou']>.25) and v['Acc50']==(v['iou']>.5);clean(v);w.writerow(v);count+=1
 assert roles==ROLE_COUNTS and count==1044
 return gzip.compress(buf.getvalue().encode(),mtime=0),{'fields':list(fields),'rows':1044,'unique_cases':116,'cohort_counts':ROLE_COUNTS,'nine_related_armoutputs_per_case_not_independent_samples':True,'gzip_mtime':0,'no_query_indices_rawIDs_or_coordinates':True,'new_content_and_regression_not_pooled_confirmation':True,'deranged_scope':'Inputintervention diagnostic; original-target score is original-label retention, not correct counterfactual-query accuracy.'}

def main(a):
 assert not a.output.exists() and a.output.resolve().is_relative_to(B.resolve()),'New bounded output required'
 # Deliberately fail before any final-result read/output creation. Root's actual
 # A/composite/scored/audit schemas are not yet independently closed. This is a
 # preparation tool, not an executable publication until that finite update.
 raise RuntimeError('R13 export remains prepared-only: actual final producer/scorer/audit schema closure pending. Use --preflight-only; do not fabricate final readiness.')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=E/'public_export');p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 if a.preflight_only:print(json.dumps(preflight(),indent=2))
 else:main(a)
