"""Finite R12 public export. Preparation is not permission to build or publish.
All writes follow final-source, immutable-seal, actual-audit and privacy gates.
"""
from pathlib import Path,PurePosixPath
from functools import lru_cache
import argparse,csv,gzip,hashlib,io,json,math
B=Path(__file__).resolve().parents[1];E=B/'research/round12/experiment';I=B/'research/round12/data_audit';V=B/'research/round12/evaluation'
STUDY='3eed_independent_instance_multiview_diagnostic_20261030'
ARMS=('source','grabcut_direct','single_instance_direct','multiview_instance_direct','wrongview_instance_direct','single_instance_half','multiview_instance_half','multiview_maskmatch_only_direct')
REQUIRED=('source_seal','mask_seal','prediction_seal','scalar_summary','score_release','input_audit','source_audit','mask_audit','unlabelled_audit','scored_audit','mechanism_summary','mechanism_seal','mechanism_gate_review','root_mechanism_review','root_report','root_decision','root_final_review')
READINESS=('actual146_source501_masks_and8_direct_arms_sealed_before_GT','actual_scored_and_independent_reduction_audits_passed','input_duplicates_erratum146_unequal_scene_counts_micro_and_equal_scene_explicit','independent_masks_and_accepted_views_do_not_prove_referring_identity_or_clock_sync','all8_roles_direct_no_neural_TTA_and_negative_zero_NA_cases_retained','root_report_decision_mechanism_and_finalreview_complete','privacy_no_originalIDs_scenes_captions_coordinates_masks_features_weights_or_media')
CODE_FILES=tuple('crossplatform/round12/'+x for x in ('capture_source.py','capture_masks.py','run_feedback.py','geometry_feedback.py','multiview_feedback.py','temporal_association.py','sensor_icp.py','sensor_pnp.py','check_real_views.py','check_real_views_icp.py','check_real_views_pnp.py','check_temporal_cpu.py','check_feedback_cpu.py'))+tuple('research/round12/data_audit/'+x for x in ('inventory_release_inputs.py','audit_all_target_input_metadata.py','check_duplicate_metadata_content.py','hash_physical_input_frames.py','prepare_physical_anchor_inputs.py','finalize_input_audit.py','audit_source146.py','audit_masks501.py','prepare_input_gallery_plan.py'))+tuple('research/round12/evaluation/'+x for x in ('metrics.py','score_feedback.py','audit_geometry.py','audit_input_selection.py','audit_scored_feedback.py','audit_mechanism_gate.py','audit_support_proxy.py','audit_final_report.py','diagnose_support_postseal.py','check_audit_geometry_cpu.py','check_pnp_independent_cpu.py','check_scored_auditor_cpu.py','check_temporal_independent_cpu.py'))+tuple('research/round12/mechanism/'+x for x in ('reduce_mechanism.py','check_reducer_cpu.py'))+('research/round12/qualitative/render_input_gallery5.py','research/round12/qualitative/render_g_change_gallery.py','tools/history/build_round12_export_pre_postseal_support.py','crossplatform/round10/rgb_support.py','crossplatform/round9/run_diagnostic.py','crossplatform/round9/fixed_q0_geometry.py','crossplatform/round9/head_exact_fit.py','crossplatform/round8/association_geometry.py','crossplatform/round8/semantic_encoder.py','crossplatform/round7/camera_projection.py','crossplatform/round7/regression_adapter.py','crossplatform/round7/experiment/runner_helpers.py','crossplatform/round6/factory.py','research/round5/audit/independent_core_v2.py','tools/obb_iou.py','tools/build_round12_export.py')+tuple('crossplatform/round12/history/'+x for x in ('preflight_G_v1_before_perpoint_corroboration/multiview_feedback.py','preflight_G_v1_before_perpoint_corroboration/run_feedback.py','preflight_G_v2_SIFT_before_PnP_backend/multiview_feedback.py','preflight_G_v2_SIFT_before_PnP_backend/run_feedback.py','preflight_PnP_v1_before_empty_B_holdout/sensor_pnp.py','preflight_v1_before_degenerate_consensus/temporal_association.py'))
STATIC_JSON=tuple('research/round12/data_audit/'+x for x in ('INPUT_PLAN_LOCK.json','INPUT_RECEIPT.json','INPUT_PREPARATION_INDEPENDENT_REPLAY.json','PRIOR_PATH_DISJOINTNESS_ERRATUM.json','CORRESPONDENCE_PREREQUISITE_AUDIT.json','PHYSICAL_INPUT_DUPLICATE_AUDIT.json','DUPLICATE_TIMESTAMP_CONTENT_AUDIT.json','TARGET_INPUT_METADATA_COVERAGE_v2.json','TIMESTAMP_CONSISTENCY_SUPPLEMENT.json','RELEASE_ARCHIVE_INPUT_ASSET_INVENTORY.json','INPUT_GALLERY5_PLAN_LOCK.json','history/PATH_ONLY_V1_PROPOSAL.json'))+tuple('research/round12/experiment/'+x for x in ('AUTHORIZATION.json','SOURCE_CAPTURE_PLAN_LOCK.json','MASK_CAPTURE_PLAN_LOCK.json','MASK_ALIAS_INTERFACE_CLOSURE.json','FEEDBACK_PLAN_LOCK.json','PROGRESSION_PREPLAN_LOCK.json','REAL_VIEW_PROBE_PLAN_LOCK.json','REAL_VIEW_ICP_PROBE_PLAN_LOCK.json','REAL_VIEW_PNP_PROBE_PLAN_LOCK.json','ICP_CPU_FIXTURE.json','PNP_CPU_FIXTURE.json','PNP_EMPTY_HOLDOUT_FIX.json','MECHANISM_READ_RELEASE.json','POSTSEAL_SUPPORT_DIAGNOSTIC_PLAN.json','support_proxy146_attempt001/SUMMARY.json','support_proxy146_attempt001/SEALED.json','POSTSEAL_PAIRED_SUPPORT_SUMMARY.json','G_CHANGE_GALLERY_PLAN.json'))+tuple('research/round12/evaluation/'+x for x in ('AUDIT_GEOMETRY_CPU_FIXTURE.json','COHORT146_EVALUATION_SUPPLEMENT.json','EVALUATION_PROPOSAL.json','FEEDBACK_IMPLEMENTATION_INDEPENDENT_REVIEW.json','FEEDBACK_PREFLIGHT_FINDING.json','ICP_INDEPENDENT_CPU_FIXTURE.json','MASK_CAPTURE_IMPLEMENTATION_REVIEW.json','MASK_CAPTURE_INPUT_CLOSURE_REVIEW.json','PERPOINT_MAPPING_INDEPENDENT_CPU_FIXTURE.json','PNP_INDEPENDENT_CPU_FIXTURE.json','PREPARATION_UPDATE_RECEIPT.json','SCALAR_AUDITOR_CODE_LOCK.json','SCALAR_AUDITOR_CPU_FIXTURE.json','SCORER_ADAPTATION_SPEC.json','SCORER_IMPLEMENTATION_REVIEW.json','SENSOR_REGISTRATION_IMPLEMENTATION_REVIEW.json','TEMPORAL_IMPLEMENTATION_INDEPENDENT_REVIEW.json','TEMPORAL_INDEPENDENT_CPU_FIXTURE.json','UNLABELLED_AUDITOR_CODE_LOCK.json','VERIFIED_MULTIVIEW_PROTOCOL_SUPPLEMENT.json'))+tuple('research/round12/mechanism/'+x for x in ('CPU_FIXTURE.json','EXECUTION_RECEIPT.json','ACTUAL146_AUTHOR_INTERPRETATION.json'))+tuple('crossplatform/round12/'+x for x in ('IMPLEMENTATION_LOCK.json','FEEDBACK_IMPLEMENTATION_LOCK.json','FEEDBACK_PLAN_TEMPLATE.json','CPU_FIXTURE.json','FEEDBACK_CPU_FIXTURE.json','history/preflight_G_v2_SIFT_before_PnP_backend/FEEDBACK_IMPLEMENTATION_LOCK.json','history/preflight_G_v2_SIFT_before_PnP_backend/FEEDBACK_CPU_FIXTURE.json','history/preflight_v1_before_degenerate_consensus/CPU_FIXTURE.json'))+('research/round12/qualitative/VISUAL_REVIEW.json','research/round12/qualitative/G_CHANGE_VISUAL_REVIEW.json')
OPTIONAL_JSON=('research/round12/evaluation/SUPPORT_PROXY_IMPLEMENTATION_REVIEW.json','research/round12/experiment/PROGRESSION_GATE.json','research/round12/experiment/ROOT_INPUT_GALLERY_REVIEW.json','research/round12/experiment/ROOT_CHANGED_GALLERY_REVIEW.json','research/round12/evaluation/FINAL_REPORT_INDEPENDENT_REVIEW.json')
DOC_FILES=('crossplatform/round12/TEMPORAL_INTERFACE.md','crossplatform/round12/FEEDBACK_INTERFACE.md','research/round12/proposal/MASK_INTERFACE.md','research/round12/proposal/OBSERVATION_AND_METHOD_PROPOSAL.md','research/round12/evaluation/MINIMAL_PROGRESSION_CARD.md','research/round12/mechanism/README.md','research/round12/mechanism/ACTUAL146_AUTHOR_INTERPRETATION.md','research/round12/mechanism/actual146_attempt001/SUMMARY.md','research/round12/experiment/EXPERIMENT_CARD.md','research/round12/experiment/REPRODUCTION.md')
PROBES=(('SIFT_two_depth','crossplatform/round12/real_view_probe5_attempt001'),('ICP_RGB','crossplatform/round12/real_view_icp_probe5_attempt001'),('one_depth_PnP','crossplatform/round12/real_view_pnp_probe5_attempt001'))
FORBIDDEN={'id','ids','case_id','scene','frame','caption','captions','utterance','ground_info','bbox_3d','gt_bbox','positive_map','GT7D_model','target7','target6','source6','box6','box7','center3','size3','source_boxes_raw6','source_boxes_eval6','arm_boxes_raw6','arm_boxes_eval6','points','sampled_points_model','raw_indices','center_features','size_features','image_intrinsic','image_extrinsic','camera_distortion','embeddings','proj_queries','proj_tokens','roi_xyxy','mask2d','uv','depth','abs_error6_m','selected_ids','pose','K','E','D','T','T_raw_B_to_A','transformation','rotation_matrix','translation_vector','coordinate_residual6','target6_modelcoords','source_box6','box_corners','provided_pose_diagnostic'}
DROP_MAPS={'files','source_cache_files','source_input_paths','weights_relative_private_location','workspace_root','programroot','rows','rows_PRIVATE','cases','candidates','pairs','selected','case_alias_notes','selected_cases','selected_rows','anchor_ids','frame_aliases','examples','timestamp_sequence_examples'}
AXES=('cx','cy','cz','sx','sy','sz')
sha=lambda x:hashlib.sha256(x).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8<<20),b''):h.update(z)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def local(rel):
 q=PurePosixPath(rel);assert q.parts and not q.is_absolute() and '..' not in q.parts;p=B/q.as_posix();assert p.resolve().is_relative_to(B.resolve()) and p.is_file(),rel;return p

def privacy_context():
 corpus=read(B/'research/round8/experiment/full5912_gpu_inputs.json');ids={r['id'] for r in corpus};scenes={r['scene'] for r in corpus};caption_hashes={sha(r['caption'].encode()) for r in corpus}
 @lru_cache(maxsize=32768)
 def textcheck(s):
  assert ('/'+'home'+'/') not in s and ('/'+'dataset_full'+'/') not in s and (chr(92)+'Users'+chr(92)) not in s and ('https:'+'//'+'1drv'+'.ms/') not in s
  assert not any(x in s for x in ids|scenes) and sha(s.encode()) not in caption_hashes,'Private input string';return s
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
    if k in ('actual_changed_axis_counts','changed_axis6','accepted_axis6') and isinstance(x,list) and len(x)==6:out[k+'_scalar_components']={a:z for a,z in zip(AXES,x)}
    else:out[k]=project(x)
   return out
  if isinstance(v,list):
   if v and all(isinstance(x,(list,tuple)) for x in v):return {'matrix_or_nested_observation_array_omitted':True,'rows':len(v)}
   if len(v) in (2,3,4,6,7,9,16) and all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in v):return {'numeric_vector_omitted_for_coordinate_privacy':True,'length':len(v)}
   return [project(x) for x in v]
  if isinstance(v,str):
   if v.startswith('/') or ('/'+'home'+'/') in v:return '[local_path_omitted]'
   return v.replace(str(B),'__WORKSPACE_ROOT__')
  return v
 return ids,scenes,textcheck,clean,project

def verify_seal(path):
 seal=read(path)
 for rel,h in seal['files'].items():q=PurePosixPath(rel);assert not q.is_absolute() and '..' not in q.parts and file_sha(path.parent/rel)==h
 return seal

def probe_projection(path,project):
 d=read(path/'RESULTS_PRIVATE.json');s=verify_seal(path/'SEALED.json');assert d['n_pairs']==5 and d['no_targetGT_or_sourceq0_or_neural_updates'] and s['files']['RESULTS_PRIVATE.json']==file_sha(path/'RESULTS_PRIVATE.json')
 rows=[]
 for i,r in enumerate(d['rows']):
  roles={}
  for role,v in r['pair_diagnostics'].items():
   assert role in ('helper_frame_alias','wrong_frame_alias');roles['candidate_neighbor' if role=='helper_frame_alias' else 'wrong_scene'] = project(v)
  rows.append({'pair_alias':f'probe_pair_{i+1:02d}','platform':r['platform'],'registration':roles})
 return {'state':d['state'],'n_pairs':5,'seconds':d['seconds'],'source_result_sha256':file_sha(path/'RESULTS_PRIVATE.json'),'source_seal_sha256':file_sha(path/'SEALED.json'),'program_sha256':d['program_sha256'],'plan_sha256':d['plan_sha256'],'mask_seal_sha256':d['mask_seal_sha256'],'accepted_real':sum(r['pair_diagnostics']['helper_frame_alias']['accepted'] for r in d['rows']),'accepted_wrong':sum(r['pair_diagnostics']['wrong_frame_alias']['accepted'] for r in d['rows']),'rows_aliased':rows,'comparison_scope':'Input-development5pairs; methods have different validation/holdout criteria, not equal-protocol superiority or task performance. No sourceq0 or targetGT. No transform or pose is exported.'}

def source_text(p):
 return p.read_text().replace(str(B),'__WORKSPACE_ROOT__').replace('crossplatform'+'/'+'dataset_full'+'/'+'3eed_dataset','__DATASET_ROOT__')

def preflight():
 _,_,textcheck,clean,project=privacy_context();checked=[]
 for rel in CODE_FILES+DOC_FILES:textcheck(source_text(local(rel)));checked.append(rel)
 for rel in STATIC_JSON+tuple(r for r in OPTIONAL_JSON if (B/r).is_file()):clean(project(read(local(rel))));checked.append(rel)
 for _,rel in PROBES:clean(probe_projection(local(rel+'/SEALED.json').parent,project));checked.append(rel+'/RESULTS_PRIVATE.json')
 return {'state':'R12_finite_allowlist_string_and_projection_preflight_pass','code_n':len(CODE_FILES),'doc_n':len(DOC_FILES),'static_json_n':len(STATIC_JSON),'probe_n':3,'checked_sources':{r:file_sha(local(r)) for r in checked},'builder_sha256':file_sha(Path(__file__)),'no_GT_private_scoredrows_model_GPU_build_or_upload':True}

def main(a):
 assert not a.output.exists() and a.output.resolve().is_relative_to(B.resolve()),'New bounded output required'
 ready=read(E/'PUBLICATION_READINESS.json')
 for k in READINESS:assert ready.get(k) is True,('Missing final gate',k)
 specs=read(E/'EXPORT_FINAL_SOURCES.json');bound={}
 for k in REQUIRED:
  spec=specs[k];assert set(spec)=={'path','sha256'};p=local(spec['path']);assert file_sha(p)==spec['sha256'];bound[k]=p
 src=verify_seal(bound['source_seal']);mask=verify_seal(bound['mask_seal']);pred=verify_seal(bound['prediction_seal']);summary=read(bound['scalar_summary']);sp=bound['scalar_summary'].parent/'SEALED.json';ss=verify_seal(sp)
 assert src['state']=='sealed_R12_source_forward_and_input_observations_no_G_no_GT' and src['n_samples']==146 and src['source_unchanged'] and src['source_repeat_bitexact']
 assert mask['state']=='sealed_R12_independent_whole_image_instance_masks_no_q0_caption_or_GT' and mask['frame_n']==501 and not mask['trained_or_updated'] and mask['mask_returned_original_image_pixels']
 assert pred['state']=='sealed_R12_8arm_independent_instance_multiview_direct_before_GT' and pred['n_samples']==146 and pred['n_arms']==8 and tuple(pred['arm_names'])==ARMS and pred['all_queries_fixed_original_q0'] and not pred['GT_model_GPU_or_neural_update_executed'] and not pred['provided_pose_used_for_registration_or_target_fusion']
 assert pred['source_seal_sha256']==file_sha(bound['source_seal']) and pred['mask_seal_sha256']==file_sha(bound['mask_seal'])
 assert src['source_checkpoint_sha256']==pred['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
 assert mask['weights_sha256']==pred['mask_weights_sha256']=='73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e'
 for s in (src,mask):assert s['strict_FP32_flags']=={'cudnn_allow_tf32':False,'matmul_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}
 assert pred['program_sha256_base']=='workspace_root'
 for r,h in pred['program_sha256'].items():assert file_sha(local(r))==h
 assert summary['state']=='R12_146_independent_instance_multiview_scored_after_complete_seal' and summary['n']==ss['n_samples']==146 and ss['prediction_seal_sha256']==file_sha(bound['prediction_seal'])
 release=read(bound['score_release']);assert release['CPU_SCORE_RELEASE'] and release['all_source_masks_geometry_arms_sealed_before_GT'] and release['stage']=='R12_independent_instance_multiview_direct' and ss['release_sha256']==file_sha(bound['score_release'])
 for k in ('source','mask','prediction'):assert release[k+'_seal_sha256']==file_sha(bound[k+'_seal'])
 assert release['scorer_sha256']==file_sha(V/'score_feedback.py') and release['metrics_sha256']==file_sha(V/'metrics.py')
 ia=read(bound['input_audit']);assert ia['state']=='PASS_R12_INDEPENDENT_INPUT_SELECTION_CONTENT_AND_SOURCE_CONTRACT' and ia['anchor_manifest_sha256']==pred['manifest_sha256'] and ia['input_receipt_sha256']==file_sha(I/'INPUT_RECEIPT.json') and ia['auditor_sha256']==file_sha(V/'audit_input_selection.py')
 for k,state,bind in (('source_audit','independent_actual_R12_source146_input_observation_binding_audit_pass','source_seal'),('mask_audit','independent_actual_R12_mask501_observation_binding_audit_pass','mask_seal'),('unlabelled_audit','PASS_R12_ACTUAL146_8ARM_UNLABELLED_AUDIT','prediction_seal')):
  d=read(bound[k]);assert d['state']==state and d[bind+'_sha256']==file_sha(bound[bind])
  auditor=I/('audit_source146.py' if k=='source_audit' else 'audit_masks501.py') if k!='unlabelled_audit' else V/'audit_geometry.py';assert d['auditor_sha256']==file_sha(auditor)
 resultpath=bound['scalar_summary'].parent/'RESULTS_PRIVATE.json';ra=read(bound['scored_audit']);assert ra['state']=='PASS_R12_INDEPENDENT_SCALAR_REDUCTION_AUDIT' and ra['n']==146 and not ra['failures'] and ra['results_sha256']==file_sha(resultpath) and ra['scored_seal_sha256']==file_sha(sp) and ra['prediction_seal_sha256']==file_sha(bound['prediction_seal']) and ra['release_sha256']==file_sha(bound['score_release']) and ra['auditor_sha256']==file_sha(V/'audit_scored_feedback.py')
 ms=verify_seal(bound['mechanism_seal']);mu=read(bound['mechanism_summary']);assert ms['state']=='sealed_R12_author_no_GT_mechanism_census' and ms['files']['SUMMARY.json']==file_sha(bound['mechanism_summary']) and ms['reducer_sha256']==file_sha(B/'research/round12/mechanism/reduce_mechanism.py') and mu['binding']['geometry_seal_sha256']==file_sha(bound['prediction_seal'])
 mg=read(bound['mechanism_gate_review']);assert mg['state']=='PASS_R12_INDEPENDENT_MECHANISM_AND_PRELOCKED_PROGRESSION_REVIEW' and mg['prediction_seal_sha256']==file_sha(bound['prediction_seal']) and mg['scored_seal_sha256']==file_sha(sp) and mg['mechanism_seal_sha256']==file_sha(bound['mechanism_seal']) and mg['mechanism_summary_sha256']==file_sha(bound['mechanism_summary']) and mg['score_audit_sha256']==file_sha(bound['scored_audit']) and mg['auditor_sha256']==file_sha(V/'audit_mechanism_gate.py')
 rm=read(bound['root_mechanism_review']);rr=read(bound['root_final_review']);assert rm['mechanism_seal_sha256']==file_sha(bound['mechanism_seal']) and rr['publication_ready'] and rr['report_sha256']==file_sha(bound['root_report']) and rr['decision_sha256']==file_sha(bound['root_decision']) and rr['mechanism_seal_sha256']==file_sha(bound['mechanism_seal'])
 supportdir=E/'support_proxy146_attempt001';supportseal=verify_seal(supportdir/'SEALED.json');support=read(supportdir/'SUMMARY.json');supportplan=read(E/'POSTSEAL_SUPPORT_DIAGNOSTIC_PLAN.json')
 assert support['state']=='postseal_descriptive_official_target_OBB_support_proxy_not_semanticGT' and support['n']==146 and support['prediction_seal_sha256']==file_sha(bound['prediction_seal']) and support['scored_seal_sha256']==file_sha(sp) and support['program_sha256']==file_sha(V/'diagnose_support_postseal.py') and support['plan_sha256']==file_sha(E/'POSTSEAL_SUPPORT_DIAGNOSTIC_PLAN.json')
 assert supportseal['files']['SUMMARY.json']==file_sha(supportdir/'SUMMARY.json') and supportseal['program_sha256']==support['program_sha256'] and supportseal['plan_sha256']==support['plan_sha256'] and supportplan['postseal_descriptive_only'] and not supportplan['method_changes_allowed']
 inputs=read(I/'ANCHOR146_GPU_INPUTS.json');assert len(inputs)==146 and file_sha(I/'ANCHOR146_GPU_INPUTS.json')==pred['manifest_sha256']==src['manifest_sha256']
 assert {p:sum(r['platform']==p for r in inputs) for p in ('drone','quad')}=={'drone':50,'quad':96}
 ids,scenes,textcheck,clean,project=privacy_context();pending={};provenance=[]
 def emit(name,data):assert name not in pending;pending[name]=data
 def js(name,obj,origin=None,mode='explicit_scalar_and_privacy_field_projection'):
  data=(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n').encode();emit(name,data)
  if origin:provenance.append({'source_relative':str(origin.relative_to(B)),'source_sha256':file_sha(origin),'export_relative':name,'export_sha256':sha(data),'mode':mode})
 for group,prefix in ((CODE_FILES,'code/'),(DOC_FILES,'docs/')):
  for rel in group:
   p=local(rel);data=textcheck(source_text(p)).encode();emit(prefix+rel,data);provenance.append({'source_relative':rel,'source_sha256':file_sha(p),'export_relative':prefix+rel,'export_sha256':sha(data),'mode':'workspace_path_projection_only'})
 for rel in STATIC_JSON+tuple(r for r in OPTIONAL_JSON if (B/r).is_file()):js('protocol/'+rel,project(read(local(rel))),local(rel))
 for label,rel in PROBES:js('input_probes/'+label+'.json',probe_projection(B/rel,project),B/rel/'RESULTS_PRIVATE.json','probe_alias_scalar_residual_projection_no_T_pose_coordinates')
 for key,p in bound.items():
  if key in ('root_report','root_decision'):
   data=textcheck(source_text(p)).encode();emit('final/'+key+p.suffix,data);provenance.append({'source_relative':str(p.relative_to(B)),'source_sha256':file_sha(p),'export_relative':'final/'+key+p.suffix,'export_sha256':sha(data),'mode':'narrative_path_projection_only'})
  else:js('final/'+key+'.json',project(read(p)),p)
 actual=read(resultpath);assert {k:v for k,v in actual.items() if k!='rows_PRIVATE'}==summary;rows=actual['rows_PRIVATE'];assert len(rows)==len({r['id'] for r in rows})==146
 aliases={r['id']:f'case_{i+1:06d}' for i,r in enumerate(inputs)};scene_alias={s:f'scene_{i+1:02d}' for i,s in enumerate(sorted({(r['platform'],r['scene']) for r in inputs}))};im={r['id']:r for r in inputs}
 fields=('case_alias','platform','scene_alias','arm','role','iou','Acc25','Acc50','delta_iou_vs_source','center_L2_error_m','size_L2_error_m','center_shift_L2_m','size_shift_L2_m','accepted_axis_bitmask','accepted_any','changed_axis_bitmask','actual_changed','raw_nonpositive_sizes','status')
 buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=fields);writer.writeheader();n=0
 for row in rows:
  assert row['platform']==im[row['id']]['platform'] and row['scene']==im[row['id']]['scene'] and tuple(row['arms'])==ARMS
  base={'case_alias':aliases[row['id']],'platform':row['platform'],'scene_alias':scene_alias[(row['platform'],row['scene'])]};src_iou=row['arms']['source']['iou']
  for arm,x in row['arms'].items():
   record={**base,'arm':arm,'role':'frozen_Waymo_source' if arm=='source' else 'direct_no_neural_'+('unfiltered_samegate_aggregation' if arm==ARMS[7] else arm),'iou':x['iou'],'Acc25':x['iou']>.25,'Acc50':x['iou']>.5,'delta_iou_vs_source':x['iou']-src_iou,'center_L2_error_m':x['coordinate_errors']['center_L2_m'],'size_L2_error_m':x['coordinate_errors']['size_L2_m'],'center_shift_L2_m':x['center_shift_L2_m'],'size_shift_L2_m':x['size_shift_L2_m'],'accepted_axis_bitmask':sum(int(z)<<j for j,z in enumerate(x['accepted_axes'])),'accepted_any':x['accepted_any'],'changed_axis_bitmask':sum(int(z)<<j for j,z in enumerate(x['actual_changed_axes'])),'actual_changed':x['actual_changed'],'raw_nonpositive_sizes':x['raw_nonpositive_sizes'],'status':x['status']};clean(record);writer.writerow(record);n+=1
 assert n==1168
 for t in summary['summary']:
  if t['population']=='all':
   rr0=[r for r in rows if r['platform']==t['platform']];values=[r['arms'][t['arm']]['iou'] for r in rr0];assert len(values)==t['n'] and abs(math.fsum(values)/len(values)-t['mean_iou'])<1e-14
   byscene={s:[r['arms'][t['arm']]['iou'] for r in rr0 if r['scene']==s] for s in {r['scene'] for r in rr0}};eq=math.fsum(math.fsum(v)/len(v) for v in byscene.values())/len(byscene);assert abs(eq-t['equal_scene_mean_iou'])<1e-14
 emit('results/ARM_SCALARS.csv.gz',gzip.compress(buf.getvalue().encode(),mtime=0));js('results/ARM_SCALARS_SCHEMA.json',{'rows':1168,'unique_cases':146,'platform_cases':{'drone':50,'quad':96},'fields':list(fields),'gzip_mtime':0,'all8_roles_direct_no_neural':True,'related_outputs_not_independent_samples':True,'scene_weighting':'Micro and equal-scene means both retained; Drone scene quotas32/18 differ.','coordinate_boundary':'Error/movement magnitudes and accepted/changed bitmasks only. No physical or GT coordinate vectors.','zero_negative_NA_fallback_cases_preserved':True})
 js('EXPORT_PROVENANCE.json',{'sources':provenance,'final_sources':{k:{'source_relative':str(p.relative_to(B)),'source_sha256':file_sha(p)} for k,p in bound.items()},'source_checkpoint_sha256':src['source_checkpoint_sha256'],'mask_weights_sha256':mask['weights_sha256'],'scope':'Finite code/protocol/alias scalars. No neuralTTA or literal semantic identity confirmation; external frozen2Dmask frontend is additional input evidence, not a3Dbox teacher.'})
 emit('README.md',(STUDY+'\nSee final report/decision and reproduction notes.146content-disjoint anchors,501frontendframes,8fixedq0direct arms. Clock synchronization/pose direction/target identity are not established by release counters, independent masks or low residuals. Probe methods have unequal validation criteria. Legacy8tharm is unfiltered samegate multiview, not anchor-only.1168relatedarmrows are not1168independent samples. Media, original IDs/scenes/captions, transforms, coordinates, features and weights are private.\n').encode())
 for name,data in pending.items():assert not any(x.encode() in data for x in ids|scenes),(name,'Private identifier');assert not name.endswith(('.pt','.pth','.bin','.safetensors','.npz','.npy','.jpg','.png','.mp4','.wav'))
 a.output.mkdir(parents=True)
 for name,data in pending.items():p=a.output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 manifest={'study':STUDY,'unique_cases':146,'arm_rows':1168,'all8_roles_direct_no_neural':True,'files':[{'path':name,'bytes':len(data),'sha256':sha(data)} for name,data in sorted(pending.items())],'source_checkpoint_sha256':src['source_checkpoint_sha256'],'mask_weights_sha256':mask['weights_sha256'],'root_final_review_sha256':file_sha(bound['root_final_review'])};(a.output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'n_files':len(pending)+1,'manifest_sha256':file_sha(a.output/'MANIFEST.json')}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=E/'public_export');p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 if a.preflight_only:print(json.dumps(preflight(),indent=2))
 else:main(a)
