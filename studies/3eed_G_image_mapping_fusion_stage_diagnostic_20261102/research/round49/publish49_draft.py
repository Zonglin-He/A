"""UNEXECUTED explicit scalar-only local public export. No network or scoring.

Requires root's final report, both sealed review phases, joined/audited results,
the single task scoring receipt, and a new export grant. No FINAL access.
Raw manifests/maps/notes/media/vectors/weights are never copied to the bundle.
"""
from pathlib import Path
import argparse,csv,datetime,hashlib,io,json,re,statistics

R=Path(__file__).resolve().parents[2];H=R/'research/round49';A=H/'independent_audit'
PREFIX='studies/3eed_G_image_mapping_fusion_stage_diagnostic_20261102'
ARMS=[f'legacy_V{v}P{p}R{r}' for v in (0,1) for p in (0,1) for r in (0,1)]+[
 'legacy_rawvv_mapped_cue_only','legacy_postclipV_mapped_cue_only','legacy_fullG_mapped_only','legacy_fullG_V_unclipped',
 'legacy_fullG_no_gray','joint_e_fullG_same_operator','joint_literal_source_plus_rawcue','joint_rawcue_mapped_only']
PARTS=['ALL336','D208','V64','M64'];PLATFORMS=['drone','quad']
OWN_CODE=[
 'crossplatform/round49/inventory49.py','crossplatform/round49/stages49.py','crossplatform/round49/predict49.py',
 'crossplatform/round49/ownership49.py','crossplatform/round49/blindjoin49.py',
 'crossplatform/round49/blind_rgb49_draft.py','crossplatform/round49/subjects49.py',
 'crossplatform/round49/copy_subject_cards49.py','research/round49/independent_audit/audit_blindjoin49.py',
 'research/round49/independent_audit/audit49_independent.py','research/round49/independent_audit/reference49.py',
 'research/round49/independent_audit/score49_independent.py','research/round49/independent_audit/audit_ownership49.py',
 'research/round49/independent_audit/structure49.py','research/round49/independent_audit/structure_ranges49.py',
 'tools/manual_p1_guard49.py','research/round49/publish49_draft.py',
 'research/round49/write_report49.py','research/round49/independent_audit/audit_saved_scalars49.py',
 'research/round49/independent_audit/score49_console_fixed.py']
STRUCT_FIELDS=['pool_mask_count','no_G_pool','mapped_query_count','no_mapped_query','q0_unmapped','pool_exists_and_q0_unmapped',
 'query_positive_weight_mask_count_max','multi_mask_mapped_queries','per_mask_positive_query_broadcast_max',
 'per_mask_positive_query_broadcast_median','raw_cue_top_tie_count','postclip_cue_top_tie_count','raw_cue_top_tie',
 'postclip_cue_top_tie','raw_postclip_winner_changed','full_vs_mapped_winner_changed','full_vs_unclipV_winner_changed',
 'full_vs_no_gray_winner_changed','full_vs_joint_same_operator_winner_changed','full_vs_joint_literal_winner_changed',
 'V_clip_query_count','P_clip_query_count','R_clip_query_count','subject_query_equals_fullcaption_exact',
 'global_relations_parsed_count','global_relations_original_implemented_count','reference_entities_parsed_count',
 'reference_relations_parsed_count','reference_relations_camera_image_count','reference_relations_unknown_frame_count',
 'reference_entities_with_original_detector_boxes','reference_entities_with_stable_detector_boxes',
 'reference_evidence_anchor_boxes_count','reference_evidence_anchor_nonzero_weights_before_mask_skip',
 'P_component_all_zero','R_component_all_zero','no_identity_error_claim']
RANGE_FLAGS=['pool_size','has_mapping','source_plus_V_vs_rawcue_same_winner','fullG_vs_rawcue_same_winner',
 'V_all_zero','P_all_zero','R_all_zero','q0_mapped','q0_all_VPR_exact_zero','unmapped_fullG_equals_source',
 'mapped_q0_no_escape_conditions_hold','fullG_winner_unmapped','source_native_top_tie_count','no_identity_or_mapping_correctness_claim']
BLIND_STATUS_FIELDS=['phase1_status','legacy_visual_vs_actual_partial_subject','legacy_raw_full_vs_actual_partial_subject',
 'legacy_gray_corrected_vs_actual_partial_subject','legacy_visual_vs_fullcaption_identity','joint_visual_vs_fullcaption_identity',
 'legacy_pool_identity_coverage_status']
BLIND_SCALARS=['raw_full_and_gray_corrected_topsets_equal','partial_compatible_candidates_in_compared_pool',
 'legacy_compared_pool_mask_count','has_legacy_visual_pool','partial_incompatible_candidates_in_compared_pool',
 'partial_UNKNOWN_candidates_in_compared_pool','partial_compatible_candidate_exists_in_compared_pool',
 'legacy_rawfull_all_mask_means_negative','legacy_rawfull_max_mask_mean_logodds','gray_corrected_e_is_not_PYes',
 'rawfull_mean_negative_is_not_all_views_negative',
 'target_mask_not_in_legacy_pool','single_target_outside_legacy_pool','ambiguous_all_alternatives_outside_legacy_pool',
 'duplicate_exact_pixelmask_hashes_n','nonexpert_blind_labels_not_gold','no_taskGT_or_IoU',
 'legacy_partial_vs_fullcaption_disagreement_not_automatically_VLMerror']
POINT_FIELDS=['allvisible','unassigned','ambiguous','foreground_exclusive_points','unassigned_over_allvisible',
 'ambiguous_over_allvisible','target_exclusive_points','other_foreground_exclusive_points','target_fraction',
 'target_share_of_exclusive_foreground','target_exclusive_points_over_allvisible','mapped','W_target_mass','R28_fixed_support_proxy_not_gold']
INVENTORY_FIELDS=['target_fullframe_exclusive_points','fullframe_exclusive_foreground_points','fullframe_unassigned_points',
 'fullframe_ambiguous_points','target_frame_point_count_is_unique_not_query_sum','target_supported_query_count',
 'any_original_query_target_supported','target_supported_queries_with_any_W_mapping',
 'target_supported_queries_with_positive_target_W_mass','target_supported_queries_without_target_W_mass',
 'fullframe_positive_target_points_does_not_certify_depth_completeness',
 'zero_target_W_mass_does_not_identify_which_coarse_or_geometric_gate_caused_it']
PHASE1_STATUSES={'SINGLE_VISIBLE_TARGET','AMBIGUOUS_TARGET','NONE','UNKNOWN'}
VISUAL_STATUSES={'UNKNOWN_no_visual_evidence_pool','UNKNOWN_no_resolved_fullcaption_identity','SUPPORTED',
 'DISJOINT_FROM_SINGLE_VISIBLE_TARGET','TIE_TARGET_OTHER','VISIBLY_INCOMPATIBLE_WITH_PARTIAL_SUBJECT',
 'UNKNOWN_OR_MIXED','UNKNOWN_known_target_not_in_compared_pool'}
POINT_STATUSES={'target_supported','other_supported','UNKNOWN_sparse_or_mixed','UNKNOWN_nonpositive_source_size'}|{'UNKNOWN_fullcaption_'+x for x in PHASE1_STATUSES}
POOL_STATUSES={'SINGLE_target_outside_pool','SINGLE_target_intersects_pool','AMBIGUOUS_all_declared_alternatives_outside_pool',
 'AMBIGUOUS_at_least_one_alternative_in_pool_or_no_declared_ids','UNKNOWN_target_coverage_NONE','UNKNOWN_target_coverage_UNKNOWN'}
RECEIPT_FIELDS=['completed_numeric_artifacts_despite_terminal_print_NameError','original_scoring_process_exit1_not_ignored','scientific_numeric_artifacts_complete_and_consistent','IoU_values_reaggregated','ordered_negative_comparisons_reproduced','scene_summary_rows_reproduced','numeric_comparisons','maximum_absolute_scalar_error','GT_geometry_metadata_media_model_FINAL_reads','state','n','arms','task_values','negative_ordered_comparisons','original_R48_native_source_IoU_reproduced',
 'no_feedback_rule_selection_or_adaptation_after_GT','blind_identity_does_not_come_from_IoU','no_FINAL_access',
 'full_score_values_bitexact','selected_query_box_values_bitexact','all_five_source_bridges_bitexact',
 'original_V_P_R_bank_all336_bitexact','legacy_historical_all336_qbox_exact','joint_historical_all336_cue_exact',
 'full_gray_subtraction_maxerror','source_scene_mask_evidence_reference_sha_verified',
 'frames_audited','source_queries','query_mask_count_cells_exact','all_source_boxes_maskhash_frame_camera_raw_input_bytebindings_verified',
 'saved_UV_to_mask_ownership_independently_recomputed','all_unassigned_ambiguous_foreground_partitions_exact',
 'nonpositive_actual_source_queries_UNKNOWN','synthetic_nonpositive_unknown_and_threeway_partition_PASS',
 'not_independent_calibration_visibility_or_worldID_truth','seconds']
PLAN_FIELDS=['state','n','cohort','canonical_source','no_GT_no_media_no_models_no_TTA','mask_access',
 'source_exact_bridge','arm_names','factorial','no_mapped_queries','semantic_boundary','measurement_stop',
 'no_GT_or_new_model_or_adaptation','identity_schema','two_visual_questions','point_support','fraction_denominators',
 'original256_support_inventory','support_inventory_limits','scope','no_selection_or_retuning',
 'no_phase1_phase2_label_modification','no_image_pixels_rawXYZ_taskGT_or_models_in_join','old_night_deadline_extended',
 'FINAL_access','meaning_of_no_media','geometry','counts','negative_or_unknown_cases','not_box_refinement',
 'frame_reuse','identity_join','scope_and_cost','failure']

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safe_text(s):return s.replace(str(R),'WORKSPACE_ROOT').replace(str(R.parent),'LOCAL_HOME')
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csv_write(p,rows,fieldnames=None):
 assert rows or fieldnames;p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);f=io.StringIO();w=csv.DictWriter(f,fieldnames or list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);p.write_text(f.getvalue())
def scalar(v):assert v is None or type(v) in (bool,int,float);return v
def prose_or_scalars(v):
 if v is None or type(v) in (bool,int,float):return v
 if isinstance(v,str):return safe_text(v)
 if isinstance(v,list):
  assert not any(isinstance(x,(dict,list)) for x in v)
  return [prose_or_scalars(x) for x in v]
 raise AssertionError('Nested dictionaries require an explicit export projection')
def project_scalars(row,keys):return {k:scalar(row[k]) for k in keys if k in row}
def project_receipt(row):
 result=project_scalars(row,[k for k in RECEIPT_FIELDS if k!='state'])
 assert isinstance(row['state'],str) and row['state'].startswith(('PASS','COMPLETE','ROOT_'))
 result['state']=row['state'];result['private_filemaps_and_inputpaths_omitted']=True;return result
def distribution(values):
 values=[v for v in values if v is not None]
 return {'n':len(values),'min':min(values) if values else None,'median':statistics.median(values) if values else None,
         'max':max(values) if values else None,'mean':statistics.mean(values) if values else None}
def aggregate_numeric(rows,fields):
 out={}
 for part in PARTS:
  out[part]={}
  for pl in ['all']+PLATFORMS:
   rr=[x for x in rows if (part=='ALL336' or x['split']==part) and (pl=='all' or x['platform']==pl)]
   out[part][pl]={'n':len(rr),'scalar_distributions':{k:distribution([x.get(k) for x in rr]) for k in fields},
     'true_counts':{k:sum(x.get(k) is True for x in rr) for k in fields if any(type(x.get(k)) is bool for x in rr)}}
 return out

def opaque_hex_matches(text):
 """Catch standalone opaque16hex, without misclassifying numeric decimals."""
 bad=[]
 for m in re.finditer(r'(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])',text,re.I):
  token=m.group();before=text[:m.start()];after=text[m.end():]
  decimal=bool(re.search(r'\d\.$',before))
  numeric=token.isdigit() and bool(re.match(r'(?:[eE][+-]?\d+)?(?:[,\s\]\})]|$)',after))
  scientific=bool(re.fullmatch(r'\d+[eE]',token)) and bool(re.match(r'[+−-]?\d+(?:[,\s\]\})]|$)',after))
  if decimal and (numeric or scientific):continue
  bad.append(m.start())
 return bad

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=int,required=True);args=ap.parse_args();assert 1<=args.attempt<=99
 planpath=H/'experiment/PUBLIC_EXPORT_PLAN_LOCK.json';plan=read(planpath);auth=read(H/'experiment/ROOT_PUBLIC_EXPORT_AUTHORIZATION.json')
 assert auth['state']=='ROOT_AUTHORIZED_R49_POSTSCORE_PUBLIC_EXPORT'
 assert auth['program_sha256']==sha(__file__) and auth['plan_sha256']==sha(planpath)
 assert plan['prefix']==PREFIX and plan['old93_and_repo_root_unchanged'] is True
 bound=plan['required_files_sha256']
 def load(rel):
  assert rel in bound and sha(R/rel)==bound[rel],('unbound_export_input',rel)
  assert 'FINAL' not in rel and 'round46/final' not in rel
  return read(R/rel)
 for rel,h in bound.items():
  assert 'FINAL' not in rel and 'round46/final' not in rel and sha(R/rel)==h
 assert set(plan['own_code'])<=set(OWN_CODE)
 for rel in plan['own_code']:assert rel in bound
 # Explicit readiness: all labels and joins are final, never peek pending data.
 for name in ['scoring_receipt','stage_audit','ownership_audit','join_audit','saved_scalar_recheck']:
  value=load(plan[name]);assert value['state']==plan['expected_states'][name]
 score_receipt=load(plan['scoring_receipt']);assert score_receipt['task_values']==5376
 for name in ['phase1_seal','phase2_seal','join_seal']:
  assert plan[name] in bound and load(plan[name])['state']==plan['expected_states'][name]
 assert load(plan['join_seal'])['files']['ROWS_PRIVATE.json']==sha(R/plan['blind_join_rows'])
 tasks=load(plan['task_rows']);assert len(tasks)==336
 mapping={r['id_PRIVATE']:f'P{i+1:04d}' for i,r in enumerate(tasks)};assert len(mapping)==336
 scenenames={pl:sorted({r['scene_PRIVATE'] for r in tasks if r['platform']==pl}) for pl in PLATFORMS}
 scene=lambda r:r['platform']+'_scene'+str(scenenames[r['platform']].index(r['scene_PRIVATE'])+1).zfill(2)
 pub=H/'publication';pub.mkdir(exist_ok=True);out=pub/f'public_export_attempt{args.attempt:03d}';out.mkdir(exist_ok=False)
 scalars=[];negative=[];vs_source=[];vs_full=[]
 for r in tasks:
  assert set(r['IoU'])==set(r['queries'])==set(ARMS)
  base={'row':mapping[r['id_PRIVATE']],'split':r['split'],'platform':r['platform'],'scene_alias':scene(r)}
  for arm in ARMS:
   value=scalar(r['IoU'][arm]);assert 0<=value<=1
   scalars.append(dict(base,arm=arm,IoU=value,changed_vs_source=r['queries'][arm]!=r['queries'][ARMS[0]],
                      changed_vs_fullG=r['queries'][arm]!=r['queries'][ARMS[7]]))
   for ref in ARMS:
    d=value-r['IoU'][ref]
    if arm!=ref and d< -1e-10:
     n=dict(base,arm=arm,reference=ref,delta_pp=float(d*100));negative.append(n)
     if ref==ARMS[0]:vs_source.append(n)
     if ref==ARMS[7]:vs_full.append(n)
 assert len(scalars)==5376 and len(negative)==score_receipt['negative_ordered_comparisons']
 csv_write(out/'results/ALL5376_TASK_SCALARS.csv',scalars)
 csv_write(out/'results/ALL_ORDERED_NEGATIVE_COMPARISONS.csv',negative,
   ['row','split','platform','scene_alias','arm','reference','delta_pp'])
 # Empty tails remain explicit rather than silently dropping the comparison.
 dump(out/'results/ALL_NEGATIVES_VS_SOURCE.json',vs_source);dump(out/'results/ALL_NEGATIVES_VS_FULL_G.json',vs_full)
 scene_rows=[]
 for part in PARTS:
  for pl in PLATFORMS:
   for sn in scenenames[pl]:
    rr=[r for r in tasks if r['platform']==pl and r['scene_PRIVATE']==sn and (part=='ALL336' or r['split']==part)]
    if not rr:continue
    for arm in ARMS:
     scene_rows.append({'split':part,'platform':pl,'scene_alias':scene(rr[0]),'n':len(rr),'arm':arm,
                        'mIoU_percent':float(statistics.mean(r['IoU'][arm] for r in rr)*100)})
 csv_write(out/'results/ALL_SCENE_TASK_SCALARS.csv',scene_rows)
 # Closed schemas for summary data; no recursive copying of arbitrary fields.
 tasksummary=load(plan['task_summary']);publicsummary={}
 metrics=['n','mIoU_percent','Acc25_percent','Acc50_percent','gain_vs_source_pp','improved','worsened','query_changes','scene_clusters']
 for part in PARTS:
  publicsummary[part]={pl:{arm:project_scalars(tasksummary[part][pl][arm],metrics) for arm in ARMS} for pl in PLATFORMS}
  publicsummary[part]['balanced_mIoU_percent']={arm:scalar(tasksummary[part]['balanced_mIoU_percent'][arm]) for arm in ARMS}
 dump(out/'results/TASK_SUMMARY.json',publicsummary)
 structure=load(plan['structure_rows']);ranges=load(plan['range_rows']);assert len(structure)==len(ranges)==336
 projected=[];rp=[]
 for i,(s,g) in enumerate(zip(structure,ranges)):
  assert s['split']==g['split']==tasks[i]['split'] and s['platform']==g['platform']==tasks[i]['platform']
  base={'row':mapping[tasks[i]['id_PRIVATE']],'split':s['split'],'platform':s['platform']}
  projected.append(dict(base,**project_scalars(s,STRUCT_FIELDS)))
  p=dict(base,**project_scalars(g,RANGE_FLAGS))
  for term in ['source','weighted2V','weighted2P','weighted2R']:
   for kind in ['all256_ptp','mapped_ptp']:p[term+'_'+kind]=scalar(g['ranges'][term][kind])
  for witness,keys in [('source_plusV_minus_rawcue_witness',['source_delta','weighted2V_delta','rawcue_delta','source_plusV_score_delta']),
      ('fullG_minus_rawcue_witness',['source_delta','weighted2V_delta','weighted2P_delta','weighted2R_delta','fullG_score_delta'])]:
   for key in keys:p[witness+'_'+key]=scalar(g[witness][key])
  rp.append(p)
 dump(out/'results/ALL336_STRUCTURE_SCALARS.json',projected);dump(out/'results/STRUCTURE_AGGREGATE.json',aggregate_numeric(projected,STRUCT_FIELDS))
 dump(out/'results/ALL336_RANGE_SOURCE_SCALARS.json',rp)
 dump(out/'results/RANGE_SOURCE_AGGREGATE.json',aggregate_numeric(rp,[k for k in rp[0] if k not in ('row','split','platform')]))
 blind=load(plan['blind_join_rows']);assert len(blind)==48
 bs=[];points=[];bi=[];tie_rows=[]
 for i,r in enumerate(blind):
  base={'blind_row':f'B{i+1:03d}','task_row':mapping[r['id_PRIVATE']],'platform':r['platform']}
  b=dict(base,**project_scalars(r,BLIND_SCALARS))
  assert r['partial_compatibility_topset_interpretation'] in {'UNKNOWN_no_visual_evidence_pool',
   'ORDERING_CHECK_WITH_CONFIRMED_COMPATIBLE_ALTERNATIVE',
   'NO_CONFIRMED_COMPATIBLE_ALTERNATIVE_FORCED_ARGMAX_NOT_BINARY_RECOGNITION_ERROR'}
  b['partial_compatibility_topset_interpretation']=r['partial_compatibility_topset_interpretation']
  for k in BLIND_STATUS_FIELDS:
   choices=PHASE1_STATUSES if k=='phase1_status' else POOL_STATUSES if k=='legacy_pool_identity_coverage_status' else VISUAL_STATUSES
   assert r[k] in choices;b[k]=r[k]
  for key in ['fullcaption_target_display_ids','actual_partial_subject_compatible_ids','actual_partial_subject_UNKNOWN_ids',
              'legacy_image_top_ids','legacy_raw_full_image_top_ids','joint_image_top_ids']:b[key.replace('_ids','_count')]=len(r[key])
  for phase in ('phase1','phase2'):
   name=r[phase+'_reviewer'];b[phase+'_reviewer']='assistant_A' if name in ('root','/root') else 'assistant_B' if name in ('influence','influence_audit','/root/influence_audit') else 'assistant_reviewer_other'
  bs.append(b)
  assert set(r['arms'])==set(ARMS)
  for arm in ARMS:
   p=r['arms'][arm];assert p['status'] in POINT_STATUSES
   points.append(dict(base,arm=arm,status=p['status'],**project_scalars(p,POINT_FIELDS)))
  z=r['original256_target_support_inventory'];assert z['status'] in {'DEFINED_SINGLE_TARGET_SUPPORT_INVENTORY'}|{'UNKNOWN_fullcaption_'+x for x in PHASE1_STATUSES}
  v=dict(base,status=z['status'],**project_scalars(z,INVENTORY_FIELDS))
  if z['all256_support_status_counts'] is not None:assert set(z['all256_support_status_counts'])<=POINT_STATUSES
  v['all256_support_status_counts']=None if z['all256_support_status_counts'] is None else {k:int(n) for k,n in z['all256_support_status_counts'].items()}
  if v['all256_support_status_counts'] is not None:assert sum(v['all256_support_status_counts'].values())==256
  bi.append(v)
  for name in ('rawcue','postclipcue'):
   t=r['cue_maxset_support_diagnostics'][name];assert set(t['maxset_support_status_counts'])<=POINT_STATUSES
   assert t['representative_support_status'] in POINT_STATUSES
   keys=['maxset_size','no_visual_maxset','contains_target_supported','target_supported_members_count',
    'representative_is_in_maxset','representative_is_target_supported_member','maxset_mixed_support_statuses',
    'source_fallback_due_empty_maxset','representative_failure_does_not_certify_entire_W_maxset_wrong']
   tie_rows.append(dict(base,cue=name,representative_support_status=t['representative_support_status'],
     maxset_support_status_counts={k:int(n) for k,n in t['maxset_support_status_counts'].items()},**project_scalars(t,keys)))
 assert len(points)==768
 dump(out/'results/ALL48_BLIND_STATUS_SCALARS.json',bs);dump(out/'results/ALL768_BLIND_POINT_SUPPORT_SCALARS.json',points)
 dump(out/'results/ALL48_ORIGINAL256_SUPPORT_INVENTORY_SCALARS.json',bi)
 assert len(tie_rows)==96;dump(out/'results/ALL96_CUE_MAXSET_SUPPORT_SCALARS.json',tie_rows)
 for name in ['scoring_receipt','stage_audit','ownership_audit','join_audit','saved_scalar_recheck']:
  dump(out/'audit'/(name.upper()+'.json'),project_receipt(load(plan[name])))
 for rel in plan['own_code']:
  dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(safe_text((R/rel).read_text()))
 for key,dest in [('report','REPORT.md'),('pipeline_report','PIPELINE_AUDIT.md')]:
  rel=plan[key];assert rel in bound;(out/dest).write_text(safe_text((R/rel).read_text()))
 decision=load(plan['decision']);dump(out/'DECISION.json',{k:prose_or_scalars(decision[k]) for k in ['state','decision','scope','limitations','method_adopted','no_FINAL_access'] if k in decision})
 for name,rel in plan['locked_plans'].items():
  assert re.fullmatch(r'[A-Z0-9_]+',name)
  raw=load(rel);data={k:prose_or_scalars(raw[k]) for k in PLAN_FIELDS if k in raw}
  data['original_private_plan_sha256']=sha(R/rel);data['private_inputmaps_not_redistributed']=True
  dump(out/'protocol'/(name+'.json'),data)
 costs=[]
 for i,rel in enumerate(plan['manual_job_receipts']):
  r=load(rel);start=datetime.datetime.fromisoformat(r['started_utc']);end=datetime.datetime.fromisoformat(r['finished_utc'])
  assert r['resource'] in ('cpu','network')
  basename=Path(r['command'][1]).name
  public_program=basename if basename in {Path(x).name for x in OWN_CODE}|{'blind_rgb49_draft.py'} else 'other_locked_manual_program'
  costs.append({'job':f'J{i+1:03d}','program':public_program,'resource':r['resource'],'seconds':(end-start).total_seconds(),
                'returncode':r['returncode'],'timed_out':r['timed_out'],'manual_human_P1':True,'old_night_deadline_not_extended':True})
 dump(out/'results/COSTS.json',{'manual_jobs':costs,'sum_job_seconds_not_single_sample_latency':sum(x['seconds'] for x in costs),
     'new_grounder_or_VLM_forwards':0,'GPU_calls':0,'optimizer_steps':0,'R15_raw_point_geometry_is_CPU_only':True,'cache_replay_cost_is_not_online_G_end_to_end_latency':True,'assistant_image_review_and_future_export_network_verification_not_in_this_CPU_sum':True})
 dump(out/'history/TECHNICAL_ATTEMPT_SUMMARY.json',{'failed_jobs':[x for x in costs if x['returncode']!=0],
     'all_completed_registered_jobs_at_preexport_freeze_included':True,'original_attempts_preserved_privately':True,
     'raw_logs_commands_filemaps_case_notes_and_annotation_scripts_not_redistributed':True})
 if plan.get('technical_failure_summary'):
  failures=load(plan['technical_failure_summary']);assert isinstance(failures,list)
  allowed=['failure_type','phase','calls_before_failure','repair','scientific_outputs_changed','original_retained']
  dump(out/'history/TECHNICAL_FAILURE_TYPES.json',[{k:prose_or_scalars(x[k]) for k in allowed if k in x} for x in failures])
 dump(out/'README.json',{'prefix':PREFIX,'scope':'old exposed336,descriptive P1,noTTA,noFINAL,new manual authorization after closed night',
   'task_values':5376,'blind_rows':48,'blind_method_point_rows':768,'all_ordered_negative_comparisons':len(negative),
   'two_assistant_nonexpert_review_not_gold':True,'point_support_not_calibration_or_complete_object_identity_gold':True,
   'old_93_studies_and_repository_root_untouched':True,'local_bundle_only_not_published':True})
 # Privacy inspection uses only exposed old336 and just-joined private aliases.
 # It never opens the closed final cohort, and exact-token checks have no numeric exception.
 oldrel='research/round24/input/ALL336_PRIVATE.json';old=load(oldrel)
 forbidden={str(r[k]) for r in old for k in ['id','alias','scene','caption','imagepath','pointpath','metapath'] if r.get(k)}
 forbidden|={r['review_alias'] for r in blind};findings=[]
 blocked={'id','id_PRIVATE','alias','review_alias','scene','scene_PRIVATE','caption','subject_query','notes','target_display_ids',
  'mask_index_PRIVATE','mask_sha256','query_PRIVATE','queries','query_indices','scores','arm_scores','source_boxes','boxes','weights',
  'token_mask','tokens','query','h','x','text_h','imagepath','pointpath','metapath','input_sha256','files','inputs_sha256'}
 def walk(value,file):
  if isinstance(value,dict):
   for k,v in value.items():
    if k in blocked:findings.append({'file':file,'kind':'forbidden_data_key','key':k})
    walk(v,file)
  elif isinstance(value,list):
   for v in value:walk(v,file)
 for p in out.rglob('*'):
  if not p.is_file():continue
  text=p.read_text();name=p.relative_to(out).as_posix()
  for token in forbidden:
   if token in text:findings.append({'file':name,'kind':'exact_old_private_token','token_sha256':hashlib.sha256(token.encode()).hexdigest()})
  if '/'+'home/' in text or '/'+'mnt/' in text:findings.append({'file':name,'kind':'local_absolute_path'})
  if opaque_hex_matches(text):findings.append({'file':name,'kind':'standalone_opaque16hex'})
  if p.suffix=='.json':walk(read(p),name)
 assert not findings,findings[:10]
 dump(out/'EXPORT_PRIVACY_AUDIT.json',{'state':'PASS_EXPLICIT_ALLOWLIST_ANONYMOUS_SCALARS_ONLY',
     'old336_and_blind_alias_exact_token_scan':True,'no_FINAL_access':True,'findings':[],
     'all_unknowns_and_negative_tails_retained':True,'raw_annotations_notes_filemaps_vectors_indices_media_weights_thirdparty_not_copied':True})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()]
 dump(out/'MANIFEST.json',{'prefix':PREFIX,'manifest_excludes_itself':True,'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];payload=pub/f'TREE_PAYLOAD_LOCAL_attempt{args.attempt:03d}.json'
 dump(payload,[{'path':PREFIX+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files])
 print(json.dumps({'state':'LOCAL_R49_PUBLIC_EXPORT_READY_NOT_PUBLISHED','files':len(files),'prefix':PREFIX,
    'payload_sha256':sha(payload),'task_values':5376,'blind_rows':48,'point_rows':768,'negative_comparisons':len(negative)}))

if __name__=='__main__':main()
