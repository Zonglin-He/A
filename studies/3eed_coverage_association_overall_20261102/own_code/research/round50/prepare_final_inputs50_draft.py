"""UNEXECUTED saved-only cost/history/private-token input preparation.

No models, media/XYZ, prediction arrays, task GT containers, TTA closing,
export, network or experiment mutation. Root must separately pin this script
and every scalar/registration input. The private token registry is not public.
"""
from pathlib import Path
import argparse,json,hashlib,datetime,math,re,collections
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round50'
FULL='crossplatform/round50/full5912_attempt001'
COST='research/round50/independent_review/full_evidence_cost_audit_attempt001'
SAVED_INPUTS={
 'full_registration':'research/round8/experiment/full5912_gpu_inputs.json',
 'dev_registration':'research/round24/input/ALL336_PRIVATE.json',
 'source_provenance':'research/round50/experiment/SOURCE_PROVENANCE_RECHECK_NO_MODEL.json',
 'dev_prep_counts':'crossplatform/round50/prepared336_attempt001/COUNTS.json',
 'dev_visual_counts':'crossplatform/round50/visual336_attempt001/COUNTS.json',
 'dev_visual_seal':'crossplatform/round50/visual336_attempt001/SEALED.json',
 'visual_load_failure':'research/round50/experiment/VISUAL_LOAD_FAILURE_RECEIPT.json',
 'visual_runtime_addendum':'research/round50/independent_review/VISUAL_RUNTIME_STATIC_ADDENDUM.json',
 'layout_failure':'research/round50/independent_review/prediction_audit_attempt001/FAILED.json',
 'layout_repair':'research/round50/independent_review/prediction_audit_preexecution_revision002/REPAIR_CONTRACT.json',
 'text_prelock_failure':'research/round50/experiment/FULL_TEXT_PRELOCK_FAILURE_RECOVERY.json',
 'text_udapi_failure':FULL+'/text_import_failure_attempt002/FAILED_all.json',
 'text_import313':'research/round50/experiment/FULL_TEXT_RUNTIME_IMPORT_SMOKE.json',
 'text_import311':'research/round50/experiment/FULL_TEXT_RUNTIME_IMPORT_SMOKE_PY311.json',
 'frames_seal':FULL+'/frames/SEALED.json','text_seal':FULL+'/text/SEALED.json',
 'geometry_seal':FULL+'/prepared/SEALED.json','source_adapter':FULL+'/source_chunks/ALL_CHUNKS_SEALED.json',
 'cost_summary':COST+'/SUMMARY.json','cost_chunk_rows':COST+'/ALL_CHUNK_COSTS.json',
 'cost_audit':COST+'/AUDIT_RESULT.json','cost_seal':COST+'/SEALED.json',
 'eng_capture_seal':'crossplatform/round50/episodic_engineering2_capture_attempt001/SEALED.json',
 'eng_fit_seal':'crossplatform/round50/episodic_engineering2_fit_attempt001/SEALED.json',
 'tta_capture_seal':'crossplatform/round50/episodic336_capture_attempt001/SEALED.json',
 'tta_fit_seal':'crossplatform/round50/episodic336_fit_attempt001/SEALED.json',
 'tta_closure_seal':'crossplatform/round50/episodic336_closure_attempt001/SEALED.json',
 'tta_audit_failure':'research/round50/independent_review/episodic336_audit_attempt001/FAILED.json',
 'tta_diag_startup_log':'research/round50/experiment/EPISODIC_PROJECTION_DTYPE_DIAGNOSTIC_ATTEMPT001.log',
 'tta_diag_startup_plan':'research/round50/experiment/EPISODIC_PROJECTION_DTYPE_DIAGNOSTIC_PLAN_LOCK.json',
 'tta_dtype_diagnostic':'research/round50/independent_review/episodic_projection_dtype_diagnostic_attempt002/DIAGNOSTIC.json',
 'tta_dtype_diag_seal':'research/round50/independent_review/episodic_projection_dtype_diagnostic_attempt002/SEALED.json',
 'tta_repaired_audit':'research/round50/independent_review/episodic336_audit_attempt002/AUDIT_RESULT.json',
 'tta_repaired_audit_seal':'research/round50/independent_review/episodic336_audit_attempt002/SEALED.json'}
CHUNKS={
 'frames':[FULL+f'/frames/chunks_PRIVATE/{j:04d}.json' for j in range(47)],
 'text':[FULL+'/text/chunks_PRIVATE/all.json'],
 'geometry':[FULL+f'/prepared/chunks_PRIVATE/{j:04d}.json' for j in range(47)]}
GUARDS={
 'FULL_SOURCE_ADAPTER':('research/round50/job_receipts/20261102T051352_1875950.json','crossplatform/round50/full_source_batch50.py'),
 'FULL_REMAINING_FRAME_BATCH':('research/round50/job_receipts/20261102T051525_1876396.json','crossplatform/round50/full_frames_batch50.py'),
 'FULL_TEXT_SOURCE_PY311':('research/round50/job_receipts/20261102T052015_1878876.json','crossplatform/round50/text_bootstrap50.py'),
 'FULL_REMAINING_GEOMETRY_BATCH':('research/round50/job_receipts/20261102T052909_1881588.json','crossplatform/round50/full_geometry_batch50.py')}
HISTORY_KEYS={'stage','attempt','status','failure_type','repair_type','valid_outputs','started_attempts',
 'entered_forward_attempts','completed_calls','source_or_prompt_or_method_changed','GT_read','prior_attempt_preserved',
 'reused_raw_outputs','new_parsed_text_rows','reused_text_graphs','physical_Stanza_neural_forwards_not_instrumented','measured_seconds'}
def local(rel):
 p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
 assert 'FINAL384' not in rel and 'round46/final' not in rel
 assert p.suffix in ('.json','.log'), 'Only explicitly registered saved metadata/registrations'
 return ROOT/p
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump_new(p,x):
 p=Path(p);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp');assert not tmp.exists();tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def number(x):assert type(x) in (int,float) and math.isfinite(x) and x>=0;return x
def integer(x):assert type(x) is int and x>=0;return x
def make_registry(full,dev):
 assert len(full)==5912 and len(dev)==336
 assert len({r['id'] for r in full})==5912 and len({r['id'] for r in dev})==336
 assert collections.Counter(r['platform'] for r in full)=={'drone':2984,'quad':2928}
 tokens=set()
 for rows in (full,dev):
  for r in rows:
   for kind,key in [('id','id'),('scene','scene'),('caption','caption'),('media_path','imagepath'),('media_path','pointpath'),('media_path','metapath')]:
    value=r[key];assert isinstance(value,str) and value;tokens.add((kind,value))
   for key in ('alias','frame_alias'):
    if key in r and r[key] is not None:
     assert isinstance(r[key],str) and r[key];tokens.add(('alias',r[key]))
 return {'tokens':[{'kind':k,'value':v} for k,v in sorted(tokens)],'coverage':{
  'full5912_ids':5912,'dev336_ids':336,'full_caption_records':5912,'full_media_path_records':17736,
  'dev_caption_records':336,'dev_media_path_records':1008,'no_protected_FINAL_specific_artifact_read':True,
  'no_media_or_GT_container_or_prediction_array_opened':True}}
def history_event(stage,attempt,status,failure_type,repair_type,**kw):
 row={'stage':stage,'attempt':attempt,'status':status,'failure_type':failure_type,'repair_type':repair_type,
  'source_or_prompt_or_method_changed':False,'GT_read':False,'prior_attempt_preserved':True,**kw}
 assert set(row)<=HISTORY_KEYS
 for k,v in row.items():
  if isinstance(v,str):assert re.fullmatch(r'[A-Za-z0-9_]{1,96}',v)
  else:assert v is None or type(v) in (bool,int,float)
 return row
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('plan','authorization'):p.add_argument('--'+key,required=True)
 args=p.parse_args();plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R50_SAVED_FINAL_INPUT_PREPARATION_ONLY'
 assert auth['program_sha256']==sha(__file__) and auth['plan_sha256']==sha(args.plan)
 assert plan['no_models_media_GT_or_prediction_arrays'] is True and plan['no_TTA_close_export_or_network'] is True
 assert plan['saved_inputs']==SAVED_INPUTS and plan['chunk_receipts']==CHUNKS
 assert plan['guard_receipts']=={k:v[0] for k,v in GUARDS.items()}
 expected_inputs=set(SAVED_INPUTS.values())|{p for paths in CHUNKS.values() for p in paths}|{v[0] for v in GUARDS.values()}
 assert set(plan['inputs_sha256'])==expected_inputs,'No unlisted raw data, task result, or arbitrary metadata input'
 consumed={}
 def get(rel):
  assert rel in plan['inputs_sha256'];fp=local(rel);h=sha(fp);assert h==plan['inputs_sha256'][rel];consumed[rel]=h
  return read(fp)
 def role(name):return get(SAVED_INPUTS[name])
 # Validate every listed input, including raw diagnostic log by hash only.
 for rel,h in plan['inputs_sha256'].items():assert sha(local(rel))==h
 source=role('source_provenance')
 assert source['state']=='PASS_R50_SAVED_SOURCE_TRAINING_REGISTRATION_AND_COMPLETION_BINDING_NO_MODEL_OR_DATA'
 assert source['saved_contract_fixed_epoch100'] and source['saved_contract_generic_RoBERTa_random_visual_grounding_no_jointtask_initialization'] and source['saved_completion_source_only_epoch100']
 counts={'full_rows':5912,'DEV_rows':336,'historical_source_epoch':100,'historical_source_training_seconds_recounted':None}
 times={};flags={'three_full_arms_share_one_collection':True,'stage_walls_overlap_and_are_not_per_method_latencies':True,
  'load_and_physical_call_times_overlap_chunk_wall':True,'cached_source_zero_incremental_forward_is_not_zero_end_to_end_cost':True,
  'historical_source_dataset_weights_and_training_not_reloaded_or_reaudited':True,
  'no_models_media_GT_prediction_arrays_network_or_TTA_closing':True,
  'physical_Stanza_neural_forwards_not_instrumented':True}
 ds=role('dev_visual_seal');dv=role('dev_visual_counts');assert ds['files']['COUNTS.json']==consumed[SAVED_INPUTS['dev_visual_counts']]
 assert dv['n']==336 and dv['total_new_input_numeric_calls']==4956 and dv['model_unchanged'] and dv['generated_tokens']==dv['optimizer_steps']==0
 counts.update(DEV_new_completed_numeric_forwards=integer(dv['total_new_input_numeric_calls']),DEV_new_numeric_this_process=integer(dv['new_numeric_calls']),
  DEV_resumed_completed_numeric_forwards=integer(dv['resumed_numeric_calls']),DEV_old_crop_views_reused=integer(dv['old_crop_views_reused']),
  DEV_extra_rejected_started_numeric_attempts=1,DEV_peak_bytes=integer(dv['peak_bytes']))
 times.update(DEV_visual_wall_seconds=number(dv['seconds']),DEV_visual_load_seconds_overlaps_wall=number(dv['load_seconds']))
 prepcounts=role('dev_prep_counts');assert prepcounts['n']==336 and prepcounts['new_numeric_forward_upper_bound']==4956
 counts['DEV_additional_mask_inventory']=integer(prepcounts['additional_masks'])
 src=role('source_adapter');assert src['n']==5912 and src['new_model_calls']==0 and src['new_features_available'] is False
 counts['full_source_adapter_new_grounder_forwards']=0
 seals={s:role(s+'_seal') for s in ('frames','text','geometry')}
 for stage in ('frames','text','geometry'):
  cc=[get(rel) for rel in CHUNKS[stage]];folder=FULL+('/prepared' if stage=='geometry' else '/'+stage)
  for rel in CHUNKS[stage]:assert seals[stage]['files'][Path(rel).relative_to(folder).as_posix()]==consumed[rel]
  assert sum(integer(c['n']) for c in cc)==5912
  times['full_'+stage+'_sum_chunk_wall_seconds']=math.fsum(number(c['seconds']) for c in cc)
  counts['full_'+stage+'_chunks']=len(cc)
  if stage=='frames':
   counts['full_new_MaskRCNN_forwards']=sum(integer(c['new_calls']) for c in cc);counts['full_reused_registered_masks']=sum(integer(c['reused']) for c in cc)
   assert counts['full_new_MaskRCNN_forwards']==5219 and counts['full_reused_registered_masks']==693
  if stage=='text':
   reused=sum(integer(c['exact_old_text_reused']) for c in cc);assert reused==336
   counts.update(full_reused_text_graphs=reused,full_new_parsed_text_rows=5912-reused,full_Stanza_physical_neural_forward_count=None)
 cseal=role('cost_seal');summary=role('cost_summary');chunks=role('cost_chunk_rows');ca=role('cost_audit')
 for name,rel in [('SUMMARY.json','cost_summary'),('ALL_CHUNK_COSTS.json','cost_chunk_rows'),('AUDIT_RESULT.json','cost_audit')]:assert cseal['files'][name]==consumed[SAVED_INPUTS[rel]]
 assert ca['state']=='PASS_R50_INDEPENDENT_FULL_FROZEN_EVIDENCE_CALL_COST_AND_RAW_ARITHMETIC_AUDIT_NO_GT' and ca['n']==5912
 for stage in ('dino','numeric'):
  ss=summary[stage];rr=[r for r in chunks if r['stage']==stage];assert sum(r['cases'] for r in rr)==5912
  for raw in ('new_input_forwards_all_processes_of_this_chunk','logical_requests','historical_exact_cache_logical_uses','historical_whole_case_logical_uses','nonhistorical_logical_uses','chunks'):
   counts['full_'+stage+'_'+raw]=integer(ss[raw])
  assert sum(r['physical_completed_forwards'] for r in rr)==ss['new_input_forwards_all_processes_of_this_chunk']
  for raw in ('sum_chunk_ledger_wall_seconds','sum_physical_call_record_seconds_overlaps_wall'):times['full_'+stage+'_'+raw]=number(ss[raw])
  counts['full_'+stage+'_max_process_peak_bytes']=integer(ss['max_process_peak_bytes'])
 counts['full_old_subject_mask_logical_inventory']=integer(summary['numeric']['old_subject_logical_budget'])//4
 counts['full_wide_subject_mask_logical_inventory']=integer(summary['numeric']['wide_subject_logical_budget'])//4
 # These are separately sealed source/fit/closure scalar headers, no states or
 # per-case predictions/quality values. They do not determine a TTA decision.
 tta={}
 for role_name,prefix in [('eng_capture_seal','TTA_engineering_capture'),('eng_fit_seal','TTA_engineering_fit'),('tta_capture_seal','TTA_DEV_capture'),('tta_fit_seal','TTA_DEV_fit'),('tta_closure_seal','TTA_installed_closure')]:
  x=role(role_name);tta[prefix]=x;counts[prefix+'_rows']=integer(x['n'])
  if 'grounder_forwards' in x:counts[prefix+'_grounder_forwards']=integer(x['grounder_forwards'])
  if 'actual_optimizer_steps' in x:counts[prefix+'_optimizer_steps']=integer(x['actual_optimizer_steps'])
  if 'seconds' in x:times[prefix+'_wall_seconds']=number(x['seconds'])
 assert tta['TTA_engineering_capture']['grounder_forwards']==4 and tta['TTA_DEV_capture']['grounder_forwards']==672
 assert tta['TTA_DEV_fit']['source_grounder_forwards_in_this_stage']==tta['TTA_DEV_fit']['new_visual_calls']==0
 assert tta['TTA_installed_closure']['grounder_forwards']<=8
 counts['TTA_new_full_grounder_forwards_including_engineering_and_closure']=sum(tta[p]['grounder_forwards'] for p in ('TTA_engineering_capture','TTA_DEV_capture','TTA_installed_closure'))
 flags['TTA_all_six_arms_share_captured_source_views_and_fixed_G']=True
 for name,(path,worker) in GUARDS.items():
  g=get(path);assert g['returncode']==0 and g['timed_out'] is False
  assert any(v in (worker,str(ROOT/worker)) for v in g['command'])
  elapsed=(datetime.datetime.fromisoformat(g['finished_utc'])-datetime.datetime.fromisoformat(g['started_utc'])).total_seconds()
  times['guard_'+name.lower()+'_wall_seconds']=number(elapsed)
 flags['remaining_batch_guard_walls_do_not_cover_first_frame_or_geometry_chunk']=True
 loadfail=role('visual_load_failure');add=role('visual_runtime_addendum');failed=add['failed_attempts_preserved']
 assert loadfail['numeric_calls_started']==0 and loadfail['first_attempt_preserved'] and len(failed)==2
 assert failed[0]['numeric_started_records']==0 and failed[1]['numeric_started_records']==1 and all(f['completed_valid_numeric_outputs']==0 for f in failed)
 assert add['no_prompt_crop_modelweights_formula_or_GT_selection_changed'] and add['archived_rejected_attempt_is_additional_not_erased_or_counted_as_valid_numeric_output']
 lf=role('layout_failure');lr=role('layout_repair');assert lr['tolerance_weakened'] is False and lr['author_predictions_or_method_changed'] is False and lr['GT_read_or_score_run'] is False
 pre=role('text_prelock_failure');ud=role('text_udapi_failure');sm313=role('text_import313');sm311=role('text_import311')
 assert pre['model_calls']==pre['fulltextcases_created']==0 and ud['completed_n']==0 and ud['error_type']=='ModuleNotFoundError'
 assert sm313['state']=='FAIL_R50_TEXT_DEPENDENCY_IMPORT_ONLY' and sm313['model_pipeline_constructed'] is False
 assert sm311['state']=='PASS_R50_TEXT_DEPENDENCY_IMPORT_ONLY' and sm311['model_pipeline_constructed'] is False
 tf=role('tta_audit_failure');diag=role('tta_dtype_diagnostic');diag_seal=role('tta_dtype_diag_seal');repair=role('tta_repaired_audit');repair_seal=role('tta_repaired_audit_seal')
 assert tf['completed_n']==0 and tf['no_GT_optimizer_or_automatic_retry'] is True
 assert diag_seal['files']['DIAGNOSTIC.json']==consumed[SAVED_INPUTS['tta_dtype_diagnostic']]
 assert diag['state']=='PASS_R50_FIRSTROW_EPISODIC_IoU_FLOAT64_CAST_CAUSE_DIAGNOSTIC_NO_GT'
 assert diag['float64_cast_only_recovers_all_stored_projection_IoU_support_coverage_and_W_bytes'] is True and diag['no_comparison_tolerance_weakened'] is True
 assert repair_seal['files']['AUDIT_RESULT.json']==consumed[SAVED_INPUTS['tta_repaired_audit']]
 assert repair['state']=='PASS_R50_INDEPENDENT_EPISODIC_NUMERIC_AND_CONTRACT_AUDIT_NO_GT' and repair['no_GT_read'] is True
 dp=role('tta_diag_startup_plan');assert dp['models_optimizers_GT_or_rescoring_allowed'] is False
 log=SAVED_INPUTS['tta_diag_startup_log'];assert log in plan['inputs_sha256'];consumed[log]=sha(local(log))
 found=False
 with local(log).open() as f:
  for line in f:
   if any(line.rstrip().startswith(v) for v in ("NameError: name 'sha' is not defined",'NameError: name "sha" is not defined')):found=True
 assert found,'Root-pinned diagnostic startup failure must match its declared enum'
 common=dict(valid_outputs=0,completed_calls=0,prior_attempt_preserved=True)
 events=[
  history_event('DEV_VISUAL',1,'TECHNICAL_FAILURE_BEFORE_VALID_NUMERIC','MODEL_LOAD_MISSING_ACCELERATE','PINNED_SEMANTIC_RUNTIME_RESTORED',started_attempts=0,entered_forward_attempts=0,**common),
  history_event('DEV_VISUAL',2,'TECHNICAL_SIGNATURE_REJECTION','FORWARD_SIGNATURE_REJECTED_LOGITS_TO_KEEP','PINNED_SEMANTIC_RUNTIME_RESTORED',started_attempts=1,entered_forward_attempts=None,**common),
  history_event('DEV_INDEPENDENT_AUDIT',1,'TECHNICAL_FAILURE_BEFORE_GT','C_F_BLAS_LAYOUT_REDUCTION_MISMATCH','RECONSTRUCT_ORIGINAL_ALLOCATION_EXACT_NOT_TOLERANCE',started_attempts=1,entered_forward_attempts=0,**common),
  history_event('FULL_TEXT',1,'PRELOCK_OR_AUTHORIZATION_STARTUP_FAILURE','WRONG_ASSET_RECEIPT_OR_MISSING_AUTH_PATH','CORRECT_REGISTERED_PATHS_BEFORE_VALID_RUN',entered_forward_attempts=0,**common),
  history_event('FULL_TEXT',2,'IMPORT_FAILURE_BEFORE_STANZA_PIPELINE','MISSING_UDAPI_DEPENDENCY','EXACT_PUREPY_DEPENDENCY_OVERLAY',entered_forward_attempts=0,**common),
  history_event('FULL_TEXT_IMPORT_SMOKE',1,'IMPORT_FAILURE_NO_NEURAL_FORWARD','ORJSON_CPYTHON_ABI_MISMATCH','RESTORE_ORIGINAL_CPYTHON311_RUNTIME',entered_forward_attempts=0,measured_seconds=number(sm313['seconds']),**common),
  history_event('FULL_TEXT_IMPORT_SMOKE',2,'SOFTWARE_IMPORT_CHECK_PASS','NONE','RESTORE_ORIGINAL_CPYTHON311_RUNTIME',valid_outputs=1,completed_calls=0,entered_forward_attempts=0,measured_seconds=number(sm311['seconds'])),
  history_event('FULL_TEXT',3,'COMPLETED_SAVED_GRAPH_PARSE','NONE','ORIGINAL_PARSER_AND_ASSETS_UNCHANGED',valid_outputs=5912,completed_calls=None,new_parsed_text_rows=counts['full_new_parsed_text_rows'],reused_text_graphs=336,physical_Stanza_neural_forwards_not_instrumented=True,measured_seconds=times['full_text_sum_chunk_wall_seconds']),
  history_event('TTA_INDEPENDENT_GEOMETRY_AUDIT',1,'TECHNICAL_FAILURE_BEFORE_GT','DETECTOR_RECT_AREA_FLOAT32_PROMOTION_OMITTED','FLOAT64_CAST_RESTORED_IN_NEW_AUDITOR_ONLY',started_attempts=1,entered_forward_attempts=None,**common),
  history_event('TTA_PROJECTION_DTYPE_DIAGNOSTIC',1,'DIAGNOSTIC_STARTUP_FAILURE','AST_NAMESPACE_MISSING_SHA','BIND_EXISTING_SHA_AND_NEW_ATTEMPT_PATHS',started_attempts=1,entered_forward_attempts=0,**common),
  history_event('TTA_PROJECTION_DTYPE_DIAGNOSTIC',2,'CAUSE_ISOLATION_PASS_NOT_TASK_RESULT','NONE','FLOAT64_CAST_ONLY_REPRODUCES_ALL_STORED_GEOMETRY',valid_outputs=1,completed_calls=0,entered_forward_attempts=0,measured_seconds=number(diag['seconds'])),
  history_event('TTA_INDEPENDENT_GEOMETRY_AUDIT',2,'COMPLETED_INDEPENDENT_REPLAY_NO_GT','NONE','FLOAT64_CAST_AND_NEW_AUDITOR_PATHS_ONLY',valid_outputs=1,completed_calls=None,entered_forward_attempts=None,measured_seconds=number(repair['seconds']))]
 counts['technical_failure_events_retained']=sum(r['status'] not in {'SOFTWARE_IMPORT_CHECK_PASS','COMPLETED_SAVED_GRAPH_PARSE','CAUSE_ISOLATION_PASS_NOT_TASK_RESULT','COMPLETED_INDEPENDENT_REPLAY_NO_GT'} for r in events)
 flags['technical_repairs_did_not_change_source_prompt_method_predictions_or_tolerances']=True
 # Do not project repair outcome metrics/update counts into a TTA adoption.
 registry=make_registry(role('full_registration'),role('dev_registration'))
 costs={'counts':counts,'timing_seconds':times,'flags':flags}
 assert set(costs)=={'counts','timing_seconds','flags'}
 for group in costs.values():
  for value in group.values():assert value is None or type(value) in (bool,int,float)
 relout=Path(plan['output_root']);assert not relout.is_absolute() and '..' not in relout.parts
 out=(ROOT/relout).resolve();assert out.is_relative_to(H) and not out.exists();out.mkdir(parents=True)
 dump_new(out/'NORMALIZED_COLLECTION_COSTS.json',costs);dump_new(out/'NORMALIZED_TECHNICAL_HISTORY.json',{'events':events})
 dump_new(out/'PRIVATE_PRIVACY_TOKEN_REGISTRY.json',registry)
 dump_new(out/'PREPARATION_RECEIPT_PRIVATE.json',{'state':'PASS_R50_SAVED_FINAL_INPUTS_PREPARED_NOT_EXPORTED',
  'program_sha256':sha(__file__),'plan_sha256':sha(args.plan),'inputs_sha256':consumed,'history_events':len(events),
  'no_GT_metrics_or_prediction_arrays_media_models_or_network_access':True,'no_TTA_closed_or_finalexport_written':True})
 dump_new(out/'SEALED.json',{'state':'R50_SAVED_FINAL_INPUT_PREPARATION_SEALED_PRIVATE_REGISTRY_NOT_PUBLIC',
  'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
 print(json.dumps({'state':'R50_SAVED_FINAL_INPUT_PREPARATION_COMPLETE_NOT_EXPORTED','history_events':len(events),'private_registry_unique_tokens':len(registry['tokens'])}))
if __name__=='__main__':main()
