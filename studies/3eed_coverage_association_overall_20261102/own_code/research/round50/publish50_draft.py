"""UNEXECUTED local whitelist export; no GitHub/network/model/GT operations.

Only root-pinned saved task scalars, anonymized audit/history projections,
reports and explicitly listed own code are published. Publication/CAS and
preserving the current remote tree are exclusively the root agent's work.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,io,json,re

ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round50';E=H/'experiment'
PREFIX='studies/3eed_coverage_association_overall_20261102'
PLAN=E/'FINAL_REPORT_EXPORT_PLAN_LOCK.json';AUTH=E/'ROOT_LOCAL_PUBLIC_EXPORT_AUTHORIZATION.json'
ARMS=['source','old_G','wideentry_oldW']
TTA_ARMS=['source','literal_A_wideG','group_A_wideG','sameinfo_consensus0','sameinfo_consensus1','sameinfo_consensus3']
OWN_CODE=[
 'crossplatform/round50/full_common50.py','crossplatform/round50/full_frontend50.py',
 'crossplatform/round50/full_frames50.py','crossplatform/round50/full_frames_batch50.py',
 'crossplatform/round50/full_source_batch50.py','crossplatform/round50/full_cpu50.py',
 'crossplatform/round50/full_geometry_batch50.py','crossplatform/round50/full_engineering50.py',
 'crossplatform/round50/full_visual_observation50.py','crossplatform/round50/text_bootstrap50.py',
 'crossplatform/round50/full_evidence50.py','crossplatform/round50/full_evidence_worklists50.py',
 'crossplatform/round50/full_evidence_batch50.py','crossplatform/round50/full_predict50.py',
 'crossplatform/round50/episodic_capture50.py','crossplatform/round50/episodic_algorithm50.py',
 'crossplatform/round50/episodic_fit50.py','crossplatform/round50/episodic_closure50.py',
 'crossplatform/round50/check_pool_weights50.py','crossplatform/round50/check_worklist_rgb_runtime50.py',
 'research/round50/independent_review/audit_full50_draft.py',
 'research/round50/independent_review/bridge_full_prepared50.py',
 'research/round50/independent_review/score_full50_draft.py',
 'research/round50/independent_review/audit_full_evidence_cost50_draft.py',
 'research/round50/independent_review/audit_episodic50_draft.py','research/round50/score_episodic50_draft.py',
 'research/round50/finalize50_draft.py','research/round50/publish50_draft.py',
 'research/round50/build_dev_export50_draft.py','tools/manual_r50_guard.py']
DENIED_DATA_KEYS={'id','id_PRIVATE','alias','aliases','scene_PRIVATE','caption','subject_query',
 'actual_head_word','token_mask','positive_mask','positive_masks','offsets','char_span',
 'queries','query_indices','query_index','point_indices','mask_indices','mask_index',
 'source_boxes','source_scores','selected_original_boxes','boxes_PRIVATE','box_PRIVATE',
 'coordinates','features','tokens','h','weights','yes_logits','no_logits','raw_answer',
 'imagepath','pointpath','metapath','raw','input_key','logical_binding','value_path'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def text(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(x)
def local(rel):
 p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
 assert 'round46/final' not in rel and 'FINAL384' not in rel
 return ROOT/p
def safe_code(x):return x.replace(str(ROOT),'WORKSPACE_ROOT').replace(str(ROOT.parent),'LOCAL_HOME')
def csvfile(p,rows,fields):
 stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
 writer.writeheader();writer.writerows(rows);text(p,stream.getvalue())
def receipt(x):return {k:v for k,v in x.items() if v is None or type(v) in (bool,int,float) or (isinstance(v,str) and (k=='state' or k.endswith('_sha256')))}
def check_data_fields(value,path='$'):
 if isinstance(value,dict):
  for k,v in value.items():
   assert k not in DENIED_DATA_KEYS,(path,k)
   check_data_fields(v,path+'.'+k)
 elif isinstance(value,list):
  for j,v in enumerate(value):check_data_fields(v,path+f'[{j}]')
def normalized_history(x):
 allowed={'stage','attempt','status','failure_type','repair_type','valid_outputs','started_attempts',
  'entered_forward_attempts','completed_calls','source_or_prompt_or_method_changed','GT_read',
  'prior_attempt_preserved','reused_raw_outputs','new_parsed_text_rows','reused_text_graphs',
  'physical_Stanza_neural_forwards_not_instrumented','measured_seconds'}
 out=[]
 for r in x['events']:
  assert set(r)<=allowed
  row={}
  for k,v in r.items():
   if isinstance(v,str):assert re.fullmatch(r'[A-Za-z0-9_]{1,96}',v),('history_enum',k)
   else:assert v is None or type(v) in (bool,int,float)
   row[k]=v
  out.append(row)
 return {'events':out,'raw_logs_call_payloads_and_vendor_overlay_excluded':True}
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--attempt',type=int,required=True);a=ap.parse_args()
 assert a.attempt>=1;auth=read(AUTH);plan=read(PLAN)
 assert auth['state']=='ROOT_AUTHORIZED_R50_LOCAL_WHITELIST_PUBLIC_EXPORT_ONLY'
 assert auth['export_program_sha256']==sha(__file__) and auth['plan_sha256']==sha(PLAN)
 pins=plan['inputs_sha256']
 def get(rel):assert rel in pins and sha(local(rel))==pins[rel];return read(local(rel))
 # The report receipt is a derived output of this same locked plan. Pin it
 # in the later export grant, not inside its own input plan (a hash cycle).
 assert sha(H/'REPORT_FINAL_RECEIPT.json')==auth['report_receipt_sha256']
 rr=read(H/'REPORT_FINAL_RECEIPT.json')
 assert rr['state']=='PASS_R50_FINAL_REPORT_FROM_LOCKED_SAVED_SCALARS' and rr['plan_sha256']==sha(PLAN)
 assert sha(H/'REPORT.md')==rr['report_sha256'] and sha(H/'DECISION.json')==rr['decision_sha256']
 privacy=get(plan['private_privacy_token_registry'])
 assert sha(local(plan['private_privacy_token_registry']))==auth['private_privacy_token_registry_sha256']
 assert privacy['coverage']['full5912_ids']==5912 and privacy['coverage']['dev336_ids']==336
 assert privacy['coverage']['full_caption_records']==5912 and privacy['coverage']['full_media_path_records']==17736
 assert privacy['coverage']['no_protected_FINAL_specific_artifact_read'] is True
 pub=H/'publication';pub.mkdir(exist_ok=True);out=pub/f'public_export_attempt{a.attempt:03d}'
 assert not out.exists();out.mkdir()
 # This backward-compatible helper generates the full seven-arm DEV scalar
 # package; its old stage-only narrative is omitted in favor of final REPORT.
 helper=local('research/round50/build_dev_export50_draft.py')
 assert sha(helper)==pins[helper.relative_to(ROOT).as_posix()]
 spec=importlib.util.spec_from_file_location('R50_LOCAL_DEV_EXPORT_HELPER',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 counts={'dev':module.build_dev(out,pins,include_narratives=False)}
 fullroot=plan['full_score_root'].rstrip('/')+'/'
 seal=get(fullroot+'SEALED.json')
 for rel,h in seal['files'].items():assert sha(local(fullroot+rel))==h
 tasks=get(fullroot+'ALL5912_TASK_ROWS_PRIVATE.json');assert len(tasks)==5912
 mapping={r['row']:f'F{i+1:05d}' for i,r in enumerate(tasks)};assert len(mapping)==5912
 scenes={pl:sorted({r['scene_PRIVATE'] for r in tasks if r['platform']==pl}) for pl in ('drone','quad')}
 assert {pl:len(v) for pl,v in scenes.items()}=={'drone':2,'quad':3}
 def scene(pl,s):return ('D' if pl=='drone' else 'Q')+'_S'+str(scenes[pl].index(s)+1).zfill(2)
 scalar=[]
 for r in tasks:
  assert set(r['IoU'])==set(r['queries'])==set(ARMS)
  for arm in ARMS:scalar.append({'row':mapping[r['row']],'part':r['part'],'platform':r['platform'],
   'scene_alias':scene(r['platform'],r['scene_PRIVATE']),'arm':arm,'IoU':r['IoU'][arm],
   'changed_vs_source':r['queries'][arm]!=r['queries']['source'],
   'changed_vs_old_G':r['queries'][arm]!=r['queries']['old_G']})
 assert len(scalar)==17736
 for pl,n in [('drone',8952),('quad',8784)]:
  group=[r for r in scalar if r['platform']==pl];assert len(group)==n
  csvfile(out/'full/results'/(pl.upper()+'_'+str(n)+'_TASK_SCALARS.csv'),group,list(scalar[0]))
 negatives=get(fullroot+'ALL_ORDERED_NEGATIVE_COMPARISONS.json')
 publicneg=[{k:r[k] for k in ('part','platform','arm','reference','delta_pp')}|{'row':mapping[r['row']]} for r in negatives]
 for pl in ('drone','quad'):
  group=[r for r in publicneg if r['platform']==pl]
  csvfile(out/'full/results'/(pl.upper()+'_ALL_ORDERED_NEGATIVE_COMPARISONS.csv'),group,['row','part','platform','arm','reference','delta_pp'])
  for ref in ('source','old_G'):
   csvfile(out/'full/results'/(pl.upper()+'_ALL_NEGATIVES_VS_'+ref.upper()+'.csv'),
    [r for r in group if r['reference']==ref],['row','part','platform','arm','reference','delta_pp'])
 summary=get(fullroot+'TASK_SUMMARY.json');allowedmetrics=['n','mIoU_percent','Acc25_percent','Acc50_percent',
  'gain_vs_source_pp','gain_vs_old_G_pp','improved_vs_source','worsened_vs_source',
  'improved_vs_old_G','worsened_vs_old_G','query_changes_vs_source','query_changes_vs_old_G','scene_clusters']
 projected={}
 for part in ('ALL5912','DEV336','NONDEV5576'):
  projected[part]={pl:{arm:{k:summary[part][pl][arm][k] for k in allowedmetrics} for arm in ARMS} for pl in ('drone','quad')}
  for kind in ('balanced_mIoU_percent','micro_mIoU_percent'):projected[part][kind]={arm:summary[part][kind][arm] for arm in ARMS}
 dump(out/'full/results/TASK_SUMMARY.json',projected)
 sr=get(fullroot+'ALL_SCENE_SUMMARIES_PRIVATE.json');sc=[]
 for r in sr:
  for arm in ARMS:sc.append({'part':r['part'],'platform':r['platform'],'scene_alias':scene(r['platform'],r['scene_PRIVATE']),
   'n':r['n'],'arm':arm,'mIoU_percent':r['mIoU_percent'][arm]})
 csvfile(out/'full/results/ALL_SCENE_ARM_SCALARS.csv',sc,['part','platform','scene_alias','n','arm','mIoU_percent'])
 score_receipt=get(fullroot+'SCORING_RECEIPT.json');assert score_receipt['task_values']==17736 and score_receipt['ordered_negative_comparisons']==len(publicneg)
 dump(out/'full/audit/SCORE.json',receipt(score_receipt))
 for key,label in [('full_independent_audit','FULL_PREDICTION'),('full_saved_scalar_audit','FULL_SAVED_SCALAR'),
                   ('full_prepared_bridge_audit','DEV_PREPARED_BRIDGE'),('full_evidence_cost_audit','EVIDENCE_COST')]:
  x=get(plan[key]);assert x['state'].startswith('PASS_');dump(out/'full/audit'/(label+'.json'),receipt(x))
 costroot=plan['full_evidence_cost_audit'].rsplit('/',1)[0]+'/'
 # These source rows are anonymous already, but project explicit fields so
 # an author adding provenance later cannot silently extend public data.
 costrows=get(costroot+'ALL_CHUNK_COSTS.json')
 cfields=['stage','chunk','cases','physical_completed_forwards','logical_uses','historical_exact_uses',
  'historical_whole_case_uses','new_or_samechunk_completed_input_uses','physical_call_cap',
  'ledger_wall_seconds_this_process','model_load_seconds_overlaps_wall','process_peak_bytes',
  'model_unchanged','generated_tokens','optimizer_steps']
 csvfile(out/'full/costs/ALL_CHUNK_COSTS.csv',[{k:r[k] for k in cfields} for r in costrows],cfields)
 costsummary=get(costroot+'SUMMARY.json')
 dump(out/'full/costs/EVIDENCE_STAGE_TOTALS.json',{stage:receipt(costsummary[stage]) for stage in ('dino','numeric')})
 logical=get(costroot+'ANONYMOUS_CASE_LOGICAL_USES.json')
 lfields=['stage','row','platform','old_pool_count','wide_pool_count','target_logical_uses','reference_logical_uses','whole_case_historical_reuse','no_generated_tokens_or_optimizer']
 assert len(logical)==11824
 csvfile(out/'full/costs/ALL11824_CASE_STAGE_LOGICAL_USE_COUNTS.csv',[{k:r[k] for k in lfields} for r in logical],lfields)
 bridge=get(plan['full_prepared_bridge_audit'])
 dump(out/'full/audit/PREPARED_FIELD_EQUALITY_COUNTS.json',receipt(bridge))
 history=get(plan['normalized_technical_history']);dump(out/'history/TECHNICAL_FAILURES_AND_REPAIRS.json',normalized_history(history))
 stagecost=get(plan['normalized_collection_costs'])
 # Closed numeric/bool stage-cost schema; no opaque IDs, commands or logs.
 assert set(stagecost)=={'counts','timing_seconds','flags'}
 assert all(type(v) in (int,float,bool) or v is None for group in stagecost.values() for v in group.values())
 dump(out/'full/costs/COLLECTION_COMPONENT_COUNTS_AND_TIMES.json',stagecost)
 gate=get(plan['full_gate_decision']);dump(out/'DECISION.json',read(H/'DECISION.json'))
 tta=get(plan['TTA_closed_status']);dump(out/'tta/STATUS.json',{'state':tta['state'],'no_missing_results_filled_as_zero':True})
 if tta['state'] in {'COMPLETED_DEV_EPISODIC_SAMEINFO','COMPLETED_DEV_VERIFIED_NO_PARAMETER_UPDATE_DIAGNOSTIC'}:
  nt=get(plan['TTA_normalized_saved_results']);assert nt['state']=='COMPLETED_R50_DEV_EPISODIC_SAMEINFO_SAVED_RESULTS' and nt['n']==336 and nt['arm_names']==TTA_ARMS
  check_data_fields(nt)
  assert len(nt['task_rows'])==2016
  dump(out/'tta/results/ALL2016_ANONYMOUS_TASK_SCALARS.json',nt['task_rows'])
  for key in ('negative_comparisons','scene_rows','summary','update_diagnostics','selection_and_gate'):
   dump(out/'tta/results'/(key.upper()+'.json'),nt[key])
  for key,name in [('TTA_independent_audit','INDEPENDENT'),('TTA_saved_scalar_audit','SAVED_SCALAR')]:
   x=get(plan[key]);assert x['state'].startswith('PASS_');dump(out/'tta/audit'/(name+'.json'),receipt(x))
  counts['tta']={'task_values':2016,'negative_comparisons':len(nt['negative_comparisons'])}
 else:assert tta['state'] in {'NOT_RUN_FULL_GATE_FAIL','NOT_RUN_ROOT_CLOSED_RESOURCE_OR_RUNTIME_LIMIT'}
 text(out/'REPORT.md',safe_code((H/'REPORT.md').read_text()))
 text(out/'README.md','# R50: coverage entrance and association overall comparison\n\nREPORT.md records all seven exposed-development configurations, the locked full three-arm comparison, and the actual conditional-TTA status. Anonymous scalar tables retain every task result and negative comparison. This folder contains no data license redistribution, model weights, raw calls, images, point clouds or private sample mappings.\n\nThe publication is one added study tree. The local builder does not contact GitHub; publication must preserve the complete then-current remote root and all old studies, verify every immutable remote byte, and only then update the branch.\n')
 dump(out/'protocol/FIXED_CONFIGURATION.json',{'source':'Waymo_only_epoch100_frozen_original256_six_coordinate_boxes',
  'DEV_n':336,'full_n':5912,'full_platform_n':{'drone':2984,'quad':2928},'known_scene_counts':{'drone':2,'quad':3},
  'full_arms':ARMS,'DEV_k_grid':[3,10],'DEV_selected_k':10,'full_selected_configuration_has_no_k':True,
  'visual_recipe':'original_partial_subject_two_crops_full_minus_gray_no_new_manual_scores',
  'relation_recipe':'original_per_mask_P_R_and_original_reference_gate',
  'new_conditioned_W_not_selected_for_full':True,'legacy_taxonomy_bug_retained_in_old_G':True,
  'wide_entry_bypasses_legacy_taxonomy_gate_so_not_pure_algorithm_causality':True,
  'allDEV_and_full_predictions_sealed_before_single_task_score':True,
  'historically_exposed_full_target_not_independent_new_scene_confirmation':True,
  'technical_failures_not_task_negative_examples':True,'source_cache_zero_incremental_forwards_not_zero_end_to_end_source_cost':True,
  'shared_collection_wall_not_three_independent_method_latencies':True,
  'Stanza_CPU_neural_inference_occurs_physical_forwards_not_instrumented':True})
 text(out/'protocol/DEPENDENCIES_AND_CODE_BASIS.md','The own-code adapters depend on the original 3EED implementation and separately obtained licensed model assets. Third-party source snapshots, vendor dependency overlays, model/tokenizer files and weights are excluded. Source code filenames ending in _draft.py reflect preparation provenance; executed status comes from audit/receipt projections, not the filename. Episodic code may remain unexecuted, as STATUS.json explicitly records.\n')
 history_code=plan.get('own_history_code',[])
 for item in history_code:
  assert set(item)=={'path','status'} and item['status'] in {'UNEXECUTED_SUPERSEDED_DRAFT','FAILED_INDEPENDENT_AUDITOR_TECHNICAL','EXECUTED_ENGINEERING_FAILURE_SNAPSHOT'}
 for rel in OWN_CODE+[x['path'] for x in history_code]+plan.get('additional_own_code',[]):
  assert rel in pins and sha(local(rel))==pins[rel]
  assert rel.endswith('.py') and not any(t in rel for t in ('vendor','runtime_overlay','runtime_accelerate','runtime_deps','upstream','annotate'))
  assert rel.startswith(('crossplatform/round50/','research/round50/')) or rel=='tools/manual_r50_guard.py'
  text(out/'own_code'/rel,safe_code(local(rel).read_text()))
 dump(out/'history/OWN_CODE_REVISION_STATUS.json',history_code)
 for rel in plan.get('public_review_documents',[]):
  assert rel in pins and sha(local(rel))==pins[rel] and rel.endswith('.md') and rel.startswith('research/round50/independent_review/')
  text(out/'review'/Path(rel).name,safe_code(local(rel).read_text()))
 counts['full']={'task_values':17736,'negative_comparisons':len(publicneg),'scene_arm_rows':len(sc),
  'case_stage_logical_count_rows':len(logical),'chunk_cost_rows':len(costrows)}
 dump(out/'SCHEMA_COUNTS.json',counts)
 # Explicit closed data projections plus exact private token scanning. Raw
 # aliases/IDs/captions/media paths are never included in the public mapping.
 files=sorted(p for p in out.rglob('*') if p.is_file())
 for p in files:
  assert p.suffix in {'.json','.md','.csv','.py'}
  if p.suffix=='.json':check_data_fields(read(p))
 joined='\n'.join(p.read_text() for p in files);findings=[]
 for item in privacy['tokens']:
  assert set(item)=={'kind','value'} and item['kind'] in {'id','alias','scene','caption','media_path'}
  value=item['value'];assert isinstance(value,str) and value
  if item['kind'] in {'id','alias','scene'}:
   hit=re.search(r'(?<![A-Za-z0-9_])'+re.escape(value)+r'(?![A-Za-z0-9_])',joined) is not None
  else:hit=value in joined
  if hit:findings.append({'kind':item['kind'],'token_sha256':hashlib.sha256(value.encode()).hexdigest()})
 assert not findings,('private_token_findings',findings)
 dump(out/'EXPORT_PRIVACY_AUDIT.json',{'state':'PASS_R50_LOCAL_EXPLICIT_WHITELIST_AND_PRIVATE_TOKEN_SCAN',
  'findings':0,'exact_private_tokens_checked':len(privacy['tokens']),
  'private_token_registry_sha256':sha(local(plan['private_privacy_token_registry'])),
  'no_raw_identifiers_captions_scene_names_query_or_mask_indices_coordinates_features_weights_media_or_vendor_files':True,
  'scan_excludes_no_text_code_csv_or_data_file':True})
 files=sorted(p for p in out.rglob('*') if p.is_file())
 dump(out/'MANIFEST.json',{'prefix':PREFIX,'manifest_excludes_itself':True,
  'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]})
 files=sorted(p for p in out.rglob('*') if p.is_file());payload=pub/f'TREE_PAYLOAD_LOCAL_attempt{a.attempt:03d}.json'
 assert not payload.exists();dump(payload,[{'path':PREFIX+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files])
 dump(pub/f'LOCAL_EXPORT_ATTEMPT{a.attempt:03d}_AUDIT.json',{'state':'PASS_R50_LOCAL_PACKAGE_READY_FOR_INDEPENDENT_REVIEW_NOT_PUBLISHED',
  'prefix':PREFIX,'files':len(files),'bytes':sum(p.stat().st_size for p in files),'counts':counts,
  'manifest_sha256':sha(out/'MANIFEST.json'),'payload_sha256':sha(payload),
  'publication_base_snapshot_sha256':auth.get('publication_base_snapshot_sha256'),
  'old_study_count_is_dynamic_and_not_inferred_by_builder':True,
  'full_task_scalar_partition_rows':{'drone':8952,'quad':8784},
  'remote_current_head_or_parent_tree_not_read_or_modified':True,'network_calls':0})
 print(json.dumps({'state':'R50_LOCAL_EXPORT_ONLY_READY','files':len(files),'counts':counts,'payload_sha256':sha(payload)}))

if __name__=='__main__':main()
