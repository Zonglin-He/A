"""Scalar allowlist export. No model/raw answer/coordinate redistribution."""
import json,csv,io,hashlib,re
from pathlib import Path
R=Path(__file__).resolve().parents[2];H=R/'research/round45';E=H/'experiment';I=H/'independent_audit';B=R/'crossplatform/round45/bboxA208_attempt001';PREFIX='studies/3eed_native_bbox_capacity_interface_factorial_20261102'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def safe(x):return x.replace('WORKSPACE_ROOT','WORKSPACE_ROOT').replace('LOCAL_3B_MODEL','LOCAL_3B_MODEL').replace('LOCAL_HOME','LOCAL_HOME')
def protocol(x):
 if isinstance(x,dict):return {k:protocol(v) for k,v in x.items() if k not in ['files','model_dir','reference3B_processor_dir','preflight2_ids_PRIVATE','ids_PRIVATE']}
 if isinstance(x,list):return [protocol(v) for v in x]
 return safe(x) if isinstance(x,str) else x
def main():
 assert read(I/'AUDIT_RESULT.json')['state']=='PASS_R45_INDEPENDENT_CPU_CONTRACT';assert read(I/'SCORING_RECEIPT.json')['task_values']==4368
 P=H/'publication';P.mkdir(exist_ok=True);O=P/'public_export_attempt001';O.mkdir(exist_ok=False)
 def cp(p,d):q=O/d;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(safe(Path(p).read_text()))
 def cpj(p,d):dump(O/d,protocol(read(p)))
 cp(E/'REPORT.md','REPORT.md');cpj(E/'DECISION.json','DECISION.json')
 for p in sorted((R/'crossplatform/round45').glob('*.py')):cp(p,p.relative_to(R))
 for p in [I/'audit45_draft.py',I/'score45_draft.py',H/'finalize45.py',Path(__file__)]:cp(p,p.relative_to(R))
 for n in ['PLAN_LOCK','IMPLEMENTATION_LOCK','IMPLEMENTATION_LOCK_ORIGINAL_PRE_REV1','EXECUTION_AUTHORIZATION','POSTSEAL_EVALUATION_AUTHORIZATION','CPU_TECH_CHECKS','AUTHOR_HANDOFF','AUTHOR_HANDOFF_REV1','REPORT_AGENT_STATIC_REVIEW','RESOURCE_SCOPE_CLARIFICATION']:cpj(E/(n+'.json'),Path('protocol')/(n+'.json'))
 for n in ['AUDIT_RESULT','SCORING_RECEIPT','TASK_SUMMARY','CAPACITY_INTERFACE_CONTRASTS','COVERAGE_UNKNOWN_SUMMARY','PARSE_STATUS_COUNTS','IDENTITY24_SUMMARY']:cpj(I/(n+'.json'),Path('results')/(n+'.json'))
 cpj(I/'DRAFT_HANDOFF.json','audit/PREEXECUTION_DRAFT_HANDOFF.json');cp(I/'DRAFT_README.md','history/PREEXECUTION_AUDIT_README.md');cp(H/'README.md','history/PREEXECUTION_AUTHOR_README.md')
 rows=read(I/'ALL208_TASK_ROWS_PRIVATE.json');assert len(rows)==208;sc={pl:sorted({x['scene_PRIVATE'] for x in rows if x['platform']==pl}) for pl in ['drone','quad']};scalars=[];flags=[]
 for r in rows:
  for arm,v in r['IoU'].items():scalars.append({'alias':r['alias'],'platform':r['platform'],'scene_alias':r['platform']+'_scene'+str(sc[r['platform']].index(r['scene_PRIVATE'])),'arm':arm,'IoU':v,'query':r['queries'][arm],'changed_vs_source':r['queries'][arm]!=r['queries']['source']})
  for arm,z in r['readouts'].items():flags.append({'alias':r['alias'],'platform':r['platform'],'arm':arm,'parse_status':r['parse_status'],'center_rectIoU_mask_conflict':r['center_rectIoU_mask_conflict'],**{k:z[k] for k in ['mapped','status','group_size','query_changed','old_subset_contains_selected_mask','full_group_nonempty']}})
 assert len(scalars)==4368 and len(flags)==832;buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(scalars[0]),lineterminator='\n');w.writeheader();w.writerows(scalars);(O/'results/ALL4368_TASK_SCALARS.csv').write_text(buf.getvalue());dump(O/'results/ALL832_READOUT_FLAGS.json',flags)
 neg=[{k:r[k] for k in ['alias','platform','arm','reference','delta_pp']} for r in read(I/'ALL_ORDERED_NEGATIVE_COMPARISONS.json')];assert len(neg)==7770;dump(O/'results/ALL7770_NEGATIVE_COMPARISONS.json',neg)
 scenes=[{'platform':r['platform'],'scene_alias':r['platform']+'_scene'+str(sc[r['platform']].index(r['scene_PRIVATE'])),'n':r['n'],'mIoU_percent':r['mIoU_percent'],'factorial_effects':r['factorial_effects'],'descriptive_only_few_existing_scenes':True} for r in read(I/'SCENE_SUMMARIES_PRIVATE.json')];dump(O/'results/SCENE_SUMMARIES.json',scenes)
 keys=['review_alias','platform','arm','annotation_status','identity_determinate','claimed_mask_matches_review','mapped','mapping_status','selected_query_in_review_group','prior_seen_artifact_access','case_specific_prior_seen','ambiguous_not_scored_correct','no_new_image_review_nonexpert_not_gold','final_IoU','source_IoU','legacy_IoU','matched3B_IoU','gain_vs_source_pp'];identity=[{k:r[k] for k in keys} for r in read(I/'IDENTITY24_TASK_JOIN_PRIVATE.json')];assert len(identity)==96;dump(O/'identity/ALL96_PROVISIONAL_IDENTITY_SCALAR_FLAGS.json',identity)
 dump(O/'identity/BOUNDARY.json',{'same_immutable_annotation_sha256':'6e382dc7cd7cdc4257a8d4811c1b640ca0d466eaab6b2b72fb1dde916c099097','n':24,'clear':18,'ambiguous':6,'not_expert_or_gold':True,'historical_exposure':True,'no_new_image_review_or_annotation':True,'no_feedback_GTfilter_or_subset_rescue':True})
 ledger=[];calls=[]
 for line in (B/'GENERATION_BUDGET.jsonl').read_text().splitlines():
  r=json.loads(line);c=read(B/'calls_PRIVATE'/(r['key']+'.json'));ledger.append({'attempt_index':r['attempt_index'],'alias':r['alias'],'phase':r['phase'],'started_unix':r['started_unix'],'single_generation_no_retry':True});calls.append({'alias':c['alias'],'phase':c['phase'],'parse_status':c['parsed']['status'],**{k:c[k] for k in ['input_tokens','generated_tokens','ended_with_eos','max_new_tokens','actual_repetition_penalty','same_input_as_cached3B','3B7B_reconstructed_CPU_input_tensors_exact','seconds_generate_including_durable_output','raw_persisted_before_parse_and_derived_hashes']}})
 assert len(calls)==208;dump(O/'results/ALL208_CALL_SCALARS.json',calls);dump(O/'results/ALL208_GENERATION_LEDGER.json',ledger)
 for n in ['FINAL_RESOURCE_RECEIPT','INPUT_WEIGHT_VERIFY_COST','PREFLIGHT2_RECEIPT']:cpj(B/(n+'.json'),Path('results')/(n+'.json'))
 load=read(B/'MODEL_LOAD_RECEIPT.json');dump(O/'results/MODEL_LOAD_SUMMARY.json',{k:load[k] for k in ['seconds_load','actual_parameter_count','loading_info','versions','local_only_BF16_SDPA_no_remote_code']}|{'effective_generation':{'do_sample':False,'num_beams':1,'max_new_tokens':96,'use_cache':True,'repetition_penalty':1.05},'stored_default_do_sample':load['model_generation_config']['do_sample'],'stored_default_not_effective_sampling':True})
 seal=read(B/'SEALED.json');dump(O/'protocol/PREDICTION_SEAL_METADATA.json',protocol(seal)|{'original_container_sha256':sha(B/'SEALED.json'),'private_filemap_excluded':True})
 dump(O/'SOURCES.json',{'official_model':'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/cc594898137f460bfe9f0759e9844b3ce807cfb5','official_code':'https://github.com/QwenLM/Qwen2.5-VL','3EED':'https://github.com/worldbench/3EED','matched_3B_bbox_and_mapping_commit':'e2dd9ba4532706664327c93d405543eaa3cfa572','point_capacity_and_download_metadata_commit':'84d67755cf7b83a4445eecbf29c2ea64d4356ce9','R42_download_16_file_metadata_and_original_STOP_resource_amendment_retained_at_prior_commit':True,'no_model_tokenizer_thirdparty_source_snapshot_redistributed':True})
 (O/'README.md').write_text('''# R45 — fixed native bbox capacity/interface factorial

STOP fixed 7B bbox-center primary, no TTA. All A208 development rows, 21 arms,
4368 task IoUs and all 7770 ordered negative comparisons retained. The latter
are repeated comparisons, not independent samples. Primary balanced mIoU
20.551200 vs legacy G 24.727373; secondary rectangle-IoU20.465032 retained.
Primary capacity × interface interaction -.087377pp, secondary -.477953pp.
115 invalid boxes, 20 nulls and 17 bad JSON outputs all remain unchanged;
strict parser uses declared resized-input range, no clipping/reordering/unitswitch.
Coverage center46/rectangle53; old-subset21/24. Immutable24identity18clear6ambig,
center4consistent2contradict12unknown, rectangle5/2/11; no accepted-only accuracy.
No new annotation or GT feedback selection. All 17 cached control arms exact.
208 single greedy calls, preflight2 reused, atomic raw/token persistence;
GPUwall148.796s includesload/CPU/persistence, verification13.429s separate.
The pre-REV1 lock/handoff and the preexecution status wording remain history;
REV1 corrected receipt counters only, before any call. No runtime retries.
The R42 publication retains original download STOP/amendment and 16 file
metadata, linked in SOURCES; no model/tokenizer/config body redistributed here.
Own code/protocol/scalar/flags only. No raw answer/token IDs, masks/coordinates,
private IDs/captions/scene paths, source vectors/features/weights/media or
thirdparty snapshots. Reproduction needs licensed/private assets. A208 exposed
few-scene development only; no FINAL/U/T access, independent confirmation or CI.
This diagnoses a fixed interface pipeline, not all7B models or native grounding.
''')
 hist=read(R/'research/round27/input/ALL528_PRIVATE.json')[:208];forbidden={str(r[k]) for r in hist for k in ['id','caption','imagepath','pointpath','metapath','scene'] if r.get(k)};findings=[]
 for p in sorted(O.rglob('*')):
  if not p.is_file():continue
  txt=p.read_text()
  for token in forbidden:
   if token in txt:findings.append({'file':p.relative_to(O).as_posix(),'kind':'private_token','token_sha256':hashlib.sha256(token.encode()).hexdigest()})
  if re.search('/'+'home/'+r'[^/\s]+/',txt):findings.append({'file':p.relative_to(O).as_posix(),'kind':'home_path'})
 assert not findings,findings[:8]
 blocked={'id','id_PRIVATE','ids_PRIVATE','preflight2_ids_PRIVATE','caption','scene','scene_PRIVATE','imagepath','pointpath','metapath','mask_index','claimed_mask_index_PRIVATE','selected_query_PRIVATE','review_target_display_ids','original_point','value','point','raw_answer_PRIVATE','generated_token_ids','pixel_values','input_ids','source_boxes','masks_bool','source_scores','scores','tokenizer','model_generation_config','image_grid_thw'};kf=[]
 def walk(x,p):
  if isinstance(x,dict):
   for k,v in x.items():
    if k in blocked:kf.append({'file':p,'key':k})
    walk(v,p)
  elif isinstance(x,list):
   for v in x:walk(v,p)
 for p in O.rglob('*.json'):walk(read(p),p.relative_to(O).as_posix())
 assert not kf,kf[:8]
 dump(O/'EXPORT_PRIVACY_AUDIT.json',{'state':'PASS','explicit_anonymous_scalar_flag_call_allowlists':True,'all4368_taskvalues_7770negatives_832readouts_96identity_208calls':True,'private_token_findings':findings,'blocked_structured_fields':kf,'no_raw_coordinates_media_features_models_tokenizer_or_thirdparty_snapshots':True,'old_strict_unknown_invalid_and_preREV1_history_preserved':True,'no_FINAL_U_T_reads_for_export':True})
 dump(O/'protocol/FINAL_CODE_SNAPSHOT.json',{'own_code_only':[{'path':p.relative_to(O).as_posix(),'sha256':sha(p)} for p in sorted(O.rglob('*.py'))]})
 files=[p for p in sorted(O.rglob('*')) if p.is_file()];dump(O/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':p.relative_to(O).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'manifest_excludes_itself':True})
 files=[p for p in sorted(O.rglob('*')) if p.is_file()];target=P/'TREE_PAYLOAD_LOCAL.json';dump(target,[{'path':PREFIX+'/'+p.relative_to(O).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files])
 for r in read(O/'MANIFEST.json')['files']:assert sha(O/r['path'])==r['sha256'] and (O/r['path']).stat().st_size==r['bytes']
 for r,p in zip(read(target),files):assert r['content']==p.read_text()
 result={'state':'PASS','n':len(files),'bytes':sum(p.stat().st_size for p in files),'payload_bytes':target.stat().st_size,'payload_sha256':sha(target),'prefix':PREFIX,'all4368values_7770negatives_832flags_96identity_208calls':True,'privacy_findings':[],'all_manifest_payload_SHAexact':True,'not_published':True};dump(P/'LOCAL_EXPORT_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
