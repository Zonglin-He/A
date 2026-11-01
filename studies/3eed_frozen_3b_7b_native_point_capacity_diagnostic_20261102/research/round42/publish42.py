"""Explicit scalar-only local export; never redistribute raw answers or models."""
import json,csv,io,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round42';E=H/'experiment';I=H/'independent_audit';B=ROOT/'crossplatform/round42/pointA208_attempt001';D=ROOT/'crossplatform/assets/Qwen2.5-VL-7B-Instruct';N=ROOT/'research/NIGHT_20261101/reviews/native_grounding_G';PREFIX='studies/3eed_frozen_3b_7b_native_point_capacity_diagnostic_20261102'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def safe(s):return s.replace('WORKSPACE_ROOT','WORKSPACE_ROOT').replace('LOCAL_3B_MODEL','LOCAL_3B_MODEL').replace('LOCAL_HOME','LOCAL_HOME')
def protocol(x):
 if isinstance(x,dict):return {k:protocol(v) for k,v in x.items() if k not in ['files','model_dir','reference3B_processor_dir','preflight2_ids_PRIVATE','ids_PRIVATE']}
 if isinstance(x,list):return [protocol(v) for v in x]
 return safe(x) if isinstance(x,str) else x
def main():
 assert read(I/'AUDIT_RESULT.json')['state']=='PASS_R42_INDEPENDENT_CPU_CONTRACT' and read(I/'SCORING_RECEIPT.json')['task_values']==1664
 pub=H/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
 def cp(p,dst):q=out/dst;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(safe(Path(p).read_text()))
 def cpj(p,dst):dump(out/dst,protocol(read(p)))
 cp(E/'REPORT.md','REPORT.md');cpj(E/'DECISION.json','DECISION.json')
 for p in sorted((ROOT/'crossplatform/round42').glob('*.py')):cp(p,p.relative_to(ROOT))
 for p in [I/'audit42_draft.py',I/'score42_draft.py',H/'finalize42.py',Path(__file__),N/'download7b.py',N/'download7b_attempt002.py']:cp(p,p.relative_to(ROOT))
 # Protocol metadata only; no actual pretrained config/tokenizer contents.
 for n in ['PLAN_LOCK','IMPLEMENTATION_LOCK','EXECUTION_AUTHORIZATION','POSTSEAL_EVALUATION_AUTHORIZATION','CPU_TECH_CHECKS','ROOT_CODE_REVIEW','AUTHOR_HANDOFF']:cpj(E/(n+'.json'),Path('protocol')/(n+'.json'))
 for n in ['ROOT_STATIC_REVIEW','DRAFT_HANDOFF']:cpj(I/(n+'.json'),Path('audit/static')/(n+'.json'))
 cp(I/'REPORT_AGENT_STATIC_REVIEW.md','audit/static/REPORT_AGENT_STATIC_REVIEW.md');cp(I/'DRAFT_README.md','history/PREEXECUTION_AUDIT_README.md');cp(H/'README.md','history/PREEXECUTION_AUTHOR_README.md')
 for n in ['AUDIT_RESULT','SCORING_RECEIPT','TASK_SUMMARY','COVERAGE_UNKNOWN_SUMMARY','IDENTITY24_SUMMARY']:cpj(I/(n+'.json'),Path('results')/(n+'.json'))
 # All task values and unknown/negative records: explicit per-row fields only.
 rows=read(I/'ALL208_TASK_ROWS_PRIVATE.json');assert len(rows)==208;sc={pl:sorted({r['scene_PRIVATE'] for r in rows if r['platform']==pl}) for pl in ['drone','quad']};scalar=[];flags=[]
 for r in rows:
  for arm,v in r['IoU'].items():scalar.append({'alias':r['alias'],'platform':r['platform'],'scene_alias':r['platform']+'_scene'+str(sc[r['platform']].index(r['scene_PRIVATE'])),'arm':arm,'IoU':v,'query':r['queries'][arm],'changed_vs_source':r['queries'][arm]!=r['queries']['source']})
  for mapping,z in r['readouts'].items():flags.append({'alias':r['alias'],'platform':r['platform'],'mapping':mapping,'parse_status':r['parse_status'],**{k:z[k] for k in ['mapped','status','group_size','query_changed','old_subset_contains_selected_mask','full_group_nonempty']}})
 assert len(scalar)==1664 and len(flags)==416;buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(scalar[0]),lineterminator='\n');w.writeheader();w.writerows(scalar);(out/'results/ALL1664_TASK_SCALARS.csv').write_text(buf.getvalue());dump(out/'results/ALL416_READOUT_FLAGS.json',flags)
 neg=[{k:r[k] for k in ['alias','platform','arm','reference','delta_pp']} for r in read(I/'ALL_ORDERED_NEGATIVE_COMPARISONS.json')];assert len(neg)==1045;dump(out/'results/ALL1045_NEGATIVE_COMPARISONS.json',neg)
 scenes=[{'platform':r['platform'],'scene_alias':r['platform']+'_scene'+str(sc[r['platform']].index(r['scene_PRIVATE'])),'n':r['n'],'mean_IoU_percent':r['mean_IoU_percent'],'descriptive_only_few_existing_scenes':True} for r in read(I/'SCENE_SUMMARIES_PRIVATE.json')];dump(out/'results/SCENE_SUMMARIES.json',scenes)
 ikeys=['review_alias','platform','arm','mapping','annotation_status','identity_determinate','claimed_mask_matches_review','point_hits_review_mask_including_overlap','hit_count','mapped','mapping_status','selected_query_in_review_group','prior_seen_artifact_access','case_specific_prior_seen','ambiguous_not_scored_correct','no_new_image_review_nonexpert_not_gold','final_IoU','source_IoU','matched3B_IoU','gain_vs_source_pp','gain_vs_matched3B_pp'];identity=[{k:r[k] for k in ikeys} for r in read(I/'IDENTITY24_TASK_JOIN_PRIVATE.json')];assert len(identity)==48;dump(out/'identity/ALL48_PROVISIONAL_IDENTITY_SCALAR_FLAGS.json',identity)
 dump(out/'identity/BOUNDARY.json',{'same_immutable_annotation_sha256':'6e382dc7cd7cdc4257a8d4811c1b640ca0d466eaab6b2b72fb1dde916c099097','n':24,'clear':18,'ambiguous':6,'not_expert_or_gold':True,'historical_artifact_exposure':True,'no_new_manual_image_review_or_annotation':True,'coverage_aware_not5of6_overall_accuracy':True,'no_feedback_selection_or_override_from_identity_or_IoU':True})
 # No original point values, images, unit positions or token IDs exported.
 units=[]
 for j,r in enumerate(read(I/'AUDITED_SYNTHETIC_ONLY.json')):units.append({'fixture_alias':f'synthetic_{j:02d}',**{k:r[k] for k in ['hit','parse_status','absolute_L2','counterfactual1000_L2']}})
 u=read(B/'COORDINATE_DIAGNOSTIC.json');dump(out/'results/SYNTHETIC8_SCALAR_DIAGNOSTIC.json',{'n':8,'7B_blue_hits':u['blue_hits7B'],'3B_cached_blue_hits':u['blue_hits3B_cached'],'rows':units,'not_perfect_hit_gate_or_unit_selection':True})
 calls=[];ledger=[]
 for j,line in enumerate((B/'GENERATION_BUDGET.jsonl').read_text().splitlines()):
  r=json.loads(line);c=read(B/'calls_PRIVATE'/(r['key']+'.json'));alias=f'synthetic_{j:02d}' if r['phase']=='fixed8_synthetic' else r['alias'];ledger.append({'attempt_index':r['attempt_index'],'alias':alias,'phase':r['phase'],'started_unix':r['started_unix'],'single_generation_no_retry':True});calls.append({'alias':alias,'phase':r['phase'],'parse_status':c['parsed']['status'],**{k:c[k] for k in ['input_tokens','generated_tokens','ended_with_eos','max_new_tokens','actual_repetition_penalty','same_input_as_cached3B','3B7B_reconstructed_CPU_input_tensors_exact','seconds_generate_including_durable_output','raw_persisted_before_parse_and_derived_hashes']}})
 assert len(calls)==216;dump(out/'results/ALL216_CALL_SCALARS.json',calls);dump(out/'results/ALL216_GENERATION_LEDGER.json',ledger)
 # Only own download provenance, not tokenizer/weight/model thirdparty payloads.
 old=read(D/'DOWNLOAD_MANIFEST.json');new=read(D/'DOWNLOAD_MANIFEST_ATTEMPT002.json');assert len(new['files'])==16
 allowold=['state','started_utc','revision','official_repository','license','public_ungated_verified','max_probe_body_bytes_each_site','max_stream_buffer_bytes','max_seconds','probes','conservative_speed_fraction','conservative_bytes_per_second','conservative_ETA_seconds','disk_free_before_bytes','total_expected_bytes','finished_utc'];dump(out/'download/ORIGINAL90MIN_STOP.json',{k:old[k] for k in allowold}|{'original_container_sha256':sha(D/'DOWNLOAD_MANIFEST.json'),'no_full_download_started_before_resource_amendment':True})
 pinned={r['rfilename']:r for r in new['pinned_files']};meta=[]
 for r in new['files']:
  p=pinned[r['filename']];meta.append({**{k:r[k] for k in ['filename','size','sha256','git_blob_sha1','host','attempt','resume_offset','network_bytes','seconds_including_verify','reused_verified']},'pinned_HF_git_blob_id':p['blobId'],'pinned_HF_LFS_sha256':p.get('lfs',{}).get('sha256'),'pinned_expected_size':p['size']})
 dump(out/'download/ALL16_FILE_METADATA.json',{'revision':new['revision'],'official_repository':new['official_repository'],'license':new['license'],'public_ungated_verified':new['public_ungated_verified'],'files':meta,'verified_total_bytes':new['verified_total_bytes'],'original_completed_manifest_sha256':sha(D/'DOWNLOAD_MANIFEST_ATTEMPT002.json'),'only_metadata_no_file_contents':True})
 for p in [N/'DOWNLOAD_7B_PREPARATION_LOCK.json',N/'DOWNLOAD_7B_ATTEMPT002_LOCK.json',N/'DOWNLOAD_7B_COMPLETION_RECEIPT.json',ROOT/'research/NIGHT_20261101/MODEL_7B_PARALLEL_DOWNLOAD_RESOURCE_AMENDMENT_LOCK.json',N/'PROCESSOR_3B_7B_SAME_INFORMATION_CHECK.json']:cpj(p,Path('download')/p.name)
 events=[]
 for name in ['DOWNLOAD_EVENTS.jsonl','DOWNLOAD_EVENTS_ATTEMPT002.jsonl']:
  for line in (D/name).read_text().splitlines():
   r=json.loads(line);allowed=['utc','kind','filename','host','attempt','resume_offset','status','size','sha256','git_blob_sha1','network_bytes','seconds_including_verify','reused_verified','error_type','message'];events.append({'attempt_history':name,**{k:r[k] for k in allowed if k in r}})
 dump(out/'download/ALL_DOWNLOAD_EVENT_METADATA.json',events)
 load=read(B/'MODEL_LOAD_RECEIPT.json');dump(out/'results/MODEL_LOAD_SUMMARY.json',{k:load[k] for k in ['seconds_load','memory_includes_load_peak_before_reset','actual_parameter_count','loading_info','versions','local_only_BF16_SDPA_no_remote_code']}|{'effective_generation_overrides':{'do_sample':False,'num_beams':1,'max_new_tokens':96,'use_cache':True,'repetition_penalty':1.05},'stored_default_generation_do_sample':load['model_generation_config']['do_sample'],'stored_default_is_not_actual_effective_sampling':True})
 for n in ['FINAL_RESOURCE_RECEIPT','INPUT_WEIGHT_VERIFY_COST','PREFLIGHT2_RECEIPT']:cpj(B/(n+'.json'),Path('results')/(n+'.json'))
 seal=read(B/'SEALED.json');dump(out/'protocol/PREDICTION_SEAL_METADATA.json',protocol(seal)|{'original_container_sha256':sha(B/'SEALED.json'),'private_binding_filemap_excluded':True})
 dump(out/'SOURCES.json',{'model':'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/cc594898137f460bfe9f0759e9844b3ce807cfb5','official_repository':'https://github.com/QwenLM/Qwen2.5-VL','3EED':'https://github.com/worldbench/3EED','prior_R37_R38_commit':'5a5fe1466471992d674439d818cc2efaecfae696','fixed_custom_point_interface_not_official_benchmark_reproduction':True,'no_thirdparty_source_or_snapshot_redistributed':True})
 (out/'README.md').write_text('''# R42 — fixed 3B/7B native point capacity diagnostic

NO_ADOPT this7Bpoint configuration,noTTA. Balanced mIoU7B22.645214 vs3B22.740349
(-.095135pp), legacyG24.727373, source19.663563. Drone-.841201/Quad+.650931pp.
SomeAcc25/50 improve: allmetrics retained,no posthocprimaryswitch. Thisis only
fixedpoint/fullimage/fulltext/prompt/mapping, not general7Bineffectiveness.

208Adevelopment+8fixedsynthetic=216singlegreedycalls,preflight2reused,raw/token
atomicdurability,noanswerretry. 8arms1664IoUs/all1045orderednegativecomparisons,
416readoutflags/all100null/outside41/parsefailure9 retained. Fullmap49vs3B160,
noacceptedsample-onlyclaim. Provisionalimmutable24identity18clear/6ambiguous:
7B5consistent1contradicts12unknown,3B8/8/2,coverage/nonexpert/historylimitskept.
Not5/6overallaccuracy; noGTfeedbackfilter ornewannotation. Synthetic3/8vs1/8
diagnosticnotgating/unitselection. Originalsource3Dboxes/6cachedcontrolsexact.

IndependentCPU216processor inputtensor/rawtoken/schema/mapping auditedPASS,
1664IoUs scored onlypostseal+rootauth. CPUno generationmodel/GPU/optimizer.
GPUwall124.58s includesload12.26s/CPUprocessing/persistence,verify18.63s separate,
peakallocated17.163GB(15.98GiB)/reserved19.046GB(17.74GiB),hostHWM4.87GiB.
Modelgenerationconfigdefaultsampling isoverriddenbyactualdo_sampleFalse/beam1.

Pinnedofficial16file metadata only,5shards firstattemptverified,16.596GBdownload
82.39min,nostreamfailure. Original90minresourcegateSTOPat195.3minconservativeETA
kept; laterexplicitroot200min/max2stream/07:45revision,not retroactivegatepass.
Modelweights/tokenizer/configbodies/thirdpartycode not redistributed. Onlyown
source,protocol,metadata,anonymousIoU/scalar/status/flags; no rawanswers/tokenIDs,
pointvalues/masks/3Dcoordinates/IDs/captions/scenes/paths/features/media/weights.
Reproduction oforiginalseals needslicensedprivateassets. A208exposed2Drone3Quad
scenes,notnewconfirmation; allcomparisonsdescriptive,noCIclaims. FINAL/U/Tuntouched.
history/PREEXECUTION* and olddraftlockstatus are chronology,finalDECISION governs.
''')
 # Scan only already-authorized A208 contents, never holdout media/labels.
 hist=read(ROOT/'research/round27/input/ALL528_PRIVATE.json')[:208];forbidden={str(r[k]) for r in hist for k in ['id','caption','imagepath','pointpath','metapath','scene'] if r.get(k)};findings=[]
 for p in sorted(out.rglob('*')):
  if not p.is_file():continue
  txt=p.read_text()
  for token in forbidden:
   if token in txt:findings.append({'file':p.relative_to(out).as_posix(),'type':'private_token','sha256':hashlib.sha256(token.encode()).hexdigest()})
  if re.search('/'+'home/'+r'[^/\s]+/',txt):findings.append({'file':p.relative_to(out).as_posix(),'type':'home_path'})
 assert not findings,findings[:8]
 blocked={'id','id_PRIVATE','ids_PRIVATE','preflight2_ids_PRIVATE','caption','scene','scene_PRIVATE','imagepath','pointpath','metapath','mask_index','claimed_mask_index_PRIVATE','selected_query_PRIVATE','review_target_display_ids','original_point','value','point','raw_answer_PRIVATE','generated_token_ids','pixel_values','input_ids','source_boxes','masks_bool','source_scores','scores','tokenizer','model_generation_config','image_grid_thw'};keyfind=[]
 def walk(x,path):
  if isinstance(x,dict):
   for k,v in x.items():
    if k in blocked:keyfind.append({'file':path,'key':k})
    walk(v,path)
  elif isinstance(x,list):
   for v in x:walk(v,path)
 for p in out.rglob('*.json'):walk(read(p),p.relative_to(out).as_posix())
 assert not keyfind,keyfind[:8]
 dump(out/'EXPORT_PRIVACY_AUDIT.json',{'state':'PASS','explicit_task_status_identity_call_download_metadata_allowlists':True,'all1664taskvalues_all1045negatives':True,'all216calls_and416readoutflags':True,'all48identityjoinflags':True,'all16download_file_metadata_no_contents':True,'original_STOP_and_resource_revision_preserved':True,'private_token_findings':findings,'blocked_structured_field_findings':keyfind,'no_weights_tokenizer_bodies_thirdparty_source_raw_coordinates_media':True,'no_FINAL_U_T_or_newGT_reads_for_export':True})
 dump(out/'protocol/FINAL_CODE_SNAPSHOT.json',{'own_code_only':[{'path':p.relative_to(out).as_posix(),'sha256':sha(p)} for p in sorted(out.rglob('*.py'))]})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];dump(out/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'manifest_excludes_itself':True})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];target=pub/'TREE_PAYLOAD_LOCAL.json';dump(target,[{'path':PREFIX+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files])
 for r in read(out/'MANIFEST.json')['files']:assert sha(out/r['path'])==r['sha256'] and (out/r['path']).stat().st_size==r['bytes']
 for r,p in zip(read(target),files):assert r['content']==p.read_text() and r['path']==PREFIX+'/'+p.relative_to(out).as_posix()
 result={'state':'PASS','n':len(files),'bytes':sum(p.stat().st_size for p in files),'payload_bytes':target.stat().st_size,'payload_sha256':sha(target),'prefix':PREFIX,'all1664values_1045negatives_216calls_48identityflags_16downloadmeta':True,'privacy_findings':[],'manifest_payload_allSHAexact':True,'not_published':True};dump(pub/'LOCAL_EXPORT_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
