"""Explicit anonymous-scalar local export only. No network publishing or scoring."""
from pathlib import Path
import json,hashlib,csv,io,re,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round48';A=H/'independent_audit';B=H/'blind_text_audit';PREFIX='studies/3eed_cached_subject_span_readout_diagnostic_20261102'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safe(s):return s.replace('WORKSPACE_ROOT','WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME')
def clean(x):
 if isinstance(x,dict):return {safe(k):clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 return safe(x) if isinstance(x,str) else x
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 assert read(A/'SCORING_RECEIPT.json')['task_values']==1680;pub=H/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt003';out.mkdir(exist_ok=False)
 def cp(p,dst):q=out/dst;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(safe(Path(p).read_text()))
 def cpj(p,dst):dump(out/dst,clean(read(p)))
 for n in ['REPORT.md','DECISION.json','SOURCES.json']:cp(H/n,n) if n.endswith('.md') else cpj(H/n,n)
 cp(H/'experiment/ENGINEERING_PROTOCOL_DRAFT.md','protocol/ENGINEERING_PROTOCOL.md');cp(B/'REPORT.md','audit/blind24/REPORT.md');cp(A/'README.md','audit/INDEPENDENT_PROTOCOL.md')
 own=[*sorted((R/'crossplatform/round48').glob('*.py')),H/'finalize48.py',Path(__file__),A/'audit48_independent.py',A/'score48_independent.py',A/'official_span_independent48.py',A/'diagnose_saved48.py',A/'schema_inventory48.py',B/'select48.py',B/'compare_sealed48.py',H/'independent_scalar_recheck/reaggregate48.py']
 assert B/'annotate48.py' not in own
 for p in own:cp(p,p.relative_to(R))
 for n in ['SEMANTIC_PLAN_LOCK','PARSER_RULES_LOCK','PLAN_LOCK','ROOT_EXECUTION_AUTHORIZATION','ROOT_INDEPENDENT_AUDIT_AUTHORIZATION','ROOT_SINGLE_TASK_SCORING_AUTHORIZATION']:cpj(H/'experiment'/(n+'.json'),Path('protocol')/(n+'.json'))
 for n in ['AUDIT_RESULT','SCORING_RECEIPT','ACTUAL_STATIC_REVIEW','INDEPENDENT_CODE_LOCK','TECHNICAL_START_FAILURE_RECEIPT']:cpj(A/(n+'.json'),Path('audit')/(n+'.json'))
 for n in ['TASK_SUMMARY','FACTOR_CONTRASTS','COVERAGE_STRATA','FIXED_MASK_PRIORITY_SUMMARY']:cpj(A/(n+'.json'),Path('results')/(n+'.json'))
 dump(out/'audit/CACHE_SCHEMA_CONTRACT_SUMMARY.json',{'observations':336,'candidate_count':256,'actual_text_token_range':[29,86],'query_embedding_dimension':64,'source_box_dimension':6,'cached_arrays_descriptive_names':['h','x','text_h','tokens','query','token_mask','boxes','scores'],'input_ids_stored':False,'numeric_vectors_not_redistributed':True})
 cpj(H/'independent_scalar_recheck/AUDIT_RESULT.json','audit/INDEPENDENT_SAVED_SCALAR_RECHECK.json')
 for n in ['TEXT_ROLE_SUMMARY','COMPARISON_RECEIPT','ANNOTATIONS_SEALED','TEXT_ROLE_REVIEW_SEALED','COHORT_LOCK']:cpj(B/(n+'.json'),Path('audit/blind24')/(n+'.json'))
 # Preserve old preexecution grammar/locks; no real-data-derived text is present.
 old=H/'experiment/preexecution_revision001'
 for n in ['PARSER_RULES_LOCK.json','PLAN_LOCK.json','ACTUAL_STATIC_REVIEW.json','parser48.py']:cpj(old/n,Path('history/preexecution_revision001')/n) if n.endswith('.json') else cp(old/n,Path('history/preexecution_revision001')/n)
 tasks=read(A/'ALL336_TASK_ROWS_PRIVATE.json');assert len(tasks)==336;mapping={r['id_PRIVATE']:'P'+str(i+1).zfill(4) for i,r in enumerate(tasks)};aliasmap={r['alias']:mapping[r['id_PRIVATE']] for r in tasks};scenes={pl:sorted({r['scene_PRIVATE'] for r in tasks if r['platform']==pl}) for pl in ['drone','quad']};diag={r['id_PRIVATE']:r for r in read(A/'ALL336_FIXED_MASK_DIAGNOSTIC_PRIVATE.json')};scalar=[];flags=[]
 for r in tasks:
  row=mapping[r['id_PRIVATE']];scene=r['platform']+'_scene'+str(scenes[r['platform']].index(r['scene_PRIVATE'])+1);base=r['queries']['current_proxy_CPU64']
  for arm,value in r['IoU'].items():scalar.append({'row':row,'split':r['split'],'platform':r['platform'],'scene_alias':scene,'arm':arm,'IoU':value,'query_changed_vs_current':r['queries'][arm]!=base})
  keys=['parser_status','structured','parser_reason','parser_fallback','official_no_match','official_literal_fallback','official_remap_fallback','literal_remap_mask_equal','current_parser_mask_equal','missing_word_offsets'];d=diag[r['id_PRIVATE']];flags.append({'row':row,'split':r['split'],'platform':r['platform'],**{k:r[k] for k in keys},'top_tie_counts':r['top_tie_counts'],'CPU64_vs_storednative_maxerror':r['CPU64_vs_storednative_maxerror'],'official_current_mask_equal':d['official_current_mask_equal'],'official_current_query_changed':d['official_current_query_changed'],'synonym_priority_reason':d['reason'],'current_term_available_in_official_list':d['proxy_term_available_in_same_official_list'],'task_delta_pp_offline_only':d['task_delta_pp_offline_only'],'task_direction':d['task_direction']})
 assert len(scalar)==1680 and len(flags)==336
 buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(scalar[0]),lineterminator='\n');w.writeheader();w.writerows(scalar);p=out/'results/ALL1680_TASK_SCALARS.csv';p.parent.mkdir(exist_ok=True);p.write_text(buf.getvalue());dump(out/'results/ALL336_STATUS_AND_FIXED_MASK_FLAGS.json',flags)
 neg=[{'row':aliasmap[r['alias']],**{k:r[k] for k in ['split','platform','arm','reference','delta_pp']}} for r in read(A/'ALL_NEGATIVES.json')];assert len(neg)==60;dump(out/'results/ALL60_NEGATIVE_COMPARISONS.json',neg)
 scene_rows=[{'split':r['split'],'platform':r['platform'],'scene_alias':r['platform']+'_scene'+str(scenes[r['platform']].index(r['scene_PRIVATE'])+1),'n':r['n'],'mIoU_percent':r['mIoU_percent']} for r in read(A/'SCENE_SUMMARIES_PRIVATE.json')];assert len(scene_rows)==20;dump(out/'results/ALL20_SCENE_SUMMARIES.json',scene_rows)
 blind=read(B/'TEXT_ROLE_COMPARISON_PRIVATE.json');blindmap={r['id_PRIVATE']:'B'+str(i+1).zfill(3) for i,r in enumerate(dict((x['id_PRIVATE'],x) for x in blind).values())};bk=['platform','stratum','arm','blind_status','nominal_span_exact_blind','effective_head_character_coverage','effective_contains_full_blind_head','effective_exact_blind_head_token_mask','effective_exact_blind_head_character_union','same_head_word_at_different_nominal_position_not_credit','same_head_word_at_other_effective_position_not_credit','selected_noncaption_tokens','declared_UNKNOWN_or_fallback','effective_fallback_to_current','mask_positive_tokens','literal_uses_actual_offsets_not_official_raw_span_as_if_aligned','nonexpert_text_roles_not_visual_or_GTidentity_gold','no_TASK_score_or_GTbox_loaded'];bs=[{'row':blindmap[r['id_PRIVATE']],**{k:r[k] for k in bk}} for r in blind];assert len(bs)==96 and len(blindmap)==24;dump(out/'audit/blind24/ALL96_ROLE_OVERLAP_SCALARS.json',bs);dump(out/'audit/blind24/ALL24_NATIVE_BRIDGE_NOT_APPLICABLE.json',[{'row':n,'arm':'cached_native_current_BRIDGE','text_span_metric':'NOT_APPLICABLE_NO_INDEPENDENT_TEXT_MASK','not_counted_as_fifth_span_prediction':True} for n in blindmap.values()])
 plan=read(H/'experiment/PLAN_LOCK.json');run=[]
 for stage,key in [('caption_only_parse','parse_output'),('five_readouts','prediction_output')]:
  p=R/plan[key]/'SEALED.json';s=read(p);run.append({'stage':stage,'state':s['state'],'n':s['n'],'created_utc':s['created_utc'],'seconds':s['seconds'],'seal_sha256':sha(p),'filemap_and_private_paths_not_redistributed':True})
 dump(out/'protocol/RUNTIME_SEAL_METADATA_SUMMARY.json',run);dump(out/'results/COSTS.json',{'CPU_parse_program_seconds':run[0]['seconds'],'CPU_readout_program_seconds':run[1]['seconds'],'independent_audit_seconds':read(A/'AUDIT_RESULT.json')['seconds'],'single_task_score_seconds':read(A/'SCORING_RECEIPT.json')['seconds'],'new_grounder_or_VLM_forwards':0,'GPU_calls':0,'optimizer_steps':0,'not_single_sample_deployment_latency':True});dump(out/'history/ALL_FAILURE_TYPES.json',{'initial_synthetic_schema_failure':{'actual_NLP_calls':0,'GTclass_or_bbox_read':False,'error':'missing true nominal head offsets caused its dependent compound to be promoted','revision':'pre-execution nominal offset schema check returns UNKNOWN','original_plan_rules_code_and_static_review_retained':True,'same_six_synthetic_checks_after_repair':'PASS'},'independent_audit_launch_failure':read(A/'TECHNICAL_START_FAILURE_RECEIPT.json'),'no_rule_change_after_task_labels':True})
 dump(out/'README.json',{'report':'REPORT.md','scope':'old exposed D208/V64/M64 only, not new confirmation','task_values':1680,'negative_comparisons':60,'all_observation_flags':336,'blind24_scalar_rows':96,'native_bridge_NA_rows':24,'raw_annotation_script_excluded_because_case_specific_head_text_and_reasons':True,'thirdparty_models_source_snapshots_weights_media_captions_coords_indices_features_and_raw_seals_not_redistributed':True,'reproduction_requires_licensed_original_data_and_registered_private_caches':True})
 # Explicit exact-token privacy check against OLD336 registration, no closed-final access.
 oldrows=read(R/'research/round24/input/ALL336_PRIVATE.json');forbidden={str(r[k]) for r in oldrows for k in ['id','alias','caption','scene','imagepath','pointpath','metapath'] if r.get(k)};findings=[]
 for p in out.rglob('*'):
  if not p.is_file():continue
  text=p.read_text()
  for token in forbidden:
   if token in text:findings.append({'file':p.relative_to(out).as_posix(),'kind':'exact_private_token','token_sha256':hashlib.sha256(token.encode()).hexdigest()})
  if '/'+'home/' in text:findings.append({'file':p.relative_to(out).as_posix(),'kind':'local_home_path'})
 blocked={'id','id_PRIVATE','alias','review_alias','caption','class','head_text','matched_word','head_span','field_span','compound_spans','actual_offsets','positive_masks','token_mask','source_boxes','boxes','queries','scores','shared_CPU64_probabilities','tokens','query','h','x','text_h','imagepath','pointpath','metapath','scene','scene_PRIVATE','nominal_charspan_PRIVATE','effective_token_charspans_PRIVATE','blind_head_charspan_PRIVATE','subject_head_text_PRIVATE','official_synonym_priority_index'};bad=[]
 def walk(x,path):
  if isinstance(x,dict):
   for k,v in x.items():
    if k in blocked:bad.append({'file':path,'key':k})
    walk(v,path)
  elif isinstance(x,list):
   for v in x:walk(v,path)
 for p in out.rglob('*.json'):walk(read(p),p.relative_to(out).as_posix())
 assert not findings,findings[:6];assert not bad,bad[:6]
 dump(out/'EXPORT_PRIVACY_AUDIT.json',{'state':'PASS_EXPLICIT_ANONYMOUS_SCALAR_ALLOWLIST_EXPORT','all_old336_private_tokens_scanned':True,'task_values':1680,'negative_comparisons':60,'status_rows':336,'blind_role_scalar_rows':96,'native_bridge_na_rows':24,'scene_summaries':20,'no_class_caption_rawhead_span_offset_mask_queryindex_box_feature_weight_media_or_rawUID_data':True,'case_specific_annotation_script_and_raw_seals_explicitly_excluded':True,'findings':findings,'forbidden_data_field_findings':bad,'no_FINAL_access_for_export_or_privacy_scan':True})
 dump(out/'protocol/FINAL_OWN_CODE_SNAPSHOT.json',[{'path':p.relative_to(out).as_posix(),'sha256':sha(p)} for p in sorted(out.rglob('*.py'))]);files=[p for p in sorted(out.rglob('*')) if p.is_file()];dump(out/'MANIFEST.json',{'prefix':PREFIX,'manifest_excludes_itself':True,'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]});files=[p for p in sorted(out.rglob('*')) if p.is_file()];payload=pub/'TREE_PAYLOAD_LOCAL.json';dump(payload,[{'path':PREFIX+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files]);result={'state':'PASS_LOCAL_R48_PUBLIC_EXPORT','prefix':PREFIX,'files':len(files),'bytes':sum(p.stat().st_size for p in files),'payload_bytes':payload.stat().st_size,'payload_sha256':sha(payload),'task_values':1680,'negative_comparisons':60,'status_rows':336,'blind_rows':96,'bridge_NA_rows':24,'privacy_findings':0,'published':False}
 for f in read(out/'MANIFEST.json')['files']:assert sha(out/f['path'])==f['sha256'] and (out/f['path']).stat().st_size==f['bytes']
 assert len(read(payload))==len(files)
 dump(pub/'LOCAL_EXPORT_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
