"""Explicit anonymous scalar export; never distribute linking map or raw data."""
import csv,io,json,hashlib,re
from pathlib import Path
R=Path(__file__).resolve().parents[3];F=R/'research/round46/final';D=R/'research/round46/independent_final';S=D/'score384_attempt001';A=D/'audit384_attempt001';B=R/'crossplatform/round46/final384_attempt001';N=R/'research/NIGHT_20261101'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def safe(x):return x.replace('WORKSPACE_ROOT','WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME')
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k!='files'}
 if isinstance(x,list):return [clean(v) for v in x]
 return safe(x) if isinstance(x,str) else x
def csvwrite(p,rows):
 rows=list(rows);assert rows;buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(buf.getvalue())
def build_results(out):
 O=Path(out);assert O.exists();rows=read(S/'ALL384_TASK_ROWS_PRIVATE.json');assert len(rows)==384;aliases={r['alias']:f'F{i+1:04d}' for i,r in enumerate(rows)};ids={r['id_PRIVATE']:aliases[r['alias']] for r in rows};byid={r['id_PRIVATE']:r for r in rows};scenes={pl:sorted({r['scene_PRIVATE'] for r in rows if r['platform']==pl}) for pl in ['drone','quad']};scenealias=lambda r:r['platform']+'_scene'+str(scenes[r['platform']].index(r['scene_PRIVATE'])+1)
 (O/'REPORT.md').write_text((F/'REPORT.md').read_text());dump(O/'DECISION.json',read(F/'RESULT_DECISION.json'))
 scalar=[]
 for r in rows:
  for arm,value in r['IoU'].items():scalar.append({'row_alias':aliases[r['alias']],'platform':r['platform'],'scene_alias':scenealias(r),'arm':arm,'IoU':value,'query_changed_vs_source':r['queries'][arm]!=r['queries']['main::source']})
 assert len(scalar)==9216;csvwrite(O/'results/ALL9216_ANONYMOUS_TASK_SCALARS.csv',scalar)
 arm_names=list(rows[0]['IoU']);arm_ids={name:f'A{i+1:02d}' for i,name in enumerate(arm_names)};assert len(arm_ids)==24;dump(O/'results/ARM_DICTIONARY.json',{'encoding':'losslessshortarmIDs_fornegativeCSV_only','arms':[{'arm_id':arm_ids[name],'arm':name,'protocol':'Aadapt_to_Bfrozen' if name.startswith('main::') else 'current_input_episodic'} for name in arm_names]});rawneg=read(S/'ALL_ORDERED_NEGATIVE_COMPARISONS.json');negatives=[{'row_alias':aliases[r['alias']],'platform':r['platform'],'arm_id':arm_ids[r['arm']],'reference_id':arm_ids[r['reference']],'delta_pp':r['delta_pp']} for r in rawneg];assert len(negatives)==37622;assert all(n['delta_pp']==r['delta_pp'] and n['arm_id']==arm_ids[r['arm']] and n['reference_id']==arm_ids[r['reference']] for n,r in zip(negatives,rawneg));csvwrite(O/'results/ALL37622_ORDERED_NEGATIVE_COMPARISONS.csv',negatives)
 for n in ['TASK_SUMMARY','FIXED_SAME_INFORMATION_PAIRED_OUTCOMES','SCORING_RECEIPT','SOURCE_POOL_DIAGNOSTIC_SUMMARY']:dump(O/'results'/(n+'.json'),read(S/(n+'.json')))
 sc=[{'platform':r['platform'],'scene_alias':scenealias(r),'n':r['n'],'mIoU_percent':r['mIoU_percent'],'descriptive_only_no_crossscene_CI':True} for r in read(S/'SCENE_SUMMARIES_PRIVATE.json')];dump(O/'results/ALL4_SCENE_SUMMARIES.json',sc)
 diag=[]
 for r in read(S/'SOURCE_POOL_ALL_ROWS_PRIVATE.json'):
  v={'row_alias':aliases[r['alias']],'platform':r['platform'],'scene_alias':scenealias(r)}
  for k in ['1','5','10']:
   for field,value in r['TopK'][k].items():v['Top'+k+'_'+field]=value
  v.update({k:r[k] for k in ['all256_oracle_IoU_GTassisted','all256_oracle_Acc25','all256_oracle_Acc50','source_top_score_tie_count','topK_are_queries_not_distinct_instances','only_frozen_sourcepool_selection_upperbound_not_MEMO_Reg_newboxes']});diag.append(v)
 assert len(diag)==384;csvwrite(O/'results/ALL384_SOURCE_POOL_DIAGNOSTIC_SCALARS.csv',diag)
 contract=[]
 for r in read(A/'ALL384_CONTRACT_ROWS_PRIVATE.json'):contract.append({'row_alias':aliases[r['alias']],**{k:v for k,v in r.items() if k not in ['alias','id_PRIVATE']}})
 assert len(contract)==384;dump(O/'audit/ALL384_CONTRACT_FLAGS.json',contract)
 for name,count,label in [('ALL768_HEAD_ARITHMETIC_SCALARS_PRIVATE',768,'ALL768_HEAD_ARITHMETIC_SCALARS'),('ALL1152_MEMORY_ARITHMETIC_SCALARS_PRIVATE',1152,'ALL1152_MEMORY_ARITHMETIC_SCALARS')]:
  rr=[{'row_alias':aliases[r['alias']],**{k:v for k,v in r.items() if k!='alias'}} for r in read(A/(name+'.json'))];assert len(rr)==count;csvwrite(O/'audit'/(label+'.csv'),rr)
 updates=[];coverage=['steps','selected_parameters','gradient_parameters','gradient_L2','parameter_delta_L2','changed_parameter_count','changed_tensor_count','nonselected_parameters_unchanged','loss','entropy','reliability_margin','first_reliable','second_reliable','recovered','second_entropy','augmentations','anchor_groups','common_groups','npg_candidates','npg_before_second_NMS','npg_accepted','noise_stable_before_confidence','printed_literal_tau1_5_rejects','CRR_low_boxes','teacher_steps','EMA_alpha','student_delta_L2','teacher_delta_L2']
 for r in read(B/'external/TRACE_PRIVATE.json'):
  for method,d in r['diagnostics'].items():
   row={'row_alias':aliases[r['alias']],'platform':r['platform'],'method':method,**{k:d.get(k) for k in coverage},'min_view_coverage':min(d['coverage']) if d.get('coverage') else None,'mean_view_coverage':sum(d['coverage'])/len(d['coverage']) if d.get('coverage') else None,'source_bridge_exact':r['source_bridge_exact']};updates.append(row)
 assert len(updates)==1536;csvwrite(O/'results/ALL1536_EXTERNAL_UPDATE_AND_SKIP_SCALARS.csv',updates)
 dump(O/'audit/INDEPENDENT_FULL_CONTRACT_AUDIT.json',read(A/'AUDIT_RESULT.json'));dump(O/'audit/INDEPENDENT_ACTUAL_CODE_LOCK.json',clean(read(D/'INDEPENDENT_IMPLEMENTATION_LOCK.json')))
 # These are protocol/hash authorizations, never the registered row manifest.
 for n in ['FINAL_PLAN_DRAFT','IMPLEMENTATION_LOCK_DRAFT','METHODS_CONTROLS_LOCK','ACTUAL_STATIC_REVIEW','ROOT_EXECUTION_AUTHORIZATION','ROOT_INDEPENDENT_AUDIT_AUTHORIZATION','ROOT_SINGLE_TASK_SCORING_AUTHORIZATION']:dump(O/'protocol'/(n+'.json'),clean(read(F/(n+'.json'))))
 own=[D/'audit46_final.py',D/'score46_final.py',R/'research/round5/audit/independent_core_v2.py',R/'tools/obb_iou.py',F/'finalize_results46.py',Path(__file__),F/'prepare_export_base46.py']
 for p in own:
  target=O/'code'/p.relative_to(R);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(safe(p.read_text()))
 # Final execution had no failed audit/score attempt; old preparation failures are
 # represented by the engineering author's scalar history, never copied raw logs.
 dump(O/'history/FINAL_SINGLE_PASS_RESULT_HISTORY.json',{'final_prediction_attempts':1,'final_audit_attempts':1,'final_score_attempts':1,'all_successful':True,'all24outputs_sealed_before_audit_then_separate_labelgrant':True,'no_method_or_primary_change_after_score':True,'oldA2_engineering_failures_separate':True,'source_pool384complete_within120seconds':True,'sourcepool_timing_includes_cache_and_row_statistics_not_pure_kernel':True})
 night=(N/'OVERNIGHT_REPORT_DRAFT.md').read_text();sources=read(N/'OVERNIGHT_REPORT_DRAFT_SOURCES.json');assert sha(N/'OVERNIGHT_REPORT_DRAFT.md')==sources['report_sha256'];night=safe(night);night=night.replace('独立最终确认','锁定新内容验证').replace('最终独立确认','锁定新内容验证').replace('最终确认','锁定新内容验证').replace('这次确认','这次锁定验证').replace('最终384条在方法锁定后一次确认','最终384条在方法锁定后一次验证');night=re.sub(r'\b(?:A|W|R|S|T|U|V|H|FINAL)_(?:drone|quad)_\d+\b','历史匿名案例',night);(O/'OVERNIGHT_REPORT.md').write_text(night);dump(O/'history/OVERNIGHT_REPORT_SOURCES.json',clean(sources)|{'public_report_sha256':sha(O/'OVERNIGHT_REPORT.md'),'publication_wording_only_edits':'localpathsanitization,historicalcasealiasanonymization,lockednewcontentvalidation_not_independentgoldconfirmation;metricsunchanged'})
 for n in ['AUDIT_RESULT','FIXED_MAIN_CONCLUSIONS']:dump(O/'audit/night_scalar_recheck'/(n+'.json'),clean(read(N/'final_scalar_audit'/(n+'.json'))))
 links=[]
 for study in ['3eed_feedback_observation_influence_diagnostic_20261102','3eed_conditional_preference_readout_20261102','3eed_local_support_partial_output_keep_TTA_20261102','3eed_source_fused_rgb_rank_preference_TTA_20261102','3eed_native_grounding_complete_mapping_schema_compatibility_20261102','3eed_native_bbox_mask_association_diagnostic_20261102','3eed_cached_numeric_grounding_readout_diagnostic_20261102','3eed_dense_distillation_coordinate_null_control_20261102','3eed_frozen_3b_7b_native_point_capacity_diagnostic_20261102','3eed_oldG_ordinal_instance_preference_transfer_20261102','3eed_native_bbox_capacity_interface_factorial_20261102','3eed_unclipped_bbox_geometric_domain_diagnostic_20261102']:links.append({'study':study,'url':'https://github.com/Zonglin-He/A/tree/main/studies/'+study})
 dump(O/'SOURCES.json',{'official_3EED':'https://github.com/worldbench/3EED','historical_completed_public_studies':links,'own_code_protocol_and_scalar_artifacts_only':True,'thirdparty_snapshots_or_models_redistributed':False,'private_linking_map_not_exported':True,'historical_positives_preserved_but_no_final_overall_parameter_gain':True})
 return {'task_scalars':9216,'negative_comparisons':37622,'source_pool_rows':384,'contract_flags':384,'head_scalar_rows':768,'memory_scalar_rows':1152,'external_update_skip_rows':1536,'anonymous_aliases':'F0001..F0384','scenes':{'drone':1,'quad':3},'no_mapping_rawvalues_or_private_identity_exported':True}
