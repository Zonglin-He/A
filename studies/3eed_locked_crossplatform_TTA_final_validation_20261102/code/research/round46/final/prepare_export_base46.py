"""Local engineering export base only; no scoring, models, network or publishing.

The result/report agent owns the final release and adds anonymous scored results.
This module never reads a row manifest, task annotations, prediction arrays,
features, model weights, raw call logs or per-observation identifiers.
"""
from pathlib import Path
import json,hashlib,argparse,re
ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'research/round46';FINAL=HERE/'final';BASE=ROOT/'crossplatform/round46/final384_attempt001'
PREFIX='studies/3eed_locked_crossplatform_TTA_final_validation_20261102'
OWN_CODE_WHITELIST=[
 'crossplatform/round46/common46.py','crossplatform/round46/controller46.py',
 'crossplatform/round46/source_heads46.py','crossplatform/round46/frames46.py',
 'crossplatform/round46/oldg46.py','crossplatform/round46/assemble46.py',
 'crossplatform/round46/external46.py','crossplatform/round46/engineering_audit46.py',
 'crossplatform/round46/fresh_audit46.py',
 'crossplatform/round24/common24.py','crossplatform/round24/fit24.py',
 'crossplatform/round24/parseM24.py','crossplatform/round24/prepareM24.py',
 'crossplatform/round24/detectM24.py','crossplatform/round24/evidenceM24.py',
 'crossplatform/round24/rankM24.py','crossplatform/round23/common23.py',
 'crossplatform/round22/common22.py','crossplatform/round21/common21.py',
 'crossplatform/round21/reference21_v3.py','crossplatform/round20/common.py',
 'crossplatform/round20/common_v2.py','crossplatform/round20/textZ.py',
 'crossplatform/round20/observations.py','crossplatform/round20/visualZ.py',
 'crossplatform/round20/prepareZ.py','crossplatform/round20/assemble_v2.py',
 'crossplatform/round20/detect.py','crossplatform/round17/pair_geometry.py',
 'crossplatform/round13/association_runner.py','crossplatform/round13/association.py',
 'crossplatform/round13/input_io.py','crossplatform/round15/scene_observation.py',
 'crossplatform/round7/camera_projection.py',
 'crossplatform/round7/experiment/runner_helpers.py','crossplatform/round9/run_diagnostic.py',
 'crossplatform/round6/factory.py','crossplatform/round18/runner.py',
 'crossplatform/round18/algorithms.py','crossplatform/round18/supplement_algorithms.py',
 'crossplatform/round18/reg_inspired.py','crossplatform/round18/geometry.py',
 'crossplatform/round12/capture_masks.py'
]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def safe_text(s):return s.replace('WORKSPACE_ROOT','WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME')
def opaque_hex_matches(text):
 """Reject opaque16hex except explicitly delimited decimal numeric mantissas.

 Exact registered identifier scans remain the unified publisher's separate gate.
 A path/quoted hex token is not exempted just because it contains digits.
 """
 for m in re.finditer(r'(?<![A-Za-z0-9])[0-9a-f]{16}(?![A-Za-z0-9])',text):
  before=text[:m.start()];after=text[m.end():];decimal_prefix=bool(re.search(r'(?<![A-Za-z0-9_.])[0-9]+\.$',before));number_end=bool(re.match(r'(?=$|[^A-Za-z0-9_./])',after));scientific_end=bool(re.match(r'[+\-−][0-9]+(?=$|[^A-Za-z0-9_./])',after))
  if decimal_prefix and (m.group().isdigit() and number_end or m.group().endswith('e') and m.group()[:-1].isdigit() and scientific_end):continue
  yield m
def build_base(output):
 """Called once by the unified publisher, into an empty local release folder."""
 out=Path(output);assert not out.exists(),'No overwrite of prior staged releases';out.mkdir(parents=True)
 plan=read(FINAL/'FINAL_PLAN_DRAFT.json');methods=read(FINAL/'METHODS_CONTROLS_LOCK.json')
 assert sha(FINAL/'FINAL_PLAN_DRAFT.json')=='f0db1dba7b121d1e493b46da51b962fccfd25ca0ab2fe58661f3c8cd9b23834c'
 assert plan['methods_controls_lock_sha256']==sha(FINAL/'METHODS_CONTROLS_LOCK.json')
 assert all(sha(ROOT/p)==h for p,h in plan['program_sha256'].items())
 finalseal=read(BASE/'SEALED.json');assert sha(BASE/'SEALED.json')=='9970abdacaf45f6a3f0fa13a91ad31e3dc3b698f990f419d09a8762596f7c4cf'
 audit=read(HERE/'independent_final/audit384_attempt001/AUDIT_RESULT.json');assert audit['state']=='PASS_R46_INDEPENDENT_FINAL_PREDICTION_CONTRACT'
 code=[]
 for rel in OWN_CODE_WHITELIST:
  assert rel.endswith('.py') and '/official/' not in rel and '/upstream/' not in rel and '/assets/' not in rel
  source=ROOT/rel;target=out/'code'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(safe_text(source.read_text()));code.append({'path':target.relative_to(out).as_posix(),'source_program_sha256':sha(source),'public_sha256':sha(target)})
 (out/'ENGINEERING_REPORT.md').write_text(safe_text((FINAL/'ENGINEERING_REPORT.md').read_text()))
 basearms=['source','R24_query_mlp_lr1_s16','R24_memory_alpha1','R30_beta1','R30_memory_E_only','R30_memory_allA_zeroU']
 external=['source','Tent_BN_statistics_first_pre','Tent_clean_post_port','SAR_LN_first_pre','SAR_LN_clean_post_port','MEMO_BN_prior_only','MEMO_augmentation_only','MEMO_AdamW1e5_clean_post_port','Reg_inspired_teacher_pre','Reg_inspired_teacher_post','Reg_inspired_student_post','Reg_inspired_direct_pseudo']
 protocol={'state':'FINAL_METHODS_FROZEN_BEFORE_DATA_AND_ALL_PREDICTIONS_SEALED_BEFORE_SINGLE_LABEL_SCORE','n':384,'platform_observations':{'drone':192,'quad':192},'historical_scene_counts':{'drone':1,'quad':3},'primary':'R30_beta1','secondary':'R24_query_mlp_lr1_s16','main_arms':basearms+[n+'_plus_current_oldG' for n in basearms],'main_protocol':'fixedA208adaptation_thenfrozenfinalforward,nocurrentupdates','external_arms':external,'external_protocol':'reset_each_current_input_nativefirst_and_addedcleanpost_separate','main_parameters_per_head':184960,'source_input':'XYZ16384+fulltext;noRGBstudentbranch','source_checkpoint_sha256':'0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd','source_training_platforms':['waymo'],'source_training_epochs':100,'readout':'GTfree_first_lexicon_noun_fallback_all_content_tokens_not_official_GTclasspositive_map','source_training_official_map_history_disclosed':True,'native_query_token_temperature':.07,'sentinel':' . not mentioned','memory':'frozenquery*normalizednoun_keyL2,CPU32perArownearestquery_thenTop4exp((cos-1)/.1),alpha1,nativeprobabilityincrements','same_current_G':'oncefrozen-sourcepool F64 F=s0+2vision+2position+2relation,sharedDelta,source+G/head+G/memory+Gallreported','current_G_resources':['sameR12MaskRCNN','originalcameraallowlistandpointgeometry','localStanza','GroundingDINO','frozenQwen2.5VL3Bnumericfullgray'],'Reg_name':'Reg-TTA3D-inspired,notoriginaldetectorreproduction','no_final_hyperparameter_or_method_selection':True,'offline_source_coverage':'postsealonlystablequeryTop1/5/10and256oracle,120scap;nofeedbackorreadoutusesGT;upperonlyfrozenpoolmain','raw_private_registration_distributed':False,'opaque_original_plan_sha256':sha(FINAL/'FINAL_PLAN_DRAFT.json'),'opaque_prediction_seal_sha256':sha(BASE/'SEALED.json')}
 dump(out/'protocol/ANONYMOUS_PROTOCOL.json',protocol)
 dump(out/'protocol/EXTERNAL_FIXED_CONFIG.json',{'Tent':{'optimizer':'SGD','lr':.001,'momentum':.9,'steps':1,'selected_parameters':42816,'BN_statistics_control_and_cleanpost':True},'SAR_LN_only':{'optimizer':'officialSAM_SGD','lr':.001,'momentum':.9,'rho':.05,'max_logical_steps':1,'entropy_margin':'.4ln256','recovery_threshold':.2,'selected_parameters':24768,'sourceBN_retained':True,'nativefirst_and_cleanpost':True},'MEMO_Transformer_AdamW':{'lr':.00001,'weight_decay':.01,'max_steps':1,'views':4,'BN_prior':16,'common_groups_min':2,'all_view_mass_coverage_min':.5,'selected_parameters':144527419,'prior_aug_cleanpost_controls_same_original_run':True},'Reg_inspired':{'optimizer':'Adam','lr':.001,'max_steps':1,'selected_parameters':1734,'substituted_RoI_yaw_class_confidence_interfaces':True,'teacherpre_teacherpost_studentpost_directpseudo_retained':True}})
 stages=read(BASE/'STAGES.json');assert len(stages)==9 and all(x['returncode']==0 for x in stages)
 trace=read(BASE/'external/TRACE_PRIVATE.json');stepnames=['Tent','SAR_LN_only','MEMO_Transformer_AdamW','Reg_inspired'];steps={n:sum(r['diagnostics'][n]['steps'] for r in trace) for n in stepnames};assert steps=={'Tent':384,'SAR_LN_only':129,'MEMO_Transformer_AdamW':291,'Reg_inspired':382}
 costs={'controller_wall_seconds':finalseal['seconds'],'stage_wall_seconds':stages,'stage_wall_sum_seconds':sum(x['seconds'] for x in stages),'control_and_final_sealing_overhead_seconds':finalseal['seconds']-sum(x['seconds'] for x in stages),'nested_warm_seconds_not_additional':{'source_and_heads':47.40892955334857,'frames':24.295549689326435,'DINO':116.74985285801813,'Qwen_numeric':234.76712180487812,'main_memories':8.563946904148906,'external':430.72999721579254,'external_per_observation_trace_sum':424.183396124281},'source_installed_head_full_forwards':1152,'MaskRCNN_forwards':384,'DINO_forwards':916,'numeric_Qwen_forwards':4272,'generated_or_nonnumeric_VLM_calls':0,'new_source_training':0,'new_final_main_optimizer_steps':0,'external_logical_steps':steps,'external_logical_steps_total':sum(steps.values()),'external_skips':{n:384-steps[n] for n in stepnames},'recorded_peak_allocated_bytes':{'source_and_heads':707024384,'Qwen':7729118720,'external_max_per_observation':max(r['peak_bytes'] for r in trace)},'MaskRCNN_DINO_independent_peak_not_recorded':True,'external_total_grounder_forward_count_not_individually_instrumented':True,'prior_source_training_and_A_adaptation_cost_not_included_not_zero_claim':True,'independent_contract_audit_seconds':audit['seconds'],'final_stage_failure_or_retry':False,'hard_outer_seconds':3300,'night_hard_deadline_local':'2026-11-02T09:30:00+08:00'}
 dump(out/'results/ENGINEERING_COSTS.json',costs)
 failures={'full_oldA2_controller':{'observations':2,'seconds':50.25052747083828,'numeric_Qwen':24,'DINO':8,'external_steps':8,'no_taskscore':True},'metadata_preparation_failure':{'attempts':1,'reason':'oldassetreceiptpath typo','model_calls':0,'failure_preserved':True},'fresh_oldA2_attempts':[{'attempt':1,'outcome':'missing_parent_directory_before_model_import','model_calls':0,'repair':'create_parent_only'},{'attempt':2,'outcome':'raw_mask_saved_then_cameraDecimal_JSON_IO_error','model_calls':1,'raw_preserved':True},{'attempt':3,'outcome':'reuse_hash_bound_prior_rawmask_and_one_remaining_forward','model_calls':1,'all_mask_camera_scene_prepare_contracts_exact':True}],'fresh_total_MaskRCNN_calls':2,'fresh_Qwen_DINO_optimizer_calls':0,'no_model_answer_selection_or_repeated_first_mask':True,'independent_fresh_comparator_failure':{'reason':'dataclass_equality_on_NumPy_array','model_calls':0,'repair':'fieldwise_recursive_comparison','original_code_log_preserved':True},'all_preexecution_lock_revisions_preserved_locally':True,'final_prediction_attempts':1,'final_stage_failures':0,'final_automatic_retries':0,'new_G_method_or_postscore_method_changes':False,'raw_failure_logs_with_private_paths_not_public':True}
 dump(out/'history/ENGINEERING_ATTEMPT_HISTORY.json',failures)
 auditkeys=['state','n','main_arms','external_protocol_controls','actual_full_model_source_head_forwards','fresh_MaskRCNN_calls','fresh_all384_image_point_metadata_mask_bindings_complete','fresh_camera_allowlist_only_and_mask_schema_nativeorder_checked','head_CPU64_maxerror','head_CPU64_vs_actual_native_argmax_differences','actual_native_GPU_argmax_preserved','memory_similarity_cells_recomputed_CPU32','memory_boundary_tie_query_counts','numeric_Qwen_calls','nonnumeric_VLM_calls','DINO_calls','external_logical_steps','external_skips','all_external_full_box_pools_and_explicit_override_contracts_checked','G_boundary','external_boundary','no_GT_taskscore_model_optimizer_or_method_change','seconds']
 dump(out/'audit/ENGINEERING_CONTRACT_SUMMARY.json',{k:audit[k] for k in auditkeys})
 dump(out/'DEPENDENCIES_ENGINEERING.json',{'runtime':'Existingtorch2.7cu128/torchvision.22 point-grounder env andexistingfrozenVLM env;localonly/noinstalls','original_3EED':'https://github.com/worldbench/3EED','official_Tent':'https://github.com/DequanWang/tent/tree/e9e926a668d85244c66a6d5c006efbd2b82e83e8','official_SAR':'https://github.com/mr-eggplant/SAR/tree/20f6e24b17525f34503510afccedc0629b67b7c4','official_MEMO':'https://github.com/zhangmarvin/memo/tree/228b2908d271c954ef8bf19cf143ede3b2fa8e3e','Reg_paper':'https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05971.pdf','reproduction':'Overlay own-code files on previous published R18/R20-R24/R30 task-port chain;provide licensed3EED,localupstreamops,officialSAMclone,source100checkpoint,Qwen/MaskRCNN/DINO/Stanzaassets andoriginalprivatepreselectedstates/registrations. Anonymousconfig isnot executableprivatecohortregistration. No privatecohort/states/rawdata distributed.','thirdparty_source_weights_paper_snapshots_redistributed':False,'private_manifest_UID_alias_caption_GT_coordinates_vectors_features_weights_excluded':True,'oldA2_engineering_and_final_contracts_not_semantic_gold':True})
 dump(out/'protocol/ENGINEERING_CODE_BASIS.json',{'source':'explicitowncodewhitelist,noimportwalk/noresultcontainercopy','files':code,'new_final_methods':False,'private_source_registration_rows_excluded':True,'scored_results_owner':'independentreportagent;separatelyintegratesanonymousreport/tables/negatives/sourcecoverage','network_or_publishing_actions':False})
 # Structural engineering-only privacy gate. Result agent applies its own exact
 # token scan to the combined package under its separate authorized export scope.
 blocked_extensions={'.npz','.npy','.pth','.pt','.safetensors','.jpg','.jpeg','.png','.pdf','.bin'}
 for f in out.rglob('*'):
  if not f.is_file():continue
  assert f.suffix.lower() not in blocked_extensions
  txt=f.read_text();assert not re.search('/'+'home/'+r'[^/\s]+/',txt),f
  assert not list(opaque_hex_matches(txt)),('unexpected_private_style_identifier',f)
 return {'state':'LOCAL_ENGINEERING_BASE_ONLY_NOT_PUBLISHED','prefix':PREFIX,'files':[p.relative_to(out).as_posix() for p in sorted(out.rglob('*')) if p.is_file()],'scored_results_not_read_or_added':True,'no_manifest_UID_alias_GT_coords_features_weights_thirdparty_copied':True}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(build_base(a.output),indent=2))
if __name__=='__main__':main()
