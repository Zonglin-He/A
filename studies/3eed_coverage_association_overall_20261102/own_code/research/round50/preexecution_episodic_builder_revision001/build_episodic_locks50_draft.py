"""UNEXECUTED root-reviewed stage-plan constructor; never runs a worker.

Import opens no data. Actual construction requires a root builder grant and
the independently rechecked, pre-registered full5912 positive gate BEFORE
reading DEV rows or emitting plans/grants. Execution auth is draft by default;
root must separately permit grant emission for the single requested stage.
"""
from pathlib import Path
import argparse,json,hashlib,math,sys,importlib.metadata,datetime
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'research/round50/experiment'
ARMS=['source','literal_A_wideG','group_A_wideG','sameinfo_consensus0','sameinfo_consensus1','sameinfo_consensus3']
STAGES=('engineering_capture','engineering_fit','dev_capture','dev_fit','dev_closure','independent_audit','single_score')
OLD='research/round24/input/ALL336_PRIVATE.json'
FULL_RULE='research/round50/experiment/FULL_GATE_AND_TTA_ADVANCE_BEFORE_FULL_GT.json'
FULL_FREEZE='research/round50/experiment/FULL_METHOD_FREEZE_BEFORE_FULL_MEDIA.json'
FULL_SCORE='research/round50/independent_score/full5912_attempt001'
FULL_RECHECK='research/round50/independent_scalar_recheck/full5912_attempt001'
SOURCE_PLAN='research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'
REVIEW='research/round50/independent_review/EPISODIC_DRAFT_STATIC_REVIEW_REV2.json'
E2_ROWS='research/round50/experiment/EPISODIC_ENGINEERING2_ROWS_PRIVATE.json'
ENG_SALT='R50episodic_engineering2_v1|'
B_SALT='R50episodic_B_v1'
CLOSURE_SALT='R50episodic_actual_state_closure_v1|'
RUNS={
 'engineering_capture':'crossplatform/round50/episodic_engineering2_capture_attempt001',
 'engineering_fit':'crossplatform/round50/episodic_engineering2_fit_attempt001',
 'dev_capture':'crossplatform/round50/episodic336_capture_attempt001',
 'dev_fit':'crossplatform/round50/episodic336_fit_attempt001',
 'dev_closure':'crossplatform/round50/episodic336_closure_attempt001',
 'independent_audit':'research/round50/independent_review/episodic336_audit_attempt001',
 'single_score':'research/round50/independent_score/episodic336_attempt001'}
PROGRAM={
 'engineering_capture':'crossplatform/round50/episodic_capture50.py',
 'engineering_fit':'crossplatform/round50/episodic_fit50.py',
 'dev_capture':'crossplatform/round50/episodic_capture50.py',
 'dev_fit':'crossplatform/round50/episodic_fit50.py',
 'dev_closure':'crossplatform/round50/episodic_closure50.py',
 'independent_audit':'research/round50/independent_review/audit_episodic50_draft.py',
 'single_score':'research/round50/score_episodic50_draft.py'}
AUTH_STATE={
 'engineering_capture':'ROOT_AUTHORIZED_R50_DEV_EPISODIC_SOURCE_CAPTURE',
 'engineering_fit':'ROOT_AUTHORIZED_R50_DEV_EPISODIC_HEAD_ONLY',
 'dev_capture':'ROOT_AUTHORIZED_R50_DEV_EPISODIC_SOURCE_CAPTURE',
 'dev_fit':'ROOT_AUTHORIZED_R50_DEV_EPISODIC_HEAD_ONLY',
 'dev_closure':'ROOT_AUTHORIZED_R50_DEV_EPISODIC_INSTALLED_HEAD_CLOSURE',
 'independent_audit':'ROOT_AUTHORIZED_R50_INDEPENDENT_DEV_EPISODIC_TINYHEAD_AUDIT_NO_GT',
 'single_score':'ROOT_AUTHORIZED_R50_SINGLE_DEV_EPISODIC_POSTSEAL_TASK_SCORE'}
PLANS={s:f'research/round50/experiment/EPISODIC_{s.upper()}_PLAN_LOCK.json' for s in STAGES}
PLANS['independent_audit']='research/round50/experiment/EPISODIC_INDEPENDENT_AUDIT_PLAN_LOCK.json'
PLANS['single_score']='research/round50/experiment/EPISODIC_SCORING_PLAN_LOCK.json'
AUTHS={s:f'research/round50/experiment/ROOT_EPISODIC_{s.upper()}_EXECUTION_AUTHORIZATION.json' for s in STAGES}
AUTHS['independent_audit']='research/round50/experiment/ROOT_INDEPENDENT_EPISODIC_AUDIT_AUTHORIZATION.json'
TIMEOUT={'engineering_capture':300,'engineering_fit':300,'dev_capture':1800,'dev_fit':1800,'dev_closure':300,'independent_audit':900,'single_score':300}
CODE=[
 'crossplatform/round50/episodic_capture50.py','crossplatform/round50/episodic_algorithm50.py',
 'crossplatform/round50/episodic_fit50.py','crossplatform/round50/episodic_closure50.py',
 'research/round50/score_episodic50_draft.py','research/round50/independent_review/audit_episodic50_draft.py',
 'crossplatform/round50/full_common50.py','crossplatform/round50/pool_weights50.py','crossplatform/round50/prepare50.py',
 'crossplatform/round49/stages49.py','crossplatform/round49/ownership49.py','crossplatform/round15/scene_observation.py',
 'crossplatform/round13/association.py','crossplatform/round7/camera_projection.py','crossplatform/round18/geometry.py',
 'crossplatform/round9/run_diagnostic.py','crossplatform/round7/regression_adapter.py',
 'crossplatform/round7/experiment/runner_helpers.py','crossplatform/round6/factory.py',
 'research/round5/audit/independent_core_v2.py','tools/manual_r50_guard.py']
BASE_INPUTS=[OLD,SOURCE_PLAN,REVIEW,
 'research/round50/TTA_EPISODIC_PROTOCOL_DRAFT.md','research/round50/EPISODIC_IMPLEMENTATION_DRAFT.json',
 'crossplatform/round24/capture336_attempt001/SEALED.json',
 'crossplatform/round50/prepared336_attempt001/SEALED.json',
 'crossplatform/round50/predicted336_attempt001/SEALED.json',
 'crossplatform/round49/ownership336_attempt001/SEALED.json','crossplatform/round49/stage336_attempt001/SEALED.json',
 'research/round17/input/A_CASES_PRIVATE.json','research/round17/input/A_FRAME_INDEX_PRIVATE.json',
 'research/round12/data_audit/FRAME_INDEX_PRIVATE.json',
 'research/round50/independent_review/prediction_audit_attempt002/AUDIT_RESULT.json']
def local(rel):
 p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
 assert 'FINAL' not in rel and 'round46/final' not in rel and 'full5912_gpu_inputs' not in rel
 return ROOT/p
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump_new(p,v):
 p=Path(p);assert not p.exists(),'Immutable stage artifact already exists; no automatic rewrite'
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');assert not t.exists()
 t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def member(folder,rel):
 folder=local(folder);s=read(folder/'SEALED.json');assert sha(folder/rel)==s['files'][rel]
 return folder/rel
def sealed_root(rel,expected_n):
 s=read(local(rel)/'SEALED.json');assert s['n']==expected_n
 return s
def require_full_gate(builder_auth):
 """Only saved aggregate receipt/summary/recheck; no GT rows or raw inputs."""
 assert builder_auth['full_gate_rule_sha256']==sha(local(FULL_RULE))
 rule=read(local(FULL_RULE));assert rule['state']=='ROOT_LOCKED_R50_FULL_ADVANCE_RULE_BEFORE_ANY_FULL_GT'
 assert rule['scientific_freeze_sha256']==sha(local(FULL_FREEZE)) and read(local(FULL_FREEZE))['selected_configuration']=='wideentry_oldW'
 assert rule['registered_full_gate']=={'candidate':'wideentry_oldW','scope':'ALL5912','metric':'two_platform_equal_weight_mIoU_percent','references':['old_G','source'],'strict_minimum_gain_pp':1e-10}
 assert rule['platform_or_case_zero_harm_gate'] is False and rule['no_new_configuration_selection_after_full_GT'] is True
 ss=local(FULL_SCORE)/'SEALED.json';rs=local(FULL_RECHECK)/'SEALED.json'
 assert sha(ss)==builder_auth['full_score_seal_sha256'] and sha(rs)==builder_auth['full_scalar_recheck_seal_sha256']
 receipt=read(member(FULL_SCORE,'SCORING_RECEIPT.json'));summary=read(member(FULL_SCORE,'TASK_SUMMARY.json'))
 check=read(member(FULL_RECHECK,'AUDIT_RESULT.json'));gate=read(member(FULL_RECHECK,'REGISTERED_FULL_GATE_RECHECK.json'))
 assert receipt['state']=='PASS_R50_SINGLE_LOCKED_FULL5912_THREE_ARM_TASK_SCORE' and receipt['n']==5912 and receipt['arms']==3 and receipt['task_values']==17736
 assert check['state']=='PASS_R50_INDEPENDENT_SAVED_FULL_SCALARS_AND_PRELOCKED_GATE_REAGGREGATION' and check['score_seal_sha256']==sha(ss)
 assert check['registered_full_gate_lock_sha256']==sha(local(FULL_RULE)) and check['full_gate_PASS'] is True
 assert gate['state']=='R50_FULL_PRELOCKED_ADVANCE_GATE_INDEPENDENTLY_RECOMPUTED_FROM_SAVED_SCALARS' and gate['full_score_seal_sha256']==sha(ss)
 assert gate['registered_gate_lock_sha256']==sha(local(FULL_RULE)) and gate['full_gate_PASS'] is True
 recomputed=read(member(FULL_RECHECK,'RECOMPUTED_TASK_SUMMARY.json'))
 m=recomputed['ALL5912']['balanced_mIoU_percent'];original=summary['ALL5912']['balanced_mIoU_percent']
 assert set(m)==set(original)=={'source','old_G','wideentry_oldW'} and all(type(x) in (float,int) and math.isfinite(x) for x in list(m.values())+list(original.values()))
 gains={ref:m['wideentry_oldW']-m[ref] for ref in ('old_G','source')}
 original_gains={ref:original['wideentry_oldW']-original[ref] for ref in ('old_G','source')}
 assert all(x>1e-10 for x in gains.values()),'STOP_FULL_GATE_NOT_POSITIVE_NO_DEV_PLAN_OR_GRANT'
 assert all(x>1e-10 for x in original_gains.values()) and gate['gate_predicate_agrees_between_independent_fmean_and_original_numpy_reduction'] is True
 assert gate['gains_vs_references_pp']==gains
 return {'state':'PASS_R50_FULL_GATE_FOR_SEPARATE_DEV_EPISODIC_PLANS','gains_vs_references_pp':gains,
  'original_numpy_gains_vs_references_pp':original_gains,'both_reduction_predicates_agree_and_PASS':True,
  'rule_sha256':sha(local(FULL_RULE)),'score_seal_sha256':sha(ss),'scalar_recheck_seal_sha256':sha(rs)}
def engineering_rows():
 rows=read(local(OLD));assert len(rows)==336 and len({r['id'] for r in rows})==336
 chosen=[min([r for r in rows if r['platform']==p],key=lambda r:hashlib.sha256((ENG_SALT+r['id']).encode()).hexdigest()) for p in ('drone','quad')]
 fp=local(E2_ROWS)
 if fp.exists():assert read(fp)==chosen
 else:dump_new(fp,chosen)
 return chosen
def engineering_bridge():
 c=sealed_root(RUNS['engineering_capture'],2);f=sealed_root(RUNS['engineering_fit'],2)
 assert c['state']=='R50_DEV_EPISODIC_TWO_SOURCE_VIEWS_CAPTURED_NO_OPTIMIZER' and c['grounder_forwards']==4 and c['source_unchanged'] is True
 assert f['state']=='R50_DEV_CURRENT_INPUT_1_3_STEP_MATCHING_ONLY_PREDICTIONS_NO_GT'
 cr=read(member(RUNS['engineering_capture'],'RECEIPTS_PRIVATE.json'));fr=read(member(RUNS['engineering_fit'],'RECEIPTS_PRIVATE.json'))
 assert len(cr)==len(fr)==2 and {r['id_PRIVATE'] for r in cr}=={r['id_PRIVATE'] for r in fr}
 assert all(r['source_A_boxes_scores_canonical_bitexact'] is True for r in cr)
 assert all(r['initial_source_and_G_queries_exact'] is True and r['zero_step_vs_literal_consensus_query_exact'] is True and r['group_A_G_literal_query_and_box_exact'] is True for r in fr)
 return {'state':'PASS_R50_ENGINEERING2_SOURCE_NATIVE_G_CONSENSUS_BRIDGES_NO_TASK_SCORE',
  'capture_seal_sha256':sha(local(RUNS['engineering_capture'])/'SEALED.json'),'fit_seal_sha256':sha(local(RUNS['engineering_fit'])/'SEALED.json')}
def pins(paths):return {p:sha(local(p)) for p in dict.fromkeys(paths)}
def stage_plan(stage,gate):
 inputs=BASE_INPUTS+[FULL_RULE,FULL_FREEZE,FULL_SCORE+'/SEALED.json',FULL_RECHECK+'/SEALED.json']
 plan={'state':'ROOT_LOCKED_R50_'+stage.upper(),'stage':stage,'output_root':RUNS[stage],
  'no_GT_newG_or_FINAL':True,'same_Waymo_source_and_GTfree_proxy':True,'canonical_R8_not_replaced':True,
  'full_advance_gate':gate,'allow_verified_no_parameter_update_branch':True,
  'matching_parameters':184960,'only_original_RankNet':True,'steps':[1,3],'lr':.0005,'weight_decay':0,'tau':.1,
  'source_buffers_modes_RNG_and_optimizer_reset':True,'B_sampling_salt':B_SALT,
  'no_new_loss_regularizer_teacher_or_layer_grid':True,'timers_remain_off':True}
 if stage.startswith('engineering_'):
  engineering_rows();inputs.append(E2_ROWS);plan.update(n=2,mode='engineering2',rows_path=E2_ROWS,engineering_selection_salt=ENG_SALT,no_task_scoring=True)
 else:plan.update(n=336,mode='DEV336')
 if stage in ('engineering_capture','dev_capture'):
  if stage=='dev_capture':
   plan['engineering_bridge']=engineering_bridge();inputs += [RUNS[s]+'/SEALED.json' for s in ('engineering_capture','engineering_fit')]
  plan.update(readout='native_GTfree_first_lexicon_noun_fallback_all_content',boxes='canonical_old336_R8_source256_raw6_no_updates',
   capture_both_views_for_every_registered_row=True,initial_head_native_max_abs_bound=2e-5,
   resource_mode='FROZEN_FULL_SOURCE_CAPTURE_ONLY_NO_OPTIMIZER',max_grounder_forwards=4 if stage=='engineering_capture' else 672)
 elif stage in ('engineering_fit','dev_fit'):
  capstage='engineering_capture' if stage=='engineering_fit' else 'dev_capture';sealed_root(RUNS[capstage],plan['n']);inputs += [RUNS[capstage]+'/SEALED.json',PLANS[capstage]]
  plan.update(capture_root=RUNS[capstage],arm_names=ARMS,initial_head_native_max_abs_bound=2e-5,
   resource_mode='DETACHED_FROZEN_FEATURE_MATCHING_HEAD_FIT_ONLY',max_optimizer_steps=3*plan['n'],new_grounder_forwards=0,new_visual_calls=0)
 elif stage=='dev_closure':
  for s in ('dev_capture','dev_fit'):sealed_root(RUNS[s],336);inputs += [RUNS[s]+'/SEALED.json',PLANS[s]]
  plan.update(capture_root=RUNS['dev_capture'],fit_root=RUNS['dev_fit'],selection_salt=CLOSURE_SALT,n_per_platform=1,
   cached_full_max_abs_bound=2e-5,max_grounder_forwards=8,resource_mode='ACTUAL_INSTALLED_EPISODIC_HEAD_FULL_GROUNDER_CLOSURE_NO_OPTIMIZER')
 elif stage=='independent_audit':
  for s,n in (('dev_capture',336),('dev_fit',336)):
   sealed_root(RUNS[s],n);inputs += [RUNS[s]+'/SEALED.json',PLANS[s]]
  assert (local(RUNS['dev_closure'])/'SEALED.json').exists();inputs += [RUNS['dev_closure']+'/SEALED.json',PLANS['dev_closure']]
  plan.update(arms=ARMS,capture_root=RUNS['dev_capture'],fit_root=RUNS['dev_fit'],closure_root=RUNS['dev_closure'],
   capture_plan=PLANS['dev_capture'],fit_plan=PLANS['dev_fit'],closure_plan=PLANS['dev_closure'],
   existing_DEV_prediction_audit='research/round50/independent_review/prediction_audit_attempt002/AUDIT_RESULT.json',
   no_GT_full_grounder_VLM_detector_or_optimizer=True,resource_mode='GPU_TINY_HEAD_FORWARD_ONLY_NO_FIT',
   runtime_versions={p:importlib.metadata.version(p) for p in ('torch','numpy')},tiny_head_p_max_abs_bound=2e-5,
   tiny_head_z_max_abs_bound=2e-4,trace_scalar_abs_bound=2e-5,parameter_norm_abs_bound=1e-6,parameter_norm_relative_bound=1e-5)
 elif stage=='single_score':
  auditpath=RUNS['independent_audit']+'/AUDIT_RESULT.json';ar=read(member(RUNS['independent_audit'],'AUDIT_RESULT.json'))
  assert ar['state']=='PASS_R50_INDEPENDENT_EPISODIC_NUMERIC_AND_CONTRACT_AUDIT_NO_GT' and ar['no_GT_read'] is True and ar['n']==336
  changed=ar['nonzero_parameter_element_count'];assert type(changed) is int and changed>=0
  if changed:assert ar['installed_closure_requirement_satisfied'] is True
  else:assert ar['all_saved_heads_source_bytes_identical'] is True and ar['all_no_update_outputs_equal_s0'] is True and ar['installed_closure_state']=='NOT_APPLICABLE_VERIFIED_NO_PARAMETER_UPDATE'
  if not changed:assert ar['installed_closure_requirement_satisfied'] is False
  assert ar['prediction_seal_sha256']==sha(local(RUNS['dev_fit'])/'SEALED.json')
  inputs += [RUNS['dev_fit']+'/SEALED.json',RUNS['independent_audit']+'/SEALED.json',auditpath,
   'research/round50/independent_score/dev336_attempt001/SEALED.json',
   'crossplatform/round5/baseline_engineering/PREPARATION_RECEIPT.json']
  plan.update(arms=ARMS,task_values=2016,prediction_root=RUNS['dev_fit'],independent_audit_result=auditpath,
   selection_rule='ALL336_balanced_max_step1_or3_exact_tie1_strictgain_vs_consensus0_and_literalAwideG',
   resource_mode='SINGLE_POSTSEAL_TASK_SCORE_ONLY_NO_MODELS',max_GT_scoring_passes=1,
   no_GT_newG_or_FINAL=False,no_newG_or_protected_FINAL_inputs=True,
   task_GT_access='ONLY_POSTSEAL_ROOT_AUTHORIZED_2016_IOU_SCORE_NO_FIT_OR_CONDITION_GATE')
 plan['programs_sha256']=pins(CODE+[Path(__file__).relative_to(ROOT).as_posix()]);plan['inputs_sha256']=pins(inputs)
 return plan
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=STAGES,required=True)
 p.add_argument('--root-builder-authorization',required=True);p.add_argument('--emit-execution-grant',action='store_true')
 p.add_argument('--grant-existing-plan',action='store_true');a=p.parse_args()
 # The builder itself has no authority to invent a root grant.
 authpath=Path(a.root_builder_authorization).resolve();assert authpath.is_relative_to(EXP)
 builder=read(authpath);assert builder['state']=='ROOT_AUTHORIZED_R50_EPISODIC_STAGE_LOCK_BUILDER_AFTER_FULL_GATE'
 assert builder['program_sha256']==sha(__file__) and a.stage in builder['allowed_stages']
 if a.emit_execution_grant:assert builder.get('allow_stage_execution_grant_emission') is True
 if a.grant_existing_plan:assert a.emit_execution_grant
 gate=require_full_gate(builder)
 review=read(local(REVIEW));assert review['state']=='PASS_R50_EPISODIC_REVISED_DRAFT_STATIC_REVIEW_ONLY_NOT_EXECUTION_AUTHORIZATION'
 for rel,h in review['programs_sha256'].items():assert sha(local(rel))==h,rel
 assert Path(sys.prefix).resolve()==(ROOT/'crossplatform/env').resolve(),'Use original source CPython3.11 metadata/runtime'
 planpath=local(PLANS[a.stage]);authout=local(AUTHS[a.stage])
 if a.grant_existing_plan:
  assert builder['existing_plan_sha256']==sha(planpath);plan=read(planpath)
  assert plan['stage']==a.stage and plan['output_root']==RUNS[a.stage] and plan['full_advance_gate']==gate
  for group in ('programs_sha256','inputs_sha256'):
   for rel,h in plan[group].items():assert sha(local(rel))==h,rel
 else:
  plan=stage_plan(a.stage,gate);plan['root_builder_authorization_sha256']=sha(authpath);dump_new(planpath,plan)
 execution={'state':AUTH_STATE[a.stage],'plan_sha256':sha(planpath)}
 if a.stage=='independent_audit':execution.update(program_sha256=sha(local(PROGRAM[a.stage])),audit_plan_sha256=sha(planpath),prediction_seal_sha256=sha(local(RUNS['dev_fit'])/'SEALED.json'))
 if a.stage=='single_score':execution.update(program_sha256=sha(local(PROGRAM[a.stage])),prediction_seal_sha256=sha(local(RUNS['dev_fit'])/'SEALED.json'),independent_audit_sha256=sha(local(RUNS['independent_audit'])/'AUDIT_RESULT.json'))
 emit=bool(a.emit_execution_grant and builder.get('allow_stage_execution_grant_emission') is True)
 if emit:
  execution['root_grant_emission_builder_authorization_sha256']=sha(authpath);dump_new(authout,execution)
 else:dump_new(authout.with_name(authout.stem+'_DRAFT.json'),{'state':'DRAFT_NOT_EXECUTION_AUTHORIZATION','intended_root_authorization':execution})
 cmd=['crossplatform/env/bin/python',PROGRAM[a.stage]]
 if a.stage!='independent_audit':cmd += ['--plan',str(planpath),'--authorization',str(authout)]
 resource='cpu' if a.stage=='single_score' else 'gpu'
 wrapper=['crossplatform/env/bin/python','tools/manual_r50_guard.py','--resource',resource,'--timeout-seconds',str(TIMEOUT[a.stage]),'--',*cmd]
 receipt_suffix='_GRANT_EMISSION_RECEIPT.json' if a.grant_existing_plan else '_CONSTRUCTION_RECEIPT.json'
 dump_new(planpath.with_name(planpath.stem+receipt_suffix),{'state':'R50_EPISODIC_STAGE_PLAN_BUILT_NOT_RUN',
  'stage':a.stage,'plan_sha256':sha(planpath),'execution_grant_emitted':emit,'command_not_executed':wrapper,
  'full_gate':gate,'new_model_optimizer_or_task_score_calls':0,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print(json.dumps({'stage':a.stage,'plan':str(planpath),'execution_grant_emitted':emit,'not_executed':wrapper}))
if __name__=='__main__':main()
