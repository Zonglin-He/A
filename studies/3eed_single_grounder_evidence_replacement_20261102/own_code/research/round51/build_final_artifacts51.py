"""Build report/export plans from completed R51 saved evidence, no new scores.

Called only after the locked full evaluation and independent saved audit.
This script never opens media, GT, original sample registrations or weights.
It does not publish, choose a method, launch models, or execute the exporter.
"""
from pathlib import Path
import argparse, hashlib, json

ROOT=Path(__file__).resolve().parents[2]
PREFIX='studies/3eed_single_grounder_evidence_replacement_20261102'
FINAL='research/round51/final_inputs_attempt001'
REPORT='research/round51/final_report_attempt001'
RUNS={
 'INITIAL_DEV':{'score_root':'research/round51/score_dev336_attempt001',
                'independent_prediction_audit':'research/round51/independent_audit/dev336_attempt001/AUDIT_RESULT.json',
                'saved_scalar_audit':'research/round51/independent_audit/saved_dev336_attempt001/AUDIT_RESULT.json'},
 'REPAIR_DEV':{'score_root':'research/round51/score_grouped_dev336_attempt001',
               'independent_prediction_audit':'research/round51/independent_audit/grouped_frames336_attempt001/AUDIT_RESULT.json',
               'saved_scalar_audit':'research/round51/independent_audit/saved_grouped_dev336_attempt001/AUDIT_RESULT.json'},
 'FULL':{'score_root':'research/round51/score_full5912_attempt001',
         'independent_prediction_audit':'research/round51/independent_audit/full5912_attempt001/AUDIT_RESULT.json',
         'saved_scalar_audit':'research/round51/independent_audit/saved_full5912_attempt001/AUDIT_RESULT.json'}}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);assert not p.exists(),p
 p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def pins(paths):return {s:sha(ROOT/s) for s in paths}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['report_plan','publication_plan'],required=True)
 ap.add_argument('--base-snapshot',default='research/round51/publication/BASE_SNAPSHOT_PRIVATE.json')
 a=ap.parse_args();evidence=read(ROOT/'research/round51/FULL_EVALUATION_ONESHOT_RECEIPT.json')
 assert evidence['state']=='COMPLETE_LOCKED_FULL_FIVE_ARM_INDEPENDENTLY_AUDITED_SINGLE_SCORE'
 for r in RUNS.values():
  for k in ('independent_prediction_audit','saved_scalar_audit'):
   assert read(ROOT/r[k])['state'].startswith('PASS_')
 for name,r in RUNS.items():
  s=read(ROOT/r['score_root']/'SEALED.json')
  for rel,h in s['files'].items():assert sha(ROOT/r['score_root']/rel)==h
 if a.mode=='report_plan':
  out=ROOT/FINAL;assert not out.exists();out.mkdir()
  cost_paths={
   'dev_raw':'crossplatform/round51/prepared336_attempt001/COSTS.json',
   'dev_capture':'crossplatform/round51/capture336_attempt001/COUNTS.json',
   'dev_ref_repair':'crossplatform/round51/reference336_headspan_attempt001/COUNTS.json',
   'diagnostic_target':'crossplatform/round51/visual_targets336_attempt002/COUNTS.json',
   'diagnostic_reference':'crossplatform/round51/visual_references336_attempt001/COUNTS.json',
   'initial_safe':'crossplatform/round51/safe336_attempt001/COSTS.json',
   'dev_grouped':'crossplatform/round51/grouped336_attempt001/COSTS.json',
   'dev_noassoc':'crossplatform/round51/noassociation336_attempt001/COSTS.json',
   'full_raw':'crossplatform/round51/prepared5912_attempt001/COSTS.json',
   'full_ref':'crossplatform/round51/reference5912_headspan_attempt001/COUNTS.json',
   'full_grouped':'crossplatform/round51/grouped5912_attempt001/COSTS.json',
   'full_noassoc':'crossplatform/round51/noassociation5912_attempt001/COSTS.json',
   'root_failed_visual_recovery':'research/round51/experiment/ROOT_DIAGNOSTIC_TARGET_VISUAL_DEV336_ATTEMPT002_AUTHORIZATION.json',
   'root_saved_audit_recovery':'research/round51/experiment/SAVED_GROUPED_DEV336_SCALAR_AUDIT_ROOT_RECOVERY.json',
   'root_coordinator_recovery':'research/round51/FULL_COORDINATOR_LOCK_RACE_RECOVERY.json'}
  c={k:read(ROOT/v) for k,v in cost_paths.items()}
  assert c['diagnostic_target']['new_calls_this_process']==67696
  assert c['diagnostic_reference']['new_calls_this_process']==830
  assert c['full_ref']['reference_forwards_all_processes']==10465
  assert c['full_ref']['source_subjects_reused_bitexact']==5912 and c['full_ref']['model_unchanged']
  costs={'counts':{
    'DEV_observations':336,'full_observations':5912,
    'diagnostic_Qwen_new_completed_target_calls':67696,
    'diagnostic_Qwen_new_completed_reference_calls':830,
    'diagnostic_Qwen_prior_failed_attempt_valid_calls':4,
    'diagnostic_Qwen_all_actual_completed_including_failed_attempt':68530,
    'DEV_grounder_capture_forwards':c['dev_capture']['physical_grounder_forwards_all_processes'],
    'DEV_reference_repair_forwards':c['dev_ref_repair']['new_reference_forwards'],
    'full_reference_calls':10465,'full_reference_UNKNOWN_heads':c['full_ref']['UNKNOWN_references_not_negative'],
    'full_subject_calls_reused':5912,'full_new_subject_forwards':0,
    'final_method_new_Mask_R_CNN_forwards':0,'final_method_new_Qwen_forwards':0,
    'final_method_new_DINO_forwards':0,'B_capture_forwards':0,'optimizer_steps':0,
    'source_training_epochs_this_round':0},
    'timing_seconds':{
      'DEV_raw_preparation':c['dev_raw']['seconds'],
      'DEV_source_reference_capture':c['dev_capture']['seconds_this_process'],
      'DEV_reference_headspan_repair':c['dev_ref_repair']['seconds'],
      'diagnostic_target_Qwen':c['diagnostic_target']['seconds'],
      'diagnostic_reference_Qwen':c['diagnostic_reference']['seconds'],
      'DEV_initial_safe_G':c['initial_safe']['seconds'],
      'DEV_grouped_G':c['dev_grouped']['seconds'],
      'DEV_noassociation':c['dev_noassoc']['seconds'],
      'full_raw_preparation':c['full_raw']['seconds'],
      'full_reference_capture':c['full_ref']['seconds_this_process'],
      'full_grouped_G':c['full_grouped']['seconds'],
      'full_noassociation':c['full_noassoc']['seconds']},
    'peak_bytes':{
      'DEV_source_model':c['dev_capture']['peak_bytes'],
      'diagnostic_target_Qwen':c['diagnostic_target']['peak_bytes'],
      'diagnostic_reference_Qwen':c['diagnostic_reference']['peak_bytes'],
      'full_reference_model':c['full_ref']['peak_bytes']},
    'flags':{'source_given_as_cache':True,'not_per_arm_independent_end_to_end_benchmark':True,
             'not_full_pipeline_peak_memory':True,'final_external_vision_cache_dependency':False,
             'diagnostic_hybrid_external_evidence_allowed':True,'TTA_gate_failed':True,
             'TTA_parameter_effectiveness_tested':False,'source_model_unchanged':True,
             'all_scenes_historically_exposed':True,'no_semantic_instance_identity_certification':True}}
  history={'events':[
    {'stage':'DEV_REFERENCE_HEADSPAN','attempt':1,'status':'COMPLETED_REPAIRED','repair_type':'pure_text_head_mask','completed_calls':573,'before_GT':True,'producer_predictions_changed':True,'prior_attempt_preserved':True},
    {'stage':'DEV_QWEN_TARGET','attempt':1,'status':'FAILED_PRESERVED','failure_type':'one_pixel_image_channel_inference','valid_outputs':4,'completed_calls':4,'before_GT':True,'prior_attempt_preserved':True},
    {'stage':'DEV_QWEN_TARGET','attempt':2,'status':'COMPLETED','repair_type':'explicit_channels_last_processor','completed_calls':67696,'before_GT':True,'tolerance_changed':False,'prior_attempt_preserved':True},
    {'stage':'DEV_RELATION_FIDELITY','attempt':2,'status':'COMPLETED','repair_type':'restore_original_one_plus_total_denominator','before_GT':True,'producer_predictions_changed':True,'prior_attempt_preserved':True},
    {'stage':'DEV_INITIAL_ASSEMBLY','attempt':1,'status':'FAILED_BEFORE_ROWS','failure_type':'missing_root_noassociation_flag','valid_outputs':0,'before_GT':True,'prior_attempt_preserved':True},
    {'stage':'DEV_CARRIER_REPAIR','attempt':1,'status':'COMPLETED_GATE_FAILED','repair_type':'whole_proposal_point_medoid','before_GT':False,'producer_predictions_changed':True,'prior_attempt_preserved':True},
    {'stage':'REPAIR_SAVED_SCALAR_AUDIT','attempt':1,'status':'FAILED_BEFORE_OUTPUT','failure_type':'root_authorization_state_string','valid_outputs':0,'before_GT':False,'tolerance_changed':False,'prior_attempt_preserved':True},
    {'stage':'FULL_COORDINATOR','attempt':1,'status':'STOPPED_WHILE_WAITING','repair_type':'wait_existing_CPU_guard_release','before_GT':True,'producer_predictions_changed':False,'tolerance_changed':False,'prior_attempt_preserved':True}]}
  closure={'TTA_executed':False,'TTA_gate_passed':False,'B_capture_executed':False,
           'selected_factor':.5,'selected_method':'grouped_replace_all_c05',
           'full_mandatory_bounded_evaluation':True,'TTA_parameter_failure_inferred':False}
  dump(out/'NORMALIZED_COSTS.json',costs);dump(out/'NORMALIZED_HISTORY.json',history);dump(out/'CLOSED_TTA_DECISION.json',closure)
  p={'state':'ROOT_LOCKED_R51_SAVED_ONLY_FINAL_REPORT','prefix':PREFIX,
     'no_GT_media_models_or_future_artifacts':True,'score_runs':RUNS,
     'closed_decision':FINAL+'/CLOSED_TTA_DECISION.json','normalized_costs':FINAL+'/NORMALIZED_COSTS.json',
     'normalized_history':FINAL+'/NORMALIZED_HISTORY.json','cost_history_source_artifacts':cost_paths,
     'output_directory':REPORT}
  paths=set(cost_paths.values())|{p[k] for k in ('closed_decision','normalized_costs','normalized_history')}
  for r in RUNS.values():
   paths|={r['independent_prediction_audit'],r['saved_scalar_audit']}
   paths|={r['score_root']+'/'+s for s in ['SEALED.json','SCORING_RECEIPT.json','TASK_SUMMARY.json','ALL_SCENE_SUMMARIES_PRIVATE.json','SHARED_CALIBRATION_SELECTION.json']}
  p['inputs_sha256']=pins(sorted(paths));pp=ROOT/'research/round51/experiment/FINAL_SAVED_REPORT_PLAN_LOCK.json';dump(pp,p)
  dump(pp.with_name('FINAL_SAVED_REPORT_AUTHORIZATION.json'),{'state':'ROOT_AUTHORIZED_R51_SAVED_ONLY_FINAL_REPORT','plan_sha256':sha(pp),'program_sha256':sha(ROOT/'research/round51/finalize51.py')})
 else:
  assert read(ROOT/REPORT/'SEALED.json')
  scalar_runs=[{'label':name,'score_root':r['score_root'],'independent_audit':r['independent_prediction_audit'],'saved_scalar_audit':r['saved_scalar_audit']} for name,r in RUNS.items()]
  code=[]
  for directory in ['crossplatform/round51','research/round51']:
   for f in sorted((ROOT/directory).glob('*.py')):
    if '_pre_' in f.name or f.name.endswith('_revision001.py'):continue
    code.append(f.relative_to(ROOT).as_posix())
  code.append('tools/manual_r51_guard.py')
  p={'prefix':PREFIX,'no_network_models_GT_or_media':True,'scalar_runs':scalar_runs,
     'private_registry':'research/round50/final_inputs_attempt001/PRIVATE_PRIVACY_TOKEN_REGISTRY.json',
     'publication_base_snapshot':a.base_snapshot,'report':REPORT+'/REPORT.md','decision':REPORT+'/DECISION.json',
     'public_fixed_configuration':REPORT+'/public_fixed_config.json','normalized_costs':FINAL+'/NORMALIZED_COSTS.json',
     'normalized_history':FINAL+'/NORMALIZED_HISTORY.json','own_code_allowlist':code}
  paths=set(code)|{p[k] for k in ['private_registry','publication_base_snapshot','report','decision','public_fixed_configuration','normalized_costs','normalized_history']}
  for r in scalar_runs:
   paths|={r['independent_audit'],r['saved_scalar_audit'],r['score_root']+'/SEALED.json'}
   paths|={r['score_root']+'/'+s for s in read(ROOT/r['score_root']/'SEALED.json')['files']}
  p['inputs_sha256']=pins(sorted(paths));pp=ROOT/'research/round51/experiment/LOCAL_PUBLIC_EXPORT_PLAN_LOCK.json';dump(pp,p)
  dump(pp.with_name('LOCAL_PUBLIC_EXPORT_AUTHORIZATION.json'),{'state':'ROOT_AUTHORIZED_R51_LOCAL_SAVED_SCALAR_WHITELIST_EXPORT',
     'plan_sha256':sha(pp),'program_sha256':sha(ROOT/'research/round51/publish51.py'),
     'private_registry_sha256':sha(ROOT/p['private_registry']),'publication_base_snapshot_sha256':sha(ROOT/a.base_snapshot)})
 print('BUILT_COMPLETED_SAVED_ARTIFACT_PLAN',a.mode,flush=True)

if __name__=='__main__':main()
