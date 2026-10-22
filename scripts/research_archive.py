"""Check and snapshot the sole maintained method and its saved evidence."""
import argparse,json,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REQUIRED=['docs/tube_fusion.md','scripts/tube_fusion/config.json','results/tube_fusion_hc64/report.json','results/tube_fusion_hc64/report.md','results/tube_fusion_hc64/ANALYSIS.md','results/tube_fusion_hc64/signed_changes.tsv','results/tube_fusion_hc64/structure.json','results/boxfixed_native_smoke.json','results/tube_temporal_hc64/report.json','results/tube_temporal_hc64/report.md','results/tube_temporal_hc64/ANALYSIS.md','results/tube_temporal_hc64/signed_changes.tsv','results/tube_temporal_hc64/structure.json','configs/legacy_opd.json','docs/boxfixed_temporal.md','results/boxfixed_migration.json','README.md','AGENTS.md','configs/default.json','docs/history.md','docs/method.md','docs/evaluation.md','docs/assets.md','docs/assets.json','docs/third_party.json','docs/migration.json','results/report.md','results/evaluation.json','results/runtime.json','results/cost.json','results/migration_checks.json','results/score_regression.json','results/cleanup_checks.json','docs/data_integrity.md','results/data_integrity.json','docs/cleanup.json','docs/publication.json','docs/formal_evaluation.md','docs/cgstvg.md','results/formal/cohorts.json','results/formal/preflight.json','results/formal/report.md','results/formal/results.json','docs/split_audit.md','results/formal/split_audit.json']
REQUIRED += ['docs/interval_pool.md','docs/temporal_audit.md','docs/temporal_sources.md',
 'scripts/interval_pool/config.json','results/interval_pool/ANALYSIS.md','results/interval_pool/decision.json',
 'results/interval_pool/cohort_provenance.json','results/interval_pool/engineering.json',
 'results/interval_pool/development/error_attribution.md','results/interval_pool/development/diagnostics.json',
 'results/interval_pool/development/signed_errors.tsv']
for panel in ['intervention_hc64','recheck_hc2','recheck_vidstg']:
 REQUIRED += [f'results/interval_pool/{panel}/{name}'for name in ['report.json','report.md','signed_changes.tsv']]
REQUIRED += ['configs/tuned_boxfixed_v1.json','docs/tuning_v1.md',
 'results/boxfixed_tuning_v1/ANALYSIS.md','results/boxfixed_tuning_v1/summary.json',
 'results/boxfixed_tuning_v1/protocol.json','results/boxfixed_tuning_v1/selected.json',
 'results/boxfixed_tuning_v1/decisions.jsonl','results/boxfixed_tuning_v1/signed_trials.tsv',
 'results/boxfixed_tuning_v1/cache_regression.json','results/boxfixed_tuning_v1/confirmation_hc2.json',
 'results/boxfixed_tuning_v1/confirmation_vidstg.json']
REQUIRED += ['docs/temporal_signal_audit.md'] + [f'results/temporal_signal_audit_v1/{name}' for name in ['report.json','report.md','protocol.json','signed_queries.tsv','sensitivity.json','verification.json','association.svg']]
REQUIRED += ['docs/joint_boundary_readout.md'] + [f'results/joint_boundary_readout_{v}/{name}' for v in ['v1','v2'] for name in ['report.json','report.md','decision.json','protocol.json','signed_queries.tsv','verification.json','formal_guard.json']]
REQUIRED += ['docs/generic_tube_readout.md'] + [f'results/{trial}/{name}' for trial in ['interval_state_readout_v1','action_interval_readout_v1','rank_interval_readout_v1','generic_tube_readout_v1'] for name in ['report.json','report.md','decision.json','protocol.json','signed_queries.tsv']]
REQUIRED += ['results/generic_tube_readout_v1/verification.json','results/interval_state_readout_v1/development_case_audit.json']
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def check():
 errors=[];warnings=[]
 for name in REQUIRED:
  if not(ROOT/name).is_file():errors.append('Missing '+name)
 for f in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]:
  for target in re.findall(r'\]\(([^)]+)\)',f.read_text()):
   target=target.split('#',1)[0]
   if not target or '://'in target:continue
   if not(f.parent/target).exists():errors.append(f'{f.relative_to(ROOT)}: missing link {target}')
 if not errors:
  cfg=json.loads((ROOT/'configs/default.json').read_text());r=json.loads((ROOT/'results/evaluation.json').read_text())
  assert cfg['method']=='BoxFixed' and cfg['temporal']['default']=='strong'
  legacy=json.loads((ROOT/'configs/legacy_opd.json').read_text())
  assert [(legacy['datasets'][ds]['lr'],legacy['datasets'][ds]['tau'],legacy['datasets'][ds]['steps'],legacy['datasets'][ds]['initialization_gamma'],legacy['datasets'][ds]['output_alpha'])for ds in ['hc2','vidstg']]==[(.014,.14,40,.5,1.),(.01,.5,30,1.,1.)]
  for ds in ['hc2','vidstg']:
   assert not {'lr','tau','steps','samples','scales','state_shape'}&set(cfg['datasets'][ds])
   assert cfg['datasets'][ds]['E_calibration']==legacy['datasets'][ds]['E_calibration']
  for ds,g in r['datasets'].items():
   assert g['parents']==g['queries']==len(g['rows'])==128
   for arm,m in g['means'].items():
    for k,v in m.items():
     x=[row['values'][arm][k]for row in g['rows']if row['values'][arm][k]is not None];assert abs(sum(x)/len(x)-v)<1e-12
  decision=json.loads((ROOT/'results/interval_pool/decision.json').read_text())
  passed=all(all(g.values())for g in decision['direction_gates'].values())
  assert decision['mix_passed_preregistered_two_direction_gate']==passed
  assert decision['selected_research_temporal']==('Mix_MBR'if passed else'Global6_Hull_MBR')
  for panel in ['intervention_hc64','recheck_hc2','recheck_vidstg']:
   saved=json.loads((ROOT/f'results/interval_pool/{panel}/report.json').read_text())
   assert saved['proof']['all_predictions_sealed_before_GT']and saved['proof']['official_dense_max_absolute_error']<2e-10
  tuning=json.loads((ROOT/'results/boxfixed_tuning_v1/summary.json').read_text())
  profile=json.loads((ROOT/'configs/tuned_boxfixed_v1.json').read_text())
  assert tuning['actual_trials']=={'hc2':14,'vidstg':17}
  for ds in ['hc2','vidstg']:
   c=json.loads((ROOT/f'results/boxfixed_tuning_v1/confirmation_{ds}.json').read_text())
   assert c['proof']['all64_predictions_sealed_before_GT']
   assert profile['datasets'][ds]==c['decision']['final_params']
   assert not c['decision']['accepted']
  audit=json.loads((ROOT/'results/temporal_signal_audit_v1/report.json').read_text())
  verify=json.loads((ROOT/'results/temporal_signal_audit_v1/verification.json').read_text())
  assert audit['proof']['all_128_label_free_signals_before_labels'] and audit['proof']['hidden_layer_checks']==1536
  assert audit['proof']['maximum_CPU_hidden_absolute_error']<1e-4
  assert not audit['proof']['CUDA_initialized'] and audit['proof']['new_expert']==audit['proof']['new_encoder']==audit['proof']['parameter_updates']==0
  assert verify['status']=='pass' and verify['independent_rank_sum_checks']==245 and verify['rank_sum_max_error']==0
  assert verify['frozen_formal_sources_unchanged']==434
  for version,count in [('v1',2304),('v2',3328)]:
   joint=json.loads((ROOT/f'results/joint_boundary_readout_{version}/report.json').read_text())
   assert joint['proof']['all128_predictions_before_GT'] and joint['proof']['official_dense_checks']==count
   assert joint['proof']['official_dense_max_error']<2e-10 and joint['proof']['network_calls']==0
   assert not joint['decision']['validated_lightweight_candidate'] and not joint['decision']['production_changed']
  generic=json.loads((ROOT/'results/generic_tube_readout_v1/report.json').read_text())
  verified=json.loads((ROOT/'results/generic_tube_readout_v1/verification.json').read_text())
  assert generic['proof']['network_calls']==generic['proof']['GPU_calls']==generic['proof']['parameter_updates']==0
  assert generic['proof']['constant_quality_temporal_utility_max_error']<2e-12
  assert not generic['decision']['validated_candidate'] and not generic['decision']['production_changed']
  assert verified['status']=='pass' and verified['frozen_formal_sources_unchanged']==434
  assert verified['unchanged_source_baseline_and_candidate_arrays']==4096
  for trial in ['interval_state_readout_v1','action_interval_readout_v1','rank_interval_readout_v1','generic_tube_readout_v1']:
   saved=json.loads((ROOT/f'results/{trial}/report.json').read_text())
   assert saved['proof']['all128_new_predictions_before_new_GT_scores'] and saved['proof']['official_dense_max_error']<2e-10
   assert not saved['decision']['validated_candidate']
  snap=ROOT/'docs/sources.json'
  if snap.exists():
   old=json.loads(snap.read_text())['files'];changed=[p for p,h in old.items()if not(ROOT/p).is_file()or sha(ROOT/p)!=h]
   if changed:warnings.append('Snapshot stale; changed '+str(len(changed))+' files')
 result=dict(status='pass'if not errors else'fail',errors=errors,warnings=warnings,scope='current file links/configuration and independent aggregation of saved scores; no inference or new experiment');print(json.dumps(result,indent=2));return result

def snapshot():
 result=check();assert not result['errors'];files={}
 for p in REQUIRED+[str(f.relative_to(ROOT))for folder in ['stvg','tools','tests','scripts']for f in (ROOT/folder).rglob('*.py')]:files[p]=sha(ROOT/p)
 dest=ROOT/'docs/sources.json';dest.write_text(json.dumps(dict(status='snapshot',files=files),indent=2)+'\n');print('Snapshot written:',len(files),'current sources')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['check','snapshot']);a=p.parse_args()
 if a.action=='snapshot':snapshot()
 else:raise SystemExit(0 if check()['status']=='pass'else 1)
