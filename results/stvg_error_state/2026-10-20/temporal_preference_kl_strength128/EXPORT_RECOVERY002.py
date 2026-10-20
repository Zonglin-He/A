"""Export-only recovery of already complete sealed scores; no GT or adaptation rerun.
Original scoring ended after RESULTS/DECISION/ROOT_AUDIT, because a global path
replacement incorrectly requested the prior experiment's COST_REVISION001.json.
This exporter reads the actual prior COST.json and retains the failed invocation.
"""
from scripts.score_stvg_temporal_preference_kl_strength128_revision001 import *
import datetime,subprocess
sys.addaudithook(forbid_GT)
plan=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');result=read(P/'RESULTS_PUBLIC.json');decision=read(P/'DECISION_PUBLIC.json')
assert len(result['rows'])==128 and read(P/'ROOT_AUDIT.json')['status']=='pass'and read(P/'ROOT_SAVED_SCORE_REAGGREGATION_PUBLIC.json')['status']=='pass'
assert all(sha(R/f)==h for f,h in plan['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files'])
means=result['means'];stats=result['paired'];process=result['process'];decomp=result['contribution_decomposition'];audit=read(P/'ROOT_AUDIT.json');cost=read(D/'COST_REVISION001.json')
failure=next(D.glob('SCORING_FAILURE_*.json'));assert read(failure)['type']=='FileNotFoundError'and 'stvg_temporal_strong_relative_preference_opd128_v1/COST_REVISION001.json'in read(failure)['message']
raw_birth=subprocess.check_output(['stat','-c','%w',str(D/'audit_score_revision001.log')],text=True).strip();start_epoch=datetime.datetime.fromisoformat(raw_birth.replace(' +','+')).timestamp();end_epoch=int(failure.stem.rsplit('_',1)[1])/1e9
fullcost=dict(**cost,CPU_audit_and_scoring_seconds=None,CPU_audit_score_invocation_wall_seconds_from_filesystem=end_epoch-start_epoch,
 wall_timing_source=dict(log_birth=raw_birth,export_failure_epoch=end_epoch,scope='whole scorer invocation to export failure, includes Python imports, not original perf_counter interval'),
 scorer_process_exit_code=1,all_scientific_scores_and_audits_completed_before_export_error=True,
 prior_coefficient1_experiment_cost=read(OLD/'COST.json'),original_source_cache_generation_cost=read(R/'artifacts/stvg_temporal_local_motion_completion128_v1/COST.json'),publication_archive_excluded=True)
from main_method.runtime import resource_guard
def check():resource_guard()
put(P/'COST_PUBLIC.json',fullcost)
lines=['# Finite Strong KL protection-strength verification: full HC128','','Only the continuous global Strong KL coefficient changes. Native offset evidence, binary agreement target, learner initialization/state, sampling/seed streams, optimizer40Adam0.1, spatial arrays and final MBR are frozen. New actual Live fits at0.1/0.3 for128parents x3seeds;1 is reused from the completed previous experiment. This is development sensitivity/selection, not confirmation. No new Direct/Fixed experiment and no claim of independent on-policy improvement at newly selected strength.','','| Coefficient | complete vIoU percent | tIoU percent | full-v delta vs Strong pp [CI95] | positive contribution pp | negative contribution pp | below-5pp parents |','|---|---:|---:|---|---:|---:|---:|',f"| Strong | {means['Strong']['v']*100:.6f} | {means['Strong']['t']*100:.6f} | reference | - | - | - |"]
for n in ['KL1','KL0p3','KL0p1']:
 v=means[n];st=stats['Mean_'+n+'_minus_Mean_Strong'];t=st['tails'];lines.append(f"| {result['coefficients'][n]} | {100*v['v']:.6f} | {100*v['t']:.6f} | {st['v']['mean']:+.6f} {st['v']['CI95']} | {t['positive_contribution_pp']:+.6f} | {t['negative_contribution_pp']:+.6f} | {t['below_minus5']} |")
lines+=['','| Comparison | complete-v pp [CI95] | tIoU pp [CI95] |','|---|---|---|']
for n in COEFFICIENTS:
 st=stats['Mean_'+n+'_minus_Mean_KL1'];lines.append(f"| {n} minus KL1 | {st['v']['mean']:+.6f} {st['v']['CI95']} | {st['t']['mean']:+.6f} {st['t']['CI95']} |")
lines+=['','All128 parents retained, including no-change/time-no-overlap cases. Match temporal/spatial seeds then mean within parent; bootstrap parents10000times, not384 independent videos. Fixed-unobserved actual-node sIoU and all512 spatial arrays remain exact. Posthoc sign groups never choose output/feedback/parameters online. Development selection maximizes complete-v only, exact ties choose stronger protection; this does not promote a method or authorize more coefficients/confirmation.','','Process and contribution expansion:','```json',json.dumps(process,indent=2),json.dumps(decomp,indent=2),json.dumps(decision,indent=2),json.dumps(fullcost,indent=2),json.dumps(audit,indent=2),'```','','All new predictions/traces sealed before new GT; independent audit verifies every weighted gradient, sampled pair/label, Adam update and MBR decision. Raw0.1/0.3 signed outputs, all seeds, negative tails, worst cases and leave-one-out retained. Zero new experts/models/checkpoints/RGB/visual forwards/GPU/neural weight training or spatial updates. CPU-only cached adaptation/scoring time excludes existing source/expert cache generation and is not end-to-end deployment latency. Both offsets remain correlated and B_o not a marginal of global Strong. No follow-up search or registry promotion.']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=plan['pins'],reused_coefficient1_commit='b65d1d6250d23fa71b3a38c19c5ce1ca0c031f3c',exposure='RC128 repeatedly exposed development',excluded=['media','captions','parentIDs','GT coordinates','weights','raw policy traces and features']))
assert all(sha(R/f)==h for f,h in plan['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files']);check();print(json.dumps(decision,indent=2),flush=True)
