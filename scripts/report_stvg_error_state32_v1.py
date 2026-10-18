"""Report sealed scalar diagnostics, without media, models or dense GT access."""
from pathlib import Path
import hashlib,json,os
os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/stvg_error_state32_v1';P=ROOT/'results/stvg_error_state/2026-10-18/cached32'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def ci(p):return f"{p['mean']:+.6f} [{p['CI95'][0]:+.6f}, {p['CI95'][1]:+.6f}]"if p['mean']is not None else'UNKNOWN'

def main():
 x=read(P/'RESULTS.json');a=read(P/'ROOT_AGGREGATE_AUDIT.json');assert a['status']=='pass'
 numerical=read(D/'ROOT_DINO_NUMERICAL_AUDIT_NO_GT.json');plan=read(D/'PLAN_LOCK.json');runtime=read(D/'RUNTIME_LOCK.json')
 dbar=read(D/'DINO_FULL_BARRIER.json');obar=read(D/'GLOBAL_PREDICTION_BARRIER.json');complete=read(D/'CPU_COMPLETION.json')
 refs=['GT_planned','GT_common','DINO_common','DINO_full'];labels=['GT planned','GT common','DINO common','DINO full']
 diagnostic={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in x['rows']if r['dataset']==ds];dd={}
  for base in ['Before','Source']:
   ratios=[r['diagnostics'][base]['variation'].get('error_to_GT_speed_ratio')for r in rr];valid=[v for v in ratios if v is not None]
   q={}
   for ref in refs:
    g=x['groups'][ds][base][ref];q[ref]={}
    for ep in ['v','unobserved_decoder']:
     values=np.array(g['global_minus']['Base'][ep]['per_parent_pp']);loo=(values.sum()-values)/(len(values)-1)
     q[ref][ep]=dict(full_mean_pp=float(values.mean()),leave_one_saved_score_mean_min_pp=float(loo.min()),leave_one_saved_score_mean_max_pp=float(loo.max()),leave_one_negative_count=int((loo<0).sum()),largest_positive_alias=rr[int(values.argmax())]['alias'],largest_positive_pp=float(values.max()),largest_negative_alias=rr[int(values.argmin())]['alias'],largest_negative_pp=float(values.min()))
   contributions={ref:{arm:{k:float(np.mean([r['contributions'][base][ref][arm][k]for r in rr]))for k in ['observed','unobserved_decoder','benchmark_interpolated']}for arm in ['LocalAnchor','GlobalError']}for ref in refs}
   dd[base]=dict(correction_error_to_GT_center_speed_ratio=dict(evaluable_parents=len(valid),slower_error_parents=sum(v<1 for v in valid),median=float(np.median(valid))if valid else None,units='ratio of normalized-image-center speeds per true second; not target-size-relative'),leave_one_saved_scores_only=q,full_vIoU_contributions_pp=contributions)
  diagnostic[ds]=dd
 put(P/'DIAGNOSTICS.json',dict(status='complete',groups=diagnostic,no_new_GT_or_scoring=True,leave_one_is_not_stream_replay=True,no_parent_removed_from_primary=True))
 verification={k:v for k,v in numerical.items()if k!='files'};verification.update(postseal_all_reference_audit='ROOT_AGGREGATE_AUDIT.json',official_metric_scalar_checks=complete['metric_checks'],fixed_clock_checks=complete['time_invariant_checks'],source_logp0_only=True,original_implementation_pins_unchanged=True)
 put(P/'VERIFICATION.json',verification)
 actual=dict(status='complete',count=32,parents_per_target=16,backbone='TA-STVG',cohort='original rollback32 exposed development parents, saved original order and inherited Before states',
  source_domains_and_checkpoint_hashes=read(ROOT/'results/stvg_posterior_decision/2026-10-18/cached64/ACTUAL_PROTOCOL.json')['source_domains_and_checkpoint_hashes'],
  cached_DINO_expert=dict(model='Grounding DINO',weights_sha256='1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3',provenance='original verified capture reused; no new checkpoint inference or expert call'),
  primary='Before/final_MBR',sensitivity_bases=['Source'],clocks=plan['clocks'],
  fixed_MBR_is_user_engineering_choice=True,previous_T0_replacement_failure_preserved=True,
  temporal_distribution='Source projection.offsets.logp0 only; full legal s<e, independent-offset product, physical envelope; all final legal grid actions including q=0',
  spatial_chart=['cx','cy','logw','logh'],parameters_per_query=4,delta='equal-weight mean of available target-minus-baseline anchor chart residuals',
  GlobalError='same delta applied to every actual decoder node; center translation, width/height scaling; no clipping',
  LocalAnchor='replace only actual reference decoder nodes; all other decoder nodes exact Base, then original benchmark xyxy interpolation',
  references=dict(DINO_full='all original admitted Top1 boxes; no new GT in prediction generation',GT_planned='exact GT at original planned positions only',GT_common='exact GT at original admitted positions with annotation',DINO_common='same original DINO targets restricted to GT_common positions; diagnostic availability filter'),
  GT_loader_boundary='historical oracle loader parses full files, but correction build() accepts only sealed sparse targets; dense evaluation starts after all oracle predictions seal',
  only_DINO_full_prediction_generation_deployment_eligible=True,not_established_deployable_benefit=True,
  no_ref='exact Base noop, full denominator',no_new_GPU_models_experts_training_or_Student_updates=True,
  support='known unobserved actual decoder frames exclude all four planned slots for every reference set; far >=.25 original input physical duration; benchmark non-decoder annotated frames separate',
  benchmark_clipping='unchanged HC lower pixel-coordinate clipping only; no extrapolation outside original sampled range',
  paired_bootstrap=dict(draws=10000,seed=20261018,unit='parent with one query each',CI=.95,multiplicity_adjusted=False,conditional_on_saved_inherited_trajectories=True,not_online_stream_replay=True),
  decision='positive means of full vIoU and unobserved actual decoder sIoU in both domains; correct-reference ability first, matched DINO reliability next; descriptive development screen, no new significance-only gate',
  costs=dict(DINO_full_generation_wall_seconds=dbar['CPU_wall_seconds'],oracle_generation_after_GT_loader_seconds=obar['CPU_wall_seconds'],score_wall_seconds=complete['CPU_score_seconds'],independent_DINO_numerical_audit_seconds=numerical['CPU_wall_seconds'],not_fresh_online_end_to_end_cost=True),
  code_pins=runtime['pins'],registries_unchanged=True,plan_sha256=sha(D/'PLAN_LOCK.json'),DINO_barrier_sha256=sha(D/'DINO_FULL_BARRIER.json'),oracle_barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json'),
  next_action=read(P/'DECISION.json')['next_action'],new_OPD_filtering_confirmation_not_authorized=True)
 put(P/'ACTUAL_PROTOCOL.json',actual)
 fig,ax=plt.subplots(2,3,figsize=(17,11));fig.subplots_adjust(left=.075,right=.98,top=.88,bottom=.11,wspace=.4,hspace=.45)
 for k,endpoint,title in [(0,'v','A  Before: Global − Base full vIoU (pp)'),(1,'unobserved_decoder','B  Before: unobserved decoder ΔsIoU (pp)')]:
  aa=ax[0,k]
  for d,ds in enumerate(['hc2','vidstg']):
   for j,ref in enumerate(refs):
    p=x['groups'][ds]['Before'][ref]['global_minus']['Base'][endpoint];y=7-(d*4+j)
    aa.errorbar(p['mean'],y,xerr=[[p['mean']-p['CI95'][0]],[p['CI95'][1]-p['mean']]],fmt='o',color='#246b9b'if d==0 else'#b65f31',capsize=3)
  aa.set_yticks(range(8),[f'{ds}: {ref}'for ds in ['Vid','HC']for ref in reversed(labels)],fontsize=8)
  aa.axvline(0,color='.5',lw=.8);aa.set_title(title,loc='left',fontsize=10);aa.spines[['top','right']].set_visible(False)
 aa=ax[0,2];width=.19
 for j,ref in enumerate(refs):
  vals=[x['groups'][ds]['Before'][ref]['global_minus']['LocalAnchor']['v']['mean']for ds in ['hc2','vidstg']]
  aa.bar(np.arange(2)+(j-1.5)*width,vals,width=width*.94,label=labels[j],color=['#628bab','#2e765a','#bd9851','#b56237'][j])
 aa.axhline(0,color='.5',lw=.8);aa.set_xticks([0,1],['HC','Vid']);aa.set_title('C  Global − LocalAnchor full ΔvIoU (pp)',loc='left',fontsize=10);aa.legend(fontsize=7.5,ncol=2);aa.spines[['top','right']].set_visible(False)
 aa=ax[1,0]
 for d,ds in enumerate(['hc2','vidstg']):
  for j,base in enumerate(['Before','Source']):
   vals=[x['groups'][ds][base][r]['global_minus']['Base']['unobserved_decoder']['mean']for r in refs]
   aa.plot(np.arange(4)+(d-.5)*.10,vals,'o-'if base=='Before'else's--',color='#246b9b'if d==0 else'#b65f31',alpha=1 if j==0 else .6,label=f'{ds} {base}')
 aa.set_xticks(range(4),labels,rotation=20,ha='right',fontsize=8);aa.axhline(0,color='.5',lw=.8);aa.set_title('D  Unobserved mean sensitivity (pp)',loc='left',fontsize=10);aa.legend(fontsize=7.5,ncol=2);aa.spines[['top','right']].set_visible(False)
 aa=ax[1,1]
 for d,ds in enumerate(['hc2','vidstg']):
  v=x['groups'][ds]['Before']['DINO_full']['global_minus']['Base']['v']['per_parent_pp'];aa.bar(np.arange(16)+d*17,v,color='#246b9b'if d==0 else'#b65f31',width=.8)
 aa.axhline(0,color='.5',lw=.8);aa.axhline(-5,color='#a63e32',ls=':',lw=.9);aa.set_xticks([7.5,24.5],['HC: original16','Vid: original16']);aa.set_title('E  DINO full signed ΔvIoU: all parents',loc='left',fontsize=10);aa.spines[['top','right']].set_visible(False)
 aa=ax[1,2]
 for d,ds in enumerate(['hc2','vidstg']):
  vals=[r['diagnostics']['Before']['variation'].get('error_to_GT_speed_ratio')for r in x['rows']if r['dataset']==ds];vals=[v for v in vals if v is not None]
  aa.scatter(np.full(len(vals),d)+np.linspace(-.11,.11,len(vals)),vals,color='#246b9b'if d==0 else'#b65f31',alpha=.7,s=18)
  aa.plot([d-.18,d+.18],[np.median(vals)]*2,color='black',lw=2)
 aa.axhline(1,color='.4',ls=':',lw=.9);aa.set_yscale('log');aa.set_xticks([0,1],['HC: 5/16 below1','Vid: 10/16 below1'],fontsize=8);aa.set_title('F  Correction-error / GT center speed',loc='left',fontsize=10);aa.set_ylabel('Normalized-image units per second ratio');aa.spines[['top','right']].set_visible(False)
 fig.suptitle('Sparse four-parameter correction: oracle ability, limited matched-reference support',fontsize=16,y=.97)
 fig.text(.075,.92,'Cached32, fixed final-distribution MBR; Before is primary. Output interventions only: no Student updates, experts, GPU or training.',fontsize=10)
 fig.text(.075,.045,'95% parent bootstrap is descriptive at saved inherited trajectories. LocalAnchor leaves unobserved decoder nodes unchanged by design.\nGT/common sets are oracle diagnostics; only DINO full avoids GT availability in generation. No OPD or filter follow-up is triggered.',fontsize=10)
 fig.savefig(P/'ERROR_STATE.svg');fig.savefig(D/'ERROR_STATE.png',dpi=145);plt.close(fig)
 lines=['# Sparse anchors and a constant four-parameter error state','',
 'Correct sparse references support a useful low-dimensional geometric correction on this exposed32-query panel. The corresponding actual DINO references do not satisfy the two-domain unobserved-decoder criterion. This is a cached output intervention, not a new Student, online TTA run, identity model or deployment result. No OPD, filtering, smoothing or confirmation is launched.','',
 '## Question and matched controls','',
 'The fixed time component is final-distribution MBR, selected by the user as an engineering choice. Its earlier T0 replacement-gate failure, HC fast-motion risk and old MAP/projection comparisons remain valid. The present spatial experiment does not retrospectively promote MBR. Native, projection and merged MAP remain sensitivity clocks for the same sealed spatial outputs; no correction is re-estimated per clock.','',
 'Each baseline is the original saved query-start Before, with its actual inherited LN effect; frozen Source is a separate sensitivity. Base keeps every original box. LocalAnchor replaces only real reference decoder nodes in normalized cxcywh, then uses the original benchmark xyxy interpolation. GlobalError fits one equal-weight residual mean in z=(cx,cy,logw,logh), and adds that constant to every original decoder node. It has four parameters per query: center translation and independent width/height scaling. There is no clipping, robust estimator or speed-dependent rule; there is no neural-network training or Student parameter update. No-reference interventions are exact Base no-ops.','',
 'LocalAnchor is not the old source24 event-relative32-knot minimum-L2 equality projection. That method constrained probability-space knots touched by W; this one edits actual decoder nodes. The old identity-trajectory proposal interpolated varying logit(cxcywh) residuals and fed stochastic OPD, also a different operation. These controls are not numerical replays of those old mechanisms.','',
 '## Reference conditions and primary effects','',
 'DINO_full uses all original admitted Top1 boxes and was sealed for all32 before this experiment opened oracle GT. GT_planned uses exact annotations at the original four planned nodes. GT_common and DINO_common use the identical admitted-node subset with exact GT availability. DINO_common therefore uses label availability and is diagnostic only. The oracle loader parses full truth files, but prediction generation accepts only sealed sparse targets; held-out dense GT does not enter the correction function. All oracle/common predictions were sealed before dense scoring.','',
 '| Target | References | Count / no-ref queries | Global−Base full Δv pp [95% CI] | Unobserved actual decoder Δs pp [95% CI] | Global−Local full Δv pp [95% CI] |','|---|---|---:|---:|---:|']
 for ds in ['hc2','vidstg']:
  for ref in refs:
   g=x['groups'][ds]['Before'][ref];p=g['global_minus'];lines.append(f"| {ds} | {ref} | {g['total_references']} / {g['no_reference_queries']} | {ci(p['Base']['v'])} | {ci(p['Base']['unobserved_decoder'])} | {ci(p['LocalAnchor']['v'])} |")
 lines+=['',
 'GT_common passes the descriptive positive-mean ability screen for full vIoU and unobserved actual decoder sIoU in both targets. Its full Global−Local gain is +4.648867pp in HC with a CI crossing zero, and +7.664849pp in Vid with a positive CI. Correct-reference diagnostic gains are not deployable effects. Even GT_common has4 negative full-v and5 negative unobserved-s parents in HC, and1 negative unobserved parent in Vid; the constant model is not universally correct.',
 '', 'DINO_common fails the HC unobserved mean (−1.949017pp); DINO_full also fails it (−2.675767pp). Both HC CIs are wide and cross zero, so these are not proofs of statistically established harm. They fail the prelocked mean screen. Full DINO improves mean full vIoU by+.689673pp HC and+1.931277pp Vid, but that is insufficient when the required unobserved endpoint fails. The matched-reference gap supports reference reliability as a limitation; it does not distinguish identity mistakes from extent/localization errors, and does not rule out limitations of the constant correction model.',
 '', 'All16 queries per domain remain. Reference counts HC/Vid are planned49/30, common43/24, full DINO56/50. GT_common/DINO_common have1/6 no-reference queries; full DINO has0/2. Zero-reference cases and unknown supports are retained, never dropped or assigned fabricated GT.',
 '', '## Actual decoder support, interpolation and tails','',
 'The primary known unobserved decoder support excludes all four original planned positions for every arm/reference condition, including unadmitted or annotation-missing slots. Within-MBR, observed, all-known and far support (at least one quarter of original input duration from every planned anchor) remain separate. Benchmark interpolation uses annotated non-decoder frames and retains original no-extrapolation behavior. Far support is available for only4 HC and7 Vid parents; empty supports remain null with counts.',
 '', 'LocalAnchor has exactly the same IoU as Base at unobserved actual decoder nodes by construction. Its benefit at benchmark-interpolated frames can still be real. This design invariant does not show that all interpolation methods are ineffective. Full Δv is independently decomposed into observed nodes, unobserved decoder nodes and benchmark-interpolated frames, with exact sums for all512 arm/reference/base cases.',
 '', 'HC DINO_full has a large positive H016 full-v change of+57.114365pp. Removing that saved score from the mean gives−3.071973pp; the primary mean still includes it. The complete leave-one-parent saved-score range is−3.071973 to+2.480409pp. This is a sensitivity summary, not online replay, refitting or case removal. Vid full-v leave-one means stay+1.464592 to+2.116971pp. All signs and severe losses below−5pp are retained.',
 '', 'HC full DINO fast-motion Q4 has6 parents, mean full Δv−5.293275pp and unobserved Δs−5.229923pp; Vid fast has3 parents. HC has no common Q1 slow group in this panel. These small/absent strata do not justify a general fast/slow routing claim. Target size, event length and query-form bins, counts, risks and CIs are all retained.',
 '', '## Does a constant correction preserve useful motion?','',
 'The correction preserves pre-metric center increments and log-size increments exactly. That is algebra: it does not show correct identity, event timing or globally smooth prediction error. In the Before baseline, correction-error center speed is below GT center speed for only5/16 HC and10/16 Vid parents. Median error/GT speed ratios are1.176150/.922498. These use normalized-image-center distance per true second, not target-size-relative speed. GT-common benefits can coexist with locally varying or badly estimated errors.',
 '', 'Source sensitivity preserves the main contrast: GT_common unobserved gains+10.685929/+25.307528pp HC/Vid; DINO_common−1.340500/+8.617848, full DINO−2.124377/+9.845364. Before and Source are different fixed starts. Neither is a replay of a new correction carried through the original stream, and the positive Source Vid CIs do not replace the primary Before endpoint.',
 '', '## Verification, cost and scope','',
 f"Independent no-GT checking uses only original Source logp0, never actionness values: {numerical['exhaustive_candidate_checks']} final legal candidates, including zero-mass decisions, pass direct physical expected-tIoU comparison (max{numerical['max_utility_error']:.6g}); independent two-dimensional CDF reconstruction differs by at most{numerical['max_q_CDF_error']:.6g}. All256 reference/base corrections pass separate constant least squares and local-node/no-op/increment checks;42 are exact no-op pairs. There are64 bit-exact original Before/Source checks and53,616 independent aggregate/CI/support/stratum/count checks. The original scorer additionally checks9,728 metric scalars and2,048 time invariants.",
 '', f"DINO_full cached generation took{dbar['CPU_wall_seconds']:.6f}s including I/O and q/MBR. Oracle generation took{obar['CPU_wall_seconds']:.6f}s after the GT loader (loader time excluded), and dense scoring{complete['CPU_score_seconds']:.6f}s. Separate independent DINO numerical checking took{numerical['CPU_wall_seconds']:.6f}s. These are CPU diagnostic stages, not fresh online end-to-end cost; do not sum them as deployed latency. New GPU/model/expert/training/Student-update counts are0.",
 '', 'The cohort is the original exposed rollback32,16 independent parents per target, with different original cross-domain Source checkpoints and inherited Before trajectories. One query per parent makes query/parent macro means coincide, but inherited-state coupling remains. Fixed-seed10,000-draw parent intervals describe these saved outputs; they do not replay a stream after deleting parents and are not multiplicity-adjusted final confirmation.',
 '', 'The permitted conclusion is geometric ability with correct sparse references and unresolved actual-reference reliability for this constant four-parameter correction. The locked matched/full-DINO two-domain condition fails, so no structured correction, OPD, filter, new expert, parameter training or automatic confirmation follows. CURRENT/MAIN/PAPER and all previous positive and negative evidence remain unchanged.',
 '', '- [Complete anonymous results, signed parent effects, CIs and strata](RESULTS.json)',
 '- [Locked descriptive decision](DECISION.json) and [actual protocol/cost](ACTUAL_PROTOCOL.json)',
 '- [Independent aggregate audit](ROOT_AGGREGATE_AUDIT.json), [numerical verification](VERIFICATION.json), [saved-score and speed diagnostics](DIAGNOSTICS.json)',
 '- [Aggregate figure](ERROR_STATE.svg) and [protocol](../../../../protocols/stvg_error_state32_v1.md)',
 '', 'No GT coordinates, original query/parent text, media, q/utilities, weights, raw tensors or personal records are exported. Oracle/common conditions are explicitly diagnostic; DINO_full only avoids GT in generation, and has not passed the measured deployment-benefit screen.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');(P/'README.md').write_text('# Sparse error-state cached32\n\n[Report](REPORT.md), [complete anonymous results](RESULTS.json), [decision](DECISION.json), [actual protocol](ACTUAL_PROTOCOL.json).\n\nCorrect-reference ability is supported on this exposed panel. Actual DINO references fail the primary two-domain unobserved-node criterion. No new expert/model/training or OPD follow-up.\n')
 files=['vg_tta/stvg_error_state_correction_v1.py','scripts/run_stvg_error_state32_v1.py','scripts/score_stvg_error_state32_v1.py','scripts/audit_stvg_error_state32_v1.py','scripts/audit_stvg_error_state_aggregates_v1.py','scripts/report_stvg_error_state32_v1.py','protocols/stvg_error_state32_v1.md']
 files += [str((P/f).relative_to(ROOT))for f in ['RESULTS.json','DECISION.json','ROOT_AGGREGATE_AUDIT.json','VERIFICATION.json','ACTUAL_PROTOCOL.json','DIAGNOSTICS.json','ERROR_STATE.svg','REPORT.md','README.md']]
 put(D/'PUBLIC_FILES.json',dict(files=files));print(json.dumps(dict(files=len(files),bytes=sum((ROOT/f).stat().st_size for f in files))))
if __name__=='__main__':main()
