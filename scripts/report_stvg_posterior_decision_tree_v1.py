"""Render only sealed anonymous T0/S0 scalars; no data/model/GT readers."""
from pathlib import Path
import hashlib, json, os
os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/stvg_posterior_decision_tree_v1'
P=ROOT/'results/stvg_posterior_decision/2026-10-18/cached64'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def interval(x,digits=6):return f"{x['mean']:+.{digits}f} [{x['CI95'][0]:+.{digits}f}, {x['CI95'][1]:+.{digits}f}]"

def main():
 x=read(P/'RESULTS_INITIAL.json');decision=read(D/'INITIAL_DECISION.json')
 assert read(P/'ROOT_AGGREGATE_AUDIT.json')['status']=='pass'
 numerical=read(D/'ROOT_NUMERICAL_AUDIT_NO_GT.json');qualification=read(D/'ROOT_CPU_QUALIFICATION.json')
 verify={k:v for k,v in numerical.items()if k!='files'}
 verify.update(synthetic_qualification=qualification,GT_coordinates_or_media_read_by_independent_audits=False,
               aggregate_audit='ROOT_AGGREGATE_AUDIT.json',old_three_readout_scalar_bit_exact_checks=6336,
               official_metric_scalar_checks=read(D/'CPU_COMPLETION_INITIAL.json')['metric_checks'])
 write(P/'VERIFICATION.json',verify);write(P/'DECISION.json',decision)
 lock=read(D/'RUNTIME_LOCK.json');plan=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_READOUT_BARRIER.json')
 old=read(ROOT/'results/stvg_interval_decision/2026-10-18/components64/ACTUAL_CONFIG.json')
 actual=dict(status='complete',scope='T0/S0 initial cached64 finite development screen',
  queries=64,unique_parents_per_domain=32,historically_exposed=True,backbone='TA-STVG',
  source_domains_and_checkpoint_hashes=old['domains'],primary_geometry='fixed original online After',
  sensitivity_geometries=['Source','Before'],no_new_GPU_models_DINO_training_or_Student_updates=True,
  probability='original float32 full endpoint logits; float64 strict s<e probabilities; independent-view product pushed through min-start/max-end envelope',
  posterior_calibration_not_established=True,decision_grid='sorted union physical frame IDs; all a<b, [fid[a],fid[b]+1), including q=0 actions',
  temporal_tie='row-major first numeric maximum',greedy_gain='.5 sum q(I) 1{f in I and current hits<2}',
  greedy_ties='exact float64 equality; greatest minimum physical distance; first point closest to input [first,last+1) midpoint; then earlier ID',
  layouts=dict(N='original Native Uniform4',U='original full-input Uniform4',P='greedy merged-q capped-two-hit coverage',projection_uniform4='auxiliary original projection Uniform4'),
  subgroup_bins=plan['critical_bins'],minimum_evaluable_parents=8,subgroup_small_sample_status='UNKNOWN, never PASS',
  T0_gates=dict(Native_gain_pp=.5,projection_noninferiority_margin_pp=-.5,critical_mean_risk_pp=-1),
  S0_gates=dict(P_minus_N=.05,baseline_P_minus_U_nonnegative=0,preGT_posterior_added_value_P_minus_U_strictly_positive=True,zero_hit_increase_limit=.05,critical_coverage_mean_risk=-.05),
  preGT_analysis_clarification=read(D/'ANALYSIS_GATE_LOCK.json'),evaluation_lock=read(D/'EVALUATION_LOCK.json'),
  bootstrap=dict(unit='parent; exactly one query per parent',draws=10000,seed=20261018,confidence=.95,multiplicity_adjusted=False),
  algorithm_CPU=dict(readouts_wall_seconds=bar['CPU_wall_seconds'],T0_sum_seconds=bar['T0_seconds'],S0_sum_seconds=bar['S0_seconds']),
  independent_numerical_audit_CPU_seconds=numerical['CPU_wall_seconds'],CPU_caps_seconds=dict(T0=3600,S0=3600),
  sole_extension=dict(prelocked_additional_queries=64,performed=False,trigger=False,selection='first32 unique parents/domain in source-case lexicographic case_id order, excluding initial parents; no GT criteria'),
  T0_matched_After_extension_available=False,retired_original_P1_payloads_not_replaced=True,
  untriggered_stages=['S1 real expert acquisition/fit','J0 crossed/Direct','online stream','second backbone','C0 confirmation'],
  production_registries_unchanged=True,GT_only_after_all64_readouts_sealed=True,
  scientific_pins=lock['pins'],plan_sha256=sha(D/'PLAN_LOCK.json'),readout_barrier_sha256=sha(D/'GLOBAL_READOUT_BARRIER.json'))
 write(P/'ACTUAL_PROTOCOL.json',actual)
 arms=['Native','projection','marginal_MAP','per_offset_MBR','final_MBR']
 names=['Native','Projection','Merged MAP','Offset MBR','Final MBR'];colors=['#718096','#236e9b','#a07923','#8c72ae','#19866b']
 fig,axs=plt.subplots(2,3,figsize=(17,10.7));fig.subplots_adjust(left=.055,right=.985,top=.88,bottom=.12,wspace=.30,hspace=.42)
 for ax,metric,title in [(axs[0,0],'v','A  Fixed After vIoU (%)'),(axs[0,1],'t','B  Fixed After tIoU (%)')]:
  for j,a in enumerate(arms):
   vals=[100*x['temporal'][ds]['After']['metrics'][a][metric]for ds in ['hc2','vidstg']]
   bars=ax.bar(np.arange(2)+(j-2)*.16,vals,width=.15,color=colors[j],label=names[j])
   for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+.45,f'{b.get_height():.2f}',ha='center',fontsize=7.5)
  ax.set_xticks([0,1],['HC (32 parents)','Vid (32 parents)']);ax.set_ylim(0,65 if metric=='t'else 32);ax.set_title(title,loc='left',fontsize=11);ax.spines[['top','right']].set_visible(False)
 axs[0,0].legend(fontsize=8,ncol=2,loc='upper right')
 ax=axs[0,2]
 for d,ds in enumerate(['hc2','vidstg']):
  for j,a in enumerate(arms[:-1]):
   p=x['temporal'][ds]['After']['paired']['final_MBR_minus_'+a];y=7-(d*4+j)
   ax.errorbar(p['mean'],y,xerr=[[p['mean']-p['CI95'][0]],[p['CI95'][1]-p['mean']]],fmt='o',color='#236e9b'if d==0 else'#b45e34',capsize=2)
 ax.axvline(0,color='.5',lw=.8);ax.axvline(-.5,color='#b45e34',ls=':',lw=.9)
 ax.set_yticks(range(8),[f'{d}: vs {a}'for d in ['Vid','HC']for a in reversed(names[:-1])],fontsize=8)
 ax.set_title('C  Final MBR paired ΔvIoU (pp), 95% CI',loc='left',fontsize=11);ax.spines[['top','right']].set_visible(False)
 layouts=['N','U','P','projection_uniform4'];lnames=['Native4','Input4','Greedy q4','Projection4']
 for ax,metric,title,scale in [(axs[1,0],'score','D  Real event capped-two-hit coverage',1),(axs[1,1],'zero_hit','E  Real event zero-hit rate (%)',100)]:
  for j,a in enumerate(layouts):
   vals=[scale*x['coverage'][ds]['metrics'][a][metric]for ds in ['hc2','vidstg']]
   bars=ax.bar(np.arange(2)+(j-1.5)*.19,vals,width=.18,color=[colors[0],colors[1],colors[4],colors[2]][j],label=lnames[j])
   for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+(.012 if scale==1 else.5),f'{b.get_height():.3f}'if scale==1 else f'{b.get_height():.1f}',ha='center',fontsize=8)
  ax.set_xticks([0,1],['HC (32 parents)','Vid (32 parents)']);ax.set_ylim(0,1.05 if scale==1 else 40);ax.set_title(title,loc='left',fontsize=11);ax.spines[['top','right']].set_visible(False)
 axs[1,0].legend(fontsize=8,ncol=2)
 ax=axs[1,2]
 for d,ds in enumerate(['hc2','vidstg']):
  for j,a in enumerate(['N','U']):
   p=x['coverage'][ds]['paired']['P_minus_'+a];y=3-(d*2+j)
   ax.errorbar(p['mean'],y,xerr=[[p['mean']-p['CI95'][0]],[p['CI95'][1]-p['mean']]],fmt='o',color='#236e9b'if d==0 else'#b45e34',capsize=3)
 ax.set_yticks([3,2,1,0],['HC: P − Native4','HC: P − Input4','Vid: P − Native4','Vid: P − Input4'],fontsize=8)
 ax.axvline(0,color='.5',lw=.8);ax.axvline(.05,color='#19866b',ls=':',lw=.9)
 ax.set_title('F  Paired real coverage Δ, 95% CI',loc='left',fontsize=11);ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Final posterior decisions and four-frame allocation: both development gates stop',fontsize=16,y=.97)
 fig.text(.055,.915,'Cached64, one query per exposed parent; all readouts sealed before GT. After is primary; Source/Before remain in RESULTS.',fontsize=10)
 fig.text(.055,.055,'HC final MBR misses projection margin and fast-motion risk gate. Critical groups with <8 parents remain UNKNOWN.\nGreedy model coverage increases, but Vid real coverage falls; no extension, DINO, fit, S1/J0/C0 or production change.',fontsize=10)
 fig.savefig(P/'DECISION.svg');fig.savefig(D/'DECISION.png',dpi=145);plt.close(fig)
 lines=['# Final posterior decisions and fixed-budget observation allocation','',
 'Both prelocked development gates stop on the original exposed64-parent panel. Final-distribution MBR improves on Native in both domains but does not meet the unified replacement criteria. Greedy posterior coverage does not improve the real-event four-frame allocation sufficiently and increases important zero-hit risks. No extension, new DINO, new Student fit or later-stage experiment is triggered.','',
 '## T0: expected utility after merging the physical distribution','',
 'The old256 experiment already constructed the merged two-offset distribution q and took its MAP. The old cached64 MBR chose each offset separately and merged two hard decisions. This experiment isolates the remaining order of operations: build the same full q first, then maximize expected physical interval tIoU over all final-grid legal decisions. q is an uncalibrated independent-view working model; correlated offsets are not newly calibrated. This is standard finite expected utility, not a new MBR theory.','',
 '| Target / fixed After | Native v/t % | Projection v/t % | Merged MAP v/t % | Offset MBR v/t % | Final MBR v/t % |','|---|---:|---:|---:|---:|---:|']
 for ds in ['hc2','vidstg']:
  vals=x['temporal'][ds]['After']['metrics'];lines.append('| '+ds+' | '+' | '.join(f"{100*vals[a]['v']:.6f} / {100*vals[a]['t']:.6f}"for a in arms)+' |')
 lines+=['','| Final MBR paired ΔvIoU pp [95% parent CI] | vs Native | vs projection | vs old merged MAP | vs old offset MBR |','|---|---:|---:|---:|---:|']
 for ds in ['hc2','vidstg']:lines.append('| '+ds+' | '+' | '.join(interval(x['temporal'][ds]['After']['paired']['final_MBR_minus_'+a])for a in arms[:-1])+' |')
 lines+=['',
 'The mean threshold is at least +0.5pp versus Native and at least −0.5pp versus projection in each target. HC misses the projection margin by .092837pp. The wide paired interval does not prove inferiority or noninferiority. Final MBR has lower mean vIoU than old merged MAP in both targets, with both differences crossing zero; neither new extra value nor stable dominance by MAP is established. It improves over the old offset-MBR means, so those two MBR orders must remain distinct.',
 '', 'HC fast-motion Q4 has8 parents: final MBR−projection is −3.304065pp [−6.468692,−.578871], triggering the prelocked −1pp mean risk. HC short0/small3 and Vid short7/fast4 are UNKNOWN under the8-parent evaluability floor, never pass. Vid small8 passes this descriptive risk test. These are exposed-panel gates, not final multiplicity-adjusted superiority claims.',
 '', 'The primary geometry is the same saved original online After in all arms. Source and Before sensitivity, all signed parent differences, event/start/motion/target-size bins, zero temporal intersections, added/lost known support, known IoU and unknown support remain in RESULTS. sIoU at fixed geometry stays unchanged when the temporal readout changes. Unknown frame support is not a negative identity label.',
 '', 'A prelocked T0 After extension cannot be executed from matching original P1 payloads because those full fit/After caches were retired. Substituting another spatial method or switching the primary endpoint to Source would change the question. No such substitution or expansion was made.',
 '', '## S0: four-frame coverage under q and under the real event','',
 'N uses the original physical-bin Uniform4 inside Native; U uses the same rule across the complete input; P greedily maximizes E_q[min(1,hits/2)] over four distinct available decoder frames. Projection Uniform4 is an auxiliary baseline. The exact gain is .5 times posterior mass of containing intervals with fewer than two previously selected hits. Deterministic physical-distance ties are fixed before scoring. No expert frames were acquired: the measured outcome here is event-time coverage, not object identity, DINO reference quality or actual adaptation benefit.','',
 '| Target | Real coverage N / U / P / projection4 | Zero-hit % N / U / P / projection4 | P−N [95% CI] | P−U [95% CI] |','|---|---:|---:|---:|---:|']
 for ds in ['hc2','vidstg']:
  c=x['coverage'][ds];lines.append('| '+ds+' | '+' / '.join(f"{c['metrics'][a]['score']:.6f}"for a in layouts)+' | '+' / '.join(f"{100*c['metrics'][a]['zero_hit']:.3f}"for a in layouts)+' | '+interval(c['paired']['P_minus_N'])+' | '+interval(c['paired']['P_minus_U'])+' |')
 lines+=['',
 'Vid has0 improving,5 worsening and27 unchanged parents for P−N coverage. Its CI upper bound is below the +.05 practical threshold; zero-hit rises from4/32 to9/32. HC P−N is positive but only+.015625, and P zero-hit5/32 exceeds U1/32. HC fast-motion P−U=−.1875 and Vid small-target P−N=−.0625 also trigger locked coverage risk. All groups and missing within-event spread values remain visible. The unique fixed extension trigger is false, so no extension is run to seek significance.',
 '', 'Model coverage and real-event coverage diverge. Mean q coverage N→P rises .987126→.991274 in HC and .996332→.999156 in Vid, while real coverage changes+.015625/−.078125. This is not proof that all posterior allocation is useless. It shows this frozen model distribution and capped-two-hit allocation did not satisfy the locked benefit/risk criteria. The failure cannot be assigned to reference identity because no new reference was examined.',
 '', 'The base noninferiority-to-U condition is P−U≥0. Before any new scoring, ANALYSIS_GATE_LOCK (bound by EVALUATION_LOCK) clarified the separate posterior-added-value requirement P−U>0, consistent with stopping a core claim when plain U is equally good. Both variants produce the same stop here. The scorer and all original locks remain unchanged.',
 '', '## Measurement, cost and verification','',
 'The original TA-STVG cross-domain sources are HC target←Vid source checkpoint5ab12c86… and Vid target←HC source47d8f158…. The panels are historically exposed and use different source checkpoints and original projection center fractions1.0/.5. The data do not identify an intrinsic domain-specific mechanism. There is one query per parent, so query and parent macro means coincide. Parent bootstrap uses10,000 fixed-seed draws; the8-parent subgroup floor is an engineering evaluability minimum, not a precision theorem. These intervals describe the saved outputs at fixed original online After trajectories; they do not replay the online stream after removing parents. One query per parent does not remove inherited-trajectory coupling.',
 '', f"All64 five-readout/four-layout outputs were sealed before new GT scoring. Readout wall time was {bar['CPU_wall_seconds']:.6f}s, with T0 sum {bar['T0_seconds']:.6f}s and S0 sum {bar['S0_seconds']:.6f}s. The separate exhaustive numerical audit took {numerical['CPU_wall_seconds']:.6f}s; it is verification overhead, not fresh model inference or an end-to-end speedup claim. New GPU/model/DINO/training/Student-update counts are all0.",
 '', 'Independent verification covered315,403 actual legal decisions by direct physical tIoU, max error3.122503e−15; Source-logit reconstruction of q differs by at most6.245005e−17; greedy physical-membership/objective/tie error is at most3.287301e−16. Thirty adjacent/irregular/point-mass/uniform/sparse/dense synthetic cases add822 utility and43,374 endpoint-partition checks. Anonymous JSON reaggregation passes6,132 means/paired-CI/count/gate checks. The old Native/projection/per-offset-MBR trajectories reproduce6,336 Source/Before/After per-query scalars bit-for-bit. The official metric implementation was also checked against DenseTube for2,880 scalars by the postseal scorer.',
 '', '## Decision scope and files','',
 'T0 and S0 both stop under the prelocked development tree. S1 actual experts/fits, crossed J0/Direct, online-stream checks, second backbone and C0 confirmation are untriggered stages, not incomplete measurements. No temperature, resolution, model, encoder or feedback variant is launched. CURRENT/MAIN/PAPER and prior positive/negative results remain unchanged. No universal impossibility claim follows.',
 '', '- [All anonymous per-parent metrics, signed differences, CIs and strata](RESULTS_INITIAL.json)',
 '- [Locked gate outcomes, risks and unknown groups](DECISION.json)',
 '- [Actual configuration, timing and pre-GT clarification](ACTUAL_PROTOCOL.json)',
 '- [Independent aggregate audit](ROOT_AGGREGATE_AUDIT.json) and [numerical verification](VERIFICATION.json)',
 '- [Aggregate figure](DECISION.svg)',
 '- [Protocol](../../../../protocols/stvg_posterior_decision_tree_v1.md)',
 '', 'Raw q/utilities/logits, GT coordinates, original query/parent identifiers, media, weights and tensors remain private. Public rows contain anonymous scalars and offline group bins only.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
 (P/'README.md').write_text('# Cached64 posterior decision tree\n\n[Measured report](REPORT.md), [complete results](RESULTS_INITIAL.json), [gates](DECISION.json), [actual protocol](ACTUAL_PROTOCOL.json), [independent checks](VERIFICATION.json).\n\nBoth T0 and S0 development gates stop. No extension, new model, DINO or training; no production change.\n')
 files=['vg_tta/stvg_posterior_decision_tree_v1.py','scripts/run_stvg_posterior_decision_tree_v1.py','scripts/score_stvg_posterior_decision_tree_v1.py','scripts/audit_stvg_posterior_decision_tree_v1.py','scripts/qualify_stvg_posterior_decision_tree_v1.py','scripts/audit_stvg_posterior_decision_aggregates_v1.py','scripts/report_stvg_posterior_decision_tree_v1.py','protocols/stvg_posterior_decision_tree_v1.md']
 files += [str((P/f).relative_to(ROOT))for f in ['RESULTS_INITIAL.json','DECISION.json','ROOT_AGGREGATE_AUDIT.json','VERIFICATION.json','ACTUAL_PROTOCOL.json','DECISION.svg','REPORT.md','README.md']]
 write(D/'PUBLIC_FILES.json',dict(files=files))
 print(json.dumps(dict(files=len(files),bytes=sum((ROOT/f).stat().st_size for f in files),status='derived_report_complete')))
if __name__=='__main__':main()
