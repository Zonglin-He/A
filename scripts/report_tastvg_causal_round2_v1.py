"""Render audited Round2 results with domain-specific estimates and six answers."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write
from scripts.run_tastvg_causal_round2_v1 import OUT,ARMS

COHORTS=[('hcstvg1_test','Vid → HC1'),('vidstg_test','HC2 → Vid'),('all','Pooled64 (secondary)')]

def num(x,d=3):return 'NA' if x is None else f'{x:.{d}f}'
def pp(x):return num(None if x is None else x*100)
def cell(x,percent=True):
    if x['mean'] is None:return 'NA'
    a=100 if percent else 1
    return f"{x['mean']*a:+.3f} [{x['ci95'][0]*a:+.3f}, {x['ci95'][1]*a:+.3f}]"

def run():
    dest=OUT/'analysis';s=read(dest/'ORACLE_SUMMARY.json');assoc=read(dest/'ASSOCIATIONS.json');pres=read(dest/'PRESERVATION.json');causal=read(dest/'PARTA_SUMMARY.json');coll=read(dest/'COLLATERAL_SUMMARY.json');rows=read(dest/'ORACLE_ROWS.json');audit=read(dest/'ORACLE_AUDIT.json');paudit=read(dest/'PARTA_AUDIT.json');gt=read(OUT/'GT_EXPOSURE.json');seconds=sum(read(p)['seconds'] for p in (OUT/'allocations').glob('*.json'))
    lines=['# TA-STVG Round2 — Causal Evidence & Correctability Audit','',
       '**Completed / independently read back.** Same64 historically exposed development parents,32 per direction. Ground truth is intentionally used for oracle optimization; these are not unlabeled TTA scores. Model weights and text H stay frozen. Round3/expert methods and PTD resumption were not run.','',
       '## Native oracle outcomes','',
       'Values are percent. Changes and95% parent bootstrap CIs are percentage points. Native sIoU scores all GT-valid boxes independently of the predicted temporal support; tube-supported sIoU additionally charges missing final-tube support. Each parent contributes one query. All finite terminal states are retained; no best-GT-metric step selection.']
    for cohort,title in COHORTS:
        x=s[cohort];lines += ['',f'### {title} (n={x["n"]})','',
        '|Arm|sIoU|tIoU|vIoU|Δs [CI]|Δt [CI]|Δv [CI]|v harm >5pp|','|---|---:|---:|---:|---|---|---|---:|',
        f"|B0|{pp(x['B0']['sIoU']['mean'])}|{pp(x['B0']['tIoU']['mean'])}|{pp(x['B0']['vIoU_corrected']['mean'])}|—|—|—|—|"]
        for arm in ARMS:
            a=x['arms'][arm];lines.append(f"|{arm}|{pp(a['absolute']['sIoU']['mean'])}|{pp(a['absolute']['tIoU']['mean'])}|{pp(a['absolute']['vIoU_corrected']['mean'])}|{cell(a['delta']['sIoU'])}|{cell(a['delta']['tIoU'])}|{cell(a['delta']['vIoU_corrected'])}|{a['tails']['vIoU_corrected']['harm_gt5pp']}|")
    lines+=['','## Preservation: useful gain and collateral effects','',
       'Useful-gain retention uses the SAME parents with positive unconstrained oracle target-metric gain. Numerator includes protected harms rather than clipping them. Zero denominators are NA. All arms share loss-descent backtracking; protected arms add only the stated complementary constraint. Oselective allocates5 S then5 T steps versus10 joint steps in OST.','',
       '|Direction|Pair|Positive oracle parents|Useful gain retained|Protected−bare target gain [pp CI]|','|---|---|---:|---:|---|']
    for cohort,title in COHORTS:
        for arm in ('OS_PT','OT_PS'):
            z=pres[cohort][arm];lines.append(f"|{title}|{arm}|{z['bare_positive_queries']}|{pp(z['useful_gain_retention'])}%|{cell(z['paired_target_gain_delta'])}|")
    lines+=['','|Direction|Arm|Mean accepted /10|Entirely no-op|Interval changes|Self box IoU on original interval|Endpoint KL|Newly lost GT support frames, mean|','|---|---|---:|---:|---:|---:|---:|---:|']
    for cohort,title in COHORTS:
        for arm in ARMS:
            a=s[cohort]['arms'][arm];c=coll[cohort]['arms'][arm];lines.append(f"|{title}|{arm}|{num(a['accepted']['mean'])}|{a['all_noop']}|{a['interval_changed']}|{num(a['self_box_iou']['mean'],6)}|{num(c['endpoint_KL_base_to_new']['mean'],6)}|{num(c['newly_lost_GT_supported_frames']['mean'])}|")
    lines+=['','## Vulnerability associations (locked before this round opened GT)','',
       'Primary VA=max-budget(mean-offset ASA1-appearance distance) using Round1 sealed preserving states. Spearman correlations and paired-parent bootstrap CIs are descriptive; no post-GT vulnerability feature selection. A positive raw loss-gradient cosine points uphill, so corrective alignment uses its negative.','',
       '|Direction|VA vs baseline spatial error|VA vs spatial oracle gain|VA vs joint oracle gain|Attack vs spatial descent cosine, mean [CI]|','|---|---|---|---|---|']
    def corr(z):return 'NA' if z['rho'] is None else f"{z['rho']:+.3f} [{z['ci95'][0]:+.3f}, {z['ci95'][1]:+.3f}]"
    for cohort,title in COHORTS:
        a=assoc[cohort];g=s[cohort]['gradient'];lines.append('|'+title+'|'+'|'.join(corr(a['VA/'+key]) for key in ('baseline_spatial_error','spatial_oracle_gain','joint_oracle_gain'))+'|'+cell(g['attack_cos_descent_S'],False)+'|')
    lines+=['','## Architecture alignment and secondary rank diagnostic','',
       'Baseline full native gradients include off-branch components. Energy concentration is a local Jacobian property; finite one-step swapped probes are reported separately. Ranks are channel-matrix energy retained, not a transferable trained subspace or a low-rank native-efficacy experiment.','',
       '|Direction|Objective|Aligned gradient energy %|Aligned first-step target gain pp|Swapped first-step target gain pp|R8/R16/R32/R64 retained %|','|---|---|---:|---:|---:|---|']
    for cohort,title in COHORTS:
        for task in ('S','T'):
            g=s[cohort]['gradient'][task];z=pres[cohort]['swapped_'+task];ranks='/'.join(pp(g['rank'][r]['mean']) for r in ('8','16','32','64'))
            lines.append(f"|{title}|{task}|{pp(g['aligned_energy']['mean'])}|{pp(z['aligned_first_step_gain']['mean'])}|{pp(z['swapped_one_step_gain']['mean'])}|{ranks}|")
    lines+=['','## PartA: ASA → Query → Decoder','',
       'Exact selected-frame equality is required in BOTH offsets: stage1 HC32/Vid30, stage2 HC27/Vid26. Strong selection drift absence does not imply equality. Stage1 intermediate-head output and route-propagated FINAL tube are distinct; stage2 head is the final output. A/H/AH denote attention-only, feature-only and joint pooled-query effects in frozen baseline context.','',
       '|Direction|Stage/appearance|Eligible|Relative Q change A/H/AH|Cancellation ratio|Decoder self-vIoU A/H/AH|Stage1 routed final self-vIoU A/H/AH|','|---|---|---:|---|---:|---|---|']
    for cohort,title in COHORTS:
        for stage in (1,2):
            x=causal[f'{cohort}/stage{stage}/app'];rel='/'.join(num(x['decomposition'][k]['relative_norm']['mean'],6) for k in ('A','H','AH'));dec='/'.join(num(x['decoder'][k]['self_vIoU']['mean'],6) for k in ('A','H','AH'));route='/'.join(num(x['decoder'][k]['final_routed']['self_vIoU']['mean'],6) for k in ('A','H','AH')) if stage==1 else 'stage2 is final'
            lines.append(f"|{title}|{stage}|{x['eligible_queries']}|{rel}|{num(x['cancellation_ratio']['mean'],6)}|{dec}|{route}|")
    lines+=['','The observed appearance decomposition has attention-only and joint query effects of similar size, while feature-only effects are smaller. This weakens dominant A/H cancellation on the eligible set. Small cosine distance alone does not imply zero vector change: relative query-norm changes are reported. Frozen-context decoder output changes are small; stage1 route-propagated predictions are unchanged in this intervention set. These data support substantial functional redundancy/attenuation locally, not proof that all ASA is irrelevant. Changed-routing cases remain outside the exact decomposition.','',
       '## Six requested questions: findings and decision','',
       '1. **Oracle correctability: yes on this finite screen.** Vid→HC1 / HC2→Vid spatial-only gains are +28.414 / +21.200pp sIoU; temporal-only gains +42.324 / +20.363pp tIoU; joint gains +19.179 / +11.333pp vIoU. All six respective bootstrap CIs are positive. Every parent improves spatial sIoU under OS. These are GT-assisted opportunities at rho=.02/K10, not unlabeled TTA gains or a theoretical upper bound.',
       '2. **Architecture alignment: strongly supported locally.** Spatial native gradient energy in appearance averages99.9973% /99.9996%; temporal energy in motion is100% /100%. Aligned versus swapped spatial first-step gains are7.294 vs0.040pp /3.598 vs0.007pp; aligned temporal first-step gains6.871 /1.875pp versus0 in the swapped appearance block. Official detach semantics are retained, so this describes the native Jacobian and these finite probes, not exclusive global correctability.',
       '3. **Preservation retains gain, but activation matters.** OS_PT equals OS on every parent: temporal protection rejects zero candidates. OT_PS retains82.823% of positive bare temporal gain in HC and100% in Vid; only HC activates it (9/32 parents,248 rejected candidates,55 no-op steps). Vid temporal protection is inactive. Oselective is5 spatial then5 temporal steps; it is not a10+10 budget.',
       '4. **Less box drift does not establish fewer task failures.** HC OT_PS improves original-interval box self-IoU from.955245 to.989212, but lowers vIoU gain by5.430pp [−9.035,−2.153] relative to OT and changes >5pp vIoU-harm count from1 to2. Mean newly lost GT support frames increases1.281 to1.500. The protected baseline interval covers only45.23% of HC GT-valid frames on average. Oselective−OST vIoU is+7.476pp [2.125,13.402] in HC but−1.185pp [−1.920,−.514] in Vid. Schedule and objective allocation also differ, so these comparisons do not isolate a benefit of preservation alone. No selective method is promoted.',
       '5. **ASA vulnerability does not support the proposed positive correctability gate.** VA versus spatial gain is−.250 [−.574,.127] in HC and−.009 [−.382,.362] in Vid; VA versus joint gain is−.223 /−.038 with both CIs spanning zero. Both fail the prelocked positive-association criterion; this is absence of support, not proof of universal independence. In Vid, VA versus baseline spatial error is−.536 [−.750,−.222], opposite a high-V bad-case indicator. Attack versus spatial DESCENT mean cosine is−.0154 /+.0004, with both CIs spanning zero. Keep this VA as a diagnostic rather than adopt it as a gate, radius rule or repair direction.',
       '6. **ASA changes are substantially attenuated downstream; dominant A/H cancellation is not supported.** On the exact-frame eligible set, attention-only query changes approximate joint changes and feature-only changes are much smaller. Pooled appearance query relative-norm changes are5.62% at stage1 and8.03% at stage2, so query vectors are not invariant merely because cosine drift is small. Stage1 interventions produce unchanged route-propagated final tubes; stage2 joint query interventions retain mean self-vIoU .996687 / .997718. This supports local functional redundancy/attenuation along the tested directions. It neither establishes universally irrelevant ASA nor labels attention as pointing at the wrong object.','',
       '## Verification and execution record','',
       f"PartA: {paudit['pooling_checks']} independently reconstructed pooled vectors, max relative error {paudit['max_pooling_relative_error']:.3g}; {paudit['metric_comparisons']} metric comparisons, max error {paudit['max_metric_error']:.3g}. Oracle:64 baseline outputs,384 final adaptations,3840 adaptation backwards plus128 initial S/T gradients and4 native-pipeline Jacobian-check backwards (3972 total backwards). {audit['independent_trial_metric_comparisons']} independent preservation comparisons, max error {audit['max_metric_error']:.3g}. Two original-pipeline Jacobian audits and four final full-pipeline reinsertion checks passed; all final text deltas zero and all radii valid.",
       '',f"Native contracts: `{audit['native_contracts']}`. Official source weights are (L1,GIoU,T)=(5,3,2) for Vid-source and(5,4,10) for HC2-source; sigma2 Gaussian temporal KL formula. Frozen source checkpoints follow the previous exact replay interface. No parameter update, expert invocation or backbone-per-step forward.",
       '',f"Cumulative Round2 GPU-process allocation: **{seconds:.3f}s ({seconds/60:.2f}min)**, including imports, source loading, engineering failures and replay checks; CPU development/readback/reporting excluded. All failures and source revisions retained. Initial PartA comparison failed on CPU-versus-GPU equality before results/GT; engineering repair moved only the comparison tensors to CPU, with exact equality unchanged.",
       '', 'Independent float64 NumPy native-loss reconstruction passed16509 scalar checks across5439 saved candidate/final/probe predictions (plus64 baselines), max absolute error1.480e-6; zero near-zero native loss changes under the registered5e-5 reporting threshold. Four CPU mathematical contract tests passed. Public scalar reconstruction is a separate publication check.',
       '', 'GT sequence: vulnerability definitions were locked first, then all PartA outputs sealed and independently read back, then the existing label container was streamed and only the64 authorized keys retained. GT is intentionally used in oracle losses/backtracking; selection is by loss and preservation, never by evaluated GT s/t/v metrics. All oracle outputs sealed before outcome scoring. No new cohort or fresh-test claim.',
       '', 'All scalar rows, collateral metrics, target exceptions, tails, baseline-good retention, mean-V secondary correlations and rank spectra are in the adjacent JSON files. No production method promoted; Round3 and PTD queues remain unstarted/paused.','',
       '![Oracle gains by direction](analysis/oracle_gains.png)','',
       '![Vulnerability versus spatial oracle gain](analysis/vulnerability_correctability.png)','']
    p=OUT/'REPORT.md';assert not p.exists();p.write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,(cohort,title) in zip(axes,COHORTS[:2]):
        x=np.arange(len(ARMS));width=.24
        for j,(metric,label,color) in enumerate([('sIoU','Spatial','steelblue'),('tIoU','Temporal','darkorange'),('vIoU_corrected','Video','seagreen')]):
            y=[100*s[cohort]['arms'][a]['delta'][metric]['mean'] for a in ARMS];ax.bar(x+(j-1)*width,y,width,label=label,color=color)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,ARMS,rotation=35,ha='right');ax.set(title=title,ylabel='Oracle gain over B0 (pp)');ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle('Round2: finite native oracle, rho .02 / 10 gradients');fig.savefig(dest/'oracle_gains.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,(cohort,title) in zip(axes,COHORTS[:2]):
        rr=[r for r in rows if r['cohort']==cohort];ax.scatter([r['VA'] for r in rr],[100*(r['arms']['OS']['metrics']['sIoU']-r['baseline']['sIoU']) for r in rr],s=30,alpha=.8);ax.axhline(0,color='grey',lw=.8);ax.set(title=title,xlabel='Prelocked ASA vulnerability VA',ylabel='Spatial oracle gain (pp)');ax.grid(alpha=.2)
    fig.savefig(dest/'vulnerability_correctability.png',dpi=170);plt.close(fig);print(p)

if __name__=='__main__':run()
