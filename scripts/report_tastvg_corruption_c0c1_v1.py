"""Render the frozen corruption anatomy and native candidate support audit."""
import sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write
from scripts.run_tastvg_corruption_c0c1_v1 import OUT,CONDITIONS,MEDIUM


def pp(x):return f'{x*100:.3f}'
def ci(x):return f"{x['mean']*100:+.3f} [{x['ci95'][0]*100:+.3f}, {x['ci95'][1]*100:+.3f}]"


def run():
    a=read(OUT/'analysis/C0_SUMMARY.json');b=read(OUT/'analysis/C1_SUMMARY.json');d=read(OUT/'DECISION.json');audit=read(OUT/'analysis/AUDIT.json');cases=read(OUT/'analysis/CASES.json');rows=read(OUT/'analysis/C1_ROWS.json');cost=sum(read(f)['seconds'] for f in (OUT/'allocations').glob('*.json'))
    lines=['# TA-STVG C0/C1 — corruption anatomy and native candidate support','',
    '**Completed / independently read back.** VidSTG-source checkpoint on the original32 historically exposed VidSTG-test development parents. C0 uses all32 x7 conditions; C1 uses the first16 parents in unchanged order x clean/three medium conditions. No expert invocation, adaptation, gradients or HC2 selection. No fresh-test claim.','',
    'The immediate experiment follows the post-Round2 corruption plan. The broader One World / Dual Evidence / reliability-aware selective adaptation direction is retained as a research agenda; candidate headroom alone does not qualify a critic or establish OPD efficacy.','',
    '## Findings and decision','',
    '**Temporal support passes; this spatial candidate generator does not.** Corrupted parent-macro temporal oracle gain is14.431pp [6.934,23.850], with10/16 parents above5pp. Spatial gain is.705pp [.299,1.200], with0/16 above5pp. This is positive but small spatial headroom, not zero correctability. The next justified critic qualification is temporal-only; improve spatial candidate generation before spending on a spatial expert. Neither action is executed in this C0/C1 run.','',
    '**The primary corruption setting still needs care.** All six C0 video-delta CIs cross zero; JPEG has no >5pp S/T/video losses in this panel. Medium defocus contains6 T-dominant,4 S-dominant,1 joint/mixed and21 robust/no-material-loss cases. There are local failures, but no stable average frozen-video degradation at these existing strengths.','',
    '**Most temporal headroom is already present on clean inputs.** Clean T-oracle gain14.810pp is close to corrupted14.431pp. Decoder layers alone provide13.440pp of corrupted temporal headroom; final-logit span additions bring it to14.431pp. Uniform candidate choice gives−.826pp temporal target gain (CI crosses zero), so the oracle result does not make the ranking problem disappear.','',
    '**Actual corruption recovery is partial.** A post-score descriptive readback of every fixed C1 cell with >5pp target damage finds5 spatial-failure cells across4 parents, with0/5 restored to clean by the spatial candidate oracle;3 temporal-failure cells across3 parents, with1/3 restored. These tiny selected-for-description counts are not a new gate. Q12 (ordinal11) under medium defocus drops from87.302% to58.286% tIoU and even the candidate oracle stays58.286%; Q05 (ordinal4) under medium noise drops from16.043% to0 and has a22.656% temporal candidate. Large mean headroom and recovery of specific corruption failures are distinct. See CORRUPTION_RECOVERY.json for every such cell.','',
    '**Minor tails are retained.** Spatial component-max selection decreases video IoU in4/64 cells, worst−.192pp; temporal and combined oracle choices have no video decreases in this panel. None has a >5pp video-harm cell. GT-selected candidates are still oracle-only outputs.','',
    '## C0: frozen corruption effects','',
    'Values are percent; deltas are corrupted minus clean in percentage points with95% paired-parent bootstrap CIs (10000 draws). Negative delta means damage. Original custom severities,224 input interface and exact physical grids are retained. The checkpoint was trained for VidSTG; this is same-domain corruption, not the earlier HC2-to-Vid mechanism panel. Native sIoU measures all GT-valid boxes, independent of predicted interval.','',
    '|Condition|sIoU|tIoU|vIoU|Δs [CI]|Δt [CI]|Δv [CI]|s/t/v losses >5pp|','|---|---:|---:|---:|---|---|---|---|']
    for cond in CONDITIONS:
        z=a[cond];lines.append('|'+cond+'|'+'|'.join(pp(z['absolute'][m]['mean']) for m in ['sIoU','tIoU','vIoU_corrected'])+'|'+'|'.join(ci(z['delta'][m]) for m in ['sIoU','tIoU','vIoU_corrected'])+'|'+'/'.join(str(z['tails'][m]['loss_gt5pp']) for m in ['sIoU','tIoU','vIoU_corrected'])+'|')
    lines+=['','Failure classes use signed losses and a fixed5pp margin; robust means no material S/T loss, not an absolutely correct prediction. The fourth group is joint/mixed when neither branch dominates by5pp. Repeated conditions are not independent parents.','',
    '|Condition|S dominant|T dominant|Joint/mixed|Robust/no material S/T loss|','|---|---:|---:|---:|---:|']
    for cond in CONDITIONS[1:]:lines.append('|'+cond+'|'+'|'.join(str(a[cond]['failure_groups'].get(g,0)) for g in ['S_dominant','T_dominant','joint_or_mixed','robust_no_material_ST_loss'])+'|')
    lines+=['','## Internal evidence drift (corrupted versus clean)','',
    'ASA comparisons use only common selected frames, with exact coverage and selected-frame Jaccard retained in scalar rows. Missing common support is NA, not zero drift. Q vectors report both cosine and relative-norm changes. Associations and categories are descriptive, not causal attribution of output failures.','',
    '|Condition|TTS app/motion JSD|ASA1/ASA2 app cosine distance|Qs1/Qs2 relative norm|Qt1/Qt2 relative norm|Selection Jaccard1/2|','|---|---|---|---|---|---|']
    def pair(e,keys):return '/'.join('NA' if e[k]['mean'] is None else f"{e[k]['mean']:.6f}" for k in keys)
    for cond in CONDITIONS[1:]:
        e=a[cond]['evidence'];lines.append('|'+cond+'|'+'|'.join(pair(e,k) for k in [('TTS_app_JSD','TTS_motion_JSD'),('ASA1_app_cosine_drift','ASA2_app_cosine_drift'),('Qs1_relative_norm','Qs2_relative_norm'),('Qt1_relative_norm','Qt2_relative_norm'),('selected_stage1_jaccard','selected_stage2_jaccard')])+'|')
    lines+=['','## C1: student candidate headroom','',
    'Spatial candidates are complete tubes from the six final-pass decoder layers. Temporal candidates start with native and other layer envelopes, then use final-layer legal-span pairs to fill up to8 unique physical intervals. These are deterministic student hypotheses, not stochastic rollouts or a calibrated policy distribution. Native is always candidate0; no per-frame GT stitching. GT selection is offline oracle only.','',
    '|Condition|Native s/t/v|Spatial oracle Δs [CI]|Temporal oracle Δt [CI]|Layer-only temporal Δt|Combined oracle Δv [CI]|Parents with >5pp S/T gain|','|---|---|---|---|---:|---|---|']
    for cond in MEDIUM+['corrupted_parent_macro']:
        z=b[cond];lines.append('|'+cond+'|'+'/'.join(pp(z['absolute']['native'][m]['mean']) for m in ['sIoU','tIoU','vIoU_corrected'])+'|'+ci(z['gain_S'])+'|'+ci(z['gain_T'])+'|'+pp(z['layer_only_temporal_gain']['mean'])+'|'+ci(z['delta']['TS_oracle']['vIoU_corrected'])+'|'+f"{z['opportunity_gt5pp']['S']}/16; {z['opportunity_gt5pp']['T']}/16"+'|')
    lines+=['','The corrupted primary summary first averages the three medium-condition gains within each parent, then bootstraps16 parents. Clean is separate. Component oracle selection can lower vIoU; all tube harms are retained. Candidate mean is a uniform selection diagnostic, not an implemented method.','',
    '|Condition|Unique S/T, mean|Uniform S/T target gain pp|S/T/combined v-loss >5pp cells|','|---|---|---|---|']
    for cond in MEDIUM+['corrupted_parent_macro']:
        z=b[cond];lines.append('|'+cond+'|'+f"{z['unique_S']['mean']:.2f}/{z['unique_T']['mean']:.2f}"+'|'+pp(z['uniform_S_gain']['mean'])+'/'+pp(z['uniform_T_gain']['mean'])+'|'+'/'.join(str(z['tails'][arm]['v_loss_gt5pp_cells']) for arm in ['S_oracle','T_oracle','TS_oracle'])+'|')
    lines+=['','## Fixed support gate and scope','',
    'The prelocked gate for later critic qualification is corrupted parent-macro target gain>=2pp, bootstrap lower bound>0 and at least4/16 parents with mean corrupted gain>5pp. A pass establishes only candidate headroom; a failure prioritizes improving this candidate generator. Neither result establishes teacher ranking quality, learnability, reliability calibration or TTA efficacy.','']
    for branch,z in d['branches'].items():lines.append(f"- {branch}: **{'PASS' if z['candidate_support_gate'] else 'NOT PASSED'}**, mean {z['mean_gain']*100:.3f}pp, CI [{z['ci95'][0]*100:.3f},{z['ci95'][1]*100:.3f}], parents>5pp {z['parents_gt5pp']}/16.")
    lines+=['','The historical light/medium panel was already known not to induce stable average video harm. Compare the C0 paired losses and clean C1 headroom before interpreting an oracle gain as corruption recovery. Do not strengthen severity or select examples after seeing this result within this run. C2 experts and all adaptation remain unstarted.','',
    '## Deterministic failure/good-case comparison','',
    'Post-score explanatory examples only:largest S loss, largest T loss and smallest total absolute S/T change for each condition; ties use original ordinal. Selection does not alter the experimental roster or candidate pool. Stable cases can still have poor absolute accuracy.','',
    '|Condition|Role|Query|Clean s/t/v|Corrupted s/t/v|Δs/Δt/Δv pp|','|---|---|---|---|---|---|']
    for r in cases:lines.append('|'+r['condition']+'|'+r['role']+'|'+r['key']+'|'+'/'.join(pp(r['baseline_clean'][m]) for m in ['sIoU','tIoU','vIoU_corrected'])+'|'+'/'.join(pp(r['native'][m]) for m in ['sIoU','tIoU','vIoU_corrected'])+'|'+'/'.join(pp(r['delta'][m]) for m in ['sIoU','tIoU','vIoU_corrected'])+'|')
    lines+=['','## Verification, data use and resources','',
    f"224 original artifact SHA checks and224 exact new-native comparisons passed, including pixels, preprocessed inputs, boxes and both offset logits. Original per-parent frame grids span20–200 frames; an initial protocol wording that generalized the48-frame smoke case was corrected without changing any input or sampling. Independent exhaustive enumeration checked64 candidate sets. All {audit['dual_implementation_metric_calls']} metric calls used two implementations. Two CPU candidate contracts passed; frozen model state hash was unchanged. All224 predictions and64 candidate sets were sealed before streaming only32 authorized GT keys for scoring. Old cache containers include historical expert payloads, but the new worker receives only sanitized native fields and invokes no experts. Constructor annotation file opens are denied.",
    '',f"Cumulative new GPU-process wall allocation **{cost:.3f}s ({cost/60:.2f}min)** includes loading, pixel transformations, capture and engineering failures. CPU development/analysis/reporting excluded. New model backwards0; optimization steps0; new expert calls0; new downloads0. Existing predictions are reused as C0 reference; missing evidence/layers required224 new two-offset full native captures, not adaptation passes.",
    '', 'All failures and revisions, if any, remain local with receipts. The original32-parent cohort is historically exposed test development, not source-training data or fresh evaluation. Official training provenance cannot certify unknown pretraining overlap. Current production methods, PTD, the previous Round1/2 evidence and HC2 transfer selection remain unchanged.','',
    '![C0 paired frozen effects](analysis/c0_effects.png)','',
    '![C1 candidate support](analysis/c1_support.png)','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained');x=np.arange(6)
    for j,(m,label) in enumerate([('sIoU','Spatial'),('tIoU','Temporal'),('vIoU_corrected','Video')]):
        vals=[a[c]['delta'][m] for c in CONDITIONS[1:]];y=np.array([v['mean']*100 for v in vals]);lo=np.array([v['ci95'][0]*100 for v in vals]);hi=np.array([v['ci95'][1]*100 for v in vals]);ax.errorbar(x+(j-1)*.2,y,yerr=[y-lo,hi-y],fmt='o',capsize=3,label=label)
    ax.axhline(0,color='grey');ax.set_xticks(x,CONDITIONS[1:],rotation=20,ha='right');ax.set(ylabel='Corrupted minus clean (pp)',title='C0: frozen Vid-source on32 VidSTG parents, paired95% CI');ax.legend();ax.grid(axis='y',alpha=.2);fig.savefig(OUT/'analysis/c0_effects.png',dpi=160);fig.savefig(OUT/'analysis/c0_effects.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained');x=np.arange(4)
    for ax,task in zip(axes,['S','T']):
        vals=[b[c]['gain_'+task] for c in MEDIUM];y=np.array([v['mean']*100 for v in vals]);lo=np.array([v['ci95'][0]*100 for v in vals]);hi=np.array([v['ci95'][1]*100 for v in vals]);ax.bar(x,y,color='steelblue' if task=='S' else 'darkorange');ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='none',color='black',capsize=3);ax.set_xticks(x,MEDIUM,rotation=25,ha='right');ax.set(title=task+' candidate oracle',ylabel='Target-metric gain over native (pp)');ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'analysis/c1_support.png',dpi=160);fig.savefig(OUT/'analysis/c1_support.svg');plt.close(fig);print(OUT/'REPORT.md')


if __name__=='__main__':run()
