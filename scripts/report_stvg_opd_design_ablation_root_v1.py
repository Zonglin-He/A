"""Final human-readable interpretation from complete, viewed, fixed evidence."""
from pathlib import Path
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_design_ablation_common_v1 import BASE,PUB,read,write,sha,verify


def pp(value):
    lo,hi=value['ci95']
    return f"{value['mean']*100:+.3f} [{lo*100:+.3f}, {hi*100:+.3f}]"


def direction(cell):
    lo,hi=cell['Full_minus_control_ci95_pp']
    if lo>0:return 'Full higher; paired interval above zero'
    if hi<0:return 'Control higher; paired interval below zero'
    return 'paired interval includes zero'


def run():
    verify()
    root=read(BASE/'actual_root/ROOT_ALL_FITS_READBACK.json')
    visual=read(BASE/'ROOT_VISUAL_REVIEW.json')
    assert root['status']==visual['status']=='pass'
    assert root['counts']['complete_fits']==5376
    assert read(BASE/'actual_fixed_case_views002/COMPLETION.json')['cases']==6
    table=read(PUB/'COMPONENT_TABLE.json');stats=read(PUB/'ALL_STATISTICS.json')
    cost=read(PUB/'COST_AND_EXPERT_QUALITY.json');coverage=read(PUB/'COVERAGE_ANALYSIS.json')
    labels={'whole_video':'w/o Event-Scoped Sampling','carry_query':'w/o Fast-State Reset','full_LN_carry':'w/o Slow-State Attenuation'}
    names=['A_hc2_cross_clean','A_vidstg_cross_clean','A_hc2_same_5percent','A_vidstg_same_5percent']
    columns=['Cross → HC2','Cross → VidSTG','Corrupt HC2','Corrupt VidSTG']
    effect_assessment={}
    for label in labels.values():
        cells=table['table'][label]
        effect_assessment[label]=dict(
            positive_paired_intervals=sum(c['Full_minus_control_ci95_pp'][0]>0 for c in cells),
            negative_paired_intervals=sum(c['Full_minus_control_ci95_pp'][1]<0 for c in cells),
            intervals_including_zero=sum(c['Full_minus_control_ci95_pp'][0]<=0<=c['Full_minus_control_ci95_pp'][1] for c in cells),
            settings=len(cells))
    coverage_metrics=[b['statistics']['metrics']['Full_minus_control_v'] for bins in coverage['bins'].values() for b in bins]
    coverage_nonempty=[m for m in coverage_metrics if m is not None]
    coverage_zero=sum(m['ci95'][0]<=0<=m['ci95'][1] for m in coverage_nonempty)
    text=['# Fixed component ablations and event-coverage Figure 3','',
        'All three requested interventions, both directions, all conditions and both fixed cross-domain orders are complete. '
        'The table combines seven preserved P2/P3 rows and three new matched rows. '
        'Uncertain and reversed differences remain in the table and companions; no parameter or checkpoint was selected from these results.','',
        'The target populations are the original 128 historically exposed parents per dataset, one fixed query per parent. '
        'Cross-domain clean averages the two original stream orders within each parent. Same-domain corruption averages '
        'frame drop, frame freeze, motion blur, occlusion and exposure at 5% before parent aggregation. '
        'There are 5,376 new complete formal fits, distinct from 72 qualification arrivals / 144 complete qualification fits. '
        'All formal arms/conditions/orders/targets sealed before new annotations. '
        'The original completed P1–P6 suite remains unchanged. TubeDETR adaptation is outside this run.','',
        'Source checkpoints and main settings are fixed: HC2 lr/sigma/tau/rounds/writeback .01/.025/.05/40/1/16; '
        'VidSTG .03/.1/.25/10/1/8, 32 antithetic actions and joint 1,792 parameters. '
        'Whole-video samples four positions from the same original observed RGB grid; it adds no decoded frames and changes only expert observation scope. '
        'Query carry inherits only the 256-coordinate residual while Adam resets and the original LN attenuation remains. '
        'Full LN carry uses alpha=1 with query/Adam reset. Native WHEN and final-round readout remain identical.','',
        '## Evidence supported by these new controls','',
        f"Slow-state attenuation has {effect_assessment['w/o Slow-State Attenuation']['positive_paired_intervals']} of four "
        'paired intervals above zero. This is the most consistent new component evidence on these saved cohorts, '
        'with unadjusted intervals; the Corrupt HC2 lower endpoint is only +0.003471 pp. '
        'It is not a claim of generalization to a new population.','',
        f"Fast-state reset has {effect_assessment['w/o Fast-State Reset']['positive_paired_intervals']} of four paired intervals above zero "
        '(Cross VidSTG); the other three include zero. Event-scoped sampling has '
        f"{effect_assessment['w/o Event-Scoped Sampling']['positive_paired_intervals']} of four paired intervals above zero: "
        'three average differences favor Full and Corrupt VidSTG favors whole-video, but all four intervals include zero. '
        'The event-scope comparison therefore does not establish a stable overall advantage.','',
        f'All {coverage_zero} of {len(coverage_nonempty)} nonempty coverage-bin paired intervals include zero. '
        'The fixed figure preserves these uncertain and reversed differences; it does not establish a monotonic '
        'coverage trend or a coverage-driven causal benefit.','',
        '## Complete component table','',
        'Absolute parent-macro m_vIoU (%). Query-macro is identical for these balanced one-query-per-parent cohorts.','',
        '| Configuration | '+' | '.join(columns)+' |','|---|---:|---:|---:|---:|']
    ordered=['Full method','w/o Event-Scoped Sampling','w/o Action-Reward Correspondence','w/o On-Policy Refresh',
        'w/o Query Residual','w/o LayerNorm Adaptation','w/o Slow-State Consolidation','w/o Fast-State Reset',
        'w/o Slow-State Attenuation','Frozen source']
    for label in ordered:text.append('| '+label+' | '+' | '.join(f"{c['mean_percent']:.3f}" for c in table['table'][label])+' |')
    text+=['','## Paired component effects','',
        'Differences are Full minus the control in percentage points, with 10,000 paired-parent bootstrap draws '
        '(fixed seed 20261011). Intervals are conditional on the saved, historically exposed cohorts and are not multiplicity adjusted. '
        'An interval including zero does not establish either equivalence or a stable advantage.','',
        '| Control | Setting | Full − control (pp), paired 95% CI | Reading |','|---|---|---:|---|']
    for arm,label in labels.items():
        for i,cell in enumerate(table['table'][label]):
            ci=cell['Full_minus_control_ci95_pp']
            text.append(f"| {label} | {columns[i]} | {cell['Full_minus_control_pp']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {direction(cell)} |")
    text+=['','## Fixed-bin event-coverage analysis','',
        'Coverage is intersection(native interval, half-open GT annotation support) divided by GT annotation duration. '
        'The annotation support ends one physical frame after the last annotated frame. This descriptive covariate differs from tIoU; '
        'the original official task scorer is unchanged. Four bins were fixed before new predictions and both orders are averaged within each parent. '
        'GT participates only in postseal grouping and evaluation. A conditional difference does not establish that coverage causes that difference.','',
        'Each control and Full retain their own declared online LayerNorm trajectory. Their paired final difference therefore includes '
        'the consequences of earlier arrivals; the coverage grouping does not hold inherited state equal at each query. '
        'The two saved cross-domain orders are averaged as fixed orders, not a sample of all possible stream orders.','',
        'The official joint vIoU numerator uses spatial overlap only inside the native/GT temporal intersection. '
        'When that intersection is absent, both paired vIoU values are zero regardless of spatial changes. '
        'A coverage-dependent vIoU difference can therefore partly reflect the joint metric’s attainable range. '
        'It cannot by itself establish better expert evidence or explain the complete mechanism.','',
        '| Direction | Coverage bin | Parents | Event-scoped − whole-video (pp), paired 95% CI |',
        '|---|---|---:|---:|']
    for name,bins in coverage['bins'].items():
        for b in bins:
            metric=b['statistics']['metrics']['Full_minus_control_v']
            interval=f"[{b['lo']:g}, {b['hi']:g}"+(']' if b['last_inclusive'] else ')')
            text.append(f"| {columns[names.index(name)]} | {interval} | {b['n_parents']} | {pp(metric) if metric else 'Unmeasured: empty bin'} |")
    text+=['','![Fixed coverage bins](figure3_event_coverage.png)','',
        '## State trajectories and negative tails','',
        'Total control effect is After−Frozen, current effect is After−Before, and inherited effect is Before−Frozen. '
        'These are decompositions of each control trajectory, not substitutes for its paired contrast with Full. '
        'Every individual condition, order and parent is retained in ALL_STATISTICS, including severe harmful tails.','',
        '| Setting / control | Total vs Frozen (pp), 95% CI | Current vs Before | Inherited vs Frozen | >5 / >20 pp harmful parents |',
        '|---|---:|---:|---:|---:|']
    for name in names:
        for arm in labels:
            m=stats[name][arm]['pooled']['metrics'];tail=m['delta_total_v']
            text.append('| '+name+' / '+arm+' | '+' | '.join(pp(m[k]) for k in ['delta_total_v','delta_current_v','delta_inherited_v'])+
                f" | {tail['harm_gt5pp_sources']} / {tail['harm_gt20pp_sources']} |")
    text+=['','## Actual expert calls and observed quality','',
        'Nominal selected observations, attempted calls, admitted references and GT-evaluable references use different denominators. '
        'Expert quality below is an unweighted mean over arrivals with at least one GT-evaluable admitted reference; '
        'unannotated frames are missing, not zero-IoU failures. State arms reuse the sealed expert references. '
        'Shared capture is counted once. Stored fit wall time includes synchronous audits and is not full cold-start deployment latency.','',
        '| Setting / control | Nominal / attempted / admitted / GT-evaluable | Expert GT IoU on valid arrivals | Fit wall seconds per arrival |',
        '|---|---:|---:|---:|']
    for name in names:
        for arm in labels:
            c=cost[name][arm];quality=c['expert_GT_IoU_mean_over_available_arrivals']
            text.append(f"| {name} / {arm} | {c['nominal_selected_observations']} / {c['attempted_observations']} / {c['admitted_references']} / {c['admitted_references_with_GT']} | "+
                (f'{quality:.6f}' if quality is not None else 'Unmeasured')+f" | {c['fit_GPU_seconds_per_arrival']:.6f} |")
    q=read(BASE/'ROOT_QUALIFICATION_READBACK.json')['actual_new_DINO_calls']
    formal=sum(read(BASE/'stages'/name/'PREDICTION_BARRIER.json')['actual_new_DINO_calls'] for name in names)
    text+=['',f'Actual new DINO calls: {q} during qualification and {formal} during formal capture ({q+formal} total). '
        'Qualification outputs never enter the formal streams. Reuse counts and measured observed/unobserved effects remain in complete anonymous rows.','',
        '## Fixed historical cases and verification','',
        'The six completed P3 cases and their display frames were fixed before any new GT read. '
        'Their historical success/current-harm/expert-mismatch labels do not label the new controls as successful or harmful. '
        'All four saved fits per case are read back; sample/GT IoU is checked independently on each arm’s GT-evaluable admitted frames. '
        'Signal curves consequently have different observation denominators. Empty/no-GT evidence is explicitly missing. '
        'Offline best samples are diagnostic and never choose a deployment output. '
        'Six private RGB sheets use identical previously fixed physical frames across Frozen, Full and the three new controls; '
        'the decoded pixels and corruption specs match the original saved inputs. RGB/query/GT geometry remain private.','',
        'The final fixed-case signal PNG/PDF uses visual003, which shortens clipped axis labels without changing '
        'any of the 24 saved curves or case/frame selection. Original provisional images and their hashes remain preserved.','',
        '![Unchanged fixed-case signals with readable labels](fixed_historical_case_signals_visual003.png)','',
        f"The actual root reread {root['counts']['complete_fits']} complete fits, {root['counts']['fit_rounds']} rounds, "
        f"{root['counts']['state_coordinates']} state coordinates, {root['counts']['matched_inputs']} matched inputs and "
        f"{root['counts']['qualified_formal_pairs']} qualified/formal complete pairs. "
        'The postseal official scorer is compared with independent dense arithmetic; source, complete query/LN/Adam lifetimes, '
        'input/native bindings, saved-math dictionaries and second byte checks pass. '
        'Actual figure and RGB views are bound by ROOT_VISUAL_REVIEW. The public audit independently recomputes anonymous '
        'scalars, parent bootstrap, coverage bins, tail counts, costs and all old/new table cells. '
        'This does not independently reimplement the complete decoder Jacobian or guarantee future GPU precision/OOM safety.','',
        'Two copied historical table display values are corrected from the complete sealed row means: '
        'shuffled-feedback Cross VidSTG 9.517→9.516 and Frozen Corrupt HC2 29.093→29.094. '
        'Original predictions, rows and scores are untouched. The failed unsealed table projection and CPU002 status writer remain preserved; '
        'the separate CPU003 projection annotates each immutable result on its initial write.','',
        'Scope: these three interventions and fixed coverage analysis only. No retuning, production promotion, extra backbone, '
        'EATA restoration or historical queue restart. Final closure additionally binds actual exact remote content and archive verification.']
    text='\n'.join(text)+'\n'
    (PUB/'ACTUAL_ROOT_REVIEW.md').write_text(text)
    (ROOT/'docs/STVG_OPD_DESIGN_ABLATION_ROOT_REVIEW.md').write_text(text)
    write(BASE/'ROOT_INTERPRETATION.json',dict(status='actual_complete_viewed_results_interpreted',
        root_readback_sha256=sha(BASE/'actual_root/ROOT_ALL_FITS_READBACK.json'),visual_sha256=sha(BASE/'ROOT_VISUAL_REVIEW.json'),
        report_sha256=sha(PUB/'ACTUAL_ROOT_REVIEW.md'),source_sha256=sha(Path(__file__)),all_negative_results_retained=True,
        component_effect_assessment=effect_assessment,
        coverage_nonempty_bins=len(coverage_nonempty),coverage_intervals_including_zero=coverage_zero,
        no_new_parameter_or_case_selection=True,CPU_only=True,model_calls=0,experiment_complete=False,time=time.time()))
    print('DESIGN_ROOT_REPORT_WRITTEN',sha(PUB/'ACTUAL_ROOT_REVIEW.md'),flush=True)


if __name__=='__main__':run()
