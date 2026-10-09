"""Write the P3 review only from complete root and actually viewed evidence."""
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,PUB,read,sha
activate()
from scripts.audit_stvg_opd_later_public_v1 import aggregate


def value(x,scale=100):
    a,b=x['ci95'];return f"{x['mean']*scale:+.3f} [{a*scale:+.3f}, {b*scale:+.3f}]"


def run():
    out=PUB/'P3';math=read(out/'ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json')
    visual=read(BASE/'P3_ROOT_VISUAL_REVIEW.json');assert math['status']==visual['status']=='pass'
    stats=read(out/'ROOT_STATISTICS.json');cost=read(out/'COST.json')
    report=['# Original fixed OPD P3: parameter scope and persistent state',
        '', 'Full exceeds Query-only in HC2 same-domain corruption and VidSTG cross-domain clean; '
        'the added query residual exceeds LN-only clearly only in VidSTG cross-domain clean. '
        'Full minus joint alpha0 is positive in HC2 cross-domain clean, with intervals containing zero in the other three setting means. '
        'The complete joint design is not uniformly better than every ablation.',
        '', 'Both targets use the original exposed 128-parent confirmation cohorts, one query per parent. '
        'Cross-domain clean has both original orders; same-domain has five physical burst families at 5%, one order. '
        'All four arms/conditions/orders/directions globally sealed before GT. '
        'There are 7,168 logical fits: 5,376 new formal fits and 1,792 exact complete-stream Full aliases. '
        'Qualification predictions are excluded.',
        '', 'TA-STVG sources remain `TASTVG_VidSTG.pth` (5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83) '
        'and `TASTVG_HCSTVG2.pth` (47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036). '
        'Cross uses the other dataset source; same-domain uses its own source. HC2 lr/sigma/tau/rounds/writeback '
        'are .01/.025/.05/40/1/16; VidSTG .03/.1/.25/10/1/8. M=32 antithetic Gaussian actions, '
        'original Uniform4, admitted Top1 frozen DINO, per-query Adam/residual reset and final-round output remain fixed. '
        'Query-only updates 256 query parameters; LN-only updates 1,536 LN parameters; joint alpha0 and Full update 1,792. '
        'Joint alpha0 discards the LN writeback; all streams start independently from source per condition/order/arm. '
        'Native WHEN is fixed and every temporal delta is zero.',
        '', 'All numbers below are vIoU percentage points with 10,000 paired parent bootstrap draws '
        '(seed 20261006), conditional on these saved histories. Conditions/orders are averaged within query before parent aggregation; '
        'query-macro equals parent-macro because this cohort has one query per parent. Intervals are not multiplicity adjusted.',
        '', '| Setting | Full − Query-only | Full − LN-only | Full − joint alpha0 |',
        '|---|---:|---:|---:|']
    for name,controls in stats['Full_minus_controls_condition_mean'].items():
        report.append('| '+name+' | '+' | '.join(value(controls[a]['metrics']['Full_minus_v']) for a in ['query_only','LN_only','joint_alpha0'])+' |')
    report+=['', '| Setting / arm | After vIoU (%) | Total vs Frozen | Current vs Before | Inherited Before vs Frozen | >5 / >20 pp harmful parents |',
        '|---|---:|---:|---:|---:|---:|']
    for name in stats['by_stage']:
        with gzip.open(PUB/name/'ROWS.jsonl.gz','rt') as f:rows=list(map(json.loads,f))
        for arm in ['query_only','LN_only','joint_alpha0','on_policy']:
            m=aggregate([r for r in rows if r['arm']==arm],['After_v','delta_total_v','delta_current_v','delta_inherited_v'])[0]['metrics']
            tail=m['delta_total_v']
            report.append('| '+name+' / '+arm+' | '+f"{m['After_v']['mean']*100:.3f}"+' | '+
                ' | '.join(value(m[k]) for k in ['delta_total_v','delta_current_v','delta_inherited_v'])+
                f" | {tail['harm_gt5pp_sources']} / {tail['harm_gt20pp_sources']} |")
    report+=['', 'LN-only retains much of the Full gain in HC2; this does not establish an incremental query benefit there. '
        'Query-only has exactly zero inherited delta because the query residual resets and LN stays at source. '
        'Joint alpha0 also has zero inherited delta, yet its current adaptation can be useful. '
        'Full Before−Frozen is a decomposition of its own trajectory, not the causal Full−alpha0 contrast: '
        'turning off carry changes subsequent adaptation states and current effects. '
        'A small paired Full−alpha0 difference must not be relabelled as the larger inherited component.',
        '', 'Severe failures, condition-specific results, per-order results, observed/unobserved effects, '
        'expert quality, motion/duration strata and query types remain in ROOT_STATISTICS, ALL_PARENT_EFFECTS, '
        'FAILURE_STRATA, COST and complete anonymous stage rows. No failing source is dropped or used to retune.',
        '', '| Setting / arm | Stored fit wall seconds / arrival | CPU math seconds / arrival |',
        '|---|---:|---:|']
    for name,arms in cost.items():
        for arm,c in arms.items():report.append(f"| {name} / {arm} | {c['real_fit_seconds_per_arrival']:.6f} | {c['independent_CPU_math_seconds_per_arrival']:.6f} |")
    report+=['', 'Stored fit wall time includes synchronous numerical recording. Shared capture is not multiplied by four; '
        'alias timing is historical cost, not new GPU compute or cold deployment latency. Full and ablations were run '
        'in serial historical streams; these cost comparisons are descriptive rather than matched hardware microbenchmarks.',
        '', f"The actual root checked {math['counts']['logical_arrivals']} complete saved mathematical dictionaries, "
        f"{math['counts']['rounds']} rounds, {math['counts']['state_coordinates']} full-state coordinates, "
        f"{math['counts']['matched_input_checks']} matched input comparisons, {math['counts']['dense_metric_scalars']} "
        f"independent dense scalars and {math['counts']['observed_unobserved_checks']} observed/unobserved checks. "
        f"The largest independent dense discrepancy is {math['maximum_dense_error']:.9g}. "
        'Receipt-aware dispatch preserves original P0/P2/006 and new scoped audit dictionaries exactly. '
        'All inactive parameters, active Adam scopes, query reset, source reset, LN inheritance/writeback and Native intervals are checked.',
        '', 'Six distinct success/harm/expert-mismatch cases were selected post hoc after all parent rows were read. '
        'Each saved Full signal is accompanied by three complete scope-control signal chains on the same query/input/order. '
        'Sample-vs-GT arithmetic is checked independently; recorded GPU actions are used when saved, while historical P0 '
        'sample logits use an explicitly labelled offline float64 sigmoid. Offline best samples never select a deployment output. '
        'Four report plot pairs and six private real RGB case sheets were actually viewed. '
        'These cases describe mechanisms and are not independent efficacy estimates.',
        '', 'Public code/configuration/anonymous rows/scalars/plots include every negative result. '
        'No RGB/query/caption/GT geometry/weights/raw logits/boxes/actions/gradient/Adam/cache payload is exported. '
        'The root performs no model/optimizer calls, proves saved parameter arithmetic and state/dense integrity, '
        'and does not independently reimplement the entire decoder Jacobian or prove CUDA transcendental kernels. '
        'Historical numerical assertion failures remain preserved; bounded audit bridges do not change the science.',
        '', 'Decision: close only P3 after actual remote publication and archive verification, then continue the original '
        'fixed P4 robustness phase. No method promotion, retuning, new algorithm/expert or restoration of paused EATA/historical queues. '
        'P4–P6 retain their own GPU qualification/global seal/CPU/root/view/public/archive obligations; paper suite remains incomplete.']
    text='\n'.join(report)+'\n';(out/'ACTUAL_ROOT_REVIEW.md').write_text(text)
    (ROOT/'docs/STVG_OPD_REVISED_P3_ROOT_REVIEW.md').write_text(text)
    print(json.dumps(dict(status='written',complete_root_sha256=sha(out/'ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json'),actual_visual_sha256=sha(BASE/'P3_ROOT_VISUAL_REVIEW.json'))))


if __name__=='__main__':run()
