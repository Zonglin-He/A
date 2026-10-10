"""P4 report from complete sealed results and actually reviewed root evidence."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,PUB,read,sha
activate()


def value(x,scale=100):
    a,b=x['ci95'];return f"{x['mean']*scale:+.3f} [{a*scale:+.3f}, {b*scale:+.3f}]"


def run():
    out=PUB/'P4';math=read(out/'ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json')
    visual=read(BASE/'P4_ROOT_VISUAL_REVIEW.json');assert math['status']==visual['status']=='pass'
    stats=read(out/'ROOT_STATISTICS.json');cost=read(out/'COST.json')
    mechanism=read(out/'ACTUAL_ROOT_ALL_CONDITION_MECHANISM_COST.json')['datasets']
    summary={}
    for name,conditions in stats['by_stage'].items():
        allmetrics=[c['on_policy']['metrics']['delta_total_v'] for c in conditions.values()]
        summary[name]=dict(positive_CI=sum(m['ci95'][0]>0 for m in allmetrics),negative_CI=sum(m['ci95'][1]<0 for m in allmetrics),
            interval_includes_zero=sum(m['ci95'][0]<=0<=m['ci95'][1] for m in allmetrics))
    hc=stats['by_stage']['P4_hc2']['clean']['on_policy']['metrics']['delta_total_v']
    vid=stats['by_stage']['P4_vidstg']['clean']['on_policy']['metrics']['delta_total_v']
    report=['# Original fixed OPD P4: same-domain physical-burst robustness','',
        f"Full exceeds Frozen in all sixteen conditions in each dataset, with positive paired parent 95% intervals. Clean gains are HC2 {value(hc)} pp and VidSTG {value(vid)} pp. "
        'On the method\'s own trajectories, the mean current-query update is negative in all HC2 conditions; inherited state supplies the positive total mean. '
        'The current-query mean is positive in VidSTG, with uncertainty reported separately. Severe individual harms and reward/task mismatches remain. '
        'This phase measures Full against its frozen same-domain source on clean input and fifteen physical-burst conditions. '
        'It tests robustness of the fixed method; it does not compare new algorithms or choose new parameters.', '',
        'All 15,504 new formal fits globally sealed before GT: HC-STVG2 validation has 237 parent movies, one clip/query per movie '
        '(3,792 arrivals), and VidSTG test has 732 parent videos, one query per parent (11,712 arrivals). '
        'There are sixteen conditions: clean and frame drop, frame freeze, motion blur, occlusion and exposure, each at 2.5%, 5% and 10% '
        'physical burst coverage. Every condition starts independently from its own same-domain source. There are no historical aliases '
        'and no qualification predictions among formal arrivals.', '',
        'TA-STVG source checkpoints are `TASTVG_HCSTVG2.pth` '
        '(47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036) and `TASTVG_VidSTG.pth` '
        '(5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83). '
        'HC2 lr/sigma/tau/rounds/writeback are .01/.025/.05/40/1/16; VidSTG .03/.1/.25/10/1/8. '
        'M=32 antithetic Gaussian actions, Uniform4, admitted Top1 frozen DINO, joint 1,792 active parameters, per-query Adam/query-residual reset, '
        'within-condition LN inheritance and final-round output stay fixed. Native WHEN remains unchanged.', '',
        'Numbers below use parent macro averaging with 10,000 paired parent bootstrap draws (seed 20261006). '
        'One query per parent makes query macro and parent macro equal. The selected query roster and saved stream order remain fixed. '
        'These intervals are conditional on the saved histories and are not multiplicity adjusted. No formal result is used to retune. '
        'The efficacy plot averages the five corruption families at each nonzero coverage; its shaded bands are marginal intervals. '
        'The paired-effect plot and complete tables retain every individual condition. The ranked-tail plot shows every parent for clean and the five 5% conditions; '
        'all sixteen condition tails remain in the complete anonymous data.', '',
        '| Dataset | Conditions with positive 95% interval | Negative interval | Interval includes zero |',
        '|---|---:|---:|---:|']
    for name,s in summary.items():report.append(f"| {name} | {s['positive_CI']} / 16 | {s['negative_CI']} / 16 | {s['interval_includes_zero']} / 16 |")
    report+=['', '| Dataset / condition | Frozen vIoU (%) | Full vIoU (%) | Full − Frozen (pp), 95% CI | Current: After − Before | Inherited: Before − Frozen | >5 / >20 pp harmful parents |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,conditions in stats['by_stage'].items():
        for condition,arms in conditions.items():
            m=arms['on_policy']['metrics'];tail=m['delta_total_v']
            report.append(f"| {name} / {condition} | {m['Frozen_v']['mean']*100:.3f} | {m['After_v']['mean']*100:.3f} | "+
                ' | '.join(value(m[k]) for k in ['delta_total_v','delta_current_v','delta_inherited_v'])+
                f" | {tail['harm_gt5pp_sources']} / {tail['harm_gt20pp_sources']} |")
    report+=['', 'Current and inherited effects decompose the method\'s own stream. Inherited Before−Frozen is not an alpha0 causal contrast. '
        'All negative parent effects, gross gain/loss and severe tails remain in the complete condition tables and anonymous rows. '
        'Temporal deltas are zero because WHEN is fixed. This phase cannot establish a benefit from temporal-head adaptation.', '',
        '| Dataset / condition | Observed-frame IoU current delta (pp) | Valid queries | Unobserved-frame IoU current delta (pp) | Valid queries | Fit wall seconds / arrival | Shared capture seconds / arrival |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def number(v):return 'NA' if v is None else f'{v*100:+.3f}'
    for ds,conditions in mechanism.items():
        for condition,m in conditions.items():report.append(f"| {ds} / {condition} | {number(m['observed_positions_iou_mean'])} | {m['observed_valid_query_count']} | {number(m['unobserved_positions_iou_mean'])} | {m['unobserved_valid_query_count']} | {m['actual_fit_wall_seconds_per_arrival']:.6f} | {m['actual_shared_capture_seconds_per_arrival']:.6f} |")
    report+=['', 'Observed and unobserved means use different frame populations; neither is a randomized causal intervention. '
        'The stored fit wall time includes synchronous numerical recording. Shared capture is reported separately. '
        'Recorded expert-forward time may include reuse and is not a cold expert latency measurement.', '',
        '| Dataset | Fit wall seconds / arrival | Shared capture seconds / arrival | CPU mathematical audit seconds / arrival | Original execution new DINO calls |',
        '|---|---:|---:|---:|---:|']
    for name,arms in cost.items():
        c=arms['on_policy'];report.append(f"| {name} | {c['real_fit_seconds_per_arrival']:.6f} | {c['shared_capture_seconds_per_arrival']:.6f} | {c['independent_CPU_math_seconds_per_arrival']:.6f} | {c['actual_new_DINO_calls_in_recorded_original_execution']} |")
    n=math['counts'];signal=read(out/'ACTUAL_ROOT_CASE_SIGNAL_CHAINS.json')
    report+=['', f"The actual root read every {n['logical_arrivals']} saved fit and complete mathematical dictionary, {n['rounds']} rounds, "
        f"{n['state_coordinates']} state coordinates, {n['input_bindings_checked']} input bindings, {n['dense_metric_scalars']} independent dense scalars, "
        f"{n['observed_unobserved_checks']} observed/unobserved effects and {n['actual_qualified_formal_pairs_bitwise']} complete qualification/formal pairs. "
        f"It also repeated {n['second_opaque_prediction_input_receipt_checks']} prediction/input/receipt SHA checks. "
        f"Maximum independent dense discrepancy was {math['maximum_dense_error']:.9g}. "
        'Each condition\'s source reset, per-query residual/Adam reset, LN inheritance/writeback, original Native intervals and receipt-aware numerical dictionaries were actually checked.', '',
        f"Six distinct post-hoc success/harm/expert-reward mismatch cases were selected only after full population readback. "
        f"All original saved GPU action sample-vs-GT chains were independently checked ({signal['independent_sample_GT_IoU_checks']} comparisons). "
        'The six private RGB sheets re-decode and apply the original physical corruption, requiring exact original input pixel SHA and corruption-spec equality. '
        'Five report plot pairs and all six real case sheets were actually viewed. '
        'Offline best samples are diagnostic only and never select deployment outputs. Cases are not independent efficacy estimates.', '',
        'Public implementation/configuration/protocols/anonymous rows/scalars/plots retain all negative findings. '
        'Private RGB/query/caption/GT geometry/weights/raw logits/boxes/actions/fit/gradient/Adam/cache payloads are excluded. '
        'The root makes no model or optimizer calls. Saved mathematical/state/dense integrity is checked without independently implementing the entire decoder Jacobian '
        'or proving CUDA transcendental kernels. Historical numerical failures remain preserved.', '',
        'Decision: close only the fixed P4 phase after actual remote verification and archive maintenance, then continue original P5 budget/cost experiments. '
        'No retuning, method promotion or new scientific algorithm is introduced. P5/P6 require their own real qualification/global deployment seal/CPU/root/view/public/archive closing. '
        'EATA and historical paused queues remain paused; the paper suite remains incomplete.']
    text='\n'.join(report)+'\n';(out/'ACTUAL_ROOT_REVIEW.md').write_text(text)
    (ROOT/'docs/STVG_OPD_REVISED_P4_ROOT_REVIEW.md').write_text(text)
    print(json.dumps(dict(status='written',conditions=summary,complete_root_sha256=sha(out/'ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json'),actual_visual_sha256=sha(BASE/'P4_ROOT_VISUAL_REVIEW.json'))))


if __name__=='__main__':run()
