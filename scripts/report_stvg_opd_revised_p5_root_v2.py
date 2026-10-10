"""P5 report from complete preserved budgets and actual root/view evidence."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,PUB,read,sha
activate()


def value(x,scale=100):
    a,b=x['ci95'];return f"{x['mean']*scale:+.3f} [{a*scale:+.3f}, {b*scale:+.3f}]"


def run():
    out=PUB/'P5';state=read(out/'ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json')
    visual=read(BASE/'P5_ROOT_VISUAL_REVIEW.json');assert state['status']==visual['status']=='pass'
    analysis=read(out/'ACTUAL_ROOT_BUDGET_CONTRASTS.json');assert analysis['status']=='pass'
    report=['# Fixed OPD P5: complete budget, cost and feedback-chain review','',
        'On the fixed historical cross-domain cohort, VidSTG has a positive paired parent interval against Frozen at every K, while all four HC2 intervals contain zero. '
        'VidSTG K4−K2 and K8−K4 intervals contain zero, so the results do not establish a uniformly increasing return to more observations. '
        'Every paired contrast and harmful-source tail is retained, and the original main Uniform4 is unchanged.','',
        'All ten deployment streams globally sealed before GT: two targets × four Uniform K budgets plus two pre-GT unified-configuration appendix streams. '
        'Each target has 128 historical parent sources, one query per parent, two locked source-blocked orders. '
        'The 2,560 logical arrivals contain 2,048 new formal fits and 512 exact complete original K4 stream aliases. '
        'Every budget/order resets to its original source; query residual and Adam reset per query, LN inheritance/writeback and configured last output stay fixed. '
        'Qualifications are never spliced into formal results. Original main Uniform4 remains selected without P5-result tuning.','',
        'Source checkpoints: VidSTG `5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83` → HC2 validation, '
        'HC-STVG2 `47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036` → VidSTG test. '
        'HC main lr .01, sigma .025, tau .05, 40 rounds, LN writeback1/16; Vid main .03/.1/.25, 10 rounds, writeback1/8. '
        'All use M32 antithetic on-policy Gaussian likelihood, detached admitted Top1 frozen-DINO IoU, joint1792 parameters and Native WHEN.','',
        'Scores and intervals below are independently read back from all anonymous rows. Query macro equals parent macro here. '
        'The 10,000 paired parent bootstrap uses seed20261006 and averages the two fixed orders within each parent before resampling. '
        'Intervals are conditional on the historically exposed roster, fixed source checkpoints and order histories; they are not fresh-source stability guarantees. '
        'The full analysis retains both individual orders and all six pairwise budget contrasts per target; intervals are not multiplicity-adjusted.','',
        '| Target | K | After−Frozen vIoU, pp [95% CI] | After−Before, pp | Before−Frozen, pp | Harm >5/>20pp parents |','|---|---:|---|---|---|---:|']
    compact={}
    for ds in ['hc2','vidstg']:
        compact[ds]={}
        for k in [1,2,4,8]:
            r=analysis['datasets'][ds][str(k)]['all_orders'];m=r['metrics'];total=m['delta_total_v']
            compact[ds][str(k)]=dict(total=total,current=m['delta_current_v'],inherited=m['delta_inherited_v'])
            report.append(f"| {ds} | {k} | {value(total)} | {value(m['delta_current_v'])} | {value(m['delta_inherited_v'])} | {total['harm_gt5pp_sources']}/{total['harm_gt20pp_sources']} |")
    report+=['','The inherited/current components describe each budget’s own state trajectory. Before−Frozen is not a Full−alpha0 causal contrast. '
        'K changes the observation support and ensuing state history. Admitted-observed versus other GT frames are different frame populations, '
        'so their mean differences are descriptive; they are not an independent causal transfer experiment.','',
        '| Target | Matched After contrast | vIoU, pp [95% CI] |','|---|---|---|']
    for ds in ['hc2','vidstg']:
        for label,group in analysis['pairwise_budget_contrasts'][ds].items():
            report.append(f"| {ds} | {label.replace('_',' ')} | {value(group['all_orders']['metrics']['v'])} |")
    report+=['','| Target | K | Recorded fit s/arrival | Recorded shared capture s/arrival | Recorded CPU math s/arrival | Original new DINO calls | Peak allocated GiB |','|---|---:|---:|---:|---:|---:|---:|']
    for ds in ['hc2','vidstg']:
        for k in [1,2,4,8]:
            c=analysis['datasets'][ds][str(k)]['all_orders']['actual_cost']
            report.append(f"| {ds} | {k} | {c['real_fit_seconds_per_arrival']:.4f} | {c['shared_capture_seconds_per_arrival']:.4f} | {c['independent_CPU_math_seconds_per_arrival']:.4f} | {c['actual_new_DINO_calls_in_recorded_original_execution']} | {c['maximum_CUDA_peak_allocated']/2**30:.3f} |")
    report+=['','These are actual recorded wall measurements, including synchronous audit recording and original cache conditions. '
        'K4 aliases retain their original measurements; they do not incur new fitting time in this P5 execution. '
        'Shared capture is not repeatedly added once per variant. Expert forward receipts can describe reused cached calls; '
        'they are not cold deployment latency. Decode/corruption, frozen STVG forward and recorded expert-forward components, available-row counts, '
        'actual backward rounds and observed/unobserved denominators remain in COST.json and ACTUAL_ROOT_BUDGET_CONTRASTS.json.','',
        '| Unified appendix target | After−Frozen vIoU, pp [95% CI] | Unified−main K4, pp [95% CI] |','|---|---|---|']
    for ds in ['hc2','vidstg']:
        r=analysis['unified_appendix'][ds]
        report.append(f"| {ds} | {value(r['statistics']['metrics']['delta_total_v'])} | {value(r['unified_minus_main_K4']['metrics']['v'])} |")
    report+=['','The unified appendix was fixed before P0 GT: lr .03, sigma .1, tau .25, 10 rounds, writeback1/16, M32, K4. '
        'It neither replaces main configurations nor constitutes a single-factor K contrast. No parameters or budget were selected from these results.','',
        'Six distinct post-hoc main-budget cases retain successes, current-query harms and cases in which expert reward rises while task score worsens. '
        'The case identity and support are verified against preserved payload/input hashes. Saved original GPU actions are evaluated against official GT '
        'only after global seal; offline best-sample curves never select deployment outputs. Private RGB redecoding must match the original pixel SHA. '
        'The full case signal chains and all negative rows/tails/strata stay in the result package. Cases illustrate mechanisms and are not efficacy estimates.','',
        'Actual root counts: `'+json.dumps(state['counts'],sort_keys=True)+'`. '
        'Every complete saved math dictionary is dispatched by its original pinned receipt, full1792 state/query-and-Adam reset/LN inheritance/writeback/input/Native interval '
        'is read back, dense metrics and observed/unobserved means are independently recomputed, and prediction/input/receipt SHA checks repeat after the full scan. '
        'The original unsealed CPU helper failures (legacy receipt without bytes, duplicate immutable statistics write, equality of differently serialized K4 input containers, and the original pre-revision Vid K4 runtime binding) remain preserved; '
        'separately pinned readback revisions preserve the original predictions/scientific runtime and compare existing statistics exactly. K4 alias/control input pairs are checked against the strict original hash-bound NPZ bridge across source-state hash, frame/pixel identity, Native interval/boxes and full packed experts; new formal input SHA checks remain exact.','',
        'Public arithmetic reproduces all anonymous source statistics, paired intervals, negative tails, strata, recorded costs and evidence bindings. '
        'It does not recreate private inference/GT/sample checks or independently implement the entire decoder Jacobian. '
        'The root makes no new model/optimizer calls and does not claim CUDA transcendental proof or future numerical/OOM safety. '
        'Private RGB/query/caption/GT geometry/boxes/actions/weights/fit/gradient/Adam/cache payloads are excluded from publication.','',
        'Decision: close only original P5 after actual remote content verification and archive maintenance, then implement and qualify the original finite P6 '
        'existing temporal-signal/head and post-deployment-seal GT-head-oracle/failure diagnostic. No retuning, new expert/algorithm/scorer/gate/memory or method promotion. '
        'EATA and all historical paused queues remain paused; the entire paper suite is still incomplete.']
    text='\n'.join(report)+'\n';(out/'ACTUAL_ROOT_REVIEW.md').write_text(text)
    (ROOT/'docs/STVG_OPD_REVISED_P5_ROOT_REVIEW.md').write_text(text)
    print(json.dumps(dict(status='written',budget_results=compact,root_sha256=sha(out/'ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json'),visual_sha256=sha(BASE/'P5_ROOT_VISUAL_REVIEW.json'))))


if __name__=='__main__':run()
