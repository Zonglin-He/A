"""Complete anonymous four-arm LN result, current tails and actual-order figures."""
import copy
import csv
import gzip
import io
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read, write, sha
from scripts.stvg_opd_mechanism_LN_order_common_v1 import BASE, PUB, stages, verify
from scripts.finalize_stvg_opd_design_ablation_v1 import statistics
from scripts.review_stvg_opd_supplement_v1 import rows
from scripts.audit_stvg_opd_LN_complete_public_v1 import REL, ARMS, ORDERS, ARM_FIELDS, FIELDS, parent_table, run as public_audit
OUT = ROOT/REL
FIG = ROOT/'output/pdf/stvg_opd_mechanism_review'


def packed(path, value, jsonl=False):
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as f:
            if jsonl:
                for row in value:
                    f.write((json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n').encode())
            else:
                f.write(json.dumps(value, separators=(',', ':'), allow_nan=False).encode())


def run():
    verify()
    root = read(BASE/'ROOT_COMPLETE_READBACK.json')
    assert root['status'] == 'pass' and root['new_formal_fits'] == 768 and root['reused_control_fits'] == 2304
    original = read(PUB/'FOUR_ARM_ORDER_STATISTICS.json')
    assert sha(PUB/'FOUR_ARM_ORDER_STATISTICS.json') == root['statistics_sha256']
    records = []
    for name in stages():
        oldname = name.replace('M_', 'O_')
        new = rows(PUB/name/'ROWS.jsonl.gz')
        oldfile = ROOT/'results/stvg_opd_supplement/2026-10-11/orders'/oldname/'ROWS.jsonl.gz'
        assert sha(oldfile) == original['statistics'][name]['old_rows_sha256']
        part = rows(oldfile)+new
        assert len(part) == 1536
        for r in part:
            r['stage'] = name
        records.extend(part)
    identity = sorted({(r['dataset'], r['source_id']) for r in records})
    assert len(identity) == 256
    ids = {key: n for n, key in enumerate(identity)}
    def anonymous(value, dataset):
        if isinstance(value, dict):
            return {('parent_id' if k == 'source_id' else k):
                    (ids[dataset, v] if k == 'source_id' else anonymous(v, dataset)) for k, v in value.items()}
        if isinstance(value, list):
            return [anonymous(v, dataset) for v in value]
        return value
    public = [dict({k: v for k, v in r.items() if k not in {'source_id', 'query_ordinal'}}, parent_id=ids[r['dataset'], r['source_id']]) for r in records]
    OUT.mkdir(parents=True, exist_ok=False)
    packed(OUT/'ROWS.jsonl.gz', public, True)
    result = copy.deepcopy(original)
    result['statistics'] = {name: anonymous(s, stages()[name]['dataset']) for name, s in original['statistics'].items()}
    packed(OUT/'FOUR_ARM_ORDER_STATISTICS.json.gz', result)
    allarms = {}
    for name, stage in stages().items():
        part = [r for r in records if r['stage'] == name]
        allarms[name] = {}
        for arm in ARMS:
            allarms[name][arm] = {}
            for order in ['pooled']+ORDERS:
                chosen = [r for r in part if r['arm'] == arm and (order == 'pooled' or r['order'] == order)]
                allarms[name][arm][order] = anonymous(statistics(chosen, ARM_FIELDS), stage['dataset'])
    packed(OUT/'ALL_ARM_STATISTICS.json.gz', dict(status='complete_cached_four_arm_current_and_absolute_outcomes',
        statistics=allarms, bootstrap=10000, seed=20261011, unadjusted_CI=True,
        new_model_calls=0, new_DINO_calls=0, new_GT_scoring_calls=0))
    table = parent_table(public)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
    with (OUT/'ALL_PARENT_CONTRASTS.csv.gz').open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as f:
            f.write(stream.getvalue().encode())
    write(OUT/'DESIGN_PUBLIC.json', dict(status='complete_original_historical_LN_component_three_actual_orders',
        targets=['hc2', 'vidstg'], parents_per_target=128, parents=256, orders=ORDERS, arms=ARMS,
        new_formal_fits=768, reused_control_fits=2304, actual_qualification_arrivals=12,
        actual_qualification_repeat_fits=24, active_LN_parameters=1536, frozen_query_parameters=256,
        stored_state_parameters=1792, original_Full_active_parameters=1792,
        config={s['dataset']: s['variant_configs']['LN_only'] for s in stages().values()},
        query_and_Adam_reset_per_query=True, source_reset_per_arm_order=True,
        same_original_source_pixel_expert_frames_config_binding=True,
        bootstrap=10000, seed=20261011, unadjusted_CI=True,
        three_orders_not_independent_parent_populations=True,
        historically_exposed_not_fresh_test=True, original_128_component_cohort_not_C_expanded_905_cohort=True,
        GT_used_to_select_cohort=False, GT_only_after_all_new_formal_global_seal=True,
        new_DINO_calls=0, cached_fit_compute_not_deployment_or_audited_cost=True,
        original_scientific_source_and_runtime_unchanged=True,
        no_new_causal_LN_state_swap=True, no_gate_created=True, production_promoted=False,
        manuscript_written=False, cost32_complete=False, whole_followup_request_complete=False,
        observation_delta_definition='legacy DenseTubeGT timeline; null retained; not P9-P0 common-U genuine-frame endpoint'))
    checked = public_audit(ROOT)
    write(OUT/'PUBLIC_SCALAR_AUDIT.json', checked)
    write(BASE/'ROOT_ANONYMOUS_SCALAR_READBACK.json', dict(checked, original_root_sha256=sha(BASE/'ROOT_COMPLETE_READBACK.json'), time=time.time()))
    lines = ['# Original LN-only: complete four-arm, three-order result', '',
        'All 768 new LN-only fits were sealed across both targets before GT. All 2,304 existing Full/reset/full-carry controls are reused exactly, with the same source, pixels, frames, Native event, detector evidence and fixed configuration. Each target has 128 historical parents and three actual full-stream permutations. The three orders are repeated outcomes of the same parents; they do not create 768 independent parents.', '',
        '## Direct final-output contrasts', '',
        'Parent averages retain all three actual orders. The 10,000 paired parent bootstrap uses the locked seed 20261011; intervals are descriptive and unadjusted. Values are percentage points.', '',
        '| Target | Order | Contrast | Difference (pp) | 95% CI | Negative parents |',
        '|---|---|---|---:|---|---:|']
    for name, s in result['statistics'].items():
        ds = stages()[name]['dataset']
        for order in ['pooled']+ORDERS:
            stat = s['pooled'] if order == 'pooled' else s['per_order'][order]
            for arm in ARMS[1:]:
                m = stat['metrics']['Full_minus_'+arm+'_v']
                lines.append(f"| {ds} | {order} | Full - {arm} | {100*m['mean']:+.3f} | [{100*m['ci95'][0]:+.3f}, {100*m['ci95'][1]:+.3f}] | {m['negative_parent_count']} / 128 |")
    lines += ['', '## Absolute outcomes and current-update negative tails', '',
        'A current-update difference is official vIoU After minus Before. A negative parent counts any negative mean across the three orders; the >5 pp and >20 pp columns separately identify substantive harm. Before minus Frozen is a descriptive entering-state readout, not a direct Full-minus-reset comparison.', '',
        '| Target | Arm | Frozen (%) | Before (%) | After (%) | Current update (pp) | Negative parents | >5 pp harm | >20 pp harm |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, s in allarms.items():
        ds = stages()[name]['dataset']
        for arm in ARMS:
            m = s[arm]['pooled']['metrics']
            d = m['delta_current_v']
            lines.append(f"| {ds} | {arm} | {100*m['Frozen_v']['mean']:.3f} | {100*m['Before_v']['mean']:.3f} | {100*m['After_v']['mean']:.3f} | {100*d['mean']:+.3f} | {d['negative_parent_count']} / 128 | {d['harm_gt5pp_sources']} | {d['harm_gt20pp_sources']} |")
    lines += ['', '## Scope and completion boundaries', '',
        'The original joint Full regime has a positive pooled final-output difference versus LN-only in VidSTG; the HC interval crosses zero. This establishes a target-specific online component effect in the locked historical cohort, not a universal need for query adaptation. It does not isolate one LN state under a fixed-query causal state swap, validate an unlabeled gate, or promote a new production method.', '',
        'Full-minus-reset intervals cross zero in both domains. Full-minus-unattenuated-carry pooled intervals are positive in both of these 128-parent, three-order component cohorts. These populations and orders differ from C Expanded-State (205/700 parents and two orders); the datasets cannot be pooled or treated as independent replications.', '',
        'The companion retains all 3,072 anonymous row outcomes, all 256 paired parents, pooled and each-order statistics, all positive and negative tails, neutral values, no-information trajectories and null diagnostics. Original observed/unobserved diagnostics use the legacy DenseTubeGT timeline, not P9-P0 common-U genuine-frame support. Historical evaluation exposure is retained; this is not a fresh untouched test.', '',
        'The original final CPU scorer counted completed first-formal pairs as if three new arms existed (36). This scope has one new arm: two domains x three orders x two first arrivals = 12. All 12 were already accepted and independently matched. The original source, logs, scoring outputs and science pins were preserved; a separate CPU dispatch corrected only the final aggregate assertion. No fit, GPU, model, detector or new GT scoring was rerun for that recovery.', '',
        'Complete fit/source/state/input/dense and statistical root passed. The figures must still receive actual PDF pixel view, and all source/anonymous outcomes/closing receipts must receive actual remote byte/SHA/Git blob/mode/tree verification and archive checks before genuine FINAL. Original cost32 and later decisions remain distinct. Cancelled formal baselines remain standby; no manuscript was written.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    FIG.mkdir(parents=True, exist_ok=True)
    figures = []
    colors = {'alpha0': '#73808b', 'alpha1': '#c36a4f', 'LN_only': '#147d92'}
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    for ax, (name, s) in zip(axs, result['statistics'].items()):
        for j, arm in enumerate(ARMS[1:]):
            for i, order in enumerate(['pooled']+ORDERS):
                stat = s['pooled'] if order == 'pooled' else s['per_order'][order]
                m = stat['metrics']['Full_minus_'+arm+'_v']
                mid = 100*m['mean']; low, high = np.array(m['ci95'])*100
                ax.errorbar(mid, i+(j-1)*.16, xerr=[[mid-low], [high-mid]], fmt='o', color=colors[arm], capsize=3,
                            label='Full - '+{'alpha0': 'reset', 'alpha1': 'full carry', 'LN_only': 'LN only'}[arm] if i == 0 else None)
        ax.axvline(0, color='#89939c', linewidth=.8)
        ax.set_yticks(range(4), ['Pooled', 'Order 3', 'Order 4', 'Order 5'])
        ax.invert_yaxis()
        ax.set_title(('HC-STVG-v2' if 'hc2' in name else 'VidSTG')+' (128 parents)')
        ax.set_xlabel('Paired final vIoU difference (pp), 95% CI')
        ax.grid(axis='x', alpha=.15)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .01))
    fig.suptitle('Four actual online regimes / three registered orders', y=.98)
    fig.tight_layout(rect=[0, .09, 1, .94])
    for extension in ['pdf', 'png']:
        path = FIG/('LN_three_order_direct_contrasts.'+extension)
        assert not path.exists()
        fig.savefig(path, dpi=220)
    plt.close(fig)
    figures.append(FIG/'LN_three_order_direct_contrasts.pdf')
    fig, axs = plt.subplots(2, 3, figsize=(11.2, 6.2), sharex=True, sharey='row')
    for rowi, (name, stage) in enumerate(stages().items()):
        for coli, order in enumerate(ORDERS):
            ax = axs[rowi, coli]
            for arm, color in [('full', '#147d92'), ('LN_only', '#c36a4f')]:
                part = [r for r in public if r['stage'] == name and r['order'] == order and r['arm'] == arm]
                for label, style in [('Before', '--'), ('After', '-')]:
                    values = [np.mean([r[label+'_v'] for r in part if begin <= r['arrival'] < begin+16])*100 for begin in range(0, 128, 16)]
                    ax.plot(np.arange(16, 129, 16), values, style, color=color, linewidth=1.6, label=('Full' if arm == 'full' else 'LN only')+' '+label)
            part = [r for r in public if r['stage'] == name and r['order'] == order and r['arm'] == 'full']
            ax.plot(np.arange(16, 129, 16), [np.mean([r['Frozen_v'] for r in part if begin <= r['arrival'] < begin+16])*100 for begin in range(0, 128, 16)], ':', color='#667580', linewidth=1, label='Frozen source')
            ax.set_title(('HC-STVG-v2' if stage['dataset'] == 'hc2' else 'VidSTG')+' / '+order.replace('order', 'Order '))
            ax.grid(alpha=.15)
            if coli == 0:
                ax.set_ylabel('Official vIoU (%)')
            if rowi == 1:
                ax.set_xlabel('Arrival (fixed 16-parent groups)')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(.5, .01), fontsize=9)
    fig.suptitle('Actual Full and LN-only before/after streams', y=.99)
    fig.tight_layout(rect=[0, .075, 1, .95])
    for extension in ['pdf', 'png']:
        path = FIG/('LN_three_order_before_after_streams.'+extension)
        assert not path.exists()
        fig.savefig(path, dpi=220)
    plt.close(fig)
    figures.append(FIG/'LN_three_order_before_after_streams.pdf')
    write(BASE/'FIGURE_EXPORT.json', dict(status='pending_actual_PDF_pixel_view',
        PDF_files={str(p.relative_to(ROOT)): dict(sha256=sha(p), bytes=p.stat().st_size) for p in figures},
        all_anonymous_rows=3072, parents=256, root_sha256=sha(BASE/'ROOT_COMPLETE_READBACK.json'),
        CPU_only=True, new_model_calls=0, new_GT_scoring_calls=0, time=time.time()))
    print('LN_COMPLETE_ANONYMOUS_REPORT_AND_FIGURES', checked['scalar_comparisons'], flush=True)


if __name__ == '__main__':
    run()
