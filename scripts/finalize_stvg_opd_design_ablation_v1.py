"""Fixed matched table, parent bootstrap, full tails/costs, coverage Figure 3."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections
import gzip
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.stvg_opd_design_ablation_common_v1 import *


def rows_from(path):
    with gzip.open(path, 'rt') as f: return [json.loads(x) for x in f]


def statistics(rows, fields, require_balanced=True):
    groups = collections.defaultdict(list); cells = set()
    for row in rows:
        key = (row['query_ordinal'], row['order'], row['condition'])
        assert key not in cells; cells.add(key)
        groups[row['source_id']].append(row)
    if not groups: return dict(sources=0, cells=0, metrics={f: None for f in fields})
    if require_balanced: assert len({len(x) for x in groups.values()}) == 1
    sources = sorted(groups); matrix = np.array([[np.mean([r[f] for r in groups[s]]) for f in fields] for s in sources])
    rng = np.random.default_rng(20261011)
    boots = np.concatenate([matrix[rng.integers(0, len(matrix), (100, len(matrix)))].mean(1) for _ in range(100)])
    ci = np.quantile(boots, [.025, .975], axis=0)
    return dict(sources=len(sources), cells=len(rows), metrics={f: dict(mean=float(matrix[:, j].mean()), ci95=ci[:, j].tolist(),
        query_macro=float(np.mean([r[f] for r in rows])), parent_values=[dict(source_id=s, value=float(matrix[i, j])) for i, s in enumerate(sources)],
        negative_parent_count=int((matrix[:, j] < 0).sum()), positive_parent_count=int((matrix[:, j] > 0).sum()),
        harm_gt5pp_sources=int((matrix[:, j] < -.05).sum()), harm_gt20pp_sources=int((matrix[:, j] < -.2).sum()),
        worst_parent=float(matrix[:, j].min()), best_parent=float(matrix[:, j].max())) for j, f in enumerate(fields)})


def existing_table():
    result = {}; expected = {
        'Full method': [23.502, 18.337, 33.024, 25.769],
        'w/o Action-Reward Correspondence': [19.665, 9.517, 26.704, 15.547],
        'w/o On-Policy Refresh': [22.378, 14.477, 29.667, 23.190],
        'w/o Query Residual': [23.672, 17.873, 33.015, 25.349],
        'w/o LayerNorm Adaptation': [22.320, 15.015, 29.082, 24.331],
        'w/o Slow-State Consolidation': [23.264, 18.079, 32.807, 25.289],
        'Frozen source': [22.156, 14.466, 29.093, 23.594]}
    specs = [('Full method', 'P3', 'on_policy', 'After_v'),
             ('w/o Action-Reward Correspondence', 'P2', 'shuffled_feedback', 'After_v'),
             ('w/o On-Policy Refresh', 'P2', 'frozen_rollout', 'After_v'),
             ('w/o Query Residual', 'P3', 'LN_only', 'After_v'),
             ('w/o LayerNorm Adaptation', 'P3', 'query_only', 'After_v'),
             ('w/o Slow-State Consolidation', 'P3', 'joint_alpha0', 'After_v'),
             ('Frozen source', 'P3', 'on_policy', 'Frozen_v')]
    columns = [('hc2', 'cross_clean'), ('vidstg', 'cross_clean'), ('hc2', 'same_5percent'), ('vidstg', 'same_5percent')]
    for label, phase, arm, field in specs:
        cells = []
        for ds, split in columns:
            path = OLDPUB / f'{phase}_{ds}_{split}' / 'ROWS.jsonl.gz'
            rows = [r for r in rows_from(path) if r['arm'] == arm]
            assert len({r['source_id'] for r in rows}) == 128
            cell = statistics(rows, [field]); mean = cell['metrics'][field]['mean'] * 100
            index = len(cells); assert abs(mean - expected[label][index]) < .00051, (label, ds, mean)
            cells.append(dict(mean_percent=mean, statistics=cell, rows_sha256=sha(path)))
        result[label] = cells
    return result


def run():
    verify(); done = read(BASE / 'CPU_SCORING_COMPLETION.json')
    assert done['status'] == 'all_postseal_math_state_input_dense_scoring_complete'
    table = existing_table(); allstats = {}; costs = {}; coverage_groups = {}
    column_order = ['A_hc2_cross_clean', 'A_vidstg_cross_clean', 'A_hc2_same_5percent', 'A_vidstg_same_5percent']
    labels = {'whole_video': 'w/o Event-Scoped Sampling', 'carry_query': 'w/o Fast-State Reset', 'full_LN_carry': 'w/o Slow-State Attenuation'}
    for name in column_order:
        path = PUB / name / 'ROWS.jsonl.gz'; assert sha(path) == done['outputs'][name]['rows_sha256']
        rows = rows_from(path); stage = stages()[name]; allstats[name] = {}; costs[name] = {}
        for arm in ARMS:
            part = [r for r in rows if r['arm'] == arm]
            fields = ['After_v', 'Before_v', 'Frozen_v', 'After_s', 'After_t', 'After_R30', 'After_R50',
                      'Full_minus_control_v', 'delta_total_v', 'delta_current_v', 'delta_inherited_v']
            stat = statistics(part, fields)
            for r in part:
                assert abs(r['delta_total_v'] - r['delta_current_v'] - r['delta_inherited_v']) < 2e-10
                assert r['Event_Full_t'] == r['After_t'] == r['Frozen_t']
            allstats[name][arm] = dict(pooled=stat,
                per_condition={c: statistics([r for r in part if r['condition'] == c], fields) for c in stage['conditions']},
                per_order={o: statistics([r for r in part if r['order'] == o], fields) for o in stage['orders']})
            table.setdefault(labels[arm], []).append(dict(mean_percent=stat['metrics']['After_v']['mean']*100,
                Full_minus_control_pp=stat['metrics']['Full_minus_control_v']['mean']*100,
                Full_minus_control_ci95_pp=[v*100 for v in stat['metrics']['Full_minus_control_v']['ci95']], statistics=stat,
                rows_sha256=sha(path)))
            quality = [r for r in part if r['admitted_expert_GT_IoU'] is not None]
            costs[name][arm] = dict(arrivals=len(part), nominal_selected_observations=sum(r['nominal_selected_observations'] for r in part),
                attempted_observations=sum(r['attempted_expert_calls'] for r in part), admitted_references=sum(r['admitted_references'] for r in part),
                admitted_references_with_GT=sum(r['admitted_references_with_GT'] for r in part),
                expert_GT_IoU_mean_over_available_arrivals=float(np.mean([r['admitted_expert_GT_IoU'] for r in quality])) if quality else None,
                actual_new_DINO_calls=sum(r['compute']['actual_new_DINO_calls'] for r in part) if arm == 'whole_video' else 0,
                fit_GPU_seconds_per_arrival=float(np.mean([r['compute']['fit_GPU_seconds'] for r in part])),
                CPU_math_seconds_per_arrival=float(np.mean([r['compute']['CPU_math_seconds'] for r in part])),
                shared_capture_counted_once=True, state_arms_reuse_original_expert=True)
        if name.endswith('cross_clean'):
            part = [r for r in rows if r['arm'] == 'whole_video']; edges = read(BASE / 'DESIGN_LOCK.json')['coverage_bins']
            # Native temporal scope is paired and order-independent on the frozen observed input.
            source_coverage = collections.defaultdict(list)
            for r in part: source_coverage[r['source_id']].append(r['native_event_coverage'])
            assert all(max(v)-min(v) == 0 for v in source_coverage.values())
            bins = []
            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                selected = [r for r in part if lo <= r['native_event_coverage'] and
                            (r['native_event_coverage'] <= hi if i == len(edges)-2 else r['native_event_coverage'] < hi)]
                stat = statistics(selected, ['Full_minus_control_v'])
                bins.append(dict(lo=lo, hi=hi, last_inclusive=i==len(edges)-2, n_parents=stat['sources'], n_cells=len(selected), statistics=stat))
            assert sum(b['n_parents'] for b in bins) == 128 and sum(b['n_cells'] for b in bins) == 256
            coverage_groups[name] = bins
    PUB.mkdir(parents=True, exist_ok=True)
    write(PUB / 'COMPONENT_TABLE.json', dict(status='complete_fixed_matched_components', table=table,
        columns=['Cross to HC2', 'Cross to VidSTG', 'Same-domain corrupt HC2', 'Same-domain corrupt VidSTG'],
        parent_sources_per_target=128, scientific_negative_results_retained=True, bootstrap=10000, time=time.time()))
    write(PUB / 'ALL_STATISTICS.json', allstats); write(PUB / 'COST_AND_EXPERT_QUALITY.json', costs)
    write(PUB / 'COVERAGE_ANALYSIS.json', dict(status='postseal_offline_fixed_bins', bins=coverage_groups,
        coverage_is_not_tIoU=True, causal_coverage_claim=False, bootstrap=10000, time=time.time()))
    ordered = ['Full method', 'w/o Event-Scoped Sampling', 'w/o Action-Reward Correspondence', 'w/o On-Policy Refresh',
               'w/o Query Residual', 'w/o LayerNorm Adaptation', 'w/o Slow-State Consolidation', 'w/o Fast-State Reset',
               'w/o Slow-State Attenuation', 'Frozen source']
    lines = ['# Component Ablation Study', '', 'Absolute parent-macro m_vIoU (%). Cross-domain: both fixed orders; same-domain: five 5% physical corruption families.', '',
             '| Configuration | Cross HC2 | Cross VidSTG | Corrupt HC2 | Corrupt VidSTG |', '|---|---:|---:|---:|---:|']
    tex = ['% Actual sealed results; values in percent.', '\\begin{tabular}{lrrrr}', '\\toprule',
           'Configuration & Cross HC2 & Cross VidSTG & Corrupt HC2 & Corrupt VidSTG \\\\', '\\midrule']
    for label in ordered:
        values = [f"{x['mean_percent']:.3f}" for x in table[label]]
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
        tex.append(label + ' & ' + ' & '.join(values) + ' \\\\')
    tex += ['\\bottomrule', '\\end{tabular}']
    (PUB / 'COMPONENT_TABLE.md').write_text('\n'.join(lines)+'\n'); (PUB / 'COMPONENT_TABLE.tex').write_text('\n'.join(tex)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.15), sharey=True)
    titles = ['VidSTG → HC-STVG-v2', 'HC-STVG-v2 → VidSTG']
    for ax, name, title in zip(axes, column_order[:2], titles):
        bins = coverage_groups[name]
        for i, b in enumerate(bins):
            metric = b['statistics']['metrics']['Full_minus_control_v']
            if metric is None: continue
            m = metric['mean'] * 100; ci = np.array(metric['ci95']) * 100
            ax.errorbar(i, m, yerr=np.array([[m-ci[0]], [ci[1]-m]]), fmt='o', color='#2b5775', capsize=4, lw=1.5, ms=6)
        ax.axhline(0, color='#dd8c35', ls='--', lw=1.2)
        ax.set_xticks(range(4), [f"{b['lo']:g}–{b['hi']:g}\nn={b['n_parents']}" for b in bins])
        ax.set_title(title, fontsize=12); ax.set_xlabel('Native event coverage')
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=.16)
    axes[0].set_ylabel('Event-scoped − Whole-video\nΔm_vIoU (pp)')
    fig.tight_layout(); fig.savefig(PUB / 'figure3_event_coverage.png', dpi=220); fig.savefig(PUB / 'figure3_event_coverage.pdf'); plt.close(fig)
    report = lines + ['', '## Paired Full-minus-control differences', '', '| Intervention | Setting | Difference (pp) | Paired 95% CI |', '|---|---|---:|---|']
    for arm, label in labels.items():
        for i, cell in enumerate(table[label]):
            ci = cell['Full_minus_control_ci95_pp']; report.append(f"| {label} | {column_order[i]} | {cell['Full_minus_control_pp']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    report += ['', 'All 5,376 new formal fits and all deployment conditions sealed before the new GT read. Original P2/P3 rows and complete Full streams are reused, not rerun.',
        'Coverage grouping is offline and descriptive; bins and bootstrap were fixed before new predictions. Confidence intervals are conditional on historical exposed parent sources, with no multiplicity adjustment.',
        'Cases without an annotated reference frame have missing expert quality, not a zero score. State-arm expert inference is reused; recorded fit costs are not full cold-start latency.',
        'All per-condition/order results, negative tails, parent effects, costs and expert counts remain in the machine-readable companion.',
        'Actual root plot review and anonymous remote publication are still required; this CPU completion is not final closure.']
    (PUB / 'REPORT.md').write_text('\n'.join(report)+'\n')
    write(BASE / 'CPU_COMPLETION.json', dict(status='pending_actual_root_visual_and_publication',
        new_formal_fits=5376, counts=done['counts'], all_negative_results_retained=True,
        output_files={str(p.relative_to(ROOT)):sha(p) for p in PUB.rglob('*') if p.is_file()},
        experiment_complete=False, original_P1_P6_suite_unchanged=True, time=time.time()))


if __name__ == '__main__': run()
