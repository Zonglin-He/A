"""Independent matched parent statistics, all tails, costs and scientific figures.

Consumes only postseal anonymous scalar rows. No winner selection or new GT
intervention is made by this finalizer. Actual plot review/publication is a root
handoff, not automatically claimed by producing a PNG.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections
import gzip
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.stvg_opd_paper_later_common_v1 import *
from scripts.finalize_stvg_opd_table1_v1 import feature_attribution


def independent(rows, fields):
    queries = {}; query_count = collections.Counter(); parent_of = {}
    keys = set()
    for row in rows:
        key = (row['query_ordinal'], row['condition'], row['order'])
        assert key not in keys, 'Duplicate logical cell'
        keys.add(key); q = row['query_ordinal']; parent_of[q] = row['source_id']
        if q not in queries:
            queries[q] = np.zeros(len(fields))
        queries[q] += np.array([row[f] for f in fields]); query_count[q] += 1
    assert queries and len(set(query_count.values())) == 1, 'Incomplete matched query/order/condition cells'
    parents = {}; parent_count = collections.Counter(); query_values = []
    for q in sorted(queries):
        value = queries[q] / query_count[q]; p = parent_of[q]
        if p not in parents:
            parents[p] = np.zeros(len(fields))
        parents[p] += value; parent_count[p] += 1; query_values.append(value)
    sources = sorted(parents); matrix = np.stack([parents[p] / parent_count[p] for p in sources])
    rng = np.random.default_rng(20261006)
    distribution = np.concatenate([matrix[rng.integers(0, len(sources), (50, len(sources)))].sum(1)
        / len(sources) for _ in range(200)])
    ci = np.percentile(distribution, [2.5, 97.5], axis=0); query_values = np.stack(query_values)
    result = dict(queries=len(queries), sources=len(sources), cells=len(rows), metrics={f: dict(
        mean=float(matrix[:, j].sum() / len(sources)), ci95=ci[:, j].tolist(),
        query_macro=float(query_values[:, j].sum() / len(queries)),
        harm_gt5pp_sources=int((matrix[:, j] < -.05).sum()) if f.startswith(('delta_', 'Full_minus_')) else None,
        harm_gt20pp_sources=int((matrix[:, j] < -.2).sum()) if f.startswith(('delta_', 'Full_minus_')) else None)
        for j, f in enumerate(fields)})
    return result, sources, matrix


def load_rows(stage_name):
    path = PUB / stage_name / 'ROWS.jsonl.gz'
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def matched_contrast(full, control):
    match = {(r['query_ordinal'], r['condition'], r['order']): r for r in full}
    assert len(match) == len(full) == len(control)
    effects = []
    for row in control:
        reference = match[row['query_ordinal'], row['condition'], row['order']]
        assert row['source_id'] == reference['source_id'] and row['arrival'] == reference['arrival']
        for field in ['Frozen_v', 'Frozen_t', 'Frozen_s']:
            assert abs(row[field] - reference[field]) < 2e-10
        effects.append(dict(query_ordinal=row['query_ordinal'], source_id=row['source_id'],
            condition=row['condition'], order=row['order'], **{
                'Full_minus_' + f: reference['After_' + f] - row['After_' + f]
                for f in ['v', 't', 's', 'R30', 'R50']}))
    return independent(effects, ['Full_minus_' + f for f in ['v', 't', 's', 'R30', 'R50']])[0]


def cost(rows):
    component_rows = [r['compute'].get('input_capture_components') for r in rows]
    observed = [r for r in component_rows if r is not None]
    fields = ['decode_corruption_seconds', 'STVG_frozen_forward_seconds', 'recorded_expert_forward_seconds']
    return dict(logical_arrivals=len(rows), real_fit_seconds_per_arrival=float(np.mean([
        r['compute']['fit_GPU_seconds'] for r in rows])),
        shared_capture_seconds_per_arrival=float(np.mean([r['compute']['shared_capture_seconds'] for r in rows])),
        independent_CPU_math_seconds_per_arrival=float(np.mean([r['compute']['CPU_math_seconds'] for r in rows])),
        actual_backward_rounds_mean=float(np.mean([r['gradient_calls'] for r in rows])),
        actual_new_DINO_calls_in_recorded_original_execution=sum(r['compute']['new_DINO_calls'] for r in rows),
        nominal_observation_budget=sorted({r['compute']['DINO_observation_budget'] for r in rows}),
        maximum_CUDA_peak_allocated=max(r['compute']['CUDA_peak_allocated'] for r in rows),
        mean_component_seconds={f: float(np.mean([r[f] for r in observed])) if observed else None for f in fields},
        capture_component_rows_available=len(observed),
        exact_stream_reuse_rows=sum(r['reused_complete_identical_stream'] for r in rows),
        reused_stream_cost_is_original_actual_measurement_not_new_GPU_time=True,
        recorded_expert_forward_may_be_reused_not_cold_latency=True,
        shared_capture_not_added_once_per_variant=True,
        planned_steps=sorted({r['config']['steps'] for r in rows}), trainable_parameters=sorted({r['active_parameters'] for r in rows}))


def run(phase):
    verify_later()
    receipt = read(BASE / f'{phase}_CPU_SCORING_COMPLETION.json')
    assert receipt['status'] == 'phase_scoring_complete_pending_independent_statistics'
    summaries, contrasts, all_effects, costs, diagnosis, pooled_contrasts = {}, {}, {}, {}, {}, {}
    checks = 0; total_rows = 0; data = {}
    for name in phases()[phase]:
        stage = stage_definition(name); output = PUB / name
        assert sha(output / 'ROWS.jsonl.gz') == receipt['outputs'][name]['rows_sha256']
        assert sha(output / 'SUMMARY.json') == receipt['outputs'][name]['summary_sha256']
        rows = load_rows(name); data[name] = rows; total_rows += len(rows)
        assert len(rows) == sum(map(len, stage['orders'].values())) * len(stage['arms']) * len(stage['conditions'])
        for row in rows:
            assert row['Frozen_t'] == row['Before_t'] == row['After_t']; checks += 1
            assert abs(row['delta_total_v'] - row['delta_current_v'] - row['delta_inherited_v']) < 2e-10; checks += 1
            assert row['config'] == stage['variant_configs'][row['arm']]
        saved = read(output / 'SUMMARY.json')
        summaries[name] = {}; contrasts[name] = {}; all_effects[name] = {}; costs[name] = {}; diagnosis[name] = {}
        for condition in stage['conditions']:
            summaries[name][condition] = {}; contrasts[name][condition] = {}; all_effects[name][condition] = {}
            part = [r for r in rows if r['condition'] == condition]
            full = [r for r in part if r['arm'] == 'on_policy']
            for arm in stage['arms']:
                arm_rows = [r for r in part if r['arm'] == arm]
                fields = list(saved['conditions'][condition][arm]['metrics'])
                result, sources, matrix = independent(arm_rows, fields)
                old = saved['conditions'][condition][arm]
                assert result['sources'] == old['sources'] and result['cells'] == old['cells']; checks += 2
                for f, value in result['metrics'].items():
                    for key in ['mean', 'query_macro']:
                        assert abs(value[key] - old['metrics'][f][key]) < 3e-12; checks += 1
                    assert np.max(np.abs(np.array(value['ci95']) - old['metrics'][f]['ci95'])) < 3e-12; checks += 1
                    if f.startswith('delta_'):
                        for key in ['harm_gt5pp_sources', 'harm_gt20pp_sources']:
                            assert value[key] == old['metrics'][f][key]; checks += 1
                summaries[name][condition][arm] = result
                column = fields.index('delta_total_v')
                all_effects[name][condition][arm] = [dict(source_id=s, delta_total_v=float(matrix[i, column]))
                    for i, s in enumerate(sources)]
                contrasts[name][condition][arm] = matched_contrast(full, arm_rows)
            diagnosis[name][condition] = feature_attribution(full)
        for arm in stage['arms']:
            costs[name][arm] = cost([r for r in rows if r['arm'] == arm])
        pooled_contrasts[name] = {arm: matched_contrast([r for r in rows if r['arm'] == 'on_policy'],
            [r for r in rows if r['arm'] == arm]) for arm in stage['arms']}
        for row in rows:
            checks += 1
            assert set(row['values']) == {'Frozen', 'Before', 'After'}
    prefix = PUB / phase
    prefix.mkdir(parents=True, exist_ok=True)
    write(prefix / 'ROOT_STATISTICS.json', dict(status='pass', independent_checks=checks,
        anonymous_logical_rows=total_rows, by_stage=summaries, Full_minus_controls=contrasts,
        Full_minus_controls_condition_mean=pooled_contrasts,
        paired_parent_bootstrap=10000, decoder_Jacobian_independently_reimplemented=False,
        no_configuration_selection=True, all_negative_cases_retained=True, time=time.time()))
    write(prefix / 'ALL_PARENT_EFFECTS.json', all_effects)
    write(prefix / 'COST.json', costs); write(prefix / 'FAILURE_STRATA.json', diagnosis)
    lines = [f'# {phase}: fixed OPD paper stage, actual postseal results', '',
        'All locked deployment arms and both directions sealed before this phase scoring. '
        'Intervals are 10000 paired parent bootstrap resamples after averaging orders within query and queries within parent. '
        'All cohorts have historical exposure; no hyperparameters or main budget are selected here.', '',
        '| Stage / condition | Arm | parent m_vIoU (%) | ΔvIoU (pp), 95% CI | Current / inherited (pp) | Parents harm >5 / >20 pp |',
        '|---|---|---:|---|---|---|']
    for name, conditions in summaries.items():
        for condition, arms in conditions.items():
            for arm, summary in arms.items():
                metric = summary['metrics']; d = metric['delta_total_v']; ci = np.array(d['ci95']) * 100
                lines.append(f'| {name} / {condition} | {arm} | {metric["After_v"]["mean"]*100:.4f} | '
                    f'{d["mean"]*100:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | '
                    f'{metric["delta_current_v"]["mean"]*100:+.4f} / {metric["delta_inherited_v"]["mean"]*100:+.4f} | '
                    f'{d["harm_gt5pp_sources"]} / {d["harm_gt20pp_sources"]} |')
    lines += ['', 'Direct uses standard 5 L1 + 2 GIoU and identical admitted support/parameter interface/Adam and fixed last-step budget. '
        'Query-only, LN-only and alpha-zero reconstruct their own state from source. Full before/current/inherited decomposition is descriptive and does not replace these interventions. '
        'Identical Full/Shuffled/Fixed or K4 streams are reused only with the complete original input, configuration, order and saved-state chain bound by SHA256.', '',
        'GT expert-quality/duration/motion/query-type strata are descriptive offline associations, never filters or online admission decisions. '
        'Corruption coverage denotes the physical time fraction, not equal pixel strength. The main Uniform4 and target configurations remain unchanged. '
        'Cached expert cost is separated from actual new DINO calls, decoder fit and CPU audit; reused measurements do not establish cold end-to-end latency.', '',
        'Root must actually inspect the generated figures and publish actual anonymous code/config/results and negative findings before closing this phase.']
    (prefix / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    render(phase, data, summaries, all_effects, costs, prefix)
    write(BASE / f'{phase}_CPU_COMPLETION.json', dict(status='pending_actual_root_visual_and_publication',
        phase=phase, anonymous_rows=total_rows, independent_checks=checks,
        outputs={str(p.relative_to(ROOT)): sha(p) for p in prefix.iterdir() if p.is_file()},
        paper_suite_complete=False, next_phase_requires_actual_root_close=True, time=time.time()))
    status(BASE / 'LATER_CPU_STAGE.json', dict(status='pending_actual_root_visual_and_publication',
        phase=phase, time=time.time()))


def render(phase, data, summaries, effects, costs, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, ds in zip(axes, DATASETS):
        if phase in ['P2', 'P3']:
            names = [n for n in data if stage_definition(n)['dataset'] == ds]
            for si, name in enumerate(names):
                arms = stage_definition(name)['arms']
                for ai, arm in enumerate(arms):
                    rr = [r for r in data[name] if r['arm'] == arm]
                    result = independent(rr, ['delta_total_v'])[0]['metrics']['delta_total_v']
                    y = si * (len(arms) + 1) + ai; lo, hi = np.array(result['ci95']) * 100
                    ax.errorbar(result['mean'] * 100, y,
                        xerr=[[max(0, result['mean'] * 100 - lo)], [max(0, hi - result['mean'] * 100)]],
                        fmt='o', capsize=3, color='#1766a0' if arm == 'on_policy' else '#708773')
            labels = [f'{stage_definition(n)["split"]}: {a}' for n in names for a in stage_definition(n)['arms']]
            positions = [si * (len(stage_definition(n)['arms']) + 1) + ai
                for si, n in enumerate(names) for ai, _ in enumerate(stage_definition(n)['arms'])]
            ax.set_yticks(positions, labels); ax.axvline(0, color='#777', lw=1)
            ax.set_xlabel('Method − matched Frozen vIoU (pp), parent CI')
        elif phase == 'P4':
            name = 'P4_' + ds; rr = data[name]
            for arm, color in [('Frozen', '#777'), ('After', '#1766a0')]:
                x, y, low, high = [], [], [], []
                for coverage in [0, 2.5, 5, 10]:
                    part = [r for r in rr if r['condition'] == 'clean'] if coverage == 0 else [
                        r for r in rr if r['condition'] != 'clean' and float(r['condition'].rsplit('_', 1)[1]) == coverage]
                    stat = independent(part, [arm + '_v'])[0]['metrics'][arm + '_v']
                    x.append(coverage); y.append(stat['mean'] * 100); low.append(stat['ci95'][0] * 100); high.append(stat['ci95'][1] * 100)
                ax.plot(x, y, 'o-', label=arm, color=color); ax.fill_between(x, low, high, alpha=.15, color=color)
            ax.set_xlabel('Physical burst coverage (%); five-family mean after clean')
            ax.set_ylabel('Parent m_vIoU (%)'); ax.legend()
        elif phase == 'P5':
            x, y, low, high = [], [], [], []
            for k in [1, 2, 4, 8]:
                result = independent(data[f'P5_{ds}_K{k}'], ['After_v'])[0]['metrics']['After_v']
                x.append(k); y.append(result['mean'] * 100); low.append(result['ci95'][0] * 100); high.append(result['ci95'][1] * 100)
            ax.plot(x, y, 'o-', color='#1766a0'); ax.fill_between(x, low, high, alpha=.15, color='#1766a0')
            ax.set_xticks(x); ax.set_xlabel('Uniform DINO observation budget K; main remains 4')
            ax.set_ylabel('Parent m_vIoU (%)')
        ax.set_title(ds + ' ' + phase); ax.grid(alpha=.15)
    fig.savefig(prefix / 'efficacy.png', dpi=180); fig.savefig(prefix / 'efficacy.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, ds in zip(axes, DATASETS):
        for name, conditions in effects.items():
            if stage_definition(name)['dataset'] != ds:
                continue
            if phase in ['P2', 'P3'] and not name.endswith('cross_clean'):
                continue
            if phase == 'P4':
                selected = ['clean', 'frame_drop_5', 'frame_freeze_5', 'motion_blur_5', 'occlusion_5', 'exposure_5']
            else:
                selected = list(conditions)
            for condition in selected:
                for arm, values in conditions[condition].items():
                    vector = np.sort([r['delta_total_v'] * 100 for r in values])
                    label = name.replace('P5_' + ds + '_', '') if phase == 'P5' else condition if phase == 'P4' else arm
                    ax.plot(np.linspace(0, 100, len(vector)), vector, label=label)
        ax.axhline(0, color='#777'); ax.axhline(-5, color='#a35', ls='--'); ax.axhline(-20, color='#a35', ls=':')
        ax.set_title(ds + ' complete tails'); ax.set_xlabel('All parent sources ranked (%)')
        ax.set_ylabel('Method − Frozen vIoU (pp)'); ax.legend(fontsize=7); ax.grid(alpha=.15)
    fig.savefig(prefix / 'complete_negative_tails.png', dpi=180)
    fig.savefig(prefix / 'complete_negative_tails.pdf'); plt.close(fig)


if __name__ == '__main__':
    run(sys.argv[1])
