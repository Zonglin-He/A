"""Post-global-phase CPU audit and official scoring of every locked logical cell."""
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
from scripts.score_stvg_opd_p1_v1 import truths, source_stats


def phase_barrier(phase):
    verify_later()
    global_barrier = read(BASE / f'{phase}_PREDICTION_BARRIER.json')
    assert global_barrier['status'] == 'sealed' and global_barrier['all_deployment_arms_and_directions']
    assert set(global_barrier['barriers']) == {
        f'stages/{n}/PREDICTION_BARRIER.json' for n in phases()[phase]}
    checked = set()
    for rel, h in global_barrier['barriers'].items():
        path = BASE / rel; assert sha(path) == h
        barrier = read(path); assert barrier['status'] == 'sealed' and not barrier['GT_read']
        stage = stage_definition(barrier['stage'])
        expected = sum(map(len, stage['orders'].values())) * len(stage['conditions']) * len(stage['arms'])
        assert len(barrier['logical_records']) == barrier['adapted_arrivals'] == expected
        assert barrier['orders'] == stage['orders'] and barrier['variant_configs'] == stage['variant_configs']
        for group in ['files', 'inputs']:
            for f, digest_value in barrier[group].items():
                if (f, digest_value) in checked:
                    continue
                actual = BASE / f; receipt = read(actual.with_suffix('.json'))
                assert sha(actual) == digest_value == receipt['sha256'] and not receipt['GT_read']
                if 'bytes' in receipt:
                    assert receipt['bytes'] == actual.stat().st_size
                else:
                    assert group == 'inputs' and set(receipt) == {
                        'sha256', 'runtime_lock_sha256', 'GT_read', 'time'}
                assert receipt['time'] <= barrier['time'] <= global_barrier['time']
                checked.add((f, digest_value))
    return global_barrier, len(checked)


def run(phase):
    import torch
    torch.set_num_threads(2)
    barrier, byte_files = phase_barrier(phase)
    runtime = read(BASE / 'LATER_CPU_RUNTIME_LOCK.json')
    for f, h in runtime['pins'].items():
        assert sha(ROOT / f) == h, f
    completion = BASE / f'{phase}_CPU_SCORING_COMPLETION.json'
    if completion.exists():
        return
    exposure = BASE / f'{phase}_GT_EXPOSURE.json'
    if not exposure.exists():
        write(exposure, dict(status='all_deployable_predictions_precede_phase_labels',
            phase_barrier_sha256=sha(BASE / f'{phase}_PREDICTION_BARRIER.json'), time=time.time()))
    from vg_tta.tastvg_oracle_event5_v1 import official, DenseTube, box_iou
    from vg_tta.tastvg_paper48_metrics_v1 import xyxy
    from vg_tta.decota_fixed_full_audit_v1 import top1_support
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from vg_tta.stvg_opd_paper_component_audit_v1 import audit
    counts = dict(logical_arrivals=0, independent_fit_math=0, independent_dense=0,
                  exact_state_chains=0, matched_inputs=0, byte_files=byte_files)
    outputs = {}; started = time.time()
    for name in phases()[phase]:
        stage = stage_definition(name); ds = stage['dataset']; destination = PUB / name
        stage_barrier = read(BASE / 'stages' / name / 'PREDICTION_BARRIER.json')
        if (BASE / 'stages' / name / 'CPU_COMPLETION.json').exists():
            previous_completion = read(BASE / 'stages' / name / 'CPU_COMPLETION.json')
            assert sha(destination / 'ROWS.jsonl.gz') == previous_completion['rows_sha256']
            outputs[name] = dict(rows_sha256=sha(destination / 'ROWS.jsonl.gz'),
                summary_sha256=sha(destination / 'SUMMARY.json'),
                CPU_completion_sha256=sha(BASE / 'stages' / name / 'CPU_COMPLETION.json'))
            for key, value in previous_completion['counts'].items():
                counts[key] += value
            continue
        destination.mkdir(parents=True, exist_ok=True)
        dense, spans, provenance = truths(ds, stage)
        plan = read(PAPER / ds / 'PLAN.json')
        source_ids = {s: i for i, s in enumerate(sorted({r['source'] for r in plan['rows']}))}
        rows = []; stage_counts = {k: 0 for k in counts if k != 'byte_files'}
        for condition in stage['conditions']:
            for order, sequence in stage['orders'].items():
                previous = {a: None for a in stage['arms']}; hashes = {a: None for a in stage['arms']}
                for at, query in enumerate(sequence):
                    row = plan['rows'][query]; gt = dense[query]; span = spans[query]
                    matched_input = None
                    for arm in stage['arms']:
                        path, binding = record_path(stage_barrier, condition, order, arm, at)
                        z = load(path); fit = z['fit']; cfg = stage['variant_configs'][arm]
                        assert z['query_ordinal'] == query and z['source'] == stage['source'] and z['dataset'] == ds
                        assert z['order'] == order and z['arrival'] == at and z['condition'] == condition and z['arm'] == arm
                        assert z['config'] == fit['config'] == cfg and fit['selected_step'] == cfg['steps']
                        assert z['previous_payload_sha256'] == hashes[arm] and not z['GT_read']
                        assert fit['active_parameters'] == {'query_only': 256, 'LN_only': 1536}.get(arm, 1792)
                        assert torch.count_nonzero(fit['initial']['spatial.query_residual']) == 0
                        if previous[arm] is not None:
                            assert all(torch.equal(v, torch.zeros_like(v) if n == 'spatial.query_residual'
                                       else previous[arm][n]) for n, v in fit['initial'].items())
                        expected = committed(fit['initial'], fit['state'], cfg['writeback'])
                        assert all(torch.equal(expected[n], z['committed'][n]) for n in expected)
                        previous[arm] = z['committed']; hashes[arm] = sha(path)
                        inp = load(BASE / z['input']['path']); expert = unpack_expert(inp['expert'])
                        assert z['input']['sha256'] == sha(BASE / z['input']['path'])
                        if matched_input is None:
                            matched_input = inp
                        else:
                            input_equivalent(matched_input, inp); stage_counts['matched_inputs'] += 1
                        result = audit(fit, expert)
                        assert result == z['math_audit']
                        stage_counts['independent_fit_math'] += 1; stage_counts['exact_state_chains'] += 1
                        values, tubes = {}, {}
                        for label, boxes in [('Frozen', inp['native_boxes']), ('Before', fit['before']), ('After', fit['final'])]:
                            values[label] = official(boxes.numpy(), row, gt, span, z['interval'], ds)
                            tubes[label] = DenseTube(boxes.numpy(), row, gt, span, clip=ds == 'hc2')
                            check = tubes[label].score(z['interval'])
                            assert all(abs(check[m] - values[label][m]) < 2e-10 for m in ['v', 't', 's'])
                            stage_counts['independent_dense'] += 1
                        assert values['Frozen']['t'] == values['Before']['t'] == values['After']['t']
                        r = dict(dataset=ds, stage=name, arm=arm, source_id=source_ids[row['source']],
                            query_ordinal=query, order=order, arrival=at, condition=condition,
                            config=cfg, active_parameters=fit['active_parameters'],
                            reused_complete_identical_stream=binding['reused_complete_identical_stream'],
                            values=values, compute=z['compute'], gradient_calls=fit['gradient_calls'], empty=fit['empty'],
                            observed_frames=len(fit['positions']), query_type=row['query_type'],
                            event_duration_fraction=len(gt) / row['input']['frame_count'])
                        for label, metrics in values.items():
                            for metric, value in metrics.items():
                                r[label + '_' + metric] = value
                            r[label + '_R30'] = float(metrics['v'] > .3)
                            r[label + '_R50'] = float(metrics['v'] > .5)
                        for metric in ['v', 's', 't']:
                            r['delta_total_' + metric] = values['After'][metric] - values['Frozen'][metric]
                            r['delta_current_' + metric] = values['After'][metric] - values['Before'][metric]
                            r['delta_inherited_' + metric] = values['Before'][metric] - values['Frozen'][metric]
                        for threshold in [.3, .5]:
                            before, after = values['Frozen']['v'] > threshold, values['After']['v'] > threshold
                            r['correct_to_wrong_' + str(threshold)] = float(before and not after)
                            r['wrong_to_correct_' + str(threshold)] = float(not before and after)
                        ids = np.array(sorted(gt)); observed = np.isin(ids, [row['frame_ids'][p] for p in fit['positions']])
                        difference = tubes['After'].iou - tubes['Before'].iou
                        r['observed_iou_delta'] = float(difference[observed].mean()) if observed.any() else None
                        r['unobserved_iou_delta'] = float(difference[~observed].mean()) if (~observed).any() else None
                        qualities = [float(box_iou(xyxy(np.array(e), row['input']['width'], row['input']['height']),
                            gt[row['frame_ids'][p]])) for p, e in top1_support(expert) if row['frame_ids'][p] in gt]
                        r['admitted_expert_GT_IoU'] = float(np.mean(qualities)) if qualities else None
                        box = np.array([gt[f] for f in sorted(gt)], float); center = (box[:, :2] + box[:, 2:]) / 2
                        r['normalized_target_motion'] = float(np.linalg.norm(np.diff(center, axis=0), axis=1).mean()
                            / np.hypot(row['input']['width'], row['input']['height'])) if len(center) > 1 else 0.
                        r['physical_corruption_spec'] = inp.get('corruption_spec')
                        if r['physical_corruption_spec'] is not None:
                            spec = r['physical_corruption_spec']
                            coverage = float(condition.rsplit('_', 1)[1])
                            assert spec['length'] == int(np.ceil(coverage / 100 * row['input']['frame_count']))
                            assert spec['actual_physical_fraction'] == spec['length'] / row['input']['frame_count']
                        rows.append(r); stage_counts['logical_arrivals'] += 1
                    if at % 10 == 0:
                        print('OPD_LATER_CPU', name, condition, order, at, len(sequence), flush=True)
        fields = [f'{label}_{m}' for label in ['Frozen', 'Before', 'After'] for m in ['v', 't', 's', 'R30', 'R50']]
        fields += [f'delta_{kind}_{m}' for kind in ['total', 'current', 'inherited'] for m in ['v', 't', 's']]
        fields += [f'{transition}_{threshold}' for transition in ['correct_to_wrong', 'wrong_to_correct'] for threshold in [.3, .5]]
        summaries = {c: {a: source_stats([r for r in rows if r['condition'] == c and r['arm'] == a], fields)
                         for a in stage['arms']} for c in stage['conditions']}
        write(destination / 'SUMMARY.json', dict(status='postseal_score_and_math_state_dense_complete',
            stage=name, dataset=ds, conditions=summaries, GT_provenance=provenance,
            phase_barrier_sha256=sha(BASE / f'{phase}_PREDICTION_BARRIER.json'),
            counts=stage_counts, all_negative_rows_retained=True, bootstrap=10000,
            historically_exposed=True, no_reselection=True, time=time.time()))
        with gzip.open(destination / 'ROWS.jsonl.gz', 'wt') as stream:
            for r in rows:
                stream.write(json.dumps(r, allow_nan=False) + '\n')
        write(BASE / 'stages' / name / 'CPU_COMPLETION.json', dict(status='postseal_complete',
            rows_sha256=sha(destination / 'ROWS.jsonl.gz'), summary_sha256=sha(destination / 'SUMMARY.json'),
            counts=stage_counts, time=time.time()))
        outputs[name] = dict(rows_sha256=sha(destination / 'ROWS.jsonl.gz'), summary_sha256=sha(destination / 'SUMMARY.json'),
            CPU_completion_sha256=sha(BASE / 'stages' / name / 'CPU_COMPLETION.json'))
        for key, value in stage_counts.items():
            counts[key] += value
    write(completion, dict(status='phase_scoring_complete_pending_independent_statistics', phase=phase,
        outputs=outputs, counts=counts, CPU_seconds=time.time() - started, time=time.time()))


if __name__ == '__main__':
    import traceback
    try:
        run(sys.argv[1])
    except BaseException:
        directory = BASE / 'later_CPU_failures' / str(time.time_ns())
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'traceback.txt').write_text(traceback.format_exc())
        status(BASE / 'LATER_CPU_STAGE.json', dict(status='failed_preserved', phase=sys.argv[1],
            evidence=str(directory), time=time.time()))
        raise
