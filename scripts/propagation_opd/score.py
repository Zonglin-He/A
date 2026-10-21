"""Independent label-gated CPU scoring of the sealed six-row HC64 comparison."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.state_opd.evaluate import (read, sha, write_json, within, score_boxes,
    mean_table, paired_stats, cost_summary)

NEW_NAMES = ['FixedPropagation', 'OPD', 'FixedKD', 'Direct']
NAMES = ['ERefStrong', 'GeometryOPD', *NEW_NAMES]


def _validate_work_and_fallback(traces, bank, predictions):
    """Independent checks of recorded work, all performed before labels."""
    assert bank['GT_read'] is False
    active, valid = np.asarray(bank['active'], dtype=bool), np.asarray(bank['valid'], dtype=bool)
    E = np.asarray(bank['E'])
    assert valid[:, 0].all() and not valid[~active, 1:].any()
    assert np.array_equal(np.asarray(bank['boxes'])[:, 0], E)
    assert set(traces) == set(NEW_NAMES)
    for arm in NEW_NAMES:
        output = traces[arm]; cost = output['costs']
        count = 0 if arm == 'FixedPropagation' else 3
        assert cost['updates'] == cost['backward'] == cost['state_only_backwards'] == count
        assert all(cost[name] == 0 for name in ('heavy_forwards', 'decoder_backwards',
                                              'decoder_calls', 'encoder_calls', 'expert_calls'))
        assert np.array_equal(predictions[arm], np.asarray(output['final']))
        assert np.array_equal(predictions[arm][~active], E[~active])
        if arm == 'FixedPropagation':
            assert cost['total_sequence_rollouts'] == 1
            paths = [output['trace']]
        else:
            assert len(output['rounds']) == len(output['trace']['training_paths']) == 3
            assert all(len(group) == 2 for group in output['trace']['training_paths'])
            assert all(np.asarray(row['gradient']).shape == (6,) and np.isfinite(np.asarray(row['gradient'])).all()
                       for row in output['rounds'])
            fixed = output['trace']['fixed_teacher_paths']
            assert len(fixed) == cost['teacher_setup_rollouts'] == (2 if arm == 'FixedKD' else 0)
            assert cost['student_training_rollouts'] == (0 if arm == 'FixedKD' else 6)
            assert cost['total_sequence_rollouts'] == (3 if arm == 'FixedKD' else 7)
            paths = [p for group in output['trace']['training_paths'] for p in group]
            paths += fixed + [output['trace']['final_rollout']]
        for path in paths:
            assert np.array_equal(np.asarray(path['boxes'])[~active], E[~active])
            assert not np.asarray(path['actions'])[~active].any()
            assert np.array_equal(np.asarray(path['memory_before'])[~active], np.asarray(path['memory_after'])[~active])


def _bank_statistics(bank):
    active = np.asarray(bank['active'], dtype=bool)
    observed = np.asarray(bank['observed'], dtype=bool)
    valid = np.asarray(bank['valid'], dtype=bool)
    confidence = np.asarray(bank['confidence'], dtype=float)
    unobs = ~observed
    return dict(nodes=len(active), observed_nodes=int(observed.sum()), unobserved_nodes=int(unobs.sum()),
        active_nodes=int(active.sum()), active_unobserved_nodes=int((active & unobs).sum()),
        active_unobserved_fraction=float(active[unobs].mean()) if unobs.any() else None,
        unobserved_with_nonE_candidate=int((unobs & valid[:, 1:].any(-1)).sum()),
        valid_slot_counts=valid.sum(0).tolist(),
        mean_unobserved_confidence=float(confidence[unobs].mean()) if unobs.any() else None)


def _shared_costs(receipts):
    stats = {}
    for field in ('evidence_seconds', 'bank_seconds', 'total_seconds'):
        values = np.asarray([row[field] for row in receipts.values()], dtype=float)
        assert np.isfinite(values).all() and (values >= 0).all()
        stats[field] = dict(valid_queries=len(values), mean=float(values.mean()), median=float(np.median(values)),
                            p90=float(np.quantile(values, .9)), maximum=float(values.max()), total=float(values.sum()))
    return dict(scope='Once per-query cached-evidence/bank construction and total observed CPU query wall time; '
                      'not full Source/expert inference latency', statistics=stats)


def _bank_summary(rows):
    total = lambda name: sum(row[name] for row in rows.values())
    unobserved = total('unobserved_nodes')
    fractions = [row['active_unobserved_fraction'] for row in rows.values() if row['active_unobserved_fraction'] is not None]
    return dict(nodes=total('nodes'), observed_nodes=total('observed_nodes'), unobserved_nodes=unobserved,
        active_nodes=total('active_nodes'), active_unobserved_nodes=total('active_unobserved_nodes'),
        micro_active_unobserved_fraction=total('active_unobserved_nodes') / unobserved if unobserved else None,
        macro_active_unobserved_fraction=float(np.mean(fractions)) if fractions else None,
        unobserved_with_nonE_candidate=total('unobserved_with_nonE_candidate'),
        valid_slot_counts=np.asarray([row['valid_slot_counts'] for row in rows.values()]).sum(0).tolist(),
        scope='Label-free bank statistics over all fixed physical nodes, excluding observed nodes in unobserved denominators')


def validate_before_labels(trial):
    """Validate every artifact/source/input before opening GT or old metrics."""
    trial = Path(trial).resolve()
    assert trial == (ROOT / 'outputs/propagation_opd_hc64').resolve()
    seal, lock = read(trial / 'sealed.json'), read(trial / 'lock.json')
    assert seal['queries'] == 64 and seal['dataset'] == 'hc2' and seal['GT_read'] is False
    assert seal['CPU_only'] is True and seal['fingerprint'] == lock['fingerprint']
    assert sha(trial / 'lock.json') == seal['lock_sha256']
    for path, digest in {**lock['frozen_sources'], **lock['sources']}.items():
        assert sha(ROOT / path) == digest, path
    assert sha(ROOT / 'outputs/formal/state.json') == lock['formal_state_sha256']
    assert sha(ROOT / 'configs/default.json') == lock['registered_config_sha256']
    assert sha(ROOT / 'outputs/state_opd_hc128/lock.json') == lock['parent_lock_sha256']
    manifest = ROOT / 'data/evaluation/development.jsonl'
    assert sha(manifest) == lock['manifest_sha256'] == seal['manifest_sha256']
    records = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    records = [r for r in records if r['dataset'] == 'hc2']
    assert len(records) == 128
    records = records[:64]
    assert lock['ids'] == [r['id'] for r in records]
    assert set(seal['files']) == {r['id'] for r in records}
    assert [x['id'] for x in lock['inputs']] == lock['ids']
    old = ROOT / 'outputs/layout_reward_dev'
    old_seal_path = old / 'scope_hc2/sealed.json'
    native_seal_path = old / 'research/native_masks/sealed.json'
    assert sha(old_seal_path) == lock['reference_seal_sha256']
    assert sha(native_seal_path) == lock['native_mask_seal_sha256']
    old_seal = read(old_seal_path)
    receipts, bank_statistics = {}, {}
    for record, item in zip(records, lock['inputs']):
        key = record['id']
        entry = seal['files'][key]
        path = within(trial, entry['path'])
        receipt_path = path.with_suffix('.json')
        receipt = read(receipt_path)
        assert receipt['fingerprint'] == lock['fingerprint'] and receipt['GT_read'] is False
        assert sha(path) == entry['sha256'] == receipt['sha256']
        assert sha(receipt_path) == entry['receipt_sha256']
        for stem in ('trace', 'bank', 'evidence'):
            assert sha(within(trial, receipt[stem + '_path'])) == receipt[stem + '_sha256']
        assert sha(old / 'student_cache' / f'{key}.pt') == item['cache_sha256'] == receipt['cache_sha256']
        assert sha(old / 'research/native_masks' / item['mask_path']) == item['native_mask_sha256'] == receipt['native_mask_sha256']
        reference = within(old, old_seal['files'][key]['path'])
        assert sha(reference) == item['reference_sha256'] == receipt['reference_sha256'] == old_seal['files'][key]['sha256']
        assert all(receipt[field] == 0 for field in ('model_calls', 'decoder_calls', 'expert_calls', 'RGB_reads'))
        assert receipt['CPU_only'] is True and set(receipt['costs']) == set(NEW_NAMES)
        with np.load(path, allow_pickle=False) as current, np.load(reference, allow_pickle=False) as prior:
            for field in ('frame_ids', 'positions8', 'Strong_interval'):
                assert np.array_equal(current[field], prior[field])
            assert current['frame_ids'].tolist() == record['metadata']['frame_ids']
            for name, field in [('SourceStrong', 'Source'), ('ERefStrong', 'E_Ref'), ('GeometryOPD', 'GeometryOPD')]:
                assert np.array_equal(current[name], prior[field])
            for name in NAMES:
                assert current[name].shape == prior['E_Ref'].shape
                assert np.isfinite(current[name]).all() and (current[name][:, 2:] > 0).all()
            predictions = {name: current[name].copy() for name in NEW_NAMES}
            expected_E = current['ERefStrong'].copy()
        bank = torch.load(within(trial, receipt['bank_path']), map_location='cpu', weights_only=False)
        boxes, valid = np.asarray(bank['boxes']), np.asarray(bank['valid'])
        assert boxes.shape == (len(record['metadata']['frame_ids']), 4, 4) and valid.shape == boxes.shape[:2]
        assert valid.dtype.kind == 'b' and valid[:, 0].all()
        assert np.isfinite(boxes[valid]).all() and (boxes[valid, 2:] > 0).all()
        assert np.array_equal(boxes[:, 0], expected_E)
        traces = torch.load(within(trial, receipt['trace_path']), map_location='cpu', weights_only=False)
        _validate_work_and_fallback(traces, bank, predictions)
        assert all(traces[arm]['costs'] == receipt['costs'][arm] for arm in NEW_NAMES)
        bank_statistics[key] = _bank_statistics(bank)
        receipts[key] = receipt
    assert not torch.cuda.is_initialized()
    proof = dict(fingerprint=seal['fingerprint'], seal_sha256=sha(trial / 'sealed.json'),
                 lock_sha256=sha(trial / 'lock.json'), manifest_sha256=sha(manifest),
                 all64_validated_before_labels=True, original_formal_task_unchanged=True,
                 source_E_geometry_baselines_exact=True, GT_used_for_online_selection=False,
                 model_calls=0, decoder_calls=0, expert_calls=0, RGB_reads=0, CUDA_initialized=False)
    return trial, seal, lock, records, old, old_seal, receipts, bank_statistics, proof


def candidate_coverage(bank, record, truth, positions, interval, span):
    """Offline GT-selected nodewise diagnostic, never an executable method."""
    from stvg.metrics import box_iou
    from stvg.metrics_base import xyxy
    boxes = np.asarray(bank['boxes'], dtype=float)
    valid = np.asarray(bank['valid'], dtype=bool)
    E = boxes[:, 0].copy()
    ids = record['metadata']['frame_ids']
    observed = {ids[int(p)] for p in positions}
    metadata = record['metadata']
    pixels = np.maximum(xyxy(boxes, metadata['width'], metadata['height']), 0)
    selected, gains, base_values, best_values = E.copy(), [], [], []
    choices = []
    for node, fid in enumerate(ids):
        if fid not in truth or fid in observed:
            continue
        candidates = np.flatnonzero(valid[node])
        scores = np.asarray([box_iou(pixels[node, k], truth[fid]) for k in candidates])
        best = int(candidates[int(scores.argmax())])
        baseline = float(box_iou(pixels[node, 0], truth[fid]))
        maximum = float(scores.max())
        selected[node] = boxes[node, best]
        base_values.append(baseline); best_values.append(maximum); gains.append(maximum - baseline)
        choices.append(dict(node=node, slot=best))
    values, error, _ = score_boxes(selected, record, truth, interval, span, positions)
    count = len(gains)
    summary = dict(valid_unobserved_GT_nodes=count,
        mean_E_IoU=float(np.mean(base_values)) if count else None,
        mean_candidate_best_IoU=float(np.mean(best_values)) if count else None,
        mean_available_improvement_pp=float(100 * np.mean(gains)) if count else None,
        nodes_with_candidate_over5pp_better=int(np.sum(np.asarray(gains) > .05)),
        fraction_nodes_with_candidate_over5pp_better=float(np.mean(np.asarray(gains) > .05)) if count else None,
        has_candidate_over5pp_better=bool(count and np.max(gains) > .05),
        GT_nodewise_candidate_readout_vIoU=values['v'],
        GT_nodewise_candidate_readout_values=values,
        online_method=False, GT_selected=True)
    return summary, error, choices


def _coverage_summary(rows):
    values = [r['candidate_coverage'] for r in rows]
    count = sum(x['valid_unobserved_GT_nodes'] for x in values)
    better = sum(x['nodes_with_candidate_over5pp_better'] for x in values)
    finite = lambda field: [x[field] for x in values if x[field] is not None]
    def average(field):
        selected = finite(field)
        return float(np.mean(selected)) if selected else None
    return dict(scope='Fixed sampled unobserved nodes with GT, excluding all eight observation positions',
        valid_unobserved_GT_nodes=count, nodes_with_candidate_over5pp_better=better,
        micro_fraction_nodes_over5pp_better=better / count if count else None,
        macro_fraction_nodes_over5pp_better=average('fraction_nodes_with_candidate_over5pp_better'),
        queries_with_candidate_over5pp_better=sum(x['has_candidate_over5pp_better'] for x in values),
        macro_mean_available_improvement_pp=average('mean_available_improvement_pp'),
        macro_GT_nodewise_candidate_readout_vIoU=average('GT_nodewise_candidate_readout_vIoU'),
        warning='GT selects a candidate independently at scored unobserved nodes; all other nodes retain E. '
                'This is an offline oracle diagnostic, not a method or online selection. Its dense interpolated '
                'vIoU is not the exact global maximum over all candidate trajectories.',
        GT_used_for_online_selection=False)


def _signed_rows(rows):
    out = []
    for index, row in enumerate(rows, 1):
        for arm in NEW_NAMES:
            values = row['values']
            entry = dict(query=f'H{index:03d}', method=arm)
            for baseline, suffix in [('ERefStrong', 'ERef'), ('GeometryOPD', 'Geometry')]:
                v = 100 * (values[arm]['v'] - values[baseline]['v'])
                a, b = values[arm]['unobserved_actual_sIoU'], values[baseline]['unobserved_actual_sIoU']
                s = None if a is None or b is None else 100 * (a - b)
                entry.update({f'vIoU_delta_vs_{suffix}_pp': v, f'unobserved_sIoU_delta_vs_{suffix}_pp': s,
                              f'vIoU_decline_vs_{suffix}': v < -1e-8,
                              f'vIoU_decline_over5pp_vs_{suffix}': v < -5})
            out.append(entry)
    return out


def render_report(public):
    lines = ['# HC64 cached propagation and sequence matching', '',
        'VidSTG → HC2: the same first 64 development parents, one query each, seed 41. '
        'All configurations are frozen before scoring. Every method uses the identical saved Strong interval; '
        'the network, RGB decoder and expert are not executed in this CPU experiment.', '',
        '| Method | Full vIoU % | tIoU % | Observed sIoU % | Unobserved sIoU % |',
        '|---|---:|---:|---:|---:|']
    for name in public['arm_order']:
        mean = public['means'][name]
        fields = ['m_vIoU', 'm_tIoU', 'm_sIoU_observed_actual', 'm_sIoU_unobserved_actual']
        fmt = lambda v: 'NA' if v is None else f'{100*v:.4f}'
        lines.append('| ' + name + ' | ' + ' | '.join(fmt(mean[k]) for k in fields) + ' |')
    lines += ['', 'Observed and unobserved sIoU use the fixed actual sampled nodes with GT, independently '
        'of time truncation. All eight expert observations are excluded from unobserved sIoU.', '',
        '| Comparison | vIoU delta pp [95% CI] | Improved / declined / unchanged | Drops >5 pp | Unobserved sIoU delta pp [95% CI] |',
        '|---|---:|---:|---:|---:|']
    comparisons = [(name, 'ERefStrong') for name in NEW_NAMES] + [('OPD', name) for name in ['GeometryOPD', 'FixedPropagation', 'FixedKD', 'Direct']]
    def interval(z):
        return 'NA' if z['mean_pp'] is None else '%+.4f [%.4f, %.4f]' % (z['mean_pp'], *z['CI95_pp'])
    for a, b in comparisons:
        p = public['paired'][a + '-' + b]; v = p['v']; s = p['unobserved_actual_sIoU']
        lines.append('| %s − %s | %s | %d / %d / %d | %d | %s |' %
                     (a, b, interval(v), v['improved'], v['declined'], v['unchanged'], v['declined_over5pp'], interval(s)))
    c = public['candidate_coverage']
    def coverage_number(field, scale=1.):
        value = c[field]
        return 'NA' if value is None else f'{scale * value:.4f}'
    lines += ['', 'All positive, negative and unchanged per-query results are retained in `hc64.json`. '
        '`signed_changes.tsv` contains every signed change for all four new methods against both ERef and Geometry, '
        'including every declining and severe-decline sample. Confidence intervals use 10,000 paired parent draws.', '',
        '## Offline candidate coverage', '', c['warning'], '',
        '| Diagnostic | Value |', '|---|---:|',
        f'| Fixed unobserved GT nodes | {c["valid_unobserved_GT_nodes"]} |',
        f'| Nodes with a candidate >5 pp better than E | {c["nodes_with_candidate_over5pp_better"]} |',
        f'| Query count with at least one such node | {c["queries_with_candidate_over5pp_better"]} / 64 |',
        f'| Micro node proportion (%) | {coverage_number("micro_fraction_nodes_over5pp_better", 100)} |',
        f'| Macro available node IoU gain (pp) | {coverage_number("macro_mean_available_improvement_pp")} |',
        f'| GT nodewise candidate readout full vIoU (%) | {coverage_number("macro_GT_nodewise_candidate_readout_vIoU", 100)} |', '',
        '## Recorded CPU costs', '', public['costs']['scope'], '',
        '| Method | Seconds/query | Updates/query |', '|---|---:|---:|']
    for arm in NEW_NAMES:
        stats = public['costs']['arms'][arm]
        def value(name):
            return 'not recorded' if name not in stats else f'{stats[name]["mean"]:.6f}'
        lines.append('| ' + arm + ' | ' + value('seconds') + ' | ' + value('updates') + ' |')
    lines += ['', 'Evidence construction and candidate-bank construction are recorded separately per query. '
        'These are incremental CPU costs with pre-existing Source/ReferDINO caches, not full deployed inference latency. '
        'FixedPropagation performs no parameter update; the geometric baseline is reused from the exact saved run. '
        'The registered method and the paused full VidSTG evaluation are unchanged.']
    lines += ['', '| Shared CPU cost | Seconds/query | Total seconds |', '|---|---:|---:|']
    for field, stats in public['shared_costs']['statistics'].items():
        lines.append(f'| {field} | {stats["mean"]:.6f} | {stats["total"]:.6f} |')
    bank = public['bank_statistics']
    lines += ['', 'Label-free bank support: active unobserved nodes '
        f'{bank["active_unobserved_nodes"]}/{bank["unobserved_nodes"]}; '
        f'valid slot counts {bank["valid_slot_counts"]}; '
        f'unobserved nodes with a non-E candidate {bank["unobserved_with_nonE_candidate"]}.']
    return '\n'.join(lines) + '\n'


def evaluate(trial, output):
    trial, seal, lock, records, old, old_seal, receipts, bank_statistics, proof = validate_before_labels(trial)
    # No GT or old scored metrics are read until every new artifact passes above.
    labels_path = ROOT / 'data/evaluation/development_labels.jsonl'
    labels_sha = sha(labels_path)
    assert labels_sha == read(old / 'scope_hc2/metrics.json')['labels_sha256']
    selected_ids = {r['id'] for r in records}
    labels = {r['id']: r for r in [json.loads(s) for s in labels_path.read_text().splitlines() if s.strip()]
              if r['dataset'] == 'hc2' and r['id'] in selected_ids}
    assert set(labels) == selected_ids
    rows, choices = [], {}
    maximum_error = 0.; checks = 0
    for record in records:
        key = record['id']; label = labels[key]
        truth = {int(k): v for k, v in label['boxes'].items()}
        with np.load(within(trial, seal['files'][key]['path']), allow_pickle=False) as prediction:
            interval, positions = prediction['Strong_interval'].tolist(), prediction['positions8'].tolist()
            values = {}
            for name in NAMES:
                values[name], error, support = score_boxes(prediction[name], record, truth, interval, label['interval'], positions)
                maximum_error = max(maximum_error, error); checks += 1
            bank = torch.load(within(trial, receipts[key]['bank_path']), map_location='cpu', weights_only=False)
            coverage, error, choices[key] = candidate_coverage(bank, record, truth, positions, interval, label['interval'])
            maximum_error = max(maximum_error, error); checks += 1
        rows.append(dict(id=key, parent=label['parent'], values=values, support=support, candidate_coverage=coverage))
    assert len(rows) == len({r['parent'] for r in rows}) == 64
    assert maximum_error < 2e-10 and sha(labels_path) == labels_sha
    assert sha(trial / 'sealed.json') == proof['seal_sha256'] and sha(trial / 'lock.json') == proof['lock_sha256']
    for key, entry in seal['files'].items():
        assert sha(within(trial, entry['path'])) == entry['sha256']
        assert sha(within(trial, entry['path']).with_suffix('.json')) == entry['receipt_sha256']
        for stem in ('trace', 'bank', 'evidence'):
            assert sha(within(trial, receipts[key][stem + '_path'])) == receipts[key][stem + '_sha256']
    assert not torch.cuda.is_initialized()
    means, counts = mean_table(rows, NAMES)
    proof.update(labels_sha256=labels_sha, official_dense_checks=checks,
                 official_dense_max_absolute_error=maximum_error, hashes_rechecked_after_scoring=True)
    signed = _signed_rows(rows)
    public = dict(status='complete_HC64_exposed_development', direction='VidSTG_to_HC2', queries=64, parents=64,
        selection='first64_original_manifest_order_one_query_per_parent', seed=41,
        arm_order=NAMES, new_arms=NEW_NAMES, means=means, valid_query_counts=counts,
        paired=paired_stats(rows, NAMES), costs=cost_summary(receipts, NEW_NAMES),
        shared_costs=_shared_costs(receipts), bank_statistics=_bank_summary(bank_statistics),
        candidate_coverage=_coverage_summary(rows), proof=proof, config=lock['config'],
        queries_anonymous=[dict(query=f'H{i+1:03d}', values=row['values'], candidate_coverage=row['candidate_coverage'],
                                bank_statistics=bank_statistics[row['id']],
                                shared_costs={field: receipts[row['id']][field] for field in
                                              ('evidence_seconds', 'bank_seconds', 'total_seconds')})
                           for i, row in enumerate(rows)], signed_changes=signed)
    write_json(trial / 'metrics.json', dict(public, rows=rows, oracle_selected_slots=choices))
    output = Path(output).resolve()
    assert output == (ROOT / 'results/propagation_opd_hc64').resolve()
    if (output / 'hc64.json').exists():
        previous = read(output / 'hc64.json')
        assert previous['proof']['fingerprint'] == proof['fingerprint'] and previous['proof']['seal_sha256'] == proof['seal_sha256']
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'hc64.json', public)
    (output / 'REPORT.md').write_text(render_report(public))
    fields = list(signed[0])
    lines = ['\t'.join(fields)]
    for row in signed:
        lines.append('\t'.join('NA' if row[k] is None else str(row[k]) for k in fields))
    (output / 'signed_changes.tsv').write_text('\n'.join(lines) + '\n')
    write_json(trial / 'score_receipt.json', dict(status='scored', queries=64, proof=proof,
        metrics_sha256=sha(trial / 'metrics.json'), public_results_sha256=sha(output / 'hc64.json'),
        public_report_sha256=sha(output / 'REPORT.md'), signed_changes_sha256=sha(output / 'signed_changes.tsv')))
    return public


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, default=ROOT / 'outputs/propagation_opd_hc64')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/propagation_opd_hc64')
    parser.add_argument('--run-after-seal', action='store_true')
    args = parser.parse_args()
    if not args.run_after_seal:
        parser.error('Scoring requires explicit --run-after-seal and all 64 sealed artifacts')
    result = evaluate(args.trial, args.output)
    print(json.dumps(dict(status=result['status'], means=result['means'], candidate_coverage=result['candidate_coverage'])))
