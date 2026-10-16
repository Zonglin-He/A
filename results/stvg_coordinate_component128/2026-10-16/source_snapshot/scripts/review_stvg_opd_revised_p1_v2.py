"""Finite root postseal readback of all P1 sources, fits and dense metrics.

No model or optimizer runs. Original receipts choose their original saved
audit revision; no historical receipts, predictions or scientific pins change.
Completion hands real report/case view and publication back to root.
"""
import collections
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate, BASE, PUB, read, write, sha, verify

DEST = BASE / 'P1_actual_root_readback'
FIELDS = ['After_v', 'After_t', 'After_s', 'After_R30', 'After_R50',
          'delta_total_v', 'delta_current_v', 'delta_inherited_v']


def rows(p):
    with gzip.open(p, 'rt') as f:
        return [json.loads(line) for line in f]


def progress(stage, **metadata):
    from scripts.decota_matrix_common_v1 import status
    status(DEST / 'STATUS.json', dict(status='running', scope='postseal CPU root readback',
        stage=stage, worker_pid=os.getpid(), GT_after_global_seal=True,
        no_model_calls=True, paper_suite_complete=False, time=time.time(), **metadata))


def matrix_summary(rr, fields=FIELDS):
    import numpy as np
    byquery = collections.defaultdict(list)
    for r in rr:
        byquery[r['query_ordinal']].append(r)
    byparent = collections.defaultdict(list)
    qvalues = []
    for q in sorted(byquery):
        group = byquery[q]
        assert len(group) == 3 and len({r['order'] for r in group}) == 3
        assert len({r['source_id'] for r in group}) == 1
        vals = [math.fsum(r[f] for r in group) / 3 for f in fields]
        byparent[group[0]['source_id']].append(vals)
        qvalues.append(vals)
    parents = sorted(byparent)
    matrix = np.array([[math.fsum(v[j] for v in byparent[p]) / len(byparent[p])
                        for j in range(len(fields))] for p in parents])
    queries = np.array(qvalues)
    rng = np.random.default_rng(20261008)
    bootstrap = np.concatenate([matrix[rng.integers(0, len(parents), (50, len(parents)))].mean(1)
                                for _ in range(200)])
    ci = np.quantile(bootstrap, [.025, .975], axis=0)
    result = dict(queries=len(byquery), parent_sources=len(parents), logical_arrivals=len(rr),
        metrics={f: dict(source_macro=float(matrix[:, j].mean()),
            official_query_macro=float(queries[:, j].mean()), paired_parent_ci95=ci[:, j].tolist())
            for j, f in enumerate(fields)})
    return result, parents, matrix


def population(design):
    from scripts import finalize_stvg_opd_table1_v1 as producer
    import numpy as np
    expected = read(PUB / 'TABLE1_ROOT_INDEPENDENT_AUDIT.json')
    assembled = read(PUB / 'TABLE1_SUMMARY.json')
    result = {}; contrasts = {}; scored = {}; case_selection = {}; checks = 0; count = 0
    for ds in ['hc2', 'vidstg']:
        progress('all_source_statistics', dataset=ds)
        methods = producer.load_methods(ds)
        full = methods['spatial_opd']
        match = {(r['query_ordinal'], r['order']): r for r in full}
        assert len(match) == design['stages']['P1_' + ds]['adapted_arrivals']
        plan = read(ROOT / 'artifacts/decota_paper_experiments_v1' / ds / 'PLAN.json')
        ids = {p: i for i, p in enumerate(sorted({r['source'] for r in plan['rows']}))}
        dev = {ids[plan['rows'][q]['source']] for q in design['datasets'][ds]['development_query_ordinals']}
        assert len(dev) == 32
        assert dev == set(read(PUB / ('P1_' + ds) / 'SUMMARY.json')['excluded_development_anonymous_source_ids'])
        result[ds] = {}; contrasts[ds] = {}
        for method, rr in methods.items():
            count += len(rr)
            assert len(rr) == len(match)
            for r in rr:
                old = match[r['query_ordinal'], r['order']]
                assert r['source_id'] == old['source_id'] and r['arrival'] == old['arrival']
                assert all(abs(r['Frozen_' + f] - old['Frozen_' + f]) < 2e-10 for f in ['v', 't', 's'])
                assert abs(r['delta_total_v'] - r['delta_current_v'] - r['delta_inherited_v']) < 2e-10
                checks += 4
            result[ds][method] = {}
            for name, part in [('all_official_queries', rr),
                               ('excluding_32_tuning_parent_sources', [r for r in rr if r['source_id'] not in dev])]:
                new, _, _ = matrix_summary(part)
                saved = assembled['datasets'][ds][method][name]
                assert new['queries'] == saved['queries'] and new['parent_sources'] == saved['parent_sources']
                checks += 2
                for f, m in new['metrics'].items():
                    for key in ['source_macro', 'official_query_macro']:
                        assert abs(m[key] - saved['metrics'][f][key]) < 3e-12
                        checks += 1
                    assert np.max(np.abs(np.array(m['paired_parent_ci95']) - saved['metrics'][f]['paired_parent_ci95'])) < 3e-12
                    checks += 2
                result[ds][method][name] = new
            pair = [dict(query_ordinal=r['query_ordinal'], source_id=r['source_id'], order=r['order'],
                **{'OPD_minus_' + k: match[r['query_ordinal'], r['order']]['After_' + k] - r['After_' + k]
                   for k in ['v', 't', 's', 'R30', 'R50']}) for r in rr]
            contrasts[ds][method], _, _ = matrix_summary(pair, ['OPD_minus_' + k for k in ['v', 't', 's', 'R30', 'R50']])
            for f, m in contrasts[ds][method]['metrics'].items():
                old = expected['paired_OPD_vs_methods'][ds][method]['metrics'][f]
                assert abs(m['source_macro'] - old['source_macro']) < 3e-12
                assert np.max(np.abs(np.array(m['paired_parent_ci95']) - old['paired_parent_ci95'])) < 3e-12
                checks += 3
        candidates = [r for r in full if r['observed_frames'] and r['central_expert_reward_round_mean_delta'] is not None]
        selected = []; used = set()
        selectors = [('success', sorted(candidates, key=lambda r: (-r['delta_current_v'], r['query_ordinal'], r['order']))),
            ('current_harm', sorted(candidates, key=lambda r: (r['delta_current_v'], r['query_ordinal'], r['order']))),
            ('expert_reward_task_mismatch', sorted([r for r in candidates if r['delta_current_v'] < 0 and
                r['central_expert_reward_round_mean_delta'] > 0], key=lambda r: (r['delta_current_v'], r['query_ordinal'], r['order'])))]
        for name, group in selectors:
            chosen = next((r for r in group if r['query_ordinal'] not in used), None)
            if chosen:
                used.add(chosen['query_ordinal'])
                selected.append(dict(kind=name, query_ordinal=chosen['query_ordinal'], order=chosen['order'],
                    arrival=chosen['arrival'], source_id=chosen['source_id'],
                    delta_current_v=chosen['delta_current_v'], delta_inherited_v=chosen['delta_inherited_v'],
                    expert_reward_delta=chosen['central_expert_reward_round_mean_delta'],
                    selection='posthoc deterministic extreme from every source, unique query per dataset; descriptive only'))
        case_selection[ds] = selected
        scored[ds] = match
    assert count == expected['anonymous_logical_rows'] == 258576
    write(PUB / 'TABLE1_ACTUAL_ROOT_POPULATION_READBACK.json', dict(status='pass', datasets=result,
        paired_OPD_vs_methods=contrasts, anonymous_logical_rows=count, scalar_comparisons=checks,
        all_sources_read=True, excluded_32_parents_independently_rebuilt=True,
        bootstrap=10000, fixed_seed=20261008, no_parameter_or_roster_selection=True, time=time.time()))
    write(PUB / 'TABLE1_ACTUAL_ROOT_CASE_SELECTION.json', dict(status='selected_after_complete_population',
        datasets=case_selection, cases_are_illustrative_not_effect_estimates=True, time=time.time()))
    return scored


def dense_metric(boxes, row, truth, span, interval, ds):
    import numpy as np
    b = np.asarray(boxes, dtype=np.float64)
    wh = np.array([row['input']['width'], row['input']['height']], dtype=np.float64)
    pxy = np.concatenate([(b[:, :2] - b[:, 2:] / 2) * wh,
                           (b[:, :2] + b[:, 2:] / 2) * wh], axis=1)
    if ds == 'hc2': pxy = np.maximum(pxy, 0)
    anchor = row['frame_ids']; ids = sorted(truth)
    estimated = np.stack([np.interp(ids, anchor, pxy[:, j]) for j in range(4)], axis=1)
    target = np.array([truth[i] for i in ids], dtype=np.float64)
    size = np.maximum(np.minimum(estimated[:, 2:], target[:, 2:]) - np.maximum(estimated[:, :2], target[:, :2]), 0)
    area = size[:, 0] * size[:, 1]
    ps = np.maximum(estimated[:, 2:] - estimated[:, :2], 0)
    gs = np.maximum(target[:, 2:] - target[:, :2], 0)
    union = ps[:, 0] * ps[:, 1] + gs[:, 0] * gs[:, 1] - area
    ious = np.divide(area, union, out=np.zeros_like(area), where=union > 0)
    ids = np.asarray(ids)
    ious[(ids < anchor[0]) | (ids > anchor[-1])] = 0
    a, z = map(int, interval); g, h = span
    overlap = max(0, min(z, h) - max(a, g))
    support = (ids >= max(a, g)) & (ids < min(z, h))
    return dict(v=float(ious[support].sum() / max(max(z, h) - min(a, g), 1)),
                s=float(ious.mean()), t=float(overlap / (z-a+h-g-overlap))), ids, ious


def all_fits(design, scored):
    import numpy as np
    import torch
    from scripts.decota_matrix_common_v1 import load
    from scripts.stvg_opd_paper_common_v1 import committed
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from vg_tta.decota_spatial_opd_precision_audit_revision001 import audit as a1, original_audit
    from vg_tta.decota_spatial_opd_precision_audit_revision002 import audit as a2
    from vg_tta.decota_spatial_opd_chart_revision003 import audit as a3, REVISION as r3
    from vg_tta.decota_spatial_opd_reward_audit_revision005 import audit as a5, REVISION as r5
    dispatch = {None: original_audit, 'precision_audit_revision001': a1,
                'precision_audit_revision002': a2, r3: a3, r5: a5}
    barrier = read(BASE / 'P1_PREDICTION_BARRIER.json')
    counts = collections.Counter(); maximum = 0.; datasets = {}; start = time.time()
    for ds in ['hc2', 'vidstg']:
        stage = design['stages']['P1_' + ds]; cfg = design['datasets'][ds]['config']
        truth, spans, provenance = truths(ds, stage)
        plan = read(ROOT / 'artifacts/decota_paper_experiments_v1' / ds / 'PLAN.json')
        dest = BASE / 'stages' / ('P1_' + ds)
        seal = read(dest / 'PREDICTION_BARRIER.json')
        assert sha(dest / 'PREDICTION_BARRIER.json') == barrier['barriers'][str((dest/'PREDICTION_BARRIER.json').relative_to(BASE))]
        first = None; revisions = collections.Counter(); bytes_read = 0
        for order, seq in stage['orders'].items():
            previous = None; predecessor = None
            for arrival, q in enumerate(seq):
                f = dest / 'clean' / order / 'on_policy' / f'{arrival:05}.pt'
                rel = str(f.relative_to(BASE)); receipt = read(f.with_suffix('.json')); digest = sha(f)
                assert digest == seal['files'][rel] == receipt['sha256']
                assert f.stat().st_size == receipt['bytes'] and not receipt['GT_read']
                assert receipt['time'] <= seal['time'] <= barrier['time']
                z = load(f); fit = z['fit']; inpfile = BASE / z['input']['path']
                assert sha(inpfile) == seal['inputs'][z['input']['path']] == z['input']['sha256']
                inp = load(inpfile); row = plan['rows'][q]
                assert z['parent'] == q and z['arrival'] == arrival and z['order'] == order and not z['GT_read']
                assert z['query_reset'] and z['Adam_reset'] and z['Native_WHEN_fixed']
                assert z['previous_payload_sha256'] == predecessor
                assert z['config'] == fit['config'] == cfg and fit['selected_step'] == cfg['steps']
                assert torch.count_nonzero(fit['initial']['spatial.query_residual']).item() == 0
                if previous is None:
                    if first is None: first = {n: v.clone() for n, v in fit['initial'].items()}
                    assert all(torch.equal(first[n], v) for n, v in fit['initial'].items())
                else:
                    assert all(torch.equal(v, torch.zeros_like(v) if n == 'spatial.query_residual' else previous[n])
                               for n, v in fit['initial'].items())
                expected = committed(fit['initial'], fit['state'], cfg['writeback'])
                assert all(torch.equal(expected[n], z['committed'][n]) for n in expected)
                assert sum(v.numel() for v in fit['initial'].values()) == 1792
                revision = z['math_audit'].get('revision'); assert revision in dispatch, revision
                checked = dispatch[revision](fit, unpack_expert(inp['expert']))
                assert checked == z['math_audit']
                revisions[str(revision)] += 1; counts['math_dictionary_exact'] += 1
                assert z['interval'] == inp['interval']
                saved = scored[ds][q, order]; curves = {}
                for name, boxes in [('Frozen', inp['native_boxes']), ('Before', fit['before']), ('After', fit['final'])]:
                    vals, ids, iou = dense_metric(boxes.numpy(), row, truth[q], spans[q], z['interval'], ds)
                    curves[name] = iou
                    for metric, value in vals.items():
                        error = abs(value - saved[name+'_'+metric]); maximum = max(maximum, error)
                        assert error < 2e-10, (ds, order, arrival, name, metric, error)
                        counts['independent_dense_scalars'] += 1
                mask = np.isin(ids, [row['frame_ids'][p] for p in fit['positions']])
                diff = curves['After'] - curves['Before']
                for label, selected in [('observed', mask), ('unobserved', ~mask)]:
                    val = float(diff[selected].mean()) if selected.any() else None
                    old = saved[label + '_iou_delta']
                    assert (val is None and old is None) or (val is not None and old is not None and abs(val-old) < 2e-10)
                    assert int(selected.sum()) == saved[label + '_GT_frames']
                    counts['observed_unobserved_checks'] += 2
                previous = z['committed']; predecessor = digest
                bytes_read += f.stat().st_size; counts['arrivals'] += 1; counts['state_coordinates'] += 3584
                if arrival % 100 == 0:
                    progress('all_fit_math_state_dense', dataset=ds, order=order, arrival=arrival,
                        dataset_arrivals=sum(revisions.values()), total_arrivals=counts['arrivals'], bytes=bytes_read)
                    print('ROOT_P1_READBACK', ds, order, arrival, len(seq), flush=True)
        datasets[ds] = dict(arrivals=sum(revisions.values()), revisions=dict(revisions), payload_bytes=bytes_read,
                            annotation_provenance=provenance, source_reset_each_order=True)
        write(DEST / ('COMPLETE_' + ds + '.json'), dict(status='pass', scope='every saved fit, state and dense metric',
            dataset=ds, counts=dict(counts), dataset_integrity=datasets[ds], GT_after_global_seal=True, time=time.time()))
    assert counts['arrivals'] == counts['math_dictionary_exact'] == 41355
    write(PUB / 'TABLE1_ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json', dict(status='pass', datasets=datasets,
        counts=dict(counts), maximum_dense_absolute_error=maximum, original_audit_revisions_dispatch_exact=True,
        GPU_model_optimizer_calls=0, all_queries_all_orders=True, GT_after_all_41355_sealed=True,
        original_GT_parser_provenance_reused=True, independent_full_decoder_Jacobian=False,
        predictions_parameters_cohort_unchanged=True, CPU_seconds=time.time()-start, time=time.time()))


def run():
    activate(); verify()
    pin = read(DEST / 'RUNTIME.json')
    for f, h in pin['pins'].items(): assert sha(ROOT / f) == h, f
    assert read(BASE / 'P1_CPU_COMPLETION.json')['status'] == 'pending_actual_root_visual_and_publication'
    assert read(BASE / 'P1_PREDICTION_BARRIER.json')['all_deployment_OPD_directions']
    import torch
    torch.set_num_threads(2)
    start = time.time(); design = read(BASE / 'DESIGN_LOCK.json')
    scored = population(design); all_fits(design, scored)
    write(DEST / 'COMPLETION.json', dict(status='pending_actual_root_signal_cases_view_publication',
        scope='all P1 source statistics and every saved fit mathematical/state/dense readback',
        artifacts={str(p.relative_to(ROOT)): sha(p) for p in PUB.glob('TABLE1_ACTUAL_ROOT*.json')},
        actual_arrivals=41355, GPU_calls=0, paper_suite_complete=False, CPU_seconds=time.time()-start, time=time.time()))
    from scripts.decota_matrix_common_v1 import status
    status(DEST / 'STATUS.json', dict(status='pending_actual_root_signal_cases_view_publication',
        scope='CPU readback complete; actual case/report view and publication remain',
        actual_arrivals=41355, paper_suite_complete=False, time=time.time()))


if __name__ == '__main__':
    try:
        run()
    except BaseException:
        failure = DEST / 'failures' / str(time.time_ns())
        failure.mkdir(parents=True, exist_ok=False)
        (failure / 'traceback.txt').write_text(traceback.format_exc())
        from scripts.decota_matrix_common_v1 import status
        status(DEST / 'STATUS.json', dict(status='failed_preserved', scope='CPU root readback',
            evidence=str(failure.relative_to(ROOT)), experiment_predictions_changed=False, time=time.time()))
        raise
