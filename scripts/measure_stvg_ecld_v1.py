"""Independent saved arithmetic readback, then separate post-seal official scores."""
import math
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_ecld_common_v1 import *


def iou(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    alo, ahi = a[..., :2]-a[..., 2:]/2, a[..., :2]+a[..., 2:]/2
    blo, bhi = b[..., :2]-b[..., 2:]/2, b[..., :2]+b[..., 2:]/2
    intersection = np.maximum(0., np.minimum(ahi, bhi)-np.maximum(alo, blo)).prod(-1)
    union = np.maximum(a[..., 2:], 0).prod(-1)+np.maximum(b[..., 2:], 0).prod(-1)-intersection
    return intersection/np.maximum(union, 1e-12)


def root_readback():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    import numpy as np
    from scripts import stvg_target_teacher_student_v1 as core
    verify_runtime()
    sys.addaudithook(guard)
    bar = read(BASE / 'STUDENT_GLOBAL_BARRIER.json')
    assert bar['status'] == 'complete' and bar['count'] == 32 and not bar['GT_read']
    expected_paths = {str((BASE/'student'/(r['probe_alias']+'.pt')).relative_to(ROOT)) for r in roster()}
    assert len(bar['files']) == 32 and {r['path'] for r in bar['files']} == expected_paths
    refs = {r['case_id']: r for r in read(BASE / 'REFERENCE_LOCK.json')['records']}
    cases = {r['probe_alias']: r for r in roster()}
    max_error = 0.
    source_pins = []
    seen_cases = set()
    checks = fit_calls = accepted_steps = 0
    totals = {v: dict(cases_with_reference=0, fit_invocations=0,
        accepted_steps=0, no_update_cases=0) for v in ['A', 'B', 'C', 'D']}
    for item in bar['files']:
        path = ROOT / item['path']
        assert sha(path) == item['sha256']
        d = torch.load(path, map_location='cpu', weights_only=False)
        cid = d['case_id']
        assert cid not in seen_cases and path == BASE/'student'/(cid+'.pt')
        seen_cases.add(cid)
        case, source_path = load_case(cases[cid])
        source_pins.extend(dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in
                           [source_path, source_path.with_suffix('.receipt.json')])
        assert sha(source_path) == d['source_file_sha256']
        assert d['source_case_sha256'] == case['source_case_sha256']
        assert not d['GT_read'] and d['source_restored']
        assert np.array_equal(d['arms']['F']['boxes'], d['arms']['T']['boxes'])
        interval = d['arms']['T']['interval']
        for v in ['A', 'B', 'C', 'D']:
            aa = refs[cid][v]['anchors']
            assert d['reference_records'][v]['anchors'] == aa
            assert np.array_equal(d['arms']['S_'+v]['boxes'], d['arms']['TS_'+v]['boxes'])
            assert d['arms']['TS_'+v]['interval'] == interval
            totals[v]['cases_with_reference'] += bool(aa)
            totals[v]['fit_invocations'] += d['receipts'][v]['actual_fit_count']
            fitted = d['fits'].get(v)
            if fitted is None:
                assert not aa and np.array_equal(d['arms']['S_'+v]['boxes'], d['arms']['F']['boxes'])
                totals[v]['no_update_cases'] += 1
                continue
            assert fitted['active_parameters'] == 1792
            assert fitted['config'] == read(BASE/'DESIGN_LOCK.json')['spatial_configs'][d['dataset']]
            assert torch.equal(d['readouts'][v]['boxes'].cpu(), fitted['final_cached_decoder_boxes'].cpu())
            totals[v]['accepted_steps'] += fitted['optimizer_steps']
            totals[v]['no_update_cases'] += fitted['optimizer_steps'] == 0
            fit_calls += d['receipts'][v]['actual_fit_count']
            accepted_steps += fitted['optimizer_steps'] * d['receipts'][v]['actual_fit_count']
            if fitted['optimizer_steps'] == 0:
                assert core._tree_equal(fitted['initial_state'], fitted['final_state'])
                assert np.array_equal(d['arms']['S_'+v]['boxes'], d['arms']['F']['boxes'])
            positions = [a['position'] for a in aa]
            evidence = np.array([a['box_cxcywh'] for a in aa])
            weights = np.array([a['weight'] for a in aa])
            for row in fitted['path']:
                if 'rollout' not in row:
                    assert row.get('skipped')
                    continue
                policy = row['rollout']
                samples = policy['samples'].numpy().astype(np.float64)
                sampled_boxes = 1/(1+np.exp(-samples))
                computed = iou(sampled_boxes, evidence[:, None, :])
                stored = policy['rewards'].numpy()
                error = float(abs(computed-stored).max())
                max_error = max(max_error, error)
                assert error < 5e-6
                checks += int(computed.size)
                before = iou(row['boxes'].numpy()[positions], evidence)
                if row.get('proposed', False):
                    after = iou(row['proposed_boxes'].numpy()[positions], evidence)
                    assert float((weights*(after-before)).sum()/weights.sum()) >= -2e-6 or not row['accepted']
                if row.get('proposed') and not row['accepted']:
                    assert core._tree_equal(row['optimizer_before'], row['optimizer_after'])
                    restored = row.get('update_state', row.get('state_after'))
                    assert core._tree_equal(row['state'], restored)
                if v in ['C', 'D']:
                    # The implementation's audit describes per-anchor admission;
                    # constant rewards are independently checked above rather
                    # than treated as task-level evidence of correct identity.
                    assert policy['samples'].shape[1] == 32
                    source_boxes = case['zero']['boxes'].cpu().numpy()[positions]
                    novelty = 1-iou(source_boxes, evidence)
                    assert abs(novelty-row['novelty'].numpy()).max() < 5e-6
                    gates = np.asarray([a['evidence_gate'] for a in aa], dtype=bool)
                    assert np.array_equal(gates, row['evidence_gate'].numpy())
                    supported = (stored.var(1) > 1e-12) & ((stored > 1e-6).sum(1) >= 1)
                    assert np.array_equal(supported, row['reward_supported'].numpy())
                    ext = weights*row['novelty'].numpy()*gates*(row['novelty'].numpy()>1e-6)
                    assert abs(ext-row['external_weight'].numpy()).max() < 5e-6
                    eff = ext*supported
                    assert abs(eff-row['effective_external_weight'].numpy()).max() < 5e-6
                    if eff.sum() > 0:
                        assert abs(eff/eff.sum()-row['normalized_outer_weight'].numpy()).max() < 5e-6
                    else:
                        assert row.get('update') is None
                    if row.get('update') is None:
                        assert core._tree_equal(row['optimizer_before'], row['optimizer_after'])
            boxes = fitted['final_cached_decoder_boxes'].numpy()
            xyxy = np.concatenate([boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2], axis=-1)
            assert np.allclose(xyxy, d['arms']['S_'+v]['boxes'], atol=1e-7, rtol=0)
    assert seen_cases == set(cases)
    readback = dict(status='pass', case_count=32, fit_invocations=fit_calls,
        accepted_optimizer_steps=accepted_steps, independent_reward_scalar_checks=checks,
        max_reward_iou_error=max_error, per_variant=totals, GT_read=False, CPU_only=True,
        per_variant_step_scope='arm-level outcomes include exact reuse; aggregate accepted_optimizer_steps counts actual invocations only',
        scope='independent sampled rewards, reference acceptance, saved state/Adam rollback and actual full-vs-cached decoder; no independent backward replay')
    write(BASE / 'ROOT_STUDENT_READBACK.json', readback)
    pins = list(bar['files']) + source_pins
    for p in ['DESIGN_LOCK.json', 'RUNTIME.json', 'ROSTER_LOCK.json', 'REFERENCE_LOCK.json',
              'TEACHER_GLOBAL_BARRIER.json', 'STUDENT_GLOBAL_BARRIER.json', 'inputs_teacher/INPUT_BARRIER_V2.json',
              'inputs_teacher/TEACHER_JOB_LOCK_V2.json', 'ROOT_STUDENT_READBACK.json']:
        pins.append(dict(path=str((BASE/p).relative_to(ROOT)), sha256=sha(BASE/p)))
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json', dict(status='complete', count=32,
        files=pins, scorer_sha256=sha(__file__), GT_read=False, time=time.time()))
    print('root', readback, flush=True)


def score():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from scripts import review_stvg_sparse_r1_v2 as scorer
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    verify_runtime()
    seal = read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    assert seal['status'] == 'complete' and seal['count'] == 32 and not seal['GT_read']
    assert seal['scorer_sha256'] == sha(__file__)
    for p in seal['files']:
        assert sha(ROOT/p['path']) == p['sha256']
    assert read(BASE/'ROOT_STUDENT_READBACK.json')['status'] == 'pass'
    bar = read(BASE/'STUDENT_GLOBAL_BARRIER.json')
    expected_paths = {str((BASE/'student'/(r['probe_alias']+'.pt')).relative_to(ROOT)) for r in roster()}
    assert bar['count'] == 32 and len(bar['files']) == 32
    assert {r['path'] for r in bar['files']} == expected_paths
    assert expected_paths <= {r['path'] for r in seal['files']}
    assert str((BASE/'STUDENT_GLOBAL_BARRIER.json').relative_to(ROOT)) in {r['path'] for r in seal['files']}
    cases, outputs = {}, {}
    for rec in roster():
        case, source_path = load_case(rec)
        cid = rec['probe_alias']
        cases[cid] = case
        outputs[cid] = torch.load(BASE/'student'/(cid+'.pt'), map_location='cpu', weights_only=False)
        assert outputs[cid]['case_id'] == cid and not outputs[cid]['GT_read']
        assert case['source_case_sha256'] == outputs[cid]['source_case_sha256']
        assert sha(source_path) == outputs[cid]['source_file_sha256']
        assert str(source_path.relative_to(ROOT)) in {r['path'] for r in seal['files']}
    write(BASE/'GT_EXPOSURE.json', dict(status='after_all_predictions_sealed',
        seal_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'), GT_read=True, time=time.time()))
    truth, provenance = scorer._truths_after_seal(cases)
    official, independent, _ = scorer._metric_functions()
    rows = []
    for cid, d in outputs.items():
        support = {a['position'] for v in ['A', 'B', 'C', 'D'] for a in d['reference_records'][v]['anchors']}
        for arm, pred in d['arms'].items():
            values, audit = metric._score_student_arm(pred, cases[cid], truth[cid], support, official, independent)
            rows.append(dict(case_id=cid, dataset=d['dataset'], arm=arm, metrics=values,
                             support=audit['support'], metric_audit=audit['metric_audit']))
    summary, paired = [], []
    for ds in ['hc2', 'vidstg']:
        rr = [r for r in rows if r['dataset'] == ds]
        ids = sorted({r['case_id'] for r in rr})
        assert len(ids) == 16
        lookup = {(r['case_id'], r['arm']): r for r in rr}
        names = sorted({r['arm'] for r in rr})
        for arm in names:
            for key in rr[0]['metrics']:
                values = [lookup[c, arm]['metrics'][key] for c in ids
                          if lookup[c, arm]['metrics'][key] is not None]
                summary.append(dict(dataset=ds, arm=arm, metric=key, count=len(values),
                    mean=math.fsum(values)/len(values) if values else None))
        comparisons = [(a, 'F') for a in names if a != 'F']
        comparisons += [('TS_'+v, 'T') for v in ['A', 'B', 'C', 'D']]
        comparisons += [(a, b) for a, b in [('TS_B','TS_A'), ('TS_C','TS_A'),
                                           ('TS_D','TS_A'), ('TS_D','TS_B'), ('TS_D','TS_C')]]
        for left, right in comparisons:
            delta = {c: (lookup[c,left]['metrics']['m_vIoU']-lookup[c,right]['metrics']['m_vIoU'])*100 for c in ids}
            vv = list(delta.values())
            paired.append(dict(dataset=ds, left=left, right=right, metric='m_vIoU',
                mean_delta_pp=math.fsum(vv)/16, positive_sources=sum(v>1e-10 for v in vv),
                negative_sources=sum(v< -1e-10 for v in vv), neutral_sources=sum(abs(v)<=1e-10 for v in vv),
                worst_delta_pp=min(vv), best_delta_pp=max(vv), per_source_delta_pp=delta))
    write(BASE/'POSTSEAL_ROWS.json', dict(status='complete', rows=rows, GT_read=True))
    write(BASE/'POSTSEAL_SUMMARY.json', dict(status='complete', summary=summary, GT_read=True,
        development_selection=True, independent_confirmation=False))
    write(BASE/'PARENT_PAIRED_DELTAS.json', dict(status='complete', records=paired,
        GT_read=True, bootstrap_executed=False))
    write(BASE/'STATUS.json', dict(status='measurement_complete_pending_review_publication',
        case_count=32, GT_read=True, full_experiment_started=False, time=time.time()), mutable=True)
    print('postseal vIoU', [r for r in summary if r['metric']=='m_vIoU'], flush=True)


if __name__ == '__main__':
    {'root': root_readback, 'score': score}[sys.argv[1]]()
