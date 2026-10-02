"""CPU-only post-seal metrics, independent reconstruction, and public export."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import time
import collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.tastvg_accumulation_common_v1 import *


FIELDS = [f'{phase}_{metric}' for phase in ['source', 'all', 'last'] for metric in ['v', 't', 's', 'correct30', 'correct50']]
FIELDS += ['all_fixed_v', 'last_fixed_v', 'delta_all_source_v', 'delta_last_source_v',
           'delta_last_all_v', 'delta_last_all_fixed_v', 'delta_last_all_s', 'delta_last_all_t']


def summary(rows):
    from scripts.score_tastvg_best_quick_v1 import source_summary
    result = source_summary(rows, FIELDS)
    result['counts'] = dict(targets=len(rows), positive=sum(r['delta_last_all_v'] > 1e-12 for r in rows),
                            negative=sum(r['delta_last_all_v'] < -1e-12 for r in rows),
                            zero=sum(abs(r['delta_last_all_v']) <= 1e-12 for r in rows),
                            harm_gt5pp=sum(r['delta_last_all_v'] < -.05 for r in rows),
                            latest_noops=sum(not r['latest_write_updated'] for r in rows),
                            intervals_changed=sum(r['last_interval'] != r['all_interval'] for r in rows))
    if rows:
        gross = [{**r, 'gross_gain': max(r['delta_last_all_v'], 0.),
                  'gross_loss': max(-r['delta_last_all_v'], 0.)} for r in rows]
        result['gross_source_macro'] = source_summary(gross, ['gross_gain', 'gross_loss'])
    result['correctness'] = {str(threshold): dict(
        all_correct=sum(r['all_v'] > threshold for r in rows),
        last_correct=sum(r['last_v'] > threshold for r in rows),
        rescued=sum(r['all_v'] <= threshold and r['last_v'] > threshold for r in rows),
        destroyed=sum(r['all_v'] > threshold and r['last_v'] <= threshold for r in rows))
        for threshold in [.3, .5]}
    return result


def aggregate(rows, conditions):
    from scripts.score_tastvg_best_quick_v1 import source_summary
    result = {}
    for group in ['corruption', 'clean', *conditions[1:]]:
        selected = [r for r in rows if (r['condition'] != 'clean' if group == 'corruption' else r['condition'] == group)]
        result[group] = {}
        for arm in ['A', 'R']:
            arm_rows = [r for r in selected if r['arm'] == arm]
            result[group][arm] = summary(arm_rows)
            result[group][arm]['at_least_two_prior_writes'] = summary([r for r in arm_rows if r['prior_writes'] >= 2])
            result[group][arm]['latest_nonzero'] = summary([r for r in arm_rows if r['latest_write_updated']])
            result[group][arm]['latest_noop'] = summary([r for r in arm_rows if not r['latest_write_updated']])
            result[group][arm]['by_prior_write_count'] = {str(n): summary([r for r in arm_rows if r['prior_writes'] == n]) for n in range(1, 9)}
        aa = {(r['condition'], r['order'], r['arrival']): r for r in selected if r['arm'] == 'A'}
        pairs = []
        for routed in [r for r in selected if r['arm'] == 'R']:
            uniform = aa[routed['condition'], routed['order'], routed['arrival']]
            assert routed['source_id'] == uniform['source_id']
            pairs.append({**routed,
                          'delta_R_A_all_v': routed['all_v'] - uniform['all_v'],
                          'delta_R_A_last_v': routed['last_v'] - uniform['last_v'],
                          'delta_R_A_last_all': routed['delta_last_all_v'] - uniform['delta_last_all_v'],
                          'delta_R_A_cancellation': routed['cancellation_ratio'] - uniform['cancellation_ratio']
                          if routed['cancellation_ratio'] is not None and uniform['cancellation_ratio'] is not None else None})
        result[group]['paired_R_minus_A'] = source_summary(pairs, ['delta_R_A_all_v', 'delta_R_A_last_v', 'delta_R_A_last_all'])
    return result


def run():
    import torch
    torch.set_num_threads(2)
    from vg_tta.tastvg_saved_write_accumulation_v1 import delta, at_origin, geometry
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.tastvg_paper48_metrics_v1 import xyxy
    from vg_tta.tastvg_paper48_hc2_metrics_v1 import HC2DenseMetric
    from vg_tta.tastvg_paper_readouts_v1 import dense_official_metrics
    from scripts.diagnose_tastvg_pipeline_cpu_v1 import evaluator
    verify()
    cohort = read(BASE / 'COHORT.json')
    barrier = read(BASE / 'GLOBAL_PREDICTION_BARRIER.json')
    assert barrier['status'] == 'sealed' and barrier['targets'] == len(cohort['rows']) == 576
    assert not barrier['GT_read'] and sha(BASE / 'COHORT.json') == barrier['cohort_sha256']
    for relative, expected in barrier['files'].items():
        file = BASE / relative
        assert sha(file) == expected and read(file)['time'] <= barrier['time']
        receipt(file.with_suffix('.pt'))
    label_path = POOL / 'hc2/GT_LABELS_search.json'
    write(BASE / 'SCORING_LOCK.json', dict(pins={str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
          'scripts/score_tastvg_best_quick_v1.py': sha(ROOT / 'scripts/score_tastvg_best_quick_v1.py'),
          'vg_tta/tastvg_paper48_hc2_metrics_v1.py': sha(ROOT / 'vg_tta/tastvg_paper48_hc2_metrics_v1.py'),
          'vg_tta/tastvg_paper_readouts_v1.py': sha(ROOT / 'vg_tta/tastvg_paper_readouts_v1.py')},
          barrier_sha256=sha(BASE / 'GLOBAL_PREDICTION_BARRIER.json'),
          labels_sha256=sha(label_path), time=time.time()))
    write(BASE / 'GT_EXPOSURE.json', dict(time=time.time(), after_global_prediction_barrier=True,
                                       historical_exposure=True, model_GT_input=False,
                                       scope='offline frozen-delta diagnosis; no reselection or promotion'))
    gt = read(label_path)
    plan = read(OLD / 'hc2/PLAN.json')
    old_public = read(ROOT / 'results/tastvg_routed_online_token/2026-10-02/hc2/ONLINE_ROWS.json')
    historical_scalars = {(r['condition'], r['order'], r['arrival']): r for r in old_public}
    geometry_records = read(BASE / 'WRITE_GEOMETRY.json')
    checks = collections.Counter(); max_error = 0.
    all_streams = {}
    for record in geometry_records:
        arm, condition, order = record['arm'], record['condition'], record['order']
        origin, writes = None, []
        previous = None
        for arrival in range(32):
            path = OLD / 'hc2' / arm / 'online' / condition / order / f'{arrival:05}.pt'
            old = load(path)
            assert sha(path) == read(path.with_suffix('.json'))['sha256']
            if origin is None:
                origin = old['pre_state']
            else:
                assert previous == old['pre_sha']
            state = at_origin(origin, writes)
            assert state_hash(state) == state_hash(old['pre_state']) == old['pre_sha']
            assert all(torch.equal(state[key], old['pre_state'][key]) for key in origin)
            checks['exact_prefix_states'] += 1
            if arrival % 4 == 0:
                writes.append(delta(old['pre_state'], old['post_state']))
            previous = old['post_sha']
        computed = geometry(writes)
        for field, value in computed.items():
            assert record[field] == value, (arm, condition, order, field)
        checks['geometry_streams'] += 1
        all_streams[arm, condition, order] = (origin, writes, record)
    metric = HC2DenseMetric()
    def score(prediction, parent, indices):
        nonlocal max_error
        row = plan['rows'][parent]; labels = gt[str(parent)]
        truth = {int(k): v for k, v in labels['truth'].items()}; span = labels['span']
        boxes = np.maximum(xyxy(prediction['boxes'], row['input']['width'], row['input']['height']), 0)
        ids = row['frame_ids']; interval = [ids[indices[0]], ids[indices[1]]+1]
        actual = metric(boxes, ids, interval, truth, span)
        independent = dense_official_metrics(boxes, ids, interval, truth, span)
        for name in ['m_vIoU', 'm_tIoU', 'vIoU@0.3', 'vIoU@0.5']:
            error = abs(actual[name] - independent[name]); assert error < 1e-10
            max_error = max(max_error, error); checks['independent_dense_scalars'] += 1
        dense_ids = np.asarray(sorted(truth))
        interpolated = np.column_stack([np.interp(dense_ids, ids, boxes[:, k]) for k in range(4)])
        label_boxes = np.asarray([truth[int(fid)] for fid in dense_ids])
        inter = np.maximum(np.minimum(interpolated[:, 2:], label_boxes[:, 2:])-np.maximum(interpolated[:, :2], label_boxes[:, :2]), 0).prod(1)
        union = np.maximum(interpolated[:, 2:]-interpolated[:, :2], 0).prod(1)+np.maximum(label_boxes[:, 2:]-label_boxes[:, :2], 0).prod(1)-inter
        ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        ious[(dense_ids < ids[0]) | (dense_ids > ids[-1])] = 0
        error = abs(actual['sIoU_dense_GT'] - float(ious.mean())); assert error < 1e-10
        max_error = max(max_error, error); checks['independent_dense_spatial'] += 1
        small = evaluator(prediction['boxes'], row, truth, span, True)(indices)
        assert abs(small['v']-actual['m_vIoU']) < 1e-10
        return dict(v=float(actual['m_vIoU']), t=float(actual['m_tIoU']),
                    s=float(actual['sIoU_dense_GT']), correct30=float(actual['vIoU@0.3']),
                    correct50=float(actual['vIoU@0.5']))
    rows = []
    for target in cohort['rows']:
        arm, condition, order, arrival = [target[k] for k in ['arm', 'condition', 'order', 'arrival']]
        file = BASE / 'predictions' / arm / condition / order / f'{arrival:05}.pt'
        receipt(file); saved = load(file)
        assert saved['target'] == target and saved['GT_read'] is False
        assert saved['parameter_updates'] == saved['new_experts'] == saved['new_backbone_forwards'] == 0
        origin, writes, geometry_record = all_streams[arm, condition, order]
        count = target['prior_writes']
        all_state = at_origin(origin, writes[:count]); last_state = at_origin(origin, [writes[count-1]])
        for phase, state in [('all', all_state), ('last', last_state)]:
            assert state_hash(state) == target[phase+'_sha'] == state_hash(saved[phase+'_state'])
            assert all(torch.equal(state[key], saved[phase+'_state'][key]) for key in origin)
            checks['counterfactual_state_binding'] += 1
        old = load(ROOT / target['old_payload'])
        cr = read(ROOT / target['capture_receipt'])
        capture = load(POOL / 'hc2' / cr['cache'])
        for phase, original in [('all', old['slow']), ('source', capture['prediction'])]:
            prediction = saved['predictions'][phase]
            assert torch.equal(prediction['boxes'], original['boxes']) and prediction['indices'] == original['indices']
            checks['bitwise_historical_predictions'] += 1
        assert old['final_indices'] == saved['predictions']['all']['indices']
        assert cr['pixel_sha256'] == target['pixel_sha256'] and cr['sha256'] == target['capture_sha256']
        checks['capture_bindings'] += 1
        row = {k: target[k] for k in ['arm', 'condition', 'order', 'arrival', 'parent', 'source_id',
                                     'prior_writes', 'latest_write_arrival', 'latest_write_updated', 'lag']}
        for phase, prediction in saved['predictions'].items():
            row.update({phase+'_'+name: value for name, value in score(prediction, target['parent'], prediction['indices']).items()})
            row[phase+'_interval'] = prediction['indices']
        fixed = saved['source_interval']
        row['all_fixed_v'] = score(saved['predictions']['all'], target['parent'], fixed)['v']
        row['last_fixed_v'] = score(saved['predictions']['last'], target['parent'], fixed)['v']
        row.update(delta_all_source_v=row['all_v']-row['source_v'],
                   delta_last_source_v=row['last_v']-row['source_v'],
                   delta_last_all_v=row['last_v']-row['all_v'],
                   delta_last_all_fixed_v=row['last_fixed_v']-row['all_fixed_v'],
                   delta_last_all_s=row['last_s']-row['all_s'],
                   delta_last_all_t=row['last_t']-row['all_t'],
                   cancellation_ratio=geometry_record['prefix_cancellation'][count-1],
                   cumulative_norm=geometry_record['prefix_norms'][count-1],
                   write_norm_sum=geometry_record['norm_sum_prefix'][count-1],
                   latest_write_norm=geometry_record['write_norms'][count-1],
                   latest_previous_cosine=geometry_record['consecutive_cosines'][count-2] if count >= 2 else None)
        if target['all_sha'] == target['last_sha']:
            assert torch.equal(saved['predictions']['all']['boxes'], saved['predictions']['last']['boxes'])
            assert abs(row['delta_last_all_v']) < 1e-15; checks['identical_state_outputs'] += 1
        if not target['latest_write_updated']:
            assert target['last_sha'] == target['source_sha']
            assert row['last_v'] == row['source_v']; checks['latest_noop_outputs'] += 1
        prior = historical_scalars[condition, order, arrival]
        assert prior['parent'] == target['parent'] and prior['source_id'] == row['source_id']
        for field, old_field in [('all_v', arm+'_m_vIoU'), ('source_v', 'Frozen_m_vIoU'),
                                 ('all_t', arm+'_m_tIoU'), ('all_s', arm+'_sIoU_dense_GT')]:
            error = abs(row[field]-prior[old_field]); assert error < 1e-10
            max_error = max(max_error, error); checks['historical_public_scalar_parity'] += 1
        rows.append(row)
    assert len(rows) == 576
    assert not torch.cuda.is_initialized()
    resources = read(BASE / 'RESOURCES.json')
    assert resources['bitwise_all_parity'] == resources['bitwise_source_parity'] == 576
    assert resources['suffix_replays'] == sum(load((BASE / path).with_suffix('.pt'))['suffix_replays'] for path in barrier['files'])
    assert sha(ROOT / 'methods/CURRENT_METHOD.json') == cohort['current_method_sha256']
    audit = dict(status='pass', checks=dict(checks), targets=576, max_metric_error=max_error,
                 prediction_barrier_sha256=sha(BASE / 'GLOBAL_PREDICTION_BARRIER.json'),
                 CPU_only=True, GT_after_global_seal=True, CURRENT_METHOD_unchanged=True, time=time.time())
    write(PUBLIC / 'ROWS.json', rows)
    write(PUBLIC / 'SUMMARY.json', aggregate(rows, cohort['conditions']))
    write(PUBLIC / 'WRITE_GEOMETRY.json', geometry_records)
    write(PUBLIC / 'RESOURCES.json', resources)
    write(PUBLIC / 'ROOT_READBACK.json', audit)
    write(PUBLIC / 'CONFIG.json', {k: value for k, value in cohort.items() if k not in ['rows', 'orders']})
    cases = {arm: {group: {'positive': sorted([r for r in rows if r['arm'] == arm and
              (r['condition'] != 'clean') == (group == 'corruption')], key=lambda r: -r['delta_last_all_v'])[:8],
             'negative': sorted([r for r in rows if r['arm'] == arm and
              (r['condition'] != 'clean') == (group == 'corruption')], key=lambda r: r['delta_last_all_v'])[:8]}
              for group in ['corruption', 'clean']} for arm in ['A', 'R']}
    write(PUBLIC / 'CASES.json', cases)
    write(BASE / 'ROOT_READBACK.json', audit)
    status(BASE / 'STATUS.json', dict(status='scored_pending_root_report_publication', done=576,
                                    total=576, time=time.time()))
    print('SCORED', audit, flush=True)


if __name__ == '__main__':
    run()
