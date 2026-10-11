"""Post-global-seal complete input/math/state/dense readback and task scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import gzip
import json
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.stvg_opd_order_supplement_common_v1 import *


def checked_barrier():
    verify()
    g = read(BASE / 'GLOBAL_PREDICTION_BARRIER.json')
    assert g['status'] == 'sealed' and g['all_three_arms_conditions_orders_and_both_targets']
    assert g['new_formal_fits'] == 2304
    assert set(g['barriers']) == {str((BASE / 'stages' / n / 'PREDICTION_BARRIER.json').relative_to(ROOT)) for n in stages()}
    checked = set()
    for rel, h in g['barriers'].items():
        assert sha(ROOT / rel) == h
        b = read(ROOT / rel); assert b['status'] == 'sealed' and not b['GT_read']
        for f, digest in {**b['files'], **b['inputs']}.items():
            if (f, digest) in checked: continue
            p = ROOT / f; rc = read(p.with_suffix('.json'))
            assert sha(p) == rc['sha256'] == digest and not rc['GT_read']
            if 'bytes' in rc: assert p.stat().st_size == rc['bytes']
            assert rc['time'] <= b['time'] <= g['time']
            checked.add((f, digest))
    return len(checked)


def coverage(native, gt):
    # Canonical half-open physical annotation support, separately from legacy task scoring.
    a, b = native; g, h = min(gt), max(gt) + 1
    assert b > a and h > g
    return max(0, min(b, h) - max(a, g)) / (h - g)


def run():
    byte_files = checked_barrier(); activate_old()
    import torch
    torch.set_num_threads(2)
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from scripts.run_stvg_opd_order_supplement_v1 import validate_lifecycle
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from vg_tta.tastvg_oracle_event5_v1 import official, DenseTube, box_iou
    from vg_tta.tastvg_paper48_metrics_v1 import xyxy
    from vg_tta.decota_fixed_full_audit_v1 import top1_support
    if not (BASE / 'GT_EXPOSURE.json').exists():
        write(BASE / 'GT_EXPOSURE.json', dict(status='all_three_arms_both_targets_global_seal_precedes_new_labels',
            global_barrier_sha256=sha(BASE / 'GLOBAL_PREDICTION_BARRIER.json'), GT_read=True,
            offline_only=True, historically_exposed=True, time=time.time()))
    counts = dict(complete_fit_math=0, complete_state_chains=0, matched_inputs=0, full_state_coordinates=0,
                  fit_rounds=0, dense_scalar_checks=0, qualified_formal_pairs=0, byte_files=byte_files)
    output = {}; tick = time.time()
    for name, stage in stages().items():
        dest = PUB / name; done = BASE / 'stages' / name / 'CPU_COMPLETION.json'
        if done.exists():
            receipt = read(done); assert sha(dest / 'ROWS.jsonl.gz') == receipt['rows_sha256']
            for k, v in receipt['counts'].items(): counts[k] += v
            output[name] = dict(rows_sha256=receipt['rows_sha256'], CPU_completion_sha256=sha(done))
            continue
        dense, spans, provenance = truths(stage['dataset'], stage)
        plan = read(PAPER / stage['dataset'] / 'PLAN.json')
        source_ids = {s: i for i, s in enumerate(sorted({r['source'] for r in plan['rows']}))}
        b = read(BASE / 'stages' / name / 'PREDICTION_BARRIER.json')
        local = {k: 0 for k in counts if k != 'byte_files'}; rows = []
        for condition in stage['conditions']:
            for order, sequence in stage['orders'].items():
                previous = {a: None for a in ARMS}; prevhash = {a: None for a in ARMS}
                for at, q in enumerate(sequence):
                    row = plan['rows'][q]; gt = dense[q]; span = spans[q]
                    for arm in ARMS:
                        binding = b['logical_records'][f'{condition}/{order}/{arm}/{at:05}']
                        path = ROOT / binding['path']; z = load(path); fit = z['fit']
                        assert sha(path) == binding['sha256'] and z['runtime_sha256'] == sha(BASE / 'RUNTIME_LOCK.json')
                        assert z['config'] == stage['variant_configs'][arm] and z['query_ordinal'] == q
                        assert z['condition'] == condition and z['order'] == order and z['arrival'] == at and z['arm'] == arm
                        validate_lifecycle(z, previous[arm], prevhash[arm])
                        previous[arm] = z['committed']; prevhash[arm] = sha(path)
                        inpfile = ROOT / z['input']['path']; assert sha(inpfile) == z['input']['sha256']
                        inp = normalized(load(inpfile)); expert = unpack_expert(inp['expert'])
                        assert inp['interval'] == z['interval'] and inp['frame_ids'] == row['frame_ids']
                        oldfile = ROOT / z['matched_full_payload_path']; assert sha(oldfile) == z['matched_full_payload_sha256']
                        oldz = load(oldfile); oldinp = normalized(load(OLD / oldz['input']['path']))
                        assert oldz['query_ordinal'] == q and oldz['interval'] == z['interval']
                        for field in ['dataset', 'source', 'pixel_sha256', 'source_model_state_sha256', 'frame_ids', 'indices', 'interval']:
                            assert inp[field] == oldinp[field], field
                        assert torch.equal(inp['native_boxes'], oldinp['native_boxes'])
                        assert inp['expert'] == oldinp['expert']
                        assert audit(fit, expert) == z['math_audit']
                        local['complete_fit_math'] += 1; local['complete_state_chains'] += 1
                        local['matched_inputs'] += 1; local['full_state_coordinates'] += 1792
                        local['fit_rounds'] += len(fit['rounds'])
                        if at < 2:
                            assert read(BASE / 'first_formal' / name / condition / order / arm / f'{at:05}.json')['status'] == 'pass'
                            local['qualified_formal_pairs'] += 1
                        values = {}; tubes = {}
                        for label, boxes in [('Frozen', inp['native_boxes']), ('Before', fit['before']), ('After', fit['final'])]:
                            values[label] = official(boxes.numpy(), row, gt, span, z['interval'], stage['dataset'])
                            tubes[label] = DenseTube(boxes.numpy(), row, gt, span, clip=stage['dataset'] == 'hc2')
                            check = tubes[label].score(z['interval'])
                            for m in ['v', 't', 's']:
                                assert abs(check[m] - values[label][m]) < 2e-10
                                local['dense_scalar_checks'] += 1
                        assert len({v['t'] for v in values.values()}) == 1
                        r = dict(dataset=stage['dataset'], stage=name, arm=arm, condition=condition, order=order,
                            arrival=at, query_ordinal=q, source_id=source_ids[row['source']], config=z['config'],
                            active_parameters=1792, values=values, native_event_coverage=coverage(z['interval'], gt),
                            compute=z['compute'], empty=fit['empty'], gradient_calls=fit['gradient_calls'],
                            nominal_selected_observations=len(expert['positions4']), attempted_expert_calls=len(expert['observations']),
                            admitted_references=len(fit['positions']), query_type=row['query_type'],
                            query_reset=z['query_reset'], Adam_reset=True, observation_scope=z['input']['observation_scope'])
                        for label, val in values.items():
                            for m, v in val.items(): r[label + '_' + m] = v
                            r[label + '_R30'] = float(val['v'] > .3); r[label + '_R50'] = float(val['v'] > .5)
                        for m in ['v', 't', 's']:
                            r['source_reset_per_arm_order'] = True
                            r['delta_total_' + m] = values['After'][m] - values['Frozen'][m]
                            r['delta_current_' + m] = values['After'][m] - values['Before'][m]
                            r['delta_inherited_' + m] = values['Before'][m] - values['Frozen'][m]
                        support = top1_support(expert); quality = []
                        for p, e in support:
                            fid = row['frame_ids'][p]
                            if fid in gt:
                                quality.append(float(box_iou(xyxy(np.asarray(e), row['input']['width'], row['input']['height']), gt[fid])))
                        r['admitted_references_with_GT'] = len(quality)
                        r['admitted_expert_GT_IoU'] = float(np.mean(quality)) if quality else None
                        gtids = np.array(sorted(gt)); observed = np.isin(gtids, [row['frame_ids'][p] for p, _ in support])
                        diff = tubes['After'].iou - tubes['Before'].iou
                        r['observed_iou_delta'] = float(diff[observed].mean()) if observed.any() else None
                        r['unobserved_iou_delta'] = float(diff[~observed].mean()) if (~observed).any() else None
                        rows.append(r)
                    if at % 16 == 0: print('ORDER_SUPPLEMENT_CPU', name, condition, order, at, flush=True)
        assert len(rows) == stage['expected_new_fits']
        dest.mkdir(parents=True, exist_ok=True)
        with gzip.open(dest / 'ROWS.jsonl.gz', 'wt') as stream:
            for r in rows: stream.write(json.dumps(r, allow_nan=False) + '\n')
        write(done, dict(status='postseal_complete_math_state_input_dense', rows_sha256=sha(dest / 'ROWS.jsonl.gz'),
            GT_provenance=provenance, counts=local, all_negative_rows_retained=True, time=time.time()))
        output[name] = dict(rows_sha256=sha(dest / 'ROWS.jsonl.gz'), CPU_completion_sha256=sha(done))
        for k, v in local.items(): counts[k] += v
    assert counts['complete_fit_math'] == 2304 and counts['qualified_formal_pairs'] == 36
    write(BASE / 'CPU_SCORING_COMPLETION.json', dict(status='all_postseal_math_state_input_dense_scoring_complete',
        outputs=output, counts=counts, CPU_seconds=time.time()-tick, time=time.time()))


if __name__ == '__main__':
    try: run()
    except BaseException:
        d = BASE / 'CPU_failures' / str(time.time_ns()); d.mkdir(parents=True, exist_ok=True)
        (d / 'traceback.txt').write_text(traceback.format_exc()); raise
