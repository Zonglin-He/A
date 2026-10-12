"""Complete CPU source/state/input/math/dense root for 768+2304 sealed fits."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import time
import collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.stvg_opd_mechanism_LN_order_common_v1 import BASE, PUB, PREVIOUS, OLD, PAPER, stages, verify, activate_old, audit, scientific, QUERY
from scripts.decota_matrix_common_v1 import read, write, status, sha, load
from scripts.review_stvg_opd_supplement_v1 import rows, independent_stats, compare_stats


def dense(boxes, row, truth, span, interval, clip):
    a = np.asarray(boxes, dtype=float)
    corners = np.concatenate([a[:, :2]-a[:, 2:]/2, a[:, :2]+a[:, 2:]/2], axis=1)
    corners *= np.array([row['input']['width'], row['input']['height']]*2)
    if clip:
        corners = np.maximum(corners, 0)
    ids = np.asarray(row['frame_ids'])
    fids = np.array(sorted(truth))
    target = np.array([truth[int(f)] for f in fids])
    p = np.stack([np.interp(fids, ids, corners[:, j]) for j in range(4)], axis=1)
    intersection = np.maximum(0, np.minimum(p[:, 2:], target[:, 2:])-np.maximum(p[:, :2], target[:, :2])).prod(axis=1)
    union = np.maximum(p[:, 2:]-p[:, :2], 0).prod(axis=1)+np.maximum(target[:, 2:]-target[:, :2], 0).prod(axis=1)-intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    iou[(fids < ids[0]) | (fids > ids[-1])] = 0
    lo, hi = interval
    g, h = span
    intersection_t = max(0, min(hi, h)-max(lo, g))
    union_t = hi-lo+h-g-intersection_t
    eligible = (fids >= max(lo, g)) & (fids < min(hi, h))
    return dict(v=float(iou[eligible].sum()/max(max(hi, h)-min(lo, g), 1)),
                t=float(intersection_t/union_t), s=float(iou.mean()))


def run():
    import torch
    torch.set_num_threads(2)
    verify()
    activate_old()
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    from scripts.stvg_opd_order_supplement_common_v1 import verify as oldverify, audit as oldaudit
    from scripts.run_stvg_opd_order_supplement_v1 import validate_lifecycle as oldlife
    from scripts.run_stvg_opd_mechanism_LN_order_v1 import validate_lifecycle as newlife
    from methods.decota_final_simplified_v1.observations import uniform_positions
    from scripts.score_stvg_opd_mechanism_LN_order_v1 import checked_barrier
    oldverify()
    assert read(BASE/'CPU_count_dispatch001/RESOLUTION_RECEIPT.json')['status'] == 'pass'
    count = collections.Counter()
    count['new_byte_files'] = checked_barrier()
    global_barrier = read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    assert read(BASE/'GT_EXPOSURE.json')['time'] >= global_barrier['time']
    scoring = read(BASE/'CPU_SCORING_COMPLETION.json')
    assert scoring['counts']['complete_fit_math'] == 768 and scoring['counts']['qualified_formal_pairs'] == 12
    saved = read(PUB/'FOUR_ARM_ORDER_STATISTICS.json')
    origins = {}
    paired_inputs = set()
    for name, stage in stages().items():
        oldname = name.replace('M_', 'O_')
        newbar = read(BASE/'stages'/name/'PREDICTION_BARRIER.json')
        oldbar = read(PREVIOUS/'stages'/oldname/'PREDICTION_BARRIER.json')
        assert newbar['source_model_state_sha256'] == oldbar['source_model_state_sha256']
        assert newbar['expert_model_state_sha256'] == oldbar['expert_model_state_sha256']
        assert newbar['actual_new_DINO_calls'] == 0
        origins0 = read(OLD/'stages'/stage['matched_full_stage']/'PREDICTION_BARRIER.json')
        origin_binding = origins0['logical_records']['clean/order1/on_policy/00000']
        origin = OLD/origin_binding['path']
        assert sha(origin) == origin_binding['sha256']
        source = load(origin)['fit']['initial']
        qualification_source = load(BASE/'qualification'/name/'SOURCE_INITIAL.pt')
        assert all(torch.equal(source[k], qualification_source[k]) for k in source)
        origins[name] = dict(source_origin_sha256=sha(origin), complete_source_reset_streams=12,
                             source_model_sha256=newbar['source_model_state_sha256'], expert_model_sha256=newbar['expert_model_state_sha256'])
        newrows = rows(PUB/name/'ROWS.jsonl.gz')
        oldrows_path = ROOT/'results/stvg_opd_supplement/2026-10-11/orders'/oldname/'ROWS.jsonl.gz'
        oldrows = rows(oldrows_path)
        assert sha(PUB/name/'ROWS.jsonl.gz') == scoring['outputs'][name]['rows_sha256']
        assert sha(oldrows_path) == saved['statistics'][name]['old_rows_sha256']
        assert len(newrows) == 384 and len(oldrows) == 1152
        cells = {(r['arm'], r['order'], r['query_ordinal']): r for r in newrows+oldrows}
        assert len(cells) == 1536
        labels, spans, provenance = truths(stage['dataset'], stage)
        plan = read(PAPER/stage['dataset']/'PLAN.json')
        oldcfg = read(PREVIOUS/'DESIGN_LOCK.json')['stages'][oldname]['variant_configs']
        old_runtime = sha(PREVIOUS/'RUNTIME_LOCK.json')
        for order, seq in stage['orders'].items():
            for arm in ['full', 'alpha0', 'alpha1', 'LN_only']:
                previous = prevhash = None
                barrier = newbar if arm == 'LN_only' else oldbar
                for at, q in enumerate(seq):
                    binding = barrier['logical_records'][f'clean/{order}/{arm}/{at:05}']
                    file = ROOT/binding['path']
                    assert sha(file) == binding['sha256']
                    rc = read(file.with_suffix('.json'))
                    assert rc['sha256'] == binding['sha256'] and not rc['GT_read'] and rc['bytes'] == file.stat().st_size
                    z = load(file)
                    assert z['query_ordinal'] == q and z['arrival'] == at and z['order'] == order and z['arm'] == arm
                    fit = z['fit']
                    assert z['runtime_sha256'] == (sha(BASE/'RUNTIME_LOCK.json') if arm == 'LN_only' else old_runtime)
                    assert z['config'] == (stage['variant_configs'][arm] if arm == 'LN_only' else oldcfg[arm])
                    (newlife if arm == 'LN_only' else oldlife)(z, previous, prevhash, source if at == 0 else None)
                    assert set(fit['initial']) == set(source) and sum(v.numel() for v in fit['initial'].values()) == 1792
                    assert torch.count_nonzero(fit['initial'][QUERY]) == 0
                    if at == 0:
                        assert all(torch.equal(fit['initial'][k], source[k]) for k in source)
                        count['source_reset_streams'] += 1
                    if arm == 'LN_only':
                        assert torch.count_nonzero(fit['state'][QUERY]) == 0
                    inpfile = ROOT/z['input']['path']
                    assert sha(inpfile) == z['input']['sha256']
                    inp = normalized(load(inpfile))
                    expert = unpack_expert(inp['expert'])
                    row = plan['rows'][q]
                    assert inp['frame_ids'] == row['frame_ids'] and inp['interval'] == z['interval']
                    assert expert['positions4'] == uniform_positions(inp['frame_ids'], inp['indices'], 4)
                    assert inp['source_model_state_sha256'] == newbar['source_model_state_sha256']
                    original = ROOT/z['matched_full_payload_path']
                    assert sha(original) == z['matched_full_payload_sha256']
                    oldz = load(original)
                    oldinpfile = OLD/oldz['input']['path']
                    assert sha(oldinpfile) == oldz['input']['sha256']
                    oldinp = normalized(load(oldinpfile))
                    for k in ['dataset', 'source', 'pixel_sha256', 'source_model_state_sha256', 'frame_ids', 'indices', 'interval', 'expert']:
                        assert inp[k] == oldinp[k], k
                    assert torch.equal(inp['native_boxes'], oldinp['native_boxes'])
                    assert (audit if arm == 'LN_only' else oldaudit)(fit, expert) == z['math_audit']
                    previous, prevhash = z['committed'], sha(file)
                    count['new_fits' if arm == 'LN_only' else 'reused_fits'] += 1
                    count['fit_rounds'] += len(fit['rounds'])
                    count['full_state_coordinates'] += 1792
                    count['input_bindings'] += 1
                    count['byte_files_checked'] += 3
                    if at < 2:
                        qualified_file = (BASE if arm == 'LN_only' else PREVIOUS)/'qualification'/(name if arm == 'LN_only' else oldname)/'clean'/order/arm/file.name
                        qualified = load(qualified_file)
                        c = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
                        equal(scientific(fit), scientific(qualified['fit']), c)
                        count['new_qualified_formal_pairs' if arm == 'LN_only' else 'reused_qualified_formal_pairs'] += 1
                        count['qualified_formal_tensor_coordinates'] += c['tensor_coordinates']
                        if arm == 'LN_only':
                            ctrl = read(qualified_file.with_suffix('.control.json'))
                            repeat = ROOT/ctrl['repeat_path']
                            assert sha(repeat) == ctrl['repeat_sha256']
                            cc = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
                            equal(scientific(qualified['fit']), scientific(load(repeat)), cc)
                            assert cc == ctrl['complete_repeat_bitwise']
                            count['new_complete_repeat_pairs'] += 1
                    record = cells[arm, order, q]
                    for label, boxes in [('Frozen', inp['native_boxes']), ('Before', fit['before']), ('After', fit['final'])]:
                        values = dense(boxes.numpy(), row, labels[q], spans[q], z['interval'], stage['dataset'] == 'hc2')
                        for metric in ['v', 't', 's']:
                            assert abs(values[metric]-record[label+'_'+metric]) < 2e-10
                            count['independent_dense_scalars'] += 1
                    if arm == 'LN_only':
                        key = f'clean/{order}/full/{at:05}'
                        assert oldbar['logical_records'][key]['input']['sha256'] == binding['input']['sha256']
                        paired_inputs.add((name, order, q))
                print('LN_ROOT_FULL_STREAM', name, order, arm, flush=True)
                status(BASE/'ROOT_STATUS.json', dict(status='running_complete_root_readback',
                    stage=name, order=order, arm=arm, counts=dict(count), CPU_only=True,
                    GT_read=True, pid=os.getpid(), time=time.time()))
        paired = []
        for r in oldrows:
            if r['arm'] != 'full':
                continue
            p = dict(r)
            for arm in ['alpha0', 'alpha1', 'LN_only']:
                c = cells[arm, r['order'], r['query_ordinal']]
                assert c['source_id'] == r['source_id'] and c['arrival'] == r['arrival']
                for metric in ['v', 's', 't']:
                    p['Full_minus_'+arm+'_'+metric] = r['After_'+metric]-c['After_'+metric]
                    count['paired_endpoint_scalars'] += 1
            paired.append(p)
        s = saved['statistics'][name]
        fields = list(s['pooled']['metrics'])
        count['statistics_scalar_comparisons'] += compare_stats(s['pooled'], independent_stats(paired, fields, True))
        for order in stage['orders']:
            count['statistics_scalar_comparisons'] += compare_stats(s['per_order'][order], independent_stats([r for r in paired if r['order'] == order], fields, True))
        for arm in ['full', 'alpha0', 'alpha1', 'LN_only']:
            fields0 = list(s['arms'][arm]['metrics'])
            count['statistics_scalar_comparisons'] += compare_stats(s['arms'][arm], independent_stats([r for r in cells.values() if r['arm'] == arm], fields0, True))
    assert count['new_fits'] == 768 and count['reused_fits'] == 2304 and len(paired_inputs) == 768
    assert count['source_reset_streams'] == 24 and count['new_qualified_formal_pairs'] == count['new_complete_repeat_pairs'] == 12
    assert count['reused_qualified_formal_pairs'] == 36 and not torch.cuda.is_initialized()
    verify()
    oldverify()
    write(BASE/'ROOT_COMPLETE_READBACK.json', dict(status='pass',
        scope='complete all768 new LN-only and2304 reused actual controls/source-origin/reset-inheritance-writeback/math/input/dense/48 qualified-formal and12 full repeat pairs/independent three-order paired statistics',
        counts=dict(count), new_formal_fits=768, reused_control_fits=2304, paired_inputs=768,
        source_bindings=origins, statistics_sha256=sha(PUB/'FOUR_ARM_ORDER_STATISTICS.json'),
        original_source_and_runtime_unchanged=True, independent_full_decoder_Jacobian=False,
        actual_new_DINO_calls=0, new_GPU_calls=0, new_model_calls=0, CPU_only=True,
        GT_read='independent offline dense audit only after768global', time=time.time()))
    revision = BASE/'ROOT_scope_dispatch001'
    if revision.exists():
        write(revision/'RESOLUTION_RECEIPT.json', dict(status='pass',
            scope='original accepted P3 LN auditor for new768; original accepted P4 joint auditor for reused2304; no fit/math/scoring/science change',
            root_sha256=sha(BASE/'ROOT_COMPLETE_READBACK.json'), counts=dict(count),
            original_source_and_runtime_unchanged=True, new_GPU_calls=0, new_model_calls=0,
            new_DINO_calls=0, CPU_only=True, GT_read=True, time=time.time()))
    status(BASE/'ROOT_STATUS.json', dict(status='pass_pending_actual_PDF_view_public_archive',
        counts=dict(count), CPU_only=True, GT_read=True, pid=os.getpid(), time=time.time()))
    print('LN_COMPLETE_ROOT_PASS', dict(count), flush=True)


if __name__ == '__main__':
    run()
