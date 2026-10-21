"""Summarize existing sealed HC64 full-box support, without GT or inference.

Only saved banks, traces, prediction arrays, receipts, hashes and configuration
are read. No candidate/path, teacher reward, model or metric is recomputed.
Outputs are anonymous aggregate structure statistics, not task performance.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'outputs/box_propagation_hc64'
OUTPUT = ROOT / 'results/box_propagation_hc64/structure.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def summary(values):
    x = torch.as_tensor(values, dtype=torch.float64).flatten()
    if not len(x): return dict(count=0)
    assert torch.isfinite(x).all()
    return dict(count=len(x), mean=float(x.mean()), median=float(x.quantile(.5)),
                minimum=float(x.min()), p10=float(x.quantile(.1)), p90=float(x.quantile(.9)), maximum=float(x.max()))


def main():
    torch.set_num_threads(1)
    begin = time.perf_counter()
    sealpath = TRIAL / 'sealed.json'
    if not sealpath.is_file(): raise SystemExit('Complete 64-query seal required; no work was performed')
    seal, lock = read(sealpath), read(TRIAL / 'lock.json')
    assert seal['queries'] == 64 and seal['GT_read'] is False and seal['fingerprint'] == lock['fingerprint']
    assert seal['lock_sha256'] == sha(TRIAL / 'lock.json') and set(seal['files']) == set(lock['ids'])
    sealsha = sha(sealpath)
    total = Counter(); reasons = Counter(); scales = {'computed_x': [], 'computed_y': [], 'accepted_x': [], 'accepted_y': [], 'rejected_scale_x': [], 'rejected_scale_y': []}
    scale_fail = Counter(); area_ratios = []; scale_products = []
    sides_confidence = []; native_area_ratios = []
    outputs = {name: Counter() for name in ('BoxOPD', 'BoxFixedKD', 'BoxDirect', 'BoxFixed')}
    output_area = {name: [] for name in outputs}
    native_selected_area = {name: [] for name in outputs}
    per_query = []
    for ordinal, key in enumerate(lock['ids']):
        entry = seal['files'][key]
        assert sha(TRIAL / entry['path']) == entry['sha256']
        rp = TRIAL / entry['receipt_path']; assert sha(rp) == entry['receipt_sha256']
        rr = read(rp)
        assert rr['GT_read'] is False and rr['fingerprint'] == lock['fingerprint'] and rr['sha256'] == entry['sha256']
        for name in ('bank', 'trace'): assert sha(TRIAL / rr[name + '_path']) == rr[name + '_sha256']
        for name in ('model_calls', 'decoder_calls', 'expert_calls', 'RGB_reads'): assert rr[name] == 0
        bank = torch.load(TRIAL / rr['bank_path'], map_location='cpu', weights_only=False)
        trace = torch.load(TRIAL / rr['trace_path'], map_location='cpu', weights_only=False)
        assert bank['GT_read'] is False
        boxes, valid, active, observed, event = (bank[k] for k in ('boxes', 'valid', 'active', 'observed', 'event'))
        T = len(boxes); positions = bank['positions8']; native = bank['native_boxes']
        assert boxes.shape == (T, 3, 4) and valid.shape == (T, 3)
        assert torch.equal(boxes[:, 0], bank['E']) and valid[:, 0].all()
        assert torch.equal(boxes[positions, 1], native) and valid[positions, 1].all()
        assert not valid[~active & ~observed, 1:].any()
        ratios = boxes[positions, 1, 2:].prod(-1) / native[:, 2:].prod(-1)
        assert torch.equal(ratios, torch.ones_like(ratios)); native_area_ratios.extend(ratios.tolist())
        alternative_count = valid[:, 1:].sum(-1)
        unobs = event & ~observed
        action_count = torch.where(active, alternative_count, torch.zeros_like(alternative_count))
        counts = dict(all_nodes=T, observations=len(positions), native_stored_exact=int(valid[positions, 1].sum()),
            observations_outside_event=int((observed & ~event).sum()), event_observed=int((event & observed).sum()),
            event_unobserved=int(unobs.sum()), event_unobserved_active=int((unobs & active).sum()),
            event_unobserved_with_alternative=int((unobs & (action_count >= 1)).sum()),
            event_unobserved_with_two_alternatives=int((unobs & (action_count == 2)).sum()),
            event_unobserved_fallback_only_E=int((unobs & (action_count == 0)).sum()),
            all_nodes_active=int(active.sum()))
        total.update(counts)
        for t, row in enumerate(bank['fit_diagnostics']):
            for side, info in enumerate(row):
                if info is None:
                    total['no_directional_anchor_sides'] += 1
                    continue
                reason = info['reason']; reasons[reason] += 1
                if reason == 'native_observation_no_fit': continue
                total['attempted_unobserved_WLS_sides'] += 1
                computed = (float(info['mass']) > 1e-15 and info['source_ESS'] >= 3 and info['target_ESS'] >= 3
                            and all(v > 1e-6 for v in info['variance']))
                if computed:
                    sx, sy = map(float, info['scale'])
                    scales['computed_x'].append(sx); scales['computed_y'].append(sy)
                    if reason == 'nonfinite_or_out_of_range_axis_scale':
                        assert np.isfinite([sx, sy]).all()
                        scales['rejected_scale_x'].append(sx); scales['rejected_scale_y'].append(sy)
                        xbad, ybad = not .5 <= sx <= 2, not .5 <= sy <= 2
                        assert xbad or ybad
                        scale_fail['x_below_half'] += int(sx < .5); scale_fail['x_above_two'] += int(sx > 2)
                        scale_fail['y_below_half'] += int(sy < .5); scale_fail['y_above_two'] += int(sy > 2)
                        scale_fail['only_x_outside'] += int(xbad and not ybad)
                        scale_fail['only_y_outside'] += int(ybad and not xbad)
                        scale_fail['both_axes_outside'] += int(xbad and ybad)
                if info['valid']:
                    assert reason == 'accepted' and bank['side_available'][t, side]
                    sx, sy = map(float, info['scale'])
                    scales['accepted_x'].append(sx); scales['accepted_y'].append(sy)
                    parent = int(bank['source_anchors'][t, side])
                    reference_area = bank['side_reference_boxes'][t, side, 2:].prod()
                    ratio = float(reference_area / native[parent, 2:].prod())
                    assert abs(ratio - sx * sy) < 1e-11
                    area_ratios.append(ratio); scale_products.append(sx * sy)
                    sides_confidence.append(float(bank['side_reliability'][t, side]))
                    total['accepted_unobserved_WLS_sides'] += 1
        with np.load(TRIAL / entry['path'], allow_pickle=False) as data:
            assert np.array_equal(data['ERefStrong'], bank['E'].numpy())
            for name in outputs:
                out = trace[name]; final = out['final']
                assert np.array_equal(data[name], final.numpy())
                assert torch.equal(final[~active], bank['E'][~active])
                assert (((boxes == final[:, None]).all(-1)) & valid).any(-1).all()
                path = out['trace'] if name == 'BoxFixed' else out['trace']['final_rollout']
                actions = path['actions']; assert not actions[~active].any()
                assert torch.equal(final, boxes[torch.arange(T), actions])
                selected_native = observed & active & (actions == 1)
                selected_count = int(selected_native.sum())
                outputs[name].update(queries=1, all_observed_nodes=int(observed.sum()),
                    eligible_observed_nodes=int((observed & active).sum()), native_slot1_selected=selected_count,
                    event_unobserved=int(unobs.sum()), event_unobserved_replaced=int((unobs & (actions != 0)).sum()))
                native_by_node = torch.zeros(T, 4, dtype=torch.float64); native_by_node[positions] = native
                ar = final[positions, 2:].prod(-1) / native[:, 2:].prod(-1)
                output_area[name].extend(ar.tolist())
                if selected_count:
                    kept = final[selected_native, 2:].prod(-1) / native_by_node[selected_native, 2:].prod(-1)
                    assert torch.equal(kept, torch.ones_like(kept))
                    native_selected_area[name].extend(kept.tolist())
        per_query.append(dict(query='H' + str(ordinal + 1).zfill(3), **counts))
    assert total['observations'] == total['native_stored_exact'] == len(native_area_ratios) == 512
    attempted = total['attempted_unobserved_WLS_sides']
    accepted = total['accepted_unobserved_WLS_sides']
    rejected_scale = reasons['nonfinite_or_out_of_range_axis_scale']
    for relative, expected in lock['sources'].items(): assert sha(ROOT / relative) == expected
    assert sha(ROOT / 'outputs/formal/state.json') == lock['formal_state_sha256']
    assert sha(ROOT / 'configs/default.json') == lock['registered_config_sha256']
    assert sha(ROOT / 'outputs/propagation_v0_hc64/gate.json') == lock['old_gate_sha256']
    assert sha(sealpath) == sealsha and not torch.cuda.is_initialized()
    result = dict(status='complete_saved_structure_audit_no_GT', queries=64, parents=64,
        observations=dict(expected=512, native_full_boxes_stored_bit_exact=512,
            stored_native_area_over_native_area=summary(native_area_ratios),
            units='normalized continuous cxcywh; no grid-edge snapping or mask-bbox crop'),
        nodes=dict(total), WLS=dict(reason_counts=dict(reasons), attempted_unobserved_sides=attempted,
            admitted_unobserved_sides=accepted, admitted_fraction=accepted / attempted,
            scale_rejected_sides=rejected_scale, scale_rejected_fraction_of_attempts=rejected_scale / attempted,
            scale_rejected_fraction_of_rejections=rejected_scale / max(1, attempted - accepted),
            scale_boundary_counts=dict(scale_fail), scale_distributions={k: summary(v) for k, v in scales.items()},
            admitted_full_box_area_over_native_area=summary(area_ratios),
            admitted_scale_product=summary(scale_products), admitted_side_cycle=summary(sides_confidence)),
        outputs={name:dict(counts=dict(outputs[name]), all_observed_output_area_over_native_area=summary(output_area[name]),
                           selected_native_area_over_native_area=summary(native_selected_area[name])) for name in outputs},
        denominators='One observation is one fixed native anchor; WLS statistics are side-level construction counts, not independent videos. Event-unobserved counts include all fallback nodes.',
        interpretation='The bank preserves complete observed native boxes; admitted transported extent changes only by its fitted axis scales. Rejected/unreliable sides fall back without constructing a mask-bbox replacement. These structure counts do not establish localization quality or the correctness of correspondences.',
        anonymous_query_structure=per_query,
        proof=dict(fingerprint=lock['fingerprint'], seal_sha256=sealsha, lock_sha256=sha(TRIAL / 'lock.json'),
            audit_source_sha256=sha(__file__), all64_hashes_validated=True, source_formal_config_old_gate_unchanged=True,
            GT_read=False, candidate_constructions=0, new_paths=0, model_calls=0, CUDA_initialized=False),
        seconds=time.perf_counter() - begin)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True); temp = OUTPUT.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n'); os.replace(temp, OUTPUT)
    print(json.dumps(dict(status=result['status'], queries=64, native_boxes=512, nodes=result['nodes'],
                          WLS_attempts=attempted, WLS_admitted=accepted, WLS_scale_rejected=rejected_scale)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args(); main()
