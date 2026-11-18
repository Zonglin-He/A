"""Replay R170 and seal comparable mapped and direct-native D assignments."""
from collections import Counter
from functools import lru_cache
import fcntl
import gc
import time

import numpy as np

from scope171 import (ARMS, HERE, INPUT, PLATFORMS, PRIOR_INPUT, PRIOR_WORK, R170, ROOT, SOURCE_SETTINGS,
                      SPLITS, WORK, atomic_json, lock, read, sha, status)
from assignment171 import assign, reference_unary
from calibrate168 import edges_of
from common160 import array_sha, npz_atomic
from crossplatform.ground_single_frame import orient_box
from geometry162 import in_box
from predict170 import old_baseline, original_evidence
from requery169 import associate, native_donors, native_forward
from replay168 import oriented
from retrieval167 import components, support_intervals, support_matrix
from select168 import solve


VIEWS = ('C', 'D')
MODES = ('baseline', 'independent', 'joint', 'independent', 'joint',
         'independent', 'fixed_reference', 'joint', 'joint')
REFERENCE_BANKS = ('', 'C_mapped', 'C_mapped', 'D_mapped', 'D_mapped',
                   'D_native', 'D_native', 'D_native_linked', 'D_native')
FILES = ('select171.py', 'assignment171.py', 'predict171.py')
HISTORICAL_NAMES = ('r158_pca', 'C_independent', 'C_joint', 'D_independent', 'D_joint')


def read_npz(p):
    with np.load(p, allow_pickle=False) as data:
        return dict(data)


@lru_cache(maxsize=1)
def old_manifest():
    r = read(R170 / 'PREDICTION_SEAL.json')
    assert not r['FINAL_access'] and not r['target_identity_labels_read']
    assert r['input_seal'] == sha(R170 / 'INPUT_SEAL.json') and r['n'] == len(r['files']) == 96
    return r


def legacy(entry, source):
    prior = old_manifest()
    name = f"{entry['case']:02d}_{source}.npz"
    path = PRIOR_WORK / 'predictions' / name
    assert sha(path) == prior['files'][name]
    old = read_npz(path)
    assert str(old['protocol']) == prior['prediction_lock'] and tuple(old['arms']) == HISTORICAL_NAMES
    return old


def query(model, entry, ev, original, arm):
    if entry['rewrite']['status'] != 'supported':
        return np.empty((0, 6), dtype=ev.boxes.dtype), np.zeros(0), 0.
    result = native_forward(model, entry['row'], ev.points, ev.indices, None, entry['rewrite'][arm])
    z = result['z']
    np.testing.assert_array_equal(z['xyz'], original['xyz'])
    np.testing.assert_array_equal(z['point_indices'], original['point_indices'])
    np.testing.assert_array_equal(z['point_indices'], ev.indices)
    ids, ranks, _ = native_donors(result)
    assert ids.size and ids[0] == result['subject_winner']
    return z['boxes'][ids], ranks, result['forward_seconds']


def camera_relation(subject_masks, reference_masks, uv, visible):
    """Apply the original factor to role-specific boxes through the same input seeds."""
    n = len(subject_masks)
    both = np.concatenate((subject_masks, reference_masks))
    low, high, confidence = support_intervals(both, visible, uv)
    a = np.arange(n)
    b = np.r_[np.arange(n, len(both)), -1]
    left, _ = components(a, b, low, high, confidence, np.zeros((len(both), len(both))))
    stats = np.stack((both.sum(1), (both & visible[None, :]).sum(1), high - low, confidence), axis=1)
    return left, stats[:n], stats[n:]


def old_control(old, role, sign, boxes, choices, field, unary):
    bank = old['bank']
    score = old['score'].copy()
    score[role, bank[role] >= 0] = field
    score[role, bank[role] == -1] = unary[-1]
    pos = np.r_[boxes, -1]
    for k, mode in enumerate(('independent', 'joint')):
        c, _ = solve(bank, score, old['left'], old['overlap'], {role: sign},
                     overlap_on=False, independent=mode == 'independent')
        s, r = choices[k]
        assert c[0] == bank[0, s] and c[role] == pos[r]


def predict_case(model, entry, settings):
    row = entry['row']
    assert sha(row['pointpath']) == row['point_sha256'] and sha(row['imagepath']) == row['image_sha256']
    ev, original, _, old = original_evidence(model, row, settings)
    expected_winner, expected_box = old_baseline(model.source)[entry['baseline_index']]
    assert (int(ev.probability.argmax()), ev.old_winner) == tuple(expected_winner)
    np.testing.assert_array_equal(ev.old_pca, expected_box)
    np.testing.assert_array_equal(ev.edges, edges_of(entry['parsed']['graph']))
    np.testing.assert_array_equal(ev.role_spans, [r['span'] for r in entry['parsed']['graph']['entities']])
    role = entry['role']
    sign = entry['sign']
    assert (old['bank'][role] == -1).sum() == 1
    subject_ids = old['bank'][0, old['bank'][0] >= 0]
    mapped_ids = old['bank'][role, old['bank'][role] >= 0]
    subject_unary = old['score'][0, old['bank'][0] >= 0]
    assert subject_ids[0] == ev.old_winner and subject_unary.argmax() == 0
    n_query = len(ev.boxes)
    whole = np.concatenate((np.c_[ev.boxes, np.zeros(n_query)], old['novel'][np.isfinite(old['novel']).all(1)]))
    keep = np.sort(np.unique(ev.seeds, axis=0, return_index=True)[1])
    seeds, visible, uv = ev.seeds[keep], ev.visible[keep], ev.uv[keep]
    inside = support_matrix(whole, seeds, n_query)
    subject_masks = inside[subject_ids]
    subject_boxes = oriented(row, old, ev.boxes, subject_ids, lambda _: ev.points)
    mapped_boxes = oriented(row, old, ev.boxes, mapped_ids, lambda _: ev.points) if len(mapped_ids) else np.empty((0, 7))
    points = np.unique(ev.points, axis=0)
    receivers = support_matrix(whole, points, n_query)[mapped_ids]
    mapped_left, subject_stats, mapped_stats = camera_relation(subject_masks, inside[mapped_ids], uv, visible)
    ref_slots = np.flatnonzero(old['bank'][role] >= -1)
    np.testing.assert_array_equal(old['bank'][role, ref_slots], np.r_[mapped_ids, -1])
    np.testing.assert_array_equal(mapped_left, old['left'][role - 1][np.ix_(np.arange(len(subject_ids)), ref_slots)])
    packet = dict(subject_boxes=subject_boxes, subject_unary=subject_unary, subject_stats=subject_stats,
                  mapped_boxes=mapped_boxes, mapped_left=mapped_left, mapped_stats=mapped_stats, seeds=seeds,
                  visible=visible, uv=uv, point_sha=array_sha(ev.points), sample_sha=array_sha(ev.indices), sign=sign)
    outputs = [np.stack((ev.old_pca, np.full(7, np.nan)))]
    choices = [(0, -1)]
    timings = {}
    for arm in VIEWS:
        raw, ranks, forward = query(model, entry, ev, original, arm)
        start = time.perf_counter()
        natural = np.stack([orient_box(ev.points, b) for b in raw]) if len(raw) else np.empty((0, 7))
        donor_mask = np.stack([in_box(points, np.r_[b, 0.]) for b in raw]) if len(raw) else np.zeros((0, len(points)), bool)
        field, links, witness = associate(donor_mask, receivers, ranks)
        mapped = reference_unary(field)
        # R169 already normalized this vector before adding its null.
        np.testing.assert_array_equal(mapped[:-1], field)
        old_choices = [assign(subject_unary, mapped, mapped_left, sign, m) for m in ('independent', 'joint')]
        old_control(old, role, sign, mapped_ids, old_choices, field, mapped)
        for s, r in old_choices:
            outputs.append(np.stack((subject_boxes[s], mapped_boxes[r] if r < len(mapped_boxes) else np.full(7, np.nan))))
        choices.extend(old_choices)
        packet.update({arm + '_natural_boxes': natural, arm + '_mapped_unary': mapped, arm + '_links': links,
                       arm + '_witness': witness, arm + '_legacy_ids': np.array(
                           [(subject_ids[s], mapped_ids[r] if r < len(mapped_ids) else -1) for s, r in old_choices])})
        timings[arm+'_forward'] = forward
        timings[arm+'_association_and_readout'] = time.perf_counter() - start
        if arm == 'D':
            started = time.perf_counter()
            native_mask = support_matrix(np.c_[raw, np.zeros(len(raw))], seeds, len(raw))
            native_left, _, native_stats = camera_relation(subject_masks, native_mask, uv, visible)
            native = reference_unary(ranks)
            linked_only = reference_unary(np.where(links >= 0, ranks, -np.inf))
            for scores, mode in ((native, 'independent'), (native, 'fixed_reference'), (linked_only, 'joint'), (native, 'joint')):
                s, r = assign(subject_unary, scores, native_left, sign, mode)
                choices.append((s, r))
                outputs.append(np.stack((subject_boxes[s], natural[r] if r < len(raw) else np.full(7, np.nan))))
            packet.update(D_native_unary=native, D_native_linked_unary=linked_only,
                          native_left=native_left, native_stats=native_stats)
            timings['D_native_geometry_and_readout'] = time.perf_counter() - started
    method_boxes = np.stack(outputs)
    assert method_boxes.shape == (len(ARMS), 2, 7)
    np.testing.assert_array_equal(method_boxes[[0, 1, 3, 5], 0], np.broadcast_to(ev.old_pca, (4, 7)))
    packet.update(method_boxes=method_boxes, choices=np.asarray(choices, dtype=np.int16), **timings)
    return packet


def verify_old(entry, source, current):
    old = legacy(entry, source)
    np.testing.assert_array_equal(old['method_boxes'], current['method_boxes'][:5])
    n = int(old['common_n'])
    np.testing.assert_array_equal(old['common'][:n], current['mapped_boxes'])
    np.testing.assert_array_equal(np.unique(old['seeds'], axis=0), np.unique(current['seeds'], axis=0))
    assert old['point_sha'] == current['point_sha'] and old['sample_sha'] == current['sample_sha']
    for i, arm in enumerate(VIEWS):
        count = int(old['native_n'][i])
        for old_key, new_key in (('natural', 'natural_boxes'), ('links', 'links'), ('witnesses', 'witness')):
            np.testing.assert_array_equal(old[old_key][i, :count], current[f'{arm}_{new_key}'])
        np.testing.assert_array_equal(old['choices'][i], current[f'{arm}_legacy_ids'])
        np.testing.assert_array_equal(old['bank_scores'][i, :n], current[f'{arm}_mapped_unary'][:-1])


def predict():
    import torch
    from evidence162 import EvidenceGrounder
    from model51 import fingerprint

    pin = lock('prediction', FILES)
    roster = read(HERE / 'INPUT_SEAL.json')
    assert roster['registry'] == sha(INPUT) and not roster['FINAL_access']
    data = dict(zip(SPLITS, (read(PRIOR_INPUT), read(INPUT)), strict=True))
    settings = read(SOURCE_SETTINGS)['sources']
    totals = Counter()
    files = {}
    status('predicting', physical_frames={split: len(x) for split, x in data.items()})
    with (ROOT / 'research/GPU_RESOURCE.lock').open('a+') as resource:
        fcntl.flock(resource, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for s in PLATFORMS:
            torch.set_num_threads(1)
            net = EvidenceGrounder(s)
            original = fingerprint(net.base), fingerprint(net.matcher)
            for split, panel in data.items():
                dest = WORK / 'predictions' / split
                dest.mkdir(parents=True, exist_ok=True)
                for entry in panel:
                    if s == entry['target']:
                        continue
                    tag = f"{entry['case']:02d}_{s}.npz"
                    started = time.perf_counter()
                    with torch.inference_mode():
                        values = predict_case(net, entry, settings[s])
                    if split == 'replay170':
                        verify_old(entry, s, values)
                    path = dest / tag
                    assert not path.exists()
                    npz_atomic(path, **values, arms=np.asarray(ARMS), design=pin)
                    files[f'{split}/{tag}'] = sha(path)
                    totals[split] += 1
                    print('R171', split, tag, entry['rewrite']['status'], round(time.perf_counter() - started, 3), flush=True)
            assert original == (fingerprint(net.base), fingerprint(net.matcher))
            del net
            gc.collect()
            torch.cuda.empty_cache()
    assert totals == Counter({key: 2 * len(rows) for key, rows in data.items()})
    atomic_json(HERE / 'PREDICTION_SEAL.json', dict(design=pin, files=files, rows=dict(totals), r170_control_parity_rows=totals['replay170'],
                r170_all_five_endpoints_and_associations_bit_exact=True, source_fingerprints_unchanged=True,
                new_truth_read_by_prediction=False, input_seal=sha(HERE / 'INPUT_SEAL.json'),
                prior_predictions=sha(R170 / 'PREDICTION_SEAL.json'), FINAL_access=False))


if __name__ == '__main__':
    predict()
