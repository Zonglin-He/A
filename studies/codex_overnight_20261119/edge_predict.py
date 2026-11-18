"""Preserve every sealed R171 record and add two frozen cross-role interventions."""
import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R171 = ROOT / 'research/round171'
PRIVATE = HERE / 'private/edge'

sys.path.insert(0, str(R171))
from scope171 import (ARMS as OLD_ARMS, INPUT, PLATFORMS, PRIOR_INPUT, SOURCE_SETTINGS,
                      SPLITS, WORK, atomic_json, read, sha)
from edges import decompose, point_support, responses, slot_proxy

SPEC = HERE / 'edge_protocol.json'
ARMS = tuple(read(SPEC)['arms'])


def pins():
    dependencies = (
        'research/round171/PREDICTION_SEAL.json',
        'research/round171/INPUT_SEAL.json',
        'research/round170/INPUT_SEAL.json',
        'research/round172/publication/PUBLICATION_RECEIPT.json',
        'research/round169/requery169.py',
        'research/round162/evidence162.py',
        'research/round76/model76.py',
        'research/round76/prepare76.py',
        'crossplatform/single_frame_sources.json',
        'crossplatform/ground_single_frame.py',
        'crossplatform/upstream/models/encoder_decoder_layers.py',
    )
    files = [HERE / f for f in ('edge_protocol.json', 'edges.py', 'edge_predict.py')]
    files.extend(ROOT / name for name in dependencies)
    from scope170 import HERE as r170
    assert sha(INPUT) == read(R171 / 'INPUT_SEAL.json')['registry']
    assert sha(PRIOR_INPUT) == read(r170 / 'INPUT_SEAL.json')['registry']
    assert not read(R171 / 'PREDICTION_SEAL.json')['FINAL_access']
    assert read(ROOT / dependencies[3])['commit'] == read(SPEC)['parent']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def projection(packet, subject, reference, forward, reverse, label):
    from assignment171 import assign

    us = packet['subject_unary']
    ur = packet['D_native_unary']
    sign, left = packet['sign'], packet['native_left']
    sc, rc = np.asarray(subject, int), np.asarray(reference, int)
    offsets = us.copy(), ur.copy()
    relation = sign * left
    direct = assign(us, ur, left, sign, 'independent')
    original = assign(us, ur, left, sign, 'joint')
    choices = [tuple(packet['choices'][0]), direct, original]
    np.testing.assert_array_equal(choices, packet['choices'][[0, OLD_ARMS.index('D_native_independent'),
                                                               OLD_ARMS.index('D_native_joint')]])
    factor = np.zeros_like(relation)
    shifted = factor.copy()
    ff, rr = factor.copy(), factor.copy()
    if len(sc) and len(rc):
        ix = np.ix_(sc, rc)
        u, v, pair = decompose(np.clip((forward + reverse) / 2, -1., 1.))
        offsets[0][sc] += u
        offsets[1][rc] += v
        factor[ix] = pair
        k = 1 + int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], 'little') % (len(rc) - 1)
        shifted[ix] = np.roll(pair, k, axis=1)
        ff[ix] = decompose(np.clip(forward, -1., 1.))[2]
        rr[ix] = decompose(np.clip(reverse, -1., 1.))[2]
    independent = offsets[0][:, None] + offsets[1][None, :]
    for score in [independent, independent + relation, independent + relation + factor,
                  independent + relation + shifted, independent + relation + ff, independent + relation + rr]:
        choices.append(np.unravel_index(score.argmax(), score.shape))
    output = []
    for i, j in choices:
        reference_box = packet['D_natural_boxes'][j] if 0 <= j < len(ur) - 1 else np.full(7, np.nan)
        output.append(np.stack((packet['subject_boxes'][i], reference_box)))
    output = np.asarray(output)
    np.testing.assert_array_equal(output[:3], packet['method_boxes'][[0, 5, 8]])
    return dict(choices=np.asarray(choices, np.int16), boxes=output, subject_valid=sc, reference_valid=rc,
                row_offset=offsets[0]-us, column_offset=offsets[1]-ur, edge=factor, mismatched=shifted,
                source_forward=forward, source_reverse=reverse)


def empty(packet, label, reason, elapsed=0.):
    return dict(**projection(packet, [], [], np.empty((0, 0)), np.empty((0, 0)), label),
                reason=reason, attention_error=0., seconds=elapsed)


def one(model, entry, setting, packet, label):
    from common160 import array_sha
    from geometry162 import in_box
    from predict170 import original_evidence
    from requery169 import native_donors, native_forward
    from replay168 import oriented
    from crossplatform.ground_single_frame import orient_box

    t0 = perf_counter()
    if entry['rewrite']['status'] != 'supported':
        return empty(packet, label, 'unsupported_rewrite')
    sv = packet['subject_stats'][:, 1] >= 3
    rv = packet['native_stats'][:, 1] >= 3
    if sv.sum() < 2 or rv.sum() < 2:
        return empty(packet, label, 'insufficient_visible_regions')
    row = entry['row']
    assert sha(row['pointpath']) == row['point_sha256'] and sha(row['imagepath']) == row['image_sha256']
    ev, original, _, prior = original_evidence(model, row, setting)
    assert array_sha(ev.points) == packet['point_sha'] and array_sha(ev.indices) == packet['sample_sha']
    role = entry['role']
    np.testing.assert_array_equal(ev.old_pca, packet['method_boxes'][0, 0])
    ids = prior['bank'][0, prior['bank'][0] >= 0]
    natural_subject = oriented(row, prior, ev.boxes, ids, lambda _: ev.points)
    np.testing.assert_array_equal(natural_subject, packet['subject_boxes'])
    np.testing.assert_array_equal(prior['score'][0, prior['bank'][0] >= 0], packet['subject_unary'])
    extra = prior['novel'][np.isfinite(prior['novel']).all(1)]
    subject_query = slot_proxy(ids, ev.boxes, extra, np.unique(ev.points, axis=0))
    sc = np.flatnonzero(sv & (subject_query >= 0))
    rc = np.flatnonzero(rv)
    if sc.size < 2:
        return empty(packet, label, 'insufficient_subject_proxies', perf_counter() - t0)
    spec = entry['rewrite']['D']
    d = native_forward(model, row, ev.points, ev.indices, None, spec)
    z = d['z']
    refs, rank, _ = native_donors(d)
    np.testing.assert_array_equal(rank-rank.max(), packet['D_native_unary'][:-1])
    np.testing.assert_array_equal(np.stack([orient_box(ev.points, z['boxes'][j]) for j in refs]),
                                  packet['D_natural_boxes'])
    np.testing.assert_array_equal(z['xyz'], original['xyz'])
    np.testing.assert_array_equal(z['point_indices'], original['point_indices'])
    if not ev.role_mask[role].any() or len(d['roles']) != 2 or not d['roles'][1].any():
        return empty(packet, label, 'unavailable_role_tokens', perf_counter() - t0)

    raw_refs = np.c_[z['boxes'][refs[rc]], np.zeros(len(rc))]
    raw_subjects = np.r_[np.c_[ev.boxes, np.zeros(len(ev.boxes))], extra][ids[sc]]
    # Interruption operates on exactly the visible point seeds inside each fixed region.
    ref_mask = point_support(raw_refs, ev.seeds) & ev.visible
    sub_mask = point_support(raw_subjects, ev.seeds) & ev.visible
    assert (ref_mask.sum(1) >= 3).all() and (sub_mask.sum(1) >= 3).all()
    a, ea = responses(model, row, original, ev.role_mask[role], ref_mask)
    rew = dict(row, caption=spec['text'], id=hashlib.sha256(spec['text'].encode()).hexdigest())
    b, eb = responses(model, rew, z, d['roles'][1], sub_mask)
    forward = a[:, subject_query[sc]].T
    reverse = b[:, refs[rc]]
    assert forward.shape == reverse.shape == (len(sc), len(rc))
    assert np.isfinite(forward).all() and np.isfinite(reverse).all()
    return dict(**projection(packet, sc, rc, forward, reverse, label), seconds=perf_counter() - t0,
                reason='eligible', attention_error=max(ea, eb))


def predict(pilot):
    import torch
    from common160 import npz_atomic
    from evidence162 import EvidenceGrounder
    from model51 import fingerprint
    from predict171 import read_npz
    from review171 import verify_predictions

    private = PRIVATE / ('pilot' if pilot else 'full')
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = private / 'LOCK.json'
    plan = pins()
    if lock.exists():
        assert read(lock) == plan, 'Different code requires a distinct attempt'
    else:
        atomic_json(lock, plan)
    old = verify_predictions()
    roster = dict(zip(SPLITS, (read(PRIOR_INPUT), read(INPUT)), strict=True))
    settings = read(SOURCE_SETTINGS)['sources']
    files = {}
    counts = Counter()
    assert not (private / 'SEAL.json').exists()
    with (ROOT / 'research/GPU_RESOURCE.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        for source in PLATFORMS:
            model = EvidenceGrounder(source)
            initial = fingerprint(model.base), fingerprint(model.matcher)
            processed = 0
            for panel, entries in roster.items():
                (private / panel).mkdir(exist_ok=True)
                for entry in entries:
                    if source == entry['target']:
                        continue
                    label = f"{panel}/{entry['case']:02d}_{source}.npz"
                    original = WORK / 'predictions' / label
                    assert sha(original) == old['files'][label]
                    packet = read_npz(original)
                    if pilot and not (entry['rewrite']['status'] == 'supported'
                                      and (packet['subject_stats'][:, 1] >= 3).sum() >= 2
                                      and (packet['native_stats'][:, 1] >= 3).sum() >= 2):
                        continue
                    dest = private / label
                    if not dest.exists():
                        with torch.inference_mode():
                            value = one(model, entry, settings[source], packet, label)
                        npz_atomic(dest, **value, arms=np.asarray(ARMS), parent=old['files'][label], pin=sha(lock))
                    else:
                        value = read_npz(dest)
                        assert value['parent'] == old['files'][label] and value['pin'] == sha(lock)
                    files[label] = sha(dest)
                    counts[str(value['reason'])] += 1
                    print(source, label, str(value['reason']), float(value['seconds']),
                          value['source_forward'].shape, flush=True)
                    processed += 1
                    if pilot and processed == 1:
                        break
                if pilot and processed == 1:
                    break
            assert initial == (fingerprint(model.base), fingerprint(model.matcher))
            del model
            gc.collect()
            torch.cuda.empty_cache()
    if not pilot:
        assert files.keys() == old['files'].keys()
    else:
        assert len(files) == len(PLATFORMS)
    atomic_json(private / 'SEAL.json', dict(lock=sha(lock), parent=sha(R171 / 'PREDICTION_SEAL.json'), files=files,
                outcomes=dict(counts), pilot=pilot, target_labels_read_by_prediction=False,
                unchanged_weights=True, protected_data_access=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    predict(parser.parse_args().pilot)
