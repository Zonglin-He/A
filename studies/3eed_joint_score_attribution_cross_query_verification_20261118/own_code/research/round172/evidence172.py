"""Construct label-free C/D correspondence on the original physical point sample."""
from collections import Counter
import hashlib

import numpy as np

from scope172 import HERE, NEW_INPUT, PLATFORMS, PRIOR_INPUT, R171, R171_WORK, WORK, atomic_json, lock, read, sha, status
from common160 import array_sha, npz_atomic
from geometry162 import in_box
from predict165 import points_for
from predict171 import read_npz
from requery169 import associate
from review171 import verify_predictions


CODE = ('evidence172.py',)


def support(boxes, points):
    return (np.stack([in_box(points, np.r_[b[:6], 0.]) for b in boxes])
            if len(boxes) else np.zeros((0, len(points)), dtype=bool))


def correspond(c, d):
    """Retain mutual highest unique links under the sealed association rule."""
    _, forward, witness = associate(c, d, np.zeros(len(c)))
    _, reverse, _ = associate(d, c, np.zeros(len(d)))
    rank = np.zeros(len(d), dtype=np.int16)
    strength = np.zeros(len(d))
    matched = np.flatnonzero(forward >= 0)
    for i in matched:
        j = forward[i]
        if reverse[j] == i:
            rank[j] = i + 1
            strength[j] = witness[i]
    return rank, strength


def scores(d, points, path):
    a = support(d['C_natural_boxes'], points)
    b = support(d['D_natural_boxes'], points)
    rank, jaccard = correspond(a, b)
    c = np.divide(1., rank, out=np.zeros(len(rank)), where=rank > 0)
    u = d['D_native_unary']
    assert len(u) == len(c)+1 and (u[:-1] <= 0).all()
    shift = 0
    if len(c) > 1:
        seed = hashlib.sha256(('R172|fixed-shift|'+path).encode()).digest()
        shift = 1 + int.from_bytes(seed[:4], 'little') % (len(c)-1)
    moved = np.roll(c, shift)
    assert np.array_equal(np.sort(c), np.sort(moved))
    return dict(c_rank=rank, overlap=jaccard, c_to_d_bonus=c, shifted_bonus=moved,
                D_only=u, C_D=np.r_[u[:-1]+c, u[-1]], shifted_C_D=np.r_[u[:-1]+moved, u[-1]],
                sample_sha=d['sample_sha'], point_sha=d['point_sha'], shift=shift)


def collect():
    stamp = lock('evidence', CODE)
    sealed = verify_predictions()
    values = {'replay170': read(PRIOR_INPUT), 'new_frames': read(NEW_INPUT)}
    status('sealing_evidence')
    written = {}
    totals = Counter()
    observed_frames = 0
    for cohort, cohort_rows in values.items():
        for entry in cohort_rows:
            row = entry['row']
            assert sha(row['pointpath']) == row['point_sha256']
            pts, indices = points_for(row)
            pt, ix = array_sha(pts), array_sha(indices)
            unique = np.unique(pts, axis=0)
            observed_frames += 1
            for source in PLATFORMS:
                if source == entry['target']:
                    continue
                path = f"{cohort}/{entry['case']:02d}_{source}.npz"
                old = R171_WORK / 'predictions' / path
                assert sha(old) == sealed['files'][path]
                d = read_npz(old)
                assert d['sample_sha'] == ix and d['point_sha'] == pt
                result = scores(d, unique, path)
                dest = WORK / 'evidence_PRIVATE' / path
                dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                assert not dest.exists()
                npz_atomic(dest, **result, design=stamp)
                written[path] = sha(dest)
                totals[cohort] += 1
    assert dict(totals) == sealed['rows'] and observed_frames == 84
    receipt = dict(files=written, parent=sha(R171 / 'PREDICTION_SEAL.json'), design=stamp,
                   rows=dict(totals), observed_frames=observed_frames, point_and_sample_hash_verified=len(written),
                   target_identity_labels_seen_by_this_stage=False, FINAL_access=False)
    atomic_json(HERE / 'EVIDENCE_SEAL.json', receipt)
    print('R172 sealed C/D links', dict(totals), flush=True)


def verify_evidence():
    stamp = lock('evidence', CODE)
    parent = verify_predictions()
    r = read(HERE / 'EVIDENCE_SEAL.json')
    assert r['design'] == stamp and r['parent'] == sha(R171 / 'PREDICTION_SEAL.json')
    assert r['rows'] == parent['rows'] and len(r['files']) == r['point_and_sample_hash_verified'] == 168
    assert r['files'].keys() == parent['files'].keys()
    assert not r['FINAL_access'] and not r['target_identity_labels_seen_by_this_stage']
    assert all(sha(WORK / 'evidence_PRIVATE' / p) == h for p, h in r['files'].items())
    return r


if __name__ == '__main__':
    collect()
