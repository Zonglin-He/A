"""Label-free stratified endpoint allocation and CPU merged-grid ceilings."""
import numpy as np


def grid_pairs(frame_ids):
    f = np.asarray(frame_ids, dtype=np.int64)
    assert len(f) >= 4 and np.all(np.diff(f) > 0)
    i, j = np.triu_indices(len(f), 1)
    return i, j, np.column_stack((f[i], f[j] + 1))


def endpoint_prior(logits, records, frame_ids):
    """Interleave two separately normalized frozen heads; not envelope policy."""
    ids = list(frame_ids)
    assert len(logits) == len(records) == 2
    p = np.zeros((len(ids), 2), dtype=np.float64)
    occupied = set()
    for z, record in zip(logits, records):
        x = np.asarray(z, dtype=np.float64).reshape(-1, 2)
        assert x.shape[0] == len(record['frame_ids']) and np.isfinite(x).all()
        q = np.exp(x - x.max(0)); q /= q.sum(0)
        for at, f in enumerate(record['frame_ids']):
            pos = ids.index(f)
            assert pos not in occupied
            occupied.add(pos); p[pos] = .5 * q[at]
    assert len(occupied) == len(ids) and (p > 0).all()
    assert np.allclose(p.sum(0), 1, rtol=0, atol=1e-12)
    return p


def allocate(native_indices, frame_ids, logits, records):
    i, j, physical = grid_pairs(frame_ids)
    f = np.asarray(frame_ids, np.int64); length = int(f[-1]+1-f[0])
    p = endpoint_prior(logits, records, frame_ids)
    score = np.log(p[i, 0]) + np.log(p[j, 1])
    ranked = np.lexsort((j, i, -score))
    native = list(map(int, native_indices))
    assert 0 <= native[0] < native[1] < len(f)
    out = [dict(indices=native, physical_interval=[int(f[native[0]]), int(f[native[1]]+1)],
                origin='current_native', fallback=False)]
    used = {tuple(native)}
    # Twice-center numerator permits exact rational bin boundaries.
    center2 = physical.sum(1) - 2*f[0]
    position = np.minimum(2, (3*center2)//(2*length))
    duration = physical[:, 1]-physical[:, 0]
    size = np.where(3*duration <= length, 0,
                    np.where(3*duration <= 2*length, 1, 2))
    strata = [(c, d) for c in range(3) for d in range(2)] + [(None, 2)]
    for center, band in strata:
        mask = (size == band) & ((position == center) if center is not None else True)
        eligible = [int(t) for t in ranked if mask[t] and (int(i[t]), int(j[t])) not in used]
        fallback = not eligible
        if fallback:
            eligible = [int(t) for t in ranked if (int(i[t]), int(j[t])) not in used]
        assert eligible, 'Fewer than eight distinct legal grid intervals'
        at = eligible[0]; pair = [int(i[at]), int(j[at])]; used.add(tuple(pair))
        out.append(dict(indices=pair, physical_interval=physical[at].tolist(),
            origin=f'position_{center}_duration_{band}', position_bin=center,
            duration_band=band, fallback=fallback, allocation_logscore=float(score[at])))
    assert len(out) == len(used) == 8
    return out


def choose(candidates, proposals, confidence):
    from vg_tta.tastvg_temporal_qualification_v1 import critic_scores
    scores = critic_scores([c['physical_interval'] for c in candidates], proposals, confidence)
    return dict(scores=scores.tolist(), selected=int(np.argmax(scores)),
                ties=int(np.count_nonzero(scores == scores.max())))


def grid_values(scorer, frame_ids):
    """Enumerate every legal endpoint pair using the literal dense denominator."""
    i, j, intervals = grid_pairs(frame_ids)
    g, h = scorer.span
    a, z = intervals.T
    prefix = np.r_[0., np.cumsum(scorer.iou)]
    left = np.searchsorted(scorer.fids, np.maximum(a, g), side='left')
    right = np.searchsorted(scorer.fids, np.minimum(z, h), side='left')
    # Non-overlapping intervals have empty intersections, not negative sums.
    sums = np.where(right >= left, prefix[right]-prefix[left], 0.)
    v = sums / np.maximum(np.maximum(z, h)-np.minimum(a, g), 1)
    return dict(i=i, j=j, intervals=intervals, values=v,
                best=int(np.argmax(v)), count=len(v))
