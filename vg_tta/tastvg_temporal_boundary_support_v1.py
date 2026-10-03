"""Predeclared CPU hypothesis coverage and local boundary proxy, without GT."""
import copy
import numpy as np
from vg_tta.tastvg_temporal_quality_old8_v1 import contrast, integral, TIE_EPS

WINDOW_SECONDS = 1.0
SUPPORT_SIZE = 32


def expanded(old, frame_ids):
    """Keep every original entry; append maximin start/end coverage on the grid."""
    f = np.asarray(frame_ids, np.int64)
    assert len(old) == 8 and len(f) >= 9 and np.all(np.diff(f) > 0)
    i, j = np.triu_indices(len(f), 1)
    width = float(f[-1] + 1 - f[0])
    xy = np.column_stack(((f[i]-f[0])/width, (f[j]+1-f[0])/width))
    lookup = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(i, j))}
    seeds = [lookup[tuple(z['indices'])] for z in old]
    for z in old:
        a, b = z['indices']
        assert z['physical_interval'] == [int(f[a]), int(f[b]+1)]
    used = np.zeros(len(i), bool); used[seeds] = True
    distance = np.min(np.sum((xy[:, None]-xy[seeds][None])**2, axis=2), axis=1)
    result = copy.deepcopy(old)
    while len(result) < SUPPORT_SIZE:
        masked = np.where(used, -1., distance)
        at = int(np.argmax(masked))  # grid order is lexicographic (start,end)
        assert not used[at] and masked[at] >= 0
        result.append(dict(indices=[int(i[at]), int(j[at])],
            physical_interval=[int(f[i[at]]), int(f[j[at]]+1)],
            origin='endpoint_maximin_append', allocation_min_squared_distance=float(distance[at])))
        used[at] = True
        distance = np.minimum(distance, np.sum((xy-xy[at])**2, axis=1))
    assert result[:8] == old and len(result) == 32
    assert len({tuple(z['indices']) for z in result[8:]}) == 24
    return result


def boundary(curve, edges, intervals, duration_seconds):
    assert duration_seconds > 0
    width = WINDOW_SECONDS / duration_seconds
    out = []
    for a, b in intervals:
        assert 0 <= a < b <= 1
        ranges = [[a, min(b, a+width)], [max(0., a-width), a],
                  [max(a, b-width), b], [b, min(1., b+width)]]
        vals = [integral(curve, edges, x, y) for x, y in ranges]
        means = [v/l if l > 0 else None for v, l in vals]
        ds = means[0]-means[1] if means[1] is not None else 0.
        de = means[2]-means[3] if means[3] is not None else 0.
        out.append(dict(score=min(ds, de), start_transition=ds, end_transition=de,
            means=means, ranges=ranges, lengths=[l for _, l in vals],
            missing_start_context=means[1] is None, missing_end_context=means[3] is None,
            both_positive=ds > 0 and de > 0))
    return out


def select(scores, curve, available, baseline):
    assert scores and np.isfinite(scores).all()
    if not available:
        return baseline, 'unavailable_embedding'
    if np.ptp(curve) <= TIE_EPS:
        return baseline, 'constant_semantic_evidence'
    top = max(scores)
    return next(i for i, v in enumerate(scores) if top-v <= TIE_EPS), 'none'


def decisions(curve, edges, intervals, duration_seconds, baseline, available=True):
    bd = contrast(curve, edges, intervals)
    dd = boundary(curve, edges, intervals, duration_seconds)
    out = {}
    for name, details in [('B', bd), ('D', dd)]:
        scores = [z['score'] for z in details]
        at, reason = select(scores, curve, available, baseline)
        out[name] = dict(scores=scores, selected=at, details=details, fallback_reason=reason)
    return out
