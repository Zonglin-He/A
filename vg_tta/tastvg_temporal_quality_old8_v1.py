"""A label-free candidate scorer from cached, projected PE features.

This is an OIC-inspired inference proxy, not AutoLoc training or an IoU head.
It never creates, changes, merges or removes a student interval.
"""
import numpy as np

ALPHA = .25
TIE_EPS = 1e-12


def activation(video, text):
    v = np.asarray(video, dtype=np.float64)
    q = np.asarray(text, dtype=np.float64)
    assert v.ndim == q.ndim == 2 and v.shape[0] == q.shape[0] == 1024
    assert v.shape[1] > 0 and q.shape[1] > 0
    assert np.isfinite(v).all() and np.isfinite(q).all()
    # UniversalVTG's poolandtoken contract prepends the projected pooled query.
    q = q[:, 0]
    vn = np.linalg.norm(v, axis=0); qn = np.linalg.norm(q)
    available = bool(qn > 0 and np.all(vn > 0))
    if not available:
        return np.zeros(v.shape[1]), False
    curve = (q / qn) @ (v / vn[None, :])
    assert np.isfinite(curve).all() and np.max(np.abs(curve)) <= 1 + 1e-12
    return curve, True


def feature_edges(count, duration):
    """Phase-zero 2-Hz observations; final bin reaches the observed window end."""
    assert duration > 0 and count == max(1, int(duration * 2))
    edges = np.r_[np.arange(count, dtype=np.float64) / 2, duration] / duration
    assert edges[0] == 0 and edges[-1] == 1 and np.all(np.diff(edges) > 0)
    return edges


def integral(curve, edges, lo, hi):
    widths = np.maximum(0., np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo))
    return float(np.dot(widths, curve)), float(widths.sum())


def contrast(curve, edges, intervals):
    curve = np.asarray(curve, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    assert len(edges) == len(curve) + 1 and np.all(np.diff(edges) > 0)
    assert edges[0] == 0 and edges[-1] == 1 and np.isfinite(curve).all()
    global_mean = float(np.dot(np.diff(edges), curve))
    results = []
    for start, end in intervals:
        assert 0 <= start < end <= 1
        width = end - start
        left = max(0., start - ALPHA * width)
        right = min(1., end + ALPHA * width)
        value, length = integral(curve, edges, start, end)
        lv, ll = integral(curve, edges, left, start)
        rv, rl = integral(curve, edges, end, right)
        inner = value / length
        # A full-window candidate has no observed outer area. Give it the
        # neutral contrast against this same window, rather than fabricating
        # observations outside the video or silently deleting the candidate.
        outer = (lv + rv) / (ll + rl) if ll + rl > 0 else global_mean
        results.append(dict(score=inner - outer, inner_mean=inner, outer_mean=outer,
            left_mean=lv / ll if ll else None, right_mean=rv / rl if rl else None,
            inside_length=length, outside_length=ll + rl,
            outer_window=[left, right], no_outer_observation=ll + rl == 0))
    return results


def decide(curve, edges, intervals, old_selected, available=True):
    assert len(intervals) == 8 and 0 <= old_selected < 8
    details = contrast(curve, edges, intervals)
    scores = [r['score'] for r in details]
    reason = 'none'
    if not available:
        selected = old_selected; reason = 'unavailable_embedding'
    elif np.ptp(curve) <= TIE_EPS:
        selected = old_selected; reason = 'constant_semantic_evidence'
    else:
        best = max(scores)
        selected = next(i for i, s in enumerate(scores) if best - s <= TIE_EPS)
    return dict(scores=scores, selected=selected, details=details,
        fallback_reason=reason, alpha=ALPHA, tie_epsilon=TIE_EPS)
