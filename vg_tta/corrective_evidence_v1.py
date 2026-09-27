"""Offline diagnostic primitives. No model calls, label access or parameter updates.

Predictive AUC and action-conditional corrective AUC are different estimands.
The fixed soft-IoU readout below is an explicit testable interface, not an
estimate of all information in a signal, nor of conditional mutual information.
"""
from __future__ import annotations

import numpy as np


def grid(frame_ids):
    ids = np.asarray(frame_ids, dtype=np.float64)
    if ids.ndim != 1 or len(ids) < 2 or not np.isfinite(ids).all() or (np.diff(ids) <= 0).any():
        raise ValueError("Need at least two finite, strictly increasing positions")
    return ids


def interval(ids, indices):
    ids = grid(ids)
    a, b = map(int, indices)
    if not 0 <= a < b < len(ids):
        raise ValueError("Interval must contain at least two grid positions")
    return [float(ids[a]), float(ids[b] + 1)]


def time_weights(frame_ids):
    ids = grid(frame_ids)
    edges = np.r_[ids[0], (ids[:-1] + ids[1:]) / 2, ids[-1] + 1]
    return np.diff(edges)


def actions(frame_ids, indices, fraction=0.05):
    """GT-free deterministic actions, retaining clipped and no-op outcomes."""
    ids = grid(frame_ids)
    a, b = map(int, indices)
    start, end = interval(ids, (a, b))
    if not 0 < fraction < 0.5:
        raise ValueError("fraction must lie in (0, .5)")
    lo, hi = float(ids[0]), float(ids[-1] + 1)
    step = fraction * (hi - lo)
    requests = {
        "noop": (start, end),
        "expand": (max(lo, start - step), min(hi, end + step)),
        "shrink": (start + step, end - step),
        "shift_left": (start + max(-step, lo - start), end + max(-step, lo - start)),
        "shift_right": (start + min(step, hi - end), end + min(step, hi - end)),
    }
    result = {}
    for name, (s, e) in requests.items():
        i = int(np.argmin(abs(ids - s)))
        j = int(np.argmin(abs((ids + 1) - e)))
        invalid = s >= e or i >= j
        if invalid:
            i, j = a, b
        realized = interval(ids, (i, j))
        result[name] = {
            "indices": [i, j], "physical": realized,
            "requested_physical": [s, e], "step_physical": step,
            "is_noop": i == a and j == b,
            "invalid_shrink_or_snap": bool(invalid),
            "snap_error": abs(realized[0] - s) + abs(realized[1] - e),
            "length_change_fraction": ((realized[1] - realized[0]) - (end - start)) / (hi - lo),
            "center_change_fraction": (sum(realized) - start - end) / (2 * (hi - lo)),
        }
    return result


def soft_iou(signal, frame_ids, indices):
    ids = grid(frame_ids)
    r = np.asarray(signal, dtype=float)
    if r.shape != ids.shape or not np.isfinite(r).all() or (r < 0).any() or (r > 1).any():
        raise ValueError("Signal must be finite, [0,1] and match the grid")
    interval(ids, indices)
    a, b = map(int, indices)
    weights = time_weights(ids)
    inside = np.zeros(len(ids), dtype=bool)
    inside[a:b + 1] = True
    numerator = float(np.sum(weights[inside] * r[inside]))
    denominator = float(np.sum(weights * r) + np.sum(weights[inside] * (1 - r[inside])))
    return numerator / denominator


def signal_action_scores(signal, frame_ids, candidates):
    baseline = soft_iou(signal, frame_ids, candidates["noop"]["indices"])
    return {name: soft_iou(signal, frame_ids, a["indices"]) - baseline
            for name, a in candidates.items()}


def select_action(scores):
    # Keep no-op on exact/numerical ties. Choice does not inspect true utility.
    if "noop" not in scores or not all(np.isfinite(v) for v in scores.values()):
        raise ValueError("Finite scores including no-op are required")
    best = "noop"
    for name, value in scores.items():
        if value > scores[best] + 1e-12:
            best = name
    return best


def weighted_auc(y, scores, weights=None):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if y.ndim != 1 or scores.shape != y.shape or weights.shape != y.shape:
        raise ValueError("Mismatched AUC inputs")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid AUC inputs")
    p, n = y == 1, y == 0
    denominator = weights[p].sum() * weights[n].sum()
    if denominator <= 0:
        return None
    comparisons = (scores[p, None] > scores[None, n]).astype(float)
    comparisons += 0.5 * (scores[p, None] == scores[None, n])
    return float(np.sum(comparisons * weights[p, None] * weights[None, n]) / denominator)


def source_mean_ci(values, sources, seed=20260911, bootstrap=1000):
    buckets = {}
    for value, source in zip(values, sources):
        if value is not None:
            if not np.isfinite(value):
                raise ValueError("Do not silently drop nonfinite measurements")
            buckets.setdefault(str(source), []).append(float(value))
    if not buckets:
        return {"mean": None, "ci95": None, "sources": 0, "queries": 0}
    vals = np.array([np.mean(v) for _, v in sorted(buckets.items())])
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(bootstrap, len(vals)))].mean(1)
    return {"mean": float(vals.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(),
            "sources": len(vals), "queries": sum(map(len, buckets.values()))}


def source_auc_ci(y, scores, sources, seed=20260911, bootstrap=1000):
    y, scores = np.asarray(y, int), np.asarray(scores, float)
    sources = np.asarray(sources, str)
    if len(y) != len(sources):
        raise ValueError("Source labels must match observations")
    unique, inverse, counts = np.unique(sources, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse] if len(y) else np.array([])
    point = weighted_auc(y, scores, weights)
    info = {"auc": point, "sources": len(unique), "observations": len(y),
            "positive": int(y.sum()), "negative": int(len(y) - y.sum()), "ci95": None,
            "eligible_bootstraps": 0}
    if point is None:
        return info
    multiplicity = np.random.default_rng(seed).multinomial(
        len(unique), np.full(len(unique), 1 / len(unique)), size=bootstrap)
    w = multiplicity[:, inverse] * weights
    positive, negative = y == 1, y == 0
    comp = (scores[positive, None] > scores[None, negative]).astype(float)
    comp += .5 * (scores[positive, None] == scores[None, negative])
    numerator = ((w[:, positive] @ comp) * w[:, negative]).sum(1)
    denominator = w[:, positive].sum(1) * w[:, negative].sum(1)
    ok = denominator > 0
    if ok.any():
        info["ci95"] = np.quantile(numerator[ok] / denominator[ok], [.025, .975]).tolist()
    info["eligible_bootstraps"] = int(ok.sum())
    return info


def oof_probe(features, labels, sources):
    """Label-assisted development diagnostic, NEVER a test-time TTA function."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x, y, groups = np.asarray(features, float), np.asarray(labels, int), np.asarray(sources, str)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite probe features")
    n = min(5, len(set(groups)))
    if n < 2 or len(set(y)) < 2:
        return {"scores": None, "folds": [], "reason": "insufficient_classes_or_sources"}
    scores = np.full(len(y), np.nan)
    audits = []
    for train, test in GroupKFold(n_splits=n).split(x, y, groups):
        assert not set(groups[train]) & set(groups[test])
        single = len(set(y[train])) < 2
        if single:
            scores[test] = y[train].mean()
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=1., solver="liblinear", max_iter=2000, random_state=20260911))
            model.fit(x[train], y[train])
            scores[test] = model.predict_proba(x[test])[:, 1]
        audits.append({"train_sources": sorted(set(groups[train])),
                       "test_sources": sorted(set(groups[test])),
                       "single_class_train_constant_prediction": single})
    assert np.isfinite(scores).all()
    return {"scores": scores.tolist(), "folds": audits, "reason": None}
