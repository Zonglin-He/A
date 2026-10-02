"""CPU temporal evidence maps and consensus-weighted native-candidate critic.

Proposal agreement is an observable heuristic, not calibrated localization
quality. Duplicate proposals are retained exactly as supplied by UniversalVTG.
All intervals use the already cached physical, half-open frame coordinates.
"""
import numpy as np


def overlap_matrix(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert (a[:, 1] > a[:, 0]).all() and (b[:, 1] > b[:, 0]).all()
    inter = np.maximum(0., np.minimum(a[:, None, 1], b[None, :, 1])
                       - np.maximum(a[:, None, 0], b[None, :, 0]))
    union = (a[:, 1]-a[:, 0])[:, None] + (b[:, 1]-b[:, 0])[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def proposal_quality(proposals, confidence):
    p = np.asarray(proposals, dtype=np.float64).reshape(-1, 2)
    c = np.asarray(confidence, dtype=np.float64)
    assert c.shape == (len(p),) and np.isfinite(c).all() and (c >= 0).all()
    mutual = overlap_matrix(p, p)
    np.fill_diagonal(mutual, 0.)
    row_sums = mutual.sum(1)
    # No second proposal means no observable cross-proposal consensus. There
    # is no invented agreement=1 or raw-confidence fallback.
    agreement = row_sums/(len(p)-1) if len(p) > 1 else np.zeros(len(p))
    quality = c*agreement
    return agreement, quality, row_sums


def quantile_indices(weights):
    w = np.asarray(weights, dtype=np.float64)
    assert w.ndim == 1 and np.isfinite(w).all() and (w >= 0).all()
    if w.sum() == 0.:
        return None
    cdf = np.cumsum(w)/w.sum()
    return np.searchsorted(cdf, [.1, .3, .5, .7, .9], side='left').tolist()


def route(candidates, frame_ids, proposals, confidence):
    ids = np.asarray(frame_ids)
    assert len(ids) > 1 and (np.diff(ids) > 0).all()
    assert 0 < len(candidates) <= 8
    spans = np.asarray([c['physical_interval'] for c in candidates], np.float64)
    p = np.asarray(proposals, np.float64).reshape(-1, 2)
    c = np.asarray(confidence, np.float64)
    # Validate endpoints even when M is zero or one.
    match = overlap_matrix(spans, p)
    agreement, q, row_sums = proposal_quality(p, c)
    sw = ((ids[None, :] >= spans[:, :1]) & (ids[None, :] < spans[:, 1:])).mean(0)
    assert sw.sum() > 0.
    ew = np.zeros(len(ids), dtype=np.float64)
    if q.sum() > 0.:
        votes = (ids[None, :] >= p[:, :1]) & (ids[None, :] < p[:, 1:])
        ew = (q[:, None]*votes).sum(0)/q.sum()
    maps = dict(S=sw, E=ew, SE=.5*sw+.5*ew)
    raw = (match*c[None, :]).max(1) if len(p) else np.zeros(len(spans))
    qc = (match*q[None, :]).max(1) if len(p) else np.zeros(len(spans))
    return dict(weights={k:v.tolist() for k,v in maps.items()},
                quantiles={k:quantile_indices(v) for k,v in maps.items()},
                proposal_agreement=agreement.tolist(), proposal_quality=q.tolist(),
                proposal_overlap_row_sums=row_sums.tolist(),
                candidate_proposal_overlap=match.tolist(),
                current_scores=raw.tolist(), qc_scores=qc.tolist(),
                current_selected=int(np.argmax(raw)), qc_selected=int(np.argmax(qc)),
                expert_available=bool(ew.sum() > 0.),
                proposal_quality_sum=float(q.sum()),
                duplicate_proposals=len(p)-len(set(map(tuple, p.tolist()))))


def support_statistics(weights, event, scored, valid, quantiles, uniform):
    w = np.asarray(weights, np.float64)
    event, scored, valid = [np.asarray(a, bool) for a in [event, scored, valid]]
    assert w.shape == event.shape == scored.shape == valid.shape
    total, inside = float(w.sum()), float(w[event].sum())
    union = total+int(event.sum())-inside
    pos = np.asarray(quantiles, int) if quantiles is not None else None
    hits = int(event[pos].sum()) if pos is not None else None
    scored_hits = int(scored[pos].sum()) if pos is not None else None
    return dict(total_mass=total, event_mass=inside, event_frames=int(event.sum()),
        scored_mass=float(w[scored].sum()), scored_frames=int(scored.sum()),
        positive_event_frames=int(((w > 0)&event).sum()),
        positive_scored_frames=int(((w > 0)&scored).sum()),
        event_recall=float((w[event] > 0).mean()) if event.any() else None,
        gt_mass=inside/total if total > 0. else None,
        scored_gt_mass=float(w[scored].sum()/total) if total > 0. else None,
        scored_recall=float((w[scored] > 0).mean()) if scored.any() else None,
        soft_iou=inside/union if union > 0. else None,
        quantile_hits=hits, quantile_coverage=hits/5 if hits is not None else None,
        quantile_scored_hits=scored_hits,
        quantile_scored_coverage=scored_hits/5 if scored_hits is not None else None,
        quantile_unique=len(set(pos.tolist())) if pos is not None else 0,
        quantile_any_gt=bool(hits) if hits is not None else False,
        uniform_hits=int(event[np.asarray(uniform, int)].sum()),
        uniform_coverage=float(event[np.asarray(uniform, int)].mean()),
        valid_reference_mass=float(w[valid].sum()),
        raw_valid_references=int(valid.sum()), raw_gt_references=int((valid&scored).sum()),
        useful_reference_discarded=bool((valid&scored).any() and w[valid].sum() == 0.),
        available=bool(total > 0.))
