"""Verify a two-role binding against the proposal that swaps its physical endpoints."""
import numpy as np
from scipy.special import logsumexp

from edges import decompose, point_support


def transfer_scores(native_boxes, regions, points, log_probability, minimum_jaccard=.25):
    """Transport comparable evidence with an average that preserves probability units."""
    native = point_support(native_boxes, points).astype(np.float32)
    wanted = point_support(regions, points).astype(np.float32)
    hits = wanted @ native.T
    union = wanted.sum(1)[:, None] + native.sum(1) - hits
    jaccard = np.divide(hits, union, out=np.zeros_like(hits), where=union > 0)
    accept = (jaccard >= minimum_jaccard) & (hits >= 3)
    weights = np.where(accept, jaccard, 0.).astype(float)
    with np.errstate(divide='ignore'):
        transported = logsumexp(np.log(weights) + log_probability, axis=1)
        transported -= np.log(np.maximum(weights.sum(1), np.finfo(float).tiny))
    assert np.array_equal(np.isfinite(transported), accept.any(1))
    return transported, weights.max(1)


def factorize(f, d, subject_logits, reference_logits, left, sign):
    """Use the same localized evidence for unaries and joint factors."""
    ns, nr = len(subject_logits), len(reference_logits)-1
    baseline = subject_logits[:, None] + reference_logits[None, :] + sign*left
    where = np.isfinite(f) & np.isfinite(d)
    subject = np.flatnonzero(where[:ns])
    reference = np.flatnonzero(where[ns:])
    parts = np.zeros((2, ns, nr+1))
    part_offsets = parts.copy()
    if len(subject) >= 2 and len(reference) >= 2:
        j = reference+ns
        ix = np.ix_(subject, reference)
        effects = (f[subject, None]-f[None, j], d[None, j]-d[subject, None])
        for k, delta in enumerate(effects):
            pair = -np.logaddexp(0, -np.clip(delta, -4., 4.))
            row, col, term = decompose(pair)
            # Additive evidence sees the full marginal contributions of the same function.
            row_base = np.zeros(ns)
            col_base = np.zeros(nr+1)
            row_base[subject] = row
            col_base[reference] = col
            part_offsets[k] = row_base[:, None] + col_base[None, :]
            parts[k][ix] = term
    else:
        subject, reference = np.empty(0, int), np.empty(0, int)
    return baseline, part_offsets, parts, subject, reference


def weights_fit(samples):
    """Fit a convex pair-order likelihood on source-only matched alternatives."""
    from scipy.optimize import minimize
    from scipy.special import expit

    def objective(w):
        grad = np.zeros(2)
        avg = 0.
        for delta, offset in samples:
            odds = offset + delta @ w
            avg += np.logaddexp(0, -odds).mean()
            grad -= np.mean(expit(-odds)[:, None] * delta, axis=0)
        return avg/len(samples) + .1 * (w@w), grad/len(samples) + .2 * w

    if not samples:
        return np.zeros(2)
    out = minimize(objective, x0=np.array([.5, .5]), bounds=((0, 1.), (0, 1.)), jac=True,
                   method='L-BFGS-B', options={'maxiter': 60, 'ftol': 1e-12})
    assert out.success and np.isfinite(out.x).all()
    return out.x


def alternative_pairs(item, subject_ids, reference_ids, truth):
    """Select source negatives with similar factual geometry and real observed identities."""
    geometry = item['sign'] * item['left'][:, :-1]
    subj = item['subject_common']
    ref = item['reference_common']
    if not len(subj) or not len(ref):
        return
    s, r = np.meshgrid(subj, ref, indexing='ij')
    sid, rid = subject_ids[s], reference_ids[r]
    correct = (sid == truth[0]) & (rid == truth[1]) & (geometry[s, r] > 0.)
    known = (sid >= 0) & (rid >= 0) & (sid != rid)
    offset, edge = item['offset'], item['edge']
    terms = (item['baseline'] + offset.sum(0))[s, r]
    if not correct.any():
        return
    best = np.argmax(np.where(correct, terms, -np.inf))
    best_s, best_r = np.unravel_index(best, terms.shape)
    matched = known & ~correct & (geometry[s, r] > 0.)
    matched &= abs(geometry[s, r]-geometry[s[best_s, best_r], r[best_s, best_r]]) <= .25
    grouping = {}
    for u, v in np.argwhere(matched):
        identity = int(sid[u,v]), int(rid[u,v])
        if identity not in grouping or terms[u,v] > terms[grouping[identity]]:
            grouping[identity] = int(u), int(v)
    if not grouping:
        return
    i, j = np.asarray(list(grouping.values())).T
    one = edge[:, s[best_s,best_r], r[best_s,best_r]]
    other = edge[:, s[i,j], r[i,j]]
    return (one[:, None]-other).T, terms[best_s,best_r]-terms[i,j]
