"""Source-balanced positive-score pairs; fixed pointwise mean-delta gate."""
import numpy as np
from sklearn.isotonic import isotonic_regression
from scripts.tastvg_anchor_certification_math_v1 import EPS, SEED, DRAWS, top1, summarize

IQR_EPS = 1e-8
Q = .05
ARMS = ['A', 'L32', 'Pair-Raw', 'Pair-Norm']

def spread(scores):
    s = np.asarray(scores, dtype=np.float64)
    assert s.shape == (32,) and np.isfinite(s).all()
    return float(np.quantile(s, .75, method='linear') - np.quantile(s, .25, method='linear'))

def pairs(rows):
    """One unordered score-strict pair, oriented toward the higher score.

    GT ties and negative GT differences are retained. Score ties alone are
    excluded. A source with no strict score pair contributes no fit weight.
    """
    out = []; metadata = []
    for source_index, r in enumerate(rows):
        s = np.asarray(r['scores'], float); t = np.asarray(r['candidate_t'], float)
        assert s.shape == t.shape == (32,) and np.isfinite(t).all()
        i, j = np.triu_indices(32, 1); good = abs(s[i] - s[j]) > EPS
        i, j = i[good], j[good]; n = len(i); scale = spread(s)
        metadata.append(dict(source_id=r['source_id'], strict_pairs=n, score_ties=496-n,
                             IQR=scale, zero_IQR=scale == 0., total_fit_weight=float(n > 0)))
        for a, b in zip(i, j):
            if s[a] < s[b]: a, b = b, a
            margin = float(s[a]-s[b])
            out.append(dict(source_index=source_index, source_id=r['source_id'],
                            high_index=int(a), low_index=int(b), raw_margin=margin,
                            norm_margin=margin/(scale+IQR_EPS), delta_t=float(t[a]-t[b]),
                            weight=1./n))
    return out, metadata

def sorted_fit(x, y, w):
    """Collapse identical x before the Cython weighted PAV solver."""
    keep = w > 0; x, y, w = x[keep], y[keep], w[keep]
    knots, inverse = np.unique(x, return_inverse=True)
    mass = np.bincount(inverse, weights=w)
    mean = np.bincount(inverse, weights=w*y)/mass
    return knots, isotonic_regression(mean, sample_weight=mass, increasing=True)

def calibrate(pair_rows, source_count, arm):
    field = 'raw_margin' if arm == 'Pair-Raw' else 'norm_margin'
    x = np.array([r[field] for r in pair_rows]); y = np.array([r['delta_t'] for r in pair_rows])
    w = np.array([r['weight'] for r in pair_rows]); ids = np.array([r['source_index'] for r in pair_rows])
    if not len(x): return dict(available=False, knots=[], mean=[], lower=[], domain=None)
    order = np.argsort(x, kind='stable'); x, y, w, ids = x[order], y[order], w[order], ids[order]
    knots, fitted = sorted_fit(x, y, w)
    rng = np.random.default_rng(SEED); boot = np.empty((DRAWS, len(knots)), dtype=np.float64)
    for draw in range(DRAWS):
        counts = np.bincount(rng.integers(0, source_count, source_count), minlength=source_count)
        weights = w * counts[ids]
        # All actual source rows have strict pairs. Do not silently replace a
        # degenerate all-empty resample in a future input cohort.
        assert weights.sum() > 0
        bx, by = sorted_fit(x, y, weights); boot[draw] = np.interp(knots, bx, by)
    lower = np.minimum(np.quantile(boot, Q, axis=0, method='linear'), fitted)
    return dict(available=True, arm=arm, knots=knots.tolist(), mean=fitted.tolist(), lower=lower.tolist(),
                domain=[float(knots[0]), float(knots[-1])], source_count=source_count,
                strict_pair_count=len(x), effective_independent_units=source_count,
                bootstrap_draws=DRAWS, seed=SEED, lower_quantile=Q,
                interpretation='pointwise bootstrap lower fitted mean delta; not individual/conformal/target safety')

def evaluate(model, margin):
    inside = model['available'] and model['knots'][0] <= margin <= model['knots'][-1]
    return dict(in_domain=bool(inside), mean=float(np.interp(margin, model['knots'], model['mean'])) if inside else None,
                lower=float(np.interp(margin, model['knots'], model['lower'])) if inside else None)

def decide(scores, anchor, models):
    k = top1(scores, anchor); raw = float(scores[k]-scores[anchor]); scale = spread(scores)
    choices = dict(A=anchor, L32=k); evidence = {}
    for arm in ARMS[2:]:
        margin = raw if arm == 'Pair-Raw' else raw/(scale+IQR_EPS)
        e = evaluate(models[arm], margin)
        reason = ('no_unique_positive_winner' if k == anchor else
                  'outside_source_pair_margin_support' if not e['in_domain'] else
                  'nonpositive_lower_mean_delta' if e['lower'] <= EPS else 'accepted')
        e.update(margin=margin, reason=reason); evidence[arm] = e
        choices[arm] = k if reason == 'accepted' else anchor
    return dict(top_index=k, raw_margin=raw, IQR=scale, evidence=evidence, choices=choices)
