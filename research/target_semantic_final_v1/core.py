"""Last fixed repair: coordinate-wise prototype stability, one native-seeded step.

Four chronological delete-group replicates are a sensitivity proxy, not a calibrated
sampling variance: native membership and neighbouring offsets are dependent.
"""
import numpy as np
from research.native_event_semantic_prototype_v1.core import EPS, membership, evidence, inclusive_scores, update

BLOCKS = 4

def moments(v, w):
    center = np.where(np.ptp(v, axis=0) == 0, v[0], v.mean(0))
    d = v-center
    if w.sum() == 0 or (1-w).sum() == 0:
        return center, center.copy(), center.copy()
    plus = center+(w[:, None]*d).sum(0)/w.sum()
    minus = center+((1-w)[:, None]*d).sum(0)/(1-w).sum()
    return center, np.clip(plus, EPS, 1-EPS), np.clip(minus, EPS, 1-EPS)

def logodds_delta(plus, minus):
    return np.log(plus)-np.log1p(-plus)-np.log(minus)+np.log1p(-minus)

def reliability(v, w):
    center, plus, minus = moments(v, w)
    eta = logodds_delta(plus, minus)
    groups = np.array_split(np.arange(len(v)), BLOCKS)
    replicates = []
    for group in groups:
        take = np.ones(len(v), dtype=bool)
        take[group] = False
        _, a, b = moments(v[take], w[take])
        replicates.append(logodds_delta(a, b))
    replicates = np.asarray(replicates)
    variance = (BLOCKS-1)/BLOCKS*((replicates-replicates.mean(0))**2).sum(0)
    denominator = eta**2+variance
    r = np.divide(eta**2, denominator, out=np.zeros_like(eta), where=denominator>0)
    return center, plus, minus, r, dict(eta=eta, delete_group_eta=replicates, variance=variance)

def stable_evidence(p, q0, *, permutation=None, disabled=False):
    if disabled:
        return evidence(p, q0)
    p = np.asarray(p, dtype=np.float64)
    assert len(p) >= 8
    w = membership(q0)
    L = np.zeros(len(p))
    stats = []
    for parity in (0, 1):
        train = np.arange(len(p)) % 2 != parity
        center, plus, minus, r, extra = reliability(p[train], w[train])
        applied = r if permutation is None else r[permutation]
        a = np.clip(center+applied*(plus-center), EPS, 1-EPS)
        b = np.clip(center+applied*(minus-center), EPS, 1-EPS)
        held = ~train
        L[held] = (p[held]*np.log(a/b)+(1-p[held])*np.log((1-a)/(1-b))).sum(1)
        stats.append(dict(heldout_parity=parity, positive=a, negative=b,
                          raw_positive=plus, raw_negative=minus, center=center,
                          reliability=r, applied_reliability=applied, **extra))
    return L, stats

def match_strength(L, target):
    norm = np.linalg.norm(L)
    factor = np.linalg.norm(target)/norm if norm else 0.
    out = L*factor
    assert abs(np.linalg.norm(out)-np.linalg.norm(target)) < 1e-11 or norm == 0
    return out, float(factor)

def candidates(p, q0, randomp, reflectedp, seed):
    L, stats = stable_evidence(p, q0)
    parentL, _ = evidence(p, q0)
    uniform, factor = match_strength(parentL, L)
    perm = np.random.default_rng(seed).permutation(p.shape[1])
    pl, ps = stable_evidence(p, q0, permutation=perm)
    pl, pf = match_strength(pl, L)
    wrong, _ = stable_evidence(p[::-1].copy(), q0)
    random, _ = stable_evidence(randomp, q0)
    reflected, _ = stable_evidence(reflectedp, q0)
    signals = dict(JackknifePrototype=L, UniformMatched=uniform, PermutedMatched=pl,
                   WrongTime=wrong, RandomSemantic=random, ReflectedSemantic=reflected)
    qs = {a: update(q0, inclusive_scores(v)) for a, v in signals.items()}
    legal = (q0>0).astype(np.float64)
    legal /= legal.sum()
    qs['DirectSameEvidence'] = update(legal, inclusive_scores(L))
    return qs, dict(L=signals, prototypes=stats, permuted_prototypes=ps,
                    permutation=perm, uniform_factor=factor, permutation_factor=pf)
