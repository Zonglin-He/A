"""One video-wide event/background prototype pair; no offset-wise fitting."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

KAPPA = 64.0
BETA = 1.0


def membership(q):
    """Soft event membership induced by the legal interval posterior."""
    return np.diag(np.cumsum(np.cumsum(q, axis=0)[:, ::-1], axis=1)[:, ::-1]).copy()


def unit(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)


def inclusive_scores(evidence):
    cumulative = np.r_[0., np.cumsum(evidence)]
    return cumulative[1:][None, :] - cumulative[:-1][:, None]


def update(prior, score):
    legal = prior > 0
    if np.all(score[legal] == 0):
        return prior.copy()
    log_mass = np.log(prior[legal]) + score[legal]
    log_mass -= log_mass.max()
    posterior = np.zeros_like(prior)
    posterior[legal] = np.exp(log_mass)
    posterior /= posterior.sum()
    return posterior


def evidence_from_membership(features, weights, kappa=KAPPA):
    """Fit ONE pair on all frames using native soft membership.

    Common-mean shrinkage is unchanged. Evaluation frames participate in the
    prototypes, so this is within-video self-consistency, not cross-validation
    or calibrated event probability. The native Source still uses two offsets.
    """
    z = np.asarray(features, dtype=np.float64)
    assert z.ndim == 2 and len(z) == len(weights) and np.isfinite(z).all()
    w = np.asarray(weights, dtype=np.float64)
    assert w.shape == (len(z),) and np.isfinite(w).all()
    center = np.where(np.ptp(z, axis=0) == 0, z[0], z.mean(0))
    x = unit(z - center)
    common = x.mean(0)
    event = (x.T @ w + kappa * common) / (w.sum() + kappa)
    background = (x.T @ (1-w) + kappa * common) / ((1-w).sum() + kappa)
    score = np.zeros(len(x))
    zero = not np.any(x) or w.sum() == 0 or (1-w).sum() == 0
    if not zero:
        score = np.log(np.maximum((1+x @ event)/2, 1e-12)
                       / np.maximum((1+x @ background)/2, 1e-12))
    stats = dict(fitting='all_video', prototype_pairs=1, event=event,
                 background=background, positive_weight=float(w.sum()),
                 negative_weight=float((1-w).sum()), zero_contrast=bool(zero),
                 self_inclusion=True)
    return score, stats


def evidence(features, q0, kappa=KAPPA):
    return evidence_from_membership(features, membership(q0), kappa)


def one_step(features, q0, *, kappa=KAPPA, beta=BETA):
    with threadpool_limits(limits=1, user_api='blas'):
        local, prototypes = evidence(features, q0, kappa)
        score = beta * inclusive_scores(local)
        posterior = update(q0, score)
    return posterior, dict(L=local, S=score, prototypes=prototypes,
                          kappa=kappa, beta=beta, rounds=1,
                          network_parameter_updates=0)
