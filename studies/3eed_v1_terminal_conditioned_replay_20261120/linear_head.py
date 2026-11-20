"""The 288-coefficient TCR residual and the source-only convex ranking objective."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def corrected_scores(base, features, weight):
    return np.asarray(base, dtype=np.float64) + np.asarray(features, dtype=np.float64) @ weight


def comparison_weights(ious):
    ious = np.asarray(ious, dtype=np.float64)
    if ious.ndim != 2 or not np.all((ious >= 0) & (ious <= 1)):
        raise ValueError('Expected one finite IoU per source row and candidate.')
    pairs = np.zeros((*ious.shape, ious.shape[1]), dtype=np.float64)
    eligible = 0
    for threshold in (.25, .50):
        positive = ious > threshold
        valid = positive[:, :, None] & ~positive[:, None, :]
        count = valid.sum(axis=(1, 2))
        eligible += np.count_nonzero(count)
        pairs += valid / np.maximum(count, 1)[:, None, None]
    if not eligible:
        raise ValueError('There are no source row/thresholds with a good and a bad candidate.')
    return pairs / eligible


def loss_gradient(weight, base, features, pair_weights, ridge):
    if ridge <= 0 or not np.isfinite(ridge):
        raise ValueError('The regularization coefficient must be finite and positive.')
    scores = corrected_scores(base, features, weight)
    margin = scores[:, :, None] - scores[:, None, :]
    loss = np.sum(pair_weights*np.logaddexp(0, -margin)) + .5*ridge*(weight@weight)
    pair_gradient = -pair_weights*expit(-margin)
    score_gradient = pair_gradient.sum(axis=2) - pair_gradient.sum(axis=1)
    gradient = np.einsum('ni,nid->d', score_gradient, features) + ridge*weight
    return float(loss), gradient


def fit_source_weight(base, features, ious, ridge, probe_decision):
    if not probe_decision['train_per_protocol']:
        raise ValueError('The sealed 192-row probe did not qualify this head for source fitting.')
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 3 or features.shape[1:] != (16, 288) or features.shape[:2] != np.shape(base):
        raise ValueError('Expected frozen scores and 16 by 288 features for each source row.')
    pairs = comparison_weights(ious)
    result = minimize(loss_gradient, np.zeros(288), args=(base, features, pairs, ridge),
                      method='L-BFGS-B', jac=True, options=dict(ftol=1e-13, gtol=1e-9, maxiter=500))
    if not result.success:
        raise RuntimeError(result.message)
    return result.x
