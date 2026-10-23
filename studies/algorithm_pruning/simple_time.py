"""One video-wide prototype pair; no cross-offset fitting, no parameter search."""
import numpy as np
from threadpoolctl import threadpool_limits
from stvg.prototype import membership, unit, inclusive_scores, update


def one_step(features, prior):
    with threadpool_limits(limits=1, user_api='blas'):
        z = np.asarray(features, dtype=np.float64)
        w = membership(prior)
        center = np.where(np.ptp(z, axis=0) == 0, z[0], z.mean(0))
        x = unit(z - center)
        common = x.mean(0)
        event = (x.T @ w + 64 * common) / (w.sum() + 64)
        background = (x.T @ (1-w) + 64 * common) / ((1-w).sum() + 64)
        evidence = np.zeros(len(x))
        if np.any(x) and w.sum() > 0 and (1-w).sum() > 0:
            evidence = np.log(np.maximum((1+x@event)/2, 1e-12)
                              / np.maximum((1+x@background)/2, 1e-12))
        return update(prior, inclusive_scores(evidence))
