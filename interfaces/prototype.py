"""Dimension-free all-video prototype feedback; no model-specific head."""
from threadpoolctl import threadpool_limits
from stvg.prototype import KAPPA, BETA, evidence_from_membership


def feedback(features, temporal):
    if len(features) != len(temporal.times):
        raise ValueError('Explicit physical-time alignment required')
    with threadpool_limits(limits=1, user_api='blas'):
        local, prototypes = evidence_from_membership(features, temporal.soft_membership())
        posterior = temporal.revise(BETA * local)
    return posterior, dict(L=local, prototypes=prototypes,
                           kappa=KAPPA, beta=BETA, rounds=1)
