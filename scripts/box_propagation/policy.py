"""Native-box and component banks use the same immutable v0 policy functions.

There is deliberately no second optimizer or sampler implementation here.
Runner fingerprints must include scripts/propagation_v0/policy.py as a frozen
source dependency. Only candidate construction and fixed bank evidence differ;
policy/config/seed/acceptance/readout remain the matched v0 implementation.
"""
from scripts.propagation_v0.policy import (
    ARMS, FEATURE_NAMES, MODES, evidence_dp, fit, fixed_propagation,
    gumbel_noise, initial_diagnostics, rollout)


IMPLEMENTATION_MODULE = 'scripts.propagation_v0.policy'
FROZEN_IMPLEMENTATION_SHA256 = '91e2a891fc211edb27ae532405c46cab381207f48f26e9fd1ff8ff235b679b4a'

__all__ = ['ARMS','FEATURE_NAMES','MODES','evidence_dp','fit','fixed_propagation',
           'gumbel_noise','initial_diagnostics','rollout','IMPLEMENTATION_MODULE',
           'FROZEN_IMPLEMENTATION_SHA256']
