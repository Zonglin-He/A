"""Only role membership is new; keep predecessor's ridge/statistic equations."""
import numpy as np
from scripts.tastvg_pr_accessibility_math_v1 import (
    EPS, SEED, DRAWS, ROLES, source_weights, fit_ridge, predict, interval_pr,
    analytic_t, pack, regression, decision,
)

FITS = ['all', 'role']

def training_indices(anchor, winner, count=32):
    """Two equal roles even for no-op. No ranking, labels or deduplication."""
    assert isinstance(anchor, (int, np.integer)) and isinstance(winner, (int, np.integer))
    assert 0 <= anchor < count and 0 <= winner < count
    return np.array([anchor, winner], dtype=int)

def role_values(values, anchor, winner):
    return dict(P_A=float(values['P'][anchor]), R_A=float(values['R'][anchor]),
                P_W=float(values['P'][winner]), R_W=float(values['R'][winner]))

def analytic_readout(z):
    a=float(analytic_t(z['P_A'],z['R_A']));w=float(analytic_t(z['P_W'],z['R_W']))
    return dict(T_A=a,T_W=w,delta_T=w-a)
