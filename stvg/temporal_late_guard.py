"""Constrain ensemble MBR to the boundary range supported by late readouts.

No posterior editing or learned/GT duration rule. The final-layer Source decision
is always feasible. The early-bracket arm is a diagnostic countercontrol.
"""
import numpy as np
from .temporal import final_mbr
from .temporal_readout import BASE as OLD_BASE, tables as old_tables
BASE = {'mode': 'global6'}
MODES = {'global6', 'late_bracket', 'early_bracket', 'last3_mean'}

def validate(p):
    if set(p) != {'mode'} or p['mode'] not in MODES:
        raise ValueError('Invalid late-guard mode')

def tables(records, logits, ids, p):
    validate(p)
    return old_tables(records, logits, ids, OLD_BASE)

def combine(a, ids, p):
    validate(p)
    a = np.asarray(a)
    if a.shape != (6, len(ids), len(ids)):
        raise ValueError('Expected original six-layer interval tables')
    return a

def decode(ids, a, p):
    validate(p)
    if p['mode'] == 'last3_mean':
        return final_mbr(ids, np.mean(a[-3:], axis=0))
    result = final_mbr(ids, np.mean(a, axis=0))
    if p['mode'] == 'global6':
        return result
    active = a[-3:] if p['mode'] == 'late_bracket' else a[:3]
    spans = np.asarray([final_mbr(ids, q)['indices'] for q in active])
    lower, upper = spans.min(axis=0), spans.max(axis=0)
    i, j = np.indices(result['utility_grid'].shape)
    legal = (i < j) & (i >= lower[0]) & (i <= upper[0]) & (j >= lower[1]) & (j <= upper[1])
    u = np.where(legal, result['utility_grid'], -np.inf)
    if not np.isfinite(u).any():
        raise ValueError('Empty deep-readout feasible set')
    k = int(np.argmax(u));s, e = divmod(k, len(ids))
    return dict(interval=[int(ids[s]), int(ids[e])+1], indices=[s,e], utility=float(u[s,e]), utility_grid=u)
