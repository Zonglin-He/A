"""Cached endpoint refinement continuation; isolated from maintained inference.

Use the last layer as anchor. Continue only log-probability changes that agree
between the final two refinement steps, bounded by the smaller observed change.
No labels, duration switch, fitted weights, posterior pruning or new forward.
"""
import numpy as np
from scipy.special import logsumexp
from .temporal import probabilities, mass_table, final_mbr
from .temporal_readout import BASE as OLD_BASE, tables as old_tables

BASE = {'mode': 'global6'}
MODES = {'source', 'global6', 'contrast', 'consistent', 'reverse'}

def validate(p):
    if set(p) != {'mode'} or p['mode'] not in MODES:
        raise ValueError('Invalid refinement mode')

def correction(logits):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 4 or z.shape[0] != 6 or z.shape[1] != 1 or z.shape[-1] != 2 or not np.isfinite(z).all():
        raise ValueError('Expected six finite cached endpoint layers')
    logp = z[:, 0] - logsumexp(z[:, 0], axis=1, keepdims=True)
    previous, latest = logp[-2] - logp[-3], logp[-1] - logp[-2]
    same = previous * latest > 0
    bounded = np.where(same, np.sign(latest) * np.minimum(abs(previous), abs(latest)), 0.)
    return logp[-1], latest, bounded

def tables(records, logits, ids, p):
    validate(p)
    if p['mode'] in {'source', 'global6'}:
        return old_tables(records, logits, ids, OLD_BASE)
    grids = probabilities(records, [z[-1] for z in logits])
    for grid, z in zip(grids, logits):
        final, latest, bounded = correction(z)
        delta = latest if p['mode'] == 'contrast' else bounded
        if p['mode'] == 'reverse':
            delta = -delta
        final = final + delta
        i, j = grid['ij']
        scores = final[i, 0] + final[j, 1]
        grid['p'] = np.exp(scores - logsumexp(scores))
    actual, q = mass_table(grids)
    if actual != list(ids):
        raise ValueError('Changed physical grid')
    return q

def combine(a, ids, p):
    validate(p)
    q = np.asarray(a, dtype=np.float64)
    if p['mode'] == 'source':
        q = q[-1]
    elif p['mode'] == 'global6':
        q = q.mean(0)
    if q.shape != (len(ids), len(ids)) or not np.isfinite(q).all() or not (q >= 0).all() or abs(q.sum() - 1) > 1e-12 or q[np.tril_indices(len(ids))].any():
        raise ValueError('Invalid posterior')
    return q

def decode(ids, q, p):
    validate(p)
    return final_mbr(ids, q)
