"""Experimental retention and update-scope operators, not registered production."""
import torch
from methods.decota_final_simplified_v1.tensors import detached
from methods.decota_final_simplified_v1.optim import fit_spatial
from vg_tta.spatial_online_state_v1 import QUERY


def consolidate(initial, fitted, alpha):
    if alpha not in (0., 1/16, 1.):
        raise ValueError('Only the three prelocked retention rules are permitted')
    result = detached(initial)
    for name in result:
        if name == QUERY:
            result[name].zero_()
        elif alpha == 1:
            result[name] = detached(fitted[name]).to(result[name])
        elif alpha:
            result[name] = initial[name] + alpha * (fitted[name].to(initial[name])-initial[name])
    return result


class ScopedReplay:
    """Restrict optimizer inputs but retain FULL parameter states and restore."""
    def __init__(self, replay, scope):
        self.replay = replay
        self.initial = replay.initial
        self.named = [(n,p) for n,p in replay.named if scope == 'full'
                      or (scope == 'query' and n == QUERY) or (scope == 'ln' and n != QUERY)]

    def values(self): return self.replay.values()
    def restore(self, state): return self.replay.restore(state)
    def state(self): return self.replay.state()


def scoped_fit(replay, zero, anchors, config, scope='full', trace=False):
    if scope not in ('full', 'query', 'ln'): raise ValueError(scope)
    proxy = ScopedReplay(replay, scope)
    selected = {n for n,_ in proxy.named}
    original = [(p,p.requires_grad) for _,p in replay.named]
    try:
        for n,p in replay.named:
            p.requires_grad_(n in selected)
            p.grad = None
        result = fit_spatial(proxy, zero, anchors, config, trace=trace)
        for n,p in replay.named:
            if n not in selected:
                assert torch.equal(result['state'][n], replay.initial[n])
        result['scope'] = scope
        result['trainable_parameters'] = sum(p.numel() for _,p in proxy.named)
        return result
    finally:
        for p,flag in original:
            p.requires_grad_(flag)
            p.grad = None
