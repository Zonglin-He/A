"""Observe task derivatives at real postcast merger values, never update weights."""
from contextlib import contextmanager
import torch


@contextmanager
def postcast_leaf(merger):
    """Register AFTER branch_injection; do not change values, dtype or support."""
    state = {'calls': 0}
    def hook(module, args, output):
        state['calls'] += 1
        assert state['calls'] == 1, 'exactly one own visual prefill required'
        assert output.is_floating_point() and not output.is_inference()
        leaf = output.detach().requires_grad_(True)
        assert leaf.is_leaf and torch.equal(leaf, output)
        assert leaf.dtype == output.dtype and leaf.shape == output.shape
        state['leaf'] = leaf
        state['values_exact'] = True
        return leaf
    handle = merger.register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()
    assert state['calls'] == 1


def transport_summary(gradient, endpoint_difference):
    g = gradient.detach().cpu().reshape(-1).double()
    pre = endpoint_difference['precast_delta'].reshape(-1).double()
    sparse = endpoint_difference['sparse_postcast']; idx = sparse['indices'].long()
    post = sparse['after'].float().double() - sparse['before'].float().double()
    assert g.shape == pre.shape and torch.isfinite(g).all()
    return {'elements': g.numel(), 'gradient_dtype': str(gradient.dtype),
            'gradient_L2': float(g.norm()), 'gradient_nonzero': int((g != 0).sum()),
            'dot_precast_delta': float(g.dot(pre)), 'dot_postcast_delta': float(g[idx].dot(post)),
            'postcast_changed_positions': idx.numel()}
