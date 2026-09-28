"""Lossless CPU checkpoints for optimizer states (integer keys are semantic)."""
from __future__ import annotations

import copy
import torch


def cpu_clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_clone(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_clone(v) for v in value)
    if isinstance(value, list):
        return [cpu_clone(v) for v in value]
    return copy.deepcopy(value)


def validate_serialized_optimizer(state):
    ids = [i for group in state['param_groups'] for i in group['params']]
    if any(type(i) is not int for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('optimizer parameter IDs must be unique integers')
    if any(type(i) is not int or i not in ids for i in state['state']):
        raise ValueError('optimizer state keys do not map to integer parameter IDs')


def assert_nested_equal(a, b):
    if type(a) is not type(b):
        raise AssertionError((type(a), type(b)))
    if isinstance(a, torch.Tensor):
        if not torch.equal(a.detach().cpu(), b.detach().cpu()):
            raise AssertionError('checkpoint tensor mismatch')
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError('checkpoint dictionary keys mismatch')
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError('checkpoint sequence length mismatch')
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    elif a != b:
        raise AssertionError((a, b))


def restore_optimizer(optimizer, state):
    """Reject ambiguous legacy string IDs instead of silently dropping momentum."""
    validate_serialized_optimizer(state)
    optimizer.load_state_dict(cpu_clone(state))
    parameters = {p for group in optimizer.param_groups for p in group['params']}
    if any(not isinstance(p, torch.Tensor) or p not in parameters for p in optimizer.state):
        raise AssertionError('orphan optimizer state is not bound to a live Parameter')
    assert_nested_equal(cpu_clone(optimizer.state_dict()), cpu_clone(state))

