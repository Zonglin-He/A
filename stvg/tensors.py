"""Small tensor utilities; no metrics, labels, or experiment selection."""

import torch


def detached(value, device=None):
    if torch.is_tensor(value):
        result = value.detach().clone()
        return result.to(device) if device is not None else result
    if isinstance(value, dict):
        return {k: detached(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [detached(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(detached(v, device) for v in value)
    return value


def floating32(value):
    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {k: floating32(v) for k, v in value.items()}
    if isinstance(value, list):
        return [floating32(v) for v in value]
    if isinstance(value, tuple):
        return tuple(floating32(v) for v in value)
    return value

