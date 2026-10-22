"""Recursive frozen tensor transfer used by the native Source."""
import torch

def transfer(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: transfer(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [transfer(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(transfer(v, device) for v in value)
    return value
