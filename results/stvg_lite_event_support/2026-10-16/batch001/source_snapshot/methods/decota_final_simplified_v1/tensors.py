"""Small tensor utilities; no metrics, labels, or experiment selection."""

import hashlib
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


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ParameterState:
    def state(self):
        return {n: p.detach().clone() for n, p in self.named}

    def restore(self, state):
        with torch.no_grad():
            for name, param in self.named:
                param.copy_(state[name].to(param))
                param.grad = None
