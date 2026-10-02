"""Finite-precision reconstruction and descriptive saved-write geometry."""
import torch


def delta(pre, post):
    assert list(pre) == list(post)
    return {key: post[key].double() - value.double() for key, value in pre.items()}


def at_origin(origin, writes):
    result = {}
    for key, value in origin.items():
        total = torch.zeros_like(value, dtype=torch.float64)
        for write in writes:
            total += write[key].double()
        result[key] = (value.double() + total).to(value.dtype)
    return result


def flat(state):
    return torch.cat([value.double().reshape(-1) for value in state.values()])


def geometry(writes):
    vectors = torch.stack([flat(write) for write in writes])
    norms = vectors.norm(dim=1)
    gram = vectors @ vectors.T
    cosines = [[float(gram[i, j] / (norms[i] * norms[j]))
                if norms[i] > 0 and norms[j] > 0 else None
                for j in range(len(writes))] for i in range(len(writes))]
    prefix = vectors.cumsum(0)
    cancellation = [float(prefix[i].norm() / norms[:i+1].sum())
                    if norms[:i+1].sum() > 0 else None for i in range(len(writes))]
    consecutive = [cosines[i-1][i] for i in range(1, len(writes))]
    return dict(write_norms=norms.tolist(), cosine_gram=cosines,
                consecutive_cosines=consecutive,
                prefix_cancellation=cancellation,
                prefix_norms=prefix.norm(dim=1).tolist(),
                norm_sum_prefix=norms.cumsum(0).tolist())
