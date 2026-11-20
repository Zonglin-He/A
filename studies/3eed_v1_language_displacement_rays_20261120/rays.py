"""Half-line readout and its matched controls. No label inputs."""
import hashlib
import numpy as np

ARMS = ('v1', 'frozen_joint', 'seed', 'origin', 'ray', 'reverse', 'mismatch')


def key(*tokens):
    return hashlib.sha256('|'.join(map(str, ('native-ray-v1', *tokens))).encode()).hexdigest()


def selection(stage, platform, size):
    return sorted(range(size), key=lambda i: key(stage, platform, i))[:64]


def derange(n, rowkey):
    rng = np.random.default_rng(int(rowkey[:16], 16))
    original = np.arange(n)
    while True:
        perm = rng.permutation(n)
        if np.all(perm != original):
            return perm


def energies(centers, orig, delta, permutation):
    """Return origin, directed, reversed and wrong-pair MSEs, in squared metres."""
    origin, v, x = (np.asarray(arr, np.float64) for arr in (orig, delta, centers))
    r = x[:, None, :] - origin
    weights = np.einsum('ki,ki->k', v, v)
    total = weights.sum()
    if total == 0:
        return np.zeros((len(x), 4), np.float64)

    def directed(directions):
        cross = np.cross(directions[None], r)
        dot = np.einsum('ki,nki->nk', directions, r)
        line = np.square(cross).sum((1, 2))
        return [(line + np.square(np.minimum(dot, 0.)).sum(1)) / total,
                (line + np.square(np.maximum(dot, 0.)).sum(1)) / total]

    spatial = np.einsum('k,nki,nki->n', weights, r, r) / total
    forward, reverse = directed(v)
    shuffled, _ = directed(v[permutation])
    result = np.column_stack((spatial, forward, reverse, shuffled))
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite energy')
    return result


def decode(boxes, joint, v1, seed_logits, origins, ends, permutation, valid):
    index = np.arange(len(joint))
    frozen = int(np.argmax(joint))
    def best(cost):
        return int(np.lexsort((index, -joint, cost))[0])
    delta = ends.astype(np.float64) - origins.astype(np.float64)
    value = energies(boxes[:, :3], origins, delta, permutation)
    picks = [best(value[:, j]) for j in range(4)] if valid and delta.any() else [frozen] * 4
    return np.array([v1, frozen, best(-seed_logits), *picks], dtype=np.int16), value
