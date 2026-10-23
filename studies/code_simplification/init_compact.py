"""The actual eight-singleton robust initialization, without unused candidates."""
from itertools import product
import numpy as np
from stvg.consensus import encode, apply
from stvg.predictor import output_writeback


def initialize(source, positions, expert_boxes, calibration, gamma):
    residuals = encode(expert_boxes) - encode(source)[positions]
    assert residuals.shape == (8, 4)
    scale = np.asarray(calibration['scale'], dtype=np.float64)
    cutoff = calibration['truncation']
    proposals = []
    # Preserve the original enumeration, objective summation and tie order.
    for assignment in product((-1, 0), repeat=8):
        active = [r for r, choice in zip(residuals, assignment) if choice == 0]
        delta = np.mean(active, axis=0) if active else np.zeros(4)
        costs = np.square((residuals - delta) / scale).sum(1)
        selected = tuple(0 if cost < cutoff else -1 for cost in costs)
        objective = float(sum(min(cutoff, float(cost)) for cost in costs))
        proposals.append((objective, selected, assignment, delta))
    objective, selected, _, delta = min(proposals, key=lambda x: (x[0], x[1], x[2]))
    result = output_writeback(source, apply(source, delta), gamma)
    return result, dict(delta=delta, selected=selected, objective=objective)


class DirectionalMatches:
    """Cache the exact original directed affinity, never a transposed surrogate."""
    def __init__(self, affinity):
        self.original = affinity
        self.cache = {}
        self.inputs = None
        self.checked = set()
        self.requests = self.hits = 0

    def __call__(self, normalized, valid, target, source):
        identity = (normalized.data_ptr(), valid.data_ptr())
        if identity not in self.checked:
            if self.inputs is None:
                self.inputs = normalized, valid
            else:
                assert np.array_equal(normalized.numpy(), self.inputs[0].numpy())
                assert np.array_equal(valid.numpy(), self.inputs[1].numpy())
            self.checked.add(identity)
        self.requests += 1
        key = int(target), int(source)
        if key not in self.cache:
            self.cache[key] = self.original(normalized, valid, target, source)
        else:
            self.hits += 1
        return self.cache[key]
