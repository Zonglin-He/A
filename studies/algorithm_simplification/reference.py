"""Readable temporal reference; no model, annotations, file access or training.

This does not replace the frozen production path. See EXACT_MATH.json for the
bound adapter audit and REFERENCE_CHECK.json for this readable implementation.
"""
import numpy as np


def hull_posterior(offsets):
    """Combine two disjoint native grids without enumerating interval pairs.

    Each offset supplies physical `ids`, legal local index pairs `ij`, and their
    probabilities `p`. Every hull endpoint belongs to exactly one offset.
    """
    assert len(offsets) == 2
    assert not set(offsets[0]['ids']).intersection(offsets[1]['ids'])
    frame_ids = sorted(set(offsets[0]['ids']).union(offsets[1]['ids']))
    index = {frame: i for i, frame in enumerate(frame_ids)}
    size = len(frame_ids)
    mass, contained, end_before, start_after = [], [], [], []

    for view in offsets:
        local = np.array([index[frame] for frame in view['ids']])
        starts, ends = view['ij']
        q = np.zeros((size, size), dtype=np.float64)
        q[local[starts], local[ends]] = view['p']

        # For a proposed hull [a,b], count strict inequalities because its
        # endpoints belong to the other view, whose physical grid is disjoint.
        inside = q[::-1].cumsum(0)[::-1].cumsum(1)
        inner_mass = np.zeros_like(q)
        inner_mass[:-1, 1:] = inside[1:, :-1]
        earlier_end = np.zeros_like(q)
        earlier_end[:, 1:] = q.cumsum(1)[:, :-1]
        later_start = np.zeros_like(q)
        later_start[:-1] = q[::-1].cumsum(0)[::-1][1:]

        mass.append(q)
        contained.append(inner_mass)
        end_before.append(earlier_end)
        start_after.append(later_start)

    same_owner_0 = mass[0] * contained[1]
    same_owner_1 = mass[1] * contained[0]
    start_0_end_1 = end_before[0] * start_after[1]
    start_1_end_0 = end_before[1] * start_after[0]
    result = same_owner_0 + same_owner_1 + start_0_end_1 + start_1_end_0
    return frame_ids, result


def unit_rows(vectors):
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 0)


def event_membership(prior):
    """w[t] = probability that a native interval contains sampled node t."""
    cumulative = prior.cumsum(0)[:, ::-1].cumsum(1)[:, ::-1]
    return np.diag(cumulative).copy()


def prototype_feedback(region_features, prior, kappa=64):
    """Estimate two vectors on the other offset and evaluate the held-out one."""
    features = np.asarray(region_features, dtype=np.float64)
    membership = event_membership(prior)
    parity = np.arange(len(features)) % 2
    feedback = np.zeros(len(features))

    for held_out in (0, 1):
        fitting = parity != held_out
        values = features[fitting]
        # Exact constant coordinates use their constant value, avoiding an
        # artificial tiny residual caused by floating-point mean reduction.
        center = np.where(np.ptp(values, axis=0) == 0, values[0], values.mean(0))
        train = unit_rows(values - center)
        test = unit_rows(features[~fitting] - center)
        positive = membership[fitting]
        negative = 1 - positive
        if not np.any(train) or positive.sum() == 0 or negative.sum() == 0:
            continue

        common_mean = train.mean(0)
        event = (train.T @ positive + kappa * common_mean) / (positive.sum() + kappa)
        background = (train.T @ negative + kappa * common_mean) / (negative.sum() + kappa)
        event_support = (1 + test @ event) / 2
        background_support = (1 + test @ background) / 2
        feedback[~fitting] = np.log(
            np.maximum(event_support, 1e-12) / np.maximum(background_support, 1e-12)
        )
    return feedback


def correct_once(prior, feedback):
    """Fixed q0 prior, inclusive frame sums, exactly one posterior correction."""
    prefix = np.r_[0., np.cumsum(feedback)]
    interval_support = prefix[1:][None, :] - prefix[:-1][:, None]
    legal = prior > 0
    if np.all(interval_support[legal] == 0):
        return prior.copy()
    log_weight = np.log(prior[legal]) + interval_support[legal]
    log_weight -= log_weight.max()
    result = np.zeros_like(prior)
    result[legal] = np.exp(log_weight)
    return result / result.sum()
