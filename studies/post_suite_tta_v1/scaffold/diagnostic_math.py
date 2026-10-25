"""Label-free numeric diagnostics for later saved-logit inspection.

These helpers do not implement a baseline or change its final readout.
"""
import math


def _row(values):
    values = tuple(float(x) for x in values)
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Expected a nonempty finite row on valid support only")
    return values


def probabilities(values):
    values = _row(values)
    m = max(values)
    e = tuple(math.exp(x - m) for x in values)
    z = sum(e)
    return tuple(x / z for x in e)


def compare_logits(before, after):
    before, after = _row(before), _row(after)
    if len(before) != len(after):
        raise ValueError("Native support changed")
    mb, ma = sum(before) / len(before), sum(after) / len(after)
    p, q = probabilities(before), probabilities(after)
    return {
        "raw_max_abs_delta": max(abs(b - a) for b, a in zip(before, after)),
        "centered_max_abs_delta": max(abs((b - mb) - (a - ma)) for b, a in zip(before, after)),
        "probability_max_abs_delta": max(abs(b - a) for b, a in zip(p, q)),
        "argmax_changed": before.index(max(before)) != after.index(max(after)),
    }


def legal_joint_margin(start_logits, end_logits):
    """Unnormalized start+end top-two score gap on i<=j, for diagnosis only."""
    starts, ends = _row(start_logits), _row(end_logits)
    if len(starts) != len(ends):
        raise ValueError("Endpoint support mismatch")
    candidates = sorted(
        ((starts[i] + ends[j], i, j) for i in range(len(starts)) for j in range(i, len(ends))),
        key=lambda x: (-x[0], x[1], x[2]),
    )
    top = candidates[0]
    return {"best_indices": [top[1], top[2]], "top_score": top[0],
            "runner_up_score": candidates[1][0] if len(candidates) > 1 else None,
            "margin": top[0] - candidates[1][0] if len(candidates) > 1 else None,
            "scope": "diagnostic score gap; original native/jointMAP implementations remain authoritative"}
