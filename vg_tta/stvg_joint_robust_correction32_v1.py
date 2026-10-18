"""Exact finite truncated-LS association; no labels, models or target calibration.

Each frame has at most three retained expert candidates. Enumerating candidate /
outlier assignments and their least-squares means gives a GLOBAL optimum: the
minimum over assignments and the continuous delta can be interchanged. Candidate
scores never weight the loss, and proximity to the zero/Student state is not a
selection prior. A deterministic lexicographic tie break is not a probability.
"""
import itertools
import numpy as np
from vg_tta.stvg_error_state_correction_v1 import encode, decode


def estimate(pools, scale, truncation, diagnostics=True):
    scale = np.asarray(scale, dtype=np.float64)
    assert scale.shape == (4,) and np.isfinite(scale).all() and (scale > 0).all()
    assert np.isfinite(truncation) and truncation > 0
    pools = [np.asarray(p, dtype=np.float64).reshape(-1, 4) for p in pools]
    assert all(np.isfinite(p).all() and len(p) <= 3 for p in pools)
    n = len(pools)
    if not n or not any(len(p) for p in pools):
        return dict(delta=np.zeros(4), objective=float(n * truncation), selected=[-1]*n,
                    support_frames=0, rejected_frames=n, assignments_evaluated=1,
                    competitor_gap=None, competitor_delta_distance=None,
                    leave_one_delta_distances=[])
    # A rejected frame contributes exactly truncation. No frame gets extra votes
    # because it has more boxes. The all-rejected assignment has canonical delta0.
    proposals = []
    for assignment in itertools.product(*[range(-1, len(p)) for p in pools]):
        selected = [p[k] for p, k in zip(pools, assignment) if k >= 0]
        delta = np.mean(selected, axis=0) if selected else np.zeros(4)
        distances = [np.sum(((p-delta)/scale)**2, axis=1) for p in pools]
        choices = [int(d.argmin()) if len(d) and d.min() < truncation else -1 for d in distances]
        cost = float(sum(min(truncation, float(d.min())) if len(d) else truncation for d in distances))
        # Discrete IDs, not ||delta||, break exact mathematical ties.
        proposals.append((cost, tuple(choices), tuple(assignment), delta))
    proposals.sort(key=lambda a: (a[0], a[1], a[2]))
    cost, choice, _, delta = proposals[0]
    active = [p[k] for p, k in zip(pools, choice) if k >= 0]
    assert active and np.max(np.abs(delta-np.mean(active, axis=0))) < 1e-10
    alternate = next((p for p in proposals[1:] if np.linalg.norm((p[3]-delta)/scale) > 1e-9), None)
    loo = []
    if diagnostics:
        for i in range(n):
            sub = estimate(pools[:i]+pools[i+1:], scale, truncation, diagnostics=False)
            loo.append(float(np.linalg.norm((sub['delta']-delta)/scale)))
    return dict(delta=delta, objective=cost, selected=list(choice),
                support_frames=len(active), rejected_frames=n-len(active),
                assignments_evaluated=len(proposals),
                competitor_gap=None if alternate is None else float(alternate[0]-cost),
                competitor_delta_distance=None if alternate is None else float(np.linalg.norm((alternate[3]-delta)/scale)),
                leave_one_delta_distances=loo)


def apply(boxes, delta):
    b = np.asarray(boxes, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    # Preserve empty/no-correction outputs byte-exactly (avoid exp(log(w)) roundoff).
    return b.copy() if np.array_equal(delta, np.zeros(4)) else decode(encode(b)+delta)


def contract():
    scale = np.ones(4); tau = 1.
    good = np.array([7., -4., .2, -.1])  # Deliberately FAR from the Student.
    pools = [np.stack((good, good+20)), np.stack((good, good-13)),
             np.stack((good, good+40)), np.array([[90., 80., 10., 10.]])]
    out = estimate(pools, scale, tau)
    assert np.array_equal(out['delta'], good) and out['selected'] == [0, 0, 0, -1]
    assert out['support_frames'] == 3 and out['objective'] == 1.
    # At a huge cutoff this reduces to the mean, including the one-frame case.
    rs = np.array([[.1, .2, .3, .4], [.3, .1, .2, .5]])
    mean = estimate([r[None] for r in rs], scale, 1e8)
    assert np.allclose(mean['delta'], rs.mean(0), atol=1e-14)
    empty = estimate([], scale, tau)
    assert np.array_equal(empty['delta'], np.zeros(4))
    singleton = estimate([good[None]], scale, tau)
    assert np.array_equal(singleton['delta'], good)
    return dict(status='pass',far_from_Student_recovery=True,one_vote_per_frame=True,
                outlier_abstention=True,large_cutoff_mean=True,empty_exact_noop=True,
                singleton_not_silently_rejected=True,GT_or_network_in_core=False)
