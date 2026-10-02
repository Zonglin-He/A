"""Query-scoped readout and conservative two-view temporal comparison.

These distributions are geometry compatibility surrogates. No correction here
changes the persistent Uniform A state or claims calibrated expert correctness.
"""
import numpy as np


def select(reward):
    if reward is None:
        return dict(selected=0, reason='empty_evidence', top_ties=[])
    r = np.asarray(reward, np.float64)
    assert r.ndim == 1 and np.isfinite(r).all()
    tied = np.flatnonzero(r == r.max()).tolist()
    # User requested central fallback on ties, even when central is not a top tie.
    return dict(selected=tied[0] if len(tied) == 1 else 0,
                reason='unique_best' if len(tied) == 1 else 'tie_keep_center', top_ties=tied)


def temporal(candidates, evidence):
    from vg_tta.tastvg_temporal_qualification_v1 import critic_scores
    assert len(evidence) == 2 and 0 < len(candidates) <= 8
    scores = np.array([critic_scores([c['physical_interval'] for c in candidates],
        e['proposals'], e['proposal_confidence']) for e in evidence])
    differences = scores - scores[:, :1]
    worst = differences.min(0)
    assert worst[0] == 0
    tied = np.flatnonzero(worst == worst.max()).tolist()
    k = tied[0] if len(tied) == 1 and worst[tied[0]] > 0 else 0
    assert k == 0 or (differences[:, k] > 0).all()
    return dict(scores=scores.tolist(), differences=differences.tolist(),
                worst=worst.tolist(), selected=k, top_ties=tied,
                native_fixed=True, coordinates='original_video_frames')


def uniform_second(length):
    assert length >= 5
    positions = np.rint((np.arange(5)+.5)/5*(length-1)).astype(int).tolist()
    assert len(set(positions)) == 5
    return positions
