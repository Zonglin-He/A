"""Exact two-role assignment in shared camera coordinates."""
import numpy as np


def reference_unary(physical):
    """Use the inherited null score and normalize the strongest available donor."""
    physical = np.asarray(physical, dtype=float).copy()
    known = np.isfinite(physical)
    if not known.any():
        return np.r_[physical, 0.]
    physical[known] -= physical[known].max()
    return np.r_[physical, -2.]


def assign(subject, reference, left, sign, mode):
    """Return positions in their own role pools. The last reference position means unknown."""
    assert left.shape == (len(subject), len(reference))
    assert len(subject) and np.isfinite(subject).all() and np.isfinite(reference[-1])
    assert np.isfinite(left).all() and (left[:, -1] == 0.).all()
    assert sign in (-1, 1) and mode in ('independent', 'fixed_reference', 'joint')
    signed = sign * left
    if mode == 'independent':
        return int(subject.argmax()), int(reference.argmax())
    if mode == 'fixed_reference':
        reference_top = int(reference.argmax())
        return int((subject + (reference[reference_top] + signed[:, reference_top])).argmax()), reference_top
    fused = reference[None, :] + signed
    reference_top = fused.argmax(1)
    best_subject = int((subject + fused[np.arange(len(subject)), reference_top]).argmax())
    return best_subject, int(reference_top[best_subject])
