"""One label-free confidence-scale diagnostic, not a trained new OPD method."""
import numpy as np


def absolute_confidence_scores(unprotected_reward,change,confidence,frame_ids,interval,change_weight=.25):
    active=(frame_ids>=interval[0])&(frame_ids<interval[1])
    if not active.any():raise ValueError('Empty fixed temporal support')
    # Evidence reliability must not disappear when the support score divided
    # by its confidence sum. This remains a heuristic exposure weight, NOT a
    # calibrated probability. No new coefficient or target label is fitted.
    exposure=float(np.mean(confidence[:,active]))
    return exposure*np.asarray(unprotected_reward)-change_weight*np.asarray(change),exposure
