"""Remove interval-extent reward that is present even without localisation."""
import numpy as np
from research.joint_event_v1.core import interval_scores


def excess_scores(mask,overlap,width,lengths):
    score,logs=interval_scores(mask,overlap,width,lengths)
    occupancy=(width[:,None]*mask).sum(0)/width.sum()
    uniform=np.broadcast_to(occupancy,mask.shape)
    _,null=interval_scores(uniform,overlap,width,lengths)
    excess=logs-null
    return excess.mean(1),excess
