"""One frozen score definition and five source objectives."""
import torch
from torch.nn import functional as F
from common import ARMS
from fit182 import multi_positive, objective as original_objective


def loss(arm, scores, ious):
    good = ious > .25
    if arm == ARMS[0]:
        return original_objective(scores, ious, torch.zeros_like(good), False)
    if arm == ARMS[1]:
        return multi_positive(scores, good)
    if arm == ARMS[4]:
        return F.binary_cross_entropy_with_logits(scores, good.to(scores.dtype))
    if arm not in ARMS[2:4]:
        raise ValueError(arm)
    per_row = -(good * scores.log_softmax(dim=-1)).sum(dim=-1)
    if arm == ARMS[3]:
        per_row = per_row / good.sum(dim=-1).clamp_min(1)
    return per_row.mean()


def pair_only(scores, ious):
    good = ious > .25
    mask = good[:, :, None] & ~good[:, None, :]
    cost = F.softplus(scores[:, None, :] - scores[:, :, None])
    return ((cost * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)).mean()
