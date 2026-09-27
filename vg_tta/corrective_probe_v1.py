"""Small GT-free diagnostics; not a production method."""
from __future__ import annotations

import math

import torch
from torchvision.ops import box_convert, nms


def post_nms_pool(boxes, scores, nms_iou=.5, maximum=5):
    boxes = torch.as_tensor(boxes).float().cpu()
    scores = torch.as_tensor(scores).float().cpu()
    if boxes.shape != (len(scores), 4) or not bool(torch.isfinite(boxes).all() and torch.isfinite(scores).all()):
        raise ValueError('finite N x 4 boxes and N scores required')
    if maximum < 1:
        raise ValueError('positive candidate budget required')
    xy = box_convert(boxes, 'cxcywh', 'xyxy').clamp(0, 1)
    keep = nms(xy, scores, float(nms_iou))[:maximum]
    return dict(indices=keep.tolist(),
                boxes=box_convert(xy[keep], 'xyxy', 'cxcywh').tolist(),
                scores=scores[keep].tolist(),
                valid=(xy[keep, 2:] > xy[keep, :2]).all(-1).tolist())


def choose_loss_iterate(losses):
    """Earliest minimum among predeclared iterates, including the native no-op."""
    if not losses or any(not math.isfinite(float(x)) for x in losses):
        raise ValueError('nonempty finite losses required')
    return min(range(len(losses)), key=lambda i: (float(losses[i]), i))
