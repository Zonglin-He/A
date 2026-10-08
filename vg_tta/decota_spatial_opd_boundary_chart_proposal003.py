"""DRAFT numerical chart proposal; not authorized or installed in any runner.

Only a sigmoid-rounded endpoint would use the finite native pre-sigmoid logit.
Interior components retain the exact original logit(box) computation. This
changes the locked endpoint-failure rule and needs a protocol authorization.
It never clamps a box or substitutes an arbitrary epsilon.
"""
import torch


def proposed_coordinates(boxes, native_logits):
    assert boxes.shape == native_logits.shape
    assert boxes.dtype == native_logits.dtype and boxes.device == native_logits.device
    assert torch.isfinite(boxes).all() and torch.isfinite(native_logits).all()
    assert ((boxes >= 0) & (boxes <= 1)).all()
    assert torch.equal(native_logits.sigmoid(), boxes), 'Must use the same native decoder outputs.'
    boundary = (boxes == 0) | (boxes == 1)
    if not boundary.any():
        return torch.logit(boxes), boundary
    mean = torch.empty_like(boxes)
    interior = ~boundary
    mean[interior] = torch.logit(boxes[interior])
    mean[boundary] = native_logits[boundary]
    assert torch.isfinite(mean).all()
    return mean, boundary
