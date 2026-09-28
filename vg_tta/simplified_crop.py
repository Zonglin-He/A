"""User-selected crop: all sampled frozen boxes, no temporal gate or expansion."""
import torch
from .geometric_spatial_teacher import tube_crop_rect


def full_video_crop_rect(boxes: torch.Tensor, image_hw):
    """Return an image-bounded, min-32px union crop over every sampled frame.

    No predicted interval is accepted. Expansion is fixed to 1.0. Existing
    clipping, integer rounding, and minimum-size safeguards are preserved.
    """
    return tube_crop_rect(boxes, list(range(len(boxes))), image_hw, expansion=1.0)
