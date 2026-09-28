"""P3 prediction structure: syntax validity and box geometry are distinct."""
import torch
from vg_tta.desta3d_v2_tta_pilot import validate_prediction as prior_structure


def validate_prediction(p, nframes):
    # Preserve shape, finite, support and interval checks. The old final clause
    # incorrectly required all boxes to have positive area when syntax was valid.
    prior_structure(dict(p, format_ok=False), nframes)
    valid = p['geometry_valid'].bool()
    boxes = p['boxes_cxcywh']
    assert torch.equal(valid, (boxes[:, 2:] > 0).all(-1))
    assert torch.equal(boxes[~valid], torch.zeros_like(boxes[~valid]))
    if p['format_ok']:
        assert p['interval'] is not None and p['positions']
    return True
