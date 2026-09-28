import pytest
import torch
from vg_tta.desta3d_v2_prediction_contract import validate_prediction


def test_valid_syntax_can_include_zero_geometry_without_exclusion():
    p = {'format_ok': True, 'positions': [0, 1], 'interval': [0, 1],
         'boxes_cxcywh': torch.tensor([[.5, .5, .2, .2], [0., 0., 0., 0.]]),
         'geometry_valid': torch.tensor([True, False])}
    assert validate_prediction(p, 2)
    with pytest.raises(AssertionError):
        validate_prediction(dict(p, geometry_valid=torch.tensor([True, True])), 2)


def test_absent_spatial_on_bad_format_is_retained():
    p = {'format_ok': False, 'positions': [], 'interval': None,
         'boxes_cxcywh': torch.empty(0, 4), 'geometry_valid': torch.empty(0, dtype=torch.bool)}
    assert validate_prediction(p, 32)
