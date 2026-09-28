import pytest
import torch
from vg_tta.desta3d_v3_large_mask_readback_v2 import assert_actual_cast

def test_native_recorded_cast_dtype_shape():
    x=torch.linspace(-2,2,24).reshape(1,2,2,2,3)
    assert_actual_cast(x.bfloat16().reshape(8,3),x)
    with pytest.raises(AssertionError):assert_actual_cast((x+1).bfloat16().reshape(8,3),x)

def test_verification_never_mutates_or_retains_grad():
    x=torch.randn(1,2,2,2,3,requires_grad=True);before=x.detach().clone();actual=x.detach().bfloat16()
    assert_actual_cast(actual,x)
    assert torch.equal(x,before) and x.grad is None
