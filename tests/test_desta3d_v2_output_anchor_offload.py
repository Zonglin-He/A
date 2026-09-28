import gc,weakref
import torch
from vg_tta.desta3d_v2_output_anchor_offload_v3 import activation_offload


def test_saved_tensor_detach_preserves_values_gradients_and_release():
    m=torch.nn.Linear(8,8).requires_grad_(False)
    x=torch.randn(2,8,requires_grad=True)
    y=m(x).sigmoid().square().sum(); y.backward();ref=x.grad.clone()
    x.grad=None
    with activation_offload(m) as info:
        z=m(x).sigmoid();w=weakref.ref(z);y=z.square().sum()
    y.backward();assert torch.equal(x.grad,ref)
    del y,z;gc.collect();assert w() is None
    assert info['tensors_offloaded']==0  # CPU checks hook semantics, not CUDA transfers.
