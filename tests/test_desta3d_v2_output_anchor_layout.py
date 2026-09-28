import torch
from vg_tta.desta3d_v2_output_anchor_layout_v5 import aligned_bias,copy_layout,layout_storage_bytes


def test_aligned_bias_preserves_shape_values_and_transfer_layout():
    mask=torch.randn(1,1,6,3635)
    out=aligned_bias(mask)
    assert torch.equal(out,mask) and out.shape==mask.shape
    assert out.stride(1)%8==0 and out.stride(2)%8==0
    copied=copy_layout(out,'cpu')
    assert copied.stride()==out.stride() and torch.equal(copied,out)
    assert layout_storage_bytes(copied)==copied.untyped_storage().nbytes()
    assert layout_storage_bytes(mask.expand(2,3,-1,-1)) is None


def test_aligned_bias_cpu_attention_value_and_gradient_equivalence():
    torch.manual_seed(1)
    q=torch.randn(1,2,6,8,requires_grad=True)
    k=torch.randn(1,2,13,8,requires_grad=True)
    v=torch.randn(1,2,13,8,requires_grad=True)
    mask=torch.randn(1,1,6,13)
    a=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
    g=torch.autograd.grad(a.square().sum(),(q,k,v))
    b=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=aligned_bias(mask))
    h=torch.autograd.grad(b.square().sum(),(q,k,v))
    assert torch.equal(a,b)
    assert all(torch.equal(x,y) for x,y in zip(g,h))
