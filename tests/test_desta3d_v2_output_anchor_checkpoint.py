import torch
from vg_tta.desta3d_v2_output_anchor_checkpoint_v4 import checkpoint_mlps


def test_stateless_checkpoint_forward_backward_after_context_exit():
    torch.manual_seed(9)
    mlps = [torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.SiLU(),
                               torch.nn.Linear(16, 8)).requires_grad_(False) for _ in range(3)]
    x = torch.randn(2, 8, requires_grad=True)
    y = x
    for m in mlps: y = y + m(y)
    loss = y.square().sum(); loss.backward(); gradient = x.grad.clone()
    expected = y.detach().clone(); x.grad = None
    with checkpoint_mlps(mlps):
        y = x
        for m in mlps: y = y + m(y)
    # Backward occurs after the original module methods have been restored.
    assert torch.equal(y, expected)
    y.square().sum().backward()
    assert torch.equal(x.grad, gradient)
    assert all(p.grad is None for m in mlps for p in m.parameters())
