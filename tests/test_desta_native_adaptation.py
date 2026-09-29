import numpy as np
import pytest
import torch
from desta3d.native_adaptation import (
    AdaptationConfig, temporal_target, spatial_target, normalized_update,
    project_branch, native_loss, grid_configs,
)


def test_physical_time_and_missing_boxes():
    y, v = temporal_target([50., 90.], [10, 11, 90])
    assert y.tolist() == [1, 2] and v.all()
    ids = list(range(150000, 151001))
    y, v = spatial_target({'10': [0., .2, .7, 1.]}, [10, 11, 90], [0, 2], ids)
    assert y[0].tolist() == [150000, 150200, 150700, 151000]
    assert v[0].all() and not v[1].any()
    assert not temporal_target(None, [1, 3])[1].any()


def test_full_normalization_before_projection_and_radius():
    q = torch.tensor([[1., 0.], [0., 1.], [0., 0.]])
    gt = torch.tensor([[1., 0., 100.]])
    gs = torch.tensor([[0., 2., 0.]])
    pt, nt = project_branch(gt, q); ps, ns = project_branch(gs, q)
    cfg = AdaptationConfig(3, .07, 2., 1.)
    c = torch.zeros(1, 2)
    expected = (2 * gt.numpy() / np.linalg.norm(gt.numpy()) + gs.numpy() / np.linalg.norm(gs.numpy())) @ q.numpy()
    for _ in range(3):
        c, meta = normalized_update(c, {'event': pt, 'spatial': ps}, {'event': nt, 'spatial': ns}, q, 20., cfg)
    expected = -expected / np.linalg.norm(expected) * .07 * 20.
    np.testing.assert_allclose(c.numpy(), expected, rtol=1e-6)
    assert float((c @ q.T).norm()) <= .07 * 20. + 1e-6
    zero, _ = normalized_update(torch.zeros_like(c), {'event': c*0, 'spatial': c*0}, {'event': 0., 'spatial': 0.}, q, 20., cfg)
    assert not zero.any()


def test_full_vocab_loss_not_restricted_and_frozen_model():
    torch.manual_seed(3)
    w = torch.randn(152775, 3)
    f = torch.randn(2, 3, requires_grad=True)
    logits = f @ w.T
    y = torch.tensor([150002, 150020]); valid = torch.ones(2, dtype=torch.bool)
    loss = native_loss(logits, y, valid)
    loss.backward()
    assert torch.isfinite(f.grad).all() and f.grad.norm() > 0 and w.grad is None
    assert len(grid_configs()) == 27
    with pytest.raises(ValueError):
        AdaptationConfig(0, .1)
