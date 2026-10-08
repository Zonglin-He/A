"""Portable CPU contracts for a draft chart proposal; no model or data access."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from vg_tta.decota_spatial_opd_boundary_chart_proposal003 import proposed_coordinates
    valid = 0
    x = torch.tensor([[-2., -.25, .1, 2.]], requires_grad=True)
    b = x.sigmoid()
    a = torch.logit(b)
    p, mask = proposed_coordinates(b, x)
    assert not mask.any() and torch.equal(a, p)
    ga = torch.autograd.grad(a.square().sum(), x, retain_graph=True)[0]
    gp = torch.autograd.grad(p.square().sum(), x)[0]
    assert torch.equal(ga, gp)
    valid += 1
    x = torch.tensor([[-100., -2., 2., 18.]], requires_grad=True)
    b = x.sigmoid()
    a, mask = proposed_coordinates(b, x)
    assert mask.tolist() == [[True, False, False, True]]
    assert torch.equal(a[mask], x[mask]) and torch.equal(a.sigmoid(), b)
    g = torch.autograd.grad(a.sum(), x)[0]
    assert torch.isfinite(a).all() and torch.isfinite(g).all()
    assert torch.equal(g[mask], torch.ones(2))
    valid += 1
    x = torch.tensor([18.])
    a, mask = proposed_coordinates(x.sigmoid(), x)
    assert a.item() == 18. and mask.item()
    valid += 1
    x = torch.empty(0, 4)
    a, mask = proposed_coordinates(x.sigmoid(), x)
    assert a.shape == x.shape == mask.shape and not mask.any()
    valid += 1
    rejected = []
    for name, box, raw in [
        ('nonfinite_box', torch.tensor([float('nan')]), torch.tensor([0.])),
        ('nonfinite_logit', torch.tensor([1.]), torch.tensor([float('inf')])),
        ('box_above_one', torch.tensor([1.01]), torch.tensor([18.])),
        ('box_below_zero', torch.tensor([-.01]), torch.tensor([-100.])),
        ('wrong_native_output', torch.tensor([.7]), torch.tensor([0.])),
        ('wrong_shape', torch.tensor([.5, .5]), torch.tensor([0.])),
        ('wrong_dtype', torch.tensor([.5]), torch.tensor([0.], dtype=torch.float64)),
    ]:
        try:
            proposed_coordinates(box, raw)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Invalid synthetic input accepted: ' + name)
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='pass', valid=valid, rejected=len(rejected), cases=rejected,
        synthetic_only=True, GPU_qualification=False, model_or_GT_access=False,
        proposal_not_installed=True, interior_values_and_gradients_bitwise=True,
        endpoint_gradient_finite=True)))


if __name__ == '__main__':
    main()
