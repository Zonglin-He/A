"""Analytic controls for subtraction, sign, no-op and FP32 SGD semantics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from vg_tta.tastvg_ur_write_decomposition_v1 import difference, sgd, gradient_geometry


def run():
    state = {'a': torch.tensor([1., -2., 3.]), 'b': torch.tensor([.3])}
    u = {'a': torch.tensor([1., 2., -1.]), 'b': torch.tensor([0.])}
    r = {'a': torch.tensor([2., 1., 1.]), 'b': torch.tensor([1.])}
    d = difference(r, u)
    assert torch.equal(d['a'], torch.tensor([1., -1., 2.]))
    assert torch.equal(sgd(state, d, .1)['a'], state['a'].clone().add_(d['a'], alpha=-.1))
    z = difference(u, u)
    assert all(torch.equal(v, state[k]) for k, v in sgd(state, z, .1).items())
    equal = gradient_geometry(u, u)
    assert abs(equal['cosine_U_R']-1.) < 1e-12 and equal['norm_specific'] == 0
    opposed = gradient_geometry(u, {k: -v for k, v in u.items()})
    assert abs(opposed['cosine_U_R']+1.) < 1e-12
    absent = gradient_geometry(u, z)
    assert absent['cosine_U_R'] is None and absent['cosine_defined'] is False
    assert torch.equal(state['a'], torch.tensor([1., -2., 3.]))
    print('PASS: 6 analytic controls; no inference, labels or CUDA')


if __name__ == '__main__':
    run()
