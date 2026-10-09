"""Portable contracts for the authorized chart, gradients and trace linkage."""
import copy
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from vg_tta.decota_spatial_opd_boundary_chart_proposal003 import proposed_coordinates
    from vg_tta.decota_spatial_opd_chart_revision003 import chart_audit, REVISION
    valid = 0
    for raw_values in [[-4., -1., .1, 4.], [-100., -2., 2., 18.],
                       [-1000., -18., 18., 1000.]]:
        x = torch.tensor([raw_values], requires_grad=True)
        b = x.sigmoid(); value, mask = proposed_coordinates(b, x)
        assert torch.isfinite(value).all() and torch.equal(value[mask], x[mask])
        gradients = torch.autograd.grad(value.square().sum(), x, retain_graph=True)[0]
        assert torch.isfinite(gradients).all()
        assert torch.equal(gradients[mask], 2 * x.detach()[mask])
        if not mask.any():
            original = torch.logit(b)
            assert torch.equal(value, original)
            assert torch.equal(gradients, torch.autograd.grad(original.square().sum(), x)[0])
        valid += 1
    x = torch.tensor([[-100., -2., 2., 18.]])
    b = x.sigmoid(); value, mask = proposed_coordinates(b, x)
    trace = [dict(round=0, phase=p, boxes=b, raw_logits=x, mean=value,
                  boundary=mask, same_native_call_verified=True) for p in ('before', 'after')]
    fit = dict(numerical_chart=dict(revision=REVISION, interior_formula='original_torch_logit',
        endpoint_formula='same_call_finite_native_logit', trace=trace),
        config=dict(steps=1), positions=[0], path=[dict(boxes=b), dict(boxes=b)],
        rounds=[dict(mean_before=value, mean_after=value)])
    assert chart_audit(fit)['endpoint_coordinates'] == 4
    valid += 1
    bad = []
    def rejected(name, modified):
        try: chart_audit(modified)
        except AssertionError: bad.append(name)
        else: raise AssertionError('Malformed chart trace accepted: ' + name)
    z = copy.deepcopy(fit); z['numerical_chart']['trace'][0]['boundary'][0, 0] = False
    rejected('wrong_endpoint_mask', z)
    z = copy.deepcopy(fit); z['numerical_chart']['trace'][0]['raw_logits'][0, 0] = -99.
    rejected('wrong_endpoint_value', z)
    z = copy.deepcopy(fit); z['numerical_chart']['trace'][0]['mean'][0, 1] += .01
    rejected('wrong_interior_chart', z)
    z = copy.deepcopy(fit); z['numerical_chart']['trace'][0]['raw_logits'][0, 1] += 1.
    rejected('different_native_output', z)
    z = copy.deepcopy(fit); z['numerical_chart']['trace'][0]['round'] = 1
    rejected('wrong_round', z)
    z = copy.deepcopy(fit); z['numerical_chart']['trace'][0]['same_native_call_verified'] = False
    rejected('unbound_native_call', z)
    z = copy.deepcopy(fit); z['path'][1]['boxes'] = z['path'][1]['boxes'] + .01
    rejected('wrong_path_readout', z)
    z = copy.deepcopy(fit); z['rounds'][0]['mean_after'] = z['rounds'][0]['mean_after'] + .01
    rejected('wrong_round_mean', z)
    z = copy.deepcopy(fit); z['numerical_chart']['trace'][0]['raw_logits'][0, 0] = float('inf')
    rejected('nonfinite_logit', z)
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='pass', valid=valid, rejected=len(bad), rejected_cases=bad,
        synthetic_only=True, real_GPU_qualification=False, model_or_GT_access=False,
        interior_values_gradients_bitwise=True, finite_endpoint_gradient=True)))


if __name__ == '__main__': main()
