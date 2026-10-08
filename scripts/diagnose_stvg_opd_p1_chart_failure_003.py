"""CPU diagnosis of the one reproduced numerical failure, with no GT/scoring."""
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, load, sha
REC = ROOT / 'artifacts/stvg_opd_paper_hc2_revision_v2/recovery/P1_box_chart_003'


def main():
    import numpy as np
    import torch
    from vg_tta.decota_spatial_opd_precision_audit_revision002 import audit
    from vg_tta.decota_fixed_full_audit_v1 import vector
    from vg_tta.decota_spatial_opd_boundary_chart_proposal003 import proposed_coordinates
    begin = time.perf_counter()
    z = load(REC / 'REPRODUCED_CHART_FAILURE.pt')
    rc = read(REC / 'REPRODUCTION_RECEIPT.json')
    assert sha(REC / 'REPRODUCED_CHART_FAILURE.pt') == rc['failed_state_sha256']
    assert z['failed_round'] == 8 and z['GT_read'] is False
    boxes = z['chart_boxes']
    raw = z['native_pre_sigmoid_logits'][z['positions']]
    assert torch.isfinite(boxes).all() and torch.isfinite(raw).all()
    bad = (boxes <= 0) | (boxes >= 1)
    assert bad.nonzero().tolist() == [[1, 3]] and boxes[1, 3].item() == 1.0
    exact64 = torch.sigmoid(raw.double())
    assert 0 < exact64[1, 3] < 1
    partial = dict(config=dict(z['config'], steps=8), selected_step=8, path=z['path'],
        rounds=z['rounds'], initial=z['initial'], state=z['path'][8]['state'],
        final=z['path'][8]['boxes'], active_parameters=1792, positions=z['positions'],
        GT_used=False, gradient_calls=sum(r['updated'] for r in z['rounds']))
    partial_check = audit(partial, z['expert'])
    names = list(z['initial'])
    m = np.zeros(1792)
    v = np.zeros(1792)
    max_error = 0.0
    for count, entry in enumerate(z['path'], 1):
        u = entry['update']
        g = u['gradient'].numpy().astype(float)
        m = .9 * m + .1 * g
        v = .999 * v + .001 * g * g
        delta = -z['config']['lr'] * (m / (1 - .9 ** count)) / (np.sqrt(v / (1 - .999 ** count)) + 1e-8)
        error = float(abs(delta - u['raw'].numpy()).max())
        assert error < 4e-6
        max_error = max(max_error, error)
    assert count == 9
    before = vector(z['path'][8]['state'], names)
    after = vector(z['failed_state'], names)
    assert np.array_equal(after - before, z['path'][8]['update']['raw'].numpy())
    assert all(torch.isfinite(value).all() for value in z['failed_state'].values())

    valid = 0
    rejected = []
    test = torch.tensor([[-2., -.25, .1, 2.]], requires_grad=True)
    native = test.sigmoid()
    old = torch.logit(native)
    proposed, mask = proposed_coordinates(native, test)
    assert not mask.any() and torch.equal(old, proposed)
    old_gradient = torch.autograd.grad(old.square().sum(), test, retain_graph=True)[0]
    new_gradient = torch.autograd.grad(proposed.square().sum(), test)[0]
    assert torch.equal(old_gradient, new_gradient)
    valid += 1
    boundary_logits = torch.tensor([[-100., -2., 2., 18.]], requires_grad=True)
    central = boundary_logits.sigmoid()
    mean, mask = proposed_coordinates(central, boundary_logits)
    assert mask.tolist() == [[True, False, False, True]]
    assert torch.equal(mean[mask], boundary_logits[mask])
    assert torch.isfinite(mean).all() and torch.equal(mean.sigmoid(), central)
    gradient = torch.autograd.grad(mean.sum(), boundary_logits)[0]
    assert torch.isfinite(gradient).all() and torch.equal(gradient[mask], torch.ones(2))
    valid += 1
    actual_logits = raw.detach().clone().requires_grad_(True)
    cpu_central = actual_logits.sigmoid()
    actual_mean, actual_mask = proposed_coordinates(cpu_central, actual_logits)
    assert actual_mask.nonzero().tolist() == [[1, 3]]
    assert actual_mean[1, 3].item() == raw[1, 3].item()
    assert torch.equal(actual_mean[~actual_mask], torch.logit(cpu_central[~actual_mask]))
    assert torch.isfinite(torch.autograd.grad(actual_mean.sum(), actual_logits)[0]).all()
    valid += 1
    for name, value, logits in [
        ('nonfinite_box', torch.tensor([float('nan')]), torch.tensor([0.])),
        ('nonfinite_logit', torch.tensor([1.]), torch.tensor([float('inf')])),
        ('box_above_one', torch.tensor([1.01]), torch.tensor([18.])),
        ('box_below_zero', torch.tensor([-.01]), torch.tensor([-100.])),
        ('mismatched_native_output', torch.tensor([.7]), torch.tensor([0.])),
        ('wrong_shape', torch.tensor([.5, .5]), torch.tensor([0.])),
    ]:
        try:
            proposed_coordinates(value, logits)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Invalid proposal input accepted: ' + name)
    assert not torch.cuda.is_initialized()
    result = dict(status='diagnosed', stage='P1_vidstg', order='order1', arrival=1882,
        query_ordinal=10220, failed_round_zero_based=8, failed_coordinate='height_of_second_admitted_frame',
        float32_box_value=boxes[1, 3].item(), native_float32_pre_sigmoid_logit=raw[1, 3].item(),
        float64_sigmoid_of_same_native_logit=exact64[1, 3].item(),
        cause='finite_native_logit_sigmoid_rounds_to_one_and_inverse_chart_rejects',
        all_boxes_logits_parameters_finite=True, first_eight_completed_rounds_CPU_audit=partial_check,
        all_nine_Adam_updates_independent_max_error=max_error,
        ninth_parameter_update_exact=True, original_failure_preserved=True,
        draft_CPU_contracts=dict(valid=valid, rejected=len(rejected), cases=rejected,
            actual_reproduced_boundary_checked=True, original_interior_values_and_gradients_bitwise=True,
            endpoint_functional_gradient_finite=True, independent_decoder_Jacobian_not_checked=True),
        proposal_status='draft_not_authorized_not_installed_in_any_GPU_runner',
        proposal_changes_locked_endpoint_failure_behavior=True,
        no_new_GPU_qualification_or_prediction_claimed=True, GT_read=False,
        CPU_seconds=time.perf_counter()-begin,
        proposal_sha256=sha(ROOT / 'vg_tta/decota_spatial_opd_boundary_chart_proposal003.py'),
        reproduced_failure_sha256=sha(REC / 'REPRODUCED_CHART_FAILURE.pt'), time=time.time())
    write(REC / 'CPU_DIAGNOSIS_AND_DRAFT_CONTRACTS.json', result)
    print(json.dumps({k: result[k] for k in ['status', 'cause', 'float32_box_value',
        'native_float32_pre_sigmoid_logit', 'float64_sigmoid_of_same_native_logit',
        'proposal_status', 'GT_read']}))


if __name__ == '__main__':
    main()
