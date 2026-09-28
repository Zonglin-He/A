"""Fixed episodic v2 TTA update interfaces; no labels or state selection."""
from __future__ import annotations
import torch
from vg_tta.desta3d_v2_tta_objective import (
    CalibrationWeights, calibration_objective, gradient_groups,
)

INTERFACES = {
    'calibration': ('branch_film', 'norm_affine'),
    'convolution': ('input_projection', 'shared_stem', 'spatial_reader', 'event_reader'),
}


def configure(adapter, interface):
    groups = adapter.parameter_groups()
    names = INTERFACES[interface]
    adapter.set_train_stage('frozen')
    adapter.zero_grad(set_to_none=True)
    for name in names:
        for p in groups[name]:
            p.requires_grad_(True)
    initial = {n: p.detach().clone() for n, p in adapter.named_parameters() if p.requires_grad}
    count = sum(p.numel() for p in initial.values())
    if interface == 'calibration':
        assert count == 66816
    assert all(not p.requires_grad for p in groups['gates'])
    return initial, count


def forward(adapter, fields):
    return adapter(fields['visual_grid'], fields['query_tokens'],
                   query_mask=fields['query_mask'], frame_times=fields['frame_times'])


def adapt(adapter, base_fields, view_fields, *, interface, alignment,
          moments, steps=3, lr=1e-5):
    """All gradients stop at the frozen captured PTD inputs. Always use step N."""
    assert steps > 0
    assert torch.equal(base_fields['frame_times'], view_fields['frame_times'])
    assert base_fields['visual_grid'].shape == view_fields['visual_grid'].shape
    assert not base_fields['visual_grid'].requires_grad and not view_fields['visual_grid'].requires_grad
    adapter.eval()
    initial_state = {n: v.detach().clone() for n, v in adapter.state_dict().items()}
    initial, count = configure(adapter, interface)
    with torch.no_grad():
        teacher = forward(adapter, base_fields)
    weights = CalibrationWeights(alignment=alignment)
    params = [p for p in adapter.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.)
    history = []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        student = forward(adapter, view_fields)
        loss, terms = calibration_objective(adapter, student, teacher, initial,
                                            weights=weights, source_moments=moments)
        loss.backward()
        gradients = gradient_groups(adapter)
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.))
        assert torch.isfinite(torch.tensor(norm))
        opt.step()
        actual = sorted({int(s['step']) for s in opt.state.values()})
        assert actual == [step]
        assert all(torch.isfinite(p).all() for p in adapter.parameters())
        history.append({'step': step, 'actual_Adam_steps': actual,
            'loss_before': float(loss.detach()),
            'terms_before': {k: float(v.detach()) for k, v in terms.items()},
            'gradient_groups': gradients, 'clip_norm_before': norm})
        del student, loss, terms
    changes = {n: not torch.equal(v, initial_state[n]) for n, v in adapter.state_dict().items()}
    assert all(not changed or n in initial for n, changed in changes.items())
    assert not changes['gate_event'] and not changes['gate_spatial']
    with torch.no_grad():
        final_loss, final_terms = calibration_objective(
            adapter, forward(adapter, view_fields), teacher, initial,
            weights=weights, source_moments=moments)
    report = {'interface': interface, 'updated_groups': list(INTERFACES[interface]),
        'parameter_count': count, 'steps': steps, 'history': history,
        'changed_tensors': changes, 'gates_unchanged': True,
        'teacher': 'fixed source-fit on same observed input; detached',
        'loss_after': float(final_loss),
        'terms_after': {k: float(v) for k, v in final_terms.items()},
        'joint': 0., 'alignment': alignment, 'GT_read': False,
        'selection': 'fixed terminal step, independent of loss or labels'}
    adapter.zero_grad(set_to_none=True)
    adapter.set_train_stage('frozen')
    return report


def validate_prediction(p, nframes):
    """Malformed model output is retained; impossible serialization is rejected."""
    positions = p['positions']
    boxes = p['boxes_cxcywh']
    valid = p['geometry_valid']
    assert boxes.shape == (len(positions), 4) and valid.shape == (len(positions),)
    assert torch.isfinite(boxes).all()
    assert len(set(positions)) == len(positions)
    assert all(isinstance(i, int) and 0 <= i < nframes for i in positions)
    if p['interval'] is not None:
        s, e = p['interval']
        assert 0 <= s <= e < nframes
    if p['format_ok']:
        assert p['interval'] is not None and positions and valid.all()
    return True
