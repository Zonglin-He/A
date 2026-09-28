from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from vg_tta.desta3d_v2 import Desta3DAdapterV2
from vg_tta.desta3d_v2_aux_backflow import (
    auxiliary_gradient_view,
    auxiliary_head_backflow,
    combine_window_gradient_buffers,
    parameter_displacement_summary,
)


def _adapter(*, hidden_dim: int = 8) -> Desta3DAdapterV2:
    torch.manual_seed(601)
    return Desta3DAdapterV2(
        in_channels=6,
        query_dim=7,
        hidden_dim=hidden_dim,
        architecture="dual3d",
        train_stage="B integration",
    )


def _batch():
    torch.manual_seed(72)
    visual = torch.randn(2, 3, 2, 2, 6)
    query = torch.randn(2, 4, 7)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    times = torch.tensor([[0.0, 0.3, 1.7], [0.0, 2.0, 5.0]])
    target_ref = torch.rand(2, 3, 2, 2)
    target_event = torch.rand(2, 3)
    target_tokens = torch.randn_like(visual)
    return visual, query, mask, times, target_ref, target_event, target_tokens


def _losses(model, batch):
    visual, query, mask, times, target_ref, target_event, target_tokens = batch
    output = model(visual, query, mask, frame_times=times)
    task_loss = (
        (output["updated_tokens_spatial"] - target_tokens).square().mean()
        + (output["updated_tokens_event"] + 0.2 * target_tokens).square().mean()
    )
    auxiliary_loss = F.binary_cross_entropy_with_logits(
        output["referent_logits"], target_ref
    ) + F.binary_cross_entropy_with_logits(output["event_logits"], target_event)
    return output, task_loss, auxiliary_loss


def _grads(loss, parameters, *, retain_graph: bool = False):
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )


def _as_cpu_buffers(names, gradients, *, scale: float = 1.0):
    return {
        name: None if grad is None else grad.detach().to(device="cpu", dtype=torch.float32) * scale
        for name, grad in zip(names, gradients, strict=True)
    }


def _assert_gradient_maps_close(actual, expected, *, scale: float = 1.0):
    assert actual.keys() == expected.keys()
    for name in actual:
        left, right = actual[name], expected[name]
        if right is None:
            assert left is None, name
        else:
            assert left is not None, name
            torch.testing.assert_close(left, right * scale, rtol=2e-6, atol=2e-8, msg=name)


def test_auxiliary_gradient_view_preserves_values_head_gradient_and_scales_input():
    torch.manual_seed(8)
    x = torch.randn(3, 4, requires_grad=True)
    head = torch.nn.Linear(4, 2)
    baseline = head(x)
    baseline_loss = baseline.square().sum()
    baseline_x, baseline_w, baseline_b = torch.autograd.grad(
        baseline_loss, (x, head.weight, head.bias)
    )

    for coefficient in (1.0, 0.0, 0.2):
        x2 = x.detach().clone().requires_grad_(True)
        output = head(auxiliary_gradient_view(x2, coefficient))
        torch.testing.assert_close(output, baseline, rtol=0, atol=0)
        grad_x, grad_w, grad_b = torch.autograd.grad(
            output.square().sum(), (x2, head.weight, head.bias)
        )
        torch.testing.assert_close(grad_x, baseline_x * coefficient, rtol=0, atol=0)
        torch.testing.assert_close(grad_w, baseline_w, rtol=0, atol=0)
        torch.testing.assert_close(grad_b, baseline_b, rtol=0, atol=0)

    for invalid in (-0.01, 1.01, float("nan"), float("inf"), True):
        with pytest.raises((TypeError, ValueError)):
            auxiliary_gradient_view(x, invalid)
    with pytest.raises(ValueError, match="detached"):
        auxiliary_gradient_view(x, torch.tensor(0.5, requires_grad=True))
    with pytest.raises(ValueError, match="exactly one"):
        auxiliary_gradient_view(x, torch.tensor([0.2, 0.3]))


def test_real_adapter_hooks_keep_task_and_head_gradients_and_scale_both_readers():
    model = _adapter().eval()
    batch = _batch()
    named = dict(model.named_parameters())
    names = tuple(named)
    params = tuple(named.values())

    base_output, base_task, base_aux = _losses(model, batch)
    base_task_grads = _as_cpu_buffers(names, _grads(base_task, params, retain_graph=True))
    base_aux_grads = _as_cpu_buffers(names, _grads(base_aux, params))
    head_parameter_ids = {
        id(parameter)
        for head in (model.referent_head, model.event_presence_head)
        for parameter in head.parameters()
    }
    head_names = {
        name for name, parameter in named.items() if id(parameter) in head_parameter_ids
    }

    for coefficient in (0.0, 0.2, 1.0):
        with auxiliary_head_backflow(model, coefficient):
            output, task_loss, aux_loss = _losses(model, batch)
            task_grads = _as_cpu_buffers(names, _grads(task_loss, params, retain_graph=True))
            aux_grads = _as_cpu_buffers(names, _grads(aux_loss, params))

        for key in ("referent_logits", "event_logits", "updated_tokens_spatial", "updated_tokens_event"):
            torch.testing.assert_close(output[key], base_output[key], rtol=0, atol=0)
        torch.testing.assert_close(task_loss, base_task, rtol=0, atol=0)
        torch.testing.assert_close(aux_loss, base_aux, rtol=0, atol=0)
        _assert_gradient_maps_close(task_grads, base_task_grads)
        for name in names:
            _assert_gradient_maps_close(
                {name: aux_grads[name]},
                {name: base_aux_grads[name]},
                scale=1.0 if name in head_names else coefficient,
            )

    assert any(
        base_aux_grads[name] is not None and base_aux_grads[name].abs().sum() > 0
        for name in named
        if name.startswith("spatial_reader.")
    )
    assert any(
        base_aux_grads[name] is not None and base_aux_grads[name].abs().sum() > 0
        for name in named
        if name.startswith("event_reader.")
    )


def test_window_buffer_b2_matches_hooked_full_objective_and_reports_diagnostics():
    model = _adapter().eval()
    batch = _batch()
    named = dict(model.named_parameters())
    names, params = tuple(named), tuple(named.values())
    output, task_loss, auxiliary_loss = _losses(model, batch)
    task_buffers = _as_cpu_buffers(names, _grads(task_loss, params, retain_graph=True))
    weighted_aux_buffers = _as_cpu_buffers(
        names, _grads(0.1 * auxiliary_loss, params), scale=1.0
    )
    expected_stem_count = model.parameter_count_by_group()["shared_stem"]
    combined, diagnostics = combine_window_gradient_buffers(
        model,
        task_buffers,
        weighted_aux_buffers,
        mode="B2",
        expected_shared_stem_numel=expected_stem_count,
    )
    assert diagnostics["mode"] == "B2"
    assert diagnostics["shared_stem_parameter_count"] == expected_stem_count
    coefficient = diagnostics["aux_reader_backflow_coefficient"]
    task_norm = diagnostics["shared_stem_task_norm"]
    aux_norm = diagnostics["shared_stem_weighted_aux_norm"]
    expected_coefficient = 0.0 if task_norm == 0 else min(1.0, 0.25 * task_norm / (aux_norm + 1e-12))
    assert coefficient == pytest.approx(expected_coefficient)
    assert diagnostics["groups"]["total"]["gT_norm"] >= 0
    assert diagnostics["groups"]["total"]["gS_norm"] >= 0
    assert "do not guarantee Adam/AdamW" in diagnostics["interpretation"]

    with auxiliary_head_backflow(model, coefficient):
        direct_output, direct_task, direct_aux = _losses(model, batch)
        direct_grads = _grads(direct_task + 0.1 * direct_aux, params)
    direct_buffers = _as_cpu_buffers(names, direct_grads)
    _assert_gradient_maps_close(combined, direct_buffers)
    for key in ("event_logits", "referent_logits"):
        torch.testing.assert_close(direct_output[key], output[key], rtol=0, atol=0)

    # A measured optimizer displacement is recordable but is not predicted by
    # the first-order diagnostics above.
    before = {name: parameter.detach().clone() for name, parameter in named.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for name, parameter in named.items():
        gradient = combined[name]
        parameter.grad = None if gradient is None else gradient.to(parameter.dtype).clone()
    optimizer.step()
    after = {name: parameter.detach().clone() for name, parameter in named.items()}
    displacement = parameter_displacement_summary(before, after)
    assert displacement["numel"] == sum(parameter.numel() for parameter in named.values())
    assert displacement["changed_parameter_tensors"] > 0
    assert math.isfinite(displacement["l2_norm"]) and displacement["l2_norm"] > 0


def test_modes_preserve_none_versus_zero_and_head_aux_gradient():
    model = _adapter()
    named = dict(model.named_parameters())
    all_none = {name: None for name in named}
    first_shared = model.parameter_groups()["shared_stem"][0]
    shared_name = next(name for name, parameter in named.items() if parameter is first_shared)
    first_head = next(parameter for parameter in model.referent_head.parameters())
    head_name = next(name for name, parameter in named.items() if parameter is first_head)
    ordinary_name = "spatial_reader.depthwise.weight"

    task = dict(all_none)
    auxiliary = dict(all_none)
    task[shared_name] = torch.zeros_like(named[shared_name], device="cpu", dtype=torch.float32)
    auxiliary[shared_name] = torch.ones_like(named[shared_name], device="cpu", dtype=torch.float32)
    auxiliary[head_name] = torch.ones_like(named[head_name], device="cpu", dtype=torch.float32)
    auxiliary[ordinary_name] = torch.ones_like(named[ordinary_name], device="cpu", dtype=torch.float32)
    expected_stem_count = model.parameter_count_by_group()["shared_stem"]

    b0, diag0 = combine_window_gradient_buffers(
        model, task, auxiliary, mode="B0", expected_shared_stem_numel=expected_stem_count
    )
    assert diag0["mode"] == "B0c1"
    torch.testing.assert_close(b0[ordinary_name], torch.ones_like(b0[ordinary_name]))
    torch.testing.assert_close(b0[head_name], torch.ones_like(b0[head_name]))

    b1, diag1 = combine_window_gradient_buffers(
        model, task, auxiliary, mode="B1", expected_shared_stem_numel=expected_stem_count
    )
    assert diag1["aux_reader_backflow_coefficient"] == 0.0
    assert b1[ordinary_name] is not None and torch.count_nonzero(b1[ordinary_name]) == 0
    assert b1[shared_name] is not None and torch.count_nonzero(b1[shared_name]) == 0
    # Head gradients keep their original (already weighted) value at c=0.
    torch.testing.assert_close(b1[head_name], torch.ones_like(b1[head_name]))
    assert b1["out_proj_spatial.weight"] is None

    # An absent task buffer stays absent, while an explicitly materialized
    # zero remains a tensor after combination.
    only_aux = dict(all_none)
    only_aux[ordinary_name] = torch.zeros_like(named[ordinary_name], dtype=torch.float32)
    preserved, _ = combine_window_gradient_buffers(
        model,
        all_none,
        only_aux,
        mode="B1c0",
        expected_shared_stem_numel=expected_stem_count,
    )
    assert preserved["out_proj_spatial.weight"] is None
    assert preserved[ordinary_name] is not None
    assert torch.count_nonzero(preserved[ordinary_name]) == 0


def test_shared_stem_guard_finite_buffers_and_parameter_groups():
    small = _adapter(hidden_dim=8)
    names = dict(small.named_parameters())
    no_grad = {name: None for name in names}
    with pytest.raises(ValueError, match="shared_stem has"):
        combine_window_gradient_buffers(small, no_grad, no_grad, mode="B2")
    with pytest.raises(ValueError, match="non-finite"):
        bad = dict(no_grad)
        key = next(iter(names))
        bad[key] = torch.full_like(names[key], float("nan"), device="cpu", dtype=torch.float32)
        combine_window_gradient_buffers(
            small,
            bad,
            no_grad,
            mode="B2",
            expected_shared_stem_numel=small.parameter_count_by_group()["shared_stem"],
        )

    production = _adapter(hidden_dim=128)
    groups = production.parameter_groups()
    stem_count = sum(parameter.numel() for parameter in groups["shared_stem"])
    assert stem_count == 19_968
    flattened = [id(parameter) for values in groups.values() for parameter in values]
    assert len(flattened) == len(set(flattened))
    production.set_train_stage("tta")
    assert all(parameter.requires_grad for parameter in groups["norm_affine"])
    assert not any(parameter.requires_grad for parameter in groups["shared_stem"])
    assert production.active_parameter_count() == production.tta_parameter_count()
