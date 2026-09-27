from __future__ import annotations

import pytest
import torch

from vg_tta.desta3d_v2 import (
    ChannelOnlyLayerNorm,
    Desta3DAdapterV2,
    asymmetric_event_referent_loss,
    physical_time_derivative,
    topk_referent_support,
)


def _inputs(*, batch: int = 1, length: int = 3):
    torch.manual_seed(17)
    visual = torch.randn(batch, 4, 3, 2, 12)
    query = torch.randn(batch, length, 10)
    mask = torch.ones(batch, length, dtype=torch.bool)
    times = torch.tensor([[0.0, 0.4, 1.7, 3.0]]).expand(batch, -1).clone()
    return visual, query, mask, times


def _model(**kwargs) -> Desta3DAdapterV2:
    architecture = kwargs.pop("architecture", "dual3d")
    return Desta3DAdapterV2(
        in_channels=12,
        query_dim=10,
        hidden_dim=8,
        architecture=architecture,
        **kwargs,
    )


def test_query_changes_reader_features_and_attention_uses_caption_tokens() -> None:
    visual, query, mask, times = _inputs()
    model = _model().eval()
    original = model(visual, query, mask, frame_times=times)

    changed_query = query.clone()
    changed_query[:, 1] += 3.0
    changed = model(visual, changed_query, mask, frame_times=times)
    assert not torch.allclose(
        original["branch_features_spatial"],
        changed["branch_features_spatial"],
        atol=1e-8,
        rtol=1e-7,
    )
    assert not torch.allclose(
        original["branch_features_event"],
        changed["branch_features_event"],
        atol=1e-8,
        rtol=1e-7,
    )

    padded_query = torch.cat((query, torch.zeros(1, 2, query.shape[-1])), dim=1)
    padded_mask = torch.cat((mask, torch.zeros(1, 2, dtype=torch.bool)), dim=1)
    padded = model(visual, padded_query, padded_mask, frame_times=times)
    torch.testing.assert_close(original["z_spatial"], padded["z_spatial"])
    torch.testing.assert_close(original["z_event"], padded["z_event"])
    torch.testing.assert_close(original["branch_features_spatial"], padded["branch_features_spatial"])
    torch.testing.assert_close(original["branch_features_event"], padded["branch_features_event"])
    assert torch.equal(padded["alpha_spatial"][:, -2:], torch.zeros(1, 2))
    assert torch.equal(padded["alpha_event"][:, -2:], torch.zeros(1, 2))


def test_each_branch_has_independent_output_projection_and_gate() -> None:
    visual, query, mask, times = _inputs()
    model = _model().eval()
    with torch.no_grad():
        baseline = model(visual, query, mask, frame_times=times)
        model.out_proj_spatial.bias.add_(0.01)
        spatial_changed = model(visual, query, mask, frame_times=times)
    assert not torch.equal(baseline["updated_tokens_spatial"], spatial_changed["updated_tokens_spatial"])
    assert torch.equal(baseline["updated_tokens_event"], spatial_changed["updated_tokens_event"])

    with torch.no_grad():
        model.out_proj_spatial.bias.sub_(0.01)
        model.out_proj_event.bias.add_(0.01)
        event_changed = model(visual, query, mask, frame_times=times)
    assert torch.equal(baseline["updated_tokens_spatial"], event_changed["updated_tokens_spatial"])
    assert not torch.equal(baseline["updated_tokens_event"], event_changed["updated_tokens_event"])


def test_channel_layer_norm_at_a_time_point_is_independent_of_other_times() -> None:
    torch.manual_seed(11)
    norm = ChannelOnlyLayerNorm(4)
    features = torch.randn(1, 4, 4, 2, 3)
    first = norm(features)
    changed_features = features.clone()
    changed_features[:, :, 3] += 100.0
    second = norm(changed_features)
    torch.testing.assert_close(first[:, :, 0], second[:, :, 0], rtol=0, atol=0)


def test_physical_derivative_handles_irregular_time_and_rejects_nonincreasing_time() -> None:
    times = torch.tensor([[0.0, 1.0, 3.0, 7.0]])
    values = (2.0 * times + 5.0).unsqueeze(-1)
    derivative = physical_time_derivative(values, times)
    torch.testing.assert_close(derivative, torch.full_like(values, 2.0))

    with pytest.raises(ValueError, match="strictly increasing"):
        physical_time_derivative(values, torch.tensor([[0.0, 1.0, 1.0, 7.0]]))


def test_frame_event_readout_and_topk_referent_support_have_direct_gradients() -> None:
    visual, query, mask, times = _inputs(batch=2)
    model = _model(p1_enabled=True).train()
    out = model(visual, query, mask, frame_times=times)
    assert out["event_logits"].shape == (2, 4)
    assert out["referent_logits"].shape == (2, 4, 3, 2)
    assert out["event_derivative"] is not None

    support = topk_referent_support(out["referent_logits"], topk=3)
    assert support.shape == (2, 4)
    loss = out["event_logits"].square().mean() + support.mean()
    loss.backward()
    assert model.event_presence_head[-1].weight.grad is not None
    assert torch.isfinite(model.event_presence_head[-1].weight.grad).all()
    assert model.event_presence_head[-1].weight.grad.abs().sum() > 0
    assert model.event_reader.depthwise.weight.grad is not None
    assert torch.isfinite(model.event_reader.depthwise.weight.grad).all()
    assert model.event_reader.depthwise.weight.grad.abs().sum() > 0


def test_a_le_r_joint_uses_topk_support_and_backpropagates() -> None:
    event_logits = torch.tensor([[4.0, 0.0]], requires_grad=True)
    referent_logits = torch.tensor(
        [[[[-2.0, -2.0], [-2.0, -2.0]], [[2.0, 2.0], [2.0, 2.0]]]],
        requires_grad=True,
    )
    loss = asymmetric_event_referent_loss(event_logits, referent_logits, topk=2)
    assert loss.isfinite()
    loss.backward()
    assert event_logits.grad is not None and event_logits.grad[0, 0] != 0
    assert referent_logits.grad is not None and referent_logits.grad[0, 0].abs().sum() > 0


def test_small_nonzero_gates_and_output_projections_reach_both_readers() -> None:
    visual, query, mask, times = _inputs()
    model = _model(train_stage="B integration").train()
    out = model(visual, query, mask, frame_times=times)
    torch.testing.assert_close(out["gate_spatial"], torch.sigmoid(torch.tensor(-6.0)))
    torch.testing.assert_close(out["gate_event"], torch.sigmoid(torch.tensor(-6.0)))
    assert model.out_proj_spatial.weight.std() < 0.002
    assert model.out_proj_event.weight.std() < 0.002

    target_s = torch.randn_like(visual)
    target_e = torch.randn_like(visual)
    loss = (out["updated_tokens_spatial"] - target_s).square().mean()
    loss = loss + (out["updated_tokens_event"] - target_e).square().mean()
    loss.backward()
    for parameter in (
        model.spatial_reader.depthwise.weight,
        model.event_reader.depthwise.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_zero_gate_override_exactly_restores_original_tokens() -> None:
    visual, query, mask, times = _inputs()
    model = _model().eval()
    out = model(visual, query, mask, frame_times=times, gate_override=0.0)
    assert torch.equal(out["delta_spatial"], torch.zeros_like(visual))
    assert torch.equal(out["delta_event"], torch.zeros_like(visual))
    assert torch.equal(out["updated_tokens_spatial"], visual)
    assert torch.equal(out["updated_tokens_event"], visual)


def test_parameter_groups_and_tta_stage_are_explicit_and_counted() -> None:
    model = Desta3DAdapterV2(
        in_channels=12,
        query_dim=10,
        hidden_dim=128,
        architecture="dual3d",
    )
    groups = model.parameter_groups()
    flattened_ids = [id(p) for params in groups.values() for p in params]
    assert len(flattened_ids) == len(set(flattened_ids))
    assert sum(len(params) for params in groups.values()) > 0

    result = model.set_train_stage("tta")
    assert result["stage"] == "tta"
    enabled_names = {
        name for name, params in groups.items() if any(p.requires_grad for p in params)
    }
    assert enabled_names == {"branch_film", "norm_affine", "gates"}
    assert model.active_parameter_count() == model.tta_parameter_count() == 66_818
    assert not any(p.requires_grad for p in groups["shared_stem"])
    assert all(p.requires_grad for p in model.norm_stem.parameters())
    assert all(p.requires_grad for p in model.norm_spatial.parameters())
    assert all(p.requires_grad for p in model.norm_event.parameters())

    all_parameter_count = sum(p.numel() for p in model.parameters())
    assert model.parameter_count_by_group()["total"] == all_parameter_count
    model.set_train_stage("frozen")
    assert model.active_parameter_count() == 0


def test_p0_and_optional_p1_event_time_paths_are_explicit() -> None:
    visual, query, mask, times = _inputs()
    p0 = _model(p1_enabled=False).eval()
    p1 = _model(p1_enabled=True).eval()
    p0_out = p0(visual, query, mask, frame_times=times)
    p1_out = p1(visual, query, mask, frame_times=times)
    assert p0_out["event_derivative"] is None
    assert p0_out["p1_enabled"] is False
    assert p1_out["event_derivative"].shape == (1, 4, 3, 2, 8)
    assert p1.event_temporal_reader.dilations == (1, 2, 4)
    assert p0.event_temporal_reader.dilations == (1,)
    assert p0.event_p1_input_proj is None
    assert p1.event_p1_input_proj is not None
    with pytest.raises(ValueError, match="strictly increasing"):
        p1(visual, query, mask, frame_times=torch.tensor([[0.0, 1.0, 1.0, 3.0]]))


def test_p1_input_projection_receives_appearance_and_physical_derivative() -> None:
    visual, query, mask, times = _inputs()
    model = _model(p1_enabled=True).eval()
    captured: list[torch.Tensor] = []
    handle = model.event_p1_input_proj.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    out = model(visual, query, mask, frame_times=times)
    handle.remove()

    z_event, _ = model.query_pool_event(query, mask)
    latent = model.input_proj(visual)
    latent_cf = latent.permute(0, 4, 1, 2, 3).contiguous()
    latent = model.norm_stem(model.shared_stem(latent_cf)).movedim(1, -1)
    appearance = model.film_event(latent, z_event)
    expected = torch.cat((appearance, out["event_derivative"]), dim=-1)
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], expected)
    assert captured[0].shape[-1] == 2 * model.hidden_dim


@pytest.mark.parametrize("architecture", ["early_factorized", "shared3d", "dual3d"])
def test_matched_architecture_modes_keep_two_output_branches(architecture: str) -> None:
    visual, query, mask, times = _inputs()
    model = _model(architecture=architecture).eval()
    out = model(visual, query, mask, frame_times=times)
    assert out["updated_tokens_spatial"].shape == visual.shape
    assert out["updated_tokens_event"].shape == visual.shape
    assert out["branch_features_spatial"].shape == (1, 4, 3, 2, 8)
    assert out["branch_features_event"].shape == (1, 4, 3, 2, 8)
    groups = model.parameter_groups()
    stem_shapes = {name: tuple(parameter.shape) for name, parameter in model.shared_stem.named_parameters()}
    reference = _model(architecture="shared3d")
    assert stem_shapes == {
        name: tuple(parameter.shape) for name, parameter in reference.shared_stem.named_parameters()
    }
    assert groups["shared_stem"]
    if architecture == "shared3d":
        assert groups["shared_reader"]
        assert not groups["spatial_reader"]
        # The event group still owns the independent temporal dilation reader.
        assert groups["event_reader"]
    else:
        assert not groups["shared_reader"]
        assert groups["spatial_reader"] and groups["event_reader"]
