import copy
import pytest
import torch

from vg_tta.native_probability_interface_v1 import NativeOutput
from vg_tta.vitta_paper_v1 import (
    REDUCTION, SourceStatistics, View, adapt, alignment_loss, capture_layer_outputs,
    feature_statistics, physical_probabilities, temporal_consistency, validate_source,
)


def output(ids, sted=None, action=None, valid=None):
    t = len(ids)
    return NativeOutput(torch.zeros(t, 2) if sted is None else sted,
                        torch.zeros(t) if action is None else action,
                        torch.ones(t, dtype=torch.bool) if valid is None else valid, tuple(ids))


def source(stats=None):
    return dict(provenance=dict(source_dataset="vidstg", checkpoint_sha256="a" * 64,
                               source_manifest_sha256="b" * 64, official_split="train",
                               source_count=2, query_count=2, target_inputs_used=False,
                               localization_labels_used=False, reduction=REDUCTION),
                statistics=stats or {"layer": dict(mean=torch.zeros(2), var=torch.ones(2))})


def toy():
    p = torch.nn.Parameter(torch.tensor([.3, -.2]), requires_grad=False)
    ids = [0, 1, 2, 3, 4, 5]
    def closure():
        result = []
        for offset in (0, 1):
            f = torch.tensor([[1., 3.], [2., 1.], [-1., 2.]]) + offset * .4
            f = f * p + .7
            native = output(ids[offset::2], f, f[:, 0])
            result.append(View(native, {"layer": f}))
        return result
    return p, ids, closure


def test_joint_feature_moments_match_independent_population_calculation():
    p, _, closure = toy()
    views = closure()
    got = feature_statistics(views)["layer"]
    rows = [r.tolist() for v in views for r in v.features["layer"]]
    means = [sum(r[c] for r in rows) / len(rows) for c in range(2)]
    variances = [sum((r[c] - means[c]) ** 2 for r in rows) / len(rows) for c in range(2)]
    assert torch.allclose(got["mean"], torch.tensor(means))
    assert torch.allclose(got["var"], torch.tensor(variances))


def test_source_estimator_equals_official_mean_of_per_video_variances():
    acc = SourceStatistics()
    acc.add({"x": dict(mean=torch.tensor([0.]), var=torch.tensor([1.]))})
    acc.add({"x": dict(mean=torch.tensor([10.]), var=torch.tensor([3.]))})
    assert acc.count == 2
    assert acc.finish()["x"]["mean"].item() == 5
    assert acc.finish()["x"]["var"].item() == 2  # not global pooled variance 27


def test_alignment_uses_mean_over_channels_then_sum_over_layers():
    a = {"x": dict(mean=torch.tensor([1., 3.]), var=torch.tensor([2., 6.]))}
    b = {"x": dict(mean=torch.zeros(2), var=torch.zeros(2))}
    assert alignment_loss(a, b).item() == 6


@pytest.mark.parametrize("field,value", [("target_inputs_used", True), ("localization_labels_used", True),
                                        ("official_split", "test"), ("reduction", "global_variance")])
def test_reject_wrong_source_provenance(field, value):
    s = source()
    s["provenance"][field] = value
    with pytest.raises(ValueError):
        validate_source(s)


def test_source_rejects_checkpoint_layer_dataset_mismatch():
    for kw in [dict(checkpoint_sha256="c" * 64), dict(source_dataset="hcstvg2"), dict(layer_names=["other"])]:
        with pytest.raises(ValueError):
            validate_source(source(), **kw)


def test_physical_rebin_preserves_endpoint_mass_with_nonuniform_ids():
    grid = [0, 1, 4, 9, 10, 20]
    z = torch.tensor([[2., -1.], [0., 1.], [-2., 3.]], requires_grad=True)
    p, a = physical_probabilities(output(grid[::2], z, z[:, 0]), grid)
    assert torch.allclose(p.sum(0), torch.ones(2))
    assert bool((p >= 0).all() and (a >= 0).all() and (a <= 1).all())
    (p[2, 0] + a[3]).backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_physical_mapping_not_equal_length_offset_index_averaging():
    grid = [0, 1, 4, 9]
    z = torch.tensor([[3., 3.], [-3., -3.]])
    a, _ = physical_probabilities(output([0, 4], z), grid)
    b, _ = physical_probabilities(output([1, 9], z), grid)
    assert not torch.equal(a, b)


def test_padding_nan_excluded_from_physical_probabilities():
    o = output([0, 2, 4], torch.tensor([[1., 0.], [2., 0.], [float("nan"), float("nan")]]),
               torch.tensor([0., 1., float("nan")]), torch.tensor([True, True, False]))
    p, a = physical_probabilities(o, [0, 1, 2, 3])
    assert torch.isfinite(p).all() and torch.isfinite(a).all()
    assert torch.allclose(p.sum(0), torch.ones(2))


def test_temporal_consistency_rejects_duplicate_view_grid():
    _, ids, closure = toy()
    views = closure()
    with pytest.raises(ValueError):
        temporal_consistency([views[0], views[0]], ids)


def test_temporal_consistency_has_live_gradient():
    p, ids, closure = toy()
    p.requires_grad_(True)
    loss = temporal_consistency(closure(), ids)
    loss.backward()
    assert loss > 0 and p.grad is not None and p.grad.abs().sum() > 0


def test_real_parameter_update_final_inference_and_exact_episodic_reset():
    p, ids, closure = toy()
    initial = p.detach().clone()
    def final_inference():
        assert not p.requires_grad  # exact source numerical inference path
        return p.clone()
    result, audit = adapt([p], closure, final_inference, source(), ids, dict(lr=.02))
    assert not torch.equal(result, initial)
    assert audit["parameter_changed"] and audit["backwards"] == 1
    assert audit["gradient_norms"][0] > 0 and audit["source_restored"]
    assert torch.equal(p, initial) and p.requires_grad is False and p.grad is None
    repeated, _ = adapt([p], closure, lambda: p.clone(), source(), ids, dict(lr=.02))
    assert torch.equal(result, repeated)


@pytest.mark.parametrize("config", [dict(lr=0), dict(steps=0)])
def test_noop_controls_are_exact(config):
    p, ids, closure = toy()
    result, audit = adapt([p], closure, lambda: p.clone(), source(), ids, config)
    assert torch.equal(result, p)
    assert not audit["parameter_changed"] and audit["source_restored"]


def test_second_step_exception_restores_all_parameter_state():
    p, ids, closure = toy()
    p.requires_grad_(True)
    p.grad = torch.tensor([3., 4.])
    initial, gradient = p.detach().clone(), p.grad.clone()
    count = [0]
    def faulty():
        count[0] += 1
        if count[0] == 2:
            raise RuntimeError("injected second step")
        return closure()
    with pytest.raises(RuntimeError, match="injected"):
        adapt([p], faulty, lambda: p.clone(), source(), ids, dict(steps=2))
    assert torch.equal(p, initial) and torch.equal(p.grad, gradient) and p.requires_grad


def test_final_inference_exception_restores_state():
    p, ids, closure = toy()
    initial = p.detach().clone()
    def failure():
        raise RuntimeError("final failed")
    with pytest.raises(RuntimeError, match="final failed"):
        adapt([p], closure, failure, source(), ids)
    assert torch.equal(p, initial) and not p.requires_grad and p.grad is None


def test_feature_hooks_removed_on_exception():
    model = torch.nn.Sequential(torch.nn.LayerNorm(2))
    original = copy.copy(model[0]._forward_hooks)
    with pytest.raises(RuntimeError):
        with capture_layer_outputs(model, ["0"]) as captured:
            model(torch.ones(3, 2))
            assert len(captured["0"]) == 1
            raise RuntimeError("injected")
    assert model[0]._forward_hooks == original


def test_episodic_contract_rejects_continual_ema():
    p, ids, closure = toy()
    with pytest.raises(ValueError, match="momentum_mvg"):
        adapt([p], closure, lambda: p.clone(), source(), ids, dict(momentum_mvg=.1))
