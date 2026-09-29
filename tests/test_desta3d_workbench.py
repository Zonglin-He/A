"""CPU contracts for the manual entry, not a new research experiment."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from desta3d.cache import SealedCache, sha256
from desta3d.config import AdapterConfig, CacheConfig, DirectionConfig, TrainingConfig, ViewConfig, load_config
from desta3d.models import DirectionMixer, build_adapter, direction_loss, make_pixel_views
from desta3d.train_cached import fit
from vg_tta.desta3d_v3_a01_triage import FitabilityDirectionMixer
from vg_tta.desta3d_v3_a02_global_mean import coefficients as global_coefficients
from vg_tta.desta3d_v3_a0_screen import coefficients as local_coefficients
from vg_tta.desta3d_v3_gap_candidates import coefficient_direction_loss
from vg_tta.desta3d_v3_policy_gate_views import build_views

torch.set_num_threads(1)


def sample(time=3):
    return {
        "z": torch.randn(1, time, 2, 3, 8),
        "qT": torch.randn(1, 8), "qS": torch.randn(1, 8),
        "evidence8": torch.randn(1, time, 2, 3, 8),
        "state33": torch.randn(1, time, 33),
        "oracle_coeff256": torch.randn(1, time, 2, 3, 4),
    }


@pytest.mark.parametrize("global_mean", [False, True])
@pytest.mark.parametrize("width", [8, 16])
def test_refactor_preserves_initialization_forward_and_gradient(global_mean, width):
    basis = torch.eye(16, 4)
    cache = sample()
    torch.manual_seed(20260928)
    old = FitabilityDirectionMixer(basis, feature_dim=8, hidden_dim=width)
    torch.manual_seed(20260928)
    new = DirectionMixer(basis, DirectionConfig(feature_dim=8, hidden_dim=width, output_rank=4, global_mean=global_mean))
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(value, new.state_dict()[key]) for key, value in old.state_dict().items())
    legacy = global_coefficients(old, cache) if global_mean else local_coefficients(old, cache)
    actual = new(cache)
    assert torch.equal(actual, legacy)
    target = cache["oracle_coeff256"]
    old_loss = coefficient_direction_loss(legacy, target)["cosine"]
    new_loss = direction_loss(actual, target)
    assert torch.equal(old_loss, new_loss)
    old_loss.backward()
    new_loss.backward()
    for (name, a), (other, b) in zip(old.named_parameters(), new.named_parameters()):
        assert name == other
        assert torch.equal(a.grad, b.grad)
    assert new.basis.grad is None
    new.load_state_dict(old.state_dict(), strict=True)


def test_fixed_norm_and_zero_projection():
    model = DirectionMixer(torch.eye(16, 4), DirectionConfig(feature_dim=8, hidden_dim=8, output_rank=4))
    fields = torch.randn(2, 3, 2, 2, 4)
    norms = torch.tensor([2.0, 5.0])
    result = model.project(fields, norms)
    torch.testing.assert_close(result.flatten(1).norm(dim=1), norms * model.radius)
    assert torch.count_nonzero(model.project(torch.zeros_like(fields), norms)) == 0
    with pytest.raises(ValueError):
        model.project(fields, torch.tensor([-1.0, 5.0]))


def test_real_width_r16_initialization_and_forward():
    from vg_tta.desta3d_v3_a04_factorized import FactorizedDirectionMixer
    union = torch.eye(2560, 256)
    channel = torch.eye(256, 16, dtype=torch.float64)
    data = sample()
    data["z"] = torch.randn(1, 3, 2, 3, 128)
    data["qT"], data["qS"] = torch.randn(1, 128), torch.randn(1, 128)
    torch.manual_seed(20260928)
    old = FactorizedDirectionMixer(union, channel)
    torch.manual_seed(20260928)
    new = DirectionMixer((union.double() @ channel).float(), DirectionConfig(output_rank=16))
    assert sum(p.numel() for p in new.parameters()) == 76688
    assert new.input.in_features == 425
    assert all(torch.equal(value, old.state_dict()[key]) for key, value in new.state_dict().items())
    assert torch.equal(new(data), old(data))


def test_adapter_tta_scope_and_pixel_parameter_wiring():
    frozen = build_adapter(AdapterConfig())
    assert frozen.active_parameter_count() == 0
    model = build_adapter(AdapterConfig(train_stage="tta"))
    assert model.active_parameter_count() == 66816
    assert not model.gate_event.requires_grad and not model.gate_spatial.requires_grad
    with_gates = build_adapter(AdapterConfig(train_stage="tta", freeze_tta_gates=False))
    assert with_gates.active_parameter_count() == 66818
    frames = np.arange(3 * 8 * 8 * 3, dtype=np.uint8).reshape(3, 8, 8, 3)
    from vg_tta.external_privileged_views import parse_teacher_text
    evidence = parse_teacher_text("{.5,.5}", [1, 2, 3])
    # Missing spatial support leaves frames unchanged; temporal dim is active.
    actual, meta = make_pixel_views(frames, [1, 2, 3], evidence, ViewConfig(0.5, 4.0))
    expected, old_meta = build_views(frames, [1, 2, 3], evidence, dim=0.5, radius=4.0)
    assert meta == old_meta
    assert all(np.array_equal(actual[k], expected[k]) for k in actual)
    assert not np.array_equal(actual["B1"], actual["T"])


@pytest.mark.parametrize("change", [
    {"oops": 1}, {"training": {"learning_rate": -1}},
    {"training": {"steps": True}}, {"model": {"global_mean": "false"}},
    {"training": {"minimum_free_gib": 7}}, {"model": {"radius": float("nan")}},
])
def test_config_rejects_typos_and_invalid_parameters(tmp_path, change):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"version": 1, "kind": "cached_direction", **change}))
    with pytest.raises(ValueError):
        load_config(path)


def fixture_cache(workspace):
    root = workspace / "cached"
    files = {}
    for split, count in (("train", 2), ("dev", 1)):
        for index in range(count):
            path = root / "cache" / split / f"{index:04}" / "CACHE.pt"
            path.parent.mkdir(parents=True)
            torch.save(sample(3 + index), path)
            files[str(path.relative_to(root))] = sha256(path)
    seal = root / "CACHE_SEAL.json"
    seal.write_text(json.dumps({"files": files}))
    basis = root / "BASIS.pt"
    torch.save(torch.eye(16, 4), basis)
    return asdict(CacheConfig(root="cached", seal_sha256=sha256(seal), basis_file="cached/BASIS.pt", basis_sha256=sha256(basis)))


def test_manual_cpu_fit_seal_counter_and_overwrite_control(tmp_path):
    config = {
        "version": 1, "kind": "cached_direction",
        "model": asdict(DirectionConfig(feature_dim=8, hidden_dim=8, output_rank=4)),
        "training": asdict(TrainingConfig(device="cpu", steps=2, batch_size=2, threads=1)),
        "cache": fixture_cache(tmp_path),
    }
    destination = fit(config, tmp_path, "synthetic")
    report = json.loads((destination / "REPORT.json").read_text())
    assert report["steps"] == 2 and report["results"]["dev"]["defined"] == 1
    assert report["native_utility_measured"] is False
    final = torch.load(destination / "FINAL.pt", weights_only=True)
    assert all(type(key) is int for key in final["optimizer"]["state"])
    assert all(int(value["step"]) == 2 for value in final["optimizer"]["state"].values())
    for relative, expected in json.loads((destination / "SEAL.json").read_text())["files"].items():
        assert sha256(destination / relative) == expected
    assert json.loads((destination / "RECEIPT.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError):
        fit(config, tmp_path, "synthetic")
    with pytest.raises(ValueError):
        fit(config, tmp_path, "../escape")
    first = tmp_path / "cached/cache/train/0000/CACHE.pt"
    first.write_bytes(first.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA256"):
        SealedCache(config["cache"], tmp_path)


def test_input_gradients_detached_and_zero_oracle():
    cache = sample()
    for value in cache.values():
        value.requires_grad_()
    model = DirectionMixer(torch.eye(16, 4), DirectionConfig(feature_dim=8, hidden_dim=8, output_rank=4))
    loss = direction_loss(model(cache), cache["oracle_coeff256"])
    loss.backward()
    assert all(value.grad is None for value in cache.values())
    assert direction_loss(model(cache), torch.zeros_like(cache["oracle_coeff256"])) == 0
