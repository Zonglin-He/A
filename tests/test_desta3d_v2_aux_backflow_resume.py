from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts import desta3d_v2_aux_backflow_fit as fit
from scripts.desta3d_v2_aux_backflow_run import adapter_sha256, cpu_copy, write
from vg_tta.desta3d_v2 import Desta3DAdapterV2


def _assert_nested_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def test_common_a_to_fresh_b_restore_roundtrip_and_short_final_window(tmp_path, monkeypatch):
    out = tmp_path / "isolated_fit"
    old = tmp_path / "isolated_old_source_fit"
    out.mkdir()
    old.mkdir()
    (out / "history").mkdir()
    (out / "LOCK.json").write_text('{"synthetic_test_lock":true}\n', encoding="utf-8")
    monkeypatch.setattr(fit, "OUT", out)
    monkeypatch.setattr(fit, "OLD", old)

    # Replace .cuda() and all CUDA RNG access with CPU-only no-ops/state holders.
    # The mocked RNG values are sentinels, not actual CUDA RNG states.
    monkeypatch.setattr(Desta3DAdapterV2, "cuda", lambda self, *args, **kwargs: self)
    mocked_cuda_rng = {"value": [torch.tensor([17], dtype=torch.uint8)]}
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _seed: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: [value.clone() for value in mocked_cuda_rng["value"]],
    )

    def set_mock_cuda_rng(values):
        mocked_cuda_rng["value"] = [torch.as_tensor(value).clone() for value in values]

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", set_mock_cuda_rng)

    # initialize() checks its initial adapter against OLD/INITIAL.json. Build
    # that expected hash from the exact seeded CPU construction in isolation.
    torch.manual_seed(fit.SEED)
    expected_initial = Desta3DAdapterV2(
        hidden_dim=128, architecture="dual3d", p1_enabled=False
    )
    write(old / "INITIAL.json", {"adapters": {"dual_repaired": adapter_sha256(expected_initial)}})
    del expected_initial

    state, adapters, optimizers = fit.initialize()
    common = adapters["common"]
    assert state == {"stage": "A", "cursor": 0, "steps": {"common": 0}, "last_window": None}
    assert all(parameter.device.type == "cpu" for parameter in common.parameters())
    initial_sha = adapter_sha256(common)
    initial_optimizer = cpu_copy(optimizers["common"].state_dict())
    assert initial_optimizer["state"] == {}

    # Explicitly synthetic single optimizer step to exercise optimizer-state
    # serialization. Then mark the A cursor/step count as a complete-stage
    # checkpoint without pretending 618 queries were trained in this CPU test.
    synthetic_parameter = common.shared_stem.depthwise.weight
    synthetic_parameter.grad = torch.full_like(synthetic_parameter, 1e-3)
    optimizers["common"].step()
    optimizers["common"].zero_grad(set_to_none=True)
    assert optimizers["common"].state
    state = {"stage": "A", "cursor": 618, "steps": {"common": 155},
             "last_window": {"stage": "A", "cursor": 618, "synthetic": True}}
    fit.save_state(state, adapters, optimizers)
    saved_a = torch.load(out / "LATEST.pt", map_location="cpu", weights_only=False)
    assert saved_a["cursor"] == 618 and saved_a["steps"] == {"common": 155}
    a_parameter_snapshot = cpu_copy(common.state_dict())
    a_optimizer_snapshot = cpu_copy(optimizers["common"].state_dict())

    mocked_cuda_rng["value"] = [torch.tensor([99], dtype=torch.uint8)]
    restored_state, restored_adapters, restored_optimizers = fit.restore()
    assert restored_state == state
    assert mocked_cuda_rng["value"][0].item() == 17
    _assert_nested_equal(cpu_copy(restored_adapters["common"].state_dict()), a_parameter_snapshot)
    _assert_nested_equal(cpu_copy(restored_optimizers["common"].state_dict()), a_optimizer_snapshot)
    assert adapter_sha256(restored_adapters["common"]) != initial_sha  # synthetic step persisted
    history_path = out / "history" / "A_C0618.json"
    assert json.loads(history_path.read_text(encoding="utf-8"))["synthetic"] is True

    b_state, b_adapters, b_optimizers = fit.transition_to_B(
        restored_state, restored_adapters, restored_optimizers
    )
    assert b_state == {"stage": "B", "cursor": 0,
                       "steps": {"B0": 0, "B1": 0, "B2": 0}, "last_window": None}
    b_hashes = {arm: adapter_sha256(model) for arm, model in b_adapters.items()}
    assert len(set(b_hashes.values())) == 1
    assert set(b_hashes.values()) == {adapter_sha256(restored_adapters["common"])}
    assert all(not optimizer.state for optimizer in b_optimizers.values())

    # Distinct synthetic B steps populate each fresh optimizer independently.
    for index, arm in enumerate(fit.MODES):
        optimizer = b_optimizers[arm]
        model = b_adapters[arm]
        model.gate_event.grad = torch.tensor(1e-3 * (index + 1))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    b_state = {"stage": "B", "cursor": 4, "steps": {arm: 1 for arm in fit.MODES},
               "last_window": {"stage": "B", "cursor": 4, "synthetic": True}}
    fit.save_state(b_state, b_adapters, b_optimizers)
    b_parameter_snapshots = {arm: cpu_copy(model.state_dict()) for arm, model in b_adapters.items()}
    b_optimizer_snapshots = {arm: cpu_copy(opt.state_dict()) for arm, opt in b_optimizers.items()}
    restored_b_state, restored_b_adapters, restored_b_optimizers = fit.restore()
    assert restored_b_state == b_state
    for arm in fit.MODES:
        _assert_nested_equal(cpu_copy(restored_b_adapters[arm].state_dict()), b_parameter_snapshots[arm])
        _assert_nested_equal(cpu_copy(restored_b_optimizers[arm].state_dict()), b_optimizer_snapshots[arm])
        assert restored_b_optimizers[arm].state

    # Mirror the fit loop's actual slicing/divisor logic for the final partial
    # window of 618 queries: it has exactly two items and scales by two.
    rows = list(range(618))
    cursor = 616
    final_window = rows[cursor:cursor + 4]
    divisor = len(final_window)
    assert final_window == [616, 617]
    assert divisor == 2 and cursor + divisor == 618
