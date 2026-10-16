"""Native-instance spatial fit adapters for the fixed TA-STVG student.

The Direct and unweighted IoU-OPD arms call the existing fit paths. The third
arm changes only the detached Gaussian-policy reward to ``r_t * IoU``. This
module is inert on import: it never opens model assets, datasets, pixels, or
annotations. Its contracts instantiate only the repository's synthetic CPU
decoder fixture.
"""
from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALGORITHMS = {"direct", "iou", "reliability_weighted_iou"}
FIT_STEPS = 5
FIT_SCOPE = "joint"
ANCHOR_LIMIT = 4
RELIABILITY_ARM = "SpatialReliabilityWeightedIoUOPD"

# The existing IoU helper verifies the frozen OPD/core/config dependencies.
# These additional pins cover its adapter code and the original Direct loss.
SOURCE_BINDINGS = {
    "scripts/stvg_video_global_spatial_fit_fp32_geometry001.py":
        "cd5f9c723c014b367db8c5b5d2e97d5aa1f688f0d095712fbf4a06afd302de53",
    "methods/decota_final_simplified_v1/objectives.py":
        "c2dc28e23e4213ff2559482f7362e32bec81e767786b5b53c4501dd6017aaeaf",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_bindings() -> dict[str, str]:
    """Verify the original OPD pins plus the local Direct/adapter pins."""
    from scripts import stvg_video_global_spatial_fit_fp32_geometry001 as opd

    actual = opd.verify_source_bindings()
    for relative, expected in SOURCE_BINDINGS.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required spatial-fit source is missing: {relative}")
        digest = _sha256_file(path)
        if digest != expected:
            raise RuntimeError(f"native-instance spatial dependency SHA differs: {relative}")
        actual[relative] = digest
    return actual


def _validate_key_and_config(key: str, config: Mapping[str, Any], scope: str,
                             reference_role: str, reference_sha256: str):
    from scripts import stvg_video_global_spatial_fit_fp32_geometry001 as opd

    prefix = "caption_query_fixed_space|"
    if not isinstance(key, str) or not key.startswith(prefix) or not key[len(prefix):]:
        raise ValueError("key must preserve caption_query_fixed_space|<case_id>")
    case_id = key[len(prefix):]
    if "|" in case_id or opd.locked_seed_key(case_id) != key:
        raise ValueError("Gaussian key differs from the original fixed spatial key")
    dataset = opd._dataset_matching_config(config)
    chosen, expected_key = opd._validate_fit_request(
        case_id=case_id, dataset=dataset, reference_role=reference_role,
        reference_sha256=reference_sha256, objective="iou", config=config,
        scope=scope)
    if expected_key != key or chosen != dict(config):
        raise ValueError("fit key/config is not the fixed original five-step setting")
    return case_id, dataset, chosen


def _validate_anchors(anchors: Sequence[Mapping[str, Any]], frame_ids: Sequence[int], torch):
    if not isinstance(anchors, (list, tuple)) or len(anchors) > ANCHOR_LIMIT:
        raise ValueError("anchors must contain zero to four sealed sparse observations")
    if not isinstance(frame_ids, (list, tuple)) or not frame_ids:
        raise ValueError("frame_ids must be the locked nonempty physical frame grid")
    ids: list[int] = []
    for value in frame_ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("physical frame IDs must be integers")
        ids.append(int(value))
    if any(right <= left for left, right in zip(ids, ids[1:])):
        raise ValueError("physical frame IDs must be strictly increasing")

    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in anchors:
        if not isinstance(raw, Mapping):
            raise ValueError("each reference anchor must be a mapping")
        position = raw.get("position")
        if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < len(ids):
            raise ValueError("anchor position is outside the fixed physical frame grid")
        if position in seen:
            raise ValueError("anchor positions must be unique")
        seen.add(int(position))
        physical_id = raw.get("physical_frame_id")
        if (isinstance(physical_id, bool) or not isinstance(physical_id, int)
                or int(physical_id) != ids[position]):
            raise ValueError("anchor physical frame binding differs from position")
        reliability = raw.get("weight", 1.0)
        if (isinstance(reliability, bool) or not isinstance(reliability, (int, float))
                or not math.isfinite(float(reliability)) or not 0.0 <= float(reliability) <= 1.0):
            raise ValueError("anchor reliability weight must be finite and in [0,1]")
        box64 = torch.as_tensor(raw.get("box")).detach().cpu().to(dtype=torch.float64)
        if box64.shape != (4,) or not bool(torch.isfinite(box64).all()):
            raise ValueError("anchor box must be a finite normalized cxcywh vector")
        cx, cy, width, height = (float(value) for value in box64.tolist())
        x1, y1 = cx - width / 2.0, cy - height / 2.0
        x2, y2 = cx + width / 2.0, cy + height / 2.0
        if not (width > 0.0 and height > 0.0
                and -1e-7 <= x1 < x2 <= 1.0 + 1e-7
                and -1e-7 <= y1 < y2 <= 1.0 + 1e-7):
            raise ValueError("anchor box must be legal normalized xyxy geometry encoded as cxcywh")
        output.append({**dict(raw), "position": int(position),
                       "physical_frame_id": int(physical_id),
                       "box": box64.to(dtype=torch.float32),
                       "weight": float(reliability)})
    return sorted(output, key=lambda row: row["position"])


def _unit_anchors(anchors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep original Direct/IoU mathematics: all admitted anchors are unit weight."""
    return [{**dict(anchor), "weight": 1.0} for anchor in anchors]


def _reliability_rollout(mean, evidence, reliability, key: str, round_index: int,
                         config: Mapping[str, Any]):
    """Use the original Gaussian samples and scale each detached IoU reward by r_t."""
    import torch
    import vg_tta.decota_spatial_opd_tunable_v1 as policy_math

    policy = policy_math.rollout(mean, evidence, key, round_index,
                                 shuffle=False, config=config)
    raw_rewards = policy["rewards"].detach()
    reliability = torch.as_tensor(reliability, dtype=raw_rewards.dtype,
                                  device=raw_rewards.device).detach()
    if reliability.shape != (raw_rewards.shape[0],):
        raise ValueError("one reliability scalar is required per active observation")
    if (not bool(torch.isfinite(reliability).all())
            or bool(((reliability < 0) | (reliability > 1)).any())):
        raise ValueError("reliability values must be finite and in [0,1]")
    # Do not divide by sum(r) or renormalize across anchors. Scaling before the
    # fixed-temperature softmax preserves the intended reliability effect.
    used_rewards = (raw_rewards * reliability[:, None]).detach()
    weights = torch.softmax(used_rewards / float(config["tau"]), dim=-1).detach()
    informative = bool((used_rewards != used_rewards[:, :1]).any())
    return {**policy,
            "raw_rewards": raw_rewards,
            "used_rewards": used_rewards,
            "weights": weights,
            "informative": informative,
            "anchor_reliability": reliability}


def _fit_reliability_weighted_iou(base, initial, anchors, frame_ids, key, config, *,
                                  reference_role: str, reference_sha256: str,
                                  resume_state=None,
                                  checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None):
    """Original five-step Gaussian IoU OPD with detached per-anchor r_t scaling."""
    import torch
    from methods.decota_final_simplified_v1.tensors import detached
    from scripts import stvg_target_teacher_student_v1 as core
    from vg_tta.decota_spatial_opd_tunable_v1 import (
        likelihood_loss, mean_coordinates, optimizer_step,
    )

    fit_started = time.perf_counter()
    base.restore(initial)
    replay, selected = core._trainable_replay(base, list(frame_ids), key, FIT_SCOPE)
    origin = replay.state()
    params = [parameter for _, parameter in selected]
    if sum(parameter.numel() for parameter in params) != 1792:
        raise RuntimeError("joint OPD parameter count differs from the original 1,792 parameters")

    active = [anchor for anchor in anchors if float(anchor["weight"]) > 0.0]
    positions = [int(anchor["position"]) for anchor in active]
    if len(set(positions)) != len(positions) or any(not 0 <= p < len(frame_ids) for p in positions):
        raise ValueError("Invalid reliability-weighted observation positions")
    if active:
        evidence = torch.stack([anchor["box"].detach().to(base.zero["boxes"]) for anchor in active])
        reliability = torch.as_tensor([anchor["weight"] for anchor in active],
                                      dtype=evidence.dtype, device=evidence.device).detach()
    else:
        evidence = base.zero["boxes"].new_empty((0, 4))
        reliability = base.zero["boxes"].new_empty((0,))

    optimizer = torch.optim.Adam(params, lr=config["lr"], betas=(.9, .999),
                                 eps=1e-8, weight_decay=0.)
    binding = {"algorithm": "reliability_weighted_iou",
               "reference_role": reference_role,
               "reference_sha256": reference_sha256,
               "reward": "r_t_times_original_aligned_iou"}
    if resume_state is not None and resume_state.get("fit_binding") != binding:
        raise ValueError("resume checkpoint belongs to another weighted-IoU reference or algorithm")
    next_step, path, rounds, optimizer_steps = core._initialize_fit_resume(
        replay, optimizer, resume_state, arm=RELIABILITY_ARM, scope=FIT_SCOPE,
        key=key, frame_ids=list(frame_ids), config=config, origin=origin)

    def publish_progress(next_index: int) -> None:
        if checkpoint_callback is not None:
            progress = core._fit_progress(
                replay, optimizer, arm=RELIABILITY_ARM, scope=FIT_SCOPE, key=key,
                frame_ids=list(frame_ids), config=config, origin=origin,
                next_step=next_index, path=path, rounds=rounds,
                optimizer_steps=optimizer_steps)
            progress["fit_binding"] = dict(binding)
            checkpoint_callback(progress)

    try:
        for step in range(next_step, int(config["steps"])):
            step_started = time.perf_counter()
            before_state = replay.state()
            before_values = replay.values()
            before_boxes = before_values["boxes"].detach().clone()
            entry = {"step": step, "boxes": before_boxes.detach().cpu(),
                     "state": detached(before_state, "cpu"),
                     "optimizer_state": core._copy_cpu(optimizer.state_dict()),
                     "rng_state_before": core._rng_snapshot()}
            if not positions:
                entry.update(skipped=True, reason="zero_or_missing_reliability_evidence",
                             step_wall_seconds=time.perf_counter() - step_started)
                rounds.append({"step": step, "no_information": True, "updated": False,
                               "active_observations": 0,
                               "wall_seconds": time.perf_counter() - step_started})
                path.append(entry)
                publish_progress(step + 1)
                continue

            mean = mean_coordinates(before_values["boxes"][positions])
            policy = _reliability_rollout(mean, evidence, reliability, key, step, config)
            loss = likelihood_loss(mean, policy["mean"], policy["samples"],
                                   policy["weights"], sigma=config["sigma"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite reliability-weighted Gaussian likelihood loss")
            policy_mean_box = torch.sigmoid(
                (policy["weights"][..., None] * policy["samples"]).sum(-2))
            policy_delta = policy_mean_box - before_boxes[positions].detach()
            optimizer_before = core._copy_cpu(optimizer.state_dict())
            before_params = detached(replay.state(), "cpu")
            update = optimizer_step(optimizer, loss, params, bool(policy["informative"]))
            after_state = replay.state()
            optimizer_after = core._copy_cpu(optimizer.state_dict())
            if update is None:
                if not core._tree_equal(before_params, detached(after_state, "cpu")):
                    raise AssertionError("Flat weighted rewards changed student parameters")
                if not core._tree_equal(optimizer_before, optimizer_after):
                    raise AssertionError("Flat weighted rewards changed Adam state")
            else:
                optimizer_steps += 1

            with torch.no_grad():
                after_boxes = replay.values()["boxes"].detach().clone()
            tangent = core._state_delta(before_state, after_state)
            expected_move = core._decoder_jvp(replay, list(before_state.values()),
                                              list(tangent.values()))
            actual_move = after_boxes - before_boxes
            unobserved = [index for index in range(len(frame_ids)) if index not in set(positions)]
            round_record = {
                "step": step,
                "updated": update is not None,
                "wall_seconds": time.perf_counter() - step_started,
                "no_information": not bool(policy["informative"]),
                "loss": float(loss.detach()),
                "reward_objective": "reliability_weighted_iou",
                "active_positions": list(positions),
                "anchor_reliability": reliability.detach().cpu(),
                "raw_iou_reward_variance": policy["raw_rewards"].var(1, unbiased=False).detach().cpu(),
                "used_reward_variance": policy["used_rewards"].var(1, unbiased=False).detach().cpu(),
                "weight_entropy": (-(policy["weights"] * policy["weights"].clamp_min(1e-30).log()).sum(1)).detach().cpu(),
                "weight_ess": (1.0 / policy["weights"].square().sum(1)).detach().cpu(),
                "expected_policy_delta_observed": policy_delta.detach().cpu(),
                "expected_delta_all_positions": expected_move.detach().cpu(),
                "actual_delta_all_positions": actual_move.detach().cpu(),
                "observed_movement": core._movement_summary(expected_move, actual_move, positions),
                "unobserved_movement": core._movement_summary(expected_move, actual_move, unobserved),
                "actions_detached": not policy["samples"].requires_grad,
                "teacher_targets_detached": not evidence.requires_grad,
                "reward_weights_detached": not policy["weights"].requires_grad,
            }
            rounds.append(round_record)
            entry.update(skipped=False, loss=float(loss.detach()),
                         rollout=detached(policy, "cpu"),
                         step_wall_seconds=time.perf_counter() - step_started,
                         optimizer_before=optimizer_before,
                         optimizer_state=optimizer_after,
                         update=None if update is None else detached(update, "cpu"),
                         update_state=detached(after_state, "cpu"),
                         expected_delta=expected_move.detach().cpu(),
                         actual_delta=actual_move.detach().cpu())
            path.append(entry)
            publish_progress(step + 1)

        final_state = replay.state()
        with torch.no_grad():
            final_cached = replay.values()["boxes"].detach().cpu()
        return {"schema": "ta-stvg-student-fit-v1",
                "arm": RELIABILITY_ARM,
                "scope": FIT_SCOPE,
                "initial_state": detached(origin, "cpu"),
                "final_state": detached(final_state, "cpu"),
                "final_cached_decoder_boxes": final_cached,
                "path": path,
                "rounds": rounds,
                "optimizer_steps": optimizer_steps,
                "selected_step": int(config["steps"]),
                "declared_steps": int(config["steps"]),
                "active_parameters": 1792,
                "config": dict(config),
                "wall_seconds": time.perf_counter() - fit_started,
                "observation_positions": list(positions),
                "input_observation_positions": [int(anchor["position"]) for anchor in anchors],
                "input_reliability": [float(anchor["weight"]) for anchor in anchors],
                "policy": "original antithetic on-policy Gaussian IoU OPD",
                "reward_objective": "r_t_times_original_iou",
                "reliability_normalized_across_anchors": False,
                "teacher_box_in_action_pool": False,
                "reference_role": reference_role,
                "reference_sha256": reference_sha256,
                "seed_key": key,
                "seed_spec": "sha256('spatial-opd-v1|{key}|{step}')[:15] as base-16",
                "GT_read": False,
                "output_source": "student native decoder readout required"}
    finally:
        replay.restore(origin)
        base.restore(initial)


def fit_spatial_instance(base, initial: Mapping[str, Any],
                         anchors: Sequence[Mapping[str, Any]], frame_ids: Sequence[int],
                         key: str, config: Mapping[str, Any], *, algorithm: str,
                         reference_role: str = "teacher",
                         reference_sha256: str | None = None,
                         scope: str = "joint", resume_state: Mapping[str, Any] | None = None,
                         checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    """Fit Direct, original IoU OPD, or reliability-weighted IoU OPD.

    ``anchors`` carry source-grid ``position``, matching ``physical_frame_id``,
    normalized ``cxcywh`` ``box`` and optional reliability ``weight`` in [0,1].
    The Direct and original-IoU arms deliberately retain the original unit
    anchor loss/reward; only ``reliability_weighted_iou`` applies ``r_t``.
    """
    import torch
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts import stvg_video_global_spatial_fit_fp32_geometry001 as opd

    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {sorted(ALGORITHMS)}")
    if scope != FIT_SCOPE:
        raise ValueError("this helper is fixed to the original joint 1,792-parameter scope")
    verify_source_bindings()
    checked = _validate_anchors(anchors, frame_ids, torch)
    case_id = key.split("|", 1)[1] if isinstance(key, str) and "|" in key else ""
    if not case_id:
        raise ValueError("key must include the original caption_query_fixed_space case key")
    if reference_sha256 is None:
        reference_sha256 = opd._anchor_payload_sha256(checked)
    dataset = opd._dataset_matching_config(config)
    chosen_config, expected_key = opd._validate_fit_request(
        case_id=case_id, dataset=dataset, reference_role=reference_role,
        reference_sha256=reference_sha256, objective="iou", config=config,
        scope=scope)
    if expected_key != key:
        raise ValueError("key differs from the original locked spatial seed key")

    reliability = [float(anchor["weight"]) for anchor in checked]
    binding = {"algorithm": algorithm, "reference_role": reference_role,
               "reference_sha256": reference_sha256}
    if resume_state is not None and algorithm in {"direct", "reliability_weighted_iou"}:
        if resume_state.get("fit_binding") != binding and algorithm == "direct":
            raise ValueError("Direct resume checkpoint belongs to another arm/reference")

    if algorithm == "direct":
        unit = _unit_anchors(checked)
        original_callback = checkpoint_callback
        direct_callback = None
        if original_callback is not None:
            def callback_with_binding(progress: Mapping[str, Any]) -> None:
                payload = dict(progress)
                payload["fit_binding"] = dict(binding)
                original_callback(payload)
            direct_callback = callback_with_binding
        fit = core._fit_direct(
            base, initial, unit, list(frame_ids), key, chosen_config, scope=scope,
            resume_state=resume_state, checkpoint_callback=direct_callback)
        fit.update({"algorithm": algorithm, "reference_role": reference_role,
                    "reference_sha256": reference_sha256,
                    "reference_reliability": reliability,
                    "reliability_applied": False})
    elif algorithm == "iou":
        unit = _unit_anchors(checked)
        fit = opd.fit_spatial_reward_opd(
            base, initial, unit, list(frame_ids), key, chosen_config,
            scope=scope, reward_kind="iou", checkpoint_callback=checkpoint_callback,
            reference_role=reference_role, reference_sha256=reference_sha256,
            resume_state=resume_state)
        fit.update({"algorithm": algorithm,
                    "reference_reliability": reliability,
                    "reliability_applied": False})
    else:
        fit = _fit_reliability_weighted_iou(
            base, initial, checked, list(frame_ids), key, chosen_config,
            reference_role=reference_role, reference_sha256=reference_sha256,
            resume_state=resume_state, checkpoint_callback=checkpoint_callback)
        fit.update({"algorithm": algorithm,
                    "reference_reliability": reliability,
                    "reliability_applied": True})
    return fit


def _tree_equal(left: Any, right: Any) -> bool:
    import numpy as np
    import torch
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right)
                and left.dtype == right.dtype and left.device.type == right.device.type
                and torch.equal(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and np.array_equal(left, right))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping)
                and set(left) == set(right)
                and all(_tree_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_tree_equal(a, b) for a, b in zip(left, right)))
    return left == right


def _without_timing(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _without_timing(child) for key, child in value.items()
                if key not in {"wall_seconds", "step_wall_seconds"}}
    if isinstance(value, list):
        return [_without_timing(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_without_timing(child) for child in value)
    return value


def run_cpu_contracts() -> dict[str, Any]:
    """Run synthetic CPU contracts only; no real source, model asset, GT, or GPU."""
    import torch
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts import stvg_video_global_spatial_fit_fp32_geometry001 as opd
    import vg_tta.decota_spatial_opd_tunable_v1 as policy_math

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU contracts require CUDA to remain uninitialized")
    bindings = verify_source_bindings()
    frame_ids = list(range(8))
    config = core.load_original_opd_config("hc2")
    config["steps"] = FIT_STEPS
    key = opd.locked_seed_key("native_instance_synthetic")
    reference_sha = "a" * 64
    base = core._synthetic_student_base(len(frame_ids))
    initial = base.initial
    anchor = {"position": 2, "physical_frame_id": frame_ids[2],
              "box": torch.tensor([.53, .49, .28, .30]), "weight": .35}

    # New API Direct must preserve the original geometric loss path exactly.
    direct = fit_spatial_instance(
        base, initial, [anchor], frame_ids, key, config, algorithm="direct",
        reference_role="teacher", reference_sha256=reference_sha)
    direct_original = core._fit_direct(
        base, initial, _unit_anchors([anchor]), frame_ids, key, config, scope=FIT_SCOPE)
    direct_fields = ("final_state", "final_cached_decoder_boxes", "path",
                     "optimizer_steps", "selected_step", "active_parameters", "config")
    assert all(_tree_equal(_without_timing(direct[field]),
                           _without_timing(direct_original[field])) for field in direct_fields)

    # Exercise the real Direct checkpoint callback boundary through a CPU
    # serialization roundtrip, then resume the one-step checkpoint and verify
    # it reconstructs the same complete five-step path as an uninterrupted fit.
    import io
    class _StopAfterCheckpoint(Exception):
        pass

    serialized_checkpoints: list[bytes] = []

    def checkpoint_roundtrip_callback(payload: Mapping[str, Any]) -> None:
        buffer = io.BytesIO()
        torch.save(dict(payload), buffer)
        serialized_checkpoints.append(buffer.getvalue())
        raise _StopAfterCheckpoint

    try:
        fit_spatial_instance(
            base, initial, [anchor], frame_ids, key, config, algorithm="direct",
            reference_role="teacher", reference_sha256=reference_sha,
            checkpoint_callback=checkpoint_roundtrip_callback)
    except _StopAfterCheckpoint:
        pass
    else:
        raise AssertionError("Direct checkpoint callback was not invoked")
    assert len(serialized_checkpoints) == 1
    checkpoint = torch.load(io.BytesIO(serialized_checkpoints[0]),
                            map_location="cpu", weights_only=False)
    direct_binding = {"algorithm": "direct", "reference_role": "teacher",
                      "reference_sha256": reference_sha}
    assert checkpoint["fit_binding"] == direct_binding
    assert checkpoint["next_step"] == 1
    assert checkpoint["path"] and checkpoint["path"][-1]["step"] == 0
    direct_resumed = fit_spatial_instance(
        base, initial, [anchor], frame_ids, key, config, algorithm="direct",
        reference_role="teacher", reference_sha256=reference_sha,
        resume_state=checkpoint)
    assert all(_tree_equal(_without_timing(direct_resumed[field]),
                           _without_timing(direct[field])) for field in direct_fields)

    # The unweighted arm is the old IoU helper/core path, byte-for-byte apart
    # from wrapper metadata and timing.
    iou = fit_spatial_instance(
        base, initial, [anchor], frame_ids, key, config, algorithm="iou",
        reference_role="teacher", reference_sha256=reference_sha)
    iou_original = core._fit_spatial_opd(
        base, initial, _unit_anchors([anchor]), frame_ids, key, config, scope=FIT_SCOPE)
    iou_fields = ("initial_state", "final_state", "final_cached_decoder_boxes", "path",
                  "rounds", "optimizer_steps", "selected_step", "active_parameters", "config")
    assert all(_tree_equal(_without_timing(iou[field]),
                           _without_timing(iou_original[field])) for field in iou_fields)

    # Formula contract: scale rewards before tau-softmax; do not normalize r
    # away across anchors or apply it after the policy distribution is formed.
    mean = policy_math.mean_coordinates(base.zero["boxes"][[2]])
    evidence = anchor["box"].to(dtype=base.zero["boxes"].dtype).reshape(1, 4)
    policy = _reliability_rollout(mean, evidence, [anchor["weight"]], key, 0, config)
    expected_weights = torch.softmax(
        policy["raw_rewards"] * anchor["weight"] / config["tau"], dim=-1)
    unweighted_weights = torch.softmax(policy["raw_rewards"] / config["tau"], dim=-1)
    assert torch.equal(policy["weights"], expected_weights)
    assert float((policy["weights"] - unweighted_weights).abs().max()) > 1e-6
    assert torch.allclose(policy["weights"].sum(-1), torch.ones(1), atol=1e-6, rtol=0)

    weighted = fit_spatial_instance(
        base, initial, [anchor], frame_ids, key, config,
        algorithm="reliability_weighted_iou", reference_role="teacher",
        reference_sha256=reference_sha)
    assert len(weighted["rounds"]) == FIT_STEPS
    assert weighted["active_parameters"] == 1792
    assert weighted["reliability_applied"] is True
    assert weighted["optimizer_steps"] > 0
    assert not _tree_equal(weighted["final_state"], iou["final_state"])

    # At r=1, the reliability arm must reduce exactly to original IoU OPD.
    # Compare full model states, every step's gradients/Adam states, sampled
    # rewards and policy weights, and the cached decoder readout.
    unit_anchor = {**anchor, "weight": 1.0}
    weighted_r1 = fit_spatial_instance(
        base, initial, [unit_anchor], frame_ids, key, config,
        algorithm="reliability_weighted_iou", reference_role="teacher",
        reference_sha256=reference_sha)
    iou_r1 = core._fit_spatial_opd(
        base, initial, _unit_anchors([unit_anchor]), frame_ids, key, config,
        scope=FIT_SCOPE)
    for field in ("initial_state", "final_state", "final_cached_decoder_boxes",
                  "optimizer_steps", "selected_step", "active_parameters", "config"):
        assert _tree_equal(_without_timing(weighted_r1[field]),
                           _without_timing(iou_r1[field])), f"r=1 mismatch in {field}"

    def _common_tree(left: Any, right: Any) -> tuple[Any, Any]:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            keys = set(left) & set(right)
            return ({key: _common_tree(left[key], right[key])[0] for key in keys},
                    {key: _common_tree(left[key], right[key])[1] for key in keys})
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                return left, right
            pairs = [_common_tree(a, b) for a, b in zip(left, right)]
            return [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        if isinstance(left, tuple) and isinstance(right, tuple):
            if len(left) != len(right):
                return left, right
            pairs = [_common_tree(a, b) for a, b in zip(left, right)]
            return tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)
        return left, right

    weighted_path_common, iou_path_common = _common_tree(
        _without_timing(weighted_r1["path"]), _without_timing(iou_r1["path"]))
    assert _tree_equal(weighted_path_common, iou_path_common), "r=1 per-step path mismatch"
    weighted_rounds_common, iou_rounds_common = _common_tree(
        _without_timing(weighted_r1["rounds"]), _without_timing(iou_r1["rounds"]))
    assert _tree_equal(weighted_rounds_common, iou_rounds_common), "r=1 per-step gradient/update mismatch"
    for weighted_entry, original_entry in zip(weighted_r1["path"], iou_r1["path"]):
        if weighted_entry.get("skipped", False):
            continue
        assert {"gradient", "raw"}.issubset(weighted_entry["update"])
        assert {"gradient", "raw"}.issubset(original_entry["update"])
        assert torch.equal(weighted_entry["rollout"]["raw_rewards"],
                           weighted_entry["rollout"]["used_rewards"])
        assert torch.equal(weighted_entry["rollout"]["raw_rewards"],
                           original_entry["rollout"]["rewards"])
    for weighted_round, original_round in zip(weighted_r1["rounds"], iou_r1["rounds"]):
        assert torch.equal(weighted_round["raw_iou_reward_variance"],
                           original_round["reward_variance"])

    # Zero reliability is removed as evidence; all-zero references produce an
    # exact no-update and a mixed set equals its positive-weight subset.
    zero = {**anchor, "position": 1, "physical_frame_id": frame_ids[1], "weight": 0.0}
    all_zero = fit_spatial_instance(
        base, initial, [zero], frame_ids, key, config,
        algorithm="reliability_weighted_iou", reference_role="teacher",
        reference_sha256=reference_sha)
    assert all_zero["optimizer_steps"] == 0
    assert len(all_zero["rounds"]) == FIT_STEPS
    assert all(not round_row["updated"] for round_row in all_zero["rounds"])
    assert _tree_equal(all_zero["initial_state"], all_zero["final_state"])
    assert all(torch.equal(row["optimizer_state"]["state"][k]["step"], torch.tensor(0.))
               for row in all_zero["path"]
               for k in row["optimizer_state"]["state"])

    mixed = fit_spatial_instance(
        base, initial, [zero, anchor], frame_ids, key, config,
        algorithm="reliability_weighted_iou", reference_role="teacher",
        reference_sha256=reference_sha)
    assert _tree_equal(mixed["final_state"], weighted["final_state"])
    assert mixed["optimizer_steps"] == weighted["optimizer_steps"]

    # Reject invalid reliability/geometry and keep physical position binding strict.
    invalid_checks = 0
    for bad_anchor in ({**anchor, "weight": -0.01}, {**anchor, "weight": 1.01},
                       {**anchor, "weight": float("nan")},
                       {**anchor, "physical_frame_id": frame_ids[3]}):
        try:
            _validate_anchors([bad_anchor], frame_ids, torch)
        except ValueError:
            invalid_checks += 1
    assert invalid_checks == 4
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU contract unexpectedly initialized CUDA")

    return {
        "status": "pass",
        "scope": "synthetic CPU only",
        "source_bindings": bindings,
        "contracts": {
            "direct_exactly_matches_original_direct_loss_path": True,
            "direct_checkpoint_callback_roundtrip_and_resume": True,
            "iou_exactly_matches_original_gaussian_iou_opd": True,
            "reliability_one_matches_original_iou_full_state_gradient_adam_and_readout": True,
            "reliability_scales_iou_before_fixed_tau_softmax": True,
            "reliability_not_normalized_across_anchors": True,
            "zero_reliability_is_exact_no_update": True,
            "zero_reliability_anchor_has_no_effect_when_mixed": True,
            "weighted_branch_uses_original_1792_parameter_joint_scope": True,
            "invalid_reliability_and_physical_binding_rejected": invalid_checks,
        },
        "synthetic_fit_steps": FIT_STEPS,
        "synthetic_active_parameters": 1792,
        "real_model_loaded": False,
        "GPU_initialized": False,
        "real_video_or_RGB_read": False,
        "GT_read": False,
        "teacher_called": False,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", action="store_true",
                        help="run only synthetic CPU contracts; no model assets or data")
    args = parser.parse_args()
    if not args.contracts:
        parser.error("this helper is an API module; use --contracts for synthetic CPU checks")
    print(json.dumps(run_cpu_contracts(), ensure_ascii=False, sort_keys=True))
