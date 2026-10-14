"""Source-bound Gaussian spatial OPD wrappers for video-global references.

IoU calls the frozen TA-STVG spatial OPD implementation directly.  DIoU
reuses its native replay, antithetic rollout, likelihood, Adam, checkpoint and
resume machinery, changing only the detached per-sample reward to
IoU - center_distance_squared / enclosing_diagonal_squared.

This module is inert on import.  It does not load a model, video, source case,
teacher response, or label.  Callers own all data and execution gates.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Frozen implementation provenance.  Line spans are human lookup pointers;
# hashes are the binding authority.
SOURCE_BINDINGS = {
    "scripts/stvg_target_teacher_student_v1.py": {
        "sha256": "e792d73414ce0639517faef351330cda8ea97298f9b7854b0ab7055de2bbbc69",
        "origin": "_initialize_fit_resume:672-704; _fit_progress:706-727; _fit_spatial_opd:827-908",
    },
    "scripts/run_stvg_support_aware_spatial_v1.py": {
        "sha256": "8b3a6f42f7a5b46dcc1c5c608e64eeaeec08608d7788b55bfd34af8e3709a6c9",
        "origin": "_anchors_for_case:143-189; main spatial fit wiring:398-444",
    },
    "vg_tta/decota_spatial_opd_tunable_v1.py": {
        "sha256": "3f49e8c95fddc3e67f1b929c9114749815e16bb1d0029161bf1edae1570ad171",
        "origin": "mean_coordinates/action_boxes/likelihood_loss/rollout/optimizer_step:17-70",
    },
    "vg_tta/tastvg_decota_critic_p0_v1.py": {
        "sha256": "78b8b20281f8ef6f667fea2a284213c1780d33f77334153a774d55c2e4d87c99",
        "origin": "aligned_iou:16-23; called by the frozen OPD rollout",
    },
    "vg_tta/c1_enabling_tricks_v1.py": {
        "sha256": "c1c25119bfcc724ffa494261d00d1e81c9a612237ee5bfbf46258496c1b20e59",
        "origin": "TrickReplay implementation used through the frozen student core",
    },
    "methods/decota_final_simplified_v1/tensors.py": {
        "sha256": "d964125abb7934d6c9385f6facd1cab9707db948e55e595c9b489fc5b909a40f",
        "origin": "detached/state helpers used by the frozen student core",
    },
    "methods/decota_spatial_opd_v1/configs.json": {
        "sha256": "fffa324298cf86e00a6445c737be92050983466ecaf20ba036ace685f9a7d1d2",
        "origin": "original per-domain lr/sigma/tau/writeback settings; steps fixed to five here",
    },
}

REFERENCE_ROLES = {"source", "teacher", "offline_oracle"}
OBJECTIVES = {"iou", "diou"}
FIT_STEPS = 5
FIT_SCOPE = "joint"
ANCHOR_LIMIT = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_bindings() -> dict[str, str]:
    """Fail closed if any copied/original fit dependency has changed."""
    actual = {}
    for relative, row in SOURCE_BINDINGS.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required frozen fit source is missing: {relative}")
        digest = sha256_file(path)
        if digest != row["sha256"]:
            raise RuntimeError(f"frozen fit source SHA differs: {relative}")
        actual[relative] = digest
    return actual


def locked_seed_key(case_id: str) -> str:
    """Common-random-number key used across Source/Teacher/Oracle references."""
    if not isinstance(case_id, str) or not case_id or "|" in case_id:
        raise ValueError("case_id must be a nonempty opaque identifier without '|'")
    return f"caption_query_fixed_space|{case_id}"


def _validate_anchors(anchors: Sequence[Mapping[str, Any]], frame_ids: Sequence[int],
                      torch) -> list[dict[str, Any]]:
    if not isinstance(anchors, (list, tuple)) or len(anchors) > ANCHOR_LIMIT:
        raise ValueError("anchors must be zero to four verified sparse observations")
    if not isinstance(frame_ids, (list, tuple)) or not frame_ids:
        raise ValueError("frame_ids must contain the locked physical video grid")
    ids = []
    for value in frame_ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("physical frame IDs must be integers")
        ids.append(int(value))
    if any(right <= left for left, right in zip(ids, ids[1:])):
        raise ValueError("physical frame IDs must be strictly increasing")

    output: list[dict[str, Any]] = []
    seen = set()
    for raw in anchors:
        if not isinstance(raw, Mapping):
            raise ValueError("each sparse observation must be a mapping")
        position = raw.get("position")
        if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < len(ids):
            raise ValueError("sparse observation position is outside the physical frame grid")
        if position in seen:
            raise ValueError("sparse observation positions must be unique")
        seen.add(position)
        physical_id = raw.get("physical_frame_id")
        if (isinstance(physical_id, bool) or not isinstance(physical_id, int)
                or physical_id != ids[position]):
            raise ValueError("sparse observation physical frame binding differs")
        if float(raw.get("weight", 1.0)) != 1.0:
            raise ValueError("the original OPD uses fixed unit observation weights")
        box = raw.get("box")
        value = torch.as_tensor(box).detach().cpu().to(dtype=torch.float64)
        if value.shape != (4,) or not torch.isfinite(value).all():
            raise ValueError("anchor box must be a finite normalized cxcywh vector")
        cx, cy, width, height = (float(x) for x in value.tolist())
        x1, y1, x2, y2 = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
        if not (width > 0 and height > 0 and 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("anchor box must be legal normalized xyxy geometry encoded as cxcywh")
        output.append({**dict(raw), "position": int(position), "box": value.to(dtype=torch.float32),
                       "weight": 1.0})
    return sorted(output, key=lambda row: row["position"])


def _validate_fit_request(*, case_id: str, dataset: str, reference_role: str,
                          reference_sha256: str, objective: str, config: Mapping[str, Any] | None,
                          scope: str) -> tuple[dict[str, Any], str]:
    if reference_role not in REFERENCE_ROLES:
        raise ValueError(f"reference_role must be one of {sorted(REFERENCE_ROLES)}")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}")
    if scope != FIT_SCOPE:
        raise ValueError("this diagnostic helper is fixed to the original 1,792-parameter joint scope")
    if (not isinstance(reference_sha256, str) or len(reference_sha256) != 64
            or any(char not in "0123456789abcdef" for char in reference_sha256)):
        raise ValueError("reference_sha256 must be a lowercase SHA-256 of the separately sealed reference receipt")

    from scripts import stvg_target_teacher_student_v1 as core

    expected = core.load_original_opd_config(dataset)
    expected["steps"] = FIT_STEPS
    chosen = dict(expected if config is None else config)
    if chosen != expected:
        raise ValueError("configuration must equal original per-domain OPD settings with exactly five steps")
    return chosen, locked_seed_key(case_id)


def aligned_diou(prediction_cxcywh, target_cxcywh):
    """Pairwise-aligned DIoU reward in normalized coordinates, lambda=1."""
    import torch
    from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou

    def xyxy(boxes):
        cx, cy, width, height = boxes.unbind(-1)
        return torch.stack((cx - width / 2, cy - height / 2,
                            cx + width / 2, cy + height / 2), dim=-1)

    pred, target = xyxy(prediction_cxcywh), xyxy(target_cxcywh)
    overlap = aligned_iou(prediction_cxcywh, target_cxcywh)
    pred_center = (pred[..., :2] + pred[..., 2:]) / 2
    target_center = (target[..., :2] + target[..., 2:]) / 2
    center_distance_sq = (pred_center - target_center).square().sum(-1)
    enclosure_lt = torch.minimum(pred[..., :2], target[..., :2])
    enclosure_rb = torch.maximum(pred[..., 2:], target[..., 2:])
    enclosure_wh = (enclosure_rb - enclosure_lt).clamp_min(0)
    diagonal_sq = enclosure_wh.square().sum(-1).clamp_min(1e-7)
    return overlap - center_distance_sq / diagonal_sq


def _diou_rollout(mean, evidence, key: str, round_index: int, config: Mapping[str, Any]):
    """Reuse the exact frozen sample path and replace only detached rewards."""
    import torch
    import vg_tta.decota_spatial_opd_tunable_v1 as policy_math

    policy = policy_math.rollout(mean, evidence, key, round_index, shuffle=False, config=config)
    samples_boxes = policy_math.action_boxes(policy["samples"])
    rewards = aligned_diou(samples_boxes, evidence[:, None, :]).detach()
    weights = torch.softmax(rewards / float(config["tau"]), dim=-1).detach()
    informative = bool((rewards != rewards[:, :1]).any())
    if not torch.isfinite(rewards).all() or not torch.isfinite(weights).all():
        raise FloatingPointError("nonfinite DIoU rewards or policy weights")
    return {**policy, "rewards": rewards, "used_rewards": rewards,
            "weights": weights, "permutations": None, "informative": informative,
            "reward_objective": "diou_lambda_1"}


def _fit_diou(base, initial, anchors, frame_ids, key, config, *,
              reference_role: str, reference_sha256: str,
              resume_state=None, checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None):
    """Minimal source-preserving copy of core._fit_spatial_opd with DIoU rewards."""
    import time
    import torch
    from methods.decota_final_simplified_v1.tensors import detached
    from scripts import stvg_target_teacher_student_v1 as core
    from vg_tta.decota_spatial_opd_tunable_v1 import (
        likelihood_loss, mean_coordinates, optimizer_step,
    )

    fit_started = time.perf_counter()
    base.restore(initial)
    replay, selected = core._trainable_replay(base, frame_ids, key, FIT_SCOPE)
    origin = replay.state()
    params = [parameter for _, parameter in selected]
    positions = [int(anchor["position"]) for anchor in anchors]
    if len(set(positions)) != len(positions) or any(not 0 <= position < len(frame_ids) for position in positions):
        raise ValueError("Invalid sparse observation positions")
    if sum(parameter.numel() for parameter in params) != 1792:
        raise RuntimeError("joint OPD parameter count differs from the original 1,792 parameters")
    if anchors:
        evidence = torch.stack([anchor["box"].detach().to(base.zero["boxes"]) for anchor in anchors])
    else:
        evidence = base.zero["boxes"].new_empty((0, 4))
    optimizer = torch.optim.Adam(params, lr=config["lr"], betas=(.9, .999), eps=1e-8, weight_decay=0.)
    arm = "SpatialDIoUOPD"
    next_step, path, rounds, optimizer_steps = core._initialize_fit_resume(
        replay, optimizer, resume_state, arm=arm, scope=FIT_SCOPE, key=key,
        frame_ids=frame_ids, config=config, origin=origin)

    def publish_progress(next_index: int) -> None:
        if checkpoint_callback is not None:
            progress = core._fit_progress(
                replay, optimizer, arm=arm, scope=FIT_SCOPE, key=key,
                frame_ids=frame_ids, config=config, origin=origin,
                next_step=next_index, path=path, rounds=rounds,
                optimizer_steps=optimizer_steps)
            progress["fit_binding"] = {"reference_role": reference_role,
                                        "reference_sha256": reference_sha256,
                                        "reward_objective": "diou_lambda_1"}
            checkpoint_callback(progress)

    try:
        for step in range(next_step, config["steps"]):
            step_started = time.perf_counter()
            before_state = replay.state()
            before_values = replay.values()
            before_boxes = before_values["boxes"].detach().clone()
            entry = {"step": step, "boxes": before_boxes.detach().cpu(),
                     "state": detached(before_state, "cpu"),
                     "optimizer_state": core._copy_cpu(optimizer.state_dict()),
                     "rng_state_before": core._rng_snapshot()}
            if not positions:
                entry.update(skipped=True, reason="no_valid_teacher_observations",
                             step_wall_seconds=time.perf_counter() - step_started)
                rounds.append({"step": step, "no_information": True, "updated": False,
                               "wall_seconds": time.perf_counter() - step_started})
                path.append(entry)
                publish_progress(step + 1)
                continue

            mu = mean_coordinates(before_values["boxes"][positions])
            policy = _diou_rollout(mu, evidence, key, step, config)
            loss = likelihood_loss(mu, policy["mean"], policy["samples"], policy["weights"],
                                   sigma=config["sigma"])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite DIoU Gaussian likelihood loss")
            policy_mean_box = (policy["weights"][..., None] * policy["samples"]).sum(-2).sigmoid()
            policy_delta = policy_mean_box - before_boxes[positions].detach()
            optimizer_before = core._copy_cpu(optimizer.state_dict())
            before_params = detached(replay.state(), "cpu")
            update = optimizer_step(optimizer, loss, params, bool(policy["informative"]))
            after_state = replay.state()
            optimizer_after = core._copy_cpu(optimizer.state_dict())
            if update is None:
                if not core._tree_equal(before_params, detached(after_state, "cpu")):
                    raise AssertionError("Equal-reward DIoU OPD updated student parameters")
                if not core._tree_equal(optimizer_before, optimizer_after):
                    raise AssertionError("Equal-reward DIoU OPD mutated Adam state")
            else:
                optimizer_steps += 1
            with torch.no_grad():
                after_boxes = replay.values()["boxes"].detach().clone()
            tangent = core._state_delta(before_state, after_state)
            expected_move = core._decoder_jvp(replay, list(before_state.values()), list(tangent.values()))
            actual_move = after_boxes - before_boxes
            unobserved = [index for index in range(len(frame_ids)) if index not in set(positions)]
            round_record = {
                "step": step, "updated": update is not None,
                "wall_seconds": time.perf_counter() - step_started,
                "no_information": not bool(policy["informative"]),
                "loss": float(loss.detach()), "reward_objective": "diou_lambda_1",
                "reward_variance": policy["rewards"].var(1, unbiased=False).detach().cpu(),
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
            entry.update(skipped=False, loss=float(loss.detach()), rollout=detached(policy, "cpu"),
                         step_wall_seconds=time.perf_counter() - step_started,
                         optimizer_before=optimizer_before, optimizer_state=optimizer_after,
                         update=None if update is None else detached(update, "cpu"),
                         update_state=detached(after_state, "cpu"),
                         expected_delta=expected_move.detach().cpu(), actual_delta=actual_move.detach().cpu())
            path.append(entry)
            publish_progress(step + 1)

        final_state = replay.state()
        with torch.no_grad():
            final_cached = replay.values()["boxes"].detach().cpu()
        return {"schema": "ta-stvg-student-fit-v1", "arm": arm, "scope": FIT_SCOPE,
                "initial_state": detached(origin, "cpu"), "final_state": detached(final_state, "cpu"),
                "final_cached_decoder_boxes": final_cached, "path": path, "rounds": rounds,
                "optimizer_steps": optimizer_steps, "selected_step": config["steps"],
                "declared_steps": config["steps"], "active_parameters": 1792,
                "config": dict(config), "wall_seconds": time.perf_counter() - fit_started,
                "observation_positions": positions, "policy": "original antithetic on-policy Gaussian OPD",
                "reward_objective": "diou_lambda_1", "teacher_box_in_action_pool": False,
                "reference_role": reference_role, "reference_sha256": reference_sha256,
                "seed_key": key, "seed_spec": "sha256('spatial-opd-v1|{key}|{step}')[:15] as base-16",
                "GT_read": False, "output_source": "student native decoder readout required"}
    finally:
        replay.restore(origin)
        base.restore(initial)


def fit_spatial_opd(base, initial: Mapping[str, Any], anchors: Sequence[Mapping[str, Any]],
                    frame_ids: Sequence[int], *, case_id: str, dataset: str,
                    reference_role: str, reference_sha256: str,
                    objective: str = "iou", config: Mapping[str, Any] | None = None,
                    scope: str = "joint", resume_state: Mapping[str, Any] | None = None,
                    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    """Fit one isolated Source/Teacher/offlineOracle reference with IoU or DIoU reward.

    ``anchors`` are up to four already verified sparse rows with normalized
    ``cxcywh`` tensors and fields ``position``, ``physical_frame_id``, and unit
    ``weight``.  An empty list means unknown reference and preserves the
    original no-information/no-optimizer-step semantics.  All three reference
    roles use the same per-case seed key for matched Gaussian noise; resume
    snapshots are separately bound to role, receipt SHA, and objective.
    """
    verify_source_bindings()
    from scripts import stvg_target_teacher_student_v1 as core
    import torch

    chosen_config, key = _validate_fit_request(
        case_id=case_id, dataset=dataset, reference_role=reference_role,
        reference_sha256=reference_sha256, objective=objective,
        config=config, scope=scope)
    if (not isinstance(frame_ids, (list, tuple)) or not frame_ids
            or any(isinstance(value, bool) or not isinstance(value, int) for value in frame_ids)):
        raise ValueError("frame_ids must be the locked physical frame sequence")
    fixed_frame_ids = [int(value) for value in frame_ids]
    checked_anchors = _validate_anchors(anchors, fixed_frame_ids, torch)
    binding = {"reference_role": reference_role, "reference_sha256": reference_sha256,
               "reward_objective": objective}
    if resume_state is not None and resume_state.get("fit_binding") != binding:
        raise ValueError("resume checkpoint belongs to another reference role, receipt, or reward objective")

    callback = None
    if checkpoint_callback is not None:
        def callback(progress: Mapping[str, Any]) -> None:
            payload = dict(progress)
            payload["fit_binding"] = binding
            checkpoint_callback(payload)

    if objective == "iou":
        fit = core._fit_spatial_opd(
            base, initial, checked_anchors, fixed_frame_ids, key, chosen_config,
            scope=FIT_SCOPE, resume_state=resume_state, checkpoint_callback=callback)
    else:
        fit = _fit_diou(
            base, initial, checked_anchors, fixed_frame_ids, key, chosen_config,
            reference_role=reference_role, reference_sha256=reference_sha256,
            resume_state=resume_state, checkpoint_callback=callback)
    fit.update({"reward_objective": "iou" if objective == "iou" else "diou_lambda_1",
                "reference_role": reference_role, "reference_sha256": reference_sha256,
                "seed_key": key, "seed_spec": "sha256('spatial-opd-v1|{key}|{step}')[:15] as base-16",
                "source_bindings": verify_source_bindings()})
    return fit


def _dataset_matching_config(config: Mapping[str, Any]) -> str:
    """Resolve the locked per-domain settings without inferring from case data."""
    from scripts import stvg_target_teacher_student_v1 as core

    matches = []
    for dataset in ("hc2", "vidstg"):
        expected = core.load_original_opd_config(dataset)
        expected["steps"] = FIT_STEPS
        if dict(config) == expected:
            matches.append(dataset)
    if len(matches) != 1:
        raise ValueError("config must match exactly one original domain's five-step OPD settings")
    return matches[0]


def _anchor_payload_sha256(anchors: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint an anchor payload only when no external receipt SHA is passed."""
    import torch

    rows = []
    for anchor in anchors:
        box = torch.as_tensor(anchor["box"]).detach().cpu().to(torch.float32).contiguous()
        rows.append({"position": int(anchor["position"]),
                     "physical_frame_id": int(anchor["physical_frame_id"]),
                     "weight": float(anchor.get("weight", 1.0)),
                     "box_sha256": hashlib.sha256(box.numpy().tobytes()).hexdigest()})
    encoded = "|".join(f"{row['position']}:{row['physical_frame_id']}:{row['weight']:.17g}:{row['box_sha256']}"
                       for row in rows).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def fit_spatial_reward_opd(base, initial, anchors, frame_ids, key, config,
                           scope="joint", reward_kind="iou", checkpoint_callback=None,
                           *, reference_role="teacher", reference_sha256=None,
                           resume_state=None):
    """Compatibility API for the P0/P1 student worker.

    Positional arguments match the existing student worker's requested call.
    ``reference_role``/``reference_sha256`` are optional extensions that keep
    Source, Teacher, and offline Oracle snapshots separate while preserving
    the same stochastic key.  If no receipt digest is supplied, a hash of the
    anchor tensor payload is used and marked as such in the returned metadata.
    """
    prefix = "caption_query_fixed_space|"
    if not isinstance(key, str) or not key.startswith(prefix) or not key[len(prefix):]:
        raise ValueError("key must preserve the original caption_query_fixed_space|<case_id> form")
    if reward_kind not in OBJECTIVES:
        raise ValueError("reward_kind must be 'iou' or 'diou'")
    if not isinstance(config, Mapping):
        raise ValueError("config must be the locked five-step OPD configuration")
    if key != locked_seed_key(key[len(prefix):]):
        raise ValueError("Gaussian seed key differs from the locked original format")
    dataset = _dataset_matching_config(config)
    digest_kind = "sealed_reference_receipt"
    if reference_sha256 is None:
        reference_sha256 = _anchor_payload_sha256(anchors)
        digest_kind = "anchor_tensor_payload_fingerprint"
    result = fit_spatial_opd(
        base, initial, anchors, frame_ids, case_id=key[len(prefix):], dataset=dataset,
        reference_role=reference_role, reference_sha256=reference_sha256,
        objective=reward_kind, config=config, scope=scope,
        resume_state=resume_state, checkpoint_callback=checkpoint_callback)
    result["reference_binding_kind"] = digest_kind
    return result


def _assert_tree_equal(left: Any, right: Any) -> None:
    import numpy as np
    import torch

    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)):
            raise AssertionError("synthetic tensors are not bitwise equal")
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and np.array_equal(left, right)):
            raise AssertionError("synthetic arrays are not bitwise equal")
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and isinstance(right, Mapping)
                and set(left) == set(right)):
            raise AssertionError("synthetic mapping schemas differ")
        for key in left:
            _assert_tree_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not (type(left) is type(right) and len(left) == len(right)):
            raise AssertionError("synthetic sequence schemas differ")
        for a, b in zip(left, right):
            _assert_tree_equal(a, b)
        return
    if left != right:
        raise AssertionError("synthetic scalar values differ")


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
    """Small deterministic CPU-only contracts; no data or model weights."""
    import torch
    from scripts import stvg_target_teacher_student_v1 as core

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU contracts require CUDA to remain uninitialized")
    verify_source_bindings()
    ids = list(range(8))
    config = {"lr": .01, "sigma": .025, "tau": .05, "steps": FIT_STEPS,
              "writeback": .0625, "samples": 32}
    anchor = [{"position": 2, "physical_frame_id": ids[2],
               "box": torch.tensor([.78, .24, .14, .19]), "weight": 1.0}]
    reference_sha = "a" * 64
    key = locked_seed_key("synthetic_case")

    base = core._synthetic_student_base(len(ids))
    rng = core._rng_snapshot()
    original = core._fit_spatial_opd(base, base.initial, anchor, ids, key, config, scope="joint")
    core.restore_rng_snapshot(rng)
    wrapped = fit_spatial_opd(base, base.initial, anchor, ids, case_id="synthetic_case",
                              dataset="hc2", reference_role="teacher",
                              reference_sha256=reference_sha, objective="iou", config=config)
    for field in ("final_state", "final_cached_decoder_boxes", "path", "rounds",
                  "optimizer_steps", "selected_step", "active_parameters", "config"):
        _assert_tree_equal(_without_timing(original[field]), _without_timing(wrapped[field]))

    # The same frozen antithetic actions produce nonconstant DIoU rewards on a
    # deliberately finite synthetic grid.
    mean = torch.zeros((1, 4), dtype=torch.float32)
    evidence = torch.tensor([[.78, .24, .14, .19]], dtype=torch.float32)
    policy = _diou_rollout(mean, evidence, key, 0, config)
    assert policy["samples"].shape == (1, 32, 4)
    assert torch.isfinite(policy["rewards"]).all()
    assert float(policy["rewards"].max() - policy["rewards"].min()) > 0.0

    # Empty/unknown reference retains the original no-information behavior for
    # IoU and DIoU, including exactly five logged steps and untouched state.
    empty_sha = "b" * 64
    iou_empty = fit_spatial_opd(base, base.initial, [], ids, case_id="synthetic_empty",
        dataset="hc2", reference_role="teacher", reference_sha256=empty_sha,
        objective="iou", config=config)
    diou_empty = fit_spatial_opd(base, base.initial, [], ids, case_id="synthetic_empty",
        dataset="hc2", reference_role="teacher", reference_sha256=empty_sha,
        objective="diou", config=config)
    for result in (iou_empty, diou_empty):
        assert result["optimizer_steps"] == 0 and result["selected_step"] == FIT_STEPS
        assert len(result["path"]) == FIT_STEPS
        _assert_tree_equal(result["initial_state"], result["final_state"])
        assert all(row.get("skipped") is True for row in result["path"])
        assert all(not row.get("optimizer_state", {}).get("state") for row in result["path"])

    # Different reference roles share the same randomness key but cannot
    # cross-resume from one another.
    assert locked_seed_key("synthetic_case") == key
    try:
        fit_spatial_opd(base, base.initial, anchor, ids, case_id="synthetic_case",
            dataset="hc2", reference_role="source", reference_sha256=reference_sha,
            objective="diou", config=config,
            resume_state={"fit_binding": {"reference_role": "teacher",
                "reference_sha256": reference_sha, "reward_objective": "diou"}})
    except ValueError:
        resume_separated = True
    else:
        raise AssertionError("fit accepted a resume checkpoint from another reference role")

    return {"status": "pass", "scope": "synthetic CPU only",
            "contracts": {"iou_wrapper_bitwise_matches_original_core": True,
                          "diou_reward_nonflat": True,
                          "empty_reference_is_unknown_noop_for_both_objectives": True,
                          "reference_roles_share_noise_but_resume_is_separated": resume_separated,
                          "worker_compatibility_api_available": callable(fit_spatial_reward_opd)},
            "model_loaded": False, "GPU_initialized": False, "video_pixels_read": False,
            "GT_read": False, "teacher_called": False}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", action="store_true",
                        help="run only synthetic CPU contracts; no model or data access")
    args = parser.parse_args()
    if not args.contracts:
        parser.error("this helper is an API module; use --contracts for local synthetic checks")
    print(json.dumps(run_cpu_contracts(), ensure_ascii=False, sort_keys=True))
