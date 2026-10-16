"""Gaussian spatial OPD with a fixed, evidence-conditioned adjacent-motion loss.

The only training-objective addition is ``lambda_motion * motion_l1`` with
``lambda_motion == 1``. IoU rollout, Gaussian samples, 1,792-parameter replay,
per-domain five-step config, random key, and Adam settings come from the frozen
spatial helper. This module is inert on import and never reads cases, media,
labels, or model weights by itself.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_BINDINGS = {
    "scripts/stvg_video_global_spatial_fit_fp32_geometry001.py":
        "cd5f9c723c014b367db8c5b5d2e97d5aa1f688f0d095712fbf4a06afd302de53",
    "scripts/stvg_target_teacher_student_v1.py":
        "e792d73414ce0639517faef351330cda8ea97298f9b7854b0ab7055de2bbbc69",
    "vg_tta/decota_spatial_opd_tunable_v1.py":
        "3f49e8c95fddc3e67f1b929c9114749815e16bb1d0029161bf1edae1570ad171",
}
ALGORITHM = "motion_iou"
LAMBDA_MOTION = 1.0
FIT_STEPS = 5
FIT_SCOPE = "joint"
PARAMETER_COUNT = 1792


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_bindings() -> dict[str, str]:
    actual = {}
    for relative, expected in SOURCE_BINDINGS.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required frozen OPD dependency is missing: {relative}")
        digest = _sha256_file(path)
        if digest != expected:
            raise RuntimeError(f"frozen OPD dependency SHA differs: {relative}")
        actual[relative] = digest
    from scripts import stvg_video_global_spatial_fit_fp32_geometry001 as spatial
    spatial.verify_source_bindings()
    return actual


def _fixed_frame_ids(frame_ids: Sequence[int]) -> list[int]:
    if not isinstance(frame_ids, (list, tuple)) or not frame_ids:
        raise ValueError("frame_ids must be the locked physical source grid")
    ids = []
    for raw in frame_ids:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError("physical frame IDs must be integers")
        ids.append(int(raw))
    if any(b <= a for a, b in zip(ids, ids[1:])):
        raise ValueError("source-grid physical frame IDs must be strictly increasing")
    return ids


def _validate_motion_evidence(motion_evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate ordered real-frame centers; None/zero is the only missing form."""
    if not isinstance(motion_evidence, (list, tuple)):
        raise ValueError("motion_evidence must be a sequence of physical-frame records")
    records = []
    last_id = None
    for index, raw in enumerate(motion_evidence):
        if not isinstance(raw, Mapping):
            raise ValueError(f"motion_evidence[{index}] must be a mapping")
        fid = raw.get("physical_frame_id")
        if isinstance(fid, bool) or not isinstance(fid, int):
            raise ValueError(f"motion_evidence[{index}].physical_frame_id must be an integer")
        fid = int(fid)
        if last_id is not None and fid <= last_id:
            raise ValueError("motion_evidence physical frame IDs must be strictly increasing")
        last_id = fid
        try:
            rho = float(raw.get("rho"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"motion_evidence[{index}].rho must be finite and nonnegative") from exc
        if not math.isfinite(rho) or rho < 0:
            raise ValueError(f"motion_evidence[{index}].rho must be finite and nonnegative")
        center = raw.get("center_xy")
        if center is None:
            if rho != 0.0:
                raise ValueError("a missing center must have rho=0; do not encode Unknown as a zero center")
            center_list = None
        else:
            if isinstance(center, (str, bytes)) or not isinstance(center, (list, tuple)) or len(center) != 2:
                raise ValueError(f"motion_evidence[{index}].center_xy must be a two-coordinate list or None")
            try:
                center_list = [float(center[0]), float(center[1])]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"motion_evidence[{index}].center_xy must be numeric") from exc
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in center_list):
                raise ValueError(f"motion_evidence[{index}].center_xy must be finite normalized coordinates")
            if rho == 0.0:
                raise ValueError("rho=0 denotes Missing and must use center_xy=None")
        records.append({"physical_frame_id": fid, "center_xy": center_list, "rho": rho})
    return records


def _evidence_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(records), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _interpolated_center(boxes, frame_ids: Sequence[int], target_id: int):
    """Piecewise-linear normalized center interpolation; never extrapolate."""
    if target_id < frame_ids[0] or target_id > frame_ids[-1]:
        raise ValueError("motion frame lies outside the sampled source grid; extrapolation is forbidden")
    for index, fid in enumerate(frame_ids):
        if target_id == fid:
            return boxes[index, :2], {"left_grid_position": index,
                                      "right_grid_position": index,
                                      "alpha": 0.0}
        if fid < target_id < frame_ids[index + 1]:
            left_id, right_id = fid, frame_ids[index + 1]
            alpha = float(target_id - left_id) / float(right_id - left_id)
            center = boxes[index, :2] * (1.0 - alpha) + boxes[index + 1, :2] * alpha
            return center, {"left_grid_position": index,
                            "right_grid_position": index + 1,
                            "alpha": alpha}
    raise AssertionError("failed to locate an in-range motion frame")


def _valid_adjacent_pairs(records: Sequence[Mapping[str, Any]], frame_ids: Sequence[int]):
    valid, excluded = [], []
    for index, (left, right) in enumerate(zip(records, records[1:])):
        left_id = int(left["physical_frame_id"])
        right_id = int(right["physical_frame_id"])
        if right_id != left_id + 1:
            excluded.append({"evidence_pair_index": index, "physical_frame_ids": [left_id, right_id],
                             "reason": "nonconsecutive_physical_frames"})
            continue
        if (left["center_xy"] is None or right["center_xy"] is None
                or float(left["rho"]) <= 0 or float(right["rho"]) <= 0):
            excluded.append({"evidence_pair_index": index, "physical_frame_ids": [left_id, right_id],
                             "reason": "missing_or_zero_rho_endpoint"})
            continue
        if left_id < frame_ids[0] or right_id > frame_ids[-1]:
            excluded.append({"evidence_pair_index": index, "physical_frame_ids": [left_id, right_id],
                             "reason": "outside_sampled_source_grid"})
            continue
        rho = min(float(left["rho"]), float(right["rho"]))
        if rho <= 0:
            excluded.append({"evidence_pair_index": index, "physical_frame_ids": [left_id, right_id],
                             "reason": "zero_pair_weight"})
            continue
        valid.append({"evidence_pair_index": index,
                      "left_physical_frame_id": left_id,
                      "right_physical_frame_id": right_id,
                      "left_center_xy": list(left["center_xy"]),
                      "right_center_xy": list(right["center_xy"]),
                      "rho_left": float(left["rho"]),
                      "rho_right": float(right["rho"]),
                      "pair_rho": rho})
    return valid, excluded


def adjacent_motion_loss(student_boxes, frame_ids: Sequence[int],
                         motion_evidence: Sequence[Mapping[str, Any]], *,
                         return_receipt: bool = False):
    """Confidence-weighted adjacent-center L1 on the real physical frame axis.

    Each coordinate pair first takes its two-dimensional mean absolute error;
    valid pair errors are then weighted by ``min(rho_left, rho_right)`` and
    divided by the sum of those pair weights. Missing or non-adjacent frames
    break the path and are never bridged.
    """
    import torch

    ids = _fixed_frame_ids(frame_ids)
    if not torch.is_tensor(student_boxes) or student_boxes.ndim != 2 or student_boxes.shape != (len(ids), 4):
        raise ValueError("student_boxes must be the sampled native (N,4) cxcywh readout")
    if not torch.isfinite(student_boxes).all():
        raise ValueError("student sampled boxes contain nonfinite values")
    records = _validate_motion_evidence(motion_evidence)
    valid, excluded = _valid_adjacent_pairs(records, ids)
    if not valid:
        zero = student_boxes.sum() * 0.0
        receipt = {"valid_pair_count": 0, "pair_weight_sum": 0.0,
                   "valid_pairs": [], "excluded_pairs": excluded}
        return (zero, receipt) if return_receipt else zero

    pair_errors = []
    pair_weights = []
    pair_receipts = []
    for pair in valid:
        left_pred, left_interp = _interpolated_center(
            student_boxes, ids, pair["left_physical_frame_id"])
        right_pred, right_interp = _interpolated_center(
            student_boxes, ids, pair["right_physical_frame_id"])
        predicted_delta = right_pred - left_pred
        target_delta = student_boxes.new_tensor(pair["right_center_xy"]) - student_boxes.new_tensor(
            pair["left_center_xy"])
        coordinate_error = (predicted_delta - target_delta).abs()
        pair_error = coordinate_error.mean()
        pair_errors.append(pair_error)
        pair_weights.append(pair["pair_rho"])
        pair_receipts.append({
            **pair,
            "left_interpolation": left_interp,
            "right_interpolation": right_interp,
            "student_delta_xy": predicted_delta.detach().cpu().tolist(),
            "evidence_delta_xy": target_delta.detach().cpu().tolist(),
            "absolute_delta_error_xy": coordinate_error.detach().cpu().tolist(),
            "mean_coordinate_l1": float(pair_error.detach().cpu()),
            "weighted_l1_contribution": float(pair_error.detach().cpu()) * pair["pair_rho"],
        })
    weight_tensor = student_boxes.new_tensor(pair_weights)
    loss = (torch.stack(pair_errors) * weight_tensor).sum() / weight_tensor.sum()
    if not torch.isfinite(loss):
        raise FloatingPointError("motion loss is nonfinite")
    receipt = {"valid_pair_count": len(valid),
               "pair_weight_sum": float(sum(pair_weights)),
               "valid_pairs": pair_receipts,
               "excluded_pairs": excluded}
    return (loss, receipt) if return_receipt else loss


def fit_motion(base, initial: Mapping[str, Any], anchors: Sequence[Mapping[str, Any]],
               frame_ids: Sequence[int], motion_evidence: Sequence[Mapping[str, Any]], *,
               case_id: str, dataset: str, reference_role: str,
               reference_sha256: str, config: Mapping[str, Any] | None = None,
               scope: str = "joint", resume_state: Mapping[str, Any] | None = None,
               checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    """Fit original Gaussian IoU-OPD plus a fixed unit adjacent-motion loss.

    The input/output contract mirrors ``fit_spatial_opd`` with one additional
    ``motion_evidence`` argument. ``initial`` is the cold source light state;
    the worker remains responsible for a fresh full-model readout.
    """
    import torch
    from methods.decota_final_simplified_v1.tensors import detached
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts import stvg_video_global_spatial_fit_fp32_geometry001 as spatial
    from vg_tta.decota_spatial_opd_tunable_v1 import (
        likelihood_loss, mean_coordinates, optimizer_step, rollout,
    )

    source_bindings = verify_source_bindings()
    chosen_config, key = spatial._validate_fit_request(
        case_id=case_id, dataset=dataset, reference_role=reference_role,
        reference_sha256=reference_sha256, objective="iou", config=config, scope=scope)
    ids = _fixed_frame_ids(frame_ids)
    checked_anchors = spatial._validate_anchors(anchors, ids, torch)
    records = _validate_motion_evidence(motion_evidence)
    evidence_sha = _evidence_sha256(records)
    motion_pairs, _ = _valid_adjacent_pairs(records, ids)
    binding = {"algorithm": ALGORITHM, "lambda_motion": LAMBDA_MOTION,
               "reference_role": reference_role, "reference_sha256": reference_sha256,
               "motion_evidence_sha256": evidence_sha}
    if resume_state is not None and resume_state.get("fit_binding") != binding:
        raise ValueError("resume checkpoint belongs to another role, reference, or motion-evidence receipt")

    base.restore(initial)
    replay, selected = core._trainable_replay(base, ids, key, FIT_SCOPE)
    origin = replay.state()
    params = [parameter for _, parameter in selected]
    names = [name for name, _ in selected]
    if sum(parameter.numel() for parameter in params) != PARAMETER_COUNT:
        raise RuntimeError("motion_iou requires exactly the original 1,792 joint decoder parameters")
    positions = [int(anchor["position"]) for anchor in checked_anchors]
    if checked_anchors:
        target_boxes = torch.stack([anchor["box"].detach().to(base.zero["boxes"])
                                    for anchor in checked_anchors])
    else:
        target_boxes = base.zero["boxes"].new_empty((0, 4))
    optimizer = torch.optim.Adam(params, lr=chosen_config["lr"], betas=(.9, .999),
                                 eps=1e-8, weight_decay=0.)
    arm = "SpatialMotionIoUOPD"
    next_step, path, rounds, optimizer_steps = core._initialize_fit_resume(
        replay, optimizer, resume_state, arm=arm, scope=FIT_SCOPE, key=key,
        frame_ids=ids, config=chosen_config, origin=origin)

    def publish_progress(next_index: int) -> None:
        if checkpoint_callback is not None:
            progress = core._fit_progress(
                replay, optimizer, arm=arm, scope=FIT_SCOPE, key=key,
                frame_ids=ids, config=chosen_config, origin=origin,
                next_step=next_index, path=path, rounds=rounds,
                optimizer_steps=optimizer_steps)
            progress["fit_binding"] = binding
            checkpoint_callback(progress)

    fit_started = time.perf_counter()
    try:
        for step in range(next_step, chosen_config["steps"]):
            step_started = time.perf_counter()
            before_state = replay.state()
            before_values = replay.values()
            before_boxes_live = before_values["boxes"]
            before_boxes = before_boxes_live.detach().clone()
            optimizer_before = core._copy_cpu(optimizer.state_dict())
            entry = {"step": step, "boxes": before_boxes.cpu(),
                     "state": detached(before_state, "cpu"),
                     "optimizer_state": optimizer_before,
                     "rng_state_before": core._rng_snapshot(),
                     "algorithm": ALGORITHM, "lambda_motion": LAMBDA_MOTION,
                     "motion_evidence_sha256": evidence_sha}

            if not positions and not motion_pairs:
                zero_motion, motion_receipt = adjacent_motion_loss(
                    before_boxes_live, ids, records, return_receipt=True)
                entry.update(skipped=True, reason="no_spatial_anchors_or_valid_motion_pairs",
                             raw_total_gradient=torch.zeros(PARAMETER_COUNT),
                             motion=motion_receipt,
                             step_wall_seconds=time.perf_counter() - step_started)
                rounds.append({"step": step, "updated": False, "no_information": True,
                               "opd_loss": 0.0, "motion_loss": float(zero_motion.detach()),
                               "total_loss": 0.0, "lambda_motion": LAMBDA_MOTION,
                               "motion_valid_pair_count": 0,
                               "motion_pair_weight_sum": 0.0,
                               "motion_pairs": motion_receipt["valid_pairs"],
                               "motion_excluded_pairs": motion_receipt["excluded_pairs"],
                               "raw_gradient_norm": 0.0,
                               "optimizer_state_unchanged": True})
                path.append(entry)
                publish_progress(step + 1)
                continue

            if positions:
                mu = mean_coordinates(before_values["boxes"][positions])
                policy = rollout(mu, target_boxes, key, step, shuffle=False, config=chosen_config)
                opd_loss = likelihood_loss(mu, policy["mean"], policy["samples"],
                                           policy["weights"], sigma=chosen_config["sigma"])
                if not torch.isfinite(opd_loss):
                    raise FloatingPointError("nonfinite original Gaussian IoU-OPD loss")
            else:
                mu = before_values["boxes"].new_empty((0, 4))
                policy = {"informative": False, "rewards": before_values["boxes"].new_empty((0, 32)),
                          "weights": before_values["boxes"].new_empty((0, 32)),
                          "samples": before_values["boxes"].new_empty((0, 32, 4)),
                          "mean": mu.detach(), "seed": None}
                opd_loss = before_values["boxes"].sum() * 0.0

            motion_loss, motion_receipt = adjacent_motion_loss(
                before_values["boxes"], ids, records, return_receipt=True)
            # Preserve the original IoU-OPD operation path exactly when no
            # valid motion pair exists. A valid motion term has fixed lambda 1.
            total_loss = (opd_loss if motion_receipt["valid_pair_count"] == 0
                          else opd_loss + LAMBDA_MOTION * motion_loss)
            if not torch.isfinite(total_loss):
                raise FloatingPointError("nonfinite combined motion_iou objective")

            policy_informative = bool(policy["informative"])
            if policy_informative:
                # The original OPD always steps for informative rewards; use
                # its exact optimizer-returned raw gradient without an extra
                # backward pass.
                raw_gradient = None
                gradient_nonzero = True
                should_step = True
            else:
                probe = torch.autograd.grad(total_loss, params, retain_graph=True,
                                            allow_unused=True)
                gradient_chunks = [torch.zeros_like(parameter).reshape(-1) if grad is None
                                   else grad.detach().reshape(-1)
                                   for parameter, grad in zip(params, probe, strict=True)]
                raw_gradient = torch.cat(gradient_chunks).detach()
                if not torch.isfinite(raw_gradient).all():
                    raise FloatingPointError("nonfinite motion_iou raw gradient")
                gradient_nonzero = bool(torch.count_nonzero(raw_gradient).item())
                should_step = gradient_nonzero

            before_params = detached(replay.state(), "cpu")
            update = optimizer_step(optimizer, total_loss, params, should_step)
            if update is not None:
                raw_gradient = update["gradient"].to(device=before_boxes.device)
            elif raw_gradient is None:
                raise AssertionError("informative OPD reward did not produce an optimizer update")
            if not torch.isfinite(raw_gradient).all():
                raise FloatingPointError("nonfinite motion_iou raw gradient")
            after_state = replay.state()
            optimizer_after = core._copy_cpu(optimizer.state_dict())
            if update is None:
                if not core._tree_equal(before_params, detached(after_state, "cpu")):
                    raise AssertionError("uninformative motion_iou changed student parameters")
                if not core._tree_equal(optimizer_before, optimizer_after):
                    raise AssertionError("uninformative motion_iou changed Adam state")
            else:
                optimizer_steps += 1
                if (not policy_informative and
                        not torch.allclose(update["gradient"].reshape(-1), raw_gradient.cpu(),
                                           atol=1e-6, rtol=1e-5)):
                    raise AssertionError("Adam raw gradient differs from the pre-step audit gradient")

            with torch.no_grad():
                after_boxes = replay.values()["boxes"].detach().clone()
            if positions:
                policy_mean_box = (policy["weights"][..., None] * policy["samples"]).sum(-2).sigmoid()
                policy_delta = policy_mean_box - before_boxes[positions].detach()
                mean_after = mean_coordinates(after_boxes[positions])
                from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
                central_reward_before = aligned_iou(before_boxes[positions], target_boxes).detach().cpu()
                central_reward_after = aligned_iou(after_boxes[positions], target_boxes).detach().cpu()
                weighted_reward = (policy["weights"] * policy["rewards"]).sum(1).mean()
                mean_movement = (mean_after - mu.detach()).norm()
            else:
                policy_delta = before_boxes.new_empty((0, 4))
                mean_after = before_boxes.new_empty((0, 4))
                central_reward_before = before_boxes.new_empty((0,))
                central_reward_after = before_boxes.new_empty((0,))
                weighted_reward = before_boxes.new_tensor(0.0)
                mean_movement = before_boxes.new_tensor(0.0)
            tangent = core._state_delta(before_state, after_state)
            expected_move = core._decoder_jvp(replay, list(before_state.values()), list(tangent.values()))
            actual_move = after_boxes - before_boxes
            unobserved = [index for index in range(len(ids)) if index not in set(positions)]
            motion_loss_value = float(motion_loss.detach().cpu())
            opd_loss_value = float(opd_loss.detach().cpu())
            total_loss_value = float(total_loss.detach().cpu())
            round_record = {
                "step": step, "algorithm": ALGORITHM,
                "updated": update is not None,
                "no_information": not should_step,
                "policy_informative": policy_informative,
                "motion_gradient_informative": gradient_nonzero and not policy_informative,
                "opd_loss": opd_loss_value,
                "motion_loss": motion_loss_value,
                "lambda_motion": LAMBDA_MOTION,
                "total_loss": total_loss_value,
                "loss": total_loss_value,
                "reward_variance": (policy["rewards"].var(1, unbiased=False).detach().cpu()
                                    if positions else torch.empty(0)),
                "weight_entropy": ((-(policy["weights"] * policy["weights"].clamp_min(1e-30).log()).sum(1)).detach().cpu()
                                   if positions else torch.empty(0)),
                "weight_ess": ((1.0 / policy["weights"].square().sum(1)).detach().cpu()
                               if positions else torch.empty(0)),
                "motion_valid_pair_count": motion_receipt["valid_pair_count"],
                "motion_pair_weight_sum": motion_receipt["pair_weight_sum"],
                "motion_pairs": motion_receipt["valid_pairs"],
                "motion_excluded_pairs": motion_receipt["excluded_pairs"],
                "raw_gradient_norm": float(raw_gradient.norm().cpu()),
                "raw_gradient_nonzero_count": int(torch.count_nonzero(raw_gradient).item()),
                "actions_detached": (not policy["samples"].requires_grad if positions else True),
                "teacher_targets_detached": (not target_boxes.requires_grad),
                "reward_weights_detached": (not policy["weights"].requires_grad if positions else True),
                "expected_policy_delta_observed": policy_delta.detach().cpu(),
                "mean_before": mu.detach().cpu(),
                "mean_after": mean_after.detach().cpu(),
                "central_reward_before": central_reward_before,
                "central_reward_after": central_reward_after,
                "weighted_reward": float(weighted_reward.detach().cpu()),
                "mean_movement": float(mean_movement.detach().cpu()),
                "expected_delta_all_positions": expected_move.detach().cpu(),
                "actual_delta_all_positions": actual_move.detach().cpu(),
                "observed_movement": core._movement_summary(expected_move, actual_move, positions),
                "unobserved_movement": core._movement_summary(expected_move, actual_move, unobserved),
                "step_wall_seconds": time.perf_counter() - step_started,
                "wall_seconds": time.perf_counter() - step_started,
            }
            rounds.append(round_record)
            entry.update(skipped=False,
                         loss=total_loss_value,
                         opd_loss=opd_loss_value,
                         motion_loss=motion_loss_value,
                         total_loss=total_loss_value,
                         lambda_motion=LAMBDA_MOTION,
                         raw_total_gradient=raw_gradient.cpu(),
                         motion=motion_receipt,
                         rollout=detached(policy, "cpu"),
                         parameter_names=names,
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
        return {
            "schema": "ta-stvg-student-fit-v1",
            "arm": arm,
            "algorithm": ALGORITHM,
            "scope": FIT_SCOPE,
            "initial_state": detached(origin, "cpu"),
            "final_state": detached(final_state, "cpu"),
            "final_cached_decoder_boxes": final_cached,
            "path": path,
            "rounds": rounds,
            "optimizer_steps": optimizer_steps,
            "selected_step": chosen_config["steps"],
            "declared_steps": chosen_config["steps"],
            "active_parameters": PARAMETER_COUNT,
            "config": dict(chosen_config),
            "wall_seconds": time.perf_counter() - fit_started,
            "observation_positions": positions,
            "policy": "original antithetic on-policy Gaussian IoU OPD",
            "reward_objective": "iou",
            "motion_objective": "rho_weighted_mean_adjacent_center_displacement_L1",
            "lambda_motion": LAMBDA_MOTION,
            "motion_evidence": records,
            "motion_evidence_sha256": evidence_sha,
            "motion_valid_pair_count": len(motion_pairs),
            "teacher_box_in_action_pool": False,
            "reference_role": reference_role,
            "reference_sha256": reference_sha256,
            "seed_key": key,
            "seed_spec": "sha256('spatial-opd-v1|{key}|{step}')[:15] as base-16",
            "source_bindings": source_bindings,
            "GT_read": False,
            "output_source": "student native decoder readout required",
            "fit_binding": binding,
        }
    finally:
        replay.restore(origin)
        base.restore(initial)


def _assert_tree_equal(left: Any, right: Any, path: str = "root") -> None:
    import numpy as np
    import torch
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and np.array_equal(left, right)):
            raise AssertionError(f"synthetic arrays differ at {path}")
        return
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)) or not torch.equal(left, right):
            raise AssertionError(f"synthetic tensors differ at {path}")
    elif isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise AssertionError(f"synthetic mapping keys differ at {path}")
        for key in left:
            _assert_tree_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        if len(left) != len(right):
            raise AssertionError(f"synthetic sequence length differs at {path}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _assert_tree_equal(a, b, f"{path}[{index}]")
    elif left != right:
        raise AssertionError(f"synthetic scalar values differ at {path}: {left!r} != {right!r}")


def run_cpu_contracts() -> dict[str, Any]:
    """Synthetic CPU math and fake-decoder contracts only; no real inputs."""
    import torch
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts import stvg_video_global_spatial_fit_fp32_geometry001 as spatial

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU contracts require CUDA to remain uninitialized")
    bindings = verify_source_bindings()

    # Differentiable interpolation uses only positions between sampled source
    # frames and the finite-difference check covers both coordinates.
    grid_ids = [10, 12, 14]
    boxes = torch.tensor([[.18, .28, .2, .2], [.47, .40, .2, .2],
                          [.77, .53, .2, .2]], dtype=torch.float64, requires_grad=True)
    evidence = [
        {"physical_frame_id": fid, "center_xy": center, "rho": rho}
        for fid, center, rho in zip(
            range(10, 15),
            ([.10, .15], [.22, .23], [.35, .32], [.48, .43], [.65, .56]),
            (1.0, .8, .6, .9, .7), strict=True)
    ]
    loss, receipt = adjacent_motion_loss(boxes, grid_ids, evidence, return_receipt=True)
    grad = torch.autograd.grad(loss, boxes)[0]
    epsilon = 1e-6
    plus = boxes.detach().clone(); plus[1, 0] += epsilon
    minus = boxes.detach().clone(); minus[1, 0] -= epsilon
    fd = (adjacent_motion_loss(plus, grid_ids, evidence)
          - adjacent_motion_loss(minus, grid_ids, evidence)) / (2 * epsilon)
    fd_error = abs(float(fd) - float(grad[1, 0]))
    if fd_error > 2e-7:
        raise AssertionError(f"motion interpolation finite-difference error too large: {fd_error}")

    # Adding the same spatial translation to every sampled center cannot alter
    # adjacent displacements.
    translated = boxes.detach().clone()
    translated[:, :2] += torch.tensor([.01, -.02], dtype=torch.float64)
    translation_loss = adjacent_motion_loss(translated, grid_ids, evidence)
    translation_delta = abs(float(translation_loss) - float(loss.detach()))
    if translation_delta > 1e-12:
        raise AssertionError("constant clip translation changed an adjacent-motion loss")

    broken = [
        {"physical_frame_id": 0, "center_xy": [.1, .1], "rho": 1.0},
        {"physical_frame_id": 1, "center_xy": [.2, .1], "rho": 1.0},
        {"physical_frame_id": 2, "center_xy": None, "rho": 0.0},
        {"physical_frame_id": 3, "center_xy": [.4, .1], "rho": 1.0},
        {"physical_frame_id": 4, "center_xy": [.5, .1], "rho": .5},
    ]
    broken_boxes = torch.tensor([[.1, .1, .1, .1], [.3, .1, .1, .1],
                                 [.5, .1, .1, .1], [.7, .1, .1, .1],
                                 [.9, .1, .1, .1]], dtype=torch.float64, requires_grad=True)
    _, broken_receipt = adjacent_motion_loss(broken_boxes, list(range(5)), broken,
                                             return_receipt=True)
    if [row["physical_frame_ids"] for row in broken_receipt["excluded_pairs"]
            if row["reason"] == "missing_or_zero_rho_endpoint"] != [[1, 2], [2, 3]]:
        raise AssertionError("Missing motion evidence was bridged")
    if broken_receipt["valid_pair_count"] != 2:
        raise AssertionError("valid motion-pair accounting differs across Missing")

    no_evidence_loss, no_evidence = adjacent_motion_loss(
        broken_boxes, list(range(5)), [], return_receipt=True)
    no_evidence_grad = torch.autograd.grad(no_evidence_loss, broken_boxes)[0]
    if no_evidence["valid_pair_count"] != 0 or torch.count_nonzero(no_evidence_grad).item() != 0:
        raise AssertionError("zero motion evidence did not produce a differentiable exact zero")

    unknown_zero_rejected = False
    try:
        _validate_motion_evidence([
            {"physical_frame_id": 0, "center_xy": [0.0, 0.0], "rho": 0.0}])
    except ValueError:
        unknown_zero_rejected = True
    if not unknown_zero_rejected:
        raise AssertionError("Unknown motion center was accepted as a fake zero coordinate")
    outside_loss, outside_receipt = adjacent_motion_loss(
        broken_boxes, list(range(5)), [
            {"physical_frame_id": -1, "center_xy": [.1, .1], "rho": 1.0},
            {"physical_frame_id": 0, "center_xy": [.2, .1], "rho": 1.0},
        ], return_receipt=True)
    if float(outside_loss.detach()) != 0.0 or outside_receipt["valid_pair_count"] != 0:
        raise AssertionError("out-of-grid motion evidence was extrapolated")

    # The no-motion path is bitwise-equivalent to the frozen original IoU fit.
    fit_ids = list(range(8))
    fit_config = {"lr": .01, "sigma": .025, "tau": .05,
                  "steps": FIT_STEPS, "writeback": .0625, "samples": 32}
    anchor = [{"position": 2, "physical_frame_id": 2,
               "box": torch.tensor([.78, .24, .14, .19]), "weight": 1.0}]
    reference_sha = "a" * 64
    base = core._synthetic_student_base(len(fit_ids))
    original_fit = spatial.fit_spatial_opd(
        base, base.initial, anchor, fit_ids, case_id="synthetic_motion_noop",
        dataset="hc2", reference_role="teacher", reference_sha256=reference_sha,
        objective="iou", config=fit_config)
    motion_empty_fit = fit_motion(
        base, base.initial, anchor, fit_ids, [], case_id="synthetic_motion_noop",
        dataset="hc2", reference_role="teacher", reference_sha256=reference_sha,
        config=fit_config)
    for field in ("initial_state", "final_state", "final_cached_decoder_boxes",
                  "path", "rounds", "optimizer_steps", "selected_step",
                  "active_parameters", "config"):
        # New audit-only loss keys differ in path entries; compare the student
        # and Adam artifacts plus exact per-step updates separately below.
        if field in ("path", "rounds"):
            continue
        _assert_tree_equal(original_fit[field], motion_empty_fit[
            {"initial_state": "initial_state", "final_state": "final_state",
             "final_cached_decoder_boxes": "final_cached_decoder_boxes",
             "optimizer_steps": "optimizer_steps", "selected_step": "selected_step",
             "active_parameters": "active_parameters", "config": "config"}[field]], field)
    if len(original_fit["path"]) != len(motion_empty_fit["path"]):
        raise AssertionError("motion fit changed the five-step source path length")
    for old_step, new_step in zip(original_fit["path"], motion_empty_fit["path"], strict=True):
        for field, value in old_step.items():
            if field in {"step_wall_seconds"}:
                continue
            _assert_tree_equal(value, new_step.get(field), f"path.{field}")
    for index, (old_round, new_round) in enumerate(
            zip(original_fit["rounds"], motion_empty_fit["rounds"], strict=True)):
        for field, value in old_round.items():
            if field == "wall_seconds":
                continue
            _assert_tree_equal(value, new_round.get(field), f"rounds[{index}].{field}")

    # A far anchor yields exactly uniform zero-IoU rewards. Without motion,
    # parameters and Adam remain untouched; with valid translating motion, the
    # same rewards permit a real gradient-driven step.
    far_anchor = [{"position": 2, "physical_frame_id": 2,
                   "box": torch.tensor([.99, .99, .01, .01]), "weight": 1.0}]
    noop = fit_motion(base, base.initial, far_anchor, fit_ids, [],
                      case_id="synthetic_uniform_no_motion", dataset="hc2",
                      reference_role="source", reference_sha256="b" * 64,
                      config=fit_config)
    if noop["optimizer_steps"] != 0:
        raise AssertionError("uniform IoU rewards with no motion evidence stepped Adam")
    for name in noop["initial_state"]:
        _assert_tree_equal(noop["initial_state"][name], noop["final_state"][name],
                           f"uniform_noop.{name}")
    if any(step.get("optimizer_state", {}).get("state") for step in noop["path"]):
        raise AssertionError("uniform no-motion fit initialized or mutated Adam state")

    moving_evidence = [
        {"physical_frame_id": fid, "center_xy": [.35 + .015 * fid, .42], "rho": 1.0}
        for fid in fit_ids
    ]
    moved = fit_motion(base, base.initial, far_anchor, fit_ids, moving_evidence,
                       case_id="synthetic_uniform_with_motion", dataset="hc2",
                       reference_role="source", reference_sha256="c" * 64,
                       config=fit_config)
    if moved["optimizer_steps"] <= 0:
        raise AssertionError("motion_iou did not step when IoU rewards were uniform but motion had signal")
    if any(float(row["reward_variance"].max()) != 0.0 for row in moved["rounds"]):
        raise AssertionError("synthetic far-anchor IoU reward unexpectedly became informative")

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU contracts unexpectedly initialized CUDA")
    return {
        "status": "pass",
        "scope": "synthetic CPU math and 1792-parameter fake decoder only",
        "contracts": {
            "differentiable_physical_frame_interpolation": True,
            "finite_difference_max_abs_error": fd_error,
            "constant_translation_invariance_max_abs_error": translation_delta,
            "missing_pair_not_bridged": True,
            "unknown_center_not_zero_filled": unknown_zero_rejected,
            "zero_motion_evidence_exact_zero": True,
            "no_motion_bitwise_original_IoU_OPD_equivalence": True,
            "uniform_reward_without_motion_exact_noop": True,
            "uniform_reward_with_motion_takes_optimizer_step": moved["optimizer_steps"] > 0,
            "same_1792_parameters_and_five_steps": True,
            "sampled_grid_interpolation_never_extrapolates": True,
        },
        "synthetic_counts": {
            "finite_difference_motion_pairs": receipt["valid_pair_count"],
            "missing_gap_valid_pairs": broken_receipt["valid_pair_count"],
            "uniform_reward_motion_optimizer_steps": moved["optimizer_steps"],
            "uniform_reward_motion_valid_pairs": moved["motion_valid_pair_count"],
        },
        "helper_sha256": _sha256_file(Path(__file__).resolve()),
        "source_bindings": bindings,
        "model_loaded": False,
        "GPU_initialized": False,
        "real_data_read": False,
        "GT_read": False,
    }


def main(argv=None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("contracts",),
                        help="only synthetic CPU contracts are exposed by this module")
    parser.add_argument("--output", type=Path,
                        help="optional JSON receipt path for synthetic contracts only")
    args = parser.parse_args(argv)
    result = run_cpu_contracts()
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
