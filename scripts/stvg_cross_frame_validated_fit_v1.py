"""Cross-frame validated wrapper for the fixed loss-weighted IoU OPD fit.

Training remains the original five-step reliability-weighted IoU likelihood.
An update is retained only when both the original weighted training-reference
IoU and a disjoint, uniform mean validation-reference IoU do not decrease by
more than 1e-7. Validation sets with fewer than two anchors fall back to the
original training-only validated fit. This is agreement with fixed references,
not semantic identity validation or a generalization guarantee.
"""
from __future__ import annotations

import copy
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIT_STEPS = 5
VALIDATION_ANCHOR_LIMIT = 8
GATE_TOLERANCE = 1e-7
OLD_HELPER = ROOT / "scripts/stvg_instance_validated_fit_v1.py"
OLD_HELPER_SHA256 = "0ecd0aedfd9fc37c84c96bc46901b706967c2c36c22fc615109cc94672b2b7c3"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_bindings() -> dict[str, str]:
    """Pin the old validated fit whose objective and train gate are reused."""
    if not OLD_HELPER.is_file():
        raise FileNotFoundError("original instance validated-fit helper is missing")
    actual = _sha256_file(OLD_HELPER)
    if OLD_HELPER_SHA256 == "TO_BE_FILLED" or actual != OLD_HELPER_SHA256:
        raise RuntimeError("original instance validated-fit helper SHA differs")
    from scripts import stvg_native_instance_spatial_fit_v1 as native
    return {"scripts/stvg_instance_validated_fit_v1.py": actual,
            **native.verify_source_bindings()}


def _validate_validation_anchors(
    anchors: Sequence[Mapping[str, Any]], frame_ids: Sequence[int], torch,
) -> list[dict[str, Any]]:
    """Validate up to eight fixed-mask references without using reliability weights."""
    if not isinstance(anchors, (list, tuple)) or len(anchors) > VALIDATION_ANCHOR_LIMIT:
        raise ValueError("validation_anchors must contain zero to eight fixed references")
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
            raise ValueError("each validation anchor must be a mapping")
        position = raw.get("position")
        if (isinstance(position, bool) or not isinstance(position, int)
                or not 0 <= position < len(ids)):
            raise ValueError("validation anchor position is outside the locked frame grid")
        if int(position) in seen:
            raise ValueError("validation anchor positions must be unique")
        seen.add(int(position))
        physical_id = raw.get("physical_frame_id")
        if (isinstance(physical_id, bool) or not isinstance(physical_id, int)
                or int(physical_id) != ids[int(position)]):
            raise ValueError("validation anchor physical frame differs from its position")
        box64 = torch.as_tensor(raw.get("box")).detach().cpu().to(dtype=torch.float64)
        if box64.shape != (4,) or not bool(torch.isfinite(box64).all()):
            raise ValueError("validation box must be a finite normalized cxcywh vector")
        cx, cy, width, height = (float(value) for value in box64.tolist())
        x1, y1 = cx - width / 2.0, cy - height / 2.0
        x2, y2 = cx + width / 2.0, cy + height / 2.0
        if not (width > 0.0 and height > 0.0
                and -1e-7 <= x1 < x2 <= 1.0 + 1e-7
                and -1e-7 <= y1 < y2 <= 1.0 + 1e-7):
            raise ValueError("validation box must be legal normalized cxcywh geometry")
        # Reliability is deliberately ignored: validation uses a uniform mean.
        output.append({**dict(raw), "position": int(position),
                       "physical_frame_id": int(physical_id),
                       "box": box64.to(dtype=torch.float32)})
    return sorted(output, key=lambda row: row["position"])


def _validation_scores(boxes, anchors: Sequence[Mapping[str, Any]], torch):
    """Return per-anchor IoU and its uniform mean, or nulls for an empty set."""
    if not anchors:
        return None, None
    from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
    positions = [int(anchor["position"]) for anchor in anchors]
    evidence = torch.stack([anchor["box"].to(boxes) for anchor in anchors])
    scores = aligned_iou(boxes[positions], evidence).detach()
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("nonfinite validation-reference IoU")
    return scores, scores.mean()


def _validation_not_decreased(before, after, tolerance: float = GATE_TOLERANCE) -> bool:
    import torch
    if before is None or after is None:
        return True
    if not bool(torch.isfinite(before).all()) or not bool(torch.isfinite(after).all()):
        return False
    return float(after) >= float(before) - tolerance


def _val_sha256(anchors: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for anchor in anchors:
        rows.append({"position": int(anchor["position"]),
                     "physical_frame_id": int(anchor["physical_frame_id"]),
                     "box": [float(x) for x in anchor["box"].tolist()]})
    payload = repr(rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fit_cross_frame(
    base, initial: Mapping[str, Any], anchors: Sequence[Mapping[str, Any]],
    validation_anchors: Sequence[Mapping[str, Any]], frame_ids: Sequence[int],
    key: str, config: Mapping[str, Any], *, reference_sha256: str,
    validation_reference_sha256: str | None = None,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the unchanged loss-weighted fit with a disjoint validation gate.

    Train anchors use the original reliability-weighted IoU gate. When at least
    two validation masks are supplied, the validation set adds a uniform-mean
    IoU non-decrease gate. Either gate can reject an Adam proposal; rejection
    restores both model parameters and Adam state exactly and stops later steps.
    With zero or one validation anchor, the original validated helper is called
    directly and the fit is explicitly marked as train-only fallback.
    """
    import torch
    from methods.decota_final_simplified_v1.tensors import detached
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts import stvg_native_instance_spatial_fit_v1 as native
    from scripts import stvg_instance_validated_fit_v1 as original
    from vg_tta.decota_spatial_opd_tunable_v1 import (
        mean_coordinates, optimizer_step, rollout,
    )
    from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou

    verify_source_bindings()
    if not isinstance(config, Mapping) or int(config.get("steps", -1)) != FIT_STEPS:
        raise ValueError("cross-frame validation is fixed to the original five-step fit")
    native._validate_key_and_config(key, config, "joint", "teacher", reference_sha256)
    checked_train = native._validate_anchors(anchors, frame_ids, torch)
    checked_val = _validate_validation_anchors(validation_anchors, frame_ids, torch)
    train_positions_all = [int(anchor["position"]) for anchor in checked_train]
    val_positions = [int(anchor["position"]) for anchor in checked_val]
    overlap = sorted(set(train_positions_all).intersection(val_positions))
    if overlap:
        raise ValueError(f"training and validation positions must be disjoint: {overlap}")
    if validation_reference_sha256 is None:
        validation_reference_sha256 = _val_sha256(checked_val)
    if (not isinstance(validation_reference_sha256, str)
            or len(validation_reference_sha256) != 64
            or any(c not in "0123456789abcdef" for c in validation_reference_sha256.lower())):
        raise ValueError("validation_reference_sha256 must be a 64-character SHA-256")

    validation_enabled = len(checked_val) >= 2
    if not validation_enabled:
        # Keep the prior implementation as the exact fallback math path.
        fit = original.fit_validated(
            base, initial, anchors, frame_ids, key, config,
            validate=True, reference_sha256=reference_sha256,
            checkpoint_callback=checkpoint_callback)
        diagnostics = []
        for row in fit["path"]:
            if checked_val and not row.get("skipped", False):
                before_scores, before_mean = _validation_scores(row["boxes"], checked_val, torch)
                proposed_scores, proposed_mean = _validation_scores(row["proposed_boxes"], checked_val, torch)
                diagnostics.append({
                    "step": int(row["step"]),
                    "validation_iou_before": before_scores.cpu(),
                    "validation_mean_iou_before": float(before_mean),
                    "validation_iou_proposed": proposed_scores.cpu(),
                    "validation_mean_iou_proposed": float(proposed_mean),
                    "used_for_acceptance": False,
                })
            else:
                diagnostics.append({"step": int(row["step"]),
                                    "validation_iou_before": None,
                                    "validation_mean_iou_before": None,
                                    "validation_iou_proposed": None,
                                    "validation_mean_iou_proposed": None,
                                    "used_for_acceptance": False})
        fit.update({
            "schema": "stvg-cross-frame-validated-fit-v1",
            "algorithm": "cross_frame_validated_loss_weighted_iou_train_only_fallback",
            "validation_enabled": False,
            "validation_fallback_count": 1,
            "validation_fallback_reason": "fewer_than_two_validation_anchors",
            "validation_anchor_count": len(checked_val),
            "validation_reference_positions": val_positions,
            "training_reference_positions": train_positions_all,
            "validation_reference_sha256": validation_reference_sha256,
            "validation_metric": "uniform_mean_aligned_iou",
            "validation_path_diagnostics": diagnostics,
            "validation_tolerance": GATE_TOLERANCE,
            "GT_read": False,
        })
        return fit

    started = time.perf_counter()
    base.restore(initial)
    replay, selected = core._trainable_replay(base, list(frame_ids), key, "joint")
    origin = replay.state()
    params = [parameter for _, parameter in selected]
    if sum(parameter.numel() for parameter in params) != 1792:
        raise RuntimeError("joint OPD parameter count differs from the original 1,792 parameters")
    active_train = [anchor for anchor in checked_train if float(anchor["weight"]) > 0.0]
    train_positions = [int(anchor["position"]) for anchor in active_train]
    train_evidence = (torch.stack([anchor["box"].to(base.zero["boxes"]) for anchor in active_train])
                      if active_train else base.zero["boxes"].new_empty((0, 4)))
    train_weights = (torch.as_tensor([anchor["weight"] for anchor in active_train],
                                     dtype=train_evidence.dtype, device=train_evidence.device).detach()
                     if active_train else base.zero["boxes"].new_empty((0,)))
    optimizer = torch.optim.Adam(params, lr=config["lr"], betas=(.9, .999),
                                 eps=1e-8, weight_decay=0.)
    path: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    validation_path: list[dict[str, Any]] = []
    proposals = accepted = 0
    stopped = False
    try:
        for step in range(FIT_STEPS):
            before_state = replay.state()
            before_boxes = replay.values()["boxes"].detach().clone()
            optimizer_before = core._copy_cpu(optimizer.state_dict())
            entry: dict[str, Any] = {
                "step": step,
                "boxes": before_boxes.detach().cpu(),
                "state": detached(before_state, "cpu"),
                "optimizer_before": optimizer_before,
            }
            val_before_scores, val_before_mean = _validation_scores(before_boxes, checked_val, torch)
            if not active_train or stopped:
                reason = "no_training_evidence" if not active_train else "stopped_after_rejected_step"
                entry.update(skipped=True, reason=reason)
                path.append(entry)
                validation_path.append({
                    "step": step,
                    "validation_iou_before": None if val_before_scores is None else val_before_scores.cpu(),
                    "validation_mean_iou_before": None if val_before_mean is None else float(val_before_mean),
                    "validation_iou_proposed": None if val_before_scores is None else val_before_scores.cpu(),
                    "validation_mean_iou_proposed": None if val_before_mean is None else float(val_before_mean),
                    "used_for_acceptance": False,
                    "skipped": True,
                })
                if checkpoint_callback is not None:
                    checkpoint_callback({
                        "next_step": step + 1, "path": path, "rounds": rounds,
                        "state": detached(replay.state(), "cpu"),
                        "optimizer_state": core._copy_cpu(optimizer.state_dict()),
                        "accepted_optimizer_steps": accepted,
                        "proposed_optimizer_steps": proposals,
                        "validation_path_diagnostics": validation_path,
                        "GT_read": False,
                    })
                continue

            before_values = replay.values()
            mean = mean_coordinates(before_values["boxes"][train_positions])
            policy = rollout(mean, train_evidence, key, step, shuffle=False, config=config)
            per_anchor = original.loss_per_anchor(mean, policy, config["sigma"])
            loss = (train_weights * per_anchor).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite original loss-weighted IoU objective")
            train_iou_before = aligned_iou(before_boxes[train_positions], train_evidence).detach()
            update = optimizer_step(optimizer, loss, params, bool(policy["informative"]))
            proposed_state = replay.state()
            with torch.no_grad():
                proposed_boxes = replay.values()["boxes"].detach().clone()
            optimizer_proposed = core._copy_cpu(optimizer.state_dict())
            train_iou_proposed = aligned_iou(proposed_boxes[train_positions], train_evidence).detach()
            val_proposed_scores, val_proposed_mean = _validation_scores(proposed_boxes, checked_val, torch)
            train_accept = (update is not None and original.accept_reference_step(
                train_iou_before, train_iou_proposed, train_weights, GATE_TOLERANCE))
            val_accept = (update is not None and _validation_not_decreased(
                val_before_mean, val_proposed_mean, GATE_TOLERANCE))
            approved = bool(train_accept and val_accept)
            if update is not None:
                proposals += 1
                if approved:
                    accepted += 1
                else:
                    replay.restore(before_state)
                    optimizer.load_state_dict(copy.deepcopy(optimizer_before))
                    stopped = True
                    if not core._tree_equal(core._copy_cpu(optimizer.state_dict()), optimizer_before):
                        raise AssertionError("rejected validation proposal did not restore Adam exactly")
                    if not core._tree_equal(detached(replay.state(), "cpu"), detached(before_state, "cpu")):
                        raise AssertionError("rejected validation proposal did not restore parameters exactly")
            after_state = replay.state()
            with torch.no_grad():
                after_boxes = replay.values()["boxes"].detach().clone()
            optimizer_after = core._copy_cpu(optimizer.state_dict())
            entry.update(
                skipped=False,
                loss=float(loss.detach()),
                anchor_loss=per_anchor.detach().cpu(),
                reliability=train_weights.detach().cpu(),
                rollout=detached(policy, "cpu"),
                update=None if update is None else detached(update, "cpu"),
                proposed_state=detached(proposed_state, "cpu"),
                proposed_boxes=proposed_boxes.detach().cpu(),
                optimizer_proposed=optimizer_proposed,
                optimizer_after=optimizer_after,
                update_state=detached(after_state, "cpu"),
                after_boxes=after_boxes.detach().cpu(),
                training_iou_before=train_iou_before.cpu(),
                training_iou_proposed=train_iou_proposed.cpu(),
                training_mean_iou_before=float((train_weights * train_iou_before).sum() / train_weights.sum()),
                training_mean_iou_proposed=float((train_weights * train_iou_proposed).sum() / train_weights.sum()),
                training_gate_accepted=bool(train_accept),
                validation_gate_accepted=bool(val_accept),
                proposed=update is not None,
                accepted=approved,
                validation_enabled=True,
            )
            path.append(entry)
            val_record = {
                "step": step,
                "validation_iou_before": val_before_scores.cpu(),
                "validation_mean_iou_before": float(val_before_mean),
                "validation_iou_proposed": val_proposed_scores.cpu(),
                "validation_mean_iou_proposed": float(val_proposed_mean),
                "validation_gate_accepted": bool(val_accept),
                "used_for_acceptance": update is not None,
                "accepted": approved,
                "skipped": False,
            }
            validation_path.append(val_record)
            rounds.append({
                "step": step,
                "proposed": update is not None,
                "accepted": approved,
                "training_gate_accepted": bool(train_accept),
                "validation_gate_accepted": bool(val_accept),
                "no_information": not bool(policy["informative"]),
                "training_mean_iou_before": entry["training_mean_iou_before"],
                "training_mean_iou_proposed": entry["training_mean_iou_proposed"],
                "validation_mean_iou_before": float(val_before_mean),
                "validation_mean_iou_proposed": float(val_proposed_mean),
            })
            if checkpoint_callback is not None:
                checkpoint_callback({
                    "next_step": step + 1, "path": path, "rounds": rounds,
                    "state": detached(after_state, "cpu"),
                    "optimizer_state": optimizer_after,
                    "accepted_optimizer_steps": accepted,
                    "proposed_optimizer_steps": proposals,
                    "validation_path_diagnostics": validation_path,
                    "GT_read": False,
                })

        final_state = replay.state()
        with torch.no_grad():
            final_boxes = replay.values()["boxes"].detach().cpu()
        return {
            "schema": "stvg-cross-frame-validated-fit-v1",
            "scope": "joint",
            "active_parameters": 1792,
            "algorithm": "cross_frame_validated_loss_weighted_iou",
            "initial_state": detached(origin, "cpu"),
            "final_state": detached(final_state, "cpu"),
            "final_cached_decoder_boxes": final_boxes,
            "path": path,
            "rounds": rounds,
            "validation_path_diagnostics": validation_path,
            "optimizer_steps": accepted,
            "proposed_optimizer_steps": proposals,
            "rejected_optimizer_steps": proposals - accepted,
            "declared_steps": FIT_STEPS,
            "config": dict(config),
            "seed_key": key,
            "training_reference_positions": train_positions_all,
            "observation_positions": train_positions,
            "validation_reference_positions": val_positions,
            "validation_anchor_count": len(checked_val),
            "validation_enabled": True,
            "validation_fallback_count": 0,
            "validation_fallback_reason": None,
            "reliability": [float(anchor["weight"]) for anchor in active_train],
            "reference_sha256": reference_sha256,
            "validation_reference_sha256": validation_reference_sha256,
            "reference_role": "teacher",
            "reward": "original_iou",
            "loss": "mean_r_times_original_per_anchor_forward_KL",
            "train_gate": "weighted_reference_IoU_non_decrease_1e-7",
            "validation_metric": "uniform_mean_aligned_iou",
            "validation_gate": "uniform_validation_mean_IoU_non_decrease_1e-7",
            "validation_tolerance": GATE_TOLERANCE,
            "adam_rollback_on_rejection": True,
            "rejection_policy": "restore_parameters_and_adam_then_stop_without_retry",
            "GT_read": False,
            "wall_seconds": time.perf_counter() - started,
        }
    finally:
        replay.restore(origin)
        base.restore(initial)


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


def contracts() -> dict[str, Any]:
    """Deterministic CPU-only contracts over the synthetic decoder fixture."""
    import torch
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts import stvg_video_global_spatial_fit_fp32_geometry001 as opd
    from scripts import stvg_instance_validated_fit_v1 as original
    from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU contracts require CUDA to remain uninitialized")
    bindings = verify_source_bindings()
    frame_ids = list(range(8))
    config = core.load_original_opd_config("hc2")
    config["steps"] = FIT_STEPS
    key = opd.locked_seed_key("cross_frame_synthetic")
    train_ref_sha = "a" * 64
    validation_ref_sha = "b" * 64
    base = core._synthetic_student_base(len(frame_ids))
    initial = {name: value.detach().clone() for name, value in base.initial.items()}
    base.restore(initial)
    synthetic_replay, _ = core._trainable_replay(
        base, frame_ids, opd.locked_seed_key("cross_frame_synthetic"), "joint")
    source_boxes = synthetic_replay.values()["boxes"].detach().clone()
    base.restore(initial)

    train_anchor = {"position": 2, "physical_frame_id": frame_ids[2],
                    "box": torch.tensor([.60, .54, .30, .28]), "weight": .7}
    one_validation = {"position": 6, "physical_frame_id": frame_ids[6],
                      "box": source_boxes[6].detach().clone()}

    # Fewer than two validation anchors must retain the original gate exactly.
    for val_anchors in ([], [one_validation]):
        old = original.fit_validated(base, initial, [train_anchor], frame_ids, key, config,
                                     validate=True, reference_sha256=train_ref_sha)
        new = fit_cross_frame(base, initial, [train_anchor], val_anchors,
                              frame_ids, key, config,
                              reference_sha256=train_ref_sha,
                              validation_reference_sha256=validation_ref_sha)
        for field in ("path", "rounds", "initial_state", "final_state",
                      "final_cached_decoder_boxes", "optimizer_steps",
                      "proposed_optimizer_steps", "rejected_optimizer_steps", "config"):
            assert _tree_equal(old[field], new[field]), f"train-only fallback changed {field}"
        assert new["validation_enabled"] is False
        assert new["validation_fallback_count"] == 1

    # Disjointness is strict, including zero-weight training anchors.
    overlapping = {"position": 2, "physical_frame_id": frame_ids[2],
                   "box": source_boxes[2].detach().clone()}
    try:
        fit_cross_frame(base, initial, [train_anchor],
                        [overlapping, one_validation], frame_ids, key, config,
                        reference_sha256=train_ref_sha,
                        validation_reference_sha256=validation_ref_sha)
    except ValueError as exc:
        assert "disjoint" in str(exc)
    else:
        raise AssertionError("overlapping train and validation positions were accepted")

    # The synthetic center-perfect train reference forces a train-gate rejection
    # after an informative proposal; parameters and Adam must return exactly.
    train_perfect = {"position": 2, "physical_frame_id": frame_ids[2],
                     "box": source_boxes[2].detach().clone(), "weight": 1.0}
    val_perfect = [
        {"position": 0, "physical_frame_id": frame_ids[0], "box": source_boxes[0].detach().clone()},
        {"position": 5, "physical_frame_id": frame_ids[5], "box": source_boxes[5].detach().clone()},
    ]
    fit_train_reject = fit_cross_frame(
        base, initial, [train_perfect], val_perfect, frame_ids, key, config,
        reference_sha256=train_ref_sha, validation_reference_sha256=validation_ref_sha)
    first = fit_train_reject["path"][0]
    assert first["proposed"] and not first["training_gate_accepted"]
    assert not first["accepted"] and fit_train_reject["rejected_optimizer_steps"] == 1
    assert _tree_equal(first["state"], first["update_state"])
    assert _tree_equal(first["optimizer_before"], first["optimizer_after"])
    assert all(row.get("reason") == "stopped_after_rejected_step"
               for row in fit_train_reject["path"][1:])

    # Construct a cross-frame conflict: training asks for a shifted box while
    # two held-out positions require their exact initial geometry. Find one
    # stable synthetic key/offset from a small, fixed test list, never data.
    val_refs = [
        {"position": 0, "physical_frame_id": frame_ids[0], "box": source_boxes[0].detach().clone()},
        {"position": 5, "physical_frame_id": frame_ids[5], "box": source_boxes[5].detach().clone()},
    ]
    val_reject = None
    val_reject_key = None
    for suffix in ("valconflict_a", "valconflict_b", "valconflict_c", "valconflict_d"):
        trial_key = opd.locked_seed_key(suffix)
        trial_anchor = {"position": 2, "physical_frame_id": frame_ids[2],
                        "box": torch.tensor([.78, .52, .24, .23]), "weight": 1.0}
        trial = fit_cross_frame(
            base, initial, [trial_anchor], val_refs, frame_ids, trial_key, config,
            reference_sha256=train_ref_sha, validation_reference_sha256=validation_ref_sha)
        row = trial["path"][0]
        if (row.get("proposed") and row.get("training_gate_accepted")
                and not row.get("validation_gate_accepted")):
            val_reject, val_reject_key = trial, trial_key
            break
    assert val_reject is not None, "synthetic held-out conflict did not exercise validation rejection"
    val_first = val_reject["path"][0]
    assert val_first["proposed"] and val_first["training_gate_accepted"]
    assert not val_first["validation_gate_accepted"] and not val_first["accepted"]
    assert val_reject["rejected_optimizer_steps"] == 1
    val_first_diag = val_reject["validation_path_diagnostics"][0]
    assert float(val_first_diag["validation_mean_iou_proposed"]) < float(
        val_first_diag["validation_mean_iou_before"]) - GATE_TOLERANCE
    assert _tree_equal(val_first["state"], val_first["update_state"])
    assert _tree_equal(val_first["optimizer_before"], val_first["optimizer_after"])

    # Tiny, isolated reference gives exactly flat (all-zero) IoU rewards in the
    # fixed Gaussian sample set, exercising informative=False and no update.
    tiny = {"position": 2, "physical_frame_id": frame_ids[2],
            "box": torch.tensor([.001, .001, .001, .001]), "weight": 1.0}
    noinfo_val = [
        {"position": 0, "physical_frame_id": frame_ids[0], "box": source_boxes[0].detach().clone()},
        {"position": 5, "physical_frame_id": frame_ids[5], "box": source_boxes[5].detach().clone()},
    ]
    noinfo_key = opd.locked_seed_key("no_information_cross_frame")
    noinfo = fit_cross_frame(
        base, initial, [tiny], noinfo_val, frame_ids, noinfo_key, config,
        reference_sha256=train_ref_sha, validation_reference_sha256=validation_ref_sha)
    assert noinfo["proposed_optimizer_steps"] == 0 and noinfo["optimizer_steps"] == 0
    assert all(row.get("no_information") for row in noinfo["rounds"])
    assert _tree_equal(noinfo["initial_state"], noinfo["final_state"])
    assert _tree_equal(noinfo["path"][0]["optimizer_before"],
                       noinfo["path"][0]["optimizer_after"])

    assert not torch.cuda.is_initialized()

    # The validation input accepts at most eight distinct source-grid masks.
    val8 = [{"position": position, "physical_frame_id": position,
             "box": torch.tensor([.5, .5, .2, .2])} for position in range(8)]
    assert len(_validate_validation_anchors(val8, list(range(8)), torch)) == 8
    try:
        _validate_validation_anchors(val8 + [{"position": 8, "physical_frame_id": 8,
                                              "box": torch.tensor([.5, .5, .2, .2])}],
                                     list(range(9)), torch)
    except ValueError as exc:
        assert "zero to eight" in str(exc)
    else:
        raise AssertionError("more than eight validation anchors were accepted")

    return {
        "status": "pass",
        "scope": "synthetic CPU cross-frame validation gate contracts",
        "checks": {
            "train_only_exact_fallback_cases": 2,
            "train_validation_disjointness": 1,
            "validation_anchor_limit_0_to_8": 1,
            "train_gate_rejection_and_parameter_adam_rollback": 1,
            "validation_gate_rejection_and_parameter_adam_rollback": 1,
            "flat_reward_exact_no_update": 1,
        },
        "validation_rejection_key": val_reject_key,
        "source_bindings": bindings,
        "GT_read": False,
        "model_weights_read": False,
        "GPU_or_CUDA_used": False,
        "cuda_initialized": False,
    }


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", action="store_true")
    parser.add_argument("--write-contract-receipt", action="store_true")
    args = parser.parse_args()
    if not args.contracts:
        parser.error("this helper only exposes --contracts; fitting is imported by the locked worker")
    result = contracts()
    print(json.dumps(result, sort_keys=True))
    if args.write_contract_receipt:
        output = ROOT / "artifacts/stvg_cross_frame_validated_fit_v1/CPU_FIT_CONTRACTS.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        receipt = dict(result)
        receipt["fit_source_sha256"] = _sha256_file(Path(__file__))
        receipt["receipt_schema"] = "stvg-cross-frame-cpu-fit-contracts-v1"
        output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
