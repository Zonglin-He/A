"""Evidence-gated, loss-level spatial OPD adapter for the ECLD study.

This module keeps the frozen five-step, 32-sample Gaussian IoU rollout and
1,792-parameter student replay.  It only changes the outer anchor weights:
reliability * source-to-reference novelty * an explicit evidence gate, with a
per-round support gate for non-flat, overlapping IoU rewards.  Importing this
module does not load real data, models, weights, or annotations.
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

from scripts.stvg_weightfree_coordinate_fit_v1 import (
    accept_reference_step,
    loss_per_anchor,
)

ALGORITHM = "evidence_aware_iou"
VALIDATED_ALGORITHM = "validated_evidence_aware_iou"
SCOPE = "joint"
EXPECTED_PARAMETERS = 1792
FIT_STEPS = 5
REWARD_VARIANCE_MIN = 1e-12
OVERLAP_IOU_MIN = 1e-6
MIN_OVERLAP_SAMPLES = 1
NOVELTY_MIN = 1e-6
VALIDATION_TOLERANCE = 1e-7
SOURCE_BINDINGS = {
    "scripts/stvg_native_instance_spatial_fit_v1.py":
        "316078ad420d6ad6fedbdb21cee119523d8b81e4807d42b42afc5942986dafeb",
    "scripts/stvg_weightfree_coordinate_fit_v1.py":
        "4ae4067ca9c35c732564e50a6463d1c6d3d19d2d5ab4f2ae3b689a829d2d707a",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_bindings() -> dict[str, str]:
    """Bind the loss/validation adapter and the original OPD implementation."""
    from scripts import stvg_native_instance_spatial_fit_v1 as native

    actual = native.verify_source_bindings()
    for relative, expected in SOURCE_BINDINGS.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required evidence-aware fit source is missing: {relative}")
        digest = _sha256_file(path)
        if digest != expected:
            raise RuntimeError(f"evidence-aware fit dependency SHA differs: {relative}")
        actual[relative] = digest
    return actual


def _resolve_evidence_gate(anchor: Mapping[str, Any], weight: float) -> tuple[bool, str]:
    """Read an explicit Boolean gate, or use the specified positive-weight fallback."""
    present = [(name, anchor[name]) for name in ("evidence_gate", "grounded") if name in anchor]
    for name, value in present:
        if not isinstance(value, bool):
            raise ValueError(f"anchor {name} must be an explicit Boolean when present")
    if len(present) == 2 and present[0][1] != present[1][1]:
        raise ValueError("anchor evidence_gate and grounded fields conflict")
    if present:
        # They are aliases in this input contract; conflicting aliases fail closed.
        return bool(present[0][1]), "+".join(name for name, _ in present)
    return bool(weight > 0.0), "positive_weight_default"


def _config_for_validation(config: Mapping[str, Any], core):
    """Apply the existing domain/config checks while preserving the caller's locked fit config."""
    if not isinstance(config, Mapping):
        raise ValueError("config must be the locked per-domain OPD mapping")
    if set(config) != {"lr", "sigma", "tau", "steps", "writeback", "samples"}:
        raise ValueError("config must preserve the original six-key OPD schema")
    if (isinstance(config["steps"], bool) or not isinstance(config["steps"], int)
            or config["steps"] != FIT_STEPS):
        raise ValueError("ECLD spatial fit is fixed at the original five Adam steps")
    if (isinstance(config["samples"], bool) or not isinstance(config["samples"], int)
            or config["samples"] != 32):
        raise ValueError("ECLD spatial fit is fixed at the original 32 antithetic samples")
    try:
        writeback = float(config["writeback"])
    except (TypeError, ValueError) as exc:
        raise ValueError("writeback must identify the locked HC or Vid domain") from exc
    if writeback == .0625:
        dataset = "hc2"
    elif writeback == .125:
        dataset = "vidstg"
    else:
        raise ValueError("writeback differs from the original locked per-domain setting")
    original = core.load_original_opd_config(dataset)
    original["steps"] = FIT_STEPS
    fixed = set(original) - {"lr", "sigma", "tau", "steps"}
    if any(config[key] != original[key] for key in fixed):
        raise ValueError("configuration changes a fixed original OPD setting")
    for name, low, high in (("lr", 0.0, .1), ("sigma", 0.0, .4), ("tau", 0.0, 2.0)):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite locked numeric value")
        if not math.isfinite(float(value)) or not low < float(value) <= high:
            raise ValueError(f"{name} is outside the reviewed fit range")
    return dataset, original


def _anchor_rows(anchors, checked, source_boxes, torch, aligned_iou):
    """Join validated anchors with source novelty and explicit evidence gates."""
    rows = []
    raw_by_position = {int(raw["position"]): raw for raw in anchors}
    for anchor in checked:
        position = int(anchor["position"])
        raw = raw_by_position[position]
        weight = float(anchor["weight"])
        gate, gate_source = _resolve_evidence_gate(raw, weight)
        reference_box = anchor["box"].detach().to(source_boxes)
        source_box = source_boxes[position].detach()
        source_iou = float(aligned_iou(source_box, reference_box).detach())
        novelty = max(0.0, min(1.0, 1.0 - source_iou))
        novelty_active = novelty > NOVELTY_MIN
        coefficient = weight * novelty * float(gate) if novelty_active else 0.0
        rows.append({
            **dict(anchor),
            "evidence_gate_resolved": gate,
            "evidence_gate_source": gate_source,
            "source_iou": source_iou,
            "novelty": novelty,
            "novelty_active": novelty_active,
            "external_weight": coefficient,
        })
    return rows


def _snapshot(replay, optimizer, core, detached, step: int, boxes, *, skipped: bool,
              reason: str | None = None) -> dict[str, Any]:
    result = {
        "step": int(step),
        "boxes": boxes.detach().cpu().clone(),
        "state": detached(replay.state(), "cpu"),
        "optimizer_before": core._copy_cpu(optimizer.state_dict()),
        "skipped": bool(skipped),
    }
    if reason is not None:
        result["reason"] = reason
    return result


def fit_evidence_aware(base, initial, anchors, frame_ids, key, config, *, validate,
                       reference_sha256, checkpoint_callback=None):
    """Fit the locked spatial student with evidence-gated outer anchor weights.

    ``anchors`` use the existing mapping schema (``position``,
    ``physical_frame_id``, normalized cxcywh ``box``, and reliability ``weight``)
    with optional Boolean ``evidence_gate`` or ``grounded``.  The latter fields
    are treated as aliases.  An anchor is weighted by
    ``weight * (1 - IoU(source_box, reference_box)) * gate``; each round then
    removes anchors whose actual 32 IoU rewards are flat or have no supported
    overlap.  Accepted proposals retain the old weighted-reference-IoU
    nondecrease gate and rollback/stop behavior.
    """
    import torch
    from methods.decota_final_simplified_v1.tensors import detached
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts import stvg_native_instance_spatial_fit_v1 as native
    from vg_tta.decota_spatial_opd_tunable_v1 import (
        mean_coordinates, optimizer_step, rollout,
    )
    from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou

    started = time.perf_counter()
    if not isinstance(validate, bool):
        raise ValueError("validate must be an explicit Boolean")
    bindings = verify_source_bindings()
    dataset, original = _config_for_validation(config, core)
    native._validate_key_and_config(key, original, SCOPE, "teacher", reference_sha256)
    checked = native._validate_anchors(anchors, frame_ids, torch)
    if len(checked) != len(anchors):
        raise AssertionError("validated anchor count changed unexpectedly")

    base.restore(initial)
    replay, selected = core._trainable_replay(base, list(frame_ids), key, SCOPE)
    origin = replay.state()
    params = [parameter for _, parameter in selected]
    if sum(parameter.numel() for parameter in params) != EXPECTED_PARAMETERS:
        raise RuntimeError("ECLD must retain the original 1,792 joint Query/LN parameters")
    config = dict(config)
    optimizer = torch.optim.Adam(params, lr=config["lr"], betas=(.9, .999),
                                 eps=1e-8, weight_decay=0.)

    try:
        with torch.no_grad():
            source_boxes = replay.values()["boxes"].detach().clone()
        anchor_rows = _anchor_rows(anchors, checked, source_boxes, torch, aligned_iou)
        # Match the original fitter: only positive-reliability anchors enter the
        # rollout and validation mean. Explicitly closed gates remain in this
        # pool for per-anchor diagnostics, but get zero loss coefficient.
        active = [row for row in anchor_rows if float(row["weight"]) > 0.0]
        positions = [int(row["position"]) for row in active]
        evidence = (torch.stack([row["box"].detach().to(base.zero["boxes"])
                                 for row in active]) if active else
                    base.zero["boxes"].new_empty((0, 4)))
        reliability = (base.zero["boxes"].new_tensor([row["weight"] for row in active]).detach()
                       if active else base.zero["boxes"].new_empty((0,)))
        external = (base.zero["boxes"].new_tensor([row["external_weight"] for row in active]).detach()
                    if active else base.zero["boxes"].new_empty((0,)))
        novelty = (base.zero["boxes"].new_tensor([row["novelty"] for row in active]).detach()
                   if active else base.zero["boxes"].new_empty((0,)))
        evidence_gates = (torch.as_tensor([row["evidence_gate_resolved"] for row in active],
                                          dtype=torch.bool, device=base.zero["boxes"].device)
                          if active else torch.empty((0,), dtype=torch.bool,
                                                     device=base.zero["boxes"].device))

        path: list[dict[str, Any]] = []
        rounds: list[dict[str, Any]] = []
        proposed_steps = accepted_steps = 0
        stopped = False

        for step in range(FIT_STEPS):
            before_state = replay.state()
            before_values = replay.values()
            before_boxes = before_values["boxes"].detach().clone()
            optimizer_before = core._copy_cpu(optimizer.state_dict())
            entry = {
                "step": step,
                "boxes": before_boxes.detach().cpu(),
                "state": detached(before_state, "cpu"),
                "optimizer_before": optimizer_before,
            }
            if not active or stopped:
                reason = "no_positive_reliability_anchors" if not active else "stopped_after_rejected_step"
                entry.update(skipped=True, reason=reason,
                             optimizer_after=core._copy_cpu(optimizer.state_dict()),
                             state_after=detached(replay.state(), "cpu"),
                             proposed=False, accepted=False)
                path.append(entry)
                rounds.append({
                    "step": step, "skipped": True, "reason": reason,
                    "reward_variance": [], "reward_min": [], "reward_max": [],
                    "overlap_fraction": [], "novelty": [],
                    "external_weight": [], "effective_external_weight": [],
                    "effective_anchor_count": 0, "loss": None,
                    "proposed": False, "accepted": False,
                    "rollback": False, "no_information": True,
                })
                if checkpoint_callback is not None:
                    checkpoint_callback({
                        "next_step": step + 1, "path": path, "rounds": rounds,
                        "state": detached(replay.state(), "cpu"),
                        "optimizer_state": core._copy_cpu(optimizer.state_dict()),
                        "accepted_optimizer_steps": accepted_steps,
                        "proposed_optimizer_steps": proposed_steps,
                        "GT_read": False,
                    })
                continue

            mean = mean_coordinates(before_values["boxes"][positions])
            policy = rollout(mean, evidence, key, step, config=config)
            rewards = policy["rewards"].detach()
            if rewards.shape != (len(active), 32):
                raise RuntimeError("original OPD rollout must return one IoU for each of 32 samples per anchor")
            reward_variance = rewards.var(dim=1, unbiased=False)
            reward_min = rewards.min(dim=1).values
            reward_max = rewards.max(dim=1).values
            overlap_count = (rewards > OVERLAP_IOU_MIN).sum(dim=1)
            overlap_fraction = overlap_count.to(rewards.dtype) / 32.0
            reward_supported = ((reward_variance > REWARD_VARIANCE_MIN)
                                & (overlap_count >= MIN_OVERLAP_SAMPLES))
            effective = external * reward_supported.to(external.dtype)
            effective_sum = effective.sum()
            per_anchor = loss_per_anchor(mean, policy, config["sigma"])
            if per_anchor.shape != (len(active),) or not bool(torch.isfinite(per_anchor).all()):
                raise FloatingPointError("original per-anchor Gaussian IoU loss is not finite")

            if float(effective_sum.detach()) > 0.0:
                normalized_outer = (effective / effective_sum).detach()
                loss = (normalized_outer * per_anchor).sum()
                if not bool(policy["informative"]):
                    raise AssertionError("supported per-anchor reward variance must be informative")
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("evidence-aware outer loss is not finite")
            else:
                normalized_outer = torch.zeros_like(effective)
                loss = None

            before_iou = aligned_iou(before_boxes[positions], evidence).detach()
            update = None
            proposed_state = before_state
            proposed_boxes = before_boxes
            optimizer_proposed = optimizer_before
            if loss is not None:
                update = optimizer_step(optimizer, loss, params, informative=True)
                proposed_state = replay.state()
                with torch.no_grad():
                    proposed_boxes = replay.values()["boxes"].detach().clone()
                optimizer_proposed = core._copy_cpu(optimizer.state_dict())

            proposed_iou = aligned_iou(proposed_boxes[positions], evidence).detach()
            is_proposal = update is not None
            approved = bool(is_proposal and
                            (not validate or accept_reference_step(
                                before_iou, proposed_iou, reliability,
                                tolerance=VALIDATION_TOLERANCE)))
            if is_proposal:
                proposed_steps += 1
                if approved:
                    accepted_steps += 1
                else:
                    replay.restore(before_state)
                    optimizer.load_state_dict(copy.deepcopy(optimizer_before))
                    stopped = True
                    if not core._tree_equal(core._copy_cpu(optimizer.state_dict()), optimizer_before):
                        raise AssertionError("rejected evidence-aware proposal did not restore Adam state")
                    if not core._tree_equal(detached(replay.state(), "cpu"),
                                            detached(before_state, "cpu")):
                        raise AssertionError("rejected evidence-aware proposal did not restore student state")

            with torch.no_grad():
                after_boxes = replay.values()["boxes"].detach().clone()
            after_state = replay.state()
            optimizer_after = core._copy_cpu(optimizer.state_dict())
            no_information = loss is None
            entry.update(
                skipped=no_information,
                reason="no_admissible_evidence_weight" if no_information else None,
                loss=None if loss is None else float(loss.detach()),
                per_anchor_loss=per_anchor.detach().cpu(),
                reliability=reliability.detach().cpu(),
                novelty=novelty.detach().cpu(),
                evidence_gate=evidence_gates.detach().cpu(),
                external_weight=external.detach().cpu(),
                effective_external_weight=effective.detach().cpu(),
                normalized_outer_weight=normalized_outer.detach().cpu(),
                reward_variance=reward_variance.detach().cpu(),
                reward_min=reward_min.detach().cpu(),
                reward_max=reward_max.detach().cpu(),
                overlap_count=overlap_count.detach().cpu(),
                overlap_fraction=overlap_fraction.detach().cpu(),
                reward_supported=reward_supported.detach().cpu(),
                effective_anchor_count=int((effective > 0).sum().item()),
                rollout=detached(policy, "cpu"),
                update=None if update is None else detached(update, "cpu"),
                proposed_state=detached(proposed_state, "cpu"),
                proposed_boxes=proposed_boxes.detach().cpu(),
                optimizer_proposed=optimizer_proposed,
                optimizer_after=optimizer_after,
                state_after=detached(after_state, "cpu"),
                boxes_after=after_boxes.detach().cpu(),
                reference_iou_before=before_iou.detach().cpu(),
                reference_iou_proposed=proposed_iou.detach().cpu(),
                proposed=is_proposal,
                accepted=approved,
                rollback=bool(is_proposal and not approved),
                validation_enabled=bool(validate),
            )
            path.append(entry)
            rounds.append({
                "step": step,
                "skipped": no_information,
                "no_information": no_information,
                "loss": None if loss is None else float(loss.detach()),
                "reward_variance": reward_variance.detach().cpu(),
                "reward_min": reward_min.detach().cpu(),
                "reward_max": reward_max.detach().cpu(),
                "overlap_count": overlap_count.detach().cpu(),
                "overlap_fraction": overlap_fraction.detach().cpu(),
                "reward_supported": reward_supported.detach().cpu(),
                "novelty": novelty.detach().cpu(),
                "evidence_gate": evidence_gates.detach().cpu(),
                "external_weight": external.detach().cpu(),
                "effective_external_weight": effective.detach().cpu(),
                "normalized_outer_weight": normalized_outer.detach().cpu(),
                "effective_anchor_count": int((effective > 0).sum().item()),
                "per_anchor_loss": per_anchor.detach().cpu(),
                "proposed": is_proposal,
                "accepted": approved,
                "rollback": bool(is_proposal and not approved),
                "no_information_policy": not bool(policy["informative"]),
                "reference_mean_before": (float((reliability * before_iou).sum() / reliability.sum())
                                          if float(reliability.sum()) > 0 else None),
                "reference_mean_proposed": (float((reliability * proposed_iou).sum() / reliability.sum())
                                             if float(reliability.sum()) > 0 else None),
            })
            if checkpoint_callback is not None:
                checkpoint_callback({
                    "next_step": step + 1, "path": path, "rounds": rounds,
                    "state": detached(after_state, "cpu"),
                    "optimizer_state": optimizer_after,
                    "accepted_optimizer_steps": accepted_steps,
                    "proposed_optimizer_steps": proposed_steps,
                    "GT_read": False,
                })

        final_state = replay.state()
        with torch.no_grad():
            final_boxes = replay.values()["boxes"].detach().cpu()
        return {
            "schema": "stvg-ecld-evidence-aware-fit-v1",
            "scope": SCOPE,
            "active_parameters": EXPECTED_PARAMETERS,
            "algorithm": VALIDATED_ALGORITHM if validate else ALGORITHM,
            "initial_state": detached(origin, "cpu"),
            "final_state": detached(final_state, "cpu"),
            "final_cached_decoder_boxes": final_boxes,
            "path": path,
            "rounds": rounds,
            "optimizer_steps": accepted_steps,
            "proposed_optimizer_steps": proposed_steps,
            "rejected_optimizer_steps": proposed_steps - accepted_steps,
            "declared_steps": FIT_STEPS,
            "config": dict(config),
            "seed_key": key,
            "observation_positions": positions,
            "input_reliability": [float(row["weight"]) for row in checked],
            "reliability": [float(row["weight"]) for row in active],
            "anchor_evidence": [{
                "position": int(row["position"]),
                "physical_frame_id": int(row["physical_frame_id"]),
                "reliability": float(row["weight"]),
                "evidence_gate": bool(row["evidence_gate_resolved"]),
                "evidence_gate_source": row["evidence_gate_source"],
                "source_iou": float(row["source_iou"]),
                "novelty": float(row["novelty"]),
                "novelty_active": bool(row["novelty_active"]),
                "external_weight": float(row["external_weight"]),
            } for row in anchor_rows],
            "reference_sha256": reference_sha256,
            "reference_role": "teacher",
            "reward": "original_aligned_iou",
            "loss": "normalized_outer_anchor_weight_times_original_per_anchor_forward_KL",
            "loss_weight": "reliability_times_source_to_reference_novelty_times_evidence_gate",
            "round_admission": {
                "reward_variance_population_strictly_above": REWARD_VARIANCE_MIN,
                "sample_overlap_iou_strictly_above": OVERLAP_IOU_MIN,
                "minimum_overlap_samples": MIN_OVERLAP_SAMPLES,
                "sample_count": 32,
            },
            "validation": ("weighted_reference_IoU_non_decrease_1e-7_then_stop"
                           if validate else "none"),
            "validation_weights": "original_positive_reliability_only",
            "dataset": dataset,
            "source_bindings": bindings,
            "GT_read": False,
            "wall_seconds": time.perf_counter() - started,
        }
    finally:
        replay.restore(origin)
        base.restore(initial)


def contracts() -> dict[str, Any]:
    """Run deterministic synthetic CPU checks; never opens research data or GT."""
    import torch
    from unittest.mock import patch
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts.stvg_weightfree_coordinate_fit_v1 import loss_per_anchor
    from vg_tta.decota_spatial_opd_tunable_v1 import mean_coordinates, rollout

    if torch.cuda.is_initialized():
        raise RuntimeError("ECLD CPU contracts must not initialize CUDA")
    base = core._synthetic_student_base(8)
    cfg = core.load_original_opd_config("hc2")
    cfg["steps"] = FIT_STEPS
    ids = list(range(8))
    key = "caption_query_fixed_space|ecld_synthetic_cpu"
    reference_sha = "a" * 64
    initial = {name: value.detach().clone() for name, value in base.initial.items()}

    # A finite nonzero gradient in the original per-anchor objective, and a
    # central-difference check independent of the decoder/optimizer path.
    initial_boxes = base.zero["boxes"].clone()
    refs = [initial_boxes[1].clone(), initial_boxes[5].clone()]
    refs[0][0] += .06
    refs[0][1] -= .02
    refs[1][0] -= .035
    refs[1][1] += .045
    anchors = [
        {"position": p, "physical_frame_id": p, "box": box,
         "weight": weight, "evidence_gate": True}
        for p, box, weight in ((1, refs[0], .8), (5, refs[1], .4))
    ]
    mean = mean_coordinates(initial_boxes[[1, 5]]).detach().requires_grad_(True)
    evidence = torch.stack(refs)
    policy = rollout(mean.detach(), evidence, key, 0, config=cfg)
    per_anchor = loss_per_anchor(mean, policy, cfg["sigma"])
    from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
    measured_iou = aligned_iou(initial_boxes[[1, 5]], torch.stack(refs))
    ext = torch.tensor([.8, .4], dtype=mean.dtype) * (1.0 - measured_iou)
    normalized = ext / ext.sum()
    objective = (normalized * per_anchor).sum()
    gradient = torch.autograd.grad(objective, mean)[0]
    eps = 2e-3
    plus, minus = mean.detach().clone(), mean.detach().clone()
    plus[0, 0] += eps
    minus[0, 0] -= eps
    fd_policy = rollout(mean.detach(), evidence, key, 0, config=cfg)
    f_plus = (normalized * loss_per_anchor(plus, fd_policy, cfg["sigma"])).sum()
    f_minus = (normalized * loss_per_anchor(minus, fd_policy, cfg["sigma"])).sum()
    finite_difference = (f_plus - f_minus) / (2 * eps)
    assert torch.isfinite(gradient).all() and float(gradient.norm()) > 0
    assert abs(float(gradient[0, 0] - finite_difference)) < 2e-3

    # Same-key policy replay is bitwise stable, and evidence weights stay
    # outside reward temperature: the underlying 32 IoU values do not change.
    policy_repeat = rollout(mean.detach(), evidence, key, 0, config=cfg)
    assert torch.equal(policy["samples"], policy_repeat["samples"])
    assert torch.equal(policy["rewards"], policy_repeat["rewards"])
    temp_scaled = dict(policy)
    scaled_rewards = policy["rewards"] * torch.tensor([.24, .16])[:, None]
    temp_scaled["weights"] = torch.softmax(scaled_rewards / cfg["tau"], dim=-1)
    loss_outer = (normalized * per_anchor.detach()).sum()
    loss_temperature = loss_per_anchor(mean.detach(), temp_scaled, cfg["sigma"]).mean()
    assert not torch.isclose(loss_outer, loss_temperature, rtol=1e-5, atol=1e-7)

    # Anchor sorting in the legacy validator must not detach a gate from its
    # physical position when callers provide an unsorted reference list.
    from scripts import stvg_native_instance_spatial_fit_v1 as native
    from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
    unsorted = [dict(anchors[1], grounded=True), dict(anchors[0], evidence_gate=False)]
    sorted_checked = native._validate_anchors(unsorted, ids, torch)
    joined = _anchor_rows(unsorted, sorted_checked, initial_boxes, torch, aligned_iou)
    assert [(row["position"], row["evidence_gate_resolved"]) for row in joined] == [(1, False), (5, True)]

    # Positive supported signal updates at least once and maintains the fixed
    # five-step/1,792-parameter interface.
    positive = fit_evidence_aware(base, initial, anchors, ids, key, cfg,
                                  validate=False, reference_sha256=reference_sha)
    assert positive["active_parameters"] == EXPECTED_PARAMETERS
    assert positive["declared_steps"] == FIT_STEPS and len(positive["path"]) == FIT_STEPS
    assert positive["proposed_optimizer_steps"] > 0
    assert positive["optimizer_steps"] == positive["proposed_optimizer_steps"]
    assert all(torch.isfinite(torch.as_tensor(row["loss"]))
               for row in positive["rounds"] if row["loss"] is not None)
    assert positive["GT_read"] is False

    validated = fit_evidence_aware(base, initial, anchors, ids, key, cfg,
                                   validate=True, reference_sha256=reference_sha)
    for row in validated["rounds"]:
        if row["accepted"]:
            assert row["reference_mean_proposed"] >= row["reference_mean_before"] - VALIDATION_TOLERANCE

    # Constant zero-IoU rewards are excluded by both variance and O>0 support,
    # and do not alter the student or create Adam moments.
    tiny_box = torch.tensor([.005, .005, .008, .008], dtype=torch.float32)
    flat_anchor = [{"position": 2, "physical_frame_id": 2, "box": tiny_box,
                    "weight": 1.0, "grounded": True}]
    flat = fit_evidence_aware(base, initial, flat_anchor, ids, key + "_flat", cfg,
                              validate=False, reference_sha256=reference_sha)
    assert flat["proposed_optimizer_steps"] == flat["optimizer_steps"] == 0
    assert all(row["loss"] is None and row["effective_anchor_count"] == 0
               for row in flat["rounds"])
    assert all(torch.equal(flat["initial_state"][name], flat["final_state"][name])
               for name in flat["initial_state"])
    assert all(not row["optimizer_after"]["state"] for row in flat["path"])
    assert all(float(row["reward_variance"][0]) == 0.0
               and float(row["overlap_fraction"][0]) == 0.0 for row in flat["rounds"])

    # Explicitly closed gates have exact no-update behavior even with a
    # positive-weight anchor that otherwise has novel geometry.
    closed = [dict(anchors[0], evidence_gate=False)]
    no_gate = fit_evidence_aware(base, initial, closed, ids, key + "_closed", cfg,
                                 validate=False, reference_sha256=reference_sha)
    assert no_gate["proposed_optimizer_steps"] == no_gate["optimizer_steps"] == 0
    assert no_gate["anchor_evidence"][0]["external_weight"] == 0.0
    assert all(torch.equal(no_gate["initial_state"][name], no_gate["final_state"][name])
               for name in no_gate["initial_state"])
    assert all(not row["optimizer_after"]["state"] for row in no_gate["path"])

    # Force one synthetic validation failure to check parameter/Adam rollback
    # and stop-after-rejection, without changing the production gate.
    with patch.object(sys.modules[__name__], "accept_reference_step", return_value=False):
        rejected = fit_evidence_aware(base, initial, anchors, ids, key, cfg,
                                      validate=True, reference_sha256=reference_sha)
    assert rejected["proposed_optimizer_steps"] == 1
    assert rejected["optimizer_steps"] == 0 and rejected["rejected_optimizer_steps"] == 1
    first = rejected["path"][0]
    assert first["proposed"] and not first["accepted"] and first["rollback"]
    assert core._tree_equal(first["state"], first["state_after"])
    assert core._tree_equal(first["optimizer_before"], first["optimizer_after"])
    assert all(row["skipped"] and row.get("reason") == "stopped_after_rejected_step"
               for row in rejected["path"][1:])

    # Gate aliases cannot silently disagree; absent aliases use weight > 0.
    checked_gate, gate_source = _resolve_evidence_gate({"grounded": True}, .25)
    assert checked_gate and gate_source == "grounded"
    default_gate, default_source = _resolve_evidence_gate({}, .25)
    assert default_gate and default_source == "positive_weight_default"
    zero_default, _ = _resolve_evidence_gate({}, 0.0)
    assert not zero_default
    conflict_rejected = False
    try:
        _resolve_evidence_gate({"grounded": True, "evidence_gate": False}, .25)
    except ValueError:
        conflict_rejected = True
    assert conflict_rejected

    assert not torch.cuda.is_initialized()
    return {
        "schema": "stvg-ecld-spatial-cpu-contracts-v1",
        "status": "pass",
        "scope": "synthetic CPU-only fitting and arithmetic checks",
        "contract_count": 11,
        "contracts": {
            "outer_loss_gradient_finite_difference": True,
            "positive_supported_update": True,
            "same_key_same_antithetic_rollout": True,
            "external_weight_is_not_reward_temperature": True,
            "unsorted_anchor_gate_keeps_position_binding": True,
            "constant_flat_reward_exact_no_update": True,
            "all_evidence_gates_closed_exact_no_update": True,
            "evidence_gate_alias_and_default_validation": True,
            "rejected_proposal_restores_state_and_adam_then_stops": True,
            "accepted_validated_step_respects_reference_gate": True,
            "fixed_five_steps_32_samples_1792_parameters": True,
        },
        "model_loaded": False,
        "GPU_initialized": False,
        "real_data_read": False,
        "GT_read": False,
        "teacher_called": False,
        "student_called": False,
        "source_bindings": verify_source_bindings(),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", action="store_true")
    args = parser.parse_args()
    if not args.contracts:
        parser.error("only --contracts is enabled; production fitting is called by the locked worker")
    print(json.dumps(contracts(), sort_keys=True, separators=(",", ":")))
