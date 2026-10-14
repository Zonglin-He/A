"""Source-preserving backtracking wrapper for the original Gaussian spatial OPD.

This module is inert at import time: it does not initialize CUDA, load a model,
or read artifacts.  The fit keeps the original Gaussian rollout, likelihood,
Adam optimizer, key, and configuration.  After each Adam proposal it checks
the actual cached native decoder boxes against the unadapted source boxes and
accepts the largest tested step scale satisfying the fixed per-frame RMS cap.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _frame_rms(boxes, source_boxes):
    """Return RMS displacement across normalized cxcywh coordinates per frame."""
    import torch

    if not torch.is_tensor(boxes) or not torch.is_tensor(source_boxes):
        raise TypeError("boxes and source_boxes must be tensors")
    if boxes.ndim != 2 or boxes.shape[-1] != 4 or source_boxes.shape != boxes.shape:
        raise ValueError("expected matching (frame_count, 4) normalized cxcywh tensors")
    delta = boxes.detach().double() - source_boxes.detach().to(boxes).double()
    return torch.sqrt(torch.mean(delta.square(), dim=-1))


def _state_at_alpha(before: Mapping[str, Any], proposal: Mapping[str, Any], alpha: float):
    import torch

    if set(before) != set(proposal):
        raise ValueError("proposal state keys differ from the pre-step student state")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    result = {}
    for name in before:
        left, right = before[name], proposal[name]
        if not torch.is_tensor(left) or not torch.is_tensor(right) or left.shape != right.shape:
            raise ValueError("student state tensor schema differs")
        result[name] = left.detach() + float(alpha) * (right.detach().to(left) - left.detach())
    return result


def _constraint_summary(boxes, source_boxes, limit: float):
    import torch

    rms = _frame_rms(boxes, source_boxes)
    finite = bool(torch.isfinite(rms).all())
    maximum = float(rms.max()) if len(rms) and finite else (0.0 if not len(rms) else None)
    return {
        "per_frame_rms": rms.detach().cpu(),
        "max_frame_rms": maximum,
        "limit": float(limit),
        "all_source_frames_within_limit": finite and maximum is not None and maximum <= float(limit),
    }


def fit_source_anchored_opd(base, initial, anchors: Sequence[Mapping[str, Any]],
                            frame_ids: Sequence[int], key: str, config: Mapping[str, Any],
                            *, scope: str = "joint", source_frame_rms_limit: float = 0.01,
                            backtracking_halvings: int = 20, resume_state=None,
                            checkpoint_callback=None):
    """Fit the original Gaussian OPD with a fixed source-displacement cap.

    Every informative original Adam proposal is tested at alpha=1, 1/2, ...,
    2^-backtracking_halvings against *all* cached source decoder frames.  The
    first feasible proposal is accepted.  If no proposal is feasible, both the
    active student state and Adam state are restored exactly.  The Gaussian
    action density remains in the original logit space; this helper makes no
    claim that the post-update constraint induces a clipped-box density.
    """
    import copy
    import torch
    from methods.decota_final_simplified_v1.tensors import detached
    from vg_tta.decota_spatial_opd_tunable_v1 import (
        likelihood_loss, mean_coordinates, optimizer_step, rollout,
    )
    from scripts import stvg_target_teacher_student_v1 as core

    cfg = dict(config)
    if scope != "joint":
        raise ValueError("the source-preserving comparison is locked to the 1,792-parameter joint scope")
    if cfg.get("steps") != 5 or cfg.get("samples") != 32:
        raise ValueError("source-anchored OPD is locked to the original 32-sample, five-step recipe")
    if set(cfg) != {"lr", "sigma", "tau", "steps", "writeback", "samples"}:
        raise ValueError("unexpected original OPD configuration keys")
    allowed_configs = []
    for domain in ("hc2", "vidstg"):
        expected = core.load_original_opd_config(domain)
        expected["steps"] = 5
        allowed_configs.append(expected)
    if cfg not in allowed_configs:
        raise ValueError("OPD settings differ from the pinned original per-domain configuration")
    if any(not math.isfinite(float(cfg[k])) or float(cfg[k]) <= 0 for k in ("lr", "sigma", "tau")):
        raise ValueError("OPD lr, sigma, and tau must be finite and positive")
    if isinstance(source_frame_rms_limit, bool) or not math.isfinite(float(source_frame_rms_limit)) or source_frame_rms_limit != 0.01:
        raise ValueError("the source-frame RMS cap is fixed at 0.01")
    if isinstance(backtracking_halvings, bool) or not isinstance(backtracking_halvings, int) or backtracking_halvings != 20:
        raise ValueError("the fixed backtracking path is alpha=1 through 2^-20")
    if len(frame_ids) < 1 or len(set(map(int, frame_ids))) != len(frame_ids):
        raise ValueError("frame_ids must be a nonempty unique physical-frame sequence")

    fit_started = core.time.perf_counter()
    from vg_tta.decota_spatial_opd_tunable_v1 import likelihood_loss, mean_coordinates, optimizer_step, rollout

    base.restore(initial)
    replay, selected = core._trainable_replay(base, frame_ids, key, scope)
    origin = replay.state()
    params = [parameter for _, parameter in selected]
    source_boxes = base.zero["boxes"].detach().clone()
    if source_boxes.shape != (len(frame_ids), 4) or not torch.isfinite(source_boxes).all():
        raise ValueError("source decoder boxes do not match the full physical frame grid")
    if bool(((source_boxes < 0) | (source_boxes > 1)).any()):
        raise ValueError("source decoder boxes must be normalized cxcywh coordinates")

    positions = [int(anchor["position"]) for anchor in anchors]
    if len(set(positions)) != len(positions) or any(not 0 <= p < len(frame_ids) for p in positions):
        raise ValueError("invalid sparse anchor positions")
    for anchor in anchors:
        if int(frame_ids[int(anchor["position"])]) != int(anchor["physical_frame_id"]):
            raise ValueError("anchor physical frame ID does not match its sparse position")
        box = anchor["box"]
        if not torch.is_tensor(box) or tuple(box.shape) != (4,) or not torch.isfinite(box).all():
            raise ValueError("anchor box must be a finite normalized cxcywh tensor")
        if bool(((box < 0) | (box > 1)).any()):
            raise ValueError("anchor box must remain in normalized cxcywh coordinates")
    evidence = (torch.stack([anchor["box"].detach().to(source_boxes) for anchor in anchors])
                if anchors else source_boxes.new_empty((0, 4)))

    optimizer = torch.optim.Adam(params, lr=cfg["lr"], betas=(.9, .999), eps=1e-8, weight_decay=0.)
    next_step, path, rounds, optimizer_steps = core._initialize_fit_resume(
        replay, optimizer, resume_state, arm="SourceAnchoredSpatialOPD", scope=scope,
        key=key, frame_ids=frame_ids, config=cfg, origin=origin)

    def publish_progress(next_index):
        if checkpoint_callback is not None:
            checkpoint_callback(core._fit_progress(
                replay, optimizer, arm="SourceAnchoredSpatialOPD", scope=scope,
                key=key, frame_ids=frame_ids, config=cfg, origin=origin,
                next_step=next_index, path=path, rounds=rounds,
                optimizer_steps=optimizer_steps))

    try:
        for step in range(next_step, cfg["steps"]):
            step_started = core.time.perf_counter()
            before_state = {name: value.detach().clone() for name, value in replay.state().items()}
            before_values = replay.values()
            before_boxes = before_values["boxes"].detach().clone()
            if not torch.equal(before_boxes, source_boxes) and step == 0:
                raise AssertionError("initial cached decoder boxes differ from the frozen source")
            optimizer_before = core._copy_cpu(optimizer.state_dict())
            entry = {
                "step": step,
                "boxes_before": before_boxes.detach().cpu(),
                "source_boxes": source_boxes.detach().cpu(),
                "state_before": detached(before_state, "cpu"),
                "optimizer_state_before": copy.deepcopy(optimizer_before),
                "rng_state_before": core._rng_snapshot(),
            }
            if not positions:
                unchanged_constraint = _constraint_summary(before_boxes, source_boxes, source_frame_rms_limit)
                entry.update(skipped=True, reason="no_verified_sparse_anchors",
                             boxes_after=before_boxes.detach().cpu(),
                             state_after=detached(before_state, "cpu"),
                             optimizer_state_after=core._copy_cpu(optimizer.state_dict()),
                             constraint=unchanged_constraint,
                             step_wall_seconds=core.time.perf_counter() - step_started)
                rounds.append({"step": step, "updated": False, "no_information": True,
                               "reason": "no_verified_sparse_anchors",
                               "constraint": unchanged_constraint,
                               "max_frame_rms": unchanged_constraint["max_frame_rms"],
                               "accepted_alpha": None})
                path.append(entry)
                publish_progress(step + 1)
                continue

            mu = mean_coordinates(before_values["boxes"][positions])
            policy = rollout(mu, evidence, key, step, shuffle=False, config=cfg)
            loss = likelihood_loss(mu, policy["mean"], policy["samples"], policy["weights"],
                                   sigma=cfg["sigma"])
            policy_mean_box = torch.sigmoid((policy["weights"][..., None] * policy["samples"]).sum(-2))
            policy_delta = policy_mean_box - before_boxes[positions].detach()
            proposal_update = optimizer_step(optimizer, loss, params, bool(policy["informative"]))
            if proposal_update is None:
                if not core._tree_equal(before_state, replay.state()):
                    raise AssertionError("no-information OPD changed the student state")
                if not core._tree_equal(optimizer_before, core._copy_cpu(optimizer.state_dict())):
                    raise AssertionError("no-information OPD changed Adam state")
                after_boxes = replay.values()["boxes"].detach().clone()
                step_state = replay.state()
                constraint = _constraint_summary(after_boxes, source_boxes, source_frame_rms_limit)
                entry.update(skipped=False, no_information=True,
                             loss=float(loss.detach()), rollout=detached(policy, "cpu"),
                             proposal_update=None, backtracking_attempts=[], accepted_alpha=None,
                             optimizer_state_after=core._copy_cpu(optimizer.state_dict()),
                             boxes_after=after_boxes.detach().cpu(),
                             state_after=detached(step_state, "cpu"),
                             constraint=constraint,
                             step_wall_seconds=core.time.perf_counter() - step_started)
                rounds.append({"step": step, "updated": False, "no_information": True,
                               "loss": float(loss.detach()), "accepted_alpha": None,
                               "backtracking_attempts": [],
                               "constraint": constraint,
                               "reward_variance": policy["rewards"].var(1, unbiased=False).detach().cpu(),
                               "weight_entropy": (-(policy["weights"] * policy["weights"].clamp_min(1e-30).log()).sum(1)).detach().cpu(),
                               "weight_ess": (1.0 / policy["weights"].square().sum(1)).detach().cpu()})
                path.append(entry)
                publish_progress(step + 1)
                continue

            proposal_state = {name: value.detach().clone() for name, value in replay.state().items()}
            proposal_boxes = replay.values()["boxes"].detach().clone()
            proposal_optimizer_state = core._copy_cpu(optimizer.state_dict())
            attempts = []
            accepted_alpha = None
            accepted_constraint = None
            accepted_boxes = None
            accepted_state = None
            for exponent in range(backtracking_halvings + 1):
                alpha = 2.0 ** (-exponent)
                replay.restore(_state_at_alpha(before_state, proposal_state, alpha))
                with torch.no_grad():
                    candidate_boxes = replay.values()["boxes"].detach().clone()
                constraint = _constraint_summary(candidate_boxes, source_boxes, source_frame_rms_limit)
                attempt = {"alpha": alpha,
                           "max_frame_rms": constraint["max_frame_rms"],
                           "all_source_frames_within_limit": constraint["all_source_frames_within_limit"],
                           "per_frame_rms": constraint["per_frame_rms"]}
                attempts.append(attempt)
                if constraint["all_source_frames_within_limit"]:
                    accepted_alpha = alpha
                    accepted_constraint = constraint
                    accepted_boxes = candidate_boxes
                    accepted_state = {name: value.detach().clone() for name, value in replay.state().items()}
                    break

            if accepted_alpha is None:
                replay.restore(before_state)
                optimizer.load_state_dict(copy.deepcopy(optimizer_before))
                optimizer.zero_grad(set_to_none=True)
                after_state = replay.state()
                after_boxes = replay.values()["boxes"].detach().clone()
                if not core._tree_equal(before_state, after_state):
                    raise AssertionError("fully rejected OPD proposal did not restore the student state")
                if not core._tree_equal(optimizer_before, core._copy_cpu(optimizer.state_dict())):
                    raise AssertionError("fully rejected OPD proposal did not restore Adam state")
                optimizer_steps_accepted = False
                constraint = _constraint_summary(after_boxes, source_boxes, source_frame_rms_limit)
            else:
                after_state = accepted_state
                after_boxes = accepted_boxes
                if not core._tree_equal(proposal_optimizer_state, core._copy_cpu(optimizer.state_dict())):
                    raise AssertionError("accepted backtracking altered the original Adam proposal state")
                optimizer_steps += 1
                optimizer_steps_accepted = True
                constraint = accepted_constraint

            actual_delta = after_boxes - before_boxes
            tangent = core._state_delta(before_state, after_state)
            expected_move = core._decoder_jvp(replay, list(before_state.values()),
                                               list(tangent.values()))
            observed_mask = [i for i in range(len(frame_ids)) if i in set(positions)]
            unobserved_mask = [i for i in range(len(frame_ids)) if i not in set(positions)]
            optimizer_after = core._copy_cpu(optimizer.state_dict())
            entry.update(
                skipped=False,
                no_information=False,
                loss=float(loss.detach()),
                rollout=detached(policy, "cpu"),
                policy_delta=policy_delta.detach().cpu(),
                proposal_update=detached(proposal_update, "cpu"),
                proposal_state=detached(proposal_state, "cpu"),
                proposal_boxes=proposal_boxes.detach().cpu(),
                proposal_optimizer_state=proposal_optimizer_state,
                backtracking_attempts=attempts,
                accepted_alpha=accepted_alpha,
                update_accepted=optimizer_steps_accepted,
                optimizer_state_after=optimizer_after,
                boxes_after=after_boxes.detach().cpu(),
                state_after=detached(after_state, "cpu"),
                expected_delta=expected_move.detach().cpu(),
                actual_delta=actual_delta.detach().cpu(),
                constraint=constraint,
                observed_movement=core._movement_summary(expected_move, actual_delta, observed_mask),
                unobserved_movement=core._movement_summary(expected_move, actual_delta, unobserved_mask),
                actions_detached=not policy["samples"].requires_grad,
                teacher_targets_detached=not evidence.requires_grad,
                reward_weights_detached=not policy["weights"].requires_grad,
                step_wall_seconds=core.time.perf_counter() - step_started)
            rounds.append({
                "step": step, "updated": optimizer_steps_accepted,
                "no_information": False, "proposal_rejected": accepted_alpha is None,
                "loss": float(loss.detach()), "accepted_alpha": accepted_alpha,
                "backtracking_attempts": attempts, "constraint": constraint,
                "reward_variance": policy["rewards"].var(1, unbiased=False).detach().cpu(),
                "weight_entropy": (-(policy["weights"] * policy["weights"].clamp_min(1e-30).log()).sum(1)).detach().cpu(),
                "weight_ess": (1.0 / policy["weights"].square().sum(1)).detach().cpu(),
                "expected_policy_delta_observed": policy_delta.detach().cpu(),
                "actual_delta_all_positions": actual_delta.detach().cpu(),
                "observed_movement": core._movement_summary(expected_move, actual_delta, observed_mask),
                "unobserved_movement": core._movement_summary(expected_move, actual_delta, unobserved_mask),
                "actions_detached": not policy["samples"].requires_grad,
                "teacher_targets_detached": not evidence.requires_grad,
                "reward_weights_detached": not policy["weights"].requires_grad,
            })
            path.append(entry)
            publish_progress(step + 1)

        final_state = {name: value.detach().clone() for name, value in replay.state().items()}
        with torch.no_grad():
            final_cached = replay.values()["boxes"].detach().cpu().clone()
        final_constraint = _constraint_summary(final_cached, source_boxes, source_frame_rms_limit)
        if not final_constraint["all_source_frames_within_limit"]:
            raise AssertionError("final cached boxes violate the source-frame RMS cap")
        return {
            "schema": "ta-stvg-source-anchored-opd-fit-v1",
            "arm": "SourceAnchoredSpatialOPD", "scope": scope, "key": key,
            "initial_state": detached(origin, "cpu"),
            "final_state": detached(final_state, "cpu"),
            "final_cached_decoder_boxes": final_cached,
            "source_decoder_boxes": source_boxes.detach().cpu(),
            "final_constraint": final_constraint,
            "path": path, "rounds": rounds,
            "optimizer_steps": optimizer_steps,
            "proposed_updates": sum(1 for row in rounds if not row.get("no_information", False)),
            "rejected_updates": sum(1 for row in rounds if row.get("proposal_rejected", False)),
            "selected_step": cfg["steps"], "declared_steps": cfg["steps"],
            "active_parameters": core.SCOPE_COUNTS[scope],
            "config": cfg, "source_frame_rms_limit": float(source_frame_rms_limit),
            "backtracking_alphas": [2.0 ** (-i) for i in range(backtracking_halvings + 1)],
            "wall_seconds": core.time.perf_counter() - fit_started,
            "observation_positions": positions,
            "policy": "original antithetic on-policy Gaussian OPD with source-frame RMS backtracking",
            "teacher_box_in_action_pool": False, "GT_read": False,
            "output_source": "fresh full native model readout required",
        }
    finally:
        replay.restore(origin)
        base.restore(initial)
