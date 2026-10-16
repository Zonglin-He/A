"""Spatial-Lite episodic fitter with explicit response and line-search recipes.

This module preserves the saved joint Query+LN, antithetic Soft-EP update and
matrix-free decoder-response direction. It removes reliability weighting,
the zero-at-anchor policy KL, the antithetic zero-gradient control variate,
response-amplitude acceptance, and default qualification readouts.
"""
from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from methods.decota_final_simplified_v1.tensors import detached
from scripts import stvg_target_teacher_student_v1 as core
from scripts.stvg_weightfree_coordinate_fit002 import accept_reference_step
from vg_tta.decota_spatial_opd_tunable_v1 import mean_coordinates, rollout
from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou


RECIPES = {
    "lite_backtrack": (1.0, 0.5, 0.25, 0.125),
    "lite_fullstep": (1.0,),
}
RESPONSE_SCOPES = {"full_grid", "event_support"}


def flatten(values):
    return torch.cat([value.reshape(-1) for value in values])


def _sync(device):
    """Synchronize only a CUDA device; a CPU invocation never initializes CUDA."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _copy_optimizer_state(state):
    return core._copy_cpu(state)


def _put(rp, selected, state, delta=None):
    """Restore a replay state, optionally adding a flat selected-parameter step."""
    offset = 0
    with torch.no_grad():
        for name, parameter in selected:
            value = state[name].to(parameter)
            if delta is not None:
                size = parameter.numel()
                value = value + delta[offset:offset + size].to(parameter).reshape_as(parameter)
                offset += size
            parameter.copy_(value)
            parameter.grad = None
    if delta is not None and offset != delta.numel():
        raise ValueError("parameter delta has the wrong length")


def _validate_config(cfg, mode):
    if mode not in {"M1", "M2"}:
        raise ValueError("mode must be M1 or M2")
    if not isinstance(cfg, dict):
        raise TypeError("config must be a mapping")
    steps = cfg.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 8:
        raise ValueError("steps must be an integer in [1, 8]")
    if cfg.get("samples") != 32:
        raise ValueError("the locked antithetic sample count must remain 32")
    if cfg.get("writeback") == 0.0625:
        dataset = "hc2"
    elif cfg.get("writeback") == 0.125:
        dataset = "vidstg"
    else:
        raise ValueError("writeback must match a locked source domain")
    original = core.load_original_opd_config(dataset)
    tunable = {"lr", "sigma", "tau", "steps"}
    if set(cfg) != set(original):
        raise ValueError("unexpected OPD config fields")
    if any(cfg[k] != original[k] for k in set(original) - tunable):
        raise ValueError("non-coordinate OPD settings differ from the original config")
    for name, low, high in (("lr", 1e-6, 1.5), ("sigma", 1e-4, 0.4), ("tau", 1e-4, 2.0)):
        value = cfg.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or not low <= float(value) <= high):
            raise ValueError(f"{name} is outside the fixed coordinate-search bounds [{low}, {high}]")
    return dataset


def _validate_settings(settings, mode):
    if not isinstance(settings, dict):
        raise TypeError("settings must be a mapping")
    if mode != "M2":
        return
    for name in ("mu", "cg_max_iter", "cg_rtol"):
        if name not in settings:
            raise ValueError(f"M2 requires settings[{name!r}]")
    mu = settings["mu"]
    max_iter = settings["cg_max_iter"]
    rtol = settings["cg_rtol"]
    if isinstance(mu, bool) or not isinstance(mu, (int, float)) or not math.isfinite(float(mu)) or mu < 0:
        raise ValueError("mu must be finite and nonnegative")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("cg_max_iter must be a positive integer")
    if isinstance(rtol, bool) or not isinstance(rtol, (int, float)) or not math.isfinite(float(rtol)) or rtol <= 0:
        raise ValueError("cg_rtol must be finite and positive")


def _validate_anchors(anchors, frame_count):
    if not isinstance(anchors, (list, tuple)):
        raise TypeError("anchors must be a list or tuple")
    seen = set()
    active = []
    for anchor in anchors:
        if not isinstance(anchor, dict) or not {"position", "box", "weight"} <= set(anchor):
            raise ValueError("each anchor needs position, box, and weight")
        pos = anchor["position"]
        if isinstance(pos, bool) or not isinstance(pos, int) or not 0 <= pos < frame_count:
            raise ValueError("anchor position is outside the supplied frame-id grid")
        if pos in seen:
            raise ValueError("anchor positions must be unique")
        seen.add(pos)
        weight = anchor["weight"]
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight)) or weight < 0):
            raise ValueError("anchor weight must be finite and nonnegative")
        box = torch.as_tensor(anchor["box"])
        if box.shape != (4,) or not torch.isfinite(box).all():
            raise ValueError("anchor box must be a finite cxcywh vector")
        if weight > 0:
            active.append({"position": pos, "box": box.detach(), "weight": float(weight)})
    return active, sorted(seen)


def _protection_positions(frame_count, active_positions, response_scope, event_support_positions):
    if response_scope not in RESPONSE_SCOPES:
        raise ValueError(f"response_scope must be one of {sorted(RESPONSE_SCOPES)}")
    if response_scope == "full_grid":
        if event_support_positions is not None:
            raise ValueError("event_support_positions must be None for full_grid")
        support = list(range(frame_count))
    else:
        if not isinstance(event_support_positions, (list, tuple, set)):
            raise ValueError("event_support requires explicit event_support_positions")
        support = []
        seen = set()
        for pos in event_support_positions:
            if isinstance(pos, bool) or not isinstance(pos, int) or not 0 <= pos < frame_count:
                raise ValueError("event support position is outside the supplied frame-id grid")
            if pos in seen:
                raise ValueError("event support positions must be unique")
            seen.add(pos)
            support.append(pos)
        support = sorted(support)
    active = set(active_positions)
    return support, [pos for pos in support if pos not in active]


def _loss_per_anchor(mean, policy, sigma):
    """Spatial-Lite score-function loss; actions and rewards are detached."""
    logp = -(policy["samples"] - mean[:, None, :]).square().sum(-1) / (2 * sigma ** 2)
    logp = logp - 2 * math.log(2 * math.pi * sigma ** 2)
    return -(policy["weights"] * logp).sum(-1)


def _flatten_grads(grads):
    return torch.cat([torch.zeros_like(parameter).reshape(-1) if grad is None else grad.reshape(-1)
                      for parameter, grad in grads])


def _qualification_witness(rp, selected, state, before_graph, protection, direction):
    """Expensive derivative checks; called only after explicit qualification=True."""
    from scripts.stvg_soft_ep_response_v1 import graph_jvp

    y = before_graph[protection].flatten() / math.sqrt(len(protection) * 4)
    norm = direction.norm()
    if not bool(torch.isfinite(norm)) or float(norm) == 0.0:
        return None, 0.0
    vector = direction.detach() / norm
    generator = torch.Generator(device="cpu").manual_seed(20261015)
    cotangent = torch.randn(len(y), generator=generator, dtype=torch.float32).to(y)
    cotangent = cotangent / cotangent.norm().clamp_min(1e-12)
    jv = graph_jvp(y, [p for _, p in selected], vector)
    vjp_grads = torch.autograd.grad(y, [p for _, p in selected], grad_outputs=cotangent,
                                    retain_graph=True, allow_unused=True)
    vjp = _flatten_grads(list(zip([p for _, p in selected], vjp_grads)))
    left = torch.dot(cotangent, jv)
    right = torch.dot(vjp, vector)
    threshold = 1e-7 + 1e-4 * max(abs(float(left)), abs(float(right)))
    result = {
        "adjoint_left": float(left.detach()),
        "adjoint_right": float(right.detach()),
        "adjoint_abs_error": float((left - right).abs().detach()),
        "adjoint_pass": bool((left - right).abs() <= threshold),
        "unit_direction": vector.detach().cpu(),
        "linear_jvp": jv.detach().cpu(),
        "output_before": y.detach().cpu(),
        "cotangent": cotangent.detach().cpu(),
        "vjp": vjp.detach().cpu(),
    }
    return result, y


def _finite_difference_witness(rp, selected, state, protection, direction, predicted,
                               read_boxes, normalize):
    rows = []
    try:
        for epsilon in (0.002, 0.001):
            _put(rp, selected, state, delta=direction * epsilon)
            plus = normalize(read_boxes("qualification_fd_plus")[protection].flatten())
            _put(rp, selected, state, delta=-direction * epsilon)
            minus = normalize(read_boxes("qualification_fd_minus")[protection].flatten())
            observed = (plus - minus) / (2 * epsilon)
            target = predicted.to(observed)
            error = (observed - target).norm()
            predicted_norm = target.norm()
            rows.append({
                "h": epsilon,
                "plus": plus.detach().cpu(),
                "minus": minus.detach().cpu(),
                "absolute_error": float(error.detach()),
                "predicted_norm": float(predicted_norm.detach()),
                "relative_error": float((error / predicted_norm.clamp_min(1e-12)).detach()),
                "gross_agreement": bool(error <= 1e-3 + 0.25 * predicted_norm),
            })
    finally:
        _put(rp, selected, state)
    return rows


def fit(base, initial, anchors, ids, key, cfg, mode, settings,
        qualification=False, checkpoint=None, *, recipe, response_scope,
        event_support_positions=None):
    """Fit one query using an explicit Spatial-Lite recipe and response support.

    The original positional fitter ABI is preserved through ``checkpoint``.
    New behavior is explicit and keyword-only: ``recipe``, ``response_scope``
    and, for event support, ``event_support_positions``.
    """
    _validate_config(cfg, mode)
    _validate_settings(settings, mode)
    if recipe not in RECIPES:
        raise ValueError(f"recipe must be one of {sorted(RECIPES)}")
    if not isinstance(qualification, bool):
        raise TypeError("qualification must be a bool")
    if not isinstance(ids, (list, tuple)) or not ids:
        raise ValueError("ids must be a nonempty frame-id sequence")
    active, all_anchor_positions = _validate_anchors(anchors, len(ids))
    active_positions = [anchor["position"] for anchor in active]
    event_positions, protection = _protection_positions(
        len(ids), active_positions, response_scope, event_support_positions)

    device = base.zero["boxes"].device
    _sync(device)
    started = time.perf_counter()
    decoder_counts = {
        "round_before": 0,
        "candidate_trial": 0,
        "qualification_fd_plus": 0,
        "qualification_fd_minus": 0,
        "final_cached_readout": 0,
    }
    decoder_seconds = {name: 0.0 for name in decoder_counts}
    solver_seconds_total = 0.0
    solver_calls = 0

    def read_boxes(purpose):
        if purpose not in decoder_counts:
            raise ValueError(f"unknown decoder accounting purpose: {purpose}")
        _sync(device)
        tick = time.perf_counter()
        values = rp.values()
        boxes = values["boxes"]
        _sync(boxes.device)
        decoder_counts[purpose] += 1
        decoder_seconds[purpose] += time.perf_counter() - tick
        return boxes

    base.restore(initial)
    rp, selected = core._trainable_replay(base, list(ids), key, "joint")
    origin = rp.state()
    if sum(parameter.numel() for _, parameter in selected) != 1792:
        raise AssertionError("Spatial-Lite requires the joint 1792-parameter Query+LN state")
    if [name for name, _ in selected] != [name for name, _ in rp.named]:
        raise AssertionError("joint selected parameters must match replay parameter order")
    params = [parameter for _, parameter in selected]
    if any(not parameter.requires_grad for parameter in params):
        raise AssertionError("all joint Query+LN parameters must require gradients")

    evidence = (torch.stack([anchor["box"].to(base.zero["boxes"]) for anchor in active])
                if active else base.zero["boxes"].new_empty((0, 4)))
    anchor_weights = base.zero["boxes"].new_ones((len(active),)).detach()
    shadows = [torch.nn.Parameter(parameter.detach().clone()) for parameter in params]
    optimizer = torch.optim.Adam(shadows, lr=cfg["lr"], betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)

    path = []
    accepted_count = 0
    proposal_count = 0
    stopped = False
    qualification_rows = []
    try:
        for step in range(cfg["steps"]):
            _sync(device)
            step_started = time.perf_counter()
            state_before = rp.state()
            optimizer_before = _copy_optimizer_state(optimizer.state_dict())
            row = {
                "step": step,
                "state_before": detached(state_before, "cpu"),
                "optimizer_before": optimizer_before,
                "optimizer_proposed": None,
                "optimizer_after": optimizer_before,
                "boxes_before": None,
                "boxes_after": None,
                "loss": None,
                "per_anchor_loss": None,
                "mean_gradient": None,
                "gradient": None,
                "original_adam_delta": None,
                "filtered_delta": None,
                "solver": None,
                "candidates": [],
                "accepted": False,
                "rollback": False,
                "accepted_scale": None,
                "skipped": False,
                "reason": None,
                "qualification_diagnostics": None,
                "decoder_counts": {name: 0 for name in decoder_counts},
                "decoder_seconds": {name: 0.0 for name in decoder_counts},
                "solver_seconds": None,
                "step_seconds": None,
            }
            counts_before = dict(decoder_counts)
            seconds_before = dict(decoder_seconds)

            if stopped:
                row.update(skipped=True, reason="stopped_after_rejected_step",
                           state_after=detached(state_before, "cpu"),
                           boxes_after=None)
            elif not active:
                row.update(skipped=True, reason="no_positive_weight_anchor",
                           state_after=detached(state_before, "cpu"),
                           boxes_after=None)
            else:
                before_graph = read_boxes("round_before")
                boxes_before = before_graph.detach().clone()
                row["boxes_before"] = boxes_before.cpu()
                mean = mean_coordinates(before_graph[active_positions])
                policy = rollout(mean, evidence, key, step, config=cfg)
                per_anchor_loss = _loss_per_anchor(mean, policy, cfg["sigma"])
                loss = per_anchor_loss.mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Spatial-Lite score-function loss is non-finite")
                reference_iou_before = aligned_iou(boxes_before[active_positions], evidence).detach()
                row.update(
                    loss=float(loss.detach()),
                    per_anchor_loss=per_anchor_loss.detach().cpu(),
                    rollout=detached(policy, "cpu"),
                    anchor_weights=anchor_weights.cpu(),
                    reference_iou_before=reference_iou_before.cpu(),
                )

                if not policy["informative"]:
                    row.update(skipped=True, reason="no_information",
                               state_after=detached(state_before, "cpu"),
                               boxes_after=boxes_before.cpu(),
                               optimizer_after=_copy_optimizer_state(optimizer.state_dict()))
                else:
                    mean_gradient = torch.autograd.grad(loss, mean, retain_graph=True)[0]
                    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    dense_grads = [torch.zeros_like(parameter) if grad is None else grad
                                   for parameter, grad in zip(params, grads)]
                    if not all(bool(torch.isfinite(grad).all()) for grad in dense_grads):
                        raise FloatingPointError("Spatial-Lite parameter gradient is non-finite")
                    optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        for shadow, parameter, grad in zip(shadows, params, dense_grads):
                            shadow.copy_(parameter)
                            shadow.grad = grad.detach().clone()
                    optimizer.step()
                    adam_delta = flatten([shadow.detach() - parameter.detach()
                                          for shadow, parameter in zip(shadows, params)])
                    optimizer_proposed = _copy_optimizer_state(optimizer.state_dict())
                    proposal_count += 1

                    solver = None
                    solver_seconds = None
                    if mode == "M2":
                        if not protection:
                            direction = adam_delta.clone()
                            solver = {"status": "identity", "reason": "empty_protection_support"}
                        elif float(adam_delta.norm()) == 0.0:
                            direction = adam_delta.clone()
                            solver = {"status": "identity", "reason": "zero_adam_direction"}
                        else:
                            from scripts.stvg_soft_ep_response_v1 import solve_from_graph
                            output = before_graph[protection].flatten() / math.sqrt(len(protection) * 4)
                            _sync(device)
                            solver_started = time.perf_counter()
                            direction, solver = solve_from_graph(
                                output, params, adam_delta, mu=float(settings["mu"]),
                                max_iter=int(settings["cg_max_iter"]),
                                rtol=float(settings["cg_rtol"]))
                            _sync(device)
                            solver_seconds = time.perf_counter() - solver_started
                            solver_calls += 1
                            helper_seconds = solver.get("elapsed_seconds")
                            solver["helper_elapsed_seconds"] = helper_seconds
                            solver["elapsed_seconds"] = solver_seconds
                            solver["timing"] = "synchronized CUDA wall time; CPU wall time on CPU"
                    else:
                        direction = adam_delta.clone()

                    if qualification and step == 0 and mode == "M2" and protection and float(adam_delta.norm()) > 0:
                        _sync(device)
                        qualification_started = time.perf_counter()
                        qcheck, _ = _qualification_witness(
                            rp, selected, state_before, before_graph, protection, adam_delta)
                        if qcheck is not None:
                            qcheck["finite_differences"] = _finite_difference_witness(
                                rp, selected, state_before, protection,
                                qcheck["unit_direction"].to(adam_delta),
                                qcheck["linear_jvp"], read_boxes,
                                lambda value: value / math.sqrt(len(protection) * 4))
                            row["qualification_diagnostics"] = qcheck
                            qualification_rows.append(qcheck)
                        _sync(device)
                        row["qualification_seconds"] = time.perf_counter() - qualification_started

                    candidates = []
                    approved = False
                    chosen_scale = None
                    chosen_boxes = boxes_before
                    for scale in RECIPES[recipe]:
                        _put(rp, selected, state_before, direction * scale)
                        candidate_boxes = read_boxes("candidate_trial").detach().clone()
                        candidate_iou = aligned_iou(candidate_boxes[active_positions], evidence).detach()
                        finite = bool(torch.isfinite(candidate_boxes).all())
                        reference_pass = accept_reference_step(
                            reference_iou_before.to(candidate_iou), candidate_iou, anchor_weights)
                        candidate_accepted = finite and reference_pass
                        candidate = {
                            "scale": scale,
                            "finite": finite,
                            "reference_pass": bool(reference_pass),
                            "accepted": bool(candidate_accepted),
                            "anchor_IoU": candidate_iou.cpu(),
                            "boxes": candidate_boxes.cpu(),
                        }
                        candidates.append(candidate)
                        if candidate_accepted:
                            approved = True
                            chosen_scale = scale
                            chosen_boxes = candidate_boxes
                            break

                    if approved:
                        accepted_count += 1
                        state_after = rp.state()
                        optimizer_after = _copy_optimizer_state(optimizer.state_dict())
                    else:
                        _put(rp, selected, state_before)
                        optimizer.load_state_dict(copy.deepcopy(optimizer_before))
                        if not core._tree_equal(_copy_optimizer_state(optimizer.state_dict()), optimizer_before):
                            raise AssertionError("rejected update did not restore the Adam state")
                        state_after = rp.state()
                        optimizer_after = _copy_optimizer_state(optimizer.state_dict())
                        stopped = True

                    row.update(
                        mean_gradient=mean_gradient.detach().cpu(),
                        gradient=flatten(dense_grads).detach().cpu(),
                        original_adam_delta=adam_delta.detach().cpu(),
                        filtered_delta=direction.detach().cpu(),
                        optimizer_proposed=optimizer_proposed,
                        optimizer_after=optimizer_after,
                        solver=solver,
                        solver_seconds=solver_seconds,
                        candidates=candidates,
                        accepted=approved,
                        rollback=not approved,
                        accepted_scale=chosen_scale,
                        state_after=detached(state_after, "cpu"),
                        boxes_after=chosen_boxes.detach().cpu(),
                    )

            row["decoder_counts"] = {name: decoder_counts[name] - counts_before[name]
                                      for name in decoder_counts}
            row["decoder_seconds"] = {name: decoder_seconds[name] - seconds_before[name]
                                       for name in decoder_counts}
            _sync(device)
            row["step_seconds"] = time.perf_counter() - step_started
            path.append(row)
            if checkpoint is not None:
                checkpoint({
                    "next_step": step + 1,
                    "path": path,
                    "state": detached(row["state_after"], "cpu"),
                    "optimizer_state": row["optimizer_after"],
                    "GT_read": False,
                })

        final_state = detached(rp.state(), "cpu")
        final_boxes = read_boxes("final_cached_readout").detach().cpu()
        _sync(device)
        elapsed = time.perf_counter() - started
        actual_decode_counts = dict(decoder_counts)
        actual_decoder_seconds = dict(decoder_seconds)
        return {
            "schema": "stvg-spatial-lite-fit-v1",
            "mode": mode,
            "recipe": recipe,
            "response_scope": response_scope,
            "event_support_positions": event_positions if response_scope == "event_support" else None,
            "initial_state": detached(origin, "cpu"),
            "final_state": final_state,
            "fit_final_state": final_state,
            "final_cached_decoder_boxes": final_boxes,
            "path": path,
            "optimizer_steps": accepted_count,
            "proposed_optimizer_steps": proposal_count,
            "active_parameters": sum(parameter.numel() for parameter in params),
            "active_names": [name for name, _ in selected],
            "config": dict(cfg),
            "settings": copy.deepcopy(settings),
            "solver_settings": ({key: settings[key] for key in ("mu", "cg_max_iter", "cg_rtol")}
                                if mode == "M2" else None),
            "seed_key": key,
            "observation_positions": active_positions,
            "anchor_positions": all_anchor_positions,
            "response_positions": event_positions if response_scope == "event_support" else list(range(len(ids))),
            "protection_positions": protection,
            "ids_count": len(ids),
            "anchor_weights": [1.0] * len(active),
            "qualification_requested": qualification,
            "qualification_diagnostics": qualification_rows if qualification_rows else None,
            "decode_counts": actual_decode_counts,
            "decoder_seconds_by_purpose": actual_decoder_seconds,
            "decode_calls_total": sum(actual_decode_counts.values()),
            "solver_calls": solver_calls,
            "solver_seconds_total": solver_seconds_total + sum(
                float(row["solver_seconds"] or 0.0) for row in path),
            "wall_seconds": elapsed,
            "GT_read": False,
            "CUDA_used": torch.device(device).type == "cuda",
        }
    finally:
        rp.restore(origin)
        base.restore(initial)

