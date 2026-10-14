"""Support-aware, CPU-testable temporal readout variants for STVG.

All non-Adam modes use the existing frozen Projection scores and final
temporal-decoder features. They keep the shared 512-parameter endpoint
readout and the original full-grid MAP decoder. ``unconstrained_ridge`` is a
closed-form residual-fitting procedure, not an Adam or candidate-KL gradient
algorithm.
"""

from __future__ import annotations

from typing import Any

import torch

from methods.decota_final_simplified_v1.objectives import project
from scripts.stvg_opd_p6_existing_common001 import config
from scripts.stvg_sparse_r1_student_v1 import _target_pair
from scripts.stvg_temporal_shared_readout_v1 import CFG as ORIGINAL_CFG
from scripts.stvg_temporal_shared_readout_v1 import fit as _original_fit


CFG = {
    "tau_T": 1.0,
    "l2": 0.001,
    "ridge_scale": 0.001,
    "ridge_lambda_floor": 1e-12,
    "full_reverse_kl_limit": 0.01,
    "steps": 5,
    "numerical_tolerance": 1e-12,
}

MODES = {
    "original_adam",
    "ridge",
    "trust",
    "unconstrained_ridge",
    "per_offset_trust",
}


def _float(value: torch.Tensor) -> float:
    return float(value.detach().double().cpu())


def _assert_cpu(value: Any, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{label} must already be on CPU; this module does not move or run tensors on GPU")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _assert_cpu(item, f"{label}[{i}]")


def _kl(target: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    if target.numel() == 0:
        return scores.sum() * 0.0
    log_p = scores.log_softmax(0)
    return (target * (target.clamp_min(1e-300).log() - log_p)).sum()


def _interval_iou(a: list[int], b: list[int]) -> float:
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return float(intersection / union) if union > 0 else 0.0


def _prepare(case: dict[str, Any], hidden: list[torch.Tensor],
             candidates: list[dict[str, Any]], rewards: list[float | None]) -> dict[str, Any]:
    assert len(hidden) == 2
    assert len(candidates) == len(rewards) >= 1
    assert all(h.ndim == 2 and h.shape[-1] == 256 for h in hidden)
    assert all(r is None or -1.0 <= float(r) <= 1.0 for r in rewards)

    evidence = project(
        case["zero"]["logits"], case["zero"]["actions"],
        case["records"], config(case["dataset"]),
    )
    offsets = evidence["offsets"]
    assert len(offsets) == 2
    hs = [h.detach().double().cpu() for h in hidden]
    for h, record in zip(hs, case["records"]):
        assert h.shape[0] == len(record["frame_ids"])

    valid = [i for i, reward in enumerate(rewards) if reward is not None]
    offset_scores: list[list[torch.Tensor]] = [[], []]
    offset_features: list[list[torch.Tensor]] = [[], []]
    for candidate_index in valid:
        candidate = candidates[candidate_index]
        candidate_scores = []
        candidate_features = []
        for offset_index, (h, evidence_offset, record) in enumerate(
                zip(hs, offsets, case["records"])):
            start, end, _ = _target_pair(record["frame_ids"], candidate["interval_frames"])
            ij = evidence_offset["ij"].detach().cpu()
            matches = torch.nonzero((ij[0] == start) & (ij[1] == end), as_tuple=False)
            assert matches.numel() == 1, "Candidate must map to one legal original span"
            at = int(matches[0, 0])
            candidate_scores.append(evidence_offset["score"].double().cpu()[at])
            candidate_features.append(torch.stack([h[start], h[end]], dim=1).reshape(512))
        for offset_index in range(2):
            offset_scores[offset_index].append(candidate_scores[offset_index])
            offset_features[offset_index].append(candidate_features[offset_index])

    if valid:
        base_by_offset = torch.stack([
            torch.stack(offset_scores[k]) for k in range(2)
        ])  # [2, N]
        phi_by_offset = torch.stack([
            torch.stack(offset_features[k]) for k in range(2)
        ])  # [2, N, 512]
        base = base_by_offset.mean(0)
        phi = phi_by_offset.mean(0)
        r = torch.tensor([float(rewards[i]) for i in valid], dtype=torch.float64)
        p0 = base.softmax(0)
        target = (base + r / CFG["tau_T"]).softmax(0).detach()
        p0_by_offset = base_by_offset.softmax(1)
        target_by_offset = (base_by_offset + r[None, :] / CFG["tau_T"]).softmax(1).detach()
        no_information = len(valid) < 2 or bool((r == r[0]).all())
        if no_information:
            target = p0.detach().clone()
            target_by_offset = p0_by_offset.detach().clone()
    else:
        base_by_offset = torch.empty((2, 0), dtype=torch.float64)
        phi_by_offset = torch.empty((2, 0, 512), dtype=torch.float64)
        base = torch.empty((0,), dtype=torch.float64)
        phi = torch.empty((0, 512), dtype=torch.float64)
        r = torch.empty((0,), dtype=torch.float64)
        p0 = base
        target = base
        p0_by_offset = base_by_offset
        target_by_offset = base_by_offset
        no_information = True

    return {
        "case": case,
        "hidden": hs,
        "candidates": candidates,
        "rewards_all": rewards,
        "valid": valid,
        "rewards": r,
        "evidence": evidence,
        "offsets": offsets,
        "base_by_offset": base_by_offset,
        "phi_by_offset": phi_by_offset,
        "base": base,
        "phi": phi,
        "p0": p0,
        "target": target,
        "p0_by_offset": p0_by_offset,
        "target_by_offset": target_by_offset,
        "no_information": no_information,
    }


def _geometry(ctx: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "per_offset_trust":
        # Preserve each offset's supervision geometry while keeping one shared
        # 512-parameter readout. This deliberately removes mean-feature
        # cancellation across offsets.
        blocks = [ctx["phi_by_offset"][k] - ctx["phi_by_offset"][k].mean(0, keepdim=True)
                  for k in range(2)] if ctx["valid"] else [torch.empty((0, 512), dtype=torch.float64)] * 2
        residual = ((ctx["rewards"] - ctx["rewards"].mean()) / CFG["tau_T"]
                    if ctx["valid"] else torch.empty((0,), dtype=torch.float64))
        design = torch.cat(blocks, dim=0)
        target_residual = torch.cat([residual, residual])
    else:
        design = ctx["phi"] - ctx["phi"].mean(0, keepdim=True) if ctx["valid"] else torch.empty((0, 512), dtype=torch.float64)
        target_residual = ((ctx["rewards"] - ctx["rewards"].mean()) / CFG["tau_T"]
                           if ctx["valid"] else torch.empty((0,), dtype=torch.float64))

    if design.shape[0] == 0:
        singular_values = torch.empty((0,), dtype=torch.float64)
        basis = torch.empty((512, 0), dtype=torch.float64)
        rank = 0
    else:
        _, singular_values, vh = torch.linalg.svd(design, full_matrices=False)
        rank = int(torch.linalg.matrix_rank(design, tol=CFG["numerical_tolerance"]))
        basis = vh[:rank].T.contiguous()
    gram_trace = float((design * design).sum())
    ridge_lambda = max(CFG["ridge_lambda_floor"], CFG["ridge_scale"] * gram_trace / max(rank, 1))
    return {
        "design": design,
        "target_residual": target_residual,
        "rank": rank,
        "singular_values": singular_values,
        "row_basis": basis,
        "gram_trace": gram_trace,
        "ridge_lambda": ridge_lambda,
    }


def _ridge_direction(geometry: dict[str, Any], theta: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    design = geometry["design"]
    if geometry["rank"] == 0 or design.shape[0] == 0:
        zero = torch.zeros((512,), dtype=torch.float64)
        return zero, {"residual_norm": 0.0, "direction_norm": 0.0, "rank_zero": True}
    residual = geometry["target_residual"] - design @ theta.reshape(512)
    gram = design @ design.T
    system = gram + geometry["ridge_lambda"] * torch.eye(gram.shape[0], dtype=torch.float64)
    direction = design.T @ torch.linalg.solve(system, residual)
    assert torch.isfinite(direction).all()
    return direction, {
        "residual_norm": _float(residual.norm()),
        "direction_norm": _float(direction.norm()),
        "rank_zero": False,
    }


def _decode(ctx: dict[str, Any], theta: torch.Tensor) -> dict[str, Any]:
    matrix = theta.reshape(256, 2)
    offset_maps = []
    for hidden, evidence_offset, record in zip(ctx["hidden"], ctx["offsets"], ctx["case"]["records"]):
        ij = evidence_offset["ij"].detach().cpu()
        scores = evidence_offset["score"].double().cpu()
        scores = scores + hidden[ij[0]] @ matrix[:, 0] + hidden[ij[1]] @ matrix[:, 1]
        at = int(scores.argmax())
        start, end = map(int, ij[:, at])
        interval = [int(record["frame_ids"][start]), int(record["frame_ids"][end]) + 1]
        offset_maps.append({"offset_index": len(offset_maps), "indices": [start, end], "interval": interval})
    interval = [min(row["interval"][0] for row in offset_maps),
                max(row["interval"][1] for row in offset_maps)]
    return {"physical_interval": interval, "offset_MAP": offset_maps}


def _scores_and_full_kl(ctx: dict[str, Any], theta: torch.Tensor) -> dict[str, Any]:
    matrix = theta.reshape(256, 2)
    full_reverse_kl = []
    score_drift = []
    offset_candidate_scores = []
    offset_candidate_probs = []
    for k, (hidden, evidence_offset) in enumerate(zip(ctx["hidden"], ctx["offsets"])):
        ij = evidence_offset["ij"].detach().cpu()
        prior_scores = evidence_offset["score"].double().cpu()
        drift = hidden[ij[0]] @ matrix[:, 0] + hidden[ij[1]] @ matrix[:, 1]
        scores = prior_scores + drift
        p_theta = scores.softmax(0)
        log_theta = scores.log_softmax(0)
        log_prior = prior_scores.log_softmax(0)
        full_reverse_kl.append((p_theta * (log_theta - log_prior)).sum())
        score_drift.append({
            "l2": _float(drift.norm()),
            "max_abs": _float(drift.abs().max()) if drift.numel() else 0.0,
            "mean_abs": _float(drift.abs().mean()) if drift.numel() else 0.0,
        })
        if ctx["valid"]:
            vals = []
            for j in ctx["valid"]:
                candidate = ctx["candidates"][j]
                start, end, _ = _target_pair(ctx["case"]["records"][k]["frame_ids"], candidate["interval_frames"])
                at = int(torch.nonzero((ij[0] == start) & (ij[1] == end), as_tuple=False)[0, 0])
                vals.append(scores[at])
            candidate_scores = torch.stack(vals)
            offset_candidate_scores.append(candidate_scores)
            offset_candidate_probs.append(candidate_scores.softmax(0))
        else:
            offset_candidate_scores.append(torch.empty((0,), dtype=torch.float64))
            offset_candidate_probs.append(torch.empty((0,), dtype=torch.float64))
    shared_candidate_scores = ctx["base"] + ctx["phi"] @ theta.reshape(512) if ctx["valid"] else ctx["base"]
    shared_probs = shared_candidate_scores.softmax(0) if ctx["valid"] else ctx["base"]
    return {
        "full_reverse_kl_by_offset": full_reverse_kl,
        "score_drift_by_offset": score_drift,
        "offset_candidate_scores": offset_candidate_scores,
        "offset_candidate_probabilities": offset_candidate_probs,
        "shared_candidate_scores": shared_candidate_scores,
        "shared_candidate_probabilities": shared_probs,
    }


def _objective(ctx: dict[str, Any], theta: torch.Tensor, mode: str) -> torch.Tensor:
    if not ctx["valid"]:
        return theta.sum() * 0.0
    if mode == "per_offset_trust":
        losses = []
        flat = theta.reshape(512)
        for k in range(2):
            scores = ctx["base_by_offset"][k] + ctx["phi_by_offset"][k] @ flat
            losses.append(_kl(ctx["target_by_offset"][k], scores))
        return torch.stack(losses).mean() + CFG["l2"] * theta.square().sum()
    scores = ctx["base"] + ctx["phi"] @ theta.reshape(512)
    return _kl(ctx["target"], scores) + CFG["l2"] * theta.square().sum()


def _compatibility_loss(ctx: dict[str, Any], theta: torch.Tensor) -> torch.Tensor:
    if not ctx["valid"]:
        return theta.sum() * 0.0
    scores = ctx["base"] + ctx["phi"] @ theta.reshape(512)
    return _kl(ctx["target"], scores) + CFG["l2"] * theta.square().sum()


def _ranking(probabilities: torch.Tensor, valid: list[int]) -> list[int]:
    if probabilities.numel() == 0:
        return []
    order = sorted(range(len(valid)), key=lambda i: (-_float(probabilities[i]), valid[i]))
    return [valid[i] for i in order]


def _pool_diagnostic(ctx: dict[str, Any], interval: list[int]) -> dict[str, Any]:
    pool = [list(map(int, c["interval_frames"])) for c in ctx["candidates"]]
    if not pool:
        return {"outside_pool": True, "exact_candidate_indices": [], "best_candidate_iou": None}
    overlaps = [_interval_iou(interval, candidate) for candidate in pool]
    best = max(overlaps)
    return {
        "outside_pool": not any(interval == candidate for candidate in pool),
        "exact_candidate_indices": [i for i, candidate in enumerate(pool) if interval == candidate],
        "best_candidate_iou": best,
        "best_candidate_indices": [i for i, score in enumerate(overlaps) if score == best],
    }


def _snapshot(ctx: dict[str, Any], theta: torch.Tensor, mode: str, step: int,
              geometry: dict[str, Any], update: dict[str, Any] | None) -> dict[str, Any]:
    decoded = _decode(ctx, theta)
    score_data = _scores_and_full_kl(ctx, theta)
    offset_probs = score_data["offset_candidate_probabilities"]
    objective = _objective(ctx, theta, mode)
    compat_loss = _compatibility_loss(ctx, theta)
    ranking = _ranking(score_data["shared_candidate_probabilities"], ctx["valid"])
    offset_rankings = [_ranking(p, ctx["valid"]) for p in offset_probs]
    details = {
        "mode": mode,
        "optimization_objective": _float(objective),
        "compatibility_loss": _float(compat_loss),
        "parameter_norm": _float(theta.norm()),
        "step": update,
        "full_reverse_kl_by_offset": score_data["full_reverse_kl_by_offset"],
        "score_drift_by_offset": score_data["score_drift_by_offset"],
        "shared_candidate_scores": score_data["shared_candidate_scores"].detach().clone(),
        "shared_candidate_probabilities": score_data["shared_candidate_probabilities"].detach().clone(),
        "candidate_ranking": ranking,
        "offset_candidate_scores": [x.detach().clone() for x in score_data["offset_candidate_scores"]],
        "offset_candidate_probabilities": [x.detach().clone() for x in offset_probs],
        "offset_candidate_rankings": offset_rankings,
        "offset_MAP": decoded["offset_MAP"],
        "decoded_outside_pool": _pool_diagnostic(ctx, decoded["physical_interval"]),
        "evidence_rank": geometry["rank"],
        "evidence_subspace_basis": geometry["row_basis"].detach().clone(),
        "centered_design_matrix": geometry["design"].detach().clone(),
        "design_singular_values": geometry["singular_values"].detach().clone(),
        "ridge_lambda": geometry["ridge_lambda"],
        "rank_tolerance": CFG["numerical_tolerance"],
    }
    return {
        "step": step,
        # Keep the legacy key reconstructible by the existing review script.
        "loss": _float(compat_loss),
        "theta": theta.reshape(256, 2).detach().clone(),
        "physical_interval": decoded["physical_interval"],
        "parameter_norm": _float(theta.norm()),
        "candidate_probabilities": score_data["shared_candidate_probabilities"].detach().clone(),
        "objective_loss": _float(objective),
        "diagnostics": details,
    }


def _try_trust_step(ctx: dict[str, Any], theta: torch.Tensor, direction: torch.Tensor,
                    mode: str) -> dict[str, Any]:
    before_objective = _float(_objective(ctx, theta, mode))
    before_kl = [_float(v) for v in _scores_and_full_kl(ctx, theta)["full_reverse_kl_by_offset"]]
    tested = []
    step_sizes = [2.0 ** (-i) for i in range(21)] + [0.0]
    for alpha in step_sizes:
        proposed = theta + alpha * direction.reshape(256, 2)
        objective = _float(_objective(ctx, proposed, mode))
        full_kl = [_float(v) for v in _scores_and_full_kl(ctx, proposed)["full_reverse_kl_by_offset"]]
        accepted = objective <= before_objective + CFG["numerical_tolerance"] and all(
            value <= CFG["full_reverse_kl_limit"] + CFG["numerical_tolerance"] for value in full_kl
        )
        tested.append({"alpha": alpha, "objective": objective, "full_reverse_kl_by_offset": full_kl,
                       "accepted": bool(accepted)})
        if accepted:
            delta = proposed - theta
            return {
                "theta": proposed,
                "alpha": alpha,
                "accepted": alpha > 0.0 and _float(delta.norm()) > CFG["numerical_tolerance"],
                "step_norm": _float(delta.norm()),
                "objective_before": before_objective,
                "objective_after": objective,
                "full_reverse_kl_before": before_kl,
                "full_reverse_kl_after": full_kl,
                "tested_step_sizes": tested,
                "reason": "accepted" if alpha > 0 else "zero_step_fallback",
            }
    return {
        "theta": theta,
        "alpha": None,
        "accepted": False,
        "step_norm": 0.0,
        "objective_before": before_objective,
        "objective_after": before_objective,
        "full_reverse_kl_before": before_kl,
        "full_reverse_kl_after": before_kl,
        "tested_step_sizes": tested,
        "reason": "no_feasible_step",
    }


def _run_closed_form(ctx: dict[str, Any], mode: str) -> dict[str, Any]:
    theta = torch.zeros((256, 2), dtype=torch.float64)
    geometry = _geometry(ctx, mode)
    optimizer_steps = 0
    update_rows: list[dict[str, Any]] = []
    snapshots = []
    stopped = False

    for step in range(CFG["steps"] + 1):
        previous_update = update_rows[-1] if step > 0 and update_rows else None
        snapshots.append(_snapshot(ctx, theta, mode, step, geometry, previous_update))
        if step == CFG["steps"]:
            break
        if stopped:
            update_rows.append({"iteration": step, "accepted": False, "reason": "already_stopped",
                                "step_norm": 0.0, "outside_subspace_norm": 0.0})
            continue
        if ctx["no_information"]:
            update_rows.append({"iteration": step, "accepted": False,
                                "reason": "constant_or_insufficient_rewards", "step_norm": 0.0,
                                "outside_subspace_norm": 0.0})
            stopped = True
            continue
        if mode == "ridge" and step > 0:
            update_rows.append({"iteration": step, "accepted": False, "reason": "one_shot_complete",
                                "step_norm": 0.0, "outside_subspace_norm": 0.0})
            stopped = True
            continue

        direction, solve_info = _ridge_direction(geometry, theta)
        direction_norm = _float(direction.norm())
        if geometry["rank"] == 0 or direction_norm <= CFG["numerical_tolerance"]:
            update_rows.append({"iteration": step, "accepted": False,
                                "reason": "zero_rank_or_direction_below_tolerance",
                                "direction_norm": direction_norm, "step_norm": 0.0,
                                "outside_subspace_norm": 0.0, **solve_info})
            stopped = True
            continue

        if mode in {"trust", "per_offset_trust"}:
            chosen = _try_trust_step(ctx, theta, direction, mode)
            delta_flat = (chosen["theta"] - theta).reshape(512)
            if chosen["accepted"]:
                theta = chosen["theta"]
                optimizer_steps += 1
            else:
                stopped = True
            inside = geometry["row_basis"] @ (geometry["row_basis"].T @ delta_flat) if geometry["rank"] else torch.zeros_like(delta_flat)
            outside_norm = _float((delta_flat - inside).norm())
            update_rows.append({
                "iteration": step,
                "direction_norm": direction_norm,
                "step_norm": _float(delta_flat.norm()),
                "outside_subspace_norm": outside_norm,
                "evidence_rank": geometry["rank"],
                **chosen,
            })
            if not chosen["accepted"] or chosen["alpha"] == 0.0:
                stopped = True
            continue

        delta_flat = direction
        inside = geometry["row_basis"] @ (geometry["row_basis"].T @ delta_flat) if geometry["rank"] else torch.zeros_like(delta_flat)
        outside_norm = _float((delta_flat - inside).norm())
        theta = theta + direction.reshape(256, 2)
        optimizer_steps += 1
        update_rows.append({
            "iteration": step,
            "accepted": True,
            "reason": "one_shot_ridge" if mode == "ridge" else "unconstrained_residual_ridge",
            "direction_norm": direction_norm,
            "step_norm": _float(delta_flat.norm()),
            "outside_subspace_norm": outside_norm,
            "evidence_rank": geometry["rank"],
            "ridge_lambda": geometry["ridge_lambda"],
            **solve_info,
        })
        if mode == "ridge":
            stopped = True

    final = _decode(ctx, theta)
    last_snapshot = snapshots[-1]
    status = "updated" if optimizer_steps else (
        "no_information_exact_zero_parameter_noop" if ctx["no_information"]
        else "no_update_zero_rank_or_no_feasible_direction"
    )
    return {
        "status": status,
        "mode": mode,
        "path": snapshots,
        "final_interval": final["physical_interval"],
        "optimizer_steps": optimizer_steps,
        "active_parameters": 512,
        "original_frozen_features": True,
        "query_reset": True,
        "shared_across_frames": True,
        "per_logit_residual": False,
        "config": {**CFG, "optimizer": "closed_form_ridge_no_adam_preconditioning"},
        "valid_candidate_indices": list(ctx["valid"]),
        "p0": ctx["p0"].detach().clone(),
        "target": ctx["target"].detach().clone(),
        "rewards": ctx["rewards"].detach().clone(),
        "no_information": ctx["no_information"],
        "diagnostics": {
            "algorithm": mode,
            "ridge_design": "stacked_per_offset_centered_features_shared_512" if mode == "per_offset_trust" else "mean_offset_centered_features_shared_512",
            "objective_definition": "per_offset_mean_candidate_KL_plus_l2" if mode == "per_offset_trust" else "legacy_candidate_KL_plus_l2",
            "path_loss_semantics": "loss is the legacy aggregate metric for review compatibility; objective_loss is the optimized objective",
            "evidence_rank": geometry["rank"],
            "evidence_subspace_basis": geometry["row_basis"].detach().clone(),
            "centered_design_matrix": geometry["design"].detach().clone(),
            "design_singular_values": geometry["singular_values"].detach().clone(),
            "ridge_lambda": geometry["ridge_lambda"],
            "gram_trace": geometry["gram_trace"],
            "rank_tolerance": CFG["numerical_tolerance"],
            "full_reverse_kl_limit": CFG["full_reverse_kl_limit"],
            "update_steps": update_rows,
            "final_snapshot": last_snapshot["diagnostics"],
            "per_offset_targets": [x.detach().clone() for x in ctx["target_by_offset"]],
            "per_offset_prior_probabilities": [x.detach().clone() for x in ctx["p0_by_offset"]],
            "GT_read": False,
        },
        "GT_read": False,
    }


def fit(case: dict[str, Any], hidden: list[torch.Tensor], candidates: list[dict[str, Any]],
        rewards: list[float | None], mode: str) -> dict[str, Any]:
    """Fit one readout variant and return six legacy-shaped snapshots.

    Modes are ``original_adam``, ``ridge`` (one unconstrained full ridge step),
    ``trust`` (five residual ridge steps with full-span trust and backtracking),
    ``unconstrained_ridge`` (five residual ridge steps without acceptance), and
    ``per_offset_trust`` (offset-specific candidate KL targets, averaged, with
    the same full-span trust rule).
    """
    if mode not in MODES:
        raise ValueError(f"Unknown temporal fitting mode: {mode!r}")
    _assert_cpu(hidden, "hidden")
    _assert_cpu(case["zero"]["logits"], "zero.logits")
    _assert_cpu(case["zero"]["actions"], "zero.actions")
    if mode == "original_adam":
        result = _original_fit(case, hidden, candidates, rewards)
        ctx = _prepare(case, hidden, candidates, rewards)
        geometry = _geometry(ctx, "ridge")
        previous = torch.zeros((256, 2), dtype=torch.float64)
        reconstructed = []
        for row in result["path"]:
            theta = row["theta"].detach().double().cpu()
            delta = theta - previous
            inside = geometry["row_basis"] @ (geometry["row_basis"].T @ delta.reshape(512)) if geometry["rank"] else torch.zeros(512, dtype=torch.float64)
            current = _snapshot(ctx, theta, "original_adam", int(row["step"]), geometry,
                                {"step_norm": _float(delta.norm()),
                                 "outside_subspace_norm": _float((delta.reshape(512) - inside).norm())})
            # Preserve all original fields while adding the new audit details.
            row["diagnostics"] = current["diagnostics"]
            row["objective_loss"] = row["loss"]
            reconstructed.append(row)
            previous = theta
        result["path"] = reconstructed
        result["mode"] = mode
        result["diagnostics"] = {
            "algorithm": "original shared-readout Adam, called without alteration",
            "objective_definition": "legacy candidate_KL_plus_l2",
            "evidence_rank": geometry["rank"],
            "evidence_subspace_basis": geometry["row_basis"].detach().clone(),
            "centered_design_matrix": geometry["design"].detach().clone(),
            "design_singular_values": geometry["singular_values"].detach().clone(),
            "ridge_lambda_reference": geometry["ridge_lambda"],
            "rank_tolerance": CFG["numerical_tolerance"],
            "per_offset_targets": [x.detach().clone() for x in ctx["target_by_offset"]],
            "GT_read": False,
        }
        return result
    ctx = _prepare(case, hidden, candidates, rewards)
    return _run_closed_form(ctx, mode)


def _synthetic_case() -> tuple[dict[str, Any], list[torch.Tensor], list[dict[str, Any]], list[float]]:
    frame_ids = [0, 1, 2]
    records = [{"frame_ids": frame_ids}, {"frame_ids": frame_ids}]
    logits = [
        torch.tensor([[[0.1, -0.1], [0.2, 0.4], [-0.2, 0.3]]], dtype=torch.float64),
        torch.tensor([[[-0.2, 0.2], [0.5, -0.1], [0.3, 0.1]]], dtype=torch.float64),
    ]
    actions = [
        torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64),
        torch.tensor([-0.2, 0.25, 0.05], dtype=torch.float64),
    ]
    evidence = project(logits, actions, records, config("hc2"))
    intervals = []
    for record, offset in zip(records, evidence["offsets"]):
        at = int(offset["score"].argmax())
        start, end = map(int, offset["ij"][:, at])
        intervals.append([record["frame_ids"][start], record["frame_ids"][end] + 1])
    case = {
        "dataset": "hc2",
        "records": records,
        "frame_ids": frame_ids,
        "Projection_interval": [min(x[0] for x in intervals), max(x[1] for x in intervals)],
        "zero": {"logits": logits, "actions": actions, "records": records},
    }
    hidden = []
    for offset in range(2):
        h = torch.zeros((3, 256), dtype=torch.float64)
        h[:, 0] = torch.tensor([0.0, 0.2 + 0.1 * offset, 0.55], dtype=torch.float64)
        h[:, 1] = torch.tensor([-0.15, 0.25, 0.4 + 0.1 * offset], dtype=torch.float64)
        hidden.append(h)
    candidates = [
        {"interval_frames": [0, 2]},
        {"interval_frames": [1, 3]},
        {"interval_frames": [0, 3]},
    ]
    return case, hidden, candidates, [0.08, -0.04, 0.02]


def contracts() -> dict[str, Any]:
    """Run deterministic synthetic CPU checks without loading project data."""
    case, hidden, candidates, rewards = _synthetic_case()
    checked_modes = sorted(MODES)
    results = {}
    for mode in checked_modes:
        fitted = fit(case, hidden, candidates, rewards, mode)
        assert len(fitted["path"]) == 6
        assert fitted["active_parameters"] == 512
        assert all({"theta", "loss", "physical_interval"} <= set(row) for row in fitted["path"])
        assert fitted["path"][0]["physical_interval"] == case["Projection_interval"]
        assert torch.count_nonzero(fitted["path"][0]["theta"]) == 0
        assert all(torch.isfinite(row["theta"]).all() for row in fitted["path"])
        assert all(torch.isfinite(torch.tensor(row["loss"], dtype=torch.float64)) for row in fitted["path"])
        if mode in {"trust", "per_offset_trust"}:
            for row in fitted["path"]:
                assert all(_float(x) <= CFG["full_reverse_kl_limit"] + CFG["numerical_tolerance"]
                           for x in row["diagnostics"]["full_reverse_kl_by_offset"])
            objectives = [row["objective_loss"] for row in fitted["path"]]
            assert all(b <= a + CFG["numerical_tolerance"] for a, b in zip(objectives, objectives[1:]))
        results[mode] = {"steps": fitted["optimizer_steps"], "final": fitted["final_interval"]}

    # The baseline mode must preserve the old fitter's actual six-step trace.
    direct = _original_fit(case, hidden, candidates, rewards)
    wrapped = fit(case, hidden, candidates, rewards, "original_adam")
    for old_row, new_row in zip(direct["path"], wrapped["path"]):
        assert old_row["step"] == new_row["step"]
        assert old_row["loss"] == new_row["loss"]
        assert old_row["physical_interval"] == new_row["physical_interval"]
        assert torch.equal(old_row["theta"], new_row["theta"])

    ctx = _prepare(case, hidden, candidates, rewards)
    geometry = _geometry(ctx, "ridge")
    direction, _ = _ridge_direction(geometry, torch.zeros((256, 2), dtype=torch.float64))
    residual = geometry["target_residual"]
    gram = geometry["design"] @ geometry["design"].T
    expected = geometry["design"].T @ torch.linalg.solve(
        gram + geometry["ridge_lambda"] * torch.eye(gram.shape[0], dtype=torch.float64), residual)
    assert torch.allclose(direction, expected, atol=CFG["numerical_tolerance"], rtol=0)
    basis = geometry["row_basis"]
    outside = direction - basis @ (basis.T @ direction) if geometry["rank"] else direction
    assert _float(outside.norm()) <= CFG["numerical_tolerance"]

    for mode in checked_modes:
        no_info = fit(case, hidden, candidates, [0.125, 0.125, 0.125], mode)
        assert no_info["no_information"] and no_info["optimizer_steps"] == 0
        assert all(torch.count_nonzero(row["theta"]) == 0 for row in no_info["path"])
    flat_hidden = [torch.ones((3, 256), dtype=torch.float64) for _ in range(2)]
    zero_rank = fit(case, flat_hidden, candidates, rewards, "trust")
    assert zero_rank["diagnostics"]["evidence_rank"] == 0
    assert zero_rank["optimizer_steps"] == 0
    assert all(torch.count_nonzero(row["theta"]) == 0 for row in zero_rank["path"])

    # Per-offset targets must remain distinct when the frozen offset priors differ.
    per = fit(case, hidden, candidates, rewards, "per_offset_trust")
    assert len(per["diagnostics"]["per_offset_targets"]) == 2
    assert not torch.allclose(per["diagnostics"]["per_offset_targets"][0],
                              per["diagnostics"]["per_offset_targets"][1], atol=1e-12, rtol=0)
    return {
        "status": "pass",
        "CPU_only": True,
        "synthetic_only": True,
        "GT_read": False,
        "modes_checked": checked_modes,
        "path_snapshots_per_mode": 6,
        "shared_parameters": 512,
        "checks": [
            "legacy result keys and six-snapshot path",
            "original_adam trace exactly matches direct legacy fit",
            "zero-init Projection readout",
            "closed-form min-norm ridge equation",
            "step lies in centered evidence subspace",
            "constant rewards exact zero-step for every mode",
            "zero-rank exact no-update",
            "trust reverse-KL bounds and non-increasing objective",
            "per-offset targets remain separated when priors differ",
        ],
        "mode_results": results,
    }


if __name__ == "__main__":
    print(contracts())
