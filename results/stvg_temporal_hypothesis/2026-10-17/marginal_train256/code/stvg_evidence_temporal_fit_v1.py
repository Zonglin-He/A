"""Evidence-constrained 512-parameter temporal fits for the locked STVG head.

The old ``original`` route delegates byte-for-byte to the existing five-step
Adam fit.  The other routes use deterministic float64 CPU objectives over the
same frozen two-offset scores and endpoint features.  Grid calculations retain
only score/probability vectors and scatter weights; they never materialize a
legal-grid by 512 feature matrix.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from methods.decota_final_simplified_v1.objectives import project
from scripts.stvg_sparse_r1_student_v1 import _target_pair
from scripts.stvg_temporal_shared_readout_v1 import fit as _legacy_table1_fit
from scripts.stvg_coordinate_temporal_fit_v1 import fit as _legacy_coordinate_fit
from scripts.stvg_opd_p6_existing_common001 import config as _method_config


ORIGINAL_CONFIG = {"lr": 0.01, "steps": 5, "tau_T": 1.0, "l2": 0.001}
PROJECTION_CONFIG = {"lr": 0.0, "steps": 0, "tau_T": 1.0, "l2": 0.0}
STABLE_CONFIG = {
    "tau_T": 1.0, "l2": 0.001, "max_iter": 50, "memory": 10,
    "gradient_tolerance": 1e-10, "armijo_c1": 1e-4,
    "initial_step": 1.0, "backtrack_factor": 0.5, "max_backtracks": 40,
}
OFFSET_TRUST_CONFIG = {
    "tau_T": 1.0, "l2": 0.001, "lambda_grid": 1.0,
    "max_iter": 50, "memory": 10, "gradient_tolerance": 1e-10,
    "armijo_c1": 1e-4, "initial_step": 1.0,
    "backtrack_factor": 0.5, "max_backtracks": 40,
}
VARIANTS = ("projection", "original", "stable", "offset_trust")


def _config(variant: str, fit_config: Mapping[str, Any] | None) -> dict[str, Any]:
    defaults = {
        "projection": {}, "original": ORIGINAL_CONFIG,
        "stable": STABLE_CONFIG, "offset_trust": OFFSET_TRUST_CONFIG,
    }[variant]
    supplied = dict(fit_config or {})
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"unsupported {variant} fit_config keys: {sorted(unknown)}")
    cfg = dict(defaults)
    cfg.update(supplied)
    # These values are protocol constants, not a tuning surface.
    for key, expected in defaults.items():
        if cfg[key] != expected:
            raise ValueError(f"{variant}.{key} is fixed at {expected!r}")
    return cfg


def _as_f64(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _logsoftmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("softmax input must be a nonempty finite vector")
    m = float(np.max(x))
    return x - (m + math.log(float(np.exp(x - m).sum())))


def _prob(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lp = _logsoftmax(x)
    return np.exp(lp), lp


def _kl(q: np.ndarray, logq: np.ndarray, logp: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    positive = q > 0
    return float(np.sum(q[positive] * (logq[positive] - logp[positive])))


def _prepare(case: Mapping[str, Any], hidden: Sequence[Any],
             candidates: Sequence[Mapping[str, Any]], rewards: Sequence[float | None]) -> dict[str, Any]:
    if len(hidden) != 2 or len(case.get("records", ())) != 2:
        raise ValueError("the frozen temporal readout requires exactly two offsets")
    if len(candidates) != len(rewards) or len(candidates) < 1:
        raise ValueError("candidates and rewards must have equal, nonzero length")
    if "zero" not in case or "Projection_interval" not in case:
        raise ValueError("case must bind zero logits/actions and Projection_interval")

    ev = project(case["zero"]["logits"], case["zero"]["actions"],
                 case["records"], _method_config(case["dataset"]))
    hs: list[np.ndarray] = []
    offsets: list[dict[str, Any]] = []
    valid: list[int] = []
    rvals: list[float] = []
    for j, reward in enumerate(rewards):
        if reward is None:
            continue
        if isinstance(reward, bool) or not isinstance(reward, (int, float, np.integer, np.floating)):
            raise ValueError("rewards must be finite numbers or None")
        value = float(reward)
        if not math.isfinite(value) or value < -1.0 or value > 1.0:
            raise ValueError("valid alignment rewards must be finite and inside [-1,1]")
        valid.append(j)
        rvals.append(value)

    for h_raw, e_raw, rec in zip(hidden, ev["offsets"], case["records"]):
        h = _as_f64(h_raw, "hidden")
        if h.ndim != 2 or h.shape[1] != 256:
            raise ValueError("each hidden table must have shape [T,256]")
        ids = [int(x) for x in rec["frame_ids"]]
        if len(ids) != h.shape[0] or len(ids) < 2:
            raise ValueError("hidden table length must match its offset frame IDs")
        ij = np.asarray(e_raw["ij"].detach().cpu().numpy(), dtype=np.int64)
        base = _as_f64(e_raw["score"], "projection grid scores").reshape(-1)
        if ij.shape != (2, len(base)) or np.any(ij[0] >= ij[1]):
            raise ValueError("projected legal-span indices have an invalid shape/order")
        if np.any(ij < 0) or np.any(ij >= len(ids)):
            raise ValueError("projected legal-span index is outside the offset grid")
        lookup = {(int(s), int(t)): i for i, (s, t) in enumerate(zip(ij[0], ij[1]))}
        rows = []
        for j in valid:
            interval = np.asarray(candidates[j].get("interval_frames"), dtype=np.float64)
            if interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] >= interval[1]:
                raise ValueError("candidate interval_frames must be a finite increasing pair")
            s, t, overlap = _target_pair(ids, interval)
            try:
                at = lookup[(int(s), int(t))]
            except KeyError as exc:
                raise ValueError("candidate target pair is not in the legal decoder grid") from exc
            rows.append((int(s), int(t), int(at), float(overlap)))
        offsets.append({
            "hidden": h, "ij": ij, "base": base, "ids": ids,
            "candidate_rows": rows,
        })
        hs.append(h)

    if len(offsets) != 2:
        raise ValueError("exactly two offset grids are required")
    r = np.asarray(rvals, dtype=np.float64)
    no_information = len(valid) < 2 or bool(np.all(r == r[0]))

    # Candidate score/features for the original shared-mean objective.
    if valid:
        bases = np.asarray([
            [off["base"][row[2]] for off, row in zip(offsets, rows)]
            for rows in zip(*(off["candidate_rows"] for off in offsets))
        ], dtype=np.float64)
        # The comprehension above is M x 2; preserve the old exact offset mean.
        mean_base = bases.mean(axis=1)
        features = np.zeros((len(valid), 256, 2), dtype=np.float64)
        for k in range(len(valid)):
            for o, off in enumerate(offsets):
                s, t, _, _ = off["candidate_rows"][k]
                features[k, :, 0] += 0.5 * off["hidden"][s]
                features[k, :, 1] += 0.5 * off["hidden"][t]
    else:
        bases = np.empty((0, 2), dtype=np.float64)
        mean_base = np.empty((0,), dtype=np.float64)
        features = np.empty((0, 256, 2), dtype=np.float64)
    return dict(
        evidence=ev, offsets=offsets, valid=valid, rewards=r,
        no_information=no_information, mean_base=mean_base,
        stable_features=features, projection_interval=list(case["Projection_interval"]),
    )


def _scores_for_offset(off: Mapping[str, Any], theta: np.ndarray) -> np.ndarray:
    # O(T + |grid|) temporaries.  In particular, do not index H by every ij
    # pair and form a [|grid|,512] feature table.
    endpoint_start = off["hidden"] @ theta[:, 0]
    endpoint_end = off["hidden"] @ theta[:, 1]
    ij = off["ij"]
    return off["base"] + endpoint_start[ij[0]] + endpoint_end[ij[1]]


def _decode(offsets: Sequence[Mapping[str, Any]], theta: np.ndarray) -> tuple[list[int], list[dict[str, Any]]]:
    intervals = []
    details = []
    for off in offsets:
        scores = _scores_for_offset(off, theta)
        at = int(np.argmax(scores))  # first maximum, matching the original decoder
        s, t = int(off["ij"][0, at]), int(off["ij"][1, at])
        interval = [off["ids"][s], off["ids"][t] + 1]
        intervals.append(interval)
        details.append({"start_index": s, "end_index": t, "physical_interval": interval,
                        "start_boundary": s == 0, "end_boundary": t == len(off["ids"]) - 1,
                        "grid_argmax": at})
    return [min(x[0] for x in intervals), max(x[1] for x in intervals)], details


def _candidate_diagnostics(pre: Mapping[str, Any], theta: np.ndarray,
                           variant: str, tau: float) -> dict[str, Any]:
    offsets = pre["offsets"]
    valid = pre["valid"]
    rewards = pre["rewards"]
    candidate_by_offset = []
    q_by_offset = []
    candidate_kl_by_offset = []
    grid_kl_by_offset = []
    grid_p0_by_offset = []
    grid_p_by_offset = []
    final_cand_logits = []
    for off in offsets:
        rows = off["candidate_rows"]
        base = np.asarray([off["base"][row[2]] for row in rows], dtype=np.float64)
        if len(valid):
            feat_score = np.asarray([
                off["hidden"][s] @ theta[:, 0] + off["hidden"][t] @ theta[:, 1]
                for s, t, _, _ in rows
            ], dtype=np.float64)
            logits = base + feat_score
            p0, lp0 = _prob(base)
            ptheta, lptheta = _prob(logits)
            q, logq = _prob(base + rewards / tau)
            if pre["no_information"]:
                q, logq = p0.copy(), lp0.copy()
            candidate_kl_by_offset.append(_kl(q, logq, lptheta))
            candidate_by_offset.append(ptheta)
            q_by_offset.append(q)
            final_cand_logits.append(logits)
        else:
            p0 = np.empty((0,), dtype=np.float64)
            ptheta = np.empty((0,), dtype=np.float64)
            q = np.empty((0,), dtype=np.float64)
            lp0 = np.empty((0,), dtype=np.float64)
            lptheta = np.empty((0,), dtype=np.float64)
            logq = np.empty((0,), dtype=np.float64)
            candidate_kl_by_offset.append(0.0)
            candidate_by_offset.append(ptheta)
            q_by_offset.append(q)
            final_cand_logits.append(np.empty((0,), dtype=np.float64))
        grid_scores0 = off["base"]
        grid_scores = _scores_for_offset(off, theta)
        pg0, lpg0 = _prob(grid_scores0)
        pg, lpg = _prob(grid_scores)
        grid_kl_by_offset.append(_kl(pg0, lpg0, lpg))
        grid_p0_by_offset.append(pg0)
        grid_p_by_offset.append(pg)

    if variant == "offset_trust":
        objective_candidate_kl = float(sum(candidate_kl_by_offset))
        objective_grid_kl = float(sum(grid_kl_by_offset))
    elif variant in {"stable", "original"} and len(valid):
        base = pre["mean_base"]
        p0mean, lp0mean = _prob(base)
        target, logtarget = _prob(base + rewards / tau)
        if pre["no_information"]:
            target, logtarget = p0mean.copy(), lp0mean.copy()
        # Stable/original optimize one shared-average candidate target.  It is
        # not the sum of the per-offset diagnostic candidate KLs above.
        mean_logits = base + np.einsum("kdc,dc->k", pre["stable_features"], theta)
        _, mean_logp = _prob(mean_logits)
        objective_candidate_kl = _kl(target, logtarget, mean_logp)
        objective_grid_kl = 0.0
    else:
        objective_candidate_kl = 0.0
        objective_grid_kl = 0.0

    return dict(
        candidate_probabilities_by_offset=candidate_by_offset,
        candidate_targets_by_offset=q_by_offset,
        per_offset_candidate_teacher_kl=candidate_kl_by_offset,
        per_offset_grid_kl_p0_to_ptheta=grid_kl_by_offset,
        full_grid_kl_p0_to_ptheta=float(sum(grid_kl_by_offset)),
        per_offset_projection_grid_probabilities=grid_p0_by_offset,
        per_offset_final_grid_probabilities=grid_p_by_offset,
        objective_candidate_kl=float(objective_candidate_kl),
        objective_grid_kl=float(objective_grid_kl),
        l2_penalty=float(0.001 * np.square(theta).sum()),
    )


def _loss_gradient(pre: Mapping[str, Any], theta_flat: np.ndarray,
                    variant: str, cfg: Mapping[str, Any]) -> tuple[float, np.ndarray, dict[str, float]]:
    theta = np.asarray(theta_flat, dtype=np.float64).reshape(256, 2)
    if variant == "stable":
        base = pre["mean_base"]
        feat = pre["stable_features"]
        if not len(base):
            return 0.0, np.zeros_like(theta_flat), {"candidate_kl": 0.0, "grid_kl": 0.0, "l2": 0.0}
        q, logq = _prob(base + pre["rewards"] / cfg["tau_T"])
        p0, logp0 = _prob(base)
        if pre["no_information"]:
            q, logq = p0, logp0
        logits = base + np.einsum("kdc,dc->k", feat, theta)
        p, logp = _prob(logits)
        cand_kl = _kl(q, logq, logp)
        diff = p - q
        grad = np.einsum("k,kdc->dc", diff, feat)
        l2 = float(cfg["l2"] * np.square(theta).sum())
        grad += 2.0 * cfg["l2"] * theta
        return cand_kl + l2, grad.reshape(-1), {"candidate_kl": cand_kl, "grid_kl": 0.0, "l2": l2}

    if variant != "offset_trust":
        raise ValueError("analytic optimizer supports stable and offset_trust only")
    total = 0.0
    grad = np.zeros((256, 2), dtype=np.float64)
    cand_total = 0.0
    grid_total = 0.0
    for off in pre["offsets"]:
        rows = off["candidate_rows"]
        if rows:
            cidx = np.asarray([x[2] for x in rows], dtype=np.int64)
            sidx = np.asarray([x[0] for x in rows], dtype=np.int64)
            eidx = np.asarray([x[1] for x in rows], dtype=np.int64)
            base = off["base"][cidx]
            candidate_logits = base + off["hidden"][sidx] @ theta[:, 0] + off["hidden"][eidx] @ theta[:, 1]
            p, logp = _prob(candidate_logits)
            q, logq = _prob(base + pre["rewards"] / cfg["tau_T"])
            if pre["no_information"]:
                q, logq = _prob(base)
            delta = p - q
            kl = _kl(q, logq, logp)
            cand_total += kl
            total += kl
            grad[:, 0] += off["hidden"][sidx].T @ delta
            grad[:, 1] += off["hidden"][eidx].T @ delta

        grid0 = off["base"]
        grid = _scores_for_offset(off, theta)
        p0, lp0 = _prob(grid0)
        ptheta, lptheta = _prob(grid)
        grid_kl = _kl(p0, lp0, lptheta)
        grid_total += grid_kl
        total += cfg["lambda_grid"] * grid_kl
        diff = cfg["lambda_grid"] * (ptheta - p0)
        starts = np.bincount(off["ij"][0], weights=diff, minlength=len(off["hidden"]))
        ends = np.bincount(off["ij"][1], weights=diff, minlength=len(off["hidden"]))
        grad[:, 0] += off["hidden"].T @ starts
        grad[:, 1] += off["hidden"].T @ ends

    l2 = float(cfg["l2"] * np.square(theta).sum())
    total += l2
    grad += 2.0 * cfg["l2"] * theta
    return total, grad.reshape(-1), {"candidate_kl": cand_total,
                                    "grid_kl": grid_total,
                                    "l2": l2}


def _lbfgs_armijo(pre: Mapping[str, Any], variant: str,
                  cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic unbounded L-BFGS (the no-active-bounds L-BFGS-B case)."""
    x = np.zeros((512,), dtype=np.float64)
    loss_fn = lambda z: _loss_gradient(pre, z, variant, cfg)
    f, g, terms = loss_fn(x)
    path = [(x.copy(), float(f), dict(terms), 0.0)]
    history: list[tuple[np.ndarray, np.ndarray, float]] = []
    rejected = 0
    termination = "max_iterations"
    accepted = 0
    if pre["no_information"]:
        return dict(x=x, path=path, accepted=0, rejected=0, termination="no_information")

    for _iteration in range(int(cfg["max_iter"])):
        if float(np.linalg.norm(g, ord=np.inf)) <= cfg["gradient_tolerance"]:
            termination = "gradient_tolerance"
            break
        q = g.copy()
        alphas: list[float] = []
        for s, y, rho in reversed(history):
            alpha = rho * float(np.dot(s, q))
            alphas.append(alpha)
            q -= alpha * y
        if history:
            s_last, y_last, _ = history[-1]
            yy = float(np.dot(y_last, y_last))
            gamma = float(np.dot(s_last, y_last)) / yy if yy > 0 else 1.0
        else:
            gamma = 1.0
        r = gamma * q
        for (s, y, rho), alpha in zip(history, reversed(alphas)):
            beta = rho * float(np.dot(y, r))
            r += s * (alpha - beta)
        direction = -r
        slope = float(np.dot(g, direction))
        if not math.isfinite(slope) or slope >= 0.0:
            direction = -g
            slope = -float(np.dot(g, g))
            history.clear()
        if not math.isfinite(slope) or slope >= 0.0:
            termination = "no_descent_direction"
            break

        step = float(cfg["initial_step"])
        accepted_trial = None
        for _bt in range(int(cfg["max_backtracks"]) + 1):
            trial = x + step * direction
            ft, gt, termst = loss_fn(trial)
            armijo = f + float(cfg["armijo_c1"]) * step * slope
            if math.isfinite(ft) and ft < f and ft <= armijo:
                accepted_trial = (trial, ft, gt, termst, step)
                break
            rejected += 1
            step *= float(cfg["backtrack_factor"])
        if accepted_trial is None:
            termination = "armijo_rejected_no_further_update"
            break

        xn, fn, gn, termsn, stepn = accepted_trial
        svec = xn - x
        yvec = gn - g
        curvature = float(np.dot(svec, yvec))
        threshold = 1e-12 * float(np.linalg.norm(svec)) * float(np.linalg.norm(yvec))
        if curvature > max(threshold, 0.0):
            history.append((svec.copy(), yvec.copy(), 1.0 / curvature))
            if len(history) > int(cfg["memory"]):
                history.pop(0)
        x, f, g, terms = xn, float(fn), gn, dict(termsn)
        accepted += 1
        path.append((x.copy(), f, dict(terms), float(stepn)))
    return dict(x=x, path=path, accepted=accepted, rejected=rejected, termination=termination)


def _torch(value: Any):
    import torch
    return torch.as_tensor(value, dtype=torch.float64, device="cpu").clone()


def _make_snapshot(step: int, theta: np.ndarray, loss: float,
                   offsets: Sequence[Mapping[str, Any]], variant: str,
                   pre: Mapping[str, Any], cfg: Mapping[str, Any],
                   step_size: float, terms: Mapping[str, float]) -> dict[str, Any]:
    interval, decodes = _decode(offsets, theta)
    diag = _candidate_diagnostics(pre, theta, variant, float(cfg.get("tau_T", 1.0)))
    if pre["valid"] and variant in {"stable", "original", "projection"}:
        mean_logits = pre["mean_base"] + np.einsum("kdc,dc->k", pre["stable_features"], theta)
        mean_probs, _ = _prob(mean_logits)
    elif pre["valid"]:
        mean_probs = np.mean(np.stack(diag["candidate_probabilities_by_offset"]), axis=0)
    else:
        mean_probs = np.empty((0,), dtype=np.float64)
    return {
        "step": int(step), "loss": float(loss), "theta": _torch(theta),
        "physical_interval": interval, "parameter_norm": float(np.linalg.norm(theta)),
        "candidate_probabilities": _torch(mean_probs),
        "candidate_probabilities_by_offset": [_torch(x) for x in diag["candidate_probabilities_by_offset"]],
        "candidate_target_probabilities_by_offset": [_torch(x) for x in diag["candidate_targets_by_offset"]],
        "candidate_kl": float(terms.get("candidate_kl", 0.0)),
        "grid_kl": float(terms.get("grid_kl", 0.0)),
        "l2_penalty": float(terms.get("l2", 0.0)),
        "per_offset_candidate_kl": list(diag["per_offset_candidate_teacher_kl"]),
        "per_offset_grid_kl": list(diag["per_offset_grid_kl_p0_to_ptheta"]),
        "offset_decodes": decodes,
        "line_search_step_size": float(step_size),
    }


def _result_from_solver(case: Mapping[str, Any], pre: Mapping[str, Any], variant: str,
                        cfg: Mapping[str, Any], solver: Mapping[str, Any]) -> dict[str, Any]:
    theta = np.asarray(solver["x"], dtype=np.float64).reshape(256, 2)
    path = []
    for step, (x, loss, terms, step_size) in enumerate(solver["path"]):
        path.append(_make_snapshot(step, np.asarray(x).reshape(256, 2), loss,
                                   pre["offsets"], variant, pre, cfg, step_size, terms))
    final_interval, final_decodes = _decode(pre["offsets"], theta)
    initial_decodes = _decode(pre["offsets"], np.zeros((256, 2), dtype=np.float64))[1]
    diagnostics = _candidate_diagnostics(pre, theta, variant, float(cfg.get("tau_T", 1.0)))
    stable_loss = _loss_gradient(pre, theta.reshape(-1), "stable", STABLE_CONFIG)[0]
    jumps = []
    for before, after in zip(initial_decodes, final_decodes):
        jumps.append({
            "start_index_delta": int(after["start_index"] - before["start_index"]),
            "end_index_delta": int(after["end_index"] - before["end_index"]),
            "start_physical_delta": int(after["physical_interval"][0] - before["physical_interval"][0]),
            "end_physical_delta": int(after["physical_interval"][1] - before["physical_interval"][1]),
            "final_start_at_grid_boundary": bool(after["start_boundary"]),
            "final_end_at_grid_boundary": bool(after["end_boundary"]),
        })
    boundary_jump_count = sum(
        int(jump["start_index_delta"] != 0 and jump["final_start_at_grid_boundary"])
        + int(jump["end_index_delta"] != 0 and jump["final_end_at_grid_boundary"])
        for jump in jumps
    )
    if pre["valid"]:
        base_mean = pre["mean_base"]
        p0, _ = _prob(base_mean)
        target, _ = _prob(base_mean + pre["rewards"] / float(cfg.get("tau_T", 1.0)))
        if pre["no_information"]:
            target = p0.copy()
    else:
        p0 = target = np.empty((0,), dtype=np.float64)
    if pre["no_information"]:
        status = "no_information_exact_zero_parameter_noop"
    elif not solver["accepted"]:
        status = "line_search_rejected_exact_zero_parameter_noop"
    else:
        status = "updated" if variant == "stable" else "offset_trust_updated"
    return {
        "status": status, "variant": variant, "path": path,
        "final_interval": final_interval, "intervals": [x["physical_interval"] for x in final_decodes],
        "loss": float(path[-1]["loss"]), "stable_loss": float(stable_loss),
        "optimizer_steps": int(solver["accepted"]),
        "accepted_steps": int(solver["accepted"]),
        "rejected_line_search_trials": int(solver["rejected"]),
        "termination": solver["termination"],
        "active_parameters": 512,
        "actual_nonzero_parameter_count": int(np.count_nonzero(theta)),
        "original_frozen_features": True, "query_reset": True,
        "shared_across_frames": True, "per_logit_residual": False,
        "config": dict(cfg), "valid_candidate_indices": list(pre["valid"]),
        "p0": _torch(p0), "target": _torch(target),
        "rewards": _torch(pre["rewards"]),
        "no_information": bool(pre["no_information"]),
        "per_offset_candidate_targets": [_torch(x) for x in diagnostics["candidate_targets_by_offset"]],
        "per_offset_candidate_probabilities": [_torch(x) for x in diagnostics["candidate_probabilities_by_offset"]],
        "candidate_probabilities": path[-1]["candidate_probabilities"],
        "candidate_probabilities_by_offset": path[-1]["candidate_probabilities_by_offset"],
        "grid_kl_per_offset": list(diagnostics["per_offset_grid_kl_p0_to_ptheta"]),
        "per_offset_grid_kl": list(diagnostics["per_offset_grid_kl_p0_to_ptheta"]),
        "per_offset_candidate_kl": list(diagnostics["per_offset_candidate_teacher_kl"]),
        "diagnostics": {
            "per_offset_candidate_teacher_kl": diagnostics["per_offset_candidate_teacher_kl"],
            "per_offset_grid_kl_p0_to_ptheta": diagnostics["per_offset_grid_kl_p0_to_ptheta"],
            "full_grid_kl_p0_to_ptheta": diagnostics["full_grid_kl_p0_to_ptheta"],
            "objective_candidate_kl": diagnostics["objective_candidate_kl"],
            "objective_grid_kl": diagnostics["objective_grid_kl"],
            "l2_penalty": float(0.001 * np.square(theta).sum()),
            "initial_offset_decodes": initial_decodes,
            "final_offset_decodes": final_decodes,
            "offset_endpoint_jumps": jumps,
            "boundary_jump_count": int(boundary_jump_count),
            "final_boundary_endpoint_count": sum(int(x["start_boundary"]) + int(x["end_boundary"]) for x in final_decodes),
            "grid_size_by_offset": [int(len(x["base"])) for x in pre["offsets"]],
            "grid_feature_matrix_materialized": False,
            "termination": solver["termination"],
        },
        "GT_read": False,
    }


def fit(case: Mapping[str, Any], hidden: Sequence[Any],
        candidates: Sequence[Mapping[str, Any]], rewards: Sequence[float | None],
        variant: str, fit_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run one locked temporal variant with the shared 512-parameter readout.

    ``projection`` is an exact zero-parameter output. ``original`` delegates
    to the original Adam implementation. ``stable`` minimizes the original
    mean-candidate KL+L2 objective with deterministic Armijo L-BFGS. The
    ``offset_trust`` objective uses per-offset candidate KL plus unit-weight
    full-legal-grid KL ``KL(p0 || p_theta)`` and the original L2 term.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    cfg = _config(variant, fit_config)
    pre = _prepare(case, hidden, candidates, rewards)
    zero = np.zeros((256, 2), dtype=np.float64)
    zero_interval, _ = _decode(pre["offsets"], zero)
    projection_interval = [int(x) for x in pre["projection_interval"]]
    if zero_interval != projection_interval:
        raise AssertionError("zero shared readout must exactly preserve Projection_interval")

    if variant == "projection":
        diagnostics = _candidate_diagnostics(pre, zero, "projection", 1.0)
        stable_loss, _, stable_terms = _loss_gradient(pre, zero.reshape(-1), "stable", STABLE_CONFIG)
        snapshot = _make_snapshot(0, zero, stable_loss, pre["offsets"], "projection", pre,
                                  {"tau_T": 1.0}, 0.0, stable_terms)
        p0, _ = _prob(pre["mean_base"]) if pre["valid"] else (np.empty((0,)), np.empty((0,)))
        projection_decodes = _decode(pre["offsets"], zero)[1]
        return {
            "status": "projection_only", "variant": variant, "path": [snapshot],
            "final_interval": zero_interval, "intervals": [x["physical_interval"] for x in projection_decodes],
            "loss": float(stable_loss), "stable_loss": float(stable_loss),
            "optimizer_steps": 0, "accepted_steps": 0,
            "rejected_line_search_trials": 0, "active_parameters": 512,
            "actual_nonzero_parameter_count": 0, "original_frozen_features": True,
            "query_reset": True, "shared_across_frames": True, "per_logit_residual": False,
            "config": dict(PROJECTION_CONFIG), "valid_candidate_indices": list(pre["valid"]),
            "p0": _torch(p0), "target": _torch(p0), "rewards": _torch(pre["rewards"]),
            "no_information": bool(pre["no_information"]), "diagnostics": {
                "per_offset_candidate_teacher_kl": diagnostics["per_offset_candidate_teacher_kl"],
                "per_offset_grid_kl_p0_to_ptheta": diagnostics["per_offset_grid_kl_p0_to_ptheta"],
                "grid_kl_per_offset": diagnostics["per_offset_grid_kl_p0_to_ptheta"],
                "full_grid_kl_p0_to_ptheta": diagnostics["full_grid_kl_p0_to_ptheta"],
                "objective_candidate_kl": stable_terms["candidate_kl"],
                "objective_grid_kl": 0.0,
                "initial_offset_decodes": projection_decodes,
                "final_offset_decodes": projection_decodes,
                "grid_size_by_offset": [int(len(x["base"])) for x in pre["offsets"]],
                "grid_feature_matrix_materialized": False,
                "boundary_jump_count": 0,
            },
            "candidate_probabilities": snapshot["candidate_probabilities"],
            "candidate_probabilities_by_offset": snapshot["candidate_probabilities_by_offset"],
            "per_offset_candidate_targets": [_torch(x) for x in diagnostics["candidate_targets_by_offset"]],
            "per_offset_candidate_probabilities": [_torch(x) for x in diagnostics["candidate_probabilities_by_offset"]],
            "grid_kl_per_offset": list(diagnostics["per_offset_grid_kl_p0_to_ptheta"]),
            "per_offset_grid_kl": list(diagnostics["per_offset_grid_kl_p0_to_ptheta"]),
            "per_offset_candidate_kl": list(diagnostics["per_offset_candidate_teacher_kl"]),
            "GT_read": False,
        }

    if variant == "original":
        # This is intentionally an exact delegation; don't recompute or
        # normalize any old path values, Adam state, or probability tensors.
        import torch
        legacy_hidden = [h if hasattr(h, "detach") else torch.as_tensor(h) for h in hidden]
        result = _legacy_table1_fit(case, legacy_hidden, candidates, rewards)
        theta = _as_f64(result["path"][-1]["theta"], "legacy final theta")
        if theta.shape != (256, 2):
            raise AssertionError("legacy readout theta must remain [256,2]")
        di = _candidate_diagnostics(pre, theta, "original", cfg["tau_T"])
        initial = _decode(pre["offsets"], zero)[1]
        final = _decode(pre["offsets"], theta)[1]
        result = dict(result)
        result.update({
            "variant": variant,
            "loss": float(result["path"][-1]["loss"]),
            "stable_loss": float(_loss_gradient(pre, theta.reshape(-1), "stable", STABLE_CONFIG)[0]),
            "intervals": [x["physical_interval"] for x in final],
            "actual_nonzero_parameter_count": int(np.count_nonzero(theta)),
            "grid_kl_per_offset": list(di["per_offset_grid_kl_p0_to_ptheta"]),
            "per_offset_grid_kl": list(di["per_offset_grid_kl_p0_to_ptheta"]),
            "per_offset_candidate_kl": list(di["per_offset_candidate_teacher_kl"]),
            "candidate_probabilities": result["path"][-1]["candidate_probabilities"],
            "candidate_probabilities_by_offset": [_torch(x) for x in di["candidate_probabilities_by_offset"]],
            "diagnostics": {
                "per_offset_candidate_teacher_kl": di["per_offset_candidate_teacher_kl"],
                "per_offset_grid_kl_p0_to_ptheta": di["per_offset_grid_kl_p0_to_ptheta"],
                "full_grid_kl_p0_to_ptheta": di["full_grid_kl_p0_to_ptheta"],
                "objective_candidate_kl": di["objective_candidate_kl"],
                "objective_grid_kl": 0.0,
                "initial_offset_decodes": initial, "final_offset_decodes": final,
                "grid_size_by_offset": [int(len(x["base"])) for x in pre["offsets"]],
                "grid_feature_matrix_materialized": False,
                "objective_warning": "original fixed five-step Adam may increase its own objective",
            },
        })
        return result

    solver = _lbfgs_armijo(pre, variant, cfg)
    result = _result_from_solver(case, pre, variant, cfg, solver)
    final_theta = solver["x"].reshape(256, 2)
    final_diag = _candidate_diagnostics(pre, final_theta, variant, cfg["tau_T"])
    result["diagnostics"].update({
        "per_offset_candidate_targets": [x.tolist() for x in final_diag["candidate_targets_by_offset"]],
        "per_offset_candidate_probabilities": [x.tolist() for x in final_diag["candidate_probabilities_by_offset"]],
        "target_objective_is_convex_for_fixed_feedback_and_features": True,
        "does_not_correct_teacher_candidate_ordering": True,
    })
    return result


def _synthetic_case(seed: int = 41) -> tuple[dict[str, Any], list[np.ndarray], list[dict[str, Any]], list[float | None]]:
    import torch
    from scripts.stvg_opd_p6_existing_common001 import config as method_config

    rng = np.random.default_rng(seed)
    ids = ([0, 2, 4, 6], [1, 3, 5, 7])
    records = [dict(offset=o, flip=False, frame_ids=list(row)) for o, row in enumerate(ids)]
    logits = [torch.tensor(rng.normal(0, 0.15, size=(1, 4, 2)), dtype=torch.float32) for _ in ids]
    actions = [torch.tensor(rng.normal(0, 0.4, size=(1, 4, 1)), dtype=torch.float32) for _ in ids]
    hidden = [rng.normal(0, 0.2, size=(4, 256)).astype(np.float64) for _ in ids]
    zero = {"logits": logits, "actions": actions}
    from methods.decota_final_simplified_v1.objectives import project
    e = project(logits, actions, records, method_config("hc2"))
    per = []
    for ex, rec in zip(e["offsets"], records):
        ij = ex["ij"].numpy()
        at = int(ex["score"].argmax())
        per.append([rec["frame_ids"][int(ij[0, at])], rec["frame_ids"][int(ij[1, at])] + 1])
    projection = [min(x[0] for x in per), max(x[1] for x in per)]
    candidates = [
        {"canonical_id": f"T{i+1:02}", "interval_frames": interval}
        for i, interval in enumerate(([0, 5], [2, 7], [1, 6], [3, 8]))
    ]
    rewards = [0.8, -0.7, 0.25, -0.35]
    case = {"dataset": "hc2", "zero": zero, "records": records,
            "Projection_interval": projection, "GT_read": False}
    return case, hidden, candidates, rewards


def contracts() -> dict[str, Any]:
    """Run synthetic CPU checks only; no case archive, GT, media, or model."""
    global _loss_gradient
    import torch

    case, hidden, candidates, rewards = _synthetic_case()
    variants = {v: fit(case, hidden, candidates, rewards, v) for v in VARIANTS}
    import torch
    legacy_hidden = [torch.as_tensor(h) for h in hidden]
    legacy = _legacy_table1_fit(case, legacy_hidden, candidates, rewards)
    coordinate = _legacy_coordinate_fit(case, legacy_hidden, candidates, rewards,
                                        fit_config=ORIGINAL_CONFIG)
    assert variants["original"]["path"][-1]["physical_interval"] == legacy["path"][-1]["physical_interval"]
    assert torch.equal(variants["original"]["path"][-1]["theta"], legacy["path"][-1]["theta"])
    assert variants["original"]["optimizer_steps"] == legacy["optimizer_steps"]
    assert len(legacy["path"]) == len(coordinate["path"]) == len(variants["original"]["path"])
    for a, b, c in zip(legacy["path"], coordinate["path"], variants["original"]["path"]):
        assert torch.equal(a["theta"], b["theta"]) and torch.equal(a["theta"], c["theta"])
        assert a["loss"] == b["loss"] == c["loss"]
        assert torch.equal(a["candidate_probabilities"], b["candidate_probabilities"])
        assert list(a["physical_interval"]) == list(b["physical_interval"]) == list(c["physical_interval"])
    assert variants["projection"]["optimizer_steps"] == 0
    assert variants["projection"]["actual_nonzero_parameter_count"] == 0
    assert variants["projection"]["final_interval"] == case["Projection_interval"]
    for name, result in variants.items():
        assert isinstance(result.get("config"), dict), name
        assert isinstance(result.get("no_information"), bool), name
        assert len(result.get("intervals", ())) == 2, name
        assert len(result.get("per_offset_grid_kl", ())) == 2, name
        assert len(result.get("per_offset_candidate_kl", ())) == 2, name
        assert math.isfinite(float(result["loss"])) and math.isfinite(float(result["stable_loss"])), name
        assert list(result["path"][-1]["physical_interval"]) == list(result["final_interval"]), name

    stable = variants["stable"]
    trust = variants["offset_trust"]
    assert stable["path"][-1]["loss"] < stable["path"][0]["loss"]
    assert trust["path"][-1]["loss"] < trust["path"][0]["loss"]
    assert all(b["loss"] < a["loss"] for a, b in zip(stable["path"], stable["path"][1:]))
    assert all(b["loss"] < a["loss"] for a, b in zip(trust["path"], trust["path"][1:]))
    assert trust["diagnostics"]["grid_feature_matrix_materialized"] is False
    assert max(trust["diagnostics"]["per_offset_grid_kl_p0_to_ptheta"]) > 0
    assert trust["diagnostics"]["per_offset_candidate_targets"][0] != trust["diagnostics"]["per_offset_candidate_targets"][1]

    equal_rewards = [0.2] * len(candidates)
    empty_rewards = [None] * len(candidates)
    equal = {v: fit(case, hidden, candidates, equal_rewards, v) for v in ("stable", "offset_trust")}
    empty = {v: fit(case, hidden, candidates, empty_rewards, v) for v in ("stable", "offset_trust")}
    for group in (equal, empty):
        for result in group.values():
            assert result["optimizer_steps"] == 0
            assert result["actual_nonzero_parameter_count"] == 0
            assert all(torch.count_nonzero(row["theta"]) == 0 for row in result["path"])
    assert empty["offset_trust"]["path"][0]["candidate_probabilities"].numel() == 0

    # Finite-difference check of the fixed offset-trust objective gradient.
    pre = _prepare(case, hidden, candidates, rewards)
    cfg = _config("offset_trust", None)
    x = np.random.default_rng(9).normal(0, 1e-3, size=512)
    f, g, _ = _loss_gradient(pre, x, "offset_trust", cfg)
    checks = 0
    for i in (0, 17, 255, 256, 300, 511):
        d = np.zeros_like(x); d[i] = 1e-6
        fp = _loss_gradient(pre, x + d, "offset_trust", cfg)[0]
        fm = _loss_gradient(pre, x - d, "offset_trust", cfg)[0]
        fd = (fp - fm) / (2e-6)
        assert abs(fd - g[i]) < 2e-7, (i, fd, g[i])
        checks += 1
    # Stable gradient is checked independently too.
    cstable = _config("stable", None)
    _, gs, _ = _loss_gradient(pre, x, "stable", cstable)
    for i in (3, 128, 257, 500):
        d = np.zeros_like(x); d[i] = 1e-6
        fp = _loss_gradient(pre, x + d, "stable", cstable)[0]
        fm = _loss_gradient(pre, x - d, "stable", cstable)[0]
        assert abs((fp - fm) / 2e-6 - gs[i]) < 2e-7
        checks += 1

    # Deterministically force every Armijo trial to be rejected and verify
    # that the solver retains its zero initialization without a partial step.
    original_loss_gradient = _loss_gradient
    def rejecting_loss_gradient(_pre, theta_flat, _variant, _cfg):
        theta_flat = np.asarray(theta_flat, dtype=np.float64)
        if np.count_nonzero(theta_flat) == 0:
            fake_gradient = np.zeros_like(theta_flat)
            fake_gradient[0] = 1.0
            return 0.0, fake_gradient, {"candidate_kl": 0.0, "grid_kl": 0.0, "l2": 0.0}
        return 1.0, np.zeros_like(theta_flat), {"candidate_kl": 1.0, "grid_kl": 0.0, "l2": 0.0}
    try:
        _loss_gradient = rejecting_loss_gradient
        rejected = _lbfgs_armijo(pre, "offset_trust", cfg)
    finally:
        _loss_gradient = original_loss_gradient
    assert rejected["accepted"] == 0 and rejected["rejected"] == cfg["max_backtracks"] + 1
    assert rejected["termination"] == "armijo_rejected_no_further_update"
    assert len(rejected["path"]) == 1 and np.count_nonzero(rejected["x"]) == 0

    # A tied zero-score grid must retain first-maximum ordering from project.
    tie_case, tie_hidden, tie_candidates, tie_rewards = _synthetic_case(seed=2)
    tie_case["zero"] = {
        "logits": [torch.zeros((1, 4, 2), dtype=torch.float32) for _ in range(2)],
        "actions": [torch.zeros((1, 4, 1), dtype=torch.float32) for _ in range(2)],
    }
    from scripts.stvg_opd_p6_existing_common001 import config as method_config
    tie_ev = project(tie_case["zero"]["logits"], tie_case["zero"]["actions"], tie_case["records"], method_config("hc2"))
    tie_per = []
    for ex, rec in zip(tie_ev["offsets"], tie_case["records"]):
        ij = ex["ij"].numpy(); at = int(ex["score"].argmax())
        tie_per.append([rec["frame_ids"][int(ij[0, at])], rec["frame_ids"][int(ij[1, at])] + 1])
    tie_case["Projection_interval"] = [min(a[0] for a in tie_per), max(a[1] for a in tie_per)]
    tie_result = fit(tie_case, tie_hidden, tie_candidates, tie_rewards, "projection")
    assert tie_result["final_interval"] == tie_case["Projection_interval"]

    return {
        "schema": "stvg-evidence-temporal-cpu-contracts-v1",
        "status": "pass", "CPU_only": True, "synthetic_only": True,
        "GT_read": False, "GPU_used": False, "model_loaded": False,
        "contract_groups": 10, "gradient_finite_difference_checks": checks,
        "armijo_rejection_trials": int(rejected["rejected"]),
        "armijo_rejection_restores_zero_parameters": True,
        "variants": list(VARIANTS), "original_exact_legacy_delegation": True,
        "stable_monotone_armijo_steps": len(stable["path"]) - 1,
        "offset_trust_monotone_armijo_steps": len(trust["path"]) - 1,
        "zero_reward_noop_variants": ["stable", "offset_trust"],
        "empty_feedback_noop_variants": ["stable", "offset_trust"],
        "shared_parameter_count": 512, "legal_grid_feature_matrix_materialized": False,
        "fixed_objective_convexity_claim": "fixed targets and hidden features only",
        "teacher_ordering_correction_claim": False,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(contracts(), indent=2, sort_keys=True))
