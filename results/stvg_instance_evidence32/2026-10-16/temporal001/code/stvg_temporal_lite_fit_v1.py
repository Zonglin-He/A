"""Combined Temporal-Lite objective over the locked 512-parameter fitter.

The upstream fitter remains unchanged. This module changes only the candidate
KL structure and disables its parameter L2 term through temporary in-memory
overrides, restored unconditionally after each fit. Full-grid trust and the
original deterministic L-BFGS/Armijo implementation remain in force.
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stvg_evidence_temporal_fit_v1 as basefit


VARIANTS = ("offset_trust", "temporal_lite")
LITE_VARIANT = "temporal_lite"


def _lite_config() -> dict[str, Any]:
    """Copy every locked offset-trust setting, changing only L2 to zero."""
    cfg = dict(basefit.OFFSET_TRUST_CONFIG)
    cfg["l2"] = 0.0
    return cfg


def _exact_equal(a: Any, b: Any) -> bool:
    """Strict recursive equality for synthetic baseline delegation checks."""
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b)
                and all(_exact_equal(a[key], b[key]) for key in a))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(_exact_equal(x, y) for x, y in zip(a, b)))
    if hasattr(a, "detach") or hasattr(b, "detach"):
        if not (hasattr(a, "detach") and hasattr(b, "detach")):
            return False
        aa, bb = a.detach().cpu(), b.detach().cpu()
        return aa.dtype == bb.dtype and tuple(aa.shape) == tuple(bb.shape) and bool((aa == bb).all())
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.dtype == b.dtype and a.shape == b.shape
                and np.array_equal(a, b, equal_nan=True))
    if isinstance(a, float) or isinstance(b, float):
        return type(a) is type(b) and (a == b or (math.isnan(a) and math.isnan(b)))
    return type(a) is type(b) and a == b


def _shared_candidate_loss_gradient(pre: Mapping[str, Any], theta_flat: np.ndarray,
                                    variant: str,
                                    cfg: Mapping[str, Any]) -> tuple[float, np.ndarray, dict[str, float]]:
    """Shared-mean candidate KL × O, plus the original full-grid trust sum."""
    if variant != "offset_trust":
        return basefit._loss_gradient(pre, theta_flat, variant, cfg)

    theta = np.asarray(theta_flat, dtype=np.float64).reshape(256, 2)
    total = 0.0
    grad = np.zeros((256, 2), dtype=np.float64)
    candidate_kl_unscaled = 0.0
    grid_kl_total = 0.0
    candidate_kl_multiplier = len(pre["offsets"])

    if len(pre["valid"]):
        mean_base = pre["mean_base"]
        q, logq = basefit._prob(mean_base + pre["rewards"] / float(cfg["tau_T"]))
        if pre["no_information"]:
            q, logq = basefit._prob(mean_base)
        logits = mean_base + np.einsum("kdc,dc->k", pre["stable_features"], theta)
        p, logp = basefit._prob(logits)
        candidate_kl_unscaled = basefit._kl(q, logq, logp)
        candidate_grad = np.einsum("k,kdc->dc", p - q, pre["stable_features"])
        candidate_kl = candidate_kl_multiplier * candidate_kl_unscaled
        grad += candidate_kl_multiplier * candidate_grad
        total += candidate_kl
    else:
        candidate_kl = 0.0

    # Preserve the original lambda_grid and sum one KL over every full legal
    # decoder grid. The only config change is l2=0.
    lambda_grid = float(cfg["lambda_grid"])
    for off in pre["offsets"]:
        p0, lp0 = basefit._prob(off["base"])
        ptheta, lptheta = basefit._prob(basefit._scores_for_offset(off, theta))
        grid_kl = basefit._kl(p0, lp0, lptheta)
        grid_kl_total += grid_kl
        total += lambda_grid * grid_kl
        diff = lambda_grid * (ptheta - p0)
        starts = np.bincount(off["ij"][0], weights=diff, minlength=len(off["hidden"]))
        ends = np.bincount(off["ij"][1], weights=diff, minlength=len(off["hidden"]))
        grad[:, 0] += off["hidden"].T @ starts
        grad[:, 1] += off["hidden"].T @ ends

    # L2 is explicitly zero and asserted here to catch accidental config drift.
    l2_weight = float(cfg["l2"])
    if l2_weight != 0.0:
        raise ValueError("Temporal-Lite requires l2=0")
    l2 = float(l2_weight * np.square(theta).sum())
    total += l2
    grad += 2.0 * l2_weight * theta
    return float(total), grad.reshape(-1), {
        "candidate_kl": float(candidate_kl),
        "candidate_kl_unscaled": float(candidate_kl_unscaled),
        "candidate_kl_multiplier": float(candidate_kl_multiplier),
        "grid_kl": float(grid_kl_total),
        "l2": float(l2),
    }


def fit(case: Mapping[str, Any], hidden: Sequence[Any],
        candidates: Sequence[Mapping[str, Any]], rewards: Sequence[float | None],
        variant_id: str) -> dict[str, Any]:
    """Run baseline Offset-Trust or Combined Temporal-Lite.

    ``offset_trust`` delegates directly to the untouched upstream fitter.
    ``temporal_lite`` keeps its 512 parameters, initialization, L-BFGS memory,
    stopping rule, backtracking and decoder; it changes only shared candidate
    supervision × number of offsets and removes the parameter L2 term.
    """
    if variant_id not in VARIANTS:
        raise ValueError(f"variant_id must be one of {VARIANTS}")
    if variant_id == "offset_trust":
        return basefit.fit(case, hidden, candidates, rewards, "offset_trust")

    actual_cfg = _lite_config()
    original_config = basefit._config
    original_loss_gradient = basefit._loss_gradient

    def config_override(name: str, supplied: Mapping[str, Any] | None):
        if name == "offset_trust" and supplied is None:
            return dict(actual_cfg)
        return original_config(name, supplied)

    def loss_override(pre, theta_flat, fit_variant, cfg):
        if fit_variant == "offset_trust":
            return _shared_candidate_loss_gradient(pre, theta_flat, fit_variant, cfg)
        return original_loss_gradient(pre, theta_flat, fit_variant, cfg)

    try:
        basefit._config = config_override
        basefit._loss_gradient = loss_override
        result = basefit.fit(case, hidden, candidates, rewards, "offset_trust")
    finally:
        basefit._config = original_config
        basefit._loss_gradient = original_loss_gradient

    result = dict(result)
    result["variant"] = LITE_VARIANT
    result["config"] = dict(actual_cfg)
    result["temporal_lite"] = {
        "schema": "stvg-temporal-lite-fit-v1",
        "variant_id": LITE_VARIANT,
        "changed_terms": [
            "per-offset candidate KL sum -> number_of_offsets * one shared-mean candidate KL",
            "parameter L2 weight -> 0",
        ],
        "candidate_kl_form": "shared_mean",
        "candidate_kl_multiplier": len(result.get("per_offset_candidate_kl", ())),
        "candidate_weight_preserved_by_offset_count": True,
        "lambda_grid": basefit.OFFSET_TRUST_CONFIG["lambda_grid"],
        "full_grid_trust_preserved": True,
        "active_parameters": 512,
        "same_initialization_optimizer_line_search_and_decoder": True,
        "GT_read": False,
    }
    result["diagnostics"] = dict(result.get("diagnostics", {}))
    result["diagnostics"].update({
        "objective_candidate_kl": float(result["path"][-1]["candidate_kl"]),
        "objective_candidate_kl_form": "shared_mean",
        "objective_candidate_kl_multiplier": len(result.get("per_offset_candidate_kl", ())),
        "per_offset_candidate_kl_role": "diagnostic_only_not_optimized_objective",
        "objective_grid_kl": float(result["path"][-1]["grid_kl"]),
        "lambda_grid_objective_weight": float(actual_cfg["lambda_grid"]),
        "l2_objective_weight": 0.0,
        "l2_penalty": 0.0,
    })
    return result


def no_fit_fallback_intervals(candidates: Sequence[Any],
                              selected_interval: Sequence[Any],
                              no_fit_record: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Preserve an independently validated original no-fit interval exactly."""
    # Keep the candidate list in the API so callers cannot mistake a missing
    # fit for a zero-reward fit. The validated no-fit record controls fallback
    # even when candidates exist but lack the required role/count structure.
    _ = candidates
    selected = list(selected_interval)
    if (no_fit_record.get("status") != "unavailable_temporal_input_native_fallback"
            or no_fit_record.get("reason") != "candidate_pool_or_role_count_not_four"
            or no_fit_record.get("GT_read") is not False
            or list(no_fit_record.get("selected_interval", [])) != selected
            or int(no_fit_record.get("optimizer_steps", -1)) != 0):
        raise ValueError("Temporal-Lite no-fit fallback must be the validated original native fallback")
    return {name: copy.deepcopy(selected) for name in VARIANTS}


def contracts() -> dict[str, Any]:
    """Synthetic-only CPU contracts; no cached source cases or labels loaded."""
    case, hidden, candidates, rewards = basefit._synthetic_case(seed=42016)
    original_config = basefit._config
    original_loss = basefit._loss_gradient
    original_offset_cfg = basefit.OFFSET_TRUST_CONFIG
    base_cfg = dict(basefit.OFFSET_TRUST_CONFIG)
    lite_cfg = _lite_config()
    assert set(lite_cfg) == set(base_cfg)
    assert {k: v for k, v in lite_cfg.items() if k != "l2"} == {
        k: v for k, v in base_cfg.items() if k != "l2"
    }
    assert lite_cfg["l2"] == 0.0
    assert lite_cfg["lambda_grid"] == base_cfg["lambda_grid"]
    assert lite_cfg["max_iter"] == base_cfg["max_iter"]
    assert lite_cfg["memory"] == base_cfg["memory"]
    assert lite_cfg["gradient_tolerance"] == base_cfg["gradient_tolerance"]
    assert lite_cfg["armijo_c1"] == base_cfg["armijo_c1"]
    assert lite_cfg["initial_step"] == base_cfg["initial_step"]
    assert lite_cfg["backtrack_factor"] == base_cfg["backtrack_factor"]
    assert lite_cfg["max_backtracks"] == base_cfg["max_backtracks"]

    baseline = fit(case, hidden, candidates, rewards, "offset_trust")
    lite = fit(case, hidden, candidates, rewards, LITE_VARIANT)
    assert basefit._config is original_config
    assert basefit._loss_gradient is original_loss
    assert basefit.OFFSET_TRUST_CONFIG is original_offset_cfg
    assert _exact_equal(baseline, basefit.fit(case, hidden, candidates, rewards, "offset_trust"))

    pre = basefit._prepare(case, hidden, candidates, rewards)
    rng = np.random.default_rng(42016)
    theta = rng.normal(0.0, 1e-3, size=512)
    objective, gradient, terms = _shared_candidate_loss_gradient(pre, theta, "offset_trust", lite_cfg)
    assert math.isfinite(objective) and np.isfinite(gradient).all()
    assert terms["candidate_kl_multiplier"] == len(pre["offsets"])
    assert math.isclose(terms["candidate_kl"],
                        len(pre["offsets"]) * terms["candidate_kl_unscaled"],
                        rel_tol=1e-13, abs_tol=1e-15)
    original_terms = basefit._loss_gradient(pre, theta, "offset_trust", base_cfg)[2]
    assert terms["grid_kl"] == original_terms["grid_kl"]
    assert terms["l2"] == 0.0
    assert terms["grid_kl"] > 0.0
    # Check every one of the 512 analytic gradient coordinates centrally.
    epsilon = 1e-6
    max_fd_error = 0.0
    for index in range(512):
        delta = np.zeros(512, dtype=np.float64)
        delta[index] = epsilon
        plus = _shared_candidate_loss_gradient(pre, theta + delta, "offset_trust", lite_cfg)[0]
        minus = _shared_candidate_loss_gradient(pre, theta - delta, "offset_trust", lite_cfg)[0]
        finite_difference = (plus - minus) / (2.0 * epsilon)
        error = abs(finite_difference - gradient[index])
        max_fd_error = max(max_fd_error, float(error))
        assert error < 4e-7, (index, finite_difference, gradient[index], error)

    # Equal rewards and an all-missing reward vector follow the original
    # no-information exact-zero-update behavior, including the decoded interval.
    equal = [0.25] * len(candidates)
    missing = [None] * len(candidates)
    no_information_checks = 0
    for reward_rows in (equal, missing):
        for variant in VARIANTS:
            result = fit(case, hidden, candidates, reward_rows, variant)
            assert result["optimizer_steps"] == 0
            assert result["actual_nonzero_parameter_count"] == 0
            assert result["final_interval"] == case["Projection_interval"]
            no_information_checks += 1

    fallback_record = {
        "status": "unavailable_temporal_input_native_fallback",
        "reason": "candidate_pool_or_role_count_not_four",
        "selected_interval": [31, 89],
        "optimizer_steps": 0,
        "GT_read": False,
    }
    fallback = no_fit_fallback_intervals([], [31, 89], fallback_record)
    assert fallback == {name: [31, 89] for name in VARIANTS}
    try:
        no_fit_fallback_intervals([], [31, 89], dict(fallback_record, selected_interval=[30, 89]))
    except ValueError:
        pass
    else:
        raise AssertionError("no-fit fallback must reject an interval change")

    return {
        "schema": "stvg-temporal-lite-cpu-contracts-v1",
        "status": "pass",
        "CPU_only": True,
        "synthetic_only": True,
        "GT_read": False,
        "GPU_used": False,
        "model_loaded": False,
        "variants": list(VARIANTS),
        "finite_difference_coordinates_checked": 512,
        "finite_difference_max_abs_error": max_fd_error,
        "shared_candidate_kl_multiplier_equals_offset_count": True,
        "original_lambda_grid_and_grid_kl_preserved": True,
        "all_other_optimizer_config_unchanged": True,
        "only_l2_disabled": True,
        "no_information_exact_zero_update_cases": no_information_checks,
        "empty_candidate_original_fallback_preserved": True,
        "upstream_temporal_module_unchanged_and_overrides_restored": True,
        "baseline_delegation_exact": True,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(contracts(), indent=2, sort_keys=True))
