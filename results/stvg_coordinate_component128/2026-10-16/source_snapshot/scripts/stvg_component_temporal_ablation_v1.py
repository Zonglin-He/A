"""Fixed, CPU temporal-component ablations over the existing 512-D fitter.

The upstream fitter is imported unchanged. Ablation configurations are
installed in memory only for the duration of one call and restored in a
``finally`` block. No model, media, labels, or optimizer retuning is involved.
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


VARIANTS = (
    "offset_trust",
    "no_grid_trust",
    "no_temporal_l2",
    "shared_offset_candidate",
    "projection_only",
)


def _shared_offset_loss_gradient(pre: Mapping[str, Any], theta_flat: np.ndarray,
                                 variant: str, cfg: Mapping[str, Any]):
    """Share the candidate KL structure without changing its total weight.

    The original objective adds one candidate KL for every offset. This
    ablation uses one shared-mean candidate KL, multiplied by the number of
    offsets, so it removes offset-specific candidate structure while keeping
    the aggregate candidate-term coefficient fixed. Full-grid KL and L2 are
    unchanged.
    """
    theta = np.asarray(theta_flat, dtype=np.float64).reshape(256, 2)
    total = 0.0
    grad = np.zeros((256, 2), dtype=np.float64)
    candidate_kl_unscaled = 0.0
    grid_kl_total = 0.0
    candidate_kl_multiplier = len(pre["offsets"])

    if len(pre["valid"]):
        base = pre["mean_base"]
        q, logq = basefit._prob(base + pre["rewards"] / cfg["tau_T"])
        if pre["no_information"]:
            q, logq = basefit._prob(base)
        logits = base + np.einsum("kdc,dc->k", pre["stable_features"], theta)
        p, logp = basefit._prob(logits)
        candidate_kl_unscaled = basefit._kl(q, logq, logp)
        candidate_grad = np.einsum("k,kdc->dc", p - q, pre["stable_features"])
        candidate_kl = candidate_kl_multiplier * candidate_kl_unscaled
        grad += candidate_kl_multiplier * candidate_grad
        total += candidate_kl
    else:
        candidate_kl = 0.0

    lambda_grid = float(cfg["lambda_grid"])
    for off in pre["offsets"]:
        grid0 = off["base"]
        grid = basefit._scores_for_offset(off, theta)
        p0, lp0 = basefit._prob(grid0)
        ptheta, lptheta = basefit._prob(grid)
        grid_kl = basefit._kl(p0, lp0, lptheta)
        grid_kl_total += grid_kl
        total += lambda_grid * grid_kl
        diff = lambda_grid * (ptheta - p0)
        starts = np.bincount(off["ij"][0], weights=diff, minlength=len(off["hidden"]))
        ends = np.bincount(off["ij"][1], weights=diff, minlength=len(off["hidden"]))
        grad[:, 0] += off["hidden"].T @ starts
        grad[:, 1] += off["hidden"].T @ ends

    l2 = float(cfg["l2"] * np.square(theta).sum())
    total += l2
    grad += 2.0 * float(cfg["l2"]) * theta
    return float(total), grad.reshape(-1), {
        "candidate_kl": float(candidate_kl),
        "candidate_kl_unscaled": float(candidate_kl_unscaled),
        "candidate_kl_multiplier": float(candidate_kl_multiplier),
        "grid_kl": float(grid_kl_total),
        "l2": float(l2),
    }


def _variant_config(variant_id: str) -> dict[str, Any]:
    cfg = dict(basefit.OFFSET_TRUST_CONFIG)
    if variant_id == "no_grid_trust":
        cfg["lambda_grid"] = 0.0
    elif variant_id == "no_temporal_l2":
        cfg["l2"] = 0.0
    return cfg


def fit(case: Mapping[str, Any], hidden: Sequence[Any],
        candidates: Sequence[Mapping[str, Any]], rewards: Sequence[float | None],
        variant_id: str) -> dict[str, Any]:
    """Fit one locked variant and return the upstream temporal-fit schema.

    ``offset_trust`` is the unchanged baseline. For the other variants the
    same source features, candidate rewards, initialization, optimizer,
    stopping rule, and decoder are used. Only the named objective term changes.
    """
    if variant_id not in VARIANTS:
        raise ValueError(f"variant_id must be one of {VARIANTS}")
    if variant_id == "projection_only":
        result = basefit.fit(case, hidden, candidates, rewards, "projection")
        result = dict(result)
        result["variant"] = variant_id
        result["component_ablation"] = {
            "schema": "stvg-temporal-component-ablation-v1",
            "variant_id": variant_id,
            "changed_terms": ["512-parameter temporal feedback disabled"],
            "GT_read": False,
        }
        return result
    if variant_id == "offset_trust":
        return basefit.fit(case, hidden, candidates, rewards, "offset_trust")

    actual_cfg = _variant_config(variant_id)
    original_config = basefit._config
    original_loss = basefit._loss_gradient

    def config_override(name: str, supplied: Mapping[str, Any] | None):
        if name == "offset_trust" and supplied is None:
            return dict(actual_cfg)
        return original_config(name, supplied)

    try:
        basefit._config = config_override
        if variant_id == "shared_offset_candidate":
            def shared_loss_override(pre, theta_flat, fit_variant, cfg):
                if fit_variant != "offset_trust":
                    return original_loss(pre, theta_flat, fit_variant, cfg)
                return _shared_offset_loss_gradient(pre, theta_flat, fit_variant, cfg)
            basefit._loss_gradient = shared_loss_override
        result = basefit.fit(case, hidden, candidates, rewards, "offset_trust")
    finally:
        basefit._config = original_config
        basefit._loss_gradient = original_loss

    result = dict(result)
    result["variant"] = variant_id
    result["config"] = dict(actual_cfg)
    result["component_ablation"] = {
        "schema": "stvg-temporal-component-ablation-v1",
        "variant_id": variant_id,
        "changed_terms": {
            "no_grid_trust": ["lambda_grid: 1.0 -> 0.0"],
            "no_temporal_l2": ["l2: 0.001 -> 0.0"],
            "shared_offset_candidate": [
                "candidate objective: per-offset KLs -> number_of_offsets * one shared-mean KL",
                "candidate KL total weight is preserved; only offset-specific structure is removed",
                "full-grid KL: unchanged, summed over both offsets",
                "L2: unchanged",
            ],
        }[variant_id],
        "active_parameters": 512,
        "same_initialization_and_optimizer": True,
        "GT_read": False,
    }
    result["component_ablation"]["objective_terms"] = {
        "lambda_grid": actual_cfg["lambda_grid"],
        "l2": actual_cfg["l2"],
        "candidate_kl_form": "shared_mean" if variant_id == "shared_offset_candidate" else "per_offset_sum",
        "candidate_kl_multiplier": len(result.get("per_offset_candidate_kl", ()))
        if variant_id == "shared_offset_candidate" else 1,
    }
    if variant_id == "shared_offset_candidate":
        # The legacy per-offset KL values remain diagnostics only. Snapshot
        # candidate_kl values and this explicit field are the optimized term.
        result["diagnostics"]["objective_candidate_kl_form"] = "shared_mean"
        result["diagnostics"]["objective_candidate_kl_multiplier"] = len(result.get("per_offset_candidate_kl", ()))
        result["diagnostics"]["per_offset_candidate_kl_role"] = "diagnostic_only_not_optimized_objective"
    else:
        result["diagnostics"]["objective_candidate_kl_form"] = "per_offset_sum"
    result["diagnostics"]["lambda_grid_objective_weight"] = actual_cfg["lambda_grid"]
    result["diagnostics"]["l2_objective_weight"] = actual_cfg["l2"]
    return result


def contracts() -> dict[str, Any]:
    """Synthetic CPU tests of objectives, gradients, state restoration and no-op flags."""
    case, hidden, candidates, rewards = basefit._synthetic_case(seed=137)
    before_config = basefit._config
    before_loss = basefit._loss_gradient
    variants = {name: fit(case, hidden, candidates, rewards, name) for name in VARIANTS}
    assert basefit._config is before_config and basefit._loss_gradient is before_loss

    baseline = basefit.fit(case, hidden, candidates, rewards, "offset_trust")
    assert variants["offset_trust"]["final_interval"] == baseline["final_interval"]
    assert variants["offset_trust"]["path"][-1]["loss"] == baseline["path"][-1]["loss"]
    assert np.array_equal(variants["no_grid_trust"]["config"]["lambda_grid"], 0.0)
    assert variants["no_grid_trust"]["config"]["l2"] == basefit.OFFSET_TRUST_CONFIG["l2"]
    assert variants["no_temporal_l2"]["config"]["l2"] == 0.0
    assert variants["no_temporal_l2"]["config"]["lambda_grid"] == basefit.OFFSET_TRUST_CONFIG["lambda_grid"]
    assert variants["projection_only"]["optimizer_steps"] == 0
    assert variants["projection_only"]["actual_nonzero_parameter_count"] == 0
    assert variants["projection_only"]["final_interval"] == case["Projection_interval"]

    pre = basefit._prepare(case, hidden, candidates, rewards)
    checks = 0
    for name in ("no_grid_trust", "no_temporal_l2", "shared_offset_candidate"):
        cfg = _variant_config(name)
        fn = _shared_offset_loss_gradient if name == "shared_offset_candidate" else basefit._loss_gradient
        x = np.random.default_rng(19).normal(0.0, 1e-3, size=512)
        _, grad, terms = fn(pre, x, "offset_trust", cfg)
        assert all(math.isfinite(float(v)) for v in terms.values())
        if name == "shared_offset_candidate":
            original_terms = basefit._loss_gradient(pre, x, "offset_trust", cfg)[2]
            assert terms["candidate_kl_multiplier"] == len(pre["offsets"])
            assert math.isclose(
                terms["candidate_kl"],
                len(pre["offsets"]) * terms["candidate_kl_unscaled"],
                rel_tol=1e-13, abs_tol=1e-15,
            )
            assert terms["grid_kl"] == original_terms["grid_kl"]
            assert terms["l2"] == original_terms["l2"]
        for i in (0, 255, 256, 511):
            delta = np.zeros_like(x); delta[i] = 1e-6
            plus = fn(pre, x + delta, "offset_trust", cfg)[0]
            minus = fn(pre, x - delta, "offset_trust", cfg)[0]
            fd = (plus - minus) / 2e-6
            assert abs(fd - grad[i]) < 3e-7, (name, i, fd, grad[i])
            checks += 1

    equal = [0.25] * len(candidates)
    missing = [None] * len(candidates)
    for name in VARIANTS[:-1]:
        for reward_rows in (equal, missing):
            result = fit(case, hidden, candidates, reward_rows, name)
            assert result["optimizer_steps"] == 0, (name, reward_rows)
            assert result["actual_nonzero_parameter_count"] == 0, (name, reward_rows)

    return {
        "schema": "stvg-temporal-component-ablation-cpu-contracts-v1",
        "status": "pass",
        "CPU_only": True,
        "synthetic_only": True,
        "GT_read": False,
        "GPU_used": False,
        "model_loaded": False,
        "variants": list(VARIANTS),
        "finite_difference_checks": checks,
        "shared_candidate_kl_weight_preserved": True,
        "shared_grid_kl_and_l2_unchanged": True,
        "patched_upstream_globals_restored": True,
        "projection_zero_parameter_noop": True,
        "equal_or_missing_rewards_no_update": True,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(contracts(), indent=2, sort_keys=True))
