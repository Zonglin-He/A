"""Independent CPU arithmetic audit for saved Spatial-Lite fitter witnesses.

This module does not import or call the Spatial-Lite fitter. It reconstructs
the antithetic rollout, geometry rewards, Lite likelihood loss and mean-space
gradient from saved inputs, then checks Adam, solver, acceptance, rollback,
state chaining and actual decoder-call accounting.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np


_RECIPE_SCALES = {
    "lite_backtrack": [1.0, 0.5, 0.25, 0.125],
    "lite_fullstep": [1.0],
}
_TOL = 2e-6


def _arr(value):
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)


def _err(left, right):
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _tree_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _iou(pred, evidence):
    p = np.asarray(pred, dtype=np.float64)
    e = np.asarray(evidence, dtype=np.float64)
    pxy = np.concatenate((p[..., :2] - p[..., 2:] / 2, p[..., :2] + p[..., 2:] / 2), axis=-1)
    exy = np.concatenate((e[..., :2] - e[..., 2:] / 2, e[..., :2] + e[..., 2:] / 2), axis=-1)
    inter = np.maximum(0, np.minimum(pxy[..., 2:], exy[..., 2:])
                       - np.maximum(pxy[..., :2], exy[..., :2])).prod(-1)
    pa = np.maximum(0, pxy[..., 2:] - pxy[..., :2]).prod(-1)
    ea = np.maximum(0, exy[..., 2:] - exy[..., :2]).prod(-1)
    return inter / np.maximum(pa + ea - inter, 1e-7)


def _assert_absent_or_none(mapping, name):
    _require(name not in mapping or mapping[name] is None,
             f"Spatial-Lite must not compute or save {name}")


def _rollout_and_loss(row, evidence, cfg, positions, key, step):
    import torch

    rollout = row["rollout"]
    before = _arr(row["boxes_before"]).astype(np.float64)
    mean = _arr(rollout["mean"]).astype(np.float64)
    samples = _arr(rollout["samples"]).astype(np.float64)
    rewards = _arr(rollout["rewards"]).astype(np.float64)
    used = _arr(rollout["used_rewards"]).astype(np.float64)
    q = _arr(rollout["weights"]).astype(np.float64)
    pos = list(positions)
    _require(rollout["permutations"] is None, "Spatial-Lite rollout must remain unshuffled")
    _require(mean.shape == (len(pos), 4) and samples.shape == (len(pos), 32, 4),
             "rollout tensor shapes differ from active anchor count")
    seed = int(hashlib.sha256(f"spatial-opd-v1|{key}|{step}".encode()).hexdigest()[:15], 16)
    _require(int(rollout["seed"]) == seed, "rollout seed mismatch")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(len(pos), 16, 4, generator=generator, dtype=torch.float32)
    noise = torch.cat((noise, -noise), dim=1).numpy().astype(np.float64)
    boxes = before[pos]
    expected_mean = np.log(boxes / (1.0 - boxes))
    expected_samples = expected_mean[:, None, :] + float(cfg["sigma"]) * noise
    sample_error = _err(samples, expected_samples)
    anti_error = _err(samples[:, :16] + samples[:, 16:],
                      np.broadcast_to(2 * mean[:, None, :], samples[:, :16].shape))
    mean_error = _err(mean, expected_mean)
    _require(sample_error <= 2e-6 and anti_error <= 2e-6 and mean_error <= 2e-6,
             f"rollout mean or antithetic samples differ from locked reconstruction: "
             f"sample={sample_error}, antithetic={anti_error}, mean={mean_error}")

    sigmoid_samples = 1.0 / (1.0 + np.exp(-samples))
    expected_rewards = _iou(sigmoid_samples, evidence[:, None, :])
    reward_error = _err(rewards, expected_rewards)
    _require(reward_error <= 3e-6 and _err(used, rewards) <= 2e-6,
             "saved geometry rewards differ from independent IoU reconstruction")
    logits = used / float(cfg["tau"])
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    expected_q = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    q_error = _err(q, expected_q)
    _require(q_error <= 2e-6 and _err(q.sum(-1), np.ones(len(pos))) <= 2e-6,
             "saved reward softmax differs from independent reconstruction")

    logp = -np.square(samples - mean[:, None, :]).sum(-1) / (2 * float(cfg["sigma"]) ** 2)
    logp -= 2 * math.log(2 * math.pi * float(cfg["sigma"]) ** 2)
    expected_per_anchor = -(q * logp).sum(-1)
    expected_loss = float(expected_per_anchor.mean())
    loss_error = max(_err(row["per_anchor_loss"], expected_per_anchor),
                     abs(float(row["loss"]) - expected_loss))
    _require(loss_error <= 3e-6, "Spatial-Lite likelihood loss differs from saved rollout")

    # At the rollout mean the old policy KL and antithetic uniform score
    # baseline have zero mean-gradient. The remaining score gradient is exact.
    expected_mean_gradient = -(
        q[:, :, None] * (samples - mean[:, None, :])
    ).sum(axis=1) / (float(cfg["sigma"]) ** 2 * len(pos))
    info = bool(np.any(used != used[:, :1]))
    if info:
        mean_gradient_error = _err(row["mean_gradient"], expected_mean_gradient)
        _require(mean_gradient_error <= 8e-5,
                 "saved mean gradient differs from the closed-form detached-score gradient")
    else:
        # Production deliberately skips backward for exactly flat rewards.
        # None is required, not treated as an observed zero gradient. The
        # caller separately checks zero proposals and unchanged state/Adam.
        _require(row["mean_gradient"] is None,
                 "flat-skip must not fabricate a computed mean gradient")
        mean_gradient_error = 0.0
    _require(bool(rollout["informative"]) == info, "informative flag differs from saved rewards")
    _require(_err(row["reference_iou_before"], _iou(boxes, evidence)) <= 2e-6,
             "pre-update pseudo-reference IoU differs from saved boxes")
    _require(_err(row["anchor_weights"], np.ones(len(pos))) == 0.0,
             "Spatial-Lite anchors must have equal positive-anchor weights")
    return {
        "mean": mean,
        "samples": samples,
        "weights": q,
        "evidence": evidence,
        "expected_loss": expected_loss,
        "informative": info,
        "mean_gradient_error": mean_gradient_error,
        "loss_error": loss_error,
        "reward_error": reward_error,
        "q_error": q_error,
    }


def _adam_audit(row, names):
    gradient = _arr(row["gradient"]).astype(np.float64).reshape(-1)
    delta = _arr(row["original_adam_delta"]).astype(np.float64).reshape(-1)
    before = row["optimizer_before"]
    proposed = row["optimizer_proposed"]
    group = proposed["param_groups"][0]
    beta1, beta2 = group["betas"]
    lr = float(group["lr"])
    epsilon = float(group["eps"])
    parameter_ids = group["params"]
    _require(len(parameter_ids) == len(names), "Adam parameter order differs from saved state order")
    moment_error = variance_error = delta_error = 0.0
    offset = 0
    for parameter_id, name in zip(parameter_ids, names):
        width = _arr(row["state_before"][name]).size
        grad = gradient[offset:offset + width]
        old = before["state"].get(parameter_id)
        new = proposed["state"][parameter_id]
        old_mean = np.zeros(width, dtype=np.float32) if old is None else _arr(old["exp_avg"]).astype(np.float32).reshape(-1)
        old_var = np.zeros(width, dtype=np.float32) if old is None else _arr(old["exp_avg_sq"]).astype(np.float32).reshape(-1)
        mean = (beta1 * old_mean + (1 - beta1) * grad).astype(np.float32)
        variance = (beta2 * old_var + (1 - beta2) * grad * grad).astype(np.float32)
        moment_error = max(moment_error, _err(mean, new["exp_avg"]))
        variance_error = max(variance_error, _err(variance, new["exp_avg_sq"]))
        old_step = 0.0 if old is None else float(_arr(old["step"]))
        step_count = float(_arr(new["step"]))
        _require(step_count == old_step + 1, "Adam step counter did not advance once")
        expected_delta = (
            -lr / (1 - beta1 ** step_count)
            * mean.astype(np.float64)
            / (np.sqrt(variance.astype(np.float64) / (1 - beta2 ** step_count)) + epsilon)
        ).astype(np.float32)
        delta_error = max(delta_error, _err(expected_delta, delta[offset:offset + width]))
        offset += width
    _require(offset == len(gradient) == len(delta), "Adam flattened vector has the wrong size")
    _require(max(moment_error, variance_error, delta_error) <= 5e-6,
             "saved Adam moments or proposal differ from independent update")
    return {"moment_error": moment_error, "variance_error": variance_error,
            "adam_delta_error": delta_error}


def _solver_audit(row, protection, mode):
    raw = _arr(row["original_adam_delta"]).astype(np.float64).reshape(-1)
    direction = _arr(row["filtered_delta"]).astype(np.float64).reshape(-1)
    solver = row["solver"]
    if mode == "M1":
        _require(solver is None, "M1 must not run the decoder-response solver")
        _require(_err(raw, direction) == 0.0, "M1 direction must equal raw Adam direction")
        return {"solver": "not_applicable"}
    _require(isinstance(solver, dict), "M2 proposal must save its solver or identity reason")
    if solver.get("status") == "identity":
        _require(solver.get("reason") in {"empty_protection_support", "zero_adam_direction"},
                 "identity solver reason is missing or invalid")
        _require(_err(raw, direction) == 0.0, "identity response direction differs from Adam")
        if solver["reason"] == "empty_protection_support":
            _require(not protection, "empty-support identity has nonempty U")
        else:
            _require(np.linalg.norm(raw) == 0.0, "zero-direction identity has nonzero Adam step")
        _assert_absent_or_none(solver, "lambda")
        _assert_absent_or_none(solver, "response_cost_adam")
        return {"solver": "identity"}

    status = solver.get("status")
    _require(status in {"converged", "max_iter", "near_zero_response_identity"},
             "unknown response solver status")
    _require(protection, "response solver was run with empty U")
    _require(_err(solver["solution"], direction) <= 2e-6,
             "saved filtered direction differs from solver solution")
    jd = _arr(solver["Jd"]).astype(np.float64).reshape(-1)
    jx = _arr(solver["Jfiltered"]).astype(np.float64).reshape(-1)
    expected_width = 4 * len(protection)
    _require(len(jd) == len(jx) == expected_width, "solver output width differs from protection support")
    _require(_err(solver["cg_rhs"], raw) <= 2e-6, "solver right-hand side differs from Adam")
    _require(abs(float(solver["response_cost_adam"]) - float(jd @ jd)) <= 2e-5,
             "saved Adam response cost differs from Jd")
    _require(abs(float(solver["response_cost_filtered"]) - float(jx @ jx)) <= 2e-5,
             "saved filtered response cost differs from Jfiltered")
    if status == "near_zero_response_identity":
        _require(_err(raw, direction) <= 2e-6, "near-zero solver should return identity direction")
        _require(float(solver["lambda"]) == 0.0, "near-zero solver lambda should be zero")
    else:
        expected_lambda = float(solver["mu"]) * float(raw @ raw) / max(float(jd @ jd), 1e-300)
        _require(abs(float(solver["lambda"]) - expected_lambda)
                 <= 2e-4 * max(1.0, abs(expected_lambda)), "normalized solver lambda mismatch")
        residual = _arr(solver["cg_residual"]).astype(np.float64).reshape(-1)
        normal_applied = _arr(solver["normal_applied"]).astype(np.float64).reshape(-1)
        _require(_err(raw - normal_applied, residual) <= 3e-6,
                 "saved conjugate-gradient residual is inconsistent")
        _require(abs(float(solver["solver_residual_abs"]) - float(np.linalg.norm(residual))) <= 3e-6,
                 "saved solver residual norm mismatch")
    return {"solver": status}


def _qualification_audit(row):
    q = row.get("qualification_diagnostics")
    if q is None:
        return None
    left = float(q["adjoint_left"])
    right = float(q["adjoint_right"])
    error = abs(left - right)
    _require(abs(error - float(q["adjoint_abs_error"])) <= 1e-7,
             "qualification adjoint error mismatch")
    bound = 1e-7 + 1e-4 * max(abs(left), abs(right))
    _require(bool(q["adjoint_pass"]) == (error <= bound), "qualification adjoint pass bit mismatch")
    for fd in q["finite_differences"]:
        observed = (_arr(fd["plus"]).astype(np.float64) - _arr(fd["minus"]).astype(np.float64)) / (2 * float(fd["h"]))
        predicted = _arr(q["linear_jvp"]).astype(np.float64)
        error = float(np.linalg.norm(observed - predicted))
        norm = float(np.linalg.norm(predicted))
        _require(abs(error - float(fd["absolute_error"])) <= 2e-6,
                 "qualification finite-difference error mismatch")
        _require(abs(norm - float(fd["predicted_norm"])) <= 2e-6,
                 "qualification predicted norm mismatch")
        _require(bool(fd["gross_agreement"]) == (error <= 1e-3 + 0.25 * norm),
                 "qualification finite-difference pass bit mismatch")
    return {"adjoint_error": error}


def fit_audit(f, ref, recipe, response_positions=None):
    """Audit a saved Lite fit without importing or invoking its fitter.

    ``response_positions`` is the independently supplied event-output support
    from the final predicted interval. For full-grid fits it must be omitted.
    """
    _require(isinstance(f, dict) and f.get("schema") == "stvg-spatial-lite-fit-v1",
             "not a saved Spatial-Lite fit")
    _require(recipe in _RECIPE_SCALES and f.get("recipe") == recipe,
             "recipe differs from saved fit")
    _require(f.get("GT_read") is False and f.get("active_parameters") == 1792,
             "fit identity or joint parameter scope is invalid")
    _require(f.get("mode") in {"M1", "M2"}, "invalid mode")
    _assert_absent_or_none(f, "original_proposal_boxes")
    _assert_absent_or_none(f, "response_original_RMS")
    _assert_absent_or_none(f, "response_budget")
    _assert_absent_or_none(f, "norm_control")

    anchors = ref.get("anchors")
    _require(isinstance(anchors, (list, tuple)), "reference must provide anchors")
    all_positions = [int(anchor["position"]) for anchor in anchors]
    active = [anchor for anchor in anchors if float(anchor["weight"]) > 0]
    positions = [int(anchor["position"]) for anchor in active]
    evidence = np.asarray([anchor.get("box_cxcywh", anchor.get("box")) for anchor in active], dtype=np.float64)
    _require(f.get("observation_positions") == positions, "positive-weight anchor order differs")
    _require(f.get("anchor_positions") == sorted(all_positions), "all supplied anchor positions differ")
    _require(f.get("anchor_weights") == [1.0] * len(positions), "Lite anchors are not equally weighted")
    _require(len(set(all_positions)) == len(all_positions), "duplicate reference anchor position")
    frame_count = int(f["ids_count"])
    _require(frame_count > 0 and all(0 <= pos < frame_count for pos in all_positions),
             "anchor lies outside the saved frame-id grid")

    if f["response_scope"] == "full_grid":
        _require(response_positions is None and f.get("event_support_positions") is None,
                 "full-grid fit must not declare event support")
        support = list(range(frame_count))
    elif f["response_scope"] == "event_support":
        saved_support = f.get("event_support_positions")
        support = list(saved_support or []) if response_positions is None else list(response_positions)
        _require(saved_support == support, "event support differs from independently supplied positions")
        _require(support == sorted(set(support)), "event support must be sorted and unique")
        _require(all(isinstance(pos, int) and 0 <= pos < frame_count for pos in support),
                 "event support position is outside the supplied grid")
    else:
        raise AssertionError("unknown response scope")
    protection = [pos for pos in support if pos not in set(positions)]
    _require(f.get("response_positions") == support, "saved response positions differ from selected scope")
    _require(f.get("protection_positions") == protection, "saved U differs from response support minus anchors")

    cfg = f["config"]
    settings = f["settings"]
    path = f["path"]
    _require(len(path) == int(cfg["steps"]), "path length differs from configured step count")
    state_names = list(f["initial_state"])
    _require(f.get("active_names") == state_names, "active Query+LN name list differs from saved state")
    _require(sum(_arr(value).size for value in f["initial_state"].values()) == 1792,
                 "saved source state is not 1792 parameters")
    if not f.get("qualification_requested"):
        _require(f.get("qualification_diagnostics") is None,
                 "disabled qualification must be saved as None, not fabricated readings")
        _require(all(row.get("qualification_diagnostics") is None for row in path),
                 "disabled row qualification must be None")

    previous_state = f["initial_state"]
    previous_optimizer = None
    previous_boxes = None
    stopped = False
    accepted = proposals = skipped = rejected = solver_calls = 0
    decode_sums = {key: 0 for key in f["decode_counts"]}
    decoder_seconds_sums = {key: 0.0 for key in f["decoder_seconds_by_purpose"]}
    solver_seconds = 0.0
    max_loss_error = max_gradient_error = max_reward_error = max_q_error = 0.0
    adam_rows = []
    solver_rows = []

    for step, row in enumerate(path):
        _require(row["step"] == step and _tree_equal(row["state_before"], previous_state),
                 "state-before chain mismatch")
        if previous_optimizer is not None:
            _require(_tree_equal(row["optimizer_before"], previous_optimizer), "optimizer-before chain mismatch")
        _require(len(row["decoder_counts"]) == len(decode_sums), "row decoder counter schema mismatch")
        for name, count in row["decoder_counts"].items():
            _require(isinstance(count, int) and count >= 0, "decoder count must be a nonnegative integer")
            decode_sums[name] += count
        for name, seconds in row["decoder_seconds"].items():
            _require(math.isfinite(float(seconds)) and float(seconds) >= 0,
                     "decoder timing must be a finite nonnegative measurement")
            decoder_seconds_sums[name] = decoder_seconds_sums.get(name, 0.0) + float(seconds)
        _assert_absent_or_none(row, "original_proposal_boxes")
        _assert_absent_or_none(row, "response_original_RMS")
        _assert_absent_or_none(row, "response_budget")
        _assert_absent_or_none(row, "norm_control")
        _qualification_audit(row)

        if stopped:
            _require(row.get("skipped") and row.get("reason") == "stopped_after_rejected_step",
                     "steps after a rejection must be skipped for this query")
            _require(row["boxes_before"] is None and row["boxes_after"] is None,
                     "stopped steps must not fabricate decoder boxes")
            _require(row["decoder_counts"]["round_before"] == 0
                     and row["decoder_counts"]["candidate_trial"] == 0,
                     "stopped step performed an unexpected decoder call")
            _require(_tree_equal(row["state_after"], row["state_before"])
                     and _tree_equal(row["optimizer_after"], row["optimizer_before"]),
                     "stopped step changed query or Adam state")
            skipped += 1
            previous_state, previous_optimizer = row["state_after"], row["optimizer_after"]
            continue

        if not active:
            _require(row.get("skipped") and row.get("reason") == "no_positive_weight_anchor",
                     "empty positive-anchor query was not skipped")
            _require(row["boxes_before"] is None and row["boxes_after"] is None
                     and "rollout" not in row,
                     "empty-anchor query has inconsistent output witnesses")
            _require(row["decoder_counts"]["round_before"] == 0,
                     "empty-anchor query should not need a per-round decoder read")
            _require(_tree_equal(row["state_after"], row["state_before"])
                     and _tree_equal(row["optimizer_after"], row["optimizer_before"]),
                     "empty-anchor query changed state")
            skipped += 1
            previous_state, previous_optimizer = row["state_after"], row["optimizer_after"]
            continue

        _require(row["decoder_counts"]["round_before"] == 1 and row["boxes_before"] is not None,
                 "active step must save one actual before readout")
        _require(row["decoder_counts"]["candidate_trial"] == len(row["candidates"]),
                 "candidate decoder-call count differs from actual trials")
        _require(row.get("rollout") is not None, "active step is missing its rollout witness")
        reconstructed = _rollout_and_loss(row, evidence, cfg, positions, f["seed_key"], step)
        max_loss_error = max(max_loss_error, reconstructed["loss_error"])
        max_gradient_error = max(max_gradient_error, reconstructed["mean_gradient_error"])
        max_reward_error = max(max_reward_error, reconstructed["reward_error"])
        max_q_error = max(max_q_error, reconstructed["q_error"])

        if not reconstructed["informative"]:
            _require(row.get("skipped") and row.get("reason") == "no_information",
                     "flat-reward step did not use the saved flat-skip rule")
            for key in ("mean_gradient", "gradient", "original_adam_delta", "filtered_delta", "optimizer_proposed"):
                _require(row.get(key) is None, f"flat-skip fabricated {key}")
            _require(not row["candidates"] and not row["accepted"] and not row["rollback"],
                     "flat-skip must not propose or reject an update")
            _require(_tree_equal(row["state_after"], row["state_before"])
                     and _tree_equal(row["optimizer_after"], row["optimizer_before"]),
                     "flat-skip changed Query or Adam state")
            _require(_err(row["boxes_after"], row["boxes_before"]) == 0.0,
                     "flat-skip boxes differ from the actual before readout")
            skipped += 1
            previous_state, previous_optimizer, previous_boxes = (
                row["state_after"], row["optimizer_after"], row["boxes_after"])
            continue

        _require(not row.get("skipped"), "informative step was unexpectedly skipped")
        _require(row["mean_gradient"] is not None and row["gradient"] is not None,
                 "proposal is missing actual gradient witnesses")
        _require(np.isfinite(_arr(row["gradient"])).all(), "saved parameter gradient is non-finite")
        _require(np.isfinite(_arr(row["original_adam_delta"])).all(), "saved Adam direction is non-finite")
        adam = _adam_audit(row, state_names)
        adam_rows.append(adam)
        solver_info = _solver_audit(row, protection, f["mode"])
        solver_rows.append(solver_info)
        if f["mode"] == "M2" and row["solver"] is not None and row["solver"].get("status") != "identity":
            solver_calls += 1
            solver_seconds += float(row["solver_seconds"] or 0.0)
        elif row["solver_seconds"] is not None:
            raise AssertionError("solver timing exists without an actual solver call")
        if row["solver"] is not None and row["solver"].get("status") == "identity":
            _require(row["solver_seconds"] is None, "identity direction must not invent solver time")

        trials = row["candidates"]
        schedule = _RECIPE_SCALES[recipe]
        _require(0 < len(trials) <= len(schedule)
                 and [float(c["scale"]) for c in trials] == schedule[:len(trials)],
                 "candidate scales differ from the selected Lite recipe")
        before_iou = _arr(row["reference_iou_before"]).astype(np.float64)
        found_accept = False
        for index, candidate in enumerate(trials):
            boxes = _arr(candidate["boxes"]).astype(np.float64)
            after_iou = _iou(boxes[positions], evidence)
            finite = bool(np.isfinite(boxes).all())
            _require(bool(candidate["finite"]) == finite, "candidate finite bit mismatch")
            _require(_err(candidate["anchor_IoU"], after_iou) <= 2e-6,
                     "candidate pseudo-reference IoU differs from saved boxes")
            reference_pass = bool(np.isfinite(after_iou).all()
                                  and float(np.mean(after_iou - before_iou)) >= -1e-7)
            _require(bool(candidate["reference_pass"]) == reference_pass,
                     "candidate accept_reference_step result mismatch")
            expected_accept = finite and reference_pass
            _require(bool(candidate["accepted"]) == expected_accept,
                     "candidate acceptance used a gate beyond finiteness and reference IoU")
            _require(index == len(trials) - 1 or not candidate["accepted"],
                     "candidate list continued after first accepted scale")
            found_accept = found_accept or expected_accept
        _require(bool(row["accepted"]) == found_accept and bool(row["rollback"]) == (not found_accept),
                 "fit accepted/rollback flags differ from candidate witnesses")
        proposals += 1
        if found_accept:
            accepted += 1
            last = trials[-1]
            _require(float(row["accepted_scale"]) == float(last["scale"]) and last["accepted"],
                     "accepted scale differs from final accepted candidate")
            direction = _arr(row["filtered_delta"]).astype(np.float64).reshape(-1)
            offset = 0
            for name, before_value in row["state_before"].items():
                before_array = _arr(before_value)
                width = before_array.size
                expected_after = before_array + (direction[offset:offset + width]
                                                   .reshape(before_array.shape) * float(row["accepted_scale"]))
                _require(_err(row["state_after"][name], expected_after) <= 3e-6,
                         f"accepted state update mismatch for {name}")
                offset += width
            _require(offset == len(direction), "filtered parameter direction has wrong size")
            _require(_tree_equal(row["optimizer_after"], row["optimizer_proposed"]),
                     "accepted update did not retain proposed Adam state")
            _require(_err(row["boxes_after"], last["boxes"]) == 0.0,
                     "accepted after-box witness differs from candidate readout")
            _require(row["decoder_counts"]["candidate_trial"] == len(trials),
                     "accepted decode count mismatch")
            previous_boxes = row["boxes_after"]
        else:
            rejected += 1
            _require(len(trials) == len(schedule), "rejected update did not exhaust recipe scales")
            _require(row["accepted_scale"] is None, "rejected update has an accepted scale")
            _require(_tree_equal(row["state_after"], row["state_before"]),
                     "rejected update did not restore parameters")
            _require(_tree_equal(row["optimizer_after"], row["optimizer_before"]),
                     "rejected update did not restore Adam moments")
            _require(_err(row["boxes_after"], row["boxes_before"]) == 0.0,
                     "rejected update did not restore its before-box readout")
            stopped = True
            previous_boxes = row["boxes_after"]

        previous_state, previous_optimizer = row["state_after"], row["optimizer_after"]

    _require(_tree_equal(previous_state, f["final_state"])
             and _tree_equal(f["fit_final_state"], f["final_state"]),
             "final state differs from accepted/rollback state chain")
    if f["qualification_requested"]:
        computed = any(row.get("qualification_diagnostics") is not None for row in path)
        _require((f["qualification_diagnostics"] is not None) == computed,
                 "top-level qualification witness should be None when no diagnostic ran")
    else:
        _require(f.get("qualification_diagnostics") is None, "disabled diagnostics must be None")

    decode_counts = f["decode_counts"]
    _require(decode_counts == decode_sums | {"final_cached_readout": 1},
             "top-level decode counts differ from per-step calls and one final cached readout")
    _require(f["decode_calls_total"] == sum(decode_counts.values()), "total decoder-call count mismatch")
    _require(decode_counts.get("final_cached_readout") == 1,
             "decode count must include exactly one final cached decoder readout")
    _require(decode_counts.get("qualification_fd_plus", 0)
             == decode_counts.get("qualification_fd_minus", 0),
             "qualification finite differences must have matched plus/minus reads")
    _require(all(isinstance(value, int) and value >= 0 for value in decode_counts.values()),
             "decode counts must be actual nonnegative integer counts")
    _require(all(math.isfinite(float(value)) and float(value) >= 0
                 for value in f["decoder_seconds_by_purpose"].values()),
             "per-purpose decoder times must be finite nonnegative measurements")
    _require(f["solver_calls"] == solver_calls
             and abs(float(f["solver_seconds_total"]) - solver_seconds) <= 2e-5,
             "solver call/time totals differ from per-step measurements")
    _require(f["optimizer_steps"] == accepted and f["proposed_optimizer_steps"] == proposals,
             "optimizer proposal/accept counts differ from path")

    return {
        "status": "pass",
        "recipe": recipe,
        "response_scope": f["response_scope"],
        "steps": len(path),
        "proposals": proposals,
        "accepted": accepted,
        "rejected": rejected,
        "skipped": skipped,
        "solver_calls": solver_calls,
        "decode_counts": dict(decode_counts),
        "max_loss_error": max_loss_error,
        "max_mean_gradient_error": max_gradient_error,
        "max_reward_error": max_reward_error,
        "max_q_error": max_q_error,
        "adam_rows": len(adam_rows),
        "solver_rows": solver_rows,
        "GT_read": False,
        "CPU_only_audit": True,
    }

