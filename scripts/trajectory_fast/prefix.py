"""Recover an exact lower-budget terminal state from a saved policy path.

This module does not run a decoder, compute a reward, sample candidates, or read
labels. It follows the call order in ``trajectory_opd.policy.fit``. A lower
budget agrees with the saved run until its last round: the round admission test
reserves 35 calls for sampled methods (32 candidates and three validations), or
two calls for point methods. Only validation can therefore be cut mid-round.

Every proposal is made from the common round-before state. An accepted
component can be copied bit for bit from that round's saved path; a component
whose acceptance has not been evaluated retains its round-before state. No
optimizer state or unsaved point proposal is needed because a truncated round
exhausts the budget and is necessarily the terminal round.
"""
from __future__ import annotations

import copy
import math
from numbers import Integral

import torch


ARMS = ("TrajectoryOPD", "Direct", "DirectMulti", "DirectMC")
SAMPLED = ("TrajectoryOPD", "DirectMC")
SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625)
ACCEPT_EPSILON = 1e-8


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _values(values, n):
    _require(isinstance(values, list) and len(values) == n,
             "One saved utility dictionary is required per component")
    for value in values:
        _require(isinstance(value, dict) and "U" in value,
                 "Saved utility dictionary is missing U")
        _require(all(isinstance(v, (int, float)) and math.isfinite(v)
                     for v in value.values()), "Nonfinite saved utility")


def _states(states, shape=None):
    _require(torch.is_tensor(states) and states.device.type == "cpu",
             "Saved states must be CPU tensors; this utility never uses CUDA")
    _require(states.is_floating_point() and states.ndim == 3,
             "Saved states must have [components, knots, channels] shape")
    _require(bool(torch.isfinite(states).all()), "Nonfinite saved state")
    if shape is not None:
        _require(tuple(states.shape) == tuple(shape), "State shape changed")


def validate_trace(trace):
    """Validate saved call accounting and component transitions, without replay.

    Returns a compact original-run account. This also verifies the saved final
    readout before a shorter prefix is recovered, so corrupt late records are
    not silently accepted merely because the requested prefix ends earlier.
    """
    arm = trace.get("arm")
    _require(arm in ARMS, "Unknown policy arm")
    sampled = arm in SAMPLED
    states = trace["initial_states"]
    _states(states)
    n = len(states)
    _require(n == (1 if arm == "Direct" else 3), "Unexpected component count")
    current = trace["initial_values"]
    _values(current, n)
    rounds, path = trace["rounds"], trace["path"]
    _require(len(rounds) == len(path), "Round/path length mismatch")
    costs = trace["costs"]
    allowance = int(costs["allowance"])
    evals, backwards, accepted, rejected = n, 0, 0, 0
    for i, (rr, pp) in enumerate(zip(rounds, path)):
        _require(rr["round"] == i and pp["step"] == i + 1,
                 "Nonconsecutive saved rounds")
        _require(evals + (35 if sampled else 2) <= allowance,
                 "Saved round violates its original admission test")
        _require(rr["before"] == current, "Round-before utility mismatch")
        _states(pp["states"], states.shape)
        _values(rr["after"], n)
        _require(pp["values"] == rr["after"], "Saved path utility mismatch")
        bt = rr["backtracking"]
        components = [item["component"] for item in bt]
        if sampled:
            _require(components == [0, 1, 2], "Sampled component order changed")
            counts = rr["counts"]
            _require(len(counts) == 3 and sum(counts) == 32,
                     "Expected the original 32-candidate cloud")
            _require(len(rr["candidate_states"]) == 3 and
                     all(len(c) == k for c, k in zip(rr["candidate_states"], counts)),
                     "Saved candidate count mismatch")
            evals += 32
            backwards += 32 if arm == "DirectMC" else 0
        else:
            _require(len(components) == 1 and components[0] in range(n),
                     "Invalid point-method component")
            _require("gradient_evaluation" in rr, "Missing point evaluation")
            evals += 1
            backwards += 1
        saved_accepted = []
        for item in bt:
            c, attempts = item["component"], item["attempts"]
            _require(len(attempts) <= len(SCALES), "Too many backtracking trials")
            for j, attempt in enumerate(attempts):
                _require(evals < allowance, "Saved trial exceeds original allowance")
                _require(attempt["scale"] == SCALES[j], "Backtracking scale/order changed")
                _require(all(isinstance(v, (int, float)) and math.isfinite(v)
                             for v in attempt.values()), "Nonfinite saved trial")
                is_accept = attempt["U"] > current[c]["U"] + ACCEPT_EPSILON
                _require(not is_accept or j == len(attempts) - 1,
                         "Saved policy continued after accepting a trial")
                evals += 1
            was_accepted = bool(attempts) and (
                attempts[-1]["U"] > current[c]["U"] + ACCEPT_EPSILON)
            _require(item["accepted"] is was_accepted, "Acceptance flag mismatch")
            if was_accepted:
                value = {k: v for k, v in attempts[-1].items() if k != "scale"}
                _require(value == rr["after"][c], "Accepted utility mismatch")
                saved_accepted.append(c)
                accepted += 1
            else:
                _require(rr["after"][c] == current[c], "Rejected utility changed")
                _require(torch.equal(pp["states"][c], states[c]),
                         "Rejected component state changed")
                rejected += 1
        for c in set(range(n)) - set(components):
            _require(rr["after"][c] == current[c] and
                     torch.equal(pp["states"][c], states[c]),
                     "Unselected component changed")
        _require(rr["accepted_components"] == saved_accepted,
                 "Accepted component list mismatch")
        _require(rr["full_trajectory_evals"] == evals,
                 "Saved round call count mismatch")
        current, states = rr["after"], pp["states"]
        best = max(range(n), key=lambda c: current[c]["U"])
        _require(pp["best_component"] == best, "Saved path readout mismatch")
    best = max(range(n), key=lambda c: current[c]["U"])
    _require(torch.equal(states, trace["final_states"]), "Saved final states mismatch")
    _require(trace["best_component"] == best and
             torch.equal(states[best], trace["final_z"]), "Saved final readout mismatch")
    for key, expected in (("full_trajectory_evals", evals),
                          ("native_offset_decoder_calls", 2 * evals),
                          ("decoder_backwards", backwards), ("updates", len(rounds)),
                          ("accepted", accepted), ("accepted_updates", accepted),
                          ("rejected", rejected)):
        _require(costs[key] == expected, f"Saved cost mismatch: {key}")
    return {"allowance": allowance, "evals": evals, "rounds": len(rounds),
            "components": n, "decoder_backwards": backwards}


def recover_prefix(trace, budget=256):
    """Return the exact terminal states under the same policy with less budget.

    ``budget`` counts full trajectory evaluations, excluding the unchanged
    replay setup checks. The result contains inherited setup call counts, but
    no extrapolated wall time or memory estimate. ``rounds_complete`` counts
    fully consumed saved rounds; ``updates`` includes a truncated last round.
    ``partial_final_round`` concerns the stored path, not scientific accuracy.
    """
    _require(isinstance(budget, Integral) and not isinstance(budget, bool),
             "Budget must be an integer")
    budget = int(budget)
    original = validate_trace(trace)
    _require(original["components"] <= budget <= original["allowance"],
             "Budget must cover initialization and not exceed saved allowance")
    arm = trace["arm"]
    sampled = arm in SAMPLED
    states = trace["initial_states"].clone()
    current = copy.deepcopy(trace["initial_values"])
    n = len(states)
    evals = n
    candidate_evals = gradient_evals = trial_evals = backwards = 0
    complete = updates = accepted = rejected = 0
    partial = False
    consumed = []
    reason = "saved_trace_end"
    for rr, pp in zip(trace["rounds"], trace["path"]):
        needed = 35 if sampled else 2
        if evals + needed > budget:
            reason = "next_round_admission_exceeds_budget"
            break
        _require(current == rr["before"], "Recovered prefix diverged before round")
        round_begin = evals
        if sampled:
            evals += 32
            candidate_evals += 32
            backwards += 32 if arm == "DirectMC" else 0
        else:
            evals += 1
            gradient_evals += 1
            backwards += 1
        details = []
        truncated = False
        for item in rr["backtracking"]:
            c = item["component"]
            available = min(len(item["attempts"]), budget - evals)
            evals += available
            trial_evals += available
            truncated |= available < len(item["attempts"])
            found = item["accepted"] and available == len(item["attempts"])
            if found:
                # Only the accepted saved component is copied; the old final
                # round state of any not-yet-validated component is never used.
                states[c].copy_(pp["states"][c])
                current[c] = copy.deepcopy(rr["after"][c])
                accepted += 1
            else:
                rejected += 1
            details.append({"component": c, "attempts_consumed": available,
                            "accepted": bool(found)})
        updates += 1
        consumed.append({"round": rr["round"], "evals_before": round_begin,
                         "evals_after": evals, "backtracking": details,
                         "complete": not truncated})
        if truncated:
            _require(evals == budget, "Partial validation must exhaust budget")
            partial = True
            reason = "budget_exhausted_during_backtracking"
            break
        complete += 1
        _require(torch.equal(states, pp["states"]) and current == rr["after"],
                 "Fully consumed round did not recover its saved state")
    best = max(range(n), key=lambda c: current[c]["U"])
    costs = trace["costs"]
    calls = dict(initial_evals=n, candidate_evals=candidate_evals,
                 gradient_evals=gradient_evals, backtracking_evals=trial_evals,
                 full_trajectory_evals=evals, native_offset_decoder_calls=2 * evals,
                 decoder_backwards=backwards, updates=updates,
                 accepted=accepted, accepted_updates=accepted, rejected=rejected,
                 allowance=budget,
                 setup_full_trajectory_evals=costs.get("setup_full_trajectory_evals", 0),
                 setup_native_offset_decoder_calls=costs.get("setup_native_offset_decoder_calls", 0))
    _require(evals == n + candidate_evals + gradient_evals + trial_evals <= budget,
             "Recovered call count exceeds budget")
    if budget == original["allowance"]:
        _require(torch.equal(states, trace["final_states"]) and
                 torch.equal(states[best], trace["final_z"]) and
                 best == trace["best_component"] and evals == original["evals"],
                 "Full allowance failed exact saved-run recovery")
    return dict(states=states, best_state=states[best].clone(), best_index=best,
                bestU=float(current[best]["U"]), current_values=current,
                evals=evals, rounds_complete=complete,
                partial_final_round=partial, callcounts=calls,
                consumed_rounds=consumed,
                explanation=dict(reason=reason, original_allowance=original["allowance"],
                                 original_evals=original["evals"],
                                 original_rounds=original["rounds"],
                                 setup_excluded_from_budget=True,
                                 new_candidates=0, new_reward_evaluations=0,
                                 decoder_calls_performed=0, labels_read=False,
                                 state_source="saved accepted component paths"))
