"""CPU-only exact-prefix tests; optional saved-receipt audit, without scoring."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import unittest
from pathlib import Path

import torch

from scripts.trajectory_fast.prefix import recover_prefix, validate_trace


def synthetic(arm="TrajectoryOPD", accepted=(True, True, True), lengths=(1, 3, 2)):
    sampled = arm in ("TrajectoryOPD", "DirectMC")
    n = 1 if arm == "Direct" else 3
    states = torch.arange(n * 4, dtype=torch.float32).reshape(n, 2, 2)
    values = [{"U": float(c)} for c in range(n)]
    trace = dict(arm=arm, initial_states=states.clone(), initial_values=copy.deepcopy(values),
                 path=[], rounds=[])
    components = list(range(3)) if sampled else [n - 1]
    bt, accepted_components = [], []
    before = copy.deepcopy(values)
    for c in components:
        length = lengths[c]
        attempts = [{"scale": (1., .5, .25, .125, .0625)[j], "U": before[c]["U"] - 1.}
                    for j in range(length)]
        if accepted[c]:
            attempts[-1]["U"] = before[c]["U"] + .5
            values[c] = {"U": attempts[-1]["U"]}
            states[c] += 10. * (c + 1)
            accepted_components.append(c)
        bt.append(dict(component=c, attempts=attempts, accepted=accepted[c]))
    evals = n + (32 if sampled else 1) + sum(len(t["attempts"]) for t in bt)
    rr = dict(round=0, before=before, after=copy.deepcopy(values),
              backtracking=bt, accepted_components=accepted_components,
              full_trajectory_evals=evals)
    if sampled:
        rr.update(counts=[12, 10, 10], candidate_states=[torch.zeros(k, 2, 2) for k in [12, 10, 10]])
    else:
        rr["gradient_evaluation"] = {"U": before[components[0]]["U"]}
    trace["rounds"].append(rr)
    best = max(range(n), key=lambda c: values[c]["U"])
    trace["path"].append(dict(step=1, states=states.clone(), values=copy.deepcopy(values), best_component=best))
    trace.update(final_states=states.clone(), final_z=states[best].clone(), best_component=best,
                 costs=dict(allowance=1321, full_trajectory_evals=evals,
                            native_offset_decoder_calls=2 * evals,
                            decoder_backwards=32 if arm == "DirectMC" else 0 if sampled else 1,
                            updates=1, accepted=len(accepted_components), accepted_updates=len(accepted_components),
                            rejected=len(components) - len(accepted_components),
                            setup_full_trajectory_evals=2, setup_native_offset_decoder_calls=4))
    return trace


class PrefixTests(unittest.TestCase):
    def test_before_candidate_admission(self):
        t = synthetic()
        r = recover_prefix(t, 37)  # 3 initial + needed35 does not fit.
        self.assertEqual((r["evals"], r["rounds_complete"]), (3, 0))
        self.assertTrue(torch.equal(r["states"], t["initial_states"]))
        self.assertFalse(r["partial_final_round"])

    def test_partial_backtracking_keeps_prior_accept_only(self):
        t = synthetic()
        r = recover_prefix(t, 38)  # Candidate32 + c0 accept + two c1 rejects.
        self.assertTrue(r["partial_final_round"])
        self.assertEqual(r["evals"], 38)
        self.assertEqual(r["rounds_complete"], 0)
        self.assertTrue(torch.equal(r["states"][0], t["final_states"][0]))
        self.assertTrue(torch.equal(r["states"][1:], t["initial_states"][1:]))
        self.assertEqual(r["callcounts"]["accepted"], 1)
        self.assertEqual(r["callcounts"]["rejected"], 2)

    def test_cut_after_second_accept_before_third(self):
        t = synthetic(lengths=(1, 2, 3))
        r = recover_prefix(t, 38)
        self.assertTrue(torch.equal(r["states"][:2], t["final_states"][:2]))
        self.assertTrue(torch.equal(r["states"][2], t["initial_states"][2]))
        self.assertEqual(r["callcounts"]["accepted"], 2)

    def test_cut_failed_trials_never_moves_component(self):
        t = synthetic(accepted=(False, True, False), lengths=(5, 2, 5))
        r = recover_prefix(t, 38)
        self.assertTrue(torch.equal(r["states"], t["initial_states"]))
        self.assertEqual(r["callcounts"]["accepted"], 0)
        self.assertEqual(r["current_values"], t["initial_values"])

    def test_exact_end_round(self):
        t = synthetic()
        r = recover_prefix(t, t["costs"]["full_trajectory_evals"])
        self.assertFalse(r["partial_final_round"])
        self.assertEqual(r["rounds_complete"], 1)
        self.assertTrue(torch.equal(r["states"], t["final_states"]))

    def test_full_allowance_all_arms(self):
        for arm in ("TrajectoryOPD", "Direct", "DirectMulti", "DirectMC"):
            with self.subTest(arm=arm):
                t = synthetic(arm)
                r = recover_prefix(t, 1321)
                self.assertTrue(torch.equal(r["states"], t["final_states"]))
                self.assertTrue(torch.equal(r["best_state"], t["final_z"]))
                self.assertEqual(r["best_index"], t["best_component"])
                self.assertEqual(r["callcounts"]["decoder_backwards"], t["costs"]["decoder_backwards"])

    def test_point_selected_component_from_record(self):
        t = synthetic("DirectMulti", lengths=(1, 1, 5))
        r = recover_prefix(t, 5)  # Initial3 + point1 + first rejected trial.
        self.assertTrue(r["partial_final_round"])
        self.assertTrue(torch.equal(r["states"], t["initial_states"]))
        self.assertEqual(r["consumed_rounds"][0]["backtracking"][0]["component"], 2)
        self.assertEqual(r["callcounts"]["gradient_evals"], 1)

    def test_tied_readout_uses_first_component(self):
        t = synthetic("DirectMulti", lengths=(1, 1, 1))
        t["initial_values"] = [{"U": 0.}] * 3
        rr = t["rounds"][0]
        rr["before"] = copy.deepcopy(t["initial_values"])
        rr["backtracking"][0]["attempts"] = [{"scale": 1., "U": .5}]
        rr["after"] = [{"U": 0.}, {"U": 0.}, {"U": .5}]
        t["path"][0]["values"] = copy.deepcopy(rr["after"])
        r = recover_prefix(t, 3)
        self.assertEqual(r["best_index"], 0)

    def test_reject_corrupt_accounting_and_invalid_budget(self):
        t = synthetic()
        for budget in (2, 1322, True, 256.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                recover_prefix(t, budget)
        t["rounds"][0]["full_trajectory_evals"] += 1
        with self.assertRaises(ValueError):
            validate_trace(t)

    def test_reject_nonfinite_and_bad_accept_flag(self):
        t = synthetic()
        t["rounds"][0]["backtracking"][0]["accepted"] = False
        with self.assertRaises(ValueError):
            recover_prefix(t)
        t = synthetic()
        t["initial_states"][0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            recover_prefix(t)


def audit_saved(root):
    """Read only complete receipt-bound traces; no NPZ, GT or decoder access."""
    receipts = sorted((root / "predictions").glob("*.json"))
    if not receipts:
        raise RuntimeError("No complete saved query receipts")
    counts, partial = {}, {}
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("GT_read") is not False:
            raise ValueError("Receipt is not an inference-only artifact")
        trace_path = root / receipt["trace_path"]
        digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        if digest != receipt["trace_sha256"]:
            raise ValueError("Saved trace hash mismatch")
        traces = torch.load(trace_path, map_location="cpu", weights_only=False)
        for arm, trace in traces.items():
            long = recover_prefix(trace, trace["costs"]["allowance"])
            short = recover_prefix(trace, 256)
            assert torch.equal(long["states"], trace["final_states"])
            assert torch.equal(long["best_state"], trace["final_z"])
            assert long["best_index"] == trace["best_component"]
            assert long["evals"] == trace["costs"]["full_trajectory_evals"]
            assert short["evals"] <= 256
            counts[arm] = counts.get(arm, 0) + 1
            partial[arm] = partial.get(arm, 0) + int(short["partial_final_round"])
    return dict(queries=len(receipts), full_allowance_exact_per_arm=counts,
                short_budget=256, partial_last_round_per_arm=partial,
                model_calls=0, new_candidates=0, reward_evaluations=0, GT_read=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PrefixTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.cached_root:
        print(json.dumps(audit_saved(args.cached_root), sort_keys=True))
