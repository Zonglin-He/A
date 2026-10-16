"""Synthetic CPU contracts for Spatial-Lite; never loads a model or reads labels."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def contracts():
    import torch
    from scripts import stvg_spatial_lite_fit_v1 as lite
    from scripts import audit_stvg_spatial_lite_fit002 as lite_audit
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts.stvg_weightfree_coordinate_fit002 import loss_per_anchor as full_loss
    from vg_tta.decota_spatial_opd_tunable_v1 import rollout
    from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou

    if torch.cuda.is_initialized():
        raise RuntimeError("Spatial-Lite synthetic contract requires CUDA to be uninitialized")
    torch.set_num_threads(2)
    checks = 0

    # Check the detached reward geometry and score-function gradient directly.
    cfg = dict(core.load_original_opd_config("hc2"), steps=2, lr=0.005)
    mean = torch.tensor([[0.30, 0.42, 0.18, 0.21],
                         [0.55, 0.38, 0.25, 0.20]], dtype=torch.float32, requires_grad=True)
    evidence = torch.tensor([[0.34, 0.40, 0.20, 0.20],
                             [0.52, 0.40, 0.22, 0.18]], dtype=torch.float32)
    policy = rollout(mean.detach(), evidence, "spatial_lite_geometry", 0, config=cfg)
    assert policy["samples"].shape == (2, 32, 4)
    assert torch.allclose(policy["samples"][:, :16] + policy["samples"][:, 16:],
                          2 * policy["mean"][:, None, :], atol=2e-7, rtol=0)
    checks += 1
    expected_rewards = aligned_iou(torch.sigmoid(policy["samples"]), evidence[:, None, :])
    assert torch.allclose(policy["rewards"], expected_rewards, atol=1e-7, rtol=0)
    assert torch.equal(policy["rewards"], policy["used_rewards"])
    expected_q = torch.softmax(expected_rewards / cfg["tau"], -1)
    assert torch.allclose(policy["weights"], expected_q, atol=1e-7, rtol=1e-6)
    checks += 1

    lite_loss = lite._loss_per_anchor(mean, policy, cfg["sigma"]).mean()
    lite_mean_grad = torch.autograd.grad(lite_loss, mean, retain_graph=True)[0]
    sigma = cfg["sigma"]
    manual_mean_grad = -(
        policy["weights"][:, :, None]
        * (policy["samples"] - mean[:, None, :])
    ).sum(1) / (sigma ** 2 * len(mean))
    assert torch.allclose(lite_mean_grad, manual_mean_grad, atol=2e-6, rtol=2e-5)
    checks += 1

    # The old control variate and policy KL have zero mean-gradient here because
    # actions are antithetic and the rollout mean is the current mean.
    old_loss = full_loss(mean, policy, sigma).mean()
    old_mean_grad = torch.autograd.grad(old_loss, mean, retain_graph=True)[0]
    assert torch.allclose(old_mean_grad, lite_mean_grad, atol=2e-6, rtol=2e-5)
    old_kl = (mean - policy["mean"]).square().sum(-1) / (2 * sigma ** 2)
    kl_grad = torch.autograd.grad(old_kl.sum(), mean, retain_graph=True)[0]
    logp = -(policy["samples"] - mean[:, None, :]).square().sum(-1) / (2 * sigma ** 2)
    logp = logp - 2 * math.log(2 * math.pi * sigma ** 2)
    uniform_baseline = (logp.mean(-1) / 32).mean()
    cv_grad = torch.autograd.grad(uniform_baseline, mean)[0]
    assert float(kl_grad.abs().max()) == 0.0
    assert float(cv_grad.abs().max()) < 2e-6
    checks += 1

    class Replay:
        def __init__(self):
            generator = torch.Generator(device="cpu").manual_seed(451)
            self.named = [
                ("spatial.query_residual", torch.nn.Parameter(torch.zeros(256))),
                ("spatial.ln.weight", torch.nn.Parameter(torch.zeros(768))),
                ("spatial.ln.bias", torch.nn.Parameter(torch.zeros(768))),
            ]
            self.matrix = torch.randn(1792, 32, generator=generator) * 0.035
            self.source_state = self.state()

        def state(self):
            return {name: parameter.detach().clone() for name, parameter in self.named}

        def restore(self, state):
            with torch.no_grad():
                for name, parameter in self.named:
                    parameter.copy_(state[name])
                    parameter.grad = None

        def values(self):
            vector = torch.cat([parameter.reshape(-1) for _, parameter in self.named])
            return {"boxes": torch.sigmoid(vector @ self.matrix).reshape(8, 4) * 0.5 + 0.15}

    class Base:
        def __init__(self):
            self.zero = {"boxes": torch.zeros(8, 4)}
            self.restore_calls = []

        def restore(self, state):
            self.restore_calls.append(copy.deepcopy(state))

    replay_instances = []
    original_trainable_replay = core._trainable_replay

    def fake_trainable_replay(base, ids, key, scope):
        assert ids == list(range(8)) and scope == "joint"
        replay = Replay()
        replay_instances.append(replay)
        return replay, replay.named

    anchors = [
        {"position": 1, "box": torch.tensor([0.35, 0.35, 0.40, 0.40]), "weight": 0.7},
        {"position": 6, "box": torch.tensor([0.35, 0.35, 0.40, 0.40]), "weight": 0.9},
    ]
    settings = {"mu": 0.125, "cg_max_iter": 3, "cg_rtol": 1e-5,
                # These old acceptance fields are present in the caller's saved
                # settings but Spatial-Lite must not consume them.
                "response_ratio": 0.0, "response_abs_tolerance": 0.0,
                "backtracking": [1.0, 0.5, 0.25, 0.125]}

    try:
        core._trainable_replay = fake_trainable_replay

        # Full-grid response positions preserve the prior behavior, while each
        # query starts with an empty Adam state and restores source state on exit.
        base = Base()
        full_a = lite.fit(base, {"source": "fixture"}, anchors, list(range(8)),
                          "lite_cpu_full", cfg, "M1", settings,
                          recipe="lite_backtrack", response_scope="full_grid")
        full_b = lite.fit(base, {"source": "fixture"}, anchors, list(range(8)),
                          "lite_cpu_full", cfg, "M1", settings,
                          recipe="lite_backtrack", response_scope="full_grid")
        assert full_a["protection_positions"] == [0, 2, 3, 4, 5, 7]
        assert full_a["path"][0]["optimizer_before"]["state"] == {}
        assert full_b["path"][0]["optimizer_before"]["state"] == {}
        assert full_a["anchor_weights"] == [1.0, 1.0]
        assert full_a["qualification_diagnostics"] is None
        assert all(row["qualification_diagnostics"] is None for row in full_a["path"])
        assert full_a["decode_counts"].get("qualification_fd_plus") == 0
        assert full_a["decode_counts"].get("qualification_fd_minus") == 0
        assert full_a["decode_counts"]["final_cached_readout"] == 1
        assert full_a["decode_calls_total"] == sum(full_a["decode_counts"].values())
        assert len(base.restore_calls) == 4
        assert all(call == {"source": "fixture"} for call in base.restore_calls)
        assert core._tree_equal(replay_instances[-1].state(), replay_instances[-1].source_state)
        assert not torch.cuda.is_initialized()
        reference_record = {"anchors": [
            {"position": item["position"], "box_cxcywh": item["box"].tolist(),
             "weight": item["weight"]} for item in anchors]}
        full_audit = lite_audit.fit_audit(full_a, reference_record, "lite_backtrack")
        assert full_audit["status"] == "pass"
        from scripts.audit_stvg_soft_ep_v1 import _optimizer
        proposal_row = next((row for row in full_a["path"] if row["gradient"] is not None), None)
        if proposal_row is not None:
            assert max(_optimizer(proposal_row, full_a["active_names"])) < 5e-6
        checks += 1

        # Explicit event support is range-checked and anchor positions are
        # removed. This reaches the actual matrix-free response solver.
        event_base = Base()
        event = lite.fit(event_base, {"source": "fixture"}, anchors, list(range(8)),
                         "lite_cpu_event", cfg, "M2", settings,
                         recipe="lite_fullstep", response_scope="event_support",
                         event_support_positions=[2, 4, 6])
        assert event["event_support_positions"] == [2, 4, 6]
        assert event["protection_positions"] == [2, 4]
        assert event["qualification_diagnostics"] is None
        assert event["decode_counts"]["round_before"] == sum(
            "rollout" in row for row in event["path"])
        assert event["decode_counts"]["candidate_trial"] == sum(
            len(row["candidates"]) for row in event["path"])
        assert event["decode_counts"]["final_cached_readout"] == 1
        assert event["decode_counts"]["qualification_fd_plus"] == 0
        assert event["decode_counts"]["qualification_fd_minus"] == 0
        assert event["solver_calls"] == sum(
            row["solver"] is not None
            and row["solver"].get("status") not in {"identity"}
            for row in event["path"])
        if event["path"][0]["solver"] is not None:
            assert event["path"][0]["solver"]["status"] in {"converged", "max_iter", "near_zero_response_identity"}
        event_audit = lite_audit.fit_audit(
            event, reference_record, "lite_fullstep", response_positions=[2, 4, 6])
        assert event_audit["status"] == "pass"
        assert not torch.cuda.is_initialized()
        checks += 1

        # Empty U is an explicit identity direction, with no solver call or
        # made-up response statistics.
        empty_u = lite.fit(Base(), {"source": "fixture"}, anchors, list(range(8)),
                           "lite_cpu_empty_u", cfg, "M2", settings,
                           recipe="lite_fullstep", response_scope="event_support",
                           event_support_positions=[1, 6])
        assert empty_u["protection_positions"] == []
        first = empty_u["path"][0]
        if first["original_adam_delta"] is not None:
            assert first["solver"] == {"status": "identity", "reason": "empty_protection_support"}
            assert torch.equal(first["original_adam_delta"], first["filtered_delta"])
            assert first["solver"].get("lambda") is None
        assert empty_u["solver_calls"] == 0
        checks += 1

        # A synthetic reject exercises rollback of both parameters and Adam
        # moments, all four backtracking candidates, and same-Query stopping.
        rejection_base = Base()
        original_accept = lite.accept_reference_step
        accept_calls = []

        def reject_for_contract(before, after, weights):
            accept_calls.append((before.detach().clone(), after.detach().clone(), weights.detach().clone()))
            return False

        lite.accept_reference_step = reject_for_contract
        try:
            rejected = lite.fit(rejection_base, {"source": "fixture"}, anchors,
                                list(range(8)), "lite_cpu_rollback", cfg, "M1", settings,
                                recipe="lite_backtrack", response_scope="full_grid")
        finally:
            lite.accept_reference_step = original_accept
        first = rejected["path"][0]
        assert len(accept_calls) == 4
        assert [item["scale"] for item in first["candidates"]] == [1.0, 0.5, 0.25, 0.125]
        assert first["rollback"] and not first["accepted"]
        assert core._tree_equal(first["state_before"], first["state_after"])
        assert torch.equal(first["boxes_before"], first["boxes_after"])
        assert core._tree_equal(first["optimizer_before"], first["optimizer_after"])
        assert first["optimizer_proposed"]["state"]
        assert rejected["path"][1]["reason"] == "stopped_after_rejected_step"
        assert rejected["path"][1]["optimizer_before"]["state"] == {}
        assert first["solver"] is None
        assert rejected["decode_counts"]["candidate_trial"] == 4
        assert rejected["decode_counts"]["final_cached_readout"] == 1
        for field in ("loss", "gradient", "optimizer_after"):
            corrupted = copy.deepcopy(full_a)
            audit_row_index = next(i for i, row in enumerate(corrupted["path"])
                                   if row["gradient"] is not None)
            audit_row = corrupted["path"][audit_row_index]
            if field == "loss":
                audit_row["loss"] += 0.1
            elif field == "gradient":
                audit_row["gradient"][0] += 0.1
            else:
                audit_row["optimizer_after"]["state"][999] = {
                    "exp_avg": torch.ones(1), "exp_avg_sq": torch.ones(1), "step": torch.tensor(1.)}
            try:
                lite_audit.fit_audit(corrupted, reference_record, "lite_backtrack")
            except (AssertionError, KeyError, IndexError):
                checks += 1
            else:
                raise AssertionError(f"independent audit accepted corrupted {field}")
        assert not torch.cuda.is_initialized()
        checks += 1

        # Invalid event positions are rejected before replay/model work.
        try:
            lite._protection_positions(8, [1, 6], "event_support", [8])
        except ValueError:
            checks += 1
        else:
            raise AssertionError("out-of-grid event support position was accepted")
    finally:
        core._trainable_replay = original_trainable_replay

    assert not torch.cuda.is_initialized()
    return {
        "status": "pass",
        "scope": "synthetic CPU contract only; no model, GT, or GPU execution",
        "CPU_only": True,
        "GT_read": False,
        "GPU_used": False,
        "model_loaded": False,
        "checks": checks,
        "antithetic_and_geometry": True,
        "lite_gradient_matches_original_at_anchor": True,
        "zero_policy_kl_and_control_variate_gradient": True,
        "rollback_restores_adam_and_state": True,
        "event_support_subset_and_empty_identity": True,
        "default_qualification_forward_count": 0,
        "source_state_restored": True,
    }


if __name__ == "__main__":
    print(json.dumps(contracts(), indent=2, sort_keys=True))
