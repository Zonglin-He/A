"""Independent CPU audit of the sealed optimizer-replay artifacts.

This module only reads the locked fit outputs/configuration and recomputes the
saved policy, reward, softmax, loss, state-chain, and decoder/cache arithmetic.
It never opens labels or runs a model.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_reference_recovery_v1/Optimizer_replay"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tree_equal(left, right, torch):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right)
                and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(_tree_equal(left[k], right[k], torch) for k in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_tree_equal(a, b, torch) for a, b in zip(left, right)))
    return left == right


def _cxcywh_to_xyxy(a, np):
    x = np.asarray(a, dtype=np.float64)
    return np.concatenate((x[..., :2] - x[..., 2:] / 2.0,
                           x[..., :2] + x[..., 2:] / 2.0), axis=-1)


def _aligned_iou_cxcywh(a, b, np):
    pa, pb = _cxcywh_to_xyxy(a, np), _cxcywh_to_xyxy(b, np)
    inter_wh = np.maximum(np.minimum(pa[..., 2:], pb[..., 2:])
                          - np.maximum(pa[..., :2], pb[..., :2]), 0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_a = np.maximum(pa[..., 2] - pa[..., 0], 0.0) * np.maximum(pa[..., 3] - pa[..., 1], 0.0)
    area_b = np.maximum(pb[..., 2] - pb[..., 0], 0.0) * np.maximum(pb[..., 3] - pb[..., 1], 0.0)
    union = np.maximum(area_a + area_b - inter, 1e-7)
    return inter / union


def _max_abs(a, b, np) -> float:
    aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise AssertionError(f"shape mismatch {aa.shape} != {bb.shape}")
    return float(np.max(np.abs(aa - bb))) if aa.size else 0.0


def _fit_audit(fit, anchors, roster_record, cfg_expected, ref_sha, torch, np):
    if fit is None:
        raise AssertionError("expected a fit payload")
    assert fit["schema"] == "stvg-reference-recovery-fit-v1"
    assert fit["profile"] == "original" and fit["scope"] == "joint"
    assert fit["dataset"] == roster_record["dataset"]
    assert fit["config"] == cfg_expected
    assert fit["reference_sha256"] == ref_sha
    assert fit["reference_role"] == "fixed_weight_free_top1_teacher"
    assert fit["GT_read"] is False
    assert fit["active_parameters"] == 1792
    assert fit["optimizer_state_reset_each_query"] is True
    assert fit["query_residual_reset_at_arrival"] is True
    assert fit["validation_enabled"] is True
    expected_key = f"caption_query_fixed_space|{roster_record['dataset']}_{roster_record['query_key_sha256']}"
    assert fit["seed_key"] == expected_key
    assert fit["declared_steps"] == cfg_expected["steps"]
    assert len(fit["path"]) == cfg_expected["steps"]
    assert len(fit["rounds"]) == cfg_expected["steps"]
    assert fit["initial_state"].keys() == fit["final_state"].keys() == fit["fit_final_state"].keys()
    assert _tree_equal(fit["final_state"], fit["fit_final_state"], torch)
    state = fit["initial_state"]
    assert set(state) == {"spatial.query_residual", "spatial.layers.5.norm1.weight",
                          "spatial.layers.5.norm1.bias", "spatial.layers.5.norm3.weight",
                          "spatial.layers.5.norm3.bias", "spatial.layers.5.norm4.weight",
                          "spatial.layers.5.norm4.bias"}
    assert state["spatial.query_residual"].numel() == 256
    assert sum(state[k].numel() for k in state if k != "spatial.query_residual") == 1536
    assert int(torch.count_nonzero(state["spatial.query_residual"])) == 0
    assert fit["initial_cached_decoder_boxes"].ndim == 2
    assert fit["initial_cached_decoder_boxes"].shape[-1] == 4
    assert fit["final_cached_decoder_boxes"].shape == fit["initial_cached_decoder_boxes"].shape

    positions = [int(a["position"]) for a in anchors]
    references = np.asarray([a["box_cxcywh"] for a in anchors], dtype=np.float64)
    anchor_weights = np.asarray([a["weight"] for a in anchors], dtype=np.float64)
    assert positions == fit["observation_positions"]
    assert np.isfinite(anchor_weights).all() and np.all((anchor_weights > 0.0) & (anchor_weights <= 1.0))
    sigma, tau = float(cfg_expected["sigma"]), float(cfg_expected["tau"])
    beta1, beta2 = 0.9, 0.999
    checks = {"reward_scalars": 0, "loss_scalars": 0, "adam_parameter_groups": 0,
              "state_chain_edges": 0, "rollback_rows": 0, "accepted_rows": 0,
              "no_information_rows": 0}
    errors = {"sample_abs": 0.0, "mean_abs": 0.0, "reward_abs": 0.0,
              "q_abs": 0.0, "per_anchor_loss_abs": 0.0, "loss_abs": 0.0,
              "reference_iou_abs": 0.0, "adam_moment_abs": 0.0}
    last_state = fit["initial_state"]
    last_boxes = fit["initial_cached_decoder_boxes"]
    accepted = proposed_count = rollback_count = no_information = 0
    stopped = False

    for index, row in enumerate(fit["path"]):
        assert row["step"] == index
        assert _tree_equal(row["state_before"], last_state, torch)
        assert torch.equal(row["boxes_before"].cpu(), last_boxes.cpu())
        assert _tree_equal(row["optimizer_before"],
                           fit["path"][index - 1]["optimizer_after"] if index else
                           {"state": {}, "param_groups": row["optimizer_before"]["param_groups"]}, torch)
        assert row["optimizer_before"]["param_groups"][0]["lr"] == cfg_expected["lr"]
        assert tuple(row["optimizer_before"]["param_groups"][0]["betas"]) == (beta1, beta2)
        assert row["optimizer_before"]["param_groups"][0]["eps"] == 1e-8
        assert row["optimizer_before"]["param_groups"][0]["weight_decay"] == 0.0
        checks["adam_parameter_groups"] += 1

        if row.get("skipped"):
            assert not row["proposed"] and not row["accepted"] and not row["rollback"]
            assert _tree_equal(row["state_before"], row["state_after"], torch)
            assert torch.equal(row["boxes_before"].cpu(), row["boxes_after"].cpu())
            assert _tree_equal(row["optimizer_before"], row["optimizer_after"], torch)
            if anchors:
                assert stopped and row["reason"] == "stopped_after_rejected_step"
            else:
                assert row["reason"] == "no_positive_weight_anchors"
            last_state, last_boxes = row["state_after"], row["boxes_after"]
            checks["state_chain_edges"] += 1
            continue

        assert row["validation_enabled"] is True
        rollout = row["rollout"]
        means = rollout["mean"].detach().cpu().numpy().astype(np.float64)
        before = row["boxes_before"].detach().cpu().numpy().astype(np.float64)[positions]
        safe_before = np.clip(before, np.finfo(np.float64).tiny, 1.0 - np.finfo(np.float64).eps)
        expected_mean = np.log(safe_before / (1.0 - safe_before))
        errors["mean_abs"] = max(errors["mean_abs"], _max_abs(means, expected_mean, np))
        assert errors["mean_abs"] < 3e-6

        seed_text = f"spatial-opd-v1|{expected_key}|{index}"
        expected_seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:15], 16)
        assert int(rollout["seed"]) == expected_seed
        gen = torch.Generator(device="cpu").manual_seed(expected_seed)
        first_noise = torch.randn((len(anchors), 16, 4), generator=gen, dtype=torch.float32)
        noise = torch.cat((first_noise, -first_noise), dim=1)
        mean32 = rollout["mean"].detach().cpu().to(dtype=torch.float32)
        expected_samples = mean32[:, None, :] + float(sigma) * noise
        actual_samples = rollout["samples"].detach().cpu()
        sample_error = float((actual_samples - expected_samples).abs().max())
        errors["sample_abs"] = max(errors["sample_abs"], sample_error)
        assert sample_error <= 2e-6
        # The stored 32 actions are antithetic around the stored Gaussian mean.
        anti_error = float((actual_samples[:, :16, :] + actual_samples[:, 16:, :]
                            - 2 * mean32[:, None, :]).abs().max())
        assert anti_error <= 2e-6

        latent = actual_samples.numpy().astype(np.float64)
        action = 1.0 / (1.0 + np.exp(-np.clip(latent, -80.0, 80.0)))
        rewards = _aligned_iou_cxcywh(action, references[:, None, :], np)
        stored_rewards = rollout["rewards"].detach().cpu().numpy().astype(np.float64)
        reward_error = _max_abs(rewards, stored_rewards, np)
        errors["reward_abs"] = max(errors["reward_abs"], reward_error)
        assert reward_error <= 5e-6
        checks["reward_scalars"] += int(rewards.size)
        used = rollout["used_rewards"].detach().cpu().numpy().astype(np.float64)
        assert _max_abs(used, stored_rewards, np) == 0.0
        assert rollout["permutations"] is None
        logits = rewards / tau
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        q = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        saved_q = rollout["weights"].detach().cpu().numpy().astype(np.float64)
        q_error = _max_abs(q, saved_q, np)
        errors["q_abs"] = max(errors["q_abs"], q_error)
        assert q_error <= 5e-6
        assert np.allclose(q.sum(axis=1), 1.0, rtol=0.0, atol=2e-7)
        informative = bool(np.any(used != used[:, :1]))
        assert bool(rollout["informative"]) == informative

        mean64 = means
        stored_mean = rollout["mean"].detach().cpu().numpy().astype(np.float64)
        kl = np.square(mean64 - stored_mean).sum(axis=-1) / (2.0 * sigma * sigma)
        logp = (-np.square(latent - mean64[:, None, :]).sum(axis=-1) /
                (2.0 * sigma * sigma) - 2.0 * math.log(2.0 * math.pi * sigma * sigma))
        per_anchor = kl - ((q - (1.0 / 32.0)) * logp).sum(axis=-1)
        stored_per_anchor = row["per_anchor_loss"].detach().cpu().numpy().astype(np.float64)
        per_error = _max_abs(per_anchor, stored_per_anchor, np)
        errors["per_anchor_loss_abs"] = max(errors["per_anchor_loss_abs"], per_error)
        assert per_error <= 2e-5
        reliability = row["reliability"].detach().cpu().numpy().astype(np.float64)
        assert _max_abs(reliability, anchor_weights, np) <= 1e-7
        expected_loss = float(np.mean(reliability * per_anchor))
        loss_error = abs(expected_loss - float(row["loss"]))
        errors["loss_abs"] = max(errors["loss_abs"], loss_error)
        assert loss_error <= 2e-5
        checks["loss_scalars"] += int(per_anchor.size + 1)

        before_iou = _aligned_iou_cxcywh(before, references, np)
        proposed_boxes = row["boxes_proposed"].detach().cpu().numpy().astype(np.float64)[positions]
        proposed_iou = _aligned_iou_cxcywh(proposed_boxes, references, np)
        before_error = _max_abs(before_iou, row["reference_iou_before"].detach().cpu().numpy(), np)
        proposed_error = _max_abs(proposed_iou, row["reference_iou_proposed"].detach().cpu().numpy(), np)
        errors["reference_iou_abs"] = max(errors["reference_iou_abs"], before_error, proposed_error)
        assert max(before_error, proposed_error) <= 5e-6

        has_proposal = row["update"] is not None
        assert bool(row["proposed"]) == has_proposal
        assert bool(row["accepted"]) == (has_proposal and
               float(np.sum(anchor_weights * (proposed_iou - before_iou)) /
                     np.sum(anchor_weights)) >= -1e-7)
        if not has_proposal:
            assert not informative
            assert _tree_equal(row["state_before"], row["state_after"], torch)
            assert _tree_equal(row["optimizer_before"], row["optimizer_after"], torch)
            no_information += 1
        elif row["accepted"]:
            accepted += 1
            assert _tree_equal(row["state_proposed"], row["state_after"], torch)
            assert _tree_equal(row["optimizer_proposed"], row["optimizer_after"], torch)
            checks["accepted_rows"] += 1
        else:
            rollback_count += 1
            stopped = True
            assert row["rollback"] is True
            assert _tree_equal(row["state_before"], row["state_after"], torch)
            assert _tree_equal(row["optimizer_before"], row["optimizer_after"], torch)
            checks["rollback_rows"] += 1

        if has_proposal:
            proposed_count += 1
            keys = list(row["state_before"])
            grad = row["update"]["gradient"].detach().cpu().reshape(-1)
            raw = row["update"]["raw"].detach().cpu().reshape(-1)
            assert grad.numel() == raw.numel() == 1792
            before_flat = torch.cat([row["state_before"][k].detach().cpu().reshape(-1) for k in keys])
            proposed_flat = torch.cat([row["state_proposed"][k].detach().cpu().reshape(-1) for k in keys])
            assert torch.equal(proposed_flat - before_flat, raw)
            state_before = row["optimizer_before"]["state"]
            state_proposed = row["optimizer_proposed"]["state"]
            for pid, key in enumerate(keys):
                g = grad[pid * 256:(pid + 1) * 256]
                prev = state_before.get(pid)
                if prev is None:
                    m0, v0, step0 = torch.zeros_like(g), torch.zeros_like(g), 0
                else:
                    m0, v0 = prev["exp_avg"].detach().cpu(), prev["exp_avg_sq"].detach().cpu()
                    step0 = int(float(prev["step"]))
                m = m0 * beta1 + g * (1.0 - beta1)
                v = v0 * beta2 + g.square() * (1.0 - beta2)
                actual = state_proposed[pid]
                assert int(float(actual["step"])) == step0 + 1
                moment_error = max(float((m - actual["exp_avg"].detach().cpu()).abs().max()),
                                   float((v - actual["exp_avg_sq"].detach().cpu()).abs().max()))
                errors["adam_moment_abs"] = max(errors["adam_moment_abs"], moment_error)
                assert moment_error <= 2e-6

        last_state, last_boxes = row["state_after"], row["boxes_after"]
        checks["state_chain_edges"] += 1

    assert fit["optimizer_steps"] == accepted
    assert fit["proposed_optimizer_steps"] == proposed_count
    assert fit["rejected_optimizer_steps"] == proposed_count - accepted == rollback_count
    assert fit["final_state"].keys() == last_state.keys()
    assert _tree_equal(last_state, fit["final_state"], torch)
    assert torch.equal(last_boxes.cpu(), fit["final_cached_decoder_boxes"].cpu())
    return {"anchor_count": len(anchors), "declared_steps": len(fit["path"]),
            "proposed_steps": proposed_count, "accepted_steps": accepted,
            "rejected_steps": rollback_count, "no_information_steps": no_information,
            "checks": checks, "max_errors": errors}


def run():
    import numpy as np
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("audit must remain CPU-only")
    torch.set_num_threads(1)
    runtime = _read(BASE / "RUNTIME.json")
    assert runtime["status"] == "locked" and runtime["GT_read"] is False
    design = _read(BASE / "DESIGN_LOCK.json")
    roster_doc = _read(BASE / "ROSTER_LOCK.json")
    ref_doc = _read(BASE / "REFERENCE_LOCK.json")
    barrier = _read(BASE / "STUDENT_GLOBAL_BARRIER.json")
    assert design["status"] == "locked" and design["GT_read"] is False
    assert roster_doc["status"] == "locked" and roster_doc["GT_read"] is False
    assert ref_doc["status"] == "locked" and ref_doc["GT_read"] is False
    assert barrier["status"] == "complete" and barrier["count"] == 32 and barrier["GT_read"] is False
    assert barrier["runtime_sha256"] == _sha(BASE / "RUNTIME.json")
    for path, digest in {**runtime["pins"], **runtime["protected"]}.items():
        p = ROOT / path
        if not p.is_file() or _sha(p) != digest:
            raise AssertionError(f"runtime pin mismatch: {path}")

    roster = roster_doc["records"]
    references = ref_doc["records"]
    items = barrier["files"]
    assert len(roster) == len(references) == len(items) == 32
    aliases = [row["probe_alias"] for row in roster]
    assert len(set(aliases)) == 32
    by_alias = {row["probe_alias"]: row for row in roster}
    refs_by_alias = {row["case_id"]: row for row in references}
    files_by_alias = {Path(row["path"]).stem: row for row in items}
    assert set(by_alias) == set(refs_by_alias) == set(files_by_alias)
    ref_sha = _sha(BASE / "REFERENCE_LOCK.json")
    original_configs = design["original_configs"]
    parent_base = ROOT / "artifacts/stvg_evidence_constrained_v1/round2_prompt009"
    parent_barrier = _read(parent_base / "STUDENT_GLOBAL_BARRIER.json")
    parent_files = {row["path"]: row for row in parent_barrier["files"]}

    checked_cases = []
    totals = {"actual_fits": 0, "anchors": 0, "declared_fit_steps": 0,
              "proposed_optimizer_steps": 0, "accepted_optimizer_steps": 0,
              "rejected_optimizer_steps": 0, "no_information_steps": 0,
              "reward_scalars_recomputed": 0, "loss_scalars_recomputed": 0,
              "state_chain_edges": 0}
    max_errors = {"sample_abs": 0.0, "mean_abs": 0.0, "reward_abs": 0.0,
                  "q_abs": 0.0, "per_anchor_loss_abs": 0.0, "loss_abs": 0.0,
                  "reference_iou_abs": 0.0, "adam_moment_abs": 0.0}
    hashes = []

    for alias in aliases:
        rec, ref, item = by_alias[alias], refs_by_alias[alias], files_by_alias[alias]
        assert ref["dataset"] == rec["dataset"]
        assert ref["GT_read"] is False
        path = ROOT / item["path"]
        assert path.is_file() and path.stat().st_size == item["bytes"] and _sha(path) == item["sha256"]
        receipt_path = path.with_suffix(".receipt.json")
        receipt = _read(receipt_path)
        assert receipt["status"] == "complete" and receipt["GT_read"] is False
        assert receipt["sha256"] == item["sha256"]
        assert receipt["file"]["sha256"] == item["sha256"] and receipt["file"]["bytes"] == item["bytes"]
        assert receipt["reference_lock_sha256"] == ref_sha
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["case_id"] == alias and payload["dataset"] == rec["dataset"]
        assert payload["GT_read"] is False and payload["source_restored"] is True
        assert payload["source_case_sha256"] == ref["source_case_sha256"]
        assert payload["reference_records"] == ref
        assert payload["source_file_sha256"] == _sha(ROOT / "artifacts/stvg_final_table1_v1/source_cases" / f"{alias}.pt")
        parent_path = parent_base / "student" / f"{alias}.pt"
        parent_rel = str(parent_path.relative_to(ROOT))
        assert parent_rel in parent_files and _sha(parent_path) == parent_files[parent_rel]["sha256"]
        assert payload["parent_student_sha256"] == parent_files[parent_rel]["sha256"]

        fit = payload["fit"]
        anchors = ref["anchors"]
        assert item["actual_fit_count"] == int(bool(anchors))
        if anchors:
            if fit is None:
                raise AssertionError(f"missing fit for anchored source {alias}")
            detail = _fit_audit(fit, anchors, rec, original_configs[rec["dataset"]],
                                ref_sha, torch, np)
            totals["actual_fits"] += 1
            totals["anchors"] += detail["anchor_count"]
            for key, target in (("declared_steps", "declared_fit_steps"),
                                ("proposed_steps", "proposed_optimizer_steps"),
                                ("accepted_steps", "accepted_optimizer_steps"),
                                ("rejected_steps", "rejected_optimizer_steps"),
                                ("no_information_steps", "no_information_steps")):
                totals[target] += detail[key]
            totals["reward_scalars_recomputed"] += detail["checks"]["reward_scalars"]
            totals["loss_scalars_recomputed"] += detail["checks"]["loss_scalars"]
            totals["state_chain_edges"] += detail["checks"]["state_chain_edges"]
            for key, value in detail["max_errors"].items():
                max_errors[key] = max(max_errors[key], value)

            # Independently connect source/cached/final decoder outputs.
            source_xy = np.asarray(payload["arms"]["F"]["boxes"], dtype=np.float64)
            initial_xy = _cxcywh_to_xyxy(fit["initial_cached_decoder_boxes"].numpy(), np)
            assert source_xy.shape == initial_xy.shape
            initial_error = _max_abs(source_xy, initial_xy, np)
            assert initial_error == 0.0
            assert torch.equal(payload["readout"]["boxes"].cpu(), fit["final_cached_decoder_boxes"].cpu())
            final_xy = _cxcywh_to_xyxy(fit["final_cached_decoder_boxes"].numpy(), np)
            assert np.array_equal(payload["arms"]["S_original"]["boxes"], final_xy)
            assert np.array_equal(payload["arms"]["TS_original"]["boxes"], final_xy)
            assert payload["arms"]["S_original"]["interval"] == payload["arms"]["F"]["interval"]
            assert payload["arms"]["TS_original"]["interval"] == payload["arms"]["T"]["interval"]
            detail["source_initial_boxes_exact"] = True
            detail["full_readout_cached_final_boxes_exact"] = True
        else:
            assert fit is None
            assert np.array_equal(payload["arms"]["S_original"]["boxes"], payload["arms"]["F"]["boxes"])
            assert np.array_equal(payload["arms"]["TS_original"]["boxes"], payload["arms"]["T"]["boxes"])
            detail = {"anchor_count": 0, "actual_fit": False,
                      "source_reuse_exact": True, "optimizer_steps": 0}

        checked_cases.append({"case_id": alias, "dataset": rec["dataset"],
                              "anchor_count": len(anchors), "actual_fit": bool(anchors),
                              "fit_readback": detail})
        hashes.extend([{"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"]},
                       {"path": str(receipt_path.relative_to(ROOT)), "sha256": _sha(receipt_path),
                        "bytes": receipt_path.stat().st_size}])

    assert totals["actual_fits"] == barrier["actual_fit_count"] == 23
    assert totals["accepted_optimizer_steps"] == barrier["accepted_optimizer_steps"] == 239
    assert totals["anchors"] == ref_doc["anchor_count"] == 49
    assert totals["declared_fit_steps"] == sum(
        original_configs[r["dataset"]]["steps"] for r in references if r["anchors"])
    assert totals["proposed_optimizer_steps"] == totals["accepted_optimizer_steps"] + totals["rejected_optimizer_steps"]

    result = {
        "schema": "stvg-reference-recovery-independent-optimizer-readback-v1",
        "status": "pass",
        "scope": "saved original-config fit math/state/Adam/cache readback; no GT, scores, model execution, or independent backward replay",
        "case_count": 32,
        "actual_fit_count": totals["actual_fits"],
        "anchor_count": totals["anchors"],
        "declared_fit_steps": totals["declared_fit_steps"],
        "proposed_optimizer_steps": totals["proposed_optimizer_steps"],
        "accepted_optimizer_steps": totals["accepted_optimizer_steps"],
        "rejected_optimizer_steps": totals["rejected_optimizer_steps"],
        "no_information_steps": totals["no_information_steps"],
        "reward_scalars_recomputed": totals["reward_scalars_recomputed"],
        "loss_scalars_recomputed": totals["loss_scalars_recomputed"],
        "state_chain_edges": totals["state_chain_edges"],
        "max_abs_errors": max_errors,
        "checks": [
            "runtime relevant pins/protected and sealed output file/receipt hashes",
            "locked original HC40/Vid10 configuration, stable key and fixed reference lock",
            "full antithetic Gaussian samples and deterministic per-step seed",
            "independent cxcywh IoU rewards, reward softmax, control-variate loss and validation IoUs",
            "Query reset, 1792-parameter schema, fresh Adam per fit, Adam moments and complete state/box chain",
            "rollback restores both parameters and optimizer state and stops later updates",
            "initial source boxes, final full-readout boxes and cached decoder boxes exact",
            "no-anchor cases preserve exact Source/Temporal source outputs with no fit",
        ],
        "CPU_only": True,
        "GT_read": False,
        "model_loaded": False,
        "GPU_used": False,
        "independent_backward_replay": False,
        "runtime_sha256": _sha(BASE / "RUNTIME.json"),
        "design_lock_sha256": _sha(BASE / "DESIGN_LOCK.json"),
        "roster_lock_sha256": _sha(BASE / "ROSTER_LOCK.json"),
        "reference_lock_sha256": ref_sha,
        "student_barrier_sha256": _sha(BASE / "STUDENT_GLOBAL_BARRIER.json"),
        "measured_files": hashes,
        "cases": checked_cases,
        "time": time.time(),
    }
    out = BASE / "ROOT_INDEPENDENT_OPTIMIZER_READBACK.json"
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({k: v for k, v in result.items() if k not in {"cases", "measured_files"}},
                     ensure_ascii=False, sort_keys=True))
