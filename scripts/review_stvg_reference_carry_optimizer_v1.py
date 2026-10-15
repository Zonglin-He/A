"""Independent CPU arithmetic readback for sealed LN-carry OPD fits.

This reader verifies only saved fit/math artifacts.  It does not load GT,
score predictions, instantiate a model, initialize CUDA, or replay backward.
Without ``--all`` it reads only already complete stream barriers and writes a
clearly marked prefix receipt.  ``--all`` requires the complete 96-arrival
prediction seal and all six stream barriers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_reference_recovery_v1/LN_carry"
ORDERS = ("order1", "order2", "order3")
DOMAINS = ("hc2", "vidstg")
STATE_KEYS = (
    "spatial.query_residual",
    "spatial.layers.5.norm1.weight", "spatial.layers.5.norm1.bias",
    "spatial.layers.5.norm3.weight", "spatial.layers.5.norm3.bias",
    "spatial.layers.5.norm4.weight", "spatial.layers.5.norm4.bias",
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _eq(left: Any, right: Any, torch) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right)
                and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape)
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping)
                and set(left) == set(right)
                and all(_eq(left[k], right[k], torch) for k in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_eq(a, b, torch) for a, b in zip(left, right)))
    return left == right


def _max_abs(a, b, np) -> float:
    aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise AssertionError(f"shape mismatch: {aa.shape} vs {bb.shape}")
    return float(np.max(np.abs(aa - bb))) if aa.size else 0.0


def _xyxy(boxes, np):
    x = np.asarray(boxes, dtype=np.float64)
    return np.concatenate((x[..., :2] - x[..., 2:] / 2.0,
                           x[..., :2] + x[..., 2:] / 2.0), axis=-1)


def _iou_cxcywh(a, b, np):
    pa, pb = _xyxy(a, np), _xyxy(b, np)
    lo = np.maximum(pa[..., :2], pb[..., :2])
    hi = np.minimum(pa[..., 2:], pb[..., 2:])
    wh = np.maximum(hi - lo, 0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(pa[..., 2] - pa[..., 0], 0.0) * np.maximum(pa[..., 3] - pa[..., 1], 0.0)
    area_b = np.maximum(pb[..., 2] - pb[..., 0], 0.0) * np.maximum(pb[..., 3] - pb[..., 1], 0.0)
    return inter / np.maximum(area_a + area_b - inter, 1e-7)


def _tensor_flat(state: Mapping[str, Any], torch):
    return torch.cat([state[k].detach().cpu().reshape(-1) for k in STATE_KEYS])


def _audit_fit(fit: Mapping[str, Any], anchors: list[Mapping[str, Any]],
               record: Mapping[str, Any], config: Mapping[str, Any],
               reference_sha: str, torch, np) -> dict[str, Any]:
    """Recompute saved 32-action policy, IoU, q, loss and Adam arithmetic."""
    assert fit["schema"] == "stvg-reference-recovery-fit-v1"
    assert fit["profile"] == "current5" and fit["scope"] == "joint"
    assert fit["dataset"] == record["dataset"]
    assert fit["config"] == config and fit["reference_sha256"] == reference_sha
    assert fit["reference_role"] == "fixed_weight_free_top1_teacher"
    assert fit["GT_read"] is False and fit["validation_enabled"] is True
    assert fit["active_parameters"] == 1792
    assert fit["optimizer_state_reset_each_query"] is True
    assert fit["query_residual_reset_at_arrival"] is True
    key = f"caption_query_fixed_space|{record['dataset']}_{record['query_key_sha256']}"
    assert fit["seed_key"] == key
    assert fit["declared_steps"] == config["steps"] == 5
    assert len(fit["path"]) == len(fit["rounds"]) == 5
    assert tuple(fit["initial_state"]) == STATE_KEYS
    assert tuple(fit["final_state"]) == tuple(fit["fit_final_state"]) == STATE_KEYS
    assert _eq(fit["final_state"], fit["fit_final_state"], torch)
    assert fit["initial_state"][STATE_KEYS[0]].numel() == 256
    assert sum(fit["initial_state"][k].numel() for k in STATE_KEYS[1:]) == 1536
    assert int(torch.count_nonzero(fit["initial_state"][STATE_KEYS[0]])) == 0
    assert tuple(fit["initial_cached_decoder_boxes"].shape) == tuple(fit["final_cached_decoder_boxes"].shape)
    assert fit["initial_cached_decoder_boxes"].ndim == 2
    assert fit["initial_cached_decoder_boxes"].shape[-1] == 4

    positions = [int(a["position"]) for a in anchors]
    refs = np.asarray([a["box_cxcywh"] for a in anchors], dtype=np.float64).reshape((-1, 4))
    weights = np.asarray([a["weight"] for a in anchors], dtype=np.float64)
    assert positions == fit["observation_positions"]
    assert np.isfinite(refs).all() and np.isfinite(weights).all()
    assert np.all((weights > 0.0) & (weights <= 1.0)) if len(weights) else True
    sigma, tau, lr = float(config["sigma"]), float(config["tau"]), float(config["lr"])
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    errors = {k: 0.0 for k in (
        "sample_abs", "mean_abs", "reward_abs", "q_abs", "per_anchor_loss_abs",
        "loss_abs", "reference_iou_abs", "adam_moment_abs", "adam_delta_abs")}
    counts = {k: 0 for k in (
        "reward_scalars", "loss_scalars", "adam_parameter_groups", "state_chain_edges",
        "proposals", "accepted", "rejected", "no_information", "skipped_no_anchor",
        "skipped_after_rejection")}
    last_state = fit["initial_state"]
    last_boxes = fit["initial_cached_decoder_boxes"]
    last_optimizer = None
    stopped = False

    for i, row in enumerate(fit["path"]):
        assert row["step"] == i
        assert _eq(row["state_before"], last_state, torch)
        assert torch.equal(row["boxes_before"].detach().cpu(), last_boxes.detach().cpu())
        opt_before = row["optimizer_before"]
        if last_optimizer is None:
            assert opt_before["state"] == {}
        else:
            assert _eq(opt_before, last_optimizer, torch)
        assert len(opt_before["param_groups"]) == 1
        group = opt_before["param_groups"][0]
        assert float(group["lr"]) == lr
        assert tuple(group["betas"]) == (beta1, beta2)
        assert float(group["eps"]) == eps and float(group["weight_decay"]) == 0.0
        counts["adam_parameter_groups"] += 1

        if row.get("skipped"):
            assert not row["proposed"] and not row["accepted"] and not row["rollback"]
            if anchors:
                assert stopped and row["reason"] == "stopped_after_rejected_step"
                counts["skipped_after_rejection"] += 1
            else:
                assert row["reason"] == "no_positive_weight_anchors"
                counts["skipped_no_anchor"] += 1
            assert _eq(row["state_before"], row["state_proposed"], torch)
            assert _eq(row["state_before"], row["state_after"], torch)
            assert torch.equal(row["boxes_before"], row["boxes_proposed"])
            assert torch.equal(row["boxes_before"], row["boxes_after"])
            assert _eq(opt_before, row["optimizer_proposed"], torch)
            assert _eq(opt_before, row["optimizer_after"], torch)
            round_row = fit["rounds"][i]
            assert round_row["no_information"] is True and round_row["loss"] is None
            last_state, last_boxes, last_optimizer = row["state_after"], row["boxes_after"], row["optimizer_after"]
            counts["state_chain_edges"] += 1
            continue

        if stopped:
            raise AssertionError("fit has a non-skipped row after a rejected step")
        assert anchors and row["validation_enabled"] is True
        rollout = row["rollout"]
        means = rollout["mean"].detach().cpu().numpy().astype(np.float64)
        before_all = row["boxes_before"].detach().cpu().numpy().astype(np.float64)
        assert np.all((before_all[positions] > 0.0) & (before_all[positions] < 1.0))
        before = before_all[positions]
        expected_mean = np.log(before / (1.0 - before))
        err = _max_abs(means, expected_mean, np)
        errors["mean_abs"] = max(errors["mean_abs"], err)
        assert err <= 3e-6

        seed_text = f"spatial-opd-v1|{key}|{i}"
        seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:15], 16)
        assert int(rollout["seed"]) == seed
        gen = torch.Generator(device="cpu").manual_seed(seed)
        half = torch.randn((len(anchors), 16, 4), generator=gen, dtype=torch.float32)
        noise = torch.cat((half, -half), dim=1)
        mean32 = rollout["mean"].detach().cpu().to(dtype=torch.float32)
        expected_samples = mean32[:, None, :] + sigma * noise
        actual_samples = rollout["samples"].detach().cpu()
        sample_err = float((actual_samples - expected_samples).abs().max())
        errors["sample_abs"] = max(errors["sample_abs"], sample_err)
        assert sample_err <= 2e-6
        anti_err = float((actual_samples[:, :16] + actual_samples[:, 16:] -
                          2.0 * mean32[:, None, :]).abs().max())
        assert anti_err <= 2e-6

        latent = actual_samples.numpy().astype(np.float64)
        actions = 1.0 / (1.0 + np.exp(-np.clip(latent, -80.0, 80.0)))
        rewards = _iou_cxcywh(actions, refs[:, None, :], np)
        saved_rewards = rollout["rewards"].detach().cpu().numpy().astype(np.float64)
        reward_err = _max_abs(rewards, saved_rewards, np)
        errors["reward_abs"] = max(errors["reward_abs"], reward_err)
        assert reward_err <= 5e-6
        counts["reward_scalars"] += int(rewards.size)
        used = rollout["used_rewards"].detach().cpu().numpy().astype(np.float64)
        assert _max_abs(used, saved_rewards, np) == 0.0
        assert rollout["permutations"] is None

        logits = used / tau
        logits -= np.max(logits, axis=1, keepdims=True)
        ex = np.exp(logits)
        q = ex / ex.sum(axis=1, keepdims=True)
        saved_q = rollout["weights"].detach().cpu().numpy().astype(np.float64)
        q_err = _max_abs(q, saved_q, np)
        errors["q_abs"] = max(errors["q_abs"], q_err)
        assert q_err <= 5e-6
        assert np.allclose(q.sum(axis=1), 1.0, rtol=0.0, atol=2e-7)
        informative = bool(np.any(used != used[:, :1]))
        assert bool(rollout["informative"]) == informative

        logp = (-np.square(latent - means[:, None, :]).sum(axis=-1) /
                (2.0 * sigma * sigma) - 2.0 * math.log(2.0 * math.pi * sigma * sigma))
        kl = np.square(means - means).sum(axis=-1) / (2.0 * sigma * sigma)
        per_anchor = kl - ((q - (1.0 / 32.0)) * logp).sum(axis=-1)
        saved_per_anchor = row["per_anchor_loss"].detach().cpu().numpy().astype(np.float64)
        per_err = _max_abs(per_anchor, saved_per_anchor, np)
        errors["per_anchor_loss_abs"] = max(errors["per_anchor_loss_abs"], per_err)
        assert per_err <= 2e-5
        reliability = row["reliability"].detach().cpu().numpy().astype(np.float64)
        assert _max_abs(reliability, weights, np) <= 1e-7
        loss = float(np.mean(reliability * per_anchor))
        loss_err = abs(loss - float(row["loss"]))
        errors["loss_abs"] = max(errors["loss_abs"], loss_err)
        assert loss_err <= 2e-5
        counts["loss_scalars"] += int(per_anchor.size + 1)

        before_iou = _iou_cxcywh(before, refs, np)
        proposed_all = row["boxes_proposed"].detach().cpu().numpy().astype(np.float64)
        proposed_iou = _iou_cxcywh(proposed_all[positions], refs, np)
        iou_err = max(_max_abs(before_iou, row["reference_iou_before"].detach().cpu().numpy(), np),
                      _max_abs(proposed_iou, row["reference_iou_proposed"].detach().cpu().numpy(), np))
        errors["reference_iou_abs"] = max(errors["reference_iou_abs"], iou_err)
        assert iou_err <= 5e-6

        update = row["update"]
        has_update = update is not None
        assert bool(row["proposed"]) == has_update
        assert has_update == informative
        accepted = bool(has_update and
                        float(np.sum(weights * (proposed_iou - before_iou)) / np.sum(weights)) >= -1e-7)
        assert bool(row["accepted"]) == accepted
        round_row = fit["rounds"][i]
        assert round_row["proposed"] is has_update and round_row["accepted"] is accepted
        assert round_row["no_information"] is (not informative)
        assert bool(row["rollback"]) is bool(has_update and not accepted)

        if not has_update:
            assert _eq(row["state_before"], row["state_proposed"], torch)
            assert _eq(row["state_before"], row["state_after"], torch)
            assert _eq(opt_before, row["optimizer_proposed"], torch)
            assert _eq(opt_before, row["optimizer_after"], torch)
            counts["no_information"] += 1
        else:
            counts["proposals"] += 1
            grad = update["gradient"].detach().cpu().reshape(-1).to(torch.float64)
            raw = update["raw"].detach().cpu().reshape(-1).to(torch.float64)
            before_flat = _tensor_flat(row["state_before"], torch)
            proposed_flat = _tensor_flat(row["state_proposed"], torch)
            assert grad.numel() == raw.numel() == before_flat.numel() == proposed_flat.numel() == 1792
            raw_err = float((raw - (proposed_flat.to(torch.float64) - before_flat.to(torch.float64))).abs().max())
            # Saved raw is formed in parameter dtype before serialization.
            assert raw_err <= 2e-6
            state_before = opt_before["state"]
            state_proposed = row["optimizer_proposed"]["state"]
            param_ids = list(group["params"])
            assert len(param_ids) == len(STATE_KEYS)
            offset = 0
            adam_delta_err = 0.0
            moment_err = 0.0
            for slot, (pid, name) in enumerate(zip(param_ids, STATE_KEYS)):
                n = row["state_before"][name].numel()
                g = grad[offset:offset+n]
                previous = state_before.get(pid)
                if previous is None:
                    m0 = torch.zeros_like(g)
                    v0 = torch.zeros_like(g)
                    step0 = 0
                else:
                    m0 = previous["exp_avg"].detach().cpu().reshape(-1).to(torch.float64)
                    v0 = previous["exp_avg_sq"].detach().cpu().reshape(-1).to(torch.float64)
                    step0 = int(float(previous["step"]))
                m = beta1 * m0 + (1.0 - beta1) * g
                v = beta2 * v0 + (1.0 - beta2) * g.square()
                actual = state_proposed[pid]
                assert int(float(actual["step"])) == step0 + 1
                am = actual["exp_avg"].detach().cpu().reshape(-1).to(torch.float64)
                av = actual["exp_avg_sq"].detach().cpu().reshape(-1).to(torch.float64)
                moment_err = max(moment_err, float((m-am).abs().max()), float((v-av).abs().max()))
                t = step0 + 1
                denom = v.sqrt() / math.sqrt(1.0 - beta2**t) + eps
                expected_delta = -(lr / (1.0 - beta1**t)) * m / denom
                actual_delta = (row["state_proposed"][name].detach().cpu().reshape(-1).to(torch.float64) -
                                row["state_before"][name].detach().cpu().reshape(-1).to(torch.float64))
                adam_delta_err = max(adam_delta_err, float((expected_delta-actual_delta).abs().max()))
                offset += n
            assert offset == 1792
            errors["adam_moment_abs"] = max(errors["adam_moment_abs"], moment_err)
            errors["adam_delta_abs"] = max(errors["adam_delta_abs"], adam_delta_err)
            assert moment_err <= 2e-6
            assert adam_delta_err <= 2e-5
            if accepted:
                counts["accepted"] += 1
                assert _eq(row["state_proposed"], row["state_after"], torch)
                assert _eq(row["optimizer_proposed"], row["optimizer_after"], torch)
            else:
                counts["rejected"] += 1
                stopped = True
                assert row["rollback"] is True
                assert _eq(row["state_before"], row["state_after"], torch)
                assert _eq(opt_before, row["optimizer_after"], torch)

        last_state, last_boxes = row["state_after"], row["boxes_after"]
        last_optimizer = row["optimizer_after"]
        counts["state_chain_edges"] += 1

    assert fit["optimizer_steps"] == counts["accepted"]
    assert fit["proposed_optimizer_steps"] == counts["proposals"]
    assert fit["rejected_optimizer_steps"] == counts["proposals"] - counts["accepted"]
    assert _eq(last_state, fit["final_state"], torch)
    assert torch.equal(last_boxes.detach().cpu(), fit["final_cached_decoder_boxes"].detach().cpu())
    return {"anchor_count": len(anchors), "fit_invocation": True, "counts": counts,
            "max_abs_errors": errors}


def _verify_runtime(runtime: Mapping[str, Any]) -> None:
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise ValueError("E3 runtime must be locked and declare GT_read=false")
    for rel, expected in {**runtime.get("pins", {}), **runtime.get("protected", {})}.items():
        path = ROOT / rel
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"runtime hash mismatch: {rel}")


def run(*, all_streams: bool = False, output: str | None = None) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU audit refuses an initialized CUDA runtime")
    torch.set_num_threads(1)
    runtime_path = BASE / "RUNTIME.json"
    design_path = BASE / "DESIGN_LOCK.json"
    roster_path = BASE / "ROSTER_LOCK.json"
    refs_path = BASE / "REFERENCE_LOCK.json"
    orders_path = BASE / "ORDERS.json"
    runtime, design = _read(runtime_path), _read(design_path)
    roster_doc, ref_doc, orders_doc = _read(roster_path), _read(refs_path), _read(orders_path)
    for label, doc in (("design", design), ("roster", roster_doc), ("reference lock", ref_doc), ("orders", orders_doc)):
        if doc.get("status") != "locked" or doc.get("GT_read") is not False:
            raise ValueError(f"{label} must be locked and explicitly GT-free")
    _verify_runtime(runtime)
    if int(roster_doc.get("count", -1)) != 32 or len(roster_doc.get("records", [])) != 32:
        raise ValueError("fixed E3 roster must contain 32 sources")
    if int(ref_doc.get("count", -1)) != 32 or len(ref_doc.get("records", [])) != 32:
        raise ValueError("fixed E3 reference lock must contain 32 source records")
    if int(orders_doc.get("count", -1)) != 6 or len(orders_doc.get("records", [])) != 6:
        raise ValueError("E3 order lock must contain the three fixed orders in both domains")

    roster = {r["probe_alias"]: r for r in roster_doc["records"]}
    refs = {r["case_id"]: r for r in ref_doc["records"]}
    assert len(roster) == len(refs) == 32 and set(roster) == set(refs)
    assert int(ref_doc.get("anchor_count", -1)) == sum(len(r["anchors"]) for r in ref_doc["records"]) == 49
    ref_sha = _sha(refs_path)
    runtime_sha = _sha(runtime_path)
    design_sha = _sha(design_path)
    roster_sha = _sha(roster_path)
    orders_sha = _sha(orders_path)

    all_stream_records = {(r["dataset"], r["order"]): r for r in orders_doc["records"]}
    assert set(all_stream_records) == {(d, o) for d in DOMAINS for o in ORDERS}
    student_root = BASE / "student"
    finished = []
    incomplete = []
    for key, entry in all_stream_records.items():
        dataset, order = key
        path = student_root / dataset / order / "STREAM_BARRIER.json"
        if not path.exists():
            incomplete.append({"dataset": dataset, "order": order, "reason": "stream_barrier_absent"})
            continue
        barrier = _read(path)
        if barrier.get("status") != "complete" or barrier.get("GT_read") is not False:
            incomplete.append({"dataset": dataset, "order": order, "reason": "stream_barrier_not_complete"})
            continue
        if barrier.get("dataset") != dataset or barrier.get("order") != order:
            raise AssertionError("stream barrier identity differs from locked order")
        if barrier.get("case_ids") != entry.get("case_ids") or barrier.get("count") != 16:
            raise AssertionError("stream barrier case sequence differs from locked order")
        for field, expected in (("roster_lock_sha256", roster_sha),
                                ("reference_lock_sha256", ref_sha),
                                ("orders_sha256", orders_sha),
                                ("runtime_sha256", runtime_sha)):
            assert barrier.get(field) == expected, f"{dataset}/{order} {field} mismatch"
        files = barrier.get("files")
        if not isinstance(files, list) or len(files) != 16:
            raise AssertionError("complete stream barrier must list 16 output files")
        expected_ids = entry["case_ids"]
        by_path = {x.get("path"): x for x in files}
        expected_paths = {str((student_root / dataset / order / f"{cid}.pt").relative_to(ROOT))
                          for cid in expected_ids}
        assert set(by_path) == expected_paths
        for cid in expected_ids:
            item = by_path[str((student_root / dataset / order / f"{cid}.pt").relative_to(ROOT))]
            case_path = ROOT / item["path"]
            receipt_path = case_path.with_suffix(".receipt.json")
            assert case_path.is_file() and case_path.stat().st_size == int(item["bytes"])
            assert _sha(case_path) == item["sha256"]
            receipt = _read(receipt_path)
            assert receipt.get("status") == "sealed" and receipt.get("GT_read") is False
            assert receipt.get("case_id") == cid and receipt.get("dataset") == dataset and receipt.get("order") == order
            assert receipt.get("path") == item["path"]
            assert receipt.get("sha256") == item["sha256"] and receipt.get("bytes") == item["bytes"]
            assert receipt.get("runtime_sha256") == runtime_sha
            assert receipt.get("reference_lock_sha256") == ref_sha
        finished.append({"dataset": dataset, "order": order, "barrier_path": str(path.relative_to(ROOT)),
                         "barrier_sha256": _sha(path), "case_ids": expected_ids, "files": files})

    if all_streams:
        if incomplete or len(finished) != 6:
            raise RuntimeError("--all requires all six complete stream barriers")
        global_path = BASE / "STUDENT_GLOBAL_BARRIER.json"
        if not global_path.is_file():
            raise RuntimeError("--all requires STUDENT_GLOBAL_BARRIER.json")
        global_barrier = _read(global_path)
        assert global_barrier.get("status") == "complete" and global_barrier.get("count") == 96
        assert global_barrier.get("GT_read") is False
        assert global_barrier.get("runtime_sha256") == runtime_sha

    parent_base = ROOT / "artifacts/stvg_evidence_constrained_v1/round2_prompt009"
    parent_barrier_path = parent_base / "STUDENT_GLOBAL_BARRIER.json"
    parent_barrier = _read(parent_barrier_path)
    assert parent_barrier.get("status") == "complete" and parent_barrier.get("count") == 32
    assert parent_barrier.get("GT_read") is False
    parent_files = {x["path"]: x for x in parent_barrier["files"]}
    parent_barrier_sha = _sha(parent_barrier_path)

    totals = {k: 0 for k in ("arrival_count", "actual_fit_count", "anchor_instances",
                             "declared_fit_steps", "proposed_steps",
                             "accepted_steps", "rejected_steps", "no_information_steps",
                             "skipped_no_anchor_steps", "skipped_after_rejection_steps",
                             "reward_scalars_recomputed",
                             "loss_scalars_recomputed", "state_chain_edges")}
    maxima = {k: 0.0 for k in ("sample_abs", "mean_abs", "reward_abs", "q_abs",
                               "per_anchor_loss_abs", "loss_abs", "reference_iou_abs",
                               "adam_moment_abs", "adam_delta_abs")}
    stream_reports = []
    output_hashes = []
    anchor_by_case = {cid: len(refs[cid]["anchors"]) for cid in refs}
    audited_case_ids: set[str] = set()
    anchored_case_ids: set[str] = set()

    for stream in finished:
        dataset, order = stream["dataset"], stream["order"]
        counts = {k: 0 for k in totals if k != "unique_anchored_sources"}
        case_reports = []
        for pos, cid in enumerate(stream["case_ids"]):
            rec, ref = roster[cid], refs[cid]
            assert rec["dataset"] == ref["dataset"] == dataset
            item = next(x for x in stream["files"] if Path(x["path"]).stem == cid)
            path = ROOT / item["path"]
            receipt_path = path.with_suffix(".receipt.json")
            receipt = _read(receipt_path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            assert payload["schema"] == "stvg-reference-carry-case-v1"
            assert payload["case_id"] == cid and payload["dataset"] == dataset
            assert payload["order"] == order and payload["stream_position"] == pos
            assert payload["GT_read"] is False and payload["source_restored"] is True
            assert payload["reference_lock_sha256"] == ref_sha
            assert payload["runtime_sha256"] == runtime_sha
            assert payload["source_case_sha256"] == ref["source_case_sha256"]
            assert payload["reference_anchors"] == ref["anchors"]
            assert receipt["source_case_sha256"] == ref["source_case_sha256"]

            fit = payload.get("fit")
            assert fit is not None, "current5 E3 invokes the fit adapter even for empty references"
            config = design["configs"][dataset]
            detail = _audit_fit(fit, ref["anchors"], rec, config, ref_sha, torch, np)
            assert _eq(payload["initial_state"], fit["initial_state"], torch)
            transition = payload["state_transition"]
            assert transition["GT_read"] is False and transition["carry_ln"] is True
            assert _eq(transition["incoming_state_before_fit"], fit["initial_state"], torch)
            assert _eq(transition["fit_final_state_before_writeback"], fit["final_state"], torch)
            assert float(transition["writeback_alpha"]) == float(config["writeback"])
            for name in STATE_KEYS:
                if name == STATE_KEYS[0]:
                    expected = torch.zeros_like(fit["initial_state"][name])
                else:
                    expected = fit["initial_state"][name] + float(config["writeback"]) * (
                        fit["final_state"][name] - fit["initial_state"][name])
                assert torch.equal(expected, payload["next_committed"][name])
                assert torch.equal(expected, transition["next_stream_state"][name])
            assert torch.equal(fit["initial_cached_decoder_boxes"], payload["readouts"]["Before_Native"]["boxes"])
            assert torch.equal(fit["final_cached_decoder_boxes"], payload["readouts"]["After_Native"]["boxes"])
            assert payload["readouts"]["full_model_redecode"] is True

            parent_path = parent_base / "student" / f"{cid}.pt"
            parent_rel = str(parent_path.relative_to(ROOT))
            assert parent_rel in parent_files and _sha(parent_path) == parent_files[parent_rel]["sha256"]
            assert payload["temporal_reuse"]["parent_prediction_sha256"] == parent_files[parent_rel]["sha256"]
            parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
            assert np.array_equal(payload["arms"]["Episodic_Native"]["boxes"],
                                  parent_payload["arms"]["S_new_crop"]["boxes"])

            counts["arrival_count"] += 1
            counts["actual_fit_count"] += 1
            counts["anchor_instances"] += detail["anchor_count"]
            counts["declared_fit_steps"] += fit["declared_steps"]
            counts["proposed_steps"] += detail["counts"]["proposals"]
            counts["accepted_steps"] += detail["counts"]["accepted"]
            counts["rejected_steps"] += detail["counts"]["rejected"]
            counts["no_information_steps"] += detail["counts"]["no_information"]
            counts["skipped_no_anchor_steps"] += detail["counts"]["skipped_no_anchor"]
            counts["skipped_after_rejection_steps"] += detail["counts"]["skipped_after_rejection"]
            counts["reward_scalars_recomputed"] += detail["counts"]["reward_scalars"]
            counts["loss_scalars_recomputed"] += detail["counts"]["loss_scalars"]
            counts["state_chain_edges"] += detail["counts"]["state_chain_edges"]
            for key, value in detail["max_abs_errors"].items():
                maxima[key] = max(maxima[key], value)
            case_reports.append({"case_id": cid, "anchor_count": detail["anchor_count"],
                                 "optimizer_steps": fit["optimizer_steps"],
                                 "counts": detail["counts"],
                                 "max_abs_errors": detail["max_abs_errors"],
                                 "student_path": item["path"], "student_sha256": item["sha256"],
                                 "student_bytes": item["bytes"],
                                 "receipt_sha256": _sha(receipt_path), "receipt_bytes": receipt_path.stat().st_size})
            output_hashes.extend([
                {"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"]},
                {"path": str(receipt_path.relative_to(ROOT)), "sha256": _sha(receipt_path),
                 "bytes": receipt_path.stat().st_size},
            ])
            audited_case_ids.add(cid)
            if anchor_by_case[cid] > 0:
                anchored_case_ids.add(cid)
        for key, value in counts.items():
            totals[key] += value
        stream_reports.append({"dataset": dataset, "order": order, "arrival_count": counts["arrival_count"],
                               "anchor_instances": counts["anchor_instances"],
                               "anchored_case_count": sum(anchor_by_case[cid] > 0 for cid in stream["case_ids"]),
                               "actual_fit_count": counts["actual_fit_count"], "counts": counts,
                               "barrier_path": stream["barrier_path"], "barrier_sha256": stream["barrier_sha256"],
                               "cases": case_reports})
        assert counts["arrival_count"] == 16
        assert counts["anchor_instances"] == sum(anchor_by_case[cid] for cid in stream["case_ids"])

    expected_arrivals = 96 if all_streams else len(finished) * 16
    assert totals["arrival_count"] == expected_arrivals
    unique_anchors_audited = sum(anchor_by_case[cid] for cid in audited_case_ids)
    if all_streams:
        assert totals["arrival_count"] == 96
        assert totals["anchor_instances"] == 49 * 3 == 147
        assert unique_anchors_audited == 49
        assert len(anchored_case_ids) == 23

    result = {
        "schema": "stvg-reference-carry-independent-optimizer-math-readback-v1",
        "status": "pass" if all_streams else "prefix_pass",
        "scope": "saved E3 current5 sampled reward/q/loss/Adam arithmetic and per-fit carry writeback; state-chain cross-query verification is assigned to a separate reader; no independent backward replay",
        "stage": "all96" if all_streams else "sealed_stream_prefix",
        "case_count": 32,
        "stream_count_expected": 6,
        "stream_count_audited": len(finished),
        "arrival_count_expected": 96,
        "arrival_count_audited": totals["arrival_count"],
        "incomplete_streams": incomplete,
        "cohort_unique_anchor_count": 49,
        "unique_anchor_specs_audited": unique_anchors_audited,
        "unique_anchored_case_count_audited": len(anchored_case_ids),
        "anchored_case_arrivals_audited": sum(
            sum(anchor_by_case[cid] > 0 for cid in s["case_ids"]) for s in finished),
        "anchor_instances_audited": totals["anchor_instances"],
        "actual_fit_count": totals["actual_fit_count"],
        "counts": {k: v for k, v in totals.items() if k != "arrival_count"},
        "max_abs_errors": maxima,
        "CPU_only": True,
        "GT_read": False,
        "model_loaded": False,
        "GPU_used": False,
        "independent_backward_replay": False,
        "runtime_sha256": runtime_sha,
        "design_lock_sha256": design_sha,
        "roster_lock_sha256": roster_sha,
        "reference_lock_sha256": ref_sha,
        "orders_sha256": orders_sha,
        "parent_prediction_barrier_sha256": parent_barrier_sha,
        "auditor_sha256": _sha(Path(__file__)),
        "streams": stream_reports,
        "time": time.time(),
    }
    if output is None:
        out = BASE / ("ROOT_INDEPENDENT_OPTIMIZER_MATH_READBACK.json" if all_streams else
                      "ROOT_INDEPENDENT_OPTIMIZER_MATH_PREFIX.json")
    else:
        out = Path(output)
        if not out.is_absolute():
            out = ROOT / out
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="require and audit all 96 sealed arrivals")
    parser.add_argument("--output", help="optional new receipt path; existing files are never overwritten")
    args = parser.parse_args()
    result = run(all_streams=args.all, output=args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "streams"},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
