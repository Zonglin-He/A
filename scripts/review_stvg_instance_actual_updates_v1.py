"""Independent CPU readback for the sealed instance-identity student updates.

This reader is intentionally fail-closed: actual artifacts are opened only with
--audit-after-seal and only after STUDENT_GLOBAL_BARRIER.json is complete.
It reads student/reference/source-summary artifacts only; never annotations,
model weights, pixels, or teacher prose beyond sealed numeric reference boxes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / "artifacts/stvg_instance_identity_update_v1"
OLD = ROOT / "artifacts/stvg_native_instance_spatial_opd_v1"
BARRIER_PATH = BASE / "STUDENT_GLOBAL_BARRIER.json"
OUTPUT_PATH = BASE / "NEW/ROOT_ACTUAL_UPDATE_READBACK.json"
FIT_PATH = ROOT / "scripts/stvg_instance_validated_fit_v1.py"
LOCK_PATH = BASE / "P1_UPDATE_LOCK.json"
REFERENCE_PATH = BASE / "REFERENCE_LOCK.json"
STUDENT_RUNTIME_PATH = BASE / "STUDENT_RUNTIME.json"
ATOL = 5e-6
RTOL = 5e-6
GATE_TOLERANCE = 1e-7


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_project_path(relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("barrier path must be a nonempty project-relative path")
    path = Path(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("barrier path escapes the project")
    resolved = (ROOT / path).resolve(strict=True)
    if not resolved.is_relative_to(ROOT.resolve()) or not resolved.is_file():
        raise ValueError("barrier path is outside the project or not a file")
    return resolved


def _tree_equal(left: Any, right: Any) -> bool:
    import numpy as np
    import torch
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right)
                and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and np.array_equal(left, right))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping)
                and set(left) == set(right)
                and all(_tree_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_tree_equal(a, b) for a, b in zip(left, right)))
    return left == right


def _max_abs(left: Any, right: Any) -> float:
    import torch
    a = torch.as_tensor(left, dtype=torch.float64).detach().cpu()
    b = torch.as_tensor(right, dtype=torch.float64).detach().cpu()
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max())


def _allclose(left: Any, right: Any, *, atol: float = ATOL, rtol: float = RTOL) -> bool:
    import torch
    a = torch.as_tensor(left).detach().cpu()
    b = torch.as_tensor(right).detach().cpu()
    return a.shape == b.shape and bool(torch.allclose(a, b, atol=atol, rtol=rtol))


def _xyxy(boxes: Any):
    """Match the locked native readout's float64 cxcywh -> continuous xyxy."""
    import numpy as np
    import torch
    value = torch.as_tensor(boxes).detach().cpu().double().numpy()
    if value.ndim != 2 or value.shape[1] != 4 or not np.isfinite(value).all():
        raise ValueError("box readout must be finite T x 4 cxcywh")
    return np.concatenate((value[:, :2] - value[:, 2:] / 2,
                           value[:, :2] + value[:, 2:] / 2), axis=1)


def _cxcywh_to_xyxy(boxes):
    import torch
    boxes = torch.as_tensor(boxes)
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2), dim=-1)


def _independent_aligned_iou(pred, evidence):
    """Standalone continuous normalized cxcywh IoU, with broadcast anchors."""
    import torch
    p = _cxcywh_to_xyxy(pred)
    e = _cxcywh_to_xyxy(evidence)
    inter_wh = (torch.minimum(p[..., 2:], e[..., 2:])
                - torch.maximum(p[..., :2], e[..., :2])).clamp_min(0)
    inter = inter_wh.prod(-1)
    p_area = (p[..., 2:] - p[..., :2]).clamp_min(0).prod(-1)
    e_area = (e[..., 2:] - e[..., :2]).clamp_min(0).prod(-1)
    union = (p_area + e_area - inter).clamp_min(1e-7)
    return inter / union


def _independent_rollout(mean, evidence, key: str, step: int, config: Mapping[str, Any]):
    """Rebuild the fixed antithetic samples and original IoU softmax on CPU."""
    import torch
    seed = int(hashlib.sha256(f"spatial-opd-v1|{key}|{step}".encode()).hexdigest()[:15], 16)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mean = torch.as_tensor(mean, dtype=torch.float32, device="cpu")
    evidence = torch.as_tensor(evidence, dtype=torch.float32, device="cpu")
    noise = torch.randn(len(mean), 16, 4, generator=gen, dtype=torch.float32)
    noise = torch.cat((noise, -noise), dim=1)
    samples = mean[:, None, :] + float(config["sigma"]) * noise
    rewards = _independent_aligned_iou(torch.sigmoid(samples), evidence[:, None, :]).detach()
    weights = torch.softmax(rewards / float(config["tau"]), dim=-1).detach()
    informative = bool((rewards != rewards[:, :1]).any())
    return {"seed": seed, "mean": mean.detach(), "samples": samples.detach(),
            "rewards": rewards, "used_rewards": rewards, "weights": weights,
            "informative": informative, "permutations": None}


def _independent_per_anchor_loss(mean, policy, sigma: float):
    import torch
    mean = torch.as_tensor(mean, dtype=torch.float32, device="cpu")
    rollout_mean = torch.as_tensor(policy["mean"], dtype=torch.float32, device="cpu")
    samples = torch.as_tensor(policy["samples"], dtype=torch.float32, device="cpu")
    weights = torch.as_tensor(policy["weights"], dtype=torch.float32, device="cpu")
    kl = (mean - rollout_mean).square().sum(-1) / (2 * sigma ** 2)
    logp = (-(samples - mean[:, None, :]).square().sum(-1) / (2 * sigma ** 2)
            - 2 * math.log(2 * math.pi * sigma ** 2))
    return kl - ((weights - 1 / 32) * logp).sum(-1)


def _weighted_reference_delta(before_iou, proposed_iou, reliability) -> float:
    import torch
    before = torch.as_tensor(before_iou, dtype=torch.float64)
    after = torch.as_tensor(proposed_iou, dtype=torch.float64)
    weights = torch.as_tensor(reliability, dtype=torch.float64)
    if before.shape != after.shape or before.shape != weights.shape or not before.numel():
        raise ValueError("validation vectors must be aligned nonempty anchor vectors")
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        return float("nan")
    if not torch.isfinite(weights).all() or bool((weights <= 0).any()):
        raise ValueError("locked center-mask references must have positive finite reliability")
    return float((weights * (after - before)).sum() / weights.sum())


def _anchor_payload(reference_record: Mapping[str, Any], frame_ids: list[int]):
    import torch
    supported = reference_record.get("presence_supported", {})
    rows = supported.get("anchors", [])
    positions = []
    physical_ids = []
    evidence = []
    reliability = []
    for row in rows:
        pos = row.get("position")
        if isinstance(pos, bool) or not isinstance(pos, int) or not 0 <= pos < len(frame_ids):
            raise ValueError("reference position is outside the locked source frame grid")
        physical = row.get("physical_frame_id")
        if isinstance(physical, bool) or not isinstance(physical, int) or physical != frame_ids[pos]:
            raise ValueError("reference physical-frame binding differs from source position")
        rel = row.get("reliability", {}).get("presence_fraction")
        if isinstance(rel, bool) or not isinstance(rel, (int, float)) or not math.isfinite(float(rel)):
            raise ValueError("presence fraction must be a finite scalar")
        if not 0.0 < float(rel) <= 1.0:
            raise ValueError("P1 lock requires strictly positive center-mask presence fraction")
        box = row.get("box_cxcywh")
        value64 = torch.as_tensor(box, dtype=torch.float64)
        if value64.shape != (4,) or not bool(torch.isfinite(value64).all()):
            raise ValueError("reference box must be finite normalized cxcywh")
        positions.append(int(pos))
        physical_ids.append(int(physical))
        evidence.append(value64.to(dtype=torch.float32))
        reliability.append(float(rel))
    if positions != sorted(set(positions)):
        raise ValueError("reference positions must be unique and sorted")
    return positions, physical_ids, (torch.stack(evidence) if evidence else torch.empty((0, 4))), reliability


def _audit_fit(fit: Mapping[str, Any], arm_name: str, case_id: str,
               reference_record: Mapping[str, Any], frame_ids: list[int],
               student_initial: Mapping[str, Any], reference_sha: str):
    import torch
    positions, physical_ids, evidence, expected_reliability = _anchor_payload(reference_record, frame_ids)
    if not positions:
        raise ValueError("_audit_fit called for an empty presence-supported reference")
    if fit.get("observation_positions") != positions:
        raise AssertionError(f"{case_id}/{arm_name}: observation positions differ from locked anchors")
    if fit.get("reliability") != [float(torch.tensor(x, dtype=torch.float32)) for x in expected_reliability]:
        raise AssertionError(f"{case_id}/{arm_name}: stored reliability differs from presence fraction")
    if not _tree_equal(fit.get("initial_state"), student_initial):
        raise AssertionError(f"{case_id}/{arm_name}: fit initial state differs from cold source state")
    if fit.get("reference_sha256") != reference_sha or fit.get("reference_role") != "teacher":
        raise AssertionError(f"{case_id}/{arm_name}: fit reference binding differs")
    if fit.get("seed_key") != f"caption_query_fixed_space|{case_id}":
        raise AssertionError(f"{case_id}/{arm_name}: fit Gaussian seed key differs")
    if fit.get("active_parameters") != 1792 or fit.get("scope") != "joint":
        raise AssertionError(f"{case_id}/{arm_name}: active parameter scope differs")
    config = fit.get("config")
    if not isinstance(config, Mapping) or int(config.get("steps", -1)) != 5 or int(config.get("samples", -1)) != 32:
        raise AssertionError(f"{case_id}/{arm_name}: original five-step/32-sample config differs")
    if fit.get("reward") != "original_iou" or fit.get("loss") != "mean_r_times_original_per_anchor_forward_KL":
        raise AssertionError(f"{case_id}/{arm_name}: reward/loss objective label differs")
    validate = arm_name == "validated_loss_weighted_iou"
    expected_algorithm = "validated_loss_weighted_iou" if validate else "loss_weighted_iou"
    if fit.get("algorithm") != expected_algorithm:
        raise AssertionError(f"{case_id}/{arm_name}: fit algorithm label differs")
    if fit.get("declared_steps") != 5:
        raise AssertionError(f"{case_id}/{arm_name}: declared step count differs")

    path = fit.get("path")
    rounds = fit.get("rounds")
    if not isinstance(path, list) or len(path) != 5 or not isinstance(rounds, list):
        raise AssertionError(f"{case_id}/{arm_name}: expected a five-row stored path")
    expected_key = f"caption_query_fixed_space|{case_id}"
    all_errors: dict[str, float] = {}
    expected_proposed = 0
    expected_accepted = 0
    executed = []
    rejected_seen = False
    round_by_step = {int(row["step"]): row for row in rounds}
    if len(round_by_step) != len(rounds):
        raise AssertionError(f"{case_id}/{arm_name}: duplicate round records")

    for step, entry in enumerate(path):
        if entry.get("step") != step:
            raise AssertionError(f"{case_id}/{arm_name}: path steps are not fixed ordered indices")
        if entry.get("skipped"):
            if not validate or not rejected_seen:
                raise AssertionError(f"{case_id}/{arm_name}: unexpected skipped path row")
            if entry.get("reason") != "stopped_after_rejected_step":
                raise AssertionError(f"{case_id}/{arm_name}: skip reason differs")
            if step in round_by_step:
                raise AssertionError(f"{case_id}/{arm_name}: skipped step has a round record")
            continue
        if rejected_seen:
            raise AssertionError(f"{case_id}/{arm_name}: a proposal occurred after a rejected update")
        executed.append(step)
        if step not in round_by_step:
            raise AssertionError(f"{case_id}/{arm_name}: executed step is missing its round record")
        row = round_by_step[step]
        before_boxes = torch.as_tensor(entry["boxes"], dtype=torch.float32, device="cpu")
        if before_boxes.ndim != 2 or before_boxes.shape[1] != 4 or not bool(torch.isfinite(before_boxes).all()):
            raise ValueError(f"{case_id}/{arm_name}: pre-step boxes are invalid")
        selected = before_boxes[positions]
        if bool(((selected <= 0) | (selected >= 1)).any()):
            raise ValueError(f"{case_id}/{arm_name}: pre-step boxes outside invertible chart")
        mean = torch.logit(selected)
        expected_policy = _independent_rollout(mean, evidence, expected_key, step, config)
        stored_policy = entry.get("rollout")
        if not isinstance(stored_policy, Mapping):
            raise AssertionError(f"{case_id}/{arm_name}: executed step omitted stored rollout")
        for key in ("seed", "informative", "permutations"):
            if stored_policy.get(key) != expected_policy[key]:
                raise AssertionError(f"{case_id}/{arm_name}/step{step}: rollout {key} differs")
        for key in ("mean", "samples", "rewards", "used_rewards", "weights"):
            err = _max_abs(stored_policy[key], expected_policy[key])
            all_errors[f"rollout_{key}"] = max(all_errors.get(f"rollout_{key}", 0.0), err)
            if not _allclose(stored_policy[key], expected_policy[key]):
                raise AssertionError(f"{case_id}/{arm_name}/step{step}: original rollout {key} differs by {err}")
        if not _allclose(stored_policy["used_rewards"], stored_policy["rewards"]):
            raise AssertionError(f"{case_id}/{arm_name}/step{step}: sampled rewards were transformed")
        if not _allclose(entry.get("reliability"), expected_reliability):
            raise AssertionError(f"{case_id}/{arm_name}/step{step}: row reliability differs")

        per_anchor = _independent_per_anchor_loss(mean, expected_policy, float(config["sigma"]))
        if not _allclose(entry.get("anchor_loss"), per_anchor):
            err = _max_abs(entry.get("anchor_loss"), per_anchor)
            raise AssertionError(f"{case_id}/{arm_name}/step{step}: per-anchor loss differs by {err}")
        expected_loss = float((torch.as_tensor(expected_reliability, dtype=torch.float32) * per_anchor).mean())
        loss_error = abs(float(entry["loss"]) - expected_loss)
        all_errors["weighted_loss"] = max(all_errors.get("weighted_loss", 0.0), loss_error)
        if not math.isclose(float(entry["loss"]), expected_loss, rel_tol=RTOL, abs_tol=ATOL):
            raise AssertionError(f"{case_id}/{arm_name}/step{step}: weighted loss differs by {loss_error}")

        before_iou = _independent_aligned_iou(before_boxes[positions], evidence).detach()
        proposed_boxes = torch.as_tensor(entry["proposed_boxes"], dtype=torch.float32, device="cpu")
        proposed_iou = _independent_aligned_iou(proposed_boxes[positions], evidence).detach()
        stored_before_iou = torch.as_tensor(entry["reference_iou_before"], dtype=torch.float32)
        stored_proposed_iou = torch.as_tensor(entry["reference_iou_proposed"], dtype=torch.float32)
        for label, actual, stored in (("gate_before_iou", before_iou, stored_before_iou),
                                      ("gate_proposed_iou", proposed_iou, stored_proposed_iou)):
            err = _max_abs(actual, stored)
            all_errors[label] = max(all_errors.get(label, 0.0), err)
            if not _allclose(actual, stored):
                raise AssertionError(f"{case_id}/{arm_name}/step{step}: {label} differs by {err}")
        delta = _weighted_reference_delta(before_iou, proposed_iou, expected_reliability)
        stored_round = row
        for key, expected in (("reference_mean_before", float((torch.tensor(expected_reliability, dtype=torch.float64) * before_iou.double()).sum() / sum(expected_reliability))),
                              ("reference_mean_proposed", float((torch.tensor(expected_reliability, dtype=torch.float64) * proposed_iou.double()).sum() / sum(expected_reliability)))):
            if not math.isclose(float(stored_round[key]), expected, rel_tol=RTOL, abs_tol=ATOL):
                raise AssertionError(f"{case_id}/{arm_name}/step{step}: round {key} differs")

        updated = entry.get("update") is not None
        if bool(entry.get("proposed")) != updated or bool(row.get("proposed")) != updated:
            raise AssertionError(f"{case_id}/{arm_name}/step{step}: proposed marker differs from optimizer update")
        if bool(row.get("no_information")) != (not expected_policy["informative"]):
            raise AssertionError(f"{case_id}/{arm_name}/step{step}: no-information marker differs")
        step_accepted = updated and (not validate or (math.isfinite(delta) and delta >= -GATE_TOLERANCE))
        if bool(entry.get("accepted")) != step_accepted or bool(row.get("accepted")) != step_accepted:
            raise AssertionError(f"{case_id}/{arm_name}/step{step}: accepted marker differs from validation rule")

        before_state = entry["state"]
        proposed_state = entry["proposed_state"]
        update_state = entry["update_state"]
        opt_before = entry["optimizer_before"]
        opt_proposed = entry["optimizer_proposed"]
        opt_after = entry["optimizer_after"]
        if updated:
            expected_proposed += 1
            if step_accepted:
                expected_accepted += 1
                if not _tree_equal(proposed_state, update_state):
                    raise AssertionError(f"{case_id}/{arm_name}/step{step}: accepted parameter state differs from proposal")
                if not _tree_equal(opt_proposed, opt_after):
                    raise AssertionError(f"{case_id}/{arm_name}/step{step}: accepted Adam state differs from proposal")
                if not _allclose(entry["proposed_boxes"], entry["after_boxes"], atol=0.0, rtol=0.0):
                    raise AssertionError(f"{case_id}/{arm_name}/step{step}: accepted boxes differ from proposal")
            else:
                rejected_seen = True
                if not _tree_equal(before_state, update_state):
                    raise AssertionError(f"{case_id}/{arm_name}/step{step}: rejected parameters were not restored")
                if not _tree_equal(opt_before, opt_after):
                    raise AssertionError(f"{case_id}/{arm_name}/step{step}: rejected Adam state was not restored")
                if _tree_equal(opt_before, opt_proposed):
                    raise AssertionError(f"{case_id}/{arm_name}/step{step}: optimizer proposal was not recorded distinctly")
                if not _tree_equal(entry["boxes"], entry["after_boxes"]):
                    raise AssertionError(f"{case_id}/{arm_name}/step{step}: rejected boxes differ after rollback")
        else:
            if step_accepted:
                raise AssertionError(f"{case_id}/{arm_name}/step{step}: no optimizer update marked accepted")
            if not _tree_equal(before_state, proposed_state) or not _tree_equal(before_state, update_state):
                raise AssertionError(f"{case_id}/{arm_name}/step{step}: no-information step changed parameters")
            if not _tree_equal(opt_before, opt_proposed) or not _tree_equal(opt_before, opt_after):
                raise AssertionError(f"{case_id}/{arm_name}/step{step}: no-information step changed Adam")
            if not _tree_equal(entry["boxes"], entry["proposed_boxes"]) or not _tree_equal(entry["boxes"], entry["after_boxes"]):
                raise AssertionError(f"{case_id}/{arm_name}/step{step}: no-information step changed boxes")

    if set(round_by_step) != set(executed):
        raise AssertionError(f"{case_id}/{arm_name}: round/path step sets differ")
    if int(fit.get("proposed_optimizer_steps", -1)) != expected_proposed:
        raise AssertionError(f"{case_id}/{arm_name}: proposed optimizer count differs")
    if int(fit.get("optimizer_steps", -1)) != expected_accepted:
        raise AssertionError(f"{case_id}/{arm_name}: accepted optimizer count differs")
    rejected = expected_proposed - expected_accepted
    if int(fit.get("rejected_optimizer_steps", -1)) != rejected:
        raise AssertionError(f"{case_id}/{arm_name}: rejected optimizer count differs")
    if rejected != int(rejected_seen):
        raise AssertionError(f"{case_id}/{arm_name}: reject marker count differs")
    if not validate and rejected:
        raise AssertionError(f"{case_id}/{arm_name}: nongated arm has rejected updates")
    if rejected_seen and not all(path[step].get("skipped") for step in range(executed[-1] + 1, 5)):
        raise AssertionError(f"{case_id}/{arm_name}: later declared steps did not stop after rejection")

    tail = path[-1]
    expected_final_state = tail["state"] if tail.get("skipped") else tail["update_state"]
    expected_final_boxes = tail["boxes"] if tail.get("skipped") else tail["after_boxes"]
    if not _tree_equal(fit.get("final_state"), expected_final_state):
        raise AssertionError(f"{case_id}/{arm_name}: final state differs from final path entry")
    if not _tree_equal(fit.get("final_cached_decoder_boxes"), expected_final_boxes):
        raise AssertionError(f"{case_id}/{arm_name}: final cached boxes differ from final path entry")
    return {
        "status": "pass",
        "anchor_count": len(positions),
        "observation_positions": positions,
        "physical_frame_ids": physical_ids,
        "executed_steps": len(executed),
        "declared_steps": 5,
        "proposed_optimizer_steps": expected_proposed,
        "accepted_optimizer_steps": expected_accepted,
        "rejected_optimizer_steps": rejected,
        "stopped_after_rejection": rejected_seen,
        "weighted_mean_iou_delta_min": min(
            [_weighted_reference_delta(
                _independent_aligned_iou(torch.as_tensor(path[s]["boxes"], dtype=torch.float32)[positions], evidence),
                _independent_aligned_iou(torch.as_tensor(path[s]["proposed_boxes"], dtype=torch.float32)[positions], evidence),
                expected_reliability) for s in executed if path[s].get("proposed")]
        ) if any(path[s].get("proposed") for s in executed) else None,
        "max_abs_recompute_errors": all_errors,
    }


def _arm_equal(student: Mapping[str, Any], arm_a: str, arm_b: str) -> bool:
    import numpy as np
    left, right = student["arms"][arm_a], student["arms"][arm_b]
    return (np.array_equal(np.asarray(left["boxes"]), np.asarray(right["boxes"]))
            and list(left["interval"]) == list(right["interval"]))


def _readout_equal_cached(student: Mapping[str, Any], arm: str, fit: Mapping[str, Any], old: Mapping[str, Any]):
    import numpy as np
    import torch
    readout = student["readouts"][arm]
    cached = torch.as_tensor(fit["final_cached_decoder_boxes"]).detach().cpu()
    native = torch.as_tensor(readout["boxes"]).detach().cpu()
    if not torch.equal(native, cached):
        raise AssertionError(f"{student['case_id']}/{arm}: full readout differs from cached decoder boxes")
    expected_xyxy = _xyxy(native)
    if not np.array_equal(expected_xyxy, np.asarray(student["arms"][arm]["boxes"])):
        raise AssertionError(f"{student['case_id']}/{arm}: serialized arm geometry differs from full readout")
    source_logits = student.get("source_logits", old.get("source_logits"))
    if source_logits is not None and not _tree_equal(readout.get("logits"), source_logits):
        raise AssertionError(f"{student['case_id']}/{arm}: temporal source logits changed on full re-decode")
    return {"cached_boxes_equal": True, "xyxy_arm_equal": True,
            "source_logits_equal": source_logits is None or _tree_equal(readout.get("logits"), source_logits)}


def _case_readback(case_id: str, student: Mapping[str, Any], old: Mapping[str, Any],
                   roster_row: Mapping[str, Any], reference_record: Mapping[str, Any],
                   reference_sha: str):
    import numpy as np
    import torch
    if student.get("GT_read") is not False or old.get("GT_read") is not False:
        raise AssertionError(f"{case_id}: a saved student payload reports GT access")
    if student.get("case_id") != case_id or old.get("case_id") != case_id:
        raise AssertionError(f"{case_id}: case identity differs")
    for key in ("source_case_sha256",):
        if student.get(key) != roster_row.get(key) or student.get(key) != old.get(key):
            raise AssertionError(f"{case_id}: {key} differs across roster and source baselines")
    if student.get("reference_sha256") != reference_sha:
        raise AssertionError(f"{case_id}: student reference lock hash differs")
    if student.get("reference_record") != reference_record:
        raise AssertionError(f"{case_id}: saved reference differs from locked reference row")
    if student.get("source_model_state_sha256") != old.get("source_model_state_sha256"):
        raise AssertionError(f"{case_id}: source model state hash differs from frozen prior source")
    if student.get("source_restored") is not True:
        raise AssertionError(f"{case_id}: source restoration assertion is false")
    if list(student.get("frame_ids", [])) != list(old.get("frame_ids", [])):
        raise AssertionError(f"{case_id}: source frame grid differs from frozen baseline")
    if student.get("pixel_sha256") != old.get("pixel_sha256"):
        raise AssertionError(f"{case_id}: source pixel binding differs from frozen baseline")
    if not _tree_equal(student.get("initial_state"), old.get("initial_state")):
        raise AssertionError(f"{case_id}: cold student state differs from frozen prior source")

    frame_ids = list(map(int, student["frame_ids"]))
    source_xyxy = _xyxy(student["source_boxes"])
    if not np.array_equal(source_xyxy, np.asarray(student["arms"]["F"]["boxes"])):
        raise AssertionError(f"{case_id}: source F display differs from stored source boxes")
    if not np.array_equal(source_xyxy, np.asarray(old["arms"]["F"]["boxes"])):
        raise AssertionError(f"{case_id}: source F differs from frozen prior baseline")
    # The new worker schema does not store a separate source_logits field;
    # compare each actual full readout to the old sealed cold-source logits.

    sup_anchors = reference_record.get("presence_supported", {}).get("anchors", [])
    fit_results = {}
    no_update = {}
    for arm_name in ("loss_weighted_iou", "validated_loss_weighted_iou"):
        receipt = student.get("receipts", {}).get(arm_name)
        if not isinstance(receipt, Mapping):
            raise AssertionError(f"{case_id}/{arm_name}: receipt is missing")
        if not sup_anchors:
            if receipt.get("status") != "exact_no_update":
                raise AssertionError(f"{case_id}/{arm_name}: empty reference is not recorded as exact no-update")
            if int(receipt.get("optimizer_steps", -1)) != 0 or int(receipt.get("proposed_optimizer_steps", -1)) != 0 or int(receipt.get("final_redecode_calls", -1)) != 0:
                raise AssertionError(f"{case_id}/{arm_name}: exact no-update has nonzero counters")
            if arm_name in student.get("fits", {}) or arm_name in student.get("readouts", {}):
                raise AssertionError(f"{case_id}/{arm_name}: no-update unexpectedly contains a fit or re-decode")
            if not _arm_equal(student, arm_name, "F") or not _arm_equal(student, f"T_{arm_name}", "T"):
                raise AssertionError(f"{case_id}/{arm_name}: no-update does not reuse exact F/T arms")
            no_update[arm_name] = {"status": "exact_no_update", "optimizer_steps": 0,
                                   "proposed_optimizer_steps": 0, "final_redecode_calls": 0}
            continue
        if receipt.get("status") != "actual_final_redecode":
            raise AssertionError(f"{case_id}/{arm_name}: nonempty reference lacks actual fit+full re-decode")
        fit = student.get("fits", {}).get(arm_name)
        if not isinstance(fit, Mapping):
            raise AssertionError(f"{case_id}/{arm_name}: stored fit is missing")
        fit_results[arm_name] = _audit_fit(fit, arm_name, case_id, reference_record,
                                           frame_ids, student["initial_state"], reference_sha)
        if int(receipt.get("optimizer_steps", -1)) != int(fit["optimizer_steps"]):
            raise AssertionError(f"{case_id}/{arm_name}: receipt accepted-step count differs")
        if int(receipt.get("proposed_optimizer_steps", -1)) != int(fit["proposed_optimizer_steps"]):
            raise AssertionError(f"{case_id}/{arm_name}: receipt proposed-step count differs")
        if int(receipt.get("final_redecode_calls", -1)) != 1:
            raise AssertionError(f"{case_id}/{arm_name}: expected one actual final full-model re-decode")
        fit_results[arm_name]["full_model_readout"] = _readout_equal_cached(student, arm_name, fit, old)
        fit_results[arm_name]["receipt_counters_match"] = True
        fit_results[arm_name]["T_interval_equals_source_T"] = (
            list(student["arms"][f"T_{arm_name}"]["interval"]) == list(student["arms"]["T"]["interval"]))
        if not fit_results[arm_name]["T_interval_equals_source_T"]:
            raise AssertionError(f"{case_id}/{arm_name}: temporal T interval changed in paired spatial readout")

    if set(student.get("fits", {})) != set(student.get("readouts", {})):
        raise AssertionError(f"{case_id}: fit and final-readout arm sets differ")
    # Any saved actual fit (including raw_iou/support controls) must also have
    # a matching full readout equal to the cache; raw-iou step math is left to
    # its original helper's audit as requested.
    for arm_name, fit in student.get("fits", {}).items():
        if arm_name in {"loss_weighted_iou", "validated_loss_weighted_iou"}:
            continue
        _readout_equal_cached(student, arm_name, fit, old)

    return {"case_id": case_id,
            "cohort": roster_row.get("cohort"),
            "source_bindings_match": True,
            "source_reset_flag": True,
            "source_boxes_match_F": True,
            "cold_logits_match": True,
            "full_readout_equals_cached_for_fit_arms": True,
            "loss_fit_arms": fit_results,
            "no_update_arms": no_update}


def run_synthetic_contracts() -> dict[str, Any]:
    """CPU-only formula checks; do not inspect sealed student or annotation data."""
    import torch
    from vg_tta.decota_spatial_opd_tunable_v1 import rollout as original_rollout
    from scripts.stvg_instance_validated_fit_v1 import loss_per_anchor
    from scripts import stvg_target_teacher_student_v1 as core
    config = {"sigma": 0.025, "tau": 0.05, "samples": 32, "steps": 5,
              "lr": 0.03, "writeback": 1 / 16}
    mean = torch.tensor([[0.12, -0.31, 0.18, -0.22], [0.09, 0.27, -0.15, 0.33]], dtype=torch.float32)
    evidence = torch.tensor([[0.52, 0.49, 0.25, 0.24], [0.44, 0.55, 0.29, 0.21]], dtype=torch.float32)
    ours = _independent_rollout(mean, evidence, "caption_query_fixed_space|reader_contract", 2, config)
    reference = original_rollout(mean, evidence, "caption_query_fixed_space|reader_contract", 2, config=config)
    for key in ("seed", "mean", "samples", "rewards", "used_rewards", "weights", "informative"):
        if not _tree_equal(ours[key], reference[key]):
            raise AssertionError(f"synthetic independent rollout differs from frozen original: {key}")
    policy = {"mean": ours["mean"], "samples": ours["samples"], "weights": ours["weights"]}
    per_anchor = _independent_per_anchor_loss(mean, policy, config["sigma"])
    module_loss = loss_per_anchor(mean, reference, config["sigma"])
    if not _tree_equal(per_anchor, module_loss):
        raise AssertionError("synthetic independent per-anchor loss differs from locked module")
    r = torch.tensor([0.2, 0.8], dtype=torch.float32)
    weighted = float((r * per_anchor).mean())
    if not math.isclose(weighted, float((0.2 * per_anchor[0] + 0.8 * per_anchor[1]) / 2), rel_tol=0, abs_tol=0):
        raise AssertionError("synthetic weighted-loss denominator differs")
    # A separate geometric fixture checks the strict threshold and weighted mean.
    before = torch.tensor([0.4, 0.5], dtype=torch.float32)
    accepted_after = torch.tensor([0.5, 0.5], dtype=torch.float32)
    rejected_after = torch.tensor([0.3, 0.5], dtype=torch.float32)
    rel = [0.2, 0.8]
    if _weighted_reference_delta(before, accepted_after, rel) < -GATE_TOLERANCE:
        raise AssertionError("synthetic nondecreasing reference gate was rejected")
    if _weighted_reference_delta(before, rejected_after, rel) >= -GATE_TOLERANCE:
        raise AssertionError("synthetic decreasing reference gate was accepted")
    if torch.cuda.is_initialized():
        raise RuntimeError("synthetic readback contracts must leave CUDA uninitialized")
    return {"status": "pass", "independent_rollout_fields": 7,
            "per_anchor_loss_formula": "pass", "mean_reliability_weighting": "pass",
            "reference_gate_accept_and_reject": "pass", "GT_read": False,
            "student_payloads_read": False, "cuda_initialized": False}


def audit_after_seal() -> dict[str, Any]:
    import numpy as np
    import torch
    from scripts.stvg_instance_identity_common_v1 import read, verify

    if torch.cuda.is_initialized():
        raise RuntimeError("actual student readback must not use initialized CUDA")
    runtime = verify("student")
    if runtime.get("GT_read") is not False:
        raise AssertionError("student runtime is not GT-free")
    barrier = _read_json(BARRIER_PATH)
    if barrier.get("status") != "complete" or barrier.get("count") != 8 or barrier.get("GT_read") is not False:
        raise RuntimeError("actual student readback requires complete eight-case GT-free barrier")
    runtime_sha = _sha(STUDENT_RUNTIME_PATH)
    if barrier.get("runtime_sha256") != runtime_sha:
        raise AssertionError("student barrier is not bound to the current runtime")
    lock = _read_json(LOCK_PATH)
    if lock.get("status") != "locked" or lock.get("GT_read") is not False or lock.get("case_count") != 8:
        raise AssertionError("P1 update lock is not complete and GT-free")
    if lock.get("fit_source_sha256") != _sha(FIT_PATH):
        raise AssertionError("P1 update lock does not bind the reviewed fit source")
    reference_sha = _sha(REFERENCE_PATH)
    references_doc = _read_json(REFERENCE_PATH)
    if references_doc.get("status") != "locked" or references_doc.get("GT_read") is not False:
        raise AssertionError("reference lock is not complete and GT-free")
    reference_by_case = {row["case_id"]: row for row in references_doc.get("records", [])}
    roster = read(OLD / "ROSTER_LOCK.json")["records"]
    roster_by_case = {row["case_id"]: row for row in roster}
    if set(reference_by_case) != set(roster_by_case) or len(roster_by_case) != 8:
        raise AssertionError("reference and source roster case sets differ")

    files = barrier.get("files")
    if not isinstance(files, list) or len(files) != 8:
        raise AssertionError("student barrier must bind exactly eight case receipts")
    receipt_by_case = {}
    for file_row in files:
        cid = file_row.get("case_id")
        if cid in receipt_by_case or cid not in roster_by_case:
            raise AssertionError("duplicate or unexpected student barrier case")
        receipt_path = _safe_project_path(file_row["path"])
        if _sha(receipt_path) != file_row.get("sha256"):
            raise AssertionError(f"{cid}: student receipt differs from barrier SHA")
        receipt = _read_json(receipt_path)
        if receipt.get("status") != "sealed" or receipt.get("GT_read") is not False:
            raise AssertionError(f"{cid}: student receipt is not sealed and GT-free")
        payload_path = _safe_project_path(receipt["path"])
        if _sha(payload_path) != receipt.get("sha256") or payload_path.stat().st_size != receipt.get("bytes"):
            raise AssertionError(f"{cid}: student payload hash/size differs from receipt")
        receipt_by_case[cid] = (receipt, payload_path)
    if set(receipt_by_case) != set(roster_by_case):
        raise AssertionError("student barrier case set differs from frozen roster")

    results = []
    aggregate = {"loss_weighted_iou_actual_fits": 0, "loss_weighted_iou_no_update": 0,
                 "validated_loss_weighted_iou_actual_fits": 0,
                 "validated_loss_weighted_iou_no_update": 0,
                 "proposed_optimizer_steps": 0, "accepted_optimizer_steps": 0,
                 "rejected_optimizer_steps": 0, "fit_step_replays": 0,
                 "full_final_readbacks_checked": 0}
    case_ids = sorted(roster_by_case)
    for cid in case_ids:
        receipt, payload_path = receipt_by_case[cid]
        student = torch.load(payload_path, map_location="cpu", weights_only=False)
        old_path = OLD / "student" / f"{cid}.pt"
        old_receipt_path = old_path.with_suffix(".receipt.json")
        old_receipt = read(old_receipt_path)
        if _sha(old_path) != old_receipt.get("sha256") or old_path.stat().st_size != old_receipt.get("bytes"):
            raise AssertionError(f"{cid}: prior cold-source student baseline differs from its receipt")
        old = torch.load(old_path, map_location="cpu", weights_only=False)
        case = _case_readback(cid, student, old, roster_by_case[cid],
                              reference_by_case[cid], reference_sha)
        for arm_name in ("loss_weighted_iou", "validated_loss_weighted_iou"):
            result = case["loss_fit_arms"].get(arm_name)
            if result is None:
                aggregate[f"{arm_name}_no_update"] += 1
                continue
            aggregate[f"{arm_name}_actual_fits"] += 1
            aggregate["proposed_optimizer_steps"] += result["proposed_optimizer_steps"]
            aggregate["accepted_optimizer_steps"] += result["accepted_optimizer_steps"]
            aggregate["rejected_optimizer_steps"] += result["rejected_optimizer_steps"]
            aggregate["fit_step_replays"] += result["executed_steps"]
            aggregate["full_final_readbacks_checked"] += 1
        results.append(case)
    receipt_totals = {
        "actual_fit_count": sum(int(r.get("actual_fit_count", -1)) for r, _ in receipt_by_case.values()),
        "optimizer_steps": sum(int(r.get("optimizer_steps", -1)) for r, _ in receipt_by_case.values()),
        "proposed_optimizer_steps": sum(int(r.get("proposed_optimizer_steps", -1)) for r, _ in receipt_by_case.values()),
        "final_redecode_calls": sum(int(r.get("final_redecode_calls", -1)) for r, _ in receipt_by_case.values()),
    }
    result = {
        "schema": "stvg-instance-actual-update-readback-v1",
        "status": "pass",
        "count": 8,
        "GT_read": False,
        "model_weights_read": False,
        "GPU_or_CUDA_used": False,
        "cuda_initialized": False,
        "bindings": {
            "student_barrier_sha256": _sha(BARRIER_PATH),
            "student_runtime_sha256": runtime_sha,
            "p1_update_lock_sha256": _sha(LOCK_PATH),
            "reference_lock_sha256": reference_sha,
            "reviewer_sha256": _sha(Path(__file__)),
            "student_worker_sha256": _sha(ROOT / "scripts/run_stvg_instance_identity_student_v1.py"),
            "fit_helper_sha256": _sha(FIT_PATH),
        },
        "aggregate": aggregate,
        "sealed_receipt_totals": receipt_totals,
        "cases": results,
        "boundaries": [
            "Original IoU rollout rewards and softmax weights were recomputed from saved pre-step boxes and locked seed keys on CPU.",
            "Per-anchor forward-KL and mean(r_t * loss) were recomputed from saved Gaussian samples; P1 references require strictly positive center-mask presence fractions.",
            "The validator checks weighted IoU against the same locked teacher reference; it is not semantic identity verification or a generalization guarantee.",
            "Source reset is verified using the sealed cold-state/hash assertions, bitwise source-frame logits, and frame/baseline bindings; no checkpoint weights or model forward were loaded by this reader.",
            "Raw-IoU step mathematics remains covered by its original helper audit; this reader independently audits only the two new loss-weighted arms and checks all actual full readouts against cached boxes.",
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    result["output_sha256"] = _sha(OUTPUT_PATH)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", action="store_true", help="run only synthetic CPU math contracts")
    parser.add_argument("--audit-after-seal", action="store_true", help="read the complete sealed eight-case student barrier once")
    args = parser.parse_args()
    if args.contracts == args.audit_after_seal:
        parser.error("choose exactly one of --contracts or --audit-after-seal")
    if args.contracts:
        print(json.dumps(run_synthetic_contracts(), sort_keys=True))
        return
    if not BARRIER_PATH.is_file():
        raise SystemExit("student barrier is not present; run only after the root confirms sealing")
    result = audit_after_seal()
    print(json.dumps({"status": result["status"], "count": result["count"],
                      "aggregate": result["aggregate"], "output_sha256": result["output_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
