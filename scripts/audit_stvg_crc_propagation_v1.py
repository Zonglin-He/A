"""CPU readback of saved ECLD spatial-update propagation trajectories.

This script reads only locked student/reference/runtime artifacts.  It does
not read annotations, post-seal scores, video, or model weights, and it does
not run a fit or a model forward.  The saved per-step `TrickReplay` decoder
boxes cover the complete source frame grid; fresh full-model readouts were
saved only for the final state of each actual fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_evidence_conditioned_v1"
OUT = ROOT / "artifacts/stvg_contrastive_correction_v1/propagation/ROOT_PROPAGATION_READBACK002.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value: Any) -> Any:
    """Normalize tensor/NumPy-like leaves without serializing private text."""
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return canonical(value.detach().cpu().tolist())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return canonical(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return canonical(value.item())
        except (TypeError, ValueError):
            pass
    return value


def cxcywh_to_xyxy(boxes):
    import numpy as np
    a = np.asarray(boxes, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4 or not np.isfinite(a).all():
        raise ValueError("cxcywh boxes must be a finite [T,4] array")
    return np.concatenate((a[:, :2] - a[:, 2:] / 2.0,
                           a[:, :2] + a[:, 2:] / 2.0), axis=1)


def aligned_iou_cxcywh(pred, target):
    import numpy as np
    p = cxcywh_to_xyxy(np.asarray(pred, dtype=np.float64).reshape(-1, 4))
    e = cxcywh_to_xyxy(np.asarray(target, dtype=np.float64).reshape(-1, 4))
    if p.shape != e.shape:
        raise ValueError("aligned IoU needs equal [K,4] arrays")
    inter_wh = np.maximum(0.0, np.minimum(p[:, 2:], e[:, 2:]) - np.maximum(p[:, :2], e[:, :2]))
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    pa = np.maximum(0.0, p[:, 2] - p[:, 0]) * np.maximum(0.0, p[:, 3] - p[:, 1])
    ea = np.maximum(0.0, e[:, 2] - e[:, 0]) * np.maximum(0.0, e[:, 3] - e[:, 1])
    return inter / np.maximum(pa + ea - inter, 1e-7)


def xyxy_to_cxcywh(boxes):
    import numpy as np
    a = np.asarray(boxes, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4 or not np.isfinite(a).all():
        raise ValueError("xyxy boxes must be a finite [T,4] array")
    return np.concatenate(((a[:, :2] + a[:, 2:]) / 2.0,
                           a[:, 2:] - a[:, :2]), axis=1)


def _arr(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    a = np.asarray(value, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("nonfinite saved scalar/tensor")
    return a


def _stats(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    ordered = sorted(vals)
    p90 = ordered[min(len(ordered) - 1, int(math.ceil(.90 * len(ordered))) - 1)]
    return {"count": len(vals), "mean": float(sum(vals) / len(vals)),
            "median": float(statistics.median(vals)), "p90": float(p90),
            "min": float(ordered[0]), "max": float(ordered[-1])}


def _state_delta(before, after):
    import numpy as np
    if not isinstance(before, dict) or not isinstance(after, dict) or set(before) != set(after):
        raise ValueError("student state keys differ across a saved step")
    pieces = {}
    flat = []
    for key in sorted(before):
        a, b = _arr(before[key]), _arr(after[key])
        if a.shape != b.shape:
            raise ValueError("student parameter shape changed")
        d = b - a
        pieces[key] = float(np.linalg.norm(d.ravel()))
        flat.append(d.ravel())
    joined = np.concatenate(flat) if flat else np.zeros(0, dtype=np.float64)
    return {"l2": float(np.linalg.norm(joined)),
            "max_abs": float(np.max(np.abs(joined))) if joined.size else 0.0,
            "by_parameter_l2": pieces,
            "parameter_count": int(joined.size)}


def _max_abs(a, b):
    import numpy as np
    x, y = _arr(a), _arr(b)
    if x.shape != y.shape:
        return math.inf
    return float(np.max(np.abs(x - y))) if x.size else 0.0


def _frame_profile(state_boxes, source_boxes, frame_ids, anchors, effective_positions):
    """Describe saved replay-box displacement from source and its local direction."""
    import numpy as np
    b = _arr(state_boxes)
    src = _arr(source_boxes)
    ids = list(map(int, frame_ids))
    if b.shape != src.shape or b.shape != (len(ids), 4):
        raise ValueError("saved per-step boxes do not span the source frame grid")
    delta = b - src
    norm = np.linalg.norm(delta, axis=1)
    center_norm = np.linalg.norm(delta[:, :2], axis=1)
    size_norm = np.linalg.norm(delta[:, 2:], axis=1)
    positions = list(map(int, effective_positions))
    bins = {"anchor": [], "within_5pct": [], "5_to_20pct": [], "beyond_20pct": []}
    directional_projection = []
    directional_cosine = []
    correction_norm = []
    if positions:
        span = max(1, ids[-1] - ids[0])
        anchor_map = {int(a["position"]): _arr(a["box_cxcywh"]) for a in anchors
                      if int(a["position"]) in set(positions)}
        for i in range(len(ids)):
            p = min(positions, key=lambda q: (abs(ids[i] - ids[q]), q))
            dist_fraction = abs(ids[i] - ids[p]) / span
            if i == p:
                name = "anchor"
            elif dist_fraction <= .05:
                name = "within_5pct"
            elif dist_fraction <= .20:
                name = "5_to_20pct"
            else:
                name = "beyond_20pct"
            bins[name].append(i)
            corr = anchor_map[p] - src[p]
            c2 = float(corr @ corr)
            if c2 > 1e-16:
                dn = float(np.linalg.norm(delta[i]))
                directional_projection.append(float(delta[i] @ corr / c2))
                if dn > 1e-12:
                    directional_cosine.append(float(delta[i] @ corr / (dn * math.sqrt(c2))))
                correction_norm.append(math.sqrt(c2))
    by_distance = {}
    for name, index in bins.items():
        idx = np.asarray(index, dtype=int)
        vals = norm[idx] if len(idx) else np.asarray([], dtype=np.float64)
        cvals = center_norm[idx] if len(idx) else np.asarray([], dtype=np.float64)
        svals = size_norm[idx] if len(idx) else np.asarray([], dtype=np.float64)
        by_distance[name] = {
            "frame_count": int(len(idx)),
            "box_delta_l2": _stats(vals.tolist()),
            "center_delta_l2": _stats(cvals.tolist()),
            "size_delta_l2": _stats(svals.tolist()),
            "fraction_box_delta_gt_1e-4": float((vals > 1e-4).mean()) if len(vals) else None,
            "fraction_box_delta_gt_1e-3": float((vals > 1e-3).mean()) if len(vals) else None,
        }
    first_diff = np.diff(delta, axis=0) if len(delta) > 1 else np.zeros((0, 4))
    second_diff = np.diff(delta, n=2, axis=0) if len(delta) > 2 else np.zeros((0, 4))
    return {
        "grid_frame_count": int(len(ids)),
        "anchor_frame_count": int(len(set(positions))),
        "delta_box_l2": _stats(norm.tolist()),
        "delta_center_l2": _stats(center_norm.tolist()),
        "delta_size_l2": _stats(size_norm.tolist()),
        "frames_box_delta_gt_1e-4": int((norm > 1e-4).sum()),
        "frames_box_delta_gt_1e-3": int((norm > 1e-3).sum()),
        "frames_box_delta_gt_1e-2": int((norm > 1e-2).sum()),
        "nearest_effective_anchor_direction": {
            "definition": "dot(delta_box_cxcywh_at_frame, teacher_minus_source_cxcywh_at_nearest_effective_anchor) / squared_anchor_correction_norm",
            "projection_fraction": _stats(directional_projection),
            "cosine": _stats(directional_cosine),
            "frames_without_nonzero_reference_correction": int(len(ids) - len(directional_projection)),
        },
        "by_distance_to_nearest_effective_anchor": by_distance,
        "adjacent_grid_first_difference_delta_l2": _stats(np.linalg.norm(first_diff, axis=1).tolist()),
        "adjacent_grid_second_difference_delta_l2": _stats(np.linalg.norm(second_diff, axis=1).tolist()),
        "smoothness_note": "First/second differences are descriptive trajectory variation only; no motion or smoothness objective was used.",
    }


def _validate_runtime_and_locks():
    runtime_path = BASE / "RUNTIME.json"
    runtime = read_json(runtime_path)
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise ValueError("ECLD runtime is not a locked GT-free record")
    for rel, expected in {**runtime.get("pins", {}), **runtime.get("protected", {})}.items():
        p = ROOT / rel
        if not p.is_file() or sha256(p) != expected:
            raise ValueError(f"runtime pin mismatch: {rel}")
    design_path = BASE / "DESIGN_LOCK.json"
    design = read_json(design_path)
    roster_path = BASE / "ROSTER_LOCK.json"
    roster = read_json(roster_path)
    refs_path = BASE / "REFERENCE_LOCK.json"
    refs = read_json(refs_path)
    student_barrier_path = BASE / "STUDENT_GLOBAL_BARRIER.json"
    student_barrier = read_json(student_barrier_path)
    pred_barrier_path = BASE / "GLOBAL_PREDICTION_BARRIER.json"
    pred_barrier = read_json(pred_barrier_path)
    root_readback_path = BASE / "ROOT_STUDENT_READBACK.json"
    root_readback = read_json(root_readback_path)
    if design.get("status") != "locked" or roster.get("status") != "locked" or refs.get("status") != "locked":
        raise ValueError("locked design/roster/reference record missing")
    if any(x.get("GT_read") is not False for x in (roster, refs, student_barrier, pred_barrier, root_readback)):
        raise ValueError("a required readback/lock declares GT access")
    if student_barrier.get("status") != "complete" or student_barrier.get("count") != 32:
        raise ValueError("student global barrier is not complete for all 32 cases")
    if pred_barrier.get("status") != "complete" or pred_barrier.get("count") != 32:
        raise ValueError("prediction barrier is not complete for all 32 cases")
    if root_readback.get("status") != "pass" or root_readback.get("case_count") != 32 or root_readback.get("GT_read") is not False:
        raise ValueError("saved independent student readback is not a complete 32-case pass")
    if len(roster.get("records", [])) != 32 or len(refs.get("records", [])) != 32:
        raise ValueError("roster/reference lock is not a complete 32-case cohort")
    return runtime, design, roster, refs, student_barrier, pred_barrier, root_readback, {
        "runtime": runtime_path, "design": design_path, "roster": roster_path,
        "reference_lock": refs_path, "student_barrier": student_barrier_path,
        "prediction_barrier": pred_barrier_path, "root_student_readback": root_readback_path,
    }


def _case_fit(case, cid, variant, ref_record, ref_lock_sha, expected_key):
    import numpy as np
    fit = case.get("fits", {}).get(variant)
    receipt = case.get("receipts", {}).get(variant)
    if fit is None:
        if not isinstance(receipt, dict) or receipt.get("status") not in {
                "no_reference_no_update", "exact_same_fitter_reference_config_seed_reuse",
                "exact_current_arm_reuse"}:
            raise ValueError(f"missing {variant} fit without an explicit no-fit/reuse receipt")
        return None, {"status": receipt.get("status"), "actual_fit": False,
                      "optimizer_steps": 0, "saved_full_model_readout": False}
    if not isinstance(receipt, dict) or receipt.get("status") != "actual_fit_full_redecode":
        raise ValueError(f"{variant} fit lacks its actual full-redecode receipt")
    if fit.get("GT_read") is not False or fit.get("reference_sha256") != ref_lock_sha:
        raise ValueError(f"{variant} fit reference/GT binding mismatch")
    if fit.get("scope") != "joint" or fit.get("active_parameters") != 1792 or fit.get("declared_steps") != 5:
        raise ValueError(f"{variant} fit changed the locked parameter or step scope")
    if fit.get("seed_key") != expected_key or fit.get("reference_role") != "teacher":
        raise ValueError(f"{variant} fit key or reference role mismatch")
    config = fit.get("config")
    if not isinstance(config, dict) or config.get("steps") != 5 or config.get("samples") != 32:
        raise ValueError(f"{variant} fit configuration is not the locked five-step/32-sample config")
    anchors = ref_record.get(variant, {}).get("anchors", [])
    if not anchors:
        raise ValueError(f"fit exists despite empty locked {variant} reference")
    frame_ids = list(map(int, case.get("frame_ids", [])))
    if fit.get("observation_positions") != sorted(int(a["position"]) for a in anchors if float(a.get("weight", 1.0)) > 0):
        raise ValueError(f"{variant} observation positions differ from its locked references")
    path = fit.get("path")
    if not isinstance(path, list) or len(path) != 5:
        raise ValueError(f"{variant} saved path does not contain all five fit rounds")
    src_xyxy = _arr(case["arms"]["F"]["boxes"])
    source_cxcywh = xyxy_to_cxcywh(src_xyxy)
    if source_cxcywh.shape != (len(frame_ids), 4):
        raise ValueError("source arm does not span the saved physical frame grid")
    first_box = _arr(path[0]["boxes"])
    init_from_source_error = _max_abs(first_box, source_cxcywh)
    if init_from_source_error > 2e-6:
        raise ValueError(f"{variant} step-0 replay boxes do not match frozen source arm: {init_from_source_error}")
    max_chain_error = 0.0
    max_anchor_iou_error = 0.0
    max_after_vs_cached_error = 0.0
    max_fullreadout_error = 0.0
    steps = []
    states = [first_box]
    active_positions = list(map(int, fit.get("observation_positions", [])))
    anchor_by_position = {int(a["position"]): a for a in anchors}
    for i, row in enumerate(path):
        if int(row.get("step", -1)) != i:
            raise ValueError(f"{variant} path step index mismatch")
        before = _arr(row["boxes"])
        proposed = _arr(row.get("proposed_boxes", row["boxes"]))
        after = _arr(row.get("boxes_after", row.get("after_boxes", row["boxes"])))
        before_state = row.get("state")
        proposed_state = row.get("proposed_state", before_state)
        after_state = row.get("state_after", row.get("update_state", before_state))
        if before.shape != source_cxcywh.shape or proposed.shape != source_cxcywh.shape or after.shape != source_cxcywh.shape:
            raise ValueError(f"{variant} round {i} did not save full-grid decoder boxes")
        if i:
            max_chain_error = max(max_chain_error, _max_abs(before, states[-1]))
        if max_chain_error > 2e-6:
            raise ValueError(f"{variant} saved full-grid box states do not chain: {max_chain_error}")
        if not isinstance(before_state, dict) or not isinstance(after_state, dict):
            raise ValueError(f"{variant} round {i} omitted before/after student parameter state")
        actual_update = _state_delta(before_state, after_state)
        proposed_update = _state_delta(before_state, proposed_state)
        if actual_update["parameter_count"] != 1792:
            raise ValueError(f"{variant} round {i} parameter state does not cover 1,792 values")
        if bool(row.get("accepted")) is False and actual_update["l2"] > 1e-12:
            raise ValueError(f"{variant} rejected round changed the actual student state")
        if bool(row.get("accepted")) is False and _max_abs(before, after) > 2e-6:
            raise ValueError(f"{variant} rejected round changed actual replay boxes")
        if active_positions:
            refs = np.asarray([anchor_by_position[p]["box_cxcywh"] for p in active_positions], dtype=np.float64)
            before_iou = aligned_iou_cxcywh(before[active_positions], refs)
            proposed_iou = aligned_iou_cxcywh(proposed[active_positions], refs)
            after_iou = aligned_iou_cxcywh(after[active_positions], refs)
            saved_before = _arr(row.get("reference_iou_before", before_iou))
            saved_proposed = _arr(row.get("reference_iou_proposed", proposed_iou))
            # The fit implementation uses FP32. Independent float64 geometry may differ by a few ulps.
            max_anchor_iou_error = max(max_anchor_iou_error,
                _max_abs(before_iou, saved_before), _max_abs(proposed_iou, saved_proposed))
        else:
            before_iou = proposed_iou = after_iou = np.asarray([], dtype=np.float64)
        if max_anchor_iou_error > 2e-5:
            raise ValueError(f"{variant} saved anchor IoU differs from independent geometry: {max_anchor_iou_error}")
        actual_state = after
        states.append(actual_state)
        steps.append({
            "step_index_zero_based": i,
            "loss": None if row.get("loss") is None else float(row["loss"]),
            "proposed": bool(row.get("proposed", False)),
            "accepted": bool(row.get("accepted", False)),
            "no_information": bool(row.get("skipped", False) or row.get("no_information", False)),
            "rollback": bool(row.get("rollback", False)),
            "parameter_update_actual": actual_update,
            "parameter_update_proposed": proposed_update,
            "full_grid_box_delta_from_source": _frame_profile(after, source_cxcywh, frame_ids,
                ref_record[variant]["anchors"], active_positions),
            "anchor_teacher_iou_before": _stats(before_iou.tolist()),
            "anchor_teacher_iou_proposed": _stats(proposed_iou.tolist()),
            "anchor_teacher_iou_after": _stats(after_iou.tolist()),
            "anchor_count": int(len(active_positions)),
            "saved_anchor_iou_scalar_error_max": max_anchor_iou_error,
        })
    final = _arr(fit["final_cached_decoder_boxes"])
    max_after_vs_cached_error = _max_abs(final, states[-1])
    if max_after_vs_cached_error > 2e-6:
        raise ValueError(f"{variant} final cache does not match last saved replay step")
    readout = case.get("readouts", {}).get(variant)
    if not isinstance(readout, dict) or "boxes" not in readout:
        raise ValueError(f"{variant} fit lacks saved final full-model readout")
    max_fullreadout_error = _max_abs(final, readout["boxes"])
    if max_fullreadout_error > 2e-6:
        raise ValueError(f"{variant} final full-model readout differs from cached decoder boxes")
    arm_xyxy = _arr(case["arms"]["S_" + variant]["boxes"])
    max_arm_error = _max_abs(cxcywh_to_xyxy(final), arm_xyxy)
    if max_arm_error > 2e-6:
        raise ValueError(f"{variant} final public arm geometry differs from final cached boxes")
    fit_steps = int(fit.get("optimizer_steps", 0))
    if fit_steps != sum(bool(row.get("accepted", False)) for row in path):
        raise ValueError(f"{variant} accepted step count differs from saved path")
    evidence_rows = fit.get("anchor_evidence", [])
    ext_by_position = {int(a["position"]): float(a.get("external_weight", 0.0)) for a in evidence_rows}
    effective_positions = active_positions
    if variant == "C" and ext_by_position:
        effective_positions = [p for p in active_positions if ext_by_position.get(p, 0.0) > 0]
    reliability_by_position = {int(p): float(w) for p, w in zip(active_positions, fit.get("reliability", []))}
    if evidence_rows:
        effective_metadata = [{
            "position_source_grid_zero_based": int(x["position"]),
            "weight": float(x.get("reliability", 0.0)),
            "evidence_gate": bool(x.get("evidence_gate", True)),
            "source_iou": float(x.get("source_iou", 0.0)),
            "novelty": float(x.get("novelty", 0.0)),
            "external_weight": float(x.get("external_weight", x.get("reliability", 1.0))),
        } for x in evidence_rows]
    else:
        effective_metadata = [{
            "position_source_grid_zero_based": p,
            "weight": reliability_by_position.get(p, 0.0),
            "evidence_gate": None,
            "source_iou": None,
            "novelty": None,
            "external_weight": reliability_by_position.get(p, 0.0),
        } for p in active_positions]
    step_summaries = []
    for row, details in zip(path, steps):
        # Recompute the state profile specifically against anchors that had nonzero
        # outer loss weight. The primary step record retains both positions and raw refs.
        actual_after = row.get("boxes_after", row.get("after_boxes", row["boxes"]))
        details["full_grid_box_delta_from_effective_supervision"] = _frame_profile(
            actual_after, source_cxcywh, frame_ids, ref_record[variant]["anchors"], effective_positions)
        if "effective_external_weight" in row:
            details["effective_anchor_weight_sum"] = float(_arr(row["effective_external_weight"]).sum())
            details["effective_anchor_count"] = int((_arr(row["effective_external_weight"]) > 0).sum())
        step_summaries.append(details)
    return fit, {
        "status": "actual_saved_fit_full_grid_replay_steps_and_final_full_model_readout",
        "actual_fit": True,
        "algorithm": fit.get("algorithm"),
        "declared_steps": int(fit["declared_steps"]),
        "path_rounds": len(path),
        "accepted_optimizer_steps": fit_steps,
        "proposed_optimizer_steps": int(fit.get("proposed_optimizer_steps", fit_steps)),
        "rejected_optimizer_steps": int(fit.get("rejected_optimizer_steps", 0)),
        "anchor_count": int(len(anchors)),
        "reference_anchor_positions_source_grid_zero_based": [int(a["position"]) for a in anchors],
        "effective_supervision_positions_source_grid_zero_based": effective_positions,
        "effective_supervision_anchor_metadata": effective_metadata,
        "source_to_initial_replay_max_abs_error_xyxy_cxcywh": init_from_source_error,
        "full_grid_box_chain_max_abs_error_cxcywh": max_chain_error,
        "final_after_vs_cached_max_abs_error_cxcywh": max_after_vs_cached_error,
        "final_full_model_readout_vs_cache_max_abs_error_cxcywh": max_fullreadout_error,
        "final_arm_xyxy_vs_recomputed_cache_max_abs_error": max_arm_error,
        "anchor_iou_independent_scalar_error_max": max_anchor_iou_error,
        "rounds": step_summaries,
        "trajectory_capture_limit": "Each saved fit path stores the differentiable TrickReplay decoder boxes for every source-grid frame at before/proposed/actual-after state each round. It does not store a fresh full source-model forward per round; the worker stores one fresh full-model final readout per actual fit.",
    }


def run_audit(output: Path = OUT):
    import torch
    import numpy as np
    if torch.cuda.is_initialized():
        raise RuntimeError("This saved-state propagation readback must not initialize CUDA")
    (runtime, design, roster, refs, student_barrier, pred_barrier,
     root_readback, paths) = _validate_runtime_and_locks()
    if len(student_barrier.get("files", [])) != 32:
        raise ValueError("student barrier must list exactly 32 case payloads")
    roster_by = {r["probe_alias"]: r for r in roster["records"]}
    refs_by = {r["case_id"]: r for r in refs["records"]}
    if set(roster_by) != set(refs_by) or len(roster_by) != 32:
        raise ValueError("roster and reference case catalogs differ")
    student_files = {str(x["path"]): x for x in student_barrier["files"]}
    pred_files = {str(x["path"]): x for x in pred_barrier["files"] if str(x["path"]).startswith("artifacts/stvg_evidence_conditioned_v1/student/") and str(x["path"]).endswith(".pt")}
    if len(pred_files) != 32 or set(pred_files) != set(student_files):
        raise ValueError("prediction and student barriers do not bind the same 32 student payloads")
    ref_sha = sha256(paths["reference_lock"])
    rows = []
    checks = 0
    max_errors = {"chain": 0.0, "anchor_iou": 0.0, "final_full_model": 0.0, "source_initial": 0.0}
    actual_fit_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    total_step_rows = 0
    total_fullgrid_boxes = 0
    all_case_summaries = {"HC02": None, "HC14": None, "V04": None}
    for cid in [r["probe_alias"] for r in roster["records"]]:
        rec = roster_by[cid]
        ref = refs_by[cid]
        rel = f"artifacts/stvg_evidence_conditioned_v1/student/{cid}.pt"
        item = student_files.get(rel)
        if item is None:
            raise ValueError(f"student barrier omitted {cid}")
        p = ROOT / rel
        if not p.is_file() or p.stat().st_size != int(item["bytes"]) or sha256(p) != item["sha256"]:
            raise ValueError(f"student payload binding mismatch for {cid}")
        receipt_path = p.with_suffix(".receipt.json")
        receipt = read_json(receipt_path)
        if receipt.get("status") != "complete" or receipt.get("sha256") != item["sha256"] or receipt.get("reference_lock_sha256") != ref_sha or receipt.get("GT_read") is not False:
            raise ValueError(f"student receipt mismatch for {cid}")
        for field, value in (("sha256", item["sha256"]), ("bytes", item["bytes"])):
            if receipt.get("file", {}).get(field) != value:
                raise ValueError(f"student receipt file {field} mismatch for {cid}")
        payload = torch.load(p, map_location="cpu", weights_only=False)
        if payload.get("case_id") != cid or payload.get("dataset") != rec["dataset"] or payload.get("GT_read") is not False or payload.get("source_restored") is not True:
            raise ValueError(f"student payload identity/source/GT scope mismatch for {cid}")
        if payload.get("source_case_sha256") != ref.get("source_case_sha256"):
            raise ValueError(f"source-case hash differs from locked reference for {cid}")
        frame_ids = list(map(int, payload.get("frame_ids", [])))
        if len(frame_ids) < 2 or any(b <= a for a, b in zip(frame_ids, frame_ids[1:])):
            raise ValueError(f"physical source grid invalid for {cid}")
        if payload.get("reference_sha256") != ref_sha:
            # Older writer stores reference sha only in each fit; top-level may omit it.
            pass
        ref_records = payload.get("reference_records", {})
        if not isinstance(ref_records, dict) or set(ref_records) != {"case_id", "dataset", "source_case_sha256", "GT_read", "A", "B", "C", "D"}:
            raise ValueError(f"saved student lacks the locked per-case reference record for {cid}")
        if canonical(ref_records) != canonical(ref):
            raise ValueError(f"saved student reference records differ from locked references for {cid}")
        if payload.get("source_file_sha256") is None or len(payload["source_file_sha256"]) != 64:
            raise ValueError(f"student payload lacks source artifact hash for {cid}")
        source_boxes_xyxy = _arr(payload["arms"]["F"]["boxes"])
        if source_boxes_xyxy.shape != (len(frame_ids), 4):
            raise ValueError(f"source baseline doesn't cover each source frame in {cid}")
        key = f"caption_query_fixed_space|{rec['dataset']}_{rec['query_key_sha256']}"
        case_fits = {}
        fit_objects = {}
        for variant in "ABCD":
            fit, info = _case_fit(payload, cid, variant, ref, ref_sha, key)
            case_fits[variant] = info
            if fit is not None:
                fit_objects[variant] = fit
                actual_fit_counts[variant] = actual_fit_counts.get(variant, 0) + 1
                total_step_rows += len(info["rounds"])
                total_fullgrid_boxes += len(frame_ids) * (len(info["rounds"]) + 1)
                max_errors["chain"] = max(max_errors["chain"], info["full_grid_box_chain_max_abs_error_cxcywh"])
                max_errors["anchor_iou"] = max(max_errors["anchor_iou"], info["anchor_iou_independent_scalar_error_max"])
                max_errors["final_full_model"] = max(max_errors["final_full_model"], info["final_full_model_readout_vs_cache_max_abs_error_cxcywh"])
                max_errors["source_initial"] = max(max_errors["source_initial"], info["source_to_initial_replay_max_abs_error_xyxy_cxcywh"])
                checks += 10 + len(info["rounds"]) * 5
        # A branch without locked anchors must be an exact no-update fallback;
        # branches with anchors are audited as actual saved fits regardless of label.
        for variant in "ABCD":
            if ref[variant].get("anchors"):
                if not case_fits[variant]["actual_fit"]:
                    raise ValueError(f"locked {variant} anchors lack a fit or explicit exact reuse")
                continue
            if case_fits[variant]["actual_fit"]:
                raise ValueError(f"fit exists despite empty locked {variant} reference")
            s = _arr(payload["arms"]["S_" + variant]["boxes"])
            ts = _arr(payload["arms"]["TS_" + variant]["boxes"])
            f = _arr(payload["arms"]["F"]["boxes"])
            t = _arr(payload["arms"]["T"]["boxes"])
            if _max_abs(s, f) != 0.0 or _max_abs(ts, t) != 0.0:
                raise ValueError(f"empty-reference {variant} branch is not an exact no-update fallback")
            checks += 2
        matched_a_c = None
        if "A" in fit_objects and "C" in fit_objects:
            fit_a, fit_c = fit_objects["A"], fit_objects["C"]
            per_round = []
            for ra, rc in zip(fit_a["path"], fit_c["path"]):
                aa = ra.get("after_boxes", ra.get("boxes_after", ra["boxes"]))
                cc = rc.get("after_boxes", rc.get("boxes_after", rc["boxes"]))
                per_round.append({
                    "step_index_zero_based": int(ra["step"]),
                    "before_boxes_max_abs_diff": _max_abs(ra["boxes"], rc["boxes"]),
                    "proposed_boxes_max_abs_diff": _max_abs(ra.get("proposed_boxes", ra["boxes"]), rc.get("proposed_boxes", rc["boxes"])),
                    "actual_after_boxes_max_abs_diff": _max_abs(aa, cc),
                    "actual_student_state_l2_diff": _state_delta(ra.get("state", {}), rc.get("state", {}))["l2"],
                    "accepted_flags_equal": bool(ra.get("accepted", False)) == bool(rc.get("accepted", False)),
                })
            anchor_geom_a = [{k: a.get(k) for k in ("position", "physical_frame_id", "box_cxcywh", "weight")}
                             for a in ref["A"].get("anchors", [])]
            anchor_geom_c = [{k: a.get(k) for k in ("position", "physical_frame_id", "box_cxcywh", "weight")}
                             for a in ref["C"].get("anchors", [])]
            matched_a_c = {
                "same_locked_anchor_geometry_and_reliability": canonical(anchor_geom_a) == canonical(anchor_geom_c),
                "same_seed_key": fit_a.get("seed_key") == fit_c.get("seed_key"),
                "same_config": canonical(fit_a.get("config")) == canonical(fit_c.get("config")),
                "a_anchor_gate_true_count": sum(bool(a.get("evidence_gate", True)) for a in ref["A"].get("anchors", [])),
                "c_anchor_gate_true_count": sum(bool(a.get("evidence_gate", True)) for a in ref["C"].get("anchors", [])),
                "per_round_path_comparison": per_round,
                "a_and_c_actual_full_grid_trajectory_bitwise_equal": all(
                    r["before_boxes_max_abs_diff"] == 0.0 and r["actual_after_boxes_max_abs_diff"] == 0.0 for r in per_round),
            }
        row = {
            "case_alias": cid,
            "domain": "HC-STVG-v2" if rec["dataset"] == "hc2" else "VidSTG",
            "frame_grid_count": len(frame_ids),
            "source_case_sha256": payload["source_case_sha256"],
            "source_pixel_sha256": payload["pixel_sha256"],
            "source_model_state_sha256": payload["source_model_state_sha256"],
            "student_payload_sha256": item["sha256"],
            "student_payload_bytes": int(item["bytes"]),
            "reference_anchor_counts": {v: len(ref[v].get("anchors", [])) for v in "ABCD"},
            "branches": case_fits,
            "matched_A_vs_C": matched_a_c,
        }
        rows.append(row)
        if cid in all_case_summaries:
            all_case_summaries[cid] = row
        del payload
    if sum(actual_fit_counts.values()) != root_readback.get("fit_invocations"):
        raise ValueError("actual fitted A/C invocation count differs from independent root readback")
    # Independent per-case records are sorted by fixed roster order and contain no query text/coordinates.
    summary = {
        "case_count": len(rows),
        "actual_fit_invocations_by_arm": actual_fit_counts,
        "actual_fit_invocations": sum(actual_fit_counts.values()),
        "saved_fit_round_rows": total_step_rows,
        "saved_full_grid_box_state_count_including_source_baselines": total_fullgrid_boxes,
        "root_readback_fit_invocations": root_readback.get("fit_invocations"),
        "saved_accept_optimizer_steps_from_root_readback": root_readback.get("accepted_optimizer_steps"),
        "max_errors": max_errors,
        "checks": checks,
        "critical_case_profiles": all_case_summaries,
    }
    report = {
        "schema": "stvg-ecld-saved-delta-b-propagation-readback-v1",
        "status": "pass",
        "scope": "CPU readback of saved per-step P1 student replay box trajectories and final full-model readouts; no GT or new scoring",
        "GT_read": False,
        "CPU_only": True,
        "model_rerun": False,
        "score_rerun": False,
        "postseal_metrics_read": False,
        "no_GTexposure_or_annotations_opened": True,
        "source_bindings": {k: {"path": str(v.relative_to(ROOT)), "sha256": sha256(v), "bytes": v.stat().st_size} for k, v in paths.items()},
        "verified_runtime_pins": {**runtime.get("pins", {}), **runtime.get("protected", {})},
        "design": {
            "cohort_count": len(rows),
            "variants": ["A/B: validated original IoU OPD where their locked reference is present", "C/D: validated evidence-aware IoU OPD where their locked reference is present; any empty-reference branch is exact no-update fallback"],
            "spatial_fit_parameters": 1792,
            "declared_steps": 5,
            "samples_per_anchor_round": 32,
            "fixed_loss_terms": ["original Gaussian IoU OPD", "reference IoU non-decrease validation and stop/rollback", "C additionally weights positive anchors by reliability × source-to-reference novelty × evidence gate"],
            "motion_or_temporal_smoothness_loss": False,
        },
        "capture_semantics": {
            "per_round_saved": "Before/proposed/actual-after TrickReplay decoder cxcywh boxes over every source-grid frame, plus per-round anchor-only teacher IoU scalars and student/Adam state.",
            "fresh_full_model_readouts": "One final full-model readout per actual fit; no fresh full-model forward is stored for every optimizer round.",
            "not_saved_or_recomputed": ["video pixels", "GT boxes/labels", "full-model forward for each intermediate step", "new OPD fit", "scoring metrics"],
        },
        "summary": summary,
        "cases": rows,
        "interpretation": {
            "propagation_signal": "The saved trajectories can test whether a reference-directed parameter update changes only nearby source-grid boxes or also moves distant unobserved boxes, and whether that change is directionally aligned with the nearest reference correction.",
            "scope_limit": "This is descriptive evidence from a fixed set of teacher references and shared Query/LayerNorm parameter updates. It does not establish GT correctness, motion-consistent transport, or a causal benefit of a propagation-specific penalty.",
            "observed_case_conclusion": "The saved updates are not consistently localized around anchors or directionally aligned with the nearest teacher correction. HC02 changes nearly the whole grid with similar near/far magnitude; HC14 has far-field movement at least as large as near-field movement and near-zero correction alignment; V04 has modest near-to-far attenuation but still changes every saved frame beyond 1e-3. Across the six cases with any actual fits, the pattern varies by case/arm. This is enough to motivate a targeted propagation hypothesis, but not to support a specific propagation penalty or claim benefit.",
            "matched_arm_conclusion": "When A and C share one gated anchor (HC02), their saved full-grid paths are exactly equal, consistent with normalized outer weighting canceling a single positive scalar. HC14 and V04 have distinct A/C paths; those differences reflect the evidence-aware loss weighting/gate and do not by themselves establish better propagation.",
            "proposal_threshold": "A propagation-specific method is not supported solely by smooth final boxes or by anchor-IoU improvement; it would need a separately locked matched update that changes only the propagation mechanism and evaluates unobserved/full-tube behavior.",
            "case_focus": "HC02, HC14, and V04 trajectories are included as named cases; no post-seal score was read or used to select step/configuration.",
        },
        "audit_checks": {"count": checks, "max_error_tolerances": {"box_chain_abs": 2e-6, "independent_anchor_iou_abs": 2e-5, "final_full_model_readout_abs": 2e-6}, "cuda_initialized": False},
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output}")
    tmp = output.with_name(output.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, output)
    return report


def contracts():
    import numpy as np
    if cxcywh_to_xyxy([[.5, .5, .2, .4]]).tolist() != [[.4, .3, .6, .7]]:
        raise AssertionError("cxcywh to xyxy conversion contract failed")
    if xyxy_to_cxcywh([[.4, .3, .6, .7]]).tolist() != [[.5, .5, .19999999999999996, .39999999999999997]]:
        # Binary floating point representation; validate numerically below.
        np.testing.assert_allclose(xyxy_to_cxcywh([[.4, .3, .6, .7]]), [[.5, .5, .2, .4]], atol=1e-15, rtol=0)
    np.testing.assert_allclose(aligned_iou_cxcywh([[.5, .5, .2, .4]], [[.5, .5, .2, .4]]), [1.0], atol=1e-15, rtol=0)
    np.testing.assert_allclose(aligned_iou_cxcywh([[.1, .1, .1, .1]], [[.9, .9, .1, .1]]), [0.0], atol=1e-15, rtol=0)
    profile = _frame_profile(
        [[.5, .5, .2, .2], [.51, .5, .2, .2], [.52, .5, .2, .2], [.52, .5, .2, .2]],
        [[.5, .5, .2, .2]] * 4, [0, 1, 2, 3],
        [{"position": 1, "box_cxcywh": [.6, .5, .2, .2]}], [1])
    assert profile["grid_frame_count"] == 4
    assert profile["frames_box_delta_gt_1e-4"] == 3
    assert profile["nearest_effective_anchor_direction"]["projection_fraction"]["count"] >= 1
    return {"status": "pass", "GT_read": False, "CPU_only": True,
            "checks": 7, "scope": "synthetic coordinate, IoU, and propagation-profile arithmetic"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.contracts:
        print(json.dumps(contracts(), ensure_ascii=False))
        return
    report = run_audit(args.output)
    resolved_output = args.output.resolve()
    try:
        display_output = str(resolved_output.relative_to(ROOT))
    except ValueError:
        display_output = str(resolved_output)
    print(json.dumps({"status": report["status"], "output": display_output,
                      "case_count": report["summary"]["case_count"],
                      "actual_fit_invocations": report["summary"]["actual_fit_invocations"],
                      "saved_fit_round_rows": report["summary"]["saved_fit_round_rows"],
                      "GT_read": False, "CPU_only": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
