"""Build an anonymous report from sealed adjacent-motion readbacks; no scoring or inference."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_adjacent_motion_opd_v1"
OLD_LK_BASE = ROOT / "artifacts/stvg_motion_guided_instance_opd_v1"
ANON = BASE / "anonymous"
DOC = ROOT / "docs/STVG_ADJACENT_MOTION_OPD_V1.md"
PDF = ROOT / "output/pdf/stvg_adjacent_motion_opd_v1.pdf"

CASE_IDS = [f"hc2_{i:03d}" for i in range(4)] + [f"vidstg_{i:03d}" for i in range(4)]
ALIASES = {f"hc2_{i:03d}": f"HC{i + 1:02d}" for i in range(4)}
ALIASES.update({f"vidstg_{i:03d}": f"V{i + 1:02d}" for i in range(4)})
ARM_ORDER = (
    "F", "Projection", "T", "teacher_direct", "teacher_iou",
    "teacher_motion_iou", "T_teacher_direct", "T_teacher_iou",
    "T_teacher_motion_iou",
)
ARM_LABELS = {
    "F": "Frozen",
    "Projection": "Projection",
    "T": "Temporal QA (reused)",
    "teacher_direct": "Spatial Direct",
    "teacher_iou": "Spatial IoU-OPD",
    "teacher_motion_iou": "Spatial Motion-OPD",
    "T_teacher_direct": "QA + Direct",
    "T_teacher_iou": "QA + IoU-OPD",
    "T_teacher_motion_iou": "QA + Motion-OPD",
}
TEACHER_KINDS = ("base", "id_swap", "color_swap", "order_swap", "joint_swap", "wrong_query")
CONTROL_FLAGS = {
    "id_swap": "id_swap_same_geometry",
    "color_swap": "color_swap_same_geometry",
    "order_swap": "order_swap_same_geometry",
    "joint_swap": "joint_swap_same_geometry",
}
NEUTRAL_CATEGORIES = ("positive", "negative", "neutral", "unknown", "baseline")
ALGORITHMS = ("direct", "iou", "motion_iou")
FIT_ROUNDS = 5

# These values are provenance, coordinates, media, raw model state, or private text.
DROP_EXACT = {
    "path", "filepath", "file_path", "source_case_path", "source_video_path",
    "physical_frame_id", "physical_frame_ids", "frame_id", "frame_ids",
    "position", "positions", "anchor_positions", "anchor_physical_frame_ids",
    "actual_valid_supervision_anchor_positions",
    "actual_valid_supervision_anchor_physical_frame_ids",
    "original4_observation_positions", "original4_observation_physical_frame_ids",
    "interval_frame_bounds_half_open", "gt_span", "truth", "truths", "gt_boxes",
    "gt_box", "ground_truth_box", "ground_truth_boxes", "gt_coordinates",
    "annotation_coordinates", "prompt", "prompt_text", "query", "query_text",
    "raw_output", "response", "raw_response", "teacher_response", "render_path",
    "video_path", "tensor_path", "image_path", "weights", "optimizer_state",
    "initial_state", "final_state", "source_logits", "logits", "state_dict",
    "parameters", "raw_gradient", "gradient", "rollout", "samples", "source_bindings",
    "candidate_input_binding", "gt_source_bindings", "file_bindings", "seals",
}
DROP_SUBSTRINGS = (
    "sha256", "_path", "filepath", "prompt", "query_prose", "raw_response",
    "teacher_response", "ground_truth_box", "gt_coordinates", "annotation_coordinates",
    "optimizer_state", "state_dict", "source_logits",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _alias(value: Any) -> Any:
    if isinstance(value, str):
        return ALIASES.get(value, value)
    return value


def _drop_key(key: str) -> bool:
    low = key.lower()
    if low in DROP_EXACT:
        return True
    if any(part in low for part in DROP_SUBSTRINGS):
        return True
    if low.endswith("_physical_frame_id") or low.endswith("_physical_frame_ids"):
        return True
    if low in {"dense_step_ids", "tracking_physical_frame_ids", "source_rgb_sha256_by_frame"}:
        return True
    return False


def _anon(value: Any, key: str = "") -> Any:
    """Recursively keep public scalars while dropping private provenance and tensors."""
    if _drop_key(key):
        return None
    if isinstance(value, Mapping):
        result = {}
        for child_key, child in value.items():
            if _drop_key(str(child_key)):
                continue
            cleaned = _anon(child, str(child_key))
            if cleaned is not None or child is None:
                result[str(child_key)] = cleaned
        return result
    if isinstance(value, (list, tuple)):
        return [_anon(item, key) for item in value]
    if isinstance(value, str):
        cleaned = _alias(value)
        if re.search(r"(?:hc2|vidstg)_\d{3}", cleaned):
            for case_id, alias in ALIASES.items():
                cleaned = cleaned.replace(case_id, alias)
        if re.search(r"(?:/home/|/attachments/|(?:^|\s)[A-Za-z]:\\)", cleaned, re.I):
            raise ValueError("private path survived anonymization")
        if re.search(r"\b[0-9a-f]{64}\b", cleaned, re.I):
            raise ValueError("hash-like value survived anonymization")
        return cleaned
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite numeric value in sealed report")
    return value


def _case_alias(case_id: str) -> str:
    _need(case_id in ALIASES, f"case is outside the fixed eight-source scope: {case_id}")
    return ALIASES[case_id]


def _load_inputs() -> dict[str, Any]:
    report = _read(BASE / "ROOT_POSTSEAL_REPORT.json")
    teacher = _read(BASE / "ROOT_TEACHER_READBACK.json")
    barrier = _read(BASE / "CANDIDATE_INPUT_BARRIER.json")
    student_barrier = _read(BASE / "METHOD_STUDENT_GLOBAL_BARRIER.json")
    old_lk_barrier = _read(OLD_LK_BASE / "CANDIDATE_INPUT_BARRIER.json")

    _need(report.get("status") == "pass" and report.get("schema") == "stvg_adjacent_motion_postseal_v1",
          "postseal metric report is not a passing adjacent-motion report")
    _need(report.get("GT_read") is True and report.get("GT_used_for_method_update") is False,
          "postseal report does not state the expected label-use boundary")
    _need(report.get("model_loaded") is False and report.get("GPU_initialized") is False,
          "postseal scoring report unexpectedly records model/GPU use")
    _need(teacher.get("status") == "pass" and teacher.get("GT_read") is False,
          "teacher readback is not a passing GT-free readback")
    _need(barrier.get("status") == "complete" and barrier.get("GT_read") is False,
          "candidate input barrier is not complete and GT-free")
    _need(barrier.get("elapsed_cpu_seconds") is not None and
          math.isfinite(float(barrier["elapsed_cpu_seconds"])),
          "candidate barrier CPU preparation timing is missing")
    _need(student_barrier.get("status") == "complete" and student_barrier.get("GT_read") is False,
          "student barrier is not complete and GT-free")
    _need(old_lk_barrier.get("status") == "complete" and old_lk_barrier.get("GT_read") is False,
          "historical LK candidate barrier is not complete and GT-free")
    _need(report.get("student_barrier_sha256") == _file_sha(BASE / "METHOD_STUDENT_GLOBAL_BARRIER.json"),
          "postseal report no longer binds the student barrier")
    _need(int(student_barrier.get("count", -1)) == 8, "student barrier count differs from the locked eight")
    _need(int(student_barrier.get("actual_final_redecode_calls", -1)) == int(report.get("actual_fit_count", -2)),
          "full final-model readout count differs from actual fit count")

    input_records = barrier.get("records")
    _need(isinstance(input_records, list) and len(input_records) == 8,
          "candidate barrier does not contain the fixed eight cases")
    input_by_id = {str(row.get("case_id")): row for row in input_records}
    _need(len(input_by_id) == 8 and set(input_by_id) == set(CASE_IDS),
          "candidate barrier case IDs differ from the fixed scope")
    for cid, row in input_by_id.items():
        _need(row.get("GT_read") is False, f"{cid}: candidate barrier row is not GT-free")
        candidates = row.get("candidates", row.get("candidate_geometry"))
        _need(isinstance(candidates, list) and candidates,
              f"{cid}: candidate geometry list is missing")
        row["_report_candidates"] = candidates
    _need(sum(len(row["_report_candidates"]) for row in input_by_id.values()) == 43,
          "candidate barrier does not contain the locked 43-candidate inventory")
    _need(sum(bool(candidate.get("is_Source")) for row in input_by_id.values()
               for candidate in row["_report_candidates"]) == 8,
          "candidate barrier does not contain exactly one Source candidate per case")
    old_input_records = old_lk_barrier.get("records")
    _need(isinstance(old_input_records, list), "historical LK barrier records are missing")
    old_by_id = {str(row.get("case_id")): row for row in old_input_records
                 if str(row.get("case_id")) in CASE_IDS}
    _need(len(old_by_id) == 8 and set(old_by_id) == set(CASE_IDS),
          "historical LK barrier does not cover the same fixed eight cases")
    for cid, row in old_by_id.items():
        _need(row.get("GT_read") is False, f"{cid}: historical LK input row is not GT-free")
        _need(isinstance(row.get("candidates"), list), f"{cid}: historical LK candidate list is missing")
    _need(sum(len(row["candidates"]) for row in old_by_id.values()) == 43,
          "historical LK barrier does not contain the matching 43-candidate inventory")
    for cid in CASE_IDS:
        current_keys = {(c.get("candidate_id"), c.get("window_id"), bool(c.get("is_Source")))
                        for c in input_by_id[cid]["_report_candidates"]}
        old_keys = {(c.get("candidate_id"), c.get("window_id"), bool(c.get("is_Source")))
                    for c in old_by_id[cid]["candidates"]}
        _need(current_keys == old_keys,
              f"{cid}: new geometry and historical LK barriers do not have identical candidate identities")

    method_rows = [r for r in report.get("case_rows", []) if r.get("phase") == "method"]
    _need(len(method_rows) == 8 and {r.get("case_id") for r in method_rows} == set(CASE_IDS),
          "case metric rows do not cover exactly the fixed eight cases")
    _need(all(set(row.get("arms", {})) >= set(ARM_ORDER) for row in method_rows),
          "one or more cases are missing a locked F/Projection/T/spatial arm")
    _need(len(report.get("metric_rows", [])) == int(report.get("metric_count", -1)),
          "metric row count does not match the sealed report")
    geometry_rows = report.get("candidate_geometry_rows", [])
    display_rows = report.get("candidate_display_rows", [])
    _need(len(geometry_rows) == int(report.get("candidate_row_count", len(geometry_rows))),
          "candidate geometry row count does not match the sealed report")
    _need(len(display_rows) == len(geometry_rows),
          "candidate display and geometry row inventories differ")
    _need(int(report.get("candidate_row_count", -1)) == 43,
          "postseal candidate geometry rows differ from the locked 43-candidate inventory")
    _need(all(row.get("phase") == "method" for row in geometry_rows + display_rows),
          "candidate rows include an unexpected phase or scope")
    declared_candidate_scope = report.get("candidate_row_scope")
    _need(declared_candidate_scope is None or
          ("method" in str(declared_candidate_scope) and "V03" in str(declared_candidate_scope)),
          "candidate rows include an unexpected declared phase or scope")
    comparisons = report.get("arm_comparison_rows", [])
    comparison_keys = [(row.get("case_id"), row.get("arm")) for row in comparisons]
    _need(len(comparison_keys) == 8 * len(ARM_ORDER) and len(set(comparison_keys)) == len(comparison_keys)
          and set(arm for _, arm in comparison_keys) == set(ARM_ORDER),
          "case comparison rows do not cover the fixed arms once per case")

    teacher_rows = teacher.get("rows", [])
    teacher_cases = teacher.get("cases", [])
    _need(int(teacher.get("count", -1)) == 48 and len(teacher_rows) == 48,
          "teacher readback is not the fixed 48-call matrix")
    _need(len(teacher_cases) == 8 and {r.get("case_id") for r in teacher_cases} == set(CASE_IDS),
          "teacher case summary does not cover the fixed eight cases")
    observed_pairs = {(r.get("case_id"), r.get("kind")) for r in teacher_rows}
    expected_pairs = {(cid, kind) for cid in CASE_IDS for kind in TEACHER_KINDS}
    _need(observed_pairs == expected_pairs and len(observed_pairs) == 48,
          "teacher control matrix has a missing or duplicate condition")
    _need(not any("raw_output" in r or "response" in r or "prompt" in r for r in teacher_rows),
          "teacher readback unexpectedly includes raw prose")

    movement = report.get("fit_movement_rows", [])
    actual_statuses = {"updated_fit", "diagnostic_update_even_if_lock_no_update"}
    actual_fits = [r for r in movement if r.get("status") in actual_statuses]
    nofit = [r for r in movement if r.get("status") == "locked_no_update_exact_F_reuse"]
    nofit_count = int(report.get("no_fit_placeholder_count", len(nofit)))
    optimizer_step_sum = int(report.get("actual_fit_optimizer_step_sum", report.get("optimizer_steps", -1)))
    _need(len(actual_fits) == int(report.get("actual_fit_count", -1)),
          "actual fit rows differ from the sealed fit count")
    _need(len(nofit) == nofit_count,
          "no-fit placeholder count differs from the sealed report")
    expected_fit_rows = {(cid, f"teacher_{algorithm}") for cid in CASE_IDS for algorithm in ALGORITHMS}
    observed_fit_rows = [(r.get("case_id"), r.get("fit")) for r in movement]
    _need(len(observed_fit_rows) == len(expected_fit_rows) and set(observed_fit_rows) == expected_fit_rows,
          "fit-movement readback is not the complete unique eight-case by three-arm matrix")
    _need(sum(int(r.get("optimizer_steps", 0)) for r in actual_fits) == optimizer_step_sum,
          "optimizer-step readback differs from actual fit rows")
    _need(int(student_barrier.get("actual_fit_count", -1)) == len(actual_fits),
          "student barrier fit count differs from postseal fit rows")
    _need(int(student_barrier.get("optimizer_steps", -1)) == optimizer_step_sum,
          "student barrier optimizer steps differ from postseal fit rows")
    motion_rows = report.get("motion_round_rows", [])
    motion_fits = [r for r in actual_fits if str(r.get("fit", "")).endswith("motion_iou")]
    _need(len(motion_rows) == len(motion_fits) * FIT_ROUNDS,
          "motion loss rounds do not cover five declared rounds per actual motion fit")
    _need(all(math.isclose(float(r["lambda_motion"]), 1.0, abs_tol=0.0) for r in motion_rows),
          "motion loss rows differ from the locked lambda_motion=1")
    _need(optimizer_step_sum == int(report.get("optimizer_steps", optimizer_step_sum)),
          "top-level optimizer-step totals disagree")

    v03 = report.get("V03_prior_control")
    _need(isinstance(v03, Mapping) and int(v03.get("new_fits_executed", -1)) == 0,
          "V03 prior-control row is missing or was rerun")
    _need(v03.get("status") == "exact_prior_evidence_reuse",
          "V03 prior control is not marked as exact evidence reuse")

    return {
        "report": report,
        "teacher": teacher,
        "barrier": barrier,
        "student_barrier": student_barrier,
        "input_records": input_by_id,
        "old_lk_input_records": old_by_id,
        "case_rows": method_rows,
        "actual_fits": actual_fits,
        "no_fit_rows": nofit,
        "no_fit_count": nofit_count,
        "optimizer_step_sum": optimizer_step_sum,
        "motion_fits": motion_fits,
    }


def _file_sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_box(box: Any) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    return (all(math.isfinite(v) for v in (x1, y1, x2, y2)) and
            x1 < x2 and y1 < y2 and min(x1, y1, x2, y2) >= -1e-7 and
            max(x1, y1, x2, y2) <= 1 + 1e-7)


def _box(box: Any) -> list[float] | None:
    if box is None:
        return None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        values = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return values


def _support(boxes: Sequence[Any]) -> dict[str, int]:
    present = sum(box is not None for box in boxes)
    valid = sum(_valid_box(box) for box in boxes if box is not None)
    return {"present_slots": int(present), "valid_slots": int(valid), "missing_slots": int(len(boxes) - present),
            "invalid_present_slots": int(present - valid), "at_least_two_slots": int(valid >= 2),
            "all_four_slots": int(valid == 4), "anchor_only": int(valid == 1)}


TRACK_COST_KEYS = (
    "attempted_transitions", "accepted_transitions", "rejected_transitions",
    "not_attempted_after_loss", "detected_corner_count", "fb_valid_corner_count",
    "ransac_inlier_count",
)


def _historical_lk_geometry(old_inputs: Mapping[str, Any]) -> dict[str, Any]:
    by_group: dict[str, Counter] = defaultdict(Counter)
    failure_reasons: dict[str, Counter] = defaultdict(Counter)
    candidate_rows = []
    case_rows = []
    for case_id in CASE_IDS:
        row = old_inputs[case_id]
        old_candidates = row["candidates"]
        case_candidates = []
        for ordinal, candidate in enumerate(old_candidates):
            is_source = bool(candidate.get("is_Source", False))
            group = "Source_display" if is_source else "nonSource_LK_track"
            boxes = candidate.get("boxes_xyxy", [])
            support = _support(boxes)
            for key, value in support.items():
                by_group[group][key] += int(value)
                by_group["all_old_candidates"][key] += int(value)
            by_group[group]["candidate_count"] += 1
            by_group["all_old_candidates"]["candidate_count"] += 1
            tracklet = candidate.get("tracklet") or {}
            trajectory = tracklet.get("trajectory", [])
            reasons = Counter(
                str(item.get("reason")) for item in trajectory
                if isinstance(item, Mapping) and item.get("reason") not in (None, "")
            )
            for reason, count in reasons.items():
                failure_reasons[group][reason] += int(count)
            direction_cost = tracklet.get("cost", {}).get("directions") or tracklet.get("directions", {})
            cost_total = {}
            for key in ("attempted_transitions", "accepted_transitions"):
                cost_total[key] = sum(
                    int(direction_cost.get(direction, {}).get(key, 0) or 0)
                    for direction in ("backward", "forward")
                )
            cost_total["rejected_transitions"] = max(
                0, cost_total["attempted_transitions"] - cost_total["accepted_transitions"]
            )
            for key, value in cost_total.items():
                if key != "profiled_corner_counts_available":
                    by_group[group][key] += int(value)
                    by_group["all_old_candidates"][key] += int(value)
            cost_total["profiled_corner_counts_available"] = False
            entry = {
                "candidate_order": ordinal + 1,
                "candidate_id": candidate.get("candidate_id"),
                "is_Source_candidate": is_source,
                "geometry_source": "original_source_display_boxes" if is_source else "historical_sparse_LK_track",
                "support": support,
                "tracklet_status": tracklet.get("status", "source_display_only" if is_source else None),
                "track_cost_by_direction": _anon({direction: {
                    "status": direction_cost.get(direction, {}).get("status"),
                    "reason": direction_cost.get(direction, {}).get("reason"),
                    "attempted_transitions": direction_cost.get(direction, {}).get("attempted_transitions", 0),
                    "accepted_transitions": direction_cost.get(direction, {}).get("accepted_transitions", 0),
                } for direction in ("backward", "forward")}),
                "track_cost_total": cost_total,
                "failure_reasons": dict(reasons),
            }
            candidate_rows.append({"case_id": _case_alias(case_id), **entry})
            case_candidates.append(entry)
        case_rows.append({
            "case_id": _case_alias(case_id),
            "candidate_count": len(old_candidates),
            "source_display_candidate_count": sum(bool(c.get("is_Source")) for c in old_candidates),
            "candidates": case_candidates,
        })
    return {
        "status": "pass",
        "scope": "same eight sources and 43 candidate IDs from the historical LK input barrier",
        "candidate_inventory_matches_new_barrier_by_case": True,
        "source_support_semantics": "Source uses its original four display boxes; it has no independent LK track",
        "nonSource_support_semantics": "nonSource boxes come from the historical sparse LK tracklet",
        "support_summary_by_group": {key: dict(value) for key, value in by_group.items()},
        "trajectory_failure_reason_counts_by_group": {
            key: dict(value) for key, value in failure_reasons.items()
        },
        "cases": case_rows,
        "candidate_rows": candidate_rows,
    }


def _candidate_geometry_and_display(inputs: Mapping[str, Any], old_inputs: Mapping[str, Any],
                                    preparation_elapsed_cpu_seconds: Any) -> dict[str, Any]:
    all_cases = []
    all_summary = defaultdict(Counter)
    tracking_cost = defaultdict(Counter)
    transition_failures = defaultdict(Counter)
    trajectory_failures = defaultdict(Counter)
    for cid in CASE_IDS:
        row = inputs[cid]
        candidates = row["_report_candidates"]
        displayed_ids = set(row.get("displayed_candidate_ids", []))
        visible = row.get("base_visible_id_by_candidate", {})
        colors = row.get("base_color_by_candidate", {})
        draw_order = list(row.get("base_draw_order", []))
        draw_rank = {candidate_id: index + 1 for index, candidate_id in enumerate(draw_order)}
        candidate_rows = []
        for ordinal, candidate in enumerate(candidates):
            cid_value = candidate.get("candidate_id")
            is_source = bool(candidate.get("is_Source", False))
            display_boxes = candidate.get("boxes_xyxy", [])
            tracked_boxes = candidate.get("geometry_boxes_xyxy")
            _need(isinstance(display_boxes, list) and len(display_boxes) == 4,
                  f"{cid}: candidate {cid_value} is missing its four display boxes")
            _need(isinstance(tracked_boxes, list) and len(tracked_boxes) == 4,
                  f"{cid}: candidate {cid_value} is missing its independent four-slot geometry")
            rho = candidate.get("rho_by_observation", [])
            slots = []
            for slot in range(4):
                display_box = display_boxes[slot] if slot < len(display_boxes) else None
                tracked_box = tracked_boxes[slot] if slot < len(tracked_boxes) else None
                slots.append({
                    "observation_slot": slot + 1,
                    "display_xyxy_normalized": _box(display_box),
                    "tracked_geometry_xyxy_normalized": _box(tracked_box),
                    "rho": float(rho[slot]) if slot < len(rho) and rho[slot] is not None else None,
                    "display_box_is_original_source": bool(is_source),
                })
            tracklet = candidate.get("tracklet") or {}
            trajectory = []
            for dense_step, item in enumerate(tracklet.get("trajectory", [])):
                if not isinstance(item, Mapping):
                    continue
                trajectory.append({
                    "dense_step": dense_step,
                    "status": item.get("status"),
                    "reason": item.get("reason"),
                    "box_xyxy_normalized": _box(item.get("box_xyxy_norm")),
                })
            transitions = []
            for index, transition in enumerate(tracklet.get("transitions", [])):
                transitions.append({
                    "transition_step": index + 1,
                    "direction": transition.get("direction"),
                    "status": transition.get("status"),
                    "reason": transition.get("reason"),
                    "attempted": transition.get("attempted"),
                    "detected_corner_count": transition.get("detected_corner_count"),
                    "fb_valid_count": transition.get("fb_valid_count"),
                    "retention": transition.get("retention"),
                    "fb_error_median_px": transition.get("fb_error_median_px"),
                    "inlier_count": transition.get("inlier_count"),
                    "inlier_fraction": transition.get("inlier_fraction"),
                    "rho": transition.get("rho"),
                    "center_dxdy_normalized": transition.get("center_dxdy_normalized"),
                    "accepted_box_xyxy_normalized": _box(transition.get("accepted_box_xyxy_norm")),
                })
            tracked_support = _support(tracked_boxes)
            display_support = _support(display_boxes)
            group = "Source" if is_source else "nonSource"
            cost = tracklet.get("cost", {}) if isinstance(tracklet, Mapping) else {}
            cost_total = cost.get("total", {}) if isinstance(cost, Mapping) else {}
            cost_directions = cost.get("directions", {}) if isinstance(cost, Mapping) else {}
            accepted_pair_count = int(cost_total.get("accepted_transitions", 0) or 0)
            candidate_failure_reasons = Counter(
                str(transition.get("reason")) for transition in tracklet.get("transitions", [])
                if transition.get("status") != "accepted" and transition.get("reason") not in (None, "")
            ) if isinstance(tracklet, Mapping) else Counter()
            candidate_trajectory_reasons = Counter(
                str(item.get("reason")) for item in tracklet.get("trajectory", [])
                if isinstance(item, Mapping) and item.get("reason") not in (None, "")
            ) if isinstance(tracklet, Mapping) else Counter()
            for reason, count in candidate_failure_reasons.items():
                transition_failures[group][reason] += int(count)
            for reason, count in candidate_trajectory_reasons.items():
                trajectory_failures[group][reason] += int(count)
            for name in TRACK_COST_KEYS:
                tracking_cost[group][name] += int(cost_total.get(name, 0) or 0)
                tracking_cost["all_candidates"][name] += int(cost_total.get(name, 0) or 0)
            for name, value in tracked_support.items():
                all_summary[group][name] += int(value)
                all_summary["all_candidates"][name] += int(value)
            all_summary[group]["candidate_count"] += 1
            all_summary["all_candidates"]["candidate_count"] += 1
            candidate_rows.append({
                "candidate_order": ordinal + 1,
                "candidate_id": cid_value,
                "window_id": candidate.get("window_id"),
                "is_Source_candidate": is_source,
                "displayed_in_teacher_video": cid_value in displayed_ids,
                "visible_numeric_id": visible.get(str(cid_value)),
                "palette_color_index": colors.get(str(cid_value)),
                "draw_order_rank": draw_rank.get(cid_value),
                "display_box_source": "original_source_observation_boxes" if is_source else "candidate_geometry",
                "tracked_geometry_source": "independent_adjacent_similarity_affine_track",
                "display_support": display_support,
                "tracked_geometry_support": tracked_support,
                "observation_slots": slots,
                "anchor_xyxy_normalized": _box(candidate.get("anchor_box_xyxy")),
                "tracklet_status": tracklet.get("status"),
                "accepted_adjacent_pair_count": accepted_pair_count,
                "track_cost_by_direction": _anon(cost_directions),
                "track_cost_total": {key: int(cost_total.get(key, 0) or 0) for key in TRACK_COST_KEYS},
                "transition_failure_reasons": dict(candidate_failure_reasons),
                "trajectory_failure_reasons": dict(candidate_trajectory_reasons),
                "tracklet_trajectory": trajectory,
                "transition_diagnostics": transitions,
            })
        source = next((c for c in candidate_rows if c["is_Source_candidate"]), None)
        source_compare = None
        if source:
            source_slots = source["observation_slots"]
            paired = [slot for slot in source_slots
                      if _valid_box(slot["display_xyxy_normalized"])
                      and _valid_box(slot["tracked_geometry_xyxy_normalized"])]
            differences = sum(any(abs(float(a) - float(b)) > 1e-12
                                  for a, b in zip(slot["display_xyxy_normalized"],
                                                  slot["tracked_geometry_xyxy_normalized"]))
                              for slot in paired)
            source_compare = {
                "display_valid_slots": (source.get("display_support") or {}).get("valid_slots", 0),
                "independently_tracked_valid_slots": (source.get("tracked_geometry_support") or {}).get("valid_slots", 0),
                "paired_valid_slots": len(paired),
                "paired_slots_differing_from_original_display": differences,
            }
        all_cases.append({
            "case_id": _case_alias(cid),
            "dataset": row["dataset"],
            "cohort": row["cohort"],
            "candidate_count": len(candidates),
            "displayed_candidate_ids": sorted(displayed_ids),
            "base_visible_id_by_candidate": _anon(visible),
            "base_color_palette_index_by_candidate": _anon(colors),
            "base_draw_order": draw_order,
            "badge_positions_fixed_across_controls": True,
            "source_display_support": source["display_support"] if source else None,
            "source_independent_track_support": source["tracked_geometry_support"] if source else None,
            "source_display_vs_independent_track": source_compare,
            "candidates": candidate_rows,
        })
    return {
        "status": "pass",
        "scope": "all original locked candidates, four original observation slots; no GT values or absolute frame IDs",
        "candidate_count": sum(row["candidate_count"] for row in all_cases),
        "displayed_candidate_count": sum(len(row["displayed_candidate_ids"]) for row in all_cases),
        "preparation_elapsed_cpu_seconds": float(preparation_elapsed_cpu_seconds),
        "support_summary_by_group": {name: dict(counts) for name, counts in all_summary.items()},
        "tracking_cost_summary_by_group": {name: dict(counts) for name, counts in tracking_cost.items()},
        "transition_failure_reason_counts_by_group": {
            name: dict(counts) for name, counts in transition_failures.items()
        },
        "trajectory_failure_reason_counts_by_group": {
            name: dict(counts) for name, counts in trajectory_failures.items()
        },
        "source_display_geometry_and_independent_track_are_separate": True,
        "normalized_geometry": "xyxy in the original normalized image plane; no clipping or GT coordinates",
        "historical_lk_comparison": _historical_lk_geometry(old_inputs),
        "cases": all_cases,
    }


def _teacher_controls(teacher: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    rows = teacher["rows"]
    cases = teacher["cases"]
    by_kind = {}
    for kind in TEACHER_KINDS:
        selected_rows = [r for r in rows if r["kind"] == kind]
        same = sum(bool(c.get(CONTROL_FLAGS[kind])) for c in cases) if kind in CONTROL_FLAGS else None
        changed = sum(bool(c.get("wrong_query_choice_changed")) for c in cases) if kind == "wrong_query" else None
        by_kind[kind] = {
            "case_denominator": len(cases),
            "same_geometry_count": same,
            "same_geometry_rate": (same / len(cases)) if same is not None and cases else None,
            "wrong_query_changed_count": changed,
            "format_valid_count": sum(bool(r["format_valid"]) for r in selected_rows),
            "truncated_count": sum(bool(r["truncated"]) for r in selected_rows),
            "abstain_or_invalid_count": sum(r.get("selected_candidate_id") is None for r in selected_rows),
            "candidate_count_total": sum(int(r.get("candidate_count", 0)) for r in selected_rows),
        }
    source_candidate_ids = {}
    for cid, row in inputs.items():
        source = next((c for c in row["_report_candidates"] if c.get("is_Source")), None)
        source_candidate_ids[cid] = source.get("candidate_id") if source else None
    base_rows = [r for r in rows if r["kind"] == "base"]
    base_breakdown = Counter()
    for row in base_rows:
        cid = row["case_id"]
        selected = row.get("selected_candidate_id")
        if selected is None:
            base_breakdown["abstain_or_invalid"] += 1
        elif selected == source_candidate_ids.get(cid):
            base_breakdown["Source"] += 1
        else:
            base_breakdown["nonSource"] += 1
    return {
        "status": teacher["status"],
        "actual_call_count": int(teacher["count"]),
        "cases": [_anon(row) for row in sorted(cases, key=lambda r: r["case_id"])],
        "requests": [_anon(row) for row in sorted(rows, key=lambda r: (r["case_id"], TEACHER_KINDS.index(r["kind"])))],
        "condition_summary": by_kind,
        "base_selection_count_by_reference_kind": dict(base_breakdown),
        "format_valid_count": int(teacher["format_valid_count"]),
        "truncated_count": int(teacher["truncated_count"]),
        "cost": _anon(teacher.get("cost", {})),
        "raw_query_and_response_prose_excluded": True,
    }


def _fit_and_motion(report: Mapping[str, Any], student_barrier: Mapping[str, Any],
                    actual_fits: Sequence[Mapping[str, Any]], no_fit_rows: Sequence[Mapping[str, Any]],
                    motion_fits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    movement = report["fit_movement_rows"]
    per_algorithm = {}
    for algorithm in ALGORITHMS:
        fit_name = f"teacher_{algorithm}"
        rows = [r for r in movement if r.get("fit") == fit_name]
        actual = [r for r in rows if r.get("status") in {"updated_fit", "diagnostic_update_even_if_lock_no_update"}]
        placeholders = [r for r in rows if r.get("status") == "locked_no_update_exact_F_reuse"]
        per_algorithm[algorithm] = {
            "actual_fit_count": len(actual),
            "no_fit_exact_F_placeholder_count": len(placeholders),
            "actual_optimizer_steps": sum(int(r.get("optimizer_steps", 0)) for r in actual),
            "full_final_model_readouts": len(actual),
        }
    motion_round_rows = report.get("motion_round_rows", [])
    by_step = []
    for step in range(FIT_ROUNDS):
        rows = [r for r in motion_round_rows if int(r["step"]) == step]
        by_step.append({
            "round": step + 1,
            "fit_round_count": len(rows),
            "mean_opd_loss": _mean(r.get("opd_loss") for r in rows),
            "mean_motion_loss": _mean(r.get("motion_loss") for r in rows),
            "mean_total_loss": _mean(r.get("total_loss") for r in rows),
            "optimizer_updated_count": sum(bool(r.get("updated")) for r in rows),
            "motion_pair_count_total": sum(int(r.get("motion_valid_pair_count", 0)) for r in rows),
            "lambda_motion_values": sorted({float(r["lambda_motion"]) for r in rows}),
        })
    return {
        "status": "pass",
        "actual_fit_count": int(report["actual_fit_count"]),
        "no_fit_placeholder_count": int(report.get("no_fit_placeholder_count", len(no_fit_rows))),
        "no_update_case_count": len({str(row["case_id"]) for row in no_fit_rows}),
        "declared_rounds": int(report["actual_fit_count"]) * FIT_ROUNDS,
        "declared_rounds_per_fit": FIT_ROUNDS,
        "actual_optimizer_steps": int(report.get("actual_fit_optimizer_step_sum", report.get("optimizer_steps", 0))),
        "actual_full_final_model_readouts": int(student_barrier["actual_final_redecode_calls"]),
        "no_update_outputs_reuse_F_exactly": True,
        "temporal_T": "exact reuse of old sealed T for this original eight-case cohort; no new temporal fit",
        "original_parameter_scope": 1792,
        "motion_lambda": 1.0,
        "algorithm_summary": per_algorithm,
        "fit_movement_rows": [_anon(row) for row in movement],
        "motion_round_rows": [_anon(row) for row in motion_round_rows],
        "motion_iou_round_summary": by_step,
        "V03_prior_control": _anon(report["V03_prior_control"]),
    }


def _mean(values: Sequence[Any]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return (sum(finite) / len(finite)) if finite else None


def _present_anchor_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "candidate_box_present_anchor_count", "candidate_box_valid_anchor_count",
        "candidate_box_missing_anchor_count", "candidate_invalid_geometry_anchor_count",
        "GT_known_original4_anchor_count", "GT_missing_original4_anchor_count",
        "candidate_present_and_GT_known_anchor_count", "candidate_missing_where_GT_known_count",
        "candidate_invalid_where_GT_known_count",
    )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped["Source" if row.get("is_Source_candidate") else "nonSource"].append(row)
    result = []
    for name, selected in grouped.items():
        result.append({
            "candidate_group": name,
            "candidate_count": len(selected),
            **{field: sum(int(row.get(field, 0) or 0) for row in selected) for field in fields},
            "support_semantics": "counts only; GT-missing remains unknown and is not replaced with zero",
        })
    return result


def _anonymous_metric_bundle(inputs: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    present_anchor_summary = report.get("candidate_present_anchor_summary")
    if not isinstance(present_anchor_summary, list):
        present_anchor_summary = _present_anchor_summary(report["candidate_geometry_rows"])
    zero_convention = report.get("candidate_missing_zero_convention")
    if zero_convention is None:
        zero_convention = (
            "Only GT-known slots among the original four enter the common denominator; candidate-missing slots "
            "are zero-filled there, while GT-missing slots are excluded and remain unknown support, not coordinate failure."
        )
    return {
        "status": "pass",
        "schema": report["schema"],
        "scope": "fixed original eight historical sources; postseal metrics only",
        "case_rows": [_anon(row) for row in sorted(report["case_rows"], key=lambda r: r["case_id"])],
        "metric_rows": [_anon(row) for row in report["metric_rows"]],
        "arm_comparison_rows": [_anon(row) for row in report["arm_comparison_rows"]],
        "groups": [_anon(row) for row in report["groups"]],
        "candidate_geometry_rows": [_anon(row) for row in report["candidate_geometry_rows"]],
        "candidate_display_rows": [_anon(row) for row in report["candidate_display_rows"]],
        "candidate_oracle_rows": [_anon(row) for row in report["candidate_oracle_rows"]],
        "candidate_present_anchor_summary": [_anon(row) for row in present_anchor_summary],
        "candidate_missing_zero_convention": _anon(zero_convention),
        "metric_count": len(report["metric_rows"]),
        "case_count": len(report["case_rows"]),
        "candidate_geometry_row_count": len(report["candidate_geometry_rows"]),
        "candidate_display_row_count": len(report["candidate_display_rows"]),
        "candidate_oracle_row_count": len(report["candidate_oracle_rows"]),
        "all_positive_negative_neutral_unknown_rows_retained": True,
        "GT_coordinates_and_source_hashes_excluded": True,
    }


def _groups(report: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    """Index group metrics by cohort, dataset, and arm.

    The postseal report contains a separate group row for each arm. Omitting
    the arm from this key silently made every summary row reuse whichever arm
    happened to be last in the serialized groups list.
    """
    return {
        (str(row["cohort"]), str(row["dataset"]), str(row["arm"])): row
        for row in report["groups"]
    }


def _metric_text(group: Mapping[str, Any] | None, name: str) -> str:
    if not group:
        return "NA"
    value = group.get("metrics_percent", {}).get(name)
    return "NA" if value is None else f"{float(value):.4f}"


def _delta_text(row: Mapping[str, Any]) -> str:
    value = row.get("delta_vIoU_pp")
    return "NA" if value is None else f"{float(value):+.4f}"


def _comparison_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    result = {}
    for arm in ARM_ORDER:
        selected = [row for row in rows if row.get("arm") == arm]
        counts = Counter(str(row.get("category", "unknown")) for row in selected)
        result[arm] = {category: int(counts.get(category, 0)) for category in NEUTRAL_CATEGORIES}
    return result


def _support_groups(geometry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return geometry["support_summary_by_group"]


def _markdown(inputs: Mapping[str, Any], geometry: Mapping[str, Any], teacher: Mapping[str, Any],
              fit: Mapping[str, Any], metrics: Mapping[str, Any], report: Mapping[str, Any]) -> str:
    groups = _groups(report)
    categories = _comparison_counts(report["arm_comparison_rows"])
    case_rows = {row["case_id"]: row for row in report["case_rows"]}
    teacher_case_rows = {row["case_id"]: row for row in teacher["cases"]}
    lines = [
        "# Adjacent-Motion Spatial OPD: No vIoU Gain Over T on the Fixed Eight-Source Cohort",
        "",
        "## Scope and reading rules",
        "",
        "This report uses the sealed eight-source cohort only. It reports the locked candidate tracks, 48 teacher-control calls, and postseal metrics. The cases are historically exposed; this is a finite diagnostic, not a fresh-source confirmation or a deployment promotion. No new temporal fit was run. Old T is reused exactly, and V03 reference controls are prior evidence reused with zero new fits.",
        "",
        f"The postseal scorer reports {len(report['metric_rows'])} metric rows, {len(report['candidate_geometry_rows'])} tracked-geometry rows, {len(report['candidate_display_rows'])} display-geometry rows, {fit['actual_fit_count']} actual fits, {fit['actual_optimizer_steps']} actual optimizer steps, and {fit['actual_full_final_model_readouts']} full final-model readouts. The fixed five-round protocol implies {fit['declared_rounds']} declared fit rounds; declared rounds and optimizer steps are different counts.",
        "",
        "All case names are aliases: HC01-HC04 and V01-V04. The anonymous data files retain per-case metric, comparison, candidate, fit, motion-round, teacher-control and Oracle rows while excluding source paths and hashes, frame IDs, GT coordinates, query prose, raw responses, model weights and optimizer tensors.",
        "",
        "## Complete-tube metrics by domain",
        "",
        "Values are percent. T is the old sealed temporal output. T + spatial arms reuse its interval and use the current spatial readout. Each domain contains four cases.",
        "",
        "|Arm|HC2 vIoU|HC2 tIoU|HC2 dense sIoU|VidSTG vIoU|VidSTG tIoU|VidSTG dense sIoU|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_ORDER:
        hc = groups.get(("old8", "hc2", arm))
        vi = groups.get(("old8", "vidstg", arm))
        values = [
            _metric_text(hc, "m_vIoU"), _metric_text(hc, "m_tIoU"), _metric_text(hc, "m_sIoU_dense_GT"),
            _metric_text(vi, "m_vIoU"), _metric_text(vi, "m_tIoU"), _metric_text(vi, "m_sIoU_dense_GT"),
        ]
        lines.append("|" + ARM_LABELS[arm] + "|" + "|".join(values) + "|")

    t_hc = groups.get(("old8", "hc2", "T"))
    t_vi = groups.get(("old8", "vidstg", "T"))
    motion_hc = groups.get(("old8", "hc2", "T_teacher_motion_iou"))
    motion_vi = groups.get(("old8", "vidstg", "T_teacher_motion_iou"))
    spatial_iou_categories = categories["teacher_iou"]
    lines += [
        "",
        "Observed result: adding Motion-OPD to T changed mean vIoU from "
        + _metric_text(t_hc, "m_vIoU") + "% to " + _metric_text(motion_hc, "m_vIoU")
        + "% in HC2 and from " + _metric_text(t_vi, "m_vIoU") + "% to "
        + _metric_text(motion_vi, "m_vIoU") + "% in VidSTG. Against F, the spatial IoU-OPD arm "
        + f"was positive in {spatial_iou_categories['positive']} case(s), negative in "
        + f"{spatial_iou_categories['negative']}; the other {fit['no_update_case_count']} cases had exact-F no-update outputs. "
        + "This is a negative aggregate complementarity result on these fixed historical sources; it does not establish route-wide failure or fresh-source performance.",
        "",
    ]

    lines += [
        "",
        "## Per-case arm readout",
        "",
        "Values are complete-tube vIoU percent. Positive, negative, neutral and unknown classifications use the scorer's descriptive tolerance; they are not a promotion gate.",
        "",
        "|Case|Domain|Frozen|Projection|T|Direct|IoU-OPD|Motion-OPD|T+Direct|T+IoU|T+Motion|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cid in CASE_IDS:
        row = case_rows[cid]
        values = [row.get("arms", {}).get(arm, {}).get("m_vIoU") for arm in ARM_ORDER]
        lines.append("|" + _case_alias(cid) + "|" + row["dataset"] + "|" + "|".join(
            "NA" if value is None else f"{100 * float(value):.4f}" for value in values
        ) + "|")
    lines += [
        "",
        "## Case-level comparison counts",
        "",
        "Each cell counts cases. F is the baseline row; T + spatial rows compare against T; spatial rows compare against F. Unknown comparisons are retained when a baseline or arm was not created.",
        "",
        "|Arm|Positive|Negative|Neutral|Unknown|Baseline|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_ORDER:
        c = categories[arm]
        lines.append(f"|{ARM_LABELS[arm]}|{c['positive']}|{c['negative']}|{c['neutral']}|{c['unknown']}|{c['baseline']}|")

    lines += [
        "",
        "## P0: candidate motion geometry and Source display",
        "",
        "Candidate support is counted from geometry_boxes_xyxy, which is the independently tracked trajectory. The Source boxes_xyxy used for display are reported separately because the Source display preserves the original four source boxes. Source display coverage is not motion-track support.",
        "",
        "|Candidate geometry source|Count|At least 2 slots|All 4 slots|Anchor only|Missing slots|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    historical_support = geometry["historical_lk_comparison"]["support_summary_by_group"]
    support_rows = [
        (_support_groups(geometry).get("all_candidates", {}), "New independent track: all candidates"),
        (_support_groups(geometry).get("Source", {}), "New independent track: Source"),
        (_support_groups(geometry).get("nonSource", {}), "New independent track: nonSource"),
        (historical_support.get("all_old_candidates", {}), "Historical LK/display: all candidates"),
        (historical_support.get("Source_display", {}), "Historical: Source display boxes"),
        (historical_support.get("nonSource_LK_track", {}), "Historical: nonSource LK track"),
    ]
    for row, label in support_rows:
        lines.append(f"|{label}|{row.get('candidate_count', 'NA')}|{row.get('at_least_two_slots', 0)}|{row.get('all_four_slots', 0)}|{row.get('anchor_only', 0)}|{row.get('missing_slots', 0)}|")
    lines += [
        "",
        "|Case|Source display valid slots|Independent Source-track valid slots|Paired valid slots|Paired slots changed from display|",
        "|---|---:|---:|---:|---:|",
    ]
    for row in geometry["cases"]:
        compare = row.get("source_display_vs_independent_track") or {}
        lines.append(f"|{row['case_id']}|{compare.get('display_valid_slots', 0)}|{compare.get('independently_tracked_valid_slots', 0)}|{compare.get('paired_valid_slots', 0)}|{compare.get('paired_slots_differing_from_original_display', 0)}|")
    lines += [
        "",
        "New adjacent-track work totals by candidate group:",
        "",
        "|Group|Attempted pairs|Accepted pairs|Rejected pairs|Not attempted after loss|Detected corners|FB-valid corners|RANSAC inliers|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group_name in ("all_candidates", "Source", "nonSource"):
        cost = geometry["tracking_cost_summary_by_group"].get(group_name, {})
        lines.append("|" + "|".join([
            group_name, *(str(int(cost.get(key, 0))) for key in TRACK_COST_KEYS)
        ]) + "|")
    lines += [
        "",
        "The historical LK support summary treats Source as the original displayed four boxes and nonSource as the old LK geometry. It is not a Source-motion estimate. The new rows track Source and nonSource independently from the selected seed. Full normalized candidate boxes, per-slot rho, ordered trajectory, accepted/rejected adjacent-pair diagnostics, per-direction work counts, failure reasons, and display-ID/color/order mappings are in `candidate_geometry_and_display.json`; physical frame IDs and absolute timestamps are removed. The new preparation elapsed CPU time is " + _number(geometry.get("preparation_elapsed_cpu_seconds")) + " seconds.",
        "",
        "## P1: teacher control matrix",
        "",
        "Each control denominator is eight cases, not the 48 total requests. Same-geometry counts require valid outputs in both the base and control response. The wrong-query condition is reported as choice changed, with abstentions also counted separately.",
        "",
        "|Condition|Same geometry / 8|Changed choice / 8|Format valid / 8|Truncated / 8|Abstain or invalid / 8|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind in TEACHER_KINDS:
        row = teacher["condition_summary"][kind]
        same = "NA" if row["same_geometry_count"] is None else str(row["same_geometry_count"])
        changed = "NA" if row["wrong_query_changed_count"] is None else str(row["wrong_query_changed_count"])
        lines.append(f"|{kind}|{same}|{changed}|{row['format_valid_count']}|{row['truncated_count']}|{row['abstain_or_invalid_count']}|")
    lines += [
        "",
        "Base selections by reference type: " + ", ".join(
            f"{key} {value}/8" for key, value in sorted(teacher["base_selection_count_by_reference_kind"].items())
        ) + ". Teacher control consistency is sensitivity evidence, not proof of instance identity.",
        "",
        "## Candidate geometry quality and postseal Oracle diagnostic",
        "",
        "The scorer keeps candidate-present IoU, same-anchor paired Frozen IoU, GT-known common-support scores and postseal pool/display Oracle rows separate. Missing candidate geometry is zero-filled only for the stated GT-known coverage diagnostic; missing GT remains unknown and is excluded from that denominator.",
        "",
        "|Case|Candidates|Displayed|Oracle on known common support|Displayed Oracle|Chosen candidate|Source baseline|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    oracle = {row["case_id"]: row for row in report["candidate_oracle_rows"]}
    for cid in CASE_IDS:
        alias = _case_alias(cid)
        row = oracle.get(cid, oracle.get(alias, {}))
        geometry_case = next(r for r in geometry["cases"] if r["case_id"] == alias)
        lines.append("|" + "|".join([
            alias, str(geometry_case["candidate_count"]), str(len(geometry_case["displayed_candidate_ids"])),
            _number(row.get("candidate_oracle_known_GT_common_support")),
            _number(row.get("display_oracle_known_GT_common_support")),
            _number(row.get("chosen_geometry_known_GT_common_support")),
            _number(row.get("Source_original_known_GT_common_support")),
        ]) + "|")

    lines += [
        "",
        "## P2: fits, actual optimizer steps and readouts",
        "",
        f"The locked spatial scope uses 1,792 parameters, the original per-domain Direct/IoU-OPD settings and five declared rounds. Motion-OPD adds the fixed adjacent-displacement loss with lambda_motion = 1.0. The report distinguishes {fit['actual_fit_count']} actual fits, {fit['no_fit_placeholder_count']} exact-F no-update placeholders, {fit['actual_optimizer_steps']} actual Adam steps, and {fit['actual_full_final_model_readouts']} full final-model readouts. Actual optimizer steps can be below declared rounds when the locked objective has no informative gradient.",
        "",
        "|Case|Eligible reference|Direct steps|IoU steps|Motion steps|Direct parameter delta|IoU parameter delta|Motion parameter delta|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    movement_by_case = defaultdict(dict)
    for row in report["fit_movement_rows"]:
        movement_by_case[row["case_id"]][row["fit"]] = row
    for cid in CASE_IDS:
        tc = teacher_case_rows[_case_alias(cid)]
        by = movement_by_case[cid]
        values = []
        for algorithm in ALGORITHMS:
            row = by.get(f"teacher_{algorithm}", {})
            values.append(str(row.get("optimizer_steps", 0)))
        deltas = [_number(by.get(f"teacher_{a}", {}).get("parameter_delta_norm")) for a in ALGORITHMS]
        lines.append(f"|{_case_alias(cid)}|{'yes' if tc.get('eligible_for_update') else 'no - exact F reuse'}|" + "|".join(values + deltas) + "|")
    lines += [
        "",
        "Motion-OPD round losses and pair-support counts are retained per case and fit. Old T was reused exactly; no new temporal model fit was made. The V03 Source/old-ID reference control is exact prior-evidence reuse with zero new fits.",
        "",
        "|Motion round|Fits|Mean OPD loss|Mean motion loss|Mean total loss|Updated fits|Valid pairs summed|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fit["motion_iou_round_summary"]:
        lines.append("|" + "|".join([
            str(row["round"]), str(row["fit_round_count"]), _number(row["mean_opd_loss"]),
            _number(row["mean_motion_loss"]), _number(row["mean_total_loss"]),
            str(row["optimizer_updated_count"]), str(row["motion_pair_count_total"]),
        ]) + "|")
    v03 = report["V03_prior_control"]
    lines += [
        "",
        f"V03 prior control status: {v03.get('status')}; new fits executed: {v03.get('new_fits_executed')}. Its previously scored fixed-reference metrics are included only as exact reuse, not as motion-selected candidate evidence.",
        "",
        "|V03 prior arm|Domain|Cases|vIoU %|tIoU %|dense sIoU %|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in v03.get("groups", []):
        m = row.get("metrics_percent", {})
        lines.append("|" + "|".join([
            str(row.get("arm", "")), str(row.get("dataset", "")), str(row.get("scored_case_count", "")),
            _number(m.get("m_vIoU")), _number(m.get("m_tIoU")), _number(m.get("m_sIoU_dense_GT")),
        ]) + "|")
    lines += [
        "",
        "|V03 locked reference role|Candidate IoU on GT-known anchors|Paired Source IoU|Difference|Known labels / unknown labels|Missing candidate where GT known|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in v03.get("anchor_rows", []):
        lines.append("|" + "|".join([
            str(row.get("reference_role", "")), _number(row.get("present_anchor_GT_IoU_mean_known_only")),
            _number(row.get("paired_F_source_same_anchor_GT_IoU_mean_known_only")),
            _number(row.get("present_minus_paired_F_source_mean_IoU_known_only")),
            f"{row.get('GT_known_original4_anchor_count', 0)} / {row.get('GT_missing_original4_anchor_count', 0)}",
            str(row.get("candidate_missing_where_GT_known_count", 0)),
        ]) + "|")
    lines += [
        "",
        "## Boundaries",
        "",
        "All case-level metric rows and every positive, negative, neutral and unknown comparison are preserved in `metric_and_case_rows.json`. Candidate geometry, display mappings, support states, candidate IoUs and Oracle diagnostics are in separate anonymous files. Teacher outputs retain parsed choices, control validity, call costs and format status, but omit raw answer/query prose. This finite historical-source diagnostic does not qualify stable performance on fresh sources or change the deployed method.",
        "",
    ]
    return "\n".join(lines)


def _number(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not math.isfinite(number) else f"{number:.6f}"


def _pdf(inputs: Mapping[str, Any], geometry: Mapping[str, Any], teacher: Mapping[str, Any],
         fit: Mapping[str, Any], report: Mapping[str, Any], destination: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph, Table, TableStyle
    from pypdf import PdfReader

    regular_candidates = (
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    bold_candidates = (
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    )
    regular = next((path for path in regular_candidates if path.is_file()), None)
    bold = next((path for path in bold_candidates if path.is_file()), None)
    _need(regular is not None and bold is not None, "LiberationSans regular/bold fonts are required")
    pdfmetrics.registerFont(TTFont("ReportSans", str(regular)))
    pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(bold)))

    page_size = landscape(A4)
    page_width, page_height = page_size
    margin = 34
    content_width = page_width - 2 * margin
    body = ParagraphStyle("ReportBody", fontName="ReportSans", fontSize=7.4, leading=9.2,
                          textColor=colors.HexColor("#253746"), spaceAfter=0)
    small = ParagraphStyle("ReportSmall", parent=body, fontSize=6.6, leading=8.0)
    title = ParagraphStyle("ReportTitle", fontName="ReportSans-Bold", fontSize=20, leading=23,
                           textColor=colors.HexColor("#12364a"))
    heading = ParagraphStyle("ReportHeading", fontName="ReportSans-Bold", fontSize=11, leading=13,
                             textColor=colors.HexColor("#12364a"))
    subtitle = ParagraphStyle("ReportSubtitle", fontName="ReportSans", fontSize=8, leading=10,
                              textColor=colors.HexColor("#526572"))
    table_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#18354a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "ReportSans-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "ReportSans"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.4),
        ("LEADING", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c6ce")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eff4f6"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ])
    pdf = canvas.Canvas(str(destination), pagesize=page_size, pageCompression=1)
    pdf.setTitle("Adjacent-Motion Spatial OPD: No vIoU Gain Over T on the Fixed Eight-Source Cohort")
    pdf.setAuthor("STVG research")

    def footer(page: int) -> None:
        pdf.setStrokeColor(colors.HexColor("#c8d2d8"))
        pdf.line(margin, 26, page_width - margin, 26)
        pdf.setFont("ReportSans", 6.5)
        pdf.setFillColor(colors.HexColor("#526572"))
        pdf.drawString(margin, 15, "Finite historical-source diagnostic | anonymous metrics")
        pdf.drawRightString(page_width - margin, 15, f"{page}/3")

    def begin_page(page: int, heading_text: str, subtext: str | None = None) -> float:
        y = page_height - margin
        p = Paragraph(escape(heading_text), title)
        _, h = p.wrap(content_width, y)
        p.drawOn(pdf, margin, y - h)
        y -= h + 4
        if subtext:
            q = Paragraph(escape(subtext), subtitle)
            _, h2 = q.wrap(content_width, y)
            q.drawOn(pdf, margin, y - h2)
            y -= h2 + 8
        footer(page)
        return y

    def add_para(text: str, y: float, style: ParagraphStyle = body, gap: float = 6) -> float:
        p = Paragraph(escape(text), style)
        _, h = p.wrap(content_width, y)
        _need(y - h >= 36, f"PDF page overflow while placing: {text[:48]}")
        p.drawOn(pdf, margin, y - h)
        return y - h - gap

    def add_table(rows: Sequence[Sequence[Any]], widths: Sequence[float], y: float,
                  font_size: float = 6.4, leading: float = 7.5) -> float:
        pstyle = ParagraphStyle("Cell", fontName="ReportSans", fontSize=font_size,
                                leading=leading, textColor=colors.HexColor("#253746"))
        hstyle = ParagraphStyle("HeadCell", parent=pstyle, fontName="ReportSans-Bold",
                                textColor=colors.white)
        cells = [[Paragraph(escape(str(v)), hstyle if row_i == 0 else pstyle)
                  for v in row] for row_i, row in enumerate(rows)]
        table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(table_style)
        _, height = table.wrap(content_width, y)
        _need(y - height >= 36, "PDF page overflow in a data table")
        table.drawOn(pdf, margin, y - height)
        return y - height - 7

    groups = _groups(report)
    case_rows = {row["case_id"]: row for row in report["case_rows"]}
    # Page 1: tube metrics and all case-level classifications.
    y = begin_page(1, "Adjacent-Motion Spatial OPD: No vIoU Gain Over T",
                   "Fixed eight-source diagnostic | HC01-HC04 and V01-V04 | postseal evaluation")
    groups = _groups(report)
    t_hc = groups.get(("old8", "hc2", "T"))
    t_vi = groups.get(("old8", "vidstg", "T"))
    motion_hc = groups.get(("old8", "hc2", "T_teacher_motion_iou"))
    motion_vi = groups.get(("old8", "vidstg", "T_teacher_motion_iou"))
    spatial_iou_comp = _comparison_counts(report["arm_comparison_rows"])["teacher_iou"]
    y = add_para(
        f"Eight historically exposed sources; {teacher['actual_call_count']} teacher calls; "
        f"{fit['actual_fit_count']} actual fits; {fit['actual_optimizer_steps']} actual optimizer steps; "
        f"{fit['actual_full_final_model_readouts']} full final-model readouts. Old T is reused exactly. "
        f"Adding Motion-OPD to T changed mean vIoU from {_metric_text(t_hc, 'm_vIoU')}% to "
        f"{_metric_text(motion_hc, 'm_vIoU')}% in HC2 and from {_metric_text(t_vi, 'm_vIoU')}% to "
        f"{_metric_text(motion_vi, 'm_vIoU')}% in VidSTG. Spatial IoU-OPD was positive in "
        f"{spatial_iou_comp['positive']} case(s), negative in {spatial_iou_comp['negative']}, with "
        f"the other {fit['no_update_case_count']} cases using exact-F no-update outputs. This is a negative aggregate result "
        "on this fixed historical cohort, not evidence of route-wide failure or fresh-source performance.",
        y, body, 7)
    y = add_para("Complete-tube metrics (percent; four cases per domain)", y, heading, 4)
    table = [["Arm", "HC2 vIoU", "HC2 tIoU", "HC2 sIoU", "VidSTG vIoU", "VidSTG tIoU", "VidSTG sIoU"]]
    for arm in ARM_ORDER:
        hc = groups.get(("old8", "hc2", arm))
        vi = groups.get(("old8", "vidstg", arm))
        table.append([ARM_LABELS[arm], _metric_text(hc, "m_vIoU"), _metric_text(hc, "m_tIoU"),
                      _metric_text(hc, "m_sIoU_dense_GT"), _metric_text(vi, "m_vIoU"),
                      _metric_text(vi, "m_tIoU"), _metric_text(vi, "m_sIoU_dense_GT")])
    y = add_table(table, [150, 80, 80, 80, 80, 80, 80], y, 6.1, 7.0)
    y = add_para("Case-level vIoU (percent)", y, heading, 4)
    table = [["Case", "Domain", "F", "Projection", "T", "Direct", "IoU", "Motion", "T+Direct", "T+IoU", "T+Motion"]]
    for cid in CASE_IDS:
        row = case_rows[cid]
        vals = [row.get("arms", {}).get(arm, {}).get("m_vIoU") for arm in ARM_ORDER]
        table.append([_case_alias(cid), row["dataset"], *["NA" if v is None else f"{float(v)*100:.2f}" for v in vals]])
    y = add_table(table, [48, 54, 58, 64, 50, 50, 48, 50, 62, 56, 56], y, 5.7, 6.8)
    y = add_para("Case comparison counts", y, heading, 4)
    ctab = [["Arm", "+", "-", "Neutral", "Unknown", "Base"]]
    counts = _comparison_counts(report["arm_comparison_rows"])
    for arm in ARM_ORDER:
        c = counts[arm]
        ctab.append([ARM_LABELS[arm], str(c["positive"]), str(c["negative"]), str(c["neutral"]), str(c["unknown"]), str(c["baseline"])])
    y = add_table(ctab, [190, 50, 50, 75, 75, 50], y, 5.8, 6.6)
    pdf.showPage()

    # Page 2: geometry, display protocol, postseal candidate quality, teacher controls.
    y = begin_page(2, "Candidate Motion Evidence and Teacher Controls", "Source display geometry and independently tracked Source geometry are separate")
    support = _support_groups(geometry)
    y = add_para("Tracked-geometry support (four original observation slots)", y, heading, 4)
    stable = [["Group", "Candidates", ">=2 slots", "All 4", "Anchor only", "Missing slots"]]
    old_support = geometry["historical_lk_comparison"]["support_summary_by_group"]
    for r, label in (
        (support.get("all_candidates", {}), "New independent track: all"),
        (support.get("Source", {}), "New independent track: Source"),
        (support.get("nonSource", {}), "New independent track: nonSource"),
        (old_support.get("all_old_candidates", {}), "Historical LK/display: all"),
        (old_support.get("Source_display", {}), "Historical Source display"),
        (old_support.get("nonSource_LK_track", {}), "Historical nonSource LK"),
    ):
        stable.append([label, str(r.get("candidate_count", "NA")), str(r.get("at_least_two_slots", 0)),
                       str(r.get("all_four_slots", 0)), str(r.get("anchor_only", 0)), str(r.get("missing_slots", 0))])
    y = add_table(stable, [190, 90, 90, 70, 90, 90], y, 5.5, 6.2)
    source_rows = []
    for row in geometry["cases"]:
        compare = row.get("source_display_vs_independent_track") or {}
        source_rows.append([row["case_id"], str(compare.get("display_valid_slots", "NA")),
                            str(compare.get("independently_tracked_valid_slots", "NA")),
                            str(compare.get("paired_slots_differing_from_original_display", "NA"))])
    y = add_para("Source display support / independent Source track support", y, heading, 4)
    y = add_table([["Case", "Displayed original", "Tracked Source", "Paired slots changed"], *source_rows],
                  [90, 180, 180, 150], y, 5.0, 5.8)
    y = add_para(
        f"Old support uses original Source display boxes and historical nonSource LK geometry. New Source boxes are independently tracked; preparation took {geometry['preparation_elapsed_cpu_seconds']:.2f} CPU seconds. Per-candidate failure reasons are in the anonymous geometry file.",
        y, small, 3)
    y = add_para("New adjacent-track work totals", y, heading, 3)
    costtab = [["Group", "Attempted", "Accepted", "Rejected", "Not attempted"]]
    for group_name in ("all_candidates", "Source", "nonSource"):
        cost = geometry["tracking_cost_summary_by_group"].get(group_name, {})
        costtab.append([group_name, *[str(int(cost.get(key, 0))) for key in
                                     ("attempted_transitions", "accepted_transitions", "rejected_transitions", "not_attempted_after_loss")]])
    y = add_table(costtab, [150, 110, 110, 110, 150], y, 4.9, 5.7)
    y = add_para("Teacher controls: denominator is eight cases per condition", y, heading, 4)
    ttab = [["Condition", "Same geom.", "Changed", "Valid", "Truncated", "Abstain/invalid"]]
    for kind in TEACHER_KINDS:
        r = teacher["condition_summary"][kind]
        ttab.append([kind, _number(r["same_geometry_count"]), _number(r["wrong_query_changed_count"]),
                     str(r["format_valid_count"]), str(r["truncated_count"]), str(r["abstain_or_invalid_count"])])
    y = add_table(ttab, [120, 100, 90, 75, 85, 120], y, 5.2, 6.0)
    y = add_para("Candidate Oracle IoU uses GT-known original-four support only. Missing candidate boxes are zero-filled only on that known-GT denominator; GT-unknown slots stay excluded.", y, small, 3)
    # The geometry bundle has already anonymized IDs (HC01/V01), while the
    # postseal Oracle rows retain source IDs. Normalize the report keys before
    # joining so valid Oracle scores are not rendered as NA.
    oracle_by_case = {_case_alias(str(row["case_id"])): row
                      for row in report["candidate_oracle_rows"]}
    otab = [["Case", "Pool Oracle", "Displayed Oracle", "Selected", "Source"]]
    for row in geometry["cases"]:
        value = oracle_by_case.get(row["case_id"], {})
        otab.append([row["case_id"], _number(value.get("candidate_oracle_known_GT_common_support")),
                     _number(value.get("display_oracle_known_GT_common_support")),
                     _number(value.get("chosen_geometry_known_GT_common_support")),
                     _number(value.get("Source_original_known_GT_common_support"))])
    y = add_table(otab, [70, 145, 145, 145, 145], y, 4.9, 5.8)
    pdf.showPage()

    # Page 3: actual fitting and motion-loss audit, prior V03 reuse, boundaries.
    y = begin_page(3, "Spatial Updates and Verification", "Direct, IoU-OPD and Motion-OPD; no temporal refit")
    y = add_para(
        f"The fixed scope uses 1,792 parameters and five declared rounds per actual fit. "
        f"There are {fit['actual_fit_count']} actual fits, {fit['no_fit_placeholder_count']} exact-F no-update placeholders, "
        f"{fit['declared_rounds']} declared fit rounds and {fit['actual_optimizer_steps']} actual Adam steps. "
        f"The student barrier records {fit['actual_full_final_model_readouts']} full final-model readouts. "
        "Motion-OPD adds the fixed adjacent displacement loss with lambda_motion = 1.0.", y, body, 7)
    y = add_para("Per-case fit movement and optimizer steps", y, heading, 4)
    move = defaultdict(dict)
    for row in report["fit_movement_rows"]:
        move[row["case_id"]][row["fit"]] = row
    mtab = [["Case", "Eligible", "D steps", "IoU steps", "Motion steps", "D param delta", "IoU param delta", "Motion param delta"]]
    for cid in CASE_IDS:
        case_teacher = next(c for c in teacher["cases"] if c["case_id"] == _case_alias(cid))
        by = move[cid]
        mtab.append([
            _case_alias(cid), "yes" if case_teacher.get("eligible_for_update") else "no / reuse F",
            *[str(by.get(f"teacher_{a}", {}).get("optimizer_steps", 0)) for a in ALGORITHMS],
            *[_number(by.get(f"teacher_{a}", {}).get("parameter_delta_norm")) for a in ALGORITHMS],
        ])
    y = add_table(mtab, [65, 105, 62, 62, 70, 105, 105, 105], y, 5.4, 6.2)
    y = add_para("Motion-OPD loss readback by round", y, heading, 4)
    rtab = [["Round", "Fits", "Mean OPD", "Mean motion", "Mean total", "Updated", "Valid pairs"]]
    for row in fit["motion_iou_round_summary"]:
        rtab.append([str(row["round"]), str(row["fit_round_count"]), _number(row["mean_opd_loss"]),
                     _number(row["mean_motion_loss"]), _number(row["mean_total_loss"]),
                     str(row["optimizer_updated_count"]), str(row["motion_pair_count_total"])])
    y = add_table(rtab, [65, 75, 100, 100, 100, 85, 100], y, 5.7, 6.5)
    v03 = report["V03_prior_control"]
    v03_groups = v03.get("groups", [])
    if v03_groups:
        y = add_para(f"V03 prior-control evidence: exact reuse; {v03.get('new_fits_executed')} new fits.", y, heading, 4)
        vtab = [["Prior arm", "Cases", "vIoU", "tIoU", "dense sIoU"]]
        for row in v03_groups:
            m = row.get("metrics_percent", {})
            vtab.append([str(row.get("arm", "")), str(row.get("scored_case_count", "")),
                         _number(m.get("m_vIoU")), _number(m.get("m_tIoU")), _number(m.get("m_sIoU_dense_GT"))])
        y = add_table(vtab, [200, 75, 100, 100, 120], y, 5.5, 6.2)
    anchor_rows = v03.get("anchor_rows", [])
    if anchor_rows:
        y = add_para("V03 locked reference anchors; missing labels remain unknown", y, heading, 3)
        atab = [["Role", "Candidate IoU", "Paired Source IoU", "Difference", "GT known/unknown", "Missing on known"]]
        for row in anchor_rows:
            atab.append([
                str(row.get("reference_role", "")),
                _number(row.get("present_anchor_GT_IoU_mean_known_only")),
                _number(row.get("paired_F_source_same_anchor_GT_IoU_mean_known_only")),
                _number(row.get("present_minus_paired_F_source_mean_IoU_known_only")),
                f"{row.get('GT_known_original4_anchor_count', 0)}/{row.get('GT_missing_original4_anchor_count', 0)}",
                str(row.get("candidate_missing_where_GT_known_count", 0)),
            ])
        y = add_table(atab, [115, 100, 110, 85, 115, 110], y, 4.8, 5.5)
    y = add_para(
        "The complete anonymous data preserve all metric and comparison rows, candidate geometry and display mappings, postseal Oracle diagnostics, teacher parsed choices/costs, fit movement and every Motion-OPD round. Raw query/response prose, source paths and hashes, absolute physical frame identifiers, GT coordinates, weights and optimizer tensors are excluded. Historical-source evidence does not establish fresh-source generalization or qualify a production route.",
        y, small, 2)
    pdf.showPage()
    pdf.save()

    reader = PdfReader(str(destination))
    _need(len(reader.pages) == 3, "generated PDF is not exactly three pages")
    for page in reader.pages:
        text = page.extract_text() or ""
        _need(bool(text.strip()), "generated PDF has an empty page")
    raw = destination.read_bytes()
    _need(b"LiberationSans" in raw or b"ReportSans" in raw,
          "generated PDF does not contain the registered LiberationSans font names")


def _public_outputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    report = inputs["report"]
    geometry = _candidate_geometry_and_display(
        inputs["input_records"], inputs["old_lk_input_records"],
        inputs["barrier"].get("elapsed_cpu_seconds"),
    )
    teacher = _teacher_controls(inputs["teacher"], inputs["input_records"])
    fit = _fit_and_motion(report, inputs["student_barrier"], inputs["actual_fits"],
                          inputs["no_fit_rows"], inputs["motion_fits"])
    metrics = _anonymous_metric_bundle(inputs["input_records"], report)
    oracle = {
        "status": "postseal_diagnostic_only",
        "missing_candidate_zero_fill": "GT-known original-four denominator only; GT-unknown excluded",
        "candidate_geometry_rows": metrics["candidate_geometry_rows"],
        "candidate_display_rows": metrics["candidate_display_rows"],
        "candidate_oracle_rows": metrics["candidate_oracle_rows"],
        "candidate_present_anchor_summary": metrics["candidate_present_anchor_summary"],
        "V03_prior_control": fit["V03_prior_control"],
    }
    manifest = {
        "status": "pass",
        "schema": "stvg_adjacent_motion_anonymous_report_v1",
        "case_aliases": {alias: "HC2" if alias.startswith("HC") else "VidSTG"
                         for alias in ALIASES.values()},
        "case_count": 8,
        "candidate_count": geometry["candidate_count"],
        "historical_LK_candidate_count": geometry["historical_lk_comparison"]["support_summary_by_group"]["all_old_candidates"]["candidate_count"],
        "tracking_preparation_elapsed_cpu_seconds": geometry["preparation_elapsed_cpu_seconds"],
        "teacher_call_count": teacher["actual_call_count"],
        "metric_row_count": metrics["metric_count"],
        "candidate_geometry_row_count": metrics["candidate_geometry_row_count"],
        "candidate_display_row_count": metrics["candidate_display_row_count"],
        "actual_fit_count": fit["actual_fit_count"],
        "no_fit_placeholder_count": fit["no_fit_placeholder_count"],
        "declared_rounds": fit["declared_rounds"],
        "actual_optimizer_steps": fit["actual_optimizer_steps"],
        "actual_full_final_model_readouts": fit["actual_full_final_model_readouts"],
        "V03_new_fits": 0,
        "private_data_fields_excluded": [
            "source/media paths and hashes", "physical frame IDs and timestamps",
            "GT coordinates", "query and raw response prose", "model weights and optimizer tensors",
        ],
    }
    return {
        "metric_and_case_rows.json": metrics,
        "candidate_geometry_and_display.json": geometry,
        "teacher_control_readback.json": teacher,
        "fit_motion_and_oracle_readback.json": {
            "fit_and_motion": fit,
            "candidate_oracle": oracle,
        },
        "report_manifest.json": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true",
                        help="replace existing report outputs after an explicit correction")
    args = parser.parse_args()
    inputs = _load_inputs()
    outputs = _public_outputs(inputs)
    markdown = _markdown(inputs["input_records"], outputs["candidate_geometry_and_display.json"],
                         outputs["teacher_control_readback.json"], outputs["fit_motion_and_oracle_readback.json"]["fit_and_motion"],
                         outputs["metric_and_case_rows.json"], inputs["report"])

    destinations = [DOC, PDF, *(ANON / name for name in outputs)]
    if not args.replace:
        existing = [str(path.relative_to(ROOT)) for path in destinations if path.exists()]
        _need(not existing, f"report outputs already exist; pass --replace only for a correction: {existing}")
    for name, payload in outputs.items():
        path = ANON / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(markdown, encoding="utf-8")
    PDF.parent.mkdir(parents=True, exist_ok=True)
    _pdf(inputs["input_records"], outputs["candidate_geometry_and_display.json"],
         outputs["teacher_control_readback.json"], outputs["fit_motion_and_oracle_readback.json"]["fit_and_motion"],
         inputs["report"], PDF)
    _audit_public_outputs(destinations)
    print(json.dumps({
        "status": "pass",
        "case_count": 8,
        "metric_rows": len(inputs["report"]["metric_rows"]),
        "candidate_geometry_rows": len(inputs["report"]["candidate_geometry_rows"]),
        "candidate_display_rows": len(inputs["report"]["candidate_display_rows"]),
        "actual_fits": outputs["fit_motion_and_oracle_readback.json"]["fit_and_motion"]["actual_fit_count"],
        "optimizer_steps": outputs["fit_motion_and_oracle_readback.json"]["fit_and_motion"]["actual_optimizer_steps"],
        "full_readouts": outputs["fit_motion_and_oracle_readback.json"]["fit_and_motion"]["actual_full_final_model_readouts"],
        "pdf_pages": 3,
    }, indent=2))


def _audit_public_outputs(paths: Sequence[Path]) -> None:
    private_case_pattern = re.compile(r"(?:hc2|vidstg)_\d{3}")
    hash_pattern = re.compile(r"\b[0-9a-f]{64}\b", re.I)
    forbidden_key_fragments = (
        "physical_frame_id", "frame_ids", "source_case_path", "source_video_path",
        "prompt", "raw_output", "raw_response", "teacher_response", "optimizer_state",
        "state_dict", "source_case_sha256", "source_model_state_sha256", "source_pixel_sha256",
    )

    def walk(value: Any, where: str = "root") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                low = str(key).lower()
                _need(not any(fragment in low for fragment in forbidden_key_fragments),
                      f"private key survived public export at {where}.{key}")
                walk(child, f"{where}.{key}")
        elif isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, f"{where}[{i}]")
        elif isinstance(value, str):
            _need(not private_case_pattern.search(value), f"private case ID survived public export at {where}")
            _need(not hash_pattern.search(value), f"hash survived public export at {where}")
            _need("/home/" not in value and "/attachments/" not in value,
                  f"private path survived public export at {where}")

    for path in paths:
        if path.suffix == ".json":
            walk(_read(path), str(path.name))
        elif path.suffix == ".md":
            content = path.read_text(encoding="utf-8")
            _need(not private_case_pattern.search(content), "private case ID survived Markdown export")
            _need(not hash_pattern.search(content), "hash survived Markdown export")
            _need("/home/" not in content and "/attachments/" not in content,
                  "private path survived Markdown export")


if __name__ == "__main__":
    main()
