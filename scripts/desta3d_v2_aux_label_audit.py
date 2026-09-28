#!/usr/bin/env python3
"""Source-train-only audit of PTD time labels, sample quantization, and panels.

This script reads only the already materialized source-training records, their
source-only time-endpoint audit, and the eight saved source gradient reports.
It never opens source-validation or target labels, videos/pixels, model weights,
or CUDA.  Outputs go to one new, fixed audit directory and existing files are
never overwritten.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/desta3d_v2/aux_backflow_v1/label_audit"
RECORDS = ROOT / "artifacts/desta3d_v1/source_fit/SOURCE_TRAIN_RECORDS.json"
ENDPOINT_AUDIT = ROOT / "artifacts/desta3d_v2/SOURCE_TIME_ENDPOINT_AUDIT.json"
GRADIENT_DIR = ROOT / "artifacts/desta3d_v2/gradient_scale_panel_v1"
SOURCE_INVENTORY = ROOT / "artifacts/desta3d_v2/SOURCE_INVENTORY.json"
SOURCE_PREP = ROOT / "scripts/desta3d_source_fit_v1.py"
OFFICIAL_BUILDER = ROOT / "external/ParallelTubeDecoding/data/prepare_vidstg.py"
ATTACHMENT = Path(
    "./private_authorization_notes/authorization.txt"
)

PANEL_SALT = "desta3d-v2-aux-label-audit-panel-20260927"
PAIR_LIMIT = 8

TIME_SPAN_RE = re.compile(r"<\|time_start\|><t(\d+)><t(\d+)><\|time_end\|>")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def parse_time_span(response: str | None) -> list[int] | None:
    if not isinstance(response, str):
        return None
    matches = TIME_SPAN_RE.findall(response)
    if len(matches) != 1:
        return None
    start, end = map(int, matches[0])
    return [start, end]


def active_from_physical_grid(frame_ids: list[int], begin_fid: int, end_fid: int) -> list[bool]:
    """The source-prep contract uses the half-open physical interval [begin,end)."""
    return [begin_fid <= int(fid) < end_fid for fid in frame_ids]


def record_facts(record: dict[str, Any]) -> dict[str, Any]:
    frame_ids = [int(x) for x in record["frame_ids"]]
    begin = int(record["event_interval"]["begin_fid"])
    end = int(record["event_interval"]["end_fid"])
    physical_active = active_from_physical_grid(frame_ids, begin, end)
    active = [bool(x) for x in record["event_active"]]
    valid = [bool(x) for x in record["box_valid"]]
    if not (len(frame_ids) == len(active) == len(valid) == len(record["boxes_xyxy"])):
        raise ValueError(f"length mismatch in {record.get('key')}")
    active_positions = [i + 1 for i, value in enumerate(active) if value]
    active_valid_positions = [
        i + 1 for i, (a, v) in enumerate(zip(active, valid)) if a and v
    ]
    active_missing_positions = [
        i + 1 for i, (a, v) in enumerate(zip(active, valid)) if a and not v
    ]
    response_positions = [int(x) for x in record.get("response_box_positions_1based", [])]
    response_span = parse_time_span(record.get("response"))
    internal_missing = [
        pos for pos in active_missing_positions
        if active_positions and active_positions[0] < pos < active_positions[-1]
    ]
    boundary_missing = [
        pos for pos in active_missing_positions
        if active_positions and pos in (active_positions[0], active_positions[-1])
    ]
    response_gap = None
    if response_positions:
        expected_positions = list(range(response_positions[0], response_positions[-1] + 1))
        if expected_positions != response_positions:
            response_gap = [x for x in expected_positions if x not in set(response_positions)]
    physical_sample_mismatch = active != physical_active

    boxes = record["boxes_xyxy"]
    areas = []
    invalid_valid_boxes = []
    for pos, (box, is_valid) in enumerate(zip(boxes, valid), start=1):
        if not is_valid:
            continue
        if len(box) != 4 or any(not math.isfinite(float(v)) for v in box):
            invalid_valid_boxes.append(pos)
            continue
        x0, y0, x1, y1 = map(float, box)
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            invalid_valid_boxes.append(pos)
            continue
        if pos in active_valid_positions:
            areas.append((x1 - x0) * (y1 - y0))
    mean_active_area = sum(areas) / len(areas) if areas else None

    eligible = bool(record.get("response_eligible"))
    response_none = record.get("response") is None
    response_span_matches_active = bool(
        response_span is not None
        and active_positions
        and response_span == [active_positions[0], active_positions[-1]]
    )
    response_span_matches_valid = bool(
        response_span is not None
        and active_valid_positions
        and response_span == [active_valid_positions[0], active_valid_positions[-1]]
    )
    response_positions_match_valid = response_positions == active_valid_positions
    contiguous_response = bool(
        response_positions
        and response_positions == list(range(response_positions[0], response_positions[-1] + 1))
    )
    left_sample = max((fid for fid in frame_ids if fid < begin), default=None)
    right_sample = min((fid for fid in frame_ids if fid >= end), default=None)

    return {
        "key": record["key"],
        "source": str(record["source"]),
        "split": record.get("split"),
        "video_sha256": record.get("video_sha256"),
        "frame_ids": frame_ids,
        "fps": float(record["fps"]),
        "event_interval": {"begin_fid": begin, "end_fid": end},
        "duration_seconds_half_open": (end - begin) / float(record["fps"]),
        "frame_count": int(record["frame_count"]),
        "physical_active_matches_record": not physical_sample_mismatch,
        "active_positions_1based": active_positions,
        "active_valid_positions_1based": active_valid_positions,
        "active_missing_positions_1based": active_missing_positions,
        "boundary_missing_positions_1based": boundary_missing,
        "internal_missing_positions_1based": internal_missing,
        "response_eligible": eligible,
        "response_is_none": response_none,
        "response_time_span_1based": response_span,
        "response_box_positions_1based": response_positions,
        "response_positions_match_active_valid": response_positions_match_valid,
        "response_span_matches_active_first_last": response_span_matches_active,
        "response_span_matches_active_valid_first_last": response_span_matches_valid,
        "response_positions_contiguous": contiguous_response,
        "response_gap_positions_1based": response_gap,
        "response_tokenization_dropped_valid_active_positions_1based": [
            p for p in active_valid_positions if p not in set(response_positions)
        ],
        "mean_normalized_active_box_area": mean_active_area,
        "invalid_box_valid_positions_1based": invalid_valid_boxes,
        "left_sample_before_interval_fid": left_sample,
        "right_sample_at_or_after_interval_fid": right_sample,
        "start_sample_quantization_frames": (
            frame_ids[active_positions[0] - 1] - begin if active_positions else None
        ),
        "end_sample_quantization_frames": (
            (end - 1) - frame_ids[active_positions[-1] - 1] if active_positions else None
        ),
    }


def quartile_ranks(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    ordered = sorted(rows, key=lambda r: (float(r[field]), r["key"]))
    n = len(ordered)
    return {r["key"]: min(3, (rank * 4) // n) + 1 for rank, r in enumerate(ordered)}


def make_panel(eligible_facts: list[dict[str, Any]]) -> dict[str, Any]:
    duration_q = quartile_ranks(eligible_facts, "duration_seconds_half_open")
    area_q = quartile_ranks(eligible_facts, "mean_normalized_active_box_area")
    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in eligible_facts:
        cells[(duration_q[r["key"]], area_q[r["key"]])].append(r)
    selected = []
    used_sources: set[str] = set()
    cell_rows = []
    for qd in range(1, 5):
        for qa in range(1, 5):
            candidates = sorted(
                cells.get((qd, qa), []),
                key=lambda r: hashlib.sha256(f"{PANEL_SALT}|{r['key']}".encode()).hexdigest(),
            )
            choice = None
            skipped_for_parent = 0
            for candidate in candidates:
                if candidate["source"] not in used_sources:
                    choice = candidate
                    break
                skipped_for_parent += 1
            cell_rows.append({
                "duration_quartile": qd,
                "area_quartile": qa,
                "candidate_queries": len(candidates),
                "selected_key": choice["key"] if choice else None,
                "selected_source": choice["source"] if choice else None,
                "skipped_due_to_parent_already_selected": skipped_for_parent,
            })
            if choice:
                used_sources.add(choice["source"])
                selected.append({
                    "key": choice["key"],
                    "source": choice["source"],
                    "duration_quartile": qd,
                    "area_quartile": qa,
                    "duration_seconds_half_open": choice["duration_seconds_half_open"],
                    "mean_normalized_active_box_area": choice["mean_normalized_active_box_area"],
                    "event_interval": choice["event_interval"],
                    "frame_ids": choice["frame_ids"],
                    "fps": choice["fps"],
                })
    return {
        "selection_rule": {
            "population": "612 response_eligible source-train records only",
            "duration": "(end_fid - begin_fid) / fps using the source half-open physical event interval",
            "area": "mean normalized xyxy area over sampled event_active && box_valid frames",
            "quartiles": "rank quartiles over the 612 eligible rows, sorting by (value,key), rank bin floor(4*rank/N)+1; ties are deterministic by key",
            "one_per_cell": "row-major duration quartile then area quartile; within a cell sort by sha256(PANEL_SALT|key); take the first source parent not already selected",
            "panel_salt": PANEL_SALT,
            "outcome_independence": "Selection uses only source metadata/labels needed for stratification; no model outputs, losses, gradients, checkpoints, or validation/target labels.",
            "max_independent_parents": 16,
        },
        "eligible_population": len(eligible_facts),
        "panel_size": len(selected),
        "independent_sources": len({r["source"] for r in selected}),
        "keys": [r["key"] for r in selected],
        "cell_summary": cell_rows,
        "selected_queries": selected,
    }


def make_pair_register(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [r for r in records if r.get("response_eligible")]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in eligible:
        by_source[str(r["source"])].append(r)
    pairs = []
    for source in sorted(by_source):
        rows = sorted(by_source[source], key=lambda r: r["key"])
        found = None
        for a, b in itertools.combinations(rows, 2):
            ia, ib = a["event_interval"], b["event_interval"]
            # Distinct, temporally disjoint half-open physical intervals are a
            # conservative metadata-only definition of different event times.
            if ia["end_fid"] <= ib["begin_fid"] or ib["end_fid"] <= ia["begin_fid"]:
                found = (a, b)
                break
        if found:
            a, b = found
            fa = [int(x) for x in a["frame_ids"]]
            fb = [int(x) for x in b["frame_ids"]]
            sa, sb = set(fa), set(fb)
            common = sorted(sa & sb)
            union = sa | sb
            same_video = a.get("video_sha256") == b.get("video_sha256")
            same_fps = float(a["fps"]) == float(b["fps"])
            exact_grid = same_video and same_fps and fa == fb
            pairs.append({
                "source": source,
                "query_a": {
                    "key": a["key"],
                    "caption": a["caption"],
                    "event_interval_half_open_fid": a["event_interval"],
                    "response_eligible": bool(a["response_eligible"]),
                },
                "query_b": {
                    "key": b["key"],
                    "caption": b["caption"],
                    "event_interval_half_open_fid": b["event_interval"],
                    "response_eligible": bool(b["response_eligible"]),
                },
                "same_video_sha256": same_video,
                "fps_a": float(a["fps"]),
                "fps_b": float(b["fps"]),
                "same_fps": same_fps,
                "frame_grid_a_count": len(fa),
                "frame_grid_b_count": len(fb),
                "frame_ids_exactly_equal": fa == fb,
                "common_physical_frame_ids": common,
                "common_frame_count": len(common),
                "union_frame_count": len(union),
                "frame_grid_jaccard": len(common) / len(union) if union else None,
                "physical_time_grid_exact_match": exact_grid,
                "selection_rule": "lexicographically first response-eligible disjoint event-interval pair for this source; source IDs in lexical order; at most one pair per source",
                "interpretation": "Metadata pairing aid only. Non-overlapping intervals do not establish semantic exclusivity; no query is assigned a negative label and mismatching queries are not treated as negatives.",
            })
            if len(pairs) >= PAIR_LIMIT:
                break
    return {
        "selection_rule": "first at most 8 lexicographically ordered source IDs with at least one pair of response-eligible, half-open, non-overlapping physical event intervals; choose lexicographically first key pair per parent",
        "candidate_scope": "618 source-train records; only response-eligible records are pair candidates",
        "candidate_sources_with_disjoint_eligible_interval_pair": len({
            source for source, rows in by_source.items()
            if any(
                x["event_interval"]["end_fid"] <= y["event_interval"]["begin_fid"]
                or y["event_interval"]["end_fid"] <= x["event_interval"]["begin_fid"]
                for x, y in itertools.combinations(rows, 2)
            )
        }),
        "registered_pair_count": len(pairs),
        "pairs": pairs,
        "negative_label_warning": "These are not contrastive negatives. Queries may share a referent or otherwise co-occur; only physical event intervals are known to be disjoint.",
    }


def recompute_gradient_direction() -> dict[str, Any]:
    rows = read_json(GRADIENT_DIR / "INPUTS.json")
    expected_summary = read_json(GRADIENT_DIR / "SUMMARY.json")
    reports = []
    for index, input_row in enumerate(rows):
        report_path = GRADIENT_DIR / "queries" / f"{index:03d}" / "SOURCE_GRADIENT_DECOMPOSITION.json"
        result = read_json(report_path)
        if result["key"] != input_row["key"] or str(result["source"]) != str(input_row["source"]):
            raise AssertionError(f"gradient report/input identity mismatch: {report_path}")
        n = result["norms"]
        r = float(result["B_weighted_aux_to_sum_task_norm_ratio"])
        cosine = float(result["angles"]["task_sum__aux_unweighted"]["cosine"])
        recomputed_r = float(n["aux_at_B_weight"]) / float(n["task_sum"])
        first_order_factor = 1.0 + r * cosine
        reports.append({
            "key": result["key"],
            "source": str(result["source"]),
            "ratio_reported_weighted_aux_over_task": r,
            "ratio_recomputed_from_norms": recomputed_r,
            "ratio_abs_error": abs(r - recomputed_r),
            "task_sum_aux_cosine": cosine,
            "first_order_direction_factor_1_plus_r_cos": first_order_factor,
            "negative_local_combined_gradient_factor": first_order_factor < 0,
            "parameter_scope": result["parameter_scope"],
            "parameter_count": result["parameters"],
            "optimizer_steps": result["optimizer_steps"],
            "source_GT_used": result["source_GT_used"],
            "target_GT_read": result["target_GT_read"],
        })
    if len(reports) != 8 or len({r["source"] for r in reports}) != 8:
        raise AssertionError("expected eight unique source parents in saved gradient panel")
    factors = [r["first_order_direction_factor_1_plus_r_cos"] for r in reports]
    ratios = [r["ratio_reported_weighted_aux_over_task"] for r in reports]
    return {
        "inputs": {
            "gradient_inputs_sha256": sha256(GRADIENT_DIR / "INPUTS.json"),
            "gradient_summary_sha256": sha256(GRADIENT_DIR / "SUMMARY.json"),
            "gradient_reports_sha256": {
                str(GRADIENT_DIR / "queries" / f"{i:03d}" / "SOURCE_GRADIENT_DECOMPOSITION.json"):
                    sha256(GRADIENT_DIR / "queries" / f"{i:03d}" / "SOURCE_GRADIENT_DECOMPOSITION.json")
                for i in range(len(rows))
            },
        },
        "definition": "g_task is event PTD task plus spatial PTD task on the same 19,968 shared-stem parameters; g_aux is 0.1 times the referent+event auxiliary loss gradient on those same parameters. Because the coefficient is positive, cosine(g_task,g_aux_weighted)=cosine(g_task,g_aux_unweighted). The checked scalar is (g_task dot (g_task + g_aux_weighted))/||g_task||^2 = 1 + r*cos.",
        "source_queries": len(reports),
        "independent_sources": len({r["source"] for r in reports}),
        "all_adapter_unchanged": bool(expected_summary["all_adapter_unchanged"]),
        "optimizer_steps": int(expected_summary["optimizer_steps"]),
        "ratio_weighted_aux_to_task_norm": {
            "min": min(ratios),
            "median": sum(sorted(ratios)[len(ratios) // 2 - 1:len(ratios) // 2 + 1]) / 2.0,
            "max": max(ratios),
        },
        "first_order_factor_negative_count": sum(f < 0 for f in factors),
        "first_order_factor_nonnegative_count": sum(f >= 0 for f in factors),
        "per_query": reports,
        "scope_limit": "This algebra is a local raw-gradient directional derivative on the measured shared stem. It is not an Adam/AdamW parameter update, trajectory effect, loss-after-step, full-model effect, or vIoU conclusion.",
    }


def counterexample_tests() -> dict[str, Any]:
    tests = []

    def case(name: str, frame_ids: list[int], begin: int, end: int,
             valid_positions: list[int], expected: dict[str, Any]) -> None:
        active = active_from_physical_grid(frame_ids, begin, end)
        active_positions = [i + 1 for i, x in enumerate(active) if x]
        missing = [p for p in active_positions if p not in valid_positions]
        boundary_missing = [p for p in missing if p in (active_positions[0], active_positions[-1])] if active_positions else []
        internal_missing = [p for p in missing if active_positions[0] < p < active_positions[-1]] if active_positions else []
        response_indices = sorted(valid_positions)
        contiguous = bool(response_indices and response_indices == list(range(response_indices[0], response_indices[-1] + 1)))
        observed = {
            "active_positions_1based": active_positions,
            "missing_active_box_positions_1based": missing,
            "boundary_missing_positions_1based": boundary_missing,
            "internal_missing_positions_1based": internal_missing,
            "response_span_if_official_builder_accepts": [response_indices[0], response_indices[-1]] if contiguous else None,
            "official_builder_contiguous_indices_condition": contiguous,
        }
        if observed != expected:
            raise AssertionError(f"counterexample {name}: observed={observed}, expected={expected}")
        tests.append({"name": name, "passed": True, "observed": observed})

    case(
        "half_open_boundary_and_quantized_response_positions",
        [9, 10, 15, 19, 20], 10, 20, [2, 3, 4],
        {"active_positions_1based": [2, 3, 4], "missing_active_box_positions_1based": [],
         "boundary_missing_positions_1based": [], "internal_missing_positions_1based": [],
         "response_span_if_official_builder_accepts": [2, 4],
         "official_builder_contiguous_indices_condition": True},
    )
    case(
        "missing_event_boundary_box_changes_builder_time_endpoint",
        [9, 10, 15, 19, 20], 10, 20, [3, 4],
        {"active_positions_1based": [2, 3, 4], "missing_active_box_positions_1based": [2],
         "boundary_missing_positions_1based": [2], "internal_missing_positions_1based": [],
         "response_span_if_official_builder_accepts": [3, 4],
         "official_builder_contiguous_indices_condition": True},
    )
    case(
        "internal_missing_box_breaks_official_contiguous_response",
        [9, 10, 15, 19, 20], 10, 20, [2, 4],
        {"active_positions_1based": [2, 3, 4], "missing_active_box_positions_1based": [3],
         "boundary_missing_positions_1based": [], "internal_missing_positions_1based": [3],
         "response_span_if_official_builder_accepts": None,
         "official_builder_contiguous_indices_condition": False},
    )
    case(
        "short_physical_event_missed_by_sampling_grid_has_no_ce_span",
        [9, 20], 10, 19, [],
        {"active_positions_1based": [], "missing_active_box_positions_1based": [],
         "boundary_missing_positions_1based": [], "internal_missing_positions_1based": [],
         "response_span_if_official_builder_accepts": None,
         "official_builder_contiguous_indices_condition": False},
    )
    return {
        "cpu_only": True,
        "no_torch_no_cuda_no_data_loading": True,
        "cases": tests,
        "interpretation": "Synthetic cases distinguish boundary-box endpoint shrinkage, internal-gap rejection by the official builder's contiguous-index guard, and an event interval with no sampled active frame. They validate the audit classification logic only; they do not create or modify source labels.",
    }


def summarize(records: list[dict[str, Any]], facts: list[dict[str, Any]], endpoint: dict[str, Any]) -> dict[str, Any]:
    if len(records) != 618 or len({r["key"] for r in records}) != 618:
        raise AssertionError(f"expected 618 unique source-train records, got {len(records)}")
    if {r.get("split") for r in records} != {"train"}:
        raise AssertionError("SOURCE_TRAIN_RECORDS contains a non-train split")
    if len({str(r["source"]) for r in records}) != 95:
        raise AssertionError("expected 95 source-train parents")
    fact_by_key = {r["key"]: r for r in facts}
    eligible = [f for f in facts if f["response_eligible"]]
    ineligible = [f for f in facts if not f["response_eligible"]]
    endpoint_rows = endpoint.get("rows", [])
    endpoint_keys = {r["key"] for r in endpoint_rows}
    eligible_keys = {r["key"] for r in eligible}
    if endpoint_keys != eligible_keys:
        raise AssertionError("existing 612-row endpoint audit key set differs from current 612 eligible records")
    endpoint_map = {r["key"]: r for r in endpoint_rows}
    prior_audit_rechecks = 0
    for f in eligible:
        row = endpoint_map[f["key"]]
        active = f["active_positions_1based"]
        span = [active[0], active[-1]]
        if row["event_sample_interval"] != span:
            raise AssertionError(f"existing endpoint audit event span mismatch: {f['key']}")
        if row["response_sample_interval"] != f["response_time_span_1based"]:
            raise AssertionError(f"existing endpoint audit response span mismatch: {f['key']}")
        prior_audit_rechecks += 1

    event_mask_mismatches = [f["key"] for f in facts if not f["physical_active_matches_record"]]
    eligible_missing_boxes = [f for f in eligible if f["active_missing_positions_1based"]]
    boundary_missing = [f for f in facts if f["boundary_missing_positions_1based"]]
    internal_missing = [f for f in facts if f["internal_missing_positions_1based"]]
    eligible_bad = [
        f for f in eligible
        if not f["response_span_matches_active_first_last"]
        or not f["response_span_matches_active_valid_first_last"]
        or not f["response_positions_match_active_valid"]
        or not f["response_positions_contiguous"]
        or f["response_tokenization_dropped_valid_active_positions_1based"]
        or f["invalid_box_valid_positions_1based"]
    ]
    no_ce_with_sampled_active = [f for f in ineligible if f["active_positions_1based"]]
    no_ce_quantized = [f for f in ineligible if not f["active_positions_1based"]]
    start_offsets = [f["start_sample_quantization_frames"] for f in eligible]
    end_offsets = [f["end_sample_quantization_frames"] for f in eligible]
    durations = sorted(f["duration_seconds_half_open"] for f in eligible)

    return {
        "scope": "source-train label/response audit only",
        "source_GT_used": True,
        "source_validation_GT_read": False,
        "target_GT_read": False,
        "pixels_or_video_opened": False,
        "gpu_or_model_forward": False,
        "record_count": len(records),
        "unique_parents": len({str(r["source"]) for r in records}),
        "response_eligible_count": len(eligible),
        "response_ineligible_count": len(ineligible),
        "existing_endpoint_audit": {
            "path": str(ENDPOINT_AUDIT.relative_to(ROOT)),
            "declared_eligible_queries": endpoint.get("eligible_queries"),
            "declared_parents": endpoint.get("parents"),
            "declared_endpoint_mismatches": endpoint.get("endpoint_mismatches"),
            "records_rechecked": prior_audit_rechecks,
            "covers_only_eligible": True,
            "does_not_establish_CE_for_all_618": True,
        },
        "physical_interval_contract": "Source preparation defines active as begin_fid <= sampled frame_id < end_fid; response <tN> indices are 1-based sampled positions enumerated by source preparation and emitted from first/last active valid box indices by the official builder.",
        "physical_event_mask_mismatch_count": len(event_mask_mismatches),
        "physical_event_mask_mismatch_keys": event_mask_mismatches,
        "eligible_time_span_matches_first_last_sampled_event_active": sum(
            f["response_span_matches_active_first_last"] for f in eligible
        ),
        "eligible_time_span_matches_first_last_active_valid_box": sum(
            f["response_span_matches_active_valid_first_last"] for f in eligible
        ),
        "eligible_response_box_positions_equal_all_active_valid_positions": sum(
            f["response_positions_match_active_valid"] for f in eligible
        ),
        "eligible_response_positions_contiguous": sum(f["response_positions_contiguous"] for f in eligible),
        "eligible_response_time_or_box_encoding_issue_count": len(eligible_bad),
        "eligible_response_issue_keys": [f["key"] for f in eligible_bad],
        "eligible_records_with_any_sampled_active_missing_box": len(eligible_missing_boxes),
        "sampled_active_missing_box_frame_count": sum(len(f["active_missing_positions_1based"]) for f in facts),
        "records_with_boundary_active_missing_box": len(boundary_missing),
        "boundary_active_missing_box_frame_count": sum(len(f["boundary_missing_positions_1based"]) for f in facts),
        "records_with_internal_active_missing_box": len(internal_missing),
        "internal_active_missing_box_frame_count": sum(len(f["internal_missing_positions_1based"]) for f in facts),
        "noneligible_with_any_sampled_active_frame": len(no_ce_with_sampled_active),
        "noneligible_with_no_sampled_active_frame_sampling_quantization": len(no_ce_quantized),
        "noneligible_quantized_examples": [
            {
                "key": f["key"],
                "source": f["source"],
                "event_interval_half_open_fid": f["event_interval"],
                "duration_seconds_half_open": f["duration_seconds_half_open"],
                "sampled_positions_active": f["active_positions_1based"],
                "nearest_sample_before_begin_fid": f["left_sample_before_interval_fid"],
                "nearest_sample_at_or_after_end_fid": f["right_sample_at_or_after_interval_fid"],
                "box_valid_active_positions": f["active_valid_positions_1based"],
                "response_is_none": f["response_is_none"],
            }
            for f in no_ce_quantized
        ],
        "sample_endpoint_quantization_frames_for_612_eligible": {
            "start_offset_min": min(start_offsets) if start_offsets else None,
            "start_offset_median": sorted(start_offsets)[len(start_offsets) // 2] if start_offsets else None,
            "start_offset_max": max(start_offsets) if start_offsets else None,
            "end_offset_min": min(end_offsets) if end_offsets else None,
            "end_offset_median": sorted(end_offsets)[len(end_offsets) // 2] if end_offsets else None,
            "end_offset_max": max(end_offsets) if end_offsets else None,
        },
        "eligible_physical_duration_seconds_quantiles_rank": {
            "min": min(durations) if durations else None,
            "q25": durations[len(durations) // 4] if durations else None,
            "median": durations[len(durations) // 2] if durations else None,
            "q75": durations[(3 * len(durations)) // 4] if durations else None,
            "max": max(durations) if durations else None,
        },
        "other_issues": {
            "eligible_response_missing_or_malformed": sum(f["response_time_span_1based"] is None for f in eligible),
            "eligible_valid_box_outside_normalized_xyxy_contract": sum(bool(f["invalid_box_valid_positions_1based"]) for f in eligible),
            "noneligible_with_sampled_active_but_no_response": len(no_ce_with_sampled_active),
            "classification": "No other source record/response discrepancy found under these checks." if not event_mask_mismatches and not eligible_bad and not no_ce_with_sampled_active else "See nonzero issue fields above.",
        },
        "per_record": facts,
    }


def render_report(summary: dict[str, Any], panel: dict[str, Any], pair_register: dict[str, Any], gradient: dict[str, Any], hashes: dict[str, Any]) -> str:
    neg = [x for x in gradient["per_query"] if x["negative_local_combined_gradient_factor"]]
    ratio_summary = gradient["ratio_weighted_aux_to_task_norm"]
    neg_text = ", ".join(
        f"{x['key']} ({x['first_order_direction_factor_1_plus_r_cos']:.4f})" for x in neg
    ) if neg else "none"
    lines = [
        "# DESTA-3D v2 source auxiliary-label audit",
        "",
        "This is a source-train-only CPU audit. It reads the saved 618-row source-training record JSON and existing source-only audit/gradient summaries; it opens no video pixels, model weights, source-validation labels, or target labels, and runs no GPU or model inference.",
        "",
        "## Time-label result",
        "",
        f"- The dataset contains **{summary['record_count']} source-train queries from {summary['unique_parents']} parents**: {summary['response_eligible_count']} have a legal PTD response/CE sequence and {summary['response_ineligible_count']} do not.",
        f"- On all records, the stored `event_active` mask exactly matches sampled `frame_ids` under the source-prep half-open physical interval `[begin_fid, end_fid)`: **{summary['physical_event_mask_mismatch_count']} mismatches**.",
        f"- For the **612 eligible queries**, the PTD `<tN>` response endpoints match both first/last sampled active position and first/last active valid-box position in **{summary['eligible_time_span_matches_first_last_sampled_event_active']}/612**; response box positions equal all active valid-box positions and are contiguous in **{summary['eligible_response_box_positions_equal_all_active_valid_positions']}/612** and **{summary['eligible_response_positions_contiguous']}/612**. Existing 612-row endpoint audit keys and endpoint values were rechecked against these records.",
        f"- Among eligible queries, sampled active missing boxes: **{summary['sampled_active_missing_box_frame_count']} frames across {summary['eligible_records_with_any_sampled_active_missing_box']} queries**; boundary cases **{summary['records_with_boundary_active_missing_box']}**, internal gaps **{summary['records_with_internal_active_missing_box']}**. Response/time/box encoding issues: **{summary['eligible_response_time_or_box_encoding_issue_count']}**.",
        f"- The six ineligible queries have **no sampled active position at all**, so the official response builder has no valid time/box sequence and source CE is absent. The physical intervals are short and fall between the fixed sampled frame IDs; they are sampling-quantization misses, not evidence that the six contain a sampled CE or that event labels are absent. There are {summary['noneligible_with_any_sampled_active_frame']} ineligible records with any sampled active position.",
        "- The 612-row result must not be generalized to all 618 as ‘all have CE’. The exact denominator is 612 eligible CE records plus 6 physically annotated intervals missed by the sample grid.",
        "",
        "## Fixed source panel and within-video pairs",
        "",
        f"- The metadata-only stratified panel contains **{panel['panel_size']} queries from {panel['independent_sources']} independent parents** (maximum 16), selected without model outcomes. Duration is physical half-open interval seconds; area is mean normalized `xyxy` area over sampled active valid boxes. Quartiles use deterministic `(value,key)` ranks and cell choice uses a fixed SHA-256 key salt.",
        f"- The pair register contains **{pair_register['registered_pair_count']}** same-video pairs with non-overlapping physical event intervals. Physical frame-grid identity was explicitly recorded (same video digest, same fps, and exactly equal sampled frame-ID lists). These are metadata pairs only: no mismatching query is treated as a negative, and interval disjointness does not establish semantic exclusivity.",
        "",
        "## Saved eight-query gradient algebra",
        "",
        f"- Recomputed `(g_task · (g_task + g_aux_weighted)) / ||g_task||² = 1 + r*cos` for {gradient['source_queries']} saved source queries on the same shared-stem parameter group. The weighted-aux/task norm ratio range is {ratio_summary['min']:.3g}–{ratio_summary['max']:.3g} (median {ratio_summary['median']:.3g}); {gradient['first_order_factor_negative_count']}/8 factors are negative: {neg_text}.",
        "- The factor is a raw local gradient directional derivative only. The source-gradient audit reports zero optimizer steps; these values are not Adam/AdamW updates, not post-step task loss, and not vIoU or full-model conclusions.",
        "",
        "## Reproducibility",
        "",
        f"- Audit script SHA-256: `{hashes['audit_script_sha256']}`.",
        f"- Source train records SHA-256: `{hashes['inputs']['source_train_records']['sha256']}`.",
        f"- Existing endpoint audit SHA-256: `{hashes['inputs']['existing_source_time_endpoint_audit']['sha256']}`.",
        "- Machine-readable results: `CPU_SUMMARY.json`, `PANEL.json`, `SAME_VIDEO_EVENT_PAIRS.json`, `GRADIENT_RECOMPUTE.json`, `COUNTEREXAMPLE_CPU_TESTS.json`, and `INPUTS.json` in this directory.",
        "",
    ]
    return "\n".join(lines)


def run() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit directory: {OUT}")
    records = read_json(RECORDS)
    endpoint = read_json(ENDPOINT_AUDIT)
    facts = [record_facts(r) for r in records]
    summary = summarize(records, facts, endpoint)
    eligible = [r for r in facts if r["response_eligible"]]
    panel = make_panel(eligible)
    pair_register = make_pair_register(records)
    gradient = recompute_gradient_direction()
    tests = counterexample_tests()

    input_paths = {
        "source_train_records": RECORDS,
        "existing_source_time_endpoint_audit": ENDPOINT_AUDIT,
        "existing_source_inventory_metadata": SOURCE_INVENTORY,
        "source_fit_preparation_code": SOURCE_PREP,
        "official_response_builder": OFFICIAL_BUILDER,
        "user_attachment": ATTACHMENT,
        "existing_gradient_panel_inputs": GRADIENT_DIR / "INPUTS.json",
        "existing_gradient_panel_summary": GRADIENT_DIR / "SUMMARY.json",
        "existing_gradient_panel_config": GRADIENT_DIR / "CONFIG.json",
        "existing_gradient_panel_lock": GRADIENT_DIR / "LOCK.json",
    }
    input_manifest = {
        "created_by": str(Path(__file__).resolve().relative_to(ROOT)),
        "created_at_utc_date": "2026-09-27",
        "audit_script_sha256": sha256(Path(__file__).resolve()),
        "input_files": {
            label: {"path": str(path.resolve()), "sha256": sha256(path)}
            for label, path in input_paths.items()
        },
        "scope": {
            "source_training_labels_read": True,
            "source_validation_labels_read": False,
            "target_labels_read": False,
            "pixels_or_video_opened": False,
            "cuda_used": False,
            "model_forward_or_prediction_scored": False,
            "outputs_are_new_and_isolated": True,
        },
    }
    hashes = {
        "audit_script_sha256": input_manifest["audit_script_sha256"],
        "inputs": {
            "source_train_records": input_manifest["input_files"]["source_train_records"],
            "existing_source_time_endpoint_audit": input_manifest["input_files"]["existing_source_time_endpoint_audit"],
        },
    }
    report = render_report(summary, panel, pair_register, gradient, hashes)

    OUT.mkdir(parents=True, exist_ok=False)
    write_json(OUT / "INPUTS.json", input_manifest)
    write_json(OUT / "CPU_SUMMARY.json", summary)
    write_json(OUT / "PANEL.json", panel)
    write_json(OUT / "SAME_VIDEO_EVENT_PAIRS.json", pair_register)
    write_json(OUT / "GRADIENT_RECOMPUTE.json", gradient)
    write_json(OUT / "COUNTEREXAMPLE_CPU_TESTS.json", tests)
    with (OUT / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write(report)
    output_files = sorted(p for p in OUT.iterdir() if p.is_file())
    write_json(OUT / "OUTPUT_MANIFEST.json", {
        "files": {p.name: sha256(p) for p in output_files},
        "audit_script_sha256": input_manifest["audit_script_sha256"],
        "all_outputs_new": True,
    })
    print(json.dumps({
        "status": "completed",
        "output": str(OUT),
        "records": summary["record_count"],
        "eligible_ce": summary["response_eligible_count"],
        "no_ce_sampling_quantization": summary["noneligible_with_no_sampled_active_frame_sampling_quantization"],
        "physical_mismatches": summary["physical_event_mask_mismatch_count"],
        "panel": panel["panel_size"],
        "pairs": pair_register["registered_pair_count"],
        "gradient_negative_factors": gradient["first_order_factor_negative_count"],
        "script_sha256": input_manifest["audit_script_sha256"],
    }, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    run()
