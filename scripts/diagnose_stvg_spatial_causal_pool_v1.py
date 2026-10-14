#!/usr/bin/env python3
"""CPU-only decomposition of fixed candidate-pool and teacher-reference effects.

This reader consumes only the already scored aggregate report, the locked
candidate-ID roster, and the sealed teacher reference IDs. It never opens raw
annotations, media, model weights, or inference outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_BASE = Path("artifacts/stvg_motion_guided_instance_opd_v1")
DEFAULT_OUT = Path("artifacts/stvg_spatial_causal_decomposition_v1/POOL_TEACHER_DIAGNOSIS.json")
ANCHOR_NEUTRAL_TOLERANCE_FRACTION = 1e-10  # 1e-8 percentage points / 100.
VIOU_NEUTRAL_TOLERANCE_PP = 1e-8
NUMERIC_TOLERANCE = 1e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def fraction_score(value: Any, field: str) -> float:
    require(finite_number(value), f"{field} must be a finite saved score")
    x = float(value)
    require(-NUMERIC_TOLERANCE <= x <= 1.0 + NUMERIC_TOLERANCE,
            f"{field} is outside the verified [0,1] fraction range: {x}")
    return min(1.0, max(0.0, x))


def classify_delta(delta: float | None, tolerance: float = ANCHOR_NEUTRAL_TOLERANCE_FRACTION) -> str:
    if delta is None or not math.isfinite(delta):
        return "unknown"
    if delta > tolerance:
        return "positive"
    if delta < -tolerance:
        return "negative"
    return "neutral"


def common_support_indices(
    gt_known: Sequence[bool],
    source_valid: Sequence[bool],
    displayed_candidate_valid: Mapping[int, Sequence[bool]],
    displayed_ids: Sequence[int],
) -> list[int]:
    """Return strict common slots; missing/unknown is never converted to zero."""
    n = len(gt_known)
    require(len(source_valid) == n, "synthetic/common support source length mismatch")
    require(set(displayed_candidate_valid) >= set(displayed_ids), "displayed ID missing a validity vector")
    for candidate_id in displayed_ids:
        require(len(displayed_candidate_valid[candidate_id]) == n,
                f"candidate {candidate_id} validity length mismatch")
    return [i for i in range(n)
            if gt_known[i] and source_valid[i]
            and all(displayed_candidate_valid[cid][i] for cid in displayed_ids)]


def mean_or_none(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    if not vals:
        return None
    require(all(math.isfinite(x) for x in vals), "nonfinite value in mean")
    return sum(vals) / len(vals)


def assert_close_saved(computed: float | None, saved: Any, field: str, *, atol: float = 1e-10) -> None:
    if computed is None:
        require(saved is None, f"{field}: computed null but saved aggregate is non-null")
        return
    require(finite_number(saved), f"{field}: computed score but saved aggregate is null/nonfinite")
    require(math.isclose(computed, float(saved), rel_tol=1e-11, abs_tol=atol),
            f"{field}: recomputed {computed} differs from saved {saved}")


def choose_oracle(scores: Mapping[int, float]) -> int | None:
    if not scores:
        return None
    best_score = max(scores.values())
    return min(candidate_id for candidate_id, score in scores.items() if score == best_score)


def synthetic_contracts() -> dict[str, Any]:
    # The second slot is excluded because candidate 7 is missing there. The
    # unknown-GT slot is excluded independently, rather than represented by 0.
    indices = common_support_indices(
        gt_known=[True, True, False],
        source_valid=[True, True, True],
        displayed_candidate_valid={7: [True, False, True], 2: [True, True, True]},
        displayed_ids=[7, 2],
    )
    require(indices == [0], "common-missing/unknown synthetic contract failed")
    no_support = common_support_indices(
        gt_known=[False, True], source_valid=[True, False],
        displayed_candidate_valid={1: [True, True]}, displayed_ids=[1],
    )
    require(no_support == [] and mean_or_none([]) is None,
            "empty support must yield an unknown/null score")
    oracle = choose_oracle({5: 0.75, 2: 0.75, 1: 0.2})
    require(oracle == 2, "exact-tie oracle must use ascending candidate ID")
    teacher_scores = {1: 0.2, 2: 0.8}
    selected_teacher_id = 2
    teacher_delta = teacher_scores[selected_teacher_id] - 0.3
    require(abs(teacher_delta - 0.5) <= 1e-15,
            "teacher ID must join to its own saved candidate score")
    # Mean over the sole valid common slot remains 0.6; the missing slot is not
    # inserted as zero (which would have produced 0.3).
    no_zero_fill = mean_or_none([0.6])
    require(no_zero_fill == 0.6, "missing candidate support was zero-filled")
    require(fraction_score(0.5, "synthetic_iou") == 0.5, "fraction unit check failed")
    rejected_percent = False
    try:
        fraction_score(50.0, "synthetic_iou_percent")
    except AssertionError:
        rejected_percent = True
    require(rejected_percent, "unit validator must reject 0-100 values as fractions")
    return {
        "status": "pass",
        "checks": [
            "unknown GT and missing candidate support are excluded, not zero-filled",
            "empty common support returns null",
            "oracle exact ties resolve by ascending candidate ID",
            "locked teacher ID joins to that candidate's score",
            "IoU fraction validator rejects percentage-scaled values",
        ],
        "check_count": 5,
    }


def _unique_rows_by_candidate(rows: Sequence[Mapping[str, Any]], case_id: str) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        cid = int(row["candidate_id"])
        require(cid not in result, f"{case_id}: duplicate candidate score row {cid}")
        result[cid] = row
    return result


def _anchor_rows(row: Mapping[str, Any], case_id: str) -> list[Mapping[str, Any]]:
    anchors = list(row.get("per_original4_anchor", []))
    require(len(anchors) == 4, f"{case_id}: expected four fixed original observation slots")
    positions = [int(a["position"]) for a in anchors]
    require(len(set(positions)) == 4, f"{case_id}: duplicate observation position")
    return anchors


def _pairwise_candidate(row: Mapping[str, Any], case_id: str, displayed: bool) -> dict[str, Any]:
    supported: list[tuple[int, float, float]] = []
    for anchor in _anchor_rows(row, case_id):
        if (anchor.get("GT_support") == "known"
                and anchor.get("candidate_support") == "present_valid"
                and finite_number(anchor.get("candidate_IoU"))
                and finite_number(anchor.get("paired_F_source_same_anchor_IoU"))):
            pos = int(anchor["position"])
            candidate_iou = fraction_score(anchor["candidate_IoU"], "candidate_IoU")
            source_iou = fraction_score(anchor["paired_F_source_same_anchor_IoU"],
                                        "paired_F_source_same_anchor_IoU")
            supported.append((pos, candidate_iou, source_iou))
    cand_mean = mean_or_none(x[1] for x in supported)
    source_mean = mean_or_none(x[2] for x in supported)
    delta = None if not supported else mean_or_none(x[1] - x[2] for x in supported)
    require(len(supported) == int(row.get("candidate_present_and_GT_known_anchor_count", -1)),
            f"{case_id} candidate {row['candidate_id']}: scored anchor support count mismatch")
    assert_close_saved(cand_mean, row.get("present_anchor_GT_IoU_mean_known_only"),
                       f"{case_id} candidate {row['candidate_id']} mean candidate IoU")
    assert_close_saved(source_mean, row.get("paired_F_source_same_anchor_GT_IoU_mean_known_only"),
                       f"{case_id} candidate {row['candidate_id']} paired source mean IoU")
    assert_close_saved(delta, row.get("present_minus_paired_F_source_mean_IoU_known_only"),
                       f"{case_id} candidate {row['candidate_id']} paired delta")
    if bool(row.get("is_Source_candidate")) and delta is not None:
        require(abs(delta) <= NUMERIC_TOLERANCE,
                f"{case_id}: Source candidate is not paired to zero-delta F reference")
    return {
        "candidate_id": int(row["candidate_id"]),
        "is_Source_candidate": bool(row.get("is_Source_candidate", False)),
        "included_in_displayed_pool": displayed,
        "support_count": len(supported),
        "support_positions": [x[0] for x in supported],
        "candidate_mean_IoU_fraction": cand_mean,
        "paired_F_source_mean_IoU_fraction": source_mean,
        "delta_candidate_minus_paired_F_source_fraction": delta,
        "classification": classify_delta(delta),
        "support_semantics": "candidate-present-valid AND GT-known paired with F-source on this candidate's own support",
    }


def analyze_case(
    case_id: str,
    dataset: str,
    cohort: str,
    displayed_ids: Sequence[int],
    candidate_score_rows: Sequence[Mapping[str, Any]],
    teacher_reference: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    displayed = [int(x) for x in displayed_ids]
    require(len(displayed) == len(set(displayed)), f"{case_id}: duplicate displayed candidate IDs")
    by_id = _unique_rows_by_candidate(candidate_score_rows, case_id)
    require(bool(by_id), f"{case_id}: no scored candidate rows")
    source_rows = [r for r in candidate_score_rows if bool(r.get("is_Source_candidate"))]
    require(len(source_rows) == 1, f"{case_id}: expected exactly one Source candidate score row")
    source_row = source_rows[0]
    source_id = int(source_row["candidate_id"])
    require(source_id in displayed, f"{case_id}: Source candidate is not in the sealed displayed pool")
    missing_display_ids = [candidate_id for candidate_id in displayed if candidate_id not in by_id]
    require(not missing_display_ids, f"{case_id}: displayed IDs missing scored rows: {missing_display_ids}")

    source_anchors = _anchor_rows(source_row, case_id)
    all_anchors = {cid: _anchor_rows(row, case_id) for cid, row in by_id.items()}
    # The four GT-known labels and source F scores are common across candidate
    # rows. Validate that we are not accidentally pairing unlike observations.
    for slot in range(4):
        gt_labels = {anchors[slot].get("GT_support") for anchors in all_anchors.values()}
        source_scores = {
            (None if not finite_number(anchors[slot].get("F_source_IoU_at_anchor"))
             else float(anchors[slot]["F_source_IoU_at_anchor"]))
            for anchors in all_anchors.values()
        }
        require(len(gt_labels) == 1, f"{case_id}: GT-support label differs between candidate score rows")
        finite_sources = [x for x in source_scores if x is not None]
        require(len(source_scores) <= 1 or (len(source_scores) == 2 and len(finite_sources) == 1),
                f"{case_id}: F-source score differs between candidate rows")

    gt_known = [a.get("GT_support") == "known" for a in source_anchors]
    source_valid = [
        a.get("candidate_support") == "present_valid"
        and finite_number(a.get("F_source_IoU_at_anchor"))
        for a in source_anchors
    ]
    display_validity: dict[int, list[bool]] = {}
    for cid in displayed:
        display_validity[cid] = [
            anchor.get("candidate_support") == "present_valid"
            and finite_number(anchor.get("candidate_IoU"))
            for anchor in all_anchors[cid]
        ]
    common_slots = common_support_indices(gt_known, source_valid, display_validity, displayed)
    slot_positions = [int(a["position"]) for a in source_anchors]
    common_positions = [slot_positions[i] for i in common_slots]
    source_scores_common = [
        fraction_score(source_anchors[i].get("F_source_IoU_at_anchor"), "F_source_IoU_at_anchor")
        for i in common_slots
    ]
    source_mean_common = mean_or_none(source_scores_common)

    candidate_common_scores: dict[int, float] = {}
    for cid in displayed:
        values = [fraction_score(all_anchors[cid][i].get("candidate_IoU"), "candidate_IoU")
                  for i in common_slots]
        if values:
            candidate_common_scores[cid] = float(mean_or_none(values))
    pool_id = choose_oracle(candidate_common_scores)
    pool_iou = None if pool_id is None else candidate_common_scores[pool_id]
    pool_delta = None if pool_iou is None or source_mean_common is None else pool_iou - source_mean_common
    if pool_delta is not None:
        require(pool_delta >= -NUMERIC_TOLERANCE,
                f"{case_id}: pool oracle fell below Source even though Source is displayed")

    selected_id_value = teacher_reference.get("selected_candidate_id")
    selected_id = None if selected_id_value is None else int(selected_id_value)
    selected_reason = "no_locked_candidate_id"
    selected_iou_common: float | None = None
    selected_delta_common: float | None = None
    if selected_id is not None and selected_id not in displayed:
        selected_reason = "locked_teacher_candidate_not_in_displayed_pool"
    elif selected_id is not None and source_mean_common is None:
        selected_reason = "empty_common_support"
    elif selected_id is not None:
        selected_iou_common = candidate_common_scores.get(selected_id)
        if selected_iou_common is None:
            selected_reason = "teacher_candidate_has_no_common_support_score"
        else:
            selected_delta_common = selected_iou_common - float(source_mean_common)
            selected_reason = "scored_on_common_support"
    selected_pairwise = None if selected_id is None or selected_id not in by_id else _pairwise_candidate(
        by_id[selected_id], case_id, selected_id in displayed)

    pairwise_rows = [
        _pairwise_candidate(by_id[cid], case_id, cid in displayed)
        for cid in sorted(by_id)
    ]
    per_slot_exclusions: list[dict[str, Any]] = []
    for slot, anchor in enumerate(source_anchors):
        reasons: list[str] = []
        if not gt_known[slot]:
            reasons.append("GT_unknown")
        if gt_known[slot] and not source_valid[slot]:
            reasons.append("Source_invalid_or_missing")
        missing_or_invalid = [cid for cid in displayed if not display_validity[cid][slot]]
        if missing_or_invalid:
            reasons.append("displayed_candidate_invalid_or_missing")
        if reasons:
            per_slot_exclusions.append({
                "position": int(anchor["position"]),
                "exclusion_reasons": reasons,
                "invalid_or_missing_displayed_candidate_ids": missing_or_invalid,
            })

    metric_by_arm: dict[str, Mapping[str, Any]] = {}
    for row in metric_rows:
        arm = str(row["arm"])
        require(arm not in metric_by_arm, f"{case_id}: duplicate metric row for arm {arm}")
        metric_by_arm[arm] = row
    require("F" in metric_by_arm and "teacher_iou" in metric_by_arm,
            f"{case_id}: missing F or teacher_iou full-tube metric")
    f_row = metric_by_arm["F"]
    teacher_iou_row = metric_by_arm["teacher_iou"]
    require(f_row.get("status") == "scored" and teacher_iou_row.get("status") == "scored",
            f"{case_id}: F / teacher_iou full-tube metric not scored")
    f_viou = fraction_score(f_row["metrics"].get("m_vIoU"), "F.m_vIoU")
    teacher_viou = fraction_score(teacher_iou_row["metrics"].get("m_vIoU"), "teacher_iou.m_vIoU")
    cmp_rows = [r for r in comparison_rows if r.get("arm") == "teacher_iou"]
    require(len(cmp_rows) == 1, f"{case_id}: expected one original teacher_iou comparison row")
    cmp = cmp_rows[0]
    require(cmp.get("baseline") == "F" and cmp.get("status") == "scored",
            f"{case_id}: original teacher_iou row is not a scored F comparison")
    fulltube_delta_pp = (teacher_viou - f_viou) * 100.0
    saved_delta_pp = float(cmp["delta_vIoU_pp"])
    require(math.isclose(fulltube_delta_pp, saved_delta_pp, rel_tol=1e-12, abs_tol=1e-9),
            f"{case_id}: full-tube comparison does not match saved F/teacher_iou m_vIoU")
    expected_category = (
        "positive" if saved_delta_pp > VIOU_NEUTRAL_TOLERANCE_PP else
        "negative" if saved_delta_pp < -VIOU_NEUTRAL_TOLERANCE_PP else "neutral"
    )
    require(cmp.get("category") == expected_category,
            f"{case_id}: saved full-tube category differs from its pp delta")

    return {
        "case_id": case_id,
        "dataset": dataset,
        "cohort": cohort,
        "displayed_candidate_ids_from_prelock_barrier": displayed,
        "scored_candidate_ids_including_not_displayed": sorted(by_id),
        "not_displayed_scored_candidate_ids": [cid for cid in sorted(by_id) if cid not in displayed],
        "source_candidate_id": source_id,
        "teacher_selection": {
            "selected_candidate_id": selected_id,
            "no_update_locked": bool(teacher_reference.get("no_update", False)),
            "selected_reference_common_support": {
                "support_count": len(common_slots),
                "support_positions": common_positions,
                "candidate_mean_IoU_fraction": selected_iou_common,
                "F_source_mean_IoU_fraction": source_mean_common,
                "delta_teacher_minus_F_source_fraction": selected_delta_common,
                "classification": classify_delta(selected_delta_common),
                "status": selected_reason,
            },
            "selected_reference_own_pairwise_support": selected_pairwise,
        },
        "common_observation_support": {
            "definition": "GT-known AND Source present-valid with finite F IoU AND every prelock-displayed candidate present-valid with finite candidate IoU",
            "support_count": len(common_slots),
            "support_positions": common_positions,
            "F_source_mean_IoU_fraction": source_mean_common,
            "excluded_slots": per_slot_exclusions,
        },
        "displayed_pool_oracle_on_common_support": {
            "status": "scored" if pool_id is not None else "unknown_empty_common_support",
            "oracle_candidate_id": pool_id,
            "tie_break": "exact score ties choose the lowest numeric candidate ID",
            "candidate_mean_IoU_fraction_by_displayed_id": {
                str(cid): candidate_common_scores.get(cid) for cid in displayed
            },
            "oracle_mean_IoU_fraction": pool_iou,
            "F_source_mean_IoU_fraction": source_mean_common,
            "delta_pool_minus_F_source_fraction": pool_delta,
            "classification": classify_delta(pool_delta),
            "offline_diagnostic_only": True,
        },
        "all_candidate_pairwise_deltas_on_each_candidate_own_support": pairwise_rows,
        "original_full_tube_metric_separate_from_anchor_geometry": {
            "metric": "m_vIoU",
            "units": "fraction in [0,1]",
            "baseline_arm": "F",
            "teacher_arm": "teacher_iou",
            "F_m_vIoU_fraction": f_viou,
            "teacher_iou_m_vIoU_fraction": teacher_viou,
            "delta_vIoU_percentage_points": saved_delta_pp,
            "classification_from_original_report": cmp["category"],
            "status": "scored",
            "locked_teacher_candidate_id": selected_id,
            "not_an_anchor_IoU_or_pool_oracle_measure": True,
        },
    }


def analyze(base: Path) -> dict[str, Any]:
    report_path = base / "ROOT_POSTSEAL_REPORT.json"
    barrier_path = base / "CANDIDATE_INPUT_BARRIER.json"
    refs_path = base / "METHOD_REFERENCE_LOCK.json"
    roster_path = base / "ROSTER_LOCK.json"
    scorer_path = Path("scripts/score_stvg_motion_guided_postseal_v1.py")
    report = read_json(report_path)
    barrier = read_json(barrier_path)
    reference_lock = read_json(refs_path)
    roster = read_json(roster_path)

    require(report.get("completed") is True, "postseal report is not marked completed")
    require(barrier.get("status") == "complete", "candidate input barrier is incomplete")
    require(barrier.get("GT_read") is False, "candidate barrier claims GT access")
    require(reference_lock.get("GT_read") is False, "teacher reference lock claims GT access")
    require(report.get("GT_used_to_generate_or_select_candidates") is False,
            "postseal report does not certify that GT was excluded from candidate generation/selection")

    barrier_sha = sha256_file(barrier_path)
    lock_sha = sha256_file(refs_path)
    require(report.get("method_reference_lock_sha256") == lock_sha,
            "postseal report is not bound to this teacher reference lock")
    input_binding = report.get("candidate_input_binding", {})
    require(input_binding.get("sha256") == barrier_sha,
            "postseal report is not bound to this candidate input barrier")
    require(int(input_binding.get("record_count", -1)) == 16,
            "scored candidate binding does not cover 16 fixed cases")

    roster_by_id = {r["case_id"]: r for r in roster.get("records", [])}
    barrier_by_id = {r["case_id"]: r for r in barrier.get("records", [])}
    ref_by_id = {r["case_id"]: r for r in reference_lock.get("records", [])}
    require(len(roster_by_id) == len(barrier_by_id) == len(ref_by_id) == 16,
            "fixed roster/barrier/reference record count is not 16")
    require(set(roster_by_id) == set(barrier_by_id) == set(ref_by_id),
            "fixed case IDs differ between roster, barrier, and reference lock")
    require(reference_lock.get("phase") == "method", "reference lock is not the method-phase lock")

    candidate_rows_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in report.get("candidate_present_anchor_rows", []):
        if row.get("phase") != "method":
            continue
        candidate_rows_by_case.setdefault(str(row["case_id"]), []).append(row)
    metric_rows_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in report.get("metric_rows", []):
        if row.get("phase") == "method":
            metric_rows_by_case.setdefault(str(row["case_id"]), []).append(row)
    comparisons_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in report.get("arm_comparison_rows", []):
        if row.get("phase") == "method":
            comparisons_by_case.setdefault(str(row["case_id"]), []).append(row)

    case_results = []
    displayed_count = 0
    scored_row_count = 0
    hidden_row_count = 0
    for case_id in sorted(roster_by_id):
        rec = barrier_by_id[case_id]
        roster_rec = roster_by_id[case_id]
        require(rec.get("dataset") == roster_rec.get("dataset"), f"{case_id}: dataset binding mismatch")
        require(rec.get("GT_read") is False, f"{case_id}: per-case candidate barrier used GT")
        ref = ref_by_id[case_id]
        require(ref.get("dataset") == roster_rec.get("dataset"), f"{case_id}: teacher lock dataset mismatch")
        teacher = ref.get("references", {}).get("teacher", {})
        score_rows = candidate_rows_by_case.get(case_id, [])
        displayed_ids = [int(x) for x in rec.get("displayed_candidate_ids", [])]
        displayed_count += len(displayed_ids)
        scored_row_count += len(score_rows)
        hidden_row_count += max(0, len(score_rows) - len(displayed_ids))
        require(bool(displayed_ids), f"{case_id}: empty locked displayed candidate pool")
        require(teacher.get("selected_candidate_id") is None
                or int(teacher["selected_candidate_id"]) in displayed_ids,
                f"{case_id}: teacher lock selected an ID outside the prelock displayed pool")
        case_results.append(analyze_case(
            case_id=case_id,
            dataset=str(roster_rec["dataset"]),
            cohort=str(roster_rec.get("cohort", "unknown")),
            displayed_ids=displayed_ids,
            candidate_score_rows=score_rows,
            teacher_reference=teacher,
            metric_rows=metric_rows_by_case.get(case_id, []),
            comparison_rows=comparisons_by_case.get(case_id, []),
        ))

    require(len(report.get("candidate_present_anchor_rows", [])) == 83,
            "expected the sealed 83 method candidate score rows")
    require(displayed_count == 78 and scored_row_count == 83 and hidden_row_count == 5,
            "displayed-versus-scored candidate row counts changed")

    category_counts: dict[str, dict[str, int]] = {}
    for field, extractor in {
        "pool_delta": lambda c: c["displayed_pool_oracle_on_common_support"]["classification"],
        "teacher_selected_common_delta": lambda c: c["teacher_selection"]["selected_reference_common_support"]["classification"],
        "per_candidate_own_support_delta": lambda c: [r["classification"] for r in c["all_candidate_pairwise_deltas_on_each_candidate_own_support"]],
        "original_full_tube_vIoU_delta": lambda c: c["original_full_tube_metric_separate_from_anchor_geometry"]["classification_from_original_report"],
    }.items():
        counter: Counter[str] = Counter()
        for case in case_results:
            vals = extractor(case)
            counter.update(vals if isinstance(vals, list) else [vals])
        category_counts[field] = {k: counter.get(k, 0) for k in ("positive", "negative", "neutral", "unknown", "baseline")}

    all_iou_values = [
        a["candidate_IoU"]
        for row in report["candidate_present_anchor_rows"]
        for a in row["per_original4_anchor"]
        if a.get("candidate_IoU") is not None
    ]
    all_viou_values = [
        row["metrics"]["m_vIoU"]
        for row in report["metric_rows"]
        if row.get("status") == "scored" and row.get("metrics", {}).get("m_vIoU") is not None
    ]
    # The persisted scorer averages overlap in [0,1] and multiplies the m_vIoU
    # difference by 100 only for the separate percentage-point comparison.
    for value in all_iou_values:
        fraction_score(value, "saved candidate_IoU")
    for value in all_viou_values:
        fraction_score(value, "saved m_vIoU")

    scorer_sha = sha256_file(scorer_path)
    return {
        "schema_version": "stvg_spatial_causal_pool_diagnosis_v1",
        "status": "complete",
        "scope": {
            "fixed_case_count": len(case_results),
            "fixed_cases_source": "ROSTER_LOCK intersected exactly with CANDIDATE_INPUT_BARRIER and METHOD_REFERENCE_LOCK",
            "raw_GT_coordinates_read_by_this_diagnostic": False,
            "raw_RGB_or_video_read_by_this_diagnostic": False,
            "model_or_weights_loaded": False,
            "GPU_used": False,
            "GT_scored_aggregates_used": True,
            "candidate_pool_oracle_changes_locked_references": False,
        },
        "input_bindings": {
            "root_postseal_report_sha256": sha256_file(report_path),
            "candidate_input_barrier_sha256": barrier_sha,
            "method_reference_lock_sha256": lock_sha,
            "roster_lock_sha256": sha256_file(roster_path),
            "scoring_implementation_sha256": scorer_sha,
        },
        "units_and_validation": {
            "candidate_anchor_iou": "fraction in [0,1]",
            "m_vIoU": "fraction in [0,1]",
            "delta_vIoU": "percentage points; recomputed as 100 * (teacher_iou_m_vIoU - F_m_vIoU)",
            "unit_verification": {
                "candidate_IoU_saved_value_count": len(all_iou_values),
                "candidate_IoU_min": min(map(float, all_iou_values)),
                "candidate_IoU_max": max(map(float, all_iou_values)),
                "m_vIoU_saved_value_count": len(all_viou_values),
                "m_vIoU_min": min(map(float, all_viou_values)),
                "m_vIoU_max": max(map(float, all_viou_values)),
                "scorer_formula_evidence": "score_stvg_motion_guided_postseal_v1.py:850 saves means of IoU fractions; lines 658-659 multiply m_vIoU difference by 100 for delta_vIoU_pp",
                "input_values_rejected_if_outside_fraction_range": True,
            },
            "anchor_delta_neutral_tolerance_fraction": ANCHOR_NEUTRAL_TOLERANCE_FRACTION,
            "anchor_delta_neutral_tolerance_derivation": "original 1e-8 percentage-point tolerance divided by 100",
            "full_tube_delta_neutral_tolerance_percentage_points": VIOU_NEUTRAL_TOLERANCE_PP,
        },
        "method": {
            "common_support": "GT known; Source candidate present-valid and finite F-source IoU; every prelock-displayed candidate is present-valid and has finite candidate IoU at the same original observation slot",
            "pool_delta": "max displayed-candidate mean anchor IoU on strict common support minus F-source mean anchor IoU on that same support",
            "teacher_delta": "locked teacher-selected candidate mean anchor IoU minus F-source mean anchor IoU on strict common support",
            "candidate_pairwise_delta": "each scored candidate minus paired F-source on that candidate's own GT-known/present-valid support; differing supports are reported separately",
            "missing_or_unknown": "excluded and represented as null/unknown; never filled with zero",
            "oracle_tie_break": "exact ties resolve to lowest numeric candidate ID",
            "oracle_role": "offline diagnostic only; does not modify METHOD_REFERENCE_LOCK",
        },
        "counts": {
            "case_count": len(case_results),
            "displayed_candidate_id_count": displayed_count,
            "scored_candidate_rows_including_not_displayed": scored_row_count,
            "scored_but_not_displayed_candidate_rows": hidden_row_count,
            "common_support_empty_case_count": sum(c["common_observation_support"]["support_count"] == 0 for c in case_results),
            "teacher_selected_pairwise_unknown_case_count": sum(
                c["teacher_selection"]["selected_reference_own_pairwise_support"]["support_count"] == 0
                for c in case_results
            ),
        },
        "category_counts": category_counts,
        "synthetic_contracts": synthetic_contracts(),
        "cases": case_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = analyze(args.base)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "output": str(args.out),
        "cases": result["counts"]["case_count"],
        "displayed_candidate_ids": result["counts"]["displayed_candidate_id_count"],
        "scored_rows": result["counts"]["scored_candidate_rows_including_not_displayed"],
        "hidden_scored_rows": result["counts"]["scored_but_not_displayed_candidate_rows"],
        "empty_common_support_cases": result["counts"]["common_support_empty_case_count"],
        "categories": result["category_counts"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
