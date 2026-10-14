"""Build an anonymous report for the sealed native-instance STVG experiment.

This entry point consumes only post-seal scalar readbacks and whitelisted timing
fields. It never loads prediction tensors, masks, media, annotations, or model
artifacts. Run only after the parent controller has sealed the score files and
authorized report generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_native_instance_spatial_opd_v1"
REPORT_PATH = ROOT / "docs/STVG_NATIVE_INSTANCE_V1.md"
FIGURE_DIR = BASE / "figures"
PDF_PATH = FIGURE_DIR / "native_instance_summary.pdf"
PNG_PATH = FIGURE_DIR / "native_instance_summary.png"

SCORE_METRICS = (
    "m_vIoU",
    "m_tIoU",
    "m_sIoU_dense_GT",
    "m_sIoU_supervised_GT",
    "m_sIoU_unsupervised_GT",
)
METRIC_LABELS = {
    "m_vIoU": "vIoU",
    "m_tIoU": "tIoU",
    "m_sIoU_dense_GT": "dense sIoU",
    "m_sIoU_supervised_GT": "supervised sIoU",
    "m_sIoU_unsupervised_GT": "unsupervised sIoU",
}
ARM_ORDER = (
    "T",
    "native_teacher_direct",
    "T_direct",
    "T_iou",
    "T_reliability_weighted_iou",
    "F",
    "direct",
    "iou",
    "reliability_weighted_iou",
)
PRIMARY_ARMS = ARM_ORDER[:5]
ATTRIBUTION_ARMS = ARM_ORDER[5:]
SPATIAL_ALGORITHMS = ("direct", "iou", "reliability_weighted_iou")
ARM_LABELS = {
    "T": "Temporal only (T)",
    "native_teacher_direct": "Native teacher direct",
    "T_direct": "T + direct spatial update",
    "T_iou": "T + IoU spatial update",
    "T_reliability_weighted_iou": "T + reliability-weighted IoU",
    "F": "Frozen source (F)",
    "direct": "Direct spatial update",
    "iou": "IoU spatial update",
    "reliability_weighted_iou": "Reliability-weighted IoU update",
}
ARM_SHORT = {
    "T": "T",
    "native_teacher_direct": "Native direct",
    "T_direct": "T + direct",
    "T_iou": "T + IoU",
    "T_reliability_weighted_iou": "T + weighted IoU",
    "F": "F",
    "direct": "Direct",
    "iou": "IoU",
    "reliability_weighted_iou": "Weighted IoU",
}
ARM_COLORS = {
    "T": "#315f8c",
    "native_teacher_direct": "#c7793b",
    "T_direct": "#3b806d",
    "T_iou": "#559b82",
    "T_reliability_weighted_iou": "#7bb39c",
    "F": "#6b7078",
    "direct": "#b28c45",
    "iou": "#c5a45c",
    "reliability_weighted_iou": "#d4bd83",
}
DATASET_LABELS = {"hc2": "HC", "vidstg": "VidSTG"}
TRACKING_RECEIPT_REL = "TRACKING/{case_id}/ACCEPTED.json"
TEACHER_RECEIPT_REL = "teacher/{case_id}.receipt.json"


class ReportInputError(RuntimeError):
    """Raised when a sealed report input fails its schema or binding checks."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportInputError(f"required report input is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise ReportInputError(f"report input must be a JSON object: {path.name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, *, minimum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not math.isfinite(float(value)):
        return False
    return minimum is None or float(value) >= minimum


def _anonymous_source(case_id: str, dataset: str) -> str:
    """Map internal program keys to the fixed report aliases without returning IDs."""
    match = re.fullmatch(r"(hc2|vidstg)_(\d{3})", case_id)
    if match is None or match.group(1) != dataset:
        raise ReportInputError("case-to-dataset binding does not match the fixed roster shape")
    slot = int(match.group(2))
    if slot not in range(4):
        raise ReportInputError("case is outside the fixed eight-source report panel")
    return ("HC" if dataset == "hc2" else "V") + f"{slot + 1:02d}"


def _validate_metric_map(metrics: Any, context: str) -> dict[str, float | None]:
    if metrics is None:
        return {name: None for name in SCORE_METRICS}
    if not isinstance(metrics, dict):
        raise ReportInputError(f"metric map has an invalid type in {context}")
    missing = set(SCORE_METRICS) - set(metrics)
    if missing:
        raise ReportInputError(f"metric map is incomplete in {context}")
    output: dict[str, float | None] = {}
    for name in SCORE_METRICS:
        value = metrics[name]
        if value is None:
            output[name] = None
        elif _finite_number(value) and 0.0 <= float(value) <= 1.0:
            output[name] = float(value)
        else:
            raise ReportInputError(f"metric value is outside its declared 0-to-1 scale in {context}")
    return output


def _load_scored_rows() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]
]:
    rows_doc = _read_json(BASE / "POSTSEAL_SCORE_ROWS.json")
    summary_doc = _read_json(BASE / "POSTSEAL_SCORE_SUMMARY.json")
    if rows_doc.get("schema") != "stvg-native-instance-postseal-rows-v1":
        raise ReportInputError("postseal rows schema is not the expected frozen schema")
    if rows_doc.get("status") != "complete" or rows_doc.get("GT_read") is not True:
        raise ReportInputError("postseal score rows are not a completed GT-based evaluation")
    if rows_doc.get("metric_scale") != "0_to_1" or rows_doc.get("case_count") != 8:
        raise ReportInputError("postseal rows do not describe the locked eight-source, 0-to-1 score")
    prediction_barrier = rows_doc.get("prediction_barrier")
    if not isinstance(prediction_barrier, dict) or not re.fullmatch(
        r"[0-9a-f]{64}", str(prediction_barrier.get("sha256", ""))
    ):
        raise ReportInputError("postseal score rows lack a valid prediction-barrier binding")
    if summary_doc.get("schema") != "stvg-native-instance-postseal-summary-v1":
        raise ReportInputError("postseal summary schema is not the expected frozen schema")
    if summary_doc.get("status") != "complete" or summary_doc.get("GT_read") is not True:
        raise ReportInputError("postseal summary is not a completed GT-based evaluation")
    if summary_doc.get("metric_scale") != "0_to_1":
        raise ReportInputError("postseal summary metric scale is not 0-to-1")

    raw_rows = rows_doc.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != 8 * len(ARM_ORDER):
        raise ReportInputError("postseal rows do not contain the expected case-by-arm grid")
    case_datasets: dict[str, str] = {}
    indexed: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, dict):
            raise ReportInputError("postseal rows contain a non-object entry")
        case_id, dataset, arm = row.get("case_id"), row.get("dataset"), row.get("arm")
        if not isinstance(case_id, str) or dataset not in DATASET_LABELS or arm not in ARM_ORDER:
            raise ReportInputError("postseal row has an unrecognized case, domain, or arm")
        _anonymous_source(case_id, dataset)
        if case_id in case_datasets and case_datasets[case_id] != dataset:
            raise ReportInputError("one source key is assigned to more than one domain")
        case_datasets[case_id] = dataset
        key = f"{case_id}\0{arm}"
        if key in indexed:
            raise ReportInputError("duplicate case-by-arm row in postseal results")
        metrics = _validate_metric_map(row.get("metrics"), key)
        support = row.get("support")
        coverage = row.get("coverage")
        fit = row.get("fit")
        if support is not None and not isinstance(support, dict):
            raise ReportInputError("postseal row support has an invalid type")
        if not isinstance(fit, dict):
            raise ReportInputError("postseal row is missing fit readback")
        if coverage is not None and not isinstance(coverage, dict):
            raise ReportInputError("postseal row coverage has an invalid type")
        if support is not None:
            for name in (
                "source_grid_positions",
                "supervised_anchor_count",
                "unsupervised_position_count",
                "dense_gt_known_frame_count",
                "supervised_gt_known_frame_count",
                "unsupervised_gt_known_frame_count",
            ):
                if name not in support:
                    raise ReportInputError(f"support field is missing: {name}")
            for name in (
                "source_grid_positions",
                "supervised_anchor_count",
                "unsupervised_position_count",
                "dense_gt_known_frame_count",
                "supervised_gt_known_frame_count",
                "unsupervised_gt_known_frame_count",
            ):
                value = support[name]
                if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                    raise ReportInputError(f"support count is invalid: {name}")
        elif any(value is not None for value in metrics.values()):
            raise ReportInputError("scored rows must include their support counts")
        if fit.get("status") not in {"not_a_fit_arm", "exact_no_update", "actual_fit"}:
            raise ReportInputError("fit status is outside the report's explicit status set")
        for name in ("optimizer_steps", "final_redecode_calls"):
            value = fit.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReportInputError(f"fit count is invalid: {name}")
        indexed[key] = {**row, "_metrics": metrics}

    if len(case_datasets) != 8 or sum(d == "hc2" for d in case_datasets.values()) != 4 or sum(
        d == "vidstg" for d in case_datasets.values()
    ) != 4:
        raise ReportInputError("postseal case roster is not four HC and four VidSTG sources")
    if set(indexed) != {
        f"{case_id}\0{arm}" for case_id in case_datasets for arm in ARM_ORDER
    }:
        raise ReportInputError("postseal case-by-arm grid is incomplete")
    _validate_summary_against_rows(summary_doc, indexed, case_datasets)
    _validate_contrast_counts(summary_doc, indexed, case_datasets)
    return (
        rows_doc,
        summary_doc,
        indexed,
        {case_id: {"dataset": dataset} for case_id, dataset in case_datasets.items()},
    )


def _summary_entries(summary: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    entries = summary.get("arm_metric_summaries")
    if not isinstance(entries, list):
        raise ReportInputError("postseal summary is missing arm metric summaries")
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ReportInputError("arm metric summary contains a non-object entry")
        key = (entry.get("dataset"), entry.get("arm"), entry.get("metric"))
        if key in output or key[0] not in DATASET_LABELS or key[1] not in ARM_ORDER or key[2] not in SCORE_METRICS:
            raise ReportInputError("arm metric summary has a duplicate or unrecognized key")
        output[key] = entry
    expected = {
        (dataset, arm, metric)
        for dataset in DATASET_LABELS
        for arm in ARM_ORDER
        for metric in SCORE_METRICS
    }
    if set(output) != expected:
        raise ReportInputError("arm metric summary does not cover all domains, arms, and metrics")
    return output


def _validate_summary_against_rows(
    summary: dict[str, Any],
    indexed: dict[str, dict[str, Any]],
    case_datasets: dict[str, str],
) -> None:
    summaries = _summary_entries(summary)
    for dataset in DATASET_LABELS:
        cases = [case_id for case_id, value in case_datasets.items() if value == dataset]
        for arm in ARM_ORDER:
            for metric in SCORE_METRICS:
                values = [indexed[f"{case_id}\0{arm}"]["_metrics"][metric] for case_id in cases]
                known = [value for value in values if value is not None]
                entry = summaries[(dataset, arm, metric)]
                if entry.get("n_valid") != len(known) or entry.get("n_unknown") != len(values) - len(known):
                    raise ReportInputError("summary valid/unknown counts disagree with source-level rows")
                mean = entry.get("mean")
                if known:
                    if not _finite_number(mean) or abs(float(mean) - sum(known) / len(known)) > 1e-10:
                        raise ReportInputError("summary mean disagrees with source-level rows")
                elif mean is not None:
                    raise ReportInputError("all-unknown metric summary must have a null mean")
                unknown_ids = entry.get("unknown_case_ids")
                expected_unknown = {
                    case_id for case_id in cases if indexed[f"{case_id}\0{arm}"]["_metrics"][metric] is None
                }
                if not isinstance(unknown_ids, list) or set(unknown_ids) != expected_unknown:
                    raise ReportInputError("summary unknown case list disagrees with source-level rows")


def _contrast_pairs() -> list[tuple[str, str, str]]:
    # Tuple order is (contrast label, comparison arm, baseline arm), matching
    # the frozen scorer's delta convention: comparison minus baseline.
    pairs = [("T_minus_F", "T", "F"), ("native_teacher_direct_minus_T", "native_teacher_direct", "T")]
    pairs.extend((f"{arm}_minus_T", arm, "T") for arm in ARM_ORDER if arm.startswith("T_"))
    pairs.extend((f"{arm}_minus_F", arm, "F") for arm in SPATIAL_ALGORITHMS)
    return pairs


def _validate_contrast_counts(
    summary: dict[str, Any], indexed: dict[str, dict[str, Any]], case_datasets: dict[str, str]
) -> None:
    contrasts = summary.get("contrasts")
    if not isinstance(contrasts, list):
        raise ReportInputError("postseal summary is missing paired contrasts")
    expected_triplets = set(_contrast_pairs())
    seen: set[tuple[str, str, str, str]] = set()
    for row in contrasts:
        if not isinstance(row, dict):
            raise ReportInputError("contrast list contains a non-object entry")
        dataset, metric = row.get("dataset"), row.get("metric")
        triplet = (row.get("contrast"), row.get("comparison_arm"), row.get("baseline_arm"))
        if dataset not in DATASET_LABELS or metric not in SCORE_METRICS or triplet not in expected_triplets:
            raise ReportInputError("contrast has an unrecognized domain, metric, or arm pair")
        key = (dataset, metric, triplet[0], triplet[1] + "\0" + triplet[2])
        if key in seen:
            raise ReportInputError("duplicate contrast record")
        seen.add(key)
        cases = [case_id for case_id, value in case_datasets.items() if value == dataset]
        deltas: list[float] = []
        unknown = 0
        for case_id in cases:
            comp = indexed[f"{case_id}\0{triplet[1]}"]["_metrics"][metric]
            base = indexed[f"{case_id}\0{triplet[2]}"]["_metrics"][metric]
            if base is None or comp is None:
                unknown += 1
            else:
                # The scorer's sign epsilon is in the raw 0-to-1 metric scale.
                deltas.append(comp - base)
        tol = row.get("neutral_tolerance")
        if not _finite_number(tol, minimum=0):
            raise ReportInputError("contrast neutral tolerance is invalid")
        positive = sum(delta > float(tol) for delta in deltas)
        negative = sum(delta < -float(tol) for delta in deltas)
        neutral = len(deltas) - positive - negative
        if (
            row.get("n_pairs") != len(deltas)
            or row.get("unknown_count") != unknown
            or row.get("positive_count") != positive
            or row.get("negative_count") != negative
            or row.get("neutral_count") != neutral
        ):
            raise ReportInputError("paired contrast counts disagree with the source-level score grid")
    expected_count = len(DATASET_LABELS) * len(SCORE_METRICS) * len(expected_triplets)
    if len(seen) != expected_count:
        raise ReportInputError("paired contrasts are missing a domain, metric, or fixed comparison")


def _load_cost_rows(
    case_datasets: dict[str, dict[str, Any]], indexed_rows: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    cost_doc = _read_json(BASE / "ROOT_COST_READBACK.json")
    if (
        cost_doc.get("schema") != "stvg-native-instance-cost-readback-v1"
        or cost_doc.get("status") != "complete"
        or cost_doc.get("GT_read") is not False
        or cost_doc.get("count") != 8
    ):
        raise ReportInputError("scalar-only student cost readback is not complete or GT-free")
    records = cost_doc.get("records")
    if not isinstance(records, list) or len(records) != 8:
        raise ReportInputError("student cost readback has an invalid record count")
    costs: dict[str, dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, dict):
            raise ReportInputError("student cost record is not an object")
        case_id = row.get("case_id")
        if case_id not in case_datasets or case_id in costs:
            raise ReportInputError("student cost readback does not match the sealed case set")
        if row.get("dataset") != case_datasets[case_id]["dataset"]:
            raise ReportInputError("student cost domain does not match the sealed case")
        if row.get("source_case_sha256") != indexed_rows[f"{case_id}\0T"]["bindings"].get("source_case_sha256"):
            raise ReportInputError("student cost source hash does not match score rows")
        if row.get("student_artifact_sha256") != indexed_rows[f"{case_id}\0T"]["bindings"].get("student_artifact_sha256"):
            raise ReportInputError("student cost artifact hash does not match score rows")
        algs = row.get("algorithm_costs")
        if not isinstance(algs, dict) or set(algs) != set(SPATIAL_ALGORITHMS):
            raise ReportInputError("student cost algorithms do not match the three frozen spatial arms")
        for algorithm, value in algs.items():
            if not isinstance(value, dict):
                raise ReportInputError("student algorithm timing record is malformed")
            seconds = value.get("seconds")
            if seconds is not None and not _finite_number(seconds, minimum=0):
                raise ReportInputError("student algorithm timing is not a finite nonnegative number")
            for name in ("optimizer_steps", "final_redecode_calls"):
                if isinstance(value.get(name), bool) or not isinstance(value.get(name), int) or value[name] < 0:
                    raise ReportInputError("student algorithm timing counts are malformed")
            fit = indexed_rows[f"{case_id}\0{algorithm}"]["fit"]
            if (
                value["optimizer_steps"] != fit["optimizer_steps"]
                or value["final_redecode_calls"] != fit["final_redecode_calls"]
            ):
                raise ReportInputError("student cost call counts do not match postseal fit readout")
            if value.get("fit_present") is not None and value.get("fit_present") is not (fit["status"] == "actual_fit"):
                raise ReportInputError("student cost fit-presence flag disagrees with postseal fit status")
        capture_seconds = row.get("source_capture_seconds")
        if capture_seconds is not None and not _finite_number(capture_seconds, minimum=0):
            raise ReportInputError("source capture time is not finite and nonnegative")
        costs[case_id] = dict(row)
    if set(costs) != set(case_datasets):
        raise ReportInputError("student cost readback omits one or more source cases")

    for case_id, source in case_datasets.items():
        tracking = _read_json(BASE / TRACKING_RECEIPT_REL.format(case_id=case_id))
        if tracking.get("status") != "pass" or tracking.get("GT_read") is not False:
            raise ReportInputError("tracking receipt is not a passing GT-free receipt")
        if tracking.get("reused") not in (True, False):
            raise ReportInputError("tracking receipt does not state whether the track was reused")
        track_seconds = tracking.get("actual_tracking_wall_s")
        if not _finite_number(track_seconds, minimum=0):
            raise ReportInputError("tracking receipt lacks a finite measured wall time")

        teacher = _read_json(BASE / TEACHER_RECEIPT_REL.format(case_id=case_id))
        if (
            teacher.get("schema") != "stvg-native-instance-teacher-receipt-v1"
            or teacher.get("status") != "complete"
            or teacher.get("GT_read") is not False
            or teacher.get("query_text_in_receipt") is not False
            or teacher.get("dataset") != source["dataset"]
            or teacher.get("source_case_sha256") != costs[case_id]["source_case_sha256"]
        ):
            raise ReportInputError("teacher receipt does not bind to the same query-free source and domain")
        teacher_cost = teacher.get("costs")
        if not isinstance(teacher_cost, dict):
            raise ReportInputError("teacher receipt has no cost object")
        selected_teacher_costs: dict[str, float | None] = {}
        for name in ("worker_wall_s", "model_load_s", "model_preprocess_s", "model_generate_s"):
            value = teacher_cost.get(name)
            if value is not None and not _finite_number(value, minimum=0):
                raise ReportInputError("teacher timing is not finite and nonnegative")
            selected_teacher_costs[name] = float(value) if value is not None else None
        if teacher.get("tracking_acceptance_sha256") != _sha256(BASE / TRACKING_RECEIPT_REL.format(case_id=case_id)):
            raise ReportInputError("teacher cost receipt is not bound to the accepted tracking receipt")
        row_teacher_sha = indexed_rows[f"{case_id}\0T"]["bindings"].get("teacher_receipt_sha256")
        if row_teacher_sha is not None and row_teacher_sha != _sha256(BASE / TEACHER_RECEIPT_REL.format(case_id=case_id)):
            raise ReportInputError("teacher receipt bytes do not match the sealed score-row binding")
        costs[case_id]["_tracking"] = {
            "reused": tracking["reused"],
            "actual_tracking_wall_s": float(track_seconds),
        }
        costs[case_id]["_teacher"] = selected_teacher_costs
    return costs


def _fmt_metric(value: float | None) -> str:
    return "unknown" if value is None else f"{100.0 * value:.3f}"


def _fmt_seconds(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.3f}"


def _safe_row_status(row: dict[str, Any]) -> str:
    known = {"scored", "scored_unknown", "teacher_invalid", "invalid", "unknown", "abstained", "truncated"}
    if any(value is not None for value in row["_metrics"].values()):
        return "scored"
    status = row.get("status")
    return status if isinstance(status, str) and status in known else "unknown"


def _format_source_table(
    indexed: dict[str, dict[str, Any]],
    cases: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "| Domain | Source | Arm | Status | vIoU | tIoU | dense sIoU | supervised sIoU | unsupervised sIoU | Source-grid positions | Supervised anchors | Unsupervised grid positions | GT-known dense / supervised / unsupervised frames | Present frames / total | Interval-present / interval total | Fit status | Steps | Final re-decodes |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|",
    ]
    ordered_cases = sorted(cases, key=lambda cid: (0 if cases[cid]["dataset"] == "hc2" else 1, _anonymous_source(cid, cases[cid]["dataset"])))
    for case_id in ordered_cases:
        dataset = cases[case_id]["dataset"]
        for arm in ARM_ORDER:
            row = indexed[f"{case_id}\0{arm}"]
            metric_cells = [_fmt_metric(row["_metrics"][name]) for name in SCORE_METRICS]
            coverage = row.get("coverage")
            if coverage is None:
                coverage_cell = "unknown"
                interval_coverage_cell = "unknown"
            else:
                coverage_cell = f"{coverage.get('present_frame_count', 'unknown')} / {coverage.get('frame_count', 'unknown')}"
                interval_coverage_cell = f"{coverage.get('interval_present_frame_count', 'unknown')} / {coverage.get('interval_frame_count', 'unknown')}"
            fit = row["fit"]
            support = row.get("support") or {}
            support_count = support.get("supervised_anchor_count")
            unknown_positions = support.get("unsupervised_position_count")
            steps = fit.get("optimizer_steps")
            redecode = fit.get("final_redecode_calls")
            cells = [
                DATASET_LABELS[dataset],
                _anonymous_source(case_id, dataset),
                ARM_SHORT[arm],
                _safe_row_status(row),
                *metric_cells,
                str(support.get("source_grid_positions")) if support.get("source_grid_positions") is not None else "unknown",
                str(support_count) if support_count is not None else "unknown",
                str(unknown_positions) if unknown_positions is not None else "unknown",
                " / ".join(
                    str(support.get(key)) if support.get(key) is not None else "unknown"
                    for key in (
                        "dense_gt_known_frame_count",
                        "supervised_gt_known_frame_count",
                        "unsupervised_gt_known_frame_count",
                    )
                ),
                coverage_cell,
                interval_coverage_cell,
                fit["status"],
                str(steps),
                str(redecode),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_arm_summary(summary: dict[str, Any]) -> str:
    entries = _summary_entries(summary)
    lines = [
        "| Domain | Arm | vIoU % (valid/unknown) | tIoU % (valid/unknown) | dense sIoU % (valid/unknown) | supervised sIoU % (valid/unknown) | unsupervised sIoU % (valid/unknown) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASET_LABELS:
        for arm in ARM_ORDER:
            values = []
            for metric in SCORE_METRICS:
                entry = entries[(dataset, arm, metric)]
                mean = entry.get("mean")
                mean_cell = "unknown" if mean is None else f"{100.0 * float(mean):.3f}"
                values.append(f"{mean_cell} ({entry['n_valid']}/{entry['n_unknown']})")
            lines.append(
                "| "
                + " | ".join([DATASET_LABELS[dataset], ARM_LABELS[arm], *values])
                + " |"
            )
    return "\n".join(lines)


def _format_contrasts(summary: dict[str, Any]) -> str:
    lines = [
        "| Domain | Contrast | Metric | Paired n | Positive | Negative | Neutral | Unknown | Mean delta (pp) | Median delta (pp) | 95% paired bootstrap CI (pp) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = summary["contrasts"]
    rows = sorted(rows, key=lambda r: (list(DATASET_LABELS).index(r["dataset"]), r["contrast"], SCORE_METRICS.index(r["metric"])))
    for row in rows:
        ci = row.get("bootstrap_ci95_delta_pp")
        if isinstance(ci, dict) and _finite_number(ci.get("lower")) and _finite_number(ci.get("upper")):
            ci_text = f"[{float(ci['lower']):+.3f}, {float(ci['upper']):+.3f}]"
        else:
            ci_text = "not estimable"
        lines.append(
            "| "
            + " | ".join(
                [
                    DATASET_LABELS[row["dataset"]],
                    row["contrast"],
                    METRIC_LABELS[row["metric"]],
                    str(row["n_pairs"]),
                    str(row["positive_count"]),
                    str(row["negative_count"]),
                    str(row["neutral_count"]),
                    str(row["unknown_count"]),
                    f"{float(row['mean_delta_pp']):+.3f}" if _finite_number(row.get("mean_delta_pp")) else "unknown",
                    f"{float(row['median_delta_pp']):+.3f}" if _finite_number(row.get("median_delta_pp")) else "unknown",
                    ci_text,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _format_cost_table(cases: dict[str, dict[str, Any]], cost_rows: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Domain | Source | Tracking history | Tracking wall (s) | Teacher worker wall (s) | Model load (s) | Preprocess (s) | Generation (s) | Source capture (s) | Direct fit+decode (s) | IoU fit+decode (s) | Weighted-IoU fit+decode (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered_cases = sorted(cases, key=lambda cid: (0 if cases[cid]["dataset"] == "hc2" else 1, _anonymous_source(cid, cases[cid]["dataset"])))
    for case_id in ordered_cases:
        dataset = cases[case_id]["dataset"]
        row = cost_rows[case_id]
        tracking = row["_tracking"]
        teacher = row["_teacher"]
        history = "reused historical" if tracking["reused"] else "current run"
        capture = row.get("source_capture_seconds")
        algorithm_values = row["algorithm_costs"]
        lines.append(
            "| "
            + " | ".join(
                [
                    DATASET_LABELS[dataset],
                    _anonymous_source(case_id, dataset),
                    history,
                    _fmt_seconds(tracking["actual_tracking_wall_s"]),
                    _fmt_seconds(teacher["worker_wall_s"]),
                    _fmt_seconds(teacher["model_load_s"]),
                    _fmt_seconds(teacher["model_preprocess_s"]),
                    _fmt_seconds(teacher["model_generate_s"]),
                    _fmt_seconds(capture),
                    _fmt_seconds(algorithm_values["direct"].get("seconds")),
                    _fmt_seconds(algorithm_values["iou"].get("seconds")),
                    _fmt_seconds(algorithm_values["reliability_weighted_iou"].get("seconds")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _seconds_total(values: list[float | None]) -> tuple[float | None, int, int]:
    observed = [value for value in values if value is not None]
    missing = len(values) - len(observed)
    return (sum(observed) if observed else None, len(observed), missing)


def _format_cost_summary(cases: dict[str, dict[str, Any]], cost_rows: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Domain | Historical tracking total (s) | Reused track count | Current-run tracking total (s) | Teacher worker total (s) | Student source capture total (s) | Direct timed fit+decode (s; timed / missing) | IoU timed fit+decode (s; timed / missing) | Weighted-IoU timed fit+decode (s; timed / missing) | Current incremental subtotal before fit (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASET_LABELS:
        ids = [cid for cid, row in cases.items() if row["dataset"] == dataset]
        selected = [cost_rows[cid] for cid in ids]
        track_all, track_all_n, track_missing = _seconds_total([r["_tracking"]["actual_tracking_wall_s"] for r in selected])
        track_current, track_current_n, current_missing = _seconds_total(
            [r["_tracking"]["actual_tracking_wall_s"] if not r["_tracking"]["reused"] else None for r in selected]
        )
        current_track_count = sum(not r["_tracking"]["reused"] for r in selected)
        teacher_total, teacher_n, teacher_missing = _seconds_total([r["_teacher"]["worker_wall_s"] for r in selected])
        capture_total, capture_n, capture_missing = _seconds_total([r.get("source_capture_seconds") for r in selected])
        reused_count = sum(bool(r["_tracking"]["reused"]) for r in selected)
        current_base_values: list[float | None] = []
        for row in selected:
            teacher_s = row["_teacher"]["worker_wall_s"]
            capture_s = row.get("source_capture_seconds")
            # A reused tracking result contributes no newly executed detector
            # time to this run; its measured historical time is shown above.
            # Cache lookup overhead is not separately timed, so no duration is
            # imputed for it.
            track_increment = 0.0 if row["_tracking"]["reused"] else row["_tracking"]["actual_tracking_wall_s"]
            current_base_values.append(
                float(track_increment) + float(teacher_s) + float(capture_s)
                if teacher_s is not None and capture_s is not None
                else None
            )
        current_base, current_base_n, current_base_missing = _seconds_total(current_base_values)
        fit_cells = []
        for algorithm in SPATIAL_ALGORITHMS:
            entries = [r["algorithm_costs"][algorithm] for r in selected]
            total, n_timed, n_missing = _seconds_total([entry.get("seconds") for entry in entries])
            no_update = sum(
                entry.get("seconds") is None
                and entry.get("optimizer_steps") == 0
                and entry.get("final_redecode_calls") == 0
                for entry in entries
            )
            untimed_fit = n_missing - no_update
            fit_cells.append(
                ("not measured" if total is None else f"{total:.3f}")
                + f" (timed {n_timed}; exact no-update {no_update}; untimed fit {untimed_fit})"
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    DATASET_LABELS[dataset],
                    ("not measured" if track_all is None else f"{track_all:.3f}") + f" ({track_all_n}/{len(selected)} timed)",
                    str(reused_count),
                    (
                        f"none (all {len(selected)} reused)"
                        if current_track_count == 0
                        else ("not measured" if track_current is None else f"{track_current:.3f}")
                        + f" ({track_current_n}/{current_track_count} timed current runs)"
                    ),
                    ("not measured" if teacher_total is None else f"{teacher_total:.3f}") + f" ({teacher_n}/{len(selected)} timed)",
                    ("not measured" if capture_total is None else f"{capture_total:.3f}") + f" ({capture_n}/{len(selected)} timed)",
                    *fit_cells,
                    ("not measured" if current_base is None else f"{current_base:.3f}")
                    + f" ({current_base_n}/{current_base_n + current_base_missing})",
                ]
            )
            + " |"
        )
        # These values are intentionally validated/read but not serialized; they
        # make missing-time state explicit without replacing it by zero.
        _ = track_missing, current_missing, teacher_missing, capture_missing
    return "\n".join(lines)


def _report_text(
    rows_doc: dict[str, Any],
    summary: dict[str, Any],
    indexed: dict[str, dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    cost_rows: dict[str, dict[str, Any]],
) -> str:
    aggregation = summary["aggregation"]
    return f"""# Native detector–tracker and semantic-reference spatial update: fixed eight-source readout

## Scope and evidence

This report covers eight historically exposed development sources, four from HC and four from VidSTG. It is not an independent confirmation set. Predictions were sealed before the final metric read. The final evaluation uses the official dataset annotations; no metric below is a GT-free estimate. The model output and training/teacher traces are not treated as proof that the chosen detector track is the semantically correct instance.

The five primary arms are Temporal-only (T), the native teacher's direct track geometry, and T combined with direct, IoU, or reliability-weighted-IoU spatial adaptation. F and the three spatial-only arms are attribution controls. F/T are the preserved source baselines; the T+S arms use the same old temporal interval while changing the spatial readout. Every source-arm row, including unknown metrics and exact no-update rows, is retained below. Domain summaries remain separate; no pooled cross-domain estimate is reported.

The R1 checkpoint is the STVG-R1 “7B” model family, with an inspected parameter count of 8,292,166,656 (about 8.292 billion parameters). The native spatial reference path also runs YOLOv12 and SAM2 tracking over all target categories, so its cost includes a separate detection/segmentation stage. The YOLO checkpoint compatibility path uses half-precision fused QKV through PyTorch SDPA and does not claim bitwise equivalence to the upstream flash-attention path. Tracking used a bounded two-frame decode cache. These are implementation constraints that affect reproducibility and resource cost.

The reliability-weighted-IoU rule uses observed mask presence and adjacent-mask IoU as a proxy weight. Rapid motion can lower adjacent-mask IoU even for a valid moving target, so the value can penalize motion. It is not calibrated confidence, identity probability, or a semantic correctness score.

The sealed metric set does not estimate full candidate-pool detection recall. A valid native R1 selection can still be scored even when no fixed QA-center mask supplies a spatial update anchor; that case remains an exact no-update student branch, not evidence that the teacher abstained.

## Per-domain arm summaries

Metrics are percentages of their 0-to-1 values. The parenthesized counts are `valid / unknown` among four historical sources. Unknown cells remain unknown; they are not imputed as zero.

{_format_arm_summary(summary)}

## Paired contrasts

Positive, negative, neutral, and unknown counts are source-level pairs. The 95% intervals use {aggregation.get('bootstrap_replicates')} paired source bootstrap replicates with seed {aggregation.get('seed')}, separately within each domain. With four historically exposed sources per domain these are exploratory descriptive intervals, not a confirmation test, selection gate, or pooled estimate.

{_format_contrasts(summary)}

## All source-level predictions and readouts

Scores are percentages. The operation columns retain support, coverage, optimizer, and final-decode status so an unknown result is distinguishable from an exact no-update branch.

{_format_source_table(indexed, cases)}

## Measured cost and reuse

Tracking wall time for a reused receipt is the actual historical tracking time; it is not free and is not this run's incremental tracking time. Current incremental subtotal excludes reused tracking and includes only current tracking where applicable, teacher worker wall time, and student source capture. Tracking-cache lookup overhead is not separately timed. Algorithm `seconds` covers that arm's fit and full final re-decode. Exact no-update receipts may have no timing, shown as `not measured`; no missing duration is replaced with zero. Domain fit totals distinguish timed fits, exact no-update arms, and untimed fits. Teacher model-load, preprocessing, and generation measurements are components of worker time and must not be added to it. The report does not count a historical cached T reconstruction as free: its retrieval/reinference latency is not measured in this scope and is excluded from all-in deployment latency.

{_format_cost_table(cases, cost_rows)}

Domain totals follow; timed fit totals include only receipts with an actual measured duration. The current incremental subtotal is reported before spatial fit/redecode time, which remains in its own algorithm columns.

{_format_cost_summary(cases, cost_rows)}

One initial teacher-worker attempt stopped at the configured 4-GiB free-GPU-memory guard before any teacher result was accepted; this was not a CUDA out-of-memory event. That attempt produced no token output, and its generation-call count is unknown. The later worker configuration changed the placement budgets from 20/4 GiB GPU/CPU to 12/8 GiB while retaining BF16, prompt, visual input, and token cap. Its failed-attempt resource cost is not represented by the successful per-source receipt totals above. Those totals therefore describe the accepted successful path, not every attempted operation.

## Limits and decision boundary

The panel is small and historically exposed. Per-domain means, source-level variation, negative cases, and unknowns must be read together. A favorable arm mean does not establish stable performance or transfer to unseen sources. Detector track availability and R1's response are not a semantic identity verification: errors in detection, mask propagation, event interpretation, or target selection can remain even when the output parses and the mask track is well formed.

The direct native teacher arm measures the selected YOLO+SAM2/R1 output without spatial optimization. The adaptation arms use only their locked reference rule and original optimizer controls; the report does not select a new threshold or promote a method. The reliability proxy is a hand-defined weighting rule, not a learned or calibrated uncertainty estimate. Deployment choice remains governed by the project method registry, not this development report.

The score rows are bound to prediction barrier SHA `{rows_doc['prediction_barrier']['sha256']}`. This report intentionally exposes scalar scores and anonymous aliases only; it does not export raw queries, media, masks, annotation coordinates, model weights, or per-case prediction tensors.
"""


def _make_figure(summary: dict[str, Any], indexed: dict[str, dict[str, Any]], cases: dict[str, dict[str, Any]]) -> None:
    summaries = _summary_entries(summary)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.3), sharey=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    for ax, dataset in zip(axes, DATASET_LABELS):
        ids = [cid for cid, value in cases.items() if value["dataset"] == dataset]
        x = list(range(len(ARM_ORDER)))
        for i, arm in enumerate(ARM_ORDER):
            entry = summaries[(dataset, arm, "m_vIoU")]
            mean = entry.get("mean")
            if mean is not None:
                ax.bar(i, 100.0 * float(mean), width=0.7, color=ARM_COLORS[arm], alpha=0.72, zorder=1)
            for j, case_id in enumerate(sorted(ids, key=lambda cid: _anonymous_source(cid, dataset))):
                value = indexed[f"{case_id}\0{arm}"]["_metrics"]["m_vIoU"]
                if value is None:
                    continue
                offset = (j - (len(ids) - 1) / 2) * 0.105
                ax.scatter(
                    [i + offset],
                    [100.0 * value],
                    color=f"C{j}",
                    edgecolor="white",
                    linewidth=0.5,
                    s=30,
                    zorder=3,
                    label=_anonymous_source(case_id, dataset) if i == 0 else None,
                )
            ax.text(i, 0.985, f"{entry['n_valid']}/4", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color="#333333")
        ax.set_xticks(x, [ARM_SHORT[arm] for arm in ARM_ORDER], rotation=42, ha="right")
        ax.set_title(f"{DATASET_LABELS[dataset]} · four exposed sources")
        ax.set_ylabel("Mean full-video vIoU (%)")
        ax.grid(axis="y", alpha=0.2, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Five primary arms and frozen/spatial attribution controls", y=0.94, fontsize=13)
    fig.text(
        0.5,
        0.015,
        "Bars show valid-source means; colored points show each anonymous source. `n/4` is printed above every arm. Unknowns are retained in the report. No pooled domain estimate.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.90))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=180, bbox_inches="tight", metadata={"Title": "Native instance STVG summary"})
    with PdfPages(PDF_PATH, metadata={"Title": "Native instance STVG summary", "Author": "STVG research report"}) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows_doc, summary, indexed, cases = _load_scored_rows()
    cost_rows = _load_cost_rows(cases, indexed)
    report = _report_text(rows_doc, summary, indexed, cases, cost_rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    _make_figure(summary, indexed, cases)
    print("Wrote anonymous report and summary figures after post-seal input validation.")


def _self_test() -> None:
    """Exercise formatting invariants without touching workspace experiment data."""
    assert _anonymous_source("hc2_000", "hc2") == "HC01"
    assert _anonymous_source("vidstg_003", "vidstg") == "V04"
    assert _fmt_metric(None) == "unknown"
    assert _fmt_seconds(None) == "not measured"
    for bad_case, bad_dataset in (("hc2_004", "hc2"), ("vidstg_000", "hc2"), ("arbitrary", "hc2")):
        try:
            _anonymous_source(bad_case, bad_dataset)
        except ReportInputError:
            pass
        else:
            raise AssertionError("non-panel case key was not rejected")
    primary = set(PRIMARY_ARMS)
    attribution = set(ATTRIBUTION_ARMS)
    assert len(primary) == 5 and len(attribution) == 4 and not (primary & attribution)
    expected = {
        "T_minus_F",
        "native_teacher_direct_minus_T",
        "T_direct_minus_T",
        "T_iou_minus_T",
        "T_reliability_weighted_iou_minus_T",
        "direct_minus_F",
        "iou_minus_F",
        "reliability_weighted_iou_minus_F",
    }
    assert {name for name, _, _ in _contrast_pairs()} == expected
    assert "unknown" in _fmt_metric(None) and "not measured" in _fmt_seconds(None)
    synthetic_cases = {
        **{f"hc2_{i:03d}": {"dataset": "hc2"} for i in range(4)},
        **{f"vidstg_{i:03d}": {"dataset": "vidstg"} for i in range(4)},
    }
    synthetic_indexed: dict[str, dict[str, Any]] = {}
    synthetic_summaries: list[dict[str, Any]] = []
    for dataset in DATASET_LABELS:
        ids = [cid for cid, row in synthetic_cases.items() if row["dataset"] == dataset]
        for arm_index, arm in enumerate(ARM_ORDER):
            for metric_index, metric in enumerate(SCORE_METRICS):
                vals = []
                unknown = []
                for case_index, case_id in enumerate(ids):
                    value = 0.25 + 0.02 * arm_index + 0.01 * case_index + 0.001 * metric_index
                    if dataset == "hc2" and case_index == 3 and arm == "T_iou" and metric == "m_sIoU_unsupervised_GT":
                        value = None
                    key = f"{case_id}\0{arm}"
                    row = synthetic_indexed.setdefault(
                        key,
                        {
                            "case_id": case_id,
                            "dataset": dataset,
                            "arm": arm,
                            "status": "unknown" if value is None else "scored",
                            "_metrics": {name: None for name in SCORE_METRICS},
                            "support": {
                                "source_grid_positions": 4,
                                "supervised_anchor_count": 0,
                                "unsupervised_position_count": 4,
                                "dense_gt_known_frame_count": 0,
                                "supervised_gt_known_frame_count": 0,
                                "unsupervised_gt_known_frame_count": 0,
                            },
                            "coverage": None,
                            "fit": {
                                "status": "exact_no_update" if arm == "iou" else "not_a_fit_arm" if arm in {"F", "T", "native_teacher_direct"} else "actual_fit",
                                "optimizer_steps": 0 if arm == "iou" else 1,
                                "final_redecode_calls": 0 if arm == "iou" else 1,
                            },
                        },
                    )
                    row["_metrics"][metric] = value
                    if value is None:
                        unknown.append(case_id)
                    else:
                        vals.append(value)
                synthetic_summaries.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "metric": metric,
                        "n_valid": len(vals),
                        "n_unknown": len(unknown),
                        "unknown_case_ids": unknown,
                        "mean": sum(vals) / len(vals) if vals else None,
                        "median": 0.0 if vals else None,
                        "min": min(vals) if vals else None,
                        "max": max(vals) if vals else None,
                    }
                )
    synthetic_contrasts: list[dict[str, Any]] = []
    for dataset in DATASET_LABELS:
        ids = [cid for cid, row in synthetic_cases.items() if row["dataset"] == dataset]
        for contrast, comp_arm, base_arm in _contrast_pairs():
            for metric in SCORE_METRICS:
                deltas = []
                unknown = 0
                for case_id in ids:
                    comp = synthetic_indexed[f"{case_id}\0{comp_arm}"]["_metrics"][metric]
                    base = synthetic_indexed[f"{case_id}\0{base_arm}"]["_metrics"][metric]
                    if comp is None or base is None:
                        unknown += 1
                    else:
                        deltas.append(comp - base)
                tol = 1e-12
                positive = sum(delta > tol for delta in deltas)
                negative = sum(delta < -tol for delta in deltas)
                neutral = len(deltas) - positive - negative
                pp = [delta * 100.0 for delta in deltas]
                synthetic_contrasts.append(
                    {
                        "dataset": dataset,
                        "contrast": contrast,
                        "comparison_arm": comp_arm,
                        "baseline_arm": base_arm,
                        "metric": metric,
                        "n_pairs": len(deltas),
                        "unknown_count": unknown,
                        "positive_count": positive,
                        "negative_count": negative,
                        "neutral_count": neutral,
                        "neutral_tolerance": tol,
                        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
                        "mean_delta_pp": sum(pp) / len(pp) if pp else None,
                        "median_delta_pp": 0.0 if pp else None,
                        "bootstrap_ci95_delta_pp": {"lower": -1.0, "upper": 1.0} if pp else None,
                    }
                )
    synthetic_summary = {
        "arm_metric_summaries": synthetic_summaries,
        "contrasts": synthetic_contrasts,
    }
    _validate_summary_against_rows(
        synthetic_summary,
        synthetic_indexed,
        {case_id: row["dataset"] for case_id, row in synthetic_cases.items()},
    )
    _validate_contrast_counts(
        synthetic_summary,
        synthetic_indexed,
        {case_id: row["dataset"] for case_id, row in synthetic_cases.items()},
    )
    assert "hc2_000" not in _format_arm_summary(synthetic_summary)
    assert "unknown" in _format_arm_summary(synthetic_summary)
    assert "HC01" in _format_source_table(synthetic_indexed, synthetic_cases)
    assert "Unknown" in _format_contrasts(synthetic_summary)
    synthetic_costs: dict[str, dict[str, Any]] = {}
    for index, (case_id, source) in enumerate(synthetic_cases.items()):
        synthetic_costs[case_id] = {
            "source_capture_seconds": 1.0,
            "algorithm_costs": {
                algorithm: {
                    "seconds": None if index == 0 and algorithm == "iou" else 2.0,
                    "optimizer_steps": 0 if index == 0 and algorithm == "iou" else 5,
                    "final_redecode_calls": 0 if index == 0 and algorithm == "iou" else 1,
                }
                for algorithm in SPATIAL_ALGORITHMS
            },
            "_tracking": {"reused": index < 2, "actual_tracking_wall_s": 3.0},
            "_teacher": {
                "worker_wall_s": 4.0,
                "model_load_s": 1.0,
                "model_preprocess_s": 0.5,
                "model_generate_s": 2.0,
            },
        }
    synthetic_cost_table = _format_cost_table(synthetic_cases, synthetic_costs)
    synthetic_cost_summary = _format_cost_summary(synthetic_cases, synthetic_costs)
    assert "reused historical" in synthetic_cost_table
    assert "exact no-update 1" in synthetic_cost_summary
    print("Synthetic report-contract checks passed; no experiment files were read.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run synthetic checks without reading experiment files")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        main()
