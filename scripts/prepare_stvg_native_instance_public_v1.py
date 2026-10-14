"""Stage an anonymous public export for the sealed native-instance STVG study.

This exporter is deliberately not run automatically. It reads only sealed
scalar readbacks, approved timing receipts, source/runtime hashes, and visual
review metadata. It does not open checkpoints, source-case tensors, teacher
JSON, annotations, masks, marker videos, or source videos, and it never pushes.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_native_instance_spatial_opd_v1"
OUT = ROOT / "results/stvg_native_instance/2026-10-14"
PUBLIC_STAGE = BASE / "public_stage"
MANIFEST_PATH = BASE / "PUBLIC_MANIFEST.json"
SOURCE_READBACK_PATH = BASE / "ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json"

REPORT_PATH = ROOT / "docs/STVG_NATIVE_INSTANCE_V1_REVIEWED.md"
PROTOCOL_PATH = ROOT / "protocols/stvg_native_instance_spatial_opd_v1.md"
PDF_PATH = BASE / "figures/native_instance_summary_reviewed.pdf"
PNG_PATH = BASE / "figures/native_instance_summary_reviewed.png"

CASE_IDS = tuple([f"hc2_{i:03d}" for i in range(4)]
                 + [f"vidstg_{i:03d}" for i in range(4)])
CASE_ALIAS = {cid: ("HC" if cid.startswith("hc2_") else "V")
              + f"{int(cid[-3:]) + 1:02d}" for cid in CASE_IDS}
DATASETS = {"hc2": "HC", "vidstg": "VidSTG"}
ARMS = ("F", "T", "native_teacher_direct", "direct", "iou",
        "reliability_weighted_iou", "T_direct", "T_iou",
        "T_reliability_weighted_iou")
SPATIAL = ("direct", "iou", "reliability_weighted_iou")
METRICS = ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT",
           "m_sIoU_supervised_GT", "m_sIoU_unsupervised_GT")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")

RUNTIME_NAMES = ("TRACKING_RUNTIME.json", "TEACHER_RUNTIME.json", "STUDENT_RUNTIME.json")
PREDICTION_BARRIER_NAME = "GLOBAL_PREDICTION_BARRIER.json"
REPORT_RECEIPT_NAME = "ROOT_POSTSEAL_READBACK.json"
VISUAL_RECEIPT_NAME = "ROOT_VISUAL_REVIEW.json"
MATH_RECEIPT_NAME = "ROOT_STUDENT_MATH_READBACK.json"
COST_RECEIPT_NAME = "ROOT_COST_READBACK.json"
GENERATION_RECEIPT_NAME = "ROOT_GENERATION_CONFIGURATION_READBACK.json"
GEOMETRY_RECEIPT_NAME = "ROOT_INDEPENDENT_GEOMETRY_READBACK.json"
POOL_COVERAGE_NAME = "POOL_CANDIDATE_COVERAGE_POSTSEAL.json"
POOL_COVERAGE_READBACK_NAME = "ROOT_POOL_COVERAGE_READBACK.json"
POOL_COVERAGE_APPENDIX_PATH = ROOT / "docs/STVG_NATIVE_INSTANCE_POOL_COVERAGE_APPENDIX.md"
POOL_COVERAGE_TOOL_PATH = "scripts/score_stvg_native_instance_pool_coverage_v1.py"
FINAL_EXECUTION_READBACK_NAME = "ROOT_FINAL_EXECUTION_READBACK.json"
FINAL_EXECUTION_SOURCES = frozenset({
    "scripts/dispatch_stvg_native_instance_postseal005.py",
    "scripts/dispatch_stvg_native_instance_postseal_dtype003.py",
    "scripts/dispatch_stvg_native_instance_postseal_dtype004.py",
    "scripts/plot_stvg_native_instance_reviewed003.py",
    "scripts/report_stvg_native_instance_reviewed002.py",
    "scripts/review_stvg_native_instance_coverage_v1.py",
    "scripts/review_stvg_native_instance_geometry_v1.py",
    "scripts/review_stvg_native_instance_math_scope002.py",
    "scripts/review_stvg_native_instance_postseal_v1.py",
    "scripts/score_stvg_native_instance_nativeclock005.py",
    "scripts/score_stvg_native_instance_pool_coverage_v1.py",
})
FIXED_PUBLIC_SOURCE_PATHS = frozenset({
    "scripts/continue_stvg_native_instance_storage002.py",
    "scripts/launch_stvg_native_instance_v1.py",
    "scripts/launch_stvg_native_instance_teacher_storage002.py",
    "scripts/prepare_stvg_native_instance_references_v1.py",
    "scripts/run_stvg_native_instance_tracking_v1.py",
    "scripts/run_stvg_native_instance_teacher_storage002.py",
    "scripts/run_stvg_native_instance_student_v1.py",
    "scripts/review_stvg_native_instance_math_v1.py",
    "scripts/review_stvg_native_instance_math_scope002.py",
    "scripts/score_stvg_native_instance_postseal_v1.py",
    "scripts/report_stvg_native_instance_v1.py",
    "scripts/report_stvg_native_instance_reviewed002.py",
    "scripts/plot_stvg_native_instance_reviewed003.py",
    "scripts/review_stvg_native_instance_postseal_v1.py",
    "scripts/review_stvg_native_instance_geometry_v1.py",
    "scripts/dispatch_stvg_native_instance_postseal_dtype003.py",
    "scripts/dispatch_stvg_native_instance_postseal_dtype004.py",
    "scripts/score_stvg_native_instance_nativeclock005.py",
    "scripts/dispatch_stvg_native_instance_postseal005.py",
    "scripts/score_stvg_native_instance_pool_coverage_v1.py",
    "scripts/review_stvg_native_instance_coverage_v1.py",
    "scripts/review_stvg_sparse_r1_v2.py",
    "scripts/score_stvg_opd_p1_v1.py",
    "scripts/diagnose_tastvg_pipeline_cpu_v1.py",
    "external/stvg_runtime/pycocotools/mask.py",
    "scripts/stvg_native_instance_spatial_fit_v1.py",
    "scripts/stvg_native_instance_common_v1.py",
    "scripts/prepare_stvg_native_instance_public_v1.py",
    "protocols/stvg_native_instance_spatial_opd_v1.md",
})

PUBLIC_SOURCE_PREFIXES = ("scripts/", "vg_tta/", "methods/", "protocols/",
                          "external/stvg-r1/")
PUBLIC_EXACT_SOURCE_PATHS = {"external/stvg_runtime/pycocotools/mask.py"}
PUBLIC_SUFFIXES = {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".txt"}
DENIED_PARTS = {"source_cases", "private_inputs", "private_views", "annotations",
                "annotation", "annos", "weights", "checkpoints", "teacher",
                "student", "TRACKING", "mask_data_filtered_1_3", "attachments",
                "conversation", "logs", "worker_exits", "partial"}
STATUSES = {"scored", "scored_unknown", "teacher_invalid", "invalid", "unknown",
            "abstained", "truncated", "valid_uncapped_selection", "invalid_output",
            "output_truncated_at_cap", "exact_no_update", "actual_fit",
            "invalid_or_abstained_no_teacher_tube", "not_a_fit_arm", "not_scored", "unscored"}
SELECTION_STATUS = {"valid", "abstain", "invalid", "invalid_output",
                    "output_truncated_at_cap", "out_of_clock_interval",
                    "unknown", "missing_status"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> Any:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required readback must be a JSON object: {path.name}")
    return value


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} schema differs from the explicit safe projection")


def _sha_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a JSON boolean")
    return value


def _int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _num(value: Any, label: str, *, minimum: float | None = None,
         maximum: float | None = None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric or null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} is below its allowed range")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label} is above its allowed range")
    return number


def _alias(case_id: Any, dataset: Any | None = None) -> str:
    if not isinstance(case_id, str) or case_id not in CASE_ALIAS:
        raise ValueError("source identity is outside the fixed eight-case panel")
    if dataset is not None and dataset != ("hc2" if case_id.startswith("hc2_") else "vidstg"):
        raise ValueError("source/domain binding differs from the fixed roster")
    return CASE_ALIAS[case_id]


def _public_path(rel: Any) -> str:
    if not isinstance(rel, str) or not rel or "\\" in rel:
        raise ValueError("source binding is not a nonempty ROOT-relative POSIX path")
    path = Path(rel)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("absolute or traversing source path is refused")
    if any(part in DENIED_PARTS for part in path.parts):
        raise ValueError("private payload path is refused")
    if (not rel.startswith(PUBLIC_SOURCE_PREFIXES) and rel not in PUBLIC_EXACT_SOURCE_PATHS
            or path.suffix.lower() not in PUBLIC_SUFFIXES
            or rel == "methods/CURRENT_METHOD.json"):
        raise ValueError("source binding is outside the public source allowlist")
    return rel


def _alias_ids(values: Any, label: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list of case identities")
    output = [_alias(value) for value in values]
    if len(output) != len(set(output)):
        raise ValueError(f"{label} contains duplicate identities")
    return sorted(output)


def _metric_map(value: Any, label: str) -> dict[str, float | None]:
    if value is None:
        return {name: None for name in METRICS}
    _exact_keys(value, set(METRICS), label)
    return {name: _num(value[name], f"{label}.{name}", minimum=0.0, maximum=1.0)
            for name in METRICS}


def _project_rows(rows_doc: Mapping[str, Any], barrier_sha: str,
                  student_barrier_sha: str) -> dict[str, Any]:
    if (rows_doc.get("schema") != "stvg-native-instance-postseal-rows-v1"
            or rows_doc.get("status") != "complete" or rows_doc.get("GT_read") is not True
            or rows_doc.get("metric_scale") != "0_to_1" or rows_doc.get("case_count") != 8
            or rows_doc.get("arm_count") != len(ARMS)):
        raise ValueError("postseal score rows are incomplete or use another frozen schema")
    barrier = rows_doc.get("prediction_barrier")
    if not isinstance(barrier, dict) or barrier.get("sha256") != barrier_sha:
        raise ValueError("postseal score rows do not bind the sealed prediction barrier")
    binds = rows_doc.get("bindings")
    binding_keys = {"roster_sha256", "reference_lock_sha256", "direct_teacher_readout_sha256",
                    "tracking_barrier_sha256", "teacher_barrier_sha256", "student_barrier_sha256"}
    _exact_keys(binds, binding_keys, "score-row bindings")
    bindings = {key: _sha_value(value, f"score bindings.{key}") for key, value in binds.items()}
    if bindings["student_barrier_sha256"] != student_barrier_sha:
        raise ValueError("score-row student barrier hash differs from sealed student phase")

    source_rows = rows_doc.get("rows")
    if not isinstance(source_rows, list) or len(source_rows) != len(CASE_IDS) * len(ARMS):
        raise ValueError("postseal rows do not contain the complete case-by-arm matrix")
    seen: set[tuple[str, str]] = set()
    bindings_by_case: dict[str, dict[str, str | None]] = {}
    output = []
    count_fields = ("source_grid_positions", "source_grid_supervised_position_count",
                    "source_grid_unsupervised_position_count", "supervised_anchor_count",
                    "unsupervised_position_count", "dense_gt_known_frame_count",
                    "source_grid_gt_known_frame_count", "off_grid_gt_known_frame_count",
                    "supervised_gt_known_frame_count", "unsupervised_gt_known_frame_count")
    coverage_counts = ("frame_count", "present_frame_count", "missing_frame_count",
                       "present_run_count", "fragmentation_gap_count", "interval_frame_count",
                       "interval_present_frame_count")
    geometry_counts = ("gt_known_frame_count", "teacher_tube_frame_overlap_count",
                       "gt_known_frames_with_present_mask_count",
                       "gt_known_frames_without_present_mask_count")
    audit_fields = {"official_independent_max_abs_error",
                    "native_missing_mask_independent_max_abs_error"}
    audit_metadata_fields = {"interval_frame_bounds_half_open", "source_grid_frames",
                             "scoring_path"}
    expected_audit_shapes = {
        frozenset({"official_independent_max_abs_error", "interval_frame_bounds_half_open",
                   "source_grid_frames"}),
        frozenset({"official_independent_max_abs_error",
                   "native_missing_mask_independent_max_abs_error", "scoring_path"}),
    }
    for row in source_rows:
        if not isinstance(row, dict):
            raise ValueError("postseal score row must be an object")
        cid, dataset, arm = row.get("case_id"), row.get("dataset"), row.get("arm")
        alias = _alias(cid, dataset)
        if arm not in ARMS:
            raise ValueError("postseal row names an unexpected arm")
        key = (alias, arm)
        if key in seen:
            raise ValueError("duplicate source-arm result row")
        seen.add(key)
        if row.get("candidate_detection_recall") != "not_measured":
            raise ValueError("candidate detection recall must remain explicitly unmeasured")
        status = row.get("status")
        if not isinstance(status, str) or status not in STATUSES:
            raise ValueError("postseal row status is not in the public status vocabulary")
        metrics = _metric_map(row.get("metrics"), f"row.{alias}.{arm}.metrics")

        support_src = row.get("support")
        support = None
        if support_src is not None:
            if not isinstance(support_src, dict):
                raise ValueError("postseal support must be an object or null")
            required_counts = set(count_fields)
            if not required_counts <= set(support_src):
                raise ValueError("postseal support count inventory is incomplete")
            support = {field: _int(support_src[field], f"support.{field}")
                       for field in count_fields}
            # Physical frame identifiers and positional lists are deliberately
            # omitted; the public table retains their aggregate counts only.

        coverage_src = row.get("coverage")
        coverage = None
        if coverage_src is not None:
            if not isinstance(coverage_src, dict):
                raise ValueError("coverage must be an object or null")
            coverage = {field: _int(coverage_src[field], f"coverage.{field}")
                        for field in coverage_counts}
            coverage["present_fraction"] = _num(
                coverage_src.get("present_fraction"), "coverage.present_fraction",
                minimum=0.0, maximum=1.0)

        teacher_geometry_src = row.get("selected_track_gt_geometry_coverage")
        teacher_geometry = None
        if teacher_geometry_src is not None:
            if not isinstance(teacher_geometry_src, dict):
                raise ValueError("selected-track coverage must be an object or null")
            teacher_geometry = {field: _int(teacher_geometry_src[field], f"coverage.{field}")
                                for field in geometry_counts}
            teacher_geometry["present_mask_fraction_of_gt_known_frames"] = _num(
                teacher_geometry_src.get("present_mask_fraction_of_gt_known_frames"),
                "selected_track_gt_geometry_coverage.present_mask_fraction",
                minimum=0.0, maximum=1.0)

        fit_src = row.get("fit")
        if not isinstance(fit_src, dict):
            raise ValueError("postseal row fit summary is missing")
        fit_status = fit_src.get("status")
        if fit_status not in {"not_a_fit_arm", "exact_no_update", "actual_fit"}:
            raise ValueError("fit status is not in the public vocabulary")
        fit = {"status": fit_status,
               "optimizer_steps": _int(fit_src.get("optimizer_steps"), "fit.optimizer_steps"),
               "final_redecode_calls": _int(fit_src.get("final_redecode_calls"),
                                             "fit.final_redecode_calls")}

        audit_src = row.get("metric_audit")
        audit = None
        if audit_src is not None:
            if (not isinstance(audit_src, dict)
                    or frozenset(audit_src) not in expected_audit_shapes
                    or not set(audit_src) <= audit_fields | audit_metadata_fields):
                raise ValueError("metric audit differs from the two locked writer schemas")
            audit = {field: _num(audit_src[field], f"metric_audit.{field}")
                     for field in sorted(set(audit_src) & audit_fields)}
            if "source_grid_frames" in audit_src:
                audit["source_grid_frame_count"] = _int(
                    audit_src["source_grid_frames"], "metric_audit.source_grid_frames")
                bounds = audit_src["interval_frame_bounds_half_open"]
                if (not isinstance(bounds, list) or len(bounds) != 2
                        or any(_num(value, "metric_audit.interval_frame_bound") is None
                               for value in bounds)
                        or float(bounds[0]) > float(bounds[1])):
                    raise ValueError("metric audit interval bounds are malformed")
            if "scoring_path" in audit_src:
                if audit_src["scoring_path"] != "official_physical_teacher_missing_as_zero":
                    raise ValueError("native teacher metric path differs from the locked scorer")
                audit["scoring_path"] = "official_physical_teacher_missing_as_zero"

        bind_src = row.get("bindings")
        row_binding_keys = {"source_case_sha256", "student_artifact_sha256",
                            "student_receipt_sha256", "teacher_record_sha256",
                            "teacher_receipt_sha256", "reference_lock_sha256"}
        _exact_keys(bind_src, row_binding_keys, "row bindings")
        row_bindings = {field: (_sha_value(value, f"row.bindings.{field}") if value is not None else None)
                        for field, value in bind_src.items()}
        if row_bindings["reference_lock_sha256"] != bindings["reference_lock_sha256"]:
            raise ValueError("row reference-lock binding differs from the report binding")
        previous = bindings_by_case.setdefault(alias, row_bindings)
        if previous != row_bindings:
            raise ValueError("source/student/teacher bindings vary across arms for one case")
        output.append({"case_id": alias, "dataset": DATASETS[dataset], "arm": arm,
                       "status": status, "metrics": metrics, "support": support,
                       "coverage": coverage,
                       "selected_track_gt_geometry_coverage": teacher_geometry,
                       "candidate_detection_recall": "not_measured", "fit": fit,
                       "metric_audit": audit, "bindings": row_bindings})
    expected = {(CASE_ALIAS[cid], arm) for cid in CASE_IDS for arm in ARMS}
    if seen != expected:
        raise ValueError("postseal public source-arm matrix is incomplete")
    return {"schema": "stvg-native-instance-public-postseal-rows-v1",
            "status": "pass", "scope": "anonymous fixed historical development panel",
            "metric_scale": "0_to_1", "case_count": 8, "arm_count": len(ARMS),
            "row_count": len(output), "candidate_detection_recall": "not_measured",
            "GT_read": True, "prediction_barrier_sha256": barrier_sha,
            "bindings": bindings, "rows": sorted(output, key=lambda row: (
                row["dataset"], row["case_id"], ARMS.index(row["arm"]))) }


def _project_summary(source: Mapping[str, Any], rows_public: Mapping[str, Any]) -> dict[str, Any]:
    if (source.get("schema") != "stvg-native-instance-postseal-summary-v1"
            or source.get("status") != "complete" or source.get("GT_read") is not True
            or source.get("metric_scale") != "0_to_1"
            or source.get("candidate_detection_recall") != "not_measured"):
        raise ValueError("postseal summary is incomplete or does not preserve recall limitation")
    aggregation = source.get("aggregation")
    expected_aggregation = {"bootstrap_replicates": 10000, "seed": 20261014,
                            "unit": "paired source case",
                            "pooled_cross_domain_gate": False,
                            "ci_used_for_arm_or_step_selection": False}
    if not isinstance(aggregation, dict) or any(aggregation.get(k) != v
                                                for k, v in expected_aggregation.items()):
        raise ValueError("summary aggregation differs from the locked descriptive-only scope")
    if source.get("case_count_by_dataset") != {"hc2": 4, "vidstg": 4}:
        raise ValueError("summary does not preserve four fixed sources per domain")
    case_datasets = {row["case_id"]: row["dataset"] for row in rows_public["rows"]
                     if row["arm"] == "F"}
    aliases_by_dataset = {domain: sorted(alias for alias, dataset in case_datasets.items()
                                         if dataset == label)
                          for domain, label in DATASETS.items()}

    arms = source.get("arm_metric_summaries")
    expected_cells = {(dataset, arm, metric) for dataset in DATASETS for arm in ARMS
                      for metric in METRICS}
    if not isinstance(arms, list) or len(arms) != len(expected_cells):
        raise ValueError("summary arm-metric matrix is incomplete")
    arm_out, seen = [], set()
    for row in arms:
        if not isinstance(row, dict):
            raise ValueError("arm summary is malformed")
        dataset, arm, metric = row.get("dataset"), row.get("arm"), row.get("metric")
        if dataset not in DATASETS or arm not in ARMS or metric not in METRICS:
            raise ValueError("arm summary has an unexpected domain, arm, or metric")
        key = (dataset, arm, metric)
        if key in seen:
            raise ValueError("duplicate arm-metric summary")
        seen.add(key)
        unknown_ids = _alias_ids(row.get("unknown_case_ids"), "summary unknown cases")
        if any(alias not in aliases_by_dataset[dataset] for alias in unknown_ids):
            raise ValueError("unknown case list belongs to another domain")
        n_valid = _int(row.get("n_valid"), "summary.n_valid")
        n_unknown = _int(row.get("n_unknown"), "summary.n_unknown")
        if n_valid + n_unknown != 4 or len(unknown_ids) != n_unknown:
            raise ValueError("summary valid/unknown counts do not cover four cases")
        fields = {name: _num(row.get(name), f"summary.{name}", minimum=0.0, maximum=1.0)
                  for name in ("mean", "median", "min", "max")}
        if n_valid == 0 and any(value is not None for value in fields.values()):
            raise ValueError("all-unknown summary must preserve null statistics")
        arm_out.append({"dataset": DATASETS[dataset], "arm": arm, "metric": metric,
                        "n_valid": n_valid, "n_unknown": n_unknown,
                        "unknown_case_ids": unknown_ids, **fields})
    if seen != expected_cells:
        raise ValueError("summary arm-metric matrix omits fixed cells")

    contrast_out, seen_contrasts = [], set()
    contrast_rows = source.get("contrasts")
    if not isinstance(contrast_rows, list) or len(contrast_rows) != 2 * 8 * len(METRICS):
        raise ValueError("summary contrast catalog has an unexpected size")
    expected_contrasts = {"T_minus_F": ("T", "F"),
                          "native_teacher_direct_minus_T": ("native_teacher_direct", "T"),
                          "T_direct_minus_T": ("T_direct", "T"),
                          "T_iou_minus_T": ("T_iou", "T"),
                          "T_reliability_weighted_iou_minus_T": ("T_reliability_weighted_iou", "T"),
                          "direct_minus_F": ("direct", "F"), "iou_minus_F": ("iou", "F"),
                          "reliability_weighted_iou_minus_F": ("reliability_weighted_iou", "F")}
    for row in contrast_rows:
        if not isinstance(row, dict):
            raise ValueError("paired contrast is malformed")
        dataset, name, metric = row.get("dataset"), row.get("contrast"), row.get("metric")
        if dataset not in DATASETS or name not in expected_contrasts or metric not in METRICS:
            raise ValueError("contrast has an unexpected domain, label, or metric")
        key = (dataset, name, metric)
        if key in seen_contrasts:
            raise ValueError("duplicate paired contrast")
        seen_contrasts.add(key)
        comp, base = expected_contrasts[name]
        if row.get("comparison_arm") != comp or row.get("baseline_arm") != base:
            raise ValueError("contrast direction differs from the fixed comparison")
        unknown_ids = _alias_ids(row.get("unknown_case_ids"), "contrast unknown cases")
        positive_ids = _alias_ids(row.get("positive_case_ids"), "contrast positive cases")
        negative_ids = _alias_ids(row.get("negative_case_ids"), "contrast negative cases")
        neutral_ids = _alias_ids(row.get("neutral_case_ids"), "contrast neutral cases")
        for group in (unknown_ids, positive_ids, negative_ids, neutral_ids):
            if any(alias not in aliases_by_dataset[dataset] for alias in group):
                raise ValueError("contrast case alias belongs to another domain")
        if (set(unknown_ids) & (set(positive_ids) | set(negative_ids) | set(neutral_ids))
                or set(positive_ids) & set(negative_ids) or set(positive_ids) & set(neutral_ids)
                or set(negative_ids) & set(neutral_ids)):
            raise ValueError("contrast case classifications overlap")
        counts = {"n_pairs": _int(row.get("n_pairs"), "contrast.n_pairs"),
                  "unknown_count": _int(row.get("unknown_count"), "contrast.unknown_count"),
                  "positive_count": _int(row.get("positive_count"), "contrast.positive_count"),
                  "negative_count": _int(row.get("negative_count"), "contrast.negative_count"),
                  "neutral_count": _int(row.get("neutral_count"), "contrast.neutral_count"),
                  "delta_le_minus_5pp_count": _int(row.get("delta_le_minus_5pp_count"), "contrast.delta_le_minus_5pp_count"),
                  "delta_ge_plus_5pp_count": _int(row.get("delta_ge_plus_5pp_count"), "contrast.delta_ge_plus_5pp_count")}
        if (counts["unknown_count"] != len(unknown_ids)
                or counts["positive_count"] != len(positive_ids)
                or counts["negative_count"] != len(negative_ids)
                or counts["neutral_count"] != len(neutral_ids)
                or counts["n_pairs"] != len(positive_ids) + len(negative_ids) + len(neutral_ids)):
            raise ValueError("contrast counts disagree with the listed anonymous cases")
        ci = row.get("bootstrap_ci95_delta_pp")
        if ci is not None:
            _exact_keys(ci, {"lower", "upper"}, "bootstrap interval")
            ci = {k: _num(v, f"bootstrap.{k}") for k, v in ci.items()}
        contrast_out.append({"dataset": DATASETS[dataset], "contrast": name,
                             "comparison_arm": comp, "baseline_arm": base,
                             "metric": metric, **counts,
                             "unknown_case_ids": unknown_ids,
                             "positive_case_ids": positive_ids,
                             "negative_case_ids": negative_ids,
                             "neutral_case_ids": neutral_ids,
                             "neutral_tolerance": _num(row.get("neutral_tolerance"),
                                                       "contrast.neutral_tolerance", minimum=0),
                             "mean_delta": _num(row.get("mean_delta"), "contrast.mean_delta"),
                             "mean_delta_pp": _num(row.get("mean_delta_pp"), "contrast.mean_delta_pp"),
                             "median_delta_pp": _num(row.get("median_delta_pp"), "contrast.median_delta_pp"),
                             "min_delta_pp": _num(row.get("min_delta_pp"), "contrast.min_delta_pp"),
                             "max_delta_pp": _num(row.get("max_delta_pp"), "contrast.max_delta_pp"),
                             "bootstrap_ci95_delta_pp": ci})
    if len(seen_contrasts) != len(expected_contrasts) * len(METRICS) * 2:
        raise ValueError("paired contrast grid is incomplete")

    teacher_summary = source.get("teacher_selection_and_coverage")
    if not isinstance(teacher_summary, list) or len(teacher_summary) != 2:
        raise ValueError("teacher selection summary must contain both domains")
    teacher_out = []
    for row in teacher_summary:
        dataset = row.get("dataset")
        if dataset not in DATASETS:
            raise ValueError("teacher summary domain is unexpected")
        status_counts = row.get("selection_status_counts")
        if not isinstance(status_counts, dict) or not set(status_counts) <= SELECTION_STATUS:
            raise ValueError("teacher selection status summary is not a closed vocabulary")
        status_counts = {key: _int(value, f"teacher.status_counts.{key}")
                         for key, value in status_counts.items()}
        teacher_out.append({"dataset": DATASETS[dataset],
                            "case_count": _int(row.get("case_count"), "teacher.case_count"),
                            "valid_uncapped_selection_count": _int(
                                row.get("valid_uncapped_selection_count"), "teacher.valid_count"),
                            "selection_rate": _num(row.get("selection_rate"), "teacher.selection_rate",
                                                    minimum=0, maximum=1),
                            "selection_status_counts": status_counts,
                            "mean_unlabeled_present_fraction_selected": _num(
                                row.get("mean_unlabeled_present_fraction_selected"),
                                "teacher.present_fraction", minimum=0, maximum=1),
                            "mean_present_runs_selected": _num(
                                row.get("mean_present_runs_selected"), "teacher.present_runs", minimum=0),
                            "unknown_or_abstained_count": _int(
                                row.get("unknown_or_abstained_count"), "teacher.unknown_count")})
    if {r["dataset"] for r in teacher_out} != set(DATASETS.values()):
        raise ValueError("teacher selection summary does not cover both domains")
    projected = {"schema": "stvg-native-instance-public-postseal-summary-v1",
            "status": "pass", "metric_scale": "0_to_1", "GT_read": True,
            "candidate_detection_recall": "not_measured",
            "scope": "exploratory descriptive paired results; no pooled gate or selection",
            "aggregation": {"bootstrap_replicates": 10000, "seed": 20261014,
                            "unit": "paired source case", "scope": "separately within each domain",
                            "historically_exposed_sources_per_domain": 4,
                            "ci_interpretation": "exploratory descriptive only",
                            "pooled_cross_domain_gate": False,
                            "ci_used_for_arm_or_step_selection": False},
            "case_count_by_dataset": {DATASETS[k]: 4 for k in DATASETS},
            "arm_metric_summaries": sorted(arm_out, key=lambda r: (
                r["dataset"], ARMS.index(r["arm"]), METRICS.index(r["metric"]))),
            "contrasts": sorted(contrast_out, key=lambda r: (
                r["dataset"], r["contrast"], METRICS.index(r["metric"]))),
            "teacher_selection_and_coverage": sorted(teacher_out, key=lambda r: r["dataset"])}
    _validate_summary_against_rows(projected, rows_public)
    return projected


def _validate_summary_against_rows(summary: Mapping[str, Any], rows: Mapping[str, Any]) -> None:
    """Recompute public group statistics and direction counts from source rows."""
    by_key = {(row["case_id"], row["arm"]): row for row in rows["rows"]}
    for item in summary["arm_metric_summaries"]:
        domain = item["dataset"]
        ids = sorted(alias for alias, arm in by_key
                     if arm == "F" and by_key[(alias, arm)]["dataset"] == domain)
        values = [by_key[(alias, item["arm"])]["metrics"][item["metric"]] for alias in ids]
        known = [float(value) for value in values if value is not None]
        unknown = sorted(ids[i] for i, value in enumerate(values) if value is None)
        if (item["n_valid"] != len(known) or item["n_unknown"] != len(unknown)
                or item["unknown_case_ids"] != unknown):
            raise ValueError("arm summary valid/unknown state differs from its source rows")
        ordered = sorted(known)
        expected = {
            "mean": sum(known) / len(known) if known else None,
            "median": ((ordered[len(ordered) // 2] if len(ordered) % 2 else
                        (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2)
                       if known else None),
            "min": min(known) if known else None,
            "max": max(known) if known else None,
        }
        for field, value in expected.items():
            actual = item[field]
            if value is None:
                if actual is not None:
                    raise ValueError("all-unknown arm summary must preserve null statistics")
            elif actual is None or not math.isclose(float(actual), value, rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"arm summary {field} differs from source-level scores")

    pairs = {"T_minus_F": ("T", "F"),
             "native_teacher_direct_minus_T": ("native_teacher_direct", "T"),
             "T_direct_minus_T": ("T_direct", "T"), "T_iou_minus_T": ("T_iou", "T"),
             "T_reliability_weighted_iou_minus_T": ("T_reliability_weighted_iou", "T"),
             "direct_minus_F": ("direct", "F"), "iou_minus_F": ("iou", "F"),
             "reliability_weighted_iou_minus_F": ("reliability_weighted_iou", "F")}
    for item in summary["contrasts"]:
        left, right = pairs[item["contrast"]]
        ids = sorted(alias for alias, arm in by_key
                     if arm == "F" and by_key[(alias, arm)]["dataset"] == item["dataset"])
        values, unknown = [], []
        for alias in ids:
            a = by_key[(alias, left)]["metrics"][item["metric"]]
            b = by_key[(alias, right)]["metrics"][item["metric"]]
            if a is None or b is None:
                unknown.append(alias)
            else:
                values.append((alias, float(a) - float(b)))
        tolerance = item["neutral_tolerance"]
        positive = sorted(alias for alias, delta in values if delta > tolerance)
        negative = sorted(alias for alias, delta in values if delta < -tolerance)
        neutral = sorted(alias for alias, delta in values if abs(delta) <= tolerance)
        if (item["unknown_case_ids"] != unknown or item["positive_case_ids"] != positive
                or item["negative_case_ids"] != negative or item["neutral_case_ids"] != neutral):
            raise ValueError("paired contrast classifications differ from source rows")
        deltas = [value for _, value in values]
        ordered = sorted(deltas)
        expected = {
            "n_pairs": len(deltas), "unknown_count": len(unknown),
            "positive_count": len(positive), "negative_count": len(negative),
            "neutral_count": len(neutral),
            "mean_delta": sum(deltas) / len(deltas) if deltas else None,
            "mean_delta_pp": 100 * sum(deltas) / len(deltas) if deltas else None,
            "median_delta_pp": (100 * (ordered[len(ordered) // 2] if len(ordered) % 2 else
                                      (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2)
                                if deltas else None),
            "min_delta_pp": 100 * min(deltas) if deltas else None,
            "max_delta_pp": 100 * max(deltas) if deltas else None,
            "delta_le_minus_5pp_count": sum(delta <= -0.05 for delta in deltas),
            "delta_ge_plus_5pp_count": sum(delta >= 0.05 for delta in deltas),
        }
        for field, value in expected.items():
            actual = item[field]
            if isinstance(value, float):
                if actual is None or not math.isclose(float(actual), value, rel_tol=0, abs_tol=1e-9):
                    raise ValueError(f"paired contrast {field} differs from source rows")
            elif actual != value:
                raise ValueError(f"paired contrast {field} differs from source rows")


def _project_math(source: Mapping[str, Any], barrier_sha: str,
                  row_bindings: Mapping[str, str]) -> dict[str, Any]:
    if (source.get("schema") != "stvg-native-instance-independent-student-math-readback-v1"
            or source.get("status") != "pass" or source.get("GT_read") is not False
            or source.get("CPU_only") is not True or source.get("GPU_initialized") is not False
            or source.get("model_replayed_by_reader") is not False or source.get("case_count") != 8
            or source.get("student_barrier_sha256") != row_bindings["student_barrier_sha256"]):
        raise ValueError("student math readback is incomplete or mismatched")
    if source.get("reference_lock_sha256") != row_bindings["reference_lock_sha256"]:
        raise ValueError("math/reference lock digest differs from score rows")
    cases_src = source.get("case_checks")
    if not isinstance(cases_src, dict) or set(cases_src) != set(CASE_IDS):
        raise ValueError("math readback does not cover the fixed eight sources")
    cases = {}
    fit_total = steps_total = readout_total = 0
    for cid, row in cases_src.items():
        if not isinstance(row, dict) or row.get("dataset") != ("hc2" if cid.startswith("hc2_") else "vidstg"):
            raise ValueError("math case check has a bad domain binding")
        count = _int(row.get("fit_count"), "math.fit_count")
        steps = _int(row.get("optimizer_steps"), "math.optimizer_steps")
        readouts = _int(row.get("saved_full_model_readout_count"), "math.readout_count")
        temporal_readouts = _int(row.get("saved_temporal_logits_preserved_count"),
                                 "math.temporal_logits_preserved_count")
        if temporal_readouts != readouts:
            raise ValueError("saved temporal-logit readouts do not match full-model readouts")
        fits = row.get("fits")
        if not isinstance(fits, dict) or not set(fits) <= set(SPATIAL):
            raise ValueError("math fit checks include an unknown arm")
        fit_public = {}
        for algorithm, check in fits.items():
            if not isinstance(check, dict) or check.get("steps") != 5:
                raise ValueError("math fit check does not retain five declared rounds")
            base_fit = {"steps": 5,
                        "optimizer_steps": _int(check.get("optimizer_steps"),
                                                 "math.fit.optimizer_steps")}
            if algorithm == "direct":
                direct_fields = ("direct_loss_independently_recomputed",
                                 "adam_moment_and_parameter_update_reconstructed",
                                 "decoder_autograd_gradient_independently_recomputed",
                                 "saved_full_model_readout_equals_cached_decoder",
                                 "model_replayed_by_reader")
                for field in direct_fields:
                    base_fit[field] = _bool(check.get(field), f"math.fit.{field}")
                for field in ("max_direct_loss_abs_error",
                              "max_direct_adam_parameter_or_state_error",
                              "max_direct_adam_second_moment_consistency_error"):
                    base_fit[field] = _num(check.get(field), f"math.fit.{field}", minimum=0)
            else:
                errors = check.get("max_abs_errors")
                if not isinstance(errors, dict):
                    raise ValueError("math fit error summary is missing")
                error_keys = {"mean_abs", "sample_abs", "reward_abs", "weight_abs",
                              "loss_abs", "adam_parameter_abs", "adam_state_abs"}
                if set(errors) != error_keys:
                    raise ValueError("OPD fit error matrix differs from its frozen schema")
                base_fit["max_abs_errors"] = {k: _num(errors[k], f"math.fit.error.{k}", minimum=0)
                                              for k in sorted(errors)}
                semantics = check.get("reward_semantics")
                if semantics not in {"original_iou", "raw_iou_times_r_t_before_fixed_tau_softmax"}:
                    raise ValueError("OPD fit reward semantics are unexpected")
                base_fit["reward_semantics"] = semantics
                base_fit["cross_anchor_reliability_normalization"] = _bool(
                    check.get("cross_anchor_reliability_normalization"),
                    "math.fit.cross_anchor_reliability_normalization")
            fit_public[algorithm] = base_fit
        if count != len(fits):
            raise ValueError("math fit count differs from per-arm math checks")
        cases[_alias(cid)] = {"dataset": DATASETS[row["dataset"]], "fit_count": count,
                              "optimizer_steps": steps, "saved_full_model_readout_count": readouts,
                              "saved_temporal_logits_preserved_count": temporal_readouts,
                              "source_state_hash_consistent": _bool(
                                  row.get("source_state_hash_consistent"), "math.source_state_hash_consistent"),
                              "saved_source_cold_receipt_self_consistent": _bool(
                                  row.get("saved_source_cold_receipt_self_consistent"),
                                  "math.saved_source_cold_receipt_self_consistent"),
                              "model_replayed_by_reader": _bool(
                                  row.get("model_replayed_by_reader"), "math.case.model_replayed_by_reader"),
                              "fits": fit_public}
        fit_total += count
        steps_total += steps
        readout_total += readouts
    if fit_total != source.get("actual_fit_count") or steps_total != source.get("saved_optimizer_steps") or readout_total != source.get("saved_full_model_readouts"):
        raise ValueError("math case totals differ from aggregate totals")
    hashes = {k: _sha_value(source.get(k), f"math.{k}")
              for k in ("runtime_sha256", "student_barrier_sha256", "reference_lock_sha256", "roster_sha256")}
    return {"schema": source["schema"], "status": "pass", "case_count": 8,
            "actual_fit_count": fit_total, "saved_optimizer_steps": steps_total,
            "saved_full_model_readouts": readout_total,
            "saved_temporal_logits_preserved": readout_total,
            "source_capture_calls": _int(source.get("source_capture_calls"), "math.source_capture_calls"),
            "GT_read": False, "GPU_initialized": False, "CPU_only": True,
            "model_replayed_by_reader": False, **hashes,
            "cases": {key: cases[key] for key in sorted(cases)}}


def _project_costs(source: Mapping[str, Any], rows: Mapping[str, Any], math_public: Mapping[str, Any],
                   barrier_sha: str) -> dict[str, Any]:
    if (source.get("schema") != "stvg-native-instance-cost-readback-v1"
            or source.get("status") != "complete" or source.get("GT_read") is not False
            or source.get("count") != 8
            or source.get("student_barrier_sha256") != math_public["student_barrier_sha256"]
            or source.get("runtime_sha256") != math_public["runtime_sha256"]):
        raise ValueError("student cost readback is incomplete or mismatched")
    by_alias = {r["case_id"]: r for r in rows["rows"] if r["arm"] == "T"}
    records = source.get("records")
    if not isinstance(records, list) or len(records) != 8:
        raise ValueError("cost readback must have exactly eight per-source records")
    costs = []
    seen = set()
    for row in records:
        case_id, dataset = row.get("case_id"), row.get("dataset")
        alias = _alias(case_id, dataset)
        if alias in seen:
            raise ValueError("duplicate per-source cost record")
        seen.add(alias)
        baseline = by_alias[alias]
        if (row.get("source_case_sha256") != baseline["bindings"]["source_case_sha256"]
                or row.get("student_artifact_sha256") != baseline["bindings"]["student_artifact_sha256"]):
            raise ValueError("cost source/student binding differs from the scored arm")
        capture = _num(row.get("source_capture_seconds"), "cost.source_capture_seconds", minimum=0)
        algorithms = row.get("algorithm_costs")
        if not isinstance(algorithms, dict) or set(algorithms) != set(SPATIAL):
            raise ValueError("cost algorithm catalog differs from the three spatial arms")
        alg_out = {}
        for name in SPATIAL:
            item = algorithms[name]
            if not isinstance(item, dict):
                raise ValueError("algorithm cost record is malformed")
            seconds = _num(item.get("seconds"), f"cost.{name}.seconds", minimum=0)
            steps = _int(item.get("optimizer_steps"), f"cost.{name}.optimizer_steps")
            redecode = _int(item.get("final_redecode_calls"), f"cost.{name}.final_redecode_calls")
            status = item.get("status")
            fit_present = _bool(item.get("fit_present"), f"cost.{name}.fit_present")
            if status not in {"exact_no_update", "actual_updated_full_redecode"}:
                raise ValueError("algorithm cost status is unexpected")
            if status == "exact_no_update" and (fit_present or seconds is not None or steps or redecode):
                raise ValueError("exact no-update cost must preserve null elapsed time and zero calls")
            alg_out[name] = {"seconds": seconds, "optimizer_steps": steps,
                             "final_redecode_calls": redecode, "status": status,
                             "fit_present": fit_present,
                             "elapsed_seconds_available": seconds is not None}
        costs.append({"case_id": alias, "dataset": DATASETS[dataset],
                      "source_case_sha256": _sha_value(row.get("source_case_sha256"), "cost.source_case_sha256"),
                      "student_artifact_sha256": _sha_value(row.get("student_artifact_sha256"), "cost.student_artifact_sha256"),
                      "source_capture_seconds": capture,
                      "source_capture_seconds_available": capture is not None,
                      "algorithm_costs": alg_out})
    if seen != set(CASE_ALIAS.values()):
        raise ValueError("cost readback omits a fixed source")
    return {"schema": "stvg-native-instance-public-costs-v1", "status": "pass",
            "GT_read": False, "case_count": 8,
            "student_source_capture_seconds_are_measured": all(
                row["source_capture_seconds_available"] for row in costs),
            "algorithm_no_update_seconds_remain_null": True,
            "records": sorted(costs, key=lambda r: r["case_id"])}


def _project_generation(source: Mapping[str, Any]) -> dict[str, Any]:
    if (source.get("status") != "pass" or source.get("GT_read") is not False
            or source.get("CPU_only") is not True):
        raise ValueError("generation configuration readback is incomplete")
    config = source.get("generation_configuration")
    overrides = source.get("explicit_generate_overrides")
    if not isinstance(config, dict) or not isinstance(overrides, dict):
        raise ValueError("generation configuration is malformed")
    expected = {"attn_implementation": "flash_attention_2", "do_sample": True,
                "top_k": 1, "top_p": 0.001}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"official generation configuration differs: {key}")
    temperature = config.get("temperature", config.get("temp"))
    if temperature != 0.1:
        raise ValueError("official generation temperature differs from the frozen configuration")
    expected["temperature"] = 0.1
    if overrides.get("max_new_tokens") != 2048:
        raise ValueError("generation token cap differs from the locked teacher configuration")
    backend = source.get("attention_backend_actual")
    if backend != "sdpa" or source.get("teacher_manual_seed_set") is not False:
        raise ValueError("actual R1 backend/manual-seed disclosure is inconsistent")
    return {"schema": source.get("schema"), "status": "pass", "GT_read": False,
            "CPU_only": True,
            "official_generation_configuration_sha256": _sha_value(
                source.get("official_generation_configuration_sha256"), "generation config SHA"),
            "generation_configuration": expected,
            "explicit_generate_overrides": {"max_new_tokens": 2048},
            "attention_backend_actual": "sdpa",
            "teacher_manual_seed_set": False,
            "interpretation": "R1 generation used the recorded SDPA backend; this is distinct from the detector QKV compatibility path."}


def _project_root_postseal(source: Mapping[str, Any], rows_path: Path,
                           summary_path: Path, barrier_sha: str) -> dict[str, Any]:
    reader = ROOT / "scripts/review_stvg_native_instance_postseal_v1.py"
    required = {"schema", "status", "GT_read", "raw_GT_read_by_this_reader", "CPU_only",
                "prediction_barrier_sha256", "rows_sha256", "summary_sha256",
                "completion_sha256", "case_count", "row_count", "arm_summary_count",
                "contrast_count", "checks_passed", "checks_failed",
                "max_metric_scalar_abs_error", "all_case_source_bindings_match",
                "all_tails_and_unknown_membership_match", "model_replayed",
                "verification_scope", "reader_sha256", "time"}
    _exact_keys(source, required, "root postseal readback")
    completion_path = BASE / "CPU_COMPLETION.json"
    if (source["schema"] != "stvg-native-instance-root-postseal-readback-v1"
            or source["status"] != "pass" or source["GT_read"] is not True
            or source["raw_GT_read_by_this_reader"] is not False
            or source["CPU_only"] is not True or source["model_replayed"] is not False
            or source["prediction_barrier_sha256"] != barrier_sha
            or source["rows_sha256"] != _sha(rows_path)
            or source["summary_sha256"] != _sha(summary_path)
            or source["completion_sha256"] != _sha(completion_path)
            or source["reader_sha256"] != _sha(reader)
            or _int(source["case_count"], "root_postseal.case_count") != 8
            or _int(source["row_count"], "root_postseal.row_count") != 72
            or _int(source["arm_summary_count"], "root_postseal.arm_summary_count") != 90
            or _int(source["contrast_count"], "root_postseal.contrast_count") != 80
            or _int(source["checks_failed"], "root_postseal.checks_failed") != 0
            or source["all_case_source_bindings_match"] is not True
            or source["all_tails_and_unknown_membership_match"] is not True
            or float(_num(source["max_metric_scalar_abs_error"],
                          "root_postseal.max_metric_scalar_abs_error")) > 2e-8):
        raise ValueError("root postseal readback does not pass all exact bindings and scalar checks")
    _num(source["time"], "root_postseal.time", minimum=0)
    return {"schema": source["schema"], "status": "pass", "GT_read": True,
            "raw_GT_read_by_this_reader": False, "CPU_only": True, "model_replayed": False,
            "prediction_barrier_sha256": barrier_sha, "rows_sha256": _sha(rows_path),
            "summary_sha256": _sha(summary_path), "completion_sha256": _sha(completion_path),
            "case_count": 8, "row_count": 72, "arm_summary_count": 90,
            "contrast_count": 80,
            "checks_passed": _int(source["checks_passed"], "root_postseal.checks_passed", minimum=1),
            "checks_failed": 0,
            "max_metric_scalar_abs_error": float(source["max_metric_scalar_abs_error"]),
            "all_case_source_bindings_match": True,
            "all_tails_and_unknown_membership_match": True,
            "verification_scope": "independent scalar aggregation, exact bindings, support partitions, and unknown-tail membership; geometric metrics were checked by the sealed scorer",
            "reader_sha256": _sha(reader)}


def _project_visual(source: Mapping[str, Any], report: Path,
                    pdf: Path, png: Path) -> dict[str, Any]:
    required = {"schema", "status", "report_sha256", "pdf_sha256", "png_sha256",
                "case_count", "private_contact_count", "all_private_contacts_reviewed",
                "pdf_pages_reviewed", "all_pdf_pages_reviewed", "raw_media_public"}
    _exact_keys(source, required, "visual review receipt")
    if (source["status"] != "pass" or _int(source["case_count"], "visual.case_count") != 8
            or _int(source["private_contact_count"], "visual.private_contact_count") != 8
            or source["all_private_contacts_reviewed"] is not True
            or _int(source["pdf_pages_reviewed"], "visual.pdf_pages_reviewed") < 1
            or source["all_pdf_pages_reviewed"] is not True
            or source["raw_media_public"] is not False
            or source["report_sha256"] != _sha(report)
            or source["pdf_sha256"] != _sha(pdf)
            or source["png_sha256"] != _sha(png)):
        raise ValueError("visual review does not pass and bind the current anonymous outputs")
    return {"schema": source["schema"], "status": "pass", "case_count": 8,
            "private_contact_count": 8, "all_private_contacts_reviewed": True,
            "pdf_pages_reviewed": _int(source["pdf_pages_reviewed"], "visual.pages"),
            "all_pdf_pages_reviewed": True, "raw_media_public": False,
            "report_sha256": _sha(report), "pdf_sha256": _sha(pdf), "png_sha256": _sha(png)}


def _project_geometry_readback(source: Mapping[str, Any]) -> dict[str, Any]:
    reader = ROOT / "scripts/review_stvg_native_instance_geometry_v1.py"
    if (source.get("status") != "pass"
            or source.get("scope") != "all selected saved RLE geometry and fixed local reliability; no semantic identity claim"
            or source.get("case_count") != 8
            or source.get("checks") != 31842 or source.get("GT_read") is not False
            or source.get("CPU_only") is not True
            or source.get("model_replayed") is not False
            or source.get("reference_lock_sha256") != _sha(BASE / "REFERENCE_LOCK.json")
            or source.get("direct_readout_sha256") != _sha(BASE / "DIRECT_TEACHER_READOUT.json")
            or source.get("reader_sha256") != _sha(reader)):
        raise ValueError("independent saved-geometry readback is incomplete or not source-bound")
    return {"schema": "stvg-native-instance-independent-geometry-readback-public-v1",
            "status": "pass", "scope": "all selected saved RLE geometry and fixed local reliability; no semantic identity claim",
            "case_count": 8, "checks": 31842, "GT_read": False,
            "CPU_only": True, "model_replayed": False,
            "reference_lock_sha256": _sha(BASE / "REFERENCE_LOCK.json"),
            "direct_readout_sha256": _sha(BASE / "DIRECT_TEACHER_READOUT.json"),
            "reader_sha256": _sha(reader), "per_case_records_published": False}


def _close(actual: Any, expected: float, label: str, *, tolerance: float = 2e-12) -> float:
    number = _num(actual, label)
    if number is None or not math.isclose(number, expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"{label} differs from its anonymous source aggregation")
    return number


def _project_pool_coverage(source: Mapping[str, Any], rows_path: Path,
                           appendix_path: Path, barrier_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    tool = ROOT / POOL_COVERAGE_TOOL_PATH
    scorer = ROOT / "scripts/score_stvg_native_instance_postseal_v1.py"
    truth_wrapper = ROOT / "scripts/review_stvg_sparse_r1_v2.py"
    truth_sources = {
        "review_stvg_sparse_r1_v2.py": ROOT / "scripts/review_stvg_sparse_r1_v2.py",
        "score_stvg_opd_p1_v1.py": ROOT / "scripts/score_stvg_opd_p1_v1.py",
        "diagnose_tastvg_pipeline_cpu_v1.py": ROOT / "scripts/diagnose_tastvg_pipeline_cpu_v1.py",
    }
    mask_source = ROOT / "external/stvg_runtime/pycocotools/mask.py"
    mask_extension = ROOT / "external/stvg_runtime/pycocotools/_mask.cpython-311-x86_64-linux-gnu.so"
    if (source.get("schema") != "stvg-native-instance-filtered-pool-coverage-v1"
            or source.get("status") != "completed_cpu_postseal"
            or source.get("GT_read") is not True
            or source.get("GT_read_after_complete_prediction_seal") is not True
            or source.get("thresholds") != [0.25, 0.5]):
        raise ValueError("filtered-pool coverage is incomplete or outside the fixed post-seal scope")
    fixed_text = {
        "coverage_estimand": "per annotated physical frame, maximum bbox IoU over all nonempty filtered tracker masks",
        "denominator_definition": "all physical frames returned by the official sealed-source annotation reader; frames outside the canonical tracker clock count as zero",
        "candidate_policy": "complete all-category filtered tracking pool; no category metadata filtering; no GT-based candidate selection",
    }
    for key, expected in fixed_text.items():
        if source.get(key) != expected:
            raise ValueError(f"coverage {key} differs from the frozen evaluation definition")
    claim_boundaries = {
        "raw_yolo_detector_recall": "not_measured",
        "semantic_identity_correctness": "not_measured",
        "oracle_fit_or_reference_change": False,
        "prediction_selection_or_change": False,
    }
    if source.get("claim_boundaries") != claim_boundaries:
        raise ValueError("coverage claim boundaries are not the locked no-selection labels")
    redactions = ("private_case_ids_exported", "source_paths_exported",
                  "queries_or_teacher_text_exported", "boxes_or_raw_annotation_labels_exported",
                  "images_or_masks_exported")
    if any(source.get(key) is not False for key in redactions):
        raise ValueError("coverage receipt declares a private artifact was exported")

    source_bindings = source.get("bindings")
    binding_keys = {"global_prediction_barrier_sha256", "roster_lock_sha256",
                    "tracking_global_barrier_sha256", "postseal_scorer_sha256",
                    "official_truth_reader_sha256", "official_gt_reader_source_sha256s",
                    "coverage_tool_sha256", "filtered_rle_decoder_source_sha256",
                    "filtered_rle_decoder_extension_sha256", "official_gt_provenance_sha256"}
    _exact_keys(source_bindings, binding_keys, "pool-coverage bindings")
    simple_paths = {
        "global_prediction_barrier_sha256": BASE / PREDICTION_BARRIER_NAME,
        "roster_lock_sha256": BASE / "ROSTER_LOCK.json",
        "tracking_global_barrier_sha256": BASE / "TRACKING_GLOBAL_BARRIER.json",
        "postseal_scorer_sha256": scorer,
        "official_truth_reader_sha256": truth_wrapper,
        "coverage_tool_sha256": tool,
        "filtered_rle_decoder_source_sha256": mask_source,
        "filtered_rle_decoder_extension_sha256": mask_extension,
    }
    checked_bindings: dict[str, Any] = {}
    for key, path in simple_paths.items():
        digest = _sha_value(source_bindings.get(key), f"pool binding.{key}")
        if digest != _sha(path):
            raise ValueError(f"pool coverage binding differs from current locked code/input: {key}")
        checked_bindings[key] = digest
    reader_hashes = source_bindings.get("official_gt_reader_source_sha256s")
    _exact_keys(reader_hashes, set(truth_sources), "official truth-reader source hashes")
    checked_reader_hashes = {}
    for name, path in truth_sources.items():
        digest = _sha_value(reader_hashes.get(name), f"pool truth source.{name}")
        if digest != _sha(path):
            raise ValueError(f"official truth-reader source changed: {name}")
        checked_reader_hashes[name] = digest
    checked_bindings["official_gt_reader_source_sha256s"] = checked_reader_hashes
    checked_bindings["official_gt_provenance_sha256"] = _sha_value(
        source_bindings.get("official_gt_provenance_sha256"), "pool official GT provenance SHA")

    row_fields = {"alias", "dataset", "known_gt_frames", "gt_frames_in_canonical_map",
                  "gt_frames_outside_canonical_map", "canonical_gt_frames_with_empty_filtered_pool",
                  "gt_frames_with_empty_filtered_pool", "sum_max_bbox_iou", "mean_max_bbox_iou",
                  "gt_frames_max_iou_ge_0_25", "fraction_gt_frames_max_iou_ge_0_25",
                  "gt_frames_max_iou_ge_0_50", "fraction_gt_frames_max_iou_ge_0_50",
                  "filtered_candidate_track_count", "nonempty_filtered_instance_masks"}
    case_rows = source.get("cases")
    if not isinstance(case_rows, list) or len(case_rows) != 8:
        raise ValueError("pool-coverage source rows do not contain all eight anonymous cases")
    rows_out = []
    seen_aliases = set()
    for row in case_rows:
        _exact_keys(row, row_fields, "pool-coverage case row")
        alias, dataset = row.get("alias"), row.get("dataset")
        if alias not in CASE_ALIAS.values() or dataset not in DATASETS:
            raise ValueError("pool-coverage row has an unknown alias or domain")
        if alias in seen_aliases:
            raise ValueError("pool-coverage aliases are duplicated or malformed")
        seen_aliases.add(alias)
        expected_domain = "HC" if dataset == "hc2" else "V"
        if not alias.startswith(expected_domain):
            raise ValueError("pool-coverage alias does not match its dataset")
        counts = {key: _int(row[key], f"pool.{alias}.{key}") for key in (
            "known_gt_frames", "gt_frames_in_canonical_map", "gt_frames_outside_canonical_map",
            "canonical_gt_frames_with_empty_filtered_pool", "gt_frames_with_empty_filtered_pool",
            "gt_frames_max_iou_ge_0_25", "gt_frames_max_iou_ge_0_50",
            "filtered_candidate_track_count", "nonempty_filtered_instance_masks")}
        known = counts["known_gt_frames"]
        if known < 1 or counts["gt_frames_in_canonical_map"] + counts["gt_frames_outside_canonical_map"] != known:
            raise ValueError("pool-coverage denominator partition is invalid")
        if (not (counts["canonical_gt_frames_with_empty_filtered_pool"] <=
                 counts["gt_frames_with_empty_filtered_pool"] <= known)
                or counts["gt_frames_with_empty_filtered_pool"] !=
                counts["canonical_gt_frames_with_empty_filtered_pool"]
                + counts["gt_frames_outside_canonical_map"]):
            raise ValueError("pool-coverage empty-pool counts are inconsistent")
        if (counts["gt_frames_max_iou_ge_0_25"] > known
                or counts["gt_frames_max_iou_ge_0_50"] > counts["gt_frames_max_iou_ge_0_25"]):
            raise ValueError("pool-coverage threshold count exceeds the denominator")
        total = _num(row["sum_max_bbox_iou"], f"pool.{alias}.sum_max_bbox_iou", minimum=0)
        mean = _num(row["mean_max_bbox_iou"], f"pool.{alias}.mean_max_bbox_iou", minimum=0, maximum=1)
        if total is None or total > known + 1e-8:
            raise ValueError("pool-coverage IoU sum exceeds its frame denominator")
        _close(mean, total / known, f"pool.{alias}.mean_max_bbox_iou")
        fractions = {}
        for threshold, key in (("0_25", "fraction_gt_frames_max_iou_ge_0_25"),
                               ("0_50", "fraction_gt_frames_max_iou_ge_0_50")):
            frac = _num(row[key], f"pool.{alias}.{key}", minimum=0, maximum=1)
            expected = counts[f"gt_frames_max_iou_ge_{threshold}"] / known
            _close(frac, expected, f"pool.{alias}.{key}")
            fractions[key] = frac
        rows_out.append({"case_id": alias, "dataset": DATASETS[dataset], **counts,
                         "sum_max_bbox_iou": float(total), "mean_max_bbox_iou": float(mean),
                         **fractions})
    if seen_aliases != set(CASE_ALIAS.values()):
        raise ValueError("pool-coverage case alias set differs from the fixed eight-source cohort")
    rows_out.sort(key=lambda row: (row["dataset"], row["case_id"]))

    agg_fields = {"case_count", "known_gt_frames", "gt_frames_in_canonical_map",
                  "gt_frames_outside_canonical_map", "canonical_gt_frames_with_empty_filtered_pool",
                  "gt_frames_with_empty_filtered_pool", "frame_weighted_mean_max_bbox_iou",
                  "case_macro_mean_max_bbox_iou", "gt_frames_max_iou_ge_0_25",
                  "fraction_gt_frames_max_iou_ge_0_25", "gt_frames_max_iou_ge_0_50",
                  "fraction_gt_frames_max_iou_ge_0_50"}
    summary_src = source.get("summary")
    if not isinstance(summary_src, dict):
        raise ValueError("pool-coverage aggregate summary is missing")
    _exact_keys(summary_src, {"all", "by_dataset"}, "pool-coverage summary")
    groups = {"all": rows_out,
              "hc2": [row for row in rows_out if row["dataset"] == DATASETS["hc2"]],
              "vidstg": [row for row in rows_out if row["dataset"] == DATASETS["vidstg"]]}
    by_dataset_src = summary_src.get("by_dataset")
    _exact_keys(by_dataset_src, {"hc2", "vidstg"}, "pool-coverage dataset summaries")
    projected_aggregates = {}
    for group, source_row in (("all", summary_src["all"]),
                              ("hc2", by_dataset_src["hc2"]),
                              ("vidstg", by_dataset_src["vidstg"])):
        _exact_keys(source_row, agg_fields, f"pool-coverage {group} summary")
        source_group = groups[group]
        expected_case_count = 8 if group == "all" else 4
        if len(source_group) != expected_case_count:
            raise ValueError("pool-coverage anonymous case grouping is incomplete")
        frame_count = sum(row["known_gt_frames"] for row in source_group)
        sum_iou = sum(row["sum_max_bbox_iou"] for row in source_group)
        expected_values = {
            "case_count": expected_case_count,
            "known_gt_frames": frame_count,
            "gt_frames_in_canonical_map": sum(row["gt_frames_in_canonical_map"] for row in source_group),
            "gt_frames_outside_canonical_map": sum(row["gt_frames_outside_canonical_map"] for row in source_group),
            "canonical_gt_frames_with_empty_filtered_pool": sum(row["canonical_gt_frames_with_empty_filtered_pool"] for row in source_group),
            "gt_frames_with_empty_filtered_pool": sum(row["gt_frames_with_empty_filtered_pool"] for row in source_group),
            "frame_weighted_mean_max_bbox_iou": sum_iou / frame_count,
            "case_macro_mean_max_bbox_iou": sum(row["mean_max_bbox_iou"] for row in source_group) / expected_case_count,
            "gt_frames_max_iou_ge_0_25": sum(row["gt_frames_max_iou_ge_0_25"] for row in source_group),
            "fraction_gt_frames_max_iou_ge_0_25": sum(row["gt_frames_max_iou_ge_0_25"] for row in source_group) / frame_count,
            "gt_frames_max_iou_ge_0_50": sum(row["gt_frames_max_iou_ge_0_50"] for row in source_group),
            "fraction_gt_frames_max_iou_ge_0_50": sum(row["gt_frames_max_iou_ge_0_50"] for row in source_group) / frame_count,
        }
        aggregate = {}
        for key, expected in expected_values.items():
            if isinstance(expected, int):
                aggregate[key] = _int(source_row[key], f"pool.summary.{group}.{key}")
                if aggregate[key] != expected:
                    raise ValueError(f"pool-coverage aggregate {group}.{key} differs from case rows")
            else:
                aggregate[key] = _close(source_row[key], expected, f"pool.summary.{group}.{key}")
        projected_aggregates[group] = aggregate

    coverage_public = {
        "schema": "stvg-native-instance-public-filtered-pool-coverage-v1",
        "status": "pass", "GT_read": True,
        "GT_read_after_complete_prediction_seal": True,
        "coverage_estimand": fixed_text["coverage_estimand"],
        "thresholds": [0.25, 0.5],
        "denominator_definition": fixed_text["denominator_definition"],
        "candidate_policy": fixed_text["candidate_policy"],
        "claim_boundaries": claim_boundaries,
        "bindings": checked_bindings,
        "summary": {"all": projected_aggregates["all"],
                    "by_dataset": {"hc2": projected_aggregates["hc2"],
                                   "vidstg": projected_aggregates["vidstg"]}},
        "cases": rows_out,
        "redactions": {key: False for key in redactions},
        "filtered_rle_extension_distributed": False,
    }

    root_receipt_path = BASE / POOL_COVERAGE_READBACK_NAME
    root_receipt = _read_json(root_receipt_path)
    _exact_keys(root_receipt, {"status", "scope", "coverage_sha256", "appendix_sha256",
                               "CPU_only", "raw_GT_read_by_this_reader"},
                "root pool-coverage readback")
    expected_scope = "independent saved pool coverage scalar aggregation and source binding readback; no new label read or prediction changes"
    if (root_receipt.get("status") != "pass"
            or root_receipt.get("coverage_sha256") != _sha(rows_path)
            or root_receipt.get("appendix_sha256") != _sha(appendix_path)
            or root_receipt.get("CPU_only") is not True
            or root_receipt.get("raw_GT_read_by_this_reader") is not False
            or root_receipt.get("scope") != expected_scope):
        raise ValueError("root pool-coverage readback does not bind the source result and appendix")
    pool_root_public = {"schema": "stvg-native-instance-root-pool-coverage-readback-public-v1",
                        "status": "pass", "coverage_sha256": _sha(rows_path),
                        "appendix_sha256": _sha(appendix_path), "CPU_only": True,
                        "raw_GT_read_by_this_reader": False,
                        "reader_sha256": _sha(ROOT / "scripts/review_stvg_native_instance_coverage_v1.py"),
                        "scope": "independent saved pool coverage scalar aggregation and source binding readback; no new label read or prediction changes"}
    return coverage_public, pool_root_public


def _project_cost_receipts(rows: Mapping[str, Any], costs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read only whitelisted timing fields from the two no-GT receipt families."""
    cases = {row["case_id"]: row for row in costs["records"]}
    by_domain = {row["case_id"]: row["dataset"] for row in rows["rows"] if row["arm"] == "F"}
    output = []
    for cid in CASE_IDS:
        alias = CASE_ALIAS[cid]
        tracking = _read_json(BASE / "TRACKING" / cid / "ACCEPTED.json")
        if tracking.get("status") != "pass" or tracking.get("GT_read") is not False:
            raise ValueError("tracking cost receipt is not a passing GT-free record")
        reused = _bool(tracking.get("reused"), "tracking.reused")
        track_seconds = _num(tracking.get("actual_tracking_wall_s"),
                             "tracking.actual_tracking_wall_s", minimum=0)
        teacher_path = BASE / "teacher" / f"{cid}.receipt.json"
        teacher = _read_json(teacher_path)
        row_baseline = next(row for row in rows["rows"]
                            if row["case_id"] == alias and row["arm"] == "F")
        raw_dataset = "hc2" if cid.startswith("hc2_") else "vidstg"
        if (teacher.get("schema") != "stvg-native-instance-teacher-receipt-v1"
                or teacher.get("status") != "complete" or teacher.get("GT_read") is not False
                or teacher.get("query_text_in_receipt") is not False
                or teacher.get("dataset") != raw_dataset
                or teacher.get("source_case_sha256") != row_baseline["bindings"]["source_case_sha256"]
                or teacher.get("tracking_acceptance_sha256") != _sha(BASE / "TRACKING" / cid / "ACCEPTED.json")
                or row_baseline["bindings"]["teacher_receipt_sha256"] != _sha(teacher_path)):
            raise ValueError("teacher timing receipt does not bind the scored query-free source")
        teacher_cost = teacher.get("costs")
        if not isinstance(teacher_cost, dict):
            raise ValueError("teacher timing receipt has no costs map")
        teacher_times = {key: _num(teacher_cost.get(key), f"teacher.costs.{key}", minimum=0)
                         for key in ("worker_wall_s", "model_load_s", "model_preprocess_s", "model_generate_s")}
        output.append({"case_id": alias, "dataset": DATASETS[raw_dataset],
                       "tracking_history": "reused_historical" if reused else "current_run",
                       "tracking_wall_seconds": track_seconds,
                       "tracking_wall_seconds_available": track_seconds is not None,
                       "teacher_receipt_sha256": _sha(teacher_path),
                       "teacher_seconds": teacher_times,
                       "teacher_seconds_available": {key: value is not None
                                                      for key, value in teacher_times.items()},
                       "source_capture_seconds": cases[alias]["source_capture_seconds"],
                       "spatial_fit_and_full_redecode_seconds": cases[alias]["algorithm_costs"]})
    return sorted(output, key=lambda row: row["case_id"])


def _project_recovery() -> dict[str, Any]:
    """Project the approved teacher stop and four reader-dispatch recoveries."""
    recovery_ids = ("postseal_roster_scope002", "postseal_numpy_dtype003",
                    "postseal_numpy_rng004", "postseal_native_clock005")
    allowed_fields = (
        "status", "scope", "GT_read", "CPU_only", "new_GPU_calls",
        "scientific_predictions_changed", "new_reader_sha256", "root_math_sha256",
        "root_cost_sha256", "old_runtime_sha256", "corrected_scorer_sha256",
        "score_rows_sha256", "root_postseal_sha256",
        "synthetic_cropped_absolute_clock_contract", "invalid_fps_rejected")
    hash_fields = {"new_reader_sha256", "root_math_sha256", "root_cost_sha256",
                   "old_runtime_sha256", "corrected_scorer_sha256",
                   "score_rows_sha256", "root_postseal_sha256"}
    bool_fields = {"GT_read", "CPU_only", "scientific_predictions_changed",
                   "synthetic_cropped_absolute_clock_contract", "invalid_fps_rejected"}
    teacher_capture = BASE / "recovery/teacher_gpu_storage002/CAPTURE.json"
    if not teacher_capture.is_file():
        raise FileNotFoundError("teacher storage-retry recovery capture is missing")
    records = [{"recovery_id": "teacher_gpu_storage002",
                "event": "initial_teacher_resource_guard_stop",
                "capture_sha256": _sha(teacher_capture),
                "aborted_teacher_process_count": 1,
                "generation_call_count": None,
                "output_token_count": None,
                "accepted_teacher_output_count": 0,
                "student_output_count_before_stop": 0,
                "gpu_headroom_guard_stop": True,
                "cuda_oom": False,
                "guard_threshold_gib": 4,
                "retry_gpu_memory_budget_gib": 12,
                "retry_cpu_memory_budget_gib": 8,
                "retry_generation_configuration_unchanged": True,
                "GT_read": False}]
    for recovery_id in recovery_ids:
        directory = BASE / "recovery" / recovery_id
        projected = {"recovery_id": recovery_id}
        for kind in ("CAPTURE", "RESOLUTION"):
            path = directory / f"{kind}.json"
            if not path.is_file():
                raise FileNotFoundError(f"authorized recovery receipt is missing: {recovery_id}/{kind}")
            source = _read_json(path)
            values = {}
            for field in allowed_fields:
                if field not in source:
                    continue
                value = source[field]
                if field in hash_fields:
                    value = None if value is None else _sha_value(value, f"{recovery_id}.{kind}.{field}")
                elif field in bool_fields:
                    value = _bool(value, f"{recovery_id}.{kind}.{field}")
                elif field == "new_GPU_calls":
                    value = _int(value, f"{recovery_id}.{kind}.{field}")
                elif field in {"status", "scope"}:
                    if not isinstance(value, str) or len(value) > 500:
                        raise ValueError(f"{recovery_id}.{kind}.{field} is not bounded text")
                values[field] = value
            if not {"status", "scope", "GT_read"} <= set(values):
                raise ValueError(f"{recovery_id}.{kind} is missing its approved status/scope/GT declaration")
            projected[kind.lower()] = {"sha256": _sha(path), **values}
        records.append(projected)
    return {"schema": "stvg-native-instance-public-recovery-v2", "status": "pass",
            "count": len(records), "records": records}


def _project_generation_file() -> dict[str, Any]:
    return _project_generation(_read_json(BASE / GENERATION_RECEIPT_NAME))


def _source_binding_rows() -> tuple[set[str], list[dict[str, Any]]]:
    paths: set[str] = set()
    bindings: list[dict[str, Any]] = []
    for runtime_name in RUNTIME_NAMES:
        runtime_path = BASE / runtime_name
        runtime = _read_json(runtime_path)
        if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
            raise ValueError(f"{runtime_name} is not a locked GT-free runtime")
        scope_count = runtime.get("count", runtime.get("cases"))
        if runtime.get("count") is not None and runtime.get("cases") is not None:
            if runtime.get("count") != runtime.get("cases"):
                raise ValueError(f"{runtime_name} declares conflicting source counts")
        if scope_count != 8:
            raise ValueError(f"{runtime_name} is not bound to the fixed eight-source scope")
        for group in ("pins", "protected"):
            mapping = runtime.get(group)
            if not isinstance(mapping, dict):
                raise ValueError(f"{runtime_name}.{group} is malformed")
            for rel, declared in sorted(mapping.items()):
                digest = _sha_value(declared, f"{runtime_name}.{group} SHA")
                row = {"runtime": runtime_name, "binding": group,
                       "sha256": digest, "published": False,
                       "source": "private_or_protected_binding",
                       "byte_exact": "runtime_declared_only"}
                try:
                    public_rel = _public_path(rel)
                except ValueError:
                    bindings.append(row)
                    continue
                source_path = ROOT / public_rel
                if (not source_path.is_file() or source_path.is_symlink()
                        or ROOT not in source_path.resolve().parents or _sha(source_path) != digest):
                    raise ValueError(f"runtime-pinned public source bytes differ: {public_rel}")
                row.update({"source": public_rel, "published": True, "byte_exact": True})
                paths.add(public_rel)
                bindings.append(row)

    # Independently verify literal source hashes embedded in the scientific
    # fit helper. Other frozen source dependencies are also byte checked by
    # their three stage runtimes above.
    helper = "scripts/stvg_native_instance_spatial_fit_v1.py"
    if helper not in paths:
        raise ValueError("native-instance spatial fit helper is absent from runtime source closure")
    module = ast.parse((ROOT / helper).read_text(encoding="utf-8"), filename=helper)
    source_map = None
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "SOURCE_BINDINGS"
                for target in node.targets):
            source_map = ast.literal_eval(node.value)
            break
    if not isinstance(source_map, dict) or not source_map:
        raise ValueError("native spatial helper has no literal SOURCE_BINDINGS")
    for rel, digest_value in sorted(source_map.items()):
        digest = digest_value.get("sha256") if isinstance(digest_value, dict) else digest_value
        digest = _sha_value(digest, "embedded scientific source SHA")
        row = {"runtime": "embedded_SOURCE_BINDINGS:" + helper,
               "binding": "scientific_source", "sha256": digest,
               "published": False, "source": "private_or_protected_binding",
               "byte_exact": "source_declared_only"}
        try:
            public_rel = _public_path(rel)
        except ValueError:
            bindings.append(row)
            continue
        source_path = ROOT / public_rel
        if (not source_path.is_file() or source_path.is_symlink()
                or ROOT not in source_path.resolve().parents or _sha(source_path) != digest):
            raise ValueError(f"embedded scientific dependency differs: {public_rel}")
        row.update({"source": public_rel, "published": True, "byte_exact": True})
        paths.add(public_rel)
        bindings.append(row)
    bound_public_paths = {row.get("source") for row in bindings
                          if row.get("published") is True}
    for rel in sorted(FIXED_PUBLIC_SOURCE_PATHS):
        _public_path(rel)
        source_path = ROOT / rel
        if (not source_path.is_file() or source_path.is_symlink()
                or ROOT not in source_path.resolve().parents):
            raise FileNotFoundError(f"fixed public source is missing or unsafe: {rel}")
        digest = _sha(source_path)
        if rel in bound_public_paths:
            matching = [row for row in bindings
                        if row.get("published") is True and row.get("source") == rel]
            if any(row.get("sha256") != digest or row.get("byte_exact") is not True
                   for row in matching):
                raise ValueError(f"runtime-bound public source hash differs: {rel}")
            continue
        bindings.append({"runtime": "explicit_public_source_closure",
                         "binding": "fixed_source", "sha256": digest,
                         "published": True, "source": rel, "byte_exact": True})
        bound_public_paths.add(rel)
        paths.add(rel)

    # Preserve the independent root execution readback as provenance for the
    # eleven new CPU dispatch/reader sources. This checks its exact source map
    # against current bytes, then records hash-only binding rows; no raw
    # runtime, private path, or execution receipt payload is exported. The
    # receipt covers post-seal work, so its GT_read flag is not a gate here.
    execution_readback = _read_json(BASE / FINAL_EXECUTION_READBACK_NAME)
    if (execution_readback.get("status") != "pass"
            or execution_readback.get("CPU_only") is not True):
        raise ValueError("final execution source readback is not a passing CPU-only receipt")
    execution_sources = execution_readback.get("source_bindings")
    if (not isinstance(execution_sources, dict)
            or set(execution_sources) != set(FINAL_EXECUTION_SOURCES)):
        raise ValueError("final execution source readback has an unexpected source inventory")
    for rel, declared in sorted(execution_sources.items()):
        digest = _sha_value(declared, "final execution source SHA")
        source_path = ROOT / rel
        if (not source_path.is_file() or source_path.is_symlink()
                or ROOT not in source_path.resolve().parents or _sha(source_path) != digest):
            raise ValueError(f"final execution source bytes differ: {rel}")
        binding = {"runtime": FINAL_EXECUTION_READBACK_NAME,
                   "binding": "root_final_execution_source", "sha256": digest,
                   "published": True, "source": rel, "byte_exact": True}
        bindings.append(binding)
        paths.add(rel)
    return paths, bindings


def _safe_summary_projection(documents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for name, doc in documents.items():
        status = doc.get("status")
        if status not in {"pass", "complete", "pending_root_review_visualization_publication"}:
            raise ValueError(f"{name} is not in an accepted terminal/pending state")
        # Explicit field-by-field scalar projection. No free-form path, text,
        # query, model output, list of frames, or artifact inventory is copied.
        item = {"name": name, "status": status}
        for field in ("schema", "case_count", "row_count", "arm_count", "actual_fit_count",
                      "saved_optimizer_steps", "saved_full_model_readouts", "source_capture_calls",
                "checks", "checks_passed", "checks_failed", "fit_count", "expected_fit_count",
                      "arm_summary_count", "contrast_count",
                      "optimizer_steps", "CPU_only", "GPU_initialized", "model_replayed_by_reader",
                      "model_replayed", "raw_GT_read_by_this_reader",
                      "all_case_source_bindings_match", "all_tails_and_unknown_membership_match",
                      "GT_read", "scoring_status", "experiment_complete"):
            if field not in doc:
                continue
            value = doc[field]
            if isinstance(value, bool) or value is None:
                item[field] = value
            elif isinstance(value, int):
                item[field] = _int(value, f"{name}.{field}")
            elif isinstance(value, str) and field == "schema" and value.startswith("stvg-"):
                item[field] = value[:100]
            elif field == "scoring_status" and value == "complete":
                item[field] = "complete"
            else:
                raise ValueError(f"{name}.{field} is outside the public scalar allowlist")
        for field in ("student_barrier_sha256", "runtime_sha256", "reference_lock_sha256",
                      "roster_sha256", "rows_sha256", "summary_sha256",
                      "prediction_barrier_sha256", "direct_readout_sha256", "completion_sha256",
                      "reader_sha256", "coverage_sha256", "appendix_sha256"):
            if field in doc:
                item[field] = _sha_value(doc[field], f"{name}.{field}")
        rows.append(item)
    return {"schema": "stvg-native-instance-public-root-readbacks-v1",
            "status": "pass", "receipts": rows}


def _scan_public_document(data: bytes, label: str) -> None:
    lower = data.lower()
    forbidden = (b"/home/wwww/", b"/tmp/", b"source_cases/", b"private_inputs/",
                 b"private_views/", b"query_text", b"raw_output", b"marker_video_path",
                 b"mask_data_dir", b"annotations/", b"source_case_path")
    if any(token in lower for token in forbidden):
        raise ValueError(f"public document contains a forbidden private marker: {label}")
    for cid in CASE_IDS:
        if cid.encode("ascii") in data:
            raise ValueError(f"public document contains an unaliased case key: {label}")


def _save_projection(name: str, value: Any) -> str:
    path = OUT / name
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _scan_public_document(encoded, path.name)
    _write_new(path, value)
    return path.relative_to(ROOT).as_posix()


def _stage(paths: set[str]) -> list[dict[str, Any]]:
    PUBLIC_STAGE.mkdir(parents=True, exist_ok=False)
    manifest = []
    for rel in sorted(paths):
        source = ROOT / rel
        if (not source.is_file() or source.is_symlink()
                or ROOT not in source.resolve().parents):
            raise FileNotFoundError(f"explicit public file is missing: {rel}")
        data = source.read_bytes()
        if rel in {REPORT_PATH.relative_to(ROOT).as_posix(),
                   PDF_PATH.relative_to(ROOT).as_posix(),
                   PNG_PATH.relative_to(ROOT).as_posix()}:
            _scan_public_document(data, rel)
        target = PUBLIC_STAGE / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest.append({"path": rel, "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest(),
                         "blob_sha": hashlib.sha1(b"blob " + str(len(data)).encode()
                                                  + b"\0" + data).hexdigest(),
                         "mode": "100644", "type": "blob",
                         "encoding": "base64" if source.suffix.lower() in {".pdf", ".png"}
                         else "utf-8"})
    return manifest


def _read_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    required = [PREDICTION_BARRIER_NAME, MATH_RECEIPT_NAME, COST_RECEIPT_NAME,
                "POSTSEAL_SCORE_ROWS.json", "POSTSEAL_SCORE_SUMMARY.json", "CPU_COMPLETION.json",
                REPORT_RECEIPT_NAME, VISUAL_RECEIPT_NAME, GENERATION_RECEIPT_NAME,
                GEOMETRY_RECEIPT_NAME, POOL_COVERAGE_NAME, POOL_COVERAGE_READBACK_NAME]
    missing = [name for name in required if not (BASE / name).is_file()]
    missing.extend(path for path in (REPORT_PATH.relative_to(ROOT).as_posix(),
                                     PROTOCOL_PATH.relative_to(ROOT).as_posix(),
                                     PDF_PATH.relative_to(ROOT).as_posix(),
                                     PNG_PATH.relative_to(ROOT).as_posix(),
                                     POOL_COVERAGE_APPENDIX_PATH.relative_to(ROOT).as_posix())
                   if not (ROOT / path).is_file())
    if missing:
        raise FileNotFoundError("native-instance public inputs are not sealed: " + ", ".join(missing))
    barrier = _read_json(BASE / PREDICTION_BARRIER_NAME)
    if barrier.get("status") != "complete" or barrier.get("count") != 8 or barrier.get("GT_read") is not False:
        raise ValueError("prediction barrier is incomplete or declares prior GT access")
    sealed_files = barrier.get("files")
    if not isinstance(sealed_files, list) or len(sealed_files) != 37:
        raise ValueError("prediction seal does not contain the locked 37-file inventory")
    paths_seen = set()
    student_phase_digests = []
    for item in sealed_files:
        _exact_keys(item, {"path", "sha256", "bytes"}, "prediction seal file entry")
        rel = item.get("path")
        if (not isinstance(rel, str) or not rel or "\\" in rel
                or Path(rel).is_absolute() or any(part in {"", ".", ".."} for part in Path(rel).parts)
                or rel in paths_seen):
            raise ValueError("prediction seal file paths are malformed or duplicated")
        paths_seen.add(rel)
        _sha_value(item.get("sha256"), "prediction seal file SHA")
        _int(item.get("bytes"), "prediction seal file bytes")
        if Path(rel).name == "STUDENT_GLOBAL_BARRIER.json":
            student_phase_digests.append(item["sha256"])
    if len(student_phase_digests) != 1:
        raise ValueError("prediction seal must bind exactly one student phase barrier")
    student_barrier_sha = student_phase_digests[0]
    barrier_sha = _sha(BASE / PREDICTION_BARRIER_NAME)

    rows_raw = _read_json(BASE / "POSTSEAL_SCORE_ROWS.json")
    summary_raw = _read_json(BASE / "POSTSEAL_SCORE_SUMMARY.json")
    rows = _project_rows(rows_raw, barrier_sha, student_barrier_sha)
    summary = _project_summary(summary_raw, rows)
    if summary_raw.get("bindings") != rows_raw.get("bindings"):
        raise ValueError("summary and row source bindings differ")
    rows_path, summary_path = BASE / "POSTSEAL_SCORE_ROWS.json", BASE / "POSTSEAL_SCORE_SUMMARY.json"

    completion = _read_json(BASE / "CPU_COMPLETION.json")
    if (completion.get("schema") != "stvg-native-instance-cpu-completion-v1"
            or completion.get("status") != "pending_root_review_visualization_publication"
            or completion.get("scoring_status") != "complete"
            or completion.get("experiment_complete") is not False
            or completion.get("GT_read") is not True or completion.get("case_count") != 8
            or completion.get("arm_count") != len(ARMS) or completion.get("row_count") != 72
            or completion.get("prediction_barrier_sha256") != barrier_sha
            or completion.get("rows_sha256") != _canonical_sha(rows_raw)
            or completion.get("summary_sha256") != _canonical_sha(summary_raw)):
        raise ValueError("CPU completion receipt does not bind the final score files")

    math_raw = _read_json(BASE / MATH_RECEIPT_NAME)
    math_public = _project_math(math_raw, barrier_sha, rows["bindings"])
    cost_raw = _read_json(BASE / COST_RECEIPT_NAME)
    cost_public = _project_costs(cost_raw, rows, math_public, barrier_sha)
    cost_detail = _project_cost_receipts(rows, cost_public)
    cost_public["tracking_and_teacher_receipts"] = cost_detail

    root_postseal_raw = _read_json(BASE / REPORT_RECEIPT_NAME)
    root_postseal = _project_root_postseal(root_postseal_raw, rows_path, summary_path, barrier_sha)
    visual_raw = _read_json(BASE / VISUAL_RECEIPT_NAME)
    visual = _project_visual(visual_raw, REPORT_PATH, PDF_PATH, PNG_PATH)
    generation = _project_generation_file()
    geometry_raw = _read_json(BASE / GEOMETRY_RECEIPT_NAME)
    geometry = _project_geometry_readback(geometry_raw)
    coverage_raw = _read_json(BASE / POOL_COVERAGE_NAME)
    coverage, pool_coverage_readback = _project_pool_coverage(
        coverage_raw, BASE / POOL_COVERAGE_NAME, POOL_COVERAGE_APPENDIX_PATH, barrier_sha)
    if rows_raw["bindings"]["teacher_barrier_sha256"] == "0" * 64:
        raise ValueError("teacher barrier binding is invalid")

    summary_docs = {
        MATH_RECEIPT_NAME: math_raw,
        COST_RECEIPT_NAME: cost_raw,
        "CPU_COMPLETION.json": completion,
        REPORT_RECEIPT_NAME: root_postseal_raw,
        VISUAL_RECEIPT_NAME: visual_raw,
        GENERATION_RECEIPT_NAME: _read_json(BASE / GENERATION_RECEIPT_NAME),
        GEOMETRY_RECEIPT_NAME: geometry_raw,
        POOL_COVERAGE_READBACK_NAME: _read_json(BASE / POOL_COVERAGE_READBACK_NAME),
    }
    root_summary = _safe_summary_projection(summary_docs)
    return rows, summary, math_public, cost_public, {
        "root_postseal": root_postseal, "visual": visual,
        "generation": generation, "root_summaries": root_summary,
        "recovery": _project_recovery(),
        "geometry": geometry,
        "pool_coverage": coverage,
        "pool_coverage_readback": pool_coverage_readback,
    }


def main() -> None:
    if OUT.exists() or PUBLIC_STAGE.exists() or MANIFEST_PATH.exists() or SOURCE_READBACK_PATH.exists():
        raise FileExistsError("public export outputs already exist; refusing overwrite")
    rows, summary, math_public, cost_public, receipts = _read_inputs()
    source_paths, binding_rows = _source_binding_rows()

    fixed_sources = FIXED_PUBLIC_SOURCE_PATHS
    source_paths |= fixed_sources
    source_paths.add(REPORT_PATH.relative_to(ROOT).as_posix())
    source_paths.add(PROTOCOL_PATH.relative_to(ROOT).as_posix())
    source_paths.add(PDF_PATH.relative_to(ROOT).as_posix())
    source_paths.add(PNG_PATH.relative_to(ROOT).as_posix())
    source_paths.add(POOL_COVERAGE_APPENDIX_PATH.relative_to(ROOT).as_posix())
    for rel in source_paths:
        if rel.startswith(PUBLIC_SOURCE_PREFIXES) or rel in PUBLIC_EXACT_SOURCE_PATHS:
            _public_path(rel)
        elif rel not in {REPORT_PATH.relative_to(ROOT).as_posix(),
                         PROTOCOL_PATH.relative_to(ROOT).as_posix(),
                         PDF_PATH.relative_to(ROOT).as_posix(),
                         PNG_PATH.relative_to(ROOT).as_posix(),
                         POOL_COVERAGE_APPENDIX_PATH.relative_to(ROOT).as_posix()}:
            raise ValueError("non-public path entered the explicit source closure")
    for rel in (REPORT_PATH.relative_to(ROOT).as_posix(),
                PROTOCOL_PATH.relative_to(ROOT).as_posix(),
                PDF_PATH.relative_to(ROOT).as_posix(),
                PNG_PATH.relative_to(ROOT).as_posix(),
                POOL_COVERAGE_APPENDIX_PATH.relative_to(ROOT).as_posix()):
        _scan_public_document((ROOT / rel).read_bytes(), rel)

    OUT.mkdir(parents=True, exist_ok=False)
    postseal_path = _save_projection("postseal_metrics.json", rows)
    summary_path = _save_projection("postseal_summary.json", summary)
    math_path = _save_projection("student_math_readback.json", math_public)
    cost_path = _save_projection("measured_costs.json", cost_public)
    roots_path = _save_projection("root_readbacks.json", receipts["root_summaries"])
    visual_path = _save_projection("visual_review.json", receipts["visual"])
    root_postseal_path = _save_projection("root_postseal_readback.json", receipts["root_postseal"])
    generation_path = _save_projection("generation_configuration.json", receipts["generation"])
    geometry_path = _save_projection("independent_geometry_readback.json", receipts["geometry"])
    pool_coverage_path = _save_projection("pool_candidate_coverage.json", receipts["pool_coverage"])
    pool_coverage_readback_path = _save_projection(
        "pool_coverage_readback.json", receipts["pool_coverage_readback"])
    recovery_path = _save_projection("recovery_metadata.json", receipts["recovery"])

    binding_summary = {"schema": "stvg-native-instance-public-source-bindings-v1",
                       "status": "pass", "runtime_sha256": {
                           name: _sha(BASE / name) for name in RUNTIME_NAMES},
                       "binding_count": len(binding_rows),
                       "public_source_count": sum(row["published"] is True for row in binding_rows),
                       "private_hash_only_count": sum(row["published"] is False for row in binding_rows),
                       "bindings": binding_rows}
    bindings_path = _save_projection("runtime_source_bindings.json", binding_summary)
    scalar = {"schema": "stvg-native-instance-public-scalar-readback-v1", "status": "pass",
              "scope": "anonymous fixed eight-source native-instance experiment",
              "case_aliases": sorted(CASE_ALIAS.values()), "case_count": 8,
              "row_count": len(rows["rows"]), "arm_count": len(ARMS),
              "arm_summary_count": len(summary["arm_metric_summaries"]),
              "contrast_count": len(summary["contrasts"]),
              "candidate_detection_recall": "not_measured", "GT_read": True,
              "GT_used_for_method_or_hyperparameter_selection": False,
              "sha256": {"postseal_metrics": _canonical_sha(rows),
                         "postseal_summary": _canonical_sha(summary),
                         "student_math": _canonical_sha(math_public),
                         "costs": _canonical_sha(cost_public),
                         "visual_review": _canonical_sha(receipts["visual"]),
                         "independent_geometry_readback": _canonical_sha(receipts["geometry"]),
                         "pool_candidate_coverage": _canonical_sha(receipts["pool_coverage"]),
                         "pool_coverage_readback": _canonical_sha(receipts["pool_coverage_readback"]),
                         "pool_coverage_appendix": _sha(POOL_COVERAGE_APPENDIX_PATH)},
              "pool_candidate_coverage_case_count": len(receipts["pool_coverage"]["cases"]),
              "pool_candidate_coverage_known_gt_frames": receipts["pool_coverage"]["summary"]["all"]["known_gt_frames"],
              "independent_geometry_checks": receipts["geometry"]["checks"]}
    if (scalar["row_count"] != 72 or scalar["arm_summary_count"] != 90
            or scalar["contrast_count"] != 80 or scalar["pool_candidate_coverage_case_count"] != 8
            or scalar["independent_geometry_checks"] != 31842):
        raise ValueError("public export counts differ from the locked 72/90/80 and 8-case readbacks")
    scalar_path = _save_projection("public_scalar_readback.json", scalar)

    extra_outputs = {postseal_path, summary_path, math_path, cost_path, roots_path,
                     visual_path, root_postseal_path, generation_path, recovery_path,
                     geometry_path, pool_coverage_path, pool_coverage_readback_path,
                     bindings_path, scalar_path}
    manifest = _stage(source_paths | extra_outputs)
    _write_new(MANIFEST_PATH, manifest)
    source_readback = {"schema": "stvg-native-instance-public-source-readback-v1",
                       "status": "pass", "files": len(manifest),
                       "bytes": sum(row["bytes"] for row in manifest),
                       "manifest_sha256": _sha(MANIFEST_PATH),
                       "public_source_binding_count": len(binding_rows),
                       "source_bindings_sha256": _sha(OUT / "runtime_source_bindings.json"),
                       "postseal_projection_sha256": _sha(OUT / "postseal_metrics.json"),
                       "summary_projection_sha256": _sha(OUT / "postseal_summary.json"),
                       "cost_projection_sha256": _sha(OUT / "measured_costs.json"),
                       "geometry_projection_sha256": _sha(OUT / "independent_geometry_readback.json"),
                       "pool_coverage_projection_sha256": _sha(OUT / "pool_candidate_coverage.json"),
                       "pool_coverage_readback_projection_sha256": _sha(OUT / "pool_coverage_readback.json"),
                       "pool_coverage_appendix_sha256": _sha(POOL_COVERAGE_APPENDIX_PATH),
                       "report_sha256": _sha(REPORT_PATH), "pdf_sha256": _sha(PDF_PATH),
                       "png_sha256": _sha(PNG_PATH),
                       "raw_prediction_barrier_staged": False,
                       "GT_coordinates_or_annotations_staged": False,
                       "raw_teacher_output_or_masks_staged": False,
                       "student_checkpoints_staged": False,
                       "RGB_or_video_staged": False, "queries_staged": False,
                       "push_performed": False}
    _write_new(SOURCE_READBACK_PATH, source_readback)
    print(json.dumps({"status": "staged_not_pushed", "files": len(manifest),
                      "bytes": source_readback["bytes"],
                      "manifest_sha256": source_readback["manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
