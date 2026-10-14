"""Independent CPU audit for the native-instance anonymous public export.

The default command is intentionally inert.  ``--contracts`` runs only on
synthetic scalar fixtures; ``--audit`` is an explicit later action after the
immutable public stage and all root receipts exist.  The audit never opens
predictions, source tensors, masks, checkpoints, images, videos, or annotations.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_native_instance_spatial_opd_v1"
OUT = ROOT / "results/stvg_native_instance/2026-10-14"
STAGE = BASE / "public_stage"
MANIFEST = BASE / "PUBLIC_MANIFEST.json"
SOURCE_RECEIPT = BASE / "ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json"
AUDIT_RECEIPT = BASE / "PUBLIC_SCALAR_AUDIT.json"
ANONYMOUS_RECEIPT = BASE / "ROOT_ANONYMOUS_SCALAR_READBACK.json"
EXPORTER = ROOT / "scripts/prepare_stvg_native_instance_public_v1.py"
REPORT = ROOT / "docs/STVG_NATIVE_INSTANCE_V1_REVIEWED.md"
PROTOCOL = ROOT / "protocols/stvg_native_instance_spatial_opd_v1.md"
PDF = BASE / "figures/native_instance_summary_reviewed.pdf"
PNG = BASE / "figures/native_instance_summary_reviewed.png"

CASE_IDS = tuple([f"hc2_{i:03d}" for i in range(4)] +
                 [f"vidstg_{i:03d}" for i in range(4)])
ALIAS = {cid: ("HC" if cid.startswith("hc2_") else "V") +
         f"{int(cid[-3:]) + 1:02d}" for cid in CASE_IDS}
DATASETS = {"hc2": "HC", "vidstg": "VidSTG"}
ARMS = ("F", "T", "native_teacher_direct", "direct", "iou",
        "reliability_weighted_iou", "T_direct", "T_iou",
        "T_reliability_weighted_iou")
SPATIAL = ("direct", "iou", "reliability_weighted_iou")
METRICS = ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT",
           "m_sIoU_supervised_GT", "m_sIoU_unsupervised_GT")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
PUBLIC_PREFIXES = ("scripts/", "vg_tta/", "methods/", "protocols/",
                   "external/stvg-r1/")
PUBLIC_EXACT_PATHS = {"external/stvg_runtime/pycocotools/mask.py"}
PUBLIC_SUFFIXES = {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".txt"}
FIXED_PUBLIC_SOURCE_PATHS = {
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
}
FINAL_EXECUTION_SOURCES = {
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
}
DENIED_PATH_PARTS = {"source_cases", "private_inputs", "private_views", "annotations",
                     "annotation", "annos", "weights", "checkpoints", "teacher",
                     "student", "TRACKING", "mask_data_filtered_1_3", "attachments",
                     "conversation", "logs", "worker_exits", "partial"}
RAW_CASE_MARKERS = tuple(cid.encode("ascii") for cid in CASE_IDS)
FORBIDDEN_BYTES = (b"/home/wwww/", b"/tmp/", b"source_cases/", b"private_inputs/",
                   b"private_views/", b"query_text", b"raw_output", b"marker_video_path",
                   b"mask_data_dir", b"annotations/", b"source_case_path")


class AuditError(RuntimeError):
    pass


def _need(ok: Any, message: str) -> None:
    if not ok:
        raise AuditError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuditError(f"invalid JSON input: {path.name}") from exc


def _keys(value: Any, expected: set[str], where: str) -> None:
    _need(isinstance(value, Mapping) and set(value) == expected,
          f"{where}: field set mismatch")


def _sha(value: Any, where: str) -> str:
    _need(isinstance(value, str) and SHA_RE.fullmatch(value) is not None,
          f"{where}: expected lowercase SHA-256")
    return value


def _num(value: Any, where: str, *, null_ok: bool = False,
         low: float | None = None, high: float | None = None) -> float | None:
    if value is None and null_ok:
        return None
    _need(not isinstance(value, bool) and isinstance(value, (int, float)),
          f"{where}: expected numeric scalar")
    out = float(value)
    _need(math.isfinite(out), f"{where}: nonfinite number")
    if low is not None:
        _need(out >= low, f"{where}: below allowed range")
    if high is not None:
        _need(out <= high, f"{where}: above allowed range")
    return out


def _count(value: Any, where: str, *, low: int = 0) -> int:
    _need(isinstance(value, int) and not isinstance(value, bool) and value >= low,
          f"{where}: expected integer count")
    return value


def _alias(case_id: Any) -> str:
    _need(isinstance(case_id, str) and case_id in ALIAS,
          "unrecognized source identity in a local scalar file")
    return ALIAS[case_id]


def _sealed_prediction_lineage() -> dict[str, Any]:
    """Bind the global seal to its three phase barriers and score inputs."""
    phase_names = {
        "tracking_barrier_sha256": "TRACKING_GLOBAL_BARRIER.json",
        "teacher_barrier_sha256": "TEACHER_GLOBAL_BARRIER.json",
        "student_barrier_sha256": "STUDENT_GLOBAL_BARRIER.json",
    }
    global_path = BASE / "GLOBAL_PREDICTION_BARRIER.json"
    doc = _read_json(global_path)
    _need(doc.get("status") == "complete" and doc.get("count") == 8
          and doc.get("GT_read") is False, "global prediction barrier is incomplete")
    files = doc.get("files")
    _need(isinstance(files, list) and len(files) == 37,
          "global prediction seal does not contain the locked file inventory")
    seen_paths: set[str] = set()
    by_name: dict[str, list[Mapping[str, Any]]] = {name: [] for name in phase_names.values()}
    for item in files:
        _keys(item, {"path", "sha256", "bytes"}, "global seal file row")
        rel = item.get("path")
        _need(isinstance(rel, str) and rel and "\\" not in rel,
              "global seal path is malformed")
        rel_path = PurePosixPath(rel)
        _need(not rel_path.is_absolute()
              and all(part not in {"", ".", ".."} for part in rel_path.parts),
              "global seal path is absolute or traverses directories")
        _need(rel not in seen_paths, "global seal repeats a file path")
        seen_paths.add(rel)
        _sha(item.get("sha256"), "global seal file digest")
        _count(item.get("bytes"), "global seal file byte count")
        if rel_path.name in by_name:
            by_name[rel_path.name].append(item)
    phase_hashes: dict[str, str] = {}
    for binding_key, filename in phase_names.items():
        matches = by_name[filename]
        _need(len(matches) == 1, f"global seal must bind exactly one {filename}")
        phase_path = BASE / filename
        digest = _sha_file(phase_path)
        _need(matches[0]["sha256"] == digest,
              f"global seal {filename} hash differs from its actual barrier")
        phase_hashes[binding_key] = digest
    input_paths = {
        "roster_sha256": BASE / "ROSTER_LOCK.json",
        "reference_lock_sha256": BASE / "REFERENCE_LOCK.json",
        "direct_teacher_readout_sha256": BASE / "DIRECT_TEACHER_READOUT.json",
        **{key: BASE / name for key, name in phase_names.items()},
    }
    input_hashes = {key: _sha_file(path) for key, path in input_paths.items()}
    return {"prediction_barrier_sha256": _sha_file(global_path),
            "bindings": input_hashes, "phase_hashes": phase_hashes}


def _privacy_scan(path: str, data: bytes) -> None:
    low = data.lower()
    _need(not any(marker in low for marker in FORBIDDEN_BYTES),
          f"private content marker found in public document: {path}")
    _need(not any(marker in data for marker in RAW_CASE_MARKERS),
          f"unaliased source enumeration found in public document: {path}")


def _path_is_public(rel: Any) -> bool:
    if not isinstance(rel, str) or "\\" in rel:
        return False
    path = PurePosixPath(rel)
    return (not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)
            and not any(part in DENIED_PATH_PARTS for part in path.parts)
            and (rel.startswith(PUBLIC_PREFIXES) or rel in PUBLIC_EXACT_PATHS)
            and path.suffix.lower() in PUBLIC_SUFFIXES
            and rel != "methods/CURRENT_METHOD.json")


def _expected_rows(source: Mapping[str, Any],
                   lineage: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Independent field-by-field projection of all 72 source-arm rows."""
    _need(source.get("schema") == "stvg-native-instance-postseal-rows-v1"
          and source.get("status") == "complete" and source.get("GT_read") is True
          and source.get("metric_scale") == "0_to_1" and source.get("case_count") == 8
          and source.get("arm_count") == 9, "postseal rows are not the complete frozen matrix")
    barrier = source.get("prediction_barrier")
    _need(isinstance(barrier, dict), "missing prediction barrier binding")
    barrier_sha = _sha(barrier.get("sha256"), "prediction barrier")
    binds = source.get("bindings")
    binding_fields = {"roster_sha256", "reference_lock_sha256", "direct_teacher_readout_sha256",
                      "tracking_barrier_sha256", "teacher_barrier_sha256", "student_barrier_sha256"}
    _keys(binds, binding_fields, "postseal global bindings")
    bindings = {k: _sha(v, "global binding") for k, v in binds.items()}
    if lineage is not None:
        _need(barrier_sha == lineage["prediction_barrier_sha256"],
              "score rows do not bind the actual global prediction barrier")
        _need(bindings == lineage["bindings"],
              "score-row source/phase bindings differ from actual frozen files")
    raw_rows = source.get("rows")
    _need(isinstance(raw_rows, list) and len(raw_rows) == 72, "expected all 72 rows")
    seen: set[tuple[str, str]] = set()
    case_binding: dict[str, dict[str, Any]] = {}
    count_fields = ("source_grid_positions", "source_grid_supervised_position_count",
                    "source_grid_unsupervised_position_count", "supervised_anchor_count",
                    "unsupervised_position_count", "dense_gt_known_frame_count",
                    "source_grid_gt_known_frame_count", "off_grid_gt_known_frame_count",
                    "supervised_gt_known_frame_count", "unsupervised_gt_known_frame_count")
    coverage_fields = ("frame_count", "present_frame_count", "missing_frame_count",
                       "present_run_count", "fragmentation_gap_count", "interval_frame_count",
                       "interval_present_frame_count")
    geometry_fields = ("gt_known_frame_count", "teacher_tube_frame_overlap_count",
                       "gt_known_frames_with_present_mask_count",
                       "gt_known_frames_without_present_mask_count")
    output = []
    for row in raw_rows:
        _need(isinstance(row, dict), "row is not an object")
        cid, dataset, arm = row.get("case_id"), row.get("dataset"), row.get("arm")
        alias = _alias(cid)
        _need(dataset in DATASETS and (cid.startswith("hc2_") == (dataset == "hc2")),
              "row dataset/case binding mismatch")
        _need(arm in ARMS, "unexpected arm")
        identity = (alias, arm)
        _need(identity not in seen, "duplicate case-arm result")
        seen.add(identity)
        _need(row.get("candidate_detection_recall") == "not_measured",
              "candidate recall must remain unmeasured")
        status = row.get("status")
        _need(isinstance(status, str) and status in {
            "scored", "scored_unknown", "teacher_invalid", "invalid", "unknown",
            "abstained", "truncated", "valid_uncapped_selection", "invalid_output",
            "output_truncated_at_cap", "exact_no_update", "actual_fit",
            "invalid_or_abstained_no_teacher_tube", "not_a_fit_arm", "not_scored", "unscored"},
            "unknown result status")
        metric_src = row.get("metrics")
        metrics = {m: None for m in METRICS} if metric_src is None else None
        if metric_src is not None:
            _keys(metric_src, set(METRICS), "row metrics")
            metrics = {m: _num(metric_src[m], f"metric.{m}", null_ok=True, low=0, high=1)
                       for m in METRICS}

        def project_counts(value: Any, fields: tuple[str, ...], label: str) -> Any:
            if value is None:
                return None
            _need(isinstance(value, dict) and set(fields) <= set(value),
                  f"{label} count inventory missing")
            return {field: _count(value[field], f"{label}.{field}") for field in fields}

        support = project_counts(row.get("support"), count_fields, "support")
        coverage = project_counts(row.get("coverage"), coverage_fields, "coverage")
        if coverage is not None:
            coverage["present_fraction"] = _num(row["coverage"].get("present_fraction"),
                                                  "coverage.present_fraction", null_ok=True, low=0, high=1)
        geo_src = row.get("selected_track_gt_geometry_coverage")
        geometry = project_counts(geo_src, geometry_fields, "selected-track geometry")
        if geometry is not None:
            geometry["present_mask_fraction_of_gt_known_frames"] = _num(
                geo_src.get("present_mask_fraction_of_gt_known_frames"),
                "geometry.present-mask fraction", null_ok=True, low=0, high=1)
        fit_src = row.get("fit")
        _need(isinstance(fit_src, dict) and fit_src.get("status") in
              {"not_a_fit_arm", "exact_no_update", "actual_fit"}, "fit status missing/invalid")
        fit = {"status": fit_src["status"],
               "optimizer_steps": _count(fit_src.get("optimizer_steps"), "optimizer_steps"),
               "final_redecode_calls": _count(fit_src.get("final_redecode_calls"), "final_redecode_calls")}
        audit_src = row.get("metric_audit")
        audit = None
        if audit_src is not None:
            audit_fields = {"official_independent_max_abs_error",
                            "native_missing_mask_independent_max_abs_error"}
            metadata_fields = {"interval_frame_bounds_half_open", "source_grid_frames", "scoring_path"}
            audit_shapes = {
                frozenset({"official_independent_max_abs_error", "interval_frame_bounds_half_open",
                           "source_grid_frames"}),
                frozenset({"official_independent_max_abs_error",
                           "native_missing_mask_independent_max_abs_error", "scoring_path"}),
            }
            _need(isinstance(audit_src, dict) and frozenset(audit_src) in audit_shapes
                  and set(audit_src) <= audit_fields | metadata_fields,
                  "metric audit differs from the two frozen scorer schemas")
            audit = {k: _num(audit_src[k], f"metric_audit.{k}", null_ok=True)
                     for k in sorted(set(audit_src) & audit_fields)}
            if "source_grid_frames" in audit_src:
                audit["source_grid_frame_count"] = _count(
                    audit_src["source_grid_frames"], "metric_audit.source_grid_frames")
                bounds = audit_src["interval_frame_bounds_half_open"]
                _need(isinstance(bounds, list) and len(bounds) == 2,
                      "metric audit interval bounds must be a pair")
                bound_values = [_num(value, "metric_audit.interval_frame_bound") for value in bounds]
                _need(all(value is not None for value in bound_values)
                      and bound_values[0] <= bound_values[1],
                      "metric audit interval bounds are malformed")
            if "scoring_path" in audit_src:
                _need(audit_src["scoring_path"] == "official_physical_teacher_missing_as_zero",
                      "native-teacher metric path differs from the fixed scorer")
                audit["scoring_path"] = "official_physical_teacher_missing_as_zero"
        raw_bind = row.get("bindings")
        row_binding_fields = {"source_case_sha256", "student_artifact_sha256", "student_receipt_sha256",
                              "teacher_record_sha256", "teacher_receipt_sha256", "reference_lock_sha256"}
        _keys(raw_bind, row_binding_fields, "row bindings")
        row_bind = {k: (_sha(v, f"row binding.{k}") if v is not None else None)
                    for k, v in raw_bind.items()}
        _need(row_bind["reference_lock_sha256"] == bindings["reference_lock_sha256"],
              "row/global reference-lock mismatch")
        prev = case_binding.setdefault(alias, row_bind)
        _need(prev == row_bind, "per-case provenance changes across arms")
        output.append({"case_id": alias, "dataset": DATASETS[dataset], "arm": arm,
                       "status": status, "metrics": metrics, "support": support,
                       "coverage": coverage, "selected_track_gt_geometry_coverage": geometry,
                       "candidate_detection_recall": "not_measured", "fit": fit,
                       "metric_audit": audit, "bindings": row_bind})
    _need(seen == {(ALIAS[c], arm) for c in CASE_IDS for arm in ARMS},
          "one or more of the 72 case-arm rows is absent")
    return {"schema": "stvg-native-instance-public-postseal-rows-v1", "status": "pass",
            "scope": "anonymous fixed historical development panel", "metric_scale": "0_to_1",
            "case_count": 8, "arm_count": 9, "row_count": 72,
            "candidate_detection_recall": "not_measured", "GT_read": True,
            "prediction_barrier_sha256": barrier_sha, "bindings": bindings,
            "rows": sorted(output, key=lambda x: (x["dataset"], x["case_id"], ARMS.index(x["arm"]))) }


def _statistics(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {k: None for k in ("mean", "median", "min", "max")}
    ordered = sorted(values)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return {"mean": sum(values) / len(values), "median": median,
            "min": min(values), "max": max(values)}


def _expected_summary(source: Mapping[str, Any], rows: Mapping[str, Any]) -> dict[str, Any]:
    _need(source.get("schema") == "stvg-native-instance-postseal-summary-v1"
          and source.get("status") == "complete" and source.get("GT_read") is True
          and source.get("metric_scale") == "0_to_1"
          and source.get("candidate_detection_recall") == "not_measured",
          "postseal summary has wrong status or limitation")
    aggregation = source.get("aggregation")
    _need(isinstance(aggregation, dict) and aggregation.get("bootstrap_replicates") == 10000
          and aggregation.get("seed") == 20261014 and aggregation.get("unit") == "paired source case"
          and aggregation.get("pooled_cross_domain_gate") is False
          and aggregation.get("ci_used_for_arm_or_step_selection") is False,
          "summary aggregation changed from the locked descriptive design")
    summary_rows = source.get("arm_metric_summaries")
    expected_arm_keys = {(d, a, m) for d in DATASETS for a in ARMS for m in METRICS}
    _need(isinstance(summary_rows, list) and len(summary_rows) == 90,
          "expected 90 domain-arm-metric summary cells")
    arms_out, seen = [], set()
    by = {(r["case_id"], r["arm"]): r for r in rows["rows"]}
    for row in summary_rows:
        _need(isinstance(row, dict), "summary row is malformed")
        key = (row.get("dataset"), row.get("arm"), row.get("metric"))
        _need(key in expected_arm_keys and key not in seen, "summary cell missing/duplicate/unknown")
        seen.add(key)
        dataset, arm, metric = key
        ids = sorted(r["case_id"] for r in rows["rows"] if r["arm"] == "F" and r["dataset"] == DATASETS[dataset])
        values = [(cid, by[(cid, arm)]["metrics"][metric]) for cid in ids]
        known = [float(v) for _, v in values if v is not None]
        unknown = sorted(cid for cid, v in values if v is None)
        stats = _statistics(known)
        stats_public = {}
        for stat, value in stats.items():
            actual = _num(row.get(stat), f"summary.{stat}", null_ok=True)
            _need(actual == value or (actual is not None and value is not None and
                                      math.isclose(actual, value, rel_tol=0, abs_tol=1e-10)),
                  "summary statistic does not recompute from the 8 source rows")
            stats_public[stat] = actual
        _need(_count(row.get("n_valid"), "summary.n_valid") == len(known)
              and _count(row.get("n_unknown"), "summary.n_unknown") == len(unknown)
              and sorted(_alias(x) for x in row.get("unknown_case_ids", [])) == unknown,
              "unknown count/aliases differ from row-level nulls")
        arms_out.append({"dataset": DATASETS[dataset], "arm": arm, "metric": metric,
                         "n_valid": len(known), "n_unknown": len(unknown),
                         "unknown_case_ids": unknown, **stats_public})
    _need(seen == expected_arm_keys, "summary does not cover all 90 cells")

    contrast_src = source.get("contrasts")
    names = {"T_minus_F": ("T", "F"),
             "native_teacher_direct_minus_T": ("native_teacher_direct", "T"),
             "T_direct_minus_T": ("T_direct", "T"), "T_iou_minus_T": ("T_iou", "T"),
             "T_reliability_weighted_iou_minus_T": ("T_reliability_weighted_iou", "T"),
             "direct_minus_F": ("direct", "F"), "iou_minus_F": ("iou", "F"),
             "reliability_weighted_iou_minus_F": ("reliability_weighted_iou", "F")}
    _need(isinstance(contrast_src, list) and len(contrast_src) == 80,
          "expected all 80 paired contrasts")
    contrast_out, seen_c = [], set()
    for row in contrast_src:
        _need(isinstance(row, dict), "contrast row malformed")
        dataset, name, metric = row.get("dataset"), row.get("contrast"), row.get("metric")
        key = (dataset, name, metric)
        _need(dataset in DATASETS and name in names and metric in METRICS and key not in seen_c,
              "contrast grid has missing/duplicate/unknown key")
        seen_c.add(key)
        left, right = names[name]
        _need(row.get("comparison_arm") == left and row.get("baseline_arm") == right,
              "contrast direction mismatch")
        ids = sorted(r["case_id"] for r in rows["rows"] if r["arm"] == "F" and r["dataset"] == DATASETS[dataset])
        deltas, unknown = [], []
        for cid in ids:
            a, b = by[(cid, left)]["metrics"][metric], by[(cid, right)]["metrics"][metric]
            if a is None or b is None:
                unknown.append(cid)
            else:
                deltas.append((cid, float(a) - float(b)))
        tolerance = _num(row.get("neutral_tolerance"), "contrast.neutral_tolerance", low=0)
        pos = sorted(cid for cid, d in deltas if d > tolerance)
        neg = sorted(cid for cid, d in deltas if d < -tolerance)
        neu = sorted(cid for cid, d in deltas if abs(d) <= tolerance)
        _need(sorted(_alias(x) for x in row.get("unknown_case_ids", [])) == unknown
              and sorted(_alias(x) for x in row.get("positive_case_ids", [])) == pos
              and sorted(_alias(x) for x in row.get("negative_case_ids", [])) == neg
              and sorted(_alias(x) for x in row.get("neutral_case_ids", [])) == neu,
              "contrast positive/negative/neutral/unknown tails lost or changed")
        vals = [d for _, d in deltas]
        stat = _statistics(vals)
        expected = {"n_pairs": len(vals), "unknown_count": len(unknown),
                    "positive_count": len(pos), "negative_count": len(neg), "neutral_count": len(neu),
                    "delta_le_minus_5pp_count": sum(d <= -.05 for d in vals),
                    "delta_ge_plus_5pp_count": sum(d >= .05 for d in vals),
                    "mean_delta": stat["mean"], "mean_delta_pp": None if stat["mean"] is None else stat["mean"] * 100,
                    "median_delta_pp": None if stat["median"] is None else stat["median"] * 100,
                    "min_delta_pp": None if stat["min"] is None else stat["min"] * 100,
                    "max_delta_pp": None if stat["max"] is None else stat["max"] * 100}
        for field, value in expected.items():
            actual = row.get(field)
            if value is None:
                _need(actual is None, f"contrast {field} should remain null")
            elif isinstance(value, int):
                _need(_count(actual, f"contrast.{field}") == value, f"contrast {field} count mismatch")
            else:
                actual_num = _num(actual, f"contrast.{field}")
                _need(math.isclose(float(actual_num), float(value), rel_tol=0, abs_tol=1e-9),
                      f"contrast {field} differs from paired rows")
        ci = row.get("bootstrap_ci95_delta_pp")
        if ci is not None:
            _keys(ci, {"lower", "upper"}, "contrast bootstrap interval")
            ci = {k: _num(ci[k], f"ci.{k}", null_ok=True) for k in ("lower", "upper")}
        contrast_out.append({"dataset": DATASETS[dataset], "contrast": name,
                             "comparison_arm": left, "baseline_arm": right, "metric": metric,
                             **{k: _count(row[k], f"contrast.{k}") for k in
                                ("n_pairs", "unknown_count", "positive_count", "negative_count",
                                 "neutral_count", "delta_le_minus_5pp_count", "delta_ge_plus_5pp_count")},
                             "unknown_case_ids": unknown, "positive_case_ids": pos,
                             "negative_case_ids": neg, "neutral_case_ids": neu,
                             "neutral_tolerance": tolerance,
                             "mean_delta": _num(row.get("mean_delta"), "contrast.mean_delta", null_ok=True),
                             "mean_delta_pp": _num(row.get("mean_delta_pp"), "contrast.mean_delta_pp", null_ok=True),
                             "median_delta_pp": _num(row.get("median_delta_pp"), "contrast.median_delta_pp", null_ok=True),
                             "min_delta_pp": _num(row.get("min_delta_pp"), "contrast.min_delta_pp", null_ok=True),
                             "max_delta_pp": _num(row.get("max_delta_pp"), "contrast.max_delta_pp", null_ok=True),
                             "bootstrap_ci95_delta_pp": ci})
    _need(len(seen_c) == 80, "paired contrast grid incomplete")
    teacher_src = source.get("teacher_selection_and_coverage")
    _need(isinstance(teacher_src, list) and len(teacher_src) == 2, "two-domain teacher coverage missing")
    teacher_out = []
    seen_teacher_domains = set()
    for row in teacher_src:
        _need(row.get("dataset") in DATASETS, "teacher summary domain invalid")
        _need(row["dataset"] not in seen_teacher_domains, "duplicate teacher summary domain")
        seen_teacher_domains.add(row["dataset"])
        status = row.get("selection_status_counts")
        _need(isinstance(status, dict) and set(status) <= {
            "valid", "abstain", "invalid", "invalid_output", "output_truncated_at_cap",
            "out_of_clock_interval", "unknown", "missing_status"},
            "teacher selection status counts missing/unknown")
        teacher_out.append({"dataset": DATASETS[row["dataset"]],
                            "case_count": _count(row.get("case_count"), "teacher.case_count"),
                            "valid_uncapped_selection_count": _count(row.get("valid_uncapped_selection_count"), "teacher.valid"),
                            "selection_rate": _num(row.get("selection_rate"), "teacher.rate", low=0, high=1),
                            "selection_status_counts": {k: _count(v, f"teacher.status.{k}") for k, v in status.items()},
                            "mean_unlabeled_present_fraction_selected": _num(row.get("mean_unlabeled_present_fraction_selected"), "teacher.fraction", null_ok=True, low=0, high=1),
                            "mean_present_runs_selected": _num(row.get("mean_present_runs_selected"), "teacher.runs", null_ok=True, low=0),
                            "unknown_or_abstained_count": _count(row.get("unknown_or_abstained_count"), "teacher.unknown")})
    _need(seen_teacher_domains == set(DATASETS), "teacher summary must have both domains exactly once")
    projected = {"schema": "stvg-native-instance-public-postseal-summary-v1", "status": "pass",
                 "metric_scale": "0_to_1", "GT_read": True, "candidate_detection_recall": "not_measured",
                 "scope": "exploratory descriptive paired results; no pooled gate or selection",
                 "aggregation": {"bootstrap_replicates": 10000, "seed": 20261014,
                                 "unit": "paired source case", "scope": "separately within each domain",
                                 "historically_exposed_sources_per_domain": 4,
                                 "ci_interpretation": "exploratory descriptive only",
                                 "pooled_cross_domain_gate": False, "ci_used_for_arm_or_step_selection": False},
                 "case_count_by_dataset": {"HC": 4, "VidSTG": 4},
                 "arm_metric_summaries": sorted(arms_out, key=lambda x: (x["dataset"], ARMS.index(x["arm"]), METRICS.index(x["metric"]))),
                 "contrasts": sorted(contrast_out, key=lambda x: (x["dataset"], x["contrast"], METRICS.index(x["metric"]))),
                 "teacher_selection_and_coverage": sorted(teacher_out, key=lambda x: x["dataset"])}
    _need(projected["aggregation"] == {
        "bootstrap_replicates": 10000, "seed": 20261014, "unit": "paired source case",
        "scope": "separately within each domain", "historically_exposed_sources_per_domain": 4,
        "ci_interpretation": "exploratory descriptive only", "pooled_cross_domain_gate": False,
        "ci_used_for_arm_or_step_selection": False}, "public bootstrap contract drift")
    return projected


def _public_json_map() -> dict[str, Path]:
    paths = {
        "postseal_metrics.json": OUT / "postseal_metrics.json",
        "postseal_summary.json": OUT / "postseal_summary.json",
        "student_math_readback.json": OUT / "student_math_readback.json",
        "measured_costs.json": OUT / "measured_costs.json",
        "root_readbacks.json": OUT / "root_readbacks.json",
        "visual_review.json": OUT / "visual_review.json",
        "root_postseal_readback.json": OUT / "root_postseal_readback.json",
        "generation_configuration.json": OUT / "generation_configuration.json",
        "recovery_metadata.json": OUT / "recovery_metadata.json",
        "runtime_source_bindings.json": OUT / "runtime_source_bindings.json",
        "public_scalar_readback.json": OUT / "public_scalar_readback.json",
    }
    return paths


def _verify_contract_fixture() -> dict[str, Any]:
    """Exercise the real row/summary projection logic on synthetic records."""
    _need(_alias("hc2_000") == "HC01" and _alias("vidstg_003") == "V04", "alias contract")
    for bad in ("query_text", "/home/wwww/private", "vidstg_000"):
        try:
            _privacy_scan("synthetic.json", bad.encode())
        except AuditError:
            continue
        raise AuditError("privacy contract accepted a forbidden synthetic marker")

    binding_names = ("roster_sha256", "reference_lock_sha256", "direct_teacher_readout_sha256",
                     "tracking_barrier_sha256", "teacher_barrier_sha256", "student_barrier_sha256")
    global_bindings = {name: hashlib.sha256(name.encode()).hexdigest() for name in binding_names}
    barrier_sha = global_bindings["student_barrier_sha256"]
    source_rows = []
    arm_shift = {name: offset for name, offset in zip(ARMS, (0, .015, .02, -.012, .007,
                                                              -.006, .021, .010, -.009))}
    count_fields = ("source_grid_positions", "source_grid_supervised_position_count",
                    "source_grid_unsupervised_position_count", "supervised_anchor_count",
                    "unsupervised_position_count", "dense_gt_known_frame_count",
                    "source_grid_gt_known_frame_count", "off_grid_gt_known_frame_count",
                    "supervised_gt_known_frame_count", "unsupervised_gt_known_frame_count")
    coverage_fields = ("frame_count", "present_frame_count", "missing_frame_count",
                       "present_run_count", "fragmentation_gap_count", "interval_frame_count",
                       "interval_present_frame_count")
    geometry_fields = ("gt_known_frame_count", "teacher_tube_frame_overlap_count",
                       "gt_known_frames_with_present_mask_count",
                       "gt_known_frames_without_present_mask_count")
    for i, cid in enumerate(CASE_IDS):
        ds = "hc2" if cid.startswith("hc2_") else "vidstg"
        case_binding = {name: hashlib.sha256(f"{cid}:{name}".encode()).hexdigest()
                        for name in ("source_case_sha256", "student_artifact_sha256",
                                     "student_receipt_sha256", "teacher_record_sha256",
                                     "teacher_receipt_sha256")}
        case_binding["reference_lock_sha256"] = global_bindings["reference_lock_sha256"]
        for arm in ARMS:
            base = .18 + i * .035 + arm_shift[arm]
            metrics = {metric: min(.92, max(.05, base + j * .009))
                       for j, metric in enumerate(METRICS)}
            if cid == "hc2_000" and arm == "direct":
                metrics["m_vIoU"] = None
            source_rows.append({"case_id": cid, "dataset": ds, "arm": arm, "status": "scored",
                                "metrics": metrics, "candidate_detection_recall": "not_measured",
                                "support": {field: 4 for field in count_fields},
                                "coverage": {**{field: 3 for field in coverage_fields},
                                             "present_fraction": .75},
                                "selected_track_gt_geometry_coverage": {
                                    **{field: 2 for field in geometry_fields},
                                    "present_mask_fraction_of_gt_known_frames": .5},
                                "fit": {"status": "actual_fit" if arm in SPATIAL else "not_a_fit_arm",
                                        "optimizer_steps": 5 if arm in SPATIAL else 0,
                                        "final_redecode_calls": 1 if arm in SPATIAL else 0},
                                "metric_audit": (
                                    None if (i == 0 and arm == "direct")
                                    or (i == 7 and arm == "iou")
                                    else ({"official_independent_max_abs_error": 0.0,
                                           "native_missing_mask_independent_max_abs_error": 0.0,
                                           "scoring_path": "official_physical_teacher_missing_as_zero"}
                                          if arm == "native_teacher_direct"
                                          else {"official_independent_max_abs_error": 0.0,
                                                "interval_frame_bounds_half_open": [0.0, 4.0],
                                                "source_grid_frames": 4})),
                                "bindings": dict(case_binding)})
    rows_raw = {"schema": "stvg-native-instance-postseal-rows-v1", "status": "complete",
                "GT_read": True, "metric_scale": "0_to_1", "case_count": 8,
                "arm_count": 9, "prediction_barrier": {"sha256": barrier_sha},
                "bindings": global_bindings, "rows": source_rows}
    rows = _expected_rows(rows_raw)
    by = {(r["case_id"], r["arm"]): r for r in rows["rows"]}
    arm_summaries = []
    for ds in DATASETS:
        aliases = sorted(ALIAS[c] for c in CASE_IDS if c.startswith(ds + "_"))
        raw_ids = sorted(c for c in CASE_IDS if c.startswith(ds + "_"))
        for arm in ARMS:
            for metric in METRICS:
                vals = [(cid, by[(ALIAS[cid], arm)]["metrics"][metric]) for cid in raw_ids]
                known = [float(value) for _, value in vals if value is not None]
                stats = _statistics(known)
                arm_summaries.append({"dataset": ds, "arm": arm, "metric": metric,
                                      "n_valid": len(known),
                                      "n_unknown": sum(value is None for _, value in vals),
                                      "unknown_case_ids": [cid for cid, value in vals if value is None],
                                      **stats})
    pair_names = {"T_minus_F": ("T", "F"),
                  "native_teacher_direct_minus_T": ("native_teacher_direct", "T"),
                  "T_direct_minus_T": ("T_direct", "T"), "T_iou_minus_T": ("T_iou", "T"),
                  "T_reliability_weighted_iou_minus_T": ("T_reliability_weighted_iou", "T"),
                  "direct_minus_F": ("direct", "F"), "iou_minus_F": ("iou", "F"),
                  "reliability_weighted_iou_minus_F": ("reliability_weighted_iou", "F")}
    contrast_rows = []
    for ds in DATASETS:
        raw_ids = sorted(c for c in CASE_IDS if c.startswith(ds + "_"))
        for name, (comp, base_arm) in pair_names.items():
            for metric in METRICS:
                deltas, unknown = [], []
                for cid in raw_ids:
                    a = by[(ALIAS[cid], comp)]["metrics"][metric]
                    b = by[(ALIAS[cid], base_arm)]["metrics"][metric]
                    if a is None or b is None:
                        unknown.append(cid)
                    else:
                        deltas.append((cid, float(a) - float(b)))
                values = [v for _, v in deltas]
                tolerance = .0001
                positive = [cid for cid, v in deltas if v > tolerance]
                negative = [cid for cid, v in deltas if v < -tolerance]
                neutral = [cid for cid, v in deltas if abs(v) <= tolerance]
                stats = _statistics(values)
                contrast_rows.append({"dataset": ds, "contrast": name, "metric": metric,
                                      "comparison_arm": comp, "baseline_arm": base_arm,
                                      "unknown_case_ids": unknown, "positive_case_ids": positive,
                                      "negative_case_ids": negative, "neutral_case_ids": neutral,
                                      "n_pairs": len(values), "unknown_count": len(unknown),
                                      "positive_count": len(positive), "negative_count": len(negative),
                                      "neutral_count": len(neutral), "delta_le_minus_5pp_count": sum(v <= -.05 for v in values),
                                      "delta_ge_plus_5pp_count": sum(v >= .05 for v in values),
                                      "neutral_tolerance": tolerance, "mean_delta": stats["mean"],
                                      "mean_delta_pp": None if stats["mean"] is None else 100 * stats["mean"],
                                      "median_delta_pp": None if stats["median"] is None else 100 * stats["median"],
                                      "min_delta_pp": None if stats["min"] is None else 100 * stats["min"],
                                      "max_delta_pp": None if stats["max"] is None else 100 * stats["max"],
                                      "bootstrap_ci95_delta_pp": {"lower": -1.0, "upper": 1.0}})
    summary_raw = {"schema": "stvg-native-instance-postseal-summary-v1", "status": "complete",
                   "GT_read": True, "metric_scale": "0_to_1", "candidate_detection_recall": "not_measured",
                   "aggregation": {"bootstrap_replicates": 10000, "seed": 20261014,
                                   "unit": "paired source case", "pooled_cross_domain_gate": False,
                                   "ci_used_for_arm_or_step_selection": False},
                   "arm_metric_summaries": arm_summaries, "contrasts": contrast_rows,
                   "teacher_selection_and_coverage": [
                       {"dataset": ds, "case_count": 4, "valid_uncapped_selection_count": 3,
                        "selection_rate": .75, "selection_status_counts": {"valid": 3, "abstain": 1},
                        "mean_unlabeled_present_fraction_selected": .6,
                        "mean_present_runs_selected": 2.0, "unknown_or_abstained_count": 1}
                       for ds in DATASETS]}
    summary = _expected_summary(summary_raw, rows)
    _need(len(rows["rows"]) == 72 and len(summary["arm_metric_summaries"]) == 90
          and len(summary["contrasts"]) == 80, "synthetic projection matrix incomplete")

    def must_reject(expected: Any, altered: Any, label: str) -> None:
        try:
            _need(_canonical(expected) == _canonical(altered), label)
        except AuditError:
            return
        raise AuditError(f"synthetic mutation was accepted: {label}")

    altered_rows = json.loads(json.dumps(rows))
    altered_rows["rows"].pop()
    must_reject(rows, altered_rows, "omitted row")
    altered_rows = json.loads(json.dumps(rows))
    null_row = next(r for r in altered_rows["rows"] if r["case_id"] == "HC01" and r["arm"] == "direct")
    null_row["metrics"]["m_vIoU"] = 0.0
    must_reject(rows, altered_rows, "null converted to zero")
    altered_summary = json.loads(json.dumps(summary))
    signed = next(r for r in altered_summary["contrasts"]
                  if r["contrast"] == "T_minus_F" and r["metric"] == "m_vIoU" and r["dataset"] == "HC")
    signed["positive_case_ids"].pop()
    signed["positive_count"] -= 1
    must_reject(summary, altered_summary, "positive/sign tail omitted")
    altered_rows = json.loads(json.dumps(rows))
    altered_rows["bindings"]["student_barrier_sha256"] = "f" * 64
    must_reject(rows, altered_rows, "source binding corrupted")
    try:
        _keys({"a": 1}, {"a", "b"}, "synthetic omission")
    except AuditError:
        pass
    else:
        raise AuditError("strict schema contract accepted an omitted key")
    try:
        _sha("0" * 63, "synthetic SHA")
    except AuditError:
        pass
    else:
        raise AuditError("invalid binding digest was accepted")
    valid_pool_counts = {
        "gt_frames_in_canonical_map": 7,
        "gt_frames_outside_canonical_map": 1,
        "canonical_gt_frames_with_empty_filtered_pool": 2,
        "gt_frames_with_empty_filtered_pool": 3,
        "gt_frames_max_iou_ge_0_25": 4,
        "gt_frames_max_iou_ge_0_50": 2,
    }
    _validate_pool_case_count_invariants(valid_pool_counts, 8, "synthetic pool")
    for key, bad_value in (("gt_frames_with_empty_filtered_pool", 2),
                           ("gt_frames_max_iou_ge_0_50", 5)):
        invalid_pool_counts = dict(valid_pool_counts)
        invalid_pool_counts[key] = bad_value
        try:
            _validate_pool_case_count_invariants(invalid_pool_counts, 8, "mutated synthetic pool")
        except AuditError:
            continue
        raise AuditError(f"synthetic pool invariant accepted an invalid {key}")
    recovery_sha = hashlib.sha256(b"synthetic-recovery").hexdigest()
    recovery_projection = _recovery_entry(
        "synthetic_recovery", "CAPTURE",
        {"status": "complete", "scope": "synthetic", "GT_read": False,
         "new_GPU_calls": 0, "private_query_text": "must not be copied"},
        recovery_sha)
    _need("private_query_text" not in recovery_projection
          and recovery_projection["sha256"] == recovery_sha,
          "recovery projection copied a non-allowlisted field")
    try:
        _recovery_entry("synthetic_recovery", "CAPTURE",
                        {"status": "complete", "scope": "synthetic", "GT_read": False,
                         "new_reader_sha256": "bad"}, recovery_sha)
    except AuditError:
        pass
    else:
        raise AuditError("recovery projection accepted an invalid source hash")
    return {"status": "pass", "synthetic_only": True,
            "checks": ["72-row projection", "90 summary cells", "80 contrast cells",
                       "unknown/null preservation", "positive/negative/neutral tail retention",
                       "omission refusal", "null-to-zero refusal", "sign-tail omission refusal",
                       "source binding corruption refusal", "privacy marker refusal",
                       "pool empty-frame partition refusal", "nested IoU-threshold refusal",
                       "recovery allowlist and hash validation"]}


def _audit_manifest() -> dict[str, Any]:
    _need(MANIFEST.is_file() and STAGE.is_dir(), "public stage or manifest is absent")
    manifest = _read_json(MANIFEST)
    _need(isinstance(manifest, list) and manifest, "manifest must be a nonempty list")
    seen: set[str] = set()
    total_bytes = 0
    for row in manifest:
        _keys(row, {"path", "bytes", "sha256", "blob_sha", "mode", "type", "encoding"}, "manifest entry")
        rel = row["path"]
        _need(isinstance(rel, str) and "\\" not in rel, "manifest path must be POSIX-relative")
        p = PurePosixPath(rel)
        _need(not p.is_absolute() and all(part not in {"", ".", ".."} for part in p.parts),
              "absolute/traversing public path")
        _need(not any(part in DENIED_PATH_PARTS for part in p.parts), "manifest includes a private path")
        _need(rel not in seen, "duplicate manifest path")
        seen.add(rel)
        src = ROOT / rel
        staged = STAGE / rel
        _need(src.is_file() and not src.is_symlink() and staged.is_file() and not staged.is_symlink(),
              f"manifest object absent or symlinked: {rel}")
        data = src.read_bytes()
        staged_data = staged.read_bytes()
        expected_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        _need(data == staged_data, f"staged bytes differ from local allowlisted file: {rel}")
        _need(row["bytes"] == len(data) and row["sha256"] == _sha_bytes(data)
              and row["blob_sha"] == expected_blob and row["mode"] == "100644"
              and row["type"] == "blob", f"manifest digest/mode mismatch: {rel}")
        if rel.endswith((".json", ".md", ".pdf", ".png")):
            _privacy_scan(rel, data)
        total_bytes += len(data)
    stage_files = {p.relative_to(STAGE).as_posix() for p in STAGE.rglob("*") if p.is_file()}
    _need(stage_files == seen, "stage contains unmanifested or missing files")
    return {"count": len(seen), "bytes": total_bytes,
            "manifest_sha256": _sha_file(MANIFEST), "paths": sorted(seen)}


def _expected_manifest_paths(bindings: Mapping[str, Any]) -> set[str]:
    tree = ast.parse(EXPORTER.read_text(encoding="utf-8"), filename=EXPORTER.name)
    fixed_sources_node = None
    fixed_public_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if "fixed_sources" in names:
                fixed_sources_node = node.value
            if "FIXED_PUBLIC_SOURCE_PATHS" in names:
                fixed_public_node = node.value
    _need(isinstance(fixed_sources_node, ast.Name)
          and fixed_sources_node.id == "FIXED_PUBLIC_SOURCE_PATHS",
          "exporter fixed-source closure alias changed")
    _need(isinstance(fixed_public_node, ast.Call)
          and isinstance(fixed_public_node.func, ast.Name)
          and fixed_public_node.func.id == "frozenset"
          and len(fixed_public_node.args) == 1 and not fixed_public_node.keywords,
          "exporter fixed public source closure is not the expected frozen literal")
    try:
        fixed = ast.literal_eval(fixed_public_node.args[0])
    except Exception as exc:
        raise AuditError("exporter fixed-source closure is not a literal set") from exc
    _need(isinstance(fixed, set), "exporter has no fixed source closure")
    sources = bindings.get("bindings")
    _need(isinstance(sources, list), "source binding rows absent")
    paths = set(fixed)
    for item in sources:
        if item.get("published") is True:
            paths.add(item.get("source"))
    paths |= {REPORT.relative_to(ROOT).as_posix(), PROTOCOL.relative_to(ROOT).as_posix(),
              PDF.relative_to(ROOT).as_posix(), PNG.relative_to(ROOT).as_posix(),
              "docs/STVG_NATIVE_INSTANCE_POOL_COVERAGE_APPENDIX.md"}
    paths |= {OUT.relative_to(ROOT).as_posix() + "/" + name for name in (
        "postseal_metrics.json", "postseal_summary.json", "student_math_readback.json",
        "measured_costs.json", "root_readbacks.json", "visual_review.json",
        "root_postseal_readback.json", "generation_configuration.json",
        "independent_geometry_readback.json", "pool_candidate_coverage.json",
        "pool_coverage_readback.json", "recovery_metadata.json",
        "runtime_source_bindings.json", "public_scalar_readback.json")}
    _need(all(isinstance(p, str) and _path_is_public(p) for p in fixed),
          "fixed source closure contains an unapproved path")
    return paths


def _expected_runtime_bindings() -> list[dict[str, Any]]:
    runtime_names = ("TRACKING_RUNTIME.json", "TEACHER_RUNTIME.json", "STUDENT_RUNTIME.json")
    result: list[dict[str, Any]] = []
    for name in runtime_names:
        runtime = _read_json(BASE / name)
        if name == "TRACKING_RUNTIME.json":
            _need(runtime.get("status") == "locked" and runtime.get("GT_read") is False
                  and runtime.get("stage") == "tracking" and runtime.get("cases") == 8
                  and runtime.get("no_new_download") is True
                  and runtime.get("original_full_frame_clock") is True
                  and runtime.get("teacher_and_student_not_yet_launched") is True,
                  f"{name} is not the locked eight-case no-GT tracking runtime")
        else:
            _need(runtime.get("status") == "locked" and runtime.get("GT_read") is False
                  and runtime.get("count") == 8, f"{name} is not a frozen no-GT runtime")
        for group in ("pins", "protected"):
            mapping = runtime.get(group)
            _need(isinstance(mapping, dict), f"{name}.{group} is malformed")
            for rel, digest in sorted(mapping.items()):
                digest = _sha(digest, f"{name}.{group} digest")
                row = {"runtime": name, "binding": group, "sha256": digest,
                       "published": False, "source": "private_or_protected_binding",
                       "byte_exact": "runtime_declared_only"}
                if _path_is_public(rel):
                    path = ROOT / rel
                    _need(path.is_file() and not path.is_symlink() and _sha_file(path) == digest,
                          f"published runtime pin bytes differ: {rel}")
                    row.update({"source": rel, "published": True, "byte_exact": True})
                result.append(row)
    helper_rel = "scripts/stvg_native_instance_spatial_fit_v1.py"
    helper = ROOT / helper_rel
    _need(helper.is_file(), "pinned scientific helper missing")
    tree = ast.parse(helper.read_text(encoding="utf-8"), filename=helper_rel)
    source_map = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == "SOURCE_BINDINGS" for target in node.targets):
            source_map = ast.literal_eval(node.value)
            break
    _need(isinstance(source_map, dict) and bool(source_map),
          "scientific helper SOURCE_BINDINGS is not a literal map")
    for rel, value in sorted(source_map.items()):
        digest = value.get("sha256") if isinstance(value, dict) else value
        digest = _sha(digest, "embedded source dependency")
        row = {"runtime": "embedded_SOURCE_BINDINGS:" + helper_rel,
               "binding": "scientific_source", "sha256": digest,
               "published": False, "source": "private_or_protected_binding",
               "byte_exact": "source_declared_only"}
        if _path_is_public(rel):
            path = ROOT / rel
            _need(path.is_file() and not path.is_symlink() and _sha_file(path) == digest,
                  f"embedded published dependency differs: {rel}")
            row.update({"source": rel, "published": True, "byte_exact": True})
        result.append(row)

    bound_public_paths = {row["source"] for row in result if row["published"] is True}
    for rel in sorted(FIXED_PUBLIC_SOURCE_PATHS):
        _need(_path_is_public(rel), f"fixed public source is outside the allowlist: {rel}")
        path = ROOT / rel
        _need(path.is_file() and not path.is_symlink() and ROOT in path.resolve().parents,
              f"fixed public source is missing or unsafe: {rel}")
        digest = _sha_file(path)
        if rel in bound_public_paths:
            matches = [row for row in result
                       if row["published"] is True and row["source"] == rel]
            _need(all(row["sha256"] == digest and row["byte_exact"] is True for row in matches),
                  f"runtime-bound fixed source differs: {rel}")
            continue
        result.append({"runtime": "explicit_public_source_closure",
                       "binding": "fixed_source", "sha256": digest,
                       "published": True, "source": rel, "byte_exact": True})
        bound_public_paths.add(rel)

    final_execution_path = BASE / "ROOT_FINAL_EXECUTION_READBACK.json"
    final_execution = _read_json(final_execution_path)
    execution_sources = final_execution.get("source_bindings")
    _need(final_execution.get("status") == "pass" and final_execution.get("CPU_only") is True
          and isinstance(execution_sources, dict)
          and set(execution_sources) == FINAL_EXECUTION_SOURCES,
          "final execution readback does not have the pinned CPU-source inventory")
    for rel, declared in sorted(execution_sources.items()):
        digest = _sha(declared, "final execution source digest")
        path = ROOT / rel
        _need(_path_is_public(rel) and path.is_file() and not path.is_symlink()
              and ROOT in path.resolve().parents and _sha_file(path) == digest,
              f"final execution source bytes differ: {rel}")
        result.append({"runtime": "ROOT_FINAL_EXECUTION_READBACK.json",
                       "binding": "root_final_execution_source", "sha256": digest,
                       "published": True, "source": rel, "byte_exact": True})
    return result


def _audit_source_readback(manifest_info: Mapping[str, Any], binding_doc: Mapping[str, Any]) -> dict[str, Any]:
    doc = _read_json(SOURCE_RECEIPT)
    expected_keys = {"schema", "status", "files", "bytes", "manifest_sha256",
                     "public_source_binding_count", "source_bindings_sha256",
                     "postseal_projection_sha256", "summary_projection_sha256",
                     "cost_projection_sha256", "geometry_projection_sha256",
                     "pool_coverage_projection_sha256", "pool_coverage_readback_projection_sha256",
                     "pool_coverage_appendix_sha256", "report_sha256", "pdf_sha256", "png_sha256",
                     "raw_prediction_barrier_staged", "GT_coordinates_or_annotations_staged",
                     "raw_teacher_output_or_masks_staged", "student_checkpoints_staged",
                     "RGB_or_video_staged", "queries_staged", "push_performed"}
    _keys(doc, expected_keys, "public source-binding readback")
    _need(doc["status"] == "pass" and doc["schema"] == "stvg-native-instance-public-source-readback-v1"
          and doc["files"] == manifest_info["count"] and doc["bytes"] == manifest_info["bytes"]
          and doc["manifest_sha256"] == manifest_info["manifest_sha256"],
          "source readback does not match current stage manifest")
    for flag in ("raw_prediction_barrier_staged", "GT_coordinates_or_annotations_staged",
                 "raw_teacher_output_or_masks_staged", "student_checkpoints_staged",
                 "RGB_or_video_staged", "queries_staged", "push_performed"):
        _need(doc[flag] is False, f"source readback privacy/scope flag not false: {flag}")
    for field in ("source_bindings_sha256", "postseal_projection_sha256", "summary_projection_sha256",
                  "cost_projection_sha256", "geometry_projection_sha256",
                  "pool_coverage_projection_sha256", "pool_coverage_readback_projection_sha256",
                  "pool_coverage_appendix_sha256", "report_sha256", "pdf_sha256", "png_sha256"):
        _sha(doc[field], f"source readback.{field}")
    exact_files = {
        "source_bindings_sha256": OUT / "runtime_source_bindings.json",
        "postseal_projection_sha256": OUT / "postseal_metrics.json",
        "summary_projection_sha256": OUT / "postseal_summary.json",
        "cost_projection_sha256": OUT / "measured_costs.json",
        "geometry_projection_sha256": OUT / "independent_geometry_readback.json",
        "pool_coverage_projection_sha256": OUT / "pool_candidate_coverage.json",
        "pool_coverage_readback_projection_sha256": OUT / "pool_coverage_readback.json",
        "pool_coverage_appendix_sha256": ROOT / "docs/STVG_NATIVE_INSTANCE_POOL_COVERAGE_APPENDIX.md",
        "report_sha256": REPORT, "pdf_sha256": PDF, "png_sha256": PNG,
    }
    for field, path in exact_files.items():
        _need(path.is_file() and doc[field] == _sha_file(path),
              f"source readback digest does not match {field}")
    _need(doc["public_source_binding_count"] == binding_doc.get("binding_count"),
          "source receipt binding count differs from public source-binding projection")
    return {"schema": doc["schema"], "status": "pass", "files": doc["files"],
            "bytes": doc["bytes"], "manifest_sha256": doc["manifest_sha256"],
            "source_bindings_sha256": doc["source_bindings_sha256"],
            "private_payload_flags_all_false": True, "push_performed": False}


def _expected_geometry_readback() -> dict[str, Any]:
    source_path = BASE / "ROOT_INDEPENDENT_GEOMETRY_READBACK.json"
    reader_path = ROOT / "scripts/review_stvg_native_instance_geometry_v1.py"
    source = _read_json(source_path)
    expected = {
        "schema": "stvg-native-instance-independent-geometry-readback-public-v1",
        "status": "pass",
        "scope": "all selected saved RLE geometry and fixed local reliability; no semantic identity claim",
        "case_count": 8, "checks": 31842, "GT_read": False, "CPU_only": True,
        "model_replayed": False,
        "reference_lock_sha256": _sha_file(BASE / "REFERENCE_LOCK.json"),
        "direct_readout_sha256": _sha_file(BASE / "DIRECT_TEACHER_READOUT.json"),
        "reader_sha256": _sha_file(reader_path), "per_case_records_published": False,
    }
    _need(source.get("status") == "pass" and source.get("case_count") == 8
          and source.get("checks") == 31842 and source.get("GT_read") is False
          and source.get("CPU_only") is True and source.get("model_replayed") is False
          and source.get("reference_lock_sha256") == expected["reference_lock_sha256"]
          and source.get("direct_readout_sha256") == expected["direct_readout_sha256"]
          and source.get("reader_sha256") == expected["reader_sha256"],
          "independent geometry receipt is not complete or source-bound")
    return expected


def _expected_postseal_receipt(rows_path: Path, summary_path: Path,
                               barrier_sha: str) -> dict[str, Any]:
    doc = _read_json(BASE / "ROOT_POSTSEAL_READBACK.json")
    reader = ROOT / "scripts/review_stvg_native_instance_postseal_v1.py"
    completion_path = BASE / "CPU_COMPLETION.json"
    required = {"schema", "status", "GT_read", "raw_GT_read_by_this_reader", "CPU_only",
                "prediction_barrier_sha256", "rows_sha256", "summary_sha256",
                "completion_sha256", "case_count", "row_count", "arm_summary_count",
                "contrast_count", "checks_passed", "checks_failed",
                "max_metric_scalar_abs_error", "all_case_source_bindings_match",
                "all_tails_and_unknown_membership_match", "model_replayed",
                "verification_scope", "reader_sha256", "time"}
    _keys(doc, required, "root postseal receipt")
    _need(doc["schema"] == "stvg-native-instance-root-postseal-readback-v1"
          and doc["status"] == "pass" and doc["GT_read"] is True
          and doc["raw_GT_read_by_this_reader"] is False and doc["CPU_only"] is True
          and doc["model_replayed"] is False
          and doc["prediction_barrier_sha256"] == barrier_sha
          and doc["rows_sha256"] == _sha_file(rows_path)
          and doc["summary_sha256"] == _sha_file(summary_path)
          and doc["completion_sha256"] == _sha_file(completion_path)
          and doc["reader_sha256"] == _sha_file(reader)
          and _count(doc["case_count"], "postseal.case_count") == 8
          and _count(doc["row_count"], "postseal.row_count") == 72
          and _count(doc["arm_summary_count"], "postseal.arm_summary_count") == 90
          and _count(doc["contrast_count"], "postseal.contrast_count") == 80
          and _count(doc["checks_failed"], "postseal.checks_failed") == 0
          and doc["all_case_source_bindings_match"] is True
          and doc["all_tails_and_unknown_membership_match"] is True
          and float(_num(doc["max_metric_scalar_abs_error"], "postseal.max_error")) <= 2e-8,
          "root postseal receipt does not pass current complete row binding")
    _num(doc["time"], "postseal.time", low=0)
    return {"schema": doc["schema"], "status": "pass", "GT_read": True,
            "raw_GT_read_by_this_reader": False, "CPU_only": True, "model_replayed": False,
            "prediction_barrier_sha256": barrier_sha, "rows_sha256": _sha_file(rows_path),
            "summary_sha256": _sha_file(summary_path), "completion_sha256": _sha_file(completion_path),
            "case_count": 8, "row_count": 72, "arm_summary_count": 90,
            "contrast_count": 80,
            "checks_passed": _count(doc["checks_passed"], "postseal.checks_passed", low=1),
            "checks_failed": 0,
            "max_metric_scalar_abs_error": float(doc["max_metric_scalar_abs_error"]),
            "all_case_source_bindings_match": True,
            "all_tails_and_unknown_membership_match": True,
            "verification_scope": "independent scalar aggregation, exact bindings, support partitions, and unknown-tail membership; geometric metrics were checked by the sealed scorer",
            "reader_sha256": _sha_file(reader)}


def _expected_visual_readback() -> dict[str, Any]:
    doc = _read_json(BASE / "ROOT_VISUAL_REVIEW.json")
    required = {"schema", "status", "report_sha256", "pdf_sha256", "png_sha256", "case_count",
                "private_contact_count", "all_private_contacts_reviewed", "pdf_pages_reviewed",
                "all_pdf_pages_reviewed", "raw_media_public"}
    _keys(doc, required, "root visual review receipt")
    _need(doc["status"] == "pass" and doc["case_count"] == 8 and doc["private_contact_count"] == 8
          and doc["all_private_contacts_reviewed"] is True and doc["all_pdf_pages_reviewed"] is True
          and doc["raw_media_public"] is False and doc["report_sha256"] == _sha_file(REPORT)
          and doc["pdf_sha256"] == _sha_file(PDF) and doc["png_sha256"] == _sha_file(PNG),
          "visual receipt does not bind the current report and figures")
    _need(_count(doc["pdf_pages_reviewed"], "visual.pdf_pages_reviewed") >= 1,
          "visual receipt says no PDF page was reviewed")
    return {"schema": doc["schema"], "status": "pass", "case_count": 8,
            "private_contact_count": 8, "all_private_contacts_reviewed": True,
            "pdf_pages_reviewed": doc["pdf_pages_reviewed"], "all_pdf_pages_reviewed": True,
            "raw_media_public": False, "report_sha256": _sha_file(REPORT),
            "pdf_sha256": _sha_file(PDF), "png_sha256": _sha_file(PNG)}


def _expected_generation_readback() -> dict[str, Any]:
    doc = _read_json(BASE / "ROOT_GENERATION_CONFIGURATION_READBACK.json")
    _need(doc.get("status") == "pass" and doc.get("GT_read") is False
          and doc.get("CPU_only") is True, "generation readback is incomplete")
    config, overrides = doc.get("generation_configuration"), doc.get("explicit_generate_overrides")
    _need(isinstance(config, dict) and isinstance(overrides, dict)
          and config.get("attn_implementation") == "flash_attention_2"
          and config.get("do_sample") is True and config.get("top_k") == 1
          and config.get("top_p") == 0.001
          and config.get("temperature", config.get("temp")) == 0.1
          and overrides.get("max_new_tokens") == 2048
          and doc.get("attention_backend_actual") == "sdpa"
          and doc.get("teacher_manual_seed_set") is False,
          "generation config is not the locked R1 setup")
    return {"schema": doc.get("schema"), "status": "pass", "GT_read": False, "CPU_only": True,
            "official_generation_configuration_sha256": _sha(doc.get("official_generation_configuration_sha256"),
                                                                "generation configuration"),
            "generation_configuration": {"attn_implementation": "flash_attention_2", "do_sample": True,
                                          "top_k": 1, "top_p": 0.001, "temperature": 0.1},
            "explicit_generate_overrides": {"max_new_tokens": 2048},
            "attention_backend_actual": "sdpa", "teacher_manual_seed_set": False,
            "interpretation": "R1 generation used the recorded SDPA backend; this is distinct from the detector QKV compatibility path."}


def _validate_pool_case_count_invariants(counts: Mapping[str, int], known: int,
                                         label: str) -> None:
    """Validate denominator partitions and nested IoU-threshold counts."""
    _need(known > 0, f"{label}: empty GT denominator")
    _need(counts["gt_frames_in_canonical_map"]
          + counts["gt_frames_outside_canonical_map"] == known,
          f"{label}: canonical/outside frame partition mismatch")
    empty = counts["gt_frames_with_empty_filtered_pool"]
    _need(counts["canonical_gt_frames_with_empty_filtered_pool"] <= empty <= known
          and empty == counts["canonical_gt_frames_with_empty_filtered_pool"]
          + counts["gt_frames_outside_canonical_map"],
          f"{label}: empty-pool denominator accounting mismatch")
    _need(counts["gt_frames_max_iou_ge_0_25"] <= known
          and counts["gt_frames_max_iou_ge_0_50"] <= counts["gt_frames_max_iou_ge_0_25"],
          f"{label}: IoU threshold counts are not nested")


def _recovery_entry(recovery_id: str, kind: str, source: Mapping[str, Any],
                    digest: str) -> dict[str, Any]:
    allowed_fields = {
        "status", "scope", "GT_read", "CPU_only", "new_GPU_calls",
        "scientific_predictions_changed", "new_reader_sha256", "root_math_sha256",
        "root_cost_sha256", "old_runtime_sha256", "corrected_scorer_sha256",
        "score_rows_sha256", "root_postseal_sha256",
        "synthetic_cropped_absolute_clock_contract", "invalid_fps_rejected",
    }
    hash_fields = {"new_reader_sha256", "root_math_sha256", "root_cost_sha256",
                   "old_runtime_sha256", "corrected_scorer_sha256",
                   "score_rows_sha256", "root_postseal_sha256"}
    bool_fields = {"GT_read", "CPU_only", "scientific_predictions_changed",
                   "synthetic_cropped_absolute_clock_contract", "invalid_fps_rejected"}
    values: dict[str, Any] = {}
    for field in allowed_fields:
        if field not in source:
            continue
        value = source[field]
        if field in hash_fields:
            value = None if value is None else _sha(value, f"{recovery_id}.{kind}.{field}")
        elif field in bool_fields:
            _need(isinstance(value, bool), f"{recovery_id}.{kind}.{field} must be boolean")
        elif field == "new_GPU_calls":
            value = _count(value, f"{recovery_id}.{kind}.{field}")
        elif field in {"status", "scope"}:
            _need(isinstance(value, str) and len(value) <= 500,
                  f"{recovery_id}.{kind}.{field} must be bounded text")
        values[field] = value
    _need({"status", "scope", "GT_read"} <= set(values),
          f"{recovery_id}.{kind} lacks required recovery declarations")
    return {"sha256": digest, **values}


def _expected_recovery_readback() -> dict[str, Any]:
    capture = BASE / "recovery/teacher_gpu_storage002/CAPTURE.json"
    _need(capture.is_file(), "approved storage-guard capture receipt is missing")
    # Hash the initial guard capture only; never parse or publish its raw contents.
    recovery_ids = ("postseal_roster_scope002", "postseal_numpy_dtype003",
                    "postseal_numpy_rng004", "postseal_native_clock005")
    records = [{"recovery_id": "teacher_gpu_storage002",
                "event": "initial_teacher_resource_guard_stop",
                "capture_sha256": _sha_file(capture),
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
        projected: dict[str, Any] = {"recovery_id": recovery_id}
        for kind in ("CAPTURE", "RESOLUTION"):
            path = BASE / "recovery" / recovery_id / f"{kind}.json"
            _need(path.is_file(), f"approved recovery receipt is missing: {recovery_id}/{kind}")
            projected[kind.lower()] = _recovery_entry(
                recovery_id, kind, _read_json(path), _sha_file(path))
        records.append(projected)
    return {"schema": "stvg-native-instance-public-recovery-v2", "status": "pass",
            "count": len(records), "records": records}


def _expected_math_projection(barrier_sha: str, row_bindings: Mapping[str, Any]) -> dict[str, Any]:
    src = _read_json(BASE / "ROOT_STUDENT_MATH_READBACK.json")
    _need(src.get("schema") == "stvg-native-instance-independent-student-math-readback-v1"
          and src.get("status") == "pass" and src.get("GT_read") is False
          and src.get("CPU_only") is True and src.get("GPU_initialized") is False
          and src.get("model_replayed_by_reader") is False and src.get("case_count") == 8
          and src.get("student_barrier_sha256") == barrier_sha
          and src.get("reference_lock_sha256") == row_bindings["reference_lock_sha256"],
          "student-math readback is incomplete or unbound")
    raw = src.get("case_checks")
    _need(isinstance(raw, dict) and set(raw) == set(CASE_IDS),
          "student-math readback does not cover the eight source cases")
    cases: dict[str, Any] = {}
    fit_total = steps_total = readout_total = 0
    for cid, row in raw.items():
        dataset = "hc2" if cid.startswith("hc2_") else "vidstg"
        _need(isinstance(row, dict) and row.get("dataset") == dataset,
              "student-math case/domain binding mismatch")
        fit_count = _count(row.get("fit_count"), "math.fit_count")
        step_count = _count(row.get("optimizer_steps"), "math.optimizer_steps")
        readouts = _count(row.get("saved_full_model_readout_count"), "math.readouts")
        temporal_readouts = _count(row.get("saved_temporal_logits_preserved_count"), "math.temporal")
        _need(temporal_readouts == readouts, "temporal readout count mismatch")
        fits_src = row.get("fits")
        _need(isinstance(fits_src, dict) and set(fits_src) <= set(SPATIAL),
              "student-math fit catalog invalid")
        fits: dict[str, Any] = {}
        for method, check in fits_src.items():
            _need(isinstance(check, dict) and check.get("steps") == 5,
                  "student fit must preserve five declared rounds")
            public: dict[str, Any] = {"steps": 5,
                                     "optimizer_steps": _count(check.get("optimizer_steps"),
                                                               "math.fit.optimizer_steps")}
            if method == "direct":
                flags = ("direct_loss_independently_recomputed",
                         "adam_moment_and_parameter_update_reconstructed",
                         "decoder_autograd_gradient_independently_recomputed",
                         "saved_full_model_readout_equals_cached_decoder", "model_replayed_by_reader")
                for key in flags:
                    _need(isinstance(check.get(key), bool), f"math.{key} must be bool")
                    public[key] = check[key]
                for key in ("max_direct_loss_abs_error", "max_direct_adam_parameter_or_state_error",
                            "max_direct_adam_second_moment_consistency_error"):
                    public[key] = _num(check.get(key), f"math.{key}", low=0)
            else:
                errors = check.get("max_abs_errors")
                _keys(errors, {"mean_abs", "sample_abs", "reward_abs", "weight_abs", "loss_abs",
                               "adam_parameter_abs", "adam_state_abs"}, "OPD math error matrix")
                public["max_abs_errors"] = {k: _num(v, f"math.{k}", low=0)
                                            for k, v in sorted(errors.items())}
                _need(check.get("reward_semantics") in
                      {"original_iou", "raw_iou_times_r_t_before_fixed_tau_softmax"},
                      "OPD reward semantics changed")
                public["reward_semantics"] = check["reward_semantics"]
                _need(isinstance(check.get("cross_anchor_reliability_normalization"), bool),
                      "cross-anchor normalization flag missing")
                public["cross_anchor_reliability_normalization"] = check[
                    "cross_anchor_reliability_normalization"]
            fits[method] = public
        _need(fit_count == len(fits), "math fit count differs from the listed method checks")
        cases[ALIAS[cid]] = {"dataset": DATASETS[dataset], "fit_count": fit_count,
                             "optimizer_steps": step_count,
                             "saved_full_model_readout_count": readouts,
                             "saved_temporal_logits_preserved_count": temporal_readouts,
                             "source_state_hash_consistent": row.get("source_state_hash_consistent"),
                             "saved_source_cold_receipt_self_consistent": row.get(
                                 "saved_source_cold_receipt_self_consistent"),
                             "model_replayed_by_reader": row.get("model_replayed_by_reader"),
                             "fits": fits}
        for key in ("source_state_hash_consistent", "saved_source_cold_receipt_self_consistent",
                    "model_replayed_by_reader"):
            _need(isinstance(cases[ALIAS[cid]][key], bool), f"math case {key} must be bool")
        fit_total += fit_count
        steps_total += step_count
        readout_total += readouts
    _need(fit_total == src.get("actual_fit_count") and steps_total == src.get("saved_optimizer_steps")
          and readout_total == src.get("saved_full_model_readouts"), "student-math totals do not reconcile")
    hashes = {k: _sha(src.get(k), f"math.{k}") for k in
              ("runtime_sha256", "student_barrier_sha256", "reference_lock_sha256", "roster_sha256")}
    return {"schema": src["schema"], "status": "pass", "case_count": 8,
            "actual_fit_count": fit_total, "saved_optimizer_steps": steps_total,
            "saved_full_model_readouts": readout_total,
            "saved_temporal_logits_preserved": readout_total,
            "source_capture_calls": _count(src.get("source_capture_calls"), "math.source_capture_calls"),
            "GT_read": False, "GPU_initialized": False, "CPU_only": True,
            "model_replayed_by_reader": False, **hashes,
            "cases": {k: cases[k] for k in sorted(cases)}}


def _expected_cost_projection(barrier_sha: str, math_public: Mapping[str, Any],
                              rows_public: Mapping[str, Any]) -> dict[str, Any]:
    src = _read_json(BASE / "ROOT_COST_READBACK.json")
    _need(src.get("schema") == "stvg-native-instance-cost-readback-v1"
          and src.get("status") == "complete" and src.get("GT_read") is False
          and src.get("count") == 8 and src.get("student_barrier_sha256") == barrier_sha
          and src.get("runtime_sha256") == math_public["runtime_sha256"],
          "cost readback is incomplete or unbound")
    scored_by_alias = {r["case_id"]: r for r in rows_public["rows"] if r["arm"] == "T"}
    records = src.get("records")
    _need(isinstance(records, list) and len(records) == 8, "cost reader must cover all eight cases")
    seen: set[str] = set()
    out = []
    for row in records:
        cid = row.get("case_id")
        alias = _alias(cid)
        dataset = "hc2" if cid.startswith("hc2_") else "vidstg"
        _need(row.get("dataset") == dataset and alias not in seen,
              "duplicate/mismatched cost record identity")
        seen.add(alias)
        baseline = scored_by_alias[alias]
        _need(row.get("source_case_sha256") == baseline["bindings"]["source_case_sha256"]
              and row.get("student_artifact_sha256") == baseline["bindings"]["student_artifact_sha256"],
              "cost row does not bind the scored source/student artifact")
        capture = _num(row.get("source_capture_seconds"), "cost.source_capture_seconds",
                       null_ok=True, low=0)
        costs_src = row.get("algorithm_costs")
        _keys(costs_src, set(SPATIAL), "algorithm cost catalog")
        costs = {}
        for name in SPATIAL:
            item = costs_src[name]
            _need(isinstance(item, dict), "algorithm cost record malformed")
            seconds = _num(item.get("seconds"), f"cost.{name}.seconds", null_ok=True, low=0)
            steps = _count(item.get("optimizer_steps"), f"cost.{name}.optimizer_steps")
            redecode = _count(item.get("final_redecode_calls"), f"cost.{name}.final_redecode_calls")
            status, fit_present = item.get("status"), item.get("fit_present")
            _need(status in {"exact_no_update", "actual_updated_full_redecode"}
                  and isinstance(fit_present, bool), "unexpected algorithm cost status")
            if status == "exact_no_update":
                _need(not fit_present and seconds is None and steps == 0 and redecode == 0,
                      "exact no-update must preserve null seconds and zero work")
            costs[name] = {"seconds": seconds, "optimizer_steps": steps,
                           "final_redecode_calls": redecode, "status": status,
                           "fit_present": fit_present,
                           "elapsed_seconds_available": seconds is not None}
        out.append({"case_id": alias, "dataset": DATASETS[dataset],
                    "source_case_sha256": _sha(row.get("source_case_sha256"), "cost source case"),
                    "student_artifact_sha256": _sha(row.get("student_artifact_sha256"), "cost student artifact"),
                    "source_capture_seconds": capture,
                    "source_capture_seconds_available": capture is not None,
                    "algorithm_costs": costs})
    _need(seen == set(ALIAS.values()), "cost readback omits a case")
    return {"schema": "stvg-native-instance-public-costs-v1", "status": "pass",
            "GT_read": False, "case_count": 8,
            "student_source_capture_seconds_are_measured": all(
                row["source_capture_seconds_available"] for row in out),
            "algorithm_no_update_seconds_remain_null": True,
            "records": sorted(out, key=lambda x: x["case_id"])}


def _expected_receipt_cost_details(rows_public: Mapping[str, Any],
                                   costs_public: Mapping[str, Any]) -> list[dict[str, Any]]:
    cost_by_case = {r["case_id"]: r for r in costs_public["records"]}
    output = []
    for cid in CASE_IDS:
        alias = ALIAS[cid]
        tracking_path = BASE / "TRACKING" / cid / "ACCEPTED.json"
        tracking = _read_json(tracking_path)
        _need(tracking.get("status") == "pass" and tracking.get("GT_read") is False,
              "tracking receipt is not complete and GT-free")
        reused = tracking.get("reused")
        _need(isinstance(reused, bool), "tracking reused flag missing")
        track_seconds = _num(tracking.get("actual_tracking_wall_s"), "tracking wall time",
                             null_ok=True, low=0)
        teacher_path = BASE / "teacher" / f"{cid}.receipt.json"
        teacher = _read_json(teacher_path)
        baseline = next(r for r in rows_public["rows"] if r["case_id"] == alias and r["arm"] == "F")
        dataset = "hc2" if cid.startswith("hc2_") else "vidstg"
        _need(teacher.get("schema") == "stvg-native-instance-teacher-receipt-v1"
              and teacher.get("status") == "complete" and teacher.get("GT_read") is False
              and teacher.get("query_text_in_receipt") is False and teacher.get("dataset") == dataset
              and teacher.get("source_case_sha256") == baseline["bindings"]["source_case_sha256"]
              and teacher.get("tracking_acceptance_sha256") == _sha_file(tracking_path)
              and baseline["bindings"]["teacher_receipt_sha256"] == _sha_file(teacher_path),
              "teacher timing receipt is not bound to the fixed source case")
        costs = teacher.get("costs")
        _need(isinstance(costs, dict), "teacher cost fields absent")
        times = {name: _num(costs.get(name), f"teacher costs.{name}", null_ok=True, low=0)
                 for name in ("worker_wall_s", "model_load_s", "model_preprocess_s", "model_generate_s")}
        output.append({"case_id": alias, "dataset": DATASETS[dataset],
                       "tracking_history": "reused_historical" if reused else "current_run",
                       "tracking_wall_seconds": track_seconds,
                       "tracking_wall_seconds_available": track_seconds is not None,
                       "teacher_receipt_sha256": _sha_file(teacher_path),
                       "teacher_seconds": times,
                       "teacher_seconds_available": {k: v is not None for k, v in times.items()},
                       "source_capture_seconds": cost_by_case[alias]["source_capture_seconds"],
                       "spatial_fit_and_full_redecode_seconds": cost_by_case[alias]["algorithm_costs"]})
    return sorted(output, key=lambda x: x["case_id"])


def _expected_root_summaries() -> dict[str, Any]:
    names = {
        "ROOT_STUDENT_MATH_READBACK.json": BASE / "ROOT_STUDENT_MATH_READBACK.json",
        "ROOT_COST_READBACK.json": BASE / "ROOT_COST_READBACK.json",
        "CPU_COMPLETION.json": BASE / "CPU_COMPLETION.json",
        "ROOT_POSTSEAL_READBACK.json": BASE / "ROOT_POSTSEAL_READBACK.json",
        "ROOT_VISUAL_REVIEW.json": BASE / "ROOT_VISUAL_REVIEW.json",
        "ROOT_GENERATION_CONFIGURATION_READBACK.json": BASE / "ROOT_GENERATION_CONFIGURATION_READBACK.json",
        "ROOT_INDEPENDENT_GEOMETRY_READBACK.json": BASE / "ROOT_INDEPENDENT_GEOMETRY_READBACK.json",
        "ROOT_POOL_COVERAGE_READBACK.json": BASE / "ROOT_POOL_COVERAGE_READBACK.json",
    }
    safe = ("schema", "case_count", "row_count", "arm_count", "actual_fit_count",
            "saved_optimizer_steps", "saved_full_model_readouts", "source_capture_calls",
            "checks", "checks_passed", "checks_failed", "fit_count", "expected_fit_count",
            "arm_summary_count", "contrast_count",
            "optimizer_steps", "CPU_only", "GPU_initialized", "model_replayed_by_reader",
            "model_replayed", "raw_GT_read_by_this_reader", "all_case_source_bindings_match",
            "all_tails_and_unknown_membership_match", "GT_read", "scoring_status",
            "experiment_complete")
    hashes = ("student_barrier_sha256", "runtime_sha256", "reference_lock_sha256",
              "roster_sha256", "rows_sha256", "summary_sha256", "prediction_barrier_sha256",
              "direct_readout_sha256", "completion_sha256", "reader_sha256", "coverage_sha256",
              "appendix_sha256")
    rows = []
    for name, path in names.items():
        doc = _read_json(path)
        status = doc.get("status")
        _need(status in {"pass", "complete", "pending_root_review_visualization_publication"},
              f"root summary source is not complete: {name}")
        item = {"name": name, "status": status}
        for field in safe:
            if field not in doc:
                continue
            value = doc[field]
            if isinstance(value, bool) or value is None:
                item[field] = value
            elif isinstance(value, int):
                item[field] = _count(value, f"{name}.{field}")
            elif isinstance(value, str) and field == "schema" and value.startswith("stvg-"):
                item[field] = value[:100]
            elif isinstance(value, str) and field == "scoring_status" and value == "complete":
                item[field] = value
            else:
                raise AuditError(f"{name}.{field} is outside the scalar readback allowlist")
        for field in hashes:
            if field in doc:
                item[field] = _sha(doc[field], f"{name}.{field}")
        rows.append(item)
    return {"schema": "stvg-native-instance-public-root-readbacks-v1",
            "status": "pass", "receipts": rows}


def _expected_pool_coverage(barrier_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute the public candidate-pool scalar projection from its sealed rows."""
    src_path = BASE / "POOL_CANDIDATE_COVERAGE_POSTSEAL.json"
    appendix = ROOT / "docs/STVG_NATIVE_INSTANCE_POOL_COVERAGE_APPENDIX.md"
    source = _read_json(src_path)
    _need(source.get("schema") == "stvg-native-instance-filtered-pool-coverage-v1"
          and source.get("status") == "completed_cpu_postseal"
          and source.get("GT_read") is True
          and source.get("GT_read_after_complete_prediction_seal") is True
          and source.get("thresholds") == [0.25, 0.5], "pool-coverage source not sealed post-prediction")
    text = {
        "coverage_estimand": "per annotated physical frame, maximum bbox IoU over all nonempty filtered tracker masks",
        "denominator_definition": "all physical frames returned by the official sealed-source annotation reader; frames outside the canonical tracker clock count as zero",
        "candidate_policy": "complete all-category filtered tracking pool; no category metadata filtering; no GT-based candidate selection",
    }
    for key, value in text.items():
        _need(source.get(key) == value, f"pool coverage {key} changed")
    boundaries = {"raw_yolo_detector_recall": "not_measured", "semantic_identity_correctness": "not_measured",
                  "oracle_fit_or_reference_change": False, "prediction_selection_or_change": False}
    _need(source.get("claim_boundaries") == boundaries, "pool coverage overstates unmeasured claims")
    redaction_keys = ("private_case_ids_exported", "source_paths_exported", "queries_or_teacher_text_exported",
                      "boxes_or_raw_annotation_labels_exported", "images_or_masks_exported")
    _need(all(source.get(k) is False for k in redaction_keys), "pool source declares a private export")

    tool = ROOT / "scripts/score_stvg_native_instance_pool_coverage_v1.py"
    scorer = ROOT / "scripts/score_stvg_native_instance_postseal_v1.py"
    wrapper = ROOT / "scripts/review_stvg_sparse_r1_v2.py"
    truth_files = {"review_stvg_sparse_r1_v2.py": ROOT / "scripts/review_stvg_sparse_r1_v2.py",
                   "score_stvg_opd_p1_v1.py": ROOT / "scripts/score_stvg_opd_p1_v1.py",
                   "diagnose_tastvg_pipeline_cpu_v1.py": ROOT / "scripts/diagnose_tastvg_pipeline_cpu_v1.py"}
    mask_src = ROOT / "external/stvg_runtime/pycocotools/mask.py"
    mask_ext = ROOT / "external/stvg_runtime/pycocotools/_mask.cpython-311-x86_64-linux-gnu.so"
    bind = source.get("bindings")
    bind_keys = {"global_prediction_barrier_sha256", "roster_lock_sha256", "tracking_global_barrier_sha256",
                 "postseal_scorer_sha256", "official_truth_reader_sha256",
                 "official_gt_reader_source_sha256s", "coverage_tool_sha256",
                 "filtered_rle_decoder_source_sha256", "filtered_rle_decoder_extension_sha256",
                 "official_gt_provenance_sha256"}
    _keys(bind, bind_keys, "pool coverage source bindings")
    expected_bind = {
        "global_prediction_barrier_sha256": _sha_file(BASE / "GLOBAL_PREDICTION_BARRIER.json"),
        "roster_lock_sha256": _sha_file(BASE / "ROSTER_LOCK.json"),
        "tracking_global_barrier_sha256": _sha_file(BASE / "TRACKING_GLOBAL_BARRIER.json"),
        "postseal_scorer_sha256": _sha_file(scorer),
        "official_truth_reader_sha256": _sha_file(wrapper),
        "coverage_tool_sha256": _sha_file(tool),
        "filtered_rle_decoder_source_sha256": _sha_file(mask_src),
        "filtered_rle_decoder_extension_sha256": _sha_file(mask_ext),
        "official_gt_reader_source_sha256s": {name: _sha_file(path) for name, path in truth_files.items()},
        "official_gt_provenance_sha256": _sha(bind.get("official_gt_provenance_sha256"),
                                               "official GT provenance digest"),
    }
    _need(bind == expected_bind and bind["global_prediction_barrier_sha256"] == barrier_sha,
          "pool coverage binding mismatch")

    row_keys = {"alias", "dataset", "known_gt_frames", "gt_frames_in_canonical_map",
                "gt_frames_outside_canonical_map", "canonical_gt_frames_with_empty_filtered_pool",
                "gt_frames_with_empty_filtered_pool", "sum_max_bbox_iou", "mean_max_bbox_iou",
                "gt_frames_max_iou_ge_0_25", "fraction_gt_frames_max_iou_ge_0_25",
                "gt_frames_max_iou_ge_0_50", "fraction_gt_frames_max_iou_ge_0_50",
                "filtered_candidate_track_count", "nonempty_filtered_instance_masks"}
    raw_cases = source.get("cases")
    _need(isinstance(raw_cases, list) and len(raw_cases) == 8, "pool coverage must have 8 cases")
    cases, seen = [], set()
    count_fields = ("known_gt_frames", "gt_frames_in_canonical_map", "gt_frames_outside_canonical_map",
                    "canonical_gt_frames_with_empty_filtered_pool", "gt_frames_with_empty_filtered_pool",
                    "gt_frames_max_iou_ge_0_25", "gt_frames_max_iou_ge_0_50",
                    "filtered_candidate_track_count", "nonempty_filtered_instance_masks")
    for row in raw_cases:
        _keys(row, row_keys, "pool coverage case row")
        alias, dataset = row.get("alias"), row.get("dataset")
        _need(alias in set(ALIAS.values()) and dataset in DATASETS and alias not in seen,
              "invalid/duplicate pool coverage alias")
        _need(alias.startswith("HC" if dataset == "hc2" else "V"), "pool alias/domain mismatch")
        seen.add(alias)
        counts = {key: _count(row[key], f"pool.{key}") for key in count_fields}
        known = counts["known_gt_frames"]
        _validate_pool_case_count_invariants(counts, known, f"pool {alias}")
        total = _num(row["sum_max_bbox_iou"], "pool.sum_max_bbox_iou", low=0)
        mean = _num(row["mean_max_bbox_iou"], "pool.mean_max_bbox_iou", low=0, high=1)
        _need(total is not None and total <= known + 1e-8 and
              math.isclose(mean, total / known, rel_tol=0, abs_tol=1e-10),
              "pool mean differs from per-frame IoU sum")
        fracs = {}
        for suffix, countkey, frac_key in (("0_25", "gt_frames_max_iou_ge_0_25", "fraction_gt_frames_max_iou_ge_0_25"),
                                           ("0_50", "gt_frames_max_iou_ge_0_50", "fraction_gt_frames_max_iou_ge_0_50")):
            frac = _num(row[frac_key], f"pool.{frac_key}", low=0, high=1)
            _need(math.isclose(frac, counts[countkey] / known, rel_tol=0, abs_tol=1e-10),
                  "pool threshold fraction mismatch")
            fracs[frac_key] = frac
        cases.append({"case_id": alias, "dataset": DATASETS[dataset], **counts,
                      "sum_max_bbox_iou": total, "mean_max_bbox_iou": mean, **fracs})
    _need(seen == set(ALIAS.values()), "pool coverage omits a fixed alias")
    cases.sort(key=lambda r: (r["dataset"], r["case_id"]))

    summary = source.get("summary")
    _keys(summary, {"all", "by_dataset"}, "pool coverage summary")
    _keys(summary["by_dataset"], {"hc2", "vidstg"}, "pool dataset summaries")
    agg_keys = {"case_count", "known_gt_frames", "gt_frames_in_canonical_map",
                "gt_frames_outside_canonical_map", "canonical_gt_frames_with_empty_filtered_pool",
                "gt_frames_with_empty_filtered_pool", "frame_weighted_mean_max_bbox_iou",
                "case_macro_mean_max_bbox_iou", "gt_frames_max_iou_ge_0_25",
                "fraction_gt_frames_max_iou_ge_0_25", "gt_frames_max_iou_ge_0_50",
                "fraction_gt_frames_max_iou_ge_0_50"}
    group_rows = {"all": cases, "hc2": [r for r in cases if r["dataset"] == "HC"],
                  "vidstg": [r for r in cases if r["dataset"] == "VidSTG"]}
    aggregates = {}
    for group, raw in (("all", summary["all"]), ("hc2", summary["by_dataset"]["hc2"]),
                       ("vidstg", summary["by_dataset"]["vidstg"])):
        _keys(raw, agg_keys, f"pool summary {group}")
        subset = group_rows[group]; ncase = 8 if group == "all" else 4
        _need(len(subset) == ncase, "pool summary group size mismatch")
        denom = sum(r["known_gt_frames"] for r in subset)
        total_iou = sum(r["sum_max_bbox_iou"] for r in subset)
        expected = {"case_count": ncase, "known_gt_frames": denom,
                    "gt_frames_in_canonical_map": sum(r["gt_frames_in_canonical_map"] for r in subset),
                    "gt_frames_outside_canonical_map": sum(r["gt_frames_outside_canonical_map"] for r in subset),
                    "canonical_gt_frames_with_empty_filtered_pool": sum(r["canonical_gt_frames_with_empty_filtered_pool"] for r in subset),
                    "gt_frames_with_empty_filtered_pool": sum(r["gt_frames_with_empty_filtered_pool"] for r in subset),
                    "frame_weighted_mean_max_bbox_iou": total_iou / denom,
                    "case_macro_mean_max_bbox_iou": sum(r["mean_max_bbox_iou"] for r in subset) / ncase,
                    "gt_frames_max_iou_ge_0_25": sum(r["gt_frames_max_iou_ge_0_25"] for r in subset),
                    "fraction_gt_frames_max_iou_ge_0_25": sum(r["gt_frames_max_iou_ge_0_25"] for r in subset) / denom,
                    "gt_frames_max_iou_ge_0_50": sum(r["gt_frames_max_iou_ge_0_50"] for r in subset),
                    "fraction_gt_frames_max_iou_ge_0_50": sum(r["gt_frames_max_iou_ge_0_50"] for r in subset) / denom}
        aggregate_out = {}
        for key, value in expected.items():
            if isinstance(value, int):
                actual = _count(raw[key], f"pool summary {group}.{key}")
                _need(actual == value,
                      "pool aggregate count mismatch")
                aggregate_out[key] = actual
            else:
                actual = _num(raw[key], f"pool summary {group}.{key}")
                _need(math.isclose(actual, value, rel_tol=0, abs_tol=1e-10),
                      "pool aggregate mean/fraction mismatch")
                aggregate_out[key] = actual
        aggregates[group] = aggregate_out

    coverage_public = {"schema": "stvg-native-instance-public-filtered-pool-coverage-v1",
                       "status": "pass", "GT_read": True,
                       "GT_read_after_complete_prediction_seal": True,
                       "coverage_estimand": text["coverage_estimand"], "thresholds": [0.25, 0.5],
                       "denominator_definition": text["denominator_definition"],
                       "candidate_policy": text["candidate_policy"], "claim_boundaries": boundaries,
                       "bindings": expected_bind,
                       "summary": {"all": aggregates["all"],
                                   "by_dataset": {"hc2": aggregates["hc2"],
                                                  "vidstg": aggregates["vidstg"]}},
                       "cases": cases, "redactions": {k: False for k in redaction_keys},
                       "filtered_rle_extension_distributed": False}
    receipt_path = BASE / "ROOT_POOL_COVERAGE_READBACK.json"
    receipt = _read_json(receipt_path)
    scope = "independent saved pool coverage scalar aggregation and source binding readback; no new label read or prediction changes"
    _need(receipt.get("status") == "pass" and receipt.get("coverage_sha256") == _sha_file(src_path)
          and receipt.get("appendix_sha256") == _sha_file(appendix)
          and receipt.get("CPU_only") is True and receipt.get("raw_GT_read_by_this_reader") is False
          and receipt.get("scope") == scope, "pool root receipt does not bind coverage and appendix")
    receipt_public = {"schema": "stvg-native-instance-root-pool-coverage-readback-public-v1",
                      "status": "pass", "coverage_sha256": _sha_file(src_path),
                      "appendix_sha256": _sha_file(appendix), "CPU_only": True,
                      "raw_GT_read_by_this_reader": False,
                      "reader_sha256": _sha_file(ROOT / "scripts/review_stvg_native_instance_coverage_v1.py"),
                      "scope": scope}
    return coverage_public, receipt_public


def _audit_actual() -> dict[str, Any]:
    """One-shot audit. Caller must invoke only after the root seals exist."""
    _need(not AUDIT_RECEIPT.exists() and not ANONYMOUS_RECEIPT.exists(),
          "audit receipts already exist; refusing overwrite")
    # Sealed score summaries contain scalars only. This audit never opens raw case tensors/media.
    lineage = _sealed_prediction_lineage()
    raw_rows = _read_json(BASE / "POSTSEAL_SCORE_ROWS.json")
    raw_summary = _read_json(BASE / "POSTSEAL_SCORE_SUMMARY.json")
    expected_rows = _expected_rows(raw_rows, lineage)
    expected_summary = _expected_summary(raw_summary, expected_rows)
    public_rows = _read_json(OUT / "postseal_metrics.json")
    public_summary = _read_json(OUT / "postseal_summary.json")
    _need(_canonical(public_rows) == _canonical(expected_rows),
          "public 72-row projection differs from independent source projection")
    _need(_canonical(public_summary) == _canonical(expected_summary),
          "public summary/contrast values differ from independent paired recomputation")

    barrier_sha = lineage["prediction_barrier_sha256"]
    student_barrier_sha = lineage["bindings"]["student_barrier_sha256"]
    math_expected = _expected_math_projection(student_barrier_sha, expected_rows["bindings"])
    math_public = _read_json(OUT / "student_math_readback.json")
    _need(_canonical(math_public) == _canonical(math_expected),
          "public student-math projection differs from the CPU root receipt")
    cost_expected = _expected_cost_projection(student_barrier_sha, math_expected, expected_rows)
    cost_expected["tracking_and_teacher_receipts"] = _expected_receipt_cost_details(
        expected_rows, cost_expected)
    costs = _read_json(OUT / "measured_costs.json")
    _need(_canonical(costs) == _canonical(cost_expected),
          "public measured costs/nulls differ from bound cost receipts")
    postseal_expected = _expected_postseal_receipt(
        BASE / "POSTSEAL_SCORE_ROWS.json", BASE / "POSTSEAL_SCORE_SUMMARY.json", barrier_sha)
    visual_expected = _expected_visual_readback()
    generation_expected = _expected_generation_readback()
    recovery_expected = _expected_recovery_readback()
    geometry_expected = _expected_geometry_readback()
    _need(_canonical(_read_json(OUT / "root_postseal_readback.json")) == _canonical(postseal_expected),
          "public root-postseal receipt differs from its source-bound receipt")
    _need(_canonical(_read_json(OUT / "visual_review.json")) == _canonical(visual_expected),
          "public visual receipt differs from the verified private-contact/PDF receipt")
    _need(_canonical(_read_json(OUT / "generation_configuration.json")) == _canonical(generation_expected),
          "public generation configuration projection differs from the frozen readback")
    _need(_canonical(_read_json(OUT / "recovery_metadata.json")) == _canonical(recovery_expected),
          "public recovery metadata differs from the hash-only approved projection")
    _need(_canonical(_read_json(OUT / "independent_geometry_readback.json")) ==
          _canonical(geometry_expected), "public geometry receipt differs from independent readback")
    _need(_canonical(_read_json(OUT / "root_readbacks.json")) ==
          _canonical(_expected_root_summaries()), "public root receipt summary lost or altered safe fields")
    pool_expected, pool_readback_expected = _expected_pool_coverage(barrier_sha)
    pool_public = _read_json(OUT / "pool_candidate_coverage.json")
    pool_readback_public = _read_json(OUT / "pool_coverage_readback.json")
    _need(_canonical(pool_public) == _canonical(pool_expected),
          "public filtered-pool scalars differ from independent denominator/aggregation readback")
    _need(_canonical(pool_readback_public) == _canonical(pool_readback_expected),
          "public pool root-readback projection differs from its receipt")

    # Check report output counts and public scalar envelope, including null-cost behavior.
    _need(len(public_rows["rows"]) == 72 and len(public_summary["arm_metric_summaries"]) == 90
          and len(public_summary["contrasts"]) == 80, "public result counts are incomplete")
    _need(all((r["metrics"] is None) or set(r["metrics"]) == set(METRICS) for r in public_rows["rows"]),
          "metric schema incomplete")
    _need(costs.get("GT_read") is False and costs.get("case_count") == 8
          and len(costs.get("records", [])) == 8, "measured cost projection incomplete")
    for record in costs["records"]:
        _need(record.get("case_id") in set(ALIAS.values()), "cost record leaks unaliased identity")
        for name in SPATIAL:
            item = record["algorithm_costs"][name]
            if item["status"] == "exact_no_update":
                _need(item["seconds"] is None and item["optimizer_steps"] == 0
                      and item["final_redecode_calls"] == 0 and item["fit_present"] is False,
                      "no-update cost null/zero contract lost")
            _need(item["elapsed_seconds_available"] is (item["seconds"] is not None),
                  "cost availability flag disagrees with null")

    # Verify the source AST/runtime pin projection and the sealed visual/public readbacks.
    bindings = _read_json(OUT / "runtime_source_bindings.json")
    _need(bindings.get("status") == "pass" and bindings.get("schema") ==
          "stvg-native-instance-public-source-bindings-v1", "source binding projection invalid")
    _need(isinstance(bindings.get("bindings"), list), "source binding rows absent")
    expected_bindings = _expected_runtime_bindings()
    _need(_canonical(bindings["bindings"]) == _canonical(expected_bindings),
          "public source bindings differ from locked runtime/embedded source hashes")
    for binding in bindings["bindings"]:
        _keys(binding, {"runtime", "binding", "sha256", "published", "source", "byte_exact"},
              "runtime source-binding row")
        _sha(binding["sha256"], "runtime source binding")
        _need(binding["published"] is (binding["source"] not in {"private_or_protected_binding"}),
              "source publication bit inconsistent")
        if binding["published"]:
            rel = binding["source"]
            _need(_path_is_public(rel)
                  and (ROOT / rel).is_file() and _sha_file(ROOT / rel) == binding["sha256"],
                  "published source differs from pinned bytes")
            _need(binding["byte_exact"] is True, "published source not marked byte-exact")
    _need(bindings["binding_count"] == len(bindings["bindings"])
          and bindings["public_source_count"] == sum(x["published"] for x in bindings["bindings"])
          and bindings["private_hash_only_count"] == sum(not x["published"] for x in bindings["bindings"]),
          "source binding counts mismatch")

    manifest_info = _audit_manifest()
    manifest_paths = set(manifest_info["paths"])
    for rel in ("docs/STVG_NATIVE_INSTANCE_V1_REVIEWED.md",
                "protocols/stvg_native_instance_spatial_opd_v1.md",
                "scripts/prepare_stvg_native_instance_public_v1.py",
                "scripts/review_stvg_native_instance_geometry_v1.py",
                PDF.relative_to(ROOT).as_posix(),
                PNG.relative_to(ROOT).as_posix()):
        _need(rel in manifest_paths, f"required source/report artifact missing from manifest: {rel}")
    for binding in expected_bindings:
        if binding["published"]:
            _need(binding["source"] in manifest_paths,
                  f"runtime-pinned source omitted from manifest: {binding['source']}")
    readback = _audit_source_readback(manifest_info, bindings)
    # Recompute the scalar envelope digests and bind the report receipts.
    scalar = _read_json(OUT / "public_scalar_readback.json")
    _need(scalar.get("status") == "pass" and scalar.get("case_count") == 8
          and scalar.get("row_count") == 72 and scalar.get("contrast_count") == 80
          and scalar.get("GT_read") is True and scalar.get("GT_used_for_method_or_hyperparameter_selection") is False,
          "public scalar envelope incomplete or overstated")
    _need(scalar.get("sha256", {}).get("postseal_metrics") == _canonical_sha(public_rows)
          and scalar["sha256"].get("postseal_summary") == _canonical_sha(public_summary)
          and scalar["sha256"].get("student_math") == _canonical_sha(math_public)
          and scalar["sha256"].get("costs") == _canonical_sha(costs)
          and scalar["sha256"].get("visual_review") == _canonical_sha(visual_expected)
          and scalar["sha256"].get("independent_geometry_readback") == _canonical_sha(geometry_expected)
          and scalar["sha256"].get("pool_candidate_coverage") == _canonical_sha(pool_expected)
          and scalar["sha256"].get("pool_coverage_readback") == _canonical_sha(pool_readback_expected)
          and scalar["sha256"].get("pool_coverage_appendix") == _sha_file(
              ROOT / "docs/STVG_NATIVE_INSTANCE_POOL_COVERAGE_APPENDIX.md")
          and scalar.get("pool_candidate_coverage_case_count") == 8
          and scalar.get("pool_candidate_coverage_known_gt_frames") ==
              pool_expected["summary"]["all"]["known_gt_frames"]
          and scalar.get("independent_geometry_checks") == 31842,
          "public scalar envelope digests differ")

    expected_paths = _expected_manifest_paths(bindings)
    _need(set(manifest_info["paths"]) == expected_paths,
          "manifest path set differs from fixed sources, pinned closure, and safe outputs")

    info = {"manifest": {key: value for key, value in manifest_info.items() if key != "paths"},
            "source_readback": readback,
            "counts": {"cases": 8, "rows": 72, "summary_cells": 90, "contrasts": 80,
                       "cost_records": 8, "bindings": len(bindings["bindings"])},
            "status": "pass", "scope": "public scalar/source/privacy audit; no model or new GT scoring"}
    audit = {"schema": "stvg-native-instance-public-scalar-audit-v1", **info,
             "public_stage_sha256": _sha_file(MANIFEST),
             "raw_score_bytes_opened": True,
             "raw_media_or_prediction_payload_opened": False,
             "new_GT_scoring": False}
    anon = {"schema": "stvg-native-instance-root-anonymous-scalar-readback-v1", "status": "pass",
            "case_aliases": sorted(ALIAS.values()), "row_count": 72,
            "summary_cell_count": 90, "contrast_count": 80,
            "unknown_and_sign_tails_preserved": True,
            "null_costs_preserved": True,
            "private_payload_staged": False,
            "manifest_sha256": manifest_info["manifest_sha256"],
            "public_projection_sha256": _sha_file(OUT / "postseal_metrics.json"),
            "coverage_root_receipt_sha256": _sha_file(BASE / "ROOT_POOL_COVERAGE_READBACK.json")}
    for path, value in ((AUDIT_RECEIPT, audit), (ANONYMOUS_RECEIPT, anon)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return {"status": "pass", "audit_receipt": AUDIT_RECEIPT.relative_to(ROOT).as_posix(),
            "anonymous_receipt": ANONYMOUS_RECEIPT.relative_to(ROOT).as_posix(),
            "rows": 72, "summaries": 90, "contrasts": 80}


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--contracts", action="store_true", help="run synthetic CPU-only checks")
    group.add_argument("--audit", action="store_true", help="audit an already sealed public stage")
    args = parser.parse_args()
    result = _verify_contract_fixture() if args.contracts else _audit_actual()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
