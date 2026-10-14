"""Prepare an explicitly allowlisted public export for matched-support STVG.

This is a staging tool only.  Run it only after the student, math, metric,
independent-readback, root-summary and visual-review inputs are complete.
It never opens a student checkpoint, roster, reference lock, source case,
annotation, image, video, teacher response, or query text.  It does not push.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_matched_support_reference_v1"
OUT = ROOT / "results/stvg_matched_support_reference/2026-10-14"
PUBLIC_STAGE = BASE / "public_stage"
MANIFEST_PATH = BASE / "PUBLIC_MANIFEST.json"
SOURCE_READBACK_PATH = BASE / "ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json"
RUNTIME_NAME = "STUDENT_RUNTIME.json"

REPORT_PATH = ROOT / "docs/STVG_MATCHED_SUPPORT_REFERENCE_V1.md"
PDF_PATH = BASE / "figures/matched_support_reference_v1.pdf"
PROTOCOL_PATH = ROOT / "protocols/stvg_matched_support_reference_v1.md"

CASE_IDS = tuple([f"hc2_{i:03d}" for i in range(4)]
                 + [f"vidstg_{i:03d}" for i in range(4)])
CASE_ALIAS = {cid: ("HC" if cid.startswith("hc2_") else "V")
              + f"{int(cid[-3:]) + 1:02d}" for cid in CASE_IDS}
ROLES = ("source", "r1", "pool_oracle", "gt_oracle")
ARMS = ("F", "T", *ROLES, *(f"T_{role}" for role in ROLES))
METRICS = ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT")

PUBLIC_SOURCE_PREFIXES = ("scripts/", "vg_tta/", "methods/", "protocols/")
PUBLIC_SUFFIXES = {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".txt"}
DENIED_PATH_PARTS = (
    "source_cases", "private_inputs", "private_views", ".pt", ".pth",
    ".safetensors", "attachments", "conversation", "annotations",
)

SUMMARY_INPUTS = {
    "prelaunch": "ROOT_PRELAUNCH_READBACK.json",
    "launched_revision": "ROOT_LAUNCHED_REVISION_READBACK.json",
    "reference_support": "ROOT_REFERENCE_SUPPORT_READBACK.json",
    "visual_review": "ROOT_VISUAL_REVIEW.json",
    "gpu_status": "STUDENT_GPU_STATUS.json",
    "live_resource_limits": "STUDENT_ACTUAL_LIVE_RESOURCE_LIMITS.json",
}

# These keys are the only generic root/visual/resource summary fields eligible
# for publication.  Free text, paths, identifiers, query/prompt fields, and
# geometry fields are not copied by the generic projection.
SAFE_SUMMARY_KEYS = {
    "schema", "status", "scope", "stage", "count", "cases", "case_count",
    "source_count", "metric_count", "support_count", "included", "excluded",
    "fit_count", "expected_fit_count", "actual_fit_count", "optimizer_steps",
    "fit_round_count", "all_28_fits_have_five_rounds", "actual_final_redecode_calls",
    "reference_quality_count", "group_count", "files", "bytes", "check_count",
    "checks_passed", "checks_failed", "pass_count", "fail_count", "warning_count",
    "role_fit_count", "role_anchor_totals", "reference_role_anchor_totals",
    "per_case_support", "reason_counts", "no_update_count", "updated_count",
    "max_metric_scalar_abs_error", "max_reference_iou_abs_error",
    "max_abs_error", "mean_abs_error", "GT_read", "GT_used_for_oracle_reference",
    "GT_used_for_support_availability_only", "GT_used_for_method_or_hyperparameter_selection",
    "GT_used_for_hyperparameters", "GT_reference_geometry_read_after_student_seal",
    "GT_loader_called_after_complete_prediction_and_math_seals",
    "official_GT_label_loader_called", "offline_diagnostic_only", "GPU_initialized",
    "GPU_used", "model_loaded", "video_or_RGB_read", "source_model_reloaded",
    "decoder_backward_recomputed", "score_metrics_recomputed", "CPU_only",
    "fit_roles", "empty_roles", "fit_round_count", "optimizer_step_count",
    "same_source_r1_anchor_count", "same_source_r1_anchor_cases_fitted_twice",
    "same_source_r1_exact_cases", "same_source_r1_exact_windows",
    "GT_derived_reference_consumed", "GT_assistance_explicit",
    "Source_selected_R1_not_skipped", "all_four_roles_identical_support",
    "source_r1_exact_same_cases", "source_r1_same_anchor_windows",
    "all_case_root", "runtime_source_binding_count", "private_hash_only_count",
    "public_source_count", "disk_free_bytes", "host_available_bytes",
    "gpu_free_bytes", "gpu_peak_bytes", "memory_current_bytes", "memory_anon_bytes",
    "own_gpu_active", "resource_guard_status", "worker_exit_status",
    "visual_review_status", "reviewed_case_count", "all_cases_reviewed",
    "input_contact_sheets_reviewed", "GT_visual_reviewed", "GT_readback_reviewed",
    "done", "total", "stage", "reference_lock_sha256", "runtime_sha256",
    "actual_full_redecode_count", "previous_prelaunch_receipts_historical", "worker_sha256",
    "pdf_sha256", "report_sha256", "PDF_pages", "all_pixels_actually_viewed",
    "private_contact_count", "raw_media_public",
}

SAFE_LABELS = {
    "pass", "complete", "locked", "sealed", "measured", "offline_diagnostic_only",
    "matched_support_reference", "student", "math", "metrics", "visual_review",
    "prelaunch", "reference_support", "gpu_status", "live_resource_limits",
    "source", "r1", "pool_oracle", "gt_oracle", "F", "T", "hc2", "vidstg",
    "exact", "unknown", "included", "excluded", "reviewed", "not_reviewed",
    "same_support_GT_known", "other_GT_known_frames", "same_support_reference_diagnosis",
    "independent_arithmetic_readback", "root_readback", "worker", "launcher", "prepared",
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _safe_rel(rel: str) -> str:
    if not isinstance(rel, str) or not rel or "\\" in rel:
        raise ValueError("source path is not a nonempty ROOT-relative POSIX path")
    path = Path(rel)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("absolute or traversing source path is refused")
    if any(part in rel for part in DENIED_PATH_PARTS):
        raise ValueError("private payload path is refused")
    return rel


def _runtime_binding_key(raw_rel: Any) -> str:
    """Validate a runtime map key without exposing private path components."""
    if not isinstance(raw_rel, str) or not raw_rel or "\\" in raw_rel:
        raise ValueError("runtime binding key is not a ROOT-relative POSIX path")
    path = Path(raw_rel)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("runtime binding key is absolute or traverses directories")
    return raw_rel


def _is_public_source(rel: str) -> bool:
    return (rel.startswith(PUBLIC_SOURCE_PREFIXES)
            and rel != "methods/CURRENT_METHOD.json"
            and Path(rel).suffix.lower() in PUBLIC_SUFFIXES
            and not any(part in rel for part in DENIED_PATH_PARTS))


def _source_binding_rows() -> tuple[set[str], list[dict[str, Any]]]:
    runtime_path = BASE / RUNTIME_NAME
    runtime = _read_json(runtime_path)
    if (runtime.get("status") != "locked" or runtime.get("GT_read") is not False
            or runtime.get("offline_diagnostic_only") is not True):
        raise ValueError("student runtime is not the locked offline-diagnostic runtime")
    pins, protected = runtime.get("pins"), runtime.get("protected")
    if not isinstance(pins, dict) or not isinstance(protected, dict):
        raise ValueError("runtime pin maps are malformed")
    paths: set[str] = set()
    rows: list[dict[str, Any]] = []
    for group, mapping in (("pin", pins), ("protected", protected)):
        for raw_rel, digest in sorted(mapping.items()):
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("runtime contains a malformed SHA-256 binding")
            rel = _runtime_binding_key(raw_rel)
            publish = _is_public_source(rel)
            if publish:
                _safe_rel(rel)
                source_path = ROOT / rel
                if not source_path.is_file() or _sha(source_path) != digest:
                    raise ValueError(f"public runtime source pin differs: {rel}")
                paths.add(rel)
                rows.append({"runtime": RUNTIME_NAME, "binding": group,
                             "source": rel, "sha256": digest, "published": True,
                             "byte_exact": True})
            else:
                # Never open or hash the private/protected target.  Preserve only
                # the digest already declared by the locked runtime.
                rows.append({"runtime": RUNTIME_NAME, "binding": group,
                             "source": "private_or_protected_binding",
                             "sha256": digest, "published": False,
                             "byte_exact": "runtime_declared_only"})

    helper_rel = "scripts/stvg_video_global_spatial_fit_fp32_geometry001.py"
    helper_path = ROOT / helper_rel
    if helper_rel not in paths or not helper_path.is_file():
        raise ValueError("the frozen fit helper is not in the public pinned source closure")
    module = ast.parse(helper_path.read_text(encoding="utf-8"), filename=helper_rel)
    literal = None
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "SOURCE_BINDINGS"
                for target in node.targets):
            literal = ast.literal_eval(node.value)
            break
    if not isinstance(literal, dict) or not literal:
        raise ValueError("frozen fit helper SOURCE_BINDINGS is missing or not literal")
    for raw_rel, value in sorted(literal.items()):
        rel = _runtime_binding_key(raw_rel)
        digest = value.get("sha256") if isinstance(value, dict) else value
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("embedded SOURCE_BINDINGS has a malformed digest")
        publish = _is_public_source(rel)
        if publish:
            _safe_rel(rel)
            source_path = ROOT / rel
            if not source_path.is_file() or _sha(source_path) != digest:
                raise ValueError(f"embedded frozen source binding differs: {rel}")
            paths.add(rel)
            rows.append({"runtime": "embedded_SOURCE_BINDINGS:" + helper_rel,
                         "binding": "scientific_source", "source": rel,
                         "sha256": digest, "published": True, "byte_exact": True})
        else:
            rows.append({"runtime": "embedded_SOURCE_BINDINGS:" + helper_rel,
                         "binding": "scientific_source",
                         "source": "private_or_protected_binding", "sha256": digest,
                         "published": False, "byte_exact": "source_declared_only"})
    return paths, rows


def _case_alias(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value in CASE_ALIAS.values():
        return value
    if not isinstance(value, str) or value not in CASE_ALIAS:
        raise ValueError("unexpected case identifier in public scalar source")
    return CASE_ALIAS[value]


def _finite_number(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("a public metric field is not numeric")
    if not math.isfinite(float(value)):
        raise ValueError("nonfinite public numeric value")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} is not a JSON boolean")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} is not a JSON integer")
    return value


def _check_alias_catalog(rows: list[Mapping[str, Any]], key_fields: tuple[str, ...],
                         expected_count: int, label: str,
                         require_all_cases: bool = True) -> None:
    identities = []
    cases = set()
    for row in rows:
        alias = _case_alias(row["case_id"])
        cases.add(alias)
        identities.append((alias, *(row[key] for key in key_fields)))
    if len(rows) != expected_count or len(set(identities)) != expected_count:
        raise ValueError(f"{label} has an incomplete or duplicate row catalog")
    if require_all_cases and cases != set(CASE_ALIAS.values()):
        raise ValueError(f"{label} does not cover the fixed anonymous case catalog")


def _exact_keys(row: Mapping[str, Any], wanted: set[str], label: str) -> None:
    if not isinstance(row, Mapping) or set(row) != wanted:
        raise ValueError(f"{label} schema differs from the explicit public projection")


def _sha256_value(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _metric_values(values: Mapping[str, Any]) -> dict[str, float]:
    _exact_keys(values, set(METRICS), "metric values")
    return {key: float(_finite_number(values[key])) for key in METRICS}


def _geometry_delta(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = {"selection_status", "GT_known_frame_count", "mean_delta_sIoU_pp",
            "positive_count", "negative_count", "neutral_count", "unknown_count"}
    _exact_keys(value, keys, "support-slice metric")
    if value["selection_status"] not in {"same_support_GT_known", "other_GT_known_frames"}:
        raise ValueError("unexpected support-slice label")
    out = {"selection_status": value["selection_status"]}
    for key in keys - {"selection_status"}:
        n = _finite_number(value[key])
        if key != "mean_delta_sIoU_pp" and (not isinstance(n, int) or n < 0):
            raise ValueError("support-slice count is invalid")
        out[key] = n
    return out


def _postseal_projection(source: Mapping[str, Any], support: Mapping[str, Any]) -> dict[str, Any]:
    required = {"status", "scope", "rows", "groups", "reference_quality", "fit_movement",
                "metric_count", "support_count", "actual_fit_count", "GT_read",
                "GT_used_for_oracle_reference", "GT_used_for_method_or_hyperparameter_selection",
                "student_barrier_sha256", "reference_lock_sha256", "root_math_sha256",
                "GT_bindings_sha256", "time"}
    _exact_keys(source, required, "root postseal report")
    if (source["status"] != "pass" or source["GT_read"] is not True
            or source["GT_used_for_oracle_reference"] is not True):
        raise ValueError("root report is not a completed, disclosed GT-assisted report")
    if source["GT_used_for_method_or_hyperparameter_selection"] is not False:
        raise ValueError("GT was declared for method or hyperparameter selection")
    for key in ("student_barrier_sha256", "reference_lock_sha256", "root_math_sha256",
                "GT_bindings_sha256"):
        _sha256_value(source[key], f"report.{key}")
    support_count = _integer(support["included"], "support.included")
    support_by_case: dict[str, int] = {alias: 0 for alias in CASE_ALIAS.values()}
    for record in support["records"]:
        if record["included"]:
            support_by_case[record["case_id"]] += 1
    if (_integer(source["metric_count"], "report.metric_count") != 80 or len(source["rows"]) != 80
            or len(source["groups"]) != 20 or len(source["reference_quality"]) != 76
            or _integer(source["support_count"], "report.support_count") != support_count
            or _integer(source["actual_fit_count"], "report.actual_fit_count") != 28):
        raise ValueError("root metric/reference counts do not match the sealed scope")
    rows = []
    row_keys = {"case_id", "dataset", "arm", "metrics", "baseline", "delta_vIoU_pp",
                "official_independent_max_abs_error", "support_count",
                "supervised_geometry", "unsupervised_geometry"}
    for row in source["rows"]:
        _exact_keys(row, row_keys, "root metric row")
        if row["dataset"] not in {"hc2", "vidstg"} or row["arm"] not in ARMS:
            raise ValueError("unexpected dataset or arm in metric row")
        expected_baseline = "T" if row["arm"] == "T" or row["arm"].startswith("T_") else "F"
        if row["baseline"] != expected_baseline:
            raise ValueError("metric row baseline does not match its frozen arm")
        row_support_count = _integer(row["support_count"], "metric.support_count")
        expected_row_support = (0 if row["arm"] in {"F", "T"}
                                else support_by_case[_case_alias(row["case_id"])])
        if row_support_count != expected_row_support:
            raise ValueError("metric row support count differs from its case support")
        rows.append({"case_id": _case_alias(row["case_id"]), "dataset": row["dataset"],
                     "arm": row["arm"], "metrics": _metric_values(row["metrics"]),
                     "baseline": row["baseline"],
                     "delta_vIoU_pp": _finite_number(row["delta_vIoU_pp"]),
                     "official_independent_max_abs_error": _finite_number(
                         row["official_independent_max_abs_error"]),
                     "support_count": int(row["support_count"]),
                     "supervised_geometry": _geometry_delta(row["supervised_geometry"]),
                     "unsupervised_geometry": _geometry_delta(row["unsupervised_geometry"])})
    group_keys = {"dataset", "arm", "count", "mean_vIoU", "mean_delta_pp",
                  "positive", "negative", "neutral"}
    groups = []
    for row in source["groups"]:
        _exact_keys(row, group_keys, "root metric group")
        if row["dataset"] not in {"hc2", "vidstg"} or row["arm"] not in ARMS:
            raise ValueError("unexpected dataset or arm in metric group")
        groups.append({key: (_finite_number(value) if key in {"mean_vIoU", "mean_delta_pp"}
                            else _integer(value, f"group.{key}") if key in
                            {"count", "positive", "negative", "neutral"} else value)
                       for key, value in row.items()})
    quality_keys = {"case_id", "dataset", "role", "slot", "candidate_id",
                    "reference_IoU", "offline_label_assisted"}
    quality = []
    for row in source["reference_quality"]:
        _exact_keys(row, quality_keys, "reference-quality row")
        if row["dataset"] not in {"hc2", "vidstg"} or row["role"] not in ROLES:
            raise ValueError("unexpected dataset or reference role")
        if (row["candidate_id"] is not None
                and (isinstance(row["candidate_id"], bool)
                     or not isinstance(row["candidate_id"], int))):
            raise ValueError("candidate identifier is not a scalar ID")
        quality.append({"case_id": _case_alias(row["case_id"]), "dataset": row["dataset"],
                        "role": row["role"], "slot": _integer(row["slot"], "quality.slot"),
                        "candidate_id": row["candidate_id"],
                        "reference_IoU": _finite_number(row["reference_IoU"]),
                        "offline_label_assisted": _boolean(
                            row["offline_label_assisted"], "quality.offline_label_assisted")})
    movement_keys = {"case_id", "role", "optimizer_steps", "parameter_delta_norm",
                     "mean_reference_IoU_before", "mean_reference_IoU_after",
                     "reference_IoU_delta_pp"}
    movement = []
    for row in source["fit_movement"]:
        _exact_keys(row, movement_keys, "fit-movement row")
        if row["role"] not in ROLES:
            raise ValueError("unexpected fit-movement role")
        movement.append({"case_id": _case_alias(row["case_id"]), "role": row["role"],
                         "optimizer_steps": _integer(row["optimizer_steps"], "movement.optimizer_steps"),
                         "parameter_delta_norm": _finite_number(row["parameter_delta_norm"]),
                         "mean_reference_IoU_before": _finite_number(
                             row["mean_reference_IoU_before"]),
                         "mean_reference_IoU_after": _finite_number(
                             row["mean_reference_IoU_after"]),
                         "reference_IoU_delta_pp": _finite_number(row["reference_IoU_delta_pp"])})
    _check_alias_catalog(rows, ("arm",), 80, "root metric rows")
    _check_alias_catalog(quality, ("role", "slot"), 76, "reference-quality rows",
                         require_all_cases=False)
    if len({(r["dataset"], r["arm"]) for r in groups}) != 20:
        raise ValueError("root group catalog is not the fixed 20-cell matrix")
    expected_group_ids = {(dataset, arm) for dataset in ("hc2", "vidstg") for arm in ARMS}
    if {(r["dataset"], r["arm"]) for r in groups} != expected_group_ids:
        raise ValueError("root group matrix has a missing or unexpected arm")
    if len(movement) != 28 or len({(r["case_id"], r["role"]) for r in movement}) != 28:
        raise ValueError("fit-movement rows are duplicated")
    if len(quality) != support_count * len(ROLES):
        raise ValueError("reference-quality rows do not match four-role matched support")
    return {
        "schema": "stvg-matched-support-public-postseal-v1", "status": "pass",
        "scope": "offline_matched_support_reference_diagnostic",
        "metric_count": 80, "group_count": 20, "reference_quality_count": 76,
        "support_count": support_count, "actual_fit_count": int(source["actual_fit_count"]),
        "GT_read": True, "GT_used_for_oracle_reference": True,
        "GT_used_for_method_or_hyperparameter_selection": False,
        "student_barrier_sha256": source["student_barrier_sha256"],
        "reference_lock_sha256": source["reference_lock_sha256"],
        "root_math_sha256": source["root_math_sha256"],
        "GT_bindings_sha256": source["GT_bindings_sha256"],
        "rows": rows, "groups": groups, "reference_quality": quality,
        "fit_movement": movement,
    }


def _support_projection(source: Mapping[str, Any]) -> dict[str, Any]:
    if (source.get("status") != "pass" or source.get("GT_read") is not True
            or source.get("GT_used_for_support_availability_only") is not True):
        raise ValueError("support readback is incomplete or its GT scope differs")
    rows = source.get("records")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("support readback must contain the sealed 32-window catalog")
    keys = {"case_id", "dataset", "slot", "position", "physical_frame_id", "GT_known",
            "all_displayed_candidates_valid", "R1_valid_nonabstain", "included",
            "selection_uses_IoU_magnitude"}
    out_rows = []
    for row in rows:
        _exact_keys(row, keys, "support window row")
        if row["dataset"] not in {"hc2", "vidstg"}:
            raise ValueError("unexpected support dataset")
        known = _boolean(row["GT_known"], "support.GT_known")
        candidates_valid = _boolean(row["all_displayed_candidates_valid"],
                                     "support.all_displayed_candidates_valid")
        r1_valid = _boolean(row["R1_valid_nonabstain"], "support.R1_valid_nonabstain")
        included_row = _boolean(row["included"], "support.included")
        uses_iou = _boolean(row["selection_uses_IoU_magnitude"],
                            "support.selection_uses_IoU_magnitude")
        if included_row != (known and candidates_valid and r1_valid) or uses_iou:
            raise ValueError("support predicate or IoU-independent selection flag differs")
        out_rows.append({"case_id": _case_alias(row["case_id"]), "dataset": row["dataset"],
                         "slot": _integer(row["slot"], "support.slot"),
                         "position": _integer(row["position"], "support.position"),
                         "physical_frame_id": _integer(row["physical_frame_id"],
                                                        "support.physical_frame_id"),
                         "GT_known": known, "all_displayed_candidates_valid": candidates_valid,
                         "R1_valid_nonabstain": r1_valid, "included": included_row,
                         "selection_uses_IoU_magnitude": uses_iou})
    _check_alias_catalog(out_rows, ("slot",), 32, "support windows")
    if len({(row["case_id"], row["slot"]) for row in out_rows}) != 32:
        raise ValueError("support window catalog contains duplicate case/slot rows")
    included = sum(row["included"] for row in out_rows)
    excluded = len(out_rows) - included
    if (_integer(source.get("count", -1), "support.count") != 32
            or _integer(source.get("included", -1), "support.included") != included
            or _integer(source.get("excluded", -1), "support.excluded") != excluded):
        raise ValueError("support included/excluded totals do not match the row catalog")
    return {"schema": "stvg-matched-support-public-support-v1", "status": "pass",
            "scope": "GT-assisted availability metadata; no coordinates exported",
            "count": 32, "included": included, "excluded": excluded, "records": out_rows}


def _math_projection(source: Mapping[str, Any]) -> dict[str, Any]:
    if (source.get("status") != "pass" or source.get("GT_read") is not True
            or source.get("offline_diagnostic_only") is not True):
        raise ValueError("independent math readback is incomplete")
    expected = {"schema", "status", "scope", "cases", "student_barrier_sha256",
                "student_runtime_sha256", "reference_lock_sha256", "support_count",
                "fit_count", "expected_fit_count", "fit_round_count",
                "all_28_fits_have_five_rounds", "role_fit_count", "optimizer_steps",
                "same_source_r1_anchor_cases_fitted_twice", "support_audit", "case_readbacks",
                "GT_read", "GT_reference_geometry_read_after_student_seal",
                "official_GT_label_loader_called", "GT_used_for_hyperparameter_or_method_selection",
                "offline_diagnostic_only", "GPU_initialized", "model_loaded",
                "source_model_reloaded", "video_or_RGB_read", "readback_scope",
                "provenance", "fit_math_auditor", "completed_utc", "case_roots", "all_case_root"}
    _exact_keys(source, expected, "math readback")
    for key in ("cases", "support_count", "fit_count", "expected_fit_count",
                "fit_round_count", "optimizer_steps", "same_source_r1_anchor_cases_fitted_twice"):
        _integer(source[key], f"math.{key}")
    if (source["cases"] != 8 or source["support_count"] != 19
            or source["fit_count"] != 28 or source["expected_fit_count"] != 28
            or source["fit_round_count"] != 140
            or _boolean(source["all_28_fits_have_five_rounds"],
                        "math.all_28_fits_have_five_rounds") is not True):
        raise ValueError("math readback counts do not match the sealed 28-fit scope")
    for key in ("student_barrier_sha256", "student_runtime_sha256", "reference_lock_sha256"):
        _sha256_value(source[key], f"math.{key}")
    if (not isinstance(source["role_fit_count"], dict)
            or set(source["role_fit_count"]) != set(ROLES)
            or any(_integer(v, f"math.role_fit_count.{k}") != 7
                   for k, v in source["role_fit_count"].items())):
        raise ValueError("math role-fit count matrix differs from the fixed scope")
    case_readbacks = source.get("case_readbacks")
    if not isinstance(case_readbacks, dict) or set(case_readbacks) != set(CASE_IDS):
        raise ValueError("math readback case catalog differs from fixed scope")
    cases: dict[str, Any] = {}
    case_keys = {"dataset", "fit_roles", "empty_roles", "fit_count", "fit_round_count",
                 "optimizer_steps", "same_source_r1_anchor_count",
                 "r1_source_all_anchors_identical", "same_target_source_r1_dynamics_max_abs_error",
                 "role_math"}
    for cid, row in case_readbacks.items():
        _exact_keys(row, case_keys, "math case summary")
        if row["dataset"] not in {"hc2", "vidstg"}:
            raise ValueError("unexpected math dataset")
        for key in ("fit_count", "fit_round_count", "optimizer_steps",
                    "same_source_r1_anchor_count"):
            _integer(row[key], f"math.case.{key}")
        if (not isinstance(row["fit_roles"], list)
                or not isinstance(row["empty_roles"], list)
                or len(set(row["fit_roles"])) != len(row["fit_roles"])
                or len(set(row["empty_roles"])) != len(row["empty_roles"])
                or not set(row["fit_roles"]) <= set(ROLES)
                or not set(row["empty_roles"]) <= set(ROLES)
                or set(row["fit_roles"]) & set(row["empty_roles"])):
            raise ValueError("math case role lists are inconsistent")
        role_math = row["role_math"]
        if not isinstance(role_math, dict) or not set(role_math) <= set(ROLES):
            raise ValueError("unexpected per-role math catalog")
        projected_roles = {}
        for role, details in role_math.items():
            _exact_keys(details, {"steps", "optimizer_steps", "max_abs_errors", "objective"},
                        "fit math summary")
            if details["objective"] != "iou":
                raise ValueError("unexpected fit objective")
            errors = details["max_abs_errors"]
            error_keys = {"mean_abs", "sample_abs", "reward_abs", "weight_abs", "loss_abs",
                          "adam_parameter_abs", "adam_state_abs"}
            _exact_keys(errors, error_keys, "fit error summary")
            steps = _integer(details["steps"], "fit_math.steps")
            optimizer_steps = _integer(details["optimizer_steps"], "fit_math.optimizer_steps")
            if steps != 5 or not 0 <= optimizer_steps <= steps:
                raise ValueError("fit math round or optimizer-step count is invalid")
            projected_roles[role] = {"steps": steps,
                                     "optimizer_steps": optimizer_steps,
                                     "objective": "iou",
                                     "max_abs_errors": {k: _finite_number(v) for k, v in errors.items()}}
        cases[_case_alias(cid)] = {
            "dataset": row["dataset"], "fit_roles": list(row["fit_roles"]),
            "empty_roles": list(row["empty_roles"]), "fit_count": int(row["fit_count"]),
            "fit_round_count": int(row["fit_round_count"]),
            "optimizer_steps": int(row["optimizer_steps"]),
            "same_source_r1_anchor_count": int(row["same_source_r1_anchor_count"]),
            "r1_source_all_anchors_identical": _boolean(
                row["r1_source_all_anchors_identical"],
                "math.case.r1_source_all_anchors_identical"),
            "same_target_source_r1_dynamics_max_abs_error": _finite_number(
                row["same_target_source_r1_dynamics_max_abs_error"]),
            "role_math": projected_roles,
        }
    if (sum(row["fit_count"] for row in cases.values()) != 28
            or sum(row["fit_round_count"] for row in cases.values()) != 140
            or sum(row["optimizer_steps"] for row in cases.values()) != source["optimizer_steps"]):
        raise ValueError("math case-level fit counts do not sum to the root totals")
    support_audit = source["support_audit"]
    if not isinstance(support_audit, dict):
        raise ValueError("math support audit is malformed")
    _exact_keys(support_audit, {"per_case_support", "source_r1_exact_same_cases",
                                "source_r1_same_anchor_windows", "anchor_record_count"},
                "math support audit")
    per_case = support_audit.get("per_case_support")
    if not isinstance(per_case, dict) or set(per_case) != set(CASE_IDS):
        raise ValueError("math support audit does not bind the fixed cases")
    projected_support = {
        "per_case_support": {_case_alias(k): _integer(v, "support_audit.per_case_support")
                             for k, v in per_case.items()},
        "source_r1_exact_same_cases": _integer(
            support_audit["source_r1_exact_same_cases"], "support_audit.source_r1_exact_same_cases"),
        "source_r1_same_anchor_windows": _integer(
            support_audit["source_r1_same_anchor_windows"], "support_audit.source_r1_same_anchor_windows"),
        "anchor_record_count": _integer(
            support_audit["anchor_record_count"], "support_audit.anchor_record_count")}
    readback_scope = source["readback_scope"]
    scope_keys = {"saved_Gaussian_rollout_and_Adam_math",
                  "full_model_readout_equals_saved_cached_decoder",
                  "F_and_T_reuse_checked_against_old_sealed_student",
                  "T_reuse_checked_against_separate_old_temporal_barrier",
                  "sampled_source_grid_pixel_sha_checked_against_both_frozen_captures",
                  "decoder_backward_recomputed",
                  "same_anchor_Source_and_R1_runs_compared_dynamically",
                  "score_metrics_recomputed"}
    _exact_keys(readback_scope, scope_keys, "math readback scope")
    readback_scope = {k: _boolean(v, f"readback_scope.{k}")
                      for k, v in readback_scope.items()}
    return {
        "schema": source["schema"], "status": source["status"],
        "scope": "offline_matched_support_reference_diagnostic",
        "cases": int(source["cases"]), "support_count": int(source["support_count"]),
        "fit_count": int(source["fit_count"]),
        "expected_fit_count": int(source["expected_fit_count"]),
        "fit_round_count": int(source["fit_round_count"]),
        "all_28_fits_have_five_rounds": True,
        "role_fit_count": dict(source["role_fit_count"]),
        "optimizer_steps": int(source["optimizer_steps"]),
        "same_source_r1_anchor_cases_fitted_twice": int(
            source["same_source_r1_anchor_cases_fitted_twice"]),
        "support_audit": projected_support, "case_readbacks": cases,
        "GT_read": True, "GT_reference_geometry_read_after_student_seal": True,
        "official_GT_label_loader_called": False,
        "GT_used_for_hyperparameter_or_method_selection": False,
        "offline_diagnostic_only": True, "GPU_initialized": False,
        "model_loaded": False, "source_model_reloaded": False,
        "video_or_RGB_read": False, "readback_scope": readback_scope,
        "all_case_root": source["all_case_root"],
    }


def _independent_projection(source: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"schema", "status", "scope", "cases", "support_count", "metric_count",
                "reference_quality_count", "fit_movement_count", "group_count",
                "max_metric_scalar_abs_error", "max_reference_iou_abs_error",
                "max_fit_movement_scalar_abs_error", "metrics", "reference_quality",
                "fit_movement", "groups", "case_roots", "all_case_root", "GT_read",
                "GT_loader_called_after_complete_prediction_and_math_seals",
                "GT_used_for_method_or_hyperparameter_selection", "offline_diagnostic_only",
                "GPU_initialized", "model_loaded", "video_or_RGB_read", "provenance",
                "completed_utc"}
    _exact_keys(source, expected, "independent postseal readback")
    if (source.get("status") != "pass" or source.get("GT_read") is not True
            or source["GT_loader_called_after_complete_prediction_and_math_seals"] is not True
            or source["GT_used_for_method_or_hyperparameter_selection"] is not False
            or source["offline_diagnostic_only"] is not True
            or source["GPU_initialized"] is not False or source["model_loaded"] is not False
            or source["video_or_RGB_read"] is not False):
        raise ValueError("independent postseal readback is incomplete")
    counts = ("cases", "support_count", "metric_count", "reference_quality_count",
              "fit_movement_count", "group_count")
    for key in counts:
        _integer(source[key], f"independent.{key}")
    if (source["cases"] != 8 or source["support_count"] != 19
            or source["metric_count"] != 80 or source["reference_quality_count"] != 76
            or source["fit_movement_count"] != 28 or source["group_count"] != 20):
        raise ValueError("independent postseal readback counts differ")
    metrics = []
    metric_keys = {"case_id", "dataset", "arm", "metrics", "baseline", "delta_vIoU_pp",
                   "official_independent_max_abs_error", "root_scalar_max_abs_error"}
    for row in source["metrics"]:
        _exact_keys(row, metric_keys, "independent metric row")
        if row["dataset"] not in {"hc2", "vidstg"} or row["arm"] not in ARMS:
            raise ValueError("unexpected independent metric dataset or arm")
        metrics.append({"case_id": _case_alias(row["case_id"]), "dataset": row["dataset"],
                        "arm": row["arm"], "metrics": _metric_values(row["metrics"]),
                        "baseline": row["baseline"],
                        "delta_vIoU_pp": _finite_number(row["delta_vIoU_pp"]),
                        "official_independent_max_abs_error": _finite_number(
                            row["official_independent_max_abs_error"]),
                        "root_scalar_max_abs_error": _finite_number(row["root_scalar_max_abs_error"])})
    quality = []
    for row in source["reference_quality"]:
        allowed = {"case_id", "dataset", "role", "slot", "candidate_id",
                   "reference_IoU", "offline_label_assisted"}
        _exact_keys(row, allowed, "independent reference-quality row")
        if row["dataset"] not in {"hc2", "vidstg"} or row["role"] not in ROLES:
            raise ValueError("unexpected independent reference-quality dataset or role")
        if row["candidate_id"] is not None and (
                isinstance(row["candidate_id"], bool) or not isinstance(row["candidate_id"], int)):
            raise ValueError("independent candidate identifier is not scalar")
        quality.append({"case_id": _case_alias(row["case_id"]), "dataset": row["dataset"],
                        "role": row["role"], "slot": _integer(row["slot"], "quality.slot"),
                        "candidate_id": row["candidate_id"],
                        "reference_IoU": _finite_number(row["reference_IoU"]),
                        "offline_label_assisted": _boolean(
                            row["offline_label_assisted"], "quality.offline_label_assisted")})
    groups = []
    for row in source["groups"]:
        allowed = {"dataset", "arm", "count", "mean_vIoU", "mean_delta_pp",
                   "positive", "negative", "neutral"}
        _exact_keys(row, allowed, "independent metric group")
        if row["dataset"] not in {"hc2", "vidstg"} or row["arm"] not in ARMS:
            raise ValueError("unexpected independent group dataset or arm")
        groups.append({key: (_finite_number(value) if key in {"mean_vIoU", "mean_delta_pp"}
                             else _integer(value, f"group.{key}") if key in
                             {"count", "positive", "negative", "neutral"} else value)
                       for key, value in row.items()})
    movement = []
    movement_keys = {"case_id", "role", "optimizer_steps", "parameter_delta_norm",
                     "mean_reference_IoU_before", "mean_reference_IoU_after",
                     "reference_IoU_delta_pp"}
    for row in source["fit_movement"]:
        _exact_keys(row, movement_keys, "independent fit-movement row")
        if row["role"] not in ROLES:
            raise ValueError("unexpected independent fit-movement role")
        movement.append({"case_id": _case_alias(row["case_id"]), "role": row["role"],
                         "optimizer_steps": _integer(row["optimizer_steps"], "movement.optimizer_steps"),
                         "parameter_delta_norm": _finite_number(row["parameter_delta_norm"]),
                         "mean_reference_IoU_before": _finite_number(
                             row["mean_reference_IoU_before"]),
                         "mean_reference_IoU_after": _finite_number(
                             row["mean_reference_IoU_after"]),
                         "reference_IoU_delta_pp": _finite_number(row["reference_IoU_delta_pp"])})
    _check_alias_catalog(metrics, ("arm",), 80, "independent metric rows")
    _check_alias_catalog(quality, ("role", "slot"), 76,
                         "independent reference-quality rows", require_all_cases=False)
    if len({(row["dataset"], row["arm"]) for row in groups}) != 20:
        raise ValueError("independent group matrix is incomplete")
    if len(movement) != 28 or len({(row["case_id"], row["role"]) for row in movement}) != 28:
        raise ValueError("independent fit-movement matrix is incomplete or duplicated")
    roots = source.get("case_roots")
    if not isinstance(roots, dict) or set(roots) != set(CASE_IDS):
        raise ValueError("independent per-case hash catalog differs")
    for cid, digest in roots.items():
        _sha256_value(digest, f"independent.case_roots.{cid}")
    _sha256_value(source["all_case_root"], "independent.all_case_root")
    if not isinstance(source["provenance"], dict):
        raise ValueError("independent provenance is malformed")
    return {"schema": source["schema"], "status": source["status"],
            "scope": "independent_arithmetic_readback_of_offline_diagnostic",
            "cases": 8, "support_count": 19, "metric_count": 80,
            "reference_quality_count": 76, "group_count": 20,
            "max_metric_scalar_abs_error": _finite_number(source["max_metric_scalar_abs_error"]),
            "max_reference_iou_abs_error": _finite_number(source["max_reference_iou_abs_error"]),
            "fit_movement_count": 28,
            "max_fit_movement_scalar_abs_error": _finite_number(
                source["max_fit_movement_scalar_abs_error"]),
            "all_case_root": source["all_case_root"],
            "GT_read": True, "GT_used_for_method_or_hyperparameter_selection": False,
            "offline_diagnostic_only": True, "GPU_initialized": False,
            "model_loaded": False, "video_or_RGB_read": False,
            "metrics": metrics, "reference_quality": quality,
            "fit_movement": movement, "groups": groups,
            "case_roots": {_case_alias(k): v for k, v in roots.items()}}


def _safe_summary_tree(value: Any, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for raw_key, child in value.items():
            if raw_key in CASE_ALIAS:
                out[CASE_ALIAS[raw_key]] = _safe_summary_tree(child, "case")
            elif raw_key in SAFE_SUMMARY_KEYS:
                out[raw_key] = _safe_summary_tree(child, raw_key)
            elif raw_key in {"hc2", "vidstg", *ROLES, *ARMS}:
                out[raw_key] = _safe_summary_tree(child, raw_key)
        return out
    if isinstance(value, list):
        return [_safe_summary_tree(item, parent_key) for item in value
                if isinstance(item, (dict, list, bool, int, float, str))]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return _finite_number(value)
    if isinstance(value, str):
        if value in CASE_ALIAS:
            return CASE_ALIAS[value]
        if value in SAFE_LABELS:
            return value
        if parent_key.endswith("_sha256"):
            try:
                return _sha256_value(value, parent_key)
            except ValueError:
                return None
        if (parent_key == "schema" and value.startswith("stvg-")
                and all(ch.islower() or ch.isdigit() or ch in "_.:-" for ch in value)):
            return value[:96]
    return None


def _summary_projection() -> dict[str, Any]:
    projected = {}
    for label, name in SUMMARY_INPUTS.items():
        doc = _read_json(BASE / name)
        # Root/visual summaries are reduced to allowlisted scalar metadata;
        # no free-form comments, paths, query strings, or geometric values pass.
        projected[label] = _safe_summary_tree(doc)
        if not projected[label].get("status"):
            raise ValueError(f"{name} lacks an allowlisted completion status")
        if projected[label]["status"] not in {"pass", "complete", "sealed", "locked"}:
            raise ValueError(f"{name} is not a completed readback")
    return {"schema": "stvg-matched-support-public-root-summaries-v1",
            "status": "pass", "summaries": projected}


def _recovery_projection() -> dict[str, Any]:
    specs = {
        "capture_pixel_scope004": {
            "capture": {"status", "scope", "reason", "time", "actual_fit_count",
                        "actual_source_capture_calls", "old_source_outputs_changed"},
            "resolution": {"status", "scope", "time", "runtime_sha256",
                           "actual_first_case_saved", "no_fitting_or_reference_changes"},
            "snapshots": {"worker": "run_stvg_matched_support_reference_v1.py",
                          "launcher": "launch_stvg_matched_support_reference_v1.py"}},
        "prelaunch_feature_receipt002": {
            "capture": {"status", "scope", "reason", "time", "new_GPU_launched",
                        "science_outputs_changed"},
            "resolution": {"status", "scope", "time", "runtime_sha256", "GPU_used"},
            "snapshots": {"worker": "run_stvg_matched_support_reference_v1.py"}},
        "prelaunch_legacy_receipt003": {
            "capture": {"status", "scope", "reason", "new_GPU_launched",
                        "science_outputs_changed"},
            "resolution": {"status", "scope", "time", "runtime_sha256", "GPU_used"},
            "snapshots": {"worker": "run_stvg_matched_support_reference_v1.py"}},
        "prelaunch_schema001": {
            "capture": {"status", "scope"},
            "resolution": {"status", "all_reference_coordinates_and_selections_exact",
                           "new_fit_launched"},
            "snapshots": {"prepare": "prepare_stvg_matched_support_reference_v1.py"}},
    }
    rows = []
    for recovery_id, spec in specs.items():
        root = BASE / "recovery" / recovery_id
        capture_path, resolution_path = root / "CAPTURE.json", root / "RESOLUTION.json"
        capture, resolution = _read_json(capture_path), _read_json(resolution_path)
        _exact_keys(capture, spec["capture"], f"{recovery_id}.CAPTURE")
        _exact_keys(resolution, spec["resolution"], f"{recovery_id}.RESOLUTION")
        if not isinstance(capture["status"], str) or not capture["status"]:
            raise ValueError(f"{recovery_id} capture status is missing")
        if not isinstance(resolution["status"], str) or not resolution["status"]:
            raise ValueError(f"{recovery_id} resolution status is missing")

        safe_capture: dict[str, Any] = {"status": "captured"}
        for key, value in capture.items():
            if key in {"status", "scope", "reason", "time"}:
                continue
            if isinstance(value, bool):
                safe_capture[key] = value
            elif isinstance(value, int) and not isinstance(value, bool):
                safe_capture[key] = _integer(value, f"{recovery_id}.capture.{key}")
            else:
                raise ValueError(f"{recovery_id} capture has an unexpected public scalar")

        safe_resolution: dict[str, Any] = {"status": "resolved"}
        for key, value in resolution.items():
            if key in {"status", "scope", "time", "runtime_sha256"}:
                if key == "runtime_sha256":
                    _sha256_value(value, f"{recovery_id}.runtime_sha256")
                continue
            if isinstance(value, bool):
                safe_resolution[key] = value
            else:
                raise ValueError(f"{recovery_id} resolution has an unexpected public scalar")

        snapshots = {}
        for alias, filename in spec["snapshots"].items():
            path = root / filename
            if not path.is_file():
                raise FileNotFoundError(f"{recovery_id} source snapshot is missing")
            snapshots[alias] = {"filename": filename, "sha256": _sha(path)}
        rows.append({"recovery_id": recovery_id,
                     "capture": safe_capture, "resolution": safe_resolution,
                     "source_snapshot_sha256": snapshots})
    if len(rows) != 4:
        raise ValueError("the fixed four-record recovery catalog is incomplete")
    return {"schema": "stvg-matched-support-public-recovery-readback-v1",
            "status": "pass", "count": 4, "records": rows}


def _scan_public_text(data: bytes, label: str) -> None:
    lower = data.lower()
    slash = b"/"
    forbidden = [b"/home" + slash + b"wwww", slash + b"tmp" + slash,
                 b"source" + b"_cases" + slash,
                 b"private" + b"_inputs" + slash,
                 b"private" + b"_views" + slash,
                 b"source" + b"_case_path", b"query" + b"_text",
                 b"raw" + b"_output"]
    for token in forbidden:
        if token in lower:
            raise ValueError(f"public document contains a forbidden private token: {label}")
    for cid in CASE_IDS:
        if cid.encode("ascii") in data:
            raise ValueError(f"public document contains an unaliased case ID: {label}")


def _assert_close(left: Any, right: Any, label: str, tolerance: float = 2e-8) -> None:
    a = float(_finite_number(left))
    b = float(_finite_number(right))
    if not math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"independent public scalar differs from root report: {label}")


def _compare_root_and_independent(report: Mapping[str, Any],
                                  independent: Mapping[str, Any]) -> None:
    roots = {(row["case_id"], row["arm"]): row for row in report["rows"]}
    audits = {(row["case_id"], row["arm"]): row for row in independent["metrics"]}
    if roots.keys() != audits.keys() or len(roots) != 80:
        raise ValueError("independent metric rows do not cover the root report identities")
    for key, root_row in roots.items():
        audit = audits[key]
        if (root_row["dataset"] != audit["dataset"]
                or root_row["baseline"] != audit["baseline"]):
            raise ValueError("independent metric row identity/baseline differs")
        for metric in METRICS:
            _assert_close(root_row["metrics"][metric], audit["metrics"][metric],
                          f"{key}.{metric}")
        for field in ("delta_vIoU_pp", "official_independent_max_abs_error"):
            _assert_close(root_row[field], audit[field], f"{key}.{field}")
    root_groups = {(row["dataset"], row["arm"]): row for row in report["groups"]}
    audit_groups = {(row["dataset"], row["arm"]): row for row in independent["groups"]}
    if root_groups.keys() != audit_groups.keys() or len(root_groups) != 20:
        raise ValueError("independent metric group identities differ from the root report")
    for key, root_group in root_groups.items():
        audit = audit_groups[key]
        for field in ("count", "positive", "negative", "neutral"):
            if root_group[field] != audit[field]:
                raise ValueError(f"independent metric group differs: {key}.{field}")
        for field in ("mean_vIoU", "mean_delta_pp"):
            _assert_close(root_group[field], audit[field], f"{key}.{field}")
    root_quality = {(row["case_id"], row["role"], row["slot"]): row
                    for row in report["reference_quality"]}
    audit_quality = {(row["case_id"], row["role"], row["slot"]): row
                     for row in independent["reference_quality"]}
    if root_quality.keys() != audit_quality.keys() or len(root_quality) != 76:
        raise ValueError("independent reference-quality identities differ from root")
    for key, root_row in root_quality.items():
        audit = audit_quality[key]
        if (root_row["dataset"] != audit["dataset"]
                or root_row["candidate_id"] != audit["candidate_id"]
                or root_row["offline_label_assisted"] != audit["offline_label_assisted"]):
            raise ValueError(f"independent reference-quality identity differs: {key}")
        _assert_close(root_row["reference_IoU"], audit["reference_IoU"],
                      f"{key}.reference_IoU")
    root_moves = {(row["case_id"], row["role"]): row for row in report["fit_movement"]}
    audit_moves = {(row["case_id"], row["role"]): row for row in independent["fit_movement"]}
    if root_moves.keys() != audit_moves.keys() or len(root_moves) != 28:
        raise ValueError("independent fit-movement identities differ from root")
    for key, root_row in root_moves.items():
        audit = audit_moves[key]
        if root_row["optimizer_steps"] != audit["optimizer_steps"]:
            raise ValueError(f"independent fit-movement step count differs: {key}")
        for field in ("parameter_delta_norm", "mean_reference_IoU_before",
                      "mean_reference_IoU_after", "reference_IoU_delta_pp"):
            _assert_close(root_row[field], audit[field], f"{key}.{field}")
    if any(float(independent[key]) > 2e-8 for key in
           ("max_metric_scalar_abs_error", "max_reference_iou_abs_error",
            "max_fit_movement_scalar_abs_error")):
        raise ValueError("independent arithmetic readback exceeds the locked scalar tolerance")


def _summary_inputs_complete() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    required = ["STUDENT_GLOBAL_BARRIER.json", "ROOT_STUDENT_MATH_READBACK.json",
                "ROOT_POSTSEAL_REPORT.json", "ROOT_POSTSEAL_INDEPENDENT_READBACK.json",
                "SUPPORT_READBACK.json"] + list(SUMMARY_INPUTS.values())
    missing = [name for name in required if not (BASE / name).is_file()]
    if missing:
        raise FileNotFoundError("public export inputs are not complete: " + ", ".join(missing))
    barrier = _read_json(BASE / "STUDENT_GLOBAL_BARRIER.json")
    math = _read_json(BASE / "ROOT_STUDENT_MATH_READBACK.json")
    root_report = _read_json(BASE / "ROOT_POSTSEAL_REPORT.json")
    support_source = _read_json(BASE / "SUPPORT_READBACK.json")
    independent = _read_json(BASE / "ROOT_POSTSEAL_INDEPENDENT_READBACK.json")
    if (barrier.get("status") != "complete" or barrier.get("count") != 8
            or barrier.get("GT_read") is not False
            or barrier.get("offline_diagnostic_only") is not True):
        raise ValueError("student barrier is not complete and correctly scoped")
    if math.get("status") != "pass" or math.get("GT_read") is not True:
        raise ValueError("independent math readback is not complete")
    if independent.get("status") != "pass" or independent.get("GT_read") is not True:
        raise ValueError("independent postseal readback is not complete")
    support = _support_projection(support_source)
    report = _postseal_projection(root_report, support)
    math_public = _math_projection(math)
    independent_public = _independent_projection(independent)
    if (report["student_barrier_sha256"] != _sha(BASE / "STUDENT_GLOBAL_BARRIER.json")
            or math.get("student_barrier_sha256") != _sha(BASE / "STUDENT_GLOBAL_BARRIER.json")
            or report["root_math_sha256"] != _sha(BASE / "ROOT_STUDENT_MATH_READBACK.json")
            or independent_public["metric_count"] != report["metric_count"]
            or independent_public["reference_quality_count"] != report["reference_quality_count"]):
        raise ValueError("public projections are not bound to the same sealed outputs")
    _compare_root_and_independent(report, independent_public)
    return support, report, math_public, independent_public


def _validate_launched_revision(report: Mapping[str, Any],
                                binding_rows: list[dict[str, Any]]) -> dict[str, Any]:
    receipt_path = BASE / "ROOT_LAUNCHED_REVISION_READBACK.json"
    receipt = _read_json(receipt_path)
    expected = {"status", "scope", "runtime_sha256", "reference_lock_sha256",
                "worker_sha256", "actual_fit_count", "actual_full_redecode_count",
                "previous_prelaunch_receipts_historical", "time"}
    _exact_keys(receipt, expected, "launched-revision readback")
    if (receipt["status"] != "pass"
            or _integer(receipt["actual_fit_count"], "revision.actual_fit_count") != 28
            or _integer(receipt["actual_full_redecode_count"],
                        "revision.actual_full_redecode_count") != 28
            or _boolean(receipt["previous_prelaunch_receipts_historical"],
                        "revision.previous_prelaunch_receipts_historical") is not True):
        raise ValueError("launched-revision receipt is incomplete")
    _sha256_value(receipt["runtime_sha256"], "revision.runtime_sha256")
    _sha256_value(receipt["reference_lock_sha256"], "revision.reference_lock_sha256")
    worker_sha = _sha256_value(receipt["worker_sha256"], "revision.worker_sha256")
    if receipt["runtime_sha256"] != _sha(BASE / RUNTIME_NAME):
        raise ValueError("launched-revision receipt does not bind the current runtime")
    if receipt["reference_lock_sha256"] != report["reference_lock_sha256"]:
        raise ValueError("launched-revision receipt does not bind the scored reference lock")
    worker_binding = [row for row in binding_rows if row.get("published") is True
                      and row.get("binding") == "pin"
                      and row.get("source") == "scripts/run_stvg_matched_support_reference_v1.py"]
    if not worker_binding or any(row["sha256"] != worker_sha for row in worker_binding):
        raise ValueError("launched-revision worker differs from the current public runtime pin")
    if report["actual_fit_count"] != receipt["actual_fit_count"]:
        raise ValueError("launched-revision fit count differs from the scored report")
    return {"schema": "stvg-matched-support-public-launched-revision-v1",
            "status": "pass", "runtime_sha256": receipt["runtime_sha256"],
            "reference_lock_sha256": receipt["reference_lock_sha256"],
            "worker_sha256": worker_sha, "actual_fit_count": 28,
            "actual_full_redecode_count": 28,
            "previous_prelaunch_receipts_historical": True}


def _save_projection(name: str, value: Any) -> str:
    path = OUT / name
    _write_new(path, value)
    return path.relative_to(ROOT).as_posix()


def _stage_bytes(paths: set[str]) -> list[dict[str, Any]]:
    PUBLIC_STAGE.mkdir(parents=True, exist_ok=False)
    manifest = []
    for rel in sorted(paths):
        _safe_rel(rel)
        source = ROOT / rel
        if not source.is_file():
            raise FileNotFoundError(f"allowlisted public file is missing: {rel}")
        data = source.read_bytes()
        if rel in {REPORT_PATH.relative_to(ROOT).as_posix(), PDF_PATH.relative_to(ROOT).as_posix()}:
            _scan_public_text(data, rel)
        target = PUBLIC_STAGE / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest.append({"path": rel, "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest(),
                         "blob_sha": hashlib.sha1(
                             b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest(),
                         "mode": "100644", "type": "blob",
                         "encoding": "base64" if source.suffix.lower() == ".pdf" else "utf-8"})
    return manifest


def main() -> None:
    if OUT.exists() or PUBLIC_STAGE.exists() or MANIFEST_PATH.exists() or SOURCE_READBACK_PATH.exists():
        raise FileExistsError("public export outputs already exist; refusing overwrite")
    support, report, math_public, independent_public = _summary_inputs_complete()
    runtime_paths, binding_rows = _source_binding_rows()
    revision_public = _validate_launched_revision(report, binding_rows)
    roots_public = _summary_projection()
    revision_summary = roots_public["summaries"]["launched_revision"]
    if any(revision_summary.get(key) != value for key, value in revision_public.items()
           if key != "schema"):
        raise ValueError("safe current-launched-revision summary differs from its bound receipt")
    recovery_public = _recovery_projection()

    # Fixed public source allowlist plus byte-exact source files required by the
    # runtime closure. The raw roster, reference lock and student checkpoints
    # are deliberately absent from this list.
    fixed_sources = {
        "scripts/contract_stvg_matched_support_reference_v1.py",
        "scripts/launch_stvg_matched_support_reference_v1.py",
        "scripts/prepare_stvg_matched_support_reference_v1.py",
        "scripts/run_stvg_matched_support_reference_v1.py",
        "scripts/review_stvg_matched_support_reference_v1.py",
        "scripts/report_stvg_matched_support_reference_v1.py",
        "scripts/score_report_stvg_matched_support_reference_v1.py",
        "scripts/prepare_stvg_matched_support_public_v1.py",
        "scripts/stvg_matched_support_public_transfer_v1.py",
        "protocols/stvg_matched_support_reference_v1.md",
    }
    source_paths = set(runtime_paths) | fixed_sources
    if "methods/CURRENT_METHOD.json" in source_paths:
        raise ValueError("the deployed method registry must remain protected")
    for rel in source_paths:
        if not _is_public_source(rel):
            raise ValueError(f"non-public source path entered the export allowlist: {rel}")

    public_documents = {
        REPORT_PATH.relative_to(ROOT).as_posix(),
        PDF_PATH.relative_to(ROOT).as_posix(),
        PROTOCOL_PATH.relative_to(ROOT).as_posix(),
    }
    staged_inputs = source_paths | public_documents
    missing_sources = sorted(rel for rel in staged_inputs if not (ROOT / rel).is_file())
    if missing_sources:
        raise FileNotFoundError("public allowlist input is missing: " + ", ".join(missing_sources))
    for rel in public_documents:
        _scan_public_text((ROOT / rel).read_bytes(), rel)

    OUT.mkdir(parents=True, exist_ok=False)
    public_support_path = _save_projection("support_availability.json", support)
    public_report_path = _save_projection("postseal_metrics.json", report)
    public_math_path = _save_projection("student_math_readback.json", math_public)
    public_independent_path = _save_projection(
        "independent_postseal_readback.json", independent_public)
    public_roots_path = _save_projection("root_readback_summaries.json", roots_public)
    public_recovery_path = _save_projection("recovery_metadata.json", recovery_public)
    binding_summary = {"schema": "stvg-matched-support-public-source-bindings-v1",
                       "status": "pass", "runtime": RUNTIME_NAME,
                       "runtime_sha256": _sha(BASE / RUNTIME_NAME),
                       "binding_count": len(binding_rows),
                       "public_source_count": sum(bool(row["published"]) for row in binding_rows),
                       "private_hash_only_count": sum(not row["published"] for row in binding_rows),
                       "bindings": binding_rows}
    public_binding_path = _save_projection("runtime_source_bindings.json", binding_summary)

    scalar_readback = {
        "schema": "stvg-matched-support-public-scalar-projection-v1",
        "status": "pass", "scope": "anonymous public scalar projection",
        "case_aliases": sorted(CASE_ALIAS.values()),
        "counts": {"support_windows": 32, "included": support["included"],
                   "excluded": support["excluded"], "metric_rows": len(report["rows"]),
                   "groups": len(report["groups"]),
                   "reference_quality_rows": len(report["reference_quality"]),
                   "fit_movement_rows": len(report["fit_movement"])},
        "sha256": {"support": hashlib.sha256(json.dumps(
            support, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "postseal": hashlib.sha256(json.dumps(
                report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "math": hashlib.sha256(json.dumps(
                math_public, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "independent": hashlib.sha256(json.dumps(
                independent_public, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()},
        "GT_read": True, "offline_diagnostic_only": True,
        "GT_used_for_oracle_reference": True,
        "GT_used_for_method_or_hyperparameter_selection": False,
    }
    public_scalar_path = _save_projection("public_scalar_readback.json", scalar_readback)

    paths = source_paths | {
        REPORT_PATH.relative_to(ROOT).as_posix(),
        PDF_PATH.relative_to(ROOT).as_posix(),
        public_support_path, public_report_path, public_math_path,
        public_independent_path, public_roots_path, public_recovery_path, public_binding_path,
        public_scalar_path,
    }
    manifest = _stage_bytes(paths)
    _write_new(MANIFEST_PATH, manifest)
    source_readback = {
        "schema": "stvg-matched-support-public-source-readback-v1",
        "status": "pass", "files": len(manifest),
        "bytes": sum(row["bytes"] for row in manifest),
        "source_binding_count": len(binding_rows),
        "source_binding_sha256": _sha(OUT / "runtime_source_bindings.json"),
        "manifest_sha256": _sha(MANIFEST_PATH),
        "raw_reference_lock_staged": False, "raw_roster_staged": False,
        "student_checkpoints_staged": False, "GT_boxes_staged": False,
        "RGB_or_video_staged": False, "queries_or_teacher_text_staged": False,
        "recovery_receipts_projected": 4,
        "source_snapshot_sha_only_count": sum(
            len(row["source_snapshot_sha256"]) for row in recovery_public["records"]),
        "push_performed": False,
    }
    _write_new(SOURCE_READBACK_PATH, source_readback)
    print(json.dumps({"status": "staged_not_pushed", "files": len(manifest),
                      "bytes": source_readback["bytes"],
                      "manifest_sha256": source_readback["manifest_sha256"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
