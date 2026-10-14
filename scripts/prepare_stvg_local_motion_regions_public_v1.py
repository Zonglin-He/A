#!/usr/bin/env python3
"""Stage an explicit, anonymized public projection of the local-motion result.

No staging occurs by default. `--self-test` is synthetic-only. `--stage` needs
both a passing visual-review receipt and a passing independent readback. The
exporter reads only the sealed scalar geometry result, source/config metadata,
and review receipts; it never opens case JSON, masks, RGB, annotations, or a
raw runtime copy for publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_local_motion_regions_v1"
PUBLIC_STAGE = BASE / "public_stage"
MANIFEST_PATH = BASE / "PUBLIC_MANIFEST.json"
SOURCE_READBACK_PATH = BASE / "ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json"
REPORT_PATH = ROOT / "docs/STVG_LOCAL_MOTION_REGIONS_V1.md"

RUNTIME_PATH = BASE / "RUNTIME.json"
BARRIER_PATH = BASE / "CPU_GENERATOR_BARRIER.json"
GEOMETRY_PATH = BASE / "ROOT_POSTSEAL_GEOMETRY.json"
FIGURE_EXPORT_PATH = BASE / "FIGURE_EXPORT.json"
VISUAL_REVIEW_DEFAULT = BASE / "ROOT_VISUAL_REVIEW.json"
TECHNICAL_DECISION_PATH = BASE / "TECHNICAL_DECISION.json"
INDEPENDENT_REVIEW_DEFAULT = BASE / "ROOT_INDEPENDENT_READBACK002.json"
HISTORICAL_INDEPENDENT_FAILURE = BASE / "ROOT_INDEPENDENT_READBACK.json"
EXPECTED_INDEPENDENT_SHA256 = "878ca9511255e343f6e7b8d2f624a9421c7e4de0f9698cc40a3f6477685f7d49"
RESULTS_REL = "results/stvg_local_motion_regions/2026-10-14"

SCHEMA = "stvg-local-motion-regions-public-export-v1"
CASE_IDS = tuple([f"hc2_{i:03d}" for i in range(4)]
                 + [f"vidstg_{i:03d}" for i in range(4)])
CASE_ALIAS = {
    cid: ("HC" if cid.startswith("hc2_") else "V") + f"{int(cid[-3:]) + 1:02d}"
    for cid in CASE_IDS
}
ALIAS_ORDER = {CASE_ALIAS[cid]: i for i, cid in enumerate(CASE_IDS)}

SOURCE_FILES = (
    "docs/STVG_LOCAL_MOTION_REGIONS_V1.md",
    "scripts/prepare_stvg_local_motion_regions_public_v1.py",
    "scripts/run_stvg_local_motion_regions_v1.py",
    "scripts/test_stvg_local_motion_regions_cpu_v1.py",
    "scripts/score_stvg_local_motion_regions_v1.py",
    "scripts/render_stvg_local_motion_regions_v1.py",
    "scripts/review_stvg_local_motion_regions_v1.py",
    "scripts/run_stvg_short_event_tracklets_v1.py",
    "scripts/prepare_stvg_short_event_tracklets_v1.py",
    "scripts/stvg_adjacent_affine_tracklet_v1.py",
    "protocols/stvg_local_motion_region_candidate_v1.proposal.md",
    "protocols/stvg_short_event_tracklets_execution_v1.md",
)
RUNTIME_PINNED_PUBLIC = {
    "scripts/run_stvg_local_motion_regions_v1.py",
    "scripts/test_stvg_local_motion_regions_cpu_v1.py",
    "protocols/stvg_local_motion_region_candidate_v1.proposal.md",
}
PUBLIC_CHART_REL = "artifacts/stvg_local_motion_regions_v1/figures/local_motion_regions.png"
RECOVERY_SNAPSHOTS = {
    "rejected_component_mask001": ("run_stvg_local_motion_regions_v1.py",),
    "postseal_center_dispatch002": ("score_stvg_local_motion_regions_v1.py",),
}
INDEPENDENT_SCHEMA_RECOVERY = "independent_schema_dispatch003"
DENIED_PATH_PARTS = {
    "cases", "private_masks", "private_views", "private_rgb", "source_cases",
    "annotations", "private_inputs", "source_cache", "student", "teacher",
}
ABSOLUTE_PRIVATE_PATH = re.compile(
    r"(?:/" + "home/" + r"|/" + "tmp/" + r"|/" + "mnt/" + r")[^\s\"']+"
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _git_mode(path: Path) -> str:
    executable = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    return "100755" if path.stat().st_mode & executable else "100644"


def _encoding_for_path(path: Path) -> str:
    return "utf-8" if path.suffix.lower() in {".py", ".md", ".txt", ".json"} else "base64"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha_bytes(raw)


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _safe_source_path(rel: str) -> Path:
    if not isinstance(rel, str) or not rel or "\\" in rel:
        raise ValueError("source path must be a nonempty ROOT-relative POSIX path")
    path = Path(rel)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("absolute or traversing source paths are refused")
    if any(part in DENIED_PATH_PARTS for part in path.parts):
        raise ValueError("private payload path is not eligible for publication")
    resolved = (ROOT / path).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError("source path escaped project root")
    return resolved


def _case_alias(raw: Any) -> str:
    if raw not in CASE_ALIAS:
        raise ValueError("sealed result contains a case outside the fixed eight-case roster")
    return CASE_ALIAS[raw]


def _dataset_alias(raw: Any) -> str:
    if raw == "hc2":
        return "HC"
    if raw == "vidstg":
        return "VidSTG"
    raise ValueError("sealed result contains an unknown dataset label")


def _load_sealed_inputs() -> tuple[dict[str, Any], ...]:
    runtime = _read_json(RUNTIME_PATH)
    barrier = _read_json(BARRIER_PATH)
    geometry = _read_json(GEOMETRY_PATH)
    figure_export = _read_json(FIGURE_EXPORT_PATH)
    visual = _read_json(VISUAL_REVIEW_DEFAULT)
    decision = _read_json(TECHNICAL_DECISION_PATH)

    if runtime.get("schema") != "stvg-local-motion-region-runtime-v1" or runtime.get("status") != "locked":
        raise ValueError("generator runtime is not locked")
    if runtime.get("case_count") != 8 or runtime.get("window_count") != 32:
        raise ValueError("runtime does not bind the fixed 8-case/32-window set")
    for key in ("GT_read", "GPU_used", "model_loaded", "teacher_called", "scoring_started"):
        if runtime.get(key) is not False:
            raise ValueError(f"generator runtime must declare {key}=false")
    config = runtime.get("config")
    if not isinstance(config, dict) or _canonical_sha(config) != runtime.get("config_sha256"):
        raise ValueError("locked generator config hash does not match its declared value")

    if barrier.get("status") != "complete" or barrier.get("schema") != "stvg-local-motion-region-cpu-barrier-v1":
        raise ValueError("CPU generator barrier is not complete")
    if barrier.get("case_count") != 8 or barrier.get("window_count") != 32:
        raise ValueError("CPU barrier does not contain the fixed eight cases and 32 windows")
    for key in ("GT_read", "GPU_used", "model_loaded", "teacher_called", "scoring_started", "video_decoded"):
        if barrier.get(key) is not False:
            raise ValueError(f"CPU generator barrier must declare {key}=false")
    if barrier.get("RGB_read") is not True:
        raise ValueError("CPU generator barrier does not declare the authorized cached-RGB read")
    barrier_files = barrier.get("files")
    if not isinstance(barrier_files, list) or len(barrier_files) != 16:
        raise ValueError("CPU barrier must bind exactly eight case summaries and eight private mask archives")
    for row in barrier_files:
        if not isinstance(row, dict) or not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64:
            raise ValueError("CPU barrier has a malformed opaque file hash")
        # Paths are validated but never emitted and the files themselves are never opened.
        rel = row.get("path")
        if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError("CPU barrier has an invalid relative file binding")
        if not rel.startswith("artifacts/stvg_local_motion_regions_v1/"):
            raise ValueError("CPU barrier file escapes its declared scope")

    if geometry.get("status") != "pass" or geometry.get("window_count") != 32:
        raise ValueError("post-seal geometry readback is not a complete pass")
    if geometry.get("candidate_count") != 221 or geometry.get("new_nonSource_count") != 37:
        raise ValueError("post-seal result differs from the locked 221-candidate/37-new-region count")
    if geometry.get("unknown_window_count") != 9 or geometry.get("actual_OPD_fits") != 0:
        raise ValueError("post-seal result has unexpected unknown or OPD counts")
    if geometry.get("teacher_calls") != 0 or geometry.get("GPU_used") is not False:
        raise ValueError("post-seal evaluation used a teacher or GPU")
    if geometry.get("GT_read") is not True or geometry.get("GT_used_for_generation") is not False:
        raise ValueError("post-seal scoring provenance is inconsistent")
    if geometry.get("runtime_sha256") != _sha(RUNTIME_PATH):
        raise ValueError("post-seal geometry readback is bound to another runtime")
    if geometry.get("barrier_sha256") != _sha(BARRIER_PATH):
        raise ValueError("post-seal geometry readback is bound to another generator barrier")

    windows = geometry.get("windows")
    candidates = geometry.get("candidate_geometry")
    groups = geometry.get("groups")
    if not isinstance(windows, list) or len(windows) != 32:
        raise ValueError("post-seal geometry readback must contain all 32 window rows")
    if not isinstance(candidates, list) or len(candidates) != 221:
        raise ValueError("post-seal geometry readback must contain all 221 candidate rows")
    if not isinstance(groups, list) or len(groups) != 6:
        raise ValueError("post-seal geometry readback must contain all six groups")
    if sum(row.get("center_GT_available") is True for row in windows) != 23:
        raise ValueError("the fixed known/unknown support split is not 23/9")
    if sum(row.get("primary_support") is True for row in windows) != 19:
        raise ValueError("primary support is not the fixed 19 centers")
    if sum(row.get("pool") == "old" for row in candidates) != 152:
        raise ValueError("old candidate count is not 152")
    if sum(row.get("pool") == "new" for row in candidates) != 69:
        raise ValueError("new candidate count is not 69")
    if sum(row.get("pool") == "new" and row.get("candidate_id", 0) > 1 for row in candidates) != 37:
        raise ValueError("new non-Source candidate count is not 37")
    if sum(row.get("pool") == "new" and row.get("candidate_id") == 1
           for row in candidates) != 32:
        raise ValueError("new pool does not preserve exactly 32 Source tracks")
    if sum(int(row.get("supported_frame_count", 0)) for row in candidates
           if row.get("pool") == "new" and row.get("candidate_id", 0) > 1) != 170:
        raise ValueError("new motion-region supported-box count is not 170")
    if sum(int(row.get("supported_frame_count", 0)) for row in candidates
           if row.get("pool") == "new" and row.get("candidate_id") == 1) != 154:
        raise ValueError("copied Source supported-box count is not 154")

    if figure_export.get("status") != "exported_pending_actual_pixel_view":
        raise ValueError("figure export receipt has an unexpected state")
    files = figure_export.get("files")
    if not isinstance(files, list):
        raise ValueError("figure export file list is malformed")
    public_figures = [row for row in files if isinstance(row, dict) and row.get("path") == PUBLIC_CHART_REL]
    if len(public_figures) != 1:
        raise ValueError("figure export must bind exactly the one public scalar chart")
    chart_row = public_figures[0]
    chart_path = ROOT / PUBLIC_CHART_REL
    if not chart_path.is_file() or _sha(chart_path) != chart_row.get("sha256"):
        raise ValueError("public scalar chart does not match its figure receipt")

    if visual.get("status") != "pass" or visual.get("all_pixels_actually_viewed") is not True:
        raise ValueError("root visual-review receipt is not a pass")
    if visual.get("private_contact_count") != 8 or visual.get("frame_count") != 160:
        raise ValueError("visual review did not cover the locked eight sheets and 160 frames")
    if visual.get("public_chart_count") != 1 or visual.get("private_media_public") is not False:
        raise ValueError("visual review does not confirm the one-chart/no-private-media boundary")
    if visual.get("figure_export_sha256") != _sha(FIGURE_EXPORT_PATH):
        raise ValueError("visual review receipt is bound to another figure export")

    if decision.get("status") != "stop_fixed_local_motion_region_recipe_before_R1_OPD":
        raise ValueError("technical decision does not stop the fixed recipe before R1 OPD")
    if decision.get("teacher_calls") != 0 or decision.get("actual_OPD_fits") != 0:
        raise ValueError("technical decision reports teacher calls or OPD fits")
    if decision.get("CURRENT_promoted") is not False:
        raise ValueError("technical decision indicates method promotion")
    if decision.get("geometry_sha256") != _sha(GEOMETRY_PATH):
        raise ValueError("technical decision is bound to another geometry readback")

    return runtime, barrier, geometry, figure_export, visual, decision


def _project_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projected = []
    for row in rows:
        alias = _case_alias(row.get("case_id"))
        if row.get("pool") not in {"old", "new"}:
            raise ValueError("candidate geometry row has an unknown pool label")
        reason = row.get("unknown_reason")
        if reason not in {None, "no_center_GT", "no_valid_center_box"}:
            raise ValueError("candidate geometry row has an unapproved unknown reason")
        iou = row.get("IoU")
        if iou is not None and (not isinstance(iou, (int, float)) or not 0 <= float(iou) <= 1):
            raise ValueError("candidate center-IoU scalar is outside [0,1]")
        projected.append({
            "case_alias": alias,
            "dataset": _dataset_alias(row.get("dataset")),
            "window_slot_zero_based": int(row["slot"]),
            "pool": row["pool"],
            "candidate_id_local_to_window": int(row["candidate_id"]),
            "is_Source": bool(row["Source"]),
            "center_box_valid": bool(row["center_valid"]),
            "supported_frame_box_count": int(row["supported_frame_count"]),
            "center_box_IoU": None if iou is None else float(iou),
            "unknown_reason": reason,
        })
    projected.sort(key=lambda r: (ALIAS_ORDER[r["case_alias"]],
                                  r["window_slot_zero_based"],
                                  0 if r["pool"] == "old" else 1,
                                  r["candidate_id_local_to_window"]))
    return projected


def _project_windows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projected = []
    for row in rows:
        reason = row.get("unknown_reason")
        if reason not in {None, "center_has_no_GT"}:
            raise ValueError("window row has an unapproved unknown reason")
        projected.append({
            "case_alias": _case_alias(row.get("case_id")),
            "dataset": _dataset_alias(row.get("dataset")),
            "window_slot_zero_based": int(row["slot"]),
            "primary_support": bool(row["primary_support"]),
            "center_annotation_available": bool(row["center_GT_available"]),
            "old_candidate_count": int(row["old_candidate_count"]),
            "new_candidate_count": int(row["new_candidate_count"]),
            "new_motion_region_count": int(row["new_nonSource_count"]),
            "source_center_IoU": row["source_IoU"],
            "old_pool_oracle_center_IoU": row["old_oracle_IoU"],
            "new_pool_oracle_center_IoU": row["new_oracle_IoU"],
            "old_pool_minus_Source_pp": row["old_delta_pool_pp"],
            "new_pool_minus_Source_pp": row["new_delta_pool_pp"],
            "new_minus_old_pool_oracle_pp": row["new_minus_old_oracle_pp"],
            "new_motion_region_supported_boxes": int(row["new_nonSource_supported_frames"]),
            "unknown_reason": reason,
        })
    projected.sort(key=lambda r: (ALIAS_ORDER[r["case_alias"]], r["window_slot_zero_based"]))
    return projected


def _project_groups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projected = []
    for row in rows:
        if row.get("support") not in {"primary19", "all_known23"}:
            raise ValueError("group row has an unknown support label")
        dataset = row.get("dataset")
        if dataset not in {"all", "hc2", "vidstg"}:
            raise ValueError("group row has an unknown dataset")
        projected.append({
            "support": "primary_19" if row["support"] == "primary19" else "all_known_23",
            "dataset": {"all": "all", "hc2": "HC", "vidstg": "VidSTG"}[dataset],
            "count": int(row["count"]),
            "old_pool_minus_Source_mean_pp": float(row["old_mean_delta_pool_pp"]),
            "new_pool_minus_Source_mean_pp": float(row["new_mean_delta_pool_pp"]),
            "new_minus_old_pool_oracle_mean_pp": float(row["mean_new_minus_old_oracle_pp"]),
            "old_positive_pool_count": int(row["old_positive_pool_count"]),
            "new_positive_pool_count": int(row["new_positive_pool_count"]),
            "new_better_count": int(row["new_better_count"]),
            "new_worse_count": int(row["new_worse_count"]),
            "equal_count": int(row["equal_count"]),
        })
    return sorted(projected, key=lambda r: (
        0 if r["support"] == "primary_19" else 1,
        {"all": 0, "HC": 1, "VidSTG": 2}[r["dataset"]],
    ))


def _project_recovery() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for recovery_id, snapshot_names in RECOVERY_SNAPSHOTS.items():
        recovery_dir = BASE / "recovery" / recovery_id
        capture_path = recovery_dir / "CAPTURE.json"
        resolution_path = recovery_dir / "RESOLUTION.json"
        capture = _read_json(capture_path)
        resolution = _read_json(resolution_path)
        if capture.get("status") != "captured" or resolution.get("status") != "pass":
            raise ValueError(f"recovery record is not captured and resolved: {recovery_id}")
        if capture.get("GPU_used") is not False or resolution.get("GPU_used") is not False:
            raise ValueError(f"recovery path used a GPU: {recovery_id}")
        if resolution.get("geometry_sha256") != _sha(GEOMETRY_PATH):
            raise ValueError(f"recovery resolution binds another geometry result: {recovery_id}")
        if resolution.get("generator_barrier_sha256") != _sha(BARRIER_PATH):
            raise ValueError(f"recovery resolution binds another generator barrier: {recovery_id}")
        snapshot_rows = []
        for name in snapshot_names:
            snapshot_path = recovery_dir / name
            if not snapshot_path.is_file():
                raise FileNotFoundError(f"recovery source snapshot is missing: {recovery_id}/{name}")
            snapshot_rows.append({"filename": name, "sha256": _sha(snapshot_path)})
        cap_projection: dict[str, Any] = {
            "status": capture["status"],
            "GPU_used": False,
        }
        for key in ("reason", "scope"):
            if isinstance(capture.get(key), str):
                cap_projection[key] = capture[key]
        for key in ("GT_read", "teacher_called", "new_candidate_generation"):
            if isinstance(capture.get(key), bool):
                cap_projection[key] = capture[key]
        if isinstance(capture.get("case_outputs"), int):
            cap_projection["case_outputs"] = int(capture["case_outputs"])
        if isinstance(capture.get("source_sha256"), str):
            cap_projection["captured_source_sha256"] = capture["source_sha256"]
        res_projection = {
            "status": resolution["status"],
            "scope": resolution.get("scope"),
            "GPU_used": False,
            "generator_barrier_sha256": resolution["generator_barrier_sha256"],
            "geometry_sha256": resolution["geometry_sha256"],
        }
        records.append({"recovery_id": recovery_id,
                        "capture_sha256": _sha(capture_path),
                        "resolution_sha256": _sha(resolution_path),
                        "capture": cap_projection,
                        "resolution": res_projection,
                        "source_snapshot_sha256": snapshot_rows})
    recovery_dir = BASE / "recovery" / INDEPENDENT_SCHEMA_RECOVERY
    capture_path = recovery_dir / "CAPTURE.json"
    resolution_path = recovery_dir / "RESOLUTION.json"
    capture = _read_json(capture_path)
    resolution = _read_json(resolution_path)
    if (capture.get("schema") != "stvg-local-motion-initial-reader-failure-preservation-v1"
            or capture.get("status") != "preserved_failure_before_reader_schema_correction"
            or capture.get("failed_receipt_status") != "fail"
            or resolution.get("status") != "pass"):
        raise ValueError("independent schema recovery does not preserve the initial failure and resolved pass")
    original_failure_sha = _sha(HISTORICAL_INDEPENDENT_FAILURE)
    passing_readback_sha = _sha(INDEPENDENT_REVIEW_DEFAULT)
    if (capture.get("failed_receipt_sha256") != original_failure_sha
            or resolution.get("original_fail_sha256") != original_failure_sha
            or resolution.get("final_pass_sha256") != passing_readback_sha
            or passing_readback_sha != EXPECTED_INDEPENDENT_SHA256):
        raise ValueError("independent schema recovery receipt hashes do not bind the preserved failure and 002 pass")
    if (capture.get("failure_count") != 93 or capture.get("failed_check_count") != 302
            or resolution.get("initial_reader_failures") != 93
            or resolution.get("final_independent_checks") != 302
            or resolution.get("final_independent_failures") != 0
            or resolution.get("GT_loader_shared") is not True
            or resolution.get("candidate_regenerated") is not False
            or resolution.get("GPU_used") is not False):
        raise ValueError("independent schema recovery summary differs from its receipts")
    snapshot_name = "review_stvg_local_motion_regions_v1.initial.py"
    snapshot_path = recovery_dir / snapshot_name
    snapshot_sha = _sha(snapshot_path)
    if (capture.get("source_sha256") != snapshot_sha
            or resolution.get("source_snapshot_sha256") != snapshot_sha):
        raise ValueError("independent schema recovery source snapshot hash does not match")
    scope = resolution.get("scope")
    if not isinstance(scope, str) or ABSOLUTE_PRIVATE_PATH.search(scope):
        raise ValueError("independent schema recovery scope is not safe to publish")
    records.append({
        "recovery_id": INDEPENDENT_SCHEMA_RECOVERY,
        "capture_sha256": _sha(capture_path),
        "resolution_sha256": _sha(resolution_path),
        "capture": {
            "status": capture["status"],
            "failed_receipt_status": "fail",
            "failed_check_count": int(capture["failed_check_count"]),
            "failure_count": int(capture["failure_count"]),
            "failed_receipt_sha256": original_failure_sha,
        },
        "resolution": {
            "status": resolution["status"],
            "scope": scope,
            "initial_reader_failures": 93,
            "final_independent_checks": 302,
            "final_independent_failures": 0,
            "GT_loader_shared": True,
            "candidate_regenerated": False,
            "GPU_used": False,
            "original_fail_sha256": original_failure_sha,
            "final_pass_sha256": passing_readback_sha,
        },
        "source_snapshot_sha256": [{"filename": snapshot_name, "sha256": snapshot_sha}],
    })
    return {"schema": "stvg-local-motion-regions-recovery-metadata-v1",
            "status": "complete", "count": len(records), "records": records}


def _project_review_receipts(independent_path: Path, visual: Mapping[str, Any],
                             decision: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    independent = _read_json(independent_path)
    independent_sha = _sha(independent_path)
    if independent_path != INDEPENDENT_REVIEW_DEFAULT or independent_sha != EXPECTED_INDEPENDENT_SHA256:
        raise ValueError("independent readback is not the root-bound 002 receipt")
    checks = independent.get("checks")
    failures = independent.get("failures")
    counts = independent.get("counts")
    if (independent.get("status") != "pass" or not isinstance(checks, list)
            or len(checks) != 302 or not isinstance(failures, list) or failures
            or not isinstance(counts, dict)):
        raise ValueError("independent 002 receipt is not the complete 302-check pass")
    if any(not isinstance(row, dict) or row.get("status") != "pass" for row in checks):
        raise ValueError("independent 002 receipt contains a non-passing check")
    expected_counts = {
        "candidate_geometry_rows": 221,
        "windows": 32,
        "GT_known_windows": 23,
        "GT_unknown_windows": 9,
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("independent 002 receipt does not bind the expected scalar set")
    if (len(independent.get("candidate_iou_rows", [])) != 221
            or len(independent.get("window_candidate_geometry", [])) != 32
            or len(independent.get("group_aggregates", [])) != 6):
        raise ValueError("independent 002 receipt is incomplete for the sealed scalar result")
    historical = _read_json(HISTORICAL_INDEPENDENT_FAILURE)
    historical_sha = _sha(HISTORICAL_INDEPENDENT_FAILURE)
    if historical.get("status") != "fail":
        raise ValueError("the initial independent failure receipt must remain preserved as failed")
    independent_projection: dict[str, Any] = {
        "sha256": independent_sha,
        "status": independent["status"],
        "check_count": len(checks),
        "failure_count": len(failures),
        "candidate_iou_row_count": len(independent["candidate_iou_rows"]),
        "window_geometry_count": len(independent["window_candidate_geometry"]),
        "group_aggregate_count": len(independent["group_aggregates"]),
        "counts": {key: int(counts[key]) for key in sorted(expected_counts)},
    }
    if isinstance(independent.get("schema"), str):
        independent_projection["schema"] = independent["schema"]
    historical_projection = {
        "sha256": historical_sha,
        "status": "fail",
        "schema": historical.get("schema"),
        "superseded_by_readback_sha256": independent_sha,
        "counts": {
            key: int(historical.get("counts", {}).get(key, 0))
            for key in ("candidate_geometry_rows", "windows", "GT_known_windows", "GT_unknown_windows")
        },
    }
    visual_projection = {
        "sha256": _sha(VISUAL_REVIEW_DEFAULT),
        "status": visual["status"],
        "reviewed_contact_sheet_count": int(visual["private_contact_count"]),
        "reviewed_frame_count": int(visual["frame_count"]),
        "reviewed_public_chart_count": int(visual["public_chart_count"]),
        "all_pixels_actually_viewed": True,
        "private_media_exported": False,
    }
    decision_projection = {
        "sha256": _sha(TECHNICAL_DECISION_PATH),
        "status": decision["status"],
        "teacher_calls": int(decision["teacher_calls"]),
        "actual_OPD_fits": int(decision["actual_OPD_fits"]),
        "CURRENT_promoted": bool(decision["CURRENT_promoted"]),
        "reason": str(decision["reason"]),
    }
    return ({"schema": "stvg-local-motion-regions-review-readback-v1",
             "status": "pass", "visual": visual_projection,
             "independent": independent_projection,
             "historical_independent_failure": historical_projection,
             "technical_decision": decision_projection}, independent)


def _source_bindings(runtime: Mapping[str, Any], opaque_barrier_hashes: Sequence[str]) -> dict[str, Any]:
    pins = runtime.get("pins")
    if not isinstance(pins, dict):
        raise ValueError("runtime pins are not a mapping")
    runtime_pins: dict[str, str] = {}
    for rel, digest in pins.items():
        if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError("runtime contains an unsafe source-binding key")
        if isinstance(digest, Mapping):
            digest = digest.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("runtime contains a malformed SHA-256 pin")
        runtime_pins[rel] = digest.lower()
    rows = []
    for rel in SOURCE_FILES:
        path = _safe_source_path(rel)
        if not path.is_file():
            raise FileNotFoundError(f"public source file is missing: {rel}")
        digest = _sha(path)
        if rel in RUNTIME_PINNED_PUBLIC and runtime_pins.get(rel) != digest:
            raise ValueError(f"runtime-pinned public source changed: {rel}")
        text = path.read_text(encoding="utf-8") if path.suffix.lower() in {".py", ".md", ".txt"} else ""
        if ABSOLUTE_PRIVATE_PATH.search(text):
            raise ValueError(f"absolute local path found in public source: {rel}")
        rows.append({"path": rel, "sha256": digest, "bytes": path.stat().st_size,
                     "runtime_pinned": rel in RUNTIME_PINNED_PUBLIC})
    opaque_runtime_hashes = {
        digest for rel, digest in runtime_pins.items()
        if rel not in RUNTIME_PINNED_PUBLIC
    }
    opaque_hashes = sorted(opaque_runtime_hashes | set(opaque_barrier_hashes))
    return {
        "schema": "stvg-local-motion-regions-source-bindings-v1",
        "status": "pass",
        "runtime_sha256": _sha(RUNTIME_PATH),
        "generator_config_sha256": runtime["config_sha256"],
        "generator_barrier_sha256": _sha(BARRIER_PATH),
        "postseal_geometry_sha256": _sha(GEOMETRY_PATH),
        "matched_reference_lock_sha256": None,
        "GT_bindings_sha256": None,
        "source_files": rows,
        "opaque_input_and_lock_hashes": {
            "count": len(opaque_hashes), "sha256": opaque_hashes,
            "paths_or_payloads_published": False,
        },
    }


def _config_projection(runtime: Mapping[str, Any], barrier: Mapping[str, Any],
                       geometry: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "stvg-local-motion-regions-public-protocol-v1",
        "status": "sealed_and_scored",
        "generator_config": runtime["config"],
        "generator_config_sha256": runtime["config_sha256"],
        "generation": {
            "case_count": 8, "window_count": 32,
            "RGB_read_from_sealed_cache": barrier["RGB_read"],
            "video_decoded": barrier["video_decoded"],
            "GT_read": barrier["GT_read"], "GPU_used": barrier["GPU_used"],
            "model_loaded": barrier["model_loaded"],
            "teacher_called": barrier["teacher_called"],
            "scoring_started": barrier["scoring_started"],
            "sealed_file_count": len(barrier["files"]),
            "file_hashes_published_as_opaque": True,
        },
        "postseal_evaluation": {
            "GT_read": geometry["GT_read"],
            "GT_used_for_generation": geometry["GT_used_for_generation"],
            "GPU_used": geometry["GPU_used"],
            "teacher_calls": geometry["teacher_calls"],
            "actual_OPD_fits": geometry["actual_OPD_fits"],
            "known_center_count": geometry["window_count"] - geometry["unknown_window_count"],
            "unknown_center_count": geometry["unknown_window_count"],
            "scope": geometry["scope"],
        },
        "decision": {
            "status": decision["status"], "reason": decision["reason"],
            "CURRENT_promoted": decision["CURRENT_promoted"],
            "independent_32_per_direction_triggered": decision["independent_32_per_direction_triggered"],
        },
        "interpretation_limit": (
            "Center-frame candidate-pool oracle diagnostic only; it is not an online selector, "
            "full-tube efficacy result, or semantic identity metric."
        ),
    }


def _summary(groups: Sequence[Mapping[str, Any]], windows: Sequence[Mapping[str, Any]],
             candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "stvg-local-motion-regions-public-summary-v1",
        "status": "sealed_and_scored",
        "case_count": len({r["case_alias"] for r in windows}),
        "window_count": len(windows),
        "primary_support_count": sum(r["primary_support"] for r in windows),
        "known_center_count": sum(r["center_annotation_available"] for r in windows),
        "unknown_center_count": sum(not r["center_annotation_available"] for r in windows),
        "old_candidate_count": sum(r["pool"] == "old" for r in candidates),
        "new_candidate_count": sum(r["pool"] == "new" for r in candidates),
        "new_motion_region_count": sum(r["pool"] == "new" and not r["is_Source"] for r in candidates),
        "new_motion_region_supported_box_count": sum(
            r["supported_frame_box_count"] for r in candidates
            if r["pool"] == "new" and not r["is_Source"]),
        "source_supported_box_count": sum(
            r["supported_frame_box_count"] for r in candidates
            if r["pool"] == "new" and r["is_Source"]),
        "known_primary_new_better_worse_equal": {
            "better": sum(r["new_better_count"] for r in groups
                           if r["support"] == "primary_19" and r["dataset"] == "all"),
            "worse": sum(r["new_worse_count"] for r in groups
                          if r["support"] == "primary_19" and r["dataset"] == "all"),
            "equal": sum(r["equal_count"] for r in groups
                          if r["support"] == "primary_19" and r["dataset"] == "all"),
        },
        "all_groups": list(groups),
    }


def _review_gate(independent_path: Path) -> tuple[dict[str, Any], ...]:
    if not independent_path.is_file():
        raise FileNotFoundError("independent readback receipt is not present; public stage is deferred")
    inputs = _load_sealed_inputs()
    review_projection, _ = _project_review_receipts(independent_path, inputs[4], inputs[5])
    return inputs + (review_projection,)


def _stage() -> dict[str, Any]:
    if PUBLIC_STAGE.exists() or MANIFEST_PATH.exists() or SOURCE_READBACK_PATH.exists():
        raise FileExistsError("public export paths already exist; immutable staging will not overwrite")
    runtime, barrier, geometry, figure_export, visual, decision, review_projection = _review_gate(INDEPENDENT_REVIEW_DEFAULT)
    geometry_rows = geometry["candidate_geometry"]
    window_rows = geometry["windows"]
    group_rows = geometry["groups"]
    candidates = _project_candidates(geometry_rows)
    windows = _project_windows(window_rows)
    groups = _project_groups(group_rows)
    if len(candidates) != 221 or len(windows) != 32 or len(groups) != 6:
        raise ValueError("projected public scalar counts differ from the sealed source")

    recovery = _project_recovery()
    source_bindings = _source_bindings(runtime, [x["sha256"] for x in barrier["files"]])
    source_bindings["matched_reference_lock_sha256"] = geometry.get("matched_reference_lock_sha256")
    source_bindings["GT_bindings_sha256"] = geometry.get("GT_bindings_sha256")
    source_bindings["review_receipts"] = {
        "visual_review_sha256": _sha(VISUAL_REVIEW_DEFAULT),
        "independent_readback_sha256": review_projection["independent"]["sha256"],
        "technical_decision_sha256": _sha(TECHNICAL_DECISION_PATH),
        "figure_export_sha256": _sha(FIGURE_EXPORT_PATH),
    }

    chart_path = ROOT / PUBLIC_CHART_REL
    chart_row = next(row for row in figure_export["files"] if row.get("path") == PUBLIC_CHART_REL)
    if _sha(chart_path) != chart_row["sha256"]:
        raise ValueError("public chart changed after review")
    projections = {
        "public_summary.json": _summary(groups, windows, candidates),
        "candidate_geometry_readback.json": {
            "schema": "stvg-local-motion-regions-candidate-geometry-v1",
            "status": "complete", "count": len(candidates),
            "coordinates_published": False,
            "rows": candidates,
        },
        "window_readback.json": {
            "schema": "stvg-local-motion-regions-window-readback-v1",
            "status": "complete", "count": len(windows), "rows": windows,
        },
        "group_readback.json": {
            "schema": "stvg-local-motion-regions-group-readback-v1",
            "status": "complete", "count": len(groups), "rows": groups,
        },
        "protocol_and_config.json": _config_projection(runtime, barrier, geometry, decision),
        "recovery_metadata.json": recovery,
        "review_receipts.json": review_projection,
        "source_bindings.json": source_bindings,
    }

    stage_tmp = BASE / ".public_stage.pending"
    if stage_tmp.exists():
        raise FileExistsError("an unfinished public-stage directory already exists")
    stage_tmp.mkdir(parents=True)
    try:
        for rel in SOURCE_FILES:
            src = _safe_source_path(rel)
            target = stage_tmp / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
            shutil.copymode(src, target)
        for name, value in projections.items():
            _write_new(stage_tmp / RESULTS_REL / name, _json_bytes(value))
        figure_target = stage_tmp / PUBLIC_CHART_REL
        figure_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(chart_path, figure_target)

        forbidden = []
        for path in stage_tmp.rglob("*"):
            if not path.is_file():
                continue
            if any(part in DENIED_PATH_PARTS for part in path.relative_to(stage_tmp).parts):
                forbidden.append(path.relative_to(stage_tmp).as_posix())
                continue
            if path.suffix.lower() in {".py", ".md", ".json", ".txt"}:
                text = path.read_text(encoding="utf-8")
                if ABSOLUTE_PRIVATE_PATH.search(text):
                    forbidden.append(path.relative_to(stage_tmp).as_posix())
        if forbidden:
            raise ValueError(f"public stage contains a denied path or absolute private path: {forbidden}")

        stage_files = []
        for path in sorted(x for x in stage_tmp.rglob("*") if x.is_file()):
            rel = path.relative_to(stage_tmp).as_posix()
            if rel.startswith("public_stage/") or Path(rel).is_absolute():
                raise ValueError("manifest paths must be repository-relative")
            data = path.read_bytes()
            stage_files.append({
                "path": rel,
                "sha256": _sha_bytes(data),
                "bytes": len(data),
                "blob_sha": _git_blob_sha(data),
                "mode": _git_mode(path),
                "type": "blob",
                "encoding": _encoding_for_path(path),
            })
        if not stage_files or not any(row["path"].endswith("STVG_LOCAL_MOTION_REGIONS_V1.md") for row in stage_files):
            raise ValueError("public stage is missing the report")
        required_manifest_fields = {"path", "bytes", "sha256", "blob_sha", "mode", "encoding"}
        if any(not required_manifest_fields.issubset(row) for row in stage_files):
            raise ValueError("public manifest row is missing repository transfer metadata")
        required_paths = {
            f"{RESULTS_REL}/{name}" for name in projections
        } | {PUBLIC_CHART_REL} | set(SOURCE_FILES)
        if {row["path"] for row in stage_files} != required_paths:
            raise ValueError("public manifest path set differs from the explicit export whitelist")
        manifest = {
            "schema": SCHEMA, "status": "staged_pending_remote_review",
            "scope": "fixed local motion-region proposal readback; no method promotion",
            "stage_file_count": len(stage_files),
            "stage_bytes": sum(row["bytes"] for row in stage_files),
            "private_media_or_masks_published": False,
            "raw_case_or_runtime_payload_published": False,
            "gt_coordinates_published": False,
            "files": stage_files,
            "visual_review_sha256": _sha(VISUAL_REVIEW_DEFAULT),
            "independent_readback_sha256": review_projection["independent"]["sha256"],
            "historical_independent_failure_sha256": review_projection["historical_independent_failure"]["sha256"],
        }
        manifest_bytes = _json_bytes(manifest)
        os.replace(stage_tmp, PUBLIC_STAGE)
        _write_new(MANIFEST_PATH, manifest_bytes)

        readback = {
            "schema": "stvg-local-motion-regions-public-source-readback-v1",
            "status": "pass", "public_source_count": len(source_bindings["source_files"]),
            "private_hash_only_count": source_bindings["opaque_input_and_lock_hashes"]["count"],
            "stage_file_count": manifest["stage_file_count"],
            "stage_bytes": manifest["stage_bytes"],
            "public_manifest_sha256": _sha(MANIFEST_PATH),
            "visual_review_sha256": _sha(VISUAL_REVIEW_DEFAULT),
            "independent_readback_sha256": review_projection["independent"]["sha256"],
            "source_files": source_bindings["source_files"],
            "opaque_input_and_lock_hashes": source_bindings["opaque_input_and_lock_hashes"],
        }
        _write_new(SOURCE_READBACK_PATH, _json_bytes(readback))
        return {"manifest_sha256": _sha(MANIFEST_PATH),
                "source_readback_sha256": _sha(SOURCE_READBACK_PATH),
                "stage_file_count": manifest["stage_file_count"],
                "stage_bytes": manifest["stage_bytes"]}
    except Exception:
        # Preserve any partial stage for explicit review; do not silently erase it.
        raise


def _self_test() -> None:
    synthetic = [
        {"case_id": "hc2_000", "dataset": "hc2", "slot": 0, "pool": "new",
         "candidate_id": 1, "Source": True, "center_valid": True,
         "supported_frame_count": 5, "IoU": 0.5, "unknown_reason": None},
        {"case_id": "vidstg_003", "dataset": "vidstg", "slot": 3, "pool": "new",
         "candidate_id": 2, "Source": False, "center_valid": False,
         "supported_frame_count": 3, "IoU": None, "unknown_reason": "no_center_GT"},
    ]
    projected = _project_candidates(synthetic)
    checks = [
        [row["case_alias"] for row in projected] == ["HC01", "V04"],
        projected[1]["center_box_IoU"] is None,
        projected[1]["unknown_reason"] == "no_center_GT",
        "case_id" not in projected[0] and "IoU" not in projected[0],
    ]
    groups = _project_groups([
        {"support": "primary19", "dataset": "all", "count": 19,
         "old_mean_delta_pool_pp": 4.099918, "new_mean_delta_pool_pp": 1.401724,
         "mean_new_minus_old_oracle_pp": -2.698194,
         "old_positive_pool_count": 7, "new_positive_pool_count": 2,
         "new_better_count": 2, "new_worse_count": 7, "equal_count": 10},
    ])
    checks.append(groups[0]["support"] == "primary_19" and groups[0]["dataset"] == "all")
    checks.append(groups[0]["new_minus_old_pool_oracle_mean_pp"] < 0)
    checks.append((groups[0]["new_better_count"], groups[0]["new_worse_count"],
                   groups[0]["equal_count"]) == (2, 7, 10))
    checks.append(RESULTS_REL == "results/stvg_local_motion_regions/2026-10-14")
    checks.append(PUBLIC_CHART_REL == "artifacts/stvg_local_motion_regions_v1/figures/local_motion_regions.png")
    checks.append(_git_blob_sha(b"hello") == "b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0")
    checks.append(_encoding_for_path(Path("report.json")) == "utf-8")
    checks.append(_encoding_for_path(Path("chart.png")) == "base64")
    try:
        _case_alias("not-a-roster-id")
    except ValueError:
        checks.append(True)
    else:
        raise AssertionError("unrecognized case IDs must fail closed")
    assert all(checks)
    print(json.dumps({"status": "pass", "synthetic_checks": len(checks),
                      "GT_read": False, "GPU_used": False}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true",
                       help="run synthetic alias/projection contracts only")
    group.add_argument("--stage", action="store_true",
                       help="create the immutable public stage after both reviews pass")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    result = _stage()
    print(json.dumps({"status": "staged_pending_remote_review", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
