"""Independent CPU readback for the sealed short-window local/associated fits.

This reader waits for the complete eight-case student barrier, verifies all
locked inputs and file hashes before deserializing any prediction tensor, and
then checks saved Gaussian IoU rollout/Adam arithmetic. It does not load a
model, decode media, read labels, or recompute decoder gradients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / "artifacts/stvg_short_event_tracklets_v1"
OLD_T_BASE = ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1"
CASE_IDS = tuple([f"hc2_{i:03d}" for i in range(4)] + [f"vidstg_{i:03d}" for i in range(4)])
ROLES = ("local_R1", "associated_R1")
EXPECTED_FEATURE_SCHEMA = "stvg-short-window-frozen-appearance-v1"
EXPECTED_STUDENT_SCHEMA = "stvg-short-window-student-v1"
READBACK_PATH = BASE / "ROOT_STUDENT_MATH_READBACK.json"
FIT_AUDITOR = ROOT / "scripts/review_stvg_scene_object_id_spatial_math_v1.py"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_cohort_identity(student: Mapping[str, Any] | None, roster_case: Mapping[str, Any],
                              manifest_case: Mapping[str, Any]) -> None:
    """Bind the execution label to the roster and historical label to the manifest."""
    _require(roster_case.get("cohort") == "old8",
             "locked roster row is not the exact old8 cohort")
    _require(manifest_case.get("cohort") == "historically_exposed_old8",
             "prepared manifest row is not labeled as historically exposed old8")
    if student is not None:
        _require(student.get("cohort") == roster_case.get("cohort"),
                 "student cohort differs from its exact locked roster row")


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("expected a ROOT-relative POSIX path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("absolute or traversing artifact path is forbidden")
    lowered = "/" + value.casefold().strip("/") + "/"
    if any(token in lowered for token in ("/annotations/", "/ground_truth/", "/gt_labels/", "/labels/")):
        raise ValueError("label-bearing path is outside this reader's scope")
    return value


def _verify_file(relative: Any, digest: Any, byte_count: Any = None) -> Path:
    relative = _safe_relative(relative)
    _require(isinstance(digest, str) and len(digest) == 64
             and all(c in "0123456789abcdef" for c in digest),
             f"malformed file digest for {relative}")
    path = ROOT / relative
    _require(path.is_file() and _sha_file(path) == digest,
             f"locked file bytes changed: {relative}")
    if byte_count is not None:
        _require(path.stat().st_size == int(byte_count),
                 f"locked file byte count changed: {relative}")
    return path


def _xyxy(cxcywh: torch.Tensor) -> torch.Tensor:
    # Match the official worker's coordinate conversion: source cxcywh values
    # are promoted to float64 before arithmetic, so float32 rounding is not
    # introduced by this independent readback.
    values = cxcywh.detach().cpu().to(dtype=torch.float64)
    cx, cy, width, height = values.unbind(-1)
    return torch.stack((cx - width / 2, cy - height / 2,
                        cx + width / 2, cy + height / 2), dim=-1)


def _normalized_anchor_box(raw_box: Any) -> torch.Tensor:
    """Mirror the worker's double-bound check followed by fp32 fit conversion."""
    box64 = torch.as_tensor(raw_box, dtype=torch.float64, device="cpu")
    _require(box64.shape == (4,) and torch.isfinite(box64).all(),
             "sparse anchor box must be finite normalized cxcywh")
    cx0, cy0, width0, height0 = map(float, box64.tolist())
    _require(width0 > 0 and height0 > 0
             and 0 <= cx0 - width0 / 2 < cx0 + width0 / 2 <= 1
             and 0 <= cy0 - height0 / 2 < cy0 + height0 / 2 <= 1,
             "raw sparse anchor box is outside normalized image bounds")
    box = box64.to(torch.float32)
    cx, cy, width, height = map(float, box.tolist())
    _require(width > 0 and height > 0
             and -1e-7 <= cx - width / 2 < cx + width / 2 <= 1 + 1e-7
             and -1e-7 <= cy - height / 2 < cy + height / 2 <= 1 + 1e-7,
             "fp32 sparse anchor box exceeds the worker's boundary-rounding tolerance")
    return box


def _as_tensor(value: Any, name: str) -> torch.Tensor:
    try:
        result = torch.as_tensor(value)
    except Exception as exc:
        raise RuntimeError(f"{name} is not a tensor-like saved value") from exc
    _require(torch.isfinite(result).all(), f"{name} contains nonfinite values")
    return result.detach().cpu().contiguous()


def _exact_tensor(left: Any, right: Any, name: str) -> None:
    a, b = _as_tensor(left, f"{name}.left"), _as_tensor(right, f"{name}.right")
    _require(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b),
             f"{name} is not exact")


def _exact_sequence(left: Any, right: Any, name: str) -> None:
    _require(isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
             and type(left) is type(right) and len(left) == len(right),
             f"{name} sequence schema differs")
    for index, (a, b) in enumerate(zip(left, right)):
        _exact_tensor(a, b, f"{name}[{index}]")


def _same_interval(left: Any, right: Any, name: str) -> None:
    _require(isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
             and len(left) == len(right) and list(left) == list(right),
             f"{name} interval is not exact")


def _feature_records(barrier: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = barrier.get("files")
    _require(isinstance(rows, Mapping) and set(rows) == set(CASE_IDS),
             "feature barrier must bind one feature file per fixed case")
    output: dict[str, dict[str, Any]] = {}
    for cid in CASE_IDS:
        row = rows[cid]
        _require(isinstance(row, Mapping), f"malformed feature receipt for {cid}")
        expected = f"artifacts/stvg_short_event_tracklets_v1/features/{cid}.pt"
        _require(row.get("path") == expected, f"feature output path changed for {cid}")
        feature_path = _verify_file(row.get("path"), row.get("sha256"), row.get("bytes"))
        receipt = row.get("receipt")
        _require(isinstance(receipt, Mapping), f"feature sidecar receipt missing for {cid}")
        sidecar_rel = f"artifacts/stvg_short_event_tracklets_v1/features/{cid}.json"
        _require(receipt.get("path") == sidecar_rel,
                 f"feature sidecar path changed for {cid}")
        sidecar_path = _verify_file(receipt.get("path"), receipt.get("sha256"), receipt.get("bytes"))
        output[cid] = {"path": feature_path, "sidecar_path": sidecar_path,
                       "sha256": row["sha256"], "sidecar_sha256": receipt["sha256"]}
    return output


def _preflight() -> dict[str, Any]:
    """Validate complete sealed catalog and all hashes before torch.load."""
    if torch.cuda.is_initialized():
        raise RuntimeError("student math readback requires a CPU-only process")

    manifest_path = BASE / "PREPARED_MANIFEST.json"
    tracklet_path = BASE / "GLOBAL_TRACKLET_BARRIER.json"
    feature_lock_path = BASE / "FEATURE_CAPTURE_LOCK.json"
    feature_runtime_path = BASE / "FEATURE_RUNTIME.json"
    feature_barrier_path = BASE / "FEATURE_CAPTURE_BARRIER.json"
    teacher_job_path = BASE / "TEACHER_JOB_LOCK.json"
    teacher_barrier_path = BASE / "TEACHER_GLOBAL_BARRIER.json"
    roster_path = BASE / "ROSTER_LOCK.json"
    reference_path = BASE / "METHOD_REFERENCE_LOCK.json"
    runtime_path = BASE / "STUDENT_RUNTIME.json"
    barrier_path = BASE / "STUDENT_GLOBAL_BARRIER.json"

    barrier = _read_json(barrier_path)
    _require(barrier.get("schema") == "stvg-short-window-student-barrier-v1"
             and barrier.get("status") == "complete"
             and barrier.get("count") == 8
             and barrier.get("case_ids") == list(CASE_IDS)
             and barrier.get("GT_read") is False,
             "student global barrier must seal the exact no-GT eight-case catalog")
    runtime = _read_json(runtime_path)
    _require(runtime.get("status") == "locked" and runtime.get("GT_read") is False,
             "student runtime is not locked as no-GT")
    runtime_sha = _sha_file(runtime_path)
    _require(barrier.get("runtime_sha256") == runtime_sha,
             "student global barrier is not bound to the exact runtime bytes")

    manifest = _read_json(manifest_path)
    roster_doc = _read_json(roster_path)
    tracklet_barrier = _read_json(tracklet_path)
    feature_lock = _read_json(feature_lock_path)
    feature_runtime = _read_json(feature_runtime_path)
    feature_barrier = _read_json(feature_barrier_path)
    teacher_job_lock = _read_json(teacher_job_path)
    teacher_barrier = _read_json(teacher_barrier_path)
    reference_lock = _read_json(reference_path)

    _require(manifest.get("status") == "prepared_manifest_not_executed"
             and manifest.get("GT_read") is False,
             "prepared manifest must remain explicitly GT-free")
    manifest_rows = manifest.get("cases")
    _require(isinstance(manifest_rows, list)
             and [row.get("case_id") for row in manifest_rows] == list(CASE_IDS),
             "prepared manifest case roster changed")
    manifest_by_id = {row["case_id"]: row for row in manifest_rows}

    _require(roster_doc.get("status") == "locked" and roster_doc.get("GT_read") is False,
             "source roster is not locked no-GT")
    roster_rows = roster_doc.get("records")
    _require(isinstance(roster_rows, list)
             and [row.get("case_id") for row in roster_rows] == list(CASE_IDS),
             "source roster does not match fixed old-eight order")
    roster_by_id = {row["case_id"]: row for row in roster_rows}
    for cid in CASE_IDS:
        _validate_cohort_identity(None, roster_by_id[cid], manifest_by_id[cid])

    _require(tracklet_barrier.get("status") == "complete"
             and tracklet_barrier.get("count") == 8
             and tracklet_barrier.get("case_ids") == list(CASE_IDS)
             and tracklet_barrier.get("GT_read") is False,
             "tracklet barrier is incomplete or outside the no-GT source scope")
    _require(feature_lock.get("status") == "locked" and feature_lock.get("GT_read") is False
             and feature_runtime.get("status") == "locked"
             and feature_runtime.get("GT_read") is False,
             "feature capture lock/runtime are not locked no-GT")
    feature_lock_sha, feature_runtime_sha = _sha_file(feature_lock_path), _sha_file(feature_runtime_path)
    _require(feature_runtime.get("feature_lock_sha256", feature_runtime.get("lock_sha256")) == feature_lock_sha,
             "feature runtime does not bind its lock")
    _require(feature_barrier.get("status") in {"sealed", "complete"}
             and feature_barrier.get("count") == 8
             and feature_barrier.get("case_ids") == list(CASE_IDS)
             and feature_barrier.get("GT_read") is False,
             "feature barrier does not seal the fixed no-GT eight-case capture")
    _require(feature_barrier.get("manifest_sha256") == _sha_file(manifest_path)
             and feature_barrier.get("feature_lock_sha256") == feature_lock_sha
             and feature_barrier.get("feature_runtime_sha256") == feature_runtime_sha,
             "feature barrier does not bind exact manifest/lock/runtime bytes")
    feature_pin_map: dict[str, str] = {}
    for document in (feature_lock, feature_runtime):
        for map_name in ("pins", "implementation_pins", "protected"):
            rows = document.get(map_name, {})
            _require(isinstance(rows, Mapping), "feature pin map is malformed")
            for relative, digest in rows.items():
                if relative in feature_pin_map:
                    _require(feature_pin_map[relative] == digest,
                             f"feature lock/runtime pin conflict: {relative}")
                feature_pin_map[str(relative)] = str(digest)
    for relative, digest in feature_pin_map.items():
        _verify_file(relative, digest)
    worker_rel = "scripts/capture_stvg_short_window_features_v1.py"
    contract_rel = "artifacts/stvg_short_event_tracklets_v1/CPU_FEATURE_CAPTURE_CONTRACTS.json"
    _require(feature_barrier.get("worker_sha256") == feature_pin_map.get(worker_rel)
             and feature_barrier.get("cpu_contracts_sha256") == feature_pin_map.get(contract_rel),
             "feature barrier does not bind its exact worker and CPU contract source")

    _require(teacher_job_lock.get("status") == "locked"
             and teacher_job_lock.get("GT_read") is False
             and teacher_job_lock.get("count") == 64
             and isinstance(teacher_job_lock.get("records"), list)
             and len(teacher_job_lock["records"]) == 64,
             "teacher job lock must seal the fixed 64 no-GT requests")
    _require(teacher_barrier.get("status") in {"sealed", "complete"}
             and teacher_barrier.get("GT_read") is False,
             "teacher barrier is incomplete or GT-exposed")
    teacher_calls = teacher_barrier.get("count", teacher_barrier.get("call_count", len(teacher_barrier.get("files", []))))
    _require(teacher_calls == 64,
             "teacher barrier does not seal the fixed 64 calls")
    _require(reference_lock.get("schema") == "stvg-short-window-reference-lock-v1"
             and reference_lock.get("status") == "locked"
             and reference_lock.get("count") == 8
             and reference_lock.get("GT_read") is False,
             "method reference lock is not sealed no-GT for old-eight")
    references = reference_lock.get("records")
    _require(isinstance(references, list)
             and [row.get("case_id") for row in references] == list(CASE_IDS),
             "reference lock case roster differs from fixed old-eight")
    reference_by_id = {row["case_id"]: row for row in references}

    binding_sha = {
        "manifest_sha256": _sha_file(manifest_path),
        "tracklet_barrier_sha256": _sha_file(tracklet_path),
        "feature_barrier_sha256": _sha_file(feature_barrier_path),
        "teacher_global_barrier_sha256": _sha_file(teacher_barrier_path),
        "teacher_job_lock_sha256": _sha_file(teacher_job_path),
    }
    for key, value in binding_sha.items():
        _require(reference_lock.get(key) == value,
                 f"method reference lock has a stale {key}")
    _require(runtime.get("reference_lock_sha256") == _sha_file(reference_path),
             "student runtime does not bind the sealed method reference lock")

    # Validate every declared execution/source pin before any case tensor is loaded.
    pins, protected = runtime.get("pins"), runtime.get("protected")
    _require(isinstance(pins, Mapping) and isinstance(protected, Mapping),
             "student runtime pin maps are malformed")
    for relative in set(pins) & set(protected):
        _require(pins[relative] == protected[relative],
                 f"runtime pin/protected hash conflict: {relative}")
    combined_pins = {**pins, **protected}
    for relative in ("scripts/run_stvg_short_window_student_v1.py",
                     "scripts/stvg_video_global_spatial_fit_fp32_geometry001.py"):
        _require(relative in pins,
                 f"student runtime omits required execution source: {relative}")
    for relative, digest in combined_pins.items():
        _verify_file(relative, digest)

    expected_fit_count = 0
    for cid in CASE_IDS:
        manifest_case, roster_case, ref_case = manifest_by_id[cid], roster_by_id[cid], reference_by_id[cid]
        _require(manifest_case.get("GT_read") is False
                 and manifest_case.get("source_case_sha256") == roster_case.get("source_case_sha256")
                 and manifest_case.get("dataset") == roster_case.get("dataset") == ref_case.get("dataset"),
                 f"source roster/manifest/reference disagree for {cid}")
        for key in ("source_case_sha256", "source_model_state_sha256", "canonical_request_sha256",
                    "canonical_video_sha256"):
            _require(ref_case.get(key) == manifest_case.get(key),
                     f"reference lock is not bound to {cid}:{key}")
        refs = ref_case.get("references")
        _require(isinstance(refs, Mapping) and set(refs) == set(ROLES),
                 f"reference roles changed for {cid}")
        for role in ROLES:
            item = refs[role]
            anchors = item.get("anchors")
            _require(isinstance(anchors, list) and len(anchors) <= 4,
                     f"{cid}/{role} sparse anchor catalog is malformed")
            _require(item.get("no_update") is (len(anchors) == 0)
                     and item.get("eligible_for_update") is (len(anchors) > 0),
                     f"{cid}/{role} no-update state differs from exact anchor emptiness")
            expected_fit_count += int(bool(anchors))
    _require(runtime.get("expected_fit_count") == expected_fit_count
             and runtime.get("steps") == 5
             and runtime.get("active_parameters") == 1792,
             "student runtime differs from locked fit matrix/5-step/1,792-coordinate protocol")

    # Check all exact global cross-bindings before opening output tensors.
    expected_global_bindings = {
        "roster_lock_sha256": _sha_file(roster_path),
        "manifest_sha256": _sha_file(manifest_path),
        "tracklet_barrier_sha256": _sha_file(tracklet_path),
        "feature_barrier_sha256": _sha_file(feature_barrier_path),
        "teacher_global_barrier_sha256": _sha_file(teacher_barrier_path),
        "teacher_job_lock_sha256": _sha_file(teacher_job_path),
        "reference_lock_sha256": _sha_file(reference_path),
        "runtime_sha256": runtime_sha,
    }
    for key, value in expected_global_bindings.items():
        _require(barrier.get(key) == value,
                 f"student global barrier has a stale {key}")
    _require(barrier.get("actual_fit_count") == expected_fit_count,
             "student barrier actual fit count differs from the sealed sparse references")

    output_rows = barrier.get("files")
    _require(isinstance(output_rows, list) and len(output_rows) == 8,
             "student barrier must bind exactly eight result files")
    case_paths: dict[str, Path] = {}
    for row in output_rows:
        _require(isinstance(row, Mapping), "malformed student output receipt")
        relative = _safe_relative(row.get("path"))
        prefix = f"artifacts/stvg_short_event_tracklets_v1/student/"
        _require(relative.startswith(prefix) and relative.endswith(".pt")
                 and "/" not in relative[len(prefix):],
                 "student result path is outside the exact case output directory")
        cid = relative[len(prefix):-3]
        _require(cid in CASE_IDS and cid not in case_paths,
                 "student barrier has an unknown or duplicated case result")
        case_paths[cid] = _verify_file(relative, row.get("sha256"), row.get("bytes"))
    _require(set(case_paths) == set(CASE_IDS), "student barrier does not seal every fixed case")

    # Verify full old temporal-reuse inputs and feature payload bytes up front.
    old_paths: dict[str, Path] = {}
    for cid in CASE_IDS:
        relative = f"artifacts/stvg_temporal_caption_query_spatial_opd_v1/student/{cid}.pt"
        digest = pins.get(relative)
        _require(isinstance(digest, str), f"student runtime omits exact old-T input pin for {cid}")
        old_paths[cid] = _verify_file(relative, digest)
    old_barrier_rel = "artifacts/stvg_temporal_caption_query_spatial_opd_v1/STUDENT_GLOBAL_BARRIER.json"
    _require(old_barrier_rel in pins,
             "student runtime omits the old temporal global barrier pin")
    old_barrier_path = _verify_file(old_barrier_rel, pins[old_barrier_rel])
    old_barrier = _read_json(old_barrier_path)
    _require(old_barrier.get("status") == "complete"
             and old_barrier.get("count") == 8
             and old_barrier.get("GT_read") is False,
             "old temporal baseline barrier is not complete no-GT old-eight")
    old_file_by_id = {}
    for row in old_barrier.get("files", []):
        if not isinstance(row, Mapping):
            continue
        rel = _safe_relative(row.get("path"))
        if rel.startswith("artifacts/stvg_temporal_caption_query_spatial_opd_v1/student/") and rel.endswith(".pt"):
            old_file_by_id[Path(rel).stem] = row
    for cid in CASE_IDS:
        old_row = old_file_by_id.get(cid)
        _require(isinstance(old_row, Mapping)
                 and old_row.get("path") == old_paths[cid].relative_to(ROOT).as_posix()
                 and old_row.get("sha256") == pins[f"artifacts/stvg_temporal_caption_query_spatial_opd_v1/student/{cid}.pt"],
                 f"old temporal barrier does not bind exact F/T baseline for {cid}")

    feature_paths = _feature_records(feature_barrier)
    hashes = {
        "student_runtime_sha256": runtime_sha,
        "student_barrier_sha256": _sha_file(barrier_path),
        "reference_lock_sha256": _sha_file(reference_path),
        "manifest_sha256": _sha_file(manifest_path),
        "tracklet_barrier_sha256": _sha_file(tracklet_path),
        "feature_lock_sha256": feature_lock_sha,
        "feature_runtime_sha256": feature_runtime_sha,
        "feature_barrier_sha256": _sha_file(feature_barrier_path),
        "teacher_job_lock_sha256": _sha_file(teacher_job_path),
        "teacher_barrier_sha256": _sha_file(teacher_barrier_path),
        "roster_lock_sha256": _sha_file(roster_path),
        "old_temporal_barrier_sha256": _sha_file(old_barrier_path),
    }
    return {
        "manifest": manifest, "manifest_by_id": manifest_by_id,
        "roster_by_id": roster_by_id, "reference_by_id": reference_by_id,
        "runtime": runtime, "barrier": barrier, "case_paths": case_paths,
        "old_paths": old_paths, "feature_paths": feature_paths,
        "hashes": hashes, "expected_fit_count": expected_fit_count,
        "feature_barrier_status": feature_barrier.get("status"),
    }


def _anchors_from_reference(reference: Mapping[str, Any], manifest_case: Mapping[str, Any],
                            frame_ids: Sequence[int]) -> tuple[tuple[list[int], torch.Tensor], str]:
    centers = manifest_case.get("centers")
    _require(isinstance(centers, list) and len(centers) == 4,
             "manifest case must have four fixed QA centers")
    center_by_slot = {int(row["slot"]): row for row in centers}
    physical_to_slot: dict[int, int] = {}
    for slot, row in center_by_slot.items():
        physical = int(row["center_physical_frame_id"])
        _require(physical not in physical_to_slot, "manifest center physical ID is duplicated")
        physical_to_slot[physical] = slot
    _require(set(center_by_slot) == {0, 1, 2, 3}, "manifest QA center slots are not 0..3")
    source_pos = {int(frame_id): pos for pos, frame_id in enumerate(frame_ids)}
    raw_anchors = reference.get("anchors")
    _require(isinstance(raw_anchors, list) and len(raw_anchors) <= 4,
             "locked reference must contain zero to four sparse anchors")
    positions: list[int] = []
    boxes: list[torch.Tensor] = []
    seen_physical: set[int] = set()
    semantics_seen: set[str] = set()
    for row in raw_anchors:
        _require(isinstance(row, Mapping), "sparse anchor must be an object")
        physical = int(row["physical_frame_id"])
        _require(physical in physical_to_slot and physical in source_pos,
                 "sparse anchor physical ID is not a locked QA center/source frame")
        slot, grid_position = physical_to_slot[physical], source_pos[physical]
        position = row.get("position")
        _require(isinstance(position, int) and not isinstance(position, bool),
                 "sparse anchor position must be an integer")
        semantics = row.get("position_semantics", reference.get("position_semantics"))
        _require(semantics is None or semantics in {"source_grid", "source_grid_position"},
                 "anchor position semantics must identify the source grid")
        _require(position == grid_position,
                 "anchor position must equal its exact source-grid index")
        semantics_seen.add("source_grid")
        if "qa_slot" in row:
            qa_slot = row["qa_slot"]
            _require(isinstance(qa_slot, int) and not isinstance(qa_slot, bool) and qa_slot == slot,
                     "explicit anchor qa_slot differs from its center")
        if "source_grid_position" in row:
            explicit_grid_position = row["source_grid_position"]
            _require(isinstance(explicit_grid_position, int)
                     and not isinstance(explicit_grid_position, bool)
                     and explicit_grid_position == grid_position,
                     "explicit anchor source-grid position differs from its frame")
        _require(physical not in seen_physical, "duplicate sparse anchor physical frame")
        seen_physical.add(physical)
        raw_box = row.get("box_cxcywh")
        box = _normalized_anchor_box(raw_box)
        _require(float(row.get("weight", 1.0)) == 1.0,
                 "the locked Gaussian OPD fit requires unit anchor weights")
        positions.append(grid_position)
        boxes.append(box)
    if not positions:
        return (([], torch.empty((0, 4), dtype=torch.float32)), "empty")
    _require(len(set(positions)) == len(positions),
             "sparse anchors collide on the source frame grid")
    semantics_summary = "+".join(sorted(semantics_seen))
    # The pinned fit helper sorts normalized anchors by source-grid position
    # before it creates Gaussian noise rows. Mirror that exact input order.
    order = sorted(range(len(positions)), key=positions.__getitem__)
    return (([positions[i] for i in order], torch.stack([boxes[i] for i in order])),
            semantics_summary)


def _audit_feature(cid: str, feature_path: Path, manifest_case: Mapping[str, Any],
                   reference_case: Mapping[str, Any], feature_hashes: Mapping[str, str]) -> dict[str, Any]:
    feature = torch.load(feature_path, map_location="cpu", weights_only=False)
    _require(isinstance(feature, Mapping) and feature.get("schema") == EXPECTED_FEATURE_SCHEMA,
             f"feature payload schema differs for {cid}")
    metadata = feature.get("metadata")
    _require(isinstance(metadata, Mapping) and metadata.get("case_id") == cid
             and metadata.get("GT_read") is False and metadata.get("backward_calls") == 0,
             f"feature metadata/no-GT claim invalid for {cid}")
    _require(metadata.get("source_case_sha256") == manifest_case.get("source_case_sha256")
             and metadata.get("source_model_state_sha256") == manifest_case.get("source_model_state_sha256"),
             f"feature payload source identity differs for {cid}")
    _require(metadata.get("manifest_case_sha256") == _canonical_sha(manifest_case),
             f"feature payload is not bound to this manifest case for {cid}")
    _require(metadata.get("canonical_request_sha256") == manifest_case.get("canonical_request_sha256")
             and metadata.get("canonical_video_sha256") == manifest_case.get("canonical_video_sha256")
             and metadata.get("source_grid_frame_ids_sha256") == manifest_case.get("source_grid_frame_ids_sha256"),
             f"feature request/video/source-grid bindings differ from manifest for {cid}")
    expected_ids = sorted({
        int(frame_id)
        for center in manifest_case.get("centers", [])
        for frame_id in center.get("physical_frame_ids", [])
    })
    _require(bool(expected_ids), f"short-window physical frame union missing for {cid}")
    _require(feature.get("frame_ids") == expected_ids
             and metadata.get("new_physical_frame_ids") == expected_ids,
             f"feature physical frame IDs differ from the fixed short-window manifest for {cid}")
    pixel_map = metadata.get("pixel_sha256_by_physical_frame")
    _require(isinstance(pixel_map, Mapping)
             and set(pixel_map) == {str(frame_id) for frame_id in expected_ids}
             and all(isinstance(value, str) and len(value) == 64 for value in pixel_map.values())
             and isinstance(metadata.get("pixel_sha256"), str)
             and len(metadata["pixel_sha256"]) == 64,
             f"feature pixel hash receipt does not cover the exact short window for {cid}")
    maps = feature.get("appearance_maps")
    _require(torch.is_tensor(maps) and maps.device.type == "cpu" and not maps.requires_grad
             and maps.ndim == 4 and maps.shape[0] == len(expected_ids)
             and maps.shape[1] == 256 and torch.isfinite(maps).all(),
             f"feature appearance map tensor is invalid for {cid}")
    offset_ids = metadata.get("native_offsets")
    _require(isinstance(offset_ids, Mapping)
             and offset_ids.get("0") == expected_ids[0::2]
             and offset_ids.get("1") == expected_ids[1::2],
             f"feature offset partition/order does not match physical frame IDs for {cid}")
    grids = metadata.get("feature_grid_size_by_offset")
    _require(isinstance(grids, Mapping)
             and set(grids) == {"0", "1"}
             and all(list(map(int, grids[key])) == list(maps.shape[-2:]) for key in ("0", "1")),
             f"feature map grid sizes disagree with saved tensor for {cid}")
    _require(metadata.get("feature_maps_shape") == list(maps.shape)
             and metadata.get("feature_maps_dtype") == str(maps.dtype),
             f"feature payload shape/dtype receipt differs for {cid}")
    digest = hashlib.sha256(maps.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    _require(metadata.get("feature_maps_sha256") == digest,
             f"feature payload tensor digest differs for {cid}")
    _require(metadata.get("feature_lock_sha256") == feature_hashes["feature_lock_sha256"]
             and metadata.get("feature_runtime_sha256") == feature_hashes["feature_runtime_sha256"]
             and metadata.get("manifest_sha256") == feature_hashes["manifest_sha256"],
             f"feature payload lock/runtime/manifest binding differs for {cid}")
    _require(reference_case.get("source_case_sha256") == manifest_case.get("source_case_sha256"),
             f"reference lock source SHA differs from feature case for {cid}")
    return {"feature_file_sha256": _sha_file(feature_path),
            "feature_tensor_sha256": digest,
            "frame_count": len(expected_ids), "shape": list(maps.shape),
            "dtype": str(maps.dtype), "no_gt_no_backward": True}


def _audit_old_reuse(new_case: Mapping[str, Any], old_case: Mapping[str, Any],
                     cid: str, old_file_sha: str) -> dict[str, Any]:
    _require(old_case.get("case_id") == cid and old_case.get("GT_read") is False,
             f"old temporal source for {cid} has wrong identity or GT scope")
    _require(new_case.get("source_case_sha256") == old_case.get("source_case_sha256"),
             f"new and old source-case SHA differ for {cid}")
    _require(list(map(int, new_case.get("frame_ids", []))) == list(map(int, old_case.get("frame_ids", []))),
             f"new and old source frame grids differ for {cid}")
    new_cold, old_cold = new_case.get("source_cold_readback"), old_case.get("source_cold_readback")
    _require(isinstance(new_cold, Mapping) and isinstance(old_cold, Mapping)
             and new_cold.get("status") == "pass" and old_cold.get("status") == "pass",
             f"source cold-readback receipts are missing for {cid}")
    _require(new_cold.get("source_model_state_sha256") == old_cold.get("full_model_state_sha256")
             and new_cold.get("pixel_sha256") == old_cold.get("pixel_sha256")
             and new_case.get("source_model_state_sha256") == new_cold.get("source_model_state_sha256")
             and new_case.get("pixel_sha256") == new_cold.get("pixel_sha256"),
             f"source checkpoint/pixel bindings differ from old exact reuse for {cid}")
    _require(new_cold.get("source_restored") is True
             and new_cold.get("frame_ids_match_manifest") is True
             and new_cold.get("initial_state_match") is True
             and new_cold.get("zero_boxes_logits_match") is True,
             f"source cold-readback did not certify initial state/readout for {cid}")
    _require(new_case.get("source_restored") is True
             and new_case.get("source_capture_calls") == 1,
             f"student did not reset and capture one source checkpoint for {cid}")
    reuse = new_case.get("temporal_exact_reuse")
    expected_old_path = f"artifacts/stvg_temporal_caption_query_spatial_opd_v1/student/{cid}.pt"
    _require(isinstance(reuse, Mapping) and reuse.get("status") == "reused_exact_old_T"
             and reuse.get("old_student_path") == expected_old_path
             and reuse.get("old_student_sha256") == old_file_sha
             and reuse.get("old_T_boxes_equal_F") is True
             and reuse.get("old_T_interval_reused") is True,
             f"temporal baseline receipt is not exact for {cid}")
    arms, old_arms = new_case.get("arms"), old_case.get("arms")
    _require(isinstance(arms, Mapping) and isinstance(old_arms, Mapping),
             f"F/T arm maps missing for {cid}")
    for arm in ("F", "T"):
        _require(arm in arms and arm in old_arms, f"{arm} arm missing for {cid}")
        _exact_tensor(arms[arm].get("boxes"), old_arms[arm].get("boxes"), f"{cid}.{arm}.boxes_exact_reuse")
        _same_interval(arms[arm].get("interval"), old_arms[arm].get("interval"), f"{cid}.{arm}")
    source_boxes = _as_tensor(new_case.get("source_boxes"), f"{cid}.source_boxes")
    _require(source_boxes.shape == (len(new_case["frame_ids"]), 4),
             f"source boxes do not cover the source grid for {cid}")
    _exact_tensor(_xyxy(source_boxes), arms["F"]["boxes"], f"{cid}.F_vs_source_boxes")
    _exact_tensor(arms["F"]["boxes"], arms["T"]["boxes"], f"{cid}.F_vs_T_boxes")
    return {"source_case_sha_exact": True,
            "source_state_sha_exact": True,
            "source_pixel_sha_exact": True,
            "source_frame_grid_exact": True,
            "F_arm_exact_reuse": True,
            "T_arm_exact_reuse": True,
            "F_T_boxes_equal": True,
            "source_checkpoint_reloaded_by_reader": False}


def _audit_case(cid: str, student: Mapping[str, Any], old_case: Mapping[str, Any],
                manifest_case: Mapping[str, Any], roster_case: Mapping[str, Any],
                reference_case: Mapping[str, Any],
                feature_path: Path, old_file_sha: str, feature_hashes: Mapping[str, str],
                reference_lock_sha: str) -> dict[str, Any]:
    _require(student.get("schema") == EXPECTED_STUDENT_SCHEMA
             and student.get("case_id") == cid
             and student.get("dataset") == manifest_case.get("dataset"),
             f"student payload identity/schema differs for {cid}")
    _validate_cohort_identity(student, roster_case, manifest_case)
    _require(student.get("GT_read") is False and student.get("online_carry") is False
             and student.get("source_restored") is True
             and student.get("reference_lock_sha256") == reference_lock_sha,
             f"student case has invalid GT/carry/source/reference scope for {cid}")
    _require(student.get("source_case_sha256") == reference_case.get("source_case_sha256")
             and student.get("source_model_state_sha256") == reference_case.get("source_model_state_sha256"),
             f"student case source SHA differs from locked reference for {cid}")
    frame_ids = student.get("frame_ids")
    _require(isinstance(frame_ids, list) and len(frame_ids) >= 2
             and all(isinstance(x, int) and not isinstance(x, bool) for x in frame_ids)
             and all(b > a for a, b in zip(frame_ids, frame_ids[1:])),
             f"student source physical frame grid is invalid for {cid}")
    _require(_canonical_sha(list(map(int, frame_ids))) == manifest_case.get("source_grid_frame_ids_sha256"),
             f"student source grid differs from manifest for {cid}")
    _require(student.get("manifest_case_sha256") == reference_case.get("manifest_case_sha256")
             and student.get("tracklet_case_sha256") == reference_case.get("tracklet_case_sha256"),
             f"student manifest/tracklet binding differs from reference lock for {cid}")

    source_cold = student.get("source_cold_readback")
    _require(isinstance(source_cold, Mapping)
             and source_cold.get("source_case_sha256") == student.get("source_case_sha256")
             and source_cold.get("source_model_state_sha256") == student.get("source_model_state_sha256"),
             f"source cold-readback has stale source hashes for {cid}")
    _audit_old_reuse(student, old_case, cid, old_file_sha)

    feature_result = _audit_feature(cid, feature_path, manifest_case, reference_case, feature_hashes)
    refs = reference_case.get("references")
    _require(isinstance(refs, Mapping) and set(refs) == set(ROLES),
             f"locked reference roles differ for {cid}")
    _require(_canonical_sha(student.get("reference_records")) == _canonical_sha(refs),
             f"student reference records differ from the lock for {cid}")
    arms = student.get("arms")
    fits = student.get("fits")
    readouts = student.get("readouts")
    receipts = student.get("reference_receipts")
    _require(all(isinstance(value, Mapping) for value in (arms, fits, readouts, receipts)),
             f"student fit/arm/readout/receipt maps are malformed for {cid}")

    locked_fits = {role for role in ROLES if refs[role].get("anchors")}
    _require(set(fits) == locked_fits and set(readouts) == locked_fits,
             f"actual fits/readouts do not equal nonempty locked references for {cid}")
    _require(student.get("actual_fit_count") == len(locked_fits)
             and student.get("optimizer_steps") == sum(int(fits[role].get("optimizer_steps", -1)) for role in locked_fits)
             and student.get("actual_final_redecode_calls") == len(readouts),
             f"case actual fit/readout counters disagree with saved tensors for {cid}")
    config_row = student.get("configuration")
    _require(isinstance(config_row, Mapping)
             and config_row.get("key") == f"caption_query_fixed_space|{cid}"
             and config_row.get("scope") == "joint"
             and config_row.get("steps") == 5
             and config_row.get("active_parameters") == 1792
             and config_row.get("method_reference_lock_sha256") == reference_lock_sha,
             f"case configuration/key differs from the locked 5-step/1,792-coordinate protocol for {cid}")
    from scripts import stvg_target_teacher_student_v1 as core
    expected_config = core.load_original_opd_config(str(student["dataset"]))
    expected_config["steps"] = 5
    _require(config_row.get("config") == expected_config,
             f"case configuration differs from original domain config for {cid}")

    role_reports: dict[str, Any] = {}
    total_steps = 0
    position_semantics: dict[str, str] = {}
    for role in ROLES:
        reference = refs[role]
        role_anchors, semantics = _anchors_from_reference(reference, manifest_case, frame_ids)
        position_semantics[role] = semantics
        if not role_anchors[0]:
            for expected_arm, baseline_arm in ((role, "F"), (f"T_{role}", "T")):
                _require(expected_arm in arms and baseline_arm in arms,
                         f"empty reference omitted {expected_arm} reuse arm for {cid}")
                _exact_tensor(arms[expected_arm].get("boxes"), arms[baseline_arm].get("boxes"),
                              f"{cid}.{expected_arm}_exact_{baseline_arm}_reuse")
                _same_interval(arms[expected_arm].get("interval"), arms[baseline_arm].get("interval"),
                               f"{cid}.{expected_arm}_exact_{baseline_arm}_reuse")
            for suffix, baseline_name in ((role, "F"), (f"T_{role}", "T")):
                receipt = receipts.get(suffix)
                _require(isinstance(receipt, Mapping)
                         and receipt.get("optimizer_steps") == 0
                         and receipt.get("final_redecode_calls") == 0
                         and receipt.get("no_update") is True,
                         f"empty {suffix} must remain a no-fit/no-redecode arm for {cid}")
                flag = "exact_F_reuse" if baseline_name == "F" else "exact_T_reuse"
                _require(receipt.get(flag) is True,
                         f"empty {suffix} does not explicitly certify {baseline_name} reuse for {cid}")
            role_reports[role] = {
                "anchor_count": 0, "fit": False,
                "status": "exact_no_update_reuse",
                "F_reuse_exact": True, "T_reuse_exact": True,
                "optimizer_steps": 0, "full_model_readout": False,
            }
            continue

        fit = fits[role]
        readout = readouts[role]
        reference_sha = _canonical_sha(reference)
        _require(fit.get("reference_role") == "teacher"
                 and fit.get("reference_sha256") == reference_sha
                 and fit.get("reward_objective") in {"iou", "original_gaussian_spatial_iou"},
                 f"{role} fit is not bound to its locked teacher reference for {cid}")
        _require(fit.get("config") == expected_config
                 and fit.get("scope") == "joint"
                 and fit.get("active_parameters") == 1792
                 and fit.get("declared_steps") == 5
                 and fit.get("selected_step") == 5,
                 f"{role} fit differs from the fixed domain-specific joint configuration for {cid}")
        key = str(config_row["key"])
        _require(fit.get("seed_key") == key,
                 f"{role} fit Gaussian seed key differs for {cid}")

        from scripts.review_stvg_scene_object_id_spatial_math_v1 import _audit_opd_fit
        fit_math = _audit_opd_fit(fit, role_anchors, frame_ids, key, "iou", student["initial_state"])
        _require(torch.is_tensor(readout.get("boxes"))
                 and torch.is_tensor(fit.get("final_cached_decoder_boxes")),
                 f"{role} omitted final decoder/readout boxes for {cid}")
        _exact_tensor(readout["boxes"], fit["final_cached_decoder_boxes"],
                      f"{cid}.{role}.final_readout_equals_cached_decoder")
        _exact_sequence(readout.get("logits"), student.get("source_logits"),
                        f"{cid}.{role}.native_temporal_logits_preserved")
        _exact_tensor(_xyxy(readout["boxes"].detach().cpu()), arms[role].get("boxes"),
                      f"{cid}.{role}.arm_matches_final_readout")
        _same_interval(readout.get("physical_interval"), student.get("Native_interval"),
                       f"{cid}.{role}.native_interval_preserved_by_logits")
        _exact_tensor(arms[role].get("boxes"), arms[f"T_{role}"].get("boxes"),
                      f"{cid}.{role}.T_role_boxes_match_spatial_readout")
        _same_interval(arms[role].get("interval"), student.get("Native_interval"),
                       f"{cid}.{role}.native_interval")
        _same_interval(arms[f"T_{role}"].get("interval"), arms["T"].get("interval"),
                       f"{cid}.{role}.old_temporal_interval_reuse")
        total_steps += int(fit["optimizer_steps"])
        role_reports[role] = {
            "anchor_count": len(role_anchors[0]),
            "anchor_position_semantics_verified": semantics,
            "fit": True, "fit_math": fit_math,
            "final_full_model_readout_equals_cached_decoder": True,
            "native_temporal_logits_preserved": True,
            "T_interval_reused_exactly": True,
            "optimizer_steps": int(fit["optimizer_steps"]),
            "decoder_backward_recomputed_by_reader": False,
        }

    _require(student.get("optimizer_steps") == total_steps,
             f"case optimizer-step sum differs from its role fits for {cid}")
    return {
        "dataset": student["dataset"],
        "fit_count": len(locked_fits),
        "optimizer_steps": total_steps,
        "roles": role_reports,
        "source_F_T_readback": {
            "source_case_state_pixel_frame_ids_bound": True,
            "F_and_T_exactly_match_prior_temporal_baseline": True,
            "source_checkpoint_reloaded_by_reader": False,
            "RGB_decoded_by_reader": False,
        },
        "short_window_feature_readback": feature_result,
        "reported_source_capture_calls": int(student["source_capture_calls"]),
        "reported_final_full_model_readouts": int(student["actual_final_redecode_calls"]),
        "reported_capture_seconds": float(student.get("source_capture_seconds", 0.0)),
        "position_semantics": position_semantics,
    }


def audit(*, write_report: bool = True) -> dict[str, Any]:
    gate = _preflight()
    if READBACK_PATH.exists() and write_report:
        raise FileExistsError(f"refusing to overwrite readback receipt: {READBACK_PATH}")
    from scripts.review_stvg_scene_object_id_spatial_math_v1 import _compare_tree

    per_case: dict[str, Any] = {}
    domain_count = {"hc2": 0, "vidstg": 0}
    fit_count = optimizer_steps = 0
    role_fit_count = {role: 0 for role in ROLES}
    role_empty_count = {role: 0 for role in ROLES}
    alias_count = {"hc2": 0, "vidstg": 0}
    old_pins = gate["runtime"]["pins"]
    reference_lock_sha = gate["hashes"]["reference_lock_sha256"]
    for cid in CASE_IDS:
        # All 8 student bytes, all 8 old baseline bytes, and all feature bytes
        # passed the global preflight hash checks before this first load.
        student = torch.load(gate["case_paths"][cid], map_location="cpu", weights_only=False)
        old = torch.load(gate["old_paths"][cid], map_location="cpu", weights_only=False)
        feature_info = gate["feature_paths"][cid]
        manifest_case = gate["manifest_by_id"][cid]
        roster_case = gate["roster_by_id"][cid]
        reference_case = gate["reference_by_id"][cid]
        old_relative = f"artifacts/stvg_temporal_caption_query_spatial_opd_v1/student/{cid}.pt"
        result = _audit_case(
            cid, student, old, manifest_case, roster_case, reference_case,
            feature_info["path"], old_pins[old_relative], gate["hashes"],
            reference_lock_sha,
        )
        domain = result["dataset"]
        domain_count[domain] += 1
        alias_count[domain] += 1
        alias = ("HC" if domain == "hc2" else "V") + f"{alias_count[domain]:02d}"
        fit_count += result["fit_count"]
        optimizer_steps += result["optimizer_steps"]
        for role in ROLES:
            if result["roles"][role]["fit"]:
                role_fit_count[role] += 1
            else:
                role_empty_count[role] += 1
        per_case[alias] = result
        del student, old

    _require(domain_count == {"hc2": 4, "vidstg": 4},
             "student readback domain counts are not the fixed balanced eight")
    _require(fit_count == gate["expected_fit_count"]
             and fit_count == gate["barrier"].get("actual_fit_count"),
             "independent fit count differs from runtime/barrier")
    _require(optimizer_steps == gate["barrier"].get("optimizer_steps"),
             "independent optimizer step count differs from student barrier")
    _require(sum(role_fit_count.values()) == fit_count
             and sum(role_empty_count.values()) == 16 - fit_count,
             "role-level fit and empty-reference counts do not partition the 16 fixed roles")

    report = {
        "schema": "stvg-short-window-student-independent-math-readback-v1",
        "status": "pass",
        "cases": 8,
        "fit_count": fit_count,
        "role_fit_count": role_fit_count,
        "role_empty_no_fit_count": role_empty_count,
        "optimizer_steps": optimizer_steps,
        "fixed_config": {"scope": "joint", "declared_steps": 5,
                          "active_parameters": 1792, "gaussian_samples": 32,
                          "seed_key_format": "caption_query_fixed_space|<case_id>"},
        "method_roles": list(ROLES),
        "cohort_scope": "historically_exposed_old8; not a fresh confirmation cohort",
        "cpu_readback_scope": {
            "saved_gaussian_rollout_reward_weight_likelihood_and_adam_transition": True,
            "uses_saved_gradient_vectors_and_adam_states": True,
            "independent_decoder_backward_recomputed": False,
            "source_model_reloaded": False,
            "full_model_readouts_replayed": False,
            "F_T_source_control_reused_exactly": True,
            "empty_reference_is_exact_no_fit_no_redecode": True,
        },
        "GT_read": False,
        "labels_read": False,
        "model_loaded": False,
        "GPU_initialized": False,
        "video_or_RGB_read": False,
        "pixels_independently_redecoded": False,
        "pixel_checks_used": "sealed SHA receipts and source-case/old-baseline metadata only",
        "feature_capture_barrier_status": gate["feature_barrier_status"],
        "counts_by_domain": domain_count,
        "actual_cost_counts_from_sealed_barrier": {
            "source_captures": sum(row["reported_source_capture_calls"] for row in per_case.values()),
            "final_full_model_readouts": gate["barrier"].get("actual_final_redecode_calls"),
            "optimizer_steps": optimizer_steps,
            "student_worker_end_to_end_seconds": gate["barrier"].get("end_to_end_seconds"),
            "sum_reported_source_capture_seconds": sum(
                row["reported_capture_seconds"] for row in per_case.values()
            ),
        },
        "case_readbacks": per_case,
        "provenance": {
            **gate["hashes"],
            "student_source_sha256": _sha_file(ROOT / "scripts/run_stvg_short_window_student_v1.py"),
            "reader_source_sha256": _sha_file(Path(__file__).resolve()),
            "reused_cpu_fit_auditor_path": FIT_AUDITOR.relative_to(ROOT).as_posix(),
            "reused_cpu_fit_auditor_sha256": _sha_file(FIT_AUDITOR),
            "student_global_barrier_files_sha256": {
                cid: _sha_file(gate["case_paths"][cid]) for cid in CASE_IDS
            },
        },
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # This aggregate is over anonymous case summaries and contains no boxes,
    # pixel bytes, captions, teacher text, query strings, or private paths.
    report["case_roots"] = {name: _canonical_sha(value) for name, value in per_case.items()}
    report["all_case_root"] = _canonical_sha(report["case_roots"])
    if write_report:
        READBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp = READBACK_PATH.with_name(READBACK_PATH.name + ".tmp")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, READBACK_PATH)
    _require(torch.cuda.is_initialized() is False,
             "CPU math reader unexpectedly initialized CUDA")
    return report


def run_cpu_contracts() -> dict[str, Any]:
    """Small synthetic checks only; does not inspect scope artifacts."""
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU contracts must run before CUDA initialization")
    _validate_cohort_identity(
        {"cohort": "old8"},
        {"cohort": "old8"},
        {"cohort": "historically_exposed_old8"},
    )
    for student, roster, manifest in (
        ({"cohort": "historically_exposed_old8"}, {"cohort": "old8"},
         {"cohort": "historically_exposed_old8"}),
        ({"cohort": "old8"}, {"cohort": "confirmation"},
         {"cohort": "historically_exposed_old8"}),
        ({"cohort": "old8"}, {"cohort": "old8"}, {"cohort": "old8"}),
    ):
        try:
            _validate_cohort_identity(student, roster, manifest)
        except RuntimeError:
            pass
        else:
            raise AssertionError("cohort validation accepted an unbound or incorrect label")
    frame_ids = [90, 100, 110, 120, 130, 140]
    centers = [
        {"slot": 0, "center_physical_frame_id": 100},
        {"slot": 1, "center_physical_frame_id": 110},
        {"slot": 2, "center_physical_frame_id": 120},
        {"slot": 3, "center_physical_frame_id": 130},
    ]
    manifest = {"centers": centers}
    box = [0.5, 0.5, 0.2, 0.2]
    grid_ref = {"position_semantics": "source_grid",
                "anchors": [{"position": 2, "source_grid_position": 2,
                             "physical_frame_id": 110,
                             "box_cxcywh": box, "weight": 1.0}]}
    grid_result, grid_semantics = _anchors_from_reference(grid_ref, manifest, frame_ids)
    assert grid_result[0] == [2]
    assert grid_semantics == "source_grid"
    qa_slot_only_ref = {"position_semantics": "qa_slot",
                        "anchors": [{"position": 1, "physical_frame_id": 110,
                                     "box_cxcywh": box, "weight": 1.0}]}
    try:
        _anchors_from_reference(qa_slot_only_ref, manifest, frame_ids)
    except RuntimeError:
        pass
    else:
        raise AssertionError("QA-slot position semantics must be rejected")
    qa_slot_implicit_ref = {"anchors": [{"position": 1, "physical_frame_id": 110,
                                         "box_cxcywh": box, "weight": 1.0}]}
    try:
        _anchors_from_reference(qa_slot_implicit_ref, manifest, frame_ids)
    except RuntimeError:
        pass
    else:
        raise AssertionError("QA-slot numeric position must not fall back by inference")
    empty, empty_semantics = _anchors_from_reference({"anchors": []}, manifest, frame_ids)
    assert empty[0] == [] and tuple(empty[1].shape) == (0, 4) and empty_semantics == "empty"
    box_late, box_early = [0.7, 0.5, 0.2, 0.2], [0.3, 0.5, 0.2, 0.2]
    permuted_ref = {"position_semantics": "source_grid_position", "anchors": [
        {"position": 4, "source_grid_position": 4, "physical_frame_id": 130,
         "box_cxcywh": box_late, "weight": 1.0},
        {"position": 1, "source_grid_position": 1, "physical_frame_id": 100,
         "box_cxcywh": box_early, "weight": 1.0},
    ]}
    permuted, _ = _anchors_from_reference(permuted_ref, manifest, frame_ids)
    assert permuted[0] == [1, 4]  # match helper _validate_anchors source-grid ordering
    assert torch.equal(permuted[1], torch.tensor([box_early, box_late], dtype=torch.float32))
    assert torch.cuda.is_initialized() is False
    fp32_box = torch.tensor([[0.46750038862228394, 0.81569904088974,
                              0.2696801722049713, 0.3566860258579254]],
                            dtype=torch.float32)
    fp64_values = fp32_box.double()
    cx, cy, width, height = fp64_values.unbind(-1)
    expected_xyxy = torch.stack((cx - width / 2, cy - height / 2,
                                 cx + width / 2, cy + height / 2), dim=-1)
    converted_xyxy = _xyxy(fp32_box)
    assert converted_xyxy.dtype == torch.float64 and torch.equal(converted_xyxy, expected_xyxy)
    edge_box = [0.6605630367994308, 0.6465563476085663,
                0.44037535786628723, 0.7068873047828674]
    edge_fit_box = _normalized_anchor_box(edge_box)
    _edge_cx, edge_cy, _edge_width, edge_height = map(float, edge_fit_box.tolist())
    edge_y2 = edge_cy + edge_height / 2
    assert edge_fit_box.dtype == torch.float32 and 1.0 < edge_y2 <= 1.0 + 1e-7
    return {"status": "passed", "checks": {
        "student_cohort_binds_exact_old8_roster_and_historical_manifest": True,
        "position_must_equal_source_grid_index_for_physical_frame": True,
        "QA_slot_semantics_and_fallback_rejected": True,
        "fit_anchor_order_matches_helper_source_grid_sort": True,
        "empty_reference_remains_empty_not_zero_box_target": True,
        "source_cxcywh_promoted_before_exact_xyxy_transform": True,
        "raw_double_anchor_bounds_precede_worker_fp32_rounding_tolerance": True,
        "cuda_initialized": False,
    }, "GT_read": False, "model_loaded": False, "video_or_RGB_read": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu-contracts", "audit"))
    args = parser.parse_args(argv)
    if args.mode == "cpu-contracts":
        print(json.dumps(run_cpu_contracts(), sort_keys=True))
        return 0
    result = audit(write_report=True)
    print(json.dumps({"status": result["status"], "cases": result["cases"],
                      "fit_count": result["fit_count"], "all_case_root": result["all_case_root"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
