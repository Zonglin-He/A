"""Independent scalar metric readback for the sealed short-window study.

The audit refuses to open targets until the completed ROOT_POSTSEAL_REPORT and
all prediction/source barriers have passed preflight. Reported metrics are
recomputed with local scalar IoU and half-open full-tube arithmetic; the
postseal scorer is never imported for metric values.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / "artifacts/stvg_short_event_tracklets_v1"
SOURCE_BASE = ROOT / "artifacts/stvg_mllm_candidate_feedback_v1/source_cases"
REPORT_PATH = BASE / "ROOT_POSTSEAL_REPORT.json"
READBACK_PATH = BASE / "ROOT_POSTSEAL_INDEPENDENT_READBACK.json"
CASE_IDS = tuple([f"hc2_{i:03d}" for i in range(4)] + [f"vidstg_{i:03d}" for i in range(4)])
ARMS = ("F", "T", "local_R1", "associated_R1", "T_local_R1", "T_associated_R1")
METRICS = ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT")
NEUTRAL_EPS = 1e-8
METRIC_TOL = 2e-9


def _sha_file(path: Path) -> str:
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
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _safe_root_path(value: Any) -> Path:
    _require(isinstance(value, str) and value and "\\" not in value,
             "artifact path must be a ROOT-relative POSIX string")
    path = Path(value)
    _require(not path.is_absolute() and not any(part in {"", ".", ".."} for part in path.parts),
             "absolute or traversing artifact path is forbidden")
    return (ROOT / path).resolve()


def _verify_file(relative: Any, digest: Any, byte_count: Any = None) -> Path:
    path = _safe_root_path(relative)
    _require(isinstance(digest, str) and len(digest) == 64
             and all(char in "0123456789abcdef" for char in digest),
             f"malformed SHA256 for {relative}")
    _require(path.is_file() and _sha_file(path) == digest,
             f"locked file bytes changed: {relative}")
    if byte_count is not None:
        _require(path.stat().st_size == int(byte_count),
                 f"locked byte count changed: {relative}")
    return path


def _float_box(value: Any, label: str) -> tuple[float, float, float, float]:
    _require(isinstance(value, (list, tuple)) and len(value) == 4,
             f"{label} must be an xyxy four-vector")
    box = tuple(float(component) for component in value)
    _require(all(math.isfinite(component) for component in box),
             f"{label} contains a non-finite coordinate")
    return box  # type: ignore[return-value]


def _scalar_iou_xyxy(first: Sequence[float], second: Sequence[float]) -> float:
    """Own scalar xyxy IoU arithmetic; degenerate/zero-union boxes score zero."""
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _pixel_predictions(boxes_xyxy_normalized: Any, width: int, height: int,
                       dataset: str) -> list[tuple[float, float, float, float]]:
    _require(dataset in {"hc2", "vidstg"} and width > 0 and height > 0,
             "unsupported dataset or dimensions")
    try:
        if torch.is_tensor(boxes_xyxy_normalized):
            raw = boxes_xyxy_normalized.detach().cpu().to(dtype=torch.float64)
        else:
            raw = torch.as_tensor(boxes_xyxy_normalized, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError("arm boxes must be CPU-convertible numeric [frames,4] values") from exc
    _require(raw.ndim == 2 and raw.shape[1] == 4 and torch.isfinite(raw).all(),
             "arm tube must be finite [frames,4] normalized xyxy")
    scale = (float(width), float(height), float(width), float(height))
    output = []
    for row in raw.tolist():
        box = tuple(float(row[index]) * scale[index] for index in range(4))
        _require(box[0] < box[2] and box[1] < box[3],
                 "arm contains non-positive normalized xyxy geometry")
        # TA-STVG's HC-STVG wrapper clamps predicted pixel coordinates below
        # zero before interpolation; it does not apply an upper-edge clamp.
        if dataset == "hc2":
            box = tuple(max(0.0, coordinate) for coordinate in box)
        output.append(box)  # type: ignore[arg-type]
    return output


def _predicted_at(frame_id: int, frame_ids: Sequence[int],
                  pixel_boxes: Sequence[Sequence[float]]) -> tuple[float, float, float, float] | None:
    if frame_id < frame_ids[0] or frame_id > frame_ids[-1]:
        return None
    index = bisect.bisect_left(frame_ids, frame_id)
    if index < len(frame_ids) and frame_ids[index] == frame_id:
        return tuple(map(float, pixel_boxes[index]))  # type: ignore[return-value]
    _require(0 < index < len(frame_ids), "interpolation bracket is absent inside source support")
    left, right = int(frame_ids[index - 1]), int(frame_ids[index])
    ratio = (float(frame_id) - left) / (right - left)
    return tuple(float(pixel_boxes[index - 1][axis])
                 + ratio * (float(pixel_boxes[index][axis]) - float(pixel_boxes[index - 1][axis]))
                 for axis in range(4))  # type: ignore[return-value]


def _interval(value: Any, label: str) -> tuple[int, int]:
    _require(isinstance(value, (list, tuple)) and len(value) == 2,
             f"{label} must be a half-open two-endpoint interval")
    a, b = map(float, value)
    _require(math.isfinite(a) and math.isfinite(b) and a.is_integer() and b.is_integer() and b > a,
             f"{label} must have increasing integer frame boundaries")
    return int(a), int(b)


def _fulltube_metrics(boxes_xyxy_normalized: Any, frame_ids: Sequence[int], width: int,
                      height: int, interval: Any, truth: Mapping[int, Sequence[float]],
                      span: Any, dataset: str) -> tuple[dict[str, float], dict[int, float]]:
    ids = [int(value) for value in frame_ids]
    _require(len(ids) > 1 and all(right > left for left, right in zip(ids, ids[1:])),
             "source frame IDs must be strictly increasing")
    pixel_boxes = _pixel_predictions(boxes_xyxy_normalized, width, height, dataset)
    _require(len(pixel_boxes) == len(ids), "arm tube does not cover its source frame grid")
    pred_start, pred_end = _interval(interval, "prediction interval")
    gt_start, gt_end = _interval(span, "GT interval")
    gt = {int(fid): _float_box(box, f"truth[{fid}]") for fid, box in truth.items()}
    _require(gt and all(right > left and bottom > top
                        for left, top, right, bottom in gt.values()),
             "GT support is empty or contains degenerate pixel boxes")
    per_frame: dict[int, float] = {}
    for fid in sorted(gt):
        prediction = _predicted_at(fid, ids, pixel_boxes)
        per_frame[fid] = 0.0 if prediction is None else _scalar_iou_xyxy(prediction, gt[fid])

    inter_start, inter_end = max(gt_start, pred_start), min(gt_end, pred_end)
    temporal_intersection = max(0, inter_end - inter_start)
    temporal_union = (gt_end - gt_start) + (pred_end - pred_start) - temporal_intersection
    mt_iou = temporal_intersection / temporal_union if temporal_union > 0 else 0.0
    interval_union = max(pred_end, gt_end) - min(pred_start, gt_start)
    viou_numerator = sum(value for fid, value in per_frame.items()
                         if inter_start <= fid < inter_end)
    m_viou = viou_numerator / max(interval_union, 1)
    s_iou = sum(per_frame.values()) / len(gt)
    return ({"m_vIoU": float(m_viou), "m_tIoU": float(mt_iou),
             "m_sIoU_dense_GT": float(s_iou)}, per_frame)


def _subset_delta(reference: Mapping[int, float], current: Mapping[int, float],
                  selected_fids: set[int] | None, selection_status: str) -> dict[str, Any]:
    gt_fids = set(reference)
    selected = set() if selected_fids is None else set(map(int, selected_fids)) & gt_fids
    deltas = [(current[fid] - reference[fid]) * 100.0 for fid in sorted(selected)]
    count = len(deltas)
    return {
        "selection_status": selection_status,
        "GT_known_frame_count": count,
        "mean_delta_sIoU_pp": sum(deltas) / count if count else None,
        "positive_count": sum(delta > NEUTRAL_EPS for delta in deltas) if count else 0,
        "negative_count": sum(delta < -NEUTRAL_EPS for delta in deltas) if count else 0,
        "neutral_count": sum(abs(delta) <= NEUTRAL_EPS for delta in deltas) if count else 0,
        "unknown_count": 1 if count == 0 else 0,
    }


def _close(actual: Any, expected: Any, label: str, tolerance: float = METRIC_TOL) -> float:
    _require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
             and isinstance(expected, (int, float)) and not isinstance(expected, bool),
             f"{label} is not a finite numeric pair")
    a, b = float(actual), float(expected)
    _require(math.isfinite(a) and math.isfinite(b), f"{label} is non-finite")
    difference = abs(a - b)
    _require(difference <= tolerance, f"{label} differs by {difference} (limit {tolerance})")
    return difference


def _same_nullable_number(actual: Any, expected: Any, label: str,
                          tolerance: float = METRIC_TOL) -> float | None:
    if actual is None or expected is None:
        _require(actual is None and expected is None, f"{label} unknown/null status differs")
        return None
    return _close(actual, expected, label, tolerance)


def _compare_subset(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> float:
    _require(actual.get("selection_status") == expected.get("selection_status"),
             f"{label} selection semantics differ")
    for key in ("GT_known_frame_count", "positive_count", "negative_count", "neutral_count", "unknown_count"):
        _require(actual.get(key) == expected.get(key), f"{label}.{key} differs")
    difference = _same_nullable_number(actual.get("mean_delta_sIoU_pp"),
                                       expected.get("mean_delta_sIoU_pp"),
                                       f"{label}.mean_delta_sIoU_pp", tolerance=2e-7)
    return 0.0 if difference is None else difference


def _load_source_metadata(manifest_case: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    relative = manifest_case.get("source_case_path")
    path = _safe_root_path(relative)
    try:
        path.relative_to(SOURCE_BASE.resolve())
    except ValueError as exc:
        raise RuntimeError(f"source case path leaves the accepted old-source directory: {case_id}") from exc
    expected_sha = manifest_case.get("source_case_sha256")
    _require(path.is_file() and _sha_file(path) == expected_sha,
             f"source case bytes differ from the prepared manifest for {case_id}")
    receipt_path = path.with_suffix(".json")
    _require(receipt_path.is_file(), f"source case sidecar is missing for {case_id}")
    receipt = _read_json(receipt_path)
    _require(receipt.get("status") == "accepted_complete_source_only_case"
             and receipt.get("GT_read") is False
             and receipt.get("sha256") == expected_sha
             and receipt.get("bytes") == path.stat().st_size,
             f"source case sidecar differs for {case_id}")
    case = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(case, Mapping) and case.get("case_id") == case_id
             and case.get("GT_read") is False
             and case.get("source_model_state_sha256") == manifest_case.get("source_model_state_sha256"),
             f"source case identity/state is invalid for {case_id}")
    input_meta = case.get("input")
    _require(isinstance(input_meta, Mapping), f"source input metadata is missing for {case_id}")
    frame_ids_raw = case.get("frame_ids")
    if torch.is_tensor(frame_ids_raw):
        frame_ids_raw = frame_ids_raw.detach().cpu().tolist()
    frame_ids = [int(value) for value in frame_ids_raw]
    _require(len(frame_ids) > 1 and all(b > a for a, b in zip(frame_ids, frame_ids[1:])),
             f"source frame grid is invalid for {case_id}")
    _require(_canonical_sha(frame_ids) == manifest_case.get("source_grid_frame_ids_sha256"),
             f"source frame grid SHA differs for {case_id}")
    dataset = str(case.get("dataset"))
    width, height = int(input_meta["width"]), int(input_meta["height"])
    query_ordinal = case.get("query_ordinal")
    _require(dataset in {"hc2", "vidstg"} and width > 0 and height > 0
             and isinstance(query_ordinal, int) and not isinstance(query_ordinal, bool),
             f"source metric metadata is malformed for {case_id}")
    frame_count = len(frame_ids)
    bounds = list(map(float, case["video_bounds"]))
    _require(len(bounds) == 2 and bounds[1] > bounds[0],
             f"source video bounds are malformed for {case_id}")
    student_frame_ids = None
    del case
    return {"case_id": case_id, "dataset": dataset, "width": width, "height": height,
            "query_ordinal": query_ordinal, "frame_ids": frame_ids,
            "frame_count": frame_count, "video_bounds": bounds}


def _preflight() -> dict[str, Any]:
    """Verify all hashes, barriers and prediction bytes before the GT loader."""
    if not REPORT_PATH.is_file():
        raise RuntimeError("ROOT_POSTSEAL_REPORT is absent; this reader will not open GT labels")
    report = _read_json(REPORT_PATH)
    _require(report.get("status") == "pass" and report.get("GT_read") is True
             and report.get("GT_used_for_selection") is False,
             "completed postseal report must record post-prediction scoring without selection")

    barrier_path = BASE / "STUDENT_GLOBAL_BARRIER.json"
    runtime_path = BASE / "STUDENT_RUNTIME.json"
    manifest_path = BASE / "PREPARED_MANIFEST.json"
    roster_path = BASE / "ROSTER_LOCK.json"
    track_barrier_path = BASE / "GLOBAL_TRACKLET_BARRIER.json"
    reference_path = BASE / "METHOD_REFERENCE_LOCK.json"
    teacher_job_path = BASE / "TEACHER_JOB_LOCK.json"
    teacher_barrier_path = BASE / "TEACHER_GLOBAL_BARRIER.json"
    math_readback_path = BASE / "ROOT_STUDENT_MATH_READBACK.json"

    barrier = _read_json(barrier_path)
    runtime = _read_json(runtime_path)
    manifest = _read_json(manifest_path)
    roster = _read_json(roster_path)
    track_barrier = _read_json(track_barrier_path)
    references = _read_json(reference_path)
    teacher_jobs = _read_json(teacher_job_path)
    teacher_barrier = _read_json(teacher_barrier_path)
    math_readback = _read_json(math_readback_path)

    barrier_sha = _sha_file(barrier_path)
    runtime_sha = _sha_file(runtime_path)
    manifest_sha = _sha_file(manifest_path)
    report_sha = _sha_file(REPORT_PATH)
    roster_sha = _sha_file(roster_path)
    track_barrier_sha = _sha_file(track_barrier_path)
    reference_sha = _sha_file(reference_path)
    teacher_job_sha = _sha_file(teacher_job_path)
    teacher_barrier_sha = _sha_file(teacher_barrier_path)
    math_readback_sha = _sha_file(math_readback_path)
    _require(barrier.get("schema") == "stvg-short-window-student-barrier-v1"
             and barrier.get("status") == "complete"
             and barrier.get("count") == 8 and barrier.get("case_ids") == list(CASE_IDS)
             and barrier.get("GT_read") is False
             and barrier.get("runtime_sha256") == runtime_sha,
             "student prediction barrier is incomplete or stale")
    _require(report.get("student_barrier_sha256") == barrier_sha,
             "postseal report does not bind the exact student barrier")
    _require(runtime.get("status") == "locked" and runtime.get("GT_read") is False,
             "student runtime is not a locked no-GT execution")
    math_provenance = math_readback.get("provenance")
    _require(math_readback.get("status") == "pass" and math_readback.get("cases") == 8
             and math_readback.get("GT_read") is False
             and math_readback.get("labels_read") is False
             and isinstance(math_provenance, Mapping)
             and math_provenance.get("student_barrier_sha256") == barrier_sha
             and math_provenance.get("student_runtime_sha256") == runtime_sha
             and math_provenance.get("manifest_sha256") == manifest_sha
             and math_provenance.get("roster_lock_sha256") == roster_sha
             and math_provenance.get("tracklet_barrier_sha256") == track_barrier_sha
             and math_provenance.get("reference_lock_sha256") == reference_sha
             and math_provenance.get("teacher_job_lock_sha256") == teacher_job_sha
             and math_provenance.get("teacher_barrier_sha256") == teacher_barrier_sha,
             "independent student math readback is absent, stale, or GT-exposed")
    _require(report.get("student_runtime_sha256") == runtime_sha
             and report.get("tracklet_barrier_sha256") == track_barrier_sha
             and report.get("teacher_job_lock_sha256") == teacher_job_sha
             and report.get("teacher_global_barrier_sha256") == teacher_barrier_sha
             and report.get("student_math_readback_sha256") == math_readback_sha,
             "postseal report does not bind its exact runtime, input barriers and prior math readback")
    _require(manifest.get("status") == "prepared_manifest_not_executed"
             and manifest.get("GT_read") is False,
             "prepared manifest is not the original GT-free source lock")
    _require(roster.get("status") == "locked" and roster.get("GT_read") is False,
             "fixed source roster is not locked no-GT")
    _require(track_barrier.get("status") == "complete" and track_barrier.get("count") == 8
             and track_barrier.get("center_count") == 32 and track_barrier.get("tracklet_count") == 152
             and track_barrier.get("GT_read") is False
             and track_barrier.get("teacher_called") is False
             and track_barrier.get("GPU_used") is False,
             "tracklet source barrier is incomplete or outside its no-GT scope")
    _require(references.get("status") == "locked" and references.get("GT_read") is False
             and references.get("count") == 8,
             "method reference lock is not a sealed no-GT roster")
    _require(teacher_jobs.get("status") == "locked" and teacher_jobs.get("GT_read") is False
             and teacher_jobs.get("count") == 64,
             "teacher job lock is not the fixed 64-request no-GT catalog")
    _require(teacher_barrier.get("status") in {"sealed", "complete"}
             and teacher_barrier.get("GT_read") is False
             and len(teacher_barrier.get("files", [])) == 64,
             "teacher barrier is not the complete fixed no-GT 64-response catalog")

    for relative, digest in {**runtime.get("pins", {}), **runtime.get("protected", {})}.items():
        _verify_file(relative, digest)

    manifest_rows = manifest.get("cases", [])
    roster_rows = roster.get("records", [])
    reference_rows = references.get("records", [])
    track_rows = track_barrier.get("case_records", [])
    _require([row.get("case_id") for row in manifest_rows] == list(CASE_IDS)
             and [row.get("case_id") for row in roster_rows] == list(CASE_IDS)
             and [row.get("case_id") for row in reference_rows] == list(CASE_IDS)
             and [row.get("case_id") for row in track_rows] == list(CASE_IDS),
             "source, manifest, reference and tracklet catalogs do not match fixed old8 order")
    manifest_by_id = {row["case_id"]: row for row in manifest_rows}
    roster_by_id = {row["case_id"]: row for row in roster_rows}
    reference_by_id = {row["case_id"]: row for row in reference_rows}
    track_by_id = {row["case_id"]: row for row in track_rows}
    for cid in CASE_IDS:
        m, r, ref, track = (manifest_by_id[cid], roster_by_id[cid],
                            reference_by_id[cid], track_by_id[cid])
        _require(r.get("cohort") == "old8"
                 and m.get("cohort") == "historically_exposed_old8",
                 f"cohort labels differ for {cid}")
        _require(r.get("source_case_sha256") == m.get("source_case_sha256")
                 and ref.get("source_case_sha256") == m.get("source_case_sha256")
                 and track.get("sha256") == ref.get("tracklet_case_sha256"),
                 f"source case SHA differs across locked records for {cid}")

    student_files = barrier.get("files", [])
    _require(len(student_files) == 8, "student barrier must list exactly eight predictions")
    case_paths: dict[str, Path] = {}
    for row in student_files:
        path = _verify_file(row.get("path"), row.get("sha256"), row.get("bytes"))
        _require(path.parent == (BASE / "student").resolve() and path.suffix == ".pt",
                 "student output path is outside the locked student case directory")
        cid = path.stem
        _require(cid in CASE_IDS and cid not in case_paths,
                 "student barrier contains an unknown or duplicate case")
        case_paths[cid] = path
    _require(set(case_paths) == set(CASE_IDS), "student predictions do not cover all old8 cases")

    track_files = {str(row.get("path")): row for row in track_barrier.get("files", [])}
    _require(len(track_files) == 16, "tracklet barrier must bind eight JSON and eight RGB sidecars")
    track_paths: dict[str, Path] = {}
    for cid in CASE_IDS:
        rec = track_by_id[cid]
        relative = rec.get("path")
        file_record = track_files.get(relative)
        _require(isinstance(file_record, Mapping), f"tracklet JSON is absent from its barrier for {cid}")
        path = _verify_file(relative, file_record.get("sha256"), file_record.get("bytes"))
        _require(path.parent == (BASE / "tracklets").resolve() and path.suffix == ".json",
                 f"tracklet JSON path is invalid for {cid}")
        _require(rec.get("sha256") == file_record.get("sha256"),
                 f"tracklet case SHA differs from its global file barrier for {cid}")
        track_case = _read_json(path)
        expected_source_sha = manifest_by_id[cid].get("source_case_sha256")
        _require(track_case.get("case_id") == cid and track_case.get("GT_read") is False
                 and isinstance(track_case.get("input_binding"), Mapping)
                 and track_case["input_binding"].get("source_case_sha256") == expected_source_sha
                 and track_case["input_binding"].get("manifest_sha256") == manifest_sha,
                 f"tracklet payload is not bound to the fixed source manifest for {cid}")
        _require(reference_by_id[cid].get("tracklet_case_sha256") == file_record.get("sha256"),
                 f"reference lock does not bind the exact tracklet case for {cid}")
        track_paths[cid] = path

    job_rows = teacher_jobs.get("records", [])
    _require(len(job_rows) == 64 and len({row.get("job_id") for row in job_rows}) == 64,
             "teacher job IDs are incomplete or duplicated")
    job_by_id = {str(row["job_id"]): row for row in job_rows}
    jobs_by_case_slot: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in job_rows:
        key = (str(row.get("case_id")), int(row.get("center_slot", -1)))
        _require(key[0] in CASE_IDS and 0 <= key[1] < 4,
                 "teacher job refers to an unknown case/window")
        jobs_by_case_slot.setdefault(key, []).append(row)
    responses: dict[str, Mapping[str, Any]] = {}
    for row in teacher_barrier.get("files", []):
        path = _verify_file(row.get("path"), row.get("sha256"), row.get("bytes"))
        response = _read_json(path)
        job_id = response.get("job_id")
        _require(isinstance(job_id, str) and job_id not in responses
                 and job_id in job_by_id,
                 "teacher response does not bind one unique locked job")
        parsed = response.get("parsed")
        _require(isinstance(parsed, Mapping) and "selected_candidate_id" in parsed,
                 "teacher response parser record is missing its selection field")
        selected = parsed.get("selected_candidate_id")
        _require(selected is None or (isinstance(selected, int) and not isinstance(selected, bool)),
                 "teacher selected ID must be integer or null")
        if selected is not None:
            allowed = job_by_id[job_id].get("visible_id_to_candidate", {}).values()
            _require(selected in allowed,
                     "teacher selection does not map to a canonical candidate in its locked job")
        responses[job_id] = {"selected_candidate_id": selected}
    _require(set(responses) == {str(row["job_id"]) for row in job_rows},
             "teacher responses do not cover the complete locked job catalog")
    for key, rows in jobs_by_case_slot.items():
        _require(len(rows) == 2 and {row.get("kind") for row in rows} == {"base", "id_rotation"},
                 f"teacher base/ID-rotation pair is incomplete for {key}")

    report_metrics = report.get("metrics")
    report_candidates = report.get("candidate_geometry")
    report_windows = report.get("window_pool_teacher")
    report_groups = report.get("groups")
    _require(isinstance(report_metrics, list) and len(report_metrics) == 48
             and report.get("metric_count") == 48,
             "postseal report does not contain 48 six-arm metrics")
    _require(isinstance(report_candidates, list) and len(report_candidates) == 152
             and report.get("candidate_count") == 152,
             "postseal report does not contain the complete 152 center-candidate set")
    _require(isinstance(report_windows, list) and len(report_windows) == 32
             and report.get("window_count") == 32,
             "postseal report does not contain all 32 teacher-pool windows")
    _require(isinstance(report_groups, list) and len(report_groups) == 12,
             "postseal report does not contain 12 dataset-by-arm groups")

    return {
        "report": report, "report_sha256": report_sha,
        "barrier": barrier, "runtime": runtime, "manifest": manifest,
        "manifest_by_id": manifest_by_id, "roster_by_id": roster_by_id,
        "reference_by_id": reference_by_id, "track_by_id": track_by_id,
        "case_paths": case_paths, "track_paths": track_paths,
        "jobs_by_case_slot": jobs_by_case_slot, "responses": responses,
        "math_readback": math_readback,
        "hashes": {
            "postseal_report_sha256": report_sha,
            "student_barrier_sha256": barrier_sha,
            "student_runtime_sha256": runtime_sha,
            "manifest_sha256": manifest_sha,
            "roster_lock_sha256": _sha_file(roster_path),
            "tracklet_barrier_sha256": _sha_file(track_barrier_path),
            "reference_lock_sha256": _sha_file(reference_path),
            "teacher_job_lock_sha256": _sha_file(teacher_job_path),
            "teacher_barrier_sha256": _sha_file(teacher_barrier_path),
            "student_math_readback_sha256": math_readback_sha,
        },
    }


def _load_gt_after_preflight(gate: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load labels only after _preflight has verified the sealed report and inputs."""
    source_cases: dict[str, dict[str, Any]] = {}
    metric_meta: dict[str, dict[str, Any]] = {}
    for cid in CASE_IDS:
        source_path = _safe_root_path(gate["manifest_by_id"][cid]["source_case_path"])
        _require(source_path.parent == SOURCE_BASE.resolve(),
                 f"source case is outside the fixed source-only directory for {cid}")
        source_case = torch.load(source_path, map_location="cpu", weights_only=False)
        _require(isinstance(source_case, Mapping) and source_case.get("GT_read") is False,
                 f"source input is not the sealed no-GT payload for {cid}")
        frame_ids = source_case.get("frame_ids")
        if torch.is_tensor(frame_ids):
            frame_ids = frame_ids.detach().cpu().tolist()
        frame_ids = list(map(int, frame_ids))
        source_cases[cid] = {
            "dataset": str(source_case["dataset"]),
            "query_ordinal": int(source_case["query_ordinal"]),
        }
        metric_meta[cid] = {
            "dataset": str(source_case["dataset"]),
            "query_ordinal": int(source_case["query_ordinal"]),
            "frame_ids": frame_ids,
            "width": int(source_case["input"]["width"]),
            "height": int(source_case["input"]["height"]),
            "video_bounds": list(map(float, source_case["video_bounds"])),
        }
        del source_case
    # This helper only reads official target labels; it is called after all
    # prediction, source, teacher, runtime and postseal-report checks above.
    from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal
    truth_by_case, gt_bindings = _truths_after_seal(source_cases)
    _require(gate["report"].get("GT_source_bindings") == gt_bindings,
             "independent official target-loader provenance differs from the postseal report")
    for cid in CASE_IDS:
        truth = truth_by_case[cid]["truth"]
        span = truth_by_case[cid]["span"]
        _require(isinstance(truth, Mapping) and len(truth) > 0,
                 f"official target frame support is empty for {cid}")
        metric_meta[cid]["truth"] = {int(fid): _float_box(box, f"GT[{cid},{fid}]")
                                      for fid, box in truth.items()}
        metric_meta[cid]["span"] = list(span)
    return metric_meta, gt_bindings, truth_by_case


def _report_maps(report: Mapping[str, Any]) -> tuple[dict[tuple[str, str], Mapping[str, Any]],
                                                     dict[tuple[str, int, int], Mapping[str, Any]],
                                                     dict[tuple[str, int], Mapping[str, Any]],
                                                     dict[tuple[str, str], Mapping[str, Any]]]:
    metric_map = {(str(row["case_id"]), str(row["arm"])): row for row in report["metrics"]}
    candidate_map = {(str(row["case_id"]), int(row["slot"]), int(row["candidate_id"])): row
                     for row in report["candidate_geometry"]}
    window_map = {(str(row["case_id"]), int(row["slot"])): row for row in report["window_pool_teacher"]}
    group_map = {(str(row["dataset"]), str(row["arm"])): row for row in report["groups"]}
    _require(len(metric_map) == 48 and len(candidate_map) == 152
             and len(window_map) == 32 and len(group_map) == 12,
             "postseal report has duplicate metric, candidate, window or group keys")
    return metric_map, candidate_map, window_map, group_map


def _load_track_case(path: Path, case_id: str) -> Mapping[str, Any]:
    row = _read_json(path)
    _require(row.get("status") == "complete" and row.get("GT_read") is False
             and row.get("teacher_called") is False and row.get("case_id") == case_id,
             f"sealed tracklet case is not source-only for {case_id}")
    centers = row.get("centers")
    tracklets = row.get("tracklets")
    _require(isinstance(centers, list) and [int(x["slot"]) for x in centers] == [0, 1, 2, 3]
             and isinstance(tracklets, list),
             f"tracklet windows are incomplete for {case_id}")
    return row


def _score_metrics(gate: Mapping[str, Any], metric_meta: Mapping[str, Any],
                   report_maps: tuple[Any, Any, Any, Any]) -> tuple[list[dict[str, Any]], float]:
    metric_map, _candidate_map, _window_map, _group_map = report_maps
    rows: list[dict[str, Any]] = []
    max_difference = 0.0
    for cid in CASE_IDS:
        student = torch.load(gate["case_paths"][cid], map_location="cpu", weights_only=False)
        meta = metric_meta[cid]
        _require(student.get("case_id") == cid and student.get("GT_read") is False
                 and student.get("source_case_sha256") == gate["manifest_by_id"][cid]["source_case_sha256"],
                 f"student source binding is invalid for {cid}")
        ids = list(map(int, student.get("frame_ids", [])))
        _require(ids == meta["frame_ids"], f"student/source physical frame grids differ for {cid}")
        _require(str(student.get("dataset")) == meta["dataset"], f"student/source dataset differs for {cid}")
        arms = student.get("arms")
        _require(isinstance(arms, Mapping) and set(arms) == set(ARMS),
                 f"student six-arm catalog differs for {cid}")
        references = gate["reference_by_id"][cid].get("references")
        _require(isinstance(references, Mapping), f"locked supervision references are missing for {cid}")
        computed: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            pred = arms[arm]
            metrics, frame_iou = _fulltube_metrics(
                pred.get("boxes"), ids, meta["width"], meta["height"], pred.get("interval"),
                meta["truth"], meta["span"], meta["dataset"])
            computed[arm] = {"metrics": metrics, "per_frame_iou": frame_iou}
            reported = metric_map.get((cid, arm))
            _require(isinstance(reported, Mapping), f"reported metric row missing for {cid}.{arm}")
            _require(reported.get("dataset") == meta["dataset"], f"reported dataset differs for {cid}.{arm}")
            for key in METRICS:
                max_difference = max(max_difference,
                                     _close(metrics[key], reported.get("metrics", {}).get(key),
                                            f"{cid}.{arm}.{key}"))
            baseline = "T" if arm.startswith("T_") or arm == "T" else "F"
            delta_pp = 100.0 * (metrics["m_vIoU"] - computed[baseline]["metrics"]["m_vIoU"])
            _require(reported.get("baseline") == baseline, f"baseline mapping differs for {cid}.{arm}")
            max_difference = max(max_difference,
                                 _close(delta_pp, reported.get("delta_vIoU_pp"),
                                        f"{cid}.{arm}.delta_vIoU_pp", tolerance=2e-7))

            role = arm.removeprefix("T_")
            sup_fids = {int(anchor["physical_frame_id"])
                        for anchor in references.get(role, {}).get("anchors", [])}
            supervised = _subset_delta(computed["F"]["per_frame_iou"], frame_iou, sup_fids,
                                       "fixed_actual_supervision_GT_known")
            unsupervised = _subset_delta(computed["F"]["per_frame_iou"], frame_iou,
                                         set(meta["truth"]) - sup_fids,
                                         "other_GT_known_frames")
            max_difference = max(max_difference,
                                 _compare_subset(reported.get("supervised_geometry", {}),
                                                 supervised, f"{cid}.{arm}.supervised_geometry"),
                                 _compare_subset(reported.get("unsupervised_geometry", {}),
                                                 unsupervised, f"{cid}.{arm}.unsupervised_geometry"))
            _require(reported.get("actual_supervision_count") == len(sup_fids),
                     f"{cid}.{arm} supervision count differs from the sealed lock")
            rows.append({
                "case_id": cid, "dataset": meta["dataset"], "arm": arm,
                "metrics": metrics, "baseline": baseline, "delta_vIoU_pp": delta_pp,
                "supervised_geometry": supervised, "unsupervised_geometry": unsupervised,
                "actual_supervision_count": len(sup_fids),
                "per_frame_iou": [{"physical_frame_id": fid, "iou": value}
                                   for fid, value in sorted(frame_iou.items())],
            })
        del student
    _require(len(rows) == 48, "independent metric recomputation did not cover all 48 arms")
    return rows, max_difference


def _score_candidates_and_windows(gate: Mapping[str, Any], metric_meta: Mapping[str, Any],
                                  report_maps: tuple[Any, Any, Any, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    _metric_map, candidate_map, window_map, _group_map = report_maps
    candidate_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    max_difference = 0.0
    for cid in CASE_IDS:
        student = torch.load(gate["case_paths"][cid], map_location="cpu", weights_only=False)
        meta = metric_meta[cid]
        ids = list(map(int, student["frame_ids"]))
        index_by_fid = {fid: index for index, fid in enumerate(ids)}
        source_boxes = torch.as_tensor(student["arms"]["F"]["boxes"],
                                       dtype=torch.float64, device="cpu")
        _require(source_boxes.shape == (len(ids), 4), f"Source boxes are misaligned for {cid}")
        tracks = _load_track_case(gate["track_paths"][cid], cid)
        by_tracklet = {str(track["tracklet_id"]): track for track in tracks["tracklets"]}
        _require(len(by_tracklet) == len(tracks["tracklets"]), f"tracklet IDs duplicate for {cid}")
        for center in tracks["centers"]:
            slot = int(center["slot"])
            center_fid = int(center["center_physical_frame_id"])
            _require(center_fid in index_by_fid, f"tracklet center is outside student source grid for {cid}")
            job_rows = gate["jobs_by_case_slot"][(cid, slot)]
            base_job = next(row for row in job_rows if row["kind"] == "base")
            rotated_job = next(row for row in job_rows if row["kind"] == "id_rotation")
            base_id = gate["responses"][str(base_job["job_id"])]["selected_candidate_id"]
            rotated_id = gate["responses"][str(rotated_job["job_id"])]["selected_candidate_id"]
            display_list = [int(value) for value in base_job["candidate_order"]]
            display = set(display_list)
            _require(len(display_list) == len(display), f"base candidate pool repeats an ID for {cid}/{slot}")
            known = center_fid in meta["truth"]
            truth_norm = None
            if known:
                x1, y1, x2, y2 = meta["truth"][center_fid]
                truth_norm = (x1 / meta["width"], y1 / meta["height"],
                              x2 / meta["width"], y2 / meta["height"])
            values: dict[int, float] = {}
            per_window_candidates = []
            track_ids = list(map(str, center["tracklet_ids"]))
            _require(len(track_ids) == len(set(track_ids)), f"tracklet IDs repeat in center {cid}/{slot}")
            for track_id in track_ids:
                track = by_tracklet.get(track_id)
                _require(isinstance(track, Mapping), f"center references missing tracklet {cid}/{slot}")
                candidate_id = int(track["candidate_id"])
                center_rows = [row for row in track["frames"]
                               if int(row["physical_frame_id"]) == center_fid]
                _require(len(center_rows) == 1, f"candidate center frame is not unique for {cid}/{slot}/{candidate_id}")
                frame_row = center_rows[0]
                valid_center = bool(frame_row.get("valid") is True and frame_row.get("box_xyxy_norm") is not None)
                candidate_iou = source_iou = delta_pp = None
                if known and valid_center:
                    candidate_box = _float_box(frame_row["box_xyxy_norm"], "tracklet center box")
                    source_box = tuple(float(value) for value in source_boxes[index_by_fid[center_fid]].tolist())
                    candidate_iou = _scalar_iou_xyxy(candidate_box, truth_norm)
                    source_iou = _scalar_iou_xyxy(source_box, truth_norm)
                    delta_pp = 100.0 * (candidate_iou - source_iou)
                    values[candidate_id] = candidate_iou
                candidate_row = {
                    "case_id": cid, "dataset": meta["dataset"], "slot": slot,
                    "candidate_id": candidate_id, "displayed": candidate_id in display,
                    "GT_known": known, "valid_center": valid_center,
                    "valid_tracklet_frames": sum(row.get("valid") is True for row in track["frames"]),
                    "candidate_IoU": candidate_iou, "source_IoU": source_iou, "delta_pp": delta_pp,
                }
                per_window_candidates.append(candidate_row)
                candidate_rows.append(candidate_row)
                reported = candidate_map.get((cid, slot, candidate_id))
                _require(isinstance(reported, Mapping), f"reported candidate metric missing for {cid}/{slot}/{candidate_id}")
                for key in ("displayed", "GT_known", "valid_center", "valid_tracklet_frames"):
                    _require(candidate_row[key] == reported.get(key),
                             f"{cid}/{slot}/{candidate_id}.{key} differs")
                for key in ("candidate_IoU", "source_IoU", "delta_pp"):
                    max_difference = max(max_difference,
                                         _same_nullable_number(candidate_row[key], reported.get(key),
                                                               f"{cid}/{slot}/{candidate_id}.{key}") or 0.0)

            common = bool(known and display and display <= set(values) and 1 in values)
            oracle_iou = max((values[candidate] for candidate in display), default=None) if common else None
            source_iou = values.get(1) if common else None
            teacher_iou = values.get(base_id) if common and base_id in display else None
            common_count = int(common)
            teacher_row = {
                "case_id": cid, "dataset": meta["dataset"], "slot": slot,
                "common_observation_count": common_count,
                "unknown_reason": None if common else "no_shared_GT_known_valid_observation",
                "source_IoU": source_iou, "oracle_IoU": oracle_iou,
                "teacher_IoU": teacher_iou,
                "delta_pool_pp": None if oracle_iou is None else 100.0 * (oracle_iou - source_iou),
                "delta_teacher_pp": None if teacher_iou is None else 100.0 * (teacher_iou - source_iou),
                "R1_candidate": base_id, "rotated_R1_candidate": rotated_id,
                "ID_invariant": None if base_id is None or rotated_id is None else base_id == rotated_id,
                "selected_Source": base_id == 1,
                "abstain_or_invalid": base_id is None,
                "candidate_count": len(display),
            }
            window_rows.append(teacher_row)
            reported_window = window_map.get((cid, slot))
            _require(isinstance(reported_window, Mapping), f"reported teacher-pool window missing for {cid}/{slot}")
            for key in ("common_observation_count", "unknown_reason", "R1_candidate", "rotated_R1_candidate",
                        "ID_invariant", "selected_Source", "abstain_or_invalid", "candidate_count"):
                _require(teacher_row[key] == reported_window.get(key), f"{cid}/{slot}.{key} differs")
            for key in ("source_IoU", "oracle_IoU", "teacher_IoU", "delta_pool_pp", "delta_teacher_pp"):
                max_difference = max(max_difference,
                                     _same_nullable_number(teacher_row[key], reported_window.get(key),
                                                           f"{cid}/{slot}.{key}") or 0.0)
        del student
    _require(len(candidate_rows) == 152 and len(window_rows) == 32,
             "independent candidate/window reconstruction has wrong total size")
    return candidate_rows, window_rows, max_difference


def _aggregate_groups(metric_rows: Sequence[Mapping[str, Any]],
                      report_groups: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    groups: list[dict[str, Any]] = []
    max_difference = 0.0
    for dataset in ("hc2", "vidstg"):
        for arm in ARMS:
            rows = [row for row in metric_rows if row["dataset"] == dataset and row["arm"] == arm]
            deltas = [float(row["delta_vIoU_pp"]) for row in rows]
            _require(len(rows) == 4, f"group {dataset}/{arm} does not contain four sources")
            current = {
                "dataset": dataset, "arm": arm, "count": len(rows),
                "mean_vIoU": statistics.mean(float(row["metrics"]["m_vIoU"]) for row in rows),
                "mean_delta_pp": statistics.mean(deltas),
                "median_delta_pp": statistics.median(deltas),
                "positive": sum(value > NEUTRAL_EPS for value in deltas),
                "negative": sum(value < -NEUTRAL_EPS for value in deltas),
                "neutral": sum(abs(value) <= NEUTRAL_EPS for value in deltas),
            }
            reported = report_groups.get((dataset, arm))
            _require(isinstance(reported, Mapping), f"reported group missing for {dataset}/{arm}")
            _require(current["count"] == reported.get("count")
                     and current["positive"] == reported.get("positive")
                     and current["negative"] == reported.get("negative")
                     and current["neutral"] == reported.get("neutral"),
                     f"group counts differ for {dataset}/{arm}")
            for key in ("mean_vIoU", "mean_delta_pp", "median_delta_pp"):
                max_difference = max(max_difference,
                                     _close(current[key], reported.get(key),
                                            f"{dataset}/{arm}.{key}", tolerance=2e-7))
            groups.append(current)
    _require(len(groups) == 12, "independent grouping did not generate 12 cells")
    return groups, max_difference


def audit(*, write_report: bool = True) -> dict[str, Any]:
    gate = _preflight()
    if READBACK_PATH.exists() and write_report:
        raise FileExistsError(f"refusing to overwrite independent metric readback: {READBACK_PATH}")
    report_maps = _report_maps(gate["report"])
    meta, gt_bindings, truth_by_case = _load_gt_after_preflight(gate)
    # Bind source frame geometry/clock metadata to the student predictions.
    for cid in CASE_IDS:
        student = torch.load(gate["case_paths"][cid], map_location="cpu", weights_only=False)
        _require(list(map(int, student["frame_ids"])) == meta[cid]["frame_ids"]
                 and student.get("source_case_sha256") == gate["manifest_by_id"][cid].get("source_case_sha256")
                 and student.get("manifest_case_sha256") == _canonical_sha(gate["manifest_by_id"][cid])
                 and student.get("tracklet_case_sha256") == gate["reference_by_id"][cid].get("tracklet_case_sha256"),
                 f"student source grid or source/manifest/tracklet binding differs for {cid}")
        del student

    metric_rows, metric_max_difference = _score_metrics(gate, meta, report_maps)
    candidate_rows, window_rows, pool_max_difference = _score_candidates_and_windows(gate, meta, report_maps)
    groups, group_max_difference = _aggregate_groups(metric_rows, report_maps[3])
    max_difference = max(metric_max_difference, pool_max_difference, group_max_difference)
    common_windows = sum(row["common_observation_count"] == 1 for row in window_rows)
    metric_pair_root = _canonical_sha(metric_rows)
    candidate_root = _canonical_sha(candidate_rows)
    window_root = _canonical_sha(window_rows)
    group_root = _canonical_sha(groups)
    output = {
        "schema": "stvg-short-window-postseal-independent-readback-v1",
        "status": "pass",
        "scope": "independent scalar arithmetic/readback after all six-arm predictions were sealed and the postseal report completed",
        "cases": 8, "metric_count": len(metric_rows), "candidate_count": len(candidate_rows),
        "window_count": len(window_rows), "common_candidate_pool_window_count": common_windows,
        "group_count": len(groups),
        "metric_tolerance": METRIC_TOL,
        "max_report_metric_difference": max_difference,
        "metric_semantics": {
            "coordinates": "normalized xyxy predictions scaled to official pixel xyxy; HC2 lower-clips predictions at zero only",
            "interpolation": "linear interpolation over captured physical frame IDs, no extrapolation outside source support",
            "temporal_intervals": "integer half-open physical-frame intervals",
            "m_vIoU": "sum of per-frame IoU on the prediction/GT interval intersection divided by the interval-union frame duration",
            "m_tIoU": "half-open temporal interval intersection-over-union",
            "m_sIoU_dense_GT": "mean per-frame IoU across all official GT frame support, missing prediction support contributes zero",
            "candidate_pool_iou": "center-frame normalized xyxy scalar IoU; unknown or invalid center remains null",
        },
        "per_case_arm_metrics_and_supervision_subsets": metric_rows,
        "center_candidate_values": candidate_rows,
        "center_pool_teacher_and_oracle": window_rows,
        "dataset_arm_group_means": groups,
        "independent_roots": {
            "case_arm_metrics": metric_pair_root,
            "center_candidate_values": candidate_root,
            "center_pool_teacher": window_root,
            "dataset_arm_groups": group_root,
        },
        "GT_source_bindings_sha256": _canonical_sha(gt_bindings),
        "GT_source_binding_count": len(gt_bindings),
        "provenance": {
            **gate["hashes"],
            "student_source_sha256": _sha_file(ROOT / "scripts/run_stvg_short_window_student_v1.py"),
            "reader_source_sha256": _sha_file(Path(__file__).resolve()),
        },
        "GT_read": True, "GT_used_for_selection": False,
        "model_loaded": False, "GPU_initialized": False, "video_or_RGB_read": False,
        "raw_GT_coordinates_saved": False, "raw_query_or_teacher_text_saved": False,
        "time": time.time(),
    }
    _require(len(truth_by_case) == 8, "official target loader did not return all sealed source labels")
    _require(torch.cuda.is_initialized() is False,
             "independent metric reader unexpectedly initialized CUDA")
    if write_report:
        temporary = READBACK_PATH.with_name(READBACK_PATH.name + ".tmp")
        temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        temporary.replace(READBACK_PATH)
    return output


def run_cpu_contracts() -> dict[str, Any]:
    """Synthetic arithmetic contracts; no report, source case, or GT is opened."""
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU contracts must run before CUDA initialization")
    assert _scalar_iou_xyxy((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert _scalar_iou_xyxy((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    truth = {1: (0.0, 0.0, 2.0, 2.0), 2: (0.5, 0.0, 2.5, 2.0),
             3: (1.0, 0.0, 3.0, 2.0), 5: (10.0, 10.0, 12.0, 12.0)}
    ids = [1, 3, 5]
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.5, 0.0, 1.5, 1.0],
                          [0.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
    metrics, per_frame = _fulltube_metrics(boxes, ids, 2, 2, [1, 4], truth, [1, 4], "vidstg")
    list_metrics, list_per_frame = _fulltube_metrics(boxes.tolist(), ids, 2, 2,
                                                     [1, 4], truth, [1, 4], "vidstg")
    assert list_metrics == metrics and list_per_frame == per_frame
    assert per_frame[1] == 1.0 and per_frame[2] > 0.0 and per_frame[3] == 1.0
    assert per_frame[5] == 0.0  # outside prediction interval despite captured source support
    assert metrics["m_tIoU"] == 1.0
    _, no_extrapolation = _fulltube_metrics(boxes, ids, 2, 2, [1, 6], truth,
                                            [1, 6], "vidstg")
    assert no_extrapolation[5] == 0.0
    # HC2 clips negative pixel edges; VidSTG preserves them.
    negative = torch.tensor([[-0.5, -0.5, 0.5, 0.5], [-0.5, -0.5, 0.5, 0.5]], dtype=torch.float64)
    clip_truth = {1: (0.0, 0.0, 1.0, 1.0), 2: (0.0, 0.0, 1.0, 1.0)}
    _, hc = _fulltube_metrics(negative, [1, 2], 1, 1, [1, 3], clip_truth, [1, 3], "hc2")
    _, vid = _fulltube_metrics(negative, [1, 2], 1, 1, [1, 3], clip_truth, [1, 3], "vidstg")
    assert hc[1] != vid[1]
    subset = _subset_delta({1: 0.5, 2: 1.0}, {1: 0.75, 2: 0.5}, {1},
                           "fixed_actual_supervision_GT_known")
    assert subset["GT_known_frame_count"] == 1 and subset["mean_delta_sIoU_pp"] == 25.0
    unknown = _subset_delta({1: 0.5}, {1: 0.7}, set(), "fixed_actual_supervision_GT_known")
    assert unknown["mean_delta_sIoU_pp"] is None and unknown["unknown_count"] == 1
    assert torch.cuda.is_initialized() is False
    return {
        "status": "passed",
        "checks": {
            "scalar_iou_identical_and_disjoint_boxes": True,
            "saved_numpy_or_sequence_box_abi": True,
            "physical_frame_interpolation_and_no_extrapolation": True,
            "half_open_temporal_iou": True,
            "HC2_lower_clip_differs_from_unclipped_VidSTG": True,
            "known_supervision_subset_and_unknown_null": True,
            "cuda_initialized": False,
        },
        "GT_read": False, "model_loaded": False, "video_or_RGB_read": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu-contracts", "audit"))
    args = parser.parse_args(argv)
    if args.mode == "cpu-contracts":
        print(json.dumps(run_cpu_contracts(), sort_keys=True))
        return 0
    result = audit(write_report=True)
    print(json.dumps({"status": result["status"], "cases": result["cases"],
                      "metrics": result["metric_count"], "candidates": result["candidate_count"],
                      "windows": result["window_count"], "groups": result["group_count"],
                      "max_difference": result["max_report_metric_difference"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
