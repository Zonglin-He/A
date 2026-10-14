#!/usr/bin/env python3
"""Independent CPU readback of sealed short-window references and associations.

The reader recomputes area-overlap ROI pooling, per-frame L2 normalization,
equal-frame descriptor means, and deterministic mutual-nearest edges without
calling the reference-lock helper. It only projects parsed teacher fields;
raw teacher output, prompt text, and query text are skipped by the JSON reader.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_short_event_tracklets_v1"
LOCK_PATH = BASE / "METHOD_REFERENCE_LOCK.json"
MANIFEST_PATH = BASE / "PREPARED_MANIFEST.json"
TRACK_BARRIER_PATH = BASE / "GLOBAL_TRACKLET_BARRIER.json"
FEATURE_BARRIER_PATH = BASE / "FEATURE_CAPTURE_BARRIER.json"
TEACHER_BARRIER_PATH = BASE / "TEACHER_GLOBAL_BARRIER.json"
JOB_LOCK_PATH = BASE / "TEACHER_JOB_LOCK.json"
OUT_PATH = BASE / "ROOT_REFERENCE_ASSOCIATION_READBACK003.json"
PREVIOUS_PATH = BASE / "recovery/association_reader_code007/PRE_RERUN_READBACK.json"
CASE_IDS = tuple([f"hc2_{i:03d}" for i in range(4)] + [f"vidstg_{i:03d}" for i in range(4)])
TOL = 1e-12


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha_bytes(payload)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# This small projection parser scans skipped JSON values only to find their
# boundaries. In particular, it never decodes job prompt/query strings or the
# teacher raw_output field.
def _ws(data: bytes, index: int) -> int:
    while index < len(data) and data[index] in b" \t\r\n":
        index += 1
    return index


def _scan_string(data: bytes, index: int) -> int:
    if data[index] != 34:
        raise ValueError("expected JSON string")
    index += 1
    while index < len(data):
        if data[index] == 92:
            index += 2
        elif data[index] == 34:
            return index + 1
        else:
            index += 1
    raise ValueError("unterminated JSON string")


def _skip_value(data: bytes, index: int) -> int:
    index = _ws(data, index)
    if index >= len(data):
        raise ValueError("missing JSON value")
    first = data[index]
    if first == 34:
        return _scan_string(data, index)
    if first in (123, 91):
        stack = [first]
        index += 1
        while index < len(data) and stack:
            value = data[index]
            if value == 34:
                index = _scan_string(data, index)
                continue
            if value in (123, 91):
                stack.append(value)
            elif value == 125:
                if stack[-1] != 123:
                    raise ValueError("mismatched JSON object")
                stack.pop()
            elif value == 93:
                if stack[-1] != 91:
                    raise ValueError("mismatched JSON array")
                stack.pop()
            index += 1
        if stack:
            raise ValueError("unterminated JSON container")
        return index
    while index < len(data) and data[index] not in b",]} \t\r\n":
        index += 1
    return index


def _decode(data: bytes, start: int, end: int) -> Any:
    return json.loads(data[start:end].decode("utf-8"))


def _project_object(data: bytes, index: int, allowed: set[str]) -> tuple[dict[str, Any], int]:
    index = _ws(data, index)
    if data[index] != 123:
        raise ValueError("expected JSON object")
    index = _ws(data, index + 1)
    result: dict[str, Any] = {}
    if data[index] == 125:
        return result, index + 1
    while True:
        index = _ws(data, index)
        key_end = _scan_string(data, index)
        key = _decode(data, index, key_end)
        index = _ws(data, key_end)
        if data[index] != 58:
            raise ValueError("expected colon")
        start = _ws(data, index + 1)
        end = _skip_value(data, start)
        if key in allowed:
            if key in result:
                raise ValueError(f"duplicate projected field: {key}")
            result[key] = _decode(data, start, end)
        index = _ws(data, end)
        if data[index] == 125:
            return result, index + 1
        if data[index] != 44:
            raise ValueError("expected object comma")
        index += 1


def _project_object_array(data: bytes, index: int, allowed: set[str]) -> tuple[list[dict[str, Any]], int]:
    index = _ws(data, index)
    if data[index] != 91:
        raise ValueError("expected JSON array")
    index = _ws(data, index + 1)
    result = []
    if data[index] == 93:
        return result, index + 1
    while True:
        item, index = _project_object(data, index, allowed)
        result.append(item)
        index = _ws(data, index)
        if data[index] == 93:
            return result, index + 1
        if data[index] != 44:
            raise ValueError("expected array comma")
        index += 1


def projected_json(path: Path, allowed: set[str], *, records_key: str | None = None,
                   record_allowed: set[str] | None = None,
                   nested_projection: dict[str, set[str]] | None = None) -> tuple[dict[str, Any], str, int]:
    data = Path(path).read_bytes()
    index = _ws(data, 0)
    if data[index] != 123:
        raise ValueError("expected top-level JSON object")
    index = _ws(data, index + 1)
    result: dict[str, Any] = {}
    while data[index] != 125:
        key_end = _scan_string(data, index)
        key = _decode(data, index, key_end)
        index = _ws(data, key_end)
        if data[index] != 58:
            raise ValueError("expected top-level colon")
        start = _ws(data, index + 1)
        if key == records_key:
            rows, end = _project_object_array(data, start, record_allowed or set())
            result[key] = rows
        elif nested_projection and key in nested_projection:
            value, end = _project_object(data, start, nested_projection[key])
            result[key] = value
        else:
            end = _skip_value(data, start)
            if key in allowed:
                result[key] = _decode(data, start, end)
        index = _ws(data, end)
        if data[index] == 125:
            break
        if data[index] != 44:
            raise ValueError("expected top-level comma")
        index = _ws(data, index + 1)
    return result, sha_bytes(data), len(data)


class Audit:
    def __init__(self) -> None:
        self.issues: list[str] = []

    def check(self, condition: Any, message: str) -> None:
        if not bool(condition):
            self.issues.append(message)


def _close(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def _expected_visible_maps(order: list[int]) -> tuple[dict[str, int], dict[str, int]]:
    n = len(order)
    base = {str(index + 1): candidate for index, candidate in enumerate(order)}
    if n == 0:
        rotated: dict[str, int] = {}
    elif n == 1:
        rotated = {"2": order[0]}
    else:
        rotated = {str((index + 1) % n + 1): candidate for index, candidate in enumerate(order)}
    return base, rotated


def _canonical_bindings(job: dict[str, Any]) -> list[tuple[Any, ...]]:
    rows = []
    visible_map = job.get("visible_id_to_candidate", {})
    for row in job.get("candidate_bindings", []):
        candidate = int(row["candidate_id"])
        visible = int(row["visible_id"])
        if visible_map.get(str(visible)) != candidate:
            raise ValueError("candidate binding visible ID does not map back to its candidate")
        rows.append((candidate, str(row["tracklet_id"]), int(row["window_id"]),
                     bool(row["is_source"]), tuple(int(v) for v in row["canonical_color_rgb"])))
    return sorted(rows)


def _pool_descriptor(frames: list[dict[str, Any]], maps_by_frame: dict[int, np.ndarray]) -> tuple[np.ndarray | None, dict[str, Any]]:
    unit_vectors: list[np.ndarray] = []
    used: list[int] = []
    zero_norm: list[int] = []
    for frame in frames:
        if frame.get("valid") is not True or frame.get("box_xyxy_norm") is None:
            continue
        frame_id = int(frame["physical_frame_id"])
        feature = maps_by_frame[frame_id]
        x1, y1, x2, y2 = map(float, frame["box_xyxy_norm"])
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("tracklet has illegal normalized coordinates")
        channels, height, width = feature.shape
        x_edges = np.arange(width + 1, dtype=np.float64) / width
        y_edges = np.arange(height + 1, dtype=np.float64) / height
        x_weight = np.maximum(0.0, np.minimum(x2, x_edges[1:]) - np.maximum(x1, x_edges[:-1]))
        y_weight = np.maximum(0.0, np.minimum(y2, y_edges[1:]) - np.maximum(y1, y_edges[:-1]))
        weights = y_weight[:, None] * x_weight[None, :]
        area = float(weights.sum(dtype=np.float64))
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("legal tracklet box has no support on the feature map")
        pooled = (feature * weights[None, :, :]).sum(axis=(1, 2), dtype=np.float64) / area
        norm = float(np.linalg.norm(pooled))
        if not math.isfinite(norm):
            raise ValueError("pooled descriptor norm is nonfinite")
        if norm <= TOL:
            zero_norm.append(frame_id)
            continue
        unit_vectors.append(pooled / norm)
        used.append(frame_id)
    if not unit_vectors:
        return None, {"status": "no_nonzero_valid_roi_descriptor", "valid_roi_frame_count": 0,
                      "used_physical_frame_ids": [], "zero_norm_physical_frame_ids": zero_norm}
    mean = np.mean(np.stack(unit_vectors, axis=0), axis=0, dtype=np.float64)
    if not np.isfinite(mean).all() or float(np.linalg.norm(mean)) <= TOL:
        return None, {"status": "mean_descriptor_canceled_to_zero", "valid_roi_frame_count": len(unit_vectors),
                      "used_physical_frame_ids": used, "zero_norm_physical_frame_ids": zero_norm}
    return mean, {"status": "valid", "valid_roi_frame_count": len(unit_vectors),
                  "used_physical_frame_ids": used, "zero_norm_physical_frame_ids": zero_norm}


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def _nearest(scores: dict[int, float]) -> int | None:
    if not scores:
        return None
    maximum = max(scores.values())
    return min(candidate_id for candidate_id, score in scores.items() if maximum - score <= TOL)


def run() -> dict[str, Any]:
    audit = Audit()
    lock = read_json(LOCK_PATH)
    manifest = read_json(MANIFEST_PATH)
    track_barrier = read_json(TRACK_BARRIER_PATH)
    feature_barrier = read_json(FEATURE_BARRIER_PATH)
    teacher_barrier = read_json(TEACHER_BARRIER_PATH)
    safe_job_keys = {
        "job_id", "case_id", "dataset", "cohort", "center_slot", "kind",
        "source_case_sha256", "source_model_state_sha256", "tracklet_case_sha256",
        "tracklet_barrier_sha256", "rgb_npz_sha256", "query_sha256", "prompt_sha256",
        "candidate_order", "visible_id_to_candidate", "candidate_bindings",
        "candidate_count", "allowed_output_ids", "source_only_non_comparable",
        "empty_candidate_pool", "cross_window_identity_claim", "GT_read", "model_or_gpu_called",
    }
    jobs_doc, job_lock_sha, _ = projected_json(
        JOB_LOCK_PATH,
        {"schema", "status", "count", "GT_read", "teacher_input_barrier_sha256", "tracklet_barrier_sha256"},
        records_key="records", record_allowed=safe_job_keys,
    )
    audit.check(lock.get("schema") == "stvg-short-window-reference-lock-v1" and lock.get("status") == "locked" and lock.get("GT_read") is False,
                "reference lock schema/status/GT flag")
    audit.check(manifest.get("GT_read") is False and track_barrier.get("GT_read") is False
                and feature_barrier.get("GT_read") is False and teacher_barrier.get("GT_read") is False,
                "an input barrier is not explicitly GT-free")
    audit.check(lock.get("count") == 8 and [row.get("case_id") for row in lock.get("records", [])] == list(CASE_IDS),
                "reference cohort size or stable order")
    audit.check(lock.get("manifest_sha256") == sha_file(MANIFEST_PATH), "manifest lock binding")
    audit.check(lock.get("tracklet_barrier_sha256") == sha_file(TRACK_BARRIER_PATH), "tracklet barrier lock binding")
    audit.check(lock.get("feature_barrier_sha256") == sha_file(FEATURE_BARRIER_PATH), "feature barrier lock binding")
    audit.check(lock.get("teacher_global_barrier_sha256") == sha_file(TEACHER_BARRIER_PATH), "teacher barrier lock binding")
    audit.check(lock.get("teacher_job_lock_sha256") == job_lock_sha, "teacher job lock binding")
    audit.check(len(jobs_doc.get("records", [])) == 64 and jobs_doc.get("count") == 64 and jobs_doc.get("GT_read") is False,
                "teacher job count/status")
    audit.check(len(teacher_barrier.get("files", [])) == 64 and teacher_barrier.get("count") == 64,
                "teacher result count")

    manifest_by = {row["case_id"]: row for row in manifest["cases"]}
    track_rows = {row["case_id"]: row for row in track_barrier["case_records"]}
    track_files = {row["path"]: row for row in track_barrier["files"]}
    job_by = {row["job_id"]: row for row in jobs_doc["records"]}
    audit.check(len(job_by) == 64, "teacher job ID uniqueness")

    # Verify all 32 base/rotation maps from their canonical candidate order.
    candidate_pairs = 0
    for case_id in CASE_IDS:
        for slot in range(4):
            base_id = f"{case_id}_window{slot:02d}_base"
            rotation_id = f"{case_id}_window{slot:02d}_id_rotation"
            base_job, rotation_job = job_by.get(base_id), job_by.get(rotation_id)
            audit.check(base_job is not None and rotation_job is not None, f"missing job pair {case_id}:{slot}")
            if base_job is None or rotation_job is None:
                continue
            audit.check((base_job.get("case_id"), base_job.get("center_slot"), base_job.get("kind")) == (case_id, slot, "base"),
                        f"base job identity {case_id}:{slot}")
            audit.check((rotation_job.get("case_id"), rotation_job.get("center_slot"), rotation_job.get("kind")) == (case_id, slot, "id_rotation"),
                        f"rotation job identity {case_id}:{slot}")
            order = list(map(int, base_job.get("candidate_order", [])))
            audit.check(order == list(map(int, rotation_job.get("candidate_order", []))) and len(order) == len(set(order)),
                        f"candidate order changed {case_id}:{slot}")
            expected_base, expected_rotation = _expected_visible_maps(order)
            audit.check(base_job.get("visible_id_to_candidate") == expected_base, f"base ID map {case_id}:{slot}")
            audit.check(rotation_job.get("visible_id_to_candidate") == expected_rotation, f"rotation ID map {case_id}:{slot}")
            n = len(order)
            allowed = [0, *range(1, n + 1)]
            audit.check(base_job.get("allowed_output_ids") == allowed and rotation_job.get("allowed_output_ids") == allowed,
                        f"allowed IDs {case_id}:{slot}")
            audit.check(base_job.get("candidate_count") == n and rotation_job.get("candidate_count") == n,
                        f"candidate count {case_id}:{slot}")
            try:
                audit.check(_canonical_bindings(base_job) == _canonical_bindings(rotation_job),
                            f"canonical candidate/color/tracklet/source bindings changed {case_id}:{slot}")
            except Exception as exc:
                audit.issues.append(f"candidate binding validation {case_id}:{slot}: {type(exc).__name__}")
            for field, expected in (
                ("source_case_sha256", manifest_by[case_id].get("source_case_sha256")),
                ("tracklet_barrier_sha256", lock.get("tracklet_barrier_sha256")),
                ("tracklet_case_sha256", track_rows[case_id].get("sha256")),
                ("rgb_npz_sha256", track_rows[case_id].get("rgb_npz_sha256")),
            ):
                audit.check(base_job.get(field) == rotation_job.get(field) == expected,
                            f"paired job binding {field} {case_id}:{slot}")
            audit.check(base_job.get("prompt_sha256") == rotation_job.get("prompt_sha256")
                        and base_job.get("query_sha256") == rotation_job.get("query_sha256"),
                        f"paired prompt/query digests {case_id}:{slot}")
            candidate_pairs += 1

    # Parse only safe output fields; output file bytes are hashed as sealed,
    # while raw_output is skipped without decoding.
    response_keys = {"job_id", "case_id", "dataset", "center_slot", "kind", "candidate_count",
                     "candidate_bindings", "parsed", "prompt_sha256", "source_case_sha256",
                     "status", "GT_read", "empty_candidate_pool", "source_only_non_comparable",
                     "output_truncated_at_cap"}
    response_by_job: dict[str, dict[str, Any]] = {}
    parsed_statuses: Counter[str] = Counter()
    parsed_visible_count = 0
    teacher_files = {row["path"]: row for row in teacher_barrier["files"]}
    for relative, file_row in sorted(teacher_files.items()):
        path = (ROOT / relative).resolve()
        audit.check(path.parent == (BASE / "teacher").resolve(), f"teacher path escaped sealed output folder {Path(relative).name}")
        if not path.is_file():
            audit.issues.append(f"teacher output missing {Path(relative).name}")
            continue
        projected, digest, size = projected_json(
            path, response_keys,
            nested_projection={"parsed": {"status", "visible_id", "selected_candidate_id"}},
        )
        audit.check(digest == file_row.get("sha256") and size == int(file_row.get("bytes", -1)),
                    f"teacher output file hash/size {Path(relative).name}")
        job = job_by.get(projected.get("job_id"))
        audit.check(job is not None, f"teacher output missing locked job {Path(relative).name}")
        if job is None:
            continue
        audit.check(projected.get("case_id") == job.get("case_id")
                    and projected.get("center_slot") == job.get("center_slot")
                    and projected.get("kind") == job.get("kind"), f"teacher output job identity {Path(relative).name}")
        audit.check(projected.get("prompt_sha256") == job.get("prompt_sha256")
                    and projected.get("source_case_sha256") == job.get("source_case_sha256"),
                    f"teacher output input digest {Path(relative).name}")
        audit.check(projected.get("candidate_count") == job.get("candidate_count")
                    and projected.get("candidate_bindings") == job.get("candidate_bindings"),
                    f"teacher output candidate binding {Path(relative).name}")
        audit.check(projected.get("GT_read") is False, f"teacher output GT flag {Path(relative).name}")
        parsed = projected.get("parsed") or {}
        status = str(parsed.get("status", "missing"))
        parsed_statuses[status] += 1
        visible = parsed.get("visible_id")
        selected = parsed.get("selected_candidate_id")
        if visible is not None:
            parsed_visible_count += 1
            mapped = job.get("visible_id_to_candidate", {}).get(str(visible))
            if mapped is None:
                audit.check(str(visible) == "0" and selected in (None, 0, "0"),
                            f"parsed unknown/abstain ID {Path(relative).name}")
            else:
                audit.check(selected == mapped, f"parsed visible ID resolves to different candidate {Path(relative).name}")
        else:
            audit.check(selected in (None, 0, "0"), f"parsed candidate lacks visible ID {Path(relative).name}")
        response_by_job[str(projected["job_id"])] = {
            "status": status, "visible_id": visible, "selected_candidate_id": selected,
            "kind": job["kind"], "case_id": job["case_id"], "center_slot": job["center_slot"],
        }
    audit.check(len(response_by_job) == 64 and parsed_statuses == Counter({"valid": 64}),
                "64 parsed teacher responses are not all valid and uniquely bound")

    # Recompute frozen CPU feature tensors and all tracklet ROI descriptors.
    descriptor_count = 0
    valid_descriptor_count = 0
    tracklet_count = 0
    feature_tensor_count = 0
    pair_boundaries = 0
    mutual_edge_count = 0
    max_edge_error = 0.0
    max_selected_error = 0.0
    selected_pair_mutual_count = 0
    selected_source_source_mutual_count = 0
    selected_non_source_support: set[tuple[str, int, int]] = set()
    selected_non_source_support_edges: set[tuple[str, int, int, int, int]] = set()
    case_findings = []
    lock_by_case = {row["case_id"]: row for row in lock["records"]}
    diag_by_case = {row["case_id"]: row for row in lock["diagnostics"]}

    for case_id in CASE_IDS:
        manifest_case = manifest_by[case_id]
        track_row = track_rows[case_id]
        lock_row = lock_by_case[case_id]
        diagnostic = diag_by_case[case_id]
        track_path = (ROOT / track_row["path"]).resolve()
        track_file = track_files.get(track_row["path"])
        audit.check(track_file is not None and track_file.get("sha256") == track_row.get("sha256"),
                    f"track case global binding {case_id}")
        audit.check(track_path.parent == (BASE / "tracklets").resolve()
                    and sha_file(track_path) == track_row.get("sha256"), f"track case file hash {case_id}")
        track_case = read_json(track_path)
        audit.check(track_case.get("GT_read") is False and track_case.get("case_id") == case_id,
                    f"track case GT/case ABI {case_id}")
        feature_row = feature_barrier["files"][case_id]
        feature_path = (ROOT / feature_row["path"]).resolve()
        receipt = feature_row["receipt"]
        receipt_path = (ROOT / receipt["path"]).resolve()
        audit.check(feature_path.parent == (BASE / "features").resolve() and feature_path.is_file()
                    and feature_path.stat().st_size == int(feature_row["bytes"])
                    and sha_file(feature_path) == feature_row["sha256"], f"feature map artifact binding {case_id}")
        audit.check(receipt_path.parent == (BASE / "features").resolve() and receipt_path.is_file()
                    and receipt_path.stat().st_size == int(receipt["bytes"])
                    and sha_file(receipt_path) == receipt["sha256"], f"feature receipt binding {case_id}")
        feature_receipt = read_json(receipt_path)
        audit.check(feature_receipt.get("GT_read") is False and feature_receipt.get("backward_calls") == 0,
                    f"feature receipt no-GT/no-backward {case_id}")
        payload = torch.load(feature_path, map_location="cpu", weights_only=True)
        feature_maps = payload["appearance_maps"]
        frame_ids = list(map(int, payload["frame_ids"]))
        metadata = payload["metadata"]
        audit.check(payload.get("schema") == "stvg-short-window-frozen-appearance-v1", f"feature schema {case_id}")
        audit.check(isinstance(feature_maps, torch.Tensor) and feature_maps.device.type == "cpu"
                    and not feature_maps.requires_grad and bool(torch.isfinite(feature_maps).all()),
                    f"feature tensor is not frozen CPU finite {case_id}")
        audit.check(list(feature_maps.shape) == metadata.get("feature_maps_shape"), f"feature tensor shape {case_id}")
        tensor_digest = sha_bytes(feature_maps.detach().contiguous().view(torch.uint8).numpy().tobytes())
        audit.check(tensor_digest == metadata.get("feature_maps_sha256") == lock_row.get("feature_maps_sha256"),
                    f"feature tensor digest {case_id}")
        audit.check(feature_row.get("sha256") == lock_row.get("feature_case_sha256"), f"feature artifact digest in lock {case_id}")
        audit.check(lock_row.get("manifest_case_sha256") == canonical_sha(manifest_case), f"manifest case digest {case_id}")
        audit.check(metadata.get("manifest_case_sha256") == canonical_sha(manifest_case)
                    and metadata.get("source_case_sha256") == manifest_case.get("source_case_sha256")
                    and metadata.get("source_model_state_sha256") == manifest_case.get("source_model_state_sha256"),
                    f"feature/source case bindings {case_id}")
        audit.check(metadata.get("GT_read") is False and metadata.get("backward_calls") == 0
                    and metadata.get("candidate_proposals_created") is False
                    and metadata.get("source_model_state_restored") is True,
                    f"feature capture invariants {case_id}")
        audit.check(frame_ids == sorted(set(frame_ids)), f"feature frame order {case_id}")
        expected_frames = sorted({int(frame_id) for center in manifest_case["centers"] for frame_id in center["physical_frame_ids"]})
        audit.check(frame_ids == expected_frames, f"feature physical-frame union {case_id}")
        local_by_physical = {int(row["physical_frame_id"]): int(row["local_frame_index"])
                             for row in track_case["frame_readback"]["selected_frames"]}
        audit.check(list(map(int, metadata.get("manifest_local_frame_ids", []))) == [local_by_physical[fid] for fid in frame_ids],
                    f"feature local/physical frame map {case_id}")
        feature_tensor_count += 1
        maps_by_frame = {fid: feature_maps[index].numpy().astype(np.float64, copy=False)
                         for index, fid in enumerate(frame_ids)}

        slot_by_tracklet: dict[str, int] = {}
        slot_candidate_ids: dict[int, set[int]] = {}
        for center in manifest_case["centers"]:
            slot = int(center["slot"])
            tids = [str(item["tracklet_id"]) for item in center["matched_tracklets"]]
            slot_candidate_ids[slot] = {int(item["candidate_id"]) for item in center["matched_tracklets"]}
            for tid in tids:
                slot_by_tracklet[tid] = slot
        tracks_by_id = {str(row["tracklet_id"]): row for row in track_case["tracklets"]}
        audit.check(set(tracks_by_id) == set(slot_by_tracklet), f"tracklet roster binding {case_id}")
        tracklet_count += len(tracks_by_id)
        descriptors: dict[int, dict[int, np.ndarray | None]] = {slot: {} for slot in range(4)}
        descriptor_diagnostics: dict[tuple[int, int], dict[str, Any]] = {}
        track_by_slot_candidate: dict[int, dict[int, dict[str, Any]]] = {slot: {} for slot in range(4)}
        for tracklet_id, tracklet in tracks_by_id.items():
            slot = slot_by_tracklet.get(tracklet_id)
            if slot is None:
                continue
            candidate_id = int(tracklet["candidate_id"])
            track_by_slot_candidate[slot][candidate_id] = tracklet
            vector, row_diag = _pool_descriptor(tracklet["frames"], maps_by_frame)
            descriptors[slot][candidate_id] = vector
            descriptor_diagnostics[(slot, candidate_id)] = row_diag
        for slot in range(4):
            audit.check(set(track_by_slot_candidate[slot]) == slot_candidate_ids[slot], f"per-center candidate roster {case_id}:{slot}")

        saved_diagnostics = {(int(row["center_slot"]), int(row["candidate_id"])): row
                            for row in diagnostic["descriptor_status_by_center_candidate"]}
        audit.check(set(saved_diagnostics) == set(descriptor_diagnostics), f"descriptor diagnostic roster {case_id}")
        for key, computed in descriptor_diagnostics.items():
            saved = saved_diagnostics.get(key, {})
            for field in ("status", "valid_roi_frame_count", "used_physical_frame_ids", "zero_norm_physical_frame_ids"):
                audit.check(saved.get(field) == computed.get(field), f"descriptor summary differs {case_id}:{key}:{field}")
            descriptor_count += 1
            valid_descriptor_count += int(computed["status"] == "valid")

        selections = {int(row["center_slot"]): row for row in diagnostic["selection_by_center"]}
        audit.check(set(selections) == {0, 1, 2, 3}, f"four base selections {case_id}")
        saved_associations = diagnostic["adjacent_window_mutual_nearest"]
        audit.check(len(saved_associations) == 3, f"three adjacent boundaries {case_id}")
        support_by_slot: dict[int, list[dict[str, Any]]] = {slot: [] for slot in range(4)}
        case_selected_pair_mutual = 0
        case_source_source_selected_mutual = 0
        for left_slot in range(3):
            right_slot = left_slot + 1
            left = {key: value for key, value in descriptors[left_slot].items() if value is not None}
            right = {key: value for key, value in descriptors[right_slot].items() if value is not None}
            scores = {(a, b): _cosine(left[a], right[b]) for a in sorted(left) for b in sorted(right)}
            left_nn = {a: _nearest({b: scores[a, b] for b in right}) for a in left}
            right_nn = {b: _nearest({a: scores[a, b] for a in left}) for b in right}
            edges = [{"left_candidate_id": a, "right_candidate_id": b, "cosine": scores[a, b]}
                     for a, b in sorted(scores) if left_nn[a] == b and right_nn[b] == a]
            saved = saved_associations[left_slot]
            audit.check((int(saved["left_center_slot"]), int(saved["right_center_slot"])) == (left_slot, right_slot),
                        f"boundary ordering {case_id}:{left_slot}")
            old_edges = saved["mutual_nearest_edges"]
            audit.check([(int(row["left_candidate_id"]), int(row["right_candidate_id"])) for row in old_edges]
                        == [(row["left_candidate_id"], row["right_candidate_id"]) for row in edges],
                        f"mutual-nearest edge IDs {case_id}:{left_slot}")
            audit.check(len(old_edges) == len(edges), f"mutual-nearest edge count {case_id}:{left_slot}")
            for old_edge, new_edge in zip(old_edges, edges):
                edge_error = abs(float(old_edge["cosine"]) - new_edge["cosine"])
                max_edge_error = max(max_edge_error, edge_error)
                audit.check(edge_error <= 2e-12,
                            f"mutual-nearest cosine {case_id}:{left_slot}")
            left_selected = selections[left_slot].get("selected_candidate_id")
            right_selected = selections[right_slot].get("selected_candidate_id")
            left_selected = None if left_selected is None else int(left_selected)
            right_selected = None if right_selected is None else int(right_selected)
            audit.check(saved.get("left_selected_candidate_id") == left_selected
                        and saved.get("right_selected_candidate_id") == right_selected,
                        f"selected pair IDs {case_id}:{left_slot}")
            selected_edge = next((edge for edge in edges if edge["left_candidate_id"] == left_selected
                                  and edge["right_candidate_id"] == right_selected), None)
            audit.check(saved.get("selected_pair_is_mutual_nearest") is (selected_edge is not None),
                        f"selected-pair mutual flag {case_id}:{left_slot}")
            expected_cos = None if selected_edge is None else selected_edge["cosine"]
            audit.check(_close(saved.get("selected_pair_cosine"), expected_cos, 2e-12),
                        f"selected-pair cosine {case_id}:{left_slot}")
            if saved.get("selected_pair_cosine") is not None and expected_cos is not None:
                selected_error = abs(float(saved["selected_pair_cosine"]) - expected_cos)
                # Match the earlier readback's reporting contract: differences
                # already within the pre-set 2e-12 verification tolerance are
                # reported as zero, not as floating-point residue.
                max_selected_error = max(max_selected_error,
                                         0.0 if selected_error <= 2e-12 else selected_error)
            if selected_edge is not None:
                selected_pair_mutual_count += 1
                case_selected_pair_mutual += 1
                if left_selected == 1 and right_selected == 1:
                    selected_source_source_mutual_count += 1
                    case_source_source_selected_mutual += 1
                for endpoint_slot, endpoint_id, neighbor_slot, neighbor_id in (
                    (left_slot, left_selected, right_slot, right_selected),
                    (right_slot, right_selected, left_slot, left_selected),
                ):
                    if endpoint_id is not None and endpoint_id != 1:
                        selected_non_source_support.add((case_id, endpoint_slot, endpoint_id))
                        selected_non_source_support_edges.add(
                            (case_id, endpoint_slot, endpoint_id, left_slot, right_slot)
                        )
                        if selections[endpoint_slot].get("reference_excluded_as_Source") is False:
                            support_by_slot[endpoint_slot].append({
                                "neighbor_center_slot": neighbor_slot,
                                "neighbor_candidate_id": neighbor_id,
                                "cosine": selected_edge["cosine"],
                            })
            mutual_edge_count += len(edges)
            pair_boundaries += 1

        local_anchors = lock["records"][CASE_IDS.index(case_id)]["references"]["local_R1"]["anchors"]
        associated_anchors = lock["records"][CASE_IDS.index(case_id)]["references"]["associated_R1"]["anchors"]
        expected_local = []
        for slot in range(4):
            selection = selections[slot]
            job_id = selection.get("base_job_id")
            response = response_by_job.get(job_id)
            audit.check(response is not None, f"locked base selection missing parsed teacher result {case_id}:{slot}")
            if response is not None:
                audit.check(response["selected_candidate_id"] == selection.get("selected_candidate_id"),
                            f"locked selection differs from parsed teacher ID {case_id}:{slot}")
            selected = selection.get("selected_candidate_id")
            if selected is None or int(selected) == 1:
                continue
            selected = int(selected)
            tracklet = track_by_slot_candidate[slot].get(selected)
            audit.check(tracklet is not None, f"selected non-Source tracklet missing {case_id}:{slot}")
            if tracklet is None:
                continue
            center_frame_id = int(manifest_case["centers"][slot]["center_physical_frame_id"])
            center_frame = next((frame for frame in tracklet["frames"]
                                 if int(frame["physical_frame_id"]) == center_frame_id), None)
            audit.check(center_frame is not None and center_frame.get("valid") is True
                        and center_frame.get("box_xyxy_norm") is not None,
                        f"selected center box missing {case_id}:{slot}")
            if center_frame is None or center_frame.get("box_xyxy_norm") is None:
                continue
            x1, y1, x2, y2 = map(float, center_frame["box_xyxy_norm"])
            expected_local.append({
                "slot": slot,
                "position": int(manifest_case["centers"][slot]["source_grid_position"]),
                "physical_frame_id": center_frame_id,
                "candidate_id": selected,
                "tracklet_id": tracklet["tracklet_id"],
                "box_cxcywh": [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1],
            })
        audit.check(len(local_anchors) == len(expected_local), f"local anchor count {case_id}")
        for expected in expected_local:
            saved = next((row for row in local_anchors if int(row["position"]) == expected["position"]), None)
            audit.check(saved is not None, f"local anchor position {case_id}:{expected['slot']}")
            if saved is not None:
                audit.check(int(saved["physical_frame_id"]) == expected["physical_frame_id"]
                            and int(saved["candidate_id"]) == expected["candidate_id"]
                            and saved["tracklet_id"] == expected["tracklet_id"],
                            f"local anchor identity {case_id}:{expected['slot']}")
                audit.check(np.allclose(np.asarray(saved["box_cxcywh"], dtype=np.float64),
                                        np.asarray(expected["box_cxcywh"], dtype=np.float64), rtol=0, atol=1e-12),
                            f"local anchor box {case_id}:{expected['slot']}")
        expected_associated = [(anchor, support_by_slot[anchor["slot"]]) for anchor in expected_local
                               if support_by_slot[anchor["slot"]]]
        audit.check(len(associated_anchors) == len(expected_associated), f"associated anchor count {case_id}")
        for expected, support in expected_associated:
            saved = next((row for row in associated_anchors if int(row["position"]) == expected["position"]), None)
            audit.check(saved is not None, f"associated anchor position {case_id}:{expected['slot']}")
            if saved is not None:
                audit.check(int(saved["candidate_id"]) == expected["candidate_id"]
                            and saved["tracklet_id"] == expected["tracklet_id"],
                            f"associated anchor identity {case_id}:{expected['slot']}")
                saved_support = saved.get("association_support", [])
                audit.check(len(saved_support) == len(support), f"associated support count {case_id}:{expected['slot']}")
                for a, b in zip(saved_support, support):
                    audit.check(a.get("neighbor_center_slot") == b["neighbor_center_slot"]
                                and a.get("neighbor_candidate_id") == b["neighbor_candidate_id"]
                                and _close(a.get("cosine"), b["cosine"], 2e-12),
                                f"associated support edge {case_id}:{expected['slot']}")
        case_findings.append({
            "case_id": case_id,
            "tracklets": len(tracks_by_id),
            "descriptors": len(descriptor_diagnostics),
            "valid_descriptors": sum(value["status"] == "valid" for value in descriptor_diagnostics.values()),
            "base_selected_non_source": len(expected_local),
            "associated_selected_non_source": len(expected_associated),
            "adjacent_boundaries": 3,
            "mutual_edges": sum(len(row["mutual_nearest_edges"]) for row in diagnostic["adjacent_window_mutual_nearest"]),
        })

    local_anchor_count = sum(len(row["references"]["local_R1"]["anchors"]) for row in lock["records"])
    associated_anchor_count = sum(len(row["references"]["associated_R1"]["anchors"]) for row in lock["records"])
    local_fit_case_count = sum(bool(row["references"]["local_R1"]["anchors"]) for row in lock["records"])
    associated_fit_case_count = sum(bool(row["references"]["associated_R1"]["anchors"]) for row in lock["records"])
    audit.check(local_anchor_count == 4 and associated_anchor_count == 0, "global R1 anchor counts")
    audit.check(lock.get("expected_fit_counts") == {"local_R1": 4, "associated_R1": 0}, "expected R1 fit counts")
    audit.check(lock.get("no_update_counts") == {"local_R1": 4, "associated_R1": 8}, "R1 no-update counts")

    source_paths = [
        "scripts/lock_stvg_short_window_references_v1.py",
        "scripts/prepare_stvg_short_window_teacher_v1.py",
        "scripts/run_stvg_short_window_teacher_v1.py",
    ]
    source_pins = {relative: sha_file(ROOT / relative) for relative in source_paths}
    source_pins.update({
        "tracklet_runner_sha256": track_barrier.get("runner_sha256"),
        "tracklet_helper_sha256": track_barrier.get("tracker_helper_sha256"),
        "feature_worker_sha256": feature_barrier.get("worker_sha256"),
        "feature_contracts_sha256": feature_barrier.get("cpu_contracts_sha256"),
        "teacher_runtime_sha256": teacher_barrier.get("runtime_sha256"),
        "teacher_job_lock_sha256": job_lock_sha,
        "teacher_global_barrier_sha256": sha_file(TEACHER_BARRIER_PATH),
        "feature_barrier_sha256": sha_file(FEATURE_BARRIER_PATH),
        "tracklet_barrier_sha256": sha_file(TRACK_BARRIER_PATH),
        "method_reference_lock_sha256": sha_file(LOCK_PATH),
    })

    previous = read_json(PREVIOUS_PATH)
    science = {
        "cases": len(CASE_IDS),
        "candidate_base_rotation_pairs_checked": candidate_pairs,
        "teacher_parsed_response_files_hash_verified": len(response_by_job),
        "teacher_parsed_status_counts": dict(parsed_statuses),
        "teacher_parsed_visible_ids_checked": parsed_visible_count,
        "feature_case_count": 8,
        "feature_tensors_cpu_verified": feature_tensor_count,
        "tracklet_count": tracklet_count,
        "independent_roi_descriptor_count": descriptor_count,
        "valid_roi_descriptor_count": valid_descriptor_count,
        "adjacent_window_boundaries_recomputed": pair_boundaries,
        "independent_mutual_nearest_edge_count": mutual_edge_count,
        "local_R1_anchor_count": local_anchor_count,
        "associated_R1_anchor_count": associated_anchor_count,
        "local_R1_fit_case_count": local_fit_case_count,
        "associated_R1_fit_case_count": associated_fit_case_count,
        "max_saved_vs_independent_edge_cosine_abs_error": max_edge_error,
        "max_saved_vs_independent_selected_cosine_abs_error": max_selected_error,
        "phase_findings": case_findings,
        "source_pins": source_pins,
    }

    previous_science_fields = tuple(science.keys())
    changed = [key for key in previous_science_fields if previous.get(key) != science[key]]
    # Clarify selected-pair support separately from pool-level mutual-nearest edges.
    selected_non_source_position_count = len(selected_non_source_support)
    selected_pair_fields = {
        "selected_pair_mutual_count": selected_pair_mutual_count,
        "selected_source_source_mutual_count": selected_source_source_mutual_count,
        "selected_non_source_support_edge_count": len(selected_non_source_support_edges),
        "selected_non_source_supported_position_count": selected_non_source_position_count,
        "selected_non_source_support_edges": [
            {"case_id": cid, "center_slot": slot, "candidate_id": candidate,
             "boundary_left_slot": left_slot, "boundary_right_slot": right_slot}
            for cid, slot, candidate, left_slot, right_slot in sorted(selected_non_source_support_edges)
        ],
        "selected_non_source_supported_positions": [
            {"case_id": cid, "center_slot": slot, "candidate_id": candidate}
            for cid, slot, candidate in sorted(selected_non_source_support)
        ],
    }
    receipt = {
        "schema": "stvg-short-window-reference-association-independent-readback-v2",
        "status": "pass" if not audit.issues and not changed else "fail",
        "scope": "CPU-only independent arithmetic and sealed-metadata readback; no student execution",
        **science,
        **selected_pair_fields,
        "scientific_fields_equal_to_previous_readback": not changed,
        "scientific_field_differences_from_previous": changed,
        "previous_readback_sha256": sha_file(PREVIOUS_PATH),
        "independent_reader_sha256": sha_file(Path(__file__).resolve()),
        "raw_teacher_output_text_read": False,
        "raw_query_text_decoded_or_inspected": False,
        "raw_RGB_pixels_decoded_or_inspected": False,
        "GT_read": False,
        "model_loaded": False,
        "GPU_used": False,
        "issues": audit.issues,
        "selected_pair_interpretation": (
            f"{selected_pair_mutual_count} selected adjacent pairs were mutual-nearest; "
            f"{selected_source_source_mutual_count} were Source-to-Source, and "
            f"{selected_non_source_position_count} selected non-Source positions received mutual-nearest support."
        ),
    }
    return receipt


def main() -> None:
    if OUT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite {OUT_PATH}")
    receipt = run()
    raw = (json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(OUT_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(BASE, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(json.dumps({
        "output": str(OUT_PATH),
        "status": receipt["status"],
        "scientific_fields_equal_to_previous_readback": receipt["scientific_fields_equal_to_previous_readback"],
        "scientific_field_differences": receipt["scientific_field_differences_from_previous"],
        "selected_pair_mutual_count": receipt["selected_pair_mutual_count"],
        "selected_source_source_mutual_count": receipt["selected_source_source_mutual_count"],
        "selected_non_source_supported_position_count": receipt["selected_non_source_supported_position_count"],
        "issues": receipt["issues"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
