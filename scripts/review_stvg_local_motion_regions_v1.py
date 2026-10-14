#!/usr/bin/env python3
"""Independent read-only custody, mask, and post-seal IoU audit.

This reader consumes only the locked local-motion outputs, their private label
archives, the earlier short-window candidate records, and the sealed GT-anchor
lock. It never loads model checkpoints, RGB/gray image tensors, or video bytes.
It writes anonymous scalar geometry only; no boxes, query text, or private paths
are copied into the receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCOPE = ROOT / "artifacts/stvg_local_motion_regions_v1"
OLD = ROOT / "artifacts/stvg_short_event_tracklets_v1"
MATCHED = ROOT / "artifacts/stvg_matched_support_reference_v1"
RUNTIME_PATH = SCOPE / "RUNTIME.json"
BARRIER_PATH = SCOPE / "CPU_GENERATOR_BARRIER.json"
GEOMETRY_PATH = SCOPE / "ROOT_POSTSEAL_GEOMETRY.json"
OUTPUT_PATH = SCOPE / "ROOT_INDEPENDENT_READBACK002.json"
TOL = 1e-12
GROUP_TOL = 1e-8

ALIASES = {
    "hc2_000": "HC01",
    "hc2_001": "HC02",
    "hc2_002": "HC03",
    "hc2_003": "HC04",
    "vidstg_000": "V01",
    "vidstg_001": "V02",
    "vidstg_002": "V03",
    "vidstg_003": "V04",
}


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object at {path}")
    return value


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _safe_path(rel: str) -> Path:
    candidate = (ROOT / rel).resolve()
    if candidate != ROOT and ROOT not in candidate.parents:
        raise ValueError(f"locked path escaped project root: {rel}")
    return candidate


def _near(a: Any, b: Any, tol: float = TOL) -> bool:
    try:
        return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _json_equal(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _near(a, b, 0.0)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    return a == b


def _box_iou_xyxy(a: Iterable[float] | None, b: Iterable[float] | None) -> float | None:
    if a is None or b is None:
        return None
    aa = np.asarray(list(a), dtype=np.float64)
    bb = np.asarray(list(b), dtype=np.float64)
    if aa.shape != (4,) or bb.shape != (4,) or not np.isfinite(aa).all() or not np.isfinite(bb).all():
        return None
    if not (aa[2] > aa[0] and aa[3] > aa[1] and bb[2] > bb[0] and bb[3] > bb[1]):
        return None
    ix1, iy1 = max(aa[0], bb[0]), max(aa[1], bb[1])
    ix2, iy2 = min(aa[2], bb[2]), min(aa[3], bb[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (aa[2] - aa[0]) * (aa[3] - aa[1])
    area_b = (bb[2] - bb[0]) * (bb[3] - bb[1])
    union = area_a + area_b - inter
    if not math.isfinite(union) or union <= 0.0:
        return None
    value = float(inter / union)
    return min(1.0, max(0.0, value))


def _cxcywh_to_xyxy(box: Iterable[float] | None) -> list[float] | None:
    if box is None:
        return None
    v = np.asarray(list(box), dtype=np.float64)
    if v.shape != (4,) or not np.isfinite(v).all() or v[2] <= 0 or v[3] <= 0:
        return None
    cx, cy, w, h = (float(x) for x in v)
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def _envelope(mask: np.ndarray) -> tuple[list[int], int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], int(xs.size)


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: dict[str, Any] | None = None) -> None:
        row = {"check": name, "status": "pass" if ok else "fail"}
        if detail:
            row["detail"] = detail
        self.checks.append(row)
        if not ok:
            self.failures.append(row)


def _load_case_data(cid: str) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    new_record = _json(SCOPE / "cases" / f"{cid}.json")
    old_record = _json(OLD / "tracklets" / f"{cid}.json")
    with np.load(SCOPE / "private_masks" / f"{cid}.npz", allow_pickle=False) as z:
        labels = np.asarray(z["frame_component_labels"])
        clean_fwd = np.asarray(z["clean_forward_masks"])
        clean_bwd = np.asarray(z["clean_backward_masks"])
        if labels.ndim != 4 or clean_fwd.ndim != 4 or clean_bwd.ndim != 4:
            raise ValueError(f"unexpected private mask dimensions for {cid}")
        if labels.shape[:2] != (4, 5) or clean_fwd.shape[:2] != (4, 4) or clean_bwd.shape[:2] != (4, 4):
            raise ValueError(f"unexpected private mask temporal dimensions for {cid}")
        arrays = {
            "labels": labels.copy(),
            "clean_fwd": clean_fwd.copy(),
            "clean_bwd": clean_bwd.copy(),
        }
    return new_record, old_record, arrays


def _verify_static_custody(audit: Audit) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    runtime = _json(RUNTIME_PATH)
    barrier = _json(BARRIER_PATH)
    geometry = _json(GEOMETRY_PATH)
    runner_sha = _sha(ROOT / "scripts/run_stvg_local_motion_regions_v1.py")
    runtime_sha = _sha(RUNTIME_PATH)
    barrier_sha = _sha(BARRIER_PATH)

    pins = runtime.get("pins", {})
    pin_errors: list[str] = []
    for rel, expected in sorted(pins.items()):
        p = _safe_path(rel)
        if not p.is_file() or _sha(p) != expected:
            pin_errors.append(rel)
    audit.check("runtime_24_pins", len(pins) == 24 and not pin_errors,
                {"declared": len(pins), "mismatches": pin_errors[:5]})
    audit.check("runtime_locked_scope", runtime.get("status") == "locked"
                and runtime.get("case_count") == 8 and runtime.get("window_count") == 32
                and runtime.get("GT_read") is False and runtime.get("GPU_used") is False
                and runtime.get("teacher_called") is False and runtime.get("model_loaded") is False,
                {"GT_read": runtime.get("GT_read"), "GPU_used": runtime.get("GPU_used"),
                 "teacher_called": runtime.get("teacher_called"), "model_loaded": runtime.get("model_loaded")})
    audit.check("barrier_runner_runtime_hashes",
                barrier.get("runner_sha256") == runner_sha
                and barrier.get("runtime_sha256") == runtime_sha
                and barrier.get("runner_sha256") == "07cae3327d2d34b236d3c2309c970e2d69da4004342f544e95d61ce3d5c6d8d5"
                and barrier.get("runtime_sha256") == "945ee3b66ee353df93300475479527e4f00373d70a16c5044fd25849a2eca9ac",
                {"runner_sha256": runner_sha, "runtime_sha256": runtime_sha})

    barrier_files = barrier.get("files", [])
    file_errors: list[str] = []
    for row in barrier_files:
        rel = str(row.get("path", ""))
        if not rel.startswith("artifacts/stvg_local_motion_regions_v1/"):
            file_errors.append(rel or "<missing-path>")
            continue
        p = _safe_path(rel)
        if not p.is_file() or p.stat().st_size != int(row.get("bytes", -1)) or _sha(p) != row.get("sha256"):
            file_errors.append(rel)
    expected_case_paths = {f"artifacts/stvg_local_motion_regions_v1/cases/{cid}.json" for cid in ALIASES}
    expected_mask_paths = {f"artifacts/stvg_local_motion_regions_v1/private_masks/{cid}.npz" for cid in ALIASES}
    actual_paths = {str(r.get("path")) for r in barrier_files}
    audit.check("barrier_16_sealed_outputs", len(barrier_files) == 16 and not file_errors
                and actual_paths == expected_case_paths | expected_mask_paths,
                {"declared_files": len(barrier_files), "mismatches": file_errors[:5]})
    audit.check("barrier_scope_status", barrier.get("status") == "complete"
                and barrier.get("case_count") == 8 and barrier.get("window_count") == 32
                and barrier.get("output_json_count") == 8 and barrier.get("private_mask_archive_count") == 8
                and barrier.get("GT_read") is False and barrier.get("GPU_used") is False
                and barrier.get("teacher_called") is False and barrier.get("scoring_started") is False,
                {"status": barrier.get("status"), "cases": barrier.get("case_count"),
                 "windows": barrier.get("window_count"), "scoring_started": barrier.get("scoring_started")})

    old_manifest = OLD / "PREPARED_MANIFEST.json"
    old_barrier_path = OLD / "GLOBAL_TRACKLET_BARRIER.json"
    old_manifest_sha, old_barrier_sha = _sha(old_manifest), _sha(old_barrier_path)
    audit.check("old_tracklet_barrier_pins",
                pins.get("artifacts/stvg_short_event_tracklets_v1/PREPARED_MANIFEST.json") == old_manifest_sha
                and pins.get("artifacts/stvg_short_event_tracklets_v1/GLOBAL_TRACKLET_BARRIER.json") == old_barrier_sha
                and barrier.get("old_manifest_sha256") == old_manifest_sha
                and barrier.get("old_global_barrier_sha256") == old_barrier_sha,
                {"manifest_sha_matches": pins.get("artifacts/stvg_short_event_tracklets_v1/PREPARED_MANIFEST.json") == old_manifest_sha,
                 "tracklet_barrier_sha_matches": pins.get("artifacts/stvg_short_event_tracklets_v1/GLOBAL_TRACKLET_BARRIER.json") == old_barrier_sha})

    audit.check("postseal_geometry_bindings",
                geometry.get("status") == "pass" and geometry.get("GT_read") is True
                and geometry.get("GT_used_for_generation") is False and geometry.get("GPU_used") is False
                and geometry.get("teacher_calls") == 0 and geometry.get("actual_OPD_fits") == 0
                and geometry.get("barrier_sha256") == barrier_sha
                and geometry.get("runtime_sha256") == runtime_sha
                and geometry.get("matched_reference_lock_sha256") == _sha(MATCHED / "REFERENCE_LOCK.json"),
                {"status": geometry.get("status"), "GT_read": geometry.get("GT_read"),
                 "GT_used_for_generation": geometry.get("GT_used_for_generation"),
                 "candidate_rows": len(geometry.get("candidate_geometry", []))})
    return runtime, barrier, geometry


def _verify_frame_provenance(audit: Audit, cid: str, new: dict[str, Any], old: dict[str, Any]) -> None:
    ib = new.get("input_bindings", {})
    old_ib = old.get("input_binding", {})
    audit.check(f"{ALIASES[cid]}_source_input_binding",
                ib.get("source_case_sha256") == old_ib.get("source_case_sha256")
                and ib.get("source_model_state_sha256") == old_ib.get("source_model_state_sha256")
                and ib.get("canonical_video_sha256") == old_ib.get("canonical_video_sha256")
                and ib.get("canonical_request_sha256") == old_ib.get("canonical_request_sha256"),
                {"source_binding_fields_checked": 4})
    audit.check(f"{ALIASES[cid]}_old_tracklet_json_binding",
                ib.get("old_tracklet_json_sha256") == _sha(OLD / "tracklets" / f"{cid}.json")
                and ib.get("old_private_rgb_npz_sha256") == _sha(OLD / "private_rgb" / f"{cid}.npz"),
                {"tracklet_and_rgb_hashes_match": True})
    old_frames = old.get("frame_readback", {}).get("selected_frames", [])
    new_frames = new.get("frame_readback", {}).get("frames", [])
    old_by_phys = {int(r["physical_frame_id"]): r for r in old_frames}
    new_by_phys = {int(r["physical_frame_id"]): r for r in new_frames}
    fields = (
        "local_frame_index", "physical_frame_id", "source_timestamp_s", "source_frame_rgb_sha256",
        "cached_rgb_sha256", "tracking_gray_sha256", "source_rgb_shape_hwc", "cached_rgb_shape_hwc", "resize",
    )
    same = set(old_by_phys) == set(new_by_phys)
    if same:
        same = all(all(_json_equal(old_by_phys[k].get(f), new_by_phys[k].get(f)) for f in fields) for k in old_by_phys)
    audit.check(f"{ALIASES[cid]}_160_selected_frame_provenance", same and len(new_frames) == 20,
                {"frame_count": len(new_frames), "matched_by_physical_frame_id": True})


def _verify_label_maps(audit: Audit, cid: str, new: dict[str, Any], masks: dict[str, np.ndarray]) -> dict[str, int]:
    labels = masks["labels"]
    clean_fwd = masks["clean_fwd"]
    clean_bwd = masks["clean_bwd"]
    windows = new.get("windows", [])
    errors = Counter()
    total_components = total_accepted = total_rejected = total_mask_pixels = 0
    for wi, window in enumerate(windows):
        rows = window.get("frame_component_readback", [])
        if len(rows) != 5:
            errors["frame_component_readback_count"] += 1
            continue
        slot = int(window.get("window_slot", wi))
        for frame_idx, frame in enumerate(rows):
            label_plane = labels[slot, frame_idx]
            if frame_idx < 4:
                clean_plane = clean_fwd[slot, frame_idx]
                expected_source = "outgoing_forward_residual"
            else:
                clean_plane = clean_bwd[slot, 3]
                expected_source = "incoming_final_backward_residual"
            if not np.array_equal(label_plane > 0, clean_plane > 0):
                errors["label_mask_union_mismatch"] += 1
            if frame.get("frame_index") != frame_idx or int(frame.get("physical_frame_id", -1)) != int(window["physical_frame_ids"][frame_idx]):
                errors["label_physical_frame_alignment"] += 1
            if frame.get("segmentation_source") != expected_source:
                errors["segmentation_source"] += 1
            comps = frame.get("components", [])
            comp_ids = {int(c["component_id"]) for c in comps}
            label_ids = set(int(x) for x in np.unique(label_plane) if int(x) != 0)
            if comp_ids != label_ids:
                errors["label_component_id_set"] += 1
            accepted_count = rejected_count = 0
            h, w = label_plane.shape
            for comp in comps:
                cid_value = int(comp["component_id"])
                region = label_plane == cid_value
                env = _envelope(region)
                if env is None:
                    errors["empty_label_component"] += 1
                    continue
                bbox_px, area = env
                x1, y1, x2, y2 = bbox_px
                if area != int(comp.get("area_px", -1)):
                    errors["component_area"] += 1
                if bbox_px != comp.get("bbox_xyxy_px"):
                    errors["component_bbox_pixels"] += 1
                norm = [x1 / w, y1 / h, x2 / w, y2 / h]
                box = comp.get("bbox_xyxy_norm")
                if box is None or len(box) != 4 or any(not _near(a, b) for a, b in zip(norm, box)):
                    errors["component_bbox_normalized"] += 1
                if list(comp.get("row_major_key", [])) != [y1, x1, y2, x2]:
                    errors["component_row_major"] += 1
                area_max = int(math.floor((h * w) * 0.25))
                should_accept = 64 <= area <= area_max
                if bool(comp.get("accepted")) != should_accept:
                    errors["component_area_acceptance"] += 1
                expected_reason = None if should_accept else ("area_below_64_px" if area < 64 else "area_above_25_percent_frame")
                if comp.get("rejection_reason") != expected_reason:
                    errors["component_rejection_reason"] += 1
                total_components += 1
                total_mask_pixels += area
                if should_accept:
                    accepted_count += 1
                    total_accepted += 1
                else:
                    rejected_count += 1
                    total_rejected += 1
            if int(frame.get("component_count", -1)) != len(comps):
                errors["component_count"] += 1
            if int(frame.get("accepted_component_count", -1)) != accepted_count:
                errors["accepted_component_count"] += 1
            if int(frame.get("rejected_component_count", -1)) != rejected_count:
                errors["rejected_component_count"] += 1
    audit.check(f"{ALIASES[cid]}_mask_label_component_audit", not errors,
                {"frames_checked": 20, "components_checked": total_components,
                 "accepted_components": total_accepted, "rejected_components": total_rejected,
                 "mask_pixels": total_mask_pixels, "errors": _counter_dict(errors)})
    return {"components": total_components, "accepted_components": total_accepted,
            "rejected_components": total_rejected, "mask_pixels": total_mask_pixels}


def _row_major_component_map(window: dict[str, Any]) -> dict[int, tuple[int, int, int, int, int]]:
    center_components = window.get("frame_component_readback", [])[2].get("components", [])
    result = {}
    for c in center_components:
        if not c.get("accepted"):
            continue
        key = tuple(int(v) for v in c["row_major_key"])
        result[int(c["component_id"])] = (*key, int(c["component_id"]))
    return result


def _verify_window(audit: Audit, cid: str, window: dict[str, Any], masks: dict[str, np.ndarray]) -> dict[str, Any]:
    slot = int(window["window_slot"])
    phys = [int(x) for x in window.get("physical_frame_ids", [])]
    if len(phys) != 5 or phys[2] != int(window.get("center_physical_frame_id", -1)):
        audit.check(f"{ALIASES[cid]}_slot{slot}_five_frame_center_binding", False,
                    {"five_frame_order": len(phys) == 5, "center_is_middle": len(phys) == 5 and phys[2] == window.get("center_physical_frame_id")})
        return {"slot": slot, "candidate_count": 0, "region_count": 0}
    audit.check(f"{ALIASES[cid]}_slot{slot}_five_frame_center_binding", True,
                {"window_center_join": "physical_frame_id", "qa_slot": slot})

    old_record = _json(OLD / "tracklets" / f"{cid}.json")
    old_tracks = []
    for track in old_record.get("tracklets", []):
        frames = track.get("frames", [])
        # `window_id` is a proposal seed index, not the QA window slot.
        if len(frames) >= 3 and int(frames[2].get("physical_frame_id", -1)) == phys[2]:
            old_tracks.append(track)
    old_by_id = {int(t["candidate_id"]): t for t in old_tracks}
    if len(old_by_id) != len(old_tracks):
        audit.check(f"{ALIASES[cid]}_slot{slot}_old_candidate_ids_unique", False)
    old_source = old_by_id.get(1)
    new_by_id = {int(c["candidate_id"]): c for c in window.get("candidates", [])}
    new_source = new_by_id.get(1)
    exact_source = old_source is not None and new_source is not None
    if exact_source:
        old_rows = {int(f["physical_frame_id"]): f for f in old_source.get("frames", [])}
        new_rows = {int(f["physical_frame_id"]): f for f in new_source.get("frames", [])}
        exact_source = set(old_rows) == set(new_rows) == set(phys)
        if exact_source:
            for p in phys:
                exact_source &= _json_equal(old_rows[p].get("box_xyxy_norm"), new_rows[p].get("box_xyxy_norm"))
                exact_source &= bool(old_rows[p].get("valid")) == bool(new_rows[p].get("valid"))
        exact_source &= bool(new_source.get("source_copy_exact")) and new_source.get("role") == "Source"
    audit.check(f"{ALIASES[cid]}_slot{slot}_Source_copy_exact", exact_source,
                {"joined_on_physical_center": True, "old_window_id_used": False})

    candidates = list(window.get("candidates", []))
    new_regions = [c for c in candidates if c.get("role") == "new_local_motion_region"]
    region_ids = [int(c["candidate_id"]) for c in new_regions]
    audit.check(f"{ALIASES[cid]}_slot{slot}_candidate_cap_and_ids",
                len(candidates) <= 5 and len(new_regions) <= 4 and int(candidates[0].get("candidate_id", -1)) == 1
                and region_ids == list(range(2, 2 + len(new_regions))),
                {"candidate_count": len(candidates), "new_regions": len(new_regions), "region_ids_stable": region_ids})

    h, w = masks["labels"][slot, 0].shape
    frame_by_phys = {int(f["physical_frame_id"]): f for f in window.get("frame_component_readback", [])}
    bbox_errors = Counter()
    edge_errors = Counter()
    for cand in new_regions:
        frame_rows = cand.get("frames", [])
        if len(frame_rows) != 5:
            bbox_errors["candidate_frame_count"] += 1
            continue
        frame_by_index = {i: frame_rows[i] for i in range(5)}
        valid_idxs = []
        for frame_idx, row in frame_by_index.items():
            if int(row.get("physical_frame_id", -1)) != phys[frame_idx]:
                bbox_errors["candidate_physical_frame_order"] += 1
                continue
            if bool(row.get("valid")):
                valid_idxs.append(frame_idx)
                comp_id = row.get("component_id")
                box = row.get("box_xyxy_norm")
                if comp_id is None or box is None:
                    bbox_errors["valid_candidate_missing_component_or_box"] += 1
                    continue
                plane = masks["labels"][slot, frame_idx]
                env = _envelope(plane == int(comp_id))
                if env is None:
                    bbox_errors["candidate_component_absent_from_label"] += 1
                    continue
                px, area = env
                x1, y1, x2, y2 = px
                expected = [x1 / w, y1 / h, x2 / w, y2 / h]
                if len(box) != 4 or any(not _near(a, b) for a, b in zip(expected, box)):
                    bbox_errors["candidate_box_not_label_envelope"] += 1
                if not all(0.0 <= float(v) <= 1.0 for v in box) or not (box[2] > box[0] and box[3] > box[1]):
                    bbox_errors["candidate_geometry_illegal"] += 1
                component = next((x for x in frame_by_phys[phys[frame_idx]].get("components", [])
                                  if int(x["component_id"]) == int(comp_id)), None)
                if component is None or not component.get("accepted") or int(component.get("area_px", -1)) != area:
                    bbox_errors["candidate_component_not_accepted"] += 1
            else:
                if row.get("box_xyxy_norm") is not None or row.get("component_id") is not None:
                    bbox_errors["missing_frame_has_geometry"] += 1
        if len(valid_idxs) != int(cand.get("supported_frame_count", -1)):
            bbox_errors["supported_frame_count"] += 1
        if not (2 in valid_idxs and (1 in valid_idxs) and (3 in valid_idxs)):
            bbox_errors["not_two_sided_center_support"] += 1
        if int(cand.get("center_component_id", -1)) != int(frame_by_index[2].get("component_id", -2)):
            bbox_errors["center_component_id_mismatch"] += 1

        edges = cand.get("accepted_edges", [])
        if len(edges) != int(cand.get("accepted_adjacent_match_count", -1)):
            edge_errors["accepted_edge_count"] += 1
        edge_keys = set()
        for edge in edges:
            i, j = int(edge.get("from_frame_index", -1)), int(edge.get("to_frame_index", -1))
            sid, did = int(edge.get("source_component_id", -1)), int(edge.get("target_component_id", -1))
            if j != i + 1 or i < 0 or j > 4:
                edge_errors["nonadjacent_or_wrong_direction"] += 1
                continue
            if not (_near(frame_by_index[i].get("component_id"), sid) and _near(frame_by_index[j].get("component_id"), did)):
                edge_errors["edge_component_path_mismatch"] += 1
            f_iou = float(edge.get("forward_warped_iou", 0.0))
            b_iou = float(edge.get("backward_warped_iou", 0.0))
            if not (f_iou > 0.0 and b_iou > 0.0):
                edge_errors["nonpositive_reciprocal_iou"] += 1
            if not _near(edge.get("reciprocal_score"), min(f_iou, b_iou)):
                edge_errors["reciprocal_score_not_min"] += 1
            key = (i, j, sid, did)
            if key in edge_keys:
                edge_errors["duplicate_track_edge"] += 1
            edge_keys.add(key)
            transition = window.get("transition_readback", [])[i].get("match_readback", {})
            stored = transition.get("accepted_edges", [])
            if not any(int(x.get("source_component_id", -1)) == sid
                       and int(x.get("target_component_id", -1)) == did
                       and _near(x.get("forward_warped_iou"), f_iou)
                       and _near(x.get("backward_warped_iou"), b_iou)
                       for x in stored):
                edge_errors["track_edge_not_in_transition_readback"] += 1
        if not any(int(e.get("from_frame_index", -1)) == 1 and int(e.get("to_frame_index", -1)) == 2 for e in edges):
            edge_errors["missing_left_center_edge"] += 1
        if not any(int(e.get("from_frame_index", -1)) == 2 and int(e.get("to_frame_index", -1)) == 3 for e in edges):
            edge_errors["missing_right_center_edge"] += 1

    audit.check(f"{ALIASES[cid]}_slot{slot}_all_new_boxes_match_private_labels", not bbox_errors,
                {"new_regions": len(new_regions), "candidate_frame_boxes_checked": sum(int(c.get("supported_frame_count", 0)) for c in new_regions),
                 "errors": _counter_dict(bbox_errors)})
    audit.check(f"{ALIASES[cid]}_slot{slot}_stored_mutual_edge_integrity", not edge_errors,
                {"candidate_edges_checked": sum(len(c.get("accepted_edges", [])) for c in new_regions),
                 "errors": _counter_dict(edge_errors),
                 "limitation": "checks saved reciprocal IoU/readback consistency; dense optical flow was not serialized for independent recomputation"})

    # Check every stored transition's accepted edges for positive reciprocal scores
    # and one-to-one mutual nearest endpoint use.
    transition_errors = Counter()
    transition_edges = 0
    for ti, transition in enumerate(window.get("transition_readback", [])):
        mr = transition.get("match_readback", {})
        edges = mr.get("accepted_edges", [])
        positive_rows = mr.get("positive_pair_edges", [])
        frame_rows = window.get("frame_component_readback", [])
        src_components = frame_rows[ti].get("components", []) if len(frame_rows) == 5 else []
        dst_components = frame_rows[ti + 1].get("components", []) if len(frame_rows) == 5 else []
        src_order = {int(c["component_id"]): tuple(int(v) for v in c["row_major_key"]) + (int(c["component_id"]),)
                     for c in src_components if c.get("accepted")}
        dst_order = {int(c["component_id"]): tuple(int(v) for v in c["row_major_key"]) + (int(c["component_id"]),)
                     for c in dst_components if c.get("accepted")}
        used_src, used_dst = set(), set()
        for edge in edges:
            transition_edges += 1
            sid, did = int(edge.get("source_component_id", -1)), int(edge.get("target_component_id", -1))
            pair = (sid, did)
            if pair in used_src or sid in used_src or did in used_dst:
                transition_errors["not_one_to_one_mutual_endpoints"] += 1
            used_src.add(sid)
            used_dst.add(did)
            if float(edge.get("forward_warped_iou", 0.0)) <= 0.0 or float(edge.get("backward_warped_iou", 0.0)) <= 0.0:
                transition_errors["nonpositive_accepted_edge"] += 1
            if not _near(edge.get("reciprocal_score"), min(float(edge.get("forward_warped_iou", 0.0)), float(edge.get("backward_warped_iou", 0.0)))):
                transition_errors["bad_reciprocal_score"] += 1
            if edge.get("status") != "accepted_mutual_nearest":
                transition_errors["accepted_edge_status"] += 1
        # `match_readback` uses the fixed per-frame segmentation masks.
        # `source_components` / `target_components` on the transition are
        # separate directional residual diagnostics, not the matching roster.
        src_ids = set(src_order)
        dst_ids = set(dst_order)
        for edge in edges:
            if int(edge.get("source_component_id", -1)) not in src_ids or int(edge.get("target_component_id", -1)) not in dst_ids:
                transition_errors["edge_endpoint_not_accepted_frame_component"] += 1
        # Recompute reciprocal-nearest selection from the serialized positive
        # directional IoUs. Source choices use forward IoU; reverse choices use
        # backward IoU; exact ties use the component row-major keys.
        eligible_positive = []
        for edge in positive_rows:
            f_iou = float(edge.get("forward_warped_iou", 0.0))
            b_iou = float(edge.get("backward_warped_iou", 0.0))
            sid = int(edge.get("source_component_id", -1))
            did = int(edge.get("target_component_id", -1))
            if f_iou > 0.0 and b_iou > 0.0:
                if sid not in src_order or did not in dst_order:
                    transition_errors["positive_pair_endpoint_not_accepted_frame_component"] += 1
                else:
                    eligible_positive.append(edge)
        source_best: dict[int, int] = {}
        target_best: dict[int, int] = {}
        for sid in src_order:
            options = [e for e in eligible_positive if int(e["source_component_id"]) == sid]
            if options:
                best_value = max(float(e["forward_warped_iou"]) for e in options)
                best = min((int(e["target_component_id"]) for e in options
                            if float(e["forward_warped_iou"]) == best_value), key=lambda did: dst_order[did])
                source_best[sid] = best
        for did in dst_order:
            options = [e for e in eligible_positive if int(e["target_component_id"]) == did]
            if options:
                best_value = max(float(e["backward_warped_iou"]) for e in options)
                best = min((int(e["source_component_id"]) for e in options
                            if float(e["backward_warped_iou"]) == best_value), key=lambda sid: src_order[sid])
                target_best[did] = best
        expected_mutual = {(sid, did) for sid, did in source_best.items() if target_best.get(did) == sid}
        actual_mutual = {(int(e.get("source_component_id", -1)), int(e.get("target_component_id", -1)))
                         for e in edges}
        if expected_mutual != actual_mutual:
            transition_errors["mutual_nearest_pair_set_mismatch"] += 1
        for edge in positive_rows:
            pair = (int(edge.get("source_component_id", -1)), int(edge.get("target_component_id", -1)))
            f_iou = float(edge.get("forward_warped_iou", 0.0))
            b_iou = float(edge.get("backward_warped_iou", 0.0))
            expected_accepted = f_iou > 0.0 and b_iou > 0.0 and pair in expected_mutual
            observed_accepted = edge.get("status") == "accepted_mutual_nearest"
            if expected_accepted != observed_accepted:
                transition_errors["positive_pair_mutual_status_mismatch"] += 1
    audit.check(f"{ALIASES[cid]}_slot{slot}_transition_edges", not transition_errors,
                {"accepted_edge_count": transition_edges, "errors": _counter_dict(transition_errors),
                 "limitation": "mutual nearest is independently recomputed from saved directional warped-IoU scores and component tie keys; dense optical flow and warped masks are not serialized, so those IoUs are not independently regenerated"})

    # Ranking/cap: derive center row-major order from private label envelopes,
    # not from the generator's ordering helper.
    all_tracks = window.get("all_center_component_tracks", [])
    center_rows = _row_major_component_map(window)
    eligible = [t for t in all_tracks if t.get("eligible_two_sided")]
    def rank_key(t: dict[str, Any]) -> tuple[Any, ...]:
        cc = int(t["center_component_id"])
        med = t.get("median_cycle_error_px")
        return (-int(t.get("supported_frame_count", 0)),
                -int(t.get("accepted_adjacent_match_count", 0)),
                float("inf") if med is None else float(med), center_rows[cc])
    sorted_eligible = sorted(eligible, key=rank_key)
    expected_selected = [int(t["center_component_id"]) for t in sorted_eligible[:4]]
    observed_selected = [int(c["center_component_id"]) for c in new_regions]
    expected_overflow = [int(t["center_component_id"]) for t in sorted_eligible[4:]]
    observed_overflow = [int(t["center_component_id"]) for t in window.get("overflow_tracks", [])]
    other_errors = Counter()
    if {int(t["center_component_id"]) for t in all_tracks} != set(center_rows):
        other_errors["center_component_roster"] += 1
    if int(window.get("eligible_track_count_before_cap", -1)) != len(eligible):
        other_errors["eligible_count"] += 1
    if expected_selected != observed_selected:
        other_errors["candidate_ranking"] += 1
    if expected_overflow != observed_overflow:
        other_errors["overflow_roster_or_order"] += 1
    if int(window.get("overflow_track_count", -1)) != len(expected_overflow):
        other_errors["overflow_count"] += 1
    if int(window.get("new_region_count", -1)) != len(expected_selected):
        other_errors["new_region_count"] += 1
    audit.check(f"{ALIASES[cid]}_slot{slot}_fixed_rank_and_cap", not other_errors,
                {"eligible_before_cap": len(eligible), "selected": len(expected_selected),
                 "overflow": len(expected_overflow), "errors": _counter_dict(other_errors)})

    return {"slot": slot, "center_physical_frame_id": phys[2],
            "old_candidate_count": len(old_tracks), "new_candidate_count": len(candidates),
            "new_region_count": len(new_regions), "old_candidate_ids": sorted(old_by_id),
            "new_candidate_ids": sorted(new_by_id), "eligible_track_count_before_cap": len(eligible),
            "overflow_track_count": len(expected_overflow)}


def _load_gt_support() -> tuple[dict[tuple[str, int], list[float]], set[tuple[str, int]], set[tuple[str, int]], dict[str, Any]]:
    """Use the official post-seal annotation loader, then normalize GT privately.

    The output receipt contains only per-candidate IoUs and binding hashes; it
    never serializes annotation coordinates or query text. The GT-oracle lock
    intentionally covers only the 19 primary positions, not every GT-known
    position in the 23-row support table.
    """
    support = _json(MATCHED / "SUPPORT_READBACK.json")
    roster = _json(MATCHED / "ROSTER_LOCK.json")
    from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
    from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal

    source_cases: dict[str, dict[str, Any]] = {}
    case_roster = {str(row["case_id"]): row for row in roster.get("records", [])}
    if set(case_roster) != set(ALIASES):
        raise ValueError("matched-support roster does not contain the exact audited eight cases")
    for cid in ALIASES:
        row = case_roster[cid]
        source_path = _safe_path(str(row["source_case_path"]))
        if not source_path.is_file() or _sha(source_path) != row.get("source_case_sha256"):
            raise ValueError(f"source case hash mismatch for {ALIASES[cid]}")
        case = load_source_case(source_path)
        if (case.get("case_id") != cid or case.get("dataset") != row.get("dataset")
                or case.get("_source_case_sha256") != row.get("source_case_sha256")):
            raise ValueError(f"source case identity/hash mismatch for {ALIASES[cid]}")
        source_cases[cid] = case
    truths, official_bindings = _truths_after_seal(source_cases)
    gt: dict[tuple[str, int], list[float]] = {}
    for cid, truth_record in truths.items():
        case = source_cases[cid]
        width, height = int(case["input"]["width"]), int(case["input"]["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"nonpositive source dimensions for {ALIASES[cid]}")
        for physical_id, pixel_box in truth_record["truth"].items():
            key = (cid, int(physical_id))
            v = np.asarray(pixel_box, dtype=np.float64)
            if (v.shape != (4,) or not np.isfinite(v).all() or
                    not (v[2] > v[0] and v[3] > v[1]) or key in gt):
                raise ValueError(f"malformed/duplicate official GT box for {ALIASES[cid]}")
            gt[key] = [float(v[0] / width), float(v[1] / height),
                       float(v[2] / width), float(v[3] / height)]
    known_from_support = set()
    primary = set()
    for row in support.get("records", []):
        key = (str(row["case_id"]), int(row["physical_frame_id"]))
        if row.get("GT_known"):
            known_from_support.add(key)
        if row.get("included"):
            primary.add(key)
    official_binding_sha = hashlib.sha256(
        json.dumps(official_bindings, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return gt, known_from_support, primary, {
        "reference_lock_sha256": _sha(MATCHED / "REFERENCE_LOCK.json"),
        "support_sha256": _sha(MATCHED / "SUPPORT_READBACK.json"),
        "roster_lock_sha256": _sha(MATCHED / "ROSTER_LOCK.json"),
        "GT_binding_sha256": official_binding_sha,
        "support_GT_binding_sha256": support.get("GT_binding_sha256"),
        "official_annotation_bindings": official_bindings,
    }


def _compute_geometry(audit: Audit, geometry: dict[str, Any]) -> dict[str, Any]:
    gt, known_support, primary_support, gt_bindings = _load_gt_support()
    window_keys = {
        (cid, int(window["center_physical_frame_id"]))
        for cid in ALIASES
        for window in _json(SCOPE / "cases" / f"{cid}.json").get("windows", [])
    }
    gt_known_at_window_centers = {key for key in window_keys if key in gt}
    ref = _json(MATCHED / "REFERENCE_LOCK.json")
    locked_primary_positions = {
        (str(case["case_id"]), int(anchor["physical_frame_id"]))
        for case in ref.get("records", [])
        for anchor in case.get("references", {}).get("gt_oracle", {}).get("anchors", [])
    }
    audit.check("GT_anchor_support_join", len(window_keys) == 32
                and gt_known_at_window_centers == known_support
                and locked_primary_positions == primary_support
                and len(known_support) == 23 and len(primary_support) == 19,
                {"official_GT_frame_records": len(gt), "audited_window_centers": len(window_keys),
                 "GT_known_support_rows": len(known_support),
                 "GT_known_audited_centers_from_official_loader": len(gt_known_at_window_centers),
                 "locked_primary_positions": len(locked_primary_positions), "primary_support_rows": len(primary_support),
                 "GT_binding_sha256": gt_bindings["GT_binding_sha256"]})
    old_candidate_rows: list[dict[str, Any]] = []
    new_candidate_rows: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    records_by_case: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for cid in ALIASES:
        new, old, _ = _load_case_data(cid)
        records_by_case[cid] = (new, old)
        for window in new.get("windows", []):
            slot = int(window["window_slot"])
            center = int(window["center_physical_frame_id"])
            old_tracks = [t for t in old.get("tracklets", []) if len(t.get("frames", [])) >= 3
                          and int(t["frames"][2].get("physical_frame_id", -1)) == center]
            gt_box = gt.get((cid, center))
            known = gt_box is not None
            primary = (cid, center) in primary_support
            source_iou: float | None = None
            old_ious = []
            for track in old_tracks:
                candidate_id = int(track["candidate_id"])
                frame = next((f for f in track.get("frames", []) if int(f.get("physical_frame_id", -1)) == center), None)
                box = frame.get("box_xyxy_norm") if frame and frame.get("valid") else None
                center_valid = bool(frame and frame.get("valid") and box is not None)
                iou = _box_iou_xyxy(box, gt_box) if known and center_valid else None
                if candidate_id == 1:
                    source_iou = iou
                if iou is not None:
                    old_ious.append(iou)
                old_candidate_rows.append({
                    "case_alias": ALIASES[cid], "dataset": "hc2" if cid.startswith("hc2") else "vidstg",
                    "slot": slot, "pool": "old", "candidate_id": candidate_id,
                    "source": candidate_id == 1, "center_valid": center_valid,
                    "supported_frame_count": sum(bool(f.get("valid")) for f in track.get("frames", [])),
                    "IoU": iou, "unknown_reason": None if iou is not None else ("no_center_GT" if not known else "no_valid_center_box"),
                })
            new_ious = []
            for candidate in window.get("candidates", []):
                candidate_id = int(candidate["candidate_id"])
                frame = next((f for f in candidate.get("frames", []) if int(f.get("physical_frame_id", -1)) == center), None)
                box = frame.get("box_xyxy_norm") if frame and frame.get("valid") else None
                center_valid = bool(frame and frame.get("valid") and box is not None)
                iou = _box_iou_xyxy(box, gt_box) if known and center_valid else None
                if iou is not None:
                    new_ious.append(iou)
                new_candidate_rows.append({
                    "case_alias": ALIASES[cid], "dataset": "hc2" if cid.startswith("hc2") else "vidstg",
                    "slot": slot, "pool": "new", "candidate_id": candidate_id,
                    "source": candidate_id == 1, "center_valid": center_valid,
                    "supported_frame_count": sum(bool(f.get("valid")) for f in candidate.get("frames", [])),
                    "IoU": iou, "unknown_reason": None if iou is not None else ("no_center_GT" if not known else "no_valid_center_box"),
                })
            old_oracle = max(old_ious) if old_ious else None
            new_oracle = max(new_ious) if new_ious else None
            if known and (source_iou is None or old_oracle is None or new_oracle is None):
                raise ValueError(f"GT-known center lacks a valid candidate box: {ALIASES[cid]} slot {slot}")
            old_delta = None if old_oracle is None else 100.0 * (old_oracle - float(source_iou))
            new_delta = None if new_oracle is None else 100.0 * (new_oracle - float(source_iou))
            new_non_source_frames = sum(sum(bool(f.get("valid")) for f in c.get("frames", []))
                                        for c in window.get("candidates", []) if c.get("role") == "new_local_motion_region")
            windows.append({
                "case_alias": ALIASES[cid], "dataset": "hc2" if cid.startswith("hc2") else "vidstg",
                "slot": slot, "primary_support": primary, "GT_known": known,
                "old_candidate_count": len(old_tracks), "new_candidate_count": len(window.get("candidates", [])),
                "new_region_count": sum(c.get("role") == "new_local_motion_region" for c in window.get("candidates", [])),
                "source_IoU": source_iou, "old_oracle_IoU": old_oracle, "new_oracle_IoU": new_oracle,
                "old_delta_pool_pp": old_delta, "new_delta_pool_pp": new_delta,
                "new_minus_old_oracle_pp": None if old_oracle is None or new_oracle is None else 100.0 * (new_oracle - old_oracle),
                "new_nonSource_supported_frames": new_non_source_frames,
                "unknown_reason": None if known else "center_has_no_GT",
            })

    all_candidate_rows = old_candidate_rows + new_candidate_rows
    expected_key = lambda r: (r["case_alias"], r["slot"], r["pool"], r["candidate_id"])
    audit.check("candidate_row_count_and_keys", len(all_candidate_rows) == 221
                and len({expected_key(r) for r in all_candidate_rows}) == 221
                and len(old_candidate_rows) == 152 and len(new_candidate_rows) == 69
                and sum(r["pool"] == "new" and not r["source"] for r in new_candidate_rows) == 37,
                {"all_candidate_rows": len(all_candidate_rows), "old_pool_rows": len(old_candidate_rows),
                 "new_pool_rows_including_source_copies": len(new_candidate_rows),
                 "new_nonSource_region_rows": sum(r["pool"] == "new" and not r["source"] for r in new_candidate_rows)})
    audit.check("all32_windows_and_unknowns", len(windows) == 32 and sum(not r["GT_known"] for r in windows) == 9
                and sum(r["GT_known"] for r in windows) == 23,
                {"window_count": len(windows), "GT_known": sum(r["GT_known"] for r in windows),
                 "GT_unknown": sum(not r["GT_known"] for r in windows)})

    # Bind independent candidate rows and window arithmetic to the post-seal report.
    root_candidates = {(ALIASES.get(r["case_id"], r["case_id"]), int(r["slot"]), str(r["pool"]), int(r["candidate_id"])): r
                       for r in geometry.get("candidate_geometry", [])}
    candidate_mismatches = []
    for row in all_candidate_rows:
        key = expected_key(row)
        root_row = root_candidates.get(key)
        if root_row is None:
            candidate_mismatches.append({"key": list(key), "field": "missing_root_row"})
            continue
        if bool(root_row.get("center_valid")) != row["center_valid"]:
            candidate_mismatches.append({"key": list(key), "field": "center_valid"})
        if int(root_row.get("supported_frame_count", -1)) != row["supported_frame_count"]:
            candidate_mismatches.append({"key": list(key), "field": "supported_frame_count"})
        if row["IoU"] is None:
            if root_row.get("IoU") is not None or root_row.get("unknown_reason") != row["unknown_reason"]:
                candidate_mismatches.append({"key": list(key), "field": "GT_unknown_state"})
        elif (root_row.get("IoU") is None or not _near(root_row.get("IoU"), row["IoU"])
              or root_row.get("unknown_reason") is not None):
            candidate_mismatches.append({"key": list(key), "field": "IoU"})
    if set(root_candidates) != {expected_key(r) for r in all_candidate_rows}:
        candidate_mismatches.append({"field": "root_candidate_key_set"})
    audit.check("root_221_candidate_geometry_match", not candidate_mismatches,
                {"rows_compared": len(all_candidate_rows), "mismatches": candidate_mismatches[:10]})

    # Per-window summaries and grouped primary/all-known arithmetic.
    root_windows = {(ALIASES.get(r["case_id"], r["case_id"]), int(r["slot"])): r for r in geometry.get("windows", [])}
    window_mismatches = []
    for row in windows:
        key = (row["case_alias"], row["slot"])
        rr = root_windows.get(key)
        if rr is None:
            window_mismatches.append({"key": key, "field": "missing_root_window"})
            continue
        expected = {
            "primary_support": row["primary_support"], "center_GT_available": row["GT_known"],
            "old_candidate_count": row["old_candidate_count"], "new_candidate_count": row["new_candidate_count"],
            "new_nonSource_count": row["new_region_count"], "source_IoU": row["source_IoU"],
            "old_oracle_IoU": row["old_oracle_IoU"], "new_oracle_IoU": row["new_oracle_IoU"],
            "old_delta_pool_pp": row["old_delta_pool_pp"], "new_delta_pool_pp": row["new_delta_pool_pp"],
            "new_minus_old_oracle_pp": row["new_minus_old_oracle_pp"],
            "new_nonSource_supported_frames": row["new_nonSource_supported_frames"],
            "unknown_reason": row["unknown_reason"],
        }
        for field, val in expected.items():
            got = rr.get(field)
            if val is None:
                equal = got is None
            elif isinstance(val, bool):
                equal = got is val
            elif isinstance(val, int):
                equal = int(got) == val if got is not None else False
            elif isinstance(val, float):
                equal = got is not None and _near(got, val, 1e-10)
            else:
                equal = got == val
            if not equal:
                window_mismatches.append({"key": key, "field": field})
    audit.check("root_32_window_geometry_match", len(root_windows) == 32 and not window_mismatches,
                {"rows_compared": len(windows), "mismatches": window_mismatches[:12]})

    groups = []
    for support_name, predicate in (("primary19", lambda x: x["primary_support"]),
                                     ("all_known23", lambda x: x["GT_known"])):
        for dataset in ("all", "hc2", "vidstg"):
            subset = [w for w in windows if predicate(w) and (dataset == "all" or w["dataset"] == dataset)]
            old_d = [float(w["old_delta_pool_pp"]) for w in subset if w["old_delta_pool_pp"] is not None]
            new_d = [float(w["new_delta_pool_pp"]) for w in subset if w["new_delta_pool_pp"] is not None]
            diffs = [float(w["new_minus_old_oracle_pp"]) for w in subset if w["new_minus_old_oracle_pp"] is not None]
            groups.append({
                "support": support_name, "dataset": dataset, "count": len(subset),
                "old_mean_delta_pool_pp": float(np.mean(old_d)) if old_d else None,
                "new_mean_delta_pool_pp": float(np.mean(new_d)) if new_d else None,
                "mean_new_minus_old_oracle_pp": float(np.mean(diffs)) if diffs else None,
                "old_positive_pool_count": sum(x > GROUP_TOL for x in old_d),
                "new_positive_pool_count": sum(x > GROUP_TOL for x in new_d),
                "new_better_count": sum(x > GROUP_TOL for x in diffs),
                "new_worse_count": sum(x < -GROUP_TOL for x in diffs),
                "equal_count": sum(abs(x) <= GROUP_TOL for x in diffs),
            })
    root_groups = {(r["support"], r["dataset"]): r for r in geometry.get("groups", [])}
    group_mismatches = []
    for row in groups:
        rr = root_groups.get((row["support"], row["dataset"]))
        if rr is None:
            group_mismatches.append({"key": [row["support"], row["dataset"]], "field": "missing_group"})
            continue
        for field, val in row.items():
            if field in ("support", "dataset"):
                continue
            got = rr.get(field)
            if val is None:
                equal = got is None
            elif isinstance(val, int):
                equal = got is not None and int(got) == val
            else:
                equal = got is not None and _near(got, val, 1e-10)
            if not equal:
                group_mismatches.append({"key": [row["support"], row["dataset"]], "field": field})
    audit.check("root_6_group_aggregates_match", len(root_groups) == 6 and not group_mismatches,
                {"groups_compared": len(groups), "mismatches": group_mismatches[:12]})

    return {
        "candidate_rows": sorted(all_candidate_rows, key=lambda r: (r["case_alias"], r["slot"], r["pool"], r["candidate_id"])),
        "window_rows": sorted(windows, key=lambda r: (r["case_alias"], r["slot"])),
        "groups": groups,
        "gt_bindings": gt_bindings,
        "counts": {"candidate_rows": len(all_candidate_rows), "old_candidate_rows": len(old_candidate_rows),
                   "new_candidate_rows_including_source_copy": len(new_candidate_rows),
                   "new_nonSource_regions": sum(r["pool"] == "new" and not r["source"] for r in new_candidate_rows),
                   "windows": len(windows), "known_windows": sum(w["GT_known"] for w in windows),
                   "unknown_windows": sum(not w["GT_known"] for w in windows),
                   "primary_support_windows": sum(w["primary_support"] for w in windows)},
    }


def run() -> dict[str, Any]:
    audit = Audit()
    runtime, barrier, geometry = _verify_static_custody(audit)
    old_barrier = _json(OLD / "GLOBAL_TRACKLET_BARRIER.json")
    old_case_records = {str(r["case_id"]): r for r in old_barrier.get("case_records", [])}
    static_counts = Counter()
    per_case = []
    window_rows = []
    mask_counts = {}
    for cid, alias in ALIASES.items():
        new, old, masks = _load_case_data(cid)
        audit.check(f"{alias}_sealed_case_state", new.get("status") == "complete"
                    and new.get("case_id") == cid and new.get("GT_read") is False
                    and new.get("scoring_started") is False and new.get("teacher_called") is False
                    and new.get("model_loaded") is False and new.get("GPU_used") is False,
                    {"status": new.get("status"), "GT_read": new.get("GT_read"),
                     "teacher_called": new.get("teacher_called"), "GPU_used": new.get("GPU_used")})
        old_rel = f"artifacts/stvg_short_event_tracklets_v1/tracklets/{cid}.json"
        old_case_sha = _sha(OLD / "tracklets" / f"{cid}.json")
        old_rgb_sha = _sha(OLD / "private_rgb" / f"{cid}.npz")
        old_input = old.get("input_binding", {})
        new_input = new.get("input_bindings", {})
        lock_row = old_case_records.get(cid, {})
        case_join_ok = (
            lock_row.get("path") == old_rel
            and lock_row.get("sha256") == old_case_sha
            and lock_row.get("rgb_npz_sha256") == old_rgb_sha
            and new_input.get("old_tracklet_json_path") == old_rel
            and new_input.get("old_tracklet_json_sha256") == old_case_sha
            and new_input.get("old_private_rgb_npz_sha256") == old_rgb_sha
            and new_input.get("candidate_barrier_record_sha256") == old_input.get("candidate_barrier_record_sha256")
            and old_input.get("candidate_barrier_record_sha256") == new_input.get("candidate_barrier_record_sha256")
            and new_input.get("old_global_barrier_sha256") == _sha(OLD / "GLOBAL_TRACKLET_BARRIER.json")
        )
        audit.check(f"{alias}_old_case_join", case_join_ok,
                    {"joined_by_case_id_and_barrier_file_sha": cid in old_case_records
                     and lock_row.get("sha256") == old_case_sha,
                     "candidate_barrier_record_binding_matches_old_case_input":
                     new_input.get("candidate_barrier_record_sha256") == old_input.get("candidate_barrier_record_sha256")})
        _verify_frame_provenance(audit, cid, new, old)
        mask_counts[alias] = _verify_label_maps(audit, cid, new, masks)
        case_windows = new.get("windows", [])
        audit.check(f"{alias}_four_windows", len(case_windows) == 4,
                    {"window_count": len(case_windows)})
        for window in case_windows:
            summary = _verify_window(audit, cid, window, masks)
            window_rows.append({"case_alias": alias, **summary})
            static_counts["windows"] += 1
            static_counts["new_regions"] += summary.get("new_region_count", 0)
            static_counts["new_candidate_instances_including_source"] += summary.get("new_candidate_count", 0)
            static_counts["old_candidate_instances"] += summary.get("old_candidate_count", 0)
        top_count = int(new.get("counts", {}).get("new_region_count", -1))
        window_sum = sum(int(w.get("new_region_count", 0)) for w in case_windows)
        audit.check(f"{alias}_case_region_count", top_count == window_sum,
                    {"case_summary_new_region_count": top_count, "sum_of_window_region_counts": window_sum})
        per_case.append({"case_alias": alias,
                         "windows": len(case_windows),
                         "new_nonSource_regions": window_sum,
                         "source_candidate_copies": sum(1 for _ in case_windows),
                         "old_candidate_instances": sum(len([t for t in old.get("tracklets", [])
                             if len(t.get("frames", [])) >= 3 and int(t["frames"][2].get("physical_frame_id", -1))
                             == int(w["center_physical_frame_id"])]) for w in case_windows),
                         "mask_components": mask_counts[alias]})
    audit.check("all_output_windows_and_candidate_counts",
                static_counts["windows"] == 32 and static_counts["new_regions"] == 37
                and static_counts["new_candidate_instances_including_source"] == 69
                and static_counts["old_candidate_instances"] == 152,
                {"windows": static_counts["windows"], "new_nonSource_regions": static_counts["new_regions"],
                 "new_candidate_instances_including_Source_copies": static_counts["new_candidate_instances_including_source"],
                 "old_candidate_instances": static_counts["old_candidate_instances"]})
    geometry_readback = _compute_geometry(audit, geometry)
    receipt = {
        "schema": "stvg-local-motion-region-independent-readback-v2",
        "status": "pass" if not audit.failures else "fail",
        "scope": "read-only source custody, private label/envelope, candidate track support/rank, and post-seal center IoU arithmetic",
        "method_limits": {
            "GT_used_for_generation": False,
            "GT_read_after_candidate_barrier": True,
            "official_truth_loader_shared_with_root_scorer": True,
            "IoU_arithmetic_independently_recomputed": True,
            "GT_coordinates_or_query_text_serialized": False,
            "GPU_or_model_used_by_reader": False,
            "RGB_or_video_decoded_by_reader": False,
            "dense_flow_recomputed": False,
            "semantic_identity_verified": False,
            "full_tube_efficacy_measured": False,
            "candidate_selection_tuned_from_GT": False,
        },
        "source_hashes": {
            "runner_sha256": _sha(ROOT / "scripts/run_stvg_local_motion_regions_v1.py"),
            "runtime_sha256": _sha(RUNTIME_PATH),
            "barrier_sha256": _sha(BARRIER_PATH),
            "postseal_geometry_sha256": _sha(GEOMETRY_PATH),
            "matched_reference_lock_sha256": _sha(MATCHED / "REFERENCE_LOCK.json"),
            "support_readback_sha256": _sha(MATCHED / "SUPPORT_READBACK.json"),
            "roster_lock_sha256": _sha(MATCHED / "ROSTER_LOCK.json"),
        },
        "counts": {
            "runtime_pins": len(runtime.get("pins", {})),
            "barrier_outputs": len(barrier.get("files", [])),
            "cases": len(per_case),
            "windows": static_counts["windows"],
            "old_candidate_instances": static_counts["old_candidate_instances"],
            "new_nonSource_regions": static_counts["new_regions"],
            "new_candidate_instances_including_Source_copies": static_counts["new_candidate_instances_including_source"],
            "candidate_geometry_rows": geometry_readback["counts"]["candidate_rows"],
            "GT_known_windows": geometry_readback["counts"]["known_windows"],
            "GT_unknown_windows": geometry_readback["counts"]["unknown_windows"],
            "primary_support_windows": geometry_readback["counts"]["primary_support_windows"],
        },
        "case_counts": per_case,
        "window_candidate_geometry": geometry_readback["window_rows"],
        "candidate_iou_rows": geometry_readback["candidate_rows"],
        "group_aggregates": geometry_readback["groups"],
        "GT_binding": geometry_readback["gt_bindings"],
        "checks": audit.checks,
        "failures": audit.failures,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT_PATH.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "status": result["status"], "counts": result["counts"],
        "failure_count": len(result["failures"]), "receipt": str(OUTPUT_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, sort_keys=True))
