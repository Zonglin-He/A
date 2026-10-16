"""Prepare a GT-free manifest for short event-local tracklet proposals.

This module does not decode video, load a model, access labels, or run a
tracking experiment when preparing the manifest.  The only tracking entrypoint
accepts already-decoded grayscale arrays and delegates unchanged to the
existing adjacent-frame helper.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_adjacent_motion_opd_v1"
SOURCE_CASES = ROOT / "artifacts/stvg_mllm_candidate_feedback_v1/source_cases"
QA_BASE = ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1"
CANONICAL_BASE = ROOT / "artifacts/stvg_target_teacher_distillation_v1/canonical_inputs"
OUT = ROOT / "artifacts/stvg_short_event_tracklets_v1"
PROTOCOL = ROOT / "protocols/stvg_short_event_tracklets_v1.proposal.json"
CASE_IDS = tuple(
    [f"hc2_{i:03d}" for i in range(4)]
    + [f"vidstg_{i:03d}" for i in range(4)]
)
SCHEMA = "stvg-short-event-tracklets-plan-v1"
TRACK_HELPER = "scripts/stvg_adjacent_affine_tracklet_v1.py"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _root_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _write_once(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prepared artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_cpu_pickle(path: Path) -> Any:
    """Load only the locally accepted no-GT metadata cache on CPU."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _source_receipt(source_path: Path, expected_sha: str) -> dict[str, Any]:
    receipt_path = source_path.with_suffix(".json")
    receipt = _read_json(receipt_path)
    if (
        receipt.get("status") != "accepted_complete_source_only_case"
        or receipt.get("GT_read") is not False
        or receipt.get("sha256") != expected_sha
        or _sha_file(source_path) != expected_sha
        or receipt.get("bytes") != source_path.stat().st_size
    ):
        raise ValueError(f"source-only receipt/hash check failed: {source_path.name}")
    return receipt


def _physical_frame_map(request: Mapping[str, Any], case_id: str):
    rows = request.get("frame_map")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"canonical request has no frame map for {case_id}")
    physical_to_local: dict[int, int] = {}
    local_to_physical: dict[int, int] = {}
    for row in rows:
        physical = int(row["physical_frame_id"])
        local = int(row["local_frame_index"])
        if physical in physical_to_local or local in local_to_physical or local < 0:
            raise ValueError(f"duplicate/invalid canonical frame map for {case_id}")
        if not math.isfinite(float(row["source_timestamp_s"])):
            raise ValueError(f"non-finite source timestamp in frame map for {case_id}")
        physical_to_local[physical] = local
        local_to_physical[local] = physical
    if set(physical_to_local.values()) != set(range(len(rows))):
        raise ValueError(f"canonical local frame map is not dense from zero for {case_id}")
    if any(
        b != a + 1
        for a, b in zip(sorted(physical_to_local), sorted(physical_to_local)[1:])
    ):
        raise ValueError(f"canonical physical frame map is not contiguous for {case_id}")
    if any(
        physical_to_local[b] != physical_to_local[a] + 1
        for a, b in zip(sorted(physical_to_local), sorted(physical_to_local)[1:])
    ):
        raise ValueError(f"canonical physical/local frame map is not adjacent for {case_id}")
    return physical_to_local, local_to_physical


def _nearest_unique_grid_centers(
    interval: Sequence[int], source_grid_ids: Sequence[int]
) -> list[dict[str, Any]]:
    start, end = int(interval[0]), int(interval[1])
    if end <= start:
        raise ValueError("QA interval must be a non-empty physical half-open interval")
    eligible = [int(frame_id) for frame_id in source_grid_ids if start <= int(frame_id) < end]
    if len(eligible) < 4:
        raise ValueError("QA interval contains fewer than four source-grid observations")
    targets = [start + (slot + 0.5) * (end - start) / 4.0 for slot in range(4)]
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for slot, target in enumerate(targets):
        center = min(eligible, key=lambda frame_id: (abs(frame_id - target), frame_id))
        if center in used:
            raise ValueError("four uniform QA targets collapse to duplicate source-grid centers")
        used.add(center)
        selected.append(
            {
                "slot": slot,
                "target_physical_frame_id": float(target),
                "center_physical_frame_id": center,
            }
        )
    if [row["center_physical_frame_id"] for row in selected] != sorted(used):
        raise ValueError("snapped QA centers are not in chronological order")
    return selected


def _read_only_request_binding(
    request_path: Path, request: Mapping[str, Any], candidate_record: Mapping[str, Any], case_id: str
) -> None:
    if _sha_file(request_path) != candidate_record.get("request_sha256"):
        raise ValueError(f"canonical request file SHA differs from the old barrier for {case_id}")


def _xyxy(value: Any, field: str) -> list[float]:
    import numpy as np

    box = np.asarray(value, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError(f"{field} must be a finite normalized xyxy box")
    return [float(v) for v in box]


def _legal_box(box: Sequence[float]) -> bool:
    x1, y1, x2, y2 = (float(v) for v in box)
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def _seed_at_center(
    candidate_id: int,
    candidate: Mapping[str, Any],
    source_box_at_center: Sequence[float],
) -> tuple[list[float] | None, str]:
    if candidate_id == 1:
        result = [float(v) for v in source_box_at_center]
        return (result, "source_native_box_at_center") if _legal_box(result) else (None, "source_box_invalid")

    seed_box = _xyxy(candidate.get("anchor_box_xyxy"), "candidate anchor")
    if not _legal_box(seed_box):
        return None, "existing_region_proposal_invalid_no_clip"
    return seed_box, "existing_anchor_box_xyxy_copied_unchanged_to_each_center"


def build_manifest() -> dict[str, Any]:
    """Build a no-decode, no-GT plan from the exact frozen old-eight inputs."""
    source_roster_path = BASE / "ROSTER_LOCK.json"
    candidate_barrier_path = BASE / "CANDIDATE_INPUT_BARRIER.json"
    qa_barrier_path = QA_BASE / "STUDENT_GLOBAL_BARRIER.json"
    roster = _read_json(source_roster_path)
    candidate_barrier = _read_json(candidate_barrier_path)
    qa_barrier = _read_json(qa_barrier_path)

    if roster.get("status") != "locked" or roster.get("count") != 8 or roster.get("GT_read") is not False:
        raise ValueError("the old-eight source roster is not locked and no-GT")
    if candidate_barrier.get("status") != "complete" or candidate_barrier.get("count") != 8 or candidate_barrier.get("GT_read") is not False:
        raise ValueError("the old adjacent-motion candidate barrier is not complete and no-GT")
    if qa_barrier.get("status") != "complete" or qa_barrier.get("count") != 8 or qa_barrier.get("GT_read") is not False:
        raise ValueError("the reused temporal-QA student barrier is not complete and no-GT")

    roster_rows = {str(row["case_id"]): row for row in roster.get("records", [])}
    candidate_rows = {str(row["case_id"]): row for row in candidate_barrier.get("records", [])}
    qa_files = {
        Path(str(row["path"])).name: row
        for row in qa_barrier.get("files", [])
    }
    if set(roster_rows) != set(CASE_IDS) or set(candidate_rows) != set(CASE_IDS):
        raise ValueError("locked old-eight rows do not match the fixed case list")
    if set(qa_files) != {f"{case_id}.pt" for case_id in CASE_IDS}:
        raise ValueError("temporal-QA barrier is not exactly the fixed old-eight set")

    cases: list[dict[str, Any]] = []
    for case_id in CASE_IDS:
        roster_row = roster_rows[case_id]
        candidate_record = candidate_rows[case_id]
        source_path = _root_path(roster_row["source_case_path"])
        if source_path.parent != SOURCE_CASES.resolve():
            raise ValueError(f"source case escaped the fixed source directory: {case_id}")
        source_sha = str(roster_row["source_case_sha256"])
        _source_receipt(source_path, source_sha)
        source_case = _load_cpu_pickle(source_path)
        if (
            source_case.get("GT_read") is not False
            or source_case.get("original_source_only") is not True
            or source_case.get("case_id") != case_id
            or candidate_record.get("GT_read") is not False
            or candidate_record.get("source_case_sha256") != source_sha
        ):
            raise ValueError(f"source/candidate binding is not no-GT and exact for {case_id}")

        qa_row = qa_files[f"{case_id}.pt"]
        qa_path = _root_path(qa_row["path"])
        if qa_path.parent != (QA_BASE / "student").resolve() or _sha_file(qa_path) != qa_row["sha256"]:
            raise ValueError(f"temporal-QA student file binding failed for {case_id}")
        qa_case = _load_cpu_pickle(qa_path)
        if (
            qa_case.get("GT_read") is not False
            or qa_case.get("case_id") != case_id
            or qa_case.get("source_case_sha256") != source_sha
            or qa_case.get("frame_ids") != list(source_case["frame_ids"])
            or qa_case.get("temporal_fit", {}).get("GT_read") is not False
        ):
            raise ValueError(f"temporal-QA record is not bound to the source grid for {case_id}")
        qa_interval = qa_case["temporal_fit"].get("final_interval")
        if not isinstance(qa_interval, (list, tuple)) or len(qa_interval) != 2:
            raise ValueError(f"temporal-QA interval missing for {case_id}")
        if any(isinstance(v, bool) or int(v) != v for v in qa_interval):
            raise ValueError(f"temporal-QA interval endpoints must be integral physical IDs for {case_id}")
        qa_interval = [int(qa_interval[0]), int(qa_interval[1])]
        source_grid_ids = [int(v) for v in source_case["frame_ids"]]
        source_grid_ids_from_qa = [int(v) for v in qa_case["frame_ids"]]
        if source_grid_ids != source_grid_ids_from_qa:
            raise ValueError(f"source grid differs from the reused QA grid for {case_id}")

        canonical_dir = (CANONICAL_BASE / case_id).resolve()
        request_path = canonical_dir / "REQUEST.json"
        request = _read_json(request_path)
        if request.get("source_video_sha256") != source_case["input"]["video_sha256"]:
            raise ValueError(f"canonical request is not bound to source video for {case_id}")
        if request.get("canonical_video_sha256") != candidate_record.get("canonical_video_sha256"):
            raise ValueError(f"canonical request differs from the old candidate barrier for {case_id}")
        _read_only_request_binding(request_path, request, candidate_record, case_id)
        video_path = Path(str(request["video_path"])).resolve()
        if video_path.parent != canonical_dir or not video_path.is_file():
            raise ValueError(f"canonical video path is outside its fixed case directory for {case_id}")
        physical_to_local, _ = _physical_frame_map(request, case_id)

        geometry_rows = candidate_record.get("candidate_geometry", [])
        by_candidate_id = {int(row["candidate_id"]): row for row in geometry_rows}
        visible_ids = [int(v) for v in candidate_record.get("displayed_candidate_ids", [])]
        if not visible_ids or visible_ids[0] != 1 or len(visible_ids) > 5 or len(visible_ids) != len(set(visible_ids)):
            raise ValueError(f"fixed displayed seed set must be Source plus at most four seeds for {case_id}")
        if any(candidate_id not in by_candidate_id for candidate_id in visible_ids):
            raise ValueError(f"displayed candidate is absent from the existing seed catalog for {case_id}")
        if by_candidate_id[1].get("is_Source") is not True:
            raise ValueError(f"candidate 1 is not the frozen Source proposal for {case_id}")

        native_boxes = source_case.get("native_boxes")
        try:
            import numpy as np

            native_boxes = np.asarray(native_boxes, dtype=np.float64)
        except Exception as exc:
            raise ValueError(f"source native boxes are unavailable for {case_id}") from exc
        if native_boxes.shape != (len(source_grid_ids), 4) or not np.isfinite(native_boxes).all():
            raise ValueError(f"source native boxes do not match the locked grid for {case_id}")

        original_observations = [int(v) for v in candidate_record.get("physical_frame_ids", [])]
        if len(original_observations) != 4 or original_observations != sorted(original_observations):
            raise ValueError(f"old four-observation binding changed for {case_id}")
        source_observation_positions = [int(v) for v in candidate_record.get("positions", [])]
        if len(source_observation_positions) != 4 or [source_grid_ids[i] for i in source_observation_positions] != original_observations:
            raise ValueError(f"source-grid observation mapping changed for {case_id}")
        source_anchor = _xyxy(by_candidate_id[1]["anchor_box_xyxy"], "frozen Source anchor")
        if source_anchor != [float(v) for v in native_boxes[source_observation_positions[1]]]:
            raise ValueError(f"existing Source seed anchor does not match the source-grid observation for {case_id}")

        if source_grid_ids != sorted(set(source_grid_ids)):
            raise ValueError(f"source grid physical IDs must be unique and ordered for {case_id}")
        covered_source_grid_ids = [frame_id for frame_id in source_grid_ids if frame_id in physical_to_local]
        centers = _nearest_unique_grid_centers(qa_interval, covered_source_grid_ids)
        available_min, available_max = min(physical_to_local), max(physical_to_local)
        center_records: list[dict[str, Any]] = []
        for center_row in centers:
            center_id = int(center_row["center_physical_frame_id"])
            if center_id not in physical_to_local:
                raise ValueError(f"uniform QA center is absent from canonical frame map for {case_id}")
            left = max(available_min, center_id - 2)
            right = min(available_max, center_id + 2)
            physical_ids = list(range(left, right + 1))
            local_indices = [physical_to_local.get(frame_id) for frame_id in physical_ids]
            if any(value is None for value in local_indices) or any(
                b != a + 1 for a, b in zip(local_indices, local_indices[1:])
            ):
                raise ValueError(f"short physical window is not fully covered by canonical input for {case_id}")
            source_grid_position = source_grid_ids.index(center_id)
            source_box = _xyxy(native_boxes[source_grid_position], "source native center box")
            tracklets: list[dict[str, Any]] = []
            for candidate_id in visible_ids:
                candidate = by_candidate_id[candidate_id]
                anchor_box, anchor_rule = _seed_at_center(
                    candidate_id, candidate, source_box
                )
                tracklets.append(
                    {
                        "tracklet_id": f"{case_id}:center{center_id}:proposal{candidate_id}",
                        "candidate_id": candidate_id,
                        "window_id": int(candidate["window_id"]),
                        "proposal_role": "Source" if candidate_id == 1 else "fixed_existing_seed",
                        "proposal_semantics": (
                            "frozen_source_prediction_at_this_center"
                            if candidate_id == 1
                            else "fixed_region_proposal_geometry_not_object_identity"
                        ),
                        "anchor_box_xyxy_norm": anchor_box,
                        "anchor_status": "valid" if anchor_box is not None else "invalid_no_clip",
                        "anchor_rule": anchor_rule,
                        "physical_frame_ids": physical_ids,
                        "local_frame_indices": [int(value) for value in local_indices],
                        "anchor_index": physical_ids.index(center_id),
                        "cross_window_identity_claim": False,
                        "same_seed_id_across_centers_means": "reused proposal hypothesis only",
                    }
                )
            center_records.append(
                {
                    **center_row,
                    "source_grid_position": source_grid_position,
                    "physical_frame_ids": physical_ids,
                    "local_frame_indices": [int(value) for value in local_indices],
                    "clip_reason": "canonical_video_physical_coverage_only",
                    "matched_tracklets": tracklets,
                    "matched_pair_key": _canonical_sha(
                        {
                            "case_id": case_id,
                            "center_physical_frame_id": center_id,
                            "tracklets": [
                                {
                                    "candidate_id": item["candidate_id"],
                                    "anchor_box_xyxy_norm": item["anchor_box_xyxy_norm"],
                                }
                                for item in tracklets
                            ],
                            "physical_frame_ids": physical_ids,
                        }
                    ),
                }
            )

        cases.append(
            {
                "case_id": case_id,
                "dataset": str(source_case["dataset"]),
                "cohort": "historically_exposed_old8",
                "source_case_path": source_path.relative_to(ROOT).as_posix(),
                "source_case_sha256": source_sha,
                "source_model_state_sha256": str(source_case["source_model_state_sha256"]),
                "source_grid_count": len(source_grid_ids),
                "source_grid_frame_ids_sha256": _canonical_sha(source_grid_ids),
                "qa_student_path": qa_path.relative_to(ROOT).as_posix(),
                "qa_student_sha256": str(qa_row["sha256"]),
                "qa_interval_physical_half_open": qa_interval,
                "qa_interval_provenance": "reused sealed temporal_fit.final_interval; no new teacher call",
                "candidate_barrier_record_sha256": _canonical_sha(candidate_record),
                "fixed_displayed_candidate_ids": visible_ids,
                "fixed_existing_window_ids": [int(by_candidate_id[i]["window_id"]) for i in visible_ids],
                "canonical_request_path": request_path.relative_to(ROOT).as_posix(),
                "canonical_request_sha256": _sha_file(request_path),
                "canonical_video_path": video_path.relative_to(ROOT).as_posix(),
                "canonical_video_sha256": str(request["canonical_video_sha256"]),
                "canonical_pixel_sha256": str(request["canonical_pixel_sha256"]),
                "source_video_sha256": str(request["source_video_sha256"]),
                "canonical_physical_coverage_half_open": [available_min, available_max + 1],
                "centers": center_records,
                "GT_read": False,
                "RGB_read": False,
                "model_loaded": False,
                "GPU_used": False,
                "video_decoded": False,
            }
        )

    return {
        "schema": SCHEMA,
        "status": "prepared_manifest_not_executed",
        "cases": cases,
        "case_count": len(cases),
        "center_count": sum(len(case["centers"]) for case in cases),
        "tracklet_proposal_count": sum(
            len(center["matched_tracklets"])
            for case in cases
            for center in case["centers"]
        ),
        "source_bindings": {
            "source_roster_lock_path": BASE.relative_to(ROOT).as_posix() + "/ROSTER_LOCK.json",
            "source_roster_lock_sha256": _sha_file(BASE / "ROSTER_LOCK.json"),
            "candidate_input_barrier_path": BASE.relative_to(ROOT).as_posix() + "/CANDIDATE_INPUT_BARRIER.json",
            "candidate_input_barrier_sha256": _sha_file(BASE / "CANDIDATE_INPUT_BARRIER.json"),
            "qa_student_barrier_path": QA_BASE.relative_to(ROOT).as_posix() + "/STUDENT_GLOBAL_BARRIER.json",
            "qa_student_barrier_sha256": _sha_file(QA_BASE / "STUDENT_GLOBAL_BARRIER.json"),
            "track_helper_path": TRACK_HELPER,
            "track_helper_sha256": _sha_file(ROOT / TRACK_HELPER),
            "proposal_path": PROTOCOL.relative_to(ROOT).as_posix(),
            "proposal_sha256": _sha_file(PROTOCOL),
        },
        "selection_contract": {
            "qa_interval": "physical frame IDs, half-open [start,end), reused without GT",
            "center_targets": "four equal-width QA-interval bin centers",
            "source_grid_snap": "nearest source-grid physical ID; ties choose lower ID; duplicate snap aborts preparation",
            "candidate_set": "exact existing P0 displayed_candidate_ids; Source plus at most four existing seeds; no refill or retrieval",
            "window": "physical center +/-2 frame IDs, clipped only to canonical video coverage",
            "cross_window_identity_claim": False,
        },
        "components": {
            "shortLK": {
                "status": "planned_after_root_review",
                "implementation": "wrapper calls existing track_adjacent unchanged for each center/seed window",
                "helper_path": TRACK_HELPER,
                "new_tracker": False,
            },
            "shortLK_frozenROI": {
                "status": "pending_feature_capture_contract",
                "implementation": None,
                "must_reuse_existing_frozen_TA_STVG_model": True,
                "new_backbone_or_tracker": False,
            },
        },
        "GT_read": False,
        "RGB_read": False,
        "model_loaded": False,
        "GPU_used": False,
        "video_decoded": False,
    }


def track_short_window_lk(
    gray_frames: Sequence[Any],
    physical_frame_ids: Sequence[int],
    center_physical_frame_id: int,
    anchor_box_xyxy_norm: Sequence[float],
    coverage_half_open: Sequence[int],
) -> dict[str, Any]:
    """Run the original LK/RANSAC helper on one independent +/-2-frame window.

    ``gray_frames`` are supplied by a separately authorized caller. This
    function performs no decoding and will not connect windows or infer object
    identity across centers.
    """
    if len(coverage_half_open) != 2:
        raise ValueError("coverage_half_open must contain [start,end)")
    coverage_start, coverage_end = map(int, coverage_half_open)
    if coverage_end <= coverage_start:
        raise ValueError("invalid canonical physical coverage")
    center = int(center_physical_frame_id)
    if not coverage_start <= center < coverage_end:
        raise ValueError("center is outside canonical physical coverage")
    expected = list(range(max(coverage_start, center - 2), min(coverage_end - 1, center + 2) + 1))
    ids = [int(value) for value in physical_frame_ids]
    if ids != expected or center not in ids:
        raise ValueError("supplied frames must be the exact clipped physical +/-2 window")
    if len(gray_frames) != len(ids) or not 1 <= len(ids) <= 5:
        raise ValueError("one grayscale frame is required for each short-window physical ID")
    anchor_box = _xyxy(anchor_box_xyxy_norm, "short-window anchor")
    if not _legal_box(anchor_box):
        raise ValueError("invalid anchors must remain explicit and must not be passed to LK")
    from scripts.stvg_adjacent_affine_tracklet_v1 import SCHEMA as TRACK_SCHEMA
    from scripts.stvg_adjacent_affine_tracklet_v1 import track_adjacent

    result = track_adjacent(gray_frames, ids, ids.index(center), anchor_box)
    return {
        "schema": "stvg-short-event-tracklet-result-v1",
        "track_helper_schema": TRACK_SCHEMA,
        "center_physical_frame_id": center,
        "physical_frame_ids": ids,
        "anchor_index": ids.index(center),
        "tracklet": result,
        "window_independent": True,
        "cross_window_identity_claim": False,
        "GT_read": False,
    }


def write_prepared_manifest(path: Path | None = None) -> dict[str, Any]:
    """Write the metadata-only plan once; this never reads RGB or runs LK."""
    output = OUT / "PREPARED_MANIFEST.json" if path is None else Path(path)
    manifest = build_manifest()
    _write_once(output, manifest)
    return manifest


if __name__ == "__main__":
    raise SystemExit(
        "Preparation is intentionally gated: root must review and authorize the proposal before calling write_prepared_manifest()."
    )
