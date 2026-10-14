"""Rebuild only the order-swap teacher inputs with fixed, base-layout badges.

CPU-only preparation. It reuses the sealed candidate geometries, original
prompts, and control ID/color/draw-order maps. It decodes the canonical source
video and never reads labels, annotations, or teacher outputs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_stvg_scene_object_id_v1 import (
    IMAGE_SIDE,
    PALETTE,
    _letterbox,
    _raster_box,
)
BASE = ROOT / "artifacts/stvg_motion_guided_instance_opd_v1"
OUT = BASE / "display_order_layout001"
CANONICAL_INPUTS = ROOT / "artifacts/stvg_target_teacher_distillation_v1/canonical_inputs"
OLD_INPUT_BARRIER = BASE / "CANDIDATE_INPUT_BARRIER.json"
OLD_JOB_LOCK = BASE / "TEACHER_JOB_LOCK.json"
TARGET_KINDS = ("order_swap", "joint_swap")
SCHEMA = "stvg-motion-display-order-layout001-v1"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_once(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _root_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("manifest path escaped repository root")
    return path


def _manifest_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("input barrier has no files list")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("malformed input file receipt")
        path = row["path"]
        if path in out:
            raise ValueError("duplicate input path in old barrier")
        out[path] = row
    return out


def _verify_manifest_file(index: Mapping[str, Mapping[str, Any]], relative: str) -> Path:
    if relative not in index:
        raise ValueError(f"old candidate barrier does not bind {relative}")
    path = _root_path(relative)
    expected = index[relative]
    if not path.is_file() or _sha_file(path) != expected.get("sha256"):
        raise ValueError(f"old input bytes do not match candidate barrier: {relative}")
    if path.stat().st_size != expected.get("bytes"):
        raise ValueError(f"old input size does not match candidate barrier: {relative}")
    return path


def _safe_video_request(case_id: str, record: Mapping[str, Any]) -> tuple[dict[str, Any], Path, dict[int, int], Path]:
    request_path = CANONICAL_INPUTS / case_id / "REQUEST.json"
    request = _read_json(request_path)
    # Read only the canonical-media fields here; query_text is deliberately unused.
    video = Path(str(request["video_path"])).resolve()
    if not video.is_relative_to(CANONICAL_INPUTS.resolve()):
        raise ValueError(f"canonical video path escaped input root for {case_id}")
    if video.parent != (CANONICAL_INPUTS / case_id).resolve():
        raise ValueError(f"canonical video is not in its case directory for {case_id}")
    if _sha_file(video) != request.get("canonical_video_sha256"):
        raise ValueError(f"canonical video SHA mismatch for {case_id}")
    if request.get("canonical_video_sha256") != record.get("canonical_video_sha256"):
        raise ValueError(f"canonical video is not the sealed source for {case_id}")
    if not isinstance(request.get("source_video_sha256"), str):
        raise ValueError(f"source video receipt missing for {case_id}")
    frame_map: dict[int, int] = {}
    for row in request["frame_map"]:
        physical = int(row["physical_frame_id"])
        local = int(row["local_frame_index"])
        if physical in frame_map or local < 0:
            raise ValueError(f"invalid canonical frame mapping for {case_id}")
        frame_map[physical] = local
    return request, video, frame_map, request_path


def _decode_keyframes(
    case_id: str,
    video: Path,
    frame_map: Mapping[int, int],
    frame_ids: list[int],
    expected_hashes: Mapping[str, str],
) -> tuple[list[Image.Image], list[str], list[list[int]]]:
    if len(frame_ids) != 4 or len(set(frame_ids)) != 4:
        raise ValueError(f"expected four unique physical frames for {case_id}")
    targets: dict[int, int] = {}
    for slot, physical in enumerate(frame_ids):
        if physical not in frame_map:
            raise ValueError(f"requested physical frame absent for {case_id}")
        local = int(frame_map[physical])
        if local in targets:
            raise ValueError(f"multiple selected frames map to one decoded frame for {case_id}")
        targets[local] = slot

    images: list[Image.Image | None] = [None] * len(frame_ids)
    hashes: list[str | None] = [None] * len(frame_ids)
    dimensions: list[list[int] | None] = [None] * len(frame_ids)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"cannot open canonical video for {case_id}")
    local_index = 0
    try:
        last = max(targets)
        while local_index <= last and targets:
            ok, bgr = cap.read()
            if not ok:
                raise OSError(f"canonical decode ended before selected frame for {case_id}")
            if local_index in targets:
                slot = targets.pop(local_index)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                digest = _sha_bytes(rgb.tobytes())
                physical = frame_ids[slot]
                if digest != expected_hashes.get(str(physical)):
                    raise ValueError(f"canonical RGB SHA differs from sealed candidate input for {case_id}")
                images[slot] = Image.fromarray(rgb)
                hashes[slot] = digest
                dimensions[slot] = [int(rgb.shape[1]), int(rgb.shape[0])]
            local_index += 1
    finally:
        cap.release()
    if targets or any(value is None for value in images + hashes + dimensions):
        raise ValueError(f"failed to decode every selected frame for {case_id}")
    return list(images), list(hashes), list(dimensions)  # type: ignore[arg-type]


def _control_metadata(case_id: str, kind: str, manifest_index: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], Path, str]:
    relative = f"artifacts/stvg_motion_guided_instance_opd_v1/private_inputs/{case_id}/{kind}_render.json"
    path = _verify_manifest_file(manifest_index, relative)
    digest = _sha_file(path)
    return _read_json(path), path, digest


def _validate_control_maps(case_id: str, kind: str, render: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    ids = render.get("id_map")
    colors = render.get("color_map")
    order = render.get("draw_order")
    if not isinstance(ids, dict) or not isinstance(colors, dict) or not isinstance(order, list):
        raise ValueError(f"malformed control maps for {case_id}/{kind}")
    keys = set(ids)
    if keys != set(colors) or keys != {str(i) for i in range(1, int(job["candidate_count"]) + 1)}:
        raise ValueError(f"control maps do not cover the frozen candidate set for {case_id}/{kind}")
    if sorted(int(v) for v in ids.values()) != list(range(1, len(keys) + 1)):
        raise ValueError(f"visible IDs are not a permutation for {case_id}/{kind}")
    if any(not isinstance(v, int) or not 0 <= v < len(PALETTE) for v in colors.values()):
        raise ValueError(f"palette index outside fixed palette for {case_id}/{kind}")
    if order != [int(x) for x in order] or sorted(order) != list(range(1, len(keys) + 1)):
        raise ValueError(f"draw order is not a candidate permutation for {case_id}/{kind}")
    bindings = {str(row["candidate_id"]): row for row in job["candidate_bindings"]}
    if set(bindings) != keys:
        raise ValueError(f"job candidate bindings differ from render for {case_id}/{kind}")
    for candidate_id in sorted(keys, key=int):
        row = bindings[candidate_id]
        if row.get("visible_id") != ids[candidate_id] or row.get("color_index") != colors[candidate_id]:
            raise ValueError(f"job ID/color binding differs from render for {case_id}/{kind}")


def _render_order_layout(
    images: list[Image.Image],
    times: list[float],
    base_render: Mapping[str, Any],
    control_render: Mapping[str, Any],
    case_id: str,
    kind: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    base_records = base_render.get("records")
    control_records = control_render.get("records")
    if len(base_records) != 4 or len(control_records) != 4 or len(images) != 4 or len(times) != 4:
        raise ValueError(f"four-frame render metadata required for {case_id}/{kind}")
    id_map = {str(k): int(v) for k, v in control_render["id_map"].items()}
    color_map = {str(k): int(v) for k, v in control_render["color_map"].items()}
    draw_order = [int(value) for value in control_render["draw_order"]]
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
    timestamp_font = ImageFont.truetype("DejaVuSans.ttf", 13)
    frames: list[np.ndarray] = []
    output_records: list[dict[str, Any]] = []

    for slot, (image, timestamp, base_row, control_row) in enumerate(
        zip(images, times, base_records, control_records)
    ):
        if int(base_row.get("slot", -1)) != slot or int(control_row.get("slot", -1)) != slot:
            raise ValueError(f"render slots are not in canonical order for {case_id}/{kind}")
        if base_row.get("source_RGB_sha256") != control_row.get("source_RGB_sha256"):
            raise ValueError(f"old controls do not share source RGB for {case_id}/{kind}")
        canvas, letterbox = _letterbox(image)
        if letterbox != base_row.get("letterbox") or letterbox != control_row.get("letterbox"):
            raise ValueError(f"canonical letterbox geometry differs from old render for {case_id}/{kind}")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((4, 4, 130, 26), fill=(24, 26, 30))
        draw.text((8, 7), f"t={timestamp:.3f}s", font=timestamp_font, fill="white")

        base_overlay = {str(row["candidate_id"]): row for row in base_row["overlays"]}
        control_overlay = {str(row["candidate_id"]): row for row in control_row["overlays"]}
        if len(base_overlay) != len(base_row["overlays"]) or len(control_overlay) != len(control_row["overlays"]):
            raise ValueError(f"duplicate candidate overlay in {case_id}/{kind}")
        if set(base_overlay) != set(control_overlay):
            raise ValueError(f"candidate availability differs across controls for {case_id}/{kind}")
        missing = control_row.get("missing_candidate_ids", [])
        if set(map(str, missing)) != set(id_map) - set(control_overlay):
            raise ValueError(f"missing-candidate declaration is inconsistent for {case_id}/{kind}")

        for candidate_id in draw_order:
            key = str(candidate_id)
            if key not in control_overlay:
                continue
            overlay = control_overlay[key]
            box = tuple(int(v) for v in overlay["box_xyxy_pixels"])
            color_index = color_map[key]
            # Verify the old control box remains the same candidate geometry.
            if key not in base_overlay or box != tuple(int(v) for v in base_overlay[key]["box_xyxy_pixels"]):
                raise ValueError(f"control changed candidate box geometry for {case_id}/{kind}")
            draw.rectangle(box, outline=PALETTE[color_index], width=2)

        new_overlays: list[dict[str, Any]] = []
        for candidate_id in draw_order:
            key = str(candidate_id)
            if key not in control_overlay:
                continue
            old_overlay = control_overlay[key]
            base_badge = tuple(int(v) for v in base_overlay[key]["badge_xyxy_pixels"])
            x1, y1, x2, y2 = base_badge
            visible_id = id_map[key]
            color_index = color_map[key]
            draw.rectangle(base_badge, fill=PALETTE[color_index])
            draw.text((x1 + 4, y1 + 2), str(visible_id), font=font, fill="white")
            new_overlays.append(
                {
                    "candidate_id": candidate_id,
                    "visible_id": visible_id,
                    "palette_index": color_index,
                    "box_xyxy_pixels": list(map(int, old_overlay["box_xyxy_pixels"])),
                    "badge_xyxy_pixels": list(base_badge),
                    "badge_position_source": "base_render_same_slot_same_candidate",
                }
            )

        rendered = np.asarray(canvas)
        frames.append(rendered)
        output_records.append(
            {
                "slot": slot,
                "source_RGB_sha256": base_row["source_RGB_sha256"],
                "rendered_RGB_sha256": _sha_bytes(rendered.tobytes()),
                "overlays": new_overlays,
                "missing_candidate_ids": list(missing),
                "letterbox": letterbox,
                "badge_position_source_render_slot": slot,
            }
        )
    tensor = np.stack(frames).transpose(0, 3, 1, 2)
    if tensor.shape != (4, 3, IMAGE_SIDE, IMAGE_SIDE) or tensor.dtype != np.uint8:
        raise ValueError(f"unexpected tensor ABI for {case_id}/{kind}")
    return tensor, output_records


def prepare() -> dict[str, Any]:
    started = time.perf_counter()
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty sub-scope: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    old_barrier = _read_json(OLD_INPUT_BARRIER)
    old_lock = _read_json(OLD_JOB_LOCK)
    if old_barrier.get("status") != "complete" or old_barrier.get("count") != 16 or old_barrier.get("GT_read") is not False:
        raise ValueError("the original 16-case candidate barrier is not a complete no-GT source")
    if old_lock.get("status") != "locked" or old_lock.get("count") != 96 or old_lock.get("GT_read") is not False:
        raise ValueError("the original 96-job lock is not the required no-GT source")
    old_barrier_sha = _sha_file(OLD_INPUT_BARRIER)
    old_job_lock_sha = _sha_file(OLD_JOB_LOCK)
    file_index = _manifest_index(old_barrier)
    records = old_barrier.get("records", [])
    if len(records) != 16 or len({row.get("case_id") for row in records}) != 16:
        raise ValueError("the original candidate barrier must contain 16 unique cases")
    job_rows = old_lock.get("records", [])
    jobs_by_id = {row.get("job_id"): row for row in job_rows}
    if len(jobs_by_id) != 96:
        raise ValueError("original job IDs are not unique")

    new_input_records: list[dict[str, Any]] = []
    new_jobs: list[dict[str, Any]] = []
    output_files: list[dict[str, Any]] = []
    case_request_hashes: dict[str, str] = {}

    for record in records:
        case_id = str(record["case_id"])
        if record.get("GT_read") is not False:
            raise ValueError(f"original candidate record is not no-GT for {case_id}")
        request, video, frame_map, request_path = _safe_video_request(case_id, record)
        frame_ids = [int(value) for value in record["physical_frame_ids"]]
        images, decoded_hashes, source_dimensions = _decode_keyframes(
            case_id, video, frame_map, frame_ids, record["source_RGB_sha256_by_frame"]
        )
        case_request_hashes[case_id] = _sha_file(request_path)

        metadata: dict[str, dict[str, Any]] = {}
        metadata_sha: dict[str, str] = {}
        jobs: dict[str, dict[str, Any]] = {}
        for kind in ("base",) + TARGET_KINDS:
            render_path = f"artifacts/stvg_motion_guided_instance_opd_v1/private_inputs/{case_id}/{kind}_render.json"
            npy_path = f"artifacts/stvg_motion_guided_instance_opd_v1/private_inputs/{case_id}/{kind}.npy"
            _verify_manifest_file(file_index, npy_path)
            render, _, digest = _control_metadata(case_id, kind, file_index)
            metadata[kind] = render
            metadata_sha[kind] = digest
            job_id = f"{case_id}_{kind}"
            if job_id not in jobs_by_id:
                raise ValueError(f"old job missing: {job_id}")
            job = jobs_by_id[job_id]
            if job.get("case_id") != case_id or job.get("kind") != kind or len(job.get("videos", [])) != 1:
                raise ValueError(f"old job metadata mismatch: {job_id}")
            old_video = job["videos"][0]
            _verify_manifest_file(file_index, old_video["path"])
            if _sha_file(_root_path(old_video["path"])) != old_video.get("sha256"):
                raise ValueError(f"old job file hash mismatch: {job_id}")
            if not isinstance(job.get("prompt"), str) or _sha_bytes(job["prompt"].encode()) != job.get("prompt_sha256"):
                raise ValueError(f"old prompt binding mismatch: {job_id}")
            _validate_control_maps(case_id, kind, render, job)
            if old_video.get("actual_times_seconds") != record.get("actual_times_seconds"):
                raise ValueError(f"old job frame times differ from candidate barrier: {job_id}")
            jobs[kind] = job

        base_render = metadata["base"]
        if len(base_render.get("records", [])) != 4:
            raise ValueError(f"base badge layout is not four frames for {case_id}")
        for slot in range(4):
            decoded_sha = decoded_hashes[slot]
            if base_render["records"][slot].get("source_RGB_sha256") != decoded_sha:
                raise ValueError(f"base render source RGB does not match canonical decode for {case_id}")
            for kind in TARGET_KINDS:
                if metadata[kind]["records"][slot].get("source_RGB_sha256") != decoded_sha:
                    raise ValueError(f"control render source RGB does not match canonical decode for {case_id}/{kind}")
        for kind in TARGET_KINDS:
            render = metadata[kind]
            _validate_control_maps(case_id, kind, render, jobs[kind])
            _verify_manifest_file(
                file_index,
                f"artifacts/stvg_motion_guided_instance_opd_v1/private_inputs/{case_id}/{kind}_render.json",
            )
            times = [float(value) for value in record["actual_times_seconds"]]
            tensor, rendered_records = _render_order_layout(
                images, times, base_render, render, case_id, kind
            )
            child_dir = OUT / "private_inputs" / case_id
            child_dir.mkdir(parents=True, exist_ok=True)
            tensor_path = child_dir / f"{kind}.npy"
            render_path = child_dir / f"{kind}_render.json"
            relative_tensor = str(tensor_path.relative_to(ROOT))
            relative_render = str(render_path.relative_to(ROOT))
            np.save(tensor_path, tensor, allow_pickle=False)
            rendered_meta = {
                "GT_read": False,
                "layout_schema": SCHEMA,
                "id_map": metadata[kind]["id_map"],
                "color_map": metadata[kind]["color_map"],
                "draw_order": metadata[kind]["draw_order"],
                "badge_layout_source": {
                    "kind": "base",
                    "path": f"artifacts/stvg_motion_guided_instance_opd_v1/private_inputs/{case_id}/base_render.json",
                    "sha256": metadata_sha["base"],
                },
                "control_layout_source": {
                    "kind": kind,
                    "path": f"artifacts/stvg_motion_guided_instance_opd_v1/private_inputs/{case_id}/{kind}_render.json",
                    "sha256": metadata_sha[kind],
                },
                "records": rendered_records,
            }
            _write_once(render_path, rendered_meta)

            tensor_sha = _sha_bytes(tensor.tobytes())
            new_video = {
                "path": relative_tensor,
                "sha256": _sha_file(tensor_path),
                "tensor_sha256": tensor_sha,
                "fps": jobs[kind]["videos"][0]["fps"],
                "actual_times_seconds": list(jobs[kind]["videos"][0]["actual_times_seconds"]),
            }
            new_job = copy.deepcopy(jobs[kind])
            old_prompt = new_job["prompt"]
            old_prompt_sha = new_job["prompt_sha256"]
            new_job["videos"] = [new_video]
            if new_job["job_id"] != f"{case_id}_{kind}" or new_job["prompt"] != old_prompt or new_job["prompt_sha256"] != old_prompt_sha:
                raise AssertionError("job ID or original prompt changed during copy")
            new_jobs.append(new_job)

            file_rows = []
            for path in (tensor_path, render_path):
                row = {"path": str(path.relative_to(ROOT)), "sha256": _sha_file(path), "bytes": path.stat().st_size}
                output_files.append(row)
                file_rows.append(row)
            new_input_records.append(
                {
                    "case_id": case_id,
                    "dataset": record["dataset"],
                    "cohort": record["cohort"],
                    "source_case_sha256": record["source_case_sha256"],
                    "source_candidate_barrier_sha256": old_barrier_sha,
                    "source_job_lock_sha256": old_job_lock_sha,
                    "canonical_video_sha256": request["canonical_video_sha256"],
                    "source_video_sha256": request["source_video_sha256"],
                    "request_sha256": case_request_hashes[case_id],
                    "physical_frame_ids": frame_ids,
                    "source_RGB_sha256_by_frame": {str(fid): decoded_hashes[i] for i, fid in enumerate(frame_ids)},
                    "source_wh_by_frame": {str(fid): source_dimensions[i] for i, fid in enumerate(frame_ids)},
                    "badge_source_render_sha256": metadata_sha["base"],
                    "control_render_sha256": metadata_sha[kind],
                    "control": kind,
                    "job_id": new_job["job_id"],
                    "candidate_ids": [int(x) for x in metadata[kind]["draw_order"]],
                    "input_tensor_sha256": tensor_sha,
                    "input_file_sha256": new_video["sha256"],
                    "render_metadata_sha256": _sha_file(render_path),
                    "GT_read": False,
                    "model_calls": 0,
                }
            )
            del tensor

    if len(new_jobs) != 32 or len(new_input_records) != 32:
        raise AssertionError("expected exactly 32 new control inputs and jobs")
    if len({row["job_id"] for row in new_jobs}) != 32:
        raise AssertionError("new job IDs are not unique")
    if {row["kind"] for row in new_jobs} != set(TARGET_KINDS):
        raise AssertionError("new jobs include an unexpected condition")

    candidate_barrier = {
        "schema": SCHEMA,
        "status": "complete",
        "count": 32,
        "case_count": 16,
        "job_count": 32,
        "records": new_input_records,
        "files": output_files,
        "source_candidate_barrier_path": str(OLD_INPUT_BARRIER.relative_to(ROOT)),
        "source_candidate_barrier_sha256": old_barrier_sha,
        "source_teacher_job_lock_path": str(OLD_JOB_LOCK.relative_to(ROOT)),
        "source_teacher_job_lock_sha256": old_job_lock_sha,
        "source_candidate_records_sha256": {
            row["case_id"]: _sha_bytes(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            )
            for row in records
        },
        "GT_read": False,
        "model_calls": 0,
        "source_pixels_decoded_and_checked": True,
        "output_roles": ["order_swap", "joint_swap"],
        "created_seconds": time.time(),
        "cpu_construction_seconds": time.perf_counter() - started,
    }
    candidate_path = OUT / "CANDIDATE_INPUT_BARRIER.json"
    _write_once(candidate_path, candidate_barrier)
    candidate_sha = _sha_file(candidate_path)

    job_lock = {
        "schema": SCHEMA,
        "status": "locked",
        "count": 32,
        "records": new_jobs,
        "source_candidate_input_barrier_sha256": candidate_sha,
        "source_candidate_barrier_sha256": old_barrier_sha,
        "source_teacher_job_lock_sha256": old_job_lock_sha,
        "GT_read": False,
        "model_calls": 0,
        "created_seconds": time.time(),
    }
    job_lock_path = OUT / "TEACHER_JOB_LOCK.json"
    _write_once(job_lock_path, job_lock)
    new_job_lock_sha = _sha_file(job_lock_path)

    capture = {
        "schema": SCHEMA,
        "status": "prepared_waiting_for_separate_teacher_calls",
        "scope": "fixed badge positions from base layout for order_swap and joint_swap only",
        "case_count": 16,
        "new_job_count": 32,
        "job_ids_unchanged": True,
        "original_prompts_and_prompt_hashes_unchanged": True,
        "original_id_color_and_box_draw_maps_reused": True,
        "badge_coordinates_from_base_render_metadata": True,
        "canonical_video_reddecode_verified": True,
        "keyframe_RGB_hashes_matched_original_candidate_barrier": True,
        "old_96_inputs_jobs_and_outputs_modified": False,
        "old_96_layout_review": "The old order-swap rendering recomputed badge placement while iterating the permuted draw order. This can couple display order to badge positions. This sub-scope holds each badge at its matching base-render slot/candidate coordinate while retaining each control's own box order and ID/color maps.",
        "teacher_calls_pending": True,
        "GT_read": False,
        "model_calls": 0,
        "source_candidate_barrier_path": str(OLD_INPUT_BARRIER.relative_to(ROOT)),
        "source_candidate_barrier_sha256": old_barrier_sha,
        "source_teacher_job_lock_path": str(OLD_JOB_LOCK.relative_to(ROOT)),
        "source_teacher_job_lock_sha256": old_job_lock_sha,
        "candidate_input_barrier_path": str(candidate_path.relative_to(ROOT)),
        "candidate_input_barrier_sha256": candidate_sha,
        "teacher_job_lock_path": str(job_lock_path.relative_to(ROOT)),
        "teacher_job_lock_sha256": new_job_lock_sha,
        "source_helper_path": str(Path(__file__).relative_to(ROOT)),
        "source_helper_sha256": _sha_file(Path(__file__)),
        "source_render_helpers_sha256": _sha_file(ROOT / "scripts/prepare_stvg_scene_object_id_v1.py"),
        "per_case_request_sha256": case_request_hashes,
        "created_seconds": time.time(),
    }
    _write_once(OUT / "CAPTURE.json", capture)
    return capture


if __name__ == "__main__":
    result = prepare()
    print(json.dumps({key: value for key, value in result.items() if key not in {"per_case_request_sha256"}}, indent=2))
