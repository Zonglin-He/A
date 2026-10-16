#!/usr/bin/env python3
"""GT-free, native STVG-R1 teacher adapter.

Importing this module is CPU-only and does not import model/runtime packages.
Actual detector, SAM2, video decoding and Qwen execution happen only when a
caller explicitly invokes the corresponding functions in an isolated worker.
This adapter never loads checkpoints or reads dataset annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "stvg-r1-native-teacher-v1"
STVG_R1_REPO = "https://github.com/XiaowZhang00/stvg-r1"
STVG_R1_REVISION = "0bea9e57d5e26e3581d9f6f94f487352f71b08c9"
STVG_R1_MODEL_REPO = "XiaowenZhang/stvg-r1-model-7b"
STVG_R1_MODEL_REVISION = "c0db00d019bf746f54b67770b82261b2161fed59"
GROUND_SAM2_REPO = "https://github.com/IDEA-Research/Grounded-SAM-2"
GROUND_SAM2_REVISION = "b7a9c29f196edff0eb54dbe14588d7ae5e3dde28"
DETECTOR_ASSET_RELATIVE = "checkpoints/STVG-R1-tracking/yolov12x.pt"
SAM2_ASSET_RELATIVE = "checkpoints/STVG-R1-tracking/sam2.1_hiera_large.pt"
SAM2_CONFIG_RELATIVE = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
TARGET_MODE = "all"
AREA_FILTER_RATIO = 1.0 / 3.0
STVG_R1_FILE_SHA256 = {
    "preprocess/dettrack.py": "d973b65922acab035bbaecf7595e9903ab639868a943f80cc5723c8675aca508",
    "preprocess/mask2marker.py": "323a247b11f44c2146fb438c39e7adb5e5965ec8c80f6c4b8afa05f774f2fd05",
}
GROUND_SAM2_FILE_SHA256 = {
    "sam2/__init__.py": "b87ca1e95cd54b81766e8c74acf0e937952639529e40d2b4088693286f49419e",
    "sam2/build_sam.py": "bc49ac8e9ebf871790fa2e5f0e70bd5734e010966eff66af0c350ceaf14f3f1e",
    "sam2/configs/sam2.1/sam2.1_hiera_l.yaml": "545e4325aa5c19a1615d43c946b07276ed4c57214eacf1437e38fa3d9374f636",
    "sam2/utils/misc.py": "a5295920c904e5673c94e42d43cf4ef44ae69f2a727e742f36b777297f36e9a4",
    "sam2/utils/transforms.py": "ba3a64f4600c62f209206a6df3b40e3fcf133edae32fad658831bb0c2a6d1146",
    "utils/mask_dictionary_model.py": "1614811da46406903d6016fa7a93570da943d922e2d2affc5fb7995f63adb4a5",
}

GROUND_TEMPLATE = """Each object in the video is marked with a red number at its center, representing its object ID. To accurately pinpoint the event \"[EVENT]\" in the video:

1.Determine the precise time period of the event occurs.

2.Identify which object ID is corresponding to the described event.

Output your thought process within the <think> </think> tags, including both a temporal analysis (e.g., specific timestamps \"xx.xx\" or time ranges \"xx.xx to xx.xx\") and a spatial analysis (e.g., object ID \"xx\").

Then, provide the start and end times (in seconds, precise to two decimal places), and object ID in the format \"Target ID: [ID], Time range: [start time to end time]\" within the <answer> </answer> tags. For example: \"Target ID: 1, Time range: 12.54 to 17.83\"."""

_REQUEST_KEYS = {
    "video_path",
    "query_text",
    "source_video_sha256",
    "canonical_video_sha256",
    "canonical_pixel_sha256",
    "local_fps",
    "local_frame_count",
    "frame_map",
}
_FRAME_MAP_KEYS = {"local_frame_index", "physical_frame_id", "source_timestamp_s"}
_FORBIDDEN_KEYS = {
    "annotation",
    "annotations",
    "bbox",
    "bboxes",
    "gt",
    "gt_id",
    "gt_bbox",
    "gt_bboxes",
    "gt_interval",
    "gt_timestamps",
    "objectid",
    "target_id",
    "target_ids",
    "timestamps",
}
_ID_RE = re.compile(r"Target ID:\s*(-?\d+)", re.IGNORECASE)
_TIME_RE = re.compile(
    r"Time range:\s*(-?\d+(?:\.\d*)?)\s*(?:to|and)\s*(-?\d+(?:\.\d*)?)",
    re.IGNORECASE,
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_sam2_hydra_config(config_name: str) -> str:
    """Fix the pinned STVG-R1 path prefix for SAM2's package-root Hydra search.

    dettrack.py computes ``sam2/configs/...`` relative to the repository root,
    while the pinned SAM2 package registers ``sam2`` itself as Hydra's config
    root. Only that exact official path is normalized; other values pass through.
    """
    if config_name == SAM2_CONFIG_RELATIVE:
        return config_name.removeprefix("sam2/")
    return config_name


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a generated/cache artifact when explicitly requested by a caller."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a strict inference-only request without opening its video path."""
    if not isinstance(payload, Mapping):
        raise TypeError("request must be a mapping")
    keys = set(payload)
    forbidden = sorted(keys & _FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError(f"GT/annotation fields are forbidden: {forbidden}")
    if keys != _REQUEST_KEYS:
        missing = sorted(_REQUEST_KEYS - keys)
        extra = sorted(keys - _REQUEST_KEYS)
        raise ValueError(f"request keys mismatch; missing={missing}, extra={extra}")

    video_path = payload["video_path"]
    query_text = payload["query_text"]
    source_video_hash = payload["source_video_sha256"]
    canonical_video_hash = payload["canonical_video_sha256"]
    canonical_pixel_hash = payload["canonical_pixel_sha256"]
    if not isinstance(video_path, str) or not video_path.strip():
        raise ValueError("video_path must be a non-empty path string")
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be a non-empty string")
    for key, digest in (
        ("source_video_sha256", source_video_hash),
        ("canonical_video_sha256", canonical_video_hash),
        ("canonical_pixel_sha256", canonical_pixel_hash),
    ):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{key} must be a caller-supplied lowercase SHA256")

    fps = payload["local_fps"]
    frame_count = payload["local_frame_count"]
    if isinstance(fps, bool) or not isinstance(fps, (float, int)) or not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("local_fps must be finite and positive")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("local_frame_count must be a positive integer")

    raw_map = payload["frame_map"]
    if not isinstance(raw_map, Sequence) or isinstance(raw_map, (str, bytes)) or len(raw_map) != frame_count:
        raise ValueError("frame_map must have one entry for every local frame")
    frame_map: list[dict[str, Any]] = []
    last_physical_id = -1
    last_timestamp = -math.inf
    for expected_index, row in enumerate(raw_map):
        if not isinstance(row, Mapping) or set(row) != _FRAME_MAP_KEYS:
            raise ValueError(f"frame_map[{expected_index}] must contain exactly {_FRAME_MAP_KEYS}")
        local_index = row["local_frame_index"]
        physical_id = row["physical_frame_id"]
        timestamp = row["source_timestamp_s"]
        if isinstance(local_index, bool) or not isinstance(local_index, int) or local_index != expected_index:
            raise ValueError("frame_map local_frame_index values must be dense and zero-based")
        if isinstance(physical_id, bool) or not isinstance(physical_id, int) or physical_id < 0:
            raise ValueError("physical_frame_id must be a non-negative integer")
        if physical_id <= last_physical_id:
            raise ValueError("physical_frame_id values must be strictly increasing")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (float, int)) or not math.isfinite(float(timestamp)):
            raise ValueError("source_timestamp_s must be finite")
        timestamp = float(timestamp)
        if timestamp < 0 or timestamp <= last_timestamp:
            raise ValueError("source_timestamp_s values must be non-negative and strictly increasing")
        frame_map.append({
            "local_frame_index": local_index,
            "physical_frame_id": physical_id,
            "source_timestamp_s": timestamp,
        })
        last_physical_id = physical_id
        last_timestamp = timestamp

    normalized = {
        "video_path": video_path,
        "query_text": query_text,
        "source_video_sha256": source_video_hash,
        "canonical_video_sha256": canonical_video_hash,
        "canonical_pixel_sha256": canonical_pixel_hash,
        "local_fps": float(fps),
        "local_frame_count": frame_count,
        "frame_map": frame_map,
    }
    normalized["query_sha256"] = _sha256_bytes(query_text.encode("utf-8"))
    normalized["frame_clock_sha256"] = _sha256_bytes(_canonical_json({
        "local_fps": normalized["local_fps"],
        "local_frame_count": frame_count,
        "frame_map": frame_map,
    }))
    return normalized


def _normalize_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Accept raw input or normalized input whose derived hashes are rechecked."""
    if set(request) == _REQUEST_KEYS:
        return validate_request(request)
    normalized_keys = _REQUEST_KEYS | {"query_sha256", "frame_clock_sha256"}
    if set(request) != normalized_keys:
        return validate_request(request)
    raw = {key: request[key] for key in _REQUEST_KEYS}
    normalized = validate_request(raw)
    if request["query_sha256"] != normalized["query_sha256"]:
        raise ValueError("query_sha256 does not bind the supplied query_text")
    if request["frame_clock_sha256"] != normalized["frame_clock_sha256"]:
        raise ValueError("frame_clock_sha256 does not bind the supplied frame map")
    return normalized


def build_native_messages(query_text: str, marker_video_path: str) -> list[dict[str, Any]]:
    """Build the official STVG-R1 single-user prompt/video message."""
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be non-empty")
    if not isinstance(marker_video_path, str) or not marker_video_path.strip():
        raise ValueError("marker_video_path must be non-empty")
    prompt = GROUND_TEMPLATE.replace("[EVENT]", query_text)
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "video": marker_video_path,
                "total_pixels": 2048 * 28 * 28,
                "min_pixels": 16 * 28 * 28,
            },
        ],
    }]


def project_interval(start_time_s: float, end_time_s: float, request: Mapping[str, Any]) -> dict[str, Any]:
    """Map official int(seconds * fps) local indices to caller-supplied source IDs."""
    req = _normalize_request(request)
    if not (math.isfinite(start_time_s) and math.isfinite(end_time_s)):
        raise ValueError("times must be finite")
    if start_time_s < 0 or end_time_s < start_time_s:
        raise ValueError("time interval must be non-negative and ordered")
    start_local = int(start_time_s * req["local_fps"])
    end_local = int(end_time_s * req["local_fps"])
    if start_local < 0 or end_local >= req["local_frame_count"]:
        raise IndexError("time interval maps outside the decoded local frame clock")
    local_indices = list(range(start_local, end_local + 1))
    rows = req["frame_map"]
    return {
        "start_local_frame": start_local,
        "end_local_frame": end_local,
        "local_frame_indices": local_indices,
        "physical_frame_ids": [rows[i]["physical_frame_id"] for i in local_indices],
    }


def parse_native_output(
    raw_text: str,
    *,
    duration_s: float,
    known_ids: set[int] | Sequence[int],
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Parse the official answer format and validate without clipping/fallback."""
    req = _normalize_request(request)
    parsed = {
        "status": "invalid_format",
        "candidate_id": None,
        "start_time_s": None,
        "end_time_s": None,
    }
    if not isinstance(raw_text, str):
        return parsed, None
    answer_blocks = re.findall(r"<answer>(.*?)</answer>", raw_text, re.DOTALL | re.IGNORECASE)
    if not answer_blocks:
        return parsed, None
    answer = answer_blocks[-1].strip()
    ids = _ID_RE.findall(answer)
    ranges = _TIME_RE.findall(answer)
    if len(ids) != 1 or len(ranges) != 1:
        return parsed, None
    candidate_id = int(ids[0])
    start_s, end_s = map(float, ranges[0])
    parsed.update({
        "candidate_id": candidate_id,
        "start_time_s": start_s,
        "end_time_s": end_s,
    })
    if candidate_id <= 0 or start_s < 0 or end_s < 0 or end_s < start_s:
        parsed["status"] = "invalid_time_or_id"
        return parsed, None
    if candidate_id not in set(known_ids):
        parsed["status"] = "unknown_candidate_id"
        return parsed, None
    if not math.isfinite(duration_s) or duration_s <= 0:
        parsed["status"] = "invalid_duration"
        return parsed, None
    try:
        projected = project_interval(start_s, end_s, req)
    except (ValueError, IndexError):
        parsed["status"] = "out_of_clock_interval"
        return parsed, None
    if end_s > duration_s:
        parsed["status"] = "out_of_duration"
        return parsed, None
    parsed["status"] = "valid"
    return parsed, projected


def bbox_from_mask_inclusive(mask: Any) -> tuple[list[int] | None, int]:
    """Return [xmin,ymin,xmax,ymax] inclusive pixels and area from a 2-D mask."""
    import numpy as np

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("decoded RLE mask must be 2-D")
    ys, xs = np.nonzero(array)
    if len(xs) == 0:
        return None, 0
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], int(len(xs))


def _load_mask_inventory(mask_data_dir: str | os.PathLike[str]) -> tuple[dict[int, Path], set[int]]:
    """Index only generated, filtered mask_####.json files; never annotation files."""
    directory = Path(mask_data_dir)
    if directory.name != "mask_data_filtered_1_3":
        raise ValueError("mask_data_dir must be the generated mask_data_filtered_1_3 directory")
    files: dict[int, Path] = {}
    known_ids: set[int] = set()
    for path in directory.iterdir():
        match = re.fullmatch(r"mask_(\d+)\.json", path.name)
        if not path.is_file() or not match:
            continue
        files[int(match.group(1))] = path
    for path in files.values():
        with path.open("r", encoding="utf-8") as stream:
            row = json.load(stream)
        objects = row.get("objects")
        if not isinstance(objects, Mapping):
            raise ValueError(f"generated mask file has no objects mapping: {path.name}")
        for key in objects:
            try:
                parsed_id = int(key)
            except (TypeError, ValueError):
                continue
            if parsed_id > 0:
                known_ids.add(parsed_id)
    return files, known_ids


def build_teacher_record(
    request: Mapping[str, Any],
    raw_output: str,
    *,
    mask_data_dir: str | os.PathLike[str],
    marker_video_sha256: str,
    costs: Mapping[str, float],
    tracker_precision: Mapping[str, Any] | None = None,
    preprocess_warnings: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """Build the frozen raw teacher row from generated masks and one native answer."""
    req = _normalize_request(request)
    if not re.fullmatch(r"[0-9a-f]{64}", marker_video_sha256):
        raise ValueError("marker_video_sha256 must be lowercase SHA256")
    expected_cost_keys = {
        "dettrack_s", "filter_marker_s", "model_preprocess_s", "model_generate_s", "total_s"
    }
    if set(costs) != expected_cost_keys:
        raise ValueError(f"costs must contain exactly {sorted(expected_cost_keys)}")
    normalized_costs = {}
    for key, value in costs.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{key} must be finite and non-negative")
        normalized_costs[key] = float(value)

    if tracker_precision is None:
        normalized_tracker_precision = {
            "autocast": "torch.autocast(device_type='cuda', dtype=torch.bfloat16)",
            "runtime_values": "not_recorded",
            "tf32_rule": "allow_tf32_matmul and allow_tf32_cudnn are enabled when CUDA compute capability major >= 8, matching official main()",
        }
    else:
        expected_tracker_precision_keys = {
            "autocast", "cuda_compute_capability_major", "allow_tf32_matmul", "allow_tf32_cudnn"
        }
        if set(tracker_precision) != expected_tracker_precision_keys:
            raise ValueError(f"tracker_precision must contain exactly {sorted(expected_tracker_precision_keys)}")
        major = tracker_precision["cuda_compute_capability_major"]
        if isinstance(major, bool) or not isinstance(major, int) or major < 0:
            raise ValueError("tracker CUDA compute capability major must be a non-negative integer")
        if not isinstance(tracker_precision["autocast"], str) or not tracker_precision["autocast"]:
            raise ValueError("tracker autocast description must be a non-empty string")
        if not isinstance(tracker_precision["allow_tf32_matmul"], bool) or not isinstance(tracker_precision["allow_tf32_cudnn"], bool):
            raise ValueError("tracker TF32 runtime flags must be booleans")
        normalized_tracker_precision = dict(tracker_precision)

    normalized_warnings: list[dict[str, str]] = []
    for warning_row in preprocess_warnings:
        if not isinstance(warning_row, Mapping) or set(warning_row) != {"category", "message"}:
            raise ValueError("each preprocessing warning must contain exactly category and message")
        if not all(isinstance(warning_row[key], str) for key in ("category", "message")):
            raise ValueError("preprocessing warning category and message must be strings")
        normalized_warnings.append({
            "category": warning_row["category"],
            "message": warning_row["message"],
        })

    mask_files, known_ids = _load_mask_inventory(mask_data_dir)
    duration_s = req["local_frame_count"] / req["local_fps"]
    parsed, projected = parse_native_output(
        raw_output,
        duration_s=duration_s,
        known_ids=known_ids,
        request=req,
    )
    tube_frames: list[dict[str, Any]] = []
    if projected is not None:
        from pycocotools import mask as mask_utils

        candidate_id = parsed["candidate_id"]
        interval_indices = set(projected["local_frame_indices"])
        for local_index, frame_map_row in enumerate(req["frame_map"]):
            row_path = mask_files.get(local_index)
            object_row = None
            if row_path is not None:
                with row_path.open("r", encoding="utf-8") as stream:
                    frame_data = json.load(stream)
                object_row = frame_data.get("objects", {}).get(str(candidate_id))
            box = None
            area = 0
            if object_row is not None:
                box, area = bbox_from_mask_inclusive(mask_utils.decode(object_row["rle"]))
            tube_frames.append({
                "local_frame_index": local_index,
                "physical_frame_id": frame_map_row["physical_frame_id"],
                "source_timestamp_s": frame_map_row["source_timestamp_s"],
                "candidate_id": candidate_id,
                "present": box is not None,
                "xyxy_inclusive_pixels": box,
                "mask_area_px": area,
            })
        interval_rows = [row for row in tube_frames if row["local_frame_index"] in interval_indices]
        present_count = sum(1 for row in interval_rows if row["present"])
        interval_count = len(interval_rows)
        if interval_count and present_count == 0:
            parsed["status"] = "candidate_missing_in_interval"
    else:
        present_count = 0
        interval_count = 0

    return {
        "schema": SCHEMA,
        "source": {
            "official_code_repo": STVG_R1_REPO,
            "official_code_revision": STVG_R1_REVISION,
            "model_repo": STVG_R1_MODEL_REPO,
            "model_revision": STVG_R1_MODEL_REVISION,
            "tracker_repo": GROUND_SAM2_REPO,
            "tracker_revision": GROUND_SAM2_REVISION,
            "detector": "YOLOv12-X",
            "detector_asset": DETECTOR_ASSET_RELATIVE,
            "segmenter": "SAM2.1 Hiera Large",
            "segmenter_asset": SAM2_ASSET_RELATIVE,
            "segmenter_config": SAM2_CONFIG_RELATIVE,
            "tracker_precision": normalized_tracker_precision,
            "preprocess_warnings": normalized_warnings,
            "target_mode": TARGET_MODE,
            "mask_area_filter_ratio": AREA_FILTER_RATIO,
        },
        "input_binding": {
            "source_video_sha256": req["source_video_sha256"],
            "canonical_video_sha256": req["canonical_video_sha256"],
            "canonical_pixel_sha256": req["canonical_pixel_sha256"],
            "marker_video_sha256": marker_video_sha256,
            "query_text": req["query_text"],
            "query_sha256": req["query_sha256"],
            "frame_clock_sha256": req["frame_clock_sha256"],
            "local_fps": req["local_fps"],
            "local_frame_count": req["local_frame_count"],
            "frame_map": req["frame_map"],
        },
        "raw_output": raw_output,
        "parsed": parsed,
        "projected_interval": projected or {
            "start_local_frame": None,
            "end_local_frame": None,
            "local_frame_indices": [],
            "physical_frame_ids": [],
        },
        "mask_tube": {
            "coordinate_convention": "xyxy_inclusive_pixels",
            "frames": tube_frames,
            "coverage": {
                "present_frame_count": present_count,
                "interval_frame_count": interval_count,
                "fraction": (present_count / interval_count) if interval_count else 0.0,
            },
        },
        "costs": normalized_costs,
    }


def run_native_mllm(
    request: Mapping[str, Any],
    marker_video_path: str,
    *,
    model: Any,
    processor: Any,
    vision_processor: Callable[..., Any],
    device: str,
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    """Run the official Qwen inference route using caller-injected loaded objects."""
    req = _normalize_request(request)
    import torch

    messages = build_native_messages(req["query_text"], marker_video_path)
    preprocess_start = time.perf_counter()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = vision_processor(messages, return_video_kwargs=True)
    fps_inputs = video_kwargs["fps"]
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=fps_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    model_preprocess_s = time.perf_counter() - preprocess_start
    generation_start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    model_generate_s = time.perf_counter() - generation_start
    generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    if len(output_text) != 1:
        raise RuntimeError("native STVG-R1 adapter expects exactly one query per call")
    return {
        "raw_output": output_text[0],
        "model_preprocess_s": model_preprocess_s,
        "model_generate_s": model_generate_s,
        "total_s": model_preprocess_s + model_generate_s,
    }


def make_official_preprocess_runner(
    official_repo: str | os.PathLike[str],
    grounded_sam2_repo: str | os.PathLike[str],
    *,
    detector_asset: str | os.PathLike[str],
    sam2_asset: str | os.PathLike[str],
) -> Callable[..., dict[str, Any]]:
    """Return a one-video, annotation-free adapter over the pinned official code.

    The returned callable imports detector/SAM2 dependencies and allocates GPU
    models only when explicitly invoked. Execute it in the root-controlled,
    isolated, serial teacher worker. It calls process_video() directly (never
    dettrack.main(), trainjson.py, rundettrack.sh or evaluate.py).
    """
    repo = Path(official_repo).resolve()
    gs2 = Path(grounded_sam2_repo).resolve()
    yolo_path = Path(detector_asset).resolve()
    sam_path = Path(sam2_asset).resolve()

    def _run(
        request: Mapping[str, Any],
        *,
        target_mode: str,
        output_dir: str | os.PathLike[str],
    ) -> dict[str, Any]:
        req = _normalize_request(request)
        if target_mode != TARGET_MODE:
            raise ValueError("native STVG-R1 smoke uses target_mode='all'")
        current = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if current != STVG_R1_REVISION:
            raise RuntimeError(f"official STVG-R1 source pin mismatch: {current}")
        grounded_head = subprocess.run(
            ["git", "-C", str(gs2), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if grounded_head != GROUND_SAM2_REVISION:
            raise RuntimeError(f"Grounded-SAM-2 source pin mismatch: {grounded_head}")
        for relative, expected_sha256 in STVG_R1_FILE_SHA256.items():
            if sha256_file(repo / relative) != expected_sha256:
                raise RuntimeError(f"official STVG-R1 file fingerprint mismatch: {relative}")
        for relative, expected_sha256 in GROUND_SAM2_FILE_SHA256.items():
            if sha256_file(gs2 / relative) != expected_sha256:
                raise RuntimeError(f"official Grounded-SAM-2 file fingerprint mismatch: {relative}")
        expected_gs2 = (repo.parent / "Grounded-SAM-2-main").resolve()
        if gs2 != expected_gs2:
            raise RuntimeError("official dettrack.py resolves Grounded-SAM-2-main as a sibling of the STVG-R1 checkout")
        if not yolo_path.is_file() or yolo_path.stat().st_size != 119_322_638:
            raise RuntimeError("pinned local YOLOv12-X file is missing or has an unexpected byte size")
        if not sam_path.is_file() or sam_path.stat().st_size != 898_083_611:
            raise RuntimeError("pinned local SAM2.1 Hiera Large file is missing or has an unexpected byte size")
        model_cfg = gs2 / SAM2_CONFIG_RELATIVE
        if not model_cfg.is_file():
            raise RuntimeError(f"official SAM2 config is missing: {model_cfg}")
        input_path = Path(req["video_path"]).resolve()
        if not input_path.is_file():
            raise FileNotFoundError("requested GT-free video path does not exist")
        root = Path(output_dir).resolve()
        if root.exists():
            raise FileExistsError("output_dir must be new; existing outputs are never overwritten")
        root.mkdir(parents=True)
        det_root = root / "dettrack"
        marker_root = root / "marker_video"
        det_root.mkdir()
        marker_root.mkdir()
        video_name = input_path.name
        video_stem = input_path.stem

        import ultralytics

        previous_yolo = ultralytics.YOLO
        old_sys_path = list(sys.path)
        old_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "utils" or name.startswith("utils.") or name == "sam2" or name.startswith("sam2.")
        }
        module_name = "_stvg_r1_native_dettrack_v1"
        marker_module_name = "_stvg_r1_native_mask2marker_v1"

        def local_yolo(model_path: Any, *args: Any, **kwargs: Any) -> Any:
            if os.fspath(model_path) != "/data/yolov12/yolov12x.pt":
                raise RuntimeError(f"unexpected YOLO asset requested by pinned source: {model_path}")
            return previous_yolo(str(yolo_path), *args, **kwargs)

        det_module = None
        marker_module = None
        dettrack_s = 0.0
        marker_s = 0.0
        try:
            ultralytics.YOLO = local_yolo
            sys.path.insert(0, str(repo / "preprocess"))
            sys.path.insert(0, str(gs2))
            det_path = repo / "preprocess" / "dettrack.py"
            spec = importlib.util.spec_from_file_location(module_name, det_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"could not load official STVG-R1 source: {det_path}")
            det_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = det_module
            spec.loader.exec_module(det_module)
            original_build_sam2 = det_module.build_sam2
            original_build_video_predictor = det_module.build_sam2_video_predictor

            def normalize_builder_config(builder: Callable[..., Any]) -> Callable[..., Any]:
                def _build(config_file: str, *args: Any, **kwargs: Any) -> Any:
                    return builder(normalize_sam2_hydra_config(config_file), *args, **kwargs)

                return _build

            # Preserve official model construction and checkpoint loading while
            # correcting only its repo-relative prefix for the package-root
            # Hydra search configured by sam2/__init__.py.
            det_module.build_sam2 = normalize_builder_config(original_build_sam2)
            det_module.build_sam2_video_predictor = normalize_builder_config(original_build_video_predictor)
            # Avoid dataset enumeration in main(): this direct single-video call
            # gets an empty metadata mapping and resolves only the whitelisted path.
            det_module.find_video_path = lambda name, _base: str(input_path) if name == video_name else None
            config = {
                "sam2_checkpoint": str(sam_path),
                "model_cfg": str(model_cfg),
                "grounding_model_id": "IDEA-Research/grounding-dino-tiny",  # unused by process_video
                "text_prompt": None,
                "target_mode": TARGET_MODE,
                "video_base_dir": str(input_path.parent),
                "base_output_dir": str(det_root),
            }
            # Official dettrack.main() enters this autocast context before its
            # video loop. Replicate that precision path around the direct,
            # annotation-free process_video() call.
            torch = det_module.torch
            compute_major = int(torch.cuda.get_device_properties(0).major)
            if compute_major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            tracker_precision = {
                "autocast": "torch.autocast(device_type='cuda', dtype=torch.bfloat16)",
                "cuda_compute_capability_major": compute_major,
                "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            }
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ok = det_module.process_video(video_name, {}, config)
            dettrack_s = time.perf_counter() - start
            native_warnings = [
                {"category": type(item.message).__name__, "message": str(item.message)}
                for item in caught_warnings
            ]
            if not ok:
                raise RuntimeError("official direct process_video() returned failure")

            marker_path = repo / "preprocess" / "mask2marker.py"
            marker_spec = importlib.util.spec_from_file_location(marker_module_name, marker_path)
            if marker_spec is None or marker_spec.loader is None:
                raise ImportError(f"could not load official marker source: {marker_path}")
            marker_module = importlib.util.module_from_spec(marker_spec)
            sys.modules[marker_module_name] = marker_module
            marker_spec.loader.exec_module(marker_module)
            marker_module.get_video_path = lambda _name, _root: str(input_path)
            start = time.perf_counter()
            filter_result = marker_module.filter_single_video((video_stem, str(det_root), AREA_FILTER_RATIO))
            if filter_result.startswith(("Error", "Skip")):
                raise RuntimeError(f"official mask area filter failed: {filter_result}")
            marker_result = marker_module.process_single_video(
                (video_stem, str(det_root), str(input_path.parent), str(marker_root))
            )
            marker_s = time.perf_counter() - start
            if marker_result.startswith(("Error", "Skip")):
                raise RuntimeError(f"official marker rendering failed: {marker_result}")
            marker_video = marker_root / f"{video_stem}.mp4"
            mask_dir = det_root / video_stem / "mask_data_filtered_1_3"
            if not marker_video.is_file() or not mask_dir.is_dir():
                raise RuntimeError("official preprocessing did not produce marker video and filtered RLE masks")
            return {
                "marker_video_path": str(marker_video),
                "marker_video_sha256": sha256_file(marker_video),
                "mask_data_dir": str(mask_dir),
                "dettrack_s": dettrack_s,
                "filter_marker_s": marker_s,
                "target_mode": TARGET_MODE,
                "metadata_input": "empty_mapping",
                "tracker_precision": tracker_precision,
                "preprocess_warnings": native_warnings,
            }
        finally:
            ultralytics.YOLO = previous_yolo
            sys.path[:] = old_sys_path
            for name in list(sys.modules):
                if name == "utils" or name.startswith("utils.") or name == "sam2" or name.startswith("sam2."):
                    if name not in old_modules:
                        del sys.modules[name]
            sys.modules.update(old_modules)
            sys.modules.pop(module_name, None)
            sys.modules.pop(marker_module_name, None)

    return _run


def run_native_teacher(
    request: Mapping[str, Any],
    *,
    preprocess_runner: Callable[..., Mapping[str, Any]],
    output_dir: str | os.PathLike[str],
    model: Any,
    processor: Any,
    vision_processor: Callable[..., Any],
    device: str,
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    """Convenience single-query path with caller-loaded Qwen objects.

    This helper runs preprocessing while the injected Qwen objects are already
    resident. For phase-separated teacher execution, call the preprocessing
    runner first, release its YOLO/SAM2 objects, load Qwen once, then call
    run_native_mllm() and build_teacher_record() separately.
    """
    total_start = time.perf_counter()
    req = _normalize_request(request)
    prep = dict(preprocess_runner(req, target_mode=TARGET_MODE, output_dir=output_dir))
    required_prep = {"marker_video_path", "marker_video_sha256", "mask_data_dir", "dettrack_s", "filter_marker_s"}
    if not required_prep <= set(prep):
        raise ValueError(f"preprocess_runner omitted required fields: {sorted(required_prep - set(prep))}")
    if prep.get("target_mode") != TARGET_MODE or prep.get("metadata_input") != "empty_mapping":
        raise ValueError("preprocess runner must certify target_mode='all' and empty metadata input")
    mllm = run_native_mllm(
        req,
        str(prep["marker_video_path"]),
        model=model,
        processor=processor,
        vision_processor=vision_processor,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    costs = {
        "dettrack_s": float(prep["dettrack_s"]),
        "filter_marker_s": float(prep["filter_marker_s"]),
        "model_preprocess_s": float(mllm["model_preprocess_s"]),
        "model_generate_s": float(mllm["model_generate_s"]),
        "total_s": time.perf_counter() - total_start,
    }
    record = build_teacher_record(
        req,
        str(mllm["raw_output"]),
        mask_data_dir=str(prep["mask_data_dir"]),
        marker_video_sha256=str(prep["marker_video_sha256"]),
        costs=costs,
        tracker_precision=prep.get("tracker_precision"),
        preprocess_warnings=prep.get("preprocess_warnings", ()),
    )
    record["costs"]["total_s"] = time.perf_counter() - total_start
    return record


def _synthetic_request() -> dict[str, Any]:
    return {
        "video_path": "/synthetic/no-file.mp4",
        "query_text": "synthetic test: the person waves",
        "source_video_sha256": "1" * 64,
        "canonical_video_sha256": "0" * 64,
        "canonical_pixel_sha256": "2" * 64,
        "local_fps": 10.0,
        "local_frame_count": 4,
        "frame_map": [
            {"local_frame_index": i, "physical_frame_id": 100 + i * 2, "source_timestamp_s": 5.0 + i * 0.2}
            for i in range(4)
        ],
    }


def run_cpu_contracts() -> dict[str, Any]:
    """Synthetic-only checks. They do not open paths or import model packages."""
    forbidden_modules = ("torch", "transformers", "cv2", "pycocotools", "ultralytics", "sam2")
    already_loaded = [name for name in sys.modules if name in forbidden_modules or any(name.startswith(m + ".") for m in forbidden_modules)]
    if already_loaded:
        raise RuntimeError(f"CPU contract process must be fresh; model/runtime modules already loaded: {already_loaded}")

    request = _synthetic_request()
    normalized = validate_request(request)
    assert normalized["query_sha256"] == _sha256_bytes(request["query_text"].encode("utf-8"))
    assert normalized["source_video_sha256"] == request["source_video_sha256"]
    assert normalized["canonical_video_sha256"] == request["canonical_video_sha256"]
    assert normalized["canonical_pixel_sha256"] == request["canonical_pixel_sha256"]
    assert len(normalized["frame_clock_sha256"]) == 64
    assert normalize_sam2_hydra_config(SAM2_CONFIG_RELATIVE) == "configs/sam2.1/sam2.1_hiera_l.yaml"
    assert normalize_sam2_hydra_config("other/config.yaml") == "other/config.yaml"
    tampered = dict(normalized, query_sha256="f" * 64)
    try:
        _normalize_request(tampered)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered query binding was accepted")

    message = build_native_messages(request["query_text"], "/synthetic/marker.mp4")[0]
    assert message["content"][0]["text"].endswith('Target ID: 1, Time range: 12.54 to 17.83\".')
    assert request["query_text"] in message["content"][0]["text"]
    assert message["content"][1]["total_pixels"] == 2048 * 28 * 28

    parsed, projected = parse_native_output(
        "<think>synthetic</think><answer>Target ID: 7, Time range: 0.1 to 0.29</answer>",
        duration_s=0.4,
        known_ids={7},
        request=normalized,
    )
    assert parsed["status"] == "valid"
    assert projected == {
        "start_local_frame": 1,
        "end_local_frame": 2,
        "local_frame_indices": [1, 2],
        "physical_frame_ids": [102, 104],
    }

    bad_parse, bad_projection = parse_native_output(
        "<answer>Target ID: 77, Time range: 0.1 to 0.2</answer>",
        duration_s=0.4,
        known_ids={7},
        request=normalized,
    )
    assert bad_parse["status"] == "unknown_candidate_id" and bad_projection is None
    out_parse, out_projection = parse_native_output(
        "<answer>Target ID: 7, Time range: 0.1 to 0.5</answer>",
        duration_s=0.4,
        known_ids={7},
        request=normalized,
    )
    assert out_parse["status"] == "out_of_clock_interval" and out_projection is None
    multi_parse, multi_projection = parse_native_output(
        "<answer>Target ID: 7, Time range: 0.1 to 0.2; Target ID: 8</answer>",
        duration_s=0.4,
        known_ids={7, 8},
        request=normalized,
    )
    assert multi_parse["status"] == "invalid_format" and multi_projection is None

    mask_box, mask_area = bbox_from_mask_inclusive(
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
    )
    assert mask_box == [1, 1, 2, 2] and mask_area == 3
    empty_box, empty_area = bbox_from_mask_inclusive([[0, 0], [0, 0]])
    assert empty_box is None and empty_area == 0

    # Exercise dense full-video mask rows and interval-only coverage with a
    # synthetic pycocotools decoder stub. No real masks or video are opened.
    fake_pycocotools = ModuleType("pycocotools")
    fake_mask_module = ModuleType("pycocotools.mask")
    fake_mask_module.decode = lambda _rle: [[0, 1], [1, 1]]
    fake_pycocotools.mask = fake_mask_module
    prior_pycocotools = sys.modules.get("pycocotools")
    prior_mask_module = sys.modules.get("pycocotools.mask")
    sys.modules["pycocotools"] = fake_pycocotools
    sys.modules["pycocotools.mask"] = fake_mask_module
    try:
        with tempfile.TemporaryDirectory(prefix="stvg-r1-cpu-contract-") as temp_root:
            mask_dir = Path(temp_root) / "mask_data_filtered_1_3"
            mask_dir.mkdir()
            for local_index in range(4):
                row = {"objects": ({"7": {"rle": {"synthetic": True}}} if local_index in {0, 1} else {})}
                (mask_dir / f"mask_{local_index}.json").write_text(
                    json.dumps(row), encoding="utf-8"
                )
            record = build_teacher_record(
                normalized,
                "<answer>Target ID: 7, Time range: 0.1 to 0.29</answer>",
                mask_data_dir=mask_dir,
                marker_video_sha256="3" * 64,
                costs={
                    "dettrack_s": 0.0,
                    "filter_marker_s": 0.0,
                    "model_preprocess_s": 0.0,
                    "model_generate_s": 0.0,
                    "total_s": 0.0,
                },
                tracker_precision={
                    "autocast": "torch.autocast(device_type='cuda', dtype=torch.bfloat16)",
                    "cuda_compute_capability_major": 8,
                    "allow_tf32_matmul": True,
                    "allow_tf32_cudnn": True,
                },
                preprocess_warnings=[{
                    "category": "UserWarning",
                    "message": "synthetic CUDA extension fallback warning",
                }],
            )
            rows = record["mask_tube"]["frames"]
            assert len(rows) == request["local_frame_count"]
            assert [row["local_frame_index"] for row in rows] == [0, 1, 2, 3]
            assert rows[1]["present"] and rows[1]["xyxy_inclusive_pixels"] == [0, 0, 1, 1]
            assert rows[2]["present"] is False and rows[2]["xyxy_inclusive_pixels"] is None
            assert rows[2]["mask_area_px"] == 0
            assert record["mask_tube"]["coverage"] == {
                "present_frame_count": 1,
                "interval_frame_count": 2,
                "fraction": 0.5,
            }
            assert record["source"]["preprocess_warnings"] == [{
                "category": "UserWarning",
                "message": "synthetic CUDA extension fallback warning",
            }]
    finally:
        if prior_pycocotools is None:
            sys.modules.pop("pycocotools", None)
        else:
            sys.modules["pycocotools"] = prior_pycocotools
        if prior_mask_module is None:
            sys.modules.pop("pycocotools.mask", None)
        else:
            sys.modules["pycocotools.mask"] = prior_mask_module

    gt_payload = dict(request, timestamps=[1.0, 2.0])
    try:
        validate_request(gt_payload)
    except ValueError:
        pass
    else:
        raise AssertionError("GT timestamp field was accepted")
    short_map = dict(request, frame_map=request["frame_map"][:-1])
    try:
        validate_request(short_map)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete frame map was accepted")

    return {
        "status": "pass",
        "mode": "synthetic_cpu_only",
        "model_imports": 0,
        "checkpoint_reads": 0,
        "rgb_reads": 0,
        "gt_reads": 0,
        "tests": [
            "strict inference request whitelist and caller-bound hashes",
            "official prompt preserves the exact query and vision pixel bounds",
            "official single-ID answer parser and int(seconds*fps) physical-frame projection",
            "query and three-way original/canonical file/pixel SHA bindings are verified",
            "unknown candidate, out-of-clock interval and multi-ID outputs fail closed",
            "synthetic mask bbox uses inclusive max pixel endpoints",
            "dense synthetic full-video tube rows retain absent frames and interval-only coverage",
            "preprocessing warnings are retained in source metadata instead of silently qualified",
            "annotation field and incomplete frame map are rejected",
            "module import does not load torch/transformers/opencv/SAM2/YOLO/pycocotools",
            "pinned SAM2 Hydra config prefix normalizes only the official known config path",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-contract", action="store_true", help="run synthetic-only CPU checks")
    args = parser.parse_args()
    if args.cpu_contract:
        print(json.dumps(run_cpu_contracts(), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    parser.error("this module has no implicit run mode; pass --cpu-contract or call run_native_teacher() explicitly")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
