"""CPU contracts and callable source/teacher/student worker APIs.

Importing this module does not load models, checkpoints, videos, or annotations.
The root-controlled worker may explicitly call the producer APIs after its
separate runtime and data gates pass. Synthetic contracts remain independent
of those model and data paths.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONTRACTS_PATH = ROOT / "artifacts/stvg_target_teacher_distillation_v1/student_contracts/CPU_CONTRACTS.json"
OPD_CONFIG_PATH = "methods/decota_spatial_opd_v1/configs.json"
OPD_CONFIG_SHA256 = "fffa324298cf86e00a6445c737be92050983466ecaf20ba036ace685f9a7d1d2"
TEACHER_SCHEMA = "stvg-r1-native-teacher-v1"
FEATURE_SCHEMA = "ta-stvg-cold-source-spatial-features-v1"
TARGET_SCHEMA = "stvg-r1-spatial-supervision-v1"
SCOPE_COUNTS = {"joint": 1792, "LN_only": 1536}
_SENSITIVE_KEY_PARTS = ("groundtruth", "annotation", "label", "oracle", "gt_", "_gt")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _key_scan(value: Any, path: str = "root") -> None:
    """Reject label/annotation fields without inspecting opaque tensor payloads."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).casefold()
            if str(key) != "GT_read" and any(part in name for part in _SENSITIVE_KEY_PARTS):
                raise ValueError(f"label-bearing field is forbidden at {path}.{key}")
            _key_scan(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            _key_scan(child, f"{path}[{i}]")


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


def load_original_opd_config(dataset: str) -> dict[str, Any]:
    """Load the repository's existing, hash-pinned per-domain OPD settings."""
    import hashlib

    path = ROOT / OPD_CONFIG_PATH
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != OPD_CONFIG_SHA256:
        raise RuntimeError("Original OPD config bytes differ from the reviewed pin")
    key = {"hcstvg2": "hc2", "hc2": "hc2", "vidstg": "vidstg"}.get(dataset)
    if key is None:
        raise ValueError(f"Unsupported target domain: {dataset}")
    root = json.loads(path.read_text(encoding="utf-8"))
    config = dict(root["datasets"][key]["config"])
    if set(config) != {"lr", "sigma", "tau", "steps", "writeback", "samples"}:
        raise ValueError("Unexpected OPD configuration schema")
    if config["samples"] != 32 or config["steps"] <= 0 or config["lr"] <= 0 or config["sigma"] <= 0 or config["tau"] <= 0:
        raise ValueError("Invalid original OPD configuration")
    return config


def validate_teacher_record(record: Mapping[str, Any], expected_binding: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the native teacher receipt while never copying its raw answer.

    The native provider's exact schema is consumed here.  Raw model text is
    required only for digesting and is not returned to student artifacts.
    """
    if not isinstance(record, Mapping) or record.get("schema") != TEACHER_SCHEMA:
        raise ValueError("Unexpected native teacher schema")
    required = {"schema", "source", "input_binding", "raw_output", "parsed",
                "projected_interval", "mask_tube", "costs"}
    if set(record) != required:
        raise ValueError(f"Native teacher record keys differ: {sorted(set(record) ^ required)}")
    metadata_only = {k: v for k, v in record.items() if k not in {"raw_output"}}
    _key_scan(metadata_only)
    if record.get("GT_read") is True:
        raise ValueError("Teacher record declares GT access")

    binding = record["input_binding"]
    if not isinstance(binding, Mapping):
        raise ValueError("Teacher input binding must be a mapping")
    expected_fields = ("query_sha256", "frame_clock_sha256", "local_fps",
                       "local_frame_count", "frame_map")
    if not set(expected_fields) <= set(binding):
        raise ValueError("Teacher input binding is missing canonical fields")
    video_hash_fields = ("source_video_sha256", "canonical_video_sha256", "canonical_pixel_sha256")
    if "video_sha256" not in binding and not set(video_hash_fields) <= set(binding):
        raise ValueError("Teacher binding needs a video hash or the three canonical video/pixel hashes")
    for name in ("video_sha256", *video_hash_fields, "query_sha256", "frame_clock_sha256"):
        if name not in binding:
            continue
        _require_sha(binding[name], f"input_binding.{name}")
    for name, expected in expected_binding.items():
        if name == "student_frame_ids":
            continue
        if name not in binding or binding[name] != expected:
            raise ValueError(f"Teacher cache differs from the expected {name}")
    if "query_text" in binding:
        if not isinstance(binding["query_text"], str) or hashlib.sha256(
                binding["query_text"].encode("utf-8")).hexdigest() != binding["query_sha256"]:
            raise ValueError("Teacher query text does not match its UTF-8 digest")
    for name in ("marker_video_sha256", "source_video_sha256", "canonical_video_sha256",
                 "canonical_pixel_sha256"):
        if name in binding:
            _require_sha(binding[name], f"input_binding.{name}")
    fps = _finite_number(binding["local_fps"], "local_fps")
    count = binding["local_frame_count"]
    if fps <= 0 or isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("Invalid local frame clock")
    frame_map = binding["frame_map"]
    if not isinstance(frame_map, list) or len(frame_map) != count:
        raise ValueError("Frame map length differs from local frame count")
    expected_map_keys = {"local_frame_index", "physical_frame_id", "source_timestamp_s"}
    physical_ids = []
    for i, row in enumerate(frame_map):
        if not isinstance(row, Mapping) or set(row) != expected_map_keys:
            raise ValueError("Invalid local-to-physical frame map")
        if row["local_frame_index"] != i:
            raise ValueError("Local frame indices must be complete and ordered")
        physical = row["physical_frame_id"]
        if isinstance(physical, bool) or not isinstance(physical, int) or physical < 0:
            raise ValueError("Physical frame IDs must be nonnegative integers")
        timestamp = _finite_number(row["source_timestamp_s"], "source_timestamp_s")
        if timestamp < 0 or (i and timestamp <= frame_map[i - 1]["source_timestamp_s"]):
            raise ValueError("Source timestamps must be nonnegative and strictly increasing")
        physical_ids.append(physical)
    if len(set(physical_ids)) != len(physical_ids) or any(b <= a for a, b in zip(physical_ids, physical_ids[1:])):
        raise ValueError("Physical frame IDs must be strictly increasing and unique")
    clock = {"local_fps": fps, "local_frame_count": count, "frame_map": frame_map}
    if hashlib.sha256(_canonical_bytes(clock)).hexdigest() != binding["frame_clock_sha256"]:
        raise ValueError("Teacher frame_clock_sha256 does not bind its supplied frame map")
    student_frame_ids = expected_binding.get("student_frame_ids")
    if student_frame_ids is not None and any(frame_id not in set(physical_ids) for frame_id in student_frame_ids):
        raise ValueError("A student source frame is absent from the teacher physical frame map")

    parsed = record["parsed"]
    if not isinstance(parsed, Mapping) or set(parsed) != {"status", "candidate_id", "start_time_s", "end_time_s"}:
        raise ValueError("Invalid parsed teacher result")
    if not isinstance(parsed["status"], str) or not parsed["status"]:
        raise ValueError("Teacher status must be a nonempty string")
    parsed_valid = parsed["status"] == "valid"
    selected = parsed["candidate_id"]
    if parsed_valid:
        if isinstance(selected, bool) or not isinstance(selected, int) or selected <= 0:
            raise ValueError("A valid teacher result must name a positive candidate ID")
        start_s = _finite_number(parsed["start_time_s"], "start_time_s")
        end_s = _finite_number(parsed["end_time_s"], "end_time_s")
        if start_s < 0 or end_s <= start_s:
            raise ValueError("Teacher interval must be finite and nonempty")
    else:
        start_s = end_s = None
        if parsed["start_time_s"] is not None:
            start_s = _finite_number(parsed["start_time_s"], "start_time_s")
        if parsed["end_time_s"] is not None:
            end_s = _finite_number(parsed["end_time_s"], "end_time_s")

    projected = record["projected_interval"]
    if not isinstance(projected, Mapping) or set(projected) != {
            "start_local_frame", "end_local_frame", "local_frame_indices", "physical_frame_ids"}:
        raise ValueError("Invalid projected teacher interval")
    start, end = projected["start_local_frame"], projected["end_local_frame"]
    local = projected["local_frame_indices"]
    projected_physical = projected["physical_frame_ids"]
    has_projected_interval = start is not None or end is not None or bool(local) or bool(projected_physical)
    if has_projected_interval:
        if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int)
                or not isinstance(end, int) or not 0 <= start <= end < count):
            raise ValueError("Projected local interval is outside the mapped frame clock")
        if local != list(range(start, end + 1)):
            raise ValueError("Projected interval must preserve the exact closed local-frame support")
        if projected_physical != [physical_ids[i] for i in local]:
            raise ValueError("Projected physical interval does not match its local frame mapping")
        if not parsed_valid and parsed["status"] not in {"candidate_missing_in_interval"}:
            raise ValueError("Nonvalid teacher output cannot carry a projected interval")
    else:
        if start is not None or end is not None or local != [] or projected_physical != []:
            raise ValueError("Invalid teacher output must use an explicitly empty interval")
        start = end = None
        local = []
        projected_physical = []
    if parsed_valid and not has_projected_interval:
        raise ValueError("Valid teacher output requires a projected interval")

    tube = record["mask_tube"]
    if not isinstance(tube, Mapping) or set(tube) != {"coordinate_convention", "frames", "coverage"}:
        raise ValueError("Invalid teacher mask tube")
    if tube["coordinate_convention"] != "xyxy_inclusive_pixels":
        raise ValueError("Teacher mask boxes must retain their inclusive-pixel convention")
    frames = tube["frames"]
    if not isinstance(frames, list):
        raise ValueError("Teacher tube frames must be a list")
    frame_keys = {"local_frame_index", "physical_frame_id", "source_timestamp_s",
                  "candidate_id", "present", "xyxy_inclusive_pixels", "mask_area_px"}
    seen_local: set[int] = set()
    present_inside_interval = 0
    target_rows = []
    for expected_local_index, row in enumerate(frames):
        if not isinstance(row, Mapping) or set(row) != frame_keys:
            raise ValueError("Invalid teacher tube frame")
        li, pi = row["local_frame_index"], row["physical_frame_id"]
        if isinstance(li, bool) or not isinstance(li, int) or not 0 <= li < count:
            raise ValueError("Teacher box is outside the physical video frame clock")
        if li != expected_local_index or li in seen_local or pi != physical_ids[li] or row["candidate_id"] != selected:
            raise ValueError("Teacher box identity or frame binding mismatch")
        seen_local.add(li)
        xyxy = row["xyxy_inclusive_pixels"]
        present_flag = row["present"]
        if not isinstance(present_flag, bool):
            raise ValueError("Teacher track presence must be an explicit boolean")
        area = row["mask_area_px"]
        timestamp = _finite_number(row["source_timestamp_s"], "tube source_timestamp_s")
        if timestamp != frame_map[li]["source_timestamp_s"]:
            raise ValueError("Tube timestamp differs from the frame-clock binding")
        if present_flag:
            if not isinstance(xyxy, list) or len(xyxy) != 4:
                raise ValueError("Present teacher masks need inclusive-pixel xyxy")
            x1, y1, x2, y2 = [_finite_number(v, "xyxy") for v in xyxy]
            if x2 < x1 or y2 < y1:
                raise ValueError("Empty or reversed teacher mask box")
            if isinstance(area, bool) or not isinstance(area, int) or area <= 0:
                raise ValueError("Present teacher masks need positive area")
            if area > (x2 - x1 + 1) * (y2 - y1 + 1):
                raise ValueError("Mask area exceeds its inclusive bounding box")
            if li in local:
                present_inside_interval += 1
                target_rows.append(dict(local_frame_index=li, physical_frame_id=pi,
                                        xyxy=list(xyxy), mask_area_px=area))
        elif xyxy is not None or isinstance(area, bool) or area != 0:
            raise ValueError("Missing teacher masks must remain unknown with null box and zero area")
    if has_projected_interval and seen_local != set(range(count)):
        raise ValueError("Teacher track tube must retain dense full-video frame presence/absence")
    if not has_projected_interval and frames:
        raise ValueError("An unprojected invalid teacher result cannot carry track frames")

    coverage = tube["coverage"]
    if not isinstance(coverage, Mapping) or set(coverage) != {
            "present_frame_count", "interval_frame_count", "fraction"}:
        raise ValueError("Invalid teacher coverage summary")
    present, interval_count = coverage["present_frame_count"], coverage["interval_frame_count"]
    if present != present_inside_interval or interval_count != len(local):
        raise ValueError("Teacher coverage counts differ from tube support")
    fraction = _finite_number(coverage["fraction"], "coverage fraction")
    expected_fraction = present / interval_count if interval_count else 0.0
    if not 0 <= fraction <= 1 or not math.isclose(fraction, expected_fraction, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Teacher coverage fraction is inconsistent")
    if parsed["status"] == "valid" and present == 0:
        raise ValueError("A valid teacher interval must contain at least one selected-track mask")
    if parsed["status"] == "candidate_missing_in_interval" and present != 0:
        raise ValueError("Teacher missing-candidate status conflicts with the dense mask tube")

    costs = record["costs"]
    cost_fields = {"dettrack_s", "filter_marker_s", "model_preprocess_s", "model_generate_s", "total_s"}
    if not isinstance(costs, Mapping) or set(costs) != cost_fields:
        raise ValueError("Teacher cost record schema differs")
    normalized_costs = {key: _finite_number(costs[key], f"costs.{key}") for key in cost_fields}
    if any(value < 0 for value in normalized_costs.values()):
        raise ValueError("Teacher stage costs cannot be negative")

    raw_output = record["raw_output"]
    if not isinstance(raw_output, str):
        raise ValueError("Native teacher raw output must be text")
    is_usable = parsed_valid and parsed["status"] == "valid" and present > 0
    return {
        "schema": TARGET_SCHEMA,
        "valid": is_usable,
        "source_record_sha256": digest_json(metadata_only),
        "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
        "input_binding_sha256": digest_json(binding),
        "status": parsed["status"],
        "selected_candidate_id_sha256": (hashlib.sha256(str(selected).encode("utf-8")).hexdigest()
                                          if selected is not None else None),
        "parsed_interval_seconds": [start_s, end_s] if start_s is not None and end_s is not None else None,
        "local_interval_closed": [start, end],
        "physical_frame_ids": list(projected_physical),
        "physical_interval": ([projected_physical[0], projected_physical[-1] + 1]
                              if projected_physical else None),
        "coverage": {"present_frame_count": present, "interval_frame_count": interval_count,
                     "fraction": fraction},
        "costs": normalized_costs,
        "target_boxes_inclusive_pixels": target_rows,
        "GT_read": False,
    }


def select_teacher_observations(frame_ids: Sequence[int], teacher_target: Mapping[str, Any], *, k: int = 4):
    """Select the fixed Uniform-K positions first, then drop unknown teacher boxes.

    Missing selected-track masks remain unknown.  They are never materialized
    as zero boxes and do not cause an outcome-dependent replacement sample.
    """
    from methods.decota_refine_uniform_v1.api import uniform_positions

    ids = list(frame_ids)
    if len(ids) < 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in ids):
        raise ValueError("Student frame IDs must be integral")
    if any(b <= a for a, b in zip(ids, ids[1:])):
        raise ValueError("Student frame IDs must be strictly increasing")
    if teacher_target.get("schema") != TARGET_SCHEMA or teacher_target.get("GT_read") is not False:
        raise ValueError("Expected a validated GT-free teacher target")
    if not teacher_target.get("valid") or teacher_target.get("physical_interval") is None:
        return {"selected_positions": [], "unknown_positions": [], "supervised_positions": [],
                "rows": [], "requested_k": k, "teacher_unavailable": True, "GT_read": False}
    interval_start, interval_end = teacher_target["physical_interval"]
    extent_positions = [i for i, frame_id in enumerate(ids)
                        if interval_start <= frame_id < interval_end]
    if not extent_positions:
        return {"selected_positions": [], "unknown_positions": [], "supervised_positions": [],
                "rows": [], "requested_k": k, "teacher_unavailable": False,
                "no_student_frames_in_teacher_interval": True, "GT_read": False}
    chosen = uniform_positions(ids, (extent_positions[0], extent_positions[-1]), k=min(k, len(extent_positions)))
    target_by_physical = {r["physical_frame_id"]: r for r in teacher_target["target_boxes_inclusive_pixels"]}
    selected_rows = []
    unknown = []
    for position in chosen:
        frame_id = ids[position]
        row = target_by_physical.get(frame_id)
        if row is None:
            unknown.append(position)
            continue
        selected_rows.append((position, row))
    return {"selected_positions": chosen, "unknown_positions": unknown,
            "supervised_positions": [p for p, _ in selected_rows],
            "rows": selected_rows, "requested_k": k,
            "GT_read": False}


def inclusive_xyxy_to_cxcywh(box: Sequence[float], width: int, height: int):
    """Convert mask inclusive pixel extrema to normalized edge-coordinate cxcywh."""
    import torch

    if isinstance(width, bool) or isinstance(height, bool) or width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive integers")
    if len(box) != 4:
        raise ValueError("Expected four inclusive xyxy coordinates")
    x1, y1, x2, y2 = [float(v) for v in box]
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("Teacher box coordinates must be finite")
    if x1 < 0 or y1 < 0 or x2 < x1 or y2 < y1 or x2 >= width or y2 >= height:
        raise ValueError("Teacher box falls outside the original image")
    # Inclusive pixel maxima become exclusive right/bottom image edges.
    left, top, right, bottom = x1, y1, x2 + 1.0, y2 + 1.0
    return torch.tensor([((left + right) / 2) / width,
                         ((top + bottom) / 2) / height,
                         (right - left) / width,
                         (bottom - top) / height], dtype=torch.float32)


def anchors_from_teacher_observations(observations: Mapping[str, Any], width: int, height: int):
    """Build only valid K4 anchors; all unobserved frames remain unlabeled."""
    return [{"position": int(position),
             "box": inclusive_xyxy_to_cxcywh(row["xyxy"], width, height),
             "weight": 1.0,
             "physical_frame_id": int(row["physical_frame_id"]),
             "mask_area_px": int(row["mask_area_px"])}
            for position, row in observations["rows"]]


def _copy_cpu(value):
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy_cpu(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_copy_cpu(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_copy_cpu(child) for child in value)
    return copy.deepcopy(value)


def _move_tree(value, device):
    import torch

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tree(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [_move_tree(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(_move_tree(child, device) for child in value)
    return copy.deepcopy(value)


def _tree_equal(left, right) -> bool:
    import torch

    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def _state_delta(before: Mapping[str, Any], after: Mapping[str, Any]):
    return {name: after[name].detach() - before[name].detach() for name in before}


def _decoder_jvp(replay, state_values, state_tangents):
    """Linearized all-frame native decoder movement for the actual Adam step."""
    import torch
    from torch.func import functional_call

    names = [name for name, _ in replay.named]
    module = replay.base.decoder.decoder
    frozen_params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())

    def decode(*values):
        current = dict(zip(names, values))
        params = dict(frozen_params)
        delta = current["spatial.query_residual"]
        for name, value in current.items():
            if name.startswith("spatial.layers."):
                params[name.removeprefix("spatial.")] = value
        per_offset = []
        for inputs in replay.base.spatial_inputs:
            kwargs = dict(inputs)
            kwargs["query_tgt"] = kwargs["query_tgt"] + delta.to(kwargs["query_tgt"])[None, None, :]
            pos = functional_call(module, (params, buffers), (), kwargs)
            per_offset.append(pos.flatten(1, 2)[-1].reshape(-1, 4))
        return torch.stack([per_offset[i % 2][i // 2] for i in range(len(replay.ids))])

    primals = tuple(v.detach().requires_grad_(True) for v in state_values)
    tangents = tuple(v.detach() for v in state_tangents)
    _, expected = torch.autograd.functional.jvp(decode, primals, tangents, create_graph=False, strict=False)
    return expected.detach()


def _movement_summary(expected, actual, positions):
    import torch

    pos = list(positions)
    if not pos:
        return {"positions": [], "expected_l2": 0.0, "actual_l2": 0.0,
                "cosine": None, "relative_error": None}
    e = expected[pos].reshape(-1).double()
    a = actual[pos].reshape(-1).double()
    en, an = torch.linalg.vector_norm(e), torch.linalg.vector_norm(a)
    cosine = None if en == 0 or an == 0 else float(torch.dot(e, a) / (en * an))
    relative = None if en == 0 else float(torch.linalg.vector_norm(a - e) / en)
    return {"positions": pos, "expected_l2": float(en), "actual_l2": float(an),
            "cosine": cosine, "relative_error": relative,
            "expected_delta": expected[pos].detach().cpu(), "actual_delta": actual[pos].detach().cpu()}


def _rng_snapshot():
    import numpy as np
    import torch

    out = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        out["torch_cuda"] = torch.cuda.get_rng_state_all()
    return _copy_cpu(out)


def restore_rng_snapshot(snapshot):
    """Restore the exact worker RNG state recorded at a complete-step boundary."""
    import numpy as np
    import torch

    required = {"python", "numpy", "torch_cpu"}
    if not isinstance(snapshot, Mapping) or not required <= set(snapshot):
        raise ValueError("Incomplete resumable RNG snapshot")
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"].cpu())
    if "torch_cuda" in snapshot:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            raise RuntimeError("Cannot resume a CUDA RNG snapshot outside the original CUDA worker")
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def save_progress_checkpoint(path, payload):
    """Atomically replace only a mutable training-progress checkpoint."""
    import os
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, tmp)
        with tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def assert_gt_free_student_batch(batch):
    """Require the same inference-only, whitelisted TA-STVG batch fields."""
    if not isinstance(batch, Mapping) or set(batch) != {"videos", "durations", "texts", "targets"}:
        raise ValueError("Student input must be the inference-only TA-STVG batch schema")
    if not isinstance(batch["texts"], (list, tuple)) or not batch["texts"] or not all(
            isinstance(text, str) and text.strip() for text in batch["texts"]):
        raise ValueError("Student inference query is missing")
    if not isinstance(batch["targets"], list) or len(batch["targets"]) != 1:
        raise ValueError("Expected one inference-only target metadata record")
    target = batch["targets"][0]
    allowed_target = {"item_id", "vid", "frame_ids", "actioness", "img_size", "ori_size"}
    if not isinstance(target, Mapping) or set(target) - allowed_target:
        raise ValueError("Target metadata contains fields outside the no-GT whitelist")
    if "actioness" in target:
        values = target["actioness"]
        if hasattr(values, "numel"):
            nonzero = bool(values.numel() and values.any())
        else:
            try:
                nonzero = any(float(value) != 0 for value in values)
            except TypeError:
                nonzero = float(values) != 0
        if nonzero:
            raise ValueError("Inference actionness metadata must be all-zero, never target labels")
    _key_scan({"target_metadata": {k: v for k, v in target.items() if k != "actioness"}})
    return True


def _cold_replay(model, frames, row, expected_source):
    """Fresh source-only capture through the project's frozen-forward producer."""
    import torch
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward

    frame_ids = list(row["frame_ids"])
    if expected_source.get("GT_read") is not False or expected_source.get("DINO_state_inherited") is not False:
        raise RuntimeError("Source record must certify no GT and no inherited DINO state")
    _require_sha(expected_source.get("checkpoint_sha256"), "checkpoint_sha256")
    _require_sha(expected_source.get("pixel_sha256"), "pixel_sha256")
    _require_sha(expected_source.get("source_model_state_sha256"), "source_model_state_sha256")
    if model.training or any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Student producer requires a cold, frozen source model")
    model_sha = getattr(model, "_fixed_full_state_hash", None)
    if not model_sha or model_sha != expected_source["source_model_state_sha256"]:
        raise RuntimeError("Loaded student is not the sealed source checkpoint")
    before_hash = state_hash(model.state_dict())
    actual_pixel_sha256 = hashlib.sha256(frames.tobytes()).hexdigest()
    if actual_pixel_sha256 != expected_source["pixel_sha256"]:
        raise RuntimeError("Fresh decoded RGB pixels differ from the source input seal")
    batch, records, base = frozen_forward(model, frames, row)
    assert_gt_free_student_batch(batch)
    native = prediction(base.zero["logits"], base.zero["boxes"], records, list(frame_ids))
    if before_hash != state_hash(model.state_dict()) or model_sha != getattr(model, "_fixed_full_state_hash", None):
        raise RuntimeError("Source capture mutated the frozen source model")
    if list(frame_ids) != list(expected_source["frame_ids"]):
        raise RuntimeError("Source feature capture is on another frame grid")
    if native["physical_interval"] != expected_source["physical_interval"]:
        raise RuntimeError("Fresh cold source temporal readout differs from sealed source")
    expected_boxes = torch.as_tensor(expected_source["boxes"]).detach().cpu()
    if not torch.equal(native["boxes"], expected_boxes):
        raise RuntimeError("Fresh cold source boxes differ from sealed source")
    expected_logits = [torch.as_tensor(value).detach().cpu() for value in expected_source["logits"]]
    if len(expected_logits) != len(native["logits"]) or any(
            not torch.equal(a, b) for a, b in zip(native["logits"], expected_logits)):
        raise RuntimeError("Fresh cold source temporal logits differ from sealed source")
    return batch, base, records, native, before_hash, actual_pixel_sha256


def snapshot_cold_source_features(base, records, native, *, binding: Mapping[str, Any],
                                  source_model_state_sha256: str, checkpoint_sha256: str):
    """Create a private raw-prefix cache for serial teacher/student residency."""
    _require_sha(source_model_state_sha256, "source_model_state_sha256")
    _require_sha(checkpoint_sha256, "checkpoint_sha256")
    _key_scan(binding)
    required_binding = {"dataset", "split", "source_parent_sha256", "pixel_sha256", "frame_ids"}
    if set(binding) != required_binding or binding["split"] != "dev":
        raise ValueError("Student source cache requires exact development input provenance")
    _require_sha(binding["source_parent_sha256"], "source_parent_sha256")
    _require_sha(binding["pixel_sha256"], "pixel_sha256")
    if len(base.views) != 2 or len(records) != 2:
        raise ValueError("Expected two source temporal offsets")
    return {"schema": FEATURE_SCHEMA,
            "binding": copy.deepcopy(dict(binding)),
            "source_model_state_sha256": source_model_state_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "views": _copy_cpu(base.views),
            "records": _copy_cpu(records),
            "source_initial": _copy_cpu(base.initial),
            "source_zero": _copy_cpu(base.zero),
            "source_prediction": _copy_cpu(native),
            "producer": {"name": "vg_tta.stvg_opd_frozen_mask_cache012.frozen_forward",
                         "source_only": True, "DINO_calls": 0, "teacher_calls": 0,
                         "student_updates": 0, "GT_read": False}}


def restore_cold_source_replay(model, feature_cache: Mapping[str, Any], *, expected_binding: Mapping[str, Any],
                               expected_checkpoint_sha256: str):
    """Rebuild a replay from cold source model weights and a verified source-only cache."""
    import torch
    from methods.decota_final_simplified_v1.replay import SpatialReplay
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.tensors import state_hash

    if feature_cache.get("schema") != FEATURE_SCHEMA or dict(feature_cache.get("binding", {})) != dict(expected_binding):
        raise ValueError("Feature cache binding mismatch")
    producer = feature_cache.get("producer", {})
    if producer != {"name": "vg_tta.stvg_opd_frozen_mask_cache012.frozen_forward",
                    "source_only": True, "DINO_calls": 0, "teacher_calls": 0,
                    "student_updates": 0, "GT_read": False}:
        raise ValueError("Feature cache is not a cold source-only capture")
    if feature_cache.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("Student checkpoint binding mismatch")
    model_sha = getattr(model, "_fixed_full_state_hash", None)
    if model.training or any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Restoration requires a fresh frozen source model")
    if model_sha != feature_cache["source_model_state_sha256"] or state_hash(model.state_dict()) != model_sha:
        raise RuntimeError("Restored student weights are not the exact cold source state")
    views = _move_tree(feature_cache["views"], next(model.parameters()).device)
    base = SpatialReplay(model, views, len(expected_binding["frame_ids"]), cached=True)
    saved = feature_cache["source_prediction"]
    native = prediction(base.zero["logits"], base.zero["boxes"], feature_cache["records"],
                        list(expected_binding["frame_ids"]))
    if not torch.equal(native["boxes"], saved["boxes"]):
        raise RuntimeError("Restored replay does not reproduce source-only box capture")
    if any(not torch.equal(a, b) for a, b in zip(native["logits"], saved["logits"])):
        raise RuntimeError("Restored replay does not reproduce source-only temporal capture")
    if any(not torch.equal(base.initial[k].cpu(), v) for k, v in feature_cache["source_initial"].items()):
        raise RuntimeError("Restored source spatial interface differs from its cold snapshot")
    if not _tree_equal(_copy_cpu(base.zero), feature_cache["source_zero"]):
        raise RuntimeError("Restored source replay does not reproduce the complete cold zero state")
    return base, list(feature_cache["records"]), native


def _trainable_replay(base, frame_ids, key, scope):
    from vg_tta.c1_enabling_tricks_v1 import TrickReplay

    if scope not in SCOPE_COUNTS:
        raise ValueError(f"Unsupported parameter scope: {scope}")
    replay = TrickReplay(base, list(frame_ids), key, {})
    selected = [(name, parameter) for name, parameter in replay.named
                if scope == "joint" or name != "spatial.query_residual"]
    if sum(parameter.numel() for _, parameter in selected) != SCOPE_COUNTS[scope]:
        raise RuntimeError("Student active parameter count differs from the declared interface")
    for name, parameter in replay.named:
        parameter.requires_grad_(any(name == selected_name for selected_name, _ in selected))
    return replay, selected


def _initialize_fit_resume(replay, optimizer, resume_state, *, arm, scope, key,
                           frame_ids, config, origin):
    from methods.decota_final_simplified_v1.tensors import state_hash

    if resume_state is None:
        return 0, [], [], 0
    expected = {"schema": "ta-stvg-fit-progress-v1", "arm": arm, "scope": scope,
                "key": key, "frame_ids": list(frame_ids), "config": dict(config),
                "initial_state_sha256": state_hash(origin)}
    for name, value in expected.items():
        if resume_state.get(name) != value:
            raise ValueError(f"Resume checkpoint does not match this fit's {name}")
    next_step = resume_state.get("next_step")
    if isinstance(next_step, bool) or not isinstance(next_step, int) or not 0 <= next_step <= config["steps"]:
        raise ValueError("Resume checkpoint next_step is outside the declared step range")
    saved_state = resume_state.get("student_state")
    if not isinstance(saved_state, Mapping) or set(saved_state) != set(origin):
        raise ValueError("Resume checkpoint student state schema differs")
    replay.restore({name: saved_state[name].to(value) for name, value in origin.items()})
    optimizer_state = resume_state.get("optimizer_state")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("Resume checkpoint omitted optimizer state")
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    restore_rng_snapshot(resume_state["rng_state"])
    path = resume_state.get("path")
    rounds = resume_state.get("rounds")
    if not isinstance(path, list) or not isinstance(rounds, list):
        raise ValueError("Resume checkpoint omitted fit history")
    updated_steps = int(resume_state.get("optimizer_steps", -1))
    if updated_steps < 0:
        raise ValueError("Resume checkpoint optimizer step count is invalid")
    return next_step, copy.deepcopy(path), copy.deepcopy(rounds), updated_steps


def _fit_progress(replay, optimizer, *, arm, scope, key, frame_ids, config,
                  origin, next_step, path, rounds, optimizer_steps):
    from methods.decota_final_simplified_v1.tensors import detached, state_hash

    return {"schema": "ta-stvg-fit-progress-v1", "arm": arm, "scope": scope,
            "key": key, "frame_ids": list(frame_ids), "config": dict(config),
            "initial_state_sha256": state_hash(origin), "next_step": next_step,
            "student_state": detached(replay.state(), "cpu"),
            "optimizer_state": _copy_cpu(optimizer.state_dict()),
            "rng_state": _rng_snapshot(), "path": copy.deepcopy(path),
            "rounds": copy.deepcopy(rounds), "optimizer_steps": optimizer_steps,
            "GT_read": False}


def _finalize_student_readout(model, batch, frame_ids, records, state):
    """Run a fresh full TA-STVG source model forward with the fitted student state."""
    import torch
    from methods.decota_final_simplified_v1.backbone import inserted_state, offset_batch
    from methods.decota_final_simplified_v1.objectives import prediction

    boxes_by_offset, logits = [], []
    with inserted_state(model, state):
        for offset in (0, 1):
            view = offset_batch(batch, offset)
            view["videos"] = view["videos"].to(next(model.parameters()).device)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(view["videos"], view["texts"], view["targets"], iteration_rate=-1)
            boxes_by_offset.append(output["pred_boxes"])
            logits.append(output["pred_sted"])
    boxes = torch.stack([boxes_by_offset[i % 2][i // 2] for i in range(len(frame_ids))]).float()
    return prediction(logits, boxes, records, list(frame_ids))


def _fit_direct(base, initial, anchors, frame_ids, key, config, scope="joint", *,
                resume_state=None, checkpoint_callback=None):
    fit_started = time.perf_counter()
    import torch
    from methods.decota_final_simplified_v1.objectives import SpatialLoss
    from methods.decota_final_simplified_v1.tensors import detached

    base.restore(initial)
    replay, selected = _trainable_replay(base, frame_ids, key, scope)
    origin = replay.state()
    params = [p for _, p in selected]
    loss_anchors = [{**anchor, "box": anchor["box"].detach().cpu().tolist()}
                    for anchor in anchors]
    loss_fn = SpatialLoss(loss_anchors, base.zero["boxes"])
    optimizer = torch.optim.Adam(params, lr=config["lr"], betas=(.9, .999), eps=1e-8, weight_decay=0.)
    next_step, path, _, optimizer_steps = _initialize_fit_resume(
        replay, optimizer, resume_state, arm="Direct", scope=scope, key=key,
        frame_ids=frame_ids, config=config, origin=origin)

    def publish_progress(next_index):
        if checkpoint_callback is not None:
            checkpoint_callback(_fit_progress(replay, optimizer, arm="Direct", scope=scope,
                key=key, frame_ids=frame_ids, config=config, origin=origin,
                next_step=next_index, path=path, rounds=[], optimizer_steps=optimizer_steps))

    try:
        for step in range(next_step, config["steps"]):
            step_started = time.perf_counter()
            before_state = replay.state()
            before_values = replay.values()
            before_boxes = before_values["boxes"].detach().clone()
            entry = {"step": step, "boxes": before_boxes.detach().cpu(),
                     "state": detached(before_state, "cpu"),
                     "optimizer_state": _copy_cpu(optimizer.state_dict()),
                     "rng_state_before": _rng_snapshot()}
            if loss_fn.empty:
                entry.update(skipped=True, reason="no_valid_teacher_observations",
                             step_wall_seconds=time.perf_counter() - step_started)
                path.append(entry)
                publish_progress(step + 1)
                continue
            prediction_boxes = replay.values()["boxes"]
            # Preserve the original fixed denominator of four planned anchors.
            # Missing teacher boxes stay unknown and do not reweight the loss.
            loss = loss_fn(prediction_boxes)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite direct distillation loss")
            grads = torch.autograd.grad(loss, params)
            if any(not torch.isfinite(grad).all() for grad in grads):
                raise FloatingPointError("Nonfinite direct distillation gradient")
            for parameter, grad in zip(params, grads):
                parameter.grad = grad.detach()
            optimizer_before = _copy_cpu(optimizer.state_dict())
            optimizer.step()
            after_state = replay.state()
            with torch.no_grad():
                after_boxes = replay.values()["boxes"].detach().clone()
            tangent = _state_delta(before_state, after_state)
            expected_move = _decoder_jvp(replay, list(before_state.values()), list(tangent.values()))
            actual_move = after_boxes - before_boxes
            expected_optimizer = _copy_cpu(optimizer.state_dict())
            entry.update(skipped=False, loss=float(loss.detach()), optimizer_before=optimizer_before,
                         optimizer_state=expected_optimizer, update_state=detached(after_state, "cpu"),
                         step_wall_seconds=time.perf_counter() - step_started,
                         expected_delta=expected_move.cpu(), actual_delta=actual_move.cpu(),
                         observed_movement=_movement_summary(expected_move, actual_move,
                             [a["position"] for a in anchors]),
                         unobserved_movement=_movement_summary(expected_move, actual_move,
                             [i for i in range(len(frame_ids)) if i not in {a["position"] for a in anchors}]))
            path.append(entry)
            optimizer_steps += 1
            publish_progress(step + 1)
        final_state = replay.state()
        with torch.no_grad():
            final_cached = replay.values()["boxes"].detach().cpu()
        return {"schema": "ta-stvg-student-fit-v1", "arm": "Direct", "scope": scope,
                "initial_state": detached(origin, "cpu"), "final_state": detached(final_state, "cpu"),
                "final_cached_decoder_boxes": final_cached, "path": path, "optimizer_steps": optimizer_steps,
                "selected_step": config["steps"], "declared_steps": config["steps"],
                "active_parameters": SCOPE_COUNTS[scope], "config": dict(config),
                "wall_seconds": time.perf_counter() - fit_started,
                "observation_positions": [a["position"] for a in anchors],
                "GT_read": False, "output_source": "student native decoder readout required"}
    finally:
        replay.restore(origin)
        base.restore(initial)


def _fit_spatial_opd(base, initial, anchors, frame_ids, key, config, scope="joint", *,
                     resume_state=None, checkpoint_callback=None):
    fit_started = time.perf_counter()
    import torch
    from methods.decota_final_simplified_v1.tensors import detached
    from vg_tta.decota_spatial_opd_tunable_v1 import (
        likelihood_loss, mean_coordinates, optimizer_step, rollout,
    )

    base.restore(initial)
    replay, selected = _trainable_replay(base, frame_ids, key, scope)
    origin = replay.state()
    params = [p for _, p in selected]
    positions = [int(a["position"]) for a in anchors]
    if len(set(positions)) != len(positions) or any(not 0 <= p < len(frame_ids) for p in positions):
        raise ValueError("Invalid K4 teacher positions")
    if anchors:
        evidence = torch.stack([a["box"].detach().to(base.zero["boxes"]) for a in anchors])
    else:
        evidence = base.zero["boxes"].new_empty((0, 4))
    optimizer = torch.optim.Adam(params, lr=config["lr"], betas=(.9, .999), eps=1e-8, weight_decay=0.)
    next_step, path, rounds, optimizer_steps = _initialize_fit_resume(
        replay, optimizer, resume_state, arm="SpatialOPD", scope=scope, key=key,
        frame_ids=frame_ids, config=config, origin=origin)

    def publish_progress(next_index):
        if checkpoint_callback is not None:
            checkpoint_callback(_fit_progress(replay, optimizer, arm="SpatialOPD", scope=scope,
                key=key, frame_ids=frame_ids, config=config, origin=origin,
                next_step=next_index, path=path, rounds=rounds, optimizer_steps=optimizer_steps))

    try:
        for step in range(next_step, config["steps"]):
            step_started = time.perf_counter()
            before_state = replay.state()
            before_values = replay.values()
            before_boxes = before_values["boxes"].detach().clone()
            entry = {"step": step, "boxes": before_boxes.detach().cpu(),
                     "state": detached(before_state, "cpu"),
                     "optimizer_state": _copy_cpu(optimizer.state_dict()),
                     "rng_state_before": _rng_snapshot()}
            if not positions:
                entry.update(skipped=True, reason="no_valid_teacher_observations",
                             step_wall_seconds=time.perf_counter() - step_started)
                rounds.append({"step": step, "no_information": True, "updated": False,
                               "wall_seconds": time.perf_counter() - step_started})
                path.append(entry)
                publish_progress(step + 1)
                continue
            mu = mean_coordinates(before_values["boxes"][positions])
            policy = rollout(mu, evidence, key, step, shuffle=False, config=config)
            loss = likelihood_loss(mu, policy["mean"], policy["samples"], policy["weights"],
                                   sigma=config["sigma"])
            policy_mean_box = torch.sigmoid((policy["weights"][..., None] * policy["samples"]).sum(-2))
            policy_delta = policy_mean_box - before_boxes[positions].detach()
            optimizer_before = _copy_cpu(optimizer.state_dict())
            before_params = detached(replay.state(), "cpu")
            update = optimizer_step(optimizer, loss, params, bool(policy["informative"]))
            after_state = replay.state()
            optimizer_after = _copy_cpu(optimizer.state_dict())
            if update is None:
                if not _tree_equal(before_params, detached(after_state, "cpu")):
                    raise AssertionError("Equal-reward OPD updated student parameters")
                if not _tree_equal(optimizer_before, optimizer_after):
                    raise AssertionError("Equal-reward OPD mutated Adam state")
            else:
                optimizer_steps += 1
            with torch.no_grad():
                after_boxes = replay.values()["boxes"].detach().clone()
            tangent = _state_delta(before_state, after_state)
            expected_move = _decoder_jvp(replay, list(before_state.values()), list(tangent.values()))
            actual_move = after_boxes - before_boxes
            mask = [i for i in range(len(frame_ids)) if i not in set(positions)]
            round_record = {"step": step, "updated": update is not None,
                "wall_seconds": time.perf_counter() - step_started,
                "no_information": not bool(policy["informative"]), "loss": float(loss.detach()),
                "reward_variance": policy["rewards"].var(1, unbiased=False).detach().cpu(),
                "weight_entropy": (-(policy["weights"] * policy["weights"].clamp_min(1e-30).log()).sum(1)).detach().cpu(),
                "weight_ess": (1.0 / policy["weights"].square().sum(1)).detach().cpu(),
                "expected_policy_delta_observed": policy_delta.detach().cpu(),
                "expected_delta_all_positions": expected_move.detach().cpu(),
                "actual_delta_all_positions": actual_move.detach().cpu(),
                "observed_movement": _movement_summary(expected_move, actual_move, positions),
                "unobserved_movement": _movement_summary(expected_move, actual_move, mask),
                "actions_detached": not policy["samples"].requires_grad,
                "teacher_targets_detached": not evidence.requires_grad,
                "reward_weights_detached": not policy["weights"].requires_grad}
            rounds.append(round_record)
            entry.update(skipped=False, loss=float(loss.detach()), rollout=detached(policy, "cpu"),
                         step_wall_seconds=time.perf_counter() - step_started,
                         optimizer_before=optimizer_before, optimizer_state=optimizer_after,
                         update=None if update is None else detached(update, "cpu"),
                         update_state=detached(after_state, "cpu"),
                         expected_delta=expected_move.detach().cpu(), actual_delta=actual_move.detach().cpu())
            path.append(entry)
            publish_progress(step + 1)
        final_state = replay.state()
        with torch.no_grad():
            final_cached = replay.values()["boxes"].detach().cpu()
        return {"schema": "ta-stvg-student-fit-v1", "arm": "SpatialOPD", "scope": scope,
                "initial_state": detached(origin, "cpu"), "final_state": detached(final_state, "cpu"),
                "final_cached_decoder_boxes": final_cached, "path": path, "rounds": rounds,
                "optimizer_steps": optimizer_steps, "selected_step": config["steps"],
                "declared_steps": config["steps"], "active_parameters": SCOPE_COUNTS[scope],
                "config": dict(config), "wall_seconds": time.perf_counter() - fit_started,
                "observation_positions": positions,
                "policy": "original antithetic on-policy Gaussian OPD",
                "teacher_box_in_action_pool": False, "GT_read": False,
                "output_source": "student native decoder readout required"}
    finally:
        replay.restore(origin)
        base.restore(initial)


def no_parameter_refine(source_boxes, frame_ids, anchors):
    """Teacher K4 piecewise-linear box control without parameter changes/extrapolation."""
    import torch

    ids = torch.as_tensor(frame_ids, dtype=torch.float64)
    boxes = source_boxes.detach().clone()
    if boxes.shape != (len(ids), 4) or not torch.isfinite(boxes).all():
        raise ValueError("Invalid source boxes")
    rows = sorted(anchors, key=lambda a: a["position"])
    if not rows:
        return {"boxes": boxes, "changed_positions": [], "unknown_positions": list(range(len(ids))),
                "GT_read": False}
    positions = [int(row["position"]) for row in rows]
    targets = torch.stack([row["box"].to(boxes) for row in rows])
    for position, target in zip(positions, targets):
        boxes[position] = target
    for left, right in zip(range(len(positions) - 1), range(1, len(positions))):
        i, j = positions[left], positions[right]
        for k in range(i + 1, j):
            alpha = ((ids[k] - ids[i]) / (ids[j] - ids[i])).to(boxes)
            boxes[k] = targets[left] + alpha * (targets[right] - targets[left])
    covered = set(range(positions[0], positions[-1] + 1))
    changed = sorted(covered)
    return {"boxes": boxes, "changed_positions": changed,
            "unknown_positions": [i for i in range(len(ids)) if i not in covered],
            "teacher_observations": positions, "interpolation_hull_only": True,
            "GT_read": False}


def fixed_time_readout(student_prediction: Mapping[str, Any], interval: Sequence[int] | None, *, source: str):
    """Pair spatial predictions with one locked interval for component isolation."""
    if source not in {"teacher_interval", "projection_interval"}:
        raise ValueError("Time component must be teacher_interval or projection_interval")
    if interval is None:
        return {"status": "unavailable", "boxes": student_prediction["boxes"].detach().cpu().clone(),
                "physical_interval": None, "time_factor_source": source,
                "GT_read": False, "reason": "teacher_interval_unavailable"}
    if len(interval) != 2 or int(interval[1]) <= int(interval[0]):
        raise ValueError("Expected a nonempty half-open physical interval")
    return {"boxes": student_prediction["boxes"].detach().cpu().clone(),
            "physical_interval": [int(interval[0]), int(interval[1])],
            "time_factor_source": source, "GT_read": False,
            "not_a_full_student_native_decode": True}


def build_case_readouts(model, batch, base, records, frame_ids, teacher_target,
                        source_prediction, width, height, dataset, key,
                        *, source_custody: Mapping[str, Any], scope="joint",
                        direct_resume=None, opd_resume=None,
                        direct_checkpoint_callback=None, opd_checkpoint_callback=None):
    """Build matched source/projection/teacher/refine/direct/OPD case outputs.

    This explicit callable is used by the separately gated root worker. It
    never chooses a best step from labels or writes teacher boxes over student
    readouts.
    """
    import torch
    from methods.decota_final_simplified_v1.objectives import project
    from scripts.stvg_opd_p6_existing_common001 import check_projection, config as projection_config

    if dataset not in {"hc2", "hcstvg2", "vidstg"}:
        raise ValueError("Unsupported student target dataset")
    custody_fields = {"checkpoint_sha256", "source_model_state_sha256", "source_input_binding_sha256",
                      "DINO_state_inherited", "GT_read"}
    if set(source_custody) != custody_fields or source_custody["DINO_state_inherited"] is not False or source_custody["GT_read"] is not False:
        raise ValueError("Missing cold source custody or DINO-free initialization proof")
    for field in ("checkpoint_sha256", "source_model_state_sha256", "source_input_binding_sha256"):
        _require_sha(source_custody[field], field)
    if source_prediction["boxes"].shape != (len(frame_ids), 4):
        raise ValueError("Source student box grid differs")
    observations = select_teacher_observations(frame_ids, teacher_target)
    anchors = anchors_from_teacher_observations(observations, width, height)
    config = load_original_opd_config(dataset)
    temporal = projection_config(dataset)
    projected = project(base.zero["logits"], base.zero["actions"], records, temporal)
    projection_math = check_projection(base.zero["logits"], base.zero["actions"], records,
                                       projected, temporal)
    projection_intervals = [row["interval"] for row in projected["offsets"]]
    projection_interval = [min(row[0] for row in projection_intervals),
                           max(row[1] for row in projection_intervals)]
    refined = no_parameter_refine(source_prediction["boxes"], frame_ids, anchors)
    initial = _copy_cpu(base.initial)
    # Model tensors may live on CUDA; copy the exact source active state back to
    # the original device rather than importing any prior adaptation state.
    initial = {name: value.to(next(base.model.parameters()).device) for name, value in initial.items()}
    direct = _fit_direct(base, initial, anchors, frame_ids, key, config, scope=scope,
                         resume_state=direct_resume, checkpoint_callback=direct_checkpoint_callback)
    opd = _fit_spatial_opd(base, initial, anchors, frame_ids, key, config, scope=scope,
                           resume_state=opd_resume, checkpoint_callback=opd_checkpoint_callback)
    native_decode_started = time.perf_counter()
    direct_readout = _finalize_student_readout(model, batch, frame_ids, records, direct["final_state"])
    opd_readout = _finalize_student_readout(model, batch, frame_ids, records, opd["final_state"])
    native_decode_seconds = time.perf_counter() - native_decode_started
    teacher_interval = teacher_target.get("physical_interval") if teacher_target.get("valid") else None
    # Direct and OPD are paired under the same projection-time component; their
    # own full native temporal decodes are saved separately and never overwritten.
    return {"schema": "stvg-target-distillation-case-v1", "dataset": dataset,
        "source": {"boxes": source_prediction["boxes"].detach().cpu(),
                   "physical_interval": list(source_prediction["physical_interval"])},
        "projection": {"boxes": source_prediction["boxes"].detach().cpu(),
                       "physical_interval": projection_interval,
                       "raw_offsets": [dict(interval=row["interval"], target_indices=row["target_indices"])
                                       for row in projected["offsets"]],
                       "independent_math_readback": projection_math},
        "teacher_readout_binding": {"record_sha256": teacher_target["source_record_sha256"],
            "status": teacher_target["status"], "valid": teacher_target["valid"],
            "candidate_id_sha256": teacher_target["selected_candidate_id_sha256"],
            "physical_interval": teacher_interval,
            "coverage": teacher_target["coverage"], "costs": teacher_target["costs"]},
        "teacher_time_only": fixed_time_readout(source_prediction, teacher_interval, source="teacher_interval"),
        "projection_time_only": fixed_time_readout(source_prediction, projection_interval, source="projection_interval"),
        "no_parameter_refine": refined,
        "direct": {"fit": direct, "native_decode": direct_readout,
                   "projection_time_paired": fixed_time_readout(direct_readout, projection_interval,
                                                                 source="projection_interval")},
        "spatial_opd": {"fit": opd, "native_decode": opd_readout,
                        "projection_time_paired": fixed_time_readout(opd_readout, projection_interval,
                                                                      source="projection_interval")},
        "observation_selection": {k: v for k, v in observations.items() if k != "rows"}
            | {"anchors": [{"position": a["position"], "physical_frame_id": a["physical_frame_id"],
                            "target": a["box"].detach().cpu(), "mask_area_px": a["mask_area_px"]} for a in anchors]},
        "configuration": {"spatial_opd": config, "spatial_opd_config_source": OPD_CONFIG_PATH,
                          "spatial_opd_config_sha256": OPD_CONFIG_SHA256,
                          "teacher_box_conversion": "inclusive pixel extrema [xmin,xmax] map to continuous image edges [xmin,xmax+1], then normalized cxcywh",
                          "spatial_scope": scope, "temporal_projection": temporal.to_dict()},
        "costs": {"direct_fit_seconds": direct["wall_seconds"],
                  "spatial_opd_fit_seconds": opd["wall_seconds"],
                  "student_full_native_readout_seconds": native_decode_seconds},
        "source_custody": dict(source_custody),
        "GT_read": False, "target_boxes_do_not_overwrite_final_student_decode": True}


def capture_cold_source_features(model, frames, row, *, expected_source: Mapping[str, Any],
                                 feature_binding: Mapping[str, Any]):
    """Cold-recapture the original TA-STVG source and return a private feature cache.

    `expected_source` must come from the already-sealed source-only screen and
    include source boxes/logits/interval, exact frame IDs, RGB pixel SHA,
    checkpoint file SHA, model state SHA, and explicit GT/DINO false flags.
    """
    capture_started = time.perf_counter()
    batch, base, records, native, state_sha, pixel_sha = _cold_replay(
        model, frames, row, expected_source)
    if list(feature_binding.get("frame_ids", [])) != list(row["frame_ids"]):
        raise ValueError("Feature binding frame IDs differ from the source input")
    if feature_binding.get("pixel_sha256") != pixel_sha:
        raise ValueError("Feature binding RGB hash differs from the cold source pixels")
    feature_cache = snapshot_cold_source_features(
        base, records, native, binding=feature_binding,
        source_model_state_sha256=state_sha,
        checkpoint_sha256=expected_source["checkpoint_sha256"])
    source_custody = {"checkpoint_sha256": expected_source["checkpoint_sha256"],
        "source_model_state_sha256": state_sha,
        "source_input_binding_sha256": digest_json(dict(feature_binding)),
        "DINO_state_inherited": False, "GT_read": False}
    return {"batch": batch, "base": base, "records": records, "source_prediction": native,
            "feature_cache": feature_cache, "source_custody": source_custody,
            "actual_pixel_sha256": pixel_sha,
            "source_capture_seconds": time.perf_counter() - capture_started,
            "GT_read": False}


def fit_teacher_student_case(model, cold_capture: Mapping[str, Any], frame_ids: Sequence[int],
                             teacher_record: Mapping[str, Any], *,
                             expected_teacher_binding: Mapping[str, Any], width: int, height: int,
                             dataset: str, key: str, scope="joint", direct_resume=None,
                             opd_resume=None, direct_checkpoint_callback=None,
                             opd_checkpoint_callback=None):
    """Validate one cached native teacher and fit actual source decoder parameters."""
    target = validate_teacher_record(teacher_record, expected_teacher_binding)
    result = build_case_readouts(model, cold_capture["batch"], cold_capture["base"],
        cold_capture["records"], frame_ids, target, cold_capture["source_prediction"],
        width, height, dataset, key, source_custody=cold_capture["source_custody"], scope=scope,
        direct_resume=direct_resume, opd_resume=opd_resume,
        direct_checkpoint_callback=direct_checkpoint_callback,
        opd_checkpoint_callback=opd_checkpoint_callback)
    if result["source_custody"] != cold_capture["source_custody"] or result["GT_read"] is not False:
        raise AssertionError("Student worker changed source custody or GT state")
    from methods.decota_final_simplified_v1.tensors import state_hash
    if state_hash(model.state_dict()) != cold_capture["source_custody"]["source_model_state_sha256"]:
        raise AssertionError("Full student readout did not restore the cold source model")
    return result


def produce_and_fit_case(model, frames, row, *, expected_source: Mapping[str, Any],
                         feature_binding: Mapping[str, Any], teacher_record: Mapping[str, Any],
                         expected_teacher_binding: Mapping[str, Any], width: int, height: int,
                         dataset: str, key: str, scope="joint", direct_resume=None,
                         opd_resume=None, direct_checkpoint_callback=None,
                         opd_checkpoint_callback=None):
    """Convenience root-worker API: cold source recapture, validated fit, no factory lookup."""
    cold = capture_cold_source_features(model, frames, row,
        expected_source=expected_source, feature_binding=feature_binding)
    result = fit_teacher_student_case(model, cold, row["frame_ids"], teacher_record,
        expected_teacher_binding=expected_teacher_binding, width=width, height=height,
        dataset=dataset, key=key, scope=scope, direct_resume=direct_resume,
        opd_resume=opd_resume, direct_checkpoint_callback=direct_checkpoint_callback,
        opd_checkpoint_callback=opd_checkpoint_callback)
    result["costs"]["cold_source_capture_seconds"] = cold["source_capture_seconds"]
    return {"schema": "ta-stvg-source-teacher-student-worker-v1",
            "feature_cache": cold["feature_cache"], "case": result,
            "source_custody": cold["source_custody"], "GT_read": False}


def _synthetic_teacher_record(binding: Mapping[str, Any], width=32, height=24):
    frame_map = binding["frame_map"]
    physical = [row["physical_frame_id"] for row in frame_map]
    return {"schema": TEACHER_SCHEMA,
        "source": {"name": "synthetic-test-only", "revision": "0"},
        "input_binding": copy.deepcopy(dict(binding)),
        "raw_output": "<synthetic-only>",
        "parsed": {"status": "valid", "candidate_id": 1, "start_time_s": 0.0,
                   "end_time_s": max(0.25, len(frame_map) / binding["local_fps"])},
        "projected_interval": {"start_local_frame": 0, "end_local_frame": len(frame_map) - 1,
            "local_frame_indices": list(range(len(frame_map))), "physical_frame_ids": physical},
        "mask_tube": {"coordinate_convention": "xyxy_inclusive_pixels", "frames":[
            {"local_frame_index": i, "physical_frame_id": row["physical_frame_id"],
             "source_timestamp_s": row["source_timestamp_s"], "candidate_id": 1,
             "present": i != 2,
             "xyxy_inclusive_pixels": [i + 1, 2, i + 4, 7] if i != 2 else None,
             "mask_area_px": 8 if i != 2 else 0}
            for i, row in enumerate(frame_map)],
            "coverage": {"present_frame_count": len(frame_map) - 1,
                         "interval_frame_count": len(frame_map),
                         "fraction": (len(frame_map) - 1) / len(frame_map)}},
        "costs": {"dettrack_s": 0.0, "filter_marker_s": 0.0, "model_preprocess_s": 0.0,
                  "model_generate_s": 0.0, "total_s": 0.0}}


def _contract_kl_identity():
    import numpy as np

    p = np.asarray([.11, .24, .29, .36], dtype=np.float64)
    reward = np.asarray([-.3, .2, .5, .8], dtype=np.float64)
    tau = .37
    logp = np.log(p)
    logq = logp + reward / tau
    logq -= np.logaddexp.reduce(logq)
    q = np.exp(logq)
    left = float(np.dot(q - p, reward))
    right = tau * float(np.sum((q - p) * (logq - logp)))
    assert np.isfinite(left) and np.isfinite(right) and abs(left - right) < 1e-12
    return {"status": "pass", "finite": True, "absolute_error": abs(left - right),
            "source": "synthetic finite categorical Gibbs identity"}


def _contract_opd_grad_and_detach():
    import torch
    from vg_tta.decota_spatial_opd_tunable_v1 import likelihood_loss, rollout

    torch.manual_seed(41)
    mean = torch.tensor([[.2, -.3, .1, .4], [-.2, .1, .3, -.4]], requires_grad=True)
    evidence = torch.tensor([[.58, .46, .18, .27], [.43, .61, .22, .30]])
    policy = rollout(mean.detach(), evidence, "synthetic-cpu", 0,
                     config={"sigma": .1, "tau": .25, "samples": 32,
                             "steps": 1, "lr": .01, "writeback": .0625})
    assert not evidence.requires_grad and not policy["samples"].requires_grad
    assert not policy["weights"].requires_grad and not policy["mean"].requires_grad
    loss = likelihood_loss(mean, policy["mean"], policy["samples"], policy["weights"], sigma=.1)
    grad = torch.autograd.grad(loss, mean)[0]
    expected = ((mean.detach() - policy["mean"]) / .1**2
                - ((policy["weights"] - 1/32)[:, :, None]
                   * (policy["samples"] - mean.detach()[:, None, :])).sum(1) / .1**2) / len(mean)
    err = float((grad - expected).abs().max())
    assert torch.isfinite(loss) and torch.isfinite(grad).all() and err < 2e-5
    assert not evidence.requires_grad
    return {"status": "pass", "gradient_max_abs_error": err,
            "teacher_target_detached": True, "action_detached": True, "softmax_weight_detached": True,
            "source": "synthetic CPU original Gaussian likelihood"}


def _contract_optimizer_noop():
    import torch
    from vg_tta.decota_spatial_opd_tunable_v1 import optimizer_step

    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.Adam([parameter], lr=.03)
    warm = (parameter.square()).sum()
    warm.backward()
    optimizer.step()
    parameter.grad = None
    before_parameter = parameter.detach().clone()
    before_state = _copy_cpu(optimizer.state_dict())
    result = optimizer_step(optimizer, (parameter.square()).sum(), [parameter], informative=False)
    after_state = _copy_cpu(optimizer.state_dict())
    assert result is None and torch.equal(parameter.detach(), before_parameter)
    assert _tree_equal(before_state, after_state)
    return {"status": "pass", "parameters_unchanged": True,
            "optimizer_state_unchanged": True, "existing_moments_preserved": True,
            "source": "synthetic Adam no-information update"}


def _contract_nonempty_fits_and_resume():
    import torch
    import vg_tta.decota_spatial_opd_tunable_v1 as opd_math

    frame_ids = list(range(8))
    base = _synthetic_student_base(len(frame_ids))
    anchors = [{"position": 2, "box": torch.tensor([.82, .22, .12, .16]),
                "weight": 1.0, "physical_frame_id": frame_ids[2], "mask_area_px": 8}]
    config = {"lr": .01, "sigma": .025, "tau": .05, "steps": 3,
              "writeback": .0625, "samples": 32}

    class SyntheticPause(Exception):
        pass

    checkpoints = []

    def stop_after_first_complete_step(payload):
        checkpoints.append(_copy_cpu(payload))
        if payload["next_step"] == 1:
            raise SyntheticPause()

    try:
        _fit_direct(base, base.initial, anchors, frame_ids, "synthetic-resume", config,
                    checkpoint_callback=stop_after_first_complete_step)
    except SyntheticPause:
        pass
    else:
        raise AssertionError("Synthetic callback did not stop at a complete optimizer boundary")
    assert len(checkpoints) == 1 and checkpoints[0]["next_step"] == 1
    resumed = _fit_direct(base, base.initial, anchors, frame_ids, "synthetic-resume", config,
                          resume_state=checkpoints[0])
    uninterrupted = _fit_direct(base, base.initial, anchors, frame_ids, "synthetic-resume", config)
    assert all(torch.equal(resumed["final_state"][name], uninterrupted["final_state"][name])
               for name in resumed["final_state"])
    assert resumed["selected_step"] == config["steps"] and len(resumed["path"]) == config["steps"]
    assert resumed["optimizer_steps"] == config["steps"]
    assert all(torch.isfinite(step["expected_delta"]).all() and torch.isfinite(step["actual_delta"]).all()
               for step in resumed["path"] if not step["skipped"])
    assert resumed["path"][0]["observed_movement"]["actual_l2"] > 0
    assert resumed["path"][0]["unobserved_movement"]["actual_l2"] >= 0

    fitted_opd = _fit_spatial_opd(base, base.initial, anchors, frame_ids, "synthetic-opd", config)
    assert fitted_opd["optimizer_steps"] > 0 and fitted_opd["selected_step"] == config["steps"]
    assert all(not step["rollout"]["samples"].requires_grad and
               not step["rollout"]["weights"].requires_grad for step in fitted_opd["path"]
               if not step["skipped"])
    normal_rollout = opd_math.rollout

    def equal_reward_rollout(mean, evidence, key, round_index, shuffle=False, *, config=None):
        sample = mean.detach()[:, None, :].expand(-1, 32, -1).clone()
        rewards = mean.new_ones((len(mean), 32))
        weights = mean.new_full((len(mean), 32), 1 / 32)
        return {"mean": mean.detach().clone(), "samples": sample, "rewards": rewards,
                "used_rewards": rewards, "weights": weights, "permutations": None,
                "informative": False, "seed": 0}

    opd_math.rollout = equal_reward_rollout
    try:
        skipped_opd = _fit_spatial_opd(base, base.initial, anchors, frame_ids,
                                       "synthetic-equal-reward", config)
    finally:
        opd_math.rollout = normal_rollout
    assert skipped_opd["optimizer_steps"] == 0
    assert all(torch.equal(skipped_opd["initial_state"][name], skipped_opd["final_state"][name])
               for name in skipped_opd["initial_state"])
    assert all(not step["optimizer_state"].get("state") for step in skipped_opd["path"])
    return {"status": "pass", "direct_actual_student_update": True,
            "spatial_opd_actual_student_update": True, "observed_and_unobserved_movement_recorded": True,
            "resumed_direct_matches_uninterrupted_last_step": True,
            "equal_reward_opd_true_noop": True, "equal_reward_optimizer_state_unchanged": True,
            "source": "synthetic CPU fake native decoder; no model weights or video"}


def _contract_teacher_mapping():
    frame_map = [{"local_frame_index": i, "physical_frame_id": 100 + 2 * i,
                  "source_timestamp_s": i / 4} for i in range(6)]
    clock = {"local_fps": 4.0, "local_frame_count": 6, "frame_map": frame_map}
    query_text = "synthetic no-GT query"
    binding = {"source_video_sha256": "1" * 64,
        "canonical_video_sha256": "4" * 64, "canonical_pixel_sha256": "5" * 64,
        "marker_video_sha256": "6" * 64, "query_text": query_text,
        "query_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
        "frame_clock_sha256": hashlib.sha256(_canonical_bytes(clock)).hexdigest(),
        "local_fps": 4.0, "local_frame_count": 6,
        "frame_map": frame_map}
    record = _synthetic_teacher_record(binding)
    target = validate_teacher_record(record, binding)
    observations = select_teacher_observations([100, 102, 104, 106, 108, 110], target, k=4)
    assert observations["selected_positions"] == [1, 2, 3, 5]
    assert 2 in observations["unknown_positions"] and observations["supervised_positions"] == [1, 3, 5]
    sparse = select_teacher_observations([100, 104, 110], target, k=4)
    assert sparse["selected_positions"] == [0, 1, 2]
    assert sparse["unknown_positions"] == [1] and sparse["supervised_positions"] == [0, 2]
    box = inclusive_xyxy_to_cxcywh([1, 2, 4, 7], 32, 24)
    assert torch_allclose_list(box.tolist(), [.09375, .2083333284, .125, .25], atol=1e-7)
    # Query/frame mismatch and GT-shaped keys must reject the record before use.
    wrong = copy.deepcopy(binding)
    wrong["query_sha256"] = "4" * 64
    try:
        validate_teacher_record(record, wrong)
    except ValueError:
        mismatch_rejected = True
    else:
        mismatch_rejected = False
    assert mismatch_rejected
    invalid = copy.deepcopy(record)
    invalid["parsed"] = {"status": "invalid_format", "candidate_id": None,
                         "start_time_s": None, "end_time_s": None}
    invalid["projected_interval"] = {"start_local_frame": None, "end_local_frame": None,
                                     "local_frame_indices": [], "physical_frame_ids": []}
    invalid["mask_tube"]["frames"] = []
    invalid["mask_tube"]["coverage"] = {"present_frame_count": 0, "interval_frame_count": 0,
                                          "fraction": 0.0}
    invalid_target = validate_teacher_record(invalid, binding)
    missing = select_teacher_observations([100, 104, 110], invalid_target, k=4)
    assert invalid_target["valid"] is False and invalid_target["physical_interval"] is None
    assert missing["teacher_unavailable"] and missing["selected_positions"] == []
    injected = copy.deepcopy(record)
    injected["mask_tube"]["frames"][0]["ground_truth_box"] = [0, 0, 1, 1]
    try:
        validate_teacher_record(injected, binding)
    except ValueError:
        gt_key_rejected = True
    else:
        gt_key_rejected = False
    assert gt_key_rejected
    return {"status": "pass", "missing_track_positions_remain_unknown": True,
            "fixed_uniform4_selected_before_missing_mask_filter": True,
            "dense_teacher_clock_maps_to_sparse_student_grid": True,
            "inclusive_pixel_conversion_checked": True,
            "query_binding_mismatch_rejected": mismatch_rejected,
            "invalid_teacher_preserved_without_interval_or_fake_boxes": True,
            "GT_shaped_field_rejected": gt_key_rejected,
            "source": "synthetic-only native teacher cache schema"}


def torch_allclose_list(actual, expected, atol):
    import torch

    return torch.allclose(torch.tensor(actual), torch.tensor(expected), atol=atol, rtol=0)


class _SyntheticReplayModule:
    """Unused placeholder to make accidental runtime imports conspicuous."""


def _synthetic_student_base(n=8):
    import torch
    from methods.decota_final_simplified_v1.tensors import ParameterState

    class Norms(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(256)
            self.norm3 = torch.nn.LayerNorm(256)
            self.norm4 = torch.nn.LayerNorm(256)

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Norms() for _ in range(6)])
            basis = torch.linspace(-1, 1, 256)
            projection = torch.stack((basis, -basis, basis.square() - basis.square().mean(),
                                      torch.sin(3 * math.pi * basis))) * .001
            self.register_buffer("projection", projection)

        def forward(self, **kwargs):
            q = kwargs["query_tgt"]
            layer = self.layers[5]
            h = (layer.norm1(q) + layer.norm3(q) + layer.norm4(q)) / 3
            value = torch.einsum("btf,of->bto", h, self.projection).sigmoid()
            return value.unsqueeze(0).expand(6, *value.shape)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = Decoder()

    class Base(ParameterState):
        def __init__(self, n):
            self.model = Model()
            self.decoder = self.model
            self.delta = torch.zeros(256, requires_grad=True)
            self.named = [("spatial.query_residual", self.delta)]
            last = self.decoder.decoder.layers[5]
            for ln in ("norm1", "norm3", "norm4"):
                for name, parameter in getattr(last, ln).named_parameters():
                    parameter.requires_grad_(True)
                    self.named.append((f"spatial.layers.5.{ln}.{name}", parameter))
            self.spatial_inputs = []
            ids = list(range(n))
            for offset in (0, 1):
                frames = ids[offset::2]
                q = torch.linspace(-.2, .2, len(frames) * 256).reshape(1, len(frames), 256)
                self.spatial_inputs.append({"query_tgt": q})
            self.zero = {"boxes": torch.full((n, 4), .5),
                         "logits": [torch.zeros(1, len(ids[0::2]), 2), torch.zeros(1, len(ids[1::2]), 2)],
                         "actions": [torch.zeros(1, len(ids[0::2]), 1), torch.zeros(1, len(ids[1::2]), 1)]}
            self.initial = self.state()

    return Base(n)


def _contract_zero_adaptation_schema():
    import torch

    ids = list(range(8))
    base = _synthetic_student_base(len(ids))
    empty_direct = _fit_direct(base, base.initial, [], ids, "zero-direct",
        {"lr": .01, "sigma": .025, "tau": .05, "steps": 3,
         "writeback": .0625, "samples": 32}, scope="joint")
    empty_opd = _fit_spatial_opd(base, base.initial, [], ids, "zero-opd",
        {"lr": .01, "sigma": .025, "tau": .05, "steps": 3,
         "writeback": .0625, "samples": 32}, scope="joint")
    for value in (empty_direct, empty_opd):
        assert value["schema"] == "ta-stvg-student-fit-v1"
        assert value["selected_step"] == 3 and value["optimizer_steps"] == 0
        assert all(not step.get("optimizer_state", {}).get("state") for step in value["path"])
        assert all(torch.equal(value["final_state"][k], value["initial_state"][k])
                   for k in value["initial_state"])
        assert torch.equal(value["final_cached_decoder_boxes"], value["path"][0]["boxes"])
    return {"status": "pass", "direct_zero_adaptation_noop": True,
            "opd_zero_adaptation_noop": True, "optimizer_state_empty": True,
            "student_native_decoder_schema_preserved": True,
            "DINO_state_inherited": False, "GT_read": False,
            "source": "synthetic CPU fake 1792-parameter decoder; not model qualification"}


def run_cpu_contracts() -> dict[str, Any]:
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU contracts must not initialize CUDA")
    start = time.time()
    result = {"status": "pass", "scope": "synthetic CPU contracts only",
        "contracts": {"finite_KL_reweighting_identity": _contract_kl_identity(),
                      "Gaussian_gradient_and_detached_teacher": _contract_opd_grad_and_detach(),
                      "equal_reward_optimizer_true_noop": _contract_optimizer_noop(),
                      "nonempty_fits_movement_and_resume": _contract_nonempty_fits_and_resume(),
                      "teacher_cache_binding_unknown_and_K4": _contract_teacher_mapping(),
                      "zero_adaptation_original_decoder_schema": _contract_zero_adaptation_schema()},
        "model_loaded": False, "GPU_initialized": False, "video_pixels_read": False,
        "GT_read": False, "teacher_called": False, "student_called": False,
        "real_weights_read": False, "time": start}
    result["time_completed"] = time.time()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("contracts",), help="Only non-model CPU mode is enabled in this revision")
    args = parser.parse_args(argv)
    result = run_cpu_contracts()
    CONTRACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONTRACTS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(CONTRACTS_PATH)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
