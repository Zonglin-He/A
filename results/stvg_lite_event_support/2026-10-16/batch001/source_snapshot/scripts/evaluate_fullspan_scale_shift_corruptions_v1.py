#!/usr/bin/env python3
"""Fixed HC->VidSTG shift-corruption evaluation for the Fullspan+scale arms.

This runner is intentionally a new evaluation consumer.  It does not import
or copy any shifted historical rows from the HC corruption experiment.  The
target roster and literal media hashes come from the locked Vid64 replication
roster; the clean evaluation episodes are used only as lineage/roster
provenance.  ``prepare`` and ``analyze`` are CPU-only.  ``run`` requires an
explicit CUDA device and the shared research GPU lock.

The prediction path is label-free.  Raw frames and frame IDs are reconstructed
from VidSTG video metadata, shifted in pixel space, normalized, and forwarded.
Only after the six legal predictions, private-head replays, no-op controls,
native reinsertion, and source-model reset have passed are annotations opened
to construct metric records.  Tent/MEMO/SAR are declared pending rather than
silently imported from a clean or differently sampled experiment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch


PROTOCOL_PATH = ROOT / "protocols/fullspan_scale_shift_corruptions_v1.json"
REPLICATION_ROOT = ROOT / "artifacts/spatial_scale_replication_v1"
PARENT_ROOT = ROOT / "artifacts/fullspan_scale_optuna_v2_evaluation"
BASELINE_ROOT = ROOT / "artifacts/baseline_expansion_v1"
SELECTION_ROOT = ROOT / "artifacts/fullspan_scale_optuna_v2"
SOURCE_ROOT = ROOT / "artifacts/spatial_affine_v1"
SOURCE_PLAN = ROOT / "data/spatial_affine_source_v1/plan.json"
GPU_LOCK = ROOT / "artifacts/spatial_tta_research_v2/gpu.lock"
FFMPEG_BIN = ROOT / ".conda/tubedetr/bin/ffmpeg"
FFMPEG_SHA256 = "bfec0e82c8497b0b979b1a7812ba1101da3646e8a3330bcb5537badb840a527e"
DEFAULT_OUT = ROOT / "artifacts/fullspan_scale_shift_corruptions_v1"

CONDITIONS = ("blur3", "low_light3", "subsample2")
METHODS = (
    "frozen",
    "temporal_only_selected",
    "spatial_only_selected",
    "fullspan_scale_selected",
    "direct_fullspan_native_boxes",
    "direct_fullspan_adapted_boxes",
)
SELECTED_KEYS = (
    "spatial_lr",
    "spatial_steps",
    "spatial_gamma",
    "spatial_rank",
    "temporal_lr",
    "temporal_steps",
    "temporal_gamma",
)
METRICS = ("vIoU_corrected", "sIoU", "tIoU", "vIoU_legacy")
EXPECTED_INDICES = (
    0, 5, 10, 19, 20, 23, 24, 26, 28, 44, 55, 60, 67, 74, 80, 91,
    93, 102, 104, 106, 108, 112, 114, 120, 123, 124, 125, 128, 131,
    135, 138, 141, 143, 144, 146, 148, 156, 161, 162, 163, 167, 173,
    174, 177, 180, 182, 185, 192, 193, 194, 195, 199, 201, 203, 204,
    207, 210, 211, 212, 214, 216, 221, 222, 223,
)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value: {value!r}")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported JSON value {type(value)!r}")


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_torch(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _raw_digest(value: np.ndarray) -> str:
    """Hash dtype, shape, and contiguous RGB bytes (the project contract)."""
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(value.shape).encode("utf-8"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _object_digest(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite")
    return result


def _protocol() -> dict[str, Any]:
    protocol = read(PROTOCOL_PATH)
    if protocol.get("schema_version") != "fullspan_scale_shift_corruptions_v1":
        raise RuntimeError("unexpected shift-corruption protocol schema")
    names = tuple(item.get("name") for item in protocol.get("conditions", ()))
    if names != CONDITIONS:
        raise RuntimeError("shift condition roster changed")
    if tuple(protocol.get("methods", ())) != METHODS:
        raise RuntimeError("shift method roster changed")
    target = protocol.get("target", {})
    if target.get("queries") != 64 or target.get("source_clusters") != 64:
        raise RuntimeError("protocol target roster count changed")
    if target.get("one_query_per_source") is not True:
        raise RuntimeError("protocol no longer requires one query per source")
    settings = protocol.get("sampling", {})
    for key, expected in {
        "official_fps": 5,
        "max_frames": 200,
        "model_stride": 2,
        "resolution": 224,
    }.items():
        if settings.get(key) != expected:
            raise RuntimeError(f"sampling setting changed: {key}")
    if protocol.get("external_baselines", {}).get("status") != "pending_safe_port":
        raise RuntimeError("external baseline status must remain explicitly pending")
    return protocol


def _validate_selected(path: Path = SELECTION_ROOT / "hc_to_vid" / "selected.json") -> tuple[dict[str, Any], Path]:
    if not path.is_file():
        raise FileNotFoundError(f"selected HC->Vid configuration is missing: {path}")
    value = read(path)
    if value.get("status") != "complete" or value.get("development_only") is not True:
        raise RuntimeError("HC->Vid selected configuration is not a complete development result")
    if value.get("test_used_for_tuning") is not False:
        raise RuntimeError("HC->Vid selected configuration is not test-free")
    tuning_lock = SELECTION_ROOT / "lock.json"
    if value.get("lock_sha256") != sha(tuning_lock):
        raise RuntimeError("HC->Vid selected configuration is stale")
    raw = value.get("selected_config")
    if not isinstance(raw, dict) or set(raw) != set(SELECTED_KEYS):
        raise RuntimeError(f"selected config must contain exactly {SELECTED_KEYS}")
    config: dict[str, Any] = {}
    for key in SELECTED_KEYS:
        if key.endswith("_steps") or key.endswith("_rank"):
            item = raw[key]
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise RuntimeError(f"invalid positive integer {key}: {item!r}")
            config[key] = int(item)
        else:
            item = _finite_number(raw[key], key)
            if item <= 0:
                raise RuntimeError(f"invalid positive value {key}: {item!r}")
            config[key] = item
    expected = {
        "spatial_lr": 0.004656726597154835,
        "spatial_steps": 1,
        "spatial_gamma": 0.0001,
        "spatial_rank": 32,
        "temporal_lr": 0.0009779330846621942,
        "temporal_steps": 5,
        "temporal_gamma": 0.19717467107590567,
    }
    if config != expected:
        raise RuntimeError(f"selected HC->Vid parameters differ from pinned config: {config!r}")
    if value.get("source_state_exact") is not True:
        raise RuntimeError("selected configuration lacks source_state_exact evidence")
    return config, path


def _resolve_literal_video(spec: Mapping[str, Any], row: Mapping[str, Any]) -> Path:
    raw = Path(str(row["video_path"]))
    candidates = [raw] if raw.is_absolute() else []
    candidates.extend((Path(str(spec["root"])) / "video" / raw, Path(str(spec["root"])) / raw))
    for candidate in candidates:
        if candidate.is_file():
            # Do not resolve symlinks: lock keys are literal media identities.
            return candidate
    raise FileNotFoundError(f"cannot resolve target video {row.get('video_path')!r}")


def _target_metadata(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    required = (
        "video_id", "original_video_id", "video_path", "caption", "fps", "width", "height",
        "start_frame", "end_frame", "qtype",
    )
    for key in required:
        if key not in row:
            raise ValueError(f"Vid metadata index {index} is missing {key}")
    result = {key: row[key] for key in required}
    # This object is deliberately GT-free: event bounds, target IDs, and
    # trajectory/box data are not copied into the pre-prediction lock.
    result["index"] = int(index)
    return result


def _load_target_roster(replication_lock: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = replication_lock.get("datasets", {}).get("vid")
    if not isinstance(spec, dict):
        raise RuntimeError("replication lock has no VidSTG target dataset")
    indices = tuple(int(item) for item in spec.get("indices", ()))
    if indices != EXPECTED_INDICES or len(set(indices)) != 64:
        raise RuntimeError("Vid target indices do not match the locked 64-query roster")
    sources = spec.get("sources", {})
    if set(sources) != {str(i) for i in indices}:
        raise RuntimeError("Vid source map does not cover exactly the locked indices")
    source_map = {str(i): str(sources[str(i)]) for i in indices}
    if len(set(source_map.values())) != 64:
        raise RuntimeError("Vid roster is not one query per distinct source")
    annotation = Path(spec["annotation"])
    if sha(annotation) != spec.get("annotation_sha256"):
        raise RuntimeError("Vid annotation hash changed")
    raw_annotation = read(annotation)
    rows = raw_annotation.get("videos")
    if not isinstance(rows, list) or max(indices) >= len(rows):
        raise RuntimeError("Vid annotation does not contain the locked indices")
    metadata = {str(i): _target_metadata(rows[i], i) for i in indices}
    for i in indices:
        path = _resolve_literal_video(spec, metadata[str(i)])
        expected = spec.get("video_sha256", {}).get(str(path))
        if expected is None:
            raise RuntimeError(f"literal Vid media path is absent from lock: {path}")
        if sha(path) != expected:
            raise RuntimeError(f"Vid media hash changed: {path}")
    target = {
        "kind": spec.get("kind"),
        "root": str(spec["root"]),
        "annotation": str(annotation),
        "annotation_sha256": spec["annotation_sha256"],
        "indices": list(indices),
        "sources": source_map,
        "video_sha256": dict(spec.get("video_sha256", {})),
        "query_metadata": metadata,
        "source_stat_overlap_excluded": spec.get("excluded_source_stat_overlap", []),
    }
    if target["kind"] != "vidstg":
        raise RuntimeError("target dataset is not VidSTG")
    return target, raw_annotation


def _validate_parent_lineage(replication_sha: str, selected: Mapping[str, Any], expected_sources: Mapping[str, str]) -> dict[str, str]:
    lock_path = PARENT_ROOT / "lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"clean parent evaluation lock is missing: {lock_path}")
    parent_lock = read(lock_path)
    if parent_lock.get("replication_lock_sha256") != replication_sha:
        raise RuntimeError("clean parent does not point at the current Vid64 replication lock")
    if parent_lock.get("evaluation_queries_used_for_tuning") is not False:
        raise RuntimeError("clean parent evaluation is not marked tuning-free")
    parent_config = parent_lock.get("selected_configs", {}).get("hc_to_vid")
    if parent_config != dict(selected):
        raise RuntimeError("clean parent HC->Vid config differs from selected config")
    selected_path = SELECTION_ROOT / "hc_to_vid" / "selected.json"
    if parent_lock.get("selected_sha256", {}).get("hc_to_vid") != sha(selected_path):
        raise RuntimeError("clean parent selected-config fingerprint differs from current selection")
    episodes = PARENT_ROOT / "hc_to_vid" / "episodes"
    result: dict[str, str] = {}
    for index in EXPECTED_INDICES:
        path = episodes / f"{index:06d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"clean parent episode missing: {path}")
        episode = read(path)
        if episode.get("index") != index or episode.get("lock_sha256") != sha(lock_path):
            raise RuntimeError(f"clean parent episode identity/lock mismatch: {path}")
        if episode.get("source") != expected_sources[str(index)]:
            raise RuntimeError(f"clean parent source roster mismatch: {path}")
        if episode.get("adaptation_GT_used") is not False or episode.get("selection_performed") is not False:
            raise RuntimeError(f"clean parent episode has invalid GT/selection flags: {path}")
        if len(episode.get("records", [])) != 20:
            raise RuntimeError(f"clean parent episode method roster changed: {path}")
        result[str(index)] = sha(path)
    return result


def _bytes_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _validate_ffmpeg() -> None:
    if not FFMPEG_BIN.is_file() or sha(FFMPEG_BIN) != FFMPEG_SHA256:
        raise RuntimeError(f"pinned TubeDETR ffmpeg binary is missing or changed: {FFMPEG_BIN}")


def _validate_clean_references(target: Mapping[str, Any], annotation: Mapping[str, Any]) -> dict[str, Any]:
    """Cross-check the target-free decoder against the canonical clean input.

    The baseline episode is consulted only for media/frame/label provenance;
    no historical metric value or prediction is copied into the shift study.
    This CPU preparation check catches a decoder/frame-grid drift before any
    CUDA model invocation.
    """
    baseline_lock_path = BASELINE_ROOT / "lock.json"
    baseline_lock = read(baseline_lock_path)
    baseline_sha = sha(baseline_lock_path)
    baseline_spec = baseline_lock.get("datasets", {}).get("vid", {})
    if [int(x) for x in baseline_spec.get("indices", [])] != list(EXPECTED_INDICES):
        raise RuntimeError("canonical clean baseline Vid roster differs from target roster")
    if {str(k): str(v) for k, v in baseline_spec.get("sources", {}).items()} != dict(target["sources"]):
        raise RuntimeError("canonical clean baseline source map differs from target roster")
    if baseline_lock.get("cells", {}).get("hc_to_vid_clean", {}).get("condition") != "clean":
        raise RuntimeError("canonical clean reference cell is not the declared clean control")
    references: dict[str, Any] = {}
    rows = annotation.get("videos")
    if not isinstance(rows, list):
        raise RuntimeError("Vid annotation has no video rows for clean QA")
    for index in EXPECTED_INDICES:
        metadata = target["query_metadata"][str(index)]
        path = _resolve_literal_video(target, metadata)
        raw, frame_ids = _decode_raw_metadata_only(metadata, path)
        episode_path = BASELINE_ROOT / "hc_to_vid_clean" / "episodes" / f"{index:06d}.json"
        if not episode_path.is_file():
            raise FileNotFoundError(f"canonical clean episode missing: {episode_path}")
        episode = read(episode_path)
        if episode.get("lock_sha256") != baseline_sha or episode.get("index") != index or episode.get("source") != target["sources"][str(index)]:
            raise RuntimeError(f"canonical clean episode identity changed: {episode_path}")
        if episode.get("gt_used") is not False:
            raise RuntimeError(f"canonical clean episode has an invalid GT flag: {episode_path}")
        if episode.get("raw_pixel_sha256") != _bytes_digest(raw):
            raise RuntimeError(f"metadata-only raw decoder differs from canonical clean pixels: {index}")
        if int(episode.get("evaluation_frames", -1)) != len(raw) or int(episode.get("sampled_input_frames", -1)) != len(raw):
            raise RuntimeError(f"canonical clean frame count differs: {index}")
        records = episode.get("records", [])
        frozen = next((item for item in records if item.get("method") == "frozen"), None)
        replay = frozen.get("metric_replay") if isinstance(frozen, dict) else None
        if not isinstance(replay, dict):
            raise RuntimeError(f"canonical clean frozen metadata replay is missing: {index}")
        if replay.get("frame_ids") != frame_ids:
            raise RuntimeError(f"canonical clean frame grid differs: {index}")
        row = rows[index]
        # Label QA is independent preparation provenance.  It is never passed
        # to the legal fit; the run phase reconstructs it after prediction.
        targets, video_target = _build_labels(annotation, row, raw, frame_ids)
        actual_target_boxes = [target_item.get("boxes", torch.empty(0, 4)).detach().float().cpu().tolist() for target_item in targets]
        if replay.get("target_boxes") != actual_target_boxes:
            raise RuntimeError(f"canonical clean labels differ from official project transform: {index}")
        if replay.get("gt_indices") != video_target["inter_idx"]:
            raise RuntimeError(f"canonical clean GT frame indices differ: {index}")
        if replay.get("gt_frame_interval") != [int(row["tube_start_frame"]), int(row["tube_end_frame"])]:
            raise RuntimeError(f"canonical clean GT interval differs: {index}")
        references[str(index)] = {
            "canonical_episode_sha256": sha(episode_path),
            "raw_bytes_sha256": _bytes_digest(raw),
            "frame_count": len(raw),
            "frame_ids_sha256": _object_digest(frame_ids),
            "labels_sha256": _object_digest({"target_boxes": actual_target_boxes, "gt_indices": video_target["inter_idx"], "gt_frame_interval": replay["gt_frame_interval"]}),
        }
    return {
        "baseline_lock": str(baseline_lock_path),
        "baseline_lock_sha256": baseline_sha,
        "cell": "hc_to_vid_clean",
        "queries": len(references),
        "references": references,
        "metric_values_copied": False,
        "purpose": "CPU decoder/frame-grid/label provenance QA only",
    }


def _validate_source(replication_lock: Mapping[str, Any], target_sources: Iterable[str] = ()) -> dict[str, Any]:
    complete_path = SOURCE_ROOT / "source_complete.json"
    stats_path = SOURCE_ROOT / "source_statistics.pt"
    head_path = SOURCE_ROOT / "source_bbox_head.pt"
    if sha(complete_path) != replication_lock.get("source_complete_sha256"):
        raise RuntimeError("spatial source-complete receipt changed")
    complete = read(complete_path)
    if not SOURCE_PLAN.is_file() or sha(SOURCE_PLAN) != complete.get("source_plan_sha256"):
        raise RuntimeError("spatial source plan is missing or changed")
    source_plan = read(SOURCE_PLAN)
    source_rows = source_plan.get("rows")
    source_ids = sorted({str(item.get("source")) for item in source_rows} if isinstance(source_rows, list) else set())
    if len(source_ids) != 64 or int(source_plan.get("source_count", -1)) != 64:
        raise RuntimeError("spatial source plan does not contain exactly 64 source clips")
    if source_plan.get("source_labels_used") is not False or source_plan.get("target_inputs_used_for_statistics") is not False:
        raise RuntimeError("spatial source plan is not label/target-free")
    target_overlap = sorted(set(source_ids) & {str(item) for item in target_sources})
    if target_overlap:
        raise RuntimeError(f"spatial source statistics overlap target source IDs: {target_overlap}")
    if sha(stats_path) != complete.get("stats_sha256") or sha(head_path) != complete.get("head_sha256"):
        raise RuntimeError("spatial source statistics/head hash changed")
    stats = load_torch(stats_path)
    if not isinstance(stats, dict) or stats.get("labels_used") is not False:
        raise RuntimeError("source statistics are not explicitly label-free")
    if int(stats.get("feature_dim", 0)) != 256 or int(stats.get("videos", 0)) != 64:
        raise RuntimeError("source statistics dimensionality/source count changed")
    return {
        "complete_path": str(complete_path),
        "complete_sha256": sha(complete_path),
        "stats_path": str(stats_path),
        "stats_sha256": sha(stats_path),
        "head_path": str(head_path),
        "head_sha256": sha(head_path),
        "source_plan_path": str(SOURCE_PLAN),
        "source_plan_sha256": sha(SOURCE_PLAN),
        "source_ids": source_ids,
        "target_source_overlap": target_overlap,
        "labels_used": False,
        "videos": int(stats["videos"]),
        "feature_dim": int(stats["feature_dim"]),
    }


def _validate_clean_reference_manifest(reference: Mapping[str, Any]) -> None:
    """Validate the already-produced CPU QA manifest without decoding/labels."""
    baseline_lock_path = BASELINE_ROOT / "lock.json"
    if reference.get("baseline_lock") != str(baseline_lock_path) or reference.get("baseline_lock_sha256") != sha(baseline_lock_path):
        raise RuntimeError("canonical clean-reference baseline lock changed")
    refs = reference.get("references")
    if reference.get("queries") != 64 or not isinstance(refs, dict) or set(refs) != {str(i) for i in EXPECTED_INDICES}:
        raise RuntimeError("canonical clean-reference manifest is incomplete")
    for index in EXPECTED_INDICES:
        item = refs[str(index)]
        if not isinstance(item, dict) or not all(key in item for key in ("canonical_episode_sha256", "raw_bytes_sha256", "frame_count", "frame_ids_sha256", "labels_sha256")):
            raise RuntimeError(f"canonical clean-reference entry is malformed: {index}")
        path = BASELINE_ROOT / "hc_to_vid_clean" / "episodes" / f"{index:06d}.json"
        if sha(path) != item["canonical_episode_sha256"]:
            raise RuntimeError(f"canonical clean-reference episode changed: {index}")


def _build_lock(out: Path, *, run_clean_qa: bool = True) -> dict[str, Any]:
    protocol = _protocol()
    selected, selected_path = _validate_selected()
    replication_path = REPLICATION_ROOT / "lock.json"
    replication = read(replication_path)
    replication_sha = sha(replication_path)
    target, raw_annotation = _load_target_roster(replication)
    parent_episode_sha = _validate_parent_lineage(replication_sha, selected, target["sources"])
    if run_clean_qa:
        clean_reference = _validate_clean_references(target, raw_annotation)
    else:
        existing_path = Path(out) / "lock.json"
        if not existing_path.is_file():
            raise RuntimeError("runtime lock is missing its prepared clean-reference manifest")
        existing = read(existing_path)
        clean_reference = existing.get("clean_reference")
        if not isinstance(clean_reference, dict):
            raise RuntimeError("prepared lock has no clean-reference manifest")
        _validate_clean_reference_manifest(clean_reference)
    source = _validate_source(replication, target["sources"].values())
    _validate_ffmpeg()
    checkpoint = Path(replication["checkpoint"])
    if sha(checkpoint) != replication.get("checkpoint_sha256"):
        raise RuntimeError("TubeDETR checkpoint hash changed")
    dependencies = [
        Path(__file__),
        PROTOCOL_PATH,
        ROOT / "scripts/evaluate_fullspan_scale_corruptions_v2.py",
        ROOT / "scripts/run_feasibility.py",
        ROOT / "vg_tta/baseline_expansion_data.py",
        ROOT / "vg_tta/geometric_video_io.py",
        ROOT / "vg_tta/fullspan_tta.py",
        ROOT / "vg_tta/spatial_affine.py",
        ROOT / "vg_tta/spatial_affine_variance.py",
        ROOT / "vg_tta/metrics.py",
        ROOT / "vg_tta/tubedetr_runtime.py",
        ROOT / "external/TubeDETR/datasets/vidstg.py",
        ROOT / "external/TubeDETR/datasets/video_transforms.py",
    ]
    if any(not item.is_file() for item in dependencies):
        raise FileNotFoundError("one or more pinned runner dependencies are missing")
    lock = {
        "status": "prepared",
        "schema_version": "fullspan_scale_shift_corruptions_v1",
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": sha(PROTOCOL_PATH),
        "lineage": {
            "replication_lock": str(replication_path),
            "replication_lock_sha256": replication_sha,
            "clean_parent_lock": str(PARENT_ROOT / "lock.json"),
            "clean_parent_lock_sha256": sha(PARENT_ROOT / "lock.json"),
            "clean_parent_episode_sha256": parent_episode_sha,
            "clean_parent_metrics_copied": False,
            "historical_exposure": "module-held-out shift evaluation; not globally untouched",
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": replication["checkpoint_sha256"],
        "target": target,
        "clean_reference": clean_reference,
        "source_statistics": source,
        "selected_config": selected,
        "selected_path": str(selected_path),
        "selected_sha256": sha(selected_path),
        "conditions": list(CONDITIONS),
        "methods": list(METHODS),
        "external_baselines": {
            "status": "pending_safe_port",
            "methods": ["tent", "memo", "sar"],
            "implemented": [],
            "counts": 0,
            "reason": "No shifted historical rows or unverified external port is imported.",
        },
        "sampling": {
            "official_fps": 5,
            "max_frames": 200,
            "model_stride": 2,
            "resolution": 224,
            "raw_decode": "VidSTG metadata-only clip start/end and fps filter",
            "timestamp_mapping": "original frame IDs; subsample positions retained; boxes lifted by timestamps",
        },
        "legal_boundary": {
            "fit_gt_used": False,
            "selection_gt_used": False,
            "labels_loaded_after_prediction_barrier": True,
            "prepare_metadata_and_label_qa": True,
            "prepare_label_qa_not_fit_or_selection": True,
            "labels_in_cache": False,
            "no_query_drops": True,
        },
        "statistics": {"bootstrap_seed": 20260907, "bootstrap_draws": 10000, "unit": "Vid source"},
        "code_sha256": {str(item): sha(item) for item in dependencies},
        "runtime": {
            "prepare_device": "cpu",
            "analyze_device": "cpu",
            "run_device": "cuda:0",
            "global_gpu_lock": str(GPU_LOCK),
            "ffmpeg_path": str(FFMPEG_BIN),
            "ffmpeg_sha256": FFMPEG_SHA256,
            "cache_not_free": True,
        },
    }
    return lock


def initialize(out: Path = DEFAULT_OUT) -> dict[str, Any]:
    """Create or validate a complete CPU manifest without touching CUDA."""
    out = Path(out).resolve()
    lock_path = out / "lock.json"
    lock = _build_lock(out, run_clean_qa=not lock_path.exists())
    path = lock_path
    if path.exists():
        existing = read(path)
        if existing != lock:
            raise RuntimeError(f"prepared lock differs; use a new output directory: {path}")
    else:
        atomic_json(path, lock)
    return lock


def verify_lock(out: Path) -> dict[str, Any]:
    out = Path(out).resolve()
    lock_path = out / "lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"missing prepared lock: {lock_path}")
    lock = read(lock_path)
    _validate_clean_reference_manifest(lock.get("clean_reference", {}))
    expected = _build_lock(out, run_clean_qa=False)
    if expected != lock:
        raise RuntimeError("prepared lock no longer matches current source/data/config lineage")
    for path, digest in lock["code_sha256"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"pinned dependency changed: {path}")
    return lock


def _sample_frame_ids(row: Mapping[str, Any], *, fps: int = 5, max_frames: int = 200) -> list[int]:
    """Copy the official VidSTG metadata-only sampling rule, without GT."""
    video_fps = float(row["fps"])
    if not math.isfinite(video_fps) or video_fps <= 0 or fps <= 0:
        raise ValueError("invalid video/sample fps")
    sampling_rate = fps / video_fps
    if sampling_rate > 1:
        raise ValueError("upsampling is not allowed by the official VidSTG loader")
    start = int(row["start_frame"])
    end = int(row["end_frame"])
    if start < 0 or end <= start:
        raise ValueError("invalid metadata clip range")
    frame_ids = [start]
    for frame_id in range(start, end):
        if int(frame_ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
            frame_ids.append(frame_id)
    if len(frame_ids) > max_frames:
        frame_ids = [(frame_ids[(j * len(frame_ids)) // max_frames]) for j in range(max_frames)]
    if not frame_ids or frame_ids != sorted(set(frame_ids)):
        raise RuntimeError("metadata sampling produced a non-increasing grid")
    return frame_ids


def _decode_raw_metadata_only(row: Mapping[str, Any], video_path: Path) -> tuple[np.ndarray, list[int]]:
    """Decode exactly the metadata clip; no trajectory/box data is accessed."""
    import ffmpeg

    frame_ids = _sample_frame_ids(row)
    width, height = int(row["width"]), int(row["height"])
    video_fps = float(row["fps"])
    clip_start, clip_end = int(row["start_frame"]), int(row["end_frame"])
    duration = (clip_end - clip_start) / video_fps
    if width <= 0 or height <= 0 or duration <= 0:
        raise ValueError("invalid metadata dimensions/duration")
    _validate_ffmpeg()
    output, _ = (
        ffmpeg.input(str(video_path), ss=clip_start / video_fps, t=duration)
        .filter("fps", fps=len(frame_ids) / duration)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True, quiet=True, cmd=str(FFMPEG_BIN))
    )
    expected = len(frame_ids) * width * height * 3
    if len(output) != expected:
        raise RuntimeError(f"raw decoder returned {len(output)} bytes; expected {expected}")
    raw = np.frombuffer(output, dtype=np.uint8).reshape(len(frame_ids), height, width, 3).copy()
    if raw.shape[0] != len(frame_ids):
        raise RuntimeError("raw decode/frame-grid mismatch")
    return raw, frame_ids


def shift_pixels(raw: np.ndarray, condition: str) -> tuple[np.ndarray, list[int]]:
    """Use the audited project pixel transform lazily (CPU-testable)."""
    from vg_tta.baseline_expansion_data import shift_pixels as _shift

    return _shift(raw, condition)


def lift_prediction(boxes: torch.Tensor, logits: torch.Tensor, positions: list[int], frame_ids: list[int]):
    from vg_tta.baseline_expansion_data import lift_prediction as _lift

    return _lift({"pred_boxes": boxes, "pred_sted": logits}, positions, frame_ids)


def _metric_record(outputs, targets, video_target, annotation, *, index, source, condition, method, runtime, peak):
    from scripts.run_feasibility import prediction_record

    record = prediction_record(
        outputs,
        targets,
        video_target,
        annotation,
        sample_index=int(index),
        condition=condition,
        method=method,
        runtime_sec=float(runtime),
        peak_vram_gb=float(peak),
    )
    record.update(
        source=str(source),
        gt_used=False,
        adaptation_gt_used=False,
        labels_used_for="evaluation_only_after_all_legal_predictions",
    )
    return record


def _snapshot(model: torch.nn.Module) -> dict[str, Any]:
    return {
        "parameters": {name: p.detach().cpu().clone() for name, p in model.named_parameters()},
        "buffers": {name: b.detach().cpu().clone() for name, b in model.named_buffers()},
        "versions": {name: int(p._version) for name, p in model.named_parameters()},
        "requires_grad": {name: bool(p.requires_grad) for name, p in model.named_parameters()},
        "modes": {name: module.training for name, module in model.named_modules()},
        "sted_id": id(model.sted_embed),
        "bbox_id": id(model.bbox_embed),
    }


def _assert_source_unchanged(model: torch.nn.Module, snapshot: Mapping[str, Any]) -> None:
    if {name: module.training for name, module in model.named_modules()} != snapshot["modes"]:
        raise AssertionError("source model training modes changed")
    if id(model.sted_embed) != snapshot["sted_id"] or id(model.bbox_embed) != snapshot["bbox_id"]:
        raise AssertionError("native source heads were not restored")
    for name, parameter in model.named_parameters():
        if int(parameter._version) != snapshot["versions"][name] or not torch.equal(parameter.detach().cpu(), snapshot["parameters"][name]):
            raise AssertionError(f"source model parameter changed: {name}")
        if bool(parameter.requires_grad) != snapshot["requires_grad"][name]:
            raise AssertionError(f"source model requires_grad flag changed: {name}")
        if parameter.grad is not None:
            raise AssertionError(f"source model received a gradient: {name}")
    for name, buffer in model.named_buffers():
        if not torch.equal(buffer.detach().cpu(), snapshot["buffers"][name]):
            raise AssertionError(f"source model buffer changed: {name}")


def _lazy_runtime():
    from scripts import evaluate_fullspan_scale_corruptions_v2 as base

    return base


def _decode_capture(model, memory, duration: int, caption: str, device: str):
    base = _lazy_runtime()
    return base._decode_capture(model, memory, duration, caption, device)


def _native_reinsert(model, memory, temporal_head, spatial_head, duration: int, caption: str, device: str):
    base = _lazy_runtime()
    return base._native_reinsert(model, memory, temporal_head, spatial_head, duration, caption, device)


def _timed(device: str, function):
    base = _lazy_runtime()
    return base._timed(device, function)


def _fit_predictions(model, source_head, source_stats, raw, frame_ids, caption, condition, config, *, device, first_for_condition, model_snapshot):
    """Produce all six predictions and audits without accessing annotation GT."""
    base = _lazy_runtime()
    shifted, positions = shift_pixels(raw, condition)
    if shifted.dtype != np.uint8 or shifted.ndim != 4 or shifted.shape[-1] != 3:
        raise RuntimeError("pixel shift did not preserve uint8 THWC")
    video = base.normalize_raw_view(shifted, resolution=224)
    if int(video.shape[1]) != len(positions):
        raise RuntimeError("shifted pixel count and retained positions disagree")
    device_obj = torch.device(device)
    memory, encoder_seconds = _timed(device, lambda: base.encode_video(
        model, video, caption, repo=ROOT / "external/TubeDETR", stride=2, device=device, use_bf16=True
    ))
    (native, hidden), decoder_seconds = _timed(device, lambda: _decode_capture(
        model, memory, len(positions), caption, device
    ))
    native_boxes = native["pred_boxes"].detach()
    native_logits = native["pred_sted"].detach()
    if not torch.isfinite(native_boxes).all() or not torch.isfinite(native_logits).all():
        raise FloatingPointError("native shifted prediction is non-finite")
    source_head_device = source_head.to(device_obj).eval().requires_grad_(False)
    spatial_head = base.AffineBoxHead(source_head_device)
    z, feature_seconds = _timed(device, lambda: base.cached_features(spatial_head, hidden.flatten(1, 2)))
    with torch.no_grad():
        identity_boxes = base.boxes_from_features(spatial_head, z)
    if not torch.equal(identity_boxes, native_boxes):
        raise AssertionError("identity spatial replay is not native-exact")
    cache_payload = {
        "index": None,
        "source": None,
        "condition": condition,
        "caption": caption,
        "frame_ids": list(frame_ids),
        "positions": list(positions),
        "raw_pixel_dtype": str(raw.dtype),
        "raw_pixel_shape": list(raw.shape),
        "raw_pixel_sha256": _raw_digest(raw),
        "shifted_pixel_dtype": str(shifted.dtype),
        "shifted_pixel_shape": list(shifted.shape),
        "shifted_pixel_sha256": _raw_digest(shifted),
        "head_input": hidden.detach().cpu(),
        "spatial_features": z.detach().cpu(),
        "frozen": {"pred_boxes": native_boxes.detach().cpu(), "pred_sted": native_logits.detach().cpu()},
        "labels_used": False,
        "gt_used": False,
    }
    spatial_config = base.AffineConfig(
        "subspace", int(config["spatial_rank"]), float(config["spatial_lr"]),
        int(config["spatial_steps"]), float(config["spatial_gamma"]),
    )
    spatial_audit, spatial_seconds = _timed(device, lambda: base.fit_variance_affine(
        spatial_head, z, source_stats, spatial_config
    ))
    if spatial_audit.get("nonfinite_fallback"):
        raise RuntimeError("selected spatial fit used a nonfinite fallback")
    with torch.no_grad():
        adapted_boxes, spatial_replay_seconds = _timed(device, lambda: base.boxes_from_features(spatial_head, z).detach())
    spatial_seconds += spatial_replay_seconds
    temporal_head = copy.deepcopy(model.sted_embed).eval()
    temporal_inputs = [{"head_input": hidden}] * int(config["temporal_steps"])
    temporal_audit, temporal_seconds = _timed(device, lambda: base.fit_fullspan_head(
        temporal_head, temporal_inputs, lr=float(config["temporal_lr"]),
        anchor_gamma=float(config["temporal_gamma"]), optimizer_eps=1e-4,
    ))
    if not temporal_audit.get("audit", {}).get("all_steps_finite", False):
        raise RuntimeError("selected temporal fit produced a non-finite step")
    with torch.no_grad():
        adapted_logits, temporal_replay_seconds = _timed(device, lambda: base.replay_temporal_head(
            temporal_head, hidden, device
        ).detach())
    temporal_seconds += temporal_replay_seconds
    full_logits = base._fullspan_logits(native_logits)
    native_reinsert = {"checked": False, "prediction_exact": None, "boxes_exact": None, "seconds": 0.0}
    if first_for_condition:
        native_adapted, native_seconds = _native_reinsert(
            model, memory, temporal_head, spatial_head, len(positions), caption, device
        )
        logits_exact = bool(torch.equal(native_adapted["pred_sted"], adapted_logits))
        boxes_exact = bool(torch.equal(native_adapted["pred_boxes"], adapted_boxes))
        native_reinsert = {
            "checked": True,
            "prediction_exact": logits_exact,
            "boxes_exact": boxes_exact,
            "seconds": float(native_seconds),
            "max_abs_logit_diff": float((native_adapted["pred_sted"].float() - adapted_logits.float()).abs().max()),
            "max_abs_box_diff": float((native_adapted["pred_boxes"].float() - adapted_boxes.float()).abs().max()),
        }
        if not logits_exact or not boxes_exact:
            raise AssertionError(f"joint native reinsertion mismatch: {native_reinsert}")
    noop = {"checked": False}
    if first_for_condition:
        noop = {"checked": True, "identity": True, "full_path_controls": {}}
        with torch.no_grad():
            identity_temporal = base.replay_temporal_head(copy.deepcopy(model.sted_embed).eval(), hidden, device)
        if not torch.equal(identity_temporal, native_logits):
            raise AssertionError("identity temporal replay is not native-exact")
        for name, zero_lr, zero_steps in (("lr0", True, False), ("steps0", False, True)):
            control_spatial = base.AffineBoxHead(source_head_device)
            control_config = base.AffineConfig(
                "subspace", int(config["spatial_rank"]),
                0.0 if zero_lr else float(config["spatial_lr"]),
                0 if zero_steps else int(config["spatial_steps"]),
                float(config["spatial_gamma"]),
            )
            control_spatial_audit = base.fit_variance_affine(control_spatial, z, source_stats, control_config)
            control_temporal = copy.deepcopy(model.sted_embed).eval()
            control_temporal_audit = base.fit_fullspan_head(
                control_temporal,
                [{"head_input": hidden}] * (0 if zero_steps else int(config["temporal_steps"])),
                lr=0.0 if zero_lr else float(config["temporal_lr"]),
                anchor_gamma=float(config["temporal_gamma"]), optimizer_eps=1e-4,
            )
            with torch.no_grad():
                control_boxes = base.boxes_from_features(control_spatial, z)
                control_logits = base.replay_temporal_head(control_temporal, hidden, device)
            spatial_exact = bool(torch.equal(control_boxes, native_boxes))
            temporal_exact = bool(torch.equal(control_logits, native_logits))
            if not spatial_exact or not temporal_exact:
                raise AssertionError(f"{name} full-path control is not exact")
            noop["full_path_controls"][name] = {
                "spatial_exact": spatial_exact,
                "temporal_exact": temporal_exact,
                "spatial_audit": control_spatial_audit,
                "temporal_audit": control_temporal_audit,
            }
    outputs = {
        "frozen": (native_boxes, native_logits),
        "temporal_only_selected": (native_boxes, adapted_logits),
        "spatial_only_selected": (adapted_boxes, native_logits),
        "fullspan_scale_selected": (adapted_boxes, adapted_logits),
        "direct_fullspan_native_boxes": (native_boxes, full_logits),
        "direct_fullspan_adapted_boxes": (adapted_boxes, full_logits),
    }
    lifted = {
        name: lift_prediction(boxes, logits, positions, frame_ids)
        for name, (boxes, logits) in outputs.items()
    }
    _assert_source_unchanged(model, model_snapshot)
    cache_payload["index"] = None
    cache_payload["source"] = None
    return {
        "lifted": lifted,
        "cache": cache_payload,
        "positions": positions,
        "raw": raw,
        "shifted": shifted,
        "native": {"pred_boxes": native_boxes, "pred_sted": native_logits},
        "audits": {
            "spatial": spatial_audit,
            "temporal": temporal_audit,
            "native_reinsertion": native_reinsert,
            "no_op_controls": noop,
            "identity_bbox_exact": True,
            "source_model_state_exact": True,
            "private_heads_fresh": True,
            "gt_used_for_adaptation": False,
            "labels_used_for_selection": False,
        },
        "timing": {
            "encoder_seconds": float(encoder_seconds),
            "frozen_decoder_seconds": float(decoder_seconds),
            "feature_cache_seconds": float(feature_seconds),
            "spatial_fit_seconds": float(spatial_seconds),
            "temporal_fit_seconds": float(temporal_seconds),
            "native_reinsertion_seconds": float(native_reinsert["seconds"]),
            "cache_not_free": True,
        },
    }


def _build_labels(annotation: Mapping[str, Any], row: Mapping[str, Any], raw: np.ndarray, frame_ids: list[int]):
    """Build official normalized labels only after the prediction barrier."""
    repo = ROOT / "external/TubeDETR"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from datasets.video_transforms import make_video_transforms, prepare

    width, height = int(row["width"]), int(row["height"])
    trajectories = annotation.get("trajectories", {})
    video_key = str(row["original_video_id"])
    target_key = str(row["target_id"])
    if video_key not in trajectories or target_key not in trajectories[video_key]:
        raise RuntimeError("Vid GT trajectory is missing after prediction barrier")
    trajectory = trajectories[video_key][target_key]
    targets = []
    gt_indices = []
    for index, frame_id in enumerate(frame_ids):
        inside = int(row["tube_start_frame"]) <= int(frame_id) < int(row["tube_end_frame"])
        if inside:
            key = str(frame_id)
            if key not in trajectory:
                raise RuntimeError(f"Vid GT trajectory is sparse at frame {frame_id}")
            targets.append(prepare(width, height, [trajectory[key]]))
            gt_indices.append(index)
        else:
            targets.append(prepare(width, height, []))
    _, normalized_targets = make_video_transforms("val", cautious=True, resolution=224)(raw, targets)
    if len(normalized_targets) != len(frame_ids) or not gt_indices:
        raise RuntimeError("Vid GT labels are empty or frame-misaligned")
    for target in normalized_targets:
        boxes = target.get("boxes")
        if boxes is None or not torch.isfinite(boxes).all():
            raise FloatingPointError("Vid GT contains non-finite boxes")
        if boxes.numel() and (boxes.min() < -1e-6 or boxes.max() > 1 + 1e-6):
            raise ValueError("Vid GT normalized boxes are outside [0,1]")
        if boxes.numel():
            boxes.clamp_(0, 1)
    video_target = {
        "video_id": row["video_id"],
        "caption": row["caption"],
        "frames_id": list(frame_ids),
        "inter_idx": [gt_indices[0], gt_indices[-1]],
    }
    return normalized_targets, video_target


def _assert_cache_without_labels(cache: Mapping[str, Any]) -> None:
    forbidden = {"targets", "annotation", "trajectory", "video_target", "gt_boxes", "gt_interval", "labels"}
    if forbidden & set(cache):
        raise AssertionError(f"legal cache contains forbidden label fields: {forbidden & set(cache)}")
    if cache.get("gt_used") is not False or cache.get("labels_used") is not False:
        raise AssertionError("legal cache is marked as label-used")


def _write_failure(out: Path, condition: str, index: int, stage: str, exc: BaseException, *, gt_read: bool) -> None:
    atomic_json(out / condition / "failures" / f"{index:06d}.json", {
        "status": "failed",
        "condition": condition,
        "index": int(index),
        "stage": stage,
        "gt_read_before_failure": bool(gt_read),
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
        "lock_sha256": sha(out / "lock.json"),
    })


def _episode_path(out: Path, condition: str, index: int) -> Path:
    return out / condition / "episodes" / f"{int(index):06d}.json"


def _cache_path(out: Path, condition: str, index: int) -> Path:
    return out / condition / "cache" / f"{int(index):06d}.pt"


def _labels_path(out: Path, condition: str, index: int) -> Path:
    return out / condition / "labels_only" / f"{int(index):06d}.pt"


def _verify_episode(out: Path, lock: Mapping[str, Any], episode: Mapping[str, Any]) -> None:
    condition, index = str(episode.get("condition")), int(episode.get("index"))
    lock_sha = sha(out / "lock.json")
    if episode.get("status") != "complete" or episode.get("lock_sha256") != lock_sha:
        raise RuntimeError(f"episode is incomplete/unlocked: {condition}/{index}")
    if condition not in CONDITIONS or index not in EXPECTED_INDICES:
        raise RuntimeError("episode identity is outside the locked roster")
    source = lock["target"]["sources"][str(index)]
    if episode.get("source") != source or episode.get("selected_config") != lock["selected_config"]:
        raise RuntimeError(f"episode source/config mismatch: {condition}/{index}")
    if episode.get("gt_used") is not False or episode.get("audits", {}).get("gt_used_for_adaptation") is not False:
        raise RuntimeError(f"episode GT boundary changed: {condition}/{index}")
    records = episode.get("records", [])
    if [record.get("method") for record in records] != list(METHODS):
        raise RuntimeError(f"episode method roster changed: {condition}/{index}")
    for record in records:
        if record.get("condition") != condition or record.get("sample_index") != index or record.get("source") != source:
            raise RuntimeError("episode record identity mismatch")
        if record.get("gt_used") is not False or record.get("adaptation_gt_used") is not False:
            raise RuntimeError("episode record GT boundary changed")
    cache = _cache_path(out, condition, index)
    labels = _labels_path(out, condition, index)
    if not cache.is_file() or not labels.is_file():
        raise RuntimeError(f"episode cache/label artifact missing: {condition}/{index}")
    if sha(cache) != episode.get("cache_sha256") or sha(labels) != episode.get("labels_sha256"):
        raise RuntimeError(f"episode cache/label hash mismatch: {condition}/{index}")
    _assert_cache_without_labels(load_torch(cache))


def _source_macro_delta(records: list[dict[str, Any]], reference: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    if len(records) != len(reference):
        raise AssertionError("source-macro rows are not aligned")
    for row, base in zip(records, reference):
        if (row.get("source"), row.get("sample_index")) != (base.get("source"), base.get("sample_index")):
            raise AssertionError("source-macro rows are not query aligned")
        grouped[str(row["source"])].append(100.0 * (float(row[metric]) - float(base[metric])))
    values = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarize empty source set")
    rng = np.random.default_rng(20260907)
    draws = values[rng.integers(values.size, size=(10000, values.size))].mean(axis=1)
    return {
        "source_count": int(values.size),
        "mean_pp": float(values.mean()),
        "median_pp": float(np.median(values)),
        "ci95_pp": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "win_neutral_loss_sources": [
            int((values > 0.1).sum()), int((np.abs(values) <= 0.1).sum()), int((values < -0.1).sum())
        ],
        "seed": 20260907,
        "bootstrap_draws": 10000,
    }


def _orphan_artifacts(out: Path, condition: str, indices: Iterable[int]) -> list[dict[str, Any]]:
    """Find partial/foreign files which must not be silently resumed."""
    expected = {f"{int(index):06d}" for index in indices}
    found: list[dict[str, Any]] = []
    for dirname, suffix in (("episodes", ".json"), ("cache", ".pt"), ("labels_only", ".pt"), ("failures", ".json")):
        directory = out / condition / dirname
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            stem = path.name[: -len(suffix)] if path.name.endswith(suffix) else ""
            if stem not in expected or not path.name.endswith(suffix):
                found.append({"condition": condition, "directory": dirname, "path": str(path), "status": "orphan"})
    return found


def analyze(out: Path = DEFAULT_OUT) -> dict[str, Any]:
    """CPU-only metric replay and source-macro summary."""
    out = Path(out).resolve()
    lock = verify_lock(out)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    failures = []
    complete_counts = {}
    orphan_artifacts: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        complete = 0
        for index in lock["target"]["indices"]:
            failure = out / condition / "failures" / f"{int(index):06d}.json"
            if failure.is_file():
                failures.append(read(failure))
            episode_path = _episode_path(out, condition, int(index))
            if not episode_path.is_file():
                continue
            episode = read(episode_path)
            _verify_episode(out, lock, episode)
            for record in episode["records"]:
                from scripts.evaluate_fullspan_scale_corruptions_v2 import _assert_metric_replay

                _assert_metric_replay(record)
                grouped[(condition, record["method"])].append(record)
            complete += 1
        complete_counts[condition] = complete
        orphan_artifacts.extend(_orphan_artifacts(out, condition, lock["target"]["indices"]))
    if orphan_artifacts:
        failures.extend(orphan_artifacts)
    cells = {}
    for condition in CONDITIONS:
        rows = {method: grouped[(condition, method)] for method in METHODS}
        count = complete_counts[condition]
        if count == 0:
            cells[condition] = {"status": "not_started", "planned_queries": 64, "completed_queries": 0}
            continue
        if any(len(rows[method]) != count for method in METHODS):
            raise RuntimeError(f"method rows are incomplete in {condition}")
        summaries = {}
        base = rows["frozen"]
        for method in METHODS:
            values = rows[method]
            summaries[method] = {
                "count": len(values),
                "mean": {metric: float(np.mean([float(item[metric]) for item in values])) for metric in METRICS},
                "delta_vs_frozen_source_macro": {
                    metric: _source_macro_delta(values, base, metric) for metric in METRICS
                },
            }
        cells[condition] = {
            "status": "complete" if count == 64 else "partial",
            "planned_queries": 64,
            "completed_queries": count,
            "source_count": len({row["source"] for row in base}),
            "methods": summaries,
            # Compare the joint arm with the same temporal adaptation and
            # native boxes.  A Frozen-only comparison would conflate the
            # temporal gain with the spatial increment.
            "spatial_increment_vs_temporal_only": {
                metric: _source_macro_delta(
                    rows["fullspan_scale_selected"], rows["temporal_only_selected"], metric
                )
                for metric in ("vIoU_corrected", "sIoU", "tIoU")
            },
        }
    result = {
        "status": "failed" if failures else ("complete" if all(item["status"] == "complete" for item in cells.values()) else "partial"),
        "experiment": str(out),
        "lock_sha256": sha(out / "lock.json"),
        "conditions": cells,
        "failures": failures,
        "orphan_artifacts": orphan_artifacts,
        "gt_isolation": "No GT in adaptation/selection; labels and metric replay occur after each query prediction barrier.",
        "external_baselines": lock["external_baselines"],
        "historical_policy": "Clean parent episodes are provenance only; no clean or shifted historical records are copied.",
        "module_held_out_not_globally_untouched": True,
        "analyzer_sha256": sha(Path(__file__)),
    }
    atomic_json(out / "analysis.json", result)
    return result


def prepare(out: Path = DEFAULT_OUT) -> dict[str, Any]:
    lock = initialize(out)
    result = {
        "status": "prepared",
        "lock_sha256": sha(Path(out) / "lock.json"),
        "planned_conditions": list(CONDITIONS),
        "planned_queries_per_condition": 64,
        "planned_legal_records": 64 * len(CONDITIONS) * len(METHODS),
        "external_baselines": lock["external_baselines"],
        "gpu_started": False,
        "annotation_json_parsed_for_prepare_qa": True,
        "prepare_label_qa": True,
        "labels_used_for_fit_or_selection": False,
    }
    atomic_json(Path(out) / "prepare.json", result)
    return result


def _run_one(out: Path, lock: Mapping[str, Any], model, source_head, source_stats, condition: str, index: int, *, device: str, first_for_condition: bool, snapshot: Mapping[str, Any], stage_state: dict[str, Any] | None = None) -> None:
    base = _lazy_runtime()
    metadata = lock["target"]["query_metadata"][str(index)]
    if stage_state is not None:
        stage_state["stage"] = "metadata_decode"
    source = lock["target"]["sources"][str(index)]
    video_path = _resolve_literal_video(lock["target"], metadata)
    expected = lock["target"]["video_sha256"].get(str(video_path))
    if expected is None or sha(video_path) != expected:
        raise RuntimeError(f"target media hash changed: {video_path}")
    raw, frame_ids = _decode_raw_metadata_only(metadata, video_path)
    # Recheck the canonical clean bytes and original-frame grid for every
    # runtime query before any model forward or legal fit.  The full decoder /
    # label QA is prepared once on CPU; this per-query check prevents a changed
    # or symlink-swapped media file from silently entering the shift cell.
    clean_reference = lock["clean_reference"]["references"].get(str(index))
    if not isinstance(clean_reference, Mapping):
        raise RuntimeError(f"canonical clean reference is missing: {index}")
    if _bytes_digest(raw) != clean_reference.get("raw_bytes_sha256"):
        raise RuntimeError(f"runtime raw bytes differ from canonical clean input: {index}")
    if len(raw) != int(clean_reference.get("frame_count", -1)):
        raise RuntimeError(f"runtime frame count differs from canonical clean input: {index}")
    if _object_digest(frame_ids) != clean_reference.get("frame_ids_sha256"):
        raise RuntimeError(f"runtime frame grid differs from canonical clean input: {index}")
    prediction = _fit_predictions(
        model, source_head, source_stats, raw, frame_ids, str(metadata["caption"]), condition,
        lock["selected_config"], device=device, first_for_condition=first_for_condition,
        model_snapshot=snapshot,
    )
    if stage_state is not None:
        stage_state["stage"] = "legal_predictions_complete"
    prediction["cache"]["index"] = index
    prediction["cache"]["source"] = source
    prediction["cache"]["lock_sha256"] = sha(out / "lock.json")
    cache_path = _cache_path(out, condition, index)
    if cache_path.exists():
        raise RuntimeError(f"orphaned cache exists before episode: {cache_path}")
    atomic_torch(cache_path, prediction["cache"])
    cache_sha = sha(cache_path)
    # -------------------- GT barrier: no annotation access above this line. --------------------
    # The target annotation is intentionally opened only after every legal
    # prediction/control has been materialized and the source model restored.
    if stage_state is not None:
        stage_state["stage"] = "labels"
        stage_state["gt_read"] = True
    annotation = read(Path(lock["target"]["annotation"]))
    rows = annotation.get("videos")
    if not isinstance(rows, list) or index >= len(rows):
        raise RuntimeError("Vid annotation rows are missing after prediction barrier")
    row = rows[index]
    if _target_metadata(row, index) != metadata:
        raise RuntimeError("post-barrier annotation metadata differs from the prepared target identity")
    labels = _build_labels(annotation, row, raw, frame_ids)
    targets, video_target = labels
    labels_payload = {
        "index": index,
        "source": source,
        "condition": condition,
        "targets": targets,
        "video_target": video_target,
        "annotation": row,
        "gt_used_for_adaptation": False,
        "labels_used_for": "evaluation_only_after_all_legal_predictions",
    }
    labels_path = _labels_path(out, condition, index)
    if labels_path.exists():
        raise RuntimeError(f"orphaned labels artifact exists before episode: {labels_path}")
    atomic_torch(labels_path, labels_payload)
    labels_sha = sha(labels_path)
    records = []
    shared = prediction["timing"]["encoder_seconds"] + prediction["timing"]["frozen_decoder_seconds"]
    timing = {
        "frozen": {"shared_seconds": shared, "adaptation_seconds": 0.0},
        "temporal_only_selected": {"shared_seconds": shared, "adaptation_seconds": prediction["timing"]["temporal_fit_seconds"]},
        "spatial_only_selected": {"shared_seconds": shared, "adaptation_seconds": prediction["timing"]["feature_cache_seconds"] + prediction["timing"]["spatial_fit_seconds"]},
        "fullspan_scale_selected": {"shared_seconds": shared, "adaptation_seconds": prediction["timing"]["feature_cache_seconds"] + prediction["timing"]["spatial_fit_seconds"] + prediction["timing"]["temporal_fit_seconds"]},
        "direct_fullspan_native_boxes": {"shared_seconds": shared, "adaptation_seconds": 0.0, "control": True},
        "direct_fullspan_adapted_boxes": {"shared_seconds": shared, "adaptation_seconds": prediction["timing"]["feature_cache_seconds"] + prediction["timing"]["spatial_fit_seconds"], "control": True},
    }
    peak = float(torch.cuda.max_memory_allocated(torch.device(device)) / 2**30)
    for method in METHODS:
        record = _metric_record(
            {"pred_boxes": prediction["lifted"][method]["pred_boxes"], "pred_sted": prediction["lifted"][method]["pred_sted"]},
            targets, video_target, row,
            index=index, source=source, condition=condition, method=method,
            runtime=timing[method]["shared_seconds"] + timing[method]["adaptation_seconds"], peak=peak,
        )
        record.update(config=lock["selected_config"], timing=timing[method])
        records.append(record)
    _assert_source_unchanged(model, snapshot)
    episode = {
        "status": "complete",
        "index": index,
        "source": source,
        "condition": condition,
        "caption": str(metadata["caption"]),
        "selected_config": lock["selected_config"],
        "selected_sha256": lock["selected_sha256"],
        "lock_sha256": sha(out / "lock.json"),
        "cache_sha256": cache_sha,
        "labels_sha256": labels_sha,
        "parent_clean_episode_sha256": lock["lineage"]["clean_parent_episode_sha256"][str(index)],
        "raw_pixel_sha256": prediction["cache"]["raw_pixel_sha256"],
        "shifted_pixel_sha256": prediction["cache"]["shifted_pixel_sha256"],
        "frame_ids": frame_ids,
        "positions": prediction["positions"],
        "records": records,
        "adaptation_audits": prediction["audits"],
        "timing": prediction["timing"],
        "audits": {
            **prediction["audits"],
            "labels_used_for_metrics_only_after_prediction": True,
            "dataset_loader_read_annotations_before_prediction": False,
            "source_model_state_exact_after_episode": True,
            "private_heads_reset_per_query": True,
        },
        "gt_used": False,
    }
    atomic_json(_episode_path(out, condition, index), episode)


def run(out: Path = DEFAULT_OUT, *, device: str = "cuda:0", conditions: Iterable[str] = CONDITIONS, limit: int | None = None) -> None:
    out = Path(out).resolve()
    lock = verify_lock(out)
    conditions = tuple(conditions)
    if not set(conditions).issubset(set(CONDITIONS)):
        raise ValueError(f"conditions must be a subset of {CONDITIONS}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("run requires CUDA; use prepare/analyze for CPU-only actions")
    base = _lazy_runtime()
    base.add_repo_to_path(ROOT / "external/TubeDETR")
    model, _ = base.build_model(ROOT / "external/TubeDETR", device=device, resolution=224, stride=2)
    report = base.load_official_checkpoint(model, lock["checkpoint"])
    if report.get("missing_keys") or report.get("unexpected_keys"):
        raise RuntimeError(f"checkpoint incompatibility: {report}")
    model.eval().requires_grad_(False)
    source_head = load_torch(Path(lock["source_statistics"]["head_path"])).to(torch.device(device)).eval().requires_grad_(False)
    source_stats = load_torch(Path(lock["source_statistics"]["stats_path"]))
    snapshot = _snapshot(model)
    for condition in conditions:
        completed = 0
        for index in lock["target"]["indices"]:
            index = int(index)
            if limit is not None and completed >= limit:
                break
            episode_path = _episode_path(out, condition, index)
            failure_path = out / condition / "failures" / f"{index:06d}.json"
            cache_path = _cache_path(out, condition, index)
            labels_path = _labels_path(out, condition, index)
            if failure_path.exists():
                raise RuntimeError(f"refusing to resume failed query: {failure_path}")
            if episode_path.exists():
                _verify_episode(out, lock, read(episode_path))
                completed += 1
                continue
            if cache_path.exists() or labels_path.exists():
                raise RuntimeError(f"orphaned artifact blocks resume: {condition}/{index}")
            stage_state = {"stage": "metadata_decode", "gt_read": False}
            try:
                _run_one(
                    out, lock, model, source_head, source_stats,
                    condition, index, device=device, first_for_condition=(completed == 0), snapshot=snapshot,
                    stage_state=stage_state,
                )
            except Exception as exc:
                _write_failure(out, condition, index, stage_state["stage"], exc, gt_read=bool(stage_state["gt_read"]))
                raise
            completed += 1
            print(f"[shift corruption] {condition} {completed}/64", flush=True)
        _assert_source_unchanged(model, snapshot)
    if all(
        _episode_path(out, condition, int(index)).is_file()
        for condition in CONDITIONS
        for index in lock["target"]["indices"]
    ):
        atomic_json(out / "complete.json", {
            "status": "complete",
            "conditions": list(CONDITIONS),
            "queries": 64 * len(CONDITIONS),
            "records": 64 * len(CONDITIONS) * len(METHODS),
            "lock_sha256": sha(out / "lock.json"),
            "external_baselines": lock["external_baselines"],
        })


class _GpuLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"shared GPU lock is busy: {self.path}") from exc
        self.handle.write(json.dumps({"pid": os.getpid(), "script": str(Path(__file__))}) + "\n")
        self.handle.flush()
        return self

    def __exit__(self, _kind, _value, _traceback):
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260907)
    torch.backends.cudnn.benchmark = False
    out = args.out.resolve()
    if args.action == "prepare":
        print(json.dumps(prepare(out), indent=2, ensure_ascii=False))
    elif args.action == "analyze":
        print(json.dumps(analyze(out), indent=2, ensure_ascii=False))
    else:
        with _GpuLock(GPU_LOCK):
            run(out, device=args.device, conditions=args.conditions, limit=args.limit)


if __name__ == "__main__":
    main()
