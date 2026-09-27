#!/usr/bin/env python3
"""Prepare a deterministic, cross-domain VidSTG support episode.

This script is deliberately independent of the experiment/runtime code.  It
reads the official VidSTG sentence annotations and VidOR validation metadata,
selects one caption for each of N distinct videos, and emits the compact
``test.json`` structure expected by TubeDETR's ``VideoModulatedSTGrounding``.
When the large validation-video ZIP is available, only the selected videos
are extracted into a private data root.  Before that ZIP is available, the
same command still emits a deterministic annotation manifest and records the
video availability as pending.

The default split is VidSTG ``test`` because those annotations point at
VidOR's validation videos, matching the HF ``validation-video.zip`` archive.
The selected data are support-only: no HC-STVG query text or HC-STVG video
stem is used.  No GPU or model forward is performed here; ``--smoke`` only
decodes one selected video through the upstream TubeDETR dataset transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDSTG_ANNOTATIONS = PROJECT_ROOT / "external" / "VidSTG-Dataset" / "annotations"
DEFAULT_VIDOR_ANNOTATION_ZIP = PROJECT_ROOT / "downloads" / "vidor" / "validation-annotation.zip"
DEFAULT_VIDOR_VIDEO_ZIP = PROJECT_ROOT / "downloads" / "vidor" / "validation-video.zip"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "vidstg_wrong_domain_support"
DEFAULT_MANIFEST = PROJECT_ROOT / "artifacts" / "phase3_vidstg_wrong_domain_support" / "manifest.json"

EXPECTED_VIDOR_ANNOTATION_BYTES = 17_212_872
EXPECTED_VIDOR_ANNOTATION_SHA256 = "65a06594c7735d8a0c384144ec2acb2227f715ccac2d03a6dee4a507de201516"
EXPECTED_VIDOR_VIDEO_BYTES = 3_056_763_915
EXPECTED_VIDOR_VIDEO_SHA256 = "40dd102b43ecd2ed70f2dc38ee08258315e25919d2f1fb8f43bb235740a83a25"

REQUIRED_VIDSTG_KEYS = {
    "vid",
    "fps",
    "frame_count",
    "used_segment",
    "width",
    "height",
    "subject/objects",
    "used_relation",
    "temporal_gt",
    "captions",
    "questions",
}
REQUIRED_VIDOR_KEYS = {
    "video_id",
    "video_path",
    "frame_count",
    "fps",
    "width",
    "height",
    "subject/objects",
    "trajectories",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_text(value: str) -> str:
    """Normalize only enough for an exact query-text overlap audit."""

    value = value.casefold().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", "", value, flags=re.UNICODE)
    return value


def float_close(left: Any, right: Any, tolerance: float = 1e-3) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git_commit(repo: Path) -> str | None:
    """Return the checked-out source commit without changing repository state."""

    head = repo / ".git" / "HEAD"
    if not head.exists():
        return None
    try:
        import subprocess

        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def collect_hc_exclusions(project_root: Path) -> dict[str, Any]:
    """Collect HC video stems and query text for a conservative overlap audit."""

    paths: list[Path] = []
    for directory in sorted(project_root.glob("data/hcstvg2*/")):
        for pattern in ("annotations/*.json", "annos/*.json"):
            paths.extend(sorted(directory.glob(pattern)))
    paths = sorted({path.resolve() for path in paths if path.is_file()})

    video_ids: set[str] = set()
    query_texts: set[str] = set()
    source_reports: list[dict[str, Any]] = []

    for path in paths:
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        entries: Iterable[tuple[str | None, dict[str, Any]]] = []
        if isinstance(data, list):
            entries = [
                (None, item)
                for item in data
                if isinstance(item, dict)
            ]
        elif isinstance(data, dict):
            entries = [
                (str(key), value)
                for key, value in data.items()
                if isinstance(value, dict)
            ]

        local_videos = 0
        local_queries = 0
        for key, item in entries:
            candidates = [
                item.get("original_video_id"),
                item.get("video_id"),
                item.get("video_path"),
                key,
            ]
            for candidate in candidates:
                if candidate is None:
                    continue
                text = str(candidate)
                stem = Path(text).stem
                if stem:
                    video_ids.add(stem)
                    local_videos += 1
                    break

            for field in ("caption", "English", "query", "text"):
                query = item.get(field)
                if isinstance(query, str) and query.strip():
                    query_texts.add(normalize_text(query))
                    local_queries += 1
                    break
        source_reports.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "video_entries_seen": local_videos,
                "query_entries_seen": local_queries,
            }
        )

    return {
        "source_files": source_reports,
        "video_stems": video_ids,
        "query_texts_normalized": query_texts,
        "video_count": len(video_ids),
        "query_count": len(query_texts),
    }


def load_vidor_annotations(annotation_zip: Path | None, annotation_root: Path | None) -> dict[str, dict[str, Any]]:
    """Load VidOR records from the official ZIP or an already extracted root."""

    records: dict[str, dict[str, Any]] = {}
    if annotation_zip is not None and annotation_zip.is_file():
        with zipfile.ZipFile(annotation_zip) as archive:
            for member in sorted(archive.namelist()):
                if not member.lower().endswith(".json"):
                    continue
                with archive.open(member) as handle:
                    record = json.load(handle)
                if not isinstance(record, dict) or "video_id" not in record:
                    continue
                records[str(record["video_id"])] = record
    elif annotation_root is not None and annotation_root.is_dir():
        for path in sorted(annotation_root.rglob("*.json")):
            record = load_json(path)
            if isinstance(record, dict) and "video_id" in record:
                records[str(record["video_id"])] = record
    else:
        raise FileNotFoundError(
            "VidOR annotation source not found; supply --vidor-annotation-zip or "
            "--vidor-annotation-root"
        )
    if not records:
        raise RuntimeError("VidOR annotation source contained no video records")
    return records


def trajectory_for_target(raw: dict[str, Any], target_id: int) -> dict[str, dict[str, Any]]:
    """Convert one VidOR target track to TubeDETR's frame-keyed bbox mapping."""

    trajectory: dict[str, dict[str, Any]] = {}
    for frame_id, frame_boxes in enumerate(raw.get("trajectories", [])):
        if not isinstance(frame_boxes, list):
            continue
        for box in frame_boxes:
            if not isinstance(box, dict) or int(box.get("tid", -1)) != int(target_id):
                continue
            bbox = box.get("bbox")
            if not isinstance(bbox, dict):
                continue
            xmin = float(bbox.get("xmin", 0))
            ymin = float(bbox.get("ymin", 0))
            xmax = float(bbox.get("xmax", 0))
            ymax = float(bbox.get("ymax", 0))
            if not (xmax > xmin and ymax > ymin):
                continue
            trajectory[str(frame_id)] = {
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "generated": int(box.get("generated", 0)),
                "tracker": box.get("tracker", "none"),
            }
            break
    return trajectory


def validate_vidstg_annotation(annotation: dict[str, Any]) -> list[str]:
    errors = sorted(REQUIRED_VIDSTG_KEYS - set(annotation))
    for field in ("used_segment", "temporal_gt"):
        if not isinstance(annotation.get(field), dict):
            errors.append(f"{field} is not an object")
    return errors


def validate_vidor_record(raw: dict[str, Any]) -> list[str]:
    return sorted(REQUIRED_VIDOR_KEYS - set(raw))


def candidate_queries(annotation: dict[str, Any], annotation_index: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for kind_order, (field, query_type, qtype) in enumerate(
        (("captions", "caption", "declarative"), ("questions", "question", "interrogative"))
    ):
        queries = annotation.get(field, [])
        if not isinstance(queries, list):
            continue
        for query_index, query in enumerate(queries):
            if not isinstance(query, dict):
                continue
            description = query.get("description")
            if not isinstance(description, str) or not description.strip():
                continue
            try:
                target_id = int(query["target_id"])
            except (KeyError, TypeError, ValueError):
                continue
            candidates.append(
                {
                    "annotation_index": annotation_index,
                    "query_index": query_index,
                    "kind_order": kind_order,
                    "query_type": query_type,
                    "qtype": qtype,
                    "query": description.strip(),
                    "target_id": target_id,
                }
            )
    return candidates


def build_candidate_pool(
    vidstg_annotations: list[dict[str, Any]],
    vidor_records: dict[str, dict[str, Any]],
    *,
    hc_exclusions: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Find one or more valid, self-contained query candidates per VidSTG video."""

    by_video: dict[str, list[dict[str, Any]]] = {}
    counters: Counter[str] = Counter()
    for annotation_index, annotation in enumerate(vidstg_annotations):
        errors = validate_vidstg_annotation(annotation)
        if errors:
            counters["invalid_vidstg_schema"] += 1
            continue
        vid = str(annotation["vid"])
        raw = vidor_records.get(vid)
        if raw is None:
            counters["missing_vidor_record"] += 1
            continue
        raw_errors = validate_vidor_record(raw)
        if raw_errors:
            counters["invalid_vidor_schema"] += 1
            continue
        if int(annotation["frame_count"]) != int(raw["frame_count"]):
            counters["metadata_frame_count_mismatch"] += 1
            continue
        if not float_close(annotation["fps"], raw["fps"]):
            counters["metadata_fps_mismatch"] += 1
            continue
        if int(annotation["width"]) != int(raw["width"]) or int(annotation["height"]) != int(raw["height"]):
            counters["metadata_dimensions_mismatch"] += 1
            continue

        used_segment = annotation["used_segment"]
        temporal_gt = annotation["temporal_gt"]
        try:
            segment_begin = int(used_segment["begin_fid"])
            segment_end = int(used_segment["end_fid"])
            tube_begin = int(temporal_gt["begin_fid"])
            tube_end = int(temporal_gt["end_fid"])
        except (KeyError, TypeError, ValueError):
            counters["invalid_frame_intervals"] += 1
            continue
        if not (0 <= segment_begin < segment_end <= int(raw["frame_count"])):
            counters["invalid_used_segment"] += 1
            continue
        if not (0 <= tube_begin < tube_end <= int(raw["frame_count"])):
            counters["invalid_temporal_gt"] += 1
            continue

        subject_objects = annotation.get("subject/objects", [])
        target_ids = {
            int(item["tid"])
            for item in subject_objects
            if isinstance(item, dict) and "tid" in item
        }
        raw_targets = {
            int(item["tid"])
            for item in raw.get("subject/objects", [])
            if isinstance(item, dict) and "tid" in item
        }
        if not target_ids or not raw_targets:
            counters["missing_target_catalog"] += 1
            continue

        for query in candidate_queries(annotation, annotation_index):
            target_id = int(query["target_id"])
            if target_id not in target_ids or target_id not in raw_targets:
                counters["query_target_missing"] += 1
                continue
            trajectory = trajectory_for_target(raw, target_id)
            interval_frames = [str(frame) for frame in range(tube_begin, tube_end)]
            if any(frame not in trajectory for frame in interval_frames):
                counters["query_target_track_gap"] += 1
                continue
            normalized_query = normalize_text(query["query"])
            if normalized_query in hc_exclusions["query_texts_normalized"]:
                counters["query_text_hc_overlap"] += 1
                continue
            if Path(str(raw["video_path"])).stem in hc_exclusions["video_stems"]:
                counters["video_stem_hc_overlap"] += 1
                continue
            candidate = {
                **query,
                "vid": vid,
                "fps": float(raw["fps"]),
                "frame_count": int(raw["frame_count"]),
                "width": int(raw["width"]),
                "height": int(raw["height"]),
                "video_path": str(raw["video_path"]),
                "segment_begin": segment_begin,
                "segment_end": segment_end,
                "tube_begin": tube_begin,
                "tube_end": tube_end,
                "trajectory": trajectory,
                "vidstg_annotation": annotation,
            }
            by_video.setdefault(vid, []).append(candidate)

    preferred: dict[str, dict[str, Any]] = {}
    for vid, candidates in by_video.items():
        candidates.sort(
            key=lambda item: (
                int(item["kind_order"]),
                int(item["annotation_index"]),
                int(item["query_index"]),
                item["query"],
            )
        )
        preferred[vid] = candidates[0]
    counters["eligible_unique_videos"] = len(preferred)
    return preferred, dict(counters)


def select_candidates(
    candidates: dict[str, dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("--num-videos must be positive")
    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            hashlib.sha256(f"phase3-wrong-domain:{seed}:{item['vid']}".encode("utf-8")).hexdigest(),
            str(item["vid"]),
        ),
    )
    if len(ranked) < count:
        raise RuntimeError(f"requested {count} videos but only {len(ranked)} eligible videos exist")
    return ranked[:count]


def archive_report(path: Path, *, expected_bytes: int, expected_sha256: str, verify: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "expected_size_bytes": expected_bytes,
        "expected_sha256": expected_sha256,
        "actual_size_bytes": path.stat().st_size if path.is_file() else None,
        "actual_sha256": None,
        "status": "missing",
        "member_count": None,
        "crc32_ok": None,
        "top_level_names": [],
    }
    if not path.is_file():
        return report
    if report["actual_size_bytes"] != expected_bytes:
        report["status"] = "size_mismatch"
        return report
    report["actual_sha256"] = sha256_file(path)
    report["status"] = "complete" if report["actual_sha256"] == expected_sha256 else "hash_mismatch"
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            report["member_count"] = len(infos)
            report["top_level_names"] = sorted({name.split("/", 1)[0] for name in archive.namelist() if name})
            if verify:
                bad_member = archive.testzip()
                report["crc32_ok"] = bad_member is None
                if bad_member is not None:
                    report["first_bad_member"] = bad_member
                    report["status"] = "crc_mismatch"
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        report["status"] = "invalid_zip"
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def find_video_member(archive: zipfile.ZipFile, video_path: str) -> str | None:
    expected = f"video/{video_path.lstrip('/')}"
    names = set(archive.namelist())
    if expected in names:
        return expected
    suffix = "/" + video_path.lstrip("/")
    matches = sorted(name for name in names if name.endswith(suffix))
    return matches[0] if len(matches) == 1 else None


def extract_selected_videos(
    archive_path: Path,
    records: list[dict[str, Any]],
    data_root: Path,
) -> dict[str, Any]:
    """Extract only selected members, preserving archive-relative video paths."""

    extracted = 0
    already_present = 0
    missing_members: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        info_by_name = {info.filename: info for info in archive.infolist()}
        for record in records:
            video_path = str(record.get("video_relpath", record["video_path"]))
            member = find_video_member(archive, video_path)
            record["video_archive_member"] = member
            if member is None:
                record["video_available"] = False
                missing_members.append(video_path)
                continue
            destination = data_root / "video" / video_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            record["video_archive_member_size_bytes"] = int(info_by_name[member].file_size)
            if destination.is_file() and destination.stat().st_size == int(info_by_name[member].file_size):
                already_present += 1
            else:
                temporary = destination.with_name(destination.name + ".part")
                with archive.open(info_by_name[member]) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                temporary.replace(destination)
                extracted += 1
            record["video_available"] = True
            record["video_file_size_bytes"] = destination.stat().st_size
            record["video_sha256"] = sha256_file(destination)
    return {
        "extracted_count": extracted,
        "already_present_count": already_present,
        "missing_member_count": len(missing_members),
        "missing_members": missing_members,
    }


def build_tubedetr_annotation(records: list[dict[str, Any]]) -> dict[str, Any]:
    videos: list[dict[str, Any]] = []
    trajectories: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for video_id, record in enumerate(records):
        vid = str(record["vid"])
        target_id = str(record["target_id"])
        videos.append(
            {
                "original_video_id": vid,
                "frame_count": int(record["frame_count"]),
                "fps": float(record["fps"]),
                "width": int(record["width"]),
                "height": int(record["height"]),
                "start_frame": int(record["segment_begin"]),
                "end_frame": int(record["segment_end"]),
                "tube_start_frame": int(record["tube_begin"]),
                "tube_end_frame": int(record["tube_end"]),
                # TubeDETR joins this path to ``data_root/video``.  The
                # manifest additionally carries an absolute ``video_path``
                # for callers; ``video_relpath`` is the on-disk contract.
                "video_path": str(record.get("video_relpath", record["video_path"])),
                "caption": str(record["query"]),
                "type": str(record["query_type"]),
                "target_id": int(record["target_id"]),
                "video_id": video_id,
                "qtype": str(record["qtype"]),
            }
        )
        trajectories.setdefault(vid, {})[target_id] = record["trajectory"]
    return {"videos": videos, "trajectories": trajectories}


def _tube_detr_frame_ids(entry: dict[str, Any], *, fps: int, max_frames: int) -> list[int]:
    """Reproduce ``VideoModulatedSTGrounding``'s deterministic frame sampling."""

    if fps <= 0 or max_frames <= 0:
        raise ValueError("fps and max_frames must be positive")
    video_fps = float(entry["fps"])
    sampling_rate = float(fps) / video_fps
    if sampling_rate > 1:
        raise ValueError(f"requested fps={fps} is above source fps={video_fps}")
    start_frame, end_frame = (int(value) for value in entry["used_segment"])
    if not (0 <= start_frame < end_frame):
        raise ValueError(f"invalid used_segment: {entry['used_segment']}")
    frame_ids = [start_frame]
    for frame_id in range(start_frame, end_frame):
        if int(frame_ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
            frame_ids.append(frame_id)
    if len(frame_ids) > max_frames:
        frame_ids = [frame_ids[(index * len(frame_ids)) // max_frames] for index in range(max_frames)]
    return frame_ids


def load_vidstg_support_video(
    entry: dict[str, Any], resolution: int = 224, max_frames: int = 200
) -> torch.Tensor:
    """Load one support entry as a CPU float ``C x T x H x W`` tensor.

    The decoder, five-FPS temporal sampling, test resize, RGB conversion, and
    ImageNet normalization are the same operations used by TubeDETR's
    ``VideoModulatedSTGrounding`` dataset.  No annotation or model forward is
    needed by this helper, which lets the Phase-3 controller load support
    frames directly from a manifest entry.

    ``entry['video_path']`` must be the absolute path emitted in the manifest;
    ``fps``, ``used_segment``, ``width``, and ``height`` are also required.
    """

    import ffmpeg
    import numpy as np
    import torch

    environment_bin = str(Path(sys.executable).resolve().parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if environment_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join([environment_bin, *path_entries])
    repo = PROJECT_ROOT / "external" / "TubeDETR"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from datasets.video_transforms import make_video_transforms

    video_path = Path(str(entry["video_path"])).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"support video not found: {video_path}")
    frame_ids = _tube_detr_frame_ids(entry, fps=5, max_frames=max_frames)
    source_fps = float(entry["fps"])
    clip_start, clip_end = (int(value) for value in entry["used_segment"])
    duration = (clip_end - clip_start) / source_fps
    if duration <= 0:
        raise ValueError("support clip duration must be positive")
    command = (
        ffmpeg.input(str(video_path), ss=clip_start / source_fps, t=duration)
        .filter("fps", fps=len(frame_ids) / duration)
    )
    output, _ = command.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
        capture_stdout=True, quiet=True
    )
    width = int(entry["width"])
    height = int(entry["height"])
    images_list = np.frombuffer(output, np.uint8).reshape([-1, height, width, 3])
    if len(images_list) != len(frame_ids):
        raise RuntimeError(
            f"TubeDETR frame sampling mismatch: decoded {len(images_list)} frames, "
            f"expected {len(frame_ids)}"
        )
    transforms = make_video_transforms("test", cautious=True, resolution=resolution)
    images, _ = transforms(images_list, None)
    if not torch.is_tensor(images) or images.ndim != 4 or images.shape[0] != 3:
        raise RuntimeError(f"unexpected transformed tensor: {getattr(images, 'shape', None)}")
    if not torch.isfinite(images).all().item():
        raise RuntimeError("transformed support tensor contains non-finite values")
    return images.float().cpu()


def run_smoke(data_root: Path, annotation_path: Path, *, resolution: int, fps: int, video_max_len: int) -> dict[str, Any]:
    """Decode/transform one real sample with the upstream TubeDETR dataset."""

    repo = PROJECT_ROOT / "external" / "TubeDETR"
    environment_bin = str(Path(sys.executable).resolve().parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if environment_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join([environment_bin, *path_entries])
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import torch

    from datasets.video_transforms import make_video_transforms
    from datasets.vidstg import VideoModulatedSTGrounding

    dataset = VideoModulatedSTGrounding(
        data_root,
        annotation_path,
        transforms=make_video_transforms("test", cautious=True, resolution=resolution),
        is_train=False,
        video_max_len=video_max_len,
        video_max_len_train=video_max_len,
        fps=fps,
        tmp_crop=False,
        tmp_loc=True,
        stride=0,
    )
    images, targets, target = dataset[0]
    if not torch.is_tensor(images) or images.ndim != 4 or images.shape[0] != 3:
        raise RuntimeError(f"unexpected transformed video tensor: {type(images)!r} {getattr(images, 'shape', None)}")
    if not torch.isfinite(images).all().item():
        raise RuntimeError("transformed video tensor contains non-finite values")
    if len(targets) != images.shape[1]:
        raise RuntimeError(f"target/frame mismatch: {len(targets)} vs {images.shape[1]}")
    if not target.get("caption"):
        raise RuntimeError("TubeDETR sample target lost query caption")
    nonempty_targets = sum(1 for frame_target in targets if len(frame_target.get("boxes", [])))
    return {
        "status": "ok",
        "sample_index": 0,
        "dataset_length": len(dataset),
        "tensor_shape_cthw": [int(value) for value in images.shape],
        "tensor_dtype": str(images.dtype),
        "tensor_min": float(images.min().item()),
        "tensor_max": float(images.max().item()),
        "target_frames": len(targets),
        "target_frames_with_boxes": nonempty_targets,
        "target_video_id": int(target["video_id"]),
        "target_query": target["caption"],
        "target_inter_idx": [int(value) for value in target["inter_idx"]],
        "tube_detr_dataset": "datasets.vidstg.VideoModulatedSTGrounding",
        "resolution": resolution,
        "fps": fps,
        "video_max_len": video_max_len,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vidstg-annotations", type=Path, default=DEFAULT_VIDSTG_ANNOTATIONS)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--vidor-annotation-zip", type=Path, default=DEFAULT_VIDOR_ANNOTATION_ZIP)
    parser.add_argument("--vidor-annotation-root", type=Path, default=None)
    parser.add_argument("--video-archive", type=Path, default=DEFAULT_VIDOR_VIDEO_ZIP)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--num-videos", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--video-max-len", type=int, default=200)
    parser.add_argument("--verify-video-zip", action="store_true")
    parser.add_argument("--verify-annotation-zip", action="store_true")
    parser.add_argument("--extract-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_dir = args.vidstg_annotations.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    annotation_zip = args.vidor_annotation_zip.expanduser().resolve() if args.vidor_annotation_zip else None
    annotation_root = args.vidor_annotation_root.expanduser().resolve() if args.vidor_annotation_root else None
    video_archive = args.video_archive.expanduser().resolve()

    split_annotation_path = annotation_dir / f"{args.split}_annotations.json"
    split_files_path = annotation_dir / f"{args.split}_files.json"
    if not split_annotation_path.is_file() or not split_files_path.is_file():
        raise FileNotFoundError(f"missing official VidSTG split files under {annotation_dir}")
    vidstg_annotations = load_json(split_annotation_path)
    split_files = [str(value) for value in load_json(split_files_path)]
    if not isinstance(vidstg_annotations, list):
        raise TypeError(f"expected a list in {split_annotation_path}")

    vidor_records = load_vidor_annotations(annotation_zip, annotation_root)
    vidstg_schema_valid_count = sum(
        not validate_vidstg_annotation(item)
        for item in vidstg_annotations
        if isinstance(item, dict)
    )
    vidor_schema_valid_count = sum(
        not validate_vidor_record(item) for item in vidor_records.values()
    )
    hc_exclusions = collect_hc_exclusions(PROJECT_ROOT)
    candidate_pool, filter_counters = build_candidate_pool(
        vidstg_annotations,
        vidor_records,
        hc_exclusions=hc_exclusions,
    )
    selected = select_candidates(candidate_pool, count=args.num_videos, seed=args.seed)

    records: list[dict[str, Any]] = []
    split_file_set = set(split_files)
    for support_index, candidate in enumerate(selected):
        relative_video_path = str(candidate["video_path"])
        absolute_video_path = (data_root / "video" / relative_video_path).resolve()
        annotation_id = (
            f"vidstg-{args.split}-a{int(candidate['annotation_index']):05d}"
            f"-q{int(candidate['query_index']):02d}"
        )
        record = {
            "support_index": support_index,
            "vid": str(candidate["vid"]),
            "annotation_id": annotation_id,
            "query": str(candidate["query"]),
            "caption": str(candidate["query"]),
            "query_type": str(candidate["query_type"]),
            "qtype": str(candidate["qtype"]),
            "target_id": int(candidate["target_id"]),
            "source_annotation_index": int(candidate["annotation_index"]),
            "source_query_index": int(candidate["query_index"]),
            # ``video_path`` is intentionally absolute so a controller can
            # import an entry without reconstructing the data root.
            "video_path": str(absolute_video_path),
            "video_relpath": relative_video_path,
            # A source cluster is one raw VidOR video.  This conservative
            # identity prevents two support entries from sharing a cluster.
            "source_cluster": str(candidate["vid"]),
            "frame_count": int(candidate["frame_count"]),
            "fps": float(candidate["fps"]),
            "width": int(candidate["width"]),
            "height": int(candidate["height"]),
            "used_segment": [int(candidate["segment_begin"]), int(candidate["segment_end"])],
            "tube_interval": [int(candidate["tube_begin"]), int(candidate["tube_end"])],
            "trajectory_frame_count": len(candidate["trajectory"]),
            "in_official_split_file": str(candidate["vid"]) in split_file_set,
            "video_archive_member": None,
            "video_available": False,
            "video_file_size_bytes": None,
            "video_sha256": None,
        }
        records.append(record)

    data_root.mkdir(parents=True, exist_ok=True)
    annotation_output_path = data_root / "annotations" / f"{args.split}.json"
    annotation_output_path.parent.mkdir(parents=True, exist_ok=True)

    video_archive_state = archive_report(
        video_archive,
        expected_bytes=EXPECTED_VIDOR_VIDEO_BYTES,
        expected_sha256=EXPECTED_VIDOR_VIDEO_SHA256,
        verify=args.verify_video_zip,
    )
    extraction_report = {
        "status": "not_attempted",
        "extracted_count": 0,
        "already_present_count": 0,
        "missing_member_count": None,
        "missing_members": [],
    }
    if args.extract_videos and video_archive_state["status"] == "complete":
        extraction_report = extract_selected_videos(video_archive, records, data_root)
        extraction_report["status"] = "ok" if extraction_report["missing_member_count"] == 0 else "partial"
    elif video_archive_state["status"] != "complete":
        extraction_report["status"] = f"blocked_{video_archive_state['status']}"

    selected_with_trajectories = []
    for record in records:
        selected_with_trajectories.append(
            {
                **record,
                "trajectory": candidate_pool[str(record["vid"])].get("trajectory", {}),
                "segment_begin": record["used_segment"][0],
                "segment_end": record["used_segment"][1],
                "tube_begin": record["tube_interval"][0],
                "tube_end": record["tube_interval"][1],
            }
        )
    tubedetr_annotation = build_tubedetr_annotation(
        [
            record for record in selected_with_trajectories
        ]
    )
    annotation_output_path.write_text(
        json.dumps(tubedetr_annotation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    smoke: dict[str, Any]
    if args.smoke and all(record["video_available"] for record in records):
        try:
            smoke = run_smoke(
                data_root,
                annotation_output_path,
                resolution=args.resolution,
                fps=args.fps,
                video_max_len=args.video_max_len,
            )
        except Exception as exc:  # report the blocker while retaining the manifest
            smoke = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    elif args.smoke:
        smoke = {
            "status": "blocked_missing_video",
            "available_video_count": sum(bool(record["video_available"]) for record in records),
            "required_video_count": len(records),
        }
    else:
        smoke = {"status": "disabled"}

    selected_identity = [
        {
            key: record[key]
            for key in (
                "support_index",
                "vid",
                "query",
                "query_type",
                "qtype",
                "target_id",
                "source_annotation_index",
                "source_query_index",
                "video_relpath",
                "frame_count",
                "fps",
                "width",
                "height",
                "used_segment",
                "tube_interval",
                "trajectory_frame_count",
            )
        }
        for record in records
    ]
    manifest = {
        "phase": "phase3",
        "purpose": "wrong_domain_vidstg_support",
        "status": "ready" if all(record["video_available"] for record in records) else "annotation_ready_video_pending",
        "selection": {
            "split": args.split,
            "requested_video_count": args.num_videos,
            "selected_video_count": len(records),
            "seed": args.seed,
            "algorithm": "one eligible declarative caption per unique VidSTG video; rank by sha256(phase3-wrong-domain:seed:vid)",
            "query_preference": "caption/declarative, then question/interrogative if a caption is unavailable",
            "selected_identity_sha256": canonical_sha256(selected_identity),
        },
        # ``source`` and ``entries`` are the stable, controller-facing API.
        # The more verbose aliases below preserve provenance for audit tools.
        "source": {
            "dataset": "VidSTG annotations + VidOR validation videos",
            "split": args.split,
            "vidstg_annotation_file": str(split_annotation_path),
            "vidstg_annotation_sha256": sha256_file(split_annotation_path),
            "vidstg_split_file": str(split_files_path),
            "vidstg_split_file_sha256": sha256_file(split_files_path),
            "vidor_annotation_archive": str(annotation_zip) if annotation_zip else None,
            "vidor_annotation_sha256": sha256_file(annotation_zip) if annotation_zip and annotation_zip.is_file() else None,
            "vidor_annotation_expected_sha256": EXPECTED_VIDOR_ANNOTATION_SHA256,
            "vidor_video_archive": str(video_archive),
            "vidor_video_sha256": video_archive_state["actual_sha256"],
            "vidor_video_expected_sha256": EXPECTED_VIDOR_VIDEO_SHA256,
            "vidstg_repo": str((PROJECT_ROOT / "external" / "VidSTG-Dataset").resolve()),
            "vidstg_repo_commit": git_commit(PROJECT_ROOT / "external" / "VidSTG-Dataset"),
        },
        # A compact hash-only provenance block for callers that need to
        # validate the annotation inputs without depending on the verbose
        # ``source``/``sources`` layouts above.
        "annotation": {
            "vidstg_annotation_file": str(split_annotation_path),
            "vidstg_annotation_sha256": sha256_file(split_annotation_path),
            "vidstg_split_file": str(split_files_path),
            "vidstg_split_file_sha256": sha256_file(split_files_path),
            "vidor_annotation_archive": str(annotation_zip) if annotation_zip else None,
            "vidor_annotation_sha256": sha256_file(annotation_zip) if annotation_zip and annotation_zip.is_file() else None,
            "vidor_annotation_expected_sha256": EXPECTED_VIDOR_ANNOTATION_SHA256,
        },
        "annotation_structure": {
            "vidstg_container": "list",
            "vidstg_required_keys": sorted(REQUIRED_VIDSTG_KEYS),
            "vidstg_schema_valid_records": vidstg_schema_valid_count,
            "vidor_container": "one JSON object per validation/####/<video_id>.json member",
            "vidor_required_keys": sorted(REQUIRED_VIDOR_KEYS),
            "vidor_schema_valid_records": vidor_schema_valid_count,
            "vidor_video_member_layout": "video/<video_path>",
        },
        "sources": {
            "vidstg_repo": str((PROJECT_ROOT / "external" / "VidSTG-Dataset").resolve()),
            "vidstg_repo_commit": git_commit(PROJECT_ROOT / "external" / "VidSTG-Dataset"),
            "vidstg_annotation_file": str(split_annotation_path),
            "vidstg_annotation_sha256": sha256_file(split_annotation_path),
            "vidstg_split_file": str(split_files_path),
            "vidstg_split_file_sha256": sha256_file(split_files_path),
            "vidor_annotation_archive": str(annotation_zip) if annotation_zip else None,
            "vidor_annotation_archive_sha256": sha256_file(annotation_zip) if annotation_zip and annotation_zip.is_file() else None,
            "vidor_annotation_expected_sha256": EXPECTED_VIDOR_ANNOTATION_SHA256,
            "vidor_video_archive": str(video_archive),
            "vidor_video_archive_sha256": video_archive_state["actual_sha256"],
            "vidor_video_archive_expected_sha256": EXPECTED_VIDOR_VIDEO_SHA256,
        },
        "counts": {
            "vidstg_annotation_records": len(vidstg_annotations),
            "vidstg_unique_video_ids": len({str(item.get("vid")) for item in vidstg_annotations if isinstance(item, dict) and item.get("vid") is not None}),
            "vidstg_split_file_video_ids": len(split_files),
            "vidor_annotation_records": len(vidor_records),
            "eligible_unique_videos": len(candidate_pool),
            "selected_unique_videos": len(records),
            "selected_query_type_counts": dict(Counter(record["query_type"] for record in records)),
            "selected_qtype_counts": dict(Counter(record["qtype"] for record in records)),
            "filter_counters": filter_counters,
        },
        "hc_query_exclusion": {
            "policy": "exact normalized caption overlap and video-stem overlap are rejected; HC data are never used as support",
            "hc_source_files": hc_exclusions["source_files"],
            "hc_video_stem_count": hc_exclusions["video_count"],
            "hc_query_text_count": hc_exclusions["query_count"],
            "selected_video_stem_overlap_count": sum(
                Path(str(record["video_path"])).stem in hc_exclusions["video_stems"] for record in records
            ),
            "selected_query_text_overlap_count": sum(
                normalize_text(str(record["query"])) in hc_exclusions["query_texts_normalized"] for record in records
            ),
            "all_selected_videos_are_from_official_vidstg_split": all(record["in_official_split_file"] for record in records),
        },
        "archives": {
            "vidor_annotation": archive_report(
                annotation_zip,
                expected_bytes=EXPECTED_VIDOR_ANNOTATION_BYTES,
                expected_sha256=EXPECTED_VIDOR_ANNOTATION_SHA256,
                verify=args.verify_annotation_zip,
            )
            if annotation_zip
            else {"status": "not_supplied"},
            "vidor_video": video_archive_state,
            "extraction": extraction_report,
        },
        "tube_detr": {
            "dataset_class": "datasets.vidstg.VideoModulatedSTGrounding",
            "repo": str((PROJECT_ROOT / "external" / "TubeDETR").resolve()),
            "data_root": str(data_root),
            "video_root": str((data_root / "video").resolve()),
            "annotation_file": str(annotation_output_path.resolve()),
            "required_layout": "data_root/video/<VidOR video_path> and data_root/annotations/test.json",
            "frame_tensor_contract": "float32 CxTxHxW, RGB, [0,1] before ImageNet normalization and normalized after transforms",
            "target_contract": "trajectories[original_video_id][str(target_id)][str(frame_id)] contains bbox [xmin,ymin,width,height]",
            "sampling": {"fps": args.fps, "video_max_len": args.video_max_len, "resolution": args.resolution},
        },
        "smoke_test": smoke,
        "records": records,
        "entries": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    if manifest["status"] == "ready" and args.smoke and smoke.get("status") != "ok":
        raise RuntimeError(f"video data are present but smoke test did not pass: {smoke}")


if __name__ == "__main__":
    main()
