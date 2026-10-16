#!/usr/bin/env python3
"""Query-independent native STVG-R1 source adapter for Table 1.

The module is inert on import.  Canonicalization, official YOLO/SAM2 tracking,
and R1 generation are separate explicit calls.  ``canonicalize`` streams the
selected FFmpeg RGB frames directly into an FFV1 encoder; it never writes a raw
RGB clip to disk.  Tracking cache identity excludes the query, while each R1
generation receives its own validated query-bearing request.

This file does not load model weights, read annotations, score predictions,
or start work at import time.  The CLI only offers synthetic contracts and a
query-free metadata comparison against the already-saved native-instance
eight-case requests/cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCHEMA = "stvg-table1-native-instance-adapter-v1"
CANONICAL_SCHEMA = "stvg-table1-canonical-input-v1"
TRACKING_SCHEMA = "stvg-table1-query-independent-tracking-v1"
WHO_SCHEMA = "stvg-table1-independent-who-v1"
MAX_NEW_TOKENS = 2048
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"Target ID:\s*(-?\d+)", re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _native():
    # This official helper is standard-library-only until a returned tracking
    # callable or model inference function is explicitly invoked.
    from scripts import stvg_r1_native_teacher_v1 as native
    return native


def _source_identity(source_case: Mapping[str, Any]) -> dict[str, Any]:
    """Read only the source video and clock metadata needed for canonical input."""
    if not isinstance(source_case, Mapping) or not isinstance(source_case.get("input"), Mapping):
        raise TypeError("source_case.input must be a mapping")
    inp = source_case["input"]
    dataset = source_case.get("dataset")
    if dataset not in {"hc2", "vidstg"}:
        raise ValueError("dataset must be hc2 or vidstg so the frozen decode policy is unambiguous")
    video_path = inp.get("video_path")
    source_sha = inp.get("video_sha256")
    fps = inp.get("fps")
    width = inp.get("width")
    height = inp.get("height")
    bounds = source_case.get("video_bounds")
    if not isinstance(video_path, str) or not video_path:
        raise ValueError("input.video_path must be a nonempty string")
    if not _is_sha(source_sha):
        raise ValueError("input.video_sha256 must be a lowercase SHA-256")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("input.fps must be finite and positive")
    for name, value in (("input.width", width), ("input.height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("video_bounds must be a two-element half-open physical frame interval")
    start, end = bounds
    if (isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
            or not math.isfinite(float(start)) or not math.isfinite(float(end))
            or int(start) != start or int(end) != end or start < 0 or end <= start):
        raise ValueError("video_bounds must be an increasing integral [start,end) interval")
    return {
        "dataset": dataset,
        "video_path": video_path,
        "source_video_sha256": source_sha,
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "start": int(start),
        "declared_end": int(end),
        "decode_policy": "ffmpeg_default_timing" if dataset == "hc2" else "ffmpeg_vsync0_original_grid",
    }


def _ffmpeg_identity() -> tuple[Path, str]:
    """Load the existing decoder locator only when canonicalization is called."""
    from vg_tta.unanchored_dense_shift_data_v1 import FFMPEG, FFMPEG_SHA
    path = Path(FFMPEG).resolve(strict=True)
    actual = _sha_file(path)
    if actual != FFMPEG_SHA:
        raise RuntimeError("FFmpeg bytes differ from the project-pinned decoder")
    return path, actual


def _canonical_key(identity: Mapping[str, Any], ffmpeg_sha256: str) -> str:
    if not _is_sha(ffmpeg_sha256):
        raise ValueError("FFmpeg identity must be a lowercase SHA-256")
    payload = {
        "schema": CANONICAL_SCHEMA,
        "source_video_sha256": identity["source_video_sha256"],
        "dataset_decode_policy": identity["decode_policy"],
        "declared_physical_bounds_half_open": [identity["start"], identity["declared_end"]],
        "fps": identity["fps"],
        "width": identity["width"],
        "height": identity["height"],
        "ffmpeg_sha256": ffmpeg_sha256,
        "rgb_format": "rgb24",
        "canonical_codec": "ffv1-level3-bgr0-matroska",
        "roundtrip_policy": "ffmpeg-vsync0-rgb24-bytehash-v1",
    }
    return _sha_bytes(_canonical_json(payload))


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable file: {path.name}")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _read_canonical_cache(cache_dir: Path, identity: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    receipt_path = cache_dir / "CANONICAL_RECEIPT.json"
    video_path = cache_dir / "input.mkv"
    if not receipt_path.exists() and not video_path.exists():
        return None
    if not receipt_path.is_file() or not video_path.is_file():
        raise RuntimeError("canonical cache is partial; preserve it and use a new cache root after review")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema", "status", "cache_key", "source_video_sha256", "dataset_decode_policy",
        "declared_physical_bounds_half_open", "available_physical_bounds_half_open",
        "metadata_missing_physical_ids", "ffmpeg_sha256", "canonical_video_sha256",
        "canonical_pixel_sha256", "local_fps", "local_frame_count", "frame_bytes",
        "canonical_video_bytes", "GT_read", "CPU_only", "query_independent",
    }
    if set(receipt) != required:
        raise RuntimeError("canonical cache receipt schema differs from the frozen adapter schema")
    frame_bytes = identity["width"] * identity["height"] * 3
    bounds = receipt.get("available_physical_bounds_half_open")
    if (receipt["schema"] != CANONICAL_SCHEMA or receipt["status"] != "complete"
            or receipt["cache_key"] != key or receipt["source_video_sha256"] != identity["source_video_sha256"]
            or receipt["dataset_decode_policy"] != identity["decode_policy"]
            or receipt["declared_physical_bounds_half_open"] != [identity["start"], identity["declared_end"]]
            or receipt["local_fps"] != identity["fps"] or receipt["frame_bytes"] != frame_bytes
            or not isinstance(bounds, list) or len(bounds) != 2
            or bounds[0] != identity["start"] or not identity["start"] < bounds[1] <= identity["declared_end"]
            or receipt["local_frame_count"] != bounds[1] - bounds[0]
            or receipt["metadata_missing_physical_ids"] != list(range(bounds[1], identity["declared_end"]))
            or not _is_sha(receipt["ffmpeg_sha256"])
            or not _is_sha(receipt["canonical_video_sha256"])
            or not _is_sha(receipt["canonical_pixel_sha256"])
            or isinstance(receipt["canonical_video_bytes"], bool)
            or not isinstance(receipt["canonical_video_bytes"], int) or receipt["canonical_video_bytes"] <= 0
            or receipt["GT_read"] is not False or receipt["CPU_only"] is not True
            or receipt["query_independent"] is not True):
        raise RuntimeError("canonical cache receipt does not match the requested source/clock binding")
    if _sha_file(video_path) != receipt["canonical_video_sha256"] or video_path.stat().st_size != receipt["canonical_video_bytes"]:
        raise RuntimeError("canonical clip bytes no longer match the saved lossless roundtrip receipt")
    return receipt


def _request_from_receipt(
    identity: Mapping[str, Any], query_text: str, video_path: Path, receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be nonempty")
    count = receipt.get("local_frame_count")
    fps = float(identity["fps"])
    start = int(receipt["available_physical_bounds_half_open"][0])
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise RuntimeError("canonical receipt has no usable local frame count")
    frame_map = [
        {
            "local_frame_index": i,
            "physical_frame_id": start + i,
            "source_timestamp_s": (start + i) / fps,
        }
        for i in range(count)
    ]
    request = {
        "video_path": str(video_path.resolve()),
        "query_text": query_text,
        "source_video_sha256": identity["source_video_sha256"],
        "canonical_video_sha256": receipt["canonical_video_sha256"],
        "canonical_pixel_sha256": receipt["canonical_pixel_sha256"],
        "local_fps": fps,
        "local_frame_count": count,
        "frame_map": frame_map,
    }
    _native().validate_request(request)
    return request


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(fd)
        yield
    finally:
        os.close(fd)
        path.unlink(missing_ok=True)


def _stream_canonicalize(
    ffmpeg: Path,
    source_path: Path,
    temp_video: Path,
    *,
    dataset: str,
    start: int,
    end: int,
    width: int,
    height: int,
    fps: float,
) -> tuple[int, str, int]:
    """Pipe one decoded RGB frame at a time directly to FFV1; no raw clip file."""
    frame_bytes = width * height * 3
    decoder_cmd = [str(ffmpeg), "-v", "error", "-i", str(source_path)]
    if dataset == "vidstg":
        decoder_cmd += ["-vsync", "0"]
    decoder_cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    encoder_cmd = [
        str(ffmpeg), "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an",
        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", "-f", "matroska", str(temp_video),
    ]
    decoder_log = temp_video.with_name("decode.stderr.log")
    encoder_log = temp_video.with_name("encode.stderr.log")
    digest = hashlib.sha256()
    decoded = selected = 0
    decoder = encoder = None
    try:
        with decoder_log.open("xb") as decoder_stderr, encoder_log.open("xb") as encoder_stderr:
            decoder = subprocess.Popen(decoder_cmd, stdout=subprocess.PIPE, stderr=decoder_stderr, bufsize=0)
            encoder = subprocess.Popen(encoder_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=encoder_stderr, bufsize=0)
            assert decoder.stdout is not None and encoder.stdin is not None
            while True:
                frame = bytearray()
                while len(frame) < frame_bytes:
                    part = decoder.stdout.read(frame_bytes - len(frame))
                    if not part:
                        break
                    frame.extend(part)
                if not frame:
                    break
                if len(frame) != frame_bytes:
                    raise RuntimeError("FFmpeg emitted a partial RGB frame")
                if start <= decoded < end:
                    digest.update(frame)
                    encoder.stdin.write(frame)
                    selected += 1
                decoded += 1
            decoder.stdout.close()
            decoder_rc = decoder.wait(timeout=60)
            if decoder_rc != 0:
                raise subprocess.CalledProcessError(decoder_rc, decoder_cmd)
            encoder.stdin.close()
            encoder_rc = encoder.wait(timeout=60)
            if encoder_rc != 0:
                raise subprocess.CalledProcessError(encoder_rc, encoder_cmd)
    except BaseException:
        for child in (decoder, encoder):
            if child is not None and child.poll() is None:
                child.kill()
        for child in (decoder, encoder):
            if child is not None:
                try:
                    child.wait(timeout=10)
                except Exception:
                    pass
        raise
    if not temp_video.is_file() or selected <= 0:
        raise RuntimeError("FFmpeg did not produce a nonempty canonical clip")
    return decoded, digest.hexdigest(), selected


def _roundtrip_rgb_hash(ffmpeg: Path, video_path: Path) -> tuple[str, int]:
    command = [str(ffmpeg), "-v", "error", "-i", str(video_path), "-vsync", "0",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    digest = hashlib.sha256()
    byte_count = 0
    with video_path.with_name("roundtrip.stderr.log").open("xb") as stderr:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr, bufsize=0)
        assert child.stdout is not None
        try:
            for chunk in iter(lambda: child.stdout.read(8 << 20), b""):
                digest.update(chunk)
                byte_count += len(chunk)
            child.stdout.close()
            code = child.wait(timeout=60)
            if code != 0:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
            raise
    return digest.hexdigest(), byte_count


def canonicalize(
    source_case: Mapping[str, Any], query_text: str, cache_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Create or reuse a query-independent FFV1 clip and return exact R1 request.

    The cache key binds source bytes, dataset-specific decode timing, physical
    ``[start,end)`` bounds, fps, dimensions, FFmpeg bytes, and roundtrip policy.
    The request itself remains query-specific and is never stored in the shared
    canonical-video cache.
    """
    identity = _source_identity(source_case)
    source_path = Path(identity["video_path"]).resolve(strict=True)
    if _sha_file(source_path) != identity["source_video_sha256"]:
        raise RuntimeError("input video bytes differ from the source-case SHA")
    ffmpeg, ffmpeg_sha256 = _ffmpeg_identity()
    key = _canonical_key(identity, ffmpeg_sha256)
    root = Path(cache_root).resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("canonical cache_root must be inside the project workspace")
    cache_dir = root / "canonical" / key
    video_path = cache_dir / "input.mkv"
    receipt = _read_canonical_cache(cache_dir, identity, key)
    if receipt is not None and receipt["ffmpeg_sha256"] != ffmpeg_sha256:
        raise RuntimeError("canonical cache was produced with a different FFmpeg binary")
    if receipt is None:
        lock_path = root / ".locks" / f"canonical-{key}.lock"
        with _exclusive_lock(lock_path):
            receipt = _read_canonical_cache(cache_dir, identity, key)
            if receipt is not None and receipt["ffmpeg_sha256"] != ffmpeg_sha256:
                raise RuntimeError("canonical cache was produced with a different FFmpeg binary")
            if receipt is None:
                cache_dir.mkdir(parents=True, exist_ok=False)
                temp_video = cache_dir / "input.partial.mkv"
                decoded, source_pixel_sha, selected = _stream_canonicalize(
                    ffmpeg, source_path, temp_video,
                    dataset=identity["dataset"], start=identity["start"], end=identity["declared_end"],
                    width=identity["width"], height=identity["height"], fps=identity["fps"],
                )
                available_end = min(identity["declared_end"], decoded)
                missing = list(range(available_end, identity["declared_end"]))
                if identity["start"] >= available_end or selected != available_end - identity["start"]:
                    raise RuntimeError("canonical decode did not cover a nonempty exact physical interval")
                roundtrip_sha, roundtrip_bytes = _roundtrip_rgb_hash(ffmpeg, temp_video)
                frame_bytes = identity["width"] * identity["height"] * 3
                if (roundtrip_sha != source_pixel_sha or roundtrip_bytes != selected * frame_bytes):
                    raise RuntimeError("FFV1 roundtrip RGB hash/byte count differs from source decoder output")
                # Keep the previous pipeline's encoded-clock metadata check.
                import cv2
                cap = cv2.VideoCapture(str(temp_video))
                cv_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cv_fps = float(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                if cv_count != selected or abs(cv_fps - identity["fps"]) > 1e-6:
                    raise RuntimeError("canonical encoded frame count/fps differs from the source physical clock")
                temp_video.replace(video_path)
                receipt = {
                    "schema": CANONICAL_SCHEMA,
                    "status": "complete",
                    "cache_key": key,
                    "source_video_sha256": identity["source_video_sha256"],
                    "dataset_decode_policy": identity["decode_policy"],
                    "declared_physical_bounds_half_open": [identity["start"], identity["declared_end"]],
                    "available_physical_bounds_half_open": [identity["start"], available_end],
                    "metadata_missing_physical_ids": missing,
                    "ffmpeg_sha256": ffmpeg_sha256,
                    "canonical_video_sha256": _sha_file(video_path),
                    "canonical_pixel_sha256": source_pixel_sha,
                    "local_fps": identity["fps"],
                    "local_frame_count": selected,
                    "frame_bytes": frame_bytes,
                    "canonical_video_bytes": video_path.stat().st_size,
                    "GT_read": False,
                    "CPU_only": True,
                    "query_independent": True,
                }
                _write_new_json(cache_dir / "CANONICAL_RECEIPT.json", receipt)
    if receipt is None:
        raise AssertionError("canonicalization ended without a receipt")
    return _request_from_receipt(identity, query_text, video_path, receipt)


def _query_free_request_binding(request: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _native()._normalize_request(request)
    return {
        "source_video_sha256": normalized["source_video_sha256"],
        "canonical_video_sha256": normalized["canonical_video_sha256"],
        "canonical_pixel_sha256": normalized["canonical_pixel_sha256"],
        "local_fps": normalized["local_fps"],
        "local_frame_count": normalized["local_frame_count"],
        "frame_map": normalized["frame_map"],
    }


def tracking_key(request: Mapping[str, Any], fingerprints: Mapping[str, str]) -> str:
    """Hash all query-independent video/clock/tracker inputs for one cache."""
    required = {
        "stvg_r1_revision", "grounded_sam2_revision", "dettrack_sha256", "mask2marker_sha256",
        "sam2_config_sha256", "sam2_build_sha256", "sam2_misc_sha256", "sam2_transforms_sha256",
        "mask_dictionary_sha256", "detector_asset_sha256", "sam2_asset_sha256",
        "bounded_frame_cache_sha256", "qkv_compat_sha256", "adapter_sha256",
    }
    if set(fingerprints) != required:
        raise ValueError(f"tracker fingerprints must have exactly {sorted(required)}")
    for name, digest in fingerprints.items():
        if name.endswith("_sha256") and not _is_sha(digest):
            raise ValueError(f"invalid SHA-256 fingerprint: {name}")
        if name.endswith("_revision") and (not isinstance(digest, str) or not digest):
            raise ValueError(f"invalid revision fingerprint: {name}")
    binding = _query_free_request_binding(request)
    payload = {
        "schema": TRACKING_SCHEMA,
        "video_clock_binding": binding,
        "official_tracker_fingerprints": dict(sorted(fingerprints.items())),
        "target_mode": "all",
        "mask_area_filter_ratio": 1.0 / 3.0,
        "frame_cache_limit": 2,
        "qkv_backend": "torch_SDPA_half_QKV",
        "query_used_by_tracker": False,
    }
    return _sha_bytes(_canonical_json(payload))


_TRACKING_FINGERPRINTS_CACHE: dict[str, str] | None = None


def _tracking_fingerprints() -> dict[str, str]:
    """Verify the current source/helpers/assets immediately before tracking."""
    global _TRACKING_FINGERPRINTS_CACHE
    if _TRACKING_FINGERPRINTS_CACHE is not None:
        return dict(_TRACKING_FINGERPRINTS_CACHE)
    native = _native()
    stvg_repo = ROOT / "external/stvg-r1"
    gs2_repo = ROOT / "external/Grounded-SAM-2-main"
    expected = {
        "stvg_r1_revision": native.STVG_R1_REVISION,
        "grounded_sam2_revision": native.GROUND_SAM2_REVISION,
        "dettrack_sha256": native.STVG_R1_FILE_SHA256["preprocess/dettrack.py"],
        "mask2marker_sha256": native.STVG_R1_FILE_SHA256["preprocess/mask2marker.py"],
        "sam2_config_sha256": native.GROUND_SAM2_FILE_SHA256["sam2/configs/sam2.1/sam2.1_hiera_l.yaml"],
        "sam2_build_sha256": native.GROUND_SAM2_FILE_SHA256["sam2/build_sam.py"],
        "sam2_misc_sha256": native.GROUND_SAM2_FILE_SHA256["sam2/utils/misc.py"],
        "sam2_transforms_sha256": native.GROUND_SAM2_FILE_SHA256["sam2/utils/transforms.py"],
        "mask_dictionary_sha256": native.GROUND_SAM2_FILE_SHA256["utils/mask_dictionary_model.py"],
    }
    for relative, expected_sha in native.STVG_R1_FILE_SHA256.items():
        if _sha_file(stvg_repo / relative) != expected_sha:
            raise RuntimeError(f"official STVG-R1 source pin mismatch: {relative}")
    for relative, expected_sha in native.GROUND_SAM2_FILE_SHA256.items():
        if _sha_file(gs2_repo / relative) != expected_sha:
            raise RuntimeError(f"official Grounded-SAM-2 source pin mismatch: {relative}")
    detector = ROOT / native.DETECTOR_ASSET_RELATIVE
    sam2 = ROOT / native.SAM2_ASSET_RELATIVE
    if not detector.is_file() or not sam2.is_file():
        raise FileNotFoundError("pinned YOLOv12-X/SAM2.1 assets are not present")
    if detector.stat().st_size != 119_322_638 or sam2.stat().st_size != 898_083_611:
        raise RuntimeError("pinned detector/segmenter asset size differs from official source readback")
    from scripts import stvg_sam_bounded_frame_cache010 as bounded
    from scripts import stvg_yolov12_qkv_checkpoint_compat_v2 as qkv
    fingerprints = {
        **expected,
        "detector_asset_sha256": _sha_file(detector),
        "sam2_asset_sha256": _sha_file(sam2),
        "bounded_frame_cache_sha256": _sha_file(Path(bounded.__file__)),
        "qkv_compat_sha256": _sha_file(Path(qkv.__file__)),
        "adapter_sha256": _sha_file(__file__),
    }
    _TRACKING_FINGERPRINTS_CACHE = dict(fingerprints)
    return fingerprints


def _safe_project_path(relative: Any, *, must_exist: bool = True) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("project-relative path must be a nonempty string")
    rel = Path(relative)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError("unsafe project-relative path")
    path = (ROOT / rel).resolve(strict=must_exist)
    if not path.is_relative_to(ROOT):
        raise ValueError("project-relative path resolves outside the workspace")
    return path


def _cache_relpath(path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("tracking cache_root must be inside the project workspace") from exc


def _write_tracking_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_json(path, value)


def _tracking_inventory(mask_dir: Path, request: Mapping[str, Any], cache_key: str) -> dict[str, Any]:
    native = _native()
    files, known_ids = native._load_mask_inventory(mask_dir)
    rows = []
    for index, path in sorted(files.items()):
        if index >= int(request["local_frame_count"]):
            raise RuntimeError("official mask index lies outside the canonical local frame clock")
        rows.append({"index": int(index), "path": _cache_relpath(path), "sha256": _sha_file(path), "bytes": path.stat().st_size})
    return {
        "schema": "stvg-table1-query-independent-mask-inventory-v1",
        "mask_count": len(rows),
        "known_ids": sorted(int(x) for x in known_ids),
        "files": rows,
        "tracking_cache_key": cache_key,
        "GT_read": False,
        "query_independent": True,
    }


def _validate_cached_tracking(cache_dir: Path, request: Mapping[str, Any], key: str,
                              fingerprints: Mapping[str, str]) -> dict[str, Any]:
    accepted_path = cache_dir / "ACCEPTED.json"
    inventory_path = cache_dir / "MASK_INVENTORY.json"
    if not accepted_path.is_file() or not inventory_path.is_file():
        raise RuntimeError("tracking cache is partial; preserve it and use a new cache root after review")
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    bind = _query_free_request_binding(request)
    expected = {
        "schema": TRACKING_SCHEMA, "status": "pass", "GT_read": False,
        "query_independent": True, "tracking_cache_key": key,
        "tracking_binding_sha256": _sha_bytes(_canonical_json(bind)),
        "fingerprints": dict(sorted(fingerprints.items())),
    }
    if any(accepted.get(k) != v for k, v in expected.items()):
        raise RuntimeError("cached tracking receipt does not match the exact query-free input/runtime key")
    if accepted.get("mask_inventory_sha256") != _sha_file(inventory_path):
        raise RuntimeError("cached mask inventory differs from the accepted tracking receipt")
    if (inventory.get("schema") != "stvg-table1-query-independent-mask-inventory-v1"
            or inventory.get("GT_read") is not False or inventory.get("query_independent") is not True):
        raise RuntimeError("cached mask inventory has the wrong privacy or query-independence marker")
    if inventory.get("tracking_cache_key") != key or inventory.get("mask_count") != len(inventory.get("files", [])):
        raise RuntimeError("cached mask inventory schema/count/key is inconsistent")
    marker = _safe_project_path(accepted.get("marker_video_path"))
    mask_dir = _safe_project_path(accepted.get("mask_data_dir"))
    if _sha_file(marker) != accepted["marker_video_sha256"]:
        raise RuntimeError("cached marker video bytes changed")
    actual_masks: set[tuple[str, str, int]] = set()
    for row in inventory["files"]:
        if set(row) != {"index", "path", "sha256", "bytes"}:
            raise RuntimeError("cached mask inventory row has an unexpected schema")
        path = _safe_project_path(row["path"])
        if (not path.is_relative_to(mask_dir.resolve()) or not path.is_file()
                or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]):
            raise RuntimeError("cached mask file differs from its exact inventory binding")
        actual_masks.add((row["path"], row["sha256"], int(row["index"])))
    indexed, ids = _native()._load_mask_inventory(mask_dir)
    if ({(_cache_relpath(p), _sha_file(p), int(i)) for i, p in indexed.items()} != actual_masks
            or sorted(ids) != inventory.get("known_ids")):
        raise RuntimeError("cached mask inventory is incomplete or its parsed ID set changed")
    return {"accepted": accepted, "mask_inventory": inventory}


def track(request: Mapping[str, Any], cache_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Reuse or create official query-independent all-category YOLO/SAM2 output.

    This function is an explicit GPU operation.  It imports the detector,
    tracker, and CUDA stack only on a cache miss; importing this module or using
    ``--self-test`` never enters that path.
    """
    native = _native()
    normalized = native._normalize_request(request)
    fingerprints = _tracking_fingerprints()
    key = tracking_key(normalized, fingerprints)
    root = Path(cache_root).resolve()
    cache_dir = root / "tracking" / key
    if not root.is_relative_to(ROOT):
        raise ValueError("tracking cache_root must be inside the project workspace")
    if cache_dir.exists():
        return _validate_cached_tracking(cache_dir, normalized, key, fingerprints)
    lock_path = root / ".locks" / f"tracking-{key}.lock"
    with _exclusive_lock(lock_path):
        if cache_dir.exists():
            return _validate_cached_tracking(cache_dir, normalized, key, fingerprints)
        cache_dir.mkdir(parents=True, exist_ok=False)
        official_output = cache_dir / "official"
        import torch
        import ultralytics
        from scripts.stvg_sam_bounded_frame_cache010 import bounded_frame_loading
        from scripts.stvg_yolov12_qkv_checkpoint_compat_v2 import install_checkpoint_qkv_aattn_compat_v2
        runner = native.make_official_preprocess_runner(
            ROOT / "external/stvg-r1", ROOT / "external/Grounded-SAM-2-main",
            detector_asset=ROOT / native.DETECTOR_ASSET_RELATIVE,
            sam2_asset=ROOT / native.SAM2_ASSET_RELATIVE,
        )
        original_yolo = ultralytics.YOLO
        compatibility = []

        def compatible_yolo(*args: Any, **kwargs: Any):
            model = original_yolo(*args, **kwargs)
            receipt = install_checkpoint_qkv_aattn_compat_v2(model.model)
            if receipt.get("aattn_count") != 16:
                raise RuntimeError("pinned YOLO checkpoint did not expose all 16 expected AAttn modules")
            compatibility.append(receipt)
            return model

        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        ultralytics.YOLO = compatible_yolo
        try:
            with bounded_frame_loading():
                prep = runner(normalized, target_mode="all", output_dir=official_output)
        finally:
            ultralytics.YOLO = original_yolo
        if len(compatibility) != 1:
            raise RuntimeError("official tracking must instantiate exactly one checkpoint-compatible YOLO model")
        marker_path = Path(prep["marker_video_path"]).resolve(strict=True)
        mask_dir = Path(prep["mask_data_dir"]).resolve(strict=True)
        if not marker_path.is_relative_to(cache_dir.resolve()) or not mask_dir.is_relative_to(cache_dir.resolve()):
            raise RuntimeError("official tracker outputs escaped the content-addressed cache directory")
        inventory = _tracking_inventory(mask_dir, normalized, key)
        inventory_path = cache_dir / "MASK_INVENTORY.json"
        _write_tracking_json(inventory_path, inventory)
        binding = _query_free_request_binding(normalized)
        accepted = {
            "schema": TRACKING_SCHEMA,
            "status": "pass",
            "GT_read": False,
            "query_independent": True,
            "tracking_cache_key": key,
            "tracking_binding_sha256": _sha_bytes(_canonical_json(binding)),
            "fingerprints": dict(sorted(fingerprints.items())),
            "source_video_sha256": normalized["source_video_sha256"],
            "canonical_video_sha256": normalized["canonical_video_sha256"],
            "canonical_pixel_sha256": normalized["canonical_pixel_sha256"],
            "frame_clock_sha256": normalized["frame_clock_sha256"],
            "local_fps": normalized["local_fps"],
            "local_frame_count": normalized["local_frame_count"],
            "target_mode": "all",
            "metadata_input": "empty_mapping",
            "marker_video_path": _cache_relpath(marker_path),
            "marker_video_sha256": prep["marker_video_sha256"],
            "mask_data_dir": _cache_relpath(mask_dir),
            "mask_inventory_path": _cache_relpath(inventory_path),
            "mask_inventory_sha256": _sha_file(inventory_path),
            "dettrack_s": float(prep["dettrack_s"]),
            "filter_marker_s": float(prep["filter_marker_s"]),
            "actual_tracking_wall_s": time.perf_counter() - started,
            "tracker_precision": prep["tracker_precision"],
            "preprocess_warnings": prep["preprocess_warnings"],
            "checkpoint_code_compatibility": {
                "source_sha256": fingerprints["qkv_compat_sha256"],
                "attention_backend": "torch_SDPA_half_QKV",
                "bitwise_equivalence_claimed": False,
                "CPU_frame_cache_limit": 2,
                "frame_cache_source_sha256": fingerprints["bounded_frame_cache_sha256"],
                "aattn_count": compatibility[0]["aattn_count"],
            },
            "peak_GPU_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        }
        _write_tracking_json(cache_dir / "ACCEPTED.json", accepted)
        return {"accepted": accepted, "mask_inventory": inventory}


def parse_who_independent(raw_output: str, known_ids: set[int] | list[int], *, truncated: bool = False) -> dict[str, Any]:
    """Parse the official Target ID independently of whether time is valid."""
    result: dict[str, Any] = {
        "schema": WHO_SCHEMA,
        "status": "invalid_format",
        "candidate_id": None,
        "answer_block_count": 0,
        "id_token_count": 0,
        "truncated_at_cap": bool(truncated),
        "GT_read": False,
    }
    if truncated:
        result["status"] = "truncated_at_cap"
        return result
    if not isinstance(raw_output, str):
        return result
    blocks = _ANSWER_RE.findall(raw_output)
    result["answer_block_count"] = len(blocks)
    if not blocks:
        return result
    ids = _ID_RE.findall(blocks[-1])
    result["id_token_count"] = len(ids)
    if len(ids) != 1:
        return result
    candidate_id = int(ids[0])
    result["candidate_id"] = candidate_id
    if candidate_id <= 0:
        result["status"] = "invalid_id"
    elif candidate_id not in {int(x) for x in known_ids}:
        result["status"] = "unknown_candidate_id"
    else:
        result["status"] = "valid"
    return result


def infer_who(
    request: Mapping[str, Any], tracking: Mapping[str, Any],
    processor: Any, model: Any, torch: Any, device: str,
) -> dict[str, Any]:
    """Run one frozen R1 call on an injected resident model and separate WHO/WHEN.

    No checkpoint is loaded here.  A capped generation fails WHO closed, but
    an independently valid Target ID is not made invalid merely because its
    time interval fails the separate official WHEN parser.
    """
    native = _native()
    normalized = native._normalize_request(request)
    accepted = tracking.get("accepted")
    inventory = tracking.get("mask_inventory")
    if not isinstance(accepted, Mapping) or not isinstance(inventory, Mapping):
        raise TypeError("tracking must contain accepted and mask_inventory mappings")
    binding = _query_free_request_binding(normalized)
    if (accepted.get("status") != "pass" or accepted.get("GT_read") is not False
            or accepted.get("query_independent") is not True
            or accepted.get("tracking_binding_sha256") != _sha_bytes(_canonical_json(binding))
            or inventory.get("tracking_cache_key") != accepted.get("tracking_cache_key")
            or inventory.get("GT_read") is not False):
        raise RuntimeError("tracking cache is not accepted for this exact canonical video/frame clock")
    known_ids = {int(x) for x in inventory.get("known_ids", [])}
    marker = _safe_project_path(accepted.get("marker_video_path"))
    if not marker.is_file() or _sha_file(marker) != accepted.get("marker_video_sha256"):
        raise RuntimeError("accepted marker video is missing or changed")
    from scripts import run_stvg_native_instance_teacher_storage002 as frozen_teacher
    generation = frozen_teacher._one_generation(
        normalized, marker, processor=processor, model=model, torch=torch, device=device,
    )
    if generation.get("generation_call_count") != 1 or generation.get("max_new_tokens") != MAX_NEW_TOKENS:
        raise RuntimeError("the frozen STVG-R1 generation call/cap changed")
    who = parse_who_independent(
        generation.get("raw_output"), known_ids,
        truncated=bool(generation.get("output_truncated_at_cap")),
    )
    when, projected = native.parse_native_output(
        generation.get("raw_output", ""),
        duration_s=normalized["local_frame_count"] / normalized["local_fps"],
        known_ids=known_ids,
        request=normalized,
    )
    return {
        "schema": SCHEMA,
        "generation": generation,
        "who": who,
        "when": when,
        "projected_interval": projected,
        "GT_read": False,
    }


def _synthetic_request(query: str) -> dict[str, Any]:
    return {
        "video_path": "/synthetic/no-open/input.mkv",
        "query_text": query,
        "source_video_sha256": "a" * 64,
        "canonical_video_sha256": "b" * 64,
        "canonical_pixel_sha256": "c" * 64,
        "local_fps": 25.0,
        "local_frame_count": 4,
        "frame_map": [
            {"local_frame_index": i, "physical_frame_id": 40 + i,
             "source_timestamp_s": (40 + i) / 25.0}
            for i in range(4)
        ],
    }


def _synthetic_fingerprints() -> dict[str, str]:
    return {
        "stvg_r1_revision": "synthetic-r1",
        "grounded_sam2_revision": "synthetic-sam2",
        "dettrack_sha256": "1" * 64,
        "mask2marker_sha256": "2" * 64,
        "sam2_config_sha256": "3" * 64,
        "sam2_build_sha256": "4" * 64,
        "sam2_misc_sha256": "5" * 64,
        "sam2_transforms_sha256": "6" * 64,
        "mask_dictionary_sha256": "7" * 64,
        "detector_asset_sha256": "8" * 64,
        "sam2_asset_sha256": "9" * 64,
        "bounded_frame_cache_sha256": "a" * 64,
        "qkv_compat_sha256": "b" * 64,
        "adapter_sha256": "c" * 64,
    }


def run_cpu_contracts() -> dict[str, Any]:
    """Pure synthetic checks; never invokes FFmpeg, model packages, or data."""
    forbidden_modules = {"torch", "transformers", "qwen_vl_utils", "cv2", "ultralytics", "sam2", "pycocotools"}
    before = forbidden_modules.intersection(sys.modules)
    if before:
        raise RuntimeError(f"model/video runtime was imported before contracts: {sorted(before)}")
    native = _native()
    request_a = _synthetic_request("synthetic query A")
    request_b = _synthetic_request("synthetic query B")
    norm_a = native.validate_request(request_a)
    norm_b = native.validate_request(request_b)
    fingerprints = _synthetic_fingerprints()
    key_a = tracking_key(norm_a, fingerprints)
    key_b = tracking_key(norm_b, fingerprints)
    assert key_a == key_b
    assert norm_a["query_sha256"] != norm_b["query_sha256"]
    assert norm_a["frame_clock_sha256"] == norm_b["frame_clock_sha256"]

    vid_identity = {
        "dataset": "vidstg", "video_path": "/synthetic/no-open/video.mp4",
        "source_video_sha256": "a" * 64, "fps": 25.0, "width": 16, "height": 12,
        "start": 40, "declared_end": 44, "decode_policy": "ffmpeg_vsync0_original_grid",
    }
    hc_identity = dict(vid_identity, dataset="hc2", decode_policy="ffmpeg_default_timing")
    assert _canonical_key(vid_identity, "d" * 64) != _canonical_key(hc_identity, "d" * 64)
    different_bounds = dict(vid_identity, declared_end=45)
    assert _canonical_key(vid_identity, "d" * 64) != _canonical_key(different_bounds, "d" * 64)
    different_video_path = dict(vid_identity, video_path="/synthetic/other/path.mp4")
    assert _canonical_key(vid_identity, "d" * 64) == _canonical_key(different_video_path, "d" * 64)
    different_tracker = dict(fingerprints, detector_asset_sha256="d" * 64)
    assert tracking_key(norm_a, fingerprints) != tracking_key(norm_a, different_tracker)
    synthetic_receipt = {
        "available_physical_bounds_half_open": [40, 44],
        "canonical_video_sha256": "b" * 64,
        "canonical_pixel_sha256": "c" * 64,
        "local_frame_count": 4,
    }
    synthetic_identity = {
        "source_video_sha256": "a" * 64,
        "fps": 25.0,
    }
    rebuilt = _request_from_receipt(synthetic_identity, "synthetic clock", Path("/synthetic/input.mkv"), synthetic_receipt)
    assert [row["physical_frame_id"] for row in rebuilt["frame_map"]] == [40, 41, 42, 43]
    assert [row["source_timestamp_s"] for row in rebuilt["frame_map"]] == [40 / 25, 41 / 25, 42 / 25, 43 / 25]
    bad_fingerprints = dict(fingerprints, qkv_compat_sha256="not-a-sha")
    try:
        tracking_key(norm_a, bad_fingerprints)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed source fingerprint was accepted")
    who = parse_who_independent("<answer>Target ID: 7, Time range: malformed</answer>", {7})
    when, projected = native.parse_native_output(
        "<answer>Target ID: 7, Time range: malformed</answer>",
        duration_s=norm_a["local_frame_count"] / norm_a["local_fps"],
        known_ids={7}, request=norm_a,
    )
    assert who["status"] == "valid" and when["status"] != "valid" and projected is None
    capped_who = parse_who_independent("<answer>Target ID: 7, Time range: 0 to 0.04</answer>", {7}, truncated=True)
    assert capped_who["status"] == "truncated_at_cap" and capped_who["candidate_id"] is None
    wrong_id = parse_who_independent("<answer>Target ID: 8, Time range: 0 to 0.04</answer>", {7})
    assert wrong_id["status"] == "unknown_candidate_id"
    after = forbidden_modules.intersection(sys.modules)
    if after:
        raise RuntimeError(f"CPU contracts imported forbidden model/video runtime: {sorted(after)}")
    return {
        "schema": "stvg-table1-native-instance-adapter-cpu-contracts-v1",
        "status": "pass",
        "checks": 10,
        "assertions": 15,
        "query_free_tracking_cache_identity": "pass",
        "dataset_specific_decode_policy_separation": "pass",
        "canonical_cache_binds_bounds_but_not_local_path": "pass",
        "tracker_asset_fingerprint_changes_cache_key": "pass",
        "physical_frame_map_preserves_absolute_source_clock": "pass",
        "fingerprint_schema_fail_closed": "pass",
        "who_independent_of_when": "pass",
        "truncated_who_fails_closed": "pass",
        "unknown_id_fails_closed": "pass",
        "forbidden_runtime_imports": [],
        "FFmpeg_or_media_decode_called": False,
        "model_or_GPU_called": False,
        "GT_read": False,
    }


def compare_existing_eight() -> dict[str, Any]:
    """Validate only old request/receipt/inventory metadata; no media or labels."""
    native = _native()
    base = ROOT / "artifacts/stvg_native_instance_spatial_opd_v1"
    roster = json.loads((base / "ROSTER_LOCK.json").read_text(encoding="utf-8"))
    barrier = json.loads((base / "TRACKING_GLOBAL_BARRIER.json").read_text(encoding="utf-8"))
    if roster.get("status") != "locked" or barrier.get("status") != "complete" or barrier.get("GT_read") is not False:
        raise RuntimeError("existing eight-case source/tracking metadata is not a complete GT-free lock")
    rows = [r for r in roster.get("records", []) if r.get("cohort") == "old8"]
    if len(rows) != 8 or barrier.get("count") != 8:
        raise RuntimeError("the existing old8 metadata roster/barrier is not exactly eight cases")
    barrier_rows = {row["case_id"]: row for row in barrier.get("files", [])}
    if len(barrier_rows) != 8:
        raise RuntimeError("the saved tracking barrier does not contain eight unique receipts")
    domains = {"hc2": 0, "vidstg": 0}
    unique_query_hashes: set[str] = set()
    binding_by_video: dict[tuple[str, str], tuple[str, str]] = {}
    for row in rows:
        case_id = row["case_id"]
        if row.get("GT_read") is True:
            raise RuntimeError("a roster row is explicitly marked as GT-read")
        req_path = (ROOT / row["request_path"]).resolve(strict=True)
        if not req_path.is_relative_to(ROOT) or _sha_file(req_path) != row["request_sha256"]:
            raise RuntimeError("saved old request path/hash differs from the locked roster")
        request = json.loads(req_path.read_text(encoding="utf-8"))
        normalized = native.validate_request(request)
        unique_query_hashes.add(normalized["query_sha256"])
        expected_barrier = barrier_rows.get(case_id)
        if not expected_barrier:
            raise RuntimeError("old tracking receipt is absent from the global barrier")
        accepted_path = (ROOT / expected_barrier["path"]).resolve(strict=True)
        if (not accepted_path.is_relative_to(base.resolve())
                or _sha_file(accepted_path) != expected_barrier["sha256"]):
            raise RuntimeError("old accepted tracking receipt differs from its global barrier")
        accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
        if (accepted.get("status") != "pass" or accepted.get("GT_read") is not False
                or accepted.get("input_request_sha256") != row["request_sha256"]
                or accepted.get("target_mode") != "all"):
            raise RuntimeError("old accepted tracking receipt does not match its GT-free query request")
        inventory_path = accepted_path.parent / "MASK_INVENTORY.json"
        if (not inventory_path.is_file()
                or _sha_file(inventory_path) != accepted.get("mask_inventory_sha256")):
            raise RuntimeError("old mask inventory receipt hash does not match accepted tracking metadata")
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        files = inventory.get("files")
        ids = inventory.get("known_ids")
        if (inventory.get("GT_read") is not False or not isinstance(files, list)
                or inventory.get("mask_count") != len(files) or not isinstance(ids, list)
                or ids != sorted(set(ids)) or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in ids)):
            raise RuntimeError("old mask inventory metadata is malformed or not GT-free")
        if any(not isinstance(f, Mapping) or set(f) != {"index", "path", "sha256", "bytes"}
               or not _is_sha(f.get("sha256")) or not isinstance(f.get("path"), str)
               or Path(f["path"]).is_absolute() or ".." in Path(f["path"]).parts for f in files):
            raise RuntimeError("old mask inventory contains an invalid path/hash entry")
        query_free = _query_free_request_binding(normalized)
        binding_sha = _sha_bytes(_canonical_json(query_free))
        cache_signature = (normalized["canonical_video_sha256"], normalized["frame_clock_sha256"])
        prior = binding_by_video.setdefault(cache_signature, (binding_sha, normalized["query_sha256"]))
        if prior[0] != binding_sha:
            raise RuntimeError("same-video/query variants do not preserve an identical tracking clock binding")
        domains[row["dataset"]] = domains.get(row["dataset"], 0) + 1
    if domains != {"hc2": 4, "vidstg": 4}:
        raise RuntimeError("existing old8 metadata does not contain four cases per domain")
    return {
        "schema": "stvg-table1-existing-eight-metadata-readback-v1",
        "status": "pass",
        "count": 8,
        "by_dataset": domains,
        "locked_request_hashes_verified": 8,
        "accepted_tracking_receipts_verified": 8,
        "mask_inventory_receipts_verified": 8,
        "distinct_query_hash_count": len(unique_query_hashes),
        "query_text_emitted": False,
        "source_media_opened_or_decoded": False,
        "mask_rle_opened_or_decoded": False,
        "GT_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true", help="run pure synthetic CPU contracts only")
    group.add_argument("--compare-existing-eight", action="store_true", help="read old request/cache metadata only")
    args = parser.parse_args()
    result = run_cpu_contracts() if args.self_test else compare_existing_eight()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
