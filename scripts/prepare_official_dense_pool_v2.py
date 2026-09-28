#!/usr/bin/env python3
"""Prepare a larger metadata-only official VidSTG pool.

The input is the official VidSTG ``test_annotations.json`` plus the local
VidOR validation annotation/video archives.  This is an intake manifest, not
an evaluation or a model run.  Query selection is source-hash based and does
not inspect metric values or use GT geometry as a preference.  GT is read
only for schema/track completeness checks; the runtime query whitelist never
contains it.  Full VidOR objects/classes/trajectories and the selected
VidSTG annotation rows are written to a separately marked scorer-only
sidecar for a later, post-prediction scorer.

The script intentionally does not extract the 3 GB video archive.  Each
selected query carries the archive member, CRC, decompressed-member SHA256,
and the metadata-only frame grid so a later runner can materialize or stream
the exact input without silently substituting another video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "official_dense_pool_v2"

VIDSTG_TEST = ROOT / "external" / "VidSTG-Dataset" / "annotations" / "test_annotations.json"
VIDOR_ANNOTATION_ZIP = ROOT / "downloads" / "vidor" / "validation-annotation.zip"
VIDOR_VIDEO_ZIP = ROOT / "downloads" / "vidor" / "validation-video.zip"
HC_OFFICIAL_METADATA = ROOT / "data" / "hcstvg2_official_metadata" / "val_v2.json"
HC_LOCAL_MEDIA = ROOT / "data" / "hcstvg2_confirm512" / "video"
VID_TARGET_DEV = ROOT / "data" / "vidstg_target_validation_v1" / "annotations" / "dev.json"
DEV_LOCK = ROOT / "artifacts" / "unanchored_fullspan_development_v1" / "lock.json"
SOURCE_INVENTORY = ROOT / "artifacts" / "marginal_confirmation_v1_intake" / "source_inventory.json"
V1_ROSTER = ROOT / "artifacts" / "dense_expansion_roster_v1" / "roster.json"

# These are provenance expectations from the already inspected official
# archives.  The actual SHA/size is always computed and compared below.
EXPECTED_ANNOTATION_BYTES = 17_212_872
EXPECTED_ANNOTATION_SHA256 = "65a06594c7735d8a0c384144ec2acb2227f715ccac2d03a6dee4a507de201516"
EXPECTED_VIDEO_BYTES = 3_056_763_915
EXPECTED_VIDEO_SHA256 = "40dd102b43ecd2ed70f2dc38ee08258315e25919d2f1fb8f43bb235740a83a25"

REQUIRED_VIDSTG_KEYS = frozenset(
    {
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
)
REQUIRED_VIDOR_KEYS = frozenset(
    {
        "video_id",
        "video_path",
        "frame_count",
        "fps",
        "width",
        "height",
        "subject/objects",
        "trajectories",
    }
)

# Keys that must not cross the later predictor boundary.  This set is
# intentionally recursive and includes semantic aliases used by older
# converted datasets.  The scorer-only sidecar is outside this boundary.
FORBIDDEN_RUNTIME_KEYS = frozenset(
    {
        "annotation",
        "annotations",
        "vidstg_annotation",
        "vidor_annotation",
        "questions",
        "subject",
        "objects",
        "trajectories",
        "trajectory",
        "relation_instances",
        "used_relation",
        "temporal_gt",
        "tube",
        "tube_start_frame",
        "tube_end_frame",
        "tube_start_time",
        "tube_end_time",
        "target_id",
        "target_ids",
        "gt",
        "gt_boxes",
        "gt_interval",
        "labels",
        "targets",
        "sparse_bbox_frame_keys",
        "reference_gt",
    }
)


class RosterContractError(RuntimeError):
    """Raised when an intake input cannot satisfy the fixed roster contract."""


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_vidstg_wrong_domain_support import (  # noqa: E402
    candidate_queries,
    find_video_member,
    load_vidor_annotations,
    trajectory_for_target,
    validate_vidor_record,
    validate_vidstg_annotation,
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def float_close(left: Any, right: Any, tolerance: float = 1e-3) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def raw_source_id(row: Mapping[str, Any]) -> str:
    """VidSTG/VidOR identity: raw original video id, never a filename prefix."""

    value = row.get("vid", row.get("video_id"))
    if value is None or not str(value):
        raise RosterContractError("empty Vid source id")
    return str(value)


def stable_source_hash(source: str, prefix: str = "dense-expansion-v2:") -> str:
    """Return the fixed source-ranking digest used by the roster."""

    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty string")
    return hashlib.sha256(f"{prefix}{source}".encode("utf-8")).hexdigest()


def rank_sources(sources: Iterable[str], prefix: str = "dense-expansion-v2:") -> list[dict[str, Any]]:
    """Sort sources by digest then source id, with no metric/GT preference."""

    unique = sorted({str(source) for source in sources})
    ranked = sorted(
        ((stable_source_hash(source, prefix), source) for source in unique),
        key=lambda item: (item[0], item[1]),
    )
    return [
        {"source": source, "source_hash": digest, "rank": rank}
        for rank, (digest, source) in enumerate(ranked)
    ]


def select_caption_candidates(candidates: Sequence[Mapping[str, Any]], max_per_source: int = 2) -> list[dict[str, Any]]:
    """Select at most N captions using stable annotation order only."""

    if max_per_source <= 0:
        raise ValueError("max_per_source must be positive")
    ordered = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda item: (
            int(item.get("annotation_index", 0)),
            int(item.get("query_index", 0)),
            str(item.get("query", "")),
        ),
    )
    return ordered[:max_per_source]


def assert_runtime_gt_free(value: Any, location: str = "query") -> None:
    """Reject annotation/GT keys recursively in runtime metadata."""

    if isinstance(value, Mapping):
        forbidden = sorted(FORBIDDEN_RUNTIME_KEYS.intersection(str(key) for key in value))
        if forbidden:
            raise RosterContractError(f"{location} contains forbidden GT fields: {forbidden}")
        for key, child in value.items():
            assert_runtime_gt_free(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_runtime_gt_free(child, f"{location}[{index}]")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise RosterContractError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RosterContractError(f"{name} is not an integer: {value!r}") from exc
    if result != float(value) or result <= 0:
        raise RosterContractError(f"{name} must be a positive integer: {value!r}")
    return result


def frame_ids_from_segment(
    *, frame_count: int, fps: float, start_frame: int, end_frame: int, sample_fps: int = 5, max_frames: int = 200
) -> list[int]:
    """Reproduce the official metadata-only VidSTG input frame sampler."""

    frame_count = _positive_int(frame_count, "frame_count")
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        raise RosterContractError(f"invalid fps={fps!r}")
    start_frame, end_frame = int(start_frame), int(end_frame)
    if sample_fps <= 0 or max_frames <= 0 or not 0 <= start_frame < end_frame <= frame_count:
        raise RosterContractError(
            f"invalid physical segment [{start_frame},{end_frame})/{frame_count}"
        )
    sampling_rate = float(sample_fps) / float(fps)
    if sampling_rate > 1:
        raise RosterContractError(f"sampler would upsample source fps={fps}")
    ids = [start_frame]
    for frame_id in range(start_frame, end_frame):
        if int(ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
            ids.append(frame_id)
    if len(ids) > max_frames:
        ids = [ids[(index * len(ids)) // max_frames] for index in range(max_frames)]
    if ids != sorted(set(ids)) or not ids or ids[0] < 0 or ids[-1] >= frame_count:
        raise RosterContractError("non-increasing or out-of-range frame grid")
    return ids


def find_member_info(archive: zipfile.ZipFile, video_path: str) -> zipfile.ZipInfo | None:
    member = find_video_member(archive, str(video_path))
    if member is None:
        return None
    try:
        return archive.getinfo(member)
    except KeyError:
        return None


def hash_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info, "r") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hc_source_id(filename: str | Path) -> str:
    """Strip one numeric HC clip prefix and retain the YouTube source id."""

    stem = Path(str(filename)).stem
    match = re.fullmatch(r"\d+_(.+)", stem)
    if match is None or not match.group(1):
        raise RosterContractError(f"HC filename has no numeric prefix: {filename!r}")
    return match.group(1)


def load_exclusion_sets() -> dict[str, Any]:
    """Load historical/current/target source ledgers without reading metrics."""

    inventory = read_json(SOURCE_INVENTORY)
    try:
        hc_hist = inventory["hcstvg2"]["historical_selected_or_reserved"]
        vid_hist = inventory["vidstg_vidor"]["historical_selected_or_reserved"]
        vid_abc: set[str] = set()
        for group in ("A", "B", "C"):
            vid_abc.update(str(value) for value in vid_hist["groups"][group]["query_source_ids"])
        vid_exposure = {str(value) for value in vid_hist["union"]["source_ids"]}
        hc_dev64 = {str(value) for value in hc_hist["dev64"]["source_ids"]}
    except (KeyError, TypeError) as exc:
        raise RosterContractError("unexpected source inventory schema") from exc
    if len(vid_abc) != 192 or len(vid_exposure) != 565 or len(hc_dev64) != 54:
        raise RosterContractError(
            f"historical source counts changed: Vid ABC={len(vid_abc)}, "
            f"Vid union={len(vid_exposure)}, HC dev64={len(hc_dev64)}"
        )

    dev_lock = read_json(DEV_LOCK)
    try:
        current_vid = {
            str(query["source"])
            for query in dev_lock["groups"]["hc_to_vid"]["queries"]
        }
        current_hc = {
            str(query["source"])
            for query in dev_lock["groups"]["vid_to_hc"]["queries"]
        }
    except (KeyError, TypeError) as exc:
        raise RosterContractError("development lock has no source queries") from exc
    if len(current_vid) != 32 or len(current_hc) != 32:
        raise RosterContractError(
            f"current development source counts changed: Vid={len(current_vid)}, HC={len(current_hc)}"
        )

    target_data = read_json(VID_TARGET_DEV)
    if not isinstance(target_data, Mapping) or not isinstance(target_data.get("videos"), list):
        raise RosterContractError("Vid target-dev annotation has no videos list")
    target_rows = target_data["videos"]
    target_vid = {str(row["original_video_id"]) for row in target_rows if isinstance(row, Mapping)}
    if len(target_rows) != 512 or len(target_vid) != 256:
        raise RosterContractError(
            f"target-dev counts changed: rows={len(target_rows)}, sources={len(target_vid)}"
        )

    return {
        "inventory_path": str(SOURCE_INVENTORY.resolve()),
        "inventory_sha256": sha256_file(SOURCE_INVENTORY),
        "dev_lock_path": str(DEV_LOCK.resolve()),
        "dev_lock_sha256": sha256_file(DEV_LOCK),
        "target_dev_path": str(VID_TARGET_DEV.resolve()),
        "target_dev_sha256": sha256_file(VID_TARGET_DEV),
        "vid_historical_abc": vid_abc,
        "vid_historical_exposure": vid_exposure,
        "vid_current_dev": current_vid,
        "vid_target_dev_reserved": target_vid,
        "hc_historical_dev64": hc_dev64,
        "hc_current_dev32": current_hc,
    }


def classify_vid_history(source: str, history: Mapping[str, Any]) -> str:
    if source in history["vid_historical_abc"]:
        return "confirmed_historical_development"
    if source in history["vid_historical_exposure"]:
        return "confirmed_historical_exposure"
    return "unknown"


def _interval(value: Any, name: str, frame_count: int) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        raise RosterContractError(f"{name} is not an object")
    try:
        start, end = int(value["begin_fid"]), int(value["end_fid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RosterContractError(f"{name} has no integer begin/end") from exc
    if not (0 <= start < end <= frame_count):
        raise RosterContractError(f"{name} outside physical frame range: {start},{end},{frame_count}")
    return start, end


def _target_ids(value: Any) -> set[int]:
    if not isinstance(value, list):
        return set()
    result: set[int] = set()
    for item in value:
        if isinstance(item, Mapping) and "tid" in item:
            try:
                result.add(int(item["tid"]))
            except (TypeError, ValueError):
                continue
    return result


def validate_annotation_candidate(
    annotation: Mapping[str, Any],
    annotation_index: int,
    raw: Mapping[str, Any] | None,
    archive: zipfile.ZipFile,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], str | None]:
    """Validate one raw row and return legal caption candidates plus reasons.

    The returned candidates contain GT-bearing internals only inside the
    in-memory validation structure.  They are never passed to the runtime
    query builder; selected internals are copied solely to the scorer sidecar.
    """

    reasons: list[str] = []
    rejected_candidates: list[dict[str, Any]] = []
    if not isinstance(annotation, Mapping):
        return [], ["vidstg_row_not_object"], [], None
    schema_errors = sorted(REQUIRED_VIDSTG_KEYS - set(annotation))
    reasons.extend(f"missing_vidstg_key:{key}" for key in schema_errors)
    source = str(annotation.get("vid", ""))
    if raw is None:
        reasons.append("missing_vidor_record")
        return [], sorted(set(reasons)), rejected_candidates, None
    raw_schema_errors = sorted(REQUIRED_VIDOR_KEYS - set(raw))
    reasons.extend(f"missing_vidor_key:{key}" for key in raw_schema_errors)
    if reasons:
        return [], sorted(set(reasons)), rejected_candidates, None

    try:
        ann_frame_count = int(annotation["frame_count"])
        raw_frame_count = int(raw["frame_count"])
    except (TypeError, ValueError):
        reasons.append("non_integer_frame_count")
        return [], sorted(set(reasons)), rejected_candidates, None
    if ann_frame_count != raw_frame_count:
        reasons.append("metadata_frame_count_mismatch")
    if str(raw.get("video_id")) != source:
        reasons.append("metadata_video_id_mismatch")
    if not float_close(annotation.get("fps"), raw.get("fps")):
        reasons.append("metadata_fps_mismatch")
    if int(annotation.get("width", -1)) != int(raw.get("width", -2)) or int(annotation.get("height", -1)) != int(raw.get("height", -2)):
        reasons.append("metadata_dimensions_mismatch")
    try:
        segment_start, segment_end = _interval(annotation["used_segment"], "used_segment", raw_frame_count)
        tube_start, tube_end = _interval(annotation["temporal_gt"], "temporal_gt", raw_frame_count)
    except RosterContractError as exc:
        reasons.append(str(exc))
        segment_start = segment_end = tube_start = tube_end = -1

    archive_member: str | None = None
    archive_info: zipfile.ZipInfo | None = None
    if isinstance(raw.get("video_path"), str):
        archive_info = find_member_info(archive, str(raw["video_path"]))
        if archive_info is None:
            reasons.append("missing_vidor_video_archive_member")
        else:
            archive_member = archive_info.filename
    else:
        reasons.append("missing_video_path")

    ann_targets = _target_ids(annotation.get("subject/objects"))
    raw_targets = _target_ids(raw.get("subject/objects"))
    if not ann_targets or not raw_targets:
        reasons.append("missing_target_catalog")

    legal: list[dict[str, Any]] = []
    captions = candidate_queries(dict(annotation), annotation_index)
    captions = [candidate for candidate in captions if candidate.get("query_type") == "caption"]
    if not captions:
        reasons.append("missing_legal_caption")
    for candidate in captions:
        candidate_reasons: list[str] = []
        target_id = int(candidate["target_id"])
        if target_id not in ann_targets:
            candidate_reasons.append("caption_target_missing_in_vidstg_catalog")
        if target_id not in raw_targets:
            candidate_reasons.append("caption_target_missing_in_vidor_catalog")
        if segment_start < 0 or tube_start < 0:
            candidate_reasons.append("invalid_physical_interval")
        else:
            trajectory = trajectory_for_target(dict(raw), target_id)
            if any(str(frame) not in trajectory for frame in range(tube_start, tube_end)):
                candidate_reasons.append("caption_target_track_gap_in_temporal_gt")
        if candidate_reasons:
            rejected_candidates.append(
                {
                    "query_index": int(candidate["query_index"]),
                    "query_type": "caption",
                    "reasons": sorted(set(candidate_reasons)),
                }
            )
            continue
        legal.append(
            {
                **candidate,
                "source": source,
                "frame_count": raw_frame_count,
                "fps": float(raw["fps"]),
                "width": int(raw["width"]),
                "height": int(raw["height"]),
                "video_path": str(raw["video_path"]),
                "video_archive_member": archive_member,
                "video_archive_member_size_bytes": int(archive_info.file_size) if archive_info else None,
                "video_archive_crc32": f"{int(archive_info.CRC) & 0xFFFFFFFF:08x}" if archive_info else None,
                "segment_start": segment_start,
                "segment_end": segment_end,
                "tube_start": tube_start,
                "tube_end": tube_end,
                # GT-bearing values stay in the internal candidate for the
                # scorer sidecar only; runtime metadata uses a strict builder.
                "target_id": target_id,
                "vidstg_annotation": dict(annotation),
                "vidor_record": dict(raw),
            }
        )
    if rejected_candidates:
        reasons.append("some_caption_candidates_invalid")
    return legal, sorted(set(reasons)), rejected_candidates, archive_member


def build_runtime_query(candidate: Mapping[str, Any], *, source_rank: int, source_hash: str, history_status: str, v1_overlap: bool) -> dict[str, Any]:
    """Build the only query object allowed to cross into a future runner."""

    frame_ids = frame_ids_from_segment(
        frame_count=int(candidate["frame_count"]),
        fps=float(candidate["fps"]),
        start_frame=int(candidate["segment_start"]),
        end_frame=int(candidate["segment_end"]),
        sample_fps=5,
        max_frames=200,
    )
    query = {
        "index": int(candidate["annotation_index"]),
        "query_id": f"vidstg-test-a{int(candidate['annotation_index']):05d}-q{int(candidate['query_index']):02d}-caption",
        "source": str(candidate["source"]),
        "kind": "vidstg",
        "query_type": "caption",
        "qtype": "declarative",
        "caption": str(candidate["query"]),
        # Raw VidOR video_id is the stable source identity.  It is deliberately
        # not replaced with a local enumerate index.
        "video_id": str(candidate["source"]),
        "original_video_id": str(candidate["source"]),
        "annotation_index": int(candidate["annotation_index"]),
        "caption_index": int(candidate["query_index"]),
        "frame_count": int(candidate["frame_count"]),
        "fps": float(candidate["fps"]),
        "width": int(candidate["width"]),
        "height": int(candidate["height"]),
        "start_frame": int(candidate["segment_start"]),
        "end_frame": int(candidate["segment_end"]),
        "frame_ids": frame_ids,
        "video_path": str(candidate["video_path"]),
        "video_archive_member": str(candidate["video_archive_member"]),
        "video_archive_member_size_bytes": int(candidate["video_archive_member_size_bytes"]),
        "video_archive_crc32": str(candidate["video_archive_crc32"]),
        "video_sha256": str(candidate["video_sha256"]),
        "video_sha_source": "VidOR validation-video.zip decompressed member bytes",
        "source_hash": str(source_hash),
        "source_hash_prefix": "dense-expansion-v2:",
        "source_rank": int(source_rank),
        "historical_exposure_status": str(history_status),
        # A local ledger cannot prove an untouched source, including sources
        # absent from the ledger.
        "historical_untouched": False,
        "v1_hc_to_vid_source_overlap": bool(v1_overlap),
        "video_materialization_required": True,
    }
    assert_runtime_gt_free(query)
    return query


def _archive_file_report(path: Path, expected_bytes: int, expected_sha: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    actual_sha = sha256_file(path)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if not infos:
            raise RosterContractError(f"archive has no members: {path}")
        return {
            "path": str(path.resolve()),
            "expected_size_bytes": int(expected_bytes),
            "actual_size_bytes": int(actual_bytes),
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "size_matches_expected": actual_bytes == expected_bytes,
            "sha256_matches_expected": actual_sha == expected_sha,
            "member_count": len(infos),
            "member_names_sha256": canonical_sha256(sorted(info.filename for info in infos)),
        }


def _load_v1_vid_sources() -> tuple[set[str], dict[str, Any]]:
    roster = read_json(V1_ROSTER)
    try:
        queries = roster["groups"]["hc_to_vid"]["queries"]
    except (KeyError, TypeError) as exc:
        raise RosterContractError("v1 roster has no hc_to_vid queries") from exc
    sources = {str(query["source"]) for query in queries}
    if len(sources) != 103:
        raise RosterContractError(f"v1 hc_to_vid source count changed: {len(sources)}")
    return sources, {"path": str(V1_ROSTER.resolve()), "sha256": sha256_file(V1_ROSTER), "source_count": len(sources)}


def build_hc_availability(history: Mapping[str, Any]) -> dict[str, Any]:
    """Inventory official HC metadata and local media; never select by GT."""

    metadata = read_json(HC_OFFICIAL_METADATA)
    if not isinstance(metadata, Mapping):
        raise RosterContractError("HC official metadata must be an object")
    official_names = sorted(str(name) for name in metadata)
    official_sources = {hc_source_id(name) for name in official_names}
    hc_excluded = set(history["hc_historical_dev64"]) | set(history["hc_current_dev32"])
    retained_names = [name for name in official_names if hc_source_id(name) not in hc_excluded]
    retained_sources = {hc_source_id(name) for name in retained_names}

    local_paths = sorted(path for path in HC_LOCAL_MEDIA.rglob("*") if path.is_file()) if HC_LOCAL_MEDIA.is_dir() else []
    local_names = {path.name for path in local_paths}
    overlap_names = sorted(set(official_names) & local_names)
    retained_overlap = sorted(set(retained_names) & local_names)
    by_source: dict[str, list[str]] = defaultdict(list)
    for name in retained_names:
        by_source[hc_source_id(name)].append(name)
    full, partial, absent = [], [], []
    missing_by_source: dict[str, list[str]] = {}
    for source in sorted(retained_sources):
        expected = sorted(by_source[source])
        present = sorted(set(expected) & local_names)
        missing = sorted(set(expected) - local_names)
        if not missing:
            full.append(source)
        elif present:
            partial.append(source)
        else:
            absent.append(source)
        if missing:
            missing_by_source[source] = missing
    return {
        "status": "availability_only",
        "annotation_path": str(HC_OFFICIAL_METADATA.resolve()),
        "annotation_sha256": sha256_file(HC_OFFICIAL_METADATA),
        "official_clip_count": len(official_names),
        "official_source_count": len(official_sources),
        "excluded_source_count": len(hc_excluded & official_sources),
        "excluded_source_ids": sorted(hc_excluded & official_sources),
        "retained_source_upper_bound": len(retained_sources),
        "retained_clip_count": len(retained_names),
        "local_media_root": str(HC_LOCAL_MEDIA.resolve()),
        "local_media_file_count": len(local_paths),
        "official_key_local_media_overlap_count": len(overlap_names),
        "retained_local_media_overlap_count": len(retained_overlap),
        "retained_sources_full_local": sorted(full),
        "retained_sources_partial_local": sorted(partial),
        "retained_sources_absent_local": sorted(absent),
        "retained_sources_full_count": len(full),
        "retained_sources_partial_count": len(partial),
        "retained_sources_absent_count": len(absent),
        "retained_missing_clip_count": sum(len(value) for value in missing_by_source.values()),
        "retained_missing_clip_names": sorted(name for values in missing_by_source.values() for name in values),
        "retained_missing_by_source": missing_by_source,
        "downloaded": False,
        "gpu_executed": False,
        "metric_based_selection": False,
        "geometry_used_for_selection": False,
        "note": "Official HC source upper bound only; absent local media is not downloaded or substituted.",
    }


def prepare(
    out: Path = OUT,
    *,
    max_sources: int = 384,
    max_captions_per_source: int = 2,
    script_path: Path | None = None,
) -> dict[str, Any]:
    """Build the official pool in a fresh output directory."""

    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise RosterContractError(f"refusing to overwrite non-empty output: {out}")
    if not 1 <= int(max_sources) <= 384:
        raise RosterContractError("max_sources must be in [1,384]")
    if not 1 <= int(max_captions_per_source) <= 2:
        raise RosterContractError("max_captions_per_source must be in [1,2]")
    script_path = Path(script_path or __file__).resolve()

    vidstg_data = read_json(VIDSTG_TEST)
    if not isinstance(vidstg_data, list):
        raise RosterContractError("official VidSTG test annotations must be a list")
    if len(vidstg_data) != 4610:
        raise RosterContractError(f"VidSTG test row count changed: {len(vidstg_data)}")

    annotation_sha = sha256_file(VIDSTG_TEST)
    annotation_zip_report = _archive_file_report(
        VIDOR_ANNOTATION_ZIP, EXPECTED_ANNOTATION_BYTES, EXPECTED_ANNOTATION_SHA256
    )
    video_zip_report = _archive_file_report(VIDOR_VIDEO_ZIP, EXPECTED_VIDEO_BYTES, EXPECTED_VIDEO_SHA256)
    for label, report in (("VidOR annotation", annotation_zip_report), ("VidOR video", video_zip_report)):
        if not report["size_matches_expected"] or not report["sha256_matches_expected"]:
            raise RosterContractError(
                f"{label} archive fingerprint mismatch: "
                f"size={report['actual_size_bytes']}/{report['expected_size_bytes']} "
                f"sha={report['actual_sha256']}/{report['expected_sha256']}"
            )
    history = load_exclusion_sets()
    v1_sources, v1_report = _load_v1_vid_sources()

    vidor_records = load_vidor_annotations(VIDOR_ANNOTATION_ZIP, None)
    if len(vidor_records) != 835:
        raise RosterContractError(f"VidOR annotation record count changed: {len(vidor_records)}")
    with zipfile.ZipFile(VIDOR_VIDEO_ZIP) as video_archive:
        source_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        invalid_records: list[dict[str, Any]] = []
        invalid_source_reasons: dict[str, Counter[str]] = defaultdict(Counter)
        validation_counters: Counter[str] = Counter()
        archive_member_by_source: dict[str, zipfile.ZipInfo] = {}

        for annotation_index, annotation in enumerate(vidstg_data):
            source = str(annotation.get("vid", "")) if isinstance(annotation, Mapping) else ""
            raw = vidor_records.get(source)
            legal, reasons, rejected, member = validate_annotation_candidate(
                annotation, annotation_index, raw, video_archive
            )
            if reasons:
                for reason in reasons:
                    validation_counters[reason] += 1
            if rejected:
                validation_counters["rejected_caption_candidates"] += len(rejected)
            if not legal:
                invalid_records.append(
                    {
                        "annotation_index": annotation_index,
                        "source": source,
                        "reasons": reasons or ["no_legal_caption_candidate"],
                        "rejected_caption_candidates": rejected,
                    }
                )
                for reason in reasons or ["no_legal_caption_candidate"]:
                    invalid_source_reasons[source][reason] += 1
                continue
            if member is not None:
                info = video_archive.getinfo(member)
                archive_member_by_source[source] = info
            for candidate in legal:
                source_candidates[source].append(candidate)

        # Source-level exclusions are applied after validity checks so invalid
        # metadata is visible rather than silently disappearing.
        exclusion_reason_sets: dict[str, list[str]] = {}
        all_vid_sources = set(source_candidates) | {str(row.get("vid", "")) for row in vidstg_data if isinstance(row, Mapping)}
        for source in sorted(all_vid_sources):
            reasons: list[str] = []
            if source in history["vid_historical_abc"]:
                reasons.append("historical_vidstg_development_ABC")
            if source in history["vid_current_dev"]:
                reasons.append("current_fullspan_vid_target_dev32")
            if source in history["vid_target_dev_reserved"]:
                reasons.append("reserved_vid_target_validation_dev256")
            if reasons:
                exclusion_reason_sets[source] = reasons

        excluded_sources_in_raw = set(exclusion_reason_sets) & all_vid_sources
        candidate_sources = sorted(set(source_candidates) - set(exclusion_reason_sets))
        ranked = rank_sources(candidate_sources)
        if len(ranked) < max_sources:
            raise RosterContractError(
                f"requested {max_sources} sources but only {len(ranked)} valid non-excluded sources exist"
            )
        selected_ranked = ranked[:max_sources]
        selected_sources = {entry["source"] for entry in selected_ranked}
        rank_by_source = {entry["source"]: entry for entry in selected_ranked}
        selected_candidates: list[dict[str, Any]] = []
        for source in sorted(selected_sources):
            selected_candidates.extend(
                select_caption_candidates(source_candidates[source], max_captions_per_source)
            )
        selected_candidates.sort(key=lambda item: (int(item["annotation_index"]), int(item["query_index"])))

        # Hash each selected unique compressed member's decompressed bytes once.
        member_sha_cache: dict[str, str] = {}
        for candidate in selected_candidates:
            member = str(candidate["video_archive_member"])
            if member not in member_sha_cache:
                member_sha_cache[member] = hash_zip_member(video_archive, video_archive.getinfo(member))
            candidate["video_sha256"] = member_sha_cache[member]

        runtime_queries: list[dict[str, Any]] = []
        for candidate in selected_candidates:
            source = str(candidate["source"])
            query = build_runtime_query(
                candidate,
                source_rank=int(rank_by_source[source]["rank"]),
                source_hash=str(rank_by_source[source]["source_hash"]),
                history_status=classify_vid_history(source, history),
                v1_overlap=source in v1_sources,
            )
            runtime_queries.append(query)

        if len(runtime_queries) != len(selected_candidates):
            raise RosterContractError("runtime query/candidate count mismatch")
        if len({int(query["index"]) for query in runtime_queries}) != len(runtime_queries):
            raise RosterContractError("selected raw annotation indices are not unique")
        for query in runtime_queries:
            assert_runtime_gt_free(query)

        # Build scorer-only provenance after the runtime query barrier.  This
        # sidecar retains the complete original VidOR record and selected raw
        # VidSTG rows, but cannot be accidentally consumed as a runtime query.
        sidecar_by_source: dict[str, dict[str, Any]] = {}
        for candidate in selected_candidates:
            source = str(candidate["source"])
            entry = sidecar_by_source.setdefault(
                source,
                {
                    "source": source,
                    "video_id": str(source),
                    "video_path": str(candidate["video_path"]),
                    "video_sha256": str(candidate["video_sha256"]),
                    "vidor_annotation": candidate["vidor_record"],
                    "selected_vidstg_rows": [],
                },
            )
            entry["selected_vidstg_rows"].append(candidate["vidstg_annotation"])
        sidecar = {
            "schema_version": "official_dense_pool_v2_scorer_only_v1",
            "scorer_only": True,
            "runtime_gt_free": False,
            "labels_read_for_validity_only": True,
            "not_used_for_source_hash_or_selection": True,
            "selection_policy": "source hash then annotation/query order; no metric or GT-geometry preference",
            "source_count": len(sidecar_by_source),
            "query_count": len(selected_candidates),
            "records": [sidecar_by_source[source] for source in sorted(sidecar_by_source)],
        }

    hc_availability = build_hc_availability(history)
    out.mkdir(parents=True, exist_ok=True)
    scorer_dir = out / "scorer_only"
    scorer_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = scorer_dir / "selected_vidor_annotations.json"
    invalid_path = out / "invalid_records.json"
    hc_path = out / "hc_availability.json"

    write_json(sidecar_path, sidecar)
    selected_annotation_indices = {int(candidate["annotation_index"]) for candidate in selected_candidates}
    invalid_annotation_indices = {int(record["annotation_index"]) for record in invalid_records}
    excluded_source_row_count = sum(
        index not in invalid_annotation_indices and str(row.get("vid", "")) in exclusion_reason_sets
        for index, row in enumerate(vidstg_data)
    )
    selected_source_row_count = len(selected_annotation_indices)
    unselected_legal_row_count = max(
        0,
        len(vidstg_data)
        - len(invalid_annotation_indices)
        - excluded_source_row_count
        - selected_source_row_count,
    )
    rows_accounted_for = (
        len(invalid_annotation_indices)
        + excluded_source_row_count
        + selected_source_row_count
        + unselected_legal_row_count
    )
    write_json(invalid_path, {
        "schema_version": "official_dense_pool_v2_invalid_records_v1",
        "records": invalid_records,
        "record_count": len(invalid_records),
        "by_reason": dict(sorted(Counter(reason for record in invalid_records for reason in record["reasons"]).items())),
        "source_reason_counts": {
            source: dict(sorted(counter.items())) for source, counter in sorted(invalid_source_reasons.items())
        },
        "selection_accounting": {
            "raw_annotation_rows": len(vidstg_data),
            "invalid_annotation_rows": len(invalid_annotation_indices),
            "excluded_source_rows": excluded_source_row_count,
            "selected_caption_rows": selected_source_row_count,
            "valid_retained_but_not_selected_rows": unselected_legal_row_count,
            "rows_accounted_for": rows_accounted_for,
            "all_rows_accounted_for": rows_accounted_for == len(vidstg_data),
        },
    })
    write_json(hc_path, hc_availability)

    source_status_counts = Counter(classify_vid_history(str(source), history) for source in selected_sources)
    v1_overlap_sources = sorted(selected_sources & v1_sources)
    excluded_reason_counts = Counter(reason for reasons in exclusion_reason_sets.values() for reason in reasons)
    source_exclusion_details = {
        source: {
            "reasons": exclusion_reason_sets[source],
            "raw_annotation_row_count": sum(1 for row in vidstg_data if str(row.get("vid", "")) == source),
        }
        for source in sorted(exclusion_reason_sets)
    }
    roster = {
        "schema_version": "official_dense_pool_v2",
        "status": "prepared",
        "passed": True,
        "gpu_executed": False,
        "model_forward_executed": False,
        "labels_loaded_runtime": False,
        "metric_based_selection": False,
        "geometry_used_for_selection": False,
        "not_untouched_confirmation": True,
        "purpose": "larger official VidSTG metadata-only intake for later cross-domain evaluation",
        "runtime_query_label_boundary": "runtime queries are GT-free; scorer-only sidecar is opened only after the full prediction barrier",
        "dataset": {
            "name": "VidSTG official test annotations joined to VidOR validation archives",
            "vidstg_annotation_path": str(VIDSTG_TEST.resolve()),
            "vidstg_annotation_sha256": annotation_sha,
            "vidor_annotation_zip": annotation_zip_report,
            "vidor_video_zip": video_zip_report,
            "video_materialization_required": True,
            "video_archive_member_sha_source": "decompressed ZIP member bytes",
        },
        "source_id_rule": "raw VidSTG vid / VidOR original video_id string; no filename-prefix parsing",
        "selection": {
            "hash_prefix": "dense-expansion-v2:",
            "rank_key": "sha256(hash_prefix + source), then source lexicographic",
            "max_sources": int(max_sources),
            "selected_source_count": len(selected_sources),
            "max_captions_per_source": int(max_captions_per_source),
            "selected_query_count": len(runtime_queries),
            "candidate_source_count_before_cap": len(ranked),
            "candidate_source_count_after_exclusions": len(candidate_sources),
            "legal_caption_candidate_count_before_cap": sum(len(source_candidates[source]) for source in candidate_sources),
            "selected_source_rank": selected_ranked,
            "source_history_status_counts": dict(sorted(source_status_counts.items())),
        },
        "raw_input_counts": {
            "vidstg_annotation_rows": len(vidstg_data),
            "vidstg_source_count": len(all_vid_sources),
            "vidor_annotation_record_count": len(vidor_records),
            "vidor_video_archive_member_count": int(video_zip_report["member_count"]),
            "valid_source_count_before_exclusion": len(set(source_candidates) | set(exclusion_reason_sets)),
            "invalid_annotation_record_count": len(invalid_records),
            "invalid_reason_counts": dict(sorted(validation_counters.items())),
        },
        "exclusions": {
            "historical_vidstg_development_ABC_source_count": len(history["vid_historical_abc"]),
            "current_fullspan_vid_target_dev32_source_count": len(history["vid_current_dev"]),
            "reserved_vid_target_validation_dev256_source_count": len(history["vid_target_dev_reserved"]),
            "exclusion_union_source_count_in_ledgers": len(
                history["vid_historical_abc"] | history["vid_current_dev"] | history["vid_target_dev_reserved"]
            ),
            "exclusion_union_source_count_present_in_raw_test": len(excluded_sources_in_raw),
            "excluded_source_reason_counts": dict(sorted(excluded_reason_counts.items())),
            "source_details": source_exclusion_details,
            "invalid_records_are_not_silent": True,
        },
        "overlap_with_v1": {
            **v1_report,
            "selected_source_overlap_count": len(v1_overlap_sources),
            "selected_source_overlap_ids": v1_overlap_sources,
            "v1_pool_is_reported_not_automatically_excluded": True,
        },
        "history_disclosure": {
            "source_inventory_path": history["inventory_path"],
            "source_inventory_sha256": history["inventory_sha256"],
            "current_dev_lock_path": history["dev_lock_path"],
            "current_dev_lock_sha256": history["dev_lock_sha256"],
            "target_dev_annotation_path": history["target_dev_path"],
            "target_dev_annotation_sha256": history["target_dev_sha256"],
            "selected_sources_unknown_history_are_still_historical_untouched_false": True,
            "historical_source_exposure_is_not_claimed_untouched": True,
        },
        "hc_official_availability": {
            "path": str(hc_path.resolve()),
            "sha256": sha256_file(hc_path),
            "retained_source_upper_bound": hc_availability["retained_source_upper_bound"],
            "local_media_present": False,
            "metadata_only": True,
        },
        "scorer_only_sidecar": {
            "path": str(sidecar_path.resolve()),
            "sha256": sha256_file(sidecar_path),
            "scorer_only": True,
            "runtime_gt_free": False,
            "contains_full_vidor_objects_classes_trajectories": True,
            "not_used_for_source_hash_or_selection": True,
        },
        "invalid_records_file": {
            "path": str(invalid_path.resolve()),
            "sha256": sha256_file(invalid_path),
            "record_count": len(invalid_records),
        },
        "groups": {
            "hc_to_vid": {
                "group": "hc_to_vid",
                "source": "hcstvg2",
                "target": "vidstg",
                "kind": "vidstg",
                "dataset_root": str(VIDOR_VIDEO_ZIP.resolve()),
                "media_kind": "archive_member_metadata_only",
                "annotation": str(VIDSTG_TEST.resolve()),
                "annotation_sha256": annotation_sha,
                "query_count": len(runtime_queries),
                "source_count": len(selected_sources),
                "queries": runtime_queries,
                "selection": "selected source hashes, then at most two captions/source in annotation order",
                "scorer_sidecar": str(sidecar_path.resolve()),
                "labels_loaded_runtime": False,
            }
        },
        "code_pins": {
            str(script_path): sha256_file(script_path),
            str((ROOT / "scripts" / "prepare_vidstg_wrong_domain_support.py").resolve()): sha256_file(ROOT / "scripts" / "prepare_vidstg_wrong_domain_support.py"),
            str(V1_ROSTER.resolve()): sha256_file(V1_ROSTER),
        },
        "provenance": {
            "prepared_by": str(script_path),
            "prepared_without_gpu": True,
            "prepared_without_model_forward": True,
            "source_hash_selection_is_deterministic": True,
            "query_order": "annotation_index then caption_index",
            "frame_grid": "5 FPS metadata-only used_segment grid, max 200, end exclusive",
            "gt_validation": "schema, metadata consistency, target catalog, and full target track coverage only; no geometry quality ranking",
        },
    }
    roster_path = out / "roster.json"
    write_json(roster_path, roster)
    receipt = {
        "schema_version": "official_dense_pool_v2_prepare_receipt_v1",
        "status": "prepared",
        "passed": True,
        "gpu_executed": False,
        "model_forward_executed": False,
        "labels_loaded_runtime": False,
        "metric_based_selection": False,
        "roster_path": str(roster_path.resolve()),
        "roster_sha256": sha256_file(roster_path),
        "scorer_only_sidecar_sha256": sha256_file(sidecar_path),
        "selected_source_count": len(selected_sources),
        "selected_query_count": len(runtime_queries),
        "invalid_record_count": len(invalid_records),
        "source_overlap_v1_count": len(v1_overlap_sources),
        "hc_retained_source_upper_bound": hc_availability["retained_source_upper_bound"],
        "archive_sha256": {
            "vidstg": annotation_sha,
            "vidor_annotation": annotation_zip_report["actual_sha256"],
            "vidor_video": video_zip_report["actual_sha256"],
        },
        "fresh_output": True,
    }
    receipt_path = out / "prepare_receipt.json"
    write_json(receipt_path, receipt)
    return receipt


def verify(out: Path = OUT) -> dict[str, Any]:
    """Verify the generated metadata-only roster without model work."""

    out = Path(out).resolve()
    roster_path, receipt_path = out / "roster.json", out / "prepare_receipt.json"
    roster = read_json(roster_path)
    receipt = read_json(receipt_path)
    if roster.get("schema_version") != "official_dense_pool_v2" or roster.get("status") != "prepared":
        raise RosterContractError("unexpected roster status/schema")
    if roster.get("gpu_executed") or roster.get("model_forward_executed") or roster.get("labels_loaded_runtime"):
        raise RosterContractError("roster claims forbidden execution")
    queries = roster.get("groups", {}).get("hc_to_vid", {}).get("queries")
    if not isinstance(queries, list) or not queries:
        raise RosterContractError("roster has no runtime queries")
    if len(queries) != int(roster["selection"]["selected_query_count"]):
        raise RosterContractError("roster query count mismatch")
    if len({int(query["index"]) for query in queries}) != len(queries):
        raise RosterContractError("roster has duplicate annotation indices")
    for query in queries:
        assert_runtime_gt_free(query)
        if not bool(query.get("historical_untouched") is False):
            raise RosterContractError("query has unsafe historical_untouched flag")
        if not isinstance(query.get("frame_ids"), list) or not query["frame_ids"]:
            raise RosterContractError("query has no frame grid")
    sidecar_path = Path(roster["scorer_only_sidecar"]["path"])
    invalid_path = Path(roster["invalid_records_file"]["path"])
    hc_path = Path(roster["hc_official_availability"]["path"])
    for path, expected in (
        (sidecar_path, roster["scorer_only_sidecar"]["sha256"]),
        (invalid_path, roster["invalid_records_file"]["sha256"]),
        (hc_path, roster["hc_official_availability"]["sha256"]),
    ):
        if sha256_file(path) != expected:
            raise RosterContractError(f"artifact hash mismatch: {path}")
    if receipt.get("roster_sha256") != sha256_file(roster_path):
        raise RosterContractError("prepare receipt roster SHA mismatch")
    if receipt.get("scorer_only_sidecar_sha256") != sha256_file(sidecar_path):
        raise RosterContractError("prepare receipt sidecar SHA mismatch")
    return {
        "status": "passed",
        "passed": True,
        "query_count": len(queries),
        "source_count": len({str(query["source"]) for query in queries}),
        "runtime_gt_free": True,
        "scorer_only_sidecar_separate": True,
        "gpu_executed": False,
        "roster_sha256": sha256_file(roster_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-sources", type=int, default=384)
    parser.add_argument("--max-captions-per-source", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = (
        prepare(args.out, max_sources=args.max_sources, max_captions_per_source=args.max_captions_per_source)
        if args.action == "prepare"
        else verify(args.out)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
