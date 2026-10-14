#!/usr/bin/env python3
"""Post-seal geometry coverage of the complete filtered native-instance pool.

The evaluator is inert at import and has two explicit entry points:
``--contracts`` runs synthetic CPU-only checks; ``--score-after-seal`` first
verifies the complete 37-file prediction seal and all source/tracking custody,
then calls the existing official truth reader.  The output is an anonymous
development diagnostic.  It is not raw detector recall, identity accuracy,
an oracle fit, or a reference-selection mechanism.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

# Make the executable path CPU-only even if a caller has CUDA installed.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / "artifacts/stvg_native_instance_spatial_opd_v1"
OUT_JSON = BASE / "POOL_CANDIDATE_COVERAGE_POSTSEAL.json"
APPENDIX = ROOT / "docs/STVG_NATIVE_INSTANCE_POOL_COVERAGE_APPENDIX.md"
THRESHOLDS = (0.25, 0.50)
MASK_NAME_RE = re.compile(r"mask_(\d+)\.json\Z")
RLE_DECODER_SOURCE = ROOT / "external/stvg_runtime/pycocotools/mask.py"
RLE_DECODER_EXTENSION = ROOT / "external/stvg_runtime/pycocotools/_mask.cpython-311-x86_64-linux-gnu.so"
RLE_DECODER_SOURCE_SHA256 = "4d4788f995971c97c5548f0bd19941f6ec1faaf5d0d7fb6ec489256ff872c617"
RLE_DECODER_EXTENSION_SHA256 = "2f690626cbf2ec65b0d0182fb7844be9101c00ef9b440ba8b94b6bbd79e869eb"
_GT_ACCESS_ENABLED = False


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _safe_root_path(relative: Any) -> Path:
    _require(isinstance(relative, str) and relative and "\\" not in relative,
             "sealed relative path is malformed")
    path = (ROOT / relative).resolve(strict=True)
    _require(path.is_relative_to(ROOT.resolve()) and path.is_file(),
             "sealed path escapes the project or is not a regular file")
    return path


def _load_mask_utils():
    """Load the repository's pinned CPU pycocotools runtime, without installing it."""
    runtime_root = ROOT / "external/stvg_runtime"
    _require(runtime_root.is_dir(), "the pinned local CPU mask runtime is unavailable")
    _require(_sha(RLE_DECODER_SOURCE) == RLE_DECODER_SOURCE_SHA256 and
             _sha(RLE_DECODER_EXTENSION) == RLE_DECODER_EXTENSION_SHA256,
             "the filtered-mask RLE decoder differs from the frozen CPU implementation")
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))
    from pycocotools import mask as mask_utils
    return mask_utils


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object in {Path(path).name}")
    return value


def _blocked_annotation_open(event: str, args: tuple[Any, ...]) -> None:
    """Block known annotation paths until every non-GT seal/custody check passes."""
    if _GT_ACCESS_ENABLED or event != "open" or not args:
        return
    raw = args[0]
    if not isinstance(raw, (str, bytes, os.PathLike)):
        return
    value = os.fsdecode(raw).replace("\\", "/").lower()
    blocked = (
        "/annos/", "/annotations/", "/anno_v2/", "/annotation/",
        "test_annotations.json", "valv2_proc.json", "val_v2.json",
        "train_v2.json", "/rows.json", "/summary.json",
    )
    if any(token in value for token in blocked):
        raise PermissionError("official annotations are inaccessible before the complete prediction seal")


def _mask_inclusive_bbox(mask: Any) -> list[int] | None:
    """Return a nonempty decoded mask's [xmin,ymin,xmax,ymax] inclusive box."""
    import numpy as np

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("decoded instance RLE must be a two-dimensional mask")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("decoded instance RLE is not binary")
    ys, xs = np.nonzero(array)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _inclusive_to_normalized_xyxy(box: Sequence[Any], width: int, height: int) -> list[float]:
    """Convert inclusive mask pixels to normalized continuous half-open XYXY."""
    if width <= 0 or height <= 0 or len(box) != 4:
        raise ValueError("mask box conversion needs positive dimensions and four coordinates")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in box):
        raise ValueError("inclusive mask coordinates must be integers")
    x0, y0, x1, y1 = [int(value) for value in box]
    if not (0 <= x0 <= x1 < width and 0 <= y0 <= y1 < height):
        raise ValueError("inclusive mask box is outside its decoded image")
    return [x0 / width, y0 / height, (x1 + 1) / width, (y1 + 1) / height]


def _normalized_xyxy_to_pixels(box: Sequence[Any], width: int, height: int) -> tuple[float, float, float, float]:
    """Map normalized half-open XYXY back to source-pixel edge coordinates."""
    if width <= 0 or height <= 0 or len(box) != 4:
        raise ValueError("normalized box conversion needs positive dimensions and four coordinates")
    values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("normalized box contains a non-finite coordinate")
    x0, y0, x1, y1 = values
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("normalized half-open box is invalid or outside the source image")
    return x0 * width, y0 * height, x1 * width, y1 * height


def _xyxy_iou(left: Sequence[Any], right: Sequence[Any]) -> float:
    """Independent continuous half-open XYXY box IoU, with no clipping."""
    if len(left) != 4 or len(right) != 4:
        raise ValueError("IoU requires two four-coordinate boxes")
    a = tuple(float(v) for v in left)
    b = tuple(float(v) for v in right)
    if not all(math.isfinite(v) for v in (*a, *b)):
        raise ValueError("IoU boxes must be finite")
    if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError("IoU boxes must have positive area")
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _coverage_stats(
    truth_by_physical: Mapping[int, Sequence[Any]],
    candidate_boxes_by_physical: Mapping[int, Sequence[Sequence[Any]]],
    canonical_physical_ids: set[int],
) -> dict[str, Any]:
    """Score every known GT frame; off-map and empty-pool frames contribute zero."""
    if not truth_by_physical:
        raise ValueError("coverage denominator cannot be empty")
    total = len(truth_by_physical)
    in_map = 0
    outside = 0
    in_map_empty = 0
    empty_all = 0
    sum_max = 0.0
    hit_025 = 0
    hit_050 = 0
    for raw_fid, gt_box in truth_by_physical.items():
        fid = int(raw_fid)
        if len(gt_box) != 4:
            raise ValueError("official GT box must have four coordinates")
        target = tuple(float(value) for value in gt_box)
        if not all(math.isfinite(value) for value in target) or target[2] <= target[0] or target[3] <= target[1]:
            raise ValueError("official GT box must be finite with positive area")
        if fid not in canonical_physical_ids:
            outside += 1
            best = 0.0
            empty_all += 1
        else:
            in_map += 1
            candidates = candidate_boxes_by_physical.get(fid, ())
            if not candidates:
                best = 0.0
                in_map_empty += 1
                empty_all += 1
            else:
                best = max(_xyxy_iou(candidate, target) for candidate in candidates)
        sum_max += best
        hit_025 += int(best >= 0.25)
        hit_050 += int(best >= 0.50)
    _require(in_map + outside == total, "GT frame partition is incomplete")
    _require(empty_all == outside + in_map_empty,
             "empty-pool GT count differs from off-map plus in-map empty frames")
    return {
        "known_gt_frames": total,
        "gt_frames_in_canonical_map": in_map,
        "gt_frames_outside_canonical_map": outside,
        "canonical_gt_frames_with_empty_filtered_pool": in_map_empty,
        "gt_frames_with_empty_filtered_pool": empty_all,
        "sum_max_bbox_iou": sum_max,
        "mean_max_bbox_iou": sum_max / total,
        "gt_frames_max_iou_ge_0_25": hit_025,
        "fraction_gt_frames_max_iou_ge_0_25": hit_025 / total,
        "gt_frames_max_iou_ge_0_50": hit_050,
        "fraction_gt_frames_max_iou_ge_0_50": hit_050 / total,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty case set")
    frames = sum(int(row["known_gt_frames"]) for row in rows)
    if frames <= 0:
        raise ValueError("aggregate GT frame denominator is empty")
    summed_iou = sum(float(row["sum_max_bbox_iou"]) for row in rows)
    hits_025 = sum(int(row["gt_frames_max_iou_ge_0_25"]) for row in rows)
    hits_050 = sum(int(row["gt_frames_max_iou_ge_0_50"]) for row in rows)
    return {
        "case_count": len(rows),
        "known_gt_frames": frames,
        "gt_frames_in_canonical_map": sum(int(row["gt_frames_in_canonical_map"]) for row in rows),
        "gt_frames_outside_canonical_map": sum(int(row["gt_frames_outside_canonical_map"]) for row in rows),
        "canonical_gt_frames_with_empty_filtered_pool": sum(
            int(row["canonical_gt_frames_with_empty_filtered_pool"]) for row in rows),
        "gt_frames_with_empty_filtered_pool": sum(
            int(row["gt_frames_with_empty_filtered_pool"]) for row in rows),
        "frame_weighted_mean_max_bbox_iou": summed_iou / frames,
        "case_macro_mean_max_bbox_iou": sum(float(row["mean_max_bbox_iou"]) for row in rows) / len(rows),
        "gt_frames_max_iou_ge_0_25": hits_025,
        "fraction_gt_frames_max_iou_ge_0_25": hits_025 / frames,
        "gt_frames_max_iou_ge_0_50": hits_050,
        "fraction_gt_frames_max_iou_ge_0_50": hits_050 / frames,
    }


def _decode_filtered_pool(
    *, case_id: str, width: int, height: int, request: Mapping[str, Any],
    inventory: Mapping[str, Any], mask_files: Mapping[int, Path],
) -> tuple[dict[int, list[tuple[float, float, float, float]]], int, int]:
    """Decode each filtered RLE sequentially; retain only boxes, never masks."""
    import numpy as np
    mask_utils = _load_mask_utils()

    frame_map = request.get("frame_map")
    frame_count = request.get("local_frame_count")
    _require(isinstance(frame_count, int) and not isinstance(frame_count, bool) and frame_count > 0 and
             isinstance(frame_map, list) and len(frame_map) == frame_count,
             f"canonical frame map is malformed for {case_id}")
    physical_by_local: dict[int, int] = {}
    physical_ids: list[int] = []
    for expected_local, row in enumerate(frame_map):
        _require(isinstance(row, Mapping) and row.get("local_frame_index") == expected_local,
                 f"canonical local frame clock is not dense for {case_id}")
        fid = row.get("physical_frame_id")
        _require(isinstance(fid, int) and not isinstance(fid, bool) and fid >= 0,
                 f"canonical physical frame ID is malformed for {case_id}")
        physical_by_local[expected_local] = int(fid)
        physical_ids.append(int(fid))
    _require(physical_ids == sorted(set(physical_ids)),
             f"canonical physical frame IDs are not strictly increasing for {case_id}")

    raw_known = inventory.get("known_ids")
    _require(isinstance(raw_known, list) and all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in raw_known),
        f"filtered inventory candidate IDs are malformed for {case_id}")
    expected_ids = set(raw_known)
    _require(len(expected_ids) == len(raw_known) and raw_known == sorted(expected_ids),
             f"filtered inventory candidate IDs are duplicated or unsorted for {case_id}")

    boxes_by_physical: dict[int, list[tuple[float, float, float, float]]] = {}
    observed_ids: set[int] = set()
    nonempty_instances = 0
    _require(set(mask_files) == set(range(frame_count)),
             f"filtered mask file set is not the complete canonical clock for {case_id}")
    for local_index in range(frame_count):
        frame = _read_json(mask_files[local_index])
        objects = frame.get("objects")
        _require(isinstance(objects, Mapping),
                 f"filtered mask frame has no object mapping for {case_id}/{local_index}")
        frame_boxes: list[tuple[float, float, float, float]] = []
        seen_local_ids: set[int] = set()
        # Sorting by the numeric tracker ID makes this deterministic.  Only a
        # single decoded bitmap exists at any point in the loop.
        try:
            keys = sorted(objects, key=lambda key: int(key))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"non-numeric filtered tracker ID for {case_id}/{local_index}") from exc
        for raw_id in keys:
            _require(isinstance(raw_id, str) and raw_id.isdecimal(),
                     f"filtered tracker ID is not a decimal string for {case_id}/{local_index}")
            candidate_id = int(raw_id)
            _require(candidate_id > 0 and candidate_id not in seen_local_ids,
                     f"filtered tracker ID is invalid or duplicated for {case_id}/{local_index}")
            seen_local_ids.add(candidate_id)
            observed_ids.add(candidate_id)
            obj = objects[raw_id]
            _require(isinstance(obj, Mapping) and isinstance(obj.get("rle"), Mapping),
                     f"filtered instance has no RLE for {case_id}/{local_index}")
            decoded = np.asarray(mask_utils.decode(obj["rle"]))
            _require(decoded.shape == (height, width),
                     f"filtered mask dimensions differ from source for {case_id}/{local_index}")
            if decoded.size:
                _require(int(decoded.min()) >= 0 and int(decoded.max()) <= 1,
                         f"filtered RLE is not binary for {case_id}/{local_index}")
            inclusive = _mask_inclusive_bbox(decoded)
            if inclusive is None:
                del decoded
                continue
            normalized = _inclusive_to_normalized_xyxy(inclusive, width, height)
            source_pixels = _normalized_xyxy_to_pixels(normalized, width, height)
            frame_boxes.append(source_pixels)
            nonempty_instances += 1
            # Do not retain or reference the decoded bitmap after this point.
            del decoded
        boxes_by_physical[physical_by_local[local_index]] = frame_boxes
    _require(observed_ids == expected_ids,
             f"decoded filtered tracker IDs differ from MASK_INVENTORY for {case_id}")
    return boxes_by_physical, len(expected_ids), nonempty_instances


def _verify_tracking_and_load_pools(locked: Mapping[str, Any], source_cases: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Verify request, accepted tracker output, inventory, all RLEs, and pool boxes."""
    from scripts import prepare_stvg_native_instance_references_v1 as refs
    from scripts import stvg_r1_native_teacher_v1 as native

    tracking_phase = locked["phases"]["TRACKING"]
    barrier_entries = {entry["case_id"]: entry for entry in tracking_phase["files"]}
    results: dict[str, dict[str, Any]] = {}
    for case_id in locked["case_ids"]:
        roster_row = locked["roster_by_case"][case_id]
        request_path = _safe_root_path(roster_row.get("request_path"))
        request_sha = roster_row.get("request_sha256")
        _require(isinstance(request_sha, str) and _sha(request_path) == request_sha,
                 f"canonical request digest differs from roster for {case_id}")
        request_raw = _read_json(request_path)
        validated_request = native.validate_request(request_raw)
        # Keep only the canonical frame clock needed by this audit. Drop the
        # query, video path, and other request text before any later processing.
        request = {
            "local_frame_count": validated_request["local_frame_count"],
            "frame_map": validated_request["frame_map"],
        }
        del validated_request, request_raw
        source = source_cases[case_id]
        _require(source.get("_source_case_sha256") == roster_row.get("source_case_sha256") and
                 source.get("dataset") == roster_row.get("dataset"),
                 f"source case differs from the locked roster for {case_id}")
        width, height = int(source["input"]["width"]), int(source["input"]["height"])
        _require(width > 0 and height > 0, f"source dimensions are invalid for {case_id}")
        _require(request.get("local_frame_count") == len(request.get("frame_map", [])),
                 f"canonical request frame count differs for {case_id}")
        teacher = locked["teacher_records"][case_id]
        receipt = locked["teacher_receipts"][case_id]
        teacher_binding = teacher.get("input_binding")
        _require(isinstance(teacher_binding, Mapping) and
                 teacher_binding.get("frame_map") == request.get("frame_map") and
                 teacher_binding.get("local_frame_count") == request.get("local_frame_count") and
                 receipt.get("request_sha256") == request_sha,
                 f"sealed teacher/request canonical frame binding differs for {case_id}")

        accepted_path = BASE / "TRACKING" / case_id / "ACCEPTED.json"
        inventory_path = BASE / "TRACKING" / case_id / "MASK_INVENTORY.json"
        entry = barrier_entries.get(case_id)
        _require(isinstance(entry, Mapping) and
                 entry.get("path") == str(accepted_path.relative_to(ROOT)) and
                 entry.get("sha256") == _sha(accepted_path),
                 f"tracking barrier does not bind the exact accepted case receipt for {case_id}")

        # This is the project's existing no-GT tracking custody verifier. It
        # checks ACCEPTED + MASK_INVENTORY hashes, every listed mask hash and
        # the permitted complete filtered-mask lineage.
        accepted, inventory, mask_files = refs._verify_tracking_case(
            roster_row, request, request_sha)
        _require(accepted.get("status") == "pass" and accepted.get("GT_read") is False and
                 accepted.get("target_mode") == "all" and
                 accepted.get("metadata_input") == "empty_mapping" and
                 accepted.get("input_request_sha256") == request_sha,
                 f"tracking acceptance is not the fixed all-category GT-free run for {case_id}")
        _require(_sha(inventory_path) == accepted.get("mask_inventory_sha256") and
                 inventory.get("GT_read") is False and
                 inventory.get("mask_count") == request.get("local_frame_count"),
                 f"MASK_INVENTORY custody or complete-clock count differs for {case_id}")

        boxes, track_count, nonempty_count = _decode_filtered_pool(
            case_id=case_id, width=width, height=height, request=request,
            inventory=inventory, mask_files=mask_files)
        results[case_id] = {
            "dataset": str(roster_row["dataset"]),
            "canonical_physical_ids": set(boxes),
            "candidate_boxes_by_physical": boxes,
            "filtered_candidate_track_count": track_count,
            "nonempty_filtered_instance_masks": nonempty_count,
            "source_dimensions": (width, height),
        }
    _require(set(results) == set(locked["case_ids"]) and len(results) == 8,
             "filtered-pool preflight did not cover the locked eight-case cohort")
    return results


def _summary_alias(case_id: str, dataset: str, position: int) -> str:
    prefix = "HC" if dataset == "hc2" else "V"
    return f"{prefix}{position:02d}"


def _anonymous_rows(locked: Mapping[str, Any], source_cases: Mapping[str, Mapping[str, Any]],
                    pools: Mapping[str, Mapping[str, Any]], truths: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    order_by_case: dict[str, int] = {}
    for dataset in ("hc2", "vidstg"):
        ids = sorted(case_id for case_id in locked["case_ids"]
                     if locked["roster_by_case"][case_id]["dataset"] == dataset)
        _require(len(ids) == 4, f"locked anonymous cohort must have four {dataset} cases")
        order_by_case.update({case_id: index + 1 for index, case_id in enumerate(ids)})

    rows: list[dict[str, Any]] = []
    for case_id in locked["case_ids"]:
        dataset = str(locked["roster_by_case"][case_id]["dataset"])
        source = source_cases[case_id]
        truth = truths[case_id].get("truth")
        _require(isinstance(truth, Mapping) and truth,
                 f"official reader returned no known GT frames for {case_id}")
        physical_truth: dict[int, Sequence[Any]] = {int(fid): box for fid, box in truth.items()}
        pool = pools[case_id]
        _require(pool["dataset"] == dataset,
                 f"filtered pool domain differs from the locked roster for {case_id}")
        stats = _coverage_stats(physical_truth, pool["candidate_boxes_by_physical"],
                                pool["canonical_physical_ids"])
        rows.append({
            "alias": _summary_alias(case_id, dataset, order_by_case[case_id]),
            "dataset": dataset,
            **stats,
            "filtered_candidate_track_count": int(pool["filtered_candidate_track_count"]),
            "nonempty_filtered_instance_masks": int(pool["nonempty_filtered_instance_masks"]),
        })
    rows.sort(key=lambda row: (row["dataset"], row["alias"]))
    _require(len(rows) == 8 and len({row["alias"] for row in rows}) == 8,
             "anonymous row set is incomplete or has duplicate aliases")
    return rows


def _markdown_appendix(payload: Mapping[str, Any], output_sha256: str) -> str:
    summary = payload["summary"]
    lines = [
        "# Filtered instance-pool geometric coverage",
        "",
        "This post-seal diagnostic measures how well the available filtered YOLOv12-X + SAM2.1 mask pool covers official ground-truth boxes geometrically. For each officially annotated physical frame, it takes the maximum axis-aligned box IoU between the GT box and any nonempty filtered instance mask box. A GT frame outside the canonical tracker clock or with no nonempty candidate mask contributes zero.",
        "",
        "The denominator is every physical frame returned by the official annotation reader for each locked case. The report includes both the count and rate at IoU ≥ 0.25 and IoU ≥ 0.50, plus the frame-weighted mean of per-frame maximum IoU. Inclusive mask pixels are converted to continuous half-open boxes with the required +1 right/bottom edge before pixel geometry is compared. No clipping or GT-based candidate selection is applied.",
        "",
        "All-category tracking used an empty metadata mapping, so category labels did not restrict the candidate pool. This is geometric availability of the filtered tracker output. It is **not raw YOLO detector recall**, does not assess semantic identity correctness, does not fit an oracle, and does not change any locked reference or prediction.",
        "",
        "## Per-case results",
        "",
        "| Case | Domain | GT frames | In clock | Outside clock | Empty pool | Mean max IoU | ≥0.25 | ≥0.50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["cases"]:
        lines.append(
            f"| {row['alias']} | {row['dataset']} | {row['known_gt_frames']} | "
            f"{row['gt_frames_in_canonical_map']} | {row['gt_frames_outside_canonical_map']} | "
            f"{row['gt_frames_with_empty_filtered_pool']} | {row['mean_max_bbox_iou']:.4f} | "
            f"{row['fraction_gt_frames_max_iou_ge_0_25']:.3f} "
            f"({row['gt_frames_max_iou_ge_0_25']}) | "
            f"{row['fraction_gt_frames_max_iou_ge_0_50']:.3f} "
            f"({row['gt_frames_max_iou_ge_0_50']}) |"
        )
    lines.extend([
        "",
        "## Cohort summary",
        "",
        "| Group | Cases | GT frames | Mean max IoU | ≥0.25 | ≥0.50 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    groups = [("All", summary["all"])] + [
        (dataset, summary["by_dataset"][dataset]) for dataset in ("hc2", "vidstg")
    ]
    for name, row in groups:
        lines.append(
            f"| {name} | {row['case_count']} | {row['known_gt_frames']} | "
            f"{row['frame_weighted_mean_max_bbox_iou']:.4f} | "
            f"{row['fraction_gt_frames_max_iou_ge_0_25']:.3f} "
            f"({row['gt_frames_max_iou_ge_0_25']}) | "
            f"{row['fraction_gt_frames_max_iou_ge_0_50']:.3f} "
            f"({row['gt_frames_max_iou_ge_0_50']}) |"
        )
    lines.extend([
        "",
        f"Machine-readable anonymous rows: `POOL_CANDIDATE_COVERAGE_POSTSEAL.json` (SHA-256 `{output_sha256}`).",
        "",
    ])
    return "\n".join(lines)


def _atomic_write_once(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing coverage output: {path.name}")
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"coverage output appeared while writing: {path.name}")
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def _run_sealed_coverage() -> dict[str, Any]:
    """Check all prediction/tracking custody, then read GT once and summarize."""
    global _GT_ACCESS_ENABLED
    if not callable(sys.addaudithook):
        raise RuntimeError("Python audit hooks are unavailable")
    sys.addaudithook(_blocked_annotation_open)

    from scripts import score_stvg_native_instance_postseal_v1 as scorer
    # This is the fail-closed 37-file barrier. It must complete before source
    # loading and, critically, before the official GT reader is reachable.
    locked = scorer._load_locked_inputs()
    source_cases = scorer._load_source_cases(locked)
    pools = _verify_tracking_and_load_pools(locked, source_cases)

    # The only GT access point follows every global seal, runtime, source,
    # request, ACCEPTED, MASK_INVENTORY, and individual filtered-RLE check.
    from scripts import review_stvg_sparse_r1_v2 as official_scoring
    _GT_ACCESS_ENABLED = True
    try:
        truth_by_case, provenance = official_scoring._truths_after_seal(source_cases)
    finally:
        _GT_ACCESS_ENABLED = False
    _require(set(truth_by_case) == set(locked["case_ids"]),
             "official GT reader did not return the exact sealed eight-case roster")

    rows = _anonymous_rows(locked, source_cases, pools, truth_by_case)
    summary = {
        "all": _aggregate(rows),
        "by_dataset": {
            dataset: _aggregate([row for row in rows if row["dataset"] == dataset])
            for dataset in ("hc2", "vidstg")
        },
    }
    _require(summary["all"]["case_count"] == 8 and
             summary["by_dataset"]["hc2"]["case_count"] == 4 and
             summary["by_dataset"]["vidstg"]["case_count"] == 4,
             "anonymous aggregate no longer represents four cases per domain")

    # The provenance map contains annotation file paths; bind its canonical
    # digest but never expose the paths or raw coordinates in the output.
    provenance_sha = hashlib.sha256(_canonical_bytes(provenance)).hexdigest()
    payload: dict[str, Any] = {
        "schema": "stvg-native-instance-filtered-pool-coverage-v1",
        "status": "completed_cpu_postseal",
        "GT_read": True,
        "GT_read_after_complete_prediction_seal": True,
        "coverage_estimand": "per annotated physical frame, maximum bbox IoU over all nonempty filtered tracker masks",
        "thresholds": list(THRESHOLDS),
        "denominator_definition": "all physical frames returned by the official sealed-source annotation reader; frames outside the canonical tracker clock count as zero",
        "candidate_policy": "complete all-category filtered tracking pool; no category metadata filtering; no GT-based candidate selection",
        "claim_boundaries": {
            "raw_yolo_detector_recall": "not_measured",
            "semantic_identity_correctness": "not_measured",
            "oracle_fit_or_reference_change": False,
            "prediction_selection_or_change": False,
        },
        "bindings": {
            "global_prediction_barrier_sha256": locked["seal_sha256"],
            "roster_lock_sha256": locked["roster_sha256"],
            "tracking_global_barrier_sha256": locked["phase_hashes"]["TRACKING"],
            "postseal_scorer_sha256": _sha(Path(scorer.__file__)),
            "official_truth_reader_sha256": _sha(Path(official_scoring.__file__)),
            "official_gt_reader_source_sha256s": {
                "review_stvg_sparse_r1_v2.py": _sha(ROOT / "scripts/review_stvg_sparse_r1_v2.py"),
                "score_stvg_opd_p1_v1.py": _sha(ROOT / "scripts/score_stvg_opd_p1_v1.py"),
                "diagnose_tastvg_pipeline_cpu_v1.py": _sha(ROOT / "scripts/diagnose_tastvg_pipeline_cpu_v1.py"),
            },
            "coverage_tool_sha256": _sha(Path(__file__)),
            "filtered_rle_decoder_source_sha256": RLE_DECODER_SOURCE_SHA256,
            "filtered_rle_decoder_extension_sha256": RLE_DECODER_EXTENSION_SHA256,
            "official_gt_provenance_sha256": provenance_sha,
        },
        "summary": summary,
        "cases": rows,
        "private_case_ids_exported": False,
        "source_paths_exported": False,
        "queries_or_teacher_text_exported": False,
        "boxes_or_raw_annotation_labels_exported": False,
        "images_or_masks_exported": False,
    }
    _require(not OUT_JSON.exists() and not APPENDIX.exists(),
             "coverage outputs already exist; immutable write-once policy forbids replacement")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    json_sha = hashlib.sha256(encoded).hexdigest()
    appendix = _markdown_appendix(payload, json_sha).encode("utf-8")
    _atomic_write_once(OUT_JSON, encoded)
    try:
        _atomic_write_once(APPENDIX, appendix)
    except BaseException:
        # Do not silently delete a successfully written receipt. The JSON is
        # the immutable authoritative result; report the missing appendix.
        raise
    try:
        import torch
        _require(not torch.cuda.is_initialized(), "coverage scoring unexpectedly initialized CUDA")
    except ImportError:
        pass
    return {
        "status": payload["status"],
        "cases": len(rows),
        "known_gt_frames": summary["all"]["known_gt_frames"],
        "coverage_json": str(OUT_JSON.relative_to(ROOT)),
        "coverage_json_sha256": json_sha,
        "appendix": str(APPENDIX.relative_to(ROOT)),
        "GT_read_after_seal": True,
        "GPU_initialized": False,
    }


def _synthetic_contracts() -> dict[str, Any]:
    """Small in-memory geometry/RLE checks; never reads project data or GT."""
    import numpy as np
    mask_utils = _load_mask_utils()

    identical = _xyxy_iou((0, 0, 2, 2), (0, 0, 2, 2))
    one_third = _xyxy_iou((0, 0, 2, 2), (1, 0, 3, 2))
    disjoint = _xyxy_iou((0, 0, 1, 1), (2, 2, 3, 3))
    if not (identical == 1.0 and abs(one_third - 1.0 / 3.0) < 1e-15 and disjoint == 0.0):
        raise AssertionError("synthetic XYXY IoU fixture failed")

    binary = np.zeros((3, 5), dtype=np.uint8)
    binary[1:3, 3:5] = 1
    rle = mask_utils.encode(np.asfortranarray(binary))
    decoded = mask_utils.decode(rle)
    inclusive = _mask_inclusive_bbox(decoded)
    normalized = _inclusive_to_normalized_xyxy(inclusive, width=5, height=3)
    source_pixels = _normalized_xyxy_to_pixels(normalized, width=5, height=3)
    expected = (3.0, 1.0, 5.0, 3.0)
    if inclusive != [3, 1, 4, 2] or any(abs(a - b) > 1e-12 for a, b in zip(source_pixels, expected)):
        raise AssertionError("synthetic inclusive/+1/normalized/source-pixel roundtrip failed")
    if abs(_xyxy_iou(source_pixels, expected) - 1.0) > 1e-15:
        raise AssertionError("synthetic non-square source dimension IoU failed")

    empty = np.zeros((2, 4), dtype=np.uint8)
    empty_decoded = mask_utils.decode(mask_utils.encode(np.asfortranarray(empty)))
    if _mask_inclusive_bbox(empty_decoded) is not None:
        raise AssertionError("empty RLE must not create a candidate box")

    stats = _coverage_stats(
        {100: [0, 0, 1, 1], 101: expected, 102: [2, 2, 4, 4]},
        {101: [expected], 102: []},
        {101, 102},
    )
    if (stats["known_gt_frames"] != 3 or stats["gt_frames_outside_canonical_map"] != 1 or
            stats["canonical_gt_frames_with_empty_filtered_pool"] != 1 or
            stats["gt_frames_with_empty_filtered_pool"] != 2 or
            abs(stats["mean_max_bbox_iou"] - 1.0 / 3.0) > 1e-15 or
            stats["gt_frames_max_iou_ge_0_25"] != 1 or
            stats["gt_frames_max_iou_ge_0_50"] != 1):
        raise AssertionError("synthetic denominator/off-clock/empty-pool fixture failed")

    # Empty candidate pools and no canonical overlap must never create a false
    # positive or divide-by-zero success; the denominator remains known GT.
    none = _coverage_stats({4: [0, 0, 1, 1]}, {}, set())
    if none["mean_max_bbox_iou"] != 0.0 or none["fraction_gt_frames_max_iou_ge_0_25"] != 0.0:
        raise AssertionError("synthetic empty/off-map pool fixture failed")

    return {
        "status": "pass",
        "synthetic_contracts": 5,
        "known_xyxy_iou_cases": 3,
        "rle_decode_cases": 2,
        "non_square_mask_roundtrip_cases": 1,
        "denominator_cases_including_off_clock_and_empty_pool": 2,
        "actual_seal_or_GT_read": False,
        "GPU_or_model_calls": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", action="store_true",
                        help="run synthetic in-memory CPU geometry/RLE contracts only")
    parser.add_argument("--score-after-seal", action="store_true",
                        help="after the complete 37-file seal and tracking custody pass, read GT and score coverage")
    args = parser.parse_args(argv)
    if args.contracts:
        print(json.dumps(_synthetic_contracts(), ensure_ascii=False, indent=2))
        return 0
    if not args.score_after_seal:
        parser.error("actual GT access is disabled unless --score-after-seal is supplied")
    print(json.dumps(_run_sealed_coverage(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
