"""Geometry and sparse-anchor contracts for Sparse STVG-R1.

This module is deliberately independent of model, video, GT, and deployment
code.  It has three separate interfaces:

* ``map_candidate_geometry`` clips an existing normalized-xyxy dense candidate
  coordinatewise into the unit image, rejects any box made degenerate, and
  retains the original geometry binding and both pre/post-map hashes.
* ``build_verified_sparse_anchors`` accepts only a separately sealed
  observation-verification receipt.  The dense candidate is never itself a
  dense target; only at most four receipt-bound positions are exposed to the
  student adapter.
* ``action_transform_logits_cxcywh`` maps Gaussian action logits through
  sigmoid(cx, cy, w, h), xyxy conversion, and fixed [0,1] clipping.  The
  Gaussian log probability remains a density in the original z space; no
  clipped-box density or Jacobian equivalence is claimed.

All hashes use canonical JSON over CPU/Python values.  The caller remains
responsible for pinning the source case and the bytes of the real verification
receipt before using any returned sparse anchors for an update.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import numbers
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


MAPPED_CANDIDATE_SCHEMA = "stvg_hierarchical_mapped_candidate_v1"
ANCHOR_RECEIPT_SCHEMA = "stvg_hierarchical_sparse_anchor_receipt_v1"
SPARSE_ANCHOR_SET_SCHEMA = "stvg_hierarchical_verified_sparse_anchor_set_v1"
ACTION_TRANSFORM = "sigmoid_logit_cxcywh_to_xyxy_then_coordinatewise_clamp_0_1_v1"
MAX_VERIFIED_ANCHORS = 4


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _jsonable(value: Any, *, where: str = "value") -> Any:
    """Convert JSON-like CPU values without interpreting opaque metadata."""
    if hasattr(value, "detach") and hasattr(value, "device"):
        if getattr(value.device, "type", None) != "cpu":
            raise ValueError(f"{where} must be CPU-resident; GPU copying is not part of this module")
        if bool(getattr(value, "requires_grad", False)):
            raise ValueError(f"{where} must be detached")
        value = value.detach().tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        value = value.tolist()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{where} contains a non-finite number")
        return result
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where} mapping keys must be strings")
            converted[key] = _jsonable(item, where=f"{where}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, where=f"{where}[]") for item in value]
    raise ValueError(f"{where} is not canonical JSON data")


def canonical_sha256(value: Any, *, where: str = "value") -> str:
    encoded = json.dumps(_jsonable(value, where=where), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    return _sha256_bytes(encoded)


def _require_sha256(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _integer_frame_ids(values: Any, *, name: str) -> list[int]:
    converted = _jsonable(values, where=name)
    if not isinstance(converted, list) or not converted:
        raise ValueError(f"{name} must be a nonempty list of physical frame IDs")
    result: list[int] = []
    for item in converted:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{name} must contain nonnegative integer physical frame IDs")
        result.append(item)
    if len(set(result)) != len(result) or any(b <= a for a, b in zip(result, result[1:])):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return result


def _xyxy_rows(boxes: Any, *, expected_count: int | None = None,
               name: str = "boxes_xyxy_normalized") -> list[list[float]]:
    rows = _jsonable(boxes, where=name)
    if not isinstance(rows, list) or (expected_count is not None and len(rows) != expected_count):
        raise ValueError(f"{name} must have one xyxy row for each source frame")
    result: list[list[float]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"{name}[{index}] must be [x1,y1,x2,y2]")
        coords: list[float] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name}[{index}] coordinates must be real numbers")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name}[{index}] coordinates must be finite")
            coords.append(value)
        result.append(coords)
    return result


def map_candidate_geometry(candidate: Mapping[str, Any], frame_ids: Any, *,
                           expected_source_case_sha256: str) -> dict[str, Any]:
    """Return one legal mapped dense geometry while preserving its opaque binding.

    Required input fields are ``candidate_id``, ``boxes_xyxy_normalized``, and
    ``opaque_geometry_binding``. Coordinates are clipped independently, in
    their existing normalized-xyxy coordinate system.  The input binding is
    copied without interpreting or rewriting it. A zero/non-positive-area
    frame rejects the entire candidate; coordinates are never swapped or
    epsilon-expanded.
    """
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate must be a mapping")
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id must be a nonempty stable string")
    source_sha = _require_sha256(expected_source_case_sha256,
                                 "expected_source_case_sha256")
    ids = _integer_frame_ids(frame_ids, name="frame_ids")
    if candidate.get("frame_ids") is not None:
        candidate_ids = _integer_frame_ids(candidate["frame_ids"],
                                           name="candidate.frame_ids")
        if candidate_ids != ids:
            raise ValueError("candidate frame_ids do not match the locked source frame axis")
    if candidate.get("source_case_sha256") is not None and candidate.get("source_case_sha256") != source_sha:
        raise ValueError("candidate belongs to a different cold source case")
    raw_boxes = _xyxy_rows(candidate.get("boxes_xyxy_normalized"),
                           expected_count=len(ids))
    if "opaque_geometry_binding" not in candidate:
        raise ValueError("candidate must preserve an explicit opaque_geometry_binding")
    opaque = _jsonable(copy.deepcopy(candidate["opaque_geometry_binding"]),
                       where="opaque_geometry_binding")
    original_binding_sha = canonical_sha256(opaque, where="opaque_geometry_binding")

    mapped: list[list[float]] = []
    clipped_coordinates = 0
    clipped_boxes = 0
    for position, (x1, y1, x2, y2) in enumerate(raw_boxes):
        clipped = [min(1.0, max(0.0, value)) for value in (x1, y1, x2, y2)]
        clipped_coordinates += sum(a != b for a, b in zip((x1, y1, x2, y2), clipped))
        clipped_boxes += int(clipped != [x1, y1, x2, y2])
        mx1, my1, mx2, my2 = clipped
        if mx2 <= mx1 or my2 <= my1:
            raise ValueError(f"candidate becomes zero-area or reversed after clipping at position {position}")
        mapped.append(clipped)

    original_geometry_sha = canonical_sha256(
        {"frame_ids": ids, "boxes_xyxy_normalized": raw_boxes},
        where="original_candidate_geometry")
    mapped_geometry_sha = canonical_sha256(
        {"frame_ids": ids, "boxes_xyxy_normalized": mapped},
        where="mapped_candidate_geometry")
    return {
        "schema": MAPPED_CANDIDATE_SCHEMA,
        "candidate_id": candidate_id,
        "source_case_sha256": source_sha,
        "frame_ids": ids,
        "boxes_xyxy_normalized": mapped,
        "original_geometry_sha256": original_geometry_sha,
        "mapped_geometry_sha256": mapped_geometry_sha,
        "opaque_geometry_binding": opaque,
        "opaque_geometry_binding_sha256": original_binding_sha,
        "mapping_policy": "coordinatewise_clamp_[0,1]_then_reject_nonpositive_area",
        "clipped_coordinate_count": clipped_coordinates,
        "clipped_box_count": clipped_boxes,
        "GT_read": False,
    }


def map_and_deduplicate_candidates(candidates: Sequence[Mapping[str, Any]],
                                   frame_ids: Any, *,
                                   expected_source_case_sha256: str) -> dict[str, Any]:
    """Map candidates first, then stable-deduplicate by mapped geometry hash.

    The first candidate in input order is canonical. Every duplicate's original
    ID, pre-map geometry hash and opaque binding remains in the canonical
    candidate's ``equivalent_original_bindings`` list and the returned alias
    table. Zero-area candidates fail closed and are returned as rejections.
    """
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("candidates must be a sequence")
    expected_source_sha = _require_sha256(expected_source_case_sha256,
                                         "expected_source_case_sha256")
    mapped_rows: list[dict[str, Any]] = []
    rejections: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id") if isinstance(candidate, Mapping) else None
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("every candidate requires a nonempty candidate_id")
        if candidate_id in seen_ids:
            raise ValueError("candidate_id values must be unique before geometry deduplication")
        seen_ids.add(candidate_id)
        candidate_source_sha = candidate.get("source_case_sha256")
        if candidate_source_sha is not None and candidate_source_sha != expected_source_sha:
            raise ValueError("candidate belongs to a different cold source case")
        try:
            mapped_rows.append(map_candidate_geometry(
                candidate, frame_ids,
                expected_source_case_sha256=expected_source_case_sha256))
        except ValueError as exc:
            rejections.append({"candidate_id": candidate_id,
                               "reason_code": "invalid_or_zero_area_after_mapping",
                               "detail": str(exc)})

    unique: list[dict[str, Any]] = []
    by_hash: dict[str, dict[str, Any]] = {}
    aliases: list[dict[str, str]] = []
    for row in mapped_rows:
        geometry_sha = row["mapped_geometry_sha256"]
        prior = by_hash.get(geometry_sha)
        if prior is None:
            row["equivalent_original_bindings"] = [{
                "candidate_id": row["candidate_id"],
                "original_geometry_sha256": row["original_geometry_sha256"],
                "opaque_geometry_binding": copy.deepcopy(row["opaque_geometry_binding"]),
                "opaque_geometry_binding_sha256": row["opaque_geometry_binding_sha256"],
            }]
            by_hash[geometry_sha] = row
            unique.append(row)
            continue
        if row["boxes_xyxy_normalized"] != prior["boxes_xyxy_normalized"] or row["frame_ids"] != prior["frame_ids"]:
            raise RuntimeError("mapped geometry SHA collision")
        member = {
            "candidate_id": row["candidate_id"],
            "original_geometry_sha256": row["original_geometry_sha256"],
            "opaque_geometry_binding": copy.deepcopy(row["opaque_geometry_binding"]),
            "opaque_geometry_binding_sha256": row["opaque_geometry_binding_sha256"],
        }
        prior["equivalent_original_bindings"].append(member)
        aliases.append({"duplicate_candidate_id": row["candidate_id"],
                        "canonical_candidate_id": prior["candidate_id"],
                        "mapped_geometry_sha256": geometry_sha})
    return {
        "schema": "stvg_hierarchical_mapped_candidate_catalog_v1",
        "source_case_sha256": expected_source_sha,
        "frame_ids": _integer_frame_ids(frame_ids, name="frame_ids"),
        "mapping_policy": "coordinatewise_clamp_[0,1]_then_reject_nonpositive_area",
        "dedup_policy": "stable_first_candidate_by_mapped_geometry_sha256",
        "input_count": len(candidates),
        "mapped_valid_count": len(mapped_rows),
        "unique_mapped_count": len(unique),
        "rejected_count": len(rejections),
        "duplicate_count": len(aliases),
        "candidates": unique,
        "aliases": aliases,
        "rejections": rejections,
        "GT_read": False,
    }


@dataclass(frozen=True)
class VerifiedSparseAnchorSet:
    """Receipt-validated supervision; dense geometry is intentionally absent."""

    status: str
    source_case_sha256: str
    candidate_id: str
    frame_ids: tuple[int, ...]
    positions: tuple[int, ...]
    physical_frame_ids: tuple[int, ...]
    boxes_xyxy_normalized: tuple[tuple[float, float, float, float], ...]
    observation_receipt_sha256s: tuple[str, ...]
    pixel_sha256s: tuple[str, ...]
    receipt_file_sha256: str | None
    receipt_canonical_sha256: str | None
    ignored_unverified_count: int = 0

    @property
    def count(self) -> int:
        return len(self.positions)

    @property
    def supervision_mask(self) -> tuple[bool, ...]:
        active = set(self.positions)
        return tuple(index in active for index in range(len(self.frame_ids)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SPARSE_ANCHOR_SET_SCHEMA,
            "status": self.status,
            "source_case_sha256": self.source_case_sha256,
            "candidate_id": self.candidate_id,
            "frame_ids": list(self.frame_ids),
            "positions": list(self.positions),
            "physical_frame_ids": list(self.physical_frame_ids),
            "boxes_xyxy_normalized": [list(row) for row in self.boxes_xyxy_normalized],
            "observation_receipt_sha256s": list(self.observation_receipt_sha256s),
            "pixel_sha256s": list(self.pixel_sha256s),
            "receipt_file_sha256": self.receipt_file_sha256,
            "receipt_canonical_sha256": self.receipt_canonical_sha256,
            "ignored_unverified_count": self.ignored_unverified_count,
            "supervision_mask": list(self.supervision_mask),
            "GT_read": False,
        }


def _empty_anchor_set(candidate: Mapping[str, Any], frame_ids: list[int], *,
                      status: str, receipt_file_sha256: str | None = None,
                      receipt_canonical_sha256: str | None = None,
                      ignored_unverified_count: int = 0) -> VerifiedSparseAnchorSet:
    return VerifiedSparseAnchorSet(
        status=status,
        source_case_sha256=candidate["source_case_sha256"],
        candidate_id=candidate["candidate_id"],
        frame_ids=tuple(frame_ids), positions=(), physical_frame_ids=(),
        boxes_xyxy_normalized=(), observation_receipt_sha256s=(), pixel_sha256s=(),
        receipt_file_sha256=receipt_file_sha256,
        receipt_canonical_sha256=receipt_canonical_sha256,
        ignored_unverified_count=ignored_unverified_count)


def build_verified_sparse_anchors(mapped_candidate: Mapping[str, Any],
                                  frame_ids: Any,
                                  receipt: Mapping[str, Any] | None) -> VerifiedSparseAnchorSet:
    """Create at most four trainable anchors from an independently verified receipt.

    Receipt contract (``ANCHOR_RECEIPT_SCHEMA``):

    ``status='verified'``, ``GT_read=False``, source/candidate/original/mapped/
    opaque-binding hashes, a pinned ``receipt_file_sha256``, and up to four
    ``verified_observations``. Each observation binds one ``frame_position``
    and ``physical_frame_id`` to a positive-area normalized-xyxy box, the exact
    mapped-candidate box hash, a source pixel SHA, and an observation-receipt
    SHA. A valid observation box must equal that candidate's mapped box at the
    same position. The returned object contains only those sparse rows.

    Missing or explicitly unavailable receipts return an empty set, never a
    dense target. Malformed or mismatched verified receipts fail closed.
    """
    if not isinstance(mapped_candidate, Mapping) or mapped_candidate.get("schema") != MAPPED_CANDIDATE_SCHEMA:
        raise ValueError("mapped_candidate must come from map_candidate_geometry")
    ids = _integer_frame_ids(frame_ids, name="frame_ids")
    candidate_ids = _integer_frame_ids(mapped_candidate.get("frame_ids"),
                                       name="mapped_candidate.frame_ids")
    if ids != candidate_ids:
        raise ValueError("receipt frame axis differs from mapped candidate frame axis")
    dense = _xyxy_rows(mapped_candidate.get("boxes_xyxy_normalized"),
                       expected_count=len(ids), name="mapped_candidate.boxes_xyxy_normalized")
    source_sha = _require_sha256(mapped_candidate.get("source_case_sha256"),
                                 "mapped_candidate.source_case_sha256")
    candidate_id = mapped_candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("mapped candidate has no stable candidate_id")
    original_sha = _require_sha256(mapped_candidate.get("original_geometry_sha256"),
                                   "mapped_candidate.original_geometry_sha256")
    mapped_sha = _require_sha256(mapped_candidate.get("mapped_geometry_sha256"),
                                 "mapped_candidate.mapped_geometry_sha256")
    binding_sha = _require_sha256(mapped_candidate.get("opaque_geometry_binding_sha256"),
                                  "mapped_candidate.opaque_geometry_binding_sha256")
    if canonical_sha256(mapped_candidate.get("opaque_geometry_binding"),
                        where="mapped_candidate.opaque_geometry_binding") != binding_sha:
        raise ValueError("opaque original geometry binding hash does not match its preserved value")
    if mapped_candidate.get("mapping_policy") != "coordinatewise_clamp_[0,1]_then_reject_nonpositive_area":
        raise ValueError("mapped candidate does not declare the fixed coordinate-clamp policy")
    for position, (x1, y1, x2, y2) in enumerate(dense):
        if x1 < 0.0 or y1 < 0.0 or x2 > 1.0 or y2 > 1.0 or x2 <= x1 or y2 <= y1:
            raise ValueError(f"mapped candidate is outside the legal unit-square domain at {position}")
    recomputed_mapped_sha = canonical_sha256(
        {"frame_ids": ids, "boxes_xyxy_normalized": dense}, where="mapped_candidate_geometry")
    if recomputed_mapped_sha != mapped_sha:
        raise ValueError("mapped candidate geometry hash does not match its coordinates")
    if receipt is None:
        return _empty_anchor_set(mapped_candidate, ids, status="no_verification_receipt")
    if not isinstance(receipt, Mapping) or receipt.get("schema") != ANCHOR_RECEIPT_SCHEMA:
        raise ValueError("unexpected sparse-anchor receipt schema")
    if receipt.get("source_case_sha256") != source_sha or receipt.get("candidate_id") != candidate_id:
        raise ValueError("sparse-anchor receipt belongs to a different source or candidate")
    if receipt.get("original_geometry_sha256") != original_sha:
        raise ValueError("sparse-anchor receipt is not bound to the original geometry")
    if receipt.get("mapped_geometry_sha256") != mapped_sha:
        raise ValueError("sparse-anchor receipt is not bound to the mapped geometry")
    if receipt.get("opaque_geometry_binding_sha256") != binding_sha:
        raise ValueError("sparse-anchor receipt is not bound to the opaque original geometry binding")
    if receipt.get("GT_read") is not False:
        raise ValueError("sparse-anchor receipt must declare GT_read=false")
    receipt_file_sha = _require_sha256(receipt.get("receipt_file_sha256"),
                                       "receipt.receipt_file_sha256")
    receipt_canonical_sha = canonical_sha256(receipt, where="sparse-anchor-receipt")
    if receipt.get("status") != "verified":
        return _empty_anchor_set(mapped_candidate, ids, status="receipt_not_verified",
                                 receipt_file_sha256=receipt_file_sha,
                                 receipt_canonical_sha256=receipt_canonical_sha)
    observations = receipt.get("verified_observations")
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise ValueError("verified_observations must be a list")
    if len(observations) > MAX_VERIFIED_ANCHORS:
        raise ValueError("at most four verified sparse anchors may supervise an update")

    rows: list[tuple[int, int, tuple[float, float, float, float], str, str]] = []
    seen_positions: set[int] = set()
    seen_frames: set[int] = set()
    for obs in observations:
        if not isinstance(obs, Mapping) or obs.get("verification_status") != "verified":
            raise ValueError("every entry in verified_observations must be verified")
        position = obs.get("frame_position")
        physical_id = obs.get("physical_frame_id")
        if (isinstance(position, bool) or not isinstance(position, int)
                or position < 0 or position >= len(ids)):
            raise ValueError("anchor frame_position is outside the mapped source grid")
        if isinstance(physical_id, bool) or not isinstance(physical_id, int):
            raise ValueError("anchor physical_frame_id must be an integer")
        if ids[position] != physical_id:
            raise ValueError("anchor position and physical frame ID do not identify the same frame")
        if position in seen_positions or physical_id in seen_frames:
            raise ValueError("verified sparse anchors must use unique source positions")
        seen_positions.add(position)
        seen_frames.add(physical_id)
        pixel_sha = _require_sha256(obs.get("pixel_sha256"), "observation.pixel_sha256")
        observation_sha = _require_sha256(obs.get("observation_receipt_sha256"),
                                          "observation.observation_receipt_sha256")
        box = _xyxy_rows([obs.get("box_xyxy_normalized")], expected_count=1,
                         name="observation.box_xyxy_normalized")[0]
        x1, y1, x2, y2 = box
        if x1 < 0.0 or y1 < 0.0 or x2 > 1.0 or y2 > 1.0 or x2 <= x1 or y2 <= y1:
            raise ValueError("verified anchor boxes must be positive-area normalized xyxy")
        candidate_box = dense[position]
        if box != candidate_box:
            raise ValueError("verified anchor box differs from mapped candidate geometry at that frame")
        candidate_box_sha = canonical_sha256(candidate_box, where="mapped_candidate_anchor_box")
        if obs.get("candidate_box_sha256") != candidate_box_sha:
            raise ValueError("verified observation does not bind the mapped candidate box bytes")
        rows.append((position, physical_id, tuple(box), observation_sha, pixel_sha))
    rows.sort(key=lambda row: row[0])
    return VerifiedSparseAnchorSet(
        status="verified" if rows else "no_verified_observations",
        source_case_sha256=source_sha,
        candidate_id=candidate_id,
        frame_ids=tuple(ids),
        positions=tuple(row[0] for row in rows),
        physical_frame_ids=tuple(row[1] for row in rows),
        boxes_xyxy_normalized=tuple(row[2] for row in rows),
        observation_receipt_sha256s=tuple(row[3] for row in rows),
        pixel_sha256s=tuple(row[4] for row in rows),
        receipt_file_sha256=receipt_file_sha,
        receipt_canonical_sha256=receipt_canonical_sha,
        ignored_unverified_count=0)


def verified_anchors_for_student_core(anchor_set: VerifiedSparseAnchorSet) -> list[dict[str, Any]]:
    """Convert only verified anchors to the existing student core's K4 record shape.

    The student core consumes normalized cxcywh ``box`` values. This converter
    emits no rows for unverified positions and never expands sparse anchors to
    the dense candidate tube.
    """
    import torch

    _validate_anchor_set(anchor_set)
    result = []
    for position, frame_id, box, obs_sha, pixel_sha in zip(
            anchor_set.positions, anchor_set.physical_frame_ids,
            anchor_set.boxes_xyxy_normalized,
            anchor_set.observation_receipt_sha256s, anchor_set.pixel_sha256s):
        x1, y1, x2, y2 = box
        cxcywh = [(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1]
        result.append({
            "position": position,
            "physical_frame_id": frame_id,
            "box": torch.tensor(cxcywh, dtype=torch.float32, device="cpu"),
            "weight": 1.0,
            "candidate_id": anchor_set.candidate_id,
            "observation_receipt_sha256": obs_sha,
            "pixel_sha256": pixel_sha,
            "receipt_file_sha256": anchor_set.receipt_file_sha256,
            "verified_observation": True,
        })
    return result


def _validate_anchor_set(anchor_set: VerifiedSparseAnchorSet) -> None:
    if not isinstance(anchor_set, VerifiedSparseAnchorSet):
        raise ValueError("anchor_set must be a VerifiedSparseAnchorSet")
    if anchor_set.status not in ("verified", "no_verified_observations",
                                 "no_verification_receipt", "receipt_not_verified"):
        raise ValueError("anchor set has an unknown status")
    _require_sha256(anchor_set.source_case_sha256, "anchor_set.source_case_sha256")
    if not isinstance(anchor_set.candidate_id, str) or not anchor_set.candidate_id:
        raise ValueError("anchor set has no stable candidate ID")
    ids = _integer_frame_ids(anchor_set.frame_ids, name="anchor_set.frame_ids")
    k = anchor_set.count
    if k > MAX_VERIFIED_ANCHORS:
        raise ValueError("verified anchor set exceeds K<=4")
    if not (len(anchor_set.physical_frame_ids) == len(anchor_set.boxes_xyxy_normalized)
            == len(anchor_set.observation_receipt_sha256s) == len(anchor_set.pixel_sha256s) == k):
        raise ValueError("anchor set fields have inconsistent sparse lengths")
    if tuple(sorted(set(anchor_set.positions))) != anchor_set.positions:
        raise ValueError("anchor positions must be sorted and unique")
    for position, frame_id, box, observation_sha, pixel_sha in zip(
            anchor_set.positions, anchor_set.physical_frame_ids,
            anchor_set.boxes_xyxy_normalized,
            anchor_set.observation_receipt_sha256s, anchor_set.pixel_sha256s):
        if position < 0 or position >= len(ids) or ids[position] != frame_id:
            raise ValueError("anchor position/frame binding is invalid")
        if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
            raise ValueError("anchor box must be finite xyxy")
        x1, y1, x2, y2 = map(float, box)
        if x1 < 0.0 or y1 < 0.0 or x2 > 1.0 or y2 > 1.0 or x2 <= x1 or y2 <= y1:
            raise ValueError("anchor box must be positive-area normalized xyxy")
        _require_sha256(observation_sha, "anchor.observation_receipt_sha256")
        _require_sha256(pixel_sha, "anchor.pixel_sha256")
    if k > 0:
        _require_sha256(anchor_set.receipt_file_sha256, "anchor_set.receipt_file_sha256")
        _require_sha256(anchor_set.receipt_canonical_sha256, "anchor_set.receipt_canonical_sha256")
        if anchor_set.status != "verified":
            raise ValueError("non-verified anchor set cannot contain trainable positions")
    elif anchor_set.status == "verified":
        raise ValueError("verified anchor set must contain at least one observation")


def gather_verified_anchor_predictions(prediction_xyxy: Any,
                                       anchor_set: VerifiedSparseAnchorSet) -> dict[str, Any]:
    """Gather only receipt-verified prediction rows; all other frames are gradient-free.

    Input is a torch tensor shaped ``[..., N, 4]`` in normalized xyxy. The
    returned prediction is an ``index_select`` over only verified positions;
    compute the downstream objective from that tensor to ensure no dense
    unverified frame can receive teacher-label gradient.
    """
    import torch

    _validate_anchor_set(anchor_set)
    if not isinstance(prediction_xyxy, torch.Tensor) or prediction_xyxy.ndim < 2:
        raise ValueError("prediction_xyxy must be a tensor shaped [..., N, 4]")
    if prediction_xyxy.shape[-1] != 4 or prediction_xyxy.shape[-2] != len(anchor_set.frame_ids):
        raise ValueError("prediction tensor does not match the anchor set frame axis")
    if not torch.isfinite(prediction_xyxy).all():
        raise ValueError("prediction_xyxy contains non-finite coordinates")
    positions = torch.tensor(anchor_set.positions, dtype=torch.long,
                             device=prediction_xyxy.device)
    selected = prediction_xyxy.index_select(-2, positions)
    targets = prediction_xyxy.new_tensor(anchor_set.boxes_xyxy_normalized).detach()
    return {"predictions": selected, "targets": targets,
            "positions": positions, "count": anchor_set.count,
            "supervision_mask": anchor_set.supervision_mask,
            "receipt_file_sha256": anchor_set.receipt_file_sha256,
            "GT_read": False}


def action_transform_logits_cxcywh(z: Any) -> Any:
    """Map action logits ``z`` deterministically to legal unit-square xyxy boxes.

    ``a(z) = clamp_xyxy( xyxy(sigmoid(z_cxcywh)), 0, 1 )``. Clamping is
    coordinatewise and fixed. Any box that becomes zero-area is rejected,
    including sigmoid underflow at a boundary; it is never epsilon-expanded.
    This operation is a many-to-one action transform and does not redefine the
    density used to score the Gaussian action.
    """
    import torch

    if not isinstance(z, torch.Tensor) or z.ndim < 1 or z.shape[-1] != 4:
        raise ValueError("z must be a torch tensor with last dimension [logit_cx,logit_cy,logit_w,logit_h]")
    if not torch.isfinite(z).all():
        raise ValueError("action logits must be finite")
    cx, cy, width, height = torch.sigmoid(z).unbind(dim=-1)
    raw = torch.stack((cx - width * 0.5, cy - height * 0.5,
                       cx + width * 0.5, cy + height * 0.5), dim=-1)
    clipped = raw.clamp(min=0.0, max=1.0)
    degenerate = (clipped[..., 2] <= clipped[..., 0]) | (clipped[..., 3] <= clipped[..., 1])
    if bool(degenerate.any()):
        raise ValueError("action box becomes zero-area after fixed unit-square clipping")
    return clipped


def diagonal_gaussian_log_prob_z(z: Any, mean_z: Any, log_std_z: Any) -> Any:
    """Evaluate diagonal Gaussian log probability in the original z space.

    No xyxy transform, clipping Jacobian, or clipped-box density is applied.
    ``z`` must be the same pre-transform random variable sampled from this
    Gaussian; callers must not pass the returned xyxy box here.
    """
    import torch

    if not all(isinstance(value, torch.Tensor) for value in (z, mean_z, log_std_z)):
        raise ValueError("z, mean_z and log_std_z must be torch tensors")
    try:
        z_b, mu_b, log_std_b = torch.broadcast_tensors(z, mean_z, log_std_z)
    except RuntimeError as exc:
        raise ValueError("Gaussian z parameters are not broadcast-compatible") from exc
    if not torch.isfinite(z_b).all() or not torch.isfinite(mu_b).all() or not torch.isfinite(log_std_b).all():
        raise ValueError("Gaussian z parameters must be finite")
    inv_std = torch.exp(-log_std_b)
    standardized = (z_b - mu_b) * inv_std
    per_coordinate = -0.5 * (standardized.square() + 2.0 * log_std_b + math.log(2.0 * math.pi))
    return per_coordinate.sum(dim=-1)


def run_cpu_contracts() -> dict[str, Any]:
    """Run synthetic CPU contracts only; this function never opens project data."""
    import torch

    assert torch.tensor([1.0], device="cpu").device.type == "cpu"
    source_sha = "a" * 64
    ids = [10, 11, 12, 13, 14, 15]
    original_a = [[-0.2, 0.1, 0.8, 0.9], [0.1, 0.1, 0.8, 0.9],
                  [0.1, 0.1, 0.8, 0.9], [0.1, 0.1, 0.8, 0.9],
                  [0.1, 0.1, 0.8, 0.9], [0.1, 0.1, 0.8, 0.9]]
    original_b = [[0.0, 0.1, 0.8, 0.9], [0.1, 0.1, 0.8, 0.9],
                  [0.1, 0.1, 0.8, 0.9], [0.1, 0.1, 0.8, 0.9],
                  [0.1, 0.1, 0.8, 0.9], [0.1, 0.1, 0.8, 0.9]]
    ca = {"candidate_id": "C-A", "boxes_xyxy_normalized": original_a,
          "opaque_geometry_binding": {"legacy_sha256": "b" * 64, "ordinal": 1}}
    cb = {"candidate_id": "C-B", "boxes_xyxy_normalized": original_b,
          "opaque_geometry_binding": {"legacy_sha256": "c" * 64, "ordinal": 2}}
    mapped_a = map_candidate_geometry(ca, ids, expected_source_case_sha256=source_sha)
    assert mapped_a["boxes_xyxy_normalized"][0] == [0.0, 0.1, 0.8, 0.9]
    assert mapped_a["clipped_coordinate_count"] == 1 and mapped_a["clipped_box_count"] == 1
    assert mapped_a["original_geometry_sha256"] != mapped_a["mapped_geometry_sha256"]
    assert mapped_a["opaque_geometry_binding"] == ca["opaque_geometry_binding"]
    assert mapped_a["opaque_geometry_binding_sha256"] == canonical_sha256(ca["opaque_geometry_binding"])
    dedup = map_and_deduplicate_candidates([ca, cb], ids,
                                           expected_source_case_sha256=source_sha)
    assert dedup["unique_mapped_count"] == 1 and dedup["duplicate_count"] == 1
    assert len(dedup["candidates"][0]["equivalent_original_bindings"]) == 2
    assert dedup["candidates"][0]["boxes_xyxy_normalized"] == mapped_a["boxes_xyxy_normalized"]

    zero_area = {"candidate_id": "zero", "boxes_xyxy_normalized": [[1.1, 0.2, 1.4, 0.7]] * 6,
                 "opaque_geometry_binding": {"legacy_sha256": "d" * 64}}
    rejected = map_and_deduplicate_candidates([zero_area], ids,
                                              expected_source_case_sha256=source_sha)
    assert rejected["rejected_count"] == 1 and rejected["unique_mapped_count"] == 0

    mapped = map_candidate_geometry(
        {"candidate_id": "anchors", "boxes_xyxy_normalized": original_b,
         "opaque_geometry_binding": {"legacy_sha256": "e" * 64}}, ids,
        expected_source_case_sha256=source_sha)
    positions = [1, 4]
    observations = []
    for position in positions:
        box = mapped["boxes_xyxy_normalized"][position]
        observations.append({
            "verification_status": "verified",
            "frame_position": position,
            "physical_frame_id": ids[position],
            "box_xyxy_normalized": box,
            "candidate_box_sha256": canonical_sha256(box),
            "pixel_sha256": hashlib.sha256(f"pixel-{position}".encode()).hexdigest(),
            "observation_receipt_sha256": hashlib.sha256(f"observation-{position}".encode()).hexdigest(),
        })
    receipt = {
        "schema": ANCHOR_RECEIPT_SCHEMA,
        "status": "verified",
        "source_case_sha256": source_sha,
        "candidate_id": mapped["candidate_id"],
        "original_geometry_sha256": mapped["original_geometry_sha256"],
        "mapped_geometry_sha256": mapped["mapped_geometry_sha256"],
        "opaque_geometry_binding_sha256": mapped["opaque_geometry_binding_sha256"],
        "receipt_file_sha256": "f" * 64,
        "verified_observations": observations,
        "GT_read": False,
    }
    anchors = build_verified_sparse_anchors(mapped, ids, receipt)
    assert anchors.count == 2 and anchors.positions == (1, 4)
    assert anchors.supervision_mask == (False, True, False, False, True, False)
    core_rows = verified_anchors_for_student_core(anchors)
    assert len(core_rows) == 2 and all(row["verified_observation"] for row in core_rows)
    no_receipt = build_verified_sparse_anchors(mapped, ids, None)
    assert no_receipt.count == 0 and no_receipt.status == "no_verification_receipt"
    no_rows = verified_anchors_for_student_core(no_receipt)
    assert no_rows == []
    empty_predictions = torch.zeros((len(ids), 4), dtype=torch.float32, device="cpu",
                                    requires_grad=True)
    empty_gather = gather_verified_anchor_predictions(empty_predictions, no_receipt)
    assert empty_gather["count"] == 0 and empty_gather["predictions"].shape == (0, 4)
    empty_gather["predictions"].sum().backward()
    assert torch.equal(empty_predictions.grad, torch.zeros_like(empty_predictions))

    predictions = torch.zeros((len(ids), 4), dtype=torch.float32, device="cpu", requires_grad=True)
    gathered = gather_verified_anchor_predictions(predictions, anchors)
    gathered["predictions"].sum().backward()
    assert gathered["count"] == 2
    active = set(anchors.positions)
    for position, grad in enumerate(predictions.grad):
        expected = torch.ones(4) if position in active else torch.zeros(4)
        assert torch.equal(grad, expected)

    too_many = copy.deepcopy(receipt)
    too_many["verified_observations"] = []
    for position in range(5):
        box = mapped["boxes_xyxy_normalized"][position]
        too_many["verified_observations"].append({
            "verification_status": "verified", "frame_position": position,
            "physical_frame_id": ids[position], "box_xyxy_normalized": box,
            "candidate_box_sha256": canonical_sha256(box),
            "pixel_sha256": hashlib.sha256(f"p-{position}".encode()).hexdigest(),
            "observation_receipt_sha256": hashlib.sha256(f"o-{position}".encode()).hexdigest(),
        })
    rejected_k = False
    try:
        build_verified_sparse_anchors(mapped, ids, too_many)
    except ValueError:
        rejected_k = True
    assert rejected_k

    # Known sigmoid chart, xyxy conversion, and fixed clipping at the left edge.
    cxcywh = torch.tensor([0.05, 0.5, 0.4, 0.3], dtype=torch.float64, device="cpu")
    z = torch.logit(cxcywh).requires_grad_(True)
    action_xyxy = action_transform_logits_cxcywh(z)
    expected_xyxy = torch.tensor([0.0, 0.35, 0.25, 0.65], dtype=torch.float64)
    assert torch.allclose(action_xyxy, expected_xyxy, atol=1e-12, rtol=0.0)
    action_xyxy.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()

    degenerate_rejected = False
    try:
        action_transform_logits_cxcywh(torch.tensor([-1000.0, 0.0, -1000.0, 0.0], device="cpu"))
    except ValueError:
        degenerate_rejected = True
    assert degenerate_rejected

    z64 = torch.tensor([[0.4, -0.3, 0.2, 0.1], [-0.1, 0.7, -0.2, 0.5]], dtype=torch.float64)
    mu64 = torch.tensor([[0.1, 0.0, -0.4, 0.2]], dtype=torch.float64)
    log_std64 = torch.tensor([-0.2, 0.1, 0.0, -0.3], dtype=torch.float64)
    logp = diagonal_gaussian_log_prob_z(z64, mu64, log_std64)
    manual = []
    for row in z64.tolist():
        manual.append(sum(-0.5 * (((v - m) * math.exp(-s)) ** 2 + 2.0 * s + math.log(2.0 * math.pi))
                          for v, m, s in zip(row, mu64[0].tolist(), log_std64.tolist())))
    assert torch.allclose(logp, torch.tensor(manual, dtype=torch.float64), atol=1e-12, rtol=0.0)

    return {
        "schema": "stvg-hierarchical-geometry-cpu-contracts-v1",
        "status": "pass",
        "synthetic_only": True,
        "CPU_only": True,
        "model_calls": 0,
        "GPU_used": False,
        "GT_read": False,
        "RGB_video_read": False,
        "checks": [
            "unit_square_xyxy_clamp_and_original_opaque_binding_preserved",
            "mapped_geometry_hash_dedup_and_stable_alias_bindings",
            "zero_area_after_clipping_rejected_without_epsilon_expansion",
            "verified_receipt_only_K2_sparse_anchor_construction",
            "missing_receipt_produces_no_supervision_or_core_anchor_rows",
            "missing_receipt_empty_gather_has_no_gradient",
            "autograd_gradient_only_at_verified_sparse_positions",
            "K_greater_than_four_receipt_rejected",
            "sigmoid_cxcywh_xyxy_fixed_clip_transform",
            "clipped_zero_area_action_rejected",
            "diagonal_gaussian_logprob_matches_manual_original_z_space",
        ],
        "action_transform": ACTION_TRANSFORM,
        "gaussian_log_probability_space": "original pre-transform z logits; no clipped-box Jacobian claim",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true",
                        help="run deterministic synthetic CPU contracts only")
    args = parser.parse_args(argv)
    if not args.selftest:
        parser.error("This is an importable geometry adapter; use --selftest for synthetic CPU contracts")
    result = run_cpu_contracts()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
