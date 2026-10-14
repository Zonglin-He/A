"""Sparse STVG-R1 pseudo-reference adapter for the frozen TA-STVG student core.

This module does not load models, videos, checkpoints, annotations, or labels at
import time.  The spatial API accepts only a semantically verified, source-bound
complete trajectory and turns the fixed Native K4 support into the existing
Gaussian spatial OPD/Direct anchor format.  Temporal adaptation is an optional
zero-initialized residual over the native two-offset time logits; it is fitted
against an independently verified physical interval and decoded again with the
unchanged native MAP decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONTRACT_PATH = ROOT / "artifacts/stvg_sparse_stvg_r1_semantic_opd_v1/student_contracts/CPU_CONTRACTS.json"
SPATIAL_SCHEMA = "stvg_sparse_candidate_pseudoreference_v1"
TEMPORAL_SCHEMA = "stvg_sparse_temporal_pseudoreference_v1"
ANCHOR_SCHEMA = "stvg_sparse_candidate_anchors_v1"
TEMPORAL_CONFIG = {"lr": 0.01, "steps": 10, "residual_l2": 1e-3}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _frame_ids(values: Sequence[int], name: str) -> list[int]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist") and not isinstance(values, (str, bytes, list, tuple)):
        values = values.tolist()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) < 2:
        raise ValueError(f"{name} must contain at least two physical frame IDs")
    result = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, numbers.Integral) or item < 0:
            raise ValueError(f"{name} must contain nonnegative integer physical frame IDs")
        result.append(int(item))
    if any(b <= a for a, b in zip(result, result[1:])):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return result


def _validate_candidate(candidate: Mapping[str, Any], frame_ids: Sequence[int],
                        native_support_frame_ids: Sequence[int],
                        expected_source_case_sha256: str) -> tuple[list[int], list[int], list[list[float]]]:
    """Validate post-seal candidate provenance and its complete normalized xyxy tube."""
    if not isinstance(candidate, Mapping) or candidate.get("schema") != SPATIAL_SCHEMA:
        raise ValueError("Unexpected sparse candidate pseudo-reference schema")
    if candidate.get("status") != "semantically_verified" or candidate.get("GT_read") is not False:
        raise ValueError("Spatial pseudo-reference is not post-seal semantic-only verified")
    if not isinstance(candidate.get("candidate_id"), str) or not candidate["candidate_id"]:
        raise ValueError("Candidate ID must be a nonempty catalog string")
    expected_sha = _require_sha(expected_source_case_sha256, "expected_source_case_sha256")
    if _require_sha(candidate.get("source_case_sha256"), "candidate.source_case_sha256") != expected_sha:
        raise ValueError("Candidate pseudo-reference belongs to a different cold source case")
    _require_sha(candidate.get("verification_receipt_sha256"),
                 "candidate.verification_receipt_sha256")

    ids = _frame_ids(frame_ids, "frame_ids")
    candidate_ids = _frame_ids(candidate.get("frame_ids", ()), "candidate.frame_ids")
    if candidate_ids != ids:
        raise ValueError("Candidate dense tube must align exactly to the full source physical-frame grid")
    support = _frame_ids(native_support_frame_ids, "native_support_frame_ids")
    if len(support) != 4:
        raise ValueError("Spatial OPD uses exactly the fixed Native K4 support")
    saved_support = candidate.get("native_support_frame_ids")
    saved_support = _frame_ids(saved_support, "candidate.native_support_frame_ids")
    if saved_support != support:
        raise ValueError("Candidate support differs from the locked Native K4 support")
    if any(frame_id not in set(ids) for frame_id in support):
        raise ValueError("Native K4 support contains a frame outside the source grid")

    raw_boxes = candidate.get("boxes_xyxy_normalized")
    if hasattr(raw_boxes, "detach"):
        raw_boxes = raw_boxes.detach().cpu().tolist()
    elif hasattr(raw_boxes, "tolist") and not isinstance(raw_boxes, (str, bytes, list, tuple)):
        raw_boxes = raw_boxes.tolist()
    if not isinstance(raw_boxes, Sequence) or isinstance(raw_boxes, (str, bytes)) or len(raw_boxes) != len(ids):
        raise ValueError("Candidate pseudo-reference must provide one dense box per source frame")
    boxes: list[list[float]] = []
    for row in raw_boxes:
        if hasattr(row, "tolist") and not isinstance(row, (list, tuple)):
            row = row.tolist()
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 4:
            raise ValueError("Every candidate box must contain normalized xyxy")
        box = []
        for coordinate in row:
            if isinstance(coordinate, bool) or not isinstance(coordinate, numbers.Real):
                raise ValueError("Candidate coordinates must be numeric")
            value = float(coordinate)
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError("Candidate coordinates must be finite and inside [0,1]")
            box.append(value)
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Candidate xyxy boxes must have positive area")
        boxes.append(box)
    return ids, support, boxes


def build_sparse_candidate_anchors(candidate: Mapping[str, Any], frame_ids: Sequence[int],
                                   native_support_frame_ids: Sequence[int], *,
                                   expected_source_case_sha256: str) -> dict[str, Any]:
    """Map a complete normalized-xyxy candidate onto fixed Native K4 positions.

    Returned ``anchors`` use the exact normalized cxcywh format consumed by
    ``stvg_target_teacher_student_v1._fit_direct`` and ``_fit_spatial_opd``.
    No support positions are reselected from candidate quality or output boxes.
    """
    import torch

    ids, support, boxes = _validate_candidate(candidate, frame_ids, native_support_frame_ids,
                                               expected_source_case_sha256)
    id_to_position = {frame_id: position for position, frame_id in enumerate(ids)}
    anchors = []
    for frame_id in support:
        position = id_to_position[frame_id]
        x1, y1, x2, y2 = boxes[position]
        anchors.append({
            "position": position,
            "physical_frame_id": frame_id,
            "box": torch.tensor([(x1 + x2) * 0.5, (y1 + y2) * 0.5,
                                 x2 - x1, y2 - y1], dtype=torch.float32),
            "weight": 1.0,
            "candidate_id": candidate["candidate_id"],
        })
    binding = {
        "schema": ANCHOR_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "source_case_sha256": candidate["source_case_sha256"],
        "verification_receipt_sha256": candidate["verification_receipt_sha256"],
        "frame_ids": ids,
        "native_support_frame_ids": support,
        "native_support_positions": [row["position"] for row in anchors],
        "boxes_xyxy_normalized_sha256": _sha256(boxes),
        "anchors_sha256": _sha256([{"position": a["position"],
                                     "physical_frame_id": a["physical_frame_id"],
                                     "box": a["box"].tolist(), "weight": 1.0}
                                    for a in anchors]),
        "GT_read": False,
    }
    return {"anchors": anchors, "binding": binding}


def _validate_temporal_reference(reference: Mapping[str, Any] | None,
                                  frame_ids: Sequence[int], *,
                                  expected_source_case_sha256: str,
                                  expected_candidate_id: str | None = None,
                                  expected_verification_receipt_sha256: str | None = None
                                  ) -> tuple[list[float], list[int]] | None:
    if reference is None:
        return None
    if not isinstance(reference, Mapping) or reference.get("schema") != TEMPORAL_SCHEMA:
        raise ValueError("Unexpected sparse temporal pseudo-reference schema")
    if reference.get("GT_read") is not False:
        raise ValueError("Temporal pseudo-reference must be explicitly GT-free")
    if reference.get("status") != "reliable":
        return None
    if not isinstance(reference.get("candidate_id"), str) or not reference["candidate_id"]:
        raise ValueError("Temporal reference must identify its catalog candidate")
    if expected_candidate_id is not None and reference["candidate_id"] != expected_candidate_id:
        raise ValueError("Temporal reference candidate does not match its spatial pseudo-reference")
    if _require_sha(reference.get("source_case_sha256"), "temporal.source_case_sha256") != _require_sha(
            expected_source_case_sha256, "expected_source_case_sha256"):
        raise ValueError("Temporal reference belongs to a different cold source case")
    _require_sha(reference.get("verification_receipt_sha256"),
                 "temporal.verification_receipt_sha256")
    if expected_verification_receipt_sha256 is not None:
        expected_receipt = _require_sha(expected_verification_receipt_sha256,
                                        "expected_temporal_verification_receipt_sha256")
        if reference["verification_receipt_sha256"] != expected_receipt:
            raise ValueError("Temporal reference differs from its sealed verification receipt")
    ids = _frame_ids(frame_ids, "frame_ids")
    interval = reference.get("interval_physical")
    if hasattr(interval, "tolist") and not isinstance(interval, (str, bytes, list, tuple)):
        interval = interval.tolist()
    if not isinstance(interval, Sequence) or isinstance(interval, (str, bytes)) or len(interval) != 2:
        raise ValueError("Temporal reference must use a half-open physical interval")
    start, end = interval
    if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, numbers.Real)
            or not isinstance(end, numbers.Real) or not math.isfinite(float(start))
            or not math.isfinite(float(end)) or float(end) <= float(start)):
        raise ValueError("Temporal reference interval must be a finite nonempty physical half-open span")
    start, end = float(start), float(end)
    clip_start, clip_end = float(ids[0]), float(ids[-1] + 1)
    if start < clip_start or end > clip_end:
        raise ValueError("Temporal reference interval extends beyond the source clip bounds")
    return [start, end], ids


def _offset_ids(records: Sequence[Mapping[str, Any]], frame_ids: Sequence[int]) -> list[list[int]]:
    if not isinstance(records, Sequence) or len(records) != 2:
        raise ValueError("Temporal residual expects the original two native offsets")
    ids = _frame_ids(frame_ids, "frame_ids")
    offsets = [_frame_ids(record.get("frame_ids", ()), f"records[{i}].frame_ids")
               for i, record in enumerate(records)]
    flattened = [value for row in offsets for value in row]
    if len(flattened) != len(set(flattened)) or sorted(flattened) != ids:
        raise ValueError("Native offset frame IDs must partition the complete source physical grid")
    return offsets


def _interval_iou(a: Sequence[float], b: Sequence[float]) -> float:
    a0, a1, b0, b1 = float(a[0]), float(a[1]), float(b[0]), float(b[1])
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - intersection
    return float(intersection / union) if union > 0 else 0.0


def _target_pair(offset_ids: Sequence[int], target_interval: Sequence[float]) -> tuple[int, int, float]:
    best = None
    for start in range(len(offset_ids) - 1):
        for end in range(start + 1, len(offset_ids)):
            span = [int(offset_ids[start]), int(offset_ids[end]) + 1]
            score = _interval_iou(span, target_interval)
            key = (-score, start, end)
            if best is None or key < best[0]:
                best = (key, start, end, score)
    if best is None:
        raise ValueError("Each native offset must have at least one legal start<end pair")
    return best[1], best[2], best[3]


def _decode_native(logits: Sequence[Any], boxes: Any | None,
                   records: Sequence[Mapping[str, Any]], frame_ids: Sequence[int]) -> dict[str, Any]:
    """Delegate interval selection to the original TA-STVG native decoder."""
    import torch
    from methods.decota_final_simplified_v1.objectives import prediction

    if boxes is None:
        boxes = torch.zeros((len(frame_ids), 4), dtype=torch.float32)
    return prediction(list(logits), boxes, list(records), list(frame_ids))


def fit_temporal_logits_residual(logits: Sequence[Any], records: Sequence[Mapping[str, Any]],
                                 frame_ids: Sequence[int], temporal_reference: Mapping[str, Any] | None,
                                 *, expected_source_case_sha256: str,
                                 expected_candidate_id: str | None = None,
                                 expected_verification_receipt_sha256: str | None = None,
                                 boxes: Any | None = None,
                                 config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fit zero-initialized per-logit residuals, then run the unchanged MAP decoder.

    A missing/unreliable reference or a reference already selected by every
    native offset is an exact no-op.  The fixed configuration is intentionally
    not tunable from case outcomes.
    """
    import torch
    import torch.nn.functional as F
    from methods.decota_final_simplified_v1.objectives import native_view_indices

    cfg = dict(TEMPORAL_CONFIG if config is None else config)
    if cfg != TEMPORAL_CONFIG:
        raise ValueError("Temporal residual hyperparameters must match the prelocked configuration")
    ids = _frame_ids(frame_ids, "frame_ids")
    if len(logits) != 2 or len(records) != 2:
        raise ValueError("Temporal residual expects exactly two native offsets")
    offset_ids = _offset_ids(records, ids)
    raw = []
    before_indices = []
    for index, row in enumerate(logits):
        if not torch.is_tensor(row) or row.ndim != 3 or tuple(row.shape[:1]) != (1,) or row.shape[-1] != 2:
            raise ValueError("Native logits must retain original [1,T,2] shape")
        if row.shape[1] != len(offset_ids[index]) or not torch.isfinite(row).all():
            raise ValueError("Native logits do not align to their physical offset frame IDs")
        raw.append(row.detach().to(dtype=torch.float32))
        before_indices.append(tuple(int(x) for x in native_view_indices(row.detach().cpu())))

    reference = _validate_temporal_reference(temporal_reference, ids,
                    expected_source_case_sha256=expected_source_case_sha256,
                    expected_candidate_id=expected_candidate_id,
                    expected_verification_receipt_sha256=expected_verification_receipt_sha256)
    if reference is None:
        native = _decode_native(raw, boxes, records, ids)
        return {"status": "no_update", "reason": "no_reliable_temporal_reference",
                "logits": [x.detach().clone() for x in raw], "optimizer_steps": 0,
                "residual_l2": 0.0, "redecoded": False, "before_indices": [list(x) for x in before_indices],
                "after_indices": [list(x) for x in before_indices], "native_readout": native,
                "config": cfg, "GT_read": False}

    target_interval, _ = reference
    before_native = _decode_native(raw, boxes, records, ids)
    if list(before_native["physical_interval"]) == list(target_interval):
        return {"status": "no_update", "reason": "reliable_reference_matches_native_interval",
                "logits": [x.detach().clone() for x in raw], "optimizer_steps": 0,
                "residual_l2": 0.0, "redecoded": False,
                "target_interval_physical": list(target_interval),
                "before_indices": [list(x) for x in before_indices],
                "after_indices": [list(x) for x in before_indices],
                "native_readout": before_native, "config": cfg, "GT_read": False}
    target_indices = [_target_pair(row, target_interval) for row in offset_ids]
    if any(score <= 0.0 for _, _, score in target_indices):
        native = _decode_native(raw, boxes, records, ids)
        return {"status": "no_update", "reason": "reference_has_no_native_offset_overlap",
                "logits": [x.detach().clone() for x in raw], "optimizer_steps": 0,
                "residual_l2": 0.0, "redecoded": False,
                "target_indices": [[s, e] for s, e, _ in target_indices],
                "before_indices": [list(x) for x in before_indices],
                "after_indices": [list(x) for x in before_indices], "native_readout": native,
                "config": cfg, "GT_read": False}

    targets = [(start, end) for start, end, _ in target_indices]
    if targets == before_indices:
        native = _decode_native(raw, boxes, records, ids)
        return {"status": "no_update", "reason": "reliable_reference_matches_native_decode",
                "logits": [x.detach().clone() for x in raw], "optimizer_steps": 0,
                "residual_l2": 0.0, "redecoded": False,
                "target_indices": [list(x) for x in targets],
                "before_indices": [list(x) for x in before_indices],
                "after_indices": [list(x) for x in before_indices], "native_readout": native,
                "config": cfg, "GT_read": False}

    residuals = [torch.zeros_like(row, dtype=torch.float32, requires_grad=True) for row in raw]
    optimizer = torch.optim.Adam(residuals, lr=cfg["lr"], betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    updates = 0
    losses = []
    for _ in range(cfg["steps"]):
        optimizer.zero_grad(set_to_none=True)
        total = raw[0].new_zeros(())
        for row, residual, (start, end) in zip(raw, residuals, targets):
            adjusted = row + residual
            start_lp = F.log_softmax(adjusted[0, :, 0], dim=0)
            end_lp = F.log_softmax(adjusted[0, :, 1], dim=0)
            pair_scores = start_lp[:, None] + end_lp[None, :]
            legal = torch.triu(torch.ones_like(pair_scores, dtype=torch.bool), diagonal=1)
            normalizer = torch.logsumexp(pair_scores[legal], dim=0)
            total = total + normalizer - pair_scores[start, end]
        penalty = sum(value.square().mean() for value in residuals) / len(residuals)
        loss = total / len(residuals) + cfg["residual_l2"] * penalty
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite temporal residual loss")
        gradients = torch.autograd.grad(loss, residuals)
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("Nonfinite temporal residual gradient")
        if not any(bool(torch.count_nonzero(gradient)) for gradient in gradients):
            break
        for residual, gradient in zip(residuals, gradients):
            residual.grad = gradient.detach()
        optimizer.step()
        updates += 1
        losses.append(float(loss.detach().cpu()))

    adapted = [(row + residual.detach()).detach() for row, residual in zip(raw, residuals)]
    changed = any(not torch.equal(before, after) for before, after in zip(raw, adapted))
    if updates == 0 or not changed:
        # Preserve exact no-op semantics when the objective supplied no usable gradient.
        adapted = [row.detach().clone() for row in raw]
        updates = 0
        residual_norm = 0.0
    else:
        residual_norm = float(sum(x.detach().square().sum().item() for x in residuals) ** 0.5)
    after_indices = [tuple(int(x) for x in native_view_indices(row.detach().cpu()))
                     for row in adapted]
    native = _decode_native(adapted, boxes, records, ids)
    return {"status": "adapted" if updates else "no_update",
            "reason": "residual_optimizer_updated" if updates else "zero_gradient_noop",
            "logits": adapted,
            "optimizer_steps": updates,
            "residual_l2": residual_norm,
            "target_interval_physical": list(target_interval),
            "target_indices": [list(x) for x in targets],
            "target_offset_iou": [float(score) for _, _, score in target_indices],
            "before_indices": [list(x) for x in before_indices],
            "after_indices": [list(x) for x in after_indices],
            "losses": losses,
            "redecoded": bool(updates),
            "native_readout": native,
            "config": cfg, "GT_read": False}


def fit_sparse_candidate_spatial_pair(model: Any, batch: Any, base: Any,
                                      records: Sequence[Mapping[str, Any]],
                                      frame_ids: Sequence[int], *,
                                      candidate: Mapping[str, Any],
                                      native_support_frame_ids: Sequence[int],
                                      expected_source_case_sha256: str,
                                      dataset: str, key: str, scope: str = "joint",
                                      direct_resume: Mapping[str, Any] | None = None,
                                      opd_resume: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run matched Direct and Gaussian SpatialOPD fits from the same cold state.

    This callable intentionally delegates the optimization/readout to the
    reviewed TA-STVG student core.  It does not introduce a new optimizer,
    change the active Query+LayerNorm scope, or overwrite a native decoder
    result with pseudo-reference boxes.
    """
    import torch
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts import stvg_target_teacher_student_v1 as core

    packed = build_sparse_candidate_anchors(candidate, frame_ids, native_support_frame_ids,
                       expected_source_case_sha256=expected_source_case_sha256)
    anchors = packed["anchors"]
    config = core.load_original_opd_config(dataset)
    initial = core._copy_cpu(base.initial)
    device = next(model.parameters()).device
    initial = {name: value.to(device) for name, value in initial.items()}

    direct_fit = core._fit_direct(base, initial, anchors, frame_ids, key, config, scope=scope,
                                  resume_state=direct_resume)
    opd_fit = core._fit_spatial_opd(base, initial, anchors, frame_ids, key, config, scope=scope,
                                    resume_state=opd_resume)
    direct_hash = state_hash(direct_fit["initial_state"])
    opd_hash = state_hash(opd_fit["initial_state"])
    if direct_hash != opd_hash:
        raise RuntimeError("Direct and SpatialOPD did not start from the same cold student state")
    direct_readout = core._finalize_student_readout(model, batch, frame_ids, records,
                                                    direct_fit["final_state"])
    opd_readout = core._finalize_student_readout(model, batch, frame_ids, records,
                                                 opd_fit["final_state"])
    return {"schema": "stvg-sparse-r1-student-spatial-pair-v1",
            "candidate_binding": packed["binding"],
            "source_case_sha256": expected_source_case_sha256,
            "dataset": dataset, "key": key, "scope": scope,
            "configuration": dict(config),
            "initial_student_state_sha256": direct_hash,
            "GT_read": False,
            "arms": {
                "Direct": {"fit": direct_fit, "native_readout": direct_readout},
                "SpatialOPD": {"fit": opd_fit, "native_readout": opd_readout},
            }}


def attach_temporal_residual(spatial_pair: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                             frame_ids: Sequence[int], temporal_reference: Mapping[str, Any] | None,
                             *, expected_source_case_sha256: str,
                             expected_temporal_candidate_id: str | None = None,
                             expected_temporal_verification_receipt_sha256: str | None = None,
                             config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Attach per-arm genuine temporal residual fits to final native readouts."""
    if spatial_pair.get("schema") != "stvg-sparse-r1-student-spatial-pair-v1":
        raise ValueError("Unexpected spatial fit result schema")
    source_sha = _require_sha(expected_source_case_sha256, "expected_source_case_sha256")
    if spatial_pair.get("source_case_sha256") != source_sha:
        raise ValueError("Spatial fit belongs to a different cold source case")
    candidate_binding = spatial_pair.get("candidate_binding")
    if not isinstance(candidate_binding, Mapping) or candidate_binding.get("source_case_sha256") != source_sha:
        raise ValueError("Spatial candidate receipt is not bound to the expected cold source case")
    _require_sha(candidate_binding.get("verification_receipt_sha256"),
                 "spatial.verification_receipt_sha256")
    arms = {}
    for arm_name, item in spatial_pair.get("arms", {}).items():
        readout = item.get("native_readout")
        if not isinstance(readout, Mapping) or not {"logits", "boxes"} <= set(readout):
            raise ValueError(f"{arm_name} omitted the full native student readout")
        adapted = fit_temporal_logits_residual(readout["logits"], records, frame_ids,
                    temporal_reference, expected_source_case_sha256=expected_source_case_sha256,
                    expected_candidate_id=expected_temporal_candidate_id,
                    expected_verification_receipt_sha256=expected_temporal_verification_receipt_sha256,
                    boxes=readout["boxes"], config=config)
        arms[arm_name] = {"native_readout": readout,
                          "temporal_residual": adapted,
                          "adapted_readout": adapted["native_readout"]}
    return {"schema": "stvg-sparse-r1-student-spatiotemporal-pair-v1",
            "candidate_binding": spatial_pair["candidate_binding"],
            "source_case_sha256": source_sha,
            "temporal_candidate_id": (temporal_reference.get("candidate_id")
                                      if isinstance(temporal_reference, Mapping) else None),
            "temporal_verification_receipt_sha256": (
                temporal_reference.get("verification_receipt_sha256")
                if isinstance(temporal_reference, Mapping) else None),
            "temporal_reference_status": (temporal_reference.get("status")
                                          if isinstance(temporal_reference, Mapping) else "unavailable"),
            "GT_read": False, "arms": arms}


def run_cpu_contracts() -> dict[str, Any]:
    """Exercise coordinate, provenance, exact no-op, OPD reward, and temporal decode contracts."""
    import torch
    from methods.decota_final_simplified_v1.objectives import native_view_indices
    from vg_tta.decota_spatial_opd_tunable_v1 import action_boxes, rollout, optimizer_step

    source_sha = "1" * 64
    receipt_sha = "2" * 64
    ids = list(range(8))
    support = [0, 2, 4, 6]
    dense = [[0.1 + 0.01 * i, 0.2, 0.5 + 0.01 * i, 0.8] for i in range(8)]
    candidate = {"schema": SPATIAL_SCHEMA, "status": "semantically_verified",
                 "candidate_id": "synthetic-candidate-1", "source_case_sha256": source_sha,
                 "verification_receipt_sha256": receipt_sha, "frame_ids": ids,
                 "native_support_frame_ids": support,
                 "boxes_xyxy_normalized": dense, "GT_read": False}
    packed = build_sparse_candidate_anchors(candidate, ids, support,
                                             expected_source_case_sha256=source_sha)
    anchors = packed["anchors"]
    assert [a["position"] for a in anchors] == [0, 2, 4, 6]
    assert [a["physical_frame_id"] for a in anchors] == support
    expected0 = torch.tensor([(dense[0][0] + dense[0][2]) / 2,
                              (dense[0][1] + dense[0][3]) / 2,
                              dense[0][2] - dense[0][0], dense[0][3] - dense[0][1]])
    assert torch.allclose(anchors[0]["box"], expected0)

    for mutation in (
        {"source_case_sha256": "3" * 64},
        {"native_support_frame_ids": [0, 1, 2, 3]},
        {"status": "unverified"},
        {"GT_read": True},
    ):
        bad = dict(candidate); bad.update(mutation)
        try:
            build_sparse_candidate_anchors(bad, ids, support,
                                          expected_source_case_sha256=source_sha)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Candidate provenance/shape mutation was accepted: {mutation}")

    # Exact native-target match must not even instantiate/step Adam.
    record0 = {"frame_ids": [0, 2, 4, 6]}
    record1 = {"frame_ids": [1, 3, 5, 7]}
    native_logits = [torch.full((1, 4, 2), -4.0), torch.full((1, 4, 2), -4.0)]
    native_logits[0][0, 0, 0] = 4.0; native_logits[0][0, 1, 1] = 4.0
    native_logits[1][0, 0, 0] = 4.0; native_logits[1][0, 1, 1] = 4.0
    native_pair = [tuple(native_view_indices(row)) for row in native_logits]
    native_span = [min(record["frame_ids"][p[0]] for record, p in zip([record0, record1], native_pair)),
                   max(record["frame_ids"][p[1]] + 1 for record, p in zip([record0, record1], native_pair))]
    same_ref = {"schema": TEMPORAL_SCHEMA, "status": "reliable",
                "candidate_id": "synthetic-candidate-1", "source_case_sha256": source_sha,
                "verification_receipt_sha256": receipt_sha,
                "interval_physical": native_span, "GT_read": False}
    no_op = fit_temporal_logits_residual(native_logits, [record0, record1], ids, same_ref,
              expected_source_case_sha256=source_sha, expected_candidate_id="synthetic-candidate-1")
    assert no_op["status"] == "no_update" and no_op["optimizer_steps"] == 0 and not no_op["redecoded"]
    assert all(torch.equal(a, b) for a, b in zip(native_logits, no_op["logits"]))

    # A reliable, mismatched interval must update a real residual and use the native decoder again.
    adapt_logits = [torch.zeros((1, 4, 2)), torch.zeros((1, 4, 2))]
    adapt_logits[0][0, 0, 0] = 3.0; adapt_logits[0][0, 1, 1] = 3.0
    adapt_logits[1][0, 0, 0] = 3.0; adapt_logits[1][0, 1, 1] = 3.0
    temporal_ref = {"schema": TEMPORAL_SCHEMA, "status": "reliable",
                    "candidate_id": "synthetic-temporal-candidate-T1", "source_case_sha256": source_sha,
                    "verification_receipt_sha256": receipt_sha,
                    "interval_physical": [4.1, 7.9], "GT_read": False}
    adapted = fit_temporal_logits_residual(adapt_logits, [record0, record1], ids, temporal_ref,
                expected_source_case_sha256=source_sha,
                expected_candidate_id="synthetic-temporal-candidate-T1")
    assert adapted["status"] == "adapted" and adapted["optimizer_steps"] > 0 and adapted["redecoded"]
    assert any(not torch.equal(a, b) for a, b in zip(adapt_logits, adapted["logits"]))
    assert adapted["native_readout"]["logits"] is not None
    assert adapted["target_interval_physical"] == [4.1, 7.9]

    # Temporal T-candidate IDs and receipts are independent from spatial S-candidates.
    spatial_pair = {"schema": "stvg-sparse-r1-student-spatial-pair-v1",
                    "source_case_sha256": source_sha,
                    "candidate_binding": {"candidate_id": "synthetic-spatial-candidate-S1",
                                          "source_case_sha256": source_sha,
                                          "verification_receipt_sha256": receipt_sha},
                    "arms": {"Direct": {"native_readout": {
                        "logits": adapt_logits, "boxes": torch.zeros((8, 4))}}}}
    attached = attach_temporal_residual(spatial_pair, [record0, record1], ids, temporal_ref,
                  expected_source_case_sha256=source_sha,
                  expected_temporal_verification_receipt_sha256=receipt_sha)
    assert attached["temporal_candidate_id"] == "synthetic-temporal-candidate-T1"
    assert attached["arms"]["Direct"]["temporal_residual"]["status"] == "adapted"
    try:
        attach_temporal_residual(spatial_pair, [record0, record1], ids, temporal_ref,
              expected_source_case_sha256=source_sha,
              expected_temporal_candidate_id="wrong-temporal-id")
    except ValueError:
        pass
    else:
        raise AssertionError("An explicitly mismatched temporal candidate ID was accepted")
    bad_clip_ref = dict(temporal_ref, interval_physical=[-0.1, 7.9])
    try:
        fit_temporal_logits_residual(adapt_logits, [record0, record1], ids, bad_clip_ref,
                    expected_source_case_sha256=source_sha)
    except ValueError:
        pass
    else:
        raise AssertionError("A temporal reference extending past clip bounds was accepted")

    # No temporal reference and zero-information OPD updates preserve exact state.
    unavailable = fit_temporal_logits_residual(adapt_logits, [record0, record1], ids, None,
                expected_source_case_sha256=source_sha)
    assert unavailable["status"] == "no_update" and unavailable["optimizer_steps"] == 0
    parameter = torch.nn.Parameter(torch.tensor([0.2, -0.1]))
    optimizer = torch.optim.Adam([parameter], lr=0.03)
    before_param = parameter.detach().clone()
    before_state = json.dumps(optimizer.state_dict(), sort_keys=True, default=str)
    skipped = optimizer_step(optimizer, parameter.square().sum(), [parameter], informative=False)
    after_state = json.dumps(optimizer.state_dict(), sort_keys=True, default=str)
    assert skipped is None and torch.equal(parameter, before_param) and before_state == after_state

    # Validate the frozen antithetic rollout reward against an independent scalar xyxy IoU.
    cxcywh = torch.tensor([[0.5, 0.5, 0.25, 0.25], [0.4, 0.6, 0.2, 0.3]])
    evidence = torch.tensor([[0.5, 0.5, 0.3, 0.25], [0.4, 0.6, 0.18, 0.28]])
    policy = rollout(torch.logit(cxcywh), evidence, "sparse-contract-key", 0,
                     shuffle=False, config={"sigma": 0.25, "tau": 0.25, "samples": 32,
                                            "steps": 10, "lr": 0.03, "writeback": 1/16})
    actions = action_boxes(policy["samples"])

    def scalar_iou(a, b):
        def corners(row):
            x, y, w, h = row
            return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        ax1, ay1, ax2, ay2 = corners(a); bx1, by1, bx2, by2 = corners(b)
        inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
        area_a = (ax2-ax1) * (ay2-ay1); area_b = (bx2-bx1) * (by2-by1)
        return inter / (area_a + area_b - inter)

    scalar_rewards = torch.tensor([[scalar_iou(actions[i, j].tolist(), evidence[i].tolist())
                                    for j in range(32)] for i in range(2)])
    assert torch.allclose(policy["rewards"].cpu(), scalar_rewards, atol=1e-6, rtol=1e-6)
    assert not torch.cuda.is_initialized()
    return {"status": "pass", "scope": "synthetic_cpu_only_student_adapter_contracts",
            "checks": ["fixed_native_k4_xyxy_to_cxcywh", "candidate_source_and_semantic_receipt_binding",
                       "dense_grid_and_support_rejects", "temporal_exact_noop_without_adam_mutation",
                       "temporal_residual_updates_then_native_redecode",
                       "independent_spatial_S_temporal_T_candidate_ids",
                       "continuous_offgrid_physical_boundaries_inside_clip",
                       "temporal_receipt_binding_and_clip_bounds", "unavailable_temporal_noop",
                       "spatial_opd_no_information_exact_noop", "opd_candidate_iou_scalar_crosscheck"],
            "CPU_only": True, "GT_read": False, "real_cases_read": 0,
            "model_calls": 0, "GPU_initialized": bool(torch.cuda.is_initialized())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("This adapter is imported by the root-controlled sparse worker; use --selftest for CPU contracts")
    result = run_cpu_contracts()
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    CONTRACT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
