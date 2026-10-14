"""Post-seal official scoring for the locked motion-guided student outputs.

All student artifacts, reference locks, source cases, and candidate inputs are
verified before the official target reader is called. This module never loads
a model, initializes CUDA, edits student artifacts, or selects candidates from
GT. GT is used only for the requested post-seal score and fixed-anchor
diagnostics.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / "artifacts/stvg_motion_guided_instance_opd_v1"
OLD_BASE = ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1"
OLD_IDS = {
    f"{domain}_{index:03d}"
    for domain in ("hc2", "vidstg")
    for index in range(4)
}
METHOD_IDS = {
    f"{domain}_{index:03d}"
    for domain in ("hc2", "vidstg")
    for index in range(8)
}
CONTROL_ID = "vidstg_002"
ALGORITHMS = ("direct", "iou", "diou")
CONTROL_ROLES = ("Source", "oldR1ID2", "oldAlternativeID3")
NEUTRAL_EPS = 1e-8

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.review_stvg_sparse_r1_v2 import (  # noqa: E402
    METRIC_CONTRACT,
    _metric_functions,
    _truths_after_seal,
    score_tube_xyxy,
)
from scripts.stvg_sparse_r1_prompt_v1 import load_source_case  # noqa: E402
from vg_tta.tastvg_oracle_event5_v1 import DenseTube, box_iou  # noqa: E402


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _safe_path(relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise RuntimeError(f"absolute artifact path is not allowed: {relative}")
    path = (ROOT / value).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise RuntimeError(f"artifact path escapes workspace: {relative}")
    return path


def _as_numpy(value: Any, label: str) -> np.ndarray:
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            raise RuntimeError(f"{label}: saved student tensor is not on CPU")
        return value.detach().numpy()
    return np.asarray(value)


def _assert_cpu_tree(value: Any, label: str = "artifact") -> None:
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            raise RuntimeError(f"{label}: artifact contains a non-CPU tensor")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _assert_cpu_tree(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_cpu_tree(child, f"{label}[{index}]")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _tensor_equal(left: Any, right: Any, label: str) -> None:
    a, b = _as_numpy(left, label), _as_numpy(right, label)
    _require(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b),
             f"{label}: exact arrays differ")


def _runtime_for_phase(phase: str, barrier: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    wanted = barrier.get("runtime_sha256")
    _require(isinstance(wanted, str) and len(wanted) == 64,
             f"{phase}: barrier has no runtime SHA")
    stem = f"{phase.upper()}_STUDENT_RUNTIME"
    matches = []
    for path in sorted(BASE.glob(stem + "*.json")):
        if _sha_file(path) == wanted:
            matches.append(path)
    _require(len(matches) == 1, f"{phase}: runtime SHA does not resolve uniquely")
    path = matches[0]
    runtime = _read_json(path)
    _require(runtime.get("status") == "locked"
             and runtime.get("phase") == phase
             and runtime.get("GT_read") is False,
             f"{phase}: selected runtime is not the expected locked GT-free runtime")
    return path, runtime


def _verify_phase_seal(phase: str, expected_ids: set[str],
                       expected_fit_count: int) -> dict[str, Any]:
    barrier_path = BASE / f"{phase.upper()}_STUDENT_GLOBAL_BARRIER.json"
    _require(barrier_path.is_file(), f"{phase}: complete global barrier is absent")
    barrier = _read_json(barrier_path)
    _require(barrier.get("status") == "complete"
             and barrier.get("GT_read") is False,
             f"{phase}: student barrier is incomplete or records GT access")
    _require(int(barrier.get("count", -1)) == len(expected_ids)
             and int(barrier.get("actual_fit_count", -1)) == expected_fit_count,
             f"{phase}: sealed case/fit counts differ")
    records = barrier.get("files")
    _require(isinstance(records, list) and len(records) == len(expected_ids),
             f"{phase}: barrier output inventory is incomplete")
    files: dict[str, dict[str, Any]] = {}
    for record in records:
        relative = record.get("path")
        _require(isinstance(relative, str), f"{phase}: malformed output path")
        path = _safe_path(relative)
        _require(path.is_file(), f"{phase}: sealed output is missing: {relative}")
        _require(_sha_file(path) == record.get("sha256"),
                 f"{phase}: sealed output hash mismatch: {relative}")
        _require(path.stat().st_size == int(record.get("bytes", -1)),
                 f"{phase}: sealed output size mismatch: {relative}")
        cid = path.stem
        _require(cid not in files, f"{phase}: duplicate output case {cid}")
        files[cid] = {"path": path, "sha256": record["sha256"], "bytes": record["bytes"]}
    _require(set(files) == expected_ids,
             f"{phase}: barrier case inventory differs from locked roster")

    runtime_path, runtime = _runtime_for_phase(phase, barrier)
    roster_path = BASE / "ROSTER_LOCK.json"
    reference_path = BASE / (
        "METHOD_REFERENCE_LOCK.json" if phase == "method"
        else "REFERENCE_CONTROL_REFERENCE_LOCK.json"
    )
    _require(runtime.get("roster_lock_sha256") == _sha_file(roster_path)
             and runtime.get("reference_lock_sha256") == _sha_file(reference_path),
             f"{phase}: runtime roster/reference lock hashes differ")
    _require(barrier.get("roster_lock_sha256") == _sha_file(roster_path)
             and barrier.get("reference_lock_sha256") == _sha_file(reference_path),
             f"{phase}: barrier roster/reference lock hashes differ")
    worker = "scripts/run_stvg_motion_guided_student002.py"
    launcher = "scripts/launch_stvg_motion_guided_student002.py"
    pins = runtime.get("pins", {})
    protected = runtime.get("protected", {})
    _require(worker in pins and launcher in pins,
             f"{phase}: runtime does not pin its actual worker and launcher")
    _require("methods/CURRENT_METHOD.json" in protected,
             f"{phase}: current method is not protected by runtime")
    verified_pins = []
    for relative, digest in {**pins, **protected}.items():
        path = _safe_path(relative)
        _require(path.is_file() and _sha_file(path) == digest,
                 f"{phase}: runtime pin mismatch: {relative}")
        verified_pins.append({"path": relative, "sha256": digest})

    readback_path = BASE / phase / "ROOT_CPU_FIT_MATH_READBACK.json"
    _require(readback_path.is_file(), f"{phase}: independent CPU fit readback is absent")
    readback = _read_json(readback_path)
    reader_path = ROOT / "scripts/review_stvg_motion_guided_math_v1.py"
    _require(readback.get("status") == "pass"
             and readback.get("phase") == phase
             and readback.get("GT_read") is False
             and readback.get("GPU_initialized") is False
             and readback.get("model_loaded") is False
             and readback.get("reader_sha256") == _sha_file(reader_path)
             and readback.get("barrier_bound_runtime") == str(runtime_path.relative_to(ROOT)),
             f"{phase}: independent CPU math readback is stale or incomplete")
    return {
        "phase": phase,
        "barrier_path": str(barrier_path.relative_to(ROOT)),
        "barrier_sha256": _sha_file(barrier_path),
        "runtime_path": str(runtime_path.relative_to(ROOT)),
        "runtime_sha256": _sha_file(runtime_path),
        "roster_lock_sha256": _sha_file(roster_path),
        "reference_lock_path": str(reference_path.relative_to(ROOT)),
        "reference_lock_sha256": _sha_file(reference_path),
        "student_file_count": len(files),
        "actual_fit_count": int(barrier["actual_fit_count"]),
        "actual_optimizer_steps": int(barrier.get("optimizer_steps", 0)),
        "student_files": [
            {"case_id": cid, "path": str(row["path"].relative_to(ROOT)),
             "sha256": row["sha256"], "bytes": int(row["bytes"])}
            for cid, row in sorted(files.items())
        ],
        "runtime_pins_verified": len(verified_pins),
        "runtime_pin_bindings": verified_pins,
        "cpu_math_readback_path": str(readback_path.relative_to(ROOT)),
        "cpu_math_readback_sha256": _sha_file(readback_path),
        "cpu_math_reader_sha256": readback["reader_sha256"],
        "barrier": barrier,
        "files": files,
    }


def _verify_candidate_input_barrier(roster_by_id: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = BASE / "CANDIDATE_INPUT_BARRIER.json"
    barrier = _read_json(path)
    _require(barrier.get("status") == "complete"
             and barrier.get("GT_read") is False
             and int(barrier.get("count", -1)) == len(METHOD_IDS),
             "candidate input barrier is not complete and GT-free")
    files = barrier.get("files")
    _require(isinstance(files, list) and files,
             "candidate input barrier has no file-hash inventory")
    input_file_bindings = []
    for record in files:
        rel = record.get("path")
        _require(isinstance(rel, str), "candidate input barrier has malformed file entry")
        target = _safe_path(rel)
        _require(target.is_file() and _sha_file(target) == record.get("sha256")
                 and target.stat().st_size == int(record.get("bytes", -1)),
                 f"candidate input artifact hash mismatch: {rel}")
        input_file_bindings.append({"path": rel, "sha256": record["sha256"],
                                    "bytes": int(record["bytes"])})
    records = barrier.get("records")
    _require(isinstance(records, list) and len(records) == len(METHOD_IDS),
             "candidate input record inventory differs from 16-case roster")
    by_id = {str(r.get("case_id")): r for r in records}
    _require(len(by_id) == len(records) and set(by_id) == METHOD_IDS,
             "candidate input case IDs differ from locked roster")
    for cid, row in by_id.items():
        locked = roster_by_id[cid]
        _require(row.get("GT_read") is False
                 and row.get("source_case_sha256") == locked["source_case_sha256"]
                 and row.get("cohort") == locked["cohort"],
                 f"{cid}: candidate input source/cohort binding differs")
        _require(len(row.get("positions", [])) == 4
                 and len(row.get("physical_frame_ids", [])) == 4,
                 f"{cid}: original-four observation support is malformed")
        _require(len(row.get("candidates", [])) > 0,
                 f"{cid}: locked candidate pool is empty")
        cand_ids = [str(x.get("candidate_id")) for x in row["candidates"]]
        _require(len(cand_ids) == len(set(cand_ids)),
                 f"{cid}: candidate IDs are duplicated")
    return by_id, {
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha_file(path),
        "status": barrier["status"],
        "GT_read": False,
        "record_count": len(by_id),
        "candidate_count": int(barrier.get("candidate_count", 0)),
        "file_count": len(input_file_bindings),
        "file_bindings": input_file_bindings,
    }


def _verify_old_temporal_seal() -> dict[str, Any]:
    path = OLD_BASE / "STUDENT_GLOBAL_BARRIER.json"
    barrier = _read_json(path)
    _require(barrier.get("status") == "complete"
             and barrier.get("GT_read") is False
             and int(barrier.get("count", -1)) == len(OLD_IDS),
             "prior temporal student barrier is incomplete")
    files = {}
    for record in barrier.get("files", []):
        target = _safe_path(record["path"])
        _require(target.is_file() and _sha_file(target) == record.get("sha256"),
                 f"old temporal output hash mismatch: {record.get('path')}")
        cid = target.stem
        files[cid] = {"path": target, "sha256": record["sha256"]}
    _require(set(files) == OLD_IDS, "old temporal output inventory differs from old eight")
    return {
        "barrier_path": str(path.relative_to(ROOT)),
        "barrier_sha256": _sha_file(path),
        "count": len(files),
        "file_bindings": [
            {"case_id": cid, "path": str(row["path"].relative_to(ROOT)),
             "sha256": row["sha256"]}
            for cid, row in sorted(files.items())
        ],
        "files": files,
    }


def _check_output_identity(output: Mapping[str, Any], cid: str,
                           roster: Mapping[str, Any], case: Mapping[str, Any],
                           reference_sha: str) -> None:
    _require(output.get("case_id") == cid
             and output.get("GT_read") is False
             and output.get("source_case_sha256") == roster["source_case_sha256"]
             and output.get("reference_lock_sha256") == reference_sha,
             f"{cid}: student output identity/lock binding differs")
    _require(output.get("source_restored") is True,
             f"{cid}: output did not receipt source restoration")
    cold = output.get("source_cold_readback", {})
    _require(cold.get("status") == "pass"
             and cold.get("source_case_sha256") == roster["source_case_sha256"]
             and cold.get("source_model_state_sha256") == case["source_model_state_sha256"]
             and cold.get("pixel_sha256") == case["pixel_sha256"]
             and cold.get("frame_ids_match") is True
             and cold.get("initial_state_match") is True
             and cold.get("zero_boxes_logits_match") is True
             and cold.get("source_restored") is True,
             f"{cid}: source cold-readback does not match the locked source case")
    _require(output.get("source_model_state_sha256") == case["source_model_state_sha256"]
             and output.get("pixel_sha256") == case["pixel_sha256"]
             and list(map(int, output.get("frame_ids", []))) == list(map(int, case["frame_ids"])),
             f"{cid}: source state, pixels, or physical frame IDs differ")


def _cxcywh_from_xyxy(boxes: Any) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    return np.concatenate(((values[:, :2] + values[:, 2:]) / 2,
                           values[:, 2:] - values[:, :2]), axis=1)


def _xyxy_from_cxcywh(boxes: Any) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    return np.concatenate((values[:, :2] - values[:, 2:] / 2,
                           values[:, :2] + values[:, 2:] / 2), axis=1)


def _state_flat(state: Mapping[str, Any], label: str) -> np.ndarray:
    chunks = []
    for key in sorted(state):
        value = _as_numpy(state[key], f"{label}.{key}").astype(np.float64, copy=False)
        chunks.append(value.reshape(-1))
    return np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float64)


def _anchor_rows(reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    anchors = reference.get("anchors", [])
    _require(isinstance(anchors, list), "reference anchors are not a list")
    rows = []
    for anchor in anchors:
        pos = anchor.get("position")
        box = anchor.get("box_cxcywh")
        _require(isinstance(pos, int) and not isinstance(pos, bool)
                 and isinstance(box, (list, tuple)) and len(box) == 4,
                 "reference anchor has invalid position/box fields")
        vals = np.asarray(box, dtype=np.float64)
        _require(np.isfinite(vals).all() and vals[2] > 0 and vals[3] > 0,
                 "reference anchor geometry is not finite and positive")
        rows.append({"position": int(pos), "box_cxcywh": vals.tolist()})
    return rows


def _fit_movement_rows(cid: str, output: Mapping[str, Any],
                       reference_record: Mapping[str, Any],
                       phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fits = output.get("fits", {})
    if phase == "method":
        refs = reference_record["references"]["teacher"]
        anchors = _anchor_rows(refs)
        expected = {f"teacher_{algorithm}" for algorithm in ALGORITHMS}
        if refs.get("no_update"):
            _require(not fits and all(output.get("reference_receipts", {}).get(name, {}).get("exact_F_reuse")
                                      for name in expected),
                     f"{cid}: locked no-update has fits or lacks exact-F receipts")
            for algorithm in ALGORITHMS:
                name = f"teacher_{algorithm}"
                rows.append({
                    "case_id": cid, "phase": phase, "fit": name,
                    "status": "locked_no_update_exact_F_reuse",
                    "optimizer_steps": 0, "valid_anchor_count": len(anchors),
                    "parameter_delta_norm": None,
                    "all_frame_box_rms_cxcywh": None,
                    "initial_reference_anchor_iou": None,
                    "final_reference_anchor_iou": None,
                    "reference_anchor_iou_is_GT": False,
                })
            return rows
        _require(set(fits) == expected,
                 f"{cid}: fit key set differs from the sealed teacher algorithms")
        source_boxes = _as_numpy(output["source_boxes"], f"{cid}.source_boxes").astype(np.float64)
        frame_ids = list(map(int, output["frame_ids"]))
        for name in sorted(expected):
            fit = fits[name]
            init = _state_flat(fit["initial_state"], f"{cid}.{name}.initial_state")
            final = _state_flat(fit["final_state"], f"{cid}.{name}.final_state")
            _require(init.shape == final.shape,
                     f"{cid}.{name}: initial/final parameter shapes differ")
            after = _as_numpy(fit["final_cached_decoder_boxes"], f"{cid}.{name}.final_boxes").astype(np.float64)
            _require(after.shape == source_boxes.shape,
                     f"{cid}.{name}: cached final/source box shapes differ")
            positions = [a["position"] for a in anchors]
            _require(all(0 <= pos < len(frame_ids) for pos in positions),
                     f"{cid}.{name}: reference position lies outside source grid")
            before_xyxy = _xyxy_from_cxcywh(source_boxes)
            after_xyxy = _xyxy_from_cxcywh(after)
            target_xyxy = _xyxy_from_cxcywh([a["box_cxcywh"] for a in anchors]) if anchors else np.empty((0, 4))
            before_ref = float(np.mean(box_iou(before_xyxy[positions], target_xyxy))) if positions else None
            after_ref = float(np.mean(box_iou(after_xyxy[positions], target_xyxy))) if positions else None
            rows.append({
                "case_id": cid, "phase": phase, "fit": name,
                "status": "updated_fit",
                "optimizer_steps": int(fit["optimizer_steps"]),
                "valid_anchor_count": len(anchors),
                "anchor_positions": positions,
                "anchor_physical_frame_ids": [frame_ids[pos] for pos in positions],
                "parameter_delta_norm": float(np.linalg.norm(final - init)),
                "all_frame_box_rms_cxcywh": float(np.sqrt(np.mean(np.square(after - source_boxes)))),
                "initial_reference_anchor_iou": before_ref,
                "final_reference_anchor_iou": after_ref,
                "reference_anchor_iou_is_GT": False,
            })
    else:
        roles = reference_record["references"]
        expected = {f"{role}_iou" for role in CONTROL_ROLES}
        _require(set(fits) == expected,
                 f"{cid}: V03 control fit key set differs from its three locked references")
        frame_ids = list(map(int, output["frame_ids"]))
        source_boxes = _as_numpy(output["source_boxes"], f"{cid}.source_boxes").astype(np.float64)
        before_xyxy = _xyxy_from_cxcywh(source_boxes)
        for role in CONTROL_ROLES:
            name = f"{role}_iou"
            anchors = _anchor_rows(roles[role])
            fit = fits[name]
            init = _state_flat(fit["initial_state"], f"{cid}.{name}.initial_state")
            final = _state_flat(fit["final_state"], f"{cid}.{name}.final_state")
            _require(init.shape == final.shape, f"{cid}.{name}: parameter shapes differ")
            after = _as_numpy(fit["final_cached_decoder_boxes"], f"{cid}.{name}.final_boxes").astype(np.float64)
            _require(after.shape == source_boxes.shape, f"{cid}.{name}: box shapes differ")
            positions = [a["position"] for a in anchors]
            _require(all(0 <= pos < len(frame_ids) for pos in positions),
                     f"{cid}.{name}: anchor position lies outside source grid")
            after_xyxy = _xyxy_from_cxcywh(after)
            targets = _xyxy_from_cxcywh([a["box_cxcywh"] for a in anchors])
            rows.append({
                "case_id": cid, "phase": phase, "fit": name,
                "reference_role": role,
                "status": "diagnostic_update_even_if_lock_no_update",
                "lock_no_update_ignored_for_control": bool(roles[role].get("no_update", False)),
                "optimizer_steps": int(fit["optimizer_steps"]),
                "valid_anchor_count": len(anchors),
                "anchor_positions": positions,
                "anchor_physical_frame_ids": [frame_ids[pos] for pos in positions],
                "parameter_delta_norm": float(np.linalg.norm(final - init)),
                "all_frame_box_rms_cxcywh": float(np.sqrt(np.mean(np.square(after - source_boxes)))),
                "initial_reference_anchor_iou": float(np.mean(box_iou(before_xyxy[positions], targets))),
                "final_reference_anchor_iou": float(np.mean(box_iou(after_xyxy[positions], targets))),
                "reference_anchor_iou_is_GT": False,
            })
    return rows


def _per_frame_iou(arm: Mapping[str, Any], frame_ids: Sequence[int],
                   width: int, height: int, dataset: str,
                   truth: Mapping[int, Sequence[float]], span: Sequence[int]) -> DenseTube:
    row = {"frame_ids": list(map(int, frame_ids)),
           "input": {"width": int(width), "height": int(height)}}
    return DenseTube(_cxcywh_from_xyxy(_as_numpy(arm["boxes"], "arm.boxes")),
                     row, truth, span, clip=(dataset == "hc2"))


def _subset_delta(reference: DenseTube, arm: DenseTube,
                  selected_fids: set[int] | None,
                  *, selection_status: str) -> dict[str, Any]:
    truth_fids = np.asarray(reference.fids, dtype=np.int64)
    delta = np.asarray(arm.iou - reference.iou, dtype=np.float64) * 100.0
    if selected_fids is None:
        mask = np.zeros(len(truth_fids), dtype=bool)
    else:
        mask = np.asarray([int(fid) in selected_fids for fid in truth_fids], dtype=bool)
    count = int(mask.sum())
    return {
        "selection_status": selection_status,
        "GT_known_frame_count": count,
        "mean_delta_sIoU_pp": float(delta[mask].mean()) if count else None,
        "positive_count": int(np.sum(delta[mask] > NEUTRAL_EPS)) if count else 0,
        "negative_count": int(np.sum(delta[mask] < -NEUTRAL_EPS)) if count else 0,
        "neutral_count": int(np.sum(np.abs(delta[mask]) <= NEUTRAL_EPS)) if count else 0,
        "unknown_count": 1 if count == 0 else 0,
    }


def _anchor_diagnostics(cid: str, arms: Mapping[str, Mapping[str, Any]],
                        frame_ids: Sequence[int], width: int, height: int,
                        dataset: str, truth: Mapping[int, Sequence[float]],
                        span: Sequence[int], anchor_positions_by_role: Mapping[str, Sequence[int]],
                        original4_positions: Sequence[int]) -> dict[str, Any]:
    frame_ids = list(map(int, frame_ids))
    _require("F" in arms, f"{cid}: spatial baseline F is absent")
    baseline = _per_frame_iou(arms["F"], frame_ids, width, height, dataset, truth, span)
    original4_positions = sorted(set(map(int, original4_positions)))
    for role, positions in anchor_positions_by_role.items():
        _require(all(0 <= int(p) < len(frame_ids) for p in positions),
                 f"{cid}.{role}: supervision position lies outside the source frame grid")
    _require(all(0 <= p < len(frame_ids) for p in original4_positions),
             f"{cid}: original-four position lies outside the source frame grid")
    original4_fids = {frame_ids[p] for p in original4_positions}
    all_fids = set(map(int, baseline.fids))
    arm_dense = {}
    for name, arm in arms.items():
        metric = _per_frame_iou(arm, frame_ids, width, height, dataset, truth, span)
        _require(np.array_equal(metric.fids, baseline.fids),
                 f"{cid}.{name}: GT frame support differs from F")
        arm_dense[name] = metric
    role_rows = {}
    for role, raw_positions in sorted(anchor_positions_by_role.items()):
        positions = sorted(set(map(int, raw_positions)))
        selected_fids = {frame_ids[p] for p in positions}
        role_rows[role] = {
            "actual_valid_supervision_anchor_positions": positions,
            "actual_valid_supervision_anchor_physical_frame_ids": [frame_ids[p] for p in positions],
            "arms": {
                name: {
                    "vs_F_observed_actual_valid_supervision_anchors": _subset_delta(
                        baseline, metric, selected_fids,
                        selection_status="no_valid_supervision_anchor" if not positions else "measured"),
                    "vs_F_unobserved_actual_valid_supervision_anchors": _subset_delta(
                        baseline, metric, all_fids - selected_fids,
                        selection_status="no_valid_supervision_anchor" if not positions else "measured"),
                }
                for name, metric in arm_dense.items()
            },
        }
    return {
        "by_reference_role": role_rows,
        "original4_observation_positions": original4_positions,
        "original4_observation_physical_frame_ids": [frame_ids[p] for p in original4_positions],
        "arms_vs_F": {
            name: {
                "at_original4_observation_positions": _subset_delta(
                    baseline, metric, original4_fids, selection_status="measured"),
                "outside_original4_observation_positions": _subset_delta(
                    baseline, metric, all_fids - original4_fids, selection_status="measured"),
            }
            for name, metric in arm_dense.items()
        },
        "interpretation": "per-frame GT IoU deltas against F; actual teacher anchors and the original four observations are separate masks",
    }


def _score_case(cid: str, cohort: str, roster: Mapping[str, Any],
                source_case: Mapping[str, Any], output: Mapping[str, Any],
                references: Mapping[str, Any], candidate_input: Mapping[str, Any],
                truths: Mapping[str, Any], official: Any, independent: Any,
                phase: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    truth = {int(fid): [float(v) for v in box]
             for fid, box in truths["truth"].items()}
    span = [int(v) for v in truths["span"]]
    width, height = int(source_case["width"]), int(source_case["height"])
    frame_ids = list(map(int, output["frame_ids"]))
    _require(frame_ids == list(map(int, source_case["frame_ids"])),
             f"{cid}: full physical source grid differs")
    _require(output.get("dataset") == roster["dataset"] == source_case["dataset"],
             f"{cid}: dataset binding differs")
    arms = output.get("arms", {})
    _require("F" in arms and "Projection" in arms,
             f"{cid}: required source F/Projection baselines are absent")
    if cid in OLD_IDS:
        _require("T" in arms, f"{cid}: old-eight exact temporal T is absent")
    else:
        _require("T" not in arms and not any(name.startswith("T_teacher_") for name in arms),
                 f"{cid}: additional-eight case unexpectedly has a T arm")

    metric_rows: list[dict[str, Any]] = []
    arm_metrics: dict[str, Any] = {}
    per_frame: dict[str, DenseTube] = {}
    metric_errors = []
    for arm_name, arm in sorted(arms.items()):
        scored = score_tube_xyxy(
            arm["boxes"], frame_ids, width, height, arm["interval"],
            truth, span, str(roster["dataset"]), official=official, independent=independent,
        )
        arm_metrics[arm_name] = scored["metrics"]
        metric_errors.append(float(scored["official_independent_max_abs_error"]))
        per_frame[arm_name] = _per_frame_iou(
            arm, frame_ids, width, height, str(roster["dataset"]), truth, span)
        metric_rows.append({
            "case_id": cid, "cohort": cohort, "dataset": roster["dataset"],
            "phase": phase, "arm": arm_name,
            "status": "scored",
            "metrics": scored["metrics"],
            "official_independent_max_abs_error": scored["official_independent_max_abs_error"],
            "source_grid_frames": scored["source_grid_frames"],
            "GT_support_frame_count": scored["gt_support_frame_count"],
            "interval_frame_bounds_half_open": scored["interval_frame_bounds_half_open"],
            "source_case_sha256": roster["source_case_sha256"],
        })

    anchor_positions_by_role: dict[str, list[int]] = {}
    if phase == "method":
        teacher_ref = references["references"]["teacher"]
        anchors = _anchor_rows(teacher_ref)
        if not teacher_ref.get("no_update"):
            anchor_positions_by_role["teacher"] = [a["position"] for a in anchors]
        else:
            anchor_positions_by_role["teacher"] = []
        _require(output.get("supervision_anchors", {}).get("teacher", [])
                 == teacher_ref.get("anchors", []),
                 f"{cid}: saved supervision anchors differ from sealed teacher reference")
        no_update = bool(teacher_ref.get("no_update"))
        if no_update:
            _require(not output.get("fits") and all(
                output.get("reference_receipts", {}).get(f"teacher_{alg}", {}).get("exact_F_reuse")
                for alg in ALGORITHMS),
                f"{cid}: no-update reference did not preserve exact F arms")
        else:
            _require(set(output.get("fits", {})) == {f"teacher_{a}" for a in ALGORITHMS},
                     f"{cid}: updating case lacks the fixed three student fits")
    else:
        _require(set(references["references"]) == set(CONTROL_ROLES),
                 f"{cid}: V03 control references differ")
        _require(set(output.get("supervision_anchors", {})) == set(CONTROL_ROLES),
                 f"{cid}: V03 output supervision anchor roles differ")
        _require(set(output.get("reference_control_roles", {})) == set(CONTROL_ROLES),
                 f"{cid}: V03 output role receipts differ")
        for role in CONTROL_ROLES:
            role_anchors = _anchor_rows(references["references"][role])
            anchor_positions_by_role[role] = [a["position"] for a in role_anchors]
            _require(output["supervision_anchors"].get(role)
                     == references["references"][role].get("anchors", []),
                     f"{cid}: saved {role} anchors differ from its fixed reference")
            _require(f"{role}_iou" in arms,
                     f"{cid}: V03 fixed reference output {role} is absent")

    original4_positions = list(map(int, candidate_input["positions"]))
    anchor_report = _anchor_diagnostics(
        cid, arms, frame_ids, width, height, str(roster["dataset"]),
        truth, span, anchor_positions_by_role, original4_positions,
    )
    fit_rows = _fit_movement_rows(cid, output, references, phase)

    comparisons = []
    for arm_name in sorted(arms):
        if arm_name == "F":
            comparisons.append({"case_id": cid, "cohort": cohort, "arm": arm_name,
                                "baseline": "F", "category": "baseline",
                                "delta_vIoU_pp": 0.0, "status": "scored"})
            continue
        baseline_name = "T" if arm_name.startswith("T_teacher_") else "F"
        if baseline_name not in arm_metrics:
            comparisons.append({"case_id": cid, "cohort": cohort, "arm": arm_name,
                                "baseline": baseline_name, "category": "unknown",
                                "delta_vIoU_pp": None, "status": "baseline_unavailable"})
            continue
        delta = (float(arm_metrics[arm_name]["m_vIoU"])
                 - float(arm_metrics[baseline_name]["m_vIoU"])) * 100.0
        category = ("positive" if delta > NEUTRAL_EPS else
                    "negative" if delta < -NEUTRAL_EPS else "neutral")
        no_update = (phase == "method"
                     and arm_name.startswith("teacher_")
                     and output.get("reference_receipts", {}).get(arm_name, {}).get("exact_F_reuse"))
        comparisons.append({
            "case_id": cid, "cohort": cohort, "dataset": roster["dataset"],
            "phase": phase, "arm": arm_name, "baseline": baseline_name,
            "category": category, "delta_vIoU_pp": delta, "status": "scored",
            "locked_no_update_exact_reuse": bool(no_update),
        })
    if phase == "method" and cid not in OLD_IDS:
        for arm_name in ("T", *(f"T_teacher_{algorithm}" for algorithm in ALGORITHMS)):
            comparisons.append({
                "case_id": cid, "cohort": cohort, "dataset": roster["dataset"],
                "phase": phase, "arm": arm_name,
                "baseline": "F" if arm_name == "T" else "T",
                "category": "unknown", "delta_vIoU_pp": None,
                "status": "not_created_by_additional8_scope",
                "explanation": "additional8 has no T arm; no new temporal output was fabricated",
            })

    return ({
        "case_id": cid, "dataset": roster["dataset"], "cohort": cohort,
        "phase": phase,
        "source_case_sha256": roster["source_case_sha256"],
        "source_model_state_sha256": output["source_model_state_sha256"],
        "source_pixel_sha256": output["pixel_sha256"],
        "frame_ids": frame_ids,
        "GT_support_frame_count": len(truth),
        "GT_span": span,
        "arms": arm_metrics,
        "anchor_diagnostics": anchor_report,
        "fit_count": len(output.get("fits", {})),
        "optimizer_steps": sum(int(f["optimizer_steps"]) for f in output.get("fits", {}).values()),
        "fit_movement_rows": fit_rows,
        "temporal_T_status": ("exact_old_T_reuse" if "T" in arms else "unavailable_for_additional8"),
        "metric_max_abs_error": max(metric_errors) if metric_errors else None,
    }, metric_rows, comparisons)


def _score_candidates(cid: str, roster: Mapping[str, Any], source_case: Mapping[str, Any],
                      candidate_input: Mapping[str, Any], output: Mapping[str, Any],
                      truths: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    truth = {int(fid): [float(v) for v in box] for fid, box in truths["truth"].items()}
    width, height = int(source_case["width"]), int(source_case["height"])
    output_fids = list(map(int, output["frame_ids"]))
    source_boxes = _as_numpy(output["arms"]["F"]["boxes"], f"{cid}.F.boxes").astype(np.float64)
    source_pos = {fid: i for i, fid in enumerate(output_fids)}
    original_positions = list(map(int, candidate_input["positions"]))
    original_fids = list(map(int, candidate_input["physical_frame_ids"]))
    _require(len(original_positions) == len(original_fids) == 4,
             f"{cid}: fixed original-four positions are malformed")
    _require(all(fid in source_pos for fid in original_fids),
             f"{cid}: an original-four candidate frame is absent from the source grid")
    rows = []
    cohort = str(roster["cohort"]) if phase == "method" else "reference_control_V03"
    for candidate_order, candidate in enumerate(candidate_input["candidates"]):
        raw_boxes = candidate.get("boxes_xyxy")
        cand_fids = list(map(int, candidate["physical_frame_ids"]))
        cand_positions = list(map(int, candidate["positions"]))
        schema_ok = (isinstance(raw_boxes, list) and len(raw_boxes) == 4
                     and cand_fids == original_fids and cand_positions == original_positions)
        box_rows: list[np.ndarray | None] = []
        box_valid_flags: list[bool] = []
        box_missing_flags: list[bool] = []
        invalid_geometry_count = 0
        if schema_ok:
            for raw_box in raw_boxes:
                if raw_box is None:
                    box_rows.append(None)
                    box_valid_flags.append(False)
                    box_missing_flags.append(True)
                    continue
                try:
                    box = np.asarray(raw_box, dtype=np.float64)
                except (TypeError, ValueError):
                    box = np.empty((0,), dtype=np.float64)
                if box.shape != (4,) or not np.isfinite(box).all():
                    box_rows.append(None)
                    box_valid_flags.append(False)
                    box_missing_flags.append(False)
                    invalid_geometry_count += 1
                elif np.any(box[2:] <= box[:2]) or np.any(box < -1e-7) or np.any(box > 1.0 + 1e-7):
                    box_rows.append(box)
                    box_valid_flags.append(False)
                    box_missing_flags.append(False)
                    invalid_geometry_count += 1
                else:
                    box_rows.append(box)
                    box_valid_flags.append(True)
                    box_missing_flags.append(False)
        else:
            box_rows = [None, None, None, None]
            box_valid_flags = [False, False, False, False]
            box_missing_flags = [False, False, False, False]
        present_rows = [box for box in box_rows if box is not None]
        positive = bool(present_rows) and all(np.all(box[2:] > box[:2]) for box in present_rows)
        in_unit = bool(present_rows) and all(np.all(box >= -1e-7) and np.all(box <= 1.0 + 1e-7)
                                              for box in present_rows)
        missing_candidate_count = sum(box_missing_flags)
        if not schema_ok:
            geometry_status = "malformed_candidate_geometry"
        elif invalid_geometry_count:
            geometry_status = "invalid_unclamped_candidate_geometry"
        elif missing_candidate_count:
            geometry_status = "partial_explicit_missing_tracklet_support"
        else:
            geometry_status = "valid_full_original4_support"
        tracklet = candidate.get("tracklet")
        tracklet_trajectory = tracklet.get("trajectory", []) if isinstance(tracklet, Mapping) else []
        tracklet_missing = sum(
            isinstance(item, Mapping) and item.get("status") == "missing"
            for item in tracklet_trajectory
        )
        per_anchor = []
        candidate_known, paired_source_known, source_all_known = [], [], []
        candidate_zeroed_known, paired_source_zeroed_known, source_all_zeroed_known = [], [], []
        gt_known_count = 0
        candidate_missing_gt_known = 0
        candidate_invalid_gt_known = 0
        gt_missing_count = 0
        for index, fid in enumerate(original_fids):
            known = fid in truth
            candidate_box = box_rows[index]
            source_iou = None
            if known:
                gt_known_count += 1
                gt_norm = np.asarray(truth[fid], dtype=np.float64) / np.asarray(
                    [width, height, width, height], dtype=np.float64)
                src_box = source_boxes[source_pos[fid]:source_pos[fid] + 1]
                source_iou = float(box_iou(src_box, gt_norm[None, :])[0])
                source_all_known.append(source_iou)
                source_all_zeroed_known.append(source_iou)
            else:
                gt_missing_count += 1
            if known and candidate_box is not None and schema_ok and box_valid_flags[index]:
                cand_iou = float(box_iou(candidate_box[None, :], gt_norm[None, :])[0])
                assert source_iou is not None
                candidate_known.append(cand_iou)
                paired_source_known.append(source_iou)
                candidate_zeroed_known.append(cand_iou)
                paired_source_zeroed_known.append(source_iou)
                candidate_support = "present_valid"
            else:
                cand_iou = None
                if known:
                    candidate_zeroed_known.append(0.0)
                    paired_source_zeroed_known.append(source_iou)
                    if box_missing_flags[index]:
                        candidate_missing_gt_known += 1
                    elif not box_valid_flags[index]:
                        candidate_invalid_gt_known += 1
                candidate_support = (
                    "unknown_missing_tracklet_support" if box_missing_flags[index] else
                    "unknown_invalid_or_malformed_geometry" if not box_valid_flags[index] else
                    "present_valid")
            per_anchor.append({
                "position": original_positions[index], "physical_frame_id": fid,
                "GT_support": "known" if known else "unknown_missing_label",
                "candidate_support": candidate_support,
                "candidate_IoU": cand_iou,
                "F_source_IoU_at_anchor": source_iou,
                "paired_F_source_same_anchor_IoU": source_iou if cand_iou is not None else None,
                "candidate_minus_paired_F_source_IoU": (
                    cand_iou - source_iou if cand_iou is not None and source_iou is not None else None),
            })
        present_known_count = len(candidate_known)
        rows.append({
            "case_id": cid, "dataset": roster["dataset"], "cohort": cohort,
            "phase": phase,
            "candidate_order_locked": candidate_order,
            "candidate_id": candidate.get("candidate_id"),
            "window_id": candidate.get("window_id"),
            "is_Source_candidate": bool(candidate.get("is_Source", False)),
            "tracklet_status": (tracklet.get("status") if isinstance(tracklet, Mapping) else "source_candidate"),
            "tracklet_missing_position_count": int(tracklet_missing),
            "candidate_valid_observation_count": candidate.get("valid_observation_count"),
            "geometry_status": geometry_status,
            "candidate_box_present_anchor_count": 4 - missing_candidate_count,
            "candidate_box_valid_anchor_count": sum(box_valid_flags),
            "candidate_box_missing_anchor_count": missing_candidate_count,
            "candidate_invalid_geometry_anchor_count": invalid_geometry_count,
            "positive_area_all_present_boxes": positive,
            "within_normalized_unit_box_all_present_boxes_with_1e7_tolerance": in_unit,
            "GT_known_original4_anchor_count": gt_known_count,
            "GT_missing_original4_anchor_count": gt_missing_count,
            "candidate_missing_where_GT_known_count": candidate_missing_gt_known,
            "candidate_invalid_where_GT_known_count": candidate_invalid_gt_known,
            "candidate_present_and_GT_known_anchor_count": present_known_count,
            "present_anchor_GT_IoU_mean_known_only": float(np.mean(candidate_known)) if present_known_count else None,
            "paired_F_source_same_anchor_GT_IoU_mean_known_only": float(np.mean(paired_source_known)) if present_known_count else None,
            "F_source_GT_IoU_mean_all_GT_known_original4": float(np.mean(source_all_known)) if source_all_known else None,
            "present_minus_paired_F_source_mean_IoU_known_only": (
                float(np.mean(np.asarray(candidate_known) - np.asarray(paired_source_known)))
                if present_known_count else None),
            "common_denominator_original4_GT_known_missing_as_zero_candidate_IoU": (
                float(np.sum(candidate_zeroed_known) / gt_known_count) if gt_known_count else None),
            "common_denominator_original4_GT_known_missing_as_zero_paired_F_source_IoU": (
                float(np.sum(paired_source_zeroed_known) / gt_known_count) if gt_known_count else None),
            "common_denominator_original4_GT_known_missing_as_zero_delta_IoU": (
                float((np.sum(candidate_zeroed_known) - np.sum(paired_source_zeroed_known)) / gt_known_count)
                if gt_known_count else None),
            "common_denominator_original4_GT_known_missing_as_zero_F_source_all_known_diagnostic": (
                float(np.sum(source_all_zeroed_known) / gt_known_count) if gt_known_count else None),
            "missing_as_zero_is_support_diagnostic_only": True,
            "GT_missing_excluded_from_zero_fill_denominator": True,
            "candidate_missing_is_zero_filled_only_in_GT_known_diagnostic": True,
            "missing_candidate_support_is_unknown_not_coordinate_failure": True,
            "per_original4_anchor": per_anchor,
            "GT_used_to_generate_or_select_candidate": False,
            "source_case_sha256": roster["source_case_sha256"],
        })
    return rows


def _score_locked_control_references(cid: str, roster: Mapping[str, Any],
                                    source_case: Mapping[str, Any],
                                    reference_record: Mapping[str, Any],
                                    candidate_input: Mapping[str, Any],
                                    output: Mapping[str, Any],
                                    truths: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Score V03's three fixed reference geometries, never its motion pool.

    The scene-object-ID candidate barrier is bound to the V03 run as context,
    but these control roles were locked independently in
    REFERENCE_CONTROL_REFERENCE_LOCK.json. Their anchor geometry is the
    correct input for the control's fixed-reference diagnostic.
    """
    refs = reference_record.get("references")
    _require(isinstance(refs, Mapping) and set(refs) == set(CONTROL_ROLES),
             f"{cid}: V03 reference-control lock does not contain the exact three roles")
    expected_positions = list(map(int, candidate_input["positions"]))
    expected_fids = list(map(int, candidate_input["physical_frame_ids"]))
    for role in CONTROL_ROLES:
        anchors = refs[role].get("anchors", [])
        _require(len(anchors) == 4,
                 f"{cid}.{role}: locked control reference must contain four anchors")
        _require([int(a["position"]) for a in anchors] == expected_positions
                 and [int(a["physical_frame_id"]) for a in anchors] == expected_fids,
                 f"{cid}.{role}: locked control anchors do not match the original four frames")

    locked_candidates = []
    for role in CONTROL_ROLES:
        ref = refs[role]
        cxcywh = np.asarray([a["box_cxcywh"] for a in ref["anchors"]], dtype=np.float64)
        _require(cxcywh.shape == (4, 4) and np.isfinite(cxcywh).all()
                 and np.all(cxcywh[:, 2:] > 0),
                 f"{cid}.{role}: locked control reference geometry is malformed")
        locked_candidates.append({
            "candidate_id": f"locked_reference:{role}",
            "old_candidate_id": ref.get("old_candidate_id"),
            "window_id": ref.get("window_id"),
            "is_Source": role == "Source",
            "reference_role": role,
            "boxes_xyxy": _xyxy_from_cxcywh(cxcywh).tolist(),
            "physical_frame_ids": expected_fids,
            "positions": expected_positions,
            "valid_observation_count": 4,
            "tracklet": {"status": "locked_reference", "trajectory": []},
        })
    locked_input = {
        "positions": expected_positions,
        "physical_frame_ids": expected_fids,
        "candidates": locked_candidates,
    }
    rows = _score_candidates(cid, roster, source_case, locked_input, output,
                             truths, "reference_control")
    for row, role, ref in zip(rows, CONTROL_ROLES,
                              (refs[name] for name in CONTROL_ROLES), strict=True):
        row["row_kind"] = "locked_reference_control_anchor_geometry"
        row["reference_role"] = role
        row["candidate_id"] = f"locked_reference:{role}"
        row["old_candidate_id"] = ref.get("old_candidate_id")
        row["candidate_geometry_source"] = "REFERENCE_CONTROL_REFERENCE_LOCK"
        row["candidate_barrier_motion_pool_member"] = False
        row["GT_used_to_generate_or_select_candidate"] = False
    return rows


def _group_summaries(case_rows: Sequence[Mapping[str, Any]],
                     comparison_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    keys = sorted({(str(row["cohort"]), str(row["dataset"])) for row in case_rows})
    for cohort, dataset in keys:
        selected = [row for row in case_rows if row["cohort"] == cohort and row["dataset"] == dataset]
        arm_names = sorted(
            {name for row in selected for name in row["arms"]}
            | {str(row["arm"]) for row in comparison_rows
               if row.get("cohort") == cohort and row.get("dataset") == dataset}
        )
        for arm_name in arm_names:
            scored = [row["arms"][arm_name] for row in selected if arm_name in row["arms"]]
            metric_means = ({
                key: statistics.mean(float(item[key]) for item in scored) * 100.0
                for key in ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT")
            } if scored else {key: None for key in ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT")})
            comps = [row for row in comparison_rows
                     if row.get("cohort") == cohort and row.get("dataset") == dataset
                     and row.get("arm") == arm_name]
            groups.append({
                "cohort": cohort, "dataset": dataset, "arm": arm_name,
                "scored_case_count": len(scored),
                "unknown_case_count": len(selected) - len(scored),
                "metrics_percent": metric_means,
                "positive_count": sum(row.get("category") == "positive" for row in comps),
                "negative_count": sum(row.get("category") == "negative" for row in comps),
                "neutral_count": sum(row.get("category") == "neutral" for row in comps),
                "unknown_count": sum(row.get("category") == "unknown" for row in comps),
                "worst_delta_vIoU_pp": min(
                    (float(row["delta_vIoU_pp"]) for row in comps
                     if row.get("delta_vIoU_pp") is not None), default=None),
                "neutral_tolerance_pp": NEUTRAL_EPS,
                "classification_is_reporting_only": True,
            })
    return groups


def run(*, replace_report: bool = False) -> dict[str, Any]:
    torch.set_num_threads(2)
    report_path = BASE / "ROOT_POSTSEAL_REPORT.json"
    previous_report_replaced = False
    previous_report_sha256 = None
    previous_report_schema = None
    previous_scoring_script_sha256 = None
    if report_path.exists():
        _require(replace_report,
                 "ROOT_POSTSEAL_REPORT.json exists; use --replace-report only to correct this scorer's prior report")
        previous = _read_json(report_path)
        _require(previous.get("schema") == "stvg_motion_guided_postseal_v1"
                 and previous.get("status") == "pass"
                 and previous.get("GT_read") is True,
                 "existing post-seal report is not a successful report from this scorer")
        previous_report_replaced = True
        previous_report_sha256 = _sha_file(report_path)
        previous_report_schema = previous.get("schema")
        previous_scoring_script_sha256 = previous.get("scoring_script_sha256")
        report_capture = BASE / "source_audit/ROOT_POSTSEAL_REPORT_PRE_SEMANTICS_FIX.json"
        _require(report_capture.is_file() and _sha_file(report_capture) == previous_report_sha256,
                 "prior report must be captured byte-for-byte before semantic correction")
        prior_script_capture = BASE / "source_audit/score_stvg_motion_guided_postseal_v1_PRE_SEMANTICS_FIX.py"
        _require(prior_script_capture.is_file()
                 and _sha_file(prior_script_capture) == previous_scoring_script_sha256,
                 "prior scoring source snapshot does not match the report's script hash")

    roster_path = BASE / "ROSTER_LOCK.json"
    roster_doc = _read_json(roster_path)
    _require(roster_doc.get("status") == "locked" and roster_doc.get("GT_read") is False,
             "16-case roster is not a locked GT-free roster")
    roster_rows = roster_doc.get("records", [])
    roster_by_id = {str(r["case_id"]): r for r in roster_rows}
    _require(len(roster_rows) == 16 and set(roster_by_id) == METHOD_IDS,
             "locked method roster differs from the fixed 16-case design")
    _require(sum(r.get("cohort") == "old8" for r in roster_rows) == 8
             and sum(r.get("cohort") == "additional8" for r in roster_rows) == 8,
             "old8/additional8 cohort counts differ")
    _require(all(r.get("GT_read") is not True for r in roster_rows),
             "roster record unexpectedly says GT was read")

    method_seal = _verify_phase_seal("method", METHOD_IDS, expected_fit_count=18)
    control_seal = _verify_phase_seal("reference_control", {CONTROL_ID}, expected_fit_count=3)
    old_seal = _verify_old_temporal_seal()
    candidate_input_by_id, candidate_input_binding = _verify_candidate_input_barrier(roster_by_id)

    method_lock_path = BASE / "METHOD_REFERENCE_LOCK.json"
    control_lock_path = BASE / "REFERENCE_CONTROL_REFERENCE_LOCK.json"
    method_lock = _read_json(method_lock_path)
    control_lock = _read_json(control_lock_path)
    _require(method_lock.get("status") == "locked" and method_lock.get("GT_read") is False
             and control_lock.get("status") == "locked" and control_lock.get("GT_read") is False,
             "sealed method/reference-control locks are not GT-free")
    method_references = {str(r["case_id"]): r for r in method_lock["records"]}
    control_references = {str(r["case_id"]): r for r in control_lock["records"]}
    _require(set(method_references) == METHOD_IDS and set(control_references) == {CONTROL_ID},
             "reference lock inventories differ from the fixed roster")
    method_outputs = {}
    for cid in sorted(METHOD_IDS):
        path = method_seal["files"][cid]["path"]
        value = torch.load(path, map_location="cpu", weights_only=False)
        _assert_cpu_tree(value, f"method.{cid}")
        method_outputs[cid] = value
    control_output_path = control_seal["files"][CONTROL_ID]["path"]
    control_output = torch.load(control_output_path, map_location="cpu", weights_only=False)
    _assert_cpu_tree(control_output, f"reference_control.{CONTROL_ID}")

    source_cases = {}
    for cid, record in roster_by_id.items():
        source_path = _safe_path(str(record["source_case_path"]))
        _require(source_path.is_file() and _sha_file(source_path) == record["source_case_sha256"],
                 f"{cid}: locked source-case file hash mismatch")
        source = load_source_case(source_path)
        _require(source.get("_source_case_sha256") == record["source_case_sha256"]
                 and source.get("GT_read") is False
                 and source.get("case_id") == cid
                 and source.get("dataset") == record["dataset"],
                 f"{cid}: source case no longer matches the locked source custody")
        source_cases[cid] = source
        _check_output_identity(method_outputs[cid], cid, record, source,
                               _sha_file(method_lock_path))
        candidate_input = candidate_input_by_id[cid]
        _require(candidate_input.get("source_model_state_sha256") == source["source_model_state_sha256"]
                 and candidate_input.get("source_pixel_sha256") == source["pixel_sha256"],
                 f"{cid}: candidate source state/pixel binding differs")
    control_record = roster_by_id[CONTROL_ID]
    _check_output_identity(control_output, CONTROL_ID, control_record,
                           source_cases[CONTROL_ID], _sha_file(control_lock_path))

    # Old eight retain their exact sealed T arm. The additional eight have no T.
    for cid in sorted(OLD_IDS):
        old_path = old_seal["files"][cid]["path"]
        old_output = torch.load(old_path, map_location="cpu", weights_only=False)
        _assert_cpu_tree(old_output, f"old_temporal.{cid}")
        new = method_outputs[cid]
        _tensor_equal(new["arms"]["T"]["boxes"], old_output["arms"]["T"]["boxes"], f"{cid}.exact_T_boxes")
        _require(new["arms"]["T"]["interval"] == old_output["arms"]["T"]["interval"],
                 f"{cid}: exact old T interval differs")
        for algorithm in ALGORITHMS:
            inherited = new["arms"][f"T_teacher_{algorithm}"]
            _require(inherited["interval"] == old_output["arms"]["T"]["interval"],
                     f"{cid}: T_teacher_{algorithm} does not retain the old T interval")
            expected_boxes = (old_output["arms"]["T"]["boxes"]
                              if f"teacher_{algorithm}" not in new.get("fits", {})
                              else new["arms"][f"teacher_{algorithm}"]["boxes"])
            _tensor_equal(inherited["boxes"], expected_boxes,
                          f"{cid}.T_teacher_{algorithm}.spatial_reuse")
    for cid in sorted(METHOD_IDS - OLD_IDS):
        _require("T" not in method_outputs[cid]["arms"]
                 and not any(key.startswith("T_teacher_") for key in method_outputs[cid]["arms"]),
                 f"{cid}: additional-eight output contains an unplanned T arm")
    _tensor_equal(control_output["arms"]["T"]["boxes"],
                  torch.load(old_seal["files"][CONTROL_ID]["path"], map_location="cpu", weights_only=False)["arms"]["T"]["boxes"],
                  "V03.exact_old_T_boxes")

    # No target annotation is requested until both new student barriers and
    # the independent CPU arithmetic readbacks have passed every check above.
    source_minimal = {
        cid: {"dataset": case["dataset"], "query_ordinal": case["query_ordinal"]}
        for cid, case in source_cases.items()
    }
    truths, gt_bindings = _truths_after_seal(source_minimal)
    _require(set(truths) == METHOD_IDS,
             "official target reader did not return dense truth for every locked case")
    official, independent, metric_contract = _metric_functions()

    case_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for cid in sorted(METHOD_IDS):
        record = roster_by_id[cid]
        row, metrics, comparisons = _score_case(
            cid, str(record["cohort"]), record, source_cases[cid], method_outputs[cid],
            method_references[cid], candidate_input_by_id[cid], truths[cid],
            official, independent, "method",
        )
        case_rows.append(row)
        metric_rows.extend(metrics)
        comparison_rows.extend(comparisons)
        fit_rows.extend(row["fit_movement_rows"])
        candidate_rows.extend(_score_candidates(
            cid, record, source_cases[cid], candidate_input_by_id[cid], method_outputs[cid],
            truths[cid], "method"))

    v03_row, v03_metrics, v03_comparisons = _score_case(
        CONTROL_ID, "reference_control_V03", control_record, source_cases[CONTROL_ID],
        control_output, control_references[CONTROL_ID], candidate_input_by_id[CONTROL_ID],
        truths[CONTROL_ID], official, independent, "reference_control",
    )
    case_rows.append(v03_row)
    metric_rows.extend(v03_metrics)
    comparison_rows.extend(v03_comparisons)
    fit_rows.extend(v03_row["fit_movement_rows"])
    control_anchor_rows = _score_locked_control_references(
        CONTROL_ID, control_record, source_cases[CONTROL_ID],
        control_references[CONTROL_ID], candidate_input_by_id[CONTROL_ID],
        control_output, truths[CONTROL_ID])

    candidate_summary = []
    for phase, cohort, dataset, is_source in sorted({
        (str(r["phase"]), str(r["cohort"]), str(r["dataset"]), bool(r["is_Source_candidate"]))
        for r in candidate_rows
    }):
        values = [r["present_anchor_GT_IoU_mean_known_only"] for r in candidate_rows
                  if r["phase"] == phase and r["cohort"] == cohort and r["dataset"] == dataset
                  and bool(r["is_Source_candidate"]) == is_source
                  and r["present_anchor_GT_IoU_mean_known_only"] is not None]
        selected_rows = [r for r in candidate_rows
                         if r["phase"] == phase and r["cohort"] == cohort and r["dataset"] == dataset
                         and bool(r["is_Source_candidate"]) == is_source]
        candidate_summary.append({
            "phase": phase, "cohort": cohort, "dataset": dataset,
            "is_Source_candidate": is_source,
            "candidate_case_rows": len(selected_rows),
            "known_support_rows": len(values),
            "unknown_support_rows": len(selected_rows) - len(values),
            "mean_present_anchor_GT_IoU_known_only": statistics.mean(values) if values else None,
            "no_GT_based_candidate_selection": True,
        })

    metric_errors = [float(row["official_independent_max_abs_error"]) for row in metric_rows]
    groups = _group_summaries(case_rows[:len(METHOD_IDS)], comparison_rows)
    control_group = _group_summaries([v03_row], v03_comparisons)
    actual_fit_statuses = {"updated_fit", "diagnostic_update_even_if_lock_no_update"}
    actual_fit_count = sum(row.get("status") in actual_fit_statuses for row in fit_rows)
    no_fit_placeholder_count = sum(
        row.get("status") == "locked_no_update_exact_F_reuse" for row in fit_rows)
    control_anchor_summary = [{
        "reference_role": row["reference_role"],
        "row_kind": row["row_kind"],
        "present_anchor_GT_IoU_mean_known_only": row["present_anchor_GT_IoU_mean_known_only"],
        "paired_F_source_same_anchor_GT_IoU_mean_known_only": row[
            "paired_F_source_same_anchor_GT_IoU_mean_known_only"],
        "present_minus_paired_F_source_mean_IoU_known_only": row[
            "present_minus_paired_F_source_mean_IoU_known_only"],
        "GT_known_original4_anchor_count": row["GT_known_original4_anchor_count"],
        "GT_missing_original4_anchor_count": row["GT_missing_original4_anchor_count"],
        "candidate_missing_where_GT_known_count": row["candidate_missing_where_GT_known_count"],
    } for row in control_anchor_rows]
    report = {
        "schema": "stvg_motion_guided_postseal_v1",
        "status": "pass",
        "completed": True,
        "student_scope": "locked 16-case method plus fixed V03 reference_control",
        "scoring_script_sha256": _sha_file(Path(__file__).resolve()),
        "previous_internal_report_replaced": previous_report_replaced,
        "prior_report_capture": ({
            "path": "artifacts/stvg_motion_guided_instance_opd_v1/source_audit/ROOT_POSTSEAL_REPORT_PRE_SEMANTICS_FIX.json",
            "sha256": previous_report_sha256,
            "schema": previous_report_schema,
            "prior_scoring_script_sha256": previous_scoring_script_sha256,
            "prior_scoring_script_capture_path": "artifacts/stvg_motion_guided_instance_opd_v1/source_audit/score_stvg_motion_guided_postseal_v1_PRE_SEMANTICS_FIX.py",
        } if previous_report_replaced else None),
        "report_semantics_correction": (
            "V03 control reference quality is scored from its three fixed geometries in "
            "REFERENCE_CONTROL_REFERENCE_LOCK; its five scene-object candidate-input rows "
            "are context only and excluded from candidate quality. Fit movement diagnostics "
            "contain actual fit rows plus locked no-update placeholders."
        ),
        "GT_read": True,
        "GT_used_for_method_update": False,
        "GT_used_to_generate_or_select_candidates": False,
        "model_loaded": False,
        "GPU_initialized": False,
        "pixels_decoded": False,
        "old8_additional8_kept_separate": True,
        "old8_T_status": "exact reuse from sealed old temporal student outputs only",
        "additional8_T_status": "no T arm created or scored",
        "seals": {"method": {k: v for k, v in method_seal.items() if k not in ("barrier", "files")},
                  "reference_control": {k: v for k, v in control_seal.items() if k not in ("barrier", "files")},
                  "old_temporal": {k: v for k, v in old_seal.items() if k != "files"}},
        "candidate_input_binding": candidate_input_binding,
        "official_metric_contract": {
            "path": str(METRIC_CONTRACT.relative_to(ROOT)),
            "sha256": _sha_file(METRIC_CONTRACT),
            "schema": metric_contract.get("schema"),
            "official_metric": "original official dense evaluator",
            "independent_metric": "separately implemented dense evaluator",
            "comparison_tolerance_absolute": 2e-9,
        },
        "GT_source_bindings": gt_bindings,
        "method_reference_lock_sha256": _sha_file(method_lock_path),
        "reference_control_lock_sha256": _sha_file(control_lock_path),
        "source_case_bindings": [
            {"case_id": cid, "cohort": roster_by_id[cid]["cohort"],
             "dataset": roster_by_id[cid]["dataset"],
             "source_case_sha256": roster_by_id[cid]["source_case_sha256"],
             "source_model_state_sha256": source_cases[cid]["source_model_state_sha256"],
             "pixel_sha256": source_cases[cid]["pixel_sha256"]}
            for cid in sorted(METHOD_IDS)
        ],
        "case_rows": case_rows,
        "metric_rows": metric_rows,
        "arm_comparison_rows": comparison_rows,
        "fit_movement_rows": fit_rows,
        "candidate_present_anchor_rows": candidate_rows,
        "reference_control_locked_anchor_rows": control_anchor_rows,
        "candidate_present_anchor_summary": candidate_summary,
        "reference_control_locked_anchor_summary": control_anchor_summary,
        "method_groups_old8_vs_additional8": groups,
        "V03_reference_control_groups": control_group,
        "metric_count": len(metric_rows),
        "candidate_row_count": len(candidate_rows),
        "candidate_row_scope": "method phase only; V03 scene-object candidate-input rows are excluded",
        "excluded_V03_candidate_context_row_count": len(candidate_input_by_id[CONTROL_ID]["candidates"]),
        "reference_control_locked_anchor_row_count": len(control_anchor_rows),
        "candidate_and_reference_control_anchor_row_count": len(candidate_rows) + len(control_anchor_rows),
        "fit_movement_row_count": len(fit_rows),
        "fit_movement_row_semantics": "movement diagnostic rows include actual fits and exact no-update placeholders",
        "actual_fit_count": actual_fit_count,
        "no_fit_placeholder_count": no_fit_placeholder_count,
        "actual_fit_optimizer_step_sum": sum(
            int(row.get("optimizer_steps", 0)) for row in fit_rows
            if row.get("status") in actual_fit_statuses),
        "official_independent_metric_max_abs_error": max(metric_errors) if metric_errors else None,
        "classification_tolerance_pp": NEUTRAL_EPS,
        "classification_is_descriptive_not_a_gate": True,
        "V03_control_roles_scored_exactly": list(CONTROL_ROLES),
        "candidate_missing_zero_convention": "only GT-known slots among the original four enter the common denominator; candidate-missing slots are zero-filled there, while GT-missing slots are excluded and remain unknown support, not coordinate failure",
        "query_prose_or_teacher_response_prose_included": False,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace-report", action="store_true",
                        help="replace this scorer's prior successful report after a structure correction")
    args = parser.parse_args()
    result = run(replace_report=args.replace_report)
    print(json.dumps({
        "status": result["status"],
        "case_rows": len(result["case_rows"]),
        "metric_rows": result["metric_count"],
        "candidate_rows": result["candidate_row_count"],
        "fit_movement_rows": result["fit_movement_row_count"],
        "official_independent_metric_max_abs_error": result["official_independent_metric_max_abs_error"],
        "report": str((BASE / "ROOT_POSTSEAL_REPORT.json").relative_to(ROOT)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
