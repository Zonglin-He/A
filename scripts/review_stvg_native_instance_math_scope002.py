"""Independent CPU arithmetic/readout reader for native-instance Student fits.

The default command waits for the sealed eight-case Student barrier, validates
all receipts and payload hashes before unpickling any result, and audits the
saved fit mathematics.  It does not load a model, decode media, or read labels.
``--contracts`` runs only synthetic CPU arithmetic checks and touches no scope
artifacts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / "artifacts/stvg_native_instance_spatial_opd_v1"
READBACK_PATH = BASE / "ROOT_STUDENT_MATH_READBACK.json"
COST_PATH = BASE / "ROOT_COST_READBACK.json"
OLD_BASE = ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1"
EXPECTED_IDS = tuple([f"hc2_{i:03d}" for i in range(4)] +
                     [f"vidstg_{i:03d}" for i in range(4)])
ALGORITHMS = ("direct", "iou", "reliability_weighted_iou")
CONFIG_SHA256 = "fffa324298cf86e00a6445c737be92050983466ecaf20ba036ace685f9a7d1d2"
TOL = {
    "mean_abs": 3e-5,
    "sample_abs": 3e-5,
    "reward_abs": 2e-4,
    "weight_abs": 2e-4,
    "loss_abs": 5e-4,
    "adam_parameter_abs": 1e-4,
    "adam_state_abs": 1e-4,
    "direct_loss_abs": 2e-5,
    "adam_moment_gradient_abs": 2e-5,
}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("expected a ROOT-relative POSIX path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("absolute or traversing path is forbidden")
    lowered = "/" + value.casefold().strip("/") + "/"
    if any(x in lowered for x in ("/annotations/", "/ground_truth/", "/gt_labels/")):
        raise ValueError("label-bearing path is outside this math reader")
    return value


def _verify_path(relative: Any, digest: Any, byte_count: Any = None) -> Path:
    rel = _safe_relative(relative)
    _need(isinstance(digest, str) and len(digest) == 64
          and all(c in "0123456789abcdef" for c in digest),
          f"malformed SHA256 for {rel}")
    path = ROOT / rel
    _need(path.is_file() and _sha_file(path) == digest,
          f"sealed bytes changed: {rel}")
    if byte_count is not None:
        _need(isinstance(byte_count, int) and not isinstance(byte_count, bool)
              and path.stat().st_size == byte_count,
              f"sealed byte count changed: {rel}")
    return path


def _exact(left: Any, right: Any, label: str) -> None:
    """Strict recursive value equality for tensors/arrays/mappings/sequences."""
    import numpy as np

    if torch.is_tensor(left) or torch.is_tensor(right):
        _need(torch.is_tensor(left) and torch.is_tensor(right), f"{label}: tensor schema differs")
        a, b = left.detach().cpu(), right.detach().cpu()
        _need(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b),
              f"{label}: tensors are not bitwise equal")
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        _need(isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
              and left.dtype == right.dtype and left.shape == right.shape
              and np.array_equal(left, right), f"{label}: arrays differ")
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        _need(isinstance(left, Mapping) and isinstance(right, Mapping)
              and set(left) == set(right), f"{label}: mapping schema differs")
        for key in left:
            _exact(left[key], right[key], f"{label}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        _need(type(left) is type(right) and len(left) == len(right),
              f"{label}: sequence schema differs")
        for i, (a, b) in enumerate(zip(left, right)):
            _exact(a, b, f"{label}[{i}]")
        return
    _need(left == right, f"{label}: values differ")


def _tree_close(left: Any, right: Any, tol: float, label: str) -> float:
    import numpy as np

    if torch.is_tensor(left) or torch.is_tensor(right):
        _need(torch.is_tensor(left) and torch.is_tensor(right), f"{label}: tensor schema differs")
        a, b = left.detach().cpu(), right.detach().cpu()
        _need(a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all(),
              f"{label}: tensor shape/nonfinite mismatch")
        error = float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
        _need(error <= tol, f"{label}: error {error:.9g} > {tol:.9g}")
        return error
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        _need(isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
              and left.shape == right.shape, f"{label}: array schema differs")
        a, b = torch.as_tensor(left), torch.as_tensor(right)
        return _tree_close(a, b, tol, label)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        _need(isinstance(left, Mapping) and isinstance(right, Mapping)
              and set(left) == set(right), f"{label}: mapping schema differs")
        return max((_tree_close(left[k], right[k], tol, f"{label}.{k}") for k in left), default=0.0)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        _need(type(left) is type(right) and len(left) == len(right),
              f"{label}: sequence schema differs")
        return max((_tree_close(a, b, tol, f"{label}[{i}]")
                    for i, (a, b) in enumerate(zip(left, right))), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        error = abs(float(left) - float(right))
        _need(math.isfinite(error) and error <= tol, f"{label}: scalar error {error:.9g} > {tol:.9g}")
        return error
    _need(left == right, f"{label}: values differ")
    return 0.0


def _opd_math():
    # This module is a pure CPU arithmetic reader; importing it does not load
    # models or read scope artifacts.  Its existing IoU/Adam primitives are
    # reused for the unweighted branch, while weighted rewards are reconstructed
    # below from raw per-sample rewards.
    from scripts import review_stvg_scene_object_id_spatial_math_v1 as math_reader
    return math_reader


def _read_runtime_pins(runtime: Mapping[str, Any]) -> dict[str, str]:
    pins, protected = runtime.get("pins"), runtime.get("protected")
    _need(isinstance(pins, Mapping) and isinstance(protected, Mapping),
          "student runtime pin maps are missing")
    for path in set(pins) & set(protected):
        _need(pins[path] == protected[path], f"runtime pin conflict at {path}")
    combined = {**pins, **protected}
    verified: dict[str, str] = {}
    for relative, digest in combined.items():
        path = _verify_path(relative, digest)
        verified[str(relative)] = str(digest)
        _need(path.is_file(), f"runtime pin is not a file: {relative}")
    return verified


def _validate_phase_lineage_pins(student_pins: Mapping[str, str],
                                 teacher_pins: Mapping[str, str]) -> None:
    """Check phase handoffs according to continue_stvg_native_instance_storage002.

    Tracking's actual-exit receipt is an input to the *teacher* runtime.  The
    student runtime pins the tracking barrier, but that exit receipt is already
    transitively bound through the teacher runtime and barrier.
    """
    base = BASE.relative_to(ROOT).as_posix()
    student_required = (
        f"{base}/TRACKING_GLOBAL_BARRIER.json",
        f"{base}/TEACHER_GLOBAL_BARRIER.json",
        f"{base}/TEACHER_ACTUAL_PHASE_EXIT.json",
        f"{base}/REFERENCE_LOCK.json",
        f"{base}/DIRECT_TEACHER_READOUT.json",
    )
    teacher_required = (
        f"{base}/TRACKING_GLOBAL_BARRIER.json",
        f"{base}/TRACKING_ACTUAL_PHASE_EXIT.json",
    )
    for relative in student_required:
        _need(relative in student_pins,
              f"student runtime does not pin sealed input lineage: {relative}")
    for relative in teacher_required:
        _need(relative in teacher_pins,
              f"teacher runtime does not pin tracking handoff lineage: {relative}")


def _preflight() -> dict[str, Any]:
    """Verify the complete output inventory and all hashes before torch.load."""
    if torch.cuda.is_initialized():
        raise RuntimeError("math reader must remain CPU-only")

    runtime_path = BASE / "STUDENT_RUNTIME.json"
    barrier_path = BASE / "STUDENT_GLOBAL_BARRIER.json"
    reference_path = BASE / "REFERENCE_LOCK.json"
    roster_path = BASE / "ROSTER_LOCK.json"
    storage_lock_path = BASE / "SERIAL_STORAGE002_LOCK.json"
    _need(runtime_path.is_file() and barrier_path.is_file()
          and reference_path.is_file() and roster_path.is_file()
          and storage_lock_path.is_file(),
          "student runtime/barrier, reference lock, roster, or storage lock is not sealed yet")
    runtime = _read_json(runtime_path)
    barrier = _read_json(barrier_path)
    reference = _read_json(reference_path)
    roster = _read_json(roster_path)
    storage_lock = _read_json(storage_lock_path)
    runtime_sha, reference_sha = _sha_file(runtime_path), _sha_file(reference_path)
    storage_lock_sha = _sha_file(storage_lock_path)
    _need(runtime.get("status") == "locked" and runtime.get("GT_read") is False
          and runtime.get("scope") == "native_instance_student" and runtime.get("count") == 8,
          "student runtime is not the exact locked eight-case no-GT scope")
    _need(barrier.get("status") == "complete" and barrier.get("count") == 8
          and barrier.get("GT_read") is False and barrier.get("runtime_sha256") == runtime_sha,
          "student barrier is incomplete, label-exposed, or stale against runtime")
    _need(reference.get("schema") == "stvg-native-instance-reference-lock-v1"
          and reference.get("status") == "locked" and reference.get("count") == 8
          and reference.get("GT_read") is False,
          "native-instance reference lock is not complete and GT-free")
    _need(roster.get("status") == "locked" and roster.get("count") == 8
          and roster.get("GT_read") is False,
          "native-instance roster is not a locked eight-case no-GT record")
    _need(storage_lock.get("status") == "locked" and storage_lock.get("count") == 8
          and storage_lock.get("GT_read") is False
          and storage_lock.get("no_automatic_GT_scoring") is True
          and storage_lock.get("no_automatic_expansion") is True,
          "SERIAL_STORAGE002_LOCK is not the sealed bounded no-GT execution lock")
    roster_rows = roster.get("records")
    ref_rows = reference.get("records")
    files = barrier.get("files")
    _need(isinstance(roster_rows, list) and len(roster_rows) == 8
          and isinstance(ref_rows, list) and len(ref_rows) == 8
          and isinstance(files, list) and len(files) == 8,
          "roster/reference/barrier must each contain exactly eight rows")
    case_ids = [row.get("case_id") for row in roster_rows]
    _need(case_ids == list(EXPECTED_IDS), "roster does not match the fixed balanced eight-case order")
    _need([row.get("case_id") for row in ref_rows] == case_ids,
          "reference lock order differs from locked roster")
    _need([row.get("case_id") for row in files] == case_ids,
          "student barrier order differs from locked roster")

    runtime_pins = _read_runtime_pins(runtime)
    rel_reference = reference_path.relative_to(ROOT).as_posix()
    rel_roster = roster_path.relative_to(ROOT).as_posix()
    _need(runtime_pins.get(rel_reference) == reference_sha,
          "student runtime does not pin the exact reference lock")
    _need(runtime_pins.get(rel_roster) == _sha_file(roster_path),
          "student runtime does not pin the exact roster")
    # The launcher barrier schema is {status,count,files,runtime_sha256,
    # GT_read,time}; it has no reference_lock_sha256 field.  Bind the lock via
    # the verified student runtime and each result payload instead.
    stage_pins = storage_lock.get("stage_pins", {}).get("student")
    storage_protected = storage_lock.get("protected")
    _need(isinstance(stage_pins, Mapping) and isinstance(storage_protected, Mapping),
          "SERIAL_STORAGE002_LOCK student pins/protected maps are missing")
    for relative, digest in stage_pins.items():
        _verify_path(relative, digest)
        _need(runtime_pins.get(relative) == digest,
              f"student runtime omits or changes SERIAL_STORAGE002_LOCK pin: {relative}")
    _need(runtime.get("protected") == storage_protected,
          "student runtime protected map differs from SERIAL_STORAGE002_LOCK")
    # Revalidate the no-GT lineage already sealed before this student phase.
    # Input JSON payloads are hash-checked only; teacher receipt text is never
    # opened by this reader.
    tracking_runtime_path = BASE / "TRACKING_RUNTIME.json"
    teacher_runtime_path = BASE / "TEACHER_RUNTIME.json"
    tracking_barrier_path = BASE / "TRACKING_GLOBAL_BARRIER.json"
    teacher_barrier_path = BASE / "TEACHER_GLOBAL_BARRIER.json"
    _need(reference.get("roster_sha256") == _sha_file(roster_path)
          and reference.get("tracking_barrier_sha256") == _sha_file(tracking_barrier_path)
          and reference.get("teacher_barrier_sha256") == _sha_file(teacher_barrier_path)
          and reference.get("tracking_runtime_sha256") == _sha_file(tracking_runtime_path)
          and reference.get("teacher_runtime_sha256") == _sha_file(teacher_runtime_path),
          "reference lock does not bind the exact roster/tracking/teacher lineage")
    tracking_runtime = _read_json(tracking_runtime_path)
    teacher_runtime = _read_json(teacher_runtime_path)
    _need(tracking_runtime.get("status") == "locked" and tracking_runtime.get("GT_read") is False
          and teacher_runtime.get("status") == "locked" and teacher_runtime.get("GT_read") is False,
          "tracking or teacher runtime is not a locked no-GT input")
    tracking_barrier = _read_json(tracking_barrier_path)
    teacher_barrier = _read_json(teacher_barrier_path)
    for stage_name, stage_barrier, stage_runtime, stage_runtime_path in (
            ("tracking", tracking_barrier, tracking_runtime, tracking_runtime_path),
            ("teacher", teacher_barrier, teacher_runtime, teacher_runtime_path)):
        _need(stage_barrier.get("status") == "complete" and stage_barrier.get("count") == 8
              and stage_barrier.get("GT_read") is False
              and stage_barrier.get("runtime_sha256") == _sha_file(stage_runtime_path),
              f"sealed {stage_name} barrier/runtime binding is invalid")
        stage_files = stage_barrier.get("files")
        _need(isinstance(stage_files, list) and len(stage_files) == 8
              and [row.get("case_id") for row in stage_files] == list(EXPECTED_IDS),
              f"sealed {stage_name} barrier does not cover the exact eight cases")
        for case_id, row in zip(EXPECTED_IDS, stage_files):
            expected_relative = (f"{BASE.relative_to(ROOT).as_posix()}/TRACKING/{case_id}/ACCEPTED.json"
                                 if stage_name == "tracking"
                                 else f"{BASE.relative_to(ROOT).as_posix()}/teacher/{case_id}.receipt.json")
            _need(row.get("path") == expected_relative,
                  f"sealed {stage_name} barrier path differs for {case_id}")
            _verify_path(row.get("path"), row.get("sha256"))
        if stage_name == "tracking":
            # Tracking predates the later two execution-runtime schemas and
            # records its stage explicitly rather than scope/count.
            _need(stage_runtime.get("stage") == "tracking"
                  and stage_runtime.get("cases") == 8,
                  "sealed tracking runtime stage field differs")
        else:
            _need(stage_runtime.get("scope") == "native_instance_teacher"
                  and stage_runtime.get("count") == 8,
                  "sealed teacher runtime scope/count differs")
    source_bindings = reference.get("source_bindings")
    _need(isinstance(source_bindings, Mapping), "reference source bindings are missing")
    expected_source_binding_names = {
        "scripts/prepare_stvg_native_instance_references_v1.py",
        "scripts/stvg_native_instance_common_v1.py",
        "scripts/stvg_r1_native_teacher_v1.py",
    }
    _need(set(source_bindings) == expected_source_binding_names
          and reference.get("created_by_sha256") == source_bindings.get(
              "scripts/prepare_stvg_native_instance_references_v1.py"),
          "reference lock preparation source bindings are malformed")
    for relative, digest in source_bindings.items():
        _verify_path(relative, digest)
    tracking_runtime_pins = _read_runtime_pins(tracking_runtime)
    teacher_runtime_pins = _read_runtime_pins(teacher_runtime)
    _validate_phase_lineage_pins(runtime_pins, teacher_runtime_pins)
    for relative, digest in storage_lock.get("stage_pins", {}).get("teacher", {}).items():
        _need(teacher_runtime_pins.get(relative) == digest,
              f"teacher runtime omits or changes SERIAL_STORAGE002_LOCK pin: {relative}")

    roster_by_id = {row["case_id"]: row for row in roster_rows}
    reference_by_id = {row["case_id"]: row for row in ref_rows}
    case_paths: dict[str, Path] = {}
    receipt_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, dict[str, Any]] = {}
    for cid, sealed_row in zip(case_ids, files):
        _need(set(sealed_row) == {"case_id", "path", "sha256"}
              and sealed_row.get("case_id") == cid,
              f"malformed student barrier entry for {cid}")
        receipt_rel = _safe_relative(sealed_row.get("path"))
        expected_receipt_rel = f"{BASE.relative_to(ROOT).as_posix()}/student/{cid}.receipt.json"
        _need(receipt_rel == expected_receipt_rel,
              f"student barrier points at an unexpected receipt path for {cid}")
        receipt_path = _verify_path(receipt_rel, sealed_row.get("sha256"))
        receipt = _read_json(receipt_path)
        _need(receipt.get("status") == "sealed" and receipt.get("GT_read") is False,
              f"student case receipt is not sealed and GT-free for {cid}")
        pt_rel = _safe_relative(receipt.get("path"))
        expected_pt_rel = f"{BASE.relative_to(ROOT).as_posix()}/student/{cid}.pt"
        _need(pt_rel == expected_pt_rel, f"student receipt points at unexpected result for {cid}")
        pt_path = _verify_path(pt_rel, receipt.get("sha256"), receipt.get("bytes"))
        _need(runtime_pins.get(pt_rel) in (None, receipt.get("sha256")),
              f"student runtime has a conflicting result pin for {cid}")
        case_paths[cid] = pt_path
        receipt_paths[cid] = receipt_path
        artifact_hashes[cid] = {
            "student_artifact_sha256": receipt["sha256"],
            "student_receipt_sha256": _sha_file(receipt_path),
            "student_artifact_bytes": receipt["bytes"],
        }
        ref_row = reference_by_id[cid]
        receipt_row = next(row for row in teacher_barrier["files"] if row["case_id"] == cid)
        _need(ref_row.get("teacher_receipt_sha256") == receipt_row.get("sha256"),
              f"reference lock does not match the sealed teacher receipt for {cid}")
        teacher_record_rel = f"{BASE.relative_to(ROOT).as_posix()}/teacher/{cid}.json"
        teacher_receipt_rel = f"{BASE.relative_to(ROOT).as_posix()}/teacher/{cid}.receipt.json"
        _verify_path(teacher_record_rel, ref_row.get("teacher_record_sha256"))
        _verify_path(teacher_receipt_rel, ref_row.get("teacher_receipt_sha256"))

    # Confirm the temporal baseline bytes are the exact old F/T inputs named by
    # the current runtime.  No old report or old score file is touched.
    old_barrier_rel = f"{OLD_BASE.relative_to(ROOT).as_posix()}/STUDENT_GLOBAL_BARRIER.json"
    _need(old_barrier_rel in runtime_pins, "student runtime omits the exact old temporal barrier pin")
    old_barrier = _read_json(ROOT / old_barrier_rel)
    _need(old_barrier.get("status") == "complete" and old_barrier.get("count") == 8
          and old_barrier.get("GT_read") is False,
          "old temporal baseline barrier is not complete and GT-free")
    old_rows = old_barrier.get("files")
    _need(isinstance(old_rows, list) and len(old_rows) == 8,
          "old temporal baseline barrier must bind eight case files")
    old_by_id = {}
    for row in old_rows:
        rel = _safe_relative(row.get("path"))
        if rel.startswith(f"{OLD_BASE.relative_to(ROOT).as_posix()}/student/") and rel.endswith(".pt"):
            cid = Path(rel).stem
            _need(cid not in old_by_id, f"duplicate old temporal case {cid}")
            old_path = _verify_path(rel, row.get("sha256"), row.get("bytes"))
            old_by_id[cid] = old_path
    _need(set(old_by_id) == set(case_ids), "old temporal barrier does not cover exact eight cases")
    for cid in case_ids:
        old_rel = old_by_id[cid].relative_to(ROOT).as_posix()
        _need(runtime_pins.get(old_rel) == _sha_file(old_by_id[cid]),
              f"current runtime does not pin exact old temporal input for {cid}")

    expected_fits = 0
    for cid in case_ids:
        rr = roster_by_id[cid]
        ref = reference_by_id[cid]
        _need(rr.get("dataset") == ref.get("dataset") in {"hc2", "vidstg"}
              and rr.get("source_case_sha256") == ref.get("source_case_sha256")
              and roster.get("GT_read") is False and ref.get("GT_read") is False,
              f"roster/reference source metadata differ for {cid}")
        _need(isinstance(rr.get("source_case_path"), str)
              and _verify_path(rr["source_case_path"], rr["source_case_sha256"]).is_file(),
              f"locked source case file/hash is missing for {cid}")
        anchors = ref.get("anchors")
        _need(isinstance(anchors, list) and len(anchors) <= 4,
              f"reference anchors are malformed for {cid}")
        _need(ref.get("no_update") is (len(anchors) == 0),
              f"reference no-update flag differs from anchor emptiness for {cid}")
        seen_pos: set[int] = set()
        for anchor in anchors:
            _need(isinstance(anchor, Mapping), f"malformed reference anchor for {cid}")
            pos = anchor.get("position")
            fid = anchor.get("physical_frame_id")
            _need(isinstance(pos, int) and not isinstance(pos, bool) and pos >= 0
                  and isinstance(fid, int) and not isinstance(fid, bool)
                  and pos not in seen_pos, f"invalid/duplicate sparse source position for {cid}")
            seen_pos.add(pos)
            box = torch.as_tensor(anchor.get("box_cxcywh"), dtype=torch.float64)
            _need(box.shape == (4,) and torch.isfinite(box).all(),
                  f"anchor box must be finite cxcywh for {cid}")
            cx, cy, width, height = map(float, box.tolist())
            _need(width > 0 and height > 0
                  and 0 <= cx - width / 2 < cx + width / 2 <= 1
                  and 0 <= cy - height / 2 < cy + height / 2 <= 1,
                  f"anchor box is outside normalized domain for {cid}")
            weight = anchor.get("weight")
            _need(isinstance(weight, (int, float)) and not isinstance(weight, bool)
                  and math.isfinite(float(weight)) and 0 <= float(weight) <= 1,
                  f"anchor reliability must be finite in [0,1] for {cid}")
        if anchors:
            expected_fits += 2 + int(any(float(a["weight"]) > 0 for a in anchors))
    # Some locked runtimes intentionally omit a planned-fit field.  The
    # authoritative count is derived from the sealed reference rows here and
    # checked against the complete case inventory after reading it.
    declared_expected_fits = runtime.get("expected_fit_count")
    if declared_expected_fits is not None:
        _need(declared_expected_fits == expected_fits,
              "student runtime expected fit count differs from sealed references")

    return {
        "runtime": runtime,
        "barrier": barrier,
        "reference": reference,
        "roster": roster,
        "runtime_sha256": runtime_sha,
        "barrier_sha256": _sha_file(barrier_path),
        "reference_sha256": reference_sha,
        "roster_sha256": _sha_file(roster_path),
        "storage_lock_sha256": storage_lock_sha,
        "runtime_pins": runtime_pins,
        "case_paths": case_paths,
        "receipt_paths": receipt_paths,
        "artifact_hashes": artifact_hashes,
        "roster_by_id": roster_by_id,
        "reference_by_id": reference_by_id,
        "old_case_paths": old_by_id,
        "expected_fit_count": expected_fits,
        "tracking_runtime_pins": tracking_runtime_pins,
        "teacher_runtime_pins": teacher_runtime_pins,
    }


def _anchors_for_audit(reference: Mapping[str, Any], frame_ids: Sequence[int]) -> tuple[list[int], torch.Tensor, list[float]]:
    positions: list[int] = []
    boxes: list[torch.Tensor] = []
    weights: list[float] = []
    for row in reference.get("anchors", []):
        position, physical = row.get("position"), row.get("physical_frame_id")
        _need(isinstance(position, int) and not isinstance(position, bool)
              and 0 <= position < len(frame_ids), "reference position is outside saved source grid")
        _need(isinstance(physical, int) and not isinstance(physical, bool)
              and frame_ids[position] == physical, "reference physical ID differs from source-grid position")
        box64 = torch.as_tensor(row.get("box_cxcywh"), dtype=torch.float64, device="cpu")
        _need(box64.shape == (4,) and torch.isfinite(box64).all(), "anchor cxcywh is malformed")
        cx, cy, w, h = map(float, box64.tolist())
        _need(w > 0 and h > 0 and 0 <= cx - w / 2 < cx + w / 2 <= 1
              and 0 <= cy - h / 2 < cy + h / 2 <= 1,
              "anchor geometry is outside normalized image bounds")
        positions.append(position)
        boxes.append(box64.to(torch.float32))
        weights.append(float(row["weight"]))
    _need(positions == sorted(positions) and len(set(positions)) == len(positions),
          "anchors must be stored in unique increasing source-grid order")
    evidence = torch.stack(boxes) if boxes else torch.empty((0, 4), dtype=torch.float32)
    return positions, evidence, weights


def _cpu_state_numel(state: Any, label: str) -> int:
    _need(isinstance(state, Mapping) and bool(state), f"{label} must be a nonempty parameter mapping")
    total = 0
    for name, value in state.items():
        _need(isinstance(name, str) and torch.is_tensor(value)
              and value.device.type == "cpu" and torch.isfinite(value).all(),
              f"{label}.{name} must be finite CPU tensor")
        total += value.numel()
    return total


def _direct_loss(boxes: torch.Tensor, positions: Sequence[int], evidence: torch.Tensor) -> torch.Tensor:
    """Recompute original Direct SpatialLoss scalar on fp32 pre-step boxes."""
    _need(torch.is_tensor(boxes) and boxes.ndim == 2 and boxes.shape[-1] == 4,
          "Direct pre-step boxes must be [T,4]")
    if not positions:
        return boxes.sum() * 0
    pred = boxes.detach().cpu().float()[list(positions)]
    target = evidence.detach().cpu().to(dtype=pred.dtype)

    def xyxy(value: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = value.unbind(-1)
        return torch.stack((cx - .5*w, cy - .5*h, cx + .5*w, cy + .5*h), -1)

    p, t = xyxy(pred), xyxy(target)
    lt = torch.maximum(p[..., :2], t[..., :2])
    rb = torch.minimum(p[..., 2:], t[..., 2:])
    intersection = (rb - lt).clamp(min=0).prod(-1)
    pa = (p[..., 2:] - p[..., :2]).clamp(min=0).prod(-1)
    ta = (t[..., 2:] - t[..., :2]).clamp(min=0).prod(-1)
    union = (pa + ta - intersection).clamp(min=1e-7)
    enc = (torch.maximum(p[..., 2:], t[..., 2:])
           - torch.minimum(p[..., :2], t[..., :2])).clamp(min=0).prod(-1).clamp(min=1e-7)
    giou = intersection / union - (enc - union) / enc
    per_anchor = 5 * (pred - target).abs().sum(-1) + 2 * (1 - giou)
    return per_anchor.sum() / 4


def _optimizer_inferred_gradients(entry: Mapping[str, Any], config: Mapping[str, Any], step: int) -> tuple[list[torch.Tensor], float]:
    """Recover Direct's gradient from Adam moments; this is not autograd proof."""
    before = entry.get("optimizer_before")
    after = entry.get("optimizer_state")
    _need(isinstance(before, Mapping) and isinstance(after, Mapping),
          f"Direct step {step} omitted Adam before/after states")
    names = list(entry["state"])
    group_before = before.get("param_groups", [{}])[0]
    group_after = after.get("param_groups", [{}])[0]
    ids_before, ids_after = group_before.get("params", []), group_after.get("params", [])
    _need(len(ids_before) == len(ids_after) == len(names),
          f"Direct step {step} Adam parameter order differs from saved 1,792 state")
    _need(abs(float(group_before.get("lr", -1)) - float(config["lr"])) < 1e-12
          and abs(float(group_after.get("lr", -1)) - float(config["lr"])) < 1e-12,
          f"Direct step {step} Adam lr differs from locked config")
    beta1, beta2 = group_after.get("betas", (None, None))
    _need(float(beta1) == .9 and float(beta2) == .999
          and float(group_after.get("eps", -1)) == 1e-8
          and float(group_after.get("weight_decay", -1)) == 0.0,
          f"Direct step {step} Adam hyperparameters differ from original optimizer")
    state_before, state_after = before.get("state", {}), after.get("state", {})
    gradients: list[torch.Tensor] = []
    max_moment_error = 0.0
    for name, pid0, pid1 in zip(names, ids_before, ids_after):
        p0 = state_before.get(pid0, state_before.get(str(pid0), {}))
        p1 = state_after.get(pid1, state_after.get(str(pid1), {}))
        _need(isinstance(p1, Mapping) and "exp_avg" in p1 and "exp_avg_sq" in p1,
              f"Direct step {step} Adam moments are incomplete")
        m1 = torch.as_tensor(p1["exp_avg"]).detach().cpu()
        v1 = torch.as_tensor(p1["exp_avg_sq"]).detach().cpu()
        param = entry["state"][name].detach().cpu()
        _need(m1.shape == v1.shape == param.shape and torch.isfinite(m1).all()
              and torch.isfinite(v1).all(), f"Direct step {step} moments malformed for {name}")
        if p0:
            m0 = torch.as_tensor(p0["exp_avg"]).detach().cpu()
            v0 = torch.as_tensor(p0["exp_avg_sq"]).detach().cpu()
        else:
            m0, v0 = torch.zeros_like(m1), torch.zeros_like(v1)
        gradient = (m1 - float(beta1) * m0) / (1.0 - float(beta1))
        expected_v = float(beta2) * v0 + (1.0 - float(beta2)) * gradient.square()
        moment_error = float((expected_v.double() - v1.double()).abs().max()) if v1.numel() else 0.0
        _need(moment_error <= 3e-5, f"Direct step {step} Adam second moment is inconsistent")
        max_moment_error = max(max_moment_error, moment_error)
        gradients.append(gradient)
    _need(sum(g.numel() for g in gradients) == 1792,
          f"Direct step {step} reconstructed gradient is not joint 1,792-dimensional")
    return gradients, max_moment_error


def _audit_direct_fit(fit: Mapping[str, Any], readout: Mapping[str, Any],
                      positions: Sequence[int], evidence: torch.Tensor,
                      expected_initial: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    _need(fit.get("algorithm") == "direct" and fit.get("arm") == "Direct"
          and fit.get("scope") == "joint" and fit.get("active_parameters") == 1792
          and fit.get("declared_steps") == fit.get("selected_step") == 5,
          "Direct fit is not the complete five-step joint fit")
    path = fit.get("path")
    _need(isinstance(path, list) and len(path) == 5,
          "Direct fit omitted one or more of its five saved steps")
    _exact(fit.get("config"), dict(config), "Direct.config")
    _exact(fit.get("initial_state"), expected_initial, "Direct.initial_state")
    _need(fit.get("observation_positions") == list(positions),
          "Direct fit positions differ from the locked sparse reference")
    _need(_cpu_state_numel(fit.get("initial_state"), "Direct.initial_state") == 1792,
          "Direct initial state does not contain exactly 1,792 trainable coordinates")
    max_loss_error = 0.0
    max_adam_error = 0.0
    max_moment_error = 0.0
    optimizer_steps = 0
    if not positions:
        for index, entry in enumerate(path):
            _need(entry.get("step") == index and entry.get("skipped") is True,
                  "empty Direct reference must retain five explicit skipped steps")
            _exact(entry.get("state"), fit.get("initial_state"), f"Direct.empty.state{index}")
            _need(not entry.get("optimizer_state", {}).get("state"),
                  "empty Direct reference must not mutate Adam")
        _exact(fit.get("final_state"), fit.get("initial_state"), "Direct.empty.final_state")
    else:
        for index, entry in enumerate(path):
            _need(entry.get("step") == index and entry.get("skipped") is False,
                  f"Direct step {index} is missing an optimizer update")
            expected_loss = _direct_loss(entry.get("boxes"), positions, evidence)
            loss_error = abs(float(entry.get("loss", float("nan"))) - float(expected_loss))
            _need(math.isfinite(loss_error) and loss_error <= TOL["direct_loss_abs"],
                  f"Direct step {index} loss differs from independent SpatialLoss: {loss_error:.9g}")
            max_loss_error = max(max_loss_error, loss_error)
            gradients, moment_error = _optimizer_inferred_gradients(entry, config, index)
            max_moment_error = max(max_moment_error, moment_error)

            params = [torch.nn.Parameter(entry["state"][name].detach().cpu().clone())
                      for name in entry["state"]]
            optimizer = torch.optim.Adam(params, lr=float(config["lr"]), betas=(.9, .999),
                                         eps=1e-8, weight_decay=0.)
            optimizer.load_state_dict(copy.deepcopy(entry["optimizer_before"]))
            _need(len(params) == len(gradients), f"Direct step {index} Adam grouping differs")
            for parameter, gradient in zip(params, gradients):
                parameter.grad = gradient.to(parameter)
            optimizer.step()
            expected_after = entry.get("update_state")
            _need(isinstance(expected_after, Mapping) and set(expected_after) == set(entry["state"]),
                  f"Direct step {index} omitted complete post-update state")
            for name, parameter in zip(entry["state"], params):
                error = float((parameter.detach().cpu().double()
                               - expected_after[name].detach().cpu().double()).abs().max())
                _need(error <= TOL["adam_parameter_abs"],
                      f"Direct step {index} Adam parameter mismatch: {error:.9g}")
                max_adam_error = max(max_adam_error, error)
            helper = _opd_math()
            errors: list[float] = []
            helper._compare_tree(optimizer.state_dict(), entry.get("optimizer_state"),
                                 tolerance=TOL["adam_state_abs"], errors=errors,
                                 name=f"Direct.step{index}.Adam")
            max_adam_error = max(max_adam_error, max(errors, default=0.0))
            optimizer_steps += 1

            if index < 4:
                next_entry = path[index + 1]
                _exact(entry.get("update_state"), next_entry.get("state"),
                       f"Direct.step{index}.state_chain")
                _exact(entry.get("optimizer_state"), next_entry.get("optimizer_before"),
                       f"Direct.step{index}.optimizer_chain")
                actual_delta = entry.get("actual_delta")
                _need(torch.is_tensor(actual_delta), f"Direct step {index} omitted decoder delta")
                _tree_close(actual_delta, next_entry.get("boxes") - entry["boxes"], 1e-7,
                            f"Direct.step{index}.decoder_delta")
        _exact(fit.get("final_state"), path[-1].get("update_state"), "Direct.final_state")
        _need(int(fit.get("optimizer_steps", -1)) == optimizer_steps,
              "Direct optimizer-step count differs from five-step path")
    _need(int(fit.get("optimizer_steps", -1)) == optimizer_steps,
          "Direct optimizer-step count is inconsistent")
    _need(torch.is_tensor(fit.get("final_cached_decoder_boxes")),
          "Direct final cached decoder boxes missing")
    _exact(readout.get("boxes"), fit["final_cached_decoder_boxes"],
           "Direct full-model readout vs cached decoder")
    return {
        "steps": 5,
        "optimizer_steps": optimizer_steps,
        "direct_loss_independently_recomputed": bool(positions),
        "adam_moment_and_parameter_update_reconstructed": bool(positions),
        "decoder_autograd_gradient_independently_recomputed": False,
        "limitation": "Direct gradient is inferred from saved Adam moments; saved fit has no raw autograd gradient or decoder features.",
        "max_direct_loss_abs_error": max_loss_error,
        "max_direct_adam_parameter_or_state_error": max_adam_error,
        "max_direct_adam_second_moment_consistency_error": max_moment_error,
        "saved_full_model_readout_equals_cached_decoder": True,
        "model_replayed_by_reader": False,
    }


def _audit_opd_fit(fit: Mapping[str, Any], positions: Sequence[int], evidence: torch.Tensor,
                   weights: Sequence[float], frame_ids: Sequence[int], key: str,
                   config: Mapping[str, Any], expected_initial: Mapping[str, Any],
                   algorithm: str) -> dict[str, Any]:
    math_reader = _opd_math()
    active_indices = list(range(len(positions)))
    if algorithm == "reliability_weighted_iou":
        active_indices = [i for i, weight in enumerate(weights) if weight > 0]
    active_positions = [positions[i] for i in active_indices]
    active_evidence = evidence[active_indices] if active_indices else evidence.new_empty((0, 4))
    active_weights = torch.tensor([weights[i] for i in active_indices], dtype=torch.float32)
    _need(fit.get("algorithm") == algorithm and fit.get("scope") == "joint"
          and fit.get("active_parameters") == 1792
          and fit.get("declared_steps") == fit.get("selected_step") == 5,
          f"{algorithm} fit is not a complete five-step joint fit")
    _exact(fit.get("config"), dict(config), f"{algorithm}.config")
    _exact(fit.get("initial_state"), expected_initial, f"{algorithm}.initial_state")
    _need(_cpu_state_numel(fit.get("initial_state"), f"{algorithm}.initial_state") == 1792,
          f"{algorithm} initial state is not exactly 1,792 coordinates")
    _need(fit.get("observation_positions") == active_positions,
          f"{algorithm} active observation positions differ from the locked references")
    _need(fit.get("seed_key") == key and fit.get("reference_role") == "teacher"
          and isinstance(fit.get("reference_sha256"), str),
          f"{algorithm} seed/reference binding is malformed")
    expected_policy = ("original antithetic on-policy Gaussian IoU OPD"
                       if algorithm == "reliability_weighted_iou"
                       else "original antithetic on-policy Gaussian OPD")
    _need(fit.get("policy") == expected_policy,
          f"{algorithm} did not use the original fixed Gaussian OPD policy")
    if algorithm == "reliability_weighted_iou":
        _need(fit.get("reward_objective") == "r_t_times_original_iou",
              "weighted arm must multiply each raw IoU reward by its own reliability")
    else:
        _need(fit.get("reward_objective") == "iou",
              f"{algorithm} must retain the original unweighted IoU objective")
    path, rounds = fit.get("path"), fit.get("rounds")
    _need(isinstance(path, list) and len(path) == 5
          and isinstance(rounds, list) and len(rounds) == 5,
          f"{algorithm} fit must retain five path and round records")
    maxima = {name: 0.0 for name in ("mean_abs", "sample_abs", "reward_abs", "weight_abs",
                                     "loss_abs", "adam_parameter_abs", "adam_state_abs")}
    optimizer_steps = 0
    for index, (entry, round_row) in enumerate(zip(path, rounds)):
        _need(entry.get("step") == round_row.get("step") == index,
              f"{algorithm} step order is malformed")
        _need(torch.is_tensor(entry.get("boxes"))
              and entry["boxes"].shape == (len(frame_ids), 4)
              and torch.isfinite(entry["boxes"]).all(), f"{algorithm} pre-step boxes missing")
        policy = entry.get("rollout")
        _need(isinstance(policy, Mapping), f"{algorithm} rollout evidence missing")
        mean = torch.logit(entry["boxes"].detach().cpu().float()[active_positions])
        maxima["mean_abs"] = max(maxima["mean_abs"], math_reader._close(
            policy["mean"], mean, TOL["mean_abs"], f"{algorithm}.step{index}.mean"))
        seed, noise = math_reader._cpu_noise(key, index, len(active_positions))
        _need(policy.get("seed") == seed, f"{algorithm} step {index} seed mismatch")
        expected_samples = policy["mean"].detach().cpu().float()[:, None, :] + float(config["sigma"]) * noise
        maxima["sample_abs"] = max(maxima["sample_abs"], math_reader._close(
            policy["samples"], expected_samples, TOL["sample_abs"],
            f"{algorithm}.step{index}.samples"))
        expected_raw = math_reader._reward(policy["samples"].detach().cpu().float().sigmoid(),
                                           active_evidence[:, None, :], "iou")
        raw_key = "raw_rewards" if algorithm == "reliability_weighted_iou" else "rewards"
        expected_used = expected_raw
        if algorithm == "reliability_weighted_iou":
            expected_used = expected_raw * active_weights[:, None]
            _need(policy.get("anchor_reliability") is not None,
                  "weighted rollout omitted per-anchor reliability")
            maxima["weight_abs"] = max(maxima["weight_abs"], math_reader._close(
                policy["anchor_reliability"], active_weights, TOL["weight_abs"],
                f"weighted.step{index}.reliability"))
            _need(fit.get("reliability_normalized_across_anchors") is False,
                  "weighted reliability must not be normalized across anchors")
            _need(fit.get("reward_objective") == "r_t_times_original_iou",
                  "weighted objective must be raw IoU multiplied by r_t")
        else:
            _need(fit.get("reliability_applied") is False,
                  "unweighted Direct/IoU must not apply anchor reliability")
        maxima["reward_abs"] = max(maxima["reward_abs"], math_reader._close(
            policy.get(raw_key), expected_raw, TOL["reward_abs"],
            f"{algorithm}.step{index}.raw_iou_reward"))
        maxima["reward_abs"] = max(maxima["reward_abs"], math_reader._close(
            policy["used_rewards"], expected_used, TOL["reward_abs"],
            f"{algorithm}.step{index}.used_reward"))
        maxima["reward_abs"] = max(maxima["reward_abs"], math_reader._close(
            policy["rewards"], expected_raw, TOL["reward_abs"],
            f"{algorithm}.step{index}.reported_raw_reward"))
        expected_weights = torch.softmax(expected_used / float(config["tau"]), dim=-1)
        maxima["weight_abs"] = max(maxima["weight_abs"], math_reader._close(
            policy["weights"], expected_weights, TOL["weight_abs"],
            f"{algorithm}.step{index}.weights"))
        informative = bool((expected_used != expected_used[:, :1]).any())
        _need(policy.get("informative") is informative,
              f"{algorithm} step {index} information flag differs from per-row rewards")
        _need(policy.get("permutations") is None,
              f"{algorithm} fixed seed path must not permute samples")
        expected_loss = math_reader._likelihood(policy["mean"].detach().cpu().float(),
            policy["mean"].detach().cpu().float(), policy["samples"].detach().cpu().float(),
            expected_weights, float(config["sigma"]))
        loss_error = abs(float(entry.get("loss", float("nan"))) - float(expected_loss))
        _need(math.isfinite(loss_error) and loss_error <= TOL["loss_abs"],
              f"{algorithm} step {index} Gaussian likelihood loss error {loss_error:.9g}")
        maxima["loss_abs"] = max(maxima["loss_abs"], loss_error)
        _need(abs(float(round_row.get("loss")) - float(entry["loss"])) <= 1e-7,
              f"{algorithm} path/round loss mismatch")
        if not informative:
            _need(entry.get("update") is None and round_row.get("updated") is False
                  and round_row.get("no_information") is True,
                  f"{algorithm} flat-reward step must be a true no-update")
            _exact(entry["state"], entry.get("update_state", entry["state"]),
                   f"{algorithm}.flat{index}.state")
            _exact(entry.get("optimizer_before"), entry.get("optimizer_state"),
                   f"{algorithm}.flat{index}.Adam")
        else:
            optimizer_steps += 1
            update = entry.get("update")
            _need(isinstance(update, Mapping) and torch.is_tensor(update.get("gradient"))
                  and torch.is_tensor(update.get("raw")),
                  f"{algorithm} step {index} omitted saved 1,792 gradient/Adam update")
            _need(update["gradient"].numel() == update["raw"].numel() == 1792,
                  f"{algorithm} step {index} gradient or raw update is not 1,792-D")
            maxima["adam_parameter_abs"], maxima["adam_state_abs"] = math_reader._audit_adam_transition(
                entry, config, maxima["adam_parameter_abs"], maxima["adam_state_abs"], index)
        if index < 4:
            _exact(entry.get("update_state", entry["state"]), path[index + 1].get("state"),
                   f"{algorithm}.step{index}.state_chain")
            _exact(entry.get("optimizer_state"), path[index + 1].get("optimizer_before", path[index + 1].get("optimizer_state")),
                   f"{algorithm}.step{index}.Adam_chain")
            _need(torch.is_tensor(entry.get("actual_delta")),
                  f"{algorithm} step {index} omitted actual decoder delta")
            _tree_close(entry["actual_delta"], path[index + 1]["boxes"] - entry["boxes"],
                        1e-7, f"{algorithm}.step{index}.decoder_delta")
    _need(int(fit.get("optimizer_steps", -1)) == optimizer_steps,
          f"{algorithm} optimizer-step count differs from independently counted updates")
    _exact(fit.get("final_state"), path[-1].get("update_state", path[-1].get("state")),
           f"{algorithm}.final_state")
    final_cached = fit.get("final_cached_decoder_boxes")
    _need(torch.is_tensor(final_cached) and final_cached.shape == (len(frame_ids), 4)
          and torch.isfinite(final_cached).all(), f"{algorithm} final cached boxes malformed")
    _tree_close(final_cached - path[-1]["boxes"], path[-1].get("actual_delta"),
                1e-7, f"{algorithm}.last_decoder_delta")
    return {"steps": 5, "optimizer_steps": optimizer_steps,
            "max_abs_errors": maxima,
            "reward_semantics": "raw_iou_times_r_t_before_fixed_tau_softmax" if algorithm == "reliability_weighted_iou" else "original_iou",
            "cross_anchor_reliability_normalization": False,
            "saved_gradient_Adam_transition_checked": True,
            "decoder_autograd_gradient_independently_recomputed": False,
            "policy_samples_rewards_weights_and_likelihood_independently_recomputed": True}


def _cxcywh_to_xyxy_numpy(boxes: torch.Tensor):
    import numpy as np
    x = boxes.detach().cpu().double().numpy()
    return np.concatenate((x[:, :2] - .5*x[:, 2:], x[:, :2] + .5*x[:, 2:]), axis=1)


def _valid_interval(value: Any, label: str) -> None:
    _need(isinstance(value, (list, tuple)) and len(value) == 2
          and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                  and math.isfinite(float(x)) for x in value)
          and float(value[0]) < float(value[1]),
          f"{label} must be a finite increasing two-endpoint interval")


def _audit_case(case: Mapping[str, Any], roster: Mapping[str, Any], reference: Mapping[str, Any],
                old_case_path: Path, reference_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cid = roster["case_id"]
    _need(case.get("case_id") == cid and case.get("dataset") == roster.get("dataset")
          and case.get("source_case_sha256") == roster.get("source_case_sha256")
          and case.get("source_case_sha256") == reference.get("source_case_sha256"),
          f"student case source identity mismatch for {cid}")
    _need(case.get("GT_read") is False and case.get("online_carry") is False
          and case.get("source_restored") is True,
          f"student case GT/carry/source restoration flags invalid for {cid}")
    _valid_interval(case.get("Native_interval"), f"{cid}.Native_interval")
    _valid_interval(case.get("Projection_interval"), f"{cid}.Projection_interval")
    _need(case.get("reference_lock_sha256") == reference_sha,
          f"student case reference lock SHA mismatch for {cid}")
    _exact(case.get("reference_record"), reference, f"{cid}.reference_record")
    frame_ids = case.get("frame_ids")
    _need(isinstance(frame_ids, list) and len(frame_ids) >= 2
          and all(isinstance(x, int) and not isinstance(x, bool) for x in frame_ids)
          and all(b > a for a, b in zip(frame_ids, frame_ids[1:])),
          f"source physical frame grid malformed for {cid}")
    centers = roster.get("centers")
    _need(isinstance(centers, list) and len(centers) == 4,
          f"roster has no exact four source-grid center bindings for {cid}")
    for center in centers:
        position = center.get("source_grid_position")
        physical = center.get("center_physical_frame_id")
        _need(isinstance(position, int) and not isinstance(position, bool)
              and 0 <= position < len(frame_ids)
              and isinstance(physical, int) and not isinstance(physical, bool)
              and frame_ids[position] == physical,
              f"source grid differs from fixed roster center for {cid}")
    positions, evidence, weights = _anchors_for_audit(reference, frame_ids)
    no_update = not positions
    _need(reference.get("no_update") is no_update,
          f"reference no-update flag differs from actual anchors for {cid}")
    # The worker stores the plain original-domain configuration mapping at
    # result.configuration; it is not a wrapper record.
    config = case.get("configuration")
    _need(isinstance(config, Mapping) and config.get("steps") == 5 and config.get("samples") == 32,
          f"student config is not original OPD five-step/32-sample setting for {cid}")
    _need(_sha_file(ROOT / "methods/decota_spatial_opd_v1/configs.json") == CONFIG_SHA256,
          "original OPD domain configuration source changed")
    from scripts import stvg_target_teacher_student_v1 as frozen_core
    expected_config = frozen_core.load_original_opd_config(roster["dataset"])
    expected_config["steps"] = 5
    _exact(config, expected_config, f"{cid}.original_domain_config")
    key = f"caption_query_fixed_space|{cid}"

    source_boxes = case.get("source_boxes")
    source_logits = case.get("source_logits")
    _need(torch.is_tensor(source_boxes) and source_boxes.shape == (len(frame_ids), 4)
          and torch.isfinite(source_boxes).all(), f"saved source native boxes malformed for {cid}")
    _need(isinstance(source_logits, (list, tuple)) and len(source_logits) == 2
          and all(torch.is_tensor(x) and torch.isfinite(x).all() for x in source_logits),
          f"source temporal logits malformed for {cid}")
    cold = case.get("source_cold_readback")
    _need(isinstance(cold, Mapping) and cold.get("status") == "pass"
          and cold.get("source_boxes_logits_bitwise") is True
          and cold.get("full_model_state_sha256") == case.get("source_model_state_sha256")
          and cold.get("pixel_sha256") == case.get("pixel_sha256"),
          f"source capture receipt fields inconsistent for {cid}")
    source_xyxy = _cxcywh_to_xyxy_numpy(source_boxes)
    _tree_close(case.get("arms", {}).get("F", {}).get("boxes"), source_xyxy, 0.0,
                f"{cid}.F_vs_source_boxes")
    _need(case.get("arms", {}).get("F", {}).get("interval") == case.get("Native_interval"),
          f"{cid}.F interval differs from Native interval")

    fits, readouts, arms, receipts = (case.get("fits"), case.get("readouts"),
                                     case.get("arms"), case.get("receipts"))
    _need(isinstance(fits, Mapping) and isinstance(readouts, Mapping)
          and isinstance(arms, Mapping) and isinstance(receipts, Mapping),
          f"student fit/readout/arm/receipt maps malformed for {cid}")
    expected_fit_keys = set()
    for algorithm in ALGORITHMS:
        eligible = bool(positions) and (algorithm != "reliability_weighted_iou" or any(w > 0 for w in weights))
        if eligible:
            expected_fit_keys.add(algorithm)
        else:
            receipt = receipts.get(algorithm)
            _need(isinstance(receipt, Mapping) and receipt.get("status") == "exact_no_update"
                  and receipt.get("optimizer_steps") == 0 and receipt.get("final_redecode_calls") == 0,
                  f"{cid}/{algorithm} missing exact no-update receipt")
            _need(algorithm not in fits and algorithm not in readouts,
                  f"{cid}/{algorithm} should not create a fit/readout when no active references")
            _tree_close(arms.get(algorithm, {}).get("boxes"), source_xyxy, 0.0,
                        f"{cid}.{algorithm}.no_update_F_reuse")
            _need(arms.get(algorithm, {}).get("interval") == case.get("Native_interval"),
                  f"{cid}.{algorithm}.no_update interval differs from Native")
    _need(set(fits) == set(readouts) == expected_fit_keys,
          f"{cid} actual fit/readout keys differ from support-locked policy")
    _need(int(case.get("actual_fit_count", -1)) == len(fits)
          and int(case.get("optimizer_steps", -1)) == sum(int(f.get("optimizer_steps", -1)) for f in fits.values())
          and int(case.get("actual_final_redecode_calls", -1)) == len(readouts),
          f"{cid} top-level actual fit/readout counts disagree")

    initial = case.get("initial_state")
    if expected_fit_keys:
        _need(_cpu_state_numel(initial, f"{cid}.initial_state") == 1792,
              f"{cid} source initial state must be 1,792 coordinates")
    fit_checks: dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        if algorithm not in expected_fit_keys:
            continue
        fit = fits[algorithm]
        readout = readouts[algorithm]
        _need(fit.get("reference_role") == "teacher"
              and fit.get("reference_sha256") == reference_sha,
              f"{cid}/{algorithm} fit is not bound to the exact teacher reference lock")
        if algorithm == "direct":
            fit_checks[algorithm] = _audit_direct_fit(fit, readout, positions, evidence,
                                                       initial, expected_config)
        else:
            fit_checks[algorithm] = _audit_opd_fit(fit, positions, evidence, weights,
                frame_ids, key, expected_config, initial, algorithm)
        _need(fit.get("reference_reliability") == weights,
              f"{cid}/{algorithm} fit reliability metadata differs from reference anchors")
        _need(readout.get("logits") is not None, f"{cid}/{algorithm} readout omitted temporal logits")
        _exact(readout.get("logits"), source_logits, f"{cid}/{algorithm}.source_temporal_logits")
        _exact(readout.get("boxes"), fit.get("final_cached_decoder_boxes"),
               f"{cid}/{algorithm}.full_model_vs_cached_boxes")
        _tree_close(arms.get(algorithm, {}).get("boxes"), _cxcywh_to_xyxy_numpy(readout["boxes"]),
                    0.0, f"{cid}/{algorithm}.arm_boxes")
        _need(arms[algorithm].get("interval") == case.get("Native_interval"),
              f"{cid}/{algorithm} arm interval must preserve Native readout interval")
        receipt = receipts.get(algorithm)
        _need(isinstance(receipt, Mapping) and receipt.get("status") == "actual_updated_full_redecode"
              and receipt.get("optimizer_steps") == fit.get("optimizer_steps")
              and receipt.get("final_redecode_calls") == 1,
              f"{cid}/{algorithm} actual fit receipt mismatch")
        _need(isinstance(receipt.get("seconds"), (int, float))
              and not isinstance(receipt.get("seconds"), bool)
              and math.isfinite(float(receipt["seconds"])) and receipt["seconds"] >= 0,
              f"{cid}/{algorithm} elapsed seconds missing or invalid")

        t_arm = arms.get(f"T_{algorithm}")
        _need(isinstance(t_arm, Mapping), f"{cid}/T_{algorithm} arm missing")
        _tree_close(t_arm.get("boxes"), arms[algorithm]["boxes"], 0.0,
                    f"{cid}/T_{algorithm}.spatial_boxes")
        _valid_interval(t_arm.get("interval"), f"{cid}/T_{algorithm}.interval")

    # Independently bind old temporal-only output, then require every T_* arm
    # to reuse that interval and the new spatial arm's exact box array.
    old_case = torch.load(old_case_path, map_location="cpu", weights_only=False)
    _need(old_case.get("case_id") == cid and old_case.get("GT_read") is False
          and old_case.get("source_case_sha256") == case.get("source_case_sha256"),
          f"{cid} old temporal output source binding mismatch")
    _need(list(old_case.get("frame_ids", [])) == frame_ids,
          f"{cid} old temporal output uses a different source frame grid")
    old_t = old_case.get("arms", {}).get("T")
    _need(isinstance(old_t, Mapping), f"{cid} old temporal-only arm is missing")
    _tree_close(arms.get("T", {}).get("boxes"), old_t.get("boxes"), 0.0,
                f"{cid}.old_T_exact_boxes")
    _need(arms.get("T", {}).get("interval") == old_t.get("interval"),
          f"{cid}.old_T_exact_interval")
    for algorithm in ALGORITHMS:
        if f"T_{algorithm}" in arms:
            _tree_close(arms[f"T_{algorithm}"].get("boxes"), arms[algorithm].get("boxes"),
                        0.0, f"{cid}.T_{algorithm}.box_reuse")
            _need(arms[f"T_{algorithm}"].get("interval") == old_t.get("interval"),
                  f"{cid}.T_{algorithm}.old_interval_reuse")

    capture_seconds = case.get("source_capture_seconds")
    _need(isinstance(capture_seconds, (int, float)) and not isinstance(capture_seconds, bool)
          and math.isfinite(float(capture_seconds)) and capture_seconds >= 0,
          f"{cid} source capture seconds missing or invalid")
    math_row = {"dataset": roster["dataset"], "fit_count": len(fits),
                "optimizer_steps": int(case["optimizer_steps"]), "fits": fit_checks,
                "source_state_hash_consistent": case.get("source_model_state_sha256") == cold.get("full_model_state_sha256"),
                "saved_source_cold_receipt_self_consistent": True,
                "saved_full_model_readout_count": len(readouts),
                "saved_temporal_logits_preserved_count": len(readouts),
                "model_replayed_by_reader": False}
    cost_row = {"case_id": cid, "dataset": roster["dataset"],
                "source_case_sha256": roster["source_case_sha256"],
                "source_capture_seconds": float(capture_seconds),
                "algorithm_costs": {}}
    for algorithm in ALGORITHMS:
        receipt = receipts[algorithm]
        fit = fits.get(algorithm)
        cost_row["algorithm_costs"][algorithm] = {
            "seconds": (float(receipt["seconds"]) if isinstance(receipt.get("seconds"), (int, float))
                        and not isinstance(receipt.get("seconds"), bool) else None),
            "optimizer_steps": int(receipt.get("optimizer_steps", 0)),
            "final_redecode_calls": int(receipt.get("final_redecode_calls", 0)),
            "status": receipt.get("status"),
            "fit_present": fit is not None,
        }
    return math_row, cost_row


def audit_and_write() -> dict[str, Any]:
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU math reader requires CUDA to remain uninitialized")
    if READBACK_PATH.exists() or COST_PATH.exists():
        raise FileExistsError("refusing to overwrite an existing native-instance readback")
    gate = _preflight()
    # All eight result and old-T hashes are verified before any torch.load.
    math_rows: dict[str, Any] = {}
    cost_rows: list[dict[str, Any]] = []
    fit_total = step_total = readout_total = 0
    for cid in EXPECTED_IDS:
        case = torch.load(gate["case_paths"][cid], map_location="cpu", weights_only=False)
        math_row, cost_row = _audit_case(case, gate["roster_by_id"][cid],
            gate["reference_by_id"][cid], gate["old_case_paths"][cid], gate["reference_sha256"])
        math_rows[cid] = math_row
        cost_row["student_artifact_sha256"] = gate["artifact_hashes"][cid]["student_artifact_sha256"]
        cost_rows.append(cost_row)
        fit_total += math_row["fit_count"]
        step_total += math_row["optimizer_steps"]
        readout_total += math_row["saved_full_model_readout_count"]
        del case
    _need(fit_total == gate["expected_fit_count"],
          "aggregate actual fit count differs from locked expected fits")
    readback = {
        "schema": "stvg-native-instance-independent-student-math-readback-v1",
        "status": "pass",
        "scope": "sealed eight-case native-instance fit arithmetic and saved-readout consistency",
        "case_count": 8,
        "actual_fit_count": fit_total,
        "saved_optimizer_steps": step_total,
        "saved_full_model_readouts": readout_total,
        "source_capture_calls": 8,
        "case_checks": math_rows,
        "runtime_sha256": gate["runtime_sha256"],
        "student_barrier_sha256": gate["barrier_sha256"],
        "serial_storage_lock_sha256": gate["storage_lock_sha256"],
        "reference_lock_sha256": gate["reference_sha256"],
        "roster_sha256": gate["roster_sha256"],
        "GT_read": False,
        "GPU_initialized": False,
        "CPU_only": True,
        "model_replayed_by_reader": False,
        "interpretation_limit": "No decoder autograd graph is independently replayed: Direct gradients are inferred from saved Adam moments, while OPD saved gradients are checked as inputs to the Adam transition and their Gaussian policy math is independently recomputed. Full-model cached/readout equality was saved by the worker and checked here without replay.",
        "time": time.time(),
    }
    cost = {
        "schema": "stvg-native-instance-cost-readback-v1",
        "status": "complete",
        "GT_read": False,
        "count": len(cost_rows),
        "records": cost_rows,
        "student_barrier_sha256": gate["barrier_sha256"],
        "runtime_sha256": gate["runtime_sha256"],
        "serial_storage_lock_sha256": gate["storage_lock_sha256"],
        "CPU_only": True,
        "GPU_initialized": False,
        "time": time.time(),
    }
    for output_path, payload in ((READBACK_PATH, readback), (COST_PATH, cost)):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                          allow_nan=False) + "\n", encoding="utf-8")
    return readback


def _contracts() -> dict[str, Any]:
    """Synthetic-only CPU fixtures for the saved three-arm fit schemas."""
    if torch.cuda.is_initialized():
        raise RuntimeError("synthetic contracts must not initialize CUDA")
    # Mirror the exact extra-pin lists passed to lock_runtime() by the serial
    # controller.  In particular, TRACKING_ACTUAL_PHASE_EXIT belongs to the
    # teacher runtime, not to the later student runtime.
    base_rel = BASE.relative_to(ROOT).as_posix()
    writer_extra_pins = {
        "teacher": (f"{base_rel}/TRACKING_GLOBAL_BARRIER.json",
                    f"{base_rel}/TRACKING_ACTUAL_PHASE_EXIT.json"),
        "student": (f"{base_rel}/TRACKING_GLOBAL_BARRIER.json",
                    f"{base_rel}/TEACHER_GLOBAL_BARRIER.json",
                    f"{base_rel}/TEACHER_ACTUAL_PHASE_EXIT.json",
                    f"{base_rel}/REFERENCE_LOCK.json",
                    f"{base_rel}/DIRECT_TEACHER_READOUT.json"),
    }
    dummy_sha = "a" * 64
    writer_teacher_pins = {path: dummy_sha for path in writer_extra_pins["teacher"]}
    writer_student_pins = {path: dummy_sha for path in writer_extra_pins["student"]}
    _validate_phase_lineage_pins(writer_student_pins, writer_teacher_pins)
    _need(f"{base_rel}/TRACKING_ACTUAL_PHASE_EXIT.json" not in writer_student_pins,
          "writer-shaped student runtime must not require the tracking phase-exit pin")
    missing_teacher_exit = dict(writer_teacher_pins)
    del missing_teacher_exit[f"{base_rel}/TRACKING_ACTUAL_PHASE_EXIT.json"]
    try:
        _validate_phase_lineage_pins(writer_student_pins, missing_teacher_exit)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("teacher runtime without tracking exit receipt must fail lineage contract")

    math_reader = _opd_math()
    raw = torch.tensor([[.0, .1, .5, .9], [.2, .3, .4, .5]], dtype=torch.float32)
    reliability = torch.tensor([.25, .8], dtype=torch.float32)
    tau = .05
    used = raw * reliability[:, None]
    weights = torch.softmax(used / tau, dim=-1)
    expected = torch.softmax((raw * reliability[:, None]) / tau, dim=-1)
    _need(torch.equal(weights, expected), "weighted rewards must be scaled before softmax")
    _need(not torch.allclose(weights, torch.softmax(raw / tau, -1) * reliability[:, None]),
          "fixture must distinguish post-softmax reliability scaling")
    _need(torch.allclose(torch.softmax(raw / tau, -1),
                         torch.softmax((raw * torch.ones(2)[:, None]) / tau, -1)),
          "r=1 must reproduce unweighted action weights")
    zero_weights = torch.softmax((raw[:1] * 0) / tau, -1)
    _need(torch.equal(zero_weights, torch.full_like(zero_weights, .25)),
          "zero reliability yields uniform weights and no reward information")
    _need(not bool((zero_weights != zero_weights[:, :1]).any()),
          "zero reliability must signal a true no-information step")
    boxes = torch.tensor([[.5, .5, .2, .3], [.4, .6, .1, .2]], dtype=torch.float32)
    targets = torch.tensor([[.48, .52, .2, .3], [.42, .58, .1, .2]], dtype=torch.float32)
    loss = _direct_loss(boxes, [0, 1], targets)
    _need(torch.isfinite(loss) and float(loss) > 0, "synthetic Direct SpatialLoss must be finite")
    # Build a tiny Adam transition to exercise moment inversion without any
    # runtime model/feature or experiment artifact.
    cfg = {"lr": .01}
    param = torch.nn.Parameter(torch.linspace(-.4, .4, 1792, dtype=torch.float32))
    opt = torch.optim.Adam([param], lr=cfg["lr"], betas=(.9, .999), eps=1e-8, weight_decay=0.)
    grad = torch.linspace(-.1, .3, 1792, dtype=torch.float32)
    before_state = {"p": param.detach().clone()}
    before_opt = copy.deepcopy(opt.state_dict())
    param.grad = grad.clone()
    opt.step()
    after = copy.deepcopy(opt.state_dict())
    entry = {"state": before_state, "optimizer_before": before_opt,
             "optimizer_state": after, "update_state": {"p": param.detach().clone()}}
    recovered, moment_error = _optimizer_inferred_gradients(entry, {"lr": .01}, 0)
    _need(torch.allclose(recovered[0], grad, atol=2e-6, rtol=0)
          and moment_error < 1e-7, "Adam moment inversion contract failed")

    # Run the fit adapters on their in-module synthetic CPU decoder so this
    # reader exercises the exact saved schema for all arms. No experiment
    # cases, weights, media, labels, or CUDA are opened.
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts import stvg_video_global_spatial_fit_fp32_geometry001 as opd
    from scripts.stvg_native_instance_spatial_fit_v1 import fit_spatial_instance
    frame_ids = list(range(8))
    base = core._synthetic_student_base(len(frame_ids))
    initial = {name: value.detach().clone() for name, value in base.initial.items()}
    key = opd.locked_seed_key("native_instance_reader_contract")
    config = core.load_original_opd_config("hc2")
    config["steps"] = 5
    anchors = [
        {"position": 2, "physical_frame_id": frame_ids[2],
         "box": torch.tensor([.56, .48, .25, .31], dtype=torch.float32), "weight": 0.0},
        {"position": 5, "physical_frame_id": frame_ids[5],
         "box": torch.tensor([.43, .57, .16, .22], dtype=torch.float32), "weight": .4},
    ]
    positions = [row["position"] for row in anchors]
    evidence = torch.stack([row["box"] for row in anchors])
    reliabilities = [row["weight"] for row in anchors]
    ref_sha = "a" * 64
    synthetic_checks: dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        fit = fit_spatial_instance(base=base, initial=initial, anchors=anchors,
            frame_ids=frame_ids, key=key, config=config, algorithm=algorithm,
            reference_role="teacher", reference_sha256=ref_sha)
        synthetic_readout = {"boxes": fit["final_cached_decoder_boxes"]}
        if algorithm == "direct":
            check = _audit_direct_fit(fit, synthetic_readout, positions, evidence,
                                      initial, config)
        else:
            check = _audit_opd_fit(fit, positions, evidence, reliabilities,
                                   frame_ids, key, config, initial, algorithm)
        _need(fit.get("algorithm") == algorithm and fit.get("reference_role") == "teacher"
              and fit.get("reference_sha256") == ref_sha,
              f"synthetic {algorithm} fit identity binding contract failed")
        _need(fit.get("reference_reliability") == reliabilities,
              f"synthetic {algorithm} reliability provenance contract failed")
        synthetic_checks[algorithm] = check
    _need(not torch.cuda.is_initialized(), "CPU-only contract initialized CUDA")
    return {"status": "pass", "CPU_only": True, "GPU_initialized": False,
            "checks": ["writer-shaped teacher/student runtime lineage pins",
                       "tracking phase-exit pin is required from teacher, not student runtime",
                       "reliability scales each raw reward before fixed-tau softmax",
                       "no cross-anchor normalization", "r=1 matches unweighted weights",
                       "r=0 gives uniform/no-information weights", "Direct SpatialLoss arithmetic",
                       "Adam first/second moment inversion",
                       "saved Direct/IoU/reliability-weighted five-step synthetic fit schemas"],
            "limitations": ["synthetic contracts do not establish actual run validity",
                            "Direct gradient reconstruction is not decoder autograd"],
            "direct_loss_fixture": float(loss), "moment_error": moment_error,
            "synthetic_fit_checks": synthetic_checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", action="store_true",
                        help="run synthetic CPU contracts only; do not read scope files")
    args = parser.parse_args()
    result = _contracts() if args.contracts else audit_and_write()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
