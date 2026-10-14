"""Independent CPU audit for the sealed matched-support reference diagnosis.

The math phase reads only the locked reference geometry and sealed student
outputs after all eight predictions and their byte hashes pass preflight.  The
metric phase is separate and must wait for the completed math receipt and the
postseal report before loading official labels.  Neither phase loads a model
or initializes CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / "artifacts/stvg_matched_support_reference_v1"
OLD = ROOT / "artifacts/stvg_short_event_tracklets_v1"
CASE_IDS = tuple([f"hc2_{i:03d}" for i in range(4)] + [f"vidstg_{i:03d}" for i in range(4)])
ROLES = ("source", "r1", "pool_oracle", "gt_oracle")
HELPER_ROLES = {"source": "source", "r1": "teacher",
                "pool_oracle": "offline_oracle", "gt_oracle": "offline_oracle"}
ARMS = ("F", "T", *ROLES, *(f"T_{role}" for role in ROLES))
REFERENCE_PATH = BASE / "REFERENCE_LOCK.json"
SUPPORT_PATH = BASE / "SUPPORT_READBACK.json"
RUNTIME_PATH = BASE / "STUDENT_RUNTIME.json"
BARRIER_PATH = BASE / "STUDENT_GLOBAL_BARRIER.json"
MATH_PATH = BASE / "ROOT_STUDENT_MATH_READBACK.json"
REPORT_PATH = BASE / "ROOT_POSTSEAL_REPORT.json"
METRIC_READBACK_PATH = BASE / "ROOT_POSTSEAL_INDEPENDENT_READBACK.json"
FIT_AUDITOR = ROOT / "scripts/review_stvg_scene_object_id_spatial_math_v1.py"


class AuditError(RuntimeError):
    pass


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_relative(value: Any) -> str:
    _require(isinstance(value, str) and value and "\\" not in value,
             "expected a ROOT-relative POSIX path")
    p = Path(value)
    _require(not p.is_absolute() and all(part not in {"", ".", ".."} for part in p.parts),
             "absolute or traversing artifact path is forbidden")
    return value


def _verify_file(relative: Any, digest: Any, byte_count: Any = None) -> Path:
    rel = _safe_relative(relative)
    _require(isinstance(digest, str) and len(digest) == 64
             and all(c in "0123456789abcdef" for c in digest),
             f"malformed SHA256 for {rel}")
    p = ROOT / rel
    _require(p.is_file() and _sha_file(p) == digest, f"locked bytes changed: {rel}")
    if byte_count is not None:
        _require(p.stat().st_size == int(byte_count), f"locked byte count changed: {rel}")
    return p


def _exact_tensor(left: Any, right: Any, name: str) -> None:
    import torch
    try:
        a, b = torch.as_tensor(left).detach().cpu(), torch.as_tensor(right).detach().cpu()
    except Exception as exc:
        raise AuditError(f"{name} is not tensor-like") from exc
    _require(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b),
             f"{name} is not exact")


def _close_tensor(left: Any, right: Any, name: str, atol: float = 1e-7) -> float:
    import torch
    a, b = torch.as_tensor(left).detach().cpu(), torch.as_tensor(right).detach().cpu()
    _require(a.shape == b.shape, f"{name} shape differs")
    if a.numel() == 0:
        return 0.0
    err = float((a.to(torch.float64) - b.to(torch.float64)).abs().max())
    _require(math.isfinite(err) and err <= atol, f"{name} differs by {err:.9g}")
    return err


def _same_interval(left: Any, right: Any, name: str) -> None:
    _require(isinstance(left, (tuple, list)) and isinstance(right, (tuple, list))
             and len(left) == len(right) and list(left) == list(right),
             f"{name} interval differs")


def _compare_numeric_tree(left: Any, right: Any, name: str, atol: float = 1e-7) -> float:
    """Compare fit dynamics recursively while ignoring role-label metadata."""
    import numpy as np
    import torch
    if torch.is_tensor(left) or torch.is_tensor(right):
        return _close_tensor(left, right, name, atol)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return _close_tensor(left, right, name, atol)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        # Wall-clock diagnostics vary across sequential identical fits and are
        # not part of the optimizer/noise/readout dynamics being compared.
        timing = {"wall_seconds", "step_wall_seconds", "elapsed_seconds",
                  "duration_seconds", "end_to_end_seconds"}
        lk, rk = set(left) - timing, set(right) - timing
        _require(lk == rk, f"{name} mapping keys differ")
        return max((_compare_numeric_tree(left[k], right[k], f"{name}.{k}", atol)
                    for k in sorted(lk)), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        _require(type(left) is type(right) and len(left) == len(right),
                 f"{name} sequence schema differs")
        return max((_compare_numeric_tree(a, b, f"{name}[{i}]", atol)
                    for i, (a, b) in enumerate(zip(left, right))), default=0.0)
    if isinstance(left, (int, float)) and not isinstance(left, bool) and \
            isinstance(right, (int, float)) and not isinstance(right, bool):
        err = abs(float(left) - float(right))
        _require(math.isfinite(err) and err <= atol, f"{name} differs by {err:.9g}")
        return err
    _require(left == right, f"{name} scalar differs")
    return 0.0


def _anchor_geometry(reference: Mapping[str, Any]) -> tuple[list[int], Any]:
    import torch
    anchors = reference.get("anchors")
    _require(isinstance(anchors, list), "reference anchors must be a list")
    rows = []
    for a in anchors:
        _require(isinstance(a, Mapping), "malformed sparse anchor")
        pos = a.get("position")
        box = a.get("box_cxcywh")
        _require(isinstance(pos, int) and not isinstance(pos, bool),
                 "anchor position must be an integer source-grid index")
        _require(isinstance(box, (list, tuple)) and len(box) == 4,
                 "anchor box must be normalized cxcywh")
        vals = [float(x) for x in box]
        _require(all(math.isfinite(x) for x in vals), "anchor box contains nonfinite geometry")
        cx, cy, w, h = vals
        _require(w > 0 and h > 0 and -1e-7 <= cx - w / 2 < cx + w / 2 <= 1 + 1e-7
                 and -1e-7 <= cy - h / 2 < cy + h / 2 <= 1 + 1e-7,
                 "anchor violates normalized image bounds")
        _require(float(a.get("weight", 1.0)) == 1.0,
                 "matched-support OPD requires unit sparse-anchor weights")
        rows.append((pos, vals))
    rows.sort(key=lambda x: x[0])
    _require(len({p for p, _ in rows}) == len(rows), "duplicate source-grid anchor position")
    return [p for p, _ in rows], torch.tensor([b for _, b in rows], dtype=torch.float32).reshape(-1, 4)


def _validate_reference_lock(lock: Mapping[str, Any], support: Mapping[str, Any]) -> dict[str, Any]:
    _require(lock.get("schema") == "stvg-matched-support-reference-lock-v1"
             and lock.get("status") == "locked"
             and lock.get("scope") == "offline_diagnostic_only"
             and lock.get("GT_read") is True
             and lock.get("GT_used_for_oracle_references") is True
             and lock.get("GT_used_for_hyperparameters") is False,
             "reference lock is not the exact disclosed offline diagnostic scope")
    _require(lock.get("roles") == list(ROLES) and lock.get("count") == 8
             and lock.get("support_count") == 19 and lock.get("expected_fit_count") == 28,
             "reference lock counts or role catalog differ")
    _require(support.get("status") == "pass" and support.get("count") == 32
             and support.get("included") == 19 and support.get("excluded") == 13
             and support.get("GT_read") is True
             and support.get("GT_used_for_support_availability_only") is True,
             "support readback is incomplete or has a different GT-use scope")
    expected_support = {(r["case_id"], int(r["slot"])): r for r in support["records"]}
    _require(len(expected_support) == 32, "support readback does not cover 32 unique centers")
    lock_by_id = {r["case_id"]: r for r in lock.get("records", [])}
    _require(tuple(lock_by_id) == CASE_IDS, "reference lock case order differs from fixed roster")
    total_anchors = 0
    per_case = {}
    repeated_source_cases = 0
    repeated_source_windows = 0
    for cid in CASE_IDS:
        row = lock_by_id[cid]
        refs = row.get("references")
        _require(isinstance(refs, Mapping) and tuple(refs) == ROLES,
                 f"{cid} reference roles differ")
        _require(row.get("GT_read") is True and row.get("scope") == "offline_diagnostic_only",
                 f"{cid} lost the disclosed offline-only GT boundary")
        role_rows = {}
        for role in ROLES:
            ref = refs[role]
            anchors = ref.get("anchors")
            _require(isinstance(anchors, list), f"{cid}.{role} anchors are malformed")
            _require(ref.get("eligible_for_update") is bool(anchors)
                     and ref.get("no_update") is (not bool(anchors)),
                     f"{cid}.{role} update/no-update flag disagrees with support")
            _require(ref.get("offline_diagnostic") is True,
                     f"{cid}.{role} missing offline diagnostic flag")
            expected_label_use = role in ("pool_oracle", "gt_oracle")
            _require(all(a.get("offline_label_assisted") is expected_label_use for a in anchors),
                     f"{cid}.{role} label-assisted provenance flag differs")
            positions, _ = _anchor_geometry(ref)
            keys = {(int(a["slot"]), int(a["position"]), int(a["physical_frame_id"]))
                    for a in anchors}
            _require(len(keys) == len(anchors), f"{cid}.{role} duplicate support binding")
            role_rows[role] = (positions, keys, anchors)
            total_anchors += len(anchors)
            for a in anchors:
                key = (cid, int(a["slot"]))
                sr = expected_support.get(key)
                _require(sr is not None and sr.get("included") is True
                         and int(sr["position"]) == int(a["position"])
                         and int(sr["physical_frame_id"]) == int(a["physical_frame_id"]),
                         f"{cid}.{role} anchor is outside the exact locked support")
                if role == "source":
                    _require(a.get("candidate_id") == 1 and a.get("selected_candidate_id") == 1
                             and a.get("is_Source") is True,
                             f"{cid} Source reference is not the original native candidate")
                elif role == "r1":
                    _require(a.get("candidate_id") == a.get("selected_candidate_id"),
                             f"{cid} R1 anchor ID differs from its locked selected ID")
                elif role == "pool_oracle":
                    _require(isinstance(a.get("candidate_id"), int)
                             and a.get("candidate_id") == a.get("selected_candidate_id"),
                             f"{cid} candidate-pool Oracle is not a displayed candidate ID")
                elif role == "gt_oracle":
                    _require(a.get("candidate_id") is None
                             and a.get("selected_candidate_id") is None,
                             f"{cid} GT-box Oracle must not invent a candidate ID")
        key_sets = {role: val[1] for role, val in role_rows.items()}
        _require(len({frozenset(v) for v in key_sets.values()}) == 1,
                 f"{cid} four reference roles do not share identical support positions")
        support_keys = next(iter(key_sets.values()))
        if not support_keys:
            _require(all(refs[x]["no_update"] for x in ROLES),
                     f"{cid} empty support is not exact NoUpdate")
        source_by_key = {(a["slot"], a["position"], a["physical_frame_id"]): a
                         for a in refs["source"]["anchors"]}
        r1_by_key = {(a["slot"], a["position"], a["physical_frame_id"]): a
                     for a in refs["r1"]["anchors"]}
        common = 0
        for key in support_keys:
            sa, ra = source_by_key[key], r1_by_key[key]
            if (sa["candidate_id"] == ra["candidate_id"]
                    and sa["box_cxcywh"] == ra["box_cxcywh"]):
                common += 1
        repeated_source_windows += common
        if support_keys and common == len(support_keys):
            repeated_source_cases += 1
        per_case[cid] = len(support_keys)
    _require(total_anchors == 19 * 4, "reference lock does not contain exactly 76 anchor records")
    _require(sum(per_case.values()) == 19, "per-case support counts do not sum to 19")
    _require(repeated_source_cases == 5 and repeated_source_windows == 17,
             "Source/R1 overlap differs from the frozen matched-support lock")
    for (cid, slot), sr in expected_support.items():
        included = (sr.get("GT_known") is True
                    and sr.get("all_displayed_candidates_valid") is True
                    and sr.get("R1_valid_nonabstain") is True)
        _require(sr.get("included") is included
                 and sr.get("selection_uses_IoU_magnitude") is False,
                 f"{cid} support center {slot} uses an invalid or IoU-conditioned predicate")
    return {"per_case_support": per_case,
            "source_r1_exact_same_cases": repeated_source_cases,
            "source_r1_same_anchor_windows": repeated_source_windows,
            "anchor_record_count": total_anchors}


def _preflight() -> dict[str, Any]:
    """Check all seals and bytes before loading any student tensor."""
    import torch
    _require(not torch.cuda.is_initialized(), "CPU readback must not initialize CUDA")
    _require(RUNTIME_PATH.is_file() and BARRIER_PATH.is_file(),
             "student runtime or complete barrier is not available yet")
    runtime = _read_json(RUNTIME_PATH)
    _require(runtime.get("status") == "locked" and runtime.get("GT_read") is False
             and runtime.get("offline_diagnostic_only") is True
             and runtime.get("expected_fit_count") == 28 and runtime.get("steps") == 5
             and runtime.get("active_parameters") == 1792,
             "runtime does not bind the fixed no-GT 28-fit execution contract")
    ref_sha = _sha_file(REFERENCE_PATH)
    _require(runtime.get("reference_lock_sha256") == ref_sha,
             "student runtime does not bind exact reference lock bytes")
    pins = runtime.get("pins", {})
    protected = runtime.get("protected", {})
    _require(isinstance(pins, Mapping) and isinstance(protected, Mapping),
             "runtime pin maps are malformed")
    all_pins = {**dict(pins), **dict(protected)}
    for rel, digest in all_pins.items():
        _verify_file(rel, digest)

    barrier = _read_json(BARRIER_PATH)
    _require(barrier.get("schema") == "stvg-matched-support-student-barrier-v1"
             and barrier.get("status") == "complete" and barrier.get("count") == 8
             and barrier.get("case_ids") == list(CASE_IDS)
             and barrier.get("GT_read") is False
             and barrier.get("GT_derived_reference_consumed") is True
             and barrier.get("annotation_files_loaded_by_worker") is False
             and barrier.get("offline_diagnostic_only") is True,
             "student global barrier is not a complete no-GT eight-case catalog")
    runtime_sha = _sha_file(RUNTIME_PATH)
    if barrier.get("runtime_sha256") is not None:
        _require(barrier.get("runtime_sha256") == runtime_sha,
                 "student barrier is not bound to exact runtime bytes")
    student_barrier_sha = _sha_file(BARRIER_PATH)
    file_rows = barrier.get("files")
    _require(isinstance(file_rows, list) and len(file_rows) == 8,
             "student barrier must bind exactly eight case outputs")
    student_paths = {}
    for item in file_rows:
        rel = _safe_relative(item.get("path"))
        _require(rel.startswith("artifacts/stvg_matched_support_reference_v1/"),
                 "student output lies outside the locked scope")
        cid = str(item.get("case_id") or Path(rel).stem)
        _require(cid in CASE_IDS and cid not in student_paths,
                 "student barrier case catalog is duplicate or unexpected")
        student_paths[cid] = _verify_file(rel, item.get("sha256"), item.get("bytes"))
    _require(tuple(student_paths) == CASE_IDS,
             "student barrier output order/catalog differs from the fixed eight cases")
    _require(barrier.get("actual_fit_count") == 28,
             "student barrier actual fit count is not the locked 28")
    _require(barrier.get("actual_final_redecode_calls") == 28
             and barrier.get("reference_lock_sha256") == ref_sha
             and barrier.get("roster_lock_sha256") == _sha_file(BASE / "ROSTER_LOCK.json"),
             "student barrier does not bind the full fit/readout count and root locks")

    # The reference lock contains GT-derived boxes. Parse it only after all
    # model outputs are sealed and byte-verified; do not invoke a GT loader here.
    lock = _read_json(REFERENCE_PATH)
    support = _read_json(SUPPORT_PATH)
    support_audit = _validate_reference_lock(lock, support)
    _require(lock.get("source_bindings") and isinstance(lock["source_bindings"], Mapping),
             "reference lock lacks its original source-binding map")
    for rel, digest in lock["source_bindings"].items():
        _verify_file(rel, digest)
    roster = _read_json(BASE / "ROSTER_LOCK.json")
    _require(roster.get("count") == 8 and roster.get("status") == "locked",
             "scope roster is not the fixed eight-case roster")
    by_id = {r["case_id"]: r for r in roster.get("records", [])}
    _require(tuple(by_id) == CASE_IDS, "scope roster identity/order differs")
    refs_by_id = {r["case_id"]: r for r in lock["records"]}
    _require(set(refs_by_id) == set(CASE_IDS), "reference lock case catalog differs")

    old_barrier_path = OLD / "STUDENT_GLOBAL_BARRIER.json"
    _require(_sha_file(old_barrier_path) == lock.get("old_student_barrier_sha256"),
             "original source/F/T student barrier bytes differ from the lock")
    old_barrier = _read_json(old_barrier_path)
    _require(old_barrier.get("status") == "complete" and old_barrier.get("count") == 8,
             "original F/T baseline barrier is incomplete")
    old_rows = old_barrier.get("files")
    _require(isinstance(old_rows, list) and len(old_rows) == 8,
             "original F/T baseline barrier file catalog is malformed")
    old_paths = {}
    for item in old_rows:
        cid = str(item.get("case_id") or Path(_safe_relative(item["path"])).stem)
        _require(cid in CASE_IDS and cid not in old_paths, "duplicate original baseline file")
        old_paths[cid] = _verify_file(item["path"], item["sha256"], item.get("bytes"))
    _require(set(old_paths) == set(CASE_IDS), "original F/T baseline catalog differs")
    old_t_barrier_path = ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1/STUDENT_GLOBAL_BARRIER.json"
    old_t_barrier = _read_json(old_t_barrier_path)
    _require(old_t_barrier.get("status") in {"complete", "sealed"}
             and old_t_barrier.get("GT_read") is False and old_t_barrier.get("count") == 8,
             "separate frozen temporal baseline barrier is incomplete")
    old_t_rows = old_t_barrier.get("files")
    _require(isinstance(old_t_rows, list) and len(old_t_rows) == 8,
             "frozen temporal baseline catalog is malformed")
    old_t_paths = {}
    for item in old_t_rows:
        rel = _safe_relative(item.get("path"))
        _require(rel.startswith("artifacts/stvg_temporal_caption_query_spatial_opd_v1/student/"),
                 "frozen temporal baseline lies outside its original scope")
        cid = str(item.get("case_id") or Path(rel).stem)
        _require(cid in CASE_IDS and cid not in old_t_paths,
                 "frozen temporal baseline catalog has duplicate/unexpected case")
        old_t_paths[cid] = _verify_file(rel, item.get("sha256"), item.get("bytes"))
    _require(tuple(old_t_paths) == CASE_IDS,
             "frozen temporal baseline catalog order differs from the fixed roster")
    old_t_barrier_rel = old_t_barrier_path.relative_to(ROOT).as_posix()
    _require(pins.get(old_t_barrier_rel) == _sha_file(old_t_barrier_path),
             "student runtime does not pin the exact separate frozen T barrier")
    return {"runtime": runtime, "barrier": barrier, "reference_lock": lock,
            "support_readback": support, "roster_by_id": by_id,
            "reference_by_id": refs_by_id, "student_paths": student_paths,
            "old_paths": old_paths, "old_t_paths": old_t_paths,
            "support_audit": support_audit,
            "hashes": {"student_runtime_sha256": runtime_sha,
                       "student_barrier_sha256": student_barrier_sha,
                       "reference_lock_sha256": ref_sha,
                       "support_readback_sha256": _sha_file(SUPPORT_PATH),
                       "old_student_barrier_sha256": _sha_file(old_barrier_path),
                       "old_t_student_barrier_sha256": _sha_file(old_t_barrier_path),
                       "reader_sha256": _sha_file(Path(__file__).resolve())}}


def _xyxy(cxcywh: Any) -> Any:
    import torch
    values = torch.as_tensor(cxcywh).detach().cpu().to(torch.float64)
    cx, cy, w, h = values.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)


def _audit_case_math(cid: str, student: Mapping[str, Any], old_case: Mapping[str, Any],
                    old_t_case: Mapping[str, Any],
                    reference_case: Mapping[str, Any], manifest_case: Mapping[str, Any],
                    reference_lock_sha: str) -> dict[str, Any]:
    import torch
    _require(student.get("case_id") == cid and student.get("dataset") == reference_case.get("dataset"),
             f"{cid} student identity/domain differs")
    _require(student.get("cohort") == "old8"
             and student.get("GT_read") is False
             and student.get("GT_derived_reference_consumed") is True
             and student.get("annotation_files_loaded_by_worker") is False
             and student.get("offline_diagnostic_only") is True
             and student.get("reference_lock_sha256") == reference_lock_sha
             and student.get("source_restored") is True,
             f"{cid} student GT/reference/source-restoration scope differs")
    for key in ("source_case_sha256", "source_model_state_sha256", "manifest_case_sha256",
                "tracklet_case_sha256"):
        _require(student.get(key) == reference_case.get(key), f"{cid}.{key} differs from reference lock")
    frame_ids = student.get("frame_ids")
    _require(isinstance(frame_ids, list) and len(frame_ids) > 1
             and all(isinstance(v, int) and not isinstance(v, bool) for v in frame_ids)
             and all(b > a for a, b in zip(frame_ids, frame_ids[1:])),
             f"{cid} source physical frame grid is malformed")
    old_t_cold = old_t_case.get("source_cold_readback")
    _require(isinstance(old_t_cold, Mapping)
             and student.get("pixel_sha256") == old_case.get("pixel_sha256")
             == old_t_cold.get("pixel_sha256"),
             f"{cid} sampled-source-grid pixel hash differs across frozen old captures")
    cold = student.get("source_cold_readback")
    _require(isinstance(cold, Mapping) and cold.get("status") == "pass"
             and cold.get("source_case_sha256") == student.get("source_case_sha256")
             and cold.get("source_model_state_sha256") == student.get("source_model_state_sha256")
             and cold.get("pixel_sha256") == student.get("pixel_sha256")
             and cold.get("frame_ids_match_manifest") is True
             and cold.get("initial_state_match") is True
             and cold.get("zero_boxes_logits_match") is True
             and cold.get("source_restored") is True,
             f"{cid} cold source capture receipt is incomplete or mismatched")
    _require(list(map(int, old_t_case.get("frame_ids", []))) == list(map(int, frame_ids))
             and old_t_case.get("source_case_sha256") == student.get("source_case_sha256")
             and old_t_cold.get("full_model_state_sha256") == student.get("source_model_state_sha256")
             and old_t_cold.get("source_boxes_logits_bitwise") is True,
             f"{cid} frozen T baseline source/frame/model identity differs")
    _require(_canonical_sha(list(map(int, frame_ids))) == manifest_case.get("source_grid_frame_ids_sha256"),
             f"{cid} student frame grid differs from manifest")
    _require(student.get("reference_records") is not None
             and _canonical_sha(student["reference_records"]) == _canonical_sha(reference_case["references"]),
             f"{cid} worker reference geometry differs from locked role references")
    temporal_reuse = student.get("temporal_exact_reuse")
    old_t_path_rel = f"artifacts/stvg_temporal_caption_query_spatial_opd_v1/student/{cid}.pt"
    _require(isinstance(temporal_reuse, Mapping)
             and temporal_reuse.get("status") == "reused_exact_old_T"
             and temporal_reuse.get("source_case_sha256") == student.get("source_case_sha256")
             and temporal_reuse.get("old_student_path") == old_t_path_rel
             and temporal_reuse.get("old_student_sha256") == _sha_file(
                 ROOT / old_t_path_rel)
             and temporal_reuse.get("old_T_boxes_equal_F") is True
             and temporal_reuse.get("old_T_interval_reused") is True,
             f"{cid} exact temporal-baseline provenance is incomplete")
    arms, fits, readouts = student.get("arms"), student.get("fits"), student.get("readouts")
    _require(all(isinstance(x, Mapping) for x in (arms, fits, readouts)),
             f"{cid} arm/fit/readout map is malformed")
    _require(set(arms) == set(ARMS), f"{cid} arm catalog differs from F/T plus four-role matrix")
    refs = reference_case["references"]
    fit_roles = {r for r in ROLES if refs[r]["anchors"]}
    _require(set(fits) == fit_roles and set(readouts) == fit_roles,
             f"{cid} actual fits/readouts do not match nonempty locked references")
    _require(student.get("actual_fit_count") == len(fit_roles)
             and student.get("actual_final_redecode_calls") == len(readouts),
             f"{cid} fit/readout counters differ from saved outputs")
    _require(student.get("optimizer_steps") == sum(int(fits[r].get("optimizer_steps", -1)) for r in fit_roles),
             f"{cid} optimizer step total differs from individual fits")
    _exact_tensor(arms["F"].get("boxes"), old_case["arms"]["F"].get("boxes"), f"{cid}.F_reuse")
    _same_interval(arms["F"].get("interval"), old_case["arms"]["F"].get("interval"), f"{cid}.F_reuse")
    _exact_tensor(arms["T"].get("boxes"), old_t_case["arms"]["T"].get("boxes"), f"{cid}.T_reuse")
    _same_interval(arms["T"].get("interval"), old_t_case["arms"]["T"].get("interval"), f"{cid}.T_reuse")
    _exact_tensor(old_t_case["arms"]["T"].get("boxes"), old_t_case["arms"]["F"].get("boxes"),
                  f"{cid}.old_T_space_exact_F_reuse")
    config_row = student.get("configuration")
    _require(isinstance(config_row, Mapping)
             and config_row.get("key") == f"caption_query_fixed_space|{cid}"
             and config_row.get("scope") == "joint"
             and config_row.get("steps") == 5
             and config_row.get("active_parameters") == 1792,
             f"{cid} fit key or declared 5-step/1,792-coordinate contract differs")
    from scripts import stvg_target_teacher_student_v1 as core
    expected_config = core.load_original_opd_config(str(student["dataset"]))
    expected_config["steps"] = 5
    _require(config_row.get("config") == expected_config,
             f"{cid} did not preserve original domain-specific Gaussian OPD config")
    for role in ROLES:
        if not refs[role]["anchors"]:
            _exact_tensor(arms[role]["boxes"], arms["F"]["boxes"], f"{cid}.{role}.empty_F_reuse")
            _same_interval(arms[role]["interval"], arms["F"]["interval"], f"{cid}.{role}.empty_F_reuse")
            _exact_tensor(arms[f"T_{role}"]["boxes"], arms["T"]["boxes"], f"{cid}.T_{role}.empty_T_reuse")
            _same_interval(arms[f"T_{role}"]["interval"], arms["T"]["interval"], f"{cid}.T_{role}.empty_T_reuse")
            continue
        fit, readout = fits[role], readouts[role]
        ref_sha = _canonical_sha(refs[role])
        _require(fit.get("config") == expected_config and fit.get("scope") == "joint"
                 and fit.get("active_parameters") == 1792
                 and fit.get("declared_steps") == 5 and fit.get("selected_step") == 5
                 and fit.get("seed_key") == config_row["key"],
                 f"{cid}.{role} fit differs from fixed key/config/steps")
        _require(fit.get("reference_sha256") in (None, ref_sha),
                 f"{cid}.{role} fit reference digest differs from the locked role reference")
        helper_role = HELPER_ROLES[role]
        _require(fit.get("reference_role") == helper_role
                 and fit.get("reference_sha256") == ref_sha,
                 f"{cid}.{role} fit role/digest does not match the exact helper reference")
        anchors = _anchor_geometry(refs[role])
        from scripts.review_stvg_scene_object_id_spatial_math_v1 import _audit_opd_fit
        math_row = _audit_opd_fit(fit, anchors, frame_ids, config_row["key"], "iou",
                                  student["initial_state"])
        _exact_tensor(readout.get("boxes"), fit.get("final_cached_decoder_boxes"),
                      f"{cid}.{role}.full_redecode_cached_boxes")
        _close_tensor(arms[role].get("boxes"), _xyxy(readout["boxes"]),
                      f"{cid}.{role}.arm_matches_full_redecode")
        _exact_tensor(arms[f"T_{role}"].get("boxes"), arms[role].get("boxes"),
                      f"{cid}.T_{role}.spatial_boxes_reuse")
        _same_interval(arms[role].get("interval"), student.get("Native_interval"),
                       f"{cid}.{role}.native_interval")
        _same_interval(arms[f"T_{role}"].get("interval"), arms["T"].get("interval"),
                       f"{cid}.{role}.frozen_T_interval")
        if student.get("source_logits") is not None and readout.get("logits") is not None:
            _exact_tensor_list(readout["logits"], student["source_logits"], f"{cid}.{role}.source_logits")
        student.setdefault("_audit_role_results", {})[role] = math_row
    same_source_r1 = bool(refs["source"]["anchors"]) and all(
        a["position"] == b["position"]
        and a["physical_frame_id"] == b["physical_frame_id"]
        and a["box_cxcywh"] == b["box_cxcywh"]
        for a, b in zip(refs["source"]["anchors"], refs["r1"]["anchors"]))
    same_fit_dynamics_error = None
    if same_source_r1:
        _require("source" in fits and "r1" in fits,
                 f"{cid} identical Source/R1 references must be run as two real fits")
        same_fit_dynamics_error = _same_fit_dynamics(fits["source"], fits["r1"], cid)
        _exact_tensor(readouts["source"]["boxes"], readouts["r1"]["boxes"],
                      f"{cid}.same_target_source_r1_full_readouts")
    return {"dataset": student["dataset"], "fit_roles": sorted(fit_roles),
            "empty_roles": sorted(set(ROLES) - fit_roles),
            "fit_count": len(fit_roles),
            "fit_round_count": sum(row["steps"] for row in student.get("_audit_role_results", {}).values()),
            "optimizer_steps": int(student["optimizer_steps"]),
            "same_source_r1_anchor_count": sum(
                a["box_cxcywh"] == b["box_cxcywh"]
                for a, b in zip(refs["source"]["anchors"], refs["r1"]["anchors"])),
            "r1_source_all_anchors_identical": bool(refs["source"]["anchors"])
                and all(a["box_cxcywh"] == b["box_cxcywh"]
                        for a, b in zip(refs["source"]["anchors"], refs["r1"]["anchors"])),
            "same_target_source_r1_dynamics_max_abs_error": same_fit_dynamics_error,
            "role_math": student.get("_audit_role_results", {})}


def _exact_tensor_list(left: Any, right: Any, name: str) -> None:
    _require(isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
             and len(left) == len(right), f"{name} sequence differs")
    for i, (a, b) in enumerate(zip(left, right)):
        _exact_tensor(a, b, f"{name}[{i}]")


def _same_fit_dynamics(a: Mapping[str, Any], b: Mapping[str, Any], cid: str) -> float:
    """The same Source/R1 target must really fit twice and produce the same dynamics."""
    maxima = []
    for key in ("initial_state", "final_state", "path", "rounds", "optimizer_steps",
                "final_cached_decoder_boxes", "declared_steps", "selected_step", "config",
                "scope", "active_parameters", "seed_key", "reward_objective"):
        maxima.append(_compare_numeric_tree(a.get(key), b.get(key),
                                            f"{cid}.source_vs_r1.{key}", atol=0.0))
    return max(maxima, default=0.0)


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"refusing to overwrite immutable receipt: {path.name}")
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def audit_math(*, write_report: bool = True) -> dict[str, Any]:
    import torch
    gate = _preflight()
    _require(not torch.cuda.is_initialized(), "math reader initialized CUDA")
    from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
    manifest = _read_json(OLD / "PREPARED_MANIFEST.json")
    manifest_by_id = {r["case_id"]: r for r in manifest["cases"]}
    old_barrier = _read_json(OLD / "STUDENT_GLOBAL_BARRIER.json")
    old_paths = {}
    for row in old_barrier["files"]:
        cid = str(row.get("case_id") or Path(row["path"]).stem)
        old_paths[cid] = _verify_file(row["path"], row["sha256"], row.get("bytes"))
    old_t_barrier = _read_json(
        ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1/STUDENT_GLOBAL_BARRIER.json")
    old_t_paths = gate["old_t_paths"]

    fit_count = optimizer_steps = fit_round_count = 0
    role_counts = {role: 0 for role in ROLES}
    cases_out: dict[str, Any] = {}
    reference_sha = gate["hashes"]["reference_lock_sha256"]
    # Every input/output file was hashed before the first tensor deserialize.
    for cid in CASE_IDS:
        student = torch.load(gate["student_paths"][cid], map_location="cpu", weights_only=False)
        old_case = torch.load(old_paths[cid], map_location="cpu", weights_only=False)
        old_t_case = torch.load(old_t_paths[cid], map_location="cpu", weights_only=False)
        ref = gate["reference_by_id"][cid]
        roster_row = gate["roster_by_id"][cid]
        manifest_row = manifest_by_id[cid]
        _require(_sha_file(ROOT / roster_row["source_case_path"]) == ref["source_case_sha256"],
                 f"{cid} source case bytes differ from the locked source hash")
        source_case = load_source_case(ROOT / roster_row["source_case_path"])
        _require(source_case.get("dataset") == ref["dataset"]
                 and source_case.get("source_model_state_sha256", ref["source_model_state_sha256"])
                 == ref["source_model_state_sha256"],
                 f"{cid} source-case metadata differs from reference lock")
        result = _audit_case_math(cid, student, old_case, old_t_case,
                                  ref, manifest_row, reference_sha)
        fit_count += result["fit_count"]
        fit_round_count += result["fit_round_count"]
        optimizer_steps += result["optimizer_steps"]
        for role in result["fit_roles"]:
            role_counts[role] += 1
        # Only scalar checks are retained; no boxes, query, labels, or paths.
        cases_out[cid] = result
        del student, old_case, old_t_case, source_case

    _require(fit_count == 28 == gate["barrier"].get("actual_fit_count"),
             "actual fit count differs from fixed seven-nonempty-case/four-role matrix")
    _require(fit_round_count == 140,
             "the 28 actual fits do not each contain all five declared Gaussian rounds")
    _require(optimizer_steps == gate["barrier"].get("optimizer_steps"),
             "independently counted Adam updates differ from the sealed barrier")
    _require(role_counts == {role: 7 for role in ROLES},
             "each role must be fitted in all seven nonempty cases, including Source-selected R1")
    same_source_case_count = sum(bool(r["r1_source_all_anchors_identical"])
                                 for r in cases_out.values())
    _require(same_source_case_count == 5,
             "five exact Source/R1 target matches were not fitted as paired real fits")
    report = {
        "schema": "stvg-matched-support-reference-independent-math-readback-v1",
        "status": "pass",
        "scope": "offline matched-support reference diagnosis on eight historically exposed sources",
        "cases": 8,
        "student_barrier_sha256": gate["hashes"]["student_barrier_sha256"],
        "student_runtime_sha256": gate["hashes"]["student_runtime_sha256"],
        "reference_lock_sha256": gate["hashes"]["reference_lock_sha256"],
        "support_count": 19,
        "fit_count": fit_count,
        "expected_fit_count": 28,
        "fit_round_count": fit_round_count,
        "all_28_fits_have_five_rounds": True,
        "role_fit_count": role_counts,
        "optimizer_steps": optimizer_steps,
        "same_source_r1_anchor_cases_fitted_twice": same_source_case_count,
        "support_audit": gate["support_audit"],
        "case_readbacks": cases_out,
        "GT_read": True,
        "GT_reference_geometry_read_after_student_seal": True,
        "official_GT_label_loader_called": False,
        "GT_used_for_hyperparameter_or_method_selection": False,
        "offline_diagnostic_only": True,
        "GPU_initialized": False,
        "model_loaded": False,
        "source_model_reloaded": False,
        "video_or_RGB_read": False,
        "readback_scope": {
            "saved_Gaussian_rollout_and_Adam_math": True,
            "full_model_readout_equals_saved_cached_decoder": True,
            "F_and_T_reuse_checked_against_old_sealed_student": True,
            "T_reuse_checked_against_separate_old_temporal_barrier": True,
            "sampled_source_grid_pixel_sha_checked_against_both_frozen_captures": True,
            "decoder_backward_recomputed": False,
            "same_anchor_Source_and_R1_runs_compared_dynamically": True,
            "score_metrics_recomputed": False,
        },
        "provenance": gate["hashes"],
        "fit_math_auditor": {
            "path": FIT_AUDITOR.relative_to(ROOT).as_posix(),
            "sha256": _sha_file(FIT_AUDITOR),
        },
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report["case_roots"] = {cid: _canonical_sha(row) for cid, row in cases_out.items()}
    report["all_case_root"] = _canonical_sha(report["case_roots"])
    if write_report:
        _write_immutable(MATH_PATH, report)
    _require(not torch.cuda.is_initialized(), "math readback initialized CUDA")
    return report


def _box_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def _compare_float(a: Any, b: Any, name: str, tol: float = 2e-9) -> float:
    x, y = float(a), float(b)
    err = abs(x - y)
    _require(math.isfinite(err) and err <= tol, f"{name} differs by {err:.9g}")
    return err


def _parameter_delta_norm(initial: Mapping[str, Any], final: Mapping[str, Any]) -> float:
    """Independent float64 L2 norm of the saved 1,792-coordinate state delta."""
    import numpy as np
    _require(set(initial) == set(final), "initial/final fit parameter names differ")
    chunks = []
    count = 0
    for name in initial:
        before = np.asarray(initial[name].detach().cpu(), dtype=np.float64).reshape(-1)
        after = np.asarray(final[name].detach().cpu(), dtype=np.float64).reshape(-1)
        _require(before.shape == after.shape and np.isfinite(before).all()
                 and np.isfinite(after).all(), f"fit parameter tensor is malformed: {name}")
        chunks.append(after - before)
        count += int(before.size)
    _require(count == 1792, f"fit parameter delta contains {count}, not 1,792, coordinates")
    return float(np.linalg.norm(np.concatenate(chunks)))


def audit_metrics(*, write_report: bool = True) -> dict[str, Any]:
    """Recompute official dense metrics after both barriers and math receipt pass."""
    import numpy as np
    import torch
    gate = _preflight()
    _require(MATH_PATH.is_file() and REPORT_PATH.is_file(),
             "postseal metric phase requires the independent math receipt and root report")
    math_readback = _read_json(MATH_PATH)
    root_report = _read_json(REPORT_PATH)
    _require(math_readback.get("status") == "pass"
             and math_readback.get("student_barrier_sha256") == gate["hashes"]["student_barrier_sha256"],
             "math receipt does not bind this complete student barrier")
    _require(math_readback.get("reference_lock_sha256") == gate["hashes"]["reference_lock_sha256"]
             and math_readback.get("student_runtime_sha256") == gate["hashes"]["student_runtime_sha256"],
             "math receipt is not bound to this exact runtime and reference lock")
    _require(root_report.get("status") == "pass" and root_report.get("metric_count") == 80
             and root_report.get("support_count") == 19
             and root_report.get("actual_fit_count") == 28,
             "root postseal report is incomplete or has different scope counts")
    _require(root_report.get("student_barrier_sha256") == gate["hashes"]["student_barrier_sha256"]
             and root_report.get("reference_lock_sha256") == gate["hashes"]["reference_lock_sha256"]
             and root_report.get("root_math_sha256") == _sha_file(MATH_PATH),
             "root report is not bound to the independent math/reference/student receipts")
    from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
    from scripts.review_stvg_sparse_r1_v2 import _metric_functions, _truths_after_seal, score_tube_xyxy
    official, independent, _ = _metric_functions()
    source_cases = {}
    roster = gate["roster_by_id"]
    for cid in CASE_IDS:
        p = ROOT / roster[cid]["source_case_path"]
        _require(_sha_file(p) == gate["reference_by_id"][cid]["source_case_sha256"],
                 f"{cid} source bytes changed after math readback")
        source_cases[cid] = load_source_case(p)
    truths, gt_bindings = _truths_after_seal(source_cases)

    report_rows = {(r["case_id"], r["arm"]): r for r in root_report["rows"]}
    _require(len(report_rows) == 80, "root metric rows are not a unique 80-row case-arm matrix")
    computed_rows = []
    quality_rows = []
    movement_rows = []
    case_roots = {}
    for cid in CASE_IDS:
        student = torch.load(gate["student_paths"][cid], map_location="cpu", weights_only=False)
        source = source_cases[cid]
        truth = truths[cid]
        frame_ids = list(map(int, student["frame_ids"]))
        _require(frame_ids == list(map(int, source["frame_ids"])),
                 f"{cid} sealed output grid differs from canonical source grid")
        arms = student["arms"]
        _require(set(arms) == set(ARMS), f"{cid} arm set changed after math readback")
        vals = {}
        for arm in ARMS:
            interval = arms[arm].get("interval")
            metric = score_tube_xyxy(arms[arm]["boxes"], frame_ids, int(source["width"]),
                int(source["height"]), interval, truth["truth"], truth["span"],
                str(source["dataset"]), official=official, independent=independent)
            vals[arm] = metric
        case_rows = []
        for arm in ARMS:
            baseline = "T" if arm == "T" or arm.startswith("T_") else "F"
            support_role = arm.removeprefix("T_")
            expected_support_count = (len(gate["reference_by_id"][cid]["references"].get(
                support_role, {}).get("anchors", [])) if support_role in ROLES else 0)
            delta = 100.0 * (vals[arm]["metrics"]["m_vIoU"] - vals[baseline]["metrics"]["m_vIoU"])
            root_row = report_rows[(cid, arm)]
            errors = {}
            for metric_key in ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT"):
                errors[metric_key] = _compare_float(vals[arm]["metrics"][metric_key],
                    root_row["metrics"][metric_key], f"{cid}.{arm}.{metric_key}")
            errors["delta_vIoU_pp"] = _compare_float(delta, root_row["delta_vIoU_pp"],
                                                        f"{cid}.{arm}.delta_vIoU_pp")
            _require(root_row.get("baseline") == baseline,
                     f"{cid}.{arm} root report baseline label differs")
            _require(root_row.get("support_count") == expected_support_count,
                     f"{cid}.{arm} root support-count label differs")
            _require(float(root_row.get("official_independent_max_abs_error", 1.0)) <= 2e-9,
                     f"{cid}.{arm} official/independent metric agreement exceeds tolerance")
            item = {"case_id": cid, "dataset": source["dataset"], "arm": arm,
                    "metrics": vals[arm]["metrics"], "baseline": baseline,
                    "delta_vIoU_pp": delta,
                    "official_independent_max_abs_error": vals[arm]["official_independent_max_abs_error"],
                    "root_scalar_max_abs_error": max(errors.values())}
            computed_rows.append(item)
            case_rows.append(item)

        ref = gate["reference_by_id"][cid]
        for role in ROLES:
            for anchor in ref["references"][role]["anchors"]:
                frame = int(anchor["physical_frame_id"])
                box = anchor["box_cxcywh"]
                cx, cy, w, h = map(float, box)
                pred = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
                gt = [float(v) for v in truth["truth"][frame]]
                width, height = float(source["width"]), float(source["height"])
                gt_norm = [gt[0] / width, gt[1] / height, gt[2] / width, gt[3] / height]
                iou = _box_iou_xyxy(pred, gt_norm)
                quality_rows.append({"case_id": cid, "dataset": source["dataset"],
                    "role": role, "slot": int(anchor["slot"]),
                    "candidate_id": anchor["candidate_id"], "reference_IoU": iou,
                    "offline_label_assisted": role in ("pool_oracle", "gt_oracle")})
            if role in student["fits"]:
                fit = student["fits"][role]
                before_values, after_values = [], []
                f_boxes = torch.as_tensor(arms["F"]["boxes"]).detach().cpu().numpy()
                role_boxes = torch.as_tensor(arms[role]["boxes"]).detach().cpu().numpy()
                for anchor in ref["references"][role]["anchors"]:
                    pos = int(anchor["position"])
                    cx, cy, width, height = map(float, anchor["box_cxcywh"])
                    target = [cx - width / 2, cy - height / 2,
                              cx + width / 2, cy + height / 2]
                    before_values.append(_box_iou_xyxy(f_boxes[pos], target))
                    after_values.append(_box_iou_xyxy(role_boxes[pos], target))
                before_mean = statistics.mean(before_values)
                after_mean = statistics.mean(after_values)
                delta_pp = 100.0 * (after_mean - before_mean)
                movement_rows.append({"case_id": cid, "role": role,
                    "optimizer_steps": int(fit["optimizer_steps"]),
                    "parameter_delta_norm": _parameter_delta_norm(
                        fit["initial_state"], fit["final_state"]),
                    "mean_reference_IoU_before": before_mean,
                    "mean_reference_IoU_after": after_mean,
                    "reference_IoU_delta_pp": delta_pp})
        case_roots[cid] = _canonical_sha(case_rows)
        del student

    _require(len(computed_rows) == 80 and len(quality_rows) == 76,
             "independent postseal table sizes differ from the locked 80/76 outputs")
    root_quality = {(r["case_id"], r["role"], int(r["slot"])): r
                    for r in root_report["reference_quality"]}
    _require(len(root_quality) == 76, "root reference-quality rows are not a unique 76-row catalog")
    quality_errors = []
    for row in quality_rows:
        key = (row["case_id"], row["role"], row["slot"])
        expected = root_quality[key]
        _require(expected.get("candidate_id") == row["candidate_id"]
                 and expected.get("offline_label_assisted") == row["offline_label_assisted"],
                 f"reference quality identity/provenance differs at {key}")
        quality_errors.append(_compare_float(row["reference_IoU"], expected["reference_IoU"],
                                             f"reference_IoU.{key}"))

    root_movements = {(row["case_id"], row["role"]): row
                      for row in root_report.get("fit_movement", [])}
    _require(len(root_movements) == 28 and len(movement_rows) == 28,
             "fit-movement table does not cover all 28 actual fits")
    movement_errors = []
    for row in movement_rows:
        key = (row["case_id"], row["role"])
        expected = root_movements.get(key)
        _require(expected is not None, f"root fit-movement row missing for {key}")
        _require(expected.get("optimizer_steps") == row["optimizer_steps"],
                 f"fit-movement optimizer count differs for {key}")
        for field in ("parameter_delta_norm", "mean_reference_IoU_before",
                      "mean_reference_IoU_after", "reference_IoU_delta_pp"):
            movement_errors.append(_compare_float(row[field], expected[field],
                                                   f"fit_movement.{key}.{field}", tol=2e-8))

    groups = []
    for dataset in ("hc2", "vidstg"):
        for arm in ARMS:
            rows = [r for r in computed_rows if r["dataset"] == dataset and r["arm"] == arm]
            deltas = [float(r["delta_vIoU_pp"]) for r in rows]
            group = {"dataset": dataset, "arm": arm, "count": len(rows),
                "mean_vIoU": statistics.mean(r["metrics"]["m_vIoU"] for r in rows),
                "mean_delta_pp": statistics.mean(deltas),
                "positive": sum(x > 1e-8 for x in deltas),
                "negative": sum(x < -1e-8 for x in deltas),
                "neutral": sum(abs(x) <= 1e-8 for x in deltas)}
            root_group = next((g for g in root_report["groups"]
                               if g["dataset"] == dataset and g["arm"] == arm), None)
            _require(root_group is not None and root_group.get("count") == len(rows),
                     f"root group missing/count differs for {dataset}.{arm}")
            for key in ("mean_vIoU", "mean_delta_pp"):
                _compare_float(group[key], root_group[key], f"group.{dataset}.{arm}.{key}")
            for key in ("positive", "negative", "neutral"):
                _require(group[key] == root_group.get(key),
                         f"group.{dataset}.{arm}.{key} tail differs")
            groups.append(group)
    _require(len(groups) == 20, "domain/arm group matrix is incomplete")
    out = {"schema": "stvg-matched-support-independent-postseal-readback-v1",
        "status": "pass", "scope": "independent arithmetic comparison of sealed same-support diagnostic",
        "cases": 8, "support_count": 19, "metric_count": len(computed_rows),
        "reference_quality_count": len(quality_rows), "fit_movement_count": len(movement_rows),
        "group_count": len(groups),
        "max_metric_scalar_abs_error": max(r["root_scalar_max_abs_error"] for r in computed_rows),
        "max_reference_iou_abs_error": max(quality_errors, default=0.0),
        "max_fit_movement_scalar_abs_error": max(movement_errors, default=0.0),
        "metrics": computed_rows, "reference_quality": quality_rows,
        "fit_movement": movement_rows, "groups": groups,
        "case_roots": case_roots, "all_case_root": _canonical_sha(case_roots),
        "GT_read": True, "GT_loader_called_after_complete_prediction_and_math_seals": True,
        "GT_used_for_method_or_hyperparameter_selection": False,
        "offline_diagnostic_only": True, "GPU_initialized": False, "model_loaded": False,
        "video_or_RGB_read": False,
        "provenance": {**gate["hashes"], "math_readback_sha256": _sha_file(MATH_PATH),
            "root_postseal_report_sha256": _sha_file(REPORT_PATH),
            "GT_binding_sha256": hashlib.sha256(json.dumps(gt_bindings, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()},
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if write_report:
        _write_immutable(METRIC_READBACK_PATH, out)
    _require(not torch.cuda.is_initialized(), "metric readback initialized CUDA")
    return out


def run_cpu_contracts() -> dict[str, Any]:
    """Exercise only synthetic support, tie-breaking, and aggregation behavior."""
    import torch
    _require(not torch.cuda.is_initialized(), "CPU contracts require CUDA to remain uninitialized")
    # Deterministic candidate-pool oracle rule, including exact ties.
    synthetic = [(1, 0.5), (2, 0.7), (3, 0.7), (4, 0.1)]
    selected = min(synthetic, key=lambda row: (-row[1], row[0]))[0]
    _require(selected == 2, "pool oracle exact-tie rule must choose the smallest numeric ID")
    # Empty support cannot create pseudo-labels, fits, or optimizer updates.
    empty = {role: {"anchors": [], "no_update": True, "eligible_for_update": False}
             for role in ROLES}
    _require(all(not row["anchors"] and row["no_update"] and not row["eligible_for_update"]
                 for row in empty.values()), "empty synthetic support was not a strict NoUpdate")
    # Same target is still a real paired fit; it is never replaced by NoUpdate.
    anchors_a = [{"position": 3, "physical_frame_id": 9,
                  "box_cxcywh": [.5, .4, .2, .3], "candidate_id": 1}]
    anchors_b = [{**anchors_a[0], "reference_role": "r1"}]
    _require(anchors_a[0]["box_cxcywh"] == anchors_b[0]["box_cxcywh"]
             and anchors_a[0]["position"] == anchors_b[0]["position"],
             "same-source/R1 synthetic anchors differ")
    x = torch.tensor([.25, -.5], dtype=torch.float64)
    y = torch.tensor([.5, -.25], dtype=torch.float64)
    expected = {"positive": int((y - x > 1e-8).sum()),
                "negative": int((y - x < -1e-8).sum()),
                "neutral": int((y - x).abs().le(1e-8).sum())}
    _require(sum(expected.values()) == 2, "synthetic signed-tail aggregation lost rows")
    return {"status": "pass", "GT_read": False, "model_loaded": False,
            "GPU_initialized": False, "support_predicate": "known & candidate-valid & R1-valid",
            "oracle_tie": "smallest_candidate_id", "empty_support": "strict_no_update",
            "same_source_r1": "paired_real_fits_required", "tail_contract": expected}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("cpu-contracts", "audit-math", "audit-metrics"))
    args = parser.parse_args(argv)
    if args.mode == "cpu-contracts":
        result = run_cpu_contracts()
    elif args.mode == "audit-math":
        result = audit_math()
    else:
        result = audit_metrics()
    print(json.dumps({k: v for k, v in result.items()
                      if k not in {"case_readbacks", "metrics", "reference_quality", "groups"}},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
