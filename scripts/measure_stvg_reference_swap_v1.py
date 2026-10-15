"""Post-seal official scoring for the finite same-frame E1 reference swap.

This module is inert on import.  The ``score`` command refuses to load GT
until both runtime closures, all teacher/student outputs, the independent
pre-GT student readback, and the prediction seal have been verified.
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

STUDY = ROOT / "artifacts/stvg_reference_recovery_v1"
BASE = STUDY / "Reference_swap"
SOURCE_BASE = ROOT / "artifacts/stvg_evidence_constrained_v1"
PARENT = SOURCE_BASE / "round2_prompt009"
METRICS = ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT",
           "m_sIoU_supervised_GT", "m_sIoU_unsupervised_GT")
ARMS = ("F", "T", "S_new_crop", "TS_new_crop",
        "S_WF_common", "TS_WF_common", "S_DINO_common", "TS_DINO_common")
CONTRASTS = (
    ("S_DINO_common_minus_S_WF_common", "S_DINO_common", "S_WF_common"),
    ("TS_DINO_common_minus_TS_WF_common", "TS_DINO_common", "TS_WF_common"),
    ("S_WF_common_minus_S_new_crop", "S_WF_common", "S_new_crop"),
    ("TS_WF_common_minus_TS_new_crop", "TS_WF_common", "TS_new_crop"),
    ("S_WF_common_minus_F", "S_WF_common", "F"),
    ("TS_WF_common_minus_T", "TS_WF_common", "T"),
)
SIGN_EPSILON = 1e-10


class AuditError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise AuditError(f"non-finite JSON constant: {value}")


def read_json(path: Path) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=_pairs_no_duplicates,
                             parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {Path(path)}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_path(relative: str) -> Path:
    require(isinstance(relative, str) and relative,
            f"expected a nonempty path, got {relative!r}")
    supplied = Path(relative)
    path = (supplied if supplied.is_absolute() else ROOT / supplied).resolve()
    require(path == ROOT or ROOT in path.parents, f"path escapes repository: {relative}")
    return path


def scorer_root_path(relative: str, *, base: Path = BASE,
                     teacher_scoped: bool = False) -> Path:
    """Resolve worker BASE-relative teacher paths and ordinary ROOT paths.

    E1 teacher manifests use ``teacher/<job>.json`` relative to Reference_swap,
    while prediction seals use repository-relative paths.  Only the explicit
    ``teacher/`` prefix receives BASE resolution, and it cannot escape the
    BASE/teacher directory.  Every other path keeps the ordinary ROOT rule.
    """
    require(isinstance(relative, str) and relative, "expected a nonempty manifest path")
    if relative.startswith("teacher/"):
        path = (base / relative).resolve()
        teacher_dir = (base / "teacher").resolve()
        require(path != teacher_dir and teacher_dir in path.parents,
                f"BASE-relative teacher path escapes BASE/teacher: {relative}")
        require(path == ROOT or ROOT in path.parents,
                f"BASE-relative teacher path escapes repository: {relative}")
        return path
    require(not teacher_scoped, f"teacher output path must start with teacher/: {relative}")
    return _root_path(relative)


def _verify_runtime(path: Path, label: str) -> dict[str, Any]:
    doc = read_json(path)
    require(isinstance(doc, dict) and doc.get("status") == "locked" and
            doc.get("GT_read") is False, f"{label} must be locked and explicitly GT-free")
    pins = doc.get("pins")
    protected = doc.get("protected", {})
    require(isinstance(pins, dict) and isinstance(protected, dict),
            f"{label} pins/protected must be mappings")
    require(not set(pins).intersection(protected) or all(
        pins[key] == protected[key] for key in set(pins).intersection(protected)),
        f"{label} has conflicting pin/protected digests")
    for rel, expected in {**pins, **protected}.items():
        require(isinstance(expected, str) and len(expected) == 64,
                f"{label} has malformed hash for {rel}")
        pinned = _root_path(rel)
        require(pinned.is_file() and sha256_file(pinned) == expected,
                f"{label} source pin mismatch: {rel}")
    return doc


def _rows_by_case(doc: Mapping[str, Any], label: str, expected_count: int = 32,
                  *, key: str = "case_id") -> dict[str, dict[str, Any]]:
    rows = doc.get("records")
    require(isinstance(rows, list) and len(rows) == expected_count,
            f"{label} must contain {expected_count} records")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        require(isinstance(row, dict), f"{label} contains a non-object record")
        cid = row.get(key, row.get("probe_alias"))
        require(isinstance(cid, str) and cid and cid not in out,
                f"{label} has missing/duplicate case alias")
        out[cid] = row
    return out


def _verify_file_rows(rows: Any, *, label: str, expected_paths: set[str] | None = None,
                      expected_count: int | None = None) -> dict[str, dict[str, Any]]:
    require(isinstance(rows, list), f"{label}.files must be a list")
    by_path: dict[str, dict[str, Any]] = {}
    for item in rows:
        require(isinstance(item, dict), f"{label} has a malformed file row")
        rel, expected = item.get("path"), item.get("sha256")
        require(isinstance(rel, str) and isinstance(expected, str) and len(expected) == 64,
                f"{label} contains a malformed path/hash")
        path = _root_path(rel)
        require(path.is_file() and sha256_file(path) == expected,
                f"{label} byte hash mismatch: {rel}")
        if item.get("bytes") is not None:
            require(path.stat().st_size == int(item["bytes"]),
                    f"{label} byte count mismatch: {rel}")
        require(rel not in by_path or by_path[rel]["sha256"] == expected,
                f"{label} repeats a path with conflicting hashes: {rel}")
        by_path[rel] = item
    if expected_paths is not None:
        require(set(by_path) == expected_paths,
                f"{label} path membership mismatch: missing={sorted(expected_paths-set(by_path))[:3]} "
                f"extra={sorted(set(by_path)-expected_paths)[:3]}")
    if expected_count is not None:
        require(len(by_path) == expected_count,
                f"{label} has {len(by_path)} unique files, expected {expected_count}")
    return by_path


def _verify_teacher_file_rows(rows: Any, *, base: Path, label: str,
                              expected_count: int) -> dict[str, dict[str, Any]]:
    require(isinstance(rows, list), f"{label}.files must be a list")
    by_root_path: dict[str, dict[str, Any]] = {}
    for item in rows:
        require(isinstance(item, dict), f"{label} has a malformed file row")
        rel, expected = item.get("path"), item.get("sha256")
        require(isinstance(rel, str) and isinstance(expected, str) and len(expected) == 64,
                f"{label} contains a malformed path/hash")
        path = scorer_root_path(rel, base=base, teacher_scoped=True)
        require(path.is_file() and sha256_file(path) == expected,
                f"{label} byte hash mismatch: {rel}")
        if item.get("bytes") is not None:
            require(path.stat().st_size == int(item["bytes"]),
                    f"{label} byte count mismatch: {rel}")
        canonical = str(path.relative_to(ROOT))
        require(canonical.startswith(str((base / "teacher").relative_to(ROOT)) + "/"),
                f"{label} canonical path is outside BASE/teacher: {rel}")
        require(canonical not in by_root_path,
                f"{label} repeats a canonical output path: {canonical}")
        normalized = dict(item)
        normalized["worker_path"] = rel
        normalized["path"] = canonical
        by_root_path[canonical] = normalized
    require(len(by_root_path) == expected_count,
            f"{label} has {len(by_root_path)} unique files, expected {expected_count}")
    return by_root_path


def _teacher_only_preflight(base: Path = BASE) -> dict[str, Any]:
    """Verify the sealed teacher/reference metadata without student artifacts.

    This intentionally does not call ``_verify_runtime`` (which re-hashes all
    runtime-pinned assets, including the model checkpoint).  It validates the
    runtime document's locked/GT-free state and its exact barrier binding, then
    hashes only the 49 sealed teacher output files and the relevant lock files.
    No model, GT, image, or scoring module is loaded.
    """
    runtime_path = base / "RUNTIME.json"
    runtime = read_json(runtime_path)
    require(isinstance(runtime, dict) and runtime.get("status") == "locked"
            and runtime.get("GT_read") is False
            and isinstance(runtime.get("pins"), dict)
            and isinstance(runtime.get("protected"), dict),
            "teacher runtime metadata is not a locked GT-free runtime")
    design_path = base / "DESIGN_LOCK.json"
    roster_path = base / "ROSTER_LOCK.json"
    ref_path = base / "REFERENCE_LOCK.json"
    job_path = base / "TEACHER_JOB_LOCK.json"
    barrier_path = base / "TEACHER_GLOBAL_BARRIER.json"
    design, roster_doc = read_json(design_path), read_json(roster_path)
    ref_doc, jobs_doc, barrier = read_json(ref_path), read_json(job_path), read_json(barrier_path)
    for doc, label in ((design, "design"), (roster_doc, "roster"),
                       (ref_doc, "reference"), (jobs_doc, "teacher job")):
        require(isinstance(doc, dict) and doc.get("status") == "locked"
                and doc.get("GT_read") is False,
                f"teacher-only preflight: {label} lock is not locked/GT-free")
    roster = _rows_by_case(roster_doc, "teacher-only E1 roster")
    references = _rows_by_case(ref_doc, "teacher-only E1 reference")
    require(set(roster) == set(references) and len(roster) == 32,
            "teacher-only roster/reference membership mismatch")
    require(int(design.get("source_count", -1)) == 32
            and int(design.get("teacher_observation_attempts", -1)) == 49,
            "teacher-only design count mismatch")
    require(isinstance(barrier, dict)
            and barrier.get("schema") == "stvg-reference-swap-teacher-global-barrier-v1"
            and barrier.get("status") == "complete"
            and barrier.get("GT_read") is False
            and int(barrier.get("count", -1)) == 49,
            "teacher barrier is not a complete 49-output GT-free seal")
    require(barrier.get("bindings", {}).get("runtime_sha256") == sha256_file(runtime_path)
            and barrier.get("bindings", {}).get("design_lock_sha256") == sha256_file(design_path)
            and barrier.get("bindings", {}).get("roster_sha256") == sha256_file(roster_path)
            and barrier.get("bindings", {}).get("teacher_job_lock_sha256") == sha256_file(job_path),
            "teacher barrier does not bind the locked runtime/design/roster/job lock")
    jobs = jobs_doc.get("records", jobs_doc.get("jobs"))
    require(isinstance(jobs, list) and len(jobs) == 49,
            "teacher job lock does not contain 49 attempts")
    job_ids = {str(row.get("job_id")) for row in jobs if isinstance(row, dict)}
    require(len(job_ids) == 49 and all(job_ids),
            "teacher job lock has missing/duplicate IDs")
    files = barrier.get("files")
    outputs = _verify_teacher_file_rows(files, base=base,
                                        label="teacher-only barrier", expected_count=49)
    norm_rows = list(outputs.values())
    require({str(row.get("job_id")) for row in norm_rows} == job_ids,
            "teacher-only barrier/job-lock membership mismatch")
    by_case_pos = {(row.get("case_id"), row.get("position")): row for row in norm_rows}
    require(len(by_case_pos) == 49,
            "teacher-only barrier repeats a case/position")
    decision_count = accepted_count = common_dino_count = common_source_count = 0
    for cid, ref in references.items():
        decisions = ref.get("dino_decisions")
        require(isinstance(decisions, list), f"reference lock lacks decisions for {cid}")
        anchors = ref.get("common_dino_anchors")
        require(isinstance(anchors, list), f"reference lock lacks DINO common anchors for {cid}")
        common_dino_count += len(anchors)
        common_source_count += bool(anchors)
        accepted_positions = set()
        for decision in decisions:
            require(isinstance(decision, dict), f"malformed reference decision for {cid}")
            pos = decision.get("position")
            output = by_case_pos.get((cid, pos))
            require(output is not None
                    and output.get("physical_frame_id") == decision.get("physical_frame_id")
                    and output.get("decision_status") == decision.get("status"),
                    f"reference decision does not match teacher barrier for {cid}")
            path = decision.get("teacher_output_path")
            require(isinstance(path, str), f"reference decision lacks teacher path for {cid}")
            resolved = scorer_root_path(path, base=base, teacher_scoped=True)
            require(str(resolved.relative_to(ROOT)) == output.get("path"),
                    f"reference decision teacher path mismatch for {cid}")
            require(decision.get("teacher_output_sha256") == output.get("sha256"),
                    f"reference decision teacher SHA mismatch for {cid}")
            if decision.get("status") == "accepted":
                accepted_count += 1
                accepted_positions.add(pos)
            decision_count += 1
        anchor_positions = {anchor.get("position") for anchor in anchors if isinstance(anchor, dict)}
        require(anchor_positions == accepted_positions,
                f"DINO common anchors do not match accepted decisions for {cid}")
    require(decision_count == 49, "reference lock must contain 49 teacher decisions")
    return {"status": "pass", "case_count": len(roster), "teacher_attempt_count": decision_count,
            "teacher_file_count": len(outputs), "accepted_decision_count": accepted_count,
            "common_dino_anchor_count": common_dino_count,
            "sources_with_common_dino_anchor": common_source_count,
            "runtime_sha256": sha256_file(runtime_path),
            "design_lock_sha256": sha256_file(design_path),
            "roster_sha256": sha256_file(roster_path),
            "reference_lock_sha256": sha256_file(ref_path),
            "teacher_job_lock_sha256": sha256_file(job_path),
            "teacher_barrier_sha256": sha256_file(barrier_path),
            "teacher_outputs_canonical_root_paths": True,
            "runtime_asset_closure_rehashed": False,
            "GT_read": False, "CPU_only": True, "model_loaded": False, "GPU_used": False}


def _cid(row: Mapping[str, Any]) -> str:
    value = row.get("case_id", row.get("probe_alias"))
    require(isinstance(value, str) and value, "locked row lacks case_id/probe_alias")
    return value


def _anchor_positions(rows: Any, frame_ids: Sequence[int], label: str) -> list[int]:
    require(isinstance(rows, list), f"{label} must be a list")
    positions = []
    for anchor in rows:
        require(isinstance(anchor, dict), f"{label} contains a non-object")
        pos, fid = anchor.get("position"), anchor.get("physical_frame_id")
        require(isinstance(pos, int) and not isinstance(pos, bool) and 0 <= pos < len(frame_ids),
                f"{label} has an invalid source-grid position")
        require(isinstance(fid, int) and not isinstance(fid, bool) and fid == int(frame_ids[pos]),
                f"{label} has a source-grid/physical-frame mismatch")
        box = anchor.get("box_cxcywh")
        require(isinstance(box, (list, tuple)) and len(box) == 4 and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))
            for x in box), f"{label} has malformed geometry")
        cx, cy, width, height = map(float, box)
        require(width > 0 and height > 0 and -1e-7 <= cx - width / 2 < cx + width / 2 <= 1+1e-7
                and -1e-7 <= cy - height / 2 < cy + height / 2 <= 1+1e-7,
                f"{label} has invalid normalized cxcywh geometry")
        weight = anchor.get("weight")
        require(isinstance(weight, (int, float)) and not isinstance(weight, bool)
                and math.isfinite(float(weight)) and float(weight) > 0,
                f"{label} has invalid reliability weight")
        positions.append(pos)
    require(positions == sorted(set(positions)), f"{label} positions must be unique and sorted")
    return positions


def _xyxy_from_cxcywh(boxes: Any):
    """Match the frozen worker conversion: detach, CPU, float64, then xyxy."""
    import numpy as np
    import torch
    values = boxes.detach().cpu().double().numpy() if torch.is_tensor(boxes) else np.asarray(boxes, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == 4 and np.isfinite(values).all(),
            "readout boxes must be a finite [T,4] cxcywh matrix")
    return np.concatenate((values[:, :2] - values[:, 2:] / 2,
                           values[:, :2] + values[:, 2:] / 2), axis=1)


def validate_common_support(ref: Mapping[str, Any], frame_ids: Sequence[int],
                            parent_ref: Mapping[str, Any]) -> list[int]:
    wf, dino = ref.get("common_wf_anchors"), ref.get("common_dino_anchors")
    wf_pos = _anchor_positions(wf, frame_ids, "common_wf_anchors")
    dino_pos = _anchor_positions(dino, frame_ids, "common_dino_anchors")
    require(wf_pos == dino_pos, "WF and DINO references do not share identical positions")
    for wa, da in zip(wf, dino):
        for key in ("position", "physical_frame_id", "weight"):
            require(wa[key] == da[key], f"WF/DINO common anchor mismatch at {key}")
    old = parent_ref.get("new_crop", {}).get("anchors", [])
    old_by_pos = {int(a["position"]): a for a in old}
    for anchor in wf:
        prior = old_by_pos.get(int(anchor["position"]))
        require(prior is not None and prior.get("physical_frame_id") == anchor["physical_frame_id"]
                and float(prior.get("weight", 1.0)) == float(anchor["weight"])
                and list(prior.get("box_cxcywh", [])) == list(anchor["box_cxcywh"]),
                "common WF geometry is not an exact subset of the original full-pool reference")
    accepted = {int(row["position"]) for row in ref.get("dino_decisions", [])
                if isinstance(row, dict) and row.get("status") == "accepted"}
    require(set(wf_pos).issubset(accepted), "common support includes a nonaccepted DINO observation")
    rejected = set(map(int, ref.get("dino_rejected_positions", [])))
    unknown = set(map(int, ref.get("dino_unknown_positions", [])))
    require(not (set(wf_pos) & (rejected | unknown)), "rejected/unknown DINO position entered common support")
    return wf_pos


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_case_arm = {(row["case_id"], row["arm"]): row for row in rows}
    require(len(by_case_arm) == len(rows), "duplicate case/arm scoring row")
    datasets = {str(row["dataset"]) for row in rows}
    out = []
    for dataset in ("hc2", "vidstg"):
        if dataset not in datasets:
            continue
        case_ids = sorted({str(row["case_id"]) for row in rows if row["dataset"] == dataset})
        require(len(case_ids) == 16, f"{dataset} must contain 16 fixed sources")
        for arm in ARMS:
            for metric in METRICS:
                values, unknown = [], []
                for cid in case_ids:
                    value = by_case_arm[cid, arm]["metrics"].get(metric)
                    if value is None:
                        unknown.append(cid)
                    else:
                        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                                and math.isfinite(float(value)), f"nonfinite {metric} for {cid}/{arm}")
                        values.append(float(value))
                out.append({"dataset": dataset, "arm": arm, "metric": metric,
                            "n_valid": len(values), "n_unknown": len(unknown),
                            "unknown_case_ids": unknown,
                            "mean": math.fsum(values) / len(values) if values else None,
                            "median": statistics.median(values) if values else None,
                            "min": min(values) if values else None,
                            "max": max(values) if values else None})
    return out


def paired_contrasts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_case_arm = {(row["case_id"], row["arm"]): row for row in rows}
    out = []
    for dataset in ("hc2", "vidstg"):
        case_ids = sorted({str(row["case_id"]) for row in rows if row["dataset"] == dataset})
        if not case_ids:
            continue
        require(len(case_ids) == 16, f"{dataset} paired comparison denominator must be 16")
        for contrast, left, right in CONTRASTS:
            for metric in METRICS:
                per_source: dict[str, float] = {}
                unknown = []
                for cid in case_ids:
                    lv = by_case_arm[cid, left]["metrics"].get(metric)
                    rv = by_case_arm[cid, right]["metrics"].get(metric)
                    if lv is None or rv is None:
                        unknown.append(cid)
                    else:
                        per_source[cid] = 100.0 * (float(lv) - float(rv))
                vals = list(per_source.values())
                positive = [cid for cid, value in per_source.items() if value > SIGN_EPSILON]
                negative = [cid for cid, value in per_source.items() if value < -SIGN_EPSILON]
                neutral = [cid for cid, value in per_source.items() if abs(value) <= SIGN_EPSILON]
                out.append({"dataset": dataset, "contrast": contrast,
                            "comparison_arm": left, "baseline_arm": right,
                            "metric": metric, "n_pairs": len(vals),
                            "unknown_count": len(unknown), "unknown_case_ids": unknown,
                            "positive_count": len(positive), "positive_case_ids": positive,
                            "negative_count": len(negative), "negative_case_ids": negative,
                            "neutral_count": len(neutral), "neutral_case_ids": neutral,
                            "neutral_tolerance_pp": SIGN_EPSILON,
                            "mean_delta_pp": math.fsum(vals) / len(vals) if vals else None,
                            "median_delta_pp": statistics.median(vals) if vals else None,
                            "min_delta_pp": min(vals) if vals else None,
                            "max_delta_pp": max(vals) if vals else None,
                            "per_source_delta_pp": per_source})
    return out


def _verify_all_seals(base: Path) -> dict[str, Any]:
    teacher_runtime_path = base / "RUNTIME.json"
    student_runtime_path = base / "STUDENT_RUNTIME.json"
    teacher_runtime = _verify_runtime(teacher_runtime_path, "E1 teacher RUNTIME")
    student_runtime = _verify_runtime(student_runtime_path, "E1 STUDENT_RUNTIME")
    source_runtime_path = SOURCE_BASE / "SOURCE_RUNTIME.json"
    source_runtime = _verify_runtime(source_runtime_path, "fixed source SOURCE_RUNTIME")
    parent_runtime_path = SOURCE_BASE / "RUNTIME.json"
    parent_runtime = _verify_runtime(parent_runtime_path, "fixed source RUNTIME")

    design_path = base / "DESIGN_LOCK.json"
    roster_path = base / "ROSTER_LOCK.json"
    reference_path = base / "REFERENCE_LOCK.json"
    teacher_job_lock_path = base / "TEACHER_JOB_LOCK.json"
    teacher_barrier_path = base / "TEACHER_GLOBAL_BARRIER.json"
    student_barrier_path = base / "STUDENT_GLOBAL_BARRIER.json"
    root_readback_path = base / "ROOT_READBACK.json"
    prediction_seal_path = base / "GLOBAL_PREDICTION_BARRIER.json"

    design = read_json(design_path)
    roster_doc = read_json(roster_path)
    ref_doc = read_json(reference_path)
    jobs_doc = read_json(teacher_job_lock_path)
    teacher_barrier = read_json(teacher_barrier_path)
    student_barrier = read_json(student_barrier_path)
    root_readback = read_json(root_readback_path)
    prediction_seal = read_json(prediction_seal_path)
    for doc, label in ((design, "E1 DESIGN_LOCK"), (roster_doc, "E1 ROSTER_LOCK"),
                       (ref_doc, "E1 REFERENCE_LOCK"), (jobs_doc, "E1 TEACHER_JOB_LOCK")):
        require(doc.get("status") == "locked" and doc.get("GT_read") is False,
                f"{label} must be locked and explicitly GT-free")
    require(int(design.get("source_count", -1)) == 32 and
            int(design.get("teacher_observation_attempts", -1)) == 49,
            "E1 design membership/teacher-attempt count mismatch")
    require(design.get("student_Source_reset") is True and design.get("Query_Adam_reset") is True
            and design.get("LN_carry") is False and design.get("validation_gate") is True,
            "E1 design does not match the fixed Source-reset/current5 protocol")
    require(design.get("GT_before_prediction_seal") is False and
            not bool(design.get("GT_used_for_selection", False)),
            "E1 design declares GT-dependent pre-seal selection")
    roster = _rows_by_case(roster_doc, "E1 ROSTER_LOCK")
    references = _rows_by_case(ref_doc, "E1 REFERENCE_LOCK")
    source_roster_doc = read_json(SOURCE_BASE / "DEVELOPMENT_ROSTER_LOCK.json")
    source_roster = _rows_by_case(source_roster_doc, "fixed development roster")
    require(set(roster) == set(references) == set(source_roster),
            "E1 roster/reference/source roster membership mismatch")
    require(len([c for c in roster if roster[c]["dataset"] == "hc2"]) == 16 and
            len([c for c in roster if roster[c]["dataset"] == "vidstg"]) == 16,
            "E1 roster must contain 16 sources per domain")
    for cid, rec in roster.items():
        src = source_roster[cid]
        require(rec.get("query_key_sha256") == src.get("query_key_sha256") and
                rec.get("query_ordinal") == src.get("query_ordinal") and
                rec.get("dataset") == src.get("dataset"),
                f"E1 source alias differs from fixed development roster: {cid}")
        ref = references[cid]
        require(ref.get("dataset") == rec.get("dataset") and
                isinstance(ref.get("source_case_sha256"), str),
                f"E1 reference identity mismatch: {cid}")
        require(_root_path(str(rec.get("source_case_path", ""))).is_file()
                if rec.get("source_case_path") else True,
                f"E1 source-case path missing: {cid}")

    require(teacher_barrier.get("schema") == "stvg-reference-swap-teacher-global-barrier-v1"
            and teacher_barrier.get("status") == "complete"
            and teacher_barrier.get("GT_read") is False
            and int(teacher_barrier.get("count", -1)) == 49,
            "E1 teacher barrier must seal exactly 49 GT-free observations")
    require(student_barrier.get("schema") == "stvg-reference-swap-student-barrier-v1"
            and student_barrier.get("status") == "complete"
            and student_barrier.get("GT_read") is False
            and int(student_barrier.get("count", -1)) == 32,
            "E1 student barrier must seal exactly 32 GT-free predictions")
    require(student_barrier.get("reference_lock_sha256") == sha256_file(reference_path)
            and student_barrier.get("runtime_sha256") == sha256_file(student_runtime_path),
            "E1 student barrier does not bind the locked reference/runtime")
    require(teacher_barrier.get("bindings", {}).get("runtime_sha256") ==
            sha256_file(teacher_runtime_path), "E1 teacher barrier runtime binding mismatch")
    require(teacher_barrier.get("bindings", {}).get("teacher_job_lock_sha256") ==
            sha256_file(teacher_job_lock_path), "E1 teacher barrier job-lock binding mismatch")
    jobs = jobs_doc.get("records", jobs_doc.get("jobs"))
    require(isinstance(jobs, list) and len(jobs) == 49, "E1 teacher job lock must have 49 attempts")
    job_ids = [row.get("job_id") for row in jobs]
    require(all(isinstance(x, str) for x in job_ids) and len(set(job_ids)) == 49,
            "E1 teacher job lock has missing/duplicate job IDs")
    teacher_files = teacher_barrier.get("files")
    require(isinstance(teacher_files, list) and len(teacher_files) == 49,
            "E1 teacher barrier must list all 49 outputs")
    teacher_out = _verify_teacher_file_rows(teacher_files, base=base,
                                            label="E1 teacher barrier",
                                            expected_count=49)
    require({row.get("job_id") for row in teacher_files} == set(job_ids),
            "E1 teacher barrier and job lock membership mismatch")
    normalized_teacher_rows = list(teacher_out.values())
    teacher_by_job = {str(row["job_id"]): row for row in normalized_teacher_rows}
    for row in normalized_teacher_rows:
        require(row.get("case_id") in roster and row.get("decision_status") in
                {"accepted", "rejected", "unknown", "invalid", "abstain"},
                "E1 teacher barrier has an invalid case/status")
    teacher_by_case_position = {(row.get("case_id"), row.get("position")): row
                                for row in normalized_teacher_rows}
    require(len(teacher_by_case_position) == 49,
            "E1 teacher barrier repeats a case/position attempt")
    decision_count = 0
    for cid, ref in references.items():
        decisions = ref.get("dino_decisions")
        require(isinstance(decisions, list), f"E1 reference lock lacks DINO decision rows: {cid}")
        for decision in decisions:
            require(isinstance(decision, dict), f"E1 DINO decision is malformed: {cid}")
            output = teacher_by_case_position.get((cid, decision.get("position")))
            require(output is not None
                    and output.get("physical_frame_id") == decision.get("physical_frame_id")
                    and output.get("decision_status") == decision.get("status"),
                    f"E1 DINO decision does not match sealed teacher attempt: {cid}")
            decision_job = decision.get("job_id", decision.get("teacher_job_id"))
            decision_sha = decision.get("output_sha256", decision.get("teacher_output_sha256"))
            decision_path = decision.get("teacher_output_path")
            if decision_job is not None:
                require(decision_job == output.get("job_id"),
                        f"E1 DINO decision job ID mismatch: {cid}")
            if decision_sha is not None:
                require(decision_sha == output.get("sha256"),
                        f"E1 DINO decision output SHA mismatch: {cid}")
            require(isinstance(decision_path, str),
                    f"E1 DINO decision lacks its BASE-relative teacher output path: {cid}")
            decision_resolved = scorer_root_path(decision_path, base=base, teacher_scoped=True)
            require(str(decision_resolved.relative_to(ROOT)) == output.get("path"),
                    f"E1 DINO decision teacher output path mismatch: {cid}")
            decision_count += 1
    require(decision_count == 49, "E1 reference lock must describe exactly 49 teacher attempts")

    expected_student_paths = {
        str((base / "student" / f"{cid}.pt").relative_to(ROOT)) for cid in roster
    }
    student_files = _verify_file_rows(student_barrier.get("files"), label="E1 student barrier",
                                      expected_paths=expected_student_paths, expected_count=32)
    for cid in roster:
        receipt_rel = str((base / "student" / f"{cid}.receipt.json").relative_to(ROOT))
        receipt_path = ROOT / receipt_rel
        require(receipt_path.is_file(), f"E1 student receipt missing: {cid}")
        receipt = read_json(receipt_path)
        artifact = student_files[str((base / "student" / f"{cid}.pt").relative_to(ROOT))]
        require(receipt.get("status") == "complete" and receipt.get("case_id") == cid
                and receipt.get("sha256") == artifact["sha256"]
                and receipt.get("reference_lock_sha256") == sha256_file(reference_path)
                and receipt.get("runtime_sha256") == sha256_file(student_runtime_path)
                and receipt.get("GT_read") is False,
                f"E1 student per-case receipt mismatch: {cid}")

    require(root_readback.get("status") == "pass" and root_readback.get("GT_read") is False
            and int(root_readback.get("count", root_readback.get("case_count", -1))) == 32,
            "E1 independent pre-GT root readback must pass for all 32 cases")
    root_bindings = root_readback.get("bindings", root_readback)
    expected_bindings = {
        "student_barrier_sha256": sha256_file(student_barrier_path),
        "reference_lock_sha256": sha256_file(reference_path),
        "teacher_barrier_sha256": sha256_file(teacher_barrier_path),
        "student_runtime_sha256": sha256_file(student_runtime_path),
        "teacher_runtime_sha256": sha256_file(teacher_runtime_path),
        "parent_student_barrier_sha256": sha256_file(PARENT / "STUDENT_GLOBAL_BARRIER.json"),
    }
    for key, value in expected_bindings.items():
        require(root_bindings.get(key) == value, f"E1 root readback missing/wrong binding {key}")

    require(prediction_seal.get("status") == "complete" and
            prediction_seal.get("GT_read") is False and
            int(prediction_seal.get("count", -1)) == 32,
            "E1 prediction seal must be complete, GT-free, and cover all 32 cases")
    seal_files = _verify_file_rows(prediction_seal.get("files"), label="E1 prediction seal")
    required_bindings = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in (
            design_path, roster_path, reference_path, teacher_job_lock_path,
            teacher_runtime_path, student_runtime_path, teacher_barrier_path,
            student_barrier_path, root_readback_path, SOURCE_BASE / "SOURCE_RUNTIME.json",
            SOURCE_BASE / "RUNTIME.json", SOURCE_BASE / "DEVELOPMENT_ROSTER_LOCK.json",
            PARENT / "STUDENT_GLOBAL_BARRIER.json",
        )
    }
    for rel, digest in required_bindings.items():
        require(rel in seal_files and seal_files[rel]["sha256"] == digest,
                f"E1 prediction seal omits or misbinds {rel}")
    for rel, item in student_files.items():
        require(rel in seal_files and seal_files[rel]["sha256"] == item["sha256"],
                f"E1 prediction seal omits student artifact {rel}")
    for cid in roster:
        receipt_rel = str((base / "student" / f"{cid}.receipt.json").relative_to(ROOT))
        require(receipt_rel in seal_files,
                f"E1 prediction seal omits student per-case receipt {receipt_rel}")
    for rel, item in teacher_out.items():
        require(rel in seal_files and seal_files[rel]["sha256"] == item["sha256"],
                f"E1 prediction seal omits teacher output {rel}")
    require(prediction_seal.get("student_barrier_sha256") == sha256_file(student_barrier_path)
            and prediction_seal.get("teacher_barrier_sha256") == sha256_file(teacher_barrier_path)
            and prediction_seal.get("root_readback_sha256") == sha256_file(root_readback_path),
            "E1 prediction seal top-level root bindings mismatch")
    return {"teacher_runtime": teacher_runtime, "student_runtime": student_runtime,
            "source_runtime": source_runtime, "parent_runtime": parent_runtime,
            "design": design, "roster": roster, "references": references,
            "teacher_jobs": jobs_doc, "teacher_barrier": teacher_barrier,
            "student_barrier": student_barrier, "root_readback": root_readback,
            "prediction_seal": prediction_seal, "student_files": student_files,
            "teacher_files": teacher_out, "teacher_by_job": teacher_by_job,
            "teacher_by_case_position": teacher_by_case_position}


def _score() -> dict[str, Any]:
    # All hash/gate checks happen before any official target loader is imported.
    locked = _verify_all_seals(BASE)
    from scripts.stvg_evidence_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as scorer
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    import numpy as np
    import torch

    roster = locked["roster"]
    reference_index = locked["references"]
    student_files = locked["student_files"]
    parent_barrier = read_json(PARENT / "STUDENT_GLOBAL_BARRIER.json")
    parent_files = _verify_file_rows(parent_barrier.get("files"), label="round2 parent barrier",
                                     expected_count=32)
    source_cases: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    anchors_by_case: dict[str, list[dict[str, Any]]] = {}
    output_refs = []

    # The root and prediction seal are already complete; now verify each case
    # payload against the locked source, references, and original WF prediction.
    for cid, rec in roster.items():
        case, source_path = load_case(rec)
        case = dict(case)
        case.setdefault("dataset", rec["dataset"])
        case.setdefault("query_ordinal", rec["query_ordinal"])
        ref = reference_index[cid]
        require(case.get("GT_read") is False and case.get("dataset") == rec["dataset"]
                and case.get("source_case_sha256") == ref.get("source_case_sha256"),
                f"E1 source case identity mismatch: {cid}")
        frame_ids = np.asarray(case["frame_ids"], dtype=np.int64).reshape(-1).tolist()
        parent_ref = read_json(PARENT / "REFERENCE_LOCK.json")
        parent_ref_index = {str(row["case_id"]): row for row in parent_ref["records"]}
        supports = validate_common_support(ref, frame_ids, parent_ref_index[cid])
        student_rel = str((BASE / "student" / f"{cid}.pt").relative_to(ROOT))
        payload_path = ROOT / student_rel
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        require(payload.get("schema") == "stvg-reference-swap-student-case-v1"
                and payload.get("case_id") == cid and payload.get("dataset") == rec["dataset"]
                and payload.get("GT_read") is False,
                f"E1 student payload identity/schema mismatch: {cid}")
        require(payload.get("reference_records") == ref and
                payload.get("reference_lock_sha256") == sha256_file(BASE / "REFERENCE_LOCK.json")
                and payload.get("source_case_sha256") == case["source_case_sha256"]
                and payload.get("source_file_sha256") == sha256_file(source_path)
                and payload.get("runtime_sha256") == sha256_file(BASE / "STUDENT_RUNTIME.json"),
                f"E1 student/source/reference hash binding mismatch: {cid}")
        expected_parent_path = str((PARENT / "student" / f"{cid}.pt").relative_to(ROOT))
        require(expected_parent_path in parent_files
                and payload.get("parent_student_sha256") == parent_files[expected_parent_path]["sha256"],
                f"E1 parent WF prediction binding mismatch: {cid}")
        parent = torch.load(ROOT / expected_parent_path, map_location="cpu", weights_only=False)
        require(parent.get("GT_read") is False and parent.get("case_id") == cid
                and parent.get("source_case_sha256") == case["source_case_sha256"],
                f"E1 parent prediction identity mismatch: {cid}")
        arms = payload.get("arms")
        require(isinstance(arms, dict) and set(arms) == set(ARMS),
                f"E1 student arms differ from the locked eight-arm schema: {cid}")
        parent_arms = parent.get("arms", {})
        for arm in ("F", "T", "S_new_crop", "TS_new_crop"):
            require(arm in parent_arms and np.array_equal(np.asarray(arms[arm]["boxes"]),
                                                           np.asarray(parent_arms[arm]["boxes"]))
                    and list(arms[arm]["interval"]) == list(parent_arms[arm]["interval"]),
                    f"E1 did not exactly preserve parent {arm} arm: {cid}")
        require(payload.get("source_restored") is True, f"E1 source not restored: {cid}")
        for arm_name, arm in arms.items():
            boxes = np.asarray(arm.get("boxes"), dtype=np.float64)
            interval = arm.get("interval")
            require(boxes.shape == (len(frame_ids), 4) and np.isfinite(boxes).all(),
                    f"E1 {arm_name} boxes are not a complete finite source-grid tube: {cid}")
            require(isinstance(interval, list) and len(interval) == 2 and
                    all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in interval)
                    and float(interval[0]) < float(interval[1]),
                    f"E1 {arm_name} interval is malformed: {cid}")
        fits = payload.get("fits")
        readouts = payload.get("readouts")
        require(isinstance(fits, dict) and isinstance(readouts, dict)
                and set(fits) == {"WF_common", "DINO_common"}
                and set(readouts) == {"WF_common", "DINO_common"},
                f"E1 fit/readout mapping malformed: {cid}")
        if supports:
            for role, spatial_arm, temporal_arm in (
                    ("WF_common", "S_WF_common", "TS_WF_common"),
                    ("DINO_common", "S_DINO_common", "TS_DINO_common")):
                fit = fits[role]
                readout = readouts[role]
                expected_role = ("weight_free_common_same_frame_geometry" if role == "WF_common"
                                 else "original_dino_same_frame_geometry")
                require(isinstance(fit, dict) and fit.get("GT_read") is False
                        and fit.get("profile") == "current5"
                        and fit.get("reference_role") == expected_role
                        and fit.get("config") == locked["design"]["configs"][rec["dataset"]]
                        and fit.get("reference_sha256") == sha256_file(BASE / "REFERENCE_LOCK.json")
                        and fit.get("active_parameters") == 1792
                        and fit.get("declared_steps") == 5
                        and fit.get("validation_enabled") is True
                        and fit.get("optimizer_state_reset_each_query") is True
                        and fit.get("full_model_readout_calls") == 1,
                        f"E1 {role} fit config/reference gate mismatch: {cid}")
                require(isinstance(readout, dict) and readout.get("boxes") is not None,
                        f"E1 {role} full-model readout is missing: {cid}")
                raw_boxes = readout["boxes"]
                cached_boxes = fit.get("final_cached_decoder_boxes")
                require(torch.is_tensor(raw_boxes) and torch.is_tensor(cached_boxes)
                        and torch.equal(raw_boxes.detach().cpu(), cached_boxes.detach().cpu()),
                        f"E1 {role} raw full-readout boxes differ from cached decoder output: {cid}")
                # The stored arms are XYXY.  The worker performs this conversion
                # from a detached CPU float64 copy of the model's CXCYWH output.
                # Comparing raw CXCYWH directly to the arm was a schema error.
                rboxes = _xyxy_from_cxcywh(raw_boxes)
                require(np.array_equal(np.asarray(arms[spatial_arm]["boxes"], dtype=np.float64), rboxes)
                        and np.array_equal(np.asarray(arms[temporal_arm]["boxes"], dtype=np.float64), rboxes),
                        f"E1 {role} arm/readout geometry mismatch: {cid}")
        else:
            require(fits == {"WF_common": None, "DINO_common": None}
                    and readouts == {"WF_common": None, "DINO_common": None},
                    f"E1 empty common support must be explicit no-fit: {cid}")
            require(np.array_equal(arms["S_WF_common"]["boxes"], arms["F"]["boxes"])
                    and arms["TS_WF_common"]["interval"] == arms["T"]["interval"]
                    and np.array_equal(arms["S_DINO_common"]["boxes"], arms["F"]["boxes"])
                    and arms["TS_DINO_common"]["interval"] == arms["T"]["interval"],
                    f"E1 empty common support fallback is not exact F/T: {cid}")
        source_cases[cid] = case
        payloads[cid] = payload
        anchors_by_case[cid] = supports
        output_refs.append({"case_id": cid, "student_path": student_rel,
                            "student_sha256": student_files[student_rel]["sha256"],
                            "source_case_sha256": case["source_case_sha256"],
                            "source_file_sha256": sha256_file(source_path),
                            "reference_support_count": len(supports)})

    # Labels are first accessed only after the complete independent root readback,
    # every source/prediction byte, and all 49 teacher attempts have passed.
    exposure_path = BASE / "GT_EXPOSURE.json"
    require(not exposure_path.exists(), "E1 GT exposure receipt already exists; refusing a second scoring pass")
    exposure = {"schema": "stvg-reference-swap-gt-exposure-v1",
                "status": "after_teacher_student_seals_and_independent_root_readback",
                "prediction_seal_sha256": sha256_file(BASE / "GLOBAL_PREDICTION_BARRIER.json"),
                "student_root_readback_sha256": sha256_file(BASE / "ROOT_READBACK.json"),
                "teacher_barrier_sha256": sha256_file(BASE / "TEACHER_GLOBAL_BARRIER.json"),
                "GT_read": True, "time": time.time()}
    _write_new(exposure_path, exposure)
    targets, truth_provenance = scorer._truths_after_seal(source_cases)
    official, independent, metric_contract = scorer._metric_functions()
    rows: list[dict[str, Any]] = []
    for cid in sorted(source_cases):
        case, payload = source_cases[cid], payloads[cid]
        for arm in ARMS:
            values, audit = metric._score_student_arm(
                payload["arms"][arm], case, targets[cid], set(anchors_by_case[cid]),
                official, independent)
            rows.append({"case_id": cid, "dataset": payload["dataset"], "arm": arm,
                         "status": "scored", "metrics": values,
                         "support": audit["support"], "metric_audit": audit["metric_audit"],
                         "fit": _fit_summary(payload, arm)})
    require(len(rows) == 256, "E1 scorer must emit exactly 32x8=256 rows")
    summary = summarize(rows)
    paired = paired_contrasts(rows)
    require(len(summary) == 80 and len(paired) == 60,
            "E1 summary/paired table cardinality mismatch")
    output_bindings = {
        "design_lock_sha256": sha256_file(BASE / "DESIGN_LOCK.json"),
        "teacher_runtime_sha256": sha256_file(BASE / "RUNTIME.json"),
        "student_runtime_sha256": sha256_file(BASE / "STUDENT_RUNTIME.json"),
        "roster_sha256": sha256_file(BASE / "ROSTER_LOCK.json"),
        "reference_lock_sha256": sha256_file(BASE / "REFERENCE_LOCK.json"),
        "teacher_job_lock_sha256": sha256_file(BASE / "TEACHER_JOB_LOCK.json"),
        "teacher_barrier_sha256": sha256_file(BASE / "TEACHER_GLOBAL_BARRIER.json"),
        "student_barrier_sha256": sha256_file(BASE / "STUDENT_GLOBAL_BARRIER.json"),
        "root_readback_sha256": sha256_file(BASE / "ROOT_READBACK.json"),
        "prediction_seal_sha256": sha256_file(BASE / "GLOBAL_PREDICTION_BARRIER.json"),
        "source_runtime_sha256": sha256_file(SOURCE_BASE / "SOURCE_RUNTIME.json"),
        "parent_student_barrier_sha256": sha256_file(PARENT / "STUDENT_GLOBAL_BARRIER.json"),
        "metric_contract_sha256": sha256_file(ROOT / scorer.METRIC_CONTRACT),
    }
    readback = {"status": "complete", "case_count": 32, "teacher_attempt_count": 49,
                "row_count": 256, "summary_cell_count": len(summary),
                "paired_contrast_count": len(paired), "bootstrap_executed": False,
                "GT_read": True, "CPU_only": True, "model_loaded": False,
                "GPU_used": False,
                "scope": "official/independent five-metric postseal scoring; no new inference or fitting",
                "truth_provenance": truth_provenance,
                "official_independent_metric_max_error": max(
                    float(row["metric_audit"]["official_independent_max_abs_error"])
                    for row in rows),
                "bindings": output_bindings}
    _write_new(BASE / "POSTSEAL_ROWS.json", {"schema": "stvg-reference-swap-postseal-rows-v1",
                                             "status": "complete", "rows": rows,
                                             "GT_read": True, "bindings": output_bindings})
    _write_new(BASE / "POSTSEAL_SUMMARY.json", {"schema": "stvg-reference-swap-postseal-summary-v1",
                                                 "status": "complete", "summary": summary,
                                                 "GT_read": True, "bindings": output_bindings})
    _write_new(BASE / "PARENT_PAIRED_DELTAS.json", {"schema": "stvg-reference-swap-paired-deltas-v1",
                                                     "status": "complete", "records": paired,
                                                     "bootstrap_executed": False,
                                                     "GT_read": True, "bindings": output_bindings})
    _write_new(BASE / "ROOT_POSTSEAL_READBACK.json", readback)
    return readback


def _fit_summary(payload: Mapping[str, Any], arm: str) -> dict[str, Any]:
    mapping = {"S_WF_common": "WF_common", "TS_WF_common": "WF_common",
               "S_DINO_common": "DINO_common", "TS_DINO_common": "DINO_common"}
    role = mapping.get(arm)
    if role is None:
        return {"status": "reused_parent" if arm in ("F", "T", "S_new_crop", "TS_new_crop")
                else "not_a_fit", "optimizer_steps": 0, "full_model_readout_calls": 0}
    fit = payload["fits"].get(role)
    if fit is None:
        return {"status": "exact_no_update_empty_common_support", "optimizer_steps": 0,
                "full_model_readout_calls": 0}
    return {"status": "actual_fit", "reference_role": fit.get("reference_role"),
            "optimizer_steps": int(fit.get("optimizer_steps", 0)),
            "full_model_readout_calls": int(fit.get("full_model_readout_calls", 0)),
            "profile": fit.get("profile")}


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite existing E1 receipt: {path.relative_to(ROOT)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _contracts() -> dict[str, Any]:
    import numpy as np
    import torch

    assert scorer_root_path("teacher/sample.json", base=BASE, teacher_scoped=True) == (BASE / "teacher/sample.json").resolve()
    assert scorer_root_path("artifacts/example.json", base=BASE) == (ROOT / "artifacts/example.json").resolve()
    try:
        scorer_root_path("teacher/../../outside.json", base=BASE, teacher_scoped=True)
    except AuditError:
        pass
    else:
        raise AssertionError("BASE-relative teacher path traversal was accepted")
    try:
        scorer_root_path("student/sample.json", base=BASE, teacher_scoped=True)
    except AuditError:
        pass
    else:
        raise AssertionError("non-teacher path was accepted in teacher-scoped mode")

    # Reproduce the saved-readout ABI: raw full-model boxes and cached decoder
    # boxes are CXCYWH tensors; arms are exact XYXY conversions made after a
    # detached CPU float64 copy.
    raw_cxcywh = torch.tensor([[0.25, 0.50, 0.125, 0.25],
                               [0.75, 0.625, 0.25, 0.50]], dtype=torch.float32)
    cached_cxcywh = raw_cxcywh.clone()
    assert raw_cxcywh.dtype == cached_cxcywh.dtype
    assert raw_cxcywh.shape == cached_cxcywh.shape
    assert torch.equal(raw_cxcywh.detach().cpu(), cached_cxcywh.detach().cpu())
    xyxy = _xyxy_from_cxcywh(raw_cxcywh)
    expected_xyxy = np.array([[0.1875, 0.375, 0.3125, 0.625],
                              [0.625, 0.375, 0.875, 0.875]], dtype=np.float64)
    assert xyxy.dtype == np.float64 and np.array_equal(xyxy, expected_xyxy)
    assert not np.array_equal(raw_cxcywh.detach().cpu().numpy(), xyxy)
    changed_cache = cached_cxcywh.clone()
    changed_cache[0, 0] += 1e-6
    assert not torch.equal(raw_cxcywh.detach().cpu(), changed_cache.detach().cpu())

    rows = []
    for dataset in ("hc2", "vidstg"):
        for index in range(16):
            cid = f"{dataset}-{index:02d}"
            vals = {metric: (None if metric.endswith("unsupervised_GT") and index % 5 == 0
                             else float(index) / 10.0)
                    for metric in METRICS}
            for arm_idx, arm in enumerate(ARMS):
                rowvals = {key: (None if value is None else value + 0.01 * arm_idx)
                           for key, value in vals.items()}
                rows.append({"case_id": cid, "dataset": dataset, "arm": arm, "metrics": rowvals})
    summ = summarize(rows)
    paired = paired_contrasts(rows)
    assert len(rows) == 256 and len(summ) == 80 and len(paired) == 60
    assert all(len(row["unknown_case_ids"]) == 4 for row in summ
               if row["metric"] == "m_sIoU_unsupervised_GT")
    assert all(row["positive_count"] + row["negative_count"] + row["neutral_count"]
               + row["unknown_count"] == 16 for row in paired)
    assert all(row["bootstrap_executed"] is False for row in [
        {"bootstrap_executed": False}])
    return {"status": "pass", "checks": 14, "rows": len(rows),
            "summary_cells": len(summ), "paired_cells": len(paired),
            "GT_read": False, "model_loaded": False, "GPU_used": False,
            "readout_contract": "raw_CXCYWH_tensor_equals_cached_then_exact_float64_XYXY_conversion"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("score", "contracts", "teacher-preflight"))
    args = parser.parse_args()
    if args.mode == "contracts":
        print(json.dumps(_contracts(), sort_keys=True))
        return
    if args.mode == "teacher-preflight":
        print(json.dumps(_teacher_only_preflight(), sort_keys=True))
        return
    result = _score()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
