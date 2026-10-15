#!/usr/bin/env python3
"""Independently seal the fixed-pool CRC labels into geometry-only references.

This is a CPU-only reader. It never imports a model runtime, GT reader, or media
decoder. Teacher text is parsed in memory and is never copied to an output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.stvg_crc_common_v1 import BASE, PARENT, ROOT, read, sha, verify_runtime, write
from scripts.stvg_crc_protocol_v1 import canonical_sha256, parse_output


def _canonical_sha(value: Any) -> str:
    return canonical_sha256(value)


def _by_case(rows: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        cid = row.get("case_id")
        if not isinstance(cid, str) or not cid or cid in out:
            raise ValueError("case records must have unique nonempty case_id values")
        out[cid] = row
    return out


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _file_index(rows: Any, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} files must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for item in rows:
        if not isinstance(item, Mapping):
            raise ValueError(f"malformed {label} file row")
        path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str) or path in result:
            raise ValueError(f"malformed or duplicate {label} path")
        result[path] = item
    return result


def _verify_file_row(row: Mapping[str, Any], *, label: str) -> Path:
    path_text = row.get("path")
    digest = row.get("sha256")
    if not isinstance(path_text, str) or not isinstance(digest, str):
        raise ValueError(f"{label} row lacks path/SHA-256")
    path = ROOT / path_text
    if not path.is_file() or sha(path) != digest:
        raise RuntimeError(f"{label} byte binding failed: {path_text}")
    if "bytes" in row and path.stat().st_size != int(row["bytes"]):
        raise RuntimeError(f"{label} byte count failed: {path_text}")
    return path


def _xyxy_to_cxcywh(box: Any) -> list[float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("candidate center box must be normalized xyxy")
    x0, y0, x1, y1 = [float(x) for x in box]
    if not all(math.isfinite(x) for x in (x0, y0, x1, y1)):
        raise ValueError("nonfinite candidate geometry")
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("illegal candidate geometry")
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def _same_box(left: Any, right: Any, tol: float = 1e-12) -> bool:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    return len(left) == len(right) == 4 and all(
        math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tol
        for a, b in zip(left, right)
    )


def _validate_input_reads(input_reads: Any, videos: Any, job_id: str) -> None:
    """Check the teacher writer's ordered two-array readback against its job."""
    if not isinstance(videos, list) or len(videos) != 2:
        raise ValueError("locked pair must have exactly two video specs")
    if not isinstance(input_reads, list) or len(input_reads) != len(videos):
        raise RuntimeError(f"CRC response must record both video reads: {job_id}")
    for read_row, video in zip(input_reads, videos):
        if not isinstance(read_row, Mapping):
            raise RuntimeError(f"malformed CRC input-read row: {job_id}")
        expected_read = {
            "path": video.get("path"),
            "file_sha256": video.get("sha256"),
            "tensor_sha256": video.get("tensor_sha256"),
            "file_bytes": video.get("bytes"),
        }
        for field, expected in expected_read.items():
            if field not in read_row or read_row[field] != expected:
                raise RuntimeError(f"CRC input-read/job mismatch {job_id}: {field}")
        shape = video.get("shape")
        if (not isinstance(shape, list) or not shape or video.get("dtype") != "uint8"
                or math.prod(int(dim) for dim in shape) <= 0):
            raise RuntimeError(f"CRC locked video tensor shape/dtype is invalid: {job_id}")
        if read_row.get("tensor_bytes") != math.prod(int(dim) for dim in shape):
            raise RuntimeError(f"CRC input-read tensor byte count differs from locked shape: {job_id}")
        if not isinstance(read_row.get("cache_hit"), bool):
            raise RuntimeError(f"CRC input-read cache metadata is malformed: {job_id}")


def _slot_anchor(slot: Mapping[str, Any], job: Mapping[str, Any],
                 parent_anchor: Mapping[str, Any], response_sha: str) -> dict[str, Any]:
    pair = slot.get("pair_job")
    if not isinstance(pair, Mapping):
        raise ValueError("Alternative decision lacks its locked pair job")
    candidate_id = int(slot["parent_selected_candidate_id"])
    if candidate_id != int(pair["alternative_candidate_id"]):
        raise ValueError("pair Alternative ID differs from frozen parent Top1 ID")
    position = int(slot["source_grid_position"])
    physical_id = int(slot["center_physical_frame_id"])
    if position != int(pair["source_grid_position"]) or physical_id != int(pair["center_physical_frame_id"]):
        raise ValueError("pair and slot center coordinates differ")
    box_xyxy = pair.get("alternative_box_xyxy_normalized")
    box = _xyxy_to_cxcywh(box_xyxy)
    if (int(parent_anchor.get("position", -1)) != position
            or int(parent_anchor.get("physical_frame_id", -1)) != physical_id
            or int(parent_anchor.get("candidate_id", -1)) != candidate_id
            or not _same_box(parent_anchor.get("box_cxcywh"), box)):
        raise ValueError("Alternative geometry does not exactly match the frozen Top1 reference")
    weight = float(slot["parent_presence"])
    if not math.isfinite(weight) or weight <= 0 or abs(weight - float(parent_anchor.get("weight", -1))) > 1e-12:
        raise ValueError("pair reliability differs from the original Top1 reference")
    return {
        "position": position,
        "physical_frame_id": physical_id,
        "box_cxcywh": box,
        "weight": weight,
        "candidate_id": candidate_id,
        "pair_job_id": pair["job_id"],
        "teacher_output_sha256": response_sha,
        "original_top1_source_sha256": slot["parent_response_sha256"],
    }


def build_readback_and_lock(*, write_outputs: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate all 128 parent slots and all 49 CRC responses, then build refs."""
    runtime = verify_runtime()
    teacher_dir = BASE / "teacher"
    lock_path = BASE / "inputs_teacher" / "TEACHER_JOB_LOCK.json"
    input_path = BASE / "inputs_teacher" / "INPUT_BARRIER.json"
    custody_path = BASE / "inputs_teacher" / "PARENT_TOP1_CUSTODY.json"
    for path in (lock_path, input_path, custody_path, BASE / "TEACHER_RUNTIME.json",
                 BASE / "TEACHER_GLOBAL_BARRIER.json", PARENT / "TEACHER_JOB_LOCK.json",
                 PARENT / "TEACHER_GLOBAL_BARRIER.json", PARENT / "CANDIDATE_INPUT_BARRIER.json",
                 PARENT / "REFERENCE_LOCK.json", PARENT / "STUDENT_GLOBAL_BARRIER.json"):
        if not path.is_file():
            raise FileNotFoundError(path)

    lock = read(lock_path)
    inputs = read(input_path)
    custody = read(custody_path)
    barrier = read(BASE / "TEACHER_GLOBAL_BARRIER.json")
    parent_jobs_doc = read(PARENT / "TEACHER_JOB_LOCK.json")
    parent_teacher_barrier = read(PARENT / "TEACHER_GLOBAL_BARRIER.json")
    parent_input_barrier = read(PARENT / "CANDIDATE_INPUT_BARRIER.json")
    parent_refs_doc = read(PARENT / "REFERENCE_LOCK.json")
    parent_student_barrier = read(PARENT / "STUDENT_GLOBAL_BARRIER.json")
    parent_teacher_runtime_path = PARENT / "TEACHER_RUNTIME.json"

    for label, doc in (("CRC job lock", lock), ("CRC input barrier", inputs),
                       ("parent top1 custody", custody), ("CRC teacher barrier", barrier),
                       ("parent teacher lock", parent_jobs_doc), ("parent teacher barrier", parent_teacher_barrier),
                       ("parent input barrier", parent_input_barrier), ("parent reference lock", parent_refs_doc),
                       ("parent student barrier", parent_student_barrier)):
        if doc.get("GT_read") is not False:
            raise RuntimeError(f"{label} must explicitly declare GT_read=false")
    if lock.get("status") != "locked" or lock.get("count") != 128 or len(lock.get("records", [])) != 128:
        raise RuntimeError("CRC job lock must contain exactly 128 locked slot rows")
    if lock.get("pair_job_count") != 49:
        raise RuntimeError("CRC job lock must contain exactly 49 pair jobs")
    if inputs.get("status") not in {"complete", "sealed"}:
        raise RuntimeError("CRC input barrier is not complete")
    if barrier.get("status") != "complete" or barrier.get("count") != 49:
        raise RuntimeError("CRC teacher barrier must seal exactly 49 responses")
    if parent_teacher_barrier.get("status") != "complete" or parent_teacher_barrier.get("count") != 512:
        raise RuntimeError("prompt009 Top1 parent teacher barrier is not complete")
    if parent_input_barrier.get("status") != "complete" or parent_input_barrier.get("teacher_job_count") != 512:
        raise RuntimeError("prompt009 parent input barrier is not complete")
    if parent_student_barrier.get("status") != "complete" or parent_student_barrier.get("count") != 32:
        raise RuntimeError("prompt009 parent student barrier is not complete")
    if runtime.get("teacher_runtime_sha256") not in (None, sha(BASE / "TEACHER_RUNTIME.json")):
        raise RuntimeError("CRC runtime does not bind the exact teacher runtime")

    # Verify every parent lock/runtime/barrier binding before consulting labels.
    parent_runtime = read(parent_teacher_runtime_path)
    if parent_runtime.get("status") != "locked" or parent_runtime.get("GT_read") is not False:
        raise RuntimeError("prompt009 teacher runtime is not locked and explicitly GT-free")
    for name, expected, actual in (
        ("parent job lock", parent_teacher_barrier.get("job_lock_sha256"), sha(PARENT / "TEACHER_JOB_LOCK.json")),
        ("parent input barrier", parent_teacher_barrier.get("input_barrier_sha256"), sha(PARENT / "CANDIDATE_INPUT_BARRIER.json")),
        ("parent runtime", parent_teacher_barrier.get("runtime_sha256"), sha(parent_teacher_runtime_path)),
    ):
        if expected != actual:
            raise RuntimeError(f"prompt009 {name} hash does not match its sealed barrier")

    expected_crc_bindings = {
        "job_lock_sha256": sha(lock_path),
        "input_barrier_sha256": sha(input_path),
        "parent_top1_custody_sha256": sha(custody_path),
        "runtime_sha256": sha(BASE / "TEACHER_RUNTIME.json"),
    }
    for field, expected in expected_crc_bindings.items():
        actual = barrier.get(field)
        if actual != expected:
            raise RuntimeError(f"CRC teacher barrier does not bind {field}")

    lock_bindings = lock.get("bindings", {})
    input_bindings = inputs.get("bindings", {})
    for field, expected in (
        ("parent_job_lock_sha256", sha(PARENT / "TEACHER_JOB_LOCK.json")),
        ("parent_teacher_barrier_sha256", sha(PARENT / "TEACHER_GLOBAL_BARRIER.json")),
        ("parent_input_barrier_sha256", sha(PARENT / "CANDIDATE_INPUT_BARRIER.json")),
        ("parent_reference_lock_sha256", sha(PARENT / "REFERENCE_LOCK.json")),
        ("parent_student_barrier_sha256", sha(PARENT / "STUDENT_GLOBAL_BARRIER.json")),
        ("parent_teacher_runtime_sha256", sha(parent_teacher_runtime_path)),
    ):
        for binding_doc, label in ((lock_bindings, "job lock"), (input_bindings, "input barrier")):
            if field in binding_doc and binding_doc[field] != expected:
                raise RuntimeError(f"CRC {label} has a stale {field}")
    for field, expected in (("input_barrier_sha256", sha(input_path)),
                            ("job_lock_sha256", sha(lock_path)),
                            ("parent_top1_custody_sha256", sha(custody_path))):
        value = barrier.get(field)
        if value != expected:
            raise RuntimeError(f"CRC teacher barrier has stale {field}")

    roster_rows = read(BASE / "ROSTER_LOCK.json").get("records", [])
    roster_index = _by_case(roster_rows)
    if len(roster_index) != 32:
        raise RuntimeError("CRC roster must contain exactly 32 cases")
    parent_jobs = parent_jobs_doc.get("records", [])
    parent_job_index = {j.get("job_id"): j for j in parent_jobs if isinstance(j, Mapping)}
    if len(parent_job_index) != 512:
        raise RuntimeError("prompt009 parent lock must contain 512 unique jobs")
    parent_output_index = _file_index(parent_teacher_barrier.get("files"), "parent teacher barrier")
    parent_input_index = _file_index(parent_input_barrier.get("files"), "parent input barrier")
    parent_ref_index = _by_case(parent_refs_doc.get("records", []))
    parent_student_index = _file_index(parent_student_barrier.get("files"), "parent student barrier")
    parent_response_custody = custody.get("records")
    if not isinstance(parent_response_custody, list) or len(parent_response_custody) != 128:
        raise RuntimeError("PARENT_TOP1_CUSTODY must retain all 128 original prompt009 slots")
    custody_by_job = {row.get("parent_job_id"): row for row in parent_response_custody if isinstance(row, Mapping)}
    if len(custody_by_job) != 128:
        raise RuntimeError("parent Top1 custody must have 128 unique jobs")

    parent_input_verified: set[str] = set()
    parent_response_verified: set[str] = set()
    pair_jobs: dict[str, Mapping[str, Any]] = {}
    decisions_by_case: dict[str, list[dict[str, Any]]] = {cid: [] for cid in roster_index}
    anchors_by_case: dict[str, list[dict[str, Any]]] = {cid: [] for cid in roster_index}
    teacher_files = _file_index(barrier.get("files"), "CRC teacher barrier")
    expected_pair_ids: set[str] = set()
    labels = {"Source": 0, "Alternative": 0, "Uncertain": 0, "invalid": 0, "no_pair": 0}

    # Hash every unique CRC input array through the sealed file list, without loading arrays.
    crc_input_files = _file_index(inputs.get("files"), "CRC input barrier")
    for rel, item in crc_input_files.items():
        _verify_file_row(item, label="CRC input")

    for slot in lock["records"]:
        if not isinstance(slot, Mapping):
            raise ValueError("malformed CRC slot row")
        cid = str(slot["case_id"])
        if cid not in roster_index:
            raise ValueError("CRC slot references a case outside its locked roster")
        roster_row = roster_index[cid]
        for field in ("dataset", "source_case_sha256"):
            if slot.get(field) != roster_row.get(field):
                raise RuntimeError(f"CRC slot/roster mismatch for {cid}: {field}")
        parent_job_id = slot.get("parent_job_id")
        parent_job = parent_job_index.get(parent_job_id)
        if parent_job is None:
            raise RuntimeError(f"missing prompt009 parent job for {cid}")
        if (parent_job.get("case_id") != cid or parent_job.get("variant") != "new_crop"
                or int(parent_job.get("center_slot", -1)) != int(slot["center_slot"])
                or int(parent_job.get("center_physical_frame_id", -1)) != int(slot["center_physical_frame_id"])
                or parent_job.get("source_case_sha256") != slot.get("source_case_sha256")
                or parent_job.get("source_request_sha256") != slot.get("source_request_sha256")):
            raise RuntimeError(f"prompt009 parent job binding differs for {parent_job_id}")
        if _canonical_sha(parent_job) != slot.get("parent_job_canonical_sha256"):
            raise RuntimeError(f"prompt009 parent job canonical hash mismatch for {parent_job_id}")

        parent_response_path = str(slot["parent_response_path"])
        parent_response_sha = str(slot["parent_response_sha256"])
        parent_item = parent_output_index.get(parent_response_path)
        if parent_item is None or parent_item.get("sha256") != parent_response_sha:
            raise RuntimeError(f"prompt009 output is not sealed for {parent_job_id}")
        parent_response = _verify_file_row(parent_item, label="prompt009 teacher output")
        parent_response_verified.add(parent_response_path)
        parent_output = read(parent_response)
        if parent_output.get("job_id") != parent_job_id or parent_output.get("GT_read") is not False:
            raise RuntimeError("prompt009 response identity/GT marker mismatch")
        parsed_parent = parent_output.get("parsed")
        if not isinstance(parsed_parent, Mapping):
            raise RuntimeError("prompt009 response lacks parsed Top1 output")
        selected = parsed_parent.get("selected_candidate_id")
        if (parsed_parent.get("status") != slot.get("parent_parsed_status")
                or selected != slot.get("parent_selected_candidate_id")):
            raise RuntimeError(f"prompt009 Top1 parse differs from custody for {parent_job_id}")
        custody_row = custody_by_job.get(parent_job_id)
        if custody_row is None or custody_row.get("response_sha256") != parent_response_sha:
            raise RuntimeError(f"prompt009 custody row mismatch for {parent_job_id}")
        if custody_row.get("selected_candidate_id") != selected:
            raise RuntimeError(f"prompt009 custody selected ID mismatch for {parent_job_id}")

        # The complete parent NPY custody is checked, then each pair reuses the exact two input arrays.
        for video in parent_job.get("videos", []):
            path = video.get("path")
            input_item = parent_input_index.get(path)
            if input_item is None or input_item.get("sha256") != video.get("sha256"):
                raise RuntimeError(f"prompt009 parent input is unsealed for {parent_job_id}")
            _verify_file_row(input_item, label="prompt009 parent input")
            parent_input_verified.add(path)

        pair = slot.get("pair_job")
        if slot.get("pair_status") == "no_pair":
            if pair is not None:
                raise RuntimeError("no_pair slot must not contain a teacher job")
            labels["no_pair"] += 1
            decisions_by_case[cid].append({
                "center_slot": int(slot["center_slot"]),
                "physical_frame_id": int(slot["center_physical_frame_id"]),
                "source_grid_position": int(slot["source_grid_position"]),
                "pair_status": "no_pair",
                "reason": slot.get("no_pair_reason"),
                "teacher_label": None,
                "teacher_output_sha256": None,
                "original_top1_source_sha256": parent_response_sha,
            })
            continue
        if slot.get("pair_status") != "locked_pair" or not isinstance(pair, Mapping):
            raise RuntimeError("locked_pair slot lacks its nested pair job")
        pair_id = pair.get("job_id")
        if not isinstance(pair_id, str) or pair_id in expected_pair_ids:
            raise RuntimeError("CRC pair job IDs must be unique")
        expected_pair_ids.add(pair_id)
        pair_jobs[pair_id] = pair
        if (int(pair.get("source_candidate_id", -1)) != 1
                or int(pair.get("alternative_candidate_id", -1)) != int(selected)
                or int(slot.get("parent_selected_candidate_id", -1)) != int(selected)
                or int(pair.get("center_physical_frame_id", -1)) != int(slot["center_physical_frame_id"])
                or int(pair.get("source_grid_position", -1)) != int(slot["source_grid_position"])
                or pair.get("source_case_sha256") != slot.get("source_case_sha256")
                or pair.get("source_request_sha256") != slot.get("source_request_sha256")):
            raise RuntimeError(f"CRC job does not match its prompt009 parent slot: {pair_id}")
        stripped = dict(pair)
        pair_hash = stripped.pop("pair_binding_sha256", None)
        if _canonical_sha(stripped) != pair_hash:
            raise RuntimeError(f"CRC pair job canonical hash mismatch: {pair_id}")
        parent_bindings = {int(x["id"]): x for x in parent_job.get("candidate_bindings", [])}
        top1_binding = parent_bindings.get(int(selected))
        if top1_binding is None or top1_binding.get("role") == "Source":
            raise RuntimeError("pairable original Top1 candidate is absent or is Source")
        if abs(float(top1_binding.get("presence", -1)) - float(slot.get("parent_presence", -1))) > 1e-12:
            raise RuntimeError("locked parent presence differs from original candidate pool")
        if int(slot["source_grid_position"]) < 0:
            raise RuntimeError("negative source-grid position")
        expected_xyxy = top1_binding.get("center_box")
        if not _same_box(pair.get("alternative_box_xyxy_normalized"), expected_xyxy):
            raise RuntimeError("CRC alternative box differs from the frozen prompt009 candidate pool")
        videos = pair.get("videos")
        if not isinstance(videos, list) or len(videos) != 2:
            raise RuntimeError("CRC pair must reuse exactly two sealed parent video inputs")
        for video in videos:
            match = next((v for v in parent_job.get("videos", [])
                          if v.get("path") == video.get("path") and v.get("sha256") == video.get("sha256")), None)
            if match is None:
                raise RuntimeError("CRC pair video differs from exact prompt009 parent input")
            if video.get("tensor_sha256") != match.get("tensor_sha256"):
                raise RuntimeError("CRC pair tensor identity differs from prompt009 parent input")

        response_item = teacher_files.get(str((BASE / "teacher" / f"{pair_id}.json").relative_to(ROOT)))
        if response_item is None or response_item.get("job_id") != pair_id:
            raise RuntimeError(f"CRC output missing from global barrier: {pair_id}")
        response_path = _verify_file_row(response_item, label="CRC teacher output")
        response = read(response_path)
        response_bindings = (
            ("schema", "stvg-crc-teacher-response-v1"),
            ("job_id", pair_id),
            ("case_id", cid),
            ("dataset", slot.get("dataset")),
            ("center_slot", slot.get("center_slot")),
            ("source_grid_position", slot.get("source_grid_position")),
            ("source_candidate_id", 1),
            ("alternative_candidate_id", int(selected)),
            ("parent_job_id", slot.get("parent_job_id")),
            ("parent_response_sha256", slot.get("parent_response_sha256")),
            ("parent_candidate_id", int(selected)),
            ("source_case_sha256", slot.get("source_case_sha256")),
            ("source_request_sha256", slot.get("source_request_sha256")),
            ("query_key_sha256", pair.get("query_key_sha256")),
            ("query_text_sha256", pair.get("query_text_sha256")),
            ("pair_binding_sha256", pair.get("pair_binding_sha256")),
            ("prompt_sha256", pair.get("prompt_sha256")),
            ("input_barrier_sha256", sha(input_path)),
            ("runtime_sha256", expected_crc_bindings["runtime_sha256"]),
        )
        for field, expected in response_bindings:
            if field not in response or response[field] != expected:
                raise RuntimeError(f"CRC response binding mismatch {pair_id}: {field}")
        if response.get("status") != "complete" or response.get("GT_read") is not False:
            raise RuntimeError(f"CRC response status/GT marker mismatch: {pair_id}")
        if response.get("teacher_frozen") is not True:
            raise RuntimeError(f"CRC response does not certify frozen teacher: {pair_id}")

        # The actual writer records the two arrays it read at top level. Match
        # those ordered receipts to the exact NPY specs in this locked job;
        # do not assume a wrapper object that the writer does not emit.
        _validate_input_reads(response.get("input_reads"), videos, pair_id)

        generation = response.get("generation")
        if not isinstance(generation, Mapping) or generation != read(BASE / "TEACHER_RUNTIME.json").get("generation"):
            raise RuntimeError(f"CRC response generation config differs from locked runtime: {pair_id}")
        if response.get("output_token_cap") != generation.get("max_new_tokens"):
            raise RuntimeError(f"CRC response token cap differs from locked generation: {pair_id}")
        output_tokens = response.get("output_tokens")
        if not isinstance(output_tokens, int) or output_tokens < 0 or output_tokens > int(response["output_token_cap"]):
            raise RuntimeError(f"CRC response output token count is invalid: {pair_id}")
        truncated = response.get("output_truncated_at_cap")
        if not isinstance(truncated, bool) or truncated != (output_tokens >= int(response["output_token_cap"])):
            raise RuntimeError(f"CRC response truncation marker disagrees with output length: {pair_id}")
        if response.get("generation_call_count") != 1:
            raise RuntimeError(f"CRC response generation call count is not one: {pair_id}")
        raw_output = response.get("raw_output")
        if not isinstance(raw_output, str) or response.get("raw_output_sha256") != hashlib.sha256(raw_output.encode("utf-8")).hexdigest():
            raise RuntimeError(f"CRC response raw-output hash mismatch: {pair_id}")
        parsed = parse_output(raw_output, pair, output_truncated_at_cap=truncated)
        if response.get("parsed") != parsed:
            raise RuntimeError(f"CRC parser replay differs from saved parse: {pair_id}")
        barrier_label = response_item.get("label")
        if response_item.get("parsed_status") != parsed["status"] or barrier_label != parsed["label"]:
            raise RuntimeError(f"CRC global barrier parse mismatch: {pair_id}")
        if parsed["status"] == "valid":
            labels[str(parsed["label"])] += 1
            if parsed["label"] == "Alternative":
                original_ref = parent_ref_index[cid].get("new_crop", {}).get("anchors", [])
                match = [a for a in original_ref if int(a.get("position", -1)) == int(slot["source_grid_position"])
                         and int(a.get("physical_frame_id", -1)) == int(slot["center_physical_frame_id"])
                         and int(a.get("candidate_id", -1)) == int(selected)]
                if len(match) != 1:
                    raise RuntimeError("Alternative requires one exact original Top1 reference anchor")
                anchor = _slot_anchor(slot, pair, match[0], sha(response_path))
                anchors_by_case[cid].append(anchor)
                reason = "alternative_selected_locked_original_candidate"
            elif parsed["label"] == "Source":
                reason = "source_selected_no_pair_reference"
            else:
                reason = "uncertain_no_pair_reference"
        else:
            labels["invalid"] += 1
            reason = "invalid_response_no_pair_reference"
        decisions_by_case[cid].append({
            "center_slot": int(slot["center_slot"]),
            "physical_frame_id": int(slot["center_physical_frame_id"]),
            "source_grid_position": int(slot["source_grid_position"]),
            "pair_status": "locked_pair",
            "label_status": parsed["status"],
            "label": parsed["label"],
            "reason": reason,
            "pair_job_id": pair_id,
            "teacher_output_sha256": sha(response_path),
            "original_top1_source_sha256": parent_response_sha,
        })

    if len(expected_pair_ids) != 49 or set(teacher_files) != {
            str((BASE / "teacher" / f"{pair_id}.json").relative_to(ROOT)) for pair_id in expected_pair_ids}:
        raise RuntimeError("CRC response set differs from the 49 locked pair jobs")
    if len(parent_response_verified) != 128:
        raise RuntimeError("not all 128 prompt009 parent Top1 outputs were byte-verified")
    expected_npy_paths = {str(video["path"]) for slot in lock["records"]
                          for video in (slot.get("pair_job") or {}).get("videos", [])}
    # Pairable input arrays must all be present in both parent custody and the CRC input barrier.
    for rel in expected_npy_paths:
        if rel not in parent_input_index or rel not in crc_input_files:
            raise RuntimeError("paired NPY input missing from a sealed input barrier")
    if len(decisions_by_case) != 32 or any(len(rows) != 4 for rows in decisions_by_case.values()):
        raise RuntimeError("each locked source must retain all four original slots")
    for cid, rows in decisions_by_case.items():
        rows.sort(key=lambda row: row["center_slot"])
        if [row["center_slot"] for row in rows] != [0, 1, 2, 3]:
            raise RuntimeError(f"missing or duplicate slot for {cid}")
        anchors_by_case[cid].sort(key=lambda row: row["position"])
        positions = [row["position"] for row in anchors_by_case[cid]]
        if len(positions) != len(set(positions)):
            raise RuntimeError(f"duplicate Pair anchor position for {cid}")

    ref_records = []
    for cid, roster_row in sorted(roster_index.items()):
        ref_records.append({
            "case_id": cid,
            "dataset": roster_row["dataset"],
            "source_case_sha256": roster_row["source_case_sha256"],
            "Pair": {"anchors": anchors_by_case[cid], "decisions": decisions_by_case[cid]},
            "GT_read": False,
        })
    reference_lock = {
        "schema": "stvg-crc-reference-lock-v1",
        "status": "locked",
        "count": 32,
        "slot_count": 128,
        "pair_job_count": 49,
        "records": ref_records,
        "job_lock_sha256": sha(lock_path),
        "input_barrier_sha256": sha(input_path),
        "parent_top1_custody_sha256": sha(custody_path),
        "teacher_barrier_sha256": sha(BASE / "TEACHER_GLOBAL_BARRIER.json"),
        "parent_teacher_lock_sha256": sha(PARENT / "TEACHER_JOB_LOCK.json"),
        "parent_teacher_barrier_sha256": sha(PARENT / "TEACHER_GLOBAL_BARRIER.json"),
        "parent_input_barrier_sha256": sha(PARENT / "CANDIDATE_INPUT_BARRIER.json"),
        "parent_reference_lock_sha256": sha(PARENT / "REFERENCE_LOCK.json"),
        "parent_student_barrier_sha256": sha(PARENT / "STUDENT_GLOBAL_BARRIER.json"),
        "parent_teacher_runtime_sha256": sha(parent_teacher_runtime_path),
        "GT_read": False,
    }
    readback = {
        "schema": "stvg-crc-root-teacher-output-readback-v1",
        "status": "pass",
        "slot_count": 128,
        "pair_job_count": 49,
        "parent_top1_outputs_byte_verified": len(parent_response_verified),
        "parent_input_arrays_byte_verified": len(parent_input_verified),
        "crc_outputs_byte_verified": len(teacher_files),
        "strict_parser_replay_count": len(teacher_files),
        "label_counts": labels,
        "alternative_anchor_count": sum(len(v) for v in anchors_by_case.values()),
        "bindings": {
            **expected_crc_bindings,
            "parent_job_lock_sha256": sha(PARENT / "TEACHER_JOB_LOCK.json"),
            "parent_teacher_barrier_sha256": sha(PARENT / "TEACHER_GLOBAL_BARRIER.json"),
            "parent_input_barrier_sha256": sha(PARENT / "CANDIDATE_INPUT_BARRIER.json"),
            "parent_reference_lock_sha256": sha(PARENT / "REFERENCE_LOCK.json"),
            "parent_student_barrier_sha256": sha(PARENT / "STUDENT_GLOBAL_BARRIER.json"),
            "parent_teacher_runtime_sha256": sha(parent_teacher_runtime_path),
        },
        "GT_read": False,
        "CPU_only": True,
        "raw_teacher_text_exported": False,
        "queries_exported": False,
        "scope": "byte custody and exact-label replay only; no semantic accuracy or GT claim",
    }
    if write_outputs:
        write(BASE / "ROOT_TEACHER_OUTPUT_READBACK.json", readback)
        write(BASE / "REFERENCE_LOCK.json", reference_lock)
    return readback, reference_lock


def contracts() -> dict[str, Any]:
    from scripts.stvg_crc_protocol_v1 import cpu_contracts
    assert _xyxy_to_cxcywh([0.1, 0.2, 0.5, 0.8]) == [0.3, 0.5, 0.4, 0.6000000000000001]
    base_slot = {
        "pair_job": {"job_id": "pair-1", "source_candidate_id": 1,
                     "alternative_candidate_id": 7, "source_grid_position": 2,
                     "center_physical_frame_id": 42,
                     "alternative_box_xyxy_normalized": [0.2, 0.3, 0.6, 0.7]},
        "parent_selected_candidate_id": 7, "source_grid_position": 2,
        "center_physical_frame_id": 42, "parent_presence": 0.75,
        "parent_response_sha256": "a" * 64,
    }
    parent_anchor = {"position": 2, "physical_frame_id": 42, "candidate_id": 7,
                     "box_cxcywh": [0.4, 0.5, 0.39999999999999997, 0.39999999999999997],
                     "weight": 0.75}
    out = _slot_anchor(base_slot, base_slot["pair_job"], parent_anchor, "b" * 64)
    assert out["candidate_id"] == 7 and out["position"] == 2 and out["weight"] == 0.75
    assert out["original_top1_source_sha256"] == "a" * 64
    fake_videos = [
        {"path": "a.npy", "sha256": "1" * 64, "tensor_sha256": "2" * 64,
         "bytes": 16, "shape": [2, 2, 2, 2], "dtype": "uint8"},
        {"path": "b.npy", "sha256": "3" * 64, "tensor_sha256": "4" * 64,
         "bytes": 24, "shape": [3, 2, 2, 2], "dtype": "uint8"},
    ]
    fake_reads = [
        {"path": v["path"], "file_sha256": v["sha256"],
         "tensor_sha256": v["tensor_sha256"], "file_bytes": v["bytes"],
         "tensor_bytes": math.prod(v["shape"]), "cache_hit": False}
        for v in fake_videos
    ]
    _validate_input_reads(fake_reads, fake_videos, "synthetic")
    bad_reads = [dict(row) for row in fake_reads]
    bad_reads[1]["tensor_sha256"] = "f" * 64
    try:
        _validate_input_reads(bad_reads, fake_videos, "synthetic")
    except RuntimeError:
        pass
    else:
        raise AssertionError("input readback accepted a mismatched tensor SHA")
    return {"status": "pass", "checks": ["strict_parser_contracts", "legal_xyxy_conversion",
            "alternative_only_anchor_geometry_and_provenance", "teacher_input_reads_match_locked_videos",
            "teacher_input_read_mismatch_rejected"], "check_count": 5,
            "parser": cpu_contracts(), "GT_read": False, "CPU_only": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", action="store_true")
    parser.add_argument("--lock", action="store_true", help="validate complete barriers and create readback/reference lock")
    args = parser.parse_args()
    if args.contracts:
        print(json.dumps(contracts(), sort_keys=True))
        return
    if not args.lock:
        parser.error("select --contracts or --lock")
    readback, reference = build_readback_and_lock(write_outputs=True)
    print(json.dumps({"readback": readback, "reference_records": len(reference["records"]),
                      "status": readback["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
