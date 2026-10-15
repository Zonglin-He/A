"""Run one locked 16-query stream for matched episodic vs. LN-carry recovery.

The module is inert on import.  It reuses the frozen source-capture and
full-model readout helpers, plus the separate frozen reference-recovery fitter.
No GT reader is imported or called here.
"""
from __future__ import annotations

import argparse
import collections
import copy
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stvg_reference_recovery_common_v1 as common

BASE = common.BASE
PARENT = common.PARENT
OLD = common.OLD
ORDERS_PATH = BASE / "ORDERS.json"
ROSTER_PATH = BASE / "ROSTER_LOCK.json"
REFERENCE_PATH = BASE / "REFERENCE_LOCK.json"
RUNTIME_PATH = BASE / "RUNTIME.json"
ORDER_NAMES = ("order1", "order2", "order3")
DOMAINS = ("hc2", "vidstg")
ARMS = (
    "F", "T", "Episodic_Native", "Episodic_T",
    "Before_Native", "Before_T", "After_Native", "After_T",
)


def _cid(record: Mapping[str, Any]) -> str:
    value = record.get("case_id", record.get("probe_alias"))
    if not isinstance(value, str) or not value:
        raise ValueError("locked source record lacks a stable case_id/probe_alias")
    return value


def _sha_state(state: Mapping[str, Any], state_hash) -> str:
    return state_hash({name: value.detach().cpu() for name, value in state.items()})


def _assert_gt_free(doc: Mapping[str, Any], label: str) -> None:
    if doc.get("GT_read") is not False:
        raise ValueError(f"{label} must explicitly declare GT_read=false")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    obj = common.read(path)
    if not isinstance(obj, dict):
        raise ValueError(f"{label} must be a JSON object")
    return obj


def _parent_files() -> dict[str, dict[str, Any]]:
    """Return the exact 32 parent prediction-file entries after barrier checks."""
    roster = _load_json(OLD / "DEVELOPMENT_ROSTER_LOCK.json", "parent development roster")
    reference = _load_json(PARENT / "REFERENCE_LOCK.json", "parent reference lock")
    barrier = _load_json(PARENT / "GLOBAL_PREDICTION_BARRIER.json", "parent prediction barrier")
    _assert_gt_free(roster, "parent roster")
    _assert_gt_free(reference, "parent reference lock")
    _assert_gt_free(barrier, "parent prediction barrier")
    if roster.get("status") != "locked" or reference.get("status") != "locked":
        raise ValueError("parent source/reference locks are not locked")
    if barrier.get("status") != "complete" or int(barrier.get("count", -1)) != 32:
        raise ValueError("the exact 32-query parent prediction barrier is not complete")
    result: dict[str, dict[str, Any]] = {}
    allowed_metadata = {
        str((PARENT / name).relative_to(ROOT))
        for name in ("REFERENCE_LOCK.json", "TEACHER_GLOBAL_BARRIER.json", "TEACHER_JOB_LOCK.json",
                     "CANDIDATE_INPUT_BARRIER.json", "ROOT_INPUT_READBACK.json",
                     "ROOT_TEMPORAL_READBACK.json", "ROOT_SPATIAL_READBACK.json", "RUNTIME.json")
    }
    allowed_metadata.update({
        str((OLD / "TEMPORAL_SELECTION_LOCK.json").relative_to(ROOT)),
        str((OLD / "DESIGN_LOCK.json").relative_to(ROOT)),
    })
    for item in barrier.get("files", []):
        rel = item.get("path")
        if not isinstance(rel, str):
            raise ValueError("parent barrier has a nonstring file path")
        path = (ROOT / rel).resolve()
        if not path.is_file() or common.sha(path) != item.get("sha256"):
            raise ValueError(f"parent barrier file SHA mismatch: {rel}")
        if item.get("bytes") is not None and path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"parent barrier byte-count mismatch: {rel}")
        if not rel.endswith(".pt"):
            if rel not in allowed_metadata:
                raise ValueError(f"parent barrier contains an unapproved metadata path: {rel}")
            continue
        case_id = path.stem
        expected_path = (PARENT / "student" / f"{case_id}.pt").resolve()
        if path != expected_path:
            raise ValueError(f"parent prediction is outside its locked student directory: {rel}")
        if case_id in result:
            raise ValueError("parent prediction barrier has duplicate case files")
        result[case_id] = {"path": path, "sha256": item["sha256"], "bytes": path.stat().st_size}
    if len(result) != 32:
        raise ValueError("parent prediction barrier must contain exactly 32 unique source files")
    return result


def _parent_reference_records() -> dict[str, dict[str, Any]]:
    """Read the frozen new_crop anchors, without opening any scored outputs."""
    doc = _load_json(PARENT / "REFERENCE_LOCK.json", "parent reference lock")
    records = doc.get("records")
    if not isinstance(records, list) or len(records) != 32:
        raise ValueError("parent reference lock must contain exactly 32 records")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        cid = str(record.get("case_id", ""))
        new_crop = record.get("new_crop")
        if not cid or cid in result or not isinstance(new_crop, dict) or not isinstance(new_crop.get("anchors"), list):
            raise ValueError("parent reference lock has duplicate IDs or malformed new_crop anchors")
        result[cid] = record
    return result


def _plan_orders(rows: list[dict[str, Any]], orders_doc: Mapping[str, Any]) -> dict[tuple[str, str], list[str]]:
    """Independently reconstruct the six ordinal streams from frozen PLAN files."""
    from scripts.stvg_evidence_common_v1 import roster as parent_roster

    original_rows = parent_roster("development")
    original_index = {_cid(r): r for r in original_rows}
    row_index = {_cid(r): r for r in rows}
    if len(original_index) != 32 or len(row_index) != 32 or set(original_index) != set(row_index):
        raise ValueError("new and original development roster membership differs")
    for cid, new in row_index.items():
        old = original_index[cid]
        for field in ("dataset", "query_ordinal", "query_key_sha256", "plan_path", "plan_sha256"):
            if new.get(field) != old.get(field):
                raise ValueError(f"development roster {field} changed for {cid}")

    by_domain = {domain: [rec for rec in rows if rec.get("dataset") == domain] for domain in DOMAINS}
    reconstructed: dict[tuple[str, str], list[str]] = {}
    for entry in orders_doc.get("records", []):
        domain = str(entry.get("dataset"))
        order = str(entry.get("order"))
        if domain not in DOMAINS or order not in ORDER_NAMES:
            raise ValueError("ORDERS.json has an unsupported domain/order")
        rows_domain = by_domain[domain]
        if len(rows_domain) != 16:
            raise ValueError(f"E3 roster must contain exactly 16 sources in {domain}")
        plan_path = (ROOT / rows_domain[0]["plan_path"]).resolve()
        if not plan_path.is_file() or common.sha(plan_path) != rows_domain[0].get("plan_sha256"):
            raise ValueError(f"source PLAN hash mismatch for {domain}")
        if any((ROOT / r["plan_path"]).resolve() != plan_path or r["plan_sha256"] != rows_domain[0]["plan_sha256"]
               for r in rows_domain):
            raise ValueError(f"source PLAN binding differs within {domain}")
        if entry.get("source_plan_path") != str(plan_path.relative_to(ROOT)) or entry.get("source_plan_sha256") != common.sha(plan_path):
            raise ValueError(f"ORDERS.json source PLAN binding mismatch for {domain}/{order}")
        plan = _load_json(plan_path, f"{domain} PLAN")
        full_order = plan.get("orders", {}).get(order)
        plan_rows = plan.get("rows")
        if not isinstance(full_order, list) or not isinstance(plan_rows, list):
            raise ValueError(f"{domain}/{order} lacks the original full ordinal sequence")
        plan_ordinals = [r.get("ordinal") for r in plan_rows]
        if (any(isinstance(x, bool) or not isinstance(x, int) for x in plan_ordinals)
                or len(set(plan_ordinals)) != len(plan_ordinals)
                or len(full_order) != len(plan_ordinals)
                or any(isinstance(x, bool) or not isinstance(x, int) for x in full_order)
                or len(set(full_order)) != len(full_order)
                or set(full_order) != set(plan_ordinals)):
            raise ValueError(f"{domain} PLAN rows/order are not a complete unique ordinal permutation")
        ordinal_to_row = {int(r["query_ordinal"]): r for r in rows_domain}
        if len(ordinal_to_row) != 16:
            raise ValueError(f"duplicate query ordinal in {domain} roster")
        selected_ordinals = set(ordinal_to_row)
        ordinals = [ordinal for ordinal in full_order if ordinal in selected_ordinals]
        if len(ordinals) != 16 or set(ordinals) != selected_ordinals:
            raise ValueError(f"{domain}/{order} does not contain the exact locked 16-source subsequence")
        ids = []
        for ordinal in ordinals:
            if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal not in ordinal_to_row:
                raise ValueError(f"{domain}/{order} contains an ordinal outside the locked source set")
            ids.append(_cid(ordinal_to_row[ordinal]))
        if len(set(ids)) != 16:
            raise ValueError(f"{domain}/{order} contains duplicate aliases")
        if list(map(int, entry.get("query_ordinals", []))) != list(ordinals):
            raise ValueError(f"ORDERS.json ordinals differ from PLAN for {domain}/{order}")
        if entry.get("case_ids") != ids:
            raise ValueError(f"ORDERS.json aliases differ from PLAN ordinal mapping for {domain}/{order}")
        if entry.get("query_key_sha256s") != [ordinal_to_row[o]["query_key_sha256"] for o in ordinals]:
            raise ValueError(f"ORDERS.json query-key bindings differ for {domain}/{order}")
        reconstructed[(domain, order)] = ids
    return reconstructed


def _read_locked_inputs():
    runtime = common.verify_runtime()
    rows = common.roster()
    roster_doc = _load_json(ROSTER_PATH, "E3 roster lock")
    ref_doc = _load_json(REFERENCE_PATH, "E3 reference lock")
    orders_doc = _load_json(ORDERS_PATH, "E3 order lock")
    for doc, label in ((roster_doc, "E3 roster"), (ref_doc, "E3 references"), (orders_doc, "E3 orders")):
        _assert_gt_free(doc, label)
        if doc.get("status") != "locked":
            raise ValueError(f"{label} is not locked")
    if len(rows) != 32 or len({_cid(r) for r in rows}) != 32:
        raise ValueError("E3 roster must contain the exact 32 distinct development sources")
    if int(roster_doc.get("count", -1)) != 32 or int(ref_doc.get("count", -1)) != 32:
        raise ValueError("E3 roster and reference locks must each have count 32")
    if roster_doc.get("records") != rows:
        raise ValueError("common roster reader does not match the locked roster records")
    parent_development_roster = OLD / "DEVELOPMENT_ROSTER_LOCK.json"
    if roster_doc.get("parent_roster_sha256") != common.sha(parent_development_roster):
        raise ValueError("E3 roster does not bind the original development roster")

    ref_rows = ref_doc.get("records")
    if not isinstance(ref_rows, list) or len(ref_rows) != 32:
        raise ValueError("E3 reference lock must contain exactly 32 records")
    ref_index: dict[str, dict[str, Any]] = {}
    base_ids = {_cid(r) for r in rows}
    row_index = {_cid(r): r for r in rows}
    for ref in ref_rows:
        cid = str(ref.get("case_id", ""))
        if cid in ref_index or cid not in base_ids:
            raise ValueError("E3 reference lock has duplicate or unexpected case IDs")
        if ref.get("dataset") != row_index[cid]["dataset"]:
            raise ValueError(f"E3 reference dataset mismatch for {cid}")
        anchors = ref.get("anchors")
        if not isinstance(anchors, list):
            raise ValueError(f"E3 reference anchors must be a list for {cid}")
        ref_index[cid] = ref
    if set(ref_index) != base_ids:
        raise ValueError("E3 reference lock membership differs from the roster")

    executable_rows = {}
    for cid, rec in row_index.items():
        if not isinstance(ref_index[cid].get("source_case_sha256"), str):
            raise ValueError(f"reference lock lacks source-case SHA for {cid}")
        executable = dict(rec)
        executable["case_id"] = cid
        executable["source_case_sha256"] = ref_index[cid]["source_case_sha256"]
        executable_rows[cid] = executable

    reconstructed = _plan_orders(rows, orders_doc)
    declared_records = orders_doc.get("records")
    if not isinstance(declared_records, list) or len(declared_records) != 6:
        raise ValueError("ORDERS.json must have exactly six dataset/order records")
    declared: dict[tuple[str, str], list[str]] = {}
    for entry in declared_records:
        key = (str(entry.get("dataset")), str(entry.get("order")))
        if key in declared or key not in reconstructed:
            raise ValueError("ORDERS.json has duplicate or unsupported stream entries")
        aliases = entry.get("case_ids")
        if not isinstance(aliases, list):
            raise ValueError("each ORDERS.json record must carry ordered case_ids")
        declared[key] = [str(cid) for cid in aliases]
    if declared != reconstructed:
        raise ValueError("ORDERS.json differs from the frozen PLAN ordinal sequences")

    parent_files = _parent_files()
    parent_refs = _parent_reference_records()
    if set(parent_files) != set(ref_index):
        raise ValueError("parent prediction set does not cover the same 32 fixed sources")
    if set(parent_refs) != set(ref_index):
        raise ValueError("parent reference set does not cover the same 32 fixed sources")
    for cid, ref in ref_index.items():
        parent_ref = parent_refs[cid]
        rec = row_index[cid]
        if (parent_ref.get("dataset") != rec["dataset"]
                or parent_ref.get("source_case_sha256") != ref["source_case_sha256"]):
            raise ValueError(f"parent reference source binding differs for {cid}")
        if ref["anchors"] != parent_ref["new_crop"]["anchors"]:
            raise ValueError(f"E3 anchors differ from the frozen parent new_crop Top1 reference for {cid}")
    parent_reference_sha = common.sha(PARENT / "REFERENCE_LOCK.json")
    if ref_doc.get("parent_reference_lock_sha256") != parent_reference_sha:
        raise ValueError("E3 reference lock does not bind the exact parent reference lock")
    parent_teacher_barrier = PARENT / "TEACHER_GLOBAL_BARRIER.json"
    if ref_doc.get("parent_teacher_barrier_sha256") != common.sha(parent_teacher_barrier):
        raise ValueError("E3 reference lock does not bind the exact parent teacher barrier")
    expected_parent_sha = orders_doc.get("parent_prediction_barrier_sha256")
    if expected_parent_sha is not None and expected_parent_sha != common.sha(PARENT / "GLOBAL_PREDICTION_BARRIER.json"):
        raise ValueError("ORDERS.json parent-prediction barrier binding mismatch")
    # Runtime must bind the complete local setup; compare only when the lock
    # declares the corresponding fields, while all pin/protected files are
    # unconditionally SHA-verified by common.verify_runtime().
    for field, path in (("roster_lock_sha256", ROSTER_PATH),
                        ("reference_lock_sha256", REFERENCE_PATH),
                        ("orders_sha256", ORDERS_PATH)):
        if field in runtime and runtime[field] != common.sha(path):
            raise ValueError(f"runtime {field} does not bind its lock")
    return rows, executable_rows, ref_index, declared, runtime, parent_files


def _load_parent_prediction(case_id: str, rec: Mapping[str, Any], parent_files: Mapping[str, Any], torch):
    entry = parent_files[case_id]
    value = torch.load(entry["path"], map_location="cpu", weights_only=False)
    if value.get("case_id") != case_id or value.get("dataset") != rec["dataset"]:
        raise ValueError(f"parent prediction identity mismatch: {case_id}")
    if value.get("source_case_sha256") != rec["source_case_sha256"]:
        raise ValueError(f"parent prediction source binding mismatch: {case_id}")
    if value.get("GT_read") is not False:
        raise ValueError(f"parent prediction must be explicitly GT-free: {case_id}")
    arms = value.get("arms")
    required = ("F", "T", "S_new_crop", "TS_new_crop")
    if not isinstance(arms, dict) or any(name not in arms for name in required):
        raise ValueError(f"parent prediction lacks the exact four reusable arms: {case_id}")
    return value


def _state_copy(state: Mapping[str, Any]):
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _same(left, right, torch) -> bool:
    return (set(left) == set(right) and all(
        torch.is_tensor(left[name]) and torch.is_tensor(right[name])
        and left[name].dtype == right[name].dtype
        and left[name].shape == right[name].shape
        and torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
        for name in left
    ))


def _state_digest(state, state_hash) -> str:
    return _sha_state(state, state_hash)


def _validate_saved_case(path: Path, receipt_path: Path, *, cid: str, dataset: str,
                         order: str, expected_previous_sha: str | None, torch,
                         state_hash, lock_sha: str) -> dict[str, Any]:
    if not path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"incomplete query output; refusing overwrite: {cid}")
    receipt = _load_json(receipt_path, f"{cid} stream receipt")
    if receipt.get("status") != "sealed" or receipt.get("GT_read") is not False:
        raise ValueError(f"existing query receipt is not a GT-free seal: {cid}")
    if receipt.get("case_id") != cid or receipt.get("dataset") != dataset or receipt.get("order") != order:
        raise ValueError(f"existing query receipt identity mismatch: {cid}")
    if receipt.get("reference_lock_sha256") != lock_sha:
        raise ValueError(f"existing query was produced under another reference lock: {cid}")
    if receipt.get("previous_result_sha256") != expected_previous_sha:
        raise ValueError(f"existing query has a broken state-chain predecessor: {cid}")
    if common.sha(path) != receipt.get("sha256") or path.stat().st_size != int(receipt.get("bytes", -1)):
        raise ValueError(f"existing query artifact SHA/size mismatch: {cid}")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("case_id") != cid or value.get("dataset") != dataset or value.get("order") != order:
        raise ValueError(f"existing query payload identity mismatch: {cid}")
    if value.get("source_case_sha256") != receipt.get("source_case_sha256"):
        raise ValueError(f"existing query source hash mismatch: {cid}")
    if value.get("GT_read") is not False or set(value.get("arms", {})) != set(ARMS):
        raise ValueError(f"existing query payload schema/GT flag mismatch: {cid}")
    next_state = value.get("next_committed")
    if not isinstance(next_state, dict) or _state_digest(next_state, state_hash) != receipt.get("next_state_sha256"):
        raise ValueError(f"existing next-state carry hash mismatch: {cid}")
    if value.get("previous_result_sha256") != expected_previous_sha:
        raise ValueError(f"existing payload state-chain predecessor mismatch: {cid}")
    return value


def _write_case(path: Path, receipt_path: Path, payload: dict[str, Any], *, cid: str,
                dataset: str, order: str, previous_result_sha: str | None,
                lock_sha: str, runtime_sha: str, next_state_sha: str,
                state_hash) -> dict[str, Any]:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite any partial/existing output: {path}")
    temp = path.with_name(path.name + ".tmp")
    if temp.exists():
        raise FileExistsError(f"unresolved partial output exists: {temp}")
    torch.save(payload, temp)
    os.replace(temp, path)
    receipt = {
        "schema": "stvg-reference-carry-case-receipt-v1",
        "status": "sealed",
        "case_id": cid,
        "dataset": dataset,
        "order": order,
        "path": str(path.relative_to(ROOT)),
        "sha256": common.sha(path),
        "bytes": path.stat().st_size,
        "source_case_sha256": payload["source_case_sha256"],
        "reference_lock_sha256": lock_sha,
        "runtime_sha256": runtime_sha,
        "previous_result_sha256": previous_result_sha,
        "incoming_state_sha256": _state_digest(payload["initial_state"], state_hash),
        "next_state_sha256": next_state_sha,
        "optimizer_steps": int(payload["fit"]["optimizer_steps"]),
        "actual_fit_count": 1,
        "actual_full_readout_calls": int(payload["actual_full_readout_calls"]),
        "GT_read": False,
        "time": time.time(),
    }
    common.write(receipt_path, receipt)
    return {"path": str(path.relative_to(ROOT)), "sha256": receipt["sha256"], "bytes": receipt["bytes"]}


def _stream_paths(dataset: str, order: str, cid: str) -> tuple[Path, Path]:
    path = BASE / "student" / dataset / order / f"{cid}.pt"
    return path, path.with_suffix(".receipt.json")


def _existing_stream_barrier(path: Path, *, dataset: str, order: str,
                             ordered_ids: list[str], lock_sha: str, runtime_sha: str,
                             torch, state_hash):
    if not path.exists():
        return None
    doc = _load_json(path, "existing stream barrier")
    if doc.get("schema") != "stvg-reference-carry-stream-barrier-v1" or doc.get("status") != "complete":
        raise ValueError("existing stream barrier is invalid; refusing overwrite")
    if doc.get("dataset") != dataset or doc.get("order") != order or doc.get("case_ids") != ordered_ids:
        raise ValueError("existing stream barrier stream identity/order mismatch")
    if doc.get("reference_lock_sha256") != lock_sha or doc.get("runtime_sha256") != runtime_sha:
        raise ValueError("existing stream barrier is bound to another lock/runtime")
    if doc.get("GT_read") is not False:
        raise ValueError("existing stream barrier must be GT-free")
    files = doc.get("files")
    if doc.get("count") != 16 or not isinstance(files, list) or len(files) != 16:
        raise ValueError("existing stream barrier must enumerate exactly 16 outputs")
    expected_previous_sha = None
    for cid, item in zip(ordered_ids, files):
        artifact, receipt = _stream_paths(dataset, order, cid)
        if item.get("path") != str(artifact.relative_to(ROOT)):
            raise ValueError(f"stream barrier path/order mismatch for {cid}")
        if not artifact.is_file() or common.sha(artifact) != item.get("sha256") or artifact.stat().st_size != int(item.get("bytes", -1)):
            raise ValueError(f"stream barrier output changed: {item.get('path')}")
        saved = _validate_saved_case(
            artifact, receipt, cid=cid, dataset=dataset, order=order,
            expected_previous_sha=expected_previous_sha, torch=torch,
            state_hash=state_hash, lock_sha=lock_sha,
        )
        expected_previous_sha = common.sha(artifact)
    return doc


def _run_stream(dataset: str, order: str, ordered_ids: list[str], row_index,
                reference_index, runtime, parent_files) -> dict[str, Any]:
    import numpy as np
    import torch
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_decota_paper_main_v1 import model_for, frames_for
    from scripts.stvg_mllm_feedback_common_v1 import raw_row
    from scripts import run_stvg_table1_source_v1 as source_module
    from scripts.run_stvg_table1_student004 import stable_fit_key
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward, transient_native_masks
    from vg_tta.stvg_opd_frozen_visual_host011 import frozen_visual_host
    from vg_tta.stvg_opd_frozen_attention_lifetime010 import frozen_attention_storage
    from vg_tta.stvg_opd_frozen_module_cpu008 import inactive_modules_on_host
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.backbone import query_subject
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts import run_stvg_adjacent_motion_student_v1 as source_helper
    from scripts.stvg_reference_recovery_fit_v1 import (
        initial_state, locked_config, fit_reference_recovery, transition_record,
    )

    stream_dir = BASE / "student" / dataset / order
    barrier_path = stream_dir / "STREAM_BARRIER.json"
    lock_sha = common.sha(REFERENCE_PATH)
    runtime_sha = common.sha(RUNTIME_PATH)
    existing_barrier = _existing_stream_barrier(
        barrier_path, dataset=dataset, order=order, ordered_ids=ordered_ids,
        lock_sha=lock_sha, runtime_sha=runtime_sha, torch=torch, state_hash=state_hash,
    )
    if existing_barrier is not None:
        return existing_barrier

    install_clean_loader()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(20261014)

    output_files: list[dict[str, Any]] = []
    previous_committed = None
    previous_result_sha: str | None = None
    model = None
    current_source = None
    total_optimizer_steps = total_full_readouts = 0
    fit_count = 0
    stream_started = time.perf_counter()

    for position, cid in enumerate(ordered_ids):
        rec = row_index[cid]
        if rec["dataset"] != dataset:
            raise ValueError(f"order lock placed {cid} in the wrong domain")
        out_path, receipt_path = _stream_paths(dataset, order, cid)
        if out_path.exists() or receipt_path.exists():
            saved = _validate_saved_case(
                out_path, receipt_path, cid=cid, dataset=dataset, order=order,
                expected_previous_sha=previous_result_sha, torch=torch,
                state_hash=state_hash, lock_sha=lock_sha,
            )
            previous_committed = saved["next_committed"]
            previous_result_sha = common.sha(out_path)
            total_optimizer_steps += int(saved["fit"]["optimizer_steps"])
            total_full_readouts += int(saved["actual_full_readout_calls"])
            fit_count += 1
            output_files.append({"path": str(out_path.relative_to(ROOT)), "sha256": previous_result_sha,
                                 "bytes": out_path.stat().st_size})
            continue

        # Load the frozen source case through the common verifier, then add
        # only the transient aliases expected by the already-validated source
        # capture helper.  The catalog override is scoped to this call.
        case, case_path = common.load_case(rec)
        source_roster_record = dict(rec)
        _, official_row = source_module._validate_roster_record(source_roster_record)
        transient_case = dict(
            case,
            source=source_module.SOURCE_MODEL_NAME[str(rec["dataset"])],
            _source_case_sha256=rec["source_case_sha256"],
            parent=official_row["source"],
        )
        capture_rec = {
            "case_id": cid,
            "dataset": rec["dataset"],
            "source_case_sha256": rec["source_case_sha256"],
        }
        previous_catalog = source_helper.catalog
        try:
            def locked_catalog(requested):
                if any(requested.get(key) != capture_rec[key] for key in capture_rec):
                    raise ValueError("source capture requested a different locked source case")
                return transient_case, {}

            source_helper.catalog = locked_catalog
            capture = source_helper._capture_source(
                capture_rec, model, current_source, torch, collections,
                model_for, frames_for, raw_row, frozen_forward, prediction, state_hash,
            )
        finally:
            source_helper.catalog = previous_catalog
        (model, current_source, captured_case, _catalog_data, row, frames, frame_ids,
         records, batch, base, native, expected_source_model_sha, pixel_sha,
         capture_seconds) = capture
        if captured_case["source_case_sha256"] != rec["source_case_sha256"]:
            raise ValueError(f"loaded source case SHA changed for {cid}")
        if frame_ids != list(map(int, case["frame_ids"])):
            raise ValueError(f"source-grid frame IDs changed for {cid}")

        parent = _load_parent_prediction(cid, rec, parent_files, torch)
        if parent.get("source_model_state_sha256") != expected_source_model_sha:
            raise ValueError(f"parent frozen source-model hash differs for {cid}")
        if list(map(int, parent.get("frame_ids", []))) != frame_ids:
            raise ValueError(f"parent source-grid frame IDs differ for {cid}")
        if parent.get("pixel_sha256") != pixel_sha:
            raise ValueError(f"parent source pixel hash differs for {cid}")
        if not np.array_equal(np.asarray(parent["arms"]["F"]["boxes"]), source_helper._xyxy(native["boxes"])):
            raise ValueError(f"parent F geometry is not exact source geometry for {cid}")
        if list(parent["arms"]["F"]["interval"]) != list(captured_case["Native_interval"]):
            raise ValueError(f"parent F interval differs from the source native interval for {cid}")
        if not np.array_equal(np.asarray(parent["arms"]["T"]["boxes"]), np.asarray(parent["arms"]["F"]["boxes"])):
            raise ValueError(f"parent T arm changes the source boxes for {cid}")
        if not np.array_equal(np.asarray(parent["arms"]["S_new_crop"]["boxes"]),
                              np.asarray(parent["arms"]["TS_new_crop"]["boxes"])):
            raise ValueError(f"parent S/TS boxes are not exact reused spatial output for {cid}")
        if list(parent["arms"]["S_new_crop"]["interval"]) != list(captured_case["Native_interval"]):
            raise ValueError(f"parent S_new_crop arm does not retain Native time for {cid}")
        if list(parent["arms"]["TS_new_crop"]["interval"]) != list(parent["arms"]["T"]["interval"]):
            raise ValueError(f"parent TS_new_crop arm does not reuse the exact parent T interval for {cid}")

        reference = reference_index[cid]
        if reference["source_case_sha256"] != rec["source_case_sha256"]:
            raise ValueError(f"E3 reference source SHA mismatch for {cid}")
        anchors = source_helper._anchors(reference, frame_ids, torch)
        if not anchors:
            # Empty references still run the locked no-information fit and
            # retain the actual online arrival state; no fake anchor is added.
            pass

        source_initial = _state_copy(base.initial)
        if previous_committed is None:
            incoming = initial_state(source_initial, None, carry_ln=False)
        else:
            incoming = initial_state(source_initial, previous_committed, carry_ln=True)
        source_helper._validate_interval(captured_case["Native_interval"], captured_case["video_bounds"], "Native")
        source_helper._validate_interval(captured_case["Projection_interval"], captured_case["video_bounds"], "Projection")

        config = locked_config(dataset, "current5")
        key = stable_fit_key(str(rec["dataset"]), str(rec["query_key_sha256"]))
        fit_started = time.perf_counter()
        fit = fit_reference_recovery(
            base=base,
            initial=incoming,
            anchors=anchors,
            frame_ids=frame_ids,
            key=key,
            config=config,
            profile="current5",
            dataset=dataset,
            validate=True,
            reference_sha256=lock_sha,
        )
        if fit.get("reference_role") != "fixed_weight_free_top1_teacher" or fit.get("GT_read") is not False:
            raise ValueError("frozen fitter returned an unexpected reference role or GT flag")
        if not _same(fit["initial_state"], incoming, torch):
            raise ValueError(f"fit arrival state differs from the explicit stream state for {cid}")
        if not torch.isfinite(fit["initial_cached_decoder_boxes"]).all() or not torch.isfinite(fit["final_cached_decoder_boxes"]).all():
            raise ValueError(f"fitter decoder cache contains nonfinite values for {cid}")

        # The fit helper restores its input state, which may contain carried
        # LN values. Reset the model's actual parameters to Source before each
        # full-model readout; each readout temporarily inserts one explicit
        # adapted state and must leave the frozen model Source-identical.
        base.restore(source_initial)
        expected_box_before = fit["initial_cached_decoder_boxes"].detach().cpu()
        before_stub = {"final_state": incoming}
        before_native = source_helper._readout(
            model, batch, frame_ids, records, before_stub, row["parses"]["subject"], core,
            torch, transient_native_masks=transient_native_masks,
            frozen_visual_host=frozen_visual_host,
            frozen_attention_storage=frozen_attention_storage,
            inactive_modules_on_host=inactive_modules_on_host,
            query_subject=query_subject,
        )
        if not torch.equal(before_native["boxes"].detach().cpu(), expected_box_before):
            raise ValueError(f"full-model Before readout differs from cached arrival decoder boxes for {cid}")
        if not source_helper._same_logits(before_native["logits"], native["logits"]):
            raise ValueError(f"Before readout changed source temporal logits for {cid}")
        if state_hash(model.state_dict()) != expected_source_model_sha:
            raise ValueError(f"Before readout failed to restore frozen Source model for {cid}")

        base.restore(source_initial)
        after_native = source_helper._readout(
            model, batch, frame_ids, records, fit, row["parses"]["subject"], core,
            torch, transient_native_masks=transient_native_masks,
            frozen_visual_host=frozen_visual_host,
            frozen_attention_storage=frozen_attention_storage,
            inactive_modules_on_host=inactive_modules_on_host,
            query_subject=query_subject,
        )
        if not torch.equal(after_native["boxes"].detach().cpu(), fit["final_cached_decoder_boxes"].detach().cpu()):
            raise ValueError(f"full-model After readout differs from cached accepted decoder boxes for {cid}")
        if not source_helper._same_logits(after_native["logits"], native["logits"]):
            raise ValueError(f"After readout changed source temporal logits for {cid}")
        if state_hash(model.state_dict()) != expected_source_model_sha:
            raise ValueError(f"After readout failed to restore frozen Source model for {cid}")

        before_boxes = source_helper._xyxy(before_native["boxes"])
        after_boxes = source_helper._xyxy(after_native["boxes"])
        native_interval = list(captured_case["Native_interval"])
        old_t_interval = list(parent["arms"]["T"]["interval"])
        if not (captured_case["video_bounds"][0] <= old_t_interval[0] < old_t_interval[1] <= captured_case["video_bounds"][1]):
            raise ValueError(f"parent T interval falls outside the source physical clock for {cid}")
        arms = {
            "F": copy.deepcopy(parent["arms"]["F"]),
            "T": copy.deepcopy(parent["arms"]["T"]),
            "Episodic_Native": copy.deepcopy(parent["arms"]["S_new_crop"]),
            "Episodic_T": copy.deepcopy(parent["arms"]["TS_new_crop"]),
            "Before_Native": source_helper._arm(before_boxes, native_interval),
            "Before_T": source_helper._arm(before_boxes, old_t_interval),
            "After_Native": source_helper._arm(after_boxes, native_interval),
            "After_T": source_helper._arm(after_boxes, old_t_interval),
        }
        if not np.array_equal(arms["Before_Native"]["boxes"], arms["After_Native"]["boxes"]) and int(fit["optimizer_steps"]) == 0:
            raise ValueError(f"zero-step fit changed the decoded spatial output for {cid}")

        transition = transition_record(
            source_initial, incoming, fit, carry_ln=True, writeback=float(config["writeback"]),
        )
        next_committed = transition["next_stream_state"]
        if not torch.count_nonzero(next_committed["spatial.query_residual"]) == 0:
            raise ValueError(f"Query residual was not reset after stream commit for {cid}")

        previous_binding = None
        if position > 0:
            previous_binding = {"case_id": ordered_ids[position - 1], "result_sha256": previous_result_sha}
        payload = {
            "schema": "stvg-reference-carry-case-v1",
            "case_id": cid,
            "dataset": dataset,
            "order": order,
            "stream_position": position,
            "source_case_sha256": rec["source_case_sha256"],
            "source_model_state_sha256": expected_source_model_sha,
            "pixel_sha256": pixel_sha,
            "frame_ids": frame_ids,
            "Native_interval": native_interval,
            "Projection_interval": list(captured_case["Projection_interval"]),
            "temporal_reuse": {
                "status": "exact_parent_round2_reuse",
                "parent_prediction_path": str(parent_files[cid]["path"].relative_to(ROOT)),
                "parent_prediction_sha256": parent_files[cid]["sha256"],
                "parent_student_barrier_sha256": common.sha(PARENT / "GLOBAL_PREDICTION_BARRIER.json"),
                "Episodic_Native_arm": "S_new_crop",
                "Episodic_T_arm": "TS_new_crop",
            },
            "reference_anchors": copy.deepcopy(reference["anchors"]),
            "reference_lock_sha256": lock_sha,
            "orders_sha256": common.sha(ORDERS_PATH),
            "roster_lock_sha256": common.sha(ROSTER_PATH),
            "runtime_sha256": runtime_sha,
            "previous_result_sha256": previous_result_sha,
            "previous_stream_item": previous_binding,
            "arms": arms,
            "fit": fit,
            "readouts": {
                "Before_Native": before_native,
                "After_Native": after_native,
                "T_interval_reuse_exact": True,
                "full_model_redecode": True,
            },
            "initial_state": _state_copy(incoming),
            "committed_before": None if previous_committed is None else _state_copy(previous_committed),
            "next_committed": _state_copy(next_committed),
            "state_transition": transition,
            "initial_state_sha256": _state_digest(incoming, state_hash),
            "next_state_sha256": _state_digest(next_committed, state_hash),
            "source_restored": True,
            "source_cold_readback": {
                "status": "pass",
                "source_case_sha256": rec["source_case_sha256"],
                "source_model_state_sha256": expected_source_model_sha,
                "pixel_sha256": pixel_sha,
                "frame_ids_match": True,
                "zero_boxes_logits_match": True,
                "Source_restored_after_Before_and_After": True,
            },
            "actual_fit_count": 1,
            "optimizer_steps": int(fit["optimizer_steps"]),
            "actual_full_readout_calls": 2,
            "fit_wall_seconds": float(time.perf_counter() - fit_started),
            "GT_read": False,
            "online_carry": True,
        }
        file_item = _write_case(
            out_path, receipt_path, payload, cid=cid, dataset=dataset, order=order,
            previous_result_sha=previous_result_sha, lock_sha=lock_sha,
            runtime_sha=runtime_sha, next_state_sha=payload["next_state_sha256"],
            state_hash=state_hash,
        )
        output_files.append(file_item)
        previous_result_sha = file_item["sha256"]
        previous_committed = payload["next_committed"]
        total_optimizer_steps += int(fit["optimizer_steps"])
        total_full_readouts += 2
        fit_count += 1
        del case, captured_case, row, frames, records, batch, base, native, parent, payload, fit
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if model is not None:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(output_files) != 16 or fit_count != 16:
        raise ValueError("stream must seal exactly 16 ordered source queries")
    if len({item["path"] for item in output_files}) != 16:
        raise ValueError("stream output paths are not unique")
    if barrier_path.exists():
        raise FileExistsError(f"refusing to overwrite stream barrier: {barrier_path}")
    barrier = {
        "schema": "stvg-reference-carry-stream-barrier-v1",
        "status": "complete",
        "dataset": dataset,
        "order": order,
        "count": 16,
        "case_ids": list(ordered_ids),
        "files": output_files,
        "actual_fit_count": fit_count,
        "optimizer_steps": total_optimizer_steps,
        "actual_full_readout_calls": total_full_readouts,
        "roster_lock_sha256": common.sha(ROSTER_PATH),
        "reference_lock_sha256": lock_sha,
        "orders_sha256": common.sha(ORDERS_PATH),
        "runtime_sha256": runtime_sha,
        "parent_prediction_barrier_sha256": common.sha(PARENT / "GLOBAL_PREDICTION_BARRIER.json"),
        "state_chain": "LN only; Query residual resets to zero per query; Source LN at stream start",
        "elapsed_seconds": time.perf_counter() - stream_started,
        "GT_read": False,
    }
    common.write(barrier_path, barrier)
    return barrier


def _synthetic_contract() -> dict[str, Any]:
    """CPU-only checks for reset, carry, writeback, and query-chain semantics."""
    import torch
    from scripts.stvg_reference_recovery_fit_v1 import initial_state, commit_state

    source = {"spatial.query_residual": torch.ones(256, dtype=torch.float32)}
    source.update({f"spatial.layers.{i}": torch.zeros(256, dtype=torch.float32) for i in range(6)})
    prior = {name: value.clone() for name, value in source.items()}
    for name in prior:
        if name != "spatial.query_residual":
            prior[name].fill_(2.0)
    prior["spatial.query_residual"].fill_(7.0)
    arrival = initial_state(source, prior, carry_ln=True)
    assert torch.count_nonzero(arrival["spatial.query_residual"]) == 0
    assert all(torch.equal(arrival[name], prior[name]) for name in source if name != "spatial.query_residual")
    fit_final = {name: value.clone() for name, value in arrival.items()}
    for name in fit_final:
        if name != "spatial.query_residual":
            fit_final[name].add_(4.0)
    fit_final["spatial.query_residual"].fill_(9.0)
    next_state = commit_state(source, arrival, fit_final, .0625, carry_ln=True)
    assert torch.count_nonzero(next_state["spatial.query_residual"]) == 0
    assert all(torch.equal(next_state[name], arrival[name] + (fit_final[name] - arrival[name]) * .0625)
               for name in source if name != "spatial.query_residual")
    first_arrival = initial_state(source, None, carry_ln=False)
    assert all(torch.equal(first_arrival[name], source[name]) for name in source if name != "spatial.query_residual")
    no_update = {name: value.clone() for name, value in first_arrival.items()}
    committed = commit_state(source, first_arrival, no_update, .125, carry_ln=True)
    assert all(torch.equal(committed[name], first_arrival[name]) for name in source)
    return {"status": "pass", "checks": 5, "GT_read": False, "CPU_only": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DOMAINS)
    parser.add_argument("--order", choices=ORDER_NAMES)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_synthetic_contract(), sort_keys=True))
        return
    if not args.dataset or not args.order:
        parser.error("--dataset and --order are required unless --self-test is used")
    rows, row_index, ref_index, orders, runtime, parent_files = _read_locked_inputs()
    ordered_ids = orders[(args.dataset, args.order)]
    sys.addaudithook(common.guard)
    result = _run_stream(args.dataset, args.order, ordered_ids, row_index, ref_index, runtime, parent_files)
    print(json.dumps({"status": result["status"], "dataset": args.dataset,
                      "order": args.order, "count": result["count"],
                      "optimizer_steps": result["optimizer_steps"],
                      "barrier": str((BASE / "student" / args.dataset / args.order / "STREAM_BARRIER.json").relative_to(ROOT)),
                      "GT_read": False}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        fail = BASE / "FAILURE_CAPTURE.json"
        if not fail.exists():
            try:
                common.write(fail, {
                    "schema": "stvg-reference-carry-failure-v1",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "GT_read": False,
                    "time": time.time(),
                })
            except Exception:
                pass
        raise
