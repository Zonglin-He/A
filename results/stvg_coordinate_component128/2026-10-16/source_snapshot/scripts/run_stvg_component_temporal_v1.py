#!/usr/bin/env python3
"""Run the locked 128-source CPU temporal-component ablation after M2 freeze.

This runner does not load a visual model or labels. It replays the original
512-parameter temporal fit from sealed rewards, requires exact agreement with
the cached fit, then combines each interval with the already sealed M2 spatial
boxes. It is a conditional readout, not a fresh end-to-end decoder pass.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stvg_soft_ep_coordinate_common_v1 import (
    BASE as SEARCH, load_case, read, roster, save_pt, sha, write, guard,
)
from scripts.stvg_component_ablation_common_v1 import BASE
from scripts.stvg_component_temporal_ablation_v1 import VARIANTS, fit as fit_variant

OUT = BASE / "temporal_component"
ROSTER_PATH = SEARCH / "ROSTER_LOCK.json"
SOURCE_BARRIER_PATH = SEARCH / "SOURCE_CAPTURE_BARRIER.json"
REFERENCE_PATH = SEARCH / "REFERENCE_LOCK.json"
TEMPORAL_PATH = SEARCH / "TEMPORAL_LOCK.json"
SELECTION_PATH = ROOT / "artifacts" / "stvg_evidence_constrained_v1" / "TEMPORAL_SELECTION_LOCK.json"
FINAL_PATH = BASE / "FINAL_PARAMETER_LOCK.json"
RUNTIME_PATH = BASE / "TEMPORAL_COMPONENT_RUNTIME.json"
VARIANT_INTERVAL_ARMS = tuple("TS_" + v for v in VARIANTS)
TEMPORAL_ARMS = ("F", "T", "S_M2", "TS_M2", *VARIANT_INTERVAL_ARMS)


def _cid(row: Mapping[str, Any]) -> str:
    value = row.get("probe_alias", row.get("case_id"))
    if not isinstance(value, str) or not value:
        raise ValueError("roster/lock row lacks case identity")
    return value


def _records(path: Path, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    doc = read(path)
    if doc.get("status") != "locked" or doc.get("GT_read") is not False:
        raise RuntimeError(f"{label} must be locked and explicitly GT-free")
    rows = doc.get("records")
    if not isinstance(rows, list):
        raise RuntimeError(f"{label}.records must be a list")
    return doc, rows


def _check_hash_map(doc: Mapping[str, Any], label: str) -> None:
    for key in ("pins", "protected"):
        mapping = doc.get(key)
        if not isinstance(mapping, dict):
            raise RuntimeError(f"{label}.{key} must be a path-to-SHA map")
        for rel, expected in mapping.items():
            path = (ROOT / rel).resolve()
            if ROOT not in path.parents or not path.is_file() or sha(path) != expected:
                raise RuntimeError(f"{label} pin mismatch: {rel}")


def _file_rows(path: Path, expected_count: int, label: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    doc = read(path)
    if (doc.get("status") != "complete" or doc.get("count") != expected_count
            or doc.get("GT_read") is not False):
        raise RuntimeError(f"{label} barrier must be complete, GT-free, count={expected_count}")
    rows = doc.get("files")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise RuntimeError(f"{label} barrier files mismatch")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("case_id", row.get("probe_alias"))
        if not isinstance(key, str) or key in indexed:
            raise RuntimeError(f"{label} barrier has missing/duplicate case ID")
        path_value = row.get("path")
        if not isinstance(path_value, str):
            raise RuntimeError(f"{label} barrier row lacks path")
        file_path = (ROOT / path_value).resolve()
        if ROOT not in file_path.parents or not file_path.is_file() or sha(file_path) != row.get("sha256"):
            raise RuntimeError(f"{label} artifact SHA mismatch: {path_value}")
        if row.get("bytes") is not None and file_path.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"{label} artifact byte mismatch: {path_value}")
        indexed[key] = row
    return doc, indexed


def _verify_locks() -> dict[str, Any]:
    if not FINAL_PATH.is_file():
        raise FileNotFoundError("FINAL_PARAMETER_LOCK.json is not present; temporal runner remains closed")
    final = read(FINAL_PATH)
    if (final.get("status") != "locked" or final.get("GT_read") is not False
            or final.get("search_finished") is not True or final.get("count") != 128):
        raise RuntimeError("FINAL_PARAMETER_LOCK must be locked, search-finished, GT-free, count=128")
    if final.get("roster_sha256") != sha(ROSTER_PATH):
        raise RuntimeError("FINAL_PARAMETER_LOCK does not bind the Coordinate128 roster")
    completion = SEARCH / "SEARCH_COMPLETION006.json"
    if not completion.is_file() or final.get("search_completion_sha256") != sha(completion):
        raise RuntimeError("FINAL_PARAMETER_LOCK does not bind completed search")
    _check_hash_map(final, "FINAL_PARAMETER_LOCK")
    selection = read(SELECTION_PATH)
    if (selection.get("status") != "locked" or selection.get("GT_read") is not False
            or not isinstance(selection.get("variant"), str) or not selection.get("variant")):
        raise RuntimeError("TEMPORAL_SELECTION_LOCK must be locked/GT-free with a selected variant")

    if not RUNTIME_PATH.is_file():
        raise FileNotFoundError("TEMPORAL_COMPONENT_RUNTIME.json is required before execution")
    runtime = read(RUNTIME_PATH)
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise RuntimeError("TEMPORAL_COMPONENT_RUNTIME must be locked/GT-free")
    _check_hash_map(runtime, "TEMPORAL_COMPONENT_RUNTIME")
    runtime_hashes = {**runtime["pins"], **runtime["protected"]}
    required = {
        "scripts/run_stvg_component_temporal_v1.py": sha(Path(__file__)),
        "scripts/stvg_component_temporal_ablation_v1.py": sha(ROOT / "scripts/stvg_component_temporal_ablation_v1.py"),
        str(FINAL_PATH.relative_to(ROOT)): sha(FINAL_PATH),
        str(ROSTER_PATH.relative_to(ROOT)): sha(ROSTER_PATH),
        str(SOURCE_BARRIER_PATH.relative_to(ROOT)): sha(SOURCE_BARRIER_PATH),
        str(REFERENCE_PATH.relative_to(ROOT)): sha(REFERENCE_PATH),
        str(TEMPORAL_PATH.relative_to(ROOT)): sha(TEMPORAL_PATH),
        str(SELECTION_PATH.relative_to(ROOT)): sha(SELECTION_PATH),
    }
    for rel, expected in required.items():
        if runtime_hashes.get(rel) != expected:
            raise RuntimeError(f"TEMPORAL_COMPONENT_RUNTIME must bind {rel}")

    roster_doc, roster_rows = _records(ROSTER_PATH, "ROSTER_LOCK")
    ref_doc, ref_rows = _records(REFERENCE_PATH, "REFERENCE_LOCK")
    temporal_doc, temporal_rows = _records(TEMPORAL_PATH, "TEMPORAL_LOCK")
    source_doc, source_rows = _file_rows(SOURCE_BARRIER_PATH, 128, "SOURCE_CAPTURE")
    for label, doc, rows in (("roster", roster_doc, roster_rows), ("reference", ref_doc, ref_rows), ("temporal", temporal_doc, temporal_rows)):
        if doc.get("count") != 128 or len(rows) != 128:
            raise RuntimeError(f"{label} lock does not contain exactly 128 rows")
    ids = [_cid(r) for r in roster_rows]
    if len(set(ids)) != 128 or set(ids) != set(source_rows):
        raise RuntimeError("source barrier and roster membership differ")
    ref_map = {_cid(x): x for x in ref_rows}
    temporal_map = {_cid(x): x for x in temporal_rows}
    if set(ref_map) != set(ids) or set(temporal_map) != set(ids):
        raise RuntimeError("reference/temporal lock membership differs from roster")
    for rec in roster_rows:
        cid = _cid(rec)
        src = source_rows[cid]
        if src.get("path") != rec.get("source_case_path"):
            raise RuntimeError(f"source path differs from roster for {cid}")
        if ref_map[cid].get("dataset", rec.get("dataset")) != rec.get("dataset"):
            raise RuntimeError(f"reference domain mismatch for {cid}")
        if temporal_map[cid].get("dataset", rec.get("dataset")) != rec.get("dataset"):
            raise RuntimeError(f"temporal domain mismatch for {cid}")
        if ref_map[cid].get("source_case_sha256") not in (None, src.get("source_case_sha256")):
            raise RuntimeError(f"reference source SHA mismatch for {cid}")
    domains = {r.get("dataset") for r in roster_rows}
    if domains != {"hc2", "vidstg"} or not all(
        isinstance(final.get(k), dict) and set(final[k]) == domains
        for k in ("configs", "response_settings", "actual_baseline_trial")
    ):
        raise RuntimeError("FINAL_PARAMETER_LOCK must contain configs/response_settings/actual_baseline_trial for both domains")
    if not all(isinstance(final["actual_baseline_trial"][d], str) and final["actual_baseline_trial"][d] for d in domains):
        raise RuntimeError("actual_baseline_trial entries must be trial IDs")

    # Validate baseline trial runtimes, locks, M2 settings, and complete output seals.
    from scripts import run_stvg_soft_ep_coordinate_trial003 as parent_runner
    parent_contexts: dict[str, Any] = {}
    for ds in sorted(domains):
        trial_id = final["actual_baseline_trial"][ds]
        ctx = parent_runner._validate_locks(trial_id)
        trial_dir = ctx[0]
        design = ctx[4]
        if final["configs"][ds] != design["configs"][ds]["M2"]:
            raise RuntimeError(f"final M2 config differs from baseline trial for {ds}")
        if final["response_settings"][ds] != design["response_settings"][ds]["M2"]:
            raise RuntimeError(f"final response settings differ from baseline trial for {ds}")
        barrier_path = trial_dir / "STUDENT_GLOBAL_BARRIER.json"
        barrier_doc, outputs = _file_rows(barrier_path, 128, f"baseline {ds} STUDENT")
        readback_path = trial_dir / "ROOT_READBACK.json"
        readback = read(readback_path)
        if readback.get("status") != "pass" or readback.get("count") != 128 or readback.get("GT_read") is not False:
            raise RuntimeError(f"baseline trial {trial_id} lacks a passing 128-case pre-GT readback")
        parent_contexts[ds] = {
            "trial_id": trial_id,
            "trial_dir": trial_dir,
            "barrier_path": barrier_path,
            "barrier": barrier_doc,
            "files": outputs,
            "barrier_sha256": sha(barrier_path),
            "readback_sha256": sha(readback_path),
            "design_sha256": sha(trial_dir / "DESIGN_LOCK.json"),
        }
    return {
        "final": final,
        "final_sha256": sha(FINAL_PATH),
        "runtime": runtime,
        "runtime_sha256": sha(RUNTIME_PATH),
        "selection": selection,
        "roster": roster_rows,
        "source_doc": source_doc,
        "source_files": source_rows,
        "references": ref_map,
        "temporals": temporal_map,
        "reference_sha256": sha(REFERENCE_PATH),
        "temporal_sha256": sha(TEMPORAL_PATH),
        "selection_sha256": sha(SELECTION_PATH),
        "source_barrier_sha256": sha(SOURCE_BARRIER_PATH),
        "parent_contexts": parent_contexts,
    }


def _exact_equal(a: Any, b: Any) -> bool:
    """Recursive strict equality including NumPy/Torch dtype and shape."""
    import numpy as np
    try:
        import torch
    except Exception:
        torch = None
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b) and all(_exact_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_exact_equal(x, y) for x, y in zip(a, b))
    if torch is not None and (isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    if isinstance(a, float) or isinstance(b, float):
        return type(a) is type(b) and (a == b or (math.isnan(a) and math.isnan(b)))
    return type(a) is type(b) and a == b


def _candidate_rewards(case: Mapping[str, Any], cached_fit: Mapping[str, Any], teacher_path: Path | None):
    """Recover full candidate reward vector without inventing missing scores."""
    from scripts import stvg_table1_temporal_adapter_v1 as ta
    candidates = case.get("temporal_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("source case has no temporal candidate list")
    indices = cached_fit.get("valid_candidate_indices")
    saved_rewards = cached_fit.get("rewards")
    if not isinstance(indices, (list, tuple)) or saved_rewards is None:
        raise RuntimeError("saved original fit lacks valid_candidate_indices/rewards")
    if hasattr(saved_rewards, "detach"):
        saved_rewards = saved_rewards.detach().cpu().reshape(-1).tolist()
    else:
        saved_rewards = list(saved_rewards)
    if len(indices) != len(saved_rewards) or len(set(int(i) for i in indices)) != len(indices):
        raise RuntimeError("cached reward vector and valid candidate indices differ")
    full: list[float | None] = [None] * len(candidates)
    for ix, value in zip(indices, saved_rewards):
        ix = int(ix)
        if ix < 0 or ix >= len(candidates):
            raise RuntimeError("saved valid candidate index is outside source candidate list")
        full[ix] = float(value)

    provenance = "sealed_temporal_fit_reward_vector"
    if teacher_path is not None and teacher_path.is_file():
        import torch
        teacher_doc = torch.load(teacher_path, map_location="cpu", weights_only=False)
        if teacher_doc.get("GT_read") is not False:
            raise RuntimeError("temporal teacher cache is not explicitly GT-free")
        t = teacher_doc.get("temporal", {})
        orders = t.get("candidate_order")
        responses = t.get("teacher_responses")
        if not isinstance(orders, list) or not isinstance(responses, list) or len(orders) != len(responses):
            raise RuntimeError("teacher cache lacks candidate_order/teacher_responses")
        cids = [x.get("canonical_id", x.get("candidate_id")) for x in candidates]
        lookup = {str(cid): i for i, cid in enumerate(cids)}
        if len(lookup) != len(candidates) or set(map(str, orders)) != set(lookup):
            raise RuntimeError("teacher response candidate IDs do not match source candidates")
        by_index: list[float | None] = [None] * len(candidates)
        for candidate_id, response in zip(orders, responses):
            ix = lookup[str(candidate_id)]
            reward = ta._alignment_reward(str(candidate_id), response)
            by_index[ix] = None if reward is None else float(reward)
        for i, value in enumerate(full):
            if value is None:
                if by_index[i] is not None:
                    raise RuntimeError("cached fit omitted a candidate that has a valid saved teacher reward")
            elif by_index[i] is None or float(value) != float(by_index[i]):
                raise RuntimeError("saved teacher reward differs from original temporal-fit reward")
        provenance = "sealed_fit_rewards_crosschecked_against_teacher_cache"
    return full, provenance


def _verify_round1_temporal_lineage(cid: str, dataset: str, cache: Mapping[str, Any],
                                   source_case_sha256: str, source_file_sha256: str,
                                   selection: Mapping[str, Any]) -> dict[str, Any]:
    """Bind legacy round2 temporal caches through their saved round1 parent.

    The old cache does not carry ``source_case_sha256``. Its provenance is the
    round1 student artifact SHA; that artifact in turn binds the current
    source-case content SHA and source-file SHA. This follows the original
    confirmation preparation chain instead of inventing a missing lock field.
    """
    import torch

    if cache.get("source_round1_sha256") is None:
        raise RuntimeError(f"legacy temporal cache lacks source_round1_sha256 for {cid}")
    round1_dir = ROOT / "artifacts" / "stvg_evidence_constrained_v1" / "round1"
    round1_path = round1_dir / "student" / f"{cid}.pt"
    barrier_path = round1_dir / "GLOBAL_PREDICTION_BARRIER.json"
    if not round1_path.is_file() or not barrier_path.is_file():
        raise FileNotFoundError(f"legacy temporal cache parent is incomplete for {cid}")
    barrier = read(barrier_path)
    if (barrier.get("status") != "complete" or barrier.get("GT_read") is not False
            or barrier.get("count") != len(barrier.get("files", []))
            or selection.get("round1_global_sha256") != sha(barrier_path)):
        raise RuntimeError(f"round1 parent barrier is not complete/GT-free for {cid}")
    expected_rel = str(round1_path.relative_to(ROOT))
    matches = [row for row in barrier.get("files", []) if row.get("path") == expected_rel]
    if len(matches) != 1:
        raise RuntimeError(f"round1 parent barrier lacks unique student row for {cid}")
    row = matches[0]
    actual_round1_sha = sha(round1_path)
    if (actual_round1_sha != cache.get("source_round1_sha256")
            or actual_round1_sha != row.get("sha256")):
        raise RuntimeError(f"legacy temporal cache round1 parent SHA mismatch for {cid}")
    if row.get("bytes") is not None and round1_path.stat().st_size != int(row["bytes"]):
        raise RuntimeError(f"round1 parent bytes mismatch for {cid}")
    parent = torch.load(round1_path, map_location="cpu", weights_only=False)
    if (parent.get("GT_read") is not False or parent.get("source_restored") is not True
            or parent.get("case_id") != cid or parent.get("dataset") != dataset
            or parent.get("source_case_sha256") != source_case_sha256
            or parent.get("source_file_sha256") != source_file_sha256):
        raise RuntimeError(f"round1 parent does not bind current source case/file for {cid}")
    variant = selection.get("variant")
    arm = parent.get("arms", {}).get(f"T_{variant}")
    if not isinstance(arm, dict) or list(arm.get("interval", [])) != list(cache.get("selected_interval", [])):
        raise RuntimeError(f"legacy temporal interval differs from its selected round1 arm for {cid}")
    return parent


def _load_temporal_cache(rec: Mapping[str, Any], cid: str, lock_row: Mapping[str, Any],
                         source_case_sha256: str, source_file_sha256: str,
                         selection: Mapping[str, Any], selection_sha256: str):
    import torch
    path = ROOT / rec["cached_temporal_path"] if rec.get("cached_temporal_path") else SEARCH / "confirmation" / "temporal" / f"{cid}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing cached Temporal fit for {cid}")
    file_sha = sha(path)
    if lock_row.get("source_temporal_sha256") != file_sha:
        raise RuntimeError(f"TEMPORAL_LOCK cache SHA mismatch for {cid}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("GT_read") is not False or cache.get("case_id") != cid or cache.get("dataset") != rec.get("dataset"):
        raise RuntimeError(f"cached Temporal fit identity/GT binding mismatch for {cid}")
    if cache.get("temporal_selection_sha256") != selection_sha256:
        raise RuntimeError(f"cached Temporal fit selection-lock SHA mismatch for {cid}")
    cached_source_case_sha = cache.get("source_case_sha256")
    if cached_source_case_sha is not None and cached_source_case_sha != source_case_sha256:
        raise RuntimeError(f"cached Temporal fit source-case content SHA mismatch for {cid}")
    round1_parent = None
    if cache.get("source_round1_sha256") is not None:
        round1_parent = _verify_round1_temporal_lineage(cid, str(rec.get("dataset")), cache,
                                                        source_case_sha256, source_file_sha256, selection)
    elif cached_source_case_sha != source_case_sha256:
        raise RuntimeError(f"cached Temporal fit lacks a valid source-case or round1 lineage for {cid}")
    fit = cache.get("fits", {}).get("offset_trust")
    if fit is not None and (not isinstance(fit, dict) or fit.get("variant") != "offset_trust" or fit.get("GT_read") is not False):
        raise RuntimeError(f"cached Temporal fit has an invalid original offset_trust record for {cid}")
    if list(cache.get("selected_interval", [])) != list(lock_row.get("selected_interval", [])):
        raise RuntimeError(f"TEMPORAL_LOCK interval differs from cached fit for {cid}")
    if fit is not None and list(fit.get("final_interval", [])) != list(lock_row.get("selected_interval", [])):
        raise RuntimeError(f"cached fit final interval differs from TEMPORAL_LOCK for {cid}")
    parent = ROOT / rec["cached_parent"] if rec.get("cached_parent") else SEARCH / "confirmation"
    teacher_path = parent / "temporal_teacher" / f"{cid}.pt"
    teacher_sha = cache.get("source_temporal_teacher_sha256")
    if teacher_sha is not None and not teacher_path.is_file():
        raise RuntimeError(f"cached temporal-teacher artifact is missing for {cid}")
    if teacher_path.is_file() and teacher_sha != sha(teacher_path):
        raise RuntimeError(f"cached temporal-teacher SHA mismatch for {cid}")
    if fit is None and cache.get("fits") != {}:
        raise RuntimeError(f"cached Temporal fit has no offset_trust entry but contains unexpected fits for {cid}")
    return path, file_sha, cache, fit, teacher_path, round1_parent


def _validated_no_fit_fallback(case: Mapping[str, Any], cache: Mapping[str, Any],
                                teacher_path: Path, source_file_sha256: str,
                                selected_interval: Any, round1_parent: Mapping[str, Any] | None = None,
                                selection: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate the explicit no-four-candidate native fallback without fabricating a fit."""
    import torch
    if teacher_path.is_file():
        teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
        return _validate_no_fit_fallback_record(case, cache, teacher, source_file_sha256, selected_interval)

    # Legacy round2 caches inherit the unavailable status through the sealed
    # round1 student artifact and its old-temporal source SHA.
    if round1_parent is None or cache.get("source_round1_sha256") is None:
        raise RuntimeError(f"no-fit fallback lacks a teacher record or round1 parent: {case.get('case_id')}")
    cid = case.get("case_id")
    old_temporal_path = ROOT / "artifacts" / "stvg_weightfree_tuning_v1" / "confirmation003" / "temporal" / f"{cid}.pt"
    if not old_temporal_path.is_file():
        raise FileNotFoundError(f"round1 no-fit fallback lacks its old temporal input: {cid}")
    if round1_parent.get("old_temporal_file_sha256") != sha(old_temporal_path):
        raise RuntimeError(f"round1 no-fit fallback old temporal SHA mismatch for {cid}")
    old_doc = torch.load(old_temporal_path, map_location="cpu", weights_only=False)
    temporal = old_doc.get("temporal", {})
    if (round1_parent.get("fits") != {}
            or old_doc.get("GT_read") is not False or old_doc.get("case_id") != cid
            or old_doc.get("dataset") != case.get("dataset")
            or old_doc.get("source_file_sha256") != source_file_sha256
            or temporal.get("GT_read") is not False
            or temporal.get("status") != "unavailable_temporal_input_native_fallback"
            or temporal.get("reason") != "candidate_pool_or_role_count_not_four"
            or int(temporal.get("optimizer_steps", -1)) != 0
            or list(temporal.get("final_interval", [])) != list(selected_interval)):
        raise RuntimeError(f"round1 inherited no-fit fallback is invalid for {cid}")
    selected_variant = (selection or {}).get("variant")
    selected_arm = round1_parent.get("arms", {}).get(f"T_{selected_variant}")
    if (not isinstance(selected_arm, dict)
            or list(selected_arm.get("interval", [])) != list(selected_interval)
            or list(round1_parent.get("arms", {}).get("F", {}).get("interval", [])) != list(case.get("Native_interval", []))):
        raise RuntimeError(f"round1 inherited fallback differs from Native interval for {cid}")
    native = list(case.get("Native_interval", []))
    if not (native == list(cache.get("old_interval", []))
            == list(cache.get("selected_interval", [])) == list(selected_interval)):
        raise RuntimeError(f"round1 inherited fallback is not the exact Source Native interval for {cid}")
    return {
        "status": "unavailable_temporal_input_native_fallback",
        "reason": "candidate_pool_or_role_count_not_four",
        "selected_interval": list(selected_interval),
        "optimizer_steps": 0,
        "GT_read": False,
        "inherited_through": "round1_student_and_old_temporal_sha_chain",
    }


def _validate_no_fit_fallback_record(case: Mapping[str, Any], cache: Mapping[str, Any],
                                     teacher: Mapping[str, Any], source_file_sha256: str,
                                     selected_interval: Any) -> dict[str, Any]:
    """Pure schema/lineage validator, also exercised by synthetic CPU contracts."""
    temporal = teacher.get("temporal", {})
    cid = case.get("case_id")
    expected_interval = list(selected_interval)
    if (cache.get("fits") != {}
            or teacher.get("GT_read") is not False or teacher.get("case_id") != cid
            or teacher.get("dataset") != case.get("dataset")
            or teacher.get("source_file_sha256") != source_file_sha256
            or temporal.get("GT_read") is not False
            or temporal.get("status") != "unavailable_temporal_input_native_fallback"
            or temporal.get("reason") != "candidate_pool_or_role_count_not_four"
            or int(temporal.get("optimizer_steps", -1)) != 0
            or list(temporal.get("final_interval", [])) != expected_interval):
        raise RuntimeError(f"no-fit teacher fallback schema/binding mismatch for {cid}")
    native = list(case.get("Native_interval", []))
    old = list(cache.get("old_interval", []))
    selected = list(cache.get("selected_interval", []))
    if not (native == old == selected == expected_interval):
        raise RuntimeError(f"no-fit fallback is not the exact Source Native interval for {cid}")
    return {
        "status": "unavailable_temporal_input_native_fallback",
        "reason": "candidate_pool_or_role_count_not_four",
        "selected_interval": expected_interval,
        "optimizer_steps": 0,
        "GT_read": False,
    }


def _load_m2_parent(cid: str, rec: Mapping[str, Any], context: Mapping[str, Any]):
    import torch
    path = context["trial_dir"] / "student" / f"{cid}.pt"
    entry = context["files"].get(cid)
    if entry is None or entry.get("path") != str(path.relative_to(ROOT)):
        raise RuntimeError(f"M2 baseline barrier lacks exact student path for {cid}")
    if sha(path) != entry.get("sha256"):
        raise RuntimeError(f"M2 baseline student SHA mismatch for {cid}")
    receipt_path = path.with_suffix(".receipt.json")
    if not receipt_path.is_file() or sha(receipt_path) != entry.get("receipt_sha256"):
        raise RuntimeError(f"M2 baseline receipt SHA mismatch for {cid}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("case_id") != cid or payload.get("dataset") != rec.get("dataset")
            or payload.get("GT_read") is not False or payload.get("source_restored") is not True):
        raise RuntimeError(f"M2 baseline identity/GT/source-restoration mismatch for {cid}")
    m2fit = payload.get("fits", {}).get("M2")
    m2readout = payload.get("readouts", {}).get("M2")
    if not isinstance(m2fit, dict) or not isinstance(m2readout, dict):
        raise RuntimeError(f"M2 baseline fit/readout missing for {cid}")
    boxes = payload.get("arms", {}).get("TS_M2", {}).get("boxes")
    interval = payload.get("arms", {}).get("TS_M2", {}).get("interval")
    if boxes is None or interval is None:
        raise RuntimeError(f"M2 baseline TS_M2 arm missing for {cid}")
    if list(interval) != list(payload.get("selected_interval", [])):
        raise RuntimeError(f"M2 baseline arm interval differs from selected interval for {cid}")
    if not torch.equal(m2readout["boxes"].detach().cpu(), m2fit["final_cached_decoder_boxes"].detach().cpu()):
        raise RuntimeError(f"M2 readout differs from its cached decoder boxes for {cid}")
    from scripts.run_stvg_adjacent_motion_student_v1 import _xyxy
    converted = _xyxy(m2readout["boxes"])
    import numpy as np
    if np.asarray(converted).shape != np.asarray(boxes).shape or not np.array_equal(np.asarray(converted), np.asarray(boxes)):
        raise RuntimeError(f"M2 TS_M2 arm boxes differ from full cached readout for {cid}")
    return path, entry, payload, np.asarray(boxes).copy(), list(interval)


def build_arms(native_boxes: Any, native_interval: Any, selected_interval: Any,
               m2_boxes: Any, variant_intervals: Mapping[str, Any]) -> dict[str, Any]:
    """Build conditional arms while sharing exact immutable box arrays."""
    import numpy as np
    native = np.asarray(native_boxes).copy()
    m2 = np.asarray(m2_boxes).copy()
    if native.ndim != 2 or native.shape[1] != 4 or m2.shape != native.shape:
        raise ValueError("native and M2 boxes must share [source_frames,4] shape")
    arms: dict[str, Any] = {
        "F": {"boxes": native, "interval": list(native_interval)},
        "T": {"boxes": native, "interval": list(selected_interval)},
        "S_M2": {"boxes": m2, "interval": list(native_interval)},
        "TS_M2": {"boxes": m2, "interval": list(selected_interval)},
    }
    if set(variant_intervals) != set(VARIANTS):
        raise ValueError("variant interval set must match all five locked temporal variants")
    for variant in VARIANTS:
        arms["TS_" + variant] = {"boxes": m2, "interval": list(variant_intervals[variant])}
    return arms


def _write_once(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(f"refusing to overwrite sealed/partial artifact: {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _readback_one(cid: str, source_sha: str, temporal_record: Mapping[str, Any],
                  reference_record: Mapping[str, Any], source_model_sha: str,
                  expected_fit_names: set[str], path: Path, expected_sha: str,
                  expected_no_fit_reason: str | None = None):
    import torch
    if sha(path) != expected_sha:
        raise RuntimeError(f"new component output SHA changed after save: {cid}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("GT_read") is not False or payload.get("case_id") != cid:
        raise RuntimeError(f"saved component case identity/GT flag mismatch: {cid}")
    if (payload.get("source_case_sha256") != source_sha
            or payload.get("source_model_state_sha256") != source_model_sha
            or payload.get("temporal_record") != dict(temporal_record)
            or payload.get("reference_record") != dict(reference_record)):
        raise RuntimeError(f"saved component provenance mismatch: {cid}")
    if set(payload.get("fits", {})) != expected_fit_names:
        raise RuntimeError(f"saved fit variants mismatch: {cid}")
    status = payload.get("temporal_update_status", {})
    if expected_no_fit_reason is None:
        if status.get("status") != "fitted" or set(status.get("fit_variants", [])) != set(VARIANTS):
            raise RuntimeError(f"saved temporal fitted-status record mismatch: {cid}")
    else:
        if (status.get("status") != "no_update_unknown"
                or status.get("reason") != expected_no_fit_reason
                or status.get("fit_variants") != []
                or set(status.get("no_update_temporal_arms", [])) != set(VARIANT_INTERVAL_ARMS)):
            raise RuntimeError(f"saved no-fit fallback status mismatch: {cid}")
        selected = list(temporal_record.get("selected_interval", []))
        for arm_name in VARIANT_INTERVAL_ARMS:
            if list(payload["arms"][arm_name].get("interval", [])) != selected:
                raise RuntimeError(f"no-fit fallback changed the frozen interval for {cid}/{arm_name}")
    if set(payload.get("arms", {})) != set(TEMPORAL_ARMS):
        raise RuntimeError(f"saved temporal arm schema mismatch: {cid}")
    return payload


def run() -> dict[str, Any]:
    import torch
    from scripts import stvg_evidence_temporal_fit_v1 as basefit
    from scripts import stvg_table1_temporal_adapter_v1 as temporal_adapter

    lock = _verify_locks()
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("temporal component output exists; no overwrite or implicit resume")
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    ref_by_id = lock["references"]
    temporal_by_id = lock["temporals"]
    context_by_domain = lock["parent_contexts"]
    rows = lock["roster"]
    output_rows = []
    audit_rows = []
    baseline_comparison_count = 0
    no_fit_fallback_count = 0

    for rec in rows:
        cid, ds = _cid(rec), rec["dataset"]
        case, source_path = load_case(rec)
        srcfile = lock["source_files"][cid]
        if (source_path != ROOT / rec["source_case_path"] or sha(source_path) != srcfile["sha256"]
                or srcfile.get("source_case_sha256") != case.get("source_case_sha256")):
            raise RuntimeError(f"Coordinate128 Source barrier mismatch for {cid}")
        ref = ref_by_id[cid]
        temp_record = temporal_by_id[cid]
        if case.get("source_case_sha256") != ref.get("source_case_sha256"):
            raise RuntimeError(f"reference/source case mismatch for {cid}")
        if case.get("dataset") != ds or case.get("GT_read") is not False:
            raise RuntimeError(f"source case identity/GT mismatch for {cid}")

        cache_path, cache_sha, cache_doc, cached_fit, teacher_path, round1_parent = _load_temporal_cache(
            rec, cid, temp_record, str(case.get("source_case_sha256")), sha(source_path),
            lock["selection"], lock["selection_sha256"])
        hidden = None
        recomputed_baseline = None
        no_fit_record = None
        if cached_fit is None:
            no_fit_record = _validated_no_fit_fallback(
                case, cache_doc, teacher_path, sha(source_path), temp_record["selected_interval"],
                round1_parent=round1_parent, selection=lock["selection"])
            reward_provenance = "unavailable_temporal_input_no_fit_no_reward_fabricated"
            temporal_results: dict[str, Any] = {}
            variant_intervals = {name: list(temp_record["selected_interval"]) for name in VARIANTS}
            no_fit_fallback_count += 1
        else:
            rewards, reward_provenance = _candidate_rewards(case, cached_fit, teacher_path)
            hidden = temporal_adapter._hidden_from_source_case(case)
            recomputed_baseline = fit_variant(case, hidden, case["temporal_candidates"], rewards, "offset_trust")
            if not _exact_equal(recomputed_baseline, cached_fit):
                raise RuntimeError(f"exact offset_trust baseline replay differs from sealed cache: {cid}")
            baseline_comparison_count += 1

            temporal_results = {"offset_trust": recomputed_baseline}
            for variant in VARIANTS:
                if variant == "offset_trust":
                    continue
                temporal_results[variant] = fit_variant(case, hidden, case["temporal_candidates"], rewards, variant)
            variant_intervals = {name: list(result["final_interval"]) for name, result in temporal_results.items()}
            if list(variant_intervals["offset_trust"]) != list(temp_record["selected_interval"]):
                raise RuntimeError(f"baseline fit interval differs from frozen TEMPORAL_LOCK for {cid}")

        parent_path, parent_entry, parent_payload, m2_boxes, m2_interval = _load_m2_parent(cid, rec, context_by_domain[ds])
        if m2_interval != list(temp_record["selected_interval"]):
            raise RuntimeError(f"M2 baseline and fixed Temporal interval differ for {cid}")
        if parent_payload.get("source_case_sha256") != case.get("source_case_sha256"):
            raise RuntimeError(f"M2 parent Source SHA mismatch for {cid}")
        if parent_payload.get("reference_lock_sha256") != lock["reference_sha256"]:
            raise RuntimeError(f"M2 parent reference lock SHA mismatch for {cid}")
        if parent_payload.get("temporal_lock_sha256") != lock["temporal_sha256"]:
            raise RuntimeError(f"M2 parent temporal lock SHA mismatch for {cid}")
        if parent_payload.get("source_restored") is not True:
            raise RuntimeError(f"M2 parent did not verify restored source state for {cid}")

        arms = build_arms(case["native_boxes_xyxy_normalized"], case["Native_interval"],
                          temp_record["selected_interval"], m2_boxes, variant_intervals)
        payload = {
            "schema": "stvg-component-temporal-case-v1",
            "case_id": cid,
            "dataset": ds,
            "arms": arms,
            "fits": temporal_results,
            "temporal_update_status": ({
                "status": "no_update_unknown",
                "reason": no_fit_record["reason"],
                "fit_variants": [],
                "no_update_temporal_arms": list(VARIANT_INTERVAL_ARMS),
                "fallback_source": no_fit_record.get("inherited_through", "sealed_teacher_native_interval"),
                "GT_read": False,
            } if no_fit_record is not None else {
                "status": "fitted",
                "fit_variants": list(VARIANTS),
                "GT_read": False,
            }),
            "temporal_no_fit_record": no_fit_record,
            "frame_ids": list(map(int, case["frame_ids"])),
            "reference_record": copy.deepcopy(ref),
            "temporal_record": copy.deepcopy(temp_record),
            "source_case_sha256": case["source_case_sha256"],
            "source_file_sha256": sha(source_path),
            "source_model_state_sha256": case["source_model_state_sha256"],
            "pixel_sha256": case.get("pixel_sha256"),
            "temporal_cache_path": str(cache_path.relative_to(ROOT)),
            "temporal_cache_sha256": cache_sha,
            "temporal_reward_provenance": reward_provenance,
            "baseline_temporal_fit_exactly_replayed": no_fit_record is None,
            "baseline_spatial_trial_id": context_by_domain[ds]["trial_id"],
            "baseline_spatial_student_path": str(parent_path.relative_to(ROOT)),
            "baseline_spatial_student_sha256": parent_entry["sha256"],
            "baseline_spatial_readout_is_cached_fullmodel_readout": True,
            "conditional_TS_readouts_reuse_M2_boxes": True,
            "full_visual_model_decode_replayed": False,
            "source_restored": True,
            "GT_read": False,
            "runtime_sha256": lock["runtime_sha256"],
            "final_parameter_lock_sha256": lock["final_sha256"],
        }
        out_path = OUT / "student" / f"{cid}.pt"
        if out_path.exists() or out_path.with_suffix(".receipt.json").exists():
            raise FileExistsError(f"case output already exists; no implicit retry: {cid}")
        item = save_pt(out_path, payload)
        receipt_path = out_path.with_suffix(".receipt.json")
        _write_once(receipt_path, {
            "schema": "stvg-component-temporal-case-receipt-v1",
            "status": "complete",
            "case_id": cid,
            "dataset": ds,
            "sha256": sha(out_path),
            "bytes": out_path.stat().st_size,
            "file": item,
            "source_case_sha256": payload["source_case_sha256"],
            "temporal_cache_sha256": cache_sha,
            "baseline_spatial_student_sha256": parent_entry["sha256"],
            "GT_read": False,
        })
        _readback_one(cid, payload["source_case_sha256"], temp_record, ref,
                      payload["source_model_state_sha256"], set(temporal_results), out_path, sha(out_path),
                      no_fit_record["reason"] if no_fit_record is not None else None)
        output_rows.append({
            "case_id": cid,
            "dataset": ds,
            "path": str(out_path.relative_to(ROOT)),
            "sha256": sha(out_path),
            "bytes": out_path.stat().st_size,
            "receipt_path": str(receipt_path.relative_to(ROOT)),
            "receipt_sha256": sha(receipt_path),
            "source_case_sha256": payload["source_case_sha256"],
            "temporal_cache_sha256": cache_sha,
            "baseline_spatial_student_sha256": parent_entry["sha256"],
        })
        audit_rows.append({"case_id": cid, "dataset": ds,
                           "baseline_fit_exact": no_fit_record is None,
                           "baseline_interval_exact": True,
                           "spatial_source_sha_exact": True,
                           "all_fit_variants": len(temporal_results),
                           "no_fit_fallback_reason": no_fit_record["reason"] if no_fit_record else None,
                           "temporal_cache_sha256": cache_sha,
                           "source_case_sha256": payload["source_case_sha256"]})
        del case, payload, parent_payload, hidden, temporal_results, recomputed_baseline
        if (len(output_rows) % 8) == 0:
            print("TEMPORAL_COMPONENT", len(output_rows), "/ 128", flush=True)

    ids = [_cid(r) for r in rows]
    if (len(output_rows) != 128 or {r["case_id"] for r in output_rows} != set(ids)
            or baseline_comparison_count + no_fit_fallback_count != 128):
        raise RuntimeError("temporal component run did not cover all 128 locked cases")
    counts = {ds: sum(r["dataset"] == ds for r in output_rows) for ds in ("hc2", "vidstg")}
    if counts != {"hc2": 64, "vidstg": 64}:
        raise RuntimeError(f"unexpected domain counts: {counts}")

    barrier_path = OUT / "TEMPORAL_COMPONENT_GLOBAL_BARRIER.json"
    barrier = {
        "schema": "stvg-component-temporal-global-barrier-v1",
        "status": "complete",
        "count": 128,
        "domain_counts": counts,
        "files": output_rows,
        "bindings": {
            "final_parameter_lock_sha256": lock["final_sha256"],
            "temporal_component_runtime_sha256": lock["runtime_sha256"],
            "roster_sha256": sha(ROSTER_PATH),
            "source_capture_barrier_sha256": lock["source_barrier_sha256"],
            "reference_lock_sha256": lock["reference_sha256"],
            "temporal_lock_sha256": lock["temporal_sha256"],
            "fit_helper_sha256": sha(ROOT / "scripts/stvg_component_temporal_ablation_v1.py"),
            "runner_sha256": sha(Path(__file__)),
            "baseline_spatial_trials": {ds: lock["parent_contexts"][ds]["trial_id"] for ds in counts},
            "baseline_spatial_barriers": {ds: lock["parent_contexts"][ds]["barrier_sha256"] for ds in counts},
            "baseline_spatial_root_readbacks": {ds: lock["parent_contexts"][ds]["readback_sha256"] for ds in counts},
        },
        "fit_variant_ids": list(VARIANTS),
        "fitted_case_count": baseline_comparison_count,
        "explicit_no_fit_fallback_count": no_fit_fallback_count,
        "conditional_TS_spatial_reference": "sealed actual-baseline M2 xyxy boxes reused for all temporal variants",
        "full_visual_model_decode_replayed": False,
        "baseline_temporal_fit_exact_replay_count": baseline_comparison_count,
        "CPU_only": True,
        "GPU_used": False,
        "model_loaded": False,
        "GT_read": False,
        "wall_seconds": time.perf_counter() - start,
    }
    _write_once(barrier_path, barrier)

    # Reopen every serialized case/receipt and independently recheck chain data.
    readback_count = 0
    receipts = []
    for row in output_rows:
        path = ROOT / row["path"]
        receipt_path = ROOT / row["receipt_path"]
        if sha(path) != row["sha256"] or sha(receipt_path) != row["receipt_sha256"]:
            raise RuntimeError(f"case or receipt changed during final readback: {row['case_id']}")
        receipt = read(receipt_path)
        if receipt.get("status") != "complete" or receipt.get("sha256") != row["sha256"] or receipt.get("GT_read") is not False:
            raise RuntimeError(f"case receipt mismatch: {row['case_id']}")
        receipts.append({"path": row["receipt_path"], "sha256": row["receipt_sha256"], "bytes": receipt_path.stat().st_size})
        readback_count += 1
    readback_path = OUT / "ROOT_TEMPORAL_COMPONENT_READBACK.json"
    _write_once(readback_path, {
        "schema": "stvg-component-temporal-cpu-readback-v1",
        "status": "pass",
        "scope": "saved baseline temporal fit exact CPU replay, variant output custody, source/reference/temporal bindings, and 128-case closure; no independent full visual-model decode",
        "count": readback_count,
        "domain_counts": counts,
        "baseline_fit_exact_replay_count": baseline_comparison_count,
        "planned_fit_variants": list(VARIANTS),
        "fitted_case_count": baseline_comparison_count,
        "explicit_no_fit_fallback_count": no_fit_fallback_count,
        "files": [dict(path=x["path"], sha256=x["sha256"], bytes=x["bytes"]) for x in output_rows],
        "receipts": receipts,
        "temporal_component_barrier_sha256": sha(barrier_path),
        "final_parameter_lock_sha256": lock["final_sha256"],
        "runtime_sha256": lock["runtime_sha256"],
        "GT_read": False,
        "CPU_only": True,
        "model_loaded": False,
        "GPU_used": False,
        "actual_scoring_performed": False,
    })

    prediction_barrier_path = OUT / "PREDICTION_TEMPORAL_COMPONENT_BARRIER.json"
    _write_once(prediction_barrier_path, {
        "schema": "stvg-component-temporal-prediction-barrier-v1",
        "status": "complete",
        "count": 128,
        "GT_read": False,
        "temporal_component_barrier_sha256": sha(barrier_path),
        "root_temporal_component_readback_sha256": sha(readback_path),
        "runtime_sha256": lock["runtime_sha256"],
        "final_parameter_lock_sha256": lock["final_sha256"],
        "files": [dict(path=x["path"], sha256=x["sha256"], bytes=x["bytes"]) for x in output_rows],
        "receipts": receipts,
    })
    _write_once(OUT / "CPU_COMPLETION.json", {
        "schema": "stvg-component-temporal-cpu-completion-v1",
        "status": "complete",
        "count": 128,
        "domain_counts": counts,
        "planned_fit_variants": list(VARIANTS),
        "baseline_fit_exact_replay_count": baseline_comparison_count,
        "explicit_no_fit_fallback_count": no_fit_fallback_count,
        "case_readback_count": readback_count,
        "GT_read": False,
        "CPU_only": True,
        "model_loaded": False,
        "GPU_used": False,
        "actual_gt_scoring_performed": False,
        "full_visual_model_decode_replayed": False,
        "prediction_barrier_sha256": sha(prediction_barrier_path),
    })
    return {"count": readback_count, "domains": counts, "baseline_exact": baseline_comparison_count}


def contracts() -> dict[str, Any]:
    import numpy as np
    from scripts.stvg_evidence_temporal_fit_v1 import _synthetic_case
    import scripts.stvg_evidence_temporal_fit_v1 as basefit

    case, hidden, candidates, rewards = basefit._synthetic_case(seed=8017)
    case = dict(case)
    case["native_boxes_xyxy_normalized"] = np.array([[.2, .2, .4, .5], [.3, .25, .5, .65]], dtype=np.float64)
    case["Native_interval"] = [10.0, 18.0]
    m2 = case["native_boxes_xyxy_normalized"] + np.array([[.01, 0, .01, 0], [0, .01, 0, .01]])
    results = {v: fit_variant(case, hidden, candidates, rewards, v) for v in VARIANTS}
    arms = build_arms(case["native_boxes_xyxy_normalized"], case["Native_interval"],
                      results["offset_trust"]["final_interval"], m2,
                      {k: results[k]["final_interval"] for k in VARIANTS})
    assert set(arms) == set(TEMPORAL_ARMS)
    assert np.array_equal(arms["F"]["boxes"], arms["T"]["boxes"])
    assert np.array_equal(arms["S_M2"]["boxes"], arms["TS_projection_only"]["boxes"])
    assert arms["F"]["interval"] == case["Native_interval"]
    assert arms["TS_projection_only"]["interval"] == case["Projection_interval"]
    original = basefit.OFFSET_TRUST_CONFIG
    _ = fit_variant(case, hidden, candidates, rewards, "shared_offset_candidate")
    assert basefit.OFFSET_TRUST_CONFIG is original
    fallback_case = {"case_id": "SYN_NOFIT", "dataset": "hc2", "Native_interval": [71, 330]}
    fallback_cache = {"fits": {}, "old_interval": [71, 330], "selected_interval": [71, 330]}
    fallback_teacher = {
        "case_id": "SYN_NOFIT", "dataset": "hc2", "source_file_sha256": "source-file-sha",
        "GT_read": False,
        "temporal": {
            "status": "unavailable_temporal_input_native_fallback",
            "reason": "candidate_pool_or_role_count_not_four",
            "final_interval": [71, 330], "optimizer_steps": 0, "GT_read": False,
        },
    }
    fallback = _validate_no_fit_fallback_record(
        fallback_case, fallback_cache, fallback_teacher, "source-file-sha", [71, 330])
    assert fallback["reason"] == "candidate_pool_or_role_count_not_four"
    try:
        _validate_no_fit_fallback_record(
            fallback_case, fallback_cache, fallback_teacher, "wrong-source-file-sha", [71, 330])
    except RuntimeError:
        pass
    else:
        raise AssertionError("no-fit fallback must reject a mismatched source file SHA")
    try:
        _validate_no_fit_fallback_record(
            dict(fallback_case, Native_interval=[72, 330]), fallback_cache,
            fallback_teacher, "source-file-sha", [71, 330])
    except RuntimeError:
        pass
    else:
        raise AssertionError("no-fit fallback must reject a non-Native interval")
    return {
        "schema": "stvg-component-temporal-runner-cpu-contracts-v1",
        "status": "pass",
        "CPU_only": True,
        "synthetic_only": True,
        "GT_read": False,
        "GPU_used": False,
        "model_loaded": False,
        "actual_source_cases_loaded": 0,
        "actual_score_rows_loaded": 0,
        "variant_count": len(VARIANTS),
        "conditional_arm_count": len(TEMPORAL_ARMS),
        "no_fit_native_fallback_contract": True,
        "baseline_replay_deterministic": True,
        "native_and_m2_box_mapping": "same source grid reused; only interval varies",
        "upstream_globals_restored": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", action="store_true", help="run synthetic CPU-only contracts")
    parser.add_argument("--run", action="store_true", help="execute only after root locks are complete")
    args = parser.parse_args()
    if args.contracts == args.run:
        parser.error("choose exactly one of --contracts or --run")
    if args.contracts:
        print(json.dumps(contracts(), indent=2, sort_keys=True))
    else:
        sys.addaudithook(guard)
        print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
