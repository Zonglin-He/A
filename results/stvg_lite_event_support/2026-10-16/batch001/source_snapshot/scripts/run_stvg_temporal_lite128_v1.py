#!/usr/bin/env python3
"""CPU-only 128-source Temporal-Lite replay and conditional readout builder.

Actual cache processing is closed unless the root scope has reviewed the code
and created the matching locked runtime record. This runner never reads GT,
loads media, scores predictions, or re-runs a visual model.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stvg_component_ablation_common_v1 import read, sha
from scripts.stvg_soft_ep_coordinate_common_v1 import save_pt
from scripts import run_stvg_component_temporal005 as legacy
from scripts import stvg_evidence_temporal_fit_v1 as basefit
from scripts import stvg_temporal_lite_fit_v1 as temporal_lite

SCOPE = ROOT / "artifacts" / "stvg_lite_event_support_v1"
OUT = SCOPE / "temporal"
DESIGN_LOCK = SCOPE / "DESIGN_LOCK.json"
RUNTIME_LOCK = OUT / "TEMPORAL_LITE_RUNTIME_LOCK.json"
FIT_SCRIPT = ROOT / "scripts" / "stvg_temporal_lite_fit_v1.py"
RUN_SCRIPT = Path(__file__).resolve()
READBACK_SCRIPT = ROOT / "scripts" / "readback_stvg_temporal_lite128_v1.py"

FIT_COUNT = 111
NO_FIT_COUNT = 17
TOTAL_COUNT = 128
CASE_ARMS = ("F", "T", "S_M2", "TS_M2", "TS_offset_trust", "TS_temporal_lite")
RUNTIME_PIN_PATHS = (
    "scripts/stvg_temporal_lite_fit_v1.py",
    "scripts/run_stvg_temporal_lite128_v1.py",
    "scripts/readback_stvg_temporal_lite128_v1.py",
    "scripts/score_stvg_temporal_lite128_v1.py",
    "scripts/run_stvg_component_temporal005.py",
    "scripts/stvg_component_temporal_ablation_v1.py",
    "scripts/stvg_evidence_temporal_fit_v1.py",
    "scripts/stvg_table1_temporal_adapter_v1.py",
    "scripts/stvg_component_ablation_common_v1.py",
)


def _write_once(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(f"refusing to overwrite existing/partial artifact: {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _input_bindings(lock: Mapping[str, Any]) -> dict[str, Any]:
    a000 = lock["a000_context"]
    return {
        "design_lock_sha256": sha(DESIGN_LOCK),
        "final_parameter_lock_sha256": lock["final_sha256"],
        "roster_lock_sha256": sha(legacy.ROSTER_PATH),
        "source_capture_barrier_sha256": lock["source_barrier_sha256"],
        "reference_lock_sha256": lock["reference_sha256"],
        "temporal_lock_sha256": lock["temporal_sha256"],
        "temporal_selection_lock_sha256": lock["selection_sha256"],
        "conditional_spatial_trial_id": "A000",
        "conditional_spatial_design_lock_sha256": a000["design_sha256"],
        "conditional_spatial_runtime_sha256": a000["runtime_sha256"],
        "conditional_spatial_student_barrier_sha256": a000["barrier_sha256"],
        "conditional_spatial_root_readback_sha256": a000["readback_sha256"],
        "conditional_spatial_prediction_barrier_sha256": a000["prediction_barrier_sha256"],
        # The final parameter lock still authenticates the original selected
        # baseline for temporal-cache lineage; its spatial boxes are not used.
        "original_final_baseline_trials": {
            ds: lock["parent_contexts"][ds]["trial_id"] for ds in ("hc2", "vidstg")
        },
    }


def _verify_a000_context(lock: Mapping[str, Any]) -> dict[str, Any]:
    import json
    trial_dir = legacy.BASE / "trials" / "A000"
    design_path = trial_dir / "DESIGN_LOCK.json"
    runtime_path = trial_dir / "RUNTIME.json"
    barrier_path = trial_dir / "STUDENT_GLOBAL_BARRIER.json"
    readback_path = trial_dir / "ROOT_READBACK.json"
    prediction_path = trial_dir / "GLOBAL_PREDICTION_BARRIER.json"
    for path in (design_path, runtime_path, barrier_path, readback_path, prediction_path):
        if not path.is_file():
            raise FileNotFoundError(f"sealed A000 M2 parent is missing {path.name}")
    design, runtime = read(design_path), read(runtime_path)
    barrier, root_readback, prediction = read(barrier_path), read(readback_path), read(prediction_path)
    if (design.get("status") != "locked" or design.get("trial_id") != "A000"
            or design.get("variant") != "baseline" or design.get("GT_read") is not False):
        raise RuntimeError("conditional spatial parent must be the locked A000 baseline trial")
    if runtime.get("status") != "locked" or runtime.get("trial_id") != "A000" or runtime.get("GT_read") is not False:
        raise RuntimeError("A000 runtime is not locked/GT-free")
    for rel, expected in {**runtime.get("pins", {}), **runtime.get("protected", {})}.items():
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"A000 runtime pin mismatch: {rel}")
    for dataset in ("hc2", "vidstg"):
        if (design.get("configs", {}).get(dataset, {}).get("M2") != lock["final"]["configs"][dataset]
                or design.get("response_settings", {}).get(dataset, {}).get("M2")
                != lock["final"]["response_settings"][dataset]):
            raise RuntimeError(f"A000 M2 configuration differs from the locked temporal-cache lineage: {dataset}")
    if (barrier.get("status") != "complete" or barrier.get("trial_id") != "A000"
            or barrier.get("count") != TOTAL_COUNT or barrier.get("GT_read") is not False
            or barrier.get("design_lock_sha256") != sha(design_path)
            or barrier.get("runtime_sha256") != sha(runtime_path)
            or barrier.get("roster_lock_sha256") != sha(legacy.ROSTER_PATH)
            or barrier.get("source_capture_barrier_sha256") != lock["source_barrier_sha256"]
            or barrier.get("reference_lock_sha256") != lock["reference_sha256"]
            or barrier.get("temporal_lock_sha256") != lock["temporal_sha256"]):
        raise RuntimeError("A000 student barrier is incomplete or not bound to the locked 128-source inputs")
    if (root_readback.get("status") != "pass" or root_readback.get("count") != TOTAL_COUNT
            or root_readback.get("GT_read") is not False
            or root_readback.get("barrier_sha256") != sha(barrier_path)
            or root_readback.get("runtime_sha256") != sha(runtime_path)):
        raise RuntimeError("A000 lacks a passing independent pre-GT root readback")
    if (prediction.get("status") != "complete" or prediction.get("count") != TOTAL_COUNT
            or prediction.get("GT_read") is not False
            or prediction.get("student_barrier_sha256") != sha(barrier_path)
            or prediction.get("root_readback_sha256") != sha(readback_path)
            or prediction.get("runtime_sha256") != sha(runtime_path)):
        raise RuntimeError("A000 lacks a complete pre-GT prediction seal")
    files_doc, files = legacy._file_rows(barrier_path, TOTAL_COUNT, "A000 M2 student")
    if set(files) != {legacy._cid(row) for row in lock["roster"]}:
        raise RuntimeError("A000 sealed file membership differs from the locked roster")
    return {
        "trial_id": "A000",
        "trial_dir": trial_dir,
        "design_sha256": sha(design_path),
        "runtime_sha256": sha(runtime_path),
        "barrier_sha256": sha(barrier_path),
        "readback_sha256": sha(readback_path),
        "prediction_barrier_sha256": sha(prediction_path),
        "barrier": files_doc,
        "files": files,
    }


def _verify_runtime_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    if not DESIGN_LOCK.is_file() or not RUNTIME_LOCK.is_file():
        raise FileNotFoundError("root DESIGN_LOCK and reviewed TEMPORAL_LITE_RUNTIME_LOCK are required before cache execution")
    design = read(DESIGN_LOCK)
    runtime = read(RUNTIME_LOCK)
    if design.get("status") != "locked" or design.get("GT_read") is not False:
        raise RuntimeError("root scope DESIGN_LOCK must be locked and explicitly GT-free")
    if design.get("study_id") != "stvg_lite_event_support_v1":
        raise RuntimeError("unexpected root scope DESIGN_LOCK identity")
    if (runtime.get("status") != "locked" or runtime.get("GT_read") is not False
            or runtime.get("execution_authorized") is not True):
        raise RuntimeError("reviewed runtime lock must explicitly authorize the CPU cache stage")
    if runtime.get("input_bindings") != _input_bindings(lock):
        raise RuntimeError("Temporal-Lite runtime does not bind the current design and original input locks")
    pins = runtime.get("pins")
    if not isinstance(pins, dict):
        raise RuntimeError("Temporal-Lite runtime pins must be a path-to-SHA map")
    for rel in RUNTIME_PIN_PATHS:
        path = ROOT / rel
        if not path.is_file() or pins.get(rel) != sha(path):
            raise RuntimeError(f"Temporal-Lite runtime pin mismatch: {rel}")
    if runtime.get("protocol") != runtime_lock_protocol():
        raise RuntimeError("Temporal-Lite runtime protocol differs from the authorized candidate")
    return runtime


def _load_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    lock = legacy._verify_locks()
    lock["a000_context"] = _verify_a000_context(lock)
    runtime = _verify_runtime_lock(lock)
    return lock, runtime


def runtime_lock_bindings() -> dict[str, Any]:
    """Return exact lock bindings for the root-owned Temporal-Lite run lock.

    This helper is read-only. It validates existing locks and sealed A000
    outputs, without loading media/GT, fitting, scoring, or writing artifacts.
    """
    lock = legacy._verify_locks()
    lock["a000_context"] = _verify_a000_context(lock)
    return _input_bindings(lock)


def runtime_lock_protocol() -> dict[str, Any]:
    """Return the exact protocol dictionary required before CPU execution."""
    return {
        "case_count": TOTAL_COUNT,
        "fit_case_count": FIT_COUNT,
        "no_fit_fallback_count": NO_FIT_COUNT,
        "active_parameters": 512,
        "lambda_grid": basefit.OFFSET_TRUST_CONFIG["lambda_grid"],
        "l2": 0.0,
        "candidate_objective": "offset_count_times_shared_mean_kl",
        "full_grid_trust": True,
        "original_lbfgs_and_armijo": True,
        "gt_scoring": False,
        "gpu_execution": False,
        "visual_model_decode": False,
    }


def runtime_source_pins() -> dict[str, str]:
    """Return SHA pins for implementation files and unchanged dependencies."""
    return {rel: sha(ROOT / rel) for rel in RUNTIME_PIN_PATHS}


def _readback_one_output(path: Path, expected_sha: str, cid: str,
                         source_case_sha256: str, source_file_sha256: str,
                         source_model_sha256: str, temp_record: Mapping[str, Any],
                         ref_record: Mapping[str, Any], expected_no_fit_reason: str | None) -> dict[str, Any]:
    import torch
    if sha(path) != expected_sha:
        raise RuntimeError(f"case output SHA changed before internal readback: {cid}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "stvg-temporal-lite-case-v1"
            or payload.get("case_id") != cid or payload.get("GT_read") is not False
            or payload.get("source_case_sha256") != source_case_sha256
            or payload.get("source_file_sha256") != source_file_sha256
            or payload.get("source_model_state_sha256") != source_model_sha256
            or payload.get("temporal_record") != dict(temp_record)
            or payload.get("reference_record") != dict(ref_record)):
        raise RuntimeError(f"case identity/provenance readback mismatch: {cid}")
    if set(payload.get("arms", {})) != set(CASE_ARMS):
        raise RuntimeError(f"case arm schema mismatch: {cid}")
    update = payload.get("temporal_update_status", {})
    if expected_no_fit_reason is None:
        if (update.get("status") != "fitted"
                or update.get("fit_variants") != list(temporal_lite.VARIANTS)
                or set(payload.get("fits", {})) != set(temporal_lite.VARIANTS)):
            raise RuntimeError(f"fitted case schema mismatch: {cid}")
    else:
        if (update.get("status") != "no_update_unknown"
                or update.get("reason") != expected_no_fit_reason
                or update.get("fit_variants") != []
                or payload.get("fits") != {}):
            raise RuntimeError(f"no-fit case schema mismatch: {cid}")
        selected = list(temp_record["selected_interval"])
        if any(list(payload["arms"][arm]["interval"]) != selected for arm in ("TS_M2", "TS_offset_trust", "TS_temporal_lite")):
            raise RuntimeError(f"no-fit fallback changed selected interval: {cid}")
    return payload


def build_arms(native_boxes: Any, native_interval: Any, selected_interval: Any,
               m2_boxes: Any, baseline_interval: Any,
               temporal_lite_interval: Any) -> dict[str, Any]:
    """Create the six paired arms while retaining the frozen M2 box arrays."""
    native = np.asarray(native_boxes).copy()
    m2 = np.asarray(m2_boxes).copy()
    if native.ndim != 2 or native.shape[1] != 4 or m2.shape != native.shape:
        raise ValueError("Native and M2 boxes must share a finite [source_frames,4] shape")
    if not np.isfinite(native).all() or not np.isfinite(m2).all():
        raise ValueError("Native and M2 boxes must be finite")
    return {
        "F": {"boxes": native.copy(), "interval": list(native_interval)},
        "T": {"boxes": native.copy(), "interval": list(selected_interval)},
        "S_M2": {"boxes": m2.copy(), "interval": list(native_interval)},
        "TS_M2": {"boxes": m2.copy(), "interval": list(selected_interval)},
        "TS_offset_trust": {"boxes": m2.copy(), "interval": list(baseline_interval)},
        "TS_temporal_lite": {"boxes": m2.copy(), "interval": list(temporal_lite_interval)},
    }


def run() -> dict[str, Any]:
    import torch
    torch.set_num_threads(2)
    lock, runtime = _load_inputs()
    if OUT.exists() and any(p.name not in {RUNTIME_LOCK.name} for p in OUT.iterdir()):
        raise FileExistsError("Temporal-Lite output already contains artifacts; no overwrite or implicit resume")
    student_dir = OUT / "student"
    student_dir.mkdir(parents=True, exist_ok=True)

    from scripts import stvg_table1_temporal_adapter_v1 as temporal_adapter
    from threadpoolctl import threadpool_limits

    rows = lock["roster"]
    ref_by_id, temporal_by_id = lock["references"], lock["temporals"]
    contexts = lock["parent_contexts"]
    records: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    fitted_count = no_fit_count = 0
    start = time.perf_counter()

    for rec in rows:
        cid, dataset = legacy._cid(rec), rec["dataset"]
        case, source_path = legacy.load_case(rec)
        srcfile = lock["source_files"][cid]
        source_file_sha = sha(source_path)
        source_case_sha = str(case.get("source_case_sha256"))
        if (source_path != ROOT / rec["source_case_path"] or source_file_sha != srcfile["sha256"]
                or srcfile.get("source_case_sha256") != source_case_sha):
            raise RuntimeError(f"Source barrier mismatch for {cid}")
        ref_record, temp_record = ref_by_id[cid], temporal_by_id[cid]
        if (case.get("GT_read") is not False or case.get("dataset") != dataset
                or ref_record.get("source_case_sha256") != source_case_sha):
            raise RuntimeError(f"Source/reference binding or GT flag mismatch for {cid}")

        cache_path, cache_sha, cache_doc, cached_fit, teacher_path, round1_parent = legacy._load_temporal_cache(
            rec, cid, temp_record, source_case_sha, source_file_sha,
            lock["selection"], lock["selection_sha256"])
        no_fit_record = None
        hidden = None
        fit_seconds: dict[str, float | None] = {
            "offset_trust_exact_replay": None,
            "temporal_lite": None,
        }
        if cached_fit is None:
            no_fit_record = legacy._validated_no_fit_fallback(
                case, cache_doc, teacher_path, source_file_sha, temp_record["selected_interval"],
                round1_parent=round1_parent, selection=lock["selection"])
            intervals = temporal_lite.no_fit_fallback_intervals(
                case.get("temporal_candidates", ()), temp_record["selected_interval"], no_fit_record)
            fit_results: dict[str, Any] = {}
            reward_provenance = "unavailable_temporal_input_no_fit_no_reward_fabricated"
            no_fit_count += 1
        else:
            rewards, reward_provenance = legacy._candidate_rewards(case, cached_fit, teacher_path)
            hidden = temporal_adapter._hidden_from_source_case(case)
            context = runtime.get("cpu_execution_context", {})
            limits = (context.get("inherited_round1_blas_threads") if round1_parent is not None
                      else context.get("new_coordinate_blas_threads"))
            if isinstance(limits, bool) or not isinstance(limits, int) or limits < 1:
                raise RuntimeError("runtime lock lacks a positive source-lineage BLAS thread limit")
            with threadpool_limits(limits=limits, user_api="blas"):
                fit_start = time.perf_counter()
                baseline = temporal_lite.fit(case, hidden, case["temporal_candidates"], rewards, "offset_trust")
                fit_seconds["offset_trust_exact_replay"] = time.perf_counter() - fit_start
                if not legacy._exact_equal(baseline, cached_fit):
                    raise RuntimeError(f"exact Offset-Trust baseline replay differs from sealed cache: {cid}")
                fit_start = time.perf_counter()
                lite = temporal_lite.fit(case, hidden, case["temporal_candidates"], rewards, "temporal_lite")
                fit_seconds["temporal_lite"] = time.perf_counter() - fit_start
            if list(baseline["final_interval"]) != list(temp_record["selected_interval"]):
                raise RuntimeError(f"baseline replay interval differs from TEMPORAL_LOCK: {cid}")
            fit_results = {"offset_trust": baseline, "temporal_lite": lite}
            intervals = {name: list(result["final_interval"]) for name, result in fit_results.items()}
            fitted_count += 1

        a000_context = lock["a000_context"]
        parent_path, parent_entry, parent_payload, m2_boxes, m2_interval = legacy._load_m2_parent(
            cid, rec, a000_context)
        if m2_interval != list(temp_record["selected_interval"]):
            raise RuntimeError(f"sealed M2 interval differs from original temporal interval for {cid}")
        if (parent_payload.get("source_case_sha256") != source_case_sha
                or parent_payload.get("reference_lock_sha256") != lock["reference_sha256"]
                or parent_payload.get("temporal_lock_sha256") != lock["temporal_sha256"]
                or parent_payload.get("source_restored") is not True):
            raise RuntimeError(f"sealed M2 spatial parent provenance mismatch: {cid}")

        native_boxes = case["native_boxes_xyxy_normalized"]
        native_interval = list(case["Native_interval"])
        lite_interval = intervals.get("temporal_lite", list(temp_record["selected_interval"]))
        baseline_interval = intervals.get("offset_trust", list(temp_record["selected_interval"]))
        arms = build_arms(native_boxes, native_interval, temp_record["selected_interval"],
                          m2_boxes, baseline_interval, lite_interval)
        payload = {
            "schema": "stvg-temporal-lite-case-v1",
            "case_id": cid,
            "dataset": dataset,
            "arms": arms,
            "fits": fit_results,
            "temporal_update_status": ({
                "status": "no_update_unknown",
                "reason": no_fit_record["reason"],
                "fit_variants": [],
                "no_update_temporal_arms": ["TS_offset_trust", "TS_temporal_lite"],
                "fallback_source": no_fit_record.get("inherited_through", "sealed_teacher_native_interval"),
                "GT_read": False,
            } if no_fit_record is not None else {
                "status": "fitted",
                "fit_variants": list(temporal_lite.VARIANTS),
                "GT_read": False,
            }),
            "temporal_no_fit_record": no_fit_record,
            "frame_ids": list(map(int, case["frame_ids"])),
            "reference_record": copy.deepcopy(ref_record),
            "temporal_record": copy.deepcopy(temp_record),
            "source_case_sha256": source_case_sha,
            "source_file_sha256": source_file_sha,
            "source_model_state_sha256": case["source_model_state_sha256"],
            "temporal_cache_path": str(cache_path.relative_to(ROOT)),
            "temporal_cache_sha256": cache_sha,
            "temporal_reward_provenance": reward_provenance,
            "matched_cpu_fit_seconds": fit_seconds,
            "matched_cpu_fit_timing_scope": "same cached Source features, hidden tables, candidates, rewards, lineage-specific BLAS limit; fit calls only",
            "baseline_temporal_fit_exactly_replayed": no_fit_record is None,
            "baseline_spatial_trial_id": "A000",
            "baseline_spatial_student_path": str(parent_path.relative_to(ROOT)),
            "baseline_spatial_student_sha256": parent_entry["sha256"],
            "conditional_TS_readouts_reuse_original_M2_boxes": True,
            "full_visual_model_decode_replayed": False,
            "source_restored": True,
            "GT_read": False,
            "runtime_lock_sha256": sha(RUNTIME_LOCK),
            "design_lock_sha256": sha(DESIGN_LOCK),
        }
        out_path = student_dir / f"{cid}.pt"
        if out_path.exists() or out_path.with_suffix(".receipt.json").exists():
            raise FileExistsError(f"case output already exists; no implicit retry: {cid}")
        file_row = save_pt(out_path, payload)
        receipt_path = out_path.with_suffix(".receipt.json")
        _write_once(receipt_path, {
            "schema": "stvg-temporal-lite-case-receipt-v1",
            "status": "complete",
            "case_id": cid,
            "dataset": dataset,
            "path": str(out_path.relative_to(ROOT)),
            "sha256": sha(out_path),
            "bytes": out_path.stat().st_size,
            "source_case_sha256": source_case_sha,
            "temporal_cache_sha256": cache_sha,
            "GT_read": False,
        })
        expected_reason = no_fit_record["reason"] if no_fit_record is not None else None
        _readback_one_output(out_path, sha(out_path), cid, source_case_sha, source_file_sha,
                             str(case["source_model_state_sha256"]), temp_record, ref_record, expected_reason)
        file_entry = {
            "case_id": cid,
            "dataset": dataset,
            "path": str(out_path.relative_to(ROOT)),
            "sha256": sha(out_path),
            "bytes": out_path.stat().st_size,
            "receipt_path": str(receipt_path.relative_to(ROOT)),
            "receipt_sha256": sha(receipt_path),
            "source_case_sha256": source_case_sha,
            "source_file_sha256": source_file_sha,
            "temporal_cache_sha256": cache_sha,
            "baseline_spatial_student_sha256": parent_entry["sha256"],
        }
        output_rows.append(file_entry)
        records.append({
            "case_id": cid,
            "dataset": dataset,
            "source_case_sha256": source_case_sha,
            "source_file_sha256": source_file_sha,
            "source_temporal_sha256": cache_sha,
            "baseline_offset_trust_interval": baseline_interval,
            "temporal_lite_interval": lite_interval,
            "selected_interval": lite_interval,
            "selected_variant": "temporal_lite_candidate",
            "fit_status": "no_update_unknown" if no_fit_record is not None else "fitted",
            "no_fit_reason": no_fit_record["reason"] if no_fit_record else None,
            "matched_cpu_fit_seconds": fit_seconds,
            "GT_read": False,
        })
        audit_rows.append({"case_id": cid, "dataset": dataset,
                           "baseline_fit_exact": no_fit_record is None,
                           "baseline_interval_exact": baseline_interval == list(temp_record["selected_interval"]),
                           "temporal_lite_interval": lite_interval,
                           "no_fit_fallback_reason": no_fit_record["reason"] if no_fit_record else None,
                           "temporal_cache_sha256": cache_sha,
                           "source_case_sha256": source_case_sha})
        del case, payload, parent_payload, hidden, fit_results
        if len(output_rows) % 8 == 0:
            print("TEMPORAL_LITE", len(output_rows), "/ 128", flush=True)

    ids = [legacy._cid(row) for row in rows]
    if (len(output_rows) != TOTAL_COUNT or {x["case_id"] for x in output_rows} != set(ids)
            or fitted_count != FIT_COUNT or no_fit_count != NO_FIT_COUNT):
        raise RuntimeError(f"expected 128 total / 111 fitted / 17 original no-fit fallbacks, got {len(output_rows)}/{fitted_count}/{no_fit_count}")
    domain_counts = {ds: sum(x["dataset"] == ds for x in output_rows) for ds in ("hc2", "vidstg")}
    if domain_counts != {"hc2": 64, "vidstg": 64}:
        raise RuntimeError(f"unexpected domain counts: {domain_counts}")

    bindings = _input_bindings(lock)
    barrier_path = OUT / "CPU_OUTPUT_BARRIER.json"
    _write_once(barrier_path, {
        "schema": "stvg-temporal-lite-cpu-output-barrier-v1",
        "status": "complete",
        "count": TOTAL_COUNT,
        "domain_counts": domain_counts,
        "files": output_rows,
        "bindings": bindings,
        "runtime_lock_sha256": sha(RUNTIME_LOCK),
        "runner_sha256": sha(RUN_SCRIPT),
        "fit_helper_sha256": sha(FIT_SCRIPT),
        "readback_script_sha256": sha(READBACK_SCRIPT),
        "baseline_exact_replay_count": fitted_count,
        "explicit_original_no_fit_fallback_count": no_fit_count,
        "fit_variant_ids": list(temporal_lite.VARIANTS),
        "conditional_TS_spatial_reference": "sealed original-baseline M2 xyxy boxes reused for both temporal fits",
        "full_visual_model_decode_replayed": False,
        "GT_read": False,
        "CPU_only": True,
        "GPU_used": False,
        "model_loaded": False,
        "actual_gt_scoring_performed": False,
        "wall_seconds": time.perf_counter() - start,
    })
    _write_once(OUT / "TEMPORAL_LOCK.json", {
        "schema": "stvg-temporal-lite-candidate-temporal-lock-v1",
        "status": "complete",
        "candidate_variant": "temporal_lite",
        "selection_basis": "pre_GT_CPU_fit_output_only",
        "not_a_deployment_or_scored_method_selection": True,
        "count": TOTAL_COUNT,
        "domain_counts": domain_counts,
        "records": records,
        "cpu_output_barrier_sha256": sha(barrier_path),
        "runtime_lock_sha256": sha(RUNTIME_LOCK),
        "GT_read": False,
    })
    _write_once(OUT / "CPU_STAGE_COMPLETION.json", {
        "schema": "stvg-temporal-lite-cpu-stage-completion-v1",
        "status": "complete",
        "count": TOTAL_COUNT,
        "domain_counts": domain_counts,
        "fitted_case_count": fitted_count,
        "explicit_original_no_fit_fallback_count": no_fit_count,
        "fit_variants": list(temporal_lite.VARIANTS),
        "GT_read": False,
        "CPU_only": True,
        "GPU_used": False,
        "model_loaded": False,
        "actual_gt_scoring_performed": False,
        "full_visual_model_decode_replayed": False,
        "cpu_output_barrier_sha256": sha(barrier_path),
        "temporal_lock_sha256": sha(OUT / "TEMPORAL_LOCK.json"),
    })
    return {"count": TOTAL_COUNT, "domain_counts": domain_counts,
            "baseline_exact_replays": fitted_count, "no_fit_fallbacks": no_fit_count,
            "cpu_output_barrier_sha256": sha(barrier_path)}


def contracts() -> dict[str, Any]:
    """Check output-arm mapping and no-fit fallback on synthetic CPU data only."""
    case, hidden, candidates, rewards = basefit._synthetic_case(seed=16420)
    case = dict(case)
    case["native_boxes_xyxy_normalized"] = np.asarray(
        [[0.12, 0.18, 0.35, 0.52], [0.21, 0.23, 0.49, 0.68]], dtype=np.float64)
    case["Native_interval"] = [2, 11]
    baseline = temporal_lite.fit(case, hidden, candidates, rewards, "offset_trust")
    lite = temporal_lite.fit(case, hidden, candidates, rewards, "temporal_lite")
    m2 = case["native_boxes_xyxy_normalized"] + np.asarray(
        [[0.01, 0.00, 0.01, 0.00], [0.00, 0.01, 0.00, 0.01]], dtype=np.float64)
    arms = build_arms(case["native_boxes_xyxy_normalized"], case["Native_interval"],
                      baseline["final_interval"], m2,
                      baseline["final_interval"], lite["final_interval"])
    assert set(arms) == set(CASE_ARMS)
    assert np.array_equal(arms["F"]["boxes"], arms["T"]["boxes"])
    assert np.array_equal(arms["S_M2"]["boxes"], arms["TS_temporal_lite"]["boxes"])
    assert arms["TS_offset_trust"]["interval"] == baseline["final_interval"]
    assert arms["TS_temporal_lite"]["interval"] == lite["final_interval"]
    fallback_record = {
        "status": "unavailable_temporal_input_native_fallback",
        "reason": "candidate_pool_or_role_count_not_four",
        "selected_interval": [2, 11], "optimizer_steps": 0, "GT_read": False,
    }
    fallback_intervals = temporal_lite.no_fit_fallback_intervals([], [2, 11], fallback_record)
    fallback_arms = build_arms(case["native_boxes_xyxy_normalized"], case["Native_interval"],
                               fallback_intervals["offset_trust"], m2,
                               fallback_intervals["offset_trust"], fallback_intervals["temporal_lite"])
    assert fallback_arms["TS_offset_trust"]["interval"] == [2, 11]
    assert fallback_arms["TS_temporal_lite"]["interval"] == [2, 11]
    return {
        "schema": "stvg-temporal-lite-runner-cpu-contracts-v1",
        "status": "pass",
        "CPU_only": True,
        "synthetic_only": True,
        "GT_read": False,
        "GPU_used": False,
        "model_loaded": False,
        "source_cases_loaded": 0,
        "score_rows_loaded": 0,
        "case_arm_count": len(CASE_ARMS),
        "conditional_arms_reuse_exactly_m2_boxes": True,
        "empty_candidate_no_fit_keeps_original_interval": True,
        "fitter_contracts": temporal_lite.contracts(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="process the locked 128-case CPU cache")
    parser.add_argument("--contracts", action="store_true", help="run synthetic-only fitter contracts")
    args = parser.parse_args()
    if args.run == args.contracts:
        parser.error("choose exactly one of --run or --contracts")
    if args.contracts:
        print(json.dumps(contracts(), indent=2, sort_keys=True))
    else:
        print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
