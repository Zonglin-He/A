#!/usr/bin/env python3
"""Independent full pre-GT readback and seal for Temporal-Lite CPU outputs."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stvg_component_ablation_common_v1 import read, sha
from scripts import run_stvg_component_temporal005 as legacy
from scripts import stvg_temporal_lite_fit_v1 as temporal_lite

SCOPE = ROOT / "artifacts" / "stvg_lite_event_support_v1"
OUT = SCOPE / "temporal"
DESIGN_LOCK = SCOPE / "DESIGN_LOCK.json"
RUNTIME_LOCK = OUT / "TEMPORAL_LITE_RUNTIME_LOCK.json"
CPU_BARRIER = OUT / "CPU_OUTPUT_BARRIER.json"
TEMPORAL_LOCK = OUT / "TEMPORAL_LOCK.json"
READBACK = OUT / "ROOT_READBACK.json"
PREDICTION_BARRIER = OUT / "PREDICTION_GLOBAL_BARRIER.json"
TOTAL_COUNT = 128
FIT_COUNT = 111
NO_FIT_COUNT = 17
ARMS = ("F", "T", "S_M2", "TS_M2", "TS_offset_trust", "TS_temporal_lite")
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


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def _write_once(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _need(not path.exists() and not path.with_name(path.name + ".tmp").exists(),
          f"refusing to overwrite existing/partial readback artifact: {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _bindings(lock: Mapping[str, Any]) -> dict[str, Any]:
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
        "original_final_baseline_trials": {
            ds: lock["parent_contexts"][ds]["trial_id"] for ds in ("hc2", "vidstg")
        },
    }


def _verify_a000_context(lock: Mapping[str, Any]) -> dict[str, Any]:
    trial_dir = legacy.BASE / "trials" / "A000"
    design_path = trial_dir / "DESIGN_LOCK.json"
    runtime_path = trial_dir / "RUNTIME.json"
    barrier_path = trial_dir / "STUDENT_GLOBAL_BARRIER.json"
    readback_path = trial_dir / "ROOT_READBACK.json"
    prediction_path = trial_dir / "GLOBAL_PREDICTION_BARRIER.json"
    for path in (design_path, runtime_path, barrier_path, readback_path, prediction_path):
        _need(path.is_file(), f"A000 parent missing {path.name}")
    design, runtime = read(design_path), read(runtime_path)
    barrier, root_readback, prediction = read(barrier_path), read(readback_path), read(prediction_path)
    _need(design.get("status") == "locked" and design.get("trial_id") == "A000"
          and design.get("variant") == "baseline" and design.get("GT_read") is False,
          "conditional spatial parent is not locked A000 baseline")
    _need(runtime.get("status") == "locked" and runtime.get("trial_id") == "A000"
          and runtime.get("GT_read") is False, "A000 runtime is not locked/GT-free")
    for rel, expected in {**runtime.get("pins", {}), **runtime.get("protected", {})}.items():
        _need(sha(ROOT / rel) == expected, f"A000 runtime pin mismatch: {rel}")
    for dataset in ("hc2", "vidstg"):
        _need(design.get("configs", {}).get(dataset, {}).get("M2") == lock["final"]["configs"][dataset]
              and design.get("response_settings", {}).get(dataset, {}).get("M2")
              == lock["final"]["response_settings"][dataset],
              f"A000 M2 config differs from temporal-cache lineage for {dataset}")
    _need(barrier.get("status") == "complete" and barrier.get("trial_id") == "A000"
          and barrier.get("count") == TOTAL_COUNT and barrier.get("GT_read") is False
          and barrier.get("design_lock_sha256") == sha(design_path)
          and barrier.get("runtime_sha256") == sha(runtime_path)
          and barrier.get("roster_lock_sha256") == sha(legacy.ROSTER_PATH)
          and barrier.get("source_capture_barrier_sha256") == lock["source_barrier_sha256"]
          and barrier.get("reference_lock_sha256") == lock["reference_sha256"]
          and barrier.get("temporal_lock_sha256") == lock["temporal_sha256"],
          "A000 student barrier not bound to source/reference/temporal inputs")
    _need(root_readback.get("status") == "pass" and root_readback.get("count") == TOTAL_COUNT
          and root_readback.get("GT_read") is False
          and root_readback.get("barrier_sha256") == sha(barrier_path)
          and root_readback.get("runtime_sha256") == sha(runtime_path),
          "A000 root readback failed")
    _need(prediction.get("status") == "complete" and prediction.get("count") == TOTAL_COUNT
          and prediction.get("GT_read") is False
          and prediction.get("student_barrier_sha256") == sha(barrier_path)
          and prediction.get("root_readback_sha256") == sha(readback_path)
          and prediction.get("runtime_sha256") == sha(runtime_path),
          "A000 prediction seal incomplete")
    files_doc, files = legacy._file_rows(barrier_path, TOTAL_COUNT, "A000 M2 student")
    _need(set(files) == {legacy._cid(row) for row in lock["roster"]}, "A000 file membership differs from roster")
    return {"trial_id": "A000", "trial_dir": trial_dir,
            "design_sha256": sha(design_path), "runtime_sha256": sha(runtime_path),
            "barrier_sha256": sha(barrier_path), "readback_sha256": sha(readback_path),
            "prediction_barrier_sha256": sha(prediction_path), "barrier": files_doc, "files": files}


def _load_locked_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    if not DESIGN_LOCK.is_file() or not RUNTIME_LOCK.is_file():
        raise FileNotFoundError("root design and Temporal-Lite runtime locks are required")
    design = read(DESIGN_LOCK)
    runtime = read(RUNTIME_LOCK)
    _need(design.get("status") == "locked" and design.get("GT_read") is False,
          "root DESIGN_LOCK must be locked/GT-free")
    _need(design.get("study_id") == "stvg_lite_event_support_v1",
          "unexpected root scope identity")
    _need(runtime.get("status") == "locked" and runtime.get("GT_read") is False
          and runtime.get("execution_authorized") is True,
          "Temporal-Lite runtime is not a reviewed, execution-authorized lock")
    pins = runtime.get("pins")
    _need(isinstance(pins, dict), "runtime pins must be a path-to-SHA map")
    for rel in RUNTIME_PIN_PATHS:
        _need(pins.get(rel) == sha(ROOT / rel), f"runtime source pin mismatch: {rel}")

    lock = legacy._verify_locks()
    lock["a000_context"] = _verify_a000_context(lock)
    _need(runtime.get("input_bindings") == _bindings(lock),
          "runtime input bindings differ from current original locks/lineage")
    expected_protocol = {
        "case_count": TOTAL_COUNT,
        "fit_case_count": FIT_COUNT,
        "no_fit_fallback_count": NO_FIT_COUNT,
        "active_parameters": 512,
        "lambda_grid": temporal_lite.basefit.OFFSET_TRUST_CONFIG["lambda_grid"],
        "l2": 0.0,
        "candidate_objective": "offset_count_times_shared_mean_kl",
        "full_grid_trust": True,
        "original_lbfgs_and_armijo": True,
        "gt_scoring": False,
        "gpu_execution": False,
        "visual_model_decode": False,
    }
    _need(runtime.get("protocol") == expected_protocol, "runtime protocol changed")
    return lock, runtime


def _read_artifact_path(row: Mapping[str, Any], field: str) -> Path:
    rel = row.get(field)
    _need(isinstance(rel, str), f"barrier row has no {field}")
    path = (ROOT / rel).resolve()
    _need(ROOT in path.parents and path.is_file(), f"artifact path missing/outside workspace: {rel}")
    return path


def _boxes_equal(actual: Any, expected: Any) -> bool:
    aa = np.asarray(actual.detach().cpu().numpy() if hasattr(actual, "detach") else actual)
    bb = np.asarray(expected.detach().cpu().numpy() if hasattr(expected, "detach") else expected)
    return aa.dtype == bb.dtype and aa.shape == bb.shape and np.array_equal(aa, bb)


def _validate_case(payload: Mapping[str, Any], cid: str, dataset: str,
                   case: Mapping[str, Any], source_path: Path,
                   source_row: Mapping[str, Any], reference: Mapping[str, Any],
                   temporal: Mapping[str, Any], lock: Mapping[str, Any],
                   cached_fit: Mapping[str, Any] | None, no_fit_record: Mapping[str, Any] | None,
                   m2_boxes: Any, baseline_trial_id: str) -> None:
    _need(payload.get("schema") == "stvg-temporal-lite-case-v1"
          and payload.get("case_id") == cid and payload.get("dataset") == dataset
          and payload.get("GT_read") is False and payload.get("source_restored") is True,
          f"prediction identity/GT/source state mismatch: {cid}")
    _need(payload.get("source_case_sha256") == case.get("source_case_sha256")
          and payload.get("source_file_sha256") == sha(source_path)
          and source_row.get("sha256") == sha(source_path)
          and source_row.get("source_case_sha256") == case.get("source_case_sha256"),
          f"source lineage mismatch: {cid}")
    _need(payload.get("reference_record") == dict(reference)
          and payload.get("temporal_record") == dict(temporal),
          f"reference/temporal record mismatch: {cid}")
    _need(payload.get("baseline_temporal_fit_exactly_replayed") is (cached_fit is not None),
          f"baseline replay status mismatch: {cid}")
    _need(payload.get("baseline_spatial_trial_id") == baseline_trial_id,
          f"baseline spatial trial mismatch: {cid}")
    _need(payload.get("conditional_TS_readouts_reuse_original_M2_boxes") is True
          and payload.get("full_visual_model_decode_replayed") is False,
          f"conditional readout provenance mismatch: {cid}")

    arms = payload.get("arms")
    _need(isinstance(arms, dict) and set(arms) == set(ARMS), f"arm set mismatch: {cid}")
    native = case["native_boxes_xyxy_normalized"]
    native_interval = list(case["Native_interval"])
    base_interval = list(temporal["selected_interval"])
    if cached_fit is not None:
        fits = payload.get("fits")
        _need(isinstance(fits, dict) and set(fits) == set(temporal_lite.VARIANTS), f"fit set mismatch: {cid}")
        _need(legacy._exact_equal(fits["offset_trust"], cached_fit), f"cached baseline fit mismatch: {cid}")
        lite = fits["temporal_lite"]
        _need(lite.get("variant") == "temporal_lite" and lite.get("GT_read") is False
              and lite.get("active_parameters") == 512
              and lite.get("config", {}).get("l2") == 0.0
              and lite.get("config", {}).get("lambda_grid") == temporal_lite.basefit.OFFSET_TRUST_CONFIG["lambda_grid"],
              f"Temporal-Lite fit protocol mismatch: {cid}")
        lite_interval = list(lite["final_interval"])
        baseline_interval = list(cached_fit["final_interval"])
        _need(baseline_interval == base_interval, f"baseline interval changed: {cid}")
        status = payload.get("temporal_update_status", {})
        _need(status.get("status") == "fitted" and status.get("fit_variants") == list(temporal_lite.VARIANTS),
              f"fitted status mismatch: {cid}")
        timings = payload.get("matched_cpu_fit_seconds", {})
        _need(set(timings) == {"offset_trust_exact_replay", "temporal_lite"}
              and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                      and math.isfinite(float(value)) and float(value) > 0 for value in timings.values()),
              f"matched CPU fit timings missing or invalid: {cid}")
    else:
        _need(payload.get("fits") == {} and isinstance(no_fit_record, dict), f"no-fit payload mismatch: {cid}")
        lite_interval = base_interval
        baseline_interval = base_interval
        status = payload.get("temporal_update_status", {})
        _need(status.get("status") == "no_update_unknown"
              and status.get("reason") == no_fit_record.get("reason")
              and status.get("fit_variants") == [], f"no-fit status mismatch: {cid}")
        _need(payload.get("matched_cpu_fit_seconds") == {
            "offset_trust_exact_replay": None, "temporal_lite": None},
            f"no-fit fallback must not report fit timing: {cid}")

    interval_specs = {
        "F": native_interval,
        "T": base_interval,
        "S_M2": native_interval,
        "TS_M2": base_interval,
        "TS_offset_trust": baseline_interval,
        "TS_temporal_lite": lite_interval,
    }
    _need(_boxes_equal(arms["F"].get("boxes"), native)
          and _boxes_equal(arms["T"].get("boxes"), native)
          and _boxes_equal(arms["S_M2"].get("boxes"), m2_boxes)
          and _boxes_equal(arms["TS_M2"].get("boxes"), m2_boxes)
          and _boxes_equal(arms["TS_offset_trust"].get("boxes"), m2_boxes)
          and _boxes_equal(arms["TS_temporal_lite"].get("boxes"), m2_boxes),
          f"conditional arm boxes differ from sealed Native/M2 arrays: {cid}")
    for name, interval in interval_specs.items():
        _need(list(arms[name].get("interval", [])) == interval, f"conditional interval mismatch: {cid}/{name}")


def run_readback() -> dict[str, Any]:
    import torch
    torch.set_num_threads(2)
    lock, runtime = _load_locked_inputs()
    _need(CPU_BARRIER.is_file() and TEMPORAL_LOCK.is_file(), "CPU output barrier and candidate TEMPORAL_LOCK are required")
    barrier = read(CPU_BARRIER)
    temporal_lock = read(TEMPORAL_LOCK)
    _need(barrier.get("status") == "complete" and barrier.get("count") == TOTAL_COUNT
          and barrier.get("GT_read") is False and barrier.get("CPU_only") is True,
          "CPU output barrier is incomplete or not explicitly pre-GT")
    _need(barrier.get("bindings") == _bindings(lock)
          and barrier.get("runtime_lock_sha256") == sha(RUNTIME_LOCK)
          and barrier.get("runner_sha256") == sha(ROOT / "scripts/run_stvg_temporal_lite128_v1.py")
          and barrier.get("fit_helper_sha256") == sha(ROOT / "scripts/stvg_temporal_lite_fit_v1.py")
          and barrier.get("readback_script_sha256") == sha(Path(__file__)),
          "CPU output barrier does not bind current code and original input locks")
    _need(temporal_lock.get("schema") == "stvg-temporal-lite-candidate-temporal-lock-v1"
          and temporal_lock.get("status") == "complete"
          and temporal_lock.get("candidate_variant") == "temporal_lite"
          and temporal_lock.get("not_a_deployment_or_scored_method_selection") is True
          and temporal_lock.get("count") == TOTAL_COUNT
          and temporal_lock.get("cpu_output_barrier_sha256") == sha(CPU_BARRIER)
          and temporal_lock.get("runtime_lock_sha256") == sha(RUNTIME_LOCK)
          and temporal_lock.get("GT_read") is False,
          "candidate TEMPORAL_LOCK is incomplete or has an invalid scope")

    files = barrier.get("files")
    _need(isinstance(files, list) and len(files) == TOTAL_COUNT, "CPU barrier must contain 128 files")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in files:
        cid = row.get("case_id")
        _need(isinstance(cid, str) and cid not in by_id, "duplicate or missing case identity in CPU barrier")
        by_id[cid] = row
        file_path = _read_artifact_path(row, "path")
        receipt_path = _read_artifact_path(row, "receipt_path")
        _need(sha(file_path) == row.get("sha256") and file_path.stat().st_size == int(row.get("bytes", -1)),
              f"case artifact checksum/size mismatch: {cid}")
        _need(sha(receipt_path) == row.get("receipt_sha256"), f"receipt checksum mismatch: {cid}")

    roster = lock["roster"]
    roster_ids = {legacy._cid(rec) for rec in roster}
    _need(set(by_id) == roster_ids, "CPU barrier membership differs from original 128-source roster")
    lock_records = temporal_lock.get("records")
    _need(isinstance(lock_records, list) and len(lock_records) == TOTAL_COUNT, "candidate TEMPORAL_LOCK record count mismatch")
    lock_records_by_id = {row.get("case_id"): row for row in lock_records}
    _need(set(lock_records_by_id) == roster_ids, "candidate TEMPORAL_LOCK membership differs from roster")

    refs, temporal_rows = lock["references"], lock["temporals"]
    source_rows = lock["source_files"]
    contexts = lock["a000_context"]
    from scripts import stvg_table1_temporal_adapter_v1 as temporal_adapter
    from threadpoolctl import threadpool_limits
    fit_count = no_fit_count = lite_replay_count = 0
    domain_counts = {"hc2": 0, "vidstg": 0}
    final_rows = []

    for rec in roster:
        cid, dataset = legacy._cid(rec), rec["dataset"]
        case, source_path = legacy.load_case(rec)
        source_row = source_rows[cid]
        temp_record, ref_record = temporal_rows[cid], refs[cid]
        src_sha = sha(source_path)
        _need(case.get("source_case_sha256") == source_row.get("source_case_sha256")
              and src_sha == source_row.get("sha256")
              and source_path == ROOT / rec["source_case_path"], f"source barrier mismatch: {cid}")

        cache_path, cache_sha, cache_doc, cached_fit, teacher_path, round1_parent = legacy._load_temporal_cache(
            rec, cid, temp_record, str(case.get("source_case_sha256")), src_sha,
            lock["selection"], lock["selection_sha256"])
        no_fit_record = None
        if cached_fit is None:
            no_fit_record = legacy._validated_no_fit_fallback(
                case, cache_doc, teacher_path, src_sha, temp_record["selected_interval"],
                round1_parent=round1_parent, selection=lock["selection"])
        else:
            rewards, _provenance = legacy._candidate_rewards(case, cached_fit, teacher_path)
            hidden = temporal_adapter._hidden_from_source_case(case)
            cpu_context = runtime.get("cpu_execution_context", {})
            limits = (cpu_context.get("inherited_round1_blas_threads") if round1_parent is not None
                      else cpu_context.get("new_coordinate_blas_threads"))
            _need(isinstance(limits, int) and not isinstance(limits, bool) and limits > 0,
                  "runtime lock lacks positive BLAS thread limit")
            with threadpool_limits(limits=limits, user_api="blas"):
                expected_lite = temporal_lite.fit(case, hidden, case["temporal_candidates"], rewards, "temporal_lite")
        parent_path, parent_entry, parent_payload, m2_boxes, m2_interval = legacy._load_m2_parent(
            cid, rec, contexts)
        _need(list(m2_interval) == list(temp_record["selected_interval"]), f"M2 parent interval mismatch: {cid}")

        file_row = by_id[cid]
        out_path = _read_artifact_path(file_row, "path")
        import torch
        payload = torch.load(out_path, map_location="cpu", weights_only=False)
        _validate_case(payload, cid, dataset, case, source_path, source_row, ref_record, temp_record,
                       lock, cached_fit, no_fit_record, m2_boxes, "A000")
        _need(payload.get("source_model_state_sha256") == case.get("source_model_state_sha256")
              and payload.get("temporal_cache_path") == str(cache_path.relative_to(ROOT))
              and payload.get("temporal_cache_sha256") == cache_sha
              and payload.get("baseline_spatial_student_path") == str(parent_path.relative_to(ROOT))
              and payload.get("baseline_spatial_student_sha256") == parent_entry.get("sha256")
              and parent_payload.get("source_restored") is True,
              f"cache/M2 parent binding mismatch: {cid}")

        candidate_row = lock_records_by_id[cid]
        expected_lite_interval = (list(expected_lite["final_interval"]) if cached_fit is not None
                                  else list(temp_record["selected_interval"]))
        expected_baseline_interval = (list(cached_fit["final_interval"]) if cached_fit is not None
                                      else list(temp_record["selected_interval"]))
        _need(candidate_row.get("dataset") == dataset
              and candidate_row.get("source_case_sha256") == case.get("source_case_sha256")
              and candidate_row.get("source_file_sha256") == src_sha
              and candidate_row.get("source_temporal_sha256") == cache_sha
              and candidate_row.get("temporal_lite_interval") == expected_lite_interval
              and candidate_row.get("baseline_offset_trust_interval") == expected_baseline_interval
              and candidate_row.get("matched_cpu_fit_seconds") == payload.get("matched_cpu_fit_seconds")
              and candidate_row.get("selected_interval") == expected_lite_interval
              and candidate_row.get("selected_variant") == "temporal_lite_candidate",
              f"candidate TEMPORAL_LOCK row mismatch: {cid}")
        if cached_fit is not None:
            _need(legacy._exact_equal(payload["fits"]["temporal_lite"], expected_lite),
                  f"independent Temporal-Lite replay differs from saved output: {cid}")
            lite_replay_count += 1
            fit_count += 1
        else:
            no_fit_count += 1
        domain_counts[dataset] += 1

        receipt_path = _read_artifact_path(file_row, "receipt_path")
        receipt = read(receipt_path)
        _need(receipt.get("schema") == "stvg-temporal-lite-case-receipt-v1"
              and receipt.get("status") == "complete"
              and receipt.get("case_id") == cid and receipt.get("dataset") == dataset
              and receipt.get("sha256") == file_row.get("sha256")
              and receipt.get("source_case_sha256") == case.get("source_case_sha256")
              and receipt.get("temporal_cache_sha256") == cache_sha
              and receipt.get("GT_read") is False,
              f"receipt content mismatch: {cid}")
        final_rows.append(dict(file_row))

    _need(fit_count == FIT_COUNT and no_fit_count == NO_FIT_COUNT and lite_replay_count == FIT_COUNT,
          f"expected 111 fitted and 17 no-fit records, got {fit_count}/{no_fit_count}/{lite_replay_count}")
    _need(domain_counts == {"hc2": 64, "vidstg": 64}, f"domain counts mismatch: {domain_counts}")

    readback_obj = {
        "schema": "stvg-temporal-lite-independent-root-readback-v1",
        "status": "pass",
        "scope": "full 128-case pre-GT output, original cache lineage, exact baseline fit, independent Temporal-Lite CPU replay, and conditional M2 box readback",
        "count": TOTAL_COUNT,
        "domain_counts": domain_counts,
        "baseline_exact_replay_count": fit_count,
        "temporal_lite_independent_replay_count": lite_replay_count,
        "explicit_original_no_fit_fallback_count": no_fit_count,
        "files": final_rows,
        "cpu_output_barrier_sha256": sha(CPU_BARRIER),
        "temporal_lock_sha256": sha(TEMPORAL_LOCK),
        "runtime_lock_sha256": sha(RUNTIME_LOCK),
        "input_bindings": _bindings(lock),
        "GT_read": False,
        "CPU_only": True,
        "GPU_used": False,
        "model_loaded": False,
        "actual_scoring_performed": False,
        "full_visual_model_decode_replayed": False,
    }
    _write_once(READBACK, readback_obj)
    prediction_obj = {
        "schema": "stvg-temporal-lite-pre-gt-prediction-barrier-v1",
        "status": "complete",
        "count": TOTAL_COUNT,
        "domain_counts": domain_counts,
        "files": final_rows,
        "receipts": [{"path": row["receipt_path"], "sha256": row["receipt_sha256"], "bytes": (ROOT / row["receipt_path"]).stat().st_size}
                     for row in final_rows],
        "cpu_output_barrier_sha256": sha(CPU_BARRIER),
        "root_readback_sha256": sha(READBACK),
        "temporal_lock_sha256": sha(TEMPORAL_LOCK),
        "runtime_lock_sha256": sha(RUNTIME_LOCK),
        "GT_read": False,
        "scoring_performed": False,
    }
    _write_once(PREDICTION_BARRIER, prediction_obj)
    return {"status": "pass", "count": TOTAL_COUNT, "domain_counts": domain_counts,
            "baseline_exact_replay_count": fit_count,
            "temporal_lite_independent_replay_count": lite_replay_count,
            "no_fit_fallback_count": no_fit_count,
            "prediction_barrier_sha256": sha(PREDICTION_BARRIER)}


if __name__ == "__main__":
    print(json.dumps(run_readback(), indent=2, sort_keys=True))
