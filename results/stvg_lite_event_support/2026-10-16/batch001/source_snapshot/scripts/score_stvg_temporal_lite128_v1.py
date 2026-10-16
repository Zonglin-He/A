#!/usr/bin/env python3
"""Score sealed Temporal-Lite conditional readouts after the full pre-GT seal."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stvg_component_ablation_common_v1 import read, sha
from scripts import run_stvg_component_temporal005 as legacy
from scripts import run_stvg_temporal_lite128_v1 as temporal_runner
from scripts import stvg_temporal_lite_fit_v1 as temporal_lite

SCOPE = ROOT / "artifacts" / "stvg_lite_event_support_v1"
OUT = SCOPE / "temporal"
CPU_BARRIER = OUT / "CPU_OUTPUT_BARRIER.json"
TEMPORAL_LOCK = OUT / "TEMPORAL_LOCK.json"
ROOT_READBACK = OUT / "ROOT_READBACK.json"
PREDICTION_BARRIER = OUT / "PREDICTION_GLOBAL_BARRIER.json"
RUNTIME_LOCK = OUT / "TEMPORAL_LITE_RUNTIME_LOCK.json"
ARMS = ("F", "T", "S_M2", "TS_M2", "TS_offset_trust", "TS_temporal_lite")
METRICS = ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT", "m_sIoU_supervised_GT", "m_sIoU_unsupervised_GT")
TOL_PP = 1e-10


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _need(not path.exists() and not path.with_name(path.name + ".tmp").exists(),
          f"refusing to overwrite existing/partial post-seal artifact: {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by = {(row["case_id"], row["arm"]): row for row in rows}
    _need(len(by) == len(rows), "duplicate scored case/arm row")
    summaries: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    for dataset in ("hc2", "vidstg"):
        cases = sorted({row["case_id"] for row in rows if row["dataset"] == dataset})
        _need(len(cases) == 64, f"{dataset} expected 64 source cases")
        for arm in ARMS:
            for metric in METRICS:
                values: list[float] = []
                unknown: list[str] = []
                for cid in cases:
                    value = by[cid, arm]["metrics"].get(metric)
                    if value is None:
                        unknown.append(cid)
                    else:
                        _need(math.isfinite(float(value)), f"nonfinite {metric} at {cid}/{arm}")
                        values.append(float(value))
                summaries.append({
                    "dataset": dataset, "arm": arm, "metric": metric,
                    "n_valid": len(values), "n_unknown": len(unknown),
                    "unknown_case_ids": unknown,
                    "mean": math.fsum(values) / len(values) if values else None,
                    "median": statistics.median(values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                })
            values = [by[cid, arm]["metrics"].get("m_vIoU") for cid in cases]
            for threshold in (0.3, 0.5):
                known = [float(value) for value in values if value is not None]
                summaries.append({
                    "dataset": dataset, "arm": arm, "metric": f"R@vIoU>={threshold:.1f}",
                    "n_valid": len(known), "n_unknown": len(values) - len(known),
                    "unknown_case_ids": [cid for cid in cases if by[cid, arm]["metrics"].get("m_vIoU") is None],
                    "mean": sum(value >= threshold for value in known) / len(known) if known else None,
                    "threshold_inclusive": True,
                    "derived_from_official_per_case_m_vIoU": True,
                })

        definitions = (
            ("TS_temporal_lite_minus_TS_offset_trust", "TS_temporal_lite", "TS_offset_trust"),
            ("TS_temporal_lite_minus_TS_M2", "TS_temporal_lite", "TS_M2"),
            ("TS_offset_trust_minus_TS_M2", "TS_offset_trust", "TS_M2"),
            ("S_M2_minus_F", "S_M2", "F"),
            ("TS_M2_minus_T", "TS_M2", "T"),
        )
        for contrast, left, right in definitions:
            for metric in METRICS:
                deltas: dict[str, float] = {}
                unknown: list[str] = []
                for cid in cases:
                    a = by[cid, left]["metrics"].get(metric)
                    b = by[cid, right]["metrics"].get(metric)
                    if a is None or b is None:
                        unknown.append(cid)
                    else:
                        deltas[cid] = 100.0 * (float(a) - float(b))
                values = list(deltas.values())
                contrasts.append({
                    "dataset": dataset, "contrast": contrast,
                    "comparison_arm": left, "baseline_arm": right,
                    "metric": metric, "n_pairs": len(deltas),
                    "unknown_count": len(unknown), "unknown_case_ids": unknown,
                    "positive_case_ids": sorted(cid for cid, value in deltas.items() if value > TOL_PP),
                    "negative_case_ids": sorted(cid for cid, value in deltas.items() if value < -TOL_PP),
                    "neutral_case_ids": sorted(cid for cid, value in deltas.items() if abs(value) <= TOL_PP),
                    "neutral_tolerance_pp": TOL_PP,
                    "mean_delta_pp": math.fsum(values) / len(values) if values else None,
                    "median_delta_pp": statistics.median(values) if values else None,
                    "worst_delta_pp": min(values) if values else None,
                    "per_source_delta_pp": deltas,
                })
    return summaries, contrasts


def _preseal_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    from scripts import stvg_lite_event_common_v1 as phase
    phase.verify()
    lock = legacy._verify_locks()
    lock["a000_context"] = temporal_runner._verify_a000_context(lock)
    design = read(temporal_runner.DESIGN_LOCK)
    runtime = read(RUNTIME_LOCK)
    _need(design.get("study_id") == "stvg_lite_event_support_v1"
          and design.get("status") == "locked" and design.get("GT_read") is False,
          "root design lock is not locked/GT-free for the expected study")
    _need(runtime.get("status") == "locked" and runtime.get("GT_read") is False
          and runtime.get("execution_authorized") is True,
          "Temporal-Lite runtime lock is not approved/GT-free")
    _need(runtime.get("input_bindings") == temporal_runner._input_bindings(lock),
          "runtime lock no longer matches original inputs and A000 spatial geometry")

    barrier = read(CPU_BARRIER)
    candidate_lock = read(TEMPORAL_LOCK)
    root_readback = read(ROOT_READBACK)
    prediction = read(PREDICTION_BARRIER)
    _need(barrier.get("status") == "complete" and barrier.get("count") == 128
          and barrier.get("GT_read") is False and barrier.get("CPU_only") is True,
          "full CPU output barrier is incomplete or not pre-GT")
    _need(root_readback.get("status") == "pass" and root_readback.get("count") == 128
          and root_readback.get("GT_read") is False,
          "independent root readback is incomplete or already GT-exposed")
    _need(prediction.get("status") == "complete" and prediction.get("count") == 128
          and prediction.get("GT_read") is False and prediction.get("scoring_performed") is False,
          "pre-GT prediction barrier is incomplete")
    _need(candidate_lock.get("schema") == "stvg-temporal-lite-candidate-temporal-lock-v1"
          and candidate_lock.get("status") == "complete"
          and candidate_lock.get("candidate_variant") == "temporal_lite"
          and candidate_lock.get("not_a_deployment_or_scored_method_selection") is True
          and candidate_lock.get("count") == 128
          and candidate_lock.get("cpu_output_barrier_sha256") == sha(CPU_BARRIER)
          and candidate_lock.get("runtime_lock_sha256") == sha(RUNTIME_LOCK)
          and candidate_lock.get("GT_read") is False,
          "candidate TEMPORAL_LOCK is incomplete or was selected after scoring")
    _need(barrier.get("bindings") == temporal_runner._input_bindings(lock)
          and barrier.get("runtime_lock_sha256") == sha(RUNTIME_LOCK)
          and root_readback.get("cpu_output_barrier_sha256") == sha(CPU_BARRIER)
          and root_readback.get("temporal_lock_sha256") == sha(TEMPORAL_LOCK)
          and prediction.get("cpu_output_barrier_sha256") == sha(CPU_BARRIER)
          and prediction.get("root_readback_sha256") == sha(ROOT_READBACK)
          and prediction.get("temporal_lock_sha256") == sha(TEMPORAL_LOCK)
          and prediction.get("runtime_lock_sha256") == sha(RUNTIME_LOCK),
          "pre-GT barrier/readback/runtime bindings mismatch")
    return lock, barrier, candidate_lock, root_readback, prediction


def score() -> dict[str, Any]:
    import torch
    from scripts import review_stvg_sparse_r1_v2 as truth_reader
    from scripts import score_stvg_native_instance_nativeclock005 as metric

    torch.set_num_threads(2)
    lock, barrier, candidate_lock, root_readback, prediction = _preseal_inputs()
    output_paths = [OUT / name for name in (
        "POSTSEAL_GT_EXPOSURE.json", "POSTSEAL_ROWS.json", "POSTSEAL_SUMMARY.json",
        "POSTSEAL_PAIRED.json", "POSTSEAL_SCORE_BARRIER.json", "CPU_COMPLETION.json",
    )]
    _need(not any(path.exists() or path.with_name(path.name + ".tmp").exists() for path in output_paths),
          "post-seal scoring outputs already exist; refusing repeat GT exposure")

    barrier_files = barrier.get("files")
    _need(isinstance(barrier_files, list) and len(barrier_files) == 128, "CPU barrier file list must be full")
    rows_by_id = {row.get("case_id"): row for row in barrier_files}
    roster = lock["roster"]
    roster_ids = {legacy._cid(row) for row in roster}
    _need(set(rows_by_id) == roster_ids, "prediction membership differs from locked roster")
    candidate_records = candidate_lock.get("records")
    _need(isinstance(candidate_records, list) and len(candidate_records) == 128
          and {row.get("case_id") for row in candidate_records} == roster_ids,
          "Temporal-Lite candidate lock membership mismatch")
    candidate_by_id = {row["case_id"]: row for row in candidate_records}
    if prediction.get("files") != barrier_files:
        raise RuntimeError("prediction barrier file rows differ from CPU output barrier")
    prediction_receipts = prediction.get("receipts")
    expected_receipts = [
        {"path": row["receipt_path"], "sha256": row["receipt_sha256"],
         "bytes": (ROOT / row["receipt_path"]).stat().st_size}
        for row in barrier_files
    ]
    _need(prediction_receipts == expected_receipts, "prediction barrier receipt set differs from the 128 output receipts")
    root_files = root_readback.get("files")
    _need(isinstance(root_files, list) and len(root_files) == 128, "root readback file list must be full")
    _need({row.get("case_id"): row for row in root_files} == rows_by_id,
          "root readback rows differ from output barrier")

    cases: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for rec in roster:
        cid = legacy._cid(rec)
        file_row = rows_by_id[cid]
        path = (ROOT / file_row["path"]).resolve()
        receipt_path = (ROOT / file_row["receipt_path"]).resolve()
        _need(ROOT in path.parents and ROOT in receipt_path.parents and path.is_file() and receipt_path.is_file(),
              f"prediction/receipt missing or outside workspace: {cid}")
        _need(sha(path) == file_row.get("sha256") and path.stat().st_size == int(file_row.get("bytes", -1))
              and sha(receipt_path) == file_row.get("receipt_sha256"), f"prediction/receipt SHA mismatch: {cid}")
        _need(receipt_path.stat().st_size == next(item["bytes"] for item in expected_receipts
                                                   if item["path"] == file_row["receipt_path"]),
              f"prediction receipt byte count mismatch: {cid}")
        case, source_path = legacy.load_case(rec)
        source_barrier_row = lock["source_files"][cid]
        _need(source_path == ROOT / rec["source_case_path"]
              and sha(source_path) == source_barrier_row.get("sha256")
              and case.get("source_case_sha256") == file_row.get("source_case_sha256")
              and file_row.get("source_file_sha256") == sha(source_path),
              f"source binding changed after seal: {cid}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _need(payload.get("GT_read") is False and payload.get("source_restored") is True
              and payload.get("case_id") == cid and payload.get("dataset") == rec["dataset"]
              and payload.get("source_case_sha256") == case.get("source_case_sha256"),
              f"prediction identity/GT/source binding mismatch: {cid}")
        _need(set(payload.get("arms", {})) == set(ARMS), f"six-arm output schema mismatch: {cid}")
        for arm_name in ARMS:
            arm = payload["arms"][arm_name]
            boxes = np.asarray(arm["boxes"].detach().cpu().numpy() if hasattr(arm["boxes"], "detach") else arm["boxes"])
            _need(boxes.shape == (len(case["frame_ids"]), 4) and np.isfinite(boxes).all(),
                  f"invalid sealed tube {cid}/{arm_name}")
            interval = np.asarray(arm.get("interval"), dtype=np.float64)
            bounds = np.asarray(case["video_bounds"], dtype=np.float64)
            _need(interval.shape == (2,) and np.isfinite(interval).all()
                  and bounds[0] <= interval[0] < interval[1] <= bounds[1],
                  f"invalid sealed interval {cid}/{arm_name}")
        _need(payload.get("conditional_TS_readouts_reuse_original_M2_boxes") is True
              and payload.get("baseline_spatial_trial_id") == "A000"
              and payload.get("full_visual_model_decode_replayed") is False,
              f"prediction is not the locked conditional A000 M2 readout: {cid}")
        candidate = candidate_by_id[cid]
        _need(candidate.get("temporal_lite_interval") == list(payload["arms"]["TS_temporal_lite"]["interval"])
              and candidate.get("baseline_offset_trust_interval") == list(payload["arms"]["TS_offset_trust"]["interval"]),
              f"candidate lock differs from sealed prediction: {cid}")
        cases[cid] = case
        payloads[cid] = payload

    bindings = {
        "prediction_barrier_sha256": sha(PREDICTION_BARRIER),
        "cpu_output_barrier_sha256": sha(CPU_BARRIER),
        "root_readback_sha256": sha(ROOT_READBACK),
        "candidate_temporal_lock_sha256": sha(TEMPORAL_LOCK),
        "runtime_lock_sha256": sha(RUNTIME_LOCK),
        "design_lock_sha256": sha(temporal_runner.DESIGN_LOCK),
    }
    _write_once(OUT / "POSTSEAL_GT_EXPOSURE.json", {
        "schema": "stvg-temporal-lite-postseal-gt-exposure-v1",
        "status": "after_full_128_prediction_barrier_and_independent_root_readback",
        "count": 128,
        "bindings": bindings,
        "GT_read": True,
        "time": time.time(),
    })

    truths, truth_provenance = truth_reader._truths_after_seal(cases)
    official, independent, metric_contract = truth_reader._metric_functions()
    refs = lock["references"]
    scored_rows: list[dict[str, Any]] = []
    for rec in roster:
        cid = legacy._cid(rec)
        case = cases[cid]
        payload = payloads[cid]
        support = {
            int(anchor["position"])
            for anchor in refs[cid].get("anchors", [])
            if float(anchor.get("weight", 1.0)) > 0
        }
        for arm_name in ARMS:
            metrics, audit = metric._score_student_arm(
                payload["arms"][arm_name], case, truths[cid], support, official, independent)
            scored_rows.append({
                "case_id": cid,
                "dataset": rec["dataset"],
                "arm": arm_name,
                "status": "scored",
                "metrics": metrics,
                "support": audit["support"],
                "metric_audit": audit["metric_audit"],
            })
    _need(len(scored_rows) == 128 * len(ARMS), "post-seal score row count mismatch")
    summaries, contrasts = _aggregate(scored_rows)
    common = {
        "schema": "stvg-temporal-lite-postseal-v1",
        "status": "complete",
        "case_count": 128,
        "row_count": len(scored_rows),
        "arm_names": list(ARMS),
        "metric_names": list(METRICS),
        "bindings": bindings,
        "GT_read": True,
        "bootstrap_executed": False,
        "conditional_spatial_reference": "original sealed A000 M2 boxes reused across temporal intervals",
        "full_visual_model_decode_replayed": False,
        "truth_provenance": truth_provenance,
        "metric_contract": metric_contract,
    }
    rows_path = OUT / "POSTSEAL_ROWS.json"
    summary_path = OUT / "POSTSEAL_SUMMARY.json"
    paired_path = OUT / "POSTSEAL_PAIRED.json"
    gt_path = OUT / "POSTSEAL_GT_EXPOSURE.json"
    barrier_path = OUT / "POSTSEAL_SCORE_BARRIER.json"
    completion_path = OUT / "CPU_COMPLETION.json"
    _write_once(rows_path, dict(common, rows=scored_rows))
    _write_once(summary_path, dict(common, summary=summaries))
    _write_once(paired_path, dict(common, paired_contrasts=contrasts))
    _write_once(barrier_path, {
        "schema": "stvg-temporal-lite-postseal-score-barrier-v1",
        "status": "complete",
        "count": len(scored_rows),
        "case_count": 128,
        "arm_count": len(ARMS),
        "GT_read": True,
        "rows_path": str(rows_path.relative_to(ROOT)),
        "rows_sha256": sha(rows_path),
        "summary_path": str(summary_path.relative_to(ROOT)),
        "summary_sha256": sha(summary_path),
        "paired_path": str(paired_path.relative_to(ROOT)),
        "paired_sha256": sha(paired_path),
        "gt_exposure_sha256": sha(gt_path),
        "prediction_bindings": bindings,
        "scorer_sha256": sha(Path(__file__)),
        "bootstrap_executed": False,
        "time": time.time(),
    })
    _write_once(completion_path, {
        "schema": "stvg-temporal-lite-postseal-cpu-completion-v1",
        "status": "complete",
        "count": 128,
        "row_count": len(scored_rows),
        "arm_count": len(ARMS),
        "score_barrier_sha256": sha(barrier_path),
        "GT_read": True,
        "CPU_only": True,
        "GPU_used": False,
        "bootstrap_executed": False,
        "time": time.time(),
    })
    return {"status": "complete", "count": 128, "row_count": len(scored_rows),
            "arm_count": len(ARMS), "summary_cells": len(summaries),
            "paired_cells": len(contrasts), "score_barrier_sha256": sha(barrier_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="store_true", help="read GT after the required full pre-GT root seal")
    args = parser.parse_args()
    if not args.score:
        parser.error("scoring requires explicit --score after the complete prediction and root readback seal")
    print(json.dumps(score(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
