#!/usr/bin/env python3
"""Sealed post-readback scorer for spatial component and temporal component trials.

No scoring is performed unless the corresponding 128-case prediction seal and
passing CPU readback are already present. The optional ``--temporal`` mode
scores conditional readouts using the fixed spatial M2 boxes.
"""
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
from scripts.stvg_component_ablation_common_v1 import BASE, SEARCH, read, roster, load_case, sha
from scripts.run_stvg_component_temporal005 import OUT as TEMP_OUT, _verify_locks, _file_rows, _cid

METRICS = ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT", "m_sIoU_supervised_GT", "m_sIoU_unsupervised_GT")
NATIVE = BASE / 'native_clock/revision002'
SPATIAL_ARMS = ("F", "T", "S_M2", "TS_M2")
TEMPORAL_ARMS = ("F", "T", "S_M2", "TS_M2", "TS_offset_trust", "TS_no_grid_trust",
                 "TS_no_temporal_l2", "TS_shared_offset_candidate", "TS_projection_only")
TOL_PP = 1e-10


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def _write_once(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _need(not path.exists() and not path.with_name(path.name + ".tmp").exists(), f"refusing to overwrite {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _filemap(doc: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    rows = doc.get("files")
    _need(isinstance(rows, list), f"{label}.files must be a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        rel, expected = row.get("path"), row.get("sha256")
        _need(isinstance(rel, str) and isinstance(expected, str), f"invalid {label} file row")
        path = (ROOT / rel).resolve()
        _need(ROOT in path.parents and path.is_file() and sha(path) == expected, f"{label} file SHA mismatch: {rel}")
        if row.get("bytes") is not None:
            _need(path.stat().st_size == int(row["bytes"]), f"{label} file bytes mismatch: {rel}")
        _need(rel not in result, f"duplicate {label} file path: {rel}")
        result[rel] = row
    return result


def _aggregate(rows: list[dict[str, Any]], arms: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by = {(r["case_id"], r["arm"]): r for r in rows}
    _need(len(by) == len(rows), "duplicate case/arm score row")
    summary = []
    for ds in ("hc2", "vidstg"):
        cases = sorted({r["case_id"] for r in rows if r["dataset"] == ds})
        _need(len(cases) == 64, f"{ds} expected 64 source cases")
        for arm in arms:
            for metric in METRICS:
                vals, unknown = [], []
                for cid in cases:
                    val = by[cid, arm]["metrics"].get(metric)
                    if val is None:
                        unknown.append(cid)
                    else:
                        _need(math.isfinite(float(val)), f"nonfinite {metric} at {cid}/{arm}")
                        vals.append(float(val))
                summary.append({"dataset": ds, "arm": arm, "metric": metric,
                                "n_valid": len(vals), "n_unknown": len(unknown),
                                "unknown_case_ids": unknown,
                                "mean": math.fsum(vals) / len(vals) if vals else None,
                                "median": statistics.median(vals) if vals else None,
                                "min": min(vals) if vals else None, "max": max(vals) if vals else None})
            vals = [by[cid, arm]["metrics"].get("m_vIoU") for cid in cases]
            for threshold in (0.3, 0.5):
                known = [float(v) for v in vals if v is not None]
                summary.append({"dataset": ds, "arm": arm, "metric": f"R@vIoU>={threshold:.1f}",
                                "n_valid": len(known), "n_unknown": len(vals) - len(known),
                                "unknown_case_ids": [cid for cid in cases if by[cid, arm]["metrics"].get("m_vIoU") is None],
                                "mean": (sum(v >= threshold for v in known) / len(known)) if known else None,
                                "threshold_inclusive": True,
                                "derived_from_official_per_case_m_vIoU": True})

    contrasts = []
    definitions = [(f"TS_{name}_minus_TS_offset_trust", f"TS_{name}", "TS_offset_trust")
                   for name in ("no_grid_trust", "no_temporal_l2", "shared_offset_candidate", "projection_only")]
    definitions += [("S_M2_minus_F", "S_M2", "F"), ("TS_M2_minus_T", "TS_M2", "T"),
                    ("TS_offset_trust_minus_TS_M2", "TS_offset_trust", "TS_M2")]
    for ds in ("hc2", "vidstg"):
        cases = sorted({r["case_id"] for r in rows if r["dataset"] == ds})
        for name, left, right in definitions:
            if left not in arms or right not in arms:
                continue
            for metric in METRICS:
                deltas, unknown = {}, []
                for cid in cases:
                    a = by[cid, left]["metrics"].get(metric)
                    b = by[cid, right]["metrics"].get(metric)
                    if a is None or b is None:
                        unknown.append(cid)
                    else:
                        deltas[cid] = 100.0 * (float(a) - float(b))
                contrasts.append({"dataset": ds, "contrast": name, "comparison_arm": left,
                                  "baseline_arm": right, "metric": metric,
                                  "n_pairs": len(deltas), "unknown_count": len(unknown),
                                  "unknown_case_ids": unknown,
                                  "positive_case_ids": sorted(c for c, v in deltas.items() if v > TOL_PP),
                                  "negative_case_ids": sorted(c for c, v in deltas.items() if v < -TOL_PP),
                                  "neutral_case_ids": sorted(c for c, v in deltas.items() if abs(v) <= TOL_PP),
                                  "neutral_tolerance_pp": TOL_PP,
                                  "mean_delta_pp": math.fsum(deltas.values()) / len(deltas) if deltas else None,
                                  "median_delta_pp": statistics.median(deltas.values()) if deltas else None,
                                  "worst_delta_pp": min(deltas.values()) if deltas else None,
                                  "per_source_delta_pp": deltas})
    return summary, contrasts


def _spatial_inputs(trial_id: str):
    from scripts.stvg_component_ablation_common_v1 import verify_final, verify_preparation
    final = verify_final(); verify_preparation()
    trial = NATIVE / "trials" / trial_id
    for name in ("DESIGN_LOCK.json", "RUNTIME.json", "ROOT_READBACK.json", "STUDENT_GLOBAL_BARRIER.json", "GLOBAL_PREDICTION_BARRIER.json"):
        _need((trial / name).is_file(), f"spatial scoring requires {name}")
    runtime = read(trial / "RUNTIME.json")
    _need(runtime.get("status") == "locked" and runtime.get("GT_read") is False, "trial runtime not locked/GT-free")
    for rel, expected in {**runtime.get("pins", {}), **runtime.get("protected", {})}.items():
        _need(sha(ROOT / rel) == expected, f"trial runtime pin mismatch: {rel}")
    root_readback = read(trial / "ROOT_READBACK.json")
    _need(root_readback.get("status") == "pass" and root_readback.get("count") == 128 and root_readback.get("GT_read") is False,
          "spatial scoring requires passing full 128-case pre-GT readback")
    student_barrier = read(trial / "STUDENT_GLOBAL_BARRIER.json")
    _need(student_barrier.get("status") == "complete" and student_barrier.get("count") == 128 and student_barrier.get("GT_read") is False,
          "spatial student barrier incomplete")
    pred = read(trial / "GLOBAL_PREDICTION_BARRIER.json")
    _need(pred.get("status") == "complete" and pred.get("count") == 128 and pred.get("GT_read") is False,
          "spatial prediction barrier incomplete")
    _need(pred.get("student_barrier_sha256") == sha(trial / "STUDENT_GLOBAL_BARRIER.json")
          and pred.get("root_readback_sha256") == sha(trial / "ROOT_READBACK.json"),
          "spatial prediction barrier does not bind readback/student seal")
    _need(pred.get("scorer_sha256") == sha(Path(__file__)), "spatial prediction barrier scorer SHA mismatch")
    out = _filemap(student_barrier, "student barrier")
    _need(len(out) == 128, "spatial student barrier file count mismatch")
    pred_files = _filemap(pred, "prediction barrier")
    _need(set(out) <= set(pred_files) and all(out[k]["sha256"] == pred_files[k]["sha256"] for k in out),
          "spatial prediction files differ from readback student barrier")
    return final, trial, student_barrier, pred, out


def _temporal_inputs():
    from scripts.stvg_component_ablation_common_v1 import verify_final, verify_preparation
    final = verify_final(); verify_preparation()
    barrier_path = TEMP_OUT / "TEMPORAL_COMPONENT_GLOBAL_BARRIER.json"
    readback_path = TEMP_OUT / "ROOT_TEMPORAL_COMPONENT_READBACK.json"
    pred_path = TEMP_OUT / "PREDICTION_TEMPORAL_COMPONENT_BARRIER.json"
    for path in (barrier_path, readback_path, pred_path):
        _need(path.is_file(), f"temporal scoring requires {path.name}")
    readback = read(readback_path)
    _need(readback.get("status") == "pass" and readback.get("count") == 128 and readback.get("GT_read") is False,
          "temporal CPU readback incomplete")
    barrier = read(barrier_path); pred = read(pred_path)
    _need(barrier.get("status") == "complete" and barrier.get("count") == 128 and barrier.get("GT_read") is False,
          "temporal component barrier incomplete")
    _need(pred.get("status") == "complete" and pred.get("count") == 128 and pred.get("GT_read") is False,
          "temporal prediction barrier incomplete")
    _need(readback.get("temporal_component_barrier_sha256") == sha(barrier_path)
          and pred.get("temporal_component_barrier_sha256") == sha(barrier_path)
          and pred.get("root_temporal_component_readback_sha256") == sha(readback_path),
          "temporal prediction/readback binding mismatch")
    _need(barrier.get("bindings", {}).get("final_parameter_lock_sha256") == sha(BASE / "FINAL_PARAMETER_LOCK.json")
          and barrier.get("bindings", {}).get("temporal_component_runtime_sha256") == sha(BASE / "TEMPORAL_COMPONENT_RUNTIME005.json"),
          "temporal seal is not bound to final/runtime lock")
    files = _filemap(barrier, "temporal component barrier")
    receipts = pred.get("receipts")
    _need(isinstance(receipts, list) and len(receipts) == 128, "temporal prediction receipt set incomplete")
    return final, barrier, pred, files, readback


def _load_case_map():
    cases = {}
    for rec in roster():
        case, source_path = load_case(rec)
        case = dict(case)
        case.setdefault("dataset", rec["dataset"])
        _need(case.get("GT_read") is False and case.get("dataset") == rec["dataset"], f"source case not GT-free: {_cid(rec)}")
        cases[_cid(rec)] = (rec, case, source_path)
    _need(len(cases) == 128, "Coordinate128 source case count mismatch")
    return cases


def _load_predictions(files: Mapping[str, Mapping[str, Any]], expected_arms: tuple[str, ...], temporal: bool):
    import torch
    cases = _load_case_map()
    ids = set(cases)
    entries_by_id = {}
    for rel, row in files.items():
        path = ROOT / rel
        try:
            z = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"cannot read sealed student payload {rel}") from exc
        cid = z.get("case_id")
        _need(cid in ids and cid not in entries_by_id, f"unexpected/duplicate case payload {cid}")
        rec, case, source_path = cases[cid]
        _need(z.get("dataset") == rec["dataset"] and z.get("source_case_sha256") == case.get("source_case_sha256"),
              f"source binding mismatch for {cid}")
        _need(z.get("GT_read") is False and z.get("source_restored") is True, f"prediction not GT-free/restored for {cid}")
        _need(z.get("source_file_sha256") == sha(source_path), f"source file SHA mismatch for {cid}")
        arms = z.get("arms")
        _need(isinstance(arms, dict) and set(arms) == set(expected_arms), f"prediction arm set mismatch for {cid}")
        for arm_name, arm in arms.items():
            boxes = arm.get("boxes")
            arr = np.asarray(boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else boxes)
            _need(arr.shape == (len(case["frame_ids"]), 4) and np.isfinite(arr).all(), f"invalid arm tube {cid}/{arm_name}")
            interval = np.asarray(arm.get("interval"), dtype=np.float64)
            bounds = np.asarray(case["video_bounds"], dtype=np.float64)
            _need(interval.shape == (2,) and np.isfinite(interval).all() and bounds[0] <= interval[0] < interval[1] <= bounds[1],
                  f"invalid arm interval {cid}/{arm_name}")
        entries_by_id[cid] = (z, rec, case)
    _need(set(entries_by_id) == ids, "sealed prediction set differs from 128-case roster")
    return cases, entries_by_id


def _baseline_comparison(final: Mapping[str, Any], cases: Mapping[str, Any], predictions: Mapping[str, Any], temporal: bool):
    import torch
    comparisons = []
    rows_by_trial = {}
    for ds in ("hc2", "vidstg"):
        trial_id = final["actual_baseline_trial"][ds]
        trial_dir = SEARCH / "trials" / trial_id
        baseline_barrier = read(trial_dir / "STUDENT_GLOBAL_BARRIER.json")
        baseline_files = _filemap(baseline_barrier, f"baseline {ds} barrier")
        baseline_rows_path = trial_dir / "POSTSEAL_SCORE_ROWS.json"
        baseline_summary_path = trial_dir / "POSTSEAL_SCORE_SUMMARY.json"
        final_pins = final.get("pins", {})
        for pinned_path in (baseline_rows_path, baseline_summary_path):
            rel = str(pinned_path.relative_to(ROOT))
            _need(pinned_path.is_file(), f"actual baseline score artifact missing: {rel}")
            _need(final_pins.get(rel) == sha(pinned_path), f"FINAL_PARAMETER_LOCK does not pin baseline score artifact: {rel}")
        rows_doc = read(baseline_rows_path)
        summary_doc = read(baseline_summary_path)
        _need(rows_doc.get("status") == "complete" and rows_doc.get("GT_read") is True,
              f"actual baseline score rows are not complete: {ds}")
        _need(summary_doc.get("status") == "complete" and summary_doc.get("GT_read") is True,
              f"actual baseline score summary is not complete: {ds}")
        baseline_rows_list = rows_doc.get("rows")
        _need(isinstance(baseline_rows_list, list) and len(baseline_rows_list) == 128 * 8,
              f"actual baseline score row count mismatch: {ds}")
        baseline_rows = {(r["case_id"], r["arm"]): r for r in baseline_rows_list}
        _need(len(baseline_rows) == len(baseline_rows_list), f"duplicate actual baseline score row: {ds}")
        _need(len(summary_doc.get("summary", [])) > 0, f"actual baseline score summary empty: {ds}")
        for cid, (payload, rec, case) in predictions.items():
            if rec["dataset"] != ds:
                continue
            rel = str((trial_dir / "student" / f"{cid}.pt").relative_to(ROOT))
            entry = baseline_files.get(rel)
            _need(entry is not None, f"actual baseline arm lacks sealed entry for {cid}")
            pp = ROOT / rel
            _need(sha(pp) == entry["sha256"], f"actual baseline M2 SHA mismatch for {cid}")
            parent = torch.load(pp, map_location="cpu", weights_only=False)
            a = payload["arms"]["TS_M2"]
            b = parent["arms"]["TS_M2"]
            aa = np.asarray(a["boxes"].detach().cpu().numpy() if hasattr(a["boxes"], "detach") else a["boxes"])
            bb = np.asarray(b["boxes"].detach().cpu().numpy() if hasattr(b["boxes"], "detach") else b["boxes"])
            exact_boxes = aa.dtype == bb.dtype and aa.shape == bb.shape and np.array_equal(aa, bb)
            exact_interval = list(a["interval"]) == list(b["interval"])
            row = {"case_id": cid, "dataset": ds, "actual_baseline_trial": trial_id,
                   "boxes_exact": bool(exact_boxes), "interval_exact": bool(exact_interval),"prediction_exact":bool(exact_boxes and exact_interval)}
            old = baseline_rows.get((cid, "TS_M2"))
            _need(old is not None, f"actual baseline TS_M2 score row missing for {cid}")
            row["baseline_metrics_available"] = True
            row["baseline_metrics"] = {m: old.get("metrics", {}).get(m) for m in METRICS}
            comparisons.append(row)
        rows_by_trial[ds] = trial_id
    return comparisons, rows_by_trial


def _score(temporal: bool, trial_id: str | None = None) -> dict[str, Any]:
    if temporal:
        final, barrier, prediction, files, readback = _temporal_inputs()
        arms = TEMPORAL_ARMS
        score_dir = TEMP_OUT
        stem = "TEMPORAL_COMPONENT"
        expected_scorer_sha = None
        bindings = {
            "prediction_barrier_sha256": sha(TEMP_OUT / "PREDICTION_TEMPORAL_COMPONENT_BARRIER.json"),
            "temporal_component_barrier_sha256": sha(TEMP_OUT / "TEMPORAL_COMPONENT_GLOBAL_BARRIER.json"),
            "cpu_readback_sha256": sha(TEMP_OUT / "ROOT_TEMPORAL_COMPONENT_READBACK.json"),
            "runtime_sha256": sha(BASE / "TEMPORAL_COMPONENT_RUNTIME005.json"),
            "final_parameter_lock_sha256": sha(BASE / "FINAL_PARAMETER_LOCK.json"),
        }
    else:
        if not trial_id:
            raise ValueError("spatial score mode requires --trial TRIAL_ID")
        final, trial, student_barrier, prediction, files = _spatial_inputs(trial_id)
        arms = SPATIAL_ARMS
        score_dir = trial
        stem = "POSTSEAL"
        readback = read(trial / "ROOT_READBACK.json")
        expected_scorer_sha = sha(Path(__file__))
        bindings = {"prediction_barrier_sha256": sha(trial / "GLOBAL_PREDICTION_BARRIER.json"),
                    "student_barrier_sha256": sha(trial / "STUDENT_GLOBAL_BARRIER.json"),
                    "root_readback_sha256": sha(trial / "ROOT_READBACK.json"),
                    "trial_runtime_sha256": sha(trial / "RUNTIME.json"),
                    "final_parameter_lock_sha256": sha(BASE / "FINAL_PARAMETER_LOCK.json")}
        _need(prediction.get("scorer_sha256") == expected_scorer_sha, "prediction seal scorer hash mismatch")

    output_paths = [score_dir / f"{stem}_ROWS.json", score_dir / f"{stem}_SUMMARY.json",
                    score_dir / f"{stem}_PAIRED.json", score_dir / f"{stem}_GT_EXPOSURE.json",
                    score_dir / f"{stem}_SCORE_BARRIER.json"]
    _need(not any(p.exists() for p in output_paths), "score output already exists; refusing repeat GT scoring")
    cases, predictions = _load_predictions(files, arms, temporal)
    from scripts import review_stvg_sparse_r1_v2 as truth_reader
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    exposure_path = score_dir / f"{stem}_GT_EXPOSURE.json"
    _write_once(exposure_path, {"schema": "stvg-component-postseal-gt-exposure-v1", "status": "after_128_case_seal_and_readback",
                                "bindings": bindings, "case_count": 128, "GT_read": True, "time": time.time()})
    source_cases = {cid: item[1] for cid, item in cases.items()}
    truths, _ = truth_reader._truths_after_seal(source_cases)
    official, independent, _contract = truth_reader._metric_functions()
    ref_doc = read(NATIVE / "REFERENCE_LOCK.json")
    refs = {_cid(r): r for r in ref_doc["records"]}
    rows = []
    for rec in roster():
        cid = _cid(rec)
        payload, _, case = predictions[cid]
        support = {int(a["position"]) for a in refs[cid].get("anchors", []) if float(a.get("weight", 1.0)) > 0}
        for arm_name in arms:
            arm = payload["arms"][arm_name]
            score, audit = metric._score_student_arm(arm, case, truths[cid], support, official, independent)
            rows.append({"case_id": cid, "dataset": rec["dataset"], "arm": arm_name,
                         "status": "scored", "metrics": score,
                         "support": audit["support"], "metric_audit": audit["metric_audit"]})
    _need(len(rows) == 128 * len(arms), "score row count mismatch")
    summary, contrasts = _aggregate(rows, arms)
    if temporal:
        baseline_comparison = None
        summary_path = score_dir / "TEMPORAL_COMPONENT_SCORE_SUMMARY.json"
        rows_path = score_dir / "TEMPORAL_COMPONENT_SCORE_ROWS.json"
        paired_path = score_dir / "TEMPORAL_COMPONENT_SCORE_PAIRED.json"
        barrier_out = score_dir / "TEMPORAL_COMPONENT_SCORE_BARRIER.json"
    else:
        baseline_comparison, baseline_trials = _baseline_comparison(final, cases, predictions, temporal=False)
        current_by = {(r["case_id"], r["arm"]): r for r in rows}
        comp_summary = []
        for comp in baseline_comparison:
            current = current_by.get((comp["case_id"], "TS_M2"))
            if current is not None and comp.get("baseline_metrics_available"):
                deltas = {}
                for met in METRICS:
                    x, y = current["metrics"].get(met), comp["baseline_metrics"].get(met)
                    deltas[met] = None if x is None or y is None else 100.0 * (float(x) - float(y))
                comp["paired_delta_pp"] = deltas
        for ds in ("hc2", "vidstg"):
            domain = [x for x in baseline_comparison if x["dataset"] == ds]
            for key in ("boxes_exact", "interval_exact","prediction_exact"):
                comp_summary.append({"dataset": ds, "comparison": "TS_M2_vs_actual_baseline_TS_M2",
                                     "field": key, "exact_count": sum(bool(x[key]) for x in domain), "count": len(domain)})
            for met in METRICS:
                vals = [x.get("paired_delta_pp", {}).get(met) for x in domain]
                known = [float(x) for x in vals if x is not None]
                comp_summary.append({"dataset": ds, "comparison": "TS_M2_vs_actual_baseline_TS_M2",
                                     "metric": met, "n_pairs": len(known), "n_unknown": len(vals)-len(known),
                                     "positive": sum(x > TOL_PP for x in known),
                                     "negative": sum(x < -TOL_PP for x in known),
                                     "neutral": sum(abs(x) <= TOL_PP for x in known),
                                     "mean_delta_pp": math.fsum(known)/len(known) if known else None,
                                     "median_delta_pp": statistics.median(known) if known else None,
                                     "worst_delta_pp": min(known) if known else None})
        summary_path = score_dir / "POSTSEAL_SUMMARY.json"
        rows_path = score_dir / "POSTSEAL_ROWS.json"
        paired_path = score_dir / "POSTSEAL_PAIRED.json"
        barrier_out = score_dir / "POSTSEAL_SCORE_BARRIER.json"
        _write_once(score_dir / "ACTUAL_BASELINE_COMPARISON.json", {
            "schema": "stvg-component-actual-baseline-comparison-v1", "status": "complete",
            "records": baseline_comparison, "summary": comp_summary,
            "baseline_trials": baseline_trials, "GT_read": True,
            "scope": "same-source TS_M2 geometry/interval and same-score per-source comparison to FINAL_PARAMETER_LOCK actual_baseline_trial"})
    common = {"schema": "stvg-component-postseal-v1", "status": "complete", "case_count": 128,
              "row_count": len(rows), "arm_names": list(arms), "metric_names": list(METRICS),
              "bindings": bindings, "GT_read": True, "bootstrap_executed": False}
    _write_once(rows_path, dict(common, rows=rows))
    _write_once(summary_path, dict(common, summary=summary))
    _write_once(paired_path, dict(common, paired_contrasts=contrasts))
    score_barrier = {
        "schema": "stvg-component-postseal-score-barrier-v1", "status": "complete",
        "count": len(rows), "case_count": 128, "arm_count": len(arms), "GT_read": True,
        "rows_path": str(rows_path.relative_to(ROOT)), "rows_sha256": sha(rows_path),
        "summary_path": str(summary_path.relative_to(ROOT)), "summary_sha256": sha(summary_path),
        "paired_path": str(paired_path.relative_to(ROOT)), "paired_sha256": sha(paired_path),
        "gt_exposure_sha256": sha(exposure_path), "prediction_bindings": bindings,
        "scorer_sha256": sha(Path(__file__)), "time": time.time(),
    }
    _write_once(barrier_out, score_barrier)
    completion_path = score_dir / ("SCORE_CPU_COMPLETION.json" if temporal else "CPU_COMPLETION.json")
    _write_once(completion_path, dict(status="complete",count=128,row_count=len(rows),
        score_barrier_sha256=sha(barrier_out),GT_read=True,CPU_only=True,GPU_used=False,
        bootstrap_executed=False,time=time.time()))
    return {"rows": len(rows), "summary_cells": len(summary), "paired_cells": len(contrasts),
            "arms": list(arms), "GT_read": True, "barrier_sha256": sha(barrier_out)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", help="spatial component trial ID, e.g. A000")
    parser.add_argument("--temporal", action="store_true", help="score sealed conditional temporal component outputs")
    parser.add_argument("--score", action="store_true", help="required explicit switch for post-seal GT scoring")
    args = parser.parse_args()
    if not args.score:
        parser.error("GT scoring requires explicit --score after independent root readback")
    if args.temporal and args.trial:
        parser.error("--trial and --temporal are mutually exclusive")
    result = _score(temporal=args.temporal, trial_id=args.trial)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
