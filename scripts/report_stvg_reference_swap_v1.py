"""Build the anonymous, post-seal E1 same-frame reference-swap report.

This is a reporting step only. It never loads videos, model weights, source
payload tensors, or GT annotations. It requires the scorer's completed,
post-seal scalar files and the already sealed E1 custody/readback receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_reference_recovery_v1/Reference_swap"
OUT = ROOT / "results/stvg_reference_swap/2026-10-15"
DOC = ROOT / "docs/STVG_REFERENCE_SWAP_V1.md"

ARMS = (
    "F", "T", "S_new_crop", "TS_new_crop", "S_WF_common",
    "TS_WF_common", "S_DINO_common", "TS_DINO_common",
)
METRICS = (
    "m_vIoU", "m_tIoU", "m_sIoU_dense_GT",
    "m_sIoU_supervised_GT", "m_sIoU_unsupervised_GT",
)
CONTRASTS = (
    ("S_DINO_common_minus_S_WF_common", "S_DINO_common", "S_WF_common"),
    ("TS_DINO_common_minus_TS_WF_common", "TS_DINO_common", "TS_WF_common"),
    ("S_WF_common_minus_S_new_crop", "S_WF_common", "S_new_crop"),
    ("TS_WF_common_minus_TS_new_crop", "TS_WF_common", "TS_new_crop"),
    ("S_WF_common_minus_F", "S_WF_common", "F"),
    ("TS_WF_common_minus_T", "TS_WF_common", "T"),
)
SIGN_EPSILON = 1e-10


class ReportError(RuntimeError):
    pass


def need(condition: Any, message: str) -> None:
    if not condition:
        raise ReportError(message)


def read(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
                ReportError(f"non-finite JSON value in {path}: {value}")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {path}: {exc}") from exc


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    if path.exists() and not replace:
        raise ReportError(f"refusing to overwrite {path}; pass --replace for an audited correction")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def alias(case_id: str) -> str:
    # These are the already-anonymous fixed roster aliases, not source IDs.
    need(case_id.startswith(("HC", "V")) and case_id[2:].isdigit(),
         f"unexpected non-anonymous E1 case alias: {case_id!r}")
    return case_id


def safe_mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def verify_and_build() -> dict[str, Any]:
    expected_files = [
        "DESIGN_LOCK.json", "ROSTER_LOCK.json", "REFERENCE_LOCK.json",
        "TEACHER_GLOBAL_BARRIER.json", "STUDENT_GLOBAL_BARRIER.json",
        "ROOT_READBACK.json", "GLOBAL_PREDICTION_BARRIER.json",
        "ROOT_TEACHER_READBACK.json", "ROOT_POSTSEAL_READBACK.json",
        "ROOT_POSTSEAL_SCALAR_READBACK.json", "ROOT_ACTUAL_EXECUTION_COST.json",
        "ROOT_INDEPENDENT_OPTIMIZER_MATH_READBACK.json",
        "POSTSEAL_ROWS.json", "POSTSEAL_SUMMARY.json", "PARENT_PAIRED_DELTAS.json",
        "POSTSEAL_REFERENCE_QUALITY.json", "GT_EXPOSURE.json",
    ]
    docs = {name: read(BASE / name) for name in expected_files}
    teacher = docs["TEACHER_GLOBAL_BARRIER.json"]
    student = docs["STUDENT_GLOBAL_BARRIER.json"]
    pre_root = docs["ROOT_READBACK.json"]
    prediction = docs["GLOBAL_PREDICTION_BARRIER.json"]
    teacher_readback = docs["ROOT_TEACHER_READBACK.json"]
    post_root = docs["ROOT_POSTSEAL_READBACK.json"]
    scalar_readback = docs["ROOT_POSTSEAL_SCALAR_READBACK.json"]
    execution_cost = docs["ROOT_ACTUAL_EXECUTION_COST.json"]
    optimizer_readback = docs["ROOT_INDEPENDENT_OPTIMIZER_MATH_READBACK.json"]
    score_rows = docs["POSTSEAL_ROWS.json"]
    score_summary = docs["POSTSEAL_SUMMARY.json"]
    score_pairs = docs["PARENT_PAIRED_DELTAS.json"]
    quality = docs["POSTSEAL_REFERENCE_QUALITY.json"]
    exposure = docs["GT_EXPOSURE.json"]
    design = docs["DESIGN_LOCK.json"]
    roster_doc = docs["ROSTER_LOCK.json"]
    refs_doc = docs["REFERENCE_LOCK.json"]

    need(teacher.get("status") == "complete" and teacher.get("GT_read") is False
         and int(teacher.get("count", -1)) == 49,
         "teacher barrier is not the complete 49-attempt GT-free seal")
    need(student.get("status") == "complete" and student.get("GT_read") is False
         and int(student.get("count", -1)) == 32,
         "student barrier is not the complete 32-source GT-free seal")
    need(pre_root.get("status") == "pass" and pre_root.get("GT_read") is False
         and int(pre_root.get("count", pre_root.get("case_count", -1))) == 32,
         "independent pre-GT readback has not passed for all 32 sources")
    need(prediction.get("status") == "complete" and prediction.get("GT_read") is False
         and int(prediction.get("count", -1)) == 32,
         "prediction seal is not complete and GT-free")
    need(teacher_readback.get("status") == "pass" and teacher_readback.get("GT_read") is False,
         "independent 49-attempt teacher readback has not passed")
    need(post_root.get("status") == "complete" and post_root.get("GT_read") is True
         and int(post_root.get("case_count", -1)) == 32
         and int(post_root.get("row_count", -1)) == 256,
         "post-seal root scoring readback is incomplete")
    need(scalar_readback.get("status") == "pass" and scalar_readback.get("GT_read") is False
         and scalar_readback.get("CPU_only") is True
         and scalar_readback.get("model_loaded") is False
         and scalar_readback.get("GPU_used") is False
         and int(scalar_readback.get("case_count", -1)) == 32
         and int(scalar_readback.get("row_count", -1)) == 256
         and int(scalar_readback.get("summary_cell_count", -1)) == 80
         and int(scalar_readback.get("paired_contrast_cell_count", -1)) == 60,
         "independent CPU scalar reaggregation is incomplete or changed scope")
    need(execution_cost.get("status") == "pass" and execution_cost.get("GT_read") is False
         and execution_cost.get("CPU_only") is True and execution_cost.get("model_called") is False
         and int(execution_cost.get("retained_fit_count", -1)) == 38
         and int(execution_cost.get("total_fit_invocations", -1)) == 40
         and int(execution_cost.get("retained_optimizer_steps", -1)) == 164
         and int(execution_cost.get("total_optimizer_steps", -1)) == 174
         and int(execution_cost.get("total_full_readout_calls", -1)) == 40
         and int(execution_cost.get("source_capture_calls", -1)) == 34
         and int(execution_cost.get("teacher_attempt_count", -1)) == 49
         and int(execution_cost.get("teacher_forward_count", -1)) == 40,
         "independent execution-cost audit is incomplete or inconsistent")
    cost_counts = execution_cost.get("counts", {})
    cost_times = execution_cost.get("timings_seconds", {})
    cost_runs = execution_cost.get("worker_runs", [])
    need(int(cost_counts.get("student005_completed_prefix_fit_invocations", -1)) == 16
         and int(cost_counts.get("student005_completed_prefix_optimizer_steps", -1)) == 69
         and int(cost_counts.get("student_new_fit_executions_007", -1)) == 22
         and int(cost_counts.get("student_partial_fit_attempts_005_006", -1)) == 2
         and int(cost_counts.get("partial_optimizer_steps_005_006", -1)) == 10
         and int(cost_counts.get("student_total_fit_records_plus_partial_attempts", -1)) == 40
         and int(cost_counts.get("student_total_readout_records_plus_partial_attempts", -1)) == 40
         and int(cost_counts.get("source_capture_calls_total", -1)) == 34,
         "execution-cost nested fit/readout accounting mismatch")
    need(isinstance(cost_runs, list) and len(cost_runs) == 7,
         "execution-cost launch/exit receipt inventory must contain seven attempts")
    cost_run_by_id = {str(row.get("run_id")): row for row in cost_runs}
    expected_run_ids = {"teacher001", "teacher002", "teacher003", "student004",
                        "student005", "student006", "student007"}
    need(set(cost_run_by_id) == expected_run_ids,
         "execution-cost launch/exit run identities mismatch")
    for run_id, row in cost_run_by_id.items():
        expected_seconds = execution_cost.get("worker_launch_to_exit_seconds", {}).get(run_id)
        need(expected_seconds is not None
             and abs(float(row.get("launch_to_exit_seconds", -1)) - float(expected_seconds)) <= 1e-9,
             f"launch/exit timing index mismatch for {run_id}")
    need(abs(math.fsum(float(row["launch_to_exit_seconds"]) for row in cost_runs)
             - float(execution_cost.get("worker_launch_to_exit_total_seconds", -1))) <= 1e-6,
         "summed worker launch/exit times differ from cost receipt")
    failed_seconds = math.fsum(float(row["launch_to_exit_seconds"]) for row in cost_runs if int(row["exit_code"]) != 0)
    need(abs(failed_seconds - float(cost_times.get("failed_launch_to_exit_total", -1))) <= 1e-6
         and abs(math.fsum(float(row["launch_to_exit_seconds"]) for row in cost_runs
                           if int(row["exit_code"]) == 0)
                 - float(cost_times.get("successful_launch_to_exit_total", -1))) <= 1e-6
         and abs(math.fsum(float(row["launch_to_exit_seconds"]) for rid, row in cost_run_by_id.items()
                           if rid.startswith("teacher"))
                 - float(cost_times.get("teacher_all_launch_to_exit_including_failed_attempts", -1))) <= 1e-6
         and abs(math.fsum(float(row["launch_to_exit_seconds"]) for rid, row in cost_run_by_id.items()
                           if rid.startswith("student"))
                 - float(cost_times.get("student_all_launch_to_exit_including_failed_attempts", -1))) <= 1e-6,
         "per-stage launch/exit totals differ from execution-cost receipt")
    teacher_failed_seconds = math.fsum(float(row["launch_to_exit_seconds"])
                                       for rid, row in cost_run_by_id.items()
                                       if rid.startswith("teacher") and int(row["exit_code"]) != 0)
    student_failed_seconds = math.fsum(float(row["launch_to_exit_seconds"])
                                       for rid, row in cost_run_by_id.items()
                                       if rid.startswith("student") and int(row["exit_code"]) != 0)
    failed_seconds = math.fsum(float(row["launch_to_exit_seconds"])
                               for row in cost_runs if int(row["exit_code"]) != 0)
    need(int(cost_counts.get("student_partial_fit_attempts_005_006", -1)) == 2
         and int(cost_counts.get("student_partial_readout_attempts_005_006", -1)) == 2
         and int(cost_counts.get("student_reused_fit_records", -1)) == 16,
         "partial/reused student work accounting mismatch")
    need(optimizer_readback.get("status") == "pass"
         and optimizer_readback.get("GT_read") is False
         and optimizer_readback.get("CPU_only") is True
         and optimizer_readback.get("GPU_used") is False
         and optimizer_readback.get("model_loaded") is False
         and optimizer_readback.get("backward_replayed") is False
         and int(optimizer_readback.get("finalized_case_count", -1)) == 32
         and int(optimizer_readback.get("finalized_fit_count", -1)) == 38
         and int(optimizer_readback.get("finalized_full_model_readout_count", -1)) == 38
         and int(optimizer_readback.get("finalized_optimizer_steps", -1)) == 164
         and int(optimizer_readback.get("preserved_partial_fit_attempt_count", -1)) == 2
         and optimizer_readback.get("partial_fit_full_model_readout_artifact_verified_by_this_reader") is False,
         "independent optimizer-math readback is incomplete or changed scope")
    need(optimizer_readback.get("student_barrier_sha256") == sha(BASE / "STUDENT_GLOBAL_BARRIER.json")
         and optimizer_readback.get("teacher_barrier_sha256") == sha(BASE / "TEACHER_GLOBAL_BARRIER.json")
         and optimizer_readback.get("reference_lock_sha256") == sha(BASE / "REFERENCE_LOCK.json"),
         "independent optimizer-math readback does not bind the sealed E1 inputs")
    need(int(optimizer_readback.get("optimizer_math_counts", {}).get("reward_scalars", -1)) == 12032
         and int(optimizer_readback.get("optimizer_math_counts", {}).get("loss_scalars", -1)) == 562
         and int(optimizer_readback.get("optimizer_math_counts", {}).get("accepted", -1)) == 164
         and int(optimizer_readback.get("optimizer_math_counts", {}).get("rejected", -1)) == 2,
         "independent optimizer-math replay scalar counts mismatch")
    need(exposure.get("GT_read") is True
         and exposure.get("prediction_seal_sha256") == sha(BASE / "GLOBAL_PREDICTION_BARRIER.json"),
         "GT exposure is absent or not bound to the complete prediction seal")
    need(score_rows.get("status") == "complete" and score_rows.get("GT_read") is True
         and score_summary.get("status") == "complete" and score_summary.get("GT_read") is True
         and score_pairs.get("status") == "complete" and score_pairs.get("GT_read") is True,
         "post-seal score tables are not complete")
    need(quality.get("status") == "complete" and quality.get("GT_read") is True
         and quality.get("bootstrap_executed") is False,
         "same-frame reference-quality audit is not complete or used an unexpected bootstrap")
    need(design.get("status") == "locked" and design.get("GT_read") is False
         and int(design.get("source_count", -1)) == 32
         and int(design.get("teacher_observation_attempts", -1)) == 49,
         "E1 design lock mismatch")
    roster = roster_doc.get("records")
    refs = refs_doc.get("records")
    need(isinstance(roster, list) and len(roster) == 32 and isinstance(refs, list) and len(refs) == 32,
         "E1 roster/reference lock must have 32 records")
    roster_by_id = {alias(str(r.get("case_id", ""))): r for r in roster}
    ref_by_id = {alias(str(r.get("case_id", ""))): r for r in refs}
    need(len(roster_by_id) == len(ref_by_id) == 32 and set(roster_by_id) == set(ref_by_id),
         "E1 roster/reference membership mismatch")
    need(sum(r.get("dataset") == "hc2" for r in roster) == 16
         and sum(r.get("dataset") == "vidstg" for r in roster) == 16,
         "E1 roster domain counts mismatch")

    decisions = [row for record in refs for row in record.get("dino_decisions", [])]
    decision_counts = {state: sum(row.get("status") == state for row in decisions)
                       for state in ("accepted", "rejected", "unknown")}
    dino_anchors = sum(len(row.get("common_dino_anchors", [])) for row in refs)
    wf_anchors = sum(len(row.get("common_wf_anchors", [])) for row in refs)
    supported = sum(bool(row.get("common_dino_anchors")) for row in refs)
    empty = len(refs) - supported
    need(len(decisions) == 49 and decision_counts == {"accepted": 38, "rejected": 2, "unknown": 9},
         f"locked DINO decision counts differ: {decision_counts}")
    need(dino_anchors == wf_anchors == 38 and supported == 19 and empty == 13,
         f"same-frame common support counts mismatch: anchors={dino_anchors}/{wf_anchors}, supported={supported}, empty={empty}")
    tr_counts = teacher_readback.get("counts", {})
    need(int(tr_counts.get("actual_model_forwards", -1)) == 40
         and int(tr_counts.get("unknown_without_model_forward", -1)) == 9,
         "teacher readback must distinguish 40 actual model forwards from 49 attempts")

    bindings = post_root.get("bindings", {})
    for key, name in (("prediction_seal_sha256", "GLOBAL_PREDICTION_BARRIER.json"),
                      ("teacher_barrier_sha256", "TEACHER_GLOBAL_BARRIER.json"),
                      ("student_barrier_sha256", "STUDENT_GLOBAL_BARRIER.json"),
                      ("root_readback_sha256", "ROOT_READBACK.json")):
        need(bindings.get(key) == sha(BASE / name), f"post-seal readback binding mismatch: {key}")
    quality_bindings = quality.get("bindings", {})
    need(quality_bindings.get("prediction_seal_sha256") == sha(BASE / "GLOBAL_PREDICTION_BARRIER.json")
         and quality_bindings.get("reference_lock_sha256") == sha(BASE / "REFERENCE_LOCK.json"),
         "reference-quality rows do not bind E1 prediction/reference seals")

    rows_in = score_rows.get("rows")
    need(isinstance(rows_in, list) and len(rows_in) == 256, "score rows must contain exactly 32x8 entries")
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    anon_rows = []
    for row in rows_in:
        cid = alias(str(row.get("case_id", "")))
        arm, dataset = row.get("arm"), row.get("dataset")
        need(cid in roster_by_id and arm in ARMS and dataset == roster_by_id[cid]["dataset"],
             f"unexpected source/arm/domain score row: {cid}/{arm}/{dataset}")
        need((cid, arm) not in rows_by_key, f"duplicate score row {cid}/{arm}")
        metrics = row.get("metrics")
        need(isinstance(metrics, dict) and set(metrics) == set(METRICS), f"metric schema mismatch {cid}/{arm}")
        clean_metrics = {}
        for metric in METRICS:
            value = metrics[metric]
            if value is not None:
                need(isinstance(value, (int, float)) and not isinstance(value, bool)
                     and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0,
                     f"invalid normalized metric {cid}/{arm}/{metric}")
                value = float(value)
            clean_metrics[metric] = value
        fit = row.get("fit", {})
        need(isinstance(fit, dict), f"fit metadata malformed {cid}/{arm}")
        fit_summary = {key: fit[key] for key in (
            "status", "reference_role", "optimizer_steps", "full_model_readout_calls", "profile"
        ) if key in fit}
        for key in ("optimizer_steps", "full_model_readout_calls"):
            if key in fit_summary:
                need(isinstance(fit_summary[key], int) and fit_summary[key] >= 0,
                     f"invalid fit count {cid}/{arm}/{key}")
        rows_by_key[cid, arm] = row
        anon_rows.append({"case_alias": cid, "domain": "HC-STVG-v2" if dataset == "hc2" else "VidSTG",
                          "arm": arm, "metrics_0_to_1": clean_metrics,
                          "fit": fit_summary})
    expected_keys = {(cid, arm) for cid in roster_by_id for arm in ARMS}
    need(set(rows_by_key) == expected_keys, "score rows do not cover every fixed source and arm")
    for cid, ref in ref_by_id.items():
        has_support = bool(ref.get("common_wf_anchors"))
        if not has_support:
            for arm in ("S_WF_common", "TS_WF_common"):
                need(rows_by_key[cid, arm]["fit"].get("status") == "exact_no_update_empty_common_support",
                     f"empty support not recorded as exact no-update: {cid}/{arm}")
            for arm, base_arm in (("S_WF_common", "F"), ("TS_WF_common", "T"),
                                  ("S_DINO_common", "F"), ("TS_DINO_common", "T")):
                for metric in METRICS:
                    need(rows_by_key[cid, arm]["metrics"][metric] == rows_by_key[cid, base_arm]["metrics"][metric],
                         f"empty-support exact fallback differs: {cid}/{arm}/{metric}")

    # Independently reconstruct the scorer's summary and paired-delta tables
    # from the sealed per-source scalar rows; no labels or model payloads used.
    computed_summary = []
    for dataset in ("hc2", "vidstg"):
        cids = sorted(cid for cid, record in roster_by_id.items() if record["dataset"] == dataset)
        for arm in ARMS:
            for metric in METRICS:
                values = [float(rows_by_key[cid, arm]["metrics"][metric]) for cid in cids
                          if rows_by_key[cid, arm]["metrics"][metric] is not None]
                unknown = [cid for cid in cids if rows_by_key[cid, arm]["metrics"][metric] is None]
                computed_summary.append({"dataset": dataset, "arm": arm, "metric": metric,
                                         "n_valid": len(values), "n_unknown": len(unknown),
                                         "unknown_case_aliases": unknown,
                                         "mean": safe_mean(values),
                                         "median": sorted(values)[(len(values)-1)//2] if values and len(values)%2 else
                                                  ((sorted(values)[len(values)//2-1] + sorted(values)[len(values)//2]) / 2.0 if values else None),
                                         "min": min(values) if values else None,
                                         "max": max(values) if values else None})
    source_summary = score_summary.get("summary")
    need(isinstance(source_summary, list) and len(source_summary) == 80,
         "scorer summary must contain 80 domain/arm/metric cells")
    s_idx = {(x.get("dataset"), x.get("arm"), x.get("metric")): x for x in source_summary}
    need(len(s_idx) == 80, "duplicate scorer summary cells")
    for cell in computed_summary:
        src = s_idx.get((cell["dataset"], cell["arm"], cell["metric"]))
        need(src is not None and int(src.get("count", src.get("n_valid", -1))) == cell["n_valid"],
             "scorer summary count differs from independently reaggregated rows")
        if cell["mean"] is None:
            need(src.get("mean") is None, "unknown-only summary mean differs")
        else:
            need(abs(float(src["mean"]) - cell["mean"]) <= 1e-12,
                 "scorer summary mean differs from independently reaggregated rows")

    computed_pairs = []
    for dataset in ("hc2", "vidstg"):
        cids = sorted(cid for cid, record in roster_by_id.items() if record["dataset"] == dataset)
        for contrast, left, right in CONTRASTS:
            for metric in METRICS:
                per_source: dict[str, float] = {}
                unknown = []
                for cid in cids:
                    lv = rows_by_key[cid, left]["metrics"][metric]
                    rv = rows_by_key[cid, right]["metrics"][metric]
                    if lv is None or rv is None:
                        unknown.append(cid)
                    else:
                        per_source[cid] = 100.0 * (float(lv) - float(rv))
                vals = list(per_source.values())
                computed_pairs.append({
                    "dataset": dataset, "contrast": contrast, "comparison_arm": left,
                    "baseline_arm": right, "metric": metric, "n_pairs": len(vals),
                    "unknown_count": len(unknown), "unknown_case_aliases": unknown,
                    "positive_count": sum(v > SIGN_EPSILON for v in vals),
                    "negative_count": sum(v < -SIGN_EPSILON for v in vals),
                    "neutral_count": sum(abs(v) <= SIGN_EPSILON for v in vals),
                    "neutral_tolerance_pp": SIGN_EPSILON,
                    "mean_delta_pp": safe_mean(vals),
                    "median_delta_pp": sorted(vals)[(len(vals)-1)//2] if vals and len(vals)%2 else
                                       ((sorted(vals)[len(vals)//2-1] + sorted(vals)[len(vals)//2]) / 2.0 if vals else None),
                    "min_delta_pp": min(vals) if vals else None,
                    "max_delta_pp": max(vals) if vals else None,
                    "per_source_delta_pp": per_source,
                })
    source_pairs = score_pairs.get("records")
    need(isinstance(source_pairs, list) and len(source_pairs) == 60,
         "scorer paired table must contain 60 contrasts")
    p_idx = {(x.get("dataset"), x.get("contrast"), x.get("metric")): x for x in source_pairs}
    need(len(p_idx) == 60, "duplicate paired-delta cells")
    for cell in computed_pairs:
        src = p_idx.get((cell["dataset"], cell["contrast"], cell["metric"]))
        need(src is not None, "paired table is missing a contrast cell")
        for key, alt in (("n_pairs", "n_pairs"), ("unknown_count", "unknown_count"),
                         ("positive_count", "positive_count"), ("negative_count", "negative_count"),
                         ("neutral_count", "neutral_count")):
            need(int(src.get(alt, -1)) == cell[key], f"paired table count mismatch for {cell['contrast']}/{cell['metric']}")
        need(src.get("bootstrap_executed", False) is False, "unexpected bootstrap in E1 paired output")
        if cell["mean_delta_pp"] is None:
            need(src.get("mean_delta_pp") is None, "paired unknown mean mismatch")
        else:
            need(abs(float(src.get("mean_delta_pp")) - cell["mean_delta_pp"]) <= 1e-10,
                 "paired mean differs from independent per-source reaggregation")
        src_deltas = src.get("per_source_delta_pp", {})
        for cid, delta in cell["per_source_delta_pp"].items():
            need(cid in src_deltas and abs(float(src_deltas[cid]) - delta) <= 1e-10,
                 "paired per-source delta differs from score rows")

    quality_rows = quality.get("rows")
    quality_summary = quality.get("summary")
    need(isinstance(quality_rows, list) and len(quality_rows) == 38,
         "reference-quality table must cover the 38 common observed-frame anchors")
    need(isinstance(quality_summary, list) and len(quality_summary) == 2,
         "reference-quality summary must contain both domains")
    q_clean = []
    for row in quality_rows:
        cid = alias(str(row.get("case_id", "")))
        need(cid in ref_by_id and row.get("dataset") == roster_by_id[cid]["dataset"],
             "reference-quality case/domain mismatch")
        if row.get("gt_observed"):
            for k in ("source_iou", "wf_iou", "dino_iou", "dino_minus_wf_pp"):
                need(isinstance(row.get(k), (int, float)) and math.isfinite(float(row[k])),
                     f"observed reference-quality scalar missing: {cid}/{k}")
        else:
            need(all(row.get(k) is None for k in ("source_iou", "wf_iou", "dino_iou", "dino_minus_wf_pp")),
                 f"unknown anchor-quality slot was imputed: {cid}")
        q_clean.append({"case_alias": cid,
                        "domain": "HC-STVG-v2" if row["dataset"] == "hc2" else "VidSTG",
                        "gt_observed": bool(row["gt_observed"]),
                        "source_anchor_iou": row.get("source_iou"),
                        "wf_anchor_iou": row.get("wf_iou"),
                        "dino_anchor_iou": row.get("dino_iou"),
                        "dino_minus_wf_anchor_iou_pp": row.get("dino_minus_wf_pp")})
    q_summary = []
    for row in quality_summary:
        need(row.get("dataset") in ("hc2", "vidstg"), "unexpected reference-quality domain")
        q_summary.append({k: row.get(k) for k in (
            "dataset", "count", "observed_count", "unknown_count", "source_iou", "wf_iou",
            "dino_iou", "mean_dino_minus_wf_pp", "positive", "negative", "neutral")})

    paired_lookup = {(row["dataset"], row["contrast"], row["metric"]): row
                     for row in computed_pairs}
    anchor_gap_vs_tube = []
    for dataset in ("hc2", "vidstg"):
        q = next(row for row in q_summary if row["dataset"] == dataset)
        ts = paired_lookup[dataset, "TS_DINO_common_minus_TS_WF_common", "m_vIoU"]
        anchor_gap_vs_tube.append({
            "dataset": dataset,
            "observed_anchor_mean_delta_pp": q["mean_dino_minus_wf_pp"],
            "TS_full_tube_vIoU_mean_delta_pp": ts["mean_delta_pp"],
            "TS_vIoU_positive_count": ts["positive_count"],
            "TS_vIoU_negative_count": ts["negative_count"],
            "TS_vIoU_neutral_count": ts["neutral_count"],
            "TS_vIoU_unknown_count": ts["unknown_count"],
        })

    # Only expose audit-level counts/limits and aggregate stage times; never copy
    # response prose, prompts, filenames, paths, frame coordinates or model logits.
    limitation = teacher_readback.get("output_field_limitations", {})
    need(limitation.get("selected_model_candidate_index_saved_explicitly") is False
         and limitation.get("target_scores_saved_for_all_raw_candidates") is False,
         "teacher output limitation fields changed or were omitted")
    teacher_public = {
        "status": "complete", "attempt_count": 49, "actual_model_forward_count": 40,
        "accepted_count": 38, "rejected_count": 2, "unknown_count": 9,
        "unknown_without_model_forward_count": 9,
        "common_anchor_count": 38, "sources_with_common_anchors": 19,
        "sources_with_empty_common_support": 13,
        "nms_candidate_stages_replayed": int(tr_counts.get("nms_candidate_stages_replayed", -1)),
        "context_pool_gate_decisions_replayed": int(tr_counts.get("context_pool_gate_decisions_replayed", -1)),
        "accepted_candidate_indices_derived": int(tr_counts.get("accepted_candidate_indices_derived_from_pool", -1)),
        "logit_and_score_storage_scope": (
            "The selected index was reconstructed from saved candidate_ids[admission_winner]. "
            "Saved processed_probe.target_scores support replay of the admitted pool score, threshold, "
            "margin, and winner. Raw logits and per-target scores for all 900 detector candidates "
            "were not serialized; the upstream all-candidate score calculation is therefore not independently replayable."),
        "GT_read": False,
    }
    teacher_elapsed = float(teacher.get("elapsed_seconds", 0.0))
    student_elapsed = float(student.get("wall_seconds", 0.0))
    need(math.isfinite(teacher_elapsed) and teacher_elapsed > 0
         and math.isfinite(student_elapsed) and student_elapsed > 0,
         "missing measured teacher or student wall time")
    need(abs(teacher_elapsed - float(cost_times.get("teacher_barrier_worker_elapsed", -1))) <= 1e-6
         and abs(student_elapsed - float(cost_times.get("student_barrier_worker_elapsed", -1))) <= 1e-6,
         "sealed stage times disagree with independently aggregated execution-cost receipt")
    fit_math_counts = optimizer_readback.get("optimizer_math_counts", {})
    math_errors = optimizer_readback.get("max_abs_errors", {})
    optimizer_math_public = {
        "schema": "stvg-reference-swap-independent-optimizer-math-anonymous-v1",
        "status": optimizer_readback["status"],
        "case_count": int(optimizer_readback["finalized_case_count"]),
        "finalized_fit_count": int(optimizer_readback["finalized_fit_count"]),
        "finalized_full_model_readout_count": int(optimizer_readback["finalized_full_model_readout_count"]),
        "finalized_optimizer_steps": int(optimizer_readback["finalized_optimizer_steps"]),
        "common_anchor_count": int(optimizer_readback["common_anchor_count"]),
        "common_support_case_count": int(optimizer_readback["common_support_case_count"]),
        "empty_common_support_case_count": int(optimizer_readback["empty_common_support_case_count"]),
        "exact_reused_case_count": int(optimizer_readback["exact_reused_case_count"]),
        "exact_reused_fit_count": int(optimizer_readback["exact_reused_fit_count"]),
        "unique_saved_fit_artifact_count_including_partials": int(
            optimizer_readback["unique_saved_fit_artifact_count_including_partials"]),
        "preserved_partial_fit_attempt_count": int(optimizer_readback["preserved_partial_fit_attempt_count"]),
        "partial_fit_full_model_readout_artifact_verified_by_this_reader": False,
        "optimizer_math_counts": {key: int(value) for key, value in fit_math_counts.items()},
        "max_abs_errors": {key: float(value) for key, value in math_errors.items()},
        "backward_replayed": False,
        "GT_read": False, "CPU_only": True, "model_loaded": False, "GPU_used": False,
        "bindings": {"optimizer_math_readback_sha256": sha(BASE / "ROOT_INDEPENDENT_OPTIMIZER_MATH_READBACK.json"),
                     "student_barrier_sha256": sha(BASE / "STUDENT_GLOBAL_BARRIER.json"),
                     "teacher_barrier_sha256": sha(BASE / "TEACHER_GLOBAL_BARRIER.json"),
                     "reference_lock_sha256": sha(BASE / "REFERENCE_LOCK.json")},
    }
    worker_seconds = execution_cost["worker_launch_to_exit_seconds"]
    teacher_failed_count = sum(int(row["exit_code"]) != 0 for rid, row in cost_run_by_id.items()
                               if rid.startswith("teacher"))
    student_failed_count = sum(int(row["exit_code"]) != 0 for rid, row in cost_run_by_id.items()
                               if rid.startswith("student"))
    costs = {
        "schema": "stvg-reference-swap-anonymous-execution-cost-v1",
        "status": "pass", "teacher_sealed_worker_elapsed_seconds": teacher_elapsed,
        "teacher_all_launch_to_exit_seconds": float(cost_times["teacher_all_launch_to_exit_including_failed_attempts"]),
        "teacher_failed_launch_count": teacher_failed_count,
        "teacher_failed_launch_to_exit_seconds": teacher_failed_seconds,
        "teacher_attempts": int(execution_cost["teacher_attempt_count"]),
        "teacher_actual_model_forwards": int(execution_cost["teacher_forward_count"]),
        "teacher_abstentions_without_forward": 9,
        "student_sealed_worker_elapsed_seconds": student_elapsed,
        "student_all_launch_to_exit_seconds": float(cost_times["student_all_launch_to_exit_including_failed_attempts"]),
        "student_failed_launch_count": student_failed_count,
        "student_failed_launch_to_exit_seconds": student_failed_seconds,
        "student_case_count": int(student.get("count", -1)),
        "retained_final_fit_count": int(execution_cost["retained_fit_count"]),
        "student005_completed_prefix_fit_invocations": int(cost_counts["student005_completed_prefix_fit_invocations"]),
        "student005_completed_prefix_optimizer_steps": int(cost_counts["student005_completed_prefix_optimizer_steps"]),
        "student007_new_fit_invocations": int(cost_counts["student_new_fit_executions_007"]),
        "partial_hc11_fit_attempts": int(cost_counts["student_partial_fit_attempts_005_006"]),
        "actual_fit_invocations_including_partial_hc11_attempts": int(execution_cost["total_fit_invocations"]),
        "final_barrier_full_model_readout_count": int(optimizer_readback["finalized_full_model_readout_count"]),
        "actual_full_model_readout_invocations_including_partial_attempts": int(execution_cost["total_full_readout_calls"]),
        "retained_optimizer_steps": int(execution_cost["retained_optimizer_steps"]),
        "total_optimizer_steps": int(execution_cost["total_optimizer_steps"]),
        "student007_new_optimizer_steps": int(cost_counts["new_optimizer_steps_007"]),
        "partial_hc11_optimizer_steps": int(cost_counts["partial_optimizer_steps_005_006"]),
        "student005_prefix_steps_reused_in_final_barrier": int(cost_counts["reused_optimizer_steps_already_in_retained_records"]),
        "full_model_readout_calls": int(student.get("full_model_readout_calls", -1)),
        "total_full_readout_calls": int(execution_cost["total_full_readout_calls"]),
        "source_capture_calls": int(execution_cost["source_capture_calls"]),
        "source_capture_seconds_serialized_for_final_case_payloads": int(
            cost_counts["source_capture_seconds_serialized_for_final_case_payloads"]),
        "failed_hc11_source_capture_seconds_unavailable": int(
            cost_counts["failed_hc11_source_capture_seconds_unavailable"]),
        "worker_launch_to_exit_seconds": {key: float(value) for key, value in worker_seconds.items()},
        "all_worker_launch_to_exit_seconds": float(execution_cost["worker_launch_to_exit_total_seconds"]),
        "all_failed_launch_count": int(sum(int(row["exit_code"]) != 0 for row in cost_runs)),
        "all_failed_launch_to_exit_seconds": failed_seconds,
        "all_launch_to_exit_including_failed_attempts_seconds": float(
            cost_times["all_launch_to_exit_including_failed_attempts"]),
        "successful_launch_to_exit_total_seconds": float(cost_times["successful_launch_to_exit_total"]),
        "partial_full_model_readout_artifact_independently_verified": False,
        "source_restored_case_count": 32,
        "historical_frozen_temporal_and_parent_student_reuse": True,
        "whole_method_cold_latency_remeasured": False,
        "bootstrap_executed": False,
        "GT_read": False,
        "limitation": ("Stage barrier elapsed time is shown separately from launch-to-exit totals. "
                       "Launch-to-exit includes failed attempts, model loading and other worker work; those "
                       "substage timings were not separately serialized. The two HC11 partial fit chains are "
                       "independently math-read, but no separate serialized partial full-model readout artifact "
                       "was verified by that reader. Historical source/temporal/parent arms are exact reuses, "
                       "so these stage timings are not fresh end-to-end method latency."),
        "bindings": {"actual_execution_cost_sha256": sha(BASE / "ROOT_ACTUAL_EXECUTION_COST.json"),
                     "optimizer_math_readback_sha256": sha(BASE / "ROOT_INDEPENDENT_OPTIMIZER_MATH_READBACK.json")},
    }
    need(costs["retained_final_fit_count"] == 38
         and costs["actual_fit_invocations_including_partial_hc11_attempts"] == 40
         and costs["retained_optimizer_steps"] == 164
         and costs["total_optimizer_steps"] == 174
         and costs["total_full_readout_calls"] == 40
         and costs["source_capture_calls"] == 34
         and costs["full_model_readout_calls"] >= 0,
         "student barrier and cost receipt fit/readout counts disagree")

    summary_public = [{k: cell[k] for k in (
        "dataset", "arm", "metric", "n_valid", "n_unknown", "unknown_case_aliases",
        "mean", "median", "min", "max")}
        for cell in computed_summary]
    paired_public = [{k: cell[k] for k in (
        "dataset", "contrast", "comparison_arm", "baseline_arm", "metric", "n_pairs",
        "unknown_count", "unknown_case_aliases", "positive_count", "negative_count",
        "neutral_count", "neutral_tolerance_pp", "mean_delta_pp", "median_delta_pp",
        "min_delta_pp", "max_delta_pp", "per_source_delta_pp")}
        for cell in computed_pairs]

    out_bindings = {
        "design_lock_sha256": sha(BASE / "DESIGN_LOCK.json"),
        "teacher_root_readback_sha256": sha(BASE / "ROOT_TEACHER_READBACK.json"),
        "postseal_scalar_readback_sha256": sha(BASE / "ROOT_POSTSEAL_SCALAR_READBACK.json"),
        "actual_execution_cost_sha256": sha(BASE / "ROOT_ACTUAL_EXECUTION_COST.json"),
        "independent_optimizer_math_readback_sha256": sha(BASE / "ROOT_INDEPENDENT_OPTIMIZER_MATH_READBACK.json"),
        "prediction_seal_sha256": sha(BASE / "GLOBAL_PREDICTION_BARRIER.json"),
        "postseal_rows_sha256": sha(BASE / "POSTSEAL_ROWS.json"),
        "postseal_summary_sha256": sha(BASE / "POSTSEAL_SUMMARY.json"),
        "paired_deltas_sha256": sha(BASE / "PARENT_PAIRED_DELTAS.json"),
        "reference_quality_sha256": sha(BASE / "POSTSEAL_REFERENCE_QUALITY.json"),
        "postseal_root_readback_sha256": sha(BASE / "ROOT_POSTSEAL_READBACK.json"),
    }
    return {
        "rows": {"schema": "stvg-reference-swap-anonymous-rows-v1", "status": "complete",
                 "rows": anon_rows, "row_count": len(anon_rows), "GT_read": True,
                 "bindings": out_bindings},
        "summary": {"schema": "stvg-reference-swap-anonymous-summary-v1", "status": "complete",
                    "summary": summary_public, "summary_cell_count": len(summary_public),
                    "GT_read": True, "bindings": out_bindings},
        "paired": {"schema": "stvg-reference-swap-anonymous-paired-v1", "status": "complete",
                   "records": paired_public, "contrast_cell_count": len(paired_public),
                   "bootstrap_executed": False, "GT_read": True, "bindings": out_bindings},
        "quality": {"schema": "stvg-reference-swap-anchor-quality-anonymous-v1", "status": "complete",
                    "scope": "descriptive IoU only at same observed anchor positions; not full-tube vIoU",
                    "rows": q_clean, "summary": q_summary, "row_count": len(q_clean),
                    "bootstrap_executed": False, "GT_read": True, "bindings": out_bindings},
        "anchor_vs_tube": anchor_gap_vs_tube,
        "teacher": teacher_public,
        "costs": costs,
        "scalar_readback": {
            "schema": "stvg-reference-swap-independent-scalar-readback-anonymous-v1",
            "status": scalar_readback["status"],
            "case_count": scalar_readback["case_count"],
            "row_count": scalar_readback["row_count"],
            "summary_cell_count": scalar_readback["summary_cell_count"],
            "paired_contrast_cell_count": scalar_readback["paired_contrast_cell_count"],
            "null_metric_cells_preserved": scalar_readback["null_metric_cells_preserved"],
            "paired_tail_counts_across_all_cells": scalar_readback["paired_tail_counts_across_all_cells"],
            "neutral_tolerance_pp": scalar_readback["neutral_tolerance_pp"],
            "GT_read": False, "CPU_only": True, "model_loaded": False, "GPU_used": False,
            "bindings": out_bindings,
        },
        "optimizer_math_readback": optimizer_math_public,
        "root_readback": {
            "schema": "stvg-reference-swap-anonymous-readback-v1", "status": "pass",
            "case_count": 32, "score_row_count": 256, "summary_cell_count": 80,
            "paired_cell_count": 60, "teacher_attempt_count": 49,
            "teacher_actual_forward_count": 40, "common_anchor_count": 38,
            "official_independent_metric_max_error": post_root.get("official_independent_metric_max_error"),
            "independent_scalar_reaggregation": {
                "status": scalar_readback["status"],
                "null_metric_cells_preserved": scalar_readback["null_metric_cells_preserved"],
                "paired_tail_counts_across_all_cells": scalar_readback["paired_tail_counts_across_all_cells"],
                "neutral_tolerance_pp": scalar_readback["neutral_tolerance_pp"],
            },
            "independent_optimizer_math": {
                "status": optimizer_readback["status"],
                "finalized_fit_count": optimizer_readback["finalized_fit_count"],
                "finalized_optimizer_steps": optimizer_readback["finalized_optimizer_steps"],
                "preserved_partial_fit_attempt_count": optimizer_readback["preserved_partial_fit_attempt_count"],
                "partial_fit_full_model_readout_artifact_verified_by_this_reader": False,
            },
            "GT_read": True, "model_loaded": False, "GPU_used": False,
            "bindings": out_bindings,
        },
        "source_bindings": {"schema": "stvg-reference-swap-anonymous-source-bindings-v1",
                            "status": "pass", "bindings": out_bindings,
                            "private_source_paths_or_source_hashes_exported": False,
                            "GT_read": False},
        "internal": {"summary": computed_summary, "paired": computed_pairs,
                     "quality_summary": q_summary, "refs": ref_by_id,
                     "rows_by_key": rows_by_key, "student_barrier": student,
                     "teacher_barrier": teacher, "teacher_readback": teacher_readback,
                     "post_root": post_root, "bindings": out_bindings},
    }


def make_report(data: Mapping[str, Any]) -> str:
    summary = data["summary"]["summary"]
    paired = data["paired"]["records"]
    qsum = data["quality"]["summary"]
    teacher = data["teacher"]
    costs = data["costs"]
    by_summary = {(x["dataset"], x["arm"], x["metric"]): x for x in summary}
    ds_names = {"hc2": "HC-STVG-v2", "vidstg": "VidSTG"}
    metric_names = {
        "m_vIoU": "vIoU", "m_tIoU": "tIoU", "m_sIoU_dense_GT": "dense sIoU",
        "m_sIoU_supervised_GT": "supervised sIoU", "m_sIoU_unsupervised_GT": "unsupervised sIoU",
    }
    lines = [
        "# E1: same-frame original-DINO versus Weight-free reference geometry", "",
        "This is a bounded comparison on 32 previously exposed development sources (16 per domain), not an untouched confirmation or the full Table 1 cohort. It tests a same-frame geometry swap under the locked current five-step Student adaptation. It does not reproduce the full original DINO/online pipeline, and it does not select or promote a production method.", "",
        "## Teacher support and audit boundary", "",
        f"The frozen DINO expert received {teacher['attempt_count']} prelocked observation attempts. {teacher['actual_model_forward_count']} had actual model forwards: {teacher['accepted_count']} passed the original admission gate and form the common support; {teacher['rejected_count']} were rejected; {teacher['unknown_count']} were parser abstentions without a forward. The accepted references provide {teacher['common_anchor_count']} same-frame anchors over {teacher['sources_with_common_anchors']} sources; {teacher['sources_with_empty_common_support']} sources have empty common support and use exact no-update F/T fallback.", "",
        "The DINO and Weight-free references are paired on identical physical observation positions and use the original Weight-free reliability weights. Rejected and unknown DINO positions are not repaired or interpolated. The selected candidate index was reconstructed from the saved admitted-pool candidate IDs and winner index. The serialized admitted-pool target scores allow replay of the saved gate, margin and selected score; raw logits and per-target scores over all 900 detector candidates were not stored, so the all-candidate score derivation cannot be independently replayed without rerunning the model.", "",
        "## Eight-arm full-tube results", "",
        "Values are source means in percent. `n` is the number of known per-source scores; `unknown` rows remain null rather than being imputed. The five metrics are kept separate because vIoU is not the product of mean tIoU and mean sIoU.", "",
    ]
    for dataset in ("hc2", "vidstg"):
        lines.extend([f"### {ds_names[dataset]}", "",
                      "| Arm | vIoU | tIoU | dense sIoU | supervised sIoU | unsupervised sIoU | n (vIoU) | unknown (vIoU) |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for arm in ARMS:
            cells = [by_summary[dataset, arm, metric] for metric in METRICS]
            formatted = ["—" if cell["mean"] is None else f"{100*cell['mean']:.6f}" for cell in cells]
            lines.append(f"| {arm} | {formatted[0]} | {formatted[1]} | {formatted[2]} | {formatted[3]} | {formatted[4]} | {cells[0]['n_valid']} | {cells[0]['n_unknown']} |")
        lines.append("")
    lines += [
        "The four parent arms F, T, S_new_crop and TS_new_crop are exact saved reuse. The WF_common and DINO_common arms use the same current5 domain configuration, 1,792 active parameters, five declared steps, 32 samples, stable query key, Source reset and per-query Adam reset. If common support is empty, no new fit is run and the arm is exact F/T. All eight arms are represented in the source-level JSON; no support case was dropped.", "",
        "## Paired differences and source-level outcomes", "",
        "Differences are in percentage points (pp). Positive, negative and neutral counts use the locked scorer tolerance of 1e−10 pp; unknown means at least one compared metric is missing. No bootstrap was run. The full JSON preserves every per-source delta and null.", "",
        "| Domain | Contrast (left − baseline) | Metric | Mean Δ (pp) | Positive | Negative | Neutral | Unknown |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for cell in paired:
        mean_text = "—" if cell["mean_delta_pp"] is None else f"{cell['mean_delta_pp']:+.6f}"
        lines.append(f"| {ds_names[cell['dataset']]} | {cell['contrast']} | {metric_names[cell['metric']]} | {mean_text} | {cell['positive_count']} | {cell['negative_count']} | {cell['neutral_count']} | {cell['unknown_count']} |")
    lines += ["", "## Same-frame anchor quality (separate from full-tube vIoU)", "",
              "The following diagnostic compares Source, WF and DINO boxes only at the shared observation positions with an actual GT annotation. It is a box IoU readout on the 38 locked common anchors, not an oracle fit, not candidate selection, and not the full-tube vIoU endpoint.", "",
              "| Domain | Common anchors | GT observed | GT unknown | Source anchor IoU | WF anchor IoU | DINO anchor IoU | DINO − WF (pp) | + / − / neutral |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in qsum:
        dname = ds_names[row["dataset"]]
        gap = "—" if row["mean_dino_minus_wf_pp"] is None else f"{row['mean_dino_minus_wf_pp']:+.6f}"
        vals = ["—" if row[key] is None else f"{100*row[key]:.6f}%" for key in ("source_iou", "wf_iou", "dino_iou")]
        signs = f"{row['positive']} / {row['negative']} / {row['neutral']}"
        lines.append(f"| {dname} | {row['count']} | {row['observed_count']} | {row['unknown_count']} | {vals[0]} | {vals[1]} | {vals[2]} | {gap} | {signs} |")
    lines.append("")
    for row in data["anchor_vs_tube"]:
        lines.append(
            f"At the annotated shared anchors, the DINO-minus-WF mean box-IoU difference was "
            f"{row['observed_anchor_mean_delta_pp']:+.6f} pp for {ds_names[row['dataset']]}; "
            f"the corresponding TS full-tube vIoU change was {row['TS_full_tube_vIoU_mean_delta_pp']:+.6f} pp "
            f"({row['TS_vIoU_positive_count']} positive, {row['TS_vIoU_negative_count']} negative, "
            f"{row['TS_vIoU_neutral_count']} neutral, {row['TS_vIoU_unknown_count']} unknown). "
            "These are distinct endpoints on this exposed panel; the anchor result does not imply an equal-sized tube-level gain."
        )
    lines += ["", "## Execution cost and limits", "",
              "| Stage | Measured work | Wall time |", "|---|---|---:|",
              f"| DINO reference worker | {costs['teacher_attempts']} attempts; {costs['teacher_actual_model_forwards']} actual forwards; {costs['teacher_abstentions_without_forward']} abstentions without a forward | {costs['teacher_sealed_worker_elapsed_seconds']:.3f} s sealed; {costs['teacher_all_launch_to_exit_seconds']:.3f} s across all launches |",
              f"| Student reference swap | {costs['student_case_count']} cases; {costs['retained_final_fit_count']} fit records sealed; {costs['actual_fit_invocations_including_partial_hc11_attempts']} actual fit invocations and {costs['actual_full_model_readout_invocations_including_partial_attempts']} reported full readouts; {costs['retained_optimizer_steps']} retained / {costs['total_optimizer_steps']} execution-chain optimizer steps | {costs['student_sealed_worker_elapsed_seconds']:.3f} s sealed; {costs['student_all_launch_to_exit_seconds']:.3f} s across all launches |",
              f"| Failed engineering launches | {costs['all_failed_launch_count']} total ({costs['teacher_failed_launch_count']} teacher, {costs['student_failed_launch_count']} student) | {costs['all_failed_launch_to_exit_seconds']:.3f} s combined |",
              f"| Full execution chain | {costs['source_capture_calls']} source-capture calls; 16 prefix fits/69 steps in student005, 22 new fits/95 steps in student007, and 2 HC11 partial fits/10 steps | {costs['all_launch_to_exit_including_failed_attempts_seconds']:.3f} s across all launches |",
              "", "The 38 retained fit records comprise 16 student005 prefix fits reused exactly in the final seal and 22 new student007 fits. Two additional HC11 Weight-free attempts in student005 and student006 each completed five optimizer rounds before their pre-seal failure. The independent CPU math reader confirms the saved partial optimizer chains and 10 extra steps, but did not independently verify a separately serialized full-model readout artifact for those attempts; the two partial readouts are recorded by the execution-cost audit. The float32 edge case used the locked 1e−7 geometry validity tolerance without clipping; all 76 paired anchor tensors were preserved exactly.", "",
              costs["limitation"], "",
              "The report is a finite analysis of exposed development sources. No confidence interval, bootstrap, full-cohort extrapolation, hyperparameter search, CURRENT change, or full evaluation was performed. Anchor-IoU quality and whole-tube vIoU answer different questions; neither is used to revise the locked teacher references or selected student checkpoint.", "",
              "## Validation and artifacts", "",
              f"The scorer outputs were checked against all 256 source/arm rows, 80 domain/arm/metric summaries and 60 paired contrasts. The independent CPU scalar receipt preserves {data['root_readback']['independent_scalar_reaggregation']['null_metric_cells_preserved']} null metric cells and reaggregates the sign counts shown above. A separate CPU optimizer-math reader checked 12,032 reward scalars, 562 loss scalars, 190 Adam parameter-group transitions, 164 accepted steps and two rejected proposals; its largest reported absolute discrepancy was {max(data['optimizer_math_readback']['max_abs_errors'].values()):.3g}. It did not rerun inference or backward passes, and it did not independently verify a serialized full-model readout artifact for the two partial HC11 attempts. Neither CPU audit read GT, loaded a model or used a GPU. This report does not independently recompute GT metrics from annotations. Teacher gate and same-frame custody were independently audited before scoring.", "",
              "![Eight-arm full-tube vIoU](../results/stvg_reference_swap/2026-10-15/REFERENCE_SWAP_VIOU.png)", "",
              "![Same-frame DINO versus WF paired deltas](../results/stvg_reference_swap/2026-10-15/REFERENCE_SWAP_CONTRASTS.png)", "",
              "[Locked protocol](../protocols/stvg_reference_swap_v1.md) · [Anonymous outputs](../results/stvg_reference_swap/2026-10-15).", ""]
    return "\n".join(lines)


def make_figures(data: Mapping[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = data["summary"]["summary"]
    paired = data["paired"]["records"]
    lookup = {(row["dataset"], row["arm"], row["metric"]): row for row in summary}
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    colors = ["#727d86", "#7c9b9b", "#8e8c73", "#6e8795", "#86a5a1", "#7b93a6", "#b17d59", "#c18d67"]
    for ax, dataset, title in zip(axes, ("hc2", "vidstg"), ("HC-STVG-v2", "VidSTG")):
        vals = [100 * lookup[dataset, arm, "m_vIoU"]["mean"] for arm in ARMS]
        bars = ax.bar(range(len(ARMS)), vals, color=colors)
        ax.set_title(title + " · 16 exposed sources")
        ax.set_xticks(range(len(ARMS)), ARMS, rotation=35, ha="right")
        ax.set_ylabel("Mean full-tube vIoU (%)")
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:.2f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "REFERENCE_SWAP_VIOU.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, dataset, title in zip(axes, ("hc2", "vidstg"), ("HC-STVG-v2", "VidSTG")):
        selected = [x for x in paired if x["dataset"] == dataset and x["metric"] == "m_vIoU"
                    and x["contrast"] in ("S_DINO_common_minus_S_WF_common",
                                          "TS_DINO_common_minus_TS_WF_common")]
        for x_pos, contrast in enumerate(("S_DINO_common_minus_S_WF_common",
                                          "TS_DINO_common_minus_TS_WF_common")):
            cell = next(x for x in selected if x["contrast"] == contrast)
            deltas = cell["per_source_delta_pp"]
            xs = [x_pos + (i - 7.5) * .018 for i in range(16)]
            ys = [deltas.get(cid) for cid in sorted(deltas)]
            ax.scatter(xs, ys, s=20, color="#b17d59" if x_pos == 0 else "#6e8795", alpha=.85)
            if cell["mean_delta_pp"] is not None:
                ax.scatter([x_pos], [cell["mean_delta_pp"]], marker="_", s=260,
                           linewidths=2.5, color="black", zorder=5)
        ax.axhline(0, color="#777", lw=.9)
        ax.set_xticks([0, 1], ["S: DINO − WF", "TS: DINO − WF"])
        ax.set_title(title + " · common-support intervention")
        ax.set_ylabel("Per-source vIoU change (pp)")
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "REFERENCE_SWAP_CONTRASTS.png", dpi=180)
    plt.close(fig)


def run(*, replace: bool = False) -> None:
    data = verify_and_build()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, key in (
        ("E1_POSTSEAL_ROWS.json", "rows"), ("E1_POSTSEAL_SUMMARY.json", "summary"),
        ("E1_PAIRED_DELTAS.json", "paired"), ("E1_REFERENCE_QUALITY.json", "quality"),
        ("E1_TEACHER_READBACK.json", "teacher"), ("E1_EXECUTION_COST.json", "costs"),
        ("E1_OPTIMIZER_MATH_READBACK.json", "optimizer_math_readback"),
        ("E1_ROOT_READBACK.json", "root_readback"), ("E1_SCALAR_READBACK.json", "scalar_readback"),
        ("E1_SOURCE_BINDINGS.json", "source_bindings"),
    ):
        write_json(OUT / name, data[key], replace=replace)
    make_figures(data)
    report = make_report(data)
    if DOC.exists() and not replace:
        existing = DOC.read_text(encoding="utf-8")
        if "<!-- E1_REPORT_PENDING_TEMPLATE -->" not in existing:
            raise ReportError(f"refusing to replace non-template report {DOC}; pass --replace")
    DOC.write_text(report, encoding="utf-8")
    write_json(OUT / "REPORT_RECEIPT.json", {
        "schema": "stvg-reference-swap-report-receipt-v1", "status": "complete",
        "row_count": 256, "summary_cell_count": 80, "paired_cell_count": 60,
        "common_anchor_count": 38, "teacher_attempt_count": 49,
        "teacher_forward_count": 40, "GT_read": True,
        "inputs": data["root_readback"]["bindings"],
        "outputs": {name: sha(OUT / name) for name in (
            "E1_POSTSEAL_ROWS.json", "E1_POSTSEAL_SUMMARY.json", "E1_PAIRED_DELTAS.json",
            "E1_REFERENCE_QUALITY.json", "E1_TEACHER_READBACK.json", "E1_EXECUTION_COST.json",
            "E1_OPTIMIZER_MATH_READBACK.json",
            "E1_ROOT_READBACK.json", "E1_SCALAR_READBACK.json", "E1_SOURCE_BINDINGS.json", "REFERENCE_SWAP_VIOU.png",
            "REFERENCE_SWAP_CONTRASTS.png")},
        "report_sha256": sha(DOC), "bootstrap_executed": False,
    }, replace=replace)
    print(f"E1 report complete: {OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true",
                        help="replace a prior report only after a reviewed correction")
    args = parser.parse_args()
    run(replace=args.replace)


if __name__ == "__main__":
    main()
