#!/usr/bin/env python3
"""Create anonymous scalar exports and two plots for the sealed Soft EP trials.

This reporting utility reads JSON receipts/results only. It never loads student
checkpoints, source tensors, videos, annotations, or model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_soft_ep_opd_v1"
OUT = ROOT / "results/stvg_soft_ep_opd/2026-10-15"
DOC = ROOT / "docs/STVG_SOFT_EP_OPD_V1.md"
PROTOCOL = ROOT / "protocols/stvg_soft_ep_opd_v1.md"
AUDIT_SCRIPT = ROOT / "scripts/audit_stvg_soft_ep_v1.py"

ARMS = ("F", "T", "S_M0", "TS_M0", "S_M1", "TS_M1", "S_M2", "TS_M2")
METRICS = (
    "m_vIoU", "m_tIoU", "m_sIoU_dense_GT",
    "m_sIoU_supervised_GT", "m_sIoU_unsupervised_GT",
)
TRIALS = (
    ("primary", "mu1", 1.0, ""),
    ("mu025", "mu025", 0.25, "mu025/"),
    ("mu4", "mu4", 4.0, "mu4/"),
)
SOURCE_FILES = (
    "scripts/audit_stvg_soft_ep_v1.py",
    "scripts/report_stvg_soft_ep_v1.py",
    "scripts/publish_stvg_soft_ep_v1.py",
    "scripts/lock_stvg_soft_ep_v1.py",
    "scripts/launch_stvg_soft_ep_v1.py",
    "scripts/review_stvg_soft_ep_v1.py",
    "scripts/run_stvg_soft_ep_v1.py",
    "scripts/lock_stvg_soft_ep_trial_v1.py",
    "scripts/launch_stvg_soft_ep_trial_v1.py",
    "scripts/run_stvg_soft_ep_trial_v1.py",
    "scripts/run_stvg_soft_ep_trial_worker_v1.py",
    "scripts/review_stvg_soft_ep_trial_v1.py",
    "scripts/stvg_soft_ep_fit_v1.py",
    "scripts/stvg_soft_ep_response_v1.py",
    "scripts/measure_stvg_soft_ep_v1.py",
    "scripts/stvg_reference_recovery_common_v1.py",
    "scripts/stvg_reference_recovery_fit_v1.py",
    "scripts/run_stvg_table1_student004.py",
    "scripts/stvg_target_teacher_student_v1.py",
    "scripts/stvg_weightfree_coordinate_fit_v1.py",
    "scripts/stvg_evidence_common_v1.py",
    "scripts/stvg_native_instance_common_v1.py",
    "methods/decota_spatial_opd_v1/configs.json",
    "methods/decota_final_simplified_v1/tensors.py",
    "vg_tta/decota_spatial_opd_tunable_v1.py",
    "vg_tta/tastvg_decota_critic_p0_v1.py",
)
OUTPUT_NAMES = (
    "ANONYMOUS_ROWS.json", "ANONYMOUS_SUMMARIES.json",
    "ANONYMOUS_PAIRED_CONTRASTS.json", "ANONYMOUS_PROCESS_READBACK.json",
    "SOURCE_BINDINGS.json", "SOFT_EP_VIOU_MEANS.png", "SOFT_EP_DIAGNOSTICS.png",
    "REPORT_RECEIPT.json",
)
FORBIDDEN_PUBLIC_JSON = (
    '"query_text"', '"caption"', '"video_path"', '"raw_output"',
    '"source_path"', '"physical_frame_ids"', '"supervised_physical_frame_ids"',
    '"box_xyxy"', '"box_cxcywh"', '"state_before"', '"gradient"',
    '"optimizer_state"', '"GT_coordinates"', '"weights"',
)


class ReportError(RuntimeError):
    pass


def need(condition: Any, message: str) -> None:
    if not condition:
        raise ReportError(message)


def read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(
            ReportError(f"non-finite JSON value in {path.name}: {x}")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {path}: {exc}") from exc


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any, replace: bool) -> None:
    if path.exists() and not replace:
        raise ReportError(f"refusing to overwrite {path.relative_to(ROOT)}; pass --replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def is_alias(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"(?:HC|V)\d{2}", value))


def _validate_trial(rel: str, label: str, expected_mu: float, primary_design_sha: str | None,
                    common: dict[str, Any] | None) -> dict[str, Any]:
    base = BASE / rel
    names = (
        "DESIGN_LOCK.json", "RUNTIME.json", "ROSTER_LOCK.json", "REFERENCE_LOCK.json",
        "STUDENT_GLOBAL_BARRIER.json", "GLOBAL_PREDICTION_BARRIER.json",
        "ROOT_READBACK.json", "CPU_COMPLETION.json", "POSTSEAL_SCORE_ROWS.json",
        "POSTSEAL_SCORE_SUMMARY.json", "POSTSEAL_PAIRED_CONTRASTS.json",
        "ROOT_INDEPENDENT_CPU_AUDIT.json", "ROOT_ROOT_REPEAT_CPU_AUDIT.json",
    )
    docs = {name: read(base / name) for name in names}
    design, runtime = docs["DESIGN_LOCK.json"], docs["RUNTIME.json"]
    roster, ref = docs["ROSTER_LOCK.json"], docs["REFERENCE_LOCK.json"]
    student, seal, pre = (docs["STUDENT_GLOBAL_BARRIER.json"],
                          docs["GLOBAL_PREDICTION_BARRIER.json"], docs["ROOT_READBACK.json"])
    completion = docs["CPU_COMPLETION.json"]
    rows, summary, paired = (docs["POSTSEAL_SCORE_ROWS.json"],
                             docs["POSTSEAL_SCORE_SUMMARY.json"],
                             docs["POSTSEAL_PAIRED_CONTRASTS.json"])
    audit, repeat = (docs["ROOT_INDEPENDENT_CPU_AUDIT.json"],
                     docs["ROOT_ROOT_REPEAT_CPU_AUDIT.json"])

    need(design.get("status") == "locked" and design.get("GT_read") is False
         and design.get("source_count") == 32 and design.get("historically_exposed") is True,
         f"{label}: design lock status/source scope mismatch")
    need(abs(float(design.get("response_settings", {}).get("mu", -1)) - expected_mu) < 1e-12,
         f"{label}: unexpected mu")
    need(runtime.get("status") == "locked" and runtime.get("GT_read") is False,
         f"{label}: runtime is not locked pre-GT")
    for relpath, expected in {**runtime.get("pins", {}), **runtime.get("protected", {})}.items():
        target = ROOT / relpath
        need(target.is_file() and sha(target) == expected, f"{label}: runtime pin mismatch: {relpath}")
    need(roster.get("status") == "locked" and roster.get("GT_read") is False
         and len(roster.get("records", [])) == 32,
         f"{label}: roster is incomplete")
    recs = roster["records"]
    ids = [r.get("probe_alias") for r in recs]
    need(len(set(ids)) == 32 and all(is_alias(x) for x in ids), f"{label}: roster aliases invalid")
    expected_ids = {f"HC{i:02d}" for i in range(1, 17)} | {f"V{i:02d}" for i in range(1, 17)}
    need(set(ids) == expected_ids, f"{label}: roster membership mismatch")
    need(sum(r.get("dataset") == "hc2" for r in recs) == 16
         and sum(r.get("dataset") == "vidstg" for r in recs) == 16,
         f"{label}: domain counts mismatch")
    need(ref.get("status") == "locked" and ref.get("GT_read") is False
         and len(ref.get("records", [])) == 32,
         f"{label}: reference lock is incomplete")
    need(student.get("status") == "complete" and student.get("count") == 32
         and student.get("GT_read") is False and student.get("runtime_sha256") == sha(base / "RUNTIME.json"),
         f"{label}: student barrier mismatch")
    need(seal.get("status") == "complete" and seal.get("count") == 32
         and seal.get("GT_read") is False
         and seal.get("student_barrier_sha256") == sha(base / "STUDENT_GLOBAL_BARRIER.json")
         and seal.get("root_readback_sha256") == sha(base / "ROOT_READBACK.json"),
         f"{label}: prediction seal mismatch")
    need(pre.get("status") == "pass" and pre.get("count") == 32
         and pre.get("GT_read") is False and pre.get("CPU_only") is True,
         f"{label}: pre-GT root readback mismatch")
    need(completion.get("status") == "complete" and completion.get("case_count") == 32
         and completion.get("row_count") == 256 and completion.get("summary_cell_count") == 80
         and completion.get("paired_contrast_count") == 120 and completion.get("GT_read") is True
         and completion.get("bootstrap_executed") is False,
         f"{label}: post-seal scoring completion mismatch")
    for doc, rows_expected, where in ((rows, 256, "rows"), (summary, 80, "summary"), (paired, 120, "paired_contrasts")):
        need(doc.get("status") == "complete" and doc.get("GT_read") is True
             and doc.get("bootstrap_executed") is False
             and len(doc.get(where, [])) == rows_expected,
             f"{label}: incomplete {where}")
    binding = completion.get("bindings", {})
    expected_bindings = {
        "runtime_sha256": sha(base / "RUNTIME.json"),
        "design_lock_sha256": sha(base / "DESIGN_LOCK.json"),
        "roster_sha256": sha(base / "ROSTER_LOCK.json"),
        "reference_lock_sha256": sha(base / "REFERENCE_LOCK.json"),
        "student_barrier_sha256": sha(base / "STUDENT_GLOBAL_BARRIER.json"),
        "root_readback_sha256": sha(base / "ROOT_READBACK.json"),
        "prediction_seal_sha256": sha(base / "GLOBAL_PREDICTION_BARRIER.json"),
    }
    need(all(binding.get(k) == v for k, v in expected_bindings.items()),
         f"{label}: completion bindings mismatch")
    need(all(d.get("bindings") == binding for d in (rows, summary, paired)),
         f"{label}: score table bindings mismatch")
    need(audit.get("status") == "pass" and audit.get("case_count") == 32
         and audit.get("arm_row_count") == 256 and audit.get("summary_count") == 80
         and audit.get("paired_count") == 120 and audit.get("new_GT_read") is False
         and audit.get("model_loaded") is False and audit.get("GPU_used") is False,
         f"{label}: independent saved-step audit incomplete")
    need(audit.get("reader_sha256") == sha(AUDIT_SCRIPT), f"{label}: audit script SHA mismatch")
    need(audit.get("scalar_audit", {}).get("status") == "pass"
         and audit["scalar_audit"].get("max_abs_reaggregation_error") == 0.0,
         f"{label}: saved audit scalar reaggregation failed")
    need(repeat.get("status") == "pass" and repeat.get("case_count") == 32
         and repeat.get("arm_row_count") == 256 and repeat.get("summary_count") == 80
         and repeat.get("paired_count") == 120 and repeat.get("reader_sha256") == audit.get("reader_sha256")
         and repeat.get("new_GT_read") is False and repeat.get("model_loaded") is False
         and repeat.get("GPU_used") is False,
         f"{label}: root repeat readback incomplete")
    need(repeat.get("bindings") == audit.get("bindings"), f"{label}: repeated audit bindings differ")

    # Only mu may differ among the three designs. Keep each trial's own lock SHA.
    if common is None:
        common = {
            "configs": design.get("configs"),
            "settings_without_mu": {k: v for k, v in design.get("response_settings", {}).items() if k != "mu"},
            "roster_sha256": sha(base / "ROSTER_LOCK.json"),
            "reference_lock_sha256": sha(base / "REFERENCE_LOCK.json"),
            "parent_student_barrier_sha256": binding.get("parent_student_barrier_sha256"),
        }
    else:
        need(design.get("configs") == common["configs"], f"{label}: domain fit config changed")
        need({k: v for k, v in design.get("response_settings", {}).items() if k != "mu"}
             == common["settings_without_mu"], f"{label}: non-mu response setting changed")
        need(sha(base / "ROSTER_LOCK.json") == common["roster_sha256"], f"{label}: roster changed")
        need(sha(base / "REFERENCE_LOCK.json") == common["reference_lock_sha256"], f"{label}: reference changed")
        need(binding.get("parent_student_barrier_sha256") == common["parent_student_barrier_sha256"],
             f"{label}: parent Student seal changed")
    if label != "mu1":
        need(design.get("primary_design_sha256") == primary_design_sha,
             f"{label}: child does not bind primary design")
        need(design.get("M1_reuse_from") == "artifacts/stvg_soft_ep_opd_v1",
             f"{label}: M1 reuse source differs")

    # Confirm exact row membership and use only the fixed anonymous roster aliases.
    row_by_key = {}
    for row in rows["rows"]:
        cid = row.get("case_id")
        need(is_alias(cid) and row.get("arm") in ARMS and row.get("dataset") in ("hc2", "vidstg"),
             f"{label}: unexpected row identity")
        key = (cid, row["arm"])
        need(key not in row_by_key, f"{label}: duplicate score row {key}")
        need(set(row.get("metrics", {})) == set(METRICS), f"{label}: metric schema mismatch")
        row_by_key[key] = row
    need(len(row_by_key) == 256 and {cid for cid, _ in row_by_key} == expected_ids
         and {arm for _, arm in row_by_key} == set(ARMS),
         f"{label}: score row roster/arm mismatch")
    summary_key = {(x["dataset"], x["arm"], x["metric"]): x for x in summary["summary"]}
    need(len(summary_key) == 80, f"{label}: summary key mismatch")
    paired_key = {(x["dataset"], x["contrast"], x["metric"]): x for x in paired["paired_contrasts"]}
    need(len(paired_key) == 120, f"{label}: paired key mismatch")
    for item in paired["paired_contrasts"]:
        for field in ("positive_case_ids", "negative_case_ids", "neutral_case_ids", "unknown_case_ids"):
            need(all(is_alias(cid) for cid in item.get(field, [])), f"{label}: non-anonymous paired alias")
        need(set(item.get("per_source_delta_pp", {})).issubset(expected_ids),
             f"{label}: unexpected paired source key")

    # Ensure M1 reuse is recorded and counted correctly for the children.
    expected_reuse = 0 if label == "mu1" else 23
    need(audit.get("reused_fit_count") == expected_reuse
         and audit.get("new_fit_invocation_count") == 46 - expected_reuse,
         f"{label}: fit reuse accounting mismatch")

    source_bindings = audit.get("source_bindings", {})
    need(set(source_bindings) == expected_ids, f"{label}: source binding alias set mismatch")
    for cid, row in source_bindings.items():
        need(set(row) == {"source_case_sha256", "source_file_sha256", "parent_student_sha256"}
             and all(re.fullmatch(r"[0-9a-f]{64}", str(v)) for v in row.values()),
             f"{label}: malformed source binding for {cid}")

    pin_set = {**runtime.get("pins", {}), **runtime.get("protected", {})}
    file_hashes = {name: sha(base / name) for name in names}
    # Keep only hashes, never serialized runtime/lock content or protected path lists.
    file_hashes["ROOT_READBACK.json"] = sha(base / "ROOT_READBACK.json")
    return {
        "label": label, "mu": expected_mu, "base": base, "docs": docs,
        "rows": rows["rows"], "summary": summary["summary"], "paired": paired["paired_contrasts"],
        "audit": audit, "repeat": repeat, "root_readback": pre,
        "roster_ids": ids, "source_bindings": source_bindings,
        "input_hashes": file_hashes,
        "runtime_pin_count": len(pin_set),
        "runtime_pin_inventory_sha256": digest_json(pin_set),
        "row_by_key": row_by_key, "summary_by_key": summary_key, "paired_by_key": paired_key,
        "common": common,
    }


def _fit_audit_public(trial: dict[str, Any]) -> dict[str, Any]:
    audit = trial["audit"]
    error_keys = (
        "max_reward_error", "max_q_error", "max_gaussian_loss_error", "max_reference_iou_error",
        "max_adam_moment_error", "max_adam_delta_error", "max_phi_update_error",
        "max_lambda_relative_error", "max_lambda_abs_error", "max_rho_error",
        "max_cg_residual_error", "max_equal_norm_error",
    )
    fits = []
    cg_rows, equal_rows = [], []
    for item in audit["fits"]:
        cid = item.get("case_id")
        need(is_alias(cid), f"{trial['label']}: invalid fit alias")
        row = {"case_alias": cid, "mode": item.get("mode"),
               "reused_from_parent": bool(item.get("reused")),
               "no_fit": bool(item.get("no_fit")),
               "step_count": int(item.get("step_count", 0)),
               "proposals": int(item.get("proposals", 0)),
               "accepted": int(item.get("accepted", 0)),
               "rejected": int(item.get("rejected", 0)),
               "skipped": int(item.get("skipped", 0)),
               "max_errors": {key: item.get(key) for key in error_keys if key in item}}
        fits.append(row)
        for step in item.get("cg_steps", []):
            cg_rows.append({"case_alias": cid, "mode": item.get("mode"), "step": int(step["step"]),
                            "mu": float(step["mu"]), "lambda": float(step["lambda_value"]),
                            "rho": float(step["rho"]), "iterations": int(step["iterations"]),
                            "solver_status": step["status"],
                            "relative_residual": float(step["residual_rel"])})
        for step in item.get("equal_norm_steps", []):
            matched = float(step["matched_U_RMS"])
            soft = float(step["soft_U_RMS"])
            equal_rows.append({"case_alias": cid, "mode": item.get("mode"),
                               "step": int(step["step"]),
                               "matched_adam_response_rms": matched,
                               "soft_response_rms": soft,
                               "soft_to_matched_response_ratio": soft / matched if matched else None})
    fd = []
    for item in trial["root_readback"].get("FD_diagnostics", []):
        need(is_alias(item.get("case_id")), f"{trial['label']}: invalid FD alias")
        fd.append({"case_alias": item["case_id"], "h": float(item["h"]),
                   "relative_error": float(item["FD_relative_error"]),
                   "relative_error_percent": 100.0 * float(item["FD_relative_error"]),
                   "absolute_error": float(item["FD_absolute_error"]),
                   "gross_agreement": bool(item["gross_agreement"])})
    ratios = [x["soft_to_matched_response_ratio"] for x in equal_rows
              if x["soft_to_matched_response_ratio"] is not None]
    count = audit["fit_count"]
    return {
        "trial": trial["label"], "mu": trial["mu"],
        "cpu_saved_step_audit": {
            "status": audit["status"], "case_count": audit["case_count"],
            "scored_row_count": audit["arm_row_count"], "summary_count": audit["summary_count"],
            "paired_count": audit["paired_count"], "max_reaggregation_error": audit["scalar_audit"]["max_abs_reaggregation_error"],
            "fit_record_count": count, "no_fit_count": audit["no_fit_count"],
            "new_fit_invocation_count": audit["new_fit_invocation_count"],
            "reused_fit_count": audit["reused_fit_count"],
            "proposal_count": audit["proposal_count"], "accepted_count": audit["accepted_count"],
            "rejected_count": audit["rejected_count"], "no_information_or_stopped_skip_count": audit["skipped_count"],
            "cg_witness_step_count": audit["cg_witness_steps"],
            "equal_norm_response_step_count": audit["equal_norm_steps"],
            "equal_norm_soft_response_lower_count": sum(x < 1.0 for x in ratios),
            "equal_norm_soft_to_matched_mean_ratio": math.fsum(ratios) / len(ratios) if ratios else None,
            "equal_norm_soft_to_matched_min_ratio": min(ratios) if ratios else None,
            "equal_norm_soft_to_matched_max_ratio": max(ratios) if ratios else None,
            "max_CG_relative_residual": trial["root_readback"].get("max_CG_relative_residual"),
            "parameter_rho_range": trial["root_readback"].get("parameter_rho_range"),
            "direction_cos_range": trial["root_readback"].get("direction_cos_range"),
            "max_saved_arithmetic_errors": {
                "reward": max((x.get("max_reward_error", 0.0) or 0.0) for x in audit["fits"]),
                "softmax_q": max((x.get("max_q_error", 0.0) or 0.0) for x in audit["fits"]),
                "Gaussian_loss": max((x.get("max_gaussian_loss_error", 0.0) or 0.0) for x in audit["fits"]),
                "reference_IoU": max((x.get("max_reference_iou_error", 0.0) or 0.0) for x in audit["fits"]),
                "Adam_moments": max((x.get("max_adam_moment_error", 0.0) or 0.0) for x in audit["fits"]),
                "Adam_delta": max((x.get("max_adam_delta_error", 0.0) or 0.0) for x in audit["fits"]),
                "accepted_parameter_update": max((x.get("max_phi_update_error", 0.0) or 0.0) for x in audit["fits"]),
                "lambda_relative": max((x.get("max_lambda_relative_error", 0.0) or 0.0) for x in audit["fits"]),
                "rho": max((x.get("max_rho_error", 0.0) or 0.0) for x in audit["fits"]),
                "CG_residual_witness": max((x.get("max_cg_residual_error", 0.0) or 0.0) for x in audit["fits"]),
                "equal_norm_vector": max((x.get("max_equal_norm_error", 0.0) or 0.0) for x in audit["fits"]),
            },
            "finite_difference_checks": fd,
            "fit_records": fits,
            "CG_step_scalars": cg_rows,
            "equal_norm_response_scalars": equal_rows,
            "backward_replayed": False,
            "full_J_reconstructed": False,
            "model_loaded": False,
            "GPU_used": False,
            "new_GT_read_by_auditor": False,
            "scope_limit": "saved arithmetic and solver witnesses only; no independent backward or full-J reconstruction",
        },
    }


def _make_exports(trials: list[dict[str, Any]]) -> dict[str, Any]:
    row_export, summary_export, paired_export, process_export = [], [], [], []
    binding_export = {"schema": "stvg-soft-ep-public-source-bindings-v1", "status": "complete",
                      "GT_payload_exported": False, "records": []}
    for t in trials:
        for row in t["rows"]:
            support = row.get("support", {})
            allowed_counts = (
                "source_grid_positions", "source_grid_supervised_position_count",
                "source_grid_unsupervised_position_count", "supervised_anchor_count",
                "unsupervised_position_count", "dense_gt_known_frame_count",
                "source_grid_gt_known_frame_count", "off_grid_gt_known_frame_count",
                "supervised_gt_known_frame_count", "unsupervised_gt_known_frame_count",
            )
            row_export.append({
                "trial": t["label"], "mu": t["mu"], "case_alias": row["case_id"],
                "dataset": row["dataset"], "arm": row["arm"], "status": row["status"],
                "metrics": {metric: row["metrics"][metric] for metric in METRICS},
                "support_counts": {key: support[key] for key in allowed_counts if key in support},
                "official_independent_metric_max_abs_error": row.get("metric_audit", {}).get("official_independent_max_abs_error"),
                "fit_kind": row.get("fit", {}).get("kind"),
            })
        for row in t["summary"]:
            summary_export.append({
                "trial": t["label"], "mu": t["mu"], "dataset": row["dataset"],
                "arm": row["arm"], "metric": row["metric"], "n_valid": row["n_valid"],
                "n_unknown": row["n_unknown"],
                "unknown_case_aliases": list(row.get("unknown_case_ids", [])),
                "mean": row.get("mean"), "median": row.get("median"),
                "min": row.get("min"), "max": row.get("max"),
            })
        for row in t["paired"]:
            per_source = row.get("per_source_delta_pp", {})
            need(all(is_alias(cid) for cid in per_source), f"{t['label']}: invalid per-source alias")
            paired_export.append({
                "trial": t["label"], "mu": t["mu"], "dataset": row["dataset"],
                "contrast": row["contrast"], "comparison_arm": row["comparison_arm"],
                "baseline_arm": row["baseline_arm"], "metric": row["metric"],
                "n_pairs": row["n_pairs"], "unknown_count": row["unknown_count"],
                "unknown_case_aliases": list(row.get("unknown_case_ids", [])),
                "positive_case_aliases": list(row.get("positive_case_ids", [])),
                "negative_case_aliases": list(row.get("negative_case_ids", [])),
                "neutral_case_aliases": list(row.get("neutral_case_ids", [])),
                "neutral_tolerance_pp": row.get("neutral_tolerance_pp"),
                "mean_delta_pp": row.get("mean_delta_pp"),
                "per_source_delta_pp": per_source,
            })
        process_export.append(_fit_audit_public(t))
        b = t["docs"]
        binding = b["CPU_COMPLETION.json"]["bindings"]
        binding_export["records"].append({
            "trial": t["label"], "mu": t["mu"],
            "input_receipt_sha256": t["input_hashes"],
            "source_code_pin_count": t["runtime_pin_count"],
            "source_code_pin_inventory_sha256": t["runtime_pin_inventory_sha256"],
            "execution_binding_sha256": {k: v for k, v in binding.items()},
            "source_case_hashes_by_alias": t["source_bindings"],
        })
    execution_costs = _execution_cost_export(trials)
    return {
        "rows": {"schema": "stvg-soft-ep-anonymous-rows-v1", "status": "complete",
                 "trial_count": len(trials), "row_count": len(row_export), "rows": row_export},
        "summaries": {"schema": "stvg-soft-ep-anonymous-summary-v1", "status": "complete",
                       "trial_count": len(trials), "summary_count": len(summary_export), "summary": summary_export},
        "paired": {"schema": "stvg-soft-ep-anonymous-paired-v1", "status": "complete",
                    "trial_count": len(trials), "paired_count": len(paired_export), "paired_contrasts": paired_export},
        "process": {"schema": "stvg-soft-ep-anonymous-process-readback-v1", "status": "complete",
                    "trial_count": len(trials), "records": process_export,
                    "execution_costs": execution_costs},
        "bindings": binding_export,
    }


def _execution_cost_export(trials: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    total_fit_invocations = 0
    total_source_captures = 0
    total_full_readouts = 0
    total_cached_m0_replays = 0
    total_reused_m1_records = 0
    total_worker_wall = 0.0
    for t in trials:
        barrier = t["docs"]["STUDENT_GLOBAL_BARRIER.json"]
        fit_count = int(barrier["new_fit_invocations"])
        capture_count = int(barrier["source_captures"])
        full_readouts = fit_count
        cached_m0 = 0
        wall_by_stage = {"development": float(barrier["wall_seconds"])}
        qualification = None
        if t["label"] == "mu1":
            qbar = read(t["base"] / "QUALIFICATION_BARRIER.json")
            qreadback = read(t["base"] / "ROOT_QUALIFICATION.json")
            need(qbar.get("status") == "complete" and qbar.get("GT_read") is False
                 and qbar.get("count") == 4 and qreadback.get("status") == "pass"
                 and qreadback.get("GT_read") is False,
                 "primary: qualification cost receipts are incomplete")
            qfits = int(qbar["new_fit_invocations"])
            qcaptures = int(qbar["source_captures"])
            qreadouts = int(qreadback["actual_full_readout_count"])
            cached_m0 = qfits - qreadouts
            need(qfits == 9 and qcaptures == 4 and qreadouts == 6 and cached_m0 == 3,
                 "primary: unexpected qualification fit/readout accounting")
            fit_count += qfits
            capture_count += qcaptures
            full_readouts += qreadouts
            wall_by_stage["qualification"] = float(qbar["wall_seconds"])
            qualification = {
                "fit_invocations": qfits, "source_capture_calls": qcaptures,
                "actual_fullmodel_final_readouts": qreadouts,
                "M0_cached_bitwise_replay_records": cached_m0,
                "worker_reported_wall_seconds": float(qbar["wall_seconds"]),
                "qualification_barrier_sha256": sha(t["base"] / "QUALIFICATION_BARRIER.json"),
                "readback_sha256": sha(t["base"] / "ROOT_QUALIFICATION.json"),
            }
        reused_m1 = 23 if t["label"] in ("mu025", "mu4") else 0
        worker_wall = sum(wall_by_stage.values())
        records.append({
            "trial": t["label"], "mu": t["mu"], "case_count": 32,
            "fit_invocations_including_qualification": fit_count,
            "new_M1_M2_fit_invocations": full_readouts,
            "source_capture_calls": capture_count,
            "actual_fullmodel_final_readouts": full_readouts,
            "M0_cached_bitwise_replay_records": cached_m0,
            "reused_M1_fit_records_from_primary": reused_m1,
            "worker_reported_wall_seconds_by_stage": wall_by_stage,
            "worker_reported_wall_seconds_sum": worker_wall,
            "student_barrier_sha256": sha(t["base"] / "STUDENT_GLOBAL_BARRIER.json"),
            "qualification": qualification,
        })
        total_fit_invocations += fit_count
        total_source_captures += capture_count
        total_full_readouts += full_readouts
        total_cached_m0_replays += cached_m0
        total_reused_m1_records += reused_m1
        total_worker_wall += worker_wall
    need(total_fit_invocations == 95 and total_source_captures == 70
         and total_full_readouts == 92 and total_cached_m0_replays == 3
         and total_reused_m1_records == 46,
         "three-trial execution cost counts differ from sealed receipts")
    return {
        "schema": "stvg-soft-ep-anonymous-execution-cost-v1", "status": "complete",
        "CPU_only_receipt_readback": True, "new_GT_read": False,
        "records": records,
        "totals": {
            "fit_invocations_including_qualification": total_fit_invocations,
            "source_capture_calls": total_source_captures,
            "actual_fullmodel_final_readouts": total_full_readouts,
            "M0_cached_bitwise_replay_records": total_cached_m0_replays,
            "reused_M1_fit_records_from_primary": total_reused_m1_records,
            "worker_reported_wall_seconds_sum": total_worker_wall,
            "new_teacher_or_temporal_forwards": 0,
            "inherited_teacher_temporal_cost_seconds": None,
            "inherited_teacher_temporal_cost_note": "Upstream Teacher/Temporal evidence was reused; its prior cost is not measured here and must not be interpreted as zero deployment cost.",
            "wall_time_scope": "sum of sealed worker-reported qualification/development stage wall_seconds; excludes upstream reused cost and is not end-to-end deployment latency",
        },
    }


def _mean_summary(trial: dict[str, Any], dataset: str, arm: str, metric: str) -> float | None:
    return trial["summary_by_key"][(dataset, arm, metric)].get("mean")


def _paired(trial: dict[str, Any], dataset: str, contrast: str, metric: str) -> dict[str, Any]:
    return trial["paired_by_key"][(dataset, contrast, metric)]


def render_markdown(trials: list[dict[str, Any]]) -> str:
    out = [
        "# Soft EP-OPD: three finite 32-source development trials",
        "",
        "This report covers the sealed primary `mu=1`, `mu=0.25`, and `mu=4` trials on the same historically exposed 32-source development panel (16 per domain). It is not the separately authorized 128-source search, a clean confirmation result, or a method freeze. The deployed method registry is unchanged.",
        "",
        "## Main vIoU means",
        "",
        "Values are source means in percent. `S_M*` denotes spatial-only; `TS_M*` denotes the original temporal branch plus the spatial arm. F/T and M0/M1 are fixed across the three `mu` trials; only M2's response coefficient changes.",
        "",
    ]
    arm_labels = {"F": "F", "T": "T", "S_M0": "S M0", "TS_M0": "TS M0",
                  "S_M1": "S M1", "TS_M1": "TS M1", "S_M2": "S M2", "TS_M2": "TS M2"}
    for dataset, title in (("hc2", "HC-STVG-v2"), ("vidstg", "VidSTG")):
        out.extend([f"### {title}", "", "| Arm | mu=1 | mu=0.25 | mu=4 |", "|---|---:|---:|---:|"])
        for arm in ARMS:
            vals = [_mean_summary(t, dataset, arm, "m_vIoU") for t in trials]
            out.append(f"| {arm_labels[arm]} | " + " | ".join("—" if x is None else f"{x*100:.6f}" for x in vals) + " |")
        out.append("")
    out.extend([
        "## Paired T+S vIoU differences",
        "",
        "Differences are in percentage points. Positive/negative/neutral counts are per source; the paired neutral tolerance is `1e-10` pp. Every metric, contrast, sign list, per-source alias delta, and null is retained in the JSON exports.",
        "",
        "| Domain | mu | Contrast | Mean delta (pp) | + | − | = | unknown |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for t in trials:
        for dataset, title in (("hc2", "HC"), ("vidstg", "Vid")):
            for contrast in ("TS_M2_minus_TS_M1", "TS_M2_minus_TS_M0"):
                row = _paired(t, dataset, contrast, "m_vIoU")
                out.append(f"| {title} | {t['mu']:g} | {contrast.replace('_minus_', ' − ')} | "
                           f"{row['mean_delta_pp']:+.6f} | {len(row['positive_case_ids'])} | "
                           f"{len(row['negative_case_ids'])} | {len(row['neutral_case_ids'])} | {row['unknown_count']} |")
    out.extend([
        "",
        "The full exports contain 768 source-arm rows (256 per trial), 240 summary cells (80 per trial), and 360 paired cells (120 per trial). The five metrics are preserved separately: vIoU, tIoU, dense sIoU, supervised-anchor sIoU, and unsupervised-frame sIoU. Unknown metric values remain null.",
        "",
        "## Saved-step and response diagnostics",
        "",
        "The independent CPU readback checked 46 fit records and 18 no-fit records per trial. The primary stages recorded 49 total fit invocations: 9 in qualification and 40 in development. Of these, 46 are M1/M2 fits with actual final full-model readouts; three M0 qualification entries are bitwise cached-output replays. Each child reuses 23 M1 fits exactly from primary and performs 23 new M2 fits. Across each trial, the saved record contains 194 proposals, 192 accepted updates, 2 full rollbacks, 36 skips, 98 CG witness steps, and 98 equal-parameter-norm response comparisons. The mean soft-to-matched-Adam output response ratios were:",
        "",
        "| mu | mean soft/matched response RMS | steps lower than matched | max CG relative residual |",
        "|---:|---:|---:|---:|",
    ])
    for t in trials:
        process = _fit_audit_public(t)["cpu_saved_step_audit"]
        out.append(f"| {t['mu']:g} | {process['equal_norm_soft_to_matched_mean_ratio']:.4f} | "
                   f"{process['equal_norm_soft_response_lower_count']}/{process['equal_norm_response_step_count']} | "
                   f"{process['max_CG_relative_residual']:.6g} |")
    costs = _execution_cost_export(trials)
    out.extend([
        "",
        "## Measured execution work",
        "",
        "Counts and stage wall times come from sealed qualification/development barriers. Primary used 49 fit invocations (9 qualification + 40 development), 24 source captures, and 46 actual full-model final readouts; the three remaining M0 qualification records were cached bitwise replays. Each child used 23 new M2 fit invocations and 23 source captures, and reused 23 M1 fit records from primary. The three trials therefore contain 95 fit invocations, 70 source captures, 92 actual full-model final readouts, and 46 child M1 reuse records.",
        "",
        "| Trial stage | Fit invocations | Source captures | Actual full-model final readouts | Worker-reported wall (s) |",
        "|---|---:|---:|---:|---:|",
    ])
    for rec in costs["records"]:
        if rec["qualification"]:
            q = rec["qualification"]
            out.append(f"| primary qualification | {q['fit_invocations']} | {q['source_capture_calls']} | "
                       f"{q['actual_fullmodel_final_readouts']} | {q['worker_reported_wall_seconds']:.3f} |")
        stage_name = "primary development" if rec["trial"] == "mu1" else rec["trial"]
        stage_fits = rec["fit_invocations_including_qualification"] - (
            rec["qualification"]["fit_invocations"] if rec["qualification"] else 0)
        out.append(f"| {stage_name} | {stage_fits} | "
                   f"{rec['source_capture_calls'] - (rec['qualification']['source_capture_calls'] if rec['qualification'] else 0)} | "
                   f"{rec['actual_fullmodel_final_readouts'] - (rec['qualification']['actual_fullmodel_final_readouts'] if rec['qualification'] else 0)} | "
                   f"{rec['worker_reported_wall_seconds_by_stage']['development']:.3f} |")
    totals = costs["totals"]
    out.append(f"| all three trials | {totals['fit_invocations_including_qualification']} | "
               f"{totals['source_capture_calls']} | {totals['actual_fullmodel_final_readouts']} | "
               f"{totals['worker_reported_wall_seconds_sum']:.3f} |")
    out.extend([
        "",
        "Teacher/Temporal inference was not repeated in these trials. Those upstream artifacts were reused, so their earlier cost is unknown here and is not zero end-to-end deployment cost. The wall-time table reports sealed worker stage seconds, not complete deployment latency.",
    ])
    primary_process = _fit_audit_public(trials[0])["cpu_saved_step_audit"]
    fd = primary_process["finite_difference_checks"]
    if fd:
        low, high = min(x["relative_error_percent"] for x in fd), max(x["relative_error_percent"] for x in fd)
        out.extend(["", f"Primary finite-difference checks cover three fixed qualification cases at two step sizes (six checks); relative errors span {low:.2f}%–{high:.2f}%. The original high-accuracy full-decoder finite-difference qualification remains false. The `mu=0.25` and `mu=4` children add no FD checks."])
    out.extend([
        "",
        "These are saved-process measurements, not correctness guarantees: a smaller equal-norm response does not establish better GT alignment. The CPU auditor verifies saved arithmetic and solver witnesses; it does not rerun backward or reconstruct the full decoder Jacobian. The earlier 13-scope numerical qualification failure remains unresolved.",
        "",
        "## Fixed design and interpretation limits",
        "",
        "The trials keep the same 32 historically exposed development cases, parent prediction seal, reference lock, temporal configuration, and domain-specific spatial settings. M2 is the only changed factor (`mu`). Query/Adam reset and the five-step/32-sample settings remain fixed. Scoring is post-seal; GT-derived metrics are not used for adaptation or selecting a best step. No bootstrap was run.",
        "",
        "The 32 cases were previously exposed during development, so these results are not blind confirmation evidence. As of this report cut on 2026-10-15, the separately authorized 128-source coordinate search (64 per domain) is still in preparation and is not represented here as a completed result. Its source-capture seal contains 128 cases: 64 new captures and 64 exact reuses. Temporal/teacher preparation has not produced a complete prediction seal or score; an initial environment attempt stopped because `accelerate` was unavailable, not because of GPU memory exhaustion, and the separately recorded continuation remains incomplete. No method is frozen or promoted here; `methods/CURRENT_METHOD.json` is unchanged.",
        "",
        "## Figures and machine-readable data",
        "",
        "![Full-tube vIoU means across the three mu trials](../results/stvg_soft_ep_opd/2026-10-15/SOFT_EP_VIOU_MEANS.png)",
        "",
        "![Paired effects and equal-norm response diagnostics](../results/stvg_soft_ep_opd/2026-10-15/SOFT_EP_DIAGNOSTICS.png)",
        "",
        "The complete anonymous source-arm rows, all summaries and paired contrasts, saved-step scalar audit, and hash-only source bindings are in `results/stvg_soft_ep_opd/2026-10-15/`. The locked protocol is [here](../protocols/stvg_soft_ep_opd_v1.md).",
        "",
    ])
    return "\n".join(out)


def plot_means(trials: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), constrained_layout=True)
    ordered = sorted(trials, key=lambda t: float(t["mu"]))
    x = [float(t["mu"]) for t in ordered]
    for ax, dataset, title in zip(axes, ("hc2", "vidstg"), ("HC-STVG-v2", "VidSTG")):
        for arm, style, label in (("TS_M0", "--", "T+S M0"), ("TS_M1", "-.", "T+S M1"),
                                  ("TS_M2", "-", "T+S M2")):
            y = [_mean_summary(t, dataset, arm, "m_vIoU") * 100 for t in ordered]
            ax.plot(x, y, marker="o", linestyle=style, linewidth=2, label=label)
        ax.axhline(_mean_summary(ordered[0], dataset, "F", "m_vIoU") * 100,
                   color="#777777", linestyle=":", linewidth=1.4, label="F")
        ax.axhline(_mean_summary(ordered[0], dataset, "T", "m_vIoU") * 100,
                   color="#222222", linestyle=":", linewidth=1.4, label="T")
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, ["0.25", "1", "4"])
        ax.set_xlabel("M2 response coefficient mu")
        ax.set_ylabel("Mean vIoU (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.24)
        ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Soft EP-OPD on the fixed 32-source development panel", fontsize=13)
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)


def plot_diagnostics(trials: list[dict[str, Any]], process_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), constrained_layout=True)
    colors = {"S_M2_minus_S_M1": "#4c78a8", "TS_M2_minus_TS_M1": "#f58518",
              "S_M2_minus_S_M0": "#54a24b", "TS_M2_minus_TS_M0": "#e45756"}
    labels = {"S_M2_minus_S_M1": "S M2 − M1", "TS_M2_minus_TS_M1": "TS M2 − M1",
              "S_M2_minus_S_M0": "S M2 − M0", "TS_M2_minus_TS_M0": "TS M2 − M0"}
    ordered = sorted(trials, key=lambda t: float(t["mu"]))
    x = [float(t["mu"]) for t in ordered]
    for col, dataset in enumerate(("hc2", "vidstg")):
        ax = axes[0, col]
        for contrast in colors:
            y = [_paired(t, dataset, contrast, "m_vIoU")["mean_delta_pp"] for t in ordered]
            ax.plot(x, y, marker="o", linewidth=1.8, color=colors[contrast], label=labels[contrast])
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, ["0.25", "1", "4"])
        ax.set_title("HC-STVG-v2 paired vIoU deltas (pp)" if dataset == "hc2" else "VidSTG paired vIoU deltas (pp)")
        ax.set_xlabel("mu")
        ax.set_ylabel("Mean paired delta (pp)")
        ax.grid(axis="y", alpha=0.24)
        ax.legend(frameon=False, fontsize=7.5, ncol=2)

        ax = axes[1, col]
        domain_prefix = "HC" if dataset == "hc2" else "V"
        trial_ids = [t["label"] for t in ordered]
        data = []
        for label in trial_ids:
            rec = next(r["cpu_saved_step_audit"] for r in process_rows if r["trial"] == label)
            vals = [x["soft_to_matched_response_ratio"] for x in rec["equal_norm_response_scalars"]
                    if x["soft_to_matched_response_ratio"] is not None
                    and str(x.get("case_alias", "")).startswith(domain_prefix)]
            data.append(vals)
        ticks = [f"{t['mu']:g}\n(n={len(vals)})" for t, vals in zip(ordered, data)]
        ax.boxplot(data, tick_labels=ticks, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "#9ecae9", "alpha": 0.75},
                   medianprops={"color": "#222222", "linewidth": 1.5})
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
        ax.set_title("Actual response RMS ratio at equal parameter norm")
        ax.set_xlabel("M2 mu")
        ax.set_ylabel("soft response / matched Adam response")
        ax.grid(axis="y", alpha=0.24)
        ax.text(0.02, 0.02, f"Domain-specific saved comparisons; process diagnostic, not GT", transform=ax.transAxes,
                fontsize=8, color="#444444")
    fig.suptitle("Paired effects and equal-norm response diagnostics", fontsize=13)
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="replace this report's generated outputs")
    args = parser.parse_args()
    need(PROTOCOL.is_file(), "locked protocol is missing")
    trials = []
    common = None
    primary_design_sha = None
    for label, sublabel, mu, rel in TRIALS:
        trial = _validate_trial(rel, sublabel, mu, primary_design_sha, common)
        if common is None:
            common = trial["common"]
            primary_design_sha = sha(trial["base"] / "DESIGN_LOCK.json")
        else:
            trial["common"] = common
        trials.append(trial)

    # The children share the same M1 parent and differ from primary only in M2 mu.
    need(trials[1]["docs"]["DESIGN_LOCK.json"].get("primary_design_sha256") == primary_design_sha
         and trials[2]["docs"]["DESIGN_LOCK.json"].get("primary_design_sha256") == primary_design_sha,
         "child trials are not bound to the primary design")
    exports = _make_exports(trials)
    OUT.mkdir(parents=True, exist_ok=True)
    data = (
        ("ANONYMOUS_ROWS.json", exports["rows"]),
        ("ANONYMOUS_SUMMARIES.json", exports["summaries"]),
        ("ANONYMOUS_PAIRED_CONTRASTS.json", exports["paired"]),
        ("ANONYMOUS_PROCESS_READBACK.json", exports["process"]),
        ("SOURCE_BINDINGS.json", exports["bindings"]),
    )
    for filename, value in data:
        write_json(OUT / filename, value, args.replace)
        raw = (OUT / filename).read_bytes()
        for token in FORBIDDEN_PUBLIC_JSON:
            need(token.encode("utf-8") not in raw, f"private field leaked in {filename}: {token}")

    process_rows = exports["process"]["records"]
    plot_means(trials, OUT / "SOFT_EP_VIOU_MEANS.png")
    plot_diagnostics(trials, process_rows, OUT / "SOFT_EP_DIAGNOSTICS.png")
    report_text = render_markdown(trials)
    if DOC.exists() and not args.replace:
        raise ReportError(f"refusing to overwrite {DOC.relative_to(ROOT)}; pass --replace")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(report_text, encoding="utf-8")

    source_manifest = []
    for relpath in SOURCE_FILES:
        source = ROOT / relpath
        need(source.is_file(), f"public source file missing: {relpath}")
        source_manifest.append({"path": relpath, "sha256": sha(source), "bytes": source.stat().st_size})
    output_hashes = {name: {"sha256": sha(OUT / name), "bytes": (OUT / name).stat().st_size}
                     for name in OUTPUT_NAMES if name != "REPORT_RECEIPT.json"}
    receipt = {
        "schema": "stvg-soft-ep-report-receipt-v1", "status": "complete",
        "scope": "three sealed 32-source development trials; reporting only",
        "trial_count": 3, "case_count_per_trial": 32, "arm_rows_per_trial": 256,
        "summary_cells_per_trial": 80, "paired_cells_per_trial": 120,
        "total_arm_rows": len(exports["rows"]["rows"]),
        "total_summary_cells": len(exports["summaries"]["summary"]),
        "total_paired_cells": len(exports["paired"]["paired_contrasts"]),
        "bootstrap_executed": False, "new_GT_read": False,
        "model_loaded": False, "GPU_used": False,
        "source_file_manifest": source_manifest,
        "input_receipts_sha256_by_trial": [x["input_hashes"] for x in trials],
        "output_files": output_hashes,
        "report_sha256": sha(DOC), "protocol_sha256": sha(PROTOCOL),
        "audit_script_sha256": sha(AUDIT_SCRIPT),
    }
    write_json(OUT / "REPORT_RECEIPT.json", receipt, args.replace)
    print(json.dumps({"status": "complete", "output": str(OUT.relative_to(ROOT)),
                      "report": str(DOC.relative_to(ROOT)),
                      "rows": receipt["total_arm_rows"], "summaries": receipt["total_summary_cells"],
                      "paired": receipt["total_paired_cells"],
                      "report_sha256": receipt["report_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
