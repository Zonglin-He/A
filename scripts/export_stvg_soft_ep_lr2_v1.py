#!/usr/bin/env python3
"""Export the sealed Soft EP LR-x2 trial as an anonymous public package.

This is an offline reader. It refuses to inspect LR2 score files until
CPU_COMPLETION.json says complete, and never loads checkpoints, video, labels,
or model weights. Run only after the LR2 scorer has written its completion
receipt and the independent saved-math readback has passed.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_soft_ep_opd_v1/lr2"
LR1 = ROOT / "artifacts/stvg_soft_ep_opd_v1/mu025"
LR1_DESIGN = LR1 / "DESIGN_LOCK.json"
PARENT = ROOT / "artifacts/stvg_evidence_constrained_v1/round2_prompt009"
OUT = ROOT / "results/stvg_soft_ep_lr2/2026-10-15"
DOC = ROOT / "docs/STVG_SOFT_EP_LR2_V1.md"
PROTOCOL = ROOT / "protocols/stvg_soft_ep_lr2_v1.md"

# The seven new LR2 execution sources pinned by lr2/RUNTIME.json.
LR2_SOURCES = (
    "scripts/contract_stvg_soft_ep_lr2_v1.py",
    "scripts/launch_stvg_soft_ep_lr2_v1.py",
    "scripts/lock_stvg_soft_ep_lr2_v1.py",
    "scripts/measure_stvg_soft_ep_lr2_v1.py",
    "scripts/prepare_stvg_soft_ep_lr2_v1.py",
    "scripts/review_stvg_soft_ep_lr2_v1.py",
    "scripts/run_stvg_soft_ep_lr2_v1.py",
)
MATH_AUDIT_SOURCE = "scripts/audit_stvg_soft_ep_lr2_math_v1.py"
ARMS = (
    "F", "T", "S_M0", "TS_M0", "S_M1", "TS_M1", "S_M2", "TS_M2",
    "S_M0matched", "TS_M0matched",
)
METRICS = (
    "m_vIoU", "m_tIoU", "m_sIoU_dense_GT",
    "m_sIoU_supervised_GT", "m_sIoU_unsupervised_GT",
)
INTRATRIAL_CONTRASTS = (
    ("S_M2_minus_S_M1", "S_M2", "S_M1"),
    ("TS_M2_minus_TS_M1", "TS_M2", "TS_M1"),
    ("S_M2_minus_S_M0", "S_M2", "S_M0"),
    ("TS_M2_minus_TS_M0", "TS_M2", "TS_M0"),
    ("S_M1_minus_S_M0", "S_M1", "S_M0"),
    ("TS_M1_minus_TS_M0", "TS_M1", "TS_M0"),
    ("S_M1_minus_S_M0matched", "S_M1", "S_M0matched"),
    ("TS_M1_minus_TS_M0matched", "TS_M1", "TS_M0matched"),
    ("S_M2_minus_S_M0matched", "S_M2", "S_M0matched"),
    ("TS_M2_minus_TS_M0matched", "TS_M2", "TS_M0matched"),
    ("S_M0matched_minus_S_M0", "S_M0matched", "S_M0"),
    ("TS_M0matched_minus_TS_M0", "TS_M0matched", "TS_M0"),
    ("S_M0matched_minus_F", "S_M0matched", "F"),
    ("TS_M0matched_minus_T", "TS_M0matched", "T"),
)
CROSS_CONTRASTS = (
    ("LR2_M1_vs_LR1_M1_S", "S_M1", "S_M1"),
    ("LR2_M1_vs_LR1_M1_TS", "TS_M1", "TS_M1"),
    ("LR2_M2_vs_LR1_M2_S", "S_M2", "S_M2"),
    ("LR2_M2_vs_LR1_M2_TS", "TS_M2", "TS_M2"),
)
EPS_PP = 1e-10


class NotReady(RuntimeError):
    pass


def need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    need(isinstance(value, dict), f"expected JSON object: {path.name}")
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def relpath(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    need(not path.exists() and not tmp.exists(), f"refusing to overwrite {path.name}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def replace_json(path: Path, value: Any) -> None:
    """Atomically replace an already exported metadata receipt."""
    tmp = path.with_name(path.name + ".refresh.tmp")
    need(not tmp.exists(), f"refusing stale refresh temporary: {tmp.name}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def replace_text(path: Path, value: str) -> None:
    tmp = path.with_name(path.name + ".refresh.tmp")
    need(not tmp.exists(), f"refusing stale refresh temporary: {tmp.name}")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def completion_gate() -> dict[str, Any]:
    """This is deliberately the first LR2 read performed by main()."""
    path = BASE / "CPU_COMPLETION.json"
    if not path.is_file():
        raise NotReady("LR2 CPU_COMPLETION.json is absent; no LR2 scores were read")
    completion = read_json(path)
    if completion.get("status") != "complete":
        raise NotReady("LR2 CPU_COMPLETION.json is not complete; no LR2 scores were read")
    expected = {
        "case_count": 32, "row_count": 320, "summary_cell_count": 100,
        "paired_contrast_count": 140, "cross_trial_contrast_count": 40,
        "GT_read": True, "CPU_only": True, "model_loaded": False,
        "GPU_used": False, "bootstrap_executed": False,
    }
    for key, want in expected.items():
        need(completion.get(key) == want, f"CPU completion field mismatch: {key}")
    return completion


def hash_bindings() -> dict[str, str]:
    paths = {
        "runtime_sha256": BASE / "RUNTIME.json",
        "design_sha256": BASE / "DESIGN_LOCK.json",
        "roster_sha256": BASE / "ROSTER_LOCK.json",
        "reference_lock_sha256": BASE / "REFERENCE_LOCK.json",
        "student_barrier_sha256": BASE / "STUDENT_GLOBAL_BARRIER.json",
        "root_readback_sha256": BASE / "ROOT_READBACK.json",
        "prediction_seal_sha256": BASE / "GLOBAL_PREDICTION_BARRIER.json",
        "lr1_runtime_sha256": LR1 / "RUNTIME.json",
        "lr1_student_barrier_sha256": LR1 / "STUDENT_GLOBAL_BARRIER.json",
        "lr1_global_prediction_barrier_sha256": LR1 / "GLOBAL_PREDICTION_BARRIER.json",
        "lr1_score_rows_sha256": LR1 / "POSTSEAL_SCORE_ROWS.json",
        "parent_global_prediction_barrier_sha256": PARENT / "GLOBAL_PREDICTION_BARRIER.json",
    }
    return {key: sha256(path) for key, path in paths.items()}


def verify_seals(completion: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bindings = hash_bindings()
    need(completion.get("bindings") == bindings, "CPU completion bindings do not match current seals")
    runtime = read_json(BASE / "RUNTIME.json")
    need(runtime.get("status") == "locked" and runtime.get("GT_read") is False, "LR2 runtime lock invalid")
    # Verify the explicitly published LR2 source closure and protected method
    # metadata only. Do not hash/load model, media, or tensor payloads here.
    for rel in LR2_SOURCES:
        digest = runtime.get("pins", {}).get(rel)
        need(isinstance(digest, str), f"LR2 source is not runtime-pinned: {rel}")
        path = (ROOT / rel).resolve()
        need(ROOT in path.parents and path.is_file() and sha256(path) == digest, f"runtime pin mismatch: {rel}")
    for rel, digest in runtime.get("protected", {}).items():
        path = (ROOT / rel).resolve()
        need(ROOT in path.parents and path.is_file() and sha256(path) == digest, f"protected file pin mismatch: {rel}")
    source_rows = []
    for rel in LR2_SOURCES:
        path = ROOT / rel
        expected = runtime.get("pins", {}).get(rel)
        need(path.is_file() and isinstance(expected, str) and sha256(path) == expected,
             f"LR2 execution source pin mismatch: {rel}")
        source_rows.append({"path": rel, "sha256": expected, "bytes": path.stat().st_size})

    math_path = ROOT / MATH_AUDIT_SOURCE
    math_readback = read_json(BASE / "ROOT_ADDITIONAL_MATH_READBACK.json")
    need(math_path.is_file() and math_readback.get("status") == "pass"
         and math_readback.get("GT_read") is False and math_readback.get("CPU_only") is True
         and math_readback.get("model_loaded") is False,
         "independent LR2 saved-math readback is missing or not pass")
    need(math_readback.get("reader_sha256") == sha256(math_path)
         and math_readback.get("runtime_sha256") == bindings["runtime_sha256"]
         and math_readback.get("student_barrier_sha256") == bindings["student_barrier_sha256"]
         and math_readback.get("count") == 32 and math_readback.get("soft_fit_count") == 46
         and math_readback.get("M0matched_fit_count") == 23,
         "independent LR2 math readback binding/count mismatch")
    source_rows.append({"path": MATH_AUDIT_SOURCE, "sha256": sha256(math_path),
                        "bytes": math_path.stat().st_size, "role": "independent_saved_math_audit"})

    root_readback = read_json(BASE / "ROOT_READBACK.json")
    need(root_readback.get("status") == "pass" and root_readback.get("count") == 32
         and root_readback.get("GT_read") is False,
         "LR2 pre-GT independent readback is missing or invalid")
    student_barrier = read_json(BASE / "STUDENT_GLOBAL_BARRIER.json")
    need(student_barrier.get("status") == "complete" and student_barrier.get("count") == 32
         and student_barrier.get("GT_read") is False
         and student_barrier.get("runtime_sha256") == bindings["runtime_sha256"],
         "LR2 student barrier invalid")
    pred = read_json(BASE / "GLOBAL_PREDICTION_BARRIER.json")
    need(pred.get("status") == "complete" and pred.get("count") == 32
         and pred.get("GT_read") is False
         and pred.get("student_barrier_sha256") == bindings["student_barrier_sha256"]
         and pred.get("root_readback_sha256") == bindings["root_readback_sha256"],
         "LR2 prediction seal invalid")
    exposure = read_json(BASE / "GT_EXPOSURE.json")
    need(exposure.get("GT_read") is True and exposure.get("bindings") == bindings,
         "LR2 GT exposure does not bind the completed seal chain")
    return {"hashes": bindings, "root_readback": root_readback,
            "student_barrier": student_barrier, "prediction_seal": pred,
            "math_readback": math_readback, "gt_exposure_sha256": sha256(BASE / "GT_EXPOSURE.json")}, source_rows


def validate_rows(doc: dict[str, Any], roster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aliases = [x["probe_alias"] for x in roster]
    domain = {x["probe_alias"]: x["dataset"] for x in roster}
    need(len(aliases) == 32 and len(set(aliases)) == 32, "LR2 roster aliases malformed")
    need(sum(domain[x] == "hc2" for x in aliases) == 16
         and sum(domain[x] == "vidstg" for x in aliases) == 16, "LR2 domain roster counts differ")
    need(doc.get("status") == "complete" and doc.get("GT_read") is True
         and doc.get("case_count") == 32 and doc.get("row_count") == 320,
         "LR2 scored rows document incomplete")
    rows = doc.get("rows")
    need(isinstance(rows, list) and len(rows) == 320, "LR2 row count mismatch")
    seen = set()
    for row in rows:
        cid, arm = row.get("case_id"), row.get("arm")
        need(cid in domain and row.get("dataset") == domain[cid] and arm in ARMS,
             "LR2 row identity/arm mismatch")
        key = (cid, arm)
        need(key not in seen, "duplicate LR2 score row")
        seen.add(key)
        metrics = row.get("metrics")
        need(isinstance(metrics, dict) and set(metrics) == set(METRICS), "LR2 metric field set mismatch")
        for metric, value in metrics.items():
            need(value is None or math.isfinite(float(value)), f"nonfinite metric {metric}")
    need(seen == {(cid, arm) for cid in aliases for arm in ARMS}, "LR2 row coverage mismatch")
    return rows


def validate_aggregate(doc: dict[str, Any], expected_count: int, key_fields: tuple[str, ...], label: str) -> list[dict[str, Any]]:
    rows = doc.get(label)
    need(doc.get("status") == "complete" and doc.get("GT_read") is True
         and doc.get("case_count") == 32,
         f"{label} document incomplete")
    need(isinstance(rows, list) and len(rows) == expected_count, f"{label} count mismatch")
    if label == "contrasts":
        need(doc.get("row_count") == expected_count, "cross-trial row_count mismatch")
    keys = [tuple(x.get(k) for k in key_fields) for x in rows]
    need(len(set(keys)) == len(keys), f"duplicate {label} key")
    for row in rows:
        for field in ("positive_case_ids", "negative_case_ids", "neutral_case_ids", "unknown_case_ids"):
            if field in row:
                need(all(isinstance(x, str) and re.fullmatch(r"(?:HC|V)\d{2}", x) for x in row[field]),
                     f"non-anonymous id in {label}.{field}")
    return rows


def project_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep all scalar metrics and safe counts; drop physical IDs/positions."""
    safe_count_keys = (
        "source_grid_positions", "source_grid_supervised_position_count",
        "source_grid_unsupervised_position_count", "supervised_anchor_count",
        "unsupervised_position_count", "dense_gt_known_frame_count",
        "source_grid_gt_known_frame_count", "off_grid_gt_known_frame_count",
        "supervised_gt_known_frame_count", "unsupervised_gt_known_frame_count",
    )
    fit_keys = ("kind", "optimizer_steps", "proposed_steps", "rejected_steps",
                "declared_steps", "active_parameters", "algorithm")
    projected = []
    for row in rows:
        support = row.get("support") or {}
        audit = row.get("metric_audit") or {}
        source_grid = audit.get("source_grid_frames")
        interval_bounds = audit.get("interval_frame_bounds_half_open")
        fit = row.get("fit") or {}
        projected.append({
            "case_id": row["case_id"], "dataset": row["dataset"], "arm": row["arm"],
            "status": row.get("status"), "metrics": {m: row["metrics"].get(m) for m in METRICS},
            "support_counts": {k: int(support[k]) for k in safe_count_keys if isinstance(support.get(k), (int, float))},
            "metric_audit": {
                "official_independent_max_abs_error": audit.get("official_independent_max_abs_error"),
                "source_grid_frame_count": len(source_grid) if isinstance(source_grid, (list, tuple)) else source_grid,
                "interval_bound_count": len(interval_bounds) if isinstance(interval_bounds, (list, tuple)) else interval_bounds,
            },
            "fit": {k: fit[k] for k in fit_keys if k in fit},
        })
    return projected


def project_aggregate(rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    fields = {
        "summary": ("dataset", "arm", "metric", "n_valid", "n_unknown", "unknown_case_ids", "mean", "median", "min", "max"),
        "paired": ("dataset", "contrast", "comparison_arm", "baseline_arm", "metric", "n_pairs", "unknown_count",
                   "unknown_case_ids", "positive_case_ids", "negative_case_ids", "neutral_case_ids", "neutral_tolerance_pp",
                   "mean_delta_pp", "per_source_delta_pp"),
        "cross": ("dataset", "contrast", "comparison_arm", "baseline_trial", "baseline_arm", "metric", "n_pairs",
                  "unknown_count", "unknown_case_ids", "positive_case_ids", "negative_case_ids", "neutral_case_ids",
                  "neutral_tolerance_pp", "mean_delta_pp", "per_source_delta_pp", "interpretation"),
    }[kind]
    out = []
    for row in rows:
        out.append({k: row[k] for k in fields if k in row})
    return out


def execution_readback(rows: list[dict[str, Any]], seals: dict[str, Any], completion: dict[str, Any]) -> dict[str, Any]:
    sb = seals["student_barrier"]
    root = seals["root_readback"]
    unique_fit = {}
    fit_row_count = 0
    exact_parent_reuse_rows = 0
    no_fit_rows = 0
    for row in rows:
        kind = (row.get("fit") or {}).get("kind")
        if kind == "reference_fit":
            fit_row_count += 1
            if row["arm"].startswith("S_"):
                unique_fit[(row["case_id"], row["arm"][2:])] = row["fit"]
        elif kind == "exact_parent_reuse":
            exact_parent_reuse_rows += 1
        else:
            no_fit_rows += 1
    by_method = {m: sum(1 for _, name in unique_fit if name == m) for m in ("M1", "M2", "M0matched")}
    row_fit = {(x["case_id"], x["arm"]): x.get("fit") for x in rows}
    for cid in {x["case_id"] for x in rows}:
        for method in ("M1", "M2", "M0matched"):
            spatial = row_fit[cid, "S_" + method]
            temporal = row_fit[cid, "TS_" + method]
            need(spatial == temporal, f"paired S/TS fit metadata differs: {cid}/{method}")
    need(sum(by_method.values()) == int(sb.get("fit_invocations", -1)) == int(completion.get("fit_invocations", -2)),
         "unique scored fits differ from worker/completion counts")
    need(int(sb.get("fit_invocations", -1)) == int(completion.get("fit_invocations", -2)),
         "fit count differs from worker/completion")
    return {
        "schema": "stvg-soft-ep-lr2-anonymous-execution-readback-v1",
        "status": "complete", "GT_read": True,
        "fit_invocations": int(sb["fit_invocations"]),
        "fit_invocations_by_method": by_method,
        "optimizer_steps_by_method": root.get("optimizer_steps_by_method"),
        "source_capture_calls": int(sb.get("source_captures")),
        "actual_full_readout_calls": int(sb.get("full_readout_calls")),
        "reference_fit_arm_rows": fit_row_count,
        "exact_parent_reuse_arm_rows": exact_parent_reuse_rows,
        "no_fit_arm_rows": no_fit_rows,
        "nonempty_reference_sources": root.get("nonempty_reference_sources"),
        "empty_reference_sources": root.get("empty_reference_sources"),
        "readout_calls_reuse_per_fit": "Each unique fitted state is decoded once, then represented by S and TS arm score rows; row count is not a decoder-call count.",
        "M1_or_M2_reused_from_LR1": 0,
        "cross_trial_contrast_cell_count": 40,
        "completion_sha256": sha256(BASE / "CPU_COMPLETION.json"),
        "student_barrier_sha256": seals["hashes"]["student_barrier_sha256"],
        "root_math_readback": {
            "status": "pass", "reader_sha256": seals["math_readback"].get("reader_sha256"),
            "readback_sha256": sha256(BASE / "ROOT_ADDITIONAL_MATH_READBACK.json"),
            "soft_fit_count": seals["math_readback"].get("soft_fit_count"),
            "M0matched_fit_count": seals["math_readback"].get("M0matched_fit_count"),
            "scope": seals["math_readback"].get("scope"),
        },
        "bootstrap_executed": False,
        "inherited_teacher_temporal_cost_seconds": None,
        "inherited_cost_note": "This LR2 trial issued no new Teacher or Temporal calls; prior shared expert cost is not measured here, not zero.",
    }


def mean_v(values: list[float]) -> float | None:
    return 100.0 * math.fsum(values) / len(values) if values else None


def report_text(summaries: list[dict[str, Any]], paired: list[dict[str, Any]], cross: list[dict[str, Any]], execution: dict[str, Any]) -> str:
    sm = {(x["dataset"], x["arm"], x["metric"]): x for x in summaries}
    pm = {(x["dataset"], x["contrast"], x["metric"]): x for x in paired}
    lines = [
        "# Soft EP-OPD LR×2 matched comparison",
        "",
        "Status: LR2 post-seal anonymous report. The 32-source roster is development data with historical exposure; this is not a blind confirmation or a full-cohort result.",
        "",
        "The trial fixes `mu=0.25` and doubles the learning rate while keeping the locked references, temporal intervals, sigma, tau, step count, writeback, sample count, and response settings fixed. It includes the historical exact-parent M0 arms (`S_M0`, `TS_M0`) and freshly fitted same-LR validated controls (`S_M0matched`, `TS_M0matched`); these are distinct conditions.",
        "",
        f"Sealed scoring contains {execution['fit_invocations']} unique fit invocations across {execution['nonempty_reference_sources']} non-empty-reference sources, {execution['actual_full_readout_calls']} actual full-model readouts, and {execution['source_capture_calls']} source captures. There are {execution['exact_parent_reuse_arm_rows']} scorer-labeled exact-parent M0 arm rows; fit/readout counts are not multiplied by paired S/TS score rows. No LR1 M1/M2 fit was reused for LR2. The prior LR1 scores are used only as matched-case comparison baselines.",
        "",
        "## Full-tube vIoU by domain",
        "",
        "Means are percentages. The complete five-metric rows, all sign tails, unknown cases, and per-source deltas are in the JSON exports.",
        "",
        "| Domain | Arm | Valid | Unknown | Mean vIoU (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds, label in (("hc2", "HC-STVG-v2"), ("vidstg", "VidSTG")):
        for arm in ARMS:
            x = sm[ds, arm, "m_vIoU"]
            mean = x.get("mean")
            lines.append(f"| {label} | {arm} | {x.get('n_valid')} | {x.get('n_unknown')} | {100*float(mean):.4f} |" if mean is not None else f"| {label} | {arm} | {x.get('n_valid')} | {x.get('n_unknown')} | null |")
    lines += ["", "## Same-LR2 matched controls", "",
              "These are paired T+S full-tube vIoU contrasts within LR2. `M0matched` is the freshly fitted validated OPD control at the same doubled learning rate. The historical `M0` rows reuse a prior-LR fit, so they do not replace this same-LR control.",
              "", "| Domain | Contrast | Pairs | + / − / = / unknown | Mean Δ (pp) |", "|---|---|---:|---:|---:|"]
    for ds, label in (("hc2", "HC-STVG-v2"), ("vidstg", "VidSTG")):
        for contrast, display in (("TS_M2_minus_TS_M1", "T+S M2 − M1"),
                                  ("TS_M2_minus_TS_M0matched", "T+S M2 − M0matched")):
            row = pm[ds, contrast, "m_vIoU"]
            tails = f"{len(row['positive_case_ids'])} / {len(row['negative_case_ids'])} / {len(row['neutral_case_ids'])} / {row['unknown_count']}"
            delta = row.get("mean_delta_pp")
            lines.append(f"| {label} | {display} | {row['n_pairs']} | {tails} | {float(delta):+.6f} |" if delta is not None else f"| {label} | {display} | {row['n_pairs']} | {tails} | null |")
    lines += ["", "M2 improves over M0matched in HC-STVG-v2 but declines in VidSTG; this does not establish a dual-domain gain over the same-LR validated OPD control.",
              "", "## Matched comparison with the prior LR×1 trial", "",
              "The baseline is the already sealed `mu=0.25` LR×1 trial. Each value is LR2 minus LR1 in percentage points; this is a cross-trial paired comparison, not an additional LR2 arm.",
              "", "| Domain | Contrast | Metric | Pairs | + / − / = / unknown | Mean Δ (pp) |", "|---|---|---|---:|---:|---:|"]
    for ds, label in (("hc2", "HC-STVG-v2"), ("vidstg", "VidSTG")):
        for row in cross:
            if row["dataset"] == ds and row["metric"] == "m_vIoU":
                tails = f"{len(row['positive_case_ids'])} / {len(row['negative_case_ids'])} / {len(row['neutral_case_ids'])} / {row['unknown_count']}"
                lines.append(f"| {label} | {row['contrast']} | m_vIoU | {row['n_pairs']} | {tails} | {float(row['mean_delta_pp']):+.4f} |" if row.get("mean_delta_pp") is not None else f"| {label} | {row['contrast']} | m_vIoU | {row['n_pairs']} | {tails} | null |")
    lines += [
        "",
        "## Limits",
        "",
        "All 32 cases are exposed development sources. The LR2 comparison does not establish unseen-source performance, method qualification, or a production change. No bootstrap was run. Unknown metric values remain null. The run uses the already fixed Teacher/Temporal references; no new expert calls were made in this trial, and the cost of the inherited upstream calls is unknown. The separate 128-source coordinate search is not represented here as a completed or scored result.",
        "",
        "The inherited finite-difference diagnostic discrepancy range is 7.64–19.58%. `high_accuracy_complete_decoder_FD_qualification=false`: the saved-math audit here checks saved reward/q/loss/Adam/CG/rollback/equal-norm witnesses; it does not qualify a high-accuracy complete-decoder Jacobian.",
        "",
        "The source/configuration hashes, exact arm rows, 100 domain-summary cells, 140 within-trial paired cells, 40 LR1 cross-trial cells, and execution counts are preserved in this directory.",
        "Exact per-domain six-scalar settings, response parameters, Adam hyperparameters, active parameter count, and the LR1 M0 versus LR2 M0matched distinction are in `ANONYMOUS_CONFIGURATION.json`.",
        "",
    ]
    return "\n".join(lines)


def protocol_text() -> str:
    return """# Soft EP-OPD LR×2 protocol\n\n## Scope and status\n\nThis is the fixed 32-source LR×2 matched comparison under `artifacts/stvg_soft_ep_opd_v1/lr2`. The roster contains 16 HC-STVG-v2 and 16 VidSTG program aliases. These sources have historical development exposure; the trial is not a blind confirmation. This document does not authorize new fitting, Teacher/Temporal inference, bootstrap, or a production-method change.\n\n## Intervention and arms\n\nThe trial holds `mu=0.25` and applies a learning-rate multiplier of 2.0. The design lock holds sigma, tau, steps, writeback, sample count, response settings, references, and temporal intervals fixed. The ten arms are F, T, S_M0, TS_M0, S_M1, TS_M1, S_M2, TS_M2, S_M0matched, and TS_M0matched. S/TS denotes spatial-only versus temporal-plus-spatial decoding. `M0` is the exact historical parent reuse; `M0matched` is a fresh validated OPD fit at the LR×2 setting. M1 and M2 are also freshly fitted at LR×2. The LR×1 M1/M2 outputs are comparison baselines only. The T+S contrasts M2−M1 and M2−M0matched are both reported within LR2; historical M0 is a prior-LR reuse and cannot replace the same-LR validated OPD control.\n\n## Outcomes and aggregation\n\nRetain all 320 case-arm rows, 100 domain-by-arm-by-metric summaries, 140 within-LR2 paired cells, and 40 matched LR2-vs-LR1 cross-trial cells. Metrics remain separate: full-tube vIoU, tIoU, dense sIoU, supervised-anchor sIoU, and unsupervised-frame sIoU. Null values remain unknown; positive, negative, neutral, and unknown case tails remain explicit. No bootstrap is used.\n\n## Execution and provenance\n\nThe exporter first requires `CPU_COMPLETION.json` with complete status and matching row, summary, and contrast counts. It then verifies the runtime and protected hashes, student/prediction seals, pre-GT readback, GT-exposure binding, saved-math audit, and the LR×1/parent comparison bindings before reading score files. It publishes only program aliases, scalar metrics/counts, and hashes; it drops frame IDs, coordinates, tensors, raw queries, and source payloads. The explicit public manifest lists the seven LR2 execution sources, the independent math auditor, and the reporting/export source.\n\n## Interpretation\n\nThis is a finite exposed-development matched comparison, not an unseen-source estimate and not a method qualification. Reused upstream Teacher/Temporal cost is unknown and must not be described as zero deployment cost. The inherited finite-difference diagnostic discrepancy range is 7.64–19.58%; `high_accuracy_complete_decoder_FD_qualification=false`, and the saved-math audit does not qualify a high-accuracy complete-decoder Jacobian. The separate 128-source coordinate search remains outside this result unless its own completed score and seal are published.\n"""





def configuration_payload() -> dict[str, Any]:
    """Export only fixed, non-private algorithm settings and their source hashes."""
    lr2 = read_json(BASE / "DESIGN_LOCK.json")
    lr1 = read_json(LR1_DESIGN)
    lr2_cfg = lr2.get("configs")
    lr1_cfg = lr1.get("configs")
    cfg_keys = {"lr", "samples", "sigma", "steps", "tau", "writeback"}
    need(isinstance(lr2_cfg, dict) and set(lr2_cfg) == {"hc2", "vidstg"}, "LR2 config domains differ")
    need(isinstance(lr1_cfg, dict) and set(lr1_cfg) == {"hc2", "vidstg"}, "LR1 config domains differ")
    for configs in (lr2_cfg, lr1_cfg):
        for dataset in ("hc2", "vidstg"):
            need(set(configs[dataset]) == cfg_keys, f"config field set differs for {dataset}")
    fitter = ROOT / "scripts/stvg_soft_ep_fit_v1.py"
    fitter_text = fitter.read_text(encoding="utf-8")
    need("betas=(.9,.999)" in fitter_text and "eps=1e-8,weight_decay=0." in fitter_text,
         "Adam settings no longer match the fitter implementation")
    runtime = read_json(BASE / "RUNTIME.json")
    need(runtime.get("pins", {}).get("scripts/stvg_soft_ep_fit_v1.py") == sha256(fitter),
         "Adam fitter source is not bound to the locked LR2 runtime")
    return {
        "schema": "stvg-soft-ep-lr2-anonymous-configuration-v1",
        "status": "complete",
        "GT_read": False,
        "lr2": {
            "configs_by_domain": lr2_cfg,
            "response_settings": lr2.get("response_settings"),
            "optimizer": {"name": "Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
            "active_parameter_count": 1792,
            "episodic": lr2.get("Episodic"),
            "query_adam_reset": lr2.get("Query_Adam_reset"),
            "ln_carry": lr2.get("LN_carry"),
            "temporal_input": lr2.get("Temporal"),
            "reference_role": lr2.get("reference_role"),
        },
        "M0_arm": {
            "fit_type": "exact historical validated current5 fit and complete Student output reuse",
            "source_lr1_configs_by_domain": lr1_cfg,
            "reused_arm_row_count": 64,
            "new_lr2_fit_invocations": 0,
        },
        "M0matched_arm": {
            "fit_type": "fresh validated OPD fit at LR2 settings",
            "configs_by_domain": lr2_cfg,
            "fit_invocation_count": 23,
        },
        "provenance": {
            "lr2_design_lock_sha256": sha256(BASE / "DESIGN_LOCK.json"),
            "lr1_design_lock_sha256": sha256(LR1_DESIGN),
            "optimizer_fitter_source": {
                "path": "scripts/stvg_soft_ep_fit_v1.py",
                "sha256": sha256(fitter),
                "bytes": fitter.stat().st_size,
            },
        },
        "limitations": {
            "inherited_finite_difference_discrepancy_percent_range": [7.64, 19.58],
            "high_accuracy_complete_decoder_FD_qualification": False,
            "saved_math_audit_scope": "saved reward/q/loss/Adam/CG/rollback/equal-norm witnesses; no independent backward or complete Jacobian qualification",
        },
    }

def generate() -> dict[str, Any]:
    # Critical ordering: do not open any LR2 score file before this gate returns.
    completion = completion_gate()
    seals, source_rows = verify_seals(completion)

    rows_doc = read_json(BASE / "POSTSEAL_SCORE_ROWS.json")
    summary_doc = read_json(BASE / "POSTSEAL_SCORE_SUMMARY.json")
    paired_doc = read_json(BASE / "POSTSEAL_PAIRED_CONTRASTS.json")
    cross_doc = read_json(BASE / "CROSS_TRIAL_LR1_CONTRASTS.json")
    roster_doc = read_json(BASE / "ROSTER_LOCK.json")
    need(summary_doc.get("bindings") == completion.get("bindings")
         and paired_doc.get("bindings") == completion.get("bindings")
         and cross_doc.get("bindings") == completion.get("bindings"),
         "LR2 score output bindings differ from CPU completion")
    roster = roster_doc.get("records", [])
    rows = validate_rows(rows_doc, roster)
    summaries = validate_aggregate(summary_doc, 100, ("dataset", "arm", "metric"), "summary")
    paired = validate_aggregate(paired_doc, 140, ("dataset", "contrast", "metric"), "paired_contrasts")
    cross = validate_aggregate(cross_doc, 40, ("dataset", "contrast", "metric"), "contrasts")
    need(rows_doc.get("bindings") == completion.get("bindings"), "LR2 row binding mismatch")
    need(cross_doc.get("baseline_trial") == "artifacts/stvg_soft_ep_opd_v1/mu025",
         "cross-trial baseline is not the locked LR1 trial")

    row_keys = {(x["case_id"], x["arm"]) for x in rows}
    execution = execution_readback(rows, seals, completion)
    projected_rows = project_rows(rows)
    projected_summaries = project_aggregate(summaries, "summary")
    projected_paired = project_aggregate(paired, "paired")
    projected_cross = project_aggregate(cross, "cross")
    configuration = configuration_payload()
    source_bindings = {
        "schema": "stvg-soft-ep-lr2-anonymous-source-bindings-v1",
        "status": "complete", "GT_read": True,
        "lr2_execution_sources": source_rows[:len(LR2_SOURCES)],
        "independent_math_audit_source": source_rows[-1],
        "exporter_source": {"path": "scripts/export_stvg_soft_ep_lr2_v1.py",
                            "sha256": sha256(Path(__file__)), "bytes": Path(__file__).stat().st_size},
        "sealed_bindings": seals["hashes"],
        "gt_exposure_sha256": seals["gt_exposure_sha256"],
        "independent_math_readback_sha256": execution["root_math_readback"]["readback_sha256"],
        "configuration_source_hashes": configuration["provenance"],
    }
    outputs = {
        "ANONYMOUS_ROWS.json": {"schema": "stvg-soft-ep-lr2-anonymous-rows-v1", "status": "complete", "row_count": len(projected_rows), "GT_read": True, "rows": projected_rows},
        "ANONYMOUS_SUMMARIES.json": {"schema": "stvg-soft-ep-lr2-anonymous-summaries-v1", "status": "complete", "summary_count": len(projected_summaries), "GT_read": True, "summary": projected_summaries},
        "ANONYMOUS_PAIRED_CONTRASTS.json": {"schema": "stvg-soft-ep-lr2-anonymous-paired-v1", "status": "complete", "paired_count": len(projected_paired), "GT_read": True, "contrast_definitions": [list(x) for x in INTRATRIAL_CONTRASTS], "paired_contrasts": projected_paired},
        "ANONYMOUS_CROSS_TRIAL_CONTRASTS.json": {"schema": "stvg-soft-ep-lr2-anonymous-cross-trial-v1", "status": "complete", "contrast_count": len(projected_cross), "GT_read": True, "baseline_trial": "mu025_LR1", "contrasts": projected_cross},
        "ANONYMOUS_EXECUTION_READBACK.json": execution,
        "ANONYMOUS_CONFIGURATION.json": configuration,
        "SOURCE_BINDINGS.json": source_bindings,
    }
    doc_text = report_text(summaries, paired, cross, execution)
    protocol = protocol_text()

    targets = [OUT / name for name in (*outputs.keys(), "REPORT_RECEIPT.json", "PUBLIC_MANIFEST.json")]
    need(not any(p.exists() for p in targets) and not DOC.exists() and not PROTOCOL.exists(),
         "LR2 export targets already exist; refusing to overwrite")
    OUT.mkdir(parents=True, exist_ok=False)
    for name, value in outputs.items():
        atomic_json(OUT / name, value)
    (DOC.parent).mkdir(parents=True, exist_ok=True)
    need(not DOC.exists() and not PROTOCOL.exists(), "LR2 docs/protocol already exist")
    DOC.write_text(doc_text, encoding="utf-8")
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(protocol, encoding="utf-8")

    generated_paths = [OUT / name for name in outputs] + [DOC, PROTOCOL]
    generated_hashes = [{"path": relpath(p), "sha256": sha256(p), "bytes": p.stat().st_size}
                        for p in generated_paths]
    receipt = {
        "schema": "stvg-soft-ep-lr2-report-receipt-v1", "status": "complete",
        "scope": "anonymous LR2 export after complete CPU/scoring receipt; no new inference",
        "case_count": 32, "arm_row_count": len(rows), "summary_count": len(summaries),
        "paired_count": len(paired), "cross_trial_count": len(cross),
        "fit_invocations": execution["fit_invocations"],
        "source_capture_calls": execution["source_capture_calls"],
        "actual_full_readout_calls": execution["actual_full_readout_calls"],
        "exact_parent_reuse_arm_rows": execution["exact_parent_reuse_arm_rows"],
        "GT_read": True, "bootstrap_executed": False,
        "cpu_completion_sha256": sha256(BASE / "CPU_COMPLETION.json"),
        "source_manifest": source_rows,
        "outputs": generated_hashes,
    }
    atomic_json(OUT / "REPORT_RECEIPT.json", receipt)
    manifest_files = generated_hashes + [{"path": relpath(OUT / "REPORT_RECEIPT.json"),
                                         "sha256": sha256(OUT / "REPORT_RECEIPT.json"),
                                         "bytes": (OUT / "REPORT_RECEIPT.json").stat().st_size}]
    exporter_row = source_bindings["exporter_source"]
    manifest = {
        "schema": "stvg-soft-ep-lr2-public-manifest-v1", "status": "complete",
        "scope": "LR2 finite matched comparison only; not the separate 128-source search",
        "lr2_execution_sources": source_rows[:len(LR2_SOURCES)],
        "independent_math_audit_source": source_rows[-1],
        "reporting_source": exporter_row,
        "public_files": manifest_files,
        "counts": {"cases": 32, "arm_rows": len(rows), "summary_cells": len(summaries),
                   "within_trial_paired_cells": len(paired), "LR1_cross_trial_cells": len(cross)},
        "GT_read": True, "bootstrap_executed": False,
    }
    atomic_json(OUT / "PUBLIC_MANIFEST.json", manifest)
    return receipt


def refresh_publication() -> dict[str, Any]:
    """Refresh documentation/configuration receipts from already-public scalars only."""
    manifest_path = OUT / "PUBLIC_MANIFEST.json"
    receipt_path = OUT / "REPORT_RECEIPT.json"
    manifest = read_json(manifest_path)
    receipt = read_json(receipt_path)
    need(manifest.get("status") == "complete" and receipt.get("status") == "complete",
         "existing LR2 public export is not complete")
    need(manifest.get("counts") == {"cases": 32, "arm_rows": 320, "summary_cells": 100,
                                     "within_trial_paired_cells": 140, "LR1_cross_trial_cells": 40},
         "existing public manifest counts differ")
    # Verify the old publication byte-for-byte before changing only report/configuration metadata.
    for row in manifest.get("lr2_execution_sources", []):
        path = ROOT / row["path"]
        need(path.is_file() and sha256(path) == row["sha256"] and path.stat().st_size == row["bytes"],
             f"LR2 execution source changed: {row.get('path')}")
    audit = manifest.get("independent_math_audit_source", {})
    audit_path = ROOT / audit.get("path", "")
    need(audit_path.is_file() and sha256(audit_path) == audit.get("sha256")
         and audit_path.stat().st_size == audit.get("bytes"), "independent math audit source changed")
    refreshable = {relpath(DOC)}
    for row in manifest.get("public_files", []):
        path = ROOT / row["path"]
        if row["path"] in refreshable:
            need(path.is_file(), "report document to refresh is missing")
            continue
        need(path.is_file() and sha256(path) == row["sha256"] and path.stat().st_size == row["bytes"],
             f"existing public file differs before refresh: {row.get('path')}")
    # Existing scalar exports are read-only inputs. No POSTSEAL score, prediction, or tensor is opened here.
    rows = read_json(OUT / "ANONYMOUS_ROWS.json")
    summaries = read_json(OUT / "ANONYMOUS_SUMMARIES.json")
    paired = read_json(OUT / "ANONYMOUS_PAIRED_CONTRASTS.json")
    cross = read_json(OUT / "ANONYMOUS_CROSS_TRIAL_CONTRASTS.json")
    execution = read_json(OUT / "ANONYMOUS_EXECUTION_READBACK.json")
    need(rows.get("row_count") == 320 and len(rows.get("rows", [])) == 320
         and summaries.get("summary_count") == 100 and len(summaries.get("summary", [])) == 100
         and paired.get("paired_count") == 140 and len(paired.get("paired_contrasts", [])) == 140
         and cross.get("contrast_count") == 40 and len(cross.get("contrasts", [])) == 40,
         "existing anonymous scalar export counts differ")
    need(execution.get("status") == "complete" and execution.get("bootstrap_executed") is False,
         "existing anonymous execution readback is incomplete")

    configuration = configuration_payload()
    doc_text = report_text(summaries["summary"], paired["paired_contrasts"], cross["contrasts"], execution)
    protocol = protocol_text()
    replace_text(DOC, doc_text)
    replace_text(PROTOCOL, protocol)
    config_path = OUT / "ANONYMOUS_CONFIGURATION.json"
    replace_json(config_path, configuration)

    source_bindings = read_json(OUT / "SOURCE_BINDINGS.json")
    source_bindings["exporter_source"] = {
        "path": "scripts/export_stvg_soft_ep_lr2_v1.py",
        "sha256": sha256(Path(__file__)), "bytes": Path(__file__).stat().st_size,
    }
    source_bindings["configuration_source_hashes"] = configuration["provenance"]
    replace_json(OUT / "SOURCE_BINDINGS.json", source_bindings)

    output_names = (
        "ANONYMOUS_ROWS.json", "ANONYMOUS_SUMMARIES.json", "ANONYMOUS_PAIRED_CONTRASTS.json",
        "ANONYMOUS_CROSS_TRIAL_CONTRASTS.json", "ANONYMOUS_EXECUTION_READBACK.json",
        "ANONYMOUS_CONFIGURATION.json", "SOURCE_BINDINGS.json",
    )
    output_rows = [{"path": relpath(OUT / name), "sha256": sha256(OUT / name),
                    "bytes": (OUT / name).stat().st_size} for name in output_names]
    output_rows.extend({"path": relpath(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                       for path in (DOC, PROTOCOL))
    receipt["outputs"] = output_rows
    receipt["source_manifest"] = manifest["lr2_execution_sources"] + [manifest["independent_math_audit_source"]]
    replace_json(receipt_path, receipt)

    manifest["reporting_source"] = source_bindings["exporter_source"]
    manifest["public_files"] = output_rows + [{"path": relpath(receipt_path),
                                                 "sha256": sha256(receipt_path),
                                                 "bytes": receipt_path.stat().st_size}]
    replace_json(manifest_path, manifest)
    return {"status": "refreshed", "row_count": 320, "summary_count": 100,
            "paired_count": 140, "cross_trial_count": 40,
            "manifest_sha256": sha256(manifest_path)}


def main() -> int:
    if sys.argv[1:] == ["--refresh-publication"]:
        result = refresh_publication()
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if sys.argv[1:]:
        raise SystemExit("usage: export_stvg_soft_ep_lr2_v1.py [--refresh-publication]")
    try:
        receipt = generate()
    except NotReady as exc:
        print(json.dumps({"status": "deferred", "reason": str(exc), "files_written": 0}, ensure_ascii=False))
        return 3
    print(json.dumps({"status": "complete", "case_count": receipt["case_count"],
                      "rows": receipt["arm_row_count"], "summaries": receipt["summary_count"],
                      "paired": receipt["paired_count"], "cross_trial": receipt["cross_trial_count"],
                      "manifest": relpath(OUT / "PUBLIC_MANIFEST.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
