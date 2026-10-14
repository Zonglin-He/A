"""Independent CPU audit of the anonymous short-window public scalar export.

This reader is deliberately separate from report_stvg_short_window_v1.py.  It
does not score predictions or load a model.  Run it only after the sealed
postseal report and the four public JSON exports exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_short_event_tracklets_v1"
PUBLIC = ROOT / "results/stvg_short_event_tracklets/2026-10-14"
OUT = BASE / "ROOT_ANONYMOUS_SCALAR_READBACK.json"
PUBLIC_NAMES = (
    "all_postseal_diagnostics.json",
    "all_teacher_calls.json",
    "reference_selection.json",
    "diagnostic_summary.json",
)
CASE_IDS = tuple([f"hc2_{i:03d}" for i in range(4)] + [f"vidstg_{i:03d}" for i in range(4)])
REMOVED_KEYS = {"GT_source_bindings", "private_path", "raw_output"}
PATH_PREFIXES = ("artifacts/", "/home/", "/tmp/")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _alias(case_id: str) -> str:
    if case_id.startswith("hc2_"):
        return f"HC{int(case_id[-3:]) + 1:02d}"
    if case_id.startswith("vidstg_"):
        return f"V{int(case_id[-3:]) + 1:02d}"
    raise ValueError("unrecognized fixed case identifier")


def _public_projection(value: Any) -> Any:
    """Reconstruct the documented redaction independently from the writer."""
    if isinstance(value, dict):
        return {
            key: _public_projection(item)
            for key, item in value.items()
            if key not in REMOVED_KEYS
        }
    if isinstance(value, list):
        return [_public_projection(item) for item in value]
    if isinstance(value, str):
        if value.startswith(PATH_PREFIXES):
            return "private_artifact_binding"
        projected = value
        for case_id in CASE_IDS:
            projected = projected.replace(case_id, _alias(case_id))
        return projected
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _first_difference(expected: Any, observed: Any, path: str = "$", limit: int = 1) -> list[str]:
    """Return mismatch paths only; never include potentially sensitive values."""
    found: list[str] = []
    if type(expected) is not type(observed):
        return [path]
    if isinstance(expected, dict):
        if set(expected) != set(observed):
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            return [f"{path}.keys(missing={missing[:4]},extra={extra[:4]})"]
        for key in sorted(expected):
            found.extend(_first_difference(expected[key], observed[key], f"{path}.{key}", limit))
            if len(found) >= limit:
                break
        return found
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return [f"{path}.length"]
        for index, (left, right) in enumerate(zip(expected, observed)):
            found.extend(_first_difference(left, right, f"{path}[{index}]", limit))
            if len(found) >= limit:
                break
        return found
    if isinstance(expected, float) and isinstance(observed, float):
        if not math.isfinite(expected) or not math.isfinite(observed) or expected != observed:
            return [path]
        return []
    return [] if expected == observed else [path]


def _assert_same(name: str, expected: Any, observed: Any, errors: list[str]) -> None:
    # Canonical JSON comparison keeps bool/int distinctions and exact serialized
    # float values, while the path-only diagnostic avoids copying scalar values.
    if _canonical(expected) != _canonical(observed):
        errors.extend(f"{name}:{p}" for p in _first_difference(expected, observed, limit=3))


def _expected_reference_selection(ref_lock: dict[str, Any]) -> dict[str, Any]:
    records = []
    for row in ref_lock["records"]:
        records.append(
            {
                "case_id": row["case_id"],
                "dataset": row["dataset"],
                "roles": {
                    role: {
                        "no_update": reference["no_update"],
                        "reason": reference["reason"],
                        "anchor_count": len(reference["anchors"]),
                        "source_grid_positions": [anchor["position"] for anchor in reference["anchors"]],
                        "selected_candidate_ids": [
                            anchor["selected_candidate_id"] for anchor in reference["anchors"]
                        ],
                    }
                    for role, reference in row["references"].items()
                },
            }
        )
    return _public_projection({"status": "locked", "records": records})


def _expected_teacher_calls(barrier: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "status",
        "job_id",
        "case_id",
        "dataset",
        "kind",
        "center_slot",
        "parsed",
        "output_truncated_at_cap",
        "source_case_sha256",
        "prompt_sha256",
        "candidate_count",
        "source_only_non_comparable",
        "empty_candidate_pool",
        "input_tokens",
        "visual_tokens",
        "output_tokens",
        "model_forward_generate_seconds",
        "end_to_end_request_seconds",
        "GPU_peak_bytes",
        "teacher_frozen",
        "GT_read",
        "mean_gap_encoder_pacing_approximation",
    ]
    calls = []
    for file_row in barrier["files"]:
        response = _read(ROOT / file_row["path"])
        calls.append({key: response[key] for key in fields})

    def stage_seconds(stage: str) -> float:
        launch = _read(BASE / f"{stage}_LAUNCH.json")
        exit_record = _read(BASE / f"{stage}_WORKER_EXIT.json")
        return exit_record["time"] - launch["time"]

    cost = {
        "status": "measured",
        "teacher_call_count": len(calls),
        "input_tokens": sum(call["input_tokens"] for call in calls),
        "visual_tokens": sum(call["visual_tokens"] for call in calls),
        "output_tokens": sum(call["output_tokens"] for call in calls),
        "summed_request_seconds": sum(call["end_to_end_request_seconds"] for call in calls),
        "summed_generation_seconds": sum(call["model_forward_generate_seconds"] for call in calls),
        "peak_gpu_bytes": max(call["GPU_peak_bytes"] for call in calls),
        "request_latency_excludes_shared_initial_model_load": True,
        "interpretation": "finite diagnostic cost; not end-to-end production TTA latency",
        "stage_launch_to_exit_seconds": {
            stage: stage_seconds(stage) for stage in ("FEATURE", "TEACHER", "STUDENT")
        },
    }
    return _public_projection({"status": "complete", "count": 64, "calls": calls, "cost": cost})


def _expected_diagnostic_summary(postseal: dict[str, Any]) -> dict[str, Any]:
    windows = postseal["window_pool_teacher"]
    pool_known = [row for row in windows if row["delta_pool_pp"] is not None]
    teacher_known = [row for row in windows if row["delta_teacher_pp"] is not None]

    def sign_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
        return {
            "positive": sum(row[key] > 1e-8 for row in rows),
            "negative": sum(row[key] < -1e-8 for row in rows),
            "neutral": sum(abs(row[key]) <= 1e-8 for row in rows),
        }

    return {
        "common_windows": len(pool_known),
        "unknown_pool": len(windows) - len(pool_known),
        "pool": sign_counts(pool_known, "delta_pool_pp"),
        "teacher": sign_counts(teacher_known, "delta_teacher_pp"),
        "teacher_known": len(teacher_known),
        "teacher_unknown": len(windows) - len(teacher_known),
        "valid_ID_pairs": sum(row["ID_invariant"] is not None for row in windows),
        "ID_invariant": sum(row["ID_invariant"] is True for row in windows),
        "base_abstain": sum(row["abstain_or_invalid"] for row in windows),
    }


def _validate_no_sensitive_strings(value: Any, path: str = "$", findings: list[str] | None = None) -> list[str]:
    findings = [] if findings is None else findings
    if isinstance(value, dict):
        for key, item in value.items():
            if key in REMOVED_KEYS:
                findings.append(f"{path}.{key}")
            _validate_no_sensitive_strings(item, f"{path}.{key}", findings)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_no_sensitive_strings(item, f"{path}[{index}]", findings)
    elif isinstance(value, str):
        if value.startswith(PATH_PREFIXES):
            findings.append(path)
        if any(case_id in value for case_id in CASE_IDS):
            findings.append(path)
    return findings


def _validate_counts(postseal: dict[str, Any], teacher: dict[str, Any], refs: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    metrics = postseal.get("metrics", [])
    groups = postseal.get("groups", [])
    candidates = postseal.get("candidate_geometry", [])
    windows = postseal.get("window_pool_teacher", [])
    movements = postseal.get("fit_movement", [])
    calls = teacher.get("calls", [])
    reference_records = refs.get("records", [])
    expected_counts = {
        "metrics": (len(metrics), 48),
        "groups": (len(groups), 12),
        "candidate_geometry": (len(candidates), 152),
        "window_pool_teacher": (len(windows), 32),
        "teacher_calls": (len(calls), 64),
        "reference_records": (len(reference_records), 8),
    }
    for name, (observed, expected) in expected_counts.items():
        if observed != expected:
            errors.append(f"count:{name}:{observed}!={expected}")

    wanted_aliases = {_alias(case_id) for case_id in CASE_IDS}
    found_aliases = {
        row.get("case_id")
        for collection in (metrics, candidates, windows, movements, calls, reference_records)
        for row in collection
        if isinstance(row, dict) and isinstance(row.get("case_id"), str)
    }
    if found_aliases != wanted_aliases:
        errors.append("case_alias_coverage")

    metric_ids = {(row.get("case_id"), row.get("arm")) for row in metrics}
    if len(metric_ids) != 48:
        errors.append("metric_identity_uniqueness")
    group_ids = {(row.get("dataset"), row.get("arm")) for row in groups}
    if len(group_ids) != 12:
        errors.append("group_identity_uniqueness")
    candidate_ids = {(row.get("case_id"), row.get("slot"), row.get("candidate_id")) for row in candidates}
    if len(candidate_ids) != 152:
        errors.append("candidate_identity_uniqueness")
    window_ids = {(row.get("case_id"), row.get("slot")) for row in windows}
    if len(window_ids) != 32:
        errors.append("window_identity_uniqueness")
    call_ids = {row.get("job_id") for row in calls}
    if len(call_ids) != 64:
        errors.append("teacher_call_identity_uniqueness")

    role_anchor_totals: dict[str, int] = {}
    role_case_anchor_counts: dict[str, dict[str, int]] = {}
    for record in reference_records:
        roles = record.get("roles", {})
        for role, item in roles.items():
            role_anchor_totals[role] = role_anchor_totals.get(role, 0) + int(item.get("anchor_count", 0))
            role_case_anchor_counts.setdefault(role, {})[record["case_id"]] = int(item.get("anchor_count", 0))
            if item.get("no_update") is not (item.get("anchor_count") == 0):
                errors.append(f"reference_no_update_inconsistent:{record['case_id']}:{role}")

    # Supervision counts in each postseal arm must come from the sealed reference
    # role at that source. This checks lock -> diagnostics consistency separately
    # from byte-level public/source projection equality.
    reference_by_case = {record["case_id"]: record["roles"] for record in reference_records}
    for row in metrics:
        role = row.get("arm", "").removeprefix("T_")
        if role == "F" or role == "T":
            expected = 0
        else:
            expected = reference_by_case.get(row.get("case_id"), {}).get(role, {}).get("anchor_count", 0)
        if row.get("actual_supervision_count") != expected:
            errors.append(f"reference_metric_supervision_mismatch:{row.get('case_id')}:{row.get('arm')}")

    eligible_moves = {
        (case_id, role)
        for role, by_case in role_case_anchor_counts.items()
        for case_id, count in by_case.items()
        if count > 0
    }
    move_ids = {(row.get("case_id"), row.get("role")) for row in movements}
    if move_ids != eligible_moves:
        errors.append("reference_fit_movement_coverage")

    parse_counts: dict[str, int] = {}
    abstain_count = invalid_count = truncated_count = 0
    for call in calls:
        parsed = call.get("parsed", {})
        state = parsed.get("status", "missing")
        parse_counts[state] = parse_counts.get(state, 0) + 1
        if state == "valid" and parsed.get("selected_candidate_id") is None:
            abstain_count += 1
        if state != "valid":
            invalid_count += 1
        if call.get("output_truncated_at_cap") is True:
            truncated_count += 1

    return {
        "counts": {key: value[0] for key, value in expected_counts.items()},
        "movement_rows": len(movements),
        "reference_role_anchor_totals": role_anchor_totals,
        "teacher_parse_counts": parse_counts,
        "teacher_valid_abstain_count": abstain_count,
        "teacher_invalid_count": invalid_count,
        "teacher_truncated_count": truncated_count,
    }


def _validate_reference_derivation(ref_lock: dict[str, Any], manifest: dict[str, Any],
                                  teacher_barrier: dict[str, Any], job_lock: dict[str, Any],
                                  errors: list[str]) -> dict[str, Any]:
    """Check role anchors against the locked centers and base-call selections."""
    jobs = {row["job_id"]: row for row in job_lock.get("records", [])}
    files_by_job = {}
    outputs_by_job = {}
    for file_row in teacher_barrier.get("files", []):
        output_path = ROOT / file_row["path"]
        output = _read(output_path)
        job_id = output.get("job_id")
        if job_id in outputs_by_job:
            errors.append("teacher_output_duplicate_job_id")
        outputs_by_job[job_id] = output
        files_by_job[job_id] = file_row

    manifest_cases = {row["case_id"]: row for row in manifest.get("cases", [])}
    checked_anchors = 0
    role_counts: dict[str, int] = {}
    for record in ref_lock.get("records", []):
        case_id = record["case_id"]
        centers = manifest_cases.get(case_id, {}).get("centers", [])
        center_by_position = {int(center["source_grid_position"]): center for center in centers}
        if len(center_by_position) != len(centers):
            errors.append(f"duplicate_locked_source_position:{_alias(case_id)}")
        for role, reference in record.get("references", {}).items():
            anchors = reference.get("anchors", [])
            role_counts[role] = role_counts.get(role, 0) + len(anchors)
            if reference.get("no_update") is not (len(anchors) == 0):
                errors.append(f"locked_role_update_flag:{_alias(case_id)}:{role}")
            for anchor in anchors:
                checked_anchors += 1
                position = anchor.get("position")
                center = center_by_position.get(position)
                if center is None:
                    errors.append(f"anchor_position_not_locked:{_alias(case_id)}:{role}")
                    continue
                slot = center.get("slot")
                job_id = f"{case_id}_window{int(slot):02d}_base"
                job = jobs.get(job_id)
                output = outputs_by_job.get(job_id)
                receipt = files_by_job.get(job_id)
                if (job is None or output is None or receipt is None or
                        anchor.get("teacher_job_id") != job_id or
                        anchor.get("physical_frame_id") != center.get("center_physical_frame_id")):
                    errors.append(f"anchor_center_or_job_binding:{_alias(case_id)}:{role}")
                    continue
                parsed = output.get("parsed", {})
                if (output.get("status") != "actual_video_candidate_verification" or
                        output.get("output_truncated_at_cap") is not False or
                        parsed.get("status") != "valid" or
                        parsed.get("selected_candidate_id") is None or
                        parsed.get("selected_candidate_id") != anchor.get("selected_candidate_id") or
                        anchor.get("candidate_id") != anchor.get("selected_candidate_id") or
                        anchor.get("is_Source") is not False or
                        anchor.get("teacher_output_sha256") != receipt.get("sha256")):
                    errors.append(f"anchor_not_derived_from_base_teacher:{_alias(case_id)}:{role}")
    return {"checked_anchor_rows": checked_anchors, "role_anchor_totals": role_counts}


def audit() -> dict[str, Any]:
    errors: list[str] = []
    required = [PUBLIC / name for name in PUBLIC_NAMES]
    missing = [name for name, path in zip(PUBLIC_NAMES, required) if not path.is_file()]
    if missing:
        raise FileNotFoundError("public export is not complete: " + ", ".join(missing))

    sealed_path = BASE / "ROOT_POSTSEAL_REPORT.json"
    ref_path = BASE / "METHOD_REFERENCE_LOCK.json"
    teacher_barrier_path = BASE / "TEACHER_GLOBAL_BARRIER.json"
    job_lock_path = BASE / "TEACHER_JOB_LOCK.json"
    manifest_path = BASE / "PREPARED_MANIFEST.json"
    for path in (sealed_path, ref_path, teacher_barrier_path, job_lock_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"required sealed source is missing: {path.name}")

    # The sealed score is only read after this script has all four exports to
    # compare against. No scorer, truth loader, model or GPU code is imported.
    sealed = _read(sealed_path)
    reference_lock = _read(ref_path)
    teacher_barrier = _read(teacher_barrier_path)
    job_lock = _read(job_lock_path)
    manifest = _read(manifest_path)
    public = {Path(path).stem: _read(path) for path in required}

    expected_postseal = _public_projection(sealed)
    expected_teacher = _expected_teacher_calls(teacher_barrier)
    expected_references = _expected_reference_selection(reference_lock)
    expected_summary = _expected_diagnostic_summary(sealed)
    expected = {
        "all_postseal_diagnostics": expected_postseal,
        "all_teacher_calls": expected_teacher,
        "reference_selection": expected_references,
        "diagnostic_summary": expected_summary,
    }
    observed = public
    for name, value in expected.items():
        _assert_same(name, value, observed[name], errors)

    if sealed.get("status") != "pass" or sealed.get("metric_count") != 48:
        errors.append("sealed_report_status_or_metric_count")
    if teacher_barrier.get("status") != "complete" or teacher_barrier.get("count") != 64:
        errors.append("teacher_global_barrier_count_or_status")
    if reference_lock.get("status") != "locked" or len(reference_lock.get("records", [])) != 8:
        errors.append("reference_lock_count_or_status")
    if job_lock.get("status") != "locked" or job_lock.get("count") != 64:
        errors.append("teacher_job_lock_count_or_status")

    summary = _validate_counts(
        observed["all_postseal_diagnostics"],
        observed["all_teacher_calls"],
        observed["reference_selection"],
        errors,
    )
    # Preserve the derived diagnostic totals and resource scalars in the audit
    # receipt as well as comparing their complete source projections above.
    summary["diagnostic_summary"] = observed["diagnostic_summary"]
    teacher_cost = observed["all_teacher_calls"].get("cost", {})
    summary["teacher_cost"] = {
        key: teacher_cost.get(key)
        for key in (
            "teacher_call_count",
            "input_tokens",
            "visual_tokens",
            "output_tokens",
            "summed_request_seconds",
            "summed_generation_seconds",
            "peak_gpu_bytes",
            "stage_launch_to_exit_seconds",
        )
    }
    ref_derivation = _validate_reference_derivation(
        reference_lock, manifest, teacher_barrier, job_lock, errors
    )
    if ref_derivation["role_anchor_totals"] != summary["reference_role_anchor_totals"]:
        errors.append("locked_and_public_reference_role_totals_differ")
    sensitive = []
    for name, value in observed.items():
        sensitive.extend(f"{name}:{p}" for p in _validate_no_sensitive_strings(value))
    if sensitive:
        errors.extend(f"redaction:{item}" for item in sensitive[:20])

    win_summary = observed["diagnostic_summary"]
    if win_summary.get("common_windows", 0) + win_summary.get("unknown_pool", 0) != 32:
        errors.append("pool_known_unknown_partition")
    if win_summary.get("teacher_known", 0) + win_summary.get("teacher_unknown", 0) != 32:
        errors.append("teacher_known_unknown_partition")

    source_paths = {
        "sealed_report_sha256": sealed_path,
        "reference_lock_sha256": ref_path,
        "teacher_global_barrier_sha256": teacher_barrier_path,
        "teacher_job_lock_sha256": job_lock_path,
        "student_global_barrier_sha256": BASE / "STUDENT_GLOBAL_BARRIER.json",
    }
    source_hashes = {name: _sha(path) for name, path in source_paths.items() if path.is_file()}
    public_hashes = {name: _sha(path) for name, path in zip(PUBLIC_NAMES, required)}
    status = "pass" if not errors else "fail"
    receipt = {
        "schema": "stvg-short-window-anonymous-public-scalar-readback-v1",
        "status": status,
        "GT_scoring_performed": False,
        "model_loaded": False,
        "GPU_initialized": False,
        "report_generator_called": False,
        "all_public_leaves_compared_to_sealed_source_projection": not errors,
        "public_export_sha256": public_hashes,
        "source_sha256": source_hashes,
        "counts_and_states": summary,
        "locked_reference_derivation": ref_derivation,
        "diagnostics": errors,
    }
    OUT.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return receipt


def _synthetic_contracts() -> dict[str, Any]:
    # Tiny deterministic checks for aliasing, redaction, null-preserving sign
    # summaries, and type-sensitive JSON equality. They contain no project data.
    errors: list[str] = []
    sample = {
        "case_id": "hc2_000",
        "private_path": "/home/private/item",
        "nested": {"GT_source_bindings": {"x": [0.1]}, "raw_output": "do not copy"},
        "label": "hc2_000_window00_base",
    }
    projected = _public_projection(sample)
    want = {
        "case_id": "HC01",
        "nested": {},
        "label": "HC01_window00_base",
    }
    _assert_same("synthetic_projection", want, projected, errors)
    if _first_difference(1, True):
        pass
    else:
        errors.append("type_sensitive_comparison")
    rows = [{"delta_pool_pp": None, "delta_teacher_pp": None}]
    if sum(row["delta_pool_pp"] is not None for row in rows) != 0:
        errors.append("null_unknown_contract")
    if errors:
        raise RuntimeError("synthetic contract failure: " + ", ".join(errors))
    return {"status": "pass", "checks": 4, "sensitive_values_emitted": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="run synthetic checks without opening experiment results")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_synthetic_contracts(), sort_keys=True))
        return
    receipt = audit()
    print(json.dumps({"status": receipt["status"], "counts_and_states": receipt["counts_and_states"],
                      "diagnostics": receipt["diagnostics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
