"""Independent scalar aggregation readback for the sealed CRC P0 outputs.

This reader consumes only the four saved postseal JSON artifacts named below.
It never opens predictions, annotations, source cases, video, model files, or a
GT loader. It verifies stored aggregates and writes only a scalar receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_contrastive_correction_v1"
DEFAULT_OUTPUT = BASE / "ROOT_POSTSEAL_SCALAR_READBACK.json"
METRICS = (
    "m_vIoU",
    "m_tIoU",
    "m_sIoU_dense_GT",
    "m_sIoU_supervised_GT",
    "m_sIoU_unsupervised_GT",
)
ARMS = ("F", "T", "S_historical", "TS_historical", "S_Pair", "TS_Pair")
DOMAINS = ("hc2", "vidstg")
EXPECTED_ALIASES = tuple([f"HC{i:02d}" for i in range(1, 17)] + [f"V{i:02d}" for i in range(1, 17)])
INPUT_NAMES = (
    "POSTSEAL_ROWS.json",
    "POSTSEAL_SUMMARY.json",
    "PARENT_PAIRED_DELTAS.json",
    "POSTSEAL_REFERENCE_DIAGNOSTICS.json",
)
FLOAT_ATOL = 1e-12
PAIR_SIGN_EPS = 1e-10
REFERENCE_SIGN_EPS = 1e-8


class AuditError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_input(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = BASE / name
    need(path.is_file(), f"required sealed scalar input is missing: {path.relative_to(ROOT)}")
    resolved = path.resolve()
    need(resolved.is_relative_to(BASE.resolve()), f"input escapes CRC scope: {name}")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except Exception as exc:
        raise AuditError(f"invalid JSON input {name}: {type(exc).__name__}") from exc
    need(isinstance(value, dict), f"input root must be an object: {name}")
    pin = {"path": str(path.relative_to(ROOT)), "sha256": sha_bytes(payload), "bytes": len(payload)}
    return value, pin


def finite_number(value: Any, context: str) -> float | None:
    if value is None:
        return None
    need(isinstance(value, (int, float)) and not isinstance(value, bool), f"non-numeric scalar at {context}")
    result = float(value)
    need(math.isfinite(result), f"non-finite scalar at {context}")
    return result


def close(actual: Any, expected: Any, context: str, atol: float = FLOAT_ATOL) -> float:
    if expected is None:
        need(actual is None, f"{context}: expected null, found a value")
        return 0.0
    av = finite_number(actual, context)
    need(av is not None, f"{context}: expected numeric scalar")
    error = abs(av - float(expected))
    need(error <= atol, f"{context}: scalar mismatch abs_error={error:.17g}")
    return error


def expected_pair_specs(arms: tuple[str, ...] = ARMS) -> tuple[tuple[str, str], ...]:
    specs = [(arm, "F") for arm in arms if arm != "F"]
    specs += [("TS_Pair", "T"), ("TS_Pair", "TS_historical"), ("S_Pair", "S_historical")]
    return tuple(specs)


def validate_rows(rows: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    need(isinstance(rows, list) and len(rows) == 32 * 6, "POSTSEAL_ROWS must contain exactly 192 rows")
    aliases_by_dataset: dict[str, set[str]] = {d: set() for d in DOMAINS}
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for i, row in enumerate(rows):
        need(isinstance(row, dict), f"row {i} must be an object")
        case_id, dataset, arm = row.get("case_id"), row.get("dataset"), row.get("arm")
        need(isinstance(case_id, str) and dataset in DOMAINS and arm in ARMS, f"bad row identity at {i}")
        pattern = r"HC\d{2}" if dataset == "hc2" else r"V\d{2}"
        need(re.fullmatch(pattern, case_id) is not None, f"case alias does not match its domain: {case_id}")
        key = (case_id, arm)
        need(key not in lookup, f"duplicate case/arm row: {key}")
        metrics = row.get("metrics")
        need(isinstance(metrics, dict) and set(metrics) == set(METRICS), f"metric schema mismatch for {key}")
        for metric in METRICS:
            finite_number(metrics[metric], f"{key}/{metric}")
        lookup[key] = row
        aliases_by_dataset[dataset].add(case_id)
    aliases = {"hc2": set(EXPECTED_ALIASES[:16]), "vidstg": set(EXPECTED_ALIASES[16:])}
    for ds in DOMAINS:
        need(aliases_by_dataset[ds] == aliases[ds], f"fixed alias cohort mismatch for {ds}")
        for cid in aliases[ds]:
            need(all((cid, arm) in lookup for arm in ARMS), f"missing arm row for {cid}")
    return lookup, {ds: "|".join(sorted(aliases[ds])) for ds in DOMAINS}


def build_metric_aggregates(lookup: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for ds in DOMAINS:
        ids = EXPECTED_ALIASES[:16] if ds == "hc2" else EXPECTED_ALIASES[16:]
        for arm in ARMS:
            for metric in METRICS:
                vals = [finite_number(lookup[cid, arm]["metrics"][metric], f"{cid}/{arm}/{metric}") for cid in ids]
                known = [v for v in vals if v is not None]
                records.append({
                    "dataset": ds,
                    "arm": arm,
                    "metric": metric,
                    "count": len(known),
                    "unknown_count": len(vals) - len(known),
                    "mean": math.fsum(known) / len(known) if known else None,
                })
    return records


def compare_metric_summary(actual_summary: Any, expected: list[dict[str, Any]]) -> tuple[int, float]:
    need(isinstance(actual_summary, list), "POSTSEAL_SUMMARY.summary must be an array")
    actual = {}
    for row in actual_summary:
        need(isinstance(row, dict), "summary row must be an object")
        key = (row.get("dataset"), row.get("arm"), row.get("metric"))
        need(key not in actual, f"duplicate summary scalar: {key}")
        actual[key] = row
    expected_keys = {(r["dataset"], r["arm"], r["metric"]) for r in expected}
    need(set(actual) == expected_keys, "summary keyset does not cover the complete 2x6x5 table")
    max_error = 0.0
    for exp in expected:
        key = (exp["dataset"], exp["arm"], exp["metric"])
        row = actual[key]
        need(row.get("count") == exp["count"], f"summary count mismatch: {key}")
        max_error = max(max_error, close(row.get("mean"), exp["mean"], f"summary mean {key}"))
    return len(actual), max_error


def build_paired_aggregates(lookup: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for ds in DOMAINS:
        ids = EXPECTED_ALIASES[:16] if ds == "hc2" else EXPECTED_ALIASES[16:]
        for left, right in expected_pair_specs():
            deltas: dict[str, float | None] = {}
            for cid in ids:
                lv = finite_number(lookup[cid, left]["metrics"]["m_vIoU"], f"{cid}/{left}/m_vIoU")
                rv = finite_number(lookup[cid, right]["metrics"]["m_vIoU"], f"{cid}/{right}/m_vIoU")
                deltas[cid] = (lv - rv) * 100.0 if lv is not None and rv is not None else None
            known = [v for v in deltas.values() if v is not None]
            records.append({
                "dataset": ds,
                "left": left,
                "right": right,
                "metric": "m_vIoU",
                "count": len(known),
                "unknown_sources": len(deltas) - len(known),
                "mean_delta_pp": math.fsum(known) / len(known) if known else None,
                "positive_sources": sum(v > PAIR_SIGN_EPS for v in known),
                "negative_sources": sum(v < -PAIR_SIGN_EPS for v in known),
                "neutral_sources": sum(abs(v) <= PAIR_SIGN_EPS for v in known),
                "worst_delta_pp": min(known) if known else None,
                "best_delta_pp": max(known) if known else None,
                "per_source_delta_pp": deltas,
            })
    return records


def compare_paired(actual_records: Any, expected: list[dict[str, Any]]) -> tuple[int, float, dict[str, int]]:
    need(isinstance(actual_records, list), "PARENT_PAIRED_DELTAS.records must be an array")
    actual = {}
    for row in actual_records:
        need(isinstance(row, dict), "paired record must be an object")
        key = (row.get("dataset"), row.get("left"), row.get("right"), row.get("metric"))
        need(key not in actual, f"duplicate paired record: {key}")
        actual[key] = row
    expected_keys = {(r["dataset"], r["left"], r["right"], r["metric"]) for r in expected}
    need(set(actual) == expected_keys, "paired record keyset does not match all fixed comparisons")
    max_error = 0.0
    tails = {"positive": 0, "negative": 0, "neutral": 0, "unknown": 0}
    for exp in expected:
        key = (exp["dataset"], exp["left"], exp["right"], exp["metric"])
        row = actual[key]
        got_map = row.get("per_source_delta_pp")
        need(isinstance(got_map, dict), f"paired per-source map missing: {key}")
        need(set(got_map) == set(exp["per_source_delta_pp"]), f"paired aliases mismatch: {key}")
        for cid, want in exp["per_source_delta_pp"].items():
            max_error = max(max_error, close(got_map[cid], want, f"paired delta {key}/{cid}"))
        for field in ("mean_delta_pp", "worst_delta_pp", "best_delta_pp"):
            max_error = max(max_error, close(row.get(field), exp[field], f"paired {field} {key}"))
        for field in ("positive_sources", "negative_sources", "neutral_sources"):
            need(row.get(field) == exp[field], f"paired sign-tail mismatch in {field}: {key}")
        if "unknown_sources" in row:
            need(row["unknown_sources"] == exp["unknown_sources"], f"paired unknown count mismatch: {key}")
        tails["positive"] += exp["positive_sources"]
        tails["negative"] += exp["negative_sources"]
        tails["neutral"] += exp["neutral_sources"]
        tails["unknown"] += exp["unknown_sources"]
    return len(actual), max_error, tails


def build_reference_aggregates(rows: Any, aliases_by_dataset: dict[str, str]) -> list[dict[str, Any]]:
    need(isinstance(rows, list), "POSTSEAL_REFERENCE_DIAGNOSTICS.rows must be an array")
    seen = set()
    grouped: dict[tuple[str, bool], list[float | None]] = {
        (ds, retained): [] for ds in DOMAINS for retained in (False, True)
    }
    valid_aliases = {ds: set(aliases_by_dataset[ds].split("|")) for ds in DOMAINS}
    for i, row in enumerate(rows):
        need(isinstance(row, dict), f"reference diagnostic row {i} must be an object")
        ds, cid, retained = row.get("dataset"), row.get("case_id"), row.get("retained")
        need(ds in DOMAINS and cid in valid_aliases[ds], f"unknown alias/domain in reference diagnostic row {i}")
        need(isinstance(retained, bool), f"retained flag must be boolean at reference row {i}")
        pos = row.get("center_position")
        need(isinstance(pos, int) and not isinstance(pos, bool), f"center position missing at reference row {i}")
        key = (ds, cid, pos, retained)
        need(key not in seen, f"duplicate reference diagnostic observation: {key}")
        seen.add(key)
        observed = row.get("gt_observed")
        need(isinstance(observed, bool), f"gt_observed must be boolean at reference row {i}")
        value = finite_number(row.get("reference_delta_iou_pp"), f"reference delta row {i}")
        need((value is not None) == observed, f"reference diagnostic unknown/value mismatch at row {i}")
        grouped[(ds, retained)].append(value)
    need(len(rows) == 49, "reference diagnostic row count must match the locked 49 anchor observations")
    records = []
    for ds in DOMAINS:
        for retained in (False, True):
            values = grouped[(ds, retained)]
            known = [v for v in values if v is not None]
            records.append({
                "dataset": ds,
                "retained": retained,
                "count": len(values),
                "observed_count": len(known),
                "unknown_count": len(values) - len(known),
                "positive": sum(v > REFERENCE_SIGN_EPS for v in known),
                "negative": sum(v < -REFERENCE_SIGN_EPS for v in known),
                "neutral": sum(abs(v) <= REFERENCE_SIGN_EPS for v in known),
                "mean_delta_pp": math.fsum(known) / len(known) if known else None,
            })
    return records


def compare_reference_summary(actual_summary: Any, expected: list[dict[str, Any]]) -> tuple[int, float]:
    need(isinstance(actual_summary, list), "reference diagnostic summary must be an array")
    key_fields = ("dataset", "retained")
    actual = {}
    for row in actual_summary:
        key = tuple(row.get(k) for k in key_fields)
        need(key not in actual, f"duplicate reference summary group: {key}")
        actual[key] = row
    expected_by = {(r["dataset"], r["retained"]): r for r in expected}
    need(set(actual) == set(expected_by), "reference summary must contain all domain x retained groups")
    max_error = 0.0
    for key, exp in expected_by.items():
        row = actual[key]
        for field in ("count", "observed_count", "unknown_count", "positive", "negative", "neutral"):
            need(row.get(field) == exp[field], f"reference summary mismatch {field}: {key}")
        max_error = max(max_error, close(row.get("mean_delta_pp"), exp["mean_delta_pp"], f"reference mean {key}"))
    return len(actual), max_error


def audit(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    need(output.resolve().is_relative_to(BASE.resolve()), "output must remain inside the CRC artifact scope")
    data: dict[str, dict[str, Any]] = {}
    pins = []
    for name in INPUT_NAMES:
        data[name], pin = load_input(name)
        pins.append(pin)
        need(data[name].get("status") == "complete", f"postseal input not complete: {name}")
        need(data[name].get("GT_read") is True, f"expected prior postseal scoring marker in {name}")

    rows_obj = data["POSTSEAL_ROWS.json"]
    summary_obj = data["POSTSEAL_SUMMARY.json"]
    paired_obj = data["PARENT_PAIRED_DELTAS.json"]
    diag_obj = data["POSTSEAL_REFERENCE_DIAGNOSTICS.json"]
    need(paired_obj.get("bootstrap_executed") is False, "unexpected paired bootstrap marker")

    row_lookup, aliases_by_dataset = validate_rows(rows_obj.get("rows"))
    metric_expected = build_metric_aggregates(row_lookup)
    metric_count, metric_error = compare_metric_summary(summary_obj.get("summary"), metric_expected)
    paired_expected = build_paired_aggregates(row_lookup)
    paired_count, paired_error, paired_tails = compare_paired(paired_obj.get("records"), paired_expected)
    reference_expected = build_reference_aggregates(diag_obj.get("rows"), aliases_by_dataset)
    reference_count, reference_error = compare_reference_summary(diag_obj.get("summary"), reference_expected)

    report = {
        "schema": "stvg-crc-postseal-scalar-readback-v1",
        "status": "pass",
        "case_count": 32,
        "case_arm_row_count": len(rows_obj["rows"]),
        "arms": list(ARMS),
        "domains": list(DOMAINS),
        "aliases_by_domain": aliases_by_dataset,
        "metric_summary_scalar_count": metric_count,
        "metric_aggregates": metric_expected,
        "metric_unknown_counts": [
            {k: r[k] for k in ("dataset", "arm", "metric", "unknown_count")}
            for r in metric_expected if r["unknown_count"]
        ],
        "paired_comparison_count": paired_count,
        "paired_tail_counts": paired_tails,
        "paired_aggregates": paired_expected,
        "reference_diagnostic_row_count": len(diag_obj["rows"]),
        "reference_group_count": reference_count,
        "reference_aggregates": reference_expected,
        "max_abs_errors": {
            "metric_summary": metric_error,
            "paired_deltas": paired_error,
            "reference_diagnostics": reference_error,
        },
        "input_files": pins,
        "GT_read": False,
        "prior_GT_derived_scalars_read": True,
        "new_scoring": False,
        "model_or_video_read": False,
        "scope": "independent reaggregation of saved CRC postseal scalar rows, summary, paired tails, and anchor-reference diagnostic groups only",
    }
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing scalar receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, output)
    return report


def _synthetic_rows() -> list[dict[str, Any]]:
    rows = []
    offsets = {"F": 0.0, "T": .005, "S_historical": -.003, "TS_historical": .001, "S_Pair": .007, "TS_Pair": -.002}
    for ds, prefix in (("hc2", "HC"), ("vidstg", "V")):
        for i in range(1, 17):
            cid = f"{prefix}{i:02d}"
            base = .2 + i * .005
            for arm_index, arm in enumerate(ARMS):
                vals = {
                    "m_vIoU": min(.95, base + offsets[arm]),
                    "m_tIoU": min(.95, .3 + i * .003 + offsets[arm] / 2),
                    "m_sIoU_dense_GT": min(.95, .4 + i * .002 + offsets[arm]),
                    "m_sIoU_supervised_GT": min(.95, .5 + i * .001 + offsets[arm]),
                    "m_sIoU_unsupervised_GT": min(.95, .6 + i * .001 + offsets[arm]),
                }
                if cid == "HC01" and arm == "F":
                    vals["m_sIoU_unsupervised_GT"] = None
                if cid == "HC02" and arm == "S_Pair":
                    vals["m_vIoU"] = None
                rows.append({"case_id": cid, "dataset": ds, "arm": arm, "metrics": vals})
    return rows


def _synthetic_reference_rows() -> list[dict[str, Any]]:
    rows = []
    pos = 0
    for ds, prefix in (("hc2", "HC"), ("vidstg", "V")):
        for retained in (False, True):
            for cid_index, delta, observed in ((1, .2, True), (2, -.1, True), (3, 0.0, True), (4, None, False)):
                pos += 1
                rows.append({
                    "case_id": f"{prefix}{cid_index:02d}",
                    "dataset": ds,
                    "center_position": pos,
                    "retained": retained,
                    "gt_observed": observed,
                    "reference_delta_iou_pp": delta,
                })
    # Fill to the locked row count with additional valid observations.
    while len(rows) < 49:
        idx = len(rows) % 16 + 1
        ds = DOMAINS[len(rows) % 2]
        prefix = "HC" if ds == "hc2" else "V"
        retained = bool((len(rows) // 2) % 2)
        pos += 1
        rows.append({"case_id": f"{prefix}{idx:02d}", "dataset": ds, "center_position": pos,
                     "retained": retained, "gt_observed": True, "reference_delta_iou_pp": (-1.0) ** len(rows) * .03})
    return rows


def contracts() -> dict[str, Any]:
    rows = _synthetic_rows()
    lookup, aliases = validate_rows(rows)
    summary = build_metric_aggregates(lookup)
    summary_count, _ = compare_metric_summary(summary, summary)
    pairs = build_paired_aggregates(lookup)
    pair_count, _, tails = compare_paired(pairs, pairs)
    ref_rows = _synthetic_reference_rows()
    ref_summary = build_reference_aggregates(ref_rows, aliases)
    ref_count, _ = compare_reference_summary(ref_summary, ref_summary)
    need(next(x for x in summary if x["dataset"] == "hc2" and x["arm"] == "F" and x["metric"] == "m_sIoU_unsupervised_GT")["unknown_count"] == 1,
         "synthetic metric unknown was not preserved")
    need(sum(x["unknown_sources"] for x in pairs) == 2, "synthetic paired unknown was not preserved")
    need(sum(x["unknown_count"] for x in ref_summary) > 0, "synthetic reference unknown was not preserved")
    need(tails["positive"] > 0 and tails["negative"] > 0 and tails["neutral"] >= 0,
         "synthetic sign tails were not represented")

    bad_summary = [dict(x) for x in summary]
    bad_summary[0] = dict(bad_summary[0], mean=(bad_summary[0]["mean"] or 0) + .01)
    try:
        compare_metric_summary(bad_summary, summary)
    except AuditError:
        pass
    else:
        raise AuditError("synthetic changed mean was not rejected")

    bad_pairs = [dict(x) for x in pairs]
    bad_pairs[0] = dict(bad_pairs[0], positive_sources=bad_pairs[0]["positive_sources"] + 1)
    try:
        compare_paired(bad_pairs, pairs)
    except AuditError:
        pass
    else:
        raise AuditError("synthetic changed sign tail was not rejected")

    bad_diag = [dict(x) for x in ref_rows]
    bad_diag[-1] = dict(bad_diag[-1], gt_observed=False, reference_delta_iou_pp=.1)
    try:
        build_reference_aggregates(bad_diag, aliases)
    except AuditError:
        pass
    else:
        raise AuditError("synthetic unknown with non-null value was not rejected")

    missing_rows = rows[:-1]
    try:
        validate_rows(missing_rows)
    except AuditError:
        pass
    else:
        raise AuditError("synthetic missing row was not rejected")
    return {
        "status": "pass",
        "CPU_only": True,
        "GT_read": False,
        "checks": 10,
        "synthetic_case_arm_rows": len(rows),
        "synthetic_summary_scalars": summary_count,
        "synthetic_paired_records": pair_count,
        "synthetic_reference_groups": ref_count,
        "unknowns_and_positive_negative_neutral_tails": True,
        "tampered_mean_sign_tail_unknown_and_missing_row_rejected": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.contracts:
        print(json.dumps(contracts(), ensure_ascii=False))
        return
    report = audit(args.output)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output.resolve().relative_to(ROOT)),
        "case_arm_row_count": report["case_arm_row_count"],
        "metric_summary_scalar_count": report["metric_summary_scalar_count"],
        "paired_comparison_count": report["paired_comparison_count"],
        "reference_diagnostic_row_count": report["reference_diagnostic_row_count"],
        "GT_read": False,
        "new_scoring": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
