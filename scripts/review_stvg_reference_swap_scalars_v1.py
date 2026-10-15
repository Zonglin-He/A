"""Independent scalar reaggregation for sealed E1 reference-swap outputs.

This reader consumes only the scorer's saved POSTSEAL_ROWS/SUMMARY/PAIRED
JSON.  It never imports a scorer, target loader, model, or annotation reader.
Run ``contracts`` before seal; run ``audit --completion <path>`` only after the
CPU completion receipt is present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_reference_recovery_v1/Reference_swap"
ARMS = ("F", "T", "S_new_crop", "TS_new_crop",
        "S_WF_common", "TS_WF_common", "S_DINO_common", "TS_DINO_common")
METRICS = ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT",
           "m_sIoU_supervised_GT", "m_sIoU_unsupervised_GT")
CONTRASTS = (
    ("S_DINO_common_minus_S_WF_common", "S_DINO_common", "S_WF_common"),
    ("TS_DINO_common_minus_TS_WF_common", "TS_DINO_common", "TS_WF_common"),
    ("S_WF_common_minus_S_new_crop", "S_WF_common", "S_new_crop"),
    ("TS_WF_common_minus_TS_new_crop", "TS_WF_common", "TS_new_crop"),
    ("S_WF_common_minus_F", "S_WF_common", "F"),
    ("TS_WF_common_minus_T", "TS_WF_common", "T"),
)
SIGN_EPSILON = 1e-10
OUT = BASE / "ROOT_POSTSEAL_SCALAR_READBACK.json"


class ReadbackError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ReadbackError(message)


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReadbackError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ReadbackError(f"nonfinite JSON token: {value}")


def read_json(path: Path) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=_pairs_no_duplicates,
                             parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadbackError(f"cannot read {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_in_root(raw: str) -> Path:
    require(isinstance(raw, str) and raw, "empty completion path")
    p = Path(raw)
    resolved = (p if p.is_absolute() else ROOT / p).resolve()
    require(resolved == ROOT or ROOT in resolved.parents,
            "completion receipt path escapes repository")
    return resolved


def _validate_rows(rows: Any) -> tuple[dict[tuple[str, str], Mapping[str, Any]], dict[str, str]]:
    require(isinstance(rows, list) and len(rows) == 256,
            "postseal rows must contain exactly 256 case/arm records")
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    dataset_by_case: dict[str, str] = {}
    null_metric_cells = 0
    for row in rows:
        require(isinstance(row, dict), "postseal row must be an object")
        cid, dataset, arm = row.get("case_id"), row.get("dataset"), row.get("arm")
        require(isinstance(cid, str) and cid and dataset in ("hc2", "vidstg") and arm in ARMS,
                "postseal row has an invalid case/domain/arm key")
        key = (cid, arm)
        require(key not in by_key, f"duplicate row: {cid}/{arm}")
        by_key[key] = row
        require(cid not in dataset_by_case or dataset_by_case[cid] == dataset,
                f"case belongs to multiple domains: {cid}")
        dataset_by_case[cid] = dataset
        metrics = row.get("metrics")
        require(isinstance(metrics, dict) and set(metrics) == set(METRICS),
                f"metric schema mismatch for {cid}/{arm}")
        for metric, value in metrics.items():
            if value is None:
                null_metric_cells += 1
            else:
                require(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(float(value)),
                        f"invalid {metric} scalar for {cid}/{arm}")
    case_ids = sorted(dataset_by_case)
    require(len(case_ids) == 32, "postseal rows must cover exactly 32 sources")
    require(sum(dataset_by_case[cid] == "hc2" for cid in case_ids) == 16 and
            sum(dataset_by_case[cid] == "vidstg" for cid in case_ids) == 16,
            "postseal row source split must be 16 HC2 and 16 VidSTG")
    require(set(by_key) == {(cid, arm) for cid in case_ids for arm in ARMS},
            "postseal rows do not contain every source/arm combination")
    return by_key, {"case_count": len(case_ids), "row_count": len(rows),
                    "null_metric_cells": null_metric_cells,
                    "case_count_hc2": 16, "case_count_vidstg": 16}


def reaggregate(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute all summaries and paired tails from the saved scalar rows."""
    by_key, _counts = _validate_rows(list(rows))
    case_to_dataset = {cid: by_key[cid, "F"]["dataset"] for cid, _arm in by_key}
    case_ids = sorted(case_to_dataset)

    summaries: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for dataset in ("hc2", "vidstg"):
        ids = [cid for cid in case_ids if case_to_dataset[cid] == dataset]
        require(len(ids) == 16, f"{dataset} does not contain 16 sources")
        for arm in ARMS:
            for metric in METRICS:
                values: list[float] = []
                unknown: list[str] = []
                for cid in ids:
                    value = by_key[cid, arm]["metrics"][metric]
                    if value is None:
                        unknown.append(cid)
                    else:
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
        for contrast, left, right in CONTRASTS:
            for metric in METRICS:
                deltas: dict[str, float] = {}
                unknown: list[str] = []
                for cid in ids:
                    lv = by_key[cid, left]["metrics"][metric]
                    rv = by_key[cid, right]["metrics"][metric]
                    if lv is None or rv is None:
                        unknown.append(cid)
                    else:
                        deltas[cid] = 100.0 * (float(lv) - float(rv))
                values = list(deltas.values())
                positives = [cid for cid, value in deltas.items() if value > SIGN_EPSILON]
                negatives = [cid for cid, value in deltas.items() if value < -SIGN_EPSILON]
                neutrals = [cid for cid, value in deltas.items()
                            if abs(value) <= SIGN_EPSILON]
                paired.append({
                    "dataset": dataset, "contrast": contrast,
                    "comparison_arm": left, "baseline_arm": right, "metric": metric,
                    "n_pairs": len(values), "unknown_count": len(unknown),
                    "unknown_case_ids": unknown,
                    "positive_count": len(positives), "positive_case_ids": positives,
                    "negative_count": len(negatives), "negative_case_ids": negatives,
                    "neutral_count": len(neutrals), "neutral_case_ids": neutrals,
                    "neutral_tolerance_pp": SIGN_EPSILON,
                    "mean_delta_pp": math.fsum(values) / len(values) if values else None,
                    "median_delta_pp": statistics.median(values) if values else None,
                    "min_delta_pp": min(values) if values else None,
                    "max_delta_pp": max(values) if values else None,
                    "per_source_delta_pp": deltas,
                })
    require(len(summaries) == 80 and len(paired) == 60,
            "reaggregated summary/paired cardinality mismatch")
    return summaries, paired


def _same_value(left: Any, right: Any, where: str) -> None:
    """Exact JSON-shape and scalar comparison; null is never coerced to zero."""
    require(type(left) is type(right) or
            (isinstance(left, (int, float)) and not isinstance(left, bool)
             and isinstance(right, (int, float)) and not isinstance(right, bool)),
            f"type mismatch at {where}")
    if isinstance(left, dict):
        require(set(left) == set(right), f"key mismatch at {where}")
        for key in left:
            _same_value(left[key], right[key], f"{where}.{key}")
    elif isinstance(left, list):
        require(len(left) == len(right), f"list length mismatch at {where}")
        for index, (a, b) in enumerate(zip(left, right)):
            _same_value(a, b, f"{where}[{index}]")
    elif isinstance(left, (float, int)) and not isinstance(left, bool):
        require(float(left) == float(right), f"scalar mismatch at {where}")
    else:
        require(left == right, f"value mismatch at {where}")


def _index_records(records: Any, key_fields: tuple[str, ...], expected: int,
                   label: str) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    require(isinstance(records, list) and len(records) == expected,
            f"{label} must contain {expected} records")
    indexed: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in records:
        require(isinstance(row, dict), f"{label} contains non-object record")
        key = tuple(row.get(field) for field in key_fields)
        require(all(item is not None for item in key) and key not in indexed,
                f"{label} has missing/duplicate key")
        indexed[key] = row
    return indexed


def _check_summary(saved: Any, expected: list[dict[str, Any]]) -> None:
    actual = _index_records(saved, ("dataset", "arm", "metric"), 80,
                            "POSTSEAL_SUMMARY.summary")
    want = _index_records(expected, ("dataset", "arm", "metric"), 80,
                          "independently recomputed summary")
    require(set(actual) == set(want), "summary key set differs from recomputation")
    for key in want:
        _same_value(dict(actual[key]), dict(want[key]), f"summary{key}")


def _check_paired(saved: Any, expected: list[dict[str, Any]]) -> dict[str, int]:
    actual = _index_records(saved, ("dataset", "contrast", "metric"), 60,
                            "PARENT_PAIRED_DELTAS.records")
    want = _index_records(expected, ("dataset", "contrast", "metric"), 60,
                          "independently recomputed paired table")
    require(set(actual) == set(want), "paired key set differs from recomputation")
    tails = {"positive": 0, "negative": 0, "neutral": 0, "unknown": 0}
    for key in want:
        _same_value(dict(actual[key]), dict(want[key]), f"paired{key}")
        row = want[key]
        tails["positive"] += int(row["positive_count"])
        tails["negative"] += int(row["negative_count"])
        tails["neutral"] += int(row["neutral_count"])
        tails["unknown"] += int(row["unknown_count"])
        require(sum(int(row[name]) for name in
                    ("positive_count", "negative_count", "neutral_count", "unknown_count")) == 16,
                f"paired tail denominator is not 16 for {key}")
    return tails


def _audit(completion_arg: str) -> dict[str, Any]:
    completion_path = _path_in_root(completion_arg)
    require(completion_path.is_file(), "CPU_COMPLETION receipt is missing")
    completion = read_json(completion_path)
    require(isinstance(completion, dict) and completion.get("status") in ("complete", "pass"),
            "CPU_COMPLETION is not complete")
    for field in ("exit_code", "returncode"):
        if field in completion:
            require(completion[field] == 0, f"CPU_COMPLETION {field} is nonzero")

    rows_path = BASE / "POSTSEAL_ROWS.json"
    summary_path = BASE / "POSTSEAL_SUMMARY.json"
    paired_path = BASE / "PARENT_PAIRED_DELTAS.json"
    docs = [read_json(path) for path in (rows_path, summary_path, paired_path)]
    rows_doc, summary_doc, paired_doc = docs
    expected_schemas = ("stvg-reference-swap-postseal-rows-v1",
                        "stvg-reference-swap-postseal-summary-v1",
                        "stvg-reference-swap-paired-deltas-v1")
    for doc, schema, label in zip(docs, expected_schemas, ("rows", "summary", "paired")):
        require(isinstance(doc, dict) and doc.get("schema") == schema
                and doc.get("status") == "complete" and doc.get("GT_read") is True,
                f"saved {label} output is not a completed postseal artifact")
    require(paired_doc.get("bootstrap_executed") is False,
            "paired postseal output unexpectedly claims a bootstrap")
    require(rows_doc.get("bindings") == summary_doc.get("bindings") == paired_doc.get("bindings"),
            "saved rows/summary/paired binding maps differ")
    rows = rows_doc.get("rows")
    by_key, counts = _validate_rows(rows)
    summary, paired = reaggregate(rows)
    _check_summary(summary_doc.get("summary"), summary)
    tails = _check_paired(paired_doc.get("records"), paired)
    return {
        "schema": "stvg-reference-swap-independent-scalar-readback-v1",
        "status": "pass", "case_count": counts["case_count"],
        "case_count_by_dataset": {"hc2": counts["case_count_hc2"],
                                  "vidstg": counts["case_count_vidstg"]},
        "row_count": counts["row_count"], "arm_count": len(ARMS),
        "metric_count": len(METRICS), "summary_cell_count": len(summary),
        "paired_contrast_cell_count": len(paired),
        "null_metric_cells_preserved": counts["null_metric_cells"],
        "paired_tail_counts_across_all_cells": tails,
        "neutral_tolerance_pp": SIGN_EPSILON,
        "bindings": {"cpu_completion_sha256": sha256_file(completion_path),
                     "postseal_rows_sha256": sha256_file(rows_path),
                     "postseal_summary_sha256": sha256_file(summary_path),
                     "paired_deltas_sha256": sha256_file(paired_path),
                     "postseal_bindings": rows_doc["bindings"]},
        "GT_read": False, "CPU_only": True, "model_loaded": False, "GPU_used": False,
        "scope": "independent scalar reaggregation from saved postseal rows/summary/paired JSON only",
    }


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite readback: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _contracts() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dataset in ("hc2", "vidstg"):
        for index in range(16):
            cid = f"{dataset}-{index:02d}"
            for arm_idx, arm in enumerate(ARMS):
                metrics = {}
                for metric_idx, metric in enumerate(METRICS):
                    if metric == "m_sIoU_unsupervised_GT" and index in (0, 7):
                        value = None
                    elif metric == "m_tIoU" and index == 2:
                        value = 0.4
                    else:
                        value = (index + 1) / 100.0 + arm_idx / 1000.0 + metric_idx / 10000.0
                    metrics[metric] = value
                rows.append({"case_id": cid, "dataset": dataset, "arm": arm,
                             "status": "scored", "metrics": metrics})
    summary, paired = reaggregate(rows)
    assert len(rows) == 256 and len(summary) == 80 and len(paired) == 60
    assert all(row["n_unknown"] == 2 for row in summary
               if row["metric"] == "m_sIoU_unsupervised_GT")
    assert all(row["unknown_count"] == 2 for row in paired
               if row["metric"] == "m_sIoU_unsupervised_GT")
    _check_summary(summary, summary)
    _check_paired(paired, paired)
    bad = [dict(row) for row in summary]
    bad[0] = dict(bad[0], mean=float(bad[0]["mean"]) + 0.001)
    try:
        _check_summary(bad, summary)
    except ReadbackError:
        pass
    else:
        raise AssertionError("summary corruption was accepted")
    bad_rows = [dict(row, metrics=dict(row["metrics"])) for row in rows]
    bad_rows[0]["metrics"][METRICS[0]] = None
    _validate_rows(bad_rows)
    # The row remains structurally valid; reaggregation must expose the
    # altered null through different summary/pair tables.
    altered_summary, altered_paired = reaggregate(bad_rows)
    assert altered_summary != summary and altered_paired != paired
    return {"status": "pass", "checks": 9, "rows": len(rows),
            "summary_cells": len(summary), "paired_cells": len(paired),
            "null_cells": sum(value is None for row in rows for value in row["metrics"].values()),
            "GT_read": False, "model_loaded": False, "GPU_used": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("contracts")
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--completion", required=True,
                              help="repo-relative CPU_COMPLETION JSON path")
    args = parser.parse_args()
    if args.mode == "contracts":
        print(json.dumps(_contracts(), sort_keys=True))
        return
    result = _audit(args.completion)
    _write_new(OUT, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
