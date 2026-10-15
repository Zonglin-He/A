"""Independent scalar reaggregation for the sealed E3 LN-carry postseal tables.

Reads only aggregate prediction/metric JSON and hashes sealed prediction files;
it does not load labels, videos, model state, or serialized predictions.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_reference_recovery_v1/LN_carry"
ORDERS = ("order1", "order2", "order3")
DOMAINS = ("hc2", "vidstg")
ARMS = ("F", "T", "Episodic_Native", "Episodic_T",
        "Before_Native", "Before_T", "After_Native", "After_T")
METRICS = ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT",
           "m_sIoU_supervised_GT", "m_sIoU_unsupervised_GT")
PAIRS = (
    ("Before_Native", "F"), ("After_Native", "Before_Native"),
    ("After_Native", "Episodic_Native"), ("Before_T", "T"),
    ("After_T", "Before_T"), ("After_T", "Episodic_T"),
)
PAIRED_METRICS = ("m_vIoU", "m_sIoU_dense_GT")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite_or_null(value: Any) -> bool:
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)
                             and math.isfinite(float(value)))


def _require_hash(path: Path, expected: str, expected_bytes: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise AssertionError(f"byte mismatch: {path}")
    actual = _sha(path)
    if actual != expected:
        raise AssertionError(f"SHA mismatch: {path}")


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def run() -> dict[str, Any]:
    row_path = BASE / "POSTSEAL_ROWS.json"
    summary_path = BASE / "POSTSEAL_SUMMARY.json"
    paired_path = BASE / "PARENT_PAIRED_DELTAS.json"
    gt_gate_path = BASE / "GT_EXPOSURE.json"
    student_barrier_path = BASE / "STUDENT_GLOBAL_BARRIER.json"
    prediction_seal_path = BASE / "GLOBAL_PREDICTION_BARRIER.json"
    math_receipt_path = BASE / "ROOT_INDEPENDENT_OPTIMIZER_MATH_READBACK.json"
    root_receipt_path = BASE / "ROOT_READBACK.json"

    rows_doc = _read(row_path)
    summary_doc = _read(summary_path)
    paired_doc = _read(paired_path)
    gt_gate = _read(gt_gate_path)
    student_barrier = _read(student_barrier_path)
    seal = _read(prediction_seal_path)
    math_receipt = _read(math_receipt_path)
    root_receipt = _read(root_receipt_path)

    assert rows_doc.get("status") == summary_doc.get("status") == paired_doc.get("status") == "complete"
    assert rows_doc.get("GT_read") is summary_doc.get("GT_read") is paired_doc.get("GT_read") is True
    assert gt_gate.get("status") == "after_all_six_stream_predictions_and_root"
    assert gt_gate.get("GT_read") is True
    assert gt_gate.get("seal_sha256") == _sha(prediction_seal_path)
    assert student_barrier.get("status") == "complete" and student_barrier.get("count") == 96
    assert student_barrier.get("GT_read") is False
    assert seal.get("status") == "complete" and seal.get("count") == 96
    assert seal.get("independent_source_count") == 32 and seal.get("GT_read") is False
    assert root_receipt.get("status") == "pass" and root_receipt.get("arrival_count") == 96
    assert root_receipt.get("stream_count") == 6 and root_receipt.get("GT_read") is False
    assert math_receipt.get("status") == "pass" and math_receipt.get("arrival_count_audited") == 96
    assert math_receipt.get("GT_read") is False

    # Verify the sealed, pre-GT prediction file closure without deserializing any .pt.
    seal_files = seal.get("files")
    assert isinstance(seal_files, list) and len(seal_files) == 301
    seen_paths: dict[str, dict[str, Any]] = {}
    verified_bytes = 0
    duplicate_seal_entries = 0
    for item in seal_files:
        rel = item["path"]
        assert isinstance(rel, str)
        if rel in seen_paths:
            previous = seen_paths[rel]
            assert previous["sha256"] == item["sha256"]
            assert previous.get("bytes") == item.get("bytes")
            duplicate_seal_entries += 1
            continue
        seen_paths[rel] = item
        p = ROOT / rel
        _require_hash(p, item["sha256"], item.get("bytes"))
        verified_bytes += p.stat().st_size
    student_files = student_barrier.get("files")
    assert isinstance(student_files, list) and len(student_files) == 96
    seal_by_path = {row["path"]: row for row in seal_files}
    assert len(seal_by_path) == len(seen_paths)
    for item in student_files:
        row = seal_by_path.get(item["path"])
        assert row is not None
        assert row["sha256"] == item["sha256"]
        assert row.get("bytes") == item.get("bytes")

    raw_rows = rows_doc.get("rows")
    assert isinstance(raw_rows, list) and len(raw_rows) == 768
    key_to_row: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    ids_by_stream: dict[tuple[str, str], set[str]] = {}
    null_counts: Counter[tuple[str, str, str]] = Counter()
    values_by_key: dict[tuple[str, str, str, str], list[float]] = {}
    for row in raw_rows:
        ds, order, cid, arm = row["dataset"], row["order"], row["case_id"], row["arm"]
        assert ds in DOMAINS and order in ORDERS and arm in ARMS
        metrics = row.get("metrics")
        assert isinstance(metrics, dict) and set(metrics) == set(METRICS)
        key = (ds, order, cid, arm)
        assert key not in key_to_row
        key_to_row[key] = row
        ids_by_stream.setdefault((ds, order), set()).add(cid)
        for metric in METRICS:
            value = metrics[metric]
            assert _finite_or_null(value), (key, metric)
            if value is None:
                null_counts[(ds, arm, metric)] += 1
            else:
                values_by_key.setdefault((ds, order, arm, metric), []).append(float(value))
    assert len(key_to_row) == 768
    expected_ids_by_stream = {key: ids for key, ids in ids_by_stream.items()}
    assert set(expected_ids_by_stream) == {(ds, order) for ds in DOMAINS for order in ORDERS}
    for key, ids in ids_by_stream.items():
        assert len(ids) == 16, key
        assert len({row["stream_position"] for row in raw_rows
                    if (row["dataset"], row["order"]) == key}) == 16
    all_ids_by_domain = {ds: set().union(*(ids_by_stream[(ds, order)] for order in ORDERS)) for ds in DOMAINS}
    assert all(len(ids) == 16 for ids in all_ids_by_domain.values())
    assert all_ids_by_domain["hc2"].isdisjoint(all_ids_by_domain["vidstg"])
    assert sum(len(ids) for ids in all_ids_by_domain.values()) == 32
    for ds in DOMAINS:
        for order in ORDERS:
            for cid in ids_by_stream[(ds, order)]:
                for arm in ARMS:
                    assert (ds, order, cid, arm) in key_to_row

    summaries = summary_doc.get("summary")
    assert isinstance(summaries, list) and len(summaries) == 240
    summary_index = {}
    for item in summaries:
        key = (item["dataset"], item["order"], item["arm"], item["metric"])
        assert key not in summary_index
        summary_index[key] = item
    assert len(summary_index) == 240
    summary_checks = 0
    summary_nulls = 0
    max_summary_abs_error = 0.0
    for ds in DOMAINS:
        for order in ORDERS:
            ids = sorted(ids_by_stream[(ds, order)])
            for arm in ARMS:
                for metric in METRICS:
                    vals = [float(key_to_row[(ds, order, cid, arm)]["metrics"][metric])
                            for cid in ids
                            if key_to_row[(ds, order, cid, arm)]["metrics"][metric] is not None]
                    item = summary_index[(ds, order, arm, metric)]
                    assert item["count"] == len(vals)
                    expected_mean = _mean(vals)
                    if expected_mean is None:
                        assert item["mean"] is None
                        summary_nulls += 1
                    else:
                        assert item["mean"] is not None
                        err = abs(float(item["mean"]) - expected_mean)
                        max_summary_abs_error = max(max_summary_abs_error, err)
                        assert err <= 1e-13
                    summary_checks += 1

    paired = paired_doc.get("records")
    assert isinstance(paired, list) and len(paired) == 72
    assert paired_doc.get("bootstrap_executed") is False
    assert paired_doc.get("independent_source_count") == 32
    assert paired_doc.get("orders_are_repeated_sources") is True
    paired_index = {}
    paired_sign_rows = []
    max_pair_mean_abs_error = 0.0
    for item in paired:
        key = (item["dataset"], item["order"], item["left"], item["right"], item["metric"])
        assert key not in paired_index
        paired_index[key] = item
    assert len(paired_index) == 72
    paired_checks = 0
    paired_null_sources = 0
    for ds in DOMAINS:
        for order in ORDERS:
            ids = sorted(ids_by_stream[(ds, order)])
            for left, right in PAIRS:
                for metric in PAIRED_METRICS:
                    key = (ds, order, left, right, metric)
                    item = paired_index[key]
                    deltas = {}
                    for cid in ids:
                        lv = key_to_row[(ds, order, cid, left)]["metrics"][metric]
                        rv = key_to_row[(ds, order, cid, right)]["metrics"][metric]
                        if lv is None or rv is None:
                            paired_null_sources += 1
                            continue
                        deltas[cid] = 100.0 * (float(lv) - float(rv))
                    assert item["count"] == len(deltas)
                    saved = item.get("per_source_delta_pp")
                    assert isinstance(saved, dict) and set(saved) == set(deltas)
                    for cid, delta in deltas.items():
                        assert _finite_or_null(saved[cid]) and abs(float(saved[cid]) - delta) <= 1e-13
                    vals = list(deltas.values())
                    mean = _mean(vals)
                    if mean is None:
                        assert item["mean_delta_pp"] is None
                        assert item["positive"] == item["negative"] == item["neutral"] == 0
                        assert item["worst"] is item["best"] is None
                        positive = negative = neutral = []
                    else:
                        err = abs(float(item["mean_delta_pp"]) - mean)
                        max_pair_mean_abs_error = max(max_pair_mean_abs_error, err)
                        assert err <= 1e-13
                        positive = sorted(cid for cid, x in deltas.items() if x > 1e-10)
                        negative = sorted(cid for cid, x in deltas.items() if x < -1e-10)
                        neutral = sorted(cid for cid, x in deltas.items() if abs(x) <= 1e-10)
                        assert item["positive"] == len(positive)
                        assert item["negative"] == len(negative)
                        assert item["neutral"] == len(neutral)
                        assert item["worst"] == min(vals)
                        assert item["best"] == max(vals)
                    paired_sign_rows.append({
                        "dataset": ds, "order": order, "left": left, "right": right,
                        "metric": metric, "count": len(deltas),
                        "positive_count": len(positive), "negative_count": len(negative),
                        "neutral_count": len(neutral),
                        "positive_cases": positive, "negative_cases": negative,
                        "neutral_cases": neutral,
                        "worst_delta_pp": item["worst"], "best_delta_pp": item["best"],
                        "mean_delta_pp": item["mean_delta_pp"],
                    })
                    paired_checks += 1

    assert summary_checks == 240 and paired_checks == 72
    rows_sha, summary_sha, paired_sha = _sha(row_path), _sha(summary_path), _sha(paired_path)
    result = {
        "schema": "stvg-reference-carry-independent-postseal-scalar-readback-v1",
        "status": "pass",
        "scope": "independent reaggregation of sealed postseal rows, summaries and per-source paired deltas; no label loader, new GT scoring, model or GPU",
        "case_count": 32,
        "orders_per_domain": 3,
        "stream_count": 6,
        "rows_expected": 768,
        "rows_audited": len(key_to_row),
        "summary_cells_expected": 240,
        "summary_cells_audited": summary_checks,
        "paired_cells_expected": 72,
        "paired_cells_audited": paired_checks,
        "arms": list(ARMS),
        "metrics": list(METRICS),
        "null_metric_cells_in_rows": {
            f"{ds}/{arm}/{metric}": null_counts[(ds, arm, metric)]
            for ds in DOMAINS for arm in ARMS for metric in METRICS
            if null_counts[(ds, arm, metric)] > 0
        },
        "summary_all_null_cells": summary_nulls,
        "paired_null_source_comparisons": paired_null_sources,
        "maximum_summary_mean_abs_error": max_summary_abs_error,
        "maximum_paired_mean_abs_error": max_pair_mean_abs_error,
        "paired_signs_and_tails": paired_sign_rows,
        "input_hashes": {
            "postseal_rows_sha256": rows_sha,
            "postseal_summary_sha256": summary_sha,
            "parent_paired_deltas_sha256": paired_sha,
            "gt_exposure_metadata_sha256": _sha(gt_gate_path),
            "student_global_barrier_sha256": _sha(student_barrier_path),
            "prediction_seal_sha256": _sha(prediction_seal_path),
            "root_state_chain_readback_sha256": _sha(root_receipt_path),
            "independent_optimizer_math_readback_sha256": _sha(math_receipt_path),
        },
        "prediction_seal_file_entries": len(seal_files),
        "duplicate_prediction_seal_entries": duplicate_seal_entries,
        "sealed_prediction_unique_files_verified": len(seen_paths),
        "sealed_prediction_file_bytes_verified": verified_bytes,
        "GT_derived_metrics_read": True,
        "new_GT_label_read": False,
        "model_loaded": False,
        "GPU_used": False,
        "time": time.time(),
    }
    out = BASE / "ROOT_CARRY_SCALAR_READBACK.json"
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
