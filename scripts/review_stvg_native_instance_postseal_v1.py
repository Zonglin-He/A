"""Independent scalar readback of sealed native-instance post-seal scores.

This reader verifies that the score documents bind to the current prediction
seal and scorer completion receipt, then independently recomputes summaries,
paired signs, unknown membership, and bootstrap intervals. Metric geometry is
checked by the scorer's official/independent implementations; this reader does
not reload labels or replay a model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.stvg_native_instance_common_v1 import BASE, read, write, sha

EXPECTED_ARMS = (
    "F", "T", "native_teacher_direct", "direct", "iou",
    "reliability_weighted_iou", "T_direct", "T_iou",
    "T_reliability_weighted_iou",
)
EXPECTED_METRICS = (
    "m_vIoU", "m_tIoU", "m_sIoU_dense_GT",
    "m_sIoU_supervised_GT", "m_sIoU_unsupervised_GT",
)
EXPECTED_CONTRASTS = (
    ("T_minus_F", "T", "F"),
    ("native_teacher_direct_minus_T", "native_teacher_direct", "T"),
    ("T_direct_minus_T", "T_direct", "T"),
    ("T_iou_minus_T", "T_iou", "T"),
    ("T_reliability_weighted_iou_minus_T", "T_reliability_weighted_iou", "T"),
    ("direct_minus_F", "direct", "F"),
    ("iou_minus_F", "iou", "F"),
    ("reliability_weighted_iou_minus_F", "reliability_weighted_iou", "F"),
)
EXPECTED_BOOTSTRAP_REPLICATES = 10_000
EXPECTED_BOOTSTRAP_SEED = 20261014
EXPECTED_SIGN_EPSILON = 1e-12


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _expected_common_bindings(lock: Mapping[str, Any]) -> dict[str, str]:
    return {
        "roster_sha256": lock["roster_sha256"],
        "reference_lock_sha256": lock["reference_lock_sha256"],
        "direct_teacher_readout_sha256": lock["direct_readout_sha256"],
        "tracking_barrier_sha256": lock["phase_hashes"]["TRACKING"],
        "teacher_barrier_sha256": lock["phase_hashes"]["TEACHER"],
        "student_barrier_sha256": lock["phase_hashes"]["STUDENT"],
    }


def _expected_row_bindings(lock: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    result = {}
    for cid in lock["case_ids"]:
        ref = lock["ref_by_case"][cid]
        result[cid] = {
            "source_case_sha256": lock["roster_by_case"][cid]["source_case_sha256"],
            "student_artifact_sha256": sha(lock["student_paths"][cid]),
            "student_receipt_sha256": sha(BASE / "student" / f"{cid}.receipt.json"),
            "teacher_record_sha256": ref["teacher_record_sha256"],
            "teacher_receipt_sha256": ref["teacher_receipt_sha256"],
            "reference_lock_sha256": lock["reference_lock_sha256"],
        }
    return result


def _validate_documents(lock: Mapping[str, Any], scorer: Any,
                        document: Mapping[str, Any], summary: Mapping[str, Any],
                        completion: Mapping[str, Any], math_receipt: Mapping[str, Any],
                        row_bindings: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """Validate writer schemas/bindings and independently recompute scalars."""
    case_ids = list(lock["case_ids"])
    _need(tuple(scorer.ARMS) == EXPECTED_ARMS and
          tuple(scorer.METRICS) == EXPECTED_METRICS and
          tuple(scorer._contrast_specs()) == EXPECTED_CONTRASTS and
          scorer.BOOTSTRAP_REPLICATES == EXPECTED_BOOTSTRAP_REPLICATES and
          scorer.BOOTSTRAP_SEED == EXPECTED_BOOTSTRAP_SEED and
          scorer.SIGN_EPSILON == EXPECTED_SIGN_EPSILON,
          "scorer arm/metric/contrast/bootstrap contract changed from the locked readback design")
    common = _expected_common_bindings(lock)
    barrier = {"path": str(scorer.SEAL.relative_to(ROOT)),
               "sha256": lock["seal_sha256"]}
    _need(document.get("schema") == scorer.ROWS_SCHEMA and
          document.get("status") == "complete" and document.get("GT_read") is True and
          document.get("metric_scale") == "0_to_1" and
          document.get("case_count") == 8 and
          document.get("arm_count") == len(EXPECTED_ARMS),
          "score rows document schema/status/scope differs from scorer writer")
    _need(summary.get("schema") == scorer.SUMMARY_SCHEMA and
          summary.get("status") == "complete" and summary.get("GT_read") is True and
          summary.get("metric_scale") == "0_to_1" and
          summary.get("case_count_by_dataset") == {"hc2": 4, "vidstg": 4},
          "score summary schema/status/scope differs from scorer writer")
    _need(document.get("prediction_barrier") == barrier and
          summary.get("prediction_barrier") == barrier,
          "score documents are stale against the current prediction barrier")
    _need(document.get("bindings") == common and summary.get("bindings") == common,
          "score document common bindings differ from current locked inputs")

    _need(completion.get("schema") == "stvg-native-instance-cpu-completion-v1" and
          completion.get("status") == "pending_root_review_visualization_publication" and
          completion.get("scoring_status") == "complete" and
          completion.get("experiment_complete") is False and
          completion.get("GT_read") is True and completion.get("case_count") == 8 and
          completion.get("arm_count") == len(EXPECTED_ARMS) and
          completion.get("row_count") == 72 and
          completion.get("prediction_barrier_sha256") == lock["seal_sha256"],
          "scorer completion receipt schema/status/bindings differ")
    _need(completion.get("rows_sha256") == _canonical_sha(document) and
          completion.get("summary_sha256") == _canonical_sha(summary),
          "canonical score-document hashes differ from CPU_COMPLETION")

    _need(math_receipt.get("schema") ==
          "stvg-native-instance-independent-student-math-readback-v1" and
          math_receipt.get("status") == "pass" and math_receipt.get("GT_read") is False and
          math_receipt.get("CPU_only") is True and math_receipt.get("GPU_initialized") is False and
          math_receipt.get("model_replayed_by_reader") is False and
          math_receipt.get("case_count") == 8 and
          math_receipt.get("student_barrier_sha256") == lock["phase_hashes"]["STUDENT"] and
          math_receipt.get("reference_lock_sha256") == lock["reference_lock_sha256"] and
          math_receipt.get("roster_sha256") == lock["roster_sha256"],
          "student math readback is stale or not bound to the current sealed predictions")

    rows = document.get("rows")
    _need(isinstance(rows, list) and len(rows) == 72,
          "score rows must contain the complete 8x9 case-arm table")
    by_key = {(r.get("case_id"), r.get("arm")): r for r in rows}
    expected_row_keys = {(cid, arm) for cid in case_ids for arm in EXPECTED_ARMS}
    _need(len(by_key) == len(rows) and set(by_key) == expected_row_keys,
          "score rows contain duplicate, missing, or unexpected case-arm keys")

    checks = 0
    max_error = 0.0

    def equal(a: Any, b: Any) -> None:
        nonlocal checks, max_error
        checks += 1
        if a is None or b is None:
            _need(a is b, f"expected both values to be null: {a!r} vs {b!r}")
            return
        _need(isinstance(a, (int, float)) and not isinstance(a, bool) and
              isinstance(b, (int, float)) and not isinstance(b, bool) and
              math.isfinite(float(a)) and math.isfinite(float(b)),
              f"expected finite numeric scalars: {a!r} vs {b!r}")
        error = abs(float(a) - float(b))
        max_error = max(max_error, error)
        _need(error <= 2e-10, f"scalar mismatch {a!r} vs {b!r}, error={error:.9g}")

    dataset_cases = {
        dataset: sorted(cid for cid in case_ids
                        if lock["roster_by_case"][cid]["dataset"] == dataset)
        for dataset in ("hc2", "vidstg")
    }
    _need({k: len(v) for k, v in dataset_cases.items()} == {"hc2": 4, "vidstg": 4},
          "locked roster is not balanced across the two datasets")
    support_by_case: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        cid = row["case_id"]
        dataset = lock["roster_by_case"][cid]["dataset"]
        _need(row.get("dataset") == dataset,
              f"row dataset differs from the locked roster for {cid}")
        _need(row.get("bindings") == row_bindings[cid],
              f"row source/student/teacher/reference bindings differ for {cid}")
        support = row.get("support")
        _need(isinstance(support, Mapping), f"row support is missing for {cid}/{row['arm']}")
        integer_fields = (
            "source_grid_positions", "source_grid_supervised_position_count",
            "source_grid_unsupervised_position_count", "supervised_anchor_count",
            "unsupervised_position_count", "dense_gt_known_frame_count",
            "source_grid_gt_known_frame_count", "off_grid_gt_known_frame_count",
            "supervised_gt_known_frame_count", "unsupervised_gt_known_frame_count",
        )
        for field in integer_fields:
            value = support.get(field)
            _need(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                  f"malformed support count {field} for {cid}/{row['arm']}")
        anchor_positions = sorted(int(a["position"]) for a in
                                  lock["ref_by_case"][cid].get("anchors", []))
        _need(support["supervised_anchor_count"] == len(anchor_positions) and
              support["source_grid_supervised_position_count"] == len(anchor_positions) and
              support.get("supervised_positions") == anchor_positions,
              f"support supervised positions differ from locked references for {cid}/{row['arm']}")
        _need(support["source_grid_supervised_position_count"] +
              support["source_grid_unsupervised_position_count"] ==
              support["source_grid_positions"] and
              support["source_grid_unsupervised_position_count"] ==
              support["unsupervised_position_count"],
              f"source-grid support partition fails for {cid}/{row['arm']}")
        _need(support["source_grid_gt_known_frame_count"] +
              support["off_grid_gt_known_frame_count"] ==
              support["dense_gt_known_frame_count"] and
              support["supervised_gt_known_frame_count"] +
              support["unsupervised_gt_known_frame_count"] ==
              support["dense_gt_known_frame_count"],
              f"known-frame support partition fails for {cid}/{row['arm']}")
        _need(support["supervised_gt_known_frame_count"] <=
              support["supervised_anchor_count"],
              f"supervised known-frame count exceeds locked anchors for {cid}/{row['arm']}")
        supervised_ids = support.get("supervised_physical_frame_ids")
        unsupervised_ids = support.get("unsupervised_physical_frame_ids")
        _need(isinstance(supervised_ids, list) and isinstance(unsupervised_ids, list) and
              all(isinstance(fid, int) and not isinstance(fid, bool)
                  for fid in supervised_ids + unsupervised_ids) and
              supervised_ids == sorted(set(supervised_ids)) and
              unsupervised_ids == sorted(set(unsupervised_ids)) and
              set(supervised_ids).isdisjoint(unsupervised_ids) and
              len(supervised_ids) == support["supervised_gt_known_frame_count"] and
              len(unsupervised_ids) == support["unsupervised_gt_known_frame_count"],
              f"known physical-frame support lists differ from their counts for {cid}/{row['arm']}")
        _need(support.get("gt_support_split") ==
              "supervised=known GT physical IDs at reference anchors; unsupervised=all other known GT physical IDs, including off-grid frames",
              f"GT support split semantics differ for {cid}/{row['arm']}")
        if cid in support_by_case:
            _need(dict(support) == dict(support_by_case[cid]),
                  f"support differs across arms for the same case {cid}")
        else:
            support_by_case[cid] = support

        metrics = row.get("metrics")
        _need(isinstance(metrics, Mapping) and set(metrics) == set(EXPECTED_METRICS),
              f"metric key set differs from scorer contract for {cid}/{row['arm']}")
        if row["arm"] == "native_teacher_direct":
            if row.get("status") == "scored":
                _need(all(isinstance(metrics[k], (int, float)) and
                          not isinstance(metrics[k], bool) and math.isfinite(float(metrics[k]))
                          for k in EXPECTED_METRICS[:3]),
                      f"scored native teacher row has unknown primary metric for {cid}")
            else:
                _need(row.get("status") == "invalid_or_abstained_no_teacher_tube" and
                      all(value is None for value in metrics.values()),
                      f"unavailable native teacher row must remain fully unknown for {cid}")
        else:
            _need(row.get("status") == "scored" and
                  all(isinstance(metrics[k], (int, float)) and
                      not isinstance(metrics[k], bool) and math.isfinite(float(metrics[k]))
                  for k in EXPECTED_METRICS[:3]),
                  f"student row status/primary metric availability differs for {cid}/{row['arm']}")
        for value in metrics.values():
            _need(value is None or (isinstance(value, (int, float)) and
                  not isinstance(value, bool) and math.isfinite(float(value)) and
                  -1e-12 <= float(value) <= 1 + 1e-12),
                  f"metric is not a finite scalar in [0,1] for {cid}/{row['arm']}")
        audit = row.get("metric_audit") or {}
        _need(isinstance(audit, Mapping), f"metric audit is malformed for {cid}/{row['arm']}")
        for key, value in audit.items():
            if key.endswith("error") and value is not None:
                _need(isinstance(value, (int, float)) and not isinstance(value, bool) and
                      math.isfinite(float(value)) and float(value) <= 2e-7,
                      f"official/independent metric error exceeds tolerance for {cid}/{key}")

    # Validate the complete arm/metric key catalog, not only its row count.
    arm_summary_rows = summary.get("arm_metric_summaries")
    expected_arm_keys = {(dataset, arm, metric)
                         for dataset in dataset_cases
                         for arm in EXPECTED_ARMS for metric in EXPECTED_METRICS}
    arm_keys = [(r.get("dataset"), r.get("arm"), r.get("metric"))
                for r in arm_summary_rows] if isinstance(arm_summary_rows, list) else []
    _need(len(arm_keys) == 90 and len(set(arm_keys)) == 90 and
          set(arm_keys) == expected_arm_keys,
          "arm summary key catalog is duplicated, incomplete, or unexpected")
    for row in arm_summary_rows:
        dataset_cases_list = dataset_cases[row["dataset"]]
        values = [by_key[cid, row["arm"]]["metrics"][row["metric"]]
                  for cid in dataset_cases_list]
        valid = [float(value) for value in values if value is not None]
        unknown_ids = [cid for cid, value in zip(dataset_cases_list, values) if value is None]
        equal(row.get("n_valid"), len(valid))
        equal(row.get("n_unknown"), len(unknown_ids))
        _need(row.get("unknown_case_ids") == unknown_ids,
              f"arm summary unknown case membership differs for {row['dataset']}/{row['arm']}/{row['metric']}")
        expected = {"mean": math.fsum(valid) / len(valid) if valid else None,
                    "median": statistics.median(valid) if valid else None,
                    "min": min(valid) if valid else None,
                    "max": max(valid) if valid else None}
        for key, value in expected.items():
            equal(row.get(key), value)

    definitions = [{"contrast": name, "comparison_arm": left, "baseline_arm": right}
                   for name, left, right in EXPECTED_CONTRASTS]
    _need(summary.get("contrast_definitions") == definitions,
          "summary contrast definitions differ from the scorer contract")
    spec_by_name = {name: (left, right) for name, left, right in EXPECTED_CONTRASTS}
    contrast_rows = summary.get("contrasts")
    expected_contrast_keys = {(dataset, name, metric)
                              for dataset in dataset_cases
                              for name, _left, _right in EXPECTED_CONTRASTS
                              for metric in EXPECTED_METRICS}
    contrast_keys = [(r.get("dataset"), r.get("contrast"), r.get("metric"))
                     for r in contrast_rows] if isinstance(contrast_rows, list) else []
    _need(len(contrast_keys) == 80 and len(set(contrast_keys)) == 80 and
          set(contrast_keys) == expected_contrast_keys,
          "contrast summary key catalog is duplicated, incomplete, or unexpected")
    for row in contrast_rows:
        left, right = spec_by_name[row["contrast"]]
        _need(row.get("comparison_arm") == left and row.get("baseline_arm") == right,
              f"contrast arms differ from the scorer definition for {row['contrast']}")
        deltas: list[tuple[str, float]] = []
        unknown_ids: list[str] = []
        for cid in dataset_cases[row["dataset"]]:
            a = by_key[cid, left]["metrics"][row["metric"]]
            b = by_key[cid, right]["metrics"][row["metric"]]
            if a is not None and b is not None:
                deltas.append((cid, float(a) - float(b)))
            else:
                unknown_ids.append(cid)
        values = [delta for _cid, delta in deltas]
        positive_ids = [cid for cid, delta in deltas if delta > EXPECTED_SIGN_EPSILON]
        negative_ids = [cid for cid, delta in deltas if delta < -EXPECTED_SIGN_EPSILON]
        neutral_ids = [cid for cid, delta in deltas if abs(delta) <= EXPECTED_SIGN_EPSILON]
        equal(row.get("n_pairs"), len(values))
        equal(row.get("unknown_count"), len(unknown_ids))
        _need(row.get("unknown_case_ids") == unknown_ids and
              row.get("positive_case_ids") == positive_ids and
              row.get("negative_case_ids") == negative_ids and
              row.get("neutral_case_ids") == neutral_ids,
              f"contrast tail/unknown case membership differs for {row['dataset']}/{row['contrast']}/{row['metric']}")
        equal(row.get("positive_count"), len(positive_ids))
        equal(row.get("negative_count"), len(negative_ids))
        equal(row.get("neutral_count"), len(neutral_ids))
        equal(row.get("neutral_tolerance"), EXPECTED_SIGN_EPSILON)
        expected_values = {
            "mean_delta": math.fsum(values) / len(values) if values else None,
            "mean_delta_pp": math.fsum(values) * 100 / len(values) if values else None,
            "median_delta_pp": statistics.median(values) * 100 if values else None,
            "min_delta_pp": min(values) * 100 if values else None,
            "max_delta_pp": max(values) * 100 if values else None,
        }
        for key, value in expected_values.items():
            equal(row.get(key), value)
        equal(row.get("delta_le_minus_5pp_count"), sum(x <= -0.05 for x in values))
        equal(row.get("delta_ge_plus_5pp_count"), sum(x >= 0.05 for x in values))
        ci = row.get("bootstrap_ci95_delta_pp")
        if values:
            seed_text = f"{EXPECTED_BOOTSTRAP_SEED}|{row['dataset']}|{row['contrast']}|{row['metric']}"
            rng = random.Random(int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big"))
            boot = sorted(math.fsum(values[rng.randrange(len(values))]
                                    for _ in values) / len(values)
                                    for _ in range(EXPECTED_BOOTSTRAP_REPLICATES))
            lower_index = int(0.025 * (EXPECTED_BOOTSTRAP_REPLICATES - 1))
            upper_index = int(0.975 * (EXPECTED_BOOTSTRAP_REPLICATES - 1))
            _need(isinstance(ci, Mapping) and set(ci) == {"lower", "upper"},
                  "paired bootstrap interval schema differs")
            equal(ci["lower"], boot[lower_index] * 100)
            equal(ci["upper"], boot[upper_index] * 100)
        else:
            _need(ci is None, "empty paired contrast must keep its interval unknown")

    aggregation = summary.get("aggregation")
    _need(isinstance(aggregation, Mapping) and
          aggregation.get("bootstrap_replicates") == scorer.BOOTSTRAP_REPLICATES and
          aggregation.get("bootstrap_replicates") == EXPECTED_BOOTSTRAP_REPLICATES and
          aggregation.get("seed") == EXPECTED_BOOTSTRAP_SEED and
          aggregation.get("unit") == "paired source case" and
          aggregation.get("pooled_cross_domain_gate") is False and
          aggregation.get("ci_used_for_arm_or_step_selection") is False,
          "summary aggregation contract differs from scorer")
    return {"checks_passed": checks, "max_scalar_abs_error": max_error,
            "row_count": len(rows), "arm_summary_count": len(arm_keys),
            "contrast_count": len(contrast_keys)}


def _expect_reject(fn) -> None:
    try:
        fn()
    except (AssertionError, KeyError, TypeError, ValueError):
        return
    raise AssertionError("synthetic invalid binding/schema unexpectedly passed")


def _synthetic_contracts() -> dict[str, Any]:
    """Use writer-shaped rows/summary/completion and test false bindings."""
    from scripts import score_stvg_native_instance_postseal_v1 as scorer

    case_ids = [f"hc2_{i:03d}" for i in range(4)] + [f"vidstg_{i:03d}" for i in range(4)]
    digest = lambda ch: ch * 64
    lock = {
        "case_ids": case_ids,
        "seal_sha256": digest("1"),
        "roster_sha256": digest("2"),
        "reference_lock_sha256": digest("3"),
        "direct_readout_sha256": digest("4"),
        "phase_hashes": {"TRACKING": digest("5"), "TEACHER": digest("6"), "STUDENT": digest("7")},
        "roster_by_case": {cid: {"dataset": "hc2" if cid.startswith("hc2") else "vidstg",
                                  "source_case_sha256": digest("8")}
                           for cid in case_ids},
        "ref_by_case": {cid: {"teacher_record_sha256": digest("9"),
                               "teacher_receipt_sha256": digest("a"),
                               "anchors": [{"position": 1}, {"position": 4}]}
                        for cid in case_ids},
    }
    row_bindings = {cid: {"source_case_sha256": digest("8"),
                          "student_artifact_sha256": digest("b"),
                          "student_receipt_sha256": digest("c"),
                          "teacher_record_sha256": digest("9"),
                          "teacher_receipt_sha256": digest("a"),
                          "reference_lock_sha256": digest("3")}
                    for cid in case_ids}
    support = {
        "source_grid_positions": 10,
        "source_grid_supervised_position_count": 2,
        "source_grid_unsupervised_position_count": 8,
        "supervised_anchor_count": 2,
        "unsupervised_position_count": 8,
        "dense_gt_known_frame_count": 5,
        "source_grid_gt_known_frame_count": 4,
        "off_grid_gt_known_frame_count": 1,
        "supervised_gt_known_frame_count": 1,
        "unsupervised_gt_known_frame_count": 4,
        "supervised_positions": [1, 4],
        "supervised_physical_frame_ids": [101],
        "unsupervised_physical_frame_ids": [102, 103, 104, 105],
        "gt_support_split": "supervised=known GT physical IDs at reference anchors; unsupervised=all other known GT physical IDs, including off-grid frames",
    }
    fake_rows = []
    teacher_status = {"hc2": [], "vidstg": []}
    for cid in case_ids:
        dataset = lock["roster_by_case"][cid]["dataset"]
        i = int(cid[-3:])
        for ai, arm in enumerate(scorer.ARMS):
            unavailable_teacher = arm == "native_teacher_direct" and cid == "hc2_003"
            metrics = {metric: (None if unavailable_teacher else
                                (None if metric == "m_sIoU_supervised_GT" and cid == "vidstg_003"
                                 else .3 + i * .01 + ai * .002))
                       for metric in scorer.METRICS}
            fake_rows.append({"case_id": cid, "dataset": dataset, "arm": arm,
                              "bindings": dict(row_bindings[cid]), "support": dict(support),
                              "metric_audit": {"official_independent_max_abs_error": 0.0},
                              "metrics": metrics,
                              "status": ("invalid_or_abstained_no_teacher_tube" if unavailable_teacher
                                         else "scored")})
        teacher_status[dataset].append({
            "case_id": cid, "selection_status": "invalid" if cid == "hc2_003" else "valid",
            "valid_uncapped_selection": cid != "hc2_003",
            "coverage_fraction": None if cid == "hc2_003" else .8,
            "present_run_count": None if cid == "hc2_003" else 1,
        })
    common = _expected_common_bindings(lock)
    barrier = {"path": str(scorer.SEAL.relative_to(ROOT)), "sha256": lock["seal_sha256"]}
    summary = scorer._summarize(fake_rows, teacher_status)
    document = {"schema": scorer.ROWS_SCHEMA, "status": "complete", "GT_read": True,
                "prediction_barrier": dict(barrier), "bindings": dict(common),
                "metric_scale": "0_to_1", "case_count": 8,
                "arm_count": len(scorer.ARMS), "rows": fake_rows}
    summary.update({"prediction_barrier": dict(barrier), "bindings": dict(common),
                    "official_metric_contract_schema": "synthetic",
                    "official_truth_binding_digest": digest("d")})
    completion = {"schema": "stvg-native-instance-cpu-completion-v1",
                  "status": "pending_root_review_visualization_publication",
                  "scoring_status": "complete", "experiment_complete": False,
                  "GT_read": True, "case_count": 8, "arm_count": len(scorer.ARMS),
                  "row_count": len(fake_rows), "prediction_barrier_sha256": lock["seal_sha256"],
                  "rows_sha256": _canonical_sha(document),
                  "summary_sha256": _canonical_sha(summary)}
    math_receipt = {"schema": "stvg-native-instance-independent-student-math-readback-v1",
                    "status": "pass", "GT_read": False, "CPU_only": True,
                    "GPU_initialized": False, "model_replayed_by_reader": False,
                    "case_count": 8,
                    "student_barrier_sha256": lock["phase_hashes"]["STUDENT"],
                    "reference_lock_sha256": lock["reference_lock_sha256"],
                    "roster_sha256": lock["roster_sha256"]}
    valid = _validate_documents(lock, scorer, document, summary, completion,
                                math_receipt, row_bindings)

    bad_completion = copy.deepcopy(completion)
    bad_completion["rows_sha256"] = digest("e")
    _expect_reject(lambda: _validate_documents(lock, scorer, document, summary,
                                               bad_completion, math_receipt, row_bindings))

    bad_barrier = copy.deepcopy(document)
    bad_barrier["prediction_barrier"]["sha256"] = digest("f")
    _expect_reject(lambda: _validate_documents(lock, scorer, bad_barrier, summary,
                                               completion, math_receipt, row_bindings))

    stale_math = dict(math_receipt, student_barrier_sha256=digest("e"))
    _expect_reject(lambda: _validate_documents(lock, scorer, document, summary,
                                               completion, stale_math, row_bindings))

    bad_row = copy.deepcopy(document)
    bad_row["rows"][0]["bindings"]["teacher_receipt_sha256"] = digest("f")
    bad_completion2 = dict(completion, rows_sha256=_canonical_sha(bad_row))
    _expect_reject(lambda: _validate_documents(lock, scorer, bad_row, summary,
                                               bad_completion2, math_receipt, row_bindings))

    bad_dataset = copy.deepcopy(document)
    bad_dataset["rows"][0]["dataset"] = "vidstg"
    bad_completion_dataset = dict(completion, rows_sha256=_canonical_sha(bad_dataset))
    _expect_reject(lambda: _validate_documents(lock, scorer, bad_dataset, summary,
                                               bad_completion_dataset, math_receipt, row_bindings))

    bad_unknown = copy.deepcopy(summary)
    arm_unknown = next(r for r in bad_unknown["arm_metric_summaries"]
                       if r["dataset"] == "hc2" and r["arm"] == "native_teacher_direct"
                       and r["metric"] == "m_vIoU")
    arm_unknown["unknown_case_ids"] = ["hc2_002"]
    bad_completion3 = dict(completion, summary_sha256=_canonical_sha(bad_unknown))
    _expect_reject(lambda: _validate_documents(lock, scorer, document, bad_unknown,
                                               bad_completion3, math_receipt, row_bindings))

    bad_tail = copy.deepcopy(summary)
    positive = next(r for r in bad_tail["contrasts"]
                    if r["dataset"] == "hc2" and r["contrast"] == "T_minus_F"
                    and r["metric"] == "m_vIoU")
    positive["positive_case_ids"] = ["hc2_001", "hc2_002", "hc2_003", "hc2_000"]
    bad_completion_tail = dict(completion, summary_sha256=_canonical_sha(bad_tail))
    _expect_reject(lambda: _validate_documents(lock, scorer, document, bad_tail,
                                               bad_completion_tail, math_receipt, row_bindings))

    bad_catalog = copy.deepcopy(summary)
    bad_catalog["arm_metric_summaries"].pop()
    bad_catalog["arm_metric_summaries"].append(copy.deepcopy(bad_catalog["arm_metric_summaries"][0]))
    bad_completion4 = dict(completion, summary_sha256=_canonical_sha(bad_catalog))
    _expect_reject(lambda: _validate_documents(lock, scorer, document, bad_catalog,
                                               bad_completion4, math_receipt, row_bindings))

    bad_contrast_catalog = copy.deepcopy(summary)
    bad_contrast_catalog["contrasts"].pop()
    bad_contrast_catalog["contrasts"].append(copy.deepcopy(bad_contrast_catalog["contrasts"][0]))
    bad_completion_contrast = dict(completion,
                                   summary_sha256=_canonical_sha(bad_contrast_catalog))
    _expect_reject(lambda: _validate_documents(lock, scorer, document, bad_contrast_catalog,
                                               bad_completion_contrast, math_receipt, row_bindings))

    bad_support = copy.deepcopy(document)
    bad_support["rows"][1]["support"]["off_grid_gt_known_frame_count"] += 1
    bad_completion5 = dict(completion, rows_sha256=_canonical_sha(bad_support))
    _expect_reject(lambda: _validate_documents(lock, scorer, bad_support, summary,
                                               bad_completion5, math_receipt, row_bindings))
    return {"schema": "stvg-native-instance-postseal-reader-contract-v1",
            "status": "pass", "CPU_only": True,
            "valid_fixture": valid,
            "false_binding_checks": ["completion canonical row digest", "prediction barrier binding",
                                     "student math barrier binding", "teacher receipt row binding",
                                     "unknown case membership", "positive tail membership",
                                     "summary arm and contrast key completeness", "support partition"],
            "actual_score_or_GT_read": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", action="store_true",
                        help="run in-memory writer-shaped CPU contracts only")
    args = parser.parse_args()
    if args.contracts:
        print(json.dumps(_synthetic_contracts(), sort_keys=True))
        return

    from scripts import score_stvg_native_instance_postseal_v1 as scorer
    lock = scorer._load_locked_inputs()
    document = read(BASE / "POSTSEAL_SCORE_ROWS.json")
    summary = read(BASE / "POSTSEAL_SCORE_SUMMARY.json")
    completion = read(BASE / "CPU_COMPLETION.json")
    math_receipt = read(BASE / "ROOT_STUDENT_MATH_READBACK.json")
    row_bindings = _expected_row_bindings(lock)
    result = _validate_documents(lock, scorer, document, summary, completion,
                                 math_receipt, row_bindings)
    receipt = dict(schema="stvg-native-instance-root-postseal-readback-v1", status="pass",
        GT_read=True, raw_GT_read_by_this_reader=False, CPU_only=True,
        prediction_barrier_sha256=lock["seal_sha256"],
        rows_sha256=sha(BASE / "POSTSEAL_SCORE_ROWS.json"),
        summary_sha256=sha(BASE / "POSTSEAL_SCORE_SUMMARY.json"),
        completion_sha256=sha(BASE / "CPU_COMPLETION.json"),
        case_count=8, row_count=result["row_count"],
        arm_summary_count=result["arm_summary_count"], contrast_count=result["contrast_count"],
        checks_passed=result["checks_passed"], checks_failed=0,
        max_metric_scalar_abs_error=result["max_scalar_abs_error"],
        all_case_source_bindings_match=True, all_tails_and_unknown_membership_match=True,
        model_replayed=False,
        verification_scope="independent scalar aggregation, exact score/completion bindings, support partitions, and tail membership; official/independent geometric metrics were checked by the sealed scorer",
        reader_sha256=sha(Path(__file__)), time=time.time())
    write(BASE / "ROOT_POSTSEAL_READBACK.json", receipt)
    print(json.dumps({"status": "pass", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
