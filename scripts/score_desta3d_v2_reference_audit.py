"""Independent CPU readback for sealed DESTA-3D v2 reference-audit runs.

This scorer intentionally does not import the source-fit worker scorer or
``vg_tta.metrics``. It refuses pilot, incomplete, unsealed, or identity-mismatched
runs before opening source labels. Run it only after the full 198-query runner
has written COMPLETE.json and PREDICTIONS_SEAL.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ART = ROOT / "artifacts/desta3d_v1"
SOURCE_FIT = ROOT / "artifacts/desta3d_v2/source_fit"
ARMS = ("independent_reference", "shared_reference_time")
METRICS = ("vIoU", "sIoU", "tIoU")
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260927


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _finite_float(value: Any, context: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"nonfinite value in {context}: {value!r}")
    return out


def iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(x) for x in a)
    bx1, by1, bx2, by2 = (float(x) for x in b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, 1e-7)


def cxcywh_to_xyxy(box: Iterable[float]) -> list[float]:
    cx, cy, width, height = (float(x) for x in box)
    return [cx - width / 2.0, cy - height / 2.0,
            cx + width / 2.0, cy + height / 2.0]


def score_tube_independently(prediction: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    """Score normalized sparse cxcywh output using scalar physical-time math.

    Predicted intervals are inclusive sampled-position pairs and map to
    ``[frame_ids[start], frame_ids[end] + 1)``. Ground-truth event intervals are
    physical-frame half-open intervals. Missing predicted boxes are zero boxes.
    """
    frame_ids = [int(x) for x in prediction["frame_ids"]]
    label_frame_ids = [int(x) for x in label["frame_ids"]]
    if frame_ids != label_frame_ids:
        raise ValueError("prediction/label sampled frame IDs differ")
    if any(b <= a for a, b in zip(frame_ids, frame_ids[1:])):
        raise ValueError("sampled physical frame IDs must be strictly increasing")
    t = len(frame_ids)
    if t == 0:
        raise ValueError("empty sampled frame sequence")
    if not (len(label["box_valid"]) == len(label["event_active"]) == len(label["boxes_xyxy"]) == t):
        raise ValueError("source label frame arrays have different lengths")
    support = [bool(valid) and bool(active)
               for valid, active in zip(label["box_valid"], label["event_active"])]
    support_count = int(sum(support))

    raw_boxes = prediction.get("boxes_cxcywh")
    boxes = _as_list(raw_boxes) if raw_boxes is not None else []
    if boxes and (not isinstance(boxes[0], (list, tuple))):
        raise ValueError("boxes_cxcywh must be a two-dimensional [N,4] array")
    if any(len(box) != 4 for box in boxes):
        raise ValueError("each cxcywh box must contain four coordinates")
    boxes = [[_finite_float(x, "boxes_cxcywh") for x in box] for box in boxes]
    positions = [int(x) for x in (prediction.get("positions") or [])]
    geometry_value = prediction.get("geometry_valid")
    geometry = None if geometry_value is None else [bool(x) for x in _as_list(geometry_value)]
    if len(positions) != len(boxes) or len(set(positions)) != len(positions):
        raise ValueError("sparse PTD positions and boxes must be unique and aligned")
    if any(pos < 0 or pos >= t for pos in positions):
        raise ValueError("sparse PTD position is outside the sampled sequence")
    if geometry is not None and len(geometry) != len(boxes):
        raise ValueError("geometry_valid and sparse boxes are not aligned")

    interval = prediction.get("interval")
    valid_interval = (isinstance(interval, (list, tuple)) and len(interval) == 2
                      and 0 <= int(interval[0]) <= int(interval[1]) < t)
    accepted_format = bool(prediction.get("format_ok", False)) and valid_interval
    if not accepted_format:
        return {
            "vIoU": 0.0, "sIoU": 0.0, "tIoU": 0.0,
            "format_ok": False, "format_declared": bool(prediction.get("format_ok", False)),
            "spatial_support_frames": support_count,
            "temporal_union_sample_count": 0,
            "predicted_interval_physical": None,
            "gt_interval_physical": [int(label["event_interval"]["begin_fid"]),
                                     int(label["event_interval"]["end_fid"])],
        }

    dense_xyxy = [[0.0, 0.0, 0.0, 0.0] for _ in range(t)]
    for source_index, (position, box) in enumerate(zip(positions, boxes)):
        if geometry is None or geometry[source_index]:
            dense_xyxy[position] = cxcywh_to_xyxy(box)

    gt_xyxy = [[_finite_float(x, "GT boxes_xyxy") for x in box]
               for box in label["boxes_xyxy"]]
    frame_iou = {
        index: iou_xyxy(dense_xyxy[index], gt_xyxy[index])
        for index, has_support in enumerate(support) if has_support
    }
    siou = sum(frame_iou.values()) / max(support_count, 1)

    predicted_start = frame_ids[int(interval[0])]
    predicted_end = frame_ids[int(interval[1])] + 1
    gt_start = int(label["event_interval"]["begin_fid"])
    gt_end = int(label["event_interval"]["end_fid"])
    if gt_end < gt_start:
        raise ValueError("ground-truth half-open event interval is reversed")
    intersection_start = max(predicted_start, gt_start)
    intersection_end = min(predicted_end, gt_end)
    union_start = min(predicted_start, gt_start)
    union_end = max(predicted_end, gt_end)
    union_samples = sum(union_start <= fid < union_end for fid in frame_ids)
    intersection_iou_sum = sum(
        value for index, value in frame_iou.items()
        if intersection_start <= frame_ids[index] < intersection_end
    )
    tiou = max(0, intersection_end - intersection_start) / max(union_end - union_start, 1)
    viou = intersection_iou_sum / max(union_samples, 1)
    return {
        "vIoU": float(viou), "sIoU": float(siou), "tIoU": float(tiou),
        "format_ok": True, "format_declared": bool(prediction.get("format_ok", False)),
        "spatial_support_frames": support_count,
        "temporal_union_sample_count": int(union_samples),
        "intersection_iou_sum": float(intersection_iou_sum),
        "predicted_interval_physical": [int(predicted_start), int(predicted_end)],
        "gt_interval_physical": [int(gt_start), int(gt_end)],
    }


def binary_auc(labels: Iterable[bool], scores: Iterable[float]) -> float | None:
    y = np.asarray(list(labels), dtype=np.bool_).reshape(-1)
    values = np.asarray(list(scores), dtype=np.float64).reshape(-1)
    if len(y) != len(values) or not np.isfinite(values).all():
        raise ValueError("AUROC labels and finite scores must be aligned")
    positive = int(y.sum())
    negative = int(len(y) - positive)
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[y].sum())
    return float((positive_rank_sum - positive * (positive + 1) / 2.0)
                 / (positive * negative))


def binary_bce(labels: Iterable[bool], logits: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=np.float64).reshape(-1)
    values = np.asarray(list(logits), dtype=np.float64).reshape(-1)
    if len(y) != len(values) or not len(y) or not np.isfinite(values).all():
        raise ValueError("BCE labels and finite logits must be nonempty and aligned")
    return float(np.mean(np.logaddexp(0.0, values) - y * values))


def average_ranks(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def correlation(x: Iterable[float], y: Iterable[float]) -> dict[str, Any]:
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("correlation vectors have different shapes")
    n = int(len(x))
    if n < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return {"n": n, "pearson_r": None, "spearman_r": None}
    return {
        "n": n,
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "spearman_r": float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1]),
    }


def paired_parent_bootstrap(
    first: dict[str, dict[str, float]],
    second: dict[str, dict[str, float]],
    metric: str,
) -> dict[str, Any]:
    parents = sorted(first)
    if parents != sorted(second) or not parents:
        raise ValueError("paired parent metric rosters differ or are empty")
    deltas = np.asarray([first[p][metric] - second[p][metric] for p in parents], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, len(parents), size=(BOOTSTRAP_REPS, len(parents)))
    boot = deltas[sampled].mean(axis=1)
    return {
        "sign": "first minus second",
        "metric": metric,
        "parents": len(parents),
        "parent_delta_pp": {p: float(deltas[i] * 100.0) for i, p in enumerate(parents)},
        "mean_delta_pp": float(deltas.mean() * 100.0),
        "bootstrap_ci95_pp": [float(x) for x in np.quantile(boot, [0.025, 0.975]) * 100.0],
        "positive_parents": int((deltas > 1e-10).sum()),
        "negative_parents": int((deltas < -1e-10).sum()),
        "zero_parents": int((np.abs(deltas) <= 1e-10).sum()),
        "severe_loss_below_minus5pp": int((deltas < -0.05).sum()),
        "worst_delta_pp": float(deltas.min() * 100.0),
        "best_delta_pp": float(deltas.max() * 100.0),
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_rng": "numpy default_rng; paired source-parent resampling with replacement",
    }


def summarize_parents(query_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        grouped[str(row["source"])].append(row)
    return {
        source: {metric: float(np.mean([r["metrics"][metric] for r in rows]))
                 for metric in METRICS}
        for source, rows in sorted(grouped.items())
    }


def summarize_arm(query_rows: list[dict[str, Any]], parents: dict[str, dict[str, float]]) -> dict[str, Any]:
    query_macro = {m: float(np.mean([r["metrics"][m] for r in query_rows])) for m in METRICS}
    parent_macro = {m: float(np.mean([p[m] for p in parents.values()])) for m in METRICS}
    return {
        "queries": len(query_rows), "parents": len(parents),
        "query_macro": query_macro, "parent_macro": parent_macro,
        "parent_rows": [{"source": source, **values} for source, values in sorted(parents.items())],
        "no_spatial_support_queries_included_with_zero_vIoU": sum(
            r["metrics"]["spatial_support_frames"] == 0 for r in query_rows
        ),
        "spatial_support_queries": sum(r["metrics"]["spatial_support_frames"] > 0 for r in query_rows),
        "spatial_support_frames": sum(r["metrics"]["spatial_support_frames"] for r in query_rows),
        "format_failures": sum(not r["metrics"]["format_ok"] for r in query_rows),
    }


def retention_against_frozen(
    predictions: dict[str, dict[str, dict[str, Any]]],
    frozen_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    good_v = {k for k, row in frozen_rows.items() if float(row["metrics"]["vIoU"]) > 0.5}
    good_t = {k for k, row in frozen_rows.items() if float(row["metrics"]["tIoU"]) > 0.5}

    def summarize(group: set[str], values: list[dict[str, Any]], metric: str) -> dict[str, Any]:
        by_key = {row["key"]: row for row in values}
        if not group.issubset(by_key):
            raise ValueError("candidate predictions do not cover Frozen-good subset")
        retained = {k for k in group if float(by_key[k]["metrics"][metric]) > 0.5}
        parents: dict[str, list[str]] = defaultdict(list)
        for key in group:
            parents[str(by_key[key]["source"])].append(key)
        parent_rates = {
            source: sum(k in retained for k in keys) / len(keys)
            for source, keys in parents.items()
        }
        return {
            "frozen_good_queries": len(group), "frozen_good_parents": len(parents),
            "retained_queries": len(retained), "lost_queries": sorted(group - retained),
            "query_retention_rate": len(retained) / len(group) if group else None,
            "parent_macro_retention_rate": float(np.mean(list(parent_rates.values()))) if parent_rates else None,
            "parent_rows": [{"source": s, "eligible_queries": len(parents[s]),
                             "retained_queries": sum(k in retained for k in parents[s]),
                             "retention_rate": parent_rates[s]} for s in sorted(parents)],
        }

    result: dict[str, Any] = {
        "definitions": {
            "Frozen_full_tube_good": "Frozen independently scored query vIoU > 0.5; candidate retained if its vIoU > 0.5.",
            "Frozen_native_temporal_good": "Frozen independently scored query tIoU > 0.5; candidate retained if its tIoU > 0.5.",
            "parent_macro_retention": "For each eligible parent, retained Frozen-good queries divided by that parent's Frozen-good queries; then macro-averaged over eligible parents.",
        },
        "frozen_full_tube_vIoU_gt_0_5": {"eligible_queries": len(good_v), "eligible_parents": len({frozen_rows[k]['source'] for k in good_v})},
        "frozen_native_temporal_tIoU_gt_0_5": {"eligible_queries": len(good_t), "eligible_parents": len({frozen_rows[k]['source'] for k in good_t})},
        "arms": {},
    }
    for arm, rows in predictions.items():
        result["arms"][arm] = {
            "full_tube_vIoU_gt_0_5": summarize(good_v, rows, "vIoU"),
            "native_temporal_tIoU_gt_0_5": summarize(good_t, rows, "tIoU"),
        }
    return result


def _rank_quadrants(points: list[dict[str, Any]], *, auc_field: str, tiou_field: str) -> dict[str, Any]:
    names = ("AUROC>0.5__tIoU>0.5", "AUROC>0.5__tIoU<=0.5",
             "AUROC<=0.5__tIoU>0.5", "AUROC<=0.5__tIoU<=0.5")
    groups = {name: [] for name in names}
    undefined = []
    for point in points:
        auc = point[auc_field]
        if auc is None:
            undefined.append(point)
            continue
        key = ("AUROC>0.5" if auc > 0.5 else "AUROC<=0.5") + "__" + (
            "tIoU>0.5" if point[tiou_field] > 0.5 else "tIoU<=0.5"
        )
        groups[key].append(point)
    return {
        "defined_AUROC_denominator": sum(len(v) for v in groups.values()),
        "undefined_AUROC_count": len(undefined),
        "counts": {name: len(groups[name]) for name in names},
        "members": {name: [p.get("key", p.get("source")) for p in groups[name]] for name in names},
        "undefined_ids": [p.get("key", p.get("source")) for p in undefined],
    }


def summarize_event_endpoint(query_points: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_points:
        by_source[str(row["source"])].append(row)
    source_points = []
    for source, rows in sorted(by_source.items()):
        logits = [x for r in rows for x in r["_logits"]]
        labels = [x for r in rows for x in r["_labels"]]
        auc = binary_auc(labels, logits)
        source_points.append({
            "source": source,
            "temporal_tIoU": float(np.mean([r["temporal_tIoU"] for r in rows])),
            "event_frame_AUROC": auc,
            "event_frame_BCE": binary_bce(labels, logits),
            "queries": len(rows), "frames": len(labels),
            "positive_frames": int(sum(labels)), "negative_frames": int(len(labels) - sum(labels)),
            "undefined_AUROC_reason": "single_class_source" if auc is None else None,
        })
    q_defined = [r for r in query_points if r["event_frame_AUROC"] is not None]
    s_defined = [r for r in source_points if r["event_frame_AUROC"] is not None]
    return {
        "query_points": [{k: v for k, v in r.items() if not k.startswith("_")} for r in query_points],
        "source_points": source_points,
        "correlations": {
            "query_AUROC_vs_tIoU": correlation([r["event_frame_AUROC"] for r in q_defined], [r["temporal_tIoU"] for r in q_defined]),
            "query_BCE_vs_tIoU": correlation([r["event_frame_BCE"] for r in query_points], [r["temporal_tIoU"] for r in query_points]),
            "source_AUROC_vs_parent_mean_tIoU": correlation([r["event_frame_AUROC"] for r in s_defined], [r["temporal_tIoU"] for r in s_defined]),
            "source_BCE_vs_parent_mean_tIoU": correlation([r["event_frame_BCE"] for r in source_points], [r["temporal_tIoU"] for r in source_points]),
        },
        "four_quadrants": {
            "threshold_note": "AUROC > 0.5 is above-random ranking only; temporal tIoU > 0.5 is a descriptive split, not a reliability or selection gate.",
            "query": _rank_quadrants(query_points, auc_field="event_frame_AUROC", tiou_field="temporal_tIoU"),
            "source": _rank_quadrants(source_points, auc_field="event_frame_AUROC", tiou_field="temporal_tIoU"),
        },
        "denominators": {
            "queries": len(query_points), "query_AUROC_defined": len(q_defined),
            "query_AUROC_undefined": len(query_points) - len(q_defined),
            "sources": len(source_points), "source_AUROC_defined": len(s_defined),
            "source_AUROC_undefined": len(source_points) - len(s_defined),
            "frames": sum(r["frames"] for r in query_points),
            "positive_frames": sum(r["positive_frames"] for r in query_points),
            "negative_frames": sum(r["negative_frames"] for r in query_points),
        },
        "parent_macro_event_AUROC": float(np.mean([r["event_frame_AUROC"] for r in s_defined])) if s_defined else None,
        "parent_macro_event_BCE": float(np.mean([r["event_frame_BCE"] for r in source_points])) if source_points else None,
        "global_frame_AUROC": binary_auc([x for r in query_points for x in r["_labels"]], [x for r in query_points for x in r["_logits"]]),
        "global_frame_BCE": binary_bce([x for r in query_points for x in r["_labels"]], [x for r in query_points for x in r["_logits"]]),
    }


def summarize_injection(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        result[arm] = {}
        for branch in ("event_injection", "spatial_injection"):
            records = []
            for row in diagnostics:
                injection = row[arm].get(branch)
                if injection is None:
                    records.append({"available": False})
                    continue
                ratio = _finite_float(injection["relative_injection_norm"], f"{arm}/{branch}/ratio")
                changed = int(injection["changed_elements"])
                elements = int(injection["elements"])
                if changed < 0 or elements < changed:
                    raise ValueError(f"invalid cast-change count for {arm}/{branch}")
                records.append({
                    "available": True, "calls": int(injection["calls"]),
                    "zero_exact": bool(injection["zero_exact"]),
                    "relative_injection_norm": ratio,
                    "changed_elements": changed, "elements": elements,
                    "cast_changed_fraction": changed / elements if elements else None,
                })
            available = [r for r in records if r["available"]]
            ratios = [r["relative_injection_norm"] for r in available]
            cast_fractions = [r["cast_changed_fraction"] for r in available if r["cast_changed_fraction"] is not None]
            result[arm][branch] = {
                "query_records": records,
                "available_queries": len(available),
                "missing_queries": len(records) - len(available),
                "zero_exact_queries": sum(r["zero_exact"] for r in available),
                "nonzero_cast_changed_queries": sum(r["changed_elements"] > 0 for r in available),
                "changed_elements_total": sum(r["changed_elements"] for r in available),
                "elements_total": sum(r["elements"] for r in available),
                "relative_injection_norm": {
                    "mean": float(np.mean(ratios)) if ratios else None,
                    "median": float(np.median(ratios)) if ratios else None,
                    "p95": float(np.quantile(ratios, .95)) if ratios else None,
                    "min": float(min(ratios)) if ratios else None,
                    "max": float(max(ratios)) if ratios else None,
                },
                "cast_changed_fraction": {
                    "mean": float(np.mean(cast_fractions)) if cast_fractions else None,
                    "median": float(np.median(cast_fractions)) if cast_fractions else None,
                },
            }
    return result


def summarize_same_prefix_residual_control(
    diagnostics: list[dict[str, Any]],
    shared_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute cached-v3 KL and coordinate deltas from sealed logits.

    The pinned runner executes the with-residual and gate-zero spatial paths
    from the same event-selected interval/reference prefix. The seal retains
    the spatial reference, interval, event completion, and both logit tensors;
    this function treats the runner's saved scalar KL/change fields only as
    values to cross-check, never as the scored measurements.
    """
    if len(diagnostics) != len(shared_predictions):
        raise ValueError("diagnostic and shared prediction rosters differ")
    rows = []
    unavailable = []
    for index, diag in enumerate(diagnostics):
        shared = diag.get("shared_reference_time", {})
        control = shared.get("same_prefix_residual_control")
        if not isinstance(control, dict) or "KL_with_to_without" not in control:
            unavailable.append(index)
            continue
        if control.get("fixed_reference_time") is not True:
            raise ValueError(f"same-prefix residual control did not freeze event reference/time at query index {index}")
        if not shared.get("event_completion") or not shared_predictions[index].get("event_completion"):
            raise ValueError(f"same-prefix control has no saved shared event prefix at query index {index}")
        if shared.get("event_completion") != shared_predictions[index].get("event_completion"):
            raise ValueError(f"same-prefix event completion differs from saved shared prediction at query index {index}")
        with_logits_value = shared.get("coordinate_logits")
        without_logits = control.get("without_logits")
        if with_logits_value is None or without_logits is None:
            raise ValueError(f"same-prefix control omitted with/without coordinate logits at query index {index}")
        with_logits = torch.as_tensor(with_logits_value, dtype=torch.float32, device="cpu")
        without_logits = torch.as_tensor(without_logits, dtype=torch.float32, device="cpu")
        if with_logits.ndim < 2 or with_logits.shape != without_logits.shape:
            raise ValueError(f"same-prefix coordinate-logit shapes differ at query index {index}: {tuple(with_logits.shape)} vs {tuple(without_logits.shape)}")
        if with_logits.shape[-1] != 1001:
            raise ValueError(f"expected cached PTD 1001-coordinate vocabulary, got {with_logits.shape[-1]} at query index {index}")
        if not torch.isfinite(with_logits).all() or not torch.isfinite(without_logits).all():
            raise ValueError(f"nonfinite with/without coordinate logits at query index {index}")
        # Recompute from the sealed tensors on CPU; do not use the runner's KL
        # scalar as the result. Keep its arithmetic direction KL(with || zero).
        with_log_probs = torch.log_softmax(with_logits, dim=-1)
        without_log_probs = torch.log_softmax(without_logits, dim=-1)
        recomputed_kl = float((with_log_probs.exp() * (with_log_probs - without_log_probs)).sum(dim=-1).mean())
        recomputed_changed = int((with_logits != without_logits).sum())
        recomputed_max_abs = float((with_logits - without_logits).abs().max())
        stored_kl = _finite_float(control["KL_with_to_without"], "stored same_prefix_residual_control KL")
        stored_max_abs = _finite_float(control["max_abs_change"], "stored same_prefix_residual_control max_abs_change")
        stored_changed = int(control["logit_changed"])
        if stored_changed < 0 or stored_max_abs < 0:
            raise ValueError(f"invalid same-prefix logit-change record at query index {index}")
        if stored_changed != recomputed_changed:
            raise ValueError(f"stored vs recomputed changed-element count mismatch at query index {index}: {stored_changed} vs {recomputed_changed}")
        if not math.isclose(stored_max_abs, recomputed_max_abs, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"stored vs recomputed max-abs mismatch at query index {index}: {stored_max_abs} vs {recomputed_max_abs}")
        if not math.isclose(stored_kl, recomputed_kl, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"stored vs recomputed KL mismatch at query index {index}: {stored_kl} vs {recomputed_kl}")
        ref_tokens = shared.get("spatial_reference_token_ids")
        interval = shared_predictions[index].get("interval")
        rows.append({
            "query_index": index,
            "fixed_interval": interval,
            "fixed_shared_spatial_reference_token_count": None if ref_tokens is None else len(ref_tokens),
            "event_prefix_matches_saved_prediction": True,
            "fixed_reference_time": True,
            "coordinate_logits_shape": list(with_logits.shape),
            "KL_with_to_without": recomputed_kl,
            "stored_KL_with_to_without": stored_kl,
            "stored_vs_recomputed_KL_abs_diff": abs(stored_kl - recomputed_kl),
            "logit_changed": recomputed_changed,
            "stored_logit_changed": stored_changed,
            "max_abs_change": recomputed_max_abs,
            "stored_max_abs_change": stored_max_abs,
            "stored_vs_recomputed_max_abs_diff": abs(stored_max_abs - recomputed_max_abs),
            "format_without": bool(control.get("format_without", False)),
            "without_logits_shape": list(without_logits.shape),
        })

    def stats(values: list[float]) -> dict[str, float | None]:
        return {
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p95": float(np.quantile(values, .95)) if values else None,
            "min": float(min(values)) if values else None,
            "max": float(max(values)) if values else None,
        }

    return {
        "definition": "Same shared event-predicted interval and shared semantic reference; spatial coordinate logits with the spatial residual compared against a gate-zero spatial decode.",
        "queries_total": len(diagnostics),
        "queries_available": len(rows),
        "queries_unavailable": len(unavailable),
        "unavailable_query_indices": unavailable,
        "format_without_ok": sum(r["format_without"] for r in rows),
        "logit_changed_queries": sum(r["logit_changed"] > 0 for r in rows),
        "logit_changed_total": sum(r["logit_changed"] for r in rows),
        "stored_recomputed_KL_max_abs_diff": max((r["stored_vs_recomputed_KL_abs_diff"] for r in rows), default=None),
        "stored_recomputed_max_abs_max_abs_diff": max((r["stored_vs_recomputed_max_abs_diff"] for r in rows), default=None),
        "KL_with_to_without": stats([r["KL_with_to_without"] for r in rows]),
        "max_abs_change": stats([r["max_abs_change"] for r in rows]),
        "query_rows": rows,
        "raw_without_logits_retained_in_diagnostics_only": True,
    }


def _write_json_once(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DESTA-3D v2 shared-reference independent source readback",
        "",
        "Independent CPU scoring of the two sealed full-run arms. This is a source-validation engineering/reference audit, not target evaluation, TTA benefit, or a method-promotion decision.",
        "",
        f"Run: `{report['provenance']['run_dir']}`  ",
        f"Prediction seal SHA-256: `{report['provenance']['prediction_seal_sha256']}`  ",
        f"Verified files: {report['provenance']['sealed_prediction_and_diagnostic_files']} / {report['provenance']['sealed_prediction_and_diagnostic_files']}; source labels were read only after seal and identity checks.",
        "",
        "## Tube metrics",
        "",
        "| Arm | Parent-macro vIoU | Parent-macro sIoU | Parent-macro tIoU | Queries / parents | Format failures | No spatial support (retained at zero) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        s = report["arms"][arm]["summary"]
        pm = s["parent_macro"]
        lines.append(f"| {arm} | {pm['vIoU']:.6f} | {pm['sIoU']:.6f} | {pm['tIoU']:.6f} | {s['queries']} / {s['parents']} | {s['format_failures']} | {s['no_spatial_support_queries_included_with_zero_vIoU']} |")
    lines += ["", "## Paired source-parent contrasts", "", "Values are first minus second in percentage points; each contrast uses all 31 paired source parents, 10,000 bootstrap samples, seed 20260927.", "", "| Contrast | Metric | Δ pp | 95% paired-parent CI pp | Negative parents | Loss below −5 pp |", "|---|---|---:|---:|---:|---:|"]
    for name, result in report["paired_parent_comparisons"].items():
        for metric in METRICS:
            x = result[metric]
            ci = x["bootstrap_ci95_pp"]
            lines.append(f"| {name} | {metric} | {x['mean_delta_pp']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {x['negative_parents']} | {x['severe_loss_below_minus5pp']} |")
    lines += ["", "## Frozen-good retention", "", "Eligibility is defined from the matched independent Frozen source baseline; candidate retention requires the same metric to remain strictly above 0.5.", "", "| Arm | Frozen vIoU-good retained | Query retention | Parent-macro retention | Frozen tIoU-good retained | Query retention | Parent-macro retention |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        r = report["frozen_good_retention"]["arms"][arm]
        v = r["full_tube_vIoU_gt_0_5"]
        t = r["native_temporal_tIoU_gt_0_5"]
        lines.append(f"| {arm} | {v['retained_queries']} / {v['frozen_good_queries']} | {v['query_retention_rate']:.3f} | {v['parent_macro_retention_rate']:.3f} | {t['retained_queries']} / {t['frozen_good_queries']} | {t['query_retention_rate']:.3f} | {t['parent_macro_retention_rate']:.3f} |")
    lines += ["", "## Reference, format, and cast-injection checks", "", "`reference_diff` counts query pairs whose spatial semantic-reference token sequences differ. `format_ok` is the end-to-end saved prediction flag. Cast changes count elements that differ after conversion back to the model token dtype; injection ratios are relative residual norms before/through the recorded hook.", "", "| Arm | Format OK | Spatial reference diff | Event hook cast changed | Spatial hook cast changed | Mean event residual ratio | Mean spatial residual ratio |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        ref = report["reference_format_diagnostics"][arm]
        inj = report["injection_diagnostics"][arm]
        e = inj["event_injection"]; s = inj["spatial_injection"]
        ev_ratio = e["relative_injection_norm"]["mean"]
        sp_ratio = s["relative_injection_norm"]["mean"]
        ev_text = "undefined" if ev_ratio is None else f"{ev_ratio:.4g}"
        sp_text = "undefined" if sp_ratio is None else f"{sp_ratio:.4g}"
        lines.append(f"| {arm} | {ref['format_ok_queries']} / {ref['queries']} | {ref['reference_diff_queries']} / {ref['queries']} | {e['nonzero_cast_changed_queries']} / {e['available_queries']} | {s['nonzero_cast_changed_queries']} / {s['available_queries']} | {ev_text} | {sp_text} |")
    residual_control = report["same_prefix_residual_control"]
    kl = residual_control["KL_with_to_without"]
    max_abs = residual_control["max_abs_change"]
    lines += ["", "## Shared same-prefix residual control", "", residual_control["definition"], "", f"Available: {residual_control['queries_available']} / {residual_control['queries_total']} queries; unavailable: {residual_control['queries_unavailable']}. Gate-zero format OK: {residual_control['format_without_ok']} / {residual_control['queries_available']}; nonzero coordinate-logit change: {residual_control['logit_changed_queries']} / {residual_control['queries_available']} queries, {residual_control['logit_changed_total']} total elements.", "", f"KL(with residual || gate zero) mean/median/p95: {kl['mean']}, {kl['median']}, {kl['p95']}; max-absolute logit change mean/median/p95: {max_abs['mean']}, {max_abs['median']}, {max_abs['p95']}. Raw gate-zero logits remain in the sealed diagnostics; JSON retains per-query summaries, not duplicate raw logits.", ""]
    lines += ["", "## Direct event readout versus temporal endpoint", "", "Event AUROC/BCE uses the saved direct `[1,T]` event logits and source `event_active` labels on every sampled frame. AUROC is undefined for one-class query/source subsets; undefined examples and all scatter points are retained in JSON. AUROC > 0.5 means above-random ranking only; tIoU > 0.5 is descriptive. Neither is a reliability standard, an online/training gate, or causal evidence that event-head quality improves tube quality.", "", "| Arm | Source AUROC (defined parents / all) | Query AUROC (defined / all) | Source Pearson AUROC–tIoU | Query Pearson AUROC–tIoU | Query Pearson BCE–tIoU |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        event = report["arms"][arm]["event_endpoint"]
        den = event["denominators"]
        cor = event["correlations"]
        def fmt(v: Any) -> str:
            return "undefined" if v is None else f"{v:+.3f}"
        lines.append(f"| {arm} | {event['parent_macro_event_AUROC']:.4f} ({den['source_AUROC_defined']}/{den['sources']}) | {den['query_AUROC_defined']}/{den['queries']} | {fmt(cor['source_AUROC_vs_parent_mean_tIoU']['pearson_r'])} | {fmt(cor['query_AUROC_vs_tIoU']['pearson_r'])} | {fmt(cor['query_BCE_vs_tIoU']['pearson_r'])} |")
    lines += ["", "## Scope and files", "", "All 198 queries, all 31 source parents, and the five queries with no valid event-box support remain in the denominator. The five unsupported queries receive vIoU=0 under the registered dense-zero spatial rule. Metrics were computed with inline scalar geometry, not the source-fit worker scorer or `vg_tta.metrics`. Full query/source rows, direct event scatter points, paired deltas/CI, residual records, undefined AUC IDs, and audit checks are in the JSON and `AUDIT.json`.", "", "The AUROC/tIoU and retention thresholds above are descriptive readout conventions only. This readback does not select a checkpoint, open target labels, or establish target generalization.", ""]
    return "\n".join(lines)


def run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    # Fail closed on partial runs before reading pins, predictions, or labels.
    complete_path = run_dir / "COMPLETE.json"
    if not complete_path.exists():
        raise RuntimeError("COMPLETE.json is absent; refusing to read or score an incomplete run")
    complete = read_json(complete_path)
    if complete.get("status") != "completed" or complete.get("interface_pass") is not True:
        raise RuntimeError("run COMPLETE does not report a completed interface-passing run")
    if complete.get("source_val_GT_read") is not False or complete.get("target_GT_read") is not False:
        raise RuntimeError("COMPLETE must declare no source-validation or target label read")

    seal_path = run_dir / "PREDICTIONS_SEAL.json"
    if not seal_path.exists():
        raise RuntimeError("PREDICTIONS_SEAL.json is absent; refusing scoring")
    seal_sha = sha256_file(seal_path)
    seal = read_json(seal_path)
    if seal.get("source_val_GT_read") is not False or seal.get("target_GT_read") is not False:
        raise RuntimeError("runner seal must declare no source-validation or target label read")
    pins = seal.get("pins")
    if not isinstance(pins, dict) or not pins:
        raise RuntimeError("prediction seal has no file pins")
    # Hash every sealed file before loading any prediction payload.
    pin_failures = []
    for raw_path, expected_hash in pins.items():
        path = Path(raw_path).resolve()
        try:
            path.relative_to(run_dir)
        except ValueError as exc:
            raise RuntimeError(f"seal pin escapes run directory: {path}") from exc
        if not path.is_file():
            pin_failures.append({"path": str(path), "reason": "missing"})
        else:
            got = sha256_file(path)
            if got != expected_hash:
                pin_failures.append({"path": str(path), "expected": expected_hash, "actual": got})
    if pin_failures:
        raise RuntimeError(f"sealed prediction/diagnostic hashes do not match: {pin_failures[:3]}")

    # Only after the whole prediction seal passes do we inspect roster and actual
    # saved prediction/input identity. Source labels are still unopened here.
    config_path = run_dir / "CONFIG.json"
    inputs_path = run_dir / "INPUTS.json"
    lock_path = run_dir / "LOCK.json"
    config, inputs, lock = read_json(config_path), read_json(inputs_path), read_json(lock_path)
    if config.get("scope") != "full":
        raise RuntimeError("only the full 198-query run may be scored; pilot runs are excluded")
    if int(seal.get("queries", -1)) != 198 or len(inputs) != 198:
        raise RuntimeError("expected a full sealed 198-query run")
    rows = sorted(inputs, key=lambda row: row["key"])
    keys = [row["key"] for row in rows]
    sources = [str(row["source"]) for row in rows]
    if len(set(keys)) != 198 or len(set(sources)) != 31:
        raise RuntimeError("full run roster must contain 198 unique queries and 31 parents")
    if any(row.get("split") != "validation" for row in rows):
        raise RuntimeError("reference audit input roster includes a non-validation query")
    if list(config.get("arms", [])) != list(ARMS):
        raise RuntimeError("run configuration arms do not match the locked reference audit")
    if "arms" in seal and list(seal["arms"]) != list(ARMS):
        raise RuntimeError("prediction seal arms do not match the locked reference audit")

    expected_paths = set()
    for index in range(len(rows)):
        for arm in ARMS:
            expected_paths.add((run_dir / "predictions" / arm / f"{index:03}.pt").resolve())
        expected_paths.add((run_dir / "diagnostics" / f"{index:03}.pt").resolve())
    sealed_paths = {Path(p).resolve() for p in pins}
    if sealed_paths != expected_paths:
        missing = sorted(str(x) for x in expected_paths - sealed_paths)
        unexpected = sorted(str(x) for x in sealed_paths - expected_paths)
        raise RuntimeError(f"seal file roster differs: missing={missing[:2]} unexpected={unexpected[:2]}")
    if len(pins) != 3 * len(rows):
        raise RuntimeError(f"expected {3*len(rows)} sealed files, found {len(pins)}")

    lock_mismatches = []
    for raw_path, expected_hash in lock.get("pins", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected_hash:
            lock_mismatches.append(str(path))
    if lock_mismatches:
        raise RuntimeError(f"registered run dependency pins changed: {lock_mismatches[:5]}")
    checkpoint = config.get("checkpoint", {})
    checkpoint_path = Path(checkpoint.get("checkpoint", ""))
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint.get("sha256"):
        raise RuntimeError("configured source checkpoint does not match its locked SHA-256")
    expected_adapter_sha = checkpoint.get("adapter_sha256")
    if not isinstance(expected_adapter_sha, str) or len(expected_adapter_sha) != 64:
        raise RuntimeError("configured checkpoint has no valid adapter SHA-256")

    complete_checks = complete.get("checks")
    if not isinstance(complete_checks, list) or len(complete_checks) != 198:
        raise RuntimeError("COMPLETE does not contain the expected 198 per-query checks")
    complete_by_key = {c["key"]: c for c in complete_checks}
    if set(complete_by_key) != set(keys):
        raise RuntimeError("COMPLETE per-query keys do not match INPUTS")

    predictions: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    diagnostics: list[dict[str, Any]] = []
    input_identity_rows = []
    reference_rows = []
    prediction_hash_rows = []
    for index, row in enumerate(rows):
        key = row["key"]
        pair = {}
        preprocesses = []
        for arm in ARMS:
            path = run_dir / "predictions" / arm / f"{index:03}.pt"
            prediction = torch.load(path, map_location="cpu", weights_only=False)
            expected_fields = {"key", "source", "frame_ids", "positions", "boxes_cxcywh", "geometry_valid",
                               "interval", "format_ok", "event_completion", "spatial_completion", "event_logits",
                               "preprocess", "adapter_sha", "GT_read", "video_sha256"}
            missing = expected_fields - set(prediction)
            if missing:
                raise RuntimeError(f"prediction {key}/{arm} missing fields {sorted(missing)}")
            if prediction["key"] != key or str(prediction["source"]) != str(row["source"]):
                raise RuntimeError(f"prediction key/source identity mismatch: {key}/{arm}")
            if [int(x) for x in prediction["frame_ids"]] != [int(x) for x in row["input"]["frame_ids"]]:
                raise RuntimeError(f"prediction sampled frame IDs differ from actual input: {key}/{arm}")
            if prediction["video_sha256"] != row["input"]["video_sha256"]:
                raise RuntimeError(f"prediction video SHA differs from actual input metadata: {key}/{arm}")
            if prediction["adapter_sha"] != expected_adapter_sha:
                raise RuntimeError(f"prediction adapter SHA differs from configured source checkpoint: {key}/{arm}")
            if prediction["GT_read"] is not False:
                raise RuntimeError(f"prediction indicates GT access: {key}/{arm}")
            preprocess = prediction["preprocess"]
            pixel_sha = preprocess.get("pixel_sha") if isinstance(preprocess, dict) else None
            if not isinstance(pixel_sha, str) or len(pixel_sha) != 64 or any(c not in "0123456789abcdef" for c in pixel_sha.lower()):
                raise RuntimeError(f"prediction does not have a valid actual pixel SHA: {key}/{arm}")
            preprocesses.append(preprocess)
            event_logits = prediction["event_logits"]
            if not isinstance(event_logits, torch.Tensor):
                event_logits = torch.as_tensor(event_logits)
            if event_logits.ndim != 2 or event_logits.shape[0] != 1 or event_logits.shape[1] != len(row["input"]["frame_ids"]):
                raise RuntimeError(f"direct event logits must be [1,T] matching input frames: {key}/{arm}")
            if not torch.isfinite(event_logits.float()).all():
                raise RuntimeError(f"nonfinite event logits: {key}/{arm}")
            pair[arm] = prediction
            predictions[arm][key] = prediction
            prediction_hash_rows.append({"key": key, "arm": arm, "path": str(path), "sha256": pins[str(path.resolve())]})
        if preprocesses[0] != preprocesses[1]:
            raise RuntimeError(f"same-frame arm predictions have different preprocessing metadata: {key}")
        check = complete_by_key[key]
        if str(check.get("source")) != str(row["source"]):
            raise RuntimeError(f"COMPLETE source identity mismatch: {key}")
        if pair[ARMS[0]]["event_completion"] != pair[ARMS[1]]["event_completion"] or not check.get("event_completion_equal"):
            raise RuntimeError(f"event completions differ or fail COMPLETE assertion: {key}")
        if pair[ARMS[0]]["interval"] != pair[ARMS[1]]["interval"] or not check.get("interval_equal"):
            raise RuntimeError(f"event intervals differ or fail COMPLETE assertion: {key}")
        actual_formats = {arm: bool(pair[arm]["format_ok"]) for arm in ARMS}
        if {arm: bool(check.get("format_ok", {}).get(arm)) for arm in ARMS} != actual_formats:
            raise RuntimeError(f"COMPLETE format flags disagree with saved predictions: {key}")

        diag_path = run_dir / "diagnostics" / f"{index:03}.pt"
        diag = torch.load(diag_path, map_location="cpu", weights_only=False)
        if not all(arm in diag for arm in ARMS):
            raise RuntimeError(f"diagnostic arm rows missing: {key}")
        old_ref = diag[ARMS[0]].get("spatial_reference_token_ids")
        new_ref = diag[ARMS[1]].get("spatial_reference_token_ids")
        old_ref = None if old_ref is None else [int(x) for x in old_ref]
        new_ref = None if new_ref is None else [int(x) for x in new_ref]
        references_equal = old_ref == new_ref
        if bool(check.get("references_equal")) != references_equal:
            raise RuntimeError(f"COMPLETE reference equality differs from sealed diagnostics: {key}")
        for arm in ARMS:
            detail = diag[arm]
            if detail.get("event_completion") != pair[arm]["event_completion"]:
                raise RuntimeError(f"diagnostic event completion differs from prediction: {key}/{arm}")
            if detail.get("spatial_completion") != pair[arm]["spatial_completion"]:
                raise RuntimeError(f"diagnostic spatial completion differs from prediction: {key}/{arm}")
        diagnostics.append(diag)
        reference_rows.append({
            "key": key, "source": str(row["source"]),
            "independent_reference_tokens": old_ref,
            "shared_reference_tokens": new_ref,
            "references_equal": references_equal,
            "reference_token_count_independent": len(old_ref or []),
            "reference_token_count_shared": len(new_ref or []),
            "format_ok": actual_formats,
            "event_completion_equal": True,
            "interval_equal": True,
        })
        input_identity_rows.append({
            "key": key, "source": str(row["source"]),
            "video_sha256": row["input"]["video_sha256"],
            "frame_count": len(row["input"]["frame_ids"]),
            "frame_ids_match": True,
            "adapter_sha_by_arm": {arm: pair[arm]["adapter_sha"] for arm in ARMS},
            "pixel_sha": preprocesses[0]["pixel_sha"],
            "preprocess": preprocesses[0],
        })

    # At this point the full prediction seal and actual adapter/input identity
    # have passed. Only now may the scorer open source validation labels.
    labels_path = SOURCE_ART / "SOURCE_LABELS_TRAINING_ONLY.json"
    labels_sha = sha256_file(labels_path)
    labels = read_json(labels_path)
    if not isinstance(labels, dict) or any(key not in labels for key in keys):
        raise RuntimeError("source validation labels do not cover the sealed 198-query roster")
    source_rows_by_key = {row["key"]: row for row in rows}
    query_scores: dict[str, dict[str, list[dict[str, Any]]]] = {arm: [] for arm in ARMS}
    event_outputs: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for key in keys:
            row = source_rows_by_key[key]
            prediction = predictions[arm][key]
            label = labels[key]
            if [int(x) for x in label["frame_ids"]] != [int(x) for x in row["input"]["frame_ids"]]:
                raise RuntimeError(f"source label frame IDs differ from the actual input roster: {key}")
            if len(label["event_active"]) != len(prediction["frame_ids"]):
                raise RuntimeError(f"event_active labels and saved logits length mismatch: {key}/{arm}")
            metric = score_tube_independently(prediction, label)
            query_scores[arm].append({"key": key, "source": str(row["source"]), "metrics": metric})
        parents = summarize_parents(query_scores[arm])
        arm_query_lookup = {r["key"]: r for r in query_scores[arm]}
        points = []
        for key in keys:
            prediction = predictions[arm][key]
            target = [bool(x) for x in labels[key]["event_active"]]
            logits = prediction["event_logits"].detach().float().cpu().numpy().reshape(-1).astype(np.float64)
            auc = binary_auc(target, logits)
            points.append({
                "key": key, "source": str(source_rows_by_key[key]["source"]),
                "temporal_tIoU": float(arm_query_lookup[key]["metrics"]["tIoU"]),
                "event_frame_AUROC": auc, "event_frame_BCE": binary_bce(target, logits),
                "frames": len(target), "positive_frames": int(sum(target)),
                "negative_frames": int(len(target) - sum(target)),
                "undefined_AUROC_reason": "single_class_query" if auc is None else None,
                "_logits": logits.tolist(), "_labels": target,
            })
        event_outputs[arm] = summarize_event_endpoint(points)

    # Frozen geometry comes from the separate independent raw readback. Verify
    # its source-label and feature-cache provenance before using its scores.
    frozen_path = SOURCE_ART / "SOURCE_FROZEN_INDEPENDENT_METRICS.json"
    frozen_artifact = read_json(frozen_path)
    barrier_path = SOURCE_ART / "SOURCE_FULL_FEATURE_BARRIER.json"
    barrier_sha = sha256_file(barrier_path)
    if frozen_artifact.get("label_sha") != labels_sha or frozen_artifact.get("barrier_sha") != barrier_sha:
        raise RuntimeError("independent Frozen readback provenance does not match source labels/feature barrier")
    barrier = read_json(barrier_path)
    frozen_rows_list = [r for r in frozen_artifact.get("rows", []) if r.get("condition") == "clean"]
    frozen_rows = {r["key"]: r for r in frozen_rows_list}
    if len(frozen_rows) != 198 or set(frozen_rows) != set(keys):
        raise RuntimeError("independent Frozen clean baseline does not match all 198 queries")
    for key in keys:
        row = frozen_rows[key]
        if str(row["source"]) != str(source_rows_by_key[key]["source"]):
            raise RuntimeError(f"Frozen baseline source mapping differs: {key}")
        cache = SOURCE_ART / "source_features" / "clean" / (hashlib.sha256(key.encode()).hexdigest() + ".pt")
        expected_cache_sha = barrier.get("files", {}).get(str(cache))
        if not expected_cache_sha or sha256_file(cache) != expected_cache_sha or row.get("cache_sha") != expected_cache_sha:
            raise RuntimeError(f"Frozen feature cache does not match its independent readback pin: {key}")
    frozen_query_scores = []
    for key in keys:
        row = frozen_rows[key]
        metric = {m: _finite_float(row["metrics"][m], f"Frozen {key}/{m}") for m in METRICS}
        frozen_query_scores.append({"key": key, "source": str(row["source"]), "metrics": metric})
    frozen_parents = summarize_parents(frozen_query_scores)
    arm_parent_scores = {
        arm: summarize_parents(query_scores[arm]) for arm in ARMS
    }
    comparisons = {}
    pair_specs = {
        "shared_minus_independent": (ARMS[1], ARMS[0]),
        "independent_minus_Frozen": (ARMS[0], "Frozen"),
        "shared_minus_Frozen": (ARMS[1], "Frozen"),
    }
    all_parent_maps = {**arm_parent_scores, "Frozen": frozen_parents}
    for name, (first, second) in pair_specs.items():
        comparisons[name] = {
            metric: paired_parent_bootstrap(all_parent_maps[first], all_parent_maps[second], metric)
            for metric in METRICS
        }

    arm_report = {}
    for arm in ARMS:
        event = event_outputs[arm]
        arm_report[arm] = {
            "query_rows": query_scores[arm],
            "summary": summarize_arm(query_scores[arm], arm_parent_scores[arm]),
            "event_endpoint": event,
        }
    frozen_retention = retention_against_frozen(query_scores, frozen_rows)
    ref_formats = {}
    for arm in ARMS:
        arm_pred = {r["key"]: r for r in query_scores[arm]}
        ref_formats[arm] = {
            "queries": len(keys),
            "format_ok_queries": sum(bool(predictions[arm][k]["format_ok"]) for k in keys),
            "format_failures": [k for k in keys if not bool(predictions[arm][k]["format_ok"])],
            "reference_diff_queries": sum(not r["references_equal"] for r in reference_rows),
            "reference_diff_keys": [r["key"] for r in reference_rows if not r["references_equal"]],
            "references_equal_queries": sum(r["references_equal"] for r in reference_rows),
            "saved_vIoU": {k: float(arm_pred[k]["metrics"]["vIoU"]) for k in keys},
        }
    injection = summarize_injection(diagnostics)

    source_n = len(set(sources))
    arm_point_counts = {arm: len(query_scores[arm]) for arm in ARMS}
    if any(n != 198 for n in arm_point_counts.values()) or any(len(p) != source_n for p in arm_parent_scores.values()):
        raise RuntimeError("scored arm denominator changed")
    audit = {
        "status": "passed",
        "run_dir": str(run_dir),
        "read_order": [
            "COMPLETE.json read and verified completed full-run status",
            "PREDICTIONS_SEAL.json read and every prediction/diagnostic pin rehashed",
            "CONFIG/INPUTS/LOCK and all saved prediction/input/adapter identities checked",
            "Only after the above checks, source labels and Frozen reference readback were opened",
        ],
        "complete_status": complete.get("status"),
        "interface_pass": complete.get("interface_pass"),
        "scope": config.get("scope"),
        "source_val_GT_read_by_runner": seal.get("source_val_GT_read"),
        "target_GT_read_by_runner": seal.get("target_GT_read"),
        "prediction_seal_sha256": seal_sha,
        "sealed_prediction_and_diagnostic_files": len(pins),
        "prediction_hash_failures": 0,
        "expected_sealed_files": 594,
        "actual_adapter_sha_expected": expected_adapter_sha,
        "actual_adapter_sha_by_arm": {arm: sorted({predictions[arm][k]["adapter_sha"] for k in keys}) for arm in ARMS},
        "actual_input_identity_rows": input_identity_rows,
        "prediction_file_hash_rows": prediction_hash_rows,
        "query_count": len(keys), "parent_count": source_n,
        "queries_per_arm": arm_point_counts,
        "diagnostic_file_count": len(diagnostics),
        "event_completion_equal_queries": sum(r["event_completion_equal"] for r in reference_rows),
        "event_interval_equal_queries": sum(r["interval_equal"] for r in reference_rows),
        "source_labels_sha256": labels_sha,
        "source_label_records_loaded_from_file": len(labels),
        "source_validation_label_queries_used": len(keys),
        "other_source_label_records_loaded_but_unused": len(labels) - len(keys),
        "target_labels_opened": False,
        "frozen_independent_readback_sha256": sha256_file(frozen_path),
        "frozen_feature_barrier_sha256": barrier_sha,
        "frozen_feature_cache_pins_verified": len(keys),
        "no_GPU_or_model_forward": True,
        "independent_geometry_method": "inline scalar normalized cxcywh-to-xyxy and physical half-open time arithmetic; no source-fit worker scorer or vg_tta.metrics import",
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "desta3d_v2_reference_audit_independent_cpu_v1",
        "provenance": audit,
        "run_configuration": config,
        "metric_method": {
            "sIoU": "Mean normalized xyxy IoU over all event_active && box_valid source frames; zero if no spatial support.",
            "vIoU": "Sparse predictions are placed on a dense all-zero sampled-frame canvas; invalid geometry remains zero. Numerator sums box IoUs on support frames whose physical frame ID lies in predicted∩GT half-open event time; denominator counts sampled frame IDs in predicted∪GT physical time.",
            "tIoU": "Continuous physical-time IoU of [frame_ids[pred_start], frame_ids[pred_end]+1) and [GT begin_fid, GT end_fid).",
            "format_failure": "Retained in every denominator with vIoU=sIoU=tIoU=0; not silently dropped.",
            "no_spatial_support": "The five unsupported queries remain in the 198-query and 31-parent metrics; they receive vIoU=0 and sIoU=0.",
            "paired_parent_bootstrap": "Paired source-parent deltas; resample 31 parent deltas with replacement; 10,000 draws, NumPy default_rng seed 20260927, percentile 95% interval.",
        },
        "frozen_reference": {
            "artifact": str(frozen_path.relative_to(ROOT)),
            "source": "prior independent raw Frozen source-validation readback; per-query clean rows matched to this exact input roster and underlying feature cache hashes reverified",
            "query_macro": {m: float(np.mean([r["metrics"][m] for r in frozen_query_scores])) for m in METRICS},
            "parent_macro": {m: float(np.mean([p[m] for p in frozen_parents.values()])) for m in METRICS},
            "parent_rows": [{"source": p, **scores} for p, scores in sorted(frozen_parents.items())],
            "query_rows": frozen_query_scores,
        },
        "arms": arm_report,
        "paired_parent_comparisons": comparisons,
        "frozen_good_retention": frozen_retention,
        "reference_format_diagnostics": ref_formats,
        "reference_pair_rows": reference_rows,
        "injection_diagnostics": injection,
        "same_prefix_residual_control": summarize_same_prefix_residual_control(
            diagnostics, [predictions[ARMS[1]][key] for key in keys]
        ),
        "interpretation_boundary": {
            "scope": "A sealed source-validation reference-contract audit of two architecture interfaces, not model training, TTA, target GT, or a method-promotion gate.",
            "direct_event_head": "Direct event AUROC/BCE and its association with tIoU are descriptive diagnostics. Higher event AUROC alone does not establish improved localization or tubes.",
            "thresholds": "Frozen-good thresholds vIoU>0.5/tIoU>0.5 are retention summaries inherited from the registered source readback; they are descriptive and not new deployment reliability thresholds.",
            "residual_cast": "changed_elements counts hook output elements that differ after casting to native token dtype; relative_injection_norm measures the output-token residual at the hook, not a quality score.",
        },
    }

    output_dir = run_dir / "independent_cpu_readback"
    if output_dir.exists():
        raise FileExistsError(f"independent readback directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    _write_json_once(output_dir / "AUDIT.json", audit)
    _write_json_once(output_dir / "INDEPENDENT_SCORE.json", report)
    (output_dir / "INDEPENDENT_SCORE.md").write_text(_render_markdown(report))
    return {"output_dir": str(output_dir), "audit": audit}


def synthetic_self_check() -> None:
    """CPU-only algebra checks; does not open project labels or prediction data."""
    label = {
        "frame_ids": [0, 2, 8, 10],
        "box_valid": [True, True, True, True],
        "event_active": [False, True, True, False],
        "boxes_xyxy": [[0.0, 0.0, 1.0, 1.0]] * 4,
        "event_interval": {"begin_fid": 2, "end_fid": 9},
    }
    prediction = {
        "frame_ids": [0, 2, 8, 10], "interval": [0, 3], "format_ok": True,
        "positions": [1], "boxes_cxcywh": torch.tensor([[0.5, 0.5, 1.0, 1.0]]),
        "geometry_valid": torch.tensor([True]),
    }
    got = score_tube_independently(prediction, label)
    assert got["vIoU"] == 0.25 and got["sIoU"] == 0.5 and math.isclose(got["tIoU"], 7 / 11)
    bad = dict(prediction, format_ok=False)
    fail = score_tube_independently(bad, label)
    assert fail["vIoU"] == fail["sIoU"] == fail["tIoU"] == 0.0
    assert fail["spatial_support_frames"] == 2
    assert binary_auc([False, True, True, False], [0.0, 1.0, 1.0, 0.0]) == 1.0
    assert binary_auc([True, True], [0.1, 0.2]) is None
    assert binary_bce([False, True], [-2.0, 2.0]) < 0.2
    parent1 = {"p1": {m: .2 for m in METRICS}, "p2": {m: .4 for m in METRICS}}
    parent2 = {"p1": {m: .1 for m in METRICS}, "p2": {m: .3 for m in METRICS}}
    ci1 = paired_parent_bootstrap(parent1, parent2, "vIoU")
    ci2 = paired_parent_bootstrap(parent1, parent2, "vIoU")
    assert ci1 == ci2 and math.isclose(ci1["mean_delta_pp"], 10.0)
    demo_frozen = {
        "q1": {"source": "p1", "metrics": {"vIoU": .6, "tIoU": .7}},
        "q2": {"source": "p1", "metrics": {"vIoU": .4, "tIoU": .7}},
    }
    demo_candidate = [
        {"key": "q1", "source": "p1", "metrics": {"vIoU": .6, "tIoU": .6}},
        {"key": "q2", "source": "p1", "metrics": {"vIoU": .3, "tIoU": .4}},
    ]
    retained = retention_against_frozen({"demo": demo_candidate}, demo_frozen)["arms"]["demo"]
    assert retained["full_tube_vIoU_gt_0_5"]["retained_queries"] == 1
    assert retained["native_temporal_tIoU_gt_0_5"]["retained_queries"] == 1
    with_logits = torch.zeros(2, 1001)
    without_logits = torch.zeros(2, 1001)
    with_logits[0, 0] = 1.0
    lp = torch.log_softmax(with_logits, dim=-1)
    lq = torch.log_softmax(without_logits, dim=-1)
    stored_kl = float((lp.exp() * (lp-lq)).sum(-1).mean())
    fake_diag = [{"shared_reference_time": {
        "coordinate_logits": with_logits,
        "event_completion": "<event>",
        "spatial_reference_token_ids": [1, 2],
        "same_prefix_residual_control": {
            "without_logits": without_logits, "KL_with_to_without": stored_kl,
            "logit_changed": 1, "max_abs_change": 1.0, "format_without": True,
            "fixed_reference_time": True}}}]
    fake_shared_prediction = [{"interval": [0, 1], "event_completion": "<event>"}]
    control = summarize_same_prefix_residual_control(fake_diag, fake_shared_prediction)
    assert control["queries_available"] == 1 and control["logit_changed_total"] == 1
    assert control["stored_recomputed_KL_max_abs_diff"] == 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="completed full 198-query run directory")
    parser.add_argument("--self-test", action="store_true", help="run synthetic CPU arithmetic checks only")
    args = parser.parse_args(argv)
    if args.self_test:
        synthetic_self_check()
        print("synthetic CPU self-check passed")
        return
    if args.run_dir is None:
        parser.error("--run-dir is required unless --self-test is used")
    torch.set_num_threads(1)
    result = run(args.run_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
