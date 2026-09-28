"""Second CPU calculation of a completed source-only auxiliary contrast report.

Runs only after the independent scalar scorer completed its input/label barrier.
Uses the pre-existing tensor task evaluator and pairwise AUROC, not the scalar
scorer's geometry or AUROC functions. Does not infer, train, or select states.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.desta3d_source_fit_v1 import _tube_metric, _frozen_prediction

RUN = ROOT / "artifacts/desta3d_v2/aux_backflow_v1/fit_recovery_v2"
SOURCE = ROOT / "artifacts/desta3d_v1"
METRICS = ("vIoU", "sIoU", "tIoU")
ARMS = ("common", "B0", "B1", "B2", "Frozen")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pairwise_auc(y, scores):
    y = np.asarray(y, bool)
    x = np.asarray(scores, np.float64)
    if not y.any() or y.all():
        return None
    diff = x[y, None] - x[None, ~y]
    return float(((diff > 0).sum() + .5 * (diff == 0).sum()) / diff.size)


def main():
    torch.set_num_threads(2)
    out = RUN / "ROOT_SCORE_CROSSCHECK.json"
    if out.exists():
        raise FileExistsError("preserve prior root crosscheck")
    directory = RUN / "independent_readback_v1"
    done = read(directory / "COMPLETE.json")
    assert done["status"] == "independently_scored"
    assert done["report_sha"] == sha(directory / "REPORT.json")
    report = read(directory / "REPORT.json")
    assert report["pre_GT_audit_sha"] == sha(directory / "PRE_GT_AUDIT.json")
    audit = read(directory / "PRE_GT_AUDIT.json")
    assert audit["status"] == "passed" and audit["predictions"] == 792
    assert audit["seal_sha"] == sha(RUN / "ALL_PREDICTIONS_SEAL.json")
    assert read(RUN / "PREDICTIONS_COMPLETE.json")["seal_sha"] == audit["seal_sha"]
    for path, digest in read(RUN / "ALL_PREDICTIONS_SEAL.json")["pins"].items():
        assert sha(Path(path)) == digest
    # Only now re-open source labels already exposed by the completed audit.
    label_path = SOURCE / "SOURCE_LABELS_TRAINING_ONLY.json"
    assert sha(label_path) == report["source_label_sha"]
    labels = read(label_path)
    rows = sorted([r for r in read(ROOT / "artifacts/desta3d_v2/source_fit/INPUTS.json")
                   if r["split"] == "validation"], key=lambda r: r["key"])
    assert len(rows) == 198 and len({r["source"] for r in rows}) == 31
    barrier = read(SOURCE / "SOURCE_FULL_FEATURE_BARRIER.json")
    assert sha(SOURCE / "SOURCE_FULL_FEATURE_BARRIER.json") == audit["frozen_barrier_sha"]
    result = {}; metric_errors = []; auc_errors = []; parent_maps = {}
    for arm in ARMS:
        lookup = {r["key"]: r for r in report["arms"][arm]["query_rows"]}
        parent_scores = {}; events = {}
        for i, row in enumerate(rows):
            key = row["key"]; parent = str(row["source"])
            pred = (_frozen_prediction(row, barrier) if arm == "Frozen" else
                    torch.load(RUN / "predictions" / arm / f"{i:03}.pt", map_location="cpu", weights_only=False))
            measured = _tube_metric(pred, labels[key])
            for metric in METRICS:
                error = abs(measured[metric] - lookup[key]["metrics"][metric]); metric_errors.append(error)
                assert error < 2e-6, (arm, key, metric, error)
            assert measured["format_ok"] == lookup[key]["metrics"]["format_ok"]
            assert measured["spatial_support_frames"] == lookup[key]["metrics"]["spatial_support_frames"]
            parent_scores.setdefault(parent, []).append([measured[m] for m in METRICS])
            if arm != "Frozen":
                e = events.setdefault(parent, [[], []])
                e[0].extend(labels[key]["event_active"])
                e[1].extend(pred["event_logits"].float().reshape(-1).tolist())
        parent_maps[arm] = {p: np.mean(values, axis=0) for p, values in parent_scores.items()}
        macro = np.mean(list(parent_maps[arm].values()), axis=0)
        assert np.allclose(macro, [report["arms"][arm]["summary"]["parent_macro"][m] for m in METRICS], atol=2e-6, rtol=0)
        if events:
            ep = {r["source"]: r for r in report["event"][arm]["source_points"]}
            for p, (ys, logits) in events.items():
                actual = pairwise_auc(ys, logits); expected = ep[p]["event_frame_AUROC"]
                assert (actual is None) == (expected is None)
                if actual is not None:
                    auc_errors.append(abs(actual - expected)); assert abs(actual - expected) < 1e-12
        result[arm] = {"queries": 198, "parents": 31, "parent_macro": dict(zip(METRICS, macro.tolist()))}
    ci_error = 0
    for name, comparison in report["comparisons"].items():
        first, second = name.split("_minus_")
        parents = sorted(parent_maps[first]); assert parents == sorted(parent_maps[second])
        delta = np.array([parent_maps[first][p] - parent_maps[second][p] for p in parents])
        index = np.random.default_rng(20260927).integers(0, len(parents), size=(10000, len(parents)))
        ci = np.quantile(delta[index].mean(axis=1), [.025, .975], axis=0) * 100
        for j, m in enumerate(METRICS):
            err = float(np.max(np.abs(ci[:, j] - comparison[m]["bootstrap_ci95_pp"])))
            ci_error = max(ci_error, err); assert err < 2e-4
    payload = {"time": time.time(), "status": "passed", "scope": "second tensor geometry + pairwise parent event AUROC; source-development only",
        "query_metric_comparisons": len(metric_errors), "maximum_query_metric_absolute_error": max(metric_errors),
        "maximum_parent_AUROC_absolute_error": max(auc_errors, default=0),
        "maximum_parent_bootstrap_CI_error_pp": ci_error, "arms": result,
        "report_sha": done["report_sha"], "source_validation_labels_read_after_completed_audit": True,
        "target_GT_read": False, "new_inference_or_updates": False,
        "script_sha": sha(Path(__file__)), "tensor_metric_sha": sha(ROOT / "vg_tta/metrics.py")}
    with out.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
