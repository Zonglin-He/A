"""Source-only independent CPU readback after the complete 792-prediction seal.

Geometry uses scalar arithmetic, not the training worker or vg_tta.metrics.
No model inference, optimization, state selection, or target labels are used.
All new arms and Frozen physical inputs are checked BEFORE opening labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import score_desta3d_v2_reference_audit as scalar

ARMS = ("common", "B0", "B1", "B2")
METRICS = scalar.METRICS
BASE = ROOT / "artifacts/desta3d_v2/aux_backflow_v1/fit_recovery_v2"
SOURCE = ROOT / "artifacts/desta3d_v1"
INPUTS = ROOT / "artifacts/desta3d_v2/source_fit/INPUTS.json"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def save_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def state_sha(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def sealed_files(run_dir, query_count=198, parent_count=31):
    """No torch payloads or labels are opened before this whole-seal check."""
    complete = scalar.read_json(run_dir / "PREDICTIONS_COMPLETE.json")
    seal_path = run_dir / "ALL_PREDICTIONS_SEAL.json"
    require(complete["seal_sha"] == scalar.sha256_file(seal_path), "completion seal SHA mismatch")
    seal = scalar.read_json(seal_path)
    require(set(seal["arms"]) == set(ARMS), "four fixed arms required")
    require(seal["queries_per_arm"] == query_count and seal["parents"] == parent_count,
            "seal roster count mismatch")
    require(seal["new_predictions"] == 4 * query_count, "not all predictions completed")
    require(seal["source_validation_GT_read"] is False and seal["target_GT_read"] is False,
            "runner reports label access")
    expected = {str((run_dir / "predictions" / arm / f"{i:03}.pt").resolve())
                for arm in ARMS for i in range(query_count)}
    require(set(seal["pins"]) == expected, "sealed prediction paths differ from fixed roster")
    for path, digest in seal["pins"].items():
        require(scalar.sha256_file(Path(path)) == digest, f"prediction SHA mismatch: {path}")
    return seal


def check_prediction(pred, row, adapter_hash):
    key = row["key"]
    require(pred["key"] == key and str(pred["source"]) == str(row["source"]), "prediction identity mismatch")
    require(pred["frame_ids"] == row["input"]["frame_ids"], "physical frame IDs mismatch")
    require(pred["video_sha256"] == row["input"]["video_sha256"], "video SHA mismatch")
    require(pred["adapter_sha"] == adapter_hash, "prediction adapter mismatch")
    require(pred["GT_read"] is False, "prediction reports GT access")
    pre = pred["preprocess"]
    digest = pre.get("pixel_sha", "")
    require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "invalid pixel SHA")
    logits = torch.as_tensor(pred["event_logits"])
    require(tuple(logits.shape) == (1, len(pred["frame_ids"])), "event logits must be [1,T]")
    require(bool(torch.isfinite(logits).all()), "nonfinite event logits")
    boxes = torch.as_tensor(pred["boxes_cxcywh"])
    valid = torch.as_tensor(pred["geometry_valid"])
    require(tuple(boxes.shape) == (len(pred["positions"]), 4), "sparse box shape mismatch")
    require(tuple(valid.shape) == (len(pred["positions"]),), "sparse validity shape mismatch")
    require(bool(torch.isfinite(boxes).all()), "nonfinite sparse boxes")
    positions = pred["positions"]
    require(len(set(positions)) == len(positions) and
            all(0 <= int(p) < len(pred["frame_ids"]) for p in positions), "invalid sparse positions")
    # Legal format failures and invalid geometry remain in the task denominator.
    return pre


def frozen_prediction(cache, row, preprocess):
    require(cache["key"] == row["key"] and str(cache["source"]) == str(row["source"]),
            "Frozen key/source mismatch")
    require(cache["frame_ids"] == row["input"]["frame_ids"], "Frozen frame grid mismatch")
    require(cache["condition"] == "clean" and cache["GT_read"] is False, "Frozen cache provenance mismatch")
    require(cache["preprocess"] == preprocess, "Frozen physical pixels/preprocessing mismatch")
    tokens = cache["box_logits"].argmax(-1).float() / 1000
    valid = (tokens[..., 2:] > tokens[..., :2]).all(-1)
    boxes = torch.cat(((tokens[..., :2] + tokens[..., 2:]) / 2,
                       tokens[..., 2:] - tokens[..., :2]), -1)
    boxes = torch.where(valid[..., None], boxes, torch.zeros_like(boxes))
    return {"key": row["key"], "source": str(row["source"]), "frame_ids": cache["frame_ids"],
            "positions": cache["spatial_positions"], "boxes_cxcywh": boxes,
            "geometry_valid": valid, "interval": cache["interval"], "format_ok": cache["format_ok"]}


def verify_before_labels(run_dir):
    seal = sealed_files(run_dir)
    lock = scalar.read_json(run_dir / "LOCK.json")
    deferred = {}
    for path, digest in lock["pins"].items():
        if Path(path).name in ("SOURCE_LABELS_TRAINING_ONLY.json", "FROZEN_SOURCE_VAL_BASELINE.json"):
            deferred[path] = digest
            continue
        require(scalar.sha256_file(Path(path)) == digest, f"active lock pin changed: {path}")
    require(str(INPUTS) in lock["pins"], "source input roster not locked")
    rows = sorted([r for r in scalar.read_json(INPUTS) if r["split"] == "validation"], key=lambda r: r["key"])
    require(len(rows) == len({r["key"] for r in rows}) == 198 and
            len({r["source"] for r in rows}) == 31, "source validation roster mismatch")
    config = scalar.read_json(run_dir / "CONFIG.json")
    require(scalar.sha256_file(run_dir / "COMMON_A.pt") == config["common_A_sha"], "common A mismatch")
    weights = torch.load(run_dir / "COMMON_A.pt", map_location="cpu", weights_only=False)
    weights.update(torch.load(run_dir / "FINAL_B.pt", map_location="cpu", weights_only=False))
    hashes = {a: state_sha(s) for a, s in weights.items()}
    require(hashes == seal["adapter_hashes"], "final weights differ from sealed adapters")
    latest = torch.load(run_dir / "LATEST.pt", map_location="cpu", weights_only=False)
    require(latest["stage"] == "validation" and latest["cursor"] == 618, "training not complete")
    for arm in ARMS[1:]:
        require(state_sha(latest["adapters"][arm]) == hashes[arm], "LATEST differs from fixed final weights")
        states = latest["optimizers"][arm]["state"]
        require(len(states) == 46 and all(type(k) is int for k in states), "invalid final Adam state keys")
        require({int(v["step"]) for v in states.values()} == {155} and latest["steps"][arm] == 155,
                "actual Adam counters must complete 155 B steps")
    del weights, latest
    barrier_path = SOURCE / "SOURCE_FULL_FEATURE_BARRIER.json"
    barrier = scalar.read_json(barrier_path)
    predictions = {a: {} for a in (*ARMS, "Frozen")}
    identities = []
    for i, row in enumerate(rows):
        key = row["key"]
        pres = []
        for arm in ARMS:
            pred = torch.load(run_dir / "predictions" / arm / f"{i:03}.pt", map_location="cpu", weights_only=False)
            pres.append(check_prediction(pred, row, hashes[arm]))
            predictions[arm][key] = pred
        require(all(p == pres[0] for p in pres), "arms have different physical pixels/preprocessing")
        cache_path = SOURCE / "source_features/clean" / (hashlib.sha256(key.encode()).hexdigest() + ".pt")
        cache_sha = scalar.sha256_file(cache_path)
        require(cache_sha == barrier["files"][str(cache_path)], "Frozen cache barrier mismatch")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        predictions["Frozen"][key] = frozen_prediction(cache, row, pres[0])
        identities.append({"key": key, "source": str(row["source"]), "pixel_sha": pres[0]["pixel_sha"],
                           "frame_ids": row["input"]["frame_ids"], "video_sha": row["input"]["video_sha256"],
                           "frozen_cache_sha": cache_sha})
        del cache
    return rows, predictions, {"status": "passed", "time": time.time(), "predictions": 792,
        "queries": 198, "parents": 31, "seal_sha": scalar.sha256_file(run_dir / "ALL_PREDICTIONS_SEAL.json"),
        "final_B_sha": scalar.sha256_file(run_dir / "FINAL_B.pt"), "adapter_hashes": hashes,
        "lock_sha": scalar.sha256_file(run_dir / "LOCK.json"), "checked_lock_pins": len(lock["pins"]) - len(deferred),
        "label_dependent_pins_deferred_until_after_identity_barrier": deferred,
        "frozen_barrier_sha": scalar.sha256_file(barrier_path), "identities": identities,
        "all_final_Adam_counters": 155, "source_validation_GT_read": False, "target_GT_read": False}


def event_summary(rows, predictions, labels, scores):
    points = []
    by_parent = {}
    for row, score in zip(rows, scores):
        key = row["key"]
        ys = [bool(x) for x in labels[key]["event_active"]]
        logits = predictions[key]["event_logits"].float().reshape(-1).tolist()
        require(len(ys) == len(logits), "event labels/logits length mismatch")
        auc = scalar.binary_auc(ys, logits)
        points.append({"key": key, "source": str(row["source"]), "temporal_tIoU": score["metrics"]["tIoU"],
            "event_frame_AUROC": auc, "event_frame_BCE": scalar.binary_bce(ys, logits),
            "frames": len(ys), "positive_frames": sum(ys), "negative_frames": len(ys) - sum(ys),
            "undefined_AUROC_reason": "single_class_query" if auc is None else None,
            "_logits": logits, "_labels": ys})
        counter = by_parent.setdefault(str(row["source"]), [0, 0, 0, 0])
        for y, logit in zip(ys, logits):
            counter[0 if y else 2] += logit >= 0
            counter[1 if y else 3] += 1
    result = scalar.summarize_event_endpoint(points)
    result["activation_at_probability_0_5"] = {
        "parent_rows": [{"source": p, "true_positive": c[0], "positive_frames": c[1],
                         "false_positive": c[2], "negative_frames": c[3]} for p, c in sorted(by_parent.items())],
        "parent_macro_detection": float(np.mean([c[0] / c[1] for c in by_parent.values() if c[1]]))
            if any(c[1] for c in by_parent.values()) else None,
        "parent_macro_false_activation": float(np.mean([c[2] / c[3] for c in by_parent.values() if c[3]]))
            if any(c[3] for c in by_parent.values()) else None,
        "positive_defined_parents": sum(c[1] > 0 for c in by_parent.values()),
        "negative_defined_parents": sum(c[3] > 0 for c in by_parent.values()),
    }
    return result


def run(run_dir, out_dir):
    require(not out_dir.exists(), "readback directory already exists; preserve previous audit")
    rows, predictions, audit = verify_before_labels(run_dir)
    # This durable barrier is written before source labels are opened.
    save_once(out_dir / "PRE_GT_AUDIT.json", audit)
    for path, digest in audit["label_dependent_pins_deferred_until_after_identity_barrier"].items():
        require(scalar.sha256_file(Path(path)) == digest, f"label-dependent lock pin changed: {path}")
    label_path = SOURCE / "SOURCE_LABELS_TRAINING_ONLY.json"
    labels = scalar.read_json(label_path)
    for row in rows:
        require(labels[row["key"]]["frame_ids"] == row["input"]["frame_ids"], "label frame grid mismatch")
    scores = {a: [{"key": r["key"], "source": str(r["source"]),
                   "metrics": scalar.score_tube_independently(predictions[a][r["key"]], labels[r["key"]])}
                  for r in rows] for a in predictions}
    parents = {a: scalar.summarize_parents(s) for a, s in scores.items()}
    pairs = [(a, "Frozen") for a in ARMS] + [(a, "common") for a in ARMS[1:]] + [
        ("B1", "B0"), ("B2", "B0"), ("B2", "B1")]
    report = {"scope": "fixed final source-development readout; not target TTA or selected best checkpoint",
        "time": time.time(), "source_label_sha": scalar.sha256_file(label_path), "target_GT_read": False,
        "pre_GT_audit_sha": scalar.sha256_file(out_dir / "PRE_GT_AUDIT.json"),
        "scorer_sha": scalar.sha256_file(Path(__file__)), "geometry_helper_sha": scalar.sha256_file(Path(scalar.__file__)),
        "arms": {a: {"summary": scalar.summarize_arm(scores[a], parents[a]), "query_rows": scores[a]}
                 for a in predictions},
        "comparisons": {f"{a}_minus_{b}": {m: scalar.paired_parent_bootstrap(parents[a], parents[b], m)
                         for m in METRICS} for a, b in pairs},
        "native_good_retention": scalar.retention_against_frozen(
            {a: scores[a] for a in ARMS}, {r["key"]: r for r in scores["Frozen"]}),
        "event": {a: event_summary(rows, predictions[a], labels, scores[a]) for a in ARMS}}
    save_once(out_dir / "REPORT.json", report)
    lines = ["# Fixed auxiliary-backflow source contrast", "", report["scope"], "",
             "All 792 new predictions and Frozen physical-input identities passed before source labels were read.",
             "", "| Arm | Parent vIoU % | sIoU % | tIoU % | Delta v vs Frozen pp [95% CI] | >5pp losses |",
             "|---|---:|---:|---:|---|---:|"]
    for a in predictions:
        m = report["arms"][a]["summary"]["parent_macro"]
        c = report["comparisons"].get(f"{a}_minus_Frozen", {}).get("vIoU")
        d = f"{c['mean_delta_pp']:+.5f} {c['bootstrap_ci95_pp']}" if c else "baseline"
        lines.append(f"|{a}|{100*m['vIoU']:.5f}|{100*m['sIoU']:.5f}|{100*m['tIoU']:.5f}|{d}|{c['severe_loss_below_minus5pp'] if c else 0}|")
    lines += ["", "Paired comparisons, direct-event AUROC defined denominators, activation rates, native-good retention, and all query/parent rows are in REPORT.json.",
              "No target labels, validation-selected checkpoints, or method promotion are involved."]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    save_once(out_dir / "COMPLETE.json", {"status": "independently_scored", "report_sha": scalar.sha256_file(out_dir / "REPORT.json")})
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=BASE)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(2)
    run(args.run_dir.resolve(), args.out_dir or args.run_dir / "independent_readback_v1")
