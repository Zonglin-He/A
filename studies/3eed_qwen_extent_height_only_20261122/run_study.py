"""Seal and score both extent clips on the locked Ordinary Validation cohort."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from live import first_depth, incumbent
from extent_clip import TOL, clip_extent, protected_points


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PAYLOAD = HERE / "private" / "predictions.jsonl.gz"
PROTOCOL = HERE / "PROTOCOL.json"
SOURCE = HERE.parent / "3eed_qwen_ray_equivalent_extent_20261122"
sys.path.insert(0, str(SOURCE))
import run_trial as reference


roster, scorer = reference.roster, reference.score_full
ARMS = ("baseline", "ray_equivalent", "height_only")
PAIRS = (("ray_equivalent", "baseline"), ("height_only", "baseline"), ("height_only", "ray_equivalent"))
PLATFORM_COUNTS = {"waymo": 3782, "drone": 2792, "quad": 2736}


def infer_one(job):
    frame, official, trace, previous, check, pins = job
    assert frame["panel"] == "validation"
    assert frame["uid"] == official["uid"] == trace["uid"] == previous["uid"]
    archived = reference.origin.prediction(frame, check, pins)
    answer = reference.origin.replies(frame, archived, reference.replies_hashes)
    original, context = incumbent(frame, answer, collect_audit=True)
    assert original == official["simple4"]
    row = dict(uid=frame["uid"], active=context is not None, full_error_m=0., variants={})
    assert row["active"] == original["applicable"] == previous["active"] == trace["production"]
    if context is None:
        return row

    box, dimensions, rotation, fit, selection, audit, transform = context
    key = "q" if trace["choose_qwen"] else "p"
    np.testing.assert_array_equal(box, trace[key])
    np.testing.assert_array_equal(box, previous["box"])
    np.testing.assert_array_equal(selection, trace["held"])
    np.testing.assert_array_equal(transform, trace["transform"])
    depth = first_depth(box, selection, transform)
    np.testing.assert_array_equal(depth, trace[key + "_depth"])
    audit_depth = first_depth(box, audit, transform)
    eye, measured, entries = protected_points(box, fit, selection, depth, transform)
    row.update(
        box=box.tolist(), selection_rays=len(selection), audit_rays=len(audit),
        protected_points=len(measured),
    )
    assert row["selection_rays"] == previous["selection_rays"]
    assert row["audit_rays"] == previous["audit_rays"]
    assert row["protected_points"] == previous["protected_points"]

    bottom = box[2] - box[5] / 2
    for arm in ARMS[1:]:
        corrected = clip_extent(box, dimensions, rotation, eye, measured, entries, height_only=arm == "height_only")
        if arm == "ray_equivalent":
            prior = previous["variants"].get(arm)
            expected = prior["box"] if prior is not None else box
            row["full_error_m"] = float(np.max(np.abs(corrected - expected)))
            assert row["full_error_m"] <= TOL and (corrected is not box) == (prior is not None)
        else:
            fixed = [0, 1, 3, 4, 6]
            np.testing.assert_array_equal(corrected[fixed], box[fixed])
        assert abs(corrected[2] - corrected[5] / 2 - bottom) <= TOL
        assert corrected[6] == box[6]
        if corrected is box:
            continue

        half = corrected[3:6] / 2
        offset = reference.local(corrected[None, :3], box)[0]
        assert (half > 0).all() and (np.abs(offset) + half <= box[3:6] / 2 + TOL).all()
        assert (np.abs(measured - offset) <= half + TOL).all()
        changed, depth_error = reference.invariance(depth, first_depth(corrected, selection, transform))
        assert changed == 0 and depth_error <= TOL
        audit_changed, _ = reference.invariance(audit_depth, first_depth(corrected, audit, transform))
        row["variants"][arm] = dict(
            box=corrected.tolist(), max_selection_depth_error_m=depth_error,
            selection_changed=changed, audit_changed=audit_changed,
            trim_height_m=float(box[5] - corrected[5]),
            volume_reduction_percent=float(100 * (1 - np.prod(corrected[3:6] / box[3:6]))),
        )
    return row


def fingerprints():
    return {name: roster.sha(HERE / name) for name in ("PROTOCOL.json", "run_study.py", "extent_clip.py", "live.py")}


def propose(workers):
    started = perf_counter()
    protocol = roster.read(PROTOCOL)
    assert not protocol["FINAL_access"]
    assert roster.sha(reference.origin.ACTIVE / "METHOD_LOCK.json") == protocol["frozen_method_sha256"]
    assert roster.sha(SOURCE / "METHOD_LOCK.json") == protocol["source_extent_lock_sha256"]
    prior_path = reference.OUT / "prediction_seal.json"
    prior = roster.read(prior_path)
    assert not prior["target_gt_opened"] and not prior["FINAL_access"]
    assert prior["payload"] == roster.sha(reference.PAYLOAD)
    trace, trace_path = reference.origin.verified_trace()
    assert trace["payload"] == prior["trace"]
    frames, pins, checks, index, official = reference.origin.inputs()
    OUT.mkdir(exist_ok=True)
    PAYLOAD.parent.mkdir(mode=0o700, exist_ok=True)
    temporary = PAYLOAD.with_suffix(".tmp")
    counts = defaultdict(int)
    max_error = 0.
    with gzip.open(trace_path, "rt") as rays, gzip.open(reference.PAYLOAD, "rt") as original, \
            temporary.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as output, \
            ProcessPoolExecutor(max_workers=workers, initializer=reference.initialize, initargs=(index,)) as pool:
        jobs = ((f, old, json.loads(t), json.loads(p), checks[f["uid"]], pins)
                for f, old, t, p in zip(frames, official, rays, original, strict=True))
        for row in pool.map(infer_one, jobs, chunksize=12):
            output.write((json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode())
            counts["descriptions"] += 1
            counts["active"] += row["active"]
            for arm in ARMS[1:]:
                counts[arm] += arm in row["variants"]
            max_error = max(max_error, row["full_error_m"])
            if counts["descriptions"] % 1000 == 0:
                print("Replayed", counts["descriptions"], "/ 9310", flush=True)
    assert counts["descriptions"] == 9310 and counts["active"] == 5910 and counts["ray_equivalent"] == 5904
    temporary.replace(PAYLOAD)
    roster.write(OUT / "prediction_seal.json", dict(
        counts=dict(counts), all_axis_max_error_m=max_error, files=fingerprints(),
        source_predictions=roster.sha(prior_path), payload=roster.sha(PAYLOAD), workers=workers,
        target_gt_opened=False, FINAL_access=False, new_Qwen_queries=0, production_prediction_changes=0,
        seconds=perf_counter() - started,
    ))
    print(dict(counts), "maximum all-axis box difference", max_error, flush=True)


def score_one(job):
    frame, sealed, previous = job
    assert frame["uid"] == sealed["uid"] == previous["uid"] and frame["panel"] == "validation"
    assert sealed["active"] == (previous["active"] == "True")
    variants = sealed["variants"]
    assert ("ray_equivalent" in variants) == (previous["changed_ray_equivalent"] == "True")
    baseline = float(previous["iou_baseline"])
    old_full = float(previous["iou_ray_equivalent"])
    scores = dict.fromkeys(ARMS[1:], baseline)
    if variants:
        path = Path(frame["calibration"])
        assert path.name == "meta_info.json" and path.parents[2] == reference.DATA / frame["target"]
        annotations = roster.read(path)["ground_info"]
        assert annotations[frame["slot"]]["class"].strip().lower() == previous["category"]
        truth = scorer.canonical_box(scorer.truth_for_slot(frame, annotations))
        values, _ = scorer.score_boxes(np.asarray([sealed["box"], *(v["box"] for v in variants.values())]), truth)
        assert abs(float(values[0]) - baseline) <= 1e-12
        scores.update(zip(variants, map(float, values[1:]), strict=True))
    assert abs(scores["ray_equivalent"] - old_full) <= 1e-10
    assert all((scores["ray_equivalent"] > t) == (old_full > t) for t in (.25, .50))

    row = {k: previous[k] for k in ("uid", "target", "recording", "category", "case")}
    row.update(active=sealed["active"], iou_baseline=baseline, gt_scored=bool(variants),
               full_error_m=sealed["full_error_m"], selection_rays=sealed.get("selection_rays", 0),
               audit_rays=sealed.get("audit_rays", 0), protected_points=sealed.get("protected_points", 0))
    for arm, score in scores.items():
        row[f"iou_{arm}"] = score
        row[f"changed_{arm}"] = arm in variants
        for key in ("selection_changed", "audit_changed", "max_selection_depth_error_m",
                    "trim_height_m", "volume_reduction_percent"):
            row[f"{arm}_{key}"] = variants.get(arm, {}).get(key, 0)
    return row


def ray_summary(rows):
    for arm in ARMS[1:]:
        changed = [r for r in rows if r[f"changed_{arm}"]]
        yield dict(
            arm=arm, n=len(rows), changed=len(changed),
            selection_rays=sum(r["selection_rays"] for r in changed),
            selection_changed=sum(r[f"{arm}_selection_changed"] for r in changed),
            max_selection_depth_error_m=max((r[f"{arm}_max_selection_depth_error_m"] for r in changed), default=0.),
            audit_rays=sum(r["audit_rays"] for r in changed),
            audit_changed=sum(r[f"{arm}_audit_changed"] for r in changed),
            protected_points=sum(r["protected_points"] for r in changed),
            median_trim_height_m=float(np.median([r[f"{arm}_trim_height_m"] for r in changed])) if changed else None,
            median_volume_reduction_percent=float(np.median([r[f"{arm}_volume_reduction_percent"] for r in changed])) if changed else None,
        )


def evaluate(workers):
    started = perf_counter()
    seal_path = OUT / "prediction_seal.json"
    seal = roster.read(seal_path)
    assert not seal["target_gt_opened"] and not seal["FINAL_access"]
    assert seal["files"] == fingerprints() and seal["payload"] == roster.sha(PAYLOAD)
    old_receipt = roster.read(reference.OUT / "evaluation_receipt.json")
    assert old_receipt["prediction_seal"] == seal["source_predictions"]
    previous = {}
    for platform, count in PLATFORM_COUNTS.items():
        path = reference.OUT / f"rows_{platform}.csv.gz"
        assert old_receipt["files"][path.name] == roster.sha(path)
        with gzip.open(path, "rt", newline="") as stream:
            items = list(csv.DictReader(stream))
        assert len(items) == count and all(r["target"] == platform for r in items)
        previous.update((r["uid"], r) for r in items)

    frames, _, _, _, _ = reference.origin.inputs()
    with gzip.open(PAYLOAD, "rt") as stream, ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = ((f, json.loads(line), previous[f["uid"]]) for f, line in zip(frames, stream, strict=True))
        rows = list(pool.map(score_one, jobs, chunksize=12))
    assert len(rows) == len({r["uid"] for r in rows}) == len(previous) == seal["counts"]["descriptions"]
    assert sum(r["active"] for r in rows) == seal["counts"]["active"]
    assert all(sum(r[f"changed_{a}"] for r in rows) == seal["counts"][a] for a in ARMS[1:])
    assert all(r["height_only_selection_changed"] == r["ray_equivalent_selection_changed"] == 0 for r in rows)

    groups = defaultdict(list)
    for row in rows:
        keys = (("macro", "all"), ("pooled", "all"), ("platform", row["target"]), ("category", row["category"]),
                ("recording", row["target"] + "/" + row["recording"]), ("case", row["case"]))
        for key in keys:
            groups[key].append(row)
    scorer.ARMS, scorer.PAIRS, scorer.OUT = ARMS, PAIRS, OUT
    metrics, paired, witness = [], [], []
    for (group, value), items in sorted(groups.items()):
        key = dict(group=group, value=value)
        metrics.extend(dict(**key, **r) for r in scorer.table_row_metrics(items, group))
        paired.extend(dict(**key, **r) for r in scorer.table_row_pairs(items, group))
        witness.extend(dict(**key, **r) for r in ray_summary(items))
    files = {name: scorer.write_rows(name, table) for name, table in (
        ("metrics.csv", metrics), ("paired.csv", paired), ("ray_checks.csv", witness),
    )}
    files["rows.csv.gz"] = scorer.write_rows("rows.csv.gz", rows, compress=True)
    roster.write(OUT / "evaluation_receipt.json", dict(
        counts=seal["counts"], ordinary_GT_descriptions=sum(r["gt_scored"] for r in rows),
        prediction_seal=roster.sha(seal_path), files=files, intervals_pp=scorer.bootstraps(rows),
        all_axis_predictions_and_thresholds_reproduced=True,
        FINAL_access=False, new_Qwen_queries=0, production_prediction_changes=0,
        seconds=perf_counter() - started,
    ))
    for pair in paired:
        if pair["group"] == "macro":
            print(pair, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("propose", "evaluate"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    (propose if args.phase == "propose" else evaluate)(args.workers)
