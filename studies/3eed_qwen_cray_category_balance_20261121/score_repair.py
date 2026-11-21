"""Score sealed alternatives with the original pair of oriented IoU implementations."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from time import perf_counter

import numpy as np

from locked_roster import HERE, OUT, WORK, locked_inputs, prediction, roster, score_full
import verify_full


ARMS = ("H", "C_pca", "C_ray", "F_complete", "B_pca", "B_ray",
        "C_ray_applicable", "C_ray_shuffled")
PAIRS = (
    ("C_pca", "H"), ("C_ray", "C_pca"), ("C_ray", "H"),
    ("F_complete", "H"), ("C_pca", "F_complete"), ("B_pca", "C_pca"),
    ("B_ray", "B_pca"), ("B_ray", "C_ray"), ("B_ray", "H"),
    ("C_ray_applicable", "H"), ("C_ray_applicable", "C_ray"),
    ("C_ray_applicable", "C_ray_shuffled"), ("C_ray_shuffled", "H"),
)


def score_one(job):
    frame, new, checksum, pins, reference = job
    old = prediction(frame, checksum, pins)
    uid = frame["uid"]
    assert new["uid"] == uid and new["previous"] == checksum
    original_boxes = old["boxes"]
    boxes = {k: original_boxes[k] for k in ARMS[:3]}
    boxes.update(new["boxes"])
    boxes["C_ray_shuffled"] = boxes["C_ray" if new["shuffled_use_ray"] else "H"]
    assert new["use_ray"] == (new["ray_eligible"] and new["label_2d_vehicle"] and new["label_3d_vehicle"])
    assert boxes["C_ray_applicable"] == boxes["C_ray" if new["use_ray"] else "H"]
    assert all(arm in boxes for arm in ARMS)
    if boxes["H"] is None:
        assert all(b is None for b in boxes.values())
        ious, error = np.zeros(len(ARMS)), 0.
    else:
        truth = score_full.canonical_box(score_full.truth_for_slot(
            frame, roster.read(frame["calibration"])["ground_info"]))
        ious, error = score_full.score_boxes(np.array([boxes[k] for k in ARMS]), truth)
    for arm, value in zip(ARMS[:3], ious[:3], strict=True):
        assert abs(value - float(reference["iou_"+arm])) < 1.e-12
    data = {k: reference[k] for k in ("uid", "target", "recording", "category", "frame")}
    data.update({k: new[k] for k in ("use_ray", "shuffled_use_ray", "ray_eligible",
                "label_2d_vehicle", "label_3d_vehicle", "bounded_pca", "bounded_qwen",
                "bounded_chooses_qwen", "fit_points")})
    data.update(full=old["full"], strict=old["strict"], old_chooses_qwen=old["chooses_qwen"])
    data.update({"iou_"+k: float(v) for k, v in zip(ARMS, ious, strict=True)})
    return data, error


def aggregation_groups(rows):
    groups = defaultdict(list)
    for r in rows:
        t, c = r["target"], r["category"]
        for key in (("pooled", "all"), ("macro", "all"), ("platform", t), ("category", c),
                    ("platform_category", t + "/" + c), ("recording", t + "/" + r["recording"])):
            groups[key].append(r)
        if r["full"] and r["strict"]:
            groups["matched_support_category", c].append(r)
    return groups


def summarize(rows):
    # Share the prior aggregation implementation with the same thresholds and bootstrap.
    score_full.ARMS = ARMS
    score_full.PAIRS = PAIRS
    groups = aggregation_groups(rows)
    metrics, pairs = [], []
    for (group, value), block in sorted(groups.items()):
        aggregation = "macro" if group == "macro" else "pooled"
        metrics.extend(dict(group=group, value=value, **r) for r in score_full.table_row_metrics(block, aggregation))
        pairs.extend(dict(group=group, value=value, **r) for r in score_full.table_row_pairs(block, aggregation))
    selection = []
    for (g, name), block in sorted(groups.items()):
        if g not in ("platform", "category", "platform_category", "recording", "pooled"):
            continue
        keys = ("use_ray", "shuffled_use_ray", "ray_eligible", "label_2d_vehicle", "label_3d_vehicle",
                "bounded_pca", "bounded_qwen", "bounded_chooses_qwen", "old_chooses_qwen", "full", "strict")
        selection.append(dict(group=g, value=name, n=len(block), **{k:sum(r[k] for r in block) for k in keys}))
    return metrics, pairs, selection


def write_table(file, rows):
    target = OUT / file
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return roster.sha(target)


def run(workers):
    start = perf_counter()
    frames, _, historical, previous, _ = locked_inputs()
    hold = roster.read(OUT/"prediction_seal.json")
    assert not hold["target_labels_read"] and not hold["FINAL_access"]
    assert hold["historical"] == roster.sha(roster.OUT/"prediction_seal.json")
    assert hold["protocol"] == roster.sha(HERE/"PROTOCOL.json")
    assert hold["code"] == roster.sha(HERE/"predict_variants.py")
    assert hold["selector"] == roster.sha(HERE/"cray.py")
    payload = WORK/"predictions.json"
    assert roster.sha(payload) == hold["payload"]
    pred = roster.read(payload)
    archived = roster.read(roster.OUT/"evaluation_receipt.json")
    source_tables = tuple("rows_"+t+".csv.gz" for t in score_full.PLATFORMS)
    assert all(roster.sha(roster.OUT/name) == archived["files"][name] for name in source_tables)
    lookup = {r["uid"]:r for name in source_tables for r in verify_full.table(name)}
    assert hold["descriptions"] == len(pred) == len(lookup) == len(frames) == 9310
    jobs = ((f, p, previous[f["uid"]], historical["pins"], lookup[f["uid"]])
            for f, p in zip(frames, pred, strict=True))
    values, max_error = [], 0.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row, error in pool.map(score_one, jobs, chunksize=24):
            values.append(row)
            max_error = max(max_error, error)
    assert len(values) == 9310 and max_error < 1.e-7
    metrics, pairs, selection = summarize(values)
    files = {"metrics.csv": write_table("metrics.csv", metrics),
             "paired.csv": write_table("paired.csv", pairs),
             "coverage.csv": write_table("coverage.csv", selection)}
    for platform in score_full.PLATFORMS:
        name = f"rows_{platform}.csv.gz"
        files[name] = write_anonymous(name, [r for r in values if r["target"] == platform])
    roster.write(OUT/"evaluation_receipt.json", dict(
        seal=roster.sha(OUT/"prediction_seal.json"), code=roster.sha(__file__), files=files,
        descriptions=len(values), recordings=len({r["recording"] for r in values}),
        max_two_iou_scorer_error=max_error, exact_old_iou_rows=len(values),
        recording_intervals_pp=score_full.bootstraps(values), FINAL_access=False,
        seconds=perf_counter()-start))
    print("Scored", len(values), "with sealed alternatives and verified historical IoUs.", flush=True)


def write_anonymous(name, rows):
    # The shared writer generates deterministic gzip bytes. Its output directory
    # belongs to the old experiment unless set explicitly here.
    previous = score_full.OUT
    score_full.OUT = OUT
    try:
        return score_full.write_rows(name, rows, compress=True)
    finally:
        score_full.OUT = previous


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    run(parser.parse_args().workers)
