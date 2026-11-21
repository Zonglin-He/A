"""Score diagnostic-only horizontal maxima against the pinned Ordinary GT."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from prepare import COHORT, GEOMETRY, HERE, OUT, PRIOR, PROTOCOL, cafe, focus_poses, roster, source_outcomes
from sweep import CHECK_TOL, HorizontalSweep, center_pieces
from score_ablation import DATA
from score_full import canonical_box, score_boxes, truth_for_slot


VARIANTS = ("free", "positive", "joint")
POSES = ("p", "q")


def constraint_check(xy, bounds, exclusions):
    lower, upper = np.asarray(bounds)
    assert ((lower-CHECK_TOL <= xy) & (xy <= upper+CHECK_TOL)).all()
    if exclusions:
        rectangles = np.asarray(exclusions)
        assert not ((rectangles[:,0]+CHECK_TOL < xy) & (xy < rectangles[:,1]-CHECK_TOL)).all(1).any()


def score_one(job):
    frame, saved, previous = job
    uid = frame["uid"]
    assert uid == saved["uid"] == previous["uid"]
    assert frame["panel"] == "validation" and frame["target"] == previous["target"]
    assert frame["recording"] == previous["recording"] and frame["frame"] == previous["frame"]
    assert saved["focus_poses"] == focus_poses(previous)
    path = Path(frame["calibration"])
    assert path.name == "meta_info.json" and path.parents[2] == DATA/frame["target"]
    items = roster.read(path)["ground_info"]
    assert items[frame["slot"]]["class"].strip().lower() == previous["category"]
    truth = canonical_box(truth_for_slot(frame, items))
    output = {k:previous[k] for k in ("uid", "target", "recording", "frame", "category")}
    output.update(original_qwen=previous["choose_qwen"]=="True", focus=any(saved["focus_poses"]))
    max_qerr = max_rerr = 0.
    for side, name, box, constraint in zip(POSES, ("pca", "qwen"), saved["boxes"], saved["constraints"], strict=True):
        index = POSES.index(side)
        box = np.asarray(box, float)
        solver = HorizontalSweep(box, truth)
        bounds, forbidden = constraint["positive"], constraint["forbidden"]
        pieces = center_pieces(bounds, forbidden)
        assert bool(pieces) == (previous[f"{side}_joint_possible"] == "True")
        assert (bounds is not None) == bool(previous[f"{side}_positive_area"])
        if bounds is not None:
            area = sum(float(np.prod(hi-lo)) for lo,hi in pieces)
            assert abs(area-float(previous[f"{side}_joint_area"])) < 1.e-9
        xy_p, value_p, err_p = solver.optimum(center_pieces(bounds, []))
        # Identical feasible sets share their exact maximizer.
        xy_j, value_j, err_j = solver.optimum(pieces) if forbidden else (xy_p, value_p, err_p)
        max_qerr = max(max_qerr, err_p, err_j)
        local_solutions = ((solver.center, solver.free_value), (xy_p, value_p), (xy_j, value_j))
        boxes = [box]
        checks = [float(previous[f"{name}_iou"])]
        attained = []
        for arm, (position, value) in zip(VARIANTS, local_solutions, strict=True):
            if position is None:
                assert value is None
                attained.append(None)
                continue
            if arm != "free":
                constraint_check(position, bounds, forbidden if arm == "joint" else [])
                assert value <= float(previous[f"{side}_{arm}_iou_upper"])+CHECK_TOL
            moved = solver.translate(position)
            np.testing.assert_array_equal(moved[2:], box[2:])
            attained.append(len(boxes))
            boxes.append(moved)
            checks.append(value)
        scores, mesh_error = score_boxes(np.asarray(boxes), truth)
        error = float(np.max(np.abs(scores-checks)))
        max_rerr = max(max_rerr, error, mesh_error)
        assert error < CHECK_TOL and np.array_equal(scores>.5, np.asarray(checks)>.5)
        assert scores[0] <= .5
        output[f"{side}_original_iou"] = float(scores[0])
        output[f"{side}_focus_pose"] = bool(saved["focus_poses"][index])
        output[f"{side}_witness_count"] = saved["witness_counts"][index]
        for arm,i in zip(VARIANTS,attained,strict=True):
            output[f"{side}_{arm}"] = None if i is None else float(scores[i])
        if xy_p is not None:
            assert output[f"{side}_positive"] <= output[f"{side}_free"]+CHECK_TOL
        if xy_j is not None:
            assert output[f"{side}_joint"] <= output[f"{side}_positive"]+CHECK_TOL
        if not forbidden:
            assert output[f"{side}_joint"] == output[f"{side}_positive"]
    return output,max_qerr,max_rerr


def within_scope(row, scope):
    if scope == "original_pose":
        return ("q",) if row["original_qwen"] else ("p",)
    if scope == "opposed_poses":
        return tuple(p for p in POSES if row[f"{p}_focus_pose"])
    return POSES


def aggregate(rows, scope):
    result = Counter()
    for r in rows:
        sides = within_scope(r, scope)
        assert sides
        variants = {}
        for variant in VARIANTS:
            legal = [r[f"{s}_{variant}"] for s in sides if r[f"{s}_{variant}"] is not None]
            result[f"{variant}_feasible"] += bool(legal)
            passed = any(x > .5 for x in legal)
            result[f"{variant}_rescues"] += passed
            variants[variant] = passed
        result["positive_rescue_removed"] += variants["positive"] and not variants["joint"]
        result["joint_only_rescue"] += variants["joint"] and not variants["positive"]
        assert not variants["positive"] or variants["free"]
    assert result["joint_only_rescue"] == 0
    return dict(descriptions=len(rows), **result)


def write_table(path, rows):
    def write(handle):
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    if path.suffix == ".gz":
        with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped, \
                io.TextIOWrapper(zipped, encoding="utf-8", newline="") as handle:
            write(handle)
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            write(handle)


def run(workers):
    started = perf_counter()
    seal_path = OUT/"constraint_seal.json"
    seal = roster.read(seal_path)
    assert not seal["FINAL_access"] and seal["new_GT_reads"] == seal["original_predictions_changed"] == 0
    assert seal["payload"] == roster.sha(GEOMETRY) and seal["code"] == roster.sha(HERE/"prepare.py")
    assert seal["protocol"] == roster.sha(PROTOCOL) and seal["frozen_method_lock"] == roster.sha(cafe.ACTIVE/"METHOD_LOCK.json")
    assert seal["previous_analysis"] == roster.sha(PRIOR/"results/analysis_receipt.json")
    previous = source_outcomes()
    all_frames, _, _, _, _ = cafe.inputs()
    frames = {f["uid"]:f for f in all_frames if f["uid"] in previous}
    assert frames.keys() == previous.keys()
    with gzip.open(GEOMETRY, "rt") as f:
        sealed = [json.loads(line) for line in f]
    assert len(sealed) == len({r["uid"] for r in sealed}) == len(previous) == 2769
    assert {r["uid"] for r in sealed} == frames.keys()
    assert sum(any(r["focus_poses"]) for r in sealed) == seal["focus"] == 388
    jobs = [(frames[r["uid"]],r,previous[r["uid"]]) for r in sealed]
    rows = []
    max_quad = max_recheck = 0.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row,quad,recheck in pool.map(score_one,jobs,chunksize=8):
            rows.append(row)
            max_quad = max(max_quad,quad)
            max_recheck = max(max_recheck,recheck)
            if len(rows)%500 == 0:
                print("Scored",len(rows),"/ 2769",flush=True)
    assert len(rows) == 2769 and Counter(r["target"] for r in rows) == COHORT
    assert sum(r["focus"] for r in rows) == 388
    summary = []
    subsets = ("all", "focus", "outside_focus")
    for subset in subsets:
        cohort = rows if subset == "all" else [r for r in rows if r["focus"] == (subset == "focus")]
        for platform in ("all", *COHORT):
            group = cohort if platform == "all" else [r for r in cohort if r["target"] == platform]
            scopes = ("either_pose", "original_pose", "opposed_poses") if subset == "focus" else ("either_pose", "original_pose")
            for scope in scopes:
                summary.append(dict(subset=subset, platform=platform, pose_scope=scope, **aggregate(group,scope)))
    written = ["summary.csv", "rows.csv.gz"]
    write_table(OUT/written[0],summary)
    write_table(OUT/written[1],rows)
    margin = min(abs(value-.5) for row in rows for arm in VARIANTS for pose in POSES
                 if (value:=row[f"{pose}_{arm}"]) is not None)
    assert max_quad < CHECK_TOL and max_recheck < CHECK_TOL and margin > 5*CHECK_TOL
    roster.write(OUT/"measurement_receipt.json",dict(
        constraint_seal=roster.sha(seal_path), score_code=roster.sha(__file__), sweep_code=roster.sha(HERE/"sweep.py"),
        descriptions=len(rows), focus=388, ordinary_GT_descriptions=len(rows), frozen_pose_descriptions=2*len(rows),
        max_quadratic_probe_error=max_quad, max_upright_iou_recheck_error=max_recheck, min_acc50_margin=margin,
        files={name:roster.sha(OUT/name) for name in written}, no_size_z_yaw_changes=True,
        original_predictions_changed=0, new_Qwen_queries=0, training=False, FINAL_access=False,
        seconds=perf_counter()-started,
    ))
    for row in summary:
        if row["subset"] != "outside_focus" and row["platform"] == "all":
            print(row,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=8)
    run(parser.parse_args().workers)
