"""Reconstruct and seal the original target and negative constraints before GT scoring."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
OUT, WORK = HERE/"results", HERE/"private"
PROTOCOL = HERE/"PROTOCOL.json"
GEOMETRY = WORK/"sealed_constraints.jsonl.gz"
PRIOR = HERE.parent/"3eed_qwen_cafe_exclusion_audit_20261121"
sys.path.insert(0, str(PRIOR))
import extract as cafe
from center_sets import TOL, local, interior, measure

roster = cafe.roster
PLATFORMS = ("waymo", "drone", "quad")
COHORT = {"waymo":425, "drone":1537, "quad":807}


def source_outcomes():
    receipt = roster.read(PRIOR/"results/analysis_receipt.json")
    assert receipt["active_pairs"] == 5910 and receipt["descriptions"] == 9310 and not receipt["FINAL_access"]
    scores = {}
    for platform in PLATFORMS:
        name = f"rows_{platform}.csv.gz"
        path = PRIOR/"results"/name
        assert roster.sha(path) == receipt["files"][name]
        with gzip.open(path, "rt", newline="") as f:
            for row in csv.DictReader(f):
                if row["original_case"] != "neither_pass":
                    continue
                assert row["target"] == platform and row["production"] == "True" and row["uid"] not in scores
                assert float(row["pca_iou"]) <= .5 and float(row["qwen_iou"]) <= .5
                scores[row["uid"]] = row
    assert Counter(r["target"] for r in scores.values()) == COHORT
    return scores


def focus_poses(row):
    return [int(row[f"witness_{key}"]) > 0 and bool(row[f"{key}_joint_iou_upper"])
            and float(row[f"{key}_joint_iou_upper"]) > .5 for key in ("p", "q")]


def materialize(box, target, witnesses, prior):
    assert measure(box, target, witnesses) == prior
    if not (prior["compatible_xy"] and prior["compatible_z"]):
        return dict(positive=None, forbidden=[])
    half = box[3:6]/2
    support = local(target, box)
    lower = support[:, :2].max(0)-half[:2]
    upper = support[:, :2].min(0)+half[:2]
    center = (lower+upper)/2
    lower, upper = np.minimum(lower, center), np.maximum(upper, center)
    neg = local(witnesses, box)
    xy = neg[np.abs(neg[:, 2]) < half[2]-TOL, :2]
    lo, hi = xy-half[:2]+TOL, xy+half[:2]-TOL
    live = ((lo < hi) & (lo < upper) & (hi > lower)).all(1)
    return dict(positive=[lower.tolist(), upper.tolist()], forbidden=np.stack((lo[live], hi[live]), axis=1).tolist())


def initialize(index):
    global replies_index
    replies_index = index


def reconstruct(job):
    frame, trace, prior, row, digest, pins = job
    uid = frame["uid"]
    assert uid == trace["uid"] == prior["uid"] == row["uid"] and frame["panel"] == "validation"
    assert trace["production"] and prior["production"]
    assert trace["choose_qwen"] == prior["choose_qwen"] == (row["choose_qwen"] == "True")
    assert roster.file_sha(frame["points"]) == frame["point_sha"]
    frozen = cafe.prediction(frame, digest, pins)
    window = cafe.replies(frame, frozen, replies_index)["2d"]["value"]
    p, q, held, matrix = (np.asarray(trace[key], float) for key in ("p", "q", "held", "transform"))
    sample, ground, support, sensor_model, sensor, target, in_win, selected = cafe.context(frame, window, held, matrix)
    in_p, in_q = interior(sensor, p), interior(sensor, q)
    xor = in_p ^ in_q
    birth, level, labels = cafe.stable_labels(sample, ground, support, sensor_model[xor])
    wp, nsp, ngp = cafe.corroborate(labels, in_p[xor])
    wq, nsq, ngq = cafe.corroborate(labels, in_q[xor])
    assert not (selected[xor] & (labels >= 0)).any()
    observed = dict(original_selection_cells=len(held), scene_selection_cells=len(sensor), target_fit_seeds=len(support),
                    birth=cafe.FOUR_SCALES[birth], fine=cafe.FOUR_SCALES[level], coarse=cafe.FOUR_SCALES[level+1],
                    xor_p=int(in_p[xor].sum()), xor_q=int(in_q[xor].sum()), stable_p=nsp, stable_q=nsq,
                    witness_p=int(wp.sum()), witness_q=int(wq.sum()), regions_p=ngp, regions_q=ngq,
                    in_window_p=int((wp & in_win[xor]).sum()), in_window_q=int((wq & in_win[xor]).sum()))
    assert all(prior[key] == value for key, value in observed.items())
    witnesses = sensor[xor][wp | wq]
    constraints = [materialize(box, target, witnesses, prior[f"{key}_set"]) for box, key in ((p,"p"),(q,"q"))]
    return dict(uid=uid, boxes=[p.tolist(), q.tolist()], constraints=constraints,
                focus_poses=focus_poses(row), witness_counts=[int(wp.sum()),int(wq.sum())])


def run(workers):
    start = perf_counter()
    protocol = roster.read(PROTOCOL)
    assert not protocol["FINAL_access"] and protocol["prediction_changes"] == 0
    lock = roster.sha(cafe.ACTIVE/"METHOD_LOCK.json")
    assert lock == protocol["source_method_sha256"]
    prior_seal_path = PRIOR/"results/evidence_seal.json"
    prior_seal = roster.read(prior_seal_path)
    ray_seal, traces = cafe.verified_trace()
    assert ray_seal["payload"] == prior_seal["source_trace_payload"]
    assert prior_seal["payload"] == roster.sha(cafe.PAYLOAD)
    assert prior_seal["extractor"] == roster.sha(PRIOR/"extract.py")
    assert prior_seal["geometry"] == roster.sha(PRIOR/"center_sets.py")
    assert not prior_seal["FINAL_access"] and not prior_seal["iou_rows_opened"]
    frames, pins, checks, versions, _ = cafe.inputs()
    scores = source_outcomes()
    temporary = GEOMETRY.with_suffix(".tmp")
    n = focus = 0
    with gzip.open(traces, "rt") as trace, gzip.open(cafe.PAYLOAD, "rt") as witnesses, temporary.open("wb") as stream, \
            gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0, compresslevel=4) as output, \
            ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(versions,)) as pool:
        jobs = ((f,json.loads(t),json.loads(w),scores[f["uid"]],checks[f["uid"]],pins)
                for f,t,w in zip(frames,trace,witnesses,strict=True) if f["uid"] in scores)
        for r in pool.map(reconstruct, jobs, chunksize=12):
            output.write((json.dumps(r, separators=(",",":"), allow_nan=False)+"\n").encode())
            n += 1
            focus += any(r["focus_poses"])
            if n%500 == 0:
                print("Replayed constraints", n, "/ 2769", flush=True)
    assert n == len(scores) == 2769 and focus == 388
    temporary.replace(GEOMETRY)
    roster.write(OUT/"constraint_seal.json", dict(
        descriptions=n, focus=focus, cohort=COHORT, payload=roster.sha(GEOMETRY), code=roster.sha(__file__),
        protocol=roster.sha(PROTOCOL), previous_witness_seal=roster.sha(prior_seal_path),
        previous_analysis=roster.sha(PRIOR/"results/analysis_receipt.json"), frozen_method_lock=lock,
        verified_original_witness_counts=True, verified_original_set_envelopes=True,
        new_GT_reads=0, original_predictions_changed=0, FINAL_access=False, seconds=perf_counter()-start,
    ))
    print("Sealed", n, "pairs, including", focus, "focus descriptions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args().workers)
