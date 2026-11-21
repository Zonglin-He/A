"""Run both readouts on the sealed ordinary cohort, then score their frozen boxes."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from cray_simple import FOUR_SCALES, TWO_SCALES, predict, vehicle_reply
from locked_roster import locked_inputs, prediction, replies, roster, score_full
import verify_full


HERE = Path(__file__).resolve().parent
OUT = HERE/"results"
WORK = HERE/"private"
BASE = HERE.parent/"3eed_qwen_cray_category_balance_20261121"
ARMS = ("current", "simple4", "simple2")
PAIRS = (("simple4", "current"), ("simple2", "simple4"), ("simple2", "current"))
TOL = 5.e-10


def source():
    frames, _, old, hashes, replies_index = locked_inputs()
    typed = roster.read(BASE/"results/prediction_seal.json")
    assert typed["historical"] == roster.sha(roster.OUT/"prediction_seal.json")
    assert typed["descriptions"] == 9310 and not typed["target_labels_read"] and not typed["FINAL_access"]
    assert roster.sha(BASE/"cray.py") == typed["selector"]
    path = BASE/"private/predictions.json"
    assert roster.sha(path) == typed["payload"]
    return frames, old["pins"], hashes, replies_index, roster.read(path)


def check_box(box, expected):
    if expected is None:
        assert box is None
        return 0., True
    assert box is not None
    error = float(np.max(np.abs(np.array(box)-expected)))
    assert error <= TOL, (error, box, expected)
    return error, box == expected


def initialize_reply_hashes(index):
    global reply_hashes
    reply_hashes = index


def decode_one(job):
    frame, checksum, pins, typed = job
    uid = frame["uid"]
    try:
        ground = prediction(frame, checksum, pins)
        assert typed["uid"] == uid and typed["previous"] == checksum
        assert roster.file_sha(frame["points"]) == frame["point_sha"]
        assert roster.file_sha(frame["rgb"]) == frame["rgb_sha"]
        shared = replies(frame, ground, reply_hashes)
        four = predict(frame, shared)
        road = vehicle_reply(shared["2d"])
        expected_ray = road and ground["selection_rays"] >= 5
        expected = ground["boxes"]["C_ray" if expected_ray else "H"]
        error, exact = check_box(four["box"], expected)
        assert four["applicable"] == expected_ray
        assert four["selection_rays"] == (ground["selection_rays"] if road else 0)
        if expected_ray:
            assert (four["heading"] == "qwen") == ground["chooses_qwen"]
        assert typed["boxes"]["C_ray_applicable"] == ground["boxes"]["C_ray" if typed["use_ray"] else "H"]
        two = predict(frame, shared, vehicle_scales=TWO_SCALES) if road else four
        return dict(
            uid=uid, current=typed["boxes"]["C_ray_applicable"], old_use_ray=typed["use_ray"],
            simple4=four, simple2=two, expected_current_geometry=expected, vehicle_2d=road,
            equivalence_error=error, byte_exact_coordinates=exact,
        )
    except Exception as e:
        raise RuntimeError(f"Prediction verification failed for {frame['target']} description {uid}") from e


def predict_all(workers):
    t0 = perf_counter()
    protocol = roster.read(HERE/"PROTOCOL.json")
    assert tuple(protocol["four_scales"]) == FOUR_SCALES and tuple(protocol["two_scales"]) == TWO_SCALES
    assert not protocol["FINAL_access"] and protocol["new_Qwen_queries"] == 0
    frames, pins, hashes, reply_index, current = source()
    jobs = ((f, hashes[f["uid"]], pins, b) for f, b in zip(frames, current, strict=True))
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_reply_hashes, initargs=(reply_index,)) as pool:
        for result in pool.map(decode_one, jobs, chunksize=8):
            rows.append(result)
            if len(rows)%1000 == 0:
                print(f"Rebuilt {len(rows)}/9310", flush=True)
    assert len(rows) == 9310 and len({r["uid"] for r in rows}) == 9310
    file = WORK/"predictions.json"
    roster.write(file, rows)
    roster.write(OUT/"prediction_seal.json", dict(
        cohort=roster.sha(roster.OUT/"input_seal.json"), model=roster.sha(roster.OUT/"qwen_seal.json"),
        current=roster.sha(BASE/"results/prediction_seal.json"), payload=roster.sha(file),
        code=roster.sha(HERE/"cray_simple.py"), evaluation_code=roster.sha(__file__),
        protocol=roster.sha(HERE/"PROTOCOL.json"), descriptions=len(rows),
        exactly_equal_to_2d_only_historical_box=sum(r["byte_exact_coordinates"] for r in rows),
        max_coordinate_error_vs_2d_only_historical_box=max(r["equivalence_error"] for r in rows),
        newly_eligible_2d_only=sum(r["simple4"]["applicable"] and not r["old_use_ray"] for r in rows),
        reused_same_fallback_for_two_scales=sum(not r["vehicle_2d"] for r in rows),
        new_Qwen_queries=0, target_labels_read=False, FINAL_access=False, seconds=perf_counter()-t0,
    ))
    print("Sealed both scale readouts on all 9310 descriptions", flush=True)


def measure_one(job):
    frame, pred, reference = job
    uid = frame["uid"]
    assert pred["uid"] == reference["uid"] == uid
    assert pred["old_use_ray"] == (reference["use_ray"] == "True")
    described = roster.read(frame["calibration"])["ground_info"]
    assert described[frame["slot"]]["class"].strip().lower() == reference["category"]
    truth = score_full.canonical_box(score_full.truth_for_slot(frame, described))
    boxes = [pred["current"], pred["simple4"]["box"], pred["simple2"]["box"], pred["expected_current_geometry"]]
    if boxes[0] is None:
        assert all(box is None for box in boxes)
        ious, error = np.zeros(4), 0.
    else:
        values = np.asarray(boxes, float)
        assert values.shape == (4, 7) and np.isfinite(values).all() and (values[:, 3:6] > 0).all()
        ious, error = score_full.score_boxes(values, truth)
    assert abs(ious[0] - float(reference["iou_C_ray_applicable"])) < 1.e-12
    assert abs(ious[1] - ious[3]) < 1.e-10
    assert all((ious[1] > threshold) == (ious[3] > threshold) for threshold in (.25, .50))
    row = {k: reference[k] for k in ("uid", "recording", "target", "frame", "category")}
    row.update({"iou_"+a: float(iou) for a, iou in zip(ARMS, ious[:3], strict=True)})
    row.update(vehicle_2d=pred["vehicle_2d"], original_uses_ray=pred["old_use_ray"])
    for name in ARMS[1:]:
        for key in ("readout", "selection_rays", "heading"):
            row[f"{name}_{key}"] = pred[name][key]
    return row, error


def groups(rows):
    result = defaultdict(list)
    for r in rows:
        t, c, rec = r["target"], r["category"], r["recording"]
        for key in (("pooled", "all"), ("macro", "all"), ("platform", t), ("category", c),
                    ("platform_category", t+"/"+c), ("recording", t+"/"+rec)):
            result[key].append(r)
    return result


def score_all(workers):
    start = perf_counter()
    frames, _, _, _, _ = source()
    seal = roster.read(OUT/"prediction_seal.json")
    assert seal["cohort"] == roster.sha(roster.OUT/"input_seal.json")
    assert seal["current"] == roster.sha(BASE/"results/prediction_seal.json")
    assert seal["model"] == roster.sha(roster.OUT/"qwen_seal.json")
    assert seal["code"] == roster.sha(HERE/"cray_simple.py") and seal["evaluation_code"] == roster.sha(__file__)
    assert seal["protocol"] == roster.sha(HERE/"PROTOCOL.json")
    assert not seal["target_labels_read"] and not seal["FINAL_access"]
    path = WORK/"predictions.json"
    assert seal["payload"] == roster.sha(path)
    inferred = roster.read(path)
    evaluation = roster.read(BASE/"results/evaluation_receipt.json")
    assert not evaluation["FINAL_access"]
    verify_full.OUT = BASE/"results"
    archived = {}
    for t in score_full.PLATFORMS:
        filename = f"rows_{t}.csv.gz"
        assert roster.sha(BASE/"results"/filename) == evaluation["files"][filename]
        archived.update({row["uid"]: row for row in verify_full.table(filename)})
    assert seal["descriptions"] == len(inferred) == len(archived) == len(frames) == 9310
    jobs = ((f, p, archived[f["uid"]]) for f,p in zip(frames, inferred, strict=True))
    rows, maximum = [], 0.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row, error in pool.map(measure_one, jobs, chunksize=24):
            rows.append(row)
            maximum = max(maximum, error)
    assert len(rows) == 9310 and maximum < 1.e-7

    score_full.ARMS, score_full.PAIRS, score_full.OUT = ARMS, PAIRS, OUT
    metrics, pairs, coverage = [], [], []
    for (level, value), group in sorted(groups(rows).items()):
        agg = "macro" if level == "macro" else "pooled"
        metrics.extend(dict(group=level, value=value, **x) for x in score_full.table_row_metrics(group, agg))
        pairs.extend(dict(group=level, value=value, **x) for x in score_full.table_row_pairs(group, agg))
        if level in ("platform", "category", "pooled"):
            for a in ARMS[1:]:
                modes = Counter(r[a+"_readout"] for r in group)
                coverage.append(dict(group=level, value=value, arm=a, n=len(group),
                                     **{mode:modes[mode] for mode in ("C_ray","H","A","missing_3d")}))
    files = {p: score_full.write_rows(p, table) for p,table in (
        ("metrics.csv", metrics), ("paired.csv", pairs), ("coverage.csv", coverage))}
    for t in score_full.PLATFORMS:
        name = f"rows_{t}.csv.gz"
        files[name] = score_full.write_rows(name, [r for r in rows if r["target"]==t], compress=True)
    roster.write(OUT/"evaluation_receipt.json", dict(
        seal=roster.sha(OUT/"prediction_seal.json"), code=roster.sha(__file__), files=files,
        descriptions=len(rows), recordings=len({r["recording"] for r in rows}),
        max_two_upright_iou_discrepancy=maximum, exact_agreement_with_old_scores=len(rows),
        intervals_pp=score_full.bootstraps(rows), FINAL_access=False, seconds=perf_counter()-start,
    ))
    print("Scored the three matched arms across",len(rows),"descriptions",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "score"))
    parser.add_argument("--workers", default=6, type=int)
    args = parser.parse_args()
    if args.stage == "predict":
        predict_all(args.workers)
    else:
        score_all(args.workers)
