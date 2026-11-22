"""Replay both deployment readouts through the normal frozen-reply interface."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from time import perf_counter

import numpy as np

from live import predict
from run_study import ARMS, OUT, PAYLOAD, fingerprints, reference, roster


def one(job):
    frame, sealed, base_row, check, pins = job
    assert frame["uid"] == base_row["uid"] == sealed["uid"] and frame["panel"] == "validation"
    cached = reference.origin.prediction(frame, check, pins)
    replies = reference.origin.replies(frame, cached, reference.replies_hashes)
    base = base_row["simple4"]
    changed = {}
    for arm in ARMS[1:]:
        actual = predict(frame, replies, height_only=arm == "height_only")
        correction = sealed["variants"].get(arm)
        box = correction["box"] if correction is not None else base["box"]
        assert actual["extent_corrected"] == (correction is not None)
        assert {k: actual[k] for k in base if k != "box"} == {k: base[k] for k in base if k != "box"}
        np.testing.assert_array_equal(actual["box"], box)
        changed[arm] = correction is not None
    return changed


def run(workers):
    started = perf_counter()
    seal_path = OUT / "prediction_seal.json"
    seal = roster.read(seal_path)
    assert seal["payload"] == roster.sha(PAYLOAD) and seal["files"] == fingerprints()
    assert not seal["target_gt_opened"] and not seal["FINAL_access"]
    frames, pins, checks, index, official = reference.origin.inputs()
    counts = dict.fromkeys(ARMS[1:], 0)
    total = 0
    with gzip.open(PAYLOAD, "rt") as stream, \
            ProcessPoolExecutor(max_workers=workers, initializer=reference.initialize, initargs=(index,)) as pool:
        jobs = ((f, json.loads(line), old, checks[f["uid"]], pins) for f, line, old in zip(frames, stream, official, strict=True))
        for changed in pool.map(one, jobs, chunksize=12):
            total += 1
            for arm, value in changed.items():
                counts[arm] += value
    assert total == seal["counts"]["descriptions"]
    assert all(n == seal["counts"][arm] for arm, n in counts.items())
    roster.write(OUT / "live_replay.json", dict(
        descriptions=total, changed=counts, exact_normal_entry_boxes=True,
        source_prediction_seal=roster.sha(seal_path), code=roster.sha(__file__),
        FINAL_access=False, new_Qwen_queries=0, seconds=perf_counter() - started,
    ))
    print("Readouts replayed exactly on", total, "descriptions", counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args().workers)
