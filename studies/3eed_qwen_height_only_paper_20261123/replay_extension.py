"""Check the unmodified live entry points on every sealed extension prediction."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from time import perf_counter

from paper_eval import OUT, extension_jobs, files, fingerprints, roster
from live import frozen_predict, predict


def replay(job):
    frame, reply, sealed = job
    assert frame["uid"] == sealed["uid"]
    original = frozen_predict(frame, reply)
    protected = predict(frame, reply)
    full = predict(frame, reply, height_only=False)
    assert original["box"] == sealed["boxes"]["baseline"]
    assert protected["box"] == sealed["boxes"]["height_only"]
    assert full["box"] == sealed["boxes"]["ray_equivalent"]
    assert bool(protected["extent_corrected"]) == bool(sealed["checks"].get("height_only", [0])[0])
    return 1


def run(workers):
    start = perf_counter()
    path, seal_path = files("extension")
    seal = roster.read(seal_path)
    assert seal["files"] == fingerprints() and seal["payload"] == roster.sha(path)
    assert not seal["target_gt_opened"] and not seal["FINAL_access"]
    with gzip.open(path, "rt") as stream, ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = ((job[0], job[1], json.loads(row)) for job, row in zip(extension_jobs(), stream, strict=True))
        exact = sum(pool.map(replay, jobs, chunksize=16))
    assert exact == sum(seal["counts"].values()) == 1536
    roster.write(OUT / "extension_live_replay.json", dict(
        matched_descriptions=exact, entrypoints=("C_ray_simple", "ray_equivalent_height_only", "ray_equivalent_full"),
        absolute_box_error_m=0., prediction_seal=roster.sha(seal_path), code_sha256=roster.sha(__file__),
        seconds=perf_counter() - start, opened_ground_truth=False, FINAL_access=False))
    print("All 1,536 descriptions reproduced exactly with the three live readouts", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args().workers)
