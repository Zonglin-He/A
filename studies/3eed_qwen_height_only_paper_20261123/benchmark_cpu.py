"""Time the locked live readouts and the extra height calculation on one CPU."""

from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from paper_eval import MAIN, OUT, PLATFORMS, locked, reference, roster, scorer
from live import first_depth, frozen_predict, incumbent, predict
from extent_clip import clip_extent, protected_points


def selected_inputs():
    frames, pins, checks, index, official = reference.origin.inputs()
    source = roster.read(locked.OUT / "prediction_seal.json")
    assert source["payload"] == roster.sha(locked.PAYLOAD) and not source["target_gt_opened"]
    blocks = defaultdict(list)
    with gzip.open(locked.PAYLOAD, "rt") as stream:
        for frame, incumbent_row, line in zip(frames, official, stream, strict=True):
            frozen = json.loads(line)
            assert frozen["uid"] == frame["uid"] == incumbent_row["uid"]
            if frozen["active"] and "ray_equivalent" not in frozen["variants"]:
                continue
            blocks[frame["target"], frozen["active"]].append((frame, incumbent_row["simple4"], frozen))
    chosen = []
    for platform in PLATFORMS:
        for active, n in ((True, 24), (False, 8)):
            ordered = sorted(blocks[platform, active], key=lambda x: roster.key("cpu-paper", x[0]["uid"]))
            for frame, original, expected in ordered[:n]:
                archived = reference.origin.prediction(frame, checks[frame["uid"]], pins)
                replies = reference.origin.replies(frame, archived, index)
                chosen.append((frame, replies, original, expected))
    assert len(chosen) == 96 and len({record[0]["uid"] for record in chosen}) == 96
    return chosen


def correct(box, dimensions, rotation, fit, selection, _, transform):
    depths = first_depth(box, selection, transform)
    witnesses = protected_points(box, fit, selection, depths, transform)
    return clip_extent(box, dimensions, rotation, *witnesses, height_only=True)


def timed(fn, *args, **kwargs):
    start = perf_counter_ns()
    value = fn(*args, **kwargs)
    return value, (perf_counter_ns() - start) / 1e6


def describe(times):
    return {k: float(x) for k, x in zip(("median_ms", "p90_ms", "mean_ms"),
                                        (np.median(times), np.quantile(times, .9), np.mean(times)))}


def run():
    assert all(os.environ.get(k) == "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"))
    core = max(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {core})
    data = selected_inputs()
    out = []
    for frame, reply, original, archived in data:
        uid = frame["uid"]
        metadata, context = incumbent(frame, reply)
        assert metadata == original and bool(context is not None) == archived["active"]
        baselines = [original["box"], *[archived["variants"].get(a, {}).get("box", original["box"])
                                      for a in ("ray_equivalent", "height_only")]]
        runners = ((frozen_predict, {}), (predict, {"height_only": False}), (predict, {"height_only": True}))
        # Warm identical decoded inputs and the OS cache before measuring the full geometry pass.
        for fn, kwargs in runners:
            fn(frame, reply, **kwargs)
        rng = np.random.default_rng(int(uid[:16], 16))
        for repeat in range(3):
            row = dict(uid=uid, target=frame["target"], active=archived["active"], repeat=repeat)
            for i in rng.permutation(3):
                fn, kwargs = runners[i]
                prediction, delay = timed(fn, frame, reply, **kwargs)
                assert prediction["box"] == baselines[i]
                row[("frozen_ms", "all_axis_ms", "height_only_ms")[i]] = delay
            row["only_extra_ms"] = None
            if context is not None:
                output, extra = timed(correct, *context)
                assert output.tolist() == baselines[2]
                row["only_extra_ms"] = extra
            row["paired_difference_ms"] = row["height_only_ms"] - row["frozen_ms"]
            out.append(row)
    assert len(out) == 288
    scorer.OUT = OUT
    checks = scorer.write_rows("time_cpu.csv", out)
    grouped = {}
    for name, rows in (("all_96", out), ("branch_72", [r for r in out if r["active"]]),
                       ("fallback_24", [r for r in out if not r["active"]])):
        grouped[name] = {k: describe([r[k] for r in rows if r[k] is not None])
                         for k in ("frozen_ms", "all_axis_ms", "height_only_ms", "only_extra_ms", "paired_difference_ms")
                         if any(r[k] is not None for r in rows)}
    source = roster.OUT / "qwen_seal.json"
    gpu = roster.read(source)
    cpu = next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
               if l.startswith("model name"))
    roster.write(OUT / "timing.json", dict(
        cpu=cpu, logical_cpu=core, cpu_threads=1, comparisons=grouped, tested_descriptions=len(data),
        repeats=3, warmed_io=True, calls_per_method=len(out), sha256=checks,
        entry_sha256=roster.sha(MAIN / "live.py"), script_sha256=roster.sha(__file__),
        qwen_reference=dict(sha256=roster.sha(source), gpu=gpu["gpu"], seconds=gpu["seconds"],
                           produced=gpu["produced_this_attempt"], peak_bytes=gpu["memory_peak_bytes"],
                           replies_per_second=gpu["produced_this_attempt"] / gpu["seconds"], batch=gpu["pins"]["batch"]),
        FINAL_access=False))
    print("Timed 96 descriptions, 3 matched repeats per readout", flush=True)


if __name__ == "__main__":
    run()
