"""Bind the new experiments to the existing main method and scoring rules."""

from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "3eed_qwen_height_only_paper_20261123"
OUT, PRIVATE = HERE / "results", HERE / "private"
PROTOCOL = HERE / "PROTOCOL.json"
sys.path.insert(0, str(SOURCE))
import paper_eval as prior


roster, scorer = prior.roster, prior.scorer


def lock(script):
    cfg = roster.read(PROTOCOL)
    pins = prior.fingerprints()
    assert cfg["method_lock_sha256"] == pins["METHOD_LOCK.json"] and not cfg["FINAL_access"]
    OUT.mkdir(exist_ok=True)
    PRIVATE.mkdir(exist_ok=True, mode=0o700)
    return dict(protocol=roster.sha(PROTOCOL), locked_method=pins["METHOD_LOCK.json"],
                previous_evaluator=pins["paper_eval.py"], code=roster.sha(script), utilities=roster.sha(__file__))


def save_payload(path, rows):
    temporary = path.with_suffix(".tmp")
    with prior.compressed(temporary) as stream:
        for row in rows:
            prior.dump(stream, row)
    temporary.replace(path)
    return roster.sha(path)


def read_payload(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def group_summary(prefix, rows, arms, pairs):
    scorer.ARMS, scorer.PAIRS, scorer.OUT = arms, pairs, OUT
    platforms = sorted({r["target"] for r in rows})
    groups = {p: [r for r in rows if r["target"] == p] for p in platforms}
    groups["pooled"] = rows
    if set(platforms) == set(prior.PLATFORMS):
        groups["macro"] = rows
    metrics, comparisons, rec = [], [], []
    for p, group in groups.items():
        metrics.extend(dict(platform=p, **r) for r in scorer.table_row_metrics(group, p))
        comparisons.extend(dict(platform=p, **r) for r in scorer.table_row_pairs(group, p))
        if p not in ("macro", "pooled"):
            by_recording = defaultdict(list)
            for r in group:
                by_recording[r["recording"]].append(r)
            for key, block in sorted(by_recording.items()):
                rec.extend(dict(platform=p, recording=key, **r) for r in scorer.table_row_pairs(block, p))
    intervals = scorer.bootstraps(rows) if "macro" in groups else scene_intervals(rows, arms, pairs)
    files = {}
    for suffix, data in (("metrics.csv", metrics), ("pairs.csv", comparisons), ("recordings.csv", rec),
                         ("rows.csv.gz", rows)):
        filename = f"{prefix}_{suffix}"
        files[filename] = scorer.write_rows(filename, data, compress=suffix.endswith(".gz"))
    return files, intervals


def scene_intervals(rows, arms, pairs):
    values = defaultdict(list)
    for r in rows:
        values[r["recording"]].append([r[f"iou_{a}"] for a in arms])
    blocks = [np.asarray(values[k], dtype=float) for k in sorted(values)]
    count = np.array([len(b) for b in blocks])
    totals = np.array([np.stack((b > .25, b > .50, b), axis=-1).sum(0) for b in blocks])
    idx = np.random.default_rng(20261123).integers(len(blocks), size=(4000, len(blocks)))
    sample = totals[idx].sum(1) / count[idx].sum(1)[:, None, None]
    result = {}
    for a, b in pairs:
        compared = 100 * (sample[:, arms.index(a)] - sample[:, arms.index(b)])
        result[f"pooled/{a}-{b}"] = dict(zip(("acc25", "acc50", "miou"), np.quantile(compared, [.025, .975], axis=0).T.tolist()))
    return result


def independent_tally(prefix, hashes):
    # Read only public-format aggregate tables and anonymous scalar ledgers.
    assert all(roster.sha(OUT / name) == sha for name, sha in hashes.items())
    with gzip.open(OUT / f"{prefix}_rows.csv.gz", "rt") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({r["uid"] for r in rows})
    counts = 0
    with (OUT / f"{prefix}_pairs.csv").open() as stream:
        for comp in csv.DictReader(stream):
            target = comp["platform"]
            groups = defaultdict(list)
            a, b = comp["arm"], comp["reference"]
            for r in rows:
                if target in ("macro", "pooled") or r["target"] == target:
                    key = r["target"] if target == "macro" else target
                    groups[key].append((float(r[f"iou_{a}"]), float(r[f"iou_{b}"])))
            assert sum(map(len, groups.values())) == int(comp["n"])
            for threshold, suffix in ((.25, "25"), (.50, "50")):
                rescues = harms = 0
                mean = []
                for g in groups.values():
                    n_up = sum(y <= threshold < x for x, y in g)
                    n_down = sum(x <= threshold < y for x, y in g)
                    mean.append(100 * (n_up - n_down) / len(g))
                    rescues += n_up
                    harms += n_down
                assert (rescues, harms) == (int(comp[f"rescue{suffix}"]), int(comp[f"harm{suffix}"]))
                assert abs(float(comp[f"delta_acc{suffix}_pp"]) - sum(mean)/len(mean)) < 1.e-10
            counts += 1
    return dict(rows=len(rows), pair_rows=counts)
