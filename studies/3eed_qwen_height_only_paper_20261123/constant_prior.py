"""Check a diagnosis with one height prior estimated on the other ordinary cohort."""

from concurrent.futures import ProcessPoolExecutor
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from paper_eval import OUT, PLATFORMS, as_roof, files, fingerprints, reference, roster, safe_iou, scorer, transfer
from audit_paper import COHORTS, ledger


def source(cohort):
    path, seal_path = files(cohort)
    seal = roster.read(seal_path)
    assert seal["payload"] == roster.sha(path) and seal["files"] == fingerprints()
    assert not seal["target_gt_opened"] and not seal["FINAL_access"]
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == sum(seal["counts"].values())
    return rows, roster.sha(seal_path)


def generate():
    inputs = {c: source(c) for c in COHORTS}
    selected, priors = {}, {}
    for cohort, donor in (COHORTS, COHORTS[::-1]):
        pool = [r["prior"] for r in inputs[donor][0] if r["eligible"]]
        prior = float(np.median(pool))
        priors[cohort] = dict(height_m=prior, donor_cohort=donor, donor_descriptions=len(pool))
        pred = {}
        for r in inputs[cohort][0]:
            b = r["boxes"]["baseline"]
            if r["eligible"]:
                b = as_roof(np.asarray(b), min(b[5], max(prior, r["floor"]))).tolist()
            pred[r["uid"]] = b
        selected[cohort] = pred
    path = transfer.PRIVATE / "constant_prior.json"
    roster.write(path, selected)
    seal = dict(description="Secondary control registered after Validation scoring, before extension scoring",
                priors=priors, source_seals={k: check for k, (_, check) in inputs.items()},
                code=roster.sha(__file__), predictions=roster.sha(path), target_gt_opened=False, FINAL_access=False)
    roster.write(OUT / "constant_prior_seal.json", seal)
    return path


def score(job):
    frame, box = job
    path = Path(frame["calibration"])
    assert path.name == "meta_info.json" and path.parents[2] == reference.DATA / frame["target"]
    items = roster.read(path)["ground_info"]
    truth = scorer.canonical_box(scorer.truth_for_slot(frame, items))
    value, _ = safe_iou([box], truth)
    return frame["uid"], float(value[0])


def run(workers):
    path = generate()
    seal = roster.read(OUT / "constant_prior_seal.json")
    assert seal["predictions"] == roster.sha(path) and seal["code"] == roster.sha(__file__)
    selected = roster.read(path)
    scorer.ARMS = ("constant_prior", "height_only", "baseline")
    scorer.PAIRS = (("constant_prior", "height_only"), ("constant_prior", "baseline"))
    scorer.OUT = OUT
    pairs, metrics, intervals = [], [], {}
    evidence = {}
    for c in COHORTS:
        frames = roster.panel()[0] if c == "validation" else transfer.read_panel()
        assert len(selected[c]) == len(frames) and set(selected[c]) == {f["uid"] for f in frames}
        jobs = ((f, selected[c][f["uid"]]) for f in frames)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            found = dict(pool.map(score, jobs, chunksize=32))
        rows = ledger(c)
        for row in rows:
            for a in scorer.ARMS[1:]:
                row[f"iou_{a}"] = float(row[f"iou_{a}"])
            row["iou_constant_prior"] = found[row["uid"]]
        for p in (*PLATFORMS, "macro", "pooled"):
            group = rows if p in ("macro", "pooled") else [r for r in rows if r["target"] == p]
            pairs.extend(dict(cohort=c, platform=p, **r) for r in scorer.table_row_pairs(group, p))
            metrics.extend(dict(cohort=c, platform=p, **r) for r in scorer.table_row_metrics(group, p)
                           if r["arm"] == "constant_prior")
        intervals[c] = scorer.bootstraps(rows)
        evidence[c] = roster.sha(OUT / f"{c}_evaluation.json")
    checks = {name: scorer.write_rows(name, rows) for name, rows in
              (("constant_prior_metrics.csv", metrics), ("constant_prior_pairs.csv", pairs))}
    roster.write(OUT / "constant_prior_evaluation.json", dict(seal=roster.sha(OUT / "constant_prior_seal.json"),
        source_scores=evidence, output_sha256=checks, recording_bootstrap_95pp=intervals, FINAL_access=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args().workers)
