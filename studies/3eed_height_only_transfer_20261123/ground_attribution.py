"""Measure ground-contact error and the oracle headroom at exactly fixed height."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import OUT, PRIVATE, group_summary, independent_tally, lock, prior, read_payload, roster, save_payload, scorer
from extent_clip import TOL, protected_points
from live import first_depth, incumbent


ARMS = ("roof", "closest_feasible_symmetric", "oracle_feasible_center")
PAIRS = (("closest_feasible_symmetric", "roof"), ("oracle_feasible_center", "roof"))


def support(job):
    frame, reply, row = job
    assert frame["uid"] == row["uid"]
    k = dict(uid=row["uid"], boxes={a: row["boxes"][a] for a in ARMS[:2]}, allowance=0., changed=bool(row["trim"]))
    if row["trim"] == 0.:
        return k
    old, ctx = incumbent(frame, reply, collect_audit=False)
    assert old["box"] == row["boxes"]["frozen"] and ctx is not None
    box, _, _, fit, selection, _, transform = ctx
    depth = first_depth(box, selection, transform)
    _, measured, hits = protected_points(box, fit, selection, depth, transform)
    clearance = max(0., float(np.concatenate((measured, hits))[:, 2].min() + box[5]/2 - TOL))
    k["allowance"] = min(row["trim"], clearance)
    pr = k["boxes"]["roof"]
    candidate = np.asarray(pr).copy()
    candidate[2] += k["allowance"]
    n, max_err = prior.reference.invariance(depth, first_depth(candidate, selection, transform))
    assert n == 0 and max_err <= TOL
    return k


def seal(cohort, workers):
    prefix = f"{cohort}_anchor"
    parent = OUT / f"{prefix}_prediction_seal.json"
    data = PRIVATE / f"{prefix}_predictions.jsonl.gz"
    existing = roster.read(parent)
    assert existing["payload"] == roster.sha(data) and not existing["target_gt_opened"]
    jobs = prior.fixed_validation_jobs() if cohort == "validation" else prior.extension_jobs()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        collection = pool.map(support, ((job[0], job[1], row) for job, row in zip(jobs, read_payload(data), strict=True)), chunksize=24)
        result = save_payload(PRIVATE / f"{cohort}_oracle_allowed.jsonl.gz", collection)
    roster.write(OUT / f"{cohort}_oracle_allowed_seal.json", dict(inputs=roster.sha(parent), payload=result,
        pins=lock(Path(__file__)), n=existing["expected"], target_gt_opened=False, FINAL_access=False))


def score(job):
    frame, candidate = job
    assert frame["uid"] == candidate["uid"]
    path = Path(frame["calibration"])
    assert path.name == "meta_info.json" and path.parents[2] == prior.reference.DATA / frame["target"]
    items = roster.read(path)["ground_info"]
    truth = scorer.canonical_box(scorer.truth_for_slot(frame, items))
    first = candidate["boxes"]["roof"]
    mid = candidate["boxes"]["closest_feasible_symmetric"]
    output = {k: frame[k] for k in ("uid", "target", "recording")}
    output.update(edited=candidate["changed"], feasible_up_m=candidate["allowance"], lower_face_error_roof_m=None,
                  lower_face_error_mid_m=None, lower_face_error_oracle_m=None, cross_checked_grid=False)
    best = first
    grid = []
    if candidate["changed"]:
        shifted = float(np.clip(truth[2]-first[2], 0., candidate["allowance"]))
        best = list(first)
        best[2] += shifted
        tb = truth[2]-truth[5]/2
        for mode, estimate in zip(("roof", "mid", "oracle"), (first, mid, best), strict=True):
            output[f"lower_face_error_{mode}_m"] = estimate[2]-estimate[5]/2-tb
        if int(frame["uid"][:8], 16) % 37 == 0:
            grid = [[*first[:2], first[2]+float(s), *first[3:]] for s in np.linspace(0, candidate["allowance"], 21)]
            output["cross_checked_grid"] = True
    result, error = prior.safe_iou([first, mid, best, *grid], truth)
    assert result[2]+1.e-11 >= max(result[:2])
    if len(grid):
        assert result[2]+1.e-11 >= max(result[3:])
    output.update({f"iou_{a}": float(x) for a, x in zip(ARMS, result[:3], strict=True)})
    return output, error


def evaluate(cohort, workers):
    path = OUT / f"{cohort}_oracle_allowed_seal.json"
    lock_ = roster.read(path)
    candidate = PRIVATE / f"{cohort}_oracle_allowed.jsonl.gz"
    assert lock_["pins"] == lock(Path(__file__)) and not lock_["target_gt_opened"]
    assert lock_["payload"] == roster.sha(candidate)
    rows = roster.panel()[0] if cohort == "validation" else prior.transfer.read_panel()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        observed = list(pool.map(score, zip(rows, read_payload(candidate), strict=True), chunksize=24))
    rows, errs = zip(*observed, strict=True)
    assert len(rows) == lock_["n"] and max(errs) < 1.e-7
    summary, ci = group_summary(f"{cohort}_fixed_height_oracle", rows, ARMS, PAIRS)
    audit = independent_tally(f"{cohort}_fixed_height_oracle", summary)
    edited = [r for r in rows if r["edited"]]
    summary_values = {a: float(np.mean([abs(r[f"lower_face_error_{a}_m"]) for r in edited]))
                      for a in ("roof", "mid", "oracle")}
    roster.write(OUT / f"{cohort}_fixed_height_oracle_evaluation.json", dict(seal=roster.sha(path), files=summary,
        independent=audit, grid_checks=sum(r["cross_checked_grid"] for r in rows), mae_bottom_m=summary_values,
        intervals=ci, oracle_used_only_for_diagnosis=True, FINAL_access=False))
    print(cohort, len(rows), "fixed-height placement oracle verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("seal", "evaluate"))
    parser.add_argument("cohort", choices=("validation", "extension"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    (seal if args.phase == "seal" else evaluate)(args.cohort, args.workers)
