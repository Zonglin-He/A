"""Keep the incumbent and final dimensions identical while varying vertical placement."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

from common import OUT, PRIVATE, group_summary, independent_tally, lock, prior, read_payload, roster, save_payload, scorer
from extent_clip import TOL, protected_points
from live import first_depth, incumbent


ARMS = ("frozen", "roof", "symmetric", "cut_bottom", "closest_feasible_symmetric")
PAIRS = tuple((a, "frozen") for a in ARMS[1:]) + tuple(("roof", a) for a in ARMS[2:])


def create(job):
    frame, reply, sealed = job
    assert frame["uid"] == sealed["uid"]
    old, roof = [sealed["boxes"][a] for a in ("baseline", "height_only")]
    row = dict(uid=frame["uid"], boxes=dict.fromkeys(ARMS, roof), stats={}, trim=0., active=sealed["active"],
               selection=0, audit=0, protected=0, symmetric_shift_projected_m=0.)
    row["boxes"]["frozen"] = old
    if old == roof:
        return row
    checked, context = incumbent(frame, reply, collect_audit=True)
    assert checked["box"] == old and context is not None
    box, _dims, _rot, fit, selection, audit, transform = context
    p = np.asarray(roof, dtype=float)
    trim = float(box[5] - p[5])
    assert trim > 0 and np.array_equal(box[[0, 1, 3, 4, 6]], p[[0, 1, 3, 4, 6]])
    assert abs(box[2] - box[5]/2 - p[2] + p[5]/2) < TOL
    d_sel = first_depth(box, selection, transform)
    d_aud = first_depth(box, audit, transform)
    _, measured, entries = protected_points(box, fit, selection, d_sel, transform)
    floor_clearance = max(0., (np.concatenate((measured, entries))[:, 2].min() + box[5]/2) - TOL)
    shift = dict(roof=0., symmetric=trim/2, cut_bottom=trim,
                 closest_feasible_symmetric=min(trim/2, floor_clearance))
    row.update(trim=trim, selection=len(selection), audit=len(audit), protected=len(measured),
               symmetric_shift_projected_m=shift["closest_feasible_symmetric"])
    for arm, dy in shift.items():
        alt = p.copy()
        alt[2] += dy
        assert np.array_equal(alt[3:7], p[3:7])
        sel, eps = prior.reference.invariance(d_sel, first_depth(alt, selection, transform))
        aud, _ = prior.reference.invariance(d_aud, first_depth(alt, audit, transform))
        lower = alt[2] - box[2] - alt[5]/2
        upper = alt[2] - box[2] + alt[5]/2
        lost = int(((measured[:, 2] < lower - 2*TOL) | (measured[:, 2] > upper + 2*TOL)).sum())
        if arm in ("roof", "closest_feasible_symmetric"):
            assert sel == lost == 0 and eps <= TOL
        row["boxes"][arm] = alt.tolist()
        row["stats"][arm] = dict(selection_changed=sel, audit_changed=aud, measured_lost=lost, shift_m=dy,
                                 max_common_depth_delta_m=eps)
    return row


def infer(cohort, workers):
    ns = lock(Path(__file__))
    path, seal_path = prior.files(cohort)
    seal = roster.read(seal_path)
    assert seal["files"] == prior.fingerprints() and roster.sha(path) == seal["payload"]
    assert not seal["target_gt_opened"]
    frames = prior.fixed_validation_jobs() if cohort == "validation" else prior.extension_jobs()
    output = PRIVATE / f"{cohort}_anchor_predictions.jsonl.gz"
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = ((v[0], v[1], s) for v, s in zip(frames, read_payload(path), strict=True))
        def progress():
            for i, row in enumerate(executor.map(create, jobs, chunksize=24), 1):
                if i % 1000 == 0:
                    print(cohort, i, "anchor rows generated", flush=True)
                yield row
        checksum = save_payload(output, progress())
    roster.write(OUT / f"{cohort}_anchor_prediction_seal.json", dict(
        cohort=cohort, source=roster.sha(seal_path), expected=sum(seal["counts"].values()), pins=ns,
        payload=checksum, target_gt_opened=False, FINAL_access=False))


def score(job):
    frame, sealed = job
    assert frame["uid"] == sealed["uid"]
    path = Path(frame["calibration"])
    assert path.name == "meta_info.json" and path.parents[2] == prior.reference.DATA / frame["target"]
    items = roster.read(path)["ground_info"]
    truth = scorer.canonical_box(scorer.truth_for_slot(frame, items))
    vals, delta = prior.safe_iou([sealed["boxes"][a] for a in ARMS], truth)
    out = {k: frame[k] for k in ("uid", "target", "recording")}
    out.update(trim_m=sealed["trim"], active=sealed["active"], selection=sealed["selection"], audit=sealed["audit"],
               protected=sealed["protected"], feasible_shift_m=sealed["symmetric_shift_projected_m"],
               category=items[frame["slot"]]["class"].lower().strip())
    out.update({f"iou_{a}": float(value) for a, value in zip(ARMS, vals, strict=True)})
    for arm in ARMS[1:]:
        for key in ("selection_changed", "audit_changed", "measured_lost", "shift_m"):
            out[f"{arm}_{key}"] = sealed["stats"].get(arm, {}).get(key, 0)
    return out, delta


def evaluate(cohort, workers):
    begin = perf_counter()
    prefix = f"{cohort}_anchor"
    predictions = PRIVATE / f"{prefix}_predictions.jsonl.gz"
    seal_path = OUT / f"{prefix}_prediction_seal.json"
    seal = roster.read(seal_path)
    assert not seal["target_gt_opened"] and seal["payload"] == roster.sha(predictions)
    assert seal["pins"] == lock(Path(__file__))
    frames = roster.panel()[0] if cohort == "validation" else prior.transfer.read_panel()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        found = list(executor.map(score, zip(frames, read_payload(predictions), strict=True), chunksize=32))
    rows, errors = zip(*found, strict=True)
    assert len(rows) == seal["expected"] and max(errors) < 1.e-7
    from audit_paper import ledger
    old = {r["uid"]: r for r in ledger(cohort)}
    max_entire = 0.
    for row in rows:
        b = old[row["uid"]]
        for a, r in (("frozen", "baseline"), ("roof", "height_only")):
            x, y = row[f"iou_{a}"], float(b[f"iou_{r}"])
            max_entire = max(max_entire, abs(x - y))
            assert all((x > t) == (y > t) for t in (.25, .50))
    assert max_entire < 1.e-10
    hashes, intervals = group_summary(prefix, rows, ARMS, PAIRS)
    checks = independent_tally(prefix, hashes)
    ray = []
    for p in (*prior.PLATFORMS, "pooled"):
        group = [r for r in rows if r["trim_m"] > 0 and (p == "pooled" or r["target"] == p)]
        for a in ARMS[1:]:
            ray.append(dict(platform=p, arm=a, changed_height=len(group),
                changed_center=sum(r[f"{a}_shift_m"] > TOL for r in group),
                selection_rays=sum(r["selection"] for r in group), audit_rays=sum(r["audit"] for r in group),
                protected_points=sum(r["protected"] for r in group),
                **{k: sum(r[f"{a}_{k}"] for r in group) for k in ("selection_changed", "audit_changed", "measured_lost")}))
    name = f"{prefix}_geometry.csv"
    hashes[name] = scorer.write_rows(name, ray)
    roster.write(OUT / f"{prefix}_evaluation.json", dict(predictions=roster.sha(seal_path), independent=checks,
        output_sha256=hashes, recording_bootstrap_95pp=intervals, max_frozen_iou_error=max_entire,
        max_two_scorer_error=max(errors), seconds=perf_counter()-begin, FINAL_access=False))
    print(cohort, len(rows), "matched-height controls scored and audited", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("infer", "evaluate"))
    parser.add_argument("cohort", choices=("validation", "extension"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    (infer if args.phase == "infer" else evaluate)(args.cohort, args.workers)
