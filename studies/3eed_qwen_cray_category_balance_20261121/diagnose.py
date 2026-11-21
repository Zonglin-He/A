"""Counterfactual component attribution on the already scored, frozen boxes."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from itertools import product

import numpy as np

from locked_roster import OUT, locked_inputs, prediction, roster, score_full


# Center (XY), horizontal size, vertical interval (Z center and height), yaw.
PARTS = ((0, 1), (3, 4), (2, 5), (6,))
NAMES = tuple("".join(str(b) for b in mask) for mask in product((0, 1), repeat=len(PARTS)))


def per_record(job):
    frame, checksum, pins = job
    original = prediction(frame, checksum, pins)
    annotations = roster.read(frame["calibration"])["ground_info"]
    truth = score_full.canonical_box(score_full.truth_for_slot(frame, annotations))
    category = annotations[frame["slot"]]["class"].lower().strip()
    row = {k: frame[k] for k in ("uid", "target", "recording")}
    row.update(category=category, changed_anchor=original["strict"], changed_yaw=original["chooses_qwen"])
    boxes = original["boxes"]
    q = boxes["H"]
    if q is None:
        row.update({f"mix_{name}": 0. for name in NAMES})
        row.update(iou_ray=0., h_span_floor=0, p_span_floor=0, h_volume_bound=0., p_volume_bound=0.,
                   h_xy_error=0., p_xy_error=0., h_z_error=0., p_z_error=0., max_scorer_error=0.)
        return row
    h, p, ray, a = map(score_full.canonical_box, (q, boxes["C_pca"], boxes["C_ray"], boxes["A"]))
    hybrids = np.tile(h, (len(NAMES), 1))
    for dest, name in zip(hybrids, NAMES, strict=True):
        for bit, part in zip(name, PARTS, strict=True):
            if bit == "1":
                dest[list(part)] = p[list(part)]
    values, error = score_full.score_boxes(np.vstack((hybrids, ray)), truth)
    row.update({f"mix_{name}": float(iou) for name, iou in zip(NAMES, values[:-1], strict=True)})
    vol = np.prod(truth[3:6])
    ratio = np.prod([h[3:6], p[3:6]], axis=1) / vol
    row.update(
        iou_ray=values[-1], h_span_floor=int(np.any(h[3:6] > a[3:6] + 1.e-6)),
        p_span_floor=int(np.any(p[3:6] > a[3:6] + 1.e-6)),
        h_volume_bound=min(ratio[0], 1/ratio[0]), p_volume_bound=min(ratio[1], 1/ratio[1]),
        h_xy_error=np.linalg.norm(h[:2]-truth[:2]), p_xy_error=np.linalg.norm(p[:2]-truth[:2]),
        h_z_error=abs(h[2]-truth[2]), p_z_error=abs(p[2]-truth[2]), max_scorer_error=error,
    )
    return row


def summarize(group, label, value):
    reference = np.array([r["mix_0000"] for r in group])
    pca = np.array([r["mix_1111"] for r in group])
    lost = (reference > .5) & (pca <= .5)
    misses = [r for r, error in zip(group, lost, strict=True) if error]
    result = []
    for arm in [*(f"mix_{name}" for name in NAMES), "iou_ray"]:
        val = np.array([r[arm] for r in group])
        result.append(dict(group=label, value=value, arm=arm, n=len(group), acc25=100*(val>.25).mean(),
                           acc50=100*(val>.50).mean(), miou=100*val.mean(),
                           rescue=int(((val>.5)&(reference<=.5)).sum()),
                           harm=int(((val<=.5)&(reference>.5)).sum()),
                           recovers_H_to_C_pca_harms=int(((val>.5)&lost).sum())))
    evidence = dict(group=label, value=value, h_to_pca_harm=len(misses),
                    h_expanded_while_p_missed=sum(r["h_span_floor"] for r in misses),
                    p_volume_bound_le50=sum(r["p_volume_bound"]<=.5 for r in misses))
    for axis in ("xy", "z"):
        for box in ("h", "p"):
            x = [r[f"{box}_{axis}_error"] for r in misses]
            evidence[f"{box}_{axis}_error_median"] = float(np.median(x)) if x else None
    return result, evidence


def run(workers):
    frames, _, seal, predictions, _ = locked_inputs()
    jobs = ((f, predictions[f["uid"]], seal["pins"]) for f in frames)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        measured = list(pool.map(per_record, jobs, chunksize=24))
    assert len(measured) == 9310 and max(r["max_scorer_error"] for r in measured) < 1.e-7
    # Independently join the previous public outcomes. Swapping equal axes is
    # harmless in geometric IoU but still needs a numerical parity check.
    lookup = {r["uid"]: r for t in score_full.PLATFORMS for r in
              __import__("verify_full").table("rows_" + t + ".csv.gz")}
    assert len(lookup) == len(measured)
    for r in measured:
        previous = lookup[r["uid"]]
        assert r["category"] == previous["category"]
        for new, old in (("mix_0000", "H"), ("mix_1111", "C_pca"), ("iou_ray", "C_ray")):
            assert abs(float(previous["iou_"+old]) - r[new]) < 1.e-7

    groups = defaultdict(list)
    for r in measured:
        for kind, key in (("category", r["category"]), ("all", "pooled")):
            groups[kind, key].append(r)
    phase, evidence = [], []
    for (kind, key), group in sorted(groups.items()):
        table, details = summarize(group, kind, key)
        phase.extend(table)
        evidence.append(details)
    for filename, rows in (("component_attribution.csv", phase), ("failure_evidence.csv", evidence)):
        with (OUT / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    roster.write(OUT / "diagnosis_receipt.json", dict(
        descriptions=len(measured), historical_seal=roster.sha(roster.OUT/"prediction_seal.json"),
        max_scorer_error=max(r["max_scorer_error"] for r in measured),
        interventions="canonical H/C_pca components; truth only for retrospective scoring",
        FINAL_access=False))
    print("Verified old predictions and scored component swaps on", len(measured), "descriptions.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    run(parser.parse_args().workers)
