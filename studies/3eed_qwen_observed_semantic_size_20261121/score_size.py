"""Score sealed Ordinary predictions and count transition-specific paired outcomes."""
from collections import Counter, defaultdict
import csv
from pathlib import Path
import sys

import numpy as np

from study import ARMS, HERE, OUT, PRIVATE, SIDE, TOL, digest, read, roster, verified_observations, write

PAIRS = (("GS", "H"), ("GS", "S"), ("GS", "GD"), ("S", "H"), ("GD", "H"), ("GD", "S"))
EPS = 1e-8


def checked_predictions():
    sealed = read(OUT / "prediction_seal.json")
    assert not sealed["target_labels_read"] and not sealed["FINAL_access"]
    assert sealed["input_seal"] == digest(OUT / "input_seal.json")
    assert sealed["model_seal"] == digest(OUT / "model_seal.json")
    assert sealed["normalized_seal"] == digest(OUT / "normalized_seal.json")
    assert sealed["code"] == digest(HERE / "complete_predictions.py")
    path = PRIVATE / "predictions.json"
    assert sealed["predictions"] == digest(path)
    return read(path)


def semantic_status(predictions, observation):
    if not observation["selected"]:
        return "no_region"
    s, gs = (predictions[arm]["semantic"] for arm in ("S", "GS"))
    if s is None or gs is None:
        return "invalid_semantic"
    if s[0] != gs[0]:
        return "category_changed"
    return "subtype_changed" if s[1] != gs[1] else "same"


def center_delta(a, b):
    return float(np.linalg.norm(np.subtract(a[:3], b[:3])))


def prior_lwh(prior):
    return [*sorted(prior[:2], reverse=True), prior[2]] if prior is not None else [None]*3


def relative_records(frame, observation, predictions, values):
    r = {k: frame[k] for k in ("uid", "target", "recording", "panel", "separation")}
    r["selected"] = observation["selected"]
    r["baseline_status"] = observation["status"]
    r["semantic_status"] = semantic_status(predictions, observation)
    r["reused_size"] = bool(predictions["S"]["request"] and predictions["S"]["request"] == predictions["GS"]["request"])
    for label, v in zip(("long", "short", "vertical"), observation.get("spans_lwh", [None]*3), strict=True):
        r[f"span_{label}_m"] = v
    r["support_points"] = observation.get("support_points", 0)
    r["observed_height_m"] = observation["observed_box_model"][5] if observation["selected"] else None
    old_box = predictions["H"]["box"]
    for arm, score in zip(ARMS, values, strict=True):
        entry = predictions[arm]
        category, subtype = entry.get("semantic") or ("", "")
        if arm in ("S", "GS"):
            r["category_"+arm] = category
            r["subtype_"+arm] = subtype
            r["semantic_format_"+arm] = entry["semantic_format"]
        r["status_"+arm] = entry["status"]
        for label, v in zip(("length", "width", "height"), prior_lwh(entry["size_prior"]), strict=True):
            r[f"prior_{label}_m_{arm}"] = v
        box = entry["box"]
        for label, v in zip(("axis1", "axis2", "vertical"), box[3:6] if box is not None else [None]*3, strict=True):
            r[f"box_{label}_m_{arm}"] = v
        r["center_shift_m_"+arm] = center_delta(box, old_box) if box is not None else 0.
        r["iou_"+arm] = float(score)

    s, gs = (predictions[arm] for arm in ("S", "GS"))
    s_prior, gs_prior = (prior_lwh(x["size_prior"]) for x in (s, gs))
    r["valid_prior_pair"] = s["size_prior"] is not None and gs["size_prior"] is not None
    r["prior_changed_GS_S"] = r["valid_prior_pair"] and s_prior != gs_prior
    r["completed_dimensions_changed_GS_S"] = s["box"] is not None and np.max(np.abs(np.subtract(gs["box"][3:6], s["box"][3:6]))) > TOL
    r["center_shift_m_GS_S"] = center_delta(gs["box"], s["box"]) if s["box"] is not None else 0.
    r["iou_delta_GS_S"] = r["iou_GS"] - r["iou_S"]
    for label, a, b in zip(("length", "width", "height"), gs_prior, s_prior, strict=True):
        r["prior_delta_GS_S_"+label+"_m"] = a - b if r["valid_prior_pair"] else None
    for k, axis in enumerate(("axis1", "axis2", "vertical"), 3):
        r["box_delta_GS_S_"+axis+"_m"] = gs["box"][k] - s["box"][k] if s["box"] is not None else None
    if r["reused_size"]:
        assert r["semantic_status"] == "same" and r["iou_delta_GS_S"] == 0 and r["center_shift_m_GS_S"] == 0
    return r


def groups(rows):
    scopes = {
        "all": rows,
        "validation": [r for r in rows if r["panel"] == "validation"],
        "recording_extension": [r for r in rows if r["panel"] == "recording_extension"],
        "recording_separated": [r for r in rows if r["separation"] == "new_recording"],
        "selected": [r for r in rows if r["selected"]],
        "semantic_changed": [r for r in rows if r["semantic_status"] in ("category_changed", "subtype_changed")]
    }
    for scope, group in scopes.items():
        if not group:
            continue
        yield scope, "macro", group
        for target in sorted({r["target"] for r in group}):
            yield scope, target, [r for r in group if r["target"] == target]


def comparison_metrics(rows, arm, reference, mean_platform):
    a, b = (np.array([r["iou_"+k] for r in rows]) for k in (arm, reference))
    report = {
        "iou_better": int(sum(a - b > EPS)), "iou_worse": int(sum(b - a > EPS)),
        "iou_tied": int(sum(abs(a-b) <= EPS)), "delta_miou_pp": 100*mean_platform(rows, a-b)
    }
    for threshold, key in ((.25, "25"), (.50, "50")):
        u, v = a > threshold, b > threshold
        report.update({f"rescue{key}": int(sum(u & ~v)), f"harm{key}": int(sum(v & ~u)),
                      f"delta_acc{key}_pp": 100*mean_platform(rows, u.astype(int)-v.astype(int))})
    return report


def cluster_ci(rows, arm, reference):
    """Resample recordings within platform, preserving pair membership."""
    by_platform = defaultdict(lambda: defaultdict(list))
    for r in rows:
        a, b = r["iou_"+arm], r["iou_"+reference]
        by_platform[r["target"]][r["recording"]].append([int(a>.25)-int(b>.25), int(a>.50)-int(b>.50), a-b])
    rng = np.random.default_rng(20261121)
    draws = []
    for _, recordings in sorted(by_platform.items()):
        clusters = [recordings[k] for k in sorted(recordings)]
        counts = np.array([len(c) for c in clusters])
        sums = np.array([np.sum(c, axis=0) for c in clusters])
        idx = rng.integers(len(counts), size=(4000, len(counts)))
        draws.append(sums[idx].sum(1) / counts[idx].sum(1)[:, None])
    interval = np.quantile(100*np.mean(draws, axis=0), [.025, .975], axis=0)
    return dict(zip(("delta_acc25_pp", "delta_acc50_pp", "delta_miou_pp"), interval.T.tolist(), strict=True))


def csv_table(file, rows):
    with file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    return digest(file)


def run():
    predictions = checked_predictions()
    observations = verified_observations()
    panel, baseline = roster()
    # The scoring module and Ordinary labels are opened only after the prediction seal checks.
    sys.path.insert(0, str(SIDE))
    from evaluate import PUBLIC_LAYOUT, canonical_box, mean_platform, score_boxes, truth_for_slot
    if PUBLIC_LAYOUT:
        import score_ablation
        roots = {Path(f["calibration"]).parents[3] for f in panel}
        assert len(roots) == 1
        score_ablation.DATA = roots.pop()

    parent_receipt = read(SIDE / "results/evaluation_receipt.json")
    parent_rows = SIDE / "results/rows.csv"
    assert parent_receipt["files"]["rows"] == digest(parent_rows) and not parent_receipt["FINAL_access"]
    with parent_rows.open() as stream:
        old = {r["uid"]: float(r["iou_H"]) for r in csv.DictReader(stream)}
    rows = []
    dual_error = baseline_error = 0.
    for f in panel:
        uid = f["uid"]
        prediction = predictions[uid]
        boxes = [prediction[arm]["box"] for arm in ARMS]
        assert boxes[0] == baseline[uid]["boxes"]["H"]
        if boxes[0] is None:
            assert all(b is None for b in boxes)
            values = np.zeros(len(ARMS))
        else:
            truth = canonical_box(truth_for_slot(f, read(f["calibration"])["ground_info"]))
            values, discrepancy = score_boxes(np.array(boxes), truth)
            dual_error = max(dual_error, discrepancy)
        row = relative_records(f, observations[uid], prediction, values)
        baseline_error = max(baseline_error, abs(row["iou_H"]-old[uid]))
        assert (row["iou_H"]>.25, row["iou_H"]>.50) == (old[uid]>.25, old[uid]>.50)
        rows.append(row)
    assert len(rows) == len(predictions) == len(old) == 416 and max(dual_error, baseline_error) < EPS

    metrics, paired, funnel, transitions = [], [], [], []
    intervals = {}
    for scope, target, group in groups(rows):
        tag = dict(scope=scope, target=target, n=len(group), recordings=len({r["recording"] for r in group}))
        for arm in ARMS:
            v = np.array([r["iou_"+arm] for r in group])
            metrics.append(dict(**tag, arm=arm, hits25=int(sum(v>.25)), hits50=int(sum(v>.50)),
                acc25=100*mean_platform(group, v>.25), acc50=100*mean_platform(group, v>.50),
                miou=100*mean_platform(group, v), pooled_acc25=100*float((v>.25).mean()),
                pooled_acc50=100*float((v>.50).mean()), pooled_miou=100*float(v.mean())))
        for arm, reference in PAIRS:
            paired.append(dict(**tag, arm=arm, reference=reference, **comparison_metrics(group, arm, reference, mean_platform)))
            if target == "macro" and scope in ("all", "recording_separated"):
                intervals[f"{scope}/{arm}-{reference}"] = cluster_ci(group, arm, reference)
        statuses = Counter(r["semantic_status"] for r in group)
        funnel.append(dict(**tag, selected=sum(r["selected"] for r in group), **dict(statuses),
            shared_requests=sum(r["reused_size"] for r in group),
            priors_differ=sum(r["prior_changed_GS_S"] for r in group),
            completed_dimensions_differ=sum(r["completed_dimensions_changed_GS_S"] for r in group),
            centers_differ=sum(r["center_shift_m_GS_S"]>TOL for r in group),
            positive_iou=sum(r["iou_delta_GS_S"]>EPS for r in group), negative_iou=sum(r["iou_delta_GS_S"]<-EPS for r in group)))
        if scope == "all":
            paths = defaultdict(list)
            for r in group:
                if r["semantic_status"] in ("category_changed", "subtype_changed"):
                    key = tuple(r[k] for k in ("category_S", "subtype_S", "category_GS", "subtype_GS"))
                    paths[key].append(r)
            for (s_cat, s_sub, gs_cat, gs_sub), sub in sorted(paths.items(), key=lambda pair: (-len(pair[1]), pair[0])):
                transitions.append(dict(target=target, n=len(sub), category_S=s_cat, subtype_S=s_sub,
                    category_GS=gs_cat, subtype_GS=gs_sub, **comparison_metrics(sub, "GS", "S", mean_platform)))

    tables = {"rows": rows, "metrics": metrics, "paired": paired, "semantic_funnel": funnel}
    if transitions:
        tables["semantic_transitions"] = transitions
    files = {k: csv_table(OUT/(k+".csv"), v) for k, v in tables.items()}
    write(OUT / "evaluation_receipt.json", dict(frames=len(rows), prediction_seal=digest(OUT / "prediction_seal.json"),
        code=digest(__file__), files=files, parent_rows=digest(parent_rows), paired_recording_intervals=intervals,
        oriented_iou_scorer_max_error=dual_error, H_score_max_replay_error=baseline_error, FINAL_access=False))
    print("Scored", len(rows), "frames. Maximum upright IoU discrepancy", dual_error, "baseline replay", baseline_error, flush=True)


if __name__ == "__main__":
    run()
