"""Read annotations only after both predictions have sealed."""
from collections import Counter
import csv
import gzip

import numpy as np

from inputs import OUT, PLATFORMS, PRIVATE, PROTOCOL, QWEN, V3, case_id, digest, load_panel, read, write
from geometry import camera, cell_points, intersections, raw_returns, rigid_xyz, to_sensor
from score import aggregate, canonical_box, delta, recorder_ci, rotation, score_boxes, truth_for_slot


ARMS = ("A_median", "C_expanded", "F_expanded", "C_fit_tree", "F_fit_tree")
COMPARISONS = (("C_expanded", "A_median"), ("F_expanded", "A_median"), ("C_expanded", "F_expanded"),
               ("C_fit_tree", "A_median"), ("F_fit_tree", "A_median"), ("C_fit_tree", "F_fit_tree"),
               ("C_fit_tree", "C_expanded"))


def table(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def save(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_reference():
    path = V3 / "results/ordinary_rows.csv.gz"
    assert digest(path) == read(V3 / "results/verification.json")["ordinary_rows_sha256"]
    with gzip.open(path, "rt", newline="") as f:
        return {(r["source"], r["case"]): r for r in csv.DictReader(f)}


def membership(cam, transform, box):
    pts = rigid_xyz(cam, np.linalg.inv(transform))
    return (np.abs((pts-box[:3])@rotation(box[6])) <= box[3:6]/2. + .05).all(1)


def depth_diagnostics(report, arm, cam_box, sensor_box, gt, gt_camera, fit_inside, ids):
    report[arm + "_fit_n"] = len(ids)
    report[arm + "_fit_purity"] = float(fit_inside[ids].mean()) if len(ids) else None
    report[arm + "_abs_depth_m"] = abs(float(cam_box[2]-gt_camera[2]))
    report[arm + "_center_m"] = float(np.linalg.norm(sensor_box[:3]-gt[:3]))


def fit_surface(boxes, held_cam, held_object):
    geometry = {arm: intersections(boxes[arm], held_cam) for arm in ("A_median", "C_fit_tree", "F_fit_tree")}
    for category, group in (("all_window_rays", np.ones(len(held_cam), bool)), ("target_rays", held_object)):
        for arm, (hits, errors) in geometry.items():
            both = group & hits & geometry["A_median"][0]
            active = group & hits
            yield dict(arm=arm, group=category, n=int(group.sum()), hits=int(active.sum()), both_hits=int(both.sum()),
                       error_sum=float(errors[active].sum()), error_sum_on_common=float(errors[both].sum()),
                       A_error_sum_on_common=float(geometry["A_median"][1][both].sum()))


def score_all():
    region_seal, v3_seal = (read(OUT / (n + ".json")) for n in ("region_seal", "v3_seal"))
    assert region_seal["protocol"] == v3_seal["stamp"]["protocol"] == digest(PROTOCOL)
    path = PRIVATE / "region_predictions.json"
    assert digest(path) == region_seal["predictions"] and not region_seal["GT_read"]
    c = read(path)
    replay = {}
    for source, sha in v3_seal["files"].items():
        file = PRIVATE / ("v3_" + source + ".json")
        assert digest(file) == sha and read(file)["stamp"] == v3_seal["stamp"]
        replay[source] = read(file)["rows"]
    previous = {r["uid"]: r for r in table(QWEN / "results/all_rows.csv")}
    v3 = read_reference()
    assert len(previous) == 96 and len(v3) == 18620

    rows, directions, surfaces = [], [], []
    maximum = dict(two_scorer_gap=0., original_A_iou_gap=0., registered_v3_iou_gap=0.)
    for frame in load_panel():
        uid, target, recording = (frame[k] for k in ("uid", "target", "recording"))
        prior = previous[uid]
        row = dict(uid=uid, target=target, recording=recording, qwen_window_identity=prior["id2d"],
                   qwen_valid_3d=int(c[uid]["baseline"] is not None), **{a: 0. for a in ARMS})
        score_input = {}
        details = c[uid]
        item = read(frame["calibration"])["ground_info"]
        gt = canonical_box(truth_for_slot(frame, item))
        if details["baseline"] is not None:
            spec, size, mat = camera(frame)
            gt_cam = rigid_xyz(gt[:3], mat)
            uv, cam = raw_returns(frame, spec, size)
            if details["variants"]:
                # Use the immutable 2D prediction decoded before ground truth.
                pixel = read(QWEN / "private/replies" / (uid + "_2d.json"))["value"]
                rect = np.asarray(pixel)*np.tile(size, 2)/1000.
                fit, held = cell_points(uv, cam, rect)
            else:
                fit, held = cam[:0], cam[:0]
            target_fit, target_held = membership(fit, mat, gt), membership(held, mat, gt)
            baseline = np.array(details["baseline"])
            boxes_cam = {arm: baseline for arm in ARMS}
            subset = {arm: np.arange(len(fit)) for arm in ARMS}
            in_region = dict(expanded=("C_expanded", "F_expanded"), fit_tree=("C_fit_tree", "F_fit_tree"))
            oracle_inputs = {}
            for variant, (arm, control) in in_region.items():
                info = details["variants"].get(variant, dict(status="missing_qwen_2d", regions=0, candidates=[]))
                chosen = info.get("chosen")
                pre = f"{variant}_"
                cand = info["candidates"]
                row.update({pre+"status": info["status"], pre+"region_n": info["regions"], pre+"candidate_n": len(cand)})
                if chosen is None:
                    continue
                selected = next(b for b in cand if b["node"] == chosen)
                boxes_cam.update({arm: np.array(info["camera_box"]), control: np.array(info["mismatch_box"])})
                subset.update({arm: np.asarray(info["roi_ids"], int), control: np.asarray(info["mismatch_ids"], int)})
                row.update({pre+"nontrivial_mismatch": int(info["nontrivial_mismatch"]),
                            pre+"scale_one_based": selected["scale"]+1,
                            pre+"visual_score": selected["visual_weight"]})
                assert len(subset[arm]) == len(subset[control]) >= 5
                assert max(np.max(subset[arm]), np.max(subset[control])) < len(fit)
                assert np.array_equal(boxes_cam[arm][3:], baseline[3:]) and np.array_equal(boxes_cam[control][3:], baseline[3:])
                coverage = []
                for b in cand:
                    assert len(b["central_ids"]) >= 5
                    coverage.append(float(target_fit[b["central_ids"]].mean()))
                    candidate = baseline.copy()
                    candidate[:3] *= b["depth"]/baseline[2]
                    oracle_inputs[(variant, b["node"])] = canonical_box(to_sensor(candidate, mat)[0])
                row.update({pre+"candidate_max_fit_purity": max(coverage),
                            pre+"high_purity_candidate": int(max(coverage) >= .5)})
            score_input.update({a: canonical_box(to_sensor(b, mat)[0]) for a, b in boxes_cam.items()})
            score_input.update(oracle_inputs)
            for arm in ARMS:
                depth_diagnostics(row, arm, boxes_cam[arm], score_input[arm], gt, gt_cam, target_fit, subset[arm])
            for rec in fit_surface(boxes_cam, held, target_held):
                surfaces.append(dict(uid=uid, target=target, recording=recording, **rec))
        else:
            row.update(expanded_status="invalid_qwen_3d", fit_tree_status="invalid_qwen_3d")
        for source in replay:
            if source != target:
                score_input[source] = np.asarray(replay[source][uid]["box"])
        scores, gap = score_boxes(np.stack(list(score_input.values())), gt)
        maximum["two_scorer_gap"] = max(maximum["two_scorer_gap"], gap)
        audit = {key: float(value) for key, value in zip(score_input, scores)}
        row.update({a: audit[a] for a in ARMS if a in audit})
        maximum["original_A_iou_gap"] = max(maximum["original_A_iou_gap"], abs(row["A_median"]-float(prior["ray_depth"])))
        for variant in ("expanded", "fit_tree"):
            options = [v for key, v in audit.items() if isinstance(key, tuple) and key[0] == variant]
            if options:
                row[variant+"_candidate_oracle_iou"] = max(options)
        for source in replay:
            if source == target:
                continue
            got, expected = replay[source][uid], v3[source, case_id(frame)]
            assert got["entered"] == (expected["active"] == "1")
            assert got["selected"] == (expected["active"] == "1" and expected["primary_fallback"] == "0")
            archived = float(expected["separate_direct"])
            maximum["registered_v3_iou_gap"] = max(maximum["registered_v3_iou_gap"], abs(audit[source]-archived))
            directions.append(dict(uid=uid, target=target, source=source, recording=recording, entered=int(got["entered"]),
                selected=int(got["selected"]), A_median=row["A_median"], B_v3=audit[source], B_archived_v3=archived,
                C_strict=row["C_expanded"] if got["entered"] else row["A_median"],
                F_strict=row["F_expanded"] if got["entered"] else row["A_median"]))
        rows.append(row)
    assert len(rows) == 96 and len(directions) == 192, maximum
    assert max(maximum["two_scorer_gap"], maximum["original_A_iou_gap"]) < 1e-8, maximum
    return rows, directions, surfaces, maximum


def grouped(rows):
    for platform in (*PLATFORMS, "macro"):
        yield platform, [r for r in rows if platform == "macro" or platform == r["target"]]


def float_summary(group, key, statistic):
    vals = [r[key] for r in group if r.get(key) is not None]
    return float(statistic(vals)) if vals else None


def run():
    rows, pairs, surface, audit = score_all()
    metrics, comparisons, geometry = [], [], []
    scope = {
        "all": rows,
        "C_expanded_used": [r for r in rows if r["expanded_status"] == "used_region"],
        "C_fit_tree_used": [r for r in rows if r["fit_tree_status"] == "used_region"],
        "2d_target": [r for r in rows if r["qwen_window_identity"] == "target"],
        "low_A_purity": [r for r in rows if r.get("A_median_fit_purity") is not None
                         and r["A_median_fit_n"] >= 5 and r["A_median_fit_purity"] < .5],
    }
    for name, available in scope.items():
        for target, group in grouped(available):
            if not group:
                continue
            for arm in ARMS:
                metrics.append(dict(scope=name, target=target, arm=arm, n=len(group),
                                    **aggregate([r[arm] for r in group])))
                geometry.append(dict(scope=name, target=target, arm=arm, n=len(group),
                    valid_box=sum(r.get(arm+"_center_m") is not None for r in group),
                    with_fit=sum(bool(r.get(arm+"_fit_n")) for r in group),
                    median_abs_depth_m=float_summary(group, arm+"_abs_depth_m", np.median),
                    median_center_m=float_summary(group, arm+"_center_m", np.median),
                    mean_inside_gt=float_summary(group, arm+"_fit_purity", np.mean),
                    median_fit_count=float_summary(group, arm+"_fit_n", np.median)))
            for arm, base in COMPARISONS:
                comparisons.append(dict(scope=name, target=target, arm=arm, reference=base, n=len(group),
                                        **delta(group, arm, base)))

    direction_table = []
    for source in (*PLATFORMS, "all"):
        for target in (*PLATFORMS, "all"):
            block = [r for r in pairs if source in ("all", r["source"]) and target in ("all", r["target"])]
            if block:
                for arm in ("A_median", "B_v3", "B_archived_v3", "C_strict", "F_strict"):
                    direction_table.append(dict(source=source, target=target, arm=arm, n=len(block),
                        entered=sum(r["entered"] for r in block), selected=sum(r["selected"] for r in block),
                        **aggregate([r[arm] for r in block]), **delta(block, arm, "A_median")))
    intervals = {f"{arm}_vs_{ref}": recorder_ci(rows, arm, ref) for arm,ref in COMPARISONS}
    focal = [(arm, delta(group, "C_expanded", "A_median")) for arm,group in grouped(rows) if arm in ("drone", "quad")]
    c_a, c_f = delta(rows, "C_expanded", "A_median"), delta(rows, "C_expanded", "F_expanded")
    gate = dict(positive_Acc50=c_a["delta50_pp"] > 0., positive_mIoU=c_a["delta_miou_pp"] > 0.,
                wins_same_count_mismatch=c_f["delta_miou_pp"] > 0.,
                drone_or_quad_task_improved=any(x["delta50_pp"] > 0. or x["delta_miou_pp"] > 0. for _,x in focal))
    gate["advance_to_D_E"] = all(gate.values())
    gates = {t: dict(n=len(g), expanded=dict(Counter(r["expanded_status"] for r in g)),
                     fit_tree=dict(Counter(r["fit_tree_status"] for r in g)),
                     fit_randomized=sum(r.get("fit_tree_nontrivial_mismatch", 0) for r in g),
                     full_randomized=sum(r.get("expanded_nontrivial_mismatch", 0) for r in g))
             for t,g in grouped(rows)}
    for name, result in (("frames",rows), ("source_pairs",pairs), ("surface",surface), ("metrics",metrics),
                         ("paired",comparisons), ("geometry",geometry), ("directions",direction_table)):
        save(OUT / (name+".csv"), result)
    write(OUT / "recording_intervals.json", intervals)
    write(OUT / "decision.json", gate)
    write(OUT / "verification.json", dict(**audit, cases=len(rows), directions=len(pairs), fit_coverage=gates,
        source_region_entries=sum(r["entered"] for r in pairs), source_region_outputs=sum(r["selected"] for r in pairs),
        all_cases_scored=True, final_access=False, qwen_generations=0, predictions_sealed_before_gt_scoring=True,
        prediction_seals={p: digest(OUT / p) for p in ("v3_seal.json", "region_seal.json")}))


if __name__ == "__main__":
    run()
