"""Seal fixed readouts, score the two ordinary cohorts, and audit roof protection."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import csv
import gzip
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
MAIN = HERE.parent / "3eed_qwen_extent_height_only_20261122"
sys.path.insert(0, str(MAIN))
import run_study as locked
from live import first_depth, geometry, incumbent
from extent_clip import TOL, clip_extent, protected_points
import transfer
import geometry_full


OUT, PRIVATE, PROTOCOL = transfer.OUT, transfer.PRIVATE, transfer.PROTOCOL
roster, scorer, reference = locked.roster, locked.scorer, locked.reference
PLATFORMS = scorer.PLATFORMS
OLD_ARMS = ("A", "H", "L7", "C_pca", "C_ray")
CORE = ("native", *OLD_ARMS, "baseline", "ray_equivalent", "height_only")
ABLATIONS = ("prior_only", "points_only", "rays_only", "no_prior", "swapped_prior", "soft_search")
RAY_COUNTS = ("rays25pct", "rays50pct")
ROOFS = ("height_only", *ABLATIONS, *RAY_COUNTS)
ARMS = (*CORE, *ABLATIONS, *RAY_COUNTS, "oracle")
PAIRS = (
    *((a, "baseline") for a in ("ray_equivalent", *ROOFS, "oracle")),
    *(("height_only", a) for a in (*CORE[:-3], "ray_equivalent", *ABLATIONS, *RAY_COUNTS)),
    ("oracle", "height_only"), ("H", "A"), ("H", "L7"), ("C_ray", "C_pca"),
)
CHECKS = ("updated", "selection_changed", "audit_changed", "lost_measured", "depth_delta")


def files(cohort):
    return PRIVATE / f"{cohort}_boxes.jsonl.gz", OUT / f"{cohort}_prediction_seal.json"


def fingerprints():
    method = roster.read(MAIN / "METHOD_LOCK.json")
    assert method["entry"]["sha256"] == roster.sha(MAIN / "live.py")
    assert method["config"]["clip_sha256"] == roster.sha(MAIN / "extent_clip.py")
    names = (PROTOCOL, HERE / "paper_eval.py", HERE / "transfer.py", MAIN / "METHOD_LOCK.json")
    return {p.name: roster.sha(p) for p in names}


@contextmanager
def compressed(path):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            yield gz


def dump(stream, row):
    stream.write((json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode())


def as_roof(box, height):
    if abs(height - box[5]) <= TOL:
        return box
    candidate = box.copy()
    candidate[2] += (height - box[5]) / 2
    candidate[5] = height
    return candidate


def lower_height(half_height, points):
    return max(TOL, half_height + max(points[:, 2], default=-np.inf) + TOL)


def ray_report(candidate, box, selection, depths, audit, audit_depths, matrix, lost):
    if np.array_equal(candidate, box):
        return [0, 0, 0, 0, 0.]
    changed, delta = reference.invariance(depths, first_depth(candidate, selection, matrix))
    audit_changed, _ = reference.invariance(audit_depths, first_depth(candidate, audit, matrix))
    return [1, changed, audit_changed, lost, delta]


def search_height(box, heights, measured, selection, matrix, support_count):
    # Adapt the published objective to the unchanged incumbent footprint and height interval.
    ray_lengths = np.linalg.norm(selection, axis=1) / selection[:, 2]
    observed = selection[:, 2]
    trials = np.unique(np.r_[np.linspace(heights["prior_only"], box[5], 17),
                            [heights[a] for a in ("points_only", "rays_only", "height_only")]])
    costs = []
    for height in trials:
        predicted = first_depth(as_roof(box, height), selection, matrix)
        hit = predicted > 0
        residual = np.mean((abs(predicted - observed) * ray_lengths)[hit]) if hit.any() else 0.
        covered = np.count_nonzero(measured[:, 2] <= height - box[5] / 2 + TOL)
        costs.append(residual + 10. * (1. - covered / support_count))
    return float(trials[np.argmin(costs)])


def seal_box(job):
    frame, replies, comparisons, expected, old = job
    uid = frame["uid"]
    original, context = incumbent(frame, replies, collect_audit=True)
    raw = replies["3d"]["value"]
    native = None
    if raw is not None:
        matrix = context[-1] if context is not None else geometry.camera(frame)[-1]
        native = geometry.to_sensor(np.asarray(raw, dtype=float), matrix)[0].tolist()
    if comparisons is None:
        comparisons = geometry_full.predict(frame, replies)["boxes"]
    baseline = original["box"]
    boxes = {"native": native, **comparisons, **dict.fromkeys(ARMS[len(OLD_ARMS) + 1:-1], baseline)}
    row = dict(uid=uid, active=context is not None, eligible=False, route=original["readout"],
               heading=original["heading"], boxes=boxes, checks={})
    if expected is not None:
        assert uid == expected["uid"] == old["uid"]
        assert original == expected["simple4"] and row["active"] == old["active"]
    if context is None:
        return row

    box, dimensions, rotation, fit, selection, audit, matrix = context
    depths = first_depth(box, selection, matrix)
    audit_depths = first_depth(box, audit, matrix)
    eye, measured, entries = protected_points(box, fit, selection, depths, matrix)
    h0 = float(box[5])
    half_height = h0 / 2
    row.update(selection=len(selection), audit=len(audit), points=len(measured), fit=len(fit),
               entry_count=len(entries), from_above=bool(eye[2] > half_height + TOL))
    if not len(entries) or (np.abs(eye) <= box[3:6] / 2 + TOL).all():
        if old is not None:
            assert not old["variants"]
        return row

    prior = float(dimensions[np.argmax(np.abs(rotation[2]))])
    hp = min(h0, lower_height(half_height, measured))
    hr = min(h0, lower_height(half_height, entries))
    lower = max(hp, hr)
    main = clip_extent(box, dimensions, rotation, eye, measured, entries, height_only=True)
    full = clip_extent(box, dimensions, rotation, eye, measured, entries, height_only=False)
    boxes["ray_equivalent"] = full.tolist()
    heights = dict(prior_only=min(h0, prior), points_only=min(h0, max(prior, hp)),
                   rays_only=min(h0, max(prior, hr)), height_only=float(main[5]), no_prior=lower)
    assert abs(main[5] - min(h0, max(prior, lower))) <= TOL
    if old is not None:
        for arm, candidate in (("height_only", main), ("ray_equivalent", full)):
            expected_box = old["variants"].get(arm, {}).get("box", box)
            np.testing.assert_allclose(candidate, expected_box, atol=TOL, rtol=0.)

    heights["soft_search"] = search_height(box, heights, measured, selection, matrix, len(fit) + len(selection))
    base_order = np.random.default_rng(int(uid[:16], 16)).permutation(len(selection))
    hits = depths > 0
    entry_by_ray = np.zeros(len(selection))
    entry_by_ray[hits] = entries[:, 2]
    subsets = {}
    for fraction, arm in zip((.25, .50), RAY_COUNTS, strict=True):
        chosen = base_order[:max(1, int(fraction * len(selection)))]
        subsets[arm] = chosen
        mask = hits[chosen]
        bound = half_height + entry_by_ray[chosen[mask]].max() + TOL if mask.any() else h0
        heights[arm] = min(h0, max(prior, hp, bound))

    for arm, height in heights.items():
        candidate = main if arm == "height_only" else as_roof(box, height)
        boxes[arm] = candidate.tolist()
        lost = int(np.count_nonzero(measured[:, 2] > candidate[5] - half_height + 2 * TOL))
        row["checks"][arm] = ray_report(candidate, box, selection, depths, audit, audit_depths, matrix, lost)
        if arm in subsets:
            indices = subsets[arm]
            subset_changes, _ = reference.invariance(depths[indices], first_depth(candidate, selection[indices], matrix))
            assert subset_changes == 0
    assert row["checks"]["height_only"][1] == row["checks"]["height_only"][3] == 0
    row.update(eligible=True, prior=prior, floor=lower, points_floor=hp, ray_floor=hr,
               prior_clip=heights["prior_only"], subsets={a: len(ix) for a, ix in subsets.items()},
               swap_geometry=[selection.tolist(), depths.tolist(), audit.tolist(), audit_depths.tolist(), matrix.tolist()])
    return row


def fixed_validation_jobs():
    frames, pins, checks, reply_index, official = reference.origin.inputs()
    seal = roster.read(locked.OUT / "prediction_seal.json")
    assert not seal["target_gt_opened"] and not seal["FINAL_access"]
    assert seal["payload"] == roster.sha(locked.PAYLOAD)
    with gzip.open(locked.PAYLOAD, "rt") as source:
        for f, expected, old in zip(frames, official, source, strict=True):
            archived = reference.origin.prediction(f, checks[f["uid"]], pins)
            replies = reference.origin.replies(f, archived, reply_index)
            yield f, replies, archived["boxes"], expected, json.loads(old)


def extension_jobs():
    frames = transfer.read_panel()
    seal = roster.read(transfer.QWEN_SEAL)
    assert not seal["target_gt_opened"] and not seal["FINAL_access"]
    assert seal["code"] == roster.sha(transfer.__file__) and seal["input"] == roster.sha(transfer.INPUT_SEAL)
    assert seal["replies"] == roster.sha(transfer.REPLY_INDEX) and seal["queries"] == 2 * len(frames)
    hashes = roster.read(transfer.REPLY_INDEX)
    for f in frames:
        yield f, transfer.replies(f, seal, hashes), None, None, None


def swapped_roof(row, prior):
    box = np.asarray(row["boxes"]["baseline"])
    selection, depths, audit, audit_depths, matrix = map(np.asarray, row.pop("swap_geometry"))
    candidate = as_roof(box, min(box[5], max(prior, row["floor"])))
    row["boxes"]["swapped_prior"] = candidate.tolist()
    row["checks"]["swapped_prior"] = ray_report(candidate, box, selection, depths, audit.reshape(-1, 3),
                                                audit_depths, matrix, 0)
    assert row["checks"]["swapped_prior"][1] == 0
    return row


def propose(cohort, workers):
    begin = perf_counter()
    payload, seal_path = files(cohort)
    config, pins = roster.read(PROTOCOL), fingerprints()
    assert not config["FINAL_access"] and config["main_lock_sha256"] == pins["METHOD_LOCK.json"]
    expected = {p: config["cohorts"][cohort][p] for p in PLATFORMS}
    by_platform, counts, active = defaultdict(dict), Counter(), Counter()
    PRIVATE.mkdir(exist_ok=True, mode=0o700)
    intermediate = PRIVATE / f"{cohort}_pre_swap.jsonl.gz"
    jobs = fixed_validation_jobs() if cohort == "validation" else extension_jobs()
    with compressed(intermediate) as output, ProcessPoolExecutor(max_workers=workers) as pool:
        def tracked():
            for job in jobs:
                platform = job[0]["target"]
                counts[platform] += 1
                platforms.append(platform)
                yield job

        platforms = []
        for i, row in enumerate(pool.map(seal_box, tracked(), chunksize=16)):
            if row["active"]:
                active[platforms[i]] += 1
            if row["eligible"]:
                by_platform[platforms[i]][row["uid"]] = row["prior"]
            dump(output, row)
            if (i + 1) % 500 == 0:
                print(cohort, "readouts", i + 1, "/", sum(expected.values()), flush=True)
    assert counts == expected

    swaps = {}
    for priors in by_platform.values():
        ids = sorted(priors, key=lambda uid: roster.key("paper-roof-donor", uid))
        if len(ids) > 1:
            swaps.update({uid: priors[ids[i - 1]] for i, uid in enumerate(ids)})
    temporary = payload.with_suffix(".tmp")
    with gzip.open(intermediate, "rt") as stream, compressed(temporary) as output:
        for line in stream:
            row = json.loads(line)
            if row["eligible"]:
                row = swapped_roof(row, swaps.get(row["uid"], row["prior"]))
            dump(output, row)
    temporary.replace(payload)
    source = locked.OUT / "prediction_seal.json" if cohort == "validation" else transfer.QWEN_SEAL
    roster.write(seal_path, dict(cohort=cohort, counts=dict(counts), active=dict(active),
        eligible={p: len(by_platform[p]) for p in PLATFORMS}, files=pins, qwen_source=roster.sha(source),
        payload=roster.sha(payload), workers=workers, seconds=perf_counter() - begin,
        target_gt_opened=False, FINAL_access=False))
    intermediate.unlink()
    print(cohort, "all boxes sealed", dict(counts), flush=True)


def safe_iou(boxes, truth):
    ious = np.zeros(len(boxes))
    valid = [i for i, box in enumerate(boxes) if box is not None]
    arr = np.asarray([boxes[i] for i in valid])
    if not valid:
        return ious, 0.
    assert arr.shape == (len(valid), 7) and np.isfinite(arr).all() and (arr[:, 3:6] > 0).all()
    ious[valid], error = scorer.score_boxes(arr, truth)
    return ious, error


def score_row(job):
    frame, sealed = job
    assert frame["uid"] == sealed["uid"]
    path = Path(frame["calibration"])
    assert path.name == "meta_info.json" and path.parents[2] == reference.DATA / frame["target"]
    items = roster.read(path)["ground_info"]
    truth = scorer.canonical_box(scorer.truth_for_slot(frame, items))
    category = items[frame["slot"]]["class"].strip().lower()
    assert re.fullmatch(r"[a-z0-9_ -]+", category)
    boxes = sealed["boxes"]
    base = boxes["baseline"]
    boxes["oracle"] = base
    eligible = sealed["eligible"]
    row = {k: frame[k] for k in ("uid", "target", "recording")}
    distance = np.linalg.norm(truth[:3])
    row.update(active=sealed["active"], eligible=eligible, route=sealed["route"], heading=sealed["heading"],
               category=category, distance=scorer.DISTANCE[sum(distance >= x for x in (15, 30, 60))],
               selection=sealed.get("selection", 0), audit=sealed.get("audit", 0),
               points=sealed.get("points", 0), entry_count=sealed.get("entry_count", 0),
               camera="above_original_top" if sealed.get("from_above") else "at_or_below_original_top",
               slack_m=None, trim_m=None, height_bias_old_m=None, height_bias_main_m=None,
               top_error_old_m=None, top_error_main_m=None, top_error_oracle_m=None,
               point_boundary_raises=False, ray_boundary_raises=False, ray_veto=False,
               full_ties_point_only=False, ray_is_strict_bound=False, prior_is_strict_bound=False)
    grid_checked = False
    if eligible:
        base = np.asarray(base, dtype=float)
        main = boxes["height_only"]
        bottom = base[2] - base[5] / 2
        gt_top = truth[2] + truth[5] / 2
        lo, hi = sealed["floor"], base[5]
        best_height = np.clip(gt_top - bottom, lo, hi)
        boxes["oracle"] = as_roof(base, best_height).tolist()
        row.update(slack_m=hi - lo, trim_m=hi - main[5], height_bias_old_m=hi - truth[5],
                   height_bias_main_m=main[5] - truth[5], top_error_old_m=bottom + hi - gt_top,
                   top_error_main_m=bottom + main[5] - gt_top, top_error_oracle_m=bottom + best_height - gt_top,
                   point_boundary_raises=boxes["points_only"][5] > boxes["prior_only"][5] + TOL,
                   ray_boundary_raises=main[5] > boxes["points_only"][5] + TOL,
                   ray_veto=main[5] >= hi - TOL and boxes["points_only"][5] < hi - TOL,
                   ray_is_strict_bound=sealed["ray_floor"] > max(sealed["points_floor"], sealed["prior_clip"]) + TOL,
                   full_ties_point_only=abs(main[5] - boxes["points_only"][5]) <= TOL,
                   prior_is_strict_bound=sealed["prior_clip"] > lo + TOL)
        if int(frame["uid"][:8], 16) % 37 == 0:
            grid = np.linspace(lo, hi, 33)
            scores, grid_error = safe_iou([as_roof(base, h) for h in grid] + [boxes["oracle"]], truth)
            assert scores[-1] + 1.e-10 >= scores[:-1].max() and grid_error < 1.e-7
            grid_checked = True

    values, error = safe_iou([boxes[a] for a in ARMS], truth)
    row.update({"iou_" + k: float(x) for k, x in zip(ARMS, values, strict=True)})
    assert row["iou_oracle"] >= max(row["iou_baseline"], row["iou_height_only"]) - 1.e-10
    for arm in ROOFS:
        detail = sealed["checks"].get(arm, [0, 0, 0, 0, 0.])
        row.update({f"{arm}_{k}": value for k, value in zip(CHECKS, detail, strict=True)})
    for arm in RAY_COUNTS:
        row[f"{arm}_preserved_subset_size"] = sealed.get("subsets", {}).get(arm, 0)
    row["outside_support"] = not eligible
    row["selection_count"] = ("fallback" if not eligible else "5_to_19" if row["selection"] < 20
                              else "20_to_49" if row["selection"] < 50 else "50_or_more")
    row["roof_slack"] = ("fallback" if not eligible else "locked" if row["slack_m"] <= TOL
                         else "0_to_0.10m" if row["slack_m"] < .1
                         else "0.10_to_0.30m" if row["slack_m"] < .3 else "0.30m_or_more")
    return row, error, grid_checked


def validate_old_rows(rows):
    path = locked.OUT / "rows.csv.gz"
    assert roster.read(locked.OUT / "evaluation_receipt.json")["files"]["rows.csv.gz"] == roster.sha(path)
    with gzip.open(path, "rt") as stream:
        old = {r["uid"]: r for r in csv.DictReader(stream)}
    assert set(old) == {r["uid"] for r in rows}
    pairs = ("baseline", "ray_equivalent", "height_only")
    err = 0.
    for row in rows:
        ref = old[row["uid"]]
        for arm in pairs:
            a, b = row[f"iou_{arm}"], float(ref[f"iou_{arm}"])
            err = max(err, abs(a - b))
            assert all((a > t) == (b > t) for t in (.25, .50))
    assert err <= 1.e-10
    return err


def leave_one_out(rows):
    scorer.ARMS = ARMS
    values = scorer.group_arrays(rows)[1]
    platforms = {p: np.array([r["target"] == p for r in rows]) for p in PLATFORMS}
    record = np.array([r["recording"] for r in rows])
    result = []
    for arm, ref in [("height_only", a) for a in ("baseline", "ray_equivalent", "no_prior", "A")]:
        delta = 100 * (values[:, ARMS.index(arm)] - values[:, ARMS.index(ref)])
        means = {p: delta[mask].mean(0) for p, mask in platforms.items()}
        for p, mask in platforms.items():
            for rec in sorted(set(record[mask])):
                keep = mask & (record != rec)
                assert keep.any()
                changed = delta[keep].mean(0)
                macro = np.mean([changed if t == p else means[t] for t in PLATFORMS], axis=0)
                result.append(dict(arm=arm, reference=ref, platform=p, removed_recording=rec,
                                   n_omitted=int(mask.sum() - keep.sum()),
                                   **{f"platform_delta_{k}_pp": float(x) for k, x in zip(scorer.STATS, changed)},
                                   **{f"macro_delta_{k}_pp": float(x) for k, x in zip(scorer.STATS, macro)}))
    return result


def geometry_table(rows):
    result = []
    for platform in (*PLATFORMS, "pooled"):
        group = [r for r in rows if (platform == "pooled" or r["target"] == platform) and r["eligible"]]
        for arm in ROOFS:
            changed = [r for r in group if r[f"{arm}_updated"]]
            result.append(dict(platform=platform, arm=arm, eligible=len(group), updated=len(changed),
                               affected_selection=sum(r["selection"] for r in changed),
                               affected_audit=sum(r["audit"] for r in changed),
                               optimized_ray_subset=sum(r.get(f"{arm}_preserved_subset_size", 0) for r in changed),
                               **{k: sum(r[f"{arm}_{k}"] for r in changed) for k in CHECKS[1:-1]},
                               max_common_depth_delta_m=max((r[f"{arm}_depth_delta"] for r in changed), default=0.)))
    return result


def evaluate(cohort, workers):
    begin = perf_counter()
    payload, seal_path = files(cohort)
    seal = roster.read(seal_path)
    assert not seal["target_gt_opened"] and not seal["FINAL_access"]
    assert seal["files"] == fingerprints() and seal["payload"] == roster.sha(payload)
    frames = roster.panel()[0] if cohort == "validation" else transfer.read_panel()
    assert Counter(f["target"] for f in frames) == seal["counts"]
    print(cohort, "prediction seals verified", flush=True)

    rows, error, oracle_checks = [], 0., 0
    with gzip.open(payload, "rt") as stream, ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = ((f, json.loads(line)) for f, line in zip(frames, stream, strict=True))
        for row, err, checked in pool.map(score_row, jobs, chunksize=32):
            rows.append(row)
            error = max(error, err)
            oracle_checks += checked
            if len(rows) % 2000 == 0:
                print(cohort, "scored", len(rows), flush=True)
    assert len(rows) == len(frames) and error < 1.e-7
    archived_difference = validate_old_rows(rows) if cohort == "validation" else None
    scorer.ARMS, scorer.PAIRS, scorer.OUT = ARMS, PAIRS, OUT
    groups = {p: [r for r in rows if r["target"] == p] for p in PLATFORMS}
    groups.update(macro=rows, pooled=rows)
    metrics, pairs, strata, records, record_pairs = [], [], [], [], []
    for name, group in groups.items():
        metrics.extend(dict(platform=name, **m) for m in scorer.table_row_metrics(group, name))
        pairs.extend(dict(platform=name, **p) for p in scorer.table_row_pairs(group, name))
        if name == "macro":
            continue
        for key in ("category", "distance", "selection_count", "camera", "roof_slack", "route"):
            blocks = defaultdict(list)
            for row in group:
                tag = row[key] if key != "camera" or row["eligible"] else "fallback"
                blocks[tag].append(row)
            for value, block in sorted(blocks.items()):
                rec = {"platform": name, "stratum": key, "value": value}
                strata.extend(dict(rec, **m) for m in scorer.table_row_metrics(block, name))
        if name == "pooled":
            continue
        blocks = defaultdict(list)
        for row in group:
            blocks[row["recording"]].append(row)
        for recording, block in sorted(blocks.items()):
            records.extend(dict(platform=name, recording=recording, **m) for m in scorer.table_row_metrics(block, name))
            record_pairs.extend(dict(platform=name, recording=recording, **p) for p in scorer.table_row_pairs(block, name))

    outputs = {}
    for name, data in dict(metrics=metrics, pairs=pairs, strata=strata, recordings=records,
                           recording_pairs=record_pairs, leave_one_out=leave_one_out(rows),
                           geometry=geometry_table(rows)).items():
        filename = f"{cohort}_{name}.csv"
        outputs[filename] = scorer.write_rows(filename, data)
    # Compact numerical ledgers keep one anonymous ID per target description.
    for platform in PLATFORMS:
        filename = f"{cohort}_rows_{platform}.csv.gz"
        public = [{k: round(v, 12) if isinstance(v, float) else v for k, v in r.items()} for r in groups[platform]]
        outputs[filename] = scorer.write_rows(filename, public, compress=True)
    roster.write(OUT / f"{cohort}_evaluation.json", dict(cohort=cohort, counts=seal["counts"],
        prediction_seal=roster.sha(seal_path), artifact_sha256=outputs, max_iou_audit_difference=error,
        max_previous_iou_difference=archived_difference, oracle_grid_checks=oracle_checks,
        recording_bootstrap_95pp=scorer.bootstraps(rows), FINAL_access=False, seconds=perf_counter()-begin))
    print(cohort, "metrics, paired counts and uncertainty recorded", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("propose", "evaluate"))
    parser.add_argument("cohort", choices=("validation", "extension"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    (propose if args.phase == "propose" else evaluate)(args.cohort, args.workers)
