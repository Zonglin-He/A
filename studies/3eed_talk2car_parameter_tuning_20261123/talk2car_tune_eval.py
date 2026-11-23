"""Seal matched geometry predictions, then score and tune across recordings."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
from itertools import product
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from responses import HERE, OUT, PRIVATE, external, identity, load, prior, read_payload, registered, roster, save_payload
import common
import cray_simple
import live


BASE_PROFILE = "original_144"
PREDICTIONS = PRIVATE / "predictions.jsonl.gz"
SEAL = OUT / "prediction_seal.json"
ADDENDUM = HERE / "SCORING_ADDENDUM.json"


def settings(cfg):
    return tuple(product(cfg["geometry"]["minimum_returns"], cfg["geometry"]["central_window_width_height_fraction"]))


def arm(readout, profile, geometry=None):
    if geometry is None:
        return f"{readout}_{profile}"
    returns, fraction = geometry
    return f"{readout}_{profile}_r{returns}_w{round(100 * fraction)}"


def pins(config):
    return dict(identity(), score_protocol=roster.sha(ADDENDUM), evaluation_code=roster.sha(__file__),
                geometry=roster.sha(live.__file__), frozen_readout=roster.sha(cray_simple.__file__),
                original_geometry_seal=roster.sha(external.PRED_SEAL),
                profiles={p: roster.sha(OUT / f"{p}_receipt.json") for p, v in config["generation"]["profiles"].items()
                          if not v.get("archived")})


def baseline_predictions():
    receipt = roster.read(external.PRED_SEAL)
    assert receipt["input"] == roster.sha(external.INPUT_SEAL) and receipt["replies"] == roster.sha(external.REPLY_SEAL)
    assert receipt["target_gt_opened"] is False and receipt["FINAL_access"] is False
    assert roster.sha(external.OUTPUT) == receipt["payload"]
    return read_payload(external.OUTPUT)


def make_geometry(job):
    frame, archived, replies, geometries = job
    assert frame["uid"] == archived["uid"] and all(r["uid"] == frame["uid"] for r in replies.values())
    boxes, diagnostics, front = {}, {}, {}
    baseline = geometries[0]
    assert baseline == (5, .50)
    original_inside = live.geometry.inside
    for name, row in replies.items():
        base = external.one_geometry((frame, row))
        if name == BASE_PROFILE:
            assert base == archived, "Locked Talk2Car output failed exact replay"
        boxes[arm("native", name)], boxes[arm("window", name)] = (base["boxes"][k] for k in ("native", "window"))
        pre = row["replies"]
        front[name] = dict(
            predicted_2d=base["predicted_2d"], status_2d=pre["2d"]["status"], status_3d=pre["3d"]["status"],
            tokens_2d=pre["2d"]["generated_tokens"], tokens_3d=pre["3d"]["generated_tokens"],
            vehicle=live.vehicle_reply(pre["2d"]),
        )
        for geom in geometries:
            count, fraction = geom
            if geom == baseline:
                current = base
            else:
                def inside(uv, rectangle, factor=1.):
                    return original_inside(uv, rectangle, fraction if factor == .50 else factor)

                with patch.object(live, "MIN_RETURNS", count), patch.object(cray_simple, "MIN_RETURNS", count), \
                        patch.object(live.geometry, "inside", inside):
                    current = external.one_geometry((frame, row))
                for control in ("native", "window"):
                    assert base["boxes"][control] == current["boxes"][control]
                assert base["predicted_2d"] == current["predicted_2d"]
            for readout in ("frozen", "roof"):
                boxes[arm(readout, name, geom)] = current["boxes"][readout]
            diagnostics[arm("roof", name, geom)] = {k: current[k] for k in
                ("route", "active", "changed", "select_rays", "audit_rays", "changed_selection", "changed_audit", "protected_points")}
    return dict(uid=frame["uid"], recording=frame["recording"], boxes=boxes, diagnostics=diagnostics, front=front)


def generate(workers):
    started = perf_counter()
    cfg = registered()
    locked = pins(cfg)
    frames, _ = external.panel()
    profiles = {name: {r["uid"]: r for r in load(name)} for name in cfg["generation"]["profiles"]}
    ids = {f["uid"] for f in frames}
    assert all(set(p) == ids for p in profiles.values())
    jobs = ((f, archived, {name: r[f["uid"]] for name, r in profiles.items()}, settings(cfg))
            for f, archived in zip(frames, baseline_predictions(), strict=True))
    counts = Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def completed():
            for row in pool.map(make_geometry, jobs, chunksize=4):
                counts["descriptions"] += 1
                assert len(row["boxes"]) == 40 and len(row["diagnostics"]) == 16
                counts["roof_candidate_records"] += len(row["diagnostics"])
                if counts["descriptions"] % 24 == 0:
                    print(f"Geometry {counts['descriptions']}/{len(frames)} descriptions", flush=True)
                yield row

        payload = save_payload(PREDICTIONS, completed())
    assert counts["descriptions"] == len(frames) == cfg["population"]["descriptions"]
    roster.write(SEAL, dict(pins=locked, payload=payload, workers=workers, **counts, seconds=perf_counter() - started,
        exact_archived_baseline_boxes=len(frames), target_gt_opened=False, FINAL_access=False))


def metrics(values):
    values = np.asarray(values, dtype=float)
    return dict(acc25=100 * np.mean(values > .25), acc50=100 * np.mean(values > .50), miou=100 * np.mean(values))


def mean_recording_metrics(recordings, choice):
    values = [metrics([r[f"iou_{choice}"] for r in group]) for group in recordings]
    return {k: float(np.mean([v[k] for v in values])) for k in values[0]}


def rank(rows, choices, complexity):
    groups = {}
    for row in rows:
        groups.setdefault(row["recording"], []).append(row)
    scores = []
    for index, choice in enumerate(choices):
        mean = mean_recording_metrics(groups.values(), choice)
        acc50, miou = mean["acc50"], mean["miou"]
        scores.append(((-round(acc50, 12), -round(miou, 12), complexity[choice], index), choice, acc50, miou))
    _, winner, acc50, miou = min(scores)
    return winner, acc50, miou


def crossfit(rows, configs, baseline_geometry, profile_defs):
    kinds = {
        "generation_only": [a for a, (_, g) in configs.items() if g == baseline_geometry],
        "geometry_only": [a for a, (p, _) in configs.items() if p == BASE_PROFILE],
        "joint": list(configs),
        "window_only": [arm("window", p) for p in profile_defs],
    }

    def profile_difference(name):
        option = profile_defs[name]
        return int(option["max_new_tokens"] != 144) + int(option["prompt"] != "original")

    complexity = {a: profile_difference(p) + sum(x != y for x, y in zip(g, baseline_geometry, strict=True))
                  for a, (p, g) in configs.items()}
    complexity.update({arm("window", p): profile_difference(p) for p in profile_defs})
    folds = []
    scenes = sorted({r["recording"] for r in rows})
    for recording in scenes:
        train = [r for r in rows if r["recording"] != recording]
        held = [r for r in rows if r["recording"] == recording]
        for kind, choices in kinds.items():
            selected, acc50, miou = rank(train, choices, complexity)
            for row in held:
                row[f"iou_loo_{kind}"] = row[f"iou_{selected}"]
                if kind == "joint":
                    name, geom = configs[selected]
                    row["iou_loo_joint_matched_window"] = row[f"iou_{arm('window', name)}"]
                    row["iou_loo_joint_matched_frozen"] = row[f"iou_{arm('frozen', name, geom)}"]
            outcomes = metrics([r[f"iou_{selected}"] for r in held])
            folds.append(dict(recording=recording, selector=kind, chosen=selected, training_descriptions=len(train),
                training_recordings=len(scenes) - 1, heldout_descriptions=len(held), train_recording_acc50=acc50,
                train_recording_miou=miou, **{f"heldout_{k}": v for k, v in outcomes.items()}))
    development = {kind: rank(rows, choices, complexity)[0] for kind, choices in kinds.items()}
    return folds, development


def evaluation(commands):
    started = perf_counter()
    cfg = registered()
    receipt = roster.read(SEAL)
    assert pins(cfg) == receipt["pins"] and roster.sha(PREDICTIONS) == receipt["payload"]
    assert receipt["target_gt_opened"] is False and receipt["FINAL_access"] is False
    frames, original = external.panel()
    assert roster.sha(external.MATCH) == original["annotation_keys"]
    identities = roster.read(external.MATCH)
    required = {(v["split"], v["command"]) for v in identities.values()}
    with gzip.open(external.OUT / "talk2car_rows.csv.gz", "rt") as stream:
        archive = {row["uid"]: row for row in csv.DictReader(stream)}
    previous = roster.read(external.OUT / "talk2car_evaluation.json")
    assert roster.sha(external.OUT / "talk2car_rows.csv.gz") == previous["output_sha256"]["talk2car_rows.csv.gz"]
    official = external.official_labels(commands, required)
    profiles = cfg["generation"]["profiles"]
    geometries = settings(cfg)
    baseline_geometry = geometries[0]
    configs = {arm("roof", p, g): (p, g) for p, g in product(profiles, geometries)}
    baseline = arm("roof", BASE_PROFILE, baseline_geometry)
    window_baseline = arm("window", BASE_PROFILE)
    arms = [arm(a, p) for p in profiles for a in ("native", "window")]
    arms += [arm(a, p, g) for p, g in product(profiles, geometries) for a in ("frozen", "roof")]
    rows, coverage, reply_status, iou2d = [], {}, {}, {}
    errors, baseline_errors = [], []
    for frame, record in zip(frames, read_payload(PREDICTIONS), strict=True):
        uid = frame["uid"]
        assert uid == record["uid"] and frame["recording"] == record["recording"]
        match = identities[uid]
        label = official[match["split"], match["command"]]
        iou, difference = prior.safe_iou([record["boxes"][a] for a in arms], external.truth_box(label, match["camera_to_lidar"]))
        errors.append(difference)
        row = dict(uid=uid, recording=frame["recording"], target="nuscenes", camera_keyframe=frame["camera_keyframe"])
        row.update({f"iou_{k}": float(v) for k, v in zip(arms, iou, strict=True)})
        for a in ("native", "window", "frozen", "roof"):
            comparator = arm(a, BASE_PROFILE, baseline_geometry if a in ("frozen", "roof") else None)
            baseline_errors.append(abs(row[f"iou_{comparator}"] - float(archive[uid][f"iou_{a}"])))
        for a, v in record["diagnostics"].items():
            values = coverage.setdefault(a, Counter())
            values.update((f"route_{v['route']}",))
            values.update({k: int(x) for k, x in v.items() if k != "route"})
        for name, front in record["front"].items():
            status = reply_status.setdefault(name, Counter())
            for task in ("2d", "3d"):
                status[f"{task}_{front['status_'+task]}"] += 1
                status[f"{task}_at_cap"] += int(front[f"tokens_{task}"] == profiles[name]["max_new_tokens"])
            status["both_valid"] += int(front["status_2d"] == front["status_3d"] == "ok")
            status["eligible_vehicle"] += int(front["vehicle"] and front["status_3d"] == "ok")
            value = external.iou_2d(front["predicted_2d"], label["box_2d"])
            if value is not None:
                iou2d.setdefault(name, []).append(float(value))
                row[f"iou2d_{name}"] = float(value)
        rows.append(row)

    assert len(rows) == len(archive) == cfg["population"]["descriptions"]
    assert max(errors) < 1e-7 and max(baseline_errors) <= 1e-12
    folds, development = crossfit(rows, configs, baseline_geometry, profiles)
    held_arms = [k.removeprefix("iou_") for k in rows[0] if k.startswith("iou_loo_")]
    pairs = {(baseline, window_baseline), ("loo_window_only", window_baseline)}
    for name, (p, g) in configs.items():
        pairs.update({(name, baseline), (name, arm("frozen", p, g)), (name, arm("window", p))})
    for a in held_arms:
        pairs.update({(a, baseline), (a, window_baseline)})
    pairs.update({("loo_joint", "loo_window_only"), ("loo_joint", "loo_joint_matched_window"),
                  ("loo_joint", "loo_joint_matched_frozen")})
    pairs = tuple(sorted((a, b) for a, b in pairs if a != b))
    all_arms = arms + held_arms
    with patch.object(common, "OUT", OUT):
        files, intervals = common.group_summary("tuning", rows, all_arms, pairs)
        audited = common.independent_tally("tuning", files)
    names = ("route_A", "route_H", "route_C_ray", "route_missing_3d", "active", "changed",
             "select_rays", "audit_rays", "changed_selection", "changed_audit", "protected_points")
    branches = [dict(arm=a, profile=p, minimum_returns=g[0], central_fraction=g[1],
                     **{k: coverage[a][k] for k in names}) for a, (p, g) in configs.items()]
    assert all(sum(x[k] for k in names[:4]) == len(rows) and x["changed_selection"] == 0 for x in branches)
    files["configuration_folds.csv"] = prior.scorer.write_rows("configuration_folds.csv", folds)
    files["coverage.csv"] = prior.scorer.write_rows("coverage.csv", branches)
    by_recording = {}
    for row in rows:
        by_recording.setdefault(row["recording"], []).append(row)
    recording_metrics = [dict(arm=a, n=len(rows), recordings=len(by_recording),
                              **mean_recording_metrics(by_recording.values(), a)) for a in all_arms]
    files["recording_macro_metrics.csv"] = prior.scorer.write_rows("recording_macro_metrics.csv", recording_metrics)
    frontend = {p: dict(counts=dict(reply_status[p]), scored_2d=len(iou2d[p]),
                        acc50_2d=100 * np.mean(np.asarray(iou2d[p]) > .50)) for p in profiles}
    roster.write(OUT / "evaluation.json", dict(pins=receipt["pins"], prediction_seal=roster.sha(SEAL), files=files,
        descriptions=len(rows), recordings=cfg["population"]["recordings"], score_audit=audited,
        max_two_iou_difference=max(errors), max_archived_baseline_iou_difference=max(baseline_errors),
        generation=frontend, development_choices=development,
        underlying_profiles=profiles, recording_bootstrap_95pp=intervals,
        seconds=perf_counter() - started, FINAL_access=False))
    print(f"{len(rows)} descriptions and {len(configs)} configurations verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("generate", "score"))
    parser.add_argument("--commands", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "generate":
        generate(args.workers)
    else:
        if args.commands is None:
            parser.error("score requires --commands")
        evaluation(args.commands)
