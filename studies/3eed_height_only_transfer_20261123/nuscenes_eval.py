"""Run the unchanged frozen Qwen geometry on the external single-frame cohort."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from common import OUT, PRIVATE, group_summary, independent_tally, lock, prior, read_payload, roster, save_payload, scorer
from nuscenes_data import INPUT_SEAL, MATCH, TALK_REVISION, annotations, panel
from live import geometry, incumbent, predict
from geometry import upright


ARMS = ("native", "window", "frozen", "roof")
PAIRS = (("window", "native"), ("frozen", "window"), ("roof", "window"), ("roof", "frozen"))
REPLIES = PRIVATE / "talk2car_replies.jsonl.gz"
REPLY_SEAL = OUT / "talk2car_qwen_seal.json"
OUTPUT = PRIVATE / "talk2car_predictions.jsonl.gz"
PRED_SEAL = OUT / "talk2car_prediction_seal.json"


def infer(weights):
    from concurrent_schedule import STREAMS, block, read_games, worker_inputs
    from run_full_qwen import load_model
    import torch

    started = perf_counter()
    pins = lock(Path(__file__))
    frames, registered = panel()
    digest = lru_cache(maxsize=None)(roster.sha)
    for frame in frames:
        assert digest(frame["rgb"]) == frame["rgb_sha"]
    processor, model, limit = load_model(weights)
    workers = worker_inputs(weights, model, processor)
    work = [(frame, task) for frame in frames for task in roster.TASKS]
    groups = [work[i:i+4] for i in range(0, len(work), 4)]
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=STREAMS) as pool:
        probe = groups[:min(STREAMS, len(groups))]
        normal = [block(workers[0], limit, g) for g in probe]
        simultaneous = read_games(pool, workers, limit, probe)
        matches = sum(a == b for x, y in zip(normal, simultaneous, strict=True) for a, b in zip(x, y, strict=True))
        tested = sum(map(len, probe))
        streams = STREAMS if matches == tested else 1
        entries = {f["uid"]: {} for f in frames}
        jobs = 0
        for i in range(0, len(groups), streams):
            batch = groups[i:i+streams]
            finished = read_games(pool, workers, limit, batch) if streams > 1 else [block(workers[0], limit, batch[0])]
            for arm, replies in zip(batch, finished, strict=True):
                for (frame, task), response in zip(arm, replies, strict=True):
                    entries[frame["uid"]][task] = response
                    jobs += 1
                if jobs % 100 == 0:
                    print("Talk2Car Qwen answers", jobs, "/", len(work), flush=True)
    assert jobs == len(work) and all(set(x) == set(roster.TASKS) for x in entries.values())
    output = save_payload(REPLIES, (dict(uid=uid, replies=entry) for uid, entry in entries.items()))
    roster.write(REPLY_SEAL, dict(payload=output, input=roster.sha(INPUT_SEAL), pins=pins,
        source_qwen_assets=roster.sha(roster.OLD / "MODEL.json"), descriptions=registered["descriptions"], queries=len(work),
        batch=4, streams=streams, exact_probe_matches=matches, probe_answers=tested, measured_gpu=torch.cuda.get_device_name(),
        peak_bytes=torch.cuda.max_memory_allocated(), seconds=perf_counter()-started, target_gt_opened=False, FINAL_access=False))


def read_replies():
    seal = roster.read(REPLY_SEAL)
    assert not seal["target_gt_opened"] and seal["pins"] == lock(Path(__file__))
    assert seal["input"] == roster.sha(INPUT_SEAL) and seal["payload"] == roster.sha(REPLIES)
    return read_payload(REPLIES)


def one_geometry(job):
    frame, row = job
    assert frame["uid"] == row["uid"]
    for path, check in (("points", "point_sha"), ("calibration", "camera_sha"), ("rgb", "rgb_sha")):
        assert roster.sha(frame[path]) == frame[check]
    replies = row["replies"]
    raw, rectangle = (replies[k]["value"] for k in ("3d", "2d"))
    original, context = incumbent(frame, replies, collect_audit=True)
    protected = predict(frame, replies)
    assert original["applicable"] == protected["applicable"]
    changed = int(protected["extent_corrected"])
    stats = dict(changed=changed, active=protected["applicable"], select_rays=0, audit_rays=0,
                 changed_selection=0, changed_audit=0, protected_points=0)
    if changed:
        box, _dims, _rot, fit, selection, audit, transform = context
        from live import first_depth
        from extent_clip import protected_points
        dp = first_depth(box, selection, transform)
        dr = first_depth(np.asarray(protected["box"]), selection, transform)
        n, d = prior.reference.invariance(dp, dr)
        assert n == 0 and d <= 1.e-8
        aud, _ = prior.reference.invariance(first_depth(box, audit, transform),
                                             first_depth(np.asarray(protected["box"]), audit, transform))
        _eye, measured, _entries = protected_points(box, fit, selection, dp, transform)
        stats.update(select_rays=len(selection), audit_rays=len(audit), changed_selection=n, changed_audit=aud,
                     protected_points=len(measured))
    native = None
    size = None
    if raw is not None or rectangle is not None:
        _, size, transform = geometry.camera(frame)
        if raw is not None:
            native = geometry.to_sensor(np.asarray(raw), transform)[0].tolist()
    reference = prior.geometry_full.predict(frame, replies)["boxes"]["A"]
    predicted2d = (np.asarray(rectangle) * np.tile(size, 2) / 1000.).tolist() if rectangle is not None else None
    return dict(uid=frame["uid"], route=original["readout"], boxes=dict(native=native, window=reference,
                frozen=original["box"], roof=protected["box"]), predicted_2d=predicted2d, **stats)


def generate(workers):
    start = perf_counter()
    inputs, seal = panel()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = pool.map(one_geometry, zip(inputs, read_replies(), strict=True), chunksize=12)
        payload = save_payload(OUTPUT, rows)
    roster.write(PRED_SEAL, dict(input=roster.sha(INPUT_SEAL), replies=roster.sha(REPLY_SEAL), payload=payload,
        pins=lock(Path(__file__)), descriptions=seal["descriptions"], geometry_replay=True,
        seconds=perf_counter()-start, target_gt_opened=False, FINAL_access=False))


def truth_box(label, transform):
    transform = np.asarray(transform, dtype=float)
    center = geometry.rigid_xyz(np.asarray(label["translation"], dtype=float), transform)
    # nuScenes stores width, length, height. The box's local x axis carries length.
    dims = np.asarray(label["size"], dtype=float)[[1, 0, 2]]
    qw, qx, qy, qz = label["rotation"]
    axes = transform[:3, :3] @ Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    return upright(center, dims, axes)


def official_labels(folders, required):
    items = {}
    checks = roster.read(folders / "sources.json")
    assert checks == roster.read(INPUT_SEAL)["language"]
    for split, record in checks.items():
        path = folders / record["filename"]
        assert TALK_REVISION in record["source"] and roster.sha(path) == record["sha256"]
        for data in roster.read(path)["commands"]:
            key = split, data["command_token"]
            if key in required:
                assert key not in items
                items[key] = dict({k: data[k] for k in ("translation", "size", "rotation", "obj_name")},
                                  box_2d=data.get("2d_box"))
    assert items.keys() == required
    return items


def iou_2d(prediction, truth):
    if truth is None:
        return None
    if prediction is None:
        return 0.
    box = np.asarray(prediction, dtype=float)
    gt = np.asarray(truth, dtype=float).copy()
    gt[2:] += gt[:2]
    inter = np.prod(np.maximum(0, np.minimum(box[2:], gt[2:]) - np.maximum(box[:2], gt[:2])))
    area = np.prod(box[2:]-box[:2])+np.prod(gt[2:]-gt[:2])-inter
    return float(inter/area) if area > 0 else 0.


def evaluate(archive, commands):
    t0 = perf_counter()
    seal = roster.read(PRED_SEAL)
    assert seal["pins"] == lock(Path(__file__)) and not seal["target_gt_opened"] and not seal["FINAL_access"]
    assert seal["input"] == roster.sha(INPUT_SEAL) and seal["replies"] == roster.sha(REPLY_SEAL)
    assert seal["payload"] == roster.sha(OUTPUT)
    frames, input_seal = panel()
    assert roster.sha(MATCH) == input_seal["annotation_keys"]
    identities = roster.read(MATCH)
    truth = annotations(archive)
    required = {(v["split"], v["command"]) for v in identities.values()}
    official = official_labels(commands, required)
    rows, errors = [], []
    matched_keyframe_boxes, missing_keyframe_annotations = [], 0
    for frame, prediction in zip(frames, read_payload(OUTPUT), strict=True):
        uid = frame["uid"]
        assert prediction["uid"] == uid
        matched = identities[uid]
        annotation = official[matched["split"], matched["command"]]
        truth3d = truth_box(annotation, matched["camera_to_lidar"])
        gt2d = annotation["box_2d"]
        if frame["camera_keyframe"]:
            backup = truth.get(matched["box"])
            if backup is not None and backup["sample_token"] == matched["sample"]:
                other_3d = truth_box(backup, matched["world_to_lidar"])
                compared, _ = prior.safe_iou([other_3d], truth3d)
                matched_keyframe_boxes.append(float(compared[0]))
            else:
                missing_keyframe_annotations += 1
        vals, err = prior.safe_iou([prediction["boxes"][a] for a in ARMS], truth3d)
        row = dict(uid=uid, target="nuscenes", recording=frame["recording"], annotation_split=frame["split"],
                   camera_keyframe=frame["camera_keyframe"], category=annotation["obj_name"],
                   predicted_2d_iou=iou_2d(prediction["predicted_2d"], gt2d))
        row.update({k: prediction[k] for k in ("route", "active", "changed", "select_rays", "audit_rays",
                   "changed_selection", "changed_audit", "protected_points")})
        row.update({f"iou_{a}": float(v) for a, v in zip(ARMS, vals, strict=True)})
        rows.append(row)
        errors.append(err)
    assert len(rows) == seal["descriptions"] and max(errors) < 1.e-7
    hashes, intervals = group_summary("talk2car", rows, ARMS, PAIRS)
    scored = independent_tally("talk2car", hashes)
    subsplit = []
    for split in sorted({r["annotation_split"] for r in rows}):
        subset = [r for r in rows if r["annotation_split"] == split]
        subsplit.extend(dict(split=split, **r) for r in scorer.table_row_metrics(subset, "pooled"))
    hashes["talk2car_annotation_splits.csv"] = scorer.write_rows("talk2car_annotation_splits.csv", subsplit)
    samples = []
    for is_keyframe in (True, False):
        group = [r for r in rows if r["camera_keyframe"] == is_keyframe]
        if group:
            samples.extend(dict(camera_keyframe=is_keyframe, **r) for r in scorer.table_row_metrics(group, "pooled"))
    hashes["talk2car_camera_timing.csv"] = scorer.write_rows("talk2car_camera_timing.csv", samples)
    provided_2d = [r["predicted_2d_iou"] for r in rows if r["predicted_2d_iou"] is not None]
    roster.write(OUT / "talk2car_evaluation.json", dict(predictions=roster.sha(PRED_SEAL), output_sha256=hashes,
        independent=scored, two_iou_max_difference=max(errors), annotated_splits=dict(Counter(f["split"] for f in frames)),
        match_camera_vs_sample_3d=dict(n=len(matched_keyframe_boxes), missing=missing_keyframe_annotations,
                                      min_iou=min(matched_keyframe_boxes, default=None)),
        qwen_2d_annotated=len(provided_2d), qwen_2d_acc50_percent=100*np.mean(np.asarray(provided_2d) > .50) if provided_2d else None,
        changed={k: sum(r[k] for r in rows) for k in ("active", "changed", "select_rays", "audit_rays", "changed_selection", "changed_audit")},
        recording_bootstrap_95pp=intervals, seconds=perf_counter()-t0, FINAL_access=False))
    print(len(rows), "external commands scored and audited", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("infer", "generate", "evaluate"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--commands", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "infer":
        infer(args.model)
    elif args.phase == "generate":
        generate(args.workers)
    else:
        evaluate(args.archive, args.commands)
