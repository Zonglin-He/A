#!/usr/bin/env python3
"""Run the locked single-frame 3EED readouts and score sealed predictions."""

import argparse
from collections import Counter
import hashlib
from importlib import import_module
import json
from pathlib import Path
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]
STUDIES = ROOT / "studies"
FRONTEND = STUDIES / "3eed_qwen_full_ordinary_validation_20261121"
QWEN = STUDIES / "3eed_frozen_qwen_metric_grounding_20261120"
ROOF = STUDIES / "3eed_qwen_extent_height_only_20261122"
ARCHIVE = STUDIES / "3eed_single_frame_crossplatform_grounding_20261118" / "own_code"
CANDIDATE_LOCK = STUDIES / "3eed_pose_recording_exposure_audit_20261121/CANDIDATE_LOCK.json"
POINT_HELPER = "crossplatform/round7/experiment/runner_helpers.py"
WAYMO_TRANSFORM_SHA256 = "574760753be97ba86b9e8b1e6e88a69ffa25520460790acb06099ee2aaa17683"
PLATFORMS = ("waymo", "drone", "quad")
VIEWS = ("F", "FL", "FR", "SL", "SR")
HISTORIC_ARMS = ("C_ray_simple", "ray_equivalent_height_only", "ray_equivalent_extent")
ARMS = HISTORIC_ARMS  # Imported by historical study runners.
PROMOTED_ARM = "ray_equivalent_height_only_2d_rank"
REGISTRATION = ROOT / "QWEN_CURRENT_METHOD.json"
INPUT_FIELDS = {"uid", "target", "text", "rgb", "points", "calibration", "suffix", "seed"}
CAMERA_FIELDS = {"image_intrinsic", "image_extrinsic", "camera_distortion"}


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def source_paths(upstream):
    view_transform = upstream / "utils/transform_waymo.py"
    if not view_transform.is_file() or digest(view_transform) != WAYMO_TRANSFORM_SHA256:
        raise ValueError(f"Missing pinned 3EED source at {upstream}")
    source = ROOT / "v1"
    sys.path[:0] = list(map(str, (
        FRONTEND, ROOF, QWEN, source, (ARCHIVE / POINT_HELPER).parent,
        source / "crossplatform/round7", source / "ablation/own_code",
        STUDIES / "3eed_v2_ordinary_val_subject_ranking_20261119/own_code",
        ARCHIVE / "research/round6/source_validation_scoring", ARCHIVE / "tools", upstream,
    )))


def check_geometry():
    import runner_helpers

    for name, expected in read(CANDIDATE_LOCK)["local_dependency_sha256"].items():
        if name == POINT_HELPER:
            path = Path(runner_helpers.__file__)
        else:
            path = (ROOT / "v1" if name.startswith("crossplatform/") else ROOT) / name
        if digest(path) != expected:
            raise ValueError(f"Geometric dependency differs from the candidate lock: {path}")


def input_frames(manifest):
    rows = read(manifest)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain a nonempty JSON list")
    frames = {}
    for row in rows:
        if set(row) != INPUT_FIELDS or row["target"] not in PLATFORMS:
            raise ValueError("Expected only the documented inference fields and a supported platform")
        uid = row["uid"]
        if not isinstance(uid, str) or not uid or uid in frames or not row["text"].strip():
            raise ValueError("Each frame must have a unique UID and a nonempty expression")
        suffix = row["suffix"]
        if type(suffix) is not int or suffix not in range(5) or (row["target"] != "waymo" and suffix != 0):
            raise ValueError(f"Invalid 3EED camera suffix for {uid}")
        if type(row["seed"]) is not int or row["seed"] < 0:
            raise ValueError(f"Invalid sampling seed for {uid}")
        for field in ("rgb", "points", "calibration"):
            row[field] = str((manifest.parent / row[field]).resolve(strict=True))
        camera = read(Path(row["calibration"]))
        if not {"image_intrinsic", "image_extrinsic"} <= camera.keys() <= CAMERA_FIELDS:
            raise ValueError(f"Camera JSON for {uid} must contain only camera fields")
        frames[uid] = dict(row, waymo_view=VIEWS[suffix])
    return frames


def predict(args):
    import torch
    import transformers

    frames = list(input_frames(args.manifest).values())
    if len(frames) % 2:
        raise ValueError("An even number of descriptions is required for complete four-query batches")
    method = read(REGISTRATION)
    frozen = read(ROOT / method["controls"]["frozen"]["method_lock"])["entry"]
    for lock in (method["prediction_entry"], method["method_lock"], frozen):
        if digest(ROOT / lock["file"]) != lock["sha256"]:
            raise ValueError(f"Locked file differs from its registered hash: {lock['file']}")
    method_lock = read(ROOT / method["method_lock"]["file"])
    for filename, expected in method_lock["config"]["historical_geometry"].items():
        if digest(ROOT / filename) != expected:
            raise ValueError(f"Registered dependency differs from lock: {filename}")
    check_geometry()

    from run_original_schedule import load_model, predict as qwen_predict
    from live import predict as roof_predict, frozen_predict

    entry = method["prediction_entry"]
    module = import_module(Path(entry["file"]).stem)
    if Path(module.__file__).resolve() != (ROOT / entry["file"]).resolve():
        raise ValueError("The registered prediction module was not loaded")
    promoted = getattr(module, entry["function"])
    args_for_entry = entry["kwargs"]
    arms = (*HISTORIC_ARMS, PROMOTED_ARM)

    started = perf_counter()
    files = {Path(frame[field]) for frame in frames for field in ("rgb", "points", "calibration")}
    hashes = {str(path): digest(path) for path in sorted(files)}
    for frame in frames:
        frame["rgb_sha"] = hashes[frame["rgb"]]
    args.output.mkdir(parents=True, mode=0o700)
    rows = []
    with torch.inference_mode():
        processor, model, limit = load_model(args.model_dir)
        for start in range(0, len(frames), 2):
            group = frames[start:start + 2]
            jobs = [(frame, task) for frame in group for task in ("3d", "2d")]
            answers = qwen_predict(processor, model, limit, jobs)
            for index, frame in enumerate(group):
                replies = dict(zip(("3d", "2d"), answers[2*index:2*index + 2], strict=True))
                predictions = (
                    frozen_predict(frame, replies),
                    roof_predict(frame, replies, height_only=True),
                    roof_predict(frame, replies, height_only=False),
                    promoted(frame, replies, **args_for_entry),
                )
                rows.append(dict(uid=frame["uid"], target=frame["target"], replies=replies,
                                 predictions=dict(zip(arms, predictions, strict=True))))
            if start == 0 or len(rows) % 128 == 0 or len(rows) == len(frames):
                print(f"Predicted {len(rows)}/{len(frames)} descriptions", flush=True)

    payload = args.output / "predictions.json"
    publish(payload, rows)
    publish(args.output / "seal.json", dict(
        predictions_sha256=digest(payload), manifest_sha256=digest(args.manifest),
        method_sha256=digest(REGISTRATION), input_file_sha256=hashes, arms=arms,
        candidate_lock_sha256=digest(CANDIDATE_LOCK),
        model_revision=method["model"]["revision"], descriptions=len(rows),
        counts=Counter(row["target"] for row in rows), target_labels_read=False,
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        transformers=transformers.__version__, python=sys.version.split()[0],
        peak_gpu_bytes=torch.cuda.max_memory_allocated(), seconds=perf_counter() - started,
    ))


def score(args):
    import numpy as np
    import score_ablation
    from score import canonical_box, score_boxes, truth_for_slot

    seal_path = args.run / "seal.json"
    seal = read(seal_path)
    payload = args.run / "predictions.json"
    method = read(REGISTRATION)
    legacy = ROOT / method["legacy_registration"]
    registration_hashes = (digest(REGISTRATION), digest(legacy))
    arms = tuple(seal.get("arms", HISTORIC_ARMS))
    if arms not in (HISTORIC_ARMS, (*HISTORIC_ARMS, PROMOTED_ARM)):
        raise ValueError("Scoring arms differ from the supported registered readouts")
    if (seal["target_labels_read"] or digest(payload) != seal["predictions_sha256"]
            or digest(args.manifest) != seal["manifest_sha256"]
            or seal["method_sha256"] not in registration_hashes
            or digest(CANDIDATE_LOCK) != seal["candidate_lock_sha256"]):
        raise ValueError("Prediction seal or manifest does not match")
    if args.output.exists():
        raise FileExistsError(args.output)
    frames = input_frames(args.manifest)
    predictions = read(payload)
    if [row["uid"] for row in predictions] != list(frames) or len(frames) != seal["descriptions"]:
        raise ValueError("Sealed predictions do not cover the manifest in order")

    score_ablation.DATA = args.dataset.resolve(strict=True)
    labels = read(args.labels)
    by_uid = {row["uid"]: row for row in labels}
    if len(labels) != len(by_uid) or by_uid.keys() != frames.keys():
        raise ValueError("Offline labels and sealed predictions must have the same unique UIDs")
    rows = []
    max_error = 0.0
    for row in predictions:
        uid = row["uid"]
        frame = frames[uid]
        label = by_uid[uid]
        if set(label) != {"uid", "metadata", "slot"} or type(label["slot"]) is not int or label["slot"] < 0:
            raise ValueError(f"Invalid offline label reference: {uid}")
        metadata = (args.labels.parent / label["metadata"]).resolve(strict=True)
        if (metadata.name != "meta_info.json" or metadata.parents[2] != score_ablation.DATA / frame["target"]
                or metadata.parent != Path(frame["rgb"]).parent or metadata.parent != Path(frame["points"]).parent):
            raise ValueError(f"Offline label path is outside the dataset platform: {uid}")
        label_frame = dict(frame, calibration=str(metadata), slot=label["slot"])
        truth = canonical_box(truth_for_slot(label_frame, read(metadata)["ground_info"]))
        boxes = {arm: row["predictions"][arm]["box"] for arm in arms}
        valid = {arm: box for arm, box in boxes.items() if box is not None}
        ious = dict.fromkeys(arms, 0.0)
        if valid:
            values, error = score_boxes(np.array(list(valid.values())), truth)
            ious.update(zip(valid, map(float, values), strict=True))
            max_error = max(max_error, error)
        rows.append(dict(uid=uid, target=frame["target"], iou=ious,
                         height_corrected=row["predictions"][HISTORIC_ARMS[1]]["extent_corrected"]))

    per_platform = {}
    for platform in PLATFORMS:
        current = [row for row in rows if row["target"] == platform]
        if current:
            metrics = {}
            for arm in arms:
                values = np.array([row["iou"][arm] for row in current])
                metrics[arm] = dict(acc25=100*float((values > .25).mean()),
                                    acc50=100*float((values > .50).mean()), miou=100*float(values.mean()))
            per_platform[platform] = dict(n=len(current), metrics=metrics)
    macro = {arm: {key: float(np.mean([group["metrics"][arm][key] for group in per_platform.values()]))
                   for key in ("acc25", "acc50", "miou")} for arm in arms}
    publish(args.output, dict(seal_sha256=digest(seal_path), n=len(rows), max_iou_check_error=max_error,
                              height_corrections=sum(row["height_corrected"] for row in rows),
                              macro=macro, per_platform=per_platform, rows=rows))
    print(json.dumps(dict(n=len(rows), macro=macro), indent=2), flush=True)


def main():
    if not __debug__:
        raise RuntimeError("The locked model and scoring checks require Python assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    infer = subparsers.add_parser("predict")
    infer.add_argument("--model-dir", required=True, type=Path)
    evaluate = subparsers.add_parser("score")
    evaluate.add_argument("--run", required=True, type=Path)
    evaluate.add_argument("--dataset", required=True, type=Path)
    evaluate.add_argument("--labels", required=True, type=Path)
    for command in (infer, evaluate):
        command.add_argument("--manifest", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source_paths(args.upstream.resolve())
    (predict if args.phase == "predict" else score)(args)


if __name__ == "__main__":
    main()
