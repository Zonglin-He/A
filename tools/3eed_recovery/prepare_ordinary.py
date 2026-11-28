#!/usr/bin/env python3
"""Prepare a fresh 3EED Ordinary Validation run using published cohort membership."""

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

from run_qwen_3eed import PLATFORMS, ROOT, ROOF, publish


def known_uids():
    # Only the opaque IDs and platform columns are used. Historical scores
    # play no part in membership, ordering, input construction, or inference.
    result = {platform: set() for platform in PLATFORMS}
    with gzip.open(ROOF / "results/rows.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            result[row["target"]].add(row["uid"])
    if {k: len(v) for k, v in result.items()} != {"waymo": 3782, "drone": 2792, "quad": 2736}:
        raise ValueError("Published Ordinary Validation membership is incomplete")
    return result


def identifier(platform, recording, frame, slot):
    record = json.dumps(("ordinary-val-description", platform, recording, frame, slot), separators=(",", ":"))
    return hashlib.sha256(record.encode()).hexdigest()[:24]


def matches(dataset, platform, membership, count):
    available = dataset / platform
    matches = []
    for recording in sorted(p for p in available.iterdir() if p.is_dir()):
        for frame in sorted(p for p in recording.iterdir() if p.is_dir()):
            for slot in range(64):
                uid = identifier(platform, recording.name, frame.name, slot)
                if uid in membership:
                    matches.append((uid, frame, slot))
                    if len(matches) == count:
                        return matches
                    # The smoke cohort uses one description per frame.
                    if count is not None:
                        break
    if count is None and {uid for uid, _, _ in matches} == membership:
        return matches
    raise ValueError(f"Could not match the requested Ordinary Validation descriptions for {platform}")


def build(dataset, output, count):
    import ijson
    import numpy as np

    sys.path.insert(0, str(ROOT / "v1"))
    from crossplatform.round7.camera_projection import read_camera_allowlist

    by_platform = known_uids()
    cases = {target: matches(dataset, target, by_platform[target], count) for target in PLATFORMS}
    output.mkdir(mode=0o700, parents=True)
    camera_dir = output / "camera"
    camera_dir.mkdir()
    inputs, labels = [], []
    for target, members in cases.items():
        for uid, frame, slot in members:
            metadata = frame / "meta_info.json"
            rgb = frame / "image.jpg"
            lidar = frame / ("lidar.npy" if target == "waymo" else "lidar.bin")
            camera = camera_dir / f"{uid}.json"
            with metadata.open("rb") as stream:
                captions = list(ijson.items(stream, "ground_info.item.caption"))
            fields = read_camera_allowlist(metadata)
            publish(camera, {key: np.asarray(value, dtype=float).tolist()
                             for key, value in fields.items() if value is not None})
            identity = f"{target}|{metadata.relative_to(dataset).as_posix()}|{slot}"
            seed = int(hashlib.sha256(("R97point|" + identity).encode()).hexdigest()[:16], 16)
            suffix = int(frame.name.rsplit("_", 1)[-1]) if target == "waymo" else 0
            inputs.append(dict(uid=uid, target=target, text=captions[slot], suffix=suffix, seed=seed,
                               rgb=os.path.relpath(rgb.resolve(strict=True), output),
                               points=os.path.relpath(lidar.resolve(strict=True), output),
                               calibration=os.path.relpath(camera, output)))
            labels.append(dict(uid=uid, metadata=os.path.relpath(metadata, output), slot=slot))
    publish(output / "manifest.json", inputs)
    publish(output / "labels.json", labels)
    print(dict(Counter(row["target"] for row in inputs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--per-platform", type=int, default=4)
    scope.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and (args.per_platform < 2 or args.per_platform % 2):
        parser.error("--per-platform must be a positive even number")
    build(args.dataset.resolve(strict=True), args.output.resolve(), None if args.all else args.per_platform)


if __name__ == "__main__":
    main()
