"""Author visual identity probes from the protected ordinary data view."""
import argparse
from collections import Counter, defaultdict, deque
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import HERE, ROOT, PRIVATE, OUT, PLATFORMS, PROTOCOL, SIDE, digest, key, read, write
from geometry import box_rect, calibration, observations, transform_xyz

sys.path.insert(0, str(ROOT / "studies/3eed_v2_ordinary_val_subject_ranking_20261119"))
from scope183 import ordinary

NOUN = {"car": "car", "pedestrian": "person", "truck": "truck", "bus": "bus",
        "cyclist": "cyclist", "motorcyclist": "motorcyclist"}
COLORS = ("#ff3c40", "#4264ff")


def overlaps(a, b):
    wh = np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    return np.prod(wh) / min(np.prod(a[2:] - a[:2]), np.prod(b[2:] - b[:2]))


def inventory(metadata, spec, transform, size):
    boxes = []
    for item in metadata["ground_info"]:
        candidates = [(item["class"], item["bbox_3d"])]
        candidates += [(obj["class_other"], obj["bbox_3d_other"]) for obj in item.get("others", ())]
        for label, raw in candidates:
            if raw is None or len(raw) < 7:
                continue
            box = np.asarray(raw[:7], float)
            noun = NOUN.get(label.strip().lower())
            if noun is None or not np.isfinite(box).all() or min(box[3:6]) <= 0.:
                continue
            duplicate = next((r for r in boxes if np.linalg.norm(r["box"][:3] - box[:3]) < .075
                              and np.max(abs(r["box"][3:6] - box[3:6])) < .15), None)
            if duplicate is not None:
                if duplicate["noun"] != noun:
                    return [], True
                continue
            rectangle = box_rect(box, spec, transform, size)
            boxes.append(dict(noun=noun, box=box, rect=rectangle))
    return [r for r in boxes if r["rect"] is not None], False


def pairs(objects, uid):
    count = Counter(obj["noun"] for obj in objects)
    eligible = []
    for a, b in combinations(objects, 2):
        if overlaps(a["rect"], b["rect"]) > .25:
            continue
        if np.linalg.norm(a["box"][:2] - b["box"][:2]) < 1.:
            continue
        unique = (count[a["noun"]] == 1) + (count[b["noun"]] == 1)
        eligible.append((-unique, key(uid, a["box"][:3].tolist(), b["box"][:3].tolist()), a, b))
    if not eligible:
        return None
    _, _, a, b = min(eligible, key=lambda x: x[:2])
    if int(key(uid, "options"), 16) % 2:
        a, b = b, a
    return a, b


def view_scans(table, platform, excluded, physical_hash):
    groups = defaultdict(list)
    for item in table:
        signature = physical_hash(item["pointpath"])
        if signature not in excluded:
            groups[signature].append(item)
    queues = defaultdict(list)
    for signature, group in groups.items():
        representative = min(group, key=lambda r: key("view", platform, signature, r["suffix"], r["seed"]))
        queues[representative["recording"]].append((signature, representative))
    return {rec: deque(sorted(items, key=lambda x: key("sample", platform, x[0])))
            for rec, items in queues.items()}


def capture():
    previous = ROOT / "studies/3eed_frozen_qwen_metric_grounding_20261120"
    qwen_source = previous / "private/panel.json"
    assert digest(qwen_source) == read(previous / "results/panel_seal.json")["panel"]
    excluded = {r["point_sha"] for r in read(qwen_source)}
    limit = read(PROTOCOL)["scope"]["physical_frames_per_platform"]
    physical_hash = lru_cache(maxsize=None)(digest)
    candidates, gold, screening = [], {}, {}
    depth_folder = PRIVATE / "observations"
    depth_folder.mkdir(exist_ok=True, mode=0o700)
    for platform, table in ordinary().items():
        queues = view_scans(table, platform, excluded, physical_hash)
        recs = sorted(queues, key=lambda r: key("recording_order", platform, r))
        screened = Counter()
        count = 0
        while count < limit and any(queues.values()):
            for rec in recs:
                while queues[rec]:
                    signature, source = queues[rec].popleft()
                    uid = key("hispatial_frame", platform, signature)
                    frame = dict(
                        uid=uid, target=platform, recording=key("hispatial_recording", platform, rec),
                        points=source["pointpath"], rgb=source["imagepath"],
                        calibration=source["metapath"], suffix=source["suffix"], point_sha=signature,
                    )
                    screened["frames_screened"] += 1
                    spec, size, transform = calibration(frame)
                    metadata = read(frame["calibration"])
                    objects, conflict = inventory(metadata, spec, transform, size)
                    pair = None if conflict else pairs(objects, uid)
                    if pair is None:
                        screened["no_visible_separated_pair"] += 1
                        continue
                    obs = observations(frame, spec, size, transform)
                    if int(obs["hold"].sum()) < 16:
                        screened["under_16_holdout_pixels"] += 1
                        continue
                    identity = Counter(obj["noun"] for obj in objects)
                    for box in pair:
                        box["noun_unique"] = identity[box["noun"]] == 1
                    labels = []
                    for box in pair:
                        labels.append({**box, "box": box["box"].tolist(), "rect": box["rect"].tolist()})
                    timestamp_record = dict(
                        shared_metadata_timestamp="timestamp" in metadata,
                        rgb_lidar_filename_stem_match=Path(metadata["image_path"]).stem == Path(metadata["lidar_path_proj"]).stem,
                        independent_rgb_and_scan_clock_fields=sum(k in metadata for k in ("camera_timestamp","lidar_timestamp")),
                        named_distortion_present=not spec.validated()[3],
                    )
                    gold[uid] = dict(objects=labels, **timestamp_record)
                    frame["rgb_sha"] = digest(frame["rgb"])
                    frame["calibration_sha"] = digest(frame["calibration"])
                    frame["size"] = size.tolist()
                    cache = depth_folder / (uid + ".npz")
                    np.savez_compressed(cache, **obs)
                    frame["observation_sha"] = digest(cache)
                    candidates.append(frame)
                    count += 1
                    break
                if count == limit:
                    break
        assert count == limit, (platform, count, dict(screened))
        screening[platform] = dict(screened)
        print(platform, count, screening[platform], flush=True)
    assert len({r["point_sha"] for r in candidates}) == len(candidates)
    write(PRIVATE / "captured.json", candidates)
    write(PRIVATE / "candidate_truth.json", gold)
    write(OUT / "capture_seal.json", dict(capture=digest(PRIVATE / "captured.json"),
          labels=digest(PRIVATE / "candidate_truth.json"), prior_qwen_frames_excluded=len(excluded),
          protocol=digest(PROTOCOL), author=digest(__file__), screening=screening, FINAL_access=False))


def review_pages():
    frames = read(PRIVATE / "captured.json")
    labels = read(PRIVATE / "candidate_truth.json")
    folder = PRIVATE / "review"
    folder.mkdir(exist_ok=True, mode=0o700)
    font = ImageFont.truetype("DejaVuSans.ttf", 14)
    width, height = 430, 310
    for platform in PLATFORMS:
        subset = [r for r in frames if r["target"] == platform]
        for start in range(0, len(subset), 6):
            canvas = Image.new("RGB", (width * 2, height * 3), "#15191e")
            for i, frame in enumerate(subset[start:start+6]):
                image = Image.open(frame["rgb"]).convert("RGB")
                image.thumbnail((width-8, 235))
                resize = np.array(image.size) / frame["size"]
                draw = ImageDraw.Draw(image)
                frame_labels = labels[frame["uid"]]["objects"]
                for side, item in enumerate(frame_labels):
                    xyxy = (np.array(item["rect"]) * np.tile(resize, 2)).tolist()
                    draw.rectangle(xyxy, outline=COLORS[side], width=3)
                    draw.text((xyxy[0]+2, max(0, xyxy[1]-17)), "AB"[side], font=font, fill=COLORS[side],
                              stroke_width=1, stroke_fill="white")
                x, y = (i % 2) * width, (i // 2) * height
                canvas.paste(image, (x+4, y+22))
                draw = ImageDraw.Draw(canvas)
                draw.text((x+6,y+3), f"{platform.upper()} {start+i+1:02d} (A red, B blue)", font=font, fill="white")
                descriptions = ["AB"[s] + " " + item["noun"] + (" [unique]" if item["noun_unique"] else "")
                                for s, item in enumerate(frame_labels)]
                draw.text((x+6,y+263), "     ".join(descriptions), font=font, fill="white")
            canvas.save(folder / f"{platform}_{start//6+1}.jpg", quality=90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "review"))
    args = parser.parse_args()
    if args.stage == "capture":
        capture()
    else:
        review_pages()
