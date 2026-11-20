"""Freeze derived questions and evaluation keys before any HiSpatial prediction."""
from collections import Counter, defaultdict
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from common import HERE, PRIVATE, OUT, PROTOCOL, PLATFORMS, SIDE, digest, key, read, write
from geometry import calibration, transform_xyz

GEOMETRY_WORDS = re.compile(
    r"\b(left|right|closest|farthest|nearer|farther|nearest|taller|shorter|distance|meters?|"
    r"beside|behind|ahead|center|middle|foreground|background|upper|lower|frontmost|rearmost)\b",
    re.I,
)


def loc(cell):
    x, y = (min(1023, round(int(v) * 1024 / SIDE)) for v in cell)
    return f"<loc{y:04}><loc{x:04}><loc{y:04}><loc{x:04}>"


def sight_pair(frame):
    with np.load(PRIVATE / "observations" / (frame["uid"] + ".npz"), allow_pickle=False) as f:
        cells, points = f["cells"], f["camera"]
        viable = np.flatnonzero(
            f["hold"] & ~f["collision"] & ((12 <= cells) & (cells < SIDE - 12)).all(1))
    rng = np.random.default_rng(int(key(frame["uid"], "probe"), 16) % 2**63)
    ordered = rng.permutation(viable)
    for a in ordered[:min(400, len(ordered))]:
        difference = abs(points[ordered, 2] - points[a, 2])
        minimum = np.maximum(1., .2 * np.minimum(points[ordered, 2], points[a, 2]))
        separates = np.max(abs(cells[ordered] - cells[a]), axis=1) >= 64
        candidates = ordered[(difference > minimum) & separates]
        if len(candidates):
            b = int(candidates[0])
            return [(int(a), cells[a], points[a]), (b, cells[b], points[b])]
    return None


def lateral_axes(transform):
    right = transform[0, :2].copy()
    right /= np.linalg.norm(right)
    forward = np.array([-right[1], right[0]])
    optical = transform[2, :2]
    assert optical @ forward / np.linalg.norm(optical) > .95
    # Right axis when yawing 90 degrees toward camera right or left.
    return right, -forward, forward


def choice(in_a_first):
    return "A" if bool(in_a_first) else "B"


def render_oracle(frame, label, path):
    with Image.open(frame["rgb"]) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(round(max(image.size) / SIDE * 3.0), 3)
    for obj, color in zip(label["objects"], ("#ff2222", "#2255ff")):
        draw.rectangle(obj["rect"], outline=color, width=width)
    image.save(path, compress_level=2)


def run():
    manifest = read(OUT / "capture_seal.json")
    assert manifest["capture"] == digest(PRIVATE / "captured.json")
    assert manifest["labels"] == digest(PRIVATE / "candidate_truth.json")
    assert manifest["protocol"] == digest(PROTOCOL)
    frames = read(PRIVATE / "captured.json")
    labels = read(PRIVATE / "candidate_truth.json")
    review = read(PRIVATE / "visual_identity_review.json")
    assert set(review) == set(labels) == {f["uid"] for f in frames}
    question_list, truth = [], {}
    quotas = defaultdict(Counter)
    images = PRIVATE / "oracle_images"
    images.mkdir(mode=0o700, exist_ok=True)
    for frame in frames:
        uid, target = frame["uid"], frame["target"]
        label = labels[uid]
        approved = review[uid]
        assert len(approved) == 2
        for value in approved:
            desc = value["description"]
            if desc is not None:
                assert value["visible"] and not GEOMETRY_WORDS.search(desc) and not re.search(r"\d", desc)
        base = {k: frame[k] for k in ("uid","target","recording")}
        group = {}
        def add(mode, task, level, prompt, value, **gold_fields):
            qid = key(uid, mode, task)
            assert qid not in truth
            question_list.append(dict(qid=qid, **base, mode=mode, task=task, level=level, prompt=prompt))
            truth[qid] = dict(value=value, **gold_fields)
            if mode == "language":
                group[task] = qid
            quotas[target][mode + "_" + level] += 1

        pair = sight_pair(frame)
        if pair is not None:
            (a, xy_a, xyz_a), (b, xy_b, xyz_b) = pair
            loc_a, loc_b = loc(xy_a), loc(xy_b)
            add("language", "pixel_xyz", "L0", f"what is the xyz of the pixel in {loc_a}",
                [float(v) for v in xyz_a], lidar_rows=[a])
            text = f"Pixel A is {loc_a}. Pixel B is {loc_b}. Give only A or B. Which pixel has "
            near_a = xyz_a[2] < xyz_b[2]
            add("language", "pixel_near", "L0", text + "smaller depth along the camera viewing axis?",
                choice(near_a), lidar_rows=[a,b])
            add("language", "pixel_far", "L0", text + "larger depth along the camera viewing axis?",
                choice(not near_a), lidar_rows=[a,b])
        else:
            quotas[target]["no_reliable_two_pixels"] += 1

        _, _, transform = calibration(frame)
        boxes = np.asarray([r["box"] for r in label["objects"]], float)
        centers = transform_xyz(boxes[:, :3], transform)
        center_ranges = np.linalg.norm(centers, axis=1)
        offset = boxes[0, :2] - boxes[1, :2]
        static, turns_right, turns_left = [float(axis @ offset) for axis in lateral_axes(transform)]
        entity = [r["description"] for r in approved]
        for i, name in enumerate(entity):
            if name:
                add("language","distance_"+"ab"[i],"L1",
                    f"What is the straight-line distance from the camera center to the center of the {name}? Answer with a single number in meters.",
                    float(center_ranges[i]), objects=[i])
                add("language","height_"+"ab"[i],"L1",
                    f"What is the physical height of the {name}, from its bottom to its top? Answer with a single number in meters.",
                    float(boxes[i,5]), objects=[i])
        fully_named = all(entity)
        if fully_named:
            assert entity[0] != entity[1]
            sentence = f"Object A is the {entity[0]}. Object B is the {entity[1]}. "
            if abs(np.diff(center_ranges)[0]) > .75:
                common = sentence + "Use the straight-line distance from the camera to the object centers. Answer with A or B only. "
                add("language", "object_near", "L2", common + "Which of the two objects is closer?",
                    choice(center_ranges[0] < center_ranges[1]), objects=[0,1])
                add("language", "object_far", "L2", common + "Which of the two objects is farther away?",
                    choice(center_ranges[0] > center_ranges[1]), objects=[0,1])
            spacing = float(np.linalg.norm(offset))
            for suffix, start, end in [("ab","A","B"),("ba","B","A")]:
                add("language","separation_"+suffix,"L2",sentence +
                    f"What is the horizontal distance from the center of object {start} to the center of object {end}? Answer with a single number in meters.",
                    spacing,objects=[0,1])
        else:
            quotas[target]["unverified_text_pair"] += 1
        for mode in ("language","oracle_identity"):
            if mode == "language":
                if not fully_named:
                    continue
                start = sentence
            else:
                if not all(x["visible"] for x in approved):
                    quotas[target]["unverified_oracle_pair"] += 1
                    continue
                path = images / (uid + ".png")
                render_oracle(frame, label, path)
                frame["oracle_rgb"] = str(path)
                frame["oracle_sha"] = digest(path)
                start = "Object A is outlined in red. Object B is outlined in blue. "
            suffix = ("Compare object centers projected onto the horizontal right axis. "
                      "Compare both even if one would be behind you. Answer with A or B only.")
            if abs(static) > .5:
                add(mode, "static_right", "L2", start+"From the original camera direction, which object is farther to the right? "+suffix,
                    choice(static > 0), objects=[0,1])
            if abs(turns_right) <= .75:
                continue
            for sign, lateral in (("right",turns_right),("left",turns_left)):
                prompt = (start + f"Imagine turning 90 degrees to your {sign} from the original camera direction "
                          "without moving your location or either object. Which object is farther to your right after the turn? " + suffix)
                add(mode, "after_turn_"+sign, "L3", prompt, choice(lateral > 0), objects=[0,1],
                    static_available=abs(static)>.5, changes_static=(lateral > 0.) != (static > 0.))
        quotas[target]["reviewed_visible_pair"] += int(all(x["visible"] for x in approved))
        quotas[target]["reviewed_text_pair"] += int(fully_named)

    # The inference manifest contains images, immutable prompts and opaque IDs
    # only. Answers and annotated boxes remain in independent private files.
    write(PRIVATE / "frames.json", frames)
    write(PRIVATE / "questions.json", question_list)
    write(PRIVATE / "answers.json", truth)
    assert len(question_list) == len(truth)
    assert all(any(q["target"] == t and q["level"] == level and q["mode"] == "language"
                   for q in question_list) for t in PLATFORMS for level in ("L0","L1","L2","L3"))
    summary = dict(
        protocol=digest(PROTOCOL), author=digest(__file__), review=digest(PRIVATE / "visual_identity_review.json"),
        frames=digest(PRIVATE / "frames.json"), questions=digest(PRIVATE / "questions.json"),
        answers=digest(PRIVATE / "answers.json"), ground_objects=manifest["labels"],
        frames_count=len(frames), question_count=len(question_list),
        quotas=dict(quotas), recordings={t:len({f["recording"] for f in frames if f["target"] == t}) for t in PLATFORMS},
        FINAL_access=False, predictions_consulted=False,
    )
    write(OUT / "panel_seal.json", summary)
    print(summary["frames_count"], summary["question_count"], summary["quotas"])


if __name__ == "__main__":
    run()
