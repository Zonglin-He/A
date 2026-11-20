"""Freeze the shared original-image questions and the explicit-coordinate probes."""
from collections import Counter
import re

import numpy as np

from shared import OUT, PRIVATE, PROTOCOL, PLATFORMS, digest, key, read, write
from geometry import calibration, transform_xyz
from questions import GEOMETRY_WORDS, choice, lateral_axes, loc, sight_pair


def run():
    seal = read(OUT / "capture_seal.json")
    assert seal["protocol"] == digest(PROTOCOL)
    assert seal["capture"] == digest(PRIVATE / "captured.json")
    assert seal["labels"] == digest(PRIVATE / "candidate_truth.json")
    frames = read(PRIVATE / "captured.json")
    labels = read(PRIVATE / "candidate_truth.json")
    visual = read(PRIVATE / "identity_review.json")
    assert all(len(visual[t]) == 32 for t in PLATFORMS)
    prompts, gold, coordinates = [], {}, []
    coverage = {t: Counter() for t in PLATFORMS}
    for frame in frames:
        uid, target = frame["uid"], frame["target"]
        objects = labels[uid]["objects"]
        names = visual[target][frame["panel_index"] - 1]
        assert len(names) == 2 and all(name is None for name in names[len(objects):])
        assert all(n is None or (not GEOMETRY_WORDS.search(n) and not re.search(r"\d", n)) for n in names)
        quotas = coverage[target]
        quotas["frames"] += 1
        quotas["named_object_count_" + str(sum(n is not None for n in names))] += 1
        metadata = {k: frame[k] for k in ("uid", "target", "recording")}

        def add(task, level, prompt, value, **extra):
            qid = key(uid, task)
            assert qid not in gold
            prompts.append(dict(qid=qid, **metadata, mode="language", task=task, level=level, prompt=prompt))
            gold[qid] = dict(value=value, **extra)
            quotas[level] += 1
            return qid

        sight = sight_pair(frame)
        if sight is not None:
            (a, cell_a, xyz_a), (b, cell_b, xyz_b) = sight
            loc_a, loc_b = map(loc, (cell_a, cell_b))
            add("pixel_xyz", "L0", f"what is the xyz of the pixel in {loc_a}",
                xyz_a.tolist(), lidar_rows=[a])
            common = f"Pixel A is {loc_a}. Pixel B is {loc_b}. Give only A or B. Which pixel has "
            first = xyz_a[2] < xyz_b[2]
            add("pixel_near", "L0", common + "smaller depth along the camera viewing axis?",
                choice(first), lidar_rows=[a, b])
            add("pixel_far", "L0", common + "larger depth along the camera viewing axis?",
                choice(not first), lidar_rows=[a, b])
        else:
            quotas["no_eligible_L0_pair"] += 1

        if not any(names):
            continue
        _, _, transform = calibration(frame)
        boxes = np.asarray([obj["box"] for obj in objects], float)
        ranges = np.linalg.norm(transform_xyz(boxes[:, :3], transform), axis=1)
        for i, name in enumerate(names):
            if name is None:
                continue
            suffix = "ab"[i]
            add("distance_" + suffix, "L1",
                f"What is the straight-line distance from the camera center to the center of the {name}? "
                "Answer with a single number in meters.", float(ranges[i]), objects=[i])
            add("height_" + suffix, "L1", f"What is the physical height of the {name}, from its bottom to its top? "
                "Answer with a single number in meters.", float(boxes[i, 5]), objects=[i])
        if not all(names):
            continue
        assert names[0] != names[1]
        prefix = f"Object A is the {names[0]}. Object B is the {names[1]}. "
        if abs(ranges[0] - ranges[1]) > .75:
            common = prefix + "Use the straight-line distance from the camera to the object centers. Answer with A or B only. "
            add("object_near", "L2", common + "Which of the two objects is closer?",
                choice(ranges[0] < ranges[1]), objects=[0, 1])
            add("object_far", "L2", common + "Which of the two objects is farther away?",
                choice(ranges[0] > ranges[1]), objects=[0, 1])
        offset = boxes[0, :2] - boxes[1, :2]
        distance = float(np.linalg.norm(offset))
        for start, end in (("A", "B"), ("B", "A")):
            add("separation_" + (start + end).lower(), "L2", prefix +
                f"What is the horizontal distance from the center of object {start} to the center of object {end}? "
                "Answer with a single number in meters.", distance, objects=[0, 1])

        right, turn_right, turn_left = lateral_axes(transform)
        static = float(right @ offset)
        turn_values = (float(turn_right @ offset), float(turn_left @ offset))
        suffix = ("Compare object centers projected onto the horizontal right axis. "
                  "Compare both even if one would be behind you. Answer with A or B only.")
        if abs(static) > .5:
            add("static_right", "L2", prefix + "From the original camera direction, which object is farther to the right? "
                + suffix, choice(static > 0), objects=[0, 1])
        if abs(turn_values[0]) <= .75:
            quotas["ambiguous_turn_margin"] += 1
            continue
        origin = -transform[:3, :3].T @ transform[:3, 3]
        centers = (boxes[:, :2] - origin[:2]) @ np.column_stack((right, turn_left))
        rounded = np.round(centers, 2)
        assert np.sign(rounded[0, 1] - rounded[1, 1]) == np.sign(turn_values[1])
        a, b = rounded
        provided = (" The original camera's ground-plane coordinates for the centers, in (rightward, forward) meters, "
                    f"are A=({a[0]:.2f}, {a[1]:.2f}) and B=({b[0]:.2f}, {b[1]:.2f}).")
        for side, lateral in zip(("right", "left"), turn_values):
            prompt = (prefix + f"Imagine turning 90 degrees to your {side} from the original camera direction "
                      "without moving your location or either object. Which object is farther to your right after the turn? "
                      + suffix)
            qid = add("after_turn_" + side, "L3", prompt, choice(lateral > 0), objects=[0, 1],
                      static_available=abs(static) > .5, changes_static=(lateral > 0) != (static > 0))
            coordinates.append(dict(qid=qid, prompt=prompt + provided))

    assert len(coordinates) == sum(q["level"] == "L3" for q in prompts)
    assert len(prompts) == len(gold)
    assert all(all(coverage[t][level] for level in ("L0", "L1", "L2", "L3")) for t in PLATFORMS)
    for name, value in (("frames", frames), ("questions", prompts), ("answers", gold), ("coordinates", coordinates)):
        write(PRIVATE / (name + ".json"), value)
    panel = dict(protocol=digest(PROTOCOL), questions=digest(PRIVATE / "questions.json"),
                 frames=digest(PRIVATE / "frames.json"), answers=digest(PRIVATE / "answers.json"),
                 coordinate_prompts=digest(PRIVATE / "coordinates.json"),
                 ground_objects=seal["labels"], visual_review=digest(PRIVATE / "identity_review.json"),
                 capture=digest(OUT / "capture_seal.json"), frames_count=len(frames), question_count=len(prompts),
                 coordinate_count=len(coordinates), quotas=coverage, recordings=seal["recordings"],
                 FINAL_access=False, predictions_consulted=False)
    write(OUT / "panel_seal.json", panel)
    print("Frozen", len(frames), len(prompts), len(coordinates), coverage, flush=True)


if __name__ == "__main__":
    run()
