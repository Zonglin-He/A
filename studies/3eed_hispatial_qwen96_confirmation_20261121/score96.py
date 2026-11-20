"""Score sealed HiSpatial outputs with distinct ordinary and oracle geometry."""
from collections import defaultdict

import numpy as np

from shared import ARMS, HERE, OUT, PRIVATE, PROTOCOL, digest, read, write
from geometry import calibration, make_xyz, transform_xyz, yaw_rotation
from maps import map_arm
from score import (FAMILY, changed, consistency_rows, csv_write, describe_error, outcome, parse,
                   pointmap_readout)


def measure(frame, labels):
    uid = frame["uid"]
    with np.load(PRIVATE / "observations" / (uid + ".npz"), allow_pickle=False) as obs:
        cells, xyz, collision = obs["cells"], obs["camera"], obs["collision"]
    _, _, transform = calibration(frame)
    points = transform_xyz(xyz, np.linalg.inv(transform))
    inside = []
    for item in labels["objects"]:
        box = np.asarray(item["box"], float)
        inside.append((abs((points - box[:3]) @ yaw_rotation(box[6])) <= box[3:6] / 2. + .05).all(1))
    results, per_object, estimates = {}, {}, {}
    with (np.load(PRIVATE / "maps" / (uid + ".npz"), allow_pickle=False) as maps,
          np.load(PRIVATE / "oracle_depth" / (uid + ".npz"), allow_pickle=False) as oracle):
        held = maps["held"]
        depth = xyz[:, 2]
        mask_group = {
            "all": held, "lt10": held & (depth < 10), "10to20": held & (depth >= 10) & (depth < 20),
            "20to40": held & (depth >= 20) & (depth < 40), "ge40": held & (depth >= 40),
            "collision": held & collision, "no_collision": held & ~collision,
        }
        y, x = cells[:, 1], cells[:, 0]
        for arm in ARMS:
            field = (make_xyz(oracle["depth"], maps["mask"], maps["rays"]) if arm == "oracle_depth"
                     else map_arm(maps, arm))
            at_cells = field[:3, y, x].T.astype(float)
            valid = field[3, y, x] == 1
            assert valid[held].all()
            estimates[arm] = at_cells, valid
            results[arm] = {group: describe_error(at_cells, xyz, keep) for group, keep in mask_group.items()}
            per_object[arm] = [describe_error(at_cells, xyz, held & keep) for keep in inside]
        assert results["oracle_depth"]["all"]["depth_mae"] in (None, 0.)
    return results, per_object, estimates


def deterministic_turn(frame, labels, task):
    _, _, matrix = calibration(frame)
    right = matrix[0, :2] / np.linalg.norm(matrix[0, :2])
    forward = np.array([-right[1], right[0]])
    origin = (-matrix[:3, :3].T @ matrix[:3, 3])[:2]
    ground = np.asarray([item["box"][:2] for item in labels["objects"]]) - origin
    centers = np.round(ground @ np.column_stack((right, forward)), 2)
    angle = np.pi/2 if task == "after_turn_right" else -np.pi/2
    points_right = centers @ [np.cos(angle), -np.sin(angle)]
    assert abs(points_right[0]-points_right[1]) > 0.
    return "A" if points_right[0] > points_right[1] else "B"


def run():
    panel, seal, analysis = (read(OUT / (name + ".json"))
                             for name in ("panel_seal", "prediction_seal", "analysis_seal"))
    for field, filename in (("answers", "answers"), ("ground_objects", "candidate_truth"),
                            ("questions", "questions"), ("frames", "frames")):
        assert panel[field] == digest(PRIVATE / (filename + ".json"))
    assert panel["protocol"] == digest(PROTOCOL)
    assert seal["stamp"]["panel"] == digest(OUT / "panel_seal.json")
    assert seal["stamp"]["analysis"] == digest(OUT / "analysis_seal.json")
    assert seal["answers_hashes"] == digest(PRIVATE / "answer_hashes.json")
    assert analysis["files"]["score96.py"] == digest(__file__)
    assert analysis["plan"] == digest(HERE / "ANALYSIS_PLAN.json")
    hashes, maps, oracle = (read(PRIVATE / (name + ".json"))
                             for name in ("answer_hashes", "map_hashes", "oracle_hashes"))
    keys = read(PRIVATE / "answers.json")
    labels = read(PRIVATE / "candidate_truth.json")
    frames = read(PRIVATE / "frames.json")
    questions = defaultdict(list)
    for q in read(PRIVATE / "questions.json"):
        questions[q["uid"]].append(q)
    qa, geometry, injected, logic, coords = [], [], [], [], []
    for frame in frames:
        uid = frame["uid"]
        main = PRIVATE / "answers_frozen" / (uid + ".json")
        for file, expected in ((main, hashes[uid]),
                               (PRIVATE / "maps" / (uid + ".npz"), maps[uid]),
                               (PRIVATE / "oracle_depth" / (uid + ".npz"), oracle[uid]),
                               (PRIVATE / "observations" / (uid + ".npz"), frame["observation_sha"])):
            assert digest(file) == expected
        answers = read(main)
        assert answers["stamp"] == seal["stamp"]
        text = {(entry["qid"], entry["arm"]): entry["answer"] for entry in answers["answers"]}
        assert len(text) == len(answers["answers"])
        outputs = {k: parse(value, keys[k[0]]["value"]) for k, value in text.items()}
        expected = {(q["qid"], arm) for q in questions[uid] for arm in ARMS}
        expected |= {(q["qid"], "oracle_centers") for q in questions[uid] if q["level"] == "L3"}
        assert set(expected) == set(outputs)

        baseline, local, map_values = measure(frame, labels[uid])
        record = {k: frame[k] for k in ("uid", "target", "recording")}
        for arm, by_group in baseline.items():
            rows = [dict(**record, arm=arm, group=group, **metrics) for group, metrics in by_group.items()]
            if arm == "oracle_depth":
                injected.extend(rows)
            else:
                geometry.extend(rows)
        logic.extend(consistency_rows(questions[uid], outputs, keys, "language", frame))
        for q in questions[uid]:
            qid = q["qid"]
            gold = keys[qid]
            meta = {k: v for k, v in q.items() if k != "prompt"}
            for arm in ARMS:
                value = outputs[qid, arm]
                evidence = [local[arm][i] for i in gold.get("objects", [])]
                reference = [local["moge"][i] for i in gold.get("objects", [])]
                n = min((x["n"] for x in evidence), default=0)
                improved = n >= 3 and all(x["depth_mae"] < y["depth_mae"] - 1e-5
                                         for x, y in zip(evidence, reference))
                direct = (outcome(pointmap_readout(q, gold, map_values[arm]), gold["value"])
                          if q["level"] == "L0" else {})
                qa.append(dict(**meta, family=FAMILY[q["task"]], arm=arm, **outcome(value, gold["value"]),
                    raw_change=int(text[qid, arm] != text[qid, "moge"]),
                    answer_change=int(changed(value, outputs[qid, "moge"])),
                    min_object_holdout=n, all_objects_depth_improved=int(improved),
                    object_depth_mae=float(np.mean([o["depth_mae"] for o in evidence])) if n >= 3 else None,
                    global_depth_mae=baseline[arm]["all"]["depth_mae"] if arm != "oracle_depth" else None,
                    direct_correct=direct.get("correct"), direct_valid=direct.get("valid"),
                    direct_depth_error=direct.get("abs_error"), direct_point_error=direct.get("point_error"),
                    static_changes=int(gold.get("changes_static", False)),
                    static_available=int(gold.get("static_available", False))))
            if q["level"] == "L3":
                deterministic = deterministic_turn(frame, labels[uid], q["task"])
                assert deterministic == gold["value"]
                coords.append(dict(**meta,
                    direct=int(outcome(outputs[qid, "prior_depth"], gold["value"])["correct"]),
                    **outcome(outputs[qid, "oracle_centers"], gold["value"]),
                    deterministic=int(deterministic == gold["value"])))
    assert len(qa) == seal["main_answers"] and len(coords) == seal["coordinate_oracle_answers"]
    groups = {"qa_rows.csv": qa, "geometry_rows.csv": geometry, "oracle_anchor_rows.csv": injected,
              "consistency_rows.csv": logic, "coordinate_qa_rows.csv": coords}
    for filename, rows in groups.items():
        csv_write(OUT / filename, rows)
    write(OUT / "score_receipt.json", dict(frames=len(frames), total_answers=len(qa)+len(coords),
          all_answer_rows_scored=True, input_predictions=digest(OUT / "prediction_seal.json"),
          files={name: digest(OUT / name) for name in groups},
          ordinary_labels_used_only_for_scoring=True, FINAL_access=False))


if __name__ == "__main__":
    run()
