"""Independently recount the tables and verify input, geometry, and semantic sharing."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ARMS = ("H", "S", "GS", "GD")
EPS = 1e-8


def read_file(file):
    return json.loads(file.read_bytes())


def file_digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def table(name):
    with (OUT / (name + ".csv")).open() as stream:
        return list(csv.DictReader(stream))


def scope_matches(r, scope):
    gates = {
        "all": True,
        "recording_separated": r["separation"] == "new_recording",
        "selected": r["selected"] == "True",
        "semantic_changed": r["semantic_status"] in ("category_changed", "subtype_changed")
    }
    return gates.get(scope, r["panel"] == scope)


def verify_public():
    receipt = read_file(OUT / "evaluation_receipt.json")
    assert receipt["code"] == file_digest(HERE / "score_size.py")
    assert receipt["prediction_seal"] == file_digest(OUT / "prediction_seal.json")
    for name, checksum in receipt["files"].items():
        assert file_digest(OUT / (name + ".csv")) == checksum
    rows = table("rows")
    assert len(rows) == len({r["uid"] for r in rows}) == 416
    assert Counter(r["target"] for r in rows) == {"waymo": 96, "drone": 160, "quad": 160}
    for r in rows:
        a, b = (float(r["iou_"+arm]) for arm in ("S", "GS"))
        semantic_equal = r["semantic_status"] == "same"
        assert semantic_equal == (r["reused_size"] == "True")
        if semantic_equal:
            assert a == b and r["center_shift_m_GS_S"] == "0.0"
        for arm in ARMS:
            assert 0. <= float(r["iou_"+arm]) <= 1. + EPS
            if r["status_"+arm] not in ("original", "completed"):
                assert r["iou_"+arm] == r["iou_H"] and r["center_shift_m_"+arm] == "0.0"
    for name in ("metrics", "paired"):
        for m in table(name):
            group = [r for r in rows if scope_matches(r, m["scope"])
                     and (m["target"] == "macro" or m["target"] == r["target"])]
            assert len(group) == int(m["n"]) and len({r["recording"] for r in group}) == int(m["recordings"])
            counts = Counter(r["target"] for r in group)
            weights = [1/(len(counts)*counts[r["target"]]) for r in group]
            a = [float(r["iou_"+m["arm"]]) for r in group]
            if name == "metrics":
                for t, suffix in ((.25, "25"), (.50, "50")):
                    passed = [x > t for x in a]
                    assert sum(passed) == int(m["hits"+suffix])
                    assert abs(100*sum(w*x for w,x in zip(weights, passed)) - float(m["acc"+suffix])) < EPS
                assert abs(100*sum(w*x for w,x in zip(weights, a)) - float(m["miou"])) < EPS
            else:
                b = [float(r["iou_"+m["reference"]]) for r in group]
                for t, suffix in ((.25, "25"), (.50, "50")):
                    assert (sum(x > t >= y for x,y in zip(a,b)), sum(y > t >= x for x,y in zip(a,b))) == (
                        int(m["rescue"+suffix]), int(m["harm"+suffix]))
                    diff = sum(w*(int(x>t)-int(y>t)) for x,y,w in zip(a,b,weights))
                    assert abs(100*diff - float(m["delta_acc"+suffix+"_pp"])) < EPS
                assert abs(100*sum(w*(x-y) for w,x,y in zip(weights,a,b))-float(m["delta_miou_pp"])) < EPS
    return dict(frames=len(rows), public_metrics=len(table("metrics")), public_pairs=len(table("paired")))


def model_coordinates_to_sensor_check(parent, frame, observed, dim):
    import numpy as np
    sensor = parent.sensor_frame_box(observed, frame["target"], frame["suffix"])
    yaw = sensor[6]
    basis = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    longer, shorter = sorted(dim[:2], reverse=True)
    horizontal = (longer, shorter) if observed[3] >= observed[4] else (shorter, longer)
    desired = np.r_[horizontal, dim[2]]
    excess = np.maximum(desired - sensor[3:6], 0.)
    mid = sensor[:2] @ basis.T
    direction = np.where(np.abs(mid)>sensor[3:5]/2, np.sign(mid), 0.)
    expected = sensor.copy()
    expected[:2] += basis.T @ (direction*excess[:2]) / 2
    expected[2] += excess[2]/2
    expected[3:6] = np.maximum(desired, sensor[3:6])
    return expected


def verify_private():
    import numpy as np
    from complete_predictions import checked_normalization, checked_replies, decoded_inputs
    from score_size import checked_predictions
    from study import PRIVATE, crop_photo, digest, geometry_interface, read, roster, verified_observations

    panel, baseline = roster()
    parent = geometry_interface()
    observations, predictions = verified_observations(), checked_predictions()
    model_seal, reply_hashes = checked_replies()
    norm_seal, norm_hashes = checked_normalization()
    rows = {r["uid"]: r for r in table("rows")}
    responses = additional = image_checks = shared = blank = 0
    max_error = 0.
    for frame in panel:
        uid = frame["uid"]
        r = rows[uid]
        pred = predictions[uid]
        box_h = baseline[uid]["boxes"]["H"]
        assert pred["H"]["box"] == box_h
        blank += int(box_h is None)
        obs = observations[uid]
        if not obs["selected"]:
            assert uid not in reply_hashes and all(pred[arm]["box"] == box_h for arm in ARMS)
            continue
        path = PRIVATE / "replies" / (uid + ".json")
        assert digest(path) == reply_hashes[uid]
        raw = read(path)
        labels, sizes, keys, formats = decoded_inputs(
            frame, obs["spans_lwh"], model_seal, reply_hashes[uid], norm_seal, norm_hashes[uid]
        )
        repaired = read(PRIVATE / "resolved" / (uid + ".json"))
        additional += len(repaired["generated"])
        assert formats == {arm: pred[arm]["semantic_format"] for arm in ("S", "GS")}
        image, rect = crop_photo(frame, baseline[uid])
        assert rect == raw["crop"] and hashlib.sha256(image.tobytes()).hexdigest() == raw["image_sha256"]
        image_checks += 1
        responses += len(raw["semantic"]) + len(raw["sizes"])
        for arm in ARMS[1:]:
            entry = pred[arm]
            key = "GD" if arm == "GD" else keys[arm]
            d = sizes.get(key)
            assert entry["semantic"] == labels.get(arm) and entry["size_prior"] == d and entry["request"] == key
            if d is None:
                assert entry["box"] == box_h
                continue
            expected = model_coordinates_to_sensor_check(parent, frame, obs["observed_box_model"], d)
            np.testing.assert_allclose(expected, entry["box"], rtol=0, atol=1.e-9)
            max_error = max(max_error, float(np.max(np.abs(expected - entry["box"]))))
            assert entry["box"][6] == box_h[6]
            assert abs(float(r["center_shift_m_"+arm])-np.linalg.norm(expected[:3]-box_h[:3])) < 1.e-9
        if labels["S"] is not None and labels["S"] == labels["GS"]:
            assert keys["S"] == keys["GS"] and pred["S"]["box"] == pred["GS"]["box"]
            shared += 1
    assert responses == model_seal["calls"] and image_checks == 320 and blank == 5
    assert additional == norm_seal["additional_size_calls"]
    assert shared == read(OUT / "prediction_seal.json")["shared_size_requests"]
    return dict(raw_responses_verified=responses+additional, original_responses=responses, added_size_responses=additional,
                regenerated_crop_hashes=image_checks, identical_semantic_reuses=shared,
                missing_original_boxes=blank, independent_sensor_completion_max_gap_m=max_error, FINAL_access=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--private", action="store_true")
    checked = verify_public()
    if cli.parse_args().private:
        checked.update(verify_private())
        (OUT/"verification.json").write_text(json.dumps(checked, indent=2)+"\n")
    print(checked, flush=True)
