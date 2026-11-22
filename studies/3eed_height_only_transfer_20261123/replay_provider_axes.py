"""Replay the complete sealed provider output under the official scoring convention."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from common import OUT, read_payload, roster, save_payload
from wilddet_eval import RAW_FILE, RAW_SEAL, evaluator_corner_error, provider_pins


def run(previous):
    first_seal = previous / "results" / RAW_SEAL.name
    first_boxes = previous / RAW_FILE.name
    source = roster.read(first_seal)
    assert not source["target_gt_opened"] and not source["FINAL_access"]
    assert source["pins"]["code"] == roster.sha(previous / "wilddet_eval.py")
    assert source["payload"] == roster.sha(first_boxes)
    pins = provider_pins()
    assert all(source["pins"][k] == v for k, v in pins.items() if k != "code")
    pins = dict(source["pins"], **pins)
    repaired = []
    maximum = 0.
    n_boxes = 0
    for row in read_payload(first_boxes):
        assert row["pins"] == source["pins"]
        errors = [0.]
        for arm in ("mono", "depth"):
            if row[arm] is None:
                continue
            old = np.asarray(row[arm], dtype=float)
            # The original archive lost neither the raw dimension values nor the rotation.
            quat = Rotation.from_euler("xyz", old[6:] * np.pi).as_quat()[[3, 0, 1, 2]]
            native = np.r_[old[:6], quat]
            fixed = old.copy()
            fixed[3:6] = old[[4, 5, 3]]
            err = evaluator_corner_error(native, fixed)
            assert err < 1.e-8
            errors.append(err)
            row[arm] = fixed.tolist()
            n_boxes += 1
        maximum = max(maximum, *errors)
        repaired.append(dict(row, pins=pins, worst_corner_error_m=max(errors)))
    assert len(repaired) == source["cases"] and n_boxes == source["filled_rgb"] + source["filled_rgb_lidar"]
    code = source["pins"]["wilddet_code"]
    base = f"https://github.com/allenai/WildDet3D/blob/{code}"
    result = dict(descriptions=len(repaired), reinterpreted_native_boxes=n_boxes, max_official_corner_error_m=maximum,
        native_dimensions="W,L,H", reinterpreted_local_axes="L,H,W", coordinate_system="OpenCV",
        original_model_seal=roster.sha(first_seal), original_prediction_payload=source["payload"],
        original_bridge_code=source["pins"]["code"], corrected_bridge_code=pins["code"],
        evaluator=base+"/wilddet3d/eval/detect3d.py#L410", coder=base+"/wilddet3d/head/coder_3d.py#L52",
        correction_code=roster.sha(__file__), new_model_calls=0, target_labels_used=False, FINAL_access=False)
    result["corrected_prediction_payload"] = save_payload(RAW_FILE, repaired)
    path = OUT / "wilddet3d_axis_replay.json"
    roster.write(path, result)
    roster.write(RAW_SEAL, dict(source, pins=pins, payload=result["corrected_prediction_payload"],
        max_camera_corner_difference_m=maximum, axis_replay=roster.sha(path), dimensions="W,L,H"))
    print(f"{n_boxes} boxes verified against the official evaluator, maximum corner difference {maximum:.3g} m", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    run(parser.parse_args().previous)
