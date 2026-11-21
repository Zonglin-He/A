"""Recover the original measured box from the sealed region, before any GT scoring."""
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from study import (BASE, HERE, OUT, PRIVATE, SPEC, complete, digest, geometry_interface,
                   read, roster, write)


def run():
    started = perf_counter()
    panel, baseline = roster()
    parent = geometry_interface()
    measured_by_uid = {}
    selected_per_platform = Counter()
    max_h_gap = 0.
    max_raw_span_gap = 0.
    ground_extension_frames = 0
    h_bigger_than_raw_height = 0
    for frame in panel:
        uid = frame["uid"]
        old = baseline[uid]
        if not old["selected"]:
            measured_by_uid[uid] = {"selected": False, "status": old["status"]}
            continue
        assert digest(frame["points"]) == frame["point_sha"]
        sample = parent.model_sample(frame)
        _, assignments, births, _ = parent.partition(sample)
        node = old["node"]
        support = sample[assignments[births[node]] == node]
        raw = np.asarray(old["support_raw"], dtype=float)
        np.testing.assert_array_equal(parent.to_raw_support(support, frame["target"], frame["suffix"]), raw)
        assert len(support) == old["region_points"]
        observed = parent.regions.measured_box(support, parent.regions.terrain(sample))
        original_size = old["boxes"]["A"][3:6]
        reconstructed = complete(parent, frame, observed, original_size)
        np.testing.assert_allclose(reconstructed, old["boxes"]["H"], atol=1e-9, rtol=0.)
        max_h_gap = max(max_h_gap, float(abs(reconstructed - old["boxes"]["H"]).max()))

        yaw = observed[6]
        axes = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        spans_xy = np.ptp((support[:, :2] - support[:, :2].mean(0)) @ axes.T, axis=0)
        height = float(np.ptp(support[:, 2]))
        sensor_yaw = old["boxes"]["H"][6]
        sensor_axes = np.array([[np.cos(sensor_yaw), np.sin(sensor_yaw)],
                                [-np.sin(sensor_yaw), np.cos(sensor_yaw)]])
        independent = np.ptp((raw[:, :2] - raw[:, :2].mean(0)) @ sensor_axes.T, axis=0)
        max_raw_span_gap = max(max_raw_span_gap, float(abs(spans_xy - independent).max()))
        np.testing.assert_allclose(np.maximum(spans_xy, 1e-6), observed[3:5], atol=1e-9, rtol=0.)
        assert observed[5] + 1e-9 >= height
        spans = [*sorted(spans_xy.tolist(), reverse=True), height]
        ground_extension_frames += int(observed[5] - height > 1e-6)
        h_bigger_than_raw_height += int(reconstructed[5] - height > 1e-6)
        selected_per_platform[frame["target"]] += 1
        measured_by_uid[uid] = {"selected": True, "status": old["status"], "observed_box_model": observed.tolist(),
                                "spans_lwh": spans, "support_points": len(support), "H_replay_gap_m": float(abs(reconstructed-old["boxes"]["H"]).max())}
    assert len(measured_by_uid) == 416 and sum(selected_per_platform.values()) == 320
    assert max_raw_span_gap < 1e-8
    path = PRIVATE / "observations.json"
    write(path, measured_by_uid)
    write(OUT / "input_seal.json", {
        "parent_prediction_seal": digest(BASE / "results/prediction_seal.json"),
        "protocol": digest(SPEC),
        "observations": digest(path),
        "code": {n: digest(HERE / n) for n in ("study.py", "prepare.py")},
        "frames": len(panel),
        "selected_by_platform": dict(selected_per_platform),
        "parents_reproduced": sum(selected_per_platform.values()),
        "ground_extended_measurement_frames": ground_extension_frames,
        "old_complete_height_exceeds_raw_span_frames": h_bigger_than_raw_height,
        "H_max_replay_gap": max_h_gap,
        "sensor_model_span_max_error_m": max_raw_span_gap,
        "seconds": perf_counter() - started,
        "target_labels_read": False,
        "FINAL_access": False
    })
    print("Prepared", len(panel), "frames and replayed", sum(selected_per_platform.values()), "selected boxes.",
          "Maximum H discrepancy:", max_h_gap, flush=True)


if __name__ == "__main__":
    run()
