"""Localize the registered-versus-live v3 differences within its frozen fallback."""
import csv
import os
from collections import Counter
from functools import lru_cache
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from inputs import HERE, OUT, PLATFORMS, PRIVATE, digest, load_panel, read, write
from crossplatform.ground_regions import Grounder, regions, sensor_frame_box

TREE = HERE.parent / "3eed_v1_physical_tree_role_decoding_20261120"
MARGIN = HERE.parent / "3eed_v1_subject_margin_attribution_20261120" / "run_PRIVATE"
R183 = HERE.parent / "3eed_v2_ordinary_val_subject_ranking_20261119" / "predictions_PRIVATE"


@lru_cache(None)
def terrain_archive(direction):
    folder = TREE / "private/ordinary/predictions"
    pinned = read(folder / "SEALED.json")
    directory = folder / direction
    seal = directory / "SEALED.json"
    path = directory / "all.npz"
    assert digest(seal) == pinned["dirs"][direction] and digest(path) == read(seal)["all"]
    i = read(TREE / "PROTOCOL.json")["controls"].index("shared_cut")
    with np.load(path, allow_pickle=False) as p:
        return p["registered"], p["measured"][:, i], p["completed"][:, i]


@lru_cache(None)
def margin_archive(direction, block):
    path = MARGIN / direction / (f"{128*block:05d}.npz")
    seal = MARGIN / direction / "SEALED.json"
    assert digest(seal) == read(MARGIN / "SEALED.json")["directions"][direction]
    assert digest(path) == read(seal)["files"][path.name]
    r183_path = R183 / direction / path.name
    assert digest(r183_path) == read(R183 / "SEALED.json")["directions"][direction]["files"][path.name]["sha256"]
    with np.load(path, allow_pickle=False) as old, np.load(r183_path, allow_pickle=False) as orig:
        take = old["slots"][np.arange(len(old["slots"])), old["choices"][:, 0]]
        boxes = old["pool"][np.arange(len(take)), take]
        return old["ordinals"], boxes, old["choices"][:, 0], orig["four"]


def fit_box(box, frame):
    return sensor_frame_box(box, frame["target"], frame["suffix"])


def score_seed(arm, fallback_box, old_box, old_choice, current_choices, old_four, new_four):
    arm["anchor_coords_equal"] = int(np.allclose(fallback_box, old_box, rtol=0., atol=1e-10))
    arm["same_slot_choice"] = int(old_choice == current_choices)
    arm["same_four_boxes"] = int(np.allclose(old_four, new_four, rtol=0., atol=1e-10))
    arm["anchor_max_xy_diff_m"] = float(abs(fallback_box[:2]-old_box[:2]).max())
    arm["anchor_max_size_diff_m"] = float(abs(fallback_box[3:6]-old_box[3:6]).max())


@torch.inference_mode()
def run():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    panel = load_panel()
    with (OUT / "source_pairs.csv").open(newline="") as f:
        score = {(r["source"], r["uid"]): r for r in csv.DictReader(f)}
    summaries = []
    for source in PLATFORMS:
        engine = Grounder(source)
        for frame in panel:
            if frame["target"] == source:
                continue
            uid, target, ix = (frame[k] for k in ("uid", "target", "ordinal"))
            row = score[source, uid]
            if int(row["entered"]):
                continue
            trace = {}
            old = regions.only_ground

            def capture_decoder(_m, args, reply):
                assert "choice" not in trace
                trace["choice"] = int(reply["choices"][0, 0])
                trace["four"] = args[0]["subject_boxes"][0].cpu().numpy().copy()

            def capture_bottom(anchor, lower):
                trace["anchor"] = anchor.copy()
                return old(anchor, lower)

            handle = engine._decoder.register_forward_hook(capture_decoder)
            try:
                with patch.object(regions, "only_ground", side_effect=capture_bottom):
                    replay = engine.predict(**{k: frame[k] for k in
                        ("target", "points", "rgb", "calibration", "text", "seed", "waymo_view")})
            finally:
                handle.remove()
            reference = read(PRIVATE / ("v3_" + source + ".json"))["rows"][uid]["box"]
            np.testing.assert_array_equal(replay, reference)
            recorded, before, after = terrain_archive(source + "_to_" + target)
            assert not recorded[ix]
            ordinals, boxes, original_choice, old_four = margin_archive(source + "_to_" + target, ix//128)
            k, = np.flatnonzero(ordinals == ix)
            np.testing.assert_array_equal(boxes[k], before[ix])
            live_anchor = fit_box(trace["anchor"], frame)
            live_four = np.stack([fit_box(b, frame) for b in trace["four"]])
            r = dict(source=source, target=target, uid=uid, iou_difference=float(row["B_v3"])-float(row["B_archived_v3"]),
                archived_floor_changed=int(not np.array_equal(before[ix], after[ix])),
                live_floor_changed=int(not np.array_equal(live_anchor,replay)),
                archived_floor_effect=float(after[ix, 5]-before[ix, 5]),
                live_floor_effect=float(replay[5]-live_anchor[5]),
                archived_live_output_equal=int(np.allclose(after[ix], replay, rtol=0., atol=1e-10)))
            score_seed(r, live_anchor, before[ix], original_choice[k], trace["choice"], old_four[k], live_four)
            summaries.append(r)
        del engine
        torch.cuda.empty_cache()
    assert len(summaries) == 184
    with (OUT / "v3_fallback_rows.csv").open("w", newline="") as f:
        output = csv.DictWriter(f, fieldnames=list(summaries[0]), lineterminator="\n")
        output.writeheader()
        output.writerows(summaries)
    discrepant = [r for r in summaries if abs(r["iou_difference"]) > 1e-8]
    counts = {tag: dict(Counter(r[tag] for r in discrepant)) for tag in
              ("anchor_coords_equal", "same_slot_choice", "same_four_boxes",
               "archived_floor_changed", "live_floor_changed", "archived_live_output_equal")}
    write(OUT / "v3_fallback_audit.json", dict(fallbacks=len(summaries), score_disagreements=len(discrepant),
        code=digest(__file__), trace_reproduced_live_boxes_exactly=True, grouped_disagreements=counts,
        all_anchor_coordinates_equal=sum(r["anchor_coords_equal"] for r in summaries),
        all_matching_fours=sum(r["same_four_boxes"] for r in summaries),
        all_matching_slot_choices=sum(r["same_slot_choice"] for r in summaries), FINAL_access=False))
    print(counts, flush=True)


if __name__ == "__main__":
    run()
