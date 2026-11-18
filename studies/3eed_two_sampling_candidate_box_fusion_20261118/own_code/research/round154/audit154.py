"""Rebuild R154's box correspondences, decisions and proposal pools from sealed inputs."""

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PARENT = ROOT / "crossplatform/round145/method"
CAPTURES = ROOT / "crossplatform/round154/captures"
PREDICTIONS = ROOT / "crossplatform/round154/predictions"
ARMS = (
    "frozen_A", "joint_zero", "direct_R145", "R145", "wrong_R145", "alternate_single",
    "geometry_zero", "geometry_TTA", "anchored_score_TTA", "union_representative_TTA",
    "fusion_zero", "fusion_TTA", "shuffled_fusion_TTA",
)
POOLS = ("original_A", "alternate", "geometry", "fusion_zero", "fusion_TTA")
PARENT_ARMS = ("Frozen_A", "norm_joint_zero", "norm_direct", "norm_TTA", "norm_wrong_TTA")


def require(test, detail):
    if not test:
        raise ValueError(detail)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(directory, relative, expected_hash):
    path = directory / relative
    require(path.resolve().is_relative_to(directory.resolve()), f"Path escapes {directory}")
    require(digest(path) == expected_hash, f"Seal hash mismatch for {relative}")


def correspondence(anchors, other):
    # Generate only intersecting pairs before computing IoU. Sorting Python tuples
    # fixes tie order explicitly and does not depend on NumPy's sorting default.
    a_min = anchors[:, :3] - anchors[:, 3:] * 0.5
    a_max = anchors[:, :3] + anchors[:, 3:] * 0.5
    b_min = other[:, :3] - other[:, 3:] * 0.5
    b_max = other[:, :3] + other[:, 3:] * 0.5
    a_valid = np.all(anchors[:, 3:] > 0, axis=1)
    b_valid = np.all(other[:, 3:] > 0, axis=1)
    intersects = a_valid[:, None] & b_valid[None, :]
    for axis in range(3):
        intersects &= a_min[:, axis, None] < b_max[None, :, axis]
        intersects &= a_max[:, axis, None] > b_min[None, :, axis]
    ia, ib = np.nonzero(intersects)
    extent = np.minimum(a_max[ia], b_max[ib]) - np.maximum(a_min[ia], b_min[ib])
    int_volume = extent.prod(axis=1)
    union = anchors[ia, 3:].prod(axis=1) + other[ib, 3:].prod(axis=1) - int_volume
    overlaps = int_volume / union
    take = overlaps > 0.55
    edges = sorted(zip(-overlaps[take], ia[take], ib[take]))
    mapping = np.full(256, -1, dtype=np.int64)
    used = set()
    for _, a_slot, b_slot in edges:
        if mapping[a_slot] < 0 and b_slot not in used:
            mapping[a_slot] = b_slot
            used.add(b_slot)
    secondary = np.array(sorted(set(range(256)) - used), dtype=np.int64)
    return mapping, secondary


def reference(anchors, other, original_scores, alt_scores, source, target, identifier):
    zero, adapted = original_scores[1], original_scores[3]
    mapping, secondary = correspondence(anchors, other)
    matches = np.flatnonzero(mapping >= 0)
    source_boxes = np.concatenate([anchors, other[secondary]])
    fused = source_boxes.copy()
    if len(matches):
        b_slots = mapping[matches]
        summed = zero[matches] + alt_scores[b_slots]
        numer = zero[matches, None] * anchors[matches] + alt_scores[b_slots, None] * other[b_slots]
        fused[matches] = np.divide(numer, summed[:, None], where=summed[:, None] > 0,
                                   out=.5 * (anchors[matches] + other[b_slots]))

    def group_order(a_scores, b_scores):
        scores = np.concatenate([a_scores, b_scores[secondary]]) * .5
        scores[matches] += .5 * b_scores[mapping[matches]]
        return np.lexsort((np.arange(len(scores)), -scores))

    by_zero = group_order(zero, alt_scores)
    by_tta = group_order(adapted, alt_scores)
    seed = int.from_bytes(hashlib.sha256(
        f"R154-shuffle|{source}|{target}|{identifier}".encode()
    ).digest()[:8], "little")
    permuted = alt_scores[np.random.default_rng(seed).permutation(256)]
    by_wrong = group_order(adapted, permuted)
    a_winner = by_tta[by_tta < 256][0]
    choices = [anchors[np.argmax(p)] for p in original_scores]
    choices.extend([
        other[np.argmax(alt_scores)], fused[np.argmax(zero)], fused[np.argmax(adapted)],
        anchors[a_winner], source_boxes[by_tta[0]], fused[by_zero[0]],
        fused[by_tta[0]], fused[by_wrong[0]],
    ])
    pools = [anchors, other, fused[:256], fused[by_zero[:256]], fused[by_tta[:256]]]
    group_info = {"fusion_TTA_winner_group": by_tta[0], "fusion_zero_winner_group": by_zero[0],
                  "n_matched": len(matches), "n_unmatched_alt_in_TTA_pool": np.sum(by_tta[:256] >= 256)}
    return mapping, np.stack(choices), np.stack(pools), group_info


def check_scores(values, identifier):
    require(values.shape == (256,), f"{identifier}: score vector shape")
    require(np.isfinite(values).all() and np.all(values >= 0), f"{identifier}: invalid score")
    require(abs(float(values.sum()) - 1) < 1e-5, f"{identifier}: probability normalization")


def compare_boxes(expected, actual, identifier):
    require(expected.shape == actual.shape, f"{identifier}: box array shape")
    require(np.isfinite(actual).all(), f"{identifier}: nonfinite boxes")
    max_error = float(np.max(np.abs(expected - actual)))
    require(np.allclose(expected, actual, atol=1e-10, rtol=2e-13), f"{identifier}: box error {max_error}")
    return max_error


def main():
    registry_file = ROOT / "research/round145/INPUT_REGISTRY_PRIVATE.json"
    lock = read_json(HERE / "experiment/DESIGN_LOCK.json")
    require(lock["arm_names"] == list(ARMS) and lock["pool_names"] == list(POOLS), "Design arm order")
    for path in [registry_file, PARENT / "SEALED.json"]:
        require(digest(path) == lock["inputs"][str(path.relative_to(ROOT))], "Design input hash")
    registry = read_json(registry_file)
    parent_seal = read_json(PARENT / "SEALED.json")
    capture_seal = read_json(CAPTURES / "SEALED_PRIVATE.json")
    seal = read_json(PREDICTIONS / "SEALED.json")
    require(seal["design_sha256"] == digest(HERE / "experiment/DESIGN_LOCK.json"), "Prediction design pin")
    require(seal["capture_seal_sha256"] == digest(CAPTURES / "SEALED_PRIVATE.json"), "Prediction capture pin")
    require(seal["code_sha256"] == digest(HERE / "predict154.py"), "Prediction code pin")
    bridge = read_json(HERE / "CAPTURE_BRIDGE_PRIVATE.json")
    require(bridge["status"] == "PASS" and bridge["pins"] == capture_seal["pins"], "Source bridge pins")
    require(bridge["sources"] == capture_seal["sources"], "Source bridge states")
    require(capture_seal["pins"]["registry_sha256"] == digest(registry_file), "Capture registry pin")
    require(capture_seal["pins"]["r145_seal_sha256"] == digest(PARENT / "SEALED.json"), "Capture parent pin")
    require(capture_seal["pins"]["capture_sha256"] == digest(HERE / "capture154.py"), "Capture code pin")
    require(list(seal["arm_names"]) == list(ARMS) and list(seal["pool_names"]) == list(POOLS),
            "Unexpected sealed endpoints")
    require(seal["n"] == len(registry) == 1798, "Cohort size")
    seen = set()
    counts = Counter()
    matches = 0
    max_chosen_error = max_pool_error = 0.
    for entry in registry:
        source, target, identifier = entry["source"], entry["target"], entry["row"]["id"]
        direction = f"{source}_to_{target}"
        filename = f"{direction}/{identifier}.npz"
        parent_filename = f"{direction}/samples_PRIVATE/{identifier}.npz"
        require(filename not in seen, "Duplicate direction-input")
        seen.add(filename)
        verify_file(PARENT, parent_filename, parent_seal["files"][parent_filename])
        verify_file(CAPTURES, filename, capture_seal["files"][filename])
        verify_file(PREDICTIONS, filename, seal["files"][filename])
        with np.load(PARENT / parent_filename, allow_pickle=False) as parent, \
             np.load(CAPTURES / filename, allow_pickle=False) as capture, \
             np.load(PREDICTIONS / filename, allow_pickle=False) as prediction:
            original_seed = entry["row"]["seed"]
            expected_seed = int.from_bytes(hashlib.sha256(f"R154|{original_seed}".encode()).digest()[:8], "big")
            require(int(capture["seed"].item()) == expected_seed, "Alternate sampling seed")
            require(int(capture["original_seed"].item()) == original_seed, "Original sampling seed")
            require(capture["new_indices_sha256"].item() != capture["original_indices_sha256"].item(),
                    "Resampling produced original indices")
            identifiers = {
                "row_id": identifier, "direction": direction,
                "raw_point_sha256": entry["A_sensor"]["point_sha256"],
                "raw_image_sha256": entry["A_sensor"]["image_sha256"],
                "capture_sha256": capture_seal["pins"]["capture_sha256"],
                "prepared_state_sha256": bridge["sources"][source]["prepared_state_sha256"],
            }
            require(all(capture[k].item() == v for k, v in identifiers.items()), "Capture provenance")
            anchors = np.asarray(parent["source_boxes"], dtype=np.float64)
            alternate = np.asarray(capture["new_boxes"], dtype=np.float64)
            require(anchors.shape == alternate.shape == (256, 6), "Invalid box shape")
            require(np.isfinite(anchors).all() and np.isfinite(alternate).all(), "Invalid box input")
            arm_names = list(parent["arm_names"])
            require(len(arm_names) == len(set(arm_names)), "Ambiguous parent arms")
            parent_p = np.asarray(parent["scores"], dtype=np.float64)
            original_scores = [parent_p[arm_names.index(arm)] for arm in PARENT_ARMS]
            alt_scores = np.asarray(capture["new_p"], dtype=np.float64)
            for scores in [*original_scores, alt_scores]:
                check_scores(scores, filename)
            m, boxes, pools, group_info = reference(anchors, alternate, original_scores, alt_scores,
                                                    source, target, identifier)
            require(np.array_equal(m, prediction["A_to_alt"]), f"{filename}: wrong association")
            require(all(prediction[k].item() == v for k, v in group_info.items()), "Group identity mismatch")
            require(list(prediction["arm_names"]) == list(ARMS), "Row arm names")
            require(list(prediction["pool_names"]) == list(POOLS), "Row pool names")
            require(bool(prediction["has_B"].item()) == bool(entry["has_B"]), "B coverage changed")
            max_chosen_error = max(max_chosen_error, compare_boxes(boxes, prediction["chosen_boxes"], filename))
            max_pool_error = max(max_pool_error, compare_boxes(pools, prediction["pools"], filename))
            matches += int(np.sum(m >= 0))
            counts[direction] += 1
    require({f for f in seal["files"] if f.endswith(".npz")} == seen, "Extra or missing predictions")
    require({f for f in capture_seal["files"] if f.endswith(".npz")} == seen, "Extra or missing alternate captures")
    result = dict(status="PASS", n=len(seen), directions=dict(sorted(counts.items())), matched_pairs=matches,
                  arm_names=ARMS, pool_names=POOLS, max_chosen_box_error=max_chosen_error,
                  max_pool_box_error=max_pool_error, seal_sha256=digest(PREDICTIONS / "SEALED.json"),
                  alternate_seal_sha256=digest(CAPTURES / "SEALED_PRIVATE.json"),
                  parent_seal_sha256=digest(PARENT / "SEALED.json"),
                  capture_forward_independently_reexecuted=False, GT_read=False, FINAL_access=False,
                  utc=datetime.now(timezone.utc).isoformat())
    path = HERE / "PREDICTION_AUDIT.json"
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
