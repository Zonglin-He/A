"""Independently replay label-free R155 endpoints and sampled local surface decisions."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FORMAL = ROOT / "crossplatform/round155"
R131 = ROOT / "crossplatform/round131/formal"
R145 = ROOT / "crossplatform/round145/method"
R154 = ROOT / "crossplatform/round154/predictions"
DESIGN = HERE / "experiment/DESIGN_LOCK.json"
DESIGN_HASH = "8a29714843f1fba5c06449fa30cbc023f617fd7322a3b4d9a4e9c79a6dc12b71"
STATES = ["joint_zero", "R145"]
ARMS = ["axis", "pca", "surface", "sign_flip"]
MODEL_STATES = ["norm_joint_zero", "norm_TTA"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def input_points(entry):
    """Recover original model points from native raw measurements and saved seed."""
    row = entry["row"]
    raw_path = Path(row["pointpath"])
    require(digest(raw_path) == entry["A_sensor"]["point_sha256"], "Raw cloud content differs")
    if raw_path.suffix == ".npy":
        raw = np.load(raw_path, allow_pickle=False)
    else:
        require(raw_path.suffix == ".bin", "Unknown cloud format")
        raw = np.fromfile(raw_path, dtype=np.float32).reshape(-1, 4)
    ii = np.random.default_rng(row["seed"]).choice(len(raw), 16384, replace=len(raw) < 16384)
    p = np.asarray(raw[ii, :3], dtype=np.float32).copy()
    if entry["target"] == "drone":
        p[:, 2] += np.float32(1.8)
    elif entry["target"] == "waymo":
        angle = np.deg2rad([0., -45., 45., -90., 90.][row["suffix"]])
        co, si = np.cos(angle), np.sin(angle)
        rotation = np.array([[co, -si, 0], [si, co, 0], [0, 0, 1]])
        p = (p @ rotation.T).astype(np.float32)
    else:
        require(entry["target"] == "quad", "Unknown frame convention")
    key = row["id"]
    direction = entry["source"] + "_to_" + entry["target"]
    rec = read(R131 / direction / "records_PRIVATE" / (key + ".json"))["source_prefix"]
    require(array_digest(p) == rec["point_sample_sha256"], "Wrong native-frame sampling transform")
    require(array_digest(ii.astype(np.int64)) == rec["point_indices_sha256"], "Wrong input point indices")
    return np.unique(p, axis=0)


def wrap(x):
    return (x + math.pi / 2) % math.pi - math.pi / 2


def box_surface(points, box, yaw):
    # For an interior point, choose the nearest face. For an exterior point use
    # the Euclidean distance to the closest point on the cuboid.
    d = points.astype(np.float64) - box[:3]
    co, si = math.cos(yaw), math.sin(yaw)
    projected = np.column_stack((co*d[:, 0] + si*d[:, 1], -si*d[:, 0] + co*d[:, 1], d[:, 2]))
    half_extent = box[3:6].astype(np.float64)/2
    closest = np.clip(projected, -half_extent, half_extent)
    distances = np.linalg.norm(projected - closest, axis=1)
    interior = np.all(np.abs(projected) <= half_extent, axis=1)
    distances[interior] = np.min(half_extent - np.abs(projected[interior]), axis=1)
    return float(np.mean(distances)/min(box[3:5]))


def physical_replay(points, box):
    center, extent = box[:3], box[3:6]
    kept = points[np.all(np.abs(points - center) <= extent / 2 + 1e-7, axis=1)]
    count = len(kept)
    if np.any(extent <= 0) or count < 8 or extent[0]/extent[1] < 1.5:
        return count, False, 0, (0., 0., 0.)
    xy = kept[:, :2].astype(np.float64)
    xy -= xy.mean(axis=0)
    covariance = xy.T @ xy / (len(xy) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[-1] <= 1e-12:
        return count, False, 0, (0., 0., 0.)
    planar_angle = np.float32(wrap(math.atan2(eigenvectors[1, -1], eigenvectors[0, -1])))
    candidates = (0., float(planar_angle), float(np.float32(wrap(float(planar_angle) + math.pi / 2))))
    costs = [box_surface(kept, box, a) for a in candidates]
    chosen = next(i for i, cost in enumerate(costs) if cost <= min(costs) + 1e-12)
    return count, True, chosen, (candidates[1], candidates[chosen], wrap(-candidates[chosen]))


def main():
    require(digest(DESIGN) == DESIGN_HASH, "Immutable design changed")
    lock = read(DESIGN)
    require(lock["states"] == STATES and lock["arms"] == ARMS, "Design ordering changed")
    regfile = ROOT / "research/round145/INPUT_REGISTRY_PRIVATE.json"
    registry = read(regfile)
    for path in (regfile, R145 / "SEALED.json", R154 / "SEALED.json"):
        require(digest(path) == lock["inputs"][str(path.relative_to(ROOT))], "Changed parent seal")
    parent = read(R145 / "SEALED.json")["files"]
    preceding = read(R154 / "SEALED.json")["files"]
    seal = read(FORMAL / "SEAL.json")
    npz = FORMAL / "predictions_PRIVATE.npz"
    pred_sha = digest(npz)
    require(seal["sha256"] == pred_sha and seal["design_sha256"] == DESIGN_HASH, "R155 seal mismatch")
    with np.load(npz, allow_pickle=False) as f:
        for key, names in (("states", STATES), ("arms", ARMS)):
            require(f[key].tolist() == names, "Endpoint name/order mismatch")
        boxes, eligible, ncrop, choice, query = [f[x] for x in ("boxes", "eligible", "n_crop", "candidate", "query")]
        has_b, ordinal = f["has_B"], f["ordinal"]
    n = len(registry)
    require(n == 1798 and boxes.shape == (n, 2, 4, 7) and boxes.dtype == np.float32, "Endpoint count/dtype")
    require(all(x.shape == (n, 2) for x in (eligible, ncrop, choice, query)), "Metadata shapes")
    require(eligible.dtype == np.bool_ and has_b.dtype == np.bool_, "Boolean types")
    require(has_b.shape == (n,) and np.array_equal(ordinal, np.arange(1, n+1)), "Input count/order")
    require(all(np.issubdtype(x.dtype, np.integer) for x in (ncrop, choice, query)), "Indices must be integral")
    require(np.isfinite(boxes).all() and np.all((0 <= ncrop) & (ncrop <= 16384)), "Numerical box/crop range")
    require(np.all(np.isin(choice, [0, 1, 2])) and np.all((0 <= query) & (query < 256)), "Metadata bounds")
    yaw = boxes[:, :, :, 6].astype(np.float64)
    require(np.all(yaw[:, :, 0] == 0) and np.max(np.abs(yaw)) <= np.pi/2 + 1e-6, "Yaw units/range")
    require(np.all(choice[~eligible] == 0) and np.all(yaw[~eligible] == 0), "Ineligible fallback")
    require(np.all(ncrop[eligible] >= 8), "Insufficient support")
    require(np.max(np.abs(wrap(yaw[:, :, 2] + yaw[:, :, 3]))) < 3e-7, "Signed control geometry")
    defined = np.where(choice == 0, 0., np.where(choice == 1, yaw[:, :, 1], yaw[:, :, 1] + np.pi/2))
    require(np.max(np.abs(wrap(yaw[:, :, 2]-defined))) < 3e-7, "Surface mode/index mismatch")
    counts = Counter()
    replayed, reasons, physical_checks, max_angle_error = set(), set(), 0, 0.
    for i, e in enumerate(registry):
        require(e["cohort"] == "EXT" and not e["protected_alias_fallback"], "Nonordinary input")
        require(e["A_sensor"]["eligible_for_extra_media"] and not e["A_sensor"]["protected_image_content_alias"],
                "Protected alias")
        require(bool(e["has_B"]) == has_b[i], "Original physical B status differs")
        direction = e["source"] + "_to_" + e["target"]
        counts[direction] += 1
        name = e["row"]["id"] + ".npz"
        parent_key = direction + "/samples_PRIVATE/" + name
        last_key = direction + "/" + name
        require(digest(R145 / parent_key) == parent[parent_key], "R145 sample differs")
        require(digest(R154 / last_key) == preceding[last_key], "R154 sample differs")
        with np.load(R145 / parent_key, allow_pickle=False) as old:
            states = old["arm_names"].tolist()
            indices = np.array([old["scores"][states.index(s)].argmax() for s in MODEL_STATES])
            original = old["source_boxes"][indices]
        with np.load(R154 / last_key, allow_pickle=False) as last:
            arms = last["arm_names"].tolist()
            require(np.array_equal(last["chosen_boxes"][[arms.index(s) for s in STATES]], original), "R154 endpoints differ")
        require(np.array_equal(query[i], indices), "Selected query differs")
        require(np.array_equal(boxes[i, :, :, :6], np.repeat(original[:, None, :], 4, axis=1)), "Center/size changed")
        # Predeclared witnesses: first, first eligible, and first non-axis endpoint per direction,
        # before scoring. Recompute both readouts for every such input.
        flags = ["first"]
        if eligible[i].any():
            flags.append("first_eligible")
        if np.any(np.abs(yaw[i, :, 2]) > 1e-7):
            flags.append("first_rotated")
        new_flags = {(direction, f) for f in flags} - reasons
        if new_flags:
            reasons.update(new_flags)
            points = input_points(e)
            replayed.add(i)
            for j in range(len(STATES)):
                count, enabled, winner, yaws = physical_replay(points, original[j])
                require(ncrop[i,j] == count and eligible[i,j] == enabled, "Independent local points/eligibility differ")
                require(choice[i,j] == winner, "Independent minimum cuboid-surface distance differs")
                err = float(np.max(np.abs(wrap(yaw[i,j,1:] - np.asarray(yaws)))))
                max_angle_error = max(max_angle_error, err)
                require(err < 3e-7, "Independent orientation differs")
                physical_checks += 1
    require(len(counts) == 6 and sum(counts.values()) == n, "Missing transfer directions")
    receipt = dict(status="PASS", prediction_sha256=pred_sha, design_sha256=DESIGN_HASH,
                   endpoint_and_query_checks=n*2, center_size_and_axis_checks=n*2*4,
                   directions=dict(counts), physical_replay_rows=len(replayed), physical_replay_endpoints=physical_checks,
                   replay_witnesses={d: sorted(k for a,k in reasons if a == d) for d in counts},
                   max_replayed_yaw_error_radians=max_angle_error, eligible_endpoints=int(eligible.sum()),
                   updated_surface_yaw_endpoints=int(np.count_nonzero(np.abs(yaw[:, :, 2]) > 1e-7)),
                   target_GT_opened=False, FINAL_opened=False, predictor_imported=False)
    out = HERE / "PREDICTION_AUDIT.json"
    with out.open("x") as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
        f.write("\n")
    print("Independent R155 prediction audit passed", n, flush=True)


if __name__ == "__main__":
    main()
