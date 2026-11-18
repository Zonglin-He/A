"""Predict four fixed-query yaw readouts from the authenticated original target A point draw."""
from collections import Counter
from functools import lru_cache
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from time import monotonic

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "research/round155"
OUT = ROOT / "crossplatform/round155"
R131 = ROOT / "crossplatform/round131/formal"
R145 = ROOT / "crossplatform/round145/method"
R154 = ROOT / "crossplatform/round154/predictions"
REG = ROOT / "research/round145/INPUT_REGISTRY_PRIVATE.json"
OLD_REG = ROOT / "research/round131/INPUT_REGISTRY_PRIVATE.json"
PCA_FILE = ROOT / "crossplatform/replication36_surface/common.py"
SURFACE_FILE = ROOT / "crossplatform/surface_dev45/refine.py"
STATES = ["joint_zero", "R145"]
ARMS = ["axis", "pca", "surface", "sign_flip"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return sha(path.read_bytes())


def array_hash(array):
    return sha(np.ascontiguousarray(array).tobytes())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=5)
def load_raw(path, expected):
    path = Path(path)
    b = path.read_bytes()
    require(sha(b) == expected, "raw point payload differs from A sensor receipt")
    value = (np.load(io.BytesIO(b), allow_pickle=False) if path.suffix == ".npy"
             else np.frombuffer(b, dtype=np.float32).reshape(-1, 4))
    require(value.ndim == 2 and value.shape[1] >= 3 and len(value) > 0, "point payload schema")
    return np.asarray(value, np.float32)


def original_points(entry, prefix, original_receipt, transform):
    row, sensor = entry["row"], entry["A_sensor"]
    require(entry["cohort"] == "EXT" and not entry["protected_alias_fallback"] and
            sensor["eligible_for_extra_media"] and not sensor["protected_image_content_alias"] and
            Path(row["pointpath"]).samefile(sensor["pointpath"]) and
            row["platform"] == sensor["platform"] == entry["target"], "ordinary A only")
    raw = load_raw(sensor["pointpath"], sensor["point_sha256"])
    indices = prefix["point_indices"]
    expected = np.random.default_rng(row["seed"]).choice(len(raw), 16384, replace=len(raw)<16384)
    require(indices.shape == (16384,) and indices.dtype == np.int64 and np.array_equal(indices, expected)
            and array_hash(indices) == original_receipt["point_indices_sha256"], "sampled indices mismatch")
    points = raw[indices, :3].copy()
    if row["platform"] == "waymo":
        view = ("F", "FL", "FR", "SL", "SR")[row["suffix"]]
        points, _ = transform(points, np.array([[0., 0., 0., 1., 1., 1., 0.]]), view)
    elif row["platform"] == "drone":
        points[:, 2] += np.float32(1.8)
    else:
        require(row["platform"] == "quad", "unknown platform")
    points = np.ascontiguousarray(points, np.float32)
    require(np.isfinite(points).all() and array_hash(points) == original_receipt["point_sample_sha256"],
            "full 16,384 XYZ sample differs from original forward")
    max_distance = float(cKDTree(points).query(prefix["xyz"][0], k=1)[0].max())
    require(max_distance < 1e-4, "model seed coordinates disagree with physical A sample")
    return points, max_distance


def yaw_readouts(points, candidate_box, pca_func, surface_func):
    # The historical functions first deduplicate then axis-crop. Pre-cropping
    # the same rows leaves exactly the same sorted unique crop, including ties.
    crop = points[np.all(abs(points-candidate_box[:3]) <= candidate_box[3:]/2+1e-7, axis=1)]
    if (candidate_box[3:] <= 0).any():
        axis = np.r_[candidate_box, np.float32(0)]
        return np.stack([axis]*4), False, len(np.unique(crop, axis=0)), 0
    axis, pca, geometry = pca_func(crop, candidate_box)
    surface, choice = surface_func(crop, axis, pca)
    sign_flip = surface.copy()
    sign_flip[6] = -surface[6]  # equivalent modulo pi for a cuboid
    require(geometry["predicted_crop_points"] == choice["crop_points"], "crop disagreement")
    eligible = choice["reason"] == "selected_by_surface_loss"
    require(eligible == (geometry["reason"] == "updated"), "eligibility disagreement")
    return np.stack((axis, pca, surface, sign_flip)), eligible, choice["crop_points"], choice["selected_candidate"]


def main():
    started = monotonic()
    design_path = HERE / "experiment/DESIGN_LOCK.json"
    design = read(design_path)
    require(design["states"] == STATES and design["arms"] == ARMS, "incompatible design")
    for path in (REG, OLD_REG, R145/"SEALED.json", R154/"SEALED.json", PCA_FILE, SURFACE_FILE):
        require(digest(path) == design["inputs"][str(path.relative_to(ROOT))], "design input mismatch")
    pca = load_module("historical_pca", PCA_FILE).refine
    surface = load_module("historical_surface", SURFACE_FILE).refine
    from utils.transform_waymo import transform_to_front_view
    entries, original = read(REG), read(OLD_REG)
    n = len(entries)
    require(n == len(original) == 1798, "cohort mismatch")
    seals = {d.name: read(d/"SEALED.json")["files"] for d in R131.iterdir() if d.is_dir()}
    r145, r154 = read(R145/"SEALED.json")["files"], read(R154/"SEALED.json")["files"]
    boxes = np.empty((n, 2, 4, 7), np.float32)
    eligible = np.empty((n, 2), bool)
    n_crop = np.empty((n, 2), np.int32)
    candidate = np.empty((n, 2), np.int8)
    query = np.empty((n, 2), np.int16)
    counts = Counter()
    max_seed_distance = 0.
    output = OUT/"predictions_PRIVATE.npz"
    seal_path = OUT/"SEAL.json"
    require(not output.exists() and not seal_path.exists(), "prediction or seal exists")
    for i, (e, old) in enumerate(zip(entries, original)):
        row = e["row"]
        src, tgt, sid = e["source"], e["target"], row["id"]
        require((src, tgt, sid) == (old["source"], old["target"], old["row"]["id"]), "registry order")
        direction = src + "_to_" + tgt
        fname = sid + ".npz"
        prefix_rel, record_rel = "prefix_PRIVATE/"+fname, "records_PRIVATE/"+sid+".json"
        parent_rel, bridge_rel = direction+"/samples_PRIVATE/"+fname, direction+"/"+fname
        prefix_path, rec_path = R131/direction/prefix_rel, R131/direction/record_rel
        require(digest(prefix_path) == seals[direction][prefix_rel] and
                digest(rec_path) == seals[direction][record_rel] and
                digest(R145/parent_rel) == r145[parent_rel] and digest(R154/bridge_rel) == r154[bridge_rel],
                "historical artifact mismatch")
        receipt = read(rec_path)["source_prefix"]
        with (np.load(prefix_path,allow_pickle=False) as prefix,
              np.load(R145/parent_rel,allow_pickle=False) as parent,
              np.load(R154/bridge_rel,allow_pickle=False) as previous):
            points, error = original_points(e, prefix, receipt, transform_to_front_view)
            max_seed_distance = max(max_seed_distance, error)
            all_boxes = parent["source_boxes"]
            require(np.array_equal(all_boxes, prefix["boxes"]) and all_boxes.dtype == np.float32,
                    "unchanged physical A256")
            names, old_names = parent["arm_names"].tolist(), previous["arm_names"].tolist()
            result_for_query = {}
            for j, (state, key) in enumerate(zip(STATES, ("norm_joint_zero", "norm_TTA"))):
                arm = names.index(key)
                q = int(np.argmax(parent["scores"][arm]))
                require(q == int(parent["queries"][arm]) and
                        np.array_equal(all_boxes[q], previous["chosen_boxes"][old_names.index(state)]),
                        "historical winner or box changed")
                if q not in result_for_query:
                    result_for_query[q] = yaw_readouts(points, all_boxes[q], pca, surface)
                boxes[i,j], eligible[i,j], n_crop[i,j], candidate[i,j] = result_for_query[q]
                require(np.all(boxes[i,j,:,:6] == all_boxes[q]), "changed center or dimensions")
                query[i,j] = q
        counts[direction] += 1
        if (i+1) % 200 == 0:
            print("R155 verified point samples and selected boxes", i+1, flush=True)
    require(len(counts) == 6 and np.isfinite(boxes).all(), "incomplete predictions")
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("xb") as f:
        np.savez_compressed(f, boxes=boxes, states=STATES, arms=ARMS, reader_names=STATES, mode_names=ARMS,
                            eligible=eligible, n_crop=n_crop, candidate=candidate, query=query,
                            has_B=np.array([e["has_B"] for e in entries], bool), ordinal=np.arange(1,n+1))
    os.replace(tmp, output)
    signature = digest(output)
    seal = dict(n=n, sha256=signature, files={output.name:signature}, design_sha256=digest(design_path),
                code_sha256=digest(Path(__file__)), directions=dict(counts), states=STATES, arms=ARMS,
                original_xyz_hash_matches=n, original_indices_hash_matches=n, original_winners_checked=2*n,
                max_seed_reconstruction_error_m=max_seed_distance, new_network_forwards=0,
                source_training=False, GT_read=False, FINAL_access=False, seconds=monotonic()-started)
    tmp = seal_path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(seal, indent=2, allow_nan=False)+"\n")
    os.replace(tmp, seal_path)
    print("R155 predictions sealed", n, signature, flush=True)


if __name__ == "__main__":
    main()
