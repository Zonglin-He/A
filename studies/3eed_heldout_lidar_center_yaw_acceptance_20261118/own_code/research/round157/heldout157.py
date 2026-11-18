"""Replay both candidate draws and obtain unused unique returns and their ray origins."""
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/round156"))
from pair156 import CAP, PINS, R131, REG, array_hash, digest, read, require, shared_points
from predict155 import load_module
transform_to_front_view = load_module("r157_view_transform", ROOT / "crossplatform/upstream/utils/transform_waymo.py").transform_to_front_view

ANCHORS = ROOT / "research/round131/ANCHORS_PRIVATE.json"
CALIBRATION = ROOT / "crossplatform/round157/M3ED_reference_calibration_PRIVATE.json"
ANCHOR_HASH = "73359180050ebe5f0eb2398681e145b56c0dd2d0c4c6d916d2dae5c93d6efcbf"
CALIBRATION_HASH = "be4ffaef1041c3659f6088ab634beb46055a27d0702a5149f7b306b413b88906"


def unused_returns(xyz, a_indices, d_indices):
    """Remove original-index union, coincident used XYZ and heldout multiplicity."""
    used = np.union1d(a_indices, d_indices)
    _, first, inverse = np.unique(xyz, axis=0, return_index=True, return_inverse=True)
    touched = np.zeros(len(first), dtype=bool)
    touched[inverse.ravel()[used]] = True
    rows = first[~touched]
    n_raw_holdout = len(xyz) - len(used)
    return rows, len(used), n_raw_holdout - len(rows)


@lru_cache(maxsize=64)
def camera_extrinsic(path):
    # Exported metadata also contains annotations. Decode the one calibration field only.
    match = re.search(rb'"image_extrinsic"\s*:\s*', data := Path(path).read_bytes())
    require(match is not None, "missing local camera-lidar calibration field")
    matrix = json.JSONDecoder().raw_decode(data[match.end():].decode())[0]
    result = np.asarray(matrix, dtype=float)
    require(result.shape == (4, 4), "calibration matrix shape")
    return result


class BeamTable:
    """Native Ouster encoder grid in the sensor frame. All distances in metres."""
    def __init__(self, metadata):
        transform = np.asarray(metadata["lidar_to_sensor_transform"], dtype=float).reshape(4, 4)
        rotation = transform[:3, :3]
        self.center = transform[:3, 3] * 0.001
        self.radius = float(metadata["lidar_origin_to_beam_origin_mm"]) * 0.001
        self.width = int(metadata["data_format"]["columns_per_frame"])
        elevation = np.deg2rad(metadata["beam_altitude_angles"])[:, None]
        offset = np.deg2rad(metadata["beam_azimuth_angles"])[:, None]
        encoder = -np.arange(self.width) * (2 * np.pi / self.width)
        azimuth = encoder[None, :] - offset
        elevation = np.broadcast_to(elevation, azimuth.shape)
        directions = np.stack((np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth),
                               np.sin(elevation)), axis=-1).reshape(-1, 3)
        axes = np.column_stack((np.cos(encoder), np.sin(encoder), np.zeros(self.width)))
        self.directions = directions @ rotation.T
        self.origins = self.center + self.radius * (axes @ rotation.T)
        self.tree = cKDTree(self.directions)

    def resolve(self, model_points, platform):
        model_shift = np.array([0., 0., 1.8 if platform == "drone" else 0.])
        sensor_points = model_points.astype(float) - model_shift
        vector = sensor_points - self.center
        length = np.linalg.norm(vector, axis=1)
        direction = vector / np.maximum(length[:, None], 1e-12)
        chord, grid = self.tree.query(direction, k=9)
        idx = grid[:, :8]
        residual_vector = sensor_points[:, None, :] - self.origins[idx % self.width]
        beams = self.directions[idx]
        along = np.sum(residual_vector * beams, axis=2)
        residual = np.linalg.norm(residual_vector - along[:, :, None]*beams, axis=2)
        residual[along <= 0] = np.inf
        nearest = np.argmin(residual, axis=1)
        row = np.arange(len(sensor_points))
        min_error = residual[row, nearest]
        runner_up = np.partition(residual, 1, axis=1)[:, 1]
        # Lower bound for the distance to every unsearched forward half-line,
        # allowing its emitter to move anywhere within the calibrated radius.
        theta = 2*np.arcsin(np.minimum(1., chord[:, 8]/2))
        unseen_bound = length*np.sin(np.minimum(theta, np.pi/2)) - self.radius
        ok = (min_error <= 1e-4) & (runner_up > 2e-4) & (unseen_bound > 2e-4)
        verified = self.origins[idx[row[ok], nearest[ok]] % self.width] + model_shift
        return verified, ok


def iter_heldout():
    require(digest(REG) == PINS[REG] and digest(ANCHORS) == ANCHOR_HASH, "ordinary registry/anchor hash")
    require(digest(CALIBRATION) == CALIBRATION_HASH, "reference calibration changed")
    require(digest(CAP / "SEALED_PRIVATE.json") == PINS[CAP / "SEALED_PRIVATE.json"], "D seal changed")
    captures = read(CAP / "SEALED_PRIVATE.json")["files"]
    records = {d.name: read(d / "SEALED.json")["files"] for d in R131.iterdir() if d.is_dir()}
    pairs = {a["A"]["frame_PRIVATE"]: a["B_pair"] for a in read(ANCHORS)}
    calibration = read(CALIBRATION)
    tables = {scene: BeamTable(c["metadata"]) for scene, c in calibration.items()}
    rows = read(REG)
    require(len(rows) == 1798, "unexpected manifest size")
    for ordinal, e in enumerate(rows, 1):
        row, sensor = e["row"], e["A_sensor"]
        platform = e["target"]
        direction = e["source"] + "_to_" + platform
        key = direction + "/" + row["id"] + ".npz"
        receipt_key = "records_PRIVATE/" + row["id"] + ".json"
        path = R131 / direction / receipt_key
        require(digest(path) == records[direction][receipt_key] and digest(CAP/key) == captures[key],
                "input receipt mismatch")
        receipt = read(path)["source_prefix"]
        _, xyz = shared_points(e, receipt, transform_to_front_view)
        require(np.isfinite(xyz).all(), "non-finite candidate point cloud")
        seed = row["seed"]
        alt_seed = int(sha256(f"R154|{seed}".encode()).hexdigest()[:16], 16)
        a_idx = np.random.default_rng(seed).choice(len(xyz), 16384, replace=len(xyz)<16384)
        d_idx = np.random.default_rng(alt_seed).choice(len(xyz), 16384, replace=len(xyz)<16384)
        with np.load(CAP/key, allow_pickle=False) as alternate:
            require(int(alternate["seed"]) == alt_seed and int(alternate["original_seed"]) == seed
                    and alternate["raw_point_sha256"].item() == sensor["point_sha256"]
                    and alternate["original_indices_sha256"].item() == receipt["point_indices_sha256"] == array_hash(a_idx)
                    and alternate["new_indices_sha256"].item() == array_hash(d_idx)
                    and alternate["new_sample_xyz_sha256"].item() == array_hash(xyz[d_idx]), "D/A support replay")
        retained, n_used, aliases = unused_returns(xyz, a_idx, d_idx)
        points = xyz[retained]
        strict_count = len(points)
        evidence = e.get("pair_PRIVATE")
        registered = bool(e["has_B"])
        attempted = pairs[sensor["frame_PRIVATE"]] is not None
        require(not registered or (attempted and evidence["B"]["point_sha256"] != sensor["point_sha256"]),
                "unsupported B or same raw point file")
        exposure = "accepted_B" if registered else ("registration_tried_no_B" if attempted else "no_registration")
        origin = None
        unresolved = 0
        reason = "unknown_waymo_emitter"
        if platform in ("drone", "quad"):
            scene = row["scene"]
            require(np.max(np.abs(camera_extrinsic(row["metapath"])-calibration[scene]["T_cam_lidar"]))<1e-9,
                    "3EED exported extrinsic differs from primary sequence calibration")
            origin, ok = tables[scene].resolve(points, platform)
            points = points[ok]
            unresolved = strict_count - len(points)
            reason = "calibrated_ouster_per_beam"
        yield points, origin, dict(ordinal=ordinal, source=e["source"], target=platform, has_B=registered,
                                   n_raw=len(xyz), n_unique_sample_indices=n_used, n_holdout_indices=len(xyz)-n_used,
                                   alias_exclusions=aliases, n_strict_unique=strict_count, n_holdout=len(points),
                                   n_origin_unresolved=unresolved, origin_reason=reason, strict_e2e=not attempted,
                                   registration_exposure=exposure)
