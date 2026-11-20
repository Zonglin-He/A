"""Checked camera/LiDAR alignment, block holdout, and dense map coordinates."""
from itertools import product
import sys

import cv2
import numpy as np
from PIL import Image

from common import ROOT, SIDE

sys.path.insert(0, str(ROOT if (ROOT / "crossplatform").is_dir() else ROOT / "v1"))
from crossplatform.round7.camera_projection import (
    CameraSpec,
    WAYMO_ANGLES,
    model_to_camera_matrix,
    project_model_points,
    raw_to_model_points,
    read_camera_allowlist,
    z_rotation,
)


def transform_xyz(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def calibration(row):
    spec = CameraSpec.from_camera_fields(
        row["target"], read_camera_allowlist(row["calibration"]),
        waymo_view_suffix=row["suffix"], missing_distortion_policy="assume_zero_declared",
    )
    with Image.open(row["rgb"]) as image:
        size = np.asarray(image.size)
    raw_to_model = np.eye(4)
    if row["target"] == "waymo":
        raw_to_model[:3, :3] = z_rotation(WAYMO_ANGLES[row["suffix"]])
    if row["target"] == "drone":
        raw_to_model[2, 3] = 1.8
    return spec, size, model_to_camera_matrix(spec) @ raw_to_model


def pixel_rays(pixels, spec):
    intrinsic, _, distortion, _ = spec.validated()
    uv = np.asarray(pixels, np.float64).reshape(-1, 2)
    if np.any(distortion):
        xy = cv2.undistortPoints(uv[:, None], intrinsic, distortion)[:, 0]
        return np.column_stack((xy, np.ones(len(xy))))
    return np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(intrinsic).T


def project_camera(points, spec):
    intrinsic, _, distortion, _ = spec.validated()
    if np.any(distortion):
        return cv2.projectPoints(points, np.zeros(3), np.zeros(3), intrinsic, distortion)[0][:, 0]
    homogeneous = points @ intrinsic.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def calibrated_grid(spec, size):
    # OpenCV nearest-neighbor resizing in the official MoGe processor picks
    # floor(i * original_size / resized_size) in each image dimension.
    x = np.floor(np.arange(SIDE) * size[0] / SIDE)
    y = np.floor(np.arange(SIDE) * size[1] / SIDE)
    pixels = np.stack(np.meshgrid(x, y), axis=-1)
    rays = pixel_rays(pixels, spec).reshape(SIDE, SIDE, 3)
    discrepancy = np.max(np.abs(project_camera(rays.reshape(-1, 3), spec) - pixels.reshape(-1, 2)))
    if discrepancy >= .05:
        raise ValueError(f"Unprojection error in original pixels: {discrepancy}")
    return rays.astype(np.float32), float(discrepancy)


def nearest_points(uv, depths, width, height, image_size):
    cell = np.floor(uv * [width, height] / image_size).astype(np.int64)
    valid = ((0 <= cell) & (cell < [width, height])).all(axis=1)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return np.empty((0, 2), dtype=np.int64), indices
    flat = cell[indices, 1] * width + cell[indices, 0]
    order = np.lexsort((indices, depths[indices], flat))
    flat = flat[order]
    first = np.r_[True, flat[1:] != flat[:-1]]
    winner = indices[order[first]]
    return cell[winner], winner


def observations(row, spec, size, raw_to_camera):
    points = row["points"]
    raw = np.load(points, allow_pickle=False) if points.endswith(".npy") else np.fromfile(points, "<f4").reshape(-1, 4)
    raw = raw[:, :3].astype(np.float64)
    raw = raw[np.isfinite(raw).all(1)]
    model = raw_to_model_points(raw, row["target"], row["suffix"])
    project = project_model_points(model, spec, size, min_depth=.25)
    camera = transform_xyz(raw, raw_to_camera)
    np.testing.assert_allclose(camera[:, 2], project["depth"], atol=2e-8, rtol=1e-10)
    valid = project["in_image"] & (camera[:, 2] <= 250.)
    camera, uv = camera[valid], project["uv"][valid]
    cell, selected = nearest_points(uv, camera[:, 2], SIDE, SIDE, size)

    # Quantify near/far competition in the camera footprint before z-buffering.
    flat = np.floor(uv * SIDE / size).astype(np.int64)
    flat = flat[:, 1] * SIDE + flat[:, 0]
    counts = np.bincount(flat, minlength=SIDE ** 2)
    min_z = np.full(SIDE ** 2, np.inf)
    max_z = np.full(SIDE ** 2, -np.inf)
    np.minimum.at(min_z, flat, camera[:, 2])
    np.maximum.at(max_z, flat, camera[:, 2])
    occupied = counts > 0
    conflict = np.zeros_like(occupied)
    conflict[occupied] = max_z[occupied] - min_z[occupied] > np.maximum(1.0, .15 * min_z[occupied])
    assert len(cell) == int(occupied.sum())

    bx, by = (cell // 8).T
    salt = int(row["uid"][:8], 16)
    heldout = ((bx * 73856093) ^ (by * 19349663) ^ salt) % 5 == 0
    assert not ({tuple(v) for v in (cell[heldout] // 8)} & {tuple(v) for v in (cell[~heldout] // 8)})
    return dict(
        cells=cell.astype(np.int16),
        uv=uv[selected].astype(np.float32),
        camera=camera[selected].astype(np.float32),
        hold=heldout,
        collision=conflict[flat[selected]],
        projected_points=int(len(uv)),
        multi_occupied_cells=int((counts > 1).sum()),
        conflict_cells=int(conflict.sum()),
        image_size=size,
    )


def yaw_rotation(yaw):
    return z_rotation(np.rad2deg(yaw))


def box_rect(box, spec, transform, size):
    signs = np.asarray(list(product((-.5, .5), repeat=3)))
    corners = (signs * box[3:6]) @ yaw_rotation(box[6]).T + box[:3]
    camera = transform_xyz(corners, transform)
    if (camera[:, 2] <= .25).any():
        return None
    pixels = project_camera(camera, spec)
    rectangle = np.r_[pixels.min(axis=0), pixels.max(axis=0)]
    clipped = np.clip(rectangle, 0., np.tile(size, 2))
    original_extent = np.maximum(1e-8, rectangle[2:] - rectangle[:2])
    extent = clipped[2:] - clipped[:2]
    if np.any(extent * SIDE / size < 8) or np.prod(extent / original_extent) < .65:
        return None
    return clipped


def make_xyz(depth, mask, rays):
    depth = np.asarray(depth, np.float32)
    if depth.shape != (SIDE, SIDE) or np.any(~np.isfinite(depth[mask])):
        raise ValueError("Dense depth is invalid on the common valid footprint")
    xyz = np.clip(rays * np.clip(depth, .25, 250.)[..., None], -250., 250.)
    xyz[~mask] = 250.
    return np.concatenate((xyz.transpose(2, 0, 1), mask[None].astype(np.float32)), axis=0)


def wrong_locations(depth, mask):
    # Swap both members of every pair so the input mask and valid-depth
    # marginal stay exactly the same after this intervention.
    paired = mask & mask[:, ::-1]
    wrong = np.where(paired, depth[:, ::-1], depth)
    np.testing.assert_array_equal(np.sort(depth[mask]), np.sort(wrong[mask]))
    return wrong
