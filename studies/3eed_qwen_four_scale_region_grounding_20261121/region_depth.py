"""Use a four-scale region to select existing Qwen fit returns without labels."""
import hashlib
import sys
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from inputs import OUT, PRIVATE, PROTOCOL, QWEN, ROOT, digest, load_panel, qwen_predictions, v3_pins, write
from geometry import camera, cell_points, front_offset, pixel_rays, raw_returns, rigid_xyz, to_sensor, variants
from crossplatform.physical_regions import regions
from crossplatform.round7.camera_projection import model_to_camera_matrix, project_model_points

sys.path[:0] = [str(ROOT / "crossplatform/round7/experiment"), str(ROOT / "crossplatform/round3/p0/upstream")]
from runner_helpers import sample_points
from utils.transform_waymo import transform_to_front_view


def z_buffer(uv, cam, ids):
    cells = (uv[ids] // 4).astype(np.int64)
    order = np.lexsort((cam[ids, 2], cells[:, 1], cells[:, 0]))
    cells = cells[order]
    ids = ids[order]
    if not len(ids):
        return ids, cells
    first = np.r_[True, (cells[1:] != cells[:-1]).any(1)]
    return ids[first], cells[first]


def held_cells(cells):
    return ((cells[:, 0]*73856093) ^ (cells[:, 1]*19349663)) % 2 == 1


def inside(uv, rectangle, factor=1.):
    center = (rectangle[:2] + rectangle[2:]) / 2.
    half = (rectangle[2:] - rectangle[:2]) * (factor/2.)
    return (np.abs(uv-center) <= half).all(1)


def model_sample(row):
    sampled, _, _ = sample_points({"pointpath": row["points"], "platform": row["target"], "seed": row["seed"]})
    if row["target"] == "waymo":
        sampled, _ = transform_to_front_view(sampled, np.empty((0, 7)), row["waymo_view"])
        sampled = sampled.astype(np.float32)
    return np.unique(sampled.astype(np.float64), axis=0)


def partition(xyz):
    ground = regions.terrain(xyz)
    height = xyz[:, 2] - xyz[:, :2]@ground[:2] - ground[2]
    cuts, labels = regions.levels(xyz, height > .20)
    assert len(cuts) == 4
    assignments = np.full_like(labels, -1)
    births, keys, counts = [], {}, []
    for level, line in enumerate(labels):
        ids = np.flatnonzero(line >= 0)
        if not len(ids):
            continue
        ids = ids[np.argsort(line[ids], kind="stable")]
        for group in np.split(ids, np.flatnonzero(np.diff(line[ids]))+1):
            key = int(group.min()), len(group)
            if key not in keys:
                keys[key] = len(births)
                births.append(level)
                counts.append(len(group))
            assignments[level, group] = keys[key]
    return cuts, assignments, np.asarray(births, dtype=int), counts


def transfer(xyz, reads, cuts, assignments, births):
    distances, nearest = cKDTree(xyz).query(reads)
    mapped = assignments[:, nearest]
    # A node that persists at later scales retains its original transfer radius.
    for level, line in enumerate(mapped):
        allowed = (line >= 0) & (distances <= cuts[level])
        ids = np.flatnonzero(allowed)
        allowed[ids] &= births[line[ids]] == level
        line[~allowed] = -1
    return mapped


def camera_box(raw, guide, point_depths):
    box = raw.copy()
    box[:3] = guide*(float(np.median(point_depths)) + front_offset(raw, guide))
    return box


def null_subset(subset, count, uid):
    seed = int(hashlib.sha256(("same_window_region_mismatch/"+uid).encode()).hexdigest()[:16], 16)
    ids = np.sort(np.random.default_rng(seed).choice(count, size=len(subset), replace=False))
    if np.array_equal(ids, subset) and len(subset) < count:
        ids = np.sort((ids+1) % count)
    assert len(ids) == len(subset)
    return ids


def localize(scene_cells, xy, sample, roi_idx, roi_uv, roi_depth, full_window, uid, raw, guide, baseline):
    info = dict(camera_box=baseline.tolist(), mismatch_box=baseline.tolist(), regions=0, candidates=[],
                chosen=None, roi_ids=[], mismatch_ids=[])
    if len(roi_idx) < 5:
        return info | {"status": "under_five_original_fit_cells"}
    if len(sample) < 9:
        return info | {"status": "under_nine_sampled_points"}
    cuts, assign, birth, sizes = partition(sample)
    count = len(birth)
    info["regions"] = count
    if count == 0:
        return info | {"status": "empty_region_tree"}
    # All visual and median cells are fit cells. The second block exactly
    # preserves A's nearest return selections at the central crop boundary.
    kind = transfer(sample, xy[np.r_[scene_cells, roi_idx]], cuts, assign, birth)
    scene_inside = inside(roi_uv[scene_cells], full_window)
    n_window = int(scene_inside.sum())
    if not n_window:
        return info | {"status": "no_visible_window_cells"}
    whole = kind[:, :len(scene_cells)]
    overlap = whole[:, scene_inside]
    visible_count = np.bincount(whole[whole >= 0], minlength=count)
    overlap_count = np.bincount(overlap[overlap >= 0], minlength=count)
    central = kind[:, len(scene_cells):]
    only = np.bincount(central[central >= 0], minlength=count)
    eligible = np.flatnonzero((only >= 5) & (overlap_count > 0))
    if not len(eligible):
        return info | {"status": "no_region_with_five_central_cells"}
    for node in eligible:
        local = np.flatnonzero(central[birth[node]] == node)
        assert len(local) == only[node] and overlap_count[node] <= visible_count[node]
        voting = float(overlap_count[node])**2 / (int(visible_count[node])*n_window)
        info["candidates"].append(dict(node=int(node), scale=int(birth[node]), sampled_size=sizes[node],
           projected_n=int(visible_count[node]), window_n=int(overlap_count[node]), central_ids=local.tolist(),
           visual_weight=voting, depth=float(camera_box(raw, guide, roi_depth[local])[2])))
    best = max(info["candidates"], key=lambda item: (item["visual_weight"], -item["node"]))
    local = np.asarray(best["central_ids"])
    mismatch = null_subset(local, len(roi_idx), uid)
    info.update(status="used_region", chosen=best["node"], roi_ids=local.tolist(), mismatch_ids=mismatch.tolist(),
                camera_box=camera_box(raw, guide, roi_depth[local]).tolist(),
                mismatch_box=camera_box(raw, guide, roi_depth[mismatch]).tolist(),
                nontrivial_mismatch=not np.array_equal(local, mismatch))
    return info


def run():
    PRIVATE.mkdir(mode=0o700, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    predictions = qwen_predictions()
    pins = v3_pins()
    panel = load_panel()
    began = perf_counter()
    outputs = {}
    for row in panel:
        uid = row["uid"]
        prior = predictions[uid]
        if not prior["camera_boxes"]:
            outputs[uid] = dict(status="invalid_qwen_3d", variants={}, baseline=None)
            continue
        spec, size, transform = camera(row)
        uv, cam = raw_returns(row, spec, size)
        raw = np.asarray(prior["camera_boxes"]["raw"], dtype=float)
        rerun, measured, _ = variants(raw, prior["pixel"], spec, size, uv, cam)
        base = rerun["ray_depth"]
        np.testing.assert_array_equal(base, prior["camera_boxes"]["ray_depth"])
        np.testing.assert_array_equal(to_sensor(base, transform)[0], prior["sensor_boxes"]["ray_depth"])
        assert measured == prior["support"]
        result = dict(status="valid_qwen_3d", variants={}, baseline=base.tolist(), fit_bins=measured["fit_bins"])
        outputs[uid] = result
        if prior["pixel"] is None:
            result["status"] = "missing_qwen_2d"
            continue
        rectangle = np.asarray(prior["pixel"], float)*np.tile(size, 2)/1000.
        guide = pixel_rays((rectangle[:2]+rectangle[2:])/2., spec)[0]
        center_ids, cells = z_buffer(uv, cam, np.flatnonzero(inside(uv, rectangle, factor=.5)))
        central_fit = center_ids[~held_cells(cells)]
        ref_fit, ref_held = cell_points(uv, cam, rectangle)
        np.testing.assert_array_equal(cam[central_fit], ref_fit)
        np.testing.assert_array_equal(cam[center_ids[held_cells(cells)]], ref_held)
        all_ids, all_cells = z_buffer(uv, cam, np.arange(len(uv)))
        scene_fit = all_ids[~held_cells(all_cells)]
        xy = rigid_xyz(cam, np.linalg.inv(model_to_camera_matrix(spec)))
        sample = model_sample(row)
        projected = project_model_points(sample, spec, size)
        visible = projected["in_image"]
        to_remove = np.zeros(len(sample), dtype=bool)
        to_remove[visible] = held_cells((projected["uv"][visible] // 4).astype(np.int64))
        for name, subset in (("expanded", sample), ("fit_tree", sample[~to_remove])):
            result["variants"][name] = localize(scene_fit, xy, subset, central_fit, uv, ref_fit[:, 2],
                                               rectangle, uid, raw, guide, base)
        result["removed_from_fit_tree"] = int(to_remove.sum())
        result["registered_sample_points"] = len(sample)
        if len(outputs) % 16 == 0:
            print(len(outputs), "/ 96 validated", flush=True)
    path = PRIVATE / "region_predictions.json"
    write(path, outputs)
    assert len(outputs) == 96 and v3_pins() == pins
    write(OUT / "region_seal.json", dict(predictions=digest(path), protocol=digest(PROTOCOL),
        code=digest(__file__), frames=len(outputs), v3_geometry_pins=pins,
        qwen_seal=digest(QWEN / "results/geometry_seal.json"), exact_A_camera_and_sensor_reconstruction=True,
        qwen_generations=0, elapsed_seconds=perf_counter()-began, GT_read=False, FINAL_access=False))


if __name__ == "__main__":
    run()
