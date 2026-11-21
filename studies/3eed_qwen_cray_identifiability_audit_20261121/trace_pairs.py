"""Recover both frozen poses and Selection first intersections without ground truth."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
import argparse
import gzip
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OUT, WORK = HERE/"results", HERE/"private"
ACTIVE = HERE.parent/"3eed_qwen_cray_simplification_20261121"
sys.path.insert(0, str(ACTIVE))
from cray_simple import (
    FOUR_SCALES, MIN_RETURNS, axial, choose_region, completed_dimensions, geometry, pose_box, vehicle_reply
)
from locked_roster import locked_inputs, prediction, replies, roster


BOX_ATOL = 5.e-10


def inputs():
    frames, _, historical, checks, reply_index = locked_inputs()
    method = roster.read(ACTIVE/"METHOD_LOCK.json")
    sealed = roster.read(ACTIVE/"results/prediction_seal.json")
    assert method["method"] == "C_ray_simple" and tuple(method["vehicle_scales"]) == FOUR_SCALES
    assert method["entry"]["sha256"] == sealed["code"] == roster.sha(ACTIVE/"cray_simple.py")
    assert sealed["model"] == roster.sha(roster.OUT/"qwen_seal.json")
    assert not sealed["target_labels_read"] and not sealed["FINAL_access"]
    bank = ACTIVE/"private/predictions.json"
    assert sealed["payload"] == roster.sha(bank)
    active = roster.read(bank)
    assert len(active) == len(frames) == 9310
    return frames, historical["pins"], checks, reply_index, active


def box_axes(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))


def first_depth(box, returns, transform):
    """Return original slab entry depths, using 0 to encode a miss."""
    inverse = np.linalg.inv(transform)
    axes = box_axes(box[6])
    direction = (returns/returns[:, 2:3]) @ inverse[:3, :3].T @ axes
    origin = (inverse[:3, 3]-box[:3]) @ axes
    half = box[3:6]/2
    parallel = np.abs(direction) < 1.e-12
    lo = np.divide(-half-origin, direction, out=np.zeros_like(direction), where=~parallel)
    hi = np.divide(half-origin, direction, out=np.zeros_like(direction), where=~parallel)
    entry = np.where(parallel, -np.inf, np.minimum(lo, hi)).max(1)
    leave = np.where(parallel, np.inf, np.maximum(lo, hi)).min(1)
    hit = ~(parallel & (np.abs(origin) > half)).any(1) & (leave >= entry) & (entry > 0)
    return np.where(hit, entry, 0.)


def clipped_residual(depth, observed, cap):
    """The locked normalized per-ray residual, with unit cost on misses."""
    return np.where(depth > 0, np.minimum(np.abs(depth-observed)/cap, 1.), 1.)


def fitted_pair(frame, response):
    spec, size, transform = geometry.camera(frame)
    original = geometry.to_sensor(np.array(response["3d"]["value"]), transform)[0]
    uv, returns = geometry.raw_returns(frame, spec, size)
    window = np.asarray(response["2d"]["value"])*np.tile(size, 2)/1000.
    central, cells = geometry.z_buffer(uv, returns, np.flatnonzero(geometry.inside(uv, window, .5)))
    central = central[geometry.reserve(cells) == 0]
    assert len(central) >= MIN_RETURNS

    indices, cells = geometry.z_buffer(uv, returns, np.arange(len(uv)))
    parts = geometry.reserve(cells)
    train = indices[parts == 0]
    select = indices[(parts == 1) & geometry.inside(uv[indices], window)]
    inside = geometry.inside(uv[train], window)
    model_inverse = np.linalg.inv(geometry.model_to_camera_matrix(spec))
    xyz = geometry.rigid_xyz(returns, model_inverse)
    fit, independence = geometry.fit_sample(frame, spec, size)
    assert not independence["retained_withheld_raw"]
    ground = geometry.regions.terrain(fit)
    points, mask = choose_region(fit, ground, FOUR_SCALES, xyz[np.r_[train, central]], inside, xyz[select])
    assert points is not None
    held = returns[select[mask]]
    dimensions = original[3:6]
    measured = geometry.regions.measured_box(points, ground)
    ext = completed_dimensions(measured[3:6], dimensions)
    pca_yaw = axial(measured[6] + (np.pi/2 if ext[1] > ext[0] else 0.))
    qwen_yaw = original[6]
    if frame["target"] == "waymo":
        qwen_yaw += np.deg2rad(geometry.WAYMO_ANGLES[frame["suffix"]])
    p, q = [
        geometry.sensor_frame_box(pose_box(points, ground, dimensions, yaw, model_inverse[:3, 3]),
                                  frame["target"], frame["suffix"])
        for yaw in (pca_yaw, qwen_yaw)
    ]
    return p, q, held, transform, max(.1, .5*min(dimensions[:2]))


def initialize(index):
    global reply_hashes
    reply_hashes = index


def trace_one(job):
    frame, digest, pins, saved = job
    uid = frame["uid"]
    try:
        frozen = prediction(frame, digest, pins)
        assert saved["uid"] == uid
        actual = saved["simple4"]
        n = frozen["selection_rays"]
        row = dict(uid=uid, selection_rays=n, production=actual["applicable"],
                   h=frozen["boxes"]["H"], production_box=actual["box"])
        if n < MIN_RETURNS:
            assert not row["production"]
            return row
        assert roster.file_sha(frame["points"]) == frame["point_sha"]
        assert roster.file_sha(frame["rgb"]) == frame["rgb_sha"]
        response = replies(frame, frozen, reply_hashes)
        p, q, held, transform, cap = fitted_pair(frame, response)
        assert row["production"] == vehicle_reply(response["2d"])
        assert len(held) == n
        assert np.isfinite(held).all() and (held[:, 2] > 0).all()
        pred = []
        for box in (p,q):
            depth = first_depth(box, held, transform)
            hit, residual = geometry.ray_residual(box, held, transform)
            np.testing.assert_array_equal(depth>0,hit)
            np.testing.assert_array_equal(np.abs(depth[hit]-held[hit,2]),residual[hit])
            pred.append(depth)
        lp,lq = (clipped_residual(d,held[:,2],cap) for d in pred)
        choose_qwen = lq.sum() < lp.sum() - 1.e-12*n
        assert choose_qwen == frozen["chooses_qwen"]
        selected = q if choose_qwen else p
        targets = [(p, frozen["boxes"]["C_pca"]), (selected, frozen["boxes"]["C_ray"])]
        if actual["applicable"]:
            assert actual["heading"] == ("qwen" if choose_qwen else "pca")
            targets.append((selected, actual["box"]))
        maximum = max(float(np.max(np.abs(a-b))) for a,b in targets)
        assert maximum <= BOX_ATOL
        row.update(p=p.tolist(), q=q.tolist(), choose_qwen=bool(choose_qwen), held=held.tolist(),
                   transform=transform.tolist(), cap=cap, p_depth=pred[0].tolist(), q_depth=pred[1].tolist(),
                   matched_box_error=maximum)
        return row
    except Exception as error:
        raise RuntimeError(f"Could not reproduce Selection rays for {frame['target']} description {uid}") from error


def run(workers):
    start = perf_counter()
    protocol = roster.read(HERE/"PROTOCOL.json")
    assert not protocol["gate_implemented"] and not protocol["FINAL_access"]
    frames,pins,checks,index,active = inputs()
    jobs = ((f,checks[f["uid"]],pins,a) for f,a in zip(frames,active,strict=True))
    path = WORK/"sealed_rays.jsonl.gz"
    partial = path.with_suffix(".tmp")
    stats = dict(descriptions=0, pairs=0, production_pairs=0, production_switches=0,
                 max_box_replay_error=0., rays=0)
    with partial.open("wb") as out, gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0, compresslevel=3) as archive:
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(index,)) as pool:
            for row in pool.map(trace_one, jobs, chunksize=8):
                archive.write((json.dumps(row, separators=(",",":"),allow_nan=False)+"\n").encode())
                stats["descriptions"] += 1
                stats["pairs"] += "p" in row
                stats["production_pairs"] += row["production"]
                stats["production_switches"] += row["production"] and row.get("choose_qwen",False)
                stats["max_box_replay_error"] = max(stats["max_box_replay_error"],row.get("matched_box_error",0.))
                stats["rays"] += row["selection_rays"] if "p" in row else 0
                if stats["descriptions"] % 1000 == 0:
                    print("Replayed", stats["descriptions"], "/9310", flush=True)
    assert stats["descriptions"] == 9310 and stats["pairs"] == 7948 and stats["production_pairs"] == 5910
    partial.replace(path)
    roster.write(OUT/"ray_seal.json", dict(
        **stats, payload=roster.sha(path), code=roster.sha(__file__),
        protocol=roster.sha(HERE/"PROTOCOL.json"), active_entry=roster.sha(ACTIVE/"cray_simple.py"),
        active_prediction_seal=roster.sha(ACTIVE/"results/prediction_seal.json"),
        historical_prediction_seal=roster.sha(roster.OUT/"prediction_seal.json"),
        ray_masks_and_depth_residuals_exact=True, gt_opened=False, FINAL_access=False,
        seconds=perf_counter()-start))
    print("Sealed",stats,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args().workers)
