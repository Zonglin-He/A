"""Rebuild the isolated fit and seal the physical and applicability controls."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse
from time import perf_counter

import numpy as np

from cray import select, vehicle_reply
from locked_roster import HERE, OUT, WORK, locked_inputs, prediction, replies, roster
from geometry_full import prior


def bounded_anchor(points, terrain, yaw, dimensions, eye):
    """The original camera-facing anchor with a measured-extent lower bound."""
    c, s = np.cos(yaw), np.sin(yaw)
    axes = np.array([[c, s], [-s, c]])
    local = points[:, :2] @ axes.T
    dims = np.r_[np.maximum(dimensions[:2], np.ptp(local, axis=0)), dimensions[2]]
    pose = prior.fixed_size_anchor(points, terrain, yaw, dims, eye)
    bottom = pose[2] - dims[2] / 2
    height = max(dims[2], points[:, 2].max() - bottom)
    pose[5] = height
    pose[2] = bottom + height / 2
    delta_xy = (points[:, :2] - pose[:2]) @ axes.T
    assert (np.abs(delta_xy) <= pose[3:5] / 2 + 1.e-8).all()
    assert (np.abs(points[:, 2] - pose[2]) <= pose[5] / 2 + 1.e-8).all()
    return pose, np.any(pose[3:6] > dimensions + 1.e-8)


def rebuild(frame, responses, base):
    arms = {k: base["boxes"]["C_pca"] for k in ("F_complete", "B_pca", "B_ray")}
    details = dict(fit_points=0, bounded_pca=False, bounded_qwen=False, bounded_chooses_qwen=False)
    if not base["strict"]:
        return arms, details
    assert roster.file_sha(frame["points"]) == frame["point_sha"]
    spec, size, transform = prior.camera(frame)
    uv, returns = prior.raw_returns(frame, spec, size)
    transform_model = prior.model_to_camera_matrix(spec)
    inverse = np.linalg.inv(transform_model)
    xyz = prior.rigid_xyz(returns, inverse)
    window = np.asarray(responses["2d"]["value"]) * np.tile(size, 2) / 1000.
    indices, cells = prior.z_buffer(uv, returns, np.arange(len(uv)))
    parts = prior.reserve(cells)
    ids, bins = prior.z_buffer(uv, returns, np.flatnonzero(prior.inside(uv, window, .5)))
    central = ids[prior.reserve(bins) == 0]
    train_ids = indices[parts == 0]
    in_window = prior.inside(uv[train_ids], window)
    combined = xyz[np.r_[train_ids, central]]
    fit, isolation = prior.fit_sample(frame, spec, size)
    assert not isolation["retained_withheld_raw"]
    catalog = prior.partition(fit)
    chosen = prior.tree_winner(fit, catalog, combined, in_window, len(central))
    assert chosen is not None
    support = chosen["points"]
    terrain = prior.regions.terrain(fit)
    a = np.array(base["boxes"]["A"])
    dims = a[3:6]
    fit_complete = prior.regions.complete(prior.regions.measured_box(support, terrain), dims)
    pca_yaw = prior.to_long_axis(fit_complete)[6]
    target, view = frame["target"], frame["suffix"]
    qwen_yaw = a[6] + (np.deg2rad(prior.WAYMO_ANGLES[view]) if target == "waymo" else 0.)

    def raw(box):
        return prior.sensor_frame_box(box, target, view)

    eye = inverse[:3, 3]
    fixed_pca = raw(prior.fixed_size_anchor(support, terrain, pca_yaw, dims, eye))
    np.testing.assert_array_equal(fixed_pca, base["boxes"]["C_pca"])
    bp, expanded_p = bounded_anchor(support, terrain, pca_yaw, dims, eye)
    bq, expanded_q = bounded_anchor(support, terrain, qwen_yaw, dims, eye)
    bp, bq = raw(bp), raw(bq)
    chosen_box = bp
    details.update(fit_points=len(support), bounded_pca=bool(expanded_p), bounded_qwen=bool(expanded_q))
    if base["selection_rays"] >= 5:
        cuts, assign, births, _ = catalog
        withheld = indices[prior.inside(uv[indices], window) & (parts > 0)]
        attached = prior.transfer(fit, xyz[withheld], cuts, assign, births)[chosen["level"]] == chosen["node"]
        attached_ids = withheld[attached]
        split = prior.reserve((uv[attached_ids]//4).astype(np.int64))
        usable = returns[attached_ids[split == 1]]
        assert len(usable) == base["selection_rays"]
        fixed_qwen = raw(prior.fixed_size_anchor(support, terrain, qwen_yaw, dims, eye))
        expected = fixed_qwen if base["chooses_qwen"] else fixed_pca
        np.testing.assert_array_equal(expected, base["boxes"]["C_ray"])
        cap = max(.1, .5 * min(dims[:2]))
        costs = [float(cost.mean()) for _, _, cost in prior.disagreement((bp, bq), usable, transform, cap)]
        uses_qwen = costs[1] < costs[0] - 1.e-12
        chosen_box = bq if uses_qwen else bp
        details.update(bounded_chooses_qwen=bool(uses_qwen))
    arms.update(F_complete=raw(fit_complete).tolist(), B_pca=bp.tolist(), B_ray=chosen_box.tolist())
    return arms, details


def work(job):
    frame, checksum, pins = job
    uid = frame["uid"]
    old = prediction(frame, checksum, pins)
    response = replies(frame, old, reply_index)
    output, details = rebuild(frame, response, old)
    selected = select(response, old)
    output["C_ray_applicable"] = selected["box"]
    return dict(uid=uid, previous=checksum, boxes=output, use_ray=selected["applicable"],
                ray_eligible=old["selection_rays"] >= 5,
                label_2d_vehicle=vehicle_reply(response["2d"]), label_3d_vehicle=vehicle_reply(response["3d"]),
                **details)


def init_worker(hashes):
    global reply_index
    reply_index = hashes


def run(workers):
    start = perf_counter()
    frames, _, seal, old, hashes = locked_inputs()
    spec = HERE / "PROTOCOL.json"
    config = roster.read(spec)
    assert not config["FINAL_access"] and not config["target_Gt_for_predictions"]
    jobs = ((f, old[f["uid"]], seal["pins"]) for f in frames)
    predictions = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(hashes,)) as pool:
        for p in pool.map(work, jobs, chunksize=8):
            predictions.append(p)
            if len(predictions) % 1000 == 0:
                print("Predicted", len(predictions), "/", len(frames), flush=True)
    assert len(predictions) == len(frames)
    groups = defaultdict(list)
    for f, p in zip(frames, predictions, strict=True):
        assert f["uid"] == p["uid"]
        if p["ray_eligible"]:
            groups[f["target"], f["recording"]].append(p)
    rng = np.random.default_rng(config["negative_control_seed"])
    for _, group in sorted(groups.items()):
        mask = np.array([p["use_ray"] for p in group])
        control = rng.permutation(mask)
        assert control.sum() == mask.sum()
        for p, flag in zip(group, control, strict=True):
            p["shuffled_use_ray"] = bool(flag)
    for p in predictions:
        p.setdefault("shuffled_use_ray", False)
    path = WORK / "predictions.json"
    roster.write(path, predictions)
    roster.write(OUT / "prediction_seal.json", dict(
        historical=roster.sha(roster.OUT/"prediction_seal.json"), payload=roster.sha(path),
        protocol=roster.sha(spec), code=roster.sha(__file__), selector=roster.sha(HERE/"cray.py"),
        descriptions=len(predictions), arms=list(config["arms"]),
        shares_frozen_old_replies=True, qwen_inferences=0, target_labels_read=False, FINAL_access=False,
        use_ray=dict(Counter(f["target"] for f, p in zip(frames, predictions, strict=True) if p["use_ray"])),
        seconds=perf_counter()-start,
    ))
    print("Sealed", len(predictions), "predictions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args().workers)
