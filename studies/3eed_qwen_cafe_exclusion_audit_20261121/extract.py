"""Seal label-free exclusion evidence for the immutable C_ray_simple pairs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from center_sets import interior, measure


HERE = Path(__file__).resolve().parent
OUT, WORK = HERE/"results", HERE/"private"
PROTOCOL = HERE/"PROTOCOL.json"
PAYLOAD = WORK/"sealed_witnesses.jsonl.gz"
SURFACE = HERE.parent/"3eed_qwen_cray_surface_rescoring_20261121"
sys.path.insert(0, str(SURFACE))
from rescore import (ACTIVE, AUDIT, FOUR_SCALES, MIN_RETURNS, choose_region, geometry,
                     inputs, prediction, replies, roster, verified_trace)
from cray_simple import region_catalog


def context(frame, window, held, transform):
    spec, size, camera = geometry.camera(frame)
    np.testing.assert_array_equal(camera, transform)
    uv, returns = geometry.raw_returns(frame, spec, size)
    window = np.asarray(window)*np.tile(size, 2)/1000.
    central, central_cells = geometry.z_buffer(uv, returns, np.flatnonzero(geometry.inside(uv, window, .5)))
    central = central[geometry.reserve(central_cells) == 0]
    ids, cells = geometry.z_buffer(uv, returns, np.arange(len(uv)))
    split = geometry.reserve(cells)
    np.testing.assert_array_equal(split > 0, geometry.held_cells(cells))
    fit_ids, scene_sel = ids[split == 0], ids[split == 1]
    in_window = geometry.inside(uv[fit_ids], window)
    sel_window = geometry.inside(uv[scene_sel], window)
    model_to_camera = geometry.model_to_camera_matrix(spec)
    xyz = geometry.rigid_xyz(returns, np.linalg.inv(model_to_camera))
    sample, audit = geometry.fit_sample(frame, spec, size)
    assert not audit["retained_withheld_raw"] and len(central) >= MIN_RETURNS
    ground = geometry.regions.terrain(sample)
    support, targets = choose_region(sample, ground, FOUR_SCALES, xyz[np.r_[fit_ids, central]],
                                     in_window, xyz[scene_sel[sel_window]])
    assert support is not None
    target_ids = scene_sel[sel_window][targets]
    np.testing.assert_array_equal(returns[target_ids], held)
    inverse = np.linalg.inv(transform)
    return (sample, ground, support, xyz[scene_sel], geometry.rigid_xyz(returns[scene_sel], inverse),
            geometry.rigid_xyz(support, inverse @ model_to_camera), sel_window, np.isin(scene_sel, target_ids))


def stable_labels(sample, ground, support, diff):
    radii, labels, births = region_catalog(sample, ground, FOUR_SCALES)
    tree = cKDTree(sample)
    target_distance, target_seeds = tree.query(support)
    assert (target_distance == 0.).all() and len(np.unique(target_seeds)) == len(support)
    target = labels[:, target_seeds[0]]
    birth_levels = [
        level for level, node in enumerate(target)
        if node >= 0 and births[node] == level and (labels[level, target_seeds] == node).all()
        and (labels[level] == node).sum() == len(support)
    ]
    (birth,) = birth_levels
    fine = min(birth, len(FOUR_SCALES)-2)
    distance, nearest = tree.query(diff)
    separated = np.ones(len(diff), dtype=bool)
    for level in (fine, fine+1):
        component = labels[level, nearest]
        used_by_target = np.unique(labels[level, target_seeds])
        separated &= (distance <= radii[level]) & (component >= 0) & ~np.isin(component, used_by_target)
    return birth, fine, np.where(separated, labels[fine, nearest], -1)


def corroborate(labels, eligible):
    use = eligible & (labels >= 0)
    groups, counts = np.unique(labels[use], return_counts=True)
    qualified = groups[counts >= MIN_RETURNS]
    return use & np.isin(labels, qualified), int(use.sum()), len(qualified)


def initialize(index):
    global reply_index
    reply_index = index


def process(job):
    frame, record, pins, check, official = job
    uid = frame["uid"]
    assert uid == record["uid"] == official["uid"]
    active = record["production"]
    assert active == official["simple4"]["applicable"]
    if not active:
        return dict(uid=uid, production=False)
    try:
        assert roster.file_sha(frame["points"]) == frame["point_sha"]
        frozen = prediction(frame, check, pins)
        original_choice = record["choose_qwen"]
        assert frozen["chooses_qwen"] == original_choice
        p, q, held, matrix = (np.asarray(record[k], float) for k in ("p", "q", "held", "transform"))
        assert len(held) == record["selection_rays"] >= MIN_RETURNS
        assert np.isfinite((p, q)).all() and (p[3:6] > 0.).all()
        np.testing.assert_array_equal(p[3:6], q[3:6])
        np.testing.assert_allclose(q if original_choice else p, official["simple4"]["box"], rtol=0., atol=5.e-10)
        window = replies(frame, frozen, reply_index)["2d"]["value"]
        sample, ground, support, model, sensor, target, in_win, selected = context(frame, window, held, matrix)
        p_inside, q_inside = interior(sensor, p), interior(sensor, q)
        xor = p_inside ^ q_inside
        birth, level, labels = stable_labels(sample, ground, support, model[xor])
        p_only, q_only = p_inside[xor], q_inside[xor]
        assert (p_only ^ q_only).all()
        in_window = in_win[xor]
        wp, nsp, ngp = corroborate(labels, p_only)
        wq, nsq, ngq = corroborate(labels, q_only)
        accepted = wp | wq
        assert not (selected[xor] & (labels >= 0)).any()
        witness = sensor[xor][accepted]
        assert len(witness) == wp.sum()+wq.sum()
        sizes = [measure(box, target, witness) for box in (p, q)]
        for tag in sizes:
            if tag["joint_possible"]:
                assert tag["joint_area"] <= tag["positive_area"]+1.e-9
                assert tag["joint_volume_change_bound"] <= tag["positive_volume_change_bound"]+1.e-9
        return dict(uid=uid, production=True, choose_qwen=original_choice,
                    original_selection_cells=len(held), scene_selection_cells=len(sensor), target_fit_seeds=len(support),
                    birth=FOUR_SCALES[birth], fine=FOUR_SCALES[level], coarse=FOUR_SCALES[level+1],
                    xor_p=int(p_only.sum()), xor_q=int(q_only.sum()), stable_p=nsp, stable_q=nsq,
                    witness_p=int(wp.sum()), witness_q=int(wq.sum()), regions_p=ngp, regions_q=ngq,
                    in_window_p=int((wp & in_window).sum()), in_window_q=int((wq & in_window).sum()),
                    p_set=sizes[0], q_set=sizes[1])
    except Exception as exc:
        raise RuntimeError(f"Counterfactual audit failed on {frame['target']} description {uid}") from exc


def run(workers):
    started = perf_counter()
    spec = roster.read(PROTOCOL)
    assert not spec["FINAL_access"] and spec["prediction_changes"] == 0
    lock = roster.sha(ACTIVE/"METHOD_LOCK.json")
    assert lock == spec["locked_method_sha256"]
    trace, source = verified_trace()
    frames, pins, checks, index, official = inputs()
    counts = dict(descriptions=0, active_pairs=0, original_qwen_choices=0, original_selection_cells=0)
    tmp = PAYLOAD.with_suffix(".tmp")
    with gzip.open(source, "rt") as stream, tmp.open("wb") as output, \
            gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=4) as archive, \
            ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(index,)) as pool:
        jobs = ((f, json.loads(t), pins, checks[f["uid"]], p)
                for f, t, p in zip(frames, stream, official, strict=True))
        for row in pool.map(process, jobs, chunksize=10):
            archive.write((json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n").encode())
            counts["descriptions"] += 1
            counts["active_pairs"] += row["production"]
            counts["original_qwen_choices"] += row.get("choose_qwen", False)
            counts["original_selection_cells"] += row.get("original_selection_cells", 0)
            if counts["descriptions"] % 1000 == 0:
                print("Audited", counts["descriptions"], "/ 9310", flush=True)
    assert counts == dict(descriptions=9310, active_pairs=5910, original_qwen_choices=2640,
                          original_selection_cells=688114)
    assert counts["active_pairs"] == trace["production_pairs"]
    tmp.replace(PAYLOAD)
    roster.write(OUT/"evidence_seal.json", dict(
        **counts, frozen_method_lock=lock, source_trace_payload=trace["payload"],
        source_trace_seal=roster.sha(AUDIT/"results/ray_seal.json"), protocol=roster.sha(PROTOCOL),
        extractor=roster.sha(__file__), geometry=roster.sha(HERE/"center_sets.py"), payload=roster.sha(PAYLOAD),
        fit_selection_disjoint=True, original_target_selection_exact=True,
        iou_rows_opened=False, additional_GT_file_reads=0, prediction_changes=0,
        new_Qwen_queries=0, FINAL_access=False, seconds=perf_counter()-started,
    ))
    print("Sealed", counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args().workers)
