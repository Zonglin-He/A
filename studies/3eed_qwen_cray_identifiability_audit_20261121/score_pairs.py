"""Read ordinary GT after the ray seal and measure per-description disagreement."""
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import argparse
import csv
import gzip
import json

import numpy as np

from trace_pairs import ACTIVE, HERE, OUT, WORK, box_axes, clipped_residual, first_depth, inputs, roster
import score_full


FEATURES = (
    "common_hits", "only_pca_hits", "only_qwen_hits", "neither_hits",
    "predicted_rms_m", "predicted_rms_scaled", "info", "signed_evidence",
    "observability_bound", "ray_margin", "common_margin", "one_hit_margin",
    "both_saturated_fraction", "yaw_difference_deg", "center_difference_m",
    "inside_target_rays", "gt_surface_rays", "gt_member_margin", "gt_surface_margin",
    "gt_first_surface_abs_error_m",
)


def archived_scores(folder):
    metadata = roster.read(folder/"results/evaluation_receipt.json")
    assert metadata["descriptions"] == 9310 and not metadata["FINAL_access"]
    result = {}
    for platform in score_full.PLATFORMS:
        name = f"rows_{platform}.csv.gz"
        path = folder/"results"/name
        assert roster.sha(path) == metadata["files"][name]
        with gzip.open(path,"rt",newline="") as f:
            for r in csv.DictReader(f):
                assert r["uid"] not in result
                result[r["uid"]] = r
    assert len(result) == 9310
    return result


def measure_one(job):
    frame, data, old, active = job
    uid = frame["uid"]
    assert data["uid"] == old["uid"] == active["uid"] == uid
    assert old["category"] == active["category"] and old["target"] == frame["target"]
    is_pair = "p" in data
    uses_qwen = data.get("choose_qwen",False)
    row = {k:old[k] for k in ("uid","target","recording","frame","category")}
    row.update(available=is_pair, production=data["production"], choose_qwen=uses_qwen,
               rays=data["selection_rays"], iou_h=float(old["iou_H"]),
               iou_current=float(active["iou_simple4"]))
    if not is_pair:
        row.update(iou_pca=None, iou_qwen=None, iou_ungated=None)
        row.update(dict.fromkeys(FEATURES))
        return row,0.

    annotation = roster.read(frame["calibration"])["ground_info"]
    assert annotation[frame["slot"]]["class"].strip().lower() == row["category"]
    truth = score_full.canonical_box(score_full.truth_for_slot(frame,annotation))
    p,q = (np.asarray(data[name],float) for name in ("p","q"))
    ious,error = score_full.score_boxes(np.asarray((p,q,data["h"],data["production_box"])),truth)
    iou_p,iou_q,iou_h,iou_current = ious
    picked = iou_q if uses_qwen else iou_p
    assert abs(iou_h-row["iou_h"])<1.e-12 and abs(iou_current-row["iou_current"])<1.e-12
    assert abs(iou_p-float(old["iou_C_pca"]))<1.e-12
    assert abs(picked-float(old["iou_C_ray"]))<1.e-12
    for actual,recorded in ((iou_p,float(old["iou_C_pca"])),(picked,float(old["iou_C_ray"])),
                            (iou_current,row["iou_current"])):
        assert (actual>.5) == (recorded>.5)

    returns = np.asarray(data["held"],float)
    tmat = np.asarray(data["transform"],float)
    z = returns[:,2]
    a,b = (np.asarray(data[name],float) for name in ("p_depth","q_depth"))
    hit_p,hit_q = a>0,b>0
    common = hit_p & hit_q
    either = hit_p ^ hit_q
    count = len(z)
    assert count == data["selection_rays"]
    cap = data["cap"]
    depth_difference = b[common]-a[common]
    scaled = depth_difference/cap
    separation = float((np.minimum(np.abs(scaled),1.).sum()+either.sum())/count)
    lp,lq = clipped_residual(a,z,cap),clipped_residual(b,z,cap)
    margin = lp-lq
    bias = float(margin.mean())
    shared_margin = float(margin[common].sum()/count)
    different_mask_margin = float(margin[either].sum()/count)
    assert abs(bias) <= separation + 1.e-12
    assert abs(bias-shared_margin-different_mask_margin) < 1.e-12
    assert np.all(np.abs(margin[common])<=np.minimum(np.abs(scaled),1.)+1.e-12)
    assert bool(bias>1.e-12) == uses_qwen

    inverse = np.linalg.inv(tmat)
    raw = returns @ inverse[:3,:3].T + inverse[:3,3]
    within = (np.abs((raw-truth[:3]) @ box_axes(truth[6])) <= truth[3:6]/2+1.e-8).all(1)
    gt_depth = first_depth(truth,returns,tmat)
    visible_gt = within & (gt_depth>0)
    enough = visible_gt.sum()>=5
    paired_gt = (clipped_residual(a[visible_gt],gt_depth[visible_gt],cap) -
                 clipped_residual(b[visible_gt],gt_depth[visible_gt],cap))
    row.update(
        iou_pca=float(iou_p), iou_qwen=float(iou_q), iou_ungated=float(picked),
        common_hits=int(common.sum()), only_pca_hits=int((hit_p&~hit_q).sum()),
        only_qwen_hits=int((hit_q&~hit_p).sum()), neither_hits=int((~hit_p&~hit_q).sum()),
        predicted_rms_m=float(np.sqrt(np.mean(depth_difference**2))) if common.any() else None,
        predicted_rms_scaled=float(np.sqrt(np.mean(scaled**2))) if common.any() else None,
        info=float(scaled@scaled) if common.any() else None,
        signed_evidence=float(((z[common]-(a[common]+b[common])/2)/cap)@scaled) if common.any() else None,
        observability_bound=separation, ray_margin=bias, common_margin=shared_margin, one_hit_margin=different_mask_margin,
        both_saturated_fraction=float(((lp==1.)&(lq==1.)).mean()),
        yaw_difference_deg=float(np.rad2deg(abs((q[6]-p[6]+np.pi/2)%np.pi-np.pi/2))),
        center_difference_m=float(np.linalg.norm(q[:3]-p[:3])),
        inside_target_rays=int(within.sum()), gt_surface_rays=int(visible_gt.sum()),
        gt_member_margin=float(margin[visible_gt].mean()) if enough else None,
        gt_surface_margin=float(paired_gt.mean()) if enough else None,
        gt_first_surface_abs_error_m=float(np.median(np.abs(z[visible_gt]-gt_depth[visible_gt]))) if enough else None,
    )
    return row,error


def run(workers):
    start = perf_counter()
    frames, _, _, _, _ = inputs()
    seal = roster.read(OUT/"ray_seal.json")
    assert seal["active_prediction_seal"] == roster.sha(ACTIVE/"results/prediction_seal.json")
    assert seal["historical_prediction_seal"] == roster.sha(roster.OUT/"prediction_seal.json")
    assert seal["code"] == roster.sha(HERE/"trace_pairs.py")
    assert seal["protocol"] == roster.sha(HERE/"PROTOCOL.json")
    assert seal["ray_masks_and_depth_residuals_exact"] and not seal["gt_opened"] and not seal["FINAL_access"]
    payload = WORK/"sealed_rays.jsonl.gz"
    assert seal["payload"] == roster.sha(payload)
    old = archived_scores(roster.HERE)
    current = archived_scores(ACTIVE)
    rows, max_error = [], 0.
    with gzip.open(payload,"rt") as f, ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = ((frame,json.loads(line),old[frame["uid"]],current[frame["uid"]])
                for frame,line in zip(frames,f,strict=True))
        for row,error in pool.map(measure_one,jobs,chunksize=16):
            rows.append(row)
            max_error = max(max_error,error)
    assert len(rows) == len({r["uid"] for r in rows}) == seal["descriptions"] == 9310
    assert sum(r["available"] for r in rows) == 7948 and sum(r["production"] for r in rows) == 5910
    assert max_error < 1.e-7
    score_full.OUT = OUT
    files = {}
    for t in score_full.PLATFORMS:
        file = f"rows_{t}.csv.gz"
        files[file] = score_full.write_rows(file,[r for r in rows if r["target"]==t],compress=True)
    roster.write(OUT/"measurement_receipt.json",dict(
        ray_seal=roster.sha(OUT/"ray_seal.json"), code=roster.sha(__file__),
        files=files, descriptions=len(rows), paired_rows=7948, production_rows=5910,
        max_two_upright_iou_discrepancy=max_error, score_gap_bound_verified=True,
        gt_controls_post_seal=True, FINAL_access=False, seconds=perf_counter()-start))
    print("Measured all 9310 descriptions and 7948 pose pairs.",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args().workers)
