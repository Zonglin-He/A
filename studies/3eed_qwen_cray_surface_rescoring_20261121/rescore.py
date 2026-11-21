"""Rescore the original two pose candidates without opening annotation files."""
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
AUDIT = HERE.parent/"3eed_qwen_cray_identifiability_audit_20261121"
sys.path.insert(0,str(AUDIT))
from trace_pairs import (
    ACTIVE, FOUR_SCALES, MIN_RETURNS, box_axes, choose_region, clipped_residual, geometry,
    inputs, prediction, replies, roster
)


ARMS = ("first","interval","fit_offset")
TIE = 1.e-12


def verified_trace():
    seal = roster.read(AUDIT/"results/ray_seal.json")
    assert seal["code"] == roster.sha(AUDIT/"trace_pairs.py")
    assert seal["active_entry"] == roster.sha(ACTIVE/"cray_simple.py")
    assert seal["active_prediction_seal"] == roster.sha(ACTIVE/"results/prediction_seal.json")
    assert seal["descriptions"] == 9310 and seal["production_pairs"] == 5910
    assert seal["ray_masks_and_depth_residuals_exact"] and not seal["gt_opened"] and not seal["FINAL_access"]
    path = AUDIT/"private/sealed_rays.jsonl.gz"
    assert roster.sha(path) == seal["payload"]
    return seal,path


def intervals(box, rays, transform):
    """The pinned slab intersection with both forward entry and exit depths."""
    inv = np.linalg.inv(transform)
    axes = box_axes(box[6])
    direction = (rays/rays[:,2:3]) @ inv[:3,:3].T @ axes
    origin = (inv[:3,3]-box[:3]) @ axes
    half = box[3:6]/2
    parallel = np.abs(direction) < 1.e-12
    lo = np.divide(-half-origin,direction,out=np.zeros_like(direction),where=~parallel)
    hi = np.divide(half-origin,direction,out=np.zeros_like(direction),where=~parallel)
    front = np.where(parallel,-np.inf,np.minimum(lo,hi)).max(1)
    back = np.where(parallel,np.inf,np.maximum(lo,hi)).min(1)
    hit = ~(parallel & (np.abs(origin)>half)).any(1) & (back>=front) & (front>0)
    return np.where(hit,front,0.),np.where(hit,back,0.)


def associated_fit(frame, window, held, transform):
    """Reproduce the original winner after withholding the original selection cells."""
    spec,size,cam = geometry.camera(frame)
    np.testing.assert_array_equal(cam,transform)
    uv,returns = geometry.raw_returns(frame,spec,size)
    window = np.asarray(window)*np.tile(size,2)/1000.
    central,cells = geometry.z_buffer(uv,returns,np.flatnonzero(geometry.inside(uv,window,.5)))
    central = central[geometry.reserve(cells)==0]
    assert len(central)>=MIN_RETURNS
    ids,cells = geometry.z_buffer(uv,returns,np.arange(len(uv)))
    parts = geometry.reserve(cells)
    train = ids[parts==0]
    in_window = geometry.inside(uv[train],window)
    fit_ids = train[in_window]
    sel_ids = ids[(parts==1) & geometry.inside(uv[ids],window)]
    assert not np.intersect1d(fit_ids,sel_ids).size
    xyz = geometry.rigid_xyz(returns,np.linalg.inv(geometry.model_to_camera_matrix(spec)))
    fit,checked = geometry.fit_sample(frame,spec,size)
    assert not checked["retained_withheld_raw"]
    ground = geometry.regions.terrain(fit)
    points,attached = choose_region(fit,ground,FOUR_SCALES,xyz[np.r_[train,central]],in_window,
                                    xyz[np.r_[fit_ids,sel_ids]])
    assert points is not None
    n = len(fit_ids)
    np.testing.assert_array_equal(returns[sel_ids[attached[n:]]],held)
    return returns[fit_ids[attached[:n]]]


def estimate_offsets(fit, fit_intervals):
    valid = (fit_intervals[0][0]>0) & (fit_intervals[1][0]>0)
    if valid.sum()<MIN_RETURNS:
        return [(0.,None,None)]*2,int(valid.sum())
    estimates = []
    for front,back in fit_intervals:
        median = float(np.median(fit[valid,2]-front[valid]))
        thickness = float(np.median(back[valid]-front[valid]))
        estimates.append((float(np.clip(median,0.,thickness)),median,thickness))
    return estimates,int(valid.sum())


def mean_costs(z, bounds, cap, offsets):
    costs = {arm:[] for arm in ARMS}
    clip_count = 0
    for (front,back),(offset,_,_) in zip(bounds,offsets,strict=True):
        original = clipped_residual(front,z,cap)
        inside = clipped_residual(np.clip(z,front,back),z,cap)
        surface = front+np.minimum(offset,back-front)
        corrected = clipped_residual(surface,z,cap)
        assert np.all(inside<=original+1.e-12)
        assert np.all((surface>=front) & (surface<=back+1.e-12))
        clip_count += int(((front>0)&(back-front<offset)).sum())
        for name,loss in zip(ARMS,(original,inside,corrected),strict=True):
            costs[name].append(float(loss.sum()/len(z)))
    return costs,clip_count


def initialize(index):
    global reply_hashes
    reply_hashes = index


def process(job):
    frame,record,pins,digest = job
    uid = frame["uid"]
    try:
        assert uid == record["uid"]
        if not record["production"]:
            return dict(uid=uid,production=False)
        frozen = prediction(frame,digest,pins)
        assert roster.file_sha(frame["points"]) == frame["point_sha"]
        window = replies(frame,frozen,reply_hashes)["2d"]["value"]
        held,transform = (np.asarray(record[k],float) for k in ("held","transform"))
        assert len(held) == record["selection_rays"] >= MIN_RETURNS
        boxes = [np.asarray(record[k],float) for k in ("p","q")]
        bounds = [intervals(box,held,transform) for box in boxes]
        for (front,_),key in zip(bounds,("p_depth","q_depth"),strict=True):
            np.testing.assert_array_equal(front,record[key])
        fit = associated_fit(frame,window,held,transform)
        fit_bounds = [intervals(box,fit,transform) for box in boxes]
        estimates,n_common = estimate_offsets(fit,fit_bounds)
        old,clipped = mean_costs(held[:,2],bounds,record["cap"],estimates)
        choices = {k:(q < p-TIE) for k,(p,q) in old.items()}
        ties = {k:(abs(p-q)<=TIE) for k,(p,q) in old.items()}
        assert choices["first"] == record["choose_qwen"] == frozen["chooses_qwen"]
        if n_common < MIN_RETURNS:
            assert old["first"] == old["fit_offset"]
        pfront,qfront = (pair[0] for pair in bounds)
        return dict(uid=uid,production=True,choice=choices,tie=ties,cost=old,
                    selection_rays=len(held),selection_miss_p=int((pfront==0).sum()),
                    selection_miss_q=int((qfront==0).sum()),selection_miss_both=int(((pfront==0)&(qfront==0)).sum()),
                    fit_rays=len(fit),fit_common_hits=n_common,offsets=estimates,selection_thickness_clips=clipped)
    except Exception as error:
        raise RuntimeError(f"Could not rescore the sealed pose pair for {frame['target']} description {uid}") from error


def run(workers):
    started = perf_counter()
    protocol = roster.read(HERE/"PROTOCOL.json")
    assert not protocol["FINAL_access"] and protocol["new_Qwen_queries"]==0
    trace,source = verified_trace()
    frames,pins,checks,index,_ = inputs()
    results = []
    with gzip.open(source,"rt") as stream, ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(index,)) as pool:
        jobs = ((f,json.loads(line),pins,checks[f["uid"]]) for f,line in zip(frames,stream,strict=True))
        for output in pool.map(process,jobs,chunksize=12):
            results.append(output)
            if len(results)%1000==0:
                print("Rescored",len(results),"/9310",flush=True)
    assert len(results)==len({r["uid"] for r in results})==9310
    active = [r for r in results if r["production"]]
    assert len(active)==5910 and sum(r["choice"]["first"] for r in active)==trace["production_switches"]==2640
    path = WORK/"sealed_scores.json"
    roster.write(path,results)
    roster.write(OUT/"score_seal.json",dict(
        trace_seal=roster.sha(AUDIT/"results/ray_seal.json"), payload=roster.sha(path),
        protocol=roster.sha(HERE/"PROTOCOL.json"),code=roster.sha(__file__),
        descriptions=len(results),active=len(active),first_qwen_switches=2640,
        original_picks_and_selection_returns_verified=True,
        fit_selection_cells_disjoint=True, offsets_supported=sum(r["fit_common_hits"]>=MIN_RETURNS for r in active),
        target_annotations_opened=False, new_Qwen_queries=0, FINAL_access=False, seconds=perf_counter()-started))
    print("Sealed all three choices on the original 9310 descriptions",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=6)
    run(parser.parse_args().workers)
