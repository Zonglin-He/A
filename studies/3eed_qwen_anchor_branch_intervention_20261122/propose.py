"""Generate label-free horizontal hypotheses by intervening on anchor support."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
OUT, WORK = HERE/"results", HERE/"private"
PROTOCOL = HERE/"PROTOCOL.json"
PAYLOAD = WORK/"predictions.jsonl.gz"
PRIOR = HERE.parent/"3eed_qwen_cafe_exclusion_audit_20261121"
SURFACE = HERE.parent/"3eed_qwen_cray_surface_rescoring_20261121"
sys.path.insert(0, str(PRIOR))
import extract as origin
from center_sets import axes, local
from trace_pairs import clipped_residual, first_depth


roster, geometry = origin.roster, origin.geometry
MIN_RETURNS, SCALES = origin.MIN_RETURNS, origin.FOUR_SCALES
FAMILIES = ("branch", "tail", "axis")
ARMS = ("baseline", "selection_only", *FAMILIES)
TIE = 1.e-12
MOVEMENT_TOL = 1.e-9


def get_support(frame, window, locked, matrix):
    spec, size, camera = geometry.camera(frame)
    np.testing.assert_array_equal(camera,matrix)
    uv, returns = geometry.raw_returns(frame,spec,size)
    window = np.asarray(window)*np.tile(size,2)/1000.
    center, cells = geometry.z_buffer(uv,returns,np.flatnonzero(geometry.inside(uv,window,.5)))
    central = center[geometry.reserve(cells)==0]
    ids, cells = geometry.z_buffer(uv,returns,np.arange(len(uv)))
    partition = geometry.reserve(cells)
    np.testing.assert_array_equal(partition>0,geometry.held_cells(cells))
    fit = ids[partition==0]
    in_window = geometry.inside(uv[ids],window)
    select = ids[(partition==1)&in_window]
    audit = ids[(partition==2)&in_window]
    model_to_camera = geometry.model_to_camera_matrix(spec)
    xyz = geometry.rigid_xyz(returns,np.linalg.inv(model_to_camera))
    sample, check = geometry.fit_sample(frame,spec,size)
    assert not check["retained_withheld_raw"]
    ground = geometry.regions.terrain(sample)
    support, target = origin.choose_region(sample,ground,SCALES,xyz[np.r_[fit,central]],in_window[partition==0],
                                           xyz[np.r_[select,audit]])
    assert support is not None
    hold = returns[select[target[:len(select)]]]
    untouched = returns[audit[target[len(select):]]]
    np.testing.assert_array_equal(hold,locked)

    radii, labels, births = origin.region_catalog(sample,ground,SCALES)
    tree = cKDTree(sample)
    _, seeds = tree.query(support)
    birth, = [k for k,node in enumerate(labels[:,seeds[0]]) if node >= 0 and births[node]==k
              and (labels[k,seeds]==node).all() and (labels[k]==node).sum()==len(seeds)]
    assert len(seeds)==len(set(seeds)) and np.array_equal(sample[seeds],support)
    distance, neighbors = tree.query(xyz[central])
    vote = np.isin(neighbors,seeds) & (distance<=radii[birth])
    assert vote.sum()>=MIN_RETURNS
    counts = np.bincount(neighbors[vote],minlength=len(sample))[seeds]
    inv = np.linalg.inv(matrix)
    raw = geometry.rigid_xyz(support,inv@model_to_camera)
    children = labels[birth-1,seeds] if birth else np.full(len(seeds),-1,dtype=int)
    return raw, children, counts, hold, untouched, inv[:3,3], SCALES[birth]


def anchor(lo, hi, eye, half):
    return np.where(eye<lo,lo+half,np.where(eye>hi,hi-half,(lo+hi)/2.))


def proposals(box, support, children, votes, viewpoint):
    coordinates = local(support,box)[:,:2]
    eye = local(viewpoint[None],box)[0,:2]
    half = box[3:5]/2.
    lo, hi = coordinates.min(0), coordinates.max(0)
    original = anchor(lo,hi,eye,half)
    np.testing.assert_allclose(original,0.,atol=5.e-9,rtol=0.)
    groups = {}
    for axis in range(2):
        extremes = [(1, lo[axis])] if eye[axis] < lo[axis] else [(-1, hi[axis])] if eye[axis] > hi[axis] \
            else [(1, lo[axis]), (-1, hi[axis])]
        for sign,knot in extremes:
            owners = np.unique(children[coordinates[:,axis]==knot])
            if len(owners)==1 and owners[0]>=0:
                groups.setdefault(int(owners[0]),(axis,sign))

    def shift(mask):
        points = coordinates[mask]
        return anchor(points.min(0),points.max(0),eye,half)

    def moved(offset):
        result = box.copy()
        result[:2] += offset @ axes(box[6])[:2,:2].T
        return result.tolist()

    result = {a:[] for a in FAMILIES}
    for child,(axis,sign) in sorted(groups.items()):
        keep = children!=child
        if votes[keep].sum()<MIN_RETURNS:
            continue
        delta = shift(keep)
        if np.linalg.norm(delta)<=MOVEMENT_TOL:
            continue
        count = int((~keep).sum())
        result["branch"].append(dict(box=moved(delta),removed=count,axis=axis,shift=float(np.linalg.norm(delta)),
                                     both_axes=bool((abs(delta)>MOVEMENT_TOL).all())))
        single = np.zeros(2)
        single[axis] = delta[axis]
        if np.linalg.norm(single)>MOVEMENT_TOL:
            result["axis"].append(dict(box=moved(single),removed=count,axis=axis,shift=float(np.linalg.norm(single))))
        retained = np.argsort(sign*coordinates[:,axis],kind="stable")[count:]
        if votes[retained].sum()>=MIN_RETURNS:
            endpoint = shift(retained)
            if np.linalg.norm(endpoint)>MOVEMENT_TOL:
                result["tail"].append(dict(box=moved(endpoint),removed=count,axis=axis,shift=float(np.linalg.norm(endpoint))))
    assert all(len(x)<=4 for x in result.values())
    return result


def cost(box, rays, transform, cap):
    return float(clipped_residual(first_depth(np.asarray(box),rays,transform),rays[:,2],cap).mean())


def setup(index):
    global replies_index
    replies_index = index


def predict_one(job):
    frame, frozen, checksum, pins = job
    uid = frame["uid"]
    assert uid == frozen["uid"] and frame["panel"] == "validation"
    if not frozen["production"]:
        return dict(uid=uid, active=False)
    assert roster.file_sha(frame["points"]) == frame["point_sha"]
    archived = origin.prediction(frame,checksum,pins)
    assert archived["chooses_qwen"] == frozen["choose_qwen"]
    b = np.array(frozen["production_box"],float)
    key = "q" if frozen["choose_qwen"] else "p"
    np.testing.assert_allclose(b,frozen[key],atol=5.e-10,rtol=0.)
    box = np.array(frozen[key],float)
    locked, transform = (np.asarray(frozen[k],float) for k in ("held","transform"))
    window = origin.replies(frame,archived,replies_index)["2d"]["value"]
    support, children, votes, selection, audit, eye, level = get_support(frame,window,locked,transform)
    assert len(selection)==frozen["selection_rays"]>=MIN_RETURNS
    options = proposals(box,support,children,votes,eye)
    cap = frozen["cap"]
    original = cost(box,selection,transform,cap)
    np.testing.assert_allclose(original,clipped_residual(np.asarray(frozen[key+"_depth"]),selection[:,2],cap).mean(),rtol=0.,atol=TIE)
    valid_audit = len(audit)>=MIN_RETURNS
    audit_base = cost(box,audit,transform,cap) if valid_audit else None
    chosen = dict.fromkeys(ARMS)
    for family,variants in options.items():
        if not variants:
            continue
        for v in variants:
            v["selection_delta"] = cost(v["box"],selection,transform,cap)-original
            v["audit_delta"] = cost(v["box"],audit,transform,cap)-audit_base if valid_audit else None
            np.testing.assert_array_equal(v["box"][2:],box[2:])
        best = min(range(len(variants)),key=lambda i:variants[i]["selection_delta"])
        v = variants[best]
        if v["selection_delta"] < -TIE:
            if family=="branch":
                chosen["selection_only"] = best
            if valid_audit and v["audit_delta"] < -TIE:
                chosen[family] = best
    return dict(uid=uid, active=True, box=box.tolist(), use_qwen=bool(frozen["choose_qwen"]), scale=level,
                fit_seeds=len(support), original_selection=len(selection), audit_cells=len(audit),
                choices=chosen, options=options)


def run(workers):
    start = perf_counter()
    protocol = roster.read(PROTOCOL)
    lock = roster.sha(origin.ACTIVE/"METHOD_LOCK.json")
    assert lock == protocol["frozen_method_sha256"] and not protocol["FINAL_access"]
    trace, payload = origin.verified_trace()
    frames,pins,checks,index,_ = origin.inputs()
    rows = 0
    active = selected = proposed = 0
    tmp = PAYLOAD.with_suffix(".tmp")
    with gzip.open(payload,"rt") as stream, tmp.open("wb") as raw, \
            gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0,compresslevel=4) as out, \
            ProcessPoolExecutor(max_workers=workers,initializer=setup,initargs=(index,)) as pool:
        jobs = ((frame,json.loads(line),checks[frame["uid"]],pins) for frame,line in zip(frames,stream,strict=True))
        for r in pool.map(predict_one,jobs,chunksize=10):
            out.write((json.dumps(r,separators=(",",":"),allow_nan=False)+"\n").encode())
            rows += 1
            active += r["active"]
            selected += r.get("original_selection",0)
            proposed += bool(r.get("options",{}).get("branch"))
            if rows%1000==0:
                print("Replayed",rows,"/ 9310",flush=True)
    assert rows==9310 and active==5910 and selected==688114
    tmp.replace(PAYLOAD)
    roster.write(OUT/"prediction_seal.json",dict(
        descriptions=rows, active=active, selection_cells=selected, connected_deletion_proposed=proposed,
        protocol=roster.sha(PROTOCOL), code=roster.sha(__file__), payload=roster.sha(PAYLOAD),
        trace=trace["payload"], locked_method=lock, untouched_audit_split=True, target_gt_opened=False,
        new_Qwen_queries=0, production_prediction_changes=0, FINAL_access=False, seconds=perf_counter()-start,
    ))
    print("Sealed",rows,"descriptions with proposals on",proposed,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=8)
    run(parser.parse_args().workers)
