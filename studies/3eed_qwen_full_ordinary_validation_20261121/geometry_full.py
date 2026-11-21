"""Run the five locked geometry readouts, reusing the original proposal operators."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import os
from pathlib import Path
import sys
from time import perf_counter, sleep

import numpy as np

from roster import (ARMS, HERE, OUT, POSE, PRIVATE, TASKS, assert_geometry_lock, file_sha, panel,
                    read, sha, write)

sys.path.insert(0,str(POSE))
import surface_grounding as prior


def predict(frame, replies):
    raw, pixel = (replies[t]["value"] for t in TASKS)
    boxes = dict.fromkeys(ARMS)
    data = dict(boxes=boxes, full=False, single=False, strict=False, chooses_qwen=False,
                selection_rays=0, isolation={}, status="missing_3d", depth_fit_bins=0,
                qwen_2d=replies["2d"]["status"], qwen_3d=replies["3d"]["status"])
    if raw is None:
        return data
    spec,size,transform = prior.camera(frame)
    uv,returns = prior.raw_returns(frame,spec,size)
    base, summary, _ = prior.variants(np.asarray(raw,float),pixel,spec,size,uv,returns)
    initial, _ = prior.to_sensor(base["ray_depth"],transform)
    data["depth_fit_bins"] = summary["fit_bins"]
    boxes.update(dict.fromkeys(ARMS,initial.tolist()))
    if pixel is None:
        data["status"] = "missing_2d"
        return data

    target,view = frame["target"],frame["suffix"]
    dims,qyaw = initial[3:6],initial[6]
    qyaw_model = qyaw + (np.deg2rad(prior.WAYMO_ANGLES[view]) if target=="waymo" else 0.)
    transform_model = prior.model_to_camera_matrix(spec)
    xyz = prior.rigid_xyz(returns,np.linalg.inv(transform_model))
    window = np.asarray(pixel)*np.tile(size,2)/1000.
    indices,cells = prior.z_buffer(uv,returns,np.arange(len(uv)))
    parts = prior.reserve(cells)
    np.testing.assert_array_equal(parts>0,prior.held_cells(cells))
    ids,bins = prior.z_buffer(uv,returns,np.flatnonzero(prior.inside(uv,window,.5)))
    central = ids[prior.reserve(bins)==0]
    if len(central)<5:
        data["status"] = "insufficient_original_center"
        return data

    def raw_box(b):
        return prior.sensor_frame_box(b,target,view).tolist()

    train_ids = indices[parts==0]
    in_window = prior.inside(uv[train_ids],window)
    combined = xyz[np.r_[train_ids,central]]
    sample = prior.model_sample(frame)
    catalog = prior.partition(sample)
    picked = prior.tree_winner(sample,catalog,combined,in_window,len(central))
    plane = prior.regions.terrain(sample)
    if picked is not None:
        support = prior.regions.measured_box(picked["points"],plane)
        boxes["H"] = raw_box(prior.regions.complete(support,dims))
        data["full"] = True

    foreground = sample[:,2]-sample[:,:2]@plane[:2]-plane[2] > .20
    radii, labels = prior.regions.levels(sample,foreground)
    k=prior.regions.FOUR_LEVEL_INDICES.index(7)
    dist, idx = prior.cKDTree(sample).query(combined)
    single = prior.one_level(sample,labels[k],radii[k],idx,dist,in_window,len(central))
    if single is not None:
        boxes["L7"] = raw_box(prior.regions.complete(prior.regions.measured_box(single,plane),dims))
        data["single"] = True

    fit,isolation = prior.fit_sample(frame,spec,size)
    data["isolation"] = isolation
    data["status"] = "no_fit_node"
    if len(fit)<9:
        return data
    fit_catalog = prior.partition(fit)
    chosen = prior.tree_winner(fit,fit_catalog,combined,in_window,len(central))
    if chosen is None:
        return data
    fit_plane = prior.regions.terrain(fit)
    measured = prior.regions.measured_box(chosen["points"],fit_plane)
    exact_pca = prior.regions.complete(measured,dims)
    pca_yaw = prior.to_long_axis(exact_pca)[6]
    eye = np.linalg.inv(transform_model)[:3,3]
    p = raw_box(prior.fixed_size_anchor(chosen["points"],fit_plane,pca_yaw,dims,eye))
    q = raw_box(prior.fixed_size_anchor(chosen["points"],fit_plane,qyaw_model,dims,eye))
    boxes.update(C_pca=p,C_ray=p)
    cuts,assign,births,_ = fit_catalog
    withheld = indices[prior.inside(uv[indices],window) & (parts>0)]
    attached = prior.transfer(fit,xyz[withheld],cuts,assign,births)[chosen["level"]]==chosen["node"]
    attached_ids = withheld[attached]
    reserved_cells = (uv[attached_ids]//4).astype(np.int64)
    split = prior.reserve(reserved_cells)
    assert (split>0).all() and len(np.unique(reserved_cells,axis=0))==len(reserved_cells)
    selector = attached_ids[split==1]
    data.update(strict=True, status="too_few_selection_rays", selection_rays=len(selector))
    if len(selector)>=5:
        clip = max(.1,.5*min(dims[:2]))
        costs = [float(cost.mean()) for _,_,cost in prior.disagreement((p,q),returns[selector],transform,clip)]
        accepted = costs[1]<costs[0]-1.e-12
        data.update(status="usable_pair", chooses_qwen=bool(accepted), selector_cost=costs)
        if accepted:
            boxes["C_ray"] = q
    return data


def parity_worker(item):
    frame, reply, expected = item
    found = predict(frame,reply)
    discrepancy = 0.
    for arm in ARMS:
        old = expected["boxes"]["P" if arm=="H" else arm]
        got = found["boxes"][arm]
        if old is None or got is None:
            assert got is old
        else:
            np.testing.assert_array_equal(got,old)
            discrepancy = max(discrepancy,float(np.max(np.abs(np.asarray(got)-old))))
    assert found["full"]==expected["full"] and found["strict"]==expected["strict"]
    assert found["selection_rays"]==expected["selection_rays"]
    assert found["chooses_qwen"]==expected["chooses"].get("C",{}).get("qwen",False)
    return discrepancy


def parity(workers):
    geometry_lock=assert_geometry_lock()
    previous_seal = read(POSE/"results/prediction_seal.json")
    source = POSE/"private/predictions.json"
    assert sha(source)==previous_seal["predictions"] and not previous_seal["target_labels_read"]
    archived = read(source)
    old_panel,_,_,hashes = prior.load_prior()
    jobs=[]
    for frame in old_panel:
        uid=frame["uid"]
        replies={}
        for task in TASKS:
            tag=uid+"_"+task
            path=prior.BASE/"private/replies"/(tag+".json")
            assert sha(path)==hashes[tag]
            replies[task]=read(path)
        jobs.append((frame,replies,archived[uid]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        error=max(pool.map(parity_worker,jobs,chunksize=5))
    write(OUT/"geometry_replay.json",dict(frames=len(jobs),arm_box_checks=len(jobs)*len(ARMS),
        max_coordinate_difference_m=error, source=sha(POSE/"results/prediction_seal.json"),
        code=sha(__file__), geometry_lock=geometry_lock, target_labels_read=False, FINAL_access=False))
    print("Geometry parity",len(jobs), "frames",len(jobs)*len(ARMS),"boxes; maximum error",error,flush=True)


def available(frame):
    return "archived" in frame or all((PRIVATE/"replies"/(frame["uid"]+"_"+t+".json")).is_file() for t in TASKS)


def geometry_worker(job):
    frame,pins=job
    uid=frame["uid"]
    assert file_sha(frame["points"])==frame["point_sha"]
    assert file_sha(frame["rgb"])==frame["rgb_sha"]
    responses, lineage = {}, {}
    for task in TASKS:
        if "archived" in frame:
            response=frame["archived"][task]
            stamp=response["source_sha256"]
        else:
            p=PRIVATE/"replies"/(uid+"_"+task+".json")
            response=read(p)
            assert response["uid"]==uid and response["task"]==task and response["pins"]==pins["model"]
            stamp=sha(p)
        responses[task]=response
        lineage[task]=stamp
    result=predict(frame,responses)
    path=PRIVATE/"predictions"/(uid+".json")
    write(path,dict(uid=uid,pins=pins,replies=lineage,**result))


def run(workers, producer_pid):
    t0=perf_counter()
    frames,input_seal=panel()
    geometry_lock=assert_geometry_lock()
    pre=read(OUT/"geometry_replay.json")
    assert pre["code"]==sha(__file__) and pre["geometry_lock"]==geometry_lock and pre["max_coordinate_difference_m"]==0.
    (PRIVATE/"predictions").mkdir(exist_ok=True,mode=0o700)
    audit=read(OUT/"batch_replay.json")
    pins=dict(model=audit["pins"], geometry=sha(__file__), input_seal=sha(OUT/"input_seal.json"),
              geometry_replay=sha(OUT/"geometry_replay.json"))
    pending=[]
    done=0
    for f in frames:
        path=PRIVATE/"predictions"/(f["uid"]+".json")
        if path.is_file():
            cached=read(path)
            assert cached["uid"]==f["uid"] and cached["pins"]==pins
            done+=1
        else:
            pending.append(f)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending:
            current=[]
            for frame in pending:
                if available(frame):
                    current.append(frame)
                    if len(current)==192:
                        break
            if not current:
                if (OUT/"qwen_seal.json").exists():
                    raise RuntimeError("Sealed model run has missing replies")
                if producer_pid and not Path(f"/proc/{producer_pid}").is_dir():
                    raise RuntimeError("Model worker stopped before sealing. Existing geometry checkpoints are preserved.")
                sleep(5)
                continue
            list(pool.map(geometry_worker,((f,pins) for f in current),chunksize=3))
            covered={f["uid"] for f in current}
            pending=[f for f in pending if f["uid"] not in covered]
            done+=len(current)
            print("Geometry done",done,"/",len(frames), "elapsed",round(perf_counter()-t0),flush=True)

    while not (OUT/"qwen_seal.json").is_file():
        if producer_pid and not Path(f"/proc/{producer_pid}").is_dir():
            raise RuntimeError("Model worker stopped before sealing.")
        sleep(2)
    model=read(OUT/"qwen_seal.json")
    assert model["queries"]==2*len(frames)==18620 and model["pins"]==pins["model"]
    bank=PRIVATE/"fresh_reply_hashes.json"
    assert sha(bank)==model["responses"] and not model["target_labels_read"]
    replies=read(bank)
    outputs={}
    counter=Counter()
    isolation=Counter()
    for f in frames:
        uid=f["uid"]
        path=PRIVATE/"predictions"/(uid+".json")
        r=read(path)
        assert r["uid"]==uid and r["pins"]==pins
        for task in TASKS:
            h=f["archived"][task]["source_sha256"] if "archived" in f else replies[uid+"_"+task]
            assert r["replies"][task]==h
        outputs[uid]=sha(path)
        counter[(f["target"],r["status"])]+=1
        isolation.update(r["isolation"])
    assert not isolation["retained_withheld_raw"]
    write(PRIVATE/"prediction_hashes.json",outputs)
    write(OUT/"prediction_seal.json",dict(pins=pins, model_seal=sha(OUT/"qwen_seal.json"), frames=input_seal["frames"],
        descriptions=len(outputs), predictions=sha(PRIVATE/"prediction_hashes.json"),
        coverage={t:{s:n for (k,s),n in counter.items() if k==t} for t in ("waymo","drone","quad")},
        held_return_isolation=dict(isolation), target_labels_read=False, FINAL_access=False,
        seconds=perf_counter()-t0))
    print("Sealed all",len(outputs),"five-arm comparisons",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("parity","run"))
    cli.add_argument("--workers",type=int,default=4)
    cli.add_argument("--producer-pid",type=int)
    args=cli.parse_args()
    if args.stage=="parity":
        parity(args.workers)
    else:
        run(args.workers,args.producer_pid)
