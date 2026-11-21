"""Score the five sealed boxes and publish paired and per-recording summaries."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
import gzip
import io
import math
import re
import sys
from time import perf_counter

import numpy as np

from roster import ARMS, HERE, OLD, OUT, PRIVATE, SPEC, TASKS, assert_geometry_lock, panel, read, sha, write

sys.path.insert(0,str(OLD))
from score import canonical_box, score_boxes, truth_for_slot

CONFIG=HERE/"SCORING.json"
PAIRS=(("H","A"),("H","L7"),("L7","A"),("C_pca","H"),("C_ray","C_pca"),("C_ray","H"))
PLATFORMS=("waymo","drone","quad")
STATS=("acc25","acc50","miou")
EPS=1e-8
DISTANCE=("lt15","15_to_lt30","30_to_lt60","60_or_more")


def verify_seals():
    frames, input_seal=panel()
    geometry_lock=assert_geometry_lock()
    seal=read(OUT/"prediction_seal.json")
    assert seal["pins"]["input_seal"]==sha(OUT/"input_seal.json") and not seal["target_labels_read"]
    assert seal["pins"]["geometry"]==sha(HERE/"geometry_full.py")
    assert seal["pins"]["geometry_replay"]==sha(OUT/"geometry_replay.json")
    assert read(OUT/"geometry_replay.json")["geometry_lock"]==geometry_lock
    assert seal["model_seal"]==sha(OUT/"qwen_seal.json") and not seal["FINAL_access"]
    hashed=PRIVATE/"prediction_hashes.json"
    assert seal["predictions"]==sha(hashed)
    checks=read(hashed)
    assert len(checks)==len(frames)==seal["descriptions"]==9310
    assert set(checks)=={r["uid"] for r in frames}
    model=read(OUT/"qwen_seal.json")
    assert model["queries"]==18620 and model["pins"]==seal["pins"]["model"]
    assert model["pins"]["schedule"]==sha(HERE/"SCHEDULE.json")
    assert sha(OUT/"batch_replay.json")==model["batch_replay"]
    if "concurrency_replay" in model:
        assert sha(OUT/"concurrency_replay.json")==model["concurrency_replay"]
        assert model["execution_code"]==sha(HERE/"concurrent_schedule.py")
        replay=read(OUT/"concurrency_replay.json")
        assert replay["accepted"] and replay["exact_outputs"]==replay["probed_outputs"]
    assert read(SPEC)["analysis"]["pairs"]==[list(p) for p in PAIRS]
    return frames,input_seal,seal,checks


def per_description(job):
    f,checksum,pins=job
    uid=f["uid"]
    path=PRIVATE/"predictions"/(uid+".json")
    assert sha(path)==checksum
    pred=read(path)
    assert pred["uid"]==uid and pred["pins"]==pins
    items=read(f["calibration"])["ground_info"]
    truth=canonical_box(truth_for_slot(f,items))
    box=items[f["slot"]]
    category=box["class"].strip().lower()
    assert re.fullmatch(r"[a-z0-9_ -]+",category),category
    objects=set()
    for item in items:
        b=item.get("bbox_3d")
        if b and len(b) in (7,9) and all(isinstance(v,(int,float)) and math.isfinite(v) for v in b) and min(b[3:6])>0:
            objects.add(tuple(b))
    assert objects
    count=len(objects)
    distance=float(np.linalg.norm(truth[:3]))
    location=DISTANCE[sum(distance>=x for x in (15,30,60))]
    trait="5+" if count>=5 else "3-4" if count>=3 else str(count)
    boxes=pred["boxes"]
    values=[boxes[a] for a in ARMS]
    if values[0] is None:
        assert all(x is None for x in values)
        ious,err=np.zeros(len(values)),0.
    else:
        arr=np.array(values)
        assert np.isfinite(arr).all() and arr.shape==(5,7) and (arr[:,3:6]>0).all()
        ious,err=score_boxes(arr,truth)
        if not pred["full"]:
            assert boxes["A"]==boxes["H"]
        if not pred["single"]:
            assert boxes["A"]==boxes["L7"]
        if not pred["strict"]:
            assert boxes["A"]==boxes["C_pca"]
        if not pred["chooses_qwen"]:
            assert boxes["C_pca"]==boxes["C_ray"]
        else:
            assert pred["selection_rays"]>=5

    row={k:f[k] for k in ("uid","recording","target","frame")}
    row.update(category=category,distance_bucket=location,object_count_bucket=trait,cache_reuse="archived" in f)
    row.update({k:pred[k] for k in ("status","full","single","strict","chooses_qwen","selection_rays","depth_fit_bins","qwen_2d","qwen_3d")})
    row.update({"iou_"+k:float(x) for k,x in zip(ARMS,ious,strict=True)})
    return row,err


def group_arrays(rows):
    vals=np.array([[r["iou_"+a] for a in ARMS] for r in rows])
    return vals,np.stack((vals>.25,vals>.50,vals),axis=-1)


def table_row_metrics(rows, group_kind):
    ious,values=group_arrays(rows)
    if group_kind=="macro":
        m=np.mean([values[[r["target"]==t for r in rows]].mean(0) for t in PLATFORMS],axis=0)
    else:
        m=values.mean(0)
    for i,a in enumerate(ARMS):
        yield dict(arm=a,n=len(rows),recordings=len({r["recording"] for r in rows}),
            hits25=int(sum(ious[:,i]>.25)),hits50=int(sum(ious[:,i]>.5)),**dict(zip(STATS,100*m[i],strict=True)))


def table_row_pairs(rows, group_kind):
    ious,values=group_arrays(rows)
    for a,b in PAIRS:
        i,j=ARMS.index(a),ARMS.index(b)
        va,vb=ious[:,i],ious[:,j]
        delta=values[:,i]-values[:,j]
        if group_kind=="macro":
            average=np.mean([delta[[r["target"]==t for r in rows]].mean(0) for t in PLATFORMS],axis=0)
        else:
            average=delta.mean(0)
        abs_diff=va-vb
        p=dict(arm=a,reference=b,n=len(rows),recordings=len({r["recording"] for r in rows}),better_iou=int(sum(abs_diff>EPS)),
               worse_iou=int(sum(abs_diff<-EPS)),same_iou=int(sum(abs(abs_diff)<=EPS)),
               **{f"delta_{k}_pp":float(100*v) for k,v in zip(STATS,average,strict=True)})
        for t,name in ((.25,"25"),(.50,"50")):
            p["rescue"+name]=int(sum((va>t)&(vb<=t)))
            p["harm"+name]=int(sum((vb>t)&(va<=t)))
        yield p


def bootstraps(rows):
    rng=np.random.default_rng(20261121)
    samples={}
    for target in PLATFORMS:
        groups=defaultdict(list)
        for r in rows:
            if r["target"]==target:
                groups[r["recording"]].append(r)
        ordered=[groups[k] for k in sorted(groups)]
        counts=np.array([len(g) for g in ordered])
        sums=np.array([group_arrays(g)[1].sum(0) for g in ordered])
        take=rng.integers(len(ordered),size=(4000,len(ordered)))
        samples[target]=sums[take].sum(1)/counts[take].sum(1)[:,None,None]
    samples["macro"]=np.mean(list(samples.values()),axis=0)
    ci={}
    for target,data in samples.items():
        for a,b in PAIRS:
            i,j=ARMS.index(a),ARMS.index(b)
            differences=100*(data[:,i]-data[:,j])
            ci[f"{target}/{a}-{b}"]=dict(zip(STATS,np.quantile(differences,[.025,.975],axis=0).T.tolist(),strict=True))
    return ci


def write_rows(name,rows,compress=False):
    path=OUT/name
    if compress:
        with path.open("wb") as fp:
            with gzip.GzipFile(filename="",fileobj=fp,mode="wb",mtime=0) as gz:
                with io.TextIOWrapper(gz,encoding="utf-8",newline="") as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    else:
        with path.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return sha(path)


def run(workers):
    t0=perf_counter()
    frames,root,seal,checks=verify_seals()
    config=read(CONFIG)
    assert config["thresholds"]==[.25,.5] and not config["FINAL_access"]
    assert config["descriptive_strata"]["target_distance_m_edges"]==[15,30,60]
    print("Verified, proceeding to scoring",len(frames),"sealed records.",flush=True)
    rows=[]
    maximum=0.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=((f,checks[f["uid"]],seal["pins"]) for f in frames)
        for r,error in pool.map(per_description,jobs,chunksize=24):
            rows.append(r)
            maximum=max(maximum,error)
            if len(rows)%2000==0:
                print("Scored",len(rows),"/",len(frames),flush=True)
    assert len(rows)==9310 and maximum < 1e-7
    assert dict(Counter(r["target"] for r in rows))==root["descriptions_by_platform"]

    groups={t:[r for r in rows if r["target"]==t] for t in PLATFORMS}
    groups.update(macro=rows,pooled=rows)
    metrics=[dict(platform=t,**m) for t,g in groups.items() for m in table_row_metrics(g,t)]
    pairs=[dict(platform=t,**p) for t,g in groups.items() for p in table_row_pairs(g,t)]
    recordmap=defaultdict(list)
    for r in rows:
        recordmap[(r["target"],r["recording"])].append(r)
    per_recording=[]
    for (t,rec),g in sorted(recordmap.items()):
        for p in table_row_pairs(g,t):
            per_recording.append(dict(platform=t,recording=rec,**p))
    recording_metrics=[dict(platform=t,recording=rec,**m) for (t,rec),g in sorted(recordmap.items())
                       for m in table_row_metrics(g,t)]

    breakdown=[]
    for field in ("category","distance_bucket","object_count_bucket","status"):
        for t in (*PLATFORMS,"pooled"):
            strata=defaultdict(list)
            for r in groups[t]:
                strata[r[field]].append(r)
            for k,g in sorted(strata.items()):
                breakdown.extend(dict(platform=t,group=field,value=k,**m) for m in table_row_metrics(g,t))
    joint=[r for r in rows if r["full"] and r["single"]]
    selectable=[r for r in rows if r["selection_rays"]>=5]
    mechanism=[]
    for name,data in (("H_L7_joint_support",joint),("pose_selection_available",selectable)):
        for t in (*PLATFORMS,"pooled","macro"):
            g=[r for r in data if t in ("macro","pooled",r["target"])]
            if g:
                mechanism.extend(dict(platform=t,group=name,**p) for p in table_row_pairs(g,t))

    files={
        "metrics.csv":write_rows("metrics.csv",metrics),
        "paired.csv":write_rows("paired.csv",pairs),
        "recording_pairs.csv":write_rows("recording_pairs.csv",per_recording),
        "recording_metrics.csv":write_rows("recording_metrics.csv",recording_metrics),
        "strata.csv":write_rows("strata.csv",breakdown),
        "eligible_pairs.csv":write_rows("eligible_pairs.csv",mechanism)
    }
    for t in PLATFORMS:
        name=f"rows_{t}.csv.gz"
        files[name]=write_rows(name,groups[t],compress=True)
    receipts=dict(prediction_seal=sha(OUT/"prediction_seal.json"), scoring_protocol=sha(CONFIG), code=sha(__file__),
        descriptions=len(rows),recordings=len(recordmap),files=files,
        coverage={t:{k:sum(bool(r[k]) for r in groups[t]) for k in ("full","single","strict","chooses_qwen","cache_reuse")} for t in PLATFORMS},
        qwen_response_status={t:{k:dict(Counter(r[k] for r in groups[t])) for k in ("qwen_2d","qwen_3d","status")} for t in PLATFORMS},
        max_two_upright_iou_discrepancy=maximum, intervals_pp=bootstraps(rows), ordinary_exposed=True,FINAL_access=False,seconds=perf_counter()-t0)
    write(OUT/"evaluation_receipt.json",receipts)
    print("Fully scored and committed",len(rows),"rows; IoU algorithms agree to",maximum,flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workers",type=int,default=4)
    run(cli.parse_args().workers)
