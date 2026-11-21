"""Test whether intrinsic axis allocation helps beyond certified shrink at the same volume."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
OUT, WORK = HERE/"results", HERE/"private"
PROTOCOL = HERE/"PROTOCOL.json"
PAYLOAD = WORK/"matched_predictions.jsonl.gz"
PREVIOUS = HERE.parent/"3eed_qwen_ray_equivalent_extent_20261122"
sys.path.insert(0,str(PREVIOUS))
import run_trial as source
from extent import TOL, allowed_fraction, contained, shrink_box, wall_motion


roster,geometry,origin = source.roster,source.geometry,source.origin
VARIANTS = ("uniform","height_first")
ARMS = ("baseline","ray_equivalent",*VARIANTS)
PAIRS = (("ray_equivalent","baseline"),)+tuple((a,"baseline") for a in VARIANTS)+tuple(("ray_equivalent",a) for a in VARIANTS)


def uniform_fraction(limits,volume):
    assert np.prod(1.-limits) <= volume+1.e-11 and volume<=1.+1.e-12
    lower,upper = 0.,float(limits.max())
    for _ in range(58):
        middle = (lower+upper)/2.
        if np.prod(1.-np.minimum(limits,middle)) > volume:
            lower = middle
        else:
            upper = middle
    return np.minimum(limits,lower)


def match_one(job):
    frame,trace,saved,pins,check = job
    uid = frame["uid"]
    assert uid==trace["uid"]==saved["uid"] and frame["panel"]=="validation"
    proposal = saved.get("variants",{}).get("ray_equivalent")
    if proposal is None:
        return dict(uid=uid,changed=False)
    pred = origin.prediction(frame,check,pins)
    pixel = origin.replies(frame,pred,source.replies_hashes)["2d"]["value"]
    matrix,held = (np.asarray(trace[k],float) for k in ("transform","held"))
    box = np.asarray(saved["box"],float)
    key = "q" if trace["choose_qwen"] else "p"
    np.testing.assert_array_equal(box,trace[key])
    support,_,_,selection,audit,_,_ = source.get_support(frame,pixel,held,matrix)
    inverse = np.linalg.inv(matrix)
    measured = geometry.rigid_xyz(selection,inverse)
    inside = np.concatenate((support[contained(support,box)],measured[contained(measured,box)]))
    depth = np.asarray(trace[key+"_depth"],float)
    hits = depth>0.
    entries = geometry.rigid_xyz(selection[hits]*(depth[hits]/selection[hits,2])[:,None],inverse)
    mandatory = source.local(np.concatenate((entries,inside)),box)
    advance,retreat = wall_motion(box,np.zeros(3),inverse[:3,3])
    limits = np.minimum(allowed_fraction(box[3:6]/2.,advance,retreat,mandatory),1.-1.e-12)
    volume = proposal["volume_fraction"]
    np.testing.assert_allclose(volume,np.prod(np.asarray(proposal["box"])[3:6]/box[3:6]),rtol=0.,atol=1.e-12)
    assert ((np.asarray(proposal["trim"])/box[3:6]) <= limits+1.e-10).all()
    vertical = min(limits[2],1.-volume)
    fractions = dict(uniform=uniform_fraction(limits,volume),
                     height_first=np.r_[uniform_fraction(limits[:2],volume/(1.-vertical)),vertical])
    options = {}
    audit_depth = source.first_depth(box,audit,matrix)
    for arm,fraction in fractions.items():
        candidate = shrink_box(box,advance,retreat,fraction)
        np.testing.assert_allclose(np.prod(candidate[3:6]/box[3:6]),volume,rtol=0.,atol=5.e-12)
        assert contained(inside,candidate).all()
        assert source.invariance(depth,source.first_depth(candidate,selection,matrix))[0]==0
        n_audit,_ = source.invariance(audit_depth,source.first_depth(candidate,audit,matrix))
        options[arm] = dict(box=candidate.tolist(), audit_delta_cells=n_audit,
                            trim_height=float(box[5]-candidate[5]))
    return dict(uid=uid,changed=True,volume=volume,boxes=options,certified=proposal["box"],
                audit_cells=len(audit),orig=box.tolist())


def propose(workers):
    started = perf_counter()
    p = PREVIOUS/"results/prediction_seal.json"
    pin = roster.read(p)
    assert not pin["target_gt_opened"] and not pin["FINAL_access"]
    assert pin["payload"]==roster.sha(source.PAYLOAD) and pin["solver"]==roster.sha(PREVIOUS/"extent.py")
    assert pin["code"]==roster.sha(PREVIOUS/"run_trial.py") and not roster.read(PROTOCOL)["FINAL_access"]
    trace,first = origin.verified_trace()
    assert trace["payload"]==pin["trace"]
    frames,pins,checks,index,_ = origin.inputs()
    count = change = 0
    temp = PAYLOAD.with_suffix(".tmp")
    with gzip.open(first,"rt") as raw, gzip.open(source.PAYLOAD,"rt") as saved, temp.open("wb") as f, \
            gzip.GzipFile(filename="",fileobj=f,mode="wb",mtime=0,compresslevel=4) as stream, \
            ProcessPoolExecutor(max_workers=workers,initializer=source.initialize,initargs=(index,)) as pool:
        jobs = ((frame,json.loads(t),json.loads(d),pins,checks[frame["uid"]]) for frame,t,d in zip(frames,raw,saved,strict=True))
        for result in pool.map(match_one,jobs,chunksize=12):
            stream.write((json.dumps(result,separators=(",",":"),allow_nan=False)+"\n").encode())
            count += 1
            change += result["changed"]
            if count%1000==0:
                print("Matched",count,"/ 9310",flush=True)
    assert count==9310 and change==5904
    temp.replace(PAYLOAD)
    roster.write(OUT/"prediction_seal.json",dict(
        descriptions=count,changed=change,per_description_volume_matched=True,selection_entries_preserved=True,
        measured_inliers_preserved=True,source_predictions=roster.sha(p),protocol=roster.sha(PROTOCOL),
        code=roster.sha(__file__),payload=roster.sha(PAYLOAD),target_gt_opened=False,FINAL_access=False,
        production_prediction_changes=0,new_Qwen_queries=0,seconds=perf_counter()-started,
    ))


def joined(job):
    frame,sealed,old = job
    assert frame["uid"]==sealed["uid"]==old["uid"] and frame["panel"]=="validation"
    result = {key:old[key] for key in ("uid","target","category","recording","frame","case")}
    baseline = float(old["iou_baseline"])
    certified = float(old["iou_ray_equivalent"])
    result.update(changed=sealed["changed"],iou_baseline=baseline,iou_ray_equivalent=certified,
                  iou_uniform=baseline,iou_height_first=baseline,audit_cells=None,
                  audit_changed_certified=None,audit_changed_uniform=None,audit_changed_height_first=None)
    assert sealed["changed"]==(old["changed_ray_equivalent"]=="True")
    if not sealed["changed"]:
        return result
    path = Path(frame["calibration"])
    assert path.name=="meta_info.json" and path.parents[2]==source.DATA/frame["target"]
    items=roster.read(path)["ground_info"]
    truth=source.score_full.canonical_box(source.score_full.truth_for_slot(frame,items))
    boxes=[sealed["orig"],sealed["certified"]]+[sealed["boxes"][arm]["box"] for arm in VARIANTS]
    values,_ = source.score_full.score_boxes(np.asarray(boxes),truth)
    np.testing.assert_allclose(values[:2],(baseline,certified),atol=1.e-12,rtol=0.)
    result.update(audit_cells=sealed["audit_cells"],audit_changed_certified=int(old["ray_equivalent_audit_changes"]))
    for arm,score in zip(VARIANTS,values[2:],strict=True):
        result[f"iou_{arm}"]=float(score)
        result[f"audit_changed_{arm}"]=sealed["boxes"][arm]["audit_delta_cells"]
    return result


def evaluate(workers):
    started = perf_counter()
    seal_path = OUT/"prediction_seal.json"
    seal = roster.read(seal_path)
    assert not seal["target_gt_opened"] and not seal["FINAL_access"]
    assert seal["protocol"]==roster.sha(PROTOCOL) and seal["code"]==roster.sha(__file__) and seal["payload"]==roster.sha(PAYLOAD)
    receipt = roster.read(PREVIOUS/"results/evaluation_receipt.json")
    assert receipt["prediction_seal"]==seal["source_predictions"]
    old={}
    for platform in source.score_full.PLATFORMS:
        name=f"rows_{platform}.csv.gz"
        path=PREVIOUS/"results"/name
        assert receipt["files"][name]==roster.sha(path)
        with gzip.open(path,"rt",newline="") as f:
            old.update({r["uid"]:r for r in csv.DictReader(f)})
    frames,_,_,_,_ = origin.inputs()
    with gzip.open(PAYLOAD,"rt") as f, ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=((frame,json.loads(line),old[frame["uid"]]) for frame,line in zip(frames,f,strict=True))
        rows=list(pool.map(joined,jobs,chunksize=10))
    assert len(rows)==len({r["uid"] for r in rows})==len(old)==9310 and sum(r["changed"] for r in rows)==5904
    module=source.score_full
    module.ARMS,module.PAIRS,module.OUT=ARMS,PAIRS,OUT
    groups=defaultdict(list)
    for r in rows:
        keys=[("macro","all"),("platform",r["target"]),("recording",r["target"]+"/"+r["recording"])]
        if r["changed"]:
            keys.append(("coverage","changed"))
        for key in keys:
            groups[key].append(r)
    metrics,paired=[],[]
    for (kind,value),items in sorted(groups.items()):
        tags=dict(group=kind,value=value)
        metrics.extend(dict(**tags,**r) for r in module.table_row_metrics(items,kind))
        paired.extend(dict(**tags,**r) for r in module.table_row_pairs(items,kind))
    files={name:module.write_rows(name,r) for name,r in (("metrics.csv",metrics),("paired.csv",paired))}
    files["rows.csv.gz"]=module.write_rows("rows.csv.gz",rows,compress=True)
    audit=dict(cells=sum(r["audit_cells"] for r in rows if r["changed"]))
    audit.update({arm:sum(r[f"audit_changed_{arm}"] for r in rows if r["changed"])
                  for arm in ("certified",*VARIANTS)})
    roster.write(OUT/"evaluation_receipt.json",dict(
        descriptions=9310,changed=5904,files=files,audit=audit,intervals_pp=module.bootstraps(rows),
        prediction_seal=roster.sha(seal_path),code=roster.sha(__file__),new_Qwen_queries=0,
        production_prediction_changes=0,FINAL_access=False,seconds=perf_counter()-started,
    ))
    print(audit,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("propose","evaluate"))
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    (propose if args.phase=="propose" else evaluate)(args.workers)
