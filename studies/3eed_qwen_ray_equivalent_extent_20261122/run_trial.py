"""Seal and evaluate an extent correction with independent ray checks."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
WORK, OUT, PROTOCOL = HERE/"private", HERE/"results", HERE/"PROTOCOL.json"
PAYLOAD = WORK/"predictions.jsonl.gz"
PREVIOUS = HERE.parent/"3eed_qwen_anchor_branch_intervention_20261122"
sys.path.insert(0,str(PREVIOUS))
from propose import first_depth, geometry, get_support, origin, roster
from evaluate_trial import archived_iou, classify, DATA, score_full
from center_sets import local
from extent import TOL, intrinsic_dimensions, wall_motion, allowed_fraction, shrink_box, contained


ARMS = ("baseline","intrinsic","support_only","full_or_stay","ray_equivalent")
VARIANTS = ARMS[1:]
PAIRS = tuple((a,"baseline") for a in VARIANTS)+tuple(("ray_equivalent",a) for a in VARIANTS[:-1])


def invariance(old, new):
    removed = (old>0.) & (new==0.)
    common = (old>0.) & (new>0.)
    changed = removed | (common & (abs(new-old)>TOL))
    if ((old==0.) & (new>0.)).any():
        raise ValueError("A nested box introduced a new intersection")
    max_depth_delta = float(np.max(abs(new[common]-old[common]))) if common.any() else 0.
    return int(changed.sum()),max_depth_delta


def contract(box, raw_dims, rotation, support, selection, audit, depth, transform):
    inv = np.linalg.inv(transform)
    measured = geometry.rigid_xyz(selection,inv)
    protected = np.concatenate((support[contained(support,box)],measured[contained(measured,box)]))
    d, vertical = intrinsic_dimensions(raw_dims,rotation)
    advance, retreat = wall_motion(box,d,inv[:3,3])
    hits = depth>0.
    if not hits.any() or not ((advance+retreat)>TOL).any() or contained(inv[:3,3][None],box)[0]:
        return {}, vertical, len(protected)
    lifted = selection[hits]*(depth[hits]/selection[hits,2])[:,None]
    first = geometry.rigid_xyz(lifted,inv)
    protected_local = local(protected,box)
    mandatory = np.concatenate((protected_local,local(first,box)))
    half = box[3:6]/2.
    support_fraction = allowed_fraction(half,advance,retreat,protected_local)
    cert_fraction = allowed_fraction(half,advance,retreat,mandatory)
    assert (cert_fraction<=support_fraction+1.e-12).all()
    fractions = dict(intrinsic=np.ones(3), support_only=support_fraction, ray_equivalent=cert_fraction,
                     full_or_stay=np.ones(3) if (cert_fraction==1.).all() else np.zeros(3))
    audit_depth = first_depth(box,audit,transform)
    audit_points = geometry.rigid_xyz(audit,inv)
    protected_audit = audit_points[contained(audit_points,box)]
    options = {}
    for arm,fraction in fractions.items():
        candidate = shrink_box(box,advance,retreat,fraction)
        if np.max(abs(candidate-box)) <= TOL:
            continue
        np.testing.assert_allclose(candidate[2]-candidate[5]/2.,box[2]-box[5]/2.,rtol=0.,atol=TOL)
        assert candidate[6]==box[6] and ((candidate[3:6] <= box[3:6]+TOL)&(candidate[3:6]>0.)).all()
        new_depth = first_depth(candidate,selection,transform)
        changed_selection, max_delta = invariance(depth,new_depth)
        changed_audit,_ = invariance(audit_depth,first_depth(candidate,audit,transform))
        lost = int((~contained(protected,candidate)).sum())
        if arm != "intrinsic":
            assert lost==0
        if arm in ("ray_equivalent","full_or_stay"):
            assert changed_selection==0 and max_delta<=TOL
        options[arm] = dict(
            box=candidate.tolist(), selection_changes=changed_selection, audit_changes=changed_audit,
            lost_points=lost, lost_audit_points=int((~contained(protected_audit,candidate)).sum()),
            trim=(box[3:6]-candidate[3:6]).tolist(), volume_fraction=float(np.prod(candidate[3:6]/box[3:6])),
        )
    return options, vertical, len(protected)


def initialize(index):
    global replies_hashes
    replies_hashes = index


def predict_row(job):
    frame,rec,pins,check = job
    uid = frame["uid"]
    assert uid==rec["uid"] and frame["panel"]=="validation"
    if not rec["production"]:
        return dict(uid=uid,active=False)
    archive = origin.prediction(frame,check,pins)
    pair = origin.replies(frame,archive,replies_hashes)
    matrix,held = (np.asarray(rec[k],float) for k in ("transform","held"))
    key = "q" if rec["choose_qwen"] else "p"
    assert archive["chooses_qwen"]==rec["choose_qwen"]
    box = np.asarray(rec[key],float)
    np.testing.assert_allclose(box,rec["production_box"],rtol=0.,atol=5.e-10)
    raw = np.asarray(pair["3d"]["value"],float)
    upright,rot = geometry.to_sensor(raw,matrix)
    np.testing.assert_array_equal(upright[3:6],box[3:6])
    support,_,_,select,audit,_,_ = get_support(frame,pair["2d"]["value"],held,matrix)
    depth = np.asarray(rec[key+"_depth"],float)
    np.testing.assert_array_equal(first_depth(box,select,matrix),depth)
    variants,vertical,points = contract(box,raw[3:6],rot,support,select,audit,depth,matrix)
    return dict(uid=uid,active=True,box=box.tolist(),qwen=rec["choose_qwen"],variants=variants,
                selection_rays=len(held),audit_rays=len(audit),protected_points=points,
                vertical_axis=vertical,vertical_alignment=float(abs(rot[2,vertical])))


def propose(workers):
    started = perf_counter()
    cfg = roster.read(PROTOCOL)
    lock = roster.sha(origin.ACTIVE/"METHOD_LOCK.json")
    assert lock==cfg["frozen_method_sha256"] and not cfg["FINAL_access"]
    branch_seal = roster.read(PREVIOUS/"results/prediction_seal.json")
    assert branch_seal["code"]==roster.sha(PREVIOUS/"propose.py")
    trace,pinned = origin.verified_trace()
    frames,pins,checks,index,_ = origin.inputs()
    n = active = cells = changed = 0
    trial = PAYLOAD.with_suffix(".tmp")
    with gzip.open(pinned,"rt") as old, trial.open("wb") as raw, \
            gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0,compresslevel=4) as out, \
            ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(index,)) as pool:
        jobs = ((f,json.loads(line),pins,checks[f["uid"]]) for f,line in zip(frames,old,strict=True))
        for row in pool.map(predict_row,jobs,chunksize=12):
            out.write((json.dumps(row,separators=(",",":"),allow_nan=False)+"\n").encode())
            n += 1
            active += row["active"]
            cells += row.get("selection_rays",0)
            changed += bool(row.get("variants"))
            if n%1000==0:
                print("Sealed inputs",n,"/ 9310",flush=True)
    assert n==9310 and active==5910 and cells==688114
    trial.replace(PAYLOAD)
    roster.write(OUT/"prediction_seal.json",dict(
        descriptions=n,active=active,selection_rays=cells,changed_any_variant=changed,trace=trace["payload"],
        locked_method=lock,protocol=roster.sha(PROTOCOL),code=roster.sha(__file__),solver=roster.sha(HERE/"extent.py"),
        support_replay_code=branch_seal["code"],payload=roster.sha(PAYLOAD),target_gt_opened=False,
        measured_points_and_locked_rays_preserved=True,production_prediction_changes=0,
        new_Qwen_queries=0,FINAL_access=False,seconds=perf_counter()-started,
    ))
    print("Sealed",n,"descriptions with changed shadow boxes on",changed,flush=True)


def score_row(job):
    frame,sealed,archived = job
    assert sealed["uid"]==frame["uid"]==archived["uid"] and frame["panel"]=="validation"
    variants = sealed.get("variants",{})
    baseline = float(archived["iou_current"])
    values = {}
    if variants:
        path = Path(frame["calibration"])
        assert path.name=="meta_info.json" and path.parents[2]==DATA/frame["target"]
        items = roster.read(path)["ground_info"]
        truth = score_full.canonical_box(score_full.truth_for_slot(frame,items))
        assert items[frame["slot"]]["class"].strip().lower()==archived["category"]
        scores,_ = score_full.score_boxes(np.asarray([sealed["box"]]+[v["box"] for v in variants.values()]),truth)
        assert abs(scores[0]-baseline)<1.e-12 and (scores[0]>.5)==(baseline>.5)
        values = dict(zip(variants,scores[1:],strict=True))
    row = {k:archived[k] for k in ("target","uid","category","recording","frame")}
    case = classify(float(archived["iou_pca"]),float(archived["iou_qwen"]),sealed["qwen"]) if sealed["active"] else "fallback"
    row.update(active=sealed["active"],case=case,gt_scored=bool(variants),iou_baseline=baseline,
               **{k:sealed.get(k) for k in ("selection_rays","audit_rays","protected_points","vertical_axis","vertical_alignment")})
    for arm in VARIANTS:
        row[f"iou_{arm}"] = float(values.get(arm,baseline))
        row[f"changed_{arm}"] = arm in variants
        item = variants.get(arm,{})
        for attr in ("selection_changes","audit_changes","lost_points","lost_audit_points","volume_fraction"):
            row[f"{arm}_{attr}"] = item.get(attr)
        for i,axis in enumerate(("long","short","height")):
            row[f"{arm}_trim_{axis}"] = item["trim"][i] if item else None
    return row


def summary(group):
    result = []
    for arm in VARIANTS:
        changes = [r for r in group if r[f"changed_{arm}"]]
        qs = [r[f"{arm}_volume_fraction"] for r in changes]
        result.append(dict(
            arm=arm,n=len(group),changed=len(changes),
            selection_cells=sum(r["selection_rays"] for r in changes),
            selection_changed_cells=sum(r[f"{arm}_selection_changes"] for r in changes),
            selection_changed_pairs=sum(r[f"{arm}_selection_changes"]>0 for r in changes),
            audit_cells=sum(r["audit_rays"] for r in changes),
            audit_changed_cells=sum(r[f"{arm}_audit_changes"] for r in changes),
            audit_changed_pairs=sum(r[f"{arm}_audit_changes"]>0 for r in changes),
            lost_points=sum(r[f"{arm}_lost_points"] for r in changes),
            lost_audit_points=sum(r[f"{arm}_lost_audit_points"] for r in changes),
            median_volume_reduction_percent=float(100*(1-np.median(qs))) if qs else None,
            **{f"median_trim_{axis}_m":float(np.median([r[f"{arm}_trim_{axis}"] for r in changes])) if changes else None
               for axis in ("long","short","height")},
        ))
    return result


def evaluate(workers):
    started = perf_counter()
    path = OUT/"prediction_seal.json"
    seal = roster.read(path)
    assert seal["code"]==roster.sha(__file__) and seal["solver"]==roster.sha(HERE/"extent.py")
    assert seal["locked_method"]==roster.sha(origin.ACTIVE/"METHOD_LOCK.json")
    assert not seal["target_gt_opened"] and not seal["FINAL_access"]
    assert seal["protocol"]==roster.sha(PROTOCOL) and seal["payload"]==roster.sha(PAYLOAD)
    previous = archived_iou()
    frames,_,_,_,_ = origin.inputs()
    rows=[]
    with gzip.open(PAYLOAD,"rt") as stream, ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = ((f,json.loads(line),previous[f["uid"]]) for f,line in zip(frames,stream,strict=True))
        for row in pool.map(score_row,jobs,chunksize=12):
            rows.append(row)
    assert len(rows)==len({r["uid"] for r in rows})==9310
    assert sum(r["gt_scored"] for r in rows)==seal["changed_any_variant"] and sum(r["active"] for r in rows)==5910
    for row in rows:
        if row["changed_ray_equivalent"]:
            assert row["ray_equivalent_selection_changes"]==row["ray_equivalent_lost_points"]==0
    groups = defaultdict(list)
    for r in rows:
        keys = [("macro","all"),("pooled","all"),("platform",r["target"]),("category",r["category"]),
                ("recording",r["target"]+"/"+r["recording"]),("case",r["case"])]
        if r["changed_ray_equivalent"]:
            keys.extend((("coverage","certified_changed"),("certified_platform",r["target"])))
        for key in keys:
            groups[key].append(r)
    score_full.ARMS,score_full.PAIRS,score_full.OUT = ARMS,PAIRS,OUT
    metrics,paired,witness = [],[],[]
    for (kind,value),items in sorted(groups.items()):
        key = dict(group=kind,value=value)
        agg = "macro" if kind=="macro" else "pooled"
        metrics.extend(dict(**key,**r) for r in score_full.table_row_metrics(items,agg))
        paired.extend(dict(**key,**r) for r in score_full.table_row_pairs(items,agg))
        witness.extend(dict(**key,**r) for r in summary(items))
    files={name:score_full.write_rows(name,table) for name,table in (("metrics.csv",metrics),("paired.csv",paired),("ray_checks.csv",witness))}
    for plat in score_full.PLATFORMS:
        name=f"rows_{plat}.csv.gz"
        files[name]=score_full.write_rows(name,[r for r in rows if r["target"]==plat],compress=True)
    roster.write(OUT/"evaluation_receipt.json",dict(
        files=files, descriptions=len(rows), active=5910, ordinary_GT_descriptions=sum(r["gt_scored"] for r in rows),
        prediction_seal=roster.sha(path), code=roster.sha(__file__), intervals_pp=score_full.bootstraps(rows),
        new_Qwen_queries=0, production_prediction_changes=0, FINAL_access=False, seconds=perf_counter()-started,
    ))
    print(summary(rows),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("propose","evaluate"))
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    (propose if args.phase=="propose" else evaluate)(args.workers)
