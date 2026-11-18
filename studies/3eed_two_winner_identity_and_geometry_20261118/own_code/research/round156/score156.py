"""Post-seal six-direction paired scoring and diagnostic upper bounds for R156."""
import csv
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT / "research/round155"))
from score155 import INPUT, aggregates, digest, ground_truth, many_iou, oriented_iou, read, require, rotation, save, table
from pair156 import OUT, DESIGN

META = ("ordinal","source","target","has_B","sampling_weight","scene_cluster","frame_cluster","point_content_cluster")
LIVE = ("A_axis","A_pca","A_full","D_axis","D_pca","D_full","A_single_pca")
ORACLES = {"oracle_R145":("A_pca","R145_pca"), "oracle_mixed":("A_pca","D_axis"),
           "oracle_shared":("A_pca","D_pca"), "oracle_full":("A_full","D_full"),
           "oracle_single_frame":("A_single_pca","D_pca"), "oracle_axis":("A_axis","D_axis")}
NAMES = ("frozen_A","R145_pca",*LIVE,*ORACLES.keys())
CONTRASTS = [("R145_pca","A_pca"),("D_axis","A_pca"),("D_pca","A_pca"),("D_full","A_full"),
             ("A_full","A_pca"),("D_full","D_pca"),("A_single_pca","A_pca"),("D_pca","D_axis"),
             ("D_pca","A_single_pca")] + [(name,pair[0]) for name,pair in ORACLES.items()]
EPS = 1e-12


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def verified_inputs():
    seal = read(OUT/"PAIR_SEAL.json")
    pp = OUT/"PAIR_PRIVATE.npz"
    require(seal["status"] == "PASS" and seal["prediction_sha256"] == digest(pp) and
            seal["design_sha256"] == digest(DESIGN), "seal mismatch")
    with np.load(pp,allow_pickle=False) as pred:
        boxes = np.array(pred["boxes"],np.float64)
        sym = np.array(pred["symmetric_A_pca"],np.float64)
        require(boxes.shape == (1798,2,3,7) and sym.shape == (1798,7) and
                np.array_equal(pred["ordinal"],np.arange(1,1799)) and
                list(pred["geometry"]) == ["axis","pca_shared_A","pca_full_raw"], "endpoint schema")
    boxes = np.concatenate((boxes.reshape(-1,6,7),sym[:,None,:]),axis=1)
    require(np.isfinite(boxes).all(),"nonfinite endpoint")
    boxes[...,3:6] = np.maximum(boxes[...,3:6],1e-6)
    old = ROOT/"research/round154/results/task_all_rows.csv"
    newer = ROOT/"research/round155/results/task_all_rows.csv"
    require(digest(old) == "3b76c59b108bfa35022e79e25ed9caefd8dec3dd5736b1e3d420fe81924d4bb2" and
            digest(newer) == "e8b15df735d2dc37a55a43617d18c78a035ff6a07683e20ef1296e2011160b9f" and
            digest(INPUT) == "38035be8022d15f4c6867f48a44b60e1eee6288f29b43be17b743559bca52e45","parent artifacts changed")
    old_rows, recent, reg = read_csv(old),read_csv(newer),read(INPUT)
    require(len(old_rows)==len(recent)==len(reg)==1798 and
            all(e["cohort"] == "EXT" and not e["protected_alias_fallback"] for e in reg),"cohort")
    for i,(e,p,q) in enumerate(zip(reg,old_rows,recent),start=1):
        require(all(p[k]==q[k] for k in META) and int(p["ordinal"])==i and
                (e["source"],e["target"])==(p["source"],p["target"]) and
                float(p["sampling_weight"])==e["sampling_weight"],"parent order or stratum")
    return boxes, reg, old_rows, recent, seal


def paired(rows):
    result = []
    for scope in ("all","has_B","no_B"):
        rr = [r for r in rows if scope == "all" or r["has_B"] == (scope=="has_B")]
        names = [(r["source"],r["target"]) for r in rr]
        for agg in ("raw","design_weighted"):
            ww = [r["sampling_weight"] if agg=="design_weighted" else 1. for r in rr]
            total = Counter()
            for k,w in zip(names,ww):
                total[k] += w
            w = np.array([a/total[k] for k,a in zip(names,ww)])
            for direction in [("all","all"),*sorted(total)]:
                ix = np.array([direction==("all","all") or k==direction for k in names])
                for new,ref in CONTRASTS:
                    a,b = (np.array([r[q] for r in rr])[ix] for q in (new,ref))
                    d = a-b
                    line = dict(subset=scope,aggregation=agg,source=direction[0],target=direction[1],
                                arm=new,control=ref,n=len(a),gain_pp=float(100*np.average(d,weights=w[ix])),
                                better=int(sum(d>EPS)),worse=int(sum(d< -EPS)),unchanged=int(sum(abs(d)<=EPS)))
                    for t,label in [(.25,"25"),(.5,"50")]:
                        line["rescues"+label]=int(sum((a>t)&(b<=t)))
                        line["harms"+label]=int(sum((a<=t)&(b>t)))
                    result.append(line)
    return result


def contributions(rows):
    weights = np.array([r["sampling_weight"] for r in rows])
    bydir = defaultdict(list)
    for i,r in enumerate(rows):
        bydir[r["source"],r["target"]].append(i)
    for ids in bydir.values():
        weights[ids] /= len(bydir)*weights[ids].sum()
    for w,r in zip(weights,rows):
        r["macro_weight"] = float(w)
        r["D_win_gain_pp"] = float(100*w*max(0.,r["D_pca"]-r["A_pca"]))


def main():
    out = HERE/"results"
    require(not out.exists(),"result directory already exists")
    boxes,reg,prev,prior,seal = verified_inputs()
    rows = []
    selected_error = reference_error = 0.
    checks = 0
    for i,(e,old,last) in enumerate(zip(reg,prev,prior)):
        frame = e["row"]
        gt = ground_truth(e["target"],frame["metapath"],frame["object_index"],frame["caption"])
        yc = many_iou(boxes[i],gt)
        g_rot = rotation(gt[6])
        independently = {}
        for box,v in zip(boxes[i],yc):
            key = tuple(box)
            if key not in independently:
                independently[key] = oriented_iou(box[:3],box[3:6],rotation(box[6]),gt[:3],gt[3:6],g_rot)
                checks += 1
            err = abs(independently[key]-v)
            selected_error = max(selected_error,err)
            require(err <= 1e-7 and all((v>t) == (independently[key]>t) for t in (.25,.5)), "independent OBB disagreement")
        for k,historical in ((0,old["joint_zero"]),(1,last["joint_zero_pca"]),(3,old["alternate_single"])):
            reference_error = max(reference_error,abs(float(historical)-yc[k]))
        require(reference_error<=1e-7, "prior endpoint not reproduced")
        row = {k:old[k] for k in META}
        row["has_B"] = old["has_B"]=="True"
        row["sampling_weight"] = float(old["sampling_weight"])
        row.update({k:int(row[k]) for k in ("ordinal","scene_cluster","frame_cluster","point_content_cluster")})
        row.update(frozen_A=float(old["frozen_A"]),R145_pca=float(last["R145_pca"]))
        row.update(zip(LIVE,yc.tolist()))
        for name,(a,b) in ORACLES.items():
            row[name]=max(row[a],row[b])
        rows.append(row)
    contributions(rows)
    summary,dirs,scenes,subsets=aggregates(rows,NAMES)
    raw_paired=paired(rows)
    staged = Path(tempfile.mkdtemp(prefix=".pairs_",dir=HERE))
    table(staged/"pair_rows.csv",rows)
    for name,items in (("pair_direction",dirs),("pair_scene",scenes),("pair_subset",subsets),("pair_contrasts",raw_paired)):
        table(staged/(name+".csv"),items)
    save(staged/"pair_summary.json",summary)
    save(staged/"PAIR_SCORE_AUDIT.json",dict(status="PASS",n=len(rows),ordinary_unique_expressions=ground_truth.cache_info().misses,
        design_sha256=digest(DESIGN),prediction_sha256=seal["prediction_sha256"],score_code_sha256=digest(Path(__file__)),
        all_endpoint_checks=len(rows)*len(LIVE),independent_OBB_calls=checks,max_oriented_error=selected_error,
        max_reference_IoU_error=reference_error,protected_frames_opened=0))
    os.rename(staged,out)
    print("R156 pair score complete",len(rows),flush=True)


if __name__=="__main__":
    main()
