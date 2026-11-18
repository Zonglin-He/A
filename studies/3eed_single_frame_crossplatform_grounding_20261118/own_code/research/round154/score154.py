"""R154 post-seal CPU evaluator. No predictions, optimization, or model imports."""
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "research/round154"
PRED = ROOT / "crossplatform/round154/predictions"
INPUT = ROOT / "research/round131/INPUT_REGISTRY_PRIVATE.json"
DATA = ROOT / "crossplatform/round96/data_view"
ARMS = ("frozen_A", "joint_zero", "direct_R145", "R145", "wrong_R145",
        "alternate_single", "geometry_zero", "geometry_TTA", "anchored_score_TTA",
        "union_representative_TTA", "fusion_zero", "fusion_TTA", "shuffled_fusion_TTA")
POOLS = ("original_A", "alternate", "geometry", "fusion_zero", "fusion_TTA")
PARENT = dict(zip(ARMS[:5], ("Frozen_A", "norm_joint_zero", "norm_direct", "norm_TTA", "norm_wrong_TTA")))
PRIMARY = "fusion_TTA"
EPS = 1e-12
sys.path.insert(0, str(ROOT / "research/round6/source_validation_scoring"))
sys.path.insert(0, str(ROOT / "tools"))
from batch_yaw_iou import many_iou
from obb_iou import oriented_iou


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


def table(path, records):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def identity(entry):
    return entry["source"], entry["target"], entry["row"]["id"]


def slug(entry):
    src, tgt, sid = identity(entry)
    return f"{src}_to_{tgt}/{sid}.npz"


def preflight():
    design = HERE / "experiment/DESIGN_LOCK.json"
    design_hash = digest(design)
    audit = read(HERE / "PREDICTION_AUDIT.json")
    sealpath = PRED / "SEALED.json"
    seal = read(sealpath)
    seal_hash = digest(sealpath)
    if audit["status"] != "PASS" or audit["seal_sha256"] != seal_hash:
        raise ValueError("prediction audit does not authorize this seal")
    if "design_lock_sha256" in seal and seal["design_lock_sha256"] != design_hash:
        raise ValueError("design-lock hash mismatch")
    lock = read(INPUT.parent / "experiment/INPUT_LOCK.json")
    if digest(INPUT) != lock["files"][INPUT.name]:
        raise ValueError("R131 ordinary input registry has changed")
    reg = read(INPUT)
    names = {slug(r) for r in reg}
    if (len(reg) != len(names) or len(reg) != 1798 or seal["n"] != 1798 or
            set(seal["files"]) != names or tuple(seal["arm_names"]) != ARMS or
            tuple(seal["pool_names"]) != POOLS):
        raise ValueError("sealed inputs, arms or proposal pools differ from protocol")
    # All files are verified before obtaining a single held-out target label.
    for e in reg:
        if e["protected_alias_fallback"] or e["cohort"] != "EXT":
            raise ValueError("unexpected input outside the ordinary cohort")
        path = PRED / slug(e)
        if path.resolve().parent.parent != PRED.resolve() or digest(path) != seal["files"][slug(e)]:
            raise ValueError("invalid prediction path or byte hash")
    return reg, seal, seal_hash, design_hash


@lru_cache(maxsize=899)
def ground_truth(platform, metapath, obj, caption):
    # Only the 851 selected ordinary frame metadata files can be opened. The
    # original R96 official loader converts the center/yaw before float32 storage.
    mp = Path(metapath)
    expected = DATA / platform
    if mp.name != "meta_info.json" or mp.parent.parent.parent != expected:
        raise ValueError("metadata is outside the selected platform data view")
    label = read(mp)["ground_info"][obj]
    if " ".join(label["caption"].replace(",", " ,").split()) != caption:
        raise ValueError("caption -> object mismatch")
    bbox = np.array(label["bbox_3d"], dtype=np.float64)
    if bbox.shape != ((7,) if platform == "waymo" else (9,)):
        raise ValueError("unexpected target box schema")
    gt = bbox[:7].copy()
    if platform == "drone":
        gt[2] += 1.8
    elif platform == "waymo":
        angle = np.deg2rad([0, -45, 45, -90, 90][int(mp.parent.name.split("_")[-1])])
        c, s = np.cos(angle), np.sin(angle)
        gt[:2] = np.array([[c, -s], [s, c]]) @ gt[:2]
        gt[6] += angle
    elif platform != "quad":
        raise ValueError("unknown platform")
    return gt.astype(np.float32).astype(np.float64)


def canonical(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 6 or not np.isfinite(boxes).all():
        raise ValueError("invalid proposal geometry")
    oriented = np.column_stack([boxes, np.zeros(len(boxes))])
    oriented[:, 3:6] = np.maximum(oriented[:, 3:6], 1e-6)
    return oriented


def metrics(y, w):
    return dict(mIoU=float(np.average(y, weights=w) * 100),
                Acc25=float(np.average(y > .25, weights=w) * 100),
                Acc50=float(np.average(y > .5, weights=w) * 100))


def aggregates(rows, names):
    method, scenes, subsets, macro = [], [], [], []
    for scope in ["all", "has_B", "no_B"]:
        selected = [r for r in rows if scope == "all" or bool(r["has_B"]) == (scope == "has_B")]
        for agg in ["raw", "design_weighted"]:
            grouped = defaultdict(list)
            for r in selected:
                grouped[r["source"], r["target"]].append(r)
            for (src, tgt), rr in sorted(grouped.items()):
                w = np.array([r["sampling_weight"] if agg == "design_weighted" else 1.0 for r in rr])
                for name in names:
                    m = dict(source=src, target=tgt, subset=scope, aggregation=agg, arm=name,
                             n=len(rr), weight_sum=float(w.sum()), **metrics(np.array([r[name] for r in rr]), w))
                    method.append(m)
                    if scope == "all":
                        for scene in sorted({r["scene_cluster"] for r in rr}):
                            idx = np.array([r["scene_cluster"] == scene for r in rr])
                            yy = np.array([r[name] for r in rr])[idx]
                            scenes.append(dict(source=src, target=tgt, scene_cluster=scene, aggregation=agg,
                                               arm=name, n=len(yy), **metrics(yy, w[idx])))
            for name in names:
                for tgt in ["six_equal", "waymo", "drone", "quad"]:
                    m = [x for x in method if x["aggregation"] == agg and x["subset"] == scope and
                         x["arm"] == name and (tgt == "six_equal" or x["target"] == tgt)]
                    if len(m) != (6 if tgt == "six_equal" else 2):
                        raise ValueError("missing direction when aggregating")
                    result = dict(aggregation=agg, group=tgt, subset=scope, arm=name, n=sum(x["n"] for x in m),
                                  **{k:float(np.mean([x[k] for x in m])) for k in ["mIoU", "Acc25", "Acc50"]})
                    (macro if scope == "all" else subsets).append(result)
    return macro, method, scenes, subsets


def paired(rows):
    contrasts = ([ (PRIMARY, arm) for arm in ARMS if arm != PRIMARY ] +
                 [("geometry_TTA", "R145"), ("geometry_zero", "joint_zero"),
                  ("geometry_TTA", "geometry_zero"), ("fusion_zero", "joint_zero"),
                  ("anchored_score_TTA", "R145"), ("alternate_single", "frozen_A")])
    output = []
    for scope in ["all", "has_B", "no_B"]:
        rr = [r for r in rows if scope == "all" or bool(r["has_B"]) == (scope == "has_B")]
        for agg in ["raw", "design_weighted"]:
            scales = defaultdict(float)
            for r in rr:
                scales[r["source"], r["target"]] += r["sampling_weight"] if agg == "design_weighted" else 1
            for src, tgt in list(sorted(scales)) + [("all", "all")]:
                sel = rr if src == "all" else [r for r in rr if (r["source"], r["target"]) == (src,tgt)]
                w = np.array([(r["sampling_weight"] if agg == "design_weighted" else 1)
                              / scales[r["source"],r["target"]] for r in sel])
                for arm, ctrl in contrasts:
                    new = np.array([r[arm] for r in sel])
                    old = np.array([r[ctrl] for r in sel])
                    dif = new - old
                    rec = dict(source=src, target=tgt, subset=scope, aggregation=agg, arm=arm, control=ctrl,
                               n=len(sel), gain_pp=float(np.average(dif, weights=w)*100),
                               better=int(sum(dif > EPS)), worse=int(sum(dif < -EPS)),
                               unchanged=int(sum(abs(dif) <= EPS)))
                    for threshold, tag in [(.25,"25"),(.5,"50")]:
                        rec["rescues"+tag] = int(sum((new > threshold) & (old <= threshold)))
                        rec["harms"+tag] = int(sum((new <= threshold) & (old > threshold)))
                    output.append(rec)
    return output


def main():
    reg, seal, seal_hash, design_hash = preflight()
    results = HERE / "results"
    if results.exists():
        raise FileExistsError(results)
    staging = Path(tempfile.mkdtemp(prefix=".scoring_", dir=HERE))
    old = {(r["source"],r["target"],r["id_PRIVATE"]):r for r in read(ROOT/"research/round145/ALL_ROWS_PRIVATE.json")}
    if set(old) != {identity(e) for e in reg}:
        raise ValueError("R145 and R131 cohort keys differ")
    raw, oracle = [], []
    clusters = defaultdict(dict)
    max_selected_error = max_pool_error = max_parent_error = 0.0
    independent_calls = 0
    for ordinal, entry in enumerate(reg, start=1):
        src, tgt, sid = identity(entry)
        pathname = PRED / slug(entry)
        if digest(pathname) != seal["files"][slug(entry)]:
            raise ValueError("prediction changed after preflight")
        with np.load(pathname, allow_pickle=False) as data:
            if tuple(data["arm_names"].tolist()) != ARMS or tuple(data["pool_names"].tolist()) != POOLS:
                raise ValueError("unexpected row schema")
            chosen = canonical(data["chosen_boxes"])
            pool = np.asarray(data["pools"],dtype=np.float64)
            has_B = bool(data["has_B"].item())
        if chosen.shape != (len(ARMS),7) or pool.shape != (len(POOLS),256,6):
            raise ValueError("invalid number of endpoints or proposals")
        prior = old[src,tgt,sid]
        if has_B != prior["has_B"] or entry["sampling_weight"] != prior["sampling_weight"]:
            raise ValueError("B availability or sampling weight changed")
        row = entry["row"]
        gt = ground_truth(tgt,row["metapath"],row["object_index"],row["caption"])
        all_boxes = np.concatenate([chosen,canonical(pool.reshape(-1,6))])
        all_y = many_iou(all_boxes,gt)
        task_y = all_y[:len(ARMS)]
        ys = all_y[len(ARMS):].reshape(len(POOLS),256)
        with np.load(ROOT/"research/round131/scores_attempt001"/f"{src}_to_{tgt}_{sid}_DIAGNOSTIC_IOUS_PRIVATE.npz",
                     allow_pickle=False) as cached:
            max_pool_error = max(max_pool_error,float(np.max(abs(ys[0]-cached["ious"]))))
        for i,name in enumerate(ARMS[:5]):
            max_parent_error = max(max_parent_error,abs(task_y[i]-prior["IoUs"][PARENT[name]]))
        if max(max_parent_error,max_pool_error) > 1e-7:
            raise ValueError(f"old-box or frozen endpoint mismatch on row {ordinal}")
        # Independent general 3D intersection checks all selected endpoints, plus
        # the chosen diagnostic best proposal of each input pool (not every proposal).
        selected_idxs = list(range(len(ARMS))) + list(len(ARMS) + np.arange(len(POOLS))*256 + ys.argmax(axis=1))
        c,s = np.cos(gt[6]),np.sin(gt[6])
        rot = np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        computed = {}
        for idx in selected_idxs:
            p = all_boxes[idx]
            geometry_key = tuple(p[:6])
            if geometry_key not in computed:
                computed[geometry_key] = oriented_iou(p[:3],p[3:6],np.eye(3),gt[:3],gt[3:6],rot)
                independent_calls += 1
            v = computed[geometry_key]
            max_selected_error = max(max_selected_error,abs(v-all_y[idx]))
            if abs(v - all_y[idx]) > 1e-7 or any((v>t) != (all_y[idx]>t) for t in [.25,.5]):
                raise ValueError(f"independent IoU mismatch on row {ordinal}")
        base = dict(ordinal=ordinal,source=src,target=tgt,has_B=has_B,sampling_weight=entry["sampling_weight"])
        for alias, value in [("scene_cluster",row["scene"]),("frame_cluster",entry["frame_PRIVATE"]),
                             ("point_content_cluster",entry["content_group_PRIVATE"])]:
            name = (tgt,value)
            base[alias] = clusters[alias].setdefault(name,len(clusters[alias])+1)
        raw.append(base | dict(zip(ARMS,task_y.tolist())))
        oracle.append(base | dict(zip(POOLS,ys.max(axis=1).tolist())))
    for family,names,rr in [("task",ARMS,raw),("oracle_only",POOLS,oracle)]:
        macro,method,scenes,subsets = aggregates(rr,names)
        table(staging/f"{family}_all_rows.csv",rr)
        table(staging/f"{family}_direction.csv",method)
        table(staging/f"{family}_scene.csv",scenes)
        table(staging/f"{family}_subset.csv",subsets)
        save(staging/f"{family}_summary.json",macro)
    table(staging/"paired.csv",paired(raw))
    save(staging/"AUDIT.json",dict(status="PASS",n=len(raw),seal_sha256=seal_hash,
        design_lock_sha256=design_hash,score_code_sha256=digest(Path(__file__)),all_predictions_verified_before_GT=True,
        ordinary_unique_expressions=ground_truth.cache_info().misses, independent_general_OBB_checks=independent_calls,
        max_selected_IoU_error=max_selected_error,max_original_pool_IoU_error=max_pool_error,
        max_parent_endpoint_IoU_error=max_parent_error,strict_thresholds=[.25,.5],
        cluster_counts={k:len(v) for k,v in clusters.items()}, protected_frames_opened=0))
    os.rename(staging,results)
    print("PASS_R154_POSTSEAL_SCORING",len(raw),flush=True)


if __name__ == "__main__":
    main()
