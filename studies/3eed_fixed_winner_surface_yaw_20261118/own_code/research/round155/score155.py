"""Post-seal evaluation of target-only local boxes on the existing ordinary cohort."""
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
sys.path.insert(0, str(ROOT / "research/round154"))
from score154 import INPUT, aggregates, digest, ground_truth, many_iou, oriented_iou, read, save, table

DATA = ROOT / "crossplatform/round155"
PREVIOUS = ROOT / "research/round154/results/task_all_rows.csv"
PARENT_SHA = "3b76c59b108bfa35022e79e25ed9caefd8dec3dd5736b1e3d420fe81924d4bb2"
REGISTRY_SHA = "38035be8022d15f4c6867f48a44b60e1eee6288f29b43be17b743559bca52e45"
STATES = ("joint_zero", "R145")
CONFIGS = ("axis", "pca", "surface", "sign_flip")
OUTPUTS = tuple(f"{state}_{arm}" for state in STATES for arm in CONFIGS)
REFS = ("frozen_A", "geometry_zero", "geometry_TTA")
NAMES = REFS + OUTPUTS
EPS = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preflight():
    pred_path = DATA / "predictions_PRIVATE.npz"
    design_path = HERE / "experiment/DESIGN_LOCK.json"
    design_sha, prediction_sha = digest(design_path), digest(pred_path)
    seal, audit = read(DATA / "SEAL.json"), read(HERE / "PREDICTION_AUDIT.json")
    require(seal["sha256"] == audit["prediction_sha256"] == prediction_sha and
            seal["design_sha256"] == design_sha and audit["status"] == "PASS", "prediction audit or design mismatch")
    require(digest(INPUT) == REGISTRY_SHA and digest(PREVIOUS) == PARENT_SHA, "registry or R154 scalar mismatch")
    registry = read(INPUT)
    require(len(registry) == 1798 and all(not e["protected_alias_fallback"] and e["cohort"] == "EXT"
            for e in registry), "registered cohort changed")
    with PREVIOUS.open(newline="") as stream:
        parent = list(csv.DictReader(stream))
    require(len(parent) == len(registry), "parent scalar count")
    aliases = defaultdict(dict)
    for k, (e, old) in enumerate(zip(registry, parent), start=1):
        target = e["target"]
        require((old["source"], old["target"], int(old["ordinal"]), float(old["sampling_weight"])) ==
                (e["source"], target, k, float(e["sampling_weight"])), "record order or survey weights differ")
        require(old["has_B"] in ("True", "False"), "invalid B eligibility")
        for alias, val in (("scene_cluster", e["row"]["scene"]), ("frame_cluster", e["frame_PRIVATE"]),
                           ("point_content_cluster", e["content_group_PRIVATE"])):
            codes = aliases[alias]
            require(int(old[alias]) == codes.setdefault((target, val), len(codes) + 1), "parent cluster differs")
    with np.load(pred_path, allow_pickle=False) as pred:
        boxes = np.array(pred["boxes"], dtype=np.float64)
        eligible = pred["eligible"]
    require(boxes.shape == (1798, 2, 4, 7) and np.isfinite(boxes).all() and eligible.shape == (1798, 2)
            and np.isin(eligible, (False, True)).all(), "prediction shapes, values or eligibility")
    require((boxes[:, :, 0, 6] == 0).all(), "historical axis control has nonzero yaw")
    # Match the R154 canonical convention for degenerate predicted extents.
    boxes[..., 3:6] = np.maximum(boxes[..., 3:6], 1e-6)
    return registry, parent, boxes, eligible, prediction_sha, design_sha


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def contrasts(rows):
    primary = "R145_surface"
    pairs = [(primary, c) for c in NAMES if c != primary]
    pairs += [("joint_zero_surface", c) for c in ("joint_zero_axis", "joint_zero_pca", "joint_zero_sign_flip")]
    pairs += [(f"R145_{c}", f"joint_zero_{c}") for c in ("axis", "pca", "sign_flip")]
    pairs += [("R145_pca", "R145_axis"), ("R145_sign_flip", "R145_axis")]
    result = []
    for subset in ("all", "has_B", "no_B"):
        kept = [r for r in rows if subset == "all" or r["has_B"] == (subset == "has_B")]
        keys = [(r["source"], r["target"]) for r in kept]
        for aggregation in ("raw", "design_weighted"):
            weights = np.array([(r["sampling_weight"] if aggregation == "design_weighted" else 1.) for r in kept])
            totals = Counter()
            for key, weight in zip(keys, weights):
                totals[key] += weight
            scaled = np.array([weight / totals[key] for key, weight in zip(keys, weights)])
            for key in [("all", "all")] + sorted(totals):
                select = np.arange(len(kept)) if key == ("all","all") else np.array([i for i, d in enumerate(keys) if d == key])
                for arm, ctrl in pairs:
                    a, b = (np.array([kept[i][name] for i in select]) for name in (arm, ctrl))
                    diff = a - b
                    line = dict(subset=subset, aggregation=aggregation, source=key[0], target=key[1],
                                arm=arm, control=ctrl, n=len(a), gain_pp=float(np.average(diff, weights=scaled[select])*100),
                                better=int((diff>EPS).sum()), worse=int((diff < -EPS).sum()), unchanged=int((abs(diff)<=EPS).sum()))
                    for t, label in ((.25,"25"),(.5,"50")):
                        line["rescues"+label] = int(((a>t)&(b<=t)).sum())
                        line["harms"+label] = int(((a<=t)&(b>t)).sum())
                    result.append(line)
    return result


def main():
    out = HERE / "results"
    require(not out.exists(), "existing results require explicit reconciliation")
    registry, prior, boxes, eligible, prediction_sha, design_sha = preflight()
    stage = Path(tempfile.mkdtemp(prefix=".scoring_",dir=HERE))
    rows = []
    maximum_pair_error = maximum_old_error = 0.
    independent = 0
    for i,(e, old) in enumerate(zip(registry, prior)):
        r = e["row"]
        gt = ground_truth(e["target"], r["metapath"], r["object_index"], r["caption"])
        candidates = boxes[i].reshape(-1,7)
        ious = many_iou(candidates, gt)
        require(np.isfinite(ious).all() and ((ious>=0)&(ious<=1)).all(), "invalid computed IoUs")
        rot_gt = rotation(gt[6])
        cache = {}
        for box, iou in zip(candidates, ious):
            key = tuple(box)
            if key not in cache:
                cache[key] = oriented_iou(box[:3],box[3:6],rotation(box[6]),gt[:3],gt[3:6],rot_gt)
                independent += 1
            err = abs(cache[key] - iou)
            maximum_pair_error = max(maximum_pair_error,err)
            require(err <= 1e-7 and all((cache[key]>cut)==(iou>cut) for cut in (.25,.5)), "oriented IoU double-check failed")
        for idx, s in ((0,"joint_zero"),(4,"R145")):
            error = abs(ious[idx] - float(old[s]))
            maximum_old_error = max(maximum_old_error,error)
            require(error <= 1e-7, "historical control mismatch")
        base = {"ordinal":i+1, "source":e["source"], "target":e["target"], "has_B":old["has_B"]=="True",
                "sampling_weight":e["sampling_weight"]}
        base.update({k:int(old[k]) for k in ("scene_cluster","frame_cluster","point_content_cluster")})
        base.update({f"eligible_{s}":bool(eligible[i,j]) for j,s in enumerate(STATES)})
        rows.append(base | {k:float(old[k]) for k in REFS} | dict(zip(OUTPUTS,ious.tolist())))
    macro,directions,scenes,subsets = aggregates(rows, NAMES)
    table(stage/"task_all_rows.csv", rows)
    for name, entries in (("task_direction", directions),("task_scene", scenes),("task_subset", subsets),("paired",contrasts(rows))):
        table(stage/f"{name}.csv",entries)
    save(stage/"task_summary.json",macro)
    save(stage/"AUDIT.json",dict(status="PASS", n=len(rows),prediction_sha256=prediction_sha,design_sha256=design_sha,
        scorer_sha256=digest(Path(__file__)),parent_scalar_sha256=PARENT_SHA,ordinary_registry_sha256=REGISTRY_SHA,
        checked_oriented_endpoints=len(rows)*len(OUTPUTS),independent_general_OBB_calls=independent,
        max_oriented_IoU_error=maximum_pair_error,max_reference_IoU_error=maximum_old_error,
        ordinary_unique_expressions=ground_truth.cache_info().misses,protected_frames_opened=0,
        strict_thresholds=[.25,.5],eligible_by_state=dict(zip(STATES, map(int,eligible.sum(axis=0))))))
    os.rename(stage,out)
    print("PASS_R155_POSTSEAL_SCORING",len(rows),flush=True)


if __name__=="__main__":
    main()
