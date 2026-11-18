"""Score the one-step conditional arm and physically identical-information direct geometry controls."""
from pathlib import Path
import os
import tempfile

import numpy as np
from score157 import (EPS, GROUPS, HERE, INPUT, LOCAL, META, N, REGISTRY_SHA, aggregates, digest,
                      gain_record, ground_truth, many_iou, oriented_iou, read, read_csv,
                      require, rotation, save, table)

ARMS = ("one_step", "box_step", "interpolate")
ALL = ("A","direct",*ARMS)
DIAGS = ("step_m","param_l2","gradient_norm")


def main():
    out = HERE / "tta_results"
    require(not out.exists(), "existing TTA results")
    asset = LOCAL / "TTA_PRIVATE.npz"
    seal = read(LOCAL / "TTA_SEAL.json")
    first = read(LOCAL / "PREDICTION_SEAL.json")
    lock = HERE / "experiment/DESIGN_LOCK.json"
    prior_score = read(HERE / "results/AUDIT.json")
    driver = seal["driver"]
    require(driver in ("ray","combined") and seal["n"] == N and seal["status"] == "PASS" and
            digest(asset) == seal["prediction_sha256"] and seal["direct_prediction_sha256"] ==
            first["prediction_sha256"] == prior_score["prediction_sha256"] and
            seal["design_sha256"] == digest(lock) and digest(INPUT) == REGISTRY_SHA, "invalid TTA evaluation bridge")
    require(digest(LOCAL / "PREDICTIONS_PRIVATE.npz") == first["prediction_sha256"], "direct prediction changed")
    direct_rows = read_csv(HERE / "results/all_rows.csv")
    direct_summary = read(HERE / "results/summary.json")
    macro = {r["arm"]:r["mIoU"] for r in direct_summary if r["group"]=="six_equal" and r["aggregation"]=="design_weighted"}
    expected = "ray" if macro["ray"] > macro["A"] else ("combined" if macro["combined"] > macro["A"] else "none")
    require(driver == expected, "unqualified TTA or different predeclared driver")
    reg = read(INPUT)
    with np.load(LOCAL / "PREDICTIONS_PRIVATE.npz",allow_pickle=False) as f:
        a = f["A"].copy()
        h = f["H"].copy()
        direct_switch = f["choose_"+driver].copy()
    with np.load(asset,allow_pickle=False) as f:
        require(list(f["arms"]) == list(ARMS) and np.array_equal(f["ordinal"],np.arange(1,N+1)), "TTA arm or order")
        boxes = f["boxes"].astype(float)
        switch = f["accepted"]
        changed = f["moved"]
        diag = {k:f[k] for k in DIAGS}
    require(np.array_equal(switch,direct_switch) and switch.dtype == changed.dtype == bool and
            len(direct_rows) == len(reg) == N and boxes.shape == (N,len(ARMS),7) and
            np.isfinite(boxes).all(), "TTA geometry or driver mismatch")
    require(np.allclose(boxes[~switch], a[~switch,None,:],atol=1e-6,rtol=0) and not np.any(changed & ~switch),
            "unaccepted TTA modified geometry")
    require(np.array_equal(boxes[:,:,2:6],np.broadcast_to(a[:,None,2:6],(N,len(ARMS),4))),
            "TTA moved vertical center or shape")
    target_yaw=np.where(switch,h[:,6],a[:,6])
    require(np.all(abs(np.sin(boxes[:,:,6]-target_yaw[:,None]))<1e-6),"TTA yaw differs from direct accepted yaw")
    require(all(v.shape==(N,) and np.isfinite(v).all() for v in diag.values()), "TTA diagnostics")
    boxes[...,3:6]=np.maximum(boxes[...,3:6],1e-6)
    rows=[]
    max_err=0.
    for i,e in enumerate(reg):
        require(e["cohort"]=="EXT" and not e["protected_alias_fallback"],"nonordinary row")
        old=direct_rows[i]
        fr=e["row"]
        require((e["source"],e["target"]) == (old["source"],old["target"]) and
                int(old["ordinal"]) == i+1, "TTA direct scope mismatch")
        gt=ground_truth(e["target"],fr["metapath"],fr["object_index"],fr["caption"])
        ref=rotation(gt[6])
        metrics=many_iou(boxes[i],gt)
        for b,v in zip(boxes[i],metrics):
            check=oriented_iou(b[:3],b[3:6],rotation(b[6]),gt[:3],gt[3:6],ref)
            max_err=max(max_err,abs(check-v))
            require(all((v>t)==(check>t) for t in (.25,.5)), "independent threshold error")
        line={k:old[k] for k in META}
        for k in ("ordinal","scene_cluster","frame_cluster","point_content_cluster"):
            line[k]=int(line[k])
        line.update(has_B=old["has_B"]=="True",sampling_weight=float(old["sampling_weight"]),
                    macro_weight=float(old["macro_weight"]),accepted=bool(switch[i]),moved=bool(changed[i]),
                    driver=driver,A=float(old["A"]),direct=float(old[driver]),**dict(zip(ARMS,map(float,metrics))))
        line.update({k:float(v[i]) for k,v in diag.items()})
        rows.append(line)
    require(max_err<1e-7,"TTA OBB audit")
    max_xy=np.max(abs(boxes[:,0,:2]-boxes[:,1,:2]),axis=1)
    max_iou=max(abs(r["one_step"]-r["box_step"]) for r in rows)
    w=np.array([r["macro_weight"] for r in rows])
    direction=np.array([r["source"]+"_to_"+r["target"] for r in rows])
    panels={"all":np.ones(N,bool),"accepted":switch,"moved":changed,"not_accepted":~switch}
    panels.update({d:direction==d for d in np.unique(direction)})
    comparisons=[]
    for group,mask in panels.items():
        if not mask.any():
            continue
        for ctrl in ("A","direct","box_step","interpolate"):
            for name in ARMS:
                if name == ctrl:
                    continue
                arrays=[np.array([r[k] for r in rows])[mask] for k in (ctrl,name)]
                comparisons.append(dict(scope=group,arm=name,control=ctrl,full_6way_weight=float(w[mask].sum()),
                                   **gain_record(*arrays,w[mask],None,w[mask])))
    summary, directions, scenes, by_b = aggregates(rows, ALL)
    tmp=Path(tempfile.mkdtemp(prefix=".tta_scored_",dir=HERE))
    table(tmp/"all_rows.csv",rows)
    table(tmp/"paired.csv",comparisons)
    table(tmp/"direction.csv",directions)
    table(tmp/"scene.csv",scenes)
    table(tmp/"subset.csv",by_b)
    save(tmp/"summary.json",summary)
    save(tmp/"AUDIT.json",dict(status="PASS",n=N,driver=driver,accepted=int(switch.sum()),moved=int(changed.sum()),
            score_after_TTA_seal=True,prediction_sha256=seal["prediction_sha256"],
            direct_prediction_sha256=first["prediction_sha256"],design_sha256=digest(lock),code_sha256=digest(Path(__file__)),
            strict_thresholds=[.25,.50], independent_endpoint_checks=N*len(ARMS),max_independent_OBB_error=max_err,
            maximum_direct_box_step_vs_parameter_xy_difference_m=float(max_xy.max()),
            maximum_direct_box_step_vs_parameter_IoU_difference=float(max_iou),
            target_final_access=False))
    os.rename(tmp,out)
    print("R157 TTA score passed",driver,int(switch.sum()),flush=True)


if __name__ == "__main__":
    main()
