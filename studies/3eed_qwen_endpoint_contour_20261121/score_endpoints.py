"""Unseal the ordinary labels and independently score the actual chosen boxes."""
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from endpoints import (ETA, HERE, MODES, OUT, PARENT, PRIVATE, SPEC, TOL, ancestry, digest, read, write)
from evaluate import EPS, PUBLIC_LAYOUT, effect, mean_platform, uncertainty
from score import canonical_box, rotation, score_boxes, truth_for_slot

ARMS=("H","span","endpoint","mismatch","span_all","previous_B")
SINGLE=tuple(k+"_single" for k in MODES)
PAIRS=(("endpoint","H"),("endpoint","span"),("endpoint","mismatch"),("endpoint","span_all"),
       ("endpoint","previous_B"),("span","H"),("mismatch","H"),("span_all","H"),("previous_B","H"),
       ("span","span_all"),("endpoint_single","H"),("endpoint_single","span_single"),
       ("endpoint_single","mismatch_single"))


def write_table(name,rows):
    path=OUT/(name+".csv")
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    return digest(path)


def groups(rows):
    scopes={"all":rows,"validation":[r for r in rows if r["panel"]=="validation"],
            "extension":[r for r in rows if r["panel"]!="validation"],
            "recording_separated":[r for r in rows if r["separation"]=="new_recording"],
            "paired_eligible":[r for r in rows if r["eligible"]],
            "visible_mismatch":[r for r in rows if r["mark_shift_px"] is not None and
                               r["eligible"] and r["mark_shift_px"]>=10.]}
    for name,scope in scopes.items():
        if scope:
            yield name,"macro",scope
        for target in sorted({r["target"] for r in scope}):
            yield name,target,[r for r in scope if r["target"]==target]


def unique_best(scores,tol):
    if not scores:
        return ""
    first,second=sorted(scores,key=scores.get,reverse=True)[:2]
    return first if scores[first]-scores[second]>tol else ""


def run():
    p=read(OUT/"prediction_seal.json")
    assert p["pins"]["inputs"]==digest(OUT/"input_seal.json")
    q=read(OUT/"input_seal.json")
    assert q["protocol"]==digest(SPEC) and q["observations"]==digest(PRIVATE/"observations.json")
    assert not p["target_labels_read"] and not p["FINAL_access"] and p["frames"]==416
    assert digest(PRIVATE/"predictions.json")==p["predictions"] and p["pins"]["code"]==digest(HERE/"run_qwen.py")
    assert all(digest(HERE/k)==v for k,v in q["code"].items())
    obs,pred=read(PRIVATE/"observations.json"),read(PRIVATE/"predictions.json")
    panel,_,prepared,possible=ancestry()
    parent_result=read(PARENT/"results/evaluation_receipt.json")
    assert parent_result["prediction_seal"]==read(SPEC)["parent_predictions"]
    assert digest(PARENT/"results/rows.csv")==parent_result["files"]["rows"]
    with (PARENT/"results/rows.csv").open() as f:
        official={r["uid"]:r for r in csv.DictReader(f)}
    if PUBLIC_LAYOUT:
        import score_ablation
        mounts={Path(f["calibration"]).parents[3] for f in panel}
        assert len(mounts)==1
        score_ablation.DATA=mounts.pop()
    rows=[]; max_gap=0.; max_parity=0.
    for f in panel:
        uid=f["uid"]; seen=obs[uid]; result=pred[uid]; old=official[uid]; parent=possible[uid]
        h=parent["H"]
        label={"H":"H","previous_B":parent["sides"].get("B","H")}
        if label["previous_B"] not in ETA:
            label["previous_B"]="H"
        boxes={"H":h,"previous_B":parent["boxes"]["B"]}
        r={k:f[k] for k in ("uid","target","recording","panel","separation")}
        r.update(upstream=seen["upstream"],selected=seen["selected"],movable=seen["movable"],
                 eligible=seen["status"]=="eligible",projection_status=seen["status"],
                 spacing_px=seen.get("spacing_px"),mark_shift_px=seen["mark_shift_px"],
                 low_visibility=seen.get("endpoints",[{}])[0].get("status","unavailable"),
                 high_visibility=seen.get("endpoints",[{},{}])[1].get("status","unavailable"))
        for arm, decision in result.items():
            label[arm]=decision["label"]
            boxes[arm]=decision["box"]
            r["decision_"+arm]=decision["status"]
            r["forward_"+arm]=decision["forward"]
            r["reverse_"+arm]=decision["reverse"]
            if arm in MODES:
                code=decision["forward"] if decision["forward"] in ETA else "H"
                label[arm+"_single"]=code
                boxes[arm+"_single"]=h if code=="H" else parent["discrete_boxes"][code]
        offsets=prepared[uid].get("geometry",{}).get("offsets",{})
        for arm,code in label.items():
            r["label_"+arm]=code
            r["shift_"+arm]=offsets.get(code,0.)
            expected=h if code=="H" else parent["discrete_boxes"][code]
            assert expected==boxes[arm]
        if h is None:
            assert all(b is None for b in boxes.values())
            scores=np.zeros(len(boxes))
        else:
            gt=canonical_box(truth_for_slot(f,read(f["calibration"])["ground_info"]))
            scores,gap=score_boxes(np.array(list(boxes.values())),gt)
            max_gap=max(max_gap,gap)
            if r["eligible"]:
                path=Path(f["points"])
                cloud=(np.load(path,mmap_mode="r",allow_pickle=False) if path.suffix==".npy"
                       else np.fromfile(path,"<f4").reshape(-1,4))
                positions=cloud[[v["raw_index"] for v in seen["endpoints"]],:3].astype(float)
                inside=(abs((positions-gt[:3])@rotation(gt[6]))<=gt[3:6]/2).all(axis=1)
                r["endpoints_inside_upright_gt"]=int(inside.sum())
        for (arm,_),score in zip(boxes.items(),scores,strict=True):
            r["iou_"+arm]=float(score)
            max_parity=max(max_parity,abs(score-float(old["iou_"+label[arm]])))
        max_parity=max(max_parity,abs(r["iou_previous_B"]-float(old["iou_B"])))
        r["iou_oracle"]=float(old["iou_oracle"])
        r["opportunity50"]=r["iou_H"]<=.5<r["iou_oracle"]
        assert r["opportunity50"]==(old["oracle_recoverable50"]=="True")
        r["best_iou_side"]=unique_best({k:float(old["iou_"+k]) for k in ETA},EPS) if seen["movable"] else ""
        r["nearest_center_side"]=unique_best({k:-float(old["center_err_m_"+k]) for k in ETA},TOL) if seen["movable"] else ""
        best_at50=[k for k in ETA if float(old["iou_"+k])>.5]
        r["successful_sides_at50"]="|".join(best_at50) if seen["movable"] else ""
        if r["opportunity50"]:
            d=result["endpoint"]
            if r["iou_endpoint"]>.5:
                reason="rescued"
            elif not r["eligible"]:
                reason=seen["status"]
            elif d["status"]!="agree":
                reason=d["status"]
            elif abs(r["shift_endpoint"])<=TOL:
                reason="agreed_original_center"
            else:
                reason="agreed_shift_below_threshold"
            r["opportunity_outcome"]=reason
        else:
            r["opportunity_outcome"]=""
        rows.append(r)
    assert len(rows)==len(official)==416 and sum(r["opportunity50"] for r in rows)==23
    assert max(max_gap,max_parity)<EPS
    metrics=[]; paired=[]; stages=[]; recordings=[]; intervals={}
    for scope,target,group in groups(rows):
        tag=dict(scope=scope,target=target,n=len(group),recordings=len({r["recording"] for r in group}))
        for arm in (*ARMS,*SINGLE,"oracle"):
            vals=np.array([r["iou_"+arm] for r in group])
            metrics.append(dict(**tag,arm=arm,hits50=int(sum(vals>.5)),acc50=100*mean_platform(group,vals>.5),
                       acc25=100*mean_platform(group,vals>.25),miou=100*mean_platform(group,vals),
                       opportunities=sum(r["opportunity50"] for r in group),
                       opportunities_rescued=sum(r["opportunity50"] and r["iou_"+arm]>.5 for r in group)))
        for a,b in PAIRS:
            paired.append(dict(**tag,arm=a,reference=b,**effect(group,a,b)))
            if a=="endpoint" and b in ("H","span","mismatch") and target=="macro":
                intervals[f"{scope}/{a}-{b}"]=uncertainty(group,a,b)
        for subset,sub in (("all",group),("opportunity50",[r for r in group if r["opportunity50"]])):
            for arm in (*MODES,"span_all"):
                c=Counter(r["decision_"+arm] for r in sub)
                unique=[r for r in sub if r["eligible"] and r["best_iou_side"]]
                strength=[r for r in unique if r["decision_"+arm]=="agree"]
                nearest=[r for r in sub if r["eligible"] and r["nearest_center_side"] and r["decision_"+arm]=="agree"]
                stages.append(dict(**tag,subset=subset,arm=arm,denominator=len(sub),eligible=sum(r["eligible"] for r in sub),
                         **{k:c[k] for k in ("agree","conflict","unknown","invalid","upstream_fallback",
                                              "no_axial_movement","missing_visible_endpoint","markers_overlap")},
                         moved=sum(abs(r["shift_"+arm])>TOL for r in sub),
                         unique_iou_cases=len(unique),answered_on_unique_iou=len(strength),
                         chose_best_iou=sum(r["label_"+arm]==r["best_iou_side"] for r in strength),
                         answered_on_unique_center=len(nearest),
                         chose_nearest_center=sum(r["label_"+arm]==r["nearest_center_side"] for r in nearest)))
    for t,rec in sorted({(r["target"],r["recording"]) for r in rows}):
        group=[r for r in rows if (r["target"],r["recording"])==(t,rec)]
        for ref in ("H","span","mismatch","previous_B"):
            recordings.append(dict(target=t,recording=rec,n=len(group),arm="endpoint",reference=ref,
                                   **effect(group,"endpoint",ref)))
    files={name:write_table(name,items) for name,items in (("rows",rows),("opportunities",[r for r in rows if r["opportunity50"]]),
                        ("metrics",metrics),("paired",paired),("funnel",stages),("recordings",recordings))}
    write(OUT/"evaluation_receipt.json",dict(seal=digest(OUT/"prediction_seal.json"),files=files,
          previous_rows=parent_result["files"]["rows"],frames=len(rows),opportunities=23,
          dual_iou_max_error=max_gap,prior_action_iou_max_error=max_parity,
          confidence_intervals_pp=intervals,code=digest(__file__),FINAL_access=False))
    print("Scored all 416 frames and recovered the exact 23 historical opportunity IDs.",flush=True)


if __name__=="__main__":
    run()
