"""Pair sealed center choices with their previously verified scores on the ordinary roster."""
import csv
from collections import Counter

import numpy as np

from endpoints import HERE, OUT, PARENT, PRIVATE, TOL, ancestry, digest, read, write
from outline_geometry import ARMS
from score_endpoints import effect, mean_platform, uncertainty, write_table

BASES=("H","span","endpoint","mismatch","previous_B")
STUDY_ARMS=(*ARMS,*(a+"_censored" for a in ARMS))
VARIANTS=(*STUDY_ARMS,*(a+"_single" for a in STUDY_ARMS))


def index(path):
    with path.open() as stream:
        return {r["uid"]:r for r in csv.DictReader(stream)}


def subsets(rows):
    def selected(r,s):
        return (s=="all" or s==r["panel"] or s=="recording_separated" and r["separation"]=="new_recording"
                or s=="extension" and r["panel"]=="recording_extension" or s=="outline_eligible" and r["outline_eligible"])
    for scope in ("all","recording_separated","validation","extension","outline_eligible"):
        data=[r for r in rows if selected(r,scope)]
        if not data:
            continue
        yield scope,"macro",data
        for platform in sorted({r["target"] for r in data}):
            yield scope,platform,[r for r in data if r["target"]==platform]


def run():
    sealed=read(OUT/"outline_prediction_seal.json")
    assert not sealed["target_labels_read"] and not sealed["FINAL_access"]
    assert sealed["source"]==digest(OUT/"outline_raw_seal.json") and sealed["code"]==digest(HERE/"outline_geometry.py")
    assert sealed["predictions"]==digest(PRIVATE/"outline_predictions.json")
    assert read(OUT/"evaluation_receipt.json")["files"]["rows"]==digest(OUT/"rows.csv")
    assert read(PARENT/"results/evaluation_receipt.json")["files"]["rows"]==digest(PARENT/"results/rows.csv")
    primary=index(OUT/"rows.csv"); parent=index(PARENT/"results/rows.csv")
    predictions=read(PRIVATE/"outline_predictions.json")
    censored=read(OUT/"censored_prediction_seal.json")
    assert not censored["target_labels_read"] and censored["predictions"]==digest(PRIVATE/"censored_predictions.json")
    assert censored["parent"]==digest(OUT/"outline_prediction_seal.json")
    for uid,arm in read(PRIVATE/"censored_predictions.json").items():
        predictions[uid].update(arm)
    panel,_,prep,boxes=ancestry()
    assert set(primary)==set(parent)==set(predictions) and len(primary)==416
    rows=[]
    for f in panel:
        uid=f["uid"]; old=primary[uid]; p=predictions[uid]
        r={k:old[k] for k in ("uid","target","recording","panel","separation")}
        r.update(outline_eligible=bool(p["outline"]["views"]),opportunity50=old["opportunity50"]=="True",
                 successful_sides_at50=old["successful_sides_at50"],endpoints_inside_upright_gt=old["endpoints_inside_upright_gt"])
        for arm in BASES:
            r["iou_"+arm]=float(old["iou_"+arm])
            r["shift_"+arm]=float(old["shift_"+arm])
        offsets=prep[uid].get("geometry",{}).get("offsets",{})
        for arm in STUDY_ARMS:
            a=p[arm]; views=a["views"]
            box=boxes[uid]["H"] if a["action"]=="H" else boxes[uid]["discrete_boxes"][a["action"]]
            assert box==a["box"]
            r["status_"+arm]=a["status"]
            for i,v in enumerate(views):
                r[f"view{i}_{arm}"]=v["reason"]
                r[f"action{i}_{arm}"]=v["action"]
            for tag,act in ((arm,a["action"]),(arm+"_single",views[0]["action"] if views else "H")):
                r["action_"+tag]=act
                r["iou_"+tag]=float(parent[uid]["iou_"+act])
                r["shift_"+tag]=offsets.get(act,0.)
            if a["action"]=="H":
                assert abs(r["iou_"+arm]-r["iou_H"])<1.e-8
        rows.append(r)
    metrics=[]; paired=[]; funnels=[]; cis={}
    for scope,t,group in subsets(rows):
        tag=dict(scope=scope,target=t,n=len(group))
        for arm in (*BASES,*VARIANTS):
            v=np.array([r["iou_"+arm] for r in group])
            metrics.append(dict(**tag,arm=arm,hits50=int(sum(v>.5)),acc50=100*mean_platform(group,v>.5),
                acc25=100*mean_platform(group,v>.25),miou=100*mean_platform(group,v),
                rescued_opportunities=sum(r["opportunity50"] and r["iou_"+arm]>.5 for r in group)))
        for a,b in ([(x,"H") for x in VARIANTS]+[("outline",b) for b in ("span","endpoint","outline_wrong","outline_rect")]+[("outline_censored",b) for b in ("span","outline","outline_wrong_censored","outline_rect_censored")]):
            paired.append(dict(**tag,arm=a,reference=b,**effect(group,a,b)))
            if t=="macro" and a in ("outline","outline_censored") and b in ("H","outline_wrong","outline_rect","outline_wrong_censored","outline_rect_censored"):
                cis[scope+"/"+a+"-"+b]=uncertainty(group,a,b)
        for cohort,sub in (("all",group),("opportunity50",[r for r in group if r["opportunity50"]])):
            for arm in STUDY_ARMS:
                counts=Counter(r["status_"+arm] for r in sub)
                funnels.append(dict(**tag,subset=cohort,arm=arm,denominator=len(sub),
                    both_in_crop=sum(r["outline_eligible"] for r in sub),**dict(sorted(counts.items())),
                    moved=sum(abs(r["shift_"+arm])>TOL for r in sub)))
    files={name:write_table(name,items) for name,items in (("outline_rows",rows),("outline_metrics",metrics),
        ("outline_paired",paired),("outline_funnel",funnels),("outline_opportunities",[r for r in rows if r["opportunity50"]] ))}
    assert len(rows)==416 and sum(r["opportunity50"] for r in rows)==23
    write(OUT/"outline_evaluation_receipt.json",dict(seal=digest(OUT/"outline_prediction_seal.json"),files=files,
           censor_seal=digest(OUT/"censored_prediction_seal.json"),
           verified_parent_scores=digest(PARENT/"results/rows.csv"),confidence_intervals_pp=cis,rows=len(rows),
           opportunities=23,code=digest(__file__),FINAL_access=False))


if __name__=="__main__":
    run()
