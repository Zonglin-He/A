"""Score sealed answers and held-out LiDAR without exposing evaluation labels."""
from collections import defaultdict
import csv
from pathlib import Path
import re

import numpy as np

from common import ARMS, OUT, PLATFORMS, PRIVATE, PROTOCOL, digest, read, write
from geometry import calibration, transform_xyz, yaw_rotation
from maps import map_arm


NUM = r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
UNIT = r"(?:millimeters?|centimeters?|kilometers?|meters?|metres?|inches?|feet|foot|cm|mm|km|ft|in|m)"
NUMBER = re.compile(r"(?<![\w.])(" + NUM + r")(?:\s*(" + UNIT + r")\b)?", re.I)
OPT = re.compile(
    r"^(?:the\s+)?(?:(?:(?:correct\s+)?answer\s*(?:is\s*)?[:=]?|option|object|pixel)\s*)?\(?([AB])\b\)?",
    re.I,
)
SPECIAL = re.compile(r"<(?:pad|eos|s|/s|end_of_turn|start_of_turn)>")
FAMILY = {
    "distance_a":"distance","distance_b":"distance","height_a":"height","height_b":"height",
    "separation_ab":"separation","separation_ba":"separation",
    "pixel_near":"pixel_order","pixel_far":"pixel_order","pixel_xyz":"pixel_xyz",
    "object_near":"object_order","object_far":"object_order","static_right":"static_right",
    "after_turn_right":"yaw_90","after_turn_left":"yaw_90",
}


def meters(number,unit):
    value = float(number.replace(",", ""))
    if not unit:
        return value
    unit = unit.lower()
    if unit == "cm" or unit.startswith("centimeter"):
        return value * .01
    if unit == "mm" or unit.startswith("millimeter"):
        return value * .001
    if unit == "km" or unit.startswith("kilometer"):
        return value * 1000
    if unit in ("foot","feet","ft"):
        return value * .3048
    if unit in ("in","inch","inches"):
        return value * .0254
    return value


def parse(raw, gold):
    clean = SPECIAL.sub("",raw).strip().strip("`*").strip()
    if isinstance(gold,str):
        choice = OPT.match(clean)
        if choice is None:
            return None
        letter = choice[1].upper()
        rest = clean[choice.end():]
        if rest and not re.match(r"^\s*(?:[).,:;!]|$|is\b|because\b|/|or\b)", rest, re.I):
            return None
        other = "AB"[letter == "A"]
        if re.match(r"^\s*(?:/|or)\s*\(?"+other+r"\b",rest,re.I):
            return None
        if re.search(r"\b(?:answer\s*(?:is|:)?|correct answer\s*(?:is|:)?)\s*\(?"+other+r"\b",rest,re.I):
            return None
        return letter
    values = [meters(x,y) for x,y in NUMBER.findall(clean)]
    if not np.isfinite(values).all():
        return None
    if isinstance(gold,list):
        return values if len(values)==3 and values[2]>0. else None
    return values[0] if len(values)==1 and values[0]>=0. else None


def outcome(answer,gold):
    if answer is None:
        return dict(correct=0,valid=0,abs_error=None,point_error=None)
    if isinstance(gold,str):
        return dict(correct=int(answer==gold),valid=1,abs_error=None,point_error=None)
    if isinstance(gold,list):
        delta=np.array(answer)-np.array(gold)
        err,point=abs(delta[2]),float(np.linalg.norm(delta))
        return dict(correct=int(err<=.25*gold[2]),valid=1,abs_error=float(err),point_error=point)
    err=abs(answer-gold)
    return dict(correct=int(err<=.25*gold),valid=1,abs_error=float(err),point_error=None)


def csv_write(path,rows):
    if not rows:
        raise ValueError(f"Empty output for {path.name}")
    with path.open("w",newline="",encoding="utf8") as f:
        wr=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)


def changed(a,b):
    if a is None or b is None or isinstance(a,str):
        return a!=b
    return np.max(abs(np.array(a)-np.array(b)))>.01


def describe_error(estimate, points, selected):
    n=int(selected.sum())
    if not n:
        return dict(n=0,depth_mae=None,depth_rmse=None,depth_absrel=None,xyz_mae=None)
    diff=estimate[selected]-points[selected]
    return dict(n=n,depth_mae=float(np.mean(abs(diff[:,2]))),depth_rmse=float(np.sqrt(np.mean(diff[:,2]**2))),
                depth_absrel=float(np.mean(abs(diff[:,2])/points[selected,2])),
                xyz_mae=float(np.mean(np.linalg.norm(diff,axis=1))))


def evaluate_map(frame, labels):
    key=frame["uid"]
    with np.load(PRIVATE/"observations"/(key+".npz"),allow_pickle=False) as f:
        point,cells,conflict = f["camera"].astype(float),f["cells"],f["collision"]
    spec,size,transform=calibration(frame)
    lidar=transform_xyz(point,np.linalg.inv(transform))
    memberships=[]
    for item in labels["objects"]:
        b=np.asarray(item["box"],float)
        in_box=(abs((lidar-b[:3])@yaw_rotation(b[6]))<=b[3:6]/2+.05).all(1)
        memberships.append(in_box)

    frame_geometry, by_obj, estimates={},{},{}
    with np.load(PRIVATE/"maps"/(key+".npz"),allow_pickle=False) as f:
        held=f["held"]
        for arm in ARMS:
            field=map_arm(f,arm)
            xy=field[:3,cells[:,1],cells[:,0]].T.astype(float)
            valid=field[3,cells[:,1],cells[:,0]]==1
            arm_rows={}
            for tag, mask in [
                ("all",held),("lt10",held&(point[:,2]<10)),("10to20",held&(point[:,2]>=10)&(point[:,2]<20)),
                ("20to40",held&(point[:,2]>=20)&(point[:,2]<40)),("ge40",held&(point[:,2]>=40)),
                ("collision",held&conflict),("no_collision",held&~conflict)
            ]:
                arm_rows[tag]=describe_error(xy,point,mask)
            frame_geometry[arm]=arm_rows
            by_obj[arm]=[describe_error(xy,point,held&mask) for mask in memberships]
            estimates[arm]=(xy,valid)
    return frame_geometry, by_obj, estimates


def consistency_rows(questions, outputs, answers, mode, frame):
    items={q["task"]:q["qid"] for q in questions if q["uid"]==frame["uid"] and q["mode"]==mode}
    pairs=[("pixel_reverse","pixel_near","pixel_far"),("distance_reverse","object_near","object_far"),
           ("separation_symmetry","separation_ab","separation_ba"),("rotations_reverse","after_turn_right","after_turn_left")]
    rows=[]
    for kind,a,b in pairs:
        if not {a,b}<=items.keys():
            continue
        p,q=items[a],items[b]
        for arm in ARMS:
            v,y=outputs[p,arm],outputs[q,arm]
            consistent=False
            if v is not None and y is not None:
                if kind=="separation_symmetry":
                    consistent=abs(v-y)<=max(.25,.10*(v+y)/2)
                    assert answers[p]["value"]==answers[q]["value"]
                else:
                    consistent=v!=y
                    assert answers[p]["value"]!=answers[q]["value"]
            rows.append(dict(target=frame["target"],mode=mode,uid=frame["uid"],recording=frame["recording"],
                task=kind,arm=arm,consistent=int(consistent),valid_both=int(v is not None and y is not None),
                both_correct=int(outcome(v,answers[p]["value"])["correct"] and outcome(y,answers[q]["value"])["correct"])))
    linked = ("distance_a","distance_b","object_near")
    if set(linked) <= items.keys():
        a, b, near = [items[name] for name in linked]
        for arm in ARMS:
            da, db, choice = [outputs[qid,arm] for qid in (a,b,near)]
            available = da is not None and db is not None and choice is not None and abs(da-db)>.05
            agrees = available and choice == ("A" if da<db else "B")
            all_correct = all(outcome(outputs[qid,arm],answers[qid]["value"])["correct"] for qid in (a,b,near))
            rows.append(dict(target=frame["target"],mode=mode,uid=frame["uid"],recording=frame["recording"],
                task="distance_vs_order",arm=arm,consistent=int(agrees),valid_both=int(available),
                both_correct=int(all_correct)))
    return rows


def pointmap_readout(question, gold, estimate):
    if question["level"]!="L0":
        return None
    chosen=gold["lidar_rows"]
    value,valid=estimate
    if not valid[chosen].all():
        return None
    if question["task"]=="pixel_xyz":
        return value[chosen[0]].tolist()
    a,b=value[chosen,2]
    if a==b:
        return None
    nearer=a<b
    return "A" if nearer==(question["task"]=="pixel_near") else "B"


def run():
    score_seal = read(OUT/"prediction_seal.json")
    panel = read(OUT/"panel_seal.json")
    scoring_plan = read(OUT/"scoring_plan.json")
    assert scoring_plan["score_code"]==digest(__file__) and scoring_plan["protocol"]==digest(PROTOCOL)
    assert score_seal["stamp"]["panel"]==digest(OUT/"panel_seal.json")
    assert panel["answers"]==digest(PRIVATE/"answers.json")
    assert panel["ground_objects"]==digest(PRIVATE/"candidate_truth.json")
    assert panel["protocol"]==digest(PROTOCOL)
    assert panel["questions"]==digest(PRIVATE/"questions.json")
    assert score_seal["answers_hashes"]==digest(PRIVATE/"answers_frozen_hashes.json")
    hashes=read(PRIVATE/"answers_frozen_hashes.json")
    qs=read(PRIVATE/"questions.json")
    by_uid=defaultdict(list)
    for q in qs:
        by_uid[q["uid"]].append(q)
    answer_key=read(PRIVATE/"answers.json")
    gt=read(PRIVATE/"candidate_truth.json")
    frames=read(PRIVATE/"frames.json")
    observed,geometry,logic=[],[],[]
    data_source=[]
    for frame in frames:
        uid=frame["uid"]
        prediction=PRIVATE/"answers_frozen"/(uid+".json")
        mapfile=PRIVATE/"maps"/(uid+".npz")
        assert digest(prediction)==hashes[uid]
        assert digest(mapfile)==read(PRIVATE/"map_hashes.json")[uid]
        assert digest(PRIVATE/"observations"/(uid+".npz"))==frame["observation_sha"]
        frame_rows=read(prediction)["answers"]
        outputs={}
        raw_text={}
        for r in frame_rows:
            k=(r["qid"],r["arm"])
            assert k not in outputs and r["arm"] in ARMS
            raw_text[k]=r["answer"]
            outputs[k]=parse(r["answer"],answer_key[r["qid"]]["value"])
        assert set(outputs)=={(q["qid"],a) for q in by_uid[uid] for a in ARMS}
        geom, per_object, estimate=evaluate_map(frame,gt[uid])
        for arm,regions in geom.items():
            for group,metrics in regions.items():
                geometry.append(dict(uid=uid,target=frame["target"],recording=frame["recording"],arm=arm,
                    group=group,**metrics))
        data_source.append({k:v for k,v in gt[uid].items() if k!="objects"} | {"target":frame["target"],"uid":uid})
        for mode in ("language","oracle_identity"):
            logic.extend(consistency_rows(by_uid[uid],outputs,answer_key,mode,frame))
        for q in by_uid[uid]:
            qid=q["qid"]
            label=answer_key[qid]
            ids=label.get("objects",[])
            evidence_base=[per_object["moge"][j] for j in ids]
            for arm in ARMS:
                ans=outputs[qid,arm]
                values=outcome(ans,label["value"])
                evidence=[per_object[arm][j] for j in ids]
                eligible=bool(evidence) and min(r["n"] for r in evidence)>=3
                n=min((r["n"] for r in evidence),default=0)
                changed_all=eligible and all(x["depth_mae"]<y["depth_mae"]-1e-5 for x,y in zip(evidence,evidence_base))
                direct=pointmap_readout(q,label,estimate[arm])
                direct_result=outcome(direct,label["value"]) if q["level"]=="L0" else {}
                observed.append({k:v for k,v in q.items() if k!="prompt"} | dict(
                    family=FAMILY[q["task"]],arm=arm,**values,
                    raw_change=int(raw_text[qid,arm]!=raw_text[qid,"moge"]),
                    answer_change=int(changed(ans,outputs[qid,"moge"])),
                    min_object_holdout=n,all_objects_depth_improved=int(changed_all),
                    object_depth_mae=float(np.mean([r["depth_mae"] for r in evidence])) if eligible else None,
                    global_depth_mae=geom[arm]["all"]["depth_mae"],
                    direct_correct=direct_result.get("correct"),direct_valid=direct_result.get("valid"),
                    direct_depth_error=direct_result.get("abs_error"),direct_point_error=direct_result.get("point_error"),
                    static_changes=int(label.get("changes_static",False)),static_available=int(label.get("static_available",False))))
    assert len(observed)==score_seal["count"]
    csv_write(OUT/"qa_rows.csv",observed)
    csv_write(OUT/"geometry_rows.csv",geometry)
    csv_write(OUT/"consistency_rows.csv",logic)
    write(OUT/"sync_audit.json",data_source)
    write(OUT/"score_receipt.json",dict(question_answers=len(observed),questions=len(qs),physical_frames=len(frames),
          qa_rows=digest(OUT/"qa_rows.csv"),geometry_rows=digest(OUT/"geometry_rows.csv"),
          consistency_rows=digest(OUT/"consistency_rows.csv"),predictions=digest(OUT/"prediction_seal.json"),
          evaluated_all=True,labels_read_after_prediction_seal_for_scoring=True,scoring_plan=digest(OUT/"scoring_plan.json"),
          FINAL_access=False))


if __name__=="__main__":
    run()
