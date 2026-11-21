"""Independent recount. Default requires only the published files and Python's standard library."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE/"results"
ARMS = ("A","B","C","D")


def load(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def records(name):
    with (OUT/(name+".csv")).open() as f:
        return list(csv.DictReader(f))


def subset(items, r):
    h = (row for row in items if (r["target"]=="macro" or row["target"]==r["target"]))
    scope = r["scope"]
    return [z for z in h if {"all":True, "validation":z["panel"]=="validation",
            "extension":z["panel"]!="validation", "recording_separated":z["separation"]=="new_recording",
            "C_D_distinct_prompt":z["C_D_same"]=="False"}[scope]]


def avg(rows, fn):
    vals = defaultdict(list)
    for r in rows:
        vals[r["target"]].append(fn(r))
    return sum(sum(v)/len(v) for v in vals.values())/len(vals)


def check_public():
    result = load(OUT/"evaluation_receipt.json")
    extra = load(OUT/"reversal_receipt.json")
    for meta in (result,extra):
        for name,h in meta["files"].items():
            assert sha(OUT/(name+".csv"))==h
    rows = records("rows")
    mirror = {r["uid"]:r for r in records("reversal_rows")}
    assert len(rows)==len({r["uid"] for r in rows})==len(mirror)==416
    for r in rows:
        other=mirror[r["uid"]]
        for arm in ARMS:
            assert other["side_"+arm]==r["side_"+arm]
            assert other["status_"+arm] in ("agree","conflict","unavailable","abstain")
            if other["status_"+arm]=="agree":
                assert float(other["primary_iou_"+arm])==float(other["relabeled_iou_"+arm])
            r["gated_"+arm]=other["agreement_only_iou_"+arm]
            expected = r["iou_"+arm] if other["status_"+arm]=="agree" else r["iou_H"]
            assert float(r["gated_"+arm])==float(expected)
        r["gated_H"]=r["iou_H"]
        if r["C_D_same"]=="True":
            assert r["side_C"]==r["side_D"] and r["iou_C"]==r["iou_D"]
    recounted=0
    for name,prefix in (("paired","iou_"),("reversal_paired","gated_")):
        for r in records(name):
            pop=subset(rows,r)
            a,b=prefix+r["arm"],prefix+r["reference"]
            assert len(pop)==int(r["n"])
            for threshold,tag in ((.5,"50"),(.25,"25")):
                plus=sum(float(z[a])>threshold>=float(z[b]) for z in pop)
                minus=sum(float(z[b])>threshold>=float(z[a]) for z in pop)
                delta=100*avg(pop, lambda z:int(float(z[a])>threshold)-int(float(z[b])>threshold))
                assert (plus,minus)==(int(r["rescue"+tag]),int(r["harm"+tag]))
                assert abs(delta-float(r["delta"+tag+"_pp"]))<1e-9
            recounted+=1
    for r in records("metrics"):
        pop=subset(rows,r)
        key="iou_"+r["arm"]
        assert sum(float(z[key])>.5 for z in pop)==int(r["hits50"])
        for threshold,k in ((.5,"acc50"),(.25,"acc25")):
            assert abs(100*avg(pop,lambda z:float(z[key])>threshold)-float(r[k]))<1e-9
        assert abs(100*avg(pop,lambda z:float(z[key]))-float(r["miou"]))<1e-9
    return dict(public_frames=len(rows),paired_table_rows_recounted=recounted,
                public_hashes_verified=True, primary_and_reversed_thresholds_match=True)


def check_private():
    import numpy as np
    from side import BASE, ETA, PRIVATE, SPEC, digest, inputs, parse, read
    primary=read(OUT/"prediction_seal.json")
    secondary=read(OUT/"reversal_seal.json")
    prep=read(OUT/"input_seal.json")
    assert primary["predictions"]==digest(PRIVATE/"predictions.json")
    assert prep["protocol"]==digest(SPEC) and prep["geometry_code"]==digest(HERE/"side.py")
    assert prep["preparation_code"]==digest(HERE/"prepare.py")
    assert primary["pins"]["inference_code"]==digest(HERE/"run_qwen.py")
    assert secondary["pins"]["code"]==digest(HERE/"sign_check.py")
    assert secondary["pins"]["plan"]==digest(HERE/"SIGN_CHECK.json")
    assert prep["prepared"]==digest(PRIVATE/"prepared.json")
    panel, original=inputs()
    prepared, predictions=read(PRIVATE/"prepared.json"),read(PRIVATE/"predictions.json")
    counts=0
    maximum=0.
    identical_raw=0
    for folder,index,seal,key in (("replies","reply_hashes",primary,"reply_hashes"),
                                  ("reversed_replies","reversed_hashes",secondary,"hashes")):
        path=PRIVATE/(index+".json")
        assert digest(path)==seal[key]
        hashes=read(path)
        assert len(hashes)==320 and len(list((PRIVATE/folder).glob("*.json")))==320
        for uid,h in hashes.items():
            raw=PRIVATE/folder/(uid+".json")
            assert digest(raw)==h
            obj=read(raw)
            assert obj["uid"]==uid and obj["pins"]==seal["pins"]
            answers=obj["answers"]
            assert set(answers)==set(ARMS)
            if folder=="replies":
                assert {a:parse(s) for a,s in answers.items()}==predictions[uid]["sides"]
                if prepared[uid]["prompts"]["C"]==prepared[uid]["prompts"]["D"]:
                    identical_raw+=int(answers["C"]==answers["D"])
            counts+=len(answers)
    for f in panel:
        uid=f["uid"]
        h=original[uid]["boxes"]["H"]
        k=prepared[uid]
        o=predictions[uid]
        assert o["H"]==k["H"]==h
        if not k["selected"]:
            assert all(b==h for b in o["boxes"].values())
            continue
        h=np.array(h)
        pts=np.array(original[uid]["support_raw"])
        g=k["geometry"]
        j=g["k"]
        angle=h[6]+j*math.pi/2
        unit=np.array([math.cos(angle),math.sin(angle),0.])
        absolute=pts@unit
        lo,hi=min(absolute),max(absolute)
        assert abs(hi-lo-g["measured"][0])<1e-6
        for a,b in o["boxes"].items():
            box=np.array(b)
            assert np.array_equal(box[2:],h[2:])
            change=box[:3]-h[:3]
            maximum=max(maximum,float(np.linalg.norm(change-unit*(change@unit))))
            side=o["sides"].get(a)
            if side not in ETA:
                assert np.array_equal(box,h)
            else:
                # Independent endpoint construction in absolute sensor coordinates.
                center={"positive":lo+h[3+j]/2, "negative":hi-h[3+j]/2,
                        "balanced":(lo+hi)/2}[side]
                assert abs(box[:3]@unit-center)<2.e-6
    assert maximum<1e-10
    return dict(raw_answers_verified=counts,identical_C_D_raw_on_identical_primary_inputs=identical_raw,
                maximum_off_axis_shift_m=maximum, all_fallbacks_and_frozen_geometry_verified=True,
                parent_seal=digest(BASE/"results/prediction_seal.json"))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--private",action="store_true")
    args=cli.parse_args()
    data=check_public()
    if args.private:
        data.update(check_private())
        (OUT/"verification.json").write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(data,indent=2))
