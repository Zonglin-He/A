"""Read back all anonymized outline scores and optionally verify both clean RGB views."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"


def load(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def table(name):
    with (OUT/(name+".csv")).open() as f:
        return list(csv.DictReader(f))


def in_scope(row,scope):
    criteria={"all":True,"recording_separated":row["separation"]=="new_recording",
              "extension":row["panel"]=="recording_extension", "outline_eligible":row["outline_eligible"]=="True"}
    return criteria.get(scope,scope==row["panel"])


def verify_public():
    receipt=load(OUT/"outline_evaluation_receipt.json")
    assert receipt["seal"]==digest(OUT/"outline_prediction_seal.json")
    for name,h in receipt["files"].items():
        assert h==digest(OUT/(name+".csv"))
    data=table("outline_rows")
    assert len(data)==len(set(r["uid"] for r in data))==416
    opp={r["uid"] for r in table("opportunities")}
    assert len(opp)==23 and opp=={r["uid"] for r in table("outline_opportunities")}
    for r in data:
        assert (r["uid"] in opp)==(r["opportunity50"]=="True")
        for arm in ("outline","outline_wrong","outline_rect","outline_censored","outline_wrong_censored","outline_rect_censored"):
            if r["action_"+arm]=="H":
                assert abs(float(r["iou_"+arm])-float(r["iou_H"]))<1.e-8
            else:
                assert r["action0_"+arm]==r["action1_"+arm]==r["action_"+arm]
    for file in ("outline_metrics","outline_paired"):
        for m in table(file):
            g=[r for r in data if (m["target"]=="macro" or r["target"]==m["target"]) and in_scope(r,m["scope"])]
            assert len(g)==int(m["n"])
            n=Counter(r["target"] for r in g)
            w=[1/(len(n)*n[r["target"]]) for r in g]
            a=[float(r["iou_"+m["arm"]]) for r in g]
            if file=="outline_metrics":
                assert sum(z>.5 for z in a)==int(m["hits50"])
                for threshold,key in ((.5,"acc50"),(.25,"acc25")):
                    assert abs(sum(k*(v>threshold) for k,v in zip(w,a))*100-float(m[key]))<1.e-8
                assert abs(sum(k*v for k,v in zip(w,a))*100-float(m["miou"]))<1.e-8
            else:
                b=[float(r["iou_"+m["reference"]]) for r in g]
                for t,key in ((.5,"50"),(.25,"25")):
                    assert int(m["rescue"+key])==sum(x>t>=y for x,y in zip(a,b))
                    assert int(m["harm"+key])==sum(y>t>=x for x,y in zip(a,b))
                    expected=sum(wi*(int(x>t)-int(y>t)) for wi,x,y in zip(w,a,b))*100
                    assert abs(expected-float(m["delta"+key+"_pp"]))<1.e-8
    return dict(ids=416,opportunities=23,paired_rows=len(table("outline_paired")),metric_rows=len(table("outline_metrics")))


def verify_private():
    from PIL import Image, ImageOps
    import numpy as np
    from endpoints import PRIVATE, ancestry
    from outline_geometry import ARMS
    panel,_,prep,states=ancestry()
    ob=load(PRIVATE/"observations.json"); pred=load(PRIVATE/"outline_predictions.json")
    hashes=load(PRIVATE/"outline_hashes.json"); rawseal=load(OUT/"outline_raw_seal.json")
    seal=load(OUT/"outline_prediction_seal.json")
    assert rawseal["replies"]==digest(PRIVATE/"outline_hashes.json") and seal["predictions"]==digest(PRIVATE/"outline_predictions.json")
    assert rawseal["pins"]["protocol"]==digest(HERE/"OUTLINE_PROTOCOL.json")
    cens=load(OUT/"censored_prediction_seal.json")
    assert cens["code"]==digest(HERE/"censored_readout.py")
    assert cens["predictions"]==digest(PRIVATE/"censored_predictions.json")
    for uid,arms in load(PRIVATE/"censored_predictions.json").items():
        pred[uid].update(arms)
    counts=Counter(); worst=0.; crops_checked=0
    for row in panel:
        uid=row["uid"]; obs=ob[uid]; old=states[uid]
        if uid in hashes:
            p=PRIVATE/"outline_replies"/(uid+".json"); assert digest(p)==hashes[uid]
            r=load(p)
            with Image.open(row["rgb"]) as photo:
                crop=photo.convert("RGB").crop(tuple(obs["views"][1]["rect"]))
                canvases=(crop,ImageOps.mirror(crop))
                assert [sha256(im.tobytes()).hexdigest() for im in canvases]==r["images"]
            crops_checked+=2
        for a in (*ARMS,*(k+"_censored" for k in ARMS)):
            decision=pred[uid][a]; label=decision["action"]
            expected=old["H"] if label=="H" else old["discrete_boxes"][label]
            assert decision["box"]==expected
            if not expected or label=="H":
                continue
            axis=np.array(prep[uid]["geometry"]["axis"])
            displacement=np.array(expected[:3])-old["H"][:3]
            worst=max(worst,float(np.linalg.norm(displacement-(displacement@axis)*axis)))
            assert expected[3:]==old["H"][3:]
            counts[a]+=1
    assert crops_checked==478 and worst<1.e-8
    return dict(fresh_rgb_crops_replayed=crops_checked,accepted_actions=dict(counts),maximum_off_axis_m=worst)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__); cli.add_argument("--private",action="store_true")
    result=verify_public()
    if cli.parse_args().private:
        result.update(verify_private(),FINAL_access=False)
        (OUT/"outline_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)
