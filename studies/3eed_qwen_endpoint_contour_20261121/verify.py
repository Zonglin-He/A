"""Recount public paired results; optionally audit the private camera inputs and boxes."""
import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"


def load(name):
    return json.loads((OUT/name).read_bytes())


def rows(name):
    with (OUT/(name+".csv")).open() as f:
        return list(csv.DictReader(f))


def subset(data,tag):
    return [r for r in data if (tag["target"]=="macro" or r["target"]==tag["target"]) and
        {"all":True,"validation":r["panel"]=="validation","extension":r["panel"]!="validation",
         "recording_separated":r["separation"]=="new_recording","paired_eligible":r["eligible"]=="True",
         "visible_mismatch":r["eligible"]=="True" and float(r["mark_shift_px"] or 0)>=10}[tag["scope"]]]


def macro(group,values):
    bins=defaultdict(list)
    for r,x in zip(group,values,strict=True):
        bins[r["target"]].append(x)
    return sum(sum(b)/len(b) for b in bins.values())/len(bins)


def verify_public():
    receipt=load("evaluation_receipt.json")
    for name,h in receipt["files"].items():
        assert sha256((OUT/(name+".csv")).read_bytes()).hexdigest()==h
    data=rows("rows")
    opp=rows("opportunities")
    assert len(data)==len({r["uid"] for r in data})==416 and len(opp)==23
    assert {r["uid"] for r in opp}=={r["uid"] for r in data if float(r["iou_H"])<=.5<float(r["iou_oracle"])}
    for arm in ("span","endpoint","mismatch","span_all"):
        for r in data:
            if r["decision_"+arm]!="agree":
                assert r["label_"+arm]=="H" and float(r["iou_"+arm])==float(r["iou_H"])
    for m in rows("metrics"):
        g=subset(data,m)
        a=[float(r["iou_"+m["arm"]]) for r in g]
        assert len(g)==int(m["n"]) and sum(x>.5 for x in a)==int(m["hits50"])
        assert abs(100*macro(g,[x>.5 for x in a])-float(m["acc50"]))<1e-9
        assert abs(100*macro(g,[x>.25 for x in a])-float(m["acc25"]))<1e-9
        assert abs(100*macro(g,a)-float(m["miou"]))<1e-9
    pairs=rows("paired")
    for row in pairs:
        g=subset(data,row)
        av=[float(z["iou_"+row["arm"]]) for z in g]
        bv=[float(z["iou_"+row["reference"]]) for z in g]
        for t,k in ((.5,"50"),(.25,"25")):
            pos=sum(a>t>=b for a,b in zip(av,bv))
            neg=sum(b>t>=a for a,b in zip(av,bv))
            delta=100*macro(g,[int(a>t)-int(b>t) for a,b in zip(av,bv)])
            assert (pos,neg)==(int(row["rescue"+k]),int(row["harm"+k]))
            assert abs(delta-float(row["delta"+k+"_pp"]))<1e-9
    return dict(frames=len(data),opportunities=len(opp),paired_rows_verified=len(pairs),
                public_files_and_metrics_verified=True)


def verify_private():
    import numpy as np
    from PIL import Image
    from endpoints import (ENCODINGS, HERE as study, MODES, OUT as public, PRIVATE, SPEC, SWAP, ancestry,
                           canvas, digest, display_uv, message, parse, read)
    from geometry import camera, rigid_xyz, sensor_pixels
    p=load("prediction_seal.json"); q=load("input_seal.json")
    assert sha256((study/"run_qwen.py").read_bytes()).hexdigest()==p["pins"]["code"]
    assert all(digest(study/k)==v for k,v in q["code"].items())
    assert q["protocol"]==digest(SPEC) and q["observations"]==digest(PRIVATE/"observations.json")
    assert digest(PRIVATE/"reply_hashes.json")==p["reply_hashes"]
    assert digest(PRIVATE/"predictions.json")==p["predictions"]
    hashes=read(PRIVATE/"reply_hashes.json")
    prediction=read(PRIVATE/"predictions.json"); observed=read(PRIVATE/"observations.json")
    panel,frozen,prep,states=ancestry()
    worst_off_axis=worst_projection=worst_endpoint=0.; calls=checked_images=0
    for f in panel:
        uid=f["uid"]; obs=observed[uid]; o=prediction[uid]; prev=states[uid]
        frame_modes=MODES if obs["status"]=="eligible" else ("span",)
        if obs["movable"]:
            path=PRIVATE/"replies"/(uid+".json"); raw=read(path)
            assert hashes[uid]==digest(path) and raw["pins"]==p["pins"]
            with Image.open(f["rgb"]) as image:
                rgb=image.convert("RGB")
            for enc in ENCODINGS:
                rev=enc=="reverse"
                assert sha256(message(f,prep[uid],rev).encode()).hexdigest()==raw["prompts"][enc]
                for arm in frame_modes:
                    key=arm+"_"+enc
                    coords=None if arm=="span" else obs["correct" if arm=="endpoint" else "wrong"]
                    im=canvas(rgb,obs["views"],coords,rev)
                    assert sha256(im.tobytes()).hexdigest()==raw["canvases"][key]
                    checked_images+=1
                    if arm=="span" and obs["status"]!="eligible":
                        arm="span_all"
                    assert parse(raw["answers"][key])==o[arm][enc]
            calls+=len(raw["answers"])
        if obs["status"]=="eligible":
            true=np.asarray(obs["correct"]); wrong=np.asarray(obs["wrong"])
            np.testing.assert_allclose(true[1]-true[0],wrong[1]-wrong[0],rtol=0.,atol=1e-9)
            for v in obs["views"]:
                assert [display_uv(x,v) is None for x in true]==[display_uv(x,v) is None for x in wrong]
            file=Path(f["points"])
            xyz=(np.load(file,mmap_mode="r",allow_pickle=False) if file.suffix==".npy"
                 else np.fromfile(file,"<f4").reshape(-1,4))
            raw_xyz=xyz[[e["raw_index"] for e in obs["endpoints"]],:3]
            spec,_,mat=camera(f)
            rep=sensor_pixels(rigid_xyz(raw_xyz.astype(float),mat),spec)
            worst_projection=max(worst_projection,float(np.max(np.linalg.norm(true-rep,axis=1))))
            axis=np.asarray(prep[uid]["geometry"]["axis"])
            values=np.array(frozen[uid]["support_raw"])@axis
            station=raw_xyz@axis
            worst_endpoint=max(worst_endpoint,float(max(abs(station-np.array([min(values),max(values)])))))
        for arm,item in o.items():
            h=prev["H"]
            if item["status"]=="agree":
                assert item["label"]==SWAP[item["reverse"]]==item["forward"]
                assert item["box"]==prev["discrete_boxes"][item["label"]]
            else:
                assert item["label"]=="H" and item["box"]==h
            if h is not None:
                assert item["box"][2:]==h[2:]
                shift=np.array(item["box"][:3])-h[:3]
                if np.any(shift):
                    u=np.array(prep[uid]["geometry"]["axis"])
                    worst_off_axis=max(worst_off_axis,float(np.linalg.norm(shift-u*(u@shift))))
    assert calls==checked_images==p["calls"] and worst_off_axis<1e-10
    assert worst_projection<1e-6 and worst_endpoint<1e-4
    return dict(raw_answers_verified=calls,rgb_inputs_rerendered=checked_images,
                maximum_endpoint_reprojection_px=worst_projection,maximum_axial_station_error_m=worst_endpoint,
                maximum_off_axis_shift_m=worst_off_axis,FINAL_access=False)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--private",action="store_true")
    flags=cli.parse_args()
    result=verify_public()
    if flags.private:
        result.update(verify_private())
        (OUT/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
