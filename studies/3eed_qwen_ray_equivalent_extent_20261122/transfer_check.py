"""Replay the locked geometry on the existing 128 recording-separated Ordinary TRAIN frames."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from live import predict
import run_trial as study
from cray_simple import predict as reference


SOURCE = study.HERE.parent/"3eed_qwen_four_scale_recording_confirmation_20261121"
LOCK = study.HERE/"METHOD_LOCK.json"
LOCAL = study.WORK/"extension_predictions.json"
roster = study.roster


def read_inputs():
    lock=roster.read(LOCK)
    assert not lock["FINAL_access"] and not lock["training"]
    assert lock["entry"]["sha256"]==roster.sha(study.HERE/"live.py")
    for name,digest in lock["dependencies"].items():
        assert digest==roster.sha(study.HERE/name)
    assert lock["frozen_reference"]["sha256"]==roster.sha(study.origin.ACTIVE/"METHOD_LOCK.json")
    assert lock["live_replay"]==roster.sha(study.OUT/"live_replay.json")
    panel=SOURCE/"private/panel.json"
    pin=roster.read(SOURCE/"results/panel_seal.json")
    model=roster.read(SOURCE/"results/model_seal.json")
    assert not pin["FINAL_access"] and not model["target_labels_read"]
    assert roster.sha(panel)==pin["panel"]==model["pins"]["panel"]
    all_rows=roster.read(panel)
    assert len(all_rows)==416
    rows=[r for r in all_rows if r["panel"]=="recording_extension"]
    assert len(rows)==len({r["uid"] for r in rows})==128
    assert Counter(r["target"] for r in rows)=={"drone":64,"quad":64}
    assert all(r["separation"]=="new_recording" for r in rows)
    val,_=roster.panel()
    key=lambda r: (r["target"],Path(r["calibration"]).parent.parent.name)
    assert not ({key(r) for r in rows}&{key(r) for r in val})
    for field in ("point_sha","rgb_sha"):
        assert not ({r[field] for r in rows}&{r[field] for r in val})
    hashes=roster.read(SOURCE/"private/reply_hashes.json")
    assert model["ordered_reply_hash"]==hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    return rows,model["pins"],hashes


def generate():
    frames,pins,hashes=read_inputs()
    output=[]
    for f in frames:
        answers={}
        assert roster.sha(f["points"])==f["point_sha"]
        for task in ("2d","3d"):
            report=f["uid"]+"_"+task
            path=SOURCE/"private/replies"/(report+".json")
            assert hashes[report]==roster.sha(path)
            response=roster.read(path)
            assert (response["uid"],response["task"],response["pins"])==(f["uid"],task,pins)
            answers[task]=response
        a,b=reference(f,answers),predict(f,answers)
        assert all(a[k]==b[k] for k in ("applicable","heading","selection_rays"))
        changed=b["extent_corrected"]
        if not changed:
            assert a["box"]==b["box"]
        output.append(dict(uid=f["uid"],baseline=a["box"],corrected=b["box"],changed=changed,applicable=a["applicable"]))
    roster.write(LOCAL,output)
    roster.write(study.OUT/"extension_seal.json",dict(
        descriptions=128,disjoint_recordings_and_observations=True,frozen_method=roster.sha(LOCK),
        source_panel=roster.sha(SOURCE/"results/panel_seal.json"),source_model=roster.sha(SOURCE/"results/model_seal.json"),
        payload=roster.sha(LOCAL),code=roster.sha(__file__),target_gt_opened=False,
        spec="existing Ordinary TRAIN Drone and Quad recording extension",new_Qwen_queries=0,FINAL_access=False,
    ))


def score():
    frames,_,_=read_inputs()
    pin=study.OUT/"extension_seal.json"
    sealed=roster.read(pin)
    assert not sealed["target_gt_opened"] and not sealed["FINAL_access"]
    assert sealed["code"]==roster.sha(__file__) and sealed["frozen_method"]==roster.sha(LOCK) and sealed["payload"]==roster.sha(LOCAL)
    output=[]
    for f,p in zip(frames,roster.read(LOCAL),strict=True):
        assert f["uid"]==p["uid"] and f["panel"]=="recording_extension"
        path=Path(f["calibration"])
        assert path.name=="meta_info.json" and path.parents[2]==study.DATA/f["target"]
        gt=study.score_full.canonical_box(study.score_full.truth_for_slot(f,roster.read(path)["ground_info"]))
        if p["baseline"] is None:
            assert p["corrected"] is None
            a,b=0.,0.
        else:
            (a,b),_=study.score_full.score_boxes(np.asarray([p["baseline"],p["corrected"]]),gt)
        output.append(dict(uid=f["uid"],target=f["target"],recording=f["recording"],changed=p["changed"],
                           applicable=p["applicable"],iou_baseline=float(a),iou_ray_equivalent=float(b)))
    assert len(output)==128
    groups={}
    for key in ("pooled","drone","quad"):
        items=[r for r in output if key in ("pooled",r["target"])]
        a,b=(np.array([r[f"iou_{arm}"] for r in items]) for arm in ("baseline","ray_equivalent"))
        groups[key]=dict(n=len(items),recordings=len({r["recording"] for r in items}),
                         changed=sum(r["changed"] for r in items),applicable=sum(r["applicable"] for r in items),
                         baseline=dict(acc25=float(100*np.mean(a>.25)),acc50=float(100*np.mean(a>.5)),miou=float(100*a.mean())),
                         ray_equivalent=dict(acc25=float(100*np.mean(b>.25)),acc50=float(100*np.mean(b>.5)),miou=float(100*b.mean())),
                         rescue50=int(((a<=.5)&(b>.5)).sum()),harm50=int(((a>.5)&(b<=.5)).sum()))
    study.score_full.OUT=study.OUT
    row_file=study.score_full.write_rows("extension_rows.csv",output)
    roster.write(study.OUT/"extension_results.json",dict(
        seal=roster.sha(pin),rows=row_file,groups=groups,scope="Previously exposed Ordinary TRAIN recording extension",
        disjoint_from_ordinary_validation=True,production_prediction_changes=0,new_Qwen_queries=0,FINAL_access=False,
    ))
    print(groups,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("propose","score"))
    (generate if parser.parse_args().phase=="propose" else score)()
