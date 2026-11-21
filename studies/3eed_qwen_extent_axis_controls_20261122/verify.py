"""Verify the equal-volume evaluation and unchanged reference on the anonymous rows."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys


HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
OLD=HERE.parent/"3eed_qwen_ray_equivalent_extent_20261122"/"results"
sys.path.insert(0,str(HERE.parent/"3eed_qwen_full_ordinary_validation_20261121"))
from verify_full import check_comparison, check_metrics


def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def rows(path):
    opener=gzip.open if path.suffix==".gz" else open
    with opener(path,"rt",encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def run():
    seal=json.loads((OUT/"prediction_seal.json").read_bytes())
    score=json.loads((OUT/"evaluation_receipt.json").read_bytes())
    parent=json.loads((OLD/"evaluation_receipt.json").read_bytes())
    assert seal["source_predictions"]==parent["prediction_seal"]==sha(OLD/"prediction_seal.json")
    assert seal["protocol"]==sha(HERE/"PROTOCOL.json") and seal["code"]==score["code"]==sha(HERE/"matched.py")
    assert sha(OUT/"prediction_seal.json")==score["prediction_seal"]
    assert seal["per_description_volume_matched"] and seal["selection_entries_preserved"] and seal["measured_inliers_preserved"]
    assert not seal["target_gt_opened"] and not seal["FINAL_access"] and not score["FINAL_access"]
    assert all(sha(OUT/name)==h for name,h in score["files"].items())
    old={}
    for plat in ("waymo","drone","quad"):
        name=f"rows_{plat}.csv.gz"
        assert parent["files"][name]==sha(OLD/name)
        old.update({r["uid"]:r for r in rows(OLD/name)})
    data=rows(OUT/"rows.csv.gz")
    assert len(data)==len({r["uid"] for r in data})==len(old)==9310
    by_group={}
    for r in data:
        ref=old[r["uid"]]
        assert all(r[k]==ref[k] for k in ("iou_baseline","iou_ray_equivalent","target","case","recording"))
        assert r["changed"]==ref["changed_ray_equivalent"]
        if r["changed"]=="True":
            assert r["audit_cells"]==ref["audit_rays"]
            assert r["audit_changed_certified"]==ref["ray_equivalent_audit_changes"]
        else:
            assert r["iou_baseline"]==r["iou_uniform"]==r["iou_height_first"]==r["iou_ray_equivalent"]
        for key in (("macro","all"),("platform",r["target"]),("recording",r["target"]+"/"+r["recording"])):
            by_group.setdefault(key,[]).append(r)
        if r["changed"]=="True":
            by_group.setdefault(("coverage","changed"),[]).append(r)
    block=by_group["coverage","changed"]
    assert len(block)==seal["changed"]==5904 and sum(int(r["audit_cells"]) for r in block)==score["audit"]["cells"]
    for a in ("certified","uniform","height_first"):
        assert sum(int(r[f"audit_changed_{a}"]) for r in block)==score["audit"][a]
    for name,checker in (("metrics.csv",check_metrics),("paired.csv",check_comparison)):
        for r in rows(OUT/name):
            checker(by_group[r["group"],r["value"]],dict(r,platform=r["group"]))
    report=dict(descriptions=len(data), equal_volume_cases=len(block), same_reference_scores=True,
                groups_recounted=len(by_group), valid=True, production_prediction_changes=0, FINAL_access=False)
    (OUT/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)


if __name__=="__main__":
    run()
