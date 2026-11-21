"""Independently recount numerical outcomes and occlusion evidence for all treatments."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
OUT = HERE/"results"
VARIANTS = ("intrinsic","support_only","full_or_stay","ray_equivalent")
sys.path.insert(0, str(HERE.parent/"3eed_qwen_full_ordinary_validation_20261121"))
from verify_full import check_comparison, check_metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(name):
    path=OUT/name
    open_file=gzip.open if path.suffix==".gz" else open
    with open_file(path,"rt",encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def match(r, group, value):
    changed=r["changed_ray_equivalent"]=="True"
    return {
        "macro": True,
        "pooled": True,
        "platform": r["target"]==value,
        "category": r["category"]==value,
        "recording": r["target"]+"/"+r["recording"]==value,
        "case": r["case"]==value,
        "coverage": changed and value=="certified_changed",
        "certified_platform": changed and r["target"]==value,
    }[group]


def run():
    seal= json.loads((OUT/"prediction_seal.json").read_bytes())
    evaluation=json.loads((OUT/"evaluation_receipt.json").read_bytes())
    assert not seal["FINAL_access"] and not evaluation["FINAL_access"] and not seal["target_gt_opened"]
    assert seal["protocol"]==sha(HERE/"PROTOCOL.json") and seal["solver"]==sha(HERE/"extent.py")
    assert seal["code"]==evaluation["code"]==sha(HERE/"run_trial.py")
    assert evaluation["prediction_seal"]==sha(OUT/"prediction_seal.json")
    assert all(sha(OUT/f)==digest for f,digest in evaluation["files"].items())
    rows=[]
    for platform, expected in (("waymo",3782),("drone",2792),("quad",2736)):
        items=table(f"rows_{platform}.csv.gz")
        assert len(items)==expected and all(r["target"]==platform for r in items)
        rows+=items
    assert len(rows)==len({r["uid"] for r in rows})==9310
    assert sum(r["active"]=="True" for r in rows)==5910
    assert sum(r["gt_scored"]=="True" for r in rows)==seal["changed_any_variant"]==5904
    certified=[r for r in rows if r["changed_ray_equivalent"]=="True"]
    for r in rows:
        for arm in VARIANTS:
            if r[f"changed_{arm}"]=="True":
                assert r["active"]==r["gt_scored"]=="True"
                assert float(r[f"{arm}_volume_fraction"])<1.
                if arm!="intrinsic":
                    assert int(r[f"{arm}_lost_points"])==0
                if arm in ("ray_equivalent","full_or_stay"):
                    assert int(r[f"{arm}_selection_changes"])==0
            else:
                assert r[f"iou_{arm}"]==r["iou_baseline"]
    blocks={}
    for r in table("ray_checks.csv"):
        group,value,arm=(r[k] for k in ("group","value","arm"))
        if (group,value) not in blocks:
            blocks[group,value]=[v for v in rows if match(v,group,value)]
        block=blocks[group,value]
        changed=[v for v in block if v[f"changed_{arm}"]=="True"]
        assert int(r["n"])==len(block) and int(r["changed"])==len(changed)
        for key,source_key in (("selection_cells","selection_rays"),("audit_cells","audit_rays"),
                               ("selection_changed_cells",f"{arm}_selection_changes"),("audit_changed_cells",f"{arm}_audit_changes"),
                               ("lost_points",f"{arm}_lost_points"),("lost_audit_points",f"{arm}_lost_audit_points")):
            assert int(r[key])==sum(int(v[source_key]) for v in changed)
    for filename, checker in (("metrics.csv",check_metrics),("paired.csv",check_comparison)):
        for r in table(filename):
            checker(blocks[r["group"],r["value"]],dict(r,platform=r["group"]))
    result=dict(descriptions=len(rows), active=5910, changed=len(certified), groups_recounted=len(blocks),
                selection_entries_and_measured_inliers_preserved=True,
                selection_cells=sum(int(r["selection_rays"]) for r in certified),
                independent_audit_cells=sum(int(r["audit_rays"]) for r in certified),
                independent_audit_changed_cells=sum(int(r["ray_equivalent_audit_changes"]) for r in certified),
                valid=True, FINAL_access=False, production_prediction_changes=0)
    (OUT/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)


if __name__=="__main__":
    run()
