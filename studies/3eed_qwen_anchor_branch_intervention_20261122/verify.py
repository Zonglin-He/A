"""Recount the complete sealed evaluation without any annotations."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
OUT = HERE/"results"
ARMS = ("baseline", "selection_only", "branch", "tail", "axis")
sys.path.insert(0, str(HERE.parent/"3eed_qwen_full_ordinary_validation_20261121"))
from verify_full import check_comparison, check_metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(name):
    path = OUT/name
    opener = gzip.open if path.suffix==".gz" else open
    with opener(path,"rt",encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def in_group(row, group, value):
    if group in ("macro","pooled"):
        return True
    if group == "recording":
        return value==row["target"]+"/"+row["recording"]
    if group in ("platform","matched_platform"):
        return value==row["target"] and (group=="platform" or row["matched_budget"]=="True")
    if group == "category":
        return value==row["category"]
    if group == "case":
        return value==row["origin_case"]
    assert group == "coverage"
    return row[{"applicable":"active","branch_proposed":"gt_scored","matched_budget":"matched_budget"}[value]]=="True"


def run():
    sealed = json.loads((OUT/"prediction_seal.json").read_bytes())
    scored = json.loads((OUT/"evaluation_receipt.json").read_bytes())
    assert not sealed["FINAL_access"] and not scored["FINAL_access"] and not sealed["target_gt_opened"]
    assert sealed["protocol"]==sha(HERE/"PROTOCOL.json") and sealed["code"]==sha(HERE/"propose.py")
    assert scored["prediction_seal"]==sha(OUT/"prediction_seal.json") and scored["code"]==sha(HERE/"evaluate_trial.py")
    assert all(sha(OUT/name)==h for name,h in scored["files"].items())
    rows = []
    for target, n in (("waymo",3782),("drone",2792),("quad",2736)):
        block = table(f"rows_{target}.csv.gz")
        assert len(block)==n and all(r["target"]==target for r in block)
        rows.extend(block)
    assert len(rows)==len({r["uid"] for r in rows})==9310
    assert len({r["recording"] for r in rows})==98 and sum(r["active"]=="True" for r in rows)==5910
    assert sum(r["gt_scored"]=="True" for r in rows)==sealed["connected_deletion_proposed"]==scored["ordinary_GT_descriptions"]
    for r in rows:
        baseline = float(r["iou_baseline"])
        for arm in ARMS[1:]:
            if r[f"changed_{arm}"]=="True":
                assert r["active"]==r["gt_scored"]=="True"
                assert float(r[f"{arm}_selection_delta"]) < -1.e-12
                if arm!="selection_only":
                    assert int(r["audit_cells"])>=5 and float(r[f"{arm}_audit_delta"]) < -1.e-12
                family = "branch" if arm=="selection_only" else arm
                assert float(r[f"iou_{arm}"])<=float(r[f"{family}_oracle"])+1.e-12
            else:
                assert float(r[f"iou_{arm}"])==baseline
        if r["changed_branch"]=="True":
            assert r["changed_selection_only"]=="True" and r["iou_branch"]==r["iou_selection_only"]
        if r["matched_budget"]=="True":
            assert r["branch_proposals"]==r["tail_proposals"]==r["axis_proposals"]
    counts = table("coverage.csv")
    by_group = {}
    for item in counts:
        tag = item["group"], item["value"]
        block = [r for r in rows if in_group(r,*tag)]
        by_group[tag] = block
        assert int(item["n"])==len(block)
        for arm in ARMS[1:]:
            assert int(item[f"{arm}_changed"])==sum(r[f"changed_{arm}"]=="True" for r in block)
        for family in ("branch","tail","axis"):
            assert int(item[f"{family}_oracle_rescues50"])==sum(float(r["iou_baseline"])<=.5<float(r[f"{family}_oracle"]) for r in block)
    for name, checker in (("metrics.csv",check_metrics),("paired.csv",check_comparison)):
        for row in table(name):
            group, value = row["group"], row["value"]
            checker(by_group[group,value], dict(row,platform=group))
    result = dict(descriptions=len(rows), active=5910, recordings=98, groups_recounted=len(by_group),
                  valid=True, production_prediction_changes=0, FINAL_access=False)
    (OUT/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)


if __name__=="__main__":
    run()
