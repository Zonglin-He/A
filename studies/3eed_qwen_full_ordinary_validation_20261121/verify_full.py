"""Recompute the complete public benchmark directly from anonymous scalar rows."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
TARGETS=("waymo","drone","quad")
ARMS=("A","H","L7","C_pca","C_ray")
EXPECTED={"waymo":3782,"drone":2792,"quad":2736}
EPS=1.e-8


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def table(name):
    link=OUT/name
    if name.endswith(".gz"):
        with gzip.open(link,"rt",encoding="utf-8",newline="") as f:
            return list(csv.DictReader(f))
    with link.open(newline="") as f:
        return list(csv.DictReader(f))


def average(rows,values,macro=False):
    if not macro:
        return mean(values)
    platform=defaultdict(list)
    for r,v in zip(rows,values,strict=True):
        platform[r["target"]].append(v)
    return mean(mean(x) for x in platform.values())


def check_metrics(group,row):
    values=[float(r["iou_"+row["arm"]]) for r in group]
    assert int(row["n"])==len(values)
    is_macro=row["platform"]=="macro"
    assert abs(100*average(group,values,is_macro)-float(row["miou"]))<EPS
    for threshold,suffix in ((.25,"25"),(.50,"50")):
        passes=[x>threshold for x in values]
        assert sum(passes)==int(row["hits"+suffix])
        assert abs(100*average(group,passes,is_macro)-float(row["acc"+suffix]))<EPS


def check_comparison(group,row):
    assert len(group)==int(row["n"])
    a,b=([float(r["iou_"+k]) for r in group] for k in (row["arm"],row["reference"]))
    is_macro=row["platform"]=="macro"
    for t,tag in ((.25,"25"),(.50,"50")):
        pos=sum(i>t>=j for i,j in zip(a,b))
        neg=sum(j>t>=i for i,j in zip(a,b))
        assert (pos,neg)==(int(row["rescue"+tag]),int(row["harm"+tag]))
        assert abs(100*average(group,[int(i>t)-int(j>t) for i,j in zip(a,b)],is_macro)-float(row["delta_acc"+tag+"_pp"]))<EPS
    diff=[x-y for x,y in zip(a,b)]
    assert abs(100*average(group,diff,is_macro)-float(row["delta_miou_pp"]))<EPS
    assert sum(x>EPS for x in diff)==int(row["better_iou"]) and sum(x<-EPS for x in diff)==int(row["worse_iou"])


def run():
    a=read(OUT/"evaluation_receipt.json")
    assert a["scoring_protocol"]==sha(HERE/"SCORING.json") and a["code"]==sha(HERE/"score_full.py")
    assert a["prediction_seal"]==sha(OUT/"prediction_seal.json")
    for file,s in a["files"].items():
        assert sha(OUT/file)==s
    coll={t:table("rows_"+t+".csv.gz") for t in TARGETS}
    assert {t:len(r) for t,r in coll.items()}==EXPECTED
    rows=sum(coll.values(),[])
    assert len(rows)==len({r["uid"] for r in rows})==a["descriptions"]==9310
    assert len({r["frame"] for r in rows})==read(OUT/"input_seal.json")["frames"]
    assert len({r["recording"] for r in rows})==a["recordings"]==98
    assert sum(r["cache_reuse"]=="True" for r in rows)==386
    for r in rows:
        for k in ARMS:
            assert 0<=float(r["iou_"+k])<=1.+EPS
        if r["qwen_3d"]!="ok":
            assert all(float(r["iou_"+a])==0. for a in ARMS)
        if r["full"]=="False":
            assert r["iou_H"]==r["iou_A"]
        if r["single"]=="False":
            assert r["iou_L7"]==r["iou_A"]
        if r["strict"]=="False":
            assert r["iou_C_pca"]==r["iou_A"]
        if r["chooses_qwen"]=="False":
            assert r["iou_C_ray"]==r["iou_C_pca"]
        else:
            assert int(r["selection_rays"])>=5
    coll.update(pooled=rows,macro=rows)
    rec=defaultdict(list)
    for r in rows:
        rec[r["target"],r["recording"]].append(r)
    rcounts=Counter()
    for name,check,key in (
        ("metrics.csv",check_metrics,lambda r:coll[r["platform"]]),
        ("paired.csv",check_comparison,lambda r:coll[r["platform"]]),
        ("recording_metrics.csv",check_metrics,lambda r:rec[r["platform"],r["recording"]]),
        ("recording_pairs.csv",check_comparison,lambda r:rec[r["platform"],r["recording"]]),
        ("strata.csv",check_metrics,lambda r:[x for x in coll[r["platform"]] if x[r["group"]]==r["value"]]),
        ("eligible_pairs.csv",check_comparison,lambda r:[x for x in coll[r["platform"]]
             if (x["full"]==x["single"]=="True" if r["group"]=="H_L7_joint_support" else int(x["selection_rays"])>=5)])):
        for r in table(name):
            check(key(r),r)
            rcounts[name]+=1
    receipt=dict(descriptions=len(rows),recordings=len(rec),recounts=dict(rcounts),
                 original_response_reuse=386,all_public_file_hashes_verified=True,FINAL_access=False)
    (OUT/"verification.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(receipt,flush=True)


if __name__=="__main__":
    run()
