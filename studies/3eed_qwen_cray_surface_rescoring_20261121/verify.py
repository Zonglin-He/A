"""Independently verify result hashes, choices, and Acc@50 case accounting."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import gzip
import hashlib
import json
import sys


HERE = Path(__file__).resolve().parent
OUT = HERE/"results"
ARMS = ("first","interval","fit_offset")
sys.path.insert(0,str(HERE.parent/"3eed_qwen_full_ordinary_validation_20261121"))
from verify_full import check_comparison, check_metrics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(file):
    with (OUT/file).open(newline="") as stream:
        return list(csv.DictReader(stream))


def run():
    seal = json.loads((OUT/"score_seal.json").read_bytes())
    report = json.loads((OUT/"evaluation_receipt.json").read_bytes())
    assert report["score_seal"]==digest(OUT/"score_seal.json")
    assert report["code"]==digest(HERE/"evaluate.py") and seal["code"]==digest(HERE/"rescore.py")
    assert seal["protocol"]==digest(HERE/"PROTOCOL.json")
    assert not seal["target_annotations_opened"] and report["additional_GT_file_reads"]==0
    assert not seal["FINAL_access"] and not report["FINAL_access"]
    for filename,sha in report["files"].items():
        assert digest(OUT/filename)==sha
    rows = []
    for t,n in (("waymo",3782),("drone",2792),("quad",2736)):
        with gzip.open(OUT/f"rows_{t}.csv.gz","rt",newline="") as f:
            block = list(csv.DictReader(f))
        assert len(block)==n and all(r["target"]==t for r in block)
        rows.extend(block)
    assert len(rows)==len({r["uid"] for r in rows})==seal["descriptions"]==9310
    assert len({r["frame"] for r in rows})==7221 and len({r["recording"] for r in rows})==98

    groups = defaultdict(list)
    cases = Counter()
    for r in rows:
        t,c,rec = (r[k] for k in ("target","category","recording"))
        tags = [("pooled","all"),("macro","all"),("platform",t),("category",c),("recording",t+"/"+rec)]
        if c in ("truck","bus","othervehicle"):
            tags.append(("target_type","other_motors"))
        if r["production"]=="True":
            tags.append(("routing","active"))
            cases[r["original_case"]] += 1
            for a in ARMS:
                use_q = r[a+"_qwen"]=="True"
                assert use_q==(float(r[a+"_margin"])>1.e-12)
                assert (r[a+"_tie"]=="True")==(abs(float(r[a+"_margin"]))<=1.e-12)
                chosen = float(r["qwen_iou" if use_q else "pca_iou"])
                assert abs(chosen-float(r["iou_"+a]))<1.e-12
            if int(r["fit_common_hits"])<5:
                assert r["fit_offset_qwen"]==r["first_qwen"]
                assert float(r["pca_offset"])==float(r["qwen_offset"])==0.
            else:
                for kind in ("pca","qwen"):
                    assert 0.<=float(r[kind+"_offset"])<=float(r[kind+"_fit_thickness"])
        else:
            assert r["iou_interval"]==r["iou_fit_offset"]==r["iou_first"]
        for tag in tags:
            groups[tag].append(r)
    assert sum(cases.values())==5910
    assert (cases["original_rescue"],cases["original_harm"],cases["missed_qwen"])==(623,64,284)
    for name,method in (("metrics.csv",check_metrics),("paired.csv",check_comparison)):
        records=table(name)
        assert len(records)==3*len(groups)
        for r in records:
            tag = r["group"],r["value"]
            method(groups[tag],dict(r,platform=tag[0]))
    for r in table("prior_case_accounting.csv"):
        block = groups[r["group"],r["value"]]
        arm = r["arm"]
        for key,label,pass_condition in (
            ("rescue_kept","original_rescue",True),("harm_fixed","original_harm",True),
            ("missed_recovered","missed_qwen",True),("new_harms","original_correct_pca",False),
        ):
            assert int(r[key])==sum(d["original_case"]==label and (float(d["iou_"+arm])>.5)==pass_condition for d in block)
    for r in table("choice_counts.csv"):
        block = groups[r["group"],r["value"]]
        assert int(r["ties"])==sum(d[r["arm"]+"_tie"]=="True" for d in block)
        assert int(r["zero_ties"])==sum(d[r["arm"]+"_zero_tie"]=="True" for d in block)
        assert int(r["zero_ties"])<=int(r["ties"])
        assert int(r["qwen"])==sum(d[r["arm"]+"_qwen"]=="True" for d in block)
    verified = dict(descriptions=len(rows),recordings=98,active=5910,matched_arms=list(ARMS),
                    original_case_counts=dict(cases),full_results_verified=True,
                    additional_GT_file_reads=0,FINAL_access=False)
    (OUT/"verification.json").write_text(json.dumps(verified,indent=2)+"\n")
    print(verified,flush=True)


if __name__ == "__main__":
    run()
