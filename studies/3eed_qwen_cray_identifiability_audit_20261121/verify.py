"""Check hashes, layered denominators and independent rank-based AUCs."""
from collections import Counter
import csv
import json

import numpy as np
from scipy.stats import rankdata

from summarize import EPS, HERE, OUT, blocks, read_rows, sha


def table(filename):
    with (OUT/filename).open(newline="") as f:
        return list(csv.DictReader(f))


def pair_outcome(row, metric):
    if metric == "acc50":
        return int(row["iou_qwen"] > .5) - int(row["iou_pca"] > .5)
    return int(row["gap"] > EPS) - int(row["gap"] < -EPS)


def run():
    seal = json.loads((OUT/"ray_seal.json").read_bytes())
    measured = json.loads((OUT/"measurement_receipt.json").read_bytes())
    analysis = json.loads((OUT/"analysis_receipt.json").read_bytes())
    assert not seal["gt_opened"] and not any(r["FINAL_access"] for r in (seal,measured,analysis))
    assert analysis["input"] == sha(OUT/"measurement_receipt.json")
    assert analysis["code"] == sha(HERE/"summarize.py")
    for filename,checksum in analysis["files"].items():
        assert sha(OUT/filename) == checksum
    rows = read_rows()
    groups = blocks(rows)
    assert len(rows) == seal["descriptions"] == 9310
    assert len(groups["all_pairs"]) == 7948 and len(groups["production"]) == 5910
    assert sum(r["rays"] for r in groups["all_pairs"]) == seal["rays"]
    for r in rows:
        assert r["available"] == (r["rays"]>=5)
        if not r["available"]:
            assert not r["production"] and r["iou_qwen"] is None
        else:
            parts = ("common_hits","only_pca_hits","only_qwen_hits","neither_hits")
            assert r["rays"] == sum(r[k] for k in parts)
            assert (r["common_hits"]>0) == (r["signed_evidence"] is not None)
            assert r["choose_qwen"] == (r["ray_margin"]>1.e-12)
            assert 0 <= r["observability_bound"] <= 1.
            assert abs(r["ray_margin"] - r["common_margin"] - r["one_hit_margin"]) < 1.e-12
            assert r["gt_surface_rays"] <= r["inside_target_rays"] <= r["rays"]
            assert (r["gt_surface_rays"]>=5) == (r["gt_surface_margin"] is not None)
        target_iou = r["iou_ungated"] if r["production"] else r["iou_h"]
        assert abs(r["iou_current"]-target_iou) < 1.e-12
    for report in table("cohorts.csv"):
        group = groups[report["scope"]]
        chosen = [x for x in group if x["choose_qwen"]]
        assert int(report["n"]) == len(group)
        assert int(report["choices_qwen"]) == len(chosen)
        for metric in ("iou","acc50"):
            counts = Counter(pair_outcome(r,metric) for r in chosen)
            for key,sign in (("rescue",1),("harm",-1),("tie",0)):
                assert int(report[f"{metric}_{key}_when_switched"]) == counts[sign]
    auc_rows = table("switch_auc.csv")
    for report in auc_rows:
        group = groups[report["scope"]]
        predicted, binary = [], []
        for r in group:
            tag = pair_outcome(r,report["outcome"])
            if r["choose_qwen"] and tag and (report["subset"]=="all_switches" or r["common_hits"]>0):
                predicted.append(r[report["feature"]])
                binary.append(tag==1)
        pos = sum(binary)
        neg = len(binary)-pos
        assert (pos,neg) == (int(report["positive"]),int(report["negative"]))
        u = rankdata(predicted,method="average")[binary].sum() - pos*(pos+1)/2
        assert abs(u/(pos*neg)-float(report["auc"])) < 1.e-12
    for report in table("gt_controls.csv"):
        wanted = 1 if report["kind"]=="rescue" else -1
        chosen = [r for r in groups[report["scope"]] if r["choose_qwen"] and pair_outcome(r,report["outcome"])==wanted]
        assert len(chosen) == int(report["n"])
        inside = [r for r in chosen if r["gt_surface_rays"]>=5]
        assert len(inside) == int(report["with_5_inside"])
        assert sum(r["gt_member_margin"]>1.e-12 for r in inside) == int(report["keep_qwen_on_inside"])
        assert sum(r["gt_surface_margin"]>1.e-12 for r in inside) == int(report["keep_qwen_on_gt_surface"])
    verified = dict(descriptions=len(rows), pairs=7948, production_pairs=5910, auc_recounted=len(auc_rows),
                    published_hashes_verified=True, missing_and_bad_outcomes_preserved=True, FINAL_access=False)
    (OUT/"verification.json").write_text(json.dumps(verified,indent=2)+"\n")
    print(verified,flush=True)


if __name__ == "__main__":
    run()
