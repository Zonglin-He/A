"""Join sealed choices to previously scored IoUs and evaluate matched outcomes."""
from collections import defaultdict
from time import perf_counter
import csv
import gzip

import numpy as np

from rescore import ARMS, AUDIT, HERE, OUT, TIE, WORK, roster
import score_full


PAIRS = (("interval","first"),("fit_offset","first"),("fit_offset","interval"))
OTHER_MOTORS = frozenset(("bus","truck","othervehicle"))


def archived_iou():
    result = {}
    receipt = roster.read(AUDIT/"results/measurement_receipt.json")
    assert receipt["ray_seal"] == roster.sha(AUDIT/"results/ray_seal.json")
    assert receipt["descriptions"] == 9310 and not receipt["FINAL_access"]
    for target in score_full.PLATFORMS:
        name = f"rows_{target}.csv.gz"
        path = AUDIT/"results"/name
        assert roster.sha(path) == receipt["files"][name]
        with gzip.open(path,"rt",newline="") as f:
            for row in csv.DictReader(f):
                assert row["uid"] not in result and row["target"]==target
                result[row["uid"]] = row
    assert len(result)==9310
    return result


def classify(p,q,selected):
    if p<=.5<q:
        return "original_rescue" if selected else "missed_qwen"
    if q<=.5<p:
        return "original_harm" if selected else "original_correct_pca"
    return "both_pass" if p>.5 else "neither_pass"


def join(scores,source):
    assert scores["uid"] == source["uid"]
    production = scores["production"]
    assert production==(source["production"]=="True")
    base = float(source["iou_current"])
    row = {key:source[key] for key in ("uid","target","recording","frame","category")}
    row.update(production=production, pca_iou=None, qwen_iou=None, original_case="fallback")
    diagnostics = ("selection_rays","selection_miss_p","selection_miss_q","selection_miss_both","fit_rays",
                   "fit_common_hits","selection_thickness_clips")
    row.update({k:scores.get(k) for k in diagnostics})
    for j,name in enumerate(("pca","qwen")):
        bounded,raw,thickness = scores["offsets"][j] if production else (None,None,None)
        row.update({name+"_offset":bounded,name+"_raw_offset":raw,name+"_fit_thickness":thickness})
    if production:
        p,q = map(float,(source["iou_pca"],source["iou_qwen"]))
        selected = source["choose_qwen"]=="True"
        assert scores["choice"]["first"] == selected
        assert abs(base-(q if selected else p)) < 1.e-12
        assert (base>.5) == ((q if selected else p)>.5)
        row.update(pca_iou=p,qwen_iou=q,original_case=classify(p,q,selected))
    for arm in ARMS:
        chose = scores["choice"][arm] if production else None
        tie = scores["tie"][arm] if production else None
        costs = scores["cost"][arm] if production else (None,None)
        if production:
            margin = costs[0]-costs[1]
            assert chose == (costs[1]<costs[0]-TIE)
            assert tie == (abs(margin)<=TIE)
            if arm=="first":
                assert abs(margin-float(source["ray_margin"]))<1.e-12
        else:
            margin = None
        zero_tie = tie and max(costs)<=TIE if production else None
        # Use the exact published main-method IoU whenever the chosen box is the same.
        value = base if not production or chose==selected else (q if chose else p)
        row.update({f"iou_{arm}":value, f"{arm}_qwen":chose, f"{arm}_tie":tie,
                    f"{arm}_zero_tie":zero_tie, f"{arm}_margin":margin})
    return row


def groups(rows):
    result = defaultdict(list)
    for r in rows:
        t,c,rec = (r[key] for key in ("target","category","recording"))
        for key in (("macro","all"),("pooled","all"),("platform",t),("category",c),("recording",t+"/"+rec)):
            result[key].append(r)
        if c in OTHER_MOTORS:
            result["target_type","other_motors"].append(r)
        if r["production"]:
            result["routing","active"].append(r)
    return result


def transitions(group,arm):
    cases = defaultdict(list)
    for row in group:
        if row["production"]:
            cases[row["original_case"]].append(row[f"iou_{arm}"]>.5)
    return dict(
        original_rescue=len(cases["original_rescue"]), rescue_kept=sum(cases["original_rescue"]),
        original_harm=len(cases["original_harm"]), harm_fixed=sum(cases["original_harm"]),
        old_missed=len(cases["missed_qwen"]), missed_recovered=sum(cases["missed_qwen"]),
        original_pca_right=len(cases["original_correct_pca"]),
        new_harms=len(cases["original_correct_pca"])-sum(cases["original_correct_pca"]),
    )


def fit_summary(group):
    support = [r for r in group if r["production"]]
    ready = [r for r in support if r["pca_raw_offset"] is not None]
    info = dict(active=len(support),selection_rays=sum(r["selection_rays"] for r in support),
                selection_miss_p=sum(r["selection_miss_p"] for r in support),
                selection_miss_q=sum(r["selection_miss_q"] for r in support),
                selection_miss_both=sum(r["selection_miss_both"] for r in support),
                fit_rays=sum(r["fit_rays"] for r in support),
                fit_common_hits=sum(r["fit_common_hits"] for r in support),
                fit_qualified=len(ready),fit_unqualified=len(support)-len(ready),
                selection_thickness_clips=sum(r["selection_thickness_clips"] for r in support))
    for name in ("pca","qwen"):
        values = [r[name+"_offset"] for r in ready]
        info[name+"_median_offset_m"] = float(np.median(values)) if values else None
        info[name+"_zero_offset"]=sum(v==0. for v in values)
        info[name+"_clipped_at_fit_thickness"]=sum(r[name+"_raw_offset"]>r[name+"_fit_thickness"] for r in ready)
    return info


def run():
    start = perf_counter()
    sealed = roster.read(OUT/"score_seal.json")
    assert sealed["trace_seal"] == roster.sha(AUDIT/"results/ray_seal.json")
    assert sealed["protocol"] == roster.sha(HERE/"PROTOCOL.json") and sealed["code"] == roster.sha(HERE/"rescore.py")
    assert sealed["original_picks_and_selection_returns_verified"] and sealed["fit_selection_cells_disjoint"]
    assert not sealed["target_annotations_opened"] and not sealed["FINAL_access"]
    path = WORK/"sealed_scores.json"
    assert sealed["payload"] == roster.sha(path)
    gold = archived_iou()
    scores = roster.read(path)
    rows = [join(r,gold[r["uid"]]) for r in scores]
    assert len(rows) == len({r["uid"] for r in rows}) == sealed["descriptions"] == 9310
    assert sum(r["production"] for r in rows) == 5910
    score_full.ARMS,score_full.PAIRS,score_full.OUT = ARMS,PAIRS,OUT
    metrics,paired,accounting,choices,fit = [],[],[],[],[]
    for (level,key),group in sorted(groups(rows).items()):
        tag = dict(group=level,value=key)
        agg = "macro" if level=="macro" else "pooled"
        metrics.extend(dict(**tag,**r) for r in score_full.table_row_metrics(group,agg))
        pairs = list(score_full.table_row_pairs(group,agg))
        paired.extend(dict(**tag,**r) for r in pairs)
        fit.append(dict(**tag,n=len(group),**fit_summary(group)))
        for arm in ARMS:
            read = transitions(group,arm)
            accounting.append(dict(**tag,arm=arm,**read))
            if arm=="first":
                assert read["rescue_kept"]==read["original_rescue"]
                assert read["harm_fixed"]==read["missed_recovered"]==read["new_harms"]==0
            else:
                compared=next(p for p in pairs if p["arm"]==arm and p["reference"]=="first")
                assert compared["rescue50"]==read["harm_fixed"]+read["missed_recovered"]
                assert compared["harm50"]==read["original_rescue"]-read["rescue_kept"]+read["new_harms"]
            active = [r for r in group if r["production"]]
            choices.append(dict(
                **tag,arm=arm,n=len(group),active=len(active),
                qwen=sum(r[arm+"_qwen"] for r in active),
                ties=sum(r[arm+"_tie"] for r in active),
                zero_ties=sum(r[arm+"_zero_tie"] for r in active),
                changed_from_first=sum(r[arm+"_qwen"]!=r["first_qwen"] for r in active),
            ))
    baseline=transitions(rows,"first")
    assert (baseline["original_rescue"],baseline["original_harm"],baseline["old_missed"])==(623,64,284)
    files = {name:score_full.write_rows(name,table) for name,table in (
        ("metrics.csv",metrics),("paired.csv",paired),("prior_case_accounting.csv",accounting),
        ("choice_counts.csv",choices),("fit_coverage.csv",fit),
    )}
    for target in score_full.PLATFORMS:
        name = f"rows_{target}.csv.gz"
        files[name] = score_full.write_rows(name,[r for r in rows if r["target"]==target],compress=True)
    roster.write(OUT/"evaluation_receipt.json",dict(
        score_seal=roster.sha(OUT/"score_seal.json"), code=roster.sha(__file__),
        candidate_iou_receipt=roster.sha(AUDIT/"results/measurement_receipt.json"),
        files=files, descriptions=len(rows), recordings=len({r["recording"] for r in rows}),
        active=5910, intervals_pp=score_full.bootstraps(rows),
        original_partition=baseline, additional_GT_file_reads=0, FINAL_access=False,
        seconds=perf_counter()-start))
    print("Evaluated first, interval and Fit offset on all 9310 descriptions",flush=True)


if __name__ == "__main__":
    run()
