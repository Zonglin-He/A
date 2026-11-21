"""Score sealed predictions and matched controls on Ordinary Validation."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from propose import ARMS, FAMILIES, HERE, OUT, PAYLOAD, PROTOCOL, SURFACE, TIE, origin, roster

sys.path.insert(0, str(SURFACE))
from evaluate import archived_iou, classify
import score_full
from score_ablation import DATA


PAIRS = (("selection_only","baseline"), ("branch","baseline"), ("tail","baseline"),
         ("axis","baseline"), ("branch","selection_only"), ("branch","tail"), ("branch","axis"))


def score(job):
    frame, sealed, archived = job
    assert frame["uid"] == sealed["uid"] == archived["uid"] and frame["panel"]=="validation"
    base = float(archived["iou_current"])
    translated = {}
    result = {k:archived[k] for k in ("uid","target","category","recording","frame")}
    result.update(active=sealed["active"], audit_cells=sealed.get("audit_cells"), origin_case="fallback",
                  fit_seeds=sealed.get("fit_seeds"), scale=sealed.get("scale"), gt_scored=False, matched_budget=False)
    choices = sealed.get("choices",{})
    options = sealed.get("options",{})
    if sealed["active"]:
        assert sealed["use_qwen"] == (archived["choose_qwen"]=="True")
        p,q = map(float,(archived["iou_pca"],archived["iou_qwen"]))
        result["origin_case"] = classify(p,q,sealed["use_qwen"])
    if options.get("branch"):
        path = Path(frame["calibration"])
        assert path.name == "meta_info.json" and path.parents[2] == DATA/frame["target"]
        items = roster.read(path)["ground_info"]
        truth = score_full.canonical_box(score_full.truth_for_slot(frame,items))
        assert items[frame["slot"]]["class"].strip().lower() == result["category"]
        ready = [sealed["box"]]+[v["box"] for name in FAMILIES for v in options[name]]
        scores, _ = score_full.score_boxes(np.asarray(ready),truth)
        assert abs(scores[0]-base)<1.e-12 and (scores[0]>.5)==(base>.5)
        begin = 1
        for family in FAMILIES:
            size = len(options[family])
            translated[family] = scores[begin:begin+size].tolist()
            begin += size
        result["gt_scored"] = True
        budgets = [len(options[family]) for family in FAMILIES]
        result["matched_budget"] = len(set(budgets))==1
    result["iou_baseline"] = base
    for family in FAMILIES:
        proposed = options.get(family,[])
        result[f"{family}_proposals"] = len(proposed)
        result[f"{family}_oracle"] = max([base]+translated.get(family,[]))
    for arm in ARMS[1:]:
        option = choices.get(arm)
        family = "branch" if arm=="selection_only" else arm
        metadata = options[family][option] if option is not None else None
        result[f"iou_{arm}"] = translated[family][option] if metadata is not None else base
        result[f"changed_{arm}"] = metadata is not None
        result[f"{arm}_selection_delta"] = metadata["selection_delta"] if metadata is not None else None
        result[f"{arm}_audit_delta"] = metadata["audit_delta"] if metadata is not None else None
        result[f"{arm}_shift"] = metadata["shift"] if metadata is not None else None
        result[f"{arm}_removed"] = metadata["removed"] if metadata is not None else None
        if metadata:
            assert metadata["selection_delta"] < -TIE
            if arm!="selection_only":
                assert metadata["audit_delta"] < -TIE
    if choices.get("branch") is not None:
        assert choices["branch"]==choices["selection_only"]
    result["branch_joint_axes"] = bool(choices.get("branch") is not None and options["branch"][choices["branch"]]["both_axes"])
    return result


def groups(rows):
    blocks = defaultdict(list)
    for r in rows:
        target = r["target"]
        keys = [("macro","all"),("pooled","all"),("platform",target),("category",r["category"]),
                ("recording",target+"/"+r["recording"]),("case",r["origin_case"])]
        if r["active"]:
            keys.append(("coverage","applicable"))
        if r["gt_scored"]:
            keys.append(("coverage","branch_proposed"))
        if r["matched_budget"]:
            keys.append(("coverage","matched_budget"))
            keys.append(("matched_platform",target))
        for key in keys:
            blocks[key].append(r)
    return blocks


def coverage(block):
    output = dict(n=len(block), active=sum(r["active"] for r in block), gt_scored=sum(r["gt_scored"] for r in block),
                  matched_budget=sum(r["matched_budget"] for r in block), joint_axes=sum(r["branch_joint_axes"] for r in block))
    for arm in ARMS[1:]:
        output[f"{arm}_changed"] = sum(r[f"changed_{arm}"] for r in block)
    for family in FAMILIES:
        output[f"{family}_available"] = sum(r[f"{family}_proposals"]>0 for r in block)
        output[f"{family}_examples"] = sum(r[f"{family}_proposals"] for r in block)
        output[f"{family}_oracle_rescues50"] = sum(r["iou_baseline"] <= .5 < r[f"{family}_oracle"] for r in block)
    return output


def run(workers):
    start = perf_counter()
    seal_file = OUT/"prediction_seal.json"
    seal = roster.read(seal_file)
    assert not seal["target_gt_opened"] and not seal["FINAL_access"] and seal["untouched_audit_split"]
    assert seal["locked_method"]==roster.sha(origin.ACTIVE/"METHOD_LOCK.json")
    assert seal["protocol"]==roster.sha(PROTOCOL) and seal["code"]==roster.sha(HERE/"propose.py")
    assert seal["payload"]==roster.sha(PAYLOAD)
    frames,_,_,_,_ = origin.inputs()
    previous = archived_iou()
    rows = []
    with gzip.open(PAYLOAD,"rt") as trace, ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = ((f,json.loads(line),previous[f["uid"]]) for f,line in zip(frames,trace,strict=True))
        for result in pool.map(score,jobs,chunksize=12):
            rows.append(result)
            if len(rows)%1000==0:
                print("Scored",len(rows),"/ 9310",flush=True)
    assert len(rows)==len({r["uid"] for r in rows})==seal["descriptions"]==9310
    assert sum(r["gt_scored"] for r in rows)==seal["connected_deletion_proposed"]
    assert sum(r["active"] for r in rows)==5910
    score_full.ARMS,score_full.PAIRS,score_full.OUT = ARMS,PAIRS,OUT
    metrics, paired, measurement = [],[],[]
    for (group,value), block in sorted(groups(rows).items()):
        tags = dict(group=group,value=value)
        domain = "macro" if group=="macro" else "pooled"
        metrics.extend(dict(**tags,**r) for r in score_full.table_row_metrics(block,domain))
        paired.extend(dict(**tags,**r) for r in score_full.table_row_pairs(block,domain))
        measurement.append(dict(**tags,**coverage(block)))
    files = {name:score_full.write_rows(name,table) for name,table in (
        ("metrics.csv",metrics),("paired.csv",paired),("coverage.csv",measurement)
    )}
    for target in score_full.PLATFORMS:
        name = f"rows_{target}.csv.gz"
        files[name] = score_full.write_rows(name,[r for r in rows if r["target"]==target],compress=True)
    roster.write(OUT/"evaluation_receipt.json",dict(
        files=files, descriptions=len(rows), active=5910, ordinary_GT_descriptions=sum(r["gt_scored"] for r in rows),
        prediction_seal=roster.sha(seal_file), code=roster.sha(__file__), intervals_pp=score_full.bootstraps(rows),
        new_Qwen_queries=0, production_prediction_changes=0, FINAL_access=False, seconds=perf_counter()-start,
    ))
    print(coverage(rows),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=8)
    run(parser.parse_args().workers)
