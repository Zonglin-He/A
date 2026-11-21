"""Summarize paired geometry, ranking diagnostics, and post-seal GT controls."""
from collections import Counter, defaultdict
from math import isfinite
from pathlib import Path
import csv
import gzip
import hashlib
import json

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OUT = HERE/"results"
PLATFORMS = ("waymo","drone","quad")
IDENTIFIERS = {"uid","target","recording","frame","category"}
FLAGS = {"available","production","choose_qwen"}
METRICS = ("ray_margin","observability_bound","predicted_rms_m","predicted_rms_scaled","info","signed_evidence")
PLAN = json.loads((HERE/"PROTOCOL.json").read_bytes())
EPS = PLAN["metric_thresholds"]["IoU_tie_absolute"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows():
    receipt = json.loads((OUT/"measurement_receipt.json").read_bytes())
    assert receipt["ray_seal"] == sha(OUT/"ray_seal.json") and not receipt["FINAL_access"]
    assert receipt["code"] == sha(HERE/"score_pairs.py")
    rows = []
    for t,n in zip(PLATFORMS,(3782,2792,2736),strict=True):
        file = OUT/f"rows_{t}.csv.gz"
        assert receipt["files"][file.name] == sha(file)
        with gzip.open(file, "rt", newline="") as f:
            block = list(csv.DictReader(f))
        assert len(block) == n
        for row in block:
            parsed = {k:(v if k in IDENTIFIERS else v=="True" if k in FLAGS else float(v) if v else None)
                      for k,v in row.items()}
            if parsed["available"]:
                assert abs(parsed["ray_margin"]) <= parsed["observability_bound"]+1.e-12
                parsed["gap"] = parsed["iou_qwen"]-parsed["iou_pca"]
                parsed["one_hit_fraction"] = (parsed["only_pca_hits"] + parsed["only_qwen_hits"])/parsed["rays"]
                parsed["inside_target_fraction"] = parsed["inside_target_rays"]/parsed["rays"]
            rows.append(parsed)
    assert len(rows) == len({r["uid"] for r in rows}) == 9310
    assert len({r["frame"] for r in rows}) == 7221 and len({r["recording"] for r in rows}) == 98
    return rows


def blocks(rows):
    pairs = [r for r in rows if r["available"]]
    active = [r for r in pairs if r["production"]]
    out = {"all_pairs":pairs, "production":active,
           "historical_only":[r for r in pairs if not r["production"]]}
    for t in PLATFORMS:
        out["production/"+t] = [r for r in active if r["target"]==t]
    for c in sorted({r["category"] for r in pairs}):
        out["class/"+c] = [r for r in pairs if r["category"]==c]
    return out


def outcome(r, kind):
    if kind=="iou":
        return 1 if r["gap"]>EPS else -1 if r["gap"]<-EPS else 0
    return int(r["iou_qwen"]>.5) - int(r["iou_pca"]>.5)


def metric(value):
    return float(value) if isfinite(value) else None


def coverage(group, name):
    switched = [r for r in group if r["choose_qwen"]]
    base = dict(scope=name,n=len(group),choices_qwen=len(switched),recordings=len({r["recording"] for r in group}),
                n_common_hit=sum(r["common_hits"]>0 for r in group),
                n_shared_hit_ge5=sum(r["common_hits"]>=5 for r in group),
                original_h_at50=sum(r["iou_h"]>.5 for r in group), pca_at50=sum(r["iou_pca"]>.5 for r in group),
                ungated_ray_at50=sum(r["iou_ungated"]>.5 for r in group),
                either_at50=sum(max(r["iou_pca"],r["iou_qwen"])>.5 for r in group),
                h_at50_neither_candidate=sum(r["iou_h"]>.5>=max(r["iou_pca"],r["iou_qwen"]) for r in group),
                near_equal_rays=sum(r["observability_bound"]<=1.e-9 for r in group),
                near_equal_rays_diff_iou=sum(r["observability_bound"]<=1.e-9 and abs(r["gap"])>EPS for r in group))
    for k in ("iou","acc50"):
        for label,v in (("rescue",1),("harm",-1),("tie",0)):
            base[f"{k}_{label}_when_switched"] = sum(outcome(r,k)==v for r in switched)
        base[f"{k}_better_qwen_left_on_pca"] = sum(not r["choose_qwen"] and outcome(r,k)==1 for r in group)
    return base


def correlations(group, name):
    for k in ("predicted_rms_m","predicted_rms_scaled","info","observability_bound","one_hit_fraction"):
        valid = [r for r in group if r[k] is not None]
        for view in ("all_valid", "distinct_IoU"):
            sample = valid if view=="all_valid" else [r for r in valid if abs(r["gap"])>EPS]
            xs = [r[k] for r in sample]
            ys = [abs(r["gap"]) for r in sample]
            rho = spearmanr(xs,ys).statistic if len(xs)>2 and len(set(xs))>1 and len(set(ys))>1 else float("nan")
            yield dict(scope=name,selection=view,feature=k,n=len(sample),spearman=metric(rho))


def auc(values, labels, weights):
    order = np.argsort(values,kind="stable")
    pos = (labels[order] == 1)
    ties = np.r_[0,np.flatnonzero(np.diff(values[order]))+1]
    positive = np.add.reduceat(weights[:,order]*pos,ties,axis=1)
    negative = np.add.reduceat(weights[:,order]*(~pos),ties,axis=1)
    pair_count = positive.sum(1)*negative.sum(1)
    wins = (positive*(np.cumsum(negative,axis=1)-.5*negative)).sum(1)
    return np.divide(wins,pair_count,out=np.full(len(weights),np.nan),where=pair_count>0)


def clustered_weights(sample, bootstrap):
    if not bootstrap:
        return np.ones((1,len(sample)), dtype=np.int16)
    k = PLAN["recording_bootstrap"]["draws"]
    rng = np.random.default_rng(PLAN["recording_bootstrap"]["seed"])
    weights = np.ones((k+1,len(sample)),dtype=np.int16)
    for t in PLATFORMS:
        ids = np.array([i for i,r in enumerate(sample) if r["target"]==t],dtype=int)
        if not len(ids):
            continue
        _, choices = np.unique([sample[i]["recording"] for i in ids],return_inverse=True)
        n = 1+int(choices.max())
        counts = rng.multinomial(n,np.full(n,1/n),size=k)
        weights[1:,ids] = counts[:,choices]
    return weights


def ranks(group, name):
    for subset in ("all_switches","common_hit_switches"):
        subset_rows = [r for r in group if r["choose_qwen"] and (subset=="all_switches" or r["common_hits"]>0)]
        keys = METRICS[:2] if subset=="all_switches" else METRICS
        for kind in ("iou","acc50"):
            sample = [r for r in subset_rows if outcome(r,kind)]
            labels = np.array([outcome(r,kind) for r in sample])
            positive,negative = int(sum(labels==1)),int(sum(labels==-1))
            if not positive or not negative:
                continue
            draws = clustered_weights(sample,bootstrap=name=="production")
            baseline = auc(np.array([r["ray_margin"] for r in sample]),labels,draws)
            for key in keys:
                score = np.array([r[key] for r in sample])
                result = auc(score,labels,draws)
                diff = result - baseline
                finite = result[1:][np.isfinite(result[1:])]
                delta = diff[1:][np.isfinite(diff[1:])]
                lower,upper = np.quantile(finite,[.025,.975]) if len(finite) else (None,None)
                ldiff,udiff = np.quantile(delta,[.025,.975]) if len(delta) else (None,None)
                yield dict(scope=name,subset=subset,outcome=kind,feature=key,positive=positive,negative=negative,
                           auc=metric(result[0]),lo=lower,hi=upper,delta_vs_margin=metric(diff[0]),
                           delta_lo=ldiff,delta_hi=udiff)


def quantiles(group,name):
    chosen = [r for r in group if r["choose_qwen"]]
    if not chosen:
        return
    x = [r["observability_bound"] for r in chosen]
    cuts = np.quantile(x,[.2,.4,.6,.8])
    for k in range(5):
        sample = [r for r,i in zip(chosen,np.searchsorted(cuts,x,side="left"),strict=True) if i==k]
        if not sample:
            continue
        a = sum(outcome(r,"iou")==1 for r in sample)
        b = sum(outcome(r,"iou")==-1 for r in sample)
        yield dict(scope=name,quintile=k+1,n=len(sample),min_bound=min(r["observability_bound"] for r in sample),
                   max_bound=max(r["observability_bound"] for r in sample),iou_better=a,iou_harm=b,
                   iou_tie=len(sample)-a-b,iou_harm_fraction=b/(a+b) if a+b else None,
                   acc50_rescue=sum(outcome(r,"acc50")==1 for r in sample),
                   acc50_harm=sum(outcome(r,"acc50")==-1 for r in sample),
                   median_predicted_rms_m=median(sample,"predicted_rms_m"),
                   median_abs_iou_gap=float(np.median([abs(r["gap"]) for r in sample])),
                   median_one_hit_fraction=median(sample,"one_hit_fraction"))


def median(rows,key):
    values = [r[key] for r in rows if r[key] is not None]
    return float(np.median(values)) if values else None


def control(group,name):
    for kind in ("iou","acc50"):
        for label in ("rescue","harm"):
            sign = 1 if label=="rescue" else -1
            chosen = [r for r in group if r["choose_qwen"] and outcome(r,kind)==sign]
            subset = [r for r in chosen if r["gt_member_margin"] is not None]
            yield dict(scope=name,outcome=kind,kind=label,n=len(chosen),with_5_inside=len(subset),
                       median_ray_margin=median(chosen,"ray_margin"),
                       median_bound=median(chosen,"observability_bound"),
                       median_predicted_rms_m=median(chosen,"predicted_rms_m"),
                       median_inside_fraction=median(chosen,"inside_target_fraction"),
                       median_both_saturated=median(chosen,"both_saturated_fraction"),
                       median_one_hit_fraction=median(chosen,"one_hit_fraction"),
                       median_gt_first_surface_error_m=median(subset,"gt_first_surface_abs_error_m"),
                       keep_qwen_on_inside=sum(r["gt_member_margin"]>1.e-12 for r in subset),
                       keep_qwen_on_gt_surface=sum(r["gt_surface_margin"]>1.e-12 for r in subset),
                       actual_and_gt_surface_differ=sum((r["gt_member_margin"]>1.e-12)!=(r["gt_surface_margin"]>1.e-12) for r in subset),
                       one_hit_contribution_exceeds_common=sum(r["one_hit_margin"]>abs(r["common_margin"]) for r in chosen))


def write(name, rows):
    file = OUT/name
    with file.open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha(file)


def run():
    rows = read_rows()
    groups = blocks(rows)
    assert len(groups["production"]) == 5910 and len(groups["all_pairs"]) == 7948
    per_class = []
    for c in sorted({r["category"] for r in rows}):
        relevant = [r for r in rows if r["category"]==c]
        per_class.append(dict(category=c,descriptions=len(relevant),pairs=sum(r["available"] for r in relevant),
                              current_uses_rays=sum(r["production"] for r in relevant)))
    outputs = {
        "cohorts.csv": [coverage(rows,group) for group,rows in groups.items()],
        "category_coverage.csv": per_class,
        "correlations.csv": [v for g,r in groups.items() for v in correlations(r,g)],
        "switch_auc.csv": [v for g,r in groups.items() for v in ranks(r,g)],
        "observability_quintiles.csv": [v for g,r in groups.items() for v in quantiles(r,g)],
        "gt_controls.csv": [v for g,r in groups.items() for v in control(r,g)],
    }
    checks = {name:write(name,table) for name,table in outputs.items()}
    receipt = dict(input=sha(OUT/"measurement_receipt.json"), files=checks, code=sha(Path(__file__)),
                   descriptions=len(rows), all_pairs=len(groups["all_pairs"]), production=len(groups["production"]),
                   groups={k:len(v) for k,v in outputs.items()}, gt_controls_diagnostic=True, FINAL_access=False)
    (OUT/"analysis_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(receipt,flush=True)


if __name__ == "__main__":
    run()
