"""Post-seal paired evaluation on the original Ordinary roster."""
from collections import defaultdict
import csv
from pathlib import Path
import sys
import numpy as np

from side import ARMS, BASE, ETA, HERE, OUT, PRIVATE, SPEC, TOL, digest, inputs, read, write
PUBLIC_LAYOUT = not (HERE.parent/"3eed_v1_component_ablation_20261119"/"score_ablation.py").is_file()
if PUBLIC_LAYOUT:
    # Identical archived scorers occupy different directories in the public export.
    sys.path[:0] = [str(HERE.parents[1]/"v1/ablation/own_code"),
                    str(HERE.parent/"3eed_waymo_source100_20261030/research/round6/source_validation_scoring")]
from score import canonical_box, score_boxes, truth_for_slot

COMPARISONS = (("B","A"), ("C","D"), *((a,"H") for a in ARMS))
EPS = 1e-8


def table(name, rows):
    path = OUT/(name+".csv")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return digest(path)


def mean_platform(rows, values):
    platforms = np.array([r["target"] for r in rows])
    v = np.asarray(values)
    return float(np.mean([v[platforms==p].mean() for p in np.unique(platforms)]))


def effect(rows, arm, ref):
    a,b = (np.array([r["iou_"+k] for r in rows]) for k in (arm, ref))
    return dict(rescue50=int(sum((a>.5)&(b<=.5))), harm50=int(sum((a<=.5)&(b>.5))),
                delta50_pp=100*mean_platform(rows, (a>.5).astype(int)-(b>.5)),
                rescue25=int(sum((a>.25)&(b<=.25))), harm25=int(sum((a<=.25)&(b>.25))),
                delta25_pp=100*mean_platform(rows, (a>.25).astype(int)-(b>.25)),
                improved_iou=int(sum(a-b>EPS)), worsened_iou=int(sum(b-a>EPS)),
                tied_iou=int(sum(abs(a-b)<=EPS)), delta_miou_pp=100*mean_platform(rows,a-b),
                different_center=sum(abs(r["shift_"+arm]-r["shift_"+ref])>TOL for r in rows))


def uncertainty(rows, arm, ref):
    # Resample whole recordings within each platform, retaining all frames and their pairing.
    rng = np.random.default_rng(20261121)
    by_platform = defaultdict(lambda:defaultdict(list))
    for row in rows:
        a,b = row["iou_"+arm], row["iou_"+ref]
        by_platform[row["target"]][row["recording"]].append([int(a>.5)-int(b>.5), a-b])
    draws = []
    for platform in sorted(by_platform):
        blocks = by_platform[platform]
        ordered = [blocks[k] for k in sorted(blocks)]
        counts = np.array([len(block) for block in ordered])
        vals = np.array([np.sum(block,axis=0) for block in ordered])
        idx = rng.integers(len(counts), size=(4000,len(counts)))
        draws.append(vals[idx].sum(1)/counts[idx].sum(1)[:,None])
    q = np.quantile(100*np.mean(draws, axis=0), [.025,.975], axis=0)
    return dict(acc50_pp=q[:,0].tolist(), miou_pp=q[:,1].tolist())


def scopes(rows):
    criteria = {"all":lambda r:True, "validation":lambda r:r["panel"]=="validation",
                "extension":lambda r:r["panel"]!="validation",
                "recording_separated":lambda r:r["separation"]=="new_recording",
                "C_D_distinct_prompt":lambda r:r["C_D_same"] is False}
    for scope, include in criteria.items():
        subset = [r for r in rows if include(r)]
        if subset:
            yield scope, "macro", subset
        for t in sorted({r["target"] for r in subset}):
            yield scope, t, [r for r in subset if r["target"] == t]


def run():
    s = read(OUT/"prediction_seal.json")
    prep = read(OUT/"input_seal.json")
    assert not s["target_labels_read"] and not s["FINAL_access"]
    assert s["predictions"] == digest(PRIVATE/"predictions.json")
    assert s["pins"]["input_seal"] == digest(OUT/"input_seal.json")
    assert s["pins"]["inference_code"] == digest(HERE/"run_qwen.py")
    assert prep["prepared"] == digest(PRIVATE/"prepared.json") and prep["protocol"] == digest(SPEC)
    outputs, prepared = read(PRIVATE/"predictions.json"), read(PRIVATE/"prepared.json")
    panel, frozen = inputs()
    if PUBLIC_LAYOUT:
        import score_ablation
        mounts = {Path(f["calibration"]).parents[3] for f in panel}
        assert len(mounts)==1
        score_ablation.DATA = mounts.pop()
    with (BASE/"results/rows.csv").open() as f:
        old = {r["uid"]:float(r["H"]) for r in csv.DictReader(f)}
    rows, max_gap, parity = [], 0., 0.
    for f in panel:
        uid = f["uid"]
        item, pred = prepared[uid], outputs[uid]
        h = frozen[uid]["boxes"]["H"]
        assert h == pred["H"] == item["H"]
        g = item.get("geometry")
        row = {k:f[k] for k in ("uid","target","panel","recording","separation")}
        row.update(selected=item["selected"], status=item["status"], promptable="prompts" in item,
                   axis_slot=g["k"] if g else None, slack_m=g["slack_m"] if g else None,
                   C_D_same=(item["prompts"]["C"]==item["prompts"]["D"]) if "prompts" in item else None,
                   shift_H=0.)
        boxmap = {"H":h, **pred["boxes"], **{k:pred["discrete_boxes"].get(k,h) for k in ETA}}
        for arm in ARMS:
            row["side_"+arm] = pred["sides"].get(arm, "unavailable")
            row["shift_"+arm] = pred["offsets_m"][arm]
        if h is None:
            assert all(b is None for b in boxmap.values())
            values = np.zeros(len(boxmap))
        else:
            truth = canonical_box(truth_for_slot(f, read(f["calibration"])["ground_info"]))
            values, gap = score_boxes(np.array(list(boxmap.values())), truth)
            max_gap = max(max_gap, gap)
            for name, box in boxmap.items():
                row["center_err_m_"+name] = float(np.linalg.norm(np.array(box[:3])-truth[:3]))
        row.update({"iou_"+k:float(v) for k,v in zip(boxmap, values, strict=True)})
        row["iou_oracle"] = max(row["iou_"+k] for k in ("H",*ETA))
        row["oracle_recoverable50"] = row["iou_H"] <= .5 < row["iou_oracle"]
        row["side_discriminates_iou"] = (max(row["iou_"+k] for k in ETA)-min(row["iou_"+k] for k in ETA)) > EPS
        parity = max(parity, abs(old[uid]-row["iou_H"]))
        rows.append(row)
    assert len(rows)==len(old)==416 and max_gap<EPS and parity<EPS
    metrics, paired, decisions, recording, ci = [],[],[],[],{}
    for scope, target, group in scopes(rows):
        label = dict(scope=scope,target=target,n=len(group),recordings=len({r["recording"] for r in group}))
        for arm in ("H",*ARMS,*ETA,"oracle"):
            a = np.array([r["iou_"+arm] for r in group])
            e = [r["center_err_m_"+arm] for r in group if "center_err_m_"+arm in r]
            metrics.append(dict(**label,arm=arm,hits50=int(sum(a>.5)),acc50=100*mean_platform(group,a>.5),
                       acc25=100*mean_platform(group,a>.25),miou=100*mean_platform(group,a),
                       center_err_n=len(e), center_error_median_m=float(np.median(e)) if e else None))
        for arm,ref in COMPARISONS:
            paired.append(dict(**label,arm=arm,reference=ref,**effect(group,arm,ref)))
            if target=="macro":
                ci[f"{scope}/{arm}-{ref}"] = uncertainty(group,arm,ref)
        for arm in ARMS:
            sides = [r["side_"+arm] for r in group]
            discriminating = [r for r in group if r["side_discriminates_iou"]]
            decisions.append(dict(**label,arm=arm,**{k:sides.count(k) for k in (*ETA,"unknown","invalid","unavailable")},
                    nonzero_slack=sum((r["slack_m"] or 0.)>TOL for r in group),
                    moved_vs_H=sum(abs(r["shift_"+arm])>TOL for r in group),
                    recoverable50=sum(r["oracle_recoverable50"] for r in group),
                    recovered50=sum(r["oracle_recoverable50"] and r["iou_"+arm]>.5 for r in group),
                    oracle_iou_distinguishable=len(discriminating),
                    explicit_best=sum(r["side_"+arm] in ETA and r["iou_"+arm]>=r["iou_oracle"]-EPS for r in discriminating)))
    for t,rec in sorted({(r["target"],r["recording"]) for r in rows}):
        g = [r for r in rows if r["target"]==t and r["recording"]==rec]
        for arm,ref in COMPARISONS:
            recording.append(dict(target=t,recording=rec,n=len(g),arm=arm,reference=ref,**effect(g,arm,ref)))
    files = {name:table(name,data) for name,data in (("rows",rows),("metrics",metrics),("paired",paired),
            ("decisions",decisions),("recordings",recording))}
    write(OUT/"evaluation_receipt.json", dict(prediction_seal=digest(OUT/"prediction_seal.json"),
           score_code=digest(__file__), files=files, intervals=ci, maximum_scorer_gap=float(max_gap),
           maximum_frozen_replay_gap=float(parity), n=len(rows), FINAL_access=False))
    print("Scored",len(rows),"frames; oriented IoU crosscheck",max_gap,"parent parity", parity,flush=True)


if __name__ == "__main__":
    run()
