"""Score the D/E panel without choosing configurations from its labels."""
import numpy as np

from inputs import OUT, PLATFORMS, PRIVATE, digest, load_panel, read, write
from followup import ARMS, PLAN
from score import aggregate, canonical_box, delta, recorder_ci, score_boxes, truth_for_slot
from score_region import save, table


def run():
    seal = read(OUT / "followup_prediction_seal.json")
    assert seal["stamp"]["protocol"] == digest(PLAN) and seal["directions"] == 192
    assert seal["stamp"]["C"] == digest(OUT / "region_seal.json")
    predictions = {}
    for source, sha in seal["files"].items():
        path = PRIVATE / "followup" / (source + ".json")
        assert digest(path) == sha and read(path)["stamp"] == seal["stamp"]
        predictions[source] = read(path)["rows"]
    previous = {r["uid"]:r for r in table(OUT / "frames.csv")}
    source_meta = {(r["source"],r["uid"]):r for r in table(OUT / "source_pairs.csv")}
    registry = {(r["source"],r["uid"]):r for r in table(OUT / "v3_text_replay_rows.csv")}
    output = []
    gap = 0.
    for row in load_panel():
        uid, target = row["uid"], row["target"]
        gt = canonical_box(truth_for_slot(row, read(row["calibration"])["ground_info"]))
        for source in PLATFORMS:
            if source == target:
                continue
            record = predictions[source][uid]
            where = dict(uid=uid,target=target,source=source,recording=row["recording"],
                         C_expanded=float(previous[uid]["C_expanded"]), A_median=float(previous[uid]["A_median"]),
                         B_v3=float(registry[source,uid]["B_canonical_caption"]),
                         official_v3_entered=int(source_meta[source,uid]["entered"]))
            where.update({k:int(v) for k,v in record.items() if isinstance(v,bool)})
            box = record["c_box"]
            if box is None:
                assert all(v is None for v in record["outputs"].values()) and where["C_expanded"] == 0.
                scored = {a:0. for a in ARMS}
            else:
                original = canonical_box(box)
                pairs = [original] + [canonical_box(record["outputs"][a]) for a in ARMS]
                iou, difference = score_boxes(np.stack(pairs),gt)
                gap = max(gap,difference)
                np.testing.assert_allclose(iou[0],where["C_expanded"],rtol=0.,atol=1e-10)
                scored = {a:float(x) for a,x in zip(ARMS,iou[1:])}
                for key, estimated in zip(ARMS, pairs[1:]):
                    where[key+"_center_error_m"] = float(np.linalg.norm(estimated[:3]-gt[:3]))
                    where[key+"_mean_extent_relative_error"] = float(np.mean(np.abs(estimated[3:6]/gt[3:6]-1.)))
                    angle = (estimated[6]-gt[6]+np.pi/2) % np.pi-np.pi/2
                    where[key+"_axis_error_degrees"] = float(np.rad2deg(abs(angle))) if gt[3]/gt[4] >= 1.2 else None
                if not record["c_applicable"]:
                    assert all(np.array_equal(original,b) for b in pairs[1:])
            output.append(where | scored)
    assert len(output) == len({(r["uid"],r["source"]) for r in output}) == 192
    strict_full = [r if r["official_v3_entered"] else r | {arm:r["A_median"] for arm in ("C_expanded",*ARMS)}
                   for r in output]
    scopes = {"all":output, "strict": [r for r in output if r["official_v3_entered"]], "strict_full":strict_full,
              "C_used": [r for r in output if r["c_applicable"]],
              "both_orig_and_qwen_shape": [r for r in output if r["original_shape"]]}
    comparisons = [("D_noun","C_expanded"),("D_joint","C_expanded"),("D_joint","D_noun")]
    comparisons += [(a,"D_joint") for a in ARMS[2:]]
    comparisons += [(a,"C_expanded") for a in ARMS[2:]]
    comparisons.extend((a,"E_v3_box") for a in ARMS[3:])
    groups = []
    metrics, paired, geometry = [], [], []
    for scope, data in scopes.items():
        for target in (*PLATFORMS,"macro"):
            rows = [r for r in data if target in (r["target"],"macro")]
            if rows:
                groups.append((scope,target,"all",rows))
        if scope == "all":
            for source in PLATFORMS:
                for target in PLATFORMS:
                    if source != target:
                        rows = [r for r in data if r["source"]==source and r["target"]==target]
                        groups.append((scope,target,source,rows))
    for scope,target,source,rows in groups:
        for a in ("A_median","B_v3","C_expanded",*ARMS):
            metrics.append(dict(scope=scope,source=source,target=target,arm=a,n=len(rows),
                                **aggregate([r[a] for r in rows])))
        for a in ARMS if scope != "strict_full" else ():
            fields = ("center_error_m","mean_extent_relative_error","axis_error_degrees")
            values = [[r[a+"_"+key] for r in rows if r.get(a+"_"+key) is not None] for key in fields]
            geometry.append(dict(scope=scope,source=source,target=target,arm=a,n=len(rows),
                **{"median_"+key:float(np.median(v)) if v else None for key,v in zip(fields,values)}))
        for a,b in comparisons:
            paired.append(dict(scope=scope,source=source,target=target,arm=a,reference=b,n=len(rows),
                               **delta(rows,a,b)))
    evidence = {t:{k:sum(r[k] for r in output if t in ("macro",r["target"])) for k in
        ("official_v3_entered","c_applicable","noun_applied","relation_applied",
         "noun_changed","relation_changed","original_shape")} for t in (*PLATFORMS,"macro")}
    intervals = {f"{a}_vs_{b}":recorder_ci(output,a,b) for a,b in comparisons}
    save(OUT / "followup_rows.csv",output)
    save(OUT / "followup_metrics.csv",metrics)
    save(OUT / "followup_paired.csv",paired)
    save(OUT / "followup_geometry.csv",geometry)
    write(OUT / "followup_score_receipt.json",dict(directions=len(output),valid_B_v3_caption_replay=True,
          target_updates=0, source_training=0, grounded_no_expert_updates=True, branch_counts=evidence,
          verify_two_scorers=gap, recording_intervals=intervals, predictions_sealed_before_scoring=True,
          prediction_seal=digest(OUT/"followup_prediction_seal.json"), FINAL_access=False))
    print(evidence,flush=True)


if __name__ == "__main__":
    run()
