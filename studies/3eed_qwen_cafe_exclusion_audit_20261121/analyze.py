"""Join immutable evidence to the pre-existing candidate IoUs."""
from collections import defaultdict
import csv
import gzip
import io
import json
import sys

from extract import ACTIVE, AUDIT, HERE, OUT, PAYLOAD, PROTOCOL, SURFACE, roster, verified_trace

sys.path.insert(0, str(SURFACE))
from evaluate import archived_iou, classify


ORIGINAL = {"original_harm": 64, "missed_qwen": 284, "original_rescue": 623,
            "original_correct_pca": 669, "neither_pass": 2769, "both_pass": 1501}
ERRORS = frozenset(("original_harm", "missed_qwen"))
FIELDS = ("original_selection_cells", "scene_selection_cells", "target_fit_seeds", "birth", "fine", "coarse",
          "xor_p", "xor_q", "stable_p", "stable_q", "witness_p", "witness_q", "regions_p", "regions_q",
          "in_window_p", "in_window_q")
SETS = ("compatible_xy", "compatible_z", "base_supports_all", "joint_possible", "positive_area", "joint_area",
        "positive_volume_change_bound", "joint_volume_change_bound")


def iou_upper(iou, replaced):
    if replaced is None:
        return None
    return min(1., (iou+replaced)/(1-replaced)) if replaced < 1. else 1.


def joined(evidence, outcome):
    uid = evidence["uid"]
    assert uid == outcome["uid"]
    identity = {k: outcome[k] for k in ("uid", "target", "frame", "recording", "category")}
    active = evidence["production"]
    assert active == (outcome["production"] == "True")
    current = float(outcome["iou_current"])
    result = dict(**identity, production=active, original_iou=current, original_case="fallback",
                  choose_qwen=None, pca_iou=None, qwen_iou=None, evidence_side=None,
                  effect_direction=None, count_direction=None, **dict.fromkeys(FIELDS))
    for name in ("p", "q"):
        result.update({f"{name}_{k}":None for k in (*SETS, "positive_iou_upper", "joint_iou_upper")})
    if not active:
        return result
    p, q = float(outcome["iou_pca"]), float(outcome["iou_qwen"])
    choose_qwen = evidence["choose_qwen"]
    assert choose_qwen == (outcome["choose_qwen"] == "True")
    assert abs(current-(q if choose_qwen else p)) < 1.e-12
    p_count, q_count = (evidence[f"witness_{name}"] for name in ("p", "q"))
    side = ("both" if q_count else "p") if p_count else ("q" if q_count else "none")
    greater = "p" if p_count > q_count else ("q" if q_count > p_count else "tie")
    failing = None
    if (p > .5) != (q > .5):
        failing = "q" if p > .5 else "p"
    count_direction = "tied" if greater == "tie" else "opposes_failing" if greater == failing else "opposes_passing"
    if failing is None:
        direction = count_direction = "same_accuracy_status"
    elif side in ("none", "both"):
        direction = side
    else:
        direction = "opposes_failing" if side == failing else "opposes_passing"
    result.update(original_case=classify(p, q, choose_qwen), choose_qwen=choose_qwen,
                  pca_iou=p, qwen_iou=q, evidence_side=side,
                  effect_direction=direction, count_direction=count_direction)
    result.update({k:evidence[k] for k in FIELDS})
    for name, base in (("p", p), ("q", q)):
        entry = evidence[name+"_set"]
        result.update({f"{name}_{k}":entry[k] for k in SETS})
        for kind in ("positive", "joint"):
            result[f"{name}_{kind}_iou_upper"] = iou_upper(base, entry[f"{kind}_volume_change_bound"])
    return result


def groups(rows):
    blocks = defaultdict(list)
    for r in rows:
        if r["production"]:
            t, case = r["target"], r["original_case"]
            keys = [("all", "all"), ("platform", t), ("recording", t+"/"+r["recording"]),
                    ("category", r["category"]), ("case", case), ("platform_case", t+"/"+case)]
            if case in ERRORS:
                keys += [("case", "previous_avoidable"), ("platform_case", t+"/previous_avoidable")]
            for key in keys:
                blocks[key].append(r)
    return blocks


def summarize(block):
    xor = sum(r["xor_p"]+r["xor_q"] > 0 for r in block)
    stable = sum(r["stable_p"]+r["stable_q"] > 0 for r in block)
    p_only = sum(r["evidence_side"] == "p" for r in block)
    q_only = sum(r["evidence_side"] == "q" for r in block)
    both = sum(r["evidence_side"] == "both" for r in block)
    supported = p_only + q_only + both
    outside_only = sum(r["evidence_side"] != "none" and r["in_window_p"]+r["in_window_q"] == 0 for r in block)
    assert supported <= stable <= xor <= len(block)
    return dict(pairs=len(block), xor_coverage=xor, stable_coverage=stable, witness_coverage=supported,
                one_sided=p_only+q_only, p_only=p_only, q_only=q_only, both=both, none=len(block)-supported,
                outside_window_only=outside_only, p_cells=sum(r["witness_p"] for r in block),
                q_cells=sum(r["witness_q"] for r in block),
                opposes_failing=sum(r["effect_direction"] == "opposes_failing" for r in block),
                opposes_passing=sum(r["effect_direction"] == "opposes_passing" for r in block),
                more_in_failing=sum(r["count_direction"] == "opposes_failing" for r in block),
                more_in_passing=sum(r["count_direction"] == "opposes_passing" for r in block))


def possible(r, key, threshold):
    return any(r[f"{side}_{key}"] is not None and r[f"{side}_{key}"] > threshold for side in ("p", "q"))


def center_summary(block):
    block = [r for r in block if r["original_case"] == "neither_pass"]
    positive = sum(any(r[f"{s}_compatible_xy"] and r[f"{s}_compatible_z"] for s in ("p", "q")) for r in block)
    joint = sum(r["p_joint_possible"] or r["q_joint_possible"] for r in block)
    positive_upper = sum(possible(r, "positive_iou_upper", .5) for r in block)
    joint_upper = sum(possible(r, "joint_iou_upper", .5) for r in block)
    assert joint_upper <= positive_upper and joint_upper <= joint <= positive <= len(block)
    return dict(
        pairs=len(block), both_horizontal_incompatible=sum(not r["p_compatible_xy"] and not r["q_compatible_xy"] for r in block),
        both_vertical_incompatible=sum(not r["p_compatible_z"] and not r["q_compatible_z"] for r in block),
        positive_feasible=positive, joint_feasible=joint, joint_conflict=positive-joint,
        joint_allows_movement=sum(possible(r, "joint_volume_change_bound", 1.e-9) for r in block),
        shrunk_area_on_any_pose=sum(any(r[f"{s}_positive_area"] is not None
                                        and r[f"{s}_positive_area"]-r[f"{s}_joint_area"]>1.e-9
                                        for s in ("p", "q")) for r in block),
        witness_available=sum(r["evidence_side"] != "none" for r in block),
        positive_bound_allows_acc50=positive_upper, joint_bound_allows_acc50=joint_upper,
        opposed_pose_bound_allows_acc50=sum(any(r[f"witness_{s}"] > 0
                                               and (r[f"{s}_joint_iou_upper"] or 0.) > .5
                                               for s in ("p", "q")) for r in block),
        ruled_out_acc50=len(block)-joint_upper,
    )


def write_table(path, rows):
    if path.suffix == ".gz":
        with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz, \
                io.TextIOWrapper(gz, encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run():
    seal_path = OUT/"evidence_seal.json"
    seal = roster.read(seal_path)
    trace, _ = verified_trace()
    assert seal["payload"] == roster.sha(PAYLOAD) and seal["protocol"] == roster.sha(PROTOCOL)
    assert seal["source_trace_payload"] == trace["payload"]
    assert seal["frozen_method_lock"] == roster.sha(ACTIVE/"METHOD_LOCK.json")
    assert seal["extractor"] == roster.sha(HERE/"extract.py") and seal["geometry"] == roster.sha(HERE/"center_sets.py")
    assert not seal["iou_rows_opened"] and not seal["FINAL_access"] and seal["prediction_changes"] == 0
    outcome = archived_iou()
    with gzip.open(PAYLOAD, "rt") as f:
        rows = []
        for line in f:
            item = json.loads(line)
            rows.append(joined(item, outcome[item["uid"]]))
    assert len(rows) == len({r["uid"] for r in rows}) == len(outcome) == 9310
    blocks = groups(rows)
    assert len(blocks["all", "all"]) == 5910
    assert all(len(blocks["case", case]) == n for case, n in ORIGINAL.items())
    assert len(blocks["case", "previous_avoidable"]) == 348
    stats = [dict(group=k, subset=v, **summarize(block)) for (k,v),block in sorted(blocks.items())]
    centers = [dict(group=k, subset=v, **center_summary(block)) for (k,v),block in sorted(blocks.items())
               if k in ("all", "platform", "recording", "category") and any(r["original_case"] == "neither_pass" for r in block)]
    written = ["exclusion_summary.csv", "both_fail_center_bounds.csv"]
    write_table(OUT/written[0], stats)
    write_table(OUT/written[1], centers)
    for platform, n in (("waymo", 3782), ("drone", 2792), ("quad", 2736)):
        subset = [r for r in rows if r["target"] == platform]
        assert len(subset) == n
        name = f"rows_{platform}.csv.gz"
        write_table(OUT/name, subset)
        written.append(name)
    coverage = summarize(blocks["case", "previous_avoidable"])
    roster.write(OUT/"analysis_receipt.json", dict(
        evidence_seal=roster.sha(seal_path), code=roster.sha(__file__), previous_analysis=roster.sha(SURFACE/"DECISION.json"),
        archived_iou_receipt=roster.sha(AUDIT/"results/measurement_receipt.json"),
        descriptions=len(rows), active_pairs=5910, original_cases=ORIGINAL,
        preceding_avoidable_failures=coverage, majority_unambiguous=2*coverage["one_sided"]>coverage["pairs"],
        both_fail_center_bounds=center_summary(blocks["all", "all"]),
        files={f:roster.sha(OUT/f) for f in written},
        new_Qwen_queries=0, training=False, additional_GT_file_reads=0, prediction_changes=0, FINAL_access=False,
    ))
    for item in stats:
        if item["group"] in ("all", "case", "platform"):
            print(item, flush=True)
    print("Both fail", center_summary(blocks["all", "all"]), flush=True)


if __name__ == "__main__":
    run()
