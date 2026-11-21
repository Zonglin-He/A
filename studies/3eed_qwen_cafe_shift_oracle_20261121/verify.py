"""Independently recount the two cohorts from the anonymized diagnostic rows."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from math import isclose
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE/"results"
OLD = HERE.parent/"3eed_qwen_cafe_exclusion_audit_20261121"/"results"
SIDES = ("p", "q")
ARMS = ("free", "positive", "joint")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    constraints = json.loads((OUT/"constraint_seal.json").read_bytes())
    scored = json.loads((OUT/"measurement_receipt.json").read_bytes())
    old = json.loads((OLD/"analysis_receipt.json").read_bytes())
    assert constraints["protocol"] == sha(HERE/"PROTOCOL.json") and constraints["code"] == sha(HERE/"prepare.py")
    assert constraints["previous_analysis"] == sha(OLD/"analysis_receipt.json")
    assert constraints["previous_witness_seal"] == sha(OLD/"evidence_seal.json")
    assert scored["constraint_seal"] == sha(OUT/"constraint_seal.json")
    assert scored["sweep_code"] == sha(HERE/"sweep.py") and scored["score_code"] == sha(HERE/"oracle.py")
    assert not constraints["FINAL_access"] and not scored["FINAL_access"]
    assert constraints["new_GT_reads"] == constraints["original_predictions_changed"] == scored["original_predictions_changed"] == 0
    assert scored["ordinary_GT_descriptions"] == constraints["descriptions"] == 2769
    assert scored["no_size_z_yaw_changes"] and not scored["training"] and scored["new_Qwen_queries"] == 0
    assert scored["max_quadratic_probe_error"] < 1.e-8 and scored["max_upright_iou_recheck_error"] < 1.e-8
    for name,digest in scored["files"].items():
        assert sha(OUT/name) == digest
    original = {}
    for platform in ("waymo","drone","quad"):
        name = f"rows_{platform}.csv.gz"
        assert sha(OLD/name) == old["files"][name]
        cases = [r for r in table(OLD/name) if r["original_case"] == "neither_pass"]
        for r in cases:
            assert r["uid"] not in original
            original[r["uid"]] = r
    rows = table(OUT/"rows.csv.gz")
    assert len(rows) == len({r["uid"] for r in rows}) == len(original) == 2769
    assert Counter(r["target"] for r in rows) == constraints["cohort"]
    all_margins = []
    focus = []
    for r in rows:
        source = original[r["uid"]]
        p_focus = []
        for side,name in zip(SIDES,("pca","qwen"),strict=True):
            assert all(r[k] == source[k] for k in ("target","category","recording","frame"))
            initial = float(r[f"{side}_original_iou"])
            assert initial <= .5 and isclose(initial,float(source[f"{name}_iou"]),rel_tol=0.,abs_tol=1.e-12)
            assert r["original_qwen"] == source["choose_qwen"]
            witness = int(source[f"witness_{side}"])
            previous_ceiling = float(source[f"{side}_joint_iou_upper"] or "0")
            assert int(r[f"{side}_witness_count"]) == witness
            p_focus.append(witness>0 and previous_ceiling>.5)
            assert (r[f"{side}_focus_pose"] == "True") == p_focus[-1]
            free,pos,joint = [float(r[f"{side}_{a}"] or "-1") for a in ARMS]
            assert initial-1.e-8 <= free <= 1. and joint <= pos+1.e-8 <= free+2.e-8
            assert (joint >= 0.) == (source[f"{side}_joint_possible"] == "True")
            assert (pos >= 0.) == bool(source[f"{side}_positive_area"])
            for arm,value in zip(ARMS,(free,pos,joint),strict=True):
                if value >= 0.:
                    all_margins.append(abs(value-.5))
                    if arm != "free":
                        assert value <= float(source[f"{side}_{arm}_iou_upper"])+1.e-8
            assert not joint>.5 or pos>.5
        assert (r["focus"] == "True") == any(p_focus)
        if any(p_focus):
            focus.append(r)
    assert len(focus) == scored["focus"] == constraints["focus"] == 388
    assert min(all_margins) == scored["min_acc50_margin"] > 5.e-8

    summary = table(OUT/"summary.csv")
    assert len(summary) == 28
    for s in summary:
        labels = []
        for r in rows:
            if s["subset"] != "all" and (r["focus"] == "True") != (s["subset"] == "focus"):
                continue
            if s["platform"] != "all" and s["platform"] != r["target"]:
                continue
            if s["pose_scope"] == "opposed_poses":
                sides = tuple(side for side in SIDES if r[f"{side}_focus_pose"] == "True")
            elif s["pose_scope"] == "original_pose":
                sides = ("q",) if r["original_qwen"] == "True" else ("p",)
            else:
                assert s["pose_scope"] == "either_pose"
                sides = SIDES
            labels.append([max(float(r[f"{side}_{arm}"] or "-1") for side in sides) for arm in ARMS])
        counts = {"descriptions":len(labels)}
        for i,arm in enumerate(ARMS):
            counts[arm+"_feasible"] = sum(r[i]>=0. for r in labels)
            counts[arm+"_rescues"] = sum(r[i]>.5 for r in labels)
        counts["positive_rescue_removed"] = sum(r[1]>.5>=r[2] for r in labels)
        counts["joint_only_rescue"] = sum(r[2]>.5>=r[1] for r in labels)
        assert counts["joint_only_rescue"] == 0
        assert all(int(s[k]) == count for k,count in counts.items())
    result = dict(
        descriptions=len(rows), focus=len(focus), groups_recounted=len(summary),
        both_original_boxes_failed=True, feasible_set_scores_nested=True,
        per_pose_focus_identity_verified=True, original_predictions_changed=0, FINAL_access=False,
    )
    (OUT/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(result)


if __name__ == "__main__":
    main()
