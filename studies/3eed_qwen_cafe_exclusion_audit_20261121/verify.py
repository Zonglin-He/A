"""Recount the public audit independently, without point clouds or target labels."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from math import fsum, isclose
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE/"results"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def covers(row, arm, variant):
    value = row[f"{arm}_{variant}_iou_upper"]
    return bool(value) and float(value) > .5


def match(row, group, name):
    target, case = row["target"], row["original_case"]
    avoidable = case in ("original_harm", "missed_qwen")
    return {
        "all": name == "all",
        "platform": name == target,
        "recording": name == f"{target}/{row['recording']}",
        "category": name == row["category"],
        "case": name == case or (name == "previous_avoidable" and avoidable),
        "platform_case": name == f"{target}/{case}" or (name == f"{target}/previous_avoidable" and avoidable),
    }[group]


def main():
    seal = json.loads((OUT/"evidence_seal.json").read_bytes())
    audit = json.loads((OUT/"analysis_receipt.json").read_bytes())
    assert seal["protocol"] == sha(HERE/"PROTOCOL.json")
    assert seal["extractor"] == sha(HERE/"extract.py") and seal["geometry"] == sha(HERE/"center_sets.py")
    assert audit["evidence_seal"] == sha(OUT/"evidence_seal.json") and audit["code"] == sha(HERE/"analyze.py")
    assert seal["original_target_selection_exact"] and seal["fit_selection_disjoint"] and not seal["iou_rows_opened"]
    for record in (seal, audit):
        assert not record["FINAL_access"] and record["additional_GT_file_reads"] == 0
        assert record["prediction_changes"] == record["new_Qwen_queries"] == 0
    for name, digest in audit["files"].items():
        assert sha(OUT/name) == digest

    all_records, macro = [], []
    for target, n in (("waymo", 3782), ("drone", 2792), ("quad", 2736)):
        part = rows(OUT/f"rows_{target}.csv.gz")
        assert len(part) == n and all(r["target"] == target for r in part)
        macro.append(sum(float(r["original_iou"]) > .5 for r in part)/n*100.)
        all_records.extend(part)
    assert len(all_records) == len({r["uid"] for r in all_records}) == 9310
    assert len({r["frame"] for r in all_records}) == 7221 and len({r["recording"] for r in all_records}) == 98
    assert isclose(fsum(macro)/3., 34.40889469558785, rel_tol=0., abs_tol=1.e-11)
    active = [r for r in all_records if r["production"] == "True"]
    assert len(active) == seal["active_pairs"] == audit["active_pairs"] == 5910
    assert sum(r["choose_qwen"] == "True" for r in active) == seal["original_qwen_choices"] == 2640
    assert sum(int(r["original_selection_cells"]) for r in active) == seal["original_selection_cells"] == 688114
    for r in all_records:
        if r["production"] != "True":
            assert r["original_case"] == "fallback" and r["evidence_side"] == r["pca_iou"] == ""
            continue
        p, q = float(r["pca_iou"]), float(r["qwen_iou"])
        selected = r["choose_qwen"] == "True"
        assert isclose(q if selected else p, float(r["original_iou"]), rel_tol=0., abs_tol=1.e-12)
        if (p > .5) == (q > .5):
            assert r["original_case"] == ("both_pass" if p > .5 else "neither_pass")
        else:
            if q > .5:
                assert r["original_case"] == ("original_rescue" if selected else "missed_qwen")
            else:
                assert r["original_case"] == ("original_harm" if selected else "original_correct_pca")
            fail = "p" if p <= .5 else "q"
            direction = r["evidence_side"]
            if direction not in ("none", "both"):
                direction = "opposes_failing" if direction == fail else "opposes_passing"
            assert r["effect_direction"] == direction
        observed = "".join(s for s in ("p", "q") if int(r[f"witness_{s}"]))
        assert r["evidence_side"] == {"": "none", "p": "p", "q": "q", "pq": "both"}[observed]
        assert int(r["xor_p"])+int(r["xor_q"]) <= int(r["scene_selection_cells"])
        for side, base in (("p", p), ("q", q)):
            assert 0 <= int(r[f"in_window_{side}"]) <= int(r[f"witness_{side}"]) <= int(r[f"stable_{side}"]) <= int(r[f"xor_{side}"])
            assert 0 <= 5*int(r[f"regions_{side}"]) <= int(r[f"witness_{side}"])
            for variant in ("positive", "joint"):
                fraction = r[f"{side}_{variant}_volume_change_bound"]
                upper = r[f"{side}_{variant}_iou_upper"]
                if fraction:
                    s = float(fraction)
                    ceiling = float(upper)
                    assert 0. <= s <= 1. and base-1.e-12 <= ceiling <= 1.
                    assert isclose(ceiling*(1-s), min(1-s, base+s), rel_tol=0., abs_tol=1.e-12)
                else:
                    assert not upper

    cases = Counter(r["original_case"] for r in active)
    assert dict(cases) == audit["original_cases"]
    stats = rows(OUT/"exclusion_summary.csv")
    for output in stats:
        block = [r for r in active if match(r,output["group"],output["subset"])]
        labels = Counter(r["evidence_side"] for r in block)
        directions = Counter(r["effect_direction"] for r in block)
        assert int(output["pairs"]) == len(block)
        assert [int(output[k]) for k in ("p_only", "q_only", "both", "none")] == [labels[k] for k in ("p", "q", "both", "none")]
        assert int(output["witness_coverage"]) == len(block)-labels["none"]
        assert int(output["one_sided"]) == labels["p"]+labels["q"]
        assert all(int(output[k]) == directions[k] for k in ("opposes_failing", "opposes_passing"))
    for output in rows(OUT/"both_fail_center_bounds.csv"):
        block = [r for r in active if r["original_case"] == "neither_pass" and match(r,output["group"],output["subset"])]
        assert int(output["pairs"]) == len(block)
        assert int(output["joint_feasible"]) == sum("True" in (r["p_joint_possible"],r["q_joint_possible"]) for r in block)
        assert all(int(output[f"{variant}_bound_allows_acc50"]) == sum(covers(r,"p",variant) or covers(r,"q",variant) for r in block)
                   for variant in ("positive", "joint"))
        assert int(output["opposed_pose_bound_allows_acc50"]) == sum(
            any(int(r[f"witness_{s}"]) > 0 and covers(r,s,"joint") for s in ("p", "q")) for r in block
        )
    checked = dict(descriptions=9310, active_pairs=5910, fallbacks=3400, original_case_counts=dict(cases),
                   original_macro_acc50=fsum(macro)/3., groups_recounted=len(stats),
                   original_predictions_changed=0, additional_GT_file_reads=0, FINAL_access=False)
    (OUT/"verification.json").write_text(json.dumps(checked, indent=2)+"\n")
    print(checked)


if __name__ == "__main__":
    main()
