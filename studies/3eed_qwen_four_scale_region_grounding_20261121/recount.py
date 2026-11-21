"""Independently reproduce C's metrics and the fixed-ray depth oracle bounds."""
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median


OUT = Path(__file__).resolve().parent / "results"
PLATFORMS = ("waymo", "drone", "quad")
METHODS = ("A_median", "C_expanded", "F_expanded", "C_fit_tree", "F_fit_tree")
LEVELS = ("expanded", "fit_tree")


def load(name):
    with (OUT / (name + ".csv")).open(newline="") as f:
        return list(csv.DictReader(f))


def metrics(values):
    raw = tuple(float(v) for v in values)
    return 100*mean(x > .25 for x in raw), 100*mean(x > .5 for x in raw), 100*mean(raw)


def structural(rows):
    old = [r for r in rows if r.get("A_median_fit_purity")
           and int(r["A_median_fit_n"]) >= 5 and float(r["A_median_fit_purity"]) < .5]
    result = {"n": len(rows), "baseline_low_purity": len(old)}
    for level, method in zip(LEVELS, ("C_expanded", "C_fit_tree")):
        exposed = [r for r in rows if r[level+"_status"] == "used_region"]
        assert all(float(r[level + "_candidate_oracle_iou"]) + 1e-12 >= float(r[method]) for r in exposed)
        pure = [r for r in old if r[level+"_status"] == "used_region" and float(r[method+"_fit_purity"]) >= .5]
        pool = [r for r in old if r.get(level+"_high_purity_candidate") == "1"]
        bounds = {f"extra_correct_at_{int(cut*100)}":
            sum(float(r["A_median"]) <= cut < float(r[level + "_candidate_oracle_iou"]) for r in exposed)
                  for cut in (.25, .50)}
        result[level] = dict(used=len(exposed), low_purity_rescued=len(pure), low_purity_recoverable=len(pool),
                             **bounds)
    return result


def surface_report(rows):
    entries = {}
    for source in ("macro", *PLATFORMS):
        entries[source] = {}
        for task in ("all_window_rays", "target_rays"):
            entries[source][task] = {}
            for arm in ("A_median", "C_fit_tree", "F_fit_tree"):
                found = [r for r in rows if source in (r["target"], "macro") and r["group"] == task and r["arm"] == arm]
                n, hits, common = [sum(int(r[key]) for r in found) for key in ("n", "hits", "both_hits")]
                entries[source][task][arm] = dict(rays=n, hits=hits, common_hits=common,
                    base_mae_common=sum(float(r["A_error_sum_on_common"]) for r in found)/common,
                    mae_common=sum(float(r["error_sum_on_common"]) for r in found)/common)
    return entries


def factorial_and_controls(source_rows, primary_rows):
    normalized = load("v3_text_replay_rows")
    assert len(normalized) == 192
    assert all(abs(float(r["B_archived"])-float(r["B_canonical_caption"])) < 1e-10 for r in normalized)
    assert sum(abs(float(r["B_archived"])-float(r["B_raw_caption"])) > 1e-8 for r in normalized) == 92
    controls, followup = load("controls_rows"), load("followup_rows")
    assert len(controls) == 96 and len(followup) == 192
    assert {r["uid"] for r in controls} == {r["uid"] for r in primary_rows}
    assert {(r["uid"],r["source"]) for r in source_rows} == {(r["uid"],r["source"]) for r in followup}
    indexed = {r["uid"]:r for r in controls}
    expected = {"controls":controls, "followup":followup}
    for prefix, rows in expected.items():
        if prefix == "followup":
            selected = [r for r in load(prefix+"_metrics")
                        if r["scope"] == "all" and r["source"] == "all" and r["target"] == "macro"]
        else:
            selected = [r for r in load(prefix+"_metrics") if r["scope"] == "all" and r["target"] == "macro"]
        for r in selected:
            values = metrics(x[r["arm"]] for x in rows)
            assert max(abs(value-float(r[col])) for value,col in zip(values,("acc25","acc50","miou"))) < 1e-10
    preserved = [r for r in followup if r["noun_changed"]=="0" and r["relation_changed"]=="0"]
    gaps = [abs(float(row[a])-float(indexed[row["uid"]][b])) for row in preserved
            for a,b in (("E_qwen_size","C_E_Qsize"),("E_qwen_pose","C_E_Qpose"))]
    assert max(gaps) < 1e-10
    assert all(r["D_joint"] == r["D_noun"] for r in followup)
    return dict(exact_normalized_v3_replays=len(normalized), raw_format_v3_iou_differences=92,
        identical_D_noun_joint_ious=len(followup), same_region_E_comparisons=len(gaps),
        max_same_region_E_iou_gap=max(gaps), source_free_completion_frames=len(controls))


def run():
    rows, pairs = load("frames"), load("source_pairs")
    assert len({r["uid"] for r in rows}) == len(rows) == 96
    assert Counter(r["target"] for r in rows) == {x: 32 for x in PLATFORMS}
    assert len({(r["source"],r["uid"]) for r in pairs}) == len(pairs) == 192
    assert sum(int(r["entered"]) for r in pairs) == sum(int(r["selected"]) for r in pairs) == 8
    for row in rows:
        for method, f, status in (("C_expanded", "F_expanded", "expanded_status"),
                                 ("C_fit_tree", "F_fit_tree", "fit_tree_status")):
            if row[status] == "used_region":
                assert row[method+"_fit_n"] == row[f+"_fit_n"]
                assert 5 <= int(row[method+"_fit_n"]) <= int(row["A_median_fit_n"])
            else:
                assert row[method] == row["A_median"] == row[f]
    summary = {r["arm"]: r for r in load("metrics") if r["scope"] == "all" and r["target"] == "macro"}
    maximum = max(abs(x-float(summary[arm][col])) for arm in METHODS
                  for x, col in zip(metrics(r[arm] for r in rows), ("acc25","acc50","miou")))
    assert maximum < 1e-10
    compare = [r for r in load("paired") if r["target"] == "macro" and r["scope"] == "all"
               and r["arm"] in METHODS]
    for group in compare:
        for cut in (25,50):
            states = [(float(row[group["arm"]])>cut/100., float(row[group["reference"]])>cut/100.) for row in rows]
            assert sum(a and not b for a,b in states) == int(group[f"rescue{cut}"])
            assert sum(b and not a for a,b in states) == int(group[f"harm{cut}"])
    active = [r for r in pairs if r["entered"]=="1"]
    assert len({r["uid"] for r in active}) == 4
    result = dict(unique_frames=96, v3_source_records=192, max_main_metric_delta_pp=maximum,
        strict_unique_frames=4, gate_coverage_verified=True, matched_subset_counts=True,
        by_platform_structure={p:structural([r for r in rows if p in (r["target"], "macro")])
                               for p in (*PLATFORMS, "macro")},
        surface=surface_report(load("surface")), additional_checks=factorial_and_controls(pairs,rows),
        FINAL_access=False)
    (OUT / "recount.json").write_text(json.dumps(result, indent=2)+"\n")
    print(result["by_platform_structure"])


if __name__ == "__main__":
    run()
