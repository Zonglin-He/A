"""Summarize anonymous paired results and the predeclared question-level rule."""
from collections import Counter, defaultdict
from statistics import mean

from shared import ARMS, HERE, OUT, PLATFORMS, digest, read, write
import summarize as prior
from uncertainty import bootstrap


LEVELS = ("L0", "L1", "L2", "L3")


def uncertainty(qa, geometry, comparisons):
    outputs, samples = defaultdict(dict), defaultdict(dict)
    for row in qa:
        outputs[row["qid"]][row["arm"]] = row
    for row in geometry:
        if row["group"] == "all":
            samples[row["uid"]][row["arm"]] = row
    intervals = []
    for arm, ref in comparisons:
        for endpoint in LEVELS + ("pixel_xyz", "pixel_order", "distance", "height", "object_order",
                                  "separation", "static_right", "yaw_90"):
            pairs = [(rows[arm], rows[ref]) for rows in outputs.values()
                     if endpoint in (rows[arm]["level"], rows[arm]["family"])]
            accuracy = [(x, [100*(int(x["correct"])-int(y["correct"]))]) for x, y in pairs]
            err = [(x, [float(x["abs_error"])-float(y["abs_error"])]) for x,y in pairs
                   if x["abs_error"] and y["abs_error"]]
            intervals.append(dict(arm=arm, reference=ref, task=endpoint,
                                accuracy=bootstrap(accuracy, ("accuracy_pp",)),
                                paired_numeric=bootstrap(err, ("mae_m",))))
        pairs = [(rows[arm], rows[ref]) for rows in samples.values() if arm in rows and ref in rows]
        errors = [(x,[float(x[k])-float(y[k]) for k in ("depth_mae", "xyz_mae")]) for x,y in pairs
                  if x["depth_mae"] and y["depth_mae"]]
        if errors:
            intervals.append(dict(arm=arm, reference=ref, task="heldout_geometry",
                                  heldout=bootstrap(errors, ("depth_mae_m","xyz_mae_m"))))
    return intervals


def policy(qa, by_level):
    items = defaultdict(dict)
    for r in qa:
        items[r["qid"]][r["arm"]] = r
    table = []
    for compare in ("moge", "prior_depth", "global_scale"):
        by_tile = defaultdict(list)
        for copies in items.values():
            old = copies[compare]
            new = copies[by_level[old["level"]]]
            by_tile[old["target"], old["level"]].append((int(new["correct"]), int(old["correct"])))
        for slice_name in (*PLATFORMS, "platform_macro", "pooled"):
            for level in (*LEVELS, "all_questions", "equal_levels"):
                blocks = [(p, group) for (p, section), group in by_tile.items()
                          if (slice_name in ("pooled", "platform_macro") or slice_name == p)
                          and (level in ("all_questions", "equal_levels") or level == section)]
                rows = [pair for _, block in blocks for pair in block]
                if level == "equal_levels":
                    rates = [mean(row[i] for row in block) for _,block in blocks for i in (0,1)]
                    model, reference = mean(rates[::2]), mean(rates[1::2])
                elif slice_name == "platform_macro":
                    grouped = defaultdict(list)
                    for p, block in blocks:
                        grouped[p].extend(block)
                    model, reference = (mean(mean(row[i] for row in block) for block in grouped.values()) for i in (0,1))
                else:
                    model, reference = (mean(row[i] for row in rows) for i in (0,1))
                table.append(dict(target=slice_name, level=level, reference=compare, n=len(rows),
                                 pass_pct=100*model, reference_pct=100*reference, delta_pp=100*(model-reference),
                                 rescue=sum((x,y)==(1,0) for x,y in rows), harm=sum((x,y)==(0,1) for x,y in rows)))
    return table


def coordinates(rows, consistency):
    data = []
    for platform, block in prior.subsets(rows):
        pair = defaultdict(list)
        for row in block:
            pair[row["uid"]].append(row)
        assert all(len(group)==2 for group in pair.values())
        delta = Counter(int(row["correct"])-int(row["direct"]) for row in block)
        ordinary_pair = [r for r in consistency if r["arm"]=="prior_depth" and
                         r["task"]=="rotations_reverse" and r["uid"] in pair]
        assert len(ordinary_pair) == len(pair)
        # With two inverse A/B questions, any two valid answers are opposite
        # exactly when either both are correct or both are incorrect.
        agreements = [sum(int(r["correct"]) for r in group) != 1 and all(int(r["valid"]) for r in group)
                      for group in pair.values()]
        data.append(dict(target=platform, n_questions=len(block), n_pairs=len(pair),
            direct_correct=sum(int(r["direct"]) for r in block), explicit_correct=sum(int(r["correct"]) for r in block),
            deterministic_correct=sum(int(r["deterministic"]) for r in block), rescue=delta[1], harm=delta[-1],
            direct_both=sum(int(r["both_correct"]) for r in ordinary_pair),
            explicit_both=sum(all(int(r["correct"]) for r in group) for group in pair.values()),
            direct_opposite=sum(int(r["consistent"]) for r in ordinary_pair), explicit_opposite=sum(agreements),
            explicit_valid=sum(int(r["valid"]) for r in block)))
    return data


def run():
    plan_file = HERE / "ANALYSIS_PLAN.json"
    plan, seal, receipt = read(plan_file), read(OUT / "analysis_seal.json"), read(OUT / "score_receipt.json")
    assert seal["files"]["diagnose96.py"] == digest(__file__) and seal["plan"] == digest(plan_file)
    for filename, expected in receipt["files"].items():
        assert digest(OUT / filename) == expected
    qa, geo, logic, coord = (prior.read_csv(name+".csv") for name in (
        "qa_rows","geometry_rows","consistency_rows","coordinate_qa_rows"))
    assert len({(r["qid"], r["arm"]) for r in qa}) == len(qa) == len({r["qid"] for r in qa})*len(ARMS)
    assert all(len({r["uid"] for r in qa if r["target"] == p}) == 32 for p in PLATFORMS)
    prior.COMPARISONS = tuple(tuple(pair) for pair in plan["same_image_comparisons"])
    tables = {
        "qa_summary.csv": prior.question_summary(qa),
        "paired.csv": prior.paired_summary(qa),
        "geometry_summary.csv": prior.geometry_summary(geo),
        "consistency_summary.csv": prior.logic_summary(logic),
        "perspective.csv": prior.perspective(qa),
        "mechanisms.csv": prior.mechanisms(qa),
        "coordinate_diagnostics.csv": coordinates(coord, logic),
        "locked_level_policy.csv": policy(qa, plan["previous_54_frame_level_readout"]),
    }
    for filename, rows in tables.items():
        prior.save_csv(filename, rows)
    write(OUT / "recording_uncertainty.json", uncertainty(qa, geo, prior.COMPARISONS))
    write(OUT / "summary_receipt.json", dict(score_receipt=digest(OUT / "score_receipt.json"),
        answers=len(qa), coordinate_answers=len(coord), tables={k:digest(OUT / k) for k in tables}))


if __name__ == "__main__":
    run()
