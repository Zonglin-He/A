"""Independently reproduce headline results using only the public scalars."""
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


OUT = Path(__file__).resolve().parent / "results"
PLATFORMS = ("waymo", "drone", "quad")
ARMS = ("moge", "global_scale", "prior_depth", "wrong_geometry", "oracle_depth")


def table(name):
    file = OUT / (name + ".csv")
    with (file.open(newline="") if file.exists() else gzip.open(str(file) + ".gz", mode="rt", newline="")) as f:
        return list(csv.DictReader(f))


def run():
    qa, geometry, anchors, coords, logic = (table(name) for name in (
        "qa_rows", "geometry_rows", "oracle_anchor_rows", "coordinate_qa_rows", "consistency_rows"))
    observed = defaultdict(dict)
    frames = defaultdict(set)
    rates = defaultdict(list)
    for row in qa:
        bundle = observed[row["qid"]]
        assert row["arm"] not in bundle and row["correct"] in ("0", "1")
        assert int(row["correct"]) <= int(row["valid"])
        bundle[row["arm"]] = row
        frames[row["target"]].add(row["uid"])
        rates[row["arm"],row["target"],row["level"]].append(int(row["correct"]))
    assert all(set(copies) == set(ARMS) and len({r["uid"] for r in copies.values()}) == 1 for copies in observed.values())
    assert all(len(frames[p]) == 32 for p in PLATFORMS)
    assert len(set.union(*frames.values())) == 96
    actual = {(r["arm"], r["task"]): float(r["acc_pct"]) for r in table("qa_summary")
              if r["target"] == "platform_macro" and r["task"] in ("L0","L1","L2","L3")}
    errors = []
    for (arm, level), value in actual.items():
        errors.append(abs(mean(mean(rates[arm,p,level]) for p in PLATFORMS)*100-value))

    seen = defaultdict(dict)
    for row in geometry:
        if row["group"] == "all":
            seen[row["uid"]][row["arm"]] = row
    assert len(seen) == 96
    assert all(set(rows) == set(ARMS[:-1]) and len({v["n"] for v in rows.values()}) == 1 for rows in seen.values())
    holdout = sum(int(rows["moge"]["n"]) for rows in seen.values())
    assert holdout == sum(int(r["n"]) for r in anchors if r["group"] == "all")
    assert all(float(r["depth_mae"]) == 0. for r in anchors if int(r["n"]) > 0)
    for row in table("geometry_summary"):
        if row["group"] != "all" or row["target"] != "platform_macro":
            continue
        arm = row["arm"]
        estimates = [mean(float(f[arm]["depth_mae"]) for f in seen.values()
                          if f[arm]["target"] == platform and f[arm]["depth_mae"]) for platform in PLATFORMS]
        assert abs(mean(estimates)-float(row["depth_mae"])) < 1e-12

    assert set(r["qid"] for r in coords) == {qid for qid, copies in observed.items() if copies["moge"]["level"] == "L3"}
    pairs = defaultdict(list)
    for r in coords:
        pairs[r["uid"]].append(r)
        assert r["deterministic"] == "1" and r["direct"] == observed[r["qid"]]["prior_depth"]["correct"]
    assert all(len(rows) == 2 for rows in pairs.values())
    report = next(r for r in table("coordinate_diagnostics") if r["target"] == "pooled")
    det = int(report["deterministic_correct"])
    assert det == len(coords) and int(report["explicit_correct"]) == sum(int(r["correct"]) for r in coords)
    assert int(report["explicit_both"]) == sum(all(r["correct"]=="1" for r in rows) for rows in pairs.values())
    for arm in ARMS:
        rows = [r for r in logic if r["arm"] == arm and r["task"] == "rotations_reverse"]
        assert {r["uid"] for r in rows} == set(pairs)
        assert sum(int(r["both_correct"]) for r in rows) <= sum(int(r["consistent"]) for r in rows)
    assert max(errors) < 1e-10
    audit = dict(physical_frames=96, questions=len(observed), answers=len(qa),
        paired_oracle_answers=len(coords), holdout_cells=holdout, max_macro_difference_pp=max(errors),
        target_counts={p:len(ids) for p,ids in frames.items()},
        arm_question_counts=Counter(r["arm"] for r in qa), all_pairs_preserved=True,
        reproduced_geometry=True, verified_oracle_separation=True,
        summary_sha256=hashlib.sha256((OUT/"qa_summary.csv").read_bytes()).hexdigest())
    (OUT/"recount.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    run()
