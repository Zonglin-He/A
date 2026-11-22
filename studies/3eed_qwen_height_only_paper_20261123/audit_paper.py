"""Recount the scored ledgers and summarize the empirical feasible roof intervals."""

import argparse
from collections import defaultdict
import csv
import gzip
import json
from math import fsum
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PLATFORMS = ("waymo", "drone", "quad")
COHORTS = ("validation", "extension")


def digest(path):
    from hashlib import file_digest

    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def load(cohort, name):
    with (OUT / f"{cohort}_{name}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def ledger(cohort):
    rows = []
    for p in PLATFORMS:
        with gzip.open(OUT / f"{cohort}_rows_{p}.csv.gz", "rt", newline="") as stream:
            after = list(csv.DictReader(stream))
            assert after and all(r["target"] == p for r in after)
            rows.extend(after)
    assert len({r["uid"] for r in rows}) == len(rows)
    return rows


def annotate(rows):
    for r in rows:
        r["before"] = float(r["iou_baseline"])
        r["after"] = float(r["iou_height_only"])
        r["upper"] = float(r["iou_oracle"])
        assert r["upper"] + 1.e-10 >= max(r["before"], r["after"])
        if r["eligible"] != "True":
            r["regime"] = "fallback"
            continue
        signed = float(r["top_error_old_m"])
        slack = float(r["slack_m"])
        if signed < -1.e-8:
            r["regime"] = "annotated_top_above_incumbent"
        elif signed > slack + 1.e-8:
            r["regime"] = "annotated_top_below_feasible_floor"
        else:
            r["regime"] = "annotated_top_inside_interval"


def verify_metrics(rows, table, pairs):
    groups = {p: [r for r in rows if r["target"] == p] for p in PLATFORMS}
    groups["pooled"] = groups["macro"] = rows

    def acc(result, p):
        if p != "macro":
            return 100 * fsum(result) / len(result)
        idx = [(t, v) for t, v in zip(rows, result, strict=True)]
        return 100 * fsum(fsum(v for t, v in idx if t["target"] == q) / len(groups[q])
                         for q in PLATFORMS) / len(PLATFORMS)

    for reported in table:
        p, arm = reported["platform"], reported["arm"]
        g = groups[p]
        vals = [float(r[f"iou_{arm}"]) for r in g]
        for c, hit, threshold in (("acc25", "hits25", .25), ("acc50", "hits50", .50)):
            passed = [v > threshold for v in vals]
            assert int(reported["n"]) == len(vals) and int(reported[hit]) == sum(passed)
            assert abs(float(reported[c]) - acc(passed, p)) < 1.e-8
        assert abs(float(reported["miou"]) - acc(vals, p)) < 1.e-8

    for reported in pairs:
        p, arm, base = [reported[k] for k in ("platform", "arm", "reference")]
        a = [float(r[f"iou_{arm}"]) for r in groups[p]]
        b = [float(r[f"iou_{base}"]) for r in groups[p]]
        for suffix, threshold in (("25", .25), ("50", .50)):
            changes = [int(v > threshold) - int(w > threshold) for v, w in zip(a, b, strict=True)]
            assert int(reported[f"rescue{suffix}"]) == changes.count(1)
            assert int(reported[f"harm{suffix}"]) == changes.count(-1)
            assert abs(float(reported[f"delta_acc{suffix}_pp"]) - acc(changes, p)) < 1.e-8
        changes = [v - w for v, w in zip(a, b, strict=True)]
        assert abs(float(reported["delta_miou_pp"]) - acc(changes, p)) < 1.e-8


def verify_rays(rows, table):
    geom_names = ("selection_changed", "audit_changed", "lost_measured")
    for reported in table:
        p, arm = reported["platform"], reported["arm"]
        group = [r for r in rows if r["eligible"] == "True" and (p == "pooled" or r["target"] == p)]
        changed = [r for r in group if r[f"{arm}_updated"] == "1"]
        assert len(group) == int(reported["eligible"]) and len(changed) == int(reported["updated"])
        for kind in geom_names:
            count = sum(int(r[f"{arm}_{kind}"]) for r in changed)
            assert count == int(reported[kind])
        for kind in ("selection", "audit"):
            assert sum(int(r[kind]) for r in changed) == int(reported[f"affected_{kind}"])
    main = [r for r in table if r["arm"] == "height_only" and r["platform"] == "pooled"][0]
    assert main["lost_measured"] == main["selection_changed"] == "0"


def diagnostics(cohort, rows):
    regimes, numbers = [], []
    for platform in (*PLATFORMS, "pooled"):
        subset = [r for r in rows if platform == "pooled" or platform == r["target"]]
        eligible = [r for r in subset if r["eligible"] == "True"]
        changed = [r for r in eligible if r["height_only_updated"] == "1"]
        names = ("point_boundary_raises", "ray_boundary_raises", "ray_veto", "prior_is_strict_bound")
        n = dict(cohort=cohort, platform=platform, n=len(subset), branch=len(eligible), changed=len(changed),
                 active_without_bound=sum(r["active"] == "True" and r["eligible"] == "False" for r in subset),
                 **{name: sum(r[name] == "True" for r in eligible) for name in names})
        n.update(median_trim_m=median(float(r["trim_m"]) for r in changed),
                 baseline_height_bias_m=median(float(r["height_bias_old_m"]) for r in eligible),
                 main_height_bias_m=median(float(r["height_bias_main_m"]) for r in eligible),
                 baseline_top_mae_m=fsum(abs(float(r["top_error_old_m"])) for r in eligible) / len(eligible),
                 main_top_mae_m=fsum(abs(float(r["top_error_main_m"])) for r in eligible) / len(eligible),
                 oracle_top_mae_m=fsum(abs(float(r["top_error_oracle_m"])) for r in eligible) / len(eligible))
        numbers.append(n)

        groups = defaultdict(list)
        for r in subset:
            groups[r["regime"]].append(r)
        for name, items in sorted(groups.items()):
            regimes.append(dict(cohort=cohort, platform=platform, annotated_top=name, n=len(items),
                original_hits=sum(r["before"] > .50 for r in items),
                main_hits=sum(r["after"] > .50 for r in items),
                oracle_hits=sum(r["upper"] > .50 for r in items),
                rescue50=sum(r["before"] <= .50 < r["after"] for r in items),
                harm50=sum(r["after"] <= .50 < r["before"] for r in items),
                oracle_rescue_remaining50=sum(r["after"] <= .50 < r["upper"] for r in items)))
    return numbers, regimes


def case_examples(cohort, rows):
    scenarios = {
        "rescued": lambda r: r["before"] <= .50 < r["after"],
        "harmed": lambda r: r["after"] <= .50 < r["before"],
        "ray_veto": lambda r: r["ray_veto"] == "True" and int(r["points_only_selection_changed"]) > 0,
        "oracle_still_fails": lambda r: .05 < r["upper"] <= .49 and r["height_only_updated"] == "1",
    }
    output = []
    for p in PLATFORMS:
        for name, select in scenarios.items():
            choices = [r for r in rows if r["target"] == p and r["eligible"] == "True" and select(r)]
            if not choices:
                continue
            # A deterministic middle-strength case, rather than a visually convenient extreme.
            choices.sort(key=lambda r: (abs(r["before"] - r["after"]), r["uid"]))
            row = choices[len(choices)//2]
            keys = ("uid", "target", "category", "before", "after", "upper", "slack_m", "trim_m",
                    "top_error_old_m", "selection", "audit", "points_only_selection_changed")
            output.append(dict(cohort=cohort, case=name, pool=len(choices), **{k: row[k] for k in keys}))
    return output


def save(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return digest(OUT / name)


def run(cohorts):
    recount, numbers, regimes, cases = {}, [], [], []
    for cohort in cohorts:
        seal = json.loads((OUT / f"{cohort}_evaluation.json").read_text())
        assert not seal["FINAL_access"]
        assert seal["prediction_seal"] == digest(OUT / f"{cohort}_prediction_seal.json")
        assert all(digest(OUT / file) == sha for file, sha in seal["artifact_sha256"].items())
        rows = ledger(cohort)
        assert len(rows) == sum(seal["counts"].values())
        metrics, comparisons, ray_counts = [load(cohort, file) for file in ("metrics", "pairs", "geometry")]
        verify_metrics(rows, metrics, comparisons)
        verify_rays(rows, ray_counts)
        annotate(rows)
        a, b = diagnostics(cohort, rows)
        numbers.extend(a)
        regimes.extend(b)
        cases.extend(case_examples(cohort, rows))
        recount[cohort] = dict(descriptions=len(rows), metrics_checked=len(metrics), comparisons_checked=len(comparisons),
                               ray_rows_checked=len(ray_counts), oracle_grid_checks=seal["oracle_grid_checks"])
    assert not (len(cohorts) == 2 and {r["uid"] for r in ledger("validation")} & {r["uid"] for r in ledger("extension")})
    files = {name: save(name, data) for name, data in (("mechanism.csv", numbers),
            ("oracle_regimes.csv", regimes), ("example_cases.csv", cases))}
    result = dict(cohorts=recount, output_sha256=files, code_sha256=digest(Path(__file__)), FINAL_access=False)
    (OUT / "independent_recount.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-only", action="store_true")
    run(COHORTS[:1] if parser.parse_args().validation_only else COHORTS)
