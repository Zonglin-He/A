"""Independently recount sealed rows and enumerate every possible positive-scale ranking."""
from collections import Counter
import csv
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
WORK = HERE / "run_PRIVATE"
OLD = HERE.parent / "3eed_v2_ordinary_val_subject_ranking_20261119"
STATES = ("hit", "none_in_A256", "only_outside_four", "blocked", "reachable_miss")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def csv_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as file:
        return list(csv.DictReader(file))


def upper_envelope(prior, scores, accepted):
    # Independent from pairwise feasibility. Rankings change only at line intersections.
    slope = np.asarray(scores, dtype=np.float64) - prior
    knots = {1.}
    for i, j in combinations(range(4), 2):
        if slope[i] != slope[j]:
            x = (prior[i] - prior[j]) / (slope[j] - slope[i])
            if x > 1:
                knots.add(x)
    stops = sorted(knots)
    tests = [1.] + [(a+b)/2 for a,b in zip(stops,stops[1:])] + [2*stops[-1]+1]
    for x in tests:
        y = prior + x*slope
        if max(y[accepted]) > max(y[~accepted]):
            return True
    return False


def duplicate_controls():
    """Compare objects with fixed cluster probability or fixed individual query logits."""
    sys.path.insert(0, str(HERE.parents[1]/"research/round158"))
    from grounding158 import select_grounding_box
    cells = []
    for m in (1,2,4,8):
        boxes = np.ones((256,6))
        boxes[:,:3] = 0
        boxes[m,0] = 20
        boxes[m+1:,0] = 50+3*np.arange(255-m)
        fixed_mass = np.full(256,.25/(255-m))
        fixed_mass[:m], fixed_mass[m] = .5/m, .25
        logits = np.full(256,-10.)
        logits[:m], logits[m] = np.log(2.), 0.
        fixed_query = np.exp(logits-logits.max())
        fixed_query /= fixed_query.sum()
        mass_scores = select_grounding_box(boxes, fixed_mass)[1]
        clone_scores = select_grounding_box(boxes, fixed_query)[1]
        values = (mass_scores[0]/mass_scores[m], clone_scores[0]/clone_scores[m])
        np.testing.assert_allclose(values, (2/m,2), rtol=1e-12,atol=1e-12)
        cells.append((m,*values))
    with (OUT/"duplicate_control.csv").open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(("identical_proposals_A","utility_ratio_fixed_object_mass","utility_ratio_fixed_query_logits"))
        writer.writerows(cells)


def by_category(panel):
    result = {}
    tables = csv_rows(OUT/f"{panel}_attribution.csv")
    index = {(r["direction"],float(r["threshold"])):r for r in tables}
    seen = set()
    for path in sorted(OUT.glob(f"{panel}_*_rows.csv.gz")):
        name = path.name.removeprefix(panel+"_").removesuffix("_rows.csv.gz")
        rows = csv_rows(path)
        assert not seen.intersection(r["row"] for r in rows)
        seen.update(r["row"] for r in rows)
        for cut in (25,50):
            key = str(cut)
            tally = Counter(r["class"+key] for r in rows)
            assert sum(tally.values()) == len(rows)
            ref = index[name, cut/100]
            assert all(tally[s] == int(ref[s]) for s in STATES)
            assert sum(r["prior_hit"+key]=="1" for r in rows)==int(ref["initial_hits"])
            assert sum(r["limiting_hit"+key]=="1" for r in rows)==int(ref["limiting_hits"])
            assert sum(r["scale_strict"+key]=="1" for r in rows)==int(ref["oracle_scaling_recovers"])
        result[name] = rows
    for cut in (25,50):
        ref = index["all_records",cut/100]
        allrows = [r for d in result.values() for r in d]
        for state in STATES:
            assert sum(r["class"+str(cut)]==state for r in allrows) == int(ref[state])
        holes = int(ref["recoverable_four_misses"])
        n, error, forced = int(ref["n"]), int(ref["errors"]), int(ref["blocked"])
        assert n==len(allrows) and error==n-int(ref["hit"])
        for denominator, field in ((n,"all"),(error,"errors"),(holes,"recoverable")):
            assert abs(forced/denominator*100 - float(ref["blocked_pct_"+field])) < 1e-10
    return result


def run():
    # Check multiple positives, exact ties, and positive slopes that can beat
    # the old winner but cannot overtake another competitor at the same scale.
    from attribute import can_win_by_scaling, classify
    duplicate_controls()
    prior = np.array([0.,-.2,-.7,-4.])
    sim = classify(prior, prior, True, np.array([0.,.8,0.,.8]), .8, .5, {})
    assert sim["cls"]=="reachable_miss" and sim["any_good_beyond_cap"]==1
    tie = classify(np.array([0.,-1.5,-2.,-3.]), np.array([0.,-1.5,-2.,-3.]),
                   False, np.array([0.,.8,0.,0.]), .8, .5, {})
    assert tie["cls"]=="blocked"
    rng = np.random.default_rng(174)
    examples = 0
    for _ in range(600):
        a = rng.integers(-20, 1, size=4).astype(float)/4
        r = rng.integers(-5, 6, size=4).astype(float)/8
        good = rng.integers(0,2,4).astype(bool)
        if good.all() or not good.any():
            continue
        assert can_win_by_scaling(a,r,good)==upper_envelope(a,a+r,good)
        examples += 1
    development = by_category("development")
    ordinary = by_category("ordinary")
    assert sum(map(len,development.values()))==3072 and sum(map(len,ordinary.values()))==18620
    assert len(development) == len(ordinary) == 6
    parent = read(OLD/"predictions_PRIVATE/SEALED.json")
    seal = read(WORK/"SEALED.json")
    assert seal["original_four_and_winners_checked"]==18620 and not seal["FINAL_access"]
    bound_checks = Counter()
    max_score_error = 0.
    max_iou_error = 0.
    for direction, public in ordinary.items():
        path = WORK/f"scored_{direction}.npz"
        audit = read(path.with_suffix(".json"))
        assert sha(path) == audit["file_sha256"] and audit["prediction_seal"] == sha(WORK/"SEALED.json")
        with np.load(path,allow_pickle=False) as verified_iou:
            iou = verified_iou["pool_iou"]
        max_iou_error = max(max_iou_error,audit["max_error"])
        bound_checks["rows_checked_by_second_geometric_scorer"] += audit["choice_and_best_rows"]
        bound_checks["second_scorer_all_256"] += audit["all_256_rows"]
        folder = WORK/direction
        assert sha(folder/"SEALED.json")==seal["directions"][direction]
        proofs=read(folder/"SEALED.json")["files"]
        for name, checksum in proofs.items():
            assert sha(folder/name)==checksum
            native=OLD/"predictions_PRIVATE"/direction/name
            assert sha(native)==parent["directions"][direction]["files"][name]["sha256"]
            with np.load(native,allow_pickle=False) as old, np.load(folder/name,allow_pickle=False) as data:
                z=dict(data)
                np.testing.assert_array_equal(z["choices"][:,0],old["choices"][:,0])
                np.testing.assert_allclose(z["pool"][np.arange(len(z["slots"]))[:,None],z["slots"]],old["four"],rtol=0.,atol=1e-10)
            assert z["support"].shape == (len(z["ordinals"]),2,8,11) and np.isfinite(z["support"]).all()
            rec = z["prior"].astype(float) + z["unary"] + z["rel"]
            max_score_error=max(max_score_error, float(np.max(np.abs(rec-z["final"]))))
            for j,ord_ in enumerate(z["ordinals"]):
                r=public[int(ord_)]
                a=z["prior"][j].astype(float)
                s=z["final"][j].astype(float)
                limit=1.5+int(z["active"][j])
                ious=iou[ord_]
                four=ious[z["slots"][j]]
                assert abs(float(r["pool_best"])-ious.max())<=5.1e-10
                assert abs(float(r["winner_iou"])-four[np.argmax(s)])<=5.1e-10
                for cut in (25,50):
                    passed=(four>cut/100)
                    status=r["class"+str(cut)]
                    limit_pick=max(range(4),key=lambda k:(s[k]-a[k], a[k], -k))
                    assert int(r["limiting_hit"+str(cut)])==int(passed[limit_pick])
                    if passed[np.argmax(s)]:
                        assert status=="hit"
                    elif not passed.any():
                        assert status==("only_outside_four" if (ious>cut/100).any() else "none_in_A256")
                    else:
                        # A good slot must beat every failing prior, simultaneously.
                        locked=all(any((a[b]-a[g]) >= limit for b in np.flatnonzero(~passed))
                                   for g in np.flatnonzero(passed))
                        assert status==("blocked" if locked else "reachable_miss")
                        assert bool(int(r["scale_strict"+str(cut)]))==upper_envelope(a,s,passed)
                        bound_checks["good_slot_errors_checked_at_both_thresholds"]+=1
                bound_checks["original_v1_scores_checked"]+=4
    assert max_score_error < 3e-6
    assert bound_checks["original_v1_scores_checked"]==74480
    assert bound_checks["rows_checked_by_second_geometric_scorer"]==18620
    audit=dict(ordinary_records=18620,development_records=3072,
               core_sha256_verified=seal["inputs"],original_winners_exact=18620,
               geometry_threshold_disagreements=0,pool_boxes_scored=18620*256,
               max_absolute_score_reconstruction_error=max_score_error,
               max_checked_iou_difference=max_iou_error, independent_counts=dict(bound_checks),
               mathematical_margin_and_scaling_examples=examples+2,
               target_parameter_updates=0,FINAL_access=False)
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit))


if __name__=="__main__":
    run()
