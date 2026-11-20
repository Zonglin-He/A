"""Post-seal attribution, including the best-prior positive and exact score-envelope test."""
import argparse
import csv
import gzip
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R183 = HERE.parent / "3eed_v2_ordinary_val_subject_ranking_20261119"
HISTORY = HERE.parent / "3eed_v1_shared_heading_overlap_20261120"
R181 = HERE.parent / "3eed_v2_readout_calibration_20261119"
sys.path[:0] = [str(R183), str(ROOT / "research/round6/source_validation_scoring")]
from scope183 import DIRECTIONS, atomic, digest, ordinary, read
from batch_yaw_iou import many_iou

OUT = HERE / "results"
PRIVATE = HERE / "run_PRIVATE"
CUTS = (.25, .50)
CLASSES = ("hit", "none_in_A256", "only_outside_four", "blocked", "reachable_miss")


def verified(path, signature):
    if digest(path) != signature:
        raise ValueError(f"Seal mismatch: {path.name}")


def identifier(panel, direction, index):
    tag = f"v1-bound-audit-20261120|{panel}|{direction}|{index}"
    return hashlib.sha256(tag.encode()).hexdigest()[:16]


def source_rows(panel):
    """Yield each row exactly once, in the ordering authenticated by the parent study."""
    if panel == "development":
        sys.path.insert(0, str(HISTORY))
        from assess_heading import inputs, roster_hash, sensor_truth
        data = HISTORY / "run_PRIVATE/pilot"
        seal, lock = read(data / "SEALED.json"), read(data / "LOCK.json")
        assert not seal["task_labels_opened"] and not seal["FINAL_access"]
        assert digest(data / "LOCK.json") == seal["lock"] and seal["n"] == 3072
        regions = R181 / "components_PRIVATE"
        active_seal = read(regions / "SEALED.json")
        for (s, t), rows in inputs("pilot").items():
            key = f"{s}_to_{t}"
            assert lock["roster"][key] == roster_hash(rows)
            check = active_seal["files"][key + ".npz"]
            verified(regions / (key + ".npz"), check if isinstance(check, str) else check["sha256"])
            with np.load(regions / (key + ".npz"), allow_pickle=False) as z:
                flags = z["has_reference"]
            for off in range(0, len(rows), 64):
                name = f"{key}/{off:05}.npz"
                verified(data / name, seal["files"][name]["sha256"])
                with np.load(data / name, allow_pickle=False) as z:
                    blocks = dict(z)
                for i, pool in enumerate(blocks["pool"]):
                    j = off+i
                    row = rows[j]
                    gt = sensor_truth(t, row["metapath"], row["caption"], row["suffix"])
                    index = blocks["ids"][i, 0]
                    s0 = blocks["logits"][i, 0]
                    p = s0.argmax()
                    assert index[p] == blocks["winners"][i, 0]
                    vals = many_iou(pool, gt)
                    yield key, j, blocks["priors"][i, 0], s0, bool(flags[j]), vals[index], vals.max(), {}
        return

    from score183 import truth
    from score_ablation import score_boxes
    cfg = read(HERE / "PROTOCOL.json")
    for path, signature in cfg["authority"].items():
        verified(ROOT / path, signature)
    pred = read(PRIVATE / "SEALED.json")
    check = read(R183 / "audited_iou_PRIVATE/SEALED.json")
    assert pred["n"] == check["n"] == 18620 and not pred["target_labels_used"]
    assert not pred["FINAL_access"] and not check["FINAL_access"]
    assert pred["inputs"]["protocol"] == digest(HERE / "PROTOCOL.json")
    all_rows = ordinary()
    for s, t in DIRECTIONS:
        key = f"{s}_to_{t}"
        folder = PRIVATE / key
        verified(folder / "SEALED.json", pred["directions"][key])
        sub = read(folder / "SEALED.json")
        assert not sub["labels_used"] and sub["inputs"] == pred["inputs"]
        old_path = R183 / "audited_iou_PRIVATE" / (key+".npz")
        verified(old_path, check["files"][old_path.name]["sha256"])
        with np.load(old_path, allow_pickle=False) as archive:
            old = dict(archive)
        separate_checks = {"all_256_rows": 0, "choice_and_best_rows": 0, "max_error": 0.}
        iou_rows = []
        for part, signature in sorted(sub["files"].items()):
            verified(folder / part, signature)
            with np.load(folder / part, allow_pickle=False) as z:
                b = dict(z)
            for i, index in enumerate(b["ordinals"]):
                j = int(index)
                row = all_rows[t][j]
                gt = truth(row["metapath"], row["object_index"], row["caption"], t, row["suffix"])
                iou = many_iou(b["pool"][i], gt)
                quartet = iou[b["slots"][i].astype(int)]
                winner = b["choices"][i, 0]
                if j % 64 == 0:
                    indices = np.arange(256)
                    separate_checks["all_256_rows"] += 1
                else:
                    indices = np.r_[b["slots"][i].astype(int), int(iou.argmax())]
                verified_values, err = score_boxes(b["pool"][i, indices], gt)
                err = max(err, float(np.max(abs(verified_values-iou[indices]))))
                separate_checks["choice_and_best_rows"] += 1
                separate_checks["max_error"] = max(err, separate_checks["max_error"])
                assert err < 1e-7
                assert int(winner) == b["final"][i].argmax() == old["chosen"][j, 0]
                assert bool(b["active"][i]) == old["active"][j]
                assert abs(quartet[winner] - old["ious"][j, 0]) < 1e-8
                assert abs(quartet.max() - old["four_best"][j]) < 1e-8
                assert iou.max() >= old["k16_best"][j] - 1e-8
                for cut in CUTS:
                    np.testing.assert_array_equal(verified_values > cut, iou[indices] > cut)
                    assert (quartet[winner] > cut) == (old["ious"][j, 0] > cut)
                    assert (quartet.max() > cut) == (old["four_best"][j] > cut)
                extra = {k:b[k][i] for k in ("unary", "rel", "support", "same_selected_box", "explicit_lr")}
                iou_rows.append(iou)
                yield key, j, b["prior"][i], b["final"][i], bool(b["active"][i]), quartet, iou.max(), extra
        path = PRIVATE / f"scored_{key}.npz"
        np.savez_compressed(path, pool_iou=np.stack(iou_rows))
        atomic(path.with_suffix(".json"), dict(**separate_checks, file_sha256=digest(path),
               prediction_seal=digest(PRIVATE / "SEALED.json")))


def can_win_by_scaling(a, delta, good):
    """Is any good score strictly above every bad score anywhere on scale >= 1?"""
    for g in np.flatnonzero(good):
        lo, hi = 1., np.inf
        for b in np.flatnonzero(~good):
            slope = delta[g] - delta[b]
            gap = a[b] - a[g]
            if slope > 0:
                lo = max(lo, gap / slope)
            elif slope < 0:
                hi = min(hi, gap / slope)
            elif gap >= 0:
                break
        else:
            if lo < hi:
                return True
    return False


def classify(a, final, active, iou, pool_best, cut, extra):
    a, final = np.asarray(a, float), np.asarray(final, float)
    assert a.shape == final.shape == (4,) and a[0] == a.max() == 0.
    winner, prior = int(final.argmax()), int(a.argmax())
    good = iou > cut
    bound = 2.5 if active else 1.5
    delta = final - a
    assert np.ptp(delta) <= bound + 4e-6
    # The limit of the same frozen-score ray at lambda -> infinity. Preserve
    # prior tie-breaking for equal slopes, and native index order for exact ties.
    limiting_winner = int(np.lexsort((np.arange(4), -a, -delta))[0])
    candidate = int(np.argmax(np.where(good, a, -np.inf))) if good.any() else None
    gap = a.max() - a[candidate] if candidate is not None else None
    fall = not bool(good[winner])
    category = ("hit" if not fall else "none_in_A256" if pool_best <= cut
                else "only_outside_four" if candidate is None else "blocked" if gap >= bound
                else "reachable_miss")
    assert not (gap is not None and gap >= bound and not fall)
    out = dict(cls=category, active=int(active), original_hit=int(good[prior]),
               final_hit=int(good[winner]), limiting_hit=int(good[limiting_winner]), switched=int(prior != winner),
               count_good=int(good.sum()), gap=gap, cap=bound, span=float(np.ptp(delta)),
               any_good_beyond_cap=int(np.any((a.max()-a >= bound) & good)),
               can_scale=None, prior_rival_beaten_in_residual=None, unary_diff=None,
               rel_diff=None, mass_diff=None, offset_diff=None, corr_diff=None,
               explicit_lr=None, same_selected_box=None)
    if fall and good.any():
        out["can_scale"] = int(can_win_by_scaling(a, delta, good))
        out["prior_rival_beaten_in_residual"] = int(delta[good].max() > delta[prior]) if not good[prior] else None
    if extra:
        out["explicit_lr"] = int(extra["explicit_lr"])
        out["same_selected_box"] = int(extra["same_selected_box"])
        if fall and good.any():
            # Choose the passing candidate with the largest initial score, fixed before inspecting moments.
            support = extra["support"][1, :4]
            assert support.shape == (4,11)
            for name, coordinate in (("mass_diff", 9), ("offset_diff", 10), ("corr_diff", 6)):
                out[name] = float(support[candidate, coordinate] - support[winner, coordinate])
            out["unary_diff"] = float(extra["unary"][candidate] - extra["unary"][winner])
            out["rel_diff"] = float(extra["rel"][candidate] - extra["rel"][winner])
    return out


def ratio(a, b):
    return 100*a/b if b else None


def med(values):
    return float(np.median(values)) if len(values) else None


def aggregate(rows):
    n = len(rows)
    counts = {k:sum(r["cls"]==k for r in rows) for k in CLASSES}
    assert sum(counts.values()) == n
    lost = [r for r in rows if r["cls"] in ("blocked","reachable_miss")]
    p_lost = [r for r in lost if not r["original_hit"]]
    relevant = [r for r in rows if r["gap"] is not None]
    falseflags = sum(r["cls"] == "reachable_miss" and r["any_good_beyond_cap"] for r in rows)
    errors = n - counts["hit"]
    out = dict(n=n, **counts, errors=errors, recoverable_four_misses=len(lost),
               blocked_pct_all=ratio(counts["blocked"],n), blocked_pct_errors=ratio(counts["blocked"],errors),
               blocked_pct_recoverable=ratio(counts["blocked"],len(lost)), any_rule_false_positives=falseflags,
               exactly_on_cap=sum(int(r["gap"]==r["cap"]) for r in relevant),
               within_two_micro_units_of_cap=sum(int(abs(r["gap"]-r["cap"])<=2e-6) for r in relevant),
               initial_hits=sum(r["original_hit"] for r in rows),
               decoder_rescues=sum(r["final_hit"] and not r["original_hit"] for r in rows),
               decoder_harms=sum(r["original_hit"] and not r["final_hit"] for r in rows),
               limiting_hits=sum(r["limiting_hit"] for r in rows),
               limiting_rescues_vs_final=sum(r["limiting_hit"] and not r["final_hit"] for r in rows),
               limiting_harms_vs_final=sum(r["final_hit"] and not r["limiting_hit"] for r in rows),
               changed_slot=sum(r["switched"] for r in rows),
               oracle_scaling_recovers=sum(r["can_scale"] for r in lost),
               blocked_oracle_scaling_recovers=sum(r["can_scale"] for r in lost if r["cls"]=="blocked"),
               prior_and_decoder_wrong_with_good=len(p_lost),
               all_good_residuals_no_higher_than_prior_winner=sum(not r["prior_rival_beaten_in_residual"] for r in p_lost),
               median_gap_recoverable=med([r["gap"] for r in lost]),
               median_residual_span=med([r["span"] for r in rows]),
               lr=sum(r["explicit_lr"]==1 for r in rows) if any(r["explicit_lr"] is not None for r in rows) else None,
               lr_exact_same_box=sum(r["explicit_lr"]==1 and r["same_selected_box"]==1 for r in rows)
                   if any(r["explicit_lr"] is not None for r in rows) else None)
    return out


def write_csv(path, rows):
    if not rows:
        raise ValueError("empty output")
    opener = gzip.open if path.suffix==".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(panel):
    groups = defaultdict(list)
    case_files = defaultdict(list)
    example_candidates = defaultdict(list)
    for direction, idx, a, score, active, iou, pool_best, extra in source_rows(panel):
        assert np.isfinite(iou).all() and np.isfinite(pool_best)
        key = identifier(panel, direction, idx)
        case = dict(row=key, winner_iou=round(float(iou[np.argmax(score)]), 9),
                    four_best=round(float(iou.max()), 9), pool_best=round(float(pool_best), 9),
                    ref=int(active))
        for cut in CUTS:
            val = classify(a, score, active, iou, pool_best, cut, extra)
            groups[direction, cut].append(val)
            prefix = str(int(cut*100))
            case["class"+prefix] = val["cls"]
            case["min_gap"+prefix] = round(val["gap"], 8) if val["gap"] is not None else None
            case["scale_strict"+prefix] = val["can_scale"]
            case["prior_hit"+prefix] = val["original_hit"]
            case["limiting_hit"+prefix] = val["limiting_hit"]
            if panel=="ordinary" and cut==.50 and val["cls"] in ("blocked","reachable_miss") and not val["original_hit"]:
                g = int(np.argmax(np.where(iou>cut, a, -np.inf)))
                p = int(np.argmax(a))
                change = float((float(score[g])-a[g])-(float(score[p])-a[p]))
                label = val["cls"] if val["cls"]=="blocked" else (
                    "reachable_favors_good" if change>0 else "reachable_favors_prior_or_ties")
                example_candidates[label].append(dict(type=label, row=key, direction=direction,
                    has_reference=int(active), cap=val["cap"], gap=float(a[p]-a[g]),
                    good_IoU=float(iou[g]), initial_winner_IoU=float(iou[p]), final_winner_IoU=float(iou[np.argmax(score)]),
                    good_minus_initial_unary=float(extra["unary"][g]-extra["unary"][p]),
                    good_minus_initial_relation=float(extra["rel"][g]-extra["rel"][p]),
                    good_minus_initial_residual=change, good_minus_initial_final=float(score[g]-score[p])))
        case_files[direction].append(case)
    tables, context, support = [], [], []
    names = [f"{s}_to_{t}" for s,t in DIRECTIONS]
    for direction in [*names, "all_records"]:
        for cut in CUTS:
            group = groups[direction, cut] if direction in names else [r for d in names for r in groups[d,cut]]
            row = dict(direction=direction, threshold=cut, **aggregate(group))
            tables.append(row)
            if direction=="all_records":
                for active in (False,True):
                    subset = [r for r in group if r["active"]==active]
                    context.append(dict(has_reference=active, threshold=cut, **aggregate(subset)))
            if panel=="ordinary":
                for cls in ("blocked","reachable_miss"):
                    subset=[r for r in group if r["cls"]==cls]
                    for name in ("unary_diff","rel_diff","mass_diff","offset_diff","corr_diff"):
                        values = np.array([r[name] for r in subset], dtype=float)
                        support.append(dict(direction=direction,threshold=cut,category=cls,
                                passing_minus_selected=name,n=len(values),median=med(values),
                                positive=int((values>0).sum()),negative=int((values<0).sum()),
                                tied=int((values==0).sum())))
    for direction,cases in case_files.items():
        write_csv(OUT / f"{panel}_{direction}_rows.csv.gz", cases)
    write_csv(OUT/f"{panel}_attribution.csv",tables)
    write_csv(OUT/f"{panel}_reference.csv",context)
    if support:
        write_csv(OUT/f"{panel}_support.csv",support)
    if example_candidates:
        representatives=[]
        for values in example_candidates.values():
            values.sort(key=lambda r:(r["gap"],r["row"]))
            representatives.append(values[(len(values)-1)//2])
        write_csv(OUT/"ordinary_representative_pairs.csv",representatives)
    macro = []
    for cut in CUTS:
        rows = [r for r in tables if r["direction"] in names and r["threshold"]==cut]
        macro.append(dict(threshold=cut, **{k:float(np.mean([100*r[k]/r["n"] for r in rows])) for k in CLASSES},
                          initial_acc=float(np.mean([100*r["initial_hits"]/r["n"] for r in rows])),
                          infinite_scale_acc=float(np.mean([100*r["limiting_hits"]/r["n"] for r in rows]))))
    atomic(OUT/f"{panel}_macro.json",macro)
    for r in tables[-2:]:
        print(json.dumps(r, ensure_ascii=False))
    print("Six directions", macro)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", choices=("ordinary","development"))
    summarize(parser.parse_args().panel)
