"""Post-seal evaluation of A, D, hybrid H, and fixed held-out return selectors."""
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAST = ROOT / "research/round156"
LOCAL = ROOT / "crossplatform/round157"
sys.path.insert(0, str(PAST))
from score156 import (INPUT, META, aggregates, digest, ground_truth, many_iou, oriented_iou,
                      read, read_csv, require, rotation, save, table)

GROUPS = ("ray", "surface", "combined")
POLICIES = (*GROUPS, *("shuffle_" + g for g in GROUPS))
BASES = ("Frozen_A", "A", "D", "H")
ORACLES = ("oracle_AH", "oracle_AD")
NAMES = (*BASES, *POLICIES, *ORACLES)
N = 1798
EPS = 1e-12
OLD_ROWS_SHA = "66ea7caa0392a38e1d59f847b93892b1597c64d5e3a64ebe8be2a9739f5063bf"
OLD_PRED_SHA = "1513f9addb268ebaa505d43337ae8a784905288cb094614a80e45991123795e9"
REGISTRY_SHA = "38035be8022d15f4c6867f48a44b60e1eee6288f29b43be17b743559bca52e45"
DIAG_NUM = ("n_raw","n_unique_sample_indices","n_holdout_indices","n_holdout","alias_exclusions","n_strict_unique","n_origin_unresolved",
            "n_cluster_points","n_cluster_voxels","n_winner_shared_voxels","n_shared_voxels")
DIAG_SCORES = ("ray_A", "ray_H", "surface_A", "surface_H", "miss_A", "miss_H", "miss_raw_A", "miss_raw_H")
DIAG_BOOL = ("strict_e2e", "origin_ok", *("eligible_" + g for g in GROUPS), *("choose_" + g for g in POLICIES))
DIAG_TEXT = ("geometry_reason","origin_reason","ray_reason","registration_exposure")


def verified_inputs():
    artifact = LOCAL / "PREDICTIONS_PRIVATE.npz"
    seal = read(LOCAL / "PREDICTION_SEAL.json")
    require(seal["n"] == N and seal["prediction_sha256"] == digest(artifact), "prediction seal mismatch")
    require(seal["design_sha256"] == digest(HERE / "experiment/DESIGN_LOCK.json"),
            "design lock changed")
    require(seal["status"] == "PASS" and seal["code_lock_sha256"] == digest(HERE / "experiment/CODE_LOCK.json"),
            "prediction code seal mismatch")
    for name, value in read(HERE / "experiment/CODE_LOCK.json")["files"].items():
        require(digest(HERE / name) == value, "code differs from lock")
    require(seal["GT_read"] is False and seal["FINAL_access"] is False, "invalid prediction boundary")
    old_pair = ROOT / "crossplatform/round156/PAIR_PRIVATE.npz"
    old_results = PAST / "results/pair_rows.csv"
    require(digest(old_results) == OLD_ROWS_SHA and digest(old_pair) == OLD_PRED_SHA
            and digest(INPUT) == REGISTRY_SHA, "parent inputs changed")
    rows = read_csv(old_results)
    reg = read(INPUT)
    n_axis_fields = (DIAG_NUM + DIAG_SCORES + DIAG_BOOL + DIAG_TEXT +
                     ("ordinal", "source", "target", "has_B"))
    with np.load(old_pair, allow_pickle=False) as f:
        anchors = f["boxes"][:, :, 1].copy()
    with np.load(artifact, allow_pickle=False) as f:
        require(all(f[k].shape == (N,) for k in n_axis_fields), "selector metadata schema")
        diags = {k: f[k].copy() for k in n_axis_fields}
        geom = np.stack([f[k] for k in ("A", "D", "H")], axis=1).astype(float)
    expected_h = anchors[:, 0].copy()
    expected_h[:, [0, 1, 6]] = anchors[:, 1, [0, 1, 6]]
    require(geom.shape == (N, 3, 7) and np.isfinite(geom).all()
            and np.array_equal(geom[:, :2], anchors) and np.array_equal(geom[:, 2], expected_h),
            "hybrid or original A/D endpoints changed")
    require(len(reg) == len(rows) == N and np.array_equal(diags["ordinal"], np.arange(1, N+1)), "ordinal")
    for key in DIAG_BOOL:
        require(diags[key].dtype == np.bool_, "nonboolean flag " + key)
    for key in DIAG_NUM:
        require(np.issubdtype(diags[key].dtype, np.integer) and np.all(diags[key] >= 0), "invalid count " + key)
    require(np.array_equal(diags["n_holdout_indices"]-diags["alias_exclusions"],diags["n_strict_unique"]) and
            np.array_equal(diags["n_strict_unique"]-diags["n_origin_unresolved"],diags["n_holdout"]),
            "held-out counts or per-beam exclusions mismatch")
    for key in DIAG_SCORES:
        require(not np.isinf(diags[key]).any(), "infinite physical score")
    for key in ("miss_A", "miss_H", "miss_raw_A", "miss_raw_H"):
        finite = np.isfinite(diags[key])
        require(np.all((diags[key][finite] >= 0) & (diags[key][finite] <= 1)), "ray miss ratio")
    for i, (r,e) in enumerate(zip(rows, reg)):
        require(e["cohort"] == "EXT" and not e["protected_alias_fallback"] and
                (r["source"],r["target"]) == (e["source"],e["target"]) ==
                (diags["source"][i],diags["target"][i]) and int(r["ordinal"]) == i+1 and
                float(r["sampling_weight"]) == e["sampling_weight"] and
                (r["has_B"] == "True") == diags["has_B"][i], "cohort or metadata mismatch")
    direction = np.char.add(np.char.add(diags["source"], "_"), diags["target"])
    for mode in GROUPS:
        eligible = diags["eligible_" + mode]
        chosen = diags["choose_" + mode]
        wrong = diags["choose_shuffle_" + mode]
        require(not np.any((chosen | wrong) & ~eligible), "choice outside eligibility")
        for d, has_b, origin_ok in set(zip(direction, diags["has_B"], diags["origin_ok"])):
            group = (direction == d) & (diags["has_B"] == has_b) & (diags["origin_ok"] == origin_ok)
            require(chosen[group].sum() == wrong[group].sum(), "switch budget mismatch " + d + mode)
    require(np.all(~diags["eligible_ray"] | diags["origin_ok"]), "ray choice lacking origin")
    for mode in ("ray", "surface"):
        flag = diags["eligible_" + mode]
        low, high = (diags[mode+"_A"], diags[mode+"_H"])
        require(np.all(np.isfinite(low[flag])) and np.all(np.isfinite(high[flag])), "missing eligible signal")
        require(np.array_equal(diags["choose_" + mode], flag & (high+1e-12 < low)), "score/decision mismatch")
    require(np.array_equal(diags["choose_combined"], np.where(diags["origin_ok"],diags["choose_ray"],
            diags["choose_surface"])), "routing error")
    geom[..., 3:6] = np.maximum(geom[..., 3:6], 1e-6)
    return geom, diags, rows, reg, seal


def evaluate(geom, diags, previous, registry):
    rows = []
    cross_error = parent_error = 0.0
    independent_calls = 0
    for i, (r, e) in enumerate(zip(previous, registry)):
        frame = e["row"]
        gt = ground_truth(e["target"], frame["metapath"], frame["object_index"], frame["caption"])
        scores = many_iou(geom[i], gt)
        grot = rotation(gt[6])
        unique_box = {}
        for box, score in zip(geom[i], scores):
            key = tuple(box)
            if key not in unique_box:
                unique_box[key] = oriented_iou(box[:3],box[3:6],rotation(box[6]),gt[:3],gt[3:6],grot)
                independent_calls += 1
            check = unique_box[key]
            cross_error = max(cross_error, abs(check-score))
            require(all((v < score) == (v < check) for v in (.25, .5)), "OBB threshold disagreement")
        parent_error = max(parent_error, abs(scores[0]-float(r["A_pca"])), abs(scores[1]-float(r["D_pca"])))
        row = {k: r[k] for k in META}
        for k in ("ordinal", "scene_cluster", "frame_cluster", "point_content_cluster"):
            row[k] = int(row[k])
        row.update(sampling_weight=float(row["sampling_weight"]), has_B=row["has_B"] == "True",
                   macro_weight=float(r["macro_weight"]), Frozen_A=float(r["frozen_A"]))
        row.update(zip(("A", "D", "H"), map(float,scores)))
        for k in DIAG_BOOL + DIAG_NUM + DIAG_TEXT:
            row[k] = diags[k][i].item()
        for k in DIAG_SCORES:
            v = float(diags[k][i])
            row[k] = v if np.isfinite(v) else ""
        for name in POLICIES:
            row[name] = row["H"] if row["choose_" + name] else row["A"]
        row["oracle_AH"] = max(row["A"], row["H"])
        row["oracle_AD"] = max(row["A"], row["D"])
        rows.append(row)
    require(cross_error <= 1e-7 and parent_error <= 1e-8, "OBB agreement or historical endpoint failure")
    return rows, dict(max_independent_OBB_error=cross_error, max_R156_IoU_error=parent_error,
                      endpoint_IoU_checks=3*N, distinct_general_OBB_calls=independent_calls)


def mean_finite(values):
    finite = [v for v in values if v != ""]
    return float(np.mean(finite)) if finite else ""


def gain_record(ref, output, weight, selector, weights_global):
    delta = output - ref
    improvement = np.maximum(delta, 0)
    harm = np.minimum(delta, 0)
    local = weight / weight.sum()
    return dict(n=len(ref), selected_H=int(np.sum(selector)) if selector is not None else "",
                local_gain_pp=float(100 * (local @ improvement)),
                local_harm_pp=float(100 * (local @ harm)),
                local_net_pp=float(100 * (local @ delta)),
                global_gain_pp=float(100 * (weights_global @ improvement)),
                global_harm_pp=float(100 * (weights_global @ harm)),
                global_net_pp=float(100 * (weights_global @ delta)),
                better=int(np.sum(delta > EPS)), worse=int(np.sum(delta < -EPS)),
                same=int(np.sum(np.abs(delta) <= EPS)),
                rescued25=int(np.sum((output > .25) & (ref <= .25))), harmed25=int(np.sum((output <= .25) & (ref > .25))),
                rescued50=int(np.sum((output > .50) & (ref <= .50))), harmed50=int(np.sum((output <= .50) & (ref > .50))))


def breakdown(rows, chosen_cases):
    columns = {k: np.array([r[k] for r in rows]) for k in tuple(NAMES) + DIAG_BOOL + DIAG_TEXT}
    w = np.array([r["macro_weight"] for r in rows])
    require(abs(w.sum()-1) <= 1e-12, "survey weighting")
    all_mask = np.ones(len(rows), bool)
    d_diff = columns["D"] - columns["A"]
    h_diff = columns["H"] - columns["A"]
    direction = np.array([r["source"] + "_to_" + r["target"] for r in rows])
    panels = {"all":all_mask, "D_better":d_diff > EPS, "D_worse":d_diff < -EPS, "D_equal":abs(d_diff) <= EPS,
              "H_better":h_diff > EPS, "H_worse":h_diff < -EPS, "H_equal":abs(h_diff) <= EPS,
              "prior15":np.array([r["ordinal"] in chosen_cases for r in rows]),
              "has_B":np.array([r["has_B"] for r in rows]), "origin_known":columns["origin_ok"],
              "origin_unknown":~columns["origin_ok"]}
    panels["no_B"] = ~panels["has_B"]
    panels["strict_e2e"] = columns["strict_e2e"]
    panels["indirect_registration"] = ~columns["strict_e2e"]
    for mode in GROUPS:
        flag = columns["eligible_" + mode]
        panels["eligible_" + mode] = flag
        panels["ineligible_" + mode] = ~flag
        panels["accepted_" + mode] = columns["choose_" + mode]
    for d in sorted(set(direction)):
        panels[d] = direction == d
    for k in DIAG_TEXT:
        for reason in sorted(set(columns[k])):
            panels[k + "_" + reason] = columns[k] == reason
    out = []
    for group,mask in panels.items():
        if not mask.any():
            continue
        for arm in ("D", "H", *POLICIES, *ORACLES):
            switch = columns["choose_" + arm][mask] if arm in POLICIES else (
                        np.ones(mask.sum(),bool) if arm == "H" else None)
            out.append(dict(scope=group, arm=arm, control="A", full_6way_weight=float(w[mask].sum()),
                         **gain_record(columns["A"][mask],columns[arm][mask],w[mask],switch,w[mask])))
    require(np.sum(panels["D_worse"]) == 513 and np.sum(panels["D_better"]) == 471, "prior subset failure")
    return out


def signal_evidence(rows):
    """Break down gross benefit, opportunity left by the gate, and switch-budget null."""
    bins = {}
    for mode in GROUPS:
        winners = []
        losers = []
        eligible_mass = 0.
        eligible_negative = 0.
        null_pp = 0.
        strata = {}
        changed = []
        for r in rows:
            if not r["eligible_" + mode]:
                continue
            gain = 100 * r["macro_weight"] * (r["H"] - r["A"])
            eligible_mass += max(0., gain)
            eligible_negative += min(0., gain)
            st = (r["source"],r["target"],r["has_B"],r["origin_ok"])
            strata.setdefault(st,[]).append(r)
            if r["choose_" + mode] and abs(r["H"]-r["A"]) > EPS:
                changed.append(r)
                rec = dict(ordinal=r["ordinal"], source=r["source"],target=r["target"],
                           weighting_contribution_pp=gain, A_IoU=r["A"], H_IoU=r["H"])
                (winners if gain>0 else losers).append(rec)
        for rr in strata.values():
            switch_rate = sum(r["choose_" + mode] for r in rr) / len(rr)
            null_pp += switch_rate * sum(100 * r["macro_weight"] * (r["H"]-r["A"]) for r in rr)
        pos = sum(r["weighting_contribution_pp"] for r in winners)
        neg = sum(r["weighting_contribution_pp"] for r in losers)
        bins[mode] = dict(evaluable=sum(map(len,strata.values())),selected=sum(r["choose_"+mode] for r in rows),
            improvements=len(winners), harms=len(losers), gain_pp=pos, harm_pp=neg, net_pp=pos+neg,
            eligible_oracle_gain_pp=eligible_mass, eligible_always_H_harm_pp=eligible_negative,
            positive_opportunity_capture=pos/eligible_mass if eligible_mass>0 else None,
            switching_on_harms_fraction=neg/eligible_negative if eligible_negative<0 else None,
            expected_random_net_pp_same_counts_by_stratum=null_pp,
            active_directions=len({(r["source"],r["target"]) for r in changed}),
            active_frame_clusters=len({r["frame_cluster"] for r in changed}),
            active_point_content_clusters=len({r["point_content_cluster"] for r in changed}),
            active_scene_clusters=len({r["scene_cluster"] for r in changed}),
            largest_three_positive=sorted(winners,key=lambda r:-r["weighting_contribution_pp"])[:3],
            largest_three_negative=sorted(losers,key=lambda r:r["weighting_contribution_pp"])[:3])
    return bins


def main():
    dest = HERE / "results"
    require(not dest.exists(), "score output already exists")
    geom, diags, earlier, reg, seal = verified_inputs()
    rows, audit = evaluate(geom, diags, earlier, reg)
    past = read(PAST / "results/geometric_attribution.json")
    require(past["groups"] == ["horizontal_center","vertical_center","dimensions","yaw"], "bit order")
    same_cases = {x["ordinal"]:x for x in past["records"]}
    require(len(same_cases)==15, "prior selected cases")
    old_h_error = max(abs(rows[k-1]["H"] - v["hybrid_ious"][9]) for k,v in same_cases.items())
    require(old_h_error < 1e-7, "prior 15 hybrid reference mismatch")
    summary, directions, scenes, by_b = aggregates(rows, NAMES)
    scope = breakdown(rows, same_cases)
    eligible_stats = []
    for group in ["all"] + sorted({r["source"]+"_to_"+r["target"] for r in rows}):
        group_rows = [r for r in rows if group == "all" or group == r["source"]+"_to_"+r["target"]]
        eligible_stats.append(dict(scope=group,n=len(group_rows),origin_known=sum(r["origin_ok"] for r in group_rows),
            **{s:sum(r[s] for r in group_rows) for s in DIAG_BOOL if s.startswith(("eligible_","choose_"))},
            **{"mean_"+c:mean_finite([r[c] for r in group_rows]) for c in DIAG_SCORES},
            **{"median_"+c:float(np.median([r[c] for r in group_rows])) for c in DIAG_NUM}))
    tmp = Path(tempfile.mkdtemp(prefix=".scored_", dir=HERE))
    table(tmp/"all_rows.csv",rows)
    table(tmp/"direction.csv",directions)
    table(tmp/"scene.csv",scenes)
    table(tmp/"subset.csv",by_b)
    table(tmp/"paired.csv",scope)
    table(tmp/"gate_counts.csv",eligible_stats)
    save(tmp/"summary.json",summary)
    save(tmp/"signal_evidence.json",signal_evidence(rows))
    save(tmp/"AUDIT.json",dict(status="PASS", records=len(rows), code_sha256=digest(Path(__file__)),
                prediction_sha256=seal["prediction_sha256"], design_sha256=seal["design_sha256"],
                score_after_seal=True, frozen_endpoints_replayed=2*N, prior_selected_hybrids_replayed=15,
                max_R156_selected_H_error=old_h_error, unique_expressions=ground_truth.cache_info().misses,
                six_direction_counts_preserved=True, strict_cutoffs=[.25,.5], target_final_access=False, **audit))
    os.rename(tmp,dest)
    print("R157 postseal score and audit",audit,flush=True)


if __name__ == "__main__":
    main()
