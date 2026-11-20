"""Recalculate the paired decisions, capacity bounds and geometry from the sealed source files."""
import csv
import gzip
import itertools
from collections import Counter
from html.parser import HTMLParser

import numpy as np
from scipy.special import expit

from cases import HERE, OUT, PRIV, PUB, ROOT, SEED, digest, pinned, read
from evaluate import score_boxes
from summarize import collect, summarize, write_json


class Plates(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []
        self.sections = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img":
            self.images.append(attrs["src"])
        if tag == "section":
            self.sections.append(attrs["id"])


def good_bad_order(v, frozen):
    return sorted(range(len(v)), key=lambda i: (-v[i], -frozen[i], i))


def main():
    orig, roster = pinned()
    seal = read(OUT / "SEALED.json")
    truth = read(SEED / "run_PRIVATE/postseal/SEALED.json")
    protocol = read(HERE / "PROTOCOL.json")
    assert truth["predictions"] == protocol["prediction_manifest_sha256"] == digest(OUT / "SEALED.json")
    assert digest(SEED / "run_PRIVATE/postseal/SEALED.json") == protocol["postseal_truth_sha256"]
    assert not seal["labels_read"] and seal["native_state_sha256"] == orig["native_state_hashes"]
    method = read(ROOT / "crossplatform/CURRENT_METHOD.json")["paper_main_method"]
    assert method["status"] == "locked"
    assert all(digest(ROOT / k) == value for k, value in method["local_core_sha256"].items())
    with gzip.open(SEED / "results/source_all_rows.csv.gz", "rt") as f:
        historical = list(csv.DictReader(f))
    wanted = {}
    for r in historical:
        selected, new, best = (float(r[k]) for k in ("frozen_joint_iou", "native_focal_abstain_iou", "pool_best_iou"))
        assert float(r["v1_iou"]) == selected
        if best > .5 >= selected:
            wanted[r["row_id"]] = "frozen_miss"
        if selected > .5 >= new:
            assert r["row_id"] not in wanted
            wanted[r["row_id"]] = "seed_harm"
    rows = collect()
    assert len(historical) == 192 and {r["case_id"]:r["cohort"] for r in rows} == wanted
    clouds, recordings = set(), set()
    errors = Counter()
    for r in rows:
        cid = r["case_id"]
        local = read(PRIV / (cid + ".json"))
        platform, ordinal = local["platform"], local["ordinal"]
        assert ordinal in roster[platform]
        d = local["data"]
        clouds.add(d["point_sha256"])
        recordings.add((platform, d["recording"]))
        gt = np.asarray(local["annotations"][0])
        rel = f"{platform}/{ordinal:05}.npz"
        assert digest(OUT/rel) == seal["files"][rel]
        with np.load(OUT / rel, allow_pickle=False) as z:
            boxes = z["boxes"].astype(float)
            iou, disagreement = score_boxes(boxes, gt)
            errors["iou_engines"] = max(errors["iou_engines"], disagreement)
            x = z["logits"].astype(float)
            probs = expit(x)
            pos = .25 * (1 - probs)**2 * np.logaddexp(0, -x)
            neg = .75 * probs**2 * np.logaddexp(0, x)
            costs = np.where(z["codes"], pos, neg).sum(axis=1)
            delta = neg.sum() - costs
            frozen = z["joint"]
            order = good_bad_order(frozen, frozen)
            seed = good_bad_order(delta.round(12), frozen)
            actual = seed[0] if delta.max() > 1e-12 else order[0]
            assert actual == local["seed"] and order[0] == local["previous"]
            score_order = order if r["cohort"] == "frozen_miss" else seed
            g = next(k for k in score_order if iou[k] > .5)
            b = next(k for k in score_order if iou[k] <= .5)
            assert (g,b) == (local["good"],local["bad"]) and score_order[0] == b
            assert abs(r["good_iou"] - iou[g]) < 1e-7 and abs(r["bad_iou"] - iou[b]) < 1e-7
            assert abs(r["gap_seed"] - (costs[g] - costs[b])) < 1e-10
            t,s = np.asarray(local["terms"]),np.asarray(local["scores"])
            assert np.abs(t[:,2]).max() <= .750001 and np.abs(t[:,3]).max() <= .500001
            if not r["has_reference"]:
                assert not t[:,3].any()
            total = t[:,0] + t[:,2] + t[:,3]
            error = float(np.abs(total - frozen).max())
            assert error < 5e-6
            errors["frozen_reconstruction"] = max(errors["frozen_reconstruction"],error)
            with np.load(PRIV / (cid + "_front.npz"),allow_pickle=False) as pre:
                delta_prior = pre["terms"].sum(1) - t[:,0]
                prior_error = float(np.ptp(delta_prior))
                assert prior_error < 3e-6 and pre["boxes4"].shape == (4,7)
                errors["front_relative_prior"] = max(errors["front_relative_prior"],prior_error)
                iou4, _ = score_boxes(pre["boxes4"].astype(float), gt)
                assert abs(iou4.max() - r["four_max_iou"]) < 1e-7
            for k, col in enumerate(("frozen", "independent", "reference_mismatch")):
                assert abs((s[b,k] - s[g,k]) - r["gap_"+col]) < 1e-6
            hit = iou > .5
            for vector,span,flag in ((t[:,0],1.5 + int(r["has_reference"]),"beyond_full_correction_range"),
                                      (t[:,0] + t[:,2],int(r["has_reference"]),"beyond_direct_reference_range")):
                # Give every passing candidate the full allowed advantage simultaneously.
                best_possible = np.argmax(vector + np.where(hit, span, 0))
                assert r[flag] == (not hit[best_possible])
            indices = ((0,1),(2,),(3,4,5),(6,))
            hybrid = []
            for mask in range(16):
                updated = boxes[b].copy()
                for k,ix in enumerate(indices):
                    if mask & (1<<k):
                        updated[list(ix)] = boxes[g,list(ix)]
                hybrid.append(updated)
            grid, cross_error = score_boxes(np.stack(hybrid), gt)
            errors["hybrid_iou_engines"] = max(errors["hybrid_iou_engines"],cross_error)
            marginals = np.zeros(4)
            for permutation in itertools.permutations(range(4)):
                mask = 0
                for k in permutation:
                    nxt = mask | (1 << k)
                    marginals[k] += (grid[nxt] - grid[mask]) / 24
                    mask = nxt
            saved = [r["geometry_" + part] for part in ("xy", "z", "size", "yaw")]
            np.testing.assert_allclose(marginals, saved, atol=1e-8, rtol=0)
    assert len(clouds) == 28 and len(recordings) == 12
    assert summarize(rows) == read(PUB / "summary.json")
    plates = Plates()
    plates.feed((PRIV / "index.html").read_text())
    assert sorted(plates.sections) == sorted(wanted) and len(plates.images) == 28
    assert set(plates.images) == {f"plates/{cid}.jpg" for cid in wanted}
    assert all((PRIV / i).is_file() for i in plates.images)
    assert all((HERE / "CASES_ZH.md").read_text().count("| " + i + " |") == 1 for i in wanted)
    decision = dict(seed_terminal_readout="retired", deployed="simplified_v1", checkpoint_changed=False,
                    diagnosis="Source-own post-seal paired analysis. Distinguish selection of another instance from geometry of the chosen instance.",
                    next_edit="Prioritize subject comparison between separate physical instances while fixing each candidate box and the reference readout. Measure within-instance localization separately.",
                    candidate_training_or_new_method=False, ordinary_target_runs=0, FINAL_access=False)
    write_json(PUB / "decision.json", decision)
    hashes = {p.name:digest(p) for p in sorted(PUB.iterdir()) if p.is_file() and p.name!="audit.json"}
    receipt = dict(parent=protocol["parent"], parent_predictions=digest(OUT / "SEALED.json"),
                   parent_gt_manifest_sha256=protocol["postseal_truth_sha256"],
                   parent_gt_linked_to_sealed_predictions=truth["predictions"]==digest(OUT / "SEALED.json"),
                   source_rows_in_parent=len(historical), paired_rows=len(rows), unique_point_clouds=len(clouds),
                   recording_groups=len(recordings), all_28_cases_have_source_rgb_and_lidar_plates=True,
                   same_target=sum(r["identity"]=="same_target" for r in rows),
                   other_instance=sum(r["identity"]=="other_instance" for r in rows),
                   unresolved=sum(r["identity"]=="unresolved" for r in rows),
                   independent_second_visual_reader=False, gt_used_to_change_selections=False, FINAL_access=False,
                   continuous_geometry_hybrids_scored=len(rows)*16, independent_permutations_per_pair=24,
                   max_numeric_errors=dict(errors), deployed_core_sha256_verified=True, hashes=hashes)
    write_json(PUB / "audit.json", receipt)
    print(f"Verified {len(rows)} pairs, {len(clouds)} frames, {len(recordings)} recordings and {len(plates.images)} plates.")


if __name__ == "__main__":
    main()
