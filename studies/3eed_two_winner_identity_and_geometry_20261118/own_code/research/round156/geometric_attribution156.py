"""Post-hoc endpoint decomposition for masked-reader agreement on same-target geometry."""
import itertools
import os
from pathlib import Path

import numpy as np

from score156 import (HERE, digest, ground_truth, many_iou, oriented_iou, read, read_csv, require,
                      rotation, save, verified_inputs)

RESULTS = HERE / "results"
GROUPS = ("horizontal_center", "vertical_center", "dimensions", "yaw")
PARTS = ((0, 1), (2,), (3, 4, 5), (6,))
PERMUTATIONS = tuple(itertools.permutations(range(len(PARTS))))
PINS = {
    "identity_rows.csv": "91d977e87e10951240667834a6c008f0fe94765eeebdb727ffee1a657917b041",
    "IDENTITY_SUMMARY.json": "db6fb46c6840f851b6ba3429f69821b3edc2cdea198b4c015348facd0088d39b",
    "pair_rows.csv": "66ea7caa0392a38e1d59f847b93892b1597c64d5e3a64ebe8be2a9739f5063bf",
}


def decomposition(a, d, gt):
    mixed = np.repeat(a[None, :], 1 << len(PARTS), axis=0)
    for mask, box in enumerate(mixed):
        for j, indices in enumerate(PARTS):
            if mask & (1 << j):
                box[list(indices)] = d[list(indices)]
    ious = many_iou(mixed, gt)
    gt_rot = rotation(gt[6])
    independent = [oriented_iou(box[:3], box[3:6], rotation(box[6]),
                                gt[:3], gt[3:6], gt_rot) for box in mixed]
    independent_err = np.max(np.abs(np.array(independent) - ious))
    require(independent_err <= 1e-7, "hybrid IoU audit failed")
    phi = np.zeros(len(PARTS))
    for order in PERMUTATIONS:
        previous = 0
        for group in order:
            following = previous | (1 << group)
            phi[group] += ious[following] - ious[previous]
            previous = following
    return ious, phi / len(PERMUTATIONS), float(independent_err)


def descriptive(values):
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                min=float(np.min(values)), max=float(np.max(values)),
                positive=int(np.sum(values > 1e-12)), negative=int(np.sum(values < -1e-12)),
                near_zero=int(np.sum(np.abs(values) <= 1e-12)))


def main():
    out = RESULTS / "geometric_attribution.json"
    require(not out.exists(), "post-hoc summary already exists")
    for filename, sha in PINS.items():
        require(digest(RESULTS / filename) == sha, "upstream evidence changed")
    endpoints, registry, _, _, pair_seal = verified_inputs()
    idrows = read_csv(RESULTS / "identity_rows.csv")
    identity = read(RESULTS / "IDENTITY_SUMMARY.json")
    total_draws = sum(int(r["positive_draws"]) for r in idrows)
    mass_pp = identity["positive_gain_oracle_mass_pp"]
    category = "SAME_TARGET_GEOMETRY"
    sample = [r for r in idrows if int(r["positive_draws"]) and
              r["reader1_category"] == r["reader2_category"] == r["consensus_category"] == category]
    expected = identity["positive"]["categories"][category]
    require(total_draws == identity["positive"]["draws"] and
            len(sample) == len({r["ordinal"] for r in sample}) == expected["unique_records"], "identity strata")
    audits = []
    records = []
    for r in sample:
        i = int(r["ordinal"]) - 1
        e = registry[i]
        frame = e["row"]
        gt = ground_truth(e["target"], frame["metapath"], frame["object_index"], frame["caption"])
        iou, phi, independent_err = decomposition(endpoints[i, 1], endpoints[i, 4], gt)
        endpoint_err = max(abs(iou[0] - float(r["A_pca"])), abs(iou[-1] - float(r["D_pca"])))
        delta = iou[-1] - iou[0]
        closure_err = abs(sum(phi) - delta)
        require(delta > 0 and closure_err <= 1e-12 and endpoint_err <= 1e-7, "endpoints or Shapley closure")
        draws = int(r["positive_draws"])
        contribution = draws / total_draws * mass_pp * phi / delta
        records.append(dict(case=r["case"], ordinal=i+1, positive_draws=draws, source=e["source"], target=e["target"],
                            A_pca=float(iou[0]), D_pca=float(iou[-1]), shapley_iou=phi.tolist(),
                            pps_oracle_pp=contribution.tolist(), hybrid_ious=iou.tolist()))
        audits.append((endpoint_err, closure_err, independent_err))
    n = np.array([r["positive_draws"] for r in records])
    unique_phi = np.array([r["shapley_iou"] for r in records])
    shares = np.array([r["pps_oracle_pp"] for r in records])
    estimates = shares.sum(axis=0)
    require(n.sum() == expected["draws"] and abs(estimates.sum()-expected["estimated_oracle_mass_pp"]) < 1e-10,
            "PPS mass does not close")
    components = {}
    for j, name in enumerate(GROUPS):
        components[name] = dict(
            estimated_oracle_pp=float(estimates[j]),
            share_of_estimated_same_target_mass=float(estimates[j] / estimates.sum()),
            unique_record_shapley_iou=descriptive(unique_phi[:, j]),
            draw_shapley_iou=descriptive(np.repeat(unique_phi[:,j], n)))
    output = dict(status="PASS", posthoc=True, data_scope="positive draws independently marked SAME_TARGET_GEOMETRY by both readers",
                  endpoint_pair="A_pca, D_pca (the shared R155 16k point set)", groups=GROUPS, hybrid_bits="bit j=1 replaces group j with D; mask 0 is A, 15 is D",
                  positive_draws=int(total_draws), positive_same_target_draws=int(n.sum()), same_target_unique_records=len(records),
                  positive_gain_oracle_mass_pp=mass_pp, estimated_same_target_geometry_pp=float(estimates.sum()),
                  aggregation="Per draw: total positive oracle mass / number of all positive draws, times Shapley IoU divided by the positive endpoint IoU difference. Sum over eligible draws including repetitions.",
                  interpretation="Signed decomposition of the observed positive endpoint difference under equal ordering of 4 component groups. Conditional on geometry read and positive endpoint improvement. No model output or deployable selection is produced.",
                  pair_prediction_sha256=pair_seal["prediction_sha256"], upstream_result_hashes=PINS,
                  code_sha256=digest(Path(__file__)), max_endpoint_IoU_error=max(v[0] for v in audits),
                  max_case_shapley_closure_error=max(v[1] for v in audits), max_independent_hybrid_IoU_error=max(v[2] for v in audits),
                  unique_hybrids_evaluated=len(records)*16,orders_averaged=len(PERMUTATIONS),components=components, records=records)
    tmp=out.with_suffix(".tmp")
    save(tmp,output)
    os.replace(tmp,out)
    print("R156 post-hoc geometry components (pp)", dict(zip(GROUPS,estimates)),flush=True)


if __name__ == "__main__":
    main()
