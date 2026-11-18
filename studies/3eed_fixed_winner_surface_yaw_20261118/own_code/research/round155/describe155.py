"""Conditional intervals and post-seal attribution using only anonymous R155 IoUs."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from score155 import HERE, digest, read, require, save


def main():
    results = HERE / "results"
    audit = read(results/"AUDIT.json")
    require(audit["status"] == "PASS" and audit["n"] == 1798, "requires audited sealed R155 scores")
    with (results/"task_all_rows.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == audit["n"], "row count")
    data = lambda key: np.array([float(r[key]) for r in rows])
    estimate_names = ["vs_frozen", "vs_axis", "vs_same_information_zero", "vs_sign_flip",
                      "vs_pca", "zero_vs_zero_axis", "original_TTA_increment", "TTA_interaction",
                      "local_choice_upper_bound_vs_axis", "positive_delta_vs_axis", "negative_delta_vs_axis"]
    new, base = data("R145_surface"), data("R145_axis")
    zero, zero_axis, sign, pca = (data(x) for x in ("joint_zero_surface", "joint_zero_axis", "R145_sign_flip", "R145_pca"))
    delta = np.stack([new - data("frozen_A"), new-base, new-zero, new-sign, new-pca, zero-zero_axis,
                      base-zero_axis, (new-zero)-(base-zero_axis), np.maximum.reduce([base,pca,new])-base,
                      np.maximum(new-base,0),np.minimum(new-base,0)],axis=1)*100.
    cluster = data("point_content_cluster").astype(int)
    w = data("sampling_weight")
    recording_groups, direction_rows = defaultdict(set), defaultdict(list)
    for index,row in enumerate(rows):
        recording_groups[(row["target"],row["scene_cluster"])].add(cluster[index])
        direction_rows[(row["source"],row["target"])].append(index)
    require(sum(map(len, recording_groups.values())) == len(set(cluster)) == 384, "clusters cross recorded strata")
    rng = np.random.default_rng(155)
    counts = np.zeros((2000,cluster.max()+1),int)
    for members in recording_groups.values():
        members = sorted(members)
        counts[:,members] = rng.multinomial(len(members), np.ones(len(members))/len(members), size=len(counts))

    def macro(weights):
        return np.mean([np.average(delta[idx],axis=0,weights=weights[idx]) for idx in direction_rows.values()],axis=0)

    boot = np.array([macro(w * v[cluster]) for v in counts])
    lo,hi = np.percentile(boot,[2.5,97.5],axis=0)
    changes = {key: dict(mIoU_gain_pp=float(point),conditional_lo_pp=float(a),conditional_hi_pp=float(b))
               for key,point,a,b in zip(estimate_names,macro(w),lo,hi)}
    eligible = np.array([r["eligible_R145"] == "True" for r in rows])
    # Distinguish an IoU that changed from one crossing the public strict task thresholds.
    by_eligibility = {}
    for name,idx in (("eligible",eligible),("fallback",~eligible),("all",np.ones(len(eligible),dtype=bool))):
        ds = new[idx]-base[idx]
        by_eligibility[name] = {"n":int(idx.sum()),"improved":int((ds>1e-12).sum()),
                                "worsened":int((ds < -1e-12).sum()),"same":int((abs(ds)<=1e-12).sum())}
        for t in (.25,.5):
            by_eligibility[name][f"rescued{int(t*100)}"] = int(((new[idx]>t)&(base[idx]<=t)).sum())
            by_eligibility[name][f"lost{int(t*100)}"] = int(((new[idx]<=t)&(base[idx]>t)).sum())
    result = {"source_scalar_sha256":digest(results/"task_all_rows.csv"),"n":len(rows),
              "uncertainty":"Six equally weighted directions, original design weights within each, paired within actual target-by-recording content clusters, 2000 percentile draws (seed 155). Conditional on the observed recordings and prior method selection.",
              "upper_bound_scope":"Post-seal, per-case better of recorded axis/PCA/surface outputs on the fixed R145 winner only. Excludes sign-flip control.",
              "changes":changes,"eligible_fixed_R145":by_eligibility}
    save(results/"ORIENTATION_DIAGNOSTICS.json",result)
    print("PASS_R155_DESCRIPTION",len(rows),flush=True)


if __name__ == "__main__":
    main()
