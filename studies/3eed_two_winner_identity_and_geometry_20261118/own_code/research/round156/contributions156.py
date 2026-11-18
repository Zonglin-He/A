"""Post-seal decomposition of the attainable two-choice gain on all ordinary rows."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    with (HERE / "results/pair_rows.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1798 and all(int(r["ordinal"]) == i for i, r in enumerate(rows, 1))
    w = np.array([float(r["sampling_weight"]) for r in rows])
    ix = defaultdict(list)
    for i, r in enumerate(rows):
        ix[r["source"], r["target"]].append(i)
    assert len(ix) == 6
    for ids in ix.values():
        w[ids] /= 6 * w[ids].sum()
    a, d = (np.array([float(r[k]) for r in rows]) for k in ("A_pca", "D_pca"))
    delta = d - a
    subsets = {
        "all": np.ones(len(rows), bool),
        "A_zero_D_nonzero": (a == 0) & (d > 0),
        "A_nonzero_D_better": (a > 0) & (d > a),
        "A_fails25_D_passes": (a <= .25) & (d > .25),
        "A_fails50_D_passes": (a <= .5) & (d > .5)
    }
    out = {}
    for name, subset in subsets.items():
        out[name] = {
            "n": int(subset.sum()), "positive": int((subset & (delta > 1e-12)).sum()),
            "negative": int((subset & (delta < -1e-12)).sum()),
            "positive_contribution_pp": float(100 * (w * np.maximum(0., delta) * subset).sum()),
            "negative_contribution_pp": float(100 * (w * np.minimum(0., delta) * subset).sum())}
    dest = HERE / "results/attainable_contributions.json"
    dest.write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
