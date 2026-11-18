"""Decompose already-scored A/D gains by intersection between the two predicted boxes."""
import csv
import json
from pathlib import Path

import numpy as np

from score156 import OUT, digest, many_iou

HERE = Path(__file__).resolve().parent


def main():
    sealed = OUT / "PAIR_PRIVATE.npz"
    assert digest(sealed) == json.loads((OUT / "PAIR_SEAL.json").read_text())["prediction_sha256"]
    with (HERE / "results/pair_rows.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    with np.load(sealed, allow_pickle=False) as f:
        boxes = f["boxes"][:, :, 1].astype(np.float64)
        assert np.array_equal(f["ordinal"], [int(r["ordinal"]) for r in rows])
    boxes[..., 3:6] = np.maximum(boxes[..., 3:6], 1e-6)
    between = np.array([many_iou(a[None, :], d)[0] for a, d in boxes])
    delta = np.array([float(r["D_pca"]) - float(r["A_pca"]) for r in rows])
    weights = np.array([float(r["macro_weight"]) for r in rows])
    assert abs(weights.sum() - 1) < 1e-12 and np.isfinite(between).all()
    out = []
    for name, mask in (("disjoint", between == 0), ("positive_to_half", (between > 0) & (between <= .5)),
                       ("above_half", between > .5)):
        d = delta[mask]
        w = weights[mask]
        out.append(dict(predicted_pair_3D_IoU=name, n=int(mask.sum()),
                        D_better=int((d > 1e-12).sum()), D_worse=int((d < -1e-12).sum()),
                        tie=int((abs(d) <= 1e-12).sum()),
                        attainable_pp=float(100 * w @ np.maximum(d, 0)),
                        damage_pp=float(100 * w @ np.minimum(d, 0))))
    assert sum(r["n"] for r in out) == len(rows) == 1798
    with (HERE / "results/pair_overlap_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out[0])
        writer.writeheader()
        writer.writerows(out)


if __name__ == "__main__":
    main()
