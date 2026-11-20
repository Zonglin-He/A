"""Prepare ordinary point maps, then the isolated all-return depth oracle."""
import argparse
from pathlib import Path
import time

import numpy as np

from shared import ASSETS, OUT, PRIVATE, PROTOCOL, SIDE, digest, read, write
from common import require_weights
from maps import atomic_npz, cuda_summary, load_frames, open_rgb, pda_sparse, requirements, run_maps, run_moge


def oracle(sources, weights_root):
    import cv2
    import torch

    requirements(sources, "prior_depth", "Prior-Depth-Anything")
    from prior_depth_anything import PriorDepthAnything

    assets = require_weights(weights_root, "prior_depth")
    completion = PriorDepthAnything(device="cuda:0", version="1.1", mde_dir=str(assets),
        ckpt_dir=str(assets), frozen_model_size="vitb", conditioned_model_size="vitb").eval().requires_grad_(False)
    torch.manual_seed(20261121)
    standard_hashes = read(PRIVATE / "map_hashes.json")
    maps_seal = read(OUT / "maps_seal.json")
    assert maps_seal["hashes"] == digest(PRIVATE / "map_hashes.json")
    folder = PRIVATE / "oracle_depth"
    folder.mkdir(mode=0o700, exist_ok=True)
    stamp = dict(ordinary_maps=digest(OUT / "maps_seal.json"), protocol=digest(PROTOCOL),
                 code=digest(__file__), assets=digest(ASSETS))
    hashes, coverage = {}, []
    started = time.perf_counter()
    with torch.inference_mode():
        for frame in load_frames():
            uid = frame["uid"]
            path = folder / (uid + ".npz")
            cache = folder / (uid + ".json")
            ordinary = PRIVATE / "maps" / (uid + ".npz")
            observations = PRIVATE / "observations" / (uid + ".npz")
            assert digest(ordinary) == standard_hashes[uid]
            assert digest(observations) == frame["observation_sha"]
            if cache.exists():
                meta = read(cache)
                assert meta["stamp"] == stamp and meta["sha"] == digest(path)
            else:
                with np.load(ordinary, allow_pickle=False) as data:
                    fit, held, mask = data["fit"], data["held"], data["mask"]
                    fallback = data["prior_depth"]
                with np.load(observations, allow_pickle=False) as data:
                    cells, pixels, depth = data["cells"], data["uv"], data["camera"][:, 2]
                observed = fit | held
                np.testing.assert_array_equal(observed, mask[cells[:, 1], cells[:, 0]])
                image, sparse = pda_sparse(open_rgb(frame), pixels[observed], depth[observed], np.asarray(frame["size"]))
                enough = int((sparse > 0).sum()) >= 5
                result = fallback.copy()
                if enough:
                    dense = completion.infer_one_sample(image=image, prior=sparse, visualize=False).cpu().numpy()
                    dense = cv2.resize(dense, (SIDE, SIDE), interpolation=cv2.INTER_NEAREST)
                    valid = np.isfinite(dense) & (dense > 0.)
                    result[valid] = np.clip(dense[valid], .25, 250.)
                y, x = cells[observed, 1], cells[observed, 0]
                result[y, x] = depth[observed]
                assert np.isfinite(result[mask]).all()
                y, x = cells[held, 1], cells[held, 0]
                np.testing.assert_array_equal(result[y, x], depth[held])
                atomic_npz(path, depth=result)
                meta = dict(stamp=stamp, sha=digest(path), target=frame["target"], anchors=int(observed.sum()),
                            injected_holdout=int(held.sum()), completion_fallback=not enough)
                write(cache, meta)
            hashes[uid] = meta["sha"]
            coverage.append({k: meta[k] for k in ("target", "anchors", "injected_holdout", "completion_fallback")})
            if len(hashes) % 16 == 0:
                print(f"Depth oracle {len(hashes)}/96", flush=True)
    write(PRIVATE / "oracle_hashes.json", hashes)
    write(OUT / "oracle_seal.json", dict(stamp=stamp, hashes=digest(PRIVATE / "oracle_hashes.json"),
          frames=len(hashes), holdout_anchor_cells=sum(row["injected_holdout"] for row in coverage),
          all_anchor_cells=sum(row["anchors"] for row in coverage),
          fallback_frames=sum(row["completion_fallback"] for row in coverage), **cuda_summary(started)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("moge", "maps", "oracle"))
    parser.add_argument("sources", type=Path)
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()
    {"moge": run_moge, "maps": run_maps, "oracle": oracle}[args.stage](args.sources, args.weights)
