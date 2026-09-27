#!/usr/bin/env python3
"""Run ONLY the fixed method on user-supplied frames, without dataset labels.

NPZ must contain uint8 frames [T,H,W,3] and integral frame_ids [T]. The output
is a new .pt artifact; an existing file is never overwritten. No loop is started.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", required=True, choices=("vid_to_hc1", "hc2_to_vid"))
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", action="store_true", help="Include hashes and all optimizer states")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    import numpy as np
    import torch
    from methods.decota_final_simplified_v1 import DeCoTAPredictor
    with np.load(args.frames, allow_pickle=False) as archive:
        frames = archive["frames"]
        ids = archive["frame_ids"].tolist()
    metadata = dict(caption=args.query, index="input", width=frames.shape[2], height=frames.shape[1])
    predictor = DeCoTAPredictor.from_pretrained(args.direction, audit=args.audit)
    result = predictor.predict(frames, ids, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        torch.save(result, handle)
    print(f"Saved {args.output}: interval={result['physical_interval']}")


if __name__ == "__main__":
    main()
