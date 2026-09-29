"""python -m desta3d: inspect config, summarize CPU model, or explicitly fit caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AdapterConfig, DirectionConfig, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "summary", "fit-cache"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-name", help="Required for fit-cache; a NEW manual output directory")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "inspect":
        print(json.dumps(config, indent=2, ensure_ascii=False))
        print("Validated configuration only: no weights, data, CUDA, or experiment loaded.")
        return
    if args.action == "summary":
        import torch
        from .models import DirectionMixer, build_adapter

        if config["kind"] == "pixel_views":
            print(json.dumps(config["views"], indent=2))
            print("Use make_pixel_views(frames, frame_ids, evidence, ViewConfig(...)).")
            return
        if config["kind"] == "adapter":
            model = build_adapter(AdapterConfig(**config["model"]))
        else:
            c = DirectionConfig(**config["model"])
            model = DirectionMixer(torch.eye(max(2560, c.output_rank), c.output_rank), c)
        print(model)
        print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()),
                          "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                          "trainable_names": [n for n, p in model.named_parameters() if p.requires_grad]}, indent=2))
        print("Random CPU structure only; this is not a loaded B1/PTD model.")
        return
    if not args.run_name:
        parser.error("fit-cache requires --run-name; old outputs are never overwritten")
    from .train_cached import fit

    destination = fit(config, args.workspace, args.run_name)
    print(f"Manual terminal fit saved to {destination}")


if __name__ == "__main__":
    main()
