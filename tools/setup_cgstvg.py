"""Verify existing official CG checkpoints and link the existing model assets.

No network downloads, model forward, new checkpoint or parameter search.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def link(destination, source):
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source:
            raise FileExistsError(destination)
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())


def main():
    from stvg.cgstvg import ROOT, CHECKPOINTS, HUB_REVISION, REVISION, checkpoint_for
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.prepare:
        zoo = ROOT / "runtime/cgstvg/model_zoo"
        zoo.mkdir(parents=True, exist_ok=True)
        link(zoo / "swin_tiny_patch244_window877_kinetics400_1k.pth",
             ROOT / "checkpoints/tastvg_model_zoo/swin_tiny_patch244_window877_kinetics400_1k.pth")
        cache = ROOT / "checkpoints/huggingface/hub/models--roberta-base"
        revision = (cache / "refs/main").read_text().strip()
        for alias in ("roberta-base", "roberta"):
            link(zoo / alias, cache / "snapshots" / revision)
    if args.check:
        rows = []
        for target in CHECKPOINTS:
            path, source = checkpoint_for(target)
            rows.append(dict(target_dataset=target, source_dataset=source,
                             filename=path.name, bytes=path.stat().st_size,
                             sha256=CHECKPOINTS[source][1]))
        out = dict(status="official_cross_domain_assets_verified", vendor_revision=REVISION,
                   hub_revision=HUB_REVISION, assets=rows, models_loaded=False,
                   model_forward_executed=False, GPU_zero_state_verified=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
    if not (args.prepare or args.check):
        p.print_help()


if __name__ == "__main__":
    main()
