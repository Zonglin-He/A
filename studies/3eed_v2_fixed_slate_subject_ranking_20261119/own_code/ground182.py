"""Run the experimental subject ranker on a registered source checkpoint."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np
import torch

from ranker182 import ARMS, FeatureGrounder, readouts
from fit182 import load_heads
from scope import PLATFORMS
from train import prepare_device
from features import WAYMO_VIEWS
from crossplatform.candidate_frontend import sensor_frame_box


class RankedGrounder(FeatureGrounder):
    def __init__(self, source, *, asset_root=None):
        super().__init__(source, asset_root=asset_root)
        self.heads = load_heads(source)

    @torch.inference_mode()
    def all_boxes(self, frame):
        row = self.collect(frame)
        features = torch.from_numpy(row['features'][None])
        residual, generic = (self.heads[a](features).numpy() for a in ('hard', 'generic'))
        incumbent = np.array([row['incumbent']])
        scores = row['scores'][None]
        new = readouts(residual, generic, scores, incumbent)
        old = dict(zip(ARMS[:4], (row['incumbent'], *row['chosen'])))
        slots = dict(old, **{a: int(v[0]) for a, v in new.items()})
        return {a: dict(box=sensor_frame_box(row['boxes'][i], frame['target'], row['view']).tolist(),
                        retains_v1=i == row['incumbent'], arm=a, k=16, frame='target_lidar',
                        length_unit='m', yaw_unit='rad') for a, i in slots.items()}

    def predict(self, *, arm='rank_joint', **frame):
        if arm not in ARMS:
            raise ValueError(f'Unknown readout {arm}')
        return self.all_boxes(frame)[arm]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, choices=PLATFORMS)
    parser.add_argument('--target', required=True, choices=PLATFORMS)
    parser.add_argument('--points', required=True, type=Path)
    parser.add_argument('--rgb', required=True, type=Path, help='image dimensions for the frozen frontend')
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--text', required=True)
    parser.add_argument('--waymo-view', choices=WAYMO_VIEWS, default='F')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--asset-root', type=Path)
    parser.add_argument('--arm', choices=ARMS, default='rank_joint')
    args = vars(parser.parse_args())
    prepare_device()
    with redirect_stdout(sys.stderr):
        method = RankedGrounder(args.pop('source'), asset_root=args.pop('asset_root'))
        output = method.predict(**args)
    print(json.dumps(output, allow_nan=False))


if __name__ == '__main__':
    main()
