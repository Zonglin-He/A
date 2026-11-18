"""Run the frozen LCIB research prototype on a single target frame."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'research/round162'))
from evidence162 import EvidenceGrounder, WAYMO_VIEWS
from manifest162 import PLATFORMS, canonical, read, sha
from crossplatform.ground_single_frame import sensor_frame_box
from run162 import PROTOCOL, SETTINGS
from solver162 import Options, solve
from support162 import SceneSupport


class SingleFrameLCIB:
    """Reuse one source model and its fixed source validation configuration."""

    def __init__(self, source):
        saved = read(SETTINGS)
        if saved['protocol_sha256'] != sha(PROTOCOL):
            raise ValueError('The protocol differs from the source parameter selection')
        selection = saved['sources'][source]
        self.model = EvidenceGrounder(source)
        self.options = Options(**selection['options'])
        self.dimensions = np.asarray(selection['log_dimensions'])
        self.sigma = np.asarray(selection['log_scale'])

    def predict(self, *, target, points, rgb, calibration, text, seed=0, waymo_view='F'):
        if target not in PLATFORMS or waymo_view not in WAYMO_VIEWS or (target!='waymo' and waymo_view!='F'):
            raise ValueError('Invalid sensor platform or Waymo view')
        if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)) or seed<0:
            raise ValueError('Provide a nonnegative integer sampling seed')
        if not isinstance(text,str) or not text.strip():
            raise ValueError('Provide a nonempty referring expression')
        paths = [str(Path(p).resolve(strict=True)) for p in (points,rgb,calibration)]
        view = WAYMO_VIEWS.index(waymo_view)
        row = dict(platform=target, suffix=view, seed=int(seed), caption=canonical(text),
                   pointpath=paths[0], imagepath=paths[1], metapath=paths[2])
        ev = self.model.evidence_frame(row)
        result = solve(SceneSupport(ev), self.options, self.dimensions, self.sigma)
        return sensor_frame_box(result.box,target,view)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, choices=PLATFORMS)
    parser.add_argument('--target', required=True, choices=PLATFORMS)
    parser.add_argument('--points', required=True, type=Path)
    parser.add_argument('--rgb', required=True, type=Path)
    parser.add_argument('--calibration', required=True, type=Path, help='3EED camera JSON')
    parser.add_argument('--text', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--waymo-view', choices=WAYMO_VIEWS, default='F')
    args = vars(parser.parse_args())
    with redirect_stdout(sys.stderr):
        source = args.pop('source')
        model = SingleFrameLCIB(source)
        box = model.predict(**args)
    print(json.dumps(dict(box=box.tolist(), frame='target_lidar', length_unit='m', yaw_unit='rad'),allow_nan=False))


if __name__ == '__main__':
    main()
