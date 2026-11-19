"""An experimental geometry selector after the v1 subject decision."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np
import torch

from quality179 import choose, pair_evidence, supported
from run179 import all_scores, calibrated, protocol, sensor_points, PLATFORMS, SELECTORS
from features import WAYMO_VIEWS, encode
from ground_instance import SourceInstanceGrounder, tensorize
from crossplatform.candidate_frontend import sensor_frame_box


class FixedInstanceGeometryGrounder(SourceInstanceGrounder):
    """Keep the v1 carrier and return one of its two geometries."""

    def __init__(self, source, *, asset_root=None):
        self.calibration = calibrated()['sources'][source]
        super().__init__(source,asset_root=asset_root)

    @torch.inference_mode()
    def predict(self, *, selector='physical_pair', **frame):
        if selector not in SELECTORS:
            raise ValueError(f'Unknown geometry selector {selector!r}')
        with self._lock:
            feature,_ = encode(self.frozen,frame)
            batch = tensorize(feature)
            proposal = self.head.pool(batch,use_votes=True,use_mask=True)
            carrier = int(feature['v1_index'])
            view = int(feature['view_index'])
            target = frame['target']
            raw = sensor_frame_box(feature['v1_box'],target,view)
            refined = sensor_frame_box(proposal['refined'][0,carrier].cpu().numpy(),target,view)
        x,counts,point_score = pair_evidence(sensor_points(feature['points'],target,view),raw,refined)
        eligible = bool(supported(counts))
        scores = all_scores(self.calibration['models'],x[None],np.array([point_score]))
        switched = bool(choose(scores[selector],np.array([eligible]),self.calibration['thresholds'][selector])[0])
        return dict(box=(refined if switched else raw).tolist(),selector=selector,
                    geometry='refined' if switched else 'raw',support_eligible=eligible,
                    carrier=carrier,frame='target_lidar',length_unit='m',yaw_unit='rad')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',required=True,choices=PLATFORMS)
    cli.add_argument('--target',required=True,choices=PLATFORMS)
    cli.add_argument('--points',required=True,type=Path)
    cli.add_argument('--rgb',required=True,type=Path,help='image dimensions used by the v1 frontend')
    cli.add_argument('--calibration',required=True,type=Path)
    cli.add_argument('--text',required=True)
    cli.add_argument('--seed',default=protocol()['geometry_pair']['point_seed'],type=int)
    cli.add_argument('--waymo-view',choices=WAYMO_VIEWS,default='F')
    cli.add_argument('--asset-root',type=Path)
    cli.add_argument('--selector',choices=SELECTORS,default='physical_pair')
    args = vars(cli.parse_args())
    with redirect_stdout(sys.stderr):
        output = FixedInstanceGeometryGrounder(args.pop('source'),asset_root=args.pop('asset_root')).predict(**args)
    print(json.dumps(output,allow_nan=False))


if __name__=='__main__':
    main()
