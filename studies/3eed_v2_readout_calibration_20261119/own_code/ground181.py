"""Experimental source-calibrated readout on the fixed R180 K=16 recipe."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np

from readout181 import ARMS, ReadoutGrounder
from calibrate181 import final_choices, load_calibration
from scope import PLATFORMS
from train import prepare_device
from features import WAYMO_VIEWS
from crossplatform.candidate_frontend import sensor_frame_box


class CalibratedGrounder(ReadoutGrounder):
    def __init__(self,source,*,asset_root=None):
        self.calibration = load_calibration()['models_by_source'][source]
        super().__init__(source,asset_root=asset_root)

    def predict(self,*,arm='calibrated_joint',**frame):
        if arm not in ARMS:
            raise ValueError(f'Unknown readout {arm}')
        row = self.collect(frame)
        choices = final_choices(row['terms'][None],row['chosen'][None],
                                np.array([row['incumbent']]),self.calibration)[0]
        slot=int(choices[ARMS.index(arm)])
        box = sensor_frame_box(row['boxes'][slot],frame['target'],row['view'])
        return dict(box=box.tolist(),arm=arm,retains_v1=slot==row['incumbent'],
                    k=16,frame='target_lidar',length_unit='m',yaw_unit='rad')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,choices=PLATFORMS)
    parser.add_argument('--target',required=True,choices=PLATFORMS)
    parser.add_argument('--points',type=Path,required=True)
    parser.add_argument('--rgb',type=Path,required=True,help='image dimensions for the frozen frontend')
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--text',required=True)
    parser.add_argument('--waymo-view',choices=WAYMO_VIEWS,default='F')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--asset-root',type=Path)
    parser.add_argument('--arm',choices=ARMS,default='calibrated_joint')
    args=vars(parser.parse_args())
    prepare_device()
    with redirect_stdout(sys.stderr):
        model=CalibratedGrounder(args.pop('source'),asset_root=args.pop('asset_root'))
        output=model.predict(**args)
    print(json.dumps(output,allow_nan=False))


if __name__=='__main__':
    main()
