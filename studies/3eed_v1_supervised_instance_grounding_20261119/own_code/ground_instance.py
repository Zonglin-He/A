"""Run the source-trained instance pipeline on one ordinary observation."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import threading

import torch

from features import PLATFORMS, WAYMO_VIEWS, encode
from crossplatform.candidate_frontend import sensor_frame_box
from crossplatform.ground import Grounder
from train import MODELS, SELECTION, decisions, load_head, prepare_device


def tensorize(feature):
    return {key:torch.as_tensor(value,device='cuda')[None] for key,value in feature.items()}


class SourceInstanceGrounder:
    """A frozen 3EED frontend and one verified source-trained instance head."""

    def __init__(self, source, *, asset_root=None, head_root=MODELS):
        prepare_device()
        self.source = source
        self.frozen = Grounder(source, asset_root=asset_root)
        self.head, self.source_readout = load_head(source,Path(head_root))
        self._lock = threading.Lock()

    @torch.inference_mode()
    def predict(self, *, readout=None, **frame):
        if readout is None:
            readout = self.source_readout
        if readout not in SELECTION:
            raise ValueError(f'Unknown readout: {readout}')
        with self._lock:
            feature,_ = encode(self.frozen,frame)
            batch = tensorize(feature)
            outputs, refs, known, _ = decisions(self.head(batch), batch)
            view = int(feature['view_index'])
            subject = sensor_frame_box(outputs[readout][0].cpu().numpy(),frame['target'],view)
            reference = None
            if known[readout][0]:
                reference = sensor_frame_box(refs[readout][0].cpu().numpy(),frame['target'],view)
        return dict(subject_box=subject.tolist(),reference_box=None if reference is None else reference.tolist(),
                    readout=readout,frame='target_lidar',length_unit='m',yaw_unit='rad')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',required=True,choices=PLATFORMS)
    cli.add_argument('--target',required=True,choices=PLATFORMS)
    cli.add_argument('--points',required=True,type=Path)
    cli.add_argument('--rgb',required=True,type=Path,help='camera-image dimensions only')
    cli.add_argument('--calibration',required=True,type=Path)
    cli.add_argument('--text',required=True)
    cli.add_argument('--seed',default=0,type=int)
    cli.add_argument('--waymo-view',default='F',choices=WAYMO_VIEWS)
    cli.add_argument('--asset-root',type=Path)
    cli.add_argument('--head-root',default=MODELS,type=Path)
    cli.add_argument('--readout',choices=SELECTION)
    args = vars(cli.parse_args())
    with redirect_stdout(sys.stderr):
        method = SourceInstanceGrounder(args.pop('source'),asset_root=args.pop('asset_root'),
                                        head_root=args.pop('head_root'))
        out = method.predict(**args)
    print(json.dumps(out,allow_nan=False))


if __name__=='__main__':
    main()
