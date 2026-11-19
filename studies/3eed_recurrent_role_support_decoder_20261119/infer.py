"""Run the isolated candidate on a single local calibrated frame and a description."""
import argparse
import json
from pathlib import Path

import torch

from common import PLATFORMS, PRIVATE, camera_view, canonical, points_for
from crossplatform.ground_single_frame import sensor_frame_box
from decoder import Decoder
from evidence import encoder, extract


class JointGrounder:
    def __init__(self, source, checkpoint):
        self.base = encoder(source)
        self.decoder = Decoder('joint').eval().requires_grad_(False)
        self.decoder.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        self.decoder.cuda()

    @torch.inference_mode()
    def predict(self, target, point_file, image_file, calibration, text, seed=0, view=0):
        if target not in PLATFORMS or not text.strip():
            raise ValueError('A supported target platform and nonempty description are required')
        row = dict(platform=target, suffix=view, seed=seed, caption=canonical(text),
                   pointpath=str(point_file), imagepath=str(image_file), metapath=str(calibration))
        points, _ = points_for(row)
        evidence = extract(self.base, [row], points[None], coordinate_audit=True)
        decoded = self.decoder(evidence)
        b = decoded['boxes'][0].cpu().numpy()
        transform = lambda box: sensor_frame_box(box, target, camera_view(row)).tolist()
        return dict(subject_box=transform(b[0]),
                    reference_box=transform(b[1]) if bool(evidence['has_reference'][0]) else None,
                    frame='target LiDAR', box_order=['x', 'y', 'z', 'length', 'width', 'height', 'axial_yaw_radians'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', choices=PLATFORMS, required=True)
    p.add_argument('--target', choices=PLATFORMS, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--points', type=Path, required=True)
    p.add_argument('--rgb', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--text', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--view', type=int, choices=range(5), default=0)
    a = p.parse_args()
    path = a.checkpoint or PRIVATE/a.source/'joint.pth'
    runner = JointGrounder(a.source, path)
    print(json.dumps(runner.predict(a.target, a.points, a.rgb, a.calibration, a.text, a.seed, a.view)))
