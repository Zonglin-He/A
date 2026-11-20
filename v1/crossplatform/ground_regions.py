"""Single-frame grounding with four physical scales and subject phrase nominations."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.special import softmax
import torch
from crossplatform.candidate_frontend import PLATFORMS, WAYMO_VIEWS, orient_box, sensor_frame_box
from crossplatform.ground import Grounder as FrozenGrounder
from crossplatform.v1_evidence import visible_pixels
from crossplatform.physical_regions import language, regions, selection, semantic


def phrase_bank(scene, observed, frame):
    from reader158 import overlaps

    points = np.unique(observed.sampled_points, axis=0)
    np.testing.assert_array_equal(scene.sampled_points, observed.sampled_points)
    boxes = np.stack([orient_box(points, box) for box in scene.boxes]).astype(float)
    overlap = overlaps(scene.boxes)
    weight = softmax(np.log(np.maximum(scene.probability,1e-300)) - np.log((overlap**2).sum(1)))
    projection = regions.box_projection(boxes,scene.target,scene.view_index,frame['calibration'],frame['rgb'])
    _, visible = visible_pixels(points,points,scene.target,scene.view_index,frame['rgb'],frame['calibration'])
    returns = regions.Returns(points[visible])
    counts = np.array([len(returns.local(box)[0]) for box in boxes])
    return dict(box=boxes,p=scene.probability,m=weight,project=projection,count=counts)


class Grounder(FrozenGrounder):
    """Predict one target LiDAR box using one frozen source model."""
    def __init__(self, source, *, asset_root=None, scales=4):
        if scales != len(regions.FOUR_LEVEL_INDICES):
            raise ValueError("This release uses four scales. The preceding eight-scale release is 4f6292f8.")
        super().__init__(source, asset_root=asset_root)

    @torch.inference_mode()
    def predict(self, *, target, points, rgb, calibration, text, seed=0, waymo_view='F'):
        frame = dict(target=target,points=points,rgb=rgb,calibration=calibration,
                     text=text,seed=seed,waymo_view=waymo_view)
        with self._lock:
            capture = {}
            scene, evidence = self._read_observation(frame,capture=capture)
            frozen = self._decoder(evidence)['boxes'][0,0].double().cpu().numpy()
            xyz = np.unique(scene.sampled_points.astype(float),axis=0)
            parsed = language.interpret(text.strip(),capture)
            words = [] if parsed is None else [language.nominal(p) for p in parsed[:2]]
            words = ['other vehicle' if w=='othervehicle' else w for w in words]
            if len(words)==2 and all(w in language.CLASSES[:-1] for w in words):
                words = tuple(language.CLASSES.index(w) for w in words)
                base = self._frontend.base
                if not hasattr(self,'_noun_banks'):
                    self._noun_banks = semantic.prompts(base),semantic.prepare_nouns(base)
                scores, word_data = semantic.observe(base,scene.sampled_points,*self._noun_banks)
                seg = regions.build(scene.sampled_points,scores,word_data['xyz'],frame)
                plane = seg['plane']
                data = regions.build_inventory(seg,word_data)
                if len(data['parent']):
                    query = 'The '+language.role_text(parsed[0])+'.'
                    nominations = self._frontend.score_frame(target=target,points=points,text=query,
                                                             seed=seed,waymo_view=waymo_view)
                    subject_proposals = {'t_'+name: value for name,value in phrase_bank(nominations,scene,frame).items()}
                    chosen,_ = selection.select(data,subject_proposals,words,*parsed[2:],
                                                regions.Returns(data['xyz']),read_reference=False)
                    if chosen>=0:
                        box = regions.complete(data['boxes'][chosen],data['dimensions'][chosen,words[0]])
                        return sensor_frame_box(box,target,scene.view_index)
            else:
                plane = regions.terrain(xyz)
            lower = regions.bottom(frozen,xyz,plane,regions.split_returns(xyz))
            return sensor_frame_box(regions.only_ground(frozen,lower),target,scene.view_index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root',type=Path)
    parser.add_argument('--scales',type=int,choices=(4,),default=4,help='four fixed physical scales (default)')
    parser.add_argument('--source',required=True,choices=PLATFORMS)
    parser.add_argument('--target',required=True,choices=PLATFORMS)
    parser.add_argument('--points',type=Path,required=True)
    parser.add_argument('--rgb',type=Path,required=True,help='camera image for sensor dimensions')
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--text',required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--waymo-view',choices=WAYMO_VIEWS,default='F')
    args = vars(parser.parse_args())
    with redirect_stdout(sys.stderr):
        grounder = Grounder(args.pop('source'),asset_root=args.pop('asset_root'),scales=args.pop('scales'))
        box = grounder.predict(**args)
    print(json.dumps(dict(box=box.tolist(),frame='target_lidar',length_unit='m',yaw_unit='rad'),allow_nan=False))


if __name__=='__main__':
    main()
