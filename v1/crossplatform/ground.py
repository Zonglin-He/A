"""Single-frame cross-platform 3D visual grounding with the v1 decoder."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import threading

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from crossplatform.candidate_frontend import (
    PLATFORMS, WAYMO_VIEWS, CandidateFrontend, file_sha256, sensor_frame_box,
)
from crossplatform.v1_decoder import JointDecoder
from crossplatform.v1_evidence import decoder_evidence, role_masks

HERE = Path(__file__).resolve().parent


def load_decoder(source, device, asset_root):
    registered = json.loads((HERE / 'CURRENT_METHOD.json').read_text())['decoders'][source]
    checkpoint = asset_root / 'crossplatform' / registered['path']
    if file_sha256(checkpoint) != registered['sha256']:
        raise ValueError(f'{source} decoder does not match the registered weights')
    decoder = JointDecoder()
    decoder.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return decoder.to(device).eval().requires_grad_(False)


class Grounder:
    """Use one frozen source model for repeated single-frame predictions."""

    def __init__(self, source, *, asset_root=None):
        self._lock = threading.Lock()
        self._frontend = CandidateFrontend(source, asset_root=asset_root)
        self._decoder = load_decoder(source, next(self._frontend.base.parameters()).device,
                                     self._frontend.asset_root)

    @torch.inference_mode()
    def _read_observation(self, frame, *, capture=None):
        from model51 import SENTINEL
        from parse97 import dependency_document, parse85, parse_subject, repair_inventory87

        frontend = self._frontend
        observed = {}

        def record_native(_module, _inputs, raw):
            if observed:
                raise RuntimeError('Expected one frozen scene forward per observation')
            observed.update(points=raw['fp2_features'].transpose(1, 2).contiguous().half(),
                            xyz=raw['fp2_xyz'], text=raw['text_feats'].half(),
                            queries=raw['last_proj_queries'], tokens=raw['proj_tokens'],
                            token_ids=raw['tokenized']['input_ids'])
            if capture is not None:
                observed['objectness'] = raw['seeds_obj_cls_logits'].squeeze(1)

        handle = frontend.base.register_forward_hook(record_native)
        try:
            scored = frontend.score_frame(target=frame['target'], points=frame['points'],
                                          text=frame['text'], seed=frame['seed'],
                                          waymo_view=frame['waymo_view'])
        finally:
            handle.remove()
        caption = frame['text'].strip()
        encoded = frontend.base.tokenizer([caption + SENTINEL], padding='longest',
                                          return_offsets_mapping=True, return_tensors='np')
        if (encoded['input_ids'] != observed['token_ids'].cpu().numpy()).any():
            raise RuntimeError('Role spans refer to a different tokenizer sequence')
        document = dependency_document(frontend.parser(caption))
        roles = repair_inventory87(caption, document,
                                   parse85(caption, document, parse_subject(caption, document)))
        valid = encoded['attention_mask'][0].astype(bool)
        masks, has_reference = role_masks(caption, roles, document, encoded['offset_mapping'][0], valid)
        if capture is not None:
            capture.update(native=observed, graph=roles, offsets=encoded['offset_mapping'][0],
                           valid=valid, document=document)
        evidence = decoder_evidence(scored, observed, masks, has_reference, valid,
                                    frame['rgb'], frame['calibration'])
        return scored, evidence

    @torch.inference_mode()
    def predict(self, *, target, points, rgb, calibration, text, seed=0, waymo_view='F'):
        """Return the selected seven-value box in the target LiDAR coordinate system."""
        frame = dict(target=target, points=points, rgb=rgb, calibration=calibration,
                     text=text, seed=seed, waymo_view=waymo_view)
        with self._lock:
            scored, evidence = self._read_observation(frame)
            result = self._decoder(evidence)
            box = result['boxes'][0, 0].cpu().numpy()
        return sensor_frame_box(box, scored.target, scored.view_index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, help='root of the local 3EED runtime and weights')
    parser.add_argument('--source', required=True, choices=PLATFORMS)
    parser.add_argument('--target', required=True, choices=PLATFORMS)
    parser.add_argument('--points', type=Path, required=True, help='.npy Nx3+ or raw float32 XYZI')
    parser.add_argument('--rgb', type=Path, required=True, help='camera image for dimensions only')
    parser.add_argument('--calibration', type=Path, required=True, help='3EED camera JSON')
    parser.add_argument('--text', required=True, help='English referring expression')
    parser.add_argument('--seed', type=int, default=0, help='LiDAR sampling seed')
    parser.add_argument('--waymo-view', choices=WAYMO_VIEWS, default='F', help='target Waymo camera')
    args = vars(parser.parse_args())
    with redirect_stdout(sys.stderr):
        model = Grounder(args.pop('source'), asset_root=args.pop('asset_root'))
        box = model.predict(**args)
    print(json.dumps(dict(box=box.tolist(), frame='target_lidar', length_unit='m', yaw_unit='rad'),
                     allow_nan=False))


if __name__ == '__main__':
    main()
