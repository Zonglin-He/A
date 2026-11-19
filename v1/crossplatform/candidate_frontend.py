"""Point-cloud candidate scores for single-frame v1 grounding."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import random
from tempfile import TemporaryDirectory
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = ('waymo', 'drone', 'quad')
WAYMO_VIEWS = ('F', 'FL', 'FR', 'SL', 'SR')
_LOAD_LOCK = threading.Lock()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sensor_frame_box(model_box, target, view_index=0):
    """Invert the training-time target coordinate convention. Yaw is axial, modulo pi."""
    from camera_projection import WAYMO_ANGLES, z_rotation
    box = np.asarray(model_box, dtype=np.float64).copy()
    if target == 'drone':
        box[2] -= 1.8
    elif target == 'waymo':
        angle = WAYMO_ANGLES[view_index]
        box[:3] = box[:3] @ z_rotation(angle)
        box[6] = (box[6] - np.deg2rad(angle) + np.pi / 2) % np.pi - np.pi / 2
    elif target != 'quad':
        raise ValueError(f'Unknown target platform: {target}')
    return box


def orient_box(sampled, box):
    """Estimate yaw using PCA to the same points used for the 3EED prediction."""
    from crossplatform.replication36_surface.common import refine
    box = np.asarray(box, dtype=np.float32)
    if (box[3:6] <= 0).any():
        # The historical metric canonically floors sizes before evaluating such boxes.
        result = np.r_[box, np.float32(0)]
        result[3:6] = np.maximum(result[3:6], 1e-6)
        return result
    crop = sampled[np.all(np.abs(sampled - box[:3]) <= box[3:6] / 2 + 1e-7, axis=1)]
    return refine(crop, box)[1]


@dataclass
class ScoredFrame:
    boxes: np.ndarray
    probability: np.ndarray
    sampled_points: np.ndarray
    target: str
    view_index: int


class CandidateFrontend:
    """Load one source. Reuse this object for independent frames and descriptions."""
    def __init__(self, source, asset_root=None):
        if source not in PLATFORMS:
            raise ValueError(f'Unknown source platform: {source}')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        self.asset_root = Path(asset_root).resolve(strict=True) if asset_root is not None else ROOT
        code_paths = ('research/round158', 'research/round97',
                      'research/round66', 'crossplatform/round51', 'crossplatform/round7/experiment')
        dependencies = ('crossplatform/round3/p0/upstream', 'crossplatform/round20/runtime')
        upstream = self.asset_root / dependencies[0]
        if not (upstream / 'models/bdetr.py').is_file():
            raise FileNotFoundError(f'Install the compiled 3EED runtime at {upstream}')
        sys.path[:0] = [str(ROOT), *(str(ROOT / p) for p in code_paths),
                        *(str(self.asset_root / p) for p in dependencies)]
        os.environ.update(TOKENIZERS_PARALLELISM='false', TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1')
        import torch
        from models import BeaUTyDETR
        from model51 import strict_runtime
        import parse97

        if not torch.cuda.is_available():
            raise RuntimeError('The frozen PointNet++ operators require CUDA in the project environment')
        self.source = source
        self._lock = threading.Lock()
        store = json.loads((ROOT / 'crossplatform/single_frame_sources.json').read_text())[source]
        checkpoint, matcher = (self.asset_root / store[k] for k in ('epoch100', 'matcher'))
        for key, path in (('epoch100', checkpoint), ('matcher', matcher)):
            if file_sha256(path) != store[key + '_sha256']:
                raise ValueError(f'{source} {key} does not match the registered source weights')
        with _LOAD_LOCK:
            # The upstream model constructor resolves its RoBERTa assets relative to cwd.
            roberta = self.asset_root / 'crossplatform/upstream/data/roberta_base'
            if not (roberta / 'model.safetensors').is_file():
                raise FileNotFoundError(f'Install the registered RoBERTa tokenizer and weights at {roberta}')
            cwd = Path.cwd()
            with TemporaryDirectory(prefix='3eed-v1-model-') as work:
                work = Path(work)
                (work / 'data').mkdir()
                (work / 'data/roberta_base').symlink_to(roberta, target_is_directory=True)
                try:
                    os.chdir(work)
                    random.seed(0)
                    np.random.seed(0)
                    torch.manual_seed(0)
                    torch.backends.cudnn.enabled = True
                    strict_runtime()
                    torch.use_deterministic_algorithms(True)
                    backbone = BeaUTyDETR(num_class=256, num_obj_class=485, input_feature_dim=3,
                                          num_queries=256, num_decoder_layers=6,
                                          self_position_embedding='loc_learned',
                                          contrastive_align_loss=True, butd=False,
                                          pointnet_ckpt=None, self_attend=True)
                finally:
                    os.chdir(cwd)
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if state['epoch'] != 100 or state['trained_platforms'] != [source]:
                raise ValueError('Checkpoint provenance differs from the selected source')
            backbone.load_state_dict(state['model'], strict=True)
            del state
            self.base = backbone.cuda().eval().requires_grad_(False)
            from torch import nn

            self.point_adapter = nn.Sequential(nn.Linear(288, 16), nn.GELU(), nn.Linear(16, 288))
            self.match_projection = deepcopy(self.base.contrastive_align_projection_image)
            with np.load(matcher, allow_pickle=False) as prepared:
                self.point_adapter.load_state_dict({
                    name: torch.from_numpy(prepared['point_adapter.' + key])
                    for name, key in (
                        ('0.weight', 'down.weight'), ('0.bias', 'down.bias'),
                        ('2.weight', 'up.weight'), ('2.bias', 'up.bias'),
                    )
                }, strict=True)
                self.match_projection.load_state_dict({
                    name: torch.from_numpy(prepared['match.' + name])
                    for name in self.match_projection.state_dict()
                }, strict=True)
            self.point_adapter.cuda().eval().requires_grad_(False)
            self.match_projection.eval().requires_grad_(False)
            assets = self.asset_root / 'crossplatform/round20/assets'
            kwargs = json.loads((assets / 'STANZA_ASSET_RECEIPT.json').read_text())['kwargs']
            for key in kwargs:
                if key.endswith('_path'):
                    kwargs[key] = str(assets / kwargs[key])
            self.parser = parse97.stanza.Pipeline(
                'en', dir=str(assets / 'stanza'), processors='tokenize,mwt,pos,lemma,depparse',
                package=None, use_gpu=False, download_method=None, verbose=False, **kwargs)

    def _point_probability(self, raw):
        """Replay the prepared point-only matcher on the original query positions."""
        import torch
        from torch.nn.functional import normalize

        device = next(self.base.parameters()).device

        def tensor(key):
            return torch.as_tensor(raw[key], device=device)

        base = self.base
        points = tensor('vis0')[0]
        text_padding = tensor('textpad')
        vision, language = base.cross_encoder(
            vis_feats=(points + self.point_adapter(points))[None],
            pos_feats=tensor('pos0'),
            padding_mask=torch.zeros((1, len(points)), device=device, dtype=torch.bool),
            text_feats=tensor('lang0'), text_padding_mask=text_padding, end_points={},
            detected_feats=None, detected_mask=None,
        )
        indices = tensor('query_points_sample_inds').long()[None]
        _, seeds, _ = base.gsample_module(
            tensor('xyz'), vision.transpose(1, 2).contiguous(), indices.int())
        query = base.decoder_query_proj(seeds).transpose(1, 2).contiguous()
        for index, layer in enumerate(base.decoder):
            query = layer(query, vision, language, tensor('position' + str(index)), None,
                          text_padding, detected_feats=None, detected_mask=None)
        with torch.autocast(device_type='cuda', enabled=False):
            q = normalize(self.match_projection(query.float())[0], dim=-1)
            tokens = normalize(base.contrastive_align_projection_text(language.float()), dim=-1)[0]
            logits = q @ tokens.T / .07
            subject = tensor('token_mask').bool()
            odds = (torch.logsumexp(logits[:, subject], dim=-1)
                    - torch.logsumexp(logits[:, ~subject], dim=-1))
        return odds.double().softmax(0)

    def score_frame(self, *, target, points, text, seed=0, waymo_view='F'):
        """Return A256 and point-cloud/language candidate probabilities."""
        import torch
        from interface66 import capture
        from model51 import SENTINEL
        from runner_helpers import sample_points
        from utils.transform_waymo import transform_to_front_view
        if target not in PLATFORMS or waymo_view not in WAYMO_VIEWS or (target != 'waymo' and waymo_view != 'F'):
            raise ValueError('Check target platform and Waymo camera view')
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError('Point sampling seed must be a nonnegative integer')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Provide a nonempty English referring expression')
        caption = text.strip()
        if len(self.base.tokenizer(caption + SENTINEL)['input_ids']) > 256:
            raise ValueError('Expression exceeds the native 256-token readout capacity')
        view = WAYMO_VIEWS.index(waymo_view)
        row = dict(platform=target, seed=int(seed), pointpath=Path(points).resolve(strict=True))
        with self._lock, torch.inference_mode():
            sampled, indices, _ = sample_points(row)
            if target == 'waymo':
                sampled, _ = transform_to_front_view(sampled, np.empty((0, 7)), waymo_view)
                sampled = np.ascontiguousarray(sampled, dtype=np.float32)
            raw, _ = capture(self.base, sampled, indices, caption)
            probability = self._point_probability(raw)
            return ScoredFrame(raw['boxes'], probability.cpu().numpy(), sampled, target, view)

