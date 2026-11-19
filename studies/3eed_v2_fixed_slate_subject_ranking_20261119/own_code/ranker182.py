"""Source-trained candidate residual using frozen v1 subject evidence."""

from pathlib import Path
import sys
import threading

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
R181 = HERE.parent / '3eed_v2_readout_calibration_20261119'
sys.path.insert(0, str(R181))

from readout181 import ComponentReader, K, ReadoutGrounder
from scope import read

WIDTH = 299
ARMS = ('v1', 'frozen_joint', 'frozen_subject', 'frozen_mismatch',
        'rank_subject', 'rank_joint', 'rank_mismatch', 'generic_joint')


def protocol():
    return read(HERE / 'PROTOCOL.json')


class FrozenSubjectReader(ComponentReader):
    def _read(self, *args):
        out = super()._read(*args)
        self.last_support = tuple(out[j][0, :K] for j in (0, 2, 3))
        return out

    def inspect(self, evidence):
        result = super().inspect(evidence)
        if not bool(evidence['has_reference'][0]):
            self.subject_support = self.last_support
        return result


def features(native, capture, masks, anchors, support):
    query = F.normalize(native['queries'][0].float(), dim=-1)
    tokens = native['tokens'][0].float()
    offsets = capture['offsets']
    valid_text = (offsets[:, 1] > offsets[:, 0]) & (offsets[:, 0] < capture['caption_length'])
    content = torch.as_tensor(valid_text & capture['valid'], device=query.device)
    selections = torch.stack((masks[0, 0], content)).float()
    reference_text = F.normalize(selections @ tokens, dim=-1)
    word, full = query * reference_text[0], query * reference_text[1]
    state, pixel, mass = support
    pose = anchors[:, :3].sign() * torch.log1p(anchors[:, :3].abs()) / 4
    shape = anchors[:, 3:6].clamp_min(.05).log()
    out = torch.cat((F.layer_norm(state.float(), (96,)), 8 * query, 16 * word, 16 * full,
                     pose, shape, (2 * pixel - 1) * mass[:, None], mass[:, None],
                     4 * word.sum(-1, keepdim=True), 4 * full.sum(-1, keepdim=True)), dim=-1)
    if out.shape != (K, WIDTH) or not torch.isfinite(out).all():
        raise ValueError('Invalid frozen subject features')
    return out.cpu().numpy()


class FeatureGrounder(ReadoutGrounder):
    """Reuse the pinned R181 reads and capture only frozen subject features."""

    def __init__(self, source, *, asset_root=None):
        super().__init__(source, asset_root=asset_root)
        self._lock = threading.RLock()
        self.reader = FrozenSubjectReader().cuda().eval().requires_grad_(False)
        self.reader.load_state_dict(self._decoder.state_dict(), strict=True)

    def _read_observation(self, frame, *, capture=None):
        capture = {} if capture is None else capture
        scored, evidence = super()._read_observation(frame, capture=capture)
        capture['caption_length'] = len(frame['text'].strip())
        self._subject_capture = capture, evidence['role_mask']
        return scored, evidence

    @torch.inference_mode()
    def collect(self, frame, *, fixed=None, ranking=None):
        with self._lock:
            self.reader.subject_support = None
            try:
                row = super().collect(frame, fixed=fixed, ranking=ranking)
                capture, masks = self._subject_capture
                raw = capture['native']
                query_ids = torch.as_tensor(row['slate'], device=raw['queries'].device)
                native = dict(queries=raw['queries'][:, query_ids], tokens=raw['tokens'])
                anchors = torch.as_tensor(row['boxes'], device=query_ids.device)
                row['features'] = features(native, capture, masks, anchors, self.reader.subject_support)
                return row
            finally:
                self._subject_capture = None
                self.reader.subject_support = None
                self.reader.last_support = None


class SubjectRanker(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = protocol()['ranker']
        width = cfg['hidden_width']
        self.score = nn.Sequential(nn.Linear(WIDTH, width), nn.GELU(), nn.Dropout(cfg['dropout']),
                                   nn.Linear(width, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, evidence):
        return self.score(evidence).squeeze(-1)


def choose(scores, incumbent):
    row = np.arange(len(scores))
    winner = np.argmax(scores, axis=1)
    return np.where(scores[row, incumbent] == scores[row, winner], incumbent, winner)


def readouts(residual, generic, frozen, incumbent):
    # Use the exact R181 scalar scores as the skip connection.
    subject = frozen[:, :, 1] + residual
    joint = frozen[:, :, 0] + residual
    mismatch = frozen[:, :, 2] + residual
    generic_joint = frozen[:, :, 0] + generic
    return {name: choose(score, incumbent) for name, score in (
        ('rank_subject', subject), ('rank_joint', joint),
        ('rank_mismatch', mismatch), ('generic_joint', generic_joint))}
