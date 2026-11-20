"""Shared frozen-v1 capture and the two matched final-subject readouts."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE.parent / '3eed_v1_supervised_instance_grounding_20261119'
ORDINARY = HERE.parent / '3eed_v2_ordinary_val_subject_ranking_20261119'
WORK = HERE / 'run_PRIVATE'
PUBLIC = HERE / 'results'
ARMS = ('v1', 'mean', 'worst')
sys.path[:0] = [str(ROOT), str(ORDINARY), str(SOURCE), str(ROOT / 'research/round158'),
                str(ROOT / 'research/round6/source_validation_scoring')]
from scope183 import atomic, digest, read
from reader158 import overlaps, probability
from batch_yaw_iou import many_iou
from crossplatform.ground import Grounder


def initialize():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    WORK.mkdir(exist_ok=True, mode=0o700)
    PUBLIC.mkdir(exist_ok=True)
    cfg = read(HERE / 'PROTOCOL.json')
    assert all(digest(ROOT / name) == pin for name, pin in cfg['pins'].items())
    return cfg


def archive_npz(path, **values):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temp.replace(path)


def anon(*values):
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def engine():
    model = Grounder('waymo')
    model._frontend.parser = lru_cache(maxsize=6000)(model._frontend.parser)
    return model


def affine(model):
    last = model._decoder.slot_score[-1]
    return torch.cat((last.weight.detach().cpu().reshape(-1), last.bias.detach().cpu()))


@torch.inference_mode()
def capture(decoder, evidence):
    hidden = []
    hook = decoder.slot_score[-1].register_forward_pre_hook(lambda _, args: hidden.append(args[0]))
    try:
        result = decoder(evidence)
    finally:
        hook.remove()
    assert len(hidden) == 2 and hidden[-1].shape == (1, 8, 96)
    return (hidden[-1][0, :4].cpu().numpy().copy(),
            result['subject_logits'][0].cpu().numpy().copy())


def logit(features, original, baseline, head):
    """Equivalent to changing just the last affine on second-round subject positions."""
    before = F.linear(features, original[:-1][None], original[-1:]).squeeze(-1)
    after = F.linear(features, head[:-1][None], head[-1:]).squeeze(-1)
    return baseline + .75 * (after.tanh() - before.tanh())


@torch.inference_mode()
def direct(decoder, evidence, head):
    """Independent check: apply the subject edit inside the real decoder."""
    calls = []
    def replace(_, args, out):
        calls.append(1)
        if len(calls) == 2:
            out = out.clone()
            out[:, :4] = F.linear(args[0][:, :4], head[:-1][None], head[-1:])
        return out
    handle = decoder.slot_score[-1].register_forward_hook(replace)
    try:
        scores = decoder(evidence)['subject_logits'][0].cpu().numpy()
    finally:
        handle.remove()
    assert len(calls) == 2
    return scores


def fixed_prior(anchors, scene):
    """Evaluate the same overlap-vote function at fixed, raw center/size boxes."""
    raw = scene.boxes.astype(np.float64)
    fixed = np.asarray(anchors, dtype=np.float64)[:, :6]
    side = raw[:, 3:].clip(min=0)
    fs = fixed[:, 3:].clip(min=0)
    lo, hi = raw[:, :3] - side / 2, raw[:, :3] + side / 2
    fl, fh = fixed[:, :3] - fs / 2, fixed[:, :3] + fs / 2
    inter = np.maximum(0, np.minimum(fh[:, None], hi) - np.maximum(fl[:, None], lo)).prod(-1)
    union = fs.prod(-1)[:, None] + side.prod(-1) - inter
    cross = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    kernel = overlaps(raw)
    mass = probability(np.log(np.maximum(scene.probability.astype(np.float64), 1e-300))
                       - np.log((kernel * kernel).sum(1)))
    maximum = np.log(max(float((kernel @ mass).max()), 1e-80))
    val = cross @ mass
    return (np.log(np.maximum(val, 1e-80)) - maximum).clip(-20, 0).astype(np.float32), val > 0


def pair_margins(scores, slots):
    idx = torch.arange(len(scores))
    a, b = slots.T
    return torch.stack((scores[idx, 0, a] - scores[idx, 0, b],
                        scores[idx, 1, b] - scores[idx, 1, a]), dim=-1)
