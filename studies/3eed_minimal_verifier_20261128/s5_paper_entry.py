"""Registered S5-pass entry consuming the two frozen Qwen replies."""
from functools import lru_cache

import numpy as np
from safetensors.torch import load_file
import torch

import live_2d  # Initialize the frozen geometric dependencies.
from provenance import V2  # Initialize the v2 geometry import paths.
from lean_geometry import describe, reduced_pool
from models import Arm, model_for
from prepare_blind import prepare


@lru_cache(maxsize=1)
def _verifier(checkpoint, cap):
    model = model_for(Arm(cap, mean=False, utility=True, lean=True))
    model.load_state_dict(load_file(checkpoint))
    return model.cuda().eval()


@torch.inference_mode()
def predict(frame, replies, *, checkpoint, cap, threshold, baseline=None):
    from minimal import predict as minimal_d

    base = minimal_d(frame, replies) if baseline is None else baseline
    row = dict(frame=frame, raw=replies, target=frame['target'])
    case = prepare((frame['uid'], row, dict(outputs=dict(axis_2d=base))))
    pool, evidence = reduced_pool(case, frame, replies, cap)
    k = len(pool)
    if k == 1:
        return dict(base, verifier_replaced=False, verifier_candidates=k, verifier_utility=None)

    x, e = describe(evidence, pool)
    nodes = np.zeros((1, cap, 14), np.float32)
    edges = np.zeros((1, cap, cap, 8), np.float32)
    nodes[0, :k], edges[0, :k, :k] = x, e
    mask = (np.arange(cap) < k)[None]
    tensors = (torch.as_tensor(a, device='cuda') for a in (nodes, edges, mask))
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    try:
        utility = _verifier(checkpoint, cap)(*tensors)[0, 1:k]
    finally:
        torch.set_float32_matmul_precision(precision)
    value, candidate = utility.max(0)
    score = float(value)
    use = score > threshold
    return dict(base, box=pool[int(candidate)+1] if use else base['box'],
                verifier_replaced=use, verifier_candidates=k, verifier_utility=score)
