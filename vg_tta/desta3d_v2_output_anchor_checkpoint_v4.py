"""Exact cached replay with stateless FFN recomputation and bounded offload.

Only language MLP calls are recomputed: attention/KV-cache operations are never
checkpointed. This preserves the native incremental cache execution.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import torch
from torch.utils.checkpoint import checkpoint
from vg_tta.desta3d_v2_output_anchor import replay_branch as replay_original
from vg_tta.desta3d_v2_output_anchor_offload_v3 import activation_offload


@contextmanager
def checkpoint_mlps(mlps):
    with ExitStack() as stack:
        for mlp in mlps:
            forward = mlp.forward
            def recompute(x, _forward=forward):
                if torch.is_grad_enabled() and x.requires_grad:
                    return checkpoint(_forward, x, use_reentrant=False,
                                      preserve_rng_state=True)
                return _forward(x)
            stack.enter_context(patch.object(mlp, 'forward', recompute))
        yield


def replay_branch(*args, **kwargs):
    model = args[0]
    mlps = [layer.mlp for layer in model.model.language_model.layers]
    assert mlps
    with activation_offload(model) as info, checkpoint_mlps(mlps):
        logits, audit = replay_original(*args, **kwargs)
    audit['lossless_activation_offload'] = dict(info)
    audit['checkpointed_stateless_language_MLPs'] = len(mlps)
    audit['attention_or_KV_checkpointed'] = False
    return logits, audit
