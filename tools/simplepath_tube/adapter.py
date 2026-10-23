"""Observe/replay the exact native visual prefix; never fabricate a time prior.

Only this isolated worker patches its own frozen model instance. Native encoder,
decoder, last-layer logits, Source MAP and eight observation policy stay native.
"""
from contextlib import contextmanager
import hashlib
import numpy as np
import torch
from stvg.tubedetr import native_forward, endpoint_posterior
from stvg.positions import observation_positions
from stvg.initialization import initialize
from stvg.predictor import output_writeback
from stvg.temporal import final_mbr
from tools.boxfixed_tube.evidence import _valid_grid_shape
from tools.llava_st.evidence import SourceEvidence
from .core.spatial import build_geometry, evidence_dp
from .core.local import LocalFrame, grid_regions, aggregate_sequence
from .core.prototype import one_step


def cpu(x):
    return x.detach().cpu().clone()


@contextmanager
def observe_prefix(model, frame_count):
    """Capture actual projected visual cells and native masks/positions pre-text."""
    backbones, projected = [], []
    def backbone_hook(module, args, output):
        features, positions = output
        _, mask = features[-1].decompose()
        backbones.append((cpu(mask), cpu(positions[-1])))
    def projection_hook(module, args, output):
        projected.append(cpu(output))
    handles = [model.backbone.register_forward_hook(backbone_hook),
               model.input_proj.register_forward_hook(projection_hook)]
    value = {}
    try:
        yield value
        if not model.fast or model.stride != 2:
            raise ValueError('This adapter requires the original full-time fast branch and stride2')
        if len(backbones) != 2 or len(projected) != 2:
            raise ValueError('Unexpected native visual prefix call structure')
        # Native calls backbone slow, fast; projection fast, slow.
        slow, fast = projected[1], projected[0]
        if len(fast) != frame_count or len(slow) != (frame_count + 1) // 2:
            raise ValueError('Real full-time projected grid unavailable')
        for name, tensor, (mask, pos) in zip(['slow', 'fast'], [slow, fast], backbones):
            if tensor.shape != pos.shape or tensor.shape[0:1] + tensor.shape[2:] != mask.shape:
                raise ValueError('Native cell/mask/position correspondence differs')
            value.update({name + '_grid': tensor, name + '_mask': mask, name + '_pos': pos})
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def replay_prefix(model, prefix):
    """Replay exact projected tensors, bypassing only the visual backbone/proj.

    All native video/text encoder, temporal and box decoder operations still run.
    The two call orders and input batch sizes are checked before accepting reuse.
    """
    from util.misc import NestedTensor
    old_backbone, old_projection = model.backbone.forward, model.input_proj.forward
    calls = []
    def backbone(samples):
        if len(calls) >= 2:
            raise ValueError('Unexpected extra backbone call')
        name = ['slow', 'fast'][len(calls)]
        grid = torch.as_tensor(prefix[name + '_grid'], device=samples.tensors.device)
        mask = torch.as_tensor(prefix[name + '_mask'], device=grid.device)
        pos = torch.as_tensor(prefix[name + '_pos'], device=grid.device)
        if len(grid) != len(samples.tensors):
            raise ValueError('Cached prefix physical support differs')
        calls.append(name)
        return [NestedTensor(grid, mask)], [pos]
    model.backbone.forward = backbone
    model.input_proj.forward = lambda x: x
    try:
        yield
        if calls != ['slow', 'fast']:
            raise ValueError('Prefix replay did not match the native call contract')
    finally:
        model.backbone.forward, model.input_proj.forward = old_backbone, old_projection


@torch.no_grad()
def capture(model, frames, ids, caption, *, prefix=None):
    cached = prefix is not None
    if prefix is None:
        with observe_prefix(model, len(ids)) as collected:
            native = native_forward(model, frames, ids, caption, 224)
        prefix = collected
    else:
        with replay_prefix(model, prefix):
            native = native_forward(model, frames, ids, caption, 224)
    args, kw = native['decoder_inputs']
    memory, pos = cpu(args[1]).float(), cpu(kw['pos']).float()
    padding = cpu(kw['memory_key_padding_mask']).bool()
    flags = pos.abs().sum(-1).any(-1)
    n = int(flags.sum())
    if n <= 0 or not flags[:n].all() or flags[n:].any():
        raise ValueError('Spatial tokens are not a validated visual prefix')
    shape = _valid_grid_shape([pos], [padding], n)
    valid = ~padding[:, :n]
    features = memory[:n].transpose(0, 1)
    features = torch.where(valid[:, :, None], features, torch.zeros_like(features))
    positions = observation_positions(ids, native['extent'])
    q0 = endpoint_posterior(native['logits'])
    return dict(Source=native['source'].numpy(), Native_interval=native['interval'],
                frame_ids=list(map(int, ids)), positions8=positions, features=features,
                valid=valid, grid_shape=shape, native_logits6=torch.stack(native['layers']),
                native_prior=q0, visual_prefix=prefix,
                prefix_cache_hit=cached)


def finish(capture, teacher, configuration):
    ids, positions = capture['frame_ids'], capture['positions8']
    features = capture['features']
    evidence = SourceEvidence.from_native_masks(features, capture['valid'], ids, positions,
                teacher['mask_logits'], teacher['teacher_image_size'], capture['grid_shape'],
                selected_query=teacher['selected_query'])
    boxes = teacher['boxes'].double().numpy()
    raw, init = initialize(capture['Source'].astype(np.float64), positions, boxes,
                           configuration['E_calibration'])
    E = output_writeback(capture['Source'], raw, configuration['initialization_gamma'])
    geometry = build_geometry(evidence, torch.from_numpy(E), torch.from_numpy(boxes))
    aux = evidence_dp(geometry.for_interval([ids[0], ids[-1] + 1]))['final'].numpy()
    prefix = capture['visual_prefix']
    grid = np.asarray(prefix['fast_grid'])
    valid = ~np.asarray(prefix['fast_mask'])
    T, C, H, W = grid.shape
    if T != len(ids) or C != 256:
        raise ValueError('Explicit full-time native pre-text grid required')
    regions = grid_regions(H, W)
    local = [LocalFrame(float(t), grid[j].transpose(1, 2, 0).reshape(-1, C),
                        regions, valid[j].ravel(), (H, W)) for j, t in enumerate(ids)]
    z, support = aggregate_sequence(local, aux, 1.5)
    posterior, trace = one_step(z, capture['native_prior'])
    interval = final_mbr(ids, posterior)['interval']
    native_interval = capture['Native_interval']
    spatial = evidence_dp(geometry.for_interval(native_interval))['final'].numpy()
    full = evidence_dp(geometry.for_interval(interval))['final'].numpy()
    return dict(Source=capture['Source'], Spatial=spatial, Temporal=capture['Source'], Full=full,
                frame_ids=np.asarray(ids), positions8=np.asarray(positions),
                Native_interval=np.asarray(native_interval), Selected_interval=np.asarray(interval),
                Native_MBR_interval=np.asarray(final_mbr(ids, capture['native_prior'])['interval']),
                native_logits6=np.asarray(capture['native_logits6']), native_prior=capture['native_prior'],
                posterior=posterior, auxiliary_path=aux, pooled_pretext=z,
                frame_feedback=trace['L'], initialization=init, support=support,
                new_teacher_calls=0, network_parameter_updates=0)
