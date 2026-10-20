"""Original frozen capture with host staging between native offsets.

Only tensor residency changes. All normalized inputs, original model calls,
hooks, offset order, dtypes and complete replay are retained.
"""
import gc
import torch


def transfer(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: transfer(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [transfer(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(transfer(v, device) for v in value)
    return value


@torch.no_grad()
def frozen_forward(model, frames, row):
    from stvg.inputs import make_batch, query_subject, offset_batch
    from stvg.source_replay import SpatialReplay
    from stvg.tensors import detached
    q = dict(row['input'])
    q.update(height=frames.shape[1], width=frames.shape[2])
    batch = make_batch(frames, row['frame_ids'], q, model)
    device = next(model.parameters()).device
    # The original GPU-normalized video is copied without any recomputation.
    # Only one original offset's video resides on CUDA during its full forward.
    batch['videos'] = batch['videos'].to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    views, records = [], []
    with query_subject(model, batch, row['parses']['subject']):
        for offset in (0, 1):
            prefixes, grounds = [], []
            hooks = [
                model.ground_encoder.encoder.norm.register_forward_pre_hook(
                    lambda m, a: prefixes.append(transfer(a[0], 'cpu'))),
                model.ground_decoder.register_forward_pre_hook(
                    lambda m, a, kw: grounds.append(transfer(kw, 'cpu')) if not grounds else None,
                    with_kwargs=True),
            ]
            view = offset_batch(batch, offset)
            view['videos'] = view['videos'].to(device)
            try:
                with torch.autocast('cuda', dtype=torch.float16):
                    model(view['videos'], view['texts'], view['targets'], iteration_rate=-1)
            finally:
                for hook in hooks:
                    hook.remove()
            if len(prefixes) != 1 or len(grounds) != 1:
                raise RuntimeError('Unexpected original native encoder/decoder routing')
            views.append(dict(prefix=prefixes[0], info=grounds[0]['encoded_info'],
                              vis_pos=grounds[0]['vis_pos']))
            records.append(dict(offset=offset, flip=False,
                                frame_ids=view['targets'][0]['frame_ids']))
            del view, prefixes, grounds, hooks
            gc.collect()
            torch.cuda.empty_cache()
        gpu_views = transfer(views, device)
        replay = SpatialReplay(model, gpu_views, len(row['frame_ids']))
    del views, gpu_views
    return batch, records, replay
