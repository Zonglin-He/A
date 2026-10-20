"""448-only frozen temporal replay without dead spatial decoding.

The caller retains its existing ``inputs.resolution(448)`` context. This module
captures the original norm-before prefix and first ground-decoder inputs under
native fp16 autocast, then preserves the original fp32 two-pass temporal replay.
It never supplies logits from the discarded native fp16 decoder suffix.
"""
import ast
import copy
import gc
import hashlib
import inspect
import textwrap
from contextlib import contextmanager

import torch
from torch import nn

from .capture import transfer
from .tensors import detached, floating32

QUERY_FORWARD_AST = '2f7204023b2b923f4538dfa9464f2ca336e17ebbef291b8a4339fa6897402ed9'


def verify_query_forward(decoder):
    source = textwrap.dedent(inspect.getsource(type(decoder).forward))
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    if actual != QUERY_FORWARD_AST:
        raise RuntimeError('Time-only extraction requires the audited QueryDecoder.forward')
    return actual


class TimeOnlyDecoder(nn.Module):
    """Exact live temporal statements from the audited native QueryDecoder.

    Omitted statements only construct position-query inputs or execute the
    independent PosDecoder. Six-layer temporal outputs/head batch shapes remain.
    """
    def __init__(self, original):
        super().__init__()
        self.d_model = original.d_model
        # Avoid copying PosDecoder/bbox modules or temporarily trainable LNs.
        self.time_fc = copy.deepcopy(original.time_fc)
        self.time_embed = copy.deepcopy(original.time_embed)
        self.time_decoder = copy.deepcopy(original.time_decoder)
        self.eval().requires_grad_(False)

    def forward(self, encoded_info, vis_pos=None, itq=None):
        encoded_feature = encoded_info['encoded_feature']
        encoded_mask = encoded_info['encoded_mask']
        n_vis_tokens = encoded_info['fea_map_size'][0] * encoded_info['fea_map_size'][1]
        encoded_pos = vis_pos.flatten(2).permute(2, 0, 1)
        encoded_pos_t = torch.cat([torch.zeros_like(encoded_feature[n_vis_tokens:-n_vis_tokens]), encoded_pos], dim=0)
        videos_cls = encoded_info['videos_cls']
        b = len(encoded_info['durations'])
        t = max(encoded_info['durations'])
        device = encoded_feature.device
        content_query = self.time_fc(videos_cls)
        content_query = content_query.expand(t, content_query.size(-1)).unsqueeze(1)
        query_mask = torch.zeros(b, t).bool().to(device)
        query_time_embed = self.time_embed(t).repeat(1, b, 1)
        encoded_mask = encoded_mask[:, :-n_vis_tokens]
        tgt_t = torch.zeros(t, b, self.d_model).to(device) if itq is None else itq[None, None, :].expand(t, 1, 256)
        outputs_time = self.time_decoder(
            query_tgt=tgt_t,
            query_content=content_query,
            query_time=query_time_embed,
            query_mask=query_mask,
            encoded_feature=encoded_feature[n_vis_tokens:],
            encoded_pos=encoded_pos_t,
            encoded_mask=encoded_mask,
        )
        return outputs_time


class TemporalReplay:
    """The unchanged FP32 temporal branch of SpatialReplay.reference_values."""
    def __init__(self, model, views, *, audit=True):
        if len(views) != 2:
            raise ValueError('Expected the two original offsets')
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('Temporal capture requires frozen eval weights')
        self.model = model
        self.views = floating32(views)
        self.norm = copy.deepcopy(model.ground_encoder.encoder.norm).eval().requires_grad_(False)
        self.decoder = TimeOnlyDecoder(model.ground_decoder)
        self.head = copy.deepcopy(model.temp_embed).float().eval().requires_grad_(False)
        self.keep_audit = audit
        self.temporal_inputs = []
        with torch.no_grad():
            self.zero = detached(self.reference_values())
        if len(self.temporal_inputs) != 2:
            raise RuntimeError('Temporal offsets were not both captured')

    @staticmethod
    def _indices(mask):
        value = torch.nonzero(mask.squeeze()).squeeze().tolist()
        return [value] if isinstance(value, int) else value

    def _temporal_query(self, H, fm, ft, chosen, count):
        # Appearance spatial activation only feeds isq/PosDecoder, never itq.
        with torch.no_grad():
            _, am = self.model.t_spatial_clas(fm[chosen], ft[:, :1])
        return (H[-count:].permute(1, 0, 2)[chosen] * am.unsqueeze(2)).mean((0, 1))

    def reference_values(self):
        logits, actions, gates, audit = [], [], [], []
        with torch.autocast('cuda', enabled=False):
            for offset, view in enumerate(self.views):
                H = self.norm(view['prefix'])
                info = dict(view['info'])
                info.update(encoded_feature=H, frames_cls=H.mean(0), videos_cls=H.mean(0).mean(0))
                h, w = info['fea_map_size']
                count, nf = h * w, H.shape[1]
                fm = H[-count:].permute(1, 2, 0).reshape(nf, 256, h, w).detach()
                fa = H[:count].permute(1, 2, 0).reshape(nf, 256, h, w).detach()
                ft = H[count:-count].mean(1).unsqueeze(0).detach()
                with torch.no_grad():
                    lm = self.model.t_temporal_clas(fm, ft)
                    la = self.model.s_temporal_clas(fa, ft)
                    prob = (lm.sigmoid() + la.sigmoid()) / 2
                    fallback = self._indices(prob > 0)
                    first = self._indices(prob > self.model.theta) or fallback
                    itq = self._temporal_query(H, fm, ft, first, count)
                    first_hidden = self.decoder(encoded_info=info, vis_pos=view['vis_pos'], itq=itq)
                    # Preserve all six layers through action head before [-1].
                    first_actions = self.model.action_embed(first_hidden)
                    ap = first_actions[-1].squeeze().sigmoid()
                    second = self._indices(ap > .5) or fallback
                itq = self._temporal_query(H, fm, ft, second, count)
                hidden = self.decoder(encoded_info=info, vis_pos=view['vis_pos'], itq=itq)
                self.temporal_inputs.append(detached(hidden))
                # Preserve all-layer GEMM batch shapes, not merely the final layer.
                all_logits = self.head(hidden)
                all_actions = self.model.action_embed(hidden)
                logits.append(all_logits[-1])
                actions.append(all_actions[-1])
                gates.append(dict(first=first, second=second))
                if self.keep_audit:
                    audit.append(dict(offset=offset, fallback=fallback, first=first, second=second,
                                      first_used_fallback=not bool(self._indices(prob > self.model.theta)),
                                      second_used_fallback=not bool(self._indices(ap > .5)),
                                      temporal_sampling_logits=detached([lm, la], 'cpu'),
                                      first_probability=detached(prob, 'cpu'),
                                      first_hidden=detached(first_hidden, 'cpu'),
                                      first_action_logits=detached(first_actions, 'cpu'),
                                      second_hidden=detached(hidden, 'cpu'),
                                      second_logits_all_layers=detached(all_logits, 'cpu'),
                                      second_action_logits_all_layers=detached(all_actions, 'cpu')))
        return dict(logits=logits, actions=actions, gates=gates, audit=audit,
                    temporal_only=True, fp32_replay_preserved=True)


class _PrefixCaptured(Exception):
    def __init__(self, token):
        self.token = token


@contextmanager
def _native_contexts(model):
    # Same context nesting and finally restoration as stvg.source.frozen_forward.
    from .source import transient_native_masks
    from ._visual import frozen_visual_host
    from ._attention import frozen_attention_storage
    from ._host import inactive_modules_on_host
    with transient_native_masks(model), frozen_visual_host(model), frozen_attention_storage(model), inactive_modules_on_host(model):
        yield


def _prefix_view(model, view, device):
    prefixes, grounds = [], []
    token = object()
    def norm_before(_module, args):
        prefixes.append(transfer(args[0], 'cpu'))
    def decoder_before(_module, args, kwargs):
        if args or grounds:
            raise RuntimeError('Unexpected native decoder prefix boundary')
        grounds.append(transfer(kwargs, 'cpu'))
        raise _PrefixCaptured(token)
    hooks = [model.ground_encoder.encoder.norm.register_forward_pre_hook(norm_before),
             model.ground_decoder.register_forward_pre_hook(decoder_before, with_kwargs=True)]
    try:
        with torch.autocast(device.type, dtype=torch.float16):
            try:
                model(view['videos'], view['texts'], view['targets'], iteration_rate=-1)
            except _PrefixCaptured as signal:
                if signal.token is not token:
                    raise
            else:
                raise RuntimeError('Original model did not reach the captured decoder boundary')
    finally:
        for hook in hooks:
            hook.remove()
    if len(prefixes) != 1 or len(grounds) != 1:
        raise RuntimeError('Unexpected original native encoder/decoder routing')
    return dict(prefix=prefixes[0], info=grounds[0]['encoded_info'], vis_pos=grounds[0]['vis_pos'])


@torch.no_grad()
def frozen_forward(model, frames, row, *, audit=True):
    """Drop-in three-tuple for the 448 call only; caller sets resolution(448).

    ``replay.zero['logits']`` is the existing public consumer. Additional
    ``replay.zero['audit']`` stores complete two-pass gate/head provenance.
    """
    from .inputs import make_batch, query_subject, offset_batch
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('Temporal capture requires frozen eval weights')
    forward_ast = verify_query_forward(model.ground_decoder)
    q = dict(row['input'])
    q.update(height=frames.shape[1], width=frames.shape[2])
    batch = make_batch(frames, row['frame_ids'], q, model)
    device = next(model.parameters()).device
    batch['videos'] = batch['videos'].to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    views, records = [], []
    with _native_contexts(model), query_subject(model, batch, row['parses']['subject']):
        for offset in (0, 1):
            view = offset_batch(batch, offset)
            view['videos'] = view['videos'].to(device)
            captured = _prefix_view(model, view, device)
            views.append(captured)
            records.append(dict(offset=offset, flip=False, frame_ids=view['targets'][0]['frame_ids']))
            del view, captured
            gc.collect()
            torch.cuda.empty_cache()
        gpu_views = transfer(views, device)
        replay = TemporalReplay(model, gpu_views, audit=audit)
    replay.forward_ast = forward_ast
    del views, gpu_views
    return batch, records, replay

# Explicit name for the single predictor.py 448-call replacement.
frozen_temporal_forward = frozen_forward
