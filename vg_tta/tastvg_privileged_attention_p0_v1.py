"""Frozen spatial attention intervention; no loss, optimizer or parameter write."""
from contextlib import contextmanager
import torch


def proposal_support(boxes, scores, threshold=.35):
    boxes = torch.as_tensor(boxes, dtype=torch.float32).detach().cpu().reshape(-1, 4)
    scores = torch.as_tensor(scores, dtype=torch.float32).detach().cpu()
    assert boxes.shape == (len(scores), 4)
    xy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                    boxes[:, :2] + boxes[:, 2:] / 2), -1).clamp(0, 1)
    valid = torch.isfinite(boxes).all(-1) & torch.isfinite(scores)
    valid &= (xy[:, 2:] > xy[:, :2]).all(-1) & (scores >= threshold)
    indices = torch.nonzero(valid).flatten()
    xy = xy[indices]
    clipped = torch.cat(((xy[:, :2] + xy[:, 2:]) / 2, xy[:, 2:] - xy[:, :2]), -1)
    weights = torch.softmax(scores[indices], 0) if len(indices) else torch.empty(0)
    return dict(indices=indices, boxes=clipped, scores=scores[indices], weights=weights)


def evidence_field(support, height, width, padding=None):
    """Spatial grid points in original image coordinates; padding is not masked."""
    pad = torch.zeros(height, width, dtype=torch.bool) if padding is None else padding.reshape(height, width).bool().cpu()
    valid = ~pad
    vh = int(valid.any(1).sum()); vw = int(valid.any(0).sum())
    assert vh > 0 and vw > 0
    rectangular = torch.arange(height)[:, None].lt(vh) & torch.arange(width)[None, :].lt(vw)
    assert torch.equal(valid, rectangular), 'Only native right/bottom padding is supported'
    if not len(support['boxes']):
        return dict(field=torch.ones(height * width), active=False, valid=valid.flatten())
    yy, xx = torch.meshgrid((torch.arange(height) + .5) / vh,
                            (torch.arange(width) + .5) / vw, indexing='ij')
    boxes = support['boxes']; sigma = boxes[:, 2:] / 2
    g = torch.exp(-.5 * (((xx[..., None] - boxes[:, 0]) / sigma[:, 0]) ** 2
                         + ((yy[..., None] - boxes[:, 1]) / sigma[:, 1]) ** 2))
    field = (g * support['weights']).sum(-1)
    assert torch.isfinite(field).all() and field.min() >= 0 and field.max() <= 1 + 1e-6
    field[pad] = 1
    return dict(field=field.flatten(), active=True, valid=valid.flatten())


class AttentionCapture:
    """Use the native additive-mask interface without altering official source."""
    def __init__(self, layer, fields, visual_tokens, *, alpha=1., epsilon=1e-6):
        self.layer = layer; self.fields = fields; self.visual_tokens = visual_tokens
        self.alpha = alpha; self.epsilon = epsilon; self.record = None; self.hooks = []

    def before(self, module, args, kwargs):
        assert not args and kwargs.get('attn_mask') is None and kwargs.get('key_padding_mask') is None
        assert module.in_proj_weight is None and module.in_proj_bias is None
        assert module.bias_k is None and not module.add_zero_attn and not module.training
        q, k = kwargs['query'], kwargs['key']; frames = q.shape[1]; heads = module.num_heads
        assert q.shape[0] == 1 and k.shape[1] == frames and len(self.fields) == frames
        bias = torch.zeros(frames, k.shape[0], device=q.device, dtype=q.dtype)
        for i, field in enumerate(self.fields):
            if field['active'] and self.alpha != 0:
                z = self.alpha * torch.log(field['field'].to(q) + self.epsilon)
                z = torch.where(field['valid'].to(q.device), z, torch.zeros_like(z))
                bias[i, :self.visual_tokens] = z
        d = q.shape[-1] // heads
        qq = (q * d ** -.5).contiguous().view(1, frames * heads, d).transpose(0, 1)
        kk = k.contiguous().view(-1, frames * heads, d).transpose(0, 1)
        raw = torch.bmm(qq, kk.transpose(1, 2)).reshape(frames, heads, -1)
        self.record = dict(raw_logits=raw.detach().cpu(), bias=bias.detach().cpu(),
                           visual_tokens=self.visual_tokens, heads=heads, layer=self.layer)
        kwargs = dict(kwargs)
        if torch.count_nonzero(bias):
            kwargs['attn_mask'] = bias.repeat_interleave(heads, 0)[:, None, :]
        return args, kwargs

    def after(self, module, args, output):
        actual = output[1].detach()
        raw = self.record['raw_logits'].to(actual.device)
        bias = self.record['bias'].to(actual.device)
        z = raw + bias[:, None, :]
        expected = (z - z.max(-1, keepdim=True).values).softmax(-1).mean(1)[:, None, :]
        error = float((actual - expected).abs().max())
        assert error < 2e-7, ('Native attention arithmetic', error)
        self.record.update(attention=actual[:, 0].cpu(), max_GPU_arithmetic_error=error)

    @contextmanager
    def installed(self, module):
        self.hooks = [module.register_forward_pre_hook(self.before, with_kwargs=True),
                      module.register_forward_hook(self.after)]
        try:
            yield self
        finally:
            for hook in self.hooks: hook.remove()


def fields_for(replay, evidence, offset):
    h, w = replay.views[offset]['info']['fea_map_size']; count = h * w
    masks = replay.spatial_inputs[offset]['encoded_mask'][:, :count]
    fields = []
    for i, pad in enumerate(masks):
        global_pos = 2 * i + offset
        support = evidence.get(global_pos, dict(boxes=torch.empty(0, 4), weights=torch.empty(0)))
        fields.append(evidence_field(support, h, w, pad))
    return fields


@torch.no_grad()
def run_attention(replay, evidence, *, privileged):
    from contextlib import ExitStack
    all_boxes = []; records = []
    assert not any(p.requires_grad or p.grad is not None for p in replay.decoder.parameters())
    assert not replay.delta.requires_grad and torch.count_nonzero(replay.delta) == 0
    for offset, inputs in enumerate(replay.spatial_inputs):
        fields = fields_for(replay, evidence, offset); trackers = []
        with ExitStack() as stack:
            for li, layer in enumerate(replay.decoder.decoder.layers):
                assert layer.from_scratch_cross_attn
                tracker = AttentionCapture(li, fields, replay.views[offset]['info']['fea_map_size'][0]
                                           * replay.views[offset]['info']['fea_map_size'][1],
                                           alpha=1. if privileged else 0.)
                stack.enter_context(tracker.installed(layer.cross_attn)); trackers.append(tracker)
            boxes = replay.decoder.decoder(**inputs).flatten(1, 2)[-1]
        all_boxes.append(boxes)
        records.append(dict(offset=offset, layers=[t.record for t in trackers],
                            active_positions=[2*i+offset for i, f in enumerate(fields) if f['active']]))
    boxes = torch.stack([all_boxes[i % 2][i // 2] for i in range(replay.n)]).float()
    return dict(boxes=boxes.detach().cpu(), attention=records)
