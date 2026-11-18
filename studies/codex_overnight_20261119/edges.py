"""Finite interventions on the two role-specific language reads of a frozen 3EED."""
import math

import numpy as np


def point_support(boxes, points):
    from geometry162 import in_box

    return np.stack([in_box(points, box) for box in boxes]) if len(boxes) else np.empty((0, len(points)), bool)


def slot_proxy(ids, native_boxes, extras, points):
    """A new box receives a query only through the fixed point-overlap criterion."""
    ids = np.asarray(ids)
    n = len(native_boxes)
    result = ids.copy()
    new = ids >= n
    if new.any():
        from requery169 import associate

        regions = point_support(extras[ids[new] - n], points)
        native = point_support(np.c_[native_boxes, np.zeros(n)], points)
        _, matched, _ = associate(regions, native, np.zeros(len(regions)))
        result[new] = matched
    return result


def decompose(scores):
    """ANOVA separation on a common support rectangle, with zero-mean edge residuals."""
    row_mean = scores.mean(axis=1, keepdims=True)
    column_mean = scores.mean(axis=0, keepdims=True)
    center = scores.mean()
    row = (row_mean - center)[:, 0]
    column = (column_mean - center)[0]
    interaction = (scores - row_mean) - (column_mean - center)
    np.testing.assert_allclose(row[:, None] + column + interaction + center, scores, atol=1e-12)
    np.testing.assert_allclose(interaction.sum(0), 0, atol=1e-12)
    np.testing.assert_allclose(interaction.sum(1), 0, atol=1e-12)
    return row, column, interaction


class RemoveRoleValues:
    """Intercept the returned text slots while leaving their attention weights untouched."""

    def __init__(self, token_mask, regions):
        self.token_mask = token_mask
        self.regions = regions
        self.delta = None
        self.selected = None
        self.error = None

    def __call__(self, module, args, kw, out):
        import torch
        import torch.nn.functional as F

        result, attention = out
        if self.delta is None:
            assert not module.training and not module.batch_first and not args
            assert kw['attn_mask'] is None and not kw['key_padding_mask'].any()
            text = kw['query'][:, 0]
            assert kw['key'] is kw['value'] or torch.equal(kw['key'], kw['value'])
            point = kw['key'][:, 0]
            assert text.shape[0] == len(self.token_mask) and point.shape[0] == self.regions.shape[-1]
            assert kw['query'].shape[1] == kw['key'].shape[1] == 1
            wq, wk, wv = module.in_proj_weight.chunk(3)
            bq, bk, bv = module.in_proj_bias.chunk(3)
            n_head, width = module.num_heads, module.head_dim
            q = F.linear(text, wq, bq).unflatten(-1, (n_head, width)).transpose(0, 1)
            k = F.linear(point, wk, bk).unflatten(-1, (n_head, width)).transpose(0, 1)
            v = F.linear(point, wv, bv).unflatten(-1, (n_head, width)).transpose(0, 1)
            a = ((q / math.sqrt(width)) @ k.transpose(1, 2)).softmax(-1)
            torch.testing.assert_close(a.mean(0), attention[0], rtol=3e-5, atol=1e-6)
            unmasked = F.linear((a @ v).transpose(0, 1).flatten(1), module.out_proj.weight, module.out_proj.bias)
            torch.testing.assert_close(unmasked, result[:, 0], rtol=3e-5, atol=2e-5)
            self.error = float((unmasked - result[:, 0]).abs().max())
            a = a[:, self.token_mask]
            selected = self.regions[:, None, None] * a
            pieces = (selected @ v).transpose(1, 2).flatten(2)
            self.delta = F.linear(pieces, module.out_proj.weight)
            return
        if self.selected is not None:
            altered = result.clone()
            altered[self.token_mask, 0] -= self.delta[self.selected]
            return altered, attention


def odds(prefix, model, pixels):
    """Native R76 score in log odds units, before competition softmax."""
    import torch

    _, q, evidence = model.matcher.main(model.base, prefix, pixels, 'bidirectional')
    logits = q @ evidence['projected_text'].T / .07
    mask = prefix['token_mask'].bool()
    scores = torch.logsumexp(logits[:, mask], dim=1) - torch.logsumexp(logits[:, ~mask], dim=1)
    return scores.detach().cpu().numpy().astype(np.float64)


def responses(model, row, z, role_tokens, regions):
    import torch
    from model76 import tensor_data, tensor_prefix
    from prepare76 import geometry

    assert model.matcher.route is None and not model.matcher.use_interface
    spatial, _ = geometry(row, z)
    prefix = tensor_prefix(z)
    pixels = tensor_data(spatial, row)
    module = model.base.cross_encoder.layers[-1].cross_layer.cross_lv
    patch = RemoveRoleValues(torch.as_tensor(role_tokens, device='cuda', dtype=torch.bool),
                             torch.as_tensor(regions, device='cuda', dtype=torch.float32))
    assert patch.token_mask.any() and not prefix['textpad'][0, patch.token_mask].any()
    hook = module.register_forward_hook(patch, with_kwargs=True)
    try:
        baseline = odds(prefix, model, pixels)
        assert patch.delta is not None and np.isfinite(baseline).all()
        response = []
        for index in range(len(regions)):
            patch.selected = index
            changed = odds(prefix, model, pixels)
            response.append(baseline - changed)
        patch.selected = None
        np.testing.assert_array_equal(odds(prefix, model, pixels), baseline)
    finally:
        hook.remove()
    return np.asarray(response), patch.error
