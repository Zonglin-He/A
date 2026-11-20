"""Evaluate withheld seeds under fixed native context and compare physical vote measures."""
from contextlib import nullcontext
from hashlib import sha256

import numpy as np
from scipy.special import logsumexp
import torch
from torch.nn.functional import normalize

ARMS = ('v1', 'private_mean', 'open_mean', 'early_mean', 'private_mass', 'density_only', 'shuffled_mean')
NATIVE = 256


def cpu(tensor):
    return tensor.detach().cpu().numpy()


def terminal_boxes(raw):
    return torch.cat([raw['last_center'], raw['last_pred_size']], dim=-1)[0]


def word_probability(query, tokens, mask):
    logits = query @ tokens.transpose(-1, -2) / .07
    return (logits.softmax(-1) * mask).sum(-1)[0]


def allowed_mask(n, device):
    # In MultiheadAttention, True means blocked. Each probe is a singleton intervention.
    mask = torch.zeros((n, n), dtype=torch.bool, device=device)
    mask[:, NATIVE:] = True
    mask.diagonal().fill_(False)
    return mask


def decode(model, raw, ids, token_mask, isolated):
    xyz = raw['fp2_xyz'][:, ids]
    features = raw['seed_features'].index_select(2, ids).contiguous()
    query = model.decoder_query_proj(features).transpose(1, 2).contiguous()
    estimate = model.proposal_head(features, base_xyz=xyz, end_points={}, prefix='proposal_')
    position = torch.cat(estimate, dim=-1)
    tokens = raw['proj_tokens']

    def read(q):
        projected = normalize(model.contrastive_align_projection_image(q), dim=-1)
        return word_probability(projected, tokens, token_mask)

    early = (cpu(position[0, NATIVE:]), cpu(read(query)[NATIVE:]))
    attn_mask = allowed_mask(len(ids), ids.device) if isolated else None

    def bind_mask(module, args, kw):
        if kw.get('attn_mask') is not None:
            raise ValueError('An unexpected native self-attention mask was supplied')
        return args, dict(kw, attn_mask=attn_mask)

    scene = raw['seed_features'].transpose(1, 2).contiguous()
    for layer, head in zip(model.decoder, model.prediction_heads, strict=True):
        hook = layer.self_attn.register_forward_pre_hook(bind_mask, with_kwargs=True) if isolated else nullcontext()
        with hook:
            query = layer(query, scene, raw['text_memory'], position, None, raw['text_attention_mask'],
                          detected_feats=None, detected_mask=None)
        center, size = head(query.transpose(1, 2).contiguous(), base_xyz=xyz, end_points={}, prefix='probe_')
        position = torch.cat((center, size), dim=-1)
    return cpu(position[0]), cpu(read(query)), early


def overlaps(anchors, proposals):
    anchors = np.asarray(anchors, dtype=np.float64)
    proposals = np.asarray(proposals, dtype=np.float64)
    side_a = anchors[:, 3:6].clip(min=0)
    side_p = proposals[:, 3:6].clip(min=0)
    lo = np.maximum((anchors[:, :3] - side_a / 2)[:, None], proposals[:, :3] - side_p / 2)
    hi = np.minimum((anchors[:, :3] + side_a / 2)[:, None], proposals[:, :3] + side_p / 2)
    intersection = np.maximum(hi - lo, 0).prod(axis=-1)
    union = side_a.prod(-1)[:, None] + side_p.prod(-1) - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def region_totals(anchors, boxes, probability):
    kernel = overlaps(anchors, boxes)
    return kernel.sum(1), kernel @ np.asarray(probability, dtype=np.float64)


def derange(values, key):
    seed = int.from_bytes(sha256(('probe-derangement|' + key).encode()).digest()[:8], 'big')
    ordering = np.random.default_rng(seed).permutation(len(values))
    result = np.empty_like(values)
    result[ordering] = values[np.roll(ordering, 1)]
    return result


def corrections(native_count, native_word, reserve_count, reserve_word):
    valid = (native_count > 0) & (native_word > 0)
    density, mass = np.zeros((2, len(valid)))
    density[valid] = np.log1p(reserve_count[valid] / native_count[valid])
    mass[valid] = np.log1p(reserve_word[valid] / native_word[valid])
    return mass - density, mass, density


def apply_delta(base, delta, valid):
    adjusted = base.copy()
    if np.count_nonzero(valid) >= 2:
        local = base[valid] + delta[valid]
        adjusted[valid] = local + logsumexp(base[valid]) - logsumexp(local)
    return adjusted


def evidence(anchors, native, private, opened, early, base, key):
    factual = region_totals(anchors, *native)
    eligible = (factual[0] > 0) & (factual[1] > 0)
    arms = [base]
    totals = [region_totals(anchors, *measure) for measure in (private, opened, early)]
    totals.append(region_totals(anchors, private[0], derange(private[1], key)))
    private_mean, mass, density = corrections(*factual, *totals[0])
    effects = (private_mean, corrections(*factual, *totals[1])[0],
               corrections(*factual, *totals[2])[0], mass, density, corrections(*factual, *totals[3])[0])
    for effect in effects:
        arms.append(apply_delta(base, effect, eligible))
    packed = np.stack((factual, *totals))
    return np.stack(arms), packed, eligible
