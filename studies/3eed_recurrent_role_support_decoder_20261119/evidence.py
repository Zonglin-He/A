"""Frozen point, RGB and language evidence. No task-label reader."""
import gc
import os
from functools import lru_cache
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from common import (CACHE, PLATFORMS, PRIVATE, ROOT, SENTINEL, array_sha, camera_view,
                    json_atomic, npz_atomic, points_for, read, sha, stable, verify_inputs)
from camera_projection import CameraSpec, project_model_points, read_camera_allowlist
from model51 import strict_runtime
from model76 import Adapter
from regression_adapter import first_noun_mask

FIELDS = ('points', 'rgb', 'xyz', 'uv', 'visible', 'text', 'valid', 'role_mask',
          'has_reference', 'left_right', 'prior', 'prior_scores', 'anchors')


def encoder(source):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import sys
    sys.path.insert(0, str(ROOT/'crossplatform/round96'))
    import factory as factory96

    reg = read(ROOT/'crossplatform/single_frame_sources.json')[source]
    backbone, matcher = (ROOT/reg[k] for k in ('epoch100', 'matcher'))
    assert sha(backbone) == reg['epoch100_sha256'] and sha(matcher) == reg['matcher_sha256']
    cwd = Path.cwd()
    try:
        factory96.configure({'seed': 0, 'torch_num_threads': 2})
        strict_runtime()
        torch.use_deterministic_algorithms(True)
        native = factory96.model()
    finally:
        os.chdir(cwd)
    state = torch.load(backbone, map_location='cpu', weights_only=True)
    assert state['epoch'] == 100 and state['trained_platforms'] == [source]
    native.load_state_dict(state['model'], strict=True)
    del state
    # The existing R76/R91 stem and image adapter, with no 3D or task label input.
    rgb = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.GELU(),
                        nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.GELU(),
                        nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU())
    stem = nn.ModuleDict({'rgb': rgb, 'image_adapter': Adapter(64)})
    with np.load(matcher, allow_pickle=False) as found:
        stem.load_state_dict({k: torch.from_numpy(found[k]) for k in stem.state_dict()}, strict=True)
    return (native.cuda().eval().requires_grad_(False), stem.cuda().eval().requires_grad_(False))


@lru_cache(maxsize=1)
def dependency_parser():
    import parse97
    assets = ROOT/'crossplatform/round20/assets'
    config = read(assets/'STANZA_ASSET_RECEIPT.json')['kwargs']
    nlp = parse97.stanza.Pipeline('en', dir=str(assets/'stanza'),
                                 processors='tokenize,mwt,pos,lemma,depparse', package=None,
                                 use_gpu=False, download_method=None, verbose=False, **config)
    return nlp, parse97


def natural_graphs(rows):
    cached = ROOT/'crossplatform/round97/roles_PRIVATE'
    graphs = [None] * len(rows)
    fresh = []
    for i, row in enumerate(rows):
        if 'spans' in row:
            continue
        f = cached/(row['id']+'.json') if 'id' in row else None
        if f and f.exists():
            item = read(f)
            from hashlib import sha256
            assert item['caption_sha256'] == sha256(row['caption'].encode()).hexdigest()
            graphs[i] = item
        else:
            fresh.append(i)
    if fresh:
        nlp, parser = dependency_parser()
        docs = nlp.bulk_process([rows[i]['caption'] for i in fresh])
        for i, doc in zip(fresh, docs, strict=True):
            txt = rows[i]['caption']
            d = parser.dependency_document(doc)
            initial = parser.parse85(txt, d, parser.parse_subject(txt, d))
            graphs[i] = {'graph': parser.repair_inventory87(txt, d, initial), 'doc': d}
    return graphs


def text_masks(rows, offsets, valid):
    graphs = natural_graphs(rows)
    masks = np.zeros((len(rows), 3, valid.shape[1]), dtype=bool)
    native = np.zeros(valid.shape, dtype=bool)
    refs, lr = np.zeros(len(rows), bool), np.zeros(len(rows), bool)
    for i, (row, parsed) in enumerate(zip(rows, graphs, strict=True)):
        a, b = offsets[i].T
        content = (b > a) & (a < len(row['caption']))
        missing = (0, 0)
        if parsed is None:
            spans = row['spans']
            refs[i] = lr[i] = True
        else:
            graph = parsed['graph']
            entities = graph['entities']
            edges = [e for e in graph['relations'] if 'T' in (e['subject'], e['reference'])]
            selected = entities[1] if len(entities) > 1 else None
            direct = edges[0] if len(edges) == 1 else None
            relation_span = missing
            if direct is not None:
                other = direct['reference'] if direct['subject'] == 'T' else direct['subject']
                selected = next((item for item in entities if item['role'] == other), selected)
                words = parsed['doc']['sentences'][direct['sentence']]['words']
                w = next((w for w in words if w['id'] == direct['governor_word']), None)
                if w and w.get('start_char') is not None and w.get('end_char') is not None:
                    relation_span = (w['start_char'], w['end_char'])
                lr[i] = direct['predicate'] in ('LEFT', 'RIGHT')
            refs[i] = selected is not None
            spans = (entities[0]['span'] if entities else (0, len(row['caption'])),
                     selected['span'] if selected else missing, relation_span)
        for k, (lo, hi) in enumerate(spans):
            if hi > lo:
                masks[i, k] = content & (a < hi) & (b > lo)
        if not masks[i, 0].any():
            masks[i, 0] = content if content.any() else valid[i]
        refs[i] &= masks[i, 1].any()
        lr[i] &= refs[i]
        try:
            native[i] = np.asarray(first_noun_mask(row['caption']+SENTINEL, offsets[i]), bool)
        except ValueError:
            native[i] = (b > a)
        native[i] &= valid[i]
        if not native[i].any():
            native[i] = masks[i, 0]
        assert native[i].any() and (valid[i] & ~native[i]).any()
    return masks, native, refs, lr


@lru_cache(maxsize=48)
def image_inputs(path, meta, platform, view):
    # Only the explicit camera-field allowlist is materialized.
    fields = read_camera_allowlist(meta)
    cam = CameraSpec.from_camera_fields(platform, fields, waymo_view_suffix=view,
                                      missing_distortion_policy='assume_zero_declared')
    with Image.open(path) as im:
        size = im.size
        pixels = np.asarray(im.convert('RGB').resize((384, 256), Image.Resampling.BILINEAR)).copy()
    return pixels, size, cam


def pixel_proof(row, seeds, points):
    rgb, size, cam = image_inputs(row['imagepath'], row['metapath'], row['platform'], camera_view(row))
    n = len(seeds)
    p = project_model_points(np.concatenate((seeds, points)), cam, size)
    valid = p['in_image']
    projected = np.where(valid[:, None], p['uv'], 0.)
    bw = (size[0] + 3)//4
    cells = np.floor(projected/4).astype(int)
    flat = cells[:, 1]*bw + cells[:, 0]
    zbuf = np.full(bw*((size[1] + 3)//4), np.inf)
    np.minimum.at(zbuf, flat[n:][valid[n:]], p['depth'][n:][valid[n:]])
    visible = valid[:n] & (p['depth'][:n] <= zbuf[flat[:n]] + .30 + .02*p['depth'][:n])
    uv = (projected[:n] / np.asarray(size)).astype(np.float32)
    # Image feature content is masked on occluded or unprojectable seed points.
    return rgb, uv, valid[:n], visible


def anchor_seeds(prior, scores, count=4):
    """Diverse symmetric initialization. Subsequent slots can read all seed points."""
    selected = []
    rank = scores.clone()
    for _ in range(count):
        i = rank.argmax(-1)
        selected.append(i)
        box = prior.gather(1, i[..., None].expand(-1, -1, 6))
        d = (prior[:, None, :, :3] - box[:, :, None, :3]) / (box[:, :, None, 3:]*.55 + .4)
        rank = rank - 4. * torch.exp(-.5 * d.square().sum(-1))
        rank.scatter_(-1, i[..., None], -torch.inf)
    return torch.stack(selected, -1)


def check_seed_coordinates(xyz, sampled):
    # Use actual coordinates rather than the upstream fp2_inds shortcut.
    from scipy.spatial import cKDTree
    error = max(float(cKDTree(a).query(x, k=1)[0].max())
                for a, x in zip(sampled, xyz, strict=True))
    if error > 1e-4:
        raise ValueError(f'Seed/sample coordinate mismatch {error:.5g}')
    return error


@torch.inference_mode()
def extract(loaded, rows, sampled, archived=None, coordinate_audit=False):
    base, stem = loaded
    pts = torch.as_tensor(sampled, device='cuda')
    utterances = [r['caption'] + SENTINEL for r in rows]
    encoder_inputs = {'point_clouds': pts, 'text': utterances}
    z = base(encoder_inputs) if archived is None else base._run_backbones(encoder_inputs)
    encoded = base.tokenizer(utterances, padding='longest', return_offsets_mapping=True,
                             return_tensors='np')
    assert np.array_equal(z['tokenized']['input_ids'].cpu(), encoded['input_ids'])
    valid = encoded['attention_mask'].astype(bool)
    offsets = encoded['offset_mapping']
    role, native, has_ref, is_lr = text_masks(rows, offsets, valid)
    seeds = z['fp2_xyz'].cpu().numpy()
    if coordinate_audit:
        check_seed_coordinates(seeds, sampled)
    px, uv, in_image, visible = (np.stack(v) for v in zip(
        *(pixel_proof(r, x, p) for r, x, p in zip(rows, seeds, sampled, strict=True)), strict=True))
    dev = lambda a: torch.as_tensor(a, device='cuda')
    pixels = dev(px).permute(0, 3, 1, 2).contiguous().float().div_(255.)
    feature_map = stem['rgb']((pixels-.5)*2)
    rgb = F.grid_sample(feature_map, dev(uv)[:, :, None]*2-1, mode='bilinear',
                        padding_mode='border', align_corners=False)[..., 0].transpose(1, 2)
    image_features = stem['image_adapter'](rgb) * dev(visible)[..., None]
    if archived is None:
        raw = torch.cat((z['last_center'], z['last_pred_size']), dim=-1)
        q, t = z['last_proj_queries'], z['proj_tokens']
    else:
        raw, q, t = map(dev, archived)
    scores = q @ t.transpose(1, 2) / .07
    nm = dev(native)[:, None]
    neg = dev(valid)[:, None] & ~nm
    odds = torch.logsumexp(scores.masked_fill(~nm, -torch.inf), -1)
    odds -= torch.logsumexp(scores.masked_fill(~neg, -torch.inf), -1)
    ref = dev(role[:, 1]).float()
    rs = (scores * ref[:, None]).sum(-1) / ref.sum(-1, keepdim=True).clamp_min(1)
    rs = torch.where(dev(has_ref)[:, None], rs, torch.zeros_like(rs))
    prior_scores = torch.stack((odds, rs), 1)
    raw = torch.cat((raw[..., :3], raw[..., 3:].clamp_min(.03)), -1)
    assert raw.shape[1:] == (256, 6) and seeds.shape[1:] == (1024, 3)
    return dict(points=z['fp2_features'].transpose(1, 2).half().contiguous(),
                rgb=image_features.half(), xyz=z['fp2_xyz'], uv=dev(uv).half(),
                visible=dev(visible), in_image=dev(in_image),
                text=z['text_feats'].half(), valid=dev(valid), role_mask=dev(role),
                has_reference=dev(has_ref), left_right=dev(is_lr), prior=raw,
                prior_scores=prior_scores, anchors=anchor_seeds(raw, prior_scores))


def source_features(source, batch_size=8, device='cuda'):
    verify_inputs()
    folder = PRIVATE/source
    cache = CACHE/source
    cache.mkdir(exist_ok=True, parents=True, mode=0o700)
    all_start = time.perf_counter()
    proof_file = folder/'FEATURE_SEAL.json'
    if proof_file.exists():
        proof = read(proof_file)
        if all((cache/name).is_file() and sha(cache/name) == digest for name, digest in proof['files'].items()):
            assert proof['roster'] == sha(folder/'ROSTER_SEAL.json')
            assert proof['capture'] == sha(Path(__file__))
            return proof
        raise RuntimeError('Existing source feature seal has missing or changed bytes')
    net = encoder(source)
    files, widths, count = {}, {}, {}
    for split in ('train', 'monitor'):
        path = folder/f'{split}_PRIVATE.json'
        assert sha(path) == read(folder/'ROSTER_SEAL.json')['rosters'][path.name]
        rows = read(path)
        width = max(map(len, net[0].tokenizer([r['caption']+SENTINEL for r in rows])['input_ids']))
        widths[split] = width
        count[split] = len(rows)
        for block in range(0, len(rows), 128):
            name = f'{split}_{block//128:04d}.npz'
            dest = cache/name
            use = rows[block:block+128]
            if dest.exists():
                with np.load(dest, allow_pickle=False) as disk:
                    assert str(disk['capture_sha']) == sha(Path(__file__))
                    assert str(disk['roster_sha']) == sha(path) and len(disk['xyz']) == len(use)
            else:
                items = []
                for start in range(0, len(use), batch_size):
                    batch = use[start:start+batch_size]
                    pts = np.stack([points_for(r)[0] for r in batch])
                    feat = extract(net, batch, pts, coordinate_audit=block == 0 and start == 0)
                    actual = feat['text'].shape[1]
                    for k in ('text', 'valid'):
                        a = feat[k]
                        feat[k] = (F.pad(a, (0, 0, 0, width-actual)) if k == 'text'
                                   else F.pad(a, (0, width-actual)))
                    feat['role_mask'] = F.pad(feat['role_mask'], (0, width-actual))
                    arrays = {k: feat[k].cpu().numpy() for k in (*FIELDS, 'in_image')}
                    arrays.update(gt=np.asarray([r['boxes'] for r in batch], np.float32),
                                  known_ref=np.asarray([r['known_reference'] for r in batch], bool),
                                  side=np.asarray([r.get('side', 0) for r in batch], np.int8))
                    items.append(arrays)
                combined = {k: np.concatenate([a[k] for a in items]) for k in items[0]}
                npz_atomic(dest, **combined, roster_sha=sha(path), capture_sha=sha(Path(__file__)))
            files[name] = sha(dest)
            print('SOURCE_FEATURES', source, split, min(block+128, len(rows)), len(rows),
                  round(time.perf_counter()-all_start, 1), flush=True)
    proof = dict(source=source, files=files, widths=widths, counts=count,
                 roster=sha(folder/'ROSTER_SEAL.json'), capture=sha(Path(__file__)), target_labels_read=False)
    json_atomic(proof_file, proof)
    del net
    dependency_parser.cache_clear()
    gc.collect()
    torch.cuda.empty_cache()
    return proof


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('source', choices=PLATFORMS)
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    source_features(args.source, args.batch_size)
