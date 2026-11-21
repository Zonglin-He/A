"""Construct untrained physical region candidates and render label-free prompts."""
import json
from pathlib import Path

import cv2
import numpy as np

from download import ROOT, STUDY, read_json, sha256, write_json
from geometry_cohort import BASELINE, RUN as DIAG, read_jsonl
from score import jsonl


RUN = ROOT / 'outputs' / 'hispatial_physical_self_grounding_20261121'
CONFIG = STUDY / 'PHYSICAL_SELF_GROUNDING.json'
MARKS = 'ABCDEFGHI'
PALETTE = ((245, 72, 50), (39, 172, 250), (215, 82, 235), (74, 202, 126),
           (251, 158, 42), (61, 213, 214), (252, 111, 169), (241, 210, 65))


def valid_xyz(xyz):
    return (xyz[3] > 0) & (xyz[2] > 0.0) & (xyz[2] < 250.0) & np.isfinite(xyz[:3]).all(axis=0)


def index_rows(path):
    raw = read_jsonl(path)
    rows = {row['id']: row for row in raw}
    assert len(rows) == len(raw)
    return rows


def proposal_regions(source, xyz, cfg):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from skimage.color import rgb2lab
    from skimage.segmentation import felzenszwalb

    stride = cfg['grid_stride']
    sampled = xyz[:, stride//2::stride, stride//2::stride]
    inside = valid_xyz(sampled)
    if not inside.any():
        return [], []
    grid_h, grid_w = inside.shape
    depth = sampled[2]
    median = float(np.median(depth[inside]))
    safe_depth = np.where(inside, depth, median)
    lab = rgb2lab(cv2.resize(source, (grid_w, grid_h), interpolation=cv2.INTER_AREA)).astype(np.float32)
    lab /= np.array([100.0, 110.0, 110.0], dtype=np.float32)
    scale = cfg['graph_feature_weights']
    relative = np.clip(np.log(safe_depth / median), -2.0, 2.0)
    rays = np.where(inside[None], np.clip(sampled[:2] / safe_depth[None], -2.0, 2.0), 0.0)
    features = np.concatenate((lab * scale['lab'], relative[..., None] * scale['log_z'],
                               np.moveaxis(rays, 0, -1) * scale['xy_over_z']), axis=-1)
    grid = np.arange(inside.size).reshape(inside.shape)
    left = np.r_[grid[:, :-1].ravel(), grid[:-1, :].ravel()]
    right = np.r_[grid[:, 1:].ravel(), grid[1:, :].ravel()]
    points = sampled[:3].reshape(3, -1)
    metric = np.linalg.norm(points[:, left] - points[:, right], axis=0)
    metric /= np.maximum(safe_depth.flat[left], safe_depth.flat[right])
    observed = inside.flat[left] & inside.flat[right]
    continuous = observed & (metric <= cfg['max_relative_xyz_step'])
    first, second = left[continuous], right[continuous]
    signal = np.linalg.norm(features.reshape(-1, features.shape[-1])[left] -
                            features.reshape(-1, features.shape[-1])[right], axis=1)
    signal = np.where(observed, signal, 0.0)
    salience = np.zeros(inside.size, dtype=np.float32)
    np.maximum.at(salience, left, signal)
    np.maximum.at(salience, right, signal)
    normalizer = float(np.median(signal[observed])) if observed.any() else 0.0
    normalizer = max(normalizer, 0.003)
    edges = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    regions = []
    fingerprints = set()
    for graph_scale in cfg['graph_scales']:
        labels = felzenszwalb(features, scale=graph_scale, sigma=cfg['graph_sigma'],
                             min_size=cfg['graph_min_component'], channel_axis=-1).ravel()
        keep = labels[first] == labels[second]
        a, b = first[keep], second[keep]
        links = coo_matrix((np.ones(len(a), dtype=np.uint8), (a, b)), shape=(inside.size, inside.size)).tocsr()
        _, components = connected_components(links, directed=False)
        counts = np.bincount(components)
        candidates = np.flatnonzero((counts >= cfg['region_min_cells']) &
                                    (counts <= inside.size * cfg['region_max_frame_fraction']))
        for region in candidates:
            support = ((components == region) & inside.ravel()).reshape(inside.shape)
            ys, xs = np.nonzero(support)
            if len(xs) < cfg['region_min_cells'] or np.ptp(xs) < 2 or np.ptp(ys) < 2:
                continue
            fingerprint = np.packbits(support).tobytes()
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            box = [int(xs.min())*stride, int(ys.min())*stride, (int(xs.max())+1)*stride, (int(ys.max())+1)*stride]
            solidity = len(xs)/(int(np.ptp(xs))+1)/(int(np.ptp(ys))+1)
            inner = support & ~cv2.erode(support.astype(np.uint8), edges).astype(bool)
            boundary = float(np.median(salience[inner.ravel()])) if inner.any() else 0.0
            fraction = len(xs)/inside.size
            quality = float(np.sqrt(solidity * min(1.0, fraction/0.012)) * np.log1p(boundary/normalizer))
            regions.append(dict(id=len(regions), scale=graph_scale, grid_area=len(xs), box=box, rank=quality,
                                mass_bin=int(fraction > .02) + int(fraction > .12),
                                tile=(min(2, int(xs.mean()*3/grid_w)), min(2, int(ys.mean()*3/grid_h))), grid_mask=support))

    selected = []
    tile_occ = {}
    size_occ = {}
    pool = list(regions)
    while pool and len(selected) < cfg['max_marked_regions']:
        best = max(pool, key=lambda r: (r['rank'] / (1 + .55*tile_occ.get(r['tile'],0)) /
                                        (1 + .40*size_occ.get(r['mass_bin'],0)), -r['id']))
        pool = [r for r in pool if r['id'] != best['id']]
        sup = best['grid_mask']
        overlap = [int((sup & r['grid_mask']).sum()) for r in selected]
        if any(i/min(best['grid_area'], r['grid_area']) > cfg['marked_max_containment'] or
               i/(best['grid_area']+r['grid_area']-i) > cfg['marked_max_iou'] for i,r in zip(overlap, selected)):
            continue
        tile_occ[best['tile']] = tile_occ.get(best['tile'],0)+1
        size_occ[best['mass_bin']] = size_occ.get(best['mass_bin'],0)+1
        selected.append(best)
    return regions, selected


def render_regions(image, supports, letters, anchors):
    marked = image.copy()
    for mask, color in zip(supports, PALETTE):
        marked[mask] = np.rint(marked[mask]*0.78 + np.asarray(color)*0.22).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(marked, contours, -1, color, 2, cv2.LINE_AA)
    for i, (letter, anchor) in enumerate(zip(letters, anchors)):
        x, y = anchor
        cv2.circle(marked, (x, y), 13, (15, 19, 21), -1, cv2.LINE_AA)
        cv2.circle(marked, (x, y), 13, PALETTE[i], 2, cv2.LINE_AA)
        cv2.putText(marked, letter, (x-7, y+6), cv2.FONT_HERSHEY_SIMPLEX, .58,
                    (255,255,255), 2, cv2.LINE_AA)
    return marked


def render_depth(image, xyz, valid):
    if not valid.any():
        return image.copy()
    low, high = np.quantile(xyz[2][valid], (.02, .98))
    stretch = np.clip((high - xyz[2])/max(high-low, 1e-6), 0, 1)
    depth_color = cv2.applyColorMap(np.uint8(np.rint(stretch*255)), cv2.COLORMAP_TURBO)[..., ::-1]
    visual = image.copy()
    visual[valid] = np.rint(visual[valid]*0.50 + depth_color[valid]*0.50).astype(np.uint8)
    return visual


def physical_readouts(xyz, regions, cfg):
    stride = cfg['grid_stride']
    valid = valid_xyz(xyz)
    sampled = xyz[:3,stride//2::stride,stride//2::stride]
    anchor_xyz = sampled.repeat(stride, axis=1).repeat(stride,axis=2)
    fused = valid & (np.linalg.norm(xyz[:3] - anchor_xyz, axis=0) <=
                     cfg['max_relative_xyz_step']*np.maximum(xyz[2], anchor_xyz[2]))
    details, supports, used = [], [], []
    for r in regions:
        support = r['grid_mask'].repeat(stride,0).repeat(stride,1) & fused
        sy, sx = np.nonzero(support)
        assert len(sy) > 0
        x0, y0, x1, y1 = int(sx.min()), int(sy.min()), int(sx.max())+1, int(sy.max())+1
        observed = xyz[:3, support].astype(np.float64)
        flat = xyz[2,y0:y1,x0:x1]
        window = flat[valid[y0:y1,x0:x1]]
        assert len(window) >= len(sy)
        interior = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5)[sy,sx]
        if used:
            minimum = np.min([(sx-x)**2+(sy-y)**2 for x,y in used], axis=0)
            if np.any(minimum >= 30**2):
                interior = np.where(minimum >= 30**2, interior,-1)
        k = int(interior.argmax())
        point = (int(np.clip(sx[k],14,xyz.shape[-1]-15)), int(np.clip(sy[k],14,xyz.shape[-2]-15)))
        used.append(point)
        supports.append(support)
        details.append(dict(id=r['id'], box=[x0,y0,x1,y1], area=len(sx),
                            region_z=float(observed[2].mean()), window_z=float(window.mean()),
                            window_points=len(window), visible_centroid=observed.mean(axis=1).tolist(),
                            anchor=list(point)))
    return valid, details, supports, used


def read_rgb(frame):
    src = BASELINE / 'embspatial/images' / frame['image_sha256']
    assert sha256(src) == frame['image_sha256']
    data = cv2.imdecode(np.frombuffer(src.read_bytes(), np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    assert data is not None
    # Benchmark images are loaded as RGB without any EXIF transform.
    from PIL import Image
    with Image.open(src) as image:
        rgb = np.asarray(image.convert('RGB'))
    return cv2.resize(rgb,(448,448)), [int(rgb.shape[1]), int(rgb.shape[0])]


def save_png(path, image):
    assert cv2.imwrite(str(path), image[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return sha256(path)


def prepare():
    cv2.setNumThreads(1)
    cfg = read_json(CONFIG)
    run = RUN
    if (run / 'input_seal.json').exists():
        raise FileExistsError(run / 'input_seal.json')
    views = run / 'views'
    views.mkdir(parents=True, exist_ok=True, mode=0o700)
    input_seal = read_json(DIAG / 'inputs_seal.json')
    pred_seal = read_json(DIAG / 'predictions_seal.json')
    assert sha256(DIAG / 'inputs_seal.json') == pred_seal['input_seal_sha256']
    for key in ('cases.jsonl','metadata.jsonl','frames.json','probe_metadata.jsonl'):
        assert sha256(DIAG / key) == input_seal['files'][key]
    assert sha256(DIAG / 'pointmap_sha256.json') == pred_seal['files']['pointmap_sha256.json']
    cases = index_rows(DIAG / 'cases.jsonl')
    meta = index_rows(DIAG / 'metadata.jsonl')
    picked = {q['case_id'] for q in read_jsonl(DIAG / 'probe_metadata.jsonl') if q['task'] == 'emb_order'}
    frames = read_json(DIAG / 'frames.json')
    maps = read_json(DIAG / 'pointmap_sha256.json')
    roster = [dict(id=q['id'], key=q['key'], prompt=q['prompt'], topic=meta[q['id']]['topic'],
                   options=meta[q['id']]['options'], order_control=q['id'] in picked)
              for q in cases.values() if q['id'].startswith('embspatial/')]
    assert len(roster)==1206 and sum(r['order_control'] for r in roster)==96
    keys = {q['key'] for q in roster}
    reverse_keys = {q['key'] for q in roster if q['order_control']}
    payload, asset_hashes = {}, {}
    for n, key in enumerate(sorted(keys),1):
        loc = DIAG / 'pointmaps' / f'{key}.npy'
        assert sha256(loc)==maps[key]
        xyz = np.load(loc, allow_pickle=False)
        assert xyz.shape == (4,448,448)
        rgb, size = read_rgb(frames[key])
        bank, selected = proposal_regions(rgb,xyz,cfg)
        valid, candidates, supports, positions = physical_readouts(xyz,selected,cfg)
        letters = MARKS[:len(selected)]
        asset_hashes[key] = {}
        images = {'marks':render_regions(rgb,supports,letters,positions),'whole':render_depth(rgb,xyz,valid)}
        if key in reverse_keys:
            images['reversed']=render_regions(rgb,supports,letters[::-1],positions)
        for view, image in images.items():
            asset_hashes[key][view] = save_png(views / f'{key}.{view}.png', image)
        mask_path = views / f'{key}.support.npy'
        mask = np.packbits(np.asarray(supports,dtype=bool).reshape(len(supports),448*448), axis=1)
        np.save(mask_path,mask,allow_pickle=False)
        asset_hashes[key]['supports'] = sha256(mask_path)
        payload[key] = dict(size=size, xyz_sha256=maps[key], marked=candidates,
                            bank=[{k:r[k] for k in ('id','box','scale','grid_area','rank')} for r in bank])
        if n % 100 == 0 or n == len(keys):
            print('geometry frames', n, '/', len(keys), flush=True)
    jsonl(run / 'roster.jsonl', roster)
    write_json(run / 'regions.json', payload)
    write_json(run / 'views_sha256.json', asset_hashes)
    write_json(run / 'input_seal.json', dict(rows=len(roster), frames=len(keys), label_access=False,
        protocol=sha256(CONFIG), diagnosis_input=sha256(DIAG / 'inputs_seal.json'),
        diagnosis_predictions=sha256(DIAG / 'predictions_seal.json'),
        files={f:sha256(run/f) for f in ('roster.jsonl','regions.json','views_sha256.json')}))


if __name__ == '__main__':
    prepare()
