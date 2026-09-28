#!/usr/bin/env python3
"""Validation-only search for Ours. Never tunes an external baseline.

The frozen three spatial views are cached once. All candidate updates use
only native head features, then validation labels score the completed output.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.run_temporal_anchor_spatial_v1 import sha, write, cpu
from scripts.run_feasibility import load_full_video, prediction_record
from scripts.run_baseline_expansion_v1 import read
from scripts.analyze_baseline_expansion_v1 import summarize
from vg_tta.baseline_expansion_data import make_dataset
from vg_tta.geometric_video_io import normalize_raw_view
from vg_tta.geometric_spatial_teacher import map_crop_boxes
from vg_tta.component_fusion import fuse_views
from vg_tta.simplified_crop import full_video_crop_rect
from vg_tta.fullspan_tta import fit_fullspan_head, replay_temporal_head
from vg_tta.tubedetr_runtime import (add_repo_to_path, build_model, load_official_checkpoint,
    forward_video, forward_video_with_temporal_head_input)

DEFAULT = ROOT / 'artifacts/ours_validation_tuning_v1'


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def query_identity(index, row, spec):
    return {'index': int(index), 'source': spec['sources'][str(index)],
            'row_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()}


def check_item(item, labels, expected):
    assert item['identity'] == expected
    assert labels['identity'] == expected
    assert item['index'] == expected['index'] and item['source'] == expected['source']
    assert item['caption'] == labels['video_target']['caption']
    assert hashlib.sha256(json.dumps(labels['annotation'], sort_keys=True).encode()).hexdigest() == expected['row_sha256']


def verify(out):
    lock = read(out / 'lock.json')
    for path, digest in lock['code_sha256'].items():
        assert sha(ROOT / path) == digest, path
    for spec in lock['datasets'].values():
        assert sha(spec['annotation']) == spec['annotation_sha256']
    paths = [Path(__file__), ROOT / 'vg_tta/fullspan_tta.py', ROOT / 'vg_tta/tubedetr_runtime.py',
             ROOT / 'vg_tta/metrics.py', ROOT / 'vg_tta/geometric_video_io.py',
             ROOT / 'vg_tta/geometric_spatial_teacher.py', ROOT / 'vg_tta/component_fusion.py',
             ROOT / 'vg_tta/simplified_crop.py']
    runtime = {'protocol_sha256': sha(out / 'lock.json'),
               'code_sha256': {str(p): sha(p) for p in paths}}
    if (out / 'runtime_lock.json').exists():
        assert read(out / 'runtime_lock.json') == runtime
    else:
        write(out / 'runtime_lock.json', runtime)
    return lock


def cache(out, groups):
    lock = verify(out)
    model, _ = build_model(ROOT / 'external/TubeDETR', device='cuda', resolution=224, stride=2)
    active_source = None
    for name in groups or lock['groups']:
        group = lock['groups'][name]
        spec = lock['datasets'][group['target']]
        if active_source != group['source']:
            checkpoint = lock['checkpoints'][group['source']]
            assert sha(checkpoint) == lock['checkpoint_sha256'][group['source']]
            loaded = load_official_checkpoint(model, checkpoint)
            assert not loaded['missing_keys'] and not loaded['unexpected_keys']
            model.eval().requires_grad_(False)
            active_source = group['source']
        data, rows, capture = make_dataset(spec)
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        head_path = folder / 'source_head.pt'
        if not head_path.exists():
            torch.save(copy.deepcopy(model.sted_embed).cpu().eval(), head_path)
        versions = {n: p._version for n, p in model.named_parameters()}
        for ordinal, index in enumerate(spec['indices']):
            path = folder / 'cache' / f'{index:06d}.pt'
            ep = folder / 'validation_labels_only' / f'{index:06d}.pt'
            identity = query_identity(index, rows[index], spec)
            if path.exists() and (ordinal != 0 or (folder / 'adapted_native_parity.json').exists()):
                c = load(path)
                assert c['runtime_lock_sha256'] == sha(out / 'runtime_lock.json') and ep.exists()
                check_item(c, load(ep), identity)
                continue
            video_path = str(Path(spec['root']) / 'video' / rows[index]['video_path'])
            assert sha(video_path) == spec['video_sha256'][video_path]
            video, targets, vt = load_full_video(data[index])
            raw = capture.raw
            assert torch.equal(video, normalize_raw_view(raw, resolution=224))
            caption = vt['caption']
            with torch.no_grad():
                output, hidden = forward_video_with_temporal_head_input(model, video, caption,
                    repo=ROOT / 'external/TubeDETR', stride=2, device='cuda')
                expected = replay_temporal_head(model.sted_embed, hidden, 'cuda')
                assert torch.equal(expected, output['pred_sted'])
            output = cpu({k: output[k] for k in ['pred_boxes', 'pred_sted']})
            h, w = raw.shape[1:3]
            rect = full_video_crop_rect(output['pred_boxes'], (h, w))
            views = {'original': output['pred_boxes'].float()}
            for key, res, crop in [('global_high', 448, None), ('crop', 320, rect)]:
                v = normalize_raw_view(raw, resolution=res, crop=crop)
                with torch.no_grad():
                    p = forward_video(model, v, caption, repo=ROOT / 'external/TubeDETR', stride=2, device='cuda')
                views[key] = p['pred_boxes'].detach().float().cpu()
                del p, v
            x0, y0, x1, y1 = rect
            views['crop'] = map_crop_boxes(views['crop'], (x0 / w, y0 / h, x1 / w, y1 / h))
            boxes = fuse_views(views, ['original', 'global_high', 'crop'])
            payload = {'index': index, 'source': spec['sources'][str(index)], 'head_input': cpu(hidden),
                'frozen': output, 'fused_boxes': boxes, 'crop_rect': rect, 'caption': caption,
                'identity': identity,
                'gt_used': False, 'runtime_lock_sha256': sha(out / 'runtime_lock.json')}
            path.parent.mkdir(parents=True, exist_ok=True)
            ep.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
            torch.save({'targets': cpu(targets), 'video_target': vt, 'annotation': rows[index], 'identity': identity}, ep)
            if ordinal == 0:
                # Check a real multistep candidate in the native network before
                # relying on extracted-head replay for the bounded search.
                source_head = model.sted_embed
                adapted = copy.deepcopy(source_head).eval()
                parity_fit = fit_fullspan_head(adapted, [{'head_input': hidden.detach()}] * 3,
                                              lr=.001, anchor_gamma=.0001)
                try:
                    model.sted_embed = adapted
                    with torch.no_grad():
                        native = forward_video(model, video, caption, repo=ROOT / 'external/TubeDETR', stride=2, device='cuda')
                        replay = replay_temporal_head(adapted, hidden, 'cuda')
                    assert torch.equal(native['pred_sted'], replay)
                    assert torch.equal(native['pred_boxes'].cpu(), output['pred_boxes'])
                    write(folder / 'adapted_native_parity.json', {'exact': True, 'index': index,
                        'audit': parity_fit['audit'], 'runtime_lock_sha256': sha(out / 'runtime_lock.json')})
                finally:
                    model.sted_embed = source_head
                del adapted, native
            assert all(p._version == versions[n] and p.grad is None for n, p in model.named_parameters())
            print(f'[tune cache {name}] {ordinal + 1}/{len(spec["indices"])}', flush=True)
        write(folder / 'cache_complete.json', {'indices': spec['indices'], 'gt_used': False,
            'query_identities': {str(i): query_identity(i, rows[i], spec) for i in spec['indices']},
            'cache_sha256': {str(i): sha(folder / 'cache' / f'{i:06d}.pt') for i in spec['indices']},
            'label_sha256': {str(i): sha(folder / 'validation_labels_only' / f'{i:06d}.pt') for i in spec['indices']},
            'source_head_sha256': sha(head_path), 'runtime_lock_sha256': sha(out / 'runtime_lock.json')})


def config_name(lr, steps, gamma):
    return f'lr{lr:g}_steps{steps}_gamma{gamma:g}'


def score(output, labels, item, method):
    result = prediction_record(output, labels['targets'], labels['video_target'], labels['annotation'],
        sample_index=item['index'], condition='clean', method=method, runtime_sec=0, peak_vram_gb=0)
    result.update(source=item['source'], gt_used=False, labels_used_for='validation_selection_only')
    return result


def candidate_order(row):
    cfg = row['config']
    return (cfg['steps'], cfg['lr'], cfg['gamma'], cfg['name'])


def choose(rows, neutral_pp=.1):
    best = max(r['summary']['vIoU'] for r in rows)
    eligible = [r for r in rows if best - r['summary']['vIoU'] <= neutral_pp]
    return min(eligible, key=candidate_order)


def fit(out, groups, stage):
    lock = verify(out)
    add_repo_to_path(ROOT / 'external/TubeDETR')
    for name in groups or lock['groups']:
        group = lock['groups'][name]
        spec = lock['datasets'][group['target']]
        folder = out / name
        complete = read(folder / 'cache_complete.json')
        assert complete['indices'] == spec['indices']
        assert complete['runtime_lock_sha256'] == sha(out / 'runtime_lock.json')
        assert read(folder / 'adapted_native_parity.json')['exact']
        assert sha(folder / 'source_head.pt') == complete['source_head_sha256']
        template = load(folder / 'source_head.pt').eval().cuda()
        items = []
        labels = []
        for index in spec['indices']:
            path = folder / 'cache' / f'{index:06d}.pt'
            ep = folder / 'validation_labels_only' / f'{index:06d}.pt'
            assert sha(path) == complete['cache_sha256'][str(index)]
            assert sha(ep) == complete['label_sha256'][str(index)]
            item = load(path)
            assert item['gt_used'] is False and not {'targets', 'annotation', 'gt_boxes'} & set(item)
            label = load(ep)
            check_item(item, label, complete['query_identities'][str(index)])
            items.append(item)
            labels.append(label)
        frozen = [score({**x['frozen'], 'pred_boxes': x['fused_boxes']}, y, x, 'frozen_fusion')
                  for x, y in zip(items, labels)]
        write(folder / 'frozen_fusion.json', {'records': frozen, 'summary': summarize(frozen, frozen)})
        coarse = [{**c, 'name': config_name(c['lr'], c['steps'], c['gamma'])} for c in lock['coarse_configs']]
        configs = coarse
        if stage == 'refine':
            old = [read(folder / 'candidates' / f'{c["name"]}.json') for c in coarse]
            top = sorted(old, key=lambda r: (-r['summary']['vIoU'], candidate_order(r)))[:3]
            extra = {}
            for row in top:
                c = row['config']
                settings = [(c['lr'] * factor, c['steps'], 1e-4) for factor in [.5, 2.]]
                settings += [(c['lr'], c['steps'], g) for g in [.001, .01, .1]]
                for lr, steps, gamma in settings:
                    key = config_name(lr, steps, gamma)
                    extra[key] = {'name': key, 'lr': lr, 'steps': steps, 'gamma': gamma}
            configs = list({c['name']: c for c in coarse + list(extra.values())}.values())
            assert len(configs) <= 31
            write(folder / 'refinement_plan.json', {'based_on': 'validation_only_coarse_top3',
                'configs': configs, 'test_results_used': False, 'top3': [r['config'] for r in top]})
        for cfg in configs:
            dest = folder / 'candidates' / f'{cfg["name"]}.json'
            if dest.exists():
                existing = read(dest)
                assert existing['config'] == cfg and existing['runtime_lock_sha256'] == sha(out / 'runtime_lock.json')
                continue
            records = []
            audits = []
            begin = time.perf_counter()
            for item, label in zip(items, labels):
                head = copy.deepcopy(template).eval()
                diagnostic = fit_fullspan_head(head, [{'head_input': item['head_input']}] * cfg['steps'],
                    lr=cfg['lr'], anchor_gamma=cfg['gamma'], optimizer_eps=1e-4)
                with torch.no_grad():
                    logits = replay_temporal_head(head, item['head_input'], 'cuda').detach().cpu()
                assert diagnostic['audit']['optimizer_steps'] == cfg['steps']
                # Completed prediction precedes validation-label evaluation.
                output = {'pred_boxes': item['fused_boxes'], 'pred_sted': logits}
                records.append(score(output, label, item, cfg['name']))
                audits.append({'index': item['index'], 'audit': diagnostic['audit']})
                del head
            result = {'config': cfg, 'records': records, 'summary': summarize(records, frozen),
                'audits': audits, 'gt_used_for_update': False, 'validation_labels_used_for_selection': True,
                'runtime_lock_sha256': sha(out / 'runtime_lock.json'), 'seconds': time.perf_counter() - begin}
            write(dest, result)
            print(f'[tune fit {name}] {cfg["name"]}: vIoU={result["summary"]["vIoU"]:.4f}', flush=True)
        rows = [read(folder / 'candidates' / f'{c["name"]}.json') for c in configs]
        selected = choose(rows)
        write(folder / f'selected_{stage}.json', {'group': name, 'config': selected['config'],
            'validation_summary': selected['summary'], 'candidate_count': len(rows),
            'selection_rule': 'source_macro_corrected_viou; within .1 pp of max prefer fewer steps then smaller lr then gamma',
            'test_metrics_used': False, 'test_evaluation_pending': True,
            'nonzero_update_configuration': True, 'runtime_lock_sha256': sha(out / 'runtime_lock.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['cache', 'coarse', 'refine'])
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--groups', nargs='*')
    a = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260907)
    np.random.seed(20260907)
    torch.backends.cudnn.benchmark = False
    if a.action == 'cache':
        cache(a.out.resolve(), a.groups)
    else:
        fit(a.out.resolve(), a.groups, a.action)
