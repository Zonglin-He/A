#!/usr/bin/env python3
"""Locked fixed-configuration STVG baseline expansion; GT-free adaptation API."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from scripts.run_temporal_anchor_spatial_v1 import cpu, sha, write
from scripts.run_feasibility import load_full_video, prediction_record
from vg_tta.baseline_expansion_data import make_dataset, select_sources, shift_pixels, lift_prediction
from vg_tta.component_fusion import fuse_views
from vg_tta.fullspan_tta import fit_fullspan_head, replay_temporal_head
from vg_tta.geometric_video_io import normalize_raw_view
from vg_tta.geometric_spatial_teacher import map_crop_boxes
from vg_tta.simplified_crop import full_video_crop_rect
from vg_tta.phase2 import aggregate_predictions, deterministic_views
from vg_tta.tubedetr_runtime import (add_repo_to_path, build_model, load_official_checkpoint,
    encode_video, decode_video, decode_video_with_temporal_head_input)

DEFAULT = ROOT / 'artifacts/baseline_expansion_v1'
CODE = ['scripts/run_baseline_expansion_v1.py', 'vg_tta/baseline_expansion_data.py',
        'vg_tta/external_tta_baselines.py', 'vg_tta/fullspan_tta.py', 'vg_tta/metrics.py',
        'vg_tta/tubedetr_runtime.py', 'vg_tta/component_fusion.py', 'vg_tta/simplified_crop.py',
        'vg_tta/geometric_video_io.py', 'vg_tta/geometric_spatial_teacher.py',
        'vg_tta/augmentations.py', 'vg_tta/phase2.py', 'scripts/anygroundbench_floor_loader.py']


def read(path):
    return json.loads(Path(path).read_text())


def synchronize():
    torch.cuda.synchronize()
    return time.perf_counter()


def prepare(out, limit):
    add_repo_to_path(ROOT / 'external/TubeDETR')
    if (out / 'lock.json').exists():
        raise FileExistsError('do not overwrite an existing protocol')
    specs = {
        'hc': {'kind': 'hcstvg', 'root': str(ROOT / 'data/hcstvg2_confirm512'),
               'annotation': str(ROOT / 'data/hcstvg2_confirm512/annotations/valv2_proc.json')},
        'vid': {'kind': 'vidstg', 'root': str(ROOT / 'data/vidstg_phase3_confirmation'),
                'annotation': str(ROOT / 'data/vidstg_phase3_confirmation/annotations/test.json')},
        'mouse': {'kind': 'anygroundbench', 'root': str(ROOT / 'data/decoder_mouse_v1'),
                  'annotation': str(ROOT / 'data/decoder_mouse_v1/annotations/heldout_test.json')},
    }
    football = read(ROOT / 'artifacts/decoder_development_v1_intake/target_data_ready_sampling_eligible_v2.json')['stages']['football_heldout_test']
    specs['football'] = {'kind': 'anygroundbench', 'root': football['runtime_root'],
                         'annotation': football['annotation_file']}
    for name, spec in specs.items():
        data, rows, _ = make_dataset(spec)
        override = None
        if name == 'football':
            override = {k: v['source_cluster'] for k, v in football['query_metadata'].items()}
        indices, sources = select_sources(rows, spec['kind'], limit, 20260907,
                                          overrides=override, all_queries=name == 'mouse' and limit > 2)
        if limit <= 2:
            indices = indices[:limit]
            sources = {str(i): sources[str(i)] for i in indices}
        spec.update(indices=indices, sources=sources, annotation_sha256=sha(spec['annotation']),
                    role='fixed-configuration expansion; prior research exposure not certified absent')
        spec['video_sha256'] = {}
        for i in indices:
            path = Path(spec['root']) / 'video' / rows[i]['video_path']
            spec['video_sha256'][str(path)] = sha(path)
    cells = {}
    for target in ['hc', 'vid']:
        for source in ['hc', 'vid']:
            conditions = ['clean', 'blur3', 'low_light3', 'subsample2'] if source == target else ['clean']
            for condition in conditions:
                cells[f'{source}_to_{target}_{condition}'] = {
                    'source': source, 'target': target, 'condition': condition,
                    'shift': 'source_clean_or_controlled' if source == target else 'compound_cross_dataset'}
    for target in ['mouse', 'football']:
        cells[f'hc_to_{target}_clean'] = {'source': 'hc', 'target': target,
                                         'condition': 'clean', 'shift': 'specialized_domain'}
    ckpts = {name: str(ROOT / f'checkpoints/tubedetr_{suffix}_res224_stride2.pth')
             for name, suffix in [('hc', 'hcstvg2'), ('vid', 'vidstg')]}
    lock = {'version': 'baseline_expansion_v1', 'datasets': specs, 'cells': cells,
        'checkpoints': ckpts, 'checkpoint_sha256': {k: sha(v) for k, v in ckpts.items()},
        'selected_model': read(ROOT / 'protocols/current_main_model.json'),
        'selection_sha256': sha(ROOT / 'protocols/current_main_model.json'),
        'code_sha256': {p: sha(ROOT / p) for p in CODE},
        'upstream_code_sha256': {str(p.relative_to(ROOT)): sha(p) for sub in ['models', 'datasets']
                                  for p in (ROOT / 'external/TubeDETR' / sub).rglob('*.py')},
        'fixed_config_not_tuned': True, 'smoke': limit <= 2,
        'settings': {'norm_scope': 'all transformer.decoder LayerNorm affine only',
            'external_lrs': {'tent': .001, 'memo': .005, 'sar': .001},
            'external_optimizers': {'tent': 'Adam', 'memo': 'SGD', 'sar': 'SAM-SGD momentum .9'},
            'external_steps': 3, 'memo_views': 4, 'augmentation_seed': 41001,
            'sar_rho': .05, 'sar_entropy_margin_fraction': .4,
            'ours_lr': .01, 'ours_steps': 1, 'ours_gamma': 1e-4,
            'mode': 'eval; no dropout or buffer-statistic adaptation',
            'resolution': 224, 'high_resolution': 448, 'crop_resolution': 320, 'stride': 2,
            'video_max_len': 200, 'fps': 5,
            'metrics': 'corrected vIoU, legacy secondary; original sampled grid for all shifts',
            'baseline_adaptation': 'TENT/MEMO/SAR task and scope ports, not original classification reproductions',
            'fusion_fairness': 'all methods also evaluated with identical frozen-prediction crop construction and three-view median; external updated decoder applied to all views',
            'episodic': True, 'labels_in_adaptation': False,
            'subsample2_evaluation': 'timestamp-linear spatial interpolation to original grid; endpoint candidates only observed indices; no GT-driven filtering',
            'blur3': 'raw-pixel Gaussian sigma=1.6*min(H,W)/224, kernel=ceil(6*sigma) made odd',
            'low_light3': 'raw pixel factor .4, round uint8',
            'pending': ['three augmentation seeds', 'baseline hyperparameter validation', 'TA-STVG verification', 'OmniGround and Industry/Surgery intake']}}
    write(out / 'lock.json', lock)
    print({k: {'queries': len(v['indices']), 'sources': len(set(v['sources'].values()))}
           for k, v in specs.items()}, flush=True)


def verify_lock(out):
    lock = read(out / 'lock.json')
    for p, h in {**lock['code_sha256'], **lock['upstream_code_sha256']}.items():
        if sha(ROOT / p) != h:
            raise RuntimeError(f'locked code changed: {p}')
    for spec in lock['datasets'].values():
        assert sha(spec['annotation']) == spec['annotation_sha256']
    return lock


def norm_parameters(model):
    names = []
    parameters = []
    model.eval().requires_grad_(False)
    for name, module in model.named_modules():
        if name.startswith('transformer.decoder.') and isinstance(module, torch.nn.LayerNorm):
            for suffix, parameter in module.named_parameters(recurse=False):
                names.append(f'{name}.{suffix}')
                parameters.append(parameter)
    if not parameters:
        raise RuntimeError('no decoder LayerNorm affine parameters found')
    return names, parameters


def adapt_predict(model, raw, caption, settings, seed, run_controls=False):
    """Only pixels/query/settings enter this function, never GT/annotation."""
    from vg_tta.external_tta_baselines import run_endpoint_tta
    names, parameters = norm_parameters(model)
    initial = [p.detach().clone() for p in parameters]
    versions = {n: p._version for n, p in model.named_parameters() if n not in names}
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    source_head = model.sted_embed
    duration = len(raw)
    view_memories = {}
    view_outputs = {}
    start = synchronize()

    def memory(key, video):
        with torch.no_grad():
            view_memories[key] = encode_video(model, video, caption, repo=ROOT / 'external/TubeDETR',
                                              stride=2, device='cuda')

    def decode(key):
        return decode_video(model, view_memories[key], duration=duration, caption=caption, device='cuda')

    video = normalize_raw_view(raw, resolution=224)
    memory('original', video)
    with torch.no_grad():
        frozen, hidden = decode_video_with_temporal_head_input(model, view_memories['original'],
            duration=duration, caption=caption, device='cuda')
    frozen = cpu({k: frozen[k] for k in ['pred_boxes', 'pred_sted']})
    timings = {'frozen': synchronize() - start}
    view_outputs['original'] = frozen
    h, w = raw.shape[1:3]
    rect = full_video_crop_rect(frozen['pred_boxes'], (h, w))
    for name, res, crop in [('global_high', 448, None), ('crop', 320, rect)]:
        memory(name, normalize_raw_view(raw, resolution=res, crop=crop))
        with torch.no_grad():
            view_outputs[name] = cpu(decode(name))
    x0, y0, x1, y1 = rect

    def fusion(outputs):
        views = {k: v['pred_boxes'].detach().float().cpu() for k, v in outputs.items()}
        views['crop'] = map_crop_boxes(views['crop'], (x0 / w, y0 / h, x1 / w, y1 / h))
        return {'pred_boxes': fuse_views(views, ['original', 'global_high', 'crop']),
                'pred_sted': outputs['original']['pred_sted'].detach().cpu()}

    predictions = {'frozen': frozen, 'frozen_fusion': fusion(view_outputs)}
    timings['frozen_fusion'] = synchronize() - start
    full_logits = torch.zeros_like(frozen['pred_sted'])
    full_logits[0, 0, 0] = 10
    full_logits[0, -1, 1] = 10
    predictions['direct_fullspan'] = {**frozen, 'pred_sted': full_logits}
    predictions['direct_fullspan_fusion'] = {**predictions['frozen_fusion'], 'pred_sted': full_logits}
    timings['direct_fullspan'] = timings['frozen']
    timings['direct_fullspan_fusion'] = timings['frozen_fusion']

    begin = synchronize()
    head = copy.deepcopy(source_head).eval()
    ours = fit_fullspan_head(head, [{'head_input': hidden.detach()}], lr=settings['ours_lr'],
                            anchor_gamma=settings['ours_gamma'])
    try:
        model.sted_embed = head
        with torch.no_grad():
            actual = cpu(decode('original'))
            replay = replay_temporal_head(head, hidden.detach(), 'cuda').detach().cpu()
        assert torch.equal(actual['pred_sted'], replay)
        assert torch.equal(actual['pred_boxes'], frozen['pred_boxes'])
        predictions['ours_temporal_only'] = {k: actual[k] for k in ['pred_boxes', 'pred_sted']}
        predictions['ours'] = {**predictions['frozen_fusion'], 'pred_sted': actual['pred_sted']}
        timings['ours_temporal_only'] = timings['frozen'] + synchronize() - begin
        timings['ours'] = timings['frozen_fusion'] + synchronize() - begin
    finally:
        model.sted_embed = source_head
    del head, hidden

    # Photometric MEMO views preserve query left/right and all temporal positions.
    memo_begin = synchronize()
    memo_outputs = [frozen]
    aug_views = deterministic_views(video, seed=seed, count=settings['memo_views'])
    for i, view in enumerate(aug_views[1:], 1):
        memory(f'memo{i}', view)
        with torch.no_grad():
            memo_outputs.append(cpu(decode(f'memo{i}')))
    predictions['memo_augmentation_only'] = cpu(aggregate_predictions(memo_outputs))
    memo_extra = synchronize() - memo_begin
    timings['memo_augmentation_only'] = timings['frozen'] + memo_extra
    del aug_views, video
    audits = {'ours': ours, 'norm_parameter_names': names,
              'norm_parameter_count': sum(p.numel() for p in parameters)}
    for method in ['tent', 'memo', 'sar']:
        method_lr = settings['external_lrs'][method]
        with torch.no_grad():
            for p, origin in zip(parameters, initial):
                p.copy_(origin)
        for p in parameters:
            p.requires_grad_(True)
        method_start = synchronize()

        def closure(index):
            key = 'original' if index == 0 else f'memo{index}'
            return decode(key)['pred_sted']

        if run_controls:
            controls = []
            for control_lr, control_steps in [(method_lr, 0), (0., 1)]:
                control = run_endpoint_tta(parameters, closure, method=method,
                    lr=control_lr, steps=control_steps,
                    num_views=settings['memo_views'] if method == 'memo' else 1,
                    rho=settings['sar_rho'], entropy_margin_fraction=settings['sar_entropy_margin_fraction'], reset=False)
                assert all(torch.equal(p, origin) for p, origin in zip(parameters, initial))
                with torch.no_grad():
                    checked = cpu(decode('original'))
                assert torch.equal(checked['pred_sted'], frozen['pred_sted'])
                assert torch.equal(checked['pred_boxes'], frozen['pred_boxes'])
                controls.append(control)
            audits[method + '_live_noops'] = controls
            method_start = synchronize()
        audit = run_endpoint_tta(parameters, closure, method=method,
            lr=method_lr, steps=settings['external_steps'],
            num_views=settings['memo_views'] if method == 'memo' else 1,
            rho=settings['sar_rho'], entropy_margin_fraction=settings['sar_entropy_margin_fraction'],
            reset=False)
        with torch.no_grad():
            outputs = {k: cpu(decode(k)) for k in ['original', 'global_high', 'crop']}
        predictions[method] = {k: outputs['original'][k] for k in ['pred_boxes', 'pred_sted']}
        predictions[method + '_fusion'] = fusion(outputs)
        elapsed = synchronize() - method_start
        timings[method] = timings['frozen'] + elapsed + (memo_extra if method == 'memo' else 0)
        timings[method + '_fusion'] = timings['frozen_fusion'] + elapsed + (memo_extra if method == 'memo' else 0)
        audits[method] = audit
        with torch.no_grad():
            for p, origin in zip(parameters, initial):
                p.copy_(origin)
                p.grad = None
                p.requires_grad_(False)
    with torch.no_grad():
        restored = cpu(decode('original'))
    assert torch.equal(restored['pred_sted'], frozen['pred_sted'])
    assert torch.equal(restored['pred_boxes'], frozen['pred_boxes'])
    assert all(p._version == versions[n] for n, p in model.named_parameters() if n in versions)
    assert all(torch.equal(b, buffers[n]) for n, b in model.named_buffers())
    assert all(not module.training for module in model.modules())
    assert all(p.grad is None for p in model.parameters())
    audits.update(reset_exact=True, frozen_parameters_unchanged=True, buffers_exact=True,
                  gt_used=False, native_head_replay_exact=True, crop_rect=list(rect),
                  total_shared_execution_sec=synchronize() - start)
    return predictions, audits, timings


def run(out, cell_names):
    lock = verify_lock(out)
    torch.set_num_threads(4)
    torch.manual_seed(20260907)
    np.random.seed(20260907)
    torch.backends.cudnn.benchmark = False
    selected = cell_names or list(lock['cells'])
    model = None
    loaded_source = None
    for cell_name in selected:
        cell = lock['cells'][cell_name]
        spec = lock['datasets'][cell['target']]
        if loaded_source != cell['source']:
            if model is None:
                model, _ = build_model(ROOT / 'external/TubeDETR', resolution=224, stride=2, device='cuda')
            assert sha(lock['checkpoints'][cell['source']]) == lock['checkpoint_sha256'][cell['source']]
            load = load_official_checkpoint(model, lock['checkpoints'][cell['source']])
            assert not load['missing_keys'] and not load['unexpected_keys'], load
            model.eval().requires_grad_(False)
            loaded_source = cell['source']
        data, rows, capture = make_dataset(spec)
        for pos, index in enumerate(spec['indices']):
            path = out / cell_name / 'episodes' / f'{index:06d}.json'
            if path.exists():
                existing = read(path)
                assert existing['lock_sha256'] == sha(out / 'lock.json')
                continue
            video_path = str(Path(spec['root']) / 'video' / rows[index]['video_path'])
            assert sha(video_path) == spec['video_sha256'][video_path]
            _, targets, target = load_full_video(data[index])
            raw = capture.raw
            assert len(raw) == len(targets)
            shifted, positions = shift_pixels(raw, cell['condition'])
            torch.cuda.reset_peak_memory_stats()
            predictions, audits, timings = adapt_predict(model, shifted, target['caption'], lock['settings'],
                                                         lock['settings']['augmentation_seed'] + index * 10,
                                                         run_controls=pos == 0)
            # Evaluation labels are accessed only after all methods have finished.
            records = []
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            for method, prediction in predictions.items():
                lifted = lift_prediction(prediction, positions, target['frames_id'])
                record = prediction_record(lifted, targets, target, rows[index], sample_index=index,
                    condition=cell['condition'], method=method, runtime_sec=timings[method], peak_vram_gb=peak)
                record.update(source=spec['sources'][str(index)], gt_used=False,
                    evaluation_role=spec['role'], observed_positions=positions,
                    runtime_caveat='cached encoder; external native cost includes extra post-update fusion decodes; compare shared total separately')
                records.append(record)
            write(path, {'lock_sha256': sha(out / 'lock.json'), 'cell': cell_name, 'index': index,
                'source': spec['sources'][str(index)], 'records': records, 'audits': audits,
                'raw_pixel_sha256': hashlib.sha256(raw).hexdigest(), 'shifted_pixel_sha256': hashlib.sha256(shifted).hexdigest(),
                'sampled_input_frames': len(shifted), 'evaluation_frames': len(raw), 'gt_used': False})
            print(f'[{cell_name}] {pos + 1}/{len(spec["indices"])} input={len(shifted)} sec={audits["total_shared_execution_sec"]:.2f}', flush=True)
            del predictions, raw, shifted, targets
        write(out / cell_name / 'complete.json', {'cell': cell_name, 'queries': len(spec['indices']),
            'sources': len(set(spec['sources'].values())), 'lock_sha256': sha(out / 'lock.json')})
    write(out / 'run_invocation_complete.json', {'cells': selected, 'pid': os.getpid(),
                                               'lock_sha256': sha(out / 'lock.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--sources', type=int, default=64)
    parser.add_argument('--cells', nargs='*')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.out.resolve(), args.sources)
    else:
        run(args.out.resolve(), args.cells)
