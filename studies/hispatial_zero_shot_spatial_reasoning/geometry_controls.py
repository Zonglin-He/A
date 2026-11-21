"""Run frozen geometry interventions on the sealed focus questions."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch

from download import read_json, sha256, write_json
from geometry_cohort import BASELINE, CONFIG, RUN, read_jsonl, stable
from predict import BATCH_N, IMG_SIZE, batch_query, cached_single, read_image, release
from score import jsonl


CONDITIONS = ('aligned', 'rgb_only', 'mismatched_xyz')


def load_cases(name, seal):
    path = RUN / f'{name}.jsonl'
    assert sha256(path) == seal['files'][path.name]
    rows = read_jsonl(path)
    assert all(set(r) == {'id', 'key', 'prompt'} for r in rows)
    assert len({r['id'] for r in rows}) == len(rows)
    return rows


def capture_pointmaps(depth, frames):
    folder = RUN / 'pointmaps'
    folder.mkdir(exist_ok=True, mode=0o700)
    receipt = RUN / 'pointmap_sha256.json'
    if receipt.exists():
        previous = read_json(receipt)
        assert set(previous) == set(frames)
        assert all(sha256(folder / f'{key}.npy') == digest for key, digest in previous.items())
        print('verified existing paired pointmaps', len(previous), flush=True)
        return folder
    known = {}
    start = time.monotonic()
    for ordinal, (key, frame) in enumerate(sorted(frames.items()), 1):
        assert stable(frame['benchmark'] + '/' + frame['image_sha256'] + '/' + frame['transform']) == key
        path = folder / f'{key}.npy'
        scene = BASELINE / frame['benchmark'] / 'images'
        with torch.inference_mode():
            image = read_image(scene, frame['image_sha256'], frame['transform'])
            xyz = depth.apply_transform(image).cpu().detach().numpy()
        assert xyz.shape == (4, IMG_SIZE, IMG_SIZE) and xyz.dtype == np.float32 and np.isfinite(xyz).all()
        with path.with_suffix('.tmp').open('wb') as stream:
            np.save(stream, xyz, allow_pickle=False)
        path.with_suffix('.tmp').replace(path)
        known[key] = sha256(path)
        if ordinal % 100 == 0 or ordinal == len(frames):
            print('pointmaps', ordinal, '/', len(frames), round(time.monotonic() - start, 1), 's', flush=True)
    write_json(receipt, known)
    return folder


def generate(predictor, prepared, rows, condition, batch_size):
    images = [prepared(case['key'])[condition] for case in rows]
    if len(images) > 1 and batch_size > 1:
        return batch_query(predictor, images, rows)
    return [cached_single(predictor, p, r['prompt']) for p, r in zip(images, rows)]


def capture(predictor, prepared, cases, probes, batch_size):
    primary = {row['id'] for row in cases}
    rows = sorted(cases + probes, key=lambda r: (r['key'], r['id']))
    main_path, probe_path = (RUN / name for name in ('condition_predictions.jsonl', 'probe_predictions.jsonl'))
    main_tmp, probe_tmp = (path.with_suffix('.jsonl.tmp') for path in (main_path, probe_path))
    start = time.monotonic()
    main_done, probe_done, next_log = 0, 0, 100
    with main_tmp.open('w') as main_stream, probe_tmp.open('w') as probe_stream:
        for index in range(0, len(rows), batch_size):
            group = rows[index:index + batch_size]
            main = [row for row in group if row['id'] in primary]
            auxiliary = [row for row in group if row['id'] not in primary]
            if main:
                aligned = generate(predictor, prepared, main, 'aligned', batch_size)
                rgb = generate(predictor, prepared, main, 'rgb_only', batch_size)
                wrong = generate(predictor, prepared, main, 'mismatched_xyz', batch_size)
                for row, a, r, m in zip(main, aligned, rgb, wrong, strict=True):
                    main_stream.write(json.dumps(dict(id=row['id'], aligned=a, rgb_only=r, mismatched_xyz=m), ensure_ascii=False) + '\n')
            if auxiliary:
                raw = generate(predictor, prepared, auxiliary, 'aligned', batch_size)
                for row, response in zip(auxiliary, raw, strict=True):
                    probe_stream.write(json.dumps(dict(id=row['id'], aligned=response), ensure_ascii=False) + '\n')
            main_stream.flush()
            probe_stream.flush()
            main_done += len(main)
            probe_done += len(group) - len(main)
            if main_done + probe_done >= next_log:
                print('matched cases', main_done, '/', len(cases), 'probes', probe_done, '/', len(probes),
                      round(time.monotonic() - start, 1), 's', flush=True)
                next_log += 100
    assert (main_done, probe_done) == (len(cases), len(probes))
    main_tmp.replace(main_path)
    probe_tmp.replace(probe_path)


def run(model_cache):
    if (RUN / 'predictions_seal.json').exists():
        raise FileExistsError(RUN / 'predictions_seal.json')
    torch.manual_seed(20261121)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    input_seal = read_json(RUN / 'inputs_seal.json')
    assert input_seal['config_sha256'] == sha256(CONFIG)
    frame_path = RUN / 'frames.json'
    assert sha256(frame_path) == input_seal['files']['frames.json']
    frames = read_json(frame_path)
    cases, probes = (load_cases(k, input_seal) for k in ('cases', 'probes'))
    assert len(cases) == input_seal['task_rows'] and len(probes) == input_seal['probe_rows']
    assert {r['key'] for r in cases} == set(frames) and {r['key'] for r in probes} <= set(frames)
    predictor, depth, runtime = release(model_cache)
    for benchmark, item in input_seal['baseline'].items():
        assert sha256(BASELINE / benchmark / 'predictions_seal.json') == item['prediction_seal']
        assert read_json(BASELINE / benchmark / 'runtime.json')['identity']['runtime'] == runtime
    print('frozen models and baseline runtime verified', flush=True)
    started = time.monotonic()
    map_dir = capture_pointmaps(depth, frames)
    model = predictor.model

    def rgb_projection(pixels, xyz, vision):
        channels = vision.last_hidden_state.shape[-1]

        def keep_rgb(_module, args):
            (tokens,) = args
            assert tokens.shape[-1] == channels * 2
            return (torch.cat((tokens[..., :channels], torch.zeros_like(tokens[..., channels:])), dim=-1),)

        projector = model.multi_modal_projector
        handle = projector.register_forward_pre_hook(keep_rgb)
        try:
            return model.get_image_features(pixels, xyz)
        finally:
            handle.remove()

    @lru_cache(maxsize=12)
    def prepared(key):
        frame = frames[key]
        image = read_image(BASELINE / frame['benchmark'] / 'images', frame['image_sha256'], frame['transform'])
        resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        own = torch.from_numpy(np.load(map_dir / f'{key}.npy', allow_pickle=False))
        donor_xyz = torch.from_numpy(np.load(map_dir / (frame['donor'] + '.npy'), allow_pickle=False))
        with torch.inference_mode():
            pixels = predictor.processor(text='<image> ', images=resized, return_tensors='pt').to(predictor.device)['pixel_values']
            vision = model.vision_tower(pixels)
            with patch.object(model.vision_tower, 'forward', return_value=vision):
                aligned = model.get_image_features(pixels, own[None].to(predictor.device))
                wrong = model.get_image_features(pixels, donor_xyz[None].to(predictor.device))
                rgb_only = rgb_projection(pixels, own[None].to(predictor.device), vision)
        return {'aligned': (resized, own, aligned), 'rgb_only': (resized, torch.zeros_like(own), rgb_only),
                'mismatched_xyz': (resized, donor_xyz, wrong)}

    # Verify the actual intervention on the same RGB tokens against two different maps.
    first = prepared(cases[0]['key'])
    image, own, _ = first['aligned']
    _, donor_xyz, _ = first['mismatched_xyz']
    with torch.inference_mode():
        pixels = predictor.processor(text='<image> ', images=image, return_tensors='pt').to(predictor.device)['pixel_values']
        vision = model.vision_tower(pixels)
        with patch.object(model.vision_tower, 'forward', return_value=vision):
            original_geometry = rgb_projection(pixels, own[None].to(predictor.device), vision)
            swapped_geometry = rgb_projection(pixels, donor_xyz[None].to(predictor.device), vision)
        torch.testing.assert_close(original_geometry, swapped_geometry, rtol=0, atol=0)
        vision_rms = (original_geometry - first['rgb_only'][2]).float().square().mean().sqrt().item()
    print('RGB-only projector is invariant to the two pointmaps', flush=True)

    anchor = []
    differences = []
    exact = dict.fromkeys(CONDITIONS, 0)
    for benchmark in input_seal['baseline']:
        sample = sorted((r for r in cases if r['id'].startswith(benchmark + '/')), key=lambda r: stable('anchor/' + r['id']))[:24]
        for start in range(0, len(sample), BATCH_N):
            rows = sample[start:start + BATCH_N]
            result = {}
            for condition in CONDITIONS:
                single = generate(predictor, prepared, rows, condition, 1)
                batched = generate(predictor, prepared, rows, condition, BATCH_N)
                exact[condition] += sum(a == b for a, b in zip(single, batched))
                result[condition] = single
            for i, row in enumerate(rows):
                both = prepared(row['key'])
                cached = both['aligned'][2]
                anchor.append({'id': row['id'], 'aligned': result['aligned'][i]})
                differences.append({k: (both[k][2] - cached).float().square().mean().sqrt().item()
                                    for k in CONDITIONS if k != 'aligned'})

    assert all(all(value > 0 for value in diff.values()) for diff in differences)
    jsonl(RUN / 'aligned_anchor.jsonl', anchor)
    sizes = {k: BATCH_N if exact[k] == len(anchor) else 1 for k in CONDITIONS}
    write_json(RUN / 'runtime.json', dict(runtime=runtime, anchor_rows=len(anchor), exact_batch_vs_single=exact,
                                        batch_sizes=sizes, rgb_only_invariant_to_geometry_swap=True,
                                        separate_rgb_recompute_rms=vision_rms, identical_recipient_vision_for_all_conditions=True,
                                        mean_fused_token_rms_vs_aligned={key: np.mean([r[key] for r in differences]).item()
                                                                       for key in differences[0]}))
    print('parity', exact, 'out of', len(anchor), 'batch sizes', sizes, flush=True)
    capture(predictor, prepared, cases, probes, min(sizes.values()))
    assert not any(p.requires_grad for p in model.parameters()) and not model.training
    files = ('condition_predictions.jsonl', 'probe_predictions.jsonl', 'aligned_anchor.jsonl', 'pointmap_sha256.json', 'runtime.json')
    write_json(RUN / 'predictions_seal.json', {
        'input_seal_sha256': sha256(RUN / 'inputs_seal.json'), 'files': {f: sha256(RUN / f) for f in files},
        'task_rows_per_condition': len(cases), 'probe_rows': len(probes), 'pointmaps': len(frames),
        'gpu_seconds_after_loading': time.monotonic() - started,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'ground_truth_read': False})
    print('predictions sealed', len(cases), len(probes), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-cache', type=Path, required=True)
    run(parser.parse_args().model_cache)
