"""Run the frozen official decoder on a sealed, answer-free benchmark manifest."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

from download import OUTPUT, PROTOCOL, ROOT, read_json, sha256, write_json


IMG_SIZE = 448
BATCH_N = 4
INPUT_FIELDS = {'id', 'image_sha256', 'transform', 'prompt'}


def release(cache):
    cfg = read_json(PROTOCOL)
    pins = {'hispatial': cfg['hispatial_commit'], 'moge': cfg['moge_commit']}
    for subdir, name in (('HiSpatial', 'hispatial'), ('MoGe', 'moge')):
        path = cache / 'sources' / subdir
        commit = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True)
        assert not dirty and commit == pins[name], (name, commit, dirty)
        sys.path.insert(0, str(path))

    verified = {}
    manifest = read_json(ROOT / cfg['weights_manifest'])
    for name in cfg['weights']:
        folder = cache / 'weights' / name
        verified[name] = manifest[name]
        for filename, metadata in manifest[name]['files'].items():
            path = folder / filename
            assert path.stat().st_size == metadata['size'], path
            if 'sha256' in metadata:
                assert sha256(path) == metadata['sha256'], path
            else:
                data = path.read_bytes()
                git_hash = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
                assert git_hash == metadata['git_blob_sha1'], path

    from accelerate import init_empty_weights
    from transformers import PaliGemmaConfig, PaliGemmaProcessor
    from hispatial.inference import HiSpatialPredictor, MoGeProcessor
    from hispatial.model import HiSpatialVLM
    from moge.model.v2 import MoGeModel

    device = torch.device('cuda:0')
    assets = cache / 'weights' / 'hispatial'
    config = PaliGemmaConfig.from_pretrained(assets, local_files_only=True)
    with init_empty_weights():
        model = HiSpatialVLM(config)
    weights = torch.load(assets / 'weights.pt', map_location='cpu', weights_only=True, mmap=True)
    assert all(torch.is_tensor(item) for item in weights.values())
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    predictor = HiSpatialPredictor.__new__(HiSpatialPredictor)
    predictor.processor = PaliGemmaProcessor.from_pretrained(assets, local_files_only=True)
    predictor.processor.tokenizer.padding_side = 'left'
    predictor.model, predictor.device, predictor.img_size = model, device, IMG_SIZE

    depth = MoGeProcessor.__new__(MoGeProcessor)
    depth.device, depth.img_size = device, IMG_SIZE
    depth.model = MoGeModel.from_pretrained(cache / 'weights' / 'moge' / 'model.pt').eval().requires_grad_(False).to(device)
    assert all(not p.requires_grad for m in (model, depth.model) for p in m.parameters())
    assert not any(module.training for m in (model, depth.model) for module in m.modules())
    return predictor, depth, {'upstream_commits': pins, 'assets': verified, 'torch': torch.__version__,
        'transformers': version('transformers'), 'cuda_device': torch.cuda.get_device_name(device),
        'pillow': version('pillow'), 'cv2': cv2.__version__, 'precision': 'float32',
        'float32_matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'max_new_tokens': 100, 'do_sample': False, 'frozen_parameters': True}


def read_image(folder, key, transform):
    data = (folder / key).read_bytes()
    assert hashlib.sha256(data).hexdigest() == key
    with Image.open(BytesIO(data)) as image:
        if transform == 'cv_exif_rgb':
            image = ImageOps.exif_transpose(image)
        elif transform == 'flip_rgb':
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return np.array(image.convert('RGB'))


def input_prompt(value):
    return value if '<image>' in value else '<image>' + value


@torch.inference_mode()
def feature(predictor, resized, xyz):
    inputs = predictor.processor(text='<image> ', images=resized, return_tensors='pt').to(predictor.device)
    return predictor.model.get_image_features(inputs['pixel_values'], xyz[None].to(predictor.device))


@torch.inference_mode()
def cached_single(predictor, prep, question):
    resized, xyz, fused = prep
    with patch.object(predictor.model, 'get_image_features', return_value=fused):
        return predictor.query(image=resized, prompt=question, xyz_values=xyz)


@torch.inference_mode()
def batch_query(predictor, images, rows):
    inputs = predictor.processor(text=[input_prompt(r['prompt']) for r in rows],
        images=[r[0] for r in images], padding=True, return_tensors='pt').to(predictor.device)
    inputs['xyz_values'] = torch.stack([r[1] for r in images]).to(predictor.device)
    fused = torch.cat([r[2] for r in images], dim=0)
    with patch.object(predictor.model, 'get_image_features', return_value=fused):
        generated = predictor.model.generate(**inputs, max_new_tokens=100, do_sample=False)
    stop = predictor.model.generation_config.eos_token_id
    eos = {stop} if isinstance(stop, int) else set(stop or ())
    output = []
    for tokens in generated[:, inputs['input_ids'].shape[1]:].tolist():
        end = next((i + 1 for i, value in enumerate(tokens) if value in eos), len(tokens))
        output.append(predictor.processor.decode(tokens[:end], skip_special_tokens=False))
    return output


def anchor_mode(predictor, prep, manifest):
    unique = [r for r in manifest if r['image_sha256']]
    probe = [unique[int(i)] for i in np.linspace(0, len(unique)-1, min(24, len(unique)), dtype=int)]
    counts = Counter()
    for start in range(0, len(probe), BATCH_N):
        group = probe[start:start+BATCH_N]
        images = [prep(x['image_sha256'], x['transform']) for x in group]
        direct = [predictor.query(image=p[0], prompt=r['prompt'], xyz_values=p[1]) for p, r in zip(images, group)]
        reuse = [cached_single(predictor, p, r['prompt']) for p, r in zip(images, group)]
        counts['single_exact'] += sum(a == b for a, b in zip(direct, reuse))
        try:
            batched = batch_query(predictor, images, group)
            counts['batch_exact'] += sum(a == b for a, b in zip(direct, batched))
        except torch.cuda.OutOfMemoryError:
            counts['batch_oom'] += 1
            torch.cuda.empty_cache()
    use = 'official_single'
    if counts['single_exact'] == len(probe):
        use = 'cached_single'
        if counts['batch_exact'] == len(probe):
            use = 'cached_batch'
    return {'probed_rows': len(probe), **counts, 'mode': use, 'batch_size': BATCH_N if use == 'cached_batch' else 1}


def run(benchmark, model_cache):
    torch.manual_seed(20261121)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    work = OUTPUT / benchmark
    assert not (work / 'predictions_seal.json').exists()
    spec = read_json(work / 'inputs_seal.json')
    manifest = work / 'inputs.jsonl'
    assert spec['inputs_sha256'] == sha256(manifest) and spec['protocol_sha256'] == sha256(PROTOCOL)
    visible = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len({r['id'] for r in visible}) == len(visible) == spec['rows']
    assert all(set(row) == INPUT_FIELDS for row in visible)
    print('inputs verified', benchmark, len(visible), spec['distinct_image_inputs'], flush=True)
    official, moge, runtime = release(model_cache)
    print('frozen models loaded', benchmark, flush=True)

    @lru_cache(maxsize=8)
    def prep(key, transform):
        with torch.inference_mode():
            rgb = read_image(work / 'images', key, transform)
            xyz = moge.apply_transform(rgb)
            if xyz.shape != (4, IMG_SIZE, IMG_SIZE) or not torch.isfinite(xyz).all().item():
                raise ValueError(f'invalid MoGe input for {key}')
            resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
            return resized, xyz, feature(official, resized, xyz)

    anchor_path = work / 'runtime.json'
    identity = {'inputs': sha256(manifest), 'protocol': sha256(PROTOCOL), 'predictor': sha256(__file__), 'runtime': runtime}
    if anchor_path.exists():
        prior = read_json(anchor_path)
        assert prior['identity'] == identity
        parity = prior['parity']
    else:
        parity = anchor_mode(official, prep, visible)
        write_json(anchor_path, {'identity': identity, 'parity': parity})
    print('exact output probe', parity, flush=True)
    begun = time.monotonic()

    path = work / 'predictions.jsonl'
    existing = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    known = {r['id'] for r in existing}
    assert len(known) == len(existing) and known <= {x['id'] for x in visible}
    chosen_mode = parity['mode']
    jobs = sorted((r for r in visible if r['id'] not in known),
                  key=lambda r: (r['image_sha256'] or '', r['transform'], r['id']))
    n = parity['batch_size']
    count = len(known)
    modes = Counter(r['mode'] for r in existing)
    next_update = 0
    with path.open('a', encoding='utf-8') as stream:
        for start in range(0, len(jobs), n):
            block = jobs[start:start+n]
            ready = [r for r in block if r['image_sha256'] is not None]
            preparation = [prep(r['image_sha256'], r['transform']) for r in ready]
            mode = chosen_mode
            if mode == 'cached_batch' and len(ready) >= 2:
                try:
                    answers = batch_query(official, preparation, ready)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    mode = 'cached_single_oom_recovery'
                    answers = [cached_single(official, p, r['prompt']) for p, r in zip(preparation, ready)]
            elif mode == 'official_single':
                answers = [official.query(image=p[0], prompt=r['prompt'], xyz_values=p[1]) for p, r in zip(preparation, ready)]
            else:
                mode = 'cached_single'
                answers = [cached_single(official, p, r['prompt']) for p, r in zip(preparation, ready)]
            assert len(answers) == len(ready)
            prepared = {r['id']: t for r, t in zip(ready, answers)}
            for row in block:
                query_id = row['id']
                record = {'id': query_id, 'response': prepared.get(query_id), 'mode': mode if query_id in prepared else 'official_image_failure'}
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                modes[record['mode']] += 1
            stream.flush()
            count += len(block)
            if count >= next_update or count == len(visible):
                print(benchmark, count, '/', len(visible), 'seconds', round(time.monotonic() - begun, 1),
                      'image_cache', prep.cache_info().hits, prep.cache_info().misses, flush=True)
                next_update = count + 100
    assert count == len(visible)
    write_json(work / 'predictions_seal.json', {'rows': count, 'predictions_sha256': sha256(path),
        'inputs_sha256': sha256(manifest), 'runtime_sha256': sha256(anchor_path), 'modes': dict(modes),
        'run_seconds': time.monotonic()-begun, 'peak_allocated_cuda_bytes': torch.cuda.max_memory_allocated(),
        'answers_loaded_by_gpu_runner': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=read_json(PROTOCOL)['order'])
    parser.add_argument('--model-cache', type=Path, required=True)
    args = parser.parse_args()
    run(args.benchmark, args.model_cache)
