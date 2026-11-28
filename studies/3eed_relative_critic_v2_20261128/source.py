"""Prepare and decode real Waymo queries for the compact-pool verifier."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from compact_pool import BASELINE, HERE, ROOT, RUN, digest, read, sealed_inputs, write

SOURCE = RUN / 'source'
WAYMO = ROOT / 'assets/data/3eed_dataset/waymo'
UPSTREAM = ROOT / 'assets/upstream-3eed'
MODEL = ROOT / 'assets/models/Qwen3-VL-4B-Instruct/ebb281ec70b05090aa6165b016eac8ec08e71b17'
SPLITS = UPSTREAM / 'data/splits'
VAL_MAX = 768
BATCH = 32


def source_guard():
    """Source preparation, feature extraction, training and validation are separate processes."""
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, Path)):
            return
        p = Path(os.path.abspath(os.fsdecode(args[0])))
        allowed_gate = {RUN/'results.json', RUN/'candidate-seal.json'}
        if (p.is_relative_to(ROOT/'runs') and not p.is_relative_to(SOURCE) and p not in allowed_gate
                or p.is_relative_to(ROOT/'assets/data') and not p.is_relative_to(WAYMO)
                or p.is_relative_to(SPLITS) and p.name not in ('waymo_train.txt','waymo_val.txt')
                or p.is_relative_to(ROOT/'assets/archives')
                or p.is_relative_to(ROOT/'assets/wildrefer-source')):
            raise PermissionError(f'Waymo-only process denied {p}')
    sys.addaudithook(audit)


def runtime():
    sys.path.insert(0, str(HERE.parents[1]/'tools/3eed_recovery'))
    from run_qwen_3eed import check_geometry, registered_predictor, source_paths

    reg = HERE.parents[1] / 'QWEN_CURRENT_METHOD.json'
    assert digest(reg) == BASELINE
    source_paths(UPSTREAM)
    check_geometry()
    return registered_predictor(read(reg))


def order(value):
    return sha256(('relative-critic-v2:20261128:'+str(value)).encode()).hexdigest()


def gate():
    evidence = read(RUN/'results.json')
    assert evidence['stage_1']['decision'] == 'go'
    assert evidence['candidate_seal'] == read(RUN/'candidate-seal.json')
    assert evidence['method_baseline'] == BASELINE


def exclusions():
    """Export Waymo RGB hashes only for excluding exact held-out frames."""
    source, _, _, pins, _ = sealed_inputs()
    assert pins['registration'] == BASELINE
    rows = [Path(r['frame']['rgb']) for r in source.values() if r['target'] == 'waymo']
    assert all(p.is_relative_to(WAYMO) for p in rows)
    with ThreadPoolExecutor(8) as workers:
        hashes = sorted(set(workers.map(digest, rows)))
    write(SOURCE/'waymo-eval-exclusions.json', dict(n_queries=len(rows), hashes=hashes, source=pins))


def queries(folder):
    from PIL import Image
    from crossplatform.round7.camera_projection import read_camera_allowlist
    from score import canonical_box

    rgb, points, metadata = (folder/name for name in ('image.jpg','lidar.npy','meta_info.json'))
    if not rgb.is_file() or not points.is_file():
        return []
    rgb_sha = digest(rgb)
    cal_path = SOURCE/'camera'/f'{order(folder)[:24]}.json'
    cal = read_camera_allowlist(metadata)
    write(cal_path, {k:np.asarray(v,float).tolist() for k,v in cal.items() if v is not None})
    with Image.open(rgb) as image:
        size = np.tile(image.size, 2)
    recording = folder.parent.name
    suffix = int(folder.name.split('_')[-1])
    rows = []
    for slot, ann in enumerate(read(metadata)['ground_info']):
        box = np.asarray(ann.get('bbox_3d'), float)
        rect = np.asarray(ann.get('bbox_2d_proj'), float)
        if (box.shape != (7,) or not np.isfinite(box).all() or (box[3:6] <= 0).any()
                or rect.shape != (4,) or not np.isfinite(rect).all() or (rect[2:]-rect[:2] < 2).any()
                or not ann.get('caption')):
            continue
        uid = order(f'{recording}/{folder.name}/{slot}')[:24]
        frame = dict(uid=uid, target='waymo', text=ann['caption'], rgb=str(rgb), points=str(points),
                     calibration=str(cal_path), suffix=suffix, waymo_view=('F','FL','FR','SL','SR')[suffix],
                     seed=int(uid[:16],16))
        rows.append(dict(uid=uid, frame=frame, rgb_sha=rgb_sha, recording=recording,
                         box=canonical_box(box).tolist()))
    return rows


def validation_cohort(available):
    grouped = defaultdict(list)
    for row in available:
        grouped[row['recording']].append(row)
    ranked = sorted(grouped, key=order)
    buckets = {key: sorted(rows, key=lambda r:order(r['uid'])) for key,rows in grouped.items()}
    take = [buckets[key][j] for j in range(max(map(len,buckets.values()))) for key in ranked if j < len(buckets[key])]
    return take[:VAL_MAX]


def prepare():
    runtime()
    blocked = set(read(SOURCE/'waymo-eval-exclusions.json')['hashes'])
    (SOURCE/'camera').mkdir(exist_ok=True)
    population = {}
    used = set()
    for part in ('train','val'):
        split = SPLITS/f'waymo_{part}.txt'
        recordings = set(split.read_text().split())
        folders = sorted(f.parent for k in recordings for f in (WAYMO/k).glob('*/meta_info.json'))
        with ThreadPoolExecutor(10) as workers:
            all_rows = [row for rows in workers.map(queries, folders) for row in rows]
        eligible = [r for r in all_rows if r['rgb_sha'] not in used and r['rgb_sha'] not in blocked]
        members = eligible if part == 'train' else validation_cohort(eligible)
        used.update(r['rgb_sha'] for r in members)
        assert len({r['uid'] for r in members}) == len(members)
        path = SOURCE/f'{part}-rows-private.json'
        write(path, members)
        population[part] = dict(n=len(members), frames=len({r['frame']['rgb'] for r in members}),
                                recordings=len({r['recording'] for r in members}), eligible=len(eligible),
                                annotated_queries=len(all_rows), available_frames=len(folders),
                                split_sha=digest(split), bank=digest(path))
        print(part, population[part], flush=True)
    write(SOURCE/'roster-seal.json', dict(population=population, source_only=True,
          exact_eval_frame_exclusions=digest(SOURCE/'waymo-eval-exclusions.json'), code=digest(__file__)))


def decode():
    runtime()
    from run_full_qwen import batch_read, load_model
    import torch

    roster = read(SOURCE/'roster-seal.json')
    plans = {part:read(SOURCE/f'{part}-rows-private.json') for part in ('train','val')}
    for part in plans:
        assert roster['population'][part]['bank'] == digest(SOURCE/f'{part}-rows-private.json')
    parts = SOURCE/'qwen'
    parts.mkdir(exist_ok=True)
    processor, model, limit = load_model(MODEL)
    start = perf_counter()
    batches = Counter()
    finished = Counter()
    with torch.inference_mode():
        for part in ('val','train'):
            rows = plans[part]
            for i in range(0, len(rows), BATCH):
                block = rows[i:i+BATCH]
                frames = [r['frame'] for r in block]
                path = parts/f'{part}-{i:04d}.json'
                receipt = dict(ids=[f['uid'] for f in frames], roster=digest(SOURCE/'roster-seal.json'), revision=MODEL.name)
                if path.exists():
                    assert read(path)['pins'] == receipt
                else:
                    new = {k:batch_read(processor, model, limit, frames, k, batches) for k in ('2d','3d')}
                    write(path, dict(pins=receipt, replies=[dict(zip(new,pair,strict=True))
                                                           for pair in zip(*new.values(),strict=True)]))
                finished[part] += len(block)
                if (i // BATCH) % 8 == 0 or finished[part] == len(rows):
                    print('Qwen source', part, finished[part], '/', len(rows),
                          round(perf_counter()-start,1), 's', flush=True)
    write(SOURCE/'qwen-seal.json', dict(roster=digest(SOURCE/'roster-seal.json'), n=dict(finished),
          checksums={p.name:digest(p) for p in sorted(parts.glob('*.json'))}, qwen_updates=0,
          batch_log=dict(batches), source_only=True, seconds=perf_counter()-start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('exclude','prepare','decode'))
    args = parser.parse_args()
    SOURCE.mkdir(exist_ok=True)
    gate()
    if args.stage == 'exclude':
        exclusions()
    else:
        source_guard()
        (prepare if args.stage == 'prepare' else decode)()


if __name__ == '__main__':
    main()
